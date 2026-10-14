"""Isolated offline NeoGraph RequestQueue/shared-index experiment.

Run: python scripts/poc_shared_index_queue.py run --neograph-root /path/to/NeoGraph
The compiler, Python dependencies and NeoGraph checkout must already exist.
Diagnostic holds are synthetic; every started operation returns a real engine result.
"""
from __future__ import annotations

import argparse
import ctypes
from contextlib import contextmanager
from dataclasses import dataclass, field
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import select
import shutil
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

SID = "11111111-2222-3333-4444-555555555555"
SENTINEL = "POC_FORBIDDEN_SENTINEL"
BACKEND = "neograph::util::RequestQueue/moodycamel::ConcurrentQueue"
SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[1]
MAX_FRAME = 65536
CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.c_void_p)


def sanitized_env() -> dict[str, str]:
    # Start with a whitelist, not a copy containing provider overrides/secrets.
    names = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "LD_LIBRARY_PATH", "VIRTUAL_ENV")
    result = {name: os.environ[name] for name in names if name in os.environ}
    result.update(PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1",
                  PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return result


def isolate_environment(root: Path) -> None:
    for name in list(os.environ):
        if name.startswith(("SELF_DIRECT_", "CODEX_", "OPENROUTER_", "OPENAI_", "ANTHROPIC_")):
            os.environ.pop(name, None)
    home = root / "home"
    home.mkdir(exist_ok=True)
    os.environ.update(HOME=str(home), XDG_CONFIG_HOME=str(home / "config"),
                      XDG_CACHE_HOME=str(home / "cache"), SELF_DIRECT_LOCAL_ONLY="true")


def make_fixture(root: Path, events: int) -> None:
    sessions = root / "sessions"
    sessions.mkdir(exist_ok=True)
    path = sessions / f"rollout-2026-09-09T01-00-00-{SID}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"type": "session_meta", "payload": {"id": SID}}) + "\n")
        for i in range(events):
            stream.write(json.dumps({"type": "response_item", "payload": {
                "type": "function_call", "name": "shell", "call_id": f"synthetic-{i}",
                "arguments": {"command": f"echo safe-{i}"}}}) + "\n")


def compile_bridge(root: Path, neograph: Path, compiler: str) -> Path:
    neograph = neograph.expanduser().resolve()
    header = neograph / "include/neograph/util/request_queue.h"
    dependency = neograph / "deps/concurrentqueue.h"
    if not header.is_file() or not dependency.is_file():
        raise ValueError("--neograph-root must contain include/neograph/util/request_queue.h and deps/concurrentqueue.h")
    executable = shutil.which(compiler)
    if executable is None:
        raise ValueError(f"GCC-compatible C++20 compiler unavailable: {compiler}")
    library = root / "poc_index_queue_native.so"
    command = [executable, "-std=c++20", "-O2", "-shared", "-fPIC", "-pthread",
               "-I", str(neograph / "include"), "-I", str(neograph / "deps"),
               str(SCRIPT.with_name("poc_index_queue_native.cpp")), "-o", str(library)]
    completed = subprocess.run(command, env=sanitized_env(), capture_output=True, text=True, timeout=120)
    if completed.returncode:
        raise RuntimeError(f"native compilation failed:\n{completed.stderr}")
    return library


class NativeQueue:
    def __init__(self, library: Path, workers: int, capacity: int, callback: Any):
        self.lib = ctypes.CDLL(str(library))  # Releases GIL for submit/take/close joins.
        self.callback = CALLBACK(callback)  # Root remains alive through destroy.
        error_type = ctypes.POINTER(ctypes.c_char)
        signatures = {
            "poc_queue_create": ([ctypes.c_size_t, ctypes.c_size_t, error_type, ctypes.c_size_t], ctypes.c_void_p),
            "poc_queue_submit": ([ctypes.c_void_p, ctypes.c_uint64, CALLBACK, ctypes.c_void_p, error_type, ctypes.c_size_t], ctypes.c_int),
            "poc_queue_take": ([ctypes.c_void_p, ctypes.c_uint64, error_type, ctypes.c_size_t], ctypes.c_int),
            "poc_queue_stats": ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)], ctypes.c_int),
            "poc_queue_close": ([ctypes.c_void_p, error_type, ctypes.c_size_t], ctypes.c_int),
            "poc_queue_destroy": ([ctypes.c_void_p], None),
            "poc_queue_backend": ([], ctypes.c_char_p),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, result
        self.backend = self.lib.poc_queue_backend().decode()
        if self.backend != BACKEND:
            raise RuntimeError(f"unexpected queue backend: {self.backend}")
        error = ctypes.create_string_buffer(2048)
        self.pointer = self.lib.poc_queue_create(workers, capacity, error, len(error))
        if not self.pointer:
            raise RuntimeError(error.value.decode(errors="replace"))

    def call(self, name: str, *args: Any) -> tuple[int, str]:
        error = ctypes.create_string_buffer(2048)
        status = getattr(self.lib, name)(self.pointer, *args, error, len(error))
        return status, error.value.decode(errors="replace")

    def stats(self) -> dict[str, int]:
        values = (ctypes.c_uint64 * 6)()
        if self.lib.poc_queue_stats(self.pointer, values) != 0:
            raise RuntimeError("native stats failed")
        return dict(zip(("pending", "active", "completed", "rejected", "workers", "capacity"), values))


@dataclass
class Job:
    job_id: int
    method: str
    params: dict[str, Any]
    deadline: float
    hold: float
    evaluation_hold: float = 0.0
    control: Any = None
    admitted: bool = False
    status: str = "submitting"
    claimed: bool = False
    started: bool = False
    engine_calls: int = 0
    cancelled: str | None = None
    result: Any = None
    failure: dict[str, str] | None = None
    native_settled: bool = False
    done: threading.Event = field(default_factory=threading.Event)


class Service:
    def __init__(self, root: Path, library: Path, workers: int, capacity: int, lock_timeout: float):
        isolate_environment(root)
        # Deliberately avoid server.py and get_settings()/credential loaders.
        from self_directing_mcp.config import Settings
        from self_directing_mcp import engine as engine_module
        from self_directing_mcp.audit import runner as audit_runner
        from self_directing_mcp.engine import SelfDirectEngine

        settings = Settings(codex_sessions_dir=root / "sessions", index_dir=root / "index",
                            grokbot_transcripts_dir=str(root / "grokbot"),
                            omp_sessions_dir=root / "omp", agy_app_data_dirs=[root / "agy"],
                            session_provider="codex", contracts_path=root / "index/contracts.json",
                            local_only=True, openrouter_api_key_file=None, openrouter_api_key=None,
                            seed_example_contracts=False, lock_wait_timeout_sec=lock_timeout,
                            _env_file=None)
        self.cv = threading.Condition(threading.RLock())
        self.admission = threading.Lock()
        self.jobs: dict[int, Job] = {}
        self.next_id = 1
        self.closing = False
        self.closed = threading.Event()
        self.collector_stop = threading.Event()
        self.native_peak = self.writer_peak = self.writer_active = 0
        self.evaluation_peak = self.evaluation_active = 0
        self.phases = threading.local()
        self.engine_calls = 0
        self.collector_error: str | None = None
        owner = self

        class MeasuredEngine(SelfDirectEngine):
            @contextmanager
            def _publication_guard(self):
                # Count the actual writer lease, not the entire public operation.
                with super()._publication_guard():
                    nested = getattr(owner.phases, "writer_depth", 0)
                    owner.phases.writer_depth = nested + 1
                    if not nested:
                        with owner.cv:
                            owner.writer_active += 1
                            owner.writer_peak = max(owner.writer_peak, owner.writer_active)
                            owner.cv.notify_all()
                    try:
                        yield
                    finally:
                        owner.phases.writer_depth -= 1
                        if not nested:
                            with owner.cv:
                                owner.writer_active -= 1
                                owner.cv.notify_all()

        original_audit = engine_module.run_audit
        original_contract = audit_runner.audit_contract

        def measured_audit(*args, **kwargs):
            with owner.cv:
                owner.evaluation_active += 1
                owner.evaluation_peak = max(owner.evaluation_peak, owner.evaluation_active)
                job = getattr(owner.phases, "job", None)
                if job is not None:
                    job.status = "evaluating"
                owner.cv.notify_all()
            try:
                return original_audit(*args, **kwargs)
            finally:
                with owner.cv:
                    owner.evaluation_active -= 1
                    owner.cv.notify_all()

        def held_contract(*args, **kwargs):
            job = getattr(owner.phases, "job", None)
            if job is not None and job.evaluation_hold and not getattr(owner.phases, "held_evaluation", False):
                owner.phases.held_evaluation = True
                with owner.cv:
                    job.status = "holding_evaluation"
                    owner.cv.notify_all()
                # Explicit local fixture inside the real native evaluation stage.
                # It cannot interrupt sleep; the shared control is checked on return.
                time.sleep(job.evaluation_hold)
            return original_contract(*args, **kwargs)

        engine_module.run_audit = measured_audit
        audit_runner.audit_contract = held_contract
        self.engine = MeasuredEngine(settings)
        self.engine.sync_session(session_id=SID, provider="codex", embed=False)
        self.engine.upsert_contracts([{"id": "poc-sentinel", "type": "must_not",
                                      "scope": "tool_call", "regex": SENTINEL}])
        self.queue = NativeQueue(library, workers, capacity, self.callback)
        self.collector = threading.Thread(target=self.collect, name="native-future-collector")
        self.collector.start()

    def callback(self, job_id: int, _user_data: Any) -> None:
        # Never let Python exceptions escape a ctypes callback into C++.
        try:
            with self.cv:
                job = self.jobs[job_id]
                job.claimed = True
                job.status = "waiting_for_engine"
                stats = self.queue.stats()
                self.native_peak = max(self.native_peak, stats["active"])
                if job.cancelled or time.monotonic() >= job.deadline:
                    job.cancelled = job.cancelled or "queued_expired"
                    self.cv.notify_all()
                    return
                self.cv.notify_all()
            from self_directing_mcp.request_control import IndexBusy, RequestStopped, request_scope
            try:
                with request_scope(job.control):
                    job.control.check()
                    with self.cv:
                        job.started = True
                        job.status = "started"
                        self.cv.notify_all()
                    self.phases.job = job
                    self.phases.held_evaluation = False
                    if job.hold:
                        # Deliberate residual contention fixture only; the normal
                        # method is invoked AFTER this writer lease is released.
                        with self.engine._publication_guard():
                            with self.cv:
                                job.status = "holding_writer"
                                self.cv.notify_all()
                            time.sleep(job.hold)
                    job.control.check()
                    with self.cv:
                        job.engine_calls += 1
                        self.engine_calls += 1
                    result = getattr(self.engine, job.method)(**job.params)
                    with self.cv:
                        job.result = result
            except Exception as exc:
                with self.cv:
                    kind = exc.reason if isinstance(exc, (IndexBusy, RequestStopped)) else "operation_timeout" if isinstance(exc, TimeoutError) else "engine_error"
                    job.failure = {"class": type(exc).__name__, "message": str(exc), "kind": kind}
            finally:
                self.phases.job = None
        except BaseException as exc:
            with self.cv:
                job = self.jobs.get(job_id)
                if job is not None:
                    job.failure = {"class": type(exc).__name__, "message": str(exc), "kind": "callback_error"}

    def collect(self) -> None:
        try:
            while not self.collector_stop.is_set():
                with self.cv:
                    candidates = [j for j in self.jobs.values() if j.admitted and not j.native_settled]
                    for job in candidates:
                        if time.monotonic() >= job.deadline:
                            job.control.cancel("request_deadline")
                            if not job.started:
                                job.cancelled = job.cancelled or "queued_expired"
                    self.native_peak = max(self.native_peak, self.queue.stats()["active"])
                for job in candidates:
                    code, error = self.queue.call("poc_queue_take", job.job_id)
                    if code == 0:
                        continue
                    with self.cv:
                        if code == 2:
                            job.failure = {"class": "NativeFutureException", "message": error,
                                           "kind": "queue_closed" if self.closing and not job.claimed else "native_error"}
                            if self.closing and not job.claimed:
                                job.cancelled = "queue_closed"
                        elif code != 1:
                            raise RuntimeError(f"native take({job.job_id}) returned {code}: {error}")
                        job.native_settled = True
                        job.status = "not_started" if job.cancelled and not job.started else "failed" if job.failure else "completed"
                        job.done.set()
                        self.cv.notify_all()
                self.collector_stop.wait(0.005)
        except BaseException as exc:
            with self.cv:
                self.collector_error = f"{type(exc).__name__}: {exc}"
                self.cv.notify_all()

    @staticmethod
    def validate(request: dict[str, Any]) -> tuple[str, dict[str, Any], float, float, float]:
        method = request.get("method")
        params = request.get("params", {})
        allowed = {"sync_session": {"session_id", "provider", "embed"},
                   "check_action": {"session_id", "provider", "action"},
                   "audit_session": {"session_id", "provider"},
                   "audit_status": {"session_id", "provider"}}
        if method not in allowed or not isinstance(params, dict) or set(params) - allowed[method]:
            raise ValueError("method/parameters are not allowlisted")
        if params.get("session_id", SID) != SID or params.get("provider", "codex") != "codex":
            raise ValueError("only the synthetic Codex session is allowed")
        params = dict(params, session_id=SID, provider="codex")
        if method == "sync_session":
            if params.get("embed", False) is not False:
                raise ValueError("embedding is prohibited")
            params["embed"] = False
        if method == "check_action":
            action = params.get("action")
            if not isinstance(action, dict) or set(action) - {"tool_name", "arguments"}:
                raise ValueError("action must contain only tool_name/arguments")
            if action.get("tool_name") != "shell" or not isinstance(action.get("arguments"), (str, dict)):
                raise ValueError("only synthetic shell proposed actions are allowed")
        ttl = request.get("timeout", 10.0)
        hold = request.get("diagnostic_hold", 0.0)
        evaluation_hold = request.get("diagnostic_evaluation_hold", 0.0)
        if any(isinstance(value, bool) for value in (ttl, hold, evaluation_hold)):
            raise ValueError("timeouts/holds must be numeric")
        ttl, hold, evaluation_hold = float(ttl), float(hold), float(evaluation_hold)
        if not all(math.isfinite(value) for value in (ttl, hold, evaluation_hold)) or not 0 < ttl <= 60 or not 0 <= hold <= 10 or not 0 <= evaluation_hold <= 10:
            raise ValueError("timeout must be (0,60]; diagnostic holds must be [0,10]")
        if evaluation_hold and method not in ("audit_session", "check_action"):
            raise ValueError("evaluation hold requires a real audit")
        return method, params, ttl, hold, evaluation_hold

    def snapshot(self, job: Job) -> dict[str, Any]:
        with self.cv:
            deadline_passed = time.monotonic() >= job.deadline
            status = job.status
            if not job.done.is_set() and deadline_passed:
                status = "still_running" if job.started else "not_started"
            output = {"job_id": job.job_id, "accepted": job.admitted, "status": status,
                      "started": job.started, "native_claimed": job.claimed,
                      "engine_calls": job.engine_calls, "native_settled": job.native_settled,
                      "final": job.done.is_set(), "deadline_passed": deadline_passed}
            if job.started and not job.done.is_set() and deadline_passed:
                output["outcome"] = "unknown"
            if job.cancelled:
                output["reason"] = job.cancelled
            if job.done.is_set():
                if job.result is not None:
                    output["result"] = job.result
                if job.failure:
                    output["failure"] = job.failure
            return output

    def submit(self, request: dict[str, Any]) -> Job:
        from self_directing_mcp.request_control import RequestControl
        method, params, ttl, hold, evaluation_hold = self.validate(request)
        with self.admission:
            with self.cv:
                deadline = time.monotonic() + ttl
                job = Job(self.next_id, method, params, deadline, hold, evaluation_hold,
                          RequestControl(deadline=deadline))
                self.next_id += 1
                self.jobs[job.job_id] = job
                if self.closing:
                    job.status, job.cancelled = "rejected", "queue_closed"
                    job.done.set()
                    return job
            code, error = self.queue.call("poc_queue_submit", job.job_id, self.queue.callback, None)
            with self.cv:
                if code == 1:
                    job.admitted = True
                    if job.status == "submitting":
                        job.status = "queued"
                else:
                    job.status = "rejected"
                    job.cancelled = "queue_full" if code == 0 else "queue_closed" if code == -2 else "admission_error"
                    if code == -1:
                        job.failure = {"class": "NativeAdmissionError", "message": error, "kind": "native_error"}
                    job.done.set()
                self.cv.notify_all()
            return job

    def wait_job(self, job: Job, connection: socket.socket, stage: str, wait: float) -> dict[str, Any] | None:
        until = time.monotonic() + min(max(wait, 0), 65)
        with self.cv:
            while not job.done.is_set():
                if (stage == "started" and job.started or stage == "claimed" and job.claimed
                        or stage == "evaluating" and job.status == "holding_evaluation"
                        or stage == "writer" and job.status == "holding_writer"):
                    break
                now = time.monotonic()
                if now >= until or stage == "deadline" and now >= job.deadline:
                    break
                readable, _, _ = select.select([connection], [], [], 0)
                disconnected = False
                if readable:
                    try:
                        disconnected = connection.recv(1, socket.MSG_PEEK) == b""
                    except (ConnectionResetError, BrokenPipeError):
                        disconnected = True
                if disconnected:
                    job.control.cancel()
                    if not job.started:
                        job.cancelled = job.cancelled or "client_disconnected"
                    return None
                self.cv.wait(min(0.02, until - now))
            return self.snapshot(job)

    def close_queue(self) -> dict[str, Any]:
        with self.admission:
            if self.closed.is_set():
                return self.metrics()
            with self.cv:
                self.closing = True
            code, error = self.queue.call("poc_queue_close")
            if code:
                raise RuntimeError(f"native close failed: {error}")
            with self.cv:
                until = time.monotonic() + 10
                while any(j.admitted and not j.native_settled for j in self.jobs.values()):
                    if self.collector_error:
                        raise RuntimeError(self.collector_error)
                    if time.monotonic() >= until:
                        raise RuntimeError("accepted native futures did not settle after close")
                    self.cv.wait(0.02)
            self.closed.set()
            return self.metrics()

    def metrics(self) -> dict[str, Any]:
        with self.cv:
            return {"native": self.queue.stats(), "native_active_peak": self.native_peak,
                    "writer_lease_peak": self.writer_peak, "evaluation_peak": self.evaluation_peak,
                    "engine_calls": self.engine_calls,
                    "backend": self.queue.backend, "closed": self.closed.is_set(),
                    "collector_error": self.collector_error, "index_dir": str(self.engine.settings.index_dir),
                    "local_only": self.engine.settings.local_only,
                    "embedder": None if self.engine.embedder is None else type(self.engine.embedder).__name__}

    def dispatch(self, request: dict[str, Any], connection: socket.socket) -> dict[str, Any] | None:
        operation = request.get("op", "submit")
        if operation == "submit":
            job = self.submit(request)
            if request.get("wait", True) and job.admitted:
                return self.wait_job(job, connection, "deadline", 65)
            return self.snapshot(job)
        if operation in ("get", "wait"):
            with self.cv:
                job = self.jobs.get(int(request["job_id"]))
            if job is None:
                raise ValueError("unknown job_id")
            if operation == "wait":
                stage = request.get("stage", "final")
                if stage not in ("final", "started", "claimed", "deadline", "evaluating", "writer"):
                    raise ValueError("unknown wait stage")
                return self.wait_job(job, connection, stage, float(request.get("wait_seconds", 15)))
            return self.snapshot(job)
        if operation == "stats":
            return self.metrics()
        if operation == "reset_metrics":
            with self.cv:
                if any(j.admitted and not j.native_settled for j in self.jobs.values()):
                    raise ValueError("cannot reset metrics while jobs remain unsettled")
                self.native_peak = self.writer_peak = self.evaluation_peak = self.engine_calls = 0
            return self.metrics()
        if operation == "close":
            return self.close_queue()
        raise ValueError("unknown IPC operation")

    def destroy(self) -> None:
        try:
            self.close_queue()
        finally:
            self.collector_stop.set()
            self.collector.join()
            self.queue.lib.poc_queue_destroy(self.queue.pointer)
            self.queue.pointer = None
            self.engine.close()


def encode(value: Any) -> bytes:
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_frame(stream: Any) -> dict[str, Any]:
    line = stream.readline(MAX_FRAME + 1)
    if not line or len(line) > MAX_FRAME or not line.endswith(b"\n"):
        raise ValueError("missing/oversized JSON frame")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("JSON frame must be an object")
    return value


def serve_at(args: argparse.Namespace, root: Path, library: Path) -> None:
    service = Service(root, library, args.workers, args.capacity, args.lock_timeout)
    token = secrets.token_hex(32)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.settimeout(70)
            try:
                request = read_frame(self.rfile)
                supplied = request.pop("token", None)
                if not isinstance(supplied, str) or not hmac.compare_digest(supplied, token):
                    self.wfile.write(encode({"error": "unauthorized"}))
                    return
                if request.get("op") == "stop":
                    response = service.close_queue()
                    self.wfile.write(encode(response))
                    self.wfile.flush()
                    threading.Thread(target=server.shutdown, name="ipc-shutdown").start()
                    return
                response = service.dispatch(request, self.connection)
                if response is not None:
                    self.wfile.write(encode(response))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                try:
                    self.wfile.write(encode({"error": type(exc).__name__, "message": str(exc)}))
                except OSError:
                    pass

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = False
        daemon_threads = False
        block_on_close = True
        request_queue_size = 128

    endpoint_path = args.endpoint_file or root / "endpoint.json"
    endpoint_created = False
    try:
        with Server(("127.0.0.1", 0), Handler) as server:
            endpoint = {"host": "127.0.0.1", "port": server.server_address[1], "token": token}
            endpoint_path = endpoint_path.resolve()
            descriptor = os.open(endpoint_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            endpoint_created = True
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(endpoint, stream)
            print(json.dumps({"ready": True, "endpoint_file": str(endpoint_path),
                              "service_pid": os.getpid(), "workers": args.workers}), flush=True)
            server.serve_forever(poll_interval=0.02)
    finally:
        try:
            service.destroy()
        finally:
            if endpoint_created:
                endpoint_path.unlink(missing_ok=True)


def serve(args: argparse.Namespace) -> None:
    if args.scratch_root:
        root = args.scratch_root.resolve()
        marker = root / "poc-owned.json"
        if not marker.is_file() or json.loads(marker.read_text())["owner_pid"] != os.getppid():
            raise ValueError("internal scratch root must be owned by the parent runner")
        if args.library is None:
            raise ValueError("internal service requires the runner-compiled library")
        serve_at(args, root, args.library.resolve())
    else:
        if args.neograph_root is None:
            raise ValueError("serve requires --neograph-root")
        with tempfile.TemporaryDirectory(prefix="self-direct-queue-serve-") as directory:
            root = Path(directory)
            make_fixture(root, args.events)
            library = compile_bridge(root, args.neograph_root, args.compiler)
            serve_at(args, root, library)


def client(args: argparse.Namespace) -> None:
    endpoint = json.loads(args.endpoint_file.read_text(encoding="utf-8"))
    if endpoint.get("host") != "127.0.0.1":
        raise ValueError("only authenticated IPv4 loopback IPC is allowed")
    request = json.loads(args.request) if args.request else json.load(sys.stdin)
    if args.start_file:
        until = time.monotonic() + 30
        while not args.start_file.exists():
            if time.monotonic() >= until:
                raise TimeoutError("client start barrier timed out")
            time.sleep(0.005)
    started = time.perf_counter()
    with socket.create_connection(("127.0.0.1", int(endpoint["port"])), timeout=70) as connection:
        connection.settimeout(70)
        connection.sendall(encode(dict(request, token=endpoint["token"])))
        if args.disconnect:
            connection.shutdown(socket.SHUT_WR)
            response = {"disconnected": True}
        else:
            with connection.makefile("rb") as stream:
                response = read_frame(stream)
    print(json.dumps({"client_pid": os.getpid(), "seconds": time.perf_counter() - started,
                      "response": response}), flush=True)


class Runner:
    def __init__(self, root: Path, library: Path, workers: int, capacity: int, lock_timeout: float):
        self.root = root
        self.lock_timeout = lock_timeout
        self.endpoint = root / f"endpoint-{workers}-{secrets.token_hex(4)}.json"
        self.env = sanitized_env()
        self.stderr = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        command = [sys.executable, "-B", str(SCRIPT), "serve", "--scratch-root", str(root),
                   "--library", str(library), "--endpoint-file", str(self.endpoint),
                   "--workers", str(workers), "--capacity", str(capacity), "--lock-timeout", str(lock_timeout)]
        self.process = subprocess.Popen(command, env=self.env, cwd=root, stdout=subprocess.PIPE,
                                        stderr=self.stderr, text=True, encoding="utf-8")
        ready, _, _ = select.select([self.process.stdout], [], [], 30)
        if not ready:
            raise RuntimeError("service did not publish readiness within 30 seconds")
        line = self.process.stdout.readline()
        if not line:
            self.process.wait(timeout=5)
            self.stderr.seek(0)
            raise RuntimeError(f"service failed before readiness: {self.stderr.read()}")
        self.readiness = json.loads(line)
        if not self.readiness.get("ready"):
            raise RuntimeError(f"invalid service readiness: {line}")
        self.service_pid = self.readiness["service_pid"]

    def spawn(self, request: dict[str, Any], barrier: Path | None = None, disconnect: bool = False):
        command = [sys.executable, "-B", str(SCRIPT), "client", "--endpoint-file", str(self.endpoint)]
        if barrier:
            command += ["--start-file", str(barrier)]
        if disconnect:
            command += ["--disconnect"]
        process = subprocess.Popen(command, env=self.env, cwd=self.root, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        process.stdin.write(json.dumps(request))
        process.stdin.close()
        process.stdin = None
        return process

    @staticmethod
    def receive(process: subprocess.Popen) -> dict[str, Any]:
        out, error = process.communicate(timeout=75)
        if process.returncode:
            raise RuntimeError(f"client {process.pid} failed: {error}")
        reply = json.loads(out)
        if "error" in reply["response"]:
            raise RuntimeError(f"IPC failed: {reply['response']}")
        return reply

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.receive(self.spawn(request))["response"]

    def wait(self, job_id: int, stage: str = "final") -> dict[str, Any]:
        return self.call({"op": "wait", "job_id": job_id, "stage": stage, "wait_seconds": 15})

    def batch(self, requests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
        barrier = self.root / f"start-{secrets.token_hex(6)}"
        children = [self.spawn(request, barrier) for request in requests]
        # Barrier prevents compilation/service startup from polluting operation time.
        started = time.perf_counter()
        barrier.touch()
        replies = [self.receive(child) for child in children]
        elapsed = time.perf_counter() - started
        barrier.unlink()
        return replies, elapsed

    def stop(self) -> dict[str, Any]:
        try:
            result = self.call({"op": "stop"})
            self.process.wait(timeout=20)
            if self.process.returncode:
                self.stderr.seek(0)
                raise RuntimeError(f"service exit {self.process.returncode}: {self.stderr.read()}")
            return result
        finally:
            self.stderr.close()
            self.process.stdout.close()


def safe(**kwargs: Any) -> dict[str, Any]:
    return dict({"method": "check_action", "params": {"action": {"tool_name": "shell", "arguments": "echo safe"}},
                 "timeout": 15}, **kwargs)


def require(condition: bool, evidence: Any) -> None:
    if not condition:
        raise AssertionError(json.dumps(evidence, default=str))


def clean(reply: dict[str, Any]) -> bool:
    value = reply.get("result", {})
    receipt = value.get("execution", {})
    return (reply.get("final") is True and value.get("verdict") == "clean"
            and value.get("coverage", {}).get("complete") is True
            and value.get("action_executed") is False and reply.get("engine_calls") == 1
            and receipt.get("executor") == "neograph-engine"
            and receipt.get("nodes") == ["select_contracts", "load_evidence",
                                         "evaluate_contracts", "aggregate_findings"])


def benchmark(runner: Runner, workers: int) -> dict[str, Any]:
    runner.call({"op": "reset_metrics"})
    requests = [safe() for _ in range(16)]
    replies, elapsed = runner.batch(requests)
    results = [reply["response"] for reply in replies]
    pids = [reply["client_pid"] for reply in replies]
    require(len(set(pids)) == 16 and runner.service_pid not in pids, pids)
    natural_successes = sum(clean(value) for value in results)
    natural_failures = [value["failure"] for value in results if value.get("failure")]
    require(natural_successes > 0 and all(clean(value) or value.get("failure", {}).get("kind") == "index_busy" for value in results), results)
    if workers == 1:
        require(natural_successes == 16 and not natural_failures, results)
    natural = runner.call({"op": "stats"})
    require(natural["writer_lease_peak"] == 1 and natural["embedder"] is None and natural["local_only"], natural)

    runner.call({"op": "reset_metrics"})
    if workers > 1:
        first = runner.call(safe(wait=False, diagnostic_evaluation_hold=3.0))
        held = runner.wait(first["job_id"], "evaluating")
        require(not held["final"] and held["started"], held)
        metadata = runner.call({"method": "audit_status"})
        require(metadata["final"] and metadata.get("result", {}).get("ok") is True, metadata)
        second = runner.call(safe(wait=False, diagnostic_evaluation_hold=1.0))
        evaluating = runner.wait(second["job_id"], "evaluating")
        require(not evaluating["final"] and evaluating["started"], evaluating)
        overlap = runner.call({"op": "stats"})
        require(overlap["evaluation_peak"] >= 2 and overlap["writer_lease_peak"] == 1, overlap)
        parallel_results = [runner.wait(first["job_id"]), runner.wait(second["job_id"])]
    else:
        serial, _ = runner.batch([safe(diagnostic_evaluation_hold=0.2) for _ in range(2)])
        parallel_results = [reply["response"] for reply in serial]
        overlap = runner.call({"op": "stats"})
        require(overlap["evaluation_peak"] == 1 and overlap["writer_lease_peak"] == 1, overlap)
        metadata = None
    require(all(clean(value) for value in parallel_results), parallel_results)

    runner.call({"op": "reset_metrics"})
    hold_seconds = max(2.0, runner.lock_timeout * 4)
    blocker = runner.call(safe(wait=False, diagnostic_hold=hold_seconds))
    holding = runner.wait(blocker["job_id"], "writer")
    require(holding["started"] and not holding["final"], holding)
    contenders, hold_elapsed = runner.batch([safe() for _ in range(3)])
    held = runner.wait(blocker["job_id"])
    require(clean(held), held)
    contenders = [reply["response"] for reply in contenders]
    failures = [r["failure"] for r in contenders if r.get("failure")]
    require(all(clean(r) or r.get("failure", {}).get("kind") == "index_busy" for r in contenders), contenders)
    if workers == 1:
        require(not failures, contenders)
    else:
        require(len(failures) >= 1, contenders)
    diagnostic = runner.call({"op": "stats"})
    require(diagnostic["writer_lease_peak"] == 1, diagnostic)
    if workers > 1:
        require(diagnostic["native_active_peak"] >= 2, diagnostic)
    return {"workers": workers, "service_pid": runner.service_pid,
        "index_dir": natural["index_dir"], "client_pids": pids, "natural": {
        "attempted_calls": 16, "successful_verdicts": natural_successes,
        "verdict_counts": {"clean": natural_successes},
        "index_busy_failures": natural_failures, "seconds": round(elapsed, 6),
        "successful_verdicts_per_second": round(natural_successes / elapsed, 3),
        "attempts_per_second": round(16 / elapsed, 3),
        "transport_seconds": [round(r["seconds"], 6) for r in replies],
        "native_active_peak": natural["native_active_peak"], "writer_lease_peak": natural["writer_lease_peak"],
        "evaluation_peak": natural["evaluation_peak"]},
        "native_execution": next(value["result"]["execution"] for value in results if clean(value)),
        "native_evaluation_overlap": {"evaluation_peak": overlap["evaluation_peak"],
            "writer_lease_peak": overlap["writer_lease_peak"], "results": parallel_results,
            "metadata_during_evaluation": metadata, "synthetic_phase_hold": True},
        "residual_writer_contention": {"hold_seconds": hold_seconds, "contenders": 3,
            "seconds": round(hold_elapsed, 6), "successful_verdicts": sum(clean(r) for r in contenders),
            "index_busy_failures": failures, "native_active_peak": diagnostic["native_active_peak"],
            "writer_lease_peak": diagnostic["writer_lease_peak"]}}


def edge_cases(runner: Runner) -> dict[str, Any]:
    # One-worker service gives deterministic unclaimed pending jobs.
    forbidden = runner.call({"method": "check_action", "params": {"action": {
        "tool_name": "shell", "arguments": f"echo {SENTINEL}"}}})
    require(forbidden.get("result", {}).get("verdict") == "violation", forbidden)
    indexed = runner.call({"method": "sync_session", "params": {"embed": False}})
    audited = runner.call({"method": "audit_session"})
    status = runner.call({"method": "audit_status"})
    require(indexed.get("result", {}).get("ok") is True, indexed)
    require(audited.get("result", {}).get("verdict") == "clean", audited)
    require(status.get("result", {}).get("embedder") is None, status)

    blocker = runner.call(safe(wait=False, diagnostic_hold=2.0))
    require(runner.wait(blocker["job_id"], "writer")["started"], blocker)
    expired = runner.call(safe(wait=False, timeout=0.12))
    queued = runner.call(safe(wait=False))
    rejected = runner.call(safe(wait=False))
    require(rejected["accepted"] is False and rejected.get("reason") == "queue_full" and rejected["engine_calls"] == 0, rejected)
    expired = runner.wait(expired["job_id"])
    queued = runner.wait(queued["job_id"])
    require(expired["status"] == "not_started" and expired["reason"] == "queued_expired" and expired["engine_calls"] == 0 and not expired["started"], expired)
    require(clean(queued) and clean(runner.wait(blocker["job_id"])), queued)

    # Half-close an accepted waiting connection; this is not ordinary durable submit.
    blocker = runner.call(safe(wait=False, diagnostic_hold=1.5))
    require(runner.wait(blocker["job_id"], "writer")["started"], blocker)
    disconnected = runner.receive(runner.spawn(safe(), disconnect=True))
    require(disconnected["response"].get("disconnected"), disconnected)
    require(clean(runner.wait(blocker["job_id"])), blocker)
    # IDs are sequential in this private service: the disconnected submit is next.
    disconnected_job = runner.wait(blocker["job_id"] + 1)
    require(disconnected_job.get("reason") == "client_disconnected" and disconnected_job["engine_calls"] == 0, disconnected_job)

    running = runner.call(safe(timeout=0.15, diagnostic_evaluation_hold=0.7))
    require(running["status"] == "still_running" and running["started"] and running.get("outcome") == "unknown" and not running["final"], running)
    final = runner.wait(running["job_id"])
    require(final["final"] and final["native_settled"] and final["deadline_passed"]
            and final.get("failure", {}).get("kind") == "request_deadline"
            and final.get("result") is None and final["engine_calls"] == 1, final)

    blocker = runner.call(safe(wait=False, diagnostic_hold=2.0))
    require(runner.wait(blocker["job_id"], "writer")["started"], blocker)
    unclaimed = [runner.call(safe(wait=False)) for _ in range(2)]
    closed = runner.call({"op": "close"})
    settlements = [runner.wait(job["job_id"]) for job in unclaimed]
    require(all(r["final"] and r["native_settled"] and not r["started"] and not r["native_claimed"] and r["engine_calls"] == 0 and r.get("reason") == "queue_closed" and r.get("failure", {}).get("kind") == "queue_closed" for r in settlements), settlements)
    require(clean(runner.wait(blocker["job_id"])), blocker)
    require(closed["native"]["pending"] == closed["native"]["active"] == 0 and closed["collector_error"] is None, closed)
    return {"sentinel_verdict": forbidden["result"]["verdict"], "session_verdict": audited["result"]["verdict"],
            "sentinel_execution": forbidden["result"]["execution"],
            "session_execution": audited["result"]["execution"],
            "sync_total_chunks": indexed["result"].get("total_chunks"), "queue_full": rejected,
            "queued_expiry": expired, "disconnect_cancellation": disconnected_job,
            "running_deadline": running, "running_settlement": final,
            "shutdown_cancellations": settlements, "native_final": closed["native"]}


def run(args: argparse.Namespace) -> None:
    if args.neograph_root is None:
        raise ValueError("run requires --neograph-root; no personal checkout is assumed")
    with tempfile.TemporaryDirectory(prefix="self-direct-shared-queue-") as directory:
        root = Path(directory)
        (root / "poc-owned.json").write_text(json.dumps({"owner_pid": os.getpid()}), encoding="utf-8")
        make_fixture(root, args.events)
        library = compile_bridge(root, args.neograph_root, args.compiler)
        comparisons = []
        for workers in (1, args.workers):
            runner = Runner(root, library, workers, 32, args.lock_timeout)
            try:
                comparisons.append(benchmark(runner, workers))
            finally:
                runner.stop()
        runner = Runner(root, library, 1, 2, args.lock_timeout)
        try:
            edges = edge_cases(runner)
        finally:
            runner.stop()
        require(comparisons[0]["workers"] == 1 and comparisons[1]["workers"] == args.workers, comparisons)
        require(comparisons[0]["index_dir"] == comparisons[1]["index_dir"] == str(root / "index"), comparisons)
        report = {"ok": True, "backend": BACKEND, "shared_index": True,
                  "index_identity": str(root / "index"), "acceptance_service_pid": runner.service_pid,
                  "events": args.events, "local_only": True, "embedding_calls": 0,
                  "comparison": comparisons, "acceptance": edges,
                  "measurement_note": "Natural throughput includes independent client startup/IPC. Writer lease and native audit evaluation phases are measured separately. Native-stage and residual writer holds are labeled synthetic fixtures; engine/NeoGraph results are real. No performance threshold.",
                  "platform_limits": "Requires POSIX IPv4 loopback, an installed GCC-compatible C++20 compiler, NeoGraph deps/concurrentqueue.h, and installed project/NeoGraph Python dependencies."}
    # This line is emitted only after service shutdown and scratch cleanup.
    report["scratch_removed"] = not root.exists()
    require(report["scratch_removed"], report)
    print(json.dumps(report, separators=(",", ":")), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode")
    for mode in ("run", "serve"):
        option = sub.add_parser(mode)
        option.add_argument("--neograph-root", type=Path)
        option.add_argument("--compiler", default="c++")
        option.add_argument("--events", type=int, default=128)
        option.add_argument("--lock-timeout", type=float, default=1.0)
        option.add_argument("--workers", type=int, default=4 if mode == "run" else 1,
                            help="Native workers (run compares one worker with this value)")
        if mode == "serve":
            option.add_argument("--capacity", type=int, default=32)
            option.add_argument("--endpoint-file", type=Path)
            option.add_argument("--scratch-root", type=Path, help=argparse.SUPPRESS)
            option.add_argument("--library", type=Path, help=argparse.SUPPRESS)
    option = sub.add_parser("client")
    option.add_argument("--endpoint-file", type=Path, required=True)
    option.add_argument("--request", help="JSON request; otherwise read JSON from stdin")
    option.add_argument("--start-file", type=Path, help=argparse.SUPPRESS)
    option.add_argument("--disconnect", action="store_true", help="Diagnostic half-close of a waiting request")
    arguments = sys.argv[1:]
    if not arguments or arguments[0].startswith("--"):
        arguments = ["run", *arguments]
    args = parser.parse_args(arguments)
    try:
        if args.mode in ("run", "serve"):
            if args.events < 1 or not math.isfinite(args.lock_timeout) or not 0 < args.lock_timeout <= 2:
                raise ValueError("events must be positive; lock-timeout must be (0,2] seconds for bounded diagnostic holds")
            if args.mode == "run" and args.workers < 2:
                raise ValueError("run requires --workers >= 2 to demonstrate parallel evaluations")
            if args.mode == "serve" and (args.workers < 1 or args.capacity < 1):
                raise ValueError("workers and capacity must be positive")
        {"run": run, "serve": serve, "client": client}[args.mode](args)
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__, "message": str(exc)}), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
