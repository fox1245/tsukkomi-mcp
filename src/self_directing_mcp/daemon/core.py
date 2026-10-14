"""The resident daemon: one process per user and OS environment (``python -m self_directing_mcp.daemon``).

What it is. A Unix-socket server (no TCP) that runs the SAME judgement code as the standalone processes:
``claude_hooks.run_hook`` and ``agy_hooks.run_hook`` for command hooks, the same FastMCP tool functions for MCP
connections. It exists to avoid paying interpreter start, imports and engine setup on every hook event, and to
replace one MCP server process per session with one shared process.

Authentication boundary. The runtime directory is private to the user (see ``rundir``) and every accepted
connection's ``SO_PEERCRED`` user id must equal the daemon's. That is all: processes of the same user are NOT
isolated from each other, exactly as they are not isolated from each other's files today.

Lifecycle. A lock file (``daemon.lock``, ``flock``) is held for the process's lifetime: a second daemon exits at
once. The daemon leaves when it has had no connection and no request for ``idle_exit_sec``, when its resident set
passes ``max_rss_mb`` after a request (it stops accepting, finishes in-flight requests, exits; the next client starts
a new one), on SIGTERM/SIGINT, or when a NEWER client asks it to ``drain``. An older client never replaces it.

Concurrency. Hook requests run on their own thread pool, MCP tool calls on another (the heavy lane), so a long
``sync_session`` never queues a ``PreToolUse``. Requests are bounded per session and in total: beyond the bound the
answer is ``unknown(queue_full)``. Engines are created lazily, one per resolved-settings key, and kept; an engine
serializes its own operations as it does in a standalone process.

Logging (``daemon.log`` in the runtime directory): event names, sizes, durations and error classes. Request and
response bodies, payloads, paths and exception messages are never logged.
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import contextvars
import fcntl
import hashlib
import json
import os
import signal
import socket
import struct
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

from self_directing_mcp import __version__
from self_directing_mcp.daemon import envview, protocol as P, rundir
from self_directing_mcp.daemon.served import ServedLedger

LOCK_WAIT_SEC = 10.0              # a freshly started daemon waits this long for a draining predecessor to exit
HELLO_TIMEOUT_SEC = 5.0
REQUEST_MARGIN_SEC = 2.0          # the daemon's cooperative deadline is the request timeout plus this
DEFAULT_REQUEST_TIMEOUT_SEC = 45.0
LOG_ROTATE_BYTES = 256 * 1024
SHUTDOWN_WATCHDOG_SEC = 20.0
DRAIN_TIMEOUT_SEC = 150.0         # a drain never waits longer than this for in-flight requests
SESSION_BOUND = 16                # queued + running hook requests per session
TOTAL_BOUND = 128                 # ... and across all sessions
HOOK_WORKERS = 8
HEAVY_WORKERS = 4
PEERCRED = struct.Struct("3i")    # pid, uid, gid
EXIT_RUNTIME_DIR = 78             # EX_CONFIG: the runtime directory is unusable
EXIT_BAD_CONFIG = 2


# --- configuration ------------------------------------------------------------------------------------------------

@dataclass
class DaemonConfig:
    idle_exit_sec: float = 900.0
    max_rss_mb: float = 400.0
    session_bound: int = SESSION_BOUND
    total_bound: int = TOTAL_BOUND
    hook_workers: int = HOOK_WORKERS
    heavy_workers: int = HEAVY_WORKERS

    @classmethod
    def from_environment(cls, environ=None) -> "DaemonConfig":
        """``SELF_DIRECT_DAEMON_IDLE_EXIT_SEC`` (900), ``..._MAX_RSS_MB`` (400), ``..._SESSION_BOUND`` (16).

        A malformed value is an error (``ValueError``), never a silent default."""
        environ = os.environ if environ is None else environ
        config = cls()
        for name, attribute, kind in (("SELF_DIRECT_DAEMON_IDLE_EXIT_SEC", "idle_exit_sec", float),
                                      ("SELF_DIRECT_DAEMON_MAX_RSS_MB", "max_rss_mb", float),
                                      ("SELF_DIRECT_DAEMON_SESSION_BOUND", "session_bound", int)):
            text = environ.get(name)
            if text is None:
                continue
            try:
                value = kind(text.strip())
            except ValueError:
                raise ValueError(f"invalid value for {name}") from None
            if not value > 0 or value != value or value == float("inf"):
                raise ValueError(f"invalid value for {name}")
            setattr(config, attribute, value)
        return config


def read_rss_mb() -> float:
    """Resident set size of this process in MiB (``/proc/self/statm``); a single seam for tests."""
    try:
        with open("/proc/self/statm", "rb") as stream:
            pages = int(stream.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return 0.0


def peer_credentials(sock) -> tuple[int, int, int]:
    """``(pid, uid, gid)`` of the peer that connected to this Unix socket."""
    return PEERCRED.unpack(sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, PEERCRED.size))


# --- logging ------------------------------------------------------------------------------------------------------

class DaemonLog:
    """JSON lines of scalars only. A string longer than 64 characters or with other than identifier characters is
    dropped, so a field can never carry request text by accident."""

    SAFE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")

    def __init__(self, runtime: rundir.RuntimeDir | None) -> None:
        self.runtime = runtime
        self._fd = None
        self._lock = threading.Lock()

    def _open(self):
        if self._fd is None and self.runtime is not None:
            self._fd = self.runtime.open_file(rundir.LOG_NAME, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        return self._fd

    def event(self, name: str, **fields) -> None:
        record = {"t": round(time.time(), 3), "ev": name}
        for key, value in fields.items():
            if isinstance(value, bool) or isinstance(value, (int, float)):
                record[key] = value
            elif isinstance(value, str) and len(value) <= 64 and set(value) <= self.SAFE:
                record[key] = value
        try:
            with self._lock:
                fd = self._open()
                if fd is None:
                    return
                if os.fstat(fd).st_size > LOG_ROTATE_BYTES:
                    os.close(fd)
                    self._fd = None
                    os.rename(rundir.LOG_NAME, rundir.LOG_NAME + ".1", src_dir_fd=self.runtime.fd,
                              dst_dir_fd=self.runtime.fd)
                    fd = self._open()
                os.write(fd, (json.dumps(record, separators=(",", ":")) + "\n").encode())
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None


# --- engines ------------------------------------------------------------------------------------------------------

def settings_key(settings) -> str:
    """Identity of a resolved settings object. Secrets enter only as a digest."""
    values = settings.model_dump()
    secret = values.get("openrouter_api_key")
    values["openrouter_api_key"] = hashlib.sha256(secret.encode()).hexdigest() if secret else None
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


class Engines:
    """One engine per resolved-settings key, created lazily and kept for reuse. The daemon is the only writer of its
    indexes, and the resident integrity worker starts once per engine (not per connection)."""

    def __init__(self, factory=None, log: DaemonLog | None = None) -> None:
        self._factory = factory
        self._lock = threading.Lock()
        self._engines: dict[str, object] = {}
        self._creating: dict[str, threading.Lock] = {}
        self._workers: dict[str, object] = {}
        self._log = log

    def get(self, settings):
        key = settings_key(settings)
        engine = self._engines.get(key)
        if engine is not None:
            return engine
        with self._lock:
            guard = self._creating.setdefault(key, threading.Lock())
        with guard:
            engine = self._engines.get(key)
            if engine is None:
                engine = self._create(settings)
                self._start_worker(key, engine)
                self._engines[key] = engine
                if self._log:
                    self._log.event("engine_created", engines=len(self._engines))
        return engine

    def _create(self, settings):
        if self._factory is not None:
            return self._factory(settings)
        from self_directing_mcp.engine import SelfDirectEngine
        return SelfDirectEngine(settings)

    def _start_worker(self, key, engine) -> None:
        if self._factory is not None:
            return
        from self_directing_mcp.integrity_worker import start_integrity_worker
        worker = start_integrity_worker(engine)
        if worker is not None:
            self._workers[key] = worker

    def __len__(self) -> int:
        return len(self._engines)

    def worker_count(self) -> int:
        return len(self._workers)

    def close_all(self) -> None:
        with self._lock:
            engines, self._engines = list(self._engines.items()), {}
        for key, engine in engines:
            worker = self._workers.pop(key, None)
            try:
                if worker is not None:
                    worker.stop(timeout=2.0)
                engine.close()
            except Exception:
                pass


# --- the daemon ---------------------------------------------------------------------------------------------------

class Daemon:
    def __init__(self, runtime: rundir.RuntimeDir, config: DaemonConfig | None = None, *, hook_runner=None,
                 engine_factory=None, version: str = __version__, proto: int = P.PROTO,
                 install_signals: bool = False, exit_watchdog: bool = False, mcp_server=None) -> None:
        self.runtime = runtime
        self.config = config or DaemonConfig()
        self.version, self.proto = version, proto
        self.log = DaemonLog(runtime)
        self.engines = Engines(engine_factory, self.log)
        self.ledger = ServedLedger()
        self._hook_runner = hook_runner or self._run_hook
        self._install_signals = install_signals
        self._exit_watchdog = exit_watchdog          # a process entry only: a hung worker must not keep the process
        self._mcp = mcp_server                  # an object with ``async serve(daemon, reader, writer, frame)``
        self._hook_pool = concurrent.futures.ThreadPoolExecutor(self.config.hook_workers,
                                                                thread_name_prefix="tsukkomi-hook")
        self._heavy_pool = concurrent.futures.ThreadPoolExecutor(self.config.heavy_workers,
                                                                 thread_name_prefix="tsukkomi-heavy")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._writers: set = set()
        self._tasks: set = set()
        self.draining = False
        self.drain_reason: str | None = None
        self.inflight = 0
        self.inflight_by_session: Counter = Counter()
        self.last_activity = time.monotonic()
        self.started_at = time.time()
        self.stats: Counter = Counter()
        self.info = {"index_schema": None, "parser_versions": None, "evaluator_version": None}

    # -- public state ----------------------------------------------------------------------------------------------
    @property
    def connections(self) -> int:
        return len(self._writers)

    def versions(self) -> tuple[int, tuple[int, ...]]:
        return self.proto, P.parse_version(self.version)

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def snapshot(self) -> dict:
        return {"pid": os.getpid(), "uptime_sec": round(time.time() - self.started_at, 1), "inflight": self.inflight,
                "connections": self.connections, "engines": len(self.engines),
                "integrity_workers": self.engines.worker_count(), "draining": self.draining,
                "drain_reason": self.drain_reason, "rss_mb": round(read_rss_mb(), 1),
                "requests": dict(self.stats)}

    # -- startup ---------------------------------------------------------------------------------------------------
    def _bind(self) -> socket.socket:
        self.runtime.unlink(rundir.SOCKET_NAME)             # the lock is ours: whatever is there is stale
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            old = os.umask(0o177)
            try:
                sock.bind(self.runtime.address())
            finally:
                os.umask(old)
            sock.setblocking(False)
        except BaseException:
            sock.close()
            raise
        return sock

    def _write_pid(self) -> None:
        fd = self.runtime.open_file(rundir.PID_NAME, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            os.write(fd, f"{os.getpid()}\n".encode())
        finally:
            os.close(fd)

    def _warm(self) -> None:
        """Import the judgement code and collect version facts while the socket already answers ``hello``."""
        try:
            from self_directing_mcp import agy, agy_hooks, claude, claude_hooks, omp, schemas, timeseries  # noqa: F401
            from self_directing_mcp.audit import incremental
            from self_directing_mcp.codex import parse as codex_parse
            from self_directing_mcp import engine, graphrag  # noqa: F401
            self.info["index_schema"] = {"incremental_state": incremental.STATE_VERSION,
                                         "analysis": timeseries.ANALYSIS_SCHEMA_VERSION,
                                         "graph": graphrag.GRAPH_SCHEMA_VERSION, "contracts": schemas.SCHEMA_VERSION}
            self.info["parser_versions"] = {"codex": codex_parse.PARSER_VERSION, "omp": omp.PARSER_VERSION,
                                            "claude": claude.PARSER_VERSION, "agy": agy.PARSER_VERSION}
            self.info["evaluator_version"] = incremental.EVALUATOR_VERSION
        except Exception as exc:
            self.log.event("warm_failed", error=type(exc).__name__)
        finally:
            self._ready.set()

    async def serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._loop.set_default_executor(self._heavy_pool)    # MCP tool calls (asyncio.to_thread) use the heavy lane
        self._stop = asyncio.Event()
        sock = self._bind()
        self._server = await asyncio.start_unix_server(self._on_connect, sock=sock, limit=P.MAX_FRAME_BYTES)
        self._write_pid()
        if self._install_signals:
            for number in (signal.SIGTERM, signal.SIGINT):
                self._loop.add_signal_handler(number, self.begin_drain, "signal")
        threading.Thread(target=self._warm, name="tsukkomi-warm", daemon=True).start()
        idle_task = asyncio.create_task(self._watch_idle())
        self.log.event("started", pid=os.getpid(), version=self.version, proto=self.proto)
        try:
            await self._stop.wait()
        finally:
            idle_task.cancel()
            await self._shutdown()

    async def _shutdown(self) -> None:
        if self._exit_watchdog:
            watchdog = threading.Timer(SHUTDOWN_WATCHDOG_SEC, os._exit, args=(1,))
            watchdog.daemon = True
            watchdog.start()
        if self._server is not None:
            self._server.close()
        self.runtime.unlink(rundir.SOCKET_NAME)
        for writer in list(self._writers):
            writer.close()
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.engines.close_all)
        self.ledger.flush()                               # the checkpoint: counters are written, engines closed
        self._hook_pool.shutdown(wait=False, cancel_futures=True)
        self.runtime.unlink(rundir.PID_NAME)
        self.log.event("stopped", reason=self.drain_reason or "stop", requests=sum(self.stats.values()))
        self.log.close()

    # -- connections -----------------------------------------------------------------------------------------------
    def _authorized(self, writer) -> bool:
        try:
            _pid, uid, _gid = peer_credentials(writer.get_extra_info("socket"))
        except (OSError, struct.error, AttributeError):
            self.log.event("peer_rejected", why="no_credentials")
            return False
        if uid != rundir.current_uid():
            self.log.event("peer_rejected", why="uid_mismatch")
            return False
        return True

    async def _on_connect(self, reader, writer) -> None:
        task = asyncio.current_task()
        self._tasks.add(task)
        self._writers.add(writer)
        self.touch()
        try:
            if self._authorized(writer):
                await self._serve_connection(reader, writer)
            else:
                self.stats["rejected_peer"] += 1
        except (P.ConnectionClosed, ConnectionError, asyncio.CancelledError):
            pass
        except (P.ProtocolError, asyncio.TimeoutError) as exc:
            self.stats["protocol_errors"] += 1
            self.log.event("protocol_error", error=type(exc).__name__)
        except Exception as exc:
            self.stats["internal_errors"] += 1
            self.log.event("connection_error", error=type(exc).__name__)
        finally:
            self._writers.discard(writer)
            self._tasks.discard(task)
            writer.close()
            self.touch()

    def hello_reply(self, relation: str) -> dict:
        return {"op": P.OP_HELLO, "ok": True, "proto": self.proto, "package_version": self.version,
                **self.info, "features": ["hook", "mcp", "drain", "status"], "pid": os.getpid(),
                "draining": self.draining, "relation": relation}

    async def _serve_connection(self, reader, writer) -> None:
        hello = await asyncio.wait_for(P.read_frame_async(reader), HELLO_TIMEOUT_SEC)
        if hello.get("op") != P.OP_HELLO:
            raise P.ProtocolError("first frame must be hello")
        relation = P.compare(P.versions_of(hello), self.versions())
        await P.write_frame_async(writer, self.hello_reply(relation))
        while True:
            frame = await P.read_frame_async(reader)
            op = frame.get("op")
            if op == P.OP_HOOK:
                reply = await self._op_hook(frame, relation)
            elif op == P.OP_DRAIN:
                reply = self._op_drain(frame, relation)
            elif op == P.OP_STATUS:
                reply = {"op": "status", "ok": True, **self.snapshot()}
            elif op == P.OP_MCP:
                await self._op_mcp(frame, relation, reader, writer)
                return
            else:
                reply = self._unknown(P.REASON_BAD_REQUEST)
            await P.write_frame_async(writer, reply)

    @staticmethod
    def _unknown(reason: str) -> dict:
        return {"op": "result", "ok": False, "status": "unknown", "reason": reason}

    # -- operations ------------------------------------------------------------------------------------------------
    def _op_drain(self, frame: dict, relation: str) -> dict:
        if relation != "newer" and frame.get("force") is not True:
            return self._unknown(P.REASON_VERSION_MISMATCH)
        self.begin_drain("version" if relation == "newer" else "requested")
        return {"op": "drain", "ok": True, "draining": True}

    async def _op_mcp(self, frame, relation, reader, writer) -> None:
        if relation != "same":
            await P.write_frame_async(writer, {**self._unknown(P.REASON_VERSION_MISMATCH), "op": "mcp"})
            return
        if self.draining:
            await P.write_frame_async(writer, {**self._unknown(P.REASON_DRAINING), "op": "mcp"})
            return
        if self._mcp is None:
            await P.write_frame_async(writer, {**self._unknown(P.REASON_BAD_REQUEST), "op": "mcp"})
            return
        await self._mcp.serve(self, reader, writer, frame)

    async def _op_hook(self, frame: dict, relation: str) -> dict:
        if relation != "same":
            self.stats["version_mismatch"] += 1
            return self._unknown(P.REASON_VERSION_MISMATCH)
        if self.draining:
            self.stats["draining"] += 1
            return self._unknown(P.REASON_DRAINING)
        host, event, payload = frame.get("host"), frame.get("event"), frame.get("payload")
        if host not in P.HOSTS or not isinstance(event, (str, type(None))) or "payload" not in frame:
            return self._unknown(P.REASON_BAD_REQUEST)
        try:
            env, argv = P.validate_environment(frame.get("env", {}), frame.get("argv", []))
        except P.ProtocolError:
            return self._unknown(P.REASON_BAD_REQUEST)
        session = P.session_key(host, payload)
        if self.inflight_by_session[session] >= self.config.session_bound or self.inflight >= self.config.total_bound:
            self.stats["queue_full"] += 1
            self.log.event("queue_full", host=host, inflight=self.inflight)
            return self._unknown(P.REASON_QUEUE_FULL)
        self.begin_request(session)
        started = time.perf_counter()
        ran = [0.0]

        def timed(*args):                                   # time spent in the lane thread: settings + judgement
            begun = time.perf_counter()
            try:
                return self._hook_runner(*args)
            finally:
                ran[0] = (time.perf_counter() - begun) * 1000

        try:
            context = contextvars.copy_context()
            response = await self._loop.run_in_executor(self._hook_pool, context.run, timed,
                                                        host, event, payload, env, argv)
        except Exception as exc:
            self.stats["hook_errors"] += 1
            self.log.event("hook_error", host=host, error=type(exc).__name__)
            return self._unknown(P.REASON_INTERNAL)
        finally:
            self.end_request(session)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.stats["hooks"] += 1
        self.log.event("hook", host=host, event=event if isinstance(event, str) else "none",
                       ms=round(elapsed_ms, 2), run_ms=round(ran[0], 2))
        self.after_request()
        return {"op": "result", "ok": True, "response": response, "served_by": "daemon", "ms": round(elapsed_ms, 2)}

    # -- hook execution (a hook-lane thread) --------------------------------------------------------------------------
    def _run_hook(self, host: str, event, payload, env: dict, argv: list) -> dict:
        """The standalone hook's body with the daemon's engine: the same ``run_hook``, the same response."""
        from self_directing_mcp.request_control import request_scope
        from self_directing_mcp.security.mask import mask_value
        from self_directing_mcp.settings_core import load_hook_settings
        if host == "claude":
            from self_directing_mcp import claude_hooks as module
            env = P.apply_claude_flags(argv, env)
            light = module.LIGHT_EVENTS
        else:
            from self_directing_mcp import agy_hooks as module
            light = ()
        with envview.request_environment(env):
            settings, engine = None, None
            try:
                settings = load_hook_settings()
                engine = (SimpleNamespace(settings=settings) if event in light else self.engines.get(settings))
            except Exception:
                # Settings or engine unavailable: ``run_hook`` with no engine takes exactly the path a standalone
                # process takes and answers with the same fail-open response.
                engine = None
            budget = (settings.request_timeout_sec if settings is not None else DEFAULT_REQUEST_TIMEOUT_SEC)
            with request_scope(deadline=time.monotonic() + budget + REQUEST_MARGIN_SEC):
                response = module.run_hook(event, payload, engine=engine)
            response = mask_value(response)
        if settings is not None:
            self.ledger.add(host, settings.index_dir, event if isinstance(event, str) else "none")
        return response

    # -- bookkeeping, idle exit, memory cap, draining --------------------------------------------------------------------
    def begin_request(self, session: str | None = None) -> None:
        self.inflight += 1
        if session is not None:
            self.inflight_by_session[session] += 1
        self.touch()

    def end_request(self, session: str | None = None) -> None:
        self.inflight -= 1
        if session is not None:
            self.inflight_by_session[session] -= 1
            if self.inflight_by_session[session] <= 0:
                del self.inflight_by_session[session]
        self.touch()

    def after_request(self) -> None:
        """The memory cap: past it, stop accepting, finish what is running, exit; the next client starts a new one."""
        if self.config.max_rss_mb and read_rss_mb() > self.config.max_rss_mb:
            self.begin_drain("rss")
        elif self.ledger.due():
            threading.Thread(target=self.ledger.flush, daemon=True).start()

    async def _watch_idle(self) -> None:
        tick = max(0.05, min(1.0, self.config.idle_exit_sec / 10))
        while True:
            await asyncio.sleep(tick)
            if (not self.draining and not self._writers and self.inflight == 0
                    and time.monotonic() - self.last_activity >= self.config.idle_exit_sec):
                self.begin_drain("idle")
                return
            if self.ledger.due():
                self._loop.run_in_executor(None, self.ledger.flush)

    def begin_drain(self, reason: str) -> None:
        """Stop accepting (the socket path disappears, so new clients start a successor), finish in-flight
        requests, then leave. Idempotent."""
        if self.draining:
            return
        self.draining, self.drain_reason = True, reason
        self.log.event("draining", reason=reason, inflight=self.inflight)
        if self._server is not None:
            self._server.close()
        self.runtime.unlink(rundir.SOCKET_NAME)
        if self._loop is not None:
            self._loop.create_task(self._finish_drain())

    async def _finish_drain(self) -> None:
        deadline = time.monotonic() + DRAIN_TIMEOUT_SEC
        while self.inflight > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        self._stop.set()


# --- process entry ----------------------------------------------------------------------------------------------------

def acquire_instance_lock(runtime: rundir.RuntimeDir, wait_sec: float = LOCK_WAIT_SEC):
    """The single-instance lock; waits for a draining predecessor. Returns the open file descriptor or None."""
    fd = runtime.open_file(rundir.LOCK_NAME, os.O_RDWR | os.O_CREAT)
    deadline = time.monotonic() + wait_sec
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                return None
            time.sleep(0.05)


def query(runtime: rundir.RuntimeDir, frame: dict, timeout: float = 3.0) -> dict | None:
    """One request to a running daemon (``--status`` / ``--stop``); None when none is reachable."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(runtime.address())
        P.write_frame(sock, P.hello_request("cli", __version__))
        P.read_frame(sock)
        P.write_frame(sock, frame)
        return P.read_frame(sock)
    except (OSError, P.ProtocolError, EOFError):
        return None
    finally:
        sock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m self_directing_mcp.daemon",
                                     description="Resident Tsukkomi daemon (experimental). Normally started by a client.")
    parser.add_argument("--status", action="store_true", help="print the running daemon's counters and exit")
    parser.add_argument("--stop", action="store_true", help="ask the running daemon to drain and exit")
    args = parser.parse_args(argv)
    try:
        runtime = rundir.open_runtime_dir()
    except rundir.RuntimeDirError as exc:
        print(f"tsukkomi daemon: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_DIR
    try:
        if args.status or args.stop:
            reply = query(runtime, {"op": P.OP_STATUS} if args.status else {"op": P.OP_DRAIN, "force": True})
            print(json.dumps(reply) if reply is not None else "no daemon is running")
            return 0 if reply is not None else 1
        try:
            config = DaemonConfig.from_environment()
        except ValueError as exc:
            print(f"tsukkomi daemon: {exc}", file=sys.stderr)
            return EXIT_BAD_CONFIG
        lock = acquire_instance_lock(runtime)
        if lock is None:
            return 0                                   # another daemon is running: nothing to do
        envview.install()
        from self_directing_mcp.daemon.mcp_session import McpHandler
        daemon = Daemon(runtime, config, mcp_server=McpHandler(), install_signals=True, exit_watchdog=True)
        try:
            asyncio.run(daemon.serve())
        finally:
            os.close(lock)
        return 0
    finally:
        runtime.close()
