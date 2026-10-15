"""Helpers for the daemon tests: isolated runtime directories, in-process daemons, synthetic worlds, hook runs.

Every daemon a test starts lives under a temporary runtime directory (``XDG_RUNTIME_DIR`` pointing into
``tmp_path``) and a temporary index; nothing here reads or touches a user's real runtime directory, sessions or
credentials. ``tests/daemon/conftest.py`` asserts at the end of every test and of the session that no daemon
process survived.
"""
from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

from self_directing_mcp.daemon import protocol as P, rundir

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
PY = sys.executable

SID = "aaaaaaaa-1111-4111-8111-aaaaaaaa0001"
AGY_SID = "11111111-1111-4111-8111-111111111111"
ENV_PREFIXES = ("SELF_DIRECT_", "CODEX_", "CLAUDE_", "OPENROUTER", "TSUKKOMI_", "XDG_")

CONTRACTS = [
    {"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"},
    {"id": "tests-before-deploy", "type": "must", "scope": "tool_call", "regex": "pytest",
     "before_regex": "deploy", "requires_success": True},
    {"id": "tests-ran", "type": "must", "scope": "tool_call", "regex": "pytest"},
]
HIGH_RISK = {"id": "no-prod-deploy", "type": "must_not", "scope": "tool_call", "regex": r"deploy\s+--prod",
             "risk": "high", "description": "synthetic high-risk contract"}
# A completion obligation that is not met in the synthetic session: a Stop must ask the agent to continue.
UNMET_AUTO_CONTINUE = {"id": "lint-ran", "type": "must", "scope": "tool_call", "regex": r"never-run-this-linter",
                       "auto_continue": True, "description": "synthetic obligation"}

_bench = None
STARTED: set[int] = set()          # every daemon or spawned process pid any test registered or left behind


def bench():
    global _bench
    if _bench is None:
        spec = importlib.util.spec_from_file_location("bench_hooks_for_daemon_tests", SCRIPTS / "bench_hooks.py")
        _bench = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _bench
        spec.loader.exec_module(_bench)
    return _bench


def pid_alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", "rb") as stream:
            return stream.read().rsplit(b")", 1)[1].split()[0] not in (b"Z", b"X")
    except (OSError, IndexError):
        return False


def is_test_process(pid: int) -> bool:
    """True only for a daemon or a test stub: a recorded pid may have been reused by anything else, and nothing else
    is ever signalled."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as stream:
            command = stream.read()
    except OSError:
        return False
    return b"self_directing_mcp.daemon" in command or b"tsukkomi-test-stub" in command


def runtime_pids(directory: Path) -> set[int]:
    """Pids a runtime directory knows about: the daemon's pid file and every recorded spawn."""
    found: set[int] = set()
    try:
        found.add(int((directory / rundir.PID_NAME).read_text().strip()))
    except (OSError, ValueError):
        pass
    try:
        for record in json.loads((directory / rundir.SPAWNS_NAME).read_text()):
            if record.get("pid"):
                found.add(int(record["pid"]))
    except (OSError, ValueError, TypeError):
        pass
    return found


class Runtime:
    """One test's private runtime area: its own ``XDG_RUNTIME_DIR`` and ``HOME``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.xdg = root / "xdg"
        self.xdg.mkdir(mode=0o700)
        os.chmod(self.xdg, 0o700)
        self.home = root / "home"
        self.home.mkdir()
        self.dir = self.xdg / "tsukkomi"
        self.extra_pids: set[int] = set()

    def environ(self) -> dict:
        return {"XDG_RUNTIME_DIR": str(self.xdg), "HOME": str(self.home)}

    def pids(self) -> set[int]:
        return runtime_pids(self.dir) | self.extra_pids

    def alive(self) -> set[int]:
        return {pid for pid in self.pids() if pid_alive(pid) and is_test_process(pid)}

    def stop(self, timeout: float = 10.0) -> None:
        """Ask a running daemon to drain (the CLI's ``--stop``), then wait for every known pid to disappear."""
        if self.dir.exists():
            try:
                runtime = rundir.open_runtime_dir(self.environ(), create=False)
            except rundir.RuntimeDirError:
                runtime = None
            if runtime is not None:
                try:
                    from self_directing_mcp.daemon import core
                    core.query(runtime, {"op": P.OP_DRAIN, "force": True}, timeout=3.0)
                finally:
                    runtime.close()
        deadline = time.monotonic() + timeout
        while self.alive() and time.monotonic() < deadline:
            time.sleep(0.05)

    def kill_leftovers(self) -> set[int]:
        left = self.alive()
        for pid in left:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        return left


class InProcessDaemon:
    """A real ``Daemon`` running on its own event loop in a thread of the test process."""

    def __init__(self, runtime: Runtime, config=None, **kwargs) -> None:
        from self_directing_mcp.daemon import core
        self.runtime_area = runtime
        config = config or core.DaemonConfig()
        if config.max_rss_mb == core.DaemonConfig().max_rss_mb:
            # The process under test is the whole pytest process: under a sanitizer (or after a long run) its real RSS
            # is far above the production cap, which would drain the daemon after its first request. Only a test that
            # sets a cap itself (with a faked reader) exercises that rule.
            config = dataclasses.replace(config, max_rss_mb=0)
        self.config = config
        self.kwargs = kwargs
        self.runtime = None
        self.lock = None
        self.daemon = None
        self.thread: threading.Thread | None = None
        self.failure: BaseException | None = None

    def start(self) -> "InProcessDaemon":
        from self_directing_mcp.daemon import core
        self.runtime = rundir.open_runtime_dir(self.runtime_area.environ())
        self.lock = core.acquire_instance_lock(self.runtime, 1.0)
        assert self.lock is not None, "another daemon holds the lock in this test's runtime directory"
        self.daemon = core.Daemon(self.runtime, self.config, **self.kwargs)

        def run():
            try:
                asyncio.run(self.daemon.serve())
            except BaseException as exc:            # noqa: BLE001 - reported by stop()
                self.failure = exc
            finally:
                if self.lock is not None:
                    os.close(self.lock)
                    self.lock = None

        self.thread = threading.Thread(target=run, name="test-daemon", daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.failure is not None:
                raise self.failure
            if self.runtime.exists(rundir.SOCKET_NAME) and self.daemon._server is not None:
                return self
            time.sleep(0.01)
        raise AssertionError("the in-process daemon did not start listening")

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def wait_exit(self, timeout: float = 10.0) -> bool:
        self.thread.join(timeout)
        return not self.thread.is_alive()

    def stop(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            loop = self.daemon._loop
            if loop is not None and not loop.is_closed():
                loop.call_soon_threadsafe(self.daemon.begin_drain, "test")
            self.thread.join(15)
        assert self.thread is None or not self.thread.is_alive(), "the in-process daemon did not stop"
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None
        if self.failure is not None and not isinstance(self.failure, asyncio.CancelledError):
            raise self.failure

    # -- raw protocol access ---------------------------------------------------------------------------------
    def connect(self, *, hello: bool = True, version: str | None = None, proto: int | None = None):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(10)
        runtime = rundir.open_runtime_dir(self.runtime_area.environ(), create=False)
        try:
            sock.connect(runtime.address())
        finally:
            runtime.close()
        if hello:
            from self_directing_mcp import __version__
            P.write_frame(sock, P.hello_request("test", version or __version__, proto=proto or P.PROTO))
            reply = P.read_frame(sock)
            return sock, reply
        return sock, None


def raw_runtime(runtime: Runtime):
    return rundir.open_runtime_dir(runtime.environ(), create=False)


# --- synthetic worlds -----------------------------------------------------------------------------------------

class World:
    """Synthetic Claude Code and AGY sessions, one contracts file and one config; recreated identically on demand."""

    def __init__(self, root: Path, runtime: Runtime, *, claude_sids=(SID,), agy_sids=(AGY_SID,),
                 contracts=None, size: int = 60_000) -> None:
        self.root, self.runtime = root, runtime
        self.projects, self.agy_app, self.index = root / "projects", root / "agy-app", root / "index"
        self.config = root / "config.toml"
        self.contracts = list(CONTRACTS if contracts is None else contracts)
        self.size = size
        b = bench()
        self.claude = {sid: b.Transcript("claude", b.claude_path(self.projects, sid), sid) for sid in claude_sids}
        self.agy = {sid: b.Transcript("agy", b.agy_path(self.agy_app, sid), sid) for sid in agy_sids}
        for transcript in (*self.claude.values(), *self.agy.values()):
            transcript.create(size)
        self.config.write_text("[paths]\n" + "\n".join([
            f"index_dir = {json.dumps(str(self.index))}", f"claude_projects_dir = {json.dumps(str(self.projects))}",
            f"agy_app_data_dirs = [{json.dumps(str(self.agy_app))}]"]) + "\n", encoding="utf-8")
        self.reset_index()

    def reset_index(self, contracts=None) -> None:
        if contracts is not None:
            self.contracts = list(contracts)
        shutil.rmtree(self.index, ignore_errors=True)
        self.index.mkdir(parents=True)
        (self.index / "contracts.json").write_text(json.dumps({"contracts": self.contracts}), encoding="utf-8")

    def new_index(self, name: str, contracts=None) -> Path:
        """A fresh, separate index directory holding the same (or the given) contracts."""
        path = self.root / f"index-{name}"
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True)
        document = {"contracts": list(self.contracts if contracts is None else contracts)}
        (path / "contracts.json").write_text(json.dumps(document), encoding="utf-8")
        return path

    def env(self, flag: str | None, **extra) -> dict:
        env = {k: v for k, v in os.environ.items() if not k.startswith(ENV_PREFIXES)}
        env.update(self.runtime.environ())
        env.update(SELF_DIRECT_CONFIG_FILE=str(self.config), SELF_DIRECT_LOCAL_ONLY="true",
                   SELF_DIRECT_INTEGRITY_BACKGROUND_VERIFY="false")
        if flag is not None:
            env["SELF_DIRECT_DAEMON"] = flag
        env.update(extra)
        return env

    def payload(self, host: str, event: str, command: str | None = None, *, sid: str | None = None,
                step_offset: int = 0) -> dict:
        """``step_offset`` gives parallel AGY calls of one conversation distinct step indexes, as a host does."""
        b = bench()
        if host == "claude":
            sid = sid or SID
            transcript = self.claude[sid]
            body = b.claude_payload(event, sid, transcript.path, transcript.last_step + 1)
            if event == "PreToolUse" and command is not None:
                body["tool_input"] = {"command": command}
            return body
        sid = sid or AGY_SID
        transcript = self.agy[sid]
        body = b.agy_payload(event, sid, transcript.path, transcript.last_step + 1 + step_offset)
        if event == "PreToolUse" and command is not None:
            body["toolCall"]["args"] = {"CommandLine": command}
        return body

    def argv(self, host: str, event: str, index: Path | None = None) -> list[str]:
        if host == "claude":
            return [PY, str(SCRIPTS / "claude_hook.py"), f"--event={event}", "--local-only",
                    f"--index-dir={index or self.index}"]
        return [PY, str(SCRIPTS / "agy_hook.py"), f"--event={event}"]

    def hook(self, host: str, event: str, command: str | None = None, *, flag: str | None = "on",
             sid: str | None = None, index: Path | None = None, timeout: float = 60.0, body: dict | None = None,
             step_offset: int = 0, **extra_env) -> "HookRun":
        body = body if body is not None else self.payload(host, event, command, sid=sid, step_offset=step_offset)
        extra_env = dict(extra_env)
        if index is not None:
            extra_env["SELF_DIRECT_INDEX_DIR"] = str(index)       # the environment beats the config file
        started = time.perf_counter()
        done = subprocess.run(self.argv(host, event, index), input=json.dumps(body), capture_output=True, text=True,
                              env=self.env(flag, **extra_env), timeout=timeout, cwd=self.root)
        return HookRun(done.stdout, done.stderr, (time.perf_counter() - started) * 1000, done.returncode)

    def start_daemon(self) -> subprocess.Popen:
        """Start a real daemon process (as a client would) and wait until it answers; it is stopped by the fixture."""
        return start_daemon_process(self.runtime)


class HookRun:
    def __init__(self, stdout: str, stderr: str, ms: float, code: int) -> None:
        self.stdout, self.stderr, self.ms, self.code = stdout, stderr, ms, code

    @property
    def json(self) -> dict:
        return json.loads(self.stdout)

    def __repr__(self) -> str:
        return f"HookRun(code={self.code}, ms={self.ms:.0f}, stdout={self.stdout[:200]!r}, stderr={self.stderr[-200:]!r})"


def daemon_log(runtime: Runtime) -> list[dict]:
    path = runtime.dir / rundir.LOG_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def start_daemon_process(runtime: Runtime, **extra_env) -> subprocess.Popen:
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT / "src"), **runtime.environ(), **extra_env}
    process = subprocess.Popen([PY, "-m", "self_directing_mcp.daemon"], env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                               cwd=str(runtime.root))
    runtime.extra_pids.add(process.pid)
    STARTED.add(process.pid)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"the daemon process exited with {process.returncode}")
        try:
            area = rundir.open_runtime_dir(runtime.environ(), create=False)
        except rundir.RuntimeDirError:
            time.sleep(0.02)
            continue
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(2)
            try:
                sock.connect(area.address())
                return process
            except OSError:
                time.sleep(0.02)
            finally:
                sock.close()
        finally:
            area.close()
    raise AssertionError("the daemon process did not start listening")


def daemon_status(runtime: Runtime) -> dict:
    from self_directing_mcp.daemon import core
    area = rundir.open_runtime_dir(runtime.environ(), create=False)
    try:
        return core.query(area, {"op": P.OP_STATUS}) or {}
    finally:
        area.close()


def make_stub(directory: Path, seconds: float = 3.0) -> Path:
    """An executable that stands in for the daemon's interpreter and never starts a daemon: it ignores its
    arguments and just stays alive for a while (so a test can observe the client not waiting for it)."""
    path = directory / "tsukkomi-test-stub"
    path.write_text(f"#!/bin/sh\nsleep {seconds}\n")
    path.chmod(0o755)
    return path
