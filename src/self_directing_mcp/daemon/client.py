"""Thin client of the daemon for command hooks. Standard library only: importing this module (or running a hook
through it) never loads the engine, pydantic, NumPy, NeoGraph or the MCP SDK.

The flag is ``SELF_DIRECT_DAEMON`` read from the process environment: ``off`` (default) leaves everything standalone.
On a platform without the daemon (anything but Linux) ``on`` behaves as ``off`` and says so once.

When the flag is on, the hook's whole answer comes from the daemon or from a visible fail-open. There is NO in-process
fallback on the hot path:

1. Connect to the daemon's socket in the verified runtime directory and send ``hello`` plus the hook request in one
   write; the daemon's ``same`` version relation means it judged the request.
2. Daemon not reachable: take the spawn lock (non-blocking), start a daemon detached (own session, stdio on
   ``/dev/null``) unless the guards forbid it, and wait for its socket for a bounded time (``AVAILABILITY_FAST_SEC``
   for ``PreToolUse`` / ``PostToolUse`` / ``UserPromptSubmit`` / ``PreInvocation``, ``AVAILABILITY_SLOW_SEC`` for
   ``Stop`` / ``SubagentStop`` / ``SessionStart``).
3. Still unavailable, a dead daemon mid-request, ``queue_full``, ``version_mismatch``: the host's fail-open response
   from ``claude_hook_state.fail_open`` / ``agy_hook_state.fail_open`` (never a deny; ``ask`` only when a matching
   contract is marked ``risk: high``; a visible warning with the reason; the failure is recorded in ``hook_health``).

Guards. More than ``CRASH_LOOP_SPAWNS`` spawns within ``CRASH_LOOP_WINDOW_SEC`` (recorded in the runtime directory)
stops spawning: the reason becomes ``daemon_unavailable:crash-loop`` and a warning is printed. A daemon started less
than ``STARTUP_GRACE_SEC`` ago and still alive is waited for, not started a second time.

Versions. A client whose proto or package version is NEWER than the daemon's asks it to drain and then starts the
successor (the answer of THIS hook is still bounded by the availability budget, and during a switchover other hooks
fail open). An OLDER or mixed client never replaces the daemon: its hooks fail open with ``version_mismatch``.
"""
from __future__ import annotations

import errno
import os
import socket
import sys
import time

from self_directing_mcp import __version__
from self_directing_mcp.daemon import protocol as P, rundir

ENV_FLAG = "SELF_DIRECT_DAEMON"
ENV_PYTHON = "SELF_DIRECT_DAEMON_PYTHON"          # interpreter that starts the daemon (default: the checkout's .venv)

FAST_EVENTS = ("PreToolUse", "PostToolUse", "PostToolUseFailure", "UserPromptSubmit", "PreInvocation")
SLOW_EVENTS = ("Stop", "SubagentStop", "SessionStart")
AVAILABILITY_FAST_SEC = 0.3
AVAILABILITY_SLOW_SEC = 3.0
CRASH_LOOP_SPAWNS = 3                  # more than this many spawns ...
CRASH_LOOP_WINDOW_SEC = 300.0          # ... within this window: do not spawn again
STARTUP_GRACE_SEC = 10.0
POLL_SEC = 0.01
CONNECT_TIMEOUT_SEC = 0.25
HELLO_TIMEOUT_SEC = 2.0
DEFAULT_REQUEST_TIMEOUT_SEC = 45.0     # = settings_core.DEFAULT_REQUEST_TIMEOUT_SEC (a test pins the equality)
RESPONSE_MARGIN_SEC = 5.0
KEEP_ENV = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LANGUAGE", "TMPDIR", "XDG_RUNTIME_DIR", "XDG_CACHE_HOME",
            "PYTHONPATH", "VIRTUAL_ENV", "PYTHONHASHSEED")
KEEP_ENV_PREFIXES = ("LC_", "SELF_DIRECT_DAEMON_")

R_UNAVAILABLE = "daemon_unavailable"
R_TIMEOUT = "daemon_timeout"
R_CRASH_LOOP = "daemon_unavailable:crash-loop"
R_RUNTIME_DIR = "daemon_unavailable:runtime_dir"


# --- the flag -----------------------------------------------------------------------------------------------------

def supported() -> bool:
    return sys.platform.startswith("linux")


def flag_state(environ=None) -> str:
    """``on`` / ``off`` / ``invalid``. Case-insensitive names, exact values: the same rule as ``settings_core``."""
    environ = os.environ if environ is None else environ
    for name, value in environ.items():
        if name.lower() == ENV_FLAG.lower():
            return value if value in ("on", "off") else "invalid"
    return "off"


def notice_once(key: str, message: str, environ=None) -> None:
    """Print ``message`` to stderr the first time only; later processes find the marker and stay quiet."""
    environ = os.environ if environ is None else environ
    home = environ.get("HOME") or os.path.expanduser("~")
    marker = os.path.join(home, ".cache", "tsukkomi", "notices", key)
    try:
        if os.path.exists(marker):
            return
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        with open(marker, "w", encoding="ascii") as stream:
            stream.write("1\n")
    except OSError:
        pass                                  # cannot remember: say it again rather than never
    print(message, file=sys.stderr)


def use_daemon(environ=None) -> bool:
    """The entry-point decision: True only for ``on`` on a supported platform."""
    state = flag_state(environ)
    if state != "on":
        return False
    if not supported():
        notice_once("daemon-unsupported", "tsukkomi: SELF_DIRECT_DAEMON=on is ignored on this platform (the resident "
                    "daemon is Linux only in this release); running standalone.", environ)
        return False
    return True


def availability_budget(event) -> float:
    return AVAILABILITY_SLOW_SEC if event in SLOW_EVENTS else AVAILABILITY_FAST_SEC


def response_timeout(environ=None) -> float:
    """The existing request budget: ``SELF_DIRECT_REQUEST_TIMEOUT_SEC`` (default 45 s) plus a margin."""
    environ = os.environ if environ is None else environ
    for name, value in environ.items():
        if name.upper() == "SELF_DIRECT_REQUEST_TIMEOUT_SEC":
            try:
                seconds = float(value.strip())
                if 0 < seconds < float("inf"):
                    return seconds + RESPONSE_MARGIN_SEC
            except ValueError:
                pass
    return DEFAULT_REQUEST_TIMEOUT_SEC + RESPONSE_MARGIN_SEC


# --- spawn --------------------------------------------------------------------------------------------------------

def _pid_alive(pid) -> bool:
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as stream:
            state = stream.read().rsplit(b")", 1)[1].split()[0]
        return state not in (b"Z", b"X")
    except (OSError, ValueError, IndexError):
        return False


def daemon_environment(environ) -> dict:
    """The daemon's own environment: no settings, no credentials; each request brings its own."""
    kept = {name: value for name, value in environ.items()
            if name in KEEP_ENV or name.startswith(KEEP_ENV_PREFIXES)}
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.isdir(os.path.join(src, "self_directing_mcp")):
        kept["PYTHONPATH"] = src + (os.pathsep + kept["PYTHONPATH"] if kept.get("PYTHONPATH") else "")
    return kept


def spawn_command(environ, python=None) -> list[str]:
    return [environ.get(ENV_PYTHON) or python or sys.executable, "-m", "self_directing_mcp.daemon"]


def _start_process(environ, python=None):
    import subprocess
    return subprocess.Popen(spawn_command(environ, python), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, env=daemon_environment(environ), cwd="/",
                            start_new_session=True, close_fds=True)


def _read_spawns(runtime) -> list[dict]:
    import json
    try:
        fd = runtime.open_file(rundir.SPAWNS_NAME, os.O_RDONLY)
    except OSError:
        return []
    try:
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            records = json.load(stream)
        return [r for r in records if isinstance(r, dict) and isinstance(r.get("t"), (int, float))]
    except (OSError, ValueError):
        return []


def _write_spawns(runtime, records) -> None:
    import json
    fd = runtime.open_file(rundir.SPAWNS_NAME, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(records, stream)


def start_daemon(runtime, environ, python=None, *, reachable=lambda: False):
    """Start a daemon unless something forbids it. ``(outcome, process)``; outcome is ``spawned``, ``busy`` (another
    client is spawning), ``starting`` (a recent daemon is still coming up), ``up``, ``crash_loop`` or ``failed``."""
    import fcntl
    lock = runtime.open_file(rundir.SPAWN_LOCK_NAME, os.O_RDWR | os.O_CREAT)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "busy", None
        if reachable():
            return "up", None
        now = time.time()
        recent = [r for r in _read_spawns(runtime) if 0 <= now - r["t"] < CRASH_LOOP_WINDOW_SEC]
        if recent and now - recent[-1]["t"] < STARTUP_GRACE_SEC and _pid_alive(recent[-1].get("pid")):
            return "starting", None
        if len(recent) > CRASH_LOOP_SPAWNS:
            return "crash_loop", None
        try:
            process = _start_process(environ, python)
            recent.append({"t": now, "pid": process.pid})
            outcome = "spawned"
        except OSError:
            process, outcome = None, "failed"
            recent.append({"t": now, "pid": 0})
        try:
            _write_spawns(runtime, recent)
        except OSError:
            pass
        return outcome, process
    finally:
        os.close(lock)


# --- one exchange --------------------------------------------------------------------------------------------------

class Unreachable(Exception):
    """No daemon is listening (or it is going away): the spawn path may help."""


class Failed(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _connect(runtime) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(CONNECT_TIMEOUT_SEC)
    try:
        sock.connect(runtime.address())
    except OSError as exc:
        sock.close()
        if exc.errno in (errno.ENOENT, errno.ECONNREFUSED, errno.ENOTDIR, errno.EAGAIN, errno.ECONNRESET):
            raise Unreachable() from None
        raise Failed(R_UNAVAILABLE) from None
    return sock


def _exchange(runtime, request: dict, timeout: float) -> dict:
    """Hello + request on one connection. Returns the daemon's result frame (with ``relation`` merged in)."""
    sock = _connect(runtime)
    try:
        try:
            sock.sendall(P.encode_frame(P.hello_request("hook", __version__)) + P.encode_frame(request))
            sock.settimeout(HELLO_TIMEOUT_SEC)
            hello = P.read_frame(sock)
            sock.settimeout(timeout)
            result = P.read_frame(sock)
        except P.ProtocolError as exc:
            raise Failed(R_UNAVAILABLE) from None
        except (socket.timeout, TimeoutError):
            raise Failed(R_TIMEOUT) from None
        except (OSError, EOFError):
            raise Failed(R_UNAVAILABLE) from None
        result = dict(result, relation=hello.get("relation"), draining=bool(hello.get("draining")))
        if result.get("ok") is False and result.get("reason") == P.REASON_VERSION_MISMATCH and \
                hello.get("relation") == "newer":
            try:                                      # a newer client asks the older daemon to leave
                sock.settimeout(HELLO_TIMEOUT_SEC)
                P.write_frame(sock, {"op": P.OP_DRAIN})
                P.read_frame(sock)
            except (OSError, EOFError, P.ProtocolError):
                pass
            result["drained"] = True
        return result
    finally:
        sock.close()


def _request_frame(host, event, payload, argv, environ) -> dict:
    return {"op": P.OP_HOOK, "host": host, "event": event, "payload": payload,
            "env": P.forwarded_environment(environ), "argv": P.forwarded_argv(argv)}


def _fail_open(host, event, payload, reason):
    kind_unavailable = reason.startswith(("daemon_unavailable", "daemon_timeout"))
    if host == "claude":
        from self_directing_mcp import claude_hook_state as state
    else:
        from self_directing_mcp import agy_hook_state as state
    return state.fail_open(event, payload, reason, state.AUDIT_UNAVAILABLE if kind_unavailable else state.CHECK_FAILED)


def serve_hook(host: str, event, payload, argv, *, environ=None, python=None) -> tuple[dict, bool]:
    """``(response, served_by_daemon)``. Always returns a response and never imports the engine. A daemon response
    is already masked; a fail-open response (``False``) carries only class names and contract ids."""
    environ = os.environ if environ is None else environ
    reason = R_UNAVAILABLE
    try:
        response, reason = _serve(host, event, payload, argv, environ, python)
    except Exception:
        response = None
    if response is not None:
        return response, True
    return _fail_open(host, event, payload, reason), False


def _serve(host, event, payload, argv, environ, python):
    """``(response, None)`` or ``(None, reason)``."""
    try:
        request = _request_frame(host, event, payload, argv, environ)
        P.encode_frame(request)                      # size check before anything is started
    except P.ProtocolError:
        return None, "frame_too_large"
    try:
        runtime = rundir.open_runtime_dir(environ)
    except rundir.RuntimeDirError as exc:
        print(f"tsukkomi: {exc}", file=sys.stderr)
        return None, R_RUNTIME_DIR
    with runtime:
        timeout = response_timeout(environ)
        deadline = time.monotonic() + availability_budget(event)
        process, spawned = None, False
        while True:
            try:
                result = _exchange(runtime, request, timeout)
            except Unreachable:
                result = None
            except Failed as exc:
                return None, exc.reason
            if result is not None:
                if result.get("ok"):
                    return result["response"], None
                reason = result.get("reason")
                if reason == P.REASON_VERSION_MISMATCH and not result.get("drained"):
                    return None, P.REASON_VERSION_MISMATCH      # older / mixed: never replaces the daemon
                if reason not in (P.REASON_DRAINING, P.REASON_VERSION_MISMATCH):
                    return None, reason or R_UNAVAILABLE         # queue_full, bad_request, internal_error ...
                # draining, or just asked to drain: a successor must be started and awaited
            if time.monotonic() >= deadline:
                return None, R_UNAVAILABLE
            if process is not None and process.poll() is not None:
                return None, R_UNAVAILABLE                       # the daemon we started died at once
            if not spawned:
                outcome, process = start_daemon(runtime, environ, python, reachable=lambda: _reachable(runtime))
                if outcome == "crash_loop":
                    print("tsukkomi: daemon crash-loop guard: more than %d daemon starts within %d s; not starting "
                          "another. Hooks fail open until it is repaired (see the daemon log in the runtime directory)."
                          % (CRASH_LOOP_SPAWNS, int(CRASH_LOOP_WINDOW_SEC)), file=sys.stderr)
                    return None, R_CRASH_LOOP
                spawned = outcome in ("spawned", "failed")
                if outcome == "failed":
                    return None, R_UNAVAILABLE
            time.sleep(POLL_SEC)


def _reachable(runtime) -> bool:
    try:
        sock = _connect(runtime)
    except (Unreachable, Failed):
        return False
    sock.close()
    return True
