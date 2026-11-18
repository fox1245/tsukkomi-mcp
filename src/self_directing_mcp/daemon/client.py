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
   from ``claude_hook_state.fail_open`` / ``agy_hook_state.fail_open`` (no deny except for Claude Code, which refuses a call
   that matches the trigger of a contract marked ``risk: high`` because the audit did not run, and AGY asks; a visible
   warning with the reason; the failure is recorded in ``hook_health``).

Guards. Spawns are recorded in the runtime directory (``spawns.json``), and a daemon announces its own orderly exit
(memory cap, idle exit, version switch, stop request, signal) in ``exits.json`` before it stops accepting. A spawn
whose daemon announced no orderly exit counts as a crash: more than ``CRASH_LOOP_SPAWNS`` of them within
``CRASH_LOOP_WINDOW_SEC`` stops spawning with ``daemon_unavailable:crash-loop``. Planned restarts have their own bound:
more than ``SELF_DIRECT_DAEMON_RESTART_LIMIT`` (default ``DEFAULT_RESTART_LIMIT``) within the same window stops spawning
with ``daemon_unavailable:restart-loop`` (a memory cap below the daemon's working set would otherwise restart it after
every request). Either prints a warning. A daemon started less than ``STARTUP_GRACE_SEC`` ago and still alive is waited
for, not started a second time.

Versions. A client whose proto or package version is NEWER than the daemon's asks it to drain and then starts the
successor (the answer of THIS hook is still bounded by the availability budget, and during a switchover other hooks
fail open). An OLDER or mixed client never replaces the daemon: its hooks fail open with ``version_mismatch``.
"""
from __future__ import annotations

import errno
import os
import sys
import time

import _socket as socket       # the C module: the socket module's Python layer costs ~2.5 ms of import per hook process

from self_directing_mcp import __version__
from self_directing_mcp.daemon import protocol as P, rundir

_monotonic = time.monotonic            # the clock and the pause of the wait in ``serve_hook`` (tests replace them)
_sleep = time.sleep

ENV_FLAG = "SELF_DIRECT_DAEMON"
ENV_PYTHON = "SELF_DIRECT_DAEMON_PYTHON"          # interpreter that starts the daemon (default: the checkout's .venv)

FAST_EVENTS = ("PreToolUse", "PostToolUse", "PostToolUseFailure", "UserPromptSubmit", "PreInvocation")
SLOW_EVENTS = ("Stop", "SubagentStop", "SessionStart")
AVAILABILITY_FAST_SEC = 0.3
AVAILABILITY_SLOW_SEC = 3.0
CRASH_LOOP_SPAWNS = 3                  # more than this many spawns without an announced orderly exit ...
CRASH_LOOP_WINDOW_SEC = 300.0          # ... within this window: do not spawn again
ENV_RESTART_LIMIT = "SELF_DIRECT_DAEMON_RESTART_LIMIT"
DEFAULT_RESTART_LIMIT = 30             # more than this many planned restarts within the window: do not spawn again
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
R_RESTART_LOOP = "daemon_unavailable:restart-loop"
R_SETTING = "daemon_unavailable:setting"
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
    if os.path.isdir(os.path.join(src, "self_directing_mcp")) and src not in kept.get("PYTHONPATH", "").split(os.pathsep):
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


def restart_limit(environ=None) -> int:
    """``SELF_DIRECT_DAEMON_RESTART_LIMIT``: planned restarts allowed within ``CRASH_LOOP_WINDOW_SEC`` (a positive
    integer, default ``DEFAULT_RESTART_LIMIT``). A malformed value is an error (``ValueError``), never a silent default."""
    environ = os.environ if environ is None else environ
    text = environ.get(ENV_RESTART_LIMIT)
    if text is None:
        return DEFAULT_RESTART_LIMIT
    try:
        value = int(text.strip())
    except ValueError:
        raise ValueError(f"invalid value for {ENV_RESTART_LIMIT}") from None
    if value <= 0:
        raise ValueError(f"invalid value for {ENV_RESTART_LIMIT}")
    return value


def _read_exits(runtime) -> list[dict]:
    import json
    try:
        fd = runtime.open_file(rundir.EXITS_NAME, os.O_RDONLY)
    except OSError:
        return []
    try:
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            records = json.load(stream)
    except (OSError, ValueError):
        return []
    if not isinstance(records, list):
        return []
    return [r for r in records if isinstance(r, dict) and isinstance(r.get("t"), (int, float))
            and isinstance(r.get("pid"), int) and not isinstance(r.get("pid"), bool)]


def record_planned_exit(runtime, pid: int, reason: str, now: float | None = None) -> None:
    """The daemon's announcement that it is leaving in order (``exits.json``). Called by the daemon while it holds the
    instance lock, before it removes its socket, so a client that finds no socket already sees it. Only the lock holder
    writes the file; it is replaced whole, so a reader never sees a partial list. Raises ``OSError``."""
    import json
    now = time.time() if now is None else now
    records = [r for r in _read_exits(runtime) if 0 <= now - r["t"] < CRASH_LOOP_WINDOW_SEC]
    records.append({"t": now, "pid": int(pid), "reason": reason})
    scratch = f"{rundir.EXITS_NAME}.{int(pid)}.tmp"
    fd = runtime.open_file(scratch, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(records, stream)
    os.replace(scratch, rundir.EXITS_NAME, src_dir_fd=runtime.fd, dst_dir_fd=runtime.fd)


def guard_outcome(recent: list[dict], exits: list[dict], limit: int) -> str | None:
    """``crash_loop``, ``restart_loop`` or None for the spawns of the window. A spawn whose daemon announced an orderly
    exit after it was started is a planned restart; any other spawn (a daemon that died without announcing, or one that
    is still around) counts as before."""
    announced: dict[int, list[float]] = {}
    for record in exits:
        announced.setdefault(record["pid"], []).append(record["t"])
    planned = sum(1 for r in recent if any(t >= r["t"] for t in announced.get(r.get("pid"), ())))
    if len(recent) - planned > CRASH_LOOP_SPAWNS:
        return "crash_loop"
    if planned > limit:
        return "restart_loop"
    return None


def start_daemon(runtime, environ, python=None, *, reachable=lambda: False):
    """Start a daemon unless something forbids it. ``(outcome, process)``; outcome is ``spawned``, ``busy`` (another
    client is spawning), ``starting`` (a recent daemon is still coming up), ``up``, ``crash_loop``, ``restart_loop``,
    ``bad_setting`` (``SELF_DIRECT_DAEMON_RESTART_LIMIT`` is malformed) or ``failed``."""
    import fcntl
    try:
        limit = restart_limit(environ)
    except ValueError:
        return "bad_setting", None
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
        refused = guard_outcome(recent, _read_exits(runtime), limit)
        if refused is not None:
            return refused, None
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


TRANSIENT_REASONS = (R_UNAVAILABLE, R_TIMEOUT)     # a daemon that is not there (yet) or did not answer in time: a retry may clear it


def _fail_open(host, event, payload, reason):
    kind_unavailable = reason.startswith(("daemon_unavailable", "daemon_timeout"))
    if host == "claude":
        from self_directing_mcp import claude_hook_state as state
        return state.fail_open(event, payload, reason, state.AUDIT_UNAVAILABLE if kind_unavailable else state.CHECK_FAILED,
                               transient=reason in TRANSIENT_REASONS)
    from self_directing_mcp import agy_hook_state as state
    return state.fail_open(event, payload, reason, state.AUDIT_UNAVAILABLE if kind_unavailable else state.CHECK_FAILED)


def _wait_again(host, event, payload, reason) -> bool:
    """Whether a call that found no daemon is worth waiting for it once more: only a Claude Code PreToolUse that a
    ``risk: high`` contract gates (every other hook costs nothing when it fails open, and waits no longer than before) and
    only for ``daemon_unavailable`` itself. ``daemon_timeout`` has used the whole response budget already, and the guards
    (``crash-loop``, ``restart-loop``, a bad setting) are decisions not to start a daemon."""
    if host != "claude" or event != "PreToolUse" or reason != R_UNAVAILABLE:
        return False
    from self_directing_mcp import claude_hook_state as state
    return state.gating_rule(event, payload) is not None


def serve_hook(host: str, event, payload, argv, *, environ=None, python=None) -> tuple[dict, bool]:
    """``(response, served_by_daemon)``. Always returns a response and never imports the engine. A daemon response
    is already masked; a fail-open response (``False``) carries only class names and contract ids.

    A gated call (``_wait_again``) that found no daemon within the usual availability budget waits for the daemon that was
    started, without starting another, until ``AVAILABILITY_SLOW_SEC`` after the hook began: the longest wait the client
    already allows any event for a daemon to come up, and far inside the response budget."""
    environ = os.environ if environ is None else environ
    started = _monotonic()
    reason = R_UNAVAILABLE
    try:
        response, reason = _serve(host, event, payload, argv, environ, python)
        while response is None and _wait_again(host, event, payload, reason):
            remaining = started + AVAILABILITY_SLOW_SEC - _monotonic()
            if remaining <= 0:
                break
            attempt = _monotonic()
            response, reason = _serve(host, event, payload, argv, environ, python, spawn=False, wait=remaining)
            if response is None and _monotonic() - attempt < AVAILABILITY_FAST_SEC:
                # An answer refused at once is not asked for again at once (nor past the end of the wait).
                _sleep(max(0.0, min(AVAILABILITY_FAST_SEC, started + AVAILABILITY_SLOW_SEC - _monotonic())))
    except Exception:
        response = None
    if response is not None:
        return response, True
    return _fail_open(host, event, payload, reason), False


def _serve(host, event, payload, argv, environ, python, *, spawn=True, wait=None):
    """``(response, None)`` or ``(None, reason)``. ``spawn=False`` only waits (up to ``wait`` seconds, instead of the
    event's availability budget) for a daemon that is already coming up."""
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
        deadline = time.monotonic() + (availability_budget(event) if wait is None else wait)
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
            if spawn and not spawned:
                outcome, process = start_daemon(runtime, environ, python, reachable=lambda: _reachable(runtime))
                if outcome == "crash_loop":
                    print("tsukkomi: daemon crash-loop guard: more than %d daemon starts within %d s; not starting "
                          "another. Hooks fail open until it is repaired (see the daemon log in the runtime directory)."
                          % (CRASH_LOOP_SPAWNS, int(CRASH_LOOP_WINDOW_SEC)), file=sys.stderr)
                    return None, R_CRASH_LOOP
                if outcome == "restart_loop":
                    print("tsukkomi: daemon restart-loop guard: more than %d planned daemon restarts within %d s (the "
                          "daemon log in the runtime directory names the reason; for \"rss\" raise "
                          "SELF_DIRECT_DAEMON_MAX_RSS_MB); not starting another. Hooks fail open until the window passes."
                          % (restart_limit(environ), int(CRASH_LOOP_WINDOW_SEC)), file=sys.stderr)
                    return None, R_RESTART_LOOP
                if outcome == "bad_setting":
                    print(f"tsukkomi: invalid value for {ENV_RESTART_LIMIT} (a positive integer); not starting a daemon.",
                          file=sys.stderr)
                    return None, R_SETTING
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
