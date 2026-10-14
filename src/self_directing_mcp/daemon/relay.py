"""stdio-to-daemon relay: what ``self-directing-mcp`` (and ``--omp-bridge``) becomes when ``SELF_DIRECT_DAEMON=on``.

Standard library only (a relay is one of N long-lived processes, so it must not load pydantic, the MCP SDK, NumPy or
the engine). It connects to the daemon (starting one under the same guards as a hook client, but with a longer
budget: a session is not a hot path), sends ``hello`` and one ``mcp`` frame carrying the client's relevant
environment (never a credential value), its working directory (for ``.env``) and the mode, and then copies bytes in
both directions: the daemon runs the MCP protocol on the connection with the same FastMCP tool functions a standalone
server uses.

Failures are never silent and never fall back to a standalone server:

* a configuration the daemon rejects (``--omp-bridge`` without the key file or native library, ...), a daemon that
  cannot be started, a version mismatch: the first JSON-RPC request on stdin (``initialize``) is answered with a
  JSON-RPC error carrying the reason, the reason goes to stderr, and the process exits non-zero (2 for configuration,
  1 otherwise);
* the daemon going away mid-session closes stdout; the host sees the server exit and restarts it as it would a crashed
  standalone server.
"""
from __future__ import annotations

import json
import os
import select
import sys
import threading
import time

from self_directing_mcp import __version__
from self_directing_mcp.daemon import client, protocol as P, rundir

AVAILABILITY_SEC = 15.0
CHUNK = 64 * 1024
EXIT_UNAVAILABLE = 1
EXIT_CONFIGURATION = 2
ERROR_CODE = -32603


class RelayError(Exception):
    def __init__(self, message: str, code: int = EXIT_UNAVAILABLE) -> None:
        super().__init__(message)
        self.message, self.code = message, code


def open_session(environ, mode: str, cwd, *, python=None, budget: float = AVAILABILITY_SEC):
    """A connected socket on which the MCP session is already established. Raises ``RelayError``."""
    try:
        runtime = rundir.open_runtime_dir(environ)
    except rundir.RuntimeDirError as exc:
        raise RelayError(f"{exc}") from None
    request = {"op": P.OP_MCP, "env": P.forwarded_environment(environ), "cwd": cwd, "mode": mode}
    with runtime:
        deadline = time.monotonic() + budget
        spawned = False
        while True:
            sock = None
            try:
                sock = client._connect(runtime)
            except client.Unreachable:
                pass
            except client.Failed:
                raise RelayError("cannot reach the daemon") from None
            if sock is not None:
                outcome = _handshake(sock, request)
                if outcome is not None:
                    return outcome
            if time.monotonic() >= deadline:
                raise RelayError("the daemon is not available (daemon_unavailable)")
            if not spawned:
                state, process = client.start_daemon(runtime, environ, python, reachable=lambda: client._reachable(runtime))
                if state == "crash_loop":
                    raise RelayError("daemon crash-loop guard: more than %d daemon starts within %d s; not starting "
                                     "another (daemon_unavailable:crash-loop)" % (client.CRASH_LOOP_SPAWNS,
                                                                                    int(client.CRASH_LOOP_WINDOW_SEC)))
                if state == "failed":
                    raise RelayError("the daemon could not be started (daemon_unavailable)")
                spawned = state in ("spawned",)
            time.sleep(client.POLL_SEC * 5)


def _handshake(sock, request):
    """The socket when the session is established, ``None`` when the daemon is going away or being replaced."""
    try:
        sock.settimeout(client.HELLO_TIMEOUT_SEC)
        P.write_frame(sock, P.hello_request("relay", __version__))
        hello = P.read_frame(sock)
        relation = hello.get("relation")
        if relation == "newer":                               # a newer relay asks the older daemon to leave
            P.write_frame(sock, {"op": P.OP_DRAIN})
            P.read_frame(sock)
            sock.close()
            return None
        if relation != "same":
            sock.close()
            raise RelayError("the running daemon is newer than this client (version_mismatch); "
                             "restart this server so that it picks up the new version")
        if hello.get("draining"):
            sock.close()
            return None
        P.write_frame(sock, request)
        sock.settimeout(60.0)                                 # settings, imports and engine setup happen here
        answer = P.read_frame(sock)
    except RelayError:
        raise
    except (OSError, EOFError, P.ProtocolError):
        sock.close()
        return None
    if answer.get("ok") is True:
        sock.settimeout(None)
        return sock
    sock.close()
    reason = answer.get("reason")
    if reason == P.REASON_DRAINING:
        return None
    message = answer.get("message") if isinstance(answer.get("message"), str) else "the daemon refused the session"
    raise RelayError(f"{message} ({reason})", EXIT_CONFIGURATION if reason == "configuration" else EXIT_UNAVAILABLE)


def _answer_initialize(message: str, wait: float = 5.0) -> None:
    """Answer the host's first request with a JSON-RPC error so that the failure surfaces as an MCP error."""
    buffer = b""
    deadline = time.monotonic() + wait
    try:
        while b"\n" not in buffer and time.monotonic() < deadline:
            if select.select([0], [], [], max(0.0, deadline - time.monotonic()))[0]:
                data = os.read(0, CHUNK)
                if not data:
                    return
                buffer += data
        request = json.loads(buffer.split(b"\n", 1)[0])
        if isinstance(request, dict) and request.get("id") is not None:
            reply = {"jsonrpc": "2.0", "id": request["id"], "error": {"code": ERROR_CODE, "message": message}}
            os.write(1, (json.dumps(reply) + "\n").encode())
    except (OSError, ValueError):
        pass


def pipe(sock) -> int:
    """Copy stdin to the daemon and the daemon to stdout until either side ends."""
    state = {"stdin_closed": False}

    def down() -> None:
        code = 0
        try:
            while True:
                data = sock.recv(CHUNK)
                if not data:
                    break
                view = memoryview(data)
                while view:
                    view = view[os.write(1, view):]
        except OSError:
            code = EXIT_UNAVAILABLE
        else:
            code = 0 if state["stdin_closed"] else EXIT_UNAVAILABLE     # the daemon left while the host still talks
        os._exit(code)

    threading.Thread(target=down, name="relay-down", daemon=True).start()
    try:
        while True:
            data = os.read(0, CHUNK)
            if not data:
                break
            sock.sendall(data)
    except OSError:
        return EXIT_UNAVAILABLE
    state["stdin_closed"] = True
    try:
        sock.shutdown(1)                                   # SHUT_WR: the daemon ends the session, `down` then exits
    except OSError:
        return 0
    time.sleep(30)                                         # `down` ends the process; this bounds a daemon that ignores EOF
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = "default"
    for arg in argv:
        if arg == "--omp-bridge":
            mode = "omp-bridge"
        else:
            print("usage: self-directing-mcp [--omp-bridge]", file=sys.stderr)
            return 2
    try:
        cwd = os.getcwd()
    except OSError:
        cwd = None
    try:
        sock = open_session(os.environ, mode, cwd)
    except RelayError as exc:
        print(f"tsukkomi-mcp: {exc.message}", file=sys.stderr)
        _answer_initialize(exc.message)
        return exc.code
    return pipe(sock)
