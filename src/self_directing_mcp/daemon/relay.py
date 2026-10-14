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
import re
import select
import sys
import threading
import time

from self_directing_mcp import __version__
from self_directing_mcp.daemon import client, protocol as P, rundir

AVAILABILITY_SEC = 15.0
CHUNK = 64 * 1024
MAX_RECOVERIES = 5
RECOVERY_WINDOW_SEC = 60.0
RESTART_ERROR = "tsukkomi daemon restarted before this request finished; retry it"
_RESPONSE_HEAD = re.compile(r'\{"jsonrpc":\s*"2\.0",\s*"id":\s*(-?\d+|"(?:[^"\\]|\\.)*"),\s*"(?:result|error)"')
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


def _line_key(value):
    return value if isinstance(value, (int, str)) else repr(value)


def _error_line(key, message: str) -> bytes:
    return (json.dumps({"jsonrpc": "2.0", "id": key, "error": {"code": ERROR_CODE, "message": message}}) + "\n").encode()


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _lines(buffer: bytearray, chunk: bytes):
    """Complete lines of a byte stream (``buffer`` keeps the unfinished tail)."""
    buffer += chunk
    while True:
        end = buffer.find(b"\n")
        if end < 0:
            return
        line = bytes(buffer[:end + 1])
        del buffer[:end + 1]
        yield line


class Relay:
    """The two byte pipes plus the bookkeeping that lets a session survive a daemon restart.

    The relay remembers the host's ``initialize`` request and ``initialized`` notification and which requests are
    still unanswered. When the daemon goes away (memory cap, version switch, crash) while the host is still talking,
    every unanswered request gets a JSON-RPC error (the host sees an ordinary failed call, not a dead server), the
    relay reconnects (starting a successor under the usual guards), replays the two handshake messages (the daemon's
    answer to the replayed ``initialize`` is consumed here: the host already has its own) and the session goes on.
    Several recoveries in a short time end the process instead (``MAX_RECOVERIES`` in ``RECOVERY_WINDOW_SEC``)."""

    def __init__(self, sock, environ, mode: str, cwd, python=None) -> None:
        self.sock, self.environ, self.mode, self.cwd, self.python = sock, environ, mode, cwd, python
        self.cond = threading.Condition()
        self.initialize: bytes | None = None
        self.initialized: bytes | None = None
        self.pending: dict = {}
        self.stdin_closed = False
        self.recoveries: list[float] = []

    # -- host -> daemon --------------------------------------------------------------------------------------
    def forward(self, line: bytes) -> None:
        try:
            message = json.loads(line)
        except ValueError:
            message = None
        request = None
        if isinstance(message, dict):
            method = message.get("method")
            if method == "initialize" and message.get("id") is not None and self.initialize is None:
                self.initialize = line
            elif method == "notifications/initialized":
                self.initialized = line
            if method and message.get("id") is not None:
                request = message["id"]
        with self.cond:
            while self.sock is None:                       # a recovery is running: hold the line until it is over
                self.cond.wait(1.0)
            sock = self.sock
            # Registered while a connection exists, so a recovery's sweep of the unanswered requests can neither
            # miss this request nor answer it and then see it sent again.
            if request is not None:
                self.pending[_line_key(request)] = request
            elif isinstance(message, dict) and message.get("method") == "notifications/cancelled":
                self.pending.pop(_line_key((message.get("params") or {}).get("requestId")), None)
        try:
            sock.sendall(line)
        except OSError:
            # The daemon went away under this line. The reader recovers; a request is answered by that recovery's
            # sweep of the unanswered requests, a notification is simply lost.
            pass

    def run_up(self) -> int:
        buffer = bytearray()
        try:
            while True:
                chunk = os.read(0, CHUNK)
                if not chunk:
                    break
                for line in _lines(buffer, chunk):
                    self.forward(line)
        except OSError:
            return EXIT_UNAVAILABLE
        self.stdin_closed = True
        with self.cond:
            sock = self.sock
        try:
            if sock is not None:
                sock.shutdown(1)                           # SHUT_WR: the daemon ends the session, `down` then exits
        except OSError:
            return 0
        time.sleep(30)                                     # `down` ends the process; this bounds a daemon that ignores EOF
        return 0

    # -- daemon -> host --------------------------------------------------------------------------------------
    def run_down(self) -> None:
        sock, buffer = self.sock, bytearray()
        while True:
            try:
                chunk = sock.recv(CHUNK)
            except OSError:
                chunk = b""
            if chunk:
                try:
                    for line in _lines(buffer, chunk):
                        self.answered(line)
                        _write_all(1, line)
                except OSError:
                    os._exit(EXIT_UNAVAILABLE)             # the host closed its end
                continue
            if self.stdin_closed:
                os._exit(0)
            sock, buffer = self.recover(), bytearray()

    def answered(self, line: bytes) -> None:
        text = line.decode("utf-8", "replace")
        head = _RESPONSE_HEAD.match(text)
        try:
            if head:
                key = _line_key(json.loads(head.group(1)))
            elif len(line) < 1 << 20:                      # another key order: parse it
                reply = json.loads(text)
                if not isinstance(reply, dict) or reply.get("id") is None or ("result" not in reply and "error" not in reply):
                    return
                key = _line_key(reply["id"])
            else:
                return
        except ValueError:
            return
        with self.cond:
            self.pending.pop(key, None)

    def recover(self):
        """Called by the reader when the daemon closed the connection: answer, reconnect, replay. Never returns on failure."""
        now = time.monotonic()
        self.recoveries = [t for t in self.recoveries if now - t < RECOVERY_WINDOW_SEC] + [now]
        with self.cond:
            self.sock = None
            unanswered, self.pending = list(self.pending.values()), {}
        try:
            for original in unanswered:
                _write_all(1, _error_line(original, RESTART_ERROR))
        except OSError:
            os._exit(EXIT_UNAVAILABLE)
        if len(self.recoveries) > MAX_RECOVERIES:
            print("tsukkomi-mcp: the daemon keeps going away; giving up (daemon_unavailable)", file=sys.stderr)
            os._exit(EXIT_UNAVAILABLE)
        try:
            sock = open_session(self.environ, self.mode, self.cwd, python=self.python)
            self._replay(sock)
        except (RelayError, OSError, EOFError, P.ProtocolError, ValueError) as exc:
            message = exc.message if isinstance(exc, RelayError) else type(exc).__name__
            print(f"tsukkomi-mcp: could not reconnect to the daemon: {message}", file=sys.stderr)
            os._exit(exc.code if isinstance(exc, RelayError) else EXIT_UNAVAILABLE)
        with self.cond:
            self.sock = sock
            self.cond.notify_all()
        if self.stdin_closed:
            try:
                sock.shutdown(1)
            except OSError:
                pass
        return sock

    def _replay(self, sock) -> None:
        if self.initialize is None:
            return
        sock.sendall(self.initialize)
        original = _line_key(json.loads(self.initialize)["id"])
        buffer = bytearray()
        sock.settimeout(60.0)
        while True:
            chunk = sock.recv(CHUNK)
            if not chunk:
                raise EOFError("daemon closed during the replayed initialize")
            for line in _lines(buffer, chunk):
                head = _RESPONSE_HEAD.match(line.decode("utf-8", "replace"))
                if head and _line_key(json.loads(head.group(1))) == original:
                    if json.loads(line).get("error"):
                        raise RelayError("the daemon refused the replayed handshake", EXIT_UNAVAILABLE)
                    if self.initialized is not None:
                        sock.sendall(self.initialized)
                    sock.settimeout(None)
                    return


def pipe(sock, environ=None, mode: str = "default", cwd=None, python=None) -> int:
    """Copy stdin to the daemon and the daemon to stdout until either side ends; survive daemon restarts."""
    relay = Relay(sock, os.environ if environ is None else environ, mode, cwd, python)
    threading.Thread(target=relay.run_down, name="relay-down", daemon=True).start()
    return relay.run_up()


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
    return pipe(sock, os.environ, mode, cwd)
