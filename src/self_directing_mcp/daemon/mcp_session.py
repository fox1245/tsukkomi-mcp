"""MCP connections of the daemon: the same FastMCP tool functions, one session per socket connection.

A client (``relay``) sends ``{"op": "mcp", "env", "cwd", "mode"}``. The daemon resolves the settings the standalone
server would have (the client's environment, ``.env`` in the client's working directory, ``--omp-bridge`` validation
for that settings key), picks the engine for them from the shared registry (one engine, and one resident integrity
worker, per settings key, shared with command hooks of the same configuration), answers ``ok`` and then speaks
newline-delimited JSON-RPC on the connection with ``server.mcp``'s low-level server: exactly what ``mcp.run("stdio")``
does for a process's stdin and stdout.

A configuration failure is answered as ``{"ok": false, "reason": "configuration", "message": ...}``; the relay
surfaces it as a JSON-RPC error to the host and exits non-zero. There is no silent fallback to anything.

While the daemon drains, a new request on an existing connection is answered with a JSON-RPC error (``-32603``) and
never started; requests already running finish.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import os
import re
import threading

from self_directing_mcp.daemon import envview, protocol as P

MODES = ("default", "omp-bridge")
DRAIN_ERROR = "tsukkomi daemon is restarting; retry the request"
_RESPONSE_HEAD = re.compile(r'\{"jsonrpc":"2\.0","id":(-?\d+|"(?:[^"\\]|\\.)*"),"(?:result|error)"')


class _Tracker:
    """Which requests of one connection are in flight (the daemon waits for them before it exits)."""

    def __init__(self, daemon) -> None:
        self.daemon = daemon
        self.pending: set = set()

    def start(self, key) -> None:
        if key not in self.pending:
            self.pending.add(key)
            self.daemon.begin_request(None)

    def finish(self, key) -> None:
        if key in self.pending:
            self.pending.discard(key)
            self.daemon.end_request(None)
            self.daemon.after_request()            # the memory cap applies after an MCP request as after a hook

    def close(self) -> None:
        for key in list(self.pending):
            self.finish(key)


class _Lines:
    """Async iterator of the lines a client sent (the shape ``mcp.server.stdio.stdio_server`` reads)."""

    def __init__(self, reader, writer, tracker: _Tracker) -> None:
        self._reader, self._writer, self._tracker = reader, writer, tracker

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        while True:
            try:
                raw = await self._reader.readline()
            except (ValueError, asyncio.LimitOverrunError):      # a line beyond the frame limit: end the session
                raise StopAsyncIteration from None
            if not raw:
                raise StopAsyncIteration
            self._tracker.daemon.touch()
            message = _parse(raw)
            if isinstance(message, dict):
                method, key = message.get("method"), message.get("id")
                if method == "notifications/cancelled":
                    self._tracker.finish(_key((message.get("params") or {}).get("requestId")))
                elif method and key is not None:
                    if self._tracker.daemon.draining:
                        self._writer.write((json.dumps({"jsonrpc": "2.0", "id": key, "error": {
                            "code": -32603, "message": DRAIN_ERROR}}) + "\n").encode())
                        continue
                    self._tracker.start(_key(key))
            return raw.decode("utf-8", errors="replace")


class _Out:
    """The write side (``write`` / ``flush`` of an ``anyio.AsyncFile``) onto the connection."""

    def __init__(self, writer, tracker: _Tracker) -> None:
        self._writer, self._tracker = writer, tracker

    async def write(self, text: str) -> None:
        match = _RESPONSE_HEAD.match(text)
        self._writer.write(text.encode("utf-8"))
        if match:
            try:
                self._tracker.finish(_key(json.loads(match.group(1))))
            except ValueError:
                pass

    async def flush(self) -> None:
        await self._writer.drain()


def _parse(raw: bytes):
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _key(value):
    return value if isinstance(value, (int, str)) else repr(value)


class McpHandler:
    def __init__(self) -> None:
        self._server = None
        self._lock = threading.Lock()

    def server_module(self):
        with self._lock:
            if self._server is None:
                from self_directing_mcp import server
                self._server = server
        return self._server

    # -- settings and engine -----------------------------------------------------------------------------------
    def open_engine(self, daemon, env: dict, cwd, mode: str):
        """Settings as the standalone server resolves them, then the shared engine. Runs in a worker thread."""
        server = self.server_module()
        from self_directing_mcp.config import Settings
        env_file = os.path.join(cwd, ".env") if isinstance(cwd, str) and os.path.isabs(cwd) else None
        with envview.request_environment(env):
            def get(**overrides):
                return Settings(_env_file=env_file, **overrides)
            settings = server.omp_bridge_settings(get) if mode == "omp-bridge" else get()
            if not settings.local_only:
                from self_directing_mcp._runtime import load_numpy
                load_numpy()                                      # before worker threads spawn children
            return daemon.engines.get(settings)

    # -- the connection ---------------------------------------------------------------------------------------
    async def serve(self, daemon, reader, writer, frame: dict) -> None:
        mode, cwd = frame.get("mode", "default"), frame.get("cwd")
        try:
            env, _argv = P.validate_environment(frame.get("env", {}), [])
            if mode not in MODES or not (cwd is None or isinstance(cwd, str)):
                raise P.ProtocolError("bad mcp request")
        except P.ProtocolError:
            await self._refuse(writer, P.REASON_BAD_REQUEST, "bad request")
            return
        loop = asyncio.get_running_loop()
        try:
            engine = await loop.run_in_executor(None, contextvars.copy_context().run, self.open_engine, daemon,
                                                env, cwd, mode)
        except (OSError, ValueError) as exc:
            daemon.stats["mcp_refused"] += 1
            daemon.log.event("mcp_refused", mode=mode, error=type(exc).__name__)
            message = (self.server_module().OMP_BRIDGE_INVALID if mode == "omp-bridge"
                       else f"server configuration is invalid ({type(exc).__name__})")
            await self._refuse(writer, "configuration", message)
            return
        server = self.server_module()
        await P.write_frame_async(writer, {"op": P.OP_MCP, "ok": True, "served_by": "daemon"})
        tracker = _Tracker(daemon)
        daemon.stats["mcp_sessions"] += 1
        daemon.log.event("mcp_open", mode=mode, sessions=daemon.stats["mcp_sessions"])
        token = server.current_engine.set(engine)
        try:
            with envview.request_environment(env):
                from mcp.server.stdio import stdio_server
                async with stdio_server(stdin=_Lines(reader, writer, tracker), stdout=_Out(writer, tracker)) as (
                        read_stream, write_stream):
                    await server.mcp._mcp_server.run(read_stream, write_stream,
                                                     server.mcp._mcp_server.create_initialization_options())
        finally:
            server.current_engine.reset(token)
            tracker.close()
            daemon.log.event("mcp_close", mode=mode)

    @staticmethod
    async def _refuse(writer, reason: str, message: str) -> None:
        await P.write_frame_async(writer, {"op": P.OP_MCP, "ok": False, "status": "error", "reason": reason,
                                           "message": message})
