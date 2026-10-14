"""MCP connections of the daemon (filled in by the MCP proxy commit)."""
from __future__ import annotations


class McpHandler:
    async def serve(self, daemon, reader, writer, frame) -> None:
        raise NotImplementedError
