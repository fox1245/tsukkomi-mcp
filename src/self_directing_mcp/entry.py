"""Console entry point: standalone MCP server, or the daemon relay when ``SELF_DIRECT_DAEMON=on``.

This module imports only the standard library (through ``daemon.client``) until it has decided, so that a relay never
pays for the MCP SDK, pydantic or the engine.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    from self_directing_mcp.daemon import client
    if client.use_daemon():
        from self_directing_mcp.daemon import relay
        raise SystemExit(relay.main(argv))
    from self_directing_mcp.server import main as standalone
    standalone(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
