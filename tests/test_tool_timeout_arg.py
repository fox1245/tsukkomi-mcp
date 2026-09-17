"""Issue #1 regression: per-call timeout_ms on long-running tools."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _server_env(tmp_path: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    sid = "12121212-3434-5656-7878-909090909090"
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    path = sessions / f"rollout-2026-09-09T01-00-00-{sid}.jsonl"
    line = json.dumps({"type": "session_meta", "payload": {"id": sid}})
    path.write_text(line + chr(10), encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("SELF_DIRECT_") and k not in ("OPENROUTER_API_KEY", "CODEX_SESSIONS_DIR")}
    env.update(SELF_DIRECT_USE_FAKE_EMBEDDER="true", SELF_DIRECT_DENSE_BACKEND="numpy", SELF_DIRECT_CODEX_SESSIONS_DIR=str(sessions),
               SELF_DIRECT_INDEX_DIR=str(tmp_path / "index"),
               PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"), PYTHONDONTWRITEBYTECODE="1")
    if extra:
        env.update(extra)
    return env


def _stdio(env, tmp_path):
    return StdioServerParameters(command=sys.executable, args=["-m", "self_directing_mcp"],
                                 env=env, cwd=str(tmp_path))


SID = "12121212-3434-5656-7878-909090909090"


def test_tools_expose_optional_timeout_ms(tmp_path):
    async def scenario():
        async with stdio_client(_stdio(_server_env(tmp_path), tmp_path)) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                schemas = {t.name: t.inputSchema for t in (await client.list_tools()).tools}
                for name in ("sync_session", "audit_session", "check_action", "search_history"):
                    props = schemas[name].get("properties", {})
                    assert "timeout_ms" in props, name
                    kinds = [opt.get("type") for opt in props["timeout_ms"].get("anyOf", [{"type": props["timeout_ms"].get("type")}])]
                    assert "integer" in kinds
                    assert "timeout_ms" not in schemas[name].get("required", [])
    asyncio.run(asyncio.wait_for(scenario(), timeout=30))


def test_timeout_ms_accepted_and_invalid_values_rejected(tmp_path):
    async def scenario():
        async with stdio_client(_stdio(_server_env(tmp_path), tmp_path)) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                # Explicit large timeout is accepted end-to-end.
                sync = await client.call_tool("sync_session",
                                              {"session_id": SID, "timeout_ms": 120000})
                assert not sync.isError
                audit = await client.call_tool("audit_session",
                                               {"session_id": SID, "timeout_ms": 120000})
                payload = json.loads(audit.content[0].text)
                assert payload["verdict"] in ("clean", "violation", "suspicious", "unknown")
                # Non-positive and over-cap values are rejected with a clear error.
                bad = await client.call_tool("audit_session", {"session_id": SID, "timeout_ms": 0})
                assert bad.isError and "positive" in bad.content[0].text
                over = await client.call_tool("sync_session",
                                              {"session_id": SID, "timeout_ms": 10_000_000})
                assert over.isError and "max_tool_timeout_sec" in over.content[0].text
    asyncio.run(asyncio.wait_for(scenario(), timeout=30))


def test_default_behavior_without_timeout_unchanged(tmp_path):
    async def scenario():
        env = _server_env(tmp_path, {"SELF_DIRECT_AUDIT_TIMEOUT_SEC": "6"})
        async with stdio_client(_stdio(env, tmp_path)) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                audit = await client.call_tool("audit_session", {"session_id": SID})
                payload = json.loads(audit.content[0].text)
                assert "timings_ms" in payload
    asyncio.run(asyncio.wait_for(scenario(), timeout=30))
