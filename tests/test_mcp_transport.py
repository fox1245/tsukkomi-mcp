"""Exercise actual MCP stdio transport, schema validation and concurrent requests."""
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_stdio_contract_to_preflight(tmp_path):
    async def scenario():
        from pathlib import Path
        sid = "11111111-2222-3333-4444-555555555555"
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        path = sessions / f"rollout-2026-09-09T01-00-00-{sid}.jsonl"
        path.write_text(json.dumps({"type": "session_meta", "payload": {"id": sid}}) + "\n", encoding="utf-8")
        # SDK v1 ClientSession / stdio_client, checked against installed SDK source.
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("SELF_DIRECT_") and key not in ("OPENROUTER_API_KEY", "CODEX_SESSIONS_DIR")}
        env.update(SELF_DIRECT_USE_FAKE_EMBEDDER="true", SELF_DIRECT_DENSE_BACKEND="numpy", SELF_DIRECT_CODEX_SESSIONS_DIR=str(sessions),
                   SELF_DIRECT_INDEX_DIR=str(tmp_path / "index"),
                   PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"), PYTHONDONTWRITEBYTECODE="1")
        params = StdioServerParameters(command=sys.executable, args=["-m", "self_directing_mcp"], env=env, cwd=str(tmp_path))
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                names = {tool.name for tool in (await client.list_tools()).tools}
                assert {"check_action", "sync_session", "audit_session"} <= names
                response = await client.call_tool("upsert_contracts", {"contracts": [
                    {"id": "no-delete", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"}
                ]})
                assert not response.isError
                responses = await asyncio.gather(*[
                    client.call_tool("check_action", {"session_id": sid, "timeout_ms": 5000, "action": {
                        "tool_name": "shell", "arguments": {"command": command}}})
                    for command in ("echo hello", "rm -rf /fictional-test-path")
                ])
                values = [json.loads(r.content[0].text) for r in responses]
                assert [v["verdict"] for v in values] == ["clean", "violation"]
                assert all(v["action_executed"] is False for v in values)
                assert all(v["coverage"]["complete"] is True for v in values)
                assert values[0]["contracts_snapshot"] == values[1]["contracts_snapshot"]
                status = await client.call_tool("audit_status", {"session_id": sid})
                assert json.loads(status.content[0].text)["session"]["chunk_count"] == 1
    asyncio.run(asyncio.wait_for(scenario(), timeout=30))
