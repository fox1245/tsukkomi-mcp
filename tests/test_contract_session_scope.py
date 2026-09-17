"""Issue #7: contract upsert/list stay in the current session plus explicit globals."""
from __future__ import annotations

import json
from pathlib import Path

from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine

SID_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
SID_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def _engine(tmp_path: Path) -> SelfDirectEngine:
    return SelfDirectEngine(Settings(
        use_fake_embedder=True,
        index_dir=tmp_path / "index",
        contracts_path=tmp_path / "index" / "contracts.json",
        _env_file=None,
    ))


def _rule(rule_id: str, session_id: str | None = None, regex: str = "NEVER") -> dict:
    payload = {"id": rule_id, "type": "must_not", "scope": "tool_call", "regex": regex}
    if session_id is not None:
        payload["session_id"] = session_id
    return payload


def test_upsert_returns_only_contracts_from_this_call(tmp_path):
    engine = _engine(tmp_path)
    try:
        engine.upsert_contracts([_rule("petfair-other", SID_B), _rule("global-no-rm")])
        result = engine.upsert_contracts([_rule("current-session", SID_A, regex="DELETE")])
        ids = [c["id"] for c in result["contracts"]]
        assert ids == ["current-session"]
        assert result["returned"] == "upserted"
        assert result["counts"]["upserted"] == 1
        assert result["counts"]["store"] == 3
        assert "global_meaning" in result
        assert "null" in result["global_meaning"]
    finally:
        engine.close()


def test_list_without_session_id_returns_only_global_rules(tmp_path):
    engine = _engine(tmp_path)
    try:
        engine.upsert_contracts([
            _rule("global-no-rm"),
            _rule("session-a", SID_A),
            _rule("session-b", SID_B),
        ])
        listed = engine.list_contracts()
        ids = [c["id"] for c in listed["contracts"]]
        assert ids == ["global-no-rm"]
        assert listed["scope"] == "global_only"
        assert listed["session_id"] is None
        assert listed["counts"]["global"] == 1
        assert listed["counts"]["session"] == 0
        assert listed["counts"]["store"] == 3
        assert "null" in listed["global_meaning"]
    finally:
        engine.close()


def test_list_for_session_includes_globals_not_other_sessions(tmp_path):
    engine = _engine(tmp_path)
    try:
        engine.upsert_contracts([
            _rule("global-no-rm"),
            _rule("session-a", SID_A),
            _rule("session-b", SID_B),
        ])
        listed = engine.list_contracts(session_id=SID_A)
        ids = [c["id"] for c in listed["contracts"]]
        assert ids == ["global-no-rm", "session-a"]
        assert listed["scope"] == "session_and_global"
        assert listed["session_id"] == SID_A
        assert listed["counts"]["global"] == 1
        assert listed["counts"]["session"] == 1
        assert listed["counts"]["store"] == 3
    finally:
        engine.close()


def test_audit_applies_current_session_and_global_rules_only(tmp_path, monkeypatch):
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / f"rollout-2026-09-09T01-00-00-{SID_A}.jsonl"
    events = [
        {"type": "session_meta", "payload": {"id": SID_A}},
        {"type": "response_item", "payload": {
            "type": "function_call", "name": "shell", "call_id": "c1",
            "arguments": {"command": "echo hello"},
        }},
    ]
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    monkeypatch.delenv("SELF_DIRECT_CODEX_SESSIONS_DIR", raising=False)
    engine = SelfDirectEngine(Settings(
        codex_sessions_dir=root,
        index_dir=tmp_path / "index",
        contracts_path=tmp_path / "index" / "contracts.json",
        use_fake_embedder=True,
        _env_file=None,
    ))
    try:
        engine.upsert_contracts([
            _rule("global-no-rm", regex=r"rm\s+-rf"),
            _rule("session-a-no-delete", SID_A, regex="DELETE"),
            _rule("session-b-no-petfair", SID_B, regex="PETFAIR"),
        ])
        engine.sync_session(session_id=SID_A, embed=False)
        audit = engine.audit_session(SID_A)
        finding_ids = {item["contract_id"] for item in audit["findings"]}
        assert finding_ids == {"global-no-rm", "session-a-no-delete"}
        assert "session-b-no-petfair" not in finding_ids
        assert audit["contract_scope"]["scope"] == "session_and_global"
        assert "null" in audit["global_meaning"]
    finally:
        engine.close()


def _server_env(tmp_path: Path) -> dict:
    import os
    sid = SID_A
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    path = sessions / f"rollout-2026-09-09T01-00-00-{sid}.jsonl"
    path.write_text(json.dumps({"type": "session_meta", "payload": {"id": sid}}) + "\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("SELF_DIRECT_") and k not in ("OPENROUTER_API_KEY", "CODEX_SESSIONS_DIR")}
    env.update(
        SELF_DIRECT_USE_FAKE_EMBEDDER="true",
        SELF_DIRECT_DENSE_BACKEND="numpy",
        SELF_DIRECT_CODEX_SESSIONS_DIR=str(sessions),
        SELF_DIRECT_INDEX_DIR=str(tmp_path / "index"),
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def test_mcp_list_contracts_session_id_and_upsert_response(tmp_path):
    import asyncio
    import sys
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def scenario():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "self_directing_mcp"],
            env=_server_env(tmp_path), cwd=str(tmp_path),
        )
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                tool = next(t for t in (await client.list_tools()).tools if t.name == "list_contracts")
                assert "session_id" in tool.inputSchema.get("properties", {})
                await client.call_tool("upsert_contracts", {"contracts": [
                    _rule("other-session", SID_B),
                    _rule("global-no-rm"),
                ]})
                upserted = await client.call_tool("upsert_contracts", {"contracts": [
                    _rule("current-session", SID_A, regex="DELETE"),
                ]})
                payload = json.loads(upserted.content[0].text)
                assert [c["id"] for c in payload["contracts"]] == ["current-session"]
                assert payload["returned"] == "upserted"
                assert payload["counts"]["store"] == 3
                listed = await client.call_tool("list_contracts", {"session_id": SID_A})
                scoped = json.loads(listed.content[0].text)
                assert [c["id"] for c in scoped["contracts"]] == ["global-no-rm", "current-session"]
                globals_only = json.loads((await client.call_tool("list_contracts", {})).content[0].text)
                assert [c["id"] for c in globals_only["contracts"]] == ["global-no-rm"]

    asyncio.run(asyncio.wait_for(scenario(), timeout=30))

