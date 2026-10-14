"""Issue #6 regression: upsert_contracts schema exposure and precise input errors."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import pytest


def _server_env(tmp_path: Path) -> dict[str, str]:
    sid = "99999999-8888-7777-6666-555555555555"
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
    return env


def test_tools_list_exposes_required_fields_and_enum(tmp_path):
    async def scenario():
        params = StdioServerParameters(command=sys.executable, args=["-m", "self_directing_mcp"],
                                       env=_server_env(tmp_path), cwd=str(tmp_path))
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                tool = next(t for t in (await client.list_tools()).tools if t.name == "upsert_contracts")
                schema = tool.inputSchema
                defs = schema.get("$defs", {})
                rule = defs.get("ContractRule") or next(iter(defs.values()))
                assert "id" in rule.get("required", [])
                assert "type" in rule.get("required", [])
                assert rule["properties"]["type"]["enum"] == ["must", "must_not"]
                assert rule.get("additionalProperties") is False
    asyncio.run(asyncio.wait_for(scenario(), timeout=30))


def test_missing_type_and_bad_enum_rejected_without_partial_save(tmp_path):
    async def scenario():
        params = StdioServerParameters(command=sys.executable, args=["-m", "self_directing_mcp"],
                                       env=_server_env(tmp_path), cwd=str(tmp_path))
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                ok = await client.call_tool("upsert_contracts", {"contracts": [
                    {"id": "baseline", "type": "must", "scope": "message", "regex": "VERIFIED"}]})
                assert not ok.isError
                bad = await client.call_tool("upsert_contracts", {"contracts": [
                    {"id": "preserve-existing-services", "description": "do not touch"}]})
                assert bad.isError
                text = bad.content[0].text
                assert "contracts[0]" in text and "preserve-existing-services" in text and "type" in text
                enum_bad = await client.call_tool("upsert_contracts", {"contracts": [
                    {"id": "wrong-enum", "type": "forbidden", "description": "x"}]})
                assert enum_bad.isError
                assert "forbidden" in enum_bad.content[0].text
                listed = await client.call_tool("list_contracts", {})
                ids = [c["id"] for c in json.loads(listed.content[0].text)["contracts"]]
                assert ids == ["baseline"]
    asyncio.run(asyncio.wait_for(scenario(), timeout=30))


def test_partial_batch_does_not_update_existing_contracts():
    from pydantic import TypeAdapter, ValidationError
    from self_directing_mcp.schemas import ContractRule

    adapter = TypeAdapter(list[ContractRule])
    batch = [
        {"id": "good", "type": "must_not", "description": "fine"},
        {"id": "bad", "description": "missing type"},
    ]
    with pytest.raises(ValidationError):
        adapter.validate_python(batch)


def test_engine_upsert_still_accepts_models_and_dicts(tmp_path):
    from self_directing_mcp.config import Settings
    from self_directing_mcp.engine import SelfDirectEngine
    from self_directing_mcp.schemas import ContractRule

    settings = Settings(use_fake_embedder=True, index_dir=tmp_path / "index",
                        contracts_path=tmp_path / "index" / "contracts.json")
    engine = SelfDirectEngine(settings=settings)
    try:
        updated = engine.upsert_contracts([
            ContractRule(id="model-rule", type="must", scope="message", regex="DONE"),
            {"id": "dict-rule", "type": "must_not", "regex": "NEVER"},
        ])
        ids = [c["id"] for c in updated["contracts"]]
        assert ids == ["model-rule", "dict-rule"]
    finally:
        engine.close()
