"""Exercise actual MCP stdio transport, schema validation and concurrent requests."""
import asyncio
import json
import os
import sys
import subprocess
from pathlib import Path

import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

def _server_env(config_path):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("SELF_DIRECT_") and key not in ("OPENROUTER_API_KEY", "CODEX_SESSIONS_DIR")}
    env.update(SELF_DIRECT_CONFIG_FILE=str(config_path),
               PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
               PYTHONDONTWRITEBYTECODE="1")
    return env



def test_stdio_contract_to_preflight(tmp_path):
    async def scenario():
        config = tmp_path / "config.toml"
        config.write_text("", encoding="utf-8")
        sid = "11111111-2222-3333-4444-555555555555"
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        path = sessions / f"rollout-2026-09-09T01-00-00-{sid}.jsonl"
        path.write_text(json.dumps({"type": "session_meta", "payload": {"id": sid}}) + "\n", encoding="utf-8")
        # SDK v1 ClientSession / stdio_client, checked against installed SDK source.
        env = _server_env(config)
        env.update(SELF_DIRECT_USE_FAKE_EMBEDDER="true", SELF_DIRECT_DENSE_BACKEND="numpy",
                   SELF_DIRECT_CODEX_SESSIONS_DIR=str(sessions), SELF_DIRECT_INDEX_DIR=str(tmp_path / "index"))
        params = StdioServerParameters(command=sys.executable, args=["-m", "self_directing_mcp"], env=env, cwd=str(tmp_path))
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
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


@pytest.mark.parametrize("env_override", [False, True])
def test_omp_bridge_uses_selected_roots_and_index(tmp_path, env_override):
    config_dir = tmp_path / "configuration"
    config_dir.mkdir()
    toml_root = config_dir / "history"
    toml_root.mkdir()
    config = config_dir / "config.toml"
    config.write_text('[paths]\nomp_sessions_dir = "history"\nindex_dir = "index"\n', encoding="utf-8")
    cwd = tmp_path / "working"
    cwd.mkdir()
    env = _server_env(config)
    env["SELF_DIRECT_LOCAL_ONLY"] = "true"
    # Explicit OMP startup settings take precedence even over a provider override.
    env["SELF_DIRECT_SESSION_PROVIDER"] = "codex"
    root, index = toml_root, config_dir / "index"
    if env_override:
        root, index = tmp_path / "environment-history", tmp_path / "environment-index"
        root.mkdir()
        env.update(SELF_DIRECT_OMP_SESSIONS_DIR=str(root), SELF_DIRECT_INDEX_DIR=str(index))
    sid = "bridge-session"
    (root / "session.jsonl").write_text(
        json.dumps({"type": "session", "id": sid}) + "\n" +
        json.dumps({"type": "message", "message": {
            "role": "user", "content": [{"type": "text", "text": "Do not delete anything."}]}}) + "\n",
        encoding="utf-8",
    )

    async def scenario():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "self_directing_mcp.server", "--omp-bridge"],
            env=env, cwd=str(cwd),
        )
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                assert not index.exists()  # Transport startup does not initialize runtime resources.
                response = await client.call_tool("upsert_contracts", {"contracts": [
                    {"id": "no-delete", "type": "must_not", "scope": "tool_call",
                     "provider": "omp", "regex": r"rm\s+-rf"}
                ]})
                assert not response.isError
                response = await client.call_tool("check_action", {
                    "session_id": sid,
                    "action": {"tool_name": "shell", "arguments": {"command": "rm -rf /fictional-test-path"}},
                })
                result = json.loads(response.content[0].text)
                assert result["verdict"] == "violation"
                assert result["coverage"]["complete"] is True
                assert result["action_executed"] is False
                response = await client.call_tool("audit_status", {"session_id": sid})
                status = json.loads(response.content[0].text)
                assert status["session"]["chunk_count"] == 1
                assert status["dense_backend"] == "disabled"
    asyncio.run(asyncio.wait_for(scenario(), timeout=30))
    assert (index / "contracts.json").is_file()
    assert not (cwd / "index").exists()
    if env_override:
        assert not (config_dir / "index").exists()


@pytest.mark.parametrize("failure", ["unselected_key", "missing_key", "empty_key", "missing_native", "environment_key", "environment_native"])
def test_omp_bridge_rejects_invalid_private_sources_without_raw_key_fallback(tmp_path, failure):
    config = tmp_path / "config.toml"
    key = tmp_path / "private-key.env"
    key.write_text("OPENROUTER_API_KEY=selected-secret-sentinel\n", encoding="utf-8")
    native = tmp_path / "vector.so"
    # Startup checks presence only; this cannot be loaded as a native extension.
    native.write_bytes(b"not a native library")
    paths = {"index_dir": "index", "openrouter_api_key_file": key.name, "sqlite_vector_path": native.name}
    if failure == "unselected_key":
        paths.pop("openrouter_api_key_file")
    elif failure == "missing_key":
        key.unlink()
    elif failure == "empty_key":
        key.write_text("OTHER_SETTING=fixture\n", encoding="utf-8")
    elif failure == "missing_native":
        native.unlink()
    config.write_text("[paths]\n" + "".join(f"{name} = {json.dumps(value)}\n" for name, value in paths.items()), encoding="utf-8")
    env = _server_env(config)
    env.update(SELF_DIRECT_LOCAL_ONLY="false", OPENROUTER_API_KEY="ambient-secret-sentinel")
    if failure == "environment_key":
        env["SELF_DIRECT_OPENROUTER_API_KEY_FILE"] = str(tmp_path / "missing-env-key")
    elif failure == "environment_native":
        env["SELF_DIRECT_SQLITE_VECTOR_PATH"] = str(tmp_path / "missing-env-native")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=cwd-secret-sentinel\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "self_directing_mcp.server", "--omp-bridge"],
        env=env, cwd=tmp_path, input="", text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert str(tmp_path) not in result.stderr
    assert "secret-sentinel" not in result.stderr
    assert not (tmp_path / "index").exists()


def test_omp_bridge_remote_startup_validates_toml_sources_without_initializing_runtime(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        '[paths]\nindex_dir = "index"\nopenrouter_api_key_file = "private.env"\nsqlite_vector_path = "vector.so"\n',
        encoding="utf-8",
    )
    (tmp_path / "private.env").write_text("OPENROUTER_API_KEY=selected-secret-sentinel\n", encoding="utf-8")
    (tmp_path / "vector.so").write_bytes(b"not a native library")
    env = _server_env(config)
    env["SELF_DIRECT_LOCAL_ONLY"] = "false"

    async def scenario():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "self_directing_mcp.server", "--omp-bridge"],
            env=env, cwd=str(tmp_path),
        )
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                assert not (tmp_path / "index").exists()
                response = await client.call_tool("audit_status", {})
                assert response.isError
                assert "selected-secret-sentinel" not in json.dumps(response.model_dump())
    asyncio.run(asyncio.wait_for(scenario(), timeout=30))
