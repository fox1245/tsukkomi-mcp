from __future__ import annotations
import copy
import json
import pytest


def test_managed_guidance_is_idempotent_and_preserves_other_text():
    from self_directing_mcp.install_codex import merge_guidance, BEGIN, END
    original = "# Personal rules\n\nKeep this sentence.\n"
    once = merge_guidance(original)
    assert once.startswith(original)
    assert once.count(BEGIN) == once.count(END) == 1
    assert merge_guidance(once) == once


def test_broken_guidance_markers_refused():
    from self_directing_mcp.install_codex import merge_guidance, BEGIN
    with pytest.raises(ValueError):
        merge_guidance("user text\n" + BEGIN)


def test_hook_merge_preserves_existing_positions_and_is_idempotent():
    from self_directing_mcp.install_codex import merge_hooks, SERVER, HOOK_TOOL
    existing = {"description": "keep", "hooks": {"UserPromptSubmit": [
        {"hooks": [{"type": "mcp_tool", "server": "other", "tool": "keep_me"}]}],
        "PostToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "echo keep"}]}]}}
    baseline = copy.deepcopy(existing)
    merged = merge_hooks(existing)
    assert existing == baseline
    assert merged["hooks"]["UserPromptSubmit"][0] == baseline["hooks"]["UserPromptSubmit"][0]
    assert merged["hooks"]["PostToolUse"] == baseline["hooks"]["PostToolUse"]
    assert merge_hooks(merged) == merged
    owned = [h for groups in merged["hooks"].values() for group in groups for h in group["hooks"]
             if h.get("server") == SERVER and h.get("tool") == HOOK_TOOL]
    assert len(owned) == 3
    assert "tool_input" in json.dumps(merged)
    assert "permissionDecision" not in json.dumps(merged)


def test_ambiguous_duplicate_hooks_are_refused():
    from self_directing_mcp.install_codex import merge_hooks
    existing = merge_hooks({})
    existing["hooks"]["PreToolUse"] *= 2
    with pytest.raises(ValueError):
        merge_hooks(existing)


def test_config_preserves_unrelated_server_settings(tmp_path):
    from self_directing_mcp.install_codex import server_config
    previous = {"startup_timeout_sec": 91, "env": {"USER_SETTING": "keep"}, "enabled": False}
    planned = server_config(tmp_path / "python.exe", tmp_path, previous)
    assert planned["startup_timeout_sec"] == 91
    assert planned["env"]["USER_SETTING"] == "keep"
    assert planned["enabled"] is True
    assert planned["env"]["SELF_DIRECT_USE_FAKE_EMBEDDER"] == "false"
    assert planned["env"]["SELF_DIRECT_DENSE_BACKEND"] == "sqlite-vector"
    assert planned["command"] == str((tmp_path / "python.exe").resolve())


@pytest.fixture
def hook_engine(tmp_path, monkeypatch):
    from self_directing_mcp.config import Settings
    from self_directing_mcp.engine import SelfDirectEngine
    for key in ("SELF_DIRECT_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    root = tmp_path / "sessions"
    root.mkdir()
    sid = "11111111-2222-3333-4444-555555555555"
    path = root / f"rollout-2026-09-09T01-00-00-{sid}.jsonl"
    path.write_text(json.dumps({"type": "session_meta", "payload": {"id": sid}}) + "\n", encoding="utf-8")
    engine = SelfDirectEngine(Settings(codex_sessions_dir=root, index_dir=tmp_path / "index",
                                      use_fake_embedder=True, _env_file=None))
    yield engine, sid, path
    engine.close()


def test_pretool_hook_reports_violation_without_blocking(hook_engine):
    from self_directing_mcp.codex_hooks import handle_hook
    engine, sid, path = hook_engine
    engine.upsert_contracts([{"id": "delete", "type": "must_not", "scope": "tool_call", "regex": "DELETE_ME"}])
    result = handle_hook(engine, "PreToolUse", sid, str(path), "Bash",
                         {"command": "echo DELETE_ME"}, turn_id="turn1")
    assert result["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert "violation" in result["hookSpecificOutput"]["additionalContext"]
    assert "permissionDecision" not in json.dumps(result)
    assert "decision" not in result
    assert engine.store.chunk_count(sid) == 1


def test_hook_does_not_guess_missing_session(hook_engine):
    from self_directing_mcp.codex_hooks import handle_hook
    engine, _, _ = hook_engine
    result = handle_hook(engine, "UserPromptSubmit", None)
    assert result == {}
    assert engine.store is None


def test_hooks_skip_their_own_mcp_calls(hook_engine):
    from self_directing_mcp.codex_hooks import handle_hook
    engine, sid, path = hook_engine
    assert handle_hook(engine, "PreToolUse", sid, str(path),
                       "mcp__self_directing_mcp__audit_session", {}) == {}


def test_stop_does_not_create_continuation_loop(hook_engine):
    from self_directing_mcp.codex_hooks import handle_hook
    engine, sid, path = hook_engine
    assert handle_hook(engine, "Stop", sid, str(path), stop_hook_active=True) == {}


def test_hook_never_uses_remote_embeddings(hook_engine, monkeypatch):
    from self_directing_mcp.codex_hooks import handle_hook
    engine, sid, path = hook_engine
    engine.ensure_ready()
    attempts = []
    def forbidden(*args):
        attempts.append(args)
        raise AssertionError("hook attempted embedding")
    monkeypatch.setattr(engine.embedder, "embed_documents", forbidden)
    monkeypatch.setattr(engine.embedder, "embed_queries", forbidden)
    context = handle_hook(engine, "UserPromptSubmit", sid, str(path), prompt="Never delete user files")
    assert "upsert_contracts" in json.dumps(context)
    assert attempts == []


def test_outside_transcript_path_is_not_read(hook_engine, tmp_path):
    from self_directing_mcp.codex_hooks import handle_hook
    engine, sid, _ = hook_engine
    path = tmp_path / "outside.jsonl"
    path.write_text("PRIVATE_SENTINEL\n", encoding="utf-8")
    engine.upsert_contracts([{"id": "deny", "type": "must_not", "scope": "tool_call", "regex": "delete"}])
    result = handle_hook(engine, "Stop", sid, str(path))
    assert "PRIVATE_SENTINEL" not in json.dumps(result)
    assert "unknown" in json.dumps(result)


def test_effective_global_override_is_selected(tmp_path):
    from self_directing_mcp.install_codex import guidance_path
    assert guidance_path(tmp_path).name == "AGENTS.md"
    (tmp_path / "AGENTS.override.md").write_text("My override", encoding="utf-8")
    assert guidance_path(tmp_path).name == "AGENTS.override.md"


def test_atomic_write_refuses_concurrent_user_edit(tmp_path):
    from self_directing_mcp.install_codex import atomic_write
    path = tmp_path / "AGENTS.md"
    path.write_bytes(b"user updated this")
    with pytest.raises(RuntimeError):
        atomic_write(path, b"our content", b"old content")
    assert path.read_bytes() == b"user updated this"


def test_dry_run_writes_nothing(tmp_path):
    from self_directing_mcp.install_codex import configure
    home = tmp_path / "new-home"
    result = configure(home, tmp_path / "python", "codex", trust_hooks=True, dry_run=True)
    assert result["dry_run"] and not home.exists()


def test_separate_servers_share_contracts_without_lost_updates(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    worker = """import sys
from pathlib import Path
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
engine=SelfDirectEngine(Settings(index_dir=Path(sys.argv[1]),use_fake_embedder=True,_env_file=None))
for i in range(8):
    engine.upsert_contracts([{'id':sys.argv[2]+'-'+str(i),'type':'must_not','regex':'NEVER'}])
engine.close()
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    processes = [subprocess.Popen([sys.executable, "-B", "-c", worker, str(tmp_path / "index"), str(i)],
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(3)]
    for process in processes:
        _, error = process.communicate(timeout=30)
        assert process.returncode == 0, error.decode(errors="replace")
    from self_directing_mcp.config import Settings
    from self_directing_mcp.engine import SelfDirectEngine
    engine = SelfDirectEngine(Settings(index_dir=tmp_path / "index", use_fake_embedder=True, _env_file=None))
    try:
        assert len(engine.list_contracts()["contracts"]) == 24
    finally:
        engine.close()
