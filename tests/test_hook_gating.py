"""Tighter hook conditions: noisy prompts filtered, risky actions audited."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_directing_mcp.codex_hooks import (
    _carries_requirement_signal,
    handle_hook,
    _is_risky_input,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SAMPLE_ROOT = FIXTURES / "sample_session"
SESSION_ID = "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def hook_engine(tmp_path: Path):
    from self_directing_mcp.config import Settings
    from self_directing_mcp.engine import SelfDirectEngine

    sessions = tmp_path / "sessions"
    (sessions / "2026" / "09" / "09").mkdir(parents=True)
    path = sessions / "2026" / "09" / "09" / f"rollout-2026-09-09T01-00-00-{SESSION_ID}.jsonl"
    lines = [
        json.dumps({"type": "session_meta", "payload": {"id": SESSION_ID}}),
        json.dumps({"type": "response_item", "payload": {"type": "message",
                   "role": "user", "content": [{"type": "input_text", "text": "hello"}]}}),
    ]
    path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    settings = Settings(codex_sessions_dir=sessions, index_dir=tmp_path / "index",
                        use_fake_embedder=True,
                        contracts_path=tmp_path / "index" / "contracts.json")
    engine = SelfDirectEngine(settings=settings)
    yield engine, SESSION_ID, path
    engine.close()


def test_requirement_signal_detection():
    assert _carries_requirement_signal("운영 DB는 수정하지 마")
    assert _carries_requirement_signal("must run tests before deploy")
    assert _carries_requirement_signal("금지된 디렉터리는 건드리지 않는다")
    assert not _carries_requirement_signal("이 함수가 뭐 하는 거야?")
    assert not _carries_requirement_signal("감사합니다")


def test_risky_input_detection():
    assert _is_risky_input({"command": "rm -rf /tmp/x"})
    assert _is_risky_input({"command": "kubectl delete pods"})
    assert _is_risky_input("deploy to production")
    assert not _is_risky_input({"command": "ls -la"})
    assert not _is_risky_input({"path": "docs/readme.md"})


def test_user_prompt_submit_qa_is_silent(hook_engine):
    """Pure Q&A does not need an audit or registration reminder."""
    engine, sid, path = hook_engine
    engine.ensure_ready()
    result = handle_hook(engine, "UserPromptSubmit", sid, str(path),
                         tool_input="이 세션에서 뭘 했지?")
    text = json.dumps(result, ensure_ascii=False)
    assert "upsert_contracts" not in text
    assert result == {}


def test_user_prompt_submit_requirement_language_keeps_guidance(hook_engine):
    engine, sid, path = hook_engine
    engine.ensure_ready()
    result = handle_hook(engine, "UserPromptSubmit", sid, str(path),
                         tool_input="테스트를 반드시 돌리고 운영 DB는 수정하지 마")
    text = json.dumps(result, ensure_ascii=False)
    assert "upsert_contracts" in text


def test_pretool_use_trivial_read_is_skipped(hook_engine):
    engine, sid, path = hook_engine
    engine.ensure_ready()
    # A pure read on a non-risky input must not run a full audit.
    result = handle_hook(engine, "PreToolUse", sid, str(path),
                         tool_name="read", tool_input={"path": "docs/readme.md"})
    assert result == {}


def test_pretool_use_risky_input_forces_audit(hook_engine):
    engine, sid, path = hook_engine
    engine.ensure_ready()
    engine.upsert_contracts([{"id": "no-delete", "type": "must_not",
                              "scope": "tool_call", "regex": r"rm\s+-rf"}])
    result = handle_hook(engine, "PreToolUse", sid, str(path),
                         tool_name="shell", tool_input={"command": "rm -rf /data"})
    assert result != {}
    assert "violation" in json.dumps(result)


