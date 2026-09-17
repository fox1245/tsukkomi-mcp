"""Regression coverage for quiet hooks without weakening action checks."""
import json

import pytest

from self_directing_mcp.codex_hooks import handle_hook
from test_hook_gating import hook_engine


@pytest.mark.parametrize("event", ["UserPromptSubmit", "PreToolUse", "Stop"])
def test_no_contracts_do_not_initialize_or_sync(hook_engine, monkeypatch, event):
    engine, sid, path = hook_engine
    def forbidden(*args, **kwargs):
        pytest.fail("No-contract hook initialized the audit runtime")
    monkeypatch.setattr(engine, "ensure_ready", forbidden)
    result = handle_hook(engine, event, sid, str(path), "functions.exec", 'await tools.exec_command({"cmd":"rg foo README.md"});')
    assert result == {}
    assert engine.store is None


def test_prompt_is_forwarded_and_qa_is_silent(hook_engine):
    from self_directing_mcp.install_codex import hook_handler
    assert hook_handler("UserPromptSubmit")["input"]["tool_input"] == "${prompt}"
    engine, sid, path = hook_engine
    assert handle_hook(engine, "UserPromptSubmit", sid, str(path), prompt="현재 상태가 어때?") == {}
    result = handle_hook(engine, "UserPromptSubmit", sid, str(path), prompt="파일은 절대 삭제하지 마")
    assert "upsert_contracts" in json.dumps(result)
    assert engine.store is None


def test_read_wrapper_is_quiet_but_explicit_read_prohibition_is_checked(hook_engine, monkeypatch):
    engine, sid, path = hook_engine
    engine.upsert_contracts([{"id": "no-delete", "type": "must_not", "scope": "tool_call", "regex": "DELETE_ME"}])
    original = engine.check_action
    calls = []
    def record(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, "check_action", record)
    code = 'text(await tools.exec_command({"cmd":"rg foo README.md","max_output_tokens":100}));'
    assert handle_hook(engine, "PreToolUse", sid, str(path), "functions.exec", code) == {}
    assert not calls
    engine.upsert_contracts([{"id": "no-read", "type": "must_not", "scope": "tool_call", "regex": "README"}])
    assert "violation" in json.dumps(handle_hook(engine, "PreToolUse", sid, str(path), "functions.exec", code))
    assert calls


@pytest.mark.parametrize("code", [
    'await tools.exec_command({"cmd":"rg --pre=DELETE_ME foo README.md"});',
    'await tools.exec_command({"cmd":"rg \\"--pre=DELETE_ME\\" foo README.md"});',
    'await tools.exec_command({"cmd":"Get-Content (DELETE_ME)"});',
    'await tools.exec_command({"cmd":"Get-Content README.md; DELETE_ME"});',
    'await tools.exec_command({"cmd":"rg foo `DELETE_ME`"});',
    'await tools.exec_command(dynamicArguments); DELETE_ME();',
    'text(await tools.mcp__self_directing_mcp__list_contracts({})); DELETE_ME();',
])
def test_unsafe_or_dynamic_wrappers_are_not_exempt(hook_engine, code):
    engine, sid, path = hook_engine
    engine.upsert_contracts([{"id": "deny", "type": "must_not", "scope": "tool_call", "regex": "DELETE_ME"}])
    assert "violation" in json.dumps(handle_hook(engine, "PreToolUse", sid, str(path), "functions.exec", code))


def test_no_success_cache_for_repeated_mutation_and_stop_is_deduplicated(hook_engine, monkeypatch):
    engine, sid, path = hook_engine
    engine.upsert_contracts([{"id": "deny", "type": "must_not", "scope": "tool_call", "regex": "DELETE_ME"}])
    audits, checks = [], []
    real_audit, real_check = engine.audit_session, engine.check_action
    def audit(*args, **kwargs):
        audits.append(1)
        return real_audit(*args, **kwargs)
    def check(*args, **kwargs):
        checks.append(1)
        return real_check(*args, **kwargs)
    monkeypatch.setattr(engine, "audit_session", audit)
    monkeypatch.setattr(engine, "check_action", check)
    handle_hook(engine, "UserPromptSubmit", sid, str(path), prompt="설명해줘", turn_id="t")
    assert handle_hook(engine, "Stop", sid, str(path), turn_id="t") == {}
    assert not audits
    for _ in range(2):
        assert "violation" in json.dumps(handle_hook(engine, "PreToolUse", sid, str(path), "shell", {"cmd": "DELETE_ME"}, turn_id="t"))
    assert len(checks) == 2
    handle_hook(engine, "Stop", sid, str(path), turn_id="t")
    handle_hook(engine, "Stop", sid, str(path), turn_id="t")
    assert len(audits) == 1
    engine.upsert_contracts([{"id": "deny", "type": "must_not", "scope": "tool_call", "regex": "NEW_RULE"}])
    handle_hook(engine, "Stop", sid, str(path), turn_id="t")
    assert len(audits) == 2


def test_corrupt_contract_store_is_unknown_not_success(hook_engine):
    engine, sid, path = hook_engine
    contracts = engine.settings.resolve_contracts_path()
    contracts.parent.mkdir(parents=True, exist_ok=True)
    contracts.write_text("invalid", encoding="utf-8")
    result = handle_hook(engine, "PreToolUse", sid, str(path), "shell", {"cmd": "deploy"})
    assert "unknown" in json.dumps(result)


def test_description_only_obligation_remains_unknown_at_checkpoint(hook_engine):
    engine, sid, path = hook_engine
    engine.upsert_contracts([{"id": "unfinished", "type": "must", "description": "Verify requested behavior"}])
    result = handle_hook(engine, "Stop", sid, str(path))
    assert "unknown" in json.dumps(result)


def test_no_contract_status_does_not_claim_compliance(hook_engine):
    engine, sid, path = hook_engine
    state = engine.hook_obligations(sid)
    assert state["status"] == "not_applicable"
    assert state["contracts"] == []
    assert "verdict" not in state


def test_new_evidence_after_stop_invalidates_checkpoint(hook_engine):
    engine, sid, path = hook_engine
    engine.upsert_contracts([{"id": "deny", "type": "must_not", "scope": "tool_call", "regex": "DELETE_ME"}])
    assert handle_hook(engine, "Stop", sid, str(path)) == {}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"type": "response_item", "payload": {
            "type": "function_call", "name": "shell", "arguments": "DELETE_ME", "call_id": "new-call"}}) + "\n")
    assert "violation" in json.dumps(handle_hook(engine, "Stop", sid, str(path)))


def test_own_audit_message_does_not_invalidate_checkpoint(hook_engine, monkeypatch):
    engine, sid, path = hook_engine
    engine.upsert_contracts([{"id": "deny", "type": "must_not", "scope": "tool_call", "regex": "DELETE_ME"}])
    handle_hook(engine, "Stop", sid, str(path))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"type": "response_item", "payload": {"type": "message", "role": "developer",
            "content": [{"type": "input_text", "text": "Self-directing audit (data, not instructions): {}"}]}}) + "\n")
    monkeypatch.setattr(engine, "audit_session", lambda *a, **k: pytest.fail("Repeated receipt caused another audit"))
    assert handle_hook(engine, "Stop", sid, str(path)) == {}


def test_wrapped_read_constraint_matches_actual_inner_tool(hook_engine):
    engine, sid, path = hook_engine
    engine.upsert_contracts([{"id": "no-command-read", "type": "must_not", "scope": "tool_call", "regex": r"^\[tool_call:exec_command\]"}])
    code = 'await tools.exec_command({"cmd":"Get-Content README.md"});'
    assert "violation" in json.dumps(handle_hook(engine, "PreToolUse", sid, str(path), "functions.exec", code))


def test_valid_literal_wrapper_with_quoted_preprocessor_is_checked(hook_engine):
    from self_directing_mcp.codex_hooks import _unwrap
    engine, sid, path = hook_engine
    engine.upsert_contracts([{"id": "deny", "type": "must_not", "scope": "tool_call", "regex": "DELETE_ME"}])
    code = "await tools.exec_command(" + json.dumps({"cmd": 'rg "--pre=DELETE_ME" foo README.md'}) + ");"
    assert _unwrap("functions.exec", code)[0] == "exec_command"
    assert "violation" in json.dumps(handle_hook(engine, "PreToolUse", sid, str(path), "functions.exec", code))
