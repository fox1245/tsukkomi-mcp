"""A hook that cannot run its audit still refuses what a high-risk contract gates, and says so once per session.

Synthetic data only. The audit failing (here: the handler raising, as a path error did in production) must not
silently allow an action that a user-selected ``risk: high`` contract is about to gate; unrelated actions keep working,
and the degraded state is reported once per session instead of on every tool call.
"""
from __future__ import annotations

from self_directing_mcp import claude_hooks

from test_claude_hooks import _raising, env, main_file, permission, run, set_mode, stop  # noqa: F401

GATED = "git push --force origin x"
GATE = {"id": "gate", "type": "must_not", "scope": "tool_call", "provider": "claude", "regex": "git push --force",
        "risk": "high"}


def _call(engine, path, command, **kwargs):
    return run(engine, path, "PreToolUse", tool_input={"command": command}, **kwargs)


def test_a_failing_audit_refuses_the_action_a_high_risk_contract_gates_and_lets_the_rest_through(env, monkeypatch):
    engine, path = env
    engine.upsert_contracts([GATE])
    monkeypatch.setattr(claude_hooks, "handle_pre_tool_use", _raising)
    refused = _call(engine, path, GATED)
    assert permission(refused) == "deny"
    reason = refused["hookSpecificOutput"]["permissionDecisionReason"]
    assert "gate" in reason and "could not be verified" in reason and "RuntimeError" in reason
    assert permission(_call(engine, path, "ls")) == "none"


def test_the_refusal_is_said_every_time_the_degraded_notice_only_once(env, monkeypatch):
    engine, path = env
    engine.upsert_contracts([GATE])
    monkeypatch.setattr(claude_hooks, "handle_pre_tool_use", _raising)
    first = _call(engine, path, "ls")
    assert "FAILED" in first["systemMessage"]
    for _ in range(3):
        assert _call(engine, path, "ls") == {}                        # nothing new to say to the user
    for _ in range(3):
        assert permission(_call(engine, path, GATED)) == "deny"
    # the model hears about the degraded state once, through the next PostToolUse
    context = run(engine, path, "PostToolUse")["hookSpecificOutput"]["additionalContext"]
    assert "FAILED" in context
    assert run(engine, path, "PostToolUse") == {}


def test_degraded_notices_are_per_session_and_per_event(env, monkeypatch):
    engine, path = env
    monkeypatch.setattr(claude_hooks, "handle_pre_tool_use", _raising)
    monkeypatch.setattr(claude_hooks, "handle_stop", _raising)
    assert "systemMessage" in _call(engine, path, "ls")
    assert _call(engine, path, "ls") == {}
    assert "systemMessage" in stop(engine, path)                      # another event of the same session
    assert stop(engine, path) == {}
    other = "bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb"
    assert "systemMessage" in run(engine, path, "PreToolUse", sid=other, tool_input={"command": "ls"})


def test_a_contract_that_names_tools_gates_only_those_tools_when_the_audit_fails(env, monkeypatch):
    engine, path = env
    engine.upsert_contracts([dict(GATE, tools=["Bash"])])
    monkeypatch.setattr(claude_hooks, "handle_pre_tool_use", _raising)
    edit = run(engine, path, "PreToolUse", tool="Edit",
               tool_input={"file_path": "notes.txt", "new_string": "text that names " + GATED})
    assert permission(edit) == "none"
    assert permission(_call(engine, path, GATED)) == "deny"


def test_advisory_mode_never_refuses_when_the_audit_fails(env, monkeypatch):
    engine, path = env
    set_mode(engine, "advisory")
    monkeypatch.setenv("SELF_DIRECT_CLAUDE_ENFORCEMENT", "advisory")
    engine.upsert_contracts([GATE])
    monkeypatch.setattr(claude_hooks, "handle_pre_tool_use", _raising)
    assert permission(_call(engine, path, GATED)) == "none"


def test_a_normal_risk_contract_does_not_refuse_when_the_audit_fails(env, monkeypatch):
    engine, path = env
    engine.upsert_contracts([dict(GATE, risk="normal")])
    monkeypatch.setattr(claude_hooks, "handle_pre_tool_use", _raising)
    assert permission(_call(engine, path, GATED)) == "none"


def test_a_path_error_on_every_call_of_a_session_refuses_the_gated_ones_and_is_noted_once(env, monkeypatch):
    """The 2026-11-05 incident: a transcript path error on every PreToolUse of a session."""
    engine, path = env
    engine.upsert_contracts([GATE])

    def broken(*args, **kwargs):
        raise OSError("transcript path has a linked component")
    monkeypatch.setattr(claude_hooks, "_validated_path", broken)
    answers = [_call(engine, path, GATED if n % 2 else "echo ok") for n in range(6)]
    assert [permission(a) for a in answers] == ["none", "deny", "none", "deny", "none", "deny"]
    assert sum("systemMessage" in a for a in answers) == 1


def test_a_high_risk_prerequisite_is_asked_not_refused_because_history_may_already_satisfy_it(env, monkeypatch):
    engine, path = env
    engine.upsert_contracts([{"id": "tests-first", "type": "must", "scope": "tool_call", "provider": "claude",
                              "before_regex": "git push", "regex": "pytest", "risk": "high"}])
    monkeypatch.setattr(claude_hooks, "handle_pre_tool_use", _raising)
    response = _call(engine, path, "git push origin x")
    assert permission(response) == "ask" and "tests-first" in response["hookSpecificOutput"]["permissionDecisionReason"]
    assert permission(_call(engine, path, "ls")) == "none"


def test_the_tools_filter_knows_the_canonical_tool_name_of_a_grok_call(tmp_path):
    import json
    from self_directing_mcp.agy_hook_state import high_risk_hit
    contracts = {"contracts": [{"id": "gate", "type": "must_not", "scope": "tool_call", "regex": "git push --force",
                                "risk": "high", "tools": ["Bash"], "provider": "grok"}]}
    path = tmp_path / "contracts.json"
    path.write_text(json.dumps(contracts), encoding="utf-8")
    call = {"name": "run_terminal_command", "args": {"command": GATED}}
    assert high_risk_hit(path, "s", call, provider="grok") == ("gate", "must_not")
    assert high_risk_hit(path, "s", {"name": "read_file", "args": {"path": GATED}}, provider="grok") is None
