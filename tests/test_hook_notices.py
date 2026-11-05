"""Repeatable hook notices are said once per session and again only when their text changes; verdicts never are.

Synthetic data only. An "unverified" notice (a contract that cannot be checked) costs the model tokens on every tool
call and carries nothing new after the first time. A denial or a violation is a verdict and stays on every call.
"""
from __future__ import annotations

from self_directing_mcp import agy_hook_state as shared_state
from self_directing_mcp import claude_hook_state as state

from test_claude_hooks import (
    ACK, OTHER, SID, _unknown, _violation, env, health, main_file, permission, run, stop, subagent_stop,  # noqa: F401
    write, human,
)


def _call(engine, path, **kwargs):
    return run(engine, path, "PreToolUse", tool_input={"command": "echo ok"}, **kwargs)


def test_an_unchanged_unverified_notice_reaches_the_user_and_the_model_once_per_session(env):
    engine, path = env
    _unknown(engine)
    first = _call(engine, path)
    assert "WARNING" in first["systemMessage"] and "[c]" in first["systemMessage"]
    assert "Unverified tool call allowed" in run(engine, path, "PostToolUse")["hookSpecificOutput"]["additionalContext"]
    for _ in range(4):
        assert _call(engine, path) == {}                       # no systemMessage, nothing queued for the model
        assert run(engine, path, "PostToolUse") == {}


def test_the_notice_is_said_again_when_the_set_of_unverified_contracts_changes(env):
    engine, path = env
    _unknown(engine)
    _call(engine, path)
    assert _call(engine, path) == {}
    engine.upsert_contracts([{"id": "d", "type": "must_not", "provider": "claude", "regex": "echo",
                              "applies_from_event_id": "not-in-this-session", "description": "a second unverified rule"}])
    changed = _call(engine, path)
    assert "[c]" in changed["systemMessage"] and "[d]" in changed["systemMessage"]
    assert _call(engine, path) == {}


def test_another_session_gets_its_own_first_notice(env):
    engine, path = env
    _unknown(engine)
    _call(engine, path)
    assert _call(engine, path) == {}
    other = write(main_file(engine.settings.resolve_claude_projects_dir(), OTHER), human("hi"))
    response = run(engine, other, "PreToolUse", sid=OTHER, tool_input={"command": "echo ok"})
    assert "systemMessage" in response


def test_a_denial_is_a_verdict_and_is_repeated_on_every_call(env):
    engine, path = env
    tool_input = _violation(engine)
    for _ in range(3):
        response = run(engine, path, "PreToolUse", tool_input=tool_input)
        assert permission(response) == "deny" and response["hookSpecificOutput"]["permissionDecisionReason"]


def test_stop_notes_about_an_unchanged_gap_are_said_once_and_a_violation_is_said_every_time(env):
    engine, path = env
    engine.upsert_contracts([ACK])
    assert "decision" in stop(engine, path)                    # the one automatic continuation
    first = stop(engine, path)
    assert "budget" in first["systemMessage"].lower() and "violation" in first["systemMessage"]
    for _ in range(3):
        later = stop(engine, path)
        assert "budget" not in later["systemMessage"].lower()
        assert "violation" in later["systemMessage"]


def test_a_stop_note_without_any_verdict_is_silent_after_the_first_time(env):
    engine, path = env
    engine.upsert_contracts([{"id": "prose", "type": "must", "provider": "claude", "scope": "any",
                              "description": "a requirement with no verifier"}])
    first = stop(engine, path)
    assert "Not verified" in first["systemMessage"] and "[prose]" in first["systemMessage"]
    assert stop(engine, path) == {}
    assert stop(engine, path) == {}


def test_subagent_stops_share_the_notice_of_their_parent_session(env):
    engine, path = env
    engine.upsert_contracts([ACK])
    assert "unavailable" in subagent_stop(engine)["systemMessage"]
    assert subagent_stop(engine) == {}


def test_claim_notice_compares_with_the_last_notice_of_the_same_kind(tmp_path):
    store = state.ClaudeHookState(tmp_path)
    assert store.claim_notice("s", "k", "text") is True
    assert store.claim_notice("s", "k", "text") is False
    assert store.claim_notice("s", "other-kind", "text") is True
    assert store.claim_notice("t", "k", "text") is True
    assert store.claim_notice("s", "k", "changed") is True
    assert store.claim_notice("s", "k", "text") is True            # compared with the last one, not with every earlier one


def test_an_unreadable_state_shows_the_notice_instead_of_dropping_it(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")
    store = shared_state.HookState(blocker)                        # its parent path is a file: the database cannot open
    assert store.claim_notice("s", "k", "text") is True
    assert store.claim_notice("s", "k", "text") is True
