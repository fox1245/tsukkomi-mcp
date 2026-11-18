"""A Claude Code PreToolUse audit that fails for a transient reason of the tool itself is retried inside the hook, and a
call that a high-risk ``must`` contract gates is then refused to the agent (``agent_review``), not asked of the user.

Synthetic data only. The failure is injected where the production one arose (the index lease: ``IndexBusy``, raised by
the engine while several writers share the index), through the same ``run_hook`` that the standalone hook and the
resident daemon both call. Time is a fake clock: nothing here sleeps.
"""
from __future__ import annotations

import json

import pytest

from self_directing_mcp import claude_hook_state as state
from self_directing_mcp import claude_hooks
from self_directing_mcp.daemon import client
from self_directing_mcp.hook_policy import HookRuntimeError
from self_directing_mcp.request_control import IndexBusy, RequestStopped

from test_claude_hooks import (  # noqa: F401  (env is a fixture)
    assistant, env, health, human, main_file, permission, run, set_mode, stop, tool_result, use, write,
)

PUSH = {"command": "git push origin x"}
GATE = {"id": "tests-first", "type": "must", "scope": "tool_call", "provider": "claude", "regex": "pytest",
        "before_regex": "git push", "risk": "high"}
FORBID = {"id": "no-force", "type": "must_not", "scope": "tool_call", "provider": "claude",
          "regex": "git push --force", "risk": "high"}
REASONS = ("audit could not run", "nothing was verified")


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(claude_hooks, "_monotonic", fake)
    monkeypatch.setattr(claude_hooks, "_sleep", fake.sleep)
    return fake


def failing(monkeypatch, error, *, times=None):
    """``handle_pre_tool_use`` raises ``error()`` ``times`` times (always when None), then answers for real."""
    real, calls = claude_hooks.handle_pre_tool_use, []

    def handler(*args, **kwargs):
        calls.append(1)
        if times is None or len(calls) <= times:
            raise error()
        return real(*args, **kwargs)
    monkeypatch.setattr(claude_hooks, "handle_pre_tool_use", handler)
    return calls


def push(engine, path, tool_input=PUSH):
    return run(engine, path, "PreToolUse", tool_input=tool_input)


def reason_of(response):
    return response["hookSpecificOutput"]["permissionDecisionReason"]


def kinds(engine):
    return {(row["kind"], row["event"]): row for row in health(engine).health()}


def write_passing_tests(path):
    write(path, assistant(use("toolu_p1", "Bash", command="pytest -q")), tool_result("toolu_p1", "5 passed"), mode="a")


def attempts_allowed(engine):
    settings = engine.settings
    return int(settings.audit_timeout_sec // settings.lock_wait_timeout_sec)


# --- the retry --------------------------------------------------------------------------------------------

def test_a_busy_index_that_frees_up_is_retried_and_the_call_gets_its_normal_audit(env, clock, monkeypatch):
    engine, path = env
    set_mode(engine, "agent_review")
    engine.upsert_contracts([GATE])
    write_passing_tests(path)
    calls = failing(monkeypatch, IndexBusy, times=2)
    response = push(engine, path)
    assert len(calls) == 3                                   # two failures, then the real audit
    assert permission(response) == "none"                    # its verdict: the prerequisite is satisfied
    assert "FAILED" not in json.dumps(response)
    assert kinds(engine) == {}                               # nothing was recorded as a failure


def test_the_retry_wraps_the_whole_hook_so_a_wrapped_busy_index_is_retried_too(env, clock, monkeypatch):
    """``read_obligations`` reports an index failure as a ``HookRuntimeError`` whose cause is the busy index."""
    engine, path = env
    set_mode(engine, "agent_review")
    engine.upsert_contracts([GATE])
    write_passing_tests(path)

    def wrapped():
        try:
            raise IndexBusy()
        except IndexBusy as busy:
            error = HookRuntimeError("contracts or checklist unavailable")
            error.__cause__ = busy
            return error
    calls = failing(monkeypatch, wrapped, times=1)
    assert permission(push(engine, path)) == "none" and len(calls) == 2


def test_retries_stay_inside_the_audit_budget_and_are_spaced_by_the_lock_wait(env, clock, monkeypatch):
    engine, path = env
    set_mode(engine, "agent_review")
    engine.upsert_contracts([GATE])
    calls = failing(monkeypatch, IndexBusy)
    started = clock.now
    push(engine, path)
    settings = engine.settings
    assert len(calls) == attempts_allowed(engine) > 1
    assert clock.now - started <= settings.audit_timeout_sec


def test_a_retry_never_outlives_the_request_deadline(env, clock, monkeypatch):
    from self_directing_mcp.request_control import request_scope
    engine, path = env
    set_mode(engine, "agent_review")
    engine.upsert_contracts([GATE])
    calls = failing(monkeypatch, IndexBusy)
    with request_scope(deadline=clock.now + 2 * engine.settings.lock_wait_timeout_sec):
        push(engine, path)
    assert len(calls) == 2


def test_a_deadline_that_has_passed_is_not_retried(env, clock, monkeypatch):
    engine, path = env
    set_mode(engine, "agent_review")
    engine.upsert_contracts([GATE])
    calls = failing(monkeypatch, lambda: RequestStopped("request_deadline"))
    response = push(engine, path)
    assert len(calls) == 1
    assert permission(response) == "deny" and "RequestStopped" in reason_of(response)


def test_only_a_pre_tool_use_audit_is_retried(env, clock, monkeypatch):
    engine, path = env
    set_mode(engine, "agent_review")
    calls = []

    def busy(*args, **kwargs):
        calls.append(1)
        raise IndexBusy()
    monkeypatch.setattr(claude_hooks, "handle_stop", busy)
    stop(engine, path)
    assert len(calls) == 1


def test_an_error_that_is_not_transient_is_not_retried(env, clock, monkeypatch):
    engine, path = env
    set_mode(engine, "agent_review")
    engine.upsert_contracts([GATE])
    calls = failing(monkeypatch, lambda: ValueError("broken"))
    response = push(engine, path)
    assert len(calls) == 1
    assert permission(response) == "ask"                     # a broken audit still goes to the user


# --- the refusal for the agent ----------------------------------------------------------------------------

def test_a_persistent_failure_refuses_a_gated_prerequisite_to_the_agent_and_never_asks(env, clock, monkeypatch):
    engine, path = env
    set_mode(engine, "agent_review")
    engine.upsert_contracts([GATE])
    failing(monkeypatch, IndexBusy)
    response = push(engine, path)
    assert permission(response) == "deny"
    reason = reason_of(response)
    assert "IndexBusy" in reason and "tests-first" in reason
    assert all(part in reason.lower() for part in REASONS)
    assert "retry the identical call" in reason and "audit_status" in reason
    assert "confirm manually" not in reason and "fail-open" not in reason
    assert permission(run(engine, path, "PreToolUse", tool_input={"command": "ls"})) == "none"


def test_each_refusal_is_recorded_in_the_health_record_under_its_own_kind(env, clock, monkeypatch):
    engine, path = env
    set_mode(engine, "agent_review")
    engine.upsert_contracts([GATE])
    failing(monkeypatch, IndexBusy)
    for _ in range(2):
        assert permission(push(engine, path)) == "deny"
    rows = kinds(engine)
    refused = rows[(state.AGENT_RETRY_REFUSED, "PreToolUse")]
    assert refused["total"] == 2 and refused["last_error_class"] == "IndexBusy"
    assert rows[(state.CHECK_FAILED, "PreToolUse")]["total"] == 2        # the failure itself is counted as before
    assert state.AGENT_RETRY_REFUSED not in (state.CHECK_FAILED, state.AUDIT_UNAVAILABLE)


def test_a_working_call_ends_the_streak_of_refusals(env, clock, monkeypatch):
    engine, path = env
    set_mode(engine, "agent_review")
    engine.upsert_contracts([GATE])
    write_passing_tests(path)
    failing(monkeypatch, IndexBusy, times=attempts_allowed(engine))
    assert permission(push(engine, path)) == "deny"
    assert permission(push(engine, path)) == "none"
    assert kinds(engine)[(state.AGENT_RETRY_REFUSED, "PreToolUse")]["consecutive"] == 0


def test_a_contract_that_leaves_unverified_calls_to_the_user_still_asks(env, clock, monkeypatch):
    engine, path = env
    set_mode(engine, "agent_review")
    engine.upsert_contracts([dict(GATE, on_unverified="human")])
    failing(monkeypatch, IndexBusy)
    response = push(engine, path)
    assert permission(response) == "ask" and "tests-first" in reason_of(response)
    assert (state.AGENT_RETRY_REFUSED, "PreToolUse") not in kinds(engine)


@pytest.mark.parametrize("mode", ["confirmed_only", "enforced"])
def test_in_the_modes_where_the_user_confirms_the_call_is_still_asked(env, clock, monkeypatch, mode):
    engine, path = env
    set_mode(engine, mode)
    engine.upsert_contracts([GATE])
    failing(monkeypatch, IndexBusy)
    response = push(engine, path)
    assert permission(response) == "ask" and "confirm manually" in reason_of(response)
    assert (state.AGENT_RETRY_REFUSED, "PreToolUse") not in kinds(engine)


def test_advisory_mode_changes_nothing(env, clock, monkeypatch):
    engine, path = env
    set_mode(engine, "advisory")
    monkeypatch.setenv("SELF_DIRECT_CLAUDE_ENFORCEMENT", "advisory")
    engine.upsert_contracts([GATE])
    failing(monkeypatch, IndexBusy)
    assert permission(push(engine, path)) == "none"
    assert (state.AGENT_RETRY_REFUSED, "PreToolUse") not in kinds(engine)


def test_a_forbidden_action_is_refused_as_before(env, clock, monkeypatch):
    engine, path = env
    set_mode(engine, "agent_review")
    engine.upsert_contracts([FORBID])
    failing(monkeypatch, IndexBusy)
    response = push(engine, path, {"command": "git push --force origin x"})
    assert permission(response) == "deny"
    assert "forbids what this action matches" in reason_of(response) and "no-force" in reason_of(response)
    assert (state.AGENT_RETRY_REFUSED, "PreToolUse") not in kinds(engine)


def test_a_call_no_high_risk_contract_gates_passes_with_a_notice(env, clock, monkeypatch):
    engine, path = env
    set_mode(engine, "agent_review")
    engine.upsert_contracts([GATE])
    failing(monkeypatch, IndexBusy)
    response = run(engine, path, "PreToolUse", tool_input={"command": "ls"})
    assert permission(response) == "none" and "FAILED" in response["systemMessage"]
    assert (state.AGENT_RETRY_REFUSED, "PreToolUse") not in kinds(engine)


def test_a_normal_risk_contract_does_not_refuse(env, clock, monkeypatch):
    engine, path = env
    set_mode(engine, "agent_review")
    engine.upsert_contracts([dict(GATE, risk="normal")])
    failing(monkeypatch, IndexBusy)
    assert permission(push(engine, path)) == "none"


# --- the other route to the same user prompt --------------------------------------------------------------

def test_a_busy_index_while_the_review_is_looked_up_is_not_turned_into_a_question_for_the_user(env, clock, monkeypatch):
    """An unverified prerequisite reaches the review step; finding the review needs the index, which is busy."""
    from self_directing_mcp import action_reviews
    engine, path = env
    set_mode(engine, "agent_review")
    engine.settings.mcp_server_name = "tsukkomi"
    engine.upsert_contracts([dict(GATE, requires_success=True)])
    write(path, assistant(use("toolu_t1", "Bash", command="pytest -q")), mode="a")      # no result yet: unknown
    monkeypatch.setattr(action_reviews.ReviewStore, "candidates",
                        lambda self, **kwargs: [{"review_id": "r1", "refusal_id": "f1", "source_offset": 0}])

    def busy(*args, **kwargs):
        raise IndexBusy()
    monkeypatch.setattr(engine, "review_origin_events", busy)
    response = push(engine, path)
    assert permission(response) == "deny" and "IndexBusy" in reason_of(response)
    assert "user decides" not in reason_of(response)


# --- the hook client (resident daemon) --------------------------------------------------------------------

@pytest.fixture
def client_env(tmp_path, monkeypatch):
    contracts = tmp_path / "contracts.json"
    contracts.write_text(json.dumps({"contracts": [GATE]}), encoding="utf-8")
    monkeypatch.setenv("SELF_DIRECT_CONTRACTS_PATH", str(contracts))
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("SELF_DIRECT_CLAUDE_ENFORCEMENT", "agent_review")
    fake = Clock()
    monkeypatch.setattr(client, "_monotonic", fake)
    monkeypatch.setattr(client, "_sleep", fake.sleep)
    return tmp_path


def served(monkeypatch, answers):
    """``_serve`` answers from ``answers`` in turn (the last one repeats); returns the list of calls."""
    calls = []

    def serve(host, event, payload, argv, environ, python, **kwargs):
        calls.append(kwargs)
        return answers[min(len(calls), len(answers)) - 1]
    monkeypatch.setattr(client, "_serve", serve)
    return calls


def payload(command="git push origin x"):
    return {"session_id": "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa", "tool_name": "Bash", "tool_input": {"command": command},
            "hook_event_name": "PreToolUse"}


def test_an_unreachable_daemon_is_waited_for_again_when_a_gated_call_depends_on_it(client_env, monkeypatch):
    calls = served(monkeypatch, [(None, client.R_UNAVAILABLE), ({"decision": "served"}, None)])
    response, from_daemon = client.serve_hook("claude", "PreToolUse", payload(), [])
    assert from_daemon is True and response == {"decision": "served"}
    assert len(calls) == 2 and calls[1].get("spawn") is False        # the daemon that was started is waited for, not restarted


def test_the_wait_for_the_daemon_ends_with_the_longest_availability_budget(client_env, monkeypatch):
    calls = served(monkeypatch, [(None, client.R_UNAVAILABLE)])
    started = client._monotonic()
    response, from_daemon = client.serve_hook("claude", "PreToolUse", payload(), [])
    assert from_daemon is False and len(calls) > 1
    assert client._monotonic() - started <= client.AVAILABILITY_SLOW_SEC
    assert permission(response) == "deny" and "daemon_unavailable" in reason_of(response)
    assert all(part in reason_of(response).lower() for part in REASONS)


def test_a_call_no_contract_gates_does_not_wait_for_the_daemon_again(client_env, monkeypatch):
    calls = served(monkeypatch, [(None, client.R_UNAVAILABLE)])
    response, from_daemon = client.serve_hook("claude", "PreToolUse", payload("ls"), [])
    assert from_daemon is False and len(calls) == 1 and permission(response) == "none"


def test_a_daemon_that_timed_out_has_used_the_budget_and_is_not_asked_again(client_env, monkeypatch):
    calls = served(monkeypatch, [(None, client.R_TIMEOUT)])
    response, _ = client.serve_hook("claude", "PreToolUse", payload(), [])
    assert len(calls) == 1
    assert permission(response) == "deny" and "daemon_timeout" in reason_of(response)


@pytest.mark.parametrize("reason", [client.R_CRASH_LOOP, client.R_RESTART_LOOP, client.R_SETTING, "version_mismatch"])
def test_a_daemon_that_is_deliberately_not_started_is_neither_waited_for_nor_refused_to_the_agent(
        client_env, monkeypatch, reason):
    calls = served(monkeypatch, [(None, reason)])
    response, _ = client.serve_hook("claude", "PreToolUse", payload(), [])
    assert len(calls) == 1
    assert permission(response) == "ask"


def test_the_client_keeps_asking_in_confirmed_only_mode(client_env, monkeypatch):
    monkeypatch.setenv("SELF_DIRECT_CLAUDE_ENFORCEMENT", "confirmed_only")
    served(monkeypatch, [(None, client.R_UNAVAILABLE)])
    response, _ = client.serve_hook("claude", "PreToolUse", payload(), [])
    assert permission(response) == "ask"


def test_the_agy_hook_client_is_unchanged(client_env, monkeypatch):
    calls = served(monkeypatch, [(None, client.R_UNAVAILABLE)])
    agy = {"conversationId": "c" * 32, "toolCall": {"name": "run_command", "args": {"command": "git push origin x"}}}
    client.serve_hook("agy", "PreToolUse", agy, [])
    assert len(calls) == 1
