"""A check of a proposed action settles when its source grows while it is being judged.

While a PreToolUse hook runs, Claude Code appends records about the proposed call itself: the call's own assistant
record, the results of the other PreToolUse hooks of that call and session bookkeeping. A check whose source grew during
its evaluation reads the appended bytes and evaluates again until one evaluation covers the source with no growth during
it; no evaluation starts later than ``SELF_DIRECT_ACTION_SETTLE_SEC`` after the first one ended. The answer is that
evaluation's, under the existing rules: a satisfied prerequisite is clean, a newer failed run of it is a violation, a newer
run without a result is unknown, and bytes that were never read (growth that does not stop, an unfinished or unparseable
record) stay unknown.

Synthetic data only. Growth is injected deterministically: ``engine.run_audit`` is wrapped so that a batch of records is
appended during an evaluation (``_audit``), right after its contracts were judged and before the engine looks at the source
again (no threads, no sleeps). The settle budget is read from an injected clock where a test needs a fixed number of passes.
"""
from __future__ import annotations

from contextlib import contextmanager
import json

import pytest

from self_directing_mcp import engine as engine_module
from test_claude_hooks import SID, assistant, env, env_record, permission, run, tool_result, use  # noqa: F401
from test_claude_hooks import write

GATE = {"id": "check-first", "provider": "claude", "type": "must", "scope": "tool_call", "regex": "make check",
        "before_regex": "git push", "requires_success": True, "risk": "high"}
PUSH = "git push origin main"


def _gate(engine, path):
    """A high-risk prerequisite that ran and succeeded before the push is proposed."""
    engine.upsert_contracts([GATE])
    write(path, assistant(use("toolu_check", "Bash", command="make check")),
          tool_result("toolu_check", "all checks passed"), mode="a")


def own_records(command=PUSH, call_id="toolu_push"):
    """What the host writes about the proposed call while its PreToolUse hooks run (kinds seen in real transcripts):
    the call's own assistant record, another PreToolUse hook's result for that call, and session bookkeeping."""
    return [assistant(use(call_id, "Bash", command=command)),
            env_record("attachment", attachment={
                "type": "hook_success", "hookEvent": "PreToolUse", "hookName": "PreToolUse:Bash", "exitCode": 0,
                "durationMs": 39, "command": "synthetic-other-hook", "content": "", "stdout": "", "stderr": "",
                "toolUseID": call_id}),
            env_record("last-prompt", lastPrompt="synthetic"), env_record("ai-title", aiTitle="synthetic"),
            env_record("mode", mode="normal"), env_record("permission-mode", permissionMode="default"),
            env_record("atis-latch", latched=True)]


def newer_run(call_id, *, failed=None):
    """A newer run of the prerequisite: failed (``True``), succeeded (``False``) or without a result yet (``None``)."""
    records = [assistant(use(call_id, "Bash", command="make check"))]
    if failed is not None:
        records.append(tool_result(call_id, "Exit code 2" if failed else "ok", is_error=failed))
    return records


def encoded(records):
    return "".join(json.dumps(record) + "\n" for record in records).encode()


def _append(path, batch):
    with path.open("ab") as stream:
        stream.write(batch if isinstance(batch, bytes) else encoded(batch))


@contextmanager
def growing(monkeypatch, path, batches):
    """Append ``batches[k]`` during the k-th evaluation of the check (``_audit``; the full verification an evaluation may
    escalate to is part of it), right after its contracts were judged and before it looks at the source again. Yields the
    list of evaluations so far (``True`` for one during which bytes were appended)."""
    audit, judge = engine_module.SelfDirectEngine._audit, engine_module.run_audit
    seen, state = [], {"depth": 0, "batch": None}

    def evaluation(self, *args, **kwargs):
        state["depth"] += 1
        try:
            if state["depth"] == 1:
                seen.append(False)
                state["batch"] = batches[len(seen) - 1] if len(seen) <= len(batches) else None
            return audit(self, *args, **kwargs)
        finally:
            state["depth"] -= 1

    def judged(*args, **kwargs):
        result = judge(*args, **kwargs)
        batch, state["batch"] = state["batch"], None
        if batch is not None:
            _append(path, batch)
            seen[-1] = True
        return result

    monkeypatch.setattr(engine_module.SelfDirectEngine, "_audit", evaluation)
    monkeypatch.setattr(engine_module, "run_audit", judged)
    yield seen


class StepClock:
    """A settle clock that advances ``step`` seconds on every read: a budget of ``2.5 * step`` allows exactly three
    passes (the first one and two that re-read)."""

    def __init__(self, step):
        self.now, self.step = 0.0, step

    def __call__(self):
        self.now += self.step
        return self.now


def check(engine, path, command=PUSH, verify="fast"):
    return engine.check_action(SID, {"tool_name": "Bash", "arguments": {"command": command}}, provider="claude",
                               path=str(path), verify=verify)


def finding(value, contract_id="check-first"):
    return next(item for item in value["findings"] if item["contract_id"] == contract_id)


# --- (a) only the proposed call's own records were appended ---------------------------------------------------

def test_growth_with_only_the_calls_own_records_settles_clean_and_the_hook_does_not_ask(env, monkeypatch):
    engine, path = env
    _gate(engine, path)
    with growing(monkeypatch, path, [own_records()]) as seen:
        response = run(engine, path, "PreToolUse", tool_input={"command": PUSH})
    assert seen[0] is True                       # the source did grow during the first evaluation
    assert response == {}, response              # no ask, no deny, no warning: Claude Code's own permissions decide
    assert permission(response) == "none"


def test_the_settled_answer_is_an_evaluation_that_covered_the_appended_records(env, monkeypatch):
    engine, path = env
    _gate(engine, path)
    with growing(monkeypatch, path, [own_records()]):
        value = check(engine, path)
    assert value["verdict"] == "clean" and value["action_verdict"] == "clean", value
    assert value["coverage"]["complete"] is True
    assert not value["coverage"].get("unread_bytes")
    assert value["coverage"]["byte_offset"] == path.stat().st_size      # the answer covers the source as it is now
    assert value["settle"]["passes"] == 2 and value["settle"]["stopped_by"] == "no_growth"
    item = finding(value)
    assert item["verdict"] == "clean" and "successful" in item["reason"]


def test_the_calls_own_record_counts_as_a_run_only_when_it_matches_the_prerequisite(env, monkeypatch):
    """Once read, the proposed call's own record is a pending run of the prerequisite only if it matches its pattern;
    then the rule for a newer run without a result applies."""
    engine, path = env
    _gate(engine, path)
    command = "make check && " + PUSH
    with growing(monkeypatch, path, [own_records(command)]):
        value = check(engine, path, command)
    assert value["coverage"]["complete"] is True and value["settle"]["passes"] == 2
    assert finding(value)["verdict"] == "unknown"
    assert "completion status" in finding(value)["reason"]
    with growing(monkeypatch, path, [own_records(command, "toolu_push_2")]):
        response = run(engine, path, "PreToolUse", tool_input={"command": command})
    assert permission(response) == "ask"


# --- (b) / (c) a newer run of the prerequisite was appended ---------------------------------------------------------

def test_a_newer_failed_run_in_the_appended_records_is_a_violation(env, monkeypatch):
    engine, path = env
    _gate(engine, path)
    with growing(monkeypatch, path, [newer_run("toolu_check_2", failed=True) + own_records()]):
        value = check(engine, path)
    assert value["coverage"]["complete"] is True and value["settle"]["passes"] == 2
    assert finding(value)["verdict"] == "violation", value
    assert value["action_verdict"] == "violation"
    with growing(monkeypatch, path, [own_records(call_id="toolu_push_2")]):
        response = run(engine, path, "PreToolUse", tool_input={"command": PUSH})
    assert permission(response) == "deny"


def test_a_newer_failed_run_appended_during_the_hook_is_denied(env, monkeypatch):
    engine, path = env
    _gate(engine, path)
    with growing(monkeypatch, path, [newer_run("toolu_check_2", failed=True) + own_records()]):
        response = run(engine, path, "PreToolUse", tool_input={"command": PUSH})
    assert permission(response) == "deny", response


def test_a_newer_run_without_a_result_in_the_appended_records_stays_unknown(env, monkeypatch):
    engine, path = env
    _gate(engine, path)
    with growing(monkeypatch, path, [newer_run("toolu_check_2") + own_records()]):
        value = check(engine, path)
    assert value["coverage"]["complete"] is True and value["settle"]["passes"] == 2
    item = finding(value)
    assert item["verdict"] == "unknown" and "completion status" in item["reason"], item
    assert value["action_verdict"] == "unknown"


def test_a_newer_run_without_a_result_appended_during_the_hook_asks(env, monkeypatch):
    engine, path = env
    _gate(engine, path)
    with growing(monkeypatch, path, [newer_run("toolu_check_2") + own_records()]):
        response = run(engine, path, "PreToolUse", tool_input={"command": PUSH})
    assert permission(response) == "ask", response


def test_a_newer_successful_run_in_the_appended_records_settles_clean(env, monkeypatch):
    engine, path = env
    _gate(engine, path)
    with growing(monkeypatch, path, [newer_run("toolu_check_2", failed=False) + own_records()]):
        value = check(engine, path)
    assert value["verdict"] == "clean" and value["coverage"]["complete"] is True


# --- (d) growth that does not stop within the budget -----------------------------------------------------------

def test_growth_that_never_stops_within_the_budget_stays_unknown(env, monkeypatch):
    engine, path = env
    _gate(engine, path)
    engine.settings.action_settle_sec = 25.0
    monkeypatch.setattr(engine_module, "SETTLE_CLOCK", StepClock(10.0))
    batches = [[env_record("mode", mode="normal")] for _ in range(12)]       # more than the budget allows
    with growing(monkeypatch, path, batches) as seen:
        value = check(engine, path)
    assert len(seen) == 3 and all(seen)          # three passes, each grew; the injector is far from exhausted
    assert value["settle"] == {**value["settle"], "passes": 3, "stopped_by": "budget"}
    assert value["coverage"]["complete"] is False and value["coverage"]["judged_prefix"] is True
    assert value["coverage"]["unread_bytes"] > 0
    item = finding(value)
    assert item["verdict"] == "unknown" and "incomplete current history" in item["reason"]
    assert value["action_verdict"] == "unknown"


def test_growth_that_never_stops_within_the_budget_still_asks(env, monkeypatch):
    engine, path = env
    _gate(engine, path)
    engine.settings.action_settle_sec = 25.0
    monkeypatch.setattr(engine_module, "SETTLE_CLOCK", StepClock(10.0))
    batches = [[env_record("mode", mode="normal")] for _ in range(12)]
    with growing(monkeypatch, path, batches) as seen:
        response = run(engine, path, "PreToolUse", tool_input={"command": PUSH})
    assert permission(response) == "ask" and "check-first" in response["hookSpecificOutput"]["permissionDecisionReason"]
    assert 2 <= len(seen) < len(batches)


def test_a_zero_budget_judges_once_as_before(env, monkeypatch):
    engine, path = env
    _gate(engine, path)
    engine.settings.action_settle_sec = 0.0
    with growing(monkeypatch, path, [own_records()]) as seen:
        value = check(engine, path)
    assert seen == [True]
    assert value["settle"]["passes"] == 1 and value["settle"]["stopped_by"] == "budget"
    assert finding(value)["verdict"] == "unknown" and value["coverage"]["judged_prefix"] is True


# --- (e) appended bytes that cannot be parsed or read ----------------------------------------------------------

def test_an_unparseable_appended_record_stays_unknown(env, monkeypatch):
    engine, path = env
    _gate(engine, path)
    # The call's own records and an unparseable line are appended while the first evaluation runs; the second reads
    # them: a parse error is not growth, so nothing is clean and no further pass is made.
    with growing(monkeypatch, path, [encoded(own_records()) + b"{not json}\n"]) as seen:
        value = check(engine, path)
    assert seen == [True, False]
    assert value["verdict"] == "unknown" and finding(value)["verdict"] == "unknown"
    assert "parse_errors" in value["coverage"]["issues"]
    assert value["coverage"].get("judged_prefix") is not True
    # The second evaluation saw no growth (it settled); what it read is incomplete, and that decides the verdict.
    assert value["settle"]["passes"] == 2 and value["settle"]["stopped_by"] == "no_growth"


def test_growth_beside_an_unparseable_record_is_not_reread(env, monkeypatch):
    """Growth is the only gap another read can close: with a parse error beside it, the check stops and stays unknown."""
    engine, path = env
    _gate(engine, path)
    batches = [encoded(own_records()) + b"{not json}\n", own_records(call_id="toolu_push_2")]
    with growing(monkeypatch, path, batches) as seen:
        value = check(engine, path)
    assert seen == [True, True]
    assert value["settle"]["passes"] == 2 and value["settle"]["stopped_by"] == "coverage_gap"
    assert {"parse_errors", "unread_bytes"} <= set(value["coverage"]["issues"])
    assert value["verdict"] == "unknown" and finding(value)["verdict"] == "unknown"


def test_an_unfinished_appended_record_stays_unknown_and_is_not_reread_forever(env, monkeypatch):
    engine, path = env
    _gate(engine, path)
    monkeypatch.setattr(engine_module, "SETTLE_CLOCK", StepClock(1.0))       # a budget of 5 s: plenty of passes left
    with growing(monkeypatch, path, [b'{"type": "assistant", "message": ']) as seen:
        value = check(engine, path)
    # The second evaluation finds the same unread tail and nothing new: another pass cannot read it either.
    assert len(seen) == 2
    assert value["settle"]["passes"] == 2 and value["settle"]["stopped_by"] == "no_progress"
    assert value["coverage"]["unread_bytes"] > 0 and value["coverage"]["complete"] is False
    assert finding(value)["verdict"] == "unknown"


# --- (f) no growth --------------------------------------------------------------------------------------------

def test_without_growth_the_check_evaluates_once(env, monkeypatch):
    engine, path = env
    _gate(engine, path)
    with growing(monkeypatch, path, []) as seen:
        value = check(engine, path)
    assert seen == [False]
    assert "settle" not in value
    assert value["verdict"] == "clean" and value["coverage"]["complete"] is True
    with growing(monkeypatch, path, []) as seen:
        assert run(engine, path, "PreToolUse", tool_input={"command": PUSH}) == {}


def test_an_action_that_does_not_depend_on_history_is_not_reevaluated(env, monkeypatch):
    """Only the action's findings that read history can be settled by reading more of it. A description-only contract
    applies to every action and is never machine-verifiable: its unknown is no reason to evaluate again."""
    engine, path = env
    engine.upsert_contracts([{"id": "no-force", "provider": "claude", "type": "must_not", "scope": "tool_call",
                              "regex": "--force"},
                             {"id": "prose", "provider": "claude", "type": "must_not", "description": "no verifier"}])
    with growing(monkeypatch, path, [own_records()]) as seen:
        value = check(engine, path)
    assert seen == [True] and value["settle"]["passes"] == 1 and value["settle"]["stopped_by"] == "decided"
    assert finding(value, "no-force")["verdict"] == "clean" and finding(value, "prose")["verdict"] == "unknown"


def test_a_violation_found_during_growth_is_not_reevaluated(env, monkeypatch):
    """A violation decides the action by itself; a second pass could not change that (and must not consume a
    one-time grant a second time)."""
    engine, path = env
    _gate(engine, path)
    engine.upsert_contracts([{"id": "no-force", "provider": "claude", "type": "must_not", "scope": "tool_call",
                              "regex": "--force"}])
    with growing(monkeypatch, path, [own_records(PUSH + " --force")]) as seen:
        value = check(engine, path, PUSH + " --force")
    assert seen == [True] and value["settle"]["stopped_by"] == "decided"
    assert value["action_verdict"] == "violation"


# --- the settle passes stay inside the request's own deadline ------------------------------------------------------

def test_the_budget_decides_whether_a_pass_starts_and_a_started_pass_finishes(env, monkeypatch):
    """The budget is checked before each pass; a pass that starts runs under the limits of the first one (a hook has no
    request deadline), so it finishes, and its answer is the check's."""
    engine, path = env
    _gate(engine, path)
    engine.settings.action_settle_sec = 10.0 + 1e-9
    monkeypatch.setattr(engine_module, "SETTLE_CLOCK", StepClock(10.0))   # a nanosecond of budget left for pass 2
    with growing(monkeypatch, path, [own_records()]) as seen:
        value = check(engine, path)
    assert seen == [True, False]
    assert value["settle"]["passes"] == 2 and value["settle"]["stopped_by"] == "no_growth"
    assert value["verdict"] == "clean" and value["coverage"]["complete"] is True


def test_a_settle_pass_runs_under_the_limits_of_the_first(env, monkeypatch):
    """No deadline of its own: under one, the comparison with the full evaluator that a hook runs in the request for a
    high-risk contract could be skipped as not fitting, and the settled answer would stand unverified."""
    from self_directing_mcp.request_control import current_request
    engine, path = env
    _gate(engine, path)
    deadlines = []

    with growing(monkeypatch, path, [own_records()]) as seen:
        audit = engine_module.SelfDirectEngine._audit          # the injector's wrapper

        def evaluation(self, *args, **kwargs):
            control = current_request()
            deadlines.append(None if control is None else control.deadline)
            return audit(self, *args, **kwargs)

        monkeypatch.setattr(engine_module.SelfDirectEngine, "_audit", evaluation)
        value = check(engine, path)
    assert value["settle"]["passes"] == 2 and seen == [True, False]
    assert deadlines and set(deadlines) == {None}


def test_a_pass_cut_by_the_requests_own_deadline_keeps_the_previous_answer(env, monkeypatch):
    """A pass that runs out of the request's time is not a failed request: the previous evaluation's answer stands."""
    engine, path = env
    _gate(engine, path)
    import time
    # The time the request keeps to return its answer leaves the second pass a nanosecond.
    monkeypatch.setattr(engine_module.incremental, "comparison_reserve", lambda remaining: remaining - 1e-9)
    with growing(monkeypatch, path, [own_records(), own_records(call_id="toolu_push_2")]) as seen:
        value = engine.check_action(SID, {"tool_name": "Bash", "arguments": {"command": PUSH}}, provider="claude",
                                    path=str(path), verify="fast", _deadline=time.monotonic() + 60)
    assert seen == [True, False]                 # the second pass started and was stopped before it judged anything
    assert value["settle"]["passes"] == 1 and value["settle"]["stopped_by"] == "deadline"
    assert finding(value)["verdict"] == "unknown" and value["coverage"]["judged_prefix"] is True


def test_a_settle_pass_stays_inside_the_requests_own_deadline(env, monkeypatch):
    """With a request deadline (an MCP call, the daemon), a pass never runs into it: the answer in hand is returned."""
    engine, path = env
    _gate(engine, path)
    import time
    with growing(monkeypatch, path, [own_records()]) as seen:
        value = engine.check_action(SID, {"tool_name": "Bash", "arguments": {"command": PUSH}}, provider="claude",
                                    path=str(path), verify="fast", _deadline=time.monotonic() + 60)
    assert value["settle"]["passes"] == 2 and value["verdict"] == "clean"
    monkeypatch.setattr(engine_module.incremental, "comparison_reserve", lambda remaining: remaining)
    with growing(monkeypatch, path, [own_records(call_id="toolu_push_2")]) as seen:
        value = engine.check_action(SID, {"tool_name": "Bash", "arguments": {"command": PUSH}}, provider="claude",
                                    path=str(path), verify="fast", _deadline=time.monotonic() + 60)
    assert seen == [True]                        # no room left before the request's deadline: no second pass
    assert value["settle"]["stopped_by"] == "deadline" and finding(value)["verdict"] == "unknown"


def test_the_settle_budget_is_a_declared_setting(monkeypatch):
    """A declared, finite, non-negative number of seconds (the hook settings parse it alike: test_hook_settings_parity)."""
    from self_directing_mcp.config import Settings
    from self_directing_mcp.settings_core import DEFAULT_ACTION_SETTLE_SEC
    monkeypatch.delenv("SELF_DIRECT_ACTION_SETTLE_SEC", raising=False)
    assert Settings(_env_file=None).action_settle_sec == DEFAULT_ACTION_SETTLE_SEC == 5.0
    monkeypatch.setenv("SELF_DIRECT_ACTION_SETTLE_SEC", "0")
    assert Settings(_env_file=None).action_settle_sec == 0.0
    for bad in ("-1", "inf", "nan", "soon"):
        monkeypatch.setenv("SELF_DIRECT_ACTION_SETTLE_SEC", bad)
        with pytest.raises(ValueError):
            Settings(_env_file=None)
