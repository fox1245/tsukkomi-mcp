"""B6(d): a call the host refused is a rejected proposal, not an executed violation.

When a hook (or the user's permission prompt) denies a tool call, the transcript still holds the ``tool_use``. A later
``must_not`` audit used to report it like an executed action. Claude Code records its decision on the linked tool
result (``permissionDecision``, kept as ``meta.permission_decision``); only that host record counts: a result that
merely says "error" is still an attempt. A refused call is reported apart (``proposals_rejected``), never makes the
verdict a violation, and leaves the verdict to the remaining evidence. Every evaluator case runs through both
evaluators (``semantic_support.evaluate``).
"""
from __future__ import annotations

import json

import pytest

from self_directing_mcp.claude_hooks import run_hook
from self_directing_mcp.audit import runner
from semantic_support import (
    SID, Claude, assistant, bash, call, denied_result, evaluate, event, finding, human, one, result, tool_result,
    use, user_turn,
)

RM = {"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"}


def denied(call_id, decision="deny", **meta):
    """The host's refusal: an error result that carries the permission decision."""
    return result(call_id, False, "blocked by a hook", permission_decision=decision, **meta)


# ---------------------------------------------------------------- both evaluators

@pytest.mark.parametrize("decision", ["deny", {"behavior": "deny"}, {"behavior": "deny", "message": "hook"}])
def test_a_refused_call_is_clean_and_reported_as_a_rejected_proposal(decision):
    first = call("rm -rf /x", "c1")
    refusal = denied("c1", decision)
    item = one(RM, [user_turn(), first, refusal])
    assert item.verdict == "clean" and item.reason.startswith("no executed forbidden match")
    assert item.reason == runner.rejected_clean(1) and item.evidence == []
    (entry,) = item.proposals_rejected
    assert entry["contract_id"] == "no-rm" and entry["event_id"] == first.chunk_id and entry["call_id"] == "c1"
    assert entry["denied_by"] == refusal.chunk_id and entry["reason"].startswith("host_denied:")
    assert entry["at_byte"] == first.key[0]


def test_an_executed_call_is_still_a_violation_and_reports_nothing_rejected():
    item = one(RM, [user_turn(), call("rm -rf /x", "c1"), result("c1", True)])
    assert item.verdict == "violation" and item.proposals_rejected == []
    assert "proposals_rejected" not in item.model_dump(mode="json")        # omitted while empty


@pytest.mark.parametrize("outcome", [
    dict(success=False, text_="error: something went wrong"),
    dict(success=None, text_="Exit code 1"),
    dict(success=False, text_="blocked by a hook", permission_decision="allow"),
    dict(success=False, text_="blocked by a hook", permission_decision={"behavior": "ask"}),
    dict(success=False, text_="blocked by a hook", permission_decision="denied"),      # not the host's word for it
    dict(success=False, text_="blocked by a hook", permission_decision=True),
])
def test_an_error_or_a_non_deny_decision_is_still_an_attempt(outcome):
    events = [user_turn(), call("rm -rf /x", "c1"), result("c1", **outcome)]
    assert one(RM, events).verdict == "violation"


def test_the_verdict_is_decided_by_the_remaining_evidence():
    refused, executed = call("rm -rf /refused", "c1"), call("rm -rf /executed", "c2")
    events = [user_turn(), refused, denied("c1"), executed, result("c2", True)]
    item = one(RM, events)
    assert item.verdict == "violation" and item.reason.startswith("forbidden regex matched 1 event(s)")
    assert [e.chunk_id for e in item.evidence] == [executed.chunk_id]
    assert [p["event_id"] for p in item.proposals_rejected] == [refused.chunk_id]


def test_a_shared_call_id_vouches_for_no_call():
    """Two calls with one id: the refusal cannot be tied to either, so both stay attempts."""
    events = [user_turn(), call("rm -rf /a", "c1"), call("rm -rf /b", "c1"), denied("c1")]
    assert one(RM, events).verdict == "violation"
    late = [user_turn(), denied("c1"), call("rm -rf /a", "c1")]                      # a result cannot refuse a later call
    assert one(RM, late).verdict == "violation"


def test_a_refusal_for_another_call_changes_nothing():
    events = [user_turn(), call("rm -rf /a", "c1"), call("ls", "c2"), denied("c2")]
    assert one(RM, events).verdict == "violation"


def test_the_exception_covers_only_what_ran():
    rule = {**RM, "exception_regex": "tmp"}
    refused_exception = [user_turn(), call("rm -rf /tmp/x", "c1"), denied("c1"), call("rm -rf /data", "c2"),
                         result("c2", True)]
    # The call the exception would have covered never ran, so it cannot waive the one that did.
    assert one(rule, refused_exception).verdict == "violation"
    ran_exception = [user_turn(), call("rm -rf /data", "c1"), denied("c1"), call("rm -rf /tmp/x", "c2"),
                     result("c2", True)]
    assert one(rule, ran_exception).verdict == "suspicious"


def test_a_window_start_hides_older_refusals_too():
    old, anchor = call("rm -rf /old", "c1"), result("c9", True)
    events = [user_turn(), old, denied("c1"), anchor, call("rm -rf /new", "c2"), denied("c2")]
    item = one({**RM, "applies_from_event_id": anchor.chunk_id}, events)
    assert item.verdict == "clean" and [p["call_id"] for p in item.proposals_rejected] == ["c2"]


def test_many_refused_calls_are_counted_and_reported_up_to_the_limit():
    events = [user_turn()]
    for number in range(30):
        events += [call(f"rm -rf /x{number}", f"c{number}"), denied(f"c{number}")]
    item = one(RM, events)
    assert item.verdict == "clean" and item.reason == runner.rejected_clean(30)
    assert len(item.proposals_rejected) == runner.REJECTED_LIMIT
    mixed = events + [call("rm -rf /real", "real"), result("real", True)]
    again = one(RM, mixed)
    assert again.verdict == "violation" and [e.snippet for e in again.evidence][0].endswith('/real"}')


def test_many_executed_calls_with_some_refused_still_list_twenty_of_them():
    events = [user_turn()]
    for number in range(40):
        events += [call(f"rm -rf /x{number}", f"c{number}"), denied(f"c{number}") if number % 3 == 0
                   else result(f"c{number}", True)]
    item = one(RM, events)
    assert item.verdict == "violation" and len(item.evidence) == 20
    assert item.reason.startswith(f"forbidden regex matched {sum(1 for n in range(40) if n % 3)} event(s)")


@pytest.mark.parametrize("provider", ["codex", "omp", "agy"])
def test_providers_without_a_host_record_do_not_read_one(provider):
    """Only Claude Code records the decision on the result; for the others the same bytes mean nothing."""
    events = [user_turn(), call("rm -rf /x", "c1"), denied("c1")]
    item = one(RM, events, provider)
    assert item.verdict == "violation" and item.proposals_rejected == []


def test_a_refused_call_still_satisfies_nothing_for_a_completion_obligation():
    """Only prohibitions are changed here; a ``must`` keeps its own rules (a failed result is not success)."""
    rule = {"id": "ran", "type": "must", "scope": "tool_call", "regex": "pytest", "requires_success": True}
    item = one(rule, [user_turn(), call("pytest", "c1"), denied("c1")])
    assert item.verdict == "violation" and item.reason == "required calls completed without success"


def test_a_result_that_echoes_the_command_still_matches_when_a_rule_reads_results():
    """Only the call is the proposal: a rule that also reads tool output keeps seeing that output."""
    rule = {**RM, "scope": "any", "roles": ["assistant", "tool"]}
    events = [user_turn(), call("rm -rf /x", "c1"), result("c1", False, "refused: rm -rf /x",
                                                          permission_decision="deny")]
    item = one(rule, events)
    assert item.verdict == "violation" and [p["call_id"] for p in item.proposals_rejected] == ["c1"]


def test_a_proposed_action_is_unaffected():
    proposed = event("tool_call", "[tool_call:Bash] " + json.dumps({"command": "rm -rf /x"}), "assistant",
                     tool_name="Bash", proposed=True)
    history = [user_turn(), call("rm -rf /y", "c1"), denied("c1")]
    item = one(RM, history, proposed=proposed)
    assert item.verdict == "violation" and item.proposals_rejected == []


def test_both_evaluators_agree_in_every_feed_with_refusals_arriving_late():
    events = [user_turn(), call("rm -rf /a", "c1"), call("ls", "c2"), result("c2", True), denied("c1"),
              call("rm -rf /b", "c3"), result("c3", True), call("rm -rf /c", "c4"), denied("c4", {"behavior": "deny"})]
    rules = [RM, {**RM, "id": "exc", "exception_regex": "/b"}, {**RM, "id": "any", "scope": "any"},
             {**RM, "id": "later", "applies_from_event_id": events[2].chunk_id}]
    findings = evaluate(rules, events, chunk_sizes=(None, 1, 2, 3, 5))
    assert [f.verdict for f in findings] == ["violation", "suspicious", "violation", "violation"]
    assert [len(f.proposals_rejected) for f in findings] == [2, 2, 2, 1]


# ---------------------------------------------------------------- end to end: parsing, the audit and the Stop hook

@pytest.fixture
def session(tmp_path):
    world = Claude(tmp_path)
    yield world
    world.close()


def stop(session):
    payload = {"session_id": SID, "transcript_path": str(session.path), "cwd": "/synthetic/project",
               "hook_event_name": "Stop", "stop_hook_active": False}
    return run_hook("Stop", payload, engine=session.engine)


def test_the_parser_keeps_the_hosts_decision_on_the_result(session):
    session.write(human("clean up"), bash("t1", "rm -rf /x"), denied_result("t1"))
    session.sync()
    (outcome,) = [c for c in session.chunks() if c.kind == "tool_result"]
    assert outcome.meta["permission_decision"] == "deny" and outcome.meta["success"] is False


def test_end_to_end_a_denied_rm_rf_is_reported_as_rejected_not_as_a_violation(session):
    session.write(human("never rm -rf"), bash("t1", "rm -rf /x"), denied_result("t1"))
    session.upsert(RM)
    audit = session.audit()
    item = finding(audit, "no-rm")
    assert item["verdict"] == "clean" and audit["verdict"] == "clean"
    assert item["proposals_rejected"][0]["call_id"] == "t1"
    assert "executed" in item["reason"] or "rejected" in item["reason"]


def test_end_to_end_the_stop_note_tells_a_rejected_proposal_from_an_executed_violation(session):
    session.write(human("never rm -rf"), bash("t1", "rm -rf /x"), denied_result("t1"))
    session.upsert(RM)
    note = stop(session)["systemMessage"]
    assert "Rejected by the host, not executed" in note and "rejected proposals, not executed violations" in note
    assert "forbidden regex matched" not in note and "Not verified" not in note

    session.append(bash("t2", "rm -rf /y"), tool_result("t2", "removed"))
    both = stop(session)["systemMessage"]
    assert "forbidden regex matched 1 event(s)" in both                       # the executed one is a violation
    assert "rejected proposals, not executed violations" in both               # ...and the refused one is apart


def test_end_to_end_an_executed_rm_rf_is_a_violation_with_no_rejection_note(session):
    session.write(human("never rm -rf"), bash("t1", "rm -rf /x"), tool_result("t1", "removed"))
    session.upsert(RM)
    item = finding(session.audit(), "no-rm")
    assert item["verdict"] == "violation" and "proposals_rejected" not in item
    note = stop(session)["systemMessage"]
    assert "forbidden regex matched" in note and "Rejected by the host" not in note


def test_end_to_end_a_plain_error_result_is_still_an_executed_attempt(session):
    session.write(human("never rm -rf"), bash("t1", "rm -rf /x"), tool_result("t1", "boom", is_error=True))
    session.upsert(RM)
    assert finding(session.audit(), "no-rm")["verdict"] == "violation"


def test_end_to_end_a_refusal_that_arrives_after_the_first_audit(session):
    """Incremental state: the refusal is a later append, and the verdict must follow it (and only it)."""
    session.write(human("never rm -rf"), bash("t1", "rm -rf /x"))
    session.upsert(RM)
    assert finding(session.audit(), "no-rm")["verdict"] == "violation"        # no decision recorded yet
    session.append(denied_result("t1"))
    item = finding(session.audit(), "no-rm")
    assert item["verdict"] == "clean" and item["proposals_rejected"][0]["call_id"] == "t1"
    session.append(bash("t2", "rm -rf /y"), tool_result("t2", "removed"))
    again = finding(session.audit(), "no-rm")
    assert again["verdict"] == "violation" and len(again["proposals_rejected"]) == 1


def test_the_hook_still_denies_the_next_proposal_exactly_as_before(session):
    session.write(human("never rm -rf"), bash("t1", "rm -rf /x"), denied_result("t1"))
    session.upsert(RM)
    payload = {"session_id": SID, "transcript_path": str(session.path), "cwd": "/synthetic/project",
               "hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "rm -rf /again"}}
    assert run_hook("PreToolUse", payload, engine=session.engine)["hookSpecificOutput"]["permissionDecision"] == "deny"


# ---------------------------------------------------------------- AGY: the Pre hook's own denial of a model proposal

def proposal(command, step):
    return event("meta", "[proposed_tool_call:shell] " + json.dumps({"command": command}), "runtime",
                 event_type="PLANNER_RESPONSE", step_index=step, status="DONE", evidence_origin="model_proposal",
                 tool_name="shell")


def hook_denial(command, step):
    return event("meta", "[tool_denied] tool call denied by pre-tool hook: policy", "runtime", event_type="GENERIC",
                 step_index=step, status="ERROR", evidence_origin="host_rejected_proposal",
                 denied_action={"name": "shell", "args": {"command": command}})


def test_agy_a_proposal_the_pre_hook_denied_is_reported_apart_and_never_an_attempt():
    refused = proposal("rm -rf /x", 1)
    refusal = hook_denial("rm -rf /x", 2)
    item = one(RM, [user_turn(), refused, refusal], "agy")
    assert item.verdict == "clean" and item.reason == "no forbidden match in inspected events"
    (entry,) = item.proposals_rejected
    assert entry == {"contract_id": "no-rm", "reason": runner.REJECTED_PROPOSAL_REASON, "tool_name": "shell",
                     "at_byte": refused.key[0], "denied_at_byte": refusal.key[0]}


def test_agy_an_undenied_proposal_is_still_an_attempt_beside_a_denied_one():
    events = [user_turn(), proposal("rm -rf /a", 1), hook_denial("rm -rf /a", 2), proposal("rm -rf /b", 3)]
    item = one(RM, events, "agy")
    assert item.verdict == "violation" and item.reason.startswith("forbidden regex matched 1 host-logged call attempt")
    assert [p["tool_name"] for p in item.proposals_rejected] == ["shell"] and len(item.proposals_rejected) == 1


def test_agy_denials_arriving_in_a_later_feed_are_matched_to_the_earlier_proposal():
    events = [user_turn(), proposal("rm -rf /a", 1), proposal("rm -rf /b", 3), hook_denial("rm -rf /b", 4),
              hook_denial("rm -rf /a", 5)]
    item = one({**RM, "exception_regex": "/b"}, events, "agy", chunk_sizes=(None, 1, 2))
    assert item.verdict == "clean" and len(item.proposals_rejected) == 2


def test_agy_an_unlinked_denial_reports_nothing():
    item = one(RM, [user_turn(), proposal("rm -rf /a", 1), hook_denial("curl x", 2)], "agy")
    assert item.verdict == "violation" and item.proposals_rejected == []


def test_agy_end_to_end_a_denied_typed_call_is_reported_as_rejected(tmp_path):
    from test_agy_parser_v6 import SID as AGY, append, finding as agy_finding, hook, make, planner, sync, user as agy_user
    from test_agy_parser_v7 import rec
    from self_directing_mcp.config import Settings
    from self_directing_mcp.engine import SelfDirectEngine
    root = tmp_path / "app"
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=tmp_path / "index",
                                       agy_app_data_dirs=[root], openrouter_api_key_file=None))
    try:
        engine.upsert_contracts([{"id": "no-rm", "provider": "agy", "type": "must_not", "scope": "tool_call",
                                  "regex": "rm -rf"}])
        path = make(root, [agy_user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "rm -rf /data"})])])
        engine.record_agy_hook(hook(path, 2, "rm -rf /data"), "PreToolUse")
        append(path, rec(2, "RUN_COMMAND", "MODEL", "ERROR", content="",
                         error="tool call denied by pre-tool hook: not allowed"))
        sync(engine, path)
        item = agy_finding(engine.audit_session(AGY, provider="agy", path=str(path)), "no-rm")
        assert item["verdict"] == "clean"
        assert [p["tool_name"] for p in item["proposals_rejected"]] == ["run_command"]
    finally:
        engine.close()
