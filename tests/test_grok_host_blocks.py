"""``audit.hostblocks``: the join of Grok's note ("a hook blocked this tool") and a call's failed result.

The class is the whole rule and both evaluators use it, so these tests feed it slim events directly (the shape
``capture_audit_events`` returns) and also run every case through ``semantic_support.evaluate``, which checks the
reference evaluator against the incremental monitors fed in pieces, with their state round-tripped through JSON as
the store does. The recorded shapes and the end-to-end audits are in ``test_grok_refused.py``.
"""
from __future__ import annotations

import json
import random

import pytest

from self_directing_mcp.audit import hostblocks, runner
from self_directing_mcp.audit import incremental as inc
from self_directing_mcp.schemas import ContractRule
from incremental_support import Session
from semantic_support import call, evaluate, event, user_turn

RULE = {"id": "no-read", "type": "must_not", "scope": "tool_call", "regex": r"secret"}
SOURCE = "project/tsukkomi"


def tool_call(call_id, tool="read_file", text="secret.env"):
    return event("tool_call", f"[tool_call:Read] {text}", "assistant", call_id=call_id, tool_name="Read",
                 native_tool_name=tool)


def failed(call_id, number, **meta):
    return event("tool_result", "[tool_result] Hook denied", "tool", call_id=call_id, success=False,
                 tool_status="failed", host_event_id=number, **meta)


def noted(number, tool="read_file", source=SOURCE, **meta):
    fields = {"evidence_origin": "host_tool_outcome", "host_event_id": number, "blocked_tool": tool, "denied_by": source}
    return event("meta", "[host_tool_outcome] blocked", "runtime", **{**fields, **meta})


def blocked(events):
    blocks = hostblocks.HostBlocks()
    for item in events:
        blocks.feed(item)
    return {call: tuple(denial) for call, denial in blocks.rejected.items()}


def verdict(events, provider="grok"):
    """The finding of RULE with both evaluators agreeing: (verdict, [(call id, denied_by)])."""
    item = evaluate([RULE], [user_turn(), *events], provider, chunk_sizes=(None, 1, 2, 3))[0]
    return item.verdict, [(entry["call_id"], entry["denied_by"]) for entry in item.proposals_rejected]


ATTEMPT = ("violation", [])
BLOCKED = ("clean", [("c1", SOURCE)])


# ---------------------------------------------------------------- the join

def test_a_note_and_a_failed_result_one_apart_for_the_same_tool_block_the_call():
    first = tool_call("c1")
    note = noted(13)
    assert blocked([note, first, failed("c1", 12)]) == {first.chunk_id: (note.chunk_id, SOURCE)}
    assert blocked([first, failed("c1", 12), note]) == {first.chunk_id: (note.chunk_id, SOURCE)}


@pytest.mark.parametrize("order", [
    lambda: [noted(13), tool_call("c1"), failed("c1", 12)],
    lambda: [tool_call("c1"), noted(13), failed("c1", 12)],
    lambda: [tool_call("c1"), failed("c1", 12), noted(13)],
], ids=["note, call, result", "call, note, result", "call, result, note"])
def test_every_order_of_the_three_records_gives_the_same_block(order):
    assert verdict(order()) == BLOCKED


@pytest.mark.parametrize("events", [
    lambda: [tool_call("c1"), failed("c1", 12)],                                                   # no note
    lambda: [noted(13), tool_call("c1")],                                                          # no result
    lambda: [noted(13, tool="write"), tool_call("c1"), failed("c1", 12)],                          # another tool
    lambda: [noted(14), tool_call("c1"), failed("c1", 12)],                                        # two apart
    lambda: [noted(12), tool_call("c1"), failed("c1", 12)],                                        # the same number
    lambda: [noted(11), tool_call("c1"), failed("c1", 12)],                                        # before the result
    lambda: [noted(13), tool_call("c1"), event("tool_result", "[tool_result] ok", "tool", call_id="c1", success=True,
                                                tool_status="completed", host_event_id=12)],       # the call ran
    lambda: [noted(13), tool_call("c1"), event("tool_result", "[tool_result] ?", "tool", call_id="c1",
                                                tool_status="in_progress", host_event_id=12)],
    lambda: [noted(13), tool_call("c1"), event("tool_result", "[tool_result] x", "tool", call_id="c1",
                                                success=False, host_event_id=12)],                 # no status
    lambda: [noted(13), tool_call("c1"), event("tool_result", "[tool_result] x", "tool", call_id="c1",
                                                success=False, tool_status="failed")],             # no number
    lambda: [failed("c1", 12), tool_call("c1"), noted(13)],                                        # the call comes later
    lambda: [noted(13), tool_call("c1", tool=None), failed("c1", 12)],                             # a call of no known tool
    lambda: [noted(13), tool_call("c1"), failed("c2", 12)],                                        # another call's result
], ids=["no-note", "no-result", "other-tool", "two-apart", "same-number", "note-before", "call-ran", "progress",
        "no-status", "no-number", "call-later", "no-tool", "other-call"])
def test_anything_less_than_the_whole_pair_is_an_attempt(events):
    assert verdict(events()) == ATTEMPT


def test_a_result_with_no_call_before_it_blocks_nothing():
    assert blocked([noted(13), failed("c1", 12)]) == {} and blocked([failed("c1", 12), noted(13)]) == {}
    assert verdict([noted(13), failed("c1", 12)]) == ("clean", [])           # nothing to match: nothing to report


@pytest.mark.parametrize("fields", [
    {"host_event_id": None}, {"host_event_id": "13"}, {"host_event_id": True}, {"host_event_id": 13.0},
    {"blocked_tool": ""}, {"blocked_tool": None}, {"blocked_tool": 7},
    {"denied_by": ""}, {"denied_by": None}, {"denied_by": ["project"]},
])
def test_a_note_without_a_number_a_tool_or_a_source_is_nothing(fields):
    base = {"evidence_origin": "host_tool_outcome", "host_event_id": 13, "blocked_tool": "read_file",
            "denied_by": SOURCE}
    odd = event("meta", "[host_tool_outcome] blocked", "runtime", **{**base, **fields})
    assert verdict([odd, tool_call("c1"), failed("c1", 12)]) == ATTEMPT


def test_a_boolean_is_not_a_number():
    assert verdict([noted(True), tool_call("c1"), failed("c1", 0)]) == ATTEMPT
    assert verdict([noted(13), tool_call("c1"), failed("c1", False)]) == ATTEMPT


def test_a_note_that_is_not_the_hosts_origin_is_nothing():
    other = noted(13, evidence_origin="model_proposal")
    assert verdict([other, tool_call("c1"), failed("c1", 12)]) == ATTEMPT
    wrong_kind = event("turn", "[host_tool_outcome] blocked", "runtime", evidence_origin="host_tool_outcome",
                       host_event_id=13, blocked_tool="read_file", denied_by=SOURCE)
    assert verdict([wrong_kind, tool_call("c1"), failed("c1", 12)]) == ATTEMPT


def test_two_calls_blocked_one_after_the_other_are_each_told_by_their_own_numbers():
    both = [noted(13), tool_call("c1"), failed("c1", 12), noted(20, tool="write"), tool_call("c2", "write"),
            failed("c2", 19)]
    assert len(blocked(both)) == 2 and {v[1] for v in blocked(both).values()} == {SOURCE}
    assert verdict(both) == ("clean", [("c1", SOURCE), ("c2", SOURCE)])
    # a third call that failed with a number nobody noted is an attempt beside them
    assert verdict([*both, tool_call("c3"), failed("c3", 30)]) == ("violation", [("c1", SOURCE), ("c2", SOURCE)])


def test_two_notes_or_two_results_with_one_number_join_nobody():
    assert verdict([noted(13), noted(13, source="another"), tool_call("c1"), failed("c1", 12)]) == ATTEMPT
    assert verdict([tool_call("c1"), tool_call("c2"), failed("c1", 12), failed("c2", 12), noted(13)]) == ATTEMPT


def test_a_call_with_two_results_is_no_block_and_a_block_already_made_is_taken_back():
    ran = lambda: event("tool_result", "[tool_result] ok", "tool", call_id="c1", success=True, tool_status="completed")
    assert verdict([noted(13), tool_call("c1"), failed("c1", 12), ran()]) == ATTEMPT
    assert verdict([noted(13), tool_call("c1"), ran(), failed("c1", 12)]) == ATTEMPT
    assert verdict([tool_call("c1"), failed("c1", 12), ran(), noted(13)]) == ATTEMPT
    assert verdict([noted(13), tool_call("c1"), failed("c1", 12), failed("c1", 12)]) == ATTEMPT


def test_a_call_id_that_two_calls_use_vouches_for_neither():
    assert verdict([noted(13), tool_call("c1"), tool_call("c1"), failed("c1", 12)]) == ATTEMPT
    assert verdict([tool_call("c1"), event("tool_result", "[tool_result] ok", "tool", call_id="c1", success=True,
                                           tool_status="completed"),
                    tool_call("c1"), noted(13), failed("c1", 12)]) == ATTEMPT


def test_a_note_and_its_result_must_lie_within_the_window_of_each_other():
    filler = lambda count: [event("meta", "[meta]", "runtime") for _ in range(count)]
    near = [noted(13), tool_call("c1"), *filler(hostblocks.WINDOW - 2), failed("c1", 12)]       # WINDOW events apart
    far = [noted(13), tool_call("c1"), *filler(hostblocks.WINDOW - 1), failed("c1", 12)]
    assert verdict(near) == BLOCKED and verdict(far) == ATTEMPT
    assert verdict([tool_call("c1"), failed("c1", 12), *filler(hostblocks.WINDOW - 1), noted(13)]) == BLOCKED
    assert verdict([tool_call("c1"), failed("c1", 12), *filler(hostblocks.WINDOW), noted(13)]) == ATTEMPT


def test_the_waiting_tables_are_bounded_and_forget_the_oldest_first():
    many = hostblocks.PENDING + 6
    notes = [noted(1000 + 10 * n, tool="read_file") for n in range(many)]
    first = [tool_call("a"), failed("a", 999)]                  # the oldest note: gone from the table
    last = [tool_call("b"), failed("b", 1000 + 10 * (many - 1) - 1)]
    assert blocked([*notes, *first]) == {} and len(blocked([*notes, *last])) == 1
    calls = [tool_call(f"x{n}") for n in range(hostblocks.CALLS + 5)]
    assert blocked([noted(13), *calls, failed("x0", 12)]) == {}                       # its call left the window
    assert len(blocked([noted(13), *calls, failed(f"x{hostblocks.CALLS + 4}", 12)])) == 1    # a call still in it
    assert len(blocked([noted(13), *calls[:3], failed("x1", 12)])) == 1


# ---------------------------------------------------------------- other hosts do not read it

@pytest.mark.parametrize("provider", ["claude", "codex", "omp", "agy"])
def test_other_providers_never_read_the_note(provider):
    events = [noted(13), tool_call("c1"), failed("c1", 12)]
    assert verdict(events, provider) == ATTEMPT


def test_grok_does_not_read_claudes_permission_decision():
    refusal = event("tool_result", "[tool_result] no", "tool", call_id="c1", success=False,
                    permission_decision={"decision": "reject", "source": "hook"})
    assert verdict([tool_call("c1"), refusal]) == ATTEMPT


def test_a_proposed_action_is_judged_alone():
    item = evaluate([RULE], [user_turn(), noted(13), tool_call("c1"), failed("c1", 12)], "grok",
                    proposed=call("cat secret.env", "p1"))[0]
    assert item.verdict == "violation" and item.proposals_rejected == []


# ---------------------------------------------------------------- the state is the same wherever it is cut

@pytest.mark.parametrize("seed", range(40))
def test_dumping_and_loading_anywhere_changes_nothing(seed):
    rng = random.Random(f"blocks/{seed}")
    session = Session("grok", rng)
    session.extend(120)
    whole = hostblocks.HostBlocks()
    for item in session.events:
        whole.feed(item)
    cut = hostblocks.HostBlocks()
    for item in session.events:
        cut.feed(item)
        if rng.random() < 0.3:
            fresh = hostblocks.HostBlocks()
            fresh.load(json.loads(json.dumps(cut.dump())))
            cut = fresh
    assert cut.dump() == whole.dump() and whole.rejected == cut.rejected
    assert runner.blocked_calls(session.events) == {c: tuple(d) for c, d in whole.rejected.items()}


def test_the_generated_sessions_do_block_calls_and_leave_others():
    """The random sessions the differential tests use are not all attempts (or all blocks)."""
    blocked_sessions = mixed = 0
    for seed in range(60):
        session = Session("grok", random.Random(f"blocks/{seed}"))
        session.extend(120)
        found = runner.blocked_calls(session.events)
        calls = [e for e in session.events if e.kind == "tool_call"]
        blocked_sessions += bool(found)
        mixed += bool(found) and len(found) < len(calls)
    assert blocked_sessions >= 30 and mixed >= 30


def test_a_table_over_its_cap_defers_to_the_reference(monkeypatch):
    monkeypatch.setattr(inc, "REJECTED_MAX", 2)
    events = [user_turn()]
    for number in range(3):
        events += [noted(10 * number + 3), tool_call(f"c{number}"), failed(f"c{number}", 10 * number + 2)]
    rule = ContractRule(**RULE)
    monitor, facts = inc.RuleMonitor(rule, "grok"), inc.make_facts("grok", [], [rule])
    inc.fold(events, [(monitor, 0)], (facts, 0), inc.Pools())
    assert facts.overflow == "blocked call table is full"
    with pytest.raises(inc.NeedFull):
        monitor.finding(None, facts)
    # the reference has no cap: it still answers
    full = runner.full_findings([rule], events, session_id="s", store=None, provider="grok", proposed=None, coverage={})
    assert full[0].verdict == "clean" and len(full[0].proposals_rejected) == 3
