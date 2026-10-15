"""Regression tests for the review of the adaptive shadow comparison (issue #18, R2).

Each test names the defect it keeps out. Everything is synthetic: the harness (``Lab``, ``lab``, ``draw``) is the one of
``test_shadow_schedule``; the idle worker is also driven with a stub engine and a clock the test moves, so that no test waits
for real time. The tests that provoke a mismatch on purpose carry ``allow_incremental_mismatch``.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from self_directing_mcp import engine as engine_module
from self_directing_mcp.audit import incremental as inc
from self_directing_mcp.audit import shadow
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index import shadow_state as shadow_sql
from self_directing_mcp.request_control import IndexBusy, RequestStopped, check_request, current_request
from self_directing_mcp.schemas import AuditResult
from self_directing_mcp.shadow_worker import BACKOFF_MAX_SEC, BACKOFF_SEC, ShadowWorker
from test_reliability import SID, call
from test_shadow_schedule import HIGH, RULES, Lab, draw, fresh, lab, settle_in, worker_for  # noqa: F401  (fixtures)

LINT = {"id": "lint-first", "type": "must", "scope": "tool_call", "regex": "lint", "before_regex": "deploy"}
DEPLOY = {"tool_name": "shell", "arguments": {"command": "deploy now"}}


def finding(payload, contract_id):
    return next(item for item in payload["findings"] if item["contract_id"] == contract_id)


def blind(monkeypatch, contract_id="no-deploy"):
    """A defect of the incremental path: the monitor of ``contract_id`` ignores forbidden calls, so its answers stay clean."""
    real = inc.RuleMonitor._forbidden
    monkeypatch.setattr(inc.RuleMonitor, "_forbidden",
                        lambda self, event: None if self.rule.id == contract_id else real(self, event))


class SlowReference:
    """The full evaluator (the reference of a comparison) takes as long as the request lets it; the incremental pass is
    not touched. Off until the test switches it on, so the set-up of a test runs at full speed."""

    def __init__(self, monkeypatch):
        self.on, self.calls = False, 0
        real = engine_module.run_audit

        def slow(**kwargs):
            if self.on and kwargs.get("plan") is None:
                self.calls += 1
                end = time.monotonic() + 5
                while time.monotonic() < end:
                    check_request()                     # cooperative, like every evaluation
                    time.sleep(0.005)
            return real(**kwargs)
        monkeypatch.setattr(engine_module, "run_audit", slow)


@pytest.fixture
def slow_reference(monkeypatch):
    return SlowReference(monkeypatch)


# --- the memo of a forced comparison (finding 1) ----------------------------------------------------------------------

def test_the_memo_covers_only_the_same_input_and_every_contract_asked_for():
    clean, bad = AuditResult(session_id="s", verdict="clean"), AuditResult(session_id="s", verdict="violation")
    key = ("generation", 1, 5)
    state = shadow.settle(fresh(), outcome="match", full_scope=False, key="audit", flagged=set(), kept=set(), at="t",
                          offset=1, state_key=key, result=clean, memo_ids={"a", "b"})
    assert state.memo["ids"] == ["a", "b"] and state.memo["request"] == "audit"
    assert shadow.memo_covers(state, key, "audit", clean)                              # no contract named: the input alone
    assert shadow.memo_covers(state, key, "audit", clean, {"a"}) and shadow.memo_covers(state, key, "audit", clean, {"a", "b"})
    assert not shadow.memo_covers(state, key, "audit", clean, {"a", "c"})              # c was never compared
    assert not shadow.memo_covers(state, ("generation", 1, 6), "audit", clean, {"a"})      # other stored events
    assert not shadow.memo_covers(state, key, "check:x", clean, {"a"})                 # another kind of request
    assert not shadow.memo_covers(state, key, "audit", bad, {"a"})                     # another answer
    assert not shadow.memo_covers(fresh(), key, "audit", clean, {"a"})                 # nothing remembered
    old = state.copy()
    old.memo.pop("ids")                                                                # a memo that names no contract
    assert not shadow.memo_covers(old, key, "audit", clean, {"a"}) and shadow.memo_covers(old, key, "audit", clean)
    unmemoized = shadow.settle(fresh(), outcome="match", full_scope=False, key="audit", flagged=set(), kept=set(), at="t",
                               offset=1)
    assert unmemoized.memo is None                                                      # no input given: nothing remembered


def faulty_lint(monkeypatch):
    """A wrong block: the incremental path says ``violation`` for a prerequisite that is met."""
    real = inc.RuleMonitor.finding

    def faulty(self, proposed, facts):
        item = real(self, proposed, facts)
        if self.rule.id == "lint-first" and item.verdict == "clean":
            return item.model_copy(update={"verdict": "violation", "reason": "injected", "evidence": []})
        return item
    monkeypatch.setattr(inc.RuleMonitor, "finding", faulty)


def lint_lab(lab, draw):
    draw.value = False
    box = lab(every="adaptive", rules=[*RULES, HIGH, LINT])
    settle_in(box, checks=True)
    box.append(call("lint .", "lint1"))                    # lint ran, so lint-first is truly met
    return box


@pytest.mark.allow_incremental_mismatch
def test_a_late_comparison_that_did_not_run_is_not_remembered_as_done(lab, draw, monkeypatch):
    """The scoped comparison of the high-risk contract ran; the one a violation asked for had no time. The next identical
    request must not be told that this input was verified."""
    box = lint_lab(lab, draw)
    faulty_lint(monkeypatch)
    real, calls = inc._shadow_affordable, []

    def once(events):
        calls.append(1)
        return len(calls) == 1                              # the high-risk comparison fits, the forced one does not
    monkeypatch.setattr(inc, "_shadow_affordable", once)
    remembered = box.engine.store.load_shadow_state(SID, "codex").memo                  # (of an earlier request, not this one)
    first = box.engine.check_action(SID, DEPLOY, provider="codex")
    report = first["incremental"]
    assert report["shadow_compared"] and report["shadow_scope"] == ["tests-first"] and report["shadow_skipped"] == "deadline"
    assert finding(first, "lint-first")["verdict"] == "violation"                       # the wrong block, unverified
    assert box.engine.store.load_shadow_state(SID, "codex").memo == remembered          # this input is not remembered as verified
    monkeypatch.setattr(inc, "_shadow_affordable", real)
    second = box.engine.check_action(SID, DEPLOY, provider="codex")
    assert second["incremental"]["incremental_mismatch"] == ["lint-first"]
    assert "shadow_skipped" not in second["incremental"]
    assert finding(second, "lint-first")["verdict"] == "clean"                          # the full evaluator's answer


def test_a_forced_comparison_that_matched_is_remembered_with_the_contracts_it_covered(lab, draw):
    box = lab(every="adaptive")
    settle_in(box)
    box.audit(call("deploy now", "bad"))
    memo = box.engine.store.load_shadow_state(SID, "codex").memo
    assert memo is not None and memo["ids"] == ["no-deploy"] and memo["request"] == "audit"
    repeated = box.engine.audit_session(SID, provider="codex")["incremental"]
    assert repeated["shadow_skipped"] == "verified_before" and "shadow_compared" not in repeated       # the same input


# --- the forced second pass (finding 2) ---------------------------------------------------------------------------------

def test_a_forced_second_pass_that_runs_out_of_time_keeps_the_first_answer(lab, draw, slow_reference):
    """The first pass answered ``violation`` within the deadline; the pass that was to verify it overran its share. The answer
    is returned (unverified, and said so) instead of a deadline error, and the schedule remembers the violation."""
    box = lab(every="adaptive")
    settle_in(box)
    box.append(call("deploy now", "bad"))
    slow_reference.on = True
    began = time.monotonic()
    payload = box.engine.audit_session(SID, provider="codex", _deadline=time.monotonic() + 2.5)
    elapsed = time.monotonic() - began
    assert payload["verdict"] == "violation" and finding(payload, "no-deploy")["verdict"] == "violation"
    report = payload["incremental"]
    assert report["shadow_skipped"] == "deadline" and "shadow_compared" not in report
    assert slow_reference.calls == 1                      # the forced pass did start, and was stopped at its share of the time
    assert elapsed < 2.5                                  # and the request still answered within its deadline
    assert box.engine.store.load_shadow_state(SID, "codex").flags == {"audit": ["no-deploy"]}
    slow_reference.on = False
    again = box.audit()
    assert again["incremental"]["shadow_reasons"] == ["violation"] and again["incremental"]["shadow_compared"]


def test_a_forced_pass_that_finds_the_index_busy_keeps_the_first_answer(lab, draw, monkeypatch):
    box = lab(every="adaptive")
    settle_in(box)
    box.append(call("deploy now", "bad"))
    real, calls = inc.AuditPlan.prepare.__func__, []

    def prepare(cls, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise IndexBusy()
        return real(cls, *args, **kwargs)
    monkeypatch.setattr(inc.AuditPlan, "prepare", classmethod(prepare))
    payload = box.engine.audit_session(SID, provider="codex")
    assert len(calls) == 2 and payload["verdict"] == "violation" and payload["incremental"]["shadow_skipped"] == "busy"
    calls.clear()
    box.append(call("deploy again", "bad2"))
    monkeypatch.setattr(inc.AuditPlan, "prepare", classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(IndexBusy())))
    with pytest.raises(IndexBusy):                       # the first pass has no answer to fall back to
        box.engine.audit_session(SID, provider="codex")


def test_a_cancelled_request_does_not_get_the_first_answer(lab, draw, slow_reference):
    box = lab(every="adaptive")
    settle_in(box)
    box.append(call("deploy now", "bad"))
    slow_reference.on = True
    cancel = threading.Event()
    timer = threading.Timer(0.4, cancel.set)
    timer.start()
    try:
        with pytest.raises(RequestStopped) as stopped:
            box.engine.audit_session(SID, provider="codex", cancel_event=cancel)
    finally:
        timer.cancel()
    assert stopped.value.reason == "request_cancelled"


def wrong_block(monkeypatch, contract_id="no-deploy"):
    """A defect of the incremental path: it says ``violation`` where the full evaluator says ``clean``."""
    real = inc.RuleMonitor.finding

    def faulty(self, proposed, facts):
        item = real(self, proposed, facts)
        if self.rule.id == contract_id and item.verdict == "clean":
            return item.model_copy(update={"verdict": "violation", "reason": "injected", "evidence": []})
        return item
    monkeypatch.setattr(inc.RuleMonitor, "finding", faulty)


@pytest.mark.allow_incremental_mismatch
def test_a_stop_while_a_found_difference_is_recorded_never_returns_the_answer_known_to_be_wrong(lab, draw, monkeypatch):
    box = lab(every="adaptive")
    settle_in(box)
    wrong_block(monkeypatch)

    def stopped(*args, **kwargs):
        raise RequestStopped("request_deadline")
    monkeypatch.setattr(box.engine.store, "record_incremental_mismatch", stopped)
    with pytest.raises(RequestStopped):                    # the first pass said violation; the forced pass found that wrong
        box.audit()


@pytest.mark.allow_incremental_mismatch
def test_a_difference_found_after_the_time_share_of_the_forced_pass_is_still_recorded(lab, draw, monkeypatch):
    """Recording what a comparison found is not part of the verification and needs none of its time share."""
    box = lab(every="adaptive")
    settle_in(box)
    wrong_block(monkeypatch)
    real, calls = inc.compare, []

    def late(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            time.sleep(3.4)                                # the first comparison ends after the share of the pass (3 s of 4)
        return real(*args, **kwargs)
    monkeypatch.setattr(inc, "compare", late)
    box.append()
    payload = box.engine.audit_session(SID, provider="codex", _deadline=time.monotonic() + 4.0)
    assert payload["incremental"]["incremental_mismatch"] == ["no-deploy"]
    assert finding(payload, "no-deploy")["verdict"] == "clean"                          # the full evaluator's answer
    assert [row["contract_id"] for row in box.status()["disabled_contracts"]] == ["no-deploy"]
    assert len(box.status()["needs_review"]) == 1 and box.status()["session"]["reset_reason"] == "mismatch"


@pytest.mark.parametrize("affordable", [False, True])
def test_a_forced_second_pass_starts_only_when_there_is_time_for_it(lab, draw, monkeypatch, affordable):
    box = lab(every="adaptive")
    settle_in(box)
    box.append(call("deploy now", "bad"))
    monkeypatch.setattr(inc, "_shadow_affordable", lambda events: affordable)
    real, calls = inc.AuditPlan.prepare.__func__, []
    monkeypatch.setattr(inc.AuditPlan, "prepare",
                        classmethod(lambda cls, *args, **kwargs: (calls.append(1), real(cls, *args, **kwargs))[1]))
    report = box.engine.audit_session(SID, provider="codex")["incremental"]
    if affordable:
        assert len(calls) == 2 and report["shadow_compared"] and report["shadow_reasons"] == ["violation"]
    else:
        assert len(calls) == 1 and report["shadow_skipped"] == "deadline" and "shadow_compared" not in report


def test_the_time_of_a_forced_pass_ends_before_the_deadline_of_the_request(monkeypatch):
    """The reserve that keeps the first answer deliverable: a forced pass gets the request's time minus the reserve (never
    more than a quarter of what is left)."""
    assert inc.comparison_reserve(100.0) == inc.FINISH_RESERVE_SEC == 1.0
    assert inc.comparison_reserve(2.0) == 0.5 and inc.comparison_reserve(0.0) == 0.0 and inc.comparison_reserve(-3.0) == 0.0
    from self_directing_mcp.request_control import request_scope
    with request_scope(deadline=time.monotonic() + 10):
        with engine_module.SelfDirectEngine._comparison_scope():
            assert current_request().deadline < time.monotonic() + 10 - 0.9
    with request_scope():
        with engine_module.SelfDirectEngine._comparison_scope():            # no deadline: nothing to reserve
            assert current_request() is None or current_request().deadline is None


# --- what a stored schedule may contain (finding 4) ----------------------------------------------------------------------

GOOD = {"premise": {"contracts": "c"}, "rate_index": 1, "streak": 2, "warmup": 0, "compared": 3,
        "flags": {"audit": ["x"]}, "memo": {"state": ["g", 1, 5], "request": "audit", "result": "abc", "ids": ["x"]},
        "last_match_at": "t", "last_match_offset": 9, "reset_reason": None}


def test_a_complete_stored_schedule_is_read_back_exactly():
    state = shadow.ShadowState.load(3, GOOD)
    assert (state.epoch, state.rate_index, state.streak, state.warmup, state.compared) == (3, 1, 2, 0, 3)
    assert state.flags == {"audit": ["x"]} and state.memo["ids"] == ["x"] and state.last_match_offset == 9
    assert shadow.ShadowState.load(3, {**GOOD, "memo": None, "flags": {}}).memo is None


@pytest.mark.parametrize("change", [
    {"rate_index": 99}, {"rate_index": "1"}, {"rate_index": True}, {"rate_index": 1.0}, {"streak": -1}, {"warmup": None},
    {"compared": "3"}, {"flags": []}, {"flags": {"audit": "x"}}, {"flags": {"audit": [1]}}, {"memo": "garbage"}, {"memo": {}},
    {"memo": {"state": "x", "request": "a", "result": "r"}}, {"memo": {"state": [], "request": 1, "result": "r"}},
    {"memo": {"state": [], "request": "a", "result": 5}}, {"memo": {"state": [], "request": "a", "result": "r", "ids": "x"}},
    {"memo": {"state": [], "request": "a", "result": "r", "ids": [1]}}, {"last_match_at": 5},
    {"last_match_offset": "9"}, {"last_match_offset": True}, {"premise": 5}])
def test_a_stored_schedule_with_an_invalid_field_is_unreadable(change):
    with pytest.raises((ValueError, TypeError)):
        shadow.ShadowState.load(3, {**GOOD, **change})


@pytest.mark.parametrize("not_an_object", [None, [], "text", 5])
def test_a_stored_schedule_that_is_not_an_object_is_unreadable(not_an_object):
    with pytest.raises(ValueError):
        shadow.ShadowState.load(3, not_an_object)


@pytest.mark.parametrize("damage", [{"rate_index": 99}, {"rate_index": "2"}, {"memo": "garbage"},
                                    {"memo": {"state": [1], "request": "audit"}}])
def test_a_schedule_with_a_bad_field_is_replaced_and_the_rate_starts_again(lab, draw, damage):
    """The bad rate (or memo) must not crash the request, and must not stay: the row is replaced by a readable one at 1/4,
    with the warm-up, even though its epoch is higher than any fresh start's."""
    draw.value = False
    box = lab(every="adaptive")
    settle_in(box)
    store = box.engine.store
    row = json.loads(store._conn.execute("SELECT state FROM audit_shadow_state").fetchone()[0])
    epoch = store.load_shadow_state(SID, "codex").epoch + 10
    with store._conn:
        store._conn.execute("UPDATE audit_shadow_state SET epoch=?, state=?", (epoch, json.dumps({**row, **damage})))
    assert store.load_shadow_state(SID, "codex").reset_reason == "unreadable"
    report = box.report(box.audit())
    assert report["shadow_reset"] == "unreadable" and report["shadow_compared"]
    restored = store.load_shadow_state(SID, "codex")
    assert restored.epoch == epoch + 1 and restored.rate_index == 0 and restored.memo is None
    assert box.status()["session"]["rate"] == "1/4" and box.status()["session"]["forced_comparisons_left"] == shadow.WARMUP_REQUESTS - 1


# --- two writers of one schedule (finding 5) -------------------------------------------------------------------------------

def test_merged_flags_keep_the_other_request_kinds_and_replace_the_writers_own():
    assert shadow.merge_flags({"a": ["x"], "b": ["y"]}, {"a": ["z"]}, {"a"}) == {"b": ["y"], "a": ["z"]}
    assert shadow.merge_flags({"a": ["x"], "b": ["y"]}, {}, {"a"}) == {"b": ["y"]}                 # a's flag cleared by its owner
    assert shadow.merge_flags({"a": ["x"]}, {"c": ["w"]}, set()) == {"c": ["w"]}                  # a writer that handled none keeps its own
    assert shadow.merge_flags(None, {"a": ["z"]}, {"a"}) == {"a": ["z"]}
    many = {f"k{n}": ["x"] for n in range(shadow.FLAG_REQUESTS)}
    merged = shadow.merge_flags(many, {"new": ["y"]}, {"new"})
    assert len(merged) == shadow.FLAG_REQUESTS and "new" in merged and "k0" not in merged       # the oldest kind falls out
    state = shadow.settle(fresh(), outcome=None, full_scope=False, key="check:a", flagged={"x"}, kept=set(), at="t", offset=1)
    assert state.touched == {"check:a"} and "touched" not in state.dump()                         # not persisted


@pytest.mark.parametrize("defer", [False, True])
def test_two_requests_of_different_kinds_keep_each_others_flags(lab, draw, defer):
    """Two hooks that planned from the same stored schedule and flagged a contract each: the second save must not erase the
    first one's flag (a later flip of that action would not be compared)."""
    draw.value = False
    box = lab(every="adaptive", defer=defer)
    settle_in(box)
    store = box.engine.store
    base = store.load_shadow_state(SID, "codex")

    def flagged(key, ids):
        return shadow.settle(base, outcome=None, full_scope=False, key=key, flagged=ids, kept=set(), at="t", offset=9)
    store.save_shadow_state(SID, "codex", flagged("check:a", {"tests-first"}))
    store.save_shadow_state(SID, "codex", flagged("check:b", {"tests-first"}))
    assert store.load_shadow_state(SID, "codex").flags == {"check:a": ["tests-first"], "check:b": ["tests-first"]}
    store.save_shadow_state(SID, "codex", flagged("check:a", set()))                              # its owner clears its own
    assert store.load_shadow_state(SID, "codex").flags == {"check:b": ["tests-first"]}
    box.engine.flush_derived()                                                                    # and the table says the same
    stored = json.loads(store._conn.execute("SELECT state FROM audit_shadow_state").fetchone()[0])
    assert stored["flags"] == {"check:b": ["tests-first"]}


@pytest.mark.parametrize("defer", [False, True])
def test_a_request_that_planned_before_a_reset_cannot_undo_it(lab, draw, defer):
    draw.value = True
    box = lab(every="adaptive", defer=defer)
    for _ in range(40):
        box.audit()
    store = box.engine.store
    stale = store.load_shadow_state(SID, "codex")
    assert shadow.RATE_DENOMINATORS[stale.rate_index] == 32
    after = shadow.settle(stale, outcome="mismatch", full_scope=True, key="audit", flagged=set(), kept=set(), at="t", offset=1)
    store.record_incremental_mismatch(SID, "codex", inc.EVALUATOR_VERSION, ["no-deploy", "#facts"], "synthetic", after, None)
    late = shadow.settle(stale, outcome="match", full_scope=True, key="audit", flagged=set(), kept=set(), at="t", offset=2)
    assert store.save_shadow_state(SID, "codex", late) == "stale"
    now = store.load_shadow_state(SID, "codex")
    assert now.epoch == stale.epoch + 1 and now.rate_index == 0 and now.reset_reason == "mismatch"


def test_a_deferring_store_never_holds_an_older_epoch_over_a_newer_one(lab, draw):
    box = lab(every="adaptive", defer=True)
    box.audit()
    store = box.engine.store
    held = store.load_shadow_state(SID, "codex")
    newer = held.copy()
    newer.epoch, newer.rate_index = held.epoch + 1, 2
    assert store.save_shadow_state(SID, "codex", newer) == "deferred"
    older = held.copy()
    older.rate_index = 3
    assert store.save_shadow_state(SID, "codex", older) == "stale"
    kept = store.load_shadow_state(SID, "codex")
    assert kept.epoch == newer.epoch and kept.rate_index == 2
    assert store.derived.hold_shadow("codex", SID, held.epoch, held.dump()) is False and store.derived.shadow_for("codex", SID)[0] == newer.epoch


def test_a_reset_stored_by_another_process_is_not_hidden_by_an_older_held_schedule(lab, draw):
    box = lab(every="adaptive", defer=True)
    for _ in range(3):
        box.audit()
    store = box.engine.store
    held = store.load_shadow_state(SID, "codex")
    other = held.copy()
    other.epoch, other.rate_index, other.reset_reason = held.epoch + 5, 0, "mismatch"
    with store._conn:                                                 # what another process stored (the table's own guard)
        assert shadow_sql.write(store._conn, "codex", SID, other.epoch, other.dump())
    seen = store.load_shadow_state(SID, "codex")
    assert seen.epoch == other.epoch and seen.reset_reason == "mismatch"


# --- a mismatch is one transaction (finding 10b) ------------------------------------------------------------------------------

REVIEW = {"source": "sampled", "since_at": "t", "since_offset": 1, "until_offset": 2, "requests": 3,
          "contracts": ["no-deploy"], "detail": "synthetic"}


def test_a_mismatch_is_recorded_all_or_nothing(lab, draw, monkeypatch):
    box = lab(every="adaptive")
    settle_in(box)
    store = box.engine.store
    before = store.load_shadow_state(SID, "codex")
    after = shadow.settle(before, outcome="mismatch", full_scope=True, key="audit", flagged=set(), kept=set(), at="t", offset=1)
    real = shadow_sql.add_range

    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(shadow_sql, "add_range", broken)
    with pytest.raises(sqlite3.OperationalError):
        store.record_incremental_mismatch(SID, "codex", inc.EVALUATOR_VERSION, ["no-deploy", "#facts"], "synthetic", after, REVIEW)
    assert box.status()["disabled_contracts"] == [] and box.status()["needs_review"] == []      # not the health row alone
    assert store.load_shadow_state(SID, "codex").epoch == before.epoch                           # and no reset without the range
    monkeypatch.setattr(shadow_sql, "add_range", real)
    store.record_incremental_mismatch(SID, "codex", inc.EVALUATOR_VERSION, ["no-deploy", "#facts"], "synthetic", after, REVIEW)
    status = box.status()
    assert [row["contract_id"] for row in status["disabled_contracts"]] == ["no-deploy"]
    assert len(status["needs_review"]) == 1 and status["session"]["reset_reason"] == "mismatch"


@pytest.mark.allow_incremental_mismatch
def test_a_mismatch_that_could_not_be_written_is_reported_and_found_again(lab, draw, monkeypatch):
    box = lab(every="adaptive")
    worker = worker_for(box)
    settle_in(box)
    blind(monkeypatch)
    box.audit(call("deploy now", "bad1"))
    box.audit()
    assert worker.snapshot()["unverified_requests"] == 2
    draw.value = True
    real = shadow_sql.add_range
    monkeypatch.setattr(shadow_sql, "add_range", lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("full")))
    payload = box.audit()
    report = payload["incremental"]
    assert report["incremental_mismatch"] == ["no-deploy"] and report["mismatch_not_recorded"] == "OperationalError"
    assert finding(payload, "no-deploy")["verdict"] == "violation"                  # the answer is the full evaluator's anyway
    assert box.status()["disabled_contracts"] == [] and box.status()["needs_review"] == []
    assert worker.snapshot()["unverified_requests"] == 2                              # still pending: nothing was recorded
    monkeypatch.setattr(shadow_sql, "add_range", real)
    again = box.audit()
    assert again["incremental"]["incremental_mismatch"] == ["no-deploy"] and "mismatch_not_recorded" not in again["incremental"]
    status = box.status()
    assert [row["contract_id"] for row in status["disabled_contracts"]] == ["no-deploy"] and len(status["needs_review"]) == 1
    assert worker.snapshot()["pending_sessions"] == 0


@pytest.mark.allow_incremental_mismatch
def test_a_cancellation_that_arrives_while_a_mismatch_is_written_does_not_undo_it(lab, draw, monkeypatch):
    box = lab(every="adaptive")
    settle_in(box)
    blind(monkeypatch)
    box.audit(call("deploy now", "bad1"))
    draw.value = True
    cancel = threading.Event()
    real = shadow_sql.add_range

    def cancelling(*args, **kwargs):
        real(*args, **kwargs)
        current_request().cancel()                        # the request is cancelled after it was admitted to write
    monkeypatch.setattr(shadow_sql, "add_range", cancelling)
    with pytest.raises(RequestStopped):
        box.engine.audit_session(SID, provider="codex", cancel_event=cancel)
    status = box.status()
    assert [row["contract_id"] for row in status["disabled_contracts"]] == ["no-deploy"]         # the three together
    assert len(status["needs_review"]) == 1 and status["session"]["rate"] == "1/4" and status["session"]["reset_reason"] == "mismatch"


def test_recording_a_mismatch_drops_the_schedule_a_deferring_store_is_holding(lab, draw):
    """A request that finished just before holds its (older) schedule in memory; the reset must not be shadowed by it."""
    draw.value = True
    box = lab(every="adaptive", defer=True)
    for _ in range(40):
        box.audit()
    store = box.engine.store
    held = store.load_shadow_state(SID, "codex")
    assert store.derived.shadow_for("codex", SID) is not None and held.rate_index == len(shadow.RATE_DENOMINATORS) - 1
    after = shadow.settle(held, outcome="mismatch", full_scope=True, key="audit", flagged=set(), kept=set(), at="t", offset=1)
    store.record_incremental_mismatch(SID, "codex", inc.EVALUATOR_VERSION, ["no-deploy", "#facts"], "synthetic", after, None)
    assert store.derived.shadow_for("codex", SID) is None
    assert store.load_shadow_state(SID, "codex").reset_reason == "mismatch"


@pytest.mark.allow_incremental_mismatch
def test_a_mismatch_in_a_deferring_engine_resets_the_schedule_for_good(lab, draw, monkeypatch):
    draw.value = True
    box = lab(every="adaptive", defer=True)
    for _ in range(40):
        box.audit()
    blind(monkeypatch)
    payload = box.audit(call("deploy now", "bad"))
    assert payload["incremental"]["incremental_mismatch"] == ["no-deploy"]
    session = box.status()["session"]
    assert session["rate"] == "1/4" and session["reset_reason"] == "mismatch" and len(box.status()["needs_review"]) == 1
    box.audit()
    assert box.status()["session"]["rate"] == "1/4" and box.status()["session"]["streak"] <= 2


# --- a violation that turned clean and could not be compared stays remembered (kept) -------------------------------------------

@pytest.mark.allow_incremental_mismatch
def test_a_flip_that_could_not_be_compared_is_remembered_for_the_next_request(lab, draw, monkeypatch):
    box = lab(every="adaptive")
    settle_in(box)
    assert box.audit(call("deploy now", "bad"))["verdict"] == "violation"
    box.engine.flush_derived()
    connection = box.engine.store._conn
    with connection:                                                                  # the state forgets its hits
        for row, state in connection.execute("SELECT rowid, state FROM audit_monitor_state WHERE monitor_id='no-deploy'").fetchall():
            data = json.loads(state)
            data.update(hits=0, hit_ev=[], hit_n=0)
            connection.execute("UPDATE audit_monitor_state SET state=? WHERE rowid=?", (json.dumps(data), row))
    box.reopen()
    real = inc._shadow_affordable
    monkeypatch.setattr(inc, "_shadow_affordable", lambda events: False)
    first = box.audit()
    assert first["incremental"]["shadow_skipped"] == "deadline" and not first["incremental"].get("incremental_mismatch")
    assert finding(first, "no-deploy")["verdict"] == "clean"                          # wrong, and nobody could check it
    assert box.engine.store.load_shadow_state(SID, "codex").flags == {"audit": ["no-deploy"]}      # but it is remembered
    monkeypatch.setattr(inc, "_shadow_affordable", real)
    second = box.audit()
    assert second["incremental"]["incremental_mismatch"] == ["no-deploy"]               # found by the flip rule, the draw is off
    assert "mismatch" in second["incremental"]["shadow_reasons"] and finding(second, "no-deploy")["verdict"] == "violation"


# --- a scoped mismatch returns the whole reference answer (exp17) ----------------------------------------------------------------

@pytest.mark.allow_incremental_mismatch
def test_a_scoped_mismatch_answers_with_every_contract_not_the_compared_one_alone(lab, draw, monkeypatch):
    box = lab(every="adaptive", rules=[*RULES, HIGH])
    settle_in(box, checks=True)
    real = inc.RuleMonitor.finding

    def broken(self, proposed, facts):
        item = real(self, proposed, facts)
        if self.rule.id == "tests-first" and item.verdict == "violation":
            return item.model_copy(update={"verdict": "clean", "reason": "injected", "evidence": []})
        return item
    monkeypatch.setattr(inc.RuleMonitor, "finding", broken)
    payload = box.check("deploy now")
    assert payload["incremental"]["incremental_mismatch"] == ["tests-first"]
    assert sorted(item["contract_id"] for item in payload["findings"]) == ["no-deploy", "no-wipe", "tests-first"]
    assert payload["applicable_contracts"] == 3 and payload["coverage"] and finding(payload, "tests-first")["verdict"] == "violation"


# --- the idle worker (findings 3, 6, 8, 10a) ---------------------------------------------------------------------------------------

class StubEngine:
    """Just what the worker asks of an engine; ``outcomes`` maps a session id to what ``verify_incremental`` does."""

    def __init__(self, outcomes=None):
        self.outcomes, self.calls, self.idle, self.accepting = outcomes or {}, [], 100.0, True

    def accepts_background_work(self):
        return self.accepting

    def idle_seconds(self):
        return self.idle

    def verify_incremental(self, session_id, provider=None, path=None, action=None, variant="audit", _deadline=None,
                           cancel_event=None):
        self.calls.append((session_id, variant))
        outcome = self.outcomes[session_id]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome(session_id, variant, cancel_event) if callable(outcome) else outcome


NOT_COMPARED = {"ok": True, "compared": False, "mismatch": [], "skipped": "deadline", "mode": "delta"}
MATCHED = {"ok": True, "compared": True, "mismatch": [], "skipped": None, "mode": "delta"}


def stub_worker(outcomes, now):
    engine = StubEngine(outcomes)
    worker = ShadowWorker(engine, idle_sec=0.0, clock=lambda: now[0])
    engine.worker = worker
    return engine, worker


def pending(worker, *sessions, variant="audit"):
    for session in sessions:
        worker.note("codex", session, None, variant, None, 1, "unverified")


def retry_in(worker, session, now):
    return worker._entries[("codex", session)].retry_at - now[0]


def test_a_job_that_cannot_compare_backs_off_exponentially_up_to_a_cap():
    now = [1000.0]
    engine, worker = stub_worker({"s": NOT_COMPARED}, now)
    pending(worker, "s")
    waits = []
    for _ in range(9):
        assert worker.tick() == 1
        waits.append(retry_in(worker, "s", now))
        assert worker.tick() == 0                                   # not before its time
        now[0] += waits[-1] - 0.5
        assert worker.tick() == 0
        now[0] += 0.5
    assert waits == [30, 60, 120, 240, 480, 900, 900, 900, 900] and BACKOFF_SEC == 30 and BACKOFF_MAX_SEC == 900
    assert worker.snapshot()["not_compared"] == 9 and worker.snapshot()["pending_sessions"] == 1
    entry = worker._entries[("codex", "s")]
    for _ in range(5000):                                              # a long-lived process fails many times: no overflow
        worker._back_off(entry)
    assert retry_in(worker, "s", now) == BACKOFF_MAX_SEC


def test_a_comparison_that_ran_clears_the_backoff():
    now = [1000.0]
    engine, worker = stub_worker({}, now)
    pending(worker, "s")
    pending(worker, "s", variant="check")
    entry = worker._entries[("codex", "s")]
    worker._back_off(entry)
    worker._back_off(entry)
    assert entry.failures == 2 and entry.retry_at == now[0] + 60
    worker.note("codex", "s", None, "audit", None, 9, "match")
    assert entry.failures == 0 and entry.retry_at == 0.0                      # it compared: no reason to wait any more
    assert ("codex", "s") in worker._entries                                  # (the other kind of request is still pending)


def test_a_preempted_job_is_not_penalized_but_an_overrun_is():
    now = [1000.0]

    def preempted(session, variant, cancel):
        cancel.set()                                                  # a request arrived
        raise RequestStopped("request_cancelled")
    engine, worker = stub_worker({"p": preempted, "o": RequestStopped("request_deadline")}, now)
    pending(worker, "p")
    assert worker.tick() == 0 and worker.snapshot()["preempted"] == 1 and worker.snapshot()["overran"] == 0
    assert worker._entries[("codex", "p")].retry_at == 0.0 and worker._entries[("codex", "p")].failures == 0
    assert worker.tick() == 0 and engine.calls == [("p", "audit"), ("p", "audit")]          # tried again at once
    pending(worker, "o")
    del worker._entries[("codex", "p")]
    assert worker.tick() == 0 and worker.snapshot()["overran"] == 1 and worker.snapshot()["preempted"] == 2
    assert retry_in(worker, "o", now) == 30 and worker.tick() == 0 and engine.calls[-1] == ("o", "audit")


def test_a_busy_index_and_an_error_back_off_too():
    now = [1000.0]
    engine, worker = stub_worker({"b": IndexBusy(), "e": RuntimeError("boom")}, now)
    pending(worker, "b", "e")
    assert worker.tick() == 0 and worker.tick() == 0
    stats = worker.snapshot()
    assert stats["busy"] == 1 and stats["errors"] == 1 and stats["last_error"] == "RuntimeError" and stats["backing_off"] == 2
    assert retry_in(worker, "b", now) == 30 and retry_in(worker, "e", now) == 30
    assert worker.tick() == 0 and len(engine.calls) == 2                                      # nothing is due


def test_a_difference_that_could_not_be_written_keeps_the_session_pending_and_backs_off():
    now = [1000.0]
    unwritten = {"ok": True, "compared": True, "mismatch": ["a"], "skipped": None, "mode": "delta", "not_recorded": "OperationalError"}
    engine, worker = stub_worker({"s": unwritten}, now)
    pending(worker, "s")
    pending(worker, "s", variant="check")
    assert worker.tick() == 1                                          # one job: it stops at the first kind of request
    assert worker.snapshot()["mismatched"] == 1 and worker.snapshot()["pending_sessions"] == 1
    assert retry_in(worker, "s", now) == 30 and engine.calls == [("s", "audit")]
    assert worker.tick() == 0                                          # not before its time


def test_sessions_are_tried_in_turn_so_that_one_cannot_starve_the_rest():
    now = [1000.0]
    order = []

    def preempted(session, variant, cancel):
        order.append(session)
        cancel.set()
        raise RequestStopped("request_cancelled")                  # never penalized: only the rotation decides
    engine, worker = stub_worker({name: preempted for name in "abc"}, now)
    pending(worker, "a", "b", "c")
    for _ in range(6):
        worker.tick()
        now[0] += 1                                                 # the clock moves, as a real one does
    assert order == ["a", "b", "c", "a", "b", "c"]


def test_a_failing_session_waits_while_the_others_are_served():
    now = [1000.0]
    done = []

    def matched(session, variant, cancel):
        done.append(session)
        worker.note("codex", session, None, variant, None, 1, "match")
        return MATCHED
    engine, worker = stub_worker({"bad": NOT_COMPARED, "x": matched, "y": matched}, now)
    pending(worker, "bad", "x", "y")
    for _ in range(4):
        worker.tick()
    assert done == ["x", "y"] and [call[0] for call in engine.calls] == ["bad", "x", "y"]     # bad once, then backing off
    assert worker.snapshot()["pending_sessions"] == 1


def test_a_session_whose_contracts_have_no_incremental_path_any_more_is_dropped_not_retried():
    now = [1000.0]
    full = {"ok": True, "compared": False, "mismatch": [], "skipped": None, "mode": "full"}
    engine, worker = stub_worker({"s": full}, now)
    pending(worker, "s")
    pending(worker, "s", variant="check")
    assert worker.snapshot()["unverified_requests"] == 2
    assert worker.tick() == 2                                           # one job per kind of request, then nothing left
    assert worker.snapshot()["pending_sessions"] == 0 and worker.snapshot()["dropped"] == 2
    assert worker.tick() == 0 and len(engine.calls) == 2


def test_a_mismatch_clears_both_kinds_of_request_and_a_match_only_its_own():
    now = [1000.0]
    engine, worker = stub_worker({}, now)
    pending(worker, "s")
    pending(worker, "s", variant="check")
    worker.note("codex", "s", None, "audit", None, 5, "match")
    assert worker.since_match("codex", "s") == {"requests": 1, "since_at": worker._entries[("codex", "s")].ledgers["check"].first_at,
                                                "since_offset": 1, "until_offset": 1}
    worker.note("codex", "s", None, "check", None, 5, "mismatch")
    assert worker.snapshot()["pending_sessions"] == 0 and worker.since_match("codex", "s") is None
    pending(worker, "s")
    pending(worker, "s", variant="check")
    worker.note("codex", "s", None, "audit", None, 5, "mismatch")                   # found by the other kind: both go
    assert worker.snapshot()["pending_sessions"] == 0 and worker.snapshot()["unverified_requests"] == 0


def test_the_requests_in_doubt_are_counted_over_both_kinds():
    now = [1000.0]
    engine, worker = stub_worker({}, now)
    worker.note("codex", "s", None, "audit", None, 10, "unverified")
    worker.note("codex", "s", None, "audit", None, 20, "unverified")
    worker.note("codex", "s", None, "check", None, 5, "unverified")
    since = worker.since_match("codex", "s")
    assert since["requests"] == 3 and since["since_offset"] == 5 and since["until_offset"] == 20
    assert worker.since_match("codex", "other") is None


def test_the_ledger_covers_only_matches_made_after_the_worker_started():
    engine, worker = stub_worker({}, [0.0])
    from datetime import timedelta
    started = worker.started_at
    assert worker.ledger_covers((started + timedelta(seconds=1)).isoformat())
    assert not worker.ledger_covers((started - timedelta(seconds=1)).isoformat())
    assert worker.ledger_covers((started + timedelta(seconds=1)).replace(tzinfo=None).isoformat())     # a naive stamp is UTC
    assert not worker.ledger_covers(None) and not worker.ledger_covers("") and not worker.ledger_covers("not a time")


def test_stopping_the_worker_cancels_the_job_it_is_running(lab, draw, monkeypatch):
    """An engine that is closing must not wait for a comparison."""
    box = lab(every="adaptive")
    worker = worker_for(box, budget_sec=30.0)
    settle_in(box)
    box.audit()
    started, release = threading.Event(), threading.Event()
    real = engine_module.run_audit

    def slow(**kwargs):
        if threading.current_thread() is not threading.main_thread() and kwargs.get("plan") is None:
            started.set()
            while not release.is_set():
                check_request()
                time.sleep(0.005)
        return real(**kwargs)
    monkeypatch.setattr(engine_module, "run_audit", slow)
    thread = threading.Thread(target=worker.tick, name="idle-job")
    worker._job_thread = thread
    thread.start()
    try:
        assert started.wait(10)
        worker.stop(timeout=0.1)
        thread.join(5)
        assert not thread.is_alive()                                         # stopped by stop(), not by the end of the test
        assert worker.snapshot()["preempted"] == 1 and worker.snapshot()["pending_sessions"] == 1
    finally:
        release.set()
        thread.join(5)


def test_a_worker_never_reopens_an_engine_that_was_closed(lab, draw):
    box = lab(every="adaptive")
    worker = worker_for(box)
    settle_in(box)
    box.audit()
    assert box.engine.accepts_background_work() and worker.snapshot()["pending_sessions"] == 1
    box.engine.close()
    assert not box.engine.accepts_background_work()
    assert worker.tick() == 0 and box.engine.store is None                      # the tick does not even try
    assert worker.snapshot()["jobs"] == 0 and worker.snapshot()["busy"] == 0
    with pytest.raises(IndexBusy):                                              # and the operation itself refuses
        box.engine.verify_incremental(SID, provider="codex")
    assert box.engine.store is None and not box.engine.accepts_background_work()
    box.engine = box.open()                                                     # a request may open an engine again
    box.audit()
    assert box.engine.accepts_background_work() is True


def test_a_request_that_arrives_between_the_idle_check_and_the_start_cancels_the_job(lab, draw, monkeypatch):
    """The cancel event is published before the idle check: a request that is already evaluating when the job starts has
    cancelled it, and the job does not evaluate at all."""
    box = lab(every="adaptive")
    worker = worker_for(box)
    settle_in(box)
    box.audit()
    assert worker.snapshot()["pending_sessions"] == 1
    in_run, release, evaluated = threading.Event(), threading.Event(), []
    real_run = engine_module.run_audit

    def watched(**kwargs):
        if threading.current_thread().name == "request-T":
            in_run.set()
            release.wait(20)
        else:
            evaluated.append(kwargs.get("plan") is None)
        return real_run(**kwargs)
    monkeypatch.setattr(engine_module, "run_audit", watched)
    calls, requests = [0], []
    real_idle = box.engine.idle_seconds

    def idle_seconds():
        calls[0] += 1
        if calls[0] == 2:                          # the check inside the job, after it published its cancel event
            thread = threading.Thread(target=lambda: box.engine.audit_session(SID, provider="codex"), name="request-T")
            thread.start()
            assert in_run.wait(20)                 # the request is evaluating, so it has already cancelled the job
            requests.append(thread)
            return 100.0                           # (the idle check itself is made to lie: the request is in flight)
        return real_idle()
    monkeypatch.setattr(box.engine, "idle_seconds", idle_seconds)
    try:
        ran = worker.tick()
    finally:
        release.set()
        for thread in requests:
            thread.join(20)
    assert requests and ran == 0 and evaluated == [] and worker.snapshot()["preempted"] == 1
    assert worker.snapshot()["matched"] == 0 and worker.snapshot()["jobs"] == 0


@pytest.mark.allow_incremental_mismatch
@pytest.mark.parametrize("resident", [True, False])
def test_a_mismatch_after_a_restart_records_the_range_from_the_stored_last_match(lab, draw, monkeypatch, resident):
    """The in-memory ledger is gone after a restart (and a hook process never had one): the range of requests in doubt starts
    at the persisted last match, with a count that says what it covers."""
    box = lab(every="adaptive")
    if resident:
        worker_for(box)
    settle_in(box)
    last_match = box.status()["session"]["last_match_offset"]
    blind(monkeypatch)
    for number in range(3):
        box.audit(call("deploy now", f"bad{number}"))                       # wrong answers, unverified (the draw is off)
    box.reopen()
    if resident:
        worker_for(box)
    draw.value = True
    payload = box.audit()
    assert payload["incremental"]["incremental_mismatch"] == ["no-deploy"]
    (review,) = box.status()["needs_review"]
    assert review["since_offset"] == last_match and review["until_offset"] > last_match and review["requests"] is None
    assert "before this process started" in review["detail"] and review["source"] == "sampled"


@pytest.mark.allow_incremental_mismatch
def test_a_restarted_engine_counts_only_what_it_answered_and_starts_the_range_at_the_last_match(lab, draw, monkeypatch):
    box = lab(every="adaptive")
    worker_for(box)
    settle_in(box)
    last = box.status()["session"]
    blind(monkeypatch)
    box.audit(call("deploy now", "bad1"))                                 # in doubt, before the restart
    box.reopen()
    worker = worker_for(box)
    box.audit(call("deploy now", "bad2"))                                 # in doubt, after it: the only one the new ledger knows
    assert worker.snapshot()["unverified_requests"] == 1
    draw.value = True
    assert box.audit()["incremental"]["incremental_mismatch"] == ["no-deploy"]
    (review,) = box.status()["needs_review"]
    assert review["requests"] == 1 and review["since_at"] == last["last_match_at"] and review["since_offset"] == last["last_match_offset"]
    assert "before this process started" in review["detail"]


@pytest.mark.allow_incremental_mismatch
def test_a_mismatch_found_by_the_first_request_after_a_match_records_no_range_when_the_ledger_covers_it(lab, draw, monkeypatch):
    box = lab(every="adaptive")
    worker = worker_for(box)
    settle_in(box)                                                           # every request so far was compared and matched
    blind(monkeypatch)
    draw.value = True
    payload = box.audit(call("deploy now", "bad"))
    assert payload["incremental"]["incremental_mismatch"] == ["no-deploy"]
    assert box.status()["needs_review"] == [] and worker.snapshot()["pending_sessions"] == 0


@pytest.mark.allow_incremental_mismatch
def test_a_mismatch_counts_the_requests_in_doubt_of_both_kinds_and_clears_the_worker(lab, draw, monkeypatch):
    box = lab(every="adaptive", rules=[*RULES, HIGH])
    worker = worker_for(box)
    settle_in(box, checks=True)
    owed = worker.snapshot()["unverified_requests"]                 # (the last check of the set-up was answered unverified)
    blind(monkeypatch)
    box.audit(call("deploy now", "bad1"))
    box.check("ls")
    box.audit()
    assert worker.snapshot()["unverified_requests"] == owed + 3
    draw.value = True
    assert box.audit()["incremental"]["incremental_mismatch"] == ["no-deploy"]
    (review,) = box.status()["needs_review"]
    assert review["requests"] == owed + 3 and "before this process started" not in review["detail"]
    assert worker.snapshot()["pending_sessions"] == 0 and worker.snapshot()["unverified_requests"] == 0


# --- wiring: who starts the worker ------------------------------------------------------------------------------------------

def test_the_daemons_engines_start_the_idle_comparison_and_stop_it_with_the_engine(tmp_path):
    from self_directing_mcp.daemon.core import Engines
    settings = Settings(_env_file=None, local_only=True, openrouter_api_key_file=None, codex_sessions_dir=tmp_path / "s",
                        index_dir=tmp_path / "i", audit_shadow_idle_sec=2.0, audit_incremental_min_events=0)
    engines = Engines(defer_derived=True)
    engine = engines.get(settings)
    worker = engine._shadow_worker
    try:
        assert worker is not None and worker.idle_sec == 2.0 and len(engines._shadow_workers) == 1
        assert worker._thread is not None and worker._thread.is_alive() and engines.get(settings) is engine
    finally:
        engines.close_all()
    assert not worker._thread.is_alive() and worker._stop.is_set() and not engines._shadow_workers
    off = Settings(_env_file=None, local_only=True, openrouter_api_key_file=None, codex_sessions_dir=tmp_path / "s",
                   index_dir=tmp_path / "j", audit_shadow_idle_sec=0.0)
    quiet = Engines(defer_derived=True)
    try:
        assert quiet.get(off)._shadow_worker is None and not quiet._shadow_workers
    finally:
        quiet.close_all()


def test_an_engine_built_by_a_test_factory_gets_no_worker(tmp_path):
    from self_directing_mcp.daemon.core import Engines
    made = []
    engines = Engines(lambda settings: made.append(object()) or made[-1])
    settings = Settings(_env_file=None, local_only=True, openrouter_api_key_file=None, index_dir=tmp_path / "i")
    assert engines.get(settings) is made[0] and not engines._shadow_workers
    engines.close_all()


def test_the_stdio_server_starts_the_idle_comparison_before_it_serves(monkeypatch):
    from self_directing_mcp import server
    from self_directing_mcp import _runtime
    from self_directing_mcp.daemon import client
    events = []
    monkeypatch.setattr(client, "use_daemon", lambda: False)
    monkeypatch.setattr(_runtime, "load_numpy", lambda: None)
    monkeypatch.setattr(server, "_mask_logging", lambda: None)
    monkeypatch.setattr(server, "start_integrity_worker", lambda engine: events.append(("integrity", engine)))
    monkeypatch.setattr(server, "start_shadow_worker", lambda engine: events.append(("shadow", engine)))
    monkeypatch.setattr(server.mcp, "run", lambda transport: events.append(("run", transport)))
    server.main([])
    assert events == [("integrity", server._engine), ("shadow", server._engine), ("run", "stdio")]


def test_only_resident_processes_start_the_idle_comparison():
    """Hook processes live for one event: nothing in them may start background work."""
    source = Path(__file__).resolve().parents[1] / "src" / "self_directing_mcp"
    users = sorted(path.relative_to(source).as_posix() for path in source.rglob("*.py")
                   if "start_shadow_worker" in path.read_text(encoding="utf-8"))
    assert users == ["daemon/core.py", "server.py", "shadow_worker.py"]


# --- the schedule on every host, with the real draw (nothing is patched) ---------------------------------------------------------------

class Host:
    """One synthetic session of a host: how to grow it, how to put a violation into it and what to pass to an audit."""

    def __init__(self, engine, sid, provider, step, violate, close, audit_kwargs=None):
        self.engine, self.sid, self.provider = engine, sid, provider
        self.step, self.violate, self.close, self.audit_kwargs = step, violate, close, audit_kwargs or {}

    def audit(self):
        return self.engine.audit_session(self.sid, provider=self.provider, **self.audit_kwargs)


def host_engine(tmp_path, **paths):
    return SelfDirectEngine(Settings(_env_file=None, local_only=True, openrouter_api_key_file=None, index_dir=tmp_path / "index",
                                     audit_shadow_every="adaptive", audit_incremental_min_events=0,
                                     audit_shadow_idle_sec=0.0, **paths), defer_derived=False)


def host_codex(tmp_path, monkeypatch):
    box = Lab(tmp_path, every="adaptive")                 # RULES: no-deploy, no-wipe, said-done (met by the first records)
    return Host(box.engine, SID, "codex", lambda n: box.append(), lambda: box.append(call("rm -rf / now", "bad")), box.close)


def host_claude(tmp_path, monkeypatch):
    from test_claude_provider import SID as CLAUDE, assistant, human, main_file, tool_result, use, write as write_claude
    for name in ("CLAUDE_CONFIG_DIR", "SELF_DIRECT_CLAUDE_PROJECTS_DIR"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "projects"
    path = main_file(root)
    write_claude(path, human("ship it"), assistant(use("t0", "Bash", command="npm test")),
                 tool_result("t0", "ran", block_extra={"is_error": False}))
    engine = host_engine(tmp_path, claude_projects_dir=root)
    engine.upsert_contracts([
        {"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": "rm -rf", "roles": ["assistant"], "provider": "claude"},
        {"id": "asked", "type": "must", "scope": "message", "regex": "ship it", "roles": ["user"], "provider": "claude"}])

    def step(n):
        write_claude(path, assistant(use(f"t{n}", "Bash", command=f"echo {n}")),
                     tool_result(f"t{n}", "ok", block_extra={"is_error": False}), mode="a")

    def violate():
        write_claude(path, assistant(use("tbad", "Bash", command="rm -rf build")),
                     tool_result("tbad", "done", block_extra={"is_error": False}), mode="a")
    return Host(engine, CLAUDE, "claude", step, violate, engine.close)


def host_omp(tmp_path, monkeypatch):
    from test_omp_local import _append, _message
    root = tmp_path / "sessions"
    root.mkdir()
    file = root / "session.jsonl"
    _append(file, {"type": "title", "title": "t"}, {"type": "session", "id": "session-1"},
            _message("user", {"type": "text", "text": "Run tests before deploy"}))
    engine = host_engine(tmp_path, omp_sessions_dir=root)
    engine.upsert_contracts([
        {"id": "no-curl", "type": "must_not", "scope": "tool_call", "regex": "curl", "provider": "omp"},
        {"id": "asked", "type": "must", "scope": "message", "regex": "Run tests", "roles": ["user"], "provider": "omp"}])

    def tool(call_id, command):
        return _message("assistant", {"type": "toolCall", "id": call_id, "name": "shell", "arguments": {"command": command}})

    return Host(engine, "session-1", "omp", lambda n: _append(file, tool(f"c{n}", f"echo {n}")),
                lambda: _append(file, tool("cbad", "curl http://example.invalid")), engine.close, {"path": str(file)})


def host_agy(tmp_path, monkeypatch):
    from test_agy_parser_v6 import SID as AGY, append, make, user
    root = tmp_path / "app"
    path = make(root, [user(0, "Run npm test before deploy.")])
    engine = host_engine(tmp_path, agy_app_data_dirs=[root])
    engine.upsert_contracts([
        {"id": "no-curl", "provider": "agy", "type": "must_not", "scope": "message", "regex": "curl", "roles": ["user"]},
        {"id": "asked", "provider": "agy", "type": "must", "scope": "message", "regex": "npm test", "roles": ["user"]}])
    counter = [0]

    def step(n):
        counter[0] += 1
        append(path, user(counter[0], f"note {n}"))

    def violate():
        counter[0] += 1
        append(path, user(counter[0], "curl the staging box"))
    return Host(engine, AGY, "agy", step, violate, engine.close)


@pytest.mark.parametrize("build", [host_codex, host_claude, host_omp, host_agy], ids=["codex", "claude", "omp", "agy"])
def test_the_adaptive_schedule_works_end_to_end_on_every_host(tmp_path, monkeypatch, build):
    """The warm-up, the real draw (recomputed here from the stored schedule, never patched), the rate bookkeeping and a
    violation's forced comparison, on the session shape of each host."""
    host = build(tmp_path, monkeypatch)
    engine, provider = host.engine, host.provider
    sid = host.sid.lower() if provider in ("codex", "agy") else host.sid
    try:
        compared, checked = [], 0
        for number in range(1, 31):
            before = engine.store.load_shadow_state(sid, provider) if engine.store is not None else None
            host.step(number)
            payload = host.audit()
            assert payload["verdict"] == "clean", payload["findings"]
            report = payload["incremental"]
            ran = bool(report.get("shadow_compared"))
            compared.append(ran)
            if before is not None and not report.get("shadow_reset") and "rebuild" not in report.get("shadow_reasons", ()):
                expected = before.warmup > 0 or shadow.sampled(before, sid, provider, payload["coverage"]["byte_offset"], None)
                assert ran == expected, (number, report, before.dump())            # the engine applies the stored schedule
                checked += 1
        assert compared[:shadow.WARMUP_REQUESTS] == [True] * shadow.WARMUP_REQUESTS
        assert checked >= 20 and not all(compared) and any(compared[shadow.WARMUP_REQUESTS:])
        status = engine.audit_status(sid, provider=provider)["incremental_verification"]["session"]
        assert status["forced_comparisons_left"] == 0 and status["comparisons"] == sum(compared)
        assert status["rate"] in ("1/4", "1/8")                                       # 30 requests: the first halving or two
        host.violate()
        payload = host.audit()
        assert payload["verdict"] == "violation" and payload["incremental"]["shadow_compared"]
        assert "violation" in payload["incremental"]["shadow_reasons"]
    finally:
        host.close()


# --- the setting (finding 9) -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["fixed", "-1", "1.5", "", "adaptive 1"])
def test_a_bad_shadow_every_value_is_refused_with_a_message_that_names_adaptive(text):
    from self_directing_mcp import settings_core
    with pytest.raises(ValueError, match="adaptive") as refused:
        settings_core._to_shadow_every("audit_shadow_every", text)
    assert "audit_shadow_every" in str(refused.value)
    assert settings_core._to_shadow_every("audit_shadow_every", " Adaptive ") == "adaptive"
    assert settings_core._to_shadow_every("audit_shadow_every", "16") == 16


# --- what a persistent violation costs (finding 11d) --------------------------------------------------------------------------------

def test_a_persistent_violation_is_compared_again_only_when_the_stored_events_change(lab, draw):
    """Every request that appends an event after a violation re-compares that contract (a scoped pass); an identical request
    is not compared twice. Declined on purpose: skipping the second pass while a violation 'persists' would trust the
    incremental state exactly where a missed event would hide."""
    box = lab(every="adaptive")
    settle_in(box)
    first = box.audit(call("deploy now", "bad"))["incremental"]
    assert first["shadow_reasons"] == ["violation"] and first["shadow_scope"] == ["no-deploy"]
    same = box.engine.audit_session(SID, provider="codex")["incremental"]
    assert same["shadow_skipped"] == "verified_before" and "shadow_compared" not in same
    for _ in range(3):
        again = box.audit()["incremental"]
        assert again["shadow_reasons"] == ["violation"] and again["shadow_scope"] == ["no-deploy"] and again["shadow_compared"]
