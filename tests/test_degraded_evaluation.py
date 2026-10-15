"""A deadline-bound audit or action check whose evidence refresh runs out of its share of the time still judges.

Before, the sync's deadline was the request's deadline: when a cold sync could not finish, everything was lost,
including the checks that need no history. Now the sync gets ``engine.SYNC_BUDGET_SHARE`` of the time left; if that runs
out while the request is live, the stored index is judged with the coverage issue ``sync_deadline`` (nothing is ever
clean; a confirmed ``must_not`` match is still a violation). The request's own stop (deadline, cancellation) and a busy
index propagate as before, and ``sync_session`` keeps reporting its own deadline.

Synthetic sessions only. The slow sync is a stand-in that waits, checking for its stop, so no large file is needed;
two tests stop the real ingest instead, by holding its parser past the end of the sync's share.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading
import time

import pytest

from test_reliability import session, write, call, result, SID
from self_directing_mcp import engine as engine_module
from self_directing_mcp import request_stats
from self_directing_mcp.codex import parse as codex_parser
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.request_control import (
    IndexBusy, RequestStopped, check_request, current_request, request_scope,
)

PROHIBITED = {"tool_name": "shell", "arguments": {"command": "rm -rf /fictional-path"}}
HARMLESS = {"tool_name": "shell", "arguments": {"command": "echo harmless"}}
NO_DELETE = {"id": "no-delete", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf",
             "description": "Never delete user files."}
RUN_TESTS = {"id": "run-tests", "type": "must", "scope": "tool_call", "regex": r"pytest",
             "description": "Run the tests."}
BUDGET = 20.0           # the request's own budget: far longer than any of these tests needs


def wait_for_stop(self, **kwargs):
    """A sync that cannot finish: it waits, checking for its stop, until a deadline or a cancellation ends it."""
    while True:
        check_request()
        time.sleep(0.005)


@pytest.fixture
def stuck(monkeypatch):
    """Every sync waits for its stop, and its share of the time is 2 % of the request's budget (0.4 s of 20 s)."""
    monkeypatch.setattr(SelfDirectEngine, "_sync_session", wait_for_stop)
    monkeypatch.setattr(engine_module, "SYNC_BUDGET_SHARE", 0.02)


def check(engine, action, **kwargs):
    kwargs.setdefault("_deadline", time.monotonic() + BUDGET)
    started = time.monotonic()
    payload = engine.check_action(SID, action, **kwargs)
    return payload, time.monotonic() - started


def audit(engine, **kwargs):
    kwargs.setdefault("_deadline", time.monotonic() + BUDGET)
    started = time.monotonic()
    payload = engine.audit_session(SID, **kwargs)
    return payload, time.monotonic() - started


def verdicts(payload):
    return {finding["contract_id"]: finding["verdict"] for finding in payload["findings"]}


# --- a session that was never synced: the stored index is empty ----------------------------------------------------

def test_a_prohibited_action_is_a_violation_although_the_sync_ran_out_of_time(session, stuck):
    engine, _ = session
    engine.upsert_contracts([NO_DELETE])
    payload, elapsed = check(engine, PROHIBITED)
    assert payload["ok"] is True and payload["verdict"] == "violation"
    assert verdicts(payload) == {"no-delete": "violation"}
    assert payload["coverage"]["complete"] is False
    assert "sync_deadline" in payload["coverage"]["issues"] and "not_synced" in payload["coverage"]["issues"]
    assert payload["integrity"]["verification"] == "unavailable" and payload["integrity"]["reason"] == "sync_deadline"
    assert payload["action_executed"] is False and payload["requires_attention"] is True
    assert elapsed < BUDGET / 4         # the sync was cut at its share (0.4 s); the verdict did not wait for the budget


def test_an_action_that_matches_nothing_is_unknown_never_clean(session, stuck):
    engine, _ = session
    engine.upsert_contracts([NO_DELETE])
    payload, _ = check(engine, HARMLESS)
    assert payload["verdict"] == "unknown" and payload["requires_attention"] is True
    assert verdicts(payload) == {"no-delete": "clean"}      # the finding is what the action says; the overall verdict is not
    assert payload["coverage"]["complete"] is False and "sync_deadline" in payload["coverage"]["issues"]
    assert "No applicable enabled contracts or no indexed events." in payload["notes"]


def test_an_audit_continues_and_answers_unknown_with_the_issue(session, stuck):
    engine, _ = session
    engine.upsert_contracts([NO_DELETE])
    payload, elapsed = audit(engine)
    assert payload["ok"] is True and payload["verdict"] == "unknown"
    assert payload["coverage"]["complete"] is False and "sync_deadline" in payload["coverage"]["issues"]
    assert not payload["coverage"].get("judged_prefix")
    assert elapsed < BUDGET / 4


def test_a_must_obligation_cannot_be_proved_missing_from_an_index_that_is_not_there(session, stuck):
    engine, _ = session
    engine.upsert_contracts([RUN_TESTS])
    payload, _ = audit(engine)
    assert verdicts(payload) == {"run-tests": "unknown"} and payload["verdict"] == "unknown"
    assert payload["coverage"]["complete"] is False


def test_the_fast_hook_check_degrades_too(session, stuck):
    engine, _ = session
    engine.upsert_contracts([NO_DELETE])
    payload, _ = check(engine, PROHIBITED, verify="fast")
    assert payload["verdict"] == "violation" and "sync_deadline" in payload["coverage"]["issues"]
    assert payload["integrity"]["mode"] == "fast" and payload["integrity"]["verification"] == "unavailable"


def test_the_degraded_answer_is_reported_to_the_request_that_asked(session, stuck):
    engine, _ = session
    engine.upsert_contracts([NO_DELETE])
    notes = request_stats.Notes()
    with request_stats.collecting(notes):
        check(engine, PROHIBITED)
    assert notes.degraded == "sync_deadline"


# --- a session with a cursor: the violation in what is stored is found, clean is impossible ------------------------

def synced(engine, path, events):
    write(path, events, "a")
    assert engine.sync_session(session_id=SID, embed=False)["ok"] is True


def test_a_violation_in_the_stored_events_is_reported_when_the_sync_cannot_finish(session, monkeypatch):
    engine, path = session
    engine.upsert_contracts([NO_DELETE])
    synced(engine, path, [call("rm -rf /fictional-path", "c1"), result("c1")])
    monkeypatch.setattr(SelfDirectEngine, "_sync_session", wait_for_stop)
    monkeypatch.setattr(engine_module, "SYNC_BUDGET_SHARE", 0.02)
    payload, _ = audit(engine)
    assert payload["verdict"] == "violation" and verdicts(payload) == {"no-delete": "violation"}
    assert payload["coverage"]["complete"] is False and "sync_deadline" in payload["coverage"]["issues"]
    assert payload["coverage"]["chunk_count"] > 0 and "not_synced" not in payload["coverage"]["issues"]


@pytest.mark.parametrize("appended", [False, True], ids=["nothing appended", "bytes appended after the cursor"])
def test_stored_events_that_match_nothing_never_make_the_answer_clean(session, monkeypatch, appended):
    engine, path = session
    engine.upsert_contracts([NO_DELETE])
    synced(engine, path, [call("echo harmless", "c1"), result("c1")])
    assert engine.audit_session(SID)["verdict"] == "clean"          # with the sync working, this history is clean
    if appended:
        write(path, [call("echo more", "c2")], "a")                 # unread bytes alone would still be judged as a prefix
    monkeypatch.setattr(SelfDirectEngine, "_sync_session", wait_for_stop)
    monkeypatch.setattr(engine_module, "SYNC_BUDGET_SHARE", 0.02)
    for payload, _ in (audit(engine), check(engine, HARMLESS)):
        assert payload["verdict"] == "unknown"
        assert payload["coverage"]["complete"] is False and "sync_deadline" in payload["coverage"]["issues"]
        assert not payload["coverage"].get("judged_prefix")             # the extra issue is what keeps "clean as of the cursor" away
        assert ("unread_bytes" in payload["coverage"]["issues"]) is appended
    payload, _ = check(engine, PROHIBITED)
    assert payload["verdict"] == "violation"


def test_a_prerequisite_met_in_the_stored_events_cannot_make_a_degraded_check_clean(session, monkeypatch):
    engine, path = session
    engine.upsert_contracts([{"id": "tests-first", "type": "must", "scope": "tool_call", "regex": r"pytest",
                              "before_regex": r"deploy", "description": "Test before you deploy."}])
    synced(engine, path, [call("pytest -q", "t1"), result("t1")])
    deploy = {"tool_name": "shell", "arguments": {"command": "deploy now"}}
    assert check(engine, deploy)[0]["verdict"] == "clean"          # with the sync working the prerequisite is met
    monkeypatch.setattr(SelfDirectEngine, "_sync_session", wait_for_stop)
    monkeypatch.setattr(engine_module, "SYNC_BUDGET_SHARE", 0.02)
    payload, elapsed = check(engine, deploy, verify="fast")
    assert payload["verdict"] == "unknown" and verdicts(payload) == {"tests-first": "unknown"}
    assert "sync_deadline" in payload["coverage"]["issues"]
    assert payload["integrity"]["mode"] == "fast" and elapsed < BUDGET / 4      # no second, doomed, full refresh


# --- a real sync that is stopped in the middle ----------------------------------------------------------------------

@contextmanager
def parse_until_the_share_is_over():
    """The real parser, followed by a wait past the end of the sync's own share: the ingest is stopped before it publishes."""
    original = codex_parser.parse_session_chunks

    def slow(*args, **kwargs):
        parsed = original(*args, **kwargs)
        while time.monotonic() < current_request().deadline + 0.05:
            time.sleep(0.005)
        return parsed

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(codex_parser, "parse_session_chunks", slow)
        patch.setattr(engine_module, "SYNC_BUDGET_SHARE", 0.025)         # 0.5 s of 20
        yield


def test_a_real_cold_sync_that_is_stopped_publishes_nothing_and_the_next_call_finishes_it(session):
    engine, path = session
    engine.upsert_contracts([NO_DELETE])
    write(path, [call("rm -rf /fictional-path", "c1"), result("c1")], "a")
    with parse_until_the_share_is_over():
        payload, elapsed = check(engine, PROHIBITED)
    assert payload["verdict"] == "violation" and "sync_deadline" in payload["coverage"]["issues"]
    assert "not_synced" in payload["coverage"]["issues"] and elapsed < BUDGET / 4
    assert engine.store.cursor_record(SID) is None and not engine.store.chunk_ids(SID)      # nothing was published
    again = engine.audit_session(SID)                                                       # no deadline: the sync completes
    assert again["verdict"] == "violation" and again["coverage"]["issues"] == []
    assert again["coverage"]["chunk_count"] > 0


def test_a_real_append_sync_that_is_stopped_leaves_the_stored_index_as_it_was(session):
    engine, path = session
    engine.upsert_contracts([NO_DELETE])
    synced(engine, path, [call("echo harmless", "c1"), result("c1")])
    before, ids = engine.store.cursor_record(SID), engine.store.chunk_ids(SID)
    write(path, [call("rm -rf /fictional-path", "c2")], "a")
    with parse_until_the_share_is_over():
        payload, _ = audit(engine)
    assert payload["verdict"] == "unknown"             # the violation is in the part of the file that was not synced
    assert {"sync_deadline", "unread_bytes"} <= set(payload["coverage"]["issues"])
    assert engine.store.cursor_record(SID) == before and engine.store.chunk_ids(SID) == ids
    again = engine.audit_session(SID)
    assert again["verdict"] == "violation" and again["coverage"]["issues"] == []


# --- the share of the time -----------------------------------------------------------------------------------------

def test_the_sync_runs_in_a_child_scope_that_ends_at_its_share_of_the_time_left(session, monkeypatch):
    engine, _ = session
    engine.upsert_contracts([NO_DELETE])
    seen = {}

    def spy(self, **kwargs):
        seen["at"], seen["deadline"] = time.monotonic(), current_request().deadline
        return wait_for_stop(self, **kwargs)

    monkeypatch.setattr(SelfDirectEngine, "_sync_session", spy)
    monkeypatch.setattr(engine_module, "SYNC_BUDGET_SHARE", 0.05)       # 1 s of 20
    parent = time.monotonic() + BUDGET
    payload = engine.check_action(SID, PROHIBITED, _deadline=parent)
    assert payload["verdict"] == "violation"
    assert seen["deadline"] < parent
    assert seen["deadline"] - seen["at"] == pytest.approx(0.05 * (parent - seen["at"]), abs=0.15)


def test_the_share_is_a_module_constant_of_sixty_percent():
    assert engine_module.SYNC_BUDGET_SHARE == 0.6


def test_a_share_of_zero_judges_at_once(session, monkeypatch):
    engine, _ = session
    engine.upsert_contracts([NO_DELETE])
    monkeypatch.setattr(SelfDirectEngine, "_sync_session", wait_for_stop)
    monkeypatch.setattr(engine_module, "SYNC_BUDGET_SHARE", 0.0)
    payload, elapsed = check(engine, PROHIBITED)
    assert payload["verdict"] == "violation" and "sync_deadline" in payload["coverage"]["issues"]
    assert elapsed < BUDGET / 4


def test_without_a_deadline_the_sync_has_no_narrower_budget(session, monkeypatch):
    engine, _ = session
    engine.upsert_contracts([NO_DELETE])
    seen = {}
    original = SelfDirectEngine._sync_session

    def spy(self, **kwargs):
        seen["control"] = current_request()
        return original(self, **kwargs)

    monkeypatch.setattr(SelfDirectEngine, "_sync_session", spy)
    payload = engine.check_action(SID, HARMLESS)
    assert seen["control"] is None
    assert "sync_deadline" not in payload["coverage"]["issues"]


def test_only_audits_and_action_checks_hand_out_a_share(session, monkeypatch):
    engine, _ = session
    engine.ensure_ready()
    parent = time.monotonic() + BUDGET
    seen = {}

    def spy(self, **kwargs):
        seen["deadline"] = current_request().deadline
        return {"ok": False, "error": "not_found", "message": "synthetic"}

    monkeypatch.setattr(SelfDirectEngine, "_sync_session", spy)
    with request_scope(deadline=parent):
        engine._refresh_evidence(SID, "codex")                           # the checklist and graph callers: no share
        assert seen["deadline"] == parent
        engine._refresh_evidence(SID, "codex", sync_share=0.5)
        assert seen["deadline"] < parent - BUDGET / 4


@pytest.mark.parametrize("idle", [False, True], ids=["a request", "the idle comparison"])
def test_the_idle_comparison_is_not_cut_short_but_a_request_is(session, monkeypatch, idle):
    engine, _ = session
    engine.ensure_ready()
    asked = []
    original = SelfDirectEngine._refresh_evidence

    def spy(self, *args, **kwargs):
        asked.append(kwargs.get("sync_share"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SelfDirectEngine, "_refresh_evidence", spy)
    with request_scope(deadline=time.monotonic() + BUDGET):
        engine._audit(SID, "codex", idle=idle)
    assert asked == [None if idle else engine_module.SYNC_BUDGET_SHARE]


# --- what still propagates -----------------------------------------------------------------------------------------

def test_the_requests_own_deadline_during_the_sync_is_still_a_request_deadline(session, monkeypatch):
    engine, _ = session
    engine.upsert_contracts([NO_DELETE])

    def deaf_sync(self, **kwargs):
        time.sleep(0.7)             # does not look until both the share (0.3 s) and the request (0.5 s) are over
        check_request()
        raise AssertionError("the request's deadline did not stop the sync")

    monkeypatch.setattr(SelfDirectEngine, "_sync_session", deaf_sync)
    monkeypatch.setattr(engine_module, "SYNC_BUDGET_SHARE", 0.6)
    notes = request_stats.Notes()
    with request_stats.collecting(notes), pytest.raises(RequestStopped) as stopped:
        engine.check_action(SID, PROHIBITED, _deadline=time.monotonic() + 0.5)
    assert stopped.value.reason == "request_deadline"
    assert notes.degraded is None          # the request was stopped, not answered from less evidence


def test_the_requests_own_deadline_during_the_fallback_is_still_a_request_deadline(session, stuck, monkeypatch):
    engine, _ = session
    engine.upsert_contracts([NO_DELETE])
    original = engine_module.coverage_from_capture

    def slow_coverage(*args, **kwargs):
        time.sleep(1.0)             # the sync ended at its share (0.2 s of 1 s), then the judging overran the request
        check_request()
        return original(*args, **kwargs)

    monkeypatch.setattr(engine_module, "SYNC_BUDGET_SHARE", 0.2)
    monkeypatch.setattr(engine_module, "coverage_from_capture", slow_coverage)
    with pytest.raises(RequestStopped) as stopped:
        engine.check_action(SID, PROHIBITED, _deadline=time.monotonic() + 1.0)
    assert stopped.value.reason == "request_deadline"


def test_a_client_cancellation_during_the_sync_is_a_request_cancelled(session, monkeypatch):
    engine, _ = session
    engine.upsert_contracts([NO_DELETE])
    monkeypatch.setattr(SelfDirectEngine, "_sync_session", wait_for_stop)
    cancel = threading.Event()
    threading.Timer(0.15, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(RequestStopped) as stopped:
        engine.check_action(SID, PROHIBITED, _deadline=time.monotonic() + BUDGET, cancel_event=cancel)
    assert stopped.value.reason == "request_cancelled" and time.monotonic() - started < BUDGET / 4


def test_a_client_cancellation_after_the_sync_ran_out_of_its_share_is_still_a_request_cancelled(session, stuck, monkeypatch):
    engine, _ = session
    engine.upsert_contracts([NO_DELETE])
    cancel, original = threading.Event(), engine_module.coverage_from_capture

    def cancelling(*args, **kwargs):
        cancel.set()                # the client goes away while the stored index is being read
        check_request()
        return original(*args, **kwargs)

    monkeypatch.setattr(engine_module, "coverage_from_capture", cancelling)
    with pytest.raises(RequestStopped) as stopped:
        engine.audit_session(SID, _deadline=time.monotonic() + BUDGET, cancel_event=cancel)
    assert stopped.value.reason == "request_cancelled"


def test_a_cancelled_request_is_not_reported_as_degraded(session, monkeypatch):
    engine, _ = session
    monkeypatch.setattr(SelfDirectEngine, "_sync_session", wait_for_stop)
    cancel = threading.Event()
    threading.Timer(0.1, cancel.set).start()
    notes = request_stats.Notes()
    with request_stats.collecting(notes), pytest.raises(RequestStopped):
        engine.audit_session(SID, _deadline=time.monotonic() + BUDGET, cancel_event=cancel)
    assert notes.degraded is None


@pytest.mark.parametrize("method", ["check_action", "audit_session"])
def test_a_busy_index_still_propagates(session, monkeypatch, method):
    engine, _ = session

    def busy(self, **kwargs):
        raise IndexBusy()

    monkeypatch.setattr(SelfDirectEngine, "_sync_session", busy)
    args = (SID, PROHIBITED) if method == "check_action" else (SID,)
    with pytest.raises(IndexBusy):
        getattr(engine, method)(*args, _deadline=time.monotonic() + BUDGET)


def test_an_ordinary_sync_failure_is_still_handled_as_before(session, monkeypatch):
    engine, _ = session
    engine.upsert_contracts([NO_DELETE])

    def broken(self, **kwargs):
        raise OSError("synthetic read failure")

    monkeypatch.setattr(SelfDirectEngine, "_sync_session", broken)
    payload, _ = check(engine, PROHIBITED)
    assert payload["verdict"] == "violation"
    assert "sync_failed:OSError" in payload["coverage"]["issues"] and "sync_deadline" not in payload["coverage"]["issues"]


def test_sync_session_still_reports_its_own_deadline_without_a_fallback(session, stuck, monkeypatch):
    engine, _ = session
    started = time.monotonic()
    notes = request_stats.Notes()
    with request_stats.collecting(notes), pytest.raises(RequestStopped) as stopped:
        engine.sync_session(session_id=SID, embed=False, _deadline=started + 0.6)
    elapsed = time.monotonic() - started
    assert stopped.value.reason == "request_deadline"
    assert 0.5 <= elapsed < 5           # at its own deadline: neither cut at a share (0.012 s here) nor given more
    assert notes.degraded is None
