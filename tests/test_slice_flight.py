"""One writer per session, and a bounded wait for everyone else (roundtable 8, P3).

A session is synced by one request at a time (``ingest.session_flight``). A request that may slice does not sit on a
heavy-call slot behind the leader: it waits at most ``min(1 s, a quarter of the time it has left)`` and then judges on
what is committed, with the issue ``sync_in_progress``. Without slicing the wait is what it always was: the waiter's own
deadline. Nobody parses twice.

An ``audit_session`` or ``check_action`` gives its sync ``engine.SYNC_BUDGET_SHARE`` of the time it has left, so the
time "left" that the wait is a quarter of is that share. The statistics count an answer that was given while the sync was
still in progress as ``degraded`` / ``sync_in_progress``, and a ``sync_session`` that met a busy session as ``incomplete``
with the same reason.
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from self_directing_mcp import engine as engine_module
from self_directing_mcp import request_stats, server
from self_directing_mcp.codex import parse as codex_parser
from self_directing_mcp.index import ingest
from self_directing_mcp.index.ingest import FlightBusy, SlicePolicy
from self_directing_mcp.request_control import IndexBusy, RequestControl, RequestStopped, request_scope
from self_directing_mcp.schemas import ProposedAction
from slicing_support import forced
from test_slice_equivalence import BIG
from test_reliability import SID, call, message, write, session as reliability_session  # noqa: F401

NO_WIPE = {"id": "no-wipe", "type": "must_not", "scope": "tool_call", "regex": "rm -rf"}


@pytest.fixture
def session(reliability_session):          # noqa: F811
    engine, path = reliability_session
    try:
        yield engine, path
    finally:
        engine.close()


class Leader:
    """A sync of the session paused inside its parse, holding the flight until ``release`` is set. With ``sliced`` it is a
    request with a deadline whose parse stops after 10 records: the leader of a session that is being synced in slices."""

    def __init__(self, engine, path, monkeypatch, sliced=False):
        write(path, [message(f"event {n}") for n in range(40)], "a")
        self.sliced = sliced
        self.entered, self.release, self.parses = threading.Event(), threading.Event(), []
        original = codex_parser.parse_session_chunks

        def paused(*args, **kwargs):
            self.parses.append(kwargs.get("start_byte", 0))
            self.entered.set()
            self.release.wait(10)
            return original(*args, **kwargs)

        monkeypatch.setattr(codex_parser, "parse_session_chunks", paused)
        self.pool = ThreadPoolExecutor(2)
        options = {}
        if sliced:
            engine.slice_policy = forced(10)
            options["_deadline"] = time.monotonic() + BIG
        self.result = self.pool.submit(engine.sync_session, session_id=SID, embed=False, derived=False, **options)
        assert self.entered.wait(5)

    def finish(self):
        self.release.set()
        outcome = self.result.result(10)
        self.pool.shutdown(wait=True)
        return outcome


@pytest.mark.parametrize("sliced", [False, True])
def test_a_second_audit_waits_a_bounded_time_and_reports_sync_in_progress(session, monkeypatch, sliced):
    engine, path = session
    engine.upsert_contracts([{"id": "no-wipe", "type": "must_not", "scope": "tool_call", "regex": "rm -rf"}])
    leader = Leader(engine, path, monkeypatch, sliced=sliced)
    engine.slice_policy = None                                          # the waiter's own sync is not forced to stop
    started = time.monotonic()
    answer = engine.audit_session(SID, _deadline=started + 2.0)         # a quarter of the sync's 60 % of 2 s: 0.3 s
    waited = time.monotonic() - started
    expected = ingest.FLIGHT_WAIT_SHARE * engine_module.SYNC_BUDGET_SHARE * 2.0
    try:
        assert expected - 0.03 <= waited < 1.5                           # bounded: it never waited for its own deadline (2 s)
        coverage = answer["coverage"]
        assert answer["verdict"] != "clean" and coverage["complete"] is False
        assert "sync_in_progress" in coverage["issues"] and not coverage.get("judged_prefix")
        assert answer["integrity"]["reason"] == "sync_in_progress"
        assert leader.parses == [0]                                      # nobody else parsed: no duplicate work
    finally:
        info = leader.finish()
    assert info["ok"] and leader.parses == [0]
    if sliced:
        assert info["sync"]["state"] == "partial" and info["new_chunks"] == 10      # the leader published its slice
        engine.slice_policy = None
        assert engine.sync_session(session_id=SID, embed=False, _deadline=time.monotonic() + BIG)["sync"]["state"] == "complete"
        assert leader.parses == [0, info["byte_offset"]]                 # the rest was parsed once, from the cursor
    else:
        assert info["total_chunks"] == 41
    assert engine.audit_session(SID)["verdict"] == "clean"                # and the next call finds everything indexed


def test_a_second_check_action_of_a_prohibited_action_is_still_a_violation(session, monkeypatch):
    """The proposed action itself is judged without any history: the bounded wait fails open into the old failure branch."""
    engine, path = session
    engine.upsert_contracts([{"id": "no-wipe", "type": "must_not", "scope": "tool_call", "regex": "rm -rf"}])
    leader = Leader(engine, path, monkeypatch)
    try:
        started = time.monotonic()
        answer = engine.check_action(SID, {"tool_name": "shell", "arguments": {"command": "rm -rf /"}},
                                     _deadline=started + 2.0)
        assert answer["verdict"] == "violation" and time.monotonic() - started < 1.5       # not its own deadline (2 s)
        assert "sync_in_progress" in answer["coverage"]["issues"]
    finally:
        leader.finish()


def test_sync_session_of_a_busy_session_is_an_incomplete_response_with_the_planned_reason(session, monkeypatch):
    engine, path = session
    leader = Leader(engine, path, monkeypatch)
    try:
        with pytest.raises(FlightBusy) as busy:
            engine.sync_session(session_id=SID, embed=False, _deadline=time.monotonic() + 0.8)
        assert isinstance(busy.value, IndexBusy) and busy.value.reason == "sync_in_progress"
        # Through the server's dispatcher: the response of a planned stop, not of a deadline overrun.
        answer = asyncio.run(server._dispatch(engine.sync_session, session_id=SID, embed=False, budget_override=0.8,
                                              heavy=False))
        assert answer["ok"] is False and answer["error"] == "sync_in_progress" and answer["verdict"] == "unknown"
        assert answer["coverage"] == {"complete": False, "issues": ["sync_in_progress"]}
        assert answer["action_executed"] is False
    finally:
        leader.finish()


def test_the_wait_is_a_quarter_of_the_time_left_and_at_most_one_second(session, monkeypatch):
    engine, path = session
    path_ = path.resolve()
    slicing = SlicePolicy()
    with request_scope(RequestControl(deadline=time.monotonic() + 60)):
        assert ingest._flight_wait(slicing, path_) == ingest.FLIGHT_WAIT_MAX_SEC == 1.0
    with request_scope(RequestControl(deadline=time.monotonic() + 3.0)):
        assert 0.5 < ingest._flight_wait(slicing, path_) <= 0.75           # a quarter of what is left (a little under 3 s)
    assert ingest._flight_wait(slicing, path_) is None                      # no deadline: wait for the leader, as before
    with request_scope(RequestControl(deadline=time.monotonic() + 60)):
        assert ingest._flight_wait(SlicePolicy(mode="off"), path_) is None  # slicing off: as before
        assert ingest._flight_wait(None, path_) is None
        assert ingest._flight_wait(slicing, path_.with_name(path_.name + ".zst")) is None   # never cut: never short-waited


def test_the_soft_deadline_of_an_audit_comes_from_the_sync_share_not_from_the_whole_request():
    """The sync of ``audit_session``/``check_action`` runs in a narrower request scope (the 60/40 split): the slice
    budget reads that deadline, so parse and commit stay inside the share and the judging keeps its 40 %."""
    clock, slicing = [100.0], SlicePolicy(clock=lambda: clock[0])
    now = time.monotonic()
    with request_scope(RequestControl(deadline=now + 8.0)):
        whole = ingest._new_budget(slicing, 0)
        with request_scope(deadline=now + engine_module.SYNC_BUDGET_SHARE * 8.0):
            share = ingest._new_budget(slicing, 0)
    assert abs((whole.soft_deadline - 100.0) - 0.5 * 8.0 / 1.3) < 0.05                  # 3.08 s of the whole request
    assert abs((share.soft_deadline - 100.0) - 0.5 * 4.8 / 1.3) < 0.05                  # 1.85 s of its sync share
    assert share.soft_deadline < whole.soft_deadline


def test_with_slicing_off_a_waiter_still_waits_for_the_whole_share_the_sync_has(session, monkeypatch):
    """Without slicing the wait is what it was: until the sync's own share of the request is spent (then the stored
    index is judged, ``sync_deadline``), not a quarter of it."""
    engine, path = session
    engine.settings.sync_slicing = "off"
    monkeypatch.setattr(engine_module, "SYNC_BUDGET_SHARE", 0.25)           # 0.5 s of 2 s; a quarter of that is 0.125 s
    leader = Leader(engine, path, monkeypatch)
    try:
        started = time.monotonic()
        answer = engine.audit_session(SID, _deadline=started + 2.0)
        waited = time.monotonic() - started
        issues = answer["coverage"]["issues"]
        assert 0.47 <= waited < 1.5
        assert "sync_deadline" in issues and "sync_in_progress" not in issues and answer["verdict"] == "unknown"
    finally:
        leader.finish()


def test_with_slicing_off_the_request_deadline_itself_still_ends_the_wait_as_before(session, monkeypatch):
    engine, path = session
    engine.settings.sync_slicing = "off"
    leader = Leader(engine, path, monkeypatch)
    try:
        started = time.monotonic()
        with pytest.raises(RequestStopped) as stopped:
            engine.sync_session(session_id=SID, embed=False, derived=False, _deadline=started + 0.5)
        assert stopped.value.reason == "request_deadline" and time.monotonic() - started >= 0.45
    finally:
        leader.finish()


def test_a_waiter_does_not_hold_a_heavy_call_slot_longer_than_its_wait(session, monkeypatch):
    """With two slots, one held by the leader: the waiter's slot is free again after its bounded wait, not at its deadline."""
    engine, path = session
    monkeypatch.setattr(server._engine.settings, "max_concurrent_calls", 2)
    monkeypatch.setattr(server, "_libc", lambda: None)
    leader = Leader(engine, path, monkeypatch)
    began, laps = time.monotonic(), []

    async def scenario():
        gate = server._gate._for_engine()
        waiter = asyncio.create_task(server._dispatch(engine.audit_session, SID, audit=True, budget_override=2.0))
        await asyncio.sleep(0.05)
        assert gate.running == 1                                   # the waiter's own slot
        answer = await waiter
        laps.append(time.monotonic() - began)
        return answer, gate.running

    try:
        answer, running_after = asyncio.run(scenario())
        assert answer["coverage"]["issues"].count("sync_in_progress") == 1 and laps[0] < 1.5     # 0.25 * 2 s, not 2 s
        assert running_after == 0                                  # released as soon as the call returned
    finally:
        leader.finish()


# --- the statistics (request_stats): the vocabulary has the new reason --------------------------------------------------

@pytest.fixture
def stats(monkeypatch):
    fresh = request_stats.RequestStats()
    monkeypatch.setattr(server, "STATS", fresh)
    monkeypatch.setattr(server._engine.settings, "max_concurrent_calls", 2)
    monkeypatch.setattr(server, "_libc", lambda: None)
    return fresh


def counted(snapshot, name):
    entry = snapshot["tools"][name]
    return (entry["calls"], {key: value for key, value in entry["outcomes"].items() if value},
            {outcome: group for outcome, group in entry["reasons"].items() if group})


def stand_in(name, result):
    def method(*args, _deadline=None, cancel_event=None, **kwargs):
        return result

    method.__name__ = name
    return method


def test_sync_in_progress_is_a_name_of_the_closed_vocabulary():
    assert "sync_in_progress" in request_stats.REASONS and request_stats.REASONS[-1] == request_stats.OTHER
    assert request_stats.fold("sync_in_progress", request_stats.REASONS) == "sync_in_progress"
    assert request_stats.fold("sync_in_progress ", request_stats.REASONS) == request_stats.OTHER     # still closed


@pytest.mark.parametrize("coverage,expected", [
    ({"complete": False, "issues": ["sync_in_progress"]}, "sync_in_progress"),
    ({"complete": False, "issues": ["unread_bytes", "sync_in_progress"]}, "sync_in_progress"),
    ({"complete": False, "issues": ["sync_deadline"]}, "sync_deadline"),
    ({"complete": False, "issues": ["sync_in_progress", "sync_deadline"]}, "sync_deadline"),     # the older reason first
    ({"complete": False, "issues": ["unread_bytes"]}, None),
    ({"complete": True, "issues": []}, None),
    ({"issues": "sync_in_progress"}, None),                    # not a list of issues
    ({"issues": None}, None),
    (None, None),
    ("sync_in_progress", None),
], ids=["only", "with another issue", "sync_deadline", "both", "another issue", "complete", "a string", "none",
        "no coverage", "a string coverage"])
def test_a_result_whose_coverage_names_sync_in_progress_is_counted_degraded(stats, coverage, expected):
    answer = {"ok": True, "verdict": "unknown", "coverage": coverage}
    assert server._unfinished_sync(answer) == expected
    assert asyncio.run(server._dispatch(stand_in("audit_session", answer), budget_override=10)) == answer
    want = (1, {"degraded": 1}, {"degraded": {expected: 1}}) if expected else (1, {"ok": 1}, {})
    assert counted(stats.snapshot(), "audit_session") == want


def test_an_audit_or_check_that_met_a_busy_session_is_counted_degraded_sync_in_progress(session, monkeypatch, stats):
    engine, path = session
    monkeypatch.setattr(server, "_engine", engine)
    engine.upsert_contracts([NO_WIPE])
    leader = Leader(engine, path, monkeypatch)
    try:
        audited = asyncio.run(server.audit_session(SID, timeout_ms=2000))
        action = ProposedAction(tool_name="shell", arguments={"command": "rm -rf /fictional-path"})
        checked = asyncio.run(server.check_action(SID, action, timeout_ms=2000))
    finally:
        leader.finish()
    assert audited["verdict"] == "unknown" and "sync_in_progress" in audited["coverage"]["issues"]
    assert checked["verdict"] == "violation" and "sync_in_progress" in checked["coverage"]["issues"]
    snapshot = stats.snapshot()
    assert counted(snapshot, "audit_session") == (1, {"degraded": 1}, {"degraded": {"sync_in_progress": 1}})
    assert counted(snapshot, "check_action") == (1, {"degraded": 1}, {"degraded": {"sync_in_progress": 1}})


def test_the_engine_tells_the_request_that_the_session_was_being_synced(session, monkeypatch):
    """Not only the coverage: the engine's own note, which reaches the dispatcher of the request that asked."""
    engine, path = session
    engine.upsert_contracts([NO_WIPE])
    leader = Leader(engine, path, monkeypatch)
    notes = request_stats.Notes()
    try:
        with request_stats.collecting(notes):
            engine.check_action(SID, {"tool_name": "shell", "arguments": {"command": "echo harmless"}},
                                _deadline=time.monotonic() + 2.0)
    finally:
        leader.finish()
    assert notes.degraded == "sync_in_progress"


def test_a_sync_session_that_met_a_busy_session_is_counted_incomplete_sync_in_progress(session, monkeypatch, stats):
    engine, path = session
    monkeypatch.setattr(server, "_engine", engine)
    leader = Leader(engine, path, monkeypatch)
    try:
        answer = asyncio.run(server.sync_session(session_id=SID, embed=False, timeout_ms=800))
    finally:
        leader.finish()
    assert answer["ok"] is False and answer["error"] == "sync_in_progress" and answer["verdict"] == "unknown"
    assert counted(stats.snapshot(), "sync_session") == (1, {"incomplete": 1}, {"incomplete": {"sync_in_progress": 1}})


def test_a_partial_sync_is_counted_degraded_sync_in_progress_and_the_one_that_finishes_it_ok(session, monkeypatch, stats):
    engine, path = session
    monkeypatch.setattr(server, "_engine", engine)
    write(path, [message(f"event {n}") for n in range(40)], "a")
    sync = dict(session_id=SID, embed=False, derived=False, budget_override=600, heavy=False)
    engine.slice_policy = forced(10)
    first = asyncio.run(server._dispatch(engine.sync_session, **sync))
    assert first["sync"]["state"] == "partial" and "sync_in_progress" in first["coverage"]["issues"]
    engine.slice_policy = None
    second = asyncio.run(server._dispatch(engine.sync_session, **sync))
    assert second["sync"]["state"] == "complete" and "sync_in_progress" not in second["coverage"]["issues"]
    assert counted(stats.snapshot(), "sync_session") == (
        2, {"degraded": 1, "ok": 1}, {"degraded": {"sync_in_progress": 1}})
