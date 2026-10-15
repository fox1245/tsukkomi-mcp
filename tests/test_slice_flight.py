"""One writer per session, and a bounded wait for everyone else (roundtable 8, P3).

A session is synced by one request at a time (``ingest.session_flight``). A request that may slice does not sit on a
heavy-call slot behind the leader: it waits at most ``min(1 s, a quarter of the time it has left)`` and then judges on
what is committed, with the issue ``sync_in_progress``. Without slicing the wait is what it always was: the waiter's own
deadline. Nobody parses twice.
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from self_directing_mcp import server
from self_directing_mcp.codex import parse as codex_parser
from self_directing_mcp.index import ingest
from self_directing_mcp.index.ingest import FlightBusy, SlicePolicy
from self_directing_mcp.request_control import IndexBusy, RequestControl, RequestStopped, request_scope
from slicing_support import forced
from test_slice_equivalence import BIG
from test_reliability import SID, call, message, write, session as reliability_session  # noqa: F401


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
    answer = engine.audit_session(SID, _deadline=started + 1.2)         # waits 0.25 * 1.2 s = 0.3 s for the leader
    waited = time.monotonic() - started
    try:
        assert 0.25 < waited < 1.0                                       # bounded: well inside its own deadline
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
                                     _deadline=started + 1.2)
        assert answer["verdict"] == "violation" and time.monotonic() - started < 1.0
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
    with request_scope(RequestControl(deadline=time.monotonic() + 0.4)):
        assert 0.05 < ingest._flight_wait(slicing, path_) <= 0.1
    assert ingest._flight_wait(slicing, path_) is None                      # no deadline: wait for the leader, as before
    with request_scope(RequestControl(deadline=time.monotonic() + 60)):
        assert ingest._flight_wait(SlicePolicy(mode="off"), path_) is None  # slicing off: as before
        assert ingest._flight_wait(None, path_) is None
        assert ingest._flight_wait(slicing, path_.with_name(path_.name + ".zst")) is None   # never cut: never short-waited


def test_with_slicing_off_a_waiter_still_waits_for_its_own_deadline(session, monkeypatch):
    engine, path = session
    engine.settings.sync_slicing = "off"
    leader = Leader(engine, path, monkeypatch)
    try:
        started = time.monotonic()
        with pytest.raises(RequestStopped) as stopped:
            engine.audit_session(SID, _deadline=started + 0.5)
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
        assert answer["coverage"]["issues"].count("sync_in_progress") == 1 and laps[0] < 1.0     # 0.25 * 2 s, not 2 s
        assert running_after == 0                                  # released as soon as the call returned
    finally:
        leader.finish()
