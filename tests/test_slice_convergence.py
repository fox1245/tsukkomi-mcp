"""Repeating a call that cannot finish now makes progress (roundtable 8, P3/P4/P7), what a partial sync may and may not
say, and the switches that restore the all-or-nothing sync.

* Convergence. A session of about 50,000 events and per-call budgets that force many slices: ``indexed_through`` strictly
  increases, every call returns inside its budget, the sync completes in at most ceil(M / q) + 1 calls for the smallest
  slice q that was observed, and the index equals the one-shot index. The same calls with slicing off make no progress.
* Verdicts. A partial prefix never yields ``clean`` (and never ``judged_prefix``), a forbidden call in it is a violation,
  a proposed prohibited action is a violation whatever the backlog, an obligation stays unknown.
* Switches. ``off``, no finite deadline, a compressed source, and every reindex keep the all-or-nothing behaviour.
"""
from __future__ import annotations

import json
import math
import os
import time

import pytest

from self_directing_mcp import session_events
from self_directing_mcp.codex import parse as codex_parser
from self_directing_mcp.index.ingest import SlicePolicy
from self_directing_mcp.request_control import RequestControl, RequestStopped, request_scope
from slicing_support import Tick, World, forced, integrity_ok, snapshot
from test_append_capture import _AppendOnFirstFstat
from test_reliability import SID, call, message, session as reliability_session, write  # noqa: F401
from test_slice_equivalence import BIG

FAR = lambda: time.monotonic() + BIG    # noqa: E731


class Allowance(RequestControl):
    """A request that is over (``request_deadline``) after ``limit`` cooperative checks: a deadline that does not depend on
    the speed of the machine. Its real deadline is a few seconds away, which is what the soft deadline is computed from."""

    def __init__(self, limit, seconds=3.0):
        super().__init__(deadline=time.monotonic() + seconds)
        self.calls, self.limit = 0, limit

    def _check(self):
        self.calls += 1
        if self.calls > self.limit:
            raise RequestStopped("request_deadline")
        super()._check()


class Tripwire(RequestControl):
    """A request that is stopped (``request_deadline``) at the first cooperative check after ``arm()``: the way to interrupt
    a reindex halfway through its parse without counting checks."""

    def __init__(self):
        super().__init__(deadline=time.monotonic() + BIG)
        self.armed = False

    def _check(self):
        if self.armed:
            raise RequestStopped("request_deadline")
        super()._check()


def watch_captures(monkeypatch, control):
    """Record the ``window`` of every capture the sync takes; arm ``control`` once a capture was taken whole."""
    windows, original = [], session_events.capture_source

    def spy(*args, **kwargs):
        windows.append(kwargs.get("window"))
        observation = original(*args, **kwargs)
        if kwargs.get("window") is None:
            control.armed = True
        return observation

    monkeypatch.setattr("self_directing_mcp.index.ingest.capture_source", spy)
    return windows


def big_world(tmp_path, monkeypatch, **options):
    return World(tmp_path, monkeypatch, "codex", seed=3, count=42000, partial=False, **options)


# --- convergence -----------------------------------------------------------------------------------------------------

def test_repeated_calls_that_made_no_progress_now_converge(tmp_path, monkeypatch):
    """Four calls that each run out of time. Before: nothing is kept, and the fifth starts from zero. Now: each keeps a slice."""
    world = big_world(tmp_path, monkeypatch)
    try:
        sizes = {}
        for mode in ("off", "auto"):
            engine = world.engine(mode, sync_slicing=mode)
            engine.slice_policy = SlicePolicy(mode=mode, clock=Tick())
            offsets = []
            for _ in range(4):
                try:
                    with request_scope(Allowance(60_000)):
                        engine.sync_session(**world.sync_args(), derived=False)
                except RequestStopped as stopped:
                    assert mode == "off" and stopped.reason == "request_deadline"
                record = engine.store.cursor_record(world.sid, "codex")
                offsets.append(record["byte_offset"] if record else None)
            sizes[mode] = (offsets, engine.store.chunk_count(world.sid, "codex"))
        assert sizes["off"] == ([None] * 4, 0)                                 # four calls, no cursor, no event
        offsets, chunks = sizes["auto"]
        assert all(offsets) and offsets == sorted(set(offsets)) and chunks > 0   # four calls, four slices, ever further
    finally:
        world.close()


def test_a_session_of_50000_events_converges_in_few_calls(tmp_path, monkeypatch):
    world = big_world(tmp_path, monkeypatch)
    try:
        one, sliced = world.engine("one-shot"), world.engine("sliced")
        for engine in (one, sliced):
            world.upsert_contracts(engine)
        one.sync_session(**world.sync_args(), derived=False)
        expected = snapshot(one, "codex", world.sid)
        assert len(expected.events) > 45_000
        size = world.path.stat().st_size
        tick, reports, previous = Tick(0.3), [], 0
        for _ in range(60):
            sliced.slice_policy = SlicePolicy(clock=tick)       # the production minimums; time moves 0.3 s per look
            started = time.monotonic()
            info = sliced.sync_session(**world.sync_args(), derived=False, _deadline=time.monotonic() + 3.0)
            elapsed = time.monotonic() - started
            sync = info["sync"]
            assert elapsed < 3.0 + 0.5                          # inside the budget, with a stated tolerance
            assert sync["indexed_through"] > previous           # strictly further every call
            assert sync["target_bytes"] == size and 0 < sync["progress"] <= 1
            previous = sync["indexed_through"]
            reports.append(sync)
            if sync["state"] == "complete":
                break
        partial = reports[:-1]
        assert len(reports) >= 5 and reports[-1]["state"] == "complete" and reports[-1]["progress"] == 1.0
        assert all(r["state"] == "partial" and r["stopped_by"] == "soft_deadline" and r["retry_after_ms"] == 500
                   for r in partial)
        assert [r["progress"] for r in partial] == sorted(r["progress"] for r in partial)
        offsets = [0] + [r["indexed_through"] for r in reports]
        q = min(b - a for a, b in zip(offsets, offsets[1:-1]))   # the smallest slice that was not the last
        assert q >= 1 << 20                                      # the minimum progress of a slice (1 MiB)
        assert len(reports) <= math.ceil(size / q) + 1
        done = snapshot(sliced, "codex", world.sid)
        assert (done.events, done.cursor, done.blocks[:2] + done.blocks[3:], done.metrics) == \
            (expected.events, expected.cursor, expected.blocks[:2] + expected.blocks[3:], expected.metrics)
        sliced.slice_policy = None
        audit = sliced.audit_session(world.sid, provider="codex", path=str(world.path), _deadline=FAR())
        reference = one.audit_session(world.sid, provider="codex", path=str(world.path))
        assert audit["verdict"] == reference["verdict"] == "violation"
        assert [(f["contract_id"], f["verdict"]) for f in audit["findings"]] == \
            [(f["contract_id"], f["verdict"]) for f in reference["findings"]]
        assert integrity_ok(sliced.settings.index_dir)
    finally:
        world.close()


def test_real_time_budgets_make_progress_and_return_inside_the_budget(tmp_path, monkeypatch):
    """The same convergence on the real clock (no counting tricks): only what any machine must satisfy is asserted."""
    world = World(tmp_path, monkeypatch, "codex", seed=5, count=20000, partial=False)
    try:
        engine = world.engine("real")
        budget, previous, calls = 3.0, 0, 0
        while True:
            calls += 1
            started = time.monotonic()
            info = engine.sync_session(**world.sync_args(), derived=False, _deadline=started + budget)
            assert time.monotonic() - started < budget + 0.25
            assert info["sync"]["indexed_through"] > previous
            previous = info["sync"]["indexed_through"]
            if info["sync"]["state"] == "complete":
                break
            assert calls < 40
        assert previous == world.path.stat().st_size and info["coverage"]["complete"] is False   # parse errors in the mix
    finally:
        world.close()


# --- verdicts --------------------------------------------------------------------------------------------------------

CONTRACTS_TEXT = [{"id": "no-wipe", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"},
                  {"id": "needs-test", "type": "must", "scope": "tool_call", "regex": "npm test"}]


def partial_session(session, first, rest=300):
    """A session whose first slice holds ``first`` and whose remainder is a backlog; returns (engine, path)."""
    engine, path = session
    write(path, [*first, *[message(f"filler {n}") for n in range(rest)]], "a")
    engine.upsert_contracts(CONTRACTS_TEXT)
    engine.slice_policy = forced(len(first) + 5)
    info = engine.sync_session(session_id=SID, embed=False, derived=False, _deadline=FAR())
    assert info["sync"]["state"] == "partial" and info["coverage"]["backlog_bytes"] > 0
    return engine, path


@pytest.fixture
def session(reliability_session):          # noqa: F811
    engine, path = reliability_session
    try:
        yield engine, path
    finally:
        engine.close()


def judged(engine, answer):
    cov = answer["coverage"]
    assert cov["complete"] is False and "sync_in_progress" in cov["issues"] and cov["backlog_bytes"] > 0
    assert not cov.get("judged_prefix")          # the one thing a backlog must never earn
    return answer


def test_a_prohibited_proposed_action_is_a_violation_whatever_the_backlog(session):
    engine, _path = partial_session(session, [call("echo ok")])
    answer = judged(engine, engine.check_action(SID, {"tool_name": "shell", "arguments": {"command": "rm -rf /data"}},
                                                _deadline=FAR()))
    assert answer["verdict"] == "violation" and answer["requires_attention"] is True


def test_a_harmless_proposed_action_is_unknown_not_clean_while_there_is_a_backlog(session):
    engine, _path = partial_session(session, [call("echo ok")])
    answer = judged(engine, engine.check_action(SID, {"tool_name": "shell", "arguments": {"command": "ls"}},
                                                _deadline=FAR()))
    assert answer["verdict"] == "unknown"
    assert {f["contract_id"]: f["verdict"] for f in answer["findings"]}["no-wipe"] == "clean"   # the finding, not the verdict


def test_an_audit_of_a_partial_prefix_is_never_clean(session):
    engine, _path = partial_session(session, [call("echo ok")])
    answer = judged(engine, engine.audit_session(SID, _deadline=FAR()))
    assert answer["verdict"] == "unknown"
    verdicts = {f["contract_id"]: f["verdict"] for f in answer["findings"]}
    assert verdicts["no-wipe"] == "unknown"             # the absence of a forbidden call is not proven for the rest


def test_a_forbidden_call_inside_the_committed_prefix_is_a_violation(session):
    engine, _path = partial_session(session, [call("rm -rf build", "c1"), call("echo ok", "c2")])
    answer = judged(engine, engine.audit_session(SID, _deadline=FAR()))
    findings = {f["contract_id"]: f["verdict"] for f in answer["findings"]}
    assert answer["verdict"] == "violation" and findings["no-wipe"] == "violation"


def test_an_obligation_that_would_be_missing_is_unknown_not_violated(session):
    engine, path = partial_session(session, [call("echo ok")])
    answer = judged(engine, engine.audit_session(SID, _deadline=FAR()))
    assert {f["contract_id"]: f["verdict"] for f in answer["findings"]}["needs-test"] == "unknown"
    # The remainder arrives: the obligation is judged for good (here it is in the part that was not indexed yet).
    write(path, [call("npm test", "late")], "a")
    engine.slice_policy = None
    final = engine.audit_session(SID, _deadline=FAR())
    assert final["verdict"] == "clean" and final["coverage"].get("backlog_bytes") is None


def test_an_obligation_met_in_the_prefix_is_still_only_unknown_overall(session):
    """Evidenced in the prefix is not enough for 'clean as of the cursor': the other contract has a backlog."""
    engine, _path = partial_session(session, [call("npm test", "c1")])
    answer = judged(engine, engine.audit_session(SID, _deadline=FAR()))
    findings = {f["contract_id"]: f["verdict"] for f in answer["findings"]}
    assert findings["needs-test"] == "clean" and answer["verdict"] == "unknown"


def test_growth_after_capture_still_yields_judged_prefix_while_slicing_is_on(session, monkeypatch):
    """The old case: a source that grows while it is captured. There is no backlog, so the verdict is clean as of the offset."""
    engine, path = session
    engine.upsert_contracts(CONTRACTS_TEXT[:1])
    write(path, [call("echo safe")], "a")
    size = path.stat().st_size
    monkeypatch.setattr(session_events, "os", _AppendOnFirstFstat(path, (json.dumps(call("rm -rf x", "late")) + "\n").encode()))
    answer = engine.audit_session(SID, _deadline=FAR())
    monkeypatch.setattr(session_events, "os", os)
    cov = answer["coverage"]
    assert answer["verdict"] == "clean" and cov["judged_prefix"] is True and cov["judged_through_byte"] == size
    assert cov["unread_bytes"] > 0 and "sync_in_progress" not in cov["issues"] and "backlog_bytes" not in cov
    assert engine.audit_session(SID, _deadline=FAR())["verdict"] == "violation"      # the next audit reads the rest


def test_a_partial_sync_says_what_it_is(session):
    engine, _path = partial_session(session, [call("echo ok")])
    info = engine.sync_session(session_id=SID, embed=False, derived=False, _deadline=FAR())
    sync, cov = info["sync"], info["coverage"]
    assert info["ok"] is True and sync["state"] == "partial" and sync["stopped_by"] == "record_cap"
    assert sync["indexed_through"] == cov["byte_offset"] < sync["target_bytes"]
    assert sync["pending_bytes"] == cov["pending_bytes"] and 0 < sync["progress"] < 1
    assert cov["complete"] is False and set(cov["issues"]) == {"unread_bytes", "sync_in_progress"}
    assert "retry_after_ms=500" in info["nudge"] and "nothing retries by itself" in info["nudge"]


# --- switches --------------------------------------------------------------------------------------------------------

def test_off_is_the_all_or_nothing_sync_of_before(tmp_path, monkeypatch):
    world = big_world(tmp_path, monkeypatch)
    try:
        engine = world.engine("off", sync_slicing="off")
        engine.slice_policy = None                         # the settings alone decide
        windows = []
        original = session_events.capture_source

        def spy(*args, **kwargs):
            windows.append(kwargs.get("window"))
            return original(*args, **kwargs)

        monkeypatch.setattr("self_directing_mcp.index.ingest.capture_source", spy)
        with pytest.raises(RequestStopped):
            with request_scope(Allowance(60_000)):
                engine.sync_session(**world.sync_args(), derived=False)
        assert engine.store.cursor_record(world.sid, "codex") is None and engine.store.chunk_count(world.sid, "codex") == 0
        info = engine.sync_session(**world.sync_args(), derived=False, _deadline=FAR())     # a deadline, but slicing is off
        assert "sync" not in info and "backlog_bytes" not in info["coverage"] and info["coverage"]["byte_offset"] > 0
        assert windows == [None, None]
    finally:
        world.close()


def test_without_a_finite_deadline_nothing_is_sliced(session, monkeypatch):
    engine, path = session
    write(path, [message(f"event {n}") for n in range(60)], "a")
    engine.slice_policy = forced(2)                            # a forced stop would cut it, if slicing were on
    info = engine.sync_session(session_id=SID, embed=False, derived=False)
    assert "sync" not in info and info["coverage"]["byte_offset"] == path.stat().st_size and info["total_chunks"] == 61


def test_a_compressed_source_is_never_sliced(tmp_path, monkeypatch):
    from test_zstd_stream import BACKENDS, make_frame
    if not BACKENDS:
        pytest.skip("needs Python 3.14+ or the zstandard package")
    world = World(tmp_path, monkeypatch, "codex", seed=2, count=300, partial=False)
    try:
        packed = world.path.with_name(world.path.name + ".zst")
        packed.write_bytes(make_frame(world.path.read_bytes()))
        world.path.unlink()
        engine = world.engine("zst")
        engine.slice_policy = forced(3, cap_bytes=1000)
        info = engine.sync_session(session_id=world.sid, provider="codex", path=str(packed), embed=False, derived=False,
                                   _deadline=FAR())
        assert info["sync"]["state"] == "complete" and info["sync"]["stopped_by"] == "eof"
        assert info["coverage"]["byte_offset"] > 1000 and "backlog_bytes" not in info["coverage"]
    finally:
        world.close()


def reindex_world(tmp_path, monkeypatch):
    """A fully indexed session (events to protect) of 1500 records."""
    world = World(tmp_path, monkeypatch, "codex", seed=8, count=1200, partial=False)
    engine = world.engine("protected")
    engine.sync_session(**world.sync_args(), derived=False)
    return world, engine, snapshot(engine, "codex", world.sid)


def test_a_parser_update_is_never_sliced_and_an_interrupted_reindex_keeps_the_old_index(tmp_path, monkeypatch):
    world, engine, before = reindex_world(tmp_path, monkeypatch)
    try:
        control = Tripwire()
        windows = watch_captures(monkeypatch, control)
        monkeypatch.setattr(codex_parser, "PARSER_VERSION", codex_parser.PARSER_VERSION + 1)
        engine.slice_policy = forced(5, cap_bytes=2000)
        with pytest.raises(RequestStopped):                   # the reindex runs out of time right after its capture
            with request_scope(control):
                engine.sync_session(**world.sync_args(), derived=False)
        after = snapshot(engine, "codex", world.sid)
        assert windows == [None]                              # the stored parser version is known: no cut capture at all
        assert (after.events, after.cursor, after.blocks, after.metrics, after.versions) == \
            (before.events, before.cursor, before.blocks, before.metrics, before.versions)      # nothing of it was kept
        info = engine.sync_session(**world.sync_args(), derived=False, _deadline=FAR())
        assert info["reindexed"] is True and info["sync"]["state"] == "complete"
        assert snapshot(engine, "codex", world.sid).versions != before.versions
    finally:
        world.close()


def test_a_changed_source_found_after_a_cut_capture_is_reindexed_whole(tmp_path, monkeypatch):
    """The capture is cut short before the cursor is compared with the file; a mismatch recaptures the file whole."""
    world, engine, before = reindex_world(tmp_path, monkeypatch)
    try:
        data = world.path.read_bytes()
        world.path.write_bytes(data[:100] + b"X" + data[101:] + b'{"type":"event_msg","payload":{"type":"agent_message","message":"more"}}\n' * 600)
        control = Tripwire()
        windows = watch_captures(monkeypatch, control)
        engine.slice_policy = forced(5, cap_bytes=2000)
        with pytest.raises(RequestStopped):
            with request_scope(control):
                engine.sync_session(**world.sync_args(), derived=False)
        assert windows[0] is not None and windows[1] is None  # cut, found changed, observed whole
        after = snapshot(engine, "codex", world.sid)
        assert (after.events, after.cursor, after.blocks, after.metrics) == \
            (before.events, before.cursor, before.blocks, before.metrics)
        info = engine.sync_session(**world.sync_args(), derived=False, _deadline=FAR())
        assert info["reindexed"] is True and info["reindex_reason"] == "source_changed" and info["sync"]["state"] == "complete"
    finally:
        world.close()


def test_an_append_to_an_indexed_session_may_be_sliced_and_keeps_the_events_it_has(tmp_path, monkeypatch):
    world, engine, before = reindex_world(tmp_path, monkeypatch)
    try:
        with world.path.open("ab") as stream:
            stream.write(b'{"type":"event_msg","payload":{"type":"agent_message","message":"more"}}\n' * 400)
        engine.slice_policy = forced(50)
        info = engine.sync_session(**world.sync_args(), derived=False, _deadline=FAR())
        assert info["sync"]["state"] == "partial" and info["reindexed"] is False
        kept = snapshot(engine, "codex", world.sid)
        assert kept.events[:len(before.events)] == before.events and len(kept.events) == len(before.events) + 50
        assert kept.generation == before.generation          # still the same history: nothing was rewritten
    finally:
        world.close()


def test_a_receipt_change_that_forces_a_reindex_is_never_sliced_either(tmp_path, monkeypatch):
    """AGY: the reindex is decided after the capture (a changed hook receipt of an indexed result), so a cut capture is
    discarded for a whole one, and an interrupted reindex leaves the old events in place."""
    from self_directing_mcp.agy_receipts import ReceiptStore
    world = World(tmp_path, monkeypatch, "agy", seed=6, count=90, partial=False)
    try:
        engine = world.engine("protected")
        world.copy_receipts(world.builder, engine)
        engine.sync_session(**world.sync_args(), derived=False)
        before = snapshot(engine, "agy", world.sid)
        top = max(json.loads(row[5])["meta"].get("step_index", -1) for row in before.events) + 1
        with world.path.open("ab") as stream:                        # more to index, so that the next capture is cut
            for step in range(top, top + 200):
                stream.write(json.dumps({"step_index": step, "source": "USER_EXPLICIT", "type": "USER_INPUT",
                                         "status": "DONE", "created_at": "2026-10-01T00:00:00Z",
                                         "content": "<USER_REQUEST>more</USER_REQUEST>"}).encode() + b"\n")
        rows = {step: json.loads(data) for step, data in ReceiptStore(engine.settings.index_dir).read_capture(world.sid)[0]}
        step, row = next((s, r) for s, r in sorted(rows.items()) if r.get("paired") and r.get("result_hash"))
        engine.record_agy_hook({"conversationId": world.sid, "transcriptPath": str(world.path), "workspacePaths": ["/synthetic"],
                                "stepIdx": step, "toolCall": json.loads(row["action"]), "error": "reported later"}, "PostToolUse")
        control = Tripwire()
        windows = watch_captures(monkeypatch, control)
        engine.slice_policy = forced(5, cap_bytes=3000)
        with pytest.raises(RequestStopped):
            with request_scope(control):
                engine.sync_session(**world.sync_args(), derived=False)
        assert windows[0] is not None and windows[-1] is None            # cut, found to need a reindex, observed whole
        after = snapshot(engine, "agy", world.sid)
        assert (after.events, after.cursor, after.metrics, after.agy) == (before.events, before.cursor, before.metrics, before.agy)
        info = engine.sync_session(**world.sync_args(), derived=False, _deadline=FAR())
        assert info["reindexed"] is True and info["reindex_reason"] == "hook_receipts_changed"
        assert info["sync"]["state"] == "complete"
    finally:
        world.close()
