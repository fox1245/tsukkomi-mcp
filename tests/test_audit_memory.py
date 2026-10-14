"""Audit loading is scoped and prerequisite lookups avoid repeated prefix copying."""
from __future__ import annotations

import re
import tracemalloc

import pytest

from self_directing_mcp.audit import runner as audit_runner
from self_directing_mcp.schemas import Chunk, ContractRule
from test_reliability import SID, call, message, result, write, session  # noqa: F401

EVENTS = 1500
BULK = "x" * 3000


def _history(path):
    events = []
    for i in range(EVENTS):
        events.append(message(f"assistant narration {i} {BULK}"))
        events.append(call(f"echo step {i}", f"c{i}"))
        events.append(result(f"c{i}"))
    write(path, events, "a")
    return sum(len(f"assistant narration {i} {BULK}") for i in range(EVENTS))


def test_audit_memory_is_bounded_by_what_the_contracts_need(session):
    engine, path = session
    engine.settings.audit_shadow_every = 0  # the shadow comparison deliberately holds the whole event list
    bulk_bytes = _history(path)
    engine.upsert_contracts([{"id": "no-delete", "type": "must_not", "scope": "tool_call", "regex": "DELETE"}])
    assert engine.audit_session(SID)["verdict"] == "clean"  # sync + warm-up outside the measurement
    tracemalloc.start()
    try:
        value = engine.audit_session(SID)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert value["verdict"] == "clean" and value["coverage"]["chunk_count"] >= EVENTS * 3
    # The contract inspects only short tool calls; the bulky narration must not be materialized.
    assert peak < 0.8 * bulk_bytes, (peak, bulk_bytes)


def test_slim_evaluation_matches_full_chunks(session):
    engine, path = session
    write(path, [call("npm test", "t1"), result("t1", 1), call("deploy now", "d1")], "a")
    engine.sync_session(session_id=SID, embed=False)
    full = engine.store.list_chunks(SID, "codex")
    rule = ContractRule(id="gate", type="must", scope="tool_call", regex="npm test", before_regex="deploy",
                        requires_success=True)
    want = audit_runner.audit_contract(rule, session_id=SID, store=engine.store, provider="codex", chunks=full)
    needs, links = audit_runner.audit_needs([rule], None)
    slim = engine.store.capture_audit_events(SID, "codex", needs, link_calls=links)
    got = audit_runner.audit_contract(rule, session_id=SID, store=slim, provider="codex")
    assert got.model_dump() == want.model_dump() and want.verdict == "violation"


def _chunks(n, success):
    out = []
    for i in range(n):
        out.append(Chunk(chunk_id=f"a{i}", session_id="s", provider="omp", kind="tool_call", text=f"npm test {i}",
                         content_hash=f"h{i}", meta={"role": "assistant", "call_id": f"c{i}"}))
        out.append(Chunk(chunk_id=f"r{i}", session_id="s", provider="omp", kind="tool_result", text="ok",
                         content_hash=f"g{i}", meta={"role": "tool", "call_id": f"c{i}", "success": success}))
        out.append(Chunk(chunk_id=f"d{i}", session_id="s", provider="omp", kind="tool_call", text=f"deploy {i}",
                         content_hash=f"d{i}", meta={"role": "assistant", "call_id": f"x{i}"}))
    return out


@pytest.mark.parametrize("success", [True, False])
def test_presence_scan_is_linear(monkeypatch, success):
    rule = ContractRule(id="gate", type="must", scope="tool_call", regex="npm test", before_regex="deploy",
                        requires_success=True)
    steps = []
    real = audit_runner.check_request
    monkeypatch.setattr(audit_runner, "check_request", lambda: (steps.append(1), real())[1])
    counts = {}
    for n in (100, 200, 400):
        steps.clear()
        audit_runner.audit_contract(rule, session_id="s", store=None, provider="omp", chunks=_chunks(n, success))
        counts[n] = len(steps)
    assert counts[400] < 4.5 * counts[100], counts
    assert counts[400] < 2.4 * counts[200], counts


class _CountCopiedItems(list):
    """Count list-slice work that check_request calls cannot observe."""

    def __init__(self, values):
        super().__init__(values)
        self.copied_items = 0
        self.largest_slice = 0

    def __getitem__(self, key):
        if isinstance(key, slice):
            count = len(range(*key.indices(len(self))))
            self.copied_items += count
            self.largest_slice = max(self.largest_slice, count)
        return super().__getitem__(key)


def _presence(chunks, *, requires_success=True):
    rule = ContractRule(id="gate", type="must", scope="tool_call", regex="npm test",
                        requires_success=requires_success)
    return audit_runner._Presence(rule, re.compile(rule.regex), chunks)


@pytest.mark.parametrize("success", [True, False])
def test_presence_checkpoints_copy_only_bounded_hit_evidence(success):
    # Instrument actual slice sizes: the former prefix copy grew quadratically
    # even though its check_request count grew only linearly.
    counts = {}
    for n in (64, 128, 256):
        presence = _presence(_chunks(n, success))
        hits = presence.hits = _CountCopiedItems(presence.hits)
        for i in range(n):
            verdict, _, supporting = presence.before(3 * i + 2)
            assert verdict == ("clean" if success else "violation")
            assert len(supporting) <= 3
        assert hits.largest_slice <= 3
        counts[n] = hits.copied_items
        assert counts[n] <= 3 * n
    assert counts[256] <= 4 * counts[64] + 9, counts


def test_presence_duplicate_results_do_not_copy_result_prefixes():
    # One call with many duplicate result rows is ambiguous, not a reason to
    # materialize all earlier rows for every later checkpoint.
    counts = {}
    for n in (64, 128, 256):
        chunks = [_chunks(1, True)[0]]
        chunks.extend(Chunk(chunk_id=f"r{i}", session_id="s", provider="omp", kind="tool_result",
                            text="ok", content_hash=f"g{i}",
                            meta={"role": "tool", "call_id": "c0", "success": True})
                      for i in range(n))
        presence = _presence(chunks)
        outputs = presence.outputs["c0"] = _CountCopiedItems(presence.outputs["c0"])
        for position in range(2, len(chunks) + 1):
            verdict, _, supporting = presence.before(position)
            assert verdict == ("clean" if position == 2 else "unknown")
            assert len(supporting) <= 2
        assert outputs.largest_slice <= 3
        counts[n] = outputs.copied_items
        assert counts[n] <= 3 * n
    assert counts[256] <= 4 * counts[64] + 9, counts


def test_presence_boundary_and_latest_completion_semantics():
    chunks = _chunks(3, True)
    chunks[4].meta["success"] = False
    presence = _presence(chunks)
    expected = {
        0: ("violation", []),  # The first call itself is outside the prefix.
        1: ("unknown", ["a0"]),  # Its result is still outside the prefix.
        2: ("clean", ["a0", "r0"]),
        3: ("clean", ["a0", "r0"]),  # A future failed call cannot invalidate it.
        4: ("unknown", ["a0", "a1"]),  # Latest call is pending.
        5: ("violation", ["a0", "a1"]),  # Latest failure overrides older success.
        7: ("unknown", ["a0", "a1", "a2"]),
        8: ("clean", ["a2", "r2"]),  # Latest success overrides older failure.
    }
    for position, (verdict, ids) in expected.items():
        got, _, supporting = presence.before(position)
        assert (got, [item.chunk_id for item in supporting]) == (verdict, ids)


@pytest.mark.parametrize("ambiguous", ["missing_id", "duplicate_call", "duplicate_result", "unknown_result"])
def test_presence_ambiguous_latest_call_preserves_bounded_evidence(ambiguous):
    chunks = _chunks(8, True)
    if ambiguous == "missing_id":
        chunks[-3].meta.pop("call_id")
    elif ambiguous == "duplicate_call":
        chunks[-1].meta["call_id"] = "c7"
    elif ambiguous == "duplicate_result":
        chunks.append(chunks[-2].model_copy(update={"chunk_id": "duplicate-r7"}))
    else:
        chunks[-2].meta["success"] = None
    verdict, _, supporting = _presence(chunks).before(len(chunks))
    assert verdict == "unknown"
    assert [item.chunk_id for item in supporting] == ["a5", "a6", "a7"]


def test_presence_without_success_uses_latest_in_prefix_only():
    chunks = _chunks(8, False)
    presence = _presence(chunks, requires_success=False)
    assert presence.before(0) == ("violation", "required evidence missing before checkpoint", [])
    for i in range(8):
        verdict, reason, supporting = presence.before(3 * i + 1)
        assert (verdict, reason) == ("clean", "required regex evidenced")
        assert [item.chunk_id for item in supporting] == [f"a{i}"]
