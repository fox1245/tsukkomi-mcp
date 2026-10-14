"""An audit loads only the rows its contracts need and scans prefixes linearly."""
from __future__ import annotations

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
