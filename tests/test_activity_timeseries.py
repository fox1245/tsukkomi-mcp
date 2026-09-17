"""Issue #4 regression: SQLite activity time-series analysis."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.checklist import ChecklistStore
from self_directing_mcp.index.ingest import SessionStore
from self_directing_mcp.schemas import Chunk
from self_directing_mcp.timeseries import analyze_activity


SID = "55555555-6666-7777-8888-999999999999"


def _chunk(kind, text, *, order=0, success=None, tool="shell", ts=None, error=None):
    return Chunk(chunk_id=f"c{order:03d}", session_id=SID, provider="codex", kind=kind,
                 text=text, content_hash=f"h{order}", timestamp=ts,
                 meta={"role": "assistant", "tool_name": tool, "success": success, "error": error})


@pytest.fixture
def store(tmp_path: Path):
    return SessionStore(tmp_path / "meta.sqlite")


def test_repeated_errors_with_retry_gaps(store):
    rows = [
        _chunk("tool_call", "[tool_call:shell] pytest", order=0),
        _chunk("tool_result", "ConnectionError timeout", order=1, success=False, error="timeout"),
        _chunk("tool_call", "[tool_call:shell] pytest again", order=2),
        _chunk("tool_result", "ConnectionError timeout", order=3, success=False, error="timeout"),
        _chunk("tool_call", "[tool_call:shell] pytest retry", order=4),
        _chunk("tool_result", "ok 3 passed", order=5, success=True),
    ]
    store.commit_events(rows, session_id=SID, provider="codex", path=Path("x.jsonl"), offset=10, digest="d")
    result = analyze_activity(store, session_id=SID, provider="codex")
    assert result["event_count"] == 6
    repeats = result["repeated_errors"]
    assert len(repeats) == 1
    assert repeats[0]["count"] == 2
    assert repeats[0]["retry_gaps_in_events"] == [2]
    assert "c001" in repeats[0]["evidence_chunk_ids"]
    chains = result["fix_reverify_sequences"]
    assert chains[0]["failed_event"] == "c001"
    assert chains[0]["reverified_ok"] == "c005"


def test_missing_timestamps_flagged_not_fabricated(store):
    rows = [
        _chunk("tool_result", "boom", order=0, success=False, error="e1", ts=None),
        _chunk("tool_result", "boom", order=1, success=False, error="e1", ts=None),
    ]
    store.commit_events(rows, session_id=SID, provider="codex", path=Path("x"), offset=5, digest="d")
    result = analyze_activity(store, session_id=SID, provider="codex")
    assert result["data_complete"] is False
    assert any("timestamps" in n for n in result["notes"])
    assert result["failure_rate_by_window"] == []


def test_failure_rate_windows_only_timestamped(store):
    rows = [
        _chunk("tool_result", "ok", order=0, success=True, ts="2026-09-10T00:00:00Z"),
        _chunk("tool_result", "fail", order=1, success=False, error="x", ts="2026-09-10T00:30:00Z"),
        _chunk("tool_result", "no-time", order=2, success=False, error="x", ts=None),
    ]
    store.commit_events(rows, session_id=SID, provider="codex", path=Path("x"), offset=5, digest="d")
    result = analyze_activity(store, session_id=SID, provider="codex")
    assert result["data_complete"] is False
    total = sum(w["tool_results"] for w in result["failure_rate_by_window"])
    assert total == 2  # timestamped only


def test_empty_session_reports_incomplete(store):
    result = analyze_activity(store, session_id="missing", provider="codex")
    assert result["data_complete"] is False
    assert any("sync_session" in n for n in result["notes"])


def test_stalled_requirements_with_checklist_items(tmp_path):
    """Regression: stalled items listed (not crashed) when checklist has open items."""
    from self_directing_mcp.checklist import ChecklistStore
    store = SessionStore(tmp_path / "meta.sqlite")
    cl = ChecklistStore(tmp_path)
    doc = cl.load(SID)
    from self_directing_mcp.checklist import RequirementItem
    doc.items.append(RequirementItem(item_id="req-001", text="A"))
    doc.items.append(RequirementItem(item_id="req-002", text="done thing", status="done",
                                     evidence_chunk_ids=["c9"]))
    cl.save(doc)
    rows = [Chunk(chunk_id="c000", session_id=SID, provider="codex", kind="tool_call",
                  text="x", content_hash="h0", meta={"role": "assistant"})]
    store.commit_events(rows, session_id=SID, provider="codex",
                        path=tmp_path / "s.jsonl", offset=5, digest="d")
    result = analyze_activity(store, session_id=SID, provider="codex")
    assert len(result["stalled_requirements"]) == 1
    assert result["stalled_requirements"][0]["item_id"] == "req-001"


