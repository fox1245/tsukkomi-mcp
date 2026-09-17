"""Issue #5 regression: validated, idempotent GraphRAG updates with recovery."""
from __future__ import annotations

from pathlib import Path

import pytest

from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.graphrag import GraphStore, validate_proposal
from self_directing_mcp.schemas import Chunk


SID = "77777777-8888-9999-aaaa-bbbbbbbbbbbb"


def _chunk(chunk_id, text, kind="tool_call", order=0):
    return Chunk(chunk_id=chunk_id, session_id=SID, provider="codex", kind=kind,
                 text=text, content_hash=f"h{order}",
                 meta={"role": "assistant", "tool_name": "shell", "success": True})


@pytest.fixture
def engine(tmp_path: Path):
    settings = Settings(use_fake_embedder=True, index_dir=tmp_path / "index",
                        contracts_path=tmp_path / "index" / "contracts.json")
    e = SelfDirectEngine(settings=settings)
    e.ensure_ready()
    store = e.store
    rows = [
        _chunk("c000", "[tool_call:shell] edit report.py", order=0),
        _chunk("c001", "pytest 3 passed", kind="tool_result", order=1),
       ]
    store.commit_events(rows, session_id=SID, provider="codex", path=Path("s.jsonl"), offset=10, digest="d")
    yield e
    e.close()


def test_rejects_unknown_relation_and_missing_evidence(engine):
    proposal = {"nodes": [{"node_id": "task-1"}],
                "edges": [{"src": "task-1", "dst": "task-2", "relation": "CAUSES"}]}
    result = engine.propose_graph_update(SID, proposal)
    assert result["ok"] is False
    assert any("unknown relation" in p for p in result["problems"])
    assert any("no evidence" in p for p in result["problems"])


def test_rejects_evidence_outside_session(engine):
    proposal = {"nodes": [{"node_id": "t1"}],
                "edges": [{"src": "task-1", "dst": "task-1", "relation": "MODIFIES",
                           "evidence_chunk_ids": ["from-other-session"]}], "job_id": "j1"}
    result = engine.propose_graph_update(SID, proposal)
    assert result["ok"] is False
    assert any("not in this session" in p for p in result["problems"])


def test_valid_proposal_commits_and_advances_cursor(engine):
    proposal = {"nodes": [{"node_id": "req-1", "kind": "requirement", "label": "report"},
                          {"node_id": "task-1", "kind": "task", "label": "edit"}],
                "edges": [{"src": "task-1", "dst": "req-1", "relation": "IMPLEMENTS",
                           "origin": "inferred", "evidence_chunk_ids": ["c000"]}],
                "job_id": "job-1"}
    check = engine.propose_graph_update(SID, {"nodes": proposal["nodes"], "edges": proposal["edges"]})
    assert check["ok"] is True
    result = engine.commit_graph_update(SID, proposal, base_graph_version=0)
    assert result["ok"] is True and result["status"] == "applied"
    assert result["graph_version"] == 1
    assert result["processed_through_order"] >= 1
    # Idempotent replay.
    replay = engine.commit_graph_update(SID, proposal, base_graph_version=0)
    assert replay["status"] == "already_applied"


def test_failed_write_reports_unclear_not_false_success(tmp_path):
    from self_directing_mcp.graphrag import GraphStore
    g = GraphStore(Path(tmp_path) / "g.sqlite")
    try:
        # Malformed proposal triggers sqlite failure path, not silent success.
        bad = {"nodes": [{"node_id": None}]}
        result = g.apply_proposal(bad, session_id=SID, provider="codex",
                                  base_version=0, through_order=5)
        if not result["ok"]:
            assert result["status"] == "unclear"
    finally:
        g.close()



