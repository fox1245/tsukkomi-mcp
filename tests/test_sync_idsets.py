"""Sync touches only the synced session's ids."""
from __future__ import annotations

from self_directing_mcp.index.sparse import SparseIndex
from self_directing_mcp.index.vector import NumpyVectorIndex
from test_reliability import SID, message, write, session  # noqa: F401


def test_sync_does_not_load_other_sessions_ids(session, monkeypatch):
    engine, path = session
    other = "99999999-2222-3333-4444-555555555555"
    for i in range(3):
        other_path = path.with_name(f"rollout-2026-09-09T02-00-0{i}-{other[:-1]}{i}.jsonl")
        sid = f"{other[:-1]}{i}"
        write(other_path, [{"type": "session_meta", "payload": {"id": sid}}, message(f"other {i}")])
        engine.sync_session(session_id=sid, embed=True)
    write(path, [message("mine")], "a")

    def forbidden(self):
        raise AssertionError("whole-index id set loaded during sync")

    monkeypatch.setattr(SparseIndex, "ids", forbidden)
    monkeypatch.setattr(NumpyVectorIndex, "ids", forbidden)
    info = engine.sync_session(session_id=SID, embed=True)
    assert info["ok"] and info["dense_restored"] >= 1 and info["sparse_updated"] >= 1


def test_search_repair_does_not_load_other_sessions_ids(session, monkeypatch):
    engine, path = session
    write(path, [message("needle in the haystack")], "a")
    engine.sync_session(session_id=SID, embed=False)

    def forbidden(self):
        raise AssertionError("whole-index id set loaded during search")

    monkeypatch.setattr(SparseIndex, "ids", forbidden)
    result = engine.search_history("needle", session_id=SID, mode="sparse")
    assert result["hits"]
