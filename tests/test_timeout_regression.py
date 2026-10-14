from __future__ import annotations

import asyncio
import time

import pytest

from test_reliability import session, write, call, SID
from self_directing_mcp.request_control import RequestStopped


def test_unchanged_sync_preserves_sparse_search(session):
    engine, path = session
    write(path, [call("UNCHANGED_EVIDENCE")], "a")
    engine.sync_session(session_id=SID, embed=False)
    again = engine.sync_session(session_id=SID, embed=False)
    assert again["new_chunks"] == 0
    assert engine.search_history("UNCHANGED_EVIDENCE", session_id=SID, mode="sparse")["hits"]


def test_append_sparse_search_preserves_old_and_new_evidence(session):
    engine, path = session
    write(path, [call("OLD_EVIDENCE", "old")], "a")
    engine.sync_session(session_id=SID, embed=False)
    write(path, [call("NEW_EVIDENCE", "new")], "a")
    engine.sync_session(session_id=SID, embed=False)
    old = engine.search_history("OLD_EVIDENCE", session_id=SID, mode="sparse")["hits"]
    new = engine.search_history("NEW_EVIDENCE", session_id=SID, mode="sparse")["hits"]
    assert old and new
    assert {h["chunk_id"] for h in old}.isdisjoint(h["chunk_id"] for h in new)


def test_sparse_repair_after_interrupted_index_write(session, monkeypatch):
    engine, path = session
    engine.sync_session(session_id=SID, embed=False)
    original = engine.sparse.upsert_many
    def fail(chunks):
        raise RuntimeError("synthetic interrupted index write")
    monkeypatch.setattr(engine.sparse, "upsert_many", fail)
    write(path, [call("RECOVER_ME")], "a")
    with pytest.raises(RuntimeError):
        engine.sync_session(session_id=SID, embed=False)
    monkeypatch.setattr(engine.sparse, "upsert_many", original)
    engine.sync_session(session_id=SID, embed=False)
    assert engine.search_history("RECOVER_ME", session_id=SID, mode="sparse")["hits"]


def test_queued_engine_call_has_bounded_lock_wait(session):
    import threading
    engine, _ = session
    engine.ensure_ready()
    acquired = threading.Event()
    release = threading.Event()
    def hold():
        with engine._lock:
            acquired.set()
            release.wait(5)
    worker = threading.Thread(target=hold)
    worker.start()
    acquired.wait(2)
    start = time.monotonic()
    try:
        with pytest.raises(RequestStopped) as stopped:
            engine.audit_status(_deadline=time.monotonic() + 0.15)
        assert stopped.value.reason == "request_deadline"
        assert time.monotonic() - start < 0.7
    finally:
        release.set()
        worker.join()


def test_supplied_invalid_transcript_does_not_fall_back_to_history(session, tmp_path):
    engine, path = session
    engine.upsert_contracts([{"id": "no-delete", "type": "must_not", "regex": "DELETE"}])
    engine.sync_session(session_id=SID, embed=False)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("do not read", encoding="utf-8")
    assert engine.audit_session(SID, path=str(outside))["verdict"] == "unknown"


def test_expired_request_never_starts_work(session, monkeypatch):
    engine, _ = session
    def forbidden():
        raise AssertionError("expired request reached engine")
    monkeypatch.setattr(engine, "ensure_ready", forbidden)
    with pytest.raises(RequestStopped) as stopped:
        engine.audit_status(_deadline=time.monotonic() - 1)
    assert stopped.value.reason == "request_deadline"


def test_deadline_returns_unknown_not_compliance(session, monkeypatch):
    import threading
    from self_directing_mcp import server
    engine, _ = session
    engine.settings.audit_timeout_sec = 0.05
    monkeypatch.setattr(server, "_engine", engine)
    held, release = threading.Event(), threading.Event()
    def hold():
        with engine._lock:
            held.set()
            release.wait(3)
    thread = threading.Thread(target=hold)
    thread.start()
    held.wait(1)
    async def exercise():
        result = await server.audit_session(SID)
        assert result["error"] == "request_deadline"
        assert result["verdict"] == "unknown"
        assert result["coverage"]["complete"] is False
        assert result["action_executed"] is False
    try:
        asyncio.run(exercise())
        assert engine.store is None
    finally:
        release.set()
        thread.join()


def test_existing_fts_database_migrates_without_rewriting_documents(tmp_path):
    import sqlite3
    from self_directing_mcp.index.sparse import SparseIndex
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, session_id UNINDEXED, kind UNINDEXED, text, tokenize='porter unicode61')")
        connection.execute("INSERT INTO chunks_fts VALUES ('old-event','session','turn','old evidence')")
        rowid = connection.execute("SELECT rowid FROM chunks_fts").fetchone()[0]
    index = SparseIndex(path)
    try:
        assert index.ids() == {"old-event"}
        assert index._conn.execute("SELECT rowid FROM chunks_fts").fetchone()[0] == rowid
        assert index.search("evidence")
        index.delete_ids(["old-event"])
        assert index.count() == 0 and index.ids() == set()
    finally:
        index._conn.close()


def test_sparse_clear_can_be_repaired_without_new_transcript_data(session):
    engine, path = session
    write(path, [call("FIND_AGAIN")], "a")
    engine.sync_session(session_id=SID, embed=False)
    engine.sparse.clear()
    engine.sync_session(session_id=SID, embed=False)
    assert engine.search_history("FIND_AGAIN", session_id=SID, mode="sparse")["hits"]


def test_close_does_not_wait_for_another_process_index_lease(session):
    from self_directing_mcp.index.locking import index_lock
    engine, _ = session
    engine.ensure_ready()
    with index_lock(engine.settings.index_dir):
        start = time.monotonic()
        engine.close()
        assert time.monotonic() - start < 0.5
