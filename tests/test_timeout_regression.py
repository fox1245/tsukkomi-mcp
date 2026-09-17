from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from test_reliability import session, write, call, SID
from self_directing_mcp.codex_hooks import handle_hook


def test_unchanged_sync_does_not_rewrite_fts(session):
    engine, path = session
    write(path, [call("echo initial")], "a")
    engine.sync_session(session_id=SID, embed=False)
    before = engine.sparse._conn.total_changes
    again = engine.sync_session(session_id=SID, embed=False)
    assert again["new_chunks"] == 0
    assert engine.sparse._conn.total_changes == before


def test_append_updates_only_new_sparse_documents(session, monkeypatch):
    engine, path = session
    write(path, [call(f"echo {i}", str(i)) for i in range(30)], "a")
    engine.sync_session(session_id=SID, embed=False)
    observed = []
    original = engine.sparse.upsert_many
    def spy(chunks):
        observed.append(len(chunks))
        return original(chunks)
    monkeypatch.setattr(engine.sparse, "upsert_many", spy)
    write(path, [call("echo newest", "new")], "a")
    engine.sync_session(session_id=SID, embed=False)
    assert sum(observed) == 1


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


def test_hook_performs_exactly_one_sync(session, monkeypatch):
    engine, path = session
    engine.upsert_contracts([{"id": "no-delete", "type": "must_not", "scope": "tool_call", "regex": "DELETE_ME"}])
    calls = []
    original = engine.sync_session
    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, "sync_session", spy)
    handle_hook(engine, "PreToolUse", SID, str(path), "Bash", {"command": "echo hi"})
    assert len(calls) == 1


def test_sparse_lookup_uses_indexed_rowid_mapping(session):
    engine, path = session
    engine.sync_session(session_id=SID, embed=False)
    plan = engine.sparse._conn.execute(
        "EXPLAIN QUERY PLAN SELECT doc_rowid FROM chunk_rows WHERE chunk_id=?", ("x",)).fetchall()
    assert "SEARCH" in str([tuple(row) for row in plan])
    assert engine.sparse.count() == engine.store.chunk_count(SID)


def test_description_audit_does_not_deserialize_full_history(session, monkeypatch):
    engine, path = session
    engine.sync_session(session_id=SID, embed=False)
    engine.upsert_contracts([{"id": "descriptive", "type": "must", "description": "Keep work local"}])
    original = engine.store.list_chunks
    counts = []
    def spy(*args, **kwargs):
        value = original(*args, **kwargs)
        counts.append(len(value))
        return value
    monkeypatch.setattr(engine.store, "list_chunks", spy)
    assert engine.audit_session(SID)["verdict"] == "unknown"
    assert counts == []


def test_mcp_async_wrapper_yields_to_other_requests(monkeypatch):
    from self_directing_mcp import server
    def slow(*args, **kwargs):
        time.sleep(0.15)
        return {"ok": True, "verdict": "clean"}
    monkeypatch.setattr(server._engine, "audit_session", slow)
    async def exercise():
        task = asyncio.create_task(server.audit_session(SID))
        await asyncio.sleep(0.02)
        assert not task.done()
        assert (await task)["ok"]
    asyncio.run(exercise())


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
        with pytest.raises(TimeoutError):
            engine.audit_status(_deadline=time.monotonic() + 0.15)
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
    with pytest.raises(TimeoutError):
        engine.audit_status(_deadline=time.monotonic() - 1)


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
