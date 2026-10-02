"""Bounded offline races through real parsing, SQLite publication and native audits."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
import sqlite3
import threading
import time

import pytest

from self_directing_mcp.audit import runner as audit_runner
from self_directing_mcp.codex import parse as codex_parser
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index.locking import index_lock
from self_directing_mcp.request_control import (
    IndexBusy, RequestControl, RequestStopped, request_scope,
)
from test_reliability import SID, call, session as reliability_session, write
from test_omp_local import _append, _message, local_engine
from test_agy_transcript import (
    SID as AGY_SID, _append_result, _deploy_check, _receipt_payload,
    session as agy_session,
)


@pytest.fixture
def session(reliability_session):
    engine, path = reliability_session
    try:
        yield engine, path
    finally:
        engine.close()



@contextmanager
def pause_first(monkeypatch, owner, name):
    """Pause after real work, before its caller can consume/publish the result."""
    entered, release = threading.Event(), threading.Event()
    gate = threading.Lock()
    first = True
    original = getattr(owner, name)

    def paused(*args, **kwargs):
        nonlocal first
        with gate:
            selected, first = first, False
        result = original(*args, **kwargs)
        if selected:
            entered.set()
            if not release.wait(10):
                raise AssertionError(f"{name} race barrier was not released")
        return result

    monkeypatch.setattr(owner, name, paused)
    pool = ThreadPoolExecutor(max_workers=3)
    try:
        yield pool, entered, release
    finally:
        release.set()
        pool.shutdown(wait=True)


def policy(engine):
    engine.upsert_contracts([{
        "id": "no-delete", "type": "must_not", "scope": "tool_call", "regex": "DELETE",
    }])


def incomplete(value):
    assert value["verdict"] != "clean", value
    assert value["coverage"]["complete"] is False, value
    assert value["action_executed"] is False, value


def test_paused_native_evaluation_allows_metadata_and_another_audit(session, monkeypatch):
    engine, path = session
    write(path, [call("echo safe")], "a")
    policy(engine)
    with closing(SelfDirectEngine(engine.settings)) as peer, pause_first(
        monkeypatch, audit_runner, "audit_contract",
    ) as (pool, entered, release):
        first = pool.submit(engine.audit_session, SID)
        assert entered.wait(5)
        # A real writer must commit, not merely acquire a Python lock. A retained
        # rollback-journal snapshot would block this SQLite publication too.
        write(path, [call("echo newer", "newer")], "a")
        synced = pool.submit(peer.sync_session, session_id=SID, embed=False).result(3)
        assert synced["coverage"]["byte_offset"] == path.stat().st_size
        second = pool.submit(peer.audit_session, SID).result(3)
        assert second["verdict"] == "clean" and second["coverage"]["complete"]
        status = pool.submit(peer.audit_status, SID).result(3)
        assert status["session"]["chunk_count"] == 3
        assert not first.done()
        release.set()
        incomplete(first.result(5))


@pytest.mark.parametrize("change", ["add", "upsert", "revoke", "missing", "corrupt"])
def test_rule_document_change_never_publishes_stale_clean(session, monkeypatch, change):
    engine, _ = session
    policy(engine)
    with pause_first(monkeypatch, audit_runner, "audit_contract") as (pool, entered, release):
        audit = pool.submit(engine.check_action, SID, {"tool_name": "shell", "arguments": "echo safe"})
        assert entered.wait(5)
        if change == "add":
            engine.upsert_contracts([{"id": "new", "type": "must_not", "regex": "safe"}])
        elif change == "upsert":
            engine.upsert_contracts([{"id": "no-delete", "type": "must_not", "regex": "safe"}])
        elif change == "revoke":
            engine.revoke_contract("no-delete")
        elif change == "missing":
            engine.settings.resolve_contracts_path().unlink()
        else:
            engine.settings.resolve_contracts_path().write_text("{invalid", encoding="utf-8")
        release.set()
        incomplete(audit.result(5))
    # Missing/corrupt documents must not become clean by reusing the old cache.
    if change in ("missing", "corrupt", "revoke"):
        assert engine.check_action(SID, {"tool_name": "shell", "arguments": "echo safe"})["verdict"] == "unknown"
    else:
        assert engine.check_action(SID, {"tool_name": "shell", "arguments": "echo safe"})["verdict"] == "violation"


@pytest.mark.parametrize("change", ["append", "rewrite", "missing"])
def test_source_change_without_index_refresh_invalidates_audit(session, monkeypatch, change):
    engine, path = session
    write(path, [call("echo safe")], "a")
    policy(engine)
    with pause_first(monkeypatch, audit_runner, "audit_contract") as (pool, entered, release):
        audit = pool.submit(engine.audit_session, SID)
        assert entered.wait(5)
        if change == "append":
            write(path, [call("DELETE", "late")], "a")
        elif change == "rewrite":
            write(path, [{"type": "session_meta", "payload": {"id": SID}}, call("DELETE")])
        else:
            path.unlink()
        release.set()
        incomplete(audit.result(5))


def test_stale_snapshot_retains_actual_violation_evidence(session, monkeypatch):
    engine, path = session
    write(path, [call("DELETE")], "a")
    policy(engine)
    with pause_first(monkeypatch, audit_runner, "audit_contract") as (pool, entered, release):
        audit = pool.submit(engine.audit_session, SID)
        assert entered.wait(5)
        evidence_ids = engine.store.chunk_ids(SID)
        engine.revoke_contract("no-delete")
        release.set()
        value = audit.result(5)
        incomplete(value)
        finding = next(f for f in value["findings"] if f["contract_id"] == "no-delete")
        assert finding["verdict"] == "violation"
        assert {e["chunk_id"] for e in finding["evidence"]} <= evidence_ids
        assert any("DELETE" in e["snippet"] for e in finding["evidence"])


@pytest.mark.parametrize("stop", ["deadline", "cancel"])
def test_parser_abandonment_cannot_late_commit_and_does_not_cancel_other_request(session, monkeypatch, stop):
    engine, path = session
    engine.sync_session(session_id=SID, embed=False)
    before = engine.store.cursor_record(SID)
    ids = engine.store.chunk_ids(SID)
    write(path, [call("LATE_EVENT")], "a")
    control = RequestControl(deadline=time.monotonic() + 0.3 if stop == "deadline" else None)

    def sync():
        with request_scope(control):
            return engine.sync_session(session_id=SID, embed=False)

    with pause_first(monkeypatch, codex_parser, "parse_session_chunks") as (pool, entered, release):
        pending = pool.submit(sync)
        assert entered.wait(5)
        if stop == "cancel":
            control.cancel()
        else:
            # Wait against the actual absolute deadline, not a scheduler guess.
            threading.Event().wait(max(0, control.deadline - time.monotonic()) + 0.02)
        independent = pool.submit(engine.audit_status, SID).result(3)
        assert independent["session"]["chunk_count"] == len(ids)
        release.set()
        with pytest.raises(RequestStopped) as stopped:
            pending.result(5)
        assert stopped.value.reason == ("request_deadline" if stop == "deadline" else "request_cancelled")
    assert engine.store.cursor_record(SID) == before
    assert engine.store.chunk_ids(SID) == ids
    engine.sync_session(session_id=SID, embed=False)
    assert engine.store.cursor_record(SID)["byte_offset"] == path.stat().st_size
    assert any("LATE_EVENT" in c.text for c in engine.store.list_chunks(SID))


@pytest.mark.parametrize("initially_synced", [False, True])
def test_concurrent_preparations_cannot_rewind_newer_cursor(session, monkeypatch, initially_synced):
    engine, path = session
    engine.ensure_ready()
    if initially_synced:
        engine.sync_session(session_id=SID, embed=False)
    write(path, [call("earlier", "earlier")], "a")
    with pause_first(monkeypatch, codex_parser, "parse_session_chunks") as (pool, entered, release):
        old = pool.submit(engine.sync_session, session_id=SID, embed=False)
        assert entered.wait(5)
        write(path, [call("newer", "newer")], "a")
        newest = pool.submit(engine.sync_session, session_id=SID, embed=False).result(3)
        final_cursor = engine.store.cursor_record(SID)
        final_ids = engine.store.chunk_ids(SID)
        assert newest["coverage"]["byte_offset"] == path.stat().st_size
        release.set()
        with pytest.raises(ValueError):
            old.result(5)
    assert engine.store.cursor_record(SID) == final_cursor
    assert engine.store.chunk_ids(SID) == final_ids
    assert {c.meta.get("call_id") for c in engine.store.list_chunks(SID) if c.kind == "tool_call"} == {"earlier", "newer"}


def test_equal_offset_rewrite_cannot_replace_newer_generation(session, monkeypatch):
    engine, path = session
    write(path, [call("old-command")], "a")
    engine.ensure_ready()
    with pause_first(monkeypatch, codex_parser, "parse_session_chunks") as (pool, entered, release):
        old = pool.submit(engine.sync_session, session_id=SID, embed=False)
        assert entered.wait(5)
        write(path, [{"type": "session_meta", "payload": {"id": SID}}, call("new-command")])
        engine.sync_session(session_id=SID, embed=False)
        latest = engine.store.cursor_record(SID)
        release.set()
        with pytest.raises(ValueError):
            old.result(5)
    assert engine.store.cursor_record(SID) == latest
    assert any("new-command" in c.text for c in engine.store.list_chunks(SID))
    assert not any("old-command" in c.text for c in engine.store.list_chunks(SID))


def test_omp_preparation_uses_exact_prior_call_scope_and_cannot_overwrite_newer_parse(local_engine, monkeypatch):
    from self_directing_mcp import omp
    engine, path = local_engine
    _append(path, _message("assistant", {"type": "toolCall", "id": "same", "name": "shell",
                                        "arguments": {"command": "prepare"}}))
    engine.sync_session(session_id="session-1", provider="omp", path=str(path), embed=False)
    _append(path, {"type": "custom", "customType": "tool_execution_start",
                   "data": {"toolCallId": "same", "toolName": "shell"}})
    with pause_first(monkeypatch, omp, "parse_session_chunks") as (pool, entered, release):
        old = pool.submit(engine.sync_session, session_id="session-1", provider="omp", path=str(path), embed=False)
        assert entered.wait(5)
        # Same call ID but wrong tool is not an authoritative prior call.
        _append(path, {"type": "custom", "customType": "tool_execution_start",
                       "data": {"toolCallId": "same", "toolName": "write"}})
        newer = engine.sync_session(session_id="session-1", provider="omp", path=str(path), embed=False)
        assert "unsupported_events" in newer["coverage"]["issues"]
        cursor = engine.store.cursor_record("session-1", "omp")
        release.set()
        with pytest.raises(ValueError):
            old.result(5)
    assert engine.store.cursor_record("session-1", "omp") == cursor
    assert engine.store.session_status("session-1", "omp")["unsupported_event_count"] == 1


def test_omp_prior_call_from_another_session_cannot_complete_coverage(local_engine, monkeypatch):
    from self_directing_mcp import omp
    engine, path = local_engine
    foreign = path.with_name("foreign.jsonl")
    _append(foreign, {"type": "session", "id": "foreign"},
            _message("assistant", {"type": "toolCall", "id": "shared", "name": "shell",
                                   "arguments": {"command": "prepare"}}))
    engine.sync_session(session_id="foreign", provider="omp", path=str(foreign), embed=False)
    engine.sync_session(session_id="session-1", provider="omp", path=str(path), embed=False)
    _append(path, {"type": "custom", "customType": "tool_execution_start",
                   "data": {"toolCallId": "shared", "toolName": "shell"}})
    with pause_first(monkeypatch, omp, "parse_session_chunks") as (pool, entered, release):
        pending = pool.submit(engine.sync_session, session_id="session-1", provider="omp",
                              path=str(path), embed=False)
        assert entered.wait(5)
        _append(foreign, _message("toolResult", {"type": "text", "text": "done"},
                                 toolCallId="shared", toolName="shell", isError=False))
        completed = engine.sync_session(session_id="foreign", provider="omp", path=str(foreign), embed=False)
        assert completed["coverage"]["complete"]
        release.set()
        own = pending.result(5)
    assert "unsupported_events" in own["coverage"]["issues"]
    assert engine.store.session_status("session-1", "omp")["unsupported_event_count"] == 1
    assert not [c for c in engine.store.list_chunks("session-1", "omp") if c.kind in ("tool_call", "tool_result")]


def test_agy_receipt_changed_during_parse_is_not_marked_consumed(agy_session, monkeypatch):
    from self_directing_mcp import agy
    engine, path = agy_session
    payload = _receipt_payload(path)
    engine.record_agy_hook(payload, "PreToolUse")
    _append_result(path)
    with pause_first(monkeypatch, agy, "parse_session_chunks") as (pool, entered, release):
        old = pool.submit(engine.sync_session, session_id=AGY_SID, provider="agy", path=str(path), embed=False)
        assert entered.wait(5)
        engine.record_agy_hook(payload, "PostToolUse")
        newer = engine.sync_session(session_id=AGY_SID, provider="agy", path=str(path), embed=False)
        assert newer["coverage"]["complete"]
        cursor = engine.store.cursor_record(AGY_SID, "agy")
        release.set()
        with pytest.raises(ValueError):
            old.result(5)
    assert engine.store.cursor_record(AGY_SID, "agy") == cursor
    assert _deploy_check(engine, path)["verdict"] == "clean"
    linked = [c for c in engine.store.list_chunks(AGY_SID, "agy")
              if c.meta.get("evidence_origin") == "host_hook_and_native_transcript"]
    assert [c.kind for c in linked] == ["tool_call", "tool_result"]
    assert linked[0].meta["call_id"] == linked[1].meta["call_id"]
    assert linked[1].meta["success"] is True


def test_cancelled_late_embedding_cannot_resurrect_removed_event(session, monkeypatch):
    engine, path = session
    write(path, [call("removed-event")], "a")
    engine.ensure_ready()
    control = RequestControl()

    def sync():
        with request_scope(control):
            return engine.sync_session(session_id=SID)

    with pause_first(monkeypatch, engine.embedder, "embed_documents") as (pool, entered, release):
        pending = pool.submit(sync)
        assert entered.wait(5)
        old_ids = engine.store.chunk_ids(SID)
        control.cancel()
        write(path, [{"type": "session_meta", "payload": {"id": SID}}, call("replacement")])
        engine.sync_session(session_id=SID, embed=False)
        removed = old_ids - engine.store.chunk_ids(SID)
        assert removed
        release.set()
        with pytest.raises(RequestStopped) as stopped:
            pending.result(5)
        assert stopped.value.reason == "request_cancelled"
    assert not removed.intersection(engine.dense.ids())
    assert not removed.intersection(engine.sparse.ids())
    assert any("replacement" in c.text for c in engine.store.list_chunks(SID))


@pytest.mark.parametrize("cause", ["busy", "deadline", "cancel"])
def test_real_writer_lock_has_distinct_stop_taxonomy(session, cause):
    engine, _ = session
    engine.ensure_ready()
    engine.settings.lock_wait_timeout_sec = 0.12
    entered, release = threading.Event(), threading.Event()

    def hold():
        with engine._lock:
            entered.set()
            release.wait(5)

    worker = threading.Thread(target=hold)
    worker.start()
    try:
        assert entered.wait(2)
        if cause == "cancel":
            cancelled = threading.Event()
            cancelled.set()
            with pytest.raises(RequestStopped) as stopped:
                engine.audit_status(cancel_event=cancelled)
            assert stopped.value.reason == "request_cancelled"
        elif cause == "deadline":
            with pytest.raises(RequestStopped) as stopped:
                engine.audit_status(_deadline=time.monotonic() + 0.04)
            assert stopped.value.reason == "request_deadline"
        else:
            with pytest.raises(IndexBusy) as stopped:
                engine.audit_status(_deadline=time.monotonic() + 3)
            assert stopped.value.reason == "index_busy"
    finally:
        release.set()
        worker.join()


def test_lockfile_io_failure_is_not_index_contention(tmp_path):
    root = tmp_path / "not-a-directory"
    root.write_text("fixture", encoding="utf-8")
    with pytest.raises(OSError) as failed:
        with index_lock(root, timeout=0.05):
            pytest.fail("file path cannot be an index directory")
    assert not isinstance(failed.value, IndexBusy)


@pytest.mark.parametrize("failure", ["busy", "missing_schema"])
def test_sqlite_operational_failure_is_not_clean_or_empty_search(session, failure):
    engine, _ = session
    engine.sync_session(session_id=SID, embed=False)
    engine.sparse._conn.execute("PRAGMA busy_timeout=10")
    other = sqlite3.connect(engine.sparse.db_path)
    try:
        if failure == "busy":
            other.execute("BEGIN EXCLUSIVE")
        else:
            other.execute("DROP TABLE chunks_fts")
            other.commit()
        with pytest.raises(sqlite3.OperationalError) as failed:
            engine.search_history("evidence", session_id=SID, mode="sparse")
        assert not isinstance(failed.value, IndexBusy)
        if failure == "busy":
            assert failed.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
    finally:
        other.rollback()
        other.close()


@pytest.mark.parametrize("cause", ["busy", "deadline", "cancel", "operation"])
def test_server_preserves_worker_timeout_reason(session, monkeypatch, cause):
    from self_directing_mcp import server
    engine, _ = session
    monkeypatch.setattr(server, "_engine", engine)
    error = IndexBusy() if cause == "busy" else RequestStopped(
        "request_deadline" if cause == "deadline" else "request_cancelled"
    ) if cause != "operation" else TimeoutError("local test operation timeout")

    def failed(*args, **kwargs):
        raise error

    monkeypatch.setattr(engine, "audit_session", failed)
    value = asyncio.run(server.audit_session(SID))
    assert value["error"] == {
        "busy": "index_busy", "deadline": "request_deadline",
        "cancel": "request_cancelled", "operation": "operation_timeout",
    }[cause]
    incomplete(value)


@pytest.mark.parametrize("stop", ["deadline", "disconnect"])
def test_server_abandoned_parser_cannot_publish_after_response(session, monkeypatch, stop):
    from self_directing_mcp import server
    engine, path = session
    engine.sync_session(session_id=SID, embed=False)
    before = engine.store.cursor_record(SID)
    before_ids = engine.store.chunk_ids(SID)
    write(path, [call("abandoned-server-event")], "a")
    monkeypatch.setattr(server, "_engine", engine)
    finished = threading.Event()
    original = engine.sync_session

    def observed(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(engine, "sync_session", observed)
    with pause_first(monkeypatch, codex_parser, "parse_session_chunks") as (_, entered, release):
        async def scenario():
            task = asyncio.create_task(server.sync_session(
                session_id=SID, embed=False, timeout_ms=500 if stop == "deadline" else 5000,
            ))
            assert await asyncio.to_thread(entered.wait, 5)
            if stop == "deadline":
                value = await task
                assert value["error"] == "request_deadline"
                incomplete(value)
            else:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            release.set()
            assert await asyncio.to_thread(finished.wait, 5)

        asyncio.run(scenario())
    assert engine.store.cursor_record(SID) == before
    assert engine.store.chunk_ids(SID) == before_ids
    # No cancellation state leaks into a later independently dispatched request.
    result = asyncio.run(server.sync_session(session_id=SID, embed=False, timeout_ms=5000))
    assert result["ok"] and result["coverage"]["byte_offset"] == path.stat().st_size


def test_cancelled_native_evaluation_preserves_typed_stop_and_request_isolation(session, monkeypatch):
    engine, path = session
    write(path, [call("echo safe")], "a")
    policy(engine)
    control = RequestControl()

    def audit():
        with request_scope(control):
            return engine.audit_session(SID)

    with pause_first(monkeypatch, audit_runner, "audit_contract") as (pool, entered, release):
        cancelled = pool.submit(audit)
        assert entered.wait(5)
        control.cancel()
        independent = pool.submit(engine.audit_session, SID).result(3)
        assert independent["verdict"] == "clean" and independent["coverage"]["complete"]
        release.set()
        with pytest.raises(RequestStopped) as stopped:
            cancelled.result(5)
        assert stopped.value.reason == "request_cancelled"
