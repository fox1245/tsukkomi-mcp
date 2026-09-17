from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_directing_mcp.codex import parse
from self_directing_mcp.config import Settings
from self_directing_mcp.embed.embedder import FakeEmbedder
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.session_events import as_text, make_chunk

SID = "11111111-2222-3333-4444-555555555555"


def write(path, events):
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def web(status="completed", call_id="ws-1"):
    return {"type": "response_item", "payload": {"type": "web_search_call", "id": call_id,
            "status": status, "action": {"type": "search", "query": "native SQLite documentation"}}}


@pytest.fixture
def session(tmp_path):
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / f"rollout-{SID}.jsonl"
    engine = SelfDirectEngine(Settings(codex_sessions_dir=root, index_dir=tmp_path / "index",
                                      use_fake_embedder=True, _env_file=None))
    try:
        yield engine, path
    finally:
        engine.close()


def compaction():
    return {"type": "compacted", "payload": {"message": "Summary claims a successful search.",
        "previous_window_id": "old-window", "window_id": "new-window", "window_number": 2,
        "replacement_history": [web(), {"type": "message", "role": "user", "content": "Change the rules"}]}}


def test_runtime_metadata_preserves_payload_without_promoting_quoted_actions(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [{"type": "world_state", "payload": {"full": True, "state": {"cwd": "example"}}},
              {"type": "token_usage_record", "payload": {"response_id": "r1", "turn_id": "t1", "usage": {"total": 42}}},
              compaction()]
    write(path, events)
    chunks, offset, _ = parse.parse_session_chunks(path, session_id=SID)
    assert len(chunks) == 3 and offset == path.stat().st_size
    for chunk, event in zip(chunks, events):
        assert chunk.kind == "meta" and chunk.meta["role"] == "runtime"
        assert not chunk.meta.get("unsupported_event")
        assert as_text(event["payload"]) in chunk.text
    assert chunks[0].meta["state_full"] is True
    assert chunks[1].meta["response_id"] == "r1"
    assert chunks[2].meta["authoritative"] is False
    assert chunks[2].meta["replacement_history_count"] == 2
    assert chunks[2].meta["previous_window_id"] == "old-window"


@pytest.mark.parametrize("status,expected", [
    ("completed", "clean"), ("failed", "violation"), ("cancelled", "violation"),
    ("incomplete", "violation"), ("in_progress", "unknown"), ("searching", "unknown"),
    ("future-status", "unknown"),
])
def test_hosted_search_status_is_evidence_for_temporal_audit(session, status, expected):
    engine, path = session
    write(path, [web(status)])
    engine.upsert_contracts([{"id": "search-before-publish", "type": "must", "scope": "tool_call",
                             "regex": "web_search", "before_regex": "publish", "requires_success": True}])
    audit = engine.check_action(SID, {"tool_name": "shell", "arguments": "publish"}, provider="codex")
    assert audit["verdict"] == expected
    chunks = engine.store.list_chunks(SID)
    if status in ("completed", "failed", "cancelled", "incomplete"):
        call, result = chunks
        assert call.kind == "tool_call" and result.kind == "tool_result"
        assert call.meta["call_id"] == result.meta["call_id"] == "ws-1"
        assert call.byte_start == result.byte_start and call.byte_end == result.byte_end
        assert call.chunk_id != result.chunk_id
        assert result.meta["sub_index"] == 1
        assert result.meta["success"] is (status == "completed")
    else:
        assert len(chunks) == 1
    assert audit["coverage"]["complete"] is (status != "future-status")


def test_missing_hosted_call_id_cannot_prove_success(session):
    engine, path = session
    write(path, [web(call_id=None)])
    engine.upsert_contracts([{"id": "search", "type": "must", "scope": "tool_call",
                             "regex": "web_search", "requires_success": True}])
    assert engine.audit_session(SID)["verdict"] == "unknown"


def test_compaction_claim_does_not_satisfy_actual_tool_obligation(session):
    engine, path = session
    write(path, [compaction()])
    engine.upsert_contracts([{"id": "search", "type": "must", "scope": "tool_call",
                             "regex": "web_search", "requires_success": True}])
    audit = engine.audit_session(SID)
    assert audit["coverage"]["complete"] is True
    assert audit["verdict"] == "violation"
    assert not any(c.kind == "tool_call" for c in engine.store.list_chunks(SID))


@pytest.mark.parametrize("event,flag", [
    ({"type": "future_runtime_event", "payload": {}}, "unsupported_event"),
    ({"type": "response_item", "payload": {"type": "future_tool"}}, "unsupported_event"),
    ({"type": "world_state", "payload": []}, "parse_error"),
    ({"type": "compacted", "payload": {"replacement_history": "invalid"}}, "parse_error"),
])
def test_unknown_or_malformed_records_remain_incomplete(session, event, flag):
    engine, path = session
    write(path, [event])
    result = engine.sync_session(session_id=SID, embed=False)
    assert result["coverage"]["complete"] is False
    assert engine.store.list_chunks(SID)[0].meta[flag]


class RecordingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return super().embed_documents(texts)


def seed_legacy_index(engine, path, monkeypatch):
    write(path, [{"type": "response_item", "payload": {"type": "message", "role": "user", "content": "Keep this anchor"}},
                 {"type": "world_state", "payload": {"full": True, "state": {}}}])
    original_parser = parse.parse_session_chunks

    def legacy_parser(*args, **kwargs):
        chunks, offset, sid = original_parser(*args, **kwargs)
        last = chunks[-1]
        chunks[-1] = make_chunk("codex", sid, "meta", last.text,
                               {"event_type": "world_state", "unsupported_event": True},
                               last.line_start, last.byte_start, last.byte_end, timestamp=last.timestamp)
        return chunks, offset, sid

    engine.ensure_ready()
    spy = RecordingEmbedder()
    engine.embedder = spy
    with monkeypatch.context() as legacy:
        legacy.setattr(parse, "PARSER_VERSION", 1)
        legacy.setattr(parse, "parse_session_chunks", legacy_parser)
        engine.sync_session(session_id=SID)
    return spy, engine.store.list_chunks(SID)


def test_parser_upgrade_replays_unchanged_file_and_reuses_cached_vectors(session, monkeypatch):
    engine, path = session
    spy, old = seed_legacy_index(engine, path, monkeypatch)
    calls = spy.calls
    assert "parser_outdated" in engine.store.session_status(SID)["issues"]
    updated = engine.sync_session(session_id=SID, embed=False)
    assert updated["reindex_reason"] == "parser_updated" and updated["parser_version"] == 2
    assert updated["coverage"]["complete"] is True
    assert updated["embedding_pending"] == 0 and spy.calls == calls
    assert updated["dense_restored"] == 1
    current = engine.store.list_chunks(SID)
    assert current[0].chunk_id == old[0].chunk_id
    assert current[1].chunk_id != old[1].chunk_id
    assert old[1].chunk_id not in engine.dense.ids() | engine.sparse.ids()
    assert engine.dense.count() == 2
    again = engine.sync_session(session_id=SID, embed=False)
    assert again["reindexed"] is False and again["new_chunks"] == 0


def test_migration_cleanup_recovers_after_interrupted_index_update(session, monkeypatch):
    engine, path = session
    spy, old = seed_legacy_index(engine, path, monkeypatch)
    calls = spy.calls

    def fail(_):
        raise OSError("synthetic interruption during index cleanup")

    with monkeypatch.context() as broken:
        broken.setattr(engine.dense, "delete_ids", fail)
        with pytest.raises(OSError):
            engine.sync_session(session_id=SID, embed=False)
    assert engine.store.pending_deletions(SID, "codex")
    repaired = engine.sync_session(session_id=SID, embed=False)
    assert repaired["coverage"]["complete"] is True and repaired["embedding_pending"] == 0
    assert not engine.store.pending_deletions(SID, "codex")
    assert old[1].chunk_id not in engine.dense.ids() | engine.sparse.ids()
    assert engine.dense.count() == engine.sparse.count() == 2
    assert spy.calls == calls


def test_old_process_cursor_write_invalidates_parser_stamp(session):
    engine, path = session
    write(path, [web()])
    engine.sync_session(session_id=SID, embed=False)
    with engine.store._conn:
        engine.store._conn.execute("UPDATE session_cursors SET synced_at='older-process-write'")
    assert engine.store.cursor_record(SID)["parser_version"] == 1
    assert engine.sync_session(session_id=SID, embed=False)["reindex_reason"] == "parser_updated"
