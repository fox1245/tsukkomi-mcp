import os
from pathlib import Path

import pytest

from self_directing_mcp import session_events
from self_directing_mcp.index import ingest
from self_directing_mcp.session_events import capture_source
from test_reliability import SID, call, write, session  # noqa: F401


def _state(engine):
    return (engine.store.cursor_record(SID)["byte_offset"], engine.store.chunk_count(SID, "codex"),
            len(engine.store.pending_deletions(SID, "codex")))


def _stale_once(monkeypatch, stale):
    real, calls = ingest.capture_source, []

    def fake(path, **kwargs):
        calls.append(1)
        return stale if len(calls) == 1 else real(path, **kwargs)
    monkeypatch.setattr(ingest, "capture_source", fake)
    return calls


def test_stale_observation_never_rewinds_newer_cursor(session, monkeypatch):
    engine, path = session
    write(path, [call("one", "c1")], "a")
    engine.sync_session(session_id=SID, embed=False)
    stale = capture_source(path)  # process A captured here
    write(path, [call("two", "c2"), call("three", "c3")], "a")
    engine.sync_session(session_id=SID, embed=False)  # process B published a later cursor
    before = _state(engine)
    assert before[0] > stale.length
    calls = _stale_once(monkeypatch, stale)
    info = engine.sync_session(session_id=SID, embed=False)
    assert len(calls) == 2  # recaptured once
    assert _state(engine) == before and before[2] == 0
    assert not info["reindexed"] and info["reindex_reason"] is None
    assert info["coverage"]["complete"] and info["coverage"]["byte_offset"] == before[0]


def test_stale_observation_gives_up_without_publishing(session, monkeypatch):
    engine, path = session
    write(path, [call("one", "c1")], "a")
    engine.sync_session(session_id=SID, embed=False)
    stale = capture_source(path)
    write(path, [call("two", "c2")], "a")
    engine.sync_session(session_id=SID, embed=False)
    before = _state(engine)
    monkeypatch.setattr(ingest, "_RECAPTURE_ATTEMPTS", 1)
    _stale_once(monkeypatch, stale)
    with pytest.raises(ValueError):
        engine.sync_session(session_id=SID, embed=False)
    assert _state(engine) == before


def test_genuine_truncation_still_reindexes(session):
    engine, path = session
    write(path, [call("one", "c1"), call("two", "c2")], "a")
    engine.sync_session(session_id=SID, embed=False)
    keep = path.read_bytes().splitlines(keepends=True)[:1]
    path.write_bytes(b"".join(keep).replace(b"one", b"uno"))
    info = engine.sync_session(session_id=SID, embed=False)
    assert info["reindexed"] and info["reindex_reason"] == "source_changed"
