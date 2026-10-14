"""Appends to a live session file are not source tampering; tampering still is."""
from __future__ import annotations

import json
import os
import tracemalloc

import pytest

from self_directing_mcp import session_events
from self_directing_mcp.audit import runner as audit_runner
from self_directing_mcp.codex import parse as codex_parser
from self_directing_mcp.session_events import capture_source, source_scope, iter_raw_lines
from test_index_concurrency import pause_first, policy, incomplete, session  # noqa: F401
from test_reliability import SID, call, message, write, session as reliability_session  # noqa: F401


class _AppendOnFirstFstat:
    """Proxy for os that appends one record right after capture's opening fstat."""

    def __init__(self, path, line):
        self.path, self.line, self.done = path, line, False

    def __getattr__(self, name):
        return getattr(os, name)

    def fstat(self, fd):
        result = os.fstat(fd)
        if not self.done:
            self.done = True
            with self.path.open("ab") as stream:
                stream.write(self.line)
        return result


def _texts(engine):
    return [c.text for c in engine.store.list_chunks(SID)]


def _cursor_end(engine):
    return engine.store.cursor_record(SID)["byte_offset"]


def test_append_during_capture_is_deferred_not_rejected(session, monkeypatch):
    engine, path = session
    write(path, [call("first", "c1")], "a")
    size_before = path.stat().st_size
    late = (json.dumps(call("LATE", "c2")) + "\n").encode()
    monkeypatch.setattr(session_events, "os", _AppendOnFirstFstat(path, late))
    info = engine.sync_session(session_id=SID, embed=False)
    monkeypatch.undo()
    assert info["ok"] is not False, info
    assert _cursor_end(engine) == size_before
    assert not any("LATE" in t for t in _texts(engine))
    assert "unread_bytes" in info["coverage"]["issues"] and info["coverage"]["complete"] is False
    engine.sync_session(session_id=SID, embed=False)
    assert _cursor_end(engine) == path.stat().st_size
    assert sum("LATE" in t for t in _texts(engine)) == 1
    assert sum("first" in t for t in _texts(engine)) == 1


def test_append_after_parse_before_publication_is_deferred(session, monkeypatch):
    engine, path = session
    write(path, [call("first", "c1")], "a")
    size_before = path.stat().st_size
    with pause_first(monkeypatch, codex_parser, "parse_session_chunks") as (pool, entered, release):
        pending = pool.submit(engine.sync_session, session_id=SID, embed=False)
        assert entered.wait(5)
        write(path, [call("LATE", "c2")], "a")
        release.set()
        info = pending.result(5)
    assert info["ok"] is not False, info
    assert _cursor_end(engine) == size_before
    assert "unread_bytes" in info["coverage"]["issues"]
    engine.sync_session(session_id=SID, embed=False)
    assert sum("LATE" in t for t in _texts(engine)) == 1
    assert _cursor_end(engine) == path.stat().st_size


def test_incomplete_last_line_completed_mid_parse(session, monkeypatch):
    engine, path = session
    write(path, [call("first", "c1")], "a")
    complete = path.stat().st_size
    encoded = (json.dumps(call("PARTIAL", "c2")) + "\n").encode()
    with path.open("ab") as stream:
        stream.write(encoded[:20])
    with pause_first(monkeypatch, codex_parser, "parse_session_chunks") as (pool, entered, release):
        pending = pool.submit(engine.sync_session, session_id=SID, embed=False)
        assert entered.wait(5)
        with path.open("ab") as stream:
            stream.write(encoded[20:])
        release.set()
        info = pending.result(5)
    assert info["ok"] is not False, info
    assert _cursor_end(engine) == complete
    assert not any("PARTIAL" in t for t in _texts(engine))
    engine.sync_session(session_id=SID, embed=False)
    assert sum("PARTIAL" in t for t in _texts(engine)) == 1


def _tamper(path, kind):
    data = path.read_bytes()
    if kind == "overwrite":
        path.write_bytes(data.replace(b"first", b"FIRST"))
    elif kind == "overwrite_append":
        path.write_bytes(data.replace(b"first", b"FIRST") + (json.dumps(call("x", "c9")) + "\n").encode())
    elif kind == "truncate":
        path.write_bytes(data[: len(data) // 2])
    elif kind == "replace":
        replacement = path.with_name("replacement.tmp")
        replacement.write_bytes(data)  # identical bytes, different inode
        os.replace(replacement, path)
    elif kind == "delete":
        path.unlink()


def _assert_index_is_the_current_file(engine, path):
    """The index holds exactly what the file holds NOW, never what an earlier, discarded preparation parsed."""
    chunks, _offset, _sid = codex_parser.parse_session_chunks(path, session_id=SID)
    assert [c.chunk_id for c in engine.store.list_chunks(SID)] == [c.chunk_id for c in chunks]
    assert [c.text for c in engine.store.list_chunks(SID)] == [c.text for c in chunks]


def _count_commits(engine, monkeypatch):
    commits = []
    real = engine.store.commit_events

    def counting(*args, **kwargs):
        commits.append(kwargs.get("replace"))
        return real(*args, **kwargs)
    monkeypatch.setattr(engine.store, "commit_events", counting)
    return commits


@pytest.mark.parametrize("kind", ["overwrite", "overwrite_append", "truncate", "replace", "delete"])
def test_tampering_after_parse_discards_the_preparation_and_observes_the_source_again(session, monkeypatch, kind):
    """A change detected between capture and publication is never published: the observation is thrown away and
    the file is captured (and fully scanned) again. A vanished file cannot be."""
    engine, path = session
    write(path, [call("first", "c1")], "a")
    engine.ensure_ready()
    commits = _count_commits(engine, monkeypatch)
    with pause_first(monkeypatch, codex_parser, "parse_session_chunks") as (pool, entered, release):
        pending = pool.submit(engine.sync_session, session_id=SID, embed=False)
        assert entered.wait(5)
        _tamper(path, kind)
        release.set()
        if kind == "delete":
            with pytest.raises((ValueError, OSError)):
                pending.result(5)
            assert engine.store.cursor_record(SID) is None and engine.store.chunk_ids(SID) == set()
            return
        info = pending.result(5)
    assert info["ok"] and len(commits) == 1  # one publication: the recapture's
    _assert_index_is_the_current_file(engine, path)
    if kind in ("overwrite", "overwrite_append"):
        assert any("FIRST" in text for text in _texts(engine)) and not any("first" in t for t in _texts(engine))


@pytest.mark.parametrize("kind", ["overwrite", "overwrite_append", "truncate", "replace", "delete"])
def test_tampering_with_appended_session_is_reindexed_never_appended_for_cursor(session, monkeypatch, kind):
    """A synced prefix that was rewritten is a reindex or an error, never an incremental append."""
    engine, path = session
    write(path, [call("first", "c1")], "a")
    engine.sync_session(session_id=SID, embed=False)
    before = engine.store.cursor_record(SID)
    write(path, [call("later", "c2")], "a")
    commits = _count_commits(engine, monkeypatch)
    with pause_first(monkeypatch, codex_parser, "parse_session_chunks") as (pool, entered, release):
        pending = pool.submit(engine.sync_session, session_id=SID, embed=False)
        assert entered.wait(5)
        _tamper(path, kind)
        release.set()
        if kind == "delete":
            with pytest.raises((ValueError, OSError)):
                pending.result(5)
            assert engine.store.cursor_record(SID) == before and commits == []
            return
        info = pending.result(5)
    assert info["ok"] and len(commits) == 1
    if kind == "replace":  # identical bytes under a new inode: the same content, so a normal incremental append
        assert not info["reindexed"]
    else:
        assert info["reindexed"] and info["reindex_reason"] == "source_changed"
    _assert_index_is_the_current_file(engine, path)


def _flip_after_every_parse(monkeypatch, path):
    """The source keeps changing: every parse is followed by a same-size rewrite (the bytes alternate)."""
    real, calls = codex_parser.parse_session_chunks, []

    def parse_then_rewrite(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(1)
        data = bytearray(path.read_bytes())
        data[40] ^= 0x01
        path.write_bytes(bytes(data))
        return result

    monkeypatch.setattr(codex_parser, "parse_session_chunks", parse_then_rewrite)
    return calls


@pytest.mark.parametrize("synced", [False, True])
def test_a_source_that_keeps_changing_is_rejected_after_bounded_retries(session, monkeypatch, synced):
    """Today's error behaviour is kept: when no stable observation can be had, nothing is published."""
    engine, path = session
    write(path, [call("first", "c1")], "a")
    engine.ensure_ready()
    before = None
    if synced:
        engine.sync_session(session_id=SID, embed=False)
        write(path, [call("later", "c2")], "a")
        before = (engine.store.cursor_record(SID), engine.store.chunk_ids(SID))
    calls = _flip_after_every_parse(monkeypatch, path)
    with pytest.raises(session_events.SourceChanged, match="source changed during preparation"):
        engine.sync_session(session_id=SID, embed=False)
    assert len(calls) == 3  # bounded: the same number of captures the stale-cursor recapture allows
    if synced:
        assert (engine.store.cursor_record(SID), engine.store.chunk_ids(SID)) == before
    else:
        assert engine.store.cursor_record(SID) is None and engine.store.chunk_ids(SID) == set()


def test_audit_never_claims_records_appended_during_audit(session, monkeypatch):
    """Growth is judged as of the captured offset; the late record is never claimed as reviewed."""
    engine, path = session
    write(path, [call("echo safe")], "a")
    policy(engine)
    size = path.stat().st_size
    with pause_first(monkeypatch, audit_runner, "audit_contract") as (pool, entered, release):
        audit = pool.submit(engine.audit_session, SID)
        assert entered.wait(5)
        write(path, [call("DELETE", "late")], "a")
        release.set()
        value = audit.result(5)
    assert value["verdict"] == "clean"
    assert value["coverage"]["complete"] is False
    assert value["coverage"]["judged_through_byte"] == size and value["coverage"]["unread_bytes"] > 0
    assert not any("DELETE" in e.get("snippet", "") for f in value["findings"] for e in f.get("evidence", []))
    # The next audit actually reviews the appended record.
    assert engine.audit_session(SID)["verdict"] == "violation"


def test_capture_is_bounded_hashed_and_does_not_retain_the_file(tmp_path):
    path = tmp_path / "big.jsonl"
    line = (json.dumps(message("x" * 1000)) + "\n").encode()
    path.write_bytes(line * 20000)  # ~20 MB
    size = path.stat().st_size
    tracemalloc.start()
    try:
        observation = capture_source(path)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 6 * 1024 * 1024, peak
    assert observation.prefix_digest(size) == __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    with path.open("ab") as stream:
        stream.write(b'{"partial"')
    assert observation.validate_prefix() is True
    assert sum(1 for _ in iter_raw_lines(path)) == 20000
    with source_scope(observation):
        assert sum(1 for _ in iter_raw_lines(path)) == 20000
