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


@pytest.mark.parametrize("kind", ["overwrite", "overwrite_append", "truncate", "replace", "delete"])
def test_tampering_after_parse_is_still_rejected(session, monkeypatch, kind):
    engine, path = session
    write(path, [call("first", "c1")], "a")
    engine.ensure_ready()
    with pause_first(monkeypatch, codex_parser, "parse_session_chunks") as (pool, entered, release):
        pending = pool.submit(engine.sync_session, session_id=SID, embed=False)
        assert entered.wait(5)
        _tamper(path, kind)
        release.set()
        with pytest.raises((ValueError, OSError)):
            pending.result(5)
    assert engine.store.cursor_record(SID) is None
    assert engine.store.chunk_ids(SID) == set()


@pytest.mark.parametrize("kind", ["overwrite", "overwrite_append", "truncate", "replace", "delete"])
def test_tampering_with_appended_session_is_still_rejected_for_cursor(session, monkeypatch, kind):
    """A synced prefix that was rewritten is a reindex or an error, never an incremental append."""
    engine, path = session
    write(path, [call("first", "c1")], "a")
    engine.sync_session(session_id=SID, embed=False)
    before = engine.store.cursor_record(SID)
    write(path, [call("later", "c2")], "a")
    with pause_first(monkeypatch, codex_parser, "parse_session_chunks") as (pool, entered, release):
        pending = pool.submit(engine.sync_session, session_id=SID, embed=False)
        assert entered.wait(5)
        _tamper(path, kind)
        release.set()
        with pytest.raises((ValueError, OSError)):
            pending.result(5)
    assert engine.store.cursor_record(SID) == before


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
