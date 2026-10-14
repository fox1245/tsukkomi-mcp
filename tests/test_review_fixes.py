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


# ---- finding 3: no whole-prefix rehash under the publication guard ----

from contextlib import contextmanager  # noqa: E402

from self_directing_mcp.session_events import SourceObservation  # noqa: E402

WINDOW = session_events._TAIL_WINDOW


def _big(tmp_path, size=WINDOW * 5):
    path = tmp_path / "big.jsonl"
    path.write_bytes((b'{"k":"' + b"x" * 90 + b'"}\n') * (size // 100))
    return path


def _hashed_bytes(monkeypatch):
    total = [0]
    real = session_events.hashlib

    class Counting:
        def sha256(self, data=b""):
            digest = real.sha256()
            total[0] += len(data)
            digest.update(data)
            outer = self

            class Wrapper:
                def update(self, block):
                    total[0] += len(block)
                    digest.update(block)

                def copy(self):
                    return digest.copy()

                def hexdigest(self):
                    return digest.hexdigest()
            return Wrapper()
    monkeypatch.setattr(session_events, "hashlib", Counting())
    return total


def test_locked_validation_reads_only_the_tail_window(tmp_path, monkeypatch):
    path = _big(tmp_path)
    observation = capture_source(path)
    with path.open("ab") as stream:
        stream.write(b'{"grown":1}\n')  # a live file: stamp differs from the verified one
    total = _hashed_bytes(monkeypatch)
    assert observation.validate_prefix(locked=True)
    assert 0 < total[0] <= WINDOW
    total[0] = 0
    assert observation.validate_prefix()  # the unlocked check still hashes everything
    assert total[0] >= observation.length


def test_tail_window_rewrite_is_caught_under_the_lock(tmp_path):
    path = _big(tmp_path)
    observation = capture_source(path)
    data = bytearray(path.read_bytes())
    data[observation.length - 10] ^= 1
    path.write_bytes(bytes(data) + b"more\n")
    assert not observation.validate_prefix(locked=True)


def test_locked_validation_rejects_truncation_and_replacement(tmp_path):
    path = _big(tmp_path)
    observation = capture_source(path)
    shorter = tmp_path / "short"
    shorter.write_bytes(path.read_bytes()[:-5])
    os.replace(shorter, path)
    assert not observation.validate_prefix(locked=True)


def test_rewrite_outside_window_is_caught_by_full_check_and_stays_failed(tmp_path):
    path = _big(tmp_path)
    observation = capture_source(path)
    data = bytearray(path.read_bytes())
    data[5] ^= 1
    path.write_bytes(bytes(data) + b"more\n")
    assert observation.validate_prefix(locked=True)  # documented residual window
    assert not observation.validate_prefix()  # the full check outside the lock catches it
    assert not observation.validate_prefix(locked=True)  # and the failure sticks


def test_sync_does_full_rehash_only_outside_the_guard(session, monkeypatch):
    engine, path = session
    write(path, [call("one", "c1")], "a")
    depth, seen = [0], []
    real_guard = engine._publication_guard

    @contextmanager
    def guard():
        with real_guard():
            depth[0] += 1
            try:
                yield
            finally:
                depth[0] -= 1
    engine._publication_guard = guard
    real = SourceObservation.validate_prefix

    def spy(self, *, locked=False):
        seen.append((locked, depth[0]))
        return real(self, locked=locked)
    monkeypatch.setattr(SourceObservation, "validate_prefix", spy)
    engine.sync_session(session_id=SID, embed=False)
    write(path, [call("two", "c2")], "a")
    engine.sync_session(session_id=SID, embed=False)
    engine.audit_session(session_id=SID)
    assert seen and any(depth for _, depth in seen)
    assert [item for item in seen if item[1] and not item[0]] == []
