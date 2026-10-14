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


def test_stage_validation_reads_only_the_tail_window_and_the_full_reread_is_explicit(tmp_path, monkeypatch):
    path = _big(tmp_path)
    observation = capture_source(path)
    with path.open("ab") as stream:
        stream.write(b'{"grown":1}\n')  # a live file: stamp differs from the verified one
    total = _hashed_bytes(monkeypatch)
    assert observation.validate_prefix(locked=True)
    assert 0 < total[0] <= WINDOW
    total[0] = 0
    assert observation.validate_prefix()  # the unlocked stage check is bounded as well: the pass was the one scan
    assert 0 < total[0] <= WINDOW
    total[0] = 0
    assert observation.validate_prefix(full=True)  # a whole-prefix re-read stays available, only when asked for
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


def test_rewrite_outside_window_is_caught_by_an_explicit_full_check_and_stays_failed(tmp_path):
    path = _big(tmp_path)
    observation = capture_source(path)
    data = bytearray(path.read_bytes())
    data[5] ^= 1
    path.write_bytes(bytes(data) + b"more\n")
    assert observation.validate_prefix(locked=True)  # documented residual window ...
    assert observation.validate_prefix()  # ... which the later bounded stage check shares (see validate_prefix) ...
    assert not observation.validate_prefix(full=True)  # ... and the next full scan (here: explicit) closes
    assert not observation.validate_prefix(locked=True)  # and the failure sticks
    assert not observation.validate_prefix()


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

    def spy(self, *, locked=False, full=False):
        seen.append((locked, depth[0]))
        return real(self, locked=locked, full=full)
    monkeypatch.setattr(SourceObservation, "validate_prefix", spy)
    engine.sync_session(session_id=SID, embed=False)
    write(path, [call("two", "c2")], "a")
    engine.sync_session(session_id=SID, embed=False)
    engine.audit_session(session_id=SID)
    assert seen and any(depth for _, depth in seen)
    assert [item for item in seen if item[1] and not item[0]] == []


# ---- finding 4: st_ino == 0 on one side carries no identity ----

from types import SimpleNamespace  # noqa: E402

from self_directing_mcp.session_events import same_identity  # noqa: E402


def _zero_fstat_ino(monkeypatch, dev=None):
    real = os.fstat

    def fake(fd):
        stat = real(fd)
        fields = {n: getattr(stat, n) for n in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}
        fields["st_ino"] = 0
        if dev is not None:
            fields["st_dev"] = dev
        return SimpleNamespace(**fields)
    monkeypatch.setattr(session_events.os, "fstat", fake)


def test_same_identity_helper():
    assert same_identity((1, 5), (1, 5))
    assert not same_identity((1, 5), (1, 6))
    assert not same_identity((1, 5), (2, 5))
    assert same_identity((1, 0), (1, 6)) and same_identity((9, 5), (1, 0))


@pytest.mark.parametrize("dev", [None, 12345])
def test_zero_inode_on_handle_side_still_captures_and_rejects_rewrites(tmp_path, monkeypatch, dev):
    path = tmp_path / "s.jsonl"
    path.write_bytes(b'{"a":1}\n' * 50)
    _zero_fstat_ino(monkeypatch, dev)
    observation = capture_source(path)
    with path.open("ab") as stream:
        stream.write(b'{"b":2}\n')
    assert observation.validate_prefix() and observation.validate_prefix(locked=True)
    assert [row[1] for row in session_events.iter_raw_lines(path)][:1] == [0]
    path.write_bytes(b'{"a":2}\n' * 10)  # size-changing rewrite
    assert not observation.validate_prefix() and not observation.validate_prefix(locked=True)


def test_nonzero_identity_mismatch_is_still_rejected(tmp_path, monkeypatch):
    path = tmp_path / "s.jsonl"
    path.write_bytes(b'{"a":1}\n')
    real = os.fstat

    def fake(fd):
        stat = real(fd)
        return SimpleNamespace(st_dev=stat.st_dev, st_ino=stat.st_ino + 1, st_size=stat.st_size,
                               st_mtime_ns=stat.st_mtime_ns, st_ctime_ns=stat.st_ctime_ns)
    monkeypatch.setattr(session_events.os, "fstat", fake)
    with pytest.raises(ValueError, match="source changed"):
        capture_source(path)
