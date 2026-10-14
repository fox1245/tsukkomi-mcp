from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace

import pytest

from self_directing_mcp import session_events
from self_directing_mcp.session_events import capture_source, source_stamp

CONTENT = b'{"a":1}\n'


def _skew(real, **changes):
    fields = {name: getattr(real, name) for name in
              ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}
    fields.update(changes)
    return SimpleNamespace(**fields)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_bytes(CONTENT)
    return path


def _fstat_sequence(monkeypatch, *transforms):
    """Replace os.fstat inside session_events; transforms apply to successive calls."""
    real = os.fstat
    calls = iter(transforms)

    def fake(fd):
        stat = real(fd)
        change = next(calls, {})
        return _skew(stat, **{k: (v(stat) if callable(v) else v) for k, v in change.items()})

    monkeypatch.setattr(session_events.os, "fstat", fake)


def _freeze_source_metadata(monkeypatch, path, stat):
    """Model a filesystem whose stamps cannot distinguish same-size rewrites."""
    real_os = session_events.os
    real_path_stat = type(path).stat

    class FrozenFileMetadata:
        def __getattr__(self, name):
            return getattr(real_os, name)

        def fstat(self, fd):
            current = real_os.fstat(fd)
            if (current.st_dev, current.st_ino) == (stat.st_dev, stat.st_ino):
                return stat
            return current

    def path_stat(candidate, *args, **kwargs):
        if candidate == path:
            return stat
        return real_path_stat(candidate, *args, **kwargs)

    monkeypatch.setattr(session_events, "os", FrozenFileMetadata())
    monkeypatch.setattr(type(path), "stat", path_stat)


def test_stable_source_is_captured(source):
    observation = capture_source(source)
    assert observation.length == len(CONTENT) and observation.complete_end == len(CONTENT)
    assert observation.prefix_digest(len(CONTENT)) == hashlib.sha256(CONTENT).hexdigest()
    assert observation.validate() and observation.stamp == source_stamp(source)


def test_fstat_with_different_ctime_semantics_is_accepted(source, monkeypatch):
    # Runtimes (e.g. 9p/drvfs) may report a different, but self-consistent, ctime via fstat.
    _fstat_sequence(monkeypatch, {"st_ctime_ns": 1}, {"st_ctime_ns": 1})
    observation = capture_source(source)
    assert observation.length == len(CONTENT)
    # The strict check compares stat with stat, so the skewed fstat ctime must not make it fail.
    assert observation.validate()


@pytest.mark.parametrize("field", ["st_mtime_ns", "st_ctime_ns"])
def test_same_size_handle_stat_change_during_read_is_rejected(source, monkeypatch, field):
    _fstat_sequence(monkeypatch, {}, {field: lambda s: getattr(s, field) + 1})
    with pytest.raises(ValueError, match="source changed"):
        capture_source(source)


@pytest.mark.parametrize("field", ["st_dev", "st_ino", "st_mtime_ns"])
def test_handle_and_path_identity_must_agree(source, monkeypatch, field):
    bump = {field: lambda s: getattr(s, field) + 1}
    _fstat_sequence(monkeypatch, bump, bump)
    with pytest.raises(ValueError, match="source changed"):
        capture_source(source)


def test_handle_larger_than_path_is_rejected(source, monkeypatch):
    bump = {"st_size": lambda s: s.st_size + 1}
    _fstat_sequence(monkeypatch, bump, bump)
    with pytest.raises(ValueError, match="source (changed|truncated)"):
        capture_source(source)


def test_rewrite_after_capture_invalidates_observation(source):
    observation = capture_source(source)
    st = source.stat()
    source.write_bytes(b'{"a":2}\n')
    os.utime(source, ns=(st.st_atime_ns, st.st_mtime_ns))
    # Metadata equality is not proof of byte equality: ctime may not advance on
    # a rapid rewrite (or may denote creation time on another filesystem).
    assert observation.validate() == (source_stamp(source) == observation.stamp)
    assert not observation.validate_prefix()


@pytest.mark.parametrize("locked", [False, True])
def test_same_stamp_rewrite_is_detected_from_actual_bytes(source, monkeypatch, locked):
    observation = capture_source(source)
    st = source.stat()
    source.write_bytes(b'{"a":2}\n')
    os.utime(source, ns=(st.st_atime_ns, st.st_mtime_ns))
    _freeze_source_metadata(monkeypatch, source, st)
    assert source.read_bytes() != CONTENT
    assert source_stamp(source) == observation.stamp
    assert observation.validate() is True  # This method checks metadata only.
    assert observation.validate_prefix(locked=locked) is False


@pytest.mark.parametrize("first_locked", [False, True])
@pytest.mark.parametrize("again_locked", [False, True])
def test_failed_prefix_does_not_heal_after_original_bytes_return(
    source, monkeypatch, first_locked, again_locked,
):
    observation = capture_source(source)
    st = source.stat()
    source.write_bytes(b'{"a":2}\n')
    # Force the first verification to read bytes even in the old implementation,
    # then restore both content and stamps to isolate the failed-check latch.
    with monkeypatch.context() as changed:
        _freeze_source_metadata(changed, source, _skew(st, st_mtime_ns=st.st_mtime_ns + 1))
        assert observation.validate_prefix(locked=first_locked) is False
    source.write_bytes(CONTENT)
    os.utime(source, ns=(st.st_atime_ns, st.st_mtime_ns))
    _freeze_source_metadata(monkeypatch, source, st)
    assert source.read_bytes() == CONTENT
    assert observation.validate() is True
    assert observation.validate_prefix(locked=again_locked) is False


@pytest.mark.parametrize("content", [b"", CONTENT, b'{"a":"' + b"x" * 80 + b'"}\n'])
@pytest.mark.parametrize("max_read", [5, 16])
def test_full_prefix_hash_reuses_one_bounded_buffer_without_reading_appends(
    source, monkeypatch, content, max_read,
):
    # Small blocks exercise multiple reads without a large CI fixture.
    monkeypatch.setattr(session_events, "_BLOCK", 16)
    source.write_bytes(content)
    observation = capture_source(source)
    with source.open("ab") as output:
        output.write(b"unobserved appended bytes\n")
    buffers, positions, lengths = [], [], []
    real_open = type(source).open

    class ReadintoProbe:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            positions.append(self.stream.tell())
            return self.stream.__exit__(*args)

        def read(self, *args):
            pytest.fail("Full-prefix verification must reuse its readinto buffer")

        def readinto(self, target):
            assert isinstance(target, memoryview)
            assert isinstance(target.obj, bytearray)
            assert len(target.obj) == min(len(content), 16)
            assert 0 < len(target) <= min(len(content) - self.stream.tell(), 16)
            buffers.append(id(target.obj))
            lengths.append(len(target))
            return self.stream.readinto(target[:max_read])

    def open_source(candidate, *args, **kwargs):
        stream = real_open(candidate, *args, **kwargs)
        return ReadintoProbe(stream) if candidate == source else stream

    monkeypatch.setattr(type(source), "open", open_source)
    assert observation.validate_prefix() is True
    assert positions == [len(content)]
    assert len(set(buffers)) == (1 if content else 0)
    if len(content) > 16:
        assert len(buffers) > 1
        assert lengths[-1] < 16


def test_full_prefix_hash_rejects_early_eof_even_with_unchanged_stamps(source, monkeypatch):
    observation = capture_source(source)
    st = source.stat()
    source.write_bytes(CONTENT[:-1])
    _freeze_source_metadata(monkeypatch, source, st)
    assert observation.validate() is True
    assert observation.validate_prefix() is False
    assert observation.validate_prefix(locked=True) is False
