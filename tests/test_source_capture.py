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


@pytest.mark.skipif(os.name == "nt", reason="Windows st_ctime is the creation time, so a same-size rewrite "
                    "that restores mtime is invisible to stat; the cursor prefix hash catches it on the next sync")
def test_rewrite_after_capture_invalidates_observation(source):
    observation = capture_source(source)
    st = source.stat()
    source.write_bytes(b'{"a":2}\n')
    os.utime(source, ns=(st.st_atime_ns, st.st_mtime_ns))  # same size, restored mtime; ctime moves
    assert not observation.validate()
    assert not observation.validate_prefix()
