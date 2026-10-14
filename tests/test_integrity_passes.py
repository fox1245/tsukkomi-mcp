"""One full scan of the source prefix per request.

Before this change one ``check_action`` on a session hashed the whole indexed prefix five times: the capture pass and
four later re-checks (two stages of ingestion, coverage and the audit's own stage check). A request now scans the
prefix once, in the capture pass; every later stage re-reads a bounded tail window. The tests count the bytes that go
through SHA-256 (``hash_meter`` replaces ``hashlib.sha256`` for the whole process, so a hash site that bypasses the
helpers cannot hide) and divide by the file size.
"""
from __future__ import annotations

import json
import os

import pytest

from self_directing_mcp import session_events
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.session_events import capture_source
from test_append_capture import _AppendOnFirstFstat

SID = "11111111-2222-3333-4444-555555555555"
ACTION = {"tool_name": "shell", "arguments": {"command": "deploy now"}}
FILLER = "synthetic harmless output evidence " * 100  # ~3.5 KB per event


def _event(index):
    return {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": f"{index} {FILLER}"}}


def _write(path, megabytes):
    rows = [{"type": "session_meta", "payload": {"id": SID}},
            {"type": "response_item", "payload": {"type": "function_call", "name": "shell", "call_id": "t1",
                                                  "arguments": {"command": "pytest -q"}}},
            {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "t1",
                                                  "output": json.dumps({"exit_code": 0, "output": "ok"})}}]
    rows += [_event(i) for i in range(int(megabytes * 1024 * 1024 / 3600))]
    path.write_bytes("".join(json.dumps(row) + "\n" for row in rows).encode())


def _append(path, count, tag="late"):
    with path.open("ab") as stream:
        for index in range(count):
            stream.write((json.dumps({"type": "response_item", "payload": {
                "type": "message", "role": "assistant", "content": f"{tag} {index}"}}) + "\n").encode())


@pytest.fixture
def world(tmp_path):
    config = tmp_path / "empty.toml"
    config.write_text("")
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl"
    _write(path, 6)  # six 1 MiB blocks: the default block size, no monkeypatching
    engine = SelfDirectEngine(Settings(codex_sessions_dir=root, index_dir=tmp_path / "index", local_only=True,
                                       _env_file=None))
    engine.upsert_contracts([
        {"id": "tests-first", "type": "must", "scope": "tool_call", "regex": "pytest", "before_regex": "deploy"},
        {"id": "no-forbidden", "type": "must_not", "scope": "tool_call", "regex": "FORBIDDEN_MARKER"}])
    try:
        yield engine, path
    finally:
        engine.close()


def _passes(meter, path):
    return meter.lap() / path.stat().st_size


def test_capturing_a_source_is_one_pass(tmp_path, hash_meter):
    path = tmp_path / "s.jsonl"
    _write(path, 3)
    capture_source(path)
    assert 1.0 <= _passes(hash_meter, path) <= 1.05  # the pass, plus the 64 KiB tail window digest


def test_later_stage_checks_never_rehash_the_prefix(tmp_path, hash_meter):
    path = tmp_path / "s.jsonl"
    _write(path, 3)
    observation = capture_source(path)
    hash_meter.lap()
    for _ in range(4):
        assert observation.validate_prefix() and observation.validate_prefix(locked=True)
    assert hash_meter.lap() <= 8 * 2 * session_events._TAIL_WINDOW


def test_check_action_scans_the_prefix_once(world, hash_meter):
    engine, path = world
    assert engine.check_action(SID, ACTION)["verdict"] == "clean"  # indexes the session
    hash_meter.lap()
    result = engine.check_action(SID, ACTION)  # nothing appended
    assert result["verdict"] == "clean"
    passes = _passes(hash_meter, path)
    assert 1.0 <= passes <= 1.1, passes  # before this change: 5.0


def test_check_action_after_a_small_append_scans_the_prefix_once(world, hash_meter):
    engine, path = world
    engine.check_action(SID, ACTION)
    _append(path, 5)
    hash_meter.lap()
    assert engine.check_action(SID, ACTION)["verdict"] == "clean"
    assert 1.0 <= _passes(hash_meter, path) <= 1.1  # the parse reads and verifies only the appended records


def test_audit_session_scans_the_prefix_once(world, hash_meter):
    engine, path = world
    engine.audit_session(SID)
    hash_meter.lap()
    assert engine.audit_session(SID)["verdict"] == "clean"
    assert 1.0 <= _passes(hash_meter, path) <= 1.1


def test_the_first_sync_scans_the_prefix_once(world, hash_meter):
    engine, path = world
    assert engine._sync_session(session_id=SID, provider="codex", embed=False, derived=False)["ok"]
    # The capture pass (1), the check of every consumed byte against the pass (1) and the digests of the event
    # texts the parser makes (1): the source prefix is scanned once, then parsed. Before this change: 6.0.
    assert _passes(hash_meter, path) <= 3.2


def test_growth_during_the_capture_pass_does_not_cost_a_second_scan(world, hash_meter, monkeypatch):
    """The old code re-hashed the whole prefix when the file grew while it was being hashed."""
    engine, path = world
    engine.check_action(SID, ACTION)
    monkeypatch.setattr(session_events, "os", _AppendOnFirstFstat(path, b'{"type":"x"}\n'))
    hash_meter.lap()
    result = engine.check_action(SID, ACTION)
    monkeypatch.undo()
    assert result["verdict"] == "clean" and "unread_bytes" in result["coverage"]["issues"]
    assert _passes(hash_meter, path) <= 1.1


def test_a_rewrite_is_still_found_by_the_single_pass(world):
    """One pass is enough to see a same-size rewrite that happened before the request started."""
    engine, path = world
    assert engine.check_action(SID, ACTION)["verdict"] == "clean"
    data = path.read_bytes()
    stat = path.stat()
    path.write_bytes(data.replace(b"pytest -q", b"pyfail -q", 1))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    info = engine.sync_session(session_id=SID, embed=False)
    assert info["reindexed"] and info["reindex_reason"] == "source_changed"
    assert engine.check_action(SID, ACTION)["verdict"] != "clean"
