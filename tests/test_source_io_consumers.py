"""A failed whole-prefix read must not be certified by a successful tail read.

Later stages of a request check the source with bounded reads; a whole-prefix re-read happens only when asked for
(``validate_prefix(full=True)``) or when the cheap signals demand it (the modification time moved while the size
did not). These tests arm a read fault and then make the file's modification time move, so the stage that must
settle it with a re-read really does, and the fault really hits it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from self_directing_mcp import agy_receipts, engine as engine_module, session_events
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine

SID = "11111111-2222-3333-4444-555555555555"
ACTION = {"tool_name": "shell", "arguments": {"command": "deploy now"}}


class _ReadFault:
    """Inject only readinto failures; bounded tail reads continue to work."""

    def __init__(self, monkeypatch, path, *, enabled=True, persistent=False):
        self.armed = False
        self.errors = 0
        real_open = Path.open
        fault = self

        class Reader:
            def __init__(self, stream):
                self.stream = stream

            def __getattr__(self, name):
                return getattr(self.stream, name)

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def readinto(self, target):
                if enabled and fault.armed and (persistent or fault.errors == 0):
                    fault.errors += 1
                    raise OSError("synthetic whole-prefix read failure")
                return self.stream.readinto(target)

        def open_source(candidate, mode="r", *args, **kwargs):
            stream = real_open(candidate, mode, *args, **kwargs)
            return Reader(stream) if candidate == path and mode == "rb" else stream

        monkeypatch.setattr(Path, "open", open_source)


def _touch(path):
    """Same bytes, newer modification time: what a rewrite that restores nothing looks like to the cheap check."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


@pytest.fixture
def source_session(tmp_path):
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / f"rollout-{SID}.jsonl"
    rows = [
        {"type": "session_meta", "payload": {"id": SID}},
        {"type": "response_item", "payload": {
            "type": "function_call", "name": "shell", "call_id": "test-ok",
            "arguments": {"command": "pytest"},
        }},
        {"type": "response_item", "payload": {
            "type": "function_call_output", "call_id": "test-ok",
            "output": json.dumps({"exit_code": 0, "output": "synthetic success"}),
        }},
        {"type": "response_item", "payload": {
            "type": "function_call", "name": "shell", "call_id": "old-deploy",
            "arguments": {"command": "deploy earlier"},
        }},
        # Push the prerequisite out of the bounded tail window that the locked check re-hashes.
        {"type": "response_item", "payload": {
            "type": "message", "role": "assistant", "content": "x" * 100000,
        }},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    engine = SelfDirectEngine(Settings(
        codex_sessions_dir=root, index_dir=tmp_path / "index", local_only=True,
        openrouter_api_key_file=None, _env_file=None,
    ))
    try:
        engine.upsert_contracts([{
            "id": "tests-first", "type": "must", "scope": "tool_call",
            "regex": "pytest", "before_regex": "deploy", "requires_success": True,
        }])
        assert engine.check_action(SID, ACTION)["verdict"] == "clean"
        yield engine, path
    finally:
        engine.close()


@pytest.mark.parametrize("operation", ["check_action", "audit_session"])
@pytest.mark.parametrize("rewrite", [False, True], ids=["unchanged", "rewritten"])
@pytest.mark.parametrize("io_failure", [False, True], ids=["read-ok", "read-error"])
def test_final_full_read_failure_cannot_publish_clean(
    source_session, monkeypatch, operation, rewrite, io_failure,
):
    engine, path = source_session
    invoke = lambda: (engine.check_action(SID, ACTION) if operation == "check_action"
                      else engine.audit_session(SID))
    assert invoke()["verdict"] == "clean"
    original = path.read_bytes()
    changed = original.replace(b"pytest", b"pyfail", 1)
    assert changed != original and len(changed) == len(original)
    assert changed[-session_events._TAIL_WINDOW:] == original[-session_events._TAIL_WINDOW:]
    fault = _ReadFault(monkeypatch, path, enabled=io_failure)
    native_run = engine_module.run_audit

    def arm_after_evaluation(*args, **kwargs):
        result = native_run(*args, **kwargs)
        if rewrite:
            path.write_bytes(changed)
        if rewrite or io_failure:
            _touch(path)  # whatever the file system's timestamp granularity: the modification time has moved
        fault.armed = True
        return result

    monkeypatch.setattr(engine_module, "run_audit", arm_after_evaluation)
    result = invoke()
    assert fault.errors == int(io_failure)
    expected = "unknown" if rewrite or io_failure else "clean"
    assert result["verdict"] == expected, result
    assert result["coverage"]["complete"] is (expected == "clean")
    assert result["requires_attention"] is (expected != "clean")
    assert result["execution"]["nodes"] == [
        "select_contracts", "load_evidence", "evaluate_contracts", "aggregate_findings",
    ]
    assert result["action_executed"] is False


@pytest.mark.parametrize("operation", ["check_action", "audit_session"])
def test_an_unchanged_source_is_not_reread_after_evaluation(source_session, monkeypatch, operation):
    """The efficiency half of the rule: with nothing moved, the stage check never re-reads the prefix, so an armed
    read fault is never hit and the verdict stays clean."""
    engine, path = source_session
    invoke = lambda: (engine.check_action(SID, ACTION) if operation == "check_action"
                      else engine.audit_session(SID))
    assert invoke()["verdict"] == "clean"
    fault = _ReadFault(monkeypatch, path, persistent=True)
    fault.armed = True
    result = invoke()
    assert fault.errors == 0 and result["verdict"] == "clean" and result["coverage"]["complete"] is True


def _checklist_change(engine):
    evidence = next(c for c in engine.store.list_chunks(SID, "codex") if c.kind == "tool_result")
    added = engine.update_checklist(SID, add=[{"text": "Verify synthetic behavior"}])
    return {"item_id": added["added"][0]["item_id"], "status": "done",
            "evidence_chunk_ids": [evidence.chunk_id]}


def test_checklist_save_never_publishes_verified_after_failed_full_read(source_session, monkeypatch):
    engine, path = source_session
    change = _checklist_change(engine)
    fault = _ReadFault(monkeypatch, path, persistent=True)
    save = engine._save_checklist

    def fail_only_final_save(*args, **kwargs):
        _touch(path)
        fault.armed = True
        try:
            return save(*args, **kwargs)
        finally:
            fault.armed = False

    monkeypatch.setattr(engine, "_save_checklist", fail_only_final_save)
    result = engine.update_checklist(SID, update=[change])
    assert result["ok"] is False and result["status"] == "checklist_conflict", result
    assert fault.errors == engine_module.CHECKLIST_EVALUATION_ATTEMPTS
    item = engine.get_checklist(SID)["items"][0]
    assert item["status"] == "open" and item["verified"] is False


def test_checklist_read_downgrades_done_after_failed_full_read(source_session, monkeypatch):
    engine, path = source_session
    result = engine.update_checklist(SID, update=[_checklist_change(engine)])
    assert result["updated"][0]["verified"] is True
    fault = _ReadFault(monkeypatch, path)
    verify = engine._verified_checklist_evidence

    def arm_after_verification(*args, **kwargs):
        verified = verify(*args, **kwargs)
        assert verified is True
        _touch(path)
        fault.armed = True
        return verified

    monkeypatch.setattr(engine, "_verified_checklist_evidence", arm_after_verification)
    item = engine.get_checklist(SID)["items"][0]
    assert fault.errors == 1
    assert item["status"] == "pending_verification" and item["verified"] is False


def test_graph_context_rejects_failed_full_read(source_session, monkeypatch):
    engine, path = source_session
    assert engine.get_graph_context(SID, path=str(path))["ok"] is True
    fault = _ReadFault(monkeypatch, path)
    refresh = engine._refresh_evidence

    def arm_after_refresh(*args, **kwargs):
        refreshed = refresh(*args, **kwargs)
        assert refreshed["ok"] and refreshed["coverage"]["complete"]
        _touch(path)
        fault.armed = True
        return refreshed

    monkeypatch.setattr(engine, "_refresh_evidence", arm_after_refresh)
    result = engine.get_graph_context(SID, path=str(path))
    assert fault.errors == 1
    assert result == {"ok": False, "error": "incomplete_history", "nodes": [], "edges": []}


@pytest.mark.parametrize("event", ["PreToolUse", "PostToolUse"])
def test_agy_receipt_preserves_prior_authority_after_failed_full_read(tmp_path, monkeypatch, event):
    path = tmp_path / "transcript.jsonl"
    path.write_text(json.dumps({"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT",
                                "status": "DONE", "content": "Synthetic user request"}) + "\n")
    receipts = agy_receipts.ReceiptStore(tmp_path / "index")
    tool = {"name": "run_command", "args": {"CommandLine": "pytest"}}
    if event == "PostToolUse":
        receipts.record(SID, path, 1, tool, "PreToolUse")
        with path.open("a") as stream:
            stream.write(json.dumps({"step_index": 1, "source": "MODEL", "type": "GENERIC",
                                     "status": "DONE", "content": "The command exited with code 0."}) + "\n")
    before = receipts.read_capture(SID)
    fault = _ReadFault(monkeypatch, path)
    completed = agy_receipts._completed_results

    def arm_after_inspection(*args, **kwargs):
        result = completed(*args, **kwargs)
        _touch(path)
        fault.armed = True
        return result

    monkeypatch.setattr(agy_receipts, "_completed_results", arm_after_inspection)
    with pytest.raises(ValueError, match="source changed during receipt preparation"):
        receipts.record(SID, path, 1, tool, event, error="" if event == "PostToolUse" else None)
    assert fault.errors == 1
    assert receipts.read_capture(SID) == before


def test_explicit_full_retry_recovers_transient_read_failure(tmp_path, monkeypatch):
    path = tmp_path / "source.jsonl"
    path.write_bytes(b'{"synthetic":true}\n')
    observation = session_events.capture_source(path)
    fault = _ReadFault(monkeypatch, path)
    fault.armed = True
    assert observation.validate_prefix(full=True) is False
    assert fault.errors == 1
    # These are distinct checks: callers must retain the failed full result even
    # when the bounded tail happens to remain readable (covered above).
    assert observation.validate_prefix(locked=True) is True
    assert observation.validate_prefix(full=True) is True
    assert observation.validate_prefix(locked=True) is True


def test_confirmed_mismatch_stays_invalid_after_read_retry(tmp_path, monkeypatch):
    path = tmp_path / "source.jsonl"
    original = b'{"synthetic":1}\n'
    path.write_bytes(original)
    observation = session_events.capture_source(path)
    path.write_bytes(b'{"synthetic":2}\n')
    assert observation.validate_prefix() is False
    path.write_bytes(original)
    fault = _ReadFault(monkeypatch, path)
    fault.armed = True
    assert observation.validate_prefix() is False
    assert observation.validate_prefix(locked=True) is False
    assert fault.errors == 0  # A proven mismatch cannot be healed by any retry.
