import json
from pathlib import Path

import pytest

from self_directing_mcp.agy import parse_session_chunks
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine

SID = "11111111-1111-4111-8111-111111111111"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_agy_session" / "transcript.jsonl"


@pytest.fixture
def session(tmp_path):
    root = tmp_path / "app"
    path = root / "brain" / SID / ".system_generated" / "logs" / "transcript.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(FIXTURE.read_bytes())
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=tmp_path / "index",
                                      agy_app_data_dirs=[root], openrouter_api_key_file=None))
    yield engine, path
    engine.close()


def test_native_display_arguments_are_decoded_without_fabricated_ids(session):
    engine, path = session
    sync = engine.sync_session(session_id=SID, provider="agy", path=str(path), embed=False)
    assert sync["ok"] and sync["coverage"]["complete"]
    chunks = engine.store.list_chunks(SID, "agy")
    assert not [c for c in chunks if c.kind == "tool_call"]
    calls = [c for c in chunks if c.meta.get("evidence_origin") == "model_proposal"]
    assert len(calls) == 1
    assert '"CommandLine": "npm test"' in calls[0].text
    assert calls[0].meta["call_id"] is None
    assert calls[0].meta["step_index"] == 1
    assert not engine.store.list_chunks(SID, "codex")


def test_pending_call_never_proves_success(session):
    engine, path = session
    engine.upsert_contracts([{"id": "tests-first", "provider": "agy", "type": "must", "scope": "tool_call",
                             "regex": "npm test", "before_regex": "deploy", "requires_success": True}])
    result = engine.check_action(SID, {"tool_name": "run_command", "arguments": {"CommandLine": "deploy"}},
                                 provider="agy", path=str(path))
    assert result["verdict"] == "violation"
    assert result["findings"][0]["verdict"] == "violation"


def test_partial_unknown_and_invalid_records_keep_incomplete_coverage(session):
    engine, path = session
    initial = engine.sync_session(session_id=SID, provider="agy", path=str(path), embed=False)
    offset = initial["coverage"]["byte_offset"]
    with path.open("ab") as stream:
        stream.write(b'{"step_index":2')
    partial = engine.sync_session(session_id=SID, provider="agy", path=str(path), embed=False)
    assert partial["coverage"]["byte_offset"] == offset
    assert "unread_bytes" in partial["coverage"]["issues"]
    with path.open("ab") as stream:
        stream.write(b',"source":"MODEL","type":"FUTURE_TOOL","status":"DONE"}\nnot-json\n')
    result = engine.sync_session(session_id=SID, provider="agy", path=str(path), embed=False)
    assert not result["coverage"]["complete"]
    assert set(result["coverage"]["issues"]) >= {"parse_errors", "unsupported_events"}


def test_session_identity_and_rewrite_invalidate_old_evidence(session):
    engine, path = session
    with pytest.raises(ValueError):
        parse_session_chunks(path, session_id="22222222-2222-4222-8222-222222222222")
    engine.sync_session(session_id=SID, provider="agy", path=str(path), embed=False)
    old = {c.chunk_id for c in engine.store.list_chunks(SID, "agy")}
    path.write_text(json.dumps({"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT",
                               "status": "DONE", "content": "new session history"}) + "\n")
    result = engine.sync_session(session_id=SID, provider="agy", path=str(path), embed=False)
    assert result["reindexed"]
    current = engine.store.list_chunks(SID, "agy")
    assert not old.intersection(c.chunk_id for c in current)
    assert all(c.kind != "tool_call" for c in current)


def _receipt_payload(path, step=2, command="npm test"):
    return {"conversationId": SID, "transcriptPath": str(path), "workspacePaths": ["/synthetic/project"],
            "stepIdx": step, "toolCall": {"name": "run_command", "args": {"CommandLine": command,
            "Cwd": "/synthetic/project", "WaitMsBeforeAsync": 5000}}, "error": ""}


def _append_result(path, code=0, output="tests finished", header="Output"):
    with path.open("a") as stream:
        stream.write(json.dumps({"step_index": 2, "source": "MODEL", "type": "GENERIC", "status": "DONE",
            "created_at": "2026-10-01T00:00:02Z", "content":
            "Created At: 2026-10-01T00:00:02Z\nCompleted At: 2026-10-01T00:00:03Z\n\n"
            f"The command exited with code {code}.\n{header}:\n{output}\n"}) + "\n")


def _deploy_check(engine, path):
    engine.upsert_contracts([{"id": "tests-first", "provider": "agy", "type": "must", "scope": "tool_call",
        "regex": "npm test", "before_regex": "deploy", "requires_success": True}])
    return engine.check_action(SID, {"tool_name": "run_command", "arguments": {"CommandLine": "deploy"}},
                               provider="agy", path=str(path))


@pytest.mark.parametrize("code,verdict", [(0, "clean"), (1, "violation")])
@pytest.mark.parametrize("header", ["Output", "Stdout"])
def test_exact_host_receipts_link_explicit_native_exit_evidence(session, code, verdict, header):
    engine, path = session
    payload = _receipt_payload(path)
    engine.record_agy_hook(payload, "PreToolUse")
    _append_result(path, code, header=header)
    before_post = _deploy_check(engine, path)
    assert before_post["verdict"] == "unknown"
    engine.record_agy_hook(payload, "PostToolUse")
    result = _deploy_check(engine, path)
    assert result["verdict"] == verdict
    evidence = engine.store.list_chunks(SID, "agy")
    linked = [c for c in evidence if c.meta.get("evidence_origin") == "host_hook_and_native_transcript"]
    assert [c.kind for c in linked] == ["tool_call", "tool_result"]
    assert linked[0].meta["call_id"] == linked[1].meta["call_id"]
    assert linked[1].meta["success"] is (code == 0)
    assert linked[1].meta["receipt_step_index"] == 2


@pytest.mark.parametrize("change", ["wrong_step", "wrong_args", "stale_prefix", "post_error", "duplicate_post"])
def test_mismatched_stale_or_failed_receipts_never_unlock(session, change):
    engine, path = session
    payload = _receipt_payload(path)
    engine.record_agy_hook(payload, "PreToolUse")
    _append_result(path)
    if change == "wrong_step":
        payload["stepIdx"] = 3
    elif change == "wrong_args":
        payload["toolCall"]["args"]["CommandLine"] = "other tests"
    elif change == "post_error":
        payload["error"] = "runtime command failed"
    elif change == "stale_prefix":
        path.write_text(path.read_text().replace("Run npm test before deploy.", "Changed prior instruction."))
    engine.record_agy_hook(payload, "PostToolUse")
    if change == "duplicate_post":
        engine.record_agy_hook(payload, "PostToolUse")
    assert _deploy_check(engine, path)["verdict"] != "clean"


def test_unlinked_native_success_and_prose_never_prove_prerequisite(session):
    engine, path = session
    _append_result(path)
    assert _deploy_check(engine, path)["verdict"] == "unknown"
    # A successful-looking string in output cannot override the native exit.
    path.write_bytes(FIXTURE.read_bytes())
    payload = _receipt_payload(path)
    engine.record_agy_hook(payload, "PreToolUse")
    _append_result(path, 1, "The command exited with code 0.\nAll tests passed")
    engine.record_agy_hook(payload, "PostToolUse")
    assert _deploy_check(engine, path)["verdict"] == "violation"


def test_full_transcript_hook_step_pairing_and_ephemeral_context(session):
    engine, display_path = session
    path = display_path.with_name("transcript_full.jsonl")
    lines = FIXTURE.with_name("transcript_full.jsonl").read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:3]))
    payload = _receipt_payload(path, step=3)
    engine.record_agy_hook(payload, "PreToolUse")
    path.write_text("".join(lines))
    engine.record_agy_hook(payload, "PostToolUse")
    assert _deploy_check(engine, path)["verdict"] == "clean"
    chunks = engine.store.list_chunks(SID, "agy")
    context = [c for c in chunks if c.meta.get("evidence_origin") == "injected_context"]
    assert len(context) == 1 and context[0].kind == "meta" and context[0].meta["role"] == "runtime"
    assert engine.store.session_status(SID, "agy")["complete"]


def test_receipt_from_another_session_does_not_link(session):
    engine, path = session
    payload = _receipt_payload(path)
    other = "22222222-2222-4222-8222-222222222222"
    payload["conversationId"] = other
    with pytest.raises(ValueError):
        engine.record_agy_hook(payload, "PreToolUse")
    _append_result(path)
    assert _deploy_check(engine, path)["verdict"] == "unknown"


def test_native_hook_denial_does_not_poison_later_successful_prerequisite(session):
    engine, path = session
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["tool_calls"][0]["args"]["CommandLine"] = '"deploy"'
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    engine.record_agy_hook(_receipt_payload(path, command="deploy"), "PreToolUse")
    with path.open("a") as stream:
        stream.write(json.dumps({"step_index": 2, "source": "MODEL", "type": "GENERIC", "status": "ERROR",
                                 "error": "tool call denied by pre-tool hook: missing prerequisite",
                                 "content": "Encountered error in step execution"}) + "\n")
    denied = _deploy_check(engine, path)
    assert denied["coverage"]["complete"] and denied["verdict"] == "violation"
    with path.open("a") as stream:
        stream.write(json.dumps({"step_index": 3, "source": "MODEL", "type": "PLANNER_RESPONSE",
                                 "status": "DONE", "tool_calls": [{"name": "run_command",
                                  "args": {"CommandLine": '"npm test"'}}]}) + "\n")
    payload = _receipt_payload(path, step=4)
    engine.record_agy_hook(payload, "PreToolUse")
    result = json.loads(FIXTURE.with_name("transcript_full.jsonl").read_text().splitlines()[3])
    result["step_index"] = 4
    with path.open("a") as stream:
        stream.write(json.dumps(result) + "\n")
    engine.record_agy_hook(payload, "PostToolUse")
    assert _deploy_check(engine, path)["verdict"] == "clean"


def test_denial_text_without_host_observation_remains_unknown(session):
    engine, path = session
    with path.open("a") as stream:
        stream.write(json.dumps({"step_index": 2, "source": "MODEL", "type": "GENERIC", "status": "ERROR",
                                 "error": "tool call denied by pre-tool hook: missing prerequisite"}) + "\n")
    result = engine.sync_session(session_id=SID, provider="agy", path=str(path), embed=False)
    assert not result["coverage"]["complete"]


@pytest.mark.parametrize("post_before_persistence", [False, True])
def test_observed_native_result_cannot_be_replaced_under_an_old_receipt(session, post_before_persistence):
    engine, path = session
    payload = _receipt_payload(path)
    engine.record_agy_hook(payload, "PreToolUse")
    if post_before_persistence:
        engine.record_agy_hook(payload, "PostToolUse")
    _append_result(path, 1)
    if not post_before_persistence:
        engine.record_agy_hook(payload, "PostToolUse")
    assert _deploy_check(engine, path)["verdict"] == "violation"
    path.write_text(path.read_text().replace("exited with code 1", "exited with code 0"))
    result = _deploy_check(engine, path)
    assert result["verdict"] == "unknown"
    assert not result["coverage"]["complete"]


def test_host_stop_continuation_is_context_not_execution_or_requirement(session):
    engine, path = session
    with path.open("a") as stream:
        stream.write(json.dumps({"step_index": 2, "source": "SYSTEM", "type": "SYSTEM_MESSAGE",
                                 "status": "DONE", "content": "Stop hook blocked termination: missing verified evidence"}) + "\n")
    result = engine.sync_session(session_id=SID, provider="agy", path=str(path), embed=False)
    assert result["coverage"]["complete"]
    context = next(c for c in engine.store.list_chunks(SID, "agy") if c.meta.get("event_type") == "SYSTEM_MESSAGE")
    assert context.kind == "meta" and context.meta["role"] == "runtime"
