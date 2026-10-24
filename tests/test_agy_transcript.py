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
    chunks = engine.store.list_chunks(SID, "agy")
    # Denial text is not proof of non-execution without the host's own Pre receipt; with no receipt at all the
    # conversation predates receipts, so the result is preserved as legacy_unlinked and result-dependent
    # obligations stay unknown (unsupported_legacy), never clean.
    assert not [c for c in chunks if c.meta.get("evidence_origin") == "host_rejected_proposal"]
    outcome = next(c for c in chunks if c.kind == "tool_result")
    assert outcome.meta["linkage"] == "legacy_unlinked" and outcome.meta["success"] is False
    assert result["coverage"]["result_dependent_complete"] is False
    assert _deploy_check(engine, path)["verdict"] == "unknown"


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


@pytest.mark.parametrize("tool_name,tool_args,result_text", [
    ("read_file", {"Path": "/synthetic/project/README.md"}, "# Synthetic Readme\n"),
    ("call_mcp_tool", {"ServerName": "tsukkomi-mcp", "ToolName": "list_contracts", "Arguments": {}}, "{\"ok\": true}"),
    ("edit_file", {"Path": "/synthetic/project/app.py", "Target": "a", "Replacement": "b"}, "Edited file"),
    ("write_file", {"Path": "/synthetic/project/new.py", "Content": "x = 1\n"}, "Created file"),
])
def test_receipt_linked_non_command_generic_results_are_supported_and_preserve_complete_coverage(
    session, tool_name, tool_args, result_text
):
    from self_directing_mcp.agy_hooks import handle_pre_invocation

    engine, path = session
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["tool_calls"] = [{"name": tool_name, "args": {k: json.dumps(v) for k, v in tool_args.items()}}]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    payload = {
        "conversationId": SID,
        "transcriptPath": str(path),
        "workspacePaths": ["/synthetic/project"],
        "stepIdx": 2,
        "toolCall": {"name": tool_name, "args": tool_args},
        "error": "",
    }
    engine.record_agy_hook(payload, "PreToolUse")
    with path.open("a") as stream:
        stream.write(json.dumps({
            "step_index": 2,
            "source": "MODEL",
            "type": "GENERIC",
            "status": "DONE",
            "created_at": "2026-10-01T00:00:02Z",
            "content": result_text,
        }) + "\n")
    engine.record_agy_hook(payload, "PostToolUse")

    synced = engine.sync_session(session_id=SID, provider="agy", path=str(path), embed=False)
    assert synced["coverage"]["complete"] is True
    assert synced["coverage"]["unsupported_event_count"] == 0
    assert "unsupported_events" not in synced["coverage"]["issues"]

    chunks = engine.store.list_chunks(SID, "agy")
    outcome = next(c for c in chunks if c.kind == "tool_result")
    assert outcome.meta["linkage"] == "host_step_receipt"
    assert outcome.meta["tool_name"] == tool_name
    assert outcome.meta["unsupported_event"] is False
    assert outcome.meta["success"] is None

    ctx = engine.get_graph_context(session_id=SID, provider="agy", path=str(path))
    assert ctx["ok"] is True

    hook_out = handle_pre_invocation(engine, SID, {
        "conversationId": SID,
        "transcriptPath": str(path),
        "workspacePaths": ["/synthetic/project"],
        "prompt": "check status",
    })
    additional = hook_out.get("hookSpecificOutput", {}).get("additionalContext", "")
    assert "AGY history/context unavailable" not in additional


@pytest.mark.parametrize("content,expected_success", [
    (
        "Created At: 2026-10-01T00:00:02+09:00\nCompleted At: 2026-10-01T00:00:03+09:00\n"
        "Command exited with code 0.\n\nCommand output:\nall passed\n",
        True,
    ),
    (
        "Created At: 2026-10-01T00:00:02+09:00\nCompleted At: 2026-10-01T00:00:03+09:00\n"
        "Command exited with code 2.\n\nCommand output:\nfailed\n",
        False,
    ),
    (
        "Created At: 2026-10-01T00:00:02+09:00\nCompleted At: 2026-10-01T00:00:03+09:00\n"
        "Command exited with code 0.\n\nNo output from the command.",
        True,
    ),
    (
        "Created At: 2026-10-01T00:00:02+09:00\nCompleted At: 2026-10-01T00:00:03+09:00\n"
        "The tool's output was truncated due to being too long. "
        "You can find the full output in a file at file:///synthetic/steps/10/output.txt.\n\n"
        "Here is a preview with the middle of the output truncated:\n"
        "Command exited with code 0.\n\nCommand output:\nhead\n<truncated>\ntail\n",
        True,
    ),
    (
        "Created At: 2026-10-01T00:00:02+09:00\n"
        "Tool is running as a background task with task id: task-123\n"
        "Task logs are available at: /synthetic/tasks/task-123.log\n",
        None,
    ),
])
def test_modern_agy_cli_run_command_headers_and_background_tasks_are_shaped(session, content, expected_success):
    engine, path = session
    payload = _receipt_payload(path)
    engine.record_agy_hook(payload, "PreToolUse")
    with path.open("a") as stream:
        stream.write(json.dumps({
            "step_index": 2,
            "source": "MODEL",
            "type": "GENERIC",
            "status": "DONE",
            "created_at": "2026-10-01T00:00:02Z",
            "content": content,
        }) + "\n")
    engine.record_agy_hook(payload, "PostToolUse")

    synced = engine.sync_session(session_id=SID, provider="agy", path=str(path), embed=False)
    assert synced["coverage"]["complete"] is True
    assert synced["coverage"]["unsupported_event_count"] == 0
    outcome = next(c for c in engine.store.list_chunks(SID, "agy") if c.kind == "tool_result")
    assert outcome.meta["unsupported_event"] is False
    assert outcome.meta["success"] is expected_success


def test_unshaped_run_command_generic_result_remains_unsupported_and_never_succeeds(session):
    engine, path = session
    payload = _receipt_payload(path)
    engine.record_agy_hook(payload, "PreToolUse")
    with path.open("a") as stream:
        stream.write(json.dumps({
            "step_index": 2,
            "source": "MODEL",
            "type": "GENERIC",
            "status": "DONE",
            "created_at": "2026-10-01T00:00:02Z",
            "content": "Command exited with code 0.\n\nCommand output:\nmissing host timestamps\n",
        }) + "\n")
    engine.record_agy_hook(payload, "PostToolUse")

    synced = engine.sync_session(session_id=SID, provider="agy", path=str(path), embed=False)
    assert synced["coverage"]["complete"] is False
    assert synced["coverage"]["unsupported_event_count"] == 1
    outcome = next(c for c in engine.store.list_chunks(SID, "agy") if c.kind == "tool_result")
    assert outcome.meta["unsupported_event"] is True
    assert outcome.meta["success"] is None


def test_parser_version_9_bump_reindexes_previously_synced_agy_session(session, monkeypatch):
    from self_directing_mcp import agy

    engine, path = session
    assert agy.PARSER_VERSION == 9
    with monkeypatch.context() as old:
        old.setattr(agy, "PARSER_VERSION", 8)
        first = engine.sync_session(session_id=SID, provider="agy", path=str(path), embed=False)
        assert first["parser_version"] == 8
    second = engine.sync_session(session_id=SID, provider="agy", path=str(path), embed=False)
    assert second["reindexed"] is True
    assert second["reindex_reason"] == "parser_updated"
    assert second["parser_version"] == 9
