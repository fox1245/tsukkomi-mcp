"""AGY parser v6: truncated-field fallback, partial observation, status values, legacy classification.

Synthetic data only; shapes mirror the host log: records are
{type, source, status, step_index, created_at, content?, tool_calls?, thinking?, error?, media?,
 truncated_fields?}. The display log (transcript.jsonl) JSON-encodes every tool-call argument value;
the full log (transcript_full.jsonl) keeps native values.
"""
import base64
import hashlib
import json
import tracemalloc
from pathlib import Path

import pytest

from self_directing_mcp import agy
from self_directing_mcp.agy import parse_session_chunks
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine

SID = "33333333-3333-4333-8333-333333333333"
SID2 = "44444444-4444-4444-8444-444444444444"


# --- synthetic record builders -------------------------------------------------------------------

def rec(step, kind, source, status="DONE", **fields):
    return {"step_index": step, "source": source, "type": kind, "status": status,
            "created_at": f"2026-10-01T00:{step // 60:02d}:{step % 60:02d}Z", **fields}


def user(step, text, **kw):
    return rec(step, "USER_INPUT", "USER_EXPLICIT", content=text, **kw)


def planner(step, content="", calls=None, *, display=True, status="DONE", **kw):
    row = rec(step, "PLANNER_RESPONSE", "MODEL", status, **kw)
    if content:
        row["content"] = content
    if calls is not None:
        row["tool_calls"] = [{"name": name, "args": ({k: json.dumps(v) for k, v in args.items()}
                                                      if display else dict(args))} for name, args in calls]
    return row


def generic(step, output="tests finished", code=0, status="DONE", **kw):
    header = (f"Created At: 2026-10-01T00:00:{step % 60:02d}Z\nCompleted At: 2026-10-01T00:00:{step % 60:02d}Z\n\n"
              f"The command exited with code {code}.\nOutput:\n{output}\n")
    return rec(step, "GENERIC", "MODEL", status, content=header, **kw)


def logs(root, sid=SID):
    folder = root / "brain" / sid / ".system_generated" / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    return path


def append(path, *records):
    with path.open("a") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")


def make(root, display, full=None, sid=SID):
    folder = logs(root, sid)
    path = write(folder / "transcript.jsonl", display)
    if full is not None:
        write(folder / "transcript_full.jsonl", full)
    return path


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "app"
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=tmp_path / "index",
                                      agy_app_data_dirs=[root], openrouter_api_key_file=None))
    yield engine, root
    engine.close()


def sync(engine, path, sid=SID):
    return engine.sync_session(session_id=sid, provider="agy", path=str(path), embed=False)


def chunks_of(engine, sid=SID):
    return engine.store.list_chunks(sid, "agy")


def hook(path, step, command, sid=SID, error=""):
    return {"conversationId": sid, "transcriptPath": str(path), "workspacePaths": ["/synthetic/project"],
            "stepIdx": step, "toolCall": {"name": "run_command", "args": {"CommandLine": command,
            "Cwd": "/synthetic/project", "WaitMsBeforeAsync": 5000}}, "error": error}


def finding(result, contract_id):
    return next(item for item in result["findings"] if item["contract_id"] == contract_id)


def must_tests_first(engine, **extra):
    engine.upsert_contracts([{"id": "tests-first", "provider": "agy", "type": "must", "scope": "tool_call",
                              "regex": "npm test", "before_regex": "deploy", "requires_success": True, **extra}])


def deploy_check(engine, path, sid=SID):
    return engine.check_action(sid, {"tool_name": "run_command", "arguments": {"CommandLine": "deploy"}},
                               provider="agy", path=str(path))


# --- 1. fallback to the sibling full transcript --------------------------------------------------

def test_parser_version_is_six():
    assert agy.PARSER_VERSION == 6


def test_truncated_fields_restore_from_sibling_full_transcript(tmp_path):
    display = make(tmp_path, [
        user(0, "Run the tests", truncated_fields=["content"]),
        planner(1, "I will run", [("run_command", {"CommandLine": "npm te"})], truncated_fields=["content", "tool_calls"]),
        generic(2, "tests fini", truncated_fields=["content"]),
    ], [
        user(0, "Run the tests, then deploy to staging only"),
        planner(1, "I will run the whole suite now", [("run_command", {"CommandLine": "npm test --coverage"})], display=False),
        generic(2, "tests finished\n" * 30),
    ])
    chunks, _offset, _sid = parse_session_chunks(display)
    first = next(c for c in chunks if c.meta["event_type"] == "USER_INPUT")
    assert first.text == "[message:user] Run the tests, then deploy to staging only"
    assert first.meta["source"] == "full" and first.meta["restored_fields"] == ["content"]
    assert "truncated_fields" not in first.meta
    message = next(c for c in chunks if c.meta["event_type"] == "PLANNER_RESPONSE" and c.kind == "turn")
    assert message.text == "[message:assistant] I will run the whole suite now" and message.meta["source"] == "full"
    proposal = next(c for c in chunks if c.meta.get("evidence_origin") == "model_proposal")
    assert '"CommandLine": "npm test --coverage"' in proposal.text and proposal.meta["source"] == "full"
    assert "truncated_fields" not in proposal.meta
    result = next(c for c in chunks if c.kind == "tool_result")
    assert result.text.count("tests finished") == 30 and result.meta["source"] == "full"
    assert not [c for c in chunks if c.meta.get("parse_error") or c.meta.get("truncated_fields")]


def test_fallback_is_per_field_and_leaves_thinking_unrestored(tmp_path):
    display = make(tmp_path, [
        planner(0, "complete", None, thinking="short", truncated_fields=["content", "thinking"]),
    ], [
        planner(0, "complete answer", None, thinking="long long thinking", display=False),
    ])
    chunks, *_ = parse_session_chunks(display)
    message = next(c for c in chunks if c.kind == "turn")
    assert message.text == "[message:assistant] complete answer"
    assert message.meta["restored_fields"] == ["content"]
    assert message.meta["truncated_fields"] == ["thinking"]


FAILURES = [
    # name, display step, full records, reason, retryable
    ("missing_full", 0, None, "full_missing", True),
    ("full_also_truncated", 0, [user(0, "Do not touch production, unless approved by", truncated_fields=["content"])],
     "full_truncated", False),
    ("wrong_type", 0, [planner(0, "Do not touch production, unless approved by the owner", None, display=False)],
     "full_step_mismatch", False),
    ("step_gap", 0, [generic(1), generic(2)], "full_missing_step", False),
    ("not_written_yet", 3, [generic(0), generic(1), generic(2)], "full_missing_step", True),
    ("not_longer", 0, [user(0, "Do not touch production, unl")], "full_not_longer", False),
    ("other_text_same_step", 0, [user(0, "Something else entirely, much longer than what the display shows")],
     "full_prefix_mismatch", False),
    ("duplicate_step", 0, [user(0, "Do not touch production, unless approved by the owner"),
                           user(0, "Do not touch production, unless approved by somebody else")],
     "full_duplicate_step", False),
]


@pytest.mark.parametrize("name,step,full,reason,retryable", FAILURES, ids=[f[0] for f in FAILURES])
def test_failed_fallback_keeps_visible_text_and_marks_field_truncated(tmp_path, name, step, full, reason, retryable):
    display_rows = [user(step, "Do not touch production, unl", truncated_fields=["content"])]
    display = make(tmp_path, display_rows, full)
    chunks, *_ = parse_session_chunks(display)
    chunk = next(c for c in chunks if c.meta.get("event_type") == "USER_INPUT")
    assert chunk.text == "[message:user] Do not touch production, unl"
    assert chunk.meta["truncated_fields"] == ["content"] and chunk.meta["source"] == "display"
    assert chunk.meta["fallback"]["reason"] == reason and chunk.meta["fallback"]["retryable"] is retryable
    assert "restored_fields" not in chunk.meta


def test_a_later_state_of_the_step_does_not_restore_a_running_record(tmp_path):
    display = make(tmp_path, [generic(2, "partial", status="RUNNING", truncated_fields=["content"])],
                   [generic(2, "partial output that finished\n" * 5, status="DONE")])
    chunks, *_ = parse_session_chunks(display)
    chunk = next(c for c in chunks if c.kind == "tool_result")
    assert chunk.meta["fallback"]["reason"] == "full_step_mismatch" and chunk.meta["pending"] is True
    assert chunk.meta["truncated_fields"] == ["content"]


def test_full_file_changed_after_indexing_fails_the_fallback(tmp_path, monkeypatch):
    display = make(tmp_path, [user(0, "Keep the logs", truncated_fields=["content"])],
                   [user(0, "Keep the logs for BBBB days")])
    full = display.with_name("transcript_full.jsonl")
    build = agy._FullTranscript._build

    def build_then_rewrite(self):
        build(self)
        full.write_bytes(full.read_bytes().replace(b"BBBB", b"CCCC"))  # same size, same inode

    monkeypatch.setattr(agy._FullTranscript, "_build", build_then_rewrite)
    chunks, *_ = parse_session_chunks(display)
    chunk = next(c for c in chunks if c.meta.get("event_type") == "USER_INPUT")
    assert chunk.meta["fallback"]["reason"] == "full_changed" and chunk.meta["fallback"]["retryable"] is True
    assert chunk.meta["truncated_fields"] == ["content"] and "CCCC" not in chunk.text


def test_sibling_must_be_the_real_file_in_the_same_directory(tmp_path):
    outside = tmp_path / "elsewhere" / "transcript_full.jsonl"
    outside.parent.mkdir()
    write(outside, [user(0, "Keep the logs for thirty days")])
    display = make(tmp_path, [user(0, "Keep the logs", truncated_fields=["content"])])
    link = display.with_name("transcript_full.jsonl")
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    chunks, *_ = parse_session_chunks(display)
    chunk = next(c for c in chunks if c.meta.get("event_type") == "USER_INPUT")
    assert chunk.meta["truncated_fields"] == ["content"] and "thirty" not in chunk.text
    assert chunk.meta["fallback"]["reason"] in ("full_missing", "full_unavailable")


def test_truncation_marker_inside_the_full_log_itself_is_not_restorable(tmp_path):
    folder = logs(tmp_path)
    path = write(folder / "transcript_full.jsonl", [user(0, "cut", truncated_fields=["content"])])
    chunks, *_ = parse_session_chunks(path)
    chunk = next(c for c in chunks if c.meta.get("event_type") == "USER_INPUT")
    assert chunk.meta["truncated_fields"] == ["content"] and chunk.meta["fallback"]["retryable"] is False


def test_tool_call_arguments_cut_mid_json_are_tolerated_only_when_flagged(tmp_path):
    cut = planner(1, "", [("run_command", {"CommandLine": "npm test"})], truncated_fields=["tool_calls"])
    cut["tool_calls"][0]["args"]["CommandLine"] = '"npm te'  # not valid JSON any more
    chunks, *_ = parse_session_chunks(make(tmp_path, [user(0, "go"), cut]))
    assert not [c for c in chunks if c.meta.get("parse_error")]
    marker = next(c for c in chunks if c.meta.get("truncated_fields") == ["tool_calls"])
    assert marker.kind == "meta" and marker.meta["role"] == "runtime"
    visible = next(c for c in chunks if c.meta.get("evidence_origin") == "model_proposal")
    assert "npm te" in visible.text
    unflagged = dict(cut)
    del unflagged["truncated_fields"]
    chunks, *_ = parse_session_chunks(make(tmp_path / "other", [user(0, "go"), unflagged]))
    assert [c for c in chunks if c.meta.get("parse_error")]


def test_unrestored_marker_survives_when_every_call_was_cut_away(tmp_path):
    row = planner(1, "", [], truncated_fields=["tool_calls"])
    chunks, *_ = parse_session_chunks(make(tmp_path, [user(0, "go"), row]))
    assert [c for c in chunks if c.meta.get("truncated_fields") == ["tool_calls"]]


# --- fallback sync behavior: retry when the full log appears, version bump ---------------------------

def test_missing_full_log_is_retried_once_when_it_appears(env):
    engine, root = env
    display = make(root, [user(0, "Keep the logs", truncated_fields=["content"])])
    first = sync(engine, display)
    assert first["coverage"]["complete"]
    assert chunks_of(engine)[0].meta["truncated_fields"] == ["content"]
    assert not sync(engine, display)["reindexed"]
    write(display.with_name("transcript_full.jsonl"), [user(0, "Keep the logs for thirty days")])
    healed = sync(engine, display)
    assert healed["reindexed"] and healed["reindex_reason"] == "truncated_fallback_retry"
    chunk = chunks_of(engine)[0]
    assert chunk.text.endswith("thirty days") and chunk.meta["source"] == "full"
    assert not sync(engine, display)["reindexed"]


def test_permanent_fallback_failure_does_not_reparse_on_every_growth(env):
    engine, root = env
    display = make(root, [user(0, "Keep the logs", truncated_fields=["content"])],
                   [user(0, "Keep the logs", truncated_fields=["content"])])
    sync(engine, display)
    append(display.with_name("transcript_full.jsonl"), generic(1))
    assert not sync(engine, display)["reindexed"]


def test_parser_version_bump_reparses_the_session(env, monkeypatch):
    engine, root = env
    display = make(root, [user(0, "Hello"), planner(1, "Hi")])
    monkeypatch.setattr(agy, "PARSER_VERSION", 5)
    assert not sync(engine, display)["reindexed"]
    assert engine.store.cursor_record(SID, "agy")["parser_version"] == 5
    monkeypatch.setattr(agy, "PARSER_VERSION", 6)
    result = sync(engine, display)
    assert result["reindexed"] and result["reindex_reason"] == "parser_updated" and result["parser_version"] == 6


def test_full_index_memory_is_bounded_by_records_not_file_size(tmp_path):
    folder = logs(tmp_path)
    records, size = 2400, 16 * 1024
    filler = "z" * size
    with (folder / "transcript_full.jsonl").open("w") as stream:
        for step in range(records):
            stream.write(json.dumps(planner(step, f"{step}:" + filler, None, display=False)) + "\n")
    display = write(folder / "transcript.jsonl", [
        planner(step, f"{step}:zzzz", None, truncated_fields=["content"]) for step in (0, 1200, records - 1)])
    total = (folder / "transcript_full.jsonl").stat().st_size
    assert total > 35 * 1024 * 1024
    tracemalloc.start()
    chunks, *_ = parse_session_chunks(display)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    restored = [c for c in chunks if c.kind == "turn"]
    assert [c.meta["source"] for c in restored] == ["full"] * 3
    assert restored[-1].text.startswith(f"[message:assistant] {records - 1}:")
    assert peak < total * 0.05, (peak, total)


# --- 2. partial-observation judgement ----------------------------------------------------------------

def test_visible_match_in_still_truncated_tool_calls_is_a_violation(env):
    engine, root = env
    path = make(root, [user(0, "tidy up"), planner(1, "", [("run_command", {"CommandLine": "rm -rf /data/ol"})],
                                                   truncated_fields=["tool_calls"])])
    engine.upsert_contracts([{"id": "no-rm", "provider": "agy", "type": "must_not", "scope": "tool_call",
                              "regex": r"rm -rf"}])
    result = engine.audit_session(SID, provider="agy", path=str(path))
    assert finding(result, "no-rm")["verdict"] == "violation"


def test_no_visible_match_in_truncated_tool_calls_is_unknown_not_clean(env):
    engine, root = env
    engine.upsert_contracts([{"id": "no-rm", "provider": "agy", "type": "must_not", "scope": "tool_call",
                              "regex": r"rm -rf"}])
    cut = make(root, [user(0, "tidy up"), planner(1, "", [("run_command", {"CommandLine": "ls"})],
                                                  truncated_fields=["tool_calls"])])
    result = engine.audit_session(SID, provider="agy", path=str(cut))
    item = finding(result, "no-rm")
    assert item["verdict"] == "unknown" and item["reason"].startswith("truncated_tool_calls")
    assert result["coverage"]["complete"]  # a partial field is not a global coverage gap
    whole = make(root, [user(0, "tidy up"), planner(1, "", [("run_command", {"CommandLine": "ls"})])])
    assert finding(engine.audit_session(SID, provider="agy", path=str(whole)), "no-rm")["verdict"] == "clean"


def test_proposed_action_is_judged_on_its_own_untruncated_arguments(env):
    engine, root = env
    path = make(root, [user(0, "tidy up"), planner(1, "", [("run_command", {"CommandLine": "ls"})],
                                                   truncated_fields=["tool_calls"])])
    engine.upsert_contracts([{"id": "no-rm", "provider": "agy", "type": "must_not", "scope": "tool_call",
                              "regex": r"rm -rf"}])
    result = engine.check_action(SID, {"tool_name": "run_command", "arguments": {"CommandLine": "ls"}},
                                 provider="agy", path=str(path))
    assert finding(result, "no-rm")["verdict"] == "clean"


def test_must_without_evidence_and_truncated_tool_calls_is_unknown_not_unmet(env):
    engine, root = env
    must_tests_first(engine)
    cut = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "echo hi"})],
                                             truncated_fields=["tool_calls"])])
    item = finding(deploy_check(engine, cut), "tests-first")
    assert item["verdict"] == "unknown" and item["reason"].startswith("truncated_tool_calls")
    whole = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "echo hi"})])])
    assert finding(deploy_check(engine, whole), "tests-first")["verdict"] == "violation"


def test_retrospective_trigger_search_cannot_be_clean_while_calls_are_hidden(env):
    engine, root = env
    must_tests_first(engine)
    cut = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "echo hi"})],
                                             truncated_fields=["tool_calls"])])
    item = finding(engine.audit_session(SID, provider="agy", path=str(cut)), "tests-first")
    assert item["verdict"] == "unknown" and item["reason"].startswith("truncated_tool_calls")


def _user_event(engine):
    return next(c.chunk_id for c in chunks_of(engine) if c.meta.get("event_type") == "USER_INPUT")


def test_truncated_user_input_never_extends_an_exception(env):
    engine, root = env
    deploy_staging = {"tool_name": "run_command", "arguments": {"CommandLine": "deploy staging"}}
    outcomes = {}
    for label, extra in (("whole", {}), ("cut", {"truncated_fields": ["content"]})):
        path = make(root, [user(0, "Never deploy, except to stag", **extra), planner(1, "ok")], sid=SID)
        sync(engine, path)
        engine.upsert_contracts([{"id": "no-deploy", "provider": "agy", "type": "must_not", "scope": "tool_call",
                                  "regex": "deploy", "exception_regex": "staging", "source_event_id": _user_event(engine)}])
        outcomes[label] = finding(engine.check_action(SID, deploy_staging, provider="agy", path=str(path)), "no-deploy")
    assert outcomes["whole"]["verdict"] == "suspicious"
    assert outcomes["cut"]["verdict"] == "violation" and outcomes["cut"]["reason"].startswith("truncated_content")


def test_truncated_user_input_never_satisfies_a_user_permission(env):
    engine, root = env
    engine.upsert_contracts([{"id": "approval", "provider": "agy", "type": "must", "scope": "message",
                              "roles": ["user"], "regex": "approved"}])
    whole = make(root, [user(0, "I approved the plan"), planner(1, "ok")])
    assert finding(engine.audit_session(SID, provider="agy", path=str(whole)), "approval")["verdict"] == "clean"
    cut = make(root, [user(0, "I approved the plan, but only if", truncated_fields=["content"]), planner(1, "ok")])
    item = finding(engine.audit_session(SID, provider="agy", path=str(cut)), "approval")
    assert item["verdict"] == "unknown" and item["reason"].startswith("truncated_content")
    missing = make(root, [user(0, "Please look at the", truncated_fields=["content"]), planner(1, "ok")])
    item = finding(engine.audit_session(SID, provider="agy", path=str(missing)), "approval")
    assert item["verdict"] == "unknown" and item["reason"].startswith("truncated_content")


def test_truncated_generic_output_never_establishes_success(env):
    engine, root = env
    must_tests_first(engine)
    path = make(root, [user(0, "Run npm test before deploy."),
                       planner(1, "", [("run_command", {"CommandLine": "npm test"})])])
    engine.record_agy_hook(hook(path, 2, "npm test"), "PreToolUse")
    append(path, generic(2, "tests fini", truncated_fields=["content"]))
    engine.record_agy_hook(hook(path, 2, "npm test"), "PostToolUse")
    item = finding(deploy_check(engine, path), "tests-first")
    assert item["verdict"] == "unknown" and item["reason"].startswith("truncated_content")
    assert any(c.meta.get("success_withheld") == "truncated_content" for c in chunks_of(engine))
    write(path.with_name("transcript_full.jsonl"), [
        user(0, "Run npm test before deploy."),
        planner(1, "", [("run_command", {"CommandLine": "npm test"})], display=False),
        generic(2, "tests finished\n" * 3)])
    healed = deploy_check(engine, path)
    assert finding(healed, "tests-first")["verdict"] == "clean"


def test_truncated_generic_output_does_not_hide_a_failure(env):
    engine, root = env
    must_tests_first(engine)
    path = make(root, [user(0, "Run npm test before deploy."),
                       planner(1, "", [("run_command", {"CommandLine": "npm test"})])])
    engine.record_agy_hook(hook(path, 2, "npm test"), "PreToolUse")
    append(path, generic(2, "boom", code=1, truncated_fields=["content"]))
    engine.record_agy_hook(hook(path, 2, "npm test"), "PostToolUse")
    assert finding(deploy_check(engine, path), "tests-first")["verdict"] == "violation"


def test_thinking_checkpoint_system_and_ephemeral_truncation_do_not_affect_verdicts(env):
    engine, root = env
    path = make(root, [
        user(0, "hello"),
        planner(1, "all good", None, thinking="hmm", truncated_fields=["thinking"]),
        rec(2, "EPHEMERAL_MESSAGE", "SYSTEM_SDK", content="ctx", truncated_fields=["content"]),
        rec(3, "SYSTEM_MESSAGE", "SYSTEM", content="note", truncated_fields=["content"]),
        rec(4, "CHECKPOINT", "SYSTEM", content="summary", truncated_fields=["content"]),
    ])
    engine.upsert_contracts([
        {"id": "no-word", "provider": "agy", "type": "must_not", "scope": "message", "regex": "forbidden-word"},
        {"id": "no-rm", "provider": "agy", "type": "must_not", "scope": "tool_call", "regex": "rm -rf"}])
    info = sync(engine, path)
    assert info["coverage"]["complete"]
    truncation = info["coverage"]["agy_evidence"]["truncation"]
    assert truncation["ignored"] == {"PLANNER_RESPONSE.thinking": 1, "EPHEMERAL_MESSAGE.content": 1,
                                     "SYSTEM_MESSAGE.content": 1, "CHECKPOINT.content": 1}
    assert truncation["unrestored"] == {} and truncation["restored_from_full"] == {}
    result = engine.audit_session(SID, provider="agy", path=str(path))
    assert finding(result, "no-word")["verdict"] == "clean" and finding(result, "no-rm")["verdict"] == "clean"
    assert result["verdict"] == "clean"


def test_coverage_counts_restored_and_unrestored_judged_fields(env):
    engine, root = env
    path = make(root, [user(0, "Keep", truncated_fields=["content"]),
                       planner(1, "cut", [("run_command", {"CommandLine": "ls"})], truncated_fields=["content", "tool_calls"])],
                [user(0, "Keep the logs")])
    info = sync(engine, path)
    truncation = info["coverage"]["agy_evidence"]["truncation"]
    assert truncation["restored_from_full"] == {"USER_INPUT.content": 1}
    assert truncation["unrestored"] == {"PLANNER_RESPONSE.content": 1, "PLANNER_RESPONSE.tool_calls": 1}


# --- 3. status values --------------------------------------------------------------------------------

def test_running_result_is_pending_and_unknown_until_a_final_record_arrives(env):
    engine, root = env
    must_tests_first(engine)
    path = make(root, [user(0, "Run npm test before deploy."),
                       planner(1, "", [("run_command", {"CommandLine": "npm test"})])])
    engine.record_agy_hook(hook(path, 2, "npm test"), "PreToolUse")
    append(path, generic(2, "still going", status="RUNNING"))
    info = sync(engine, path)
    assert info["coverage"]["complete"] and info["coverage"]["unsupported_event_count"] == 0
    running = next(c for c in chunks_of(engine) if c.meta.get("status") == "RUNNING")
    assert running.kind == "tool_result" and running.meta["pending"] is True and running.meta["success"] is None
    item = finding(deploy_check(engine, path), "tests-first")
    assert item["verdict"] == "unknown" and item["reason"].startswith("pending_running")
    append(path, generic(2, "tests finished", code=0))
    engine.record_agy_hook(hook(path, 2, "npm test"), "PostToolUse")
    assert finding(deploy_check(engine, path), "tests-first")["verdict"] == "clean"


def test_error_result_is_never_success_but_the_attempt_remains_a_violation(env):
    engine, root = env
    engine.upsert_contracts([{"id": "no-deploy", "provider": "agy", "type": "must_not", "scope": "tool_call",
                              "regex": "deploy"}])
    path = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "deploy"})])])
    engine.record_agy_hook(hook(path, 2, "deploy"), "PreToolUse")
    append(path, rec(2, "GENERIC", "MODEL", "ERROR", error="deploy blew up", content="Encountered error in step execution"))
    engine.record_agy_hook(hook(path, 2, "deploy", error="deploy blew up"), "PostToolUse")
    result = engine.audit_session(SID, provider="agy", path=str(path))
    assert finding(result, "no-deploy")["verdict"] == "violation"
    outcome = next(c for c in chunks_of(engine) if c.kind == "tool_result")
    assert outcome.meta["success"] is False and "deploy blew up" in outcome.text


def test_invalid_call_is_never_success_and_a_forbidden_attempt_is_a_violation(env):
    engine, root = env
    engine.upsert_contracts([{"id": "no-deploy", "provider": "agy", "type": "must_not", "scope": "tool_call",
                              "regex": "deploy"}])
    path = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "deploy"})]),
                       rec(2, "GENERIC", "MODEL", "INVALID", error="invalid arguments")])
    info = sync(engine, path)
    assert info["coverage"]["complete"] and info["coverage"]["parse_error_count"] == 0
    outcome = next(c for c in chunks_of(engine) if c.kind == "tool_result")
    assert outcome.meta["success"] is False and "invalid arguments" in outcome.text
    assert finding(engine.audit_session(SID, provider="agy", path=str(path)), "no-deploy")["verdict"] == "violation"


def test_attempt_blocked_by_the_pre_tool_hook_is_not_counted_as_a_violation(env):
    engine, root = env
    engine.upsert_contracts([{"id": "no-deploy", "provider": "agy", "type": "must_not", "scope": "tool_call",
                              "regex": "deploy"}])
    path = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "deploy"})])])
    engine.record_agy_hook(hook(path, 2, "deploy"), "PreToolUse")
    append(path, rec(2, "GENERIC", "MODEL", "ERROR", error="tool call denied by pre-tool hook: not allowed",
                     content="Encountered error in step execution"))
    result = engine.audit_session(SID, provider="agy", path=str(path))
    assert finding(result, "no-deploy")["verdict"] == "clean"


def test_checkpoint_error_message_and_media_parse_without_errors_or_unsupported_events(env):
    engine, root = env
    path = make(root, [
        user(0, "go"), planner(1, "working"),
        rec(2, "CHECKPOINT", "SYSTEM", content="Conversation summary so far"),
        rec(3, "ERROR_MESSAGE", "SYSTEM", error="model quota exceeded", content="retrying"),
        rec(4, "ERROR_MESSAGE", "SYSTEM", error="stream closed"),
        generic(5, "screenshot taken", media=[{"mime_type": "image/png", "data": "QUJD" * 100}]),
    ])
    info = sync(engine, path)
    coverage = info["coverage"]
    assert coverage["parse_error_count"] == 0 and coverage["unsupported_event_count"] == 0 and coverage["complete"]
    by_type = {c.meta["event_type"]: c for c in chunks_of(engine) if c.meta.get("event_type") in ("CHECKPOINT", "ERROR_MESSAGE")}
    assert by_type["CHECKPOINT"].kind == "meta" and by_type["CHECKPOINT"].meta["role"] == "runtime"
    assert by_type["CHECKPOINT"].meta["evidence_origin"] == "checkpoint"
    errors = [c for c in chunks_of(engine) if c.meta.get("event_type") == "ERROR_MESSAGE"]
    assert len(errors) == 2 and all(c.meta["role"] == "runtime" for c in errors)
    assert "model quota exceeded" in errors[0].text and "retrying" in errors[0].text
    assert coverage["agy_evidence"]["media_placeholders"] == 1


def test_checkpoint_and_error_message_do_not_enter_default_judgement(env):
    engine, root = env
    path = make(root, [user(0, "go"), planner(1, "working"),
                       rec(2, "CHECKPOINT", "SYSTEM", content="we must never rm -rf anything"),
                       rec(3, "ERROR_MESSAGE", "SYSTEM", error="rm -rf failed in a sandbox")])
    engine.upsert_contracts([{"id": "no-rm", "provider": "agy", "type": "must_not", "scope": "any", "regex": "rm -rf"}])
    result = engine.audit_session(SID, provider="agy", path=str(path))
    assert finding(result, "no-rm")["verdict"] == "clean"


def test_media_payload_is_never_indexed(env):
    engine, root = env
    blob = "QUJDREVG" * 4000
    raw = base64.b64encode(hashlib.sha512(b"seed").digest() * 100).decode()  # a long run of real base64 text
    content = ("Created At: x\nCompleted At: y\n\nThe command exited with code 0.\nOutput:\nsaved data:image/png;base64,"
               + blob + " ok " + raw + "\n")
    path = make(root, [user(0, "go"), planner(1, "ok"),
                       rec(2, "GENERIC", "MODEL", content=content,
                           media=[{"mime_type": "image/png", "data": blob, "name": "shot.png"}])])
    sync(engine, path)
    rows = engine.store._conn.execute("SELECT data FROM session_events").fetchall()
    assert rows and not any("QUJDREVG" in row[0] or raw[:200] in row[0] for row in rows)
    result = next(c for c in chunks_of(engine) if c.kind == "tool_result")
    assert "[media #0 type=image/png" in result.text and "blob omitted" in result.text
    assert result.meta["media_count"] == 1


# --- 4. legacy classification ------------------------------------------------------------------------

def _legacy_session(root):
    return make(root, [
        user(0, "Run npm test before deploy."),
        planner(1, "", [("run_command", {"CommandLine": "npm test"})]), generic(2, "tests finished"),
        planner(3, "", [("run_command", {"CommandLine": "deploy"})]), generic(4, "deployed"),
    ])


def test_results_without_any_receipt_are_legacy_unlinked_not_unsupported(env):
    engine, root = env
    info = sync(engine, _legacy_session(root))
    coverage = info["coverage"]
    assert coverage["complete"] and "unsupported_events" not in coverage["issues"]
    assert coverage["unsupported_event_count"] == 0
    assert coverage["legacy_unlinked_count"] == 2 and coverage["receipts_since"] is None
    assert coverage["result_dependent_complete"] is False
    results = [c for c in chunks_of(engine) if c.kind == "tool_result"]
    assert [c.meta["linkage"] for c in results] == ["legacy_unlinked"] * 2
    assert all(c.meta["call_id"] is None and c.meta["success"] is not False for c in results)


def test_legacy_keeps_call_side_judgement_and_marks_result_dependent_ones(env):
    engine, root = env
    path = _legacy_session(root)
    engine.upsert_contracts([{"id": "no-deploy", "provider": "agy", "type": "must_not", "scope": "tool_call",
                              "regex": "deploy"}])
    must_tests_first(engine)
    audit = engine.audit_session(SID, provider="agy", path=str(path))
    assert finding(audit, "no-deploy")["verdict"] == "violation"
    item = finding(audit, "tests-first")
    assert item["verdict"] == "unknown" and item["reason"].startswith("unsupported_legacy")
    checked = finding(deploy_check(engine, path), "tests-first")
    assert checked["verdict"] == "unknown" and checked["reason"].startswith("unsupported_legacy")
    assert deploy_check(engine, path)["coverage"]["unknown_breakdown"] == {"current_session": 0, "unsupported_legacy": 1}


def test_receipts_since_splits_a_conversation_into_legacy_and_current_defects(env):
    engine, root = env
    path = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "npm test"})]),
                       generic(2, "old run"), planner(3, "", [("run_command", {"CommandLine": "npm test"})])])
    engine.record_agy_hook(hook(path, 4, "npm test"), "PreToolUse")
    append(path, generic(4, "tests finished"))
    engine.record_agy_hook(hook(path, 4, "npm test"), "PostToolUse")
    append(path, planner(5, "", [("run_command", {"CommandLine": "ls"})]), generic(6, "listing"))
    info = sync(engine, path)
    coverage = info["coverage"]
    assert coverage["receipts_since"]["step_index"] == 4 and coverage["receipts_since"]["byte_offset"] > 0
    assert coverage["legacy_unlinked_count"] == 1
    # step 6 follows the first receipt but has none: that is a current-session defect, not legacy.
    assert coverage["unsupported_event_count"] == 1 and "unsupported_events" in coverage["issues"]
    assert not coverage["complete"]
    linkage = {c.meta["step_index"]: c.meta["linkage"] for c in chunks_of(engine) if c.kind == "tool_result"}
    assert linkage == {2: "legacy_unlinked", 4: "host_step_receipt", 6: "native_log_omits_call_id"}


def test_new_conversation_does_not_inherit_another_conversations_legacy(env):
    engine, root = env
    sync(engine, _legacy_session(root))
    other = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "npm test"})])], sid=SID2)
    engine.record_agy_hook(hook(other, 2, "npm test", sid=SID2), "PreToolUse")
    append(other, generic(2, "tests finished"))
    engine.record_agy_hook(hook(other, 2, "npm test", sid=SID2), "PostToolUse")
    info = sync(engine, other, sid=SID2)
    coverage = info["coverage"]
    assert coverage["complete"] and coverage["legacy_unlinked_count"] == 0
    assert coverage["receipts_since"]["step_index"] == 2 and coverage["result_dependent_complete"] is True
    must_tests_first(engine)
    assert finding(deploy_check(engine, other, sid=SID2), "tests-first")["verdict"] == "clean"


def test_legacy_never_shrinks_the_evaluation_window_of_a_must(env):
    engine, root = env
    path = make(root, [
        user(0, "Run npm test before deploy."),
        planner(1, "", [("run_command", {"CommandLine": "npm test"})]), generic(2, "tests finished"),
        planner(3, "", [("run_command", {"CommandLine": "ls"})])])
    engine.record_agy_hook(hook(path, 4, "ls"), "PreToolUse")
    append(path, generic(4, "listing"))
    engine.record_agy_hook(hook(path, 4, "ls"), "PostToolUse")
    must_tests_first(engine)
    # The earlier (unlinked) test run is in the window: unknown, neither clean nor a missing-evidence violation.
    item = finding(deploy_check(engine, path), "tests-first")
    assert item["verdict"] == "unknown" and item["reason"].startswith("unsupported_legacy")
    # A rule that starts after the legacy region does not see it, so the missing run is a plain violation.
    sync(engine, path)
    anchor = next(c.chunk_id for c in chunks_of(engine) if c.meta.get("step_index") == 2 and c.kind == "tool_result")
    must_tests_first(engine, applies_from_event_id=anchor)
    assert finding(deploy_check(engine, path), "tests-first")["verdict"] == "violation"


def test_persisted_evidence_counts_match_a_scan_across_incremental_syncs(env):
    engine, root = env
    path = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "ls"})], truncated_fields=["tool_calls"]),
                       generic(2, "listing"), rec(3, "EPHEMERAL_MESSAGE", "SYSTEM_SDK", content="c", truncated_fields=["content"])])
    sync(engine, path)
    append(path, generic(4, "more", media=[{"mime_type": "image/png", "data": "QUJD" * 50}]),
           user(5, "again", truncated_fields=["content"]))
    info = sync(engine, path)
    assert not info["reindexed"]
    persisted = info["coverage"]["agy_evidence"]
    engine.store._conn.execute("DELETE FROM session_agy_stats")  # force the scan fallback
    assert engine.store.agy_evidence_stats(SID) == persisted
    assert persisted["legacy_unlinked_results"] == 2 and persisted["media_placeholders"] == 1
    assert persisted["truncation"]["unrestored"] == {"PLANNER_RESPONSE.tool_calls": 1, "USER_INPUT.content": 1}
    assert persisted["truncation"]["ignored"] == {"EPHEMERAL_MESSAGE.content": 1}


def test_invalid_result_with_host_receipts_keeps_the_call_and_never_succeeds(env):
    engine, root = env
    engine.upsert_contracts([{"id": "no-deploy", "provider": "agy", "type": "must_not", "scope": "tool_call",
                              "regex": "deploy"}])
    path = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "deploy"})])])
    engine.record_agy_hook(hook(path, 2, "deploy"), "PreToolUse")
    append(path, rec(2, "GENERIC", "MODEL", "INVALID", error="bad arguments"))
    engine.record_agy_hook(hook(path, 2, "deploy", error="bad arguments"), "PostToolUse")
    info = sync(engine, path)
    assert info["coverage"]["complete"] and info["coverage"]["unsupported_event_count"] == 0
    kinds = [(c.kind, c.meta.get("success")) for c in chunks_of(engine) if c.meta.get("step_index") == 2]
    assert kinds == [("tool_call", None), ("tool_result", False)]
    assert finding(engine.audit_session(SID, provider="agy", path=str(path)), "no-deploy")["verdict"] == "violation"


def test_legacy_session_where_the_required_call_was_never_proposed_is_a_real_absence(env):
    engine, root = env
    path = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "ls"})]), generic(2, "listing")])
    must_tests_first(engine)
    item = finding(deploy_check(engine, path), "tests-first")
    # The call side proves no `npm test` was ever attempted, whatever the unlinked results say.
    assert item["verdict"] == "violation" and not item["reason"].startswith("unsupported_legacy")


def test_unrelated_running_result_does_not_mask_a_missing_obligation(env):
    engine, root = env
    must_tests_first(engine)
    path = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "sleep 600"})])])
    engine.record_agy_hook(hook(path, 2, "sleep 600"), "PreToolUse")
    append(path, generic(2, "waiting", status="RUNNING"))
    item = finding(deploy_check(engine, path), "tests-first")
    assert item["verdict"] == "violation"


def test_must_not_on_tool_results_sees_visible_matches_and_stays_unknown_when_output_is_cut(env):
    engine, root = env
    engine.upsert_contracts([{"id": "no-secret", "provider": "agy", "type": "must_not", "scope": "tool_result",
                              "roles": ["tool"], "regex": "hunter2"}])
    cut = make(root, [user(0, "go"), generic(1, "all fine", truncated_fields=["content"])])
    item = finding(engine.audit_session(SID, provider="agy", path=str(cut)), "no-secret")
    assert item["verdict"] == "unknown" and item["reason"].startswith("truncated_content")
    leaked = make(root, [user(0, "go"), generic(1, "password is hunter2", truncated_fields=["content"])])
    assert finding(engine.audit_session(SID, provider="agy", path=str(leaked)), "no-secret")["verdict"] == "violation"
    whole = make(root, [user(0, "go"), generic(1, "all fine")])
    assert finding(engine.audit_session(SID, provider="agy", path=str(whole)), "no-secret")["verdict"] == "clean"


def test_unverifiable_contract_stays_unknown_and_is_not_overridden_by_attempts(env):
    engine, root = env
    engine.upsert_contracts([{"id": "no-deploy", "provider": "agy", "type": "must_not", "scope": "tool_call",
                              "regex": "deploy", "source_event_id": "agy:missing:0:0:abcd"}])
    path = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "deploy"})])])
    item = finding(engine.audit_session(SID, provider="agy", path=str(path)), "no-deploy")
    assert item["verdict"] == "unknown"


def test_unrecognized_tool_result_status_is_unsupported_not_guessed(env):
    engine, root = env
    path = make(root, [user(0, "go"), generic(1, "x", status="TELEPORTED")])
    info = sync(engine, path)
    assert info["coverage"]["unsupported_event_count"] == 1 and not info["coverage"]["complete"]
