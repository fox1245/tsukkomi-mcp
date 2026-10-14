"""AGY parser v7: the IDE's typed result steps and context-loading steps.

Synthetic data only. The IDE (``antigravity-ide`` app data) writes the result of a tool call as a step of its own
type instead of the CLI's ``GENERIC``: ``RUN_COMMAND`` (with a host ``exit_code``), ``VIEW_FILE`` (with a
``truncated_fields`` list), ``CODE_ACTION``, ``MCP_TOOL`` and ``LIST_DIRECTORY``. They follow the evidence rules of
``GENERIC``: no call id exists, so a result is linked to its call only through a host receipt. ``CONVERSATION_HISTORY``
and ``KNOWLEDGE_ARTIFACTS`` are content-free context-loading steps. Any other step type stays unsupported.
"""
import json

import pytest

from self_directing_mcp import agy
from self_directing_mcp.agy import parse_session_chunks

from test_agy_parser_v6 import (  # noqa: F401  (env is a fixture)
    SID, append, chunks_of, deploy_check, env, finding, generic, hook, logs, make, must_tests_first, planner, rec,
    sync, user, write)


# --- synthetic IDE records -----------------------------------------------------------------------

def run_command(step, output="tests finished\n", code=0, status="DONE", **kw):
    fields = {"content": output, **({} if code is None else {"exit_code": code})}
    return rec(step, "RUN_COMMAND", "MODEL", status, **fields, **kw)


def view_file(step, text="def main():\n    pass\n", **kw):
    kw.setdefault("truncated_fields", [])
    return rec(step, "VIEW_FILE", "MODEL", content=text, **kw)


def code_action(step, text="The following changes were made", **kw):
    return rec(step, "CODE_ACTION", "MODEL", content=text, **kw)


def mcp_tool(step, text='{"ok": true}', **kw):
    return rec(step, "MCP_TOOL", "MODEL", content=text, **kw)


def list_dir(step, text="a.py\nb.py\n", **kw):
    return rec(step, "LIST_DIRECTORY", "MODEL", content=text, **kw)


def context_step(step, kind, **kw):
    return rec(step, kind, "SYSTEM", **kw)


def tool_hook(path, step, name, args, error=""):
    return {"conversationId": SID, "transcriptPath": str(path), "workspacePaths": ["/synthetic/project"],
            "stepIdx": step, "toolCall": {"name": name, "args": args}, "error": error}


def run_session(engine, root, result, *, hooks=True, post_error="", call="npm test"):
    """user, planner proposing ``call``, then ``result`` at step 2 (hook receipts around it when ``hooks``)."""
    path = make(root, [user(0, "Run npm test before deploy."), planner(1, "", [("run_command", {"CommandLine": call})])])
    if hooks:
        engine.record_agy_hook(hook(path, 2, call), "PreToolUse")
    append(path, result)
    if hooks:
        engine.record_agy_hook(hook(path, 2, call, error=post_error), "PostToolUse")
    return path


def result_chunk(engine, step=2):
    return next(c for c in chunks_of(engine) if c.kind == "tool_result" and c.meta["step_index"] == step)


# --- 1. a whole IDE conversation -------------------------------------------------------------------

def ide_conversation():
    return [
        user(0, "Tidy the project"),
        context_step(1, "CONVERSATION_HISTORY"), context_step(2, "KNOWLEDGE_ARTIFACTS"),
        planner(3, "Looking", [("view_file", {"AbsolutePath": "/synthetic/project/app.py"})]), view_file(4),
        planner(5, "Running", [("run_command", {"CommandLine": "ls"})]), run_command(6, "a.py\n"),
        planner(7, "Editing", [("replace_file_content", {"TargetFile": "/synthetic/project/app.py"})]), code_action(8),
        planner(9, "Asking", [("some_server_tool", {"q": "x"})]), mcp_tool(10),
        planner(11, "Listing", [("list_dir", {"DirectoryPath": "/synthetic/project"})]), list_dir(12),
        rec(13, "CHECKPOINT", "SYSTEM", content="summary"),
        rec(14, "SYSTEM_MESSAGE", "SYSTEM", content="note"),
        rec(15, "ERROR_MESSAGE", "SYSTEM", error="stream closed"),
    ]


def test_an_ide_conversation_parses_without_unsupported_events(env):
    engine, root = env
    info = sync(engine, make(root, ide_conversation()))
    coverage = info["coverage"]
    assert coverage["parse_error_count"] == 0 and coverage["unsupported_event_count"] == 0 and coverage["complete"]
    results = [c for c in chunks_of(engine) if c.kind == "tool_result"]
    assert [c.meta["event_type"] for c in results] == ["VIEW_FILE", "RUN_COMMAND", "CODE_ACTION", "MCP_TOOL",
                                                       "LIST_DIRECTORY"]
    assert all(c.meta["role"] == "tool" and c.meta["result_type"] == c.meta["event_type"] for c in results)
    assert all(c.text.startswith("[tool_result] ") for c in results)
    # No host receipt exists in this conversation: every result is legacy, a known limitation, never a defect.
    assert [c.meta["linkage"] for c in results] == ["legacy_unlinked"] * 5
    assert all(c.meta["call_id"] is None and c.meta["success"] is not True for c in results)
    assert coverage["legacy_unlinked_count"] == 5 and coverage["result_dependent_complete"] is False
    assert coverage["agy_evidence"]["legacy_unlinked_results"] == 5


def test_the_cli_generic_result_keeps_its_exact_shape(env):
    engine, root = env
    sync(engine, make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "ls"})]), generic(2, "x")]))
    result = result_chunk(engine)
    assert result.meta["event_type"] == "GENERIC" and "result_type" not in result.meta  # CLI chunk ids are unchanged


def test_generic_and_typed_results_mix_in_one_history(env):
    engine, root = env
    path = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "npm test"})])])
    engine.record_agy_hook(hook(path, 2, "npm test"), "PreToolUse")
    append(path, generic(2, "tests finished", code=0))
    engine.record_agy_hook(hook(path, 2, "npm test"), "PostToolUse")
    append(path, planner(3, "", [("run_command", {"CommandLine": "npm test"})]))
    engine.record_agy_hook(hook(path, 4, "npm test"), "PreToolUse")
    append(path, run_command(4, "tests finished", code=0))
    engine.record_agy_hook(hook(path, 4, "npm test"), "PostToolUse")
    info = sync(engine, path)
    assert info["coverage"]["complete"] and info["coverage"]["legacy_unlinked_count"] == 0
    results = [c for c in chunks_of(engine) if c.kind == "tool_result"]
    assert [(c.meta["event_type"], c.meta["linkage"], c.meta["success"]) for c in results] == [
        ("GENERIC", "host_step_receipt", True), ("RUN_COMMAND", "host_step_receipt", True)]


# --- 2. context-loading steps ------------------------------------------------------------------------

@pytest.mark.parametrize("kind,label", [("CONVERSATION_HISTORY", "conversation_history"),
                                        ("KNOWLEDGE_ARTIFACTS", "knowledge_artifacts")])
def test_context_loading_steps_are_runtime_metadata_not_unsupported(env, kind, label):
    engine, root = env
    path = make(root, [user(0, "go"), context_step(1, kind), planner(2, "ok")])
    info = sync(engine, path)
    assert info["coverage"]["complete"] and info["coverage"]["unsupported_event_count"] == 0
    chunk = next(c for c in chunks_of(engine) if c.meta.get("event_type") == kind)
    assert chunk.kind == "meta" and chunk.meta["role"] == "runtime" and chunk.meta["evidence_origin"] == label
    assert chunk.text == f"[{label}]" and not chunk.meta.get("unsupported_event")


def test_context_loading_steps_are_never_user_authority_or_evidence(env):
    engine, root = env
    phrase = "approved-by-the-owner"
    path = make(root, [user(0, "go"),
                       context_step(1, "KNOWLEDGE_ARTIFACTS", content=f"notes say {phrase}"),
                       context_step(2, "CONVERSATION_HISTORY", content=f"earlier: {phrase}"),
                       run_command(3, f"echo {phrase}\n"), view_file(4, f"{phrase}\n"),
                       planner(5, "ok")])
    engine.upsert_contracts([
        {"id": "needs-approval", "provider": "agy", "type": "must", "scope": "any", "roles": ["user"], "regex": phrase},
        {"id": "no-approval-word", "provider": "agy", "type": "must_not", "scope": "any", "roles": ["user"],
         "regex": phrase}])
    result = engine.audit_session(SID, provider="agy", path=str(path))
    assert finding(result, "needs-approval")["verdict"] == "violation"  # nothing a user said satisfies it
    assert finding(result, "no-approval-word")["verdict"] == "clean"   # and nothing a user said violates it
    assert not [c for c in chunks_of(engine) if c.meta.get("role") == "user" and phrase in c.text]


# --- 3. unknown shapes stay visible ------------------------------------------------------------------

UNSUPPORTED = {
    "future_step_type": lambda: rec(1, "FUTURE_TOOL", "MODEL", content="x"),
    "future_typed_result_lookalike": lambda: rec(1, "RUN_COMMAND_V2", "MODEL", content="x", exit_code=0),
    "typed_result_from_the_user": lambda: rec(1, "RUN_COMMAND", "USER_EXPLICIT", content="x", exit_code=0),
    "typed_result_from_the_system": lambda: rec(1, "VIEW_FILE", "SYSTEM", content="x"),
    "context_step_from_the_model": lambda: rec(1, "CONVERSATION_HISTORY", "MODEL"),
    "context_step_from_the_user": lambda: rec(1, "KNOWLEDGE_ARTIFACTS", "USER_EXPLICIT"),
    "unknown_result_status": lambda: rec(1, "RUN_COMMAND", "MODEL", "TELEPORTED", content="x", exit_code=0),
    "unknown_view_status": lambda: rec(1, "VIEW_FILE", "MODEL", "TELEPORTED", content="x", truncated_fields=[]),
}


@pytest.mark.parametrize("name", sorted(UNSUPPORTED))
def test_unknown_step_types_sources_and_statuses_stay_unsupported(env, name):
    engine, root = env
    path = make(root, [user(0, "go"), UNSUPPORTED[name](), planner(2, "ok")])
    info = sync(engine, path)
    assert info["coverage"]["unsupported_event_count"] == 1 and not info["coverage"]["complete"]
    assert "unsupported_events" in info["coverage"]["issues"]
    chunk = next(c for c in chunks_of(engine) if c.meta.get("unsupported_event"))
    assert chunk.text.startswith("[unsupported_agy_event] ") and chunk.meta["role"] == "runtime"
    engine.upsert_contracts([{"id": "no-rm", "provider": "agy", "type": "must_not", "scope": "tool_call", "regex": "rm"}])
    assert engine.audit_session(SID, provider="agy", path=str(path))["verdict"] == "unknown"


@pytest.mark.parametrize("fields", [{"exit_code": True}, {"exit_code": "0"}, {"exit_code": 0.0}, {"exit_code": [0]},
                                    {"content": 5}, {"content": ["x"]}, {"truncated_fields": "content"},
                                    {"truncated_fields": [1]}, {"status": 5}],
                         ids=lambda f: json.dumps(f))
def test_malformed_typed_results_are_parse_errors_not_guessed(env, fields):
    engine, root = env
    row = {**run_command(1), **fields}
    info = sync(engine, make(root, [user(0, "go"), row, planner(2, "ok")]))
    assert info["coverage"]["parse_error_count"] == 1 and not info["coverage"]["complete"]
    assert not [c for c in chunks_of(engine) if c.kind == "tool_result"]


# --- 4. RUN_COMMAND: the host exit code ----------------------------------------------------------------

def test_a_linked_run_command_with_exit_zero_is_success_evidence(env):
    engine, root = env
    must_tests_first(engine)
    path = run_session(engine, root, run_command(2, "tests finished\n", code=0))
    info = sync(engine, path)
    assert info["coverage"]["complete"] and info["coverage"]["unsupported_event_count"] == 0
    result = result_chunk(engine)
    assert result.meta["success"] is True and result.meta["linkage"] == "host_step_receipt"
    assert result.meta["call_id"].startswith("agy-hook:") and result.meta["tool_name"] == "run_command"
    assert result.meta["exit_code"] == 0 and result.meta["result_type"] == "RUN_COMMAND"
    call = next(c for c in chunks_of(engine) if c.kind == "tool_call")
    assert call.meta["call_id"] == result.meta["call_id"] and call.text.startswith("[tool_call:run_command] ")
    item = finding(deploy_check(engine, path), "tests-first")
    assert item["verdict"] == "clean" and "successful" in item["reason"]


@pytest.mark.parametrize("code", [1, 2, 127, -1])
def test_a_linked_run_command_with_a_nonzero_exit_code_is_a_failure(env, code):
    engine, root = env
    must_tests_first(engine)
    path = run_session(engine, root, run_command(2, "boom\n", code=code))
    sync(engine, path)
    assert result_chunk(engine).meta["success"] is False and result_chunk(engine).meta["exit_code"] == code
    assert finding(deploy_check(engine, path), "tests-first")["verdict"] == "violation"


def test_a_failure_recorded_by_the_post_hook_overrides_a_zero_exit_code(env):
    engine, root = env
    path = run_session(engine, root, run_command(2, "ok\n", code=0), post_error="hook reported an error")
    sync(engine, path)
    assert result_chunk(engine).meta["success"] is False


def test_the_exit_code_is_not_read_from_output_prose(env):
    engine, root = env
    path = run_session(engine, root, run_command(2, "The command exited with code 0.\nOutput:\nfine\n", code=3))
    sync(engine, path)
    assert result_chunk(engine).meta["success"] is False  # the host field, not text a command could print


@pytest.mark.parametrize("status", ["ERROR", "INVALID", "CANCELED"])
def test_a_failed_status_is_never_success_even_with_exit_zero(env, status):
    engine, root = env
    must_tests_first(engine)
    path = run_session(engine, root, run_command(2, "x", code=0, status=status), post_error="failed")
    sync(engine, path)
    assert result_chunk(engine).meta["success"] is False
    assert finding(deploy_check(engine, path), "tests-first")["verdict"] == "violation"
    bare = make(root, [user(0, "go"), run_command(1, "x", code=0, status=status)])
    sync(engine, bare)
    unlinked = next(c for c in chunks_of(engine) if c.kind == "tool_result" and c.meta["step_index"] == 1)
    assert unlinked.meta["success"] is False


def test_an_unlinked_run_command_is_never_success(env):
    engine, root = env
    must_tests_first(engine)
    path = make(root, [user(0, "Run npm test before deploy."), planner(1, "", [("run_command", {"CommandLine": "npm test"})]),
                       run_command(2, "tests finished\n", code=0)])
    info = sync(engine, path)
    result = result_chunk(engine)
    assert result.meta["linkage"] == "legacy_unlinked" and result.meta["call_id"] is None
    assert result.meta["success"] is None and result.meta["success_withheld"] == "unlinked"
    assert result.meta["exit_code"] == 0
    assert info["coverage"]["complete"] and info["coverage"]["legacy_unlinked_count"] == 1
    item = finding(deploy_check(engine, path), "tests-first")
    assert item["verdict"] == "unknown" and item["reason"].startswith("unsupported_legacy")
    # A nonzero host exit code is still recorded as a failure; it can only make a verdict worse.
    failing = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "npm test"})]),
                          run_command(2, "boom", code=1)])
    sync(engine, failing)
    assert result_chunk(engine).meta["success"] is False


def test_a_run_command_after_the_first_receipt_without_one_is_a_current_defect(env):
    engine, root = env
    path = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "ls"})]), run_command(2, "old run"),
                       planner(3, "", [("run_command", {"CommandLine": "npm test"})])])
    engine.record_agy_hook(hook(path, 4, "npm test"), "PreToolUse")
    append(path, run_command(4, "tests finished"))
    engine.record_agy_hook(hook(path, 4, "npm test"), "PostToolUse")
    append(path, planner(5, "", [("run_command", {"CommandLine": "ls"})]), run_command(6, "listing"))
    info = sync(engine, path)
    coverage = info["coverage"]
    assert coverage["receipts_since"]["step_index"] == 4 and coverage["legacy_unlinked_count"] == 1
    assert coverage["unsupported_event_count"] == 1 and "unsupported_events" in coverage["issues"]
    linkage = {c.meta["step_index"]: c.meta["linkage"] for c in chunks_of(engine) if c.kind == "tool_result"}
    assert linkage == {2: "legacy_unlinked", 4: "host_step_receipt", 6: "native_log_omits_call_id"}
    assert next(c for c in chunks_of(engine) if c.meta["step_index"] == 6 and c.kind == "tool_result").meta["success"] is None


@pytest.mark.parametrize("code", [None, "missing"])
def test_a_run_command_without_a_host_exit_code_is_unsupported_when_linked(env, code):
    engine, root = env
    must_tests_first(engine)
    row = run_command(2, "tests finished\n", code=None)
    if code == "missing":
        row["exit_code"] = None
    path = run_session(engine, root, row)
    info = sync(engine, path)
    result = result_chunk(engine)
    assert result.meta["success"] is None and result.meta["unsupported_event"] is True
    assert info["coverage"]["unsupported_event_count"] == 1 and not info["coverage"]["complete"]
    assert finding(deploy_check(engine, path), "tests-first")["verdict"] != "clean"


def test_a_receipt_for_another_tool_never_makes_a_run_command_success(env):
    engine, root = env
    path = make(root, [user(0, "go"), planner(1, "", [("view_file", {"AbsolutePath": "/synthetic/a"})])])
    engine.record_agy_hook(tool_hook(path, 2, "view_file", {"AbsolutePath": "/synthetic/a"}), "PreToolUse")
    append(path, run_command(2, "fine", code=0))
    engine.record_agy_hook(tool_hook(path, 2, "view_file", {"AbsolutePath": "/synthetic/a"}), "PostToolUse")
    sync(engine, path)
    result = result_chunk(engine)
    assert result.meta["linkage"] == "host_step_receipt" and result.meta["tool_name"] == "view_file"
    assert result.meta["success"] is None


def test_a_running_run_command_is_pending_then_decided_by_its_final_record(env):
    engine, root = env
    must_tests_first(engine)
    path = make(root, [user(0, "Run npm test before deploy."), planner(1, "", [("run_command", {"CommandLine": "npm test"})])])
    engine.record_agy_hook(hook(path, 2, "npm test"), "PreToolUse")
    append(path, run_command(2, "still going", code=0, status="RUNNING"))
    info = sync(engine, path)
    assert info["coverage"]["complete"] and info["coverage"]["unsupported_event_count"] == 0
    running = result_chunk(engine)
    assert running.meta["pending"] is True and running.meta["success"] is None and running.meta["linkage"] == "pending"
    assert "exit_code" not in running.meta  # a running command has no exit status yet, whatever the field says
    assert info["coverage"]["agy_evidence"]["running_results"] == 1
    item = finding(deploy_check(engine, path), "tests-first")
    assert item["verdict"] == "unknown" and item["reason"].startswith("pending_running")
    append(path, run_command(2, "tests finished", code=0))
    engine.record_agy_hook(hook(path, 2, "npm test"), "PostToolUse")
    assert finding(deploy_check(engine, path), "tests-first")["verdict"] == "clean"


def test_a_running_run_command_without_an_exit_code_is_pending_too(env):
    engine, root = env
    must_tests_first(engine)
    path = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "npm test"})]),
                       run_command(2, "still going", code=None, status="RUNNING")])
    info = sync(engine, path)
    assert info["coverage"]["complete"]
    assert finding(deploy_check(engine, path), "tests-first")["verdict"] == "unknown"
    assert result_chunk(engine).meta["pending"] is True


def test_an_unrelated_running_run_command_does_not_mask_a_missing_obligation(env):
    engine, root = env
    must_tests_first(engine)
    path = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "sleep 600"})])])
    engine.record_agy_hook(hook(path, 2, "sleep 600"), "PreToolUse")
    append(path, run_command(2, "waiting", code=None, status="RUNNING"))
    assert finding(deploy_check(engine, path), "tests-first")["verdict"] == "violation"


def test_a_forbidden_command_attempt_is_a_violation_whatever_the_ide_recorded(env):
    engine, root = env
    engine.upsert_contracts([{"id": "no-deploy", "provider": "agy", "type": "must_not", "scope": "tool_call",
                              "regex": "deploy"}])
    for status in ("ERROR", "INVALID", "DONE"):
        path = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "deploy"})]),
                           run_command(2, "x", code=0 if status == "DONE" else None, status=status)])
        assert finding(engine.audit_session(SID, provider="agy", path=str(path)), "no-deploy")["verdict"] == "violation"


def test_a_prior_result_at_the_step_cannot_be_adopted_by_a_later_receipt(env):
    engine, root = env
    path = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "npm test"})]),
                       run_command(2, "tests finished", code=0)])
    engine.record_agy_hook(hook(path, 2, "npm test"), "PreToolUse")  # the result already exists: not an execution
    engine.record_agy_hook(hook(path, 2, "npm test"), "PostToolUse")
    sync(engine, path)
    result = result_chunk(engine)
    assert result.meta["success"] is None and result.meta["call_id"] is None
    assert result.meta["linkage"] == "native_log_omits_call_id" and result.meta["unsupported_event"] is True


def test_rewriting_a_hash_bound_typed_result_voids_its_receipt(env):
    engine, root = env
    path = run_session(engine, root, run_command(2, "tests finished", code=1))
    first = sync(engine, path)
    assert first["coverage"]["complete"] and result_chunk(engine).meta["success"] is False
    path.write_bytes(path.read_bytes().replace(b'"exit_code": 1', b'"exit_code": 0'))
    sync(engine, path)
    result = result_chunk(engine)
    assert result.meta["linkage"] != "host_step_receipt" and result.meta["success"] is not True


def test_a_cut_run_command_output_never_establishes_success_but_does_not_hide_a_failure(env):
    engine, root = env
    must_tests_first(engine)
    path = run_session(engine, root, run_command(2, "tests fini", code=0, truncated_fields=["content"]))
    sync(engine, path)
    result = result_chunk(engine)
    assert result.meta["success"] is None and result.meta["success_withheld"] == "truncated_content"
    item = finding(deploy_check(engine, path), "tests-first")
    assert item["verdict"] == "unknown" and item["reason"].startswith("truncated_content")
    # The sibling full log restores the output: the exit code then counts.
    write(path.with_name("transcript_full.jsonl"), [
        user(0, "Run npm test before deploy."), planner(1, "", [("run_command", {"CommandLine": "npm test"})], display=False),
        run_command(2, "tests finished\n" * 3, code=0)])
    healed = deploy_check(engine, path)
    assert finding(healed, "tests-first")["verdict"] == "clean"
    assert result_chunk(engine).meta["source"] == "full"
    failing = run_session(engine, root, run_command(2, "boom", code=1, truncated_fields=["content"]))
    sync(engine, failing)
    assert finding(deploy_check(engine, failing), "tests-first")["verdict"] == "violation"


# --- 5. VIEW_FILE and truncated fields ---------------------------------------------------------------

SECRET_RULE = {"id": "no-secret", "provider": "agy", "type": "must_not", "scope": "tool_result", "roles": ["tool"],
               "regex": "hunter2"}


def test_a_truncated_view_file_without_a_full_transcript_is_partial_observation(env):
    engine, root = env
    engine.upsert_contracts([SECRET_RULE])
    cut = make(root, [user(0, "go"), view_file(1, "all fi", truncated_fields=["content"])])
    info = sync(engine, cut)
    assert info["coverage"]["complete"]  # a cut field is a classified limit, not a coverage gap
    chunk = next(c for c in chunks_of(engine) if c.kind == "tool_result")
    assert chunk.meta["truncated_fields"] == ["content"] and chunk.meta["source"] == "display"
    assert chunk.meta["fallback"] == {"reason": "full_missing", "retryable": True, "full_size": None}
    assert info["coverage"]["agy_evidence"]["truncation"]["unrestored"] == {"VIEW_FILE.content": 1}
    item = finding(engine.audit_session(SID, provider="agy", path=str(cut)), "no-secret")
    assert item["verdict"] == "unknown" and item["reason"].startswith("truncated_content")
    leaked = make(root, [user(0, "go"), view_file(1, "password is hunter2", truncated_fields=["content"])])
    assert finding(engine.audit_session(SID, provider="agy", path=str(leaked)), "no-secret")["verdict"] == "violation"
    whole = make(root, [user(0, "go"), view_file(1, "all fine")])
    assert finding(engine.audit_session(SID, provider="agy", path=str(whole)), "no-secret")["verdict"] == "clean"


def test_a_truncated_view_file_is_restored_from_the_same_step_of_the_full_transcript(env):
    engine, root = env
    engine.upsert_contracts([SECRET_RULE])
    full_text = "line\n" * 40 + "password is hunter2\n"
    path = make(root, [user(0, "go"), view_file(1, "line\nline\n", truncated_fields=["content"])],
                [user(0, "go"), view_file(1, full_text)])
    info = sync(engine, path)
    chunk = next(c for c in chunks_of(engine) if c.kind == "tool_result")
    assert chunk.text == "[tool_result] " + full_text and chunk.meta["source"] == "full"
    assert chunk.meta["restored_fields"] == ["content"] and "truncated_fields" not in chunk.meta
    assert chunk.meta["full_ref"]["byte_end"] > chunk.meta["full_ref"]["byte_start"]
    assert info["coverage"]["agy_evidence"]["truncation"]["restored_from_full"] == {"VIEW_FILE.content": 1}
    assert info["coverage"]["agy_evidence"]["truncation"]["unrestored"] == {}
    # The secret sat only in the part the display cut: the restored text shows it.
    assert finding(engine.audit_session(SID, provider="agy", path=str(path)), "no-secret")["verdict"] == "violation"


def test_a_failed_view_file_fallback_keeps_the_visible_text_and_says_why(env):
    engine, root = env
    path = make(root, [user(0, "go"), view_file(1, "line\nline\n", truncated_fields=["content"])],
                [user(0, "go"), run_command(1, "line\nline\nmuch longer than the display shows\n")])
    chunks, *_ = parse_session_chunks(path)
    chunk = next(c for c in chunks if c.kind == "tool_result")
    assert chunk.text == "[tool_result] line\nline\n" and chunk.meta["truncated_fields"] == ["content"]
    assert chunk.meta["fallback"]["reason"] == "full_step_mismatch" and chunk.meta["fallback"]["retryable"] is False


def test_the_full_transcript_itself_cannot_restore_a_truncated_typed_result(tmp_path):
    folder = logs(tmp_path)
    path = write(folder / "transcript_full.jsonl", [view_file(0, "cut", truncated_fields=["content"])])
    chunks, *_ = parse_session_chunks(path)
    chunk = next(c for c in chunks if c.kind == "tool_result")
    assert chunk.meta["truncated_fields"] == ["content"] and chunk.meta["fallback"]["reason"] == "source_is_full"


def test_an_empty_or_absent_truncated_fields_list_is_not_a_truncation(env):
    engine, root = env
    plain = view_file(1, "text")
    absent = {k: v for k, v in view_file(3, "text").items() if k != "truncated_fields"}
    info = sync(engine, make(root, [user(0, "go"), plain, planner(2, "ok"), absent]))
    assert info["coverage"]["complete"]
    assert not [c for c in chunks_of(engine) if "truncated_fields" in c.meta]
    assert info["coverage"]["agy_evidence"]["truncation"] == {"restored_from_full": {}, "unrestored": {}, "ignored": {}}


def test_a_cut_field_that_no_verdict_depends_on_is_only_counted(env):
    engine, root = env
    path = make(root, [user(0, "go"), view_file(1, "x", truncated_fields=["thinking"]),
                       context_step(2, "KNOWLEDGE_ARTIFACTS", truncated_fields=["content"])])
    info = sync(engine, path)
    assert info["coverage"]["complete"]
    assert info["coverage"]["agy_evidence"]["truncation"]["ignored"] == {"VIEW_FILE.thinking": 1,
                                                                        "KNOWLEDGE_ARTIFACTS.content": 1}
    assert info["coverage"]["agy_evidence"]["truncation"]["unrestored"] == {}


def test_a_cut_view_file_does_not_hide_a_missing_run_obligation(env):
    engine, root = env
    must_tests_first(engine)
    path = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "ls"})]),
                       view_file(2, "x", truncated_fields=["content"])])
    # Only results that can carry success matter to a success obligation; the cut file view does not.
    assert finding(deploy_check(engine, path), "tests-first")["verdict"] == "violation"


# --- 6. the other typed results ----------------------------------------------------------------------

TYPED = [
    ("VIEW_FILE", view_file, "view_file", {"AbsolutePath": "/synthetic/project/app.py"}),
    ("CODE_ACTION", code_action, "replace_file_content", {"TargetFile": "/synthetic/project/app.py"}),
    ("MCP_TOOL", mcp_tool, "some_server_tool", {"query": "x"}),
    ("LIST_DIRECTORY", list_dir, "list_dir", {"DirectoryPath": "/synthetic/project"}),
]


def typed_session(engine, root, build, name, args, *, status="DONE", hooks=True, post_error="", **kw):
    path = make(root, [user(0, "go"), planner(1, "", [(name, args)])])
    if hooks:
        engine.record_agy_hook(tool_hook(path, 2, name, args), "PreToolUse")
    append(path, build(2, status=status, **kw))
    if hooks:
        engine.record_agy_hook(tool_hook(path, 2, name, args, post_error), "PostToolUse")
    return path


@pytest.mark.parametrize("kind,build,name,args", TYPED, ids=[t[0] for t in TYPED])
def test_a_linked_typed_result_keeps_its_call_and_is_not_an_objective_proof(env, kind, build, name, args):
    engine, root = env
    path = typed_session(engine, root, build, name, args)
    info = sync(engine, path)
    assert info["coverage"]["complete"] and info["coverage"]["unsupported_event_count"] == 0
    result = result_chunk(engine)
    assert result.meta["event_type"] == kind and result.meta["result_type"] == kind
    assert result.meta["linkage"] == "host_step_receipt" and result.meta["tool_name"] == name
    assert result.meta["call_id"].startswith("agy-hook:")
    assert result.meta["success"] is None  # tool transport success proves nothing about the objective
    call = next(c for c in chunks_of(engine) if c.kind == "tool_call")
    assert call.text.startswith(f"[tool_call:{name}] ") and call.meta["call_id"] == result.meta["call_id"]


@pytest.mark.parametrize("kind,build,name,args", TYPED, ids=[t[0] for t in TYPED])
def test_an_unlinked_typed_result_is_legacy_and_never_success(env, kind, build, name, args):
    engine, root = env
    path = typed_session(engine, root, build, name, args, hooks=False)
    info = sync(engine, path)
    assert info["coverage"]["complete"] and info["coverage"]["legacy_unlinked_count"] == 1
    result = result_chunk(engine)
    assert result.meta["linkage"] == "legacy_unlinked" and result.meta["call_id"] is None
    assert result.meta["success"] is None and not result.meta.get("unsupported_event")


@pytest.mark.parametrize("kind,build,name,args", TYPED, ids=[t[0] for t in TYPED])
def test_a_typed_result_after_the_first_receipt_without_one_is_a_current_defect(env, kind, build, name, args):
    engine, root = env
    path = typed_session(engine, root, code_action, "replace_file_content", {"TargetFile": "/synthetic/x"})
    append(path, planner(3, "", [(name, args)]), build(4))
    info = sync(engine, path)
    assert info["coverage"]["unsupported_event_count"] == 1 and not info["coverage"]["complete"]
    assert result_chunk(engine, 4).meta["linkage"] == "native_log_omits_call_id"


@pytest.mark.parametrize("status", ["ERROR", "INVALID", "CANCELED"])
@pytest.mark.parametrize("kind,build,name,args", TYPED, ids=[t[0] for t in TYPED])
def test_a_failed_typed_result_is_never_success(env, kind, build, name, args, status):
    engine, root = env
    path = typed_session(engine, root, build, name, args, status=status, post_error="failed")
    info = sync(engine, path)
    assert info["coverage"]["complete"]
    assert result_chunk(engine).meta["success"] is False


@pytest.mark.parametrize("kind,build,name,args", TYPED, ids=[t[0] for t in TYPED])
def test_a_running_typed_result_is_pending(env, kind, build, name, args):
    engine, root = env
    path = typed_session(engine, root, build, name, args, status="RUNNING", hooks=False)
    info = sync(engine, path)
    assert info["coverage"]["complete"] and info["coverage"]["agy_evidence"]["running_results"] == 1
    result = result_chunk(engine)
    assert result.meta["pending"] is True and result.meta["linkage"] == "pending" and result.meta["success"] is None


def test_the_hook_denial_of_a_typed_call_is_recorded_as_non_execution(env):
    engine, root = env
    engine.upsert_contracts([{"id": "no-rm", "provider": "agy", "type": "must_not", "scope": "tool_call",
                              "regex": "rm -rf"}])
    path = make(root, [user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "rm -rf /data"})])])
    engine.record_agy_hook(hook(path, 2, "rm -rf /data"), "PreToolUse")
    append(path, rec(2, "RUN_COMMAND", "MODEL", "ERROR", content="", error="tool call denied by pre-tool hook: not allowed"))
    sync(engine, path)
    denial = next(c for c in chunks_of(engine) if c.meta.get("evidence_origin") == "host_rejected_proposal")
    assert denial.text.startswith("[tool_denied] ") and denial.kind == "meta"
    assert finding(engine.audit_session(SID, provider="agy", path=str(path)), "no-rm")["verdict"] == "clean"


# --- 7. audits and judgement -------------------------------------------------------------------------

def test_typed_result_text_is_tool_role_evidence_for_result_rules(env):
    engine, root = env
    engine.upsert_contracts([SECRET_RULE])
    for builder in (lambda: view_file(1, "token hunter2"), lambda: run_command(1, "hunter2\n"),
                    lambda: code_action(1, "hunter2"), lambda: mcp_tool(1, "hunter2"), lambda: list_dir(1, "hunter2")):
        path = make(root, [user(0, "go"), builder()])
        assert finding(engine.audit_session(SID, provider="agy", path=str(path)), "no-secret")["verdict"] == "violation"
        engine.upsert_contracts([{**SECRET_RULE, "roles": ["assistant"]}])
        assert finding(engine.audit_session(SID, provider="agy", path=str(path)), "no-secret")["verdict"] == "clean"
        engine.upsert_contracts([SECRET_RULE])


def test_the_result_type_reaches_audit_events(env):
    engine, root = env
    sync(engine, make(root, ide_conversation()))
    events = engine.store.capture_audit_events(SID, "agy", [("tool_result", ("tool",))]).events
    kinds = [event.meta.get("result_type") for event in events if event.kind == "tool_result"]
    assert kinds == ["VIEW_FILE", "RUN_COMMAND", "CODE_ACTION", "MCP_TOOL", "LIST_DIRECTORY"]


# --- 8. versions, reparsing and the cache --------------------------------------------------------------

def test_a_parser_bump_turns_unsupported_ide_steps_into_supported_ones(env, monkeypatch):
    engine, root = env
    path = make(root, ide_conversation())
    with monkeypatch.context() as old:
        # Parser v6 knew neither the typed result steps nor the context-loading steps.
        old.setattr(agy, "PARSER_VERSION", 6)
        old.setattr(agy, "RESULT_TYPES", frozenset({"GENERIC"}))
        old.setattr(agy, "CONTEXT_TYPES", {})
        first = sync(engine, path)
    assert first["coverage"]["unsupported_event_count"] == 7 and first["parser_version"] == 6
    second = sync(engine, path)
    assert second["reindexed"] and second["reindex_reason"] == "parser_updated" and second["parser_version"] == agy.PARSER_VERSION
    assert second["coverage"]["unsupported_event_count"] == 0 and second["coverage"]["complete"]
    assert not sync(engine, path)["reindexed"]
    persisted = second["coverage"]["agy_evidence"]
    engine.store._conn.execute("DELETE FROM session_agy_stats")  # force the scan fallback
    assert engine.store.agy_evidence_stats(SID) == persisted
