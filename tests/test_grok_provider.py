"""Grok CLI session provider: ``updates.jsonl`` only, streamed deltas joined, calls linked by ``toolCallId``.

The recorded shapes come from ``tests/fixtures/grok`` (a headless Grok CLI 1.0.52 probe, sanitized: placeholder paths
and session ids). Everything else here is synthetic and follows those shapes. ``chat_history.jsonl`` and
``system_prompt.txt`` are never read; a test pins that the resolver refuses them.
"""
from __future__ import annotations

import json
import os
import shutil

import pytest

from self_directing_mcp import grok
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index.ingest import current_parser_version
from grok_support import (
    FIXTURES, GROUP, OTHER, SID, T0, agent, annotation, bash_output, call, completed, hook_execution, rec,
    session_file, thought, update, user, write,
)

PHRASE = "FORBIDDEN-PHRASE"


def parse(path, **kwargs):
    return grok.parse_session_chunks(path, **kwargs)[0]


def dump(chunks):
    return "\n".join(chunk_.model_dump_json() for chunk_ in chunks)


def shape(chunks):
    return [(c.kind, c.meta.get("role"), c.text) for c in chunks]


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def root(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    return sessions


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("SELF_DIRECT_GROK_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER", "GROK_HOME", "OPENROUTER_API_KEY",
                 "SELF_DIRECT_OPENROUTER_API_KEY_FILE", "SELF_DIRECT_LOCAL_ONLY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def engine(tmp_path, root):
    instance = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=tmp_path / "index",
                                         grok_sessions_dir=root, openrouter_api_key_file=None))
    yield instance
    instance.close()


def sync(engine, path, sid=SID):
    return engine.sync_session(session_id=sid, provider="grok", path=str(path), embed=False)


def stored(engine, sid=SID):
    return engine.store.list_chunks(sid, "grok")


def fixture_copy(root, name, sid):
    path = session_file(root, sid)
    path.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / name, path)
    return path


# ---------------------------------------------------------------- the recorded sessions

def test_recorded_session_with_a_denied_read(root):
    chunks = parse(fixture_copy(root, "updates_session2.jsonl", SID))
    kinds = [(c.kind, c.meta.get("role")) for c in chunks]
    # the host's note of the blocked read is the one `_x.ai` record that is indexed; the file shows it before the call
    assert kinds == [("turn", "user"), ("meta", "runtime"), ("turn", "assistant"), ("tool_call", "assistant"),
                     ("meta", "runtime"), ("tool_call", "assistant"), ("tool_result", "tool"),
                     ("tool_call", "assistant"), ("tool_result", "tool"), ("tool_result", "tool"),
                     ("turn", "assistant"), ("meta", "runtime")]
    texts = [c.text for c in chunks]
    assert texts[3] == '[tool_call:Read] {"file_path": "/work/proj/notes.txt", "limit": 5}'
    assert texts[7] == '[tool_call:Bash] {"command": "echo probe-ok"}'
    # no `_x.ai` hook report is indexed; the host's note of the blocked read is, without the reason a hook wrote
    assert "hook_execution" not in dump(chunks)
    (note,) = [c for c in chunks if c.meta.get("evidence_origin") == "host_tool_outcome"]
    assert "secret.env" not in note.model_dump_json() and note.text == "[host_tool_outcome] blocked" and note.meta["role"] == "runtime"
    assert (note.meta["host_event_id"], note.meta["blocked_tool"], note.meta["denied_by"]) == (
        34, "read_file", "project/hook probe")
    # the denied read: the call is an attempt, its refusal a failed result with the host's text, linked by id
    denied = {c.meta["call_id"]: c for c in chunks if c.kind == "tool_result"}
    assert denied["call-0000-1"].meta["success"] is False and denied["call-0000-1"].text.startswith("[tool_result] Hook denied")
    assert denied["call-0000-1"].meta["host_event_id"] == 33 and "host_event_id" not in denied["call-0000-0"].meta
    assert denied["call-0000-0"].meta["success"] is True and denied["call-0000-2"].meta["success"] is True
    assert denied["call-0000-2"].meta["exit_code"] == 0 and denied["call-0000-2"].text == "[tool_result] probe-ok\n"
    # the call and its result share only the id; every event keeps its own record's byte range
    calls = {c.meta["call_id"]: c for c in chunks if c.kind == "tool_call"}
    assert calls["call-0000-2"].byte_end <= denied["call-0000-2"].byte_start
    assert all(c.provider == "grok" and c.session_id == SID for c in chunks)
    assert chunks[-1].meta["stop_reason"] == "end_turn" and chunks[-1].meta["totalTokens"] == 70019


def test_recorded_session_without_hooks_is_the_same_story(root):
    chunks = parse(fixture_copy(root, "updates_session1.jsonl", "01a00000-0000-7000-8000-000000000003"))
    assert [c.kind for c in chunks].count("tool_result") == 3 and all(
        c.meta["success"] is True for c in chunks if c.kind == "tool_result")
    assert not any(c.meta.get("unsupported_event") or c.meta.get("parse_error") for c in chunks)


def test_recorded_cancelled_edit_is_a_failed_result_that_never_ran(root):
    chunks = parse(fixture_copy(root, "updates_session3.jsonl", OTHER))
    result = next(c for c in chunks if c.kind == "tool_result" and c.meta["call_id"] == "call-0000-2")
    assert result.meta["success"] is False and result.meta["tool_not_run"] == "cancelled"
    assert result.text.startswith("[tool_result] User cancelled")
    # the richer start of the edit carried a diff, but only a status-less update: progress, not a result
    assert not any(c.meta.get("diff_paths") for c in chunks)
    edit = next(c for c in chunks if c.kind == "tool_call" and c.meta["call_id"] == "call-0000-2")
    assert edit.text.startswith("[tool_call:Edit] ") and edit.meta["native_tool_name"] == "search_replace"
    assert chunks[-1].meta["stop_reason"] == "cancelled"
    # `list_dir` finished with only a rawOutput: the output text is not invented from structured output
    listing = next(c for c in chunks if c.kind == "tool_result" and c.meta["call_id"] == "call-0000-0")
    assert listing.text == "[tool_result] " and listing.meta["success"] is True


# ---------------------------------------------------------------- streamed deltas

def test_contiguous_deltas_of_one_kind_are_one_event(root):
    path = write(session_file(root), user("hello"), agent("Hel"), agent("lo, "), hook_execution(), agent("world"),
                 completed())
    chunks = parse(path)
    assert shape(chunks) == [("turn", "user", "[message:user] hello"),
                             ("turn", "assistant", "[message:assistant] Hello, world"),
                             ("meta", "runtime", "[turn_completed] stop_reason=end_turn")]
    joined = chunks[1]
    assert joined.meta["chunk_count"] == 3 and joined.line_start == 2 and joined.line_end == 5
    raw = path.read_bytes().splitlines(keepends=True)
    assert joined.byte_start == len(raw[0]) and joined.byte_end == sum(len(line) for line in raw[:5])


def test_a_different_kind_a_tool_call_a_new_prompt_and_the_turn_end_each_flush(root):
    path = write(session_file(root), agent("one"), thought("hmm"), agent("two"), call("c1", "grep", {"pattern": "x"}),
                 agent("three"), agent("four", prompt="p2"), completed("p2"), agent("five", prompt="p3"))
    assert [c.text for c in parse(path) if c.kind == "turn"] == [
        "[message:assistant] one", "[message:assistant] two", "[message:assistant] three", "[message:assistant] four",
        "[message:assistant] five"]


def test_two_user_prompts_are_never_one_message(root):
    chunks = parse(write(session_file(root), user("first", 0), user("second", 1), user("third", 1)))
    assert [c.text for c in chunks] == ["[message:user] first", "[message:user] secondthird"]
    assert [c.meta["prompt_index"] for c in chunks] == [0, 1]


def test_thoughts_are_runtime_placeholders_and_the_reasoning_is_not_indexed(root):
    chunks = parse(write(session_file(root), thought(f"I should {PHRASE} "), thought("secretly")))
    (event,) = chunks
    assert (event.kind, event.text, event.meta["role"], event.meta["evidence_origin"]) == (
        "meta", "[thinking]", "runtime", "thinking")
    assert event.meta["text_length"] == len(f"I should {PHRASE} secretly") and PHRASE not in dump(chunks)


def test_an_empty_assistant_message_is_a_placeholder(root):
    (event,) = parse(write(session_file(root), agent("")))
    assert (event.kind, event.text, event.meta["evidence_origin"]) == ("meta", "[message:assistant]", "empty_message")


def test_non_text_content_is_unsupported_and_flushes_the_run(root):
    path = write(session_file(root), agent("before"), rec({"sessionUpdate": "agent_message_chunk", "content": {
        "type": "image", "data": "B64-SENTINEL", "mimeType": "image/png"}}), agent("after"))
    chunks = parse(path)
    assert [c.text for c in chunks] == ["[message:assistant] before", "[unsupported_content:image] keys=data,mimeType,type",
                                        "[message:assistant] after"]
    assert chunks[1].meta["unsupported_event"] is True and "B64-SENTINEL" not in dump(chunks)


# ---------------------------------------------------------------- tool calls and results

def test_calls_and_results_link_by_tool_call_id_whatever_the_order(root):
    path = write(session_file(root), call("a", "run_terminal_command", {"command": "pytest -q"}),
                 call("b", "read_file", {"target_file": "/x"}),
                 update("b", "completed", "data", raw_output={"type": "ReadFile"}),
                 update("a", "completed", "3 passed", raw_output=bash_output(0, "3 passed\n")))
    chunks = parse(path)
    results = {c.meta["call_id"]: c for c in chunks if c.kind == "tool_result"}
    assert set(results) == {"a", "b"} and all(c.meta["tool_name"] is None for c in results.values())
    assert [c.meta["call_id"] for c in chunks if c.kind == "tool_call"] == ["a", "b"]


@pytest.mark.parametrize("status,raw_output,expected", [
    ("completed", bash_output(0), True),
    ("completed", bash_output(2), False),                                  # a non-zero exit is `completed` for the host
    ("completed", {"type": "Bash", "timed_out": True}, False),
    ("completed", {"type": "Bash", "signal": 9}, False),
    ("completed", {"type": "ReadFile", "FileContent": {"content": "x"}}, True),
    ("completed", None, True),
    ("failed", bash_output(0), False),
    ("failed", None, False),
])
def test_success_follows_the_status_and_the_tools_own_exit_code(root, status, raw_output, expected):
    chunks = parse(write(session_file(root), call("c", "run_terminal_command", {"command": "x"}),
                         update("c", status, "out", raw_output=raw_output)))
    assert chunks[1].meta["success"] is expected and chunks[1].meta["tool_status"] == status


def test_a_call_without_a_result_is_in_flight_or_cancelled_never_success(root, engine):
    engine.upsert_contracts([{"id": "tests-ok", "type": "must", "scope": "tool_call", "regex": "pytest",
                              "requires_success": True, "provider": "grok"}])
    path = write(session_file(root), user("run the tests"), call("c1", "run_terminal_command", {"command": "pytest -q"}))
    sync(engine, path)
    assert engine.audit_session(SID, provider="grok", path=str(path))["verdict"] != "clean"        # in flight
    write(path, update("c1", "completed", "ok", raw_output=bash_output(0)), mode="a")
    assert engine.audit_session(SID, provider="grok", path=str(path))["verdict"] == "clean"
    cancelled = write(session_file(root, OTHER), user("run the tests"),
                      call("c1", "run_terminal_command", {"command": "pytest -q"}),
                      update("c1", "failed", "cancelled", extra={"_meta": {"toolNotRun": "cancelled"}}))
    sync(engine, cancelled, OTHER)
    assert engine.audit_session(OTHER, provider="grok", path=str(cancelled))["verdict"] != "clean"
    failing = write(session_file(root, "01a00000-0000-7000-8000-000000000003"), user("run the tests"),
                    call("c1", "run_terminal_command", {"command": "pytest -q"}),
                    update("c1", "completed", "1 failed", raw_output=bash_output(1, "1 failed\n")))
    sync(engine, failing, "01a00000-0000-7000-8000-000000000003")
    assert engine.audit_session("01a00000-0000-7000-8000-000000000003", provider="grok",
                                path=str(failing))["verdict"] != "clean"


def test_a_backgrounded_command_is_not_evidence_that_the_tests_passed(root, engine):
    """``is_background`` is rendered as Claude's ``run_in_background``, which the success analysis already reads."""
    engine.upsert_contracts([{"id": "tests-ok", "type": "must", "scope": "tool_call", "regex": "pytest",
                              "requires_success": True, "provider": "grok"}])
    path = write(session_file(root), user("run the tests"),
                 call("c1", "run_terminal_command", {"command": "pytest -q", "is_background": True}),
                 update("c1", "completed", "started", raw_output=bash_output(0, "started\n")))
    sync(engine, path)
    audit = engine.audit_session(SID, provider="grok", path=str(path))
    assert audit["verdict"] == "unknown" and "success_unverifiable" in json.dumps(audit["findings"])
    masked = write(session_file(root, OTHER), user("run the tests"),
                   call("c1", "run_terminal_command", {"command": "pytest -q || true"}),
                   update("c1", "completed", "x", raw_output=bash_output(0, "x\n")))
    sync(engine, masked, OTHER)
    assert engine.audit_session(OTHER, provider="grok", path=str(masked))["verdict"] == "unknown"


def test_an_edit_result_names_the_file_and_does_not_repeat_the_diff(root):
    diff = [{"type": "diff", "path": "/work/proj/app.py", "oldText": f"return {PHRASE}", "newText": "return 1"}]
    chunks = parse(write(session_file(root), call("e", "search_replace", {"file_path": "app.py", "old_string": "a",
                                                                       "new_string": "b"}),
                         update("e", "completed", content=diff, raw_output={"type": "SearchReplace"})))
    assert chunks[1].text == "[tool_result] [diff path=/work/proj/app.py]" and chunks[1].meta["diff_paths"] == [
        "/work/proj/app.py"]
    assert PHRASE not in chunks[1].text


def test_a_subagent_tool_is_the_parents_call_only(root):
    chunks = parse(write(session_file(root), call("s", "spawn_subagent", {"description": "look", "prompt": "find"}),
                         update("s", "completed", "found it")))
    assert chunks[0].text == '[tool_call:Task] {"description": "look", "prompt": "find"}'
    assert chunks[0].meta["native_tool_name"] == "spawn_subagent" and chunks[1].meta["success"] is True
    assert {c.meta.get("agent_id") for c in chunks} == {None}


def test_mcp_and_unknown_tools(root):
    chunks = parse(write(session_file(root), call("m", "linear__save_issue", {"title": "t"}),
                         call("u", "todo_write", {"todos": []})))
    assert [c.text for c in chunks] == ['[tool_call:mcp__linear__save_issue] {"title": "t"}',
                                        '[tool_call:todo_write] {"todos": []}']
    assert chunks[0].meta["tool_name"] == "mcp__linear__save_issue" and "native_tool_name" in chunks[0].meta
    assert "native_tool_name" not in chunks[1].meta


def test_the_shell_description_is_prose_kept_out_of_the_call_text(root):
    (event,) = parse(write(session_file(root), call("c", "run_terminal_command", {
        "command": "ls", "description": "without connecting to it with sqlite3"})))
    assert "sqlite3" not in event.text and event.meta["tool_prose"] == {
        "description": "without connecting to it with sqlite3"}


@pytest.mark.parametrize("record", [
    call("", "read_file"), call("c", ""), rec({"sessionUpdate": "tool_call", "toolCallId": "c", "title": "x",
                                                "rawInput": "a string"}),
    rec({"sessionUpdate": "tool_call_update", "status": "completed"}),
])
def test_malformed_tool_records_are_unsupported_not_guessed(root, record):
    (event,) = parse(write(session_file(root), record))
    assert event.kind == "meta" and event.meta["unsupported_event"] is True and event.meta["role"] == "runtime"


# ---------------------------------------------------------------- records that are not evidence, and unknown ones

def test_host_hook_records_never_produce_events_and_never_break_a_run(root):
    path = write(session_file(root), agent("a"), hook_execution(), annotation(f"`read_file` blocked: {PHRASE}"),
                 agent("b"), hook_execution("stop"), completed())
    chunks = parse(path)
    assert [c.text for c in chunks] == ["[message:assistant] ab", "[turn_completed] stop_reason=end_turn"]
    assert PHRASE not in dump(chunks)


def test_progress_updates_without_a_terminal_status_are_skipped(root):
    chunks = parse(write(session_file(root), call("c", "read_file", {"target_file": "/x"}),
                         update("c", None, extra={"kind": "read", "title": "Read `/x`"}),
                         update("c", "in_progress"), update("c", "pending")))
    assert [c.kind for c in chunks] == ["tool_call"]


UNKNOWN = [
    ("plan", rec({"sessionUpdate": "plan", "entries": [{"content": "step"}]})),
    ("mode", rec({"sessionUpdate": "current_mode_update", "currentModeId": "plan"})),
    ("compaction", rec({"sessionUpdate": "compaction", "summary": f"summary {PHRASE}"})),
    ("rewind", rec({"sessionUpdate": "rewind", "to": 3}, method="_x.ai/session/update", prompt=None)),
    ("method", rec({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": PHRASE}},
                   method="x.ai/other")),
    ("status", update("c", "teleported")),
    ("extension_chunk", rec({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": PHRASE}},
                            method="_x.ai/session/update", prompt=None)),
]


@pytest.mark.parametrize("name,record", UNKNOWN, ids=[case[0] for case in UNKNOWN])
def test_unknown_record_kinds_are_counted_and_never_crash(root, engine, name, record):
    path = write(session_file(root), agent("before"), record, agent("after"))
    chunks = parse(path)
    odd = [c for c in chunks if c.meta.get("unsupported_event")]
    assert len(odd) == 1 and odd[0].kind == "meta" and odd[0].meta["role"] == "runtime"
    assert PHRASE not in dump(chunks)                                       # the payload is not indexed
    assert [c.text for c in chunks if c.kind == "turn"] == ["[message:assistant] before", "[message:assistant] after"]
    info = sync(engine, path)
    assert info["coverage"]["unsupported_event_count"] == 1 and "unsupported_events" in info["coverage"]["issues"]
    assert info["coverage"]["complete"] is False                            # unknown never becomes clean


MALFORMED = [
    ("not json", "{broken"),
    ("array", "[1, 2]"),
    ("no method", json.dumps({"timestamp": 1, "params": {"update": {"sessionUpdate": "plan"}}})),
    ("no update", json.dumps({"timestamp": 1, "method": "session/update", "params": {}})),
    ("no kind", json.dumps({"timestamp": 1, "method": "session/update", "params": {"update": {}}})),
    ("bad timestamp", json.dumps({**agent("x"), "timestamp": True})),
    ("bad session id", json.dumps({"timestamp": 1, "method": "session/update", "params": {
        "sessionId": 5, "update": {"sessionUpdate": "plan"}}})),
    ("bad meta", json.dumps({"timestamp": 1, "method": "session/update", "params": {
        "_meta": [], "update": {"sessionUpdate": "plan"}}})),
    ("deep", "[" * 200000),
]


@pytest.mark.parametrize("name,line", MALFORMED, ids=[case[0] for case in MALFORMED])
def test_a_malformed_record_is_one_parse_error_and_flushes_the_run(root, name, line):
    path = session_file(root)
    write(path, agent("before"))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    write(path, agent("after"), mode="a")
    chunks = parse(path)
    assert [(c.kind, bool(c.meta.get("parse_error"))) for c in chunks] == [("turn", False), ("meta", True), ("turn", False)]
    assert chunks[1].text.startswith("[parse_error] ")
    assert [c.text for c in chunks if c.kind == "turn"] == ["[message:assistant] before", "[message:assistant] after"]


def test_invalid_utf8_and_blank_lines(root):
    path = write(session_file(root), agent("a"))
    with path.open("ab") as stream:
        stream.write(b'\n   \n{"x": "\xff\xfe"}\n')
    write(path, agent("b"), mode="a")
    chunks = parse(path)
    assert [c.text for c in chunks if c.kind == "turn"] == ["[message:assistant] a", "[message:assistant] b"]
    assert sum(1 for c in chunks if c.meta.get("parse_error")) == 1


def test_timestamps_are_iso_utc_and_strings_are_kept(root):
    chunks = parse(write(session_file(root), agent("a", ts=0), agent("b", ts="2026-10-01T00:00:00Z"),
                         agent("c", ts=1.5), call("c1", "grep", {}, ts=T0)))
    assert chunks[0].timestamp == "1970-01-01T00:00:00Z"
    assert chunks[-1].timestamp.startswith("2026-")
    chunks = parse(write(session_file(root), agent("b", ts="2026-10-01T00:00:00Z")))
    assert chunks[0].timestamp == "2026-10-01T00:00:00Z"


# ---------------------------------------------------------------- authority

def test_a_typed_prompt_is_a_user_turn_and_only_that(root):
    path = write(session_file(root), user(f"please {PHRASE}", 0),
                 rec({"sessionUpdate": "user_message_chunk", "content": {"type": "text", "text": "no index"}},
                     prompt=None),
                 user("<system-reminder>injected by a hook</system-reminder>", 1),
                 user("  \n<System-Reminder source='hook'>x</System-Reminder>", 2),
                 user("", 3))
    chunks = parse(path)
    assert [(c.kind, c.meta["role"], c.meta.get("evidence_origin")) for c in chunks] == [
        ("turn", "user", None), ("meta", "runtime", "user_message_unattributed"),
        ("meta", "runtime", "injected_system_reminder"), ("meta", "runtime", "injected_system_reminder"),
        ("meta", "runtime", "empty_message")]
    assert chunks[0].text == f"[message:user] please {PHRASE}" and chunks[0].meta["prompt_index"] == 0


def test_a_record_from_another_session_is_inherited_history_never_authority(root):
    path = write(session_file(root), user("mine", 0), user(f"from the parent {PHRASE}", 1, sid=OTHER),
                 call("c", "grep", {"pattern": "x"}, sid=OTHER))
    chunks = parse(path)
    assert chunks[0].meta["role"] == "user" and not chunks[0].meta.get("inherited")
    assert (chunks[1].kind, chunks[1].meta["role"], chunks[1].meta["evidence_origin"]) == (
        "meta", "runtime", "inherited_history")
    assert chunks[1].meta["inherited"] is True and chunks[1].meta["record_session_id"] == OTHER
    assert chunks[2].kind == "tool_call" and chunks[2].meta["inherited"] is True       # still evidence of what ran


def test_only_a_user_message_can_anchor_a_contract(root, engine):
    path = write(session_file(root), user("never run rm -rf", 0), agent("ok"),
                 rec({"sessionUpdate": "user_message_chunk", "content": {"type": "text", "text": "hook text"}}, prompt=None))
    sync(engine, path)
    turns = {c.text: c.chunk_id for c in stored(engine)}
    rule = {"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf", "provider": "grok"}
    assert engine.upsert_contracts([{**rule, "source_event_id": turns["[message:user] never run rm -rf"]}])["ok"]
    with pytest.raises(ValueError, match="not a user message"):
        engine.upsert_contracts([{**rule, "id": "x1", "source_event_id": turns["[message:assistant] ok"]}])
    unattributed = next(c.chunk_id for c in stored(engine) if c.text.startswith("[user_message_unattributed]"))
    with pytest.raises(ValueError, match="not a user message"):
        engine.upsert_contracts([{**rule, "id": "x2", "source_event_id": unattributed}])


# ---------------------------------------------------------------- unicode, large lines, determinism

def test_unicode_survives_in_messages_calls_and_results(root):
    path = write(session_file(root), user("테스트를 실행하고 rm -rf 는 쓰지 마 \U0001F600", 0),
                 agent("알겠습니다 → 계속합니다"),
                 call("c", "run_terminal_command", {"command": "echo '한글 é'"}),
                 update("c", "completed", "한글 é\n", raw_output=bash_output(0)))
    texts = [c.text for c in parse(path)]
    assert texts[0] == "[message:user] 테스트를 실행하고 rm -rf 는 쓰지 마 \U0001F600"
    assert texts[2] == '[tool_call:Bash] {"command": "echo \'한글 é\'"}' and texts[3] == "[tool_result] 한글 é\n"


def test_very_long_lines_and_very_long_runs_parse_whole(root):
    big = "x" * (3 * 1024 * 1024)
    path = write(session_file(root), call("c", "run_terminal_command", {"command": "cat big"}),
                 update("c", "completed", big, raw_output=bash_output(0, "y\n")),
                 *[agent(f"{i:05d}") for i in range(3000)], completed())
    chunks = parse(path)
    assert chunks[1].text == "[tool_result] " + big
    joined = next(c for c in chunks if c.kind == "turn")
    assert joined.meta["chunk_count"] == 3000 and joined.text == "[message:assistant] " + "".join(
        f"{i:05d}" for i in range(3000))


def test_parsing_is_deterministic_and_never_reads_anything_but_updates_jsonl(root):
    path = fixture_copy(root, "updates_session2.jsonl", SID)
    sibling = path.with_name("chat_history.jsonl")
    sibling.write_text("SYSTEM-PROMPT-SENTINEL\n", encoding="utf-8")
    first, second = parse(path), parse(path)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second] and "SYSTEM-PROMPT-SENTINEL" not in dump(first)
    with pytest.raises(ValueError, match="official Grok transcript location"):
        grok.resolve_session_path(root, path=str(sibling))


# ---------------------------------------------------------------- resume appends, truncation, the engine

def test_appending_at_a_record_boundary_is_the_same_as_parsing_it_whole(root):
    records = [user("go", 0), agent("thinking out loud"), call("c", "grep", {"pattern": "x"}),
               update("c", "completed", "2 matches"), agent("done"), completed(), user("again", 1), agent("sure")]
    path = write(session_file(root), *records)
    whole = parse(path)
    # every cut that falls between two events (not inside a streamed run) reproduces the whole parse exactly
    lines = path.read_bytes().splitlines(keepends=True)
    cut_path = session_file(root, SID, "%2Fcopy")
    for cut in (1, 3, 5, 6, 7):
        cut_path.parent.mkdir(parents=True, exist_ok=True)
        cut_path.write_bytes(b"".join(lines[:cut]))
        first, offset, _sid = grok.parse_session_chunks(cut_path, session_id=SID)
        cut_path.write_bytes(b"".join(lines))
        second, end, _sid = grok.parse_session_chunks(cut_path, session_id=SID, start_byte=offset)
        assert end == len(b"".join(lines))
        assert [(c.kind, c.text, c.byte_start, c.byte_end) for c in first + second] == [
            (c.kind, c.text, c.byte_start, c.byte_end) for c in whole], cut


def test_a_cut_inside_a_streamed_run_splits_it_where_the_cut_fell(root):
    path = write(session_file(root), agent("Hel"), agent("lo"), agent("!"))
    lines = path.read_bytes().splitlines(keepends=True)
    cut_path = session_file(root, SID, "%2Fcopy")
    cut_path.parent.mkdir(parents=True)
    cut_path.write_bytes(b"".join(lines[:2]))
    first, offset, _sid = grok.parse_session_chunks(cut_path, session_id=SID)
    cut_path.write_bytes(b"".join(lines))
    second, _end, _sid = grok.parse_session_chunks(cut_path, session_id=SID, start_byte=offset)
    assert [c.text for c in first + second] == ["[message:assistant] Hello", "[message:assistant] !"]
    assert offset == len(lines[0]) + len(lines[1])               # the cursor is the end of the last record read


def test_an_unfinished_last_line_is_left_unread(root):
    path = write(session_file(root), agent("a"))
    with path.open("ab") as stream:
        stream.write(json.dumps(agent("half")).encode()[:20])
    chunks, offset, _sid = grok.parse_session_chunks(path)
    assert [c.text for c in chunks] == ["[message:assistant] a"] and offset == len(json.dumps(agent("a"))) + 1


def test_the_engine_indexes_appends_and_reindexes_when_the_file_shrinks(root, engine):
    records = [user("go", 0), call("c", "run_terminal_command", {"command": "pytest -q"}),
               update("c", "completed", "ok", raw_output=bash_output(0)), agent("done"), completed()]
    path = write(session_file(root), *records[:3])
    first = sync(engine, path)
    assert first["ok"] and first["new_chunks"] == 3 and first["coverage"]["complete"] is True
    write(path, *records[3:], mode="a")
    second = sync(engine, path)
    assert second["new_chunks"] == 2 and second["total_chunks"] == 5 and not second.get("reindexed")
    unchanged = sync(engine, path)
    assert unchanged["new_chunks"] == 0 and unchanged["total_chunks"] == 5
    shrunk = write(path, records[0])                                       # the file was rewritten shorter
    again = sync(engine, shrunk)
    assert again["reindexed"] is True and again["reindex_reason"] == "source_changed" and again["total_chunks"] == 1
    assert [c.text for c in stored(engine)] == ["[message:user] go"]


def test_parser_upgrade_reindexes(root, engine, monkeypatch):
    path = write(session_file(root), user("go", 0), agent("x"))
    sync(engine, path)
    assert current_parser_version("grok") == grok.PARSER_VERSION == 2
    monkeypatch.setattr(grok, "PARSER_VERSION", 3)
    assert current_parser_version("grok") == 3
    info = sync(engine, path)
    assert info["reindexed"] is True and info["reindex_reason"] == "parser_updated"


def test_contracts_written_for_claude_names_match_grok_history(root, engine):
    path = write(session_file(root), user("go", 0),
                 call("c", "run_terminal_command", {"command": "rm -rf build", "description": "clean"}),
                 call("r", "read_file", {"target_file": "/work/proj/secret.env"}))
    engine.upsert_contracts([
        {"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": r"\[tool_call:Bash\].*rm\s+-rf"},
        {"id": "no-env", "type": "must_not", "scope": "tool_call", "regex": r"\[tool_call:Read\].*\.env"},
        {"id": "no-native", "type": "must_not", "scope": "tool_call", "regex": "run_terminal_command|read_file|target_file"},
    ])
    sync(engine, path)
    audit = engine.audit_session(SID, provider="grok", path=str(path))
    verdicts = {f["contract_id"]: f["verdict"] for f in audit["findings"]}
    assert verdicts == {"no-rm": "violation", "no-env": "violation", "no-native": "clean"}


def test_discovery_and_resolution(root, tmp_path):
    a = write(session_file(root, SID, GROUP), user("a", 0))
    b = write(session_file(root, OTHER, "%2Fother"), user("b", 0))
    write(root / GROUP / "not-a-session" / "updates.jsonl", user("c", 0))
    write(root / GROUP / SID / "chat_history.jsonl", user("secret", 0))
    assert grok.discover_session_files(root) == sorted([a.resolve(), b.resolve()])
    assert grok.resolve_session_path(root, session_id=SID) == a.resolve()
    assert grok.resolve_session_path(root, session_id=SID, path=str(a)) == a.resolve()
    assert grok.peek_session_id(a) == SID and grok.peek_session_id(a.with_name("chat_history.jsonl")) is None
    for bad in (root / "updates.jsonl", root / GROUP / "updates.jsonl", root / GROUP / "x" / "updates.jsonl",
                root / GROUP / SID / "sub" / "updates.jsonl"):
        with pytest.raises(ValueError):
            grok.resolve_session_path(root, path=str(bad), require_exists=False)
    with pytest.raises(grok.PathTraversalError):
        grok.resolve_session_path(root, path=str(tmp_path / "elsewhere" / GROUP / SID / "updates.jsonl"))
    with pytest.raises(grok.PathTraversalError):
        grok.resolve_session_path(root, path=str(root / ".." / "x" / GROUP / SID / "updates.jsonl"))
    with pytest.raises(ValueError, match="invalid Grok session id"):
        grok.resolve_session_path(root, session_id="../etc")
    with pytest.raises(FileNotFoundError):
        grok.resolve_session_path(root, session_id="01a00000-0000-7000-8000-0000000000ff")


def test_a_session_id_found_twice_is_ambiguous_not_guessed(root):
    write(session_file(root, SID, GROUP), user("a", 0))
    write(session_file(root, SID, "%2Fcopy"), user("a", 0))
    with pytest.raises(FileNotFoundError, match="ambiguous"):
        grok.resolve_session_path(root, session_id=SID)


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privileges on Windows")
def test_symlinks_below_the_root_are_refused_and_not_followed(root, tmp_path):
    real = tmp_path / "real"
    write(real / SID / "updates.jsonl", user("a", 0))
    (root / GROUP).symlink_to(real, target_is_directory=True)
    assert grok.discover_session_files(root) == []
    with pytest.raises(grok.PathTraversalError, match="symlinks"):
        grok.resolve_session_path(root, path=str(root / GROUP / SID / "updates.jsonl"))
    with pytest.raises(FileNotFoundError):
        grok.resolve_session_path(root, session_id=SID)


def test_the_engine_resolves_the_provider_from_the_layout_and_refuses_other_files(root, engine):
    path = write(session_file(root), user("go", 0))
    info = engine.sync_session(path=str(path), embed=False)                    # no provider: auto-detected from the layout
    assert info["ok"] and info["provider"] == "grok" and info["session_id"] == SID
    sibling = write(root / GROUP / SID / "chat_history.jsonl", user("secret", 0))
    refused = engine.sync_session(path=str(sibling), provider="grok", embed=False)
    assert refused["ok"] is False and refused["error"] == "bad_request"
    outside = engine.sync_session(path=str(root.parent / "x" / GROUP / SID / "updates.jsonl"), provider="grok", embed=False)
    assert outside["ok"] is False and outside["error"] == "path_traversal"
    assert engine.sync_session(session_id="nonsense", provider="grok", embed=False)["error"] == "bad_request"
    assert engine.sync_session(session_id="01a00000-0000-7000-8000-0000000000ff", provider="grok",
                               embed=False)["error"] == "not_found"


# ---------------------------------------------------------------- settings

def test_the_sessions_dir_setting(tmp_path, monkeypatch):
    from self_directing_mcp import settings_core
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    assert settings_core.default_grok_sessions_dir() == tmp_path / "home" / ".grok" / "sessions"
    monkeypatch.setenv("GROK_HOME", str(tmp_path / "gh"))
    assert settings_core.default_grok_sessions_dir() == tmp_path / "gh" / "sessions"
    monkeypatch.setenv("SELF_DIRECT_GROK_SESSIONS_DIR", str(tmp_path / "explicit"))
    assert settings_core.default_grok_sessions_dir() == tmp_path / "explicit"
    assert Settings(_env_file=None).resolve_grok_sessions_dir() == tmp_path / "explicit"
    assert settings_core.load_hook_settings().resolve_grok_sessions_dir() == tmp_path / "explicit"
    monkeypatch.delenv("SELF_DIRECT_GROK_SESSIONS_DIR")
    config = tmp_path / "config.toml"
    config.write_text('[paths]\ngrok_sessions_dir = "relative/sessions"\n', encoding="utf-8")
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(config))
    assert Settings(_env_file=None).resolve_grok_sessions_dir() == (tmp_path / "relative" / "sessions").resolve()
    assert settings_core.load_hook_settings().grok_sessions_dir == (tmp_path / "relative" / "sessions").resolve()
    assert "grok" in settings_core.SESSION_PROVIDERS
    monkeypatch.setenv("SELF_DIRECT_SESSION_PROVIDER", "grok")
    assert Settings(_env_file=None).resolve_session_provider() == "grok"
