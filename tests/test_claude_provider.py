"""Claude Code session provider: authority comes from observed origin, shapes are strict, sidechains stay apart.

All fixtures are synthetic. Payload sentinels prove that opaque or runtime-only content is never indexed.
"""
from __future__ import annotations

import itertools
import json
import tracemalloc
from pathlib import Path

import pytest

from self_directing_mcp import claude
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index.ingest import current_parser_version

SID = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"
OTHER = "bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb"
AGENT_ID = "a1b2c3d4e5f60718"
STEM = f"agent-{AGENT_ID}"
CHILD = f"{SID}_{STEM}"
PROJECT = "-synthetic-project"
PHRASE = "FORBIDDEN-PHRASE"

_counter = itertools.count(1)


# ---------------------------------------------------------------- synthetic record builders

def env(kind, *, sid=SID, sub=False, **extra):
    n = next(_counter)
    record = {"type": kind, "uuid": f"00000000-0000-4000-8000-{n:012d}", "parentUuid": None, "sessionId": sid,
              "timestamp": f"2026-10-01T00:{(n // 60) % 60:02d}:{n % 60:02d}.000Z", "cwd": "/synthetic/project",
              "isSidechain": sub, "userType": "external", "version": "9.9.9"}
    if sub:
        record["agentId"] = AGENT_ID
    record.update(extra)
    return record


def user(content, *, sub=False, **extra):
    return env("user", sub=sub, message={"role": "user", "content": content}, **extra)


def human(text, *, sub=False, **extra):
    return user(text, sub=sub, origin={"kind": "human"}, promptId="prompt-1", **extra)


def assistant(*blocks, sub=False, **extra):
    message = {"id": "msg_synthetic_1", "role": "assistant", "model": "claude-synthetic", "content": list(blocks),
               "stop_reason": "end_turn", "stop_sequence": None}
    return env("assistant", sub=sub, message=message, requestId="req_synthetic_1", **extra)


def text(value):
    return {"type": "text", "text": value}


def use(call_id, name, **arguments):
    return {"type": "tool_use", "id": call_id, "name": name, "input": arguments}


def tool_result(call_id, content, *, sub=False, block_extra=None, **extra):
    block = {"type": "tool_result", "tool_use_id": call_id, "content": content, **(block_extra or {})}
    return user([block], sub=sub, sourceToolAssistantUUID="00000000-0000-4000-8000-0000000000aa", **extra)


def attachment(kind, *, sub=False, sid=SID, **fields):
    return env("attachment", sub=sub, sid=sid, attachment={"type": kind, **fields})


def write(path, *records, mode="w", raw=b""):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    if raw:
        with path.open("ab") as stream:
            stream.write(raw)
    return path


def main_file(root, sid=SID, project=PROJECT):
    return root / project / f"{sid}.jsonl"


def sub_file(root, sid=SID, stem=STEM, project=PROJECT):
    return root / project / sid / "subagents" / f"{stem}.jsonl"


def parse(path, **kwargs):
    return claude.parse_session_chunks(path, **kwargs)[0]


def dump(chunks):
    return "\n".join(chunk.model_dump_json() for chunk in chunks)


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def root(tmp_path):
    projects = tmp_path / "projects"
    projects.mkdir()
    return projects


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("SELF_DIRECT_CLAUDE_PROJECTS_DIR", "SELF_DIRECT_SESSION_PROVIDER", "OPENROUTER_API_KEY",
                 "SELF_DIRECT_OPENROUTER_API_KEY_FILE", "SELF_DIRECT_LOCAL_ONLY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def engine(tmp_path, root):
    instance = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=tmp_path / "index",
                                         claude_projects_dir=root, openrouter_api_key_file=None))
    yield instance
    instance.close()


def sync(engine, path, sid=SID):
    return engine.sync_session(session_id=sid, provider="claude", path=str(path), embed=False)


def stored(engine, sid=SID):
    return engine.store.list_chunks(sid, "claude")


def must_not_user(engine, phrase=PHRASE):
    engine.upsert_contracts([{"id": "user-ban", "type": "must_not", "scope": "message", "regex": phrase,
                              "roles": ["user"], "provider": "claude"}])


# ---------------------------------------------------------------- user authority

def test_human_origin_prompt_is_a_user_turn(root):
    path = write(main_file(root), human(f"please do {PHRASE}"))
    (chunk,) = parse(path)
    assert (chunk.kind, chunk.meta["role"], chunk.provider, chunk.session_id) == ("turn", "user", "claude", SID)
    assert chunk.text == f"[message:user] please do {PHRASE}"
    assert chunk.meta["prompt_id"] == "prompt-1" and "evidence_origin" not in chunk.meta
    assert chunk.timestamp.startswith("2026-10-01T")


def test_human_text_block_list_is_one_user_turn_per_block_and_images_are_placeholders(root):
    path = write(main_file(root), human([text("first"), {"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": "B64-IMAGE-SENTINEL"}}, text("second")]))
    chunks = parse(path)
    assert [c.text for c in chunks] == ["[message:user] first", "[image:image/png]", "[message:user] second"]
    assert [(c.kind, c.meta["role"]) for c in chunks] == [("turn", "user"), ("meta", "runtime"), ("turn", "user")]
    assert chunks[1].meta["evidence_origin"] == "image"
    assert "B64-IMAGE-SENTINEL" not in dump(chunks)


def test_legacy_prompt_without_origin_in_main_file_is_inferred_user(root):
    path = write(main_file(root), user("old style prompt"), user([text("old style list prompt")]))
    chunks = parse(path)
    assert [(c.kind, c.meta["role"], c.meta["attribution"]) for c in chunks] == [
        ("turn", "user", "inferred_legacy")] * 2


def _peer():
    return user(f"peer says {PHRASE}", isMeta=True, origin={
        "kind": "peer", "body": "b", "from": "agent-x", "handback": False, "name": "reviewer", "senderTaskId": "t1"})


NON_USER = [
    ("peer", False, _peer, "peer_message"),
    ("task_notification", False, lambda: user(f"<task-notification>{PHRASE}</task-notification>", origin={
        "kind": "task-notification", "producer": "bash", "runId": "run-1"}), "task_notification"),
    ("coordinator", True, lambda: user(f"coordinator says {PHRASE}", sub=True, origin={"kind": "coordinator"}),
     "coordinator"),
    ("sub_task_notification", True, lambda: user(f"notice {PHRASE}", sub=True, origin={
        "kind": "task-notification", "producer": "bash", "runId": "run-2"}), "task_notification"),
    ("is_meta", False, lambda: user(f"injected {PHRASE}", isMeta=True), "injected_meta"),
    ("compact_summary", False, lambda: user(f"summary {PHRASE}", isCompactSummary=True), "compaction"),
    ("subagent_parent_task", True, lambda: user(f"parent asks {PHRASE}", sub=True), "parent_task"),
    ("subagent_parent_task_blocks", True, lambda: user([text(f"parent asks {PHRASE}")], sub=True), "parent_task"),
    ("transcript_only", False, lambda: user(f"display {PHRASE}", origin={"kind": "human"},
                                           isVisibleInTranscriptOnly=True), "transcript_only"),
]


@pytest.mark.parametrize("name,sub,build,origin", NON_USER, ids=[case[0] for case in NON_USER])
def test_text_not_from_the_human_is_never_user_authority(root, name, sub, build, origin):
    path = write(sub_file(root) if sub else main_file(root), build())
    chunks = parse(path)
    assert len(chunks) == 1
    (chunk,) = chunks
    assert (chunk.kind, chunk.meta["role"], chunk.meta["evidence_origin"]) == ("meta", "runtime", origin)
    assert not chunk.meta.get("unsupported_event") and not chunk.meta.get("parse_error")
    assert chunk.meta["role"] != "user" and chunk.kind != "turn"


def test_compaction_summary_is_a_non_authoritative_context_projection(root):
    path = write(main_file(root), user("earlier discussion", isCompactSummary=True))
    (chunk,) = parse(path)
    assert chunk.meta["authoritative"] is False and chunk.meta["evidence_origin"] == "compaction"
    assert chunk.text.startswith("[compact_summary]")


def test_contradictory_flags_with_human_origin_never_grant_authority(root):
    path = write(main_file(root),
                 user("meta but human", isMeta=True, origin={"kind": "human"}),
                 user("compact but human", isCompactSummary=True, origin={"kind": "human"}))
    chunks = parse(path)
    assert len(chunks) == 2
    assert all(c.kind == "meta" and c.meta["role"] == "runtime" and c.meta["unsupported_event"] for c in chunks)


@pytest.mark.parametrize("origin", [
    {"kind": "robot"}, {"kind": 7}, {}, "human", ["human"],
    {"kind": "human", "extra": "x"}, {"kind": "peer", "mystery": 1}, {"kind": "task-notification", "producer": 3},
])
def test_unknown_or_malformed_origin_stays_unsupported_and_unindexed(root, origin):
    path = write(main_file(root), user("ORIGIN-PAYLOAD-SENTINEL", origin=origin))
    (chunk,) = parse(path)
    assert chunk.meta["unsupported_event"] is True and chunk.kind == "meta" and chunk.meta["role"] == "runtime"
    assert "ORIGIN-PAYLOAD-SENTINEL" not in chunk.text


@pytest.mark.parametrize("flags", [{"isMeta": "yes"}, {"isCompactSummary": 1}, {"isVisibleInTranscriptOnly": "x"}])
def test_malformed_flags_degrade_coverage_instead_of_guessing(root, flags):
    path = write(main_file(root), human("hi", **flags))
    (chunk,) = parse(path)
    assert chunk.meta.get("parse_error") is True or chunk.meta.get("unsupported_event") is True
    assert chunk.meta.get("role") != "user"


def test_text_sibling_of_a_tool_result_is_never_user_text(root):
    path = write(main_file(root), user([
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
        text(f"<system-reminder>{PHRASE}</system-reminder>")]))
    chunks = parse(path)
    assert [c.kind for c in chunks] == ["tool_result", "meta"]
    assert chunks[1].meta["role"] == "runtime" and chunks[1].meta["evidence_origin"] == "tool_message_text"
    assert not chunks[1].meta.get("unsupported_event")


def test_human_queued_command_is_a_user_turn_and_others_are_runtime(root):
    path = write(
        main_file(root),
        attachment("queued_command", prompt=f"queued {PHRASE}", commandMode="prompt", origin={"kind": "human"},
                   delivery_id="d1", source_uuid="s1", runId="r1"),
        attachment("queued_command", prompt="note", commandMode="task-notification",
                   origin={"kind": "task-notification", "producer": "bash", "runId": "r2"}, delivery_id="d2"),
        attachment("queued_command", prompt=f"peer {PHRASE}", commandMode="prompt",
                   origin={"kind": "peer", "from": "agent-x", "name": "n", "body": "b", "handback": True,
                           "senderTaskId": "t"}),
        attachment("queued_command", prompt=f"no origin {PHRASE}", commandMode="prompt"),
    )
    chunks = parse(path)
    assert [(c.kind, c.meta["role"]) for c in chunks] == [
        ("turn", "user"), ("meta", "runtime"), ("meta", "runtime"), ("meta", "runtime")]
    assert chunks[0].text == f"[message:user] queued {PHRASE}" and chunks[0].meta["delivery_id"] == "d1"
    assert [c.meta["evidence_origin"] for c in chunks[1:]] == ["task_notification", "peer_message", "queued_command"]
    assert not any(c.meta.get("unsupported_event") for c in chunks)


@pytest.mark.parametrize("fields", [
    {"prompt": "x", "commandMode": "task-notification", "origin": {"kind": "human"}},   # human origin, wrong mode
    {"prompt": "x", "origin": {"kind": "human"}},                                       # human origin, no mode
    {"prompt": "x", "commandMode": "future-mode"},
    {"prompt": 5, "commandMode": "prompt", "origin": {"kind": "human"}},
    {"commandMode": "prompt", "origin": {"kind": "human"}},
])
def test_malformed_queued_command_is_unsupported_never_a_user_turn(root, fields):
    path = write(main_file(root), attachment("queued_command", **fields))
    (chunk,) = parse(path)
    assert chunk.meta["unsupported_event"] is True and chunk.meta["role"] == "runtime" and chunk.kind == "meta"


QUEUED_BASE = {"delivery_id": "d1", "source_uuid": "s1", "timestamp": "2026-10-01T00:00:00.000Z"}
PEER = {"kind": "peer", "body": "b", "from": "agent-x", "handback": False, "name": "reviewer", "senderTaskId": "t1"}


def queued(**fields):
    return attachment("queued_command", **fields)


# Key sets observed in real transcripts: (record, user turn?, evidence_origin).
QUEUED_VARIANTS = {
    "task_notification": (queued(commandMode="task-notification", origin={"kind": "task-notification",
                                                                        "producer": "bash"},
                                 prompt="done", runId="r1", **QUEUED_BASE), False, "task_notification"),
    "task_notification_with_usage_dict": (queued(commandMode="task-notification", origin={
        "kind": "task-notification", "producer": "bash"}, prompt="done", runId="r1", usage={"tokens": "USAGE-SENTINEL"},
        **QUEUED_BASE), False, "task_notification"),
    "task_notification_with_usage_int": (queued(commandMode="task-notification", origin={
        "kind": "task-notification", "producer": "bash"}, prompt="done", runId="r1", usage=12, **QUEUED_BASE),
        False, "task_notification"),
    "task_notification_with_usage_str": (queued(commandMode="task-notification", origin={
        "kind": "task-notification", "producer": "bash"}, prompt="done", runId="r1", usage="USAGE-SENTINEL",
        **QUEUED_BASE), False, "task_notification"),
    "task_notification_without_run_id": (queued(commandMode="task-notification", origin={
        "kind": "task-notification", "producer": "bash"}, prompt="done", **QUEUED_BASE), False, "task_notification"),
    "human_turn_bool": (queued(commandMode="prompt", origin={"kind": "human"}, prompt=f"mid {PHRASE}", humanTurn=True,
                               **QUEUED_BASE), True, None),
    "human_turn_str": (queued(commandMode="prompt", origin={"kind": "human"}, prompt=f"mid {PHRASE}", humanTurn="3",
                              **QUEUED_BASE), True, None),
    "human_turn_int": (queued(commandMode="prompt", origin={"kind": "human"}, prompt=f"mid {PHRASE}", humanTurn=3,
                              **QUEUED_BASE), True, None),
    "coordinator_reminder": (queued(isMeta=True, origin={"kind": "coordinator"}, prompt=f"remind {PHRASE}",
                                    reminderId="rem-1", source_uuid="s1"), False, "coordinator"),
    "peer_prompt_mode_meta": (queued(commandMode="prompt", isMeta=True, origin=PEER, prompt=f"peer {PHRASE}",
                                     reminderId="rem-2", **QUEUED_BASE), False, "peer_message"),
    "peer_reminder": (queued(isMeta=True, origin=PEER, prompt=f"peer {PHRASE}", reminderId="rem-3",
                             source_uuid="s1"), False, "peer_message"),
    "peer_reminder_without_handback": (queued(isMeta=True, origin={k: v for k, v in PEER.items() if k != "handback"},
                                              prompt=f"peer {PHRASE}", reminderId="rem-4", source_uuid="s1"),
                                       False, "peer_message"),
    "meta_without_origin": (queued(isMeta=True, prompt=f"meta {PHRASE}", reminderId="rem-5", source_uuid="s1"),
                            False, "injected_meta"),
}


@pytest.mark.parametrize("name", sorted(QUEUED_VARIANTS))
def test_observed_queued_command_variants_are_supported_and_only_human_prompts_are_user(root, name):
    record, is_user, origin = QUEUED_VARIANTS[name]
    chunks = parse(write(main_file(root), record))
    assert len(chunks) == 1
    (chunk,) = chunks
    assert not chunk.meta.get("unsupported_event") and not chunk.meta.get("parse_error")
    if is_user:
        assert (chunk.kind, chunk.meta["role"]) == ("turn", "user") and "evidence_origin" not in chunk.meta
        assert chunk.text.startswith("[message:user] mid ")
    else:
        assert (chunk.kind, chunk.meta["role"], chunk.meta["evidence_origin"]) == ("meta", "runtime", origin)
    assert "USAGE-SENTINEL" not in dump(chunks)


def test_queued_command_extra_fields_become_meta_not_text(root):
    chunks = parse(write(main_file(root), QUEUED_VARIANTS["human_turn_str"][0], QUEUED_VARIANTS["peer_reminder"][0]))
    assert chunks[0].meta["human_turn"] == "3" and chunks[0].meta["delivery_id"] == "d1"
    assert chunks[1].meta["reminder_id"] == "rem-3" and chunks[1].meta["is_meta"] is True
    assert chunks[1].meta["peer_from"] == "agent-x" and chunks[1].meta["peer_name"] == "reviewer"


@pytest.mark.parametrize("fields", [
    {"commandMode": "prompt", "origin": {"kind": "human"}, "prompt": "x", "isMeta": True},   # meta is never the human
    {"commandMode": "prompt", "origin": {"kind": "human"}, "prompt": "x", "isMeta": "no"},
    {"commandMode": "prompt", "origin": {"kind": "human"}, "prompt": "x", "humanTurn": [1]},
    {"commandMode": "prompt", "origin": {"kind": "human"}, "prompt": "x", "humanTurn": {"n": 1}},
    {"commandMode": "prompt", "origin": {"kind": "human"}, "prompt": "x", "usage": [1]},
    {"commandMode": "prompt", "origin": {"kind": "human"}, "prompt": "x", "usage": True},
    {"origin": PEER, "prompt": "x", "isMeta": True, "reminderId": ["r"]},
    {"origin": PEER, "prompt": "x", "isMeta": True, "reminderId": True},
    {"origin": PEER, "prompt": "x", "isMeta": True, "mystery": 1},
    {"origin": {"kind": "coordinator", "extra": 1}, "prompt": "x", "isMeta": True},
])
def test_malformed_queued_command_variants_stay_unsupported_and_never_user(root, fields):
    (chunk,) = parse(write(main_file(root), queued(**fields)))
    assert chunk.meta["unsupported_event"] is True and chunk.kind == "meta" and chunk.meta["role"] == "runtime"


def test_attachments_rendered_as_user_role_are_never_user_authority(root):
    path = write(main_file(root),
                 attachment("session_context", context=f"project rules {PHRASE}", renderedRole="user"),
                 env("attachment", attachment={"type": "instructions"}, renderedRole="user",
                     rendered=f"instructions {PHRASE}"),
                 env("attachment", attachment={"type": "session_context", "context": f"more {PHRASE}"},
                     renderedRole=7))
    chunks = parse(path)
    assert all(c.kind == "meta" and c.meta.get("role") != "user" for c in chunks)
    assert all(c.meta["role"] == "runtime" for c in chunks[:2])
    assert [c.meta.get("rendered_role") for c in chunks[:2]] == ["user", "user"]
    assert chunks[2].meta.get("parse_error") is True or chunks[2].meta.get("unsupported_event") is True
    assert PHRASE not in dump(chunks)


# ---------------------------------------------------------------- tools

def test_tool_use_and_result_are_linked_and_success_comes_from_is_error(root):
    path = write(
        main_file(root),
        assistant(use("toolu_ok", "Bash", command="npm test", description="run tests")),
        tool_result("toolu_ok", "all passed", block_extra={"is_error": False},
                    permissionDecision="allow"),
        assistant(use("toolu_bad", "Bash", command="npm test")),
        tool_result("toolu_bad", "Exit code 1\nboom", block_extra={"is_error": True}),
        assistant(use("toolu_text", "Bash", command="make")),
        tool_result("toolu_text", "built\nExit code 3\n"),
        assistant(use("toolu_unknown", "Read", file_path="/synthetic/a.txt")),
        tool_result("toolu_unknown", [text("contents")]),
    )
    chunks = parse(path)
    calls = [c for c in chunks if c.kind == "tool_call"]
    results = [c for c in chunks if c.kind == "tool_result"]
    assert [c.meta["call_id"] for c in calls] == [c.meta["call_id"] for c in results] == [
        "toolu_ok", "toolu_bad", "toolu_text", "toolu_unknown"]
    assert calls[0].text == '[tool_call:Bash] {"command": "npm test", "description": "run tests"}'
    assert calls[0].meta["tool_name"] == "Bash" and calls[0].meta["role"] == "assistant"
    assert calls[0].meta["message_id"] == "msg_synthetic_1" and calls[0].meta["request_id"] == "req_synthetic_1"
    assert [c.meta["success"] for c in results] == [True, False, False, None]
    assert all(c.meta["role"] == "tool" for c in results)
    assert results[0].text == "[tool_result] all passed"
    assert results[0].meta["permission_decision"] == "allow"
    assert results[0].meta["source_tool_assistant_uuid"].startswith("00000000-")
    assert not any(c.meta.get("unsupported_event") for c in chunks)


def test_permission_decision_keeps_only_short_scalars(root):
    path = write(main_file(root), tool_result("t1", "ok", permissionDecision={
        "behavior": "ask", "message": "m" * 200, "rule": {"toolName": "Bash"}, "count": 2, "flag": True}))
    (chunk,) = parse(path)
    assert chunk.meta["permission_decision"] == {"behavior": "ask", "count": 2, "flag": True}


def test_tool_result_content_forms_and_placeholders(root):
    path = write(main_file(root),
                 tool_result("t1", [text("line one"), {"type": "image", "source": {
                     "type": "base64", "media_type": "image/jpeg", "data": "B64-RESULT-SENTINEL"}}, text("line two")]),
                 tool_result("t2", [{"type": "tool_reference", "tool_name": "WebFetch"}]),
                 tool_result("t3", ""),
                 tool_result("t4", [{"type": "future_result_block", "payload": "RESULT-BLOCK-SENTINEL"}]))
    chunks = parse(path)
    assert chunks[0].text == "[tool_result] line one [image:image/jpeg] line two"
    assert chunks[1].text == "[tool_result] [tool_reference:WebFetch]"
    assert chunks[2].text == "[tool_result] " and chunks[2].meta["success"] is None
    assert chunks[3].meta["unsupported_event"] is True
    assert "B64-RESULT-SENTINEL" not in dump(chunks) and "RESULT-BLOCK-SENTINEL" not in dump(chunks)
    assert not any(c.meta.get("unsupported_event") for c in chunks[:3])


@pytest.mark.parametrize("block", [
    {"type": "tool_result", "content": "no id"},
    {"type": "tool_result", "tool_use_id": "", "content": "empty id"},
    {"type": "tool_result", "tool_use_id": "t", "content": 12},
    {"type": "tool_result", "tool_use_id": "t", "content": "x", "is_error": "no"},
    {"type": "tool_result", "tool_use_id": "t", "content": "x", "mystery": 1},
    {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "image", "source": {"type": "base64"}}]},
])
def test_malformed_tool_results_stay_unsupported(root, block):
    path = write(main_file(root), user([block]))
    chunks = parse(path)
    assert any(c.meta.get("unsupported_event") or c.meta.get("parse_error") for c in chunks)


@pytest.mark.parametrize("block", [
    {"type": "tool_use", "id": "t", "name": "Bash", "input": "not-a-dict"},
    {"type": "tool_use", "id": "", "name": "Bash", "input": {}},
    {"type": "tool_use", "id": "t", "name": "", "input": {}},
    {"type": "tool_use", "id": "t", "name": "Bash", "input": {}, "mystery": 1},
])
def test_malformed_tool_use_is_not_a_tool_call(root, block):
    chunks = parse(write(main_file(root), assistant(block)))
    assert not [c for c in chunks if c.kind == "tool_call"]
    assert any(c.meta.get("unsupported_event") or c.meta.get("parse_error") for c in chunks)


def test_assistant_text_thinking_and_error_messages(root):
    path = write(
        main_file(root),
        assistant({"type": "thinking", "thinking": "THINKING-TEXT-SENTINEL", "signature": "SIGNATURE-SENTINEL"},
                  text("visible answer")),
        assistant({"type": "redacted_thinking", "data": "REDACTED-DATA-SENTINEL"}),
        assistant(text("API Error: 529 overloaded"), isApiErrorMessage=True),
        env("assistant", message={"id": "m2", "role": "assistant", "model": "<synthetic>",
                                  "content": [text("No response requested.")]}),
        assistant(),
    )
    chunks = parse(path)
    assert [c.text for c in chunks] == [
        "[thinking]", "[message:assistant] visible answer", "[redacted_thinking]",
        "[api_error] API Error: 529 overloaded", "[synthetic_assistant] No response requested.", "[message:assistant]"]
    assert [(c.kind, c.meta["role"]) for c in chunks] == [
        ("meta", "runtime"), ("turn", "assistant"), ("meta", "runtime"),
        ("meta", "runtime"), ("meta", "runtime"), ("meta", "runtime")]
    assert [c.meta.get("evidence_origin") for c in chunks] == [
        "thinking", None, "thinking", "api_error", "synthetic_assistant", "empty_message"]
    blob = dump(chunks)
    for sentinel in ("THINKING-TEXT-SENTINEL", "SIGNATURE-SENTINEL", "REDACTED-DATA-SENTINEL"):
        assert sentinel not in blob
    assert not any(c.meta.get("unsupported_event") for c in chunks)


def test_tool_use_inside_an_api_error_message_is_not_a_call(root):
    chunks = parse(write(main_file(root), assistant(use("toolu_x", "Bash", command="rm -rf /"),
                                                    isApiErrorMessage=True)))
    assert not [c for c in chunks if c.kind == "tool_call"]
    assert chunks[0].meta["unsupported_event"] is True


# ---------------------------------------------------------------- newer attachment types

def test_file_attachments_index_paths_only(root):
    path = write(main_file(root),
                 attachment("file", filename="/synthetic/a.txt", displayPath="a.txt", content={
                     "type": "text", "file": {"content": "FILE-BODY-SENTINEL"}}),
                 attachment("compact_file_reference", filename="/synthetic/b.txt", displayPath="b.txt"))
    chunks = parse(path)
    assert [c.text for c in chunks] == [
        "[attachment:file] display_path=a.txt filename=/synthetic/a.txt",
        "[attachment:compact_file_reference] display_path=b.txt filename=/synthetic/b.txt"]
    assert all(c.kind == "meta" and c.meta["role"] == "runtime" and not c.meta.get("unsupported_event") for c in chunks)
    assert [c.meta["attachment_type"] for c in chunks] == ["file", "compact_file_reference"]
    assert "FILE-BODY-SENTINEL" not in dump(chunks)


def test_plan_mode_indexes_the_reminder_type_and_flags_only(root):
    path = write(main_file(root),
                 attachment("plan_mode", isSubAgent=False, planExists=True, planFilePath="/synthetic/PLAN-PATH.md",
                            reminderType="full"),
                 attachment("plan_mode", isSubAgent=True, planExists=False, planFilePath="/synthetic/none.md",
                            reminderType="sparse"))
    chunks = parse(path)
    assert [c.text for c in chunks] == [
        "[attachment:plan_mode] reminder_type=full plan_exists=true",
        "[attachment:plan_mode] reminder_type=sparse plan_exists=false"]
    assert all((c.kind, c.meta["role"]) == ("meta", "runtime") and not c.meta.get("unsupported_event") for c in chunks)
    assert chunks[0].meta["plan_file_path"] == "/synthetic/PLAN-PATH.md" and chunks[0].meta["is_sub_agent"] is False
    assert chunks[1].meta["plan_exists"] is False and chunks[1].meta["is_sub_agent"] is True
    assert "PLAN-PATH" not in " ".join(c.text for c in chunks)


def test_informational_system_record_is_a_placeholder(root):
    path = write(main_file(root), env("system", subtype="informational", level="notice",
                                      content=f"host says {PHRASE}", isMeta=False, entrypoint="cli",
                                      gitBranch="main", slug="a-b-c"))
    (chunk,) = parse(path)
    assert chunk.text == "[system:informational]" and (chunk.kind, chunk.meta["role"]) == ("meta", "runtime")
    assert chunk.meta["system_subtype"] == "informational" and chunk.meta["level"] == "notice"
    assert not chunk.meta.get("unsupported_event") and PHRASE not in dump([chunk])


def test_continued_in_links_session_continuations_as_metadata(root):
    path = write(main_file(root), env("continued-in", continuedInSessionId=OTHER))
    (chunk,) = parse(path)
    assert chunk.text == "[claude_metadata:continued-in]" and (chunk.kind, chunk.meta["role"]) == ("meta", "runtime")
    assert chunk.meta["continued_in_session_id"] == OTHER and not chunk.meta.get("unsupported_event")
    for bad in ({}, {"continuedInSessionId": 5}, {"continuedInSessionId": ""}, {"continuedInSessionId": "a b"},
                {"continuedInSessionId": ["x"]}):
        (flagged,) = parse(write(main_file(root), env("continued-in", **bad)))
        assert flagged.meta.get("unsupported_event") is True and "continued_in_session_id" not in flagged.meta


def test_hook_injected_attachments_keep_only_event_and_name(root):
    path = write(main_file(root),
                 attachment("hook_additional_context", content=[f"tool output {PHRASE}"], hookName="UserPromptSubmit",
                            toolUseID="toolu_h1", hookEvent="UserPromptSubmit"),
                 attachment("hook_system_message", content=f"system says {PHRASE}", hookName="PostToolUse:Bash",
                            toolUseID="toolu_h2", hookEvent="PostToolUse"))
    chunks = parse(path)
    assert [c.text for c in chunks] == [
        "[attachment:hook_additional_context] event=UserPromptSubmit name=UserPromptSubmit",
        "[attachment:hook_system_message] event=PostToolUse name=PostToolUse:Bash"]
    assert all((c.kind, c.meta["role"]) == ("meta", "runtime") for c in chunks)
    assert [c.meta["tool_use_id"] for c in chunks] == ["toolu_h1", "toolu_h2"]
    assert PHRASE not in dump(chunks)


def test_thinking_drop_and_remote_session_change_are_metadata_only(root):
    path = write(main_file(root),
                 attachment("thinking_drop", requestId="req_1", thinkingBlocksSent=3, thinkingTurnsSent=2,
                            newlyDropped={"h": "DROP-SENTINEL"}, blockHashes=["DROP-SENTINEL"],
                            firstReportForThreadInProcess=True, clientChange={"c": "DROP-SENTINEL"},
                            model="claude-synthetic"),
                 attachment("remote_session_change", url="https://remote.invalid/DROP-SENTINEL", commit="abc123",
                            pr="DROP-SENTINEL", sendUserFileHint=True, managedCommit=False, managedPr=True))
    chunks = parse(path)
    assert [c.text for c in chunks] == ["[attachment:thinking_drop]", "[attachment:remote_session_change]"]
    assert (chunks[0].meta["thinking_blocks_sent"], chunks[0].meta["thinking_turns_sent"]) == (3, 2)
    assert chunks[1].meta["managed_pr"] is True and chunks[1].meta["managed_commit"] is False
    assert "DROP-SENTINEL" not in dump(chunks) and "abc123" not in dump(chunks)
    assert not any(c.meta.get("unsupported_event") for c in chunks)


def test_read_truncation_notice_records_the_truncated_result_link(root):
    path = write(main_file(root),
                 assistant(use("toolu_read", "Read", file_path="/synthetic/big.txt")),
                 tool_result("toolu_read", "first lines", block_extra={"is_error": False}),
                 attachment("read_truncation_notice", banner="BANNER-SENTINEL", toolUseID="toolu_read"))
    chunks = parse(path)
    notice = chunks[-1]
    assert notice.text == "[attachment:read_truncation_notice] truncated_result_for=toolu_read"
    assert notice.meta["truncated_result_for"] == "toolu_read" and notice.meta["role"] == "runtime"
    assert notice.kind == "meta" and not notice.meta.get("unsupported_event")
    assert "BANNER-SENTINEL" not in dump(chunks)
    # The result itself is untouched at parse time: the link lives on the notice for the audit layer.
    assert chunks[1].meta["call_id"] == "toolu_read" and "truncated_result_for" not in chunks[1].meta


NEW_ATTACHMENT_FIELDS = {
    "file": dict(filename="/s/a.txt", displayPath="a.txt", content={"type": "text", "file": {}}),
    "compact_file_reference": dict(filename="/s/a.txt", displayPath="a.txt"),
    "hook_additional_context": dict(content=["x"], hookName="n", toolUseID="t", hookEvent="UserPromptSubmit"),
    "hook_system_message": dict(content="x", hookName="n", toolUseID="t", hookEvent="PostToolUse"),
    "thinking_drop": dict(requestId="r", thinkingBlocksSent=1, thinkingTurnsSent=1, newlyDropped={}, blockHashes=[],
                          firstReportForThreadInProcess=False, clientChange={}, model="m"),
    "remote_session_change": dict(url=None, commit="c", pr="p", sendUserFileHint=False, managedCommit=False,
                                  managedPr=False),
    "read_truncation_notice": dict(banner="b", toolUseID="t"),
    "plan_mode": dict(isSubAgent=False, planExists=True, planFilePath="/s/plan.md", reminderType="full"),
}
NEW_ATTACHMENT_BREAKS = {
    "file": [dict(filename=3), dict(displayPath=None), dict(content="text"), dict(content={"mystery": 1}),
             dict(mystery=1), "filename"],
    "compact_file_reference": [dict(displayPath=3), dict(mystery=1), "displayPath"],
    "hook_additional_context": [dict(content="str-not-list"), dict(hookName=3), dict(toolUseID=3),
                                dict(hookEvent=None), dict(mystery=1), "hookName"],
    "hook_system_message": [dict(content=["list-not-str"]), dict(hookEvent=3), dict(mystery=1), "toolUseID"],
    "thinking_drop": [dict(thinkingBlocksSent="1"), dict(thinkingTurnsSent=True), dict(newlyDropped=[]),
                      dict(blockHashes={}), dict(firstReportForThreadInProcess=1), dict(clientChange=[]),
                      dict(model=3), dict(querySource=3), dict(mystery=1), "requestId"],
    "remote_session_change": [dict(url=3), dict(commit=None), dict(pr=3), dict(sendUserFileHint="no"),
                              dict(managedCommit=1), dict(managedPr=None), dict(mystery=1), "commit"],
    "read_truncation_notice": [dict(banner=3), dict(toolUseID=""), dict(toolUseID=3), dict(mystery=1), "toolUseID"],
    "plan_mode": [dict(isSubAgent="no"), dict(planExists=1), dict(planFilePath=None), dict(reminderType=3),
                  dict(mystery=1), "reminderType", "planFilePath", "planExists", "isSubAgent"],
}


@pytest.mark.parametrize("kind", sorted(NEW_ATTACHMENT_FIELDS))
def test_new_attachment_types_validate_their_shape_strictly(root, kind):
    valid = NEW_ATTACHMENT_FIELDS[kind]
    (good,) = parse(write(main_file(root), attachment(kind, **valid)))
    assert not good.meta.get("unsupported_event") and not good.meta.get("parse_error")
    for change in NEW_ATTACHMENT_BREAKS[kind]:
        fields = {key: value for key, value in valid.items() if key != change} if isinstance(change, str) \
            else {**valid, **change}
        (bad,) = parse(write(main_file(root), attachment(kind, **fields)))
        assert bad.meta.get("unsupported_event") is True and bad.meta["role"] == "runtime", (kind, change)
        assert bad.kind == "meta"


# ---------------------------------------------------------------- opaque and runtime-only payloads

SENTINELS = {
    "hook_out": "HOOK-STDOUT-SENTINEL", "hook_err": "HOOK-STDERR-SENTINEL", "hook_cmd": "HOOK-COMMAND-SENTINEL",
    "hook_content": "HOOK-CONTENT-SENTINEL", "snapshot": "FILE-SNAPSHOT-SENTINEL", "snippet": "SNIPPET-SENTINEL",
    "last_prompt": "LAST-PROMPT-SENTINEL", "title": "AI-TITLE-SENTINEL", "system": "SYSTEM-CONTENT-SENTINEL",
    "queue": "QUEUE-CONTENT-SENTINEL", "env": "ENV-SNAPSHOT-SENTINEL", "tokens": "TOKENS-REMINDER-SENTINEL",
    "tasks": "TASK-REMINDER-SENTINEL", "tools": "DEFERRED-TOOLS-SENTINEL", "mcp": "MCP-BLOCK-SENTINEL",
    "context": "SESSION-CONTEXT-SENTINEL", "pr": "PR-LINK-SENTINEL", "away": "AWAY-SUMMARY-SENTINEL",
    "local": "LOCAL-COMMAND-SENTINEL", "silent": "SILENT-TURN-SENTINEL", "model": "MODEL-ATTACH-SENTINEL",
    "agent_name": "AGENT-NAME-SENTINEL", "cost": "COST-STATE-SENTINEL", "delta": "FILE-DELTA-SENTINEL",
    "file_content": "FILE-CONTENT-SENTINEL", "hook_ctx": "HOOK-ADDITIONAL-CONTEXT-SENTINEL",
    "hook_sys": "HOOK-SYSTEM-MESSAGE-SENTINEL", "banner": "READ-BANNER-SENTINEL", "block_hash": "BLOCK-HASH-SENTINEL",
    "client_change": "CLIENT-CHANGE-SENTINEL", "remote_url": "REMOTE-URL-SENTINEL",
    "remote_commit": "REMOTE-COMMIT-SENTINEL", "remote_pr": "REMOTE-PR-SENTINEL", "usage": "USAGE-PAYLOAD-SENTINEL",
    "informational": "INFORMATIONAL-HOST-TEXT-SENTINEL", "plan_text": "PLAN-FILE-CONTENT-SENTINEL",
}


def corpus(sub=False):
    """One synthetic record for every supported shape, each carrying a payload sentinel where payloads exist."""
    s = SENTINELS
    base = {"sub": sub}
    records = [
        human("HUMAN-TEXT-POSITIVE-CONTROL", **base) if not sub else user("parent task text", sub=True),
        assistant({"type": "thinking", "thinking": "private", "signature": "sig"}, text("assistant text"),
                  use("toolu_c1", "Bash", command="ls"), **base),
        tool_result("toolu_c1", "listing", block_extra={"is_error": False}, **base),
        attachment("hook_success", hookEvent="PostToolUse", hookName="PostToolUse:Bash", exitCode=0, durationMs=12,
                   command=s["hook_cmd"], content=s["hook_content"], stdout=s["hook_out"], stderr=s["hook_err"],
                   **base),
        attachment("total_tokens_reminder", text=s["tokens"], **base),
        attachment("prompt_snapshot", snapshot=s["snapshot"], **base),
        attachment("deferred_tools_record", entries=[s["tools"]], toolInputCopies=[s["tools"]], **base),
        attachment("deferred_tools_delta", added=[s["tools"]], **base),
        attachment("silent_turn_reminder", text=s["silent"], **base),
        attachment("edited_text_file", filename="/synthetic/edited.txt", snippet=s["snippet"], **base),
        attachment("task_reminder", content=s["tasks"], itemCount=2, **base),
        attachment("mcp_instructions_delta", addedBlocks=[s["mcp"]], addedNames=["srv-a"], removedNames=["srv-b"],
                   **base),
        attachment("model", identity="claude-synthetic", text=s["model"], **base),
        attachment("environment", snapshot=s["env"], **base),
        attachment("session_context", context=s["context"], renderedRole="user", **base),
        attachment("instructions", renderedRole="user", rendered=s["context"], **base),
        attachment("date", date="2026-10-01", **base),
        attachment("credential_org", org="synthetic", **base),
        attachment("agent_listing_delta", added=["x"], **base),
        attachment("auto_mode", enabled=True, **base),
        attachment("skill_listing", skills=[s["mcp"]], **base),
        attachment("task_status", status="running", **base),
        attachment("plan_mode", isSubAgent=sub, planExists=True, planFilePath="/synthetic/plan.md",
                   reminderType="full", **base),
        env("system", subtype="informational", level="notice", content=s["informational"], isMeta=False, sub=sub),
        env("continued-in", continuedInSessionId=OTHER, sub=sub),
        attachment("file", filename="/synthetic/read.txt", displayPath="read.txt", content={
            "type": "text", "file": {"filePath": "/synthetic/read.txt", "content": s["file_content"]}}, **base),
        attachment("compact_file_reference", filename="/synthetic/ref.txt", displayPath="ref.txt", **base),
        attachment("hook_additional_context", content=[s["hook_ctx"]], hookName="UserPromptSubmit",
                   toolUseID="toolu_c1", hookEvent="UserPromptSubmit", **base),
        attachment("hook_system_message", content=s["hook_sys"], hookName="PostToolUse:Bash", toolUseID="toolu_c1",
                   hookEvent="PostToolUse", **base),
        attachment("thinking_drop", requestId="req_1", thinkingBlocksSent=3, thinkingTurnsSent=2,
                   newlyDropped={"a": s["block_hash"]}, blockHashes=[s["block_hash"]],
                   firstReportForThreadInProcess=True, clientChange={"a": s["client_change"]},
                   model="claude-synthetic", querySource="repl", **base),
        attachment("thinking_drop", requestId="req_2", thinkingBlocksSent=0, thinkingTurnsSent=0, newlyDropped={},
                   blockHashes=[], firstReportForThreadInProcess=False, clientChange={}, model="claude-synthetic",
                   **base),
        attachment("remote_session_change", url=s["remote_url"], commit=s["remote_commit"], pr=s["remote_pr"],
                   sendUserFileHint=False, managedCommit=True, managedPr=False, **base),
        attachment("remote_session_change", url=None, commit="", pr="", sendUserFileHint=True, managedCommit=False,
                   managedPr=False, **base),
        attachment("read_truncation_notice", banner=s["banner"], toolUseID="toolu_c1", **base),
        attachment("queued_command", commandMode="task-notification", origin={"kind": "task-notification",
                                                                             "producer": "bash"},
                   prompt="done", runId="r1", usage={"n": s["usage"]}, delivery_id="d", source_uuid="s",
                   timestamp="2026-10-01T00:00:00.000Z", **base),
        attachment("queued_command", isMeta=True, origin={"kind": "coordinator"}, prompt="remind", reminderId="r1",
                   source_uuid="s", **base),
        env("system", subtype="turn_duration", durationMs=1234, level="info", isMeta=False, content=s["system"],
            sub=sub),
        env("system", subtype="stop_hook_summary", level="info", isMeta=False, content=s["system"], sub=sub),
        env("system", subtype="away_summary", level="notice", isMeta=False, content=s["away"], sub=sub),
        env("system", subtype="compact_boundary", level="info", isMeta=False, content=s["system"], sub=sub,
            compactMetadata={"trigger": "auto", "preTokens": 1000}),
        env("system", subtype="local_command", level="suggestion", isMeta=False, content=s["local"], sub=sub),
        env("queue-operation", operation="enqueue", content=s["queue"], commandUuid="c1", sub=sub),
        env("queue-operation", operation="dequeue", deliveryId="d1", sub=sub),
        env("queue-operation", operation="remove", reason="cancelled", sub=sub),
        env("mode", mode="plan", sub=sub),
        env("permission-mode", permissionMode="acceptEdits", sub=sub),
        env("ai-title", aiTitle=s["title"], sub=sub),
        env("agent-name", agentName=s["agent_name"], sub=sub),
        env("pr-link", prUrl=s["pr"], prNumber=7, sub=sub),
        env("last-prompt", lastPrompt=s["last_prompt"], sub=sub),
        env("atis-latch", latched=True, sub=sub),
        env("cost-state", costUsd=1.5, detail=s["cost"], sub=sub),
        env("file-history-snapshot", messageId="m1", snapshot={"trackedFileBackups": {"a": s["snapshot"]}},
            isSnapshotUpdate=False, sub=sub),
        env("file-history-delta", delta={"a": s["delta"]}, sub=sub),
        user(f"meta {PHRASE}", isMeta=True, sub=sub),
    ]
    return records


@pytest.mark.parametrize("sub", [False, True], ids=["main", "subagent"])
def test_payload_sentinels_are_never_indexed_and_every_valid_shape_is_supported(root, sub):
    path = write(sub_file(root) if sub else main_file(root), *corpus(sub))
    chunks = parse(path)
    blob = dump(chunks)
    for name, sentinel in SENTINELS.items():
        assert sentinel not in blob, name
    for chunk in chunks:
        assert not chunk.meta.get("parse_error") and not chunk.meta.get("unsupported_event"), chunk.text
        assert chunk.meta["role"] != "user" or (not sub and chunk.text == "[message:user] HUMAN-TEXT-POSITIVE-CONTROL")
    assert any("assistant text" in c.text for c in chunks) and any("ls" in c.text for c in chunks)
    if not sub:
        assert any(c.text == "[message:user] HUMAN-TEXT-POSITIVE-CONTROL" for c in chunks)
    # Hook runtime records keep only the event, name and exit code.
    (hook,) = [c for c in chunks if c.meta.get("attachment_type") == "hook_success"]
    assert hook.text == "[attachment:hook_success] event=PostToolUse name=PostToolUse:Bash exit=0"
    assert (hook.meta["hook_event"], hook.meta["hook_name"], hook.meta["exit_code"]) == (
        "PostToolUse", "PostToolUse:Bash", 0)
    assert hook.kind == "meta" and hook.meta["role"] == "runtime"
    # Small identifiers that are deliberately kept.
    texts = {c.text for c in chunks}
    assert "[attachment:edited_text_file] /synthetic/edited.txt" in texts
    assert "[attachment:mcp_instructions_delta] added=srv-a removed=srv-b" in texts
    assert "[system:turn_duration]" in texts and "[queue_operation:enqueue]" in texts
    assert "[claude_metadata:last-prompt]" in texts and "[claude_metadata:file-history-snapshot]" in texts
    modes = {c.meta.get("permission_mode") for c in chunks if c.meta.get("event_type") == "permission-mode"}
    assert modes == {"acceptEdits"}


@pytest.mark.parametrize("sub", [False, True], ids=["main", "subagent"])
def test_complete_coverage_for_every_valid_shape_end_to_end(engine, root, sub):
    path = write(sub_file(root) if sub else main_file(root), *corpus(sub))
    result = sync(engine, path, CHILD if sub else SID)
    assert result["ok"], result
    coverage = result["coverage"]
    assert coverage["complete"] is True, coverage
    assert coverage["parse_error_count"] == 0 and coverage["unsupported_event_count"] == 0
    assert coverage["provider"] == "claude" and coverage["parser_version"] == claude.PARSER_VERSION


UNKNOWN_SHAPES = {
    "record_type": lambda: env("future-record", payload="UNKNOWN-SHAPE-SENTINEL"),
    "attachment_type": lambda: attachment("future_attachment", payload="UNKNOWN-SHAPE-SENTINEL"),
    "attachment_not_a_dict": lambda: env("attachment", attachment="UNKNOWN-SHAPE-SENTINEL"),
    "assistant_block": lambda: assistant({"type": "server_tool_use", "id": "s1", "name": "web_search",
                                          "input": {"q": "UNKNOWN-SHAPE-SENTINEL"}}),
    "assistant_block_extra_key": lambda: assistant({"type": "text", "text": "x", "mystery": "UNKNOWN-SHAPE-SENTINEL"}),
    "user_block": lambda: user([{"type": "document", "source": {"data": "UNKNOWN-SHAPE-SENTINEL"}}]),
    "image_without_source": lambda: user([{"type": "image", "data": "UNKNOWN-SHAPE-SENTINEL"}]),
    "image_non_image_media": lambda: user([{"type": "image", "source": {
        "type": "base64", "media_type": "text/plain", "data": "UNKNOWN-SHAPE-SENTINEL"}}]),
    "system_subtype": lambda: env("system", subtype="future_subtype", content="UNKNOWN-SHAPE-SENTINEL"),
    "system_level_type": lambda: env("system", subtype="turn_duration", level=5, content="UNKNOWN-SHAPE-SENTINEL"),
    "queue_operation": lambda: env("queue-operation", operation="teleport", content="UNKNOWN-SHAPE-SENTINEL"),
    "hook_success_types": lambda: attachment("hook_success", hookEvent=5, hookName="n", exitCode="0",
                                             stdout="UNKNOWN-SHAPE-SENTINEL"),
    "message_not_a_dict": lambda: env("assistant", message="UNKNOWN-SHAPE-SENTINEL"),
    "message_wrong_role": lambda: env("user", message={"role": "assistant", "content": "UNKNOWN-SHAPE-SENTINEL"}),
    "content_not_str_or_list": lambda: env("user", message={"role": "user", "content": {"x": "UNKNOWN-SHAPE-SENTINEL"}}),
    "bad_timestamp": lambda: human("UNKNOWN-SHAPE-SENTINEL", timestamp=12345),
}


@pytest.mark.parametrize("name", sorted(UNKNOWN_SHAPES))
def test_unknown_or_malformed_shapes_keep_coverage_incomplete_and_unindexed(engine, root, name):
    path = write(main_file(root), human("a normal prompt"), UNKNOWN_SHAPES[name]())
    result = sync(engine, path)
    assert result["ok"], result
    coverage = result["coverage"]
    assert coverage["complete"] is False
    assert {"unsupported_events", "parse_errors"} & set(coverage["issues"]), coverage
    blob = dump(stored(engine))
    assert "UNKNOWN-SHAPE-SENTINEL" not in blob
    engine.upsert_contracts([{"id": "x", "type": "must_not", "scope": "tool_call", "regex": "never-matches-anything"}])
    verdict = engine.audit_session(SID, provider="claude", path=str(path))
    assert verdict["verdict"] == "unknown", "unknown shapes must never become clean"


def test_unreadable_lines_degrade_coverage(engine, root):
    path = main_file(root)
    write(path, human("fine"))
    with path.open("ab") as stream:
        stream.write(b"not json\n[1, 2]\n\xff\xfe\n")
    chunks = parse(path)
    assert sum(1 for c in chunks if c.meta.get("parse_error")) == 3
    assert sync(engine, path)["coverage"]["complete"] is False


# ---------------------------------------------------------------- session identity and sidechains

def test_session_id_comes_from_layout_and_subagent_ids_embed_the_parent(root):
    assert claude.peek_session_id(main_file(root)) == SID
    assert claude.peek_session_id(sub_file(root)) == CHILD
    assert claude.session_parts(CHILD) == (SID, STEM)
    assert claude.session_parts(SID) == (SID, None)
    for bad in (root / PROJECT / "notes.jsonl", root / PROJECT / SID.upper() / "subagents" / "x.jsonl",
                root / PROJECT / SID / "subagents" / "deeper" / "x.jsonl", root / PROJECT / SID / "x.jsonl",
                root / PROJECT / f"{SID.upper()}.jsonl", root / PROJECT / SID / "subagents" / ".hidden.jsonl",
                root / PROJECT / SID / "subagents" / "bad name.jsonl", root / PROJECT / f"{SID}.json"):
        assert claude.peek_session_id(bad) is None, bad
    assert len(CHILD) <= 128 and claude.valid_id(CHILD) and claude.valid_id(SID)
    assert not claude.valid_id("../x") and not claude.valid_id(SID.upper()) and not claude.valid_id(f"{SID}_")


def test_subagent_file_is_its_own_session_with_parent_metadata(root):
    path = write(sub_file(root), user("parent task", sub=True),
                 assistant(text("child answer"), use("toolu_s1", "Bash", command="ls"), sub=True))
    chunks, offset, sid = claude.parse_session_chunks(path)
    assert sid == CHILD and offset == path.stat().st_size
    assert {c.session_id for c in chunks} == {CHILD}
    assert all(c.meta["parent_session_id"] == SID and c.meta["agent_id"] == AGENT_ID and c.meta["sidechain"] is True
               for c in chunks)
    parent_chunks = parse(write(main_file(root), human("main prompt"), assistant(text("main answer"))))
    assert not ({c.chunk_id for c in chunks} & {c.chunk_id for c in parent_chunks})
    assert all("parent_session_id" not in c.meta and c.meta["sidechain"] is False for c in parent_chunks)


def test_agent_id_falls_back_to_the_file_stem(root):
    record = user("parent task", sub=True)
    del record["agentId"]
    (chunk,) = parse(write(sub_file(root), record))
    assert chunk.meta["agent_id"] == AGENT_ID


def test_parse_rejects_identity_mismatch(root):
    path = write(main_file(root), human("x"))
    with pytest.raises(ValueError):
        claude.parse_session_chunks(path, session_id=OTHER)
    with pytest.raises(ValueError):
        claude.parse_session_chunks(path.parent / "notes.jsonl")
    assert claude.parse_session_chunks(path, session_id=SID)[2] == SID


def test_records_that_claim_another_session_are_errors_not_evidence(root):
    path = write(main_file(root), human("mine"), human("inherited", sid=OTHER),
                 env("mode", mode="plan", sid=OTHER), env("mode", mode="plan"))
    chunks = parse(path)
    assert [c.text for c in chunks][0] == "[message:user] mine"
    assert [bool(c.meta.get("parse_error")) for c in chunks] == [False, True, True, False]
    assert "inherited" not in dump(chunks)
    assert all("inherited" not in c.meta and "record_session_id" not in c.meta for c in chunks)


def test_sidechain_flag_must_agree_with_the_file_layout(root):
    in_sub = parse(write(sub_file(root), user("task", sub=True), env("mode", mode="x", sub=False, isSidechain=False)))
    assert not in_sub[0].meta.get("parse_error") and in_sub[1].meta.get("parse_error") is True
    in_main = parse(write(main_file(root), human("embedded child", isSidechain=True)))
    assert in_main[0].meta["unsupported_event"] is True and in_main[0].meta["role"] == "runtime"
    assert "embedded child" not in dump(in_main)


def test_subagent_file_with_inherited_session_ids_is_accepted_when_the_agent_is_constant(engine, root):
    path = write(
        sub_file(root),
        user("parent task", sub=True),
        user("older task", sub=True, sid=OTHER),
        user("older coordinator", sub=True, sid=OTHER, origin={"kind": "coordinator"}),
        assistant(text("older answer"), use("toolu_old", "Bash", command="ls"), sub=True, sid=OTHER),
        tool_result("toolu_old", "ok", sub=True, sid=OTHER, block_extra={"is_error": False}),
        attachment("hook_success", sub=True, sid=OTHER, hookEvent="Stop", hookName="Stop", exitCode=0),
        env("mode", mode="plan", sub=True, sid=OTHER),
        assistant(text("current answer"), sub=True),
    )
    chunks = parse(path)
    assert not any(c.meta.get("parse_error") or c.meta.get("unsupported_event") for c in chunks)
    assert {c.session_id for c in chunks} == {CHILD}
    inherited = [c for c in chunks if c.meta.get("inherited")]
    assert len(inherited) == 7 and {c.meta["record_session_id"] for c in inherited} == {OTHER}
    own = [c for c in chunks if not c.meta.get("inherited")]
    assert len(own) == 2 and {c.meta["record_session_id"] for c in own} == {SID}
    assert all(c.meta["inherited"] is True for c in inherited) and all("inherited" not in c.meta for c in own)
    assert all(c.meta["parent_session_id"] == SID and c.meta["agent_id"] == AGENT_ID for c in chunks)
    # Authority is exactly what origin and flags say: an inherited prompt is no more a user turn than its own.
    assert [(c.kind, c.meta["role"], c.meta.get("evidence_origin")) for c in chunks[:3]] == [
        ("meta", "runtime", "parent_task"), ("meta", "runtime", "parent_task"), ("meta", "runtime", "coordinator")]
    assert [c.kind for c in chunks[3:6]] == ["turn", "tool_call", "tool_result"]
    result = sync(engine, path, CHILD)
    assert result["ok"] and result["coverage"]["complete"] is True, result["coverage"]


def test_inherited_history_never_upgrades_authority(engine, root):
    path = write(sub_file(root),
                 human(f"older human {PHRASE}", sub=True, sid=OTHER),
                 user(f"older parent task {PHRASE}", sub=True, sid=OTHER),
                 user(f"older meta {PHRASE}", sub=True, sid=OTHER, isMeta=True),
                 user(f"older compact {PHRASE}", sub=True, sid=OTHER, isCompactSummary=True))
    chunks = parse(path)
    assert [(c.kind, c.meta["role"]) for c in chunks] == [("turn", "user")] + [("meta", "runtime")] * 3
    assert [c.meta.get("evidence_origin") for c in chunks[1:]] == ["parent_task", "injected_meta", "compaction"]
    assert all(c.meta["inherited"] and c.meta["record_session_id"] == OTHER for c in chunks)


def test_inherited_session_ids_need_a_constant_agent_id(root):
    other_agent = user("older task", sub=True, sid=OTHER)
    other_agent["agentId"] = "ffffffffffffffff"
    unattributed = user("older task", sub=True, sid=OTHER)
    del unattributed["agentId"]
    first_inherited = user("older task", sub=True, sid=OTHER)
    first_inherited["agentId"] = "ffffffffffffffff"
    cases = {
        "agent_changes_on_inherited_record": [user("task", sub=True), other_agent],
        "inherited_record_without_agent_id": [user("task", sub=True), unattributed],
        "agent_changes_after_inherited_record": [first_inherited, user("task", sub=True)],
    }
    for name, records in cases.items():
        chunks = parse(write(sub_file(root), *records))
        assert [bool(c.meta.get("parse_error")) for c in chunks] == [False, True], name
        assert "inherited" not in chunks[1].meta and chunks[1].kind == "meta"


def test_main_files_keep_the_strict_session_check_even_with_an_agent_id(root):
    chunks = parse(write(main_file(root), human("mine"), human("foreign", sid=OTHER, agentId=AGENT_ID)))
    assert [bool(c.meta.get("parse_error")) for c in chunks] == [False, True]


def test_agent_id_must_not_change_inside_one_file(root):
    other = user("another agent", sub=True)
    other["agentId"] = "ffffffffffffffff"
    chunks = parse(write(sub_file(root), user("task", sub=True), other))
    assert chunks[1].meta.get("parse_error") is True


def test_parent_and_child_never_mix_evidence_in_the_index(engine, root):
    write(main_file(root), human(f"main says {PHRASE}"), assistant(text("main answer")))
    write(sub_file(root), user("child parent-task text CHILD-ONLY", sub=True), assistant(text("child answer CHILD-ONLY"), sub=True))
    assert sync(engine, main_file(root), SID)["coverage"]["complete"]
    assert sync(engine, sub_file(root), CHILD)["coverage"]["complete"]
    main_text, child_text = dump(stored(engine, SID)), dump(stored(engine, CHILD))
    assert "CHILD-ONLY" not in main_text and PHRASE not in child_text
    assert {c.session_id for c in stored(engine, SID)} == {SID} and {c.session_id for c in stored(engine, CHILD)} == {CHILD}
    must_not_user(engine)
    assert engine.audit_session(SID, provider="claude", path=str(main_file(root)))["verdict"] == "violation"
    assert engine.audit_session(CHILD, provider="claude", path=str(sub_file(root)))["verdict"] == "clean"
    hits = engine.search_history("CHILD-ONLY", session_id=SID, provider="claude", mode="sparse")["hits"]
    assert hits == []


def test_copy_of_the_parent_id_in_a_child_file_cannot_masquerade_as_the_parent(engine, root):
    write(sub_file(root), user("task", sub=True))
    refused = engine.sync_session(session_id=SID, provider="claude", path=str(sub_file(root)), embed=False)
    assert refused["ok"] is False and refused["error"] == "bad_request"


# ---------------------------------------------------------------- authority through the audit

def test_audit_only_a_human_prompt_triggers_a_user_role_prohibition(engine, root):
    path = main_file(root)
    write(path, human("start"))
    must_not_user(engine)
    assert engine.audit_session(SID, provider="claude", path=str(path))["verdict"] == "clean"
    write(path, *[case[2]() for case in NON_USER if not case[1]], mode="a")
    write(path, assistant(text(f"assistant echoes {PHRASE}"), use("toolu_e", "Bash", command=f"echo {PHRASE}")),
          tool_result("toolu_e", f"{PHRASE}"), mode="a")
    result = engine.audit_session(SID, provider="claude", path=str(path))
    assert result["coverage"]["complete"] is True, result["coverage"]
    assert result["verdict"] == "clean", [f["evidence"] for f in result["findings"]]
    write(path, human(f"now please {PHRASE}"), mode="a")
    result = engine.audit_session(SID, provider="claude", path=str(path))
    assert result["verdict"] == "violation" and result["coverage"]["complete"] is True
    assert len(result["findings"][0]["evidence"]) == 1


def test_audit_legacy_prompt_and_human_queued_command_trigger_it(engine, root):
    must_not_user(engine)
    legacy = write(main_file(root), user(f"legacy {PHRASE}"))
    assert engine.audit_session(SID, provider="claude", path=str(legacy))["verdict"] == "violation"
    queued = write(main_file(root, OTHER), attachment("queued_command", prompt=f"queued {PHRASE}", sid=OTHER,
                                                      commandMode="prompt", origin={"kind": "human"}))
    assert engine.audit_session(OTHER, provider="claude", path=str(queued))["verdict"] == "violation"


def test_audit_subagent_prompts_never_trigger_user_prohibitions(engine, root):
    path = write(sub_file(root), user(f"parent asks {PHRASE}", sub=True),
                 user(f"coordinator {PHRASE}", sub=True, origin={"kind": "coordinator"}),
                 user(f"notify {PHRASE}", sub=True, origin={"kind": "task-notification", "producer": "p", "runId": "r"}))
    must_not_user(engine)
    result = engine.audit_session(CHILD, provider="claude", path=str(path))
    assert result["verdict"] == "clean" and result["coverage"]["complete"] is True


def test_check_action_uses_linked_success_for_temporal_requirements(engine, root):
    path = main_file(root)
    engine.upsert_contracts([{"id": "tests-first", "provider": "claude", "type": "must", "scope": "tool_call",
                              "regex": "npm test", "before_regex": "deploy", "requires_success": True}])
    action = {"tool_name": "Bash", "arguments": {"command": "deploy now"}}
    write(path, human("ship it"), assistant(use("toolu_t", "Bash", command="npm test")))
    pending = engine.check_action(SID, action, provider="claude", path=str(path))
    assert pending["verdict"] == "unknown" and pending["coverage"]["complete"] is True
    write(path, tool_result("toolu_t", "ran", block_extra={"is_error": False}), mode="a")
    assert engine.check_action(SID, action, provider="claude", path=str(path))["verdict"] == "clean"
    path2 = main_file(root, OTHER)
    write(path2, human("ship it", sid=OTHER), assistant(use("toolu_t", "Bash", command="npm test"), sid=OTHER),
          tool_result("toolu_t", "Exit code 1", block_extra={"is_error": True}, sid=OTHER))
    failed = engine.check_action(OTHER, action, provider="claude", path=str(path2))
    assert failed["verdict"] == "violation"


# ---------------------------------------------------------------- discovery and path validation

def test_resolve_by_session_id_for_main_and_subagent_sessions(root):
    write(main_file(root), human("main"))
    write(sub_file(root), user("task", sub=True))
    assert claude.resolve_session_path(root, session_id=SID) == main_file(root)
    assert claude.resolve_session_path(root, session_id=CHILD) == sub_file(root)
    assert claude.resolve_session_path(root, path=str(sub_file(root))) == sub_file(root)
    assert claude.resolve_session_path(root, session_id=SID, path=str(main_file(root))) == main_file(root)


def test_resolve_failures_missing_ambiguous_and_mismatched(root, tmp_path):
    write(main_file(root), human("main"))
    with pytest.raises(FileNotFoundError):
        claude.resolve_session_path(root, session_id=OTHER)
    with pytest.raises(FileNotFoundError):
        claude.resolve_session_path(root, session_id=f"{SID}_agent-missing")
    write(main_file(root, project="-copy-of-project"), human("duplicate id in another project"))
    with pytest.raises(FileNotFoundError, match="ambiguous"):
        claude.resolve_session_path(root, session_id=SID)
    with pytest.raises(ValueError):
        claude.resolve_session_path(root, session_id="../etc/passwd")
    with pytest.raises(ValueError):
        claude.resolve_session_path(root, session_id=SID.upper())
    with pytest.raises(ValueError):
        claude.resolve_session_path(root)
    with pytest.raises(ValueError):
        claude.resolve_session_path(root, session_id=OTHER, path=str(main_file(root)))
    with pytest.raises(FileNotFoundError):
        claude.resolve_session_path(root, path=str(main_file(root, OTHER)))


def test_path_validation_rejects_traversal_escapes_symlinks_and_foreign_layouts(root, tmp_path):
    inside = write(main_file(root), human("main"))
    outside = write(tmp_path / "elsewhere" / PROJECT / f"{OTHER}.jsonl", human("outside"))
    with pytest.raises(claude.PathTraversalError):
        claude.resolve_session_path(root, path=str(outside))
    with pytest.raises(claude.PathTraversalError):
        claude.resolve_session_path(root, path=str(root / PROJECT / ".." / PROJECT / f"{SID}.jsonl"))
    with pytest.raises(claude.PathTraversalError):
        claude.resolve_session_path(root, path=f"{PROJECT}/{SID}.jsonl")
    with pytest.raises(claude.PathTraversalError):
        claude.resolve_session_path(root, path=str(root / ".." / "projects" / PROJECT / f"{SID}.jsonl"))
    # A symlinked file may not redirect evidence to another session or escape the root.
    link = root / PROJECT / f"{OTHER}.jsonl"
    link.symlink_to(inside)
    with pytest.raises(claude.PathTraversalError):
        claude.resolve_session_path(root, path=str(link))
    escape = root / PROJECT / "cccccccc-3333-4333-8333-cccccccccccc.jsonl"
    escape.symlink_to(outside)
    with pytest.raises(claude.PathTraversalError):
        claude.resolve_session_path(root, path=str(escape))
    # So may a symlinked project or session directory.
    (root / "-linked-project").symlink_to(outside.parent, target_is_directory=True)
    with pytest.raises(claude.PathTraversalError):
        claude.resolve_session_path(root, path=str(root / "-linked-project" / f"{OTHER}.jsonl"))
    real = write(sub_file(root, "dddddddd-4444-4444-8444-dddddddddddd"), user("task", sub=True,
                                                                           sid="dddddddd-4444-4444-8444-dddddddddddd"))
    (root / PROJECT / "eeeeeeee-5555-4555-8555-eeeeeeeeeeee").symlink_to(real.parent.parent, target_is_directory=True)
    with pytest.raises(claude.PathTraversalError):
        claude.resolve_session_path(root, path=str(root / PROJECT / "eeeeeeee-5555-4555-8555-eeeeeeeeeeee"
                                                   / "subagents" / real.name))
    # The configured root itself may be a symlink.
    alias = tmp_path / "alias-root"
    alias.symlink_to(root, target_is_directory=True)
    assert claude.resolve_session_path(alias, path=str(inside)) == inside
    # Layouts that are not official Claude Code transcript locations.
    for bad in (root / PROJECT / "notes.jsonl", root / PROJECT / SID / "nested" / "x.jsonl",
                root / f"{SID}.jsonl", root / PROJECT / "a" / "b" / SID / "subagents" / f"{STEM}.jsonl"):
        write(bad, human("x"))
        with pytest.raises(ValueError):
            claude.resolve_session_path(root, path=str(bad))
    with pytest.raises(ValueError):
        claude.resolve_session_path(root, path=str(root))


def test_engine_reports_path_errors_without_reading_outside_the_root(engine, root, tmp_path):
    outside = write(tmp_path / "elsewhere" / PROJECT / f"{OTHER}.jsonl", human("outside"))
    assert engine.sync_session(session_id=OTHER, provider="claude", path=str(outside), embed=False)["error"] == "path_traversal"
    assert engine.sync_session(session_id="../x", provider="claude", embed=False)["error"] == "bad_request"
    assert engine.sync_session(session_id=SID, provider="claude", embed=False)["error"] == "not_found"
    write(main_file(root), human("hello"))
    by_id = engine.sync_session(session_id=SID, provider="claude", embed=False)
    assert by_id["ok"] and by_id["provider"] == "claude" and by_id["path"] == str(main_file(root))


def test_discovery_lists_only_valid_unlinked_transcripts(root, tmp_path):
    write(main_file(root), human("a"))
    write(main_file(root, OTHER, project="-second"), human("b"))
    write(sub_file(root), user("task", sub=True))
    write(root / PROJECT / "notes.jsonl", human("not a session"))
    write(root / PROJECT / SID / "subagents" / "bad name.jsonl", human("bad"))
    write(root / PROJECT / SID / "other" / f"{STEM}.jsonl", human("bad layout"))
    link = main_file(root, "cccccccc-3333-4333-8333-cccccccccccc")
    link.symlink_to(main_file(root))
    assert claude.discover_session_files(root) == sorted([main_file(root), main_file(root, OTHER, "-second"), sub_file(root)])
    assert claude.discover_session_files(root, include_subagents=False) == sorted(
        [main_file(root), main_file(root, OTHER, "-second")])
    assert claude.discover_session_files(tmp_path / "missing") == []


def test_provider_is_auto_detected_from_a_path_under_the_claude_root(engine, root, tmp_path):
    write(main_file(root), human("hello"))
    write(sub_file(root), user("task", sub=True))
    assert engine._resolve_provider(None, path=str(main_file(root))) == "claude"
    assert engine._resolve_provider(None, path=str(sub_file(root))) == "claude"
    assert engine._resolve_provider(None, path=str(root / PROJECT / "notes.jsonl")) == "codex"
    assert engine._resolve_provider("CLAUDE") == "claude"
    assert sync_without_provider(engine, main_file(root))["provider"] == "claude"
    with pytest.raises(ValueError, match="claude"):
        engine._resolve_provider("claude-code")


def sync_without_provider(engine, path):
    return engine.sync_session(session_id=SID, path=str(path), embed=False)


# ---------------------------------------------------------------- configuration

def test_claude_projects_dir_default_environment_and_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    assert Settings(_env_file=None).resolve_claude_projects_dir() == tmp_path / "home" / ".claude" / "projects"
    monkeypatch.setenv("SELF_DIRECT_CLAUDE_PROJECTS_DIR", str(tmp_path / "from-env"))
    assert Settings(_env_file=None).resolve_claude_projects_dir() == tmp_path / "from-env"
    assert Settings(_env_file=None, claude_projects_dir=tmp_path / "explicit").resolve_claude_projects_dir() == (
        tmp_path / "explicit")


def test_claude_projects_dir_from_toml_is_config_relative(tmp_path, monkeypatch):
    selected = tmp_path / "configuration" / "settings.toml"
    selected.parent.mkdir()
    selected.write_text('[paths]\nclaude_projects_dir = "../claude-data"\n', encoding="utf-8")
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(selected))
    assert Settings(_env_file=None).resolve_claude_projects_dir() == tmp_path / "claude-data"
    monkeypatch.setenv("SELF_DIRECT_CLAUDE_PROJECTS_DIR", str(tmp_path / "env-wins"))
    assert Settings(_env_file=None).resolve_claude_projects_dir() == tmp_path / "env-wins"


def test_claude_is_a_valid_session_provider_setting_and_chunk_provider():
    assert Settings(_env_file=None, session_provider="claude").resolve_session_provider() == "claude"
    from self_directing_mcp.schemas import Chunk, ContractRule
    assert ContractRule(id="r", type="must_not", provider="claude", regex="x").provider == "claude"
    assert Chunk(chunk_id="c", session_id="s", provider="claude", kind="meta", text="t", content_hash="h").provider == "claude"


# ---------------------------------------------------------------- incremental parsing, versioning, memory

def test_cursor_resumes_at_record_boundaries_and_ignores_unfinished_records(engine, root):
    path = write(main_file(root), human("one"), assistant(text("two")))
    first = sync(engine, path)
    assert first["new_chunks"] == 2 and first["coverage"]["complete"]
    write(path, human("three"), mode="a")
    with path.open("ab") as stream:
        stream.write(b'{"type":"user","message":')
    second = sync(engine, path)
    assert second["new_chunks"] == 1 and not second["reindexed"]
    assert "unread_bytes" in second["coverage"]["issues"] and second["coverage"]["complete"] is False
    chunks, offset, _ = claude.parse_session_chunks(path, start_byte=first["byte_offset"])
    assert [c.text for c in chunks] == ["[message:user] three"] and offset < path.stat().st_size


def test_parser_version_is_registered_and_a_bump_reindexes(engine, root, monkeypatch):
    assert current_parser_version("claude") == claude.PARSER_VERSION >= 1
    path = write(main_file(root), human("one"))
    monkeypatch.setattr(claude, "PARSER_VERSION", claude.PARSER_VERSION + 1)
    assert sync(engine, path)["parser_version"] == claude.PARSER_VERSION
    old_ids = {c.chunk_id for c in stored(engine)}
    monkeypatch.setattr(claude, "PARSER_VERSION", claude.PARSER_VERSION + 1)
    upgraded = sync(engine, path)
    assert upgraded["reindexed"] and upgraded["reindex_reason"] == "parser_updated"
    assert {c.chunk_id for c in stored(engine)} == old_ids and upgraded["coverage"]["complete"]


def _large_session(path, count, bulk="x" * 900):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for i in range(count):
            stream.write(json.dumps(human(f"prompt {i:05d} {bulk}")) + "\n")
            stream.write(json.dumps(assistant(use(f"toolu_{i}", "Bash", command=f"echo {i} {bulk}"))) + "\n")
            stream.write(json.dumps(tool_result(f"toolu_{i}", f"output {i} {bulk}",
                                                block_extra={"is_error": False})) + "\n")
    return path.stat().st_size


def test_parsing_a_large_synthetic_session_holds_one_record_at_a_time(root):
    path = main_file(root)
    size = _large_session(path, 5000)
    assert size > 15_000_000

    class Sink(list):
        seen = 0

        def append(self, chunk):  # discard: only the parser's own working set is measured
            self.seen += 1

    sink = Sink()
    tracemalloc.start()
    try:
        claude.parse_session_chunks(path, sink=sink)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert sink.seen == 15000 and len(sink) == 0
    assert peak < 1_000_000, (peak, size)  # independent of the file size


def test_large_synthetic_session_is_indexed_with_bounded_memory(engine, root):
    path = main_file(root)
    size = _large_session(path, 5000)
    tracemalloc.start()
    try:
        result = sync(engine, path)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert result["ok"] and result["total_chunks"] == 15000 and result["coverage"]["complete"]
    assert peak < 0.5 * size, (peak, size)
