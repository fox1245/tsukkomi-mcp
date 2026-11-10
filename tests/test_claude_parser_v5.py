"""Claude Code parser v5: record shapes written by Claude Code 2.1.31x.

Synthetic data only; every value below is made up. Each new shape is accepted only through a strict shape check (key
set, value types, identifier patterns) and is classified by what it carries:

* identifier-only bookkeeping (``session-owner``, ``tools-used``, the list form of an api-request's ``cacheControl``)
  is session metadata, and its identifiers are never indexed;
* host notices and injected text (effort-mode reminders, the advisor tool change, command permission grants, re-injected
  skills, cancelled hooks, plan-mode exits, stripped thinking or advisor blocks, killed agents, an emptied queue, an
  auto-continued turn) are runtime evidence and never user authority;
* a peer message stays a peer message whatever new fields its origin carries, and a human prompt queued with a pasted
  image is a user turn;
* a hosted tool call (``server_tool_use``) is a call and its advisor result (encrypted advice, or an error code) a
  result, linked by id.

Every deviation (an extra key, a wrong value type, a malformed identifier) stays ``unsupported_event``.
"""
from __future__ import annotations

import pytest

from self_directing_mcp import claude
from self_directing_mcp.index.ingest import current_parser_version

from test_claude_provider import (  # noqa: F401  (clean_environment is autouse, engine/root are fixtures)
    CHILD, PHRASE, SID, assistant, attachment, clean_environment, dump, engine, env, human, main_file, parse, root,
    stored, sub_file, sync, text, tool_result, use, user, write)

ACCOUNT = "5e000000-0000-4000-8000-0000000000a1"
ORGANIZATION = "5e000000-0000-4000-8000-0000000000b2"
TOOL_NAME = "Tools-Used-Sentinel"
PARAMS = "API-PARAMS-SENTINEL"
SKILL_BODY = "SKILL-BODY-SENTINEL"
SKILL_PATH = "/synthetic/skills/SKILL-PATH-SENTINEL.md"
ENCRYPTED = "ENCRYPTED-ADVICE-SENTINEL"
PDF_DATA = "PDF-DATA-SENTINEL"
IMAGE_DATA = "PASTED-IMAGE-SENTINEL"
PEER_BODY = "PEER-BODY-SENTINEL"
PROC_START = "PROC-START-SENTINEL"
RULE = "Bash(rule-sentinel:*)"
SENTINELS = (ACCOUNT, ORGANIZATION, TOOL_NAME, PARAMS, SKILL_BODY, "SKILL-PATH-SENTINEL", ENCRYPTED, PDF_DATA,
             IMAGE_DATA, PEER_BODY, PROC_START, "rule-sentinel", "HANDBACK-REPORT-SENTINEL", "HOOK-COMMAND-SENTINEL",
             "QUEUED-POPALL-SENTINEL")
POP_ALL_TEXT = "QUEUED-POPALL-SENTINEL"

PEER = {"kind": "peer", "body": PEER_BODY, "from": "agent-x", "name": "reviewer",
        "msg_id": "00000000-0000-4000-8000-0000000000c3", "fromMode": "bypass", "verifiedPeerPid": 4242,
        "verifiedPeerProcStart": PROC_START, "hopChain": ["session-a", "session-b"]}
BREAKPOINTS = [{"path": ["system", 0], "value": {"type": "ephemeral", "ttl": "1h", "scope": "global"}},
               {"path": ["system", 1], "value": {"type": "ephemeral", "ttl": "1h"}},
               {"path": ["messages", 3, "content", 2], "value": {"type": "ephemeral", "ttl": "5m"}}]
PDF = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": PDF_DATA}}
IMAGE = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": IMAGE_DATA}}


def session_owner(sub=False, **fields):
    return env("session-owner", sub=sub, **{"accountUuid": ACCOUNT, "organizationUuid": ORGANIZATION, **fields})


def tools_used(sub=False, **fields):
    return env("tools-used", sub=sub, **{"tools": ["Bash", "Read", TOOL_NAME], **fields})


def api_request(sub=False, **fields):
    record = {"id": "00000000-0000-4000-8000-0000000000d4", "base": "00000000-0000-4000-8000-0000000000d5",
              "tail": ["00000000-0000-4000-8000-0000000000d6"], "keep": 2, "querySource": "repl_main_thread",
              "requestDigest": "digest-1", "shapeHash": "shape-1", "params": {"max_tokens": 1, "note": PARAMS},
              "cacheControl": BREAKPOINTS}
    record.update(fields)
    return env("api-request", sub=sub, **{key: value for key, value in record.items() if value is not ...})


def file_attachment(sub=False, **fields):
    a = {"filename": "/synthetic/a.txt", "displayPath": "a.txt", "savedCopyFraming": {"enveloped": True},
         "content": {"type": "text", "file": {"filePath": "/synthetic/a.txt", "content": "FILE-BODY", "numLines": 1,
                                              "startLine": 1, "totalLines": 1}}}
    a.update(fields)
    return attachment("file", sub=sub, **a)


def advisor_tool(sub=False, **fields):
    return attachment("advisor_tool", sub=sub, **{"available": True, "model": "claude-synthetic", "toolChange": "add",
                                                  **fields})


def invoked_skills(sub=False, skills=None, **fields):
    skills = [{"content": SKILL_BODY, "name": "skill-one", "path": SKILL_PATH},
              {"content": SKILL_BODY, "name": "skill two", "path": SKILL_PATH}] if skills is None else skills
    return attachment("invoked_skills", sub=sub, skills=skills, **fields)


def agents_killed(sub=False, **fields):
    return env("system", sub=sub, **{"subtype": "agents_killed", "isMeta": False, "entrypoint": "cli",
                                     "gitBranch": "main", "slug": "a-b-c", **fields})


def server_tool_use(sub=False, **fields):
    return assistant({"type": "server_tool_use", "id": "srvtoolu_1", "name": "advisor", "input": {}, **fields}, sub=sub)


def advisor_result(sub=False, content=None, **fields):
    content = {"type": "advisor_redacted_result", "encrypted_content": ENCRYPTED} if content is None else content
    return assistant({"type": "advisor_tool_result", "tool_use_id": "srvtoolu_1", "content": content, **fields}, sub=sub)


HANDBACK_PEER = {"kind": "peer", "body": PEER_BODY, "from": "agent-y", "name": "worker", "senderTaskId": "task-1",
                 "handback": True, "handbackReportText": "HANDBACK-REPORT-SENTINEL"}
PLAN_PATH = "/synthetic/plans/plan.md"


def hook_cancelled(sub=False, **fields):
    return attachment("hook_cancelled", sub=sub, **{
        "command": "HOOK-COMMAND-SENTINEL", "durationMs": 5003, "hookEvent": "UserPromptSubmit",
        "hookName": "UserPromptSubmit", "timedOut": True, "timeoutMs": 5000,
        "toolUseID": "00000000-0000-4000-8000-0000000000e7", **fields})


def thinking_stripped(sub=False, **fields):
    return attachment("thinking_stripped", sub=sub, **{"from": {"messageId": "msg_synthetic_1", "thinkingIndex": 0},
                                                       "scope": "partial", **fields})


def peer_user(sub=False, origin=None):
    return user(f"peer says {PHRASE}", sub=sub, isMeta=True, origin=PEER if origin is None else origin,
                promptSource="system", turnOrigin="peer")


def queued(sub=False, **fields):
    return attachment("queued_command", sub=sub, **{"delivery_id": "d1", "source_uuid": "s1",
                                                    "timestamp": "2026-10-01T00:00:00.000Z", **fields})


def peer_queued(sub=False, origin=None):
    return queued(sub=sub, commandMode="prompt", isMeta=True, origin=PEER if origin is None else origin,
                  prompt=f"peer {PHRASE}", reminderId="rem-1")


def notification_queued(sub=False, **fields):
    return queued(sub=sub, **{"commandMode": "task-notification", "runId": "r1", "prompt": "done",
                              "origin": {"kind": "task-notification", "producer": "session-task"},
                              "toolsUsedDelta": ["Bash", TOOL_NAME], "usage": {"totalTokens": 1}, **fields})


def human_queued_image(**fields):
    return queued(**{"commandMode": "prompt", "origin": {"kind": "human"}, "humanTurn": True, "imagePasteIds": [1],
                     "prompt": [IMAGE, text(f"look at this {PHRASE}")], **fields})


def document_result(content=None):
    return tool_result("toolu_pdf", [PDF, text("page text")] if content is None else content,
                       block_extra={"is_error": False})


# name -> (builder taking sub, expected kind, role, evidence_origin, expected text, expected meta subset, meta keys that
# must be absent)
ACCEPTED = {
    "session-owner": (session_owner, "meta", "runtime", "session_metadata", "[claude_metadata:session-owner]", {}, ()),
    "tools-used": (tools_used, "meta", "runtime", "session_metadata", "[claude_metadata:tools-used]",
                   {"tool_count": 3}, ()),
    "api-request-breakpoints": (api_request, "meta", "runtime", "session_metadata", "[claude_metadata:api-request]",
                                {"query_source": "repl_main_thread"}, ()),
    "api-request-without-base": (lambda sub: api_request(sub, base=...), "meta", "runtime", "session_metadata",
                                 "[claude_metadata:api-request]", {}, ()),
    "file-saved-copy-framing": (file_attachment, "meta", "runtime", "attachment:file",
                                "[attachment:file] display_path=a.txt filename=/synthetic/a.txt",
                                {"saved_copy_enveloped": True}, ()),
    "ultra_effort_enter": (lambda sub: attachment("ultra_effort_enter", sub=sub, reminderType="sparse"), "meta",
                           "runtime", "attachment:ultra_effort_enter",
                           "[attachment:ultra_effort_enter] reminder_type=sparse", {"reminder_type": "sparse"}, ()),
    "ultra_effort_exit": (lambda sub: attachment("ultra_effort_exit", sub=sub), "meta", "runtime",
                          "attachment:ultra_effort_exit", "[attachment:ultra_effort_exit]", {}, ()),
    "advisor_tool": (advisor_tool, "meta", "runtime", "attachment:advisor_tool",
                     "[attachment:advisor_tool] tool_change=add available=true",
                     {"tool_change": "add", "available": True, "advisor_model": "claude-synthetic"}, ("by_base",)),
    "advisor_tool_by_base": (lambda sub: advisor_tool(sub, toolChange="remove", available=False, byBase=True), "meta",
                             "runtime", "attachment:advisor_tool",
                             "[attachment:advisor_tool] tool_change=remove available=false", {"by_base": True}, ()),
    "command_permissions": (lambda sub: attachment("command_permissions", sub=sub, allowedTools=["Read", RULE]), "meta",
                            "runtime", "attachment:command_permissions", "[attachment:command_permissions] allowed_tools=2",
                            {"allowed_tool_count": 2, "allowed_tool_names": ["Read"]}, ()),
    "command_permissions_empty": (lambda sub: attachment("command_permissions", sub=sub, allowedTools=[]), "meta",
                                  "runtime", "attachment:command_permissions",
                                  "[attachment:command_permissions] allowed_tools=0", {"allowed_tool_count": 0}, ()),
    "invoked_skills": (invoked_skills, "meta", "runtime", "attachment:invoked_skills",
                       "[attachment:invoked_skills] count=2 names=skill-one",
                       {"skill_count": 2, "skill_names": ["skill-one"]}, ()),
    "agents_killed": (agents_killed, "meta", "runtime", "runtime_metadata", "[system:agents_killed]",
                      {"system_subtype": "agents_killed"}, ()),
    "peer_origin_user": (peer_user, "meta", "runtime", "peer_message", f"[peer_message] peer says {PHRASE}",
                         {"origin_kind": "peer", "peer_from": "agent-x", "peer_name": "reviewer",
                          "peer_msg_id": "00000000-0000-4000-8000-0000000000c3", "peer_from_mode": "bypass",
                          "peer_verified_pid": 4242, "peer_hop_count": 2}, ()),
    "peer_origin_queued": (peer_queued, "meta", "runtime", "peer_message", f"[peer_message] peer {PHRASE}",
                           {"peer_msg_id": "00000000-0000-4000-8000-0000000000c3", "queued_command": True}, ()),
    "notification_tools_delta": (notification_queued, "meta", "runtime", "task_notification",
                                 "[task_notification] done", {"tools_used_delta_count": 2, "run_id": "r1"}, ()),
    "server_tool_use": (server_tool_use, "tool_call", "assistant", "hosted_tool", "[tool_call:advisor] {}",
                        {"tool_name": "advisor", "call_id": "srvtoolu_1"}, ()),
    "advisor_tool_result": (advisor_result, "tool_result", "tool", "hosted_tool",
                            "[tool_result] [advisor_redacted_result]", {"call_id": "srvtoolu_1", "success": True}, ()),
    "advisor_tool_result_stop_reason": (
        lambda sub: advisor_result(sub, content={"type": "advisor_redacted_result", "encrypted_content": ENCRYPTED,
                                                 "stop_reason": "end_turn"}),
        "tool_result", "tool", "hosted_tool", "[tool_result] [advisor_redacted_result]", {"success": True}, ()),
    "advisor_tool_result_error": (
        lambda sub: advisor_result(sub, content={"type": "advisor_tool_result_error",
                                                 "error_code": "execution_time_exceeded"}),
        "tool_result", "tool", "hosted_tool", "[tool_result] [advisor_tool_result_error] error_code=execution_time_exceeded",
        {"success": False, "error_code": "execution_time_exceeded", "call_id": "srvtoolu_1"}, ()),
    "peer_origin_handback_report": (
        lambda sub: peer_user(sub, origin=HANDBACK_PEER), "meta", "runtime", "peer_message",
        f"[peer_message] peer says {PHRASE}", {"peer_task_id": "task-1", "peer_from": "agent-y"}, ()),
    "auto_continuation_origin": (
        lambda sub: user(f"continue {PHRASE}", sub=sub, isMeta=True, origin={"kind": "auto-continuation"},
                         queuePriority="later", turnOrigin="auto_continuation"),
        "meta", "runtime", "auto_continuation", f"[auto_continuation] continue {PHRASE}",
        {"origin_kind": "auto-continuation"}, ()),
    "hook_cancelled": (hook_cancelled, "meta", "runtime", "attachment:hook_cancelled",
                       "[attachment:hook_cancelled] event=UserPromptSubmit name=UserPromptSubmit timed_out=true",
                       {"hook_event": "UserPromptSubmit", "timed_out": True, "timeout_ms": 5000,
                        "tool_use_id": "00000000-0000-4000-8000-0000000000e7"}, ()),
    "plan_mode_exit": (lambda sub: attachment("plan_mode_exit", sub=sub, planExists=True, planFilePath=PLAN_PATH),
                       "meta", "runtime", "attachment:plan_mode_exit", "[attachment:plan_mode_exit] plan_exists=true",
                       {"plan_exists": True, "plan_file_path": PLAN_PATH}, ()),
    "thinking_stripped": (thinking_stripped, "meta", "runtime", "attachment:thinking_stripped",
                          "[attachment:thinking_stripped] scope=partial",
                          {"scope": "partial", "stripped_message_id": "msg_synthetic_1", "thinking_index": 0}, ()),
    "advisor_stripped": (lambda sub: attachment("advisor_stripped", sub=sub), "meta", "runtime",
                         "attachment:advisor_stripped", "[attachment:advisor_stripped]", {}, ()),
    "queue_pop_all": (lambda sub: env("queue-operation", sub=sub, operation="popAll", content=POP_ALL_TEXT),
                      "meta", "runtime", "runtime_metadata", "[queue_operation:popAll]", {"operation": "popAll"}, ()),
    "queue_pop_all_bare": (lambda sub: env("queue-operation", sub=sub, operation="popAll"), "meta", "runtime",
                           "runtime_metadata", "[queue_operation:popAll]", {"operation": "popAll"}, ()),
}


def test_parser_version_is_five_and_registered():
    assert claude.PARSER_VERSION == 5 and current_parser_version("claude") == 5


@pytest.mark.parametrize("sub", [False, True], ids=["main", "subagent"])
@pytest.mark.parametrize("name", sorted(ACCEPTED))
def test_new_shapes_are_supported_and_classified_by_what_they_carry(root, name, sub):
    build, kind, role, origin, expected_text, meta, absent = ACCEPTED[name]
    chunks = parse(write(sub_file(root) if sub else main_file(root), build(sub)))
    assert len(chunks) == 1, [c.text for c in chunks]
    (chunk,) = chunks
    assert not chunk.meta.get("unsupported_event") and not chunk.meta.get("parse_error"), chunk.text
    assert (chunk.kind, chunk.meta["role"], chunk.meta.get("evidence_origin")) == (kind, role, origin)
    assert chunk.text == expected_text
    assert {key: chunk.meta.get(key) for key in meta} == meta
    assert not set(absent) & set(chunk.meta)
    blob = dump(chunks)
    for sentinel in SENTINELS:
        assert sentinel not in blob, sentinel
    assert chunk.meta["role"] != "user"


def test_identifiers_of_bookkeeping_records_are_kept_nowhere(root):
    chunks = parse(write(main_file(root), session_owner(), tools_used(), api_request()))
    blob = dump(chunks)
    for value in (ACCOUNT, ORGANIZATION, TOOL_NAME, "Bash", "Read", PARAMS, "ephemeral", "global"):
        assert value not in blob, value
    assert not any(key in c.meta for c in chunks for key in ("accountUuid", "organizationUuid", "account_uuid", "tools"))


def test_a_document_in_a_tool_result_is_a_placeholder(root):
    (chunk,) = parse(write(main_file(root), document_result()))
    assert chunk.kind == "tool_result" and chunk.meta["role"] == "tool" and not chunk.meta.get("unsupported_event")
    assert chunk.text == "[tool_result] [document:application/pdf] page text"
    assert chunk.meta["success"] is True and PDF_DATA not in dump([chunk])


def test_a_human_prompt_queued_with_a_pasted_image_is_a_user_turn(root):
    chunks = parse(write(main_file(root), human_queued_image()))
    assert [(c.kind, c.meta["role"], c.text) for c in chunks] == [
        ("meta", "runtime", "[image:image/png]"), ("turn", "user", f"[message:user] look at this {PHRASE}")]
    assert chunks[1].meta["image_paste_count"] == 1 and chunks[1].meta["human_turn"] is True
    assert not any(c.meta.get("unsupported_event") for c in chunks) and IMAGE_DATA not in dump(chunks)


# --- deviations: every one stays unsupported, never user, never indexed -------------------------------------------

def _bad_peer(**change):
    return {**PEER, **change}


DEVIATIONS = {
    "session-owner": [lambda: session_owner(mystery="x"), lambda: session_owner(accountUuid=5),
                      lambda: session_owner(organizationUuid="not-a-uuid"),
                      lambda: session_owner(accountUuid=ACCOUNT.upper()),
                      lambda: env("session-owner", accountUuid=ACCOUNT)],
    "tools-used": [lambda: tools_used(mystery=1), lambda: tools_used(tools="Bash"), lambda: tools_used(tools=["Bash", 3]),
                   lambda: tools_used(tools=["a tool with spaces"]), lambda: env("tools-used")],
    "api-request": [lambda: api_request(cacheControl=[{"path": ["system", 0]}]),
                    lambda: api_request(cacheControl=[{"path": ["system", 0], "value": {"type": "ephemeral"},
                                                       "extra": 1}]),
                    lambda: api_request(cacheControl=[{"path": [], "value": {"type": "ephemeral"}}]),
                    lambda: api_request(cacheControl=[{"path": ["system", True], "value": {"type": "ephemeral"}}]),
                    lambda: api_request(cacheControl=[{"path": ["a b"], "value": {"type": "ephemeral"}}]),
                    lambda: api_request(cacheControl=[{"path": ["system"], "value": {"ttl": "1h"}}]),
                    lambda: api_request(cacheControl=[{"path": ["system"], "value": {"type": "ephemeral", "x": "y"}}]),
                    lambda: api_request(cacheControl=[{"path": ["system"], "value": {"type": 5}}]),
                    lambda: api_request(cacheControl=["ephemeral"]), lambda: api_request(mystery=1)],
    "file": [lambda: file_attachment(savedCopyFraming={"enveloped": "yes"}),
             lambda: file_attachment(savedCopyFraming={"enveloped": True, "extra": 1}),
             lambda: file_attachment(savedCopyFraming=True), lambda: file_attachment(savedCopyFraming={})],
    "ultra_effort_enter": [lambda: attachment("ultra_effort_enter", reminderType="sparse", mystery=1),
                           lambda: attachment("ultra_effort_enter", reminderType=3),
                           lambda: attachment("ultra_effort_enter", reminderType="two words"),
                           lambda: attachment("ultra_effort_enter")],
    "ultra_effort_exit": [lambda: attachment("ultra_effort_exit", mystery=1),
                          lambda: attachment("ultra_effort_exit", reminderType="full")],
    "advisor_tool": [lambda: advisor_tool(mystery=1), lambda: advisor_tool(available="yes"), lambda: advisor_tool(model=3),
                     lambda: advisor_tool(toolChange="add tool"), lambda: advisor_tool(byBase=1),
                     lambda: attachment("advisor_tool", available=True, model="m")],
    "command_permissions": [lambda: attachment("command_permissions", allowedTools=["Read"], mystery=1),
                            lambda: attachment("command_permissions", allowedTools="Read"),
                            lambda: attachment("command_permissions", allowedTools=["Read", 3]),
                            lambda: attachment("command_permissions")],
    "invoked_skills": [lambda: invoked_skills(mystery=1), lambda: invoked_skills(skills="skill-one"),
                       lambda: invoked_skills(skills=[{"content": SKILL_BODY, "name": "a", "path": "p", "extra": 1}]),
                       lambda: invoked_skills(skills=[{"content": SKILL_BODY, "name": 3, "path": "p"}]),
                       lambda: invoked_skills(skills=[{"content": SKILL_BODY, "name": "a"}]),
                       lambda: invoked_skills(skills=["skill-one"])],
    "agents_killed": [lambda: agents_killed(content=f"killed {PHRASE}"), lambda: agents_killed(mystery=1),
                      lambda: agents_killed(gitBranch=3), lambda: agents_killed(session_id="another-session"),
                      lambda: agents_killed(isMeta="no"), lambda: agents_killed(level=2)],
    "peer_origin": [lambda: peer_user(origin=_bad_peer(verifiedPeerPid="4242")),
                    lambda: peer_user(origin=_bad_peer(verifiedPeerPid=True)),
                    lambda: peer_user(origin=_bad_peer(msg_id=7)), lambda: peer_user(origin=_bad_peer(fromMode=None)),
                    lambda: peer_user(origin=_bad_peer(verifiedPeerProcStart=1)),
                    lambda: peer_user(origin=_bad_peer(hopChain="session-a")),
                    lambda: peer_user(origin=_bad_peer(hopChain=["a session"])),
                    lambda: peer_user(origin=_bad_peer(hopChain=[1])),
                    lambda: peer_user(origin=_bad_peer(mystery=1)),
                    lambda: peer_queued(origin=_bad_peer(verifiedPeerPid="4242")),
                    lambda: peer_queued(origin=_bad_peer(mystery=1))],
    "queued_command": [lambda: human_queued_image(imagePasteIds=["1"]), lambda: human_queued_image(imagePasteIds=[True]),
                       lambda: human_queued_image(imagePasteIds=1),
                       lambda: notification_queued(toolsUsedDelta=["a tool"]),
                       lambda: notification_queued(toolsUsedDelta=[3]), lambda: notification_queued(toolsUsedDelta="Bash")],
    "server_tool_use": [lambda: server_tool_use(mystery=1), lambda: server_tool_use(id=""), lambda: server_tool_use(id=3),
                        lambda: server_tool_use(name="two words"), lambda: server_tool_use(name=""),
                        lambda: server_tool_use(input="query"),
                        lambda: assistant({"type": "server_tool_use", "id": "s", "name": "advisor"}),
                        lambda: server_tool_use(cache_control={"type": "ephemeral"})],
    "advisor_tool_result": [lambda: advisor_result(mystery=1), lambda: advisor_result(tool_use_id=""),
                            lambda: advisor_result(tool_use_id=5),
                            lambda: advisor_result(content={"type": "advisor_redacted_result"}),
                            lambda: advisor_result(content={"type": "advisor_redacted_result", "encrypted_content": 3}),
                            lambda: advisor_result(content={"type": "advisor_redacted_result",
                                                            "encrypted_content": ENCRYPTED, "extra": 1}),
                            lambda: advisor_result(content={"type": "advisor_redacted_result",
                                                            "encrypted_content": ENCRYPTED, "stop_reason": "a b"}),
                            lambda: advisor_result(content={"type": "advisor_result", "text": f"advice {PHRASE}"}),
                            lambda: advisor_result(content={"type": "advisor_tool_result_error",
                                                            "error_code": "two words"}),
                            lambda: advisor_result(content={"type": "advisor_tool_result_error", "error_code": 5}),
                            lambda: advisor_result(content={"type": "advisor_tool_result_error",
                                                            "error_code": "unavailable", "detail": "x"}),
                            lambda: advisor_result(content={"type": "advisor_tool_result_error"}),
                            lambda: advisor_result(content=ENCRYPTED)],
    "peer_handback": [lambda: peer_user(origin={**HANDBACK_PEER, "handbackReportText": 3}),
                      lambda: peer_queued(origin={**HANDBACK_PEER, "handbackReportText": ["x"]})],
    "auto_continuation": [lambda: user("x", isMeta=True, origin={"kind": "auto-continuation", "extra": 1}),
                          lambda: user("x", isMeta=True, origin={"kind": "auto_continuation"})],
    "hook_cancelled": [lambda: hook_cancelled(mystery=1), lambda: hook_cancelled(timedOut="yes"),
                       lambda: hook_cancelled(durationMs="5"), lambda: hook_cancelled(timeoutMs=True),
                       lambda: hook_cancelled(toolUseID=""), lambda: hook_cancelled(command=None),
                       lambda: attachment("hook_cancelled", hookEvent="Stop", hookName="Stop")],
    "plan_mode_exit": [lambda: attachment("plan_mode_exit", planExists="yes", planFilePath=PLAN_PATH),
                       lambda: attachment("plan_mode_exit", planExists=True, planFilePath=None),
                       lambda: attachment("plan_mode_exit", planExists=True, planFilePath=PLAN_PATH, mystery=1),
                       lambda: attachment("plan_mode_exit", planExists=True)],
    "thinking_stripped": [lambda: thinking_stripped(mystery=1), lambda: thinking_stripped(scope="two words"),
                          lambda: thinking_stripped(scope=3),
                          lambda: thinking_stripped(**{"from": {"messageId": "m", "thinkingIndex": "0"}}),
                          lambda: thinking_stripped(**{"from": {"messageId": "m", "thinkingIndex": True}}),
                          lambda: thinking_stripped(**{"from": {"messageId": 1, "thinkingIndex": 0}}),
                          lambda: thinking_stripped(**{"from": {"messageId": "m", "thinkingIndex": 0, "x": 1}}),
                          lambda: thinking_stripped(**{"from": "m"})],
    "advisor_stripped": [lambda: attachment("advisor_stripped", mystery=1)],
    "queue_pop_all": [lambda: env("queue-operation", operation="popAll", content=5),
                      lambda: env("queue-operation", operation="popAll", content=POP_ALL_TEXT, mystery=1),
                      lambda: env("queue-operation", operation="popAll", gitBranch=3)],
    "document": [lambda: document_result([{**PDF, "source": {**PDF["source"], "media_type": "text/plain"}}]),
                 lambda: document_result([{**PDF, "source": {**PDF["source"], "type": "url"}}]),
                 lambda: document_result([{**PDF, "source": {**PDF["source"], "extra": 1}}]),
                 lambda: document_result([{**PDF, "source": {**PDF["source"], "media_type": ["application/pdf"]}}]),
                 lambda: document_result([{**PDF, "title": "t"}]),
                 lambda: user([PDF])],
}
DEVIATION_CASES = [(name, index) for name in sorted(DEVIATIONS) for index in range(len(DEVIATIONS[name]))]


@pytest.mark.parametrize("name,index", DEVIATION_CASES, ids=[f"{name}-{index}" for name, index in DEVIATION_CASES])
def test_each_deviation_stays_unsupported_and_unindexed(root, name, index):
    chunks = parse(write(main_file(root), DEVIATIONS[name][index]()))
    assert any(c.meta.get("unsupported_event") or c.meta.get("parse_error") for c in chunks), [c.text for c in chunks]
    assert not any(c.kind == "turn" and c.meta.get("role") == "user" for c in chunks)
    assert not any(c.kind == "tool_call" for c in chunks)
    blob = dump(chunks)
    for sentinel in SENTINELS + (PHRASE,):
        assert sentinel not in blob, (sentinel, [c.text for c in chunks])


@pytest.mark.parametrize("name", ["session-owner", "advisor_tool_result", "peer_origin", "agents_killed"])
def test_a_deviation_keeps_coverage_incomplete_and_the_audit_unknown(engine, root, name):
    path = write(main_file(root), human("a normal prompt"), DEVIATIONS[name][0]())
    info = sync(engine, path)
    assert info["coverage"]["complete"] is False and "unsupported_events" in info["coverage"]["issues"]
    engine.upsert_contracts([{"id": "x", "type": "must_not", "scope": "tool_call", "regex": "never-matches-anything"}])
    assert engine.audit_session(SID, provider="claude", path=str(path))["verdict"] == "unknown"


# --- authority ---------------------------------------------------------------------------------------------------

def user_rule(kind, phrase, scope="message"):
    return {"id": f"user-{kind}-{scope}", "type": kind, "scope": scope, "roles": ["user"], "regex": phrase,
            "provider": "claude"}


def test_runtime_text_of_the_new_shapes_never_satisfies_or_violates_a_user_role_contract(engine, root):
    phrase = "approved-by-the-owner"
    path = write(main_file(root), human("please summarise the repository"),
                 peer_user(origin={**PEER, "body": phrase}), peer_queued(),
                 queued(commandMode="prompt", isMeta=True, origin=PEER, prompt=phrase, reminderId="rem-2"),
                 invoked_skills(skills=[{"content": phrase, "name": "skill-one", "path": SKILL_PATH}]),
                 attachment("command_permissions", allowedTools=[phrase]),
                 attachment("ultra_effort_enter", reminderType="full"), advisor_tool(),
                 agents_killed(), session_owner(), tools_used(tools=[phrase]),
                 user(f"{phrase} {PHRASE}", isMeta=True, origin={"kind": "auto-continuation"}),
                 user(f"{phrase} {PHRASE}", origin={"kind": "auto-continuation"}),
                 peer_user(origin={**HANDBACK_PEER, "handbackReportText": phrase}),
                 env("queue-operation", operation="popAll", content=f"{phrase} {PHRASE}"),
                 hook_cancelled(command=phrase), thinking_stripped(scope=phrase))
    engine.upsert_contracts([user_rule("must", phrase), user_rule("must", phrase, "any"),
                             user_rule("must_not", PHRASE), user_rule("must_not", PHRASE, "any")])
    result = engine.audit_session(SID, provider="claude", path=str(path))
    assert result["coverage"]["complete"] is True, result["coverage"]
    assert {f["contract_id"]: f["verdict"] for f in result["findings"]} == {
        "user-must-message": "violation", "user-must-any": "violation",
        "user-must_not-message": "clean", "user-must_not-any": "clean"}


def test_a_queued_human_prompt_with_an_image_satisfies_and_violates_user_role_contracts(engine, root):
    path = write(main_file(root), human("first"), human_queued_image())
    engine.upsert_contracts([user_rule("must", "look at this"), user_rule("must_not", PHRASE)])
    result = engine.audit_session(SID, provider="claude", path=str(path))
    assert result["coverage"]["complete"] is True, result["coverage"]
    assert {f["contract_id"]: f["verdict"] for f in result["findings"]} == {
        "user-must-message": "clean", "user-must_not-message": "violation"}


def test_a_runtime_role_rule_sees_only_placeholders_of_injected_payloads(engine, root):
    path = write(main_file(root), human("go"), invoked_skills(), attachment("command_permissions", allowedTools=[RULE]),
                 advisor_result(), document_result(), session_owner())
    engine.upsert_contracts([{"id": "no-payload", "type": "must_not", "scope": "any",
                              "regex": "SKILL-BODY|SKILL-PATH|rule-sentinel|ENCRYPTED-ADVICE|PDF-DATA|5e000000",
                              "roles": ["runtime", "user", "assistant", "tool"], "provider": "claude"}])
    result = engine.audit_session(SID, provider="claude", path=str(path))
    assert result["coverage"]["complete"] is True and result["findings"][0]["verdict"] == "clean"


# --- the reported symptom: a satisfied prerequisite of a proposed action stays clean ------------------------------

def _session_with_every_new_shape(path):
    return write(path, human("ship it"), session_owner(), tools_used(), api_request(), api_request(base=...),
                 file_attachment(), attachment("ultra_effort_enter", reminderType="sparse"),
                 attachment("ultra_effort_exit"), advisor_tool(), attachment("command_permissions", allowedTools=[]),
                 invoked_skills(), agents_killed(), peer_user(), peer_queued(), notification_queued(),
                 human_queued_image(), server_tool_use(), advisor_result(), peer_user(origin=HANDBACK_PEER),
                 user("continue", isMeta=True, origin={"kind": "auto-continuation"}), hook_cancelled(),
                 attachment("plan_mode_exit", planExists=False, planFilePath=PLAN_PATH), thinking_stripped(),
                 attachment("advisor_stripped"), env("queue-operation", operation="popAll", content=POP_ALL_TEXT),
                 server_tool_use(id="srvtoolu_2"),
                 advisor_result(tool_use_id="srvtoolu_2", content={"type": "advisor_tool_result_error",
                                                                   "error_code": "overloaded"}),
                 assistant(use("toolu_pdf", "Read", file_path="/synthetic/a.pdf")), document_result(),
                 assistant(use("toolu_t", "Bash", command="npm test")),
                 tool_result("toolu_t", "ran", block_extra={"is_error": False}))


def test_a_satisfied_prerequisite_of_a_proposed_action_is_clean_next_to_every_new_shape(engine, root):
    path = _session_with_every_new_shape(main_file(root))
    engine.upsert_contracts([{"id": "tests-first", "provider": "claude", "type": "must", "scope": "tool_call",
                              "regex": "npm test", "before_regex": "deploy", "requires_success": True}])
    action = {"tool_name": "Bash", "arguments": {"command": "deploy now"}}
    result = engine.check_action(SID, action, provider="claude", path=str(path))
    assert result["coverage"]["complete"] is True, result["coverage"]
    assert result["coverage"]["unsupported_event_count"] == 0 and result["verdict"] == "clean"


def test_a_hosted_call_and_its_result_are_linked_evidence(engine, root):
    path = _session_with_every_new_shape(main_file(root))
    sync(engine, path)
    calls = {c.meta["call_id"]: c for c in stored(engine) if c.kind == "tool_call"}
    results = {c.meta["call_id"]: c for c in stored(engine) if c.kind == "tool_result"}
    assert calls["srvtoolu_1"].meta["evidence_origin"] == "hosted_tool"
    assert results["srvtoolu_1"].meta["success"] is True and results["srvtoolu_1"].meta["role"] == "tool"
    assert results["srvtoolu_2"].meta["success"] is False and results["srvtoolu_2"].meta["error_code"] == "overloaded"


@pytest.mark.parametrize("sub", [False, True], ids=["main", "subagent"])
def test_complete_coverage_end_to_end_for_main_and_subagent_files(engine, root, sub):
    records = [build(sub) for build, *_expected in ACCEPTED.values()]
    path = write(sub_file(root) if sub else main_file(root), *records)
    info = sync(engine, path, CHILD if sub else SID)
    assert info["ok"] and info["coverage"]["complete"] is True, info["coverage"]
    assert info["coverage"]["unsupported_event_count"] == 0 and info["coverage"]["parse_error_count"] == 0
    blob = dump(stored(engine, CHILD if sub else SID))
    for sentinel in SENTINELS:
        assert sentinel not in blob, sentinel


# --- versions and reparsing ----------------------------------------------------------------------------------------

def test_the_parser_bump_reindexes_a_session_parsed_by_v4(engine, root, monkeypatch):
    path = write(main_file(root), human("hello"), session_owner(), assistant(text("done")))
    with monkeypatch.context() as old:
        old.setattr(claude, "PARSER_VERSION", 4)
        old.delitem(claude._HANDLERS, "session-owner")
        first = sync(engine, path)
    assert first["parser_version"] == 4 and first["coverage"]["unsupported_event_count"] == 1
    assert first["coverage"]["complete"] is False
    second = sync(engine, path)
    assert second["reindexed"] and second["reindex_reason"] == "parser_updated" and second["parser_version"] == 5
    assert second["coverage"]["complete"] is True and second["coverage"]["unsupported_event_count"] == 0
    assert ACCOUNT not in dump(stored(engine))
