"""Synthetic Grok CLI records and payloads shared by the Grok tests (the recorded shapes of ``fixtures/grok``)."""
from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "grok"
SID = "01a00000-0000-7000-8000-000000000001"
OTHER = "01a00000-0000-7000-8000-000000000002"
GROUP = "%2Fwork%2Fproj"
T0 = 1792100000


def rec(update, *, method="session/update", sid=SID, ts=T0, prompt="p1", envelope=None):
    params = {"sessionId": sid, "update": update}
    meta = {"promptId": prompt, **(envelope or {})} if method == "session/update" and prompt else (envelope or {})
    if meta:
        params["_meta"] = meta
    return {"timestamp": ts, "method": method, "params": params}


def chunk(kind, text, **kw):
    return rec({"sessionUpdate": kind, "content": {"type": "text", "text": text}}, **kw)


def user(text, index=0, **kw):
    return rec({"sessionUpdate": "user_message_chunk", "content": {"type": "text", "text": text},
                "_meta": {"modelId": "grok-4.7", "promptIndex": index}}, prompt=None, **kw)


def agent(text, **kw):
    return chunk("agent_message_chunk", text, **kw)


def thought(text, **kw):
    return chunk("agent_thought_chunk", text, **kw)


def call(call_id, title, raw_input=None, **kw):
    return rec({"sessionUpdate": "tool_call", "toolCallId": call_id, "title": title,
                "rawInput": {} if raw_input is None else raw_input,
                "_meta": {"x.ai/tool": {"version": 1, "name": title, "namespace": "grok_build"}}}, **kw)


def update(call_id, status, text=None, *, raw_output=None, extra=None, content=None, **kw):
    body = {"sessionUpdate": "tool_call_update", "toolCallId": call_id}
    if status is not None:
        body["status"] = status
    if text is not None:
        body["content"] = [{"type": "content", "content": {"type": "text", "text": text}}]
    if content is not None:
        body["content"] = content
    if raw_output is not None:
        body["rawOutput"] = raw_output
    body.update(extra or {})
    return rec(body, **kw)


def bash_output(code=0, text="ok\n"):
    return {"type": "Bash", "output": list(text.encode()), "output_for_prompt": f"exit: {code}\n{text}",
            "exit_code": code, "truncated": False, "signal": None, "timed_out": False}


def completed(prompt="p1", stop="end_turn", **kw):
    return rec({"sessionUpdate": "turn_completed", "prompt_id": prompt, "stop_reason": stop,
                "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15, "modelCalls": 1, "numTurns": 1,
                          "modelUsage": {"m": {"inputTokens": 10}}}, "elapsed_ms": 1},
               method="_x.ai/session/update", prompt=None, **kw)


def hook_execution(event_name="pre_tool_use", **kw):
    return rec({"sessionUpdate": "hook_execution", "event_name": event_name, "tool_name": "read_file",
                "runs": [{"name": "global/hook", "status": {"status": "success", "elapsed_ms": 4}}]},
               method="_x.ai/session/update", prompt=None, **kw)


def annotation(message, **kw):
    return rec({"sessionUpdate": "hook_annotation", "message": message, "kind": "tool_outcome"},
               method="_x.ai/session/update", prompt=None, **kw)


def session_file(root, sid=SID, group=GROUP):
    return root / group / sid / "updates.jsonl"


def write(path, *records, mode="w", raw=b""):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if raw:
        with path.open("ab") as stream:
            stream.write(raw)
    return path




# ---------------------------------------------------------------- hook payloads (a superset of Claude Code's)

NATIVE_EVENT = {"SessionStart": "session_start", "UserPromptSubmit": "user_prompt_submit", "PreToolUse": "pre_tool_use",
                "PostToolUse": "post_tool_use", "PostToolUseFailure": "post_tool_use_failure", "Stop": "stop"}


def hook_payload(event, path=None, *, tool="run_terminal_command", tool_input=None, sid=SID, **extra):
    """The stdin of a Grok command hook: camelCase and snake_case twins, as recorded in ``hook_payloads_session*.jsonl``."""
    base = {"hookEventName": NATIVE_EVENT[event], "sessionId": sid, "cwd": "/work/proj", "workspaceRoot": "/work/proj",
            "timestamp": "2026-01-01T00:00:00+00:00", "permissionMode": "default", "hook_event_name": event,
            "session_id": sid, "permission_mode": "default"}
    if event in ("PreToolUse", "PostToolUse", "PostToolUseFailure", "Stop") and path is not None:
        base.update(transcriptPath=str(path), transcript_path=str(path))
    if event in ("PreToolUse", "PostToolUse", "PostToolUseFailure"):
        arguments = tool_input if tool_input is not None else {"command": "echo ok", "description": "say ok"}
        base.update(toolName=tool, toolUseId="call-0000-0", toolInput=arguments, toolInputTruncated=False,
                    tool_name=tool, tool_input=arguments, tool_use_id="call-0000-0")
    if event == "PostToolUse":
        result = {"type": "Bash", "output_for_prompt": "exit: 0\nok\n", "exit_code": 0}
        base.update(toolResult=result, tool_response=result, toolResultTruncated=False, durationMs=3, duration_ms=3,
                    isBackgrounded=False)
    if event == "PostToolUseFailure":
        base.update(error="boom")
    if event == "UserPromptSubmit":
        base.update(prompt="continue please", promptId="00000000-0000-4000-8000-0000000000b1")
    if event == "SessionStart":
        base["source"] = "new"
    if event == "Stop":
        base.update(reason="end_turn", stopHookActive=False, lastAssistantMessage="done", backgroundTasks=[],
                    sessionCrons=[], promptId="00000000-0000-4000-8000-0000000000b1")
    base.update(extra)
    return base
