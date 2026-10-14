"""Local OMP JSONL evidence; never infer tool execution from a call alone."""
from __future__ import annotations

import json
import re
from pathlib import Path

from self_directing_mcp.session_events import as_text, iter_raw_lines, make_chunk, parse_error, result_success

# Bumping this triggers a lazy per-session reparse on the next sync/audit of each already-indexed
# session (index.ingest compares the recorded version); unchanged event ids, anchors and vectors survive.
# v7: host runtime notices (custom/custom_message/mode_change shapes), strict image blocks, steering
# user messages, session_init/model_usage/branch_summary records and redactedThinking blocks.
# v8: every agent-attributed user-role message is runtime text, partial steering metadata is rejected,
# and unsupported opaque blocks keep only their key names.
PARSER_VERSION = 8
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_EVENT_FIELDS = {"id", "parentId", "timestamp", "type"}
_ADMIN_FIELDS = {
    "title_change": _EVENT_FIELDS | {"title", "source", "previousTitle", "trigger"},
    "credential_pin": _EVENT_FIELDS | {"provider", "hash"},
    "service_tier_change": _EVENT_FIELDS | {"serviceTier"},
    "ttsr_injection": _EVENT_FIELDS | {"injectedRules"},
    "mode_change": _EVENT_FIELDS | {"mode", "data"},
    "compaction": _EVENT_FIELDS | {"summary", "shortSummary", "firstKeptEntryId",
                                   "tokensBefore", "tokensAfter", "method",
                                   "providerReplayThroughEntryId", "details",
                                   "fromExtension", "preserveData"},
}
_SERVICE_TIER_PROVIDERS = {"openai", "anthropic", "google"}
_SERVICE_TIER_VALUES = {"auto", "default", "flex", "scale", "priority", "ultrafast"}
_CUSTOM_MESSAGE_FIELDS = _EVENT_FIELDS | {"customType", "attribution", "display", "content", "details"}
_TODO_HUD_FIELDS = _EVENT_FIELDS | {"customType", "data"}
# Host notices: runtime metadata only, never user instructions or execution evidence. New notice
# kinds are deliberately absent so they surface as unsupported until reviewed.
_HOST_NOTICE_TYPES = {"launch-completion", "todo-error-reminder", "image-attachment", "tool-roster-notice",
                      "resolve-reminder", "xdev-mount-notice", "goal-continuation", "thinking-loop-redirect"}
_GOAL_COMPLETED_TYPES = {"objective": (str,), "timeUsedSeconds": (int, float), "tokensUsed": (int, float)}


def _custom_data_ok(custom_type, data):
    if custom_type == "user_todo_edit":
        return isinstance(data, dict) and set(data) == {"phases"} and isinstance(data["phases"], list)
    if custom_type == "system-prompt-digest":
        return isinstance(data, str)
    return (custom_type == "goal-completed" and isinstance(data, dict) and set(data) == set(_GOAL_COMPLETED_TYPES)
            and all(isinstance(data[key], kinds) and type(data[key]) is not bool
                    for key, kinds in _GOAL_COMPLETED_TYPES.items()))


_SESSION_INIT_FIELDS = _EVENT_FIELDS | {"agent", "task", "systemPrompt", "tools", "spawns", "resolvedModel", "readOnly",
                                        "modelRole", "outputSchema", "outputSchemaMode", "readSummarize"}
_SESSION_INIT_TYPES = {"agent": str, "task": str, "tools": list, "spawns": str, "resolvedModel": str,
                       "readOnly": bool, "modelRole": str, "outputSchema": dict, "outputSchemaMode": str,
                       "readSummarize": bool}
_MODEL_USAGE_TYPES = {"api": str, "model": str, "provider": str, "purpose": str, "stopReason": str, "usage": dict,
                      "role": str, "errorMessage": str}
_STEERING_FIELDS = {"role", "content", "timestamp", "attribution", "steering", "liveSteered"}
_OPAQUE_KINDS = {"image", "redactedThinking", "thinking"}


def _unsupported_text(kind, block):
    """Diagnostics for an unsupported block. Image, reasoning and other ``data`` payloads are never indexed."""
    if isinstance(block, dict) and (kind in _OPAQUE_KINDS or "data" in block):
        return "keys=" + ",".join(sorted(map(str, block)))
    return as_text(block)


def _typed_fields_ok(obj, allowed, required, types):
    return (set(obj) <= allowed and required <= set(obj)
            and all(type(obj[key]) is kind for key, kind in types.items() if key in obj))


def _session_init_ok(obj):
    return (_typed_fields_ok(obj, _SESSION_INIT_FIELDS, _SESSION_INIT_FIELDS - _EVENT_FIELDS
                             - {"modelRole", "outputSchema", "outputSchemaMode", "readSummarize"}, _SESSION_INIT_TYPES)
            and isinstance(obj["systemPrompt"], (str, list)))


def _model_usage_ok(obj):
    return _typed_fields_ok(obj, _EVENT_FIELDS | set(_MODEL_USAGE_TYPES),
                            set(_MODEL_USAGE_TYPES) - {"role", "errorMessage"}, _MODEL_USAGE_TYPES)


def _branch_summary_ok(obj):
    details = obj.get("details")
    return (set(obj) <= _EVENT_FIELDS | {"summary", "fromId", "details"}
            and isinstance(obj.get("summary"), str) and isinstance(obj.get("fromId"), str)
            and isinstance(details, dict) and set(details) == {"discardedEntryId", "kind"}
            and all(isinstance(value, str) for value in details.values()))


def _image_ok(block):
    """Strict image block; base64 data is validated but never indexed."""
    meta = block.get("_meta", {"codex/imageDetail": ""}) if isinstance(block, dict) else None
    return (isinstance(block, dict) and {"type", "data", "mimeType"} <= set(block) <= {"type", "data", "mimeType", "_meta"}
            and block["type"] == "image"
            and isinstance(meta, dict) and set(meta) == {"codex/imageDetail"}
            and isinstance(meta["codex/imageDetail"], (str, dict))
            and isinstance(block["data"], str) and isinstance(block["mimeType"], str)
            and block["mimeType"].startswith("image/") and len(block["mimeType"]) > 6)


_DEVELOPER_MESSAGE_FIELDS = {"role", "content", "timestamp", "attribution", "synthetic", "userInitiated"}


class PathTraversalError(ValueError):
    pass


def valid_id(session_id: str) -> bool:
    return isinstance(session_id, str) and bool(_ID.fullmatch(session_id)) and session_id not in (".", "..")


def peek_session_id(path: Path) -> str | None:
    # OMP may write a presentation title before the session header.
    for _line, _start, _end, raw in iter_raw_lines(path):
        if not raw.strip():
            continue
        try:
            header = json.loads(raw)
        except (UnicodeError, ValueError):
            return None
        if not isinstance(header, dict):
            return None
        if header.get("type") == "title":
            continue
        sid = header.get("id") if header.get("type") == "session" else None
        return sid if valid_id(sid) else None
    return None


def resolve_session_path(root: Path, *, session_id: str | None = None, path: str | None = None) -> Path:
    if session_id is not None and not valid_id(session_id):
        raise ValueError("invalid OMP session id")
    root = Path(root).expanduser().resolve()
    if path is None:
        if not session_id:
            raise ValueError("OMP session_id or path is required")
        # OMP may persist sessions in nested workspaces; match the header, not a filename guess.
        matches = []
        for candidate in root.rglob("*.jsonl") if root.is_dir() else ():
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                continue
            if peek_session_id(resolved) == session_id:
                matches.append(resolved)
        if len(matches) != 1:
            raise FileNotFoundError("OMP session not found or ambiguous under configured root")
        return matches[0]
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise PathTraversalError("OMP session path is outside configured root")
    if not resolved.is_file() or resolved.suffix.lower() != ".jsonl":
        raise FileNotFoundError("OMP session JSONL not found")
    claimed = peek_session_id(resolved)
    if claimed is None:
        raise ValueError("OMP session header missing or invalid")
    if session_id is not None and claimed != session_id:
        raise ValueError("OMP session header id does not match requested session")
    return resolved


def parse_session_chunks(path: Path, *, session_id=None, start_byte=0, known_hashes=None, prior_call=None, sink=None):
    sid = peek_session_id(path)
    if sid is None or (session_id is not None and session_id != sid):
        raise ValueError("OMP session header missing or id does not match requested session")
    chunks, offset, seen_header = ([] if sink is None else sink), start_byte, start_byte > 0
    seen_calls: set[tuple[str, str]] = set()
    for line, start, end, raw in iter_raw_lines(path, start_byte=start_byte):
        offset = end
        if not raw.strip():
            continue
        try:
            if any(0xDC80 <= ord(char) <= 0xDCFF for char in raw):
                raise ValueError("invalid UTF-8")
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError("JSONL record is not an object")
            if obj.get("type") == "session":
                if seen_header or obj.get("id") != sid:
                    raise ValueError("unexpected OMP session header")
                seen_header = True
                continue
            if not seen_header and obj.get("type") != "title":
                raise ValueError("OMP event before session header")
            if obj.get("type") != "message" or not isinstance(obj.get("message"), dict):
                record_type = obj.get("type")
                custom_type = obj.get("customType")
                # Agent notifications are runtime evidence, not assistant actions.
                # Keep their content searchable without treating it as a tool call.
                agent_message = (record_type == "custom_message"
                                 and set(obj) <= _CUSTOM_MESSAGE_FIELDS
                                 and ((custom_type in ("async-result", "irc:incoming",
                                                       "mid-run-todo-nudge", "goal-mode-context")
                                       and obj.get("attribution") == "agent")
                                      or (custom_type in _HOST_NOTICE_TYPES
                                          and isinstance(obj.get("attribution"), str)
                                          and type(obj.get("display")) is bool))
                                 and isinstance(obj.get("content"), str)
                                 and (obj.get("details") is None or isinstance(obj["details"], dict)))
                metadata_only = record_type in (
                    "title", "model_change", "thinking_level_change",
                ) or (record_type == "title_change" and set(obj) <= _ADMIN_FIELDS["title_change"]
                      and isinstance(obj.get("title"), str)) or (
                    record_type == "credential_pin" and set(obj) <= _ADMIN_FIELDS["credential_pin"]
                    and isinstance(obj.get("hash"), str) and isinstance(obj.get("provider"), str)
                ) or (record_type == "service_tier_change"
                      and set(obj) <= _ADMIN_FIELDS["service_tier_change"]
                      and "serviceTier" in obj
                      and (obj["serviceTier"] is None or (
                          isinstance(obj["serviceTier"], dict)
                          and set(obj["serviceTier"]) <= _SERVICE_TIER_PROVIDERS
                          and all(isinstance(tier, str) and tier in _SERVICE_TIER_VALUES
                                  for tier in obj["serviceTier"].values())
                      ))
                ) or (record_type == "mode_change" and set(obj) <= _ADMIN_FIELDS["mode_change"]
                      and isinstance(obj.get("mode"), str)
                      and (obj.get("data") is None or (isinstance(obj["data"], dict)
                                                       and set(obj["data"]) <= {"goal"}))) or (
                    record_type == "compaction" and set(obj) <= _ADMIN_FIELDS["compaction"]
                    and isinstance(obj.get("summary"), str)
                    and isinstance(obj.get("firstKeptEntryId"), str)
                ) or (record_type == "custom" and custom_type == "session_exit") or (
                    record_type == "custom" and custom_type == "todo_hud_state"
                    and set(obj) <= _TODO_HUD_FIELDS and isinstance(obj.get("data"), dict)
                    and set(obj["data"]) <= {"sourceEntryId", "fingerprint", "visibility"}
                ) or (record_type == "session_init" and _session_init_ok(obj)
                ) or (record_type == "model_usage" and _model_usage_ok(obj)
                ) or (record_type == "branch_summary" and _branch_summary_ok(obj)
                ) or (record_type == "custom" and set(obj) == _TODO_HUD_FIELDS
                      and _custom_data_ok(custom_type, obj.get("data"))) or agent_message
                if record_type == "ttsr_injection":
                    metadata_only = (set(obj) <= _ADMIN_FIELDS["ttsr_injection"]
                                     and isinstance(obj.get("injectedRules"), list)
                                     and all(isinstance(rule, str) and rule for rule in obj["injectedRules"]))
                if record_type == "custom" and custom_type == "tool_execution_start":
                    data = obj.get("data")
                    if isinstance(data, dict):
                        call = (data.get("toolCallId"), data.get("toolName"))
                        metadata_only = call in seen_calls or (
                            prior_call is not None and all(isinstance(part, str) and part for part in call)
                            and prior_call(*call, start))
                text = (f"[omp_event:{record_type}:{custom_type}] {obj['content']}"
                        if agent_message else f"[omp_event:{record_type}]")
                if record_type == "session_init" and metadata_only:
                    # Only the parent's instruction is searchable; prompt, tools and schema are never indexed.
                    text = f"[omp_event:session_init] [subagent_task] {obj['task']}"
                meta = {"unsupported_event": not metadata_only, "event_type": record_type, "role": "runtime"}
                if metadata_only and record_type in ("session_init", "model_usage", "branch_summary"):
                    meta["evidence_origin"] = {"session_init": "subagent_init"}.get(record_type, record_type)
                elif metadata_only and (custom_type in _HOST_NOTICE_TYPES or record_type == "mode_change"
                                      or (record_type == "custom" and custom_type in (
                                          "user_todo_edit", "system-prompt-digest", "goal-completed"))):
                    meta["evidence_origin"] = f"host_notice:{custom_type or record_type}"
                chunks.append(make_chunk("omp", sid, "meta", text, meta, line, start, end))
                continue
            msg = obj["message"]
            role = msg.get("role")
            content = msg.get("content")
            timestamp = obj.get("timestamp") or msg.get("timestamp")
            timestamp = str(timestamp) if timestamp is not None else None
            if role == "user" and "attribution" in msg and msg["attribution"] not in ("agent", "user"):
                raise ValueError("unsupported OMP user attribution")
            # Any steering metadata takes the strict path, so a partial record is rejected, not trusted.
            steering = role == "user" and ("steering" in msg or "liveSteered" in msg)
            if steering:
                # Mid-run steering: only a user-attributed message is a user turn; agent-attributed is host text.
                if (not set(msg) <= _STEERING_FIELDS or msg.get("steering") is not True
                        or msg.get("attribution") not in ("agent", "user")
                        or ("liveSteered" in msg and msg["liveSteered"] is not True
                            and not isinstance(msg["liveSteered"], str))):
                    raise ValueError("unsupported OMP steering message shape")
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
            if role not in ("user", "assistant", "toolResult", "developer") or not isinstance(content, list):
                raise ValueError("unsupported OMP message shape")
            base = {"event_type": "message", "role": "tool" if role == "toolResult" else role}
            if role == "developer":
                if (not set(msg) <= _DEVELOPER_MESSAGE_FIELDS
                        or (msg.get("attribution") is not None and not isinstance(msg["attribution"], str))
                        or any(flag in msg and type(msg[flag]) is not bool
                               for flag in ("synthetic", "userInitiated"))
                        or any(not isinstance(block, dict) or not set(block) <= {"type", "text"}
                               or block.get("type") != "text" or not isinstance(block.get("text"), str)
                               for block in content)):
                    raise ValueError("unsupported OMP developer message shape")
                base.update(role="runtime", evidence_origin="developer", attribution=msg.get("attribution"))
            # A user-role message the host attributes to an agent (subagent tasks, agent steering) is never
            # user authority, with or without steering metadata.
            agent_said = role == "user" and msg.get("attribution") == "agent"
            if agent_said:
                base.update(role="runtime", evidence_origin="agent_steering" if steering else "agent_message")
            if role == "toolResult":
                call_id = msg.get("toolCallId")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("tool result lacks toolCallId")
                name = msg.get("toolName")
                if name is not None and not isinstance(name, str):
                    raise ValueError("invalid toolName")
                text_blocks = [block for block in content if isinstance(block, dict) and block.get("type") == "text"]
                images = [block for block in content if _image_ok(block)]
                # Image placeholders only; success is judged from the text blocks alone, as before images.
                text = " ".join([as_text(block.get("text")) for block in text_blocks]
                                + [f"[image:{block['mimeType']}]" for block in images])
                unsupported = ("details" in msg and not isinstance(msg["details"], dict)) or any(not _image_ok(block) and (not isinstance(block, dict) or block.get("type") != "text" or not isinstance(block.get("text"), str)) for block in content)
                success = not msg["isError"] if type(msg.get("isError")) is bool else result_success(text_blocks)
                chunks.append(make_chunk("omp", sid, "tool_result", f"[tool_result:{name or ''}] {text}",
                                         {**base, "call_id": call_id, "tool_name": name, "success": success,
                                          "unsupported_event": unsupported}, line, start, end, timestamp=timestamp))
                continue
            if not content:
                chunks.append(make_chunk("omp", sid, "meta", f"[message:{role}]",
                                         {**base, "role": "runtime"}, line, start, end, timestamp=timestamp))
            for sub, block in enumerate(content):
                if not isinstance(block, dict):
                    block = {"type": "unknown", "value": block}
                kind = block.get("type")
                meta = dict(base)
                if kind == "text" and isinstance(block.get("text"), str):
                    event_kind = "meta" if role == "developer" or agent_said else "turn"
                    label = ("steering:agent" if steering else "message:agent") if agent_said else f"message:{role}"
                    text = f"[{label}] {block['text']}"
                elif kind == "image" and role == "user" and _image_ok(block):
                    # Attachments are not instructions: keep as runtime meta, never base64.
                    meta.update(role="runtime", evidence_origin=base.get("evidence_origin", "image"))
                    event_kind, text = "meta", f"[image:{block['mimeType']}]"
                elif kind == "toolCall" and role == "assistant" and isinstance(block.get("id"), str) and block["id"] and isinstance(block.get("name"), str) and block["name"]:
                    name = block["name"]
                    meta.update(tool_name=name, call_id=block["id"])
                    event_kind, text = "tool_call", f"[tool_call:{name}] {as_text(block.get('arguments'))}"
                    seen_calls.add((block["id"], name))
                elif kind == "thinking" and isinstance(block.get("thinking"), str):
                    # Retain ordering but do not treat private reasoning as an assistant action.
                    meta.update(role="runtime", evidence_origin="thinking")
                    event_kind, text = "meta", "[thinking]"
                elif (kind == "redactedThinking" and role == "assistant" and set(block) == {"type", "data"}
                      and isinstance(block["data"], str)):
                    meta.update(role="runtime", evidence_origin="thinking")
                    event_kind, text = "meta", "[redacted_thinking]"
                else:
                    meta["unsupported_event"] = True
                    event_kind, text = "meta", f"[unsupported_content:{kind}] {_unsupported_text(kind, block)}"
                chunks.append(make_chunk("omp", sid, event_kind, text, meta, line, start, end,
                                         sub=sub, timestamp=timestamp))
        except (ValueError, TypeError) as exc:
            chunks.append(parse_error("omp", sid, line, start, end, type(exc).__name__))
    return chunks, offset, sid
