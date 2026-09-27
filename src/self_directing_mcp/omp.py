"""Local OMP JSONL evidence; never infer tool execution from a call alone."""
from __future__ import annotations

import json
import re
from pathlib import Path

from self_directing_mcp.session_events import as_text, iter_raw_lines, make_chunk, parse_error, result_success

PARSER_VERSION = 3
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_EVENT_FIELDS = {"id", "parentId", "timestamp", "type"}
_ADMIN_FIELDS = {
    "title_change": _EVENT_FIELDS | {"title", "source", "previousTitle", "trigger"},
    "credential_pin": _EVENT_FIELDS | {"provider", "hash"},
    "mode_change": _EVENT_FIELDS | {"mode", "data"},
    "compaction": _EVENT_FIELDS | {"summary", "shortSummary", "firstKeptEntryId",
                                   "tokensBefore", "tokensAfter", "method",
                                   "providerReplayThroughEntryId", "details",
                                   "fromExtension", "preserveData"},
}
_CUSTOM_MESSAGE_FIELDS = _EVENT_FIELDS | {"customType", "attribution", "display", "content", "details"}
_TODO_HUD_FIELDS = _EVENT_FIELDS | {"customType", "data"}


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


def parse_session_chunks(path: Path, *, session_id=None, start_byte=0, known_hashes=None, prior_call=None):
    sid = peek_session_id(path)
    if sid is None or (session_id is not None and session_id != sid):
        raise ValueError("OMP session header missing or id does not match requested session")
    chunks, offset, seen_header = [], start_byte, start_byte > 0
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
                                 and custom_type in ("async-result", "irc:incoming",
                                                     "mid-run-todo-nudge", "goal-mode-context")
                                 and obj.get("attribution") == "agent"
                                 and isinstance(obj.get("content"), str)
                                 and (obj.get("details") is None or isinstance(obj["details"], dict)))
                metadata_only = record_type in (
                    "title", "model_change", "thinking_level_change",
                ) or (record_type == "title_change" and set(obj) <= _ADMIN_FIELDS["title_change"]
                      and isinstance(obj.get("title"), str)) or (
                    record_type == "credential_pin" and set(obj) <= _ADMIN_FIELDS["credential_pin"]
                    and isinstance(obj.get("hash"), str) and isinstance(obj.get("provider"), str)
                ) or (record_type == "mode_change" and set(obj) <= _ADMIN_FIELDS["mode_change"]
                      and isinstance(obj.get("mode"), str) and isinstance(obj.get("data"), dict)
                      and set(obj["data"]) <= {"goal"}) or (
                    record_type == "compaction" and set(obj) <= _ADMIN_FIELDS["compaction"]
                    and isinstance(obj.get("summary"), str)
                    and isinstance(obj.get("firstKeptEntryId"), str)
                ) or (record_type == "custom" and custom_type == "session_exit") or (
                    record_type == "custom" and custom_type == "todo_hud_state"
                    and set(obj) <= _TODO_HUD_FIELDS and isinstance(obj.get("data"), dict)
                    and set(obj["data"]) <= {"sourceEntryId", "fingerprint", "visibility"}
                ) or agent_message
                if record_type == "custom" and custom_type == "tool_execution_start":
                    data = obj.get("data")
                    if isinstance(data, dict):
                        call = (data.get("toolCallId"), data.get("toolName"))
                        metadata_only = call in seen_calls or (
                            prior_call is not None and all(isinstance(part, str) and part for part in call)
                            and prior_call(*call, start))
                text = (f"[omp_event:{record_type}:{custom_type}] {obj['content']}"
                        if agent_message else f"[omp_event:{record_type}]")
                chunks.append(make_chunk("omp", sid, "meta", text,
                                         {"unsupported_event": not metadata_only, "event_type": record_type,
                                          "role": "runtime"}, line, start, end))
                continue
            msg = obj["message"]
            role = msg.get("role")
            content = msg.get("content")
            timestamp = obj.get("timestamp") or msg.get("timestamp")
            timestamp = str(timestamp) if timestamp is not None else None
            if role not in ("user", "assistant", "toolResult") or not isinstance(content, list):
                raise ValueError("unsupported OMP message shape")
            base = {"event_type": "message", "role": "tool" if role == "toolResult" else role}
            if role == "toolResult":
                call_id = msg.get("toolCallId")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("tool result lacks toolCallId")
                name = msg.get("toolName")
                if name is not None and not isinstance(name, str):
                    raise ValueError("invalid toolName")
                text = " ".join(as_text(block.get("text")) for block in content if isinstance(block, dict) and block.get("type") == "text")
                unsupported = any(not isinstance(block, dict) or block.get("type") != "text" or not isinstance(block.get("text"), str) for block in content)
                success = not msg["isError"] if type(msg.get("isError")) is bool else result_success(content)
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
                    event_kind, text = "turn", f"[message:{role}] {block['text']}"
                elif kind == "toolCall" and role == "assistant" and isinstance(block.get("id"), str) and block["id"] and isinstance(block.get("name"), str) and block["name"]:
                    name = block["name"]
                    meta.update(tool_name=name, call_id=block["id"])
                    event_kind, text = "tool_call", f"[tool_call:{name}] {as_text(block.get('arguments'))}"
                    seen_calls.add((block["id"], name))
                elif kind == "thinking" and isinstance(block.get("thinking"), str):
                    # Retain ordering but do not treat private reasoning as an assistant action.
                    meta.update(role="runtime", evidence_origin="thinking")
                    event_kind, text = "meta", "[thinking]"
                else:
                    meta["unsupported_event"] = True
                    event_kind, text = "meta", f"[unsupported_content:{kind}] {as_text(block)}"
                chunks.append(make_chunk("omp", sid, event_kind, text, meta, line, start, end,
                                         sub=sub, timestamp=timestamp))
        except (ValueError, TypeError) as exc:
            chunks.append(parse_error("omp", sid, line, start, end, type(exc).__name__))
    return chunks, offset, sid
