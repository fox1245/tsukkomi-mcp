from __future__ import annotations

import json
from pathlib import Path

from self_directing_mcp.session_events import (
    as_text, iter_raw_lines, make_chunk, parse_error, result_success,
)

PARSER_VERSION = 2
_WEB_TERMINAL = {"completed": True, "failed": False, "cancelled": False, "incomplete": False}
_WEB_PENDING = {"in_progress", "searching", "queued"}


def peek_session_id(path: Path) -> str | None:
    try:
        with Path(path).open(encoding="utf-8", errors="replace") as stream:
            for i, line in enumerate(stream):
                if i > 50:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("type") == "session_meta":
                    payload = obj.get("payload")
                    if isinstance(payload, dict) and isinstance(payload.get("id"), str):
                        return payload["id"]
    except OSError:
        pass
    return None


def _classify_and_text(obj):
    etype = obj.get("type", "unknown")
    payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
    meta = {"event_type": etype}
    if etype in ("world_state", "token_usage_record", "compacted"):
        if not isinstance(obj.get("payload"), dict):
            raise ValueError("runtime event payload must be an object")
        meta.update(role="runtime", evidence_origin="runtime_metadata")
        if etype == "world_state":
            meta["state_full"] = payload.get("full")
        elif etype == "token_usage_record":
            meta["response_id"] = payload.get("response_id")
            meta["turn_id"] = payload.get("turn_id")
        else:
            # A context replacement is a historical projection, not new user
            # instructions or proof that its quoted tool calls were executed.
            history = payload.get("replacement_history")
            if history is not None and not isinstance(history, list):
                raise ValueError("replacement_history must be an array")
            meta.update(evidence_origin="compaction", authoritative=False,
                        replacement_history_count=len(history or []))
            for key in ("compaction_response_id", "first_window_id", "previous_window_id", "window_id", "window_number"):
                if key in payload:
                    meta[key] = payload[key]
        # Keep the complete payload and the previous text/cache identity.
        return "meta", f"[{etype}] {as_text(payload)}", meta
    if etype in ("session_meta", "config_snapshot", "turn_context"):
        kind = "policy" if etype == "config_snapshot" else "meta"
        return kind, f"[{etype}] {as_text(payload or obj)}", meta
    if etype == "event_msg":
        role = payload.get("role") or payload.get("type", "event")
        role = {"agent_message": "assistant", "user_message": "user"}.get(role, role)
        meta["role"] = role
        return "turn", f"[event_msg:{role}] {as_text(payload.get('message') or payload.get('text') or payload)}", meta
    if etype in ("response_item", "input_item"):
        item = payload or obj
        item_type = str(item.get("type") or item.get("item_type") or "unknown")
        meta["item_type"] = item_type
        meta["call_id"] = item.get("call_id") or item.get("tool_call_id") or item.get("tool_use_id")
        if item_type == "web_search_call" and etype == "response_item":
            action = item.get("action")
            status = item.get("status")
            meta.update(role="assistant", tool_name="web_search", call_id=item.get("id"),
                        evidence_origin="hosted_tool", status=status)
            if not isinstance(action, dict) or not isinstance(action.get("type"), str):
                meta["unsupported_event"] = True
                meta["unsupported_reason"] = "web search action schema"
            if not isinstance(status, str) or status not in (_WEB_TERMINAL.keys() | _WEB_PENDING):
                meta["unsupported_event"] = True
                meta["unsupported_reason"] = "web search status schema"
            return "tool_call", f"[tool_call:web_search] {as_text(action)}", meta
        if item_type in ("function_call_output", "custom_tool_call_output", "tool_result", "tool_output"):
            body = item.get("output", item.get("content", item.get("text", "")))
            meta.update(role="tool", success=result_success(item))
            return "tool_result", f"[tool_result] {as_text(body)}", meta
        if item_type in ("function_call", "custom_tool_call", "tool_call", "shell_call", "command") or "function_call" in item:
            call = item.get("function_call") if isinstance(item.get("function_call"), dict) else item
            name = call.get("name") or call.get("tool_name") or item_type
            args = call.get("arguments", call.get("input", call.get("command", call.get("content"))))
            meta.update(role="assistant", tool_name=name)
            return "tool_call", f"[tool_call:{name}] {as_text(args)}", meta
        content = item.get("content", item.get("text", item.get("output", item)))
        meta["role"] = item.get("role", "assistant")
        if item_type not in ("message", "input", "reasoning"):
            meta["unsupported_event"] = True
        return "turn", f"[{etype}:{item_type}:{meta['role']}] {as_text(content)}", meta
    meta["unsupported_event"] = True
    return "meta", f"[{etype}] {as_text(payload or obj)}", meta


def parse_session_chunks(path: Path, *, session_id=None, start_byte=0, known_hashes=None):
    """Retain every occurrence; known_hashes is accepted for legacy callers only."""
    from self_directing_mcp.codex.discover import session_id_from_filename

    sid = (session_id or peek_session_id(path) or session_id_from_filename(Path(path)) or "unknown").lower()
    chunks, offset = [], start_byte
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
            if obj.get("type") == "session_meta":
                payload = obj.get("payload") or {}
                claimed = payload.get("id") if isinstance(payload, dict) else None
                if claimed and str(claimed).lower() != sid:
                    raise ValueError("session id does not match requested session")
            kind, text, meta = _classify_and_text(obj)
            timestamp = obj.get("timestamp") or obj.get("created_at") or obj.get("time")
            chunks.append(make_chunk("codex", sid, kind, text, meta, line, start, end,
                                     timestamp=str(timestamp) if timestamp is not None else None))
            if meta.get("evidence_origin") == "hosted_tool" and not meta.get("unsupported_event"):
                status = meta["status"]
                if status in _WEB_TERMINAL:
                    # Codex reports native-tool completion in the call record.
                    # Both projections point to the same original byte range.
                    completion = {"id": meta["call_id"], "status": status}
                    chunks.append(make_chunk(
                        "codex", sid, "tool_result", f"[tool_result:web_search] {as_text(completion)}",
                        {**meta, "role": "tool", "success": _WEB_TERMINAL[status],
                         "evidence_origin": "hosted_tool_status"}, line, start, end, sub=1,
                        timestamp=str(timestamp) if timestamp is not None else None))
        except (ValueError, TypeError) as exc:
            chunks.append(parse_error("codex", sid, line, start, end, type(exc).__name__))
    return chunks, offset, sid

