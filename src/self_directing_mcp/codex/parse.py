from __future__ import annotations

import io
import json
from pathlib import Path

from self_directing_mcp.compressed_source import CompressedSourceError, is_compressed, open_source
from self_directing_mcp.tool_payload import tool_call_text
from self_directing_mcp.session_events import (
    as_text, has_invalid_utf8, iter_raw_lines, make_chunk, parse_error, result_success,
)

# Bumping this triggers a lazy per-session reparse on the next sync/audit of each already-indexed
# session (index.ingest compares the recorded version); unchanged event ids, anchors and vectors survive.
# v3: multi-agent messaging, tool-search discovery, inter-agent metadata and new-style session headers.
PARSER_VERSION = 3
_WEB_TERMINAL = {"completed": True, "failed": False, "cancelled": False, "incomplete": False}
_WEB_PENDING = {"in_progress", "searching", "queued"}


def _header_id(stream):
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
    return None


def read_session_id(path: Path) -> str | None:
    """``session_meta.payload.id`` from the first lines of a rollout. A ``.zst`` rollout is decoded as a stream
    (only its first lines); an unreadable one raises ``CompressedSourceError``."""
    if is_compressed(path):
        with io.TextIOWrapper(open_source(path, max_bytes=None), encoding="utf-8", errors="replace") as stream:
            return _header_id(stream)
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        return _header_id(stream)


def peek_session_id(path: Path) -> str | None:
    try:
        return read_session_id(path)
    except (OSError, CompressedSourceError):
        return None


_TOP_FIELDS = {"timestamp", "type", "payload", "ordinal", "metadata"}
_PASSTHROUGH = "internal_chat_message_metadata_passthrough"
_HEADER_STR = ("id", "session_id", "timestamp", "cwd", "originator", "cli_version", "model_provider", "history_mode",
               "thread_source", "agent_nickname", "agent_path", "multi_agent_version", "parent_thread_id",
               "forked_from_id", "creator_account_id", "creator_user_id")
_HEADER_KINDS = {**dict.fromkeys(_HEADER_STR, str), "base_instructions": dict, "context_window": dict, "git": dict,
                 "history_base": dict, "dynamic_tools": list, "runtime_workspace_roots": list}
_HEADER_REQUIRED = ("id", "session_id", "timestamp", "cwd", "originator", "cli_version", "model_provider",
                    "base_instructions", "context_window", "history_mode")
# Indexed header fields. System prompt, tool schemas, context window, git/history data, account ids and
# workspace roots are validated for shape but never indexed.
_HEADER_INDEXED = ("id", "session_id", "timestamp", "cwd", "originator", "cli_version", "model_provider",
                   "history_mode", "thread_source", "agent_nickname", "agent_path", "multi_agent_version",
                   "parent_thread_id", "forked_from_id", "subagent_history_start_ordinal")


def _plain_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _header_new_style(payload):
    return bool({"session_id", "base_instructions", "context_window", "history_mode"} & set(payload))


def _header_ok(payload):
    """Type-check known header keys; new-style headers must also carry the always-present set."""
    if not isinstance(payload, dict):
        return False
    if _header_new_style(payload) and not set(_HEADER_REQUIRED) <= set(payload):
        return False
    if any(type(payload[key]) is not kind for key, kind in _HEADER_KINDS.items() if key in payload):
        return False
    if "source" in payload and not isinstance(payload["source"], (str, dict)):
        return False
    return "subagent_history_start_ordinal" not in payload or _plain_int(payload["subagent_history_start_ordinal"])


def _top_ok(obj):
    """Top-level envelope for the newly supported shapes; client-authored records have unclear semantics."""
    meta = obj.get("metadata", {})
    return (set(obj) <= _TOP_FIELDS and ("ordinal" not in obj or _plain_int(obj["ordinal"]))
            and isinstance(meta, dict) and set(meta) <= {"client_authored", "user_input_order"}
            and meta.get("client_authored", False) is False
            and (meta.get("user_input_order") is None or _plain_int(meta["user_input_order"])))


def _passthrough_ok(item):
    value = item.get(_PASSTHROUGH, {})
    return (isinstance(value, dict) and set(value) <= {"create_time", "turn_id"}
            and all(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in value.values()))


def _optional_str(item, *keys):
    return all(isinstance(item[key], str) for key in keys if key in item)


def _agent_message(obj, item):
    """Inter-agent text is runtime evidence: never user authority and never tool evidence."""
    content = item.get("content")
    ok = (_top_ok(obj) and set(item) <= {"type", "id", "author", "recipient", "content", _PASSTHROUGH}
          and _optional_str(item, "id") and isinstance(item.get("author"), str) and isinstance(item.get("recipient"), str)
          and _passthrough_ok(item) and isinstance(content, list)
          and all(isinstance(block, dict) and (
              (set(block) == {"type", "text"} and block["type"] == "input_text" and isinstance(block["text"], str))
              or (set(block) == {"type", "encrypted_content"} and block["type"] == "encrypted_content"
                  and isinstance(block["encrypted_content"], str))) for block in content))
    if not ok:
        return None
    # Encrypted blocks keep a placeholder only.
    body = " ".join(block["text"] if block["type"] == "input_text" else "[encrypted_content]" for block in content)
    return f"[agent_message:{item['author']}->{item['recipient']}] {body}"


def _tool_search(obj, item, item_type):
    """Tool discovery, not an action on the user's system; tool lists are never indexed."""
    args = item.get("arguments")
    common = (_top_ok(obj) and _optional_str(item, "id", "status", "execution") and isinstance(item.get("call_id"), str)
              and _passthrough_ok(item))
    fields = {"type", "id", "call_id", "status", "execution", _PASSTHROUGH}
    if item_type == "tool_search_call":
        if not (common and set(item) <= fields | {"arguments"} and isinstance(args, dict)
                and set(args) <= {"query", "limit"} and isinstance(args.get("query"), str)
                and ("limit" not in args or _plain_int(args["limit"]))):
            return None
        return f"[tool_search_call] {args['query']}"
    if not (common and set(item) <= fields | {"tools"} and isinstance(item.get("tools"), list)):
        return None
    return "[tool_search_output] [tools omitted]"


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
    if etype == "inter_agent_communication_metadata":
        meta.update(role="runtime", evidence_origin="inter_agent_metadata")
        if not (_top_ok(obj) and isinstance(obj.get("payload"), dict) and set(payload) == {"trigger_turn"}
                and isinstance(payload["trigger_turn"], bool)):
            meta["unsupported_event"] = True
        return "meta", "[inter_agent_communication_metadata]", meta
    if etype == "session_meta":
        meta.update(role="runtime", evidence_origin="session_meta")
        if not _header_ok(obj.get("payload")):
            meta["unsupported_event"] = True
        safe = {key: payload[key] for key in _HEADER_INDEXED if key in payload}
        if isinstance(payload.get("source"), str):
            safe["source"] = payload["source"]
        return "meta", f"[session_meta] {as_text(safe)}" if safe else "[session_meta]", meta
    if etype in ("config_snapshot", "turn_context"):
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
        if etype == "response_item" and item_type in ("agent_message", "tool_search_call", "tool_search_output"):
            origin = "agent_message" if item_type == "agent_message" else "tool_search"
            meta.update(role="runtime", evidence_origin=origin)
            text = (_agent_message(obj, item) if item_type == "agent_message" and isinstance(obj.get("payload"), dict)
                    else _tool_search(obj, item, item_type) if isinstance(obj.get("payload"), dict) else None)
            if text is None:
                meta["unsupported_event"] = True
                text = f"[response_item:{item_type}] [unsupported shape]"
            return "meta", text, meta
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
            return "tool_call", tool_call_text("codex", name, args), meta
        content = item.get("content", item.get("text", item.get("output", item)))
        meta["role"] = item.get("role", "assistant")
        if item_type not in ("message", "input", "reasoning"):
            meta["unsupported_event"] = True
        return "turn", f"[{etype}:{item_type}:{meta['role']}] {as_text(content)}", meta
    meta["unsupported_event"] = True
    return "meta", f"[{etype}] {as_text(payload or obj)}", meta


def parse_session_chunks(path: Path, *, session_id=None, start_byte=0, known_hashes=None, sink=None):
    """Retain every occurrence; known_hashes is accepted for legacy callers only."""
    from self_directing_mcp.codex.discover import session_id_from_filename

    sid = (session_id or peek_session_id(path) or session_id_from_filename(Path(path)) or "unknown").lower()
    chunks, offset = ([] if sink is None else sink), start_byte
    for line, start, end, raw in iter_raw_lines(path, start_byte=start_byte):
        offset = end
        if not raw.strip():
            continue
        try:
            if has_invalid_utf8(raw):
                raise ValueError("invalid UTF-8")
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError("JSONL record is not an object")
            foreign = False
            if obj.get("type") == "session_meta":
                payload = obj.get("payload") or {}
                claimed = payload.get("id") if isinstance(payload, dict) else None
                foreign = bool(claimed and str(claimed).lower() != sid)
                # Only a complete new-style header after the first record may belong to another thread
                # (e.g. inherited parent history); a first-line or legacy mismatch is still an error.
                if foreign and not (start > 0 and isinstance(payload, dict) and _header_new_style(payload)
                                    and _header_ok(payload)):
                    raise ValueError("session id does not match requested session")
            kind, text, meta = _classify_and_text(obj)
            if obj.get("type") == "session_meta" and foreign:
                meta["evidence_origin"] = "inherited_session_meta"
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

