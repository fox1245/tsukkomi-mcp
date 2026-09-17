from __future__ import annotations

import json
from pathlib import Path

from self_directing_mcp.session_events import (
    as_text, content_hash, iter_raw_lines, make_chunk, parse_error, result_success,
)


def peek_session_id(path):
    from self_directing_mcp.grokbot.discover import session_id_from_path
    return session_id_from_path(Path(path))


def _classify_line(obj):
    role = str(obj.get("role") or "unknown")
    message = obj.get("message", obj.get("content"))
    content = message.get("content", message) if isinstance(message, dict) else message
    base = {"role": role}
    if "role" not in obj and "message" not in obj:
        return [("meta", f"[meta] {as_text(obj)}", {})]
    if role == "tool" and not isinstance(content, list):
        return [("tool_result", f"[tool_result] {as_text(content)}",
                 {**base, "call_id": obj.get("tool_call_id"), "success": result_success(content)})]
    if not isinstance(content, list):
        return [("turn", f"[{role}] {as_text(content)}", base)]
    parts = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(("turn", f"[{role}] {as_text(block)}", base))
            continue
        btype = block.get("type", "text")
        if btype == "tool_use":
            name = block.get("name", "unknown")
            args = block.get("input", block.get("arguments"))
            parts.append(("tool_call", f"[tool_call:{name}] {as_text(args)}",
                          {**base, "tool_name": name, "call_id": block.get("id"),
                           "block_type": btype}))
        elif btype == "tool_result":
            body = block.get("content", block.get("output", block.get("text")))
            parts.append(("tool_result", f"[tool_result] {as_text(body)}",
                          {**base, "role": "tool", "call_id": block.get("tool_use_id") or block.get("tool_call_id"),
                           "success": result_success(block), "block_type": btype}))
        else:
            parts.append(("turn", f"[{role}] {as_text(block.get('text', block))}",
                          {**base, **({"unsupported_event": True} if btype not in ("text", "thinking") else {})}))
    return parts or [("meta", f"[empty:{role}]", base)]


def parse_session_chunks(path, *, session_id=None, start_byte=0, known_hashes=None):
    sid = str(session_id or peek_session_id(path) or "unknown")
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
            timestamp = obj.get("timestamp") or obj.get("created_at")
            for sub, (kind, text, meta) in enumerate(_classify_line(obj)):
                chunks.append(make_chunk("grokbot", sid, kind, text, meta, line, start, end,
                                         sub, str(timestamp) if timestamp is not None else None))
        except (ValueError, TypeError) as exc:
            chunks.append(parse_error("grokbot", sid, line, start, end, type(exc).__name__))
    return chunks, offset, sid
