"""Antigravity transcript discovery; only official brain/session log layouts."""
from __future__ import annotations

import re
import json
from pathlib import Path

from self_directing_mcp.session_events import as_text, iter_raw_lines, make_chunk, parse_error

PARSER_VERSION = 5
_ID = re.compile(r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\Z")


class PathTraversalError(ValueError):
    pass


def valid_id(value: str) -> bool:
    return isinstance(value, str) and bool(_ID.fullmatch(value))


def peek_session_id(path: Path) -> str | None:
    parts = Path(path).parts
    if (len(parts) >= 5 and parts[-5] == "brain"
            and parts[-3:-1] == (".system_generated", "logs")
            and parts[-1] in ("transcript.jsonl", "transcript_full.jsonl")
            and valid_id(parts[-4])):
        return parts[-4].lower()
    return None


def resolve_session_path(roots: list[Path], *, session_id: str | None = None,
                         path: str | None = None, require_exists: bool = True) -> Path:
    if session_id is not None and not valid_id(session_id):
        raise ValueError("invalid AGY conversation UUID")
    sid = session_id.lower() if session_id else None
    roots = [Path(root).expanduser().resolve() for root in roots]
    if path is None:
        if sid is None:
            raise ValueError("AGY conversationId or transcriptPath is required")
        matches = []
        for root in roots:
            logs = root / "brain" / sid / ".system_generated" / "logs"
            # Current CLI hooks pass the full log. These are session mirrors,
            # not two sessions; explicit hook paths remain authoritative.
            for name in ("transcript_full.jsonl", "transcript.jsonl"):
                candidate = logs / name
                if candidate.is_file():
                    matches.append(resolve_session_path(roots, session_id=sid, path=str(candidate)))
                    break
        matches = list(dict.fromkeys(matches))
        if len(matches) != 1:
            raise FileNotFoundError("AGY transcript missing or ambiguous across configured app roots")
        return matches[0]
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise PathTraversalError("AGY transcript must be an absolute path without traversal")
    claimed = peek_session_id(candidate)
    if claimed is None or (sid is not None and claimed != sid):
        raise ValueError("AGY transcript layout or conversationId mismatch")
    # Check both the requested and resolved layouts: a symlink must not change the
    # conversation identity, escape an app root, or redirect the evidence file.
    resolved = candidate.resolve()
    if not any(candidate.is_relative_to(root) and resolved ==
               root / "brain" / candidate.parts[-4] / ".system_generated" / "logs" / candidate.name
               for root in roots):
        raise PathTraversalError("AGY transcript escapes its configured session root")
    if require_exists and not resolved.is_file():
        raise FileNotFoundError("AGY transcript not yet persisted")
    return resolved


def parse_session_chunks(path: Path, *, session_id=None, start_byte=0, known_hashes=None, receipts=None):
    """Read the host's display JSONL, not lifecycle hook payloads.

    CLI 1.2.9 emits JSON-encoded values in tool_calls[].args and omits call IDs.
    Missing IDs remain missing; step order alone is not proof of result linkage.
    """
    sid = peek_session_id(path)
    if sid is None or (session_id is not None and sid != session_id.lower()):
        raise ValueError("AGY transcript session identity mismatch")
    chunks, offset = [], start_byte
    for line, start, end, raw in iter_raw_lines(path, start_byte=start_byte):
        offset = end
        if not raw.strip():
            continue
        try:
            if any(0xDC80 <= ord(c) <= 0xDCFF for c in raw):
                raise ValueError("invalid UTF-8")
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise ValueError("transcript record is not an object")
            step, source, kind = record.get("step_index"), record.get("source"), record.get("type")
            if type(step) is not int or step < 0:
                raise ValueError("missing native step index")
            timestamp = record.get("created_at")
            if timestamp is not None and not isinstance(timestamp, str):
                raise ValueError("invalid timestamp")
            base = {"event_type": kind, "step_index": step, "status": record.get("status")}
            content = record.get("content", "")
            if not isinstance(content, str):
                raise ValueError("invalid transcript content")
            if kind == "USER_INPUT" and source == "USER_EXPLICIT":
                chunks.append(make_chunk("agy", sid, "turn", "[message:user] " + content,
                                         {**base, "role": "user"}, line, start, end, timestamp=timestamp))
            elif kind == "EPHEMERAL_MESSAGE" and source == "SYSTEM_SDK":
                chunks.append(make_chunk("agy", sid, "meta", "[ephemeral_context] " + content,
                                         {**base, "role": "runtime", "evidence_origin": "injected_context"},
                                         line, start, end, timestamp=timestamp))
            elif kind == "SYSTEM_MESSAGE" and source == "SYSTEM":
                # Observed host-injected Stop continuation is context, not a
                # user instruction, executed action, or successful evidence.
                chunks.append(make_chunk("agy", sid, "meta", "[host_system_context] " + content,
                                         {**base, "role": "runtime", "evidence_origin": "host_system_context"},
                                         line, start, end, timestamp=timestamp))
            elif kind == "PLANNER_RESPONSE" and source == "MODEL":
                chunks.append(make_chunk("agy", sid, "turn" if content else "meta",
                                         "[message:assistant] " + content,
                                         {**base, "role": "assistant" if content else "runtime"},
                                         line, start, end, timestamp=timestamp))
                calls = record.get("tool_calls", [])
                if not isinstance(calls, list):
                    raise ValueError("invalid tool_calls")
                for sub, call in enumerate(calls, 1):
                    if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not call["name"]:
                        raise ValueError("invalid tool call")
                    args = call.get("args")
                    if not isinstance(args, dict):
                        raise ValueError("invalid tool arguments")
                    if path.name == "transcript_full.jsonl":
                        decoded = args
                    else:
                        if any(not isinstance(value, str) for value in args.values()):
                            raise ValueError("invalid encoded tool arguments")
                        decoded = {key: json.loads(value) for key, value in args.items()}
                    # A planner proposal may subsequently be denied or cancelled.
                    # Preserve it for retrieval, never as observed execution.
                    chunks.append(make_chunk("agy", sid, "meta",
                                             f"[proposed_tool_call:{call['name']}] {as_text(decoded)}",
                                             {**base, "role": "runtime", "tool_name": call["name"],
                                              "call_id": None, "evidence_origin": "model_proposal"},
                                             line, start, end, sub=sub, timestamp=timestamp))
            elif kind == "GENERIC" and source == "MODEL":
                from self_directing_mcp.agy_receipts import linked_receipt
                observation = linked_receipt(receipts or {}, step, path, start,
                                             result_record=record, require_paired=False)
                if (observation and not observation.get("paired") and record.get("status") == "ERROR"
                        and isinstance(record.get("error"), str)
                        and record["error"].startswith("tool call denied by pre-tool hook: ")):
                    # Native host rejection plus the anchored Pre receipt proves
                    # non-execution. A denied call has no PostToolUse to pair.
                    chunks.append(make_chunk("agy", sid, "meta", "[tool_denied] " + record["error"],
                                             {**base, "role": "runtime", "receipt_id": observation["receipt_id"],
                                              "evidence_origin": "host_rejected_proposal"},
                                             line, start, end, timestamp=timestamp))
                    continue
                # Match only the host prefix, never exit-code prose in output.
                exit_match = re.match(
                    r"\ACreated At: [^\n]+\nCompleted At: [^\n]+\n\n"
                    r"The command exited with code (-?\d+)\.\n(?:Output|Stdout):\n", content)
                success = (int(exit_match.group(1)) == 0
                           if exit_match and record.get("status") == "DONE" else None)
                if record.get("status") in ("ERROR", "CANCELED"):
                    success = False
                receipt = observation if observation and observation.get("paired") else None
                call_id, name, provenance = None, None, {}
                if receipt:
                    action = json.loads(receipt["action"])
                    name = action["name"]
                    call_id = "agy-hook:" + receipt["receipt_id"]
                    provenance = {"receipt_id": receipt["receipt_id"],
                                  "evidence_origin": "host_hook_and_native_transcript",
                                  "receipt_step_index": step,
                                  "receipt_prefix_hash": receipt["prefix_hash"]}
                    chunks.append(make_chunk("agy", sid, "tool_call",
                                             f"[tool_call:{name}] {as_text(action['args'])}",
                                             {**base, **provenance, "role": "assistant",
                                              "tool_name": name, "call_id": call_id},
                                             line, start, end, timestamp=timestamp))
                    if receipt.get("post_error"):
                        success = False
                    elif name != "run_command":
                        # An empty hook error is tool transport success, not a
                        # proof that an arbitrary tool fulfilled its objective.
                        success = None
                chunks.append(make_chunk("agy", sid, "tool_result", "[tool_result] " + content,
                                         {**base, **provenance, "role": "tool", "call_id": call_id,
                                          "tool_name": name, "success": success,
                                          "linkage": "host_step_receipt" if receipt else "native_log_omits_call_id",
                                          "unsupported_event": not bool(exit_match) or receipt is None},
                                         line, start, end, sub=1 if receipt else 0, timestamp=timestamp))
            else:
                chunks.append(make_chunk("agy", sid, "meta", "[unsupported_agy_event] " + as_text(record),
                                         {**base, "role": "runtime", "unsupported_event": True},
                                         line, start, end, timestamp=timestamp))
        except (ValueError, TypeError) as exc:
            chunks.append(parse_error("agy", sid, line, start, end, type(exc).__name__))
    return chunks, offset, sid
