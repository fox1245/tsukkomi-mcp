"""Claude Code session transcripts (``~/.claude/projects``). Authority comes from observed origin only.

Rules that every shape below follows:

* Only a record the host stamped ``origin.kind == "human"`` (or, in a main file, a legacy prompt that predates
  origin stamps) is a user turn. Task notifications, peer/coordinator messages, injected meta messages,
  compaction summaries, attachments, hook output, tool results and a subagent's task prompt are runtime
  evidence, never user authority, whatever role the host renders them with.
* Shapes are validated strictly. An unknown record type, attachment type, content block or origin kind stays
  ``unsupported_event`` so coverage reports it; unknown never becomes clean.
* Opaque or runtime-only payloads (thinking, signatures, base64 images, hook stdout/stderr/command, file
  snapshots, session context, system and queue payloads, titles, last-prompt copies) are validated for shape
  and replaced by placeholders. Only small identifiers are kept.
* A subagent transcript is its own session whose id embeds the parent's, so the two never share evidence.
"""
from __future__ import annotations

import os
import json
import re
from pathlib import Path

from self_directing_mcp.codex import discover as _codex_discover
from self_directing_mcp.session_events import as_text, iter_raw_lines, make_chunk, parse_error, result_success

# Bumping this triggers a lazy per-session reparse on the next sync/audit of each already-indexed session.
# v1: initial Claude Code provider (main and subagent transcripts, origin-based user authority).
PARSER_VERSION = 1

SUBAGENTS = "subagents"
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}\Z")
_HOOK_NAME = re.compile(r"[A-Za-z0-9_.:*/@+-]{1,200}\Z")


class PathTraversalError(ValueError):
    pass


# ----------------------------------------------------------------------------- identity and discovery
#
# Session ids come from the transcript's location, never from record contents:
#   <project>/<uuid>.jsonl                          -> "<uuid>"
#   <project>/<uuid>/subagents/<stem>.jsonl         -> "<uuid>_<stem>"   (child of session <uuid>)
# Every record of a subagent file carries the PARENT's ``sessionId`` and ``agentId``, so neither is unique per
# file; the composite id keeps parent and child in separate (provider, session) index partitions, keeps
# duplicate agent ids under different parents apart, and still matches the workflow id alphabet.

def session_parts(session_id) -> tuple[str, str | None] | None:
    """``(session uuid, subagent stem or None)`` for a canonical Claude Code session id, else None."""
    if not isinstance(session_id, str):
        return None
    if _UUID.fullmatch(session_id):
        return session_id, None
    if len(session_id) > 37 and session_id[36] == "_" and _UUID.fullmatch(session_id[:36]) \
            and _STEM.fullmatch(session_id[37:]):
        return session_id[:36], session_id[37:]
    return None


def valid_id(session_id) -> bool:
    return session_parts(session_id) is not None


def _id_from_parts(parts) -> str | None:
    """Session id for the trailing components of a transcript path (layout only; no I/O)."""
    if not parts or not parts[-1].endswith(".jsonl"):
        return None
    stem = parts[-1][:-len(".jsonl")]
    if len(parts) >= 3 and parts[-2] == SUBAGENTS and _UUID.fullmatch(parts[-3]) and _STEM.fullmatch(stem):
        return f"{parts[-3]}_{stem}"
    return stem if _UUID.fullmatch(stem) else None


def peek_session_id(path: Path) -> str | None:
    return _id_from_parts(Path(path).parts)


def _layout_id(relative_parts) -> str | None:
    """Session id for a path relative to the projects root; only the two official layouts qualify."""
    if len(relative_parts) == 2:
        return _id_from_parts(relative_parts) if _UUID.fullmatch(relative_parts[1][:-len(".jsonl")]) else None
    if len(relative_parts) == 4 and relative_parts[2] == SUBAGENTS:
        return _id_from_parts(relative_parts[1:])
    return None


def _linked(path: Path) -> bool:
    return path.is_symlink() or os.path.isjunction(path)


def _find_by_id(root: Path, session_id: str) -> list[Path]:
    parent, stem = session_parts(session_id)
    found: list[Path] = []
    try:
        projects = sorted(entry for entry in root.iterdir() if entry.is_dir() and not _linked(entry))
    except OSError:
        return found
    for project in projects:
        candidate = project / (f"{session_id}.jsonl" if stem is None else f"{parent}/{SUBAGENTS}/{stem}.jsonl")
        if _linked(candidate) or not candidate.is_file():
            continue
        try:
            found.append(resolve_session_path(root, session_id=session_id, path=str(candidate)))
        except (ValueError, OSError):
            continue
    return list(dict.fromkeys(found))


def resolve_session_path(root: Path, *, session_id: str | None = None, path: str | None = None,
                         require_exists: bool = True) -> Path:
    """Resolve a transcript below ``root`` by id or explicit path.

    The path must be absolute, free of ``..``, inside the configured root, in an official layout, and no
    component below the root may be a symlink or junction (a link could redirect the evidence file or change
    which session it claims). The root itself may be a link.
    """
    if session_id is not None and not valid_id(session_id):
        raise ValueError("invalid Claude Code session id")
    configured = Path(os.path.abspath(Path(root).expanduser()))
    root = Path(root).expanduser().resolve()
    if path is None:
        if not session_id:
            raise ValueError("Claude Code session_id or path is required")
        matches = _find_by_id(root, session_id)
        if len(matches) != 1:
            raise FileNotFoundError("Claude Code session not found or ambiguous under configured root")
        return matches[0]
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise PathTraversalError("Claude Code transcript must be an absolute path without traversal")
    for prefix in dict.fromkeys((root, configured)):
        if candidate.is_relative_to(prefix):
            below = candidate.relative_to(prefix).parts
            break
    else:
        raise PathTraversalError("Claude Code transcript is outside the configured projects root")
    current = prefix
    for part in below:
        current = current / part
        if _linked(current):
            raise PathTraversalError("Claude Code transcript path may not contain symlinks")
    resolved = root.joinpath(*below)
    try:
        _codex_discover.ensure_under_root(resolved, root)
    except _codex_discover.PathTraversalError as exc:
        raise PathTraversalError(str(exc)) from exc
    claimed = _layout_id(below)
    if claimed is None:
        raise ValueError("not an official Claude Code transcript location")
    if session_id is not None and claimed != session_id:
        raise ValueError("Claude Code session id does not match the transcript location")
    if require_exists and not resolved.is_file():
        raise FileNotFoundError("Claude Code transcript not found")
    return resolved


def discover_session_files(root: Path, *, include_subagents: bool = True) -> list[Path]:
    """Transcripts in official layouts below ``root``. Symlinked files and directories are never followed."""
    root = Path(root).expanduser().resolve()

    def entries(directory):
        try:
            with os.scandir(directory) as scan:
                return sorted(scan, key=lambda entry: entry.name)
        except OSError:
            return []

    def plain(entry, *, directory):
        try:
            return (entry.is_dir(follow_symlinks=False) if directory else entry.is_file(follow_symlinks=False)) \
                and not os.path.isjunction(entry.path)
        except OSError:
            return False

    found: list[Path] = []
    for project in entries(root):
        if not plain(project, directory=True):
            continue
        for entry in entries(project.path):
            if plain(entry, directory=False) and entry.name.endswith(".jsonl") \
                    and _UUID.fullmatch(entry.name[:-len(".jsonl")]):
                found.append(Path(entry.path))
            elif include_subagents and plain(entry, directory=True) and _UUID.fullmatch(entry.name):
                nested = next((e for e in entries(entry.path) if e.name == SUBAGENTS and plain(e, directory=True)), None)
                for child in entries(nested.path) if nested is not None else ():
                    if plain(child, directory=False) and child.name.endswith(".jsonl") \
                            and _STEM.fullmatch(child.name[:-len(".jsonl")]):
                        found.append(Path(child.path))
    return sorted(found)


# ----------------------------------------------------------------------------- shape helpers

def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _label(value) -> str | None:
    return value if isinstance(value, str) and _LABEL.fullmatch(value) else None


def _keys(value) -> str:
    """Key names only, for diagnostics about shapes whose payload must not be indexed."""
    if not isinstance(value, dict):
        return "keys=-"
    names = sorted(str(key) if _LABEL.fullmatch(str(key)) else "?" for key in value)
    return "keys=" + ",".join(names[:30])


def _flag(record, key) -> bool:
    value = record.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _opt_str(container, *keys) -> None:
    for key in keys:
        if key in container and not isinstance(container[key], str):
            raise ValueError(f"{key} must be a string")


class _Event:
    """One chunk to emit: kind, text, meta. ``sub`` is assigned in emission order."""
    __slots__ = ("kind", "text", "meta")

    def __init__(self, kind, text, meta):
        self.kind, self.text, self.meta = kind, text, meta


def _runtime(base, text, origin, **extra):
    return _Event("meta", text, {**base, "role": "runtime", "evidence_origin": origin, **extra})


def _unsupported(base, text, **extra):
    return _Event("meta", text, {**base, "role": "runtime", "unsupported_event": True, **extra})


class _File:
    """Per-parse state. A subagent file's records carry the parent's session id and one agent id."""

    def __init__(self, sid):
        self.sid = sid
        self.parent, self.stem = session_parts(sid)
        self.sub = self.stem is not None
        stem = self.stem or ""
        self.agent_id = stem[len("agent-"):] if stem.startswith("agent-") else stem
        self.pinned_agent: str | None = None


def _envelope(record, state):
    """Validated metadata common to every record; raises ValueError on identity or type violations."""
    _opt_str(record, "sessionId")
    claimed = record.get("sessionId")
    expected = state.parent if state.sub else state.sid
    inherited = claimed is not None and claimed != expected
    if inherited and not state.sub:
        raise ValueError("record session id does not match the transcript's session")
    _opt_str(record, "uuid", "timestamp", "cwd", "requestId", "promptId", "agentId", "version")
    if "parentUuid" in record and record["parentUuid"] is not None and not isinstance(record["parentUuid"], str):
        raise ValueError("parentUuid must be a string or null")
    flag = record.get("isSidechain")
    if flag is not None and not isinstance(flag, bool):
        raise ValueError("isSidechain must be a boolean")
    if state.sub and flag is False:
        raise ValueError("record contradicts the subagent transcript it is in")
    agent = record.get("agentId")
    if inherited and agent is None:
        # A subagent file may replay another session's history, but only under one constant agent id.
        raise ValueError("inherited record carries no agent id")
    if state.sub and agent is not None:
        if state.pinned_agent is not None and agent != state.pinned_agent:
            raise ValueError("agent id changed inside one transcript")
        state.pinned_agent = agent
    meta = {"event_type": record.get("type"), "sidechain": state.sub or flag is True}
    for key, name in (("uuid", "claude_uuid"), ("parentUuid", "parent_uuid"), ("promptId", "prompt_id"),
                      ("requestId", "request_id"), ("cwd", "cwd"), ("version", "claude_version")):
        if record.get(key) is not None:
            meta[name] = record[key]
    if state.sub:
        meta.update(parent_session_id=state.parent, agent_id=agent or state.pinned_agent or state.agent_id)
        if claimed is not None:
            meta["record_session_id"] = claimed
        if inherited:
            meta["inherited"] = True  # replayed history: authority is still decided by origin and flags alone
    elif agent is not None:
        meta["agent_id"] = agent
    return meta, (flag is True and not state.sub)


# ----------------------------------------------------------------------------- origin and authority

_NON_HUMAN = {"task-notification": "task_notification", "peer": "peer_message", "coordinator": "coordinator"}
_ORIGIN_KEYS = {"human": set(), "coordinator": set(), "peer": {"body", "from", "handback", "name", "senderTaskId"},
                "task-notification": {"producer", "runId"}}


def _origin(value):
    """``(status, kind, meta)``; status is "none", "ok" or "bad". Unknown or malformed origins are never trusted."""
    if value is None:
        return "none", None, {}
    kind = value.get("kind") if isinstance(value, dict) else None
    if not isinstance(kind, str) or kind not in _ORIGIN_KEYS or not set(value) <= {"kind"} | _ORIGIN_KEYS[kind]:
        return "bad", None, {}
    extra = {}
    for key in _ORIGIN_KEYS[kind] - {"body", "handback"}:
        if key in value:
            if not isinstance(value[key], str):
                return "bad", None, {}
            extra[{"from": "peer_from", "name": "peer_name", "senderTaskId": "peer_task_id",
                   "producer": "producer", "runId": "run_id"}[key]] = value[key]
    if "body" in value and not isinstance(value["body"], str):
        return "bad", None, {}
    if "handback" in value and not isinstance(value["handback"], (bool, str, dict)):
        return "bad", None, {}
    return "ok", kind, extra


def _classify(*, origin, compact, transcript_only, is_meta, sidechain, has_tool_result):
    """Authority of one text block: ``(kind, label, meta)``; label None means a placeholder with no text."""
    unsupported = origin == "human" and (compact or is_meta)
    if compact:
        meta = {"role": "runtime", "evidence_origin": "compaction", "authoritative": False}
        return "meta", "compact_summary", meta, unsupported
    if transcript_only:
        return "meta", None, {"role": "runtime", "evidence_origin": "transcript_only"}, False
    if is_meta:
        name = _NON_HUMAN.get(origin, "injected_meta")
        return "meta", name, {"role": "runtime", "evidence_origin": name}, unsupported
    if origin in _NON_HUMAN:
        name = _NON_HUMAN[origin]
        return "meta", name, {"role": "runtime", "evidence_origin": name}, False
    if has_tool_result:
        # A text block riding with tool results is host-injected, never typed by the human.
        return "meta", "tool_message_text", {"role": "runtime", "evidence_origin": "tool_message_text"}, \
            origin == "human"
    if origin == "human":
        return "turn", "message:user", {"role": "user"}, False
    if sidechain:
        # A subagent's prompt without origin is the parent agent's task, not the human.
        return "meta", "parent_task", {"role": "runtime", "evidence_origin": "parent_task"}, False
    return "turn", "message:user", {"role": "user", "attribution": "inferred_legacy"}, False


def _text_block_ok(block) -> bool:
    return (isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
            and set(block) <= {"type", "text", "citations", "cache_control"})


def _image_placeholder(block) -> str | None:
    """Placeholder for a strictly valid image block; the payload is validated and never indexed."""
    if not (isinstance(block, dict) and block.get("type") == "image" and set(block) <= {"type", "source", "cache_control"}
            and isinstance(block.get("source"), dict)):
        return None
    source = block["source"]
    if source.get("type") == "base64" and set(source) == {"type", "media_type", "data"} \
            and isinstance(source["data"], str) and isinstance(source["media_type"], str) \
            and source["media_type"].startswith("image/") and len(source["media_type"]) > 6:
        return f"[image:{source['media_type']}]"
    if source.get("type") == "url" and set(source) == {"type", "url"} and isinstance(source["url"], str):
        return "[image:url]"
    return None


def _unsupported_block(base, kind, block, **extra):
    name = _label(kind) or "?"
    return _unsupported(base, f"[unsupported_content:{name}] {_keys(block)}", **extra)


def _permission(value):
    """Short scalars only: a hook adapter may need the decision, never arbitrary payload."""
    if isinstance(value, str):
        return value if len(value) <= 64 else None
    if isinstance(value, dict):
        kept = {key: item for key, item in value.items()
                if isinstance(key, str) and re.fullmatch(r"[A-Za-z0-9_]{1,40}", key)
                and (isinstance(item, (bool, int)) or (isinstance(item, str) and len(item) <= 64))}
        return kept or None
    return None


def _tool_result(base, block, record):
    """A tool result block inside a user message. It is runtime evidence linked by tool_use_id."""
    ok = (isinstance(block, dict) and set(block) <= {"type", "tool_use_id", "content", "is_error", "cache_control"}
          and isinstance(block.get("tool_use_id"), str) and block["tool_use_id"]
          and ("is_error" not in block or isinstance(block["is_error"], bool)))
    content = block.get("content") if isinstance(block, dict) else None
    if not ok or not (content is None or isinstance(content, (str, list))):
        raise ValueError("invalid tool_result block")
    parts, text_blocks, unsupported = [], [], False
    if isinstance(content, str):
        parts.append(content)
    for item in content if isinstance(content, list) else ():
        if _text_block_ok(item):
            parts.append(item["text"])
            text_blocks.append(item)
        elif (placeholder := _image_placeholder(item)) is not None:
            parts.append(placeholder)
        elif isinstance(item, dict) and item.get("type") == "tool_reference" and set(item) <= {
                "type", "tool_name", "cache_control"} and isinstance(item.get("tool_name"), str):
            parts.append(f"[tool_reference:{item['tool_name']}]")
        else:
            unsupported = True
            parts.append(f"[unsupported_content:{_label(item.get('type') if isinstance(item, dict) else None) or '?'}]")
    if "is_error" in block:
        success = not block["is_error"]
    else:
        success = result_success(content if isinstance(content, str) else text_blocks)
    meta = {**base, "role": "tool", "call_id": block["tool_use_id"], "tool_name": None, "success": success}
    if unsupported:
        meta["unsupported_event"] = True
    decision = _permission(record.get("permissionDecision"))
    if decision is not None:
        meta["permission_decision"] = decision
    if isinstance(record.get("sourceToolAssistantUUID"), str):
        meta["source_tool_assistant_uuid"] = record["sourceToolAssistantUUID"]
    return _Event("tool_result", "[tool_result] " + " ".join(parts), meta)


def _user_blocks(base, content, *, origin, compact, transcript_only, is_meta, sidechain, record, override=None):
    """Events for the content of a user-role message, or for a queued prompt (``override`` fixes authority).

    A queued prompt (``override``) can never carry tool results: those only come from the host in user messages.
    """
    if isinstance(content, str):
        content = [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        raise ValueError("invalid message content")
    if not content:
        yield _runtime(base, "[message:user]", "empty_message")
        return
    has_tool_result = override is None and any(
        isinstance(block, dict) and block.get("type") == "tool_result" for block in content)
    for block in content:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "tool_result" and override is None:
            yield _tool_result(base, block, record)
        elif _text_block_ok(block):
            if override is not None:
                event_kind, label, meta, unsupported = override
            else:
                event_kind, label, meta, unsupported = _classify(
                    origin=origin, compact=compact, transcript_only=transcript_only, is_meta=is_meta,
                    sidechain=sidechain, has_tool_result=has_tool_result)
            meta = {**base, **meta, **({"unsupported_event": True} if unsupported else {})}
            if event_kind == "turn" and not block["text"]:
                yield _runtime(base, "[message:user]", "empty_message")
            else:
                yield _Event(event_kind, f"[{label}] {block['text']}" if label else "[transcript_only_message]", meta)
        elif (placeholder := _image_placeholder(block)) is not None:
            yield _runtime(base, placeholder, "image")
        else:
            yield _unsupported_block(base, kind, block)


def _user(base, record, state):
    message = record.get("message")
    if not isinstance(message, dict) or message.get("role") != "user":
        raise ValueError("invalid user message")
    flags = dict(compact=_flag(record, "isCompactSummary"), transcript_only=_flag(record, "isVisibleInTranscriptOnly"),
                 is_meta=_flag(record, "isMeta"))
    status, origin, extra = _origin(record.get("origin"))
    if status == "bad":
        yield _unsupported(base, "[unsupported_user_origin]")
        return
    base = {**base, **extra, **({"origin_kind": origin} if origin else {})}
    yield from _user_blocks(base, message.get("content"), origin=origin, sidechain=state.sub, record=record, **flags)


def _assistant(base, record, state):
    message = record.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("invalid assistant message")
    _opt_str(message, "id", "model")
    content = message.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        raise ValueError("invalid message content")
    api_error = _flag(record, "isApiErrorMessage")
    synthetic = message.get("model") == "<synthetic>"
    base = {**base, **{name: message[key] for key, name in (("id", "message_id"), ("model", "model"))
                       if key in message}}
    if not content:
        yield _runtime(base, "[message:assistant]", "empty_message")
        return
    for block in content:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text" and _text_block_ok(block):
            if api_error or synthetic:
                origin = "api_error" if api_error else "synthetic_assistant"
                yield _runtime(base, f"[{origin}] {block['text']}", origin)
            else:
                yield _Event("turn", f"[message:assistant] {block['text']}", {**base, "role": "assistant"})
        elif kind == "thinking" and set(block) <= {"type", "thinking", "signature"} and isinstance(
                block.get("thinking"), str) and isinstance(block.get("signature", ""), str):
            yield _runtime(base, "[thinking]", "thinking")
        elif kind == "redacted_thinking" and set(block) == {"type", "data"} and isinstance(block["data"], str):
            yield _runtime(base, "[redacted_thinking]", "thinking")
        elif kind == "tool_use" and set(block) <= {"type", "id", "name", "input", "caller", "cache_control"} \
                and isinstance(block.get("id"), str) and block["id"] and isinstance(block.get("name"), str) \
                and block["name"] and isinstance(block.get("input"), dict):
            if api_error or synthetic:
                yield _unsupported(base, "[unsupported_content:tool_use] keys=-")
            else:
                yield _Event("tool_call", f"[tool_call:{block['name']}] {as_text(block['input'])}",
                             {**base, "role": "assistant", "tool_name": block["name"], "call_id": block["id"]})
        else:
            yield _unsupported_block(base, kind, block)


# ----------------------------------------------------------------------------- attachments

def _typed(fields, **types):
    """Every present field has its declared type (``bool`` never passes as ``int``)."""
    for key, kind in types.items():
        if key in fields:
            value = fields[key]
            if isinstance(value, bool) and bool not in (kind if isinstance(kind, tuple) else (kind,)):
                return False
            if not isinstance(value, kind):
                return False
    return True


def _names(value):
    return value if isinstance(value, list) and all(isinstance(item, str) for item in value) else None


def _shape(a, required, optional=None):
    """Exactly the known keys (``type`` aside), required ones present, every value of its declared type."""
    optional = optional or {}
    return (set(a) <= set(required) | set(optional) | {"type"} and set(required) <= set(a)
            and _typed(a, **required, **optional))


def _one_line(value: str) -> str:
    return re.sub(r"[\x00-\x1f\x7f]", " ", value)


def _hook_ref(a):
    """Event and name of a hook; the name is indexed only when it is a plain identifier (never a command line)."""
    event = a["hookEvent"] if _HOOK_NAME.fullmatch(a["hookEvent"]) else None
    name = a["hookName"] if _HOOK_NAME.fullmatch(a["hookName"]) else None
    return f" event={event or '-'} name={name or '-'}", {"hook_event": event, "hook_name": name}


def _hook_success(a):
    if not (_typed(a, hookEvent=str, hookName=str, exitCode=int, durationMs=(int, float), command=str, content=str,
                   stdout=str, stderr=str) and isinstance(a.get("hookEvent"), str) and isinstance(a.get("hookName"), str)):
        return None
    text, meta = _hook_ref(a)
    return f"{text} exit={a['exitCode'] if 'exitCode' in a else '-'}", {**meta, "exit_code": a.get("exitCode")}


def _hook_injection(content_type):
    """Hook-injected context/messages: runtime only. The payload may carry other tools' output; never indexed."""
    def check(a):
        if not (_shape(a, {"content": content_type, "hookName": str, "toolUseID": str, "hookEvent": str})
                and a["toolUseID"]):
            return None
        text, meta = _hook_ref(a)
        return text, {**meta, "tool_use_id": a["toolUseID"]}
    return check


def _file_reference(with_content):
    """A file attached to the context: its path is evidence of what was shown, its content never is."""
    required = {"filename": str, "displayPath": str, **({"content": dict} if with_content else {})}

    def check(a):
        if not _shape(a, required):
            return None
        if with_content and not (set(a["content"]) == {"file", "type"} and isinstance(a["content"]["type"], str)
                                 and isinstance(a["content"]["file"], dict)):
            return None
        return (f" display_path={_one_line(a['displayPath'])} filename={_one_line(a['filename'])}",
                {"filename": a["filename"], "display_path": a["displayPath"]})
    return check


def _thinking_drop(a):
    required = {"requestId": str, "thinkingBlocksSent": int, "thinkingTurnsSent": int, "newlyDropped": dict,
                "blockHashes": list, "firstReportForThreadInProcess": bool, "clientChange": dict, "model": str}
    if not _shape(a, required, {"querySource": str}):
        return None
    return "", {"thinking_blocks_sent": a["thinkingBlocksSent"], "thinking_turns_sent": a["thinkingTurnsSent"],
                "model": _label(a["model"]), "query_source": _label(a.get("querySource"))}


def _remote_session_change(a):
    required = {"url": (str, type(None)), "commit": str, "pr": str, "sendUserFileHint": bool,
                "managedCommit": bool, "managedPr": bool}
    if not _shape(a, required):
        return None
    return "", {"send_user_file_hint": a["sendUserFileHint"], "managed_commit": a["managedCommit"],
                "managed_pr": a["managedPr"]}


def _plan_mode(a):
    """Plan-mode reminder: the type and whether a plan exists are indexed; the plan file is never read."""
    required = {"isSubAgent": bool, "planExists": bool, "planFilePath": str, "reminderType": str}
    if not _shape(a, required):
        return None
    kind = _label(a["reminderType"])
    return (f" reminder_type={kind or '-'} plan_exists={'true' if a['planExists'] else 'false'}",
            {"reminder_type": kind, "plan_exists": a["planExists"], "is_sub_agent": a["isSubAgent"],
             "plan_file_path": a["planFilePath"]})


def _read_truncation_notice(a):
    """The Read result linked by ``toolUseID`` was cut. The link is kept for the audit layer; the banner is not."""
    if not (_shape(a, {"banner": str, "toolUseID": str}) and a["toolUseID"]):
        return None
    return f" truncated_result_for={_one_line(a['toolUseID'])}", {"truncated_result_for": a["toolUseID"]}


def _edited_text_file(a):
    if not (_typed(a, filename=str, snippet=str) and isinstance(a.get("filename"), str)):
        return None
    return f" {a['filename']}", {}


def _mcp_instructions_delta(a):
    if not _typed(a, addedBlocks=list) or (added := _names(a.get("addedNames", []))) is None \
            or (removed := _names(a.get("removedNames", []))) is None:
        return None
    return f" added={','.join(added)} removed={','.join(removed)}", {}


def _task_reminder(a):
    return ("", {"item_count": a["itemCount"]} if "itemCount" in a else {}) if _typed(a, itemCount=int) else None


def _date(a):
    return (f" {a['date']}", {}) if _typed(a, date=str) and isinstance(a.get("date"), str) else None


def _plain(**types):
    return lambda a: ("", {}) if _typed(a, **types) else None


# Known attachment types. Payloads are validated where their shape is known and never indexed.
_ATTACHMENTS = {
    "hook_success": _hook_success,
    "total_tokens_reminder": _plain(text=str),
    "silent_turn_reminder": _plain(text=str),
    "edited_text_file": _edited_text_file,
    "task_reminder": _task_reminder,
    "mcp_instructions_delta": _mcp_instructions_delta,
    "model": _plain(text=str),
    "date": _date,
    "deferred_tools_record": _plain(entries=list),
    "file": _file_reference(True),
    "compact_file_reference": _file_reference(False),
    "hook_additional_context": _hook_injection(list),
    "hook_system_message": _hook_injection(str),
    "thinking_drop": _thinking_drop,
    "remote_session_change": _remote_session_change,
    "read_truncation_notice": _read_truncation_notice,
    "plan_mode": _plan_mode,
    **{name: _plain() for name in ("prompt_snapshot", "deferred_tools_delta", "environment", "session_context",
                                    "instructions", "credential_org", "agent_listing_delta", "auto_mode",
                                    "skill_listing", "task_status")},
}
_QUEUED_KEYS = {"type", "commandMode", "delivery_id", "origin", "prompt", "runId", "source_uuid", "timestamp", "usage",
                "humanTurn", "isMeta", "reminderId"}
_QUEUED_TYPES = {"delivery_id": str, "runId": str, "source_uuid": str, "timestamp": str, "isMeta": bool,
                 "reminderId": (str, int), "humanTurn": (bool, str, int), "usage": (dict, int, float, str, type(None))}


def _queued_command(base, a, state):
    """A prompt queued mid-turn. Only a human prompt that is not meta is a user turn; all else is runtime."""
    status, origin, extra = _origin(a.get("origin"))
    mode, prompt = a.get("commandMode"), a.get("prompt")
    if (status == "bad" or not set(a) <= _QUEUED_KEYS or mode not in (None, "prompt", "task-notification")
            or not isinstance(prompt, (str, list)) or not _typed(a, **_QUEUED_TYPES)):
        yield _unsupported(base, "[unsupported_attachment:queued_command]", attachment_type="queued_command")
        return
    is_meta = a.get("isMeta", False)
    base = {**base, **extra, "attachment_type": "queued_command", "queued_command": True}
    for key, name in (("delivery_id", "delivery_id"), ("runId", "run_id"), ("reminderId", "reminder_id")):
        if key in a:
            base[name] = a[key]
    if isinstance(a.get("humanTurn"), (bool, int)) or (isinstance(a.get("humanTurn"), str) and len(a["humanTurn"]) <= 64):
        base["human_turn"] = a["humanTurn"]
    if "isMeta" in a:
        base["is_meta"] = is_meta
    if mode is not None:
        base["command_mode"] = mode
    if origin:
        base["origin_kind"] = origin
    name = _NON_HUMAN.get(origin) or ("injected_meta" if is_meta else "queued_command")
    if origin == "human" and (mode != "prompt" or is_meta):
        # A human origin without the prompt mode, or on a meta record, is a contradiction: never a user turn.
        yield _unsupported(base, "[unsupported_attachment:queued_command]", evidence_origin=name)
        return
    if origin == "human":
        override = ("turn", "message:user", {"role": "user"}, False)
    else:
        override = ("meta", name, {"role": "runtime", "evidence_origin": name}, False)
    yield from _user_blocks(base, prompt, origin=origin, compact=False, transcript_only=False, is_meta=False,
                            sidechain=state.sub, record={}, override=override)


def _attachment(base, record, state):
    a = record.get("attachment")
    kind = a.get("type") if isinstance(a, dict) else None
    if not isinstance(kind, str):
        yield _unsupported(base, "[unsupported_attachment:?] " + _keys(a))
        return
    rendered_role = record.get("renderedRole", a.get("renderedRole"))
    if rendered_role is not None and not isinstance(rendered_role, str):
        raise ValueError("renderedRole must be a string")
    if rendered_role is not None:
        base = {**base, "rendered_role": rendered_role}  # a rendering hint; never user authority
    if kind == "queued_command":
        yield from _queued_command(base, a, state)
        return
    check = _ATTACHMENTS.get(kind)
    parsed = check(a) if check is not None else None
    if parsed is None:
        yield _unsupported(base, f"[unsupported_attachment:{_label(kind) or '?'}] {_keys(a)}", attachment_type=kind)
        return
    suffix, meta = parsed
    yield _runtime(base, f"[attachment:{kind}]{suffix}", f"attachment:{kind}", attachment_type=kind,
                   **{key: value for key, value in meta.items() if value is not None})


# ----------------------------------------------------------------------------- runtime and metadata records

_SYSTEM_SUBTYPES = {"turn_duration", "stop_hook_summary", "away_summary", "compact_boundary", "local_command",
                    "informational"}
_QUEUE_OPERATIONS = {"enqueue", "dequeue", "remove"}
_METADATA = {"mode", "permission-mode", "ai-title", "agent-name", "pr-link", "file-history-snapshot",
             "file-history-delta", "cost-state", "atis-latch", "last-prompt", "continued-in"}
_SESSION_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def _system(base, record, state):
    subtype, level = record.get("subtype"), record.get("level", "")
    if subtype not in _SYSTEM_SUBTYPES or not isinstance(level, str) or not isinstance(record.get("isMeta", False), bool):
        yield _unsupported(base, f"[unsupported_system:{_label(subtype) or '?'}] {_keys(record)}")
        return
    extra = {"system_subtype": subtype, "level": _label(level) or None}
    if _is_int(record.get("durationMs")):
        extra["duration_ms"] = record["durationMs"]
    if subtype == "compact_boundary":
        extra.update(authoritative=False)
    yield _runtime(base, f"[system:{subtype}]",
                   "compaction" if subtype == "compact_boundary" else "runtime_metadata",
                   **{key: value for key, value in extra.items() if value is not None})


def _queue_operation(base, record, state):
    operation = record.get("operation")
    if operation not in _QUEUE_OPERATIONS:
        yield _unsupported(base, f"[unsupported_queue_operation:{_label(operation) or '?'}] {_keys(record)}")
        return
    extra = {"operation": operation}
    for key, name in (("deliveryId", "delivery_id"), ("reason", "reason")):
        if _label(record.get(key)):
            extra[name] = record[key]
    yield _runtime(base, f"[queue_operation:{operation}]", "runtime_metadata", **extra)


def _metadata(base, record, state):
    kind = record["type"]
    extra = {}
    if kind == "permission-mode" and _label(record.get("permissionMode")):
        extra["permission_mode"] = record["permissionMode"]
    if kind == "mode" and _label(record.get("mode")):
        extra["session_mode"] = record["mode"]
    if kind == "continued-in":
        # Links a session to the one that continues it; the id is an identifier, never text.
        target = record.get("continuedInSessionId")
        if not (isinstance(target, str) and _SESSION_REF.fullmatch(target)):
            yield _unsupported(base, f"[unsupported_claude_event:continued-in] {_keys(record)}")
            return
        extra["continued_in_session_id"] = target
    yield _runtime(base, f"[claude_metadata:{kind}]", "session_metadata", **extra)


_HANDLERS = {"user": _user, "assistant": _assistant, "attachment": _attachment, "system": _system,
             "queue-operation": _queue_operation, **{name: _metadata for name in _METADATA}}


def _events(record, state):
    kind = record.get("type")
    if not isinstance(kind, str):
        raise ValueError("record type must be a string")
    base, embedded = _envelope(record, state)
    if embedded:
        # Sidechain records inside a main transcript would mix a child's evidence into the parent session.
        yield _unsupported(base, f"[unsupported_claude_event:embedded_sidechain:{_label(kind) or '?'}]")
        return
    handler = _HANDLERS.get(kind)
    if handler is None:
        yield _unsupported(base, f"[unsupported_claude_event:{_label(kind) or '?'}] {_keys(record)}")
        return
    yield from handler(base, record, state)


# ----------------------------------------------------------------------------- parser

def parse_session_chunks(path: Path, *, session_id=None, start_byte=0, known_hashes=None, sink=None):
    """Parse one transcript (main or subagent file). ``known_hashes`` is accepted for legacy callers only.

    Each record is classified atomically: a malformed record yields one parse_error chunk and no partial
    evidence. Chunks are appended to ``sink`` one at a time so large sessions stream.
    """
    sid = peek_session_id(path)
    if sid is None or (session_id is not None and session_id != sid):
        raise ValueError("Claude Code transcript layout or session id mismatch")
    state = _File(sid)
    chunks, offset = ([] if sink is None else sink), start_byte
    for line, start, end, raw in iter_raw_lines(path, start_byte=start_byte):
        offset = end
        if not raw.strip():
            continue
        try:
            if any(0xDC80 <= ord(char) <= 0xDCFF for char in raw):
                raise ValueError("invalid UTF-8")
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise ValueError("JSONL record is not an object")
            timestamp = record.get("timestamp")
            if timestamp is not None and not isinstance(timestamp, str):
                raise ValueError("invalid timestamp")
            events = list(_events(record, state))
        except (ValueError, TypeError) as exc:
            chunks.append(parse_error("claude", sid, line, start, end, type(exc).__name__))
            continue
        for index, event in enumerate(events):
            chunks.append(make_chunk("claude", sid, event.kind, event.text, event.meta, line, start, end,
                                     sub=index, timestamp=timestamp))
    return chunks, offset, sid
