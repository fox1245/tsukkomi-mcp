"""Grok CLI session transcripts: ``<sessions dir>/<url-encoded cwd>/<session id>/updates.jsonl``.

``updates.jsonl`` is the ACP session update stream Grok appends to (``session/update``) with its own extension
records (``_x.ai/session/update``). It is the only file this provider reads: ``chat_history.jsonl`` and
``system_prompt.txt`` hold the rendered system prompt and are never opened (the resolver refuses any other name).

Rules every shape below follows:

* Session ids come from the directory that holds ``updates.jsonl``, never from record contents. A record that
  names another session (a fork or resume replaying history) is kept as evidence and marked ``inherited``; its user
  messages are never user authority.
* A ``user_message_chunk`` is a user turn (authority for contract anchors) only when Grok stamped it with its prompt
  index (the recorded shape of a typed prompt) and it does not open with the ``<system-reminder>`` wrapper Grok puts
  around hook-injected text. Anything else is runtime text, never authority.
* ``agent_message_chunk`` / ``agent_thought_chunk`` / ``user_message_chunk`` records are streamed deltas: contiguous
  records of the same kind (and prompt) are one event, flushed by any other record that produces an event, by a
  parse error and by the end of what was read. A run still open at the end of a read is emitted as it stands; chunks
  that arrive later start a new event (the cursor is a record boundary like every other provider's). Thoughts are
  runtime evidence with a placeholder text (the reasoning is never indexed, as with Claude Code's thinking blocks).
* ``tool_call`` {toolCallId, title (the tool name), rawInput} is a call; ``tool_call_update`` with a terminal
  ``status`` (``completed`` / ``failed``) is its result, linked by ``toolCallId``. An update without a status is
  progress and is skipped. A call that never gets a result is in flight or was cancelled: it has no result event, so
  it is never success. The tool name and keys are rendered in Claude Code's vocabulary (``grok_tools``).
* ``_x.ai/session/update`` ``hook_execution`` and ``hook_annotation`` are the host reporting on hooks (including
  this server's own denials): skipped, never indexed. ``turn_completed`` there is runtime metadata.
* Every other record kind is a placeholder flagged ``unsupported_event`` so coverage reports it; unknown never
  becomes clean. A malformed record is one ``parse_error`` chunk and no partial evidence.
* A ``spawn_subagent`` child is a separate session directory: only the parent's own file is read here.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path

from self_directing_mcp import grok_tools
from self_directing_mcp.codex import discover as _codex_discover
from self_directing_mcp.grok_ids import SESSION_FILE, UUID as _UUID, valid_id  # noqa: F401
from self_directing_mcp.session_events import has_invalid_utf8, iter_raw_lines, make_chunk, parse_error, result_success
from self_directing_mcp.tool_payload import tool_call_parts

# Bumping this triggers a lazy per-session reparse on the next sync/audit of each already-indexed session.
# v1: initial Grok CLI provider (updates.jsonl; tool names rendered by grok_tools.TOOL_MAP_VERSION 1).
PARSER_VERSION = 1

_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}\Z")
SESSION_UPDATE = "session/update"
EXTENSION_UPDATE = "_x.ai/session/update"
_REMINDER = re.compile(r"\s*<system-reminder[\s>]", re.IGNORECASE)
_CHUNKS = {"user_message_chunk": "user", "agent_message_chunk": "assistant", "agent_thought_chunk": "thought"}
_QUIET_EXTENSION = frozenset({"hook_execution", "hook_annotation"})     # the host reporting on hooks: not evidence
_PROGRESS = frozenset({"pending", "in_progress"})
_USAGE_FIELDS = ("inputTokens", "outputTokens", "totalTokens", "modelCalls", "numTurns")


class PathTraversalError(ValueError):
    pass


# ----------------------------------------------------------------------------- identity and discovery

def _layout_id(relative_parts) -> str | None:
    """Session id for a path relative to the sessions root; only ``<group>/<id>/updates.jsonl`` qualifies."""
    if len(relative_parts) == 3 and relative_parts[2] == SESSION_FILE and relative_parts[0] not in ("", ".", "..") \
            and valid_id(relative_parts[1]):
        return relative_parts[1]
    return None


def peek_session_id(path: Path) -> str | None:
    """The session id a file name and its directory claim (layout only; no I/O, no root)."""
    parts = Path(path).parts
    if len(parts) >= 2 and parts[-1] == SESSION_FILE and valid_id(parts[-2]):
        return parts[-2]
    return None


def _linked(path: Path) -> bool:
    return path.is_symlink() or os.path.isjunction(path)


def _find_by_id(root: Path, session_id: str) -> list[Path]:
    found: list[Path] = []
    try:
        groups = sorted(entry for entry in root.iterdir() if entry.is_dir() and not _linked(entry))
    except OSError:
        return found
    for group in groups:
        candidate = group / session_id / SESSION_FILE
        if _linked(group / session_id) or _linked(candidate) or not candidate.is_file():
            continue
        try:
            found.append(resolve_session_path(root, session_id=session_id, path=str(candidate)))
        except (ValueError, OSError):
            continue
    return list(dict.fromkeys(found))


def resolve_session_path(root: Path, *, session_id: str | None = None, path: str | None = None,
                         require_exists: bool = True) -> Path:
    """Resolve a transcript below ``root`` by id or explicit path.

    The path must be absolute, free of ``..``, inside the configured root, in the official layout, named exactly
    ``updates.jsonl`` (no other file of a session directory is evidence), and no component below the root may be a
    symlink or junction (a link could redirect the evidence file or change which session it claims). The root
    itself may be a link.
    """
    if session_id is not None and not valid_id(session_id):
        raise ValueError("invalid Grok session id")
    configured = Path(os.path.abspath(Path(root).expanduser()))
    root = Path(root).expanduser().resolve()
    if path is None:
        if not session_id:
            raise ValueError("Grok session_id or path is required")
        matches = _find_by_id(root, session_id)
        if len(matches) != 1:
            raise FileNotFoundError("Grok session not found or ambiguous under configured root")
        return matches[0]
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise PathTraversalError("Grok transcript must be an absolute path without traversal")
    for prefix in dict.fromkeys((root, configured)):
        if candidate.is_relative_to(prefix):
            below = candidate.relative_to(prefix).parts
            break
    else:
        raise PathTraversalError("Grok transcript is outside the configured sessions root")
    current = prefix
    for part in below:
        current = current / part
        if _linked(current):
            raise PathTraversalError("Grok transcript path may not contain symlinks")
    resolved = root.joinpath(*below)
    try:
        _codex_discover.ensure_under_root(resolved, root)
    except _codex_discover.PathTraversalError as exc:
        raise PathTraversalError(str(exc)) from exc
    claimed = _layout_id(below)
    if claimed is None:
        raise ValueError("not an official Grok transcript location (<group>/<session id>/updates.jsonl)")
    if session_id is not None and claimed != session_id:
        raise ValueError("Grok session id does not match the transcript location")
    if require_exists and not resolved.is_file():
        raise FileNotFoundError("Grok transcript not found")
    return resolved


def discover_session_files(root: Path) -> list[Path]:
    """Transcripts in the official layout below ``root``. Symlinked files and directories are never followed."""
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
    for group in entries(root):
        if not plain(group, directory=True):
            continue
        for session in entries(group.path):
            if not (plain(session, directory=True) and valid_id(session.name)):
                continue
            transcript = next((e for e in entries(session.path) if e.name == SESSION_FILE and plain(e, directory=False)),
                              None)
            if transcript is not None:
                found.append(Path(transcript.path))
    return sorted(found)


# ----------------------------------------------------------------------------- shape helpers

def _label(value) -> str | None:
    return value if isinstance(value, str) and _LABEL.fullmatch(value) else None


def _keys(value) -> str:
    """Key names only, for diagnostics about shapes whose payload must not be indexed."""
    if not isinstance(value, dict):
        return "keys=-"
    names = sorted(str(key) if _LABEL.fullmatch(str(key)) else "?" for key in value)
    return "keys=" + ",".join(names[:30])


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _timestamp(record) -> str | None:
    """Grok stamps whole seconds since the epoch; a string is kept as it is; anything else is malformed."""
    value = record.get("timestamp")
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid timestamp")
    try:
        return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, OSError, ValueError):
        raise ValueError("invalid timestamp") from None


class _Event:
    """One chunk to emit: kind, text, meta. ``sub`` is assigned in emission order."""
    __slots__ = ("kind", "text", "meta")

    def __init__(self, kind, text, meta):
        self.kind, self.text, self.meta = kind, text, meta


class _Part:
    """One streamed delta, not yet an event: the run it belongs to decides what it becomes."""
    __slots__ = ("flavor", "key", "text", "meta")

    def __init__(self, flavor, key, text, meta):
        self.flavor, self.key, self.text, self.meta = flavor, key, text, meta


class _Run:
    """Contiguous deltas of one kind. Positions are those of the first and last record."""
    __slots__ = ("flavor", "key", "texts", "meta", "line", "last_line", "start", "end", "timestamp")

    def __init__(self, part, line, start, end, timestamp):
        self.flavor, self.key, self.meta = part.flavor, part.key, part.meta
        self.texts, self.line, self.last_line = [part.text], line, line
        self.start, self.end, self.timestamp = start, end, timestamp

    def add(self, part, line, end):
        self.texts.append(part.text)
        self.last_line, self.end = line, end


def _runtime(base, text, origin, **extra):
    return _Event("meta", text, {**base, "role": "runtime", "evidence_origin": origin, **extra})


def _unsupported(base, text, **extra):
    return _Event("meta", text, {**base, "role": "runtime", "unsupported_event": True, **extra})


class _File:
    """Per-parse state."""

    def __init__(self, sid):
        self.sid = sid
        self.run: _Run | None = None


def _run_event(run: _Run) -> _Event:
    text = "".join(run.texts)
    base = {**run.meta, "chunk_count": len(run.texts)}
    flavor = run.flavor
    if flavor == "thought":
        return _runtime(base, "[thinking]", "thinking", text_length=len(text))
    if flavor == "assistant":
        if not text:
            return _runtime(base, "[message:assistant]", "empty_message")
        return _Event("turn", f"[message:assistant] {text}", {**base, "role": "assistant"})
    # user chunks
    if base.pop("_authority", False) is True:
        if not text:
            return _runtime(base, "[message:user]", "empty_message")
        return _Event("turn", f"[message:user] {text}", {**base, "role": "user"})
    origin = base.pop("_origin", "user_message_unattributed")
    return _runtime(base, f"[{origin}] {text}" if text else f"[{origin}]", origin)


# ----------------------------------------------------------------------------- records

def _envelope(record, state):
    """Validated parts common to every record: ``(method, params, update, meta)``; ValueError on violations."""
    method, params = record.get("method"), record.get("params")
    if not isinstance(method, str) or not isinstance(params, dict):
        raise ValueError("record has no method or params")
    update = params.get("update")
    if not isinstance(update, dict) or not isinstance(update.get("sessionUpdate"), str):
        raise ValueError("record has no sessionUpdate")
    claimed = params.get("sessionId")
    if claimed is not None and not isinstance(claimed, str):
        raise ValueError("sessionId must be a string")
    envelope = params.get("_meta")
    if envelope is not None and not isinstance(envelope, dict):
        raise ValueError("_meta must be an object")
    meta = {"event_type": update["sessionUpdate"]}
    if claimed is not None and claimed != state.sid:
        meta["inherited"] = True            # replayed history of another session: evidence, never authority
        meta["record_session_id"] = claimed[:64]
    prompt = (envelope or {}).get("promptId")
    if isinstance(prompt, str) and prompt:
        meta["prompt_id"] = prompt[:80]
    return method, params, update, meta


def _chunk_text(update):
    content = update.get("content")
    if isinstance(content, dict) and content.get("type") == "text" and isinstance(content.get("text"), str):
        return content["text"]
    return None


def _chunk(base, update, state):
    kind = update["sessionUpdate"]
    flavor = _CHUNKS[kind]
    text = _chunk_text(update)
    if text is None:
        content = update.get("content")
        label = _label(content.get("type")) if isinstance(content, dict) else None
        yield _unsupported(base, f"[unsupported_content:{label or '?'}] {_keys(content)}")
        return
    inner = update.get("_meta")
    inner = inner if isinstance(inner, dict) else {}
    meta = dict(base)
    key = (flavor, base.get("prompt_id"), base.get("inherited", False))
    if flavor == "user":
        index = inner.get("promptIndex")
        if _is_int(index):
            meta["prompt_index"] = index
            key += (index,)
        if _label(inner.get("modelId")):
            meta["model_id"] = inner["modelId"]
        if base.get("inherited"):
            origin = "inherited_history"
        elif _REMINDER.match(text):
            origin = "injected_system_reminder"
        elif not _is_int(index):
            origin = "user_message_unattributed"
        else:
            origin = None
        meta["_authority"] = origin is None
        if origin is not None:
            meta["_origin"] = origin
    yield _Part(flavor, key, text, meta)


def _content_text(content):
    """``(text, unsupported)`` of a tool update's ``content`` list; diffs are named, their text is not repeated."""
    parts, unsupported, diff_paths = [], False, []
    for item in content if isinstance(content, list) else ():
        kind = item.get("type") if isinstance(item, dict) else None
        if kind == "content" and isinstance(item.get("content"), dict) and item["content"].get("type") == "text" \
                and isinstance(item["content"].get("text"), str):
            parts.append(item["content"]["text"])
        elif kind == "diff" and isinstance(item.get("path"), str):
            parts.append(f"[diff path={item['path']}]")
            diff_paths.append(item["path"])
        else:
            unsupported = True
            parts.append(f"[unsupported_content:{_label(kind) or '?'}]")
    return "\n".join(parts), unsupported, diff_paths


def _success(status, raw_output):
    """Failed is failure. Completed is success unless the tool's own output says the run failed (exit code,
    timeout, signal): a non-zero shell exit is ``completed`` for the host."""
    if status != "completed":
        return False
    if isinstance(raw_output, dict):
        code = result_success(raw_output)
        if code is not None:
            return code
        if raw_output.get("timed_out") is True or (raw_output.get("signal") not in (None, 0, "")):
            return False
    return True


def _tool_call(base, update):
    call_id, title = update.get("toolCallId"), update.get("title")
    arguments = update.get("rawInput")
    if arguments is None:
        arguments = {}
    if not (isinstance(call_id, str) and call_id and isinstance(title, str) and title and isinstance(arguments, dict)):
        yield _unsupported(base, "[unsupported_tool_call] " + _keys(update))
        return
    canonical = grok_tools.canonical_call(title, arguments)
    text, extra = tool_call_parts("grok", title, arguments)
    yield _Event("tool_call", text, {**base, "role": "assistant", "tool_name": canonical.name, "call_id": call_id,
                                     **extra})


def _tool_update(base, update):
    call_id, status = update.get("toolCallId"), update.get("status")
    if not (isinstance(call_id, str) and call_id):
        yield _unsupported(base, "[unsupported_tool_call_update] " + _keys(update))
        return
    if status is None or status in _PROGRESS:
        return                                # progress or the richer start of a call: not a result
    if status not in ("completed", "failed"):
        yield _unsupported(base, f"[unsupported_tool_status:{_label(status) or '?'}] " + _keys(update),
                           call_id=call_id)
        return
    raw_output = update.get("rawOutput")
    text, unsupported, diff_paths = _content_text(update.get("content"))
    if not text and isinstance(raw_output, dict) and isinstance(raw_output.get("output_for_prompt"), str):
        text = raw_output["output_for_prompt"]
    meta = {**base, "role": "tool", "call_id": call_id, "tool_name": None, "success": _success(status, raw_output),
            "tool_status": status}
    if unsupported:
        meta["unsupported_event"] = True
    if diff_paths:
        meta["diff_paths"] = diff_paths[:20]
    inner = update.get("_meta")
    if isinstance(inner, dict) and _label(inner.get("toolNotRun")):
        meta["tool_not_run"] = inner["toolNotRun"]          # the call was cancelled before it ran
    if isinstance(raw_output, dict):
        if _label(raw_output.get("type")):
            meta["raw_output_type"] = raw_output["type"]
        if _is_int(raw_output.get("exit_code")):
            meta["exit_code"] = raw_output["exit_code"]
        if raw_output.get("truncated") is True:
            meta["output_truncated"] = True
    yield _Event("tool_result", "[tool_result] " + text, meta)


def _turn_completed(base, update):
    stop = update.get("stop_reason")
    extra = {"stop_reason": stop} if _label(stop) else {}
    if _label(update.get("prompt_id")):
        extra["prompt_id"] = update["prompt_id"]
    usage = update.get("usage")
    if isinstance(usage, dict):
        extra.update({name: usage[name] for name in _USAGE_FIELDS if _is_int(usage.get(name))})
    yield _runtime(base, f"[turn_completed] stop_reason={extra.get('stop_reason', '?')}", "turn_completed", **extra)


def _items(record, state):
    """Events and streamed deltas of one record; an empty result skips the record without ending a run."""
    method, params, update, base = _envelope(record, state)
    kind = update["sessionUpdate"]
    if method == SESSION_UPDATE:
        if kind in _CHUNKS:
            yield from _chunk(base, update, state)
        elif kind == "tool_call":
            yield from _tool_call(base, update)
        elif kind == "tool_call_update":
            yield from _tool_update(base, update)
        else:
            yield _unsupported(base, f"[unsupported_grok_event:{_label(kind) or '?'}] {_keys(update)}")
    elif method == EXTENSION_UPDATE:
        if kind in _QUIET_EXTENSION:
            return
        if kind == "turn_completed":
            yield from _turn_completed(base, update)
        else:
            yield _unsupported(base, f"[unsupported_grok_extension:{_label(kind) or '?'}] {_keys(update)}")
    else:
        yield _unsupported(base, f"[unsupported_grok_method:{_label(method) or '?'}] {_keys(update)}")


# ----------------------------------------------------------------------------- parser

def parse_session_chunks(path: Path, *, session_id=None, start_byte=0, known_hashes=None, sink=None):
    """Parse one ``updates.jsonl``. ``known_hashes`` is accepted for legacy callers only.

    Each record is classified atomically: a malformed record yields one parse_error chunk and no partial evidence.
    Chunks are appended to ``sink`` one at a time so large sessions stream. The returned offset is the end of the
    last record read, like every other provider's: a streamed run open at that point is emitted as it stands.
    """
    sid = peek_session_id(path)
    if sid is None or (session_id is not None and session_id != sid):
        raise ValueError("Grok transcript layout or session id mismatch")
    state = _File(sid)
    chunks, offset = ([] if sink is None else sink), start_byte

    def flush():
        run, state.run = state.run, None
        if run is not None:
            chunk = make_chunk("grok", sid, *_split(_run_event(run)), run.line, run.start, run.end,
                               sub=0, timestamp=run.timestamp)
            chunk.line_end = run.last_line
            chunks.append(chunk)

    for line, start, end, raw in iter_raw_lines(path, start_byte=start_byte):
        offset = end
        if not raw.strip():
            continue
        try:
            if has_invalid_utf8(raw):
                raise ValueError("invalid UTF-8")
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise ValueError("JSONL record is not an object")
            timestamp = _timestamp(record)
            items = list(_items(record, state))
        except (ValueError, TypeError, RecursionError) as exc:
            flush()
            chunks.append(parse_error("grok", sid, line, start, end, type(exc).__name__))
            continue
        sub = 0
        for item in items:
            if isinstance(item, _Part):
                run = state.run
                if run is not None and run.key == item.key:
                    run.add(item, line, end)
                    continue
                flush()
                state.run = _Run(item, line, start, end, timestamp)
                continue
            flush()
            chunks.append(make_chunk("grok", sid, item.kind, item.text, item.meta, line, start, end,
                                     sub=sub, timestamp=timestamp))
            sub += 1
    flush()
    return chunks, offset, sid


def _split(event: _Event):
    return event.kind, event.text, event.meta
