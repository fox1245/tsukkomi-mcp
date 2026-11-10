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

from self_directing_mcp.claude_ids import (  # noqa: F401
    RUN as _RUN, STEM as _STEM, SUBAGENTS, UUID as _UUID, WORKFLOWS, session_parts, valid_id,
)
from self_directing_mcp.codex import discover as _codex_discover
from self_directing_mcp.session_events import has_invalid_utf8, iter_raw_lines, make_chunk, parse_error, result_success
from self_directing_mcp.tool_payload import tool_call_parts

# Bumping this triggers a lazy per-session reparse on the next sync/audit of each already-indexed session.
# v1: initial Claude Code provider (main and subagent transcripts, origin-based user authority).
# v2: the ``system`` record ``scheduled_task_fire`` (a cron task fired) as runtime metadata.
# v3: a tool call's model-written ``description`` is kept out of the call text (``tool_payload``) and recorded in
#     ``meta.tool_prose``; contracts match the payload only.
# v4: Claude Code 2.1.30x runtime metadata records (``api-request``, ``api-request-blob``, ``api-request-shape``,
#     ``fork-context-ref``).
# v5: Claude Code 2.1.31x shapes: the ``session-owner`` and ``tools-used`` records, the list form of an api-request's
#     ``cacheControl``, the newer peer origin fields, ``queued_command`` image paste ids and tool deltas, the
#     ``file`` attachment's ``savedCopyFraming``, the ``ultra_effort_enter``/``ultra_effort_exit``, ``advisor_tool``,
#     ``command_permissions``, ``invoked_skills``, ``hook_cancelled``, ``plan_mode_exit``, ``thinking_stripped`` and
#     ``advisor_stripped`` attachments, the ``agents_killed`` system record, the ``popAll`` queue operation, the
#     ``auto-continuation`` origin, PDF documents in tool results, and the hosted ``server_tool_use`` call with its
#     encrypted or failed ``advisor_tool_result``.
PARSER_VERSION = 5

_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}\Z")
_HOOK_NAME = re.compile(r"[A-Za-z0-9_.:*/@+-]{1,200}\Z")
_CRON = re.compile(r"[0-9*/,?# -]{1,100}\Z")  # digits and cron punctuation only: no word can ride along


class PathTraversalError(ValueError):
    pass


# ----------------------------------------------------------------------------- identity and discovery
#
# Session ids come from the transcript's location, never from record contents:
#   <project>/<uuid>.jsonl                                      -> "<uuid>"
#   <project>/<uuid>/subagents/<stem>.jsonl                     -> "<uuid>_<stem>"   (child of session <uuid>)
#   <project>/<uuid>/subagents/workflows/<run>/<stem>.jsonl     -> "<uuid>_<stem>"   (workflow child of session <uuid>)
# Every record of a subagent file carries the PARENT's ``sessionId`` and ``agentId``, so neither is unique per
# file; the composite id keeps parent and child in separate (provider, session) index partitions, keeps
# duplicate agent ids under different parents apart, and still matches the workflow id alphabet.

def _id_from_parts(parts) -> str | None:
    """Session id for the trailing components of a transcript path (layout only; no I/O)."""
    if not parts or not parts[-1].endswith(".jsonl"):
        return None
    stem = parts[-1][:-len(".jsonl")]
    if len(parts) >= 3 and parts[-2] == SUBAGENTS and _UUID.fullmatch(parts[-3]) and _STEM.fullmatch(stem):
        return f"{parts[-3]}_{stem}"
    if (len(parts) >= 5 and parts[-3] == WORKFLOWS and parts[-4] == SUBAGENTS
            and _UUID.fullmatch(parts[-5]) and _RUN.fullmatch(parts[-2]) and _STEM.fullmatch(stem)):
        return f"{parts[-5]}_{stem}"
    for i in range(len(parts) - 1):
        if parts[i] == SUBAGENTS and ((i > 0 and _UUID.fullmatch(parts[i - 1]))
                                      or (i + 1 < len(parts) - 1 and parts[i + 1] == WORKFLOWS)):
            return None
    return stem if _UUID.fullmatch(stem) else None


def peek_session_id(path: Path) -> str | None:
    return _id_from_parts(Path(path).parts)


def _layout_id(relative_parts) -> str | None:
    """Session id for a path relative to the projects root; only the three official layouts qualify."""
    if len(relative_parts) == 2:
        return _id_from_parts(relative_parts[1:]) if _UUID.fullmatch(relative_parts[1][:-len(".jsonl")]) else None
    if len(relative_parts) == 4 and relative_parts[2] == SUBAGENTS:
        return _id_from_parts(relative_parts[1:])
    if len(relative_parts) == 6 and relative_parts[2] == SUBAGENTS and relative_parts[3] == WORKFLOWS:
        return _id_from_parts(relative_parts[1:])
    return None


def _linked(path: Path) -> bool:
    return path.is_symlink() or os.path.isjunction(path)


def _linked_project_target(link: Path, *roots: Path) -> Path:
    """The real project directory a linked project directory stands for, or PathTraversalError.

    Exactly one hop is read (``os.readlink``; a relative target is taken against the link's own directory). The
    target must be spelled as a direct child of the projects root (``roots``: its resolved and configured
    spellings) with no ``..`` and no other link in its path, and must be an existing directory that is
    not itself a link. A chain, a hop that leaves the root, the root itself, a directory deeper down and a
    dangling link are refused: every extra hop is a path another party may control and may change between this
    check and the read. Returns the target in the resolved root's spelling.
    """
    refused = PathTraversalError("Claude Code transcript path may not contain symlinks")
    try:
        hop = os.readlink(link)
    except OSError as exc:
        raise refused from exc
    target = Path(os.path.join(link.parent, hop))
    if not target.is_absolute() or target.name in ("", ".", ".."):
        raise refused
    if target.parent not in roots or _linked(target):
        raise refused
    try:
        if not target.is_dir():
            raise refused
    except OSError as exc:
        raise refused from exc
    return roots[0] / target.name


def _entries(directory):
    try:
        with os.scandir(directory) as scan:
            return sorted(scan, key=lambda entry: entry.name)
    except OSError:
        return []


def _plain_entry(entry, *, directory: bool) -> bool:
    try:
        return (entry.is_dir(follow_symlinks=False) if directory else entry.is_file(follow_symlinks=False)) \
            and not os.path.isjunction(entry.path)
    except OSError:
        return False


def _subagent_candidates(sub_dir: Path, *, allow_ambiguous: bool = False) -> list[tuple[str, Path]]:
    """``(stem, path)`` for subagent transcripts under ``sub_dir`` (ordinary and ``workflows/<run>/``).

    When ``allow_ambiguous`` is False, a stem that appears in more than one place under ``sub_dir`` is omitted
    completely so discovery never picks one copy arbitrarily.
    """
    raw: list[tuple[str, Path]] = []
    counts: dict[str, int] = {}
    for entry in _entries(sub_dir):
        if _plain_entry(entry, directory=False) and entry.name.endswith(".jsonl"):
            stem = entry.name[:-len(".jsonl")]
            if _STEM.fullmatch(stem):
                raw.append((stem, Path(entry.path)))
                counts[stem] = counts.get(stem, 0) + 1
        elif entry.name == WORKFLOWS and _plain_entry(entry, directory=True):
            for run in _entries(entry.path):
                if not (_plain_entry(run, directory=True) and _RUN.fullmatch(run.name)):
                    continue
                for child in _entries(run.path):
                    if _plain_entry(child, directory=False) and child.name.endswith(".jsonl"):
                        stem = child.name[:-len(".jsonl")]
                        if _STEM.fullmatch(stem):
                            raw.append((stem, Path(child.path)))
                            counts[stem] = counts.get(stem, 0) + 1
    if allow_ambiguous:
        return raw
    return [(stem, path) for stem, path in raw if counts[stem] == 1]


def _find_by_id(root: Path, session_id: str) -> list[Path]:
    parent, stem = session_parts(session_id)
    found: list[Path] = []
    try:
        projects = sorted(entry for entry in root.iterdir() if entry.is_dir() and not _linked(entry))
    except OSError:
        return found
    for project in projects:
        if stem is None:
            candidates = [project / f"{session_id}.jsonl"]
        else:
            parent_dir = project / parent
            sub_dir = parent_dir / SUBAGENTS
            if _linked(parent_dir) or _linked(sub_dir) or not sub_dir.is_dir():
                continue
            candidates = [path for candidate_stem, path in _subagent_candidates(sub_dir, allow_ambiguous=True)
                          if candidate_stem == stem]
        for candidate in candidates:
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

    One link is accepted: the project directory, the first component below the root, when it resolves to an
    existing real directory that is a direct child of the resolved root (see ``_linked_project_target``). The
    answer is then the canonical path through that target, so a file has one spelling whichever way it was named.
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
    for index, part in enumerate(below):
        current = current / part
        if not _linked(current):
            continue
        if index != 0 or len(below) < 2:  # only the project directory, never the transcript or anything deeper
            raise PathTraversalError("Claude Code transcript path may not contain symlinks")
        target = _linked_project_target(current, root, configured)
        below = (target.name, *below[1:])
        current = target
        for rest in below[1:]:
            current = current / rest
            if _linked(current):
                raise PathTraversalError("Claude Code transcript path may not contain symlinks")
        break
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
    found: list[Path] = []
    for project in _entries(root):
        if not _plain_entry(project, directory=True):
            continue
        for entry in _entries(project.path):
            if _plain_entry(entry, directory=False) and entry.name.endswith(".jsonl") \
                    and _UUID.fullmatch(entry.name[:-len(".jsonl")]):
                found.append(Path(entry.path))
            elif include_subagents and _plain_entry(entry, directory=True) and _UUID.fullmatch(entry.name):
                nested = next((e for e in _entries(entry.path) if e.name == SUBAGENTS and _plain_entry(e, directory=True)),
                              None)
                if nested is not None:
                    found.extend(path for _stem, path in _subagent_candidates(Path(nested.path)))
    return sorted(found)


def discover_subagent_files(root: Path, session_id: str, *, path: Path | None = None) -> list[tuple[str, Path]]:
    """Return ``(subagent_session_id, resolved_path)`` pairs for all subagent transcripts of ``session_id``."""
    parts = session_parts(session_id)
    if parts is None:
        return []
    parent = parts[0]
    root = Path(root).expanduser().resolve()
    candidate_dirs: list[Path] = []
    if path is not None:
        try:
            resolved_main = resolve_session_path(root, path=str(path))
            main_parts = session_parts(peek_session_id(resolved_main))
            if main_parts is not None and main_parts[0] == parent:
                project_dir = root / resolved_main.relative_to(root).parts[0]
                candidate_dirs.append(project_dir / parent / SUBAGENTS)
        except (ValueError, OSError):
            pass
    if not candidate_dirs:
        try:
            projects = sorted(entry for entry in root.iterdir() if entry.is_dir() and not _linked(entry))
        except OSError:
            projects = []
        for project in projects:
            parent_dir = project / parent
            sub_dir = parent_dir / SUBAGENTS
            if not _linked(parent_dir) and not _linked(sub_dir) and sub_dir.is_dir():
                candidate_dirs.append(sub_dir)
    seen: set[str] = set()
    found: list[tuple[str, Path]] = []
    for sub_dir in candidate_dirs:
        if _linked(sub_dir.parent) or _linked(sub_dir) or not sub_dir.is_dir():
            continue
        for stem, entry in _subagent_candidates(sub_dir):
            sub_sid = f"{parent}_{stem}"
            if sub_sid in seen:
                continue
            try:
                resolved = resolve_session_path(root, session_id=sub_sid, path=str(entry))
            except (ValueError, OSError):
                continue
            seen.add(sub_sid)
            found.append((sub_sid, resolved))
    return found


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

_NON_HUMAN = {"task-notification": "task_notification", "peer": "peer_message", "coordinator": "coordinator",
              "auto-continuation": "auto_continuation"}
# The fields an origin of each kind may carry: ``key -> (declared type, meta name)``; a meta name of None means the field
# is validated and never kept. A peer origin also names the peer's message id, the mode the sender runs in, the sender
# process the host verified, the chain of sessions that relayed it and the text of a handed-back report; none of them
# makes a peer the user. ``auto-continuation`` is a turn the host continued by itself.
_ORIGIN_FIELDS = {
    "human": {},
    "coordinator": {},
    "auto-continuation": {},
    "peer": {"body": (str, None), "from": (str, "peer_from"), "handback": ((bool, str, dict), None),
             "handbackReportText": (str, None), "name": (str, "peer_name"), "senderTaskId": (str, "peer_task_id"),
             "msg_id": (str, "peer_msg_id"), "fromMode": (str, "peer_from_mode"),
             "verifiedPeerPid": (int, "peer_verified_pid"), "verifiedPeerProcStart": (str, None),
             "hopChain": (list, None)},
    "task-notification": {"producer": (str, "producer"), "runId": (str, "run_id")},
}


def _origin(value):
    """``(status, kind, meta)``; status is "none", "ok" or "bad". Unknown or malformed origins are never trusted."""
    if value is None:
        return "none", None, {}
    kind = value.get("kind") if isinstance(value, dict) else None
    fields = _ORIGIN_FIELDS.get(kind) if isinstance(kind, str) else None
    if fields is None or not set(value) <= {"kind"} | set(fields) \
            or not _typed(value, **{key: declared for key, (declared, _name) in fields.items()}):
        return "bad", None, {}
    if "hopChain" in value and _labels(value["hopChain"]) is None:
        return "bad", None, {}
    extra = {name: value[key] for key, (_declared, name) in fields.items() if name is not None and key in value}
    if "hopChain" in value:
        extra["peer_hop_count"] = len(value["hopChain"])
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


# Media types of a base64 ``document`` block in a tool result (a Read of a PDF returns one).
_DOCUMENT_MEDIA_TYPES = {"application/pdf"}


def _document_placeholder(block) -> str | None:
    """Placeholder for a strictly valid base64 document block in a tool result; the payload is never indexed."""
    if not (isinstance(block, dict) and block.get("type") == "document" and set(block) <= {"type", "source", "cache_control"}
            and isinstance(block.get("source"), dict)):
        return None
    source = block["source"]
    if source.get("type") == "base64" and set(source) == {"type", "media_type", "data"} \
            and isinstance(source["data"], str) and isinstance(source["media_type"], str) \
            and source["media_type"] in _DOCUMENT_MEDIA_TYPES:
        return f"[document:{source['media_type']}]"
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
        elif (placeholder := _image_placeholder(item)) is not None or (placeholder := _document_placeholder(item)) is not None:
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
                text, prose = tool_call_parts("claude", block["name"], block["input"])
                yield _Event("tool_call", text, {**base, "role": "assistant", "tool_name": block["name"],
                                                 "call_id": block["id"], **prose})
        elif kind == "server_tool_use" and _server_tool_use_ok(block):
            # A tool the API runs itself (hosted), called by the model: a call, never a local execution or user text.
            if api_error or synthetic:
                yield _unsupported(base, "[unsupported_content:server_tool_use] keys=-")
            else:
                text, prose = tool_call_parts("claude", block["name"], block["input"])
                yield _Event("tool_call", text, {**base, "role": "assistant", "tool_name": block["name"],
                                                 "call_id": block["id"], "evidence_origin": "hosted_tool", **prose})
        elif kind == "advisor_tool_result" and (result := _advisor_result(block)) is not None:
            if api_error or synthetic:
                yield _unsupported(base, "[unsupported_content:advisor_tool_result] keys=-")
            else:
                content_type, success, error_code = result
                suffix = f" error_code={error_code}" if error_code else ""
                yield _Event("tool_result", f"[tool_result] [{content_type}]{suffix}",
                             {**base, "role": "tool", "call_id": block["tool_use_id"], "tool_name": None,
                              "success": success, "evidence_origin": "hosted_tool",
                              **({"error_code": error_code} if error_code else {})})
        else:
            yield _unsupported_block(base, kind, block)


_SERVER_TOOL_USE_KEYS = {"type", "id", "name", "input"}
# Advisor results by ``content.type``: required fields, optional fields and what the result says about success. The
# encrypted advice cannot be read and is never indexed; a failed consultation names its error code. The plaintext form
# (``advisor_result``) stays unsupported until a transcript shows it.
_ADVISOR_RESULTS = {"advisor_redacted_result": ({"encrypted_content": str}, {"stop_reason": str}, True),
                    "advisor_tool_result_error": ({"error_code": str}, {}, False)}
_ADVISOR_LABELS = ("stop_reason", "error_code")  # fields of an advisor result that must be plain identifiers


def _server_tool_use_ok(block) -> bool:
    return (set(block) == _SERVER_TOOL_USE_KEYS and isinstance(block["id"], str) and bool(block["id"])
            and _label(block["name"]) is not None and isinstance(block["input"], dict))


def _advisor_result(block):
    """``(content type, success, error code or None)`` of a strictly valid advisor tool result block, or None."""
    content = block.get("content")
    if not (set(block) == {"type", "tool_use_id", "content"} and isinstance(block["tool_use_id"], str)
            and block["tool_use_id"] and isinstance(content, dict) and isinstance(content.get("type"), str)):
        return None
    known = _ADVISOR_RESULTS.get(content["type"])
    if known is None:
        return None
    required, optional, success = known
    if not _shape(content, required, optional) or any(
            key in content and _label(content[key]) is None for key in _ADVISOR_LABELS):
        return None
    return content["type"], success, content.get("error_code")


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


def _labels(value):
    """A list of plain identifiers (``_LABEL``), or None."""
    return value if isinstance(value, list) and all(_label(item) is not None for item in value) else None


def _ints(value):
    return value if isinstance(value, list) and all(_is_int(item) for item in value) else None


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


def _hook_cancelled(a):
    """A hook the host cancelled (it ran out of time): event, name and the timing are kept, the command never is."""
    if not (_shape(a, {"command": str, "durationMs": int, "hookEvent": str, "hookName": str, "timedOut": bool,
                       "timeoutMs": int, "toolUseID": str}) and a["toolUseID"]):
        return None
    text, meta = _hook_ref(a)
    return (f"{text} timed_out={'true' if a['timedOut'] else 'false'}",
            {**meta, "tool_use_id": a["toolUseID"], "timed_out": a["timedOut"], "duration_ms": a["durationMs"],
             "timeout_ms": a["timeoutMs"]})


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
    """A file attached to the context: its path is evidence of what was shown, its content never is.

    ``savedCopyFraming`` (``{"enveloped": bool}``) says how the host framed the copy of the content it saved; it is kept
    as a flag."""
    required = {"filename": str, "displayPath": str, **({"content": dict} if with_content else {})}
    optional = {"savedCopyFraming": dict} if with_content else {}

    def check(a):
        if not _shape(a, required, optional):
            return None
        if with_content and not (set(a["content"]) == {"file", "type"} and isinstance(a["content"]["type"], str)
                                 and isinstance(a["content"]["file"], dict)):
            return None
        framing = a.get("savedCopyFraming")
        if framing is not None and not (set(framing) == {"enveloped"} and isinstance(framing["enveloped"], bool)):
            return None
        return (f" display_path={_one_line(a['displayPath'])} filename={_one_line(a['filename'])}",
                {"filename": a["filename"], "display_path": a["displayPath"],
                 **({"saved_copy_enveloped": framing["enveloped"]} if framing is not None else {})})
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


def _plan_mode_exit(a):
    """Plan mode ended: whether a plan exists is indexed; the plan file is never read."""
    if not _shape(a, {"planExists": bool, "planFilePath": str}):
        return None
    return (f" plan_exists={'true' if a['planExists'] else 'false'}",
            {"plan_exists": a["planExists"], "plan_file_path": a["planFilePath"]})


def _thinking_stripped(a):
    """The host removed a thinking block from the model's context: which one (message id, index) and the scope."""
    origin = a.get("from")
    if not (_shape(a, {"from": dict, "scope": str}) and (scope := _label(a["scope"])) is not None
            and set(origin) == {"messageId", "thinkingIndex"} and isinstance(origin["messageId"], str)
            and _is_int(origin["thinkingIndex"])):
        return None
    return f" scope={scope}", {"scope": scope, "stripped_message_id": origin["messageId"],
                               "thinking_index": origin["thinkingIndex"]}


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


def _bare(a):
    """An attachment that carries nothing but its type."""
    return ("", {}) if _shape(a, {}) else None


def _effort_mode(a):
    """The host reminds the model that a higher-effort mode began: the reminder type is kept, the reminder text is not."""
    if not _shape(a, {"reminderType": str}) or (kind := _label(a["reminderType"])) is None:
        return None
    return f" reminder_type={kind}", {"reminder_type": kind}


def _advisor_tool(a):
    """The host added or removed the advisor tool. A change of the model's tools, not something the user said."""
    if not _shape(a, {"available": bool, "model": str, "toolChange": str}, {"byBase": bool}) \
            or (change := _label(a["toolChange"])) is None:
        return None
    return (f" tool_change={change} available={'true' if a['available'] else 'false'}",
            {"tool_change": change, "available": a["available"], "advisor_model": _label(a["model"]),
             "by_base": a.get("byBase")})


def _command_permissions(a):
    """Tools a command the user invoked may use. A host permission grant, never a user instruction: the text carries the
    count, and only rules that are plain names are kept (in meta)."""
    if not _shape(a, {"allowedTools": list}) or _names(a["allowedTools"]) is None:
        return None
    rules = a["allowedTools"]
    return (f" allowed_tools={len(rules)}",
            {"allowed_tool_count": len(rules), "allowed_tool_names": [rule for rule in rules if _label(rule)]})


def _invoked_skills(a):
    """Skills invoked earlier, re-injected by the host. Their content is injected instruction text, not something the
    user said: validated and never indexed. Skill names that are plain identifiers are kept."""
    skills = a.get("skills")
    if not _shape(a, {"skills": list}) or not all(
            isinstance(skill, dict) and set(skill) == {"content", "name", "path"}
            and all(isinstance(value, str) for value in skill.values()) for skill in skills):
        return None
    names = [skill["name"] for skill in skills if _label(skill["name"])]
    return f" count={len(skills)} names={','.join(names)}", {"skill_count": len(skills), "skill_names": names}


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
    "hook_cancelled": _hook_cancelled,
    "plan_mode_exit": _plan_mode_exit,
    "thinking_stripped": _thinking_stripped,
    "advisor_stripped": _bare,
    "ultra_effort_enter": _effort_mode,
    "ultra_effort_exit": _bare,
    "advisor_tool": _advisor_tool,
    "command_permissions": _command_permissions,
    "invoked_skills": _invoked_skills,
    **{name: _plain() for name in ("prompt_snapshot", "deferred_tools_delta", "environment", "session_context",
                                    "instructions", "credential_org", "agent_listing_delta", "auto_mode",
                                    "skill_listing", "task_status")},
}
_QUEUED_KEYS = {"type", "commandMode", "delivery_id", "origin", "prompt", "runId", "source_uuid", "timestamp", "usage",
                "humanTurn", "isMeta", "reminderId", "imagePasteIds", "toolsUsedDelta"}
_QUEUED_TYPES = {"delivery_id": str, "runId": str, "source_uuid": str, "timestamp": str, "isMeta": bool,
                 "reminderId": (str, int), "humanTurn": (bool, str, int), "usage": (dict, int, float, str, type(None)),
                 "imagePasteIds": list, "toolsUsedDelta": list}


def _queued_command(base, a, state):
    """A prompt queued mid-turn. Only a human prompt that is not meta is a user turn; all else is runtime."""
    status, origin, extra = _origin(a.get("origin"))
    mode, prompt = a.get("commandMode"), a.get("prompt")
    if (status == "bad" or not set(a) <= _QUEUED_KEYS or mode not in (None, "prompt", "task-notification")
            or not isinstance(prompt, (str, list)) or not _typed(a, **_QUEUED_TYPES)
            or ("imagePasteIds" in a and _ints(a["imagePasteIds"]) is None)
            or ("toolsUsedDelta" in a and _labels(a["toolsUsedDelta"]) is None)):
        yield _unsupported(base, "[unsupported_attachment:queued_command]", attachment_type="queued_command")
        return
    is_meta = a.get("isMeta", False)
    base = {**base, **extra, "attachment_type": "queued_command", "queued_command": True}
    for key, name in (("delivery_id", "delivery_id"), ("runId", "run_id"), ("reminderId", "reminder_id")):
        if key in a:
            base[name] = a[key]
    # Pasted images are blocks of the prompt (placeholders); the ids and the tools a finished task used are counts only.
    for key, name in (("imagePasteIds", "image_paste_count"), ("toolsUsedDelta", "tools_used_delta_count")):
        if key in a:
            base[name] = len(a[key])
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
                    "informational", "scheduled_task_fire", "agents_killed"}
_QUEUE_OPERATIONS = {"enqueue", "dequeue", "remove", "popAll"}
_METADATA = {"mode", "permission-mode", "ai-title", "agent-name", "pr-link", "file-history-snapshot",
             "file-history-delta", "cost-state", "atis-latch", "last-prompt", "continued-in",
             "api-request", "api-request-blob", "api-request-shape", "fork-context-ref", "session-owner", "tools-used"}
_SESSION_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_ENVELOPE_KEYS = {"type", "sessionId", "timestamp", "version", "uuid", "parentUuid", "cwd",
                  "agentId", "isSidechain", "userType", "promptId", "requestId"}
# Keys the host stamps on most records besides the envelope (all text). ``session_id`` repeats ``sessionId``.
_HOST_KEYS = {"entrypoint": str, "gitBranch": str, "slug": str, "session_id": str}
_API_REQUEST_KEYS = _ENVELOPE_KEYS | {"id", "base", "tail", "keep", "params", "cacheControl",
                                      "querySource", "requestDigest", "shapeHash"}
_API_REQUEST_BLOB_KEYS = _ENVELOPE_KEYS | {"hash", "message"}
_API_REQUEST_SHAPE_KEYS = _ENVELOPE_KEYS | {"shape", "shapeHash"}
_FORK_CONTEXT_REF_KEYS = _ENVELOPE_KEYS | {"contextLength", "parentLastUuid", "parentSessionId"}
_SESSION_OWNER_KEYS = _ENVELOPE_KEYS | {"accountUuid", "organizationUuid"}
_TOOLS_USED_KEYS = _ENVELOPE_KEYS | {"tools"}
_AGENTS_KILLED_KEYS = _ENVELOPE_KEYS | set(_HOST_KEYS) | {"subtype", "level", "isMeta"}
_POP_ALL_KEYS = _ENVELOPE_KEYS | set(_HOST_KEYS) | {"operation", "content"}
_CACHE_SETTING_KEYS = {"type", "ttl", "scope"}


def _cache_breakpoints(value) -> bool:
    """The list form of an api-request's ``cacheControl``: where in the request each cache breakpoint sits (a path of
    field names and indices) and its setting (labels). Nothing of it is kept."""
    return isinstance(value, list) and all(
        isinstance(item, dict) and set(item) == {"path", "value"}
        and isinstance(item["path"], list) and bool(item["path"])
        and all(_is_int(step) or _label(step) is not None for step in item["path"])
        and isinstance(item["value"], dict) and "type" in item["value"] and set(item["value"]) <= _CACHE_SETTING_KEYS
        and all(_label(setting) is not None for setting in item["value"].values())
        for item in value)


def _scheduled_task_fire(record):
    """Small identifiers of a scheduled-task announcement, or None when a payload field is not text.

    ``prompt`` (the scheduled automation text) and ``content`` are validated and never indexed: like every other
    system payload they are runtime text, not something the user said. Only short identifiers survive, and an
    odd one is dropped rather than making the announcement unsupported.
    """
    if any(key in record and not isinstance(record[key], str)
           for key in ("prompt", "content", "cron", "cronKind", "taskId", "taskKind")):
        return None
    found = {name: record[key] for key, name in (("taskId", "task_id"), ("taskKind", "task_kind"),
                                                 ("cronKind", "cron_kind")) if _label(record.get(key))}
    if isinstance(record.get("cron"), str) and _CRON.fullmatch(record["cron"]):
        found["cron"] = " ".join(record["cron"].split())
    return found


def _system(base, record, state):
    subtype, level = record.get("subtype"), record.get("level", "")
    if subtype not in _SYSTEM_SUBTYPES or not isinstance(level, str) or not isinstance(record.get("isMeta", False), bool):
        yield _unsupported(base, f"[unsupported_system:{_label(subtype) or '?'}] {_keys(record)}")
        return
    if subtype == "agents_killed" and not (
            set(record) <= _AGENTS_KILLED_KEYS and _typed(record, **_HOST_KEYS)
            and record.get("session_id", record.get("sessionId")) == record.get("sessionId")):
        # The host stopped the session's background agents. The notice carries no text; any other field is unknown.
        yield _unsupported(base, "[unsupported_system:agents_killed] " + _keys(record))
        return
    if subtype == "scheduled_task_fire":
        found = _scheduled_task_fire(record)
        if found is None:
            yield _unsupported(base, "[unsupported_system:scheduled_task_fire] " + _keys(record))
            return
        text = "[system:scheduled_task_fire]" + "".join(
            f' {name}="{value}"' if name == "cron" else f" {name}={value}" for name, value in found.items())
        yield _runtime(base, text, "scheduled_task_fire", system_subtype=subtype, **found,
                       **({"level": _label(level)} if _label(level) else {}))
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
    if operation not in _QUEUE_OPERATIONS or (operation == "popAll" and not (
            set(record) <= _POP_ALL_KEYS and _typed(record, content=str, **_HOST_KEYS))):
        # ``popAll`` (the queue was emptied at once; its ``content`` is the queued text, never indexed) is checked
        # strictly; the older operations keep their original check.
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
    if kind == "session-owner":
        # The account and organization the session runs under. Identifiers only: validated, never kept in the text or
        # in meta (they say nothing about what anyone asked for or did).
        if not (set(record) <= _SESSION_OWNER_KEYS and all(
                isinstance(record.get(key), str) and _UUID.fullmatch(record[key])
                for key in ("accountUuid", "organizationUuid"))):
            yield _unsupported(base, f"[unsupported_claude_event:session-owner] {_keys(record)}")
            return
    if kind == "tools-used":
        # Bookkeeping of the tools the session has used so far; the calls themselves are the evidence. Only the count is
        # kept, so a tool name here can never match a contract.
        if not (set(record) <= _TOOLS_USED_KEYS and _labels(record.get("tools")) is not None):
            yield _unsupported(base, f"[unsupported_claude_event:tools-used] {_keys(record)}")
            return
        extra["tool_count"] = len(record["tools"])
    if kind == "api-request":
        cache_control = record.get("cacheControl")
        if not (set(record) <= _API_REQUEST_KEYS
                and _typed(record, id=str, querySource=str, requestDigest=str, shapeHash=str,
                           base=(str, dict, list, int, type(None)), tail=(str, dict, list, int, type(None)),
                           keep=(int, list, dict, bool, str, type(None)), params=(dict, type(None)),
                           cacheControl=(dict, str, bool, list, type(None)))
                and (not isinstance(cache_control, list) or _cache_breakpoints(cache_control))):
            yield _unsupported(base, f"[unsupported_claude_event:api-request] {_keys(record)}")
            return
        if _label(record.get("querySource")):
            extra["query_source"] = record["querySource"]
        if isinstance(record.get("id"), str) and _SESSION_REF.fullmatch(record["id"]):
            extra["api_request_id"] = record["id"]
    if kind == "api-request-blob":
        # ``message`` duplicates turn payloads and must never be indexed as user/assistant text.
        if not (set(record) <= _API_REQUEST_BLOB_KEYS
                and isinstance(record.get("hash"), str) and bool(record["hash"])
                and isinstance(record.get("message"), (dict, list, str))):
            yield _unsupported(base, f"[unsupported_claude_event:api-request-blob] {_keys(record)}")
            return
    if kind == "api-request-shape":
        if not (set(record) <= _API_REQUEST_SHAPE_KEYS
                and isinstance(record.get("shapeHash"), str) and bool(record["shapeHash"])
                and isinstance(record.get("shape"), (dict, list, str))):
            yield _unsupported(base, f"[unsupported_claude_event:api-request-shape] {_keys(record)}")
            return
    if kind == "fork-context-ref":
        parent_sid = record.get("parentSessionId")
        parent_last = record.get("parentLastUuid")
        ctx_len = record.get("contextLength")
        if not (set(record) <= _FORK_CONTEXT_REF_KEYS
                and isinstance(parent_sid, str) and _SESSION_REF.fullmatch(parent_sid)
                and (not state.sub or state.parent is None or parent_sid == state.parent)
                and (parent_last is None or (isinstance(parent_last, str) and _SESSION_REF.fullmatch(parent_last)))
                and (ctx_len is None or (_is_int(ctx_len) and ctx_len >= 0))):
            yield _unsupported(base, f"[unsupported_claude_event:fork-context-ref] {_keys(record)}")
            return
        extra["parent_session_id"] = parent_sid
        if parent_last is not None:
            extra["parent_last_uuid"] = parent_last
        if ctx_len is not None:
            extra["context_length"] = ctx_len
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
            if has_invalid_utf8(raw):
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
