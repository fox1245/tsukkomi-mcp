"""Antigravity transcript discovery; only official brain/session log layouts."""
from __future__ import annotations

import array
import hashlib
import json
import os
import re
from pathlib import Path

from self_directing_mcp.request_control import check_request
from self_directing_mcp.session_events import (
    _identity, as_text, iter_raw_lines, make_chunk, parse_error, same_identity)

# v7: host NUL padding in front of a record, and the IDE's typed result and context steps.
PARSER_VERSION = 7
DISPLAY_NAME, FULL_NAME = "transcript.jsonl", "transcript_full.jsonl"
KNOWN_STATUS = frozenset({"DONE", "RUNNING", "ERROR", "INVALID", "CANCELED"})
# The result of a tool call. The CLI writes every result as GENERIC; the IDE writes one step type per tool. Both
# are linked to their call only through a host receipt (no call ids exist) and follow the same evidence rules.
IDE_RESULT_TYPES = frozenset({"RUN_COMMAND", "VIEW_FILE", "CODE_ACTION", "MCP_TOOL", "LIST_DIRECTORY"})
RESULT_TYPES = frozenset({"GENERIC"}) | IDE_RESULT_TYPES
# Result steps that can carry a success verdict: the CLI's exit-code header, the IDE's host ``exit_code``.
SUCCESS_TYPES = frozenset({"GENERIC", "RUN_COMMAND"})
# Content-free context-loading steps of the IDE: runtime metadata, like CHECKPOINT; the value is the label.
CONTEXT_TYPES = {"CONVERSATION_HISTORY": "conversation_history", "KNOWLEDGE_ARTIFACTS": "knowledge_artifacts"}
# Fields a fallback to transcript_full.jsonl is attempted for. thinking, CHECKPOINT, SYSTEM_MESSAGE and
# EPHEMERAL_MESSAGE are context, not evidence: they are kept visible and only counted when truncated.
FALLBACK_FIELDS = {"USER_INPUT": ("content",), "PLANNER_RESPONSE": ("content", "tool_calls"), "GENERIC": ("content",),
                   **{kind: ("content",) for kind in IDE_RESULT_TYPES}}
# (step type, field) pairs a verdict can depend on; the runner keys its partial-observation rules on these.
JUDGED_FIELDS = frozenset({("USER_INPUT", "content"), ("PLANNER_RESPONSE", "content"),
                           ("PLANNER_RESPONSE", "tool_calls")} | {(kind, "content") for kind in RESULT_TYPES})
_ID = re.compile(r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\Z")
# ``iter_raw_lines`` decodes with surrogateescape: a byte that is not UTF-8 becomes one of these code points.
_UNDECODABLE = re.compile("[\udc80-\udcff]")


def has_undecodable(raw: str) -> bool:
    """Does the line hold a byte that was not valid UTF-8? Pure ASCII (almost every record) needs no search."""
    return not raw.isascii() and _UNDECODABLE.search(raw) is not None


class PathTraversalError(ValueError):
    pass


def split_host_padding(raw):
    """``(body, nul_count)`` for a line (``str`` or ``bytes``): a leading contiguous run of NULs is removed.

    Only a run at the very start counts. NULs anywhere else stay in ``body``, where they keep the line from
    parsing; the caller decides what a padded line must still contain.
    """
    body = raw.lstrip("\x00" if isinstance(raw, str) else b"\x00")
    return body, len(raw) - len(body)


def load_line(raw):
    """``(record, padding)`` for one JSONL line, tolerating the host's leading NUL run (see ``split_host_padding``).

    A line that is only padding (nothing but whitespace after the run) is an error, as is any remainder that is
    not JSON: the padding is never evidence that a record exists, and neighbouring step indexes explain nothing.
    """
    body, padding = split_host_padding(raw)
    if padding and not body.strip():
        raise ValueError("host padding without a record")
    return json.loads(body), padding


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


class _FullTranscript:
    """The sibling transcript_full.jsonl of a display log, indexed in one bounded-memory streaming pass.

    The pass keeps, per complete line, only (step_index, start, end, 8-byte digest) in three compact arrays and
    one step -> slot dict; no record text is retained, so memory is O(records) plus one record in flight, never
    O(file). A fetch re-reads exactly the indexed byte range from a file with the same identity and requires the
    same digest: the bytes consumed are the bytes indexed, or the fallback fails (``full_changed``) and the
    partial-observation rules apply. Nothing about the full file is persisted; see ``full_transcript_size`` for
    how a failed fallback is retried.
    """

    def __init__(self, display: Path, sid: str):
        self.sid = sid
        self.path = Path(display).resolve().parent / FULL_NAME  # same directory only
        self.size: int | None = None
        self.max_step = -1
        self.failure: tuple[str, bool] | None = None  # (reason, retryable): the sibling cannot be used at all
        self._identity = None
        self._slots: dict[int, int] = {}
        self._starts, self._ends = array.array("q"), array.array("q")
        self._sums = array.array("Q")
        self._duplicates: set[int] = set()
        self._built = False

    def _build(self) -> None:
        self._built = True
        path = self.path
        try:
            if path.is_symlink():
                self.failure = ("full_unavailable", False)
            elif not path.is_file():
                self.failure = ("full_missing", True)
            elif path.resolve() != path or peek_session_id(path) != self.sid:
                self.failure = ("full_unavailable", False)
            if self.failure:
                return
            self.size = path.stat().st_size  # recorded even if the pass fails: see full_transcript_size()
            with path.open("rb") as stream:
                before = os.fstat(stream.fileno())
                self._identity, self.size = _identity(before), before.st_size
                position = 0
                while position < self.size:
                    check_request()
                    raw = stream.readline(self.size - position)
                    if not raw.endswith(b"\n"):
                        break  # an unfinished final record is not indexed
                    start, position = position, position + len(raw)
                    self._add(start, position, raw)
                after = os.fstat(stream.fileno())
            if not same_identity(_identity(after), self._identity) or after.st_size < self.size:
                self.failure = ("full_changed", True)
        except OSError:
            self.failure = ("full_unavailable", True)

    def _add(self, start: int, end: int, raw: bytes) -> None:
        if not raw.strip():
            return
        try:
            row, _padding = load_line(raw)
        except ValueError:
            return  # an unreadable line cannot be matched to a step
        step = row.get("step_index") if isinstance(row, dict) else None
        if type(step) is not int or step < 0:
            return
        self.max_step = max(self.max_step, step)
        if step in self._slots or step in self._duplicates:
            self._duplicates.add(step)  # ambiguous: never restore from a repeated step
            self._slots.pop(step, None)
            return
        self._slots[step] = len(self._starts)
        self._starts.append(start)
        self._ends.append(end)
        self._sums.append(int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big"))

    def fetch(self, step: int, kind: str, source, status=None) -> tuple[dict | None, dict | None, tuple[str, bool] | None]:
        """(record, byte-range reference, None) or (None, None, (reason, retryable))."""
        if not self._built:
            self._build()
        if self.failure:
            return None, None, self.failure
        if step in self._duplicates:
            return None, None, ("full_duplicate_step", False)
        slot = self._slots.get(step)
        if slot is None:
            # Not there yet only if the full log has not reached this step; a gap behind its end stays a gap.
            return None, None, ("full_missing_step", step > self.max_step)
        start, end = self._starts[slot], self._ends[slot]
        try:
            with self.path.open("rb") as stream:
                if not same_identity(_identity(os.fstat(stream.fileno())), self._identity):
                    return None, None, ("full_changed", True)
                stream.seek(start)
                raw = stream.read(end - start)
        except OSError:
            return None, None, ("full_unavailable", True)
        if len(raw) != end - start or int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big") != self._sums[slot]:
            return None, None, ("full_changed", True)
        try:
            row, _padding = load_line(raw)
        except ValueError:
            return None, None, ("full_changed", True)
        if (not isinstance(row, dict) or row.get("step_index") != step or row.get("type") != kind
                or (source is not None and row.get("source") not in (None, source))
                or (status is not None and row.get("status") not in (None, status))):
            # Same step, type, source and status: a later state of the step (RUNNING -> DONE) is not this record.
            return None, None, ("full_step_mismatch", False)
        return row, {"byte_start": start, "byte_end": end, "sha256": hashlib.sha256(raw).hexdigest()[:16]}, None


def full_transcript_size(path: Path) -> int | None:
    """Size of the usable sibling full log, or None. Recorded on unresolved fallbacks to detect a retry.

    A fallback that failed for a transient reason (full log missing or not yet at this step) is retried exactly
    when this value differs from the one recorded with the chunk; permanent failures are never retried.
    """
    path = Path(path)
    if path.name != DISPLAY_NAME:
        return None
    sibling = path.resolve().parent / FULL_NAME
    try:
        if sibling.is_symlink() or not sibling.is_file() or sibling.resolve() != sibling:
            return None
        return sibling.stat().st_size
    except OSError:
        return None


_DATA_URI = re.compile(r"data:[A-Za-z0-9.+/-]+(?:;[A-Za-z0-9=._+-]+)*;base64,[A-Za-z0-9+/=_-]{16,}")
_LONG_BASE64 = re.compile(r"[A-Za-z0-9+/]{4096,}={0,2}")
_MIME = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+/_-]{0,63}")


def _scrub(text: str) -> str:
    """Binary payloads are never indexed: data URIs and very long base64 runs become placeholders."""
    if "base64," in text:
        text = _DATA_URI.sub(lambda m: f"[blob omitted: {len(m.group())} chars]", text)
    if len(text) >= 4096:
        # Repeated filler (a long run of one or two characters) is text, not a payload.
        text = _LONG_BASE64.sub(lambda m: f"[blob omitted: {len(m.group())} chars]" if len(set(m.group())) >= 16
                                else m.group(), text)
    return text


def _media(value) -> list[str]:
    """One placeholder per media item: type, size and a digest, never the payload."""
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("invalid media")
    out = []
    for index, item in enumerate(value):
        mime = next((item[key] for key in ("mime_type", "mimeType", "content_type", "type")
                     if isinstance(item.get(key), str) and _MIME.fullmatch(item[key])), "unknown")
        chars = sum(len(text) for text in item.values() if isinstance(text, str))
        digest = hashlib.sha256(json.dumps(item, sort_keys=True, default=str).encode()).hexdigest()[:12]
        out.append(f"[media #{index} type={mime} chars={chars} sha256={digest}]")
    return out


def _marks(record: dict) -> list[str]:
    marks = record.get("truncated_fields")
    if marks is None:
        return []
    if not isinstance(marks, list) or any(not isinstance(mark, str) or not mark for mark in marks):
        raise ValueError("invalid truncated_fields")
    return list(dict.fromkeys(marks))


def _decode_calls(calls, *, encoded: bool, lenient: bool) -> list[tuple[str, dict]]:
    """Tool calls as (name, decoded arguments). ``lenient`` (a record flagged as truncated) keeps what is
    visible: an undecodable or malformed piece is kept as its raw text or skipped instead of failing the record."""
    if calls is None or (lenient and not isinstance(calls, list)):
        return []
    if not isinstance(calls, list):
        raise ValueError("invalid tool_calls")
    out = []
    for call in calls:
        if (not isinstance(call, dict) or not isinstance(call.get("name"), str) or not call["name"]
                or not isinstance(call.get("args"), dict)):
            if lenient:
                continue
            raise ValueError("invalid tool call")
        args = {}
        for key, value in call["args"].items():
            if not encoded:
                args[key] = value
            elif isinstance(value, str):
                try:
                    args[key] = json.loads(value)
                except ValueError:
                    if not lenient:
                        raise
                    args[key] = value
            elif lenient:
                args[key] = value
            else:
                raise ValueError("invalid encoded tool arguments")
        out.append((call["name"], args))
    return out


def _restore(field: str, visible, row: dict):
    """(value, None) when ``row`` holds a complete, longer value of ``field``; else (None, (reason, retryable))."""
    marks = row.get("truncated_fields")
    if marks is not None and (not isinstance(marks, list) or any(not isinstance(m, str) for m in marks) or field in marks):
        return None, ("full_truncated", False)
    if field not in row:
        return None, ("full_field_missing", False)
    value = row[field]
    if field == "tool_calls":
        try:
            value = _decode_calls(value, encoded=False, lenient=False)
        except ValueError:
            return None, ("full_field_invalid", False)
        seen = [name for name, _args in visible]
        if [name for name, _args in value][:len(seen)] != seen:
            return None, ("full_prefix_mismatch", False)  # the same step number, but not the same calls
        longer = len(as_text(value)) > len(as_text(visible))
    elif isinstance(value, str):
        head = min(16, len(visible))
        if value[:head] != visible[:head]:
            return None, ("full_prefix_mismatch", False)  # guards a renumbered log: same step and type, other text
        longer = len(value) > len(visible)
    else:
        return None, ("full_field_invalid", False)
    # The host flagged the display value as cut; a full value that is not longer restores nothing.
    return (value, None) if longer else (None, ("full_not_longer", False))


class _Truncation:
    """Outcome of the fallback for one record: restored values, fields still cut, and why a fallback failed."""

    def __init__(self, record, kind, source, step, marks, get_full, visible):
        self.kind, self.step = kind, step
        self.values: dict = {}
        self.restored: list[str] = []
        self.still = list(marks)
        self.failures: dict[str, tuple[str, bool]] = {}
        self.ref = None
        self.full_size = None
        wanted = [field for field in marks if field in FALLBACK_FIELDS.get(kind, ())]
        if not wanted:
            return
        full = get_full()
        if full is None:
            self.failures = {field: ("source_is_full", False) for field in wanted}
            return
        row, self.ref, failure = full.fetch(step, kind, source, record.get("status"))
        self.full_size = full.size
        for field in wanted:
            why = failure
            if row is not None:
                value, why = _restore(field, visible[field], row)
                if why is None:
                    self.values[field] = value
                    self.restored.append(field)
                    self.still.remove(field)
            if why is not None:
                self.failures[field] = why

    def meta(self, own=None) -> dict:
        still = [f for f in self.still if own is None or f in own]
        restored = [f for f in self.restored if own is None or f in own]
        if not still and not restored:
            return {}
        out: dict = {}
        if still:
            out["truncated_fields"] = still
        if restored:
            out.update(restored_fields=restored, full_ref=self.ref)
        open_judged = any((self.kind, f) in JUDGED_FIELDS for f in still)
        out["source"] = "full" if restored and not open_judged else "display"
        failed = [self.failures[f] for f in still if f in self.failures]
        if failed:
            out["fallback"] = {"reason": failed[0][0], "retryable": any(retry for _reason, retry in failed),
                               "full_size": self.full_size}
        return out


_EXIT_HEADER = re.compile(r"\ACreated At: [^\n]+\nCompleted At: [^\n]+\n\n"
                          r"The command exited with code (-?\d+)\.\n(?:Output|Stdout):\n")


def parse_session_chunks(path: Path, *, session_id=None, start_byte=0, known_hashes=None, receipts=None, sink=None,
                         uses=None):
    """Read the host's display JSONL, not lifecycle hook payloads.

    CLI 1.2.9 emits JSON-encoded values in tool_calls[].args and omits call IDs.
    Missing IDs remain missing; step order alone is not proof of result linkage.

    Truncated fields (``truncated_fields``) fall back to the same step of the sibling full log; whatever stays
    cut keeps its visible text and is marked in the chunk meta. GENERIC results with no host receipt are
    ``legacy_unlinked`` only when they precede this conversation's first receipt (or it has none); a result after
    its first receipt that lacks one stays an unsupported, current-session defect.

    ``uses`` (a set) receives the step of every result record whose output was derived from ``receipts``: exactly the
    records an index would have to rewrite if the receipt of that step changed. Together with ``first_receipt``
    (which decides ``legacy_unlinked``) that is everything the parse reads from ``receipts``.
    """
    sid = peek_session_id(path)
    if sid is None or (session_id is not None and sid != session_id.lower()):
        raise ValueError("AGY transcript session identity mismatch")
    path = Path(path)
    is_full = path.name == FULL_NAME
    receipts = receipts or {}
    first_receipt = min(receipts) if receipts else None
    resolved = str(path.resolve())  # compared with every receipt's path; one lookup, not one per result
    siblings: list[_FullTranscript] = []

    def get_full():
        if is_full:
            return None
        if not siblings:
            siblings.append(_FullTranscript(path, sid))
        return siblings[0]

    chunks, offset = ([] if sink is None else sink), start_byte
    for line, start, end, raw in iter_raw_lines(path, start_byte=start_byte):
        offset = end
        if not raw.strip():
            continue
        try:
            if has_undecodable(raw):
                raise ValueError("invalid UTF-8")
            record, padding = load_line(raw)
            if not isinstance(record, dict):
                raise ValueError("transcript record is not an object")
            step, source, kind = record.get("step_index"), record.get("source"), record.get("type")
            if type(step) is not int or step < 0:
                raise ValueError("missing native step index")
            timestamp = record.get("created_at")
            if timestamp is not None and not isinstance(timestamp, str):
                raise ValueError("invalid timestamp")
            status = record.get("status")
            if status is not None and not isinstance(status, str):
                raise ValueError("invalid status")
            base = {"event_type": kind, "step_index": step, "status": status}
            if padding:
                base["host_padding_bytes"] = padding  # a NUL run the host wrote before this record; see load_line
            content = record.get("content", "")
            if content is None:
                content = ""
            if not isinstance(content, str):
                raise ValueError("invalid transcript content")
            marks = _marks(record)
            # Only a tool result's status carries semantics (success, pending); an unrecognized one is unsupported.
            known = kind not in RESULT_TYPES or status is None or status in KNOWN_STATUS

            def truncation(visible=None):
                return _Truncation(record, kind, source, step, marks, get_full, visible or {"content": content})

            if known and kind == "USER_INPUT" and source == "USER_EXPLICIT":
                cut = truncation()
                content = cut.values.get("content", content)
                chunks.append(make_chunk("agy", sid, "turn", "[message:user] " + _scrub(content),
                                         {**base, "role": "user", **cut.meta()}, line, start, end, timestamp=timestamp))
            elif known and kind == "EPHEMERAL_MESSAGE" and source == "SYSTEM_SDK":
                chunks.append(make_chunk("agy", sid, "meta", "[ephemeral_context] " + _scrub(content),
                                         {**base, "role": "runtime", "evidence_origin": "injected_context",
                                          **truncation().meta()}, line, start, end, timestamp=timestamp))
            elif known and kind == "SYSTEM_MESSAGE" and source == "SYSTEM":
                # Observed host-injected Stop continuation is context, not a
                # user instruction, executed action, or successful evidence.
                chunks.append(make_chunk("agy", sid, "meta", "[host_system_context] " + _scrub(content),
                                         {**base, "role": "runtime", "evidence_origin": "host_system_context",
                                          **truncation().meta()}, line, start, end, timestamp=timestamp))
            elif known and kind == "CHECKPOINT" and source == "SYSTEM":
                # A context projection: never a new user instruction, never observed execution.
                chunks.append(make_chunk("agy", sid, "meta", "[checkpoint] " + _scrub(content),
                                         {**base, "role": "runtime", "evidence_origin": "checkpoint",
                                          **truncation().meta()}, line, start, end, timestamp=timestamp))
            elif known and kind == "ERROR_MESSAGE" and source == "SYSTEM":
                error = record.get("error")
                if error is not None and not isinstance(error, str):
                    raise ValueError("invalid error")
                text = "[host_error_message] " + _scrub(error or "") + (("\n" + _scrub(content)) if content else "")
                chunks.append(make_chunk("agy", sid, "meta", text,
                                         {**base, "role": "runtime", "evidence_origin": "host_error_message",
                                          **truncation().meta()}, line, start, end, timestamp=timestamp))
            elif known and kind in CONTEXT_TYPES and source == "SYSTEM":
                # IDE context loading (history, knowledge): a context projection, never a user instruction and
                # never observed execution. The observed records carry no content at all.
                label = CONTEXT_TYPES[kind]
                chunks.append(make_chunk("agy", sid, "meta", f"[{label}]" + (" " + _scrub(content) if content else ""),
                                         {**base, "role": "runtime", "evidence_origin": label, **truncation().meta()},
                                         line, start, end, timestamp=timestamp))
            elif known and kind == "PLANNER_RESPONSE" and source == "MODEL":
                lenient = "tool_calls" in marks
                visible_calls = _decode_calls(record.get("tool_calls", []), encoded=not is_full, lenient=lenient)
                cut = truncation({"content": content, "tool_calls": visible_calls})
                content = cut.values.get("content", content)
                calls = cut.values.get("tool_calls", visible_calls)
                chunks.append(make_chunk("agy", sid, "turn" if content else "meta",
                                         "[message:assistant] " + _scrub(content),
                                         {**base, "role": "assistant" if content else "runtime",
                                          **cut.meta(("content", "thinking"))},
                                         line, start, end, timestamp=timestamp))
                for sub, (name, decoded) in enumerate(calls, 1):
                    # A planner proposal may subsequently be denied or cancelled.
                    # Preserve it for retrieval, never as observed execution.
                    meta = {**base, "role": "runtime", "tool_name": name, "call_id": None,
                            "evidence_origin": "model_proposal"}
                    if "tool_calls" in cut.restored:
                        meta["source"] = "full"
                        if sub == 1:
                            meta.update(restored_fields=["tool_calls"], full_ref=cut.ref)
                    chunks.append(make_chunk("agy", sid, "meta", f"[proposed_tool_call:{name}] {as_text(decoded)}",
                                             meta, line, start, end, sub=sub, timestamp=timestamp))
                if "tool_calls" in cut.still:
                    # Calls may be hidden beyond the visible ones, so even zero visible calls leave a marker.
                    chunks.append(make_chunk("agy", sid, "meta", f"[truncated_tool_calls] {len(calls)} visible",
                                             {**base, "role": "runtime", "evidence_origin": "model_proposal_truncated",
                                              **cut.meta(("tool_calls",))},
                                             line, start, end, sub=len(calls) + 1, timestamp=timestamp))
            elif known and kind in RESULT_TYPES and source == "MODEL":
                from self_directing_mcp.agy_receipts import linked_receipt
                cut = truncation()
                content = cut.values.get("content", content)
                error = record.get("error")
                if error is not None and not isinstance(error, str):
                    raise ValueError("invalid error")
                exit_code = record.get("exit_code") if kind == "RUN_COMMAND" else None
                if exit_code is not None and type(exit_code) is not int:
                    raise ValueError("invalid exit_code")
                if uses is not None:
                    uses.add(step)
                observation = linked_receipt(receipts, step, path, start, result_record=record,
                                             require_paired=False, resolved=resolved)
                if (observation and not observation.get("paired") and status == "ERROR"
                        and isinstance(error, str) and error.startswith("tool call denied by pre-tool hook: ")):
                    # Native host rejection plus the anchored Pre receipt proves
                    # non-execution. A denied call has no PostToolUse to pair.
                    denied = json.loads(observation["action"])
                    chunks.append(make_chunk("agy", sid, "meta", "[tool_denied] " + error,
                                             {**base, "role": "runtime", "receipt_id": observation["receipt_id"],
                                              "evidence_origin": "host_rejected_proposal",
                                              "denied_action": {"name": denied.get("name"), "args": denied.get("args")}},
                                             line, start, end, timestamp=timestamp))
                    continue
                if kind == "RUN_COMMAND":
                    # The IDE records the exit status as a host field. A command that is still running has none
                    # yet, so the field is not read then; output prose is never read at all.
                    shaped = exit_code is not None
                    if status == "RUNNING":
                        exit_code = None
                    if exit_code is None:
                        success = None
                    elif exit_code != 0:
                        success = False  # a host-recorded failure, linked or not; it can only make a verdict worse
                    else:
                        success = True if status == "DONE" else None
                else:
                    # Match only the host prefix, never exit-code prose in output.
                    exit_match = _EXIT_HEADER.match(content)
                    shaped = kind != "GENERIC" or exit_match is not None  # typed IDE results carry no header
                    success = int(exit_match.group(1)) == 0 if exit_match and status == "DONE" else None
                if status in ("ERROR", "CANCELED", "INVALID"):
                    success = False  # never success evidence; the call attempt itself is preserved elsewhere
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
                    elif kind == "GENERIC" and name != "run_command":
                        # An empty hook error is tool transport success, not a
                        # proof that an arbitrary tool fulfilled its objective.
                        success = None
                    elif kind != "GENERIC" and (kind != "RUN_COMMAND" or name != "run_command"):
                        success = False if success is False else None  # a typed result stays a failure, never a success
                extra = {}
                if kind == "RUN_COMMAND" and success is True and not receipt:
                    success, extra["success_withheld"] = None, "unlinked"  # a host exit code is no proof of which call ran
                if success is True and "content" in cut.still:
                    success, extra["success_withheld"] = None, "truncated_content"  # a cut output proves nothing
                pending = status == "RUNNING"
                if receipt:
                    linkage = "host_step_receipt"
                elif pending:
                    linkage = "pending"
                elif first_receipt is None or step < first_receipt:
                    linkage = "legacy_unlinked"  # predates this conversation's receipts; never a fabricated link
                else:
                    linkage = "native_log_omits_call_id"
                unsupported = (not shaped and status in (None, "DONE")) if receipt else linkage == "native_log_omits_call_id"
                if pending:
                    extra["pending"] = True
                placeholders = _media(record.get("media"))
                if placeholders:
                    extra["media_count"] = len(placeholders)
                text = "[tool_result] " + _scrub(content)
                if error:
                    text += "\n[error] " + _scrub(error)
                for placeholder in placeholders:
                    text += "\n" + placeholder
                if kind != "GENERIC":
                    extra["result_type"] = kind  # the IDE's step type; a GENERIC result's event_type already says so
                    if exit_code is not None:
                        extra["exit_code"] = exit_code
                chunks.append(make_chunk("agy", sid, "tool_result", text,
                                         {**base, **provenance, "role": "tool", "call_id": call_id,
                                          "tool_name": name, "success": success, "linkage": linkage,
                                          "unsupported_event": bool(unsupported), **extra, **cut.meta()},
                                         line, start, end, sub=1 if receipt else 0, timestamp=timestamp))
            else:
                chunks.append(make_chunk("agy", sid, "meta", "[unsupported_agy_event] " + _scrub(as_text(record)),
                                         {**base, "role": "runtime", "unsupported_event": True},
                                         line, start, end, timestamp=timestamp))
        except (ValueError, TypeError) as exc:
            chunks.append(parse_error("agy", sid, line, start, end, type(exc).__name__))
    return chunks, offset, sid
