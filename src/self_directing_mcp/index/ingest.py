from __future__ import annotations

import hashlib
from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace as dataclass_replace
import json
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from self_directing_mcp import journal as _journal, reindex_spool, settings_core
from self_directing_mcp._runtime import np

from self_directing_mcp.compressed_source import CompressedSourceError, is_compressed
from self_directing_mcp.audit.shadow import ShadowState
from self_directing_mcp.index import shadow_state as shadow_sql
from self_directing_mcp.index.deferred import DerivedBuffer
from self_directing_mcp.schemas import AuditEvent, Chunk
from self_directing_mcp.security.mask import mask_secrets
from self_directing_mcp.request_control import (
    IndexBusy, RequestStopped, check_request, current_request, publication, request_operation,
)
from self_directing_mcp.session_events import (
    SourceChanged, capture_source, capture_source_fast, observed_source, source_scope,
)
from self_directing_mcp.eval_cost import COSTS
from self_directing_mcp.slice_budget import SliceBudget, current_sizing, slice_scope
from self_directing_mcp.source_blocks import BlockDraft, BlockRecord, BlockTable, FastFallback, is_block_hash


@dataclass(frozen=True)
class AuditEventSnapshot:
    """Detached ordered slim events of one session; see ``SessionStore.capture_audit_events``."""
    session_id: str | None
    provider: str | None
    events: tuple
    count: int

    def list_chunks(self, session_id=None, provider=None):
        if session_id not in (None, self.session_id) or provider not in (None, self.provider):
            return []
        return list(self.events)

    def get_chunk(self, chunk_id):
        return next((event for event in self.events if event.chunk_id == chunk_id), None)

    def chunk_count(self, session_id=None, provider=None):
        if session_id not in (None, self.session_id) or provider not in (None, self.provider):
            return 0
        return self.count


_SCOPE_KINDS = {"tool_call": "tool_call", "tool_result": "tool_result", "message": "turn"}
# Meta keys the audit helper for AGY evidence classes needs on every slim event.
_AGY_AUDIT_META = ("event_type", "step_index", "status", "pending", "linkage", "evidence_origin",
                   "tool_name", "truncated_fields", "denied_action", "result_type")
# Claude Code: a truncation notice names the tool result (``call_id``) the host cut; a tool result carries the
# permission decision the host recorded for its call (a refused call is a rejected proposal); see audit.runner.
# Grok: a call's tool name, a result's status and Grok's own event number, and the host's note that a hook blocked a
# tool (its tool, its source and its number, ``evidence_origin``): what ``audit.hostblocks`` joins.
_GROK_AUDIT_META = ("native_tool_name", "tool_status", "host_event_id", "evidence_origin", "blocked_tool", "denied_by")
_AUDIT_META = {"agy": _AGY_AUDIT_META, "claude": ("truncated_result_for", "call_id", "permission_decision"),
               "grok": _GROK_AUDIT_META}


@dataclass(frozen=True)
class AuditEvidenceSnapshot:
    session_id: str | None
    provider: str | None
    rows: tuple[tuple[str, str], ...]
    count: int
    checklist_raw: bytes | None = None

    def list_chunks(self, session_id=None, provider=None):
        chunks = []
        for _cid, raw in self.rows:
            check_request()
            chunk = Chunk.model_validate_json(raw)
            if ((session_id is None or chunk.session_id == session_id)
                    and (provider is None or chunk.provider == provider)):
                chunks.append(chunk)
        return chunks

    def get_chunk(self, chunk_id):
        check_request()
        return next((Chunk.model_validate_json(raw) for cid, raw in self.rows if cid == chunk_id), None)

    def chunk_count(self, session_id=None, provider=None):
        if session_id not in (None, self.session_id) or provider not in (None, self.provider):
            return 0
        return self.count

    def chunk_ids(self, session_id, provider):
        if session_id == self.session_id and provider == self.provider:
            return {cid for cid, _raw in self.rows}
        return {chunk.chunk_id for chunk in self.list_chunks(session_id, provider)}

    def checklist_items(self, session_id, provider):
        if session_id != self.session_id or provider != self.provider or self.checklist_raw is None:
            return []
        from self_directing_mcp.checklist import ChecklistDocument
        return [item.model_dump(mode="json") for item in ChecklistDocument.model_validate_json(self.checklist_raw).items]

    def event_frames(self, session_id, provider):
        return [_event_frame(order, chunk) for order, chunk in enumerate(self.list_chunks(session_id, provider))]


def _event_frame(order, chunk):
    """The one shape of a normalized event row: what time-series analysis and the graph batches read."""
    meta = chunk.meta or {}
    return {"chunk_id": chunk.chunk_id, "order": order, "kind": chunk.kind, "text": chunk.text,
            "timestamp": chunk.timestamp, "tool_name": meta.get("tool_name"), "success": meta.get("success"),
            "error_text": meta.get("error") or meta.get("error_text")}


_BATCH = 500


def _load_deps(text):
    """The stored receipt dependencies (see agy_receipts.dependencies), or None when absent or unreadable."""
    try:
        deps = json.loads(text) if text else None
        if isinstance(deps, dict) and isinstance(deps.get("steps"), dict):
            deps["steps"] = {int(step): digest for step, digest in deps["steps"].items()}
            return deps
    except (TypeError, ValueError):
        pass
    return None


def _batches(items, size=_BATCH):
    for start in range(0, len(items), size):
        check_request()
        yield items[start:start + size]


def current_parser_version(provider):
    if provider == "codex":
        from self_directing_mcp.codex.parse import PARSER_VERSION
        return PARSER_VERSION
    if provider == "omp":
        from self_directing_mcp.omp import PARSER_VERSION
        return PARSER_VERSION
    if provider == "agy":
        from self_directing_mcp.agy import PARSER_VERSION
        return PARSER_VERSION
    if provider == "claude":
        from self_directing_mcp.claude import PARSER_VERSION
        return PARSER_VERSION
    if provider == "grok":
        from self_directing_mcp.grok import PARSER_VERSION
        return PARSER_VERSION
    return 1


def prefix_digest(path: Path, length: int) -> str:
    observation = observed_source(path)
    if observation is not None:
        return observation.prefix_digest(length)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = length
        while remaining:
            check_request()
            block = stream.read(min(remaining, 1024 * 1024))
            if not block:
                raise ValueError("source truncated during read")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def agy_tally(linkage, pending, media_count, event_type, still, restored, fallback=None, padding=None, sub=0):
    """Counter increments for one AGY event's evidence classes; shared by commit and the scan fallback."""
    from self_directing_mcp.agy import JUDGED_FIELDS
    out = []
    if padding and not sub:
        out.append(("host_padded_records", 1))  # one per padded line: every chunk of it carries the padding
    if linkage == "legacy_unlinked":
        out.append(("legacy_unlinked_results", 1))
    if pending:
        out.append(("running_results", 1))
    if media_count:
        out.append(("media_placeholders", media_count))
    for field in still or ():
        out.append((("unrestored|" if (event_type, field) in JUDGED_FIELDS else "ignored|") + f"{event_type}.{field}", 1))
    for field in restored or ():
        out.append((f"restored|{event_type}.{field}", 1))
    if isinstance(fallback, dict) and fallback.get("retryable"):
        out.append((f"retry|{fallback.get('full_size')}", 1))  # the size of the full log that failed transiently
    return out


def _agy_stats_dict(counter):
    def bucket(prefix):
        return {key.split("|", 1)[1]: count for key, count in sorted(counter.items()) if key.startswith(prefix + "|")}
    return {"legacy_unlinked_results": counter["legacy_unlinked_results"], "running_results": counter["running_results"],
            "media_placeholders": counter["media_placeholders"], "host_padded_records": counter["host_padded_records"],
            "truncation": {"restored_from_full": bucket("restored"), "unrestored": bucket("unrestored"),
                           "ignored": bucket("ignored")}}


class RowSpool:
    """Prepared event rows spooled to disk (memory below 1 MiB); parsers append, publication replays in batches.

    ``track_calls``: also keep the earliest byte of every tool call (``calls``, OMP) which the parse of the next slice of a
    staged re-index needs (``reindex_spool``)."""

    def __init__(self, track_calls=False):
        self._file = tempfile.SpooledTemporaryFile(max_size=1 << 20, mode="w+", encoding="utf-8", newline="\n")
        self.count, self.ids, self.flags, self.tallies = 0, set(), {}, {}
        self.calls = {} if track_calls else None

    @classmethod
    def from_chunks(cls, chunks):
        spool = cls()
        for chunk in chunks:
            spool.append(chunk)
        return spool

    def append(self, chunk):
        check_request()
        row = [chunk.chunk_id, chunk.provider, chunk.session_id, chunk.byte_start or 0,
               chunk.meta.get("sub_index", 0), chunk.model_dump_json()]
        self._file.write(json.dumps(row) + "\n")
        self.count += 1
        self.ids.add(chunk.chunk_id)
        if self.calls is not None and chunk.kind == "tool_call":
            key, start = (chunk.meta.get("call_id"), chunk.meta.get("tool_name")), chunk.byte_start or 0
            self.calls[key] = min(self.calls.get(key, start), start)
        if chunk.meta.get("parse_error") or chunk.meta.get("unsupported_event"):
            self.flags[chunk.chunk_id] = (bool(chunk.meta.get("parse_error")), bool(chunk.meta.get("unsupported_event")))
        if chunk.provider == "agy":
            meta = chunk.meta
            tally = agy_tally(meta.get("linkage"), meta.get("pending"), meta.get("media_count"), meta.get("event_type"),
                              meta.get("truncated_fields"), meta.get("restored_fields"), meta.get("fallback"),
                              meta.get("host_padding_bytes"), meta.get("sub_index", 0))
            if tally:
                self.tallies[chunk.chunk_id] = tally

    def __len__(self):
        return self.count

    def batches(self, size=256):
        self._file.flush()
        self._file.seek(0)
        batch = []
        for line in self._file:
            batch.append(tuple(json.loads(line)))
            if len(batch) == size:
                check_request()
                yield batch
                batch = []
        if batch:
            yield batch

    def blocks(self, limit=1 << 20):
        """The rows as they are in the file, ``(text as bytes, rows)`` in blocks of about ``limit`` bytes (a row is one
        ASCII line): what a staged re-index copies into its spool without reading a row."""
        self._file.flush()
        self._file.seek(0)
        pending, size = [], 0
        for line in self._file:
            pending.append(line)
            size += len(line)
            if size >= limit:
                yield "".join(pending).encode("ascii"), len(pending)
                pending, size = [], 0
        if pending:
            yield "".join(pending).encode("ascii"), len(pending)

    def close(self):
        self._file.close()


_flights, _flights_gate = {}, threading.Lock()
_embed_flights, _embed_flights_gate = {}, threading.Lock()
_progress_map, _progress_lock = {}, threading.Lock()


def _format_progress_entry(entry: dict, now: float) -> dict:
    res = {k: v for k, v in entry.items() if not k.startswith("_")}
    elapsed_ms = round(max(0.0, (now - entry["_started_mono"]) * 1000.0), 1)
    res["elapsed_ms"] = elapsed_ms
    eta_ms = None
    if res.get("stage") == "embedding" and res.get("embedded_total", 0) > 0 and res.get("embedded_done", 0) > 0:
        rem = max(0, res["embedded_total"] - res["embedded_done"])
        embed_start = entry.get("_embed_started_mono", entry["_started_mono"])
        embed_elapsed_ms = max(0.0, (now - embed_start) * 1000.0)
        eta_ms = round((embed_elapsed_ms / res["embedded_done"]) * rem, 1)
    elif res.get("bytes_total", 0) > 0 and res.get("bytes_done", 0) > 0 and res["bytes_done"] < res["bytes_total"]:
        rem = max(0, res["bytes_total"] - res["bytes_done"])
        eta_ms = round((elapsed_ms / res["bytes_done"]) * rem, 1)
    res["eta_ms"] = eta_ms
    return res


def update_session_progress(
    session_id: str | None,
    provider: str | None,
    *,
    flight_key=None,
    owner: str | None = None,
    stage: str | None = None,
    bytes_done: int | None = None,
    bytes_total: int | None = None,
    chunks_done: int | None = None,
    chunks_total: int | None = None,
    embedded_done: int | None = None,
    embedded_total: int | None = None,
    started_at: float | None = None,
    clear: bool = False,
    force: bool = False,
) -> dict | None:
    keys = []
    if session_id and provider:
        sid_norm = session_id.lower() if provider in ("codex", "agy") else session_id
        keys.append(("session", provider, sid_norm))
    if flight_key is not None:
        keys.append(("flight", flight_key))
    if not keys:
        return None
    now = time.monotonic()
    with _progress_lock:
        existing = None
        for k in keys:
            if k in _progress_map:
                existing = _progress_map[k]
                break
        if clear:
            if existing is not None and existing.get("stage") == "embedding" and not force:
                return None
            for k in keys:
                cur = _progress_map.get(k)
                if cur is not None and owner is not None and cur.get("_owner") not in (None, owner):
                    continue
                _progress_map.pop(k, None)
            return None
        if existing is not None and existing.get("stage") == "embedding" and stage not in (None, "embedding") and not force:
            return _format_progress_entry(existing, now)
        base = dict(existing) if existing is not None else {
            "stage": stage or "syncing",
            "bytes_done": 0,
            "bytes_total": 0,
            "chunks_done": 0,
            "chunks_total": 0,
            "embedded_done": 0,
            "embedded_total": 0,
            "_started_mono": started_at if started_at is not None else now,
        }
        if session_id:
            base["session_id"] = session_id
        if provider:
            base["provider"] = provider
        if owner is not None:
            base["_owner"] = owner
        base["_updated_mono"] = now
        if stage is not None:
            if stage == "embedding" and (base.get("stage") != "embedding" or "_embed_started_mono" not in base or embedded_done == 0):
                base["_embed_started_mono"] = now
            base["stage"] = stage
        if bytes_done is not None:
            base["bytes_done"] = max(0, int(bytes_done))
        if bytes_total is not None:
            base["bytes_total"] = max(0, int(bytes_total))
        if chunks_done is not None:
            base["chunks_done"] = max(0, int(chunks_done))
        if chunks_total is not None:
            base["chunks_total"] = max(0, int(chunks_total))
        if embedded_done is not None:
            base["embedded_done"] = max(0, int(embedded_done))
        if embedded_total is not None:
            base["embedded_total"] = max(0, int(embedded_total))
        formatted = _format_progress_entry(base, now)
        base["elapsed_ms"] = formatted["elapsed_ms"]
        base["eta_ms"] = formatted["eta_ms"]
        for k in keys:
            cur = _progress_map.get(k)
            if (
                cur is not None
                and k[0] == "session"
                and owner is not None
                and cur.get("_owner") not in (None, owner)
                and cur.get("stage") == "embedding"
                and stage != "embedding"
            ):
                continue
            _progress_map[k] = dict(base)
        return formatted


def get_session_progress(session_id: str | None, provider: str | None = None) -> dict | None:
    if not session_id:
        return None
    now = time.monotonic()
    with _progress_lock:
        if provider:
            sid_norm = session_id.lower() if provider in ("codex", "agy") else session_id
            entry = _progress_map.get(("session", provider, sid_norm))
            return _format_progress_entry(entry, now) if entry is not None else None
        matches = [
            entry
            for k, entry in _progress_map.items()
            if k[0] == "session"
            and (
                k[2] == session_id
                or (k[1] in ("codex", "agy") and k[2] == session_id.lower())
            )
        ]
        if matches:
            best = max(matches, key=lambda e: e.get("_updated_mono", e.get("_started_mono", 0.0)))
            return _format_progress_entry(best, now)
    return None


def get_flight_progress(flight_key) -> dict | None:
    now = time.monotonic()
    with _progress_lock:
        entry = _progress_map.get(("flight", flight_key))
        if entry is None and isinstance(flight_key, tuple) and len(flight_key) >= 4 and flight_key[2] == "session":
            entry = _progress_map.get(("session", flight_key[1], flight_key[3]))
        if entry is not None:
            return _format_progress_entry(entry, now)
    return None


class FlightBusy(IndexBusy):
    """The session is being synced by another request and the bounded wait of a request that may slice ran out.

    Private to the sync path. ``engine._refresh_evidence`` judges on the prefix that is committed when it meets this
    (coverage incomplete, issue ``sync_in_progress``); a caller that lets it pass is answered with an incomplete
    response whose reason is ``sync_in_progress``, a planned stop, not a deadline overrun.
    """
    reason = "sync_in_progress"

    def __init__(self, message: str = "", *, progress: dict | None = None):
        super().__init__()
        if message:
            self.args = (message,)
        self.progress = progress


@contextmanager
def session_flight(key, *, max_wait=None):
    """Single-flight per (index, provider, session): at most one capture/parse/publish at a time."""
    with _flights_gate:
        entry = _flights.setdefault(key, [threading.Lock(), 0])
        entry[1] += 1
    try:
        patience = None if max_wait is None else time.monotonic() + max_wait
        while not entry[0].acquire(timeout=0.05 if patience is None else max(0.0, min(0.05, patience - time.monotonic()))):
            check_request()
            if patience is not None and time.monotonic() >= patience:
                raise FlightBusy(progress=get_flight_progress(key))
        try:
            check_request()
            yield
        finally:
            entry[0].release()
    finally:
        with _flights_gate:
            entry[1] -= 1
            if not entry[1] and _flights.get(key) is entry:
                del _flights[key]


_embed_commit_seq = 0


def _notify_embed_flight_commit(key, state_snapshot: tuple, *, request_embed: bool = False) -> int:
    global _embed_commit_seq
    with _embed_flights_gate:
        _embed_commit_seq += 1
        seq = _embed_commit_seq
        if key is not None:
            entry = _embed_flights.get(key)
            if entry is not None:
                if seq >= entry[7]:
                    entry[4] = state_snapshot
                    entry[7] = seq
                if entry[2]:
                    entry[3] = True
                    if request_embed:
                        entry[5] = True
        return seq


@contextmanager
def _embed_flight(key, *, blocking: bool = True, state_snapshot: tuple | None = None, commit_seq: int = 0):
    """Separate single-flight for dense embedding so remote embedding never blocks canonical/sparse syncs."""
    with _embed_flights_gate:
        # entry layout: [0:Lock, 1:refs, 2:active, 3:rerun_requested, 4:latest_state_snapshot, 5:rerun_embed, 6:cache_pass_after_fail, 7:latest_commit_seq]
        entry = _embed_flights.setdefault(key, [threading.Lock(), 0, False, False, state_snapshot, False, False, commit_seq])
        entry[1] += 1
        if state_snapshot is not None and (entry[4] is None or commit_seq >= entry[7]):
            entry[4] = state_snapshot
            entry[7] = max(entry[7], commit_seq)
        if entry[2]:
            entry[3] = True
            if blocking:
                entry[5] = True
            if not blocking:
                busy_fast = True
            else:
                busy_fast = False
        else:
            busy_fast = False
            if not blocking:
                acquired_fast = entry[0].acquire(blocking=False)
                if acquired_fast:
                    entry[2] = True
                    entry[3] = False
                    entry[6] = False
                else:
                    entry[3] = True
                    busy_fast = True
    holder_state = [False]
    try:
        if busy_fast:
            yield False, entry, holder_state
            return
        if not blocking:
            holder_state[0] = True
            yield True, entry, holder_state
            return
        while not entry[0].acquire(timeout=0.05):
            check_request()
        holder_state[0] = True
        with _embed_flights_gate:
            if state_snapshot is not None and (entry[4] is None or commit_seq >= entry[7]):
                entry[4] = state_snapshot
                entry[7] = max(entry[7], commit_seq)
            entry[2] = True
            entry[6] = False
        yield True, entry, holder_state
    finally:
        with _embed_flights_gate:
            if holder_state[0]:
                holder_state[0] = False
                entry[2] = False
                entry[0].release()
            entry[1] -= 1
            if not entry[1] and _embed_flights.get(key) is entry:
                del _embed_flights[key]


_STATE_FIELDS = ("monitor_id", "contract_revision", "fingerprint", "parser_version", "evaluator_version", "generation",
                 "byte_offset", "prefix_hash", "event_count", "path")

# One statement for every write of a monitor state: a state is never written over a newer one of the same generation.
_UPSERT_STATE = (
    "INSERT INTO audit_monitor_state VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    " ON CONFLICT(provider,session_id,monitor_id,needs_digest) DO UPDATE SET"
    " contract_revision=excluded.contract_revision, fingerprint=excluded.fingerprint,"
    " parser_version=excluded.parser_version, evaluator_version=excluded.evaluator_version,"
    " generation=excluded.generation, byte_offset=excluded.byte_offset,"
    " prefix_hash=excluded.prefix_hash, event_count=excluded.event_count, path=excluded.path,"
    " state=excluded.state, updated_at=excluded.updated_at"
    " WHERE audit_monitor_state.generation < excluded.generation OR"
    " (audit_monitor_state.generation = excluded.generation AND"
    "  audit_monitor_state.byte_offset <= excluded.byte_offset)")


def _decode_held_state(row):
    """A held monitor-state row in the shape ``load_audit_states`` returns (the state parsed afresh, so a caller that
    edits it cannot reach the held copy); ``None`` when the held text is unreadable."""
    try:
        state = json.loads(row["state"])
    except ValueError:
        return None
    record = {name: row[name] for name in _STATE_FIELDS}
    record["state"] = state
    return record


class SessionStore:
    """v2 events preserve every occurrence; legacy tables remain untouched for recovery."""

    def __init__(self, db_path: Path, *, max_decompressed_bytes: int | None = None, defer_derived: bool = False,
                 journal: str | None = None):
        """``journal``: ``"auto"`` (write-ahead logging on a verified local file system, else the rollback journal) or
        ``"delete"``; ``None`` is ``journal.DEFAULT_POLICY``. Always ``synchronous=FULL``; see ``journal``."""
        check_request()
        self.db_path = Path(db_path)
        # A resident engine (the daemon) holds derived and advisory rows in memory and writes them with the next
        # authority commit instead of committing each on its own; see ``index.deferred`` for why that is sound.
        self.defer_derived = defer_derived
        self.derived = DerivedBuffer()
        # Bound on the declared decompressed size of a .zst session source (None: compressed_source's default).
        self.max_decompressed_bytes = max_decompressed_bytes
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.journal_policy = _journal.validate_policy(_journal.DEFAULT_POLICY if journal is None else journal)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._counts = {}
        # Before the first transaction: the mode changes only here (or at ``retry_journal``), never inside one.
        self.journal = _journal.apply_journal(self._conn, self.db_path, self.journal_policy)
        try:
            with publication():
                self._initialize_schema()
            stat = self.db_path.stat()
            self._authority_identity = (stat.st_dev, stat.st_ino)
        except BaseException:
            self._conn.close()
            raise

    def retry_journal(self) -> bool:
        """Try again to reach the wanted journal mode after a refused switch (another connection held the file). Call
        it from a quiet point holding the engine's lease; a connection inside a transaction is left alone. Rate-limited,
        never raises. True when the database is in the wanted mode afterwards."""
        if self.journal.pending is not None and self.journal.retry_due():
            _journal.reconcile(self._conn, self.journal)
        return self.journal.pending is None

    def journal_report(self) -> dict:
        return self.journal.report()

    def close(self) -> None:
        """Fold the write-ahead log into the main file (a no-op in the rollback-journal mode), then close. Never raises."""
        try:
            _journal.checkpoint(self._conn)
        finally:
            self._conn.close()

    def _initialize_schema(self):
        self._conn.executescript("""
            BEGIN;
            CREATE TABLE IF NOT EXISTS session_events (
                chunk_id TEXT PRIMARY KEY, provider TEXT NOT NULL, session_id TEXT NOT NULL,
                byte_start INTEGER NOT NULL, sub_index INTEGER NOT NULL, data TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS events_session ON session_events(provider, session_id, byte_start, sub_index);
            CREATE TABLE IF NOT EXISTS session_cursors (
                provider TEXT NOT NULL, session_id TEXT NOT NULL, path TEXT NOT NULL,
                byte_offset INTEGER NOT NULL, prefix_hash TEXT NOT NULL, synced_at TEXT NOT NULL,
                PRIMARY KEY(provider, session_id)
            );
            CREATE TABLE IF NOT EXISTS embedding_cache (
                cache_key TEXT NOT NULL, text_hash TEXT NOT NULL, vector BLOB NOT NULL,
                PRIMARY KEY(cache_key, text_hash)
            );
            CREATE TABLE IF NOT EXISTS session_metrics (
                provider TEXT NOT NULL, session_id TEXT NOT NULL,
                chunk_count INTEGER NOT NULL, parse_errors INTEGER NOT NULL,
                unsupported_events INTEGER NOT NULL, PRIMARY KEY(provider,session_id)
            );
            CREATE TABLE IF NOT EXISTS session_parser_versions (
                provider TEXT NOT NULL, session_id TEXT NOT NULL,
                parser_version INTEGER NOT NULL, synced_at TEXT NOT NULL,
                PRIMARY KEY(provider,session_id)
            );
            CREATE TABLE IF NOT EXISTS session_agy_stats (
                session_id TEXT PRIMARY KEY, stats TEXT NOT NULL, synced_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_receipt_versions (
                provider TEXT NOT NULL, session_id TEXT NOT NULL,
                digest TEXT, synced_at TEXT NOT NULL,
                PRIMARY KEY(provider,session_id)
            );
            -- What the AGY events of a session were derived from besides its bytes: the receipt of each result step
            -- and the first receipt (agy_receipts.dependencies). Bound to a cursor by synced_at like the receipt
            -- digest above; a cursor written by an older process leaves it stale, which means "reparse", never "reuse".
            CREATE TABLE IF NOT EXISTS session_receipt_deps (
                provider TEXT NOT NULL, session_id TEXT NOT NULL,
                deps TEXT, synced_at TEXT NOT NULL,
                PRIMARY KEY(provider,session_id)
            );
            CREATE TABLE IF NOT EXISTS pending_index_deletions (
                provider TEXT NOT NULL, session_id TEXT NOT NULL, chunk_id TEXT NOT NULL,
                PRIMARY KEY(provider,session_id,chunk_id)
            );
            -- Block digests of the indexed prefix (see source_blocks). Bound to a cursor by synced_at; a cursor
            -- written by an older process leaves it stale, which is detected, never trusted.
            CREATE TABLE IF NOT EXISTS source_blocks (
                provider TEXT NOT NULL, session_id TEXT NOT NULL, path TEXT NOT NULL,
                dev INTEGER NOT NULL, ino INTEGER NOT NULL, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
                block_size INTEGER NOT NULL, covered INTEGER NOT NULL, lines INTEGER NOT NULL,
                digests BLOB NOT NULL, tail BLOB, root TEXT NOT NULL, synced_at TEXT NOT NULL,
                full_verified_at TEXT, full_verified_through INTEGER,
                PRIMARY KEY(provider, session_id)
            );
            -- Incremental audit support (see audit/incremental.py). Everything below is derived data: the events
            -- and the cursor stay the authority, and a missing or disagreeing row only means "evaluate in full".
            -- A generation counts the times a session's stored events were replaced; the commit log is the chain
            -- of append-only commits inside the current generation.
            CREATE TABLE IF NOT EXISTS session_generations (
                provider TEXT NOT NULL, session_id TEXT NOT NULL, generation INTEGER NOT NULL,
                PRIMARY KEY(provider,session_id)
            );
            CREATE TABLE IF NOT EXISTS session_commit_log (
                provider TEXT NOT NULL, session_id TEXT NOT NULL, generation INTEGER NOT NULL,
                byte_offset INTEGER NOT NULL, prefix_hash TEXT NOT NULL, event_count INTEGER NOT NULL,
                synced_at TEXT NOT NULL,
                PRIMARY KEY(provider,session_id,generation,byte_offset)
            );
            CREATE TABLE IF NOT EXISTS audit_monitor_state (
                provider TEXT NOT NULL, session_id TEXT NOT NULL, monitor_id TEXT NOT NULL,
                needs_digest TEXT NOT NULL, contract_revision INTEGER NOT NULL, fingerprint TEXT NOT NULL,
                parser_version INTEGER NOT NULL, evaluator_version INTEGER NOT NULL, generation INTEGER NOT NULL,
                byte_offset INTEGER NOT NULL, prefix_hash TEXT NOT NULL, event_count INTEGER NOT NULL,
                path TEXT NOT NULL, state TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(provider,session_id,monitor_id,needs_digest)
            );
            CREATE TABLE IF NOT EXISTS audit_incremental_health (
                provider TEXT NOT NULL, session_id TEXT NOT NULL, monitor_id TEXT NOT NULL,
                evaluator_version INTEGER NOT NULL, status TEXT NOT NULL, detail TEXT NOT NULL, at TEXT NOT NULL,
                PRIMARY KEY(provider,session_id,monitor_id,evaluator_version)
            );
        """ + shadow_sql.SCHEMA)
        check_request()
        self._conn.commit()

    @contextmanager
    def _read_boundary(self):
        """Pin only the raw capture; never retain a read transaction for evaluation."""
        check_request()
        stat = self.db_path.stat()
        if (stat.st_dev, stat.st_ino) != self._authority_identity:
            raise ValueError("metadata authority replaced")
        owned = not self._conn.in_transaction
        if owned:
            check_request()
            self._conn.execute("BEGIN")
        try:
            yield
        finally:
            if owned:
                self._conn.rollback()

    def cursor_record(self, session_id, provider="codex"):
        with self._read_boundary():
            return self._cursor_record(session_id, provider)

    def _cursor_record(self, session_id, provider):
        row = self._conn.execute("SELECT * FROM session_cursors WHERE provider=? AND session_id=?",
                                 (provider, session_id)).fetchone()
        if row is None:
            return None
        record = dict(row)
        version = self._conn.execute(
            "SELECT parser_version,synced_at FROM session_parser_versions WHERE provider=? AND session_id=?",
            (provider, session_id)).fetchone()
        # An older process can still write the six-column cursor table. Its
        # timestamp won't match this stamp, so the new reader reparses safely.
        record["parser_version"] = version[0] if version and version[1] == record["synced_at"] else 1
        if provider == "agy":
            receipt = self._conn.execute(
                "SELECT digest,synced_at FROM session_receipt_versions WHERE provider=? AND session_id=?",
                (provider, session_id)).fetchone()
            record["receipt_digest"] = receipt[0] if receipt and receipt[1] == record["synced_at"] else None
        return record

    def receipt_deps(self, session_id, provider, synced_at):
        """What the indexed events of an AGY session were derived from (``agy_receipts.dependencies``), or None:
        absent, unreadable, or written for another publication than the cursor stamped ``synced_at``."""
        with self._read_boundary():
            row = self._conn.execute("SELECT deps,synced_at FROM session_receipt_deps WHERE provider=? AND session_id=?",
                                     (provider, session_id)).fetchone()
        return _load_deps(row[0]) if row and row[1] == synced_at else None

    def get_cursor(self, session_id, provider="codex"):
        record = self.cursor_record(session_id, provider)
        return (record["byte_offset"], record["path"]) if record else (0, None)

    def source_block_record(self, session_id, provider="codex"):
        """The stored block table of a session, or None. The caller must check ``bound_to(cursor)``."""
        with self._read_boundary():
            return self._source_block_record(session_id, provider)

    def _source_block_record(self, session_id, provider):
        row = self._conn.execute("SELECT * FROM source_blocks WHERE provider=? AND session_id=?",
                                 (provider, session_id)).fetchone()
        if row is None:
            return None
        table = BlockTable(covered=row["covered"], lines=row["lines"], digests=bytes(row["digests"]),
                           tail=None if row["tail"] is None else bytes(row["tail"]), root=row["root"],
                           block_size=row["block_size"])
        record = BlockRecord(provider=provider, session_id=session_id, path=row["path"], dev=_unsigned64(row["dev"]),
                             ino=_unsigned64(row["ino"]),
                             size=row["size"], mtime_ns=row["mtime_ns"], table=table, synced_at=row["synced_at"],
                             full_verified_at=row["full_verified_at"], full_verified_through=row["full_verified_through"])
        stamp = self.derived.stamp_for(provider, session_id, row["synced_at"], row["root"])
        if stamp is not None:
            # A verification this process made and has not written yet; applied only to the table it was made for.
            record = dataclass_replace(record, dev=stamp["dev"], ino=stamp["ino"], size=stamp["size"],
                                       mtime_ns=stamp["mtime_ns"], full_verified_at=stamp["at"],
                                       full_verified_through=stamp["through"])
        return record

    def mark_full_verified(self, session_id, provider, record, *, at, through, dev, ino, size, mtime_ns):
        """Record that EVERY block of ``record``'s table matched the source at ``at``. Compare-and-set on the table's
        publication and root: a table replaced meanwhile is never stamped."""
        if self.defer_derived:
            with self._read_boundary():
                current = self._conn.execute("SELECT synced_at,root FROM source_blocks WHERE provider=? AND session_id=?",
                                             (provider, session_id)).fetchone()
            if current is None or (current[0], current[1]) != (record.synced_at, record.table.root):
                return False
            self.derived.hold_stamp(provider, session_id, {
                "synced_at": record.synced_at, "root": record.table.root, "at": at, "through": through,
                "dev": dev, "ino": ino, "size": size, "mtime_ns": mtime_ns})
            return True
        with self._conn, publication():
            changed = self._conn.execute(
                "UPDATE source_blocks SET full_verified_at=?, full_verified_through=?, dev=?, ino=?, size=?, mtime_ns=?"
                " WHERE provider=? AND session_id=? AND synced_at=? AND root=?",
                (at, through, _signed64(dev), _signed64(ino), size, mtime_ns, provider, session_id, record.synced_at,
                 record.table.root)).rowcount
            check_request()
        return changed == 1

    def commit_events(self, chunks, *, session_id, provider, path, offset, digest, replace=False,
                      parser_version=None, expected_token=None, prepared_rows=None, receipt_digest=None, blocks=None,
                      receipt_deps=None):
        """``blocks``: the ``BlockDraft`` published with the cursor in the same transaction. A publication without
        one (a source whose cursor is a plain digest) removes the session's block row.

        ``receipt_deps`` (AGY): the JSON of ``agy_receipts.dependencies`` for the events as they stand after this
        commit, written with the cursor. A commit without it leaves none, which makes the next receipt change reparse."""
        # ``prepared_rows``: a RowSpool or any row source of its interface (``reindex_spool.StagedRows``).
        spool = prepared_rows if hasattr(prepared_rows, "batches") else RowSpool.from_chunks(chunks)
        try:
            self._commit_spool(spool, session_id=session_id, provider=provider, path=path, offset=offset, digest=digest,
                               replace=replace, parser_version=parser_version, expected_token=expected_token,
                               receipt_digest=receipt_digest, blocks=blocks, receipt_deps=receipt_deps)
        finally:
            if spool is not prepared_rows:
                spool.close()

    def _commit_spool(self, spool, *, session_id, provider, path, offset, digest, replace, parser_version,
                      expected_token, receipt_digest, blocks=None, receipt_deps=None):
        check_request()
        self._counts.clear()
        try:
            self._commit_rows(spool, session_id=session_id, provider=provider, path=path, offset=offset, digest=digest,
                              replace=replace, parser_version=parser_version, expected_token=expected_token,
                              receipt_digest=receipt_digest, blocks=blocks, receipt_deps=receipt_deps)
        finally:
            self._counts.clear()  # a rolled-back write leaves no trace in total_changes

    def _commit_rows(self, spool, *, session_id, provider, path, offset, digest, replace, parser_version,
                     expected_token, receipt_digest, blocks=None, receipt_deps=None):
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            if expected_token is not None and self.state_token(session_id, provider) != expected_token:
                raise ValueError("stale session preparation")
            metrics = self.session_metrics(session_id, provider)
            # ``known``: stored ids of this session as they stand before this commit. A replacement needs all of them
            # (those the new parse no longer has are journalled); an append only asks about the ids it brings.
            known = (self.chunk_ids(session_id, provider) if replace
                     else self.live_ids(session_id, provider, spool.ids))
            previous = self._conn.execute("SELECT byte_offset FROM session_cursors WHERE provider=? AND session_id=?",
                                          (provider, session_id)).fetchone()
            # An append only adds rows at or after the previous cursor. Anything else (a replacement, a row that
            # overwrites a stored id, a row before the cursor) changes the history audit state was built from.
            touched_prefix = replace or previous is None
            prior = self._agy_tally(session_id) if provider == "agy" and not replace else Counter()
            tally = Counter()
            with publication():
                if replace:
                    for cid in known - spool.ids:
                        check_request()
                        self._conn.execute("INSERT OR IGNORE INTO pending_index_deletions VALUES (?,?,?)",
                                           (provider, session_id, cid))
                    self._conn.execute("DELETE FROM session_events WHERE provider=? AND session_id=?",
                                       (provider, session_id))
                added = errors = unsupported = 0
                for batch in spool.batches():
                    check_request()
                    self._conn.executemany("INSERT OR REPLACE INTO session_events VALUES (?,?,?,?,?,?)", batch)
                    for row in batch:
                        if row[1] != provider or row[2] != session_id:
                            raise ValueError("event scope differs from session publication")
                        if not touched_prefix and (row[0] in known or row[3] < previous[0]):
                            touched_prefix = True
                        if replace or row[0] not in known:
                            added += 1
                            flag = spool.flags.get(row[0], (False, False))
                            errors += flag[0]
                            unsupported += flag[1]
                            for key, count in spool.tallies.get(row[0], ()):
                                tally[key] += count
                base = (0, 0, 0) if replace else metrics
                metrics = (base[0] + added, base[1] + errors, base[2] + unsupported)
                synced_at = datetime.now(timezone.utc).isoformat()
                self._conn.execute("INSERT OR REPLACE INTO session_cursors VALUES (?,?,?,?,?,?)",
                                   (provider, session_id, str(path.resolve()), offset, digest, synced_at))
                self._conn.execute("INSERT OR REPLACE INTO session_parser_versions VALUES (?,?,?,?)",
                                   (provider, session_id, current_parser_version(provider) if parser_version is None else parser_version, synced_at))
                if blocks is None:
                    self._conn.execute("DELETE FROM source_blocks WHERE provider=? AND session_id=?",
                                       (provider, session_id))
                else:
                    table = blocks.table
                    self._conn.execute(
                        "INSERT OR REPLACE INTO source_blocks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (provider, session_id, str(path.resolve()), _signed64(blocks.dev), _signed64(blocks.ino), blocks.size,
                         blocks.mtime_ns, table.block_size, table.covered, table.lines, table.digests, table.tail,
                         table.root, synced_at, blocks.full_verified_at, blocks.full_verified_through))
                if provider == "agy":
                    self._conn.execute("INSERT OR REPLACE INTO session_receipt_versions VALUES (?,?,?,?)",
                                       (provider, session_id, receipt_digest, synced_at))
                    self._conn.execute("INSERT OR REPLACE INTO session_receipt_deps VALUES (?,?,?,?)",
                                       (provider, session_id, receipt_deps, synced_at))
                    self._conn.execute("INSERT OR REPLACE INTO session_agy_stats VALUES (?,?,?)",
                                       (session_id, json.dumps(dict(prior + tally), sort_keys=True), synced_at))
                self._conn.execute("INSERT OR REPLACE INTO session_metrics VALUES (?,?,?,?,?)",
                                   (provider, session_id, *metrics))
                self._record_commit(provider, session_id, offset, digest, metrics[0], synced_at, touched_prefix)
                # The block table of this session was just replaced (or removed): a held stamp described the old one,
                # and the new table carries the verification state its draft was built from.
                self.derived.forget_stamp(provider, session_id)
                if len(self.derived):
                    self._write_pending()   # rides this commit: no synchronous commit of its own
                check_request()
                self._conn.commit()

    _COMMIT_LOG_KEEP = 256

    def _record_commit(self, provider, session_id, offset, digest, event_count, synced_at, rewrite):
        """Chain this commit onto the session's append history, in the transaction that published it."""
        row = self._conn.execute("SELECT generation FROM session_generations WHERE provider=? AND session_id=?",
                                 (provider, session_id)).fetchone()
        if rewrite or row is None:
            generation = (row[0] if row else 0) + 1
            self._conn.execute("INSERT OR REPLACE INTO session_generations VALUES (?,?,?)",
                               (provider, session_id, generation))
            self._conn.execute("DELETE FROM session_commit_log WHERE provider=? AND session_id=?", (provider, session_id))
            self._conn.execute("DELETE FROM audit_monitor_state WHERE provider=? AND session_id=?", (provider, session_id))
            self.derived.forget_session(provider, session_id)
        else:
            generation = row[0]
        self._conn.execute("INSERT OR REPLACE INTO session_commit_log VALUES (?,?,?,?,?,?,?)",
                           (provider, session_id, generation, offset, digest, event_count, synced_at))
        self._conn.execute(
            "DELETE FROM session_commit_log WHERE provider=? AND session_id=? AND generation=? AND byte_offset <"
            " COALESCE((SELECT byte_offset FROM session_commit_log WHERE provider=? AND session_id=? AND generation=?"
            " ORDER BY byte_offset DESC LIMIT 1 OFFSET ?), -1)",
            (provider, session_id, generation, provider, session_id, generation, self._COMMIT_LOG_KEEP - 1))

    def pending_deletions(self, session_id, provider):
        return [row[0] for row in self._conn.execute(
            "SELECT chunk_id FROM pending_index_deletions WHERE provider=? AND session_id=?",
            (provider, session_id))]

    def finish_deletions(self, session_id, provider, handled_ids):
        with self._conn, publication():
            for cid in handled_ids:
                check_request()
                self._conn.execute("DELETE FROM pending_index_deletions WHERE provider=? AND session_id=? AND chunk_id=?",
                                   (provider, session_id, cid))
            check_request()

    def chunk_ids(self, session_id=None, provider=None):
        with self._read_boundary():
            return self._chunk_ids(session_id, provider)

    @staticmethod
    def _family_bounds(session_id: str, provider: str) -> tuple[str, str, str]:
        from self_directing_mcp import claude_ids
        parts = claude_ids.session_parts(session_id) if provider == "claude" else None
        parent = parts[0] if parts is not None else session_id
        return parent, f"{parent}_", f"{parent}`"

    def family_session_ids(self, session_id: str, provider: str) -> list[str]:
        """Return ``session_id``'s parent and all indexed child subagent session ids for ``provider``."""
        parent, low, high = self._family_bounds(session_id, provider)
        with self._read_boundary():
            rows = self._conn.execute(
                "SELECT session_id FROM session_cursors WHERE provider=? AND (session_id=? OR (session_id>=? AND session_id<?)) "
                "UNION "
                "SELECT DISTINCT session_id FROM session_events WHERE provider=? AND (session_id=? OR (session_id>=? AND session_id<?))",
                (provider, parent, low, high, provider, parent, low, high),
            ).fetchall()
        found = sorted({row[0] for row in rows})
        return found if found else [parent]

    def family_chunk_ids(self, session_id: str, provider: str) -> set[str]:
        """Return all ``chunk_id``s in ``session_id``'s parent-and-subagent family in one indexed range scan."""
        parent, low, high = self._family_bounds(session_id, provider)
        with self._read_boundary():
            return {
                row[0]
                for row in self._conn.execute(
                    "SELECT chunk_id FROM session_events WHERE provider=? AND (session_id=? OR (session_id>=? AND session_id<?))",
                    (provider, parent, low, high),
                )
            }

    def capture_family_evidence(self, session_id: str, provider: str) -> AuditEvidenceSnapshot:
        """Capture all events across ``session_id``'s parent-and-subagent family in a single read boundary."""
        parent, low, high = self._family_bounds(session_id, provider)
        with self._read_boundary():
            check_request()
            rows = tuple(
                tuple(row)
                for row in self._conn.execute(
                    "SELECT chunk_id, data FROM session_events "
                    "WHERE provider=? AND (session_id=? OR (session_id>=? AND session_id<?)) "
                    "ORDER BY session_id, byte_start, sub_index, chunk_id",
                    (provider, parent, low, high),
                ).fetchall()
            )
            return AuditEvidenceSnapshot(None, provider, rows, len(rows))

    def _chunk_ids(self, session_id, provider):
        clauses, params = [], []
        if session_id is not None:
            clauses.append("session_id=?")
            params.append(session_id)
        if provider is not None:
            clauses.append("provider=?")
            params.append(provider)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return {row[0] for row in self._conn.execute("SELECT chunk_id FROM session_events" + where, params)}

    def live_ids(self, session_id, provider, ids):
        ids = list(ids)
        out = set()
        with self._read_boundary():
            for start in range(0, len(ids), 500):
                check_request()
                batch = ids[start:start + 500]
                out.update(row[0] for row in self._conn.execute(
                    "SELECT chunk_id FROM session_events WHERE +provider=? AND +session_id=? AND chunk_id IN ("
                    + ",".join("?" * len(batch)) + ")", [provider, session_id, *batch]))
        return out

    def existing_ids(self, ids):
        """The subset of ``ids`` the store holds, whatever provider and session (the primary key answers each lookup, in
        bounded batches). A derived index asks it before it deletes a document: a stored event is never an orphan."""
        ids = list(ids)
        out = set()
        with self._read_boundary():
            for start in range(0, len(ids), 500):
                check_request()
                batch = ids[start:start + 500]
                out.update(row[0] for row in self._conn.execute(
                    "SELECT chunk_id FROM session_events WHERE chunk_id IN (" + ",".join("?" * len(batch)) + ")", batch))
        return out

    def has_prior_tool_call(self, session_id, provider, call_id, name, before_byte):
        row = self._conn.execute(
            """SELECT 1 FROM session_events WHERE provider=? AND session_id=? AND byte_start<?
               AND json_extract(data,'$.kind')='tool_call'
               AND json_extract(data,'$.meta.call_id')=?
               AND json_extract(data,'$.meta.tool_name')=? LIMIT 1""",
            (provider, session_id, before_byte, call_id, name),
        ).fetchone()
        return row is not None

    def prior_calls(self, session_id, provider):
        return tuple(tuple(row) for row in self._conn.execute(
            """SELECT byte_start,json_extract(data,'$.meta.call_id'),json_extract(data,'$.meta.tool_name')
               FROM session_events WHERE provider=? AND session_id=? AND json_extract(data,'$.kind')='tool_call'""",
            (provider, session_id)).fetchall())

    def state_token(self, session_id, provider):
        with self._read_boundary():
            values = []
            for table in ("session_cursors", "session_parser_versions", "session_receipt_versions", "session_metrics"):
                row = self._conn.execute(f"SELECT * FROM {table} WHERE provider=? AND session_id=?",
                                         (provider, session_id)).fetchone()
                values.append(tuple(row) if row else None)
            values.append(self.chunk_count(session_id, provider))
            return tuple(values)

    def capture_evidence(self, session_id=None, provider=None, *, history=True, anchors=()):
        with self._read_boundary():
            return self._capture_evidence(session_id, provider, history=history, anchors=anchors)

    def _capture_evidence(self, session_id, provider, *, history, anchors):
        clauses, params = [], []
        if session_id is not None:
            clauses.append("session_id=?")
            params.append(session_id)
        if provider is not None:
            clauses.append("provider=?")
            params.append(provider)
        count = self.chunk_count(session_id, provider)
        if not history:
            anchors = tuple(dict.fromkeys(anchors))
            if not anchors:
                return AuditEvidenceSnapshot(session_id, provider, (), count)
            rows = []
            for start in range(0, len(anchors), 256):
                check_request()
                batch = anchors[start:start + 256]
                # ``+`` keeps the planner on the primary key: without table statistics it prefers the two equalities
                # (a scan of the whole session per batch) over the id list, which made a derived pass quadratic.
                selected = ["+" + clause for clause in clauses] + ["chunk_id IN (" + ",".join("?" for _ in batch) + ")"]
                rows.extend(tuple(row) for row in self._conn.execute(
                    "SELECT chunk_id,data FROM session_events WHERE " + " AND ".join(selected),
                    params + list(batch)).fetchall())
            return AuditEvidenceSnapshot(session_id, provider, tuple(rows), count)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        check_request()
        rows = tuple(tuple(row) for row in self._conn.execute(
            "SELECT chunk_id,data FROM session_events" + where + " ORDER BY provider,session_id,byte_start,sub_index", params).fetchall())
        return AuditEvidenceSnapshot(session_id, provider, rows, count)

    audit_rows_fetched = 0  # slim audit rows read by this store object; tests and benchmarks read it

    def capture_audit_events(self, session_id, provider, needs, *, link_calls=False):
        """Ordered slim events; text is loaded only for events some ``(scope, roles)`` need can inspect.

        ``link_calls`` also keeps text for tool calls/results (evidence of success-linked obligations).
        Rows are paged through a cursor and never held as raw JSON or pydantic models.
        """
        with self._read_boundary():
            count = self.chunk_count(session_id, provider)
            events = self._fetch_audit_events(session_id, provider, needs, link_calls, None)
            return AuditEventSnapshot(session_id, provider, tuple(events), count)

    def capture_audit_tail(self, session_id, provider, needs, *, link_calls=False, from_byte=0):
        """The slim events whose records start at ``from_byte`` or later, in order; call inside a read boundary.

        Reads rows in proportion to the appended records, not to the session: the range is served by the
        ``(provider, session_id, byte_start, sub_index)`` index.
        """
        return self._fetch_audit_events(session_id, provider, needs, link_calls, from_byte)

    def _fetch_audit_events(self, session_id, provider, needs, link_calls, from_byte):
        terms, term_params = [], []
        for scope, roles in needs:
            if scope == "attempt":
                # Host-logged model proposals (forbidden-attempt matching); never every runtime-role event.
                terms.append("json_extract(data,'$.meta.evidence_origin')='model_proposal'")
                continue
            kind = _SCOPE_KINDS.get(scope)
            parts = []
            if kind is not None:
                parts.append("json_extract(data,'$.kind')=?")
                term_params.append(kind)
            parts.append("json_extract(data,'$.meta.role') IN (" + ",".join("?" for _ in roles) + ")" if roles else "0")
            term_params.extend(roles)
            terms.append("(" + " AND ".join(parts) + ")")
        if link_calls:
            terms.append("json_extract(data,'$.kind') IN ('tool_call','tool_result')")
        wanted = " OR ".join(terms) if terms else "0"
        check_request()
        # Provider evidence classes (AGY truncation, pending, legacy, denial; Claude cut results; Grok blocked calls)
        # are read with one extra packed column.
        packed_keys = _AUDIT_META.get(provider)
        extra = ("json_extract(data,'" + "','".join("$.meta." + key for key in packed_keys) + "')"
                 if packed_keys else "NULL")
        tail, tail_params = ("", []) if from_byte is None else (" AND byte_start>=?", [from_byte])
        cursor = self._conn.execute(
            "SELECT chunk_id, json_extract(data,'$.kind'), json_extract(data,'$.meta.role'),"
            " json_extract(data,'$.meta.call_id'), json_type(data,'$.meta.success'),"
            " json_extract(data,'$.meta.success'), CASE WHEN " + wanted + " THEN json_extract(data,'$.text') END,"
            " " + extra + ", byte_start, sub_index FROM session_events WHERE session_id=? AND provider=?" + tail
            + " ORDER BY provider,session_id,byte_start,sub_index", term_params + [session_id, provider] + tail_params)
        events = []
        while True:
            check_request()
            page = cursor.fetchmany(512)
            if not page:
                break
            for cid, kind, role, call_id, success_type, success, text, packed, byte_start, sub_index in page:
                if success_type == "true":
                    success = True
                elif success_type == "false":
                    success = False
                elif success_type in (None, "null"):
                    success = None
                meta = {"role": role, "call_id": call_id, "success": success}
                if packed is not None:
                    meta.update((key, value) for key, value in zip(packed_keys, json.loads(packed))
                                if value is not None)
                events.append(AuditEvent(cid, kind, text or "", meta, (byte_start, sub_index), session_id=session_id))
            self.audit_rows_fetched += len(page)
        return events

    # --- Incremental audit state ------------------------------------------------------------------------------
    # Derived data only. The events and the cursor are the authority; nothing here may make an audit cleaner.

    def audit_lineage(self, session_id, provider):
        """The cursor, the session's generation and its newest commit-log row, as of this read boundary."""
        cursor = self._cursor_record(session_id, provider)
        row = self._conn.execute("SELECT generation FROM session_generations WHERE provider=? AND session_id=?",
                                 (provider, session_id)).fetchone()
        latest = None
        if row is not None:
            found = self._conn.execute(
                "SELECT byte_offset,prefix_hash,event_count,synced_at FROM session_commit_log"
                " WHERE provider=? AND session_id=? AND generation=? ORDER BY byte_offset DESC LIMIT 1",
                (provider, session_id, row[0])).fetchone()
            latest = dict(found) if found is not None else None
        return {"cursor": cursor, "generation": row[0] if row is not None else None, "latest": latest}

    def audit_progress(self, session_id, provider):
        """``(events indexed, events the newest monitor state covers, its byte offset or None)`` as of this read boundary.

        Only a state of the session's CURRENT generation counts (a rewrite drops the others), and a state held in memory
        (``defer_derived``) counts like a stored one. A state belongs to one contract set; this does not look at which, so
        what it reports for one set can only be too much covered, never too little."""
        with self._read_boundary():
            row = self._conn.execute("SELECT generation FROM session_generations WHERE provider=? AND session_id=?",
                                     (provider, session_id)).fetchone()
            total = self.chunk_count(session_id, provider)
            covered, through = 0, None
            if row is not None:
                found = self._conn.execute(
                    "SELECT MAX(event_count), MAX(byte_offset) FROM audit_monitor_state"
                    " WHERE provider=? AND session_id=? AND generation=?", (provider, session_id, row[0])).fetchone()
                covered, through = found[0] or 0, found[1]
                for item in list(self.derived.states.get((provider, session_id), {}).values()):
                    if item["generation"] == row[0]:
                        covered = max(covered, item["event_count"])
                        through = item["byte_offset"] if through is None else max(through, item["byte_offset"])
        return total, covered, through

    def audit_debt(self, session_id, provider):
        """Events the index holds that no monitor state covers (all of them without a state): what the next incremental
        audit has to fold before it can answer, and what a request sizes its sync by."""
        total, covered, _through = self.audit_progress(session_id, provider)
        return max(0, total - covered)

    def lineage_row(self, session_id, provider, generation, offset):
        found = self._conn.execute(
            "SELECT byte_offset,prefix_hash,event_count,synced_at FROM session_commit_log"
            " WHERE provider=? AND session_id=? AND generation=? AND byte_offset=?",
            (provider, session_id, generation, offset)).fetchone()
        return dict(found) if found is not None else None

    def load_audit_states(self, session_id, provider, needs_digest):
        rows = self._conn.execute(
            "SELECT monitor_id,contract_revision,fingerprint,parser_version,evaluator_version,generation,byte_offset,"
            "prefix_hash,event_count,path,state FROM audit_monitor_state"
            " WHERE provider=? AND session_id=? AND needs_digest=?", (provider, session_id, needs_digest)).fetchall()
        out = {}
        for row in rows:
            record = dict(row)
            try:
                record["state"] = json.loads(record["state"])
            except ValueError:
                continue  # an unreadable cache row is the same as no row
            out[record["monitor_id"]] = record
        return self.derived.overlay_states(provider, session_id, needs_digest, out, _decode_held_state)

    def audit_health(self, session_id, provider, evaluator_version):
        return {row[0]: (row[1], row[2]) for row in self._conn.execute(
            "SELECT monitor_id,status,detail FROM audit_incremental_health"
            " WHERE provider=? AND session_id=? AND evaluator_version=?", (provider, session_id, evaluator_version))}

    def record_audit_health(self, session_id, provider, evaluator_version, monitor_ids, status, detail):
        """Persist 'do not trust incremental evaluation of this contract here' (until the evaluator changes)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn, publication():
            self._conn.execute("BEGIN IMMEDIATE")
            for monitor_id in monitor_ids:
                self._conn.execute("INSERT OR REPLACE INTO audit_incremental_health VALUES (?,?,?,?,?,?,?)",
                                   (provider, session_id, monitor_id, evaluator_version, status, detail[:2000], now))
                self._conn.execute("DELETE FROM audit_monitor_state WHERE provider=? AND session_id=? AND monitor_id=?",
                                   (provider, session_id, monitor_id))
            self.derived.forget_monitors(provider, session_id, monitor_ids)
            check_request()

    def drop_audit_states(self, session_id, provider, monitor_ids):
        """Forget stored monitor states (all needs variants) without marking the contract unhealthy."""
        with self._conn, publication():
            self._conn.execute("BEGIN IMMEDIATE")
            for monitor_id in monitor_ids:
                self._conn.execute("DELETE FROM audit_monitor_state WHERE provider=? AND session_id=? AND monitor_id=?",
                                   (provider, session_id, monitor_id))
            self.derived.forget_monitors(provider, session_id, monitor_ids)
            check_request()

    # -- the shadow-comparison schedule and the review ranges (audit/shadow.py, index/shadow_state.py) ----------------

    def load_shadow_state(self, session_id, provider):
        """The stored schedule, or ``None``. A deferring store may hold a newer one; the larger epoch wins, so a reset that
        another process stored is not hidden by an older held state. A row that cannot be read (damaged, written for other
        constants, an invalid field) comes back as a placeholder that keeps its epoch (``ShadowState.unreadable``): the policy
        starts again at 1/4 with its warm-up, and the state that replaces it is not refused by the epoch guard."""
        held = self.derived.shadow_for(provider, session_id)
        found = shadow_sql.load(self._conn, provider, session_id)
        if held is not None and (found is None or found[0] <= held[0]):
            found = held
        if found is None:
            return None
        try:
            return ShadowState.load(found[0], found[1])
        except (KeyError, TypeError, ValueError, AttributeError):
            return ShadowState.unreadable(found[0])

    def save_shadow_state(self, session_id, provider, state):
        """Persist a schedule: held in a deferring store (written with the next authority commit), else committed. A writer
        that read before a reset (an older epoch) is refused (``"stale"``); one of the same epoch keeps the flags of the
        request kinds it did not handle (``ShadowState.touched``)."""
        if self.defer_derived:
            if self.derived.shadow_for(provider, session_id) is None:
                # Nothing is held (a reset dropped it, or it was flushed): the stored row is the authority. While one is held,
                # the holder applies the same guard to its own epoch, and the flush applies the table's.
                found = shadow_sql.load(self._conn, provider, session_id)
                if found is not None and found[0] > state.epoch:
                    return "stale"
            if not self.derived.hold_shadow(provider, session_id, state.epoch, state.dump(), state.touched):
                return "stale"
            if self.derived.full():
                self.flush_derived()
            return "deferred"
        with self._conn, publication():
            self._conn.execute("BEGIN IMMEDIATE")
            written = shadow_sql.write(self._conn, provider, session_id, state.epoch, state.dump(),
                                       touched=state.touched)
            check_request()
        return "saved" if written else "stale"

    def record_incremental_mismatch(self, session_id, provider, evaluator_version, monitor_ids, detail, state=None,
                                    review=None):
        """Everything a mismatch changes, in ONE transaction (all of it or none): the health rows that keep the contracts on the
        full evaluator until the evaluator version changes, their deleted monitor states, the schedule reset to 1/4 and the range
        of requests that need review. It is not undone by a cancellation that arrives once it was admitted: a contract must
        never stay disabled without the reset that goes with it."""
        now = datetime.now(timezone.utc).isoformat()
        self.derived.drop_shadow(provider, session_id)
        with self._conn, publication():
            self._conn.execute("BEGIN IMMEDIATE")
            for monitor_id in monitor_ids:
                self._conn.execute("INSERT OR REPLACE INTO audit_incremental_health VALUES (?,?,?,?,?,?,?)",
                                   (provider, session_id, monitor_id, evaluator_version, "incremental_mismatch",
                                    detail[:2000], now))
                self._conn.execute("DELETE FROM audit_monitor_state WHERE provider=? AND session_id=? AND monitor_id=?",
                                   (provider, session_id, monitor_id))
            self.derived.forget_monitors(provider, session_id, monitor_ids)
            if state is not None:
                shadow_sql.write(self._conn, provider, session_id, state.epoch, state.dump(), now)
            if review is not None:
                shadow_sql.add_range(self._conn, provider, session_id, now=now, **review)

    def shadow_report(self, session_id, provider, evaluator_version):
        """What ``audit_status`` shows of the schedule of one session: the state, the contracts whose incremental path is
        switched off (for this evaluator version) and the ranges that need review."""
        state = self.load_shadow_state(session_id, provider)
        rows = self._conn.execute(
            "SELECT monitor_id,status,detail,at FROM audit_incremental_health WHERE provider=? AND session_id=?"
            " AND evaluator_version=? AND monitor_id NOT LIKE '#%'", (provider, session_id, evaluator_version)).fetchall()
        return {"state": state, "disabled": [{"contract_id": r[0], "status": r[1], "detail": r[2], "at": r[3]} for r in rows],
                "needs_review": shadow_sql.ranges(self._conn, provider, session_id)}

    def save_audit_states(self, session_id, provider, needs_digest, rows, snapshot, shadow=None):
        """Write monitor states and the evaluated cursor in one transaction; returns ``"saved"`` or why not.
        ``shadow``: the session's ``ShadowState``, written in the same transaction (held when the store defers).

        ``snapshot`` is what the evaluation read: ``generation`` (``None`` when no valid commit chain existed),
        ``offset``, ``prefix_hash``, ``synced_at``, ``path`` and ``count``. The write is refused unless the index
        still contains exactly that point: same chain and log row, or (when the chain is new) the same cursor. A
        state is never written over a newer one of the same generation.
        """
        now = datetime.now(timezone.utc).isoformat()
        if self.defer_derived and snapshot["generation"] is not None:
            if shadow is not None:
                self.save_shadow_state(session_id, provider, shadow)
            return self._hold_audit_states(session_id, provider, needs_digest, rows, snapshot, now)
        with self._conn, publication():
            self._conn.execute("BEGIN IMMEDIATE")
            cursor = self._conn.execute(
                "SELECT byte_offset,prefix_hash,synced_at,path FROM session_cursors WHERE provider=? AND session_id=?",
                (provider, session_id)).fetchone()
            row = self._conn.execute("SELECT generation FROM session_generations WHERE provider=? AND session_id=?",
                                     (provider, session_id)).fetchone()
            if snapshot["generation"] is not None:
                if row is None or row[0] != snapshot["generation"]:
                    return "stale_generation"
                logged = self._conn.execute(
                    "SELECT prefix_hash,event_count FROM session_commit_log"
                    " WHERE provider=? AND session_id=? AND generation=? AND byte_offset=?",
                    (provider, session_id, snapshot["generation"], snapshot["offset"])).fetchone()
                if logged is None or logged[0] != snapshot["prefix_hash"] or logged[1] != snapshot["count"]:
                    return "stale_lineage"
                generation = snapshot["generation"]
            else:
                if cursor is None or tuple(cursor) != (snapshot["offset"], snapshot["prefix_hash"],
                                                        snapshot["synced_at"], snapshot["path"]):
                    return "stale_cursor"
                generation = (row[0] if row is not None else 0) + 1
                self._conn.execute("INSERT OR REPLACE INTO session_generations VALUES (?,?,?)",
                                   (provider, session_id, generation))
                self._conn.execute("DELETE FROM session_commit_log WHERE provider=? AND session_id=?", (provider, session_id))
                self._conn.execute("DELETE FROM audit_monitor_state WHERE provider=? AND session_id=?", (provider, session_id))
                self.derived.forget_session(provider, session_id)
                self._conn.execute("INSERT OR REPLACE INTO session_commit_log VALUES (?,?,?,?,?,?,?)",
                                   (provider, session_id, generation, snapshot["offset"], snapshot["prefix_hash"],
                                    snapshot["count"], snapshot["synced_at"]))
            for item in rows:
                check_request()
                self._conn.execute(_UPSERT_STATE, (
                    provider, session_id, item["monitor_id"], needs_digest, item["contract_revision"],
                    item["fingerprint"], item["parser_version"], item["evaluator_version"], generation,
                    snapshot["offset"], snapshot["prefix_hash"], snapshot["count"], snapshot["path"],
                    json.dumps(item["state"], separators=(",", ":")), now))
            self._trim_state_variants(provider, session_id)
            if shadow is not None:
                shadow_sql.write(self._conn, provider, session_id, shadow.epoch, shadow.dump(), now,
                                 touched=shadow.touched)
            check_request()
            return "saved"

    def _hold_audit_states(self, session_id, provider, needs_digest, rows, snapshot, now):
        """``save_audit_states`` for a store that defers its derived writes: the same checks against the lineage, then the
        rows are held in memory (read back through ``load_audit_states``) instead of committed. Returns ``"deferred"``."""
        with self._read_boundary():
            row = self._conn.execute("SELECT generation FROM session_generations WHERE provider=? AND session_id=?",
                                     (provider, session_id)).fetchone()
            if row is None or row[0] != snapshot["generation"]:
                return "stale_generation"
            logged = self._conn.execute(
                "SELECT prefix_hash,event_count FROM session_commit_log"
                " WHERE provider=? AND session_id=? AND generation=? AND byte_offset=?",
                (provider, session_id, snapshot["generation"], snapshot["offset"])).fetchone()
            if logged is None or logged[0] != snapshot["prefix_hash"] or logged[1] != snapshot["count"]:
                return "stale_lineage"
        for item in rows:
            check_request()
            self.derived.hold_state(provider, session_id, needs_digest, item["monitor_id"], {
                "monitor_id": item["monitor_id"], "needs_digest": needs_digest,
                "contract_revision": item["contract_revision"], "fingerprint": item["fingerprint"],
                "parser_version": item["parser_version"], "evaluator_version": item["evaluator_version"],
                "generation": snapshot["generation"], "byte_offset": snapshot["offset"],
                "prefix_hash": snapshot["prefix_hash"], "event_count": snapshot["count"], "path": snapshot["path"],
                "state": json.dumps(item["state"], separators=(",", ":")), "updated_at": now})
        if self.derived.full():
            self.flush_derived()
        return "deferred"

    def _trim_state_variants(self, provider, session_id):
        """Keep the newest few needs variants of a monitor (audit and action checks differ), drop the rest."""
        self._conn.execute(
            "DELETE FROM audit_monitor_state WHERE provider=? AND session_id=? AND rowid IN ("
            " SELECT rowid FROM (SELECT rowid, ROW_NUMBER() OVER (PARTITION BY monitor_id ORDER BY updated_at DESC) AS n"
            " FROM audit_monitor_state WHERE provider=? AND session_id=?) WHERE n > 4)",
            (provider, session_id, provider, session_id))

    # -- deferred writes (``defer_derived``) ------------------------------------------------------------------------

    def has_pending_derived(self) -> bool:
        return bool(len(self.derived))

    def flush_derived(self) -> int:
        """Write what ``defer_derived`` is holding, in one transaction; returns how many rows were written.

        Callable at any time, from any thread that holds the engine's lease (``SelfDirectEngine.flush_derived``). A
        failure loses cache rows, never authority, and is not raised for a SQLite error."""
        if not len(self.derived):
            return 0
        try:
            with self._conn, publication():
                self._conn.execute("BEGIN IMMEDIATE")
                written = self._write_pending()
                check_request()
            return written
        except sqlite3.Error:
            return 0

    def _write_pending(self) -> int:
        """Inside an open write transaction: validate every held row once more against the lineage as it stands NOW and
        write those that still fit; the rest are dropped (they are cache). Nothing here may fail the transaction it
        rides in, so the work runs in a savepoint that a SQLite error rolls back."""
        states, stamps = self.derived.drain()
        shadows = self.derived.drain_shadows()
        if not states and not stamps and not shadows:
            return 0
        written, touched = 0, set()
        self._conn.execute("SAVEPOINT derived_flush")
        try:
            for provider, session_id, epoch, state, flags_touched in shadows:   # self-validating (premise, epoch): no lineage check
                written += bool(shadow_sql.write(self._conn, provider, session_id, epoch, state, touched=flags_touched))
            lineage = {}
            for provider, session_id, needs_digest, monitor_id, row in states:
                key = (provider, session_id, row["generation"], row["byte_offset"])
                if key not in lineage:
                    current = self._conn.execute(
                        "SELECT generation FROM session_generations WHERE provider=? AND session_id=?",
                        (provider, session_id)).fetchone()
                    logged = None
                    if current is not None and current[0] == row["generation"]:
                        logged = self._conn.execute(
                            "SELECT prefix_hash,event_count FROM session_commit_log"
                            " WHERE provider=? AND session_id=? AND generation=? AND byte_offset=?",
                            (provider, session_id, row["generation"], row["byte_offset"])).fetchone()
                    lineage[key] = logged
                logged = lineage[key]
                if logged is None or logged[0] != row["prefix_hash"] or logged[1] != row["event_count"]:
                    continue    # the lineage moved on (a rewrite, a trimmed log row): this row describes nothing stored
                self._conn.execute(_UPSERT_STATE, (
                    provider, session_id, monitor_id, needs_digest, row["contract_revision"], row["fingerprint"],
                    row["parser_version"], row["evaluator_version"], row["generation"], row["byte_offset"],
                    row["prefix_hash"], row["event_count"], row["path"], row["state"], row["updated_at"]))
                touched.add((provider, session_id))
                written += 1
            for provider, session_id in touched:
                self._trim_state_variants(provider, session_id)
            for provider, session_id, stamp in stamps:
                written += self._conn.execute(
                    "UPDATE source_blocks SET full_verified_at=?, full_verified_through=?, dev=?, ino=?, size=?, mtime_ns=?"
                    " WHERE provider=? AND session_id=? AND synced_at=? AND root=?",
                    (stamp["at"], stamp["through"], _signed64(stamp["dev"]), _signed64(stamp["ino"]), stamp["size"],
                     stamp["mtime_ns"], provider, session_id, stamp["synced_at"], stamp["root"])).rowcount
            self._conn.execute("RELEASE derived_flush")
        except sqlite3.Error:
            self._conn.execute("ROLLBACK TO derived_flush")
            self._conn.execute("RELEASE derived_flush")
            return 0
        return written

    _CALL_INDEX = ("CREATE INDEX IF NOT EXISTS events_call ON session_events(provider, session_id,"
                   " json_extract(data,'$.meta.call_id')) WHERE json_extract(data,'$.meta.call_id') IS NOT NULL")

    def ensure_call_index(self):
        """Index events by call id so a monitor can ask which calls and results share one. Built once per index.

        Never call this inside a read boundary: creating the index writes.
        """
        if self._conn.in_transaction:
            raise RuntimeError("ensure_call_index must run outside a read boundary")
        if self._conn.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name='events_call'").fetchone():
            return
        with self._conn, publication():
            self._conn.execute(self._CALL_INDEX)
            check_request()

    def audit_call_events(self, session_id, provider, call_ids, before_byte):
        """``{call_id: [(kind, (byte_start, sub_index), chunk_id)]}`` for stored calls and results of these ids."""
        found = {}
        ids = list(call_ids)
        for start in range(0, len(ids), 400):
            check_request()
            batch = ids[start:start + 400]
            rows = self._conn.execute(
                "SELECT json_extract(data,'$.meta.call_id'), json_extract(data,'$.kind'), byte_start, sub_index, chunk_id"
                # ``+byte_start``: keep the planner on the call-id index; a range on byte_start would scan the session.
                " FROM session_events WHERE provider=? AND session_id=? AND +byte_start<?"
                " AND json_extract(data,'$.meta.call_id') IN (" + ",".join("?" for _ in batch) + ")"
                " AND json_extract(data,'$.kind') IN ('tool_call','tool_result')",
                [provider, session_id, before_byte, *batch]).fetchall()
            for call_id, kind, byte_start, sub_index, chunk_id in rows:
                found.setdefault(call_id, []).append((kind, (byte_start, sub_index), chunk_id))
        self.audit_rows_fetched += sum(len(v) for v in found.values())
        return found

    PROPOSAL_LOOKBACK = 4096  # events scanned back from the first appended one
    PROPOSAL_ROWS = 128       # proposals of one tool kept from that window

    def anchor_chunks(self, session_id, provider, anchors):
        """The stored events named by ``anchors`` that belong to this session, as full chunks (no session scan)."""
        found = []
        for anchor in anchors:
            check_request()
            chunk = self.get_chunk(anchor)
            if chunk is not None and chunk.session_id == session_id and chunk.provider == provider:
                found.append(chunk)
        return found

    def audit_proposals_before(self, session_id, provider, name, before_byte):
        """Model proposals of tool ``name`` among the last ``PROPOSAL_LOOKBACK`` events before ``before_byte``.

        Returns ``(rows, complete)``, newest first, ``(key, text)`` each, at most ``PROPOSAL_ROWS``; ``complete`` is
        false when the window did not reach the start of the session or held more proposals than that, so an older
        one may exist.
        """
        check_request()
        edge = self._conn.execute(
            "SELECT 1 FROM (SELECT 1 FROM session_events WHERE provider=? AND session_id=? AND byte_start<?"
            " ORDER BY byte_start DESC, sub_index DESC LIMIT ?) ", (provider, session_id, before_byte,
                                                                    self.PROPOSAL_LOOKBACK + 1)).fetchall()
        complete = len(edge) <= self.PROPOSAL_LOOKBACK
        rows = self._conn.execute(
            "SELECT byte_start, sub_index, json_extract(data,'$.text') FROM (SELECT byte_start, sub_index, data"
            " FROM session_events WHERE provider=? AND session_id=? AND byte_start<?"
            " ORDER BY byte_start DESC, sub_index DESC LIMIT ?)"
            " WHERE json_extract(data,'$.meta.evidence_origin')='model_proposal'"
            " AND json_extract(data,'$.meta.tool_name') IS ? ORDER BY byte_start DESC, sub_index DESC LIMIT ?",
            (provider, session_id, before_byte, self.PROPOSAL_LOOKBACK, name, self.PROPOSAL_ROWS + 1)).fetchall()
        if len(rows) > self.PROPOSAL_ROWS:
            rows, complete = rows[:self.PROPOSAL_ROWS], False
        self.audit_rows_fetched += len(rows)
        return [((row[0], row[1]), row[2] or "") for row in rows], complete

    def agy_fallback_retries(self, session_id):
        """Sizes (None: missing) of the sibling full log recorded with this session's transient failed fallbacks."""
        with self._read_boundary():
            return [None if size == "None" else int(size) for size in
                    (key.split("|", 1)[1] for key in self._agy_tally(session_id) if key.startswith("retry|"))]

    def _agy_tally(self, session_id):
        """Evidence-class counts: the row published with the current cursor, else a scan of the stored events
        (a still-running older process, or an index from before parser v6)."""
        row = self._conn.execute(
            "SELECT s.stats FROM session_agy_stats s JOIN session_cursors c ON c.provider='agy'"
            " AND c.session_id=s.session_id AND c.synced_at=s.synced_at WHERE s.session_id=?", (session_id,)).fetchone()
        if row is not None:
            return Counter(json.loads(row[0]))
        tally = Counter()
        cursor = self._conn.execute(
            "SELECT json_extract(data,'$.meta.linkage','$.meta.pending','$.meta.media_count','$.meta.event_type',"
            "'$.meta.truncated_fields','$.meta.restored_fields','$.meta.fallback','$.meta.host_padding_bytes',"
            "'$.meta.sub_index') FROM session_events"
            " WHERE provider='agy' AND session_id=?", (session_id,))
        while True:
            check_request()
            page = cursor.fetchmany(1024)
            if not page:
                return tally
            for (packed,) in page:
                for key, count in agy_tally(*json.loads(packed)):
                    tally[key] += count

    def agy_evidence_stats(self, session_id):
        """Legacy/pending results, media placeholders and truncation counts of one AGY session."""
        with self._read_boundary():
            return _agy_stats_dict(self._agy_tally(session_id))

    def session_metrics(self, session_id, provider):
        row = self._conn.execute(
            "SELECT chunk_count,parse_errors,unsupported_events FROM session_metrics WHERE provider=? AND session_id=?",
            (provider, session_id)).fetchone()
        if row is not None:
            return tuple(row)
        # Read-only legacy fallback; no hidden mutation from a metadata reader.
        row = self._conn.execute("""
            SELECT COUNT(*), COALESCE(SUM(json_extract(data,'$.meta.parse_error') = 1),0),
                   COALESCE(SUM(json_extract(data,'$.meta.unsupported_event') = 1),0)
            FROM session_events WHERE provider=? AND session_id=?""", (provider, session_id)).fetchone()
        return tuple(row)

    def get_chunk(self, chunk_id):
        row = self._conn.execute("SELECT data FROM session_events WHERE chunk_id=?", (chunk_id,)).fetchone()
        return Chunk.model_validate_json(row["data"]) if row else None

    WORKING_DIR_SCAN = 50           # events at the start of a session that may name its working directory

    def session_working_dir(self, session_id, provider):
        """The working directory the session's own first records name, or ``None`` (never guessed).

        Claude Code stamps ``cwd`` on its records; a Codex rollout names it in its ``session_meta`` record. Other
        providers' parsers do not index one. Only the first few events are looked at, so a session that never names
        a directory costs a bounded read.
        """
        if provider not in ("claude", "codex"):
            return None
        with self._read_boundary():
            rows = self._conn.execute(
                "SELECT json_extract(data,'$.meta.cwd') AS cwd, json_extract(data,'$.meta.evidence_origin') AS origin,"
                " json_extract(data,'$.text') AS text FROM (SELECT data FROM session_events WHERE provider=? AND "
                "session_id=? ORDER BY byte_start, sub_index LIMIT ?)", (provider, session_id, self.WORKING_DIR_SCAN))
            for row in rows:
                if provider == "claude":
                    found = row["cwd"]
                elif row["origin"] == "session_meta" and isinstance(row["text"], str):
                    try:
                        found = json.loads(row["text"].partition(" ")[2]).get("cwd")
                    except (ValueError, AttributeError):
                        found = None
                else:
                    continue
                if isinstance(found, str) and 0 < len(found) <= 4096:
                    return found
        return None

    def list_chunks(self, session_id=None, provider=None):
        clauses, params = [], []
        if session_id is not None:
            clauses.append("session_id=?")
            params.append(session_id)
        if provider is not None:
            clauses.append("provider=?")
            params.append(provider)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            "SELECT data FROM session_events" + where + " ORDER BY provider,session_id,byte_start,sub_index", params)
        return [Chunk.model_validate_json(row["data"]) for row in rows]

    def event_frames(self, session_id, provider):
        """Normalized event rows for time-series analysis.

        Original order (byte_start, sub_index) is the sequence; timestamps are
        returned as-is and never fabricated. rowid is not used as an identifier.
        """
        rows = self._conn.execute(
            """SELECT data FROM session_events WHERE provider=? AND session_id=?
               ORDER BY byte_start, sub_index""", (provider, session_id))
        return [_event_frame(order, Chunk.model_validate_json(row["data"])) for order, row in enumerate(rows)]

    def event_frames_range(self, session_id, provider, start, max_events, max_chars=None, *, cost=None):
        """``event_frames(session_id, provider)[start:]``, cut after a batch: the same frames with the same ``order``
        numbering, but only the rows of the batch are read and parsed, however long the session is.

        Frames are taken in event order from ``start`` while there are at most ``max_events`` of them and, when
        ``max_chars`` is given, their summed ``cost(frame)`` (default: the length of the text) stays within it. The
        first frame that would not fit is returned as ``blocked`` and never cut; with no frame taken it is the
        event that alone exceeds the budget. Returns ``(frames, blocked)``; ``blocked`` is ``None`` when the batch
        ended by count or at the end of the session. The ``start`` rows before the batch are stepped over in the
        ``events_session`` index and their data is not read. One read boundary covers the whole read.
        """
        if start < 0 or max_events < 1:
            raise ValueError("start must be >= 0 and max_events >= 1")
        cost = cost or (lambda frame: len(frame["text"]))
        check_request()
        with self._read_boundary():
            rows = self._conn.execute(
                "SELECT data FROM session_events WHERE provider=? AND session_id=?"
                " ORDER BY byte_start, sub_index LIMIT ? OFFSET ?", (provider, session_id, max_events, start))
            frames, size = [], 0
            try:
                for order, row in enumerate(rows, start):
                    check_request()
                    frame = _event_frame(order, Chunk.model_validate_json(row["data"]))
                    spent = size + cost(frame)
                    if max_chars is not None and spent > max_chars:
                        return frames, frame
                    size = spent
                    frames.append(frame)
            finally:
                rows.close()       # a batch cut short leaves unread rows; release the statement inside the boundary
            return frames, None

    def checklist_items(self, session_id, provider):
        """Load checklist items for stalled-requirement analysis (best effort)."""
        try:
            from self_directing_mcp.checklist import ChecklistStore
        except ImportError:
            return []
        try:
            doc = ChecklistStore(Path(self.db_path).parent).load(session_id)
            return [item.model_dump(mode="json") for item in doc.items]
        except Exception:
            return []

    def chunk_count(self, session_id=None, provider=None):
        clauses, params = [], []
        if session_id is not None:
            clauses.append("session_id=?")
            params.append(session_id)
        if provider is not None:
            clauses.append("provider=?")
            params.append(provider)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        # COUNT(*) walks the whole session's index range; a state token asks for it several times per request. The
        # answer cannot change while neither this connection nor another one has written (``data_version`` moves
        # for other connections, ``total_changes`` for this one), and this store's own writes drop the cache.
        version = (self._conn.execute("PRAGMA data_version").fetchone()[0], self._conn.total_changes)
        cached = self._counts.get((session_id, provider))
        if cached is not None and cached[0] == version:
            return cached[1]
        count = self._conn.execute("SELECT COUNT(*) FROM session_events" + where, params).fetchone()[0]
        self._counts[(session_id, provider)] = (version, count)
        return count

    def known_hashes(self, session_id, provider=None):
        return {c.content_hash for c in self.list_chunks(session_id, provider)}

    def cached_vector(self, cache_key, text_hash):
        row = self._conn.execute("SELECT vector FROM embedding_cache WHERE cache_key=? AND text_hash=?",
                                 (cache_key, text_hash)).fetchone()
        return np.frombuffer(row["vector"], dtype=np.float64).copy() if row else None

    def cache_vectors(self, cache_key, pairs):
        payload = [(cache_key, key, np.asarray(vec, dtype=np.float64).tobytes()) for key, vec in pairs]
        with self._conn, publication():
            for start in range(0, len(payload), 256):
                check_request()
                self._conn.executemany("INSERT OR REPLACE INTO embedding_cache VALUES (?,?,?)", payload[start:start + 256])
            check_request()

    def legacy_count(self):
        exists = self._conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'").fetchone()
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] if exists else 0

    def session_status(self, session_id, provider="codex"):
        with self._read_boundary():
            record = self.cursor_record(session_id, provider)
            metrics = self.session_metrics(session_id, provider)
            stats = self.agy_evidence_stats(session_id) if provider == "agy" and record else None
        coverage = coverage_from_capture(session_id, provider, record, metrics, max_bytes=self.max_decompressed_bytes)
        if stats is not None:
            # No receipt capture here: the counts are exact, the first-receipt position is reported by sync/audit.
            coverage.update(legacy_unlinked_count=stats["legacy_unlinked_results"],
                            result_dependent_complete=not stats["legacy_unlinked_results"], agy_evidence=stats)
        return coverage


@dataclass(frozen=True)
class IntegrityPolicy:
    """How source integrity is verified (see ``source_blocks`` and the README, "Source integrity")."""

    recent_blocks: int = 4  # fast path: the latest complete blocks that are always compared
    sample_blocks: int = 8  # fast path: random other blocks compared (drawn fresh for every call)
    full_verify_interval_sec: float = 300.0  # a fast check is only allowed this long after the last full one
    clock: Any = lambda: datetime.now(timezone.utc)
    rng: Any = None  # None: fresh system randomness per call

    @classmethod
    def from_settings(cls, settings):
        return cls(recent_blocks=settings.integrity_fast_recent_blocks,
                   sample_blocks=settings.integrity_fast_sample_blocks,
                   full_verify_interval_sec=settings.integrity_full_verify_interval_sec)

    def new_rng(self):
        return self.rng() if self.rng is not None else None


# Slicing (``slice_budget``). A slice's soft deadline is ``SYNC_SHARE`` of the time the request has left when the parse
# starts, divided by ``1 + COMMIT_RATIO`` because the commit of what was parsed costs about that fraction of the parse:
# parse and commit use about half the budget and at least a quarter remains for the evaluation and the response. Never
# less than ``SOFT_DEADLINE_MIN_SEC`` after the start. That share is the UPPER bound: a request that will also judge
# what it syncs (``slice_budget.Sizing``) ends the slice earlier when parsing, committing and evaluating the records it
# holds would no longer fit the request (``SliceBudget.cost_budget``). Module constants so tests can move them.
SYNC_SHARE = 0.5
COMMIT_RATIO = 0.3
SOFT_DEADLINE_MIN_SEC = 0.25
# A request that may slice waits for the session's flight at most min(FLIGHT_WAIT_MAX_SEC, FLIGHT_WAIT_SHARE * time left).
FLIGHT_WAIT_MAX_SEC = 1.0
FLIGHT_WAIT_SHARE = 0.25
RETRY_AFTER_MS = 500    # the hint a partial sync gives; advisory, nothing retries by itself
# Staging a re-index (``reindex_spool``): the parse of a re-index goes into a spool slice by slice and ONE ordinary replace
# commit publishes it. That commit cannot be cut, so it is started only when its predicted time fits what the request has
# left. Module constants, so tests can move them.
COMMIT_SECONDS_PER_ROW = 18e-6   # measured on synthetic sessions (ext4): the replace commit costs about 18 microseconds per event
COMMIT_SAFETY = 1.5              # the prediction is multiplied by this before it is compared with the time left
COMMIT_RESERVE = 0.1             # the share of the time left that a commit leaves for the answer and the derived work
REINDEX_STATES = ("staging", "ready_to_commit", "committed", "discarded", "disabled")


@dataclass(frozen=True)
class SlicePolicy:
    """Whether a sync may publish a prefix instead of all or nothing, and how a slice is bounded.

    ``mode``: ``"auto"`` (a request with a finite deadline and a plain file: appends are published in slices, a re-index
    is staged in a spool and published by one commit) or ``"off"`` (exactly the all-or-nothing sync of before).
    ``cap_bytes``: the most a slice observes past the indexed offset
    (``SELF_DIRECT_SYNC_SLICE_BYTES``). The rest bounds the parse (``slice_budget.SliceBudget``); the defaults are the
    production ones, and tests set ``max_records``/``max_bytes`` to force deterministic stops and ``clock`` to move time.
    """

    mode: str = settings_core.DEFAULT_SYNC_SLICING
    cap_bytes: int = settings_core.DEFAULT_SYNC_SLICE_BYTES
    min_records: int = 256
    min_bytes: int = 1 << 20
    max_records: int | None = None
    max_bytes: int | None = None
    clock: Any = time.monotonic

    def __post_init__(self):
        if self.mode not in settings_core.SYNC_SLICING_MODES or self.cap_bytes < 1:
            raise ValueError("invalid slice policy")

    @classmethod
    def from_settings(cls, settings):
        return cls(mode=getattr(settings, "sync_slicing", settings_core.DEFAULT_SYNC_SLICING),
                   cap_bytes=getattr(settings, "sync_slice_bytes", settings_core.DEFAULT_SYNC_SLICE_BYTES))


def _may_slice(slicing, path) -> tuple[bool, bool]:
    """``(reports, slices)``: whether a call reports a ``sync`` object, and whether it may cut the sync short.

    Both need ``auto`` and a request with a finite deadline (no deadline: no budget to run out of); only a plain file
    is ever cut (a ``.zst`` source is decoded from its start, so a forward seek would decode the prefix again)."""
    control = current_request()
    reports = (slicing is not None and slicing.mode == "auto" and control is not None
               and control.deadline is not None)
    return reports, reports and not is_compressed(path)


def _flight_wait(slicing, path):
    """Seconds a caller waits for the flight of its session (None: until the leader is done, as without slicing)."""
    if not _may_slice(slicing, path)[1]:
        return None
    return min(FLIGHT_WAIT_MAX_SEC, FLIGHT_WAIT_SHARE * max(0.0, current_request().deadline - time.monotonic()))


def _new_budget(slicing, start, pending=None, judged=True) -> SliceBudget:
    """The budget of a slice that starts at ``start`` and has ``pending`` bytes to parse (``None``: not known). A parse
    shorter than the minimum progress is never cut, so it does not need the cost numbers (and a request that only ever
    appends a little never reads the contracts for them).

    ``judged``: whether the request judges what this slice publishes. A staged slice of a re-index does not: its rows wait in
    a spool, the index stays as it is, and the evaluation that follows reads the OLD index. Such a slice is sized by the
    share alone, and neither the contracts nor the index are read for numbers that would not be used."""
    remaining = max(0.0, current_request().deadline - time.monotonic())
    started = slicing.clock()
    options = {}
    sizing = current_sizing() if judged else None
    if sizing is not None and (pending is None or pending > slicing.min_bytes) and sizing.resolve():
        # The request judges what this slice publishes: parse + commit + evaluating the records must fit what is left
        # of the REQUEST (not of the narrower share its sync runs in), less the reserve for the response and less the
        # evaluation of what the index already holds without a monitor state. ``look`` applies it with the parse speed
        # this very slice shows, so a guess about the parse cannot break it. The share above stays the upper bound.
        whole = max(0.0, sizing.deadline - time.monotonic())
        options = dict(started=started, cost_per_record=sizing.eval_per_event, commit_factor=1.0 + COMMIT_RATIO,
                       cost_budget=max(0.0, (1.0 - sizing.reserve) * whole - sizing.debt_events * sizing.eval_per_event))
    return SliceBudget(start, soft_deadline=started + max(SOFT_DEADLINE_MIN_SEC,
                                                         SYNC_SHARE * remaining / (1.0 + COMMIT_RATIO)),
                       min_records=slicing.min_records, min_bytes=slicing.min_bytes, max_bytes=slicing.max_bytes,
                       max_records=slicing.max_records, clock=slicing.clock, **options)


def _sync_report(indexed, target, pending, budget, observation, reindex=None, provider=None) -> dict:
    """The ``sync`` object of a result: how far the index got, how far this sync means to go, and why it stopped.

    ``progress`` is the share of the target that is COMMITTED (what a parse read but did not publish does not count). While a
    re-index is staged and not yet published (``reindex``: its report, see ``_Stage.report``) the index in force is still the
    old one: the sync is partial whatever the cursor says, ``progress`` is the staged share, and a staged range that reaches the
    end of the source stopped at ``eof``.

    ``cost`` is what sized the slice: ``slice_us`` the expected microseconds to parse and commit an event, ``eval_us`` the
    expected microseconds to judge one (0: nothing is judged after this sync; so it is for a staged re-index, whose rows no
    audit reads until they are published), ``debt_events`` the events already indexed that no monitor state covers, and
    ``limited_by`` which limit ended it (``share``: the soft deadline, ``cost``: the cost bound, ``None``: neither, e.g. the
    end of the file)."""
    waiting = reindex is not None and reindex["state"] in ("staging", "discarded", "ready_to_commit")
    partial = indexed < target or waiting
    stopped = "eof"
    if waiting:
        stopped = ("eof" if reindex["state"] == "ready_to_commit" else
                   budget.stopped_by if budget is not None else None) or "slice_cap"
    elif partial:
        stopped = (budget.stopped_by if budget is not None and indexed < observation.complete_end else None) or "slice_cap"
    share = reindex["staged_through"] if waiting else indexed
    report = {"state": "partial" if partial else "complete", "indexed_through": indexed, "pending_bytes": pending,
              "target_bytes": target, "progress": round(min(1.0, share / target), 4) if partial and target else 1.0,
              "retry_after_ms": RETRY_AFTER_MS if partial else None, "stopped_by": stopped}
    if reindex is not None:
        report["reindex"] = reindex
    if budget is not None:
        sizing = current_sizing()
        report["cost"] = {"slice_us": round(COSTS.slice_cost(provider) * 1e6, 1),
                          "eval_us": round(budget.cost_per_record * 1e6, 1),
                          "debt_events": sizing.debt_events if sizing is not None and budget.cost_budget is not None else 0,
                          "limited_by": budget.bound if partial else None}
    return report


def _commit_fits(rows: int) -> bool:
    """Whether the replace commit of ``rows`` staged events is predicted to fit what the request has left. It is never
    started otherwise: a transaction that starts and is rolled back by the deadline only spends the budget."""
    left = max(0.0, current_request().deadline - time.monotonic())
    return COMMIT_SECONDS_PER_ROW * rows * COMMIT_SAFETY <= left * (1.0 - COMMIT_RESERVE)


class _Stage:
    """The staged re-index of one sync (``reindex_spool``): the spool, what this request did with it, what it reports.

    ``spool`` is None when staging is off for this request (``disabled``: the index volume has no room, or the directory is
    not private): the sync is then the all-or-nothing re-index of before."""

    def __init__(self):
        self.spool = None
        self.disabled = None        # why staging is off for this request
        self.discarded = None       # why a spool that was found has been thrown away
        self.complete = False       # the staged range reaches the end of what the capture observed
        self.committed = False
        self.rows = 0               # rows published, once committed

    @property
    def active(self) -> bool:
        return self.spool is not None

    @property
    def start(self) -> int:
        """The source offset this request's parse starts at: where the staged range ends (0 for a spool to be started)."""
        return self.spool.staged_through if self.spool is not None and self.spool.valid else 0

    @classmethod
    def begin(cls, *, store, provider, sid, cursor, observation, path, kind, parser_version, receipt_digest, receipt_rows):
        """Open and validate the spool of this re-index, or find that staging is off for this request.

        ``receipt_rows`` (AGY): a spool whose rows were parsed with other receipts survives only if the receipts that changed
        are none of those its events were derived from (``agy_receipts.affects_index``); a new receipt for a later step must
        not restart the staging of a session that is still being written."""
        stage, index_dir, length = cls(), store.db_path.parent, observation.length
        head = min(reindex_spool.HEAD_BYTES, length)
        binding = reindex_spool.SpoolBinding(
            provider=provider, session_id=sid, target_version=parser_version, base_version=cursor["parser_version"],
            base_stamp=cursor["synced_at"], hash_kind=kind, dev=observation.identity[0], ino=observation.identity[1],
            path=str(path), size=length, head_length=head, head_hash=observation.head_digest(head),
            receipt_digest=receipt_digest)
        try:
            spool = reindex_spool.ReindexSpool.open(index_dir, binding, source_length=length,
                                                    head_hash=observation.head_digest,
                                                    cursor_hash=lambda offset: observation.cursor_hash(offset, kind))
        except reindex_spool.SpoolBusy:
            raise FlightBusy() from None       # another process is staging this session: judge on what is committed
        except reindex_spool.SpoolUnsafe:
            stage.disabled = "unsafe"
            return stage
        except (RequestStopped, IndexBusy):
            raise
        except OSError:
            stage.disabled = "unavailable"     # the directory cannot be made or read: the re-index is the one of before
            return stage
        try:
            if spool.valid and receipt_rows is not None and spool.slices and spool.receipt_digest != receipt_digest:
                from self_directing_mcp.agy_receipts import DEPENDENCIES_VERSION, affects_index
                staged = {"v": DEPENDENCIES_VERSION, "first": spool.deps_first, "steps": dict(spool.deps_steps)}
                if affects_index(staged, receipt_rows):
                    spool.discard("receipts_changed")
            if not reindex_spool.has_room(index_dir, observation.complete_end - (spool.staged_through if spool.valid else 0)):
                if spool.valid:
                    spool.discard("disk")      # the space it holds is given back; the sync is the all-or-nothing one
                stage.disabled = "disk"
        except BaseException:
            spool.close()
            raise
        stage.discarded = spool.discard_reason
        if stage.disabled:
            spool.close()
        else:
            stage.spool = spool
        return stage

    def persist(self, rows, start, end, digest, *, deps, receipt_digest) -> None:
        """Make the slice just parsed (``[start, end)``) durable: its rows, then the trailer that names them."""
        try:
            self.spool.append_slice(rows, end, digest, start_offset=start, deps=deps, receipt_digest=receipt_digest)
        except reindex_spool.SpoolBusy:
            raise FlightBusy() from None

    def published(self, rows) -> None:
        """The re-index is in the index: the spool (if any file exists) has done its job."""
        self.committed, self.rows = True, rows
        if self.spool is not None:
            self.spool.finish()

    def report(self, target, offset) -> dict:
        """The ``reindex`` object of the ``sync`` report. States, in order of precedence: ``disabled`` (no spool for this
        request, ``reason`` says why); ``committed``; ``ready_to_commit`` (everything is staged, the commit did not fit the
        time left); ``discarded`` (a spool that no longer matched was thrown away and the staging started over, ``reason`` says
        why); ``staging``."""
        spool = self.spool
        through, rows = (spool.staged_through, spool.rows) if spool is not None else (0, 0)
        if self.disabled:
            state, reason = "disabled", self.disabled
        elif self.committed:
            state, reason, through, rows = "committed", None, offset, self.rows
        elif self.complete:
            state, reason = "ready_to_commit", "commit_exceeds_budget"
        elif self.discarded:
            state, reason = "discarded", self.discarded
        else:
            state, reason = "staging", None
        return {"state": state, "staged_through": through, "target_bytes": target, "rows": rows, "reason": reason}

    def abort(self) -> None:
        """The request stopped inside a slice: cut what it wrote after the last trailer, let go of the lock."""
        if self.spool is not None:
            self.spool.abort()

    def close(self) -> None:
        if self.spool is not None:
            self.spool.close()


class _NoRows:
    """The row source of a request that publishes nothing (a slice was staged; the index stays as it is)."""
    count, ids, flags, tallies = 0, frozenset(), {}, {}

    def batches(self, size=256):
        return iter(())

    def close(self):
        pass


def _signed64(value: int) -> int:
    """SQLite integers are signed 64-bit; device and file ids may use the whole unsigned range (stored bit for bit)."""
    return value - (1 << 64) if value >= (1 << 63) else value


def _unsigned64(value: int) -> int:
    return value + (1 << 64) if value < 0 else value


def _blockable(provider, path) -> bool:
    """Plain files of every provider but AGY keep a block table. AGY receipts bind to plain prefix digests, and a
    ``.zst`` snapshot is validated by the digest of its (compressed) bytes; both keep full verification."""
    from self_directing_mcp.compressed_source import is_compressed
    return provider != "agy" and not is_compressed(path)


def _cursor_kind(cursor):
    """``"blocks"`` / ``"sha"`` by the cursor's stored hash; None without a cursor."""
    if cursor is None:
        return None
    return "blocks" if is_block_hash(cursor["prefix_hash"]) else "sha"


def capture_for_cursor(provider, path, cursor, checkpoints, max_bytes, *, publish=False, window=None):
    """One full capture collecting what is needed to compare with ``cursor`` (and to publish, with ``publish``).

    ``window=(floor, cap)`` cuts the observation of a plain file short (``session_events.capture_source``)."""
    kind, blockable = _cursor_kind(cursor), _blockable(provider, path)
    blocks = kind == "blocks" or (publish and blockable)
    sha = kind == "sha" or (publish and not blockable) or not blocks
    return capture_source(path, checkpoints=checkpoints, max_bytes=max_bytes, sha=sha, blocks=blocks, window=window)


def _hash_matches(observation, offset, stored) -> bool:
    return observation.cursor_hash(offset, "blocks" if is_block_hash(stored) else "sha") == stored


def _parse_time(value):
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def _plan(provider, path, cursor, record, verify, policy):
    """``(mode, reason)``: whether this request may verify in the cheap way. ``reason`` explains a full scan that a
    fast one could have replaced; None when full was asked for."""
    if not _blockable(provider, path):
        return "full", "legacy_source"
    if cursor is None:
        return "full", "first_sync"
    if verify != "fast":
        return "full", None
    if cursor["path"] != str(path):
        return "full", "path_changed"
    if cursor["parser_version"] != current_parser_version(provider):
        return "full", "parser_updated"
    if not is_block_hash(cursor["prefix_hash"]):
        return "full", "migration"
    if record is None or not record.bound_to(cursor):
        return "full", "no_record"
    verified = _parse_time(record.full_verified_at)
    if verified is None:
        return "full", "unverified"
    age = (policy.clock() - verified).total_seconds()
    # An interval of 0 means every check is full; a clock that jumped back is not evidence of recency.
    if policy.full_verify_interval_sec <= 0 or age > policy.full_verify_interval_sec or age < -60:
        return "full", "interval_elapsed"
    return "fast", None


def index_generation(cursor) -> str:
    """Names one publication of one session's index state (changes with every cursor write)."""
    stamp = f"{cursor['synced_at']}|{cursor['byte_offset']}|{cursor['parser_version']}"
    return "i1:" + hashlib.sha256(stamp.encode()).hexdigest()[:16]


def integrity_report(observation, cursor, record) -> dict:
    """The ``integrity`` object of a response.

    ``mode`` is how THIS request checked the source. ``full_verified_*`` describe the last verification of every
    byte of the indexed prefix, a past event: they say nothing about the source as it is now. A fast request
    compared only the blocks listed in ``fast_checked``.
    """
    at = through = None
    if record is not None and record.bound_to(cursor) and cursor["path"] == str(observation.path):
        at, through = record.full_verified_at, record.full_verified_through
    elif observation.mode == "full":
        # No stored table for this file (a source form without one, or a copy of the indexed source): the scan this
        # request just did is the verification, of as much of the indexed prefix as this file holds.
        at, through = observation.scanned_at, min(cursor["byte_offset"], observation.length)
    report = {"mode": observation.mode, "full_verified_through_byte": through, "full_verified_at": at,
              "source_generation": observation.generation, "index_generation": index_generation(cursor)}
    if observation.mode == "fast":
        report["fast_checked"] = observation.fast_checked
    elif observation.full_reason:
        report["full_reason"] = observation.full_reason
        if observation.lag_sec is not None:
            report["verification_lag_sec"] = round(observation.lag_sec, 3)
    return report


def coverage_from_capture(session_id, provider, record, metrics, observation=None, *, alias=False, max_bytes=None):
    """``alias``: ``observation`` is a verified copy or continuation of the indexed source, which lives at another path.

    ``max_bytes`` bounds the declared decompressed size when an indexed ``.zst`` source is observed again. An
    unreadable compressed source is reported by its own code (``zstd_corrupt``, ``zstd_unavailable``, ...)."""
    count, errors, unsupported = metrics
    result = dict(record or {"provider": provider, "session_id": session_id, "path": None, "byte_offset": 0})
    result.update(chunk_count=count, parse_error_count=errors, unsupported_event_count=unsupported)
    issues = []
    if record is None:
        issues.append("not_synced")
    else:
        if record["parser_version"] != current_parser_version(provider):
            issues.append("parser_outdated")
        try:
            observation = observation or capture_for_cursor(provider, Path(record["path"]), record,
                                                            (record["byte_offset"],), max_bytes)
            if alias:
                result.update(indexed_path=record["path"], requested_path=str(observation.path))
            if (str(observation.path) != record["path"] and not alias) or not observation.validate_prefix():
                issues.append("source_changed")
            # Bytes appended after the observation are unread, never reviewed or lost.
            result["file_size"] = max(observation.length, observation.current_size())
            result["pending_bytes"] = max(0, result["file_size"] - record["byte_offset"])
            if result["pending_bytes"]:
                issues.append("unread_bytes")
            if not _hash_matches(observation, record["byte_offset"], record["prefix_hash"]):
                issues.append("source_changed")
        except RequestStopped:
            raise
        except CompressedSourceError as exc:
            issues.append(exc.code)
        except (OSError, ValueError):
            issues.append("source_unavailable_or_truncated")
    if errors:
        issues.append("parse_errors")
    if unsupported:
        issues.append("unsupported_events")
    result.update(issues=issues, complete=not issues)
    return result


def _full_size(path):
    from self_directing_mcp.agy import full_transcript_size
    return full_transcript_size(path)


def agy_coverage(coverage, stats, receipts=None):
    """AGY-only coverage facts. They do not make ``complete`` false: a legacy result is a classified, known
    limitation (the unsupported_legacy verdict reason), not a current-session defect. They do make the
    result-dependent half of the evidence honest: ``result_dependent_complete`` is false while any result
    predates this conversation's receipts."""
    first = min(receipts) if receipts else None
    coverage["receipts_since"] = (None if first is None else
                                  {"step_index": first, "byte_offset": receipts[first].get("prefix_length")})
    coverage["legacy_unlinked_count"] = stats["legacy_unlinked_results"]
    coverage["result_dependent_complete"] = not stats["legacy_unlinked_results"]
    coverage["agy_evidence"] = stats
    return coverage


@request_operation
def ingest_session(*, path, session_id, store, sparse, dense, embedder, provider="codex",
                   embed=True, derived=True, guard=nullcontext, retain_dense_deletions=False,
                   verify="full", policy=None, slicing=None):
    """Serialize per (index, provider, session). Later callers find the cursor current and skip the parse.

    A parser-version bump therefore reparses a session once per process no matter how many hooks race on it;
    other sessions reparse lazily, each on its own first sync, never in a storm. Copies of one session id in
    several folders share a flight; across processes the state token still decides who publishes.

    ``retain_dense_deletions``: this process has no dense index but the index directory does (a local-only hook
    sharing a server's index). Obsolete ids are removed from the sparse index and stay journalled until a process
    that can also remove their vectors syncs the session.

    ``verify``: ``"full"`` scans the whole source prefix once; ``"fast"`` (hook-time checks) may compare only the
    tail, the latest blocks and a random sample with the stored block digests (``IntegrityPolicy``), and falls
    back to a full scan whenever it cannot vouch. The result's ``integrity`` object says which happened.

    A source change detected while preparing (a rewrite, truncation or replacement between capture and
    publication) discards the observation and recaptures, in full, a bounded number of times.

    ``slicing`` (``SlicePolicy``; None: off): with ``auto``, a request that has a finite deadline syncs a plain file
    in slices. The parse ends at a record boundary when its soft deadline passes (``slice_budget``) and the prefix
    it read is published, so the next call continues from there; a capture is cut short past ``slicing.cap_bytes``.
    Nothing is ever published over events that exist, but a reindex keeps its progress too: its slices wait in a spool
    next to the index (``reindex_spool``) and ONE ordinary replace commit publishes them once they are all there and the
    commit is predicted to fit the time left; until then the index is the old one, byte for byte. The result then has
    a ``sync`` object (with a ``reindex`` object while a reindex is staged), and ``coverage`` says how many bytes of the
    capture are still to be indexed (``backlog_bytes``, the issue ``sync_in_progress``) or that a staged reindex waits for
    its commit (``reindex_ready_to_commit``). A waiter on the session's flight gives up after a bounded wait with
    ``FlightBusy``.
    """
    policy = policy or IntegrityPolicy()
    requested = Path(path).resolve()
    wait = _flight_wait(slicing, requested)
    f_key = _flight_key(store, provider, requested, session_id)
    owner_token = f"{f_key}:{time.monotonic_ns()}"
    info = None
    with session_flight(f_key, **({} if wait is None else {"max_wait": wait})):
        try:
            init_bytes = requested.stat().st_size if requested.exists() else 0
        except OSError:
            init_bytes = 0
        update_session_progress(session_id, provider, flight_key=f_key, owner=owner_token, stage="capturing", bytes_done=0, bytes_total=init_bytes)
        seen, target, captured = [], requested, None
        attempt, whole = 0, False
        try:
            while attempt < _RECAPTURE_ATTEMPTS:
                attempt += 1
                handed, captured = captured, None
                try:
                    info = _ingest_session(path=target, session_id=session_id, store=store, sparse=sparse, dense=dense,
                                           embedder=embedder, provider=provider, embed=embed, derived=derived,
                                           guard=guard, seen=seen, observation=handed, verify=verify, policy=policy,
                                           retain_dense_deletions=retain_dense_deletions, slicing=slicing, whole=whole,
                                           f_key=f_key, owner_token=owner_token)
                except _NeedsWholeCapture:
                    whole, attempt = True, attempt - 1   # not a lost race: the next observation is simply not cut short
                    continue
                except _StaleObservation:
                    continue  # another process published past our capture: observe the file again
                except _IndexedElsewhere as elsewhere:
                    target, captured = elsewhere.path, elsewhere.observation  # a strict prefix: report the indexed source
                    continue
                except SourceChanged:
                    verify = "full"  # whatever moved, the next attempt does not rest on a cheap check
                    continue
                if target != requested:
                    _mark_prefix_alias(info, requested)
                break
            else:
                raise SourceChanged("source changed during preparation")
        except BaseException:
            update_session_progress(session_id, provider, flight_key=f_key, owner=owner_token, clear=True, force=False)
            raise
    deferred = info.pop("_deferred_dense", None) if info is not None else None
    if deferred is not None:
        sid_resolved, prov_resolved, f_key_used, state_snap_used, receipt_store_used, owner_token_used, commit_seq_used = deferred
        return _ingest_dense(
            info=info,
            sid=sid_resolved,
            provider=prov_resolved,
            f_key=f_key_used,
            state_snapshot=state_snap_used,
            receipt_store=receipt_store_used,
            owner_token=owner_token_used,
            commit_seq=commit_seq_used,
            store=store,
            dense=dense,
            embedder=embedder,
            embed=embed,
            guard=guard,
        )
    return info


_RECAPTURE_ATTEMPTS = 3


class _StaleObservation(Exception):
    """The committed cursor lies beyond the captured prefix: our capture predates another process's publication."""


class _NeedsWholeCapture(Exception):
    """A capture cut short for a slice turned out to need a reindex (or belongs to another file than planned): a
    reindex replaces the stored events, so it is all or nothing, as before, and needs the file observed whole."""


class _IndexedElsewhere(Exception):
    """The requested file is a strict prefix of the indexed source: carry that source and its fresh observation."""

    def __init__(self, path, observation):
        super().__init__(str(path))
        self.path, self.observation = path, observation


class AmbiguousSessionSource(ValueError):
    """Two files claim one session id and neither continues the other. The index and cursor are left as they are."""

    def __init__(self, session_id, indexed_path, requested_path):
        super().__init__(f"session {session_id} is indexed from {indexed_path}; {requested_path} claims the same id "
                         "but neither file is a prefix of the other")
        self.session_id, self.indexed_path, self.requested_path = session_id, str(indexed_path), str(requested_path)


def _mark_prefix_alias(info, requested):
    """The request was a strict prefix of the indexed source: the index holds events it does not contain."""
    coverage = info["coverage"]
    info.update(path=str(requested), alias_of=coverage["path"])
    coverage.update(indexed_path=coverage["path"], requested_path=str(requested))
    coverage["issues"].append("requested_source_is_prefix_of_indexed")
    coverage["complete"] = False


def _other_source(cursor, path, observation, session_id, max_bytes=None, provider="codex"):
    """Another file claims the session id whose cursor indexes ``cursor["path"]``. Streaming hashes only; no writes.

    "copy": this file starts with the indexed prefix and the indexed file still exists, "moved": the same but that
    file is gone, "replaced": it is gone and this file does not continue it (reindex, as before). A strict prefix of
    the indexed file raises _IndexedElsewhere; two files that disagree raise AmbiguousSessionSource.
    """
    indexed, offset, kind = Path(cursor["path"]), cursor["byte_offset"], _cursor_kind(cursor)
    continues = observation.length >= offset and _hash_matches(observation, offset, cursor["prefix_hash"])
    present = indexed.is_file()
    if continues:
        return "copy" if present else "moved"
    if not present:
        return "replaced"
    other = capture_for_cursor(provider, indexed, cursor, (offset, observation.length), max_bytes)
    if (other.length >= offset and _hash_matches(other, offset, cursor["prefix_hash"])
            and observation.length < offset
            and other.cursor_hash(observation.length, kind) == observation.cursor_hash(observation.length, kind)):
        raise _IndexedElsewhere(indexed, other)
    raise AmbiguousSessionSource(session_id, indexed, path)


def _flight_key(store, provider, path, session_id):
    """One flight per session id when it is known cheaply, else per file."""
    try:
        sid = session_id or _provider_parser(provider)[1](path)
    except (OSError, ValueError):
        sid = None
    if sid:
        return (str(store.db_path), provider, "session", sid.lower() if provider in ("codex", "agy") else sid)
    return (str(store.db_path), provider, "path", str(path))


def _provider_parser(provider):
    if provider == "agy":
        from self_directing_mcp.agy import parse_session_chunks, peek_session_id
    elif provider == "omp":
        from self_directing_mcp.omp import parse_session_chunks, peek_session_id
    elif provider == "claude":
        from self_directing_mcp.claude import parse_session_chunks, peek_session_id
    elif provider == "grok":
        from self_directing_mcp.grok import parse_session_chunks, peek_session_id
    elif provider == "grokbot":
        from self_directing_mcp.grokbot.parse import parse_session_chunks, peek_session_id
    else:
        from self_directing_mcp.codex.parse import parse_session_chunks, peek_session_id
    return parse_session_chunks, peek_session_id


def _draft(observation, offset, record):
    """The block table to publish with a cursor at ``offset``. A full scan compared every block just now; a fast
    check keeps the previous verification stamps (it verified nothing in full)."""
    table = observation.table_at(offset)
    stamp = observation.stamp
    if observation.mode == "full":
        verified_at, through = observation.scanned_at, offset
    elif record is not None:
        verified_at, through = record.full_verified_at, record.full_verified_through
    else:
        verified_at = through = None
    return BlockDraft(table, stamp[0], stamp[1], stamp[2], stamp[3], verified_at, through)


def _capture_observation(path, provider, known, record, hint, verify, policy, max_bytes, window=None):
    """The request's capture: a fast check when allowed and possible, otherwise the one full scan of the prefix.

    ``window=(floor, cap)``: a slice's capture, cut short past ``floor`` (the indexed offset) by ``cap`` bytes."""
    mode, reason = _plan(provider, path, known, record, verify, policy)
    lag = None
    if reason == "interval_elapsed":
        lag = (policy.clock() - _parse_time(record.full_verified_at)).total_seconds() - policy.full_verify_interval_sec
    if mode == "fast":
        try:
            observation = capture_source_fast(path, record, recent=policy.recent_blocks, samples=policy.sample_blocks,
                                              rng=policy.new_rng(), cap=None if window is None else window[1])
        except FastFallback as fallback:
            reason = fallback.reason  # the cheap signals (or a block) say this must be settled by a full scan
        else:
            observation.scanned_at = policy.clock().isoformat()
            return observation
    observation = capture_for_cursor(provider, path, known, hint, max_bytes, publish=True, window=window)
    observation.full_reason, observation.lag_sec = reason, lag
    observation.scanned_at = policy.clock().isoformat()
    return observation


def _ingest_session(*, path, session_id, store, sparse, dense, embedder, provider, embed, derived, guard, seen,
                    observation=None, retain_dense_deletions=False, verify="full", policy=None, slicing=None,
                    whole=False, f_key=None, owner_token=None):
    if embed and (dense is None or embedder is None):
        raise ValueError("embedding unavailable in local-only mode")
    parse_session_chunks, peek_session_id = _provider_parser(provider)
    path = Path(path).resolve()
    policy = policy or IntegrityPolicy()
    reports, sliceable = _may_slice(slicing, path)
    stageable = sliceable                 # a re-index may be staged (a capture that is whole is needed, and taken, either way)
    sliceable = sliceable and not whole   # ``whole``: this call observes the file whole (it needs a reindex)
    planned = None  # the cursor a fast capture was planned from; it must still be the current one
    if observation is None:
        hint, known, record, sid_hint = (), None, None, session_id
        if not sid_hint:
            try:
                sid_hint = peek_session_id(path)
            except (OSError, ValueError):
                sid_hint = None
        looked = bool(sid_hint)   # the cursor of this session was read, so what is "indexed" is known
        if sid_hint:
            try:
                with guard(), store._read_boundary():
                    known = store.cursor_record(sid_hint.lower() if provider in ("codex", "agy") else sid_hint, provider)
                    record = store.source_block_record(known["session_id"], provider) if known else None
                hint = (known["byte_offset"],) if known and type(known["byte_offset"]) is int else ()
                if provider == "agy":
                    # Every receipt anchors to a prefix digest and the ledger to a verified length: hash the capture
                    # at those offsets and none of them costs a second read of the file.
                    from self_directing_mcp.agy_receipts import ReceiptStore
                    hint += ReceiptStore(store.db_path.parent, held=store.derived if store.defer_derived else None,
                                         journal=store.journal_policy).checkpoints(sid_hint.lower(), path)
                if stageable and known:
                    # A staged re-index continues from where its spool ends: that offset's hash is a checkpoint of the
                    # one pass over the source, and nothing is read twice to verify the staged range.
                    staged = reindex_spool.ReindexSpool.peek(store.db_path.parent, provider, known["session_id"],
                                                             current_parser_version(provider))
                    if staged:
                        hint += (staged,)
            except (OSError, sqlite3.Error):
                hint, known, record, looked = (), None, None, False
        # A capture is cut short only where a prefix commit can only add: nothing indexed yet, or an append to the
        # file the cursor is on with the current parser. A cursor that is known to need a reindex is not cut.
        window = None
        if sliceable and looked and (known is None or (
                known["path"] == str(path) and type(known["byte_offset"]) is int
                and known["parser_version"] == current_parser_version(provider))):
            window = (0 if known is None else known["byte_offset"], slicing.cap_bytes)
        observation = _capture_observation(path, provider, known, record, hint, verify, policy,
                                           store.max_decompressed_bytes, window)
        planned = known
    if observation.scanned_at is None:  # an observation handed in by a retry: its scan finished before now
        observation.scanned_at = policy.clock().isoformat()
    with source_scope(observation):
        sid = session_id or peek_session_id(path)
        if not sid:
            from self_directing_mcp.codex.discover import session_id_from_filename
            sid = session_id_from_filename(path) if provider == "codex" else None
        if not sid:
            raise ValueError("cannot determine session id")
        sid = sid.lower() if provider in ("codex", "agy") else sid
        update_session_progress(sid, provider, flight_key=f_key, owner=owner_token, stage="parsing", bytes_done=0, bytes_total=observation.target_end)
        claimed = peek_session_id(path)
        if claimed is not None and (claimed.lower() if provider in ("codex", "agy") else claimed) != sid:
            raise ValueError("session identity differs from observed source")
        receipt_store, receipts, fallback_retries, cursor_deps = None, {}, (), None
        with guard(), store._read_boundary():
            cursor = store.cursor_record(sid, provider)
            record = store.source_block_record(sid, provider) if cursor else None
            expected_token = store.state_token(sid, provider)
            prior_calls = (store.prior_calls(sid, provider)
                           if provider == "omp" and cursor and observation.complete_end > cursor["byte_offset"] else ())
            if provider == "agy":
                from self_directing_mcp.agy_receipts import ReceiptStore
                fallback_retries = store.agy_fallback_retries(sid) if cursor else ()
                cursor_deps = store.receipt_deps(sid, provider, cursor["synced_at"]) if cursor else None
                receipt_store = ReceiptStore(store.db_path.parent, held=store.derived if store.defer_derived else None,
                                             journal=store.journal_policy)
                receipt_capture = receipt_store.read_capture(sid)
                receipt_authority_before = receipt_store.authority_token(sid)
                if (receipt_authority_before[0] != "present"
                        or receipt_authority_before[2] != receipt_store.rows_digest(receipt_capture[0])):
                    raise ValueError("receipt authority changed during capture")
        if observation.mode == "fast" and (cursor is None or planned is None
                                           or cursor["synced_at"] != planned["synced_at"]):
            raise _StaleObservation()  # the table the check used belongs to a publication that is gone
        if cursor and cursor["path"] == str(path) and type(cursor["byte_offset"]) is int \
                and cursor["byte_offset"] > observation.length:
            # Never rewind a newer cursor from an older capture. Only a recapture that is still shorter than
            # the cursor we already saw is a genuine truncation, which reindexes below as before.
            if not seen or observation.length >= seen[-1]:
                seen.append(cursor["byte_offset"])
                raise _StaleObservation()
        # A file with another path but the same session id: AGY hook receipts bind to an exact path, so it keeps
        # reindexing on a path change; elsewhere a copy or continuation of the indexed prefix is not a new source.
        relation = None
        if observation.limited_from is not None and cursor and cursor["path"] != str(path):
            raise _NeedsWholeCapture()   # the cut was planned for an append to this file; the cursor is on another
        if cursor and provider != "agy" and cursor["path"] != str(path) and type(cursor["byte_offset"]) is int:
            relation = _other_source(cursor, path, observation, sid, store.max_decompressed_bytes, provider)
        parser_version = current_parser_version(provider)
        start = cursor["byte_offset"] if cursor else 0
        reindexed, reindex_reason = False, None
        if cursor:
            try:
                changed = relation not in ("copy", "moved") and (
                    str(path) != cursor["path"] or not _hash_matches(observation, start, cursor["prefix_hash"]))
            except ValueError:
                changed = True
            if changed:
                start, reindexed, reindex_reason = 0, True, "source_changed"
            elif cursor["parser_version"] != parser_version:
                start, reindexed, reindex_reason = 0, True, "parser_updated"
            elif fallback_retries and any(size != _full_size(path) for size in fallback_retries):
                # A fallback that failed because the full log was missing or not yet at that step: it is retried
                # once per change of the full log, and only for those transient failures.
                start, reindexed, reindex_reason = 0, True, "truncated_fallback_retry"
        receipt_digest = receipt_ledger = None
        receipts_dirty = False
        if receipt_store is not None:
            receipt_rows, receipts, receipt_digest, indexed_digest, receipt_ledger = receipt_store.prepare_bindings(
                receipt_capture, path, session_id=sid, observation=observation)
            from self_directing_mcp.agy_receipts import affects_index
            receipts_dirty = bool(cursor) and (receipt_digest != indexed_digest
                                               or cursor.get("receipt_digest") != receipt_digest)
            # The receipts moved since the indexed events were built. Those events are still what a fresh parse
            # would produce unless a receipt they were derived from changed (agy_receipts.dependencies): the usual
            # case is a receipt for a step that has no result yet. Anything unproven reparses from byte 0.
            if receipts_dirty and (reindexed or affects_index(cursor_deps, receipt_rows)):
                start, reindexed, reindex_reason = 0, True, "hook_receipts_changed"
        if reindexed and observation.limited_from is not None:
            raise _NeedsWholeCapture()   # a reindex replaces events: never a prefix over an index that exists
        index_dir, kind = store.db_path.parent, "blocks" if _blockable(provider, path) else "sha"
        stage, stage_only = None, False
        if not reindexed:
            # The index is current for this source and parser: a spool of this session is an orphan (another process
            # finished the re-index, or the source went back to what is indexed).
            reindex_spool.ReindexSpool.discard_session(index_dir, provider, sid)
        elif stageable and observation.mode == "full":
            # A re-index replaces what is stored: it is staged slice by slice in a spool and published by ONE commit, so
            # a request that runs out of time keeps its parse and the index is not touched until then. (A capture that did
            # not scan the whole prefix cannot vouch for the staged range: that re-index is the all-or-nothing one.)
            stage = _Stage.begin(store=store, provider=provider, sid=sid, cursor=cursor, observation=observation, path=path,
                                 kind=kind, parser_version=parser_version, receipt_digest=receipt_digest,
                                 receipt_rows=receipt_rows if receipt_store is not None else None)
        staging = stage is not None and stage.active
        if staging:
            start = stage.start          # where the staged range ends: the parse continues from there
        # The parse that adds to what is stored, and a staged slice of a re-index, may stop early at their soft deadline;
        # any other re-index is all or nothing, as before.
        budget, chunks, uses = None, _NoRows(), set()
        try:   # from here on whatever ends the request lets go of the spool (its lock is a file descriptor)
            if (sliceable and not reindexed) or staging:
                # What a staged slice holds is judged by nobody until it is published (the audit that follows reads the OLD
                # index), so only a slice that adds to what is stored is sized by what judging it costs, and only that one
                # reads the contracts and the index for the numbers.
                budget = _new_budget(slicing, start, observation.complete_end - start, judged=not staging)
            chunks = RowSpool(track_calls=staging and provider == "omp")
            parse_started = time.perf_counter()
            with slice_scope(budget):
                if cursor and not reindexed and provider != "agy" and start >= observation.complete_end:
                    offset, parsed_sid = start, sid
                elif staging and start >= observation.complete_end:
                    offset, parsed_sid = start, sid       # everything is staged already: only the commit is left
                elif provider == "agy":
                    _, offset, parsed_sid = parse_session_chunks(path, session_id=sid, start_byte=start,
                                                                 receipts=receipts, sink=chunks, uses=uses)
                elif provider == "omp" and start:
                    # This lookup is request-owned and bound to the captured cursor generation (or, for a staged re-index,
                    # to the calls of the slices staged so far).
                    earliest = dict(stage.spool.calls) if staging else {}
                    for before, call_id, name in () if staging else prior_calls:
                        key = (call_id, name)
                        earliest[key] = min(earliest.get(key, before), before)
                    _, offset, parsed_sid = parse_session_chunks(
                        path, session_id=sid, start_byte=start, sink=chunks,
                        prior_call=lambda call_id, name, before_byte: earliest.get((call_id, name), float("inf")) < before_byte)
                else:
                    _, offset, parsed_sid = parse_session_chunks(path, session_id=sid, start_byte=start, sink=chunks)
            if parsed_sid != sid:
                raise ValueError("session id changed while parsing")
            spent = time.perf_counter() - parse_started       # parse + commit of this slice, for the cost model
            blockable = _blockable(provider, path)
            digest = observation.cursor_hash(offset, "blocks" if blockable else "sha")
            draft = _draft(observation, offset, record) if blockable else None
            receipt_deps = slice_deps = None
            if receipt_store is not None:
                from self_directing_mcp.agy_receipts import dependencies
                # An append extends what the stored events depend on; without stored dependencies (an index from
                # before they were kept) there is nothing to extend and the next receipt change reparses.
                prior = None if reindexed or not cursor else cursor_deps
                if reindexed or not cursor or prior is not None:
                    # A staged re-index: every slice has used some receipts; the events depend on all of them.
                    used = uses | set(stage.spool.deps_steps) if staging else uses
                    receipt_deps = json.dumps(dependencies(receipt_rows, used, prior), sort_keys=True)
                    slice_deps = dependencies(receipt_rows, uses, None) if staging else None
            if staging:
                stage.complete = offset >= observation.complete_end
                # A re-index that is parsed whole in this request and whose commit fits is published straight from memory,
                # as before: no file is made for it. Anything else is staged, so that the work outlives this request.
                if not (stage.complete and not stage.spool.slices and _commit_fits(chunks.count)):
                    stage.persist(chunks, start, offset, digest, deps=slice_deps, receipt_digest=receipt_digest)
                    chunks.close()
                    if stage.complete and _commit_fits(stage.spool.rows):
                        # The capture is the verification of [0, target_end): it scanned the whole prefix in this request,
                        # and every staged slice was bound to it by the hash of the range it ended at.
                        if not (observation.mode == "full" and observation.limited_from is None
                                and offset == observation.complete_end == observation.target_end):
                            raise AssertionError("a staged re-index is committed only against a whole capture")
                        chunks = stage.spool.as_row_source()
                    else:
                        chunks, stage_only = _NoRows(), True    # the index is left as it is; the staged rows wait
        except BaseException:
            chunks.close()
            if stage is not None:
                stage.abort()
            raise
    try:
        check_request()
        # Bounded re-checks only: the capture was this request's one full scan of the prefix (see validate_prefix).
        if not observation.validate_prefix():
            raise SourceChanged("source changed during preparation")
        with guard():
            if not observation.validate_prefix(locked=True):
                raise SourceChanged("source changed during preparation")
            current = store.cursor_record(sid, provider)
            # A request that only stages publishes nothing, so it has no state to compare-and-swap.
            changed_token = not stage_only and store.state_token(sid, provider) != expected_token
            equivalent = bool(changed_token and current and current["path"] == str(path)
                              and current["byte_offset"] == offset and current["prefix_hash"] == digest
                              and current["parser_version"] == parser_version)
            if equivalent:
                for batch in chunks.batches():
                    live_rows = dict(store.capture_evidence(sid, provider, history=False,
                                                           anchors=[row[0] for row in batch]).rows)
                    if not all(live_rows.get(row[0]) == row[-1] for row in batch):
                        equivalent = False
                        break
                if receipt_store is not None:
                    equivalent = (equivalent and current.get("receipt_digest") == receipt_digest
                                  and receipt_store.authority_token(sid)[1] == receipt_authority_before[1]
                                  and receipt_store.read_capture(sid)[0] == receipt_rows)
            if changed_token and not equivalent:
                raise ValueError("stale session preparation")
            if stage_only:
                pass    # the staged slice is on disk; the index is not touched until the re-index is complete
            elif not equivalent:
                if receipt_store is not None:
                    if receipt_store.authority_token(sid) != receipt_authority_before:
                        raise ValueError("receipt authority changed during preparation")
                    receipt_store.publish_bindings(sid, receipt_capture, receipt_rows, receipt_ledger)
                # A cursor without a block table (written before this version, by an older process, or after a
                # change of source form) is republished with one, even when nothing else changed.
                own = draft is not None and cursor is not None and cursor["path"] == str(path) and relation is None
                bound = own and record is not None and record.bound_to(cursor)
                needs_table = own and not bound
                if (chunks.count or reindexed or cursor is None or offset != cursor["byte_offset"]
                        or relation == "moved" or needs_table or receipts_dirty):
                    # receipts_dirty: the events stand, but they now belong to these receipts (and their dependencies).
                    committing = time.perf_counter()
                    try:
                        store.commit_events((), session_id=sid, provider=provider, path=path, offset=offset,
                                            digest=digest, replace=reindexed, parser_version=parser_version,
                                            expected_token=expected_token, prepared_rows=chunks,
                                            receipt_digest=receipt_digest, blocks=draft, receipt_deps=receipt_deps)
                    except reindex_spool.SpoolError:
                        if staging:
                            stage.spool.discard("corrupt")   # a frame did not verify: the commit rolled back, start over
                        raise
                    if not reindexed:       # what an appended event costs to parse and commit (a replace costs more)
                        COSTS.observe_slice(provider, chunks.count, spent + time.perf_counter() - committing)
                    if staging:
                        stage.published(chunks.count)
                    elif reindexed:
                        # Not staged (slicing off, no deadline, a compressed source): a spool of this session that an
                        # earlier request left is moot now.
                        reindex_spool.ReindexSpool.discard_session(index_dir, provider, sid)
                elif bound and observation.mode == "full":
                    # Nothing to publish, but every block was just compared: stamp the stored table.
                    stamp = observation.stamp
                    store.mark_full_verified(sid, provider, record, at=observation.scanned_at,
                                             through=record.table.covered, dev=stamp[0], ino=stamp[1],
                                             size=stamp[2], mtime_ns=stamp[3])
                if receipt_store is not None:
                    receipt_store.mark_indexed(sid, receipt_digest)
            else:
                if staging:
                    stage.published(chunks.count)    # another process published this very re-index meanwhile
                chunks.count = 0
            with store._read_boundary():
                committed_token = store.state_token(sid, provider)
                committed_cursor = store.cursor_record(sid, provider)
                committed_record = store.source_block_record(sid, provider)
                metrics = store.session_metrics(sid, provider)
                receipt_authority = receipt_store.authority_token(sid) if receipt_store is not None else None
                agy_stats = store.agy_evidence_stats(sid) if provider == "agy" else None
    finally:
        new_chunks = chunks.count
        chunks.close()
        if stage is not None:
            stage.close()
    if stage_only:
        # Nothing was replaced: the cursor and the events are the old ones. ``reindex_reason`` keeps saying why a re-index is
        # under way, ``reindexed`` that it has not happened yet.
        offset, reindexed = committed_cursor["byte_offset"], False
    alias = committed_cursor["path"] != str(path)  # an equal or shorter copy never moves the cursor
    coverage = coverage_from_capture(sid, provider, committed_cursor, metrics, observation, alias=alias,
                                     max_bytes=store.max_decompressed_bytes)
    if agy_stats is not None:
        agy_coverage(coverage, agy_stats, receipts)
    sync = None
    if reports:
        # What the capture saw and the index does not hold yet. Not the same as ``unread_bytes`` (everything past the
        # cursor, appended after the capture included): a backlog means this sync is not done, so the verdict can never
        # be "clean as of the cursor" (``engine._mark_growth`` keeps ``judged_prefix`` for pure append only).
        indexed = committed_cursor["byte_offset"]
        backlog = max(0, observation.target_end - indexed)
        if backlog:   # absent when there is none: a finished sync's coverage is what it always was
            coverage["backlog_bytes"] = backlog
            coverage["issues"].append("sync_in_progress")
            coverage["complete"] = False
        reindex = stage.report(observation.target_end, offset) if stage is not None else None
        if reindex is not None and reindex["state"] in ("staging", "discarded", "ready_to_commit"):
            # The old index is still the one in force (coverage keeps ``parser_outdated`` / ``source_changed``): the verdict
            # is never clean while the re-index waits, and the issue says what it waits for.
            waiting = "reindex_ready_to_commit" if reindex["state"] == "ready_to_commit" else "sync_in_progress"
            if waiting not in coverage["issues"]:
                coverage["issues"].append(waiting)
            coverage["complete"] = False
        sync = _sync_report(indexed, observation.target_end, coverage.get("pending_bytes", backlog), budget, observation,
                            reindex, provider)
    info = {"session_id": sid, "provider": provider, "path": str(path), "byte_offset": offset,
            "new_chunks": new_chunks, "embedded": 0, "total_chunks": coverage["chunk_count"],
            "reindexed": reindexed, "parser_version": parser_version, "reindex_reason": reindex_reason,
            "dense_restored": 0, "embedding_pending": 0, "embedding_error": None,
            "sparse_updated": 0, "coverage": coverage,
            "integrity": integrity_report(observation, committed_cursor, committed_record),
            "_source_observation": observation, "_metadata_token": committed_token, "_receipt_token": receipt_authority}
    if sync is not None:
        info["sync"] = sync
    if reindexed:
        # What this request's verification found: the index was rebuilt from the source as it is now. For
        # "source_changed" that is a mismatch of the stored digests, never a runtime failure.
        info["integrity"]["reindex_reason"] = reindex_reason
    if alias:
        info["alias_of"] = committed_cursor["path"]
    state_snap = (committed_token, observation, receipt_authority, offset)
    commit_seq = _notify_embed_flight_commit(f_key, state_snap, request_embed=bool(derived and embed))
    update_session_progress(
        sid,
        provider,
        flight_key=f_key,
        owner=owner_token,
        stage="sparse_indexing" if derived else "done",
        bytes_done=offset,
        bytes_total=observation.target_end,
        chunks_done=coverage["chunk_count"],
        chunks_total=coverage["chunk_count"],
    )
    if not derived:
        update_session_progress(sid, provider, flight_key=f_key, owner=owner_token, clear=True)
        return info

    def finish():
        source_ok = observation.validate_prefix()  # bounded; see validate_prefix
        with guard():
            issues = []
            if store.state_token(sid, provider) != committed_token:
                issues.append("metadata_changed_during_derived_work")
            if not (source_ok and observation.validate_prefix(locked=True)):
                issues.append("source_changed_during_derived_work")
            elif observation.current_size() > info["coverage"].get("file_size", 0) and "unread_bytes" not in info["coverage"]["issues"]:
                info["coverage"]["pending_bytes"] = max(0, observation.current_size() - offset)
                issues.append("unread_bytes")
            if receipt_store is not None and receipt_store.authority_token(sid) != receipt_authority:
                issues.append("receipts_changed_during_derived_work")
            if issues:
                info["coverage"]["complete"] = False
                info["coverage"]["issues"].extend(issues)
        return info

    # Metadata-first repair remains replayable. Acknowledge exactly this batch.
    with guard():
        if dense is not None:
            dense.refresh()
        pending = store.pending_deletions(sid, provider)
        stale = list(set(pending) - store.chunk_ids(sid, provider))
        if stale:
            check_request()
            sparse.delete_ids(stale)
            if dense is not None:
                dense.delete_ids(stale)
        if pending and not (dense is None and retain_dense_deletions):
            store.finish_deletions(sid, provider, pending)
        session_ids = sorted(store.chunk_ids(sid, provider))
        indexed = sparse.present(session_ids)
        missing_ids = [cid for cid in session_ids if cid not in indexed]
    for batch in _batches(missing_ids):
        with guard():
            snapshot = store.capture_evidence(sid, provider, history=False, anchors=batch)
        documents = snapshot.list_chunks(sid, provider)
        with guard():
            live = store.live_ids(sid, provider, [chunk.chunk_id for chunk in documents])
            documents = [chunk for chunk in documents if chunk.chunk_id in live]
            if documents:
                sparse.upsert_many(documents)
        info["sparse_updated"] += len(documents)
    # Sparse index deletions and upserts above are completed strictly inside session_flight.
    update_session_progress(sid, provider, flight_key=f_key, owner=owner_token, clear=True, force=False)
    if embedder is None:
        return finish()
    info["_deferred_dense"] = (sid, provider, f_key, state_snap, receipt_store, owner_token, commit_seq)
    return info


def _finalize_post_derived_coverage(
    info: dict,
    *,
    sid: str,
    provider: str,
    store,
    guard,
    expected_token,
    observation,
    receipt_store,
    receipt_authority,
    offset: int,
) -> dict:
    source_ok = True if observation is None else observation.validate_prefix()
    with guard():
        issues = []
        if store.state_token(sid, provider) != expected_token:
            issues.append("metadata_changed_during_derived_work")
        if observation is not None:
            if not (source_ok and observation.validate_prefix(locked=True)):
                issues.append("source_changed_during_derived_work")
            elif observation.current_size() > info["coverage"].get("file_size", 0) and "unread_bytes" not in info["coverage"]["issues"]:
                info["coverage"]["pending_bytes"] = max(0, observation.current_size() - offset)
                issues.append("unread_bytes")
        if receipt_store is not None and receipt_store.authority_token(sid) != receipt_authority:
            issues.append("receipts_changed_during_derived_work")
        if issues:
            info["coverage"]["complete"] = False
            for iss in issues:
                if iss not in info["coverage"]["issues"]:
                    info["coverage"]["issues"].append(iss)
    return info


def _ingest_dense(
    *,
    info,
    sid,
    provider,
    f_key,
    state_snapshot,
    receipt_store,
    owner_token,
    commit_seq=0,
    store,
    dense,
    embedder,
    embed,
    guard,
):
    caller_token, caller_obs, caller_receipt_auth, caller_offset = state_snapshot
    with _embed_flight(f_key, blocking=embed, state_snapshot=state_snapshot, commit_seq=commit_seq) as (acquired, flight_entry, holder_state):
        if not acquired:
            with guard():
                dense.refresh()
                session_ids = sorted(store.chunk_ids(sid, provider))
                indexed = dense.present(session_ids)
                info["embedding_pending"] = sum(1 for cid in session_ids if cid not in indexed)
            prog = get_session_progress(sid, provider)
            if prog is None and info["embedding_pending"] > 0:
                prog = {
                    "stage": "embedding",
                    "session_id": sid,
                    "provider": provider,
                    "embedded_done": 0,
                    "embedded_total": info["embedding_pending"],
                    "elapsed_ms": 0.0,
                    "eta_ms": None,
                }
            if prog is not None:
                info["sync_progress"] = prog
            return _finalize_post_derived_coverage(
                info,
                sid=sid,
                provider=provider,
                store=store,
                guard=guard,
                expected_token=caller_token,
                observation=caller_obs,
                receipt_store=receipt_store,
                receipt_authority=caller_receipt_auth,
                offset=caller_offset,
            )
        active_embed = bool(embed)
        try:
            cache_key = embedder.cache_key
            embedding_failed = False
            stale_abort = False
            newer_commit_restarts = 0
            while True:
                check_request()
                with _embed_flights_gate:
                    if active_embed:
                        flight_entry[3] = False
                        flight_entry[5] = False
                    latest_snap = flight_entry[4] or state_snapshot
                active_token, active_obs, active_receipt_auth, active_offset = latest_snap
                with guard():
                    dense.refresh()
                    session_ids = sorted(store.chunk_ids(sid, provider))
                    indexed = dense.present(session_ids)
                    candidate_ids = [cid for cid in session_ids if cid not in indexed]
                if active_embed:
                    update_session_progress(
                        sid,
                        provider,
                        flight_key=f_key,
                        owner=owner_token,
                        stage="embedding",
                        embedded_done=0,
                        embedded_total=len(candidate_ids),
                        force=True,
                    )
                restored_in_pass = 0
                restart_for_newer_commit = False
                for batch in _batches(candidate_ids):
                    with guard():
                        snapshot = store.capture_evidence(sid, provider, history=False, anchors=batch)
                    candidates = snapshot.list_chunks(sid, provider)
                    text_by_hash, chunk_hashes = {}, {}
                    for chunk in candidates:
                        check_request()
                        sanitized = mask_secrets(chunk.text)
                        key = hashlib.sha256(sanitized.encode("utf-8")).hexdigest()
                        text_by_hash[key], chunk_hashes[chunk.chunk_id] = sanitized, key
                    with guard():
                        vectors = {key: store.cached_vector(cache_key, key) for key in text_by_hash}
                    missing = [key for key, value in vectors.items() if value is None]
                    try:
                        for start_idx in range(0, len(missing) if active_embed and not embedding_failed else 0, 32):
                            check_request()
                            keys = missing[start_idx:start_idx + 32]
                            values = embedder.embed_documents([text_by_hash[key] for key in keys])
                            check_request()
                            if len(values) != len(keys):
                                raise ValueError("embedding response count mismatch")
                            if any(np.asarray(v).shape != (embedder.dim,) or not np.isfinite(v).all() for v in values):
                                raise ValueError("invalid embedding shape or values")
                            with guard():
                                store.cache_vectors(cache_key, zip(keys, values))
                            vectors.update(zip(keys, (np.asarray(v, dtype=np.float32) for v in values)))
                            info["embedded"] += len(keys)
                            if active_embed:
                                batch_done = sum(1 for c in candidates if vectors.get(chunk_hashes[c.chunk_id]) is not None)
                                update_session_progress(
                                    sid,
                                    provider,
                                    flight_key=f_key,
                                    owner=owner_token,
                                    stage="embedding",
                                    embedded_done=min(len(candidate_ids), restored_in_pass + batch_done),
                                    embedded_total=len(candidate_ids),
                                    force=True,
                                )
                    except (RequestStopped, IndexBusy):
                        raise
                    except Exception as exc:
                        info["embedding_error"] = type(exc).__name__
                        embedding_failed = True
                    restored_all = [(c.chunk_id, vectors[chunk_hashes[c.chunk_id]]) for c in candidates
                                    if vectors[chunk_hashes[c.chunk_id]] is not None]
                    obs_ok_unlocked = True if active_obs is None else active_obs.validate_prefix()
                    with _embed_flights_gate:
                        newest_snap = flight_entry[4]
                    newest_obs = newest_snap[1] if newest_snap is not None else None
                    newest_obs_ok_unlocked = True if newest_obs is None else newest_obs.validate_prefix()
                    with guard():
                        check_request()
                        dense.refresh()
                        cur_store_token = store.state_token(sid, provider)
                        token_ok = (cur_store_token == active_token)
                        obs_ok = obs_ok_unlocked and (True if active_obs is None else active_obs.validate_prefix(locked=True))
                        all_cand_ids = [c.chunk_id for c in candidates]
                        live = store.live_ids(sid, provider, all_cand_ids)
                        already = dense.present(all_cand_ids)
                        if token_ok and obs_ok:
                            restored = [pair for pair in restored_all if pair[0] in live and pair[0] not in already]
                            if restored:
                                dense.upsert_many(restored)
                        elif (
                            active_embed
                            and not token_ok
                            and newest_snap is not None
                            and newest_snap[0] == cur_store_token
                            and cur_store_token is not None
                            and newest_obs_ok_unlocked
                            and (True if newest_obs is None else newest_obs.validate_prefix(locked=True))
                            and newer_commit_restarts < 8
                        ):
                            # An in-process follower committed a strictly newer session state while we were embedding.
                            # Verify live chunk hashes against the current store snapshot before upserting, then restart pass on newest_snap.
                            cur_snap = store.capture_evidence(sid, provider, history=False, anchors=all_cand_ids)
                            cur_hashes = {
                                c.chunk_id: hashlib.sha256(mask_secrets(c.text).encode("utf-8")).hexdigest()
                                for c in cur_snap.list_chunks(sid, provider)
                            }
                            restored = [
                                pair for pair in restored_all
                                if pair[0] in live
                                and pair[0] not in already
                                and cur_hashes.get(pair[0]) == chunk_hashes.get(pair[0])
                            ]
                            if restored:
                                dense.upsert_many(restored)
                            restart_for_newer_commit = True
                            newer_commit_restarts += 1
                        else:
                            restored = []
                            stale_abort = True
                    info["dense_restored"] += len(restored)
                    restored_in_pass += len(restored)
                    if active_embed:
                        update_session_progress(
                            sid,
                            provider,
                            flight_key=f_key,
                            owner=owner_token,
                            stage="embedding",
                            embedded_done=min(len(candidate_ids), restored_in_pass),
                            embedded_total=len(candidate_ids),
                            force=True,
                        )
                    if stale_abort or restart_for_newer_commit:
                        break
                if restart_for_newer_commit and not stale_abort:
                    continue
                with guard():
                    dense.refresh()
                    final_sids = sorted(store.chunk_ids(sid, provider))
                    final_indexed = dense.present(final_sids)
                    info["embedding_pending"] = sum(1 for cid in final_sids if cid not in final_indexed)
                with _embed_flights_gate:
                    can_rerun = (
                        active_embed
                        and not stale_abort
                        and bool(flight_entry[3])
                        and (not embedding_failed or not flight_entry[6])
                        and newer_commit_restarts < 8
                    )
                    if can_rerun:
                        newer_commit_restarts += 1
                        if embedding_failed:
                            flight_entry[6] = True
                        continue
                    # Atomically clear holder_state, active flag, and release lock under _embed_flights_gate so finally will not touch successor's lock.
                    if holder_state[0]:
                        holder_state[0] = False
                        flight_entry[2] = False
                        flight_entry[0].release()
                    break
            return _finalize_post_derived_coverage(
                info,
                sid=sid,
                provider=provider,
                store=store,
                guard=guard,
                expected_token=caller_token,
                observation=caller_obs,
                receipt_store=receipt_store,
                receipt_authority=caller_receipt_auth,
                offset=caller_offset,
            )
        finally:
            if active_embed or embed:
                update_session_progress(sid, provider, flight_key=f_key, owner=owner_token, clear=True, force=True)
