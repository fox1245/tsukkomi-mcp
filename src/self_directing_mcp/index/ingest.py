from __future__ import annotations

import hashlib
from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import json
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from self_directing_mcp.schemas import AuditEvent, Chunk
from self_directing_mcp.security.mask import mask_secrets
from self_directing_mcp.request_control import IndexBusy, RequestStopped, check_request, publication, request_operation
from self_directing_mcp.session_events import capture_source, observed_source, source_scope


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
                   "tool_name", "truncated_fields", "denied_action")


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
        frames = []
        for order, chunk in enumerate(self.list_chunks(session_id, provider)):
            meta = chunk.meta
            frames.append({"chunk_id": chunk.chunk_id, "order": order, "kind": chunk.kind,
                           "text": chunk.text, "timestamp": chunk.timestamp,
                           "tool_name": meta.get("tool_name"), "success": meta.get("success"),
                           "error_text": meta.get("error") or meta.get("error_text")})
        return frames


_BATCH = 500


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


def agy_tally(linkage, pending, media_count, event_type, still, restored, fallback=None):
    """Counter increments for one AGY event's evidence classes; shared by commit and the scan fallback."""
    from self_directing_mcp.agy import JUDGED_FIELDS
    out = []
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
            "media_placeholders": counter["media_placeholders"],
            "truncation": {"restored_from_full": bucket("restored"), "unrestored": bucket("unrestored"),
                           "ignored": bucket("ignored")}}


class RowSpool:
    """Prepared event rows spooled to disk (memory below 1 MiB); parsers append, publication replays in batches."""

    def __init__(self):
        self._file = tempfile.SpooledTemporaryFile(max_size=1 << 20, mode="w+", encoding="utf-8", newline="\n")
        self.count, self.ids, self.flags, self.tallies = 0, set(), {}, {}

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
        if chunk.meta.get("parse_error") or chunk.meta.get("unsupported_event"):
            self.flags[chunk.chunk_id] = (bool(chunk.meta.get("parse_error")), bool(chunk.meta.get("unsupported_event")))
        if chunk.provider == "agy":
            meta = chunk.meta
            tally = agy_tally(meta.get("linkage"), meta.get("pending"), meta.get("media_count"), meta.get("event_type"),
                              meta.get("truncated_fields"), meta.get("restored_fields"), meta.get("fallback"))
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

    def close(self):
        self._file.close()


_flights, _flights_gate = {}, threading.Lock()


@contextmanager
def session_flight(key):
    """Single-flight per (index, provider, session): at most one capture/parse/publish at a time.

    Waiters poll with their own request control, so a waiter's deadline or cancellation raises for that waiter
    only, and a leader's failure just passes the baton: the next caller becomes leader and finds the cursor
    current (no second parse) or retries. Cross-process safety is still the state token/CAS at publication.
    """
    with _flights_gate:
        entry = _flights.setdefault(key, [threading.Lock(), 0])
        entry[1] += 1
    try:
        while not entry[0].acquire(timeout=0.05):
            check_request()
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


class SessionStore:
    """v2 events preserve every occurrence; legacy tables remain untouched for recovery."""

    def __init__(self, db_path: Path):
        check_request()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            with publication():
                self._initialize_schema()
            stat = self.db_path.stat()
            self._authority_identity = (stat.st_dev, stat.st_ino)
        except BaseException:
            self._conn.close()
            raise

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
            CREATE TABLE IF NOT EXISTS pending_index_deletions (
                provider TEXT NOT NULL, session_id TEXT NOT NULL, chunk_id TEXT NOT NULL,
                PRIMARY KEY(provider,session_id,chunk_id)
            );
        """)
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

    def get_cursor(self, session_id, provider="codex"):
        record = self.cursor_record(session_id, provider)
        return (record["byte_offset"], record["path"]) if record else (0, None)

    def commit_events(self, chunks, *, session_id, provider, path, offset, digest, replace=False,
                      parser_version=None, expected_token=None, prepared_rows=None, receipt_digest=None):
        spool = prepared_rows if isinstance(prepared_rows, RowSpool) else RowSpool.from_chunks(chunks)
        try:
            self._commit_spool(spool, session_id=session_id, provider=provider, path=path, offset=offset, digest=digest,
                               replace=replace, parser_version=parser_version, expected_token=expected_token,
                               receipt_digest=receipt_digest)
        finally:
            if spool is not prepared_rows:
                spool.close()

    def _commit_spool(self, spool, *, session_id, provider, path, offset, digest, replace, parser_version,
                      expected_token, receipt_digest):
        check_request()
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            if expected_token is not None and self.state_token(session_id, provider) != expected_token:
                raise ValueError("stale session preparation")
            metrics = self.session_metrics(session_id, provider)
            known = self.chunk_ids(session_id, provider)
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
                if provider == "agy":
                    self._conn.execute("INSERT OR REPLACE INTO session_receipt_versions VALUES (?,?,?,?)",
                                       (provider, session_id, receipt_digest, synced_at))
                    self._conn.execute("INSERT OR REPLACE INTO session_agy_stats VALUES (?,?,?)",
                                       (session_id, json.dumps(dict(prior + tally), sort_keys=True), synced_at))
                self._conn.execute("INSERT OR REPLACE INTO session_metrics VALUES (?,?,?,?,?)",
                                   (provider, session_id, *metrics))
                check_request()
                self._conn.commit()

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
                    "SELECT chunk_id FROM session_events WHERE provider=? AND session_id=? AND chunk_id IN ("
                    + ",".join("?" * len(batch)) + ")", [provider, session_id, *batch]))
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
                selected = clauses + ["chunk_id IN (" + ",".join("?" for _ in batch) + ")"]
                rows.extend(tuple(row) for row in self._conn.execute(
                    "SELECT chunk_id,data FROM session_events WHERE " + " AND ".join(selected),
                    params + list(batch)).fetchall())
            return AuditEvidenceSnapshot(session_id, provider, tuple(rows), count)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        check_request()
        rows = tuple(tuple(row) for row in self._conn.execute(
            "SELECT chunk_id,data FROM session_events" + where + " ORDER BY provider,session_id,byte_start,sub_index", params).fetchall())
        return AuditEvidenceSnapshot(session_id, provider, rows, count)

    def capture_audit_events(self, session_id, provider, needs, *, link_calls=False):
        """Ordered slim events; text is loaded only for events some ``(scope, roles)`` need can inspect.

        ``link_calls`` also keeps text for tool calls/results (evidence of success-linked obligations).
        Rows are paged through a cursor and never held as raw JSON or pydantic models.
        """
        with self._read_boundary():
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
            count = self.chunk_count(session_id, provider)
            check_request()
            # AGY evidence classes (truncation, pending, legacy, denial) are read with one extra packed column.
            extra = ("json_extract(data,'" + "','".join("$.meta." + key for key in _AGY_AUDIT_META) + "')"
                     if provider == "agy" else "NULL")
            cursor = self._conn.execute(
                "SELECT chunk_id, json_extract(data,'$.kind'), json_extract(data,'$.meta.role'),"
                " json_extract(data,'$.meta.call_id'), json_type(data,'$.meta.success'),"
                " json_extract(data,'$.meta.success'), CASE WHEN " + wanted + " THEN json_extract(data,'$.text') END,"
                " " + extra + " FROM session_events WHERE session_id=? AND provider=?"
                " ORDER BY provider,session_id,byte_start,sub_index", term_params + [session_id, provider])
            events = []
            while True:
                check_request()
                page = cursor.fetchmany(512)
                if not page:
                    break
                for cid, kind, role, call_id, success_type, success, text, packed in page:
                    if success_type == "true":
                        success = True
                    elif success_type == "false":
                        success = False
                    elif success_type in (None, "null"):
                        success = None
                    meta = {"role": role, "call_id": call_id, "success": success}
                    if packed is not None:
                        meta.update((key, value) for key, value in zip(_AGY_AUDIT_META, json.loads(packed))
                                    if value is not None)
                    events.append(AuditEvent(cid, kind, text or "", meta))
            return AuditEventSnapshot(session_id, provider, tuple(events), count)

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
            "'$.meta.truncated_fields','$.meta.restored_fields','$.meta.fallback') FROM session_events"
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
        frames = []
        for order, row in enumerate(rows):
            chunk = Chunk.model_validate_json(row["data"])
            meta = chunk.meta or {}
            frames.append({
                "chunk_id": chunk.chunk_id,
                "order": order,
                "kind": chunk.kind,
                "text": chunk.text,
                "timestamp": chunk.timestamp,
                "tool_name": meta.get("tool_name"),
                "success": meta.get("success"),
                "error_text": meta.get("error") or meta.get("error_text"),
            })
        return frames

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
        return self._conn.execute("SELECT COUNT(*) FROM session_events" + where, params).fetchone()[0]

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
        coverage = coverage_from_capture(session_id, provider, record, metrics)
        if stats is not None:
            # No receipt capture here: the counts are exact, the first-receipt position is reported by sync/audit.
            coverage.update(legacy_unlinked_count=stats["legacy_unlinked_results"],
                            result_dependent_complete=not stats["legacy_unlinked_results"], agy_evidence=stats)
        return coverage


def coverage_from_capture(session_id, provider, record, metrics, observation=None, *, alias=False):
    """``alias``: ``observation`` is a verified copy or continuation of the indexed source, which lives at another path."""
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
            observation = observation or capture_source(Path(record["path"]), checkpoints=(record["byte_offset"],))
            if alias:
                result.update(indexed_path=record["path"], requested_path=str(observation.path))
            if (str(observation.path) != record["path"] and not alias) or not observation.validate_prefix():
                issues.append("source_changed")
            # Bytes appended after the observation are unread, never reviewed or lost.
            result["file_size"] = max(observation.length, observation.current_size())
            result["pending_bytes"] = max(0, result["file_size"] - record["byte_offset"])
            if result["pending_bytes"]:
                issues.append("unread_bytes")
            if observation.prefix_digest(record["byte_offset"]) != record["prefix_hash"]:
                issues.append("source_changed")
        except RequestStopped:
            raise
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
                   embed=True, derived=True, guard=nullcontext):
    """Serialize per (index, provider, session). Later callers find the cursor current and skip the parse.

    A parser-version bump therefore reparses a session once per process no matter how many hooks race on it;
    other sessions reparse lazily, each on its own first sync, never in a storm. Copies of one session id in
    several folders share a flight; across processes the state token still decides who publishes.
    """
    requested = Path(path).resolve()
    with session_flight(_flight_key(store, provider, requested, session_id)):
        seen, target, captured = [], requested, None
        for _ in range(_RECAPTURE_ATTEMPTS):
            handed, captured = captured, None
            try:
                info = _ingest_session(path=target, session_id=session_id, store=store, sparse=sparse, dense=dense,
                                       embedder=embedder, provider=provider, embed=embed, derived=derived,
                                       guard=guard, seen=seen, observation=handed)
            except _StaleObservation:
                continue  # another process published past our capture: observe the file again
            except _IndexedElsewhere as elsewhere:
                target, captured = elsewhere.path, elsewhere.observation  # a strict prefix: report the indexed source
                continue
            if target != requested:
                _mark_prefix_alias(info, requested)
            return info
        raise ValueError("source changed during preparation")


_RECAPTURE_ATTEMPTS = 3


class _StaleObservation(Exception):
    """The committed cursor lies beyond the captured prefix: our capture predates another process's publication."""


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


def _other_source(cursor, path, observation, session_id):
    """Another file claims the session id whose cursor indexes ``cursor["path"]``. Streaming hashes only; no writes.

    "copy": this file starts with the indexed prefix and the indexed file still exists, "moved": the same but that
    file is gone, "replaced": it is gone and this file does not continue it (reindex, as before). A strict prefix of
    the indexed file raises _IndexedElsewhere; two files that disagree raise AmbiguousSessionSource.
    """
    indexed, offset = Path(cursor["path"]), cursor["byte_offset"]
    continues = observation.length >= offset and observation.prefix_digest(offset) == cursor["prefix_hash"]
    present = indexed.is_file()
    if continues:
        return "copy" if present else "moved"
    if not present:
        return "replaced"
    other = capture_source(indexed, checkpoints=(offset, observation.length))
    if (other.length >= offset and other.prefix_digest(offset) == cursor["prefix_hash"]
            and observation.length < offset
            and other.prefix_digest(observation.length) == observation.prefix_digest(observation.length)):
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
    elif provider == "grokbot":
        from self_directing_mcp.grokbot.parse import parse_session_chunks, peek_session_id
    else:
        from self_directing_mcp.codex.parse import parse_session_chunks, peek_session_id
    return parse_session_chunks, peek_session_id


def _ingest_session(*, path, session_id, store, sparse, dense, embedder, provider, embed, derived, guard, seen,
                    observation=None):
    if embed and (dense is None or embedder is None):
        raise ValueError("embedding unavailable in local-only mode")
    parse_session_chunks, peek_session_id = _provider_parser(provider)
    path = Path(path).resolve()
    if observation is None:
        hint = ()
        if session_id:
            try:
                with guard(), store._read_boundary():
                    known = store.cursor_record(session_id.lower() if provider in ("codex", "agy") else session_id, provider)
                hint = (known["byte_offset"],) if known and type(known["byte_offset"]) is int else ()
            except (OSError, sqlite3.Error):
                hint = ()
        observation = capture_source(path, checkpoints=hint)
    with source_scope(observation):
        sid = session_id or peek_session_id(path)
        if not sid:
            from self_directing_mcp.codex.discover import session_id_from_filename
            sid = session_id_from_filename(path) if provider == "codex" else None
        if not sid:
            raise ValueError("cannot determine session id")
        sid = sid.lower() if provider in ("codex", "agy") else sid
        claimed = peek_session_id(path)
        if claimed is not None and (claimed.lower() if provider in ("codex", "agy") else claimed) != sid:
            raise ValueError("session identity differs from observed source")
        receipt_store, receipts, fallback_retries = None, {}, ()
        with guard(), store._read_boundary():
            cursor = store.cursor_record(sid, provider)
            expected_token = store.state_token(sid, provider)
            prior_calls = (store.prior_calls(sid, provider)
                           if provider == "omp" and cursor and observation.complete_end > cursor["byte_offset"] else ())
            if provider == "agy":
                from self_directing_mcp.agy_receipts import ReceiptStore
                fallback_retries = store.agy_fallback_retries(sid) if cursor else ()
                receipt_store = ReceiptStore(store.db_path.parent)
                receipt_capture = receipt_store.read_capture(sid)
                receipt_authority_before = receipt_store.authority_token(sid)
                if (receipt_authority_before[0] != "present"
                        or receipt_authority_before[2] != receipt_store.rows_digest(receipt_capture[0])):
                    raise ValueError("receipt authority changed during capture")
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
        if cursor and provider != "agy" and cursor["path"] != str(path) and type(cursor["byte_offset"]) is int:
            relation = _other_source(cursor, path, observation, sid)
        parser_version = current_parser_version(provider)
        start = cursor["byte_offset"] if cursor else 0
        reindexed, reindex_reason = False, None
        if cursor:
            try:
                changed = relation not in ("copy", "moved") and (
                    str(path) != cursor["path"] or observation.prefix_digest(start) != cursor["prefix_hash"])
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
        receipt_digest = None
        if receipt_store is not None:
            receipt_rows, receipts, receipt_digest, indexed_digest = receipt_store.prepare_bindings(receipt_capture, path)
            if cursor and (receipt_digest != indexed_digest or cursor.get("receipt_digest") != receipt_digest):
                start, reindexed, reindex_reason = 0, True, "hook_receipts_changed"
        chunks = RowSpool()
        try:
            if cursor and not reindexed and provider != "agy" and start >= observation.complete_end:
                offset, parsed_sid = start, sid
            elif provider == "agy":
                _, offset, parsed_sid = parse_session_chunks(path, session_id=sid, start_byte=start, receipts=receipts,
                                                             sink=chunks)
            elif provider == "omp" and start:
                # This lookup is request-owned and bound to the captured cursor generation.
                earliest = {}
                for before, call_id, name in prior_calls:
                    key = (call_id, name)
                    earliest[key] = min(earliest.get(key, before), before)
                _, offset, parsed_sid = parse_session_chunks(
                    path, session_id=sid, start_byte=start, sink=chunks,
                    prior_call=lambda call_id, name, before_byte: earliest.get((call_id, name), float("inf")) < before_byte)
            else:
                _, offset, parsed_sid = parse_session_chunks(path, session_id=sid, start_byte=start, sink=chunks)
            if parsed_sid != sid:
                raise ValueError("session id changed while parsing")
            digest = observation.prefix_digest(offset)
            # The whole-prefix rehash stays outside the publication lock; the lock only re-checks identity, size and a
            # bounded tail window (a rewrite elsewhere in that gap is caught by the next sync's cursor prefix_hash).
            if not observation.validate_prefix():
                raise ValueError("source changed during preparation")
        except BaseException:
            chunks.close()
            raise
    try:
        check_request()
        if not observation.validate_prefix():  # full rehash, off the shared lock
            raise ValueError("source changed during preparation")
        with guard():
            if not observation.validate_prefix(locked=True):
                raise ValueError("source changed during preparation")
            current = store.cursor_record(sid, provider)
            changed_token = store.state_token(sid, provider) != expected_token
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
            if not equivalent:
                if receipt_store is not None:
                    if receipt_store.authority_token(sid) != receipt_authority_before:
                        raise ValueError("receipt authority changed during preparation")
                    receipt_store.publish_bindings(sid, receipt_capture, receipt_rows)
                if chunks.count or reindexed or cursor is None or offset != cursor["byte_offset"] or relation == "moved":
                    store.commit_events((), session_id=sid, provider=provider, path=path, offset=offset,
                                        digest=digest, replace=reindexed, parser_version=parser_version,
                                        expected_token=expected_token, prepared_rows=chunks, receipt_digest=receipt_digest)
                if receipt_store is not None:
                    receipt_store.mark_indexed(sid, receipt_digest)
            else:
                chunks.count = 0
            with store._read_boundary():
                committed_token = store.state_token(sid, provider)
                committed_cursor = store.cursor_record(sid, provider)
                metrics = store.session_metrics(sid, provider)
                receipt_authority = receipt_store.authority_token(sid) if receipt_store is not None else None
                agy_stats = store.agy_evidence_stats(sid) if provider == "agy" else None
    finally:
        new_chunks = chunks.count
        chunks.close()
    alias = committed_cursor["path"] != str(path)  # an equal or shorter copy never moves the cursor
    coverage = coverage_from_capture(sid, provider, committed_cursor, metrics, observation, alias=alias)
    if agy_stats is not None:
        agy_coverage(coverage, agy_stats, receipts)
    info = {"session_id": sid, "provider": provider, "path": str(path), "byte_offset": offset,
            "new_chunks": new_chunks, "embedded": 0, "total_chunks": coverage["chunk_count"],
            "reindexed": reindexed, "parser_version": parser_version, "reindex_reason": reindex_reason,
            "dense_restored": 0, "embedding_pending": 0, "embedding_error": None,
            "sparse_updated": 0, "coverage": coverage,
            "_source_observation": observation, "_metadata_token": committed_token, "_receipt_token": receipt_authority}
    if alias:
        info["alias_of"] = committed_cursor["path"]
    if not derived:
        return info

    def finish():
        source_ok = observation.validate_prefix()  # full rehash, off the shared lock
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
        if pending:
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
    if embedder is None:
        return finish()
    with guard():
        dense.refresh()
        session_ids = sorted(store.chunk_ids(sid, provider))
        indexed = dense.present(session_ids)
        candidate_ids = [cid for cid in session_ids if cid not in indexed]
    cache_key = embedder.cache_key
    embedding_failed = False
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
            for start_idx in range(0, len(missing) if embed and not embedding_failed else 0, 32):
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
        except (RequestStopped, IndexBusy):
            raise
        except Exception as exc:
            info["embedding_error"] = type(exc).__name__
            embedding_failed = True
        restored = [(c.chunk_id, vectors[chunk_hashes[c.chunk_id]]) for c in candidates
                    if vectors[chunk_hashes[c.chunk_id]] is not None]
        with guard():
            check_request()
            live = store.live_ids(sid, provider, [cid for cid, _ in restored])
            restored = [pair for pair in restored if pair[0] in live]
            if restored:
                dense.refresh()
                dense.upsert_many(restored)
        info["dense_restored"] += len(restored)
        info["embedding_pending"] += len(candidates) - len(restored)
    return finish()
