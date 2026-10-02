from __future__ import annotations

import hashlib
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from self_directing_mcp.schemas import Chunk
from self_directing_mcp.security.mask import mask_secrets
from self_directing_mcp.request_control import IndexBusy, RequestStopped, check_request, publication, request_operation
from self_directing_mcp.session_events import capture_source, observed_source, source_scope


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
        chunks = list(chunks)
        if any(chunk.provider != provider or chunk.session_id != session_id for chunk in chunks):
            raise ValueError("event scope differs from session publication")
        rows = prepared_rows if prepared_rows is not None else [
            (c.chunk_id, c.provider, c.session_id, c.byte_start or 0,
             c.meta.get("sub_index", 0), c.model_dump_json()) for c in chunks]
        check_request()
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            if expected_token is not None and self.state_token(session_id, provider) != expected_token:
                raise ValueError("stale session preparation")
            metrics = self.session_metrics(session_id, provider)
            known = self.chunk_ids(session_id, provider)
            additions = chunks if replace else [c for c in chunks if c.chunk_id not in known]
            base = (0, 0, 0) if replace else metrics
            metrics = (base[0] + len(additions),
                       base[1] + sum(bool(c.meta.get("parse_error")) for c in additions),
                       base[2] + sum(bool(c.meta.get("unsupported_event")) for c in additions))
            with publication():
                if replace:
                    stale = known - {c.chunk_id for c in chunks}
                    for cid in stale:
                        check_request()
                        self._conn.execute("INSERT OR IGNORE INTO pending_index_deletions VALUES (?,?,?)",
                                           (provider, session_id, cid))
                    self._conn.execute("DELETE FROM session_events WHERE provider=? AND session_id=?",
                                       (provider, session_id))
                for start in range(0, len(rows), 256):
                    check_request()
                    self._conn.executemany("INSERT OR REPLACE INTO session_events VALUES (?,?,?,?,?,?)", rows[start:start + 256])
                synced_at = datetime.now(timezone.utc).isoformat()
                self._conn.execute("INSERT OR REPLACE INTO session_cursors VALUES (?,?,?,?,?,?)",
                                   (provider, session_id, str(path.resolve()), offset, digest, synced_at))
                self._conn.execute("INSERT OR REPLACE INTO session_parser_versions VALUES (?,?,?,?)",
                                   (provider, session_id, current_parser_version(provider) if parser_version is None else parser_version, synced_at))
                if provider == "agy":
                    self._conn.execute("INSERT OR REPLACE INTO session_receipt_versions VALUES (?,?,?,?)",
                                       (provider, session_id, receipt_digest, synced_at))
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
        return coverage_from_capture(session_id, provider, record, metrics)


def coverage_from_capture(session_id, provider, record, metrics, observation=None):
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
            observation = observation or capture_source(Path(record["path"]))
            if str(observation.path) != record["path"] or not observation.validate():
                issues.append("source_changed")
            result["file_size"] = len(observation.data)
            result["pending_bytes"] = max(0, len(observation.data) - record["byte_offset"])
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


@request_operation
def ingest_session(*, path, session_id, store, sparse, dense, embedder, provider="codex",
                   embed=True, derived=True, guard=nullcontext):
    if embed and (dense is None or embedder is None):
        raise ValueError("embedding unavailable in local-only mode")
    if provider == "agy":
        from self_directing_mcp.agy import parse_session_chunks, peek_session_id
    elif provider == "omp":
        from self_directing_mcp.omp import parse_session_chunks, peek_session_id
    elif provider == "grokbot":
        from self_directing_mcp.grokbot.parse import parse_session_chunks, peek_session_id
    else:
        from self_directing_mcp.codex.parse import parse_session_chunks, peek_session_id
    path = Path(path).resolve()
    observation = capture_source(path)
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
        receipt_store = None
        with guard(), store._read_boundary():
            cursor = store.cursor_record(sid, provider)
            expected_token = store.state_token(sid, provider)
            prior_calls = (store.prior_calls(sid, provider)
                           if provider == "omp" and cursor and len(observation.data) > cursor["byte_offset"] else ())
            if provider == "agy":
                from self_directing_mcp.agy_receipts import ReceiptStore
                receipt_store = ReceiptStore(store.db_path.parent)
                receipt_capture = receipt_store.read_capture(sid)
                receipt_authority_before = receipt_store.authority_token(sid)
                if (receipt_authority_before[0] != "present"
                        or receipt_authority_before[2] != receipt_store.rows_digest(receipt_capture[0])):
                    raise ValueError("receipt authority changed during capture")
        parser_version = current_parser_version(provider)
        start = cursor["byte_offset"] if cursor else 0
        reindexed, reindex_reason = False, None
        if cursor:
            try:
                changed = str(path) != cursor["path"] or observation.prefix_digest(start) != cursor["prefix_hash"]
            except ValueError:
                changed = True
            if changed:
                start, reindexed, reindex_reason = 0, True, "source_changed"
            elif cursor["parser_version"] != parser_version:
                start, reindexed, reindex_reason = 0, True, "parser_updated"
        receipt_digest = None
        if receipt_store is not None:
            receipt_rows, receipts, receipt_digest, indexed_digest = receipt_store.prepare_bindings(receipt_capture, path)
            if cursor and (receipt_digest != indexed_digest or cursor.get("receipt_digest") != receipt_digest):
                start, reindexed, reindex_reason = 0, True, "hook_receipts_changed"
        if provider == "agy":
            chunks, offset, parsed_sid = parse_session_chunks(path, session_id=sid, start_byte=start, receipts=receipts)
        elif provider == "omp" and start:
            # This lookup is request-owned and bound to the captured cursor generation.
            earliest = {}
            for before, call_id, name in prior_calls:
                key = (call_id, name)
                earliest[key] = min(earliest.get(key, before), before)
            chunks, offset, parsed_sid = parse_session_chunks(
                path, session_id=sid, start_byte=start,
                prior_call=lambda call_id, name, before_byte: earliest.get((call_id, name), float("inf")) < before_byte)
        else:
            chunks, offset, parsed_sid = parse_session_chunks(path, session_id=sid, start_byte=start)
        if parsed_sid != sid:
            raise ValueError("session id changed while parsing")
        digest = observation.prefix_digest(offset)
        prepared_rows = []
        for chunk in chunks:
            check_request()
            prepared_rows.append((chunk.chunk_id, chunk.provider, chunk.session_id, chunk.byte_start or 0,
                                  chunk.meta.get("sub_index", 0), chunk.model_dump_json()))
    check_request()
    with guard():
        if not observation.validate():
            raise ValueError("source changed during preparation")
        current = store.cursor_record(sid, provider)
        changed_token = store.state_token(sid, provider) != expected_token
        equivalent = bool(changed_token and current and current["path"] == str(path)
                          and current["byte_offset"] == offset and current["prefix_hash"] == digest
                          and current["parser_version"] == parser_version)
        if equivalent:
            live_rows = dict(store.capture_evidence(sid, provider, history=False,
                                                   anchors=(row[0] for row in prepared_rows)).rows)
            equivalent = all(live_rows.get(row[0]) == row[-1] for row in prepared_rows)
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
            if chunks or reindexed or cursor is None or offset != cursor["byte_offset"]:
                store.commit_events(chunks, session_id=sid, provider=provider, path=path, offset=offset,
                                    digest=digest, replace=reindexed, parser_version=parser_version,
                                    expected_token=expected_token, prepared_rows=prepared_rows, receipt_digest=receipt_digest)
            if receipt_store is not None:
                receipt_store.mark_indexed(sid, receipt_digest)
        else:
            chunks = []
        with store._read_boundary():
            committed_token = store.state_token(sid, provider)
            committed_cursor = store.cursor_record(sid, provider)
            metrics = store.session_metrics(sid, provider)
            receipt_authority = receipt_store.authority_token(sid) if receipt_store is not None else None
    coverage = coverage_from_capture(sid, provider, committed_cursor, metrics, observation)
    info = {"session_id": sid, "provider": provider, "path": str(path), "byte_offset": offset,
            "new_chunks": len(chunks), "embedded": 0, "total_chunks": coverage["chunk_count"],
            "reindexed": reindexed, "parser_version": parser_version, "reindex_reason": reindex_reason,
            "dense_restored": 0, "embedding_pending": 0, "embedding_error": None,
            "sparse_updated": 0, "coverage": coverage,
            "_source_observation": observation, "_metadata_token": committed_token, "_receipt_token": receipt_authority}
    if not derived:
        return info

    def finish():
        with guard():
            issues = []
            if store.state_token(sid, provider) != committed_token:
                issues.append("metadata_changed_during_derived_work")
            if not observation.validate():
                issues.append("source_changed_during_derived_work")
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
        missing_ids = store.chunk_ids(sid, provider) - sparse.ids()
        sparse_snapshot = store.capture_evidence(sid, provider, history=False, anchors=missing_ids)
    to_index = sparse_snapshot.list_chunks(sid, provider)
    with guard():
        live = store.chunk_ids(sid, provider)
        to_index = [chunk for chunk in to_index if chunk.chunk_id in live]
        if to_index:
            sparse.upsert_many(to_index)
        info["sparse_updated"] = len(to_index)
        if embedder is None:
            return finish()
        dense.refresh()
        candidates_snapshot = store.capture_evidence(sid, provider, history=False,
                                                     anchors=store.chunk_ids(sid, provider) - dense.ids())
    candidates = candidates_snapshot.list_chunks(sid, provider)
    text_by_hash, chunk_hashes = {}, {}
    for chunk in candidates:
        check_request()
        sanitized = mask_secrets(chunk.text)
        key = hashlib.sha256(sanitized.encode("utf-8")).hexdigest()
        text_by_hash[key], chunk_hashes[chunk.chunk_id] = sanitized, key
    cache_key = embedder.cache_key
    with guard():
        vectors = {key: store.cached_vector(cache_key, key) for key in text_by_hash}
    missing = [key for key, value in vectors.items() if value is None]
    try:
        for start_idx in range(0, len(missing) if embed else 0, 32):
            check_request()
            batch = missing[start_idx:start_idx + 32]
            values = embedder.embed_documents([text_by_hash[key] for key in batch])
            check_request()
            if len(values) != len(batch):
                raise ValueError("embedding response count mismatch")
            if any(np.asarray(v).shape != (embedder.dim,) or not np.isfinite(v).all() for v in values):
                raise ValueError("invalid embedding shape or values")
            with guard():
                store.cache_vectors(cache_key, zip(batch, values))
            vectors.update(zip(batch, values))
            info["embedded"] += len(batch)
    except (RequestStopped, IndexBusy):
        raise
    except Exception as exc:
        info["embedding_error"] = type(exc).__name__
    restored = [(c.chunk_id, vectors[chunk_hashes[c.chunk_id]]) for c in candidates
                if vectors[chunk_hashes[c.chunk_id]] is not None]
    with guard():
        check_request()
        live = store.chunk_ids(sid, provider)
        restored = [pair for pair in restored if pair[0] in live]
        if restored:
            dense.refresh()
            dense.upsert_many(restored)
    info["dense_restored"] = len(restored)
    info["embedding_pending"] = len(candidates) - len(restored)
    return finish()
