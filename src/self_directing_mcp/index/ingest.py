from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from self_directing_mcp.schemas import Chunk
from self_directing_mcp.security.mask import mask_secrets


def current_parser_version(provider):
    if provider == "codex":
        from self_directing_mcp.codex.parse import PARSER_VERSION
        return PARSER_VERSION
    return 1


def prefix_digest(path: Path, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = length
        while remaining:
            block = stream.read(min(remaining, 1024 * 1024))
            if not block:
                raise ValueError("source truncated during read")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


class SessionStore:
    """v2 events preserve every occurrence; legacy tables remain untouched for recovery."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript("""
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
            CREATE TABLE IF NOT EXISTS pending_index_deletions (
                provider TEXT NOT NULL, session_id TEXT NOT NULL, chunk_id TEXT NOT NULL,
                PRIMARY KEY(provider,session_id,chunk_id)
            );
        """)
        self._conn.commit()

    def cursor_record(self, session_id, provider="codex"):
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
        return record

    def get_cursor(self, session_id, provider="codex"):
        record = self.cursor_record(session_id, provider)
        return (record["byte_offset"], record["path"]) if record else (0, None)

    def commit_events(self, chunks, *, session_id, provider, path, offset, digest, replace=False, parser_version=None):
        metrics = self.session_metrics(session_id, provider)
        known = self.chunk_ids(session_id, provider)
        additions = list(chunks) if replace else [c for c in chunks if c.chunk_id not in known]
        if replace:
            metrics = (0, 0, 0)
        metrics = (metrics[0] + len(additions),
                   metrics[1] + sum(bool(c.meta.get("parse_error")) for c in additions),
                   metrics[2] + sum(bool(c.meta.get("unsupported_event")) for c in additions))
        with self._conn:
            if replace:
                stale = known - {c.chunk_id for c in chunks}
                self._conn.executemany("INSERT OR IGNORE INTO pending_index_deletions VALUES (?,?,?)",
                                       [(provider, session_id, cid) for cid in stale])
                self._conn.execute("DELETE FROM session_events WHERE provider=? AND session_id=?",
                                   (provider, session_id))
            self._conn.executemany(
                "INSERT OR REPLACE INTO session_events VALUES (?,?,?,?,?,?)",
                [(c.chunk_id, c.provider, c.session_id, c.byte_start or 0,
                  c.meta.get("sub_index", 0), c.model_dump_json()) for c in chunks])
            synced_at = datetime.now(timezone.utc).isoformat()
            self._conn.execute("INSERT OR REPLACE INTO session_cursors VALUES (?,?,?,?,?,?)",
                               (provider, session_id, str(path.resolve()), offset, digest,
                                synced_at))
            self._conn.execute("INSERT OR REPLACE INTO session_parser_versions VALUES (?,?,?,?)",
                               (provider, session_id, current_parser_version(provider) if parser_version is None else parser_version, synced_at))
            self._conn.execute("INSERT OR REPLACE INTO session_metrics VALUES (?,?,?,?,?)",
                               (provider, session_id, *metrics))

    def pending_deletions(self, session_id, provider):
        return [row[0] for row in self._conn.execute(
            "SELECT chunk_id FROM pending_index_deletions WHERE provider=? AND session_id=?",
            (provider, session_id))]

    def finish_deletions(self, session_id, provider):
        with self._conn:
            self._conn.execute("DELETE FROM pending_index_deletions WHERE provider=? AND session_id=?",
                               (provider, session_id))

    def chunk_ids(self, session_id, provider):
        return {row[0] for row in self._conn.execute(
            "SELECT chunk_id FROM session_events WHERE provider=? AND session_id=?", (provider, session_id))}

    def session_metrics(self, session_id, provider):
        row = self._conn.execute(
            "SELECT chunk_count,parse_errors,unsupported_events FROM session_metrics WHERE provider=? AND session_id=?",
            (provider, session_id)).fetchone()
        if row is not None:
            return tuple(row)
        # One-time backfill for v0.3 indexes, without constructing Python Chunk objects.
        row = self._conn.execute("""
            SELECT COUNT(*), COALESCE(SUM(json_extract(data,'$.meta.parse_error') = 1),0),
                   COALESCE(SUM(json_extract(data,'$.meta.unsupported_event') = 1),0)
            FROM session_events WHERE provider=? AND session_id=?""", (provider, session_id)).fetchone()
        with self._conn:
            self._conn.execute("INSERT OR REPLACE INTO session_metrics VALUES (?,?,?,?,?)",
                               (provider, session_id, *tuple(row)))
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
        with self._conn:
            self._conn.executemany("INSERT OR REPLACE INTO embedding_cache VALUES (?,?,?)",
                                  [(cache_key, key, np.asarray(vec, dtype=np.float64).tobytes()) for key, vec in pairs])

    def legacy_count(self):
        exists = self._conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'").fetchone()
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] if exists else 0

    def session_status(self, session_id, provider="codex"):
        record = self.cursor_record(session_id, provider)
        count, errors, unsupported = self.session_metrics(session_id, provider)
        result = dict(record or {"provider": provider, "session_id": session_id, "path": None, "byte_offset": 0})
        result.update(chunk_count=count, parse_error_count=errors, unsupported_event_count=unsupported)
        issues = []
        if record is None:
            issues.append("not_synced")
        else:
            if record["parser_version"] != current_parser_version(provider):
                issues.append("parser_outdated")
            try:
                path = Path(record["path"])
                result["file_size"] = path.stat().st_size
                result["pending_bytes"] = max(0, result["file_size"] - record["byte_offset"])
                if result["pending_bytes"]:
                    issues.append("unread_bytes")
                if prefix_digest(path, record["byte_offset"]) != record["prefix_hash"]:
                    issues.append("source_changed")
            except (OSError, ValueError):
                issues.append("source_unavailable_or_truncated")
        if result["parse_error_count"]:
            issues.append("parse_errors")
        if result["unsupported_event_count"]:
            issues.append("unsupported_events")
        result["issues"] = issues
        result["complete"] = not issues
        return result


def ingest_session(*, path, session_id, store, sparse, dense, embedder, provider="codex", embed=True):
    if provider == "grokbot":
        from self_directing_mcp.grokbot.parse import parse_session_chunks, peek_session_id
    else:
        from self_directing_mcp.codex.parse import parse_session_chunks, peek_session_id
    path = Path(path)
    sid = session_id or peek_session_id(path)
    if not sid:
        from self_directing_mcp.codex.discover import session_id_from_filename
        sid = session_id_from_filename(path) if provider == "codex" else None
    if not sid:
        raise ValueError("cannot determine session id")
    sid = sid.lower() if provider == "codex" else sid
    cursor = store.cursor_record(sid, provider)
    parser_version = current_parser_version(provider)
    start, reindexed = (cursor["byte_offset"] if cursor else 0), False
    reindex_reason = None
    if cursor:
        try:
            changed = str(path.resolve()) != cursor["path"] or prefix_digest(path, start) != cursor["prefix_hash"]
        except (OSError, ValueError):
            changed = True
        if changed:
            start, reindexed = 0, True
            reindex_reason = "source_changed"
        elif cursor["parser_version"] != parser_version:
            start, reindexed = 0, True
            reindex_reason = "parser_updated"
    # Detect a rewrite concurrent with parsing before publishing any new cursor.
    initial_stat = path.stat()
    chunks, offset, parsed_sid = parse_session_chunks(path, session_id=sid, start_byte=start)
    if parsed_sid != sid:
        raise ValueError("session id changed while parsing")
    digest = prefix_digest(path, offset)
    final_stat = path.stat()
    if final_stat.st_size < offset or (final_stat.st_mtime_ns != initial_stat.st_mtime_ns and final_stat.st_size <= initial_stat.st_size):
        raise ValueError("source changed during sync; retry")
    if chunks or reindexed or cursor is None or offset != cursor["byte_offset"]:
        store.commit_events(chunks, session_id=sid, provider=provider, path=path, offset=offset,
                            digest=digest, replace=reindexed, parser_version=parser_version)
    stale_ids = store.pending_deletions(sid, provider)
    if stale_ids:
        # Replayable cleanup also repairs a crash after metadata was committed.
        # Preserve vectors and anchors for unchanged event IDs.
        stale_ids = list(set(stale_ids) - store.chunk_ids(sid, provider))
        sparse.delete_ids(stale_ids)
        dense.delete_ids(stale_ids)
        store.finish_deletions(sid, provider)
    # Immutable event ids permit delta updates. Missing entries also repair an
    # interrupted metadata->FTS write without reindexing existing documents.
    missing_ids = store.chunk_ids(sid, provider) - sparse.ids()
    to_index = [store.get_chunk(cid) for cid in missing_ids]
    if to_index:
        sparse.upsert_many(to_index)
    coverage = store.session_status(sid, provider)
    cache_key = embedder.cache_key
    existing_ids = dense.ids()
    candidates = [store.get_chunk(cid) for cid in sorted(store.chunk_ids(sid, provider) - existing_ids)]
    text_by_hash = {}
    chunk_hashes = {}
    for chunk in candidates:
        sanitized = mask_secrets(chunk.text)
        key = hashlib.sha256(sanitized.encode("utf-8")).hexdigest()
        text_by_hash[key] = sanitized
        chunk_hashes[chunk.chunk_id] = key
    vectors = {key: store.cached_vector(cache_key, key) for key in text_by_hash}
    missing = [key for key, value in vectors.items() if value is None]
    embedded, embedding_error = 0, None
    try:
        for start_idx in range(0, len(missing) if embed else 0, 32):
            batch = missing[start_idx:start_idx + 32]
            values = embedder.embed_documents([text_by_hash[key] for key in batch])
            if len(values) != len(batch):
                raise ValueError("embedding response count mismatch")
            if any(np.asarray(v).shape != (embedder.dim,) or not np.isfinite(v).all() for v in values):
                raise ValueError("invalid embedding shape or values")
            store.cache_vectors(cache_key, zip(batch, values))
            vectors.update(zip(batch, values))
            embedded += len(batch)
    except Exception as exc:
        # Local evidence is committed and remains usable for deterministic audit.
        # Do not echo remote response bodies or tokens in an error.
        embedding_error = type(exc).__name__
    restored = [(c.chunk_id, vectors[chunk_hashes[c.chunk_id]]) for c in candidates
                if vectors[chunk_hashes[c.chunk_id]] is not None]
    if restored:
        dense.upsert_many(restored)
    return {"session_id": sid, "provider": provider, "path": str(path.resolve()),
            "byte_offset": offset, "new_chunks": len(chunks), "embedded": embedded,
            "total_chunks": coverage["chunk_count"], "reindexed": reindexed,
            "parser_version": parser_version, "reindex_reason": reindex_reason,
            "dense_restored": len(restored), "embedding_pending": len(candidates) - len(restored),
            "embedding_error": embedding_error, "sparse_updated": len(to_index), "coverage": coverage}
