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

from self_directing_mcp import journal as _journal
from self_directing_mcp._runtime import np

from self_directing_mcp.compressed_source import CompressedSourceError
from self_directing_mcp.index.deferred import DerivedBuffer
from self_directing_mcp.schemas import AuditEvent, Chunk
from self_directing_mcp.security.mask import mask_secrets
from self_directing_mcp.request_control import IndexBusy, RequestStopped, check_request, publication, request_operation
from self_directing_mcp.session_events import (
    SourceChanged, capture_source, capture_source_fast, observed_source, source_scope,
)
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
_AUDIT_META = {"agy": _AGY_AUDIT_META, "claude": ("truncated_result_for", "call_id", "permission_decision")}


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
        spool = prepared_rows if isinstance(prepared_rows, RowSpool) else RowSpool.from_chunks(chunks)
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
        # Provider evidence classes (AGY truncation, pending, legacy, denial; Claude cut results) are read
        # with one extra packed column.
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
                events.append(AuditEvent(cid, kind, text or "", meta, (byte_start, sub_index)))
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

    def save_audit_states(self, session_id, provider, needs_digest, rows, snapshot):
        """Write monitor states and the evaluated cursor in one transaction; returns ``"saved"`` or why not.

        ``snapshot`` is what the evaluation read: ``generation`` (``None`` when no valid commit chain existed),
        ``offset``, ``prefix_hash``, ``synced_at``, ``path`` and ``count``. The write is refused unless the index
        still contains exactly that point: same chain and log row, or (when the chain is new) the same cursor. A
        state is never written over a newer one of the same generation.
        """
        now = datetime.now(timezone.utc).isoformat()
        if self.defer_derived and snapshot["generation"] is not None:
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
        if not states and not stamps:
            return 0
        written, touched = 0, set()
        self._conn.execute("SAVEPOINT derived_flush")
        try:
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


def capture_for_cursor(provider, path, cursor, checkpoints, max_bytes, *, publish=False):
    """One full capture collecting what is needed to compare with ``cursor`` (and to publish, with ``publish``)."""
    kind, blockable = _cursor_kind(cursor), _blockable(provider, path)
    blocks = kind == "blocks" or (publish and blockable)
    sha = kind == "sha" or (publish and not blockable) or not blocks
    return capture_source(path, checkpoints=checkpoints, max_bytes=max_bytes, sha=sha, blocks=blocks)


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
                   verify="full", policy=None):
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
    """
    policy = policy or IntegrityPolicy()
    requested = Path(path).resolve()
    with session_flight(_flight_key(store, provider, requested, session_id)):
        seen, target, captured = [], requested, None
        for _ in range(_RECAPTURE_ATTEMPTS):
            handed, captured = captured, None
            try:
                info = _ingest_session(path=target, session_id=session_id, store=store, sparse=sparse, dense=dense,
                                       embedder=embedder, provider=provider, embed=embed, derived=derived,
                                       guard=guard, seen=seen, observation=handed, verify=verify, policy=policy,
                                       retain_dense_deletions=retain_dense_deletions)
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
            return info
        raise SourceChanged("source changed during preparation")


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


def _capture_observation(path, provider, known, record, hint, verify, policy, max_bytes):
    """The request's capture: a fast check when allowed and possible, otherwise the one full scan of the prefix."""
    mode, reason = _plan(provider, path, known, record, verify, policy)
    lag = None
    if reason == "interval_elapsed":
        lag = (policy.clock() - _parse_time(record.full_verified_at)).total_seconds() - policy.full_verify_interval_sec
    if mode == "fast":
        try:
            observation = capture_source_fast(path, record, recent=policy.recent_blocks, samples=policy.sample_blocks,
                                              rng=policy.new_rng())
        except FastFallback as fallback:
            reason = fallback.reason  # the cheap signals (or a block) say this must be settled by a full scan
        else:
            observation.scanned_at = policy.clock().isoformat()
            return observation
    observation = capture_for_cursor(provider, path, known, hint, max_bytes, publish=True)
    observation.full_reason, observation.lag_sec = reason, lag
    observation.scanned_at = policy.clock().isoformat()
    return observation


def _ingest_session(*, path, session_id, store, sparse, dense, embedder, provider, embed, derived, guard, seen,
                    observation=None, retain_dense_deletions=False, verify="full", policy=None):
    if embed and (dense is None or embedder is None):
        raise ValueError("embedding unavailable in local-only mode")
    parse_session_chunks, peek_session_id = _provider_parser(provider)
    path = Path(path).resolve()
    policy = policy or IntegrityPolicy()
    planned = None  # the cursor a fast capture was planned from; it must still be the current one
    if observation is None:
        hint, known, record, sid_hint = (), None, None, session_id
        if not sid_hint:
            try:
                sid_hint = peek_session_id(path)
            except (OSError, ValueError):
                sid_hint = None
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
            except (OSError, sqlite3.Error):
                hint, known, record = (), None, None
        observation = _capture_observation(path, provider, known, record, hint, verify, policy,
                                           store.max_decompressed_bytes)
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
        chunks = RowSpool()
        uses = set()
        try:
            if cursor and not reindexed and provider != "agy" and start >= observation.complete_end:
                offset, parsed_sid = start, sid
            elif provider == "agy":
                _, offset, parsed_sid = parse_session_chunks(path, session_id=sid, start_byte=start, receipts=receipts,
                                                             sink=chunks, uses=uses)
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
            blockable = _blockable(provider, path)
            digest = observation.cursor_hash(offset, "blocks" if blockable else "sha")
            draft = _draft(observation, offset, record) if blockable else None
            receipt_deps = None
            if receipt_store is not None:
                from self_directing_mcp.agy_receipts import dependencies
                # An append extends what the stored events depend on; without stored dependencies (an index from
                # before they were kept) there is nothing to extend and the next receipt change reparses.
                prior = None if reindexed or not cursor else cursor_deps
                if reindexed or not cursor or prior is not None:
                    receipt_deps = json.dumps(dependencies(receipt_rows, uses, prior), sort_keys=True)
        except BaseException:
            chunks.close()
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
                    receipt_store.publish_bindings(sid, receipt_capture, receipt_rows, receipt_ledger)
                # A cursor without a block table (written before this version, by an older process, or after a
                # change of source form) is republished with one, even when nothing else changed.
                own = draft is not None and cursor is not None and cursor["path"] == str(path) and relation is None
                bound = own and record is not None and record.bound_to(cursor)
                needs_table = own and not bound
                if (chunks.count or reindexed or cursor is None or offset != cursor["byte_offset"]
                        or relation == "moved" or needs_table or receipts_dirty):
                    # receipts_dirty: the events stand, but they now belong to these receipts (and their dependencies).
                    store.commit_events((), session_id=sid, provider=provider, path=path, offset=offset,
                                        digest=digest, replace=reindexed, parser_version=parser_version,
                                        expected_token=expected_token, prepared_rows=chunks, receipt_digest=receipt_digest,
                                        blocks=draft, receipt_deps=receipt_deps)
                elif bound and observation.mode == "full":
                    # Nothing to publish, but every block was just compared: stamp the stored table.
                    stamp = observation.stamp
                    store.mark_full_verified(sid, provider, record, at=observation.scanned_at,
                                             through=record.table.covered, dev=stamp[0], ino=stamp[1],
                                             size=stamp[2], mtime_ns=stamp[3])
                if receipt_store is not None:
                    receipt_store.mark_indexed(sid, receipt_digest)
            else:
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
    alias = committed_cursor["path"] != str(path)  # an equal or shorter copy never moves the cursor
    coverage = coverage_from_capture(sid, provider, committed_cursor, metrics, observation, alias=alias,
                                     max_bytes=store.max_decompressed_bytes)
    if agy_stats is not None:
        agy_coverage(coverage, agy_stats, receipts)
    info = {"session_id": sid, "provider": provider, "path": str(path), "byte_offset": offset,
            "new_chunks": new_chunks, "embedded": 0, "total_chunks": coverage["chunk_count"],
            "reindexed": reindexed, "parser_version": parser_version, "reindex_reason": reindex_reason,
            "dense_restored": 0, "embedding_pending": 0, "embedding_error": None,
            "sparse_updated": 0, "coverage": coverage,
            "integrity": integrity_report(observation, committed_cursor, committed_record),
            "_source_observation": observation, "_metadata_token": committed_token, "_receipt_token": receipt_authority}
    if reindexed:
        # What this request's verification found: the index was rebuilt from the source as it is now. For
        # "source_changed" that is a mismatch of the stored digests, never a runtime failure.
        info["integrity"]["reindex_reason"] = reindex_reason
    if alias:
        info["alias_of"] = committed_cursor["path"]
    if not derived:
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
