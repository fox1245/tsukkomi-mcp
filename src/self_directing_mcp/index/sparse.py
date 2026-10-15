from __future__ import annotations

from contextlib import closing, contextmanager
import functools
import re
import sqlite3
from pathlib import Path

from self_directing_mcp.request_control import check_request, publication, request_operation
from self_directing_mcp.security.mask import mask_secrets

LEGACY_FILE = "sparse.sqlite"          # the first format: FTS5 table that stores a copy of every text
COMPACT_FILE = "sparse-v2.sqlite"      # the compact format: FTS5 contentless table, nothing but the inverted index
COMPACT_FORMAT = "compact-1"           # the value of sparse_meta.format in a compact file
COMPACT_MIN_SQLITE = (3, 43, 0)        # FTS5 contentless_delete


def _fts_query(raw: str) -> str:
    toks = re.findall(r"[A-Za-z0-9_\-]+", raw)
    cleaned: list[str] = []
    for t in toks:
        check_request()
        if len(t) < 2:
            continue
        cleaned.append(f'"{t}"')
    if not cleaned:
        return '""'
    return " OR ".join(cleaned)


class SparseIndex:
    """SQLite FTS5 BM25 sparse index over session chunks: the first ("legacy") format, ``sparse.sqlite``.

    The FTS5 table stores a copy of every (masked) text next to its inverted index. ``CompactSparseIndex`` returns the
    same results from a file about 70 % smaller; see its docstring and README "Sparse index formats".
    """

    format = "legacy"

    def __init__(self, db_path: Path) -> None:
        check_request()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            self._init_schema()
        except Exception:
            self._conn.close()
            raise

    def _init_schema(self) -> None:
        check_request()
        with self._conn:
            self._conn.execute("BEGIN")
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    session_id UNINDEXED,
                    kind UNINDEXED,
                    text,
                    tokenize = 'porter unicode61'
                )
                """
            )
            # Preserve the FTS table; migrate only a compact id -> rowid map.
            check_request()
            exists = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk_rows'").fetchone()
            if not exists:
                self._conn.execute("CREATE TABLE chunk_rows(chunk_id TEXT PRIMARY KEY, doc_rowid INTEGER UNIQUE NOT NULL)")
                self._conn.execute("INSERT INTO chunk_rows SELECT chunk_id,rowid FROM chunks_fts")
            with publication():
                self._conn.commit()


    def clear(self) -> None:
        check_request()
        with self._conn:
            self._conn.execute("DELETE FROM chunks_fts")
            self._conn.execute("DELETE FROM chunk_rows")
            with publication():
                self._conn.commit()

    @request_operation
    def ids(self) -> set[str]:
        out = set()
        with closing(self._conn.execute("SELECT chunk_id FROM chunk_rows")) as rows:
            for row in rows:
                check_request()
                out.add(row[0])
        return out

    @request_operation
    def present(self, ids) -> set[str]:
        """Subset of ``ids`` already indexed; queries only the given ids, in bounded batches."""
        ids = list(ids)
        out = set()
        for start in range(0, len(ids), 500):
            check_request()
            batch = ids[start:start + 500]
            out.update(row[0] for row in self._conn.execute(
                "SELECT chunk_id FROM chunk_rows WHERE chunk_id IN (" + ",".join("?" * len(batch)) + ")", batch))
        return out

    def upsert(self, chunk_id: str, session_id: str, kind: str, text: str) -> None:
        check_request()
        with self._conn:
            row = self._conn.execute("SELECT doc_rowid FROM chunk_rows WHERE chunk_id=?", (chunk_id,)).fetchone()
            if row:
                self._conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (row[0],))
            inserted = self._conn.execute(
                "INSERT INTO chunks_fts(chunk_id,session_id,kind,text) VALUES (?,?,?,?)",
                (chunk_id, session_id, kind, mask_secrets(text)))
            self._conn.execute("INSERT OR REPLACE INTO chunk_rows VALUES (?,?)", (chunk_id, inserted.lastrowid))
            with publication():
                self._conn.commit()

    def delete_ids(self, ids: list[str]) -> None:
        check_request()
        with self._conn:
            for cid in ids:
                check_request()
                row = self._conn.execute("SELECT doc_rowid FROM chunk_rows WHERE chunk_id=?", (cid,)).fetchone()
                if row:
                    self._conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (row[0],))
                    self._conn.execute("DELETE FROM chunk_rows WHERE chunk_id=?", (cid,))
            with publication():
                self._conn.commit()

    def upsert_many(self, chunks) -> None:
        check_request()
        with self._conn:
            for chunk in chunks:
                check_request()
                row = self._conn.execute("SELECT doc_rowid FROM chunk_rows WHERE chunk_id=?", (chunk.chunk_id,)).fetchone()
                if row:
                    self._conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (row[0],))
                inserted = self._conn.execute(
                    "INSERT INTO chunks_fts(chunk_id,session_id,kind,text) VALUES (?,?,?,?)",
                    (chunk.chunk_id, chunk.session_id, chunk.kind, mask_secrets(chunk.text)))
                self._conn.execute("INSERT OR REPLACE INTO chunk_rows VALUES (?,?)",
                                   (chunk.chunk_id, inserted.lastrowid))
            with publication():
                self._conn.commit()

    @request_operation
    def search(
        self,
        query: str,
        top_k: int = 20,
        *,
        session_id: str | None = None,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float, int]]:
        q = _fts_query(query)
        if session_id:
            cur = self._conn.execute(
                """
                SELECT chunk_id, bm25(chunks_fts) AS score
                FROM chunks_fts
                WHERE chunks_fts MATCH ? AND session_id = ?
                ORDER BY score
                LIMIT ?
                """,
                (q, session_id, -1 if allowed_ids is not None else top_k),
            )
        else:
            cur = self._conn.execute(
                """
                SELECT chunk_id, bm25(chunks_fts) AS score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (q, -1 if allowed_ids is not None else top_k),
            )
        out: list[tuple[str, float, int]] = []
        with closing(cur):
            for row in cur:
                check_request()
                if allowed_ids is not None and row["chunk_id"] not in allowed_ids:
                    continue
                if len(out) >= top_k:
                    break
                out.append((row["chunk_id"], float(-row["score"]), len(out) + 1))
        return out

    @request_operation
    def count(self, session_id: str | None = None) -> int:
        if session_id:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM chunks_fts WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        else:
            # chunk_rows mirrors every FTS document in the same transaction; counting the FTS5 table
            # itself scans the whole index while callers may hold the shared index lock.
            row = self._conn.execute("SELECT COUNT(*) AS c FROM chunk_rows").fetchone()
        return int(row["c"]) if row else 0


class SparseFormatError(RuntimeError):
    """The sparse index cannot be opened in the requested format. The message names the repair."""


# ------------------------------------------------------------------------------------------------------------------
# FTS5 statistics
#
# bm25() needs the number of documents and the total number of tokens ("averages": row 1 of ``<fts>_data``, the varints
# nTotalRow and one total per column). A contentless_delete table does not take a deleted row out of them (FTS5 does not
# know its tokens), so after a delete the scores drift away from an index that never held the row. The size of every
# deleted document is still in ``<fts>_docsize``, so the record is corrected in the transaction of the delete, before its
# first FTS5 write (FTS5 keeps its own copy of the totals once it has written and saves it at commit). The record format
# is part of the SQLite file format, which is stable; ``compact_support`` proves the whole mechanism on the running SQLite
# before a compact file is opened, and a self-check (``CompactSparseIndex.verify_statistics``) compares the record with
# the document sizes.
# ------------------------------------------------------------------------------------------------------------------
def _read_varints(blob: bytes) -> list[int]:
    out, i, n = [], 0, len(blob)
    while i < n:
        value = 0
        for position in range(9):
            if i >= n:
                raise SparseFormatError("truncated varint in an FTS5 statistics record")
            byte = blob[i]
            i += 1
            if position == 8:               # the ninth byte of a SQLite varint carries all eight bits
                value = (value << 8) | byte
                break
            value = (value << 7) | (byte & 0x7F)
            if not byte & 0x80:
                break
        out.append(value)
    return out


def _varint(value: int) -> bytes:
    if value < 0 or value >= 1 << 56:
        raise SparseFormatError("FTS5 statistic out of range")
    groups = [value & 0x7F]
    value >>= 7
    while value:
        groups.append(value & 0x7F)
        value >>= 7
    groups.reverse()
    return bytes([g | 0x80 for g in groups[:-1]] + [groups[-1]])


def _retire_statistics(conn: sqlite3.Connection, fts: str, rowids: list[int]) -> None:
    """Take the documents ``rowids`` out of the FTS5 statistics of the contentless table ``fts``. Call it inside the
    write transaction, before the first write to the table and before the documents are deleted."""
    rows = tokens = 0
    for start in range(0, len(rowids), 500):
        batch = rowids[start:start + 500]
        for _id, size in conn.execute(f"SELECT id, sz FROM {fts}_docsize WHERE id IN (" + ",".join("?" * len(batch)) + ")", batch):
            sizes = _read_varints(size)
            if len(sizes) != 1:
                raise SparseFormatError("unexpected FTS5 document size record")
            rows += 1
            tokens += sizes[0]
    if not rows:
        return
    row = conn.execute(f"SELECT block FROM {fts}_data WHERE id=1").fetchone()
    totals = _read_varints(row[0]) if row else []
    if len(totals) != 2 or totals[0] < rows or totals[1] < tokens:
        raise SparseFormatError(f"unexpected FTS5 statistics record {totals} while retiring {rows} documents")
    conn.execute(f"UPDATE {fts}_data SET block=? WHERE id=1", (_varint(totals[0] - rows) + _varint(totals[1] - tokens),))


def sqlite_version_info() -> tuple[int, ...]:
    """The version of the running SQLite (a function so that tests can pose as an old one)."""
    return tuple(sqlite3.sqlite_version_info)


@functools.lru_cache(maxsize=8)
def _statistics_probe(runtime: str) -> tuple[str | None, bool]:
    """Prove on ``runtime`` (the SQLite version, the cache key) what the compact format needs: contentless_delete, and
    bm25 scores after a delete that equal those of an index that never held the row. Returns ``(reason, needs_patch)``:
    a reason when something is missing, else whether this SQLite leaves the statistics stale on delete."""
    try:
        conn = sqlite3.connect(":memory:", isolation_level=None)
        try:
            documents = {1: "alpha beta gamma delta", 2: "alpha alpha epsilon zeta eta", 3: "beta gamma", 4: "alpha theta iota kappa lambda mu",
                         5: "gamma gamma gamma", 6: "alpha"}
            for name in ("ref", "plain", "patched"):
                conn.execute(f"CREATE VIRTUAL TABLE {name} USING fts5(text, content='', contentless_delete=1, tokenize='porter unicode61')")
                for rowid, text in documents.items():
                    if name != "ref" or rowid != 2:
                        conn.execute(f"INSERT INTO {name}(rowid, text) VALUES (?,?)", (rowid, text))
            conn.execute("DELETE FROM plain WHERE rowid=2")
            conn.execute("BEGIN IMMEDIATE")
            _retire_statistics(conn, "patched", [2])
            conn.execute("DELETE FROM patched WHERE rowid=2")
            conn.execute("COMMIT")

            def scores(name):
                return conn.execute(f"SELECT rowid, bm25({name}) AS score FROM {name} WHERE {name} MATCH ? ORDER BY score, rowid",
                                    ('"alpha" OR "gamma" OR "theta"',)).fetchall()

            expected = scores("ref")
            if scores("plain") == expected:
                return None, False
            if scores("patched") == expected:
                return None, True
            return ("this SQLite does not keep FTS5 bm25 statistics exact after a contentless delete and they cannot be corrected here",
                    False)
        finally:
            conn.close()
    except (sqlite3.Error, SparseFormatError) as exc:
        return f"{type(exc).__name__}: {exc}", False


_REPAIR = ("use a Python build with SQLite 3.43 or newer (a newer Python, or a different SQLite library), or keep the previous "
           "format with SELF_DIRECT_SPARSE_FORMAT=legacy")


def compact_support() -> str | None:
    """None when this process can read and write the compact format, else the reason and the repair."""
    version = sqlite_version_info()
    if tuple(version) < COMPACT_MIN_SQLITE:
        return (f"SQLite {'.'.join(map(str, version))} is older than {'.'.join(map(str, COMPACT_MIN_SQLITE))}, which the compact "
                f"sparse format needs (FTS5 contentless_delete): {_REPAIR}")
    reason, _needs_patch = _statistics_probe(sqlite3.sqlite_version)
    if reason is not None:
        return f"the compact sparse format cannot be used with SQLite {sqlite3.sqlite_version} ({reason}): {_REPAIR}"
    return None


class CompactSparseIndex(SparseIndex):
    """The compact format, ``sparse-v2.sqlite``: the same search results as ``SparseIndex`` from a file about 70 % smaller.

    * ``chunks_fts`` is a contentless FTS5 table (``content=''``, ``contentless_delete=1``) over the masked text, with the
      same tokenizer and so the same stemming: it stores the inverted index and document sizes, **no text**.
    * ``chunk_rows(rowid, chunk_id, session_no, kind)`` maps a chunk to its FTS rowid (the same number); sessions and kinds
      are small integers resolved through ``sparse_sessions`` and ``sparse_kinds``. ``sparse_meta`` holds the format
      marker and the SQLite version that created the file.

    Equivalence with the legacy format is the contract (tests/test_sparse_compact.py): the same chunks, in the same order
    of writes, give the same ranked ids and the same bm25 scores, including ties (ascending rowid, which is what the
    legacy query returns), after deletes and re-upserts (the statistics are kept exact, see ``_retire_statistics``),
    session and ``allowed_ids`` filters and ``top_k``. The file never holds more text than the legacy one did, since it
    holds none: ``mask_secrets`` still runs on every text before it is tokenized.

    Opening a file in this format on a SQLite that cannot use it raises ``SparseFormatError`` naming the repair; nothing
    is substituted.
    """

    format = "compact"

    def __init__(self, db_path: Path) -> None:
        reason = compact_support()          # before the file is created: an unsupported SQLite leaves nothing behind
        if reason is not None:
            raise SparseFormatError(reason)
        self._patch_statistics = _statistics_probe(sqlite3.sqlite_version)[1]
        super().__init__(db_path)

    def _init_schema(self) -> None:
        check_request()
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            existing = self._conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sparse_meta'").fetchone()
            if existing:
                found = self._conn.execute("SELECT value FROM sparse_meta WHERE key='format'").fetchone()
                if not found or found[0] != COMPACT_FORMAT:
                    raise SparseFormatError(f"{self.db_path.name} is not a {COMPACT_FORMAT} sparse index "
                                            f"(format {found[0] if found else 'missing'!r}); it is left untouched")
            elif self._conn.execute("SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchone():
                raise SparseFormatError(f"{self.db_path.name} holds tables of another format; it is left untouched")
            self._conn.execute("CREATE TABLE IF NOT EXISTS sparse_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID")
            self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='', "
                               "contentless_delete=1, tokenize='porter unicode61')")
            self._conn.execute("CREATE TABLE IF NOT EXISTS chunk_rows(rowid INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE, "
                               "session_no INTEGER NOT NULL, kind_no INTEGER NOT NULL)")
            self._conn.execute("CREATE TABLE IF NOT EXISTS sparse_sessions(session_no INTEGER PRIMARY KEY, session_id TEXT NOT NULL UNIQUE)")
            self._conn.execute("CREATE TABLE IF NOT EXISTS sparse_kinds(kind_no INTEGER PRIMARY KEY, kind TEXT NOT NULL UNIQUE)")
            self._conn.execute("INSERT OR IGNORE INTO sparse_meta(key, value) VALUES ('format', ?)", (COMPACT_FORMAT,))
            self._conn.execute("INSERT OR IGNORE INTO sparse_meta(key, value) VALUES ('sqlite_version', ?)", (sqlite3.sqlite_version,))
            with publication():
                self._conn.commit()

    # ---- writes -----------------------------------------------------------------------------------------------------
    @contextmanager
    def _write(self):
        """One write transaction. The write lock is taken first (the statistics are read and corrected inside it)."""
        check_request()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            with publication():
                self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise

    def _intern(self, table: str, column: str, value: str, local: dict, *, create: bool) -> int | None:
        key = (table, value)
        if key in local:
            return local[key]
        row = self._conn.execute(f"SELECT rowid FROM {table} WHERE {column}=?", (value,)).fetchone()
        number = row[0] if row else (self._conn.execute(f"INSERT INTO {table}({column}) VALUES (?)", (value,)).lastrowid if create else None)
        local[key] = number
        return number

    def _rowids(self, ids) -> list[int]:
        out: list[int] = []
        for start in range(0, len(ids), 500):
            check_request()
            batch = ids[start:start + 500]
            out += [row[0] for row in self._conn.execute(
                "SELECT rowid FROM chunk_rows WHERE chunk_id IN (" + ",".join("?" * len(batch)) + ")", batch)]
        return out

    def _retire(self, ids: list[str]) -> list[int]:
        """The rowids of the stored ``ids``, taken out of the FTS statistics (before any FTS5 write of the transaction)."""
        rowids = self._rowids(ids)
        if rowids and self._patch_statistics:
            _retire_statistics(self._conn, "chunks_fts", rowids)
        return rowids

    def _drop(self, rowids: list[int]) -> None:
        for rowid in rowids:
            check_request()
            self._conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (rowid,))
            self._conn.execute("DELETE FROM chunk_rows WHERE rowid=?", (rowid,))

    def _add(self, chunk_id: str, session_id: str, kind: str, text: str, local: dict) -> None:
        masked = mask_secrets(text)
        session_no = self._intern("sparse_sessions", "session_id", session_id, local, create=True)
        kind_no = self._intern("sparse_kinds", "kind", kind, local, create=True)
        rowid = self._conn.execute("INSERT INTO chunk_rows(chunk_id, session_no, kind_no) VALUES (?,?,?)",
                                   (chunk_id, session_no, kind_no)).lastrowid
        self._conn.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?,?)", (rowid, masked))

    def clear(self) -> None:
        with self._write():
            self._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")     # also resets the statistics
            self._conn.execute("DELETE FROM chunk_rows")

    def upsert(self, chunk_id: str, session_id: str, kind: str, text: str) -> None:
        with self._write():
            self._drop(self._retire([chunk_id]))
            self._add(chunk_id, session_id, kind, text, {})

    def delete_ids(self, ids: list[str]) -> None:
        ids = list(ids)
        check_request()
        with self._write():
            self._drop(self._retire(ids))

    def upsert_many(self, chunks) -> None:
        latest: dict = {}
        for chunk in chunks:                  # a repeated id keeps its last text, at its last position (as the legacy loop does)
            latest.pop(chunk.chunk_id, None)
            latest[chunk.chunk_id] = chunk
        with self._write():
            self._drop(self._retire(list(latest)))
            local: dict = {}
            for chunk in latest.values():
                check_request()
                self._add(chunk.chunk_id, chunk.session_id, chunk.kind, chunk.text, local)

    # ---- reads ------------------------------------------------------------------------------------------------------
    @request_operation
    def search(
        self,
        query: str,
        top_k: int = 20,
        *,
        session_id: str | None = None,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float, int]]:
        q = _fts_query(query)
        sql = ("SELECT r.chunk_id AS chunk_id, bm25(chunks_fts) AS score "
               "FROM chunks_fts CROSS JOIN chunk_rows AS r ON r.rowid = chunks_fts.rowid "   # the FTS5 scan drives; rowid is r's key
               "WHERE chunks_fts MATCH ?")
        args: list = [q]
        if session_id:
            row = self._conn.execute("SELECT session_no FROM sparse_sessions WHERE session_id=?", (session_id,)).fetchone()
            if row is None:
                return []
            sql += " AND r.session_no = ?"
            args.append(row[0])
        # Ties rank in ascending rowid, which is the order the legacy table's query returns them in (rowids are assigned in
        # write order in both formats).
        sql += " ORDER BY score, chunks_fts.rowid LIMIT ?"
        args.append(-1 if allowed_ids is not None else top_k)
        cur = self._conn.execute(sql, args)
        out: list[tuple[str, float, int]] = []
        with closing(cur):
            for row in cur:
                check_request()
                if allowed_ids is not None and row["chunk_id"] not in allowed_ids:
                    continue
                if len(out) >= top_k:
                    break
                out.append((row["chunk_id"], float(-row["score"]), len(out) + 1))
        return out

    @request_operation
    def count(self, session_id: str | None = None) -> int:
        if session_id:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM chunk_rows WHERE session_no = (SELECT session_no FROM sparse_sessions WHERE session_id=?)",
                (session_id,)).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM chunk_rows").fetchone()
        return int(row["c"]) if row else 0

    def meta(self) -> dict[str, str]:
        return {row[0]: row[1] for row in self._conn.execute("SELECT key, value FROM sparse_meta")}

    @request_operation
    def verify_statistics(self) -> dict:
        """Compare the FTS5 statistics record with the stored document sizes and the chunk map. ``ok`` is the bar the
        bm25 parity rests on; the rebuild command and the tests call it."""
        rows = tokens = 0
        with closing(self._conn.execute("SELECT sz FROM chunks_fts_docsize")) as cur:
            for (size,) in cur:
                check_request()
                rows += 1
                tokens += _read_varints(size)[0]
        record = self._conn.execute("SELECT block FROM chunks_fts_data WHERE id=1").fetchone()
        totals = _read_varints(record[0]) if record and record[0] else [0, 0]
        mapped = self.count()
        return {"ok": totals == [rows, tokens] and mapped == rows, "documents": rows, "tokens": tokens,
                "recorded": totals, "mapped": mapped}


def sparse_file(index_dir, fmt: str) -> Path:
    return Path(index_dir) / (COMPACT_FILE if fmt == "compact" else LEGACY_FILE)


def open_sparse_index(index_dir, setting: str = "auto") -> tuple[SparseIndex, dict]:
    """Open the sparse index of ``index_dir`` in the format ``setting`` asks for (``auto``, ``legacy``, ``compact``).

    ``compact`` raises ``SparseFormatError`` when this SQLite cannot use it. ``auto`` uses the compact format when it can
    and otherwise the legacy one, and says so in the returned resolution (``reason``) so that ``audit_status`` shows it.
    The file of the format that is not opened is never touched."""
    if setting not in ("auto", "legacy", "compact"):
        raise ValueError(f"SELF_DIRECT_SPARSE_FORMAT must be auto, legacy or compact, not {setting!r}")
    reason = None
    if setting == "legacy":
        fmt = "legacy"
    else:
        reason = compact_support()
        if reason is None:
            fmt = "compact"
        elif setting == "compact":
            raise SparseFormatError(reason)
        else:
            fmt = "legacy"
    cls = CompactSparseIndex if fmt == "compact" else SparseIndex
    return cls(sparse_file(index_dir, fmt)), {"setting": setting, "format": fmt, "reason": reason}


def count_sparse_file(path) -> int | None:
    """Documents in a sparse file this process does not use (``audit_status`` shows both formats). Read-only: the file is
    never created, written or recovered, and only its id map is read, not the FTS5 tables. None when it cannot be read."""
    try:
        conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=0.25)
    except (sqlite3.Error, ValueError):
        return None
    try:
        return int(conn.execute("SELECT COUNT(*) FROM chunk_rows").fetchone()[0])
    except sqlite3.Error:
        return None
    finally:
        conn.close()
