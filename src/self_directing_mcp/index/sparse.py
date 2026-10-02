from __future__ import annotations

from contextlib import closing
import re
import sqlite3
from pathlib import Path

from self_directing_mcp.request_control import check_request, publication, request_operation


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
    """SQLite FTS5 BM25 sparse index over session chunks."""

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

    def upsert(self, chunk_id: str, session_id: str, kind: str, text: str) -> None:
        check_request()
        with self._conn:
            row = self._conn.execute("SELECT doc_rowid FROM chunk_rows WHERE chunk_id=?", (chunk_id,)).fetchone()
            if row:
                self._conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (row[0],))
            inserted = self._conn.execute(
                "INSERT INTO chunks_fts(chunk_id,session_id,kind,text) VALUES (?,?,?,?)",
                (chunk_id, session_id, kind, text))
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
                    (chunk.chunk_id, chunk.session_id, chunk.kind, chunk.text))
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
            row = self._conn.execute("SELECT COUNT(*) AS c FROM chunks_fts").fetchone()
        return int(row["c"]) if row else 0
