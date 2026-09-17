from __future__ import annotations

import re
import sqlite3
from pathlib import Path


def _fts_query(raw: str) -> str:
    toks = re.findall(r"[A-Za-z0-9_\-]+", raw)
    cleaned: list[str] = []
    for t in toks:
        if len(t) < 2:
            continue
        cleaned.append(f'"{t}"')
    if not cleaned:
        return '""'
    return " OR ".join(cleaned)


class SparseIndex:
    """SQLite FTS5 BM25 sparse index over session chunks."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
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
        self._conn.commit()
        # Preserve the existing FTS table; migrate only a compact id -> rowid map.
        exists = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk_rows'").fetchone()
        if not exists:
            with self._conn:
                self._conn.execute("CREATE TABLE chunk_rows(chunk_id TEXT PRIMARY KEY, doc_rowid INTEGER UNIQUE NOT NULL)")
                self._conn.execute("INSERT INTO chunk_rows SELECT chunk_id,rowid FROM chunks_fts")

    def clear(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM chunks_fts")
            self._conn.execute("DELETE FROM chunk_rows")

    def ids(self) -> set[str]:
        return {row[0] for row in self._conn.execute("SELECT chunk_id FROM chunk_rows")}

    def upsert(self, chunk_id: str, session_id: str, kind: str, text: str) -> None:
        with self._conn:
            row = self._conn.execute("SELECT doc_rowid FROM chunk_rows WHERE chunk_id=?", (chunk_id,)).fetchone()
            if row:
                self._conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (row[0],))
            inserted = self._conn.execute(
                "INSERT INTO chunks_fts(chunk_id,session_id,kind,text) VALUES (?,?,?,?)",
                (chunk_id, session_id, kind, text))
            self._conn.execute("INSERT OR REPLACE INTO chunk_rows VALUES (?,?)", (chunk_id, inserted.lastrowid))

    def delete_ids(self, ids: list[str]) -> None:
        with self._conn:
            for cid in ids:
                row = self._conn.execute("SELECT doc_rowid FROM chunk_rows WHERE chunk_id=?", (cid,)).fetchone()
                if row:
                    self._conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (row[0],))
                    self._conn.execute("DELETE FROM chunk_rows WHERE chunk_id=?", (cid,))

    def upsert_many(self, chunks) -> None:
        with self._conn:
            for chunk in chunks:
                row = self._conn.execute("SELECT doc_rowid FROM chunk_rows WHERE chunk_id=?", (chunk.chunk_id,)).fetchone()
                if row:
                    self._conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (row[0],))
                inserted = self._conn.execute(
                    "INSERT INTO chunks_fts(chunk_id,session_id,kind,text) VALUES (?,?,?,?)",
                    (chunk.chunk_id, chunk.session_id, chunk.kind, chunk.text))
                self._conn.execute("INSERT OR REPLACE INTO chunk_rows VALUES (?,?)",
                                   (chunk.chunk_id, inserted.lastrowid))

    def search(
        self,
        query: str,
        top_k: int = 20,
        *,
        session_id: str | None = None,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float, int]]:
        q = _fts_query(query)
        try:
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
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            return []
        out: list[tuple[str, float, int]] = []
        rows = [r for r in rows if allowed_ids is None or r["chunk_id"] in allowed_ids][:top_k]
        for i, row in enumerate(rows, start=1):
            out.append((row["chunk_id"], float(-row["score"]), i))
        return out

    def count(self, session_id: str | None = None) -> int:
        if session_id:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM chunks_fts WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM chunks_fts").fetchone()
        return int(row["c"]) if row else 0
