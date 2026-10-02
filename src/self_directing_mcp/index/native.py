"""Load sqliteai/sqlite-vector directly; no Python wheel is required."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
from urllib.parse import quote

import numpy as np

from self_directing_mcp.request_control import check_request, publication, request_operation


class NativeVectorStore:
    table: str
    id_column: str

    def __init__(self, db_path: Path, dim: int = 1024, *, extension_path: Path | None = None):
        check_request()
        self.dim = dim
        self.db_path = Path(db_path)
        self.extension_path = str(Path(extension_path).expanduser().resolve()) if extension_path else "vector"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            self._load(self._conn)
            check_request()
            self._conn.execute("BEGIN")
            self._conn.execute(f"CREATE TABLE IF NOT EXISTS {self.table} ({self.id_column} TEXT PRIMARY KEY, embedding BLOB NOT NULL)")
            with publication():
                self._conn.commit()
            check_request()
            self._initialize(self._conn)
            check_request()
        except Exception:
            self._conn.close()
            raise

    def _load(self, conn):
        try:
            conn.enable_load_extension(True)
            try:
                conn.load_extension(self.extension_path)
            finally:
                conn.enable_load_extension(False)
            # Reject similarly named but incompatible extensions such as sqlite-vec.
            self.extension_version = conn.execute("SELECT vector_version()").fetchone()[0]
            self.compute_backend = conn.execute("SELECT vector_backend()").fetchone()[0]
        except (sqlite3.Error, AttributeError, OSError) as exc:
            raise RuntimeError(
                "sqliteai/sqlite-vector could not be loaded. Install the native extension with "
                "scripts/setup_sqlite_vector.py and set "
                "SELF_DIRECT_SQLITE_VECTOR_PATH to its absolute DLL/SO/dylib path. "
                "NumPy requires an explicit SELF_DIRECT_DENSE_BACKEND=numpy selection."
            ) from exc

    def _initialize(self, conn):
        conn.execute("SELECT vector_init(?, 'embedding', ?)",
                     (self.table, f"type=FLOAT32,dimension={self.dim},distance=COSINE"))

    def _blob(self, vector):
        arr = np.asarray(vector, dtype=np.float32)
        if arr.shape != (self.dim,) or not np.isfinite(arr).all():
            raise ValueError(f"Expected a finite vector with dimension {self.dim}")
        return arr.tobytes()

    def refresh(self):
        check_request()  # SQLite reads are current automatically.

    @request_operation
    def ids(self):
        out = set()
        with closing(self._conn.execute(f"SELECT {self.id_column} FROM {self.table}")) as rows:
            for row in rows:
                check_request()
                out.add(row[0])
        return out

    def clear(self):
        check_request()
        with self._conn:
            self._conn.execute(f"DELETE FROM {self.table}")
            with publication():
                self._conn.commit()

    def upsert(self, item_id, vector):
        self.upsert_many([(item_id, vector)])

    def upsert_many(self, pairs):
        rows = []
        for item_id, vector in pairs:
            check_request()
            rows.append((item_id, self._blob(vector)))
        with self._conn:
            for offset in range(0, len(rows), 128):
                check_request()
                self._conn.executemany(
                    f"INSERT OR REPLACE INTO {self.table}({self.id_column}, embedding) VALUES (?, ?)",
                    rows[offset:offset + 128])
            with publication():
                self._conn.commit()

    def delete_ids(self, ids):
        check_request()
        with self._conn:
            batch = []
            for item_id in ids:
                check_request()
                batch.append((item_id,))
                if len(batch) == 128:
                    self._conn.executemany(f"DELETE FROM {self.table} WHERE {self.id_column}=?", batch)
                    batch.clear()
            if batch:
                self._conn.executemany(f"DELETE FROM {self.table} WHERE {self.id_column}=?", batch)
            with publication():
                self._conn.commit()

    def _search(self, conn, query, top_k, allowed_ids=None):
        check_request()
        if top_k <= 0 or allowed_ids == set():
            return []
        blob = self._blob(query)
        # Streaming exact scan: filter before LIMIT, including session-scoped searches.
        # SQL ordering also resolves ties across the complete population deterministically.
        where = "" if allowed_ids is None else f"WHERE e.{self.id_column} IN (SELECT value FROM json_each(?))"
        params = [self.table, blob]
        if allowed_ids is not None:
            params.append(json.dumps(sorted(allowed_ids)))
        params.append(top_k)
        out = []
        with closing(conn.execute(
                f"SELECT e.{self.id_column}, 1.0 - v.distance AS score "
                f"FROM vector_full_scan(?, 'embedding', ?) AS v "
                f"JOIN {self.table} AS e ON e.rowid = v.rowid {where} "
                f"ORDER BY v.distance, e.{self.id_column} LIMIT ?", params)) as rows:
            for item_id, score in rows:
                check_request()
                out.append((item_id, float(score)))
        check_request()
        return out

    @request_operation
    def search(self, query, top_k=20, allowed_ids=None):
        return self._search(self._conn, query, top_k, allowed_ids)

    @request_operation
    def search_readonly(self, query, top_k=20):
        uri = "file:" + quote(self.db_path.resolve().as_posix(), safe="/:") + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            self._load(conn)
            self._initialize(conn)
            return self._search(conn, query, top_k)

    @request_operation
    def count(self):
        return int(self._conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0])

    def close(self):
        self._conn.close()
