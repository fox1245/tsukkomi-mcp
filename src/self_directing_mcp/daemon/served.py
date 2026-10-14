"""``served_by`` accounting: which process judged a hook.

Host hook JSON must not carry extra top-level keys, so the answer's origin is recorded in the hook state database
that already holds ``hook_health`` (``<index_dir>/claude-hook-state.sqlite`` / ``agy-hook-state.sqlite``): a
``served_by`` table with one counter row per event and origin. The daemon counts in memory and writes at most every
``FLUSH_SEC`` and at exit, so a judgement never waits for a database write. A standalone hook process records
nothing: a database without ``served_by="daemon"`` rows was never served by the daemon.
"""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3
import threading
import time

FLUSH_SEC = 30.0
STATE_FILES = {"claude": "claude-hook-state.sqlite", "agy": "agy-hook-state.sqlite"}


class ServedLedger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str, str, str], list[float]] = {}
        self._flushed_at = time.monotonic()

    def add(self, host: str, index_dir, event: str, served_by: str = "daemon") -> None:
        key = (str(index_dir), STATE_FILES[host], str(event)[:40], served_by)
        with self._lock:
            entry = self._counts.setdefault(key, [0, 0.0])
            entry[0] += 1
            entry[1] = time.time()

    def due(self) -> bool:
        return time.monotonic() - self._flushed_at >= FLUSH_SEC

    def flush(self) -> int:
        """Write and clear the counters; returns how many rows could not be written (they are kept)."""
        with self._lock:
            pending, self._counts = self._counts, {}
            self._flushed_at = time.monotonic()
        failed = 0
        for (index_dir, state_file, event, served_by), (count, last_at) in pending.items():
            try:
                self._write(Path(index_dir) / state_file, event, served_by, int(count), last_at)
            except Exception:
                failed += 1
                with self._lock:                     # keep the count for the next flush
                    entry = self._counts.setdefault((index_dir, state_file, event, served_by), [0, 0.0])
                    entry[0] += count
                    entry[1] = max(entry[1], last_at)
        return failed

    @staticmethod
    def _write(path: Path, event: str, served_by: str, count: int, last_at: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=2.0, isolation_level=None)) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS served_by (event TEXT NOT NULL, served_by TEXT NOT NULL, "
                         "total INTEGER NOT NULL, last_at REAL NOT NULL, PRIMARY KEY (event, served_by))")
            conn.execute("INSERT INTO served_by VALUES (?,?,?,?) ON CONFLICT(event, served_by) DO UPDATE SET "
                         "total=total+excluded.total, last_at=excluded.last_at", (event, served_by, count, last_at))


def read(index_dir, host: str) -> list[dict]:
    """The recorded counters (for diagnostics and tests)."""
    path = Path(index_dir) / STATE_FILES[host]
    if not path.exists():
        return []
    try:
        with closing(sqlite3.connect(path, timeout=2.0)) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute("SELECT * FROM served_by ORDER BY event, served_by")]
    except sqlite3.Error:
        return []
