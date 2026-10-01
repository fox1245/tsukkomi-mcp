"""Host hook receipts supplement, but never replace, native AGY transcript evidence.

The index directory must be outside agent write authority in enforced deployments.
Receipts are local host observations, not model-supplied successful-result claims.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from self_directing_mcp.session_events import iter_raw_lines


def _prefix(path: Path, step: int) -> tuple[int, str]:
    offset = 0
    for _line, start, end, raw in iter_raw_lines(path):
        row = json.loads(raw)
        index = row.get("step_index")
        if type(index) is not int:
            raise ValueError("cannot anchor receipt to unknown transcript layout")
        if index >= step:
            break
        offset = end
    from self_directing_mcp.index.ingest import prefix_digest
    return offset, prefix_digest(path, offset)


def result_digest(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _completed_results(path: Path) -> dict[int, str]:
    results = {}
    for _line, _start, _end, raw in iter_raw_lines(path):
        record = json.loads(raw)
        if (record.get("type") == "GENERIC" and record.get("source") == "MODEL"
                and record.get("status") in ("DONE", "ERROR", "CANCELED")
                and type(record.get("step_index")) is int):
            step = record["step_index"]
            if step in results:
                raise ValueError("ambiguous duplicate native execution step")
            results[step] = result_digest(record)
    return results


class ReceiptStore:
    def __init__(self, index_dir: Path):
        self.path = Path(index_dir) / "agy-receipts.sqlite"

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute("CREATE TABLE IF NOT EXISTS receipts (session_id TEXT, step INTEGER, data TEXT NOT NULL, PRIMARY KEY(session_id,step))")
        conn.execute("CREATE TABLE IF NOT EXISTS indexed (session_id TEXT PRIMARY KEY, digest TEXT NOT NULL)")
        return conn

    def record(self, session_id: str, path: Path, step: int, tool_call: dict,
               event: str, error: str | None = None) -> None:
        action = json.dumps(tool_call, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        with closing(self._connect()) as conn, conn:
            if event == "PreToolUse":
                length, digest = _prefix(path, step) if path.is_file() else (0, "")
                row = {"receipt_id": str(uuid4()), "path": str(path), "action": action,
                       "prefix_length": length, "prefix_hash": digest, "paired": False,
                       "step_index": step, "session_id": session_id,
                       "preexisting_result": path.is_file() and step in _completed_results(path)}
            else:
                previous = conn.execute("SELECT data FROM receipts WHERE session_id=? AND step=?", (session_id, step)).fetchone()
                row = json.loads(previous[0]) if previous else {}
                # A mismatched/duplicate Post must invalidate, never preserve,
                # an old successful pair at the same native execution index.
                paired = (row.get("action") == action and row.get("path") == str(path)
                          and row.get("paired") is False and not row.get("preexisting_result")
                          and isinstance(error, str))
                row.update(paired=paired, post_error=error,
                           post_action=action, post_path=str(path))
                if paired and not row.get("result_hash") and path.is_file():
                    observed = _completed_results(path).get(step)
                    if observed is not None:
                        row["result_hash"] = observed
            conn.execute("INSERT OR REPLACE INTO receipts VALUES (?,?,?)",
                         (session_id, step, json.dumps(row, sort_keys=True)))

    def snapshot(self, session_id: str, path: Path) -> tuple[dict[int, dict], str, str | None]:
        with closing(self._connect()) as conn, conn:
            rows = conn.execute("SELECT step,data FROM receipts WHERE session_id=? ORDER BY step", (session_id,)).fetchall()
            unbound = [(step, data, json.loads(data)) for step, data in rows
                       if not json.loads(data).get("result_hash")]
            if unbound:
                completed = _completed_results(path)
                for step, original, row in unbound:
                    if row.get("receipt_id") and row.get("path") == str(path) and step in completed:
                        row["result_hash"] = completed[step]
                        # First observation is immutable. Never rebind on rewrite.
                        conn.execute("UPDATE receipts SET data=? WHERE session_id=? AND step=? AND data=?",
                                     (json.dumps(row, sort_keys=True), session_id, step, original))
                rows = conn.execute("SELECT step,data FROM receipts WHERE session_id=? ORDER BY step", (session_id,)).fetchall()
            digest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
            indexed = conn.execute("SELECT digest FROM indexed WHERE session_id=?", (session_id,)).fetchone()
            return {step: json.loads(data) for step, data in rows}, digest, indexed[0] if indexed else None

    def mark_indexed(self, session_id: str, digest: str):
        with closing(self._connect()) as conn, conn:
            conn.execute("INSERT OR REPLACE INTO indexed VALUES (?,?)", (session_id, digest))


def linked_receipt(receipts: dict[int, dict], step: int, path: Path, before_byte: int,
                   *, result_record: dict, require_paired: bool = True) -> dict | None:
    row = receipts.get(step)
    if (not row or row.get("preexisting_result") or (require_paired and not row.get("paired"))
            or row.get("path") != str(path.resolve()) or row.get("result_hash") != result_digest(result_record)):
        return None
    length = row.get("prefix_length", 0)
    if not length or length > before_byte:
        return None
    from self_directing_mcp.index.ingest import prefix_digest
    if prefix_digest(path, length) != row.get("prefix_hash"):
        return None
    return row
