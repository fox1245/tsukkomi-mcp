"""Host hook receipts supplement, but never replace, native AGY transcript evidence.

The index directory must be outside agent write authority in enforced deployments.
Receipts are local host observations, not model-supplied successful-result claims.
"""
from __future__ import annotations

from contextlib import closing, nullcontext
import hashlib
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from self_directing_mcp.session_events import capture_source, iter_raw_lines, source_scope
from self_directing_mcp.request_control import check_request, publication


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
        check_request()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        try:
            with publication():
                conn.execute("CREATE TABLE IF NOT EXISTS receipts (session_id TEXT, step INTEGER, data TEXT NOT NULL, PRIMARY KEY(session_id,step))")
                conn.execute("CREATE TABLE IF NOT EXISTS indexed (session_id TEXT PRIMARY KEY, digest TEXT NOT NULL)")
            return conn
        except BaseException:
            conn.close()
            raise

    def record(self, session_id: str, path: Path, step: int, tool_call: dict,
               event: str, error: str | None = None, *, guard=nullcontext) -> None:
        action = json.dumps(tool_call, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        with guard(), closing(self._connect()) as conn:
            previous = conn.execute("SELECT data FROM receipts WHERE session_id=? AND step=?", (session_id, step)).fetchone()
            original = previous[0] if previous else None
        # Transcript inspection and receipt serialization are preparation, not writes.
        check_request()
        observation = capture_source(path) if path.is_file() else None
        with source_scope(observation) if observation is not None else nullcontext():
            if event == "PreToolUse":
                length, digest = _prefix(path, step) if observation is not None else (0, "")
                row = {"receipt_id": str(uuid4()), "path": str(path), "action": action,
                       "prefix_length": length, "prefix_hash": digest, "paired": False,
                       "step_index": step, "session_id": session_id,
                       "preexisting_result": observation is not None and step in _completed_results(path)}
            else:
                row = json.loads(original) if original else {}
                # A mismatched/duplicate Post invalidates an old successful pair.
                paired = (row.get("action") == action and row.get("path") == str(path)
                          and row.get("paired") is False and not row.get("preexisting_result")
                          and isinstance(error, str))
                row.update(paired=paired, post_error=error, post_action=action, post_path=str(path))
                if paired and not row.get("result_hash") and observation is not None:
                    observed = _completed_results(path).get(step)
                    if observed is not None:
                        row["result_hash"] = observed
        payload = json.dumps(row, sort_keys=True)
        with guard(), closing(self._connect()) as conn, conn:
            if (observation is not None and not observation.validate()) or (observation is None and path.is_file()):
                raise ValueError("source changed during receipt preparation")
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT data FROM receipts WHERE session_id=? AND step=?", (session_id, step)).fetchone()
            if (current[0] if current else None) != original:
                raise ValueError("stale hook receipt preparation")
            with publication():
                conn.execute("INSERT OR REPLACE INTO receipts VALUES (?,?,?)", (session_id, step, payload))
                check_request()
                conn.commit()

    def authority_token(self, session_id: str):
        """Read exact scoped evidence without recreating lost receipt authority."""
        check_request()
        try:
            before = self.path.stat()
        except FileNotFoundError:
            return ("missing", None, None)
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            rows = tuple(conn.execute("SELECT step,data FROM receipts WHERE session_id=? ORDER BY step",
                                      (session_id,)).fetchall())
        try:
            after = self.path.stat()
        except FileNotFoundError:
            return ("missing", None, None)
        identity = (after.st_dev, after.st_ino)
        if identity != (before.st_dev, before.st_ino):
            return ("changed", identity, None)
        return ("present", identity, self.rows_digest(rows))

    @staticmethod
    def rows_digest(rows) -> str:
        return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()

    def read_capture(self, session_id: str):
        with closing(self._connect()) as conn:
            rows = tuple(conn.execute("SELECT step,data FROM receipts WHERE session_id=? ORDER BY step", (session_id,)).fetchall())
            indexed = conn.execute("SELECT digest FROM indexed WHERE session_id=?", (session_id,)).fetchone()
        return rows, indexed[0] if indexed else None

    def prepare_bindings(self, capture, path: Path):
        rows, indexed = capture
        decoded = [(step, original, json.loads(original)) for step, original in rows]
        unbound = [item for item in decoded if not item[2].get("result_hash")]
        completed = _completed_results(path) if unbound else {}
        prepared = []
        for step, original, row in decoded:
            check_request()
            if (not row.get("result_hash") and row.get("receipt_id")
                    and row.get("path") == str(path) and step in completed):
                row["result_hash"] = completed[step]
                prepared.append((step, json.dumps(row, sort_keys=True)))
            else:
                prepared.append((step, original))
        prepared = tuple(prepared)
        return prepared, {step: json.loads(raw) for step, raw in prepared}, self.rows_digest(prepared), indexed

    def publish_bindings(self, session_id: str, capture, prepared_rows):
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = tuple(conn.execute("SELECT step,data FROM receipts WHERE session_id=? ORDER BY step", (session_id,)).fetchall())
            indexed = conn.execute("SELECT digest FROM indexed WHERE session_id=?", (session_id,)).fetchone()
            current = (rows, indexed[0] if indexed else None)
            if current != capture:
                raise ValueError("stale hook receipt preparation")
            if rows != prepared_rows:
                with publication():
                    for step, payload in prepared_rows:
                        check_request()
                        conn.execute("UPDATE receipts SET data=? WHERE session_id=? AND step=?", (payload, session_id, step))
                    check_request()
                    conn.commit()

    def snapshot(self, session_id: str, path: Path) -> tuple[dict[int, dict], str, str | None]:
        capture = self.read_capture(session_id)
        prepared, receipts, digest, indexed = self.prepare_bindings(capture, path)
        self.publish_bindings(session_id, capture, prepared)
        return receipts, digest, indexed

    def mark_indexed(self, session_id: str, digest: str):
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = tuple(conn.execute("SELECT step,data FROM receipts WHERE session_id=? ORDER BY step", (session_id,)).fetchall())
            if self.rows_digest(rows) != digest:
                raise ValueError("hook receipts changed before indexed marker")
            with publication():
                conn.execute("INSERT OR REPLACE INTO indexed VALUES (?,?)", (session_id, digest))
                check_request()
                conn.commit()


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
