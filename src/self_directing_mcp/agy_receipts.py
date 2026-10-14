"""Host hook receipts supplement, but never replace, native AGY transcript evidence.

The index directory must be outside agent write authority in enforced deployments.
Receipts are local host observations, not model-supplied successful-result claims.

A receipt anchors to the transcript bytes before its step (``prefix_hash``) and, once the host wrote the result, to
that result's digest (``result_hash``). Finding both used to parse the whole transcript at every hook. The
``Ledger`` keeps what those parses established about a *verified prefix* so a hook reads only what was appended;
see its docstring for why that cannot change what a receipt says.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import closing, nullcontext
import hashlib
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from self_directing_mcp.agy import RESULT_TYPES, load_line
from self_directing_mcp.session_events import capture_source, iter_raw_lines, same_identity, source_scope
from self_directing_mcp.request_control import check_request, publication

COMPLETED = ("DONE", "ERROR", "INVALID", "CANCELED")


def result_digest(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _scan(path: Path, start: int = 0):
    """One parse pass over the complete records from ``start``: ``(end, max_step, bad, results)``.

    ``results`` maps the step of every completed native result to its digest; two of them for one step are
    ambiguous and an error. ``bad`` says that a record has no integer ``step_index``. A line that is not a JSON
    object raises, as it always did: receipts are never anchored to a layout that cannot be read. Run it inside
    ``source_scope`` so every consumed byte is checked against the capture.
    """
    end, max_step, bad, results = start, -1, False, {}
    for _line, _begin, stop, raw in iter_raw_lines(path, start_byte=start):
        record, _padding = load_line(raw)
        index = record.get("step_index")
        if type(index) is int:
            if index > max_step:
                max_step = index
            if (record.get("type") in RESULT_TYPES and record.get("source") == "MODEL"
                    and record.get("status") in COMPLETED):
                if index in results:
                    raise ValueError("ambiguous duplicate native execution step")
                results[index] = result_digest(record)
        else:
            bad = True
        end = stop
    return end, max_step, bad, results


def _completed_results(path: Path) -> dict[int, str]:
    """The reference: every completed result of the whole transcript, by a full parse (no ledger)."""
    return _scan(path)[3]


def _first_at_or_after(path: Path, step: int) -> int:
    """End of the last record before the first one numbered ``step`` or more (the whole file when none is).

    A full parse. The ledger answers the usual question (a step beyond everything written) without it; this is
    what it falls back to for a step the transcript already reached, where the position matters."""
    offset = 0
    for _line, _start, end, raw in iter_raw_lines(path):
        row, _padding = load_line(raw)
        index = row.get("step_index")
        if type(index) is not int:
            raise ValueError("cannot anchor receipt to unknown transcript layout")
        if index >= step:
            break
        offset = end
    return offset


def _identity_text(identity) -> tuple[str, str]:
    return f"{identity[0]:x}", f"{identity[1]:x}"


class Ledger:
    """What a parse of one transcript's complete records established, reusable only while its prefix is intact.

    Stored per (conversation, file): the length of the scanned prefix and the SHA-256 of exactly those bytes, the
    file identity, the highest step number, whether any record lacked an integer ``step_index`` and, per completed
    native result, its digest. It is a cache of ``_scan``, never evidence:

    * it is used only after the digest of ``[0, length)`` was recomputed from the file in the same pass that hashed
      it for the current request (``observation.prefix_digest``; free when ``length`` was a capture checkpoint,
      otherwise a re-read), and only for the same path and file identity. Anything else parses the whole file;
    * the bytes appended since are parsed under the request's ``source_scope``, so what is read is what was hashed;
    * a result at or below the stored maximum step, an error, or any doubt rebuilds from byte 0, which is exactly
      the old full parse. A failed scan stores nothing.

    Writes are compare-and-set on the stored (length, digest) in the same transaction as the receipt or binding
    that needed them, so concurrent hooks can only lose a cache update, never corrupt one.
    """

    def __init__(self, session_id, path, state, persisted, asked):
        self.session_id, self.path = session_id, str(path)
        self.state, self._persisted, self._asked = state, persisted, frozenset(asked)
        self.length = self.digest = self.max_step = self.bad = None
        self._fresh, self._rebuilt = {}, False
        self.update = None
        self.advanced = False

    @staticmethod
    def read(conn, session_id, path, steps=()) -> "Ledger":
        """The stored state and the stored digests of ``steps`` (the only results this ledger can be asked for)."""
        row = conn.execute("SELECT dev,ino,length,digest,max_step,bad FROM scan_state WHERE session_id=? AND path=?",
                           (session_id, str(path))).fetchone()
        state = None if row is None else {"identity": (int(row[0], 16), int(row[1], 16)), "length": row[2],
                                          "digest": row[3], "max_step": row[4], "bad": bool(row[5])}
        steps, persisted = sorted(set(steps)), {}
        if state is not None:
            for start in range(0, len(steps), 400):
                batch = steps[start:start + 400]
                persisted.update(conn.execute(
                    "SELECT step,digest FROM scan_results WHERE session_id=? AND path=? AND step IN ("
                    + ",".join("?" * len(batch)) + ")", (session_id, str(path), *batch)))
        return Ledger(session_id, path, state, persisted, steps)

    def checkpoint(self):
        """The stored length: hash the capture there too and the verification below costs nothing."""
        return None if self.state is None else self.state["length"]

    def advance(self, observation) -> "Ledger":
        """Bring the facts up to ``observation.complete_end``. Call inside ``source_scope(observation)``."""
        if self.advanced:
            return self
        state, end, path = self.state, observation.complete_end, Path(self.path)
        usable = (state is not None and observation.path == path
                  and same_identity(state["identity"], observation.identity) and state["length"] <= end
                  and observation.prefix_digest(state["length"]) == state["digest"])
        rebuilt = not usable
        if usable:
            tail_end, tail_max, tail_bad, tail = _scan(path, state["length"]) if state["length"] < end else (
                end, -1, False, {})
            if any(step <= state["max_step"] for step in tail):
                rebuilt = True  # out of order or repeated: only a full parse can tell which is which
        if rebuilt:
            tail_end, tail_max, tail_bad, tail = _scan(path)
            base_max, base_bad = -1, False
        else:
            base_max, base_bad = state["max_step"], state["bad"]
        if tail_end != end:
            raise ValueError("transcript scan did not reach the observed record boundary")
        self.length, self.digest = end, observation.prefix_digest(end)
        self.max_step, self.bad = max(base_max, tail_max), base_bad or tail_bad
        self._fresh, self._rebuilt, self.advanced = tail, rebuilt, True
        if rebuilt or tail_end != state["length"]:
            self.update = {"old": None if state is None else (state["length"], state["digest"]),
                           "new": (end, self.digest, self.max_step, self.bad,
                                   *_identity_text(observation.identity)),
                           "results": tail, "rebuilt": rebuilt}
        return self

    def completed(self, step: int):
        """Digest of the completed result at ``step`` anywhere in the verified prefix, else None."""
        found = self._fresh.get(step)
        if found is None and not self._rebuilt:
            if step not in self._asked:
                raise LookupError("this ledger was not read for that step")
            found = self._persisted.get(step)
        return found

    def persist(self, conn) -> None:
        """Store what ``advance`` learned, inside the caller's write transaction. A cache: it is skipped when the
        stored state is no longer the one this ledger started from."""
        update = self.update
        if update is None:
            return
        stored = conn.execute("SELECT length,digest FROM scan_state WHERE session_id=? AND path=?",
                              (self.session_id, self.path)).fetchone()
        if (tuple(stored) if stored else None) != update["old"]:
            return
        if update["rebuilt"]:
            conn.execute("DELETE FROM scan_results WHERE session_id=? AND path=?", (self.session_id, self.path))
        conn.executemany("INSERT OR REPLACE INTO scan_results VALUES (?,?,?,?)",
                         [(self.session_id, self.path, step, digest) for step, digest in update["results"].items()])
        length, digest, max_step, bad, dev, ino = update["new"]
        conn.execute("INSERT OR REPLACE INTO scan_state VALUES (?,?,?,?,?,?,?,?)",
                     (self.session_id, self.path, dev, ino, length, digest, max_step, int(bad)))


def _anchor(ledger: Ledger, observation, path: Path, step: int) -> tuple[int, str]:
    """``(prefix_length, prefix_hash)`` for a receipt of ``step``: the end of the last record before it."""
    ledger.advance(observation)
    if step > ledger.max_step:
        if ledger.bad:
            raise ValueError("cannot anchor receipt to unknown transcript layout")
        offset = ledger.length  # nothing written yet is numbered ``step`` or more: the end of the complete records
    else:
        offset = _first_at_or_after(path, step)
    # Free when the offset is where the capture stopped (the usual case); a plain re-read otherwise.
    return offset, observation.prefix_digest(offset)


_last_digest: tuple | None = None  # (rows, digest) of the latest ``ReceiptStore.rows_digest``; replaced whole, never edited


class _Receipts(Mapping):
    """The receipts of a conversation by step, decoded when first used.

    A long conversation has thousands of receipts and a sync looks at a handful: the results it parses and the lowest
    step. Equal to the dict of every decoded receipt for every use the parser makes of it."""

    def __init__(self, rows):
        self._raw, self._decoded = dict(rows), {}

    def __getitem__(self, step):
        found = self._decoded.get(step)
        if found is None:
            found = self._decoded[step] = json.loads(self._raw[step])  # KeyError for a step with no receipt
        return found

    def __iter__(self):
        return iter(self._raw)

    def __len__(self):
        return len(self._raw)


_BOUND_KEY = '"result_hash": "'


def _is_bound(original: str) -> bool:
    """Is this receipt's ``result_hash`` already a nonempty string? Exact for the ``json.dumps(sort_keys=True)`` text
    receipts are stored as; ``False`` means "decode it and look", never "unbound".

    A quote that is not preceded by a backslash is always structural in JSON text, and a string value is never
    followed by a colon, so the key text at such a position is the key of this flat object, not part of a value."""
    at = original.find(_BOUND_KEY)
    return at >= 0 and original[at - 1:at] != "\\" and original[at + len(_BOUND_KEY):at + len(_BOUND_KEY) + 1] != '"'


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
                # The Ledger (derived, per conversation and file): a verified prefix and what a parse found in it.
                conn.execute("CREATE TABLE IF NOT EXISTS scan_state (session_id TEXT NOT NULL, path TEXT NOT NULL,"
                             " dev TEXT NOT NULL, ino TEXT NOT NULL, length INTEGER NOT NULL, digest TEXT NOT NULL,"
                             " max_step INTEGER NOT NULL, bad INTEGER NOT NULL, PRIMARY KEY(session_id,path))")
                conn.execute("CREATE TABLE IF NOT EXISTS scan_results (session_id TEXT NOT NULL, path TEXT NOT NULL,"
                             " step INTEGER NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(session_id,path,step))")
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
            ledger = Ledger.read(conn, session_id, Path(path).resolve(), (step,))
        # Transcript inspection and receipt serialization are preparation, not writes.
        check_request()
        # One pass over the file: its SHA-256, with a state at the verified length of the ledger, so what the
        # ledger knows is checked against these very bytes and only the appended records are parsed.
        marks = () if ledger.checkpoint() is None else (ledger.checkpoint(),)
        observation = capture_source(path, checkpoints=marks) if path.is_file() else None
        with source_scope(observation) if observation is not None else nullcontext():
            if event == "PreToolUse":
                length, digest = _anchor(ledger, observation, path, step) if observation is not None else (0, "")
                row = {"receipt_id": str(uuid4()), "path": str(path), "action": action,
                       "prefix_length": length, "prefix_hash": digest, "paired": False,
                       "step_index": step, "session_id": session_id,
                       "preexisting_result": observation is not None and ledger.advance(observation).completed(step) is not None}
            else:
                row = json.loads(original) if original else {}
                # A mismatched/duplicate Post invalidates an old successful pair.
                paired = (row.get("action") == action and row.get("path") == str(path)
                          and row.get("paired") is False and not row.get("preexisting_result")
                          and isinstance(error, str))
                row.update(paired=paired, post_error=error, post_action=action, post_path=str(path))
                if paired and not row.get("result_hash") and observation is not None:
                    observed = ledger.advance(observation).completed(step)
                    if observed is not None:
                        row["result_hash"] = observed
        payload = json.dumps(row, sort_keys=True)
        # A failed full read (even a transient one) is not repaired by a successful bounded tail check.
        source_ok = observation is None or observation.validate_prefix()  # bounded stage check (identity, size, tail; see validate_prefix), off the shared lock
        with guard(), closing(self._connect()) as conn, conn:
            if (not source_ok or (observation is not None and not observation.validate_prefix(locked=True))
                    or (observation is None and path.is_file())):
                raise ValueError("source changed during receipt preparation")
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT data FROM receipts WHERE session_id=? AND step=?", (session_id, step)).fetchone()
            if (current[0] if current else None) != original:
                raise ValueError("stale hook receipt preparation")
            with publication():
                conn.execute("INSERT OR REPLACE INTO receipts VALUES (?,?,?)", (session_id, step, payload))
                ledger.persist(conn)
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
        if not same_identity(identity, (before.st_dev, before.st_ino)):
            return ("changed", identity, None)
        return ("present", identity, self.rows_digest(rows))

    @staticmethod
    def rows_digest(rows) -> str:
        """SHA-256 of the receipts as ``(step, json)`` rows. A request checks its receipts against this several times
        (authority tokens, bindings, the indexed marker): the digest of rows equal to the last ones is that digest,
        and comparing is a memory scan where hashing re-encodes every receipt."""
        global _last_digest
        last = _last_digest
        if last is not None and last[0] == rows:
            return last[1]
        digest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
        _last_digest = (rows, digest)
        return digest

    def read_capture(self, session_id: str):
        with closing(self._connect()) as conn:
            rows = tuple(conn.execute("SELECT step,data FROM receipts WHERE session_id=? ORDER BY step", (session_id,)).fetchall())
            indexed = conn.execute("SELECT digest FROM indexed WHERE session_id=?", (session_id,)).fetchone()
        return rows, indexed[0] if indexed else None

    def checkpoints(self, session_id: str, path: Path) -> tuple[int, ...]:
        """Offsets whose prefix digests a sync of ``path`` is going to ask for: every receipt's anchor, and the
        verified length of the ledger. A capture that hashes the file with a state at each of them makes those
        digests free; one that does not still gets them right, by re-reading."""
        with closing(self._connect()) as conn:
            marks = {row[0] for row in conn.execute(
                "SELECT json_extract(data,'$.prefix_length') FROM receipts WHERE session_id=?", (session_id,))}
            length = conn.execute("SELECT length FROM scan_state WHERE session_id=? AND path=?",
                                  (session_id, str(path))).fetchone()
        if length:
            marks.add(length[0])
        return tuple(sorted(mark for mark in marks if type(mark) is int and mark > 0))

    def prepare_bindings(self, capture, path: Path, *, session_id: str | None = None, observation=None):
        """``(rows, receipts, digest, indexed, ledger)``: ``rows`` with every receipt whose result is now in the
        transcript bound to its digest. With ``session_id`` and the request's ``observation`` (inside its
        ``source_scope``) the ledger finds those results by parsing only what was appended; otherwise the whole
        transcript is parsed. ``ledger`` (or None) is published with the rows by ``publish_bindings``."""
        rows, indexed = capture
        # Only a receipt that has no result digest yet, and names this file, can be bound now: of thousands,
        # nearly all are bound already and are not even decoded.
        unbound = {}
        for step, original in rows:
            if _is_bound(original):
                continue
            row = json.loads(original)
            if not row.get("result_hash") and row.get("receipt_id") and row.get("path") == str(path):
                unbound[step] = row
        completed, ledger = {}, None
        if unbound and observation is not None and session_id is not None:
            with closing(self._connect()) as conn:
                ledger = Ledger.read(conn, session_id, Path(path).resolve(), unbound)
            ledger.advance(observation)
            completed = {step: found for step in unbound if (found := ledger.completed(step)) is not None}
        elif unbound:
            completed = _completed_results(path)
        prepared = []
        for step, original in rows:
            check_request()
            if step in unbound and step in completed:
                row = unbound[step]
                row["result_hash"] = completed[step]
                prepared.append((step, json.dumps(row, sort_keys=True)))
            else:
                prepared.append((step, original))
        prepared = tuple(prepared)
        return prepared, _Receipts(prepared), self.rows_digest(prepared), indexed, ledger

    def publish_bindings(self, session_id: str, capture, prepared_rows, ledger: Ledger | None = None):
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = tuple(conn.execute("SELECT step,data FROM receipts WHERE session_id=? ORDER BY step", (session_id,)).fetchall())
            indexed = conn.execute("SELECT digest FROM indexed WHERE session_id=?", (session_id,)).fetchone()
            current = (rows, indexed[0] if indexed else None)
            if current != capture:
                raise ValueError("stale hook receipt preparation")
            if rows != prepared_rows or (ledger is not None and ledger.update is not None):
                with publication():
                    if rows != prepared_rows:
                        for step, payload in prepared_rows:
                            check_request()
                            conn.execute("UPDATE receipts SET data=? WHERE session_id=? AND step=?", (payload, session_id, step))
                    if ledger is not None:
                        ledger.persist(conn)
                    check_request()
                    conn.commit()

    def snapshot(self, session_id: str, path: Path) -> tuple[dict[int, dict], str, str | None]:
        capture = self.read_capture(session_id)
        prepared, receipts, digest, indexed, ledger = self.prepare_bindings(capture, path)
        self.publish_bindings(session_id, capture, prepared, ledger)
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
                   *, result_record: dict, require_paired: bool = True, resolved: str | None = None) -> dict | None:
    """The receipt that ties the result record at ``before_byte`` to a call, or None.

    ``resolved`` is ``str(path.resolve())`` when the caller already has it. The anchor is the plain SHA-256 of
    the file's bytes ``[0, prefix_length)`` as they are NOW: ``prefix_digest`` serves it from the request's capture
    pass when the capture stopped at that offset and otherwise reads the file again. It never consults a block
    table, a sample or a cache, so a cheap observation cannot create a link a full check would reject.
    """
    row = receipts.get(step)
    if (not row or row.get("preexisting_result") or (require_paired and not row.get("paired"))
            or row.get("path") != (resolved if resolved is not None else str(path.resolve()))
            or row.get("result_hash") != result_digest(result_record)):
        return None
    length = row.get("prefix_length", 0)
    if not length or length > before_byte:
        return None
    from self_directing_mcp.index.ingest import prefix_digest
    if prefix_digest(path, length) != row.get("prefix_hash"):
        return None
    return row
