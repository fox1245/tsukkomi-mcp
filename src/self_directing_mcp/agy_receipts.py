"""Host hook receipts supplement, but never replace, native AGY transcript evidence.

The index directory must be outside agent write authority in enforced deployments.
Receipts are local host observations, not model-supplied successful-result claims.

A receipt anchors to the transcript bytes before its step (``prefix_hash``) and, once the host wrote the result, to
that result's digest (``result_hash``). Finding both used to parse the whole transcript at every hook. The
``Ledger`` keeps what those parses established about a *verified prefix* so a hook reads only what was appended;
see its docstring for why that cannot change what a receipt says.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import closing, nullcontext
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
from uuid import uuid4

from self_directing_mcp import journal as _journal
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
        """The stored state and the stored digests of ``steps`` (the only results this ledger can be asked for),
        read in one transaction so they describe the same publication."""
        owned = not conn.in_transaction
        if owned:
            conn.execute("BEGIN")
        try:
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
        finally:
            if owned:
                conn.rollback()
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
        rows = [(self.session_id, self.path, step, digest) for step, digest in update["results"].items()]
        for start in range(0, len(rows), 500):
            check_request()  # a rebuild writes one row per result; a stop rolls the whole transaction back
            conn.executemany("INSERT OR REPLACE INTO scan_results VALUES (?,?,?,?)", rows[start:start + 500])
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


_MISS = object()


class _Reader:
    """A resident engine's persistent read-only connection to one receipts database, and what it last read from it.

    Every request used to read all the receipts of a conversation (thousands of rows) a dozen times: the authority
    tokens that detect a change between two phases, the capture of a sync, the checkpoint list. Here a value is read once
    and served again until the database changes. The witness is SQLite's own: ``PRAGMA data_version`` differs between two
    calls on one connection exactly when ANOTHER connection committed in between, and every writer of this database
    (``record``, ``publish_bindings``, ``mark_indexed``, another process) is another connection. The version is read
    BEFORE the value, so a commit that lands between the two only makes the next call read again. The file identity is
    compared with the path on every use (``_reader_for``): a database that was deleted, replaced or recreated is a new
    reader, never served from the old one's values. A write that bypasses SQLite (an editor on the file) moves no
    version; the same assumption the index store's row-count cache already makes."""

    def __init__(self, connection, identity) -> None:
        self.connection, self.identity = connection, identity
        self.lock = threading.Lock()
        self.values: OrderedDict = OrderedDict()

    def get(self, key, load):
        with self.lock:
            version = self.connection.execute("PRAGMA data_version").fetchone()[0]
            held = self.values.get(key)
            if held is not None and held[0] == version:
                self.values.move_to_end(key)
                return held[1]
            value = load(self.connection)
            self.values[key] = (version, value)
            self.values.move_to_end(key)
            while len(self.values) > _READER_VALUES:
                self.values.popitem(last=False)
            return value

    def close(self) -> None:
        with self.lock:
            try:
                self.connection.close()
            except sqlite3.Error:
                pass


_READER_VALUES = 12     # (kind, session) values one reader keeps
_READER_LIMIT = 4       # receipts databases kept open by one process
_READERS: OrderedDict[str, _Reader] = OrderedDict()
_READERS_LOCK = threading.Lock()


def _reader_for(path: Path):
    """The reader of this receipts database, opened on first use; None when it cannot be (no file, SQLite refuses)."""
    key = str(path)
    try:
        stat = os.stat(path)
    except OSError:
        return None
    identity = (stat.st_dev, stat.st_ino)
    with _READERS_LOCK:
        reader = _READERS.get(key)
        if reader is not None and reader.identity != identity:
            _READERS.pop(key).close()               # the file at this path is not the one the reader holds open
            reader = None
        if reader is None:
            try:
                connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False)
                again = os.stat(path)
            except (sqlite3.Error, OSError):
                return None
            if (again.st_dev, again.st_ino) != identity:      # replaced while opening: the next call starts over
                connection.close()
                return None
            reader = _READERS[key] = _Reader(connection, identity)
            while len(_READERS) > _READER_LIMIT:
                _READERS.popitem(last=False)[1].close()
        else:
            _READERS.move_to_end(key)
        return reader


def _drop_reader(path: Path, reader) -> None:
    with _READERS_LOCK:
        if _READERS.get(str(path)) is reader:
            del _READERS[str(path)]
    reader.close()


class ReceiptStore:
    def __init__(self, index_dir: Path, *, held=None, journal: str | None = None):
        """``held``: the ``index.deferred.DerivedBuffer`` of a resident engine, or None. The "indexed" marker (which receipts
        digest the index was built on) is advisory, so such an engine holds it there and writes it with the next
        receipt it records, at idle or at close, instead of committing it by itself; every read goes through
        ``_indexed``, which sees the held value first."""
        self.path = Path(index_dir) / "agy-receipts.sqlite"
        self.held = held
        # ``journal``: "auto" or "delete" (``None``: ``journal.DEFAULT_POLICY``); applied to every connection this store
        # opens, always with ``synchronous=FULL``. See the module ``journal``.
        self.journal = _journal.validate_policy(_journal.DEFAULT_POLICY if journal is None else journal)

    @staticmethod
    def release(index_dir: Path) -> None:
        """Close this process's persistent reader of the index directory's receipts database (engine close)."""
        path = Path(index_dir) / "agy-receipts.sqlite"
        with _READERS_LOCK:
            reader = _READERS.pop(str(path), None)
        if reader is not None:
            reader.close()

    def journal_report(self) -> dict:
        """The journal mode the receipts database is in now, what the policy and file system call for, and whether
        they differ (``pending``). Reads the file; creates nothing."""
        return _journal.inspect_journal(self.path, self.journal)

    def checkpoint(self) -> bool:
        """Fold the write-ahead log into the main file (engine close, daemon idle exit); cheap without a ``-wal``."""
        return _journal.checkpoint_file(self.path)

    def _cached(self, key, load):
        """``load(connection)`` through a persistent reader (resident engines only, i.e. with ``held``), else ``_MISS``:
        the caller then reads as it always did. Any SQLite trouble also means ``_MISS`` and drops the reader."""
        if self.held is None:
            return _MISS
        reader = _reader_for(self.path)
        if reader is None:
            return _MISS
        try:
            return reader.get(key, load)
        except sqlite3.Error:
            _drop_reader(self.path, reader)
            return _MISS

    def _rows(self, session_id, connection=None):
        """The ``(step, json)`` receipts of a conversation, ordered by step."""
        def load(conn):
            return tuple(conn.execute("SELECT step,data FROM receipts WHERE session_id=? ORDER BY step",
                                      (session_id,)).fetchall())
        rows = self._cached(("rows", session_id), load)
        return rows if rows is not _MISS else load(connection)

    def _indexed(self, conn, session_id):
        """The receipts digest the index is marked to be built on: the value this process holds, else the stored one."""
        held = self.held.marker_for(str(self.path), session_id) if self.held is not None else None
        if held is not None:
            return held
        row = conn.execute("SELECT digest FROM indexed WHERE session_id=?", (session_id,)).fetchone()
        return row[0] if row else None

    def flush_held(self) -> int:
        """Write the markers this process holds for this database (idle, close); a failure loses only markers."""
        if self.held is None or not any(key[0] == str(self.path) for key in list(self.held.markers)):
            return 0
        try:
            with closing(self._connect()) as conn, conn:
                conn.execute("BEGIN IMMEDIATE")
                return self._write_held(conn)
        except sqlite3.Error:
            return 0

    def _write_held(self, conn) -> int:
        taken = self.held.drain_markers(str(self.path)) if self.held is not None else []
        for session_id, digest in taken:
            conn.execute("INSERT OR REPLACE INTO indexed VALUES (?,?)", (session_id, digest))
        return len(taken)

    def _connect(self):
        check_request()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        try:
            _journal.apply_journal(conn, self.path, self.journal)    # before any transaction; never fails the request
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
                self._write_held(conn)    # rides this commit
                check_request()
                conn.commit()

    def authority_token(self, session_id: str):
        """Read exact scoped evidence without recreating lost receipt authority."""
        check_request()
        try:
            before = self.path.stat()
        except FileNotFoundError:
            return ("missing", None, None)
        rows = self._cached(("rows", session_id), lambda conn: tuple(conn.execute(
            "SELECT step,data FROM receipts WHERE session_id=? ORDER BY step", (session_id,)).fetchall()))
        if rows is _MISS:
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
        with closing(self._connect()) as conn:        # also makes sure the tables exist before any reader asks
            rows = self._rows(session_id, conn)
            indexed = self._indexed(conn, session_id)
        return rows, indexed

    def checkpoints(self, session_id: str, path: Path) -> tuple[int, ...]:
        """Offsets whose prefix digests a sync of ``path`` is going to ask for: every receipt's anchor, and the
        verified length of the ledger. A capture that hashes the file with a state at each of them makes those
        digests free; one that does not still gets them right, by re-reading."""
        def load(conn):
            return frozenset(row[0] for row in conn.execute(
                "SELECT json_extract(data,'$.prefix_length') FROM receipts WHERE session_id=?", (session_id,)))

        with closing(self._connect()) as conn:
            cached = self._cached(("marks", session_id), load)
            marks = set(cached if cached is not _MISS else load(conn))
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
            current = (rows, self._indexed(conn, session_id))
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
        if self.held is not None:
            # Same check, but the marker is held and written with the next commit of this database (see ``__init__``).
            with closing(self._connect()) as conn:
                rows = self._rows(session_id, conn)
                if self.rows_digest(rows) != digest:
                    raise ValueError("hook receipts changed before indexed marker")
                stored = conn.execute("SELECT digest FROM indexed WHERE session_id=?", (session_id,)).fetchone()
            if stored is not None and stored[0] == digest:
                self.held.drop_marker(str(self.path), session_id)
            else:
                self.held.hold_marker(str(self.path), session_id, digest)
            return
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = tuple(conn.execute("SELECT step,data FROM receipts WHERE session_id=? ORDER BY step", (session_id,)).fetchall())
            if self.rows_digest(rows) != digest:
                raise ValueError("hook receipts changed before indexed marker")
            marked = conn.execute("SELECT digest FROM indexed WHERE session_id=?", (session_id,)).fetchone()
            if marked is not None and marked[0] == digest:
                return  # already this digest: rewriting the identical row would only cost a synchronous commit
            with publication():
                conn.execute("INSERT OR REPLACE INTO indexed VALUES (?,?)", (session_id, digest))
                check_request()
                conn.commit()


# What an index of this transcript rests on, besides the transcript itself (see ``dependencies``).
DEPENDENCIES_VERSION = 1


def _row_digest(payload: str | None) -> str:
    return "" if payload is None else hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def dependencies(rows, uses, prior: dict | None = None) -> dict:
    """What the indexed events were derived from, to be stored with the cursor they belong to.

    The parse reads receipts in two ways only (see ``parse_session_chunks``): the receipt of the step of each result
    record it turned into events (``uses``), and the first receipt of the conversation, which decides whether an
    unlinked result predates the receipts. So an index stays exactly what a fresh parse would build for as long
    as neither changes. ``rows`` are the receipts the parse saw as ``(step, json)``; ``prior`` the dependencies
    of the events already indexed (an append adds to them; a full parse starts from none).
    """
    by_step = dict(rows)
    steps = dict(prior["steps"]) if prior else {}
    for step in uses:
        steps[step] = _row_digest(by_step.get(step))
    return {"v": DEPENDENCIES_VERSION, "first": min(by_step) if by_step else None, "steps": steps}


def affects_index(deps: dict | None, rows) -> bool:
    """May a fresh parse with ``rows`` differ from the indexed events ``deps`` describes? True when unsure.

    Differs only if the receipt of an indexed result step changed (added, removed or modified), or the first
    receipt moved past an indexed result: such a result is ``legacy_unlinked`` exactly when its step is below the
    first receipt, so only steps between the old and the new first receipt can flip. Missing or foreign
    dependencies (an index from before they were kept) answer True: the caller reparses, as it always did.
    """
    if not isinstance(deps, dict) or deps.get("v") != DEPENDENCIES_VERSION or not isinstance(deps.get("steps"), dict):
        return True
    by_step = dict(rows)
    steps = deps["steps"]
    for step, digest in steps.items():
        if _row_digest(by_step.get(step)) != digest:
            return True
    then, now = deps.get("first"), (min(by_step) if by_step else None)
    if then != now:
        moved = [first for first in (then, now) if first is not None]
        low, high = min(moved), (max(moved) if len(moved) == 2 else None)
        return any(step >= low and (high is None or step < high) for step in steps)
    return False


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
