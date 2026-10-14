"""AGY hooks cost what was appended, not the size of the transcript.

The mechanisms (``agy_receipts.Ledger``, receipt dependencies, capture checkpoints) and the guards that make them
safe: a cache is used only after its prefix digest was recomputed from the file in the same pass that hashed it,
every doubt falls back to the old whole-file parse, and a reparse still happens whenever a receipt change can alter
an event that is already indexed. ``test_agy_append_equivalence.py`` proves the end state equals a fresh reindex;
these tests pin the individual decisions and the costs.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3

import pytest

from self_directing_mcp import agy_hooks, agy_receipts
from self_directing_mcp.agy import RESULT_TYPES
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.session_events import SourceChanged, capture_source, source_scope

SID = "55555555-5555-4555-8555-555555555555"
CALL = {"name": "run_command", "args": {"CommandLine": "npm test", "Cwd": "/synthetic"}}


def user(step):
    return {"step_index": step, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE",
            "content": f"<USER_REQUEST>task {step}</USER_REQUEST>"}


def planner(step, command="npm test"):
    return {"step_index": step, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE", "content": "",
            "tool_calls": [{"name": "run_command", "args": {"CommandLine": json.dumps(command),
                                                             "Cwd": json.dumps("/synthetic")}}]}


def result(step, status="DONE", code=0, **extra):
    return {"step_index": step, "source": "MODEL", "type": "GENERIC", "status": status,
            "content": f"Created At: t\nCompleted At: t\n\nThe command exited with code {code}.\nOutput:\nstep {step}\n",
            **extra}


def write(path, records):
    path.write_bytes(b"".join(json.dumps(r, separators=(",", ":")).encode() + b"\n" for r in records))


def append(path, *records):
    with path.open("ab") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")).encode() + b"\n")


class Session:
    def __init__(self, tmp_path):
        self.app = tmp_path / "app"
        self.path = self.app / "brain" / SID / ".system_generated" / "logs" / "transcript.jsonl"
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b"")
        self.index = tmp_path / "index"
        self.engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=self.index,
                                                agy_app_data_dirs=[self.app], openrouter_api_key_file=None))
        self.receipts = agy_receipts.ReceiptStore(self.index)
        self.engine.ensure_ready()
        self.database = self.engine.store.db_path

    def payload(self, step, call=CALL, error=None):
        body = {"conversationId": SID, "transcriptPath": str(self.path), "workspacePaths": ["/synthetic"],
                "stepIdx": step, "toolCall": call}
        if error is not None:
            body["error"] = error
        return body

    def pre(self, step, call=CALL):
        self.engine.record_agy_hook(self.payload(step, call), "PreToolUse")

    def post(self, step, call=CALL, error=""):
        self.engine.record_agy_hook(self.payload(step, call, error), "PostToolUse")

    def sync(self):
        return self.engine.sync_session(session_id=SID, provider="agy", path=str(self.path), embed=False)

    def row(self, step):
        return json.loads(dict(self.receipts.read_capture(SID)[0])[step])

    def ledger_rows(self):
        with sqlite3.connect(self.receipts.path) as conn:
            return (conn.execute("SELECT * FROM scan_state").fetchall(),
                    conn.execute("SELECT * FROM scan_results ORDER BY step").fetchall())

    def linkages(self):
        return {c.meta["step_index"]: c.meta["linkage"] for c in self.engine.store.list_chunks(SID, "agy")
                if c.kind == "tool_result"}


@pytest.fixture
def session(tmp_path):
    world = Session(tmp_path)
    yield world
    world.engine.close()


@pytest.fixture
def scans(monkeypatch):
    """Where each parse of the transcript started (``_scan`` is the ledger's only parse)."""
    starts = []
    real = agy_receipts._scan

    def spy(path, start=0):
        starts.append(start)
        return real(path, start)

    monkeypatch.setattr(agy_receipts, "_scan", spy)
    return starts


def blocks(session, count):
    """A transcript of ``count`` completed tool calls with a receipt for each, as hooks leave it."""
    records, step = [user(0)], 1
    for number in range(count):
        records += [planner(step, f"echo {number}")]
        write(session.path, records)
        call = {"name": "run_command", "args": {"CommandLine": f"echo {number}", "Cwd": "/synthetic"}}
        session.pre(step + 1, call)
        records += [result(step + 1)]
        write(session.path, records)
        session.post(step + 1, call)
        step += 2
    return step


# --- the hook side: what a receipt reads ---------------------------------------------------------------------

def test_a_receipt_parses_only_what_was_appended(session, scans):
    step = blocks(session, 6)
    size = session.path.stat().st_size
    scans.clear()
    append(session.path, planner(step))
    session.pre(step + 1)
    assert scans and min(scans) > 0 and min(scans) <= size  # from the verified length of the ledger, never byte 0
    row = session.row(step + 1)
    assert row["prefix_length"] == session.path.stat().st_size
    assert row["prefix_hash"] == hashlib.sha256(session.path.read_bytes()).hexdigest()
    assert row["preexisting_result"] is False


def test_the_ledger_follows_post_and_finds_the_result_it_appended_before(session, scans):
    step = blocks(session, 3)
    append(session.path, planner(step))
    session.pre(step + 1)
    append(session.path, result(step + 1))
    scans.clear()
    session.post(step + 1)
    assert scans and min(scans) > 0
    row = session.row(step + 1)
    assert row["paired"] is True and row["result_hash"] == agy_receipts.result_digest(result(step + 1))
    assert hashlib.sha256(session.path.read_bytes()[:row["prefix_length"]]).hexdigest() == row["prefix_hash"]


def test_a_changed_prefix_is_not_trusted(session, scans):
    step = blocks(session, 4)
    append(session.path, planner(step))
    session.pre(step + 1)
    data = bytearray(session.path.read_bytes())
    at = data.index(b"echo 1")
    data[at:at + 6] = b"echo 9"  # same length, different bytes, early in the file
    session.path.write_bytes(bytes(data))
    scans.clear()
    append(session.path, result(step + 1))
    session.post(step + 1)
    assert 0 in scans  # the ledger's digest no longer matched: everything is parsed again
    assert session.row(step + 1)["result_hash"] == agy_receipts.result_digest(result(step + 1))


def test_a_replaced_file_is_not_trusted(session, scans, tmp_path):
    step = blocks(session, 3)
    session.pre(step + 1)
    copy = tmp_path / "copy"
    copy.write_bytes(session.path.read_bytes())
    os.replace(copy, session.path)  # same bytes, another inode
    scans.clear()
    session.pre(step + 2)
    assert 0 in scans


def test_a_hole_below_the_stored_step_rebuilds_instead_of_merging(session, scans):
    write(session.path, [user(0), planner(1), result(2)])
    session.pre(2)
    append(session.path, planner(3), result(4), result(5, status="RUNNING"))
    session.pre(5)
    scans.clear()
    append(session.path, result(3))  # a completed result numbered below the highest step seen: only a full parse knows
    session.post(5)
    assert 0 in scans
    state, results = session.ledger_rows()
    assert [step for _, _, step, _ in results] == [2, 3, 4]


@pytest.mark.parametrize("known", ["stored", "tail"])
def test_two_completed_results_for_one_step_stay_an_error_and_store_nothing(session, known):
    write(session.path, [user(0), planner(1), result(2)])
    session.pre(2)
    before = session.ledger_rows()
    append(session.path, planner(3), *([result(2)] if known == "stored" else [result(4), result(4)]))
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        session.pre(4)
    assert session.ledger_rows() == before
    assert 4 not in dict(session.receipts.read_capture(SID)[0])


def test_an_unreadable_line_fails_the_receipt_as_it_always_did(session):
    write(session.path, [user(0), planner(1)])
    session.pre(2)
    before, ledger = session.receipts.read_capture(SID), session.ledger_rows()
    with session.path.open("ab") as stream:
        stream.write(b"this is not json\n")
    with pytest.raises(ValueError):
        session.pre(2)
    assert session.receipts.read_capture(SID) == before and session.ledger_rows() == ledger


def test_a_record_without_a_step_number_refuses_to_anchor_beyond_it_but_not_before_it(session):
    write(session.path, [user(0), planner(1), result(2)])
    append(session.path, {"source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE", "content": "no step"},
           planner(5))
    with pytest.raises(ValueError, match="unknown transcript layout"):
        session.pre(6)  # beyond every step: the whole layout must be readable
    session.pre(2)  # the first record numbered 2 comes before the odd one, exactly as the whole-file walk found it
    assert session.row(2)["prefix_length"] == len(b"".join(
        json.dumps(r, separators=(",", ":")).encode() + b"\n" for r in [user(0), planner(1)]))


def test_a_step_the_transcript_already_reached_anchors_where_the_old_walk_did(session):
    step = blocks(session, 3)  # steps 0 user, 1 planner, 2 result, 3 planner, 4 result, 5 planner, 6 result
    session.pre(4)  # inside the transcript: the position of the first record numbered 4 matters
    row = session.row(4)
    lines = session.path.read_bytes().splitlines(keepends=True)
    cut = sum(len(line) for line in lines[:4])  # records numbered 0, 1, 2 and 3 precede the first one numbered 4
    assert row["prefix_length"] == cut
    assert row["prefix_hash"] == hashlib.sha256(session.path.read_bytes()[:cut]).hexdigest()
    assert row["preexisting_result"] is True and step == 7


def test_a_ledger_update_is_skipped_when_another_writer_got_there_first(session):
    write(session.path, [user(0), planner(1)])
    session.pre(2)
    stale = None
    with sqlite3.connect(session.receipts.path) as conn:
        stale = agy_receipts.Ledger.read(conn, SID, session.path.resolve(), (2,))
    append(session.path, result(2))
    observation = capture_source(session.path, checkpoints=(stale.checkpoint(),))
    with source_scope(observation):
        stale.advance(observation)
    session.post(2)  # a concurrent writer persists first
    state_then = session.ledger_rows()
    with sqlite3.connect(session.receipts.path) as conn:
        stale.persist(conn)
    assert session.ledger_rows() == state_then


# --- a cheap observation cannot create what a full check rejects -----------------------------------------------

def test_an_anchor_is_checked_against_the_bytes_of_the_file_not_a_block_table(session):
    write(session.path, [user(0), planner(1)])
    session.pre(2)
    append(session.path, result(2))
    session.post(2)
    row = session.row(2)
    record = result(2)
    start = session.path.read_bytes().rindex(b'{"step_index":2')
    honest = capture_source(session.path, sha=False, blocks=True)  # no whole-file digest: only block digests
    with source_scope(honest):
        assert agy_receipts.linked_receipt({2: row}, 2, session.path, start, result_record=record) == row
    data = bytearray(session.path.read_bytes())
    data[3] ^= 1  # a byte of the anchored prefix changes
    session.path.write_bytes(bytes(data))
    tampered = capture_source(session.path, sha=False, blocks=True)
    assert tampered.prefix_digest(row["prefix_length"]) != row["prefix_hash"]  # read from the bytes, not remembered
    with source_scope(tampered):
        assert agy_receipts.linked_receipt({2: row}, 2, session.path, start, result_record=record) is None
    assert agy_receipts.linked_receipt({2: row}, 2, session.path, start, result_record=record) is None


# --- the sync side ---------------------------------------------------------------------------------------------

def test_a_hook_cycle_never_reparses_the_transcript(session, scans, monkeypatch):
    session.engine.upsert_contracts([{"id": "no-wipe", "type": "must_not", "scope": "tool_call", "provider": "agy",
                                      "regex": r"rm\s+-rf"}])
    step = blocks(session, 4)
    info = session.sync()
    cursor = info["byte_offset"]
    from self_directing_mcp import agy as agy_module
    parses = []
    real = agy_module.parse_session_chunks
    monkeypatch.setattr(agy_module, "parse_session_chunks",
                        lambda *args, **kwargs: parses.append(kwargs.get("start_byte")) or real(*args, **kwargs))
    scans.clear()
    for number in range(3):
        call = {"name": "run_command", "args": {"CommandLine": f"echo late {number}", "Cwd": "/synthetic"}}
        append(session.path, planner(step, f"echo late {number}"))
        assert isinstance(agy_hooks.run_hook("PreToolUse", session.payload(step + 1, call), engine=session.engine), dict)
        append(session.path, result(step + 1))
        assert agy_hooks.run_hook("PostToolUse", session.payload(step + 1, call, ""), engine=session.engine) == {}
        step += 2
    assert parses and all(start >= cursor > 0 for start in parses)  # the chunk parser began at the cursor each time
    assert scans and min(scans) > 0  # and so did the ledger's
    info = session.sync()
    assert info["reindexed"] is False and info["reindex_reason"] is None
    assert info["new_chunks"] == 0 and "unread_bytes" not in info["coverage"]["issues"]
    assert set(session.linkages().values()) == {"host_step_receipt"}


def test_the_stored_receipt_digest_follows_the_receipts_without_a_reparse(session):
    step = blocks(session, 2)
    session.sync()
    append(session.path, planner(step))
    session.pre(step + 1)
    info = session.sync()
    assert info["reindexed"] is False
    coverage = info["coverage"]
    assert "receipts_not_indexed" not in session.engine.audit_status(SID, provider="agy")["session"]["issues"]
    assert coverage["receipt_digest"] == session.receipts.authority_token(SID)[2]


def test_a_post_that_pairs_a_result_already_indexed_reparses_and_links_it(session):
    write(session.path, [user(0), planner(1)])
    session.pre(2)
    append(session.path, result(2))
    info = session.sync()  # the result is indexed while its receipt is still unpaired
    assert session.linkages() == {2: "native_log_omits_call_id"} and info["reindexed"] is False
    session.post(2)
    info = session.sync()
    assert info["reindexed"] is True and info["reindex_reason"] == "hook_receipts_changed"
    assert session.linkages() == {2: "host_step_receipt"}


@pytest.mark.parametrize("step,reparsed", [(1, True), (9, False)])
def test_a_first_receipt_changes_the_index_only_for_results_below_or_above_it(session, step, reparsed):
    write(session.path, [user(0), planner(1), result(2), planner(3), result(4)])
    assert session.sync()["new_chunks"] > 0
    assert set(session.linkages().values()) == {"legacy_unlinked"}
    session.pre(step)  # results 2 and 4 are indexed; a receipt for step 1 predates them, one for step 9 does not
    info = session.sync()
    assert info["reindexed"] is reparsed
    expected = "native_log_omits_call_id" if reparsed else "legacy_unlinked"
    assert set(session.linkages().values()) == {expected}


def test_an_index_that_never_stored_dependencies_reparses_once_and_then_stops(session):
    step = blocks(session, 2)
    session.sync()
    with sqlite3.connect(session.database) as conn:  # an index written before dependencies existed
        conn.execute("DELETE FROM session_receipt_deps")
    append(session.path, planner(step))
    session.pre(step + 1)
    info = session.sync()
    assert info["reindexed"] is True and info["reindex_reason"] == "hook_receipts_changed"
    append(session.path, result(step + 1))
    session.post(step + 1)
    assert session.sync()["reindexed"] is False


def test_dependencies_written_by_another_publication_are_not_used(session):
    step = blocks(session, 2)
    session.sync()
    with sqlite3.connect(session.database) as conn:  # an older process republished the cursor
        conn.execute("UPDATE session_receipt_deps SET synced_at = 'another publication'")
    append(session.path, planner(step))
    session.pre(step + 1)
    assert session.sync()["reindex_reason"] == "hook_receipts_changed"


def test_a_receipt_of_another_file_does_not_make_a_sync_read_the_transcript(session, scans):
    write(session.path, [user(0), planner(1)])
    session.pre(2)
    other = session.path.with_name("transcript_full.jsonl")
    write(other, [user(0), planner(1), result(2)])
    session.engine.record_agy_hook({**session.payload(2), "transcriptPath": str(other)}, "PreToolUse")
    session.sync()
    session.sync()
    scans.clear()
    session.sync()
    assert scans == []  # nothing is unbound for this file: no scan at all


# --- units -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fields,bound", [
    ({"result_hash": "ab12", "z": 1}, True),
    ({"result_hash": "", "z": 1}, False),
    ({"result_hash": None, "z": 1}, False),
    ({"z": 1}, False),
    ({"action": '{"result_hash": "ab"}', "z": 1}, False),       # the words inside a value are not the key
    ({"action": 'x\\"result_hash": "ab"'}, False),
    ({"action": "ends with a backslash \\", "result_hash": "ab"}, True),
])
def test_the_bound_check_never_calls_an_unbound_receipt_bound(fields, bound):
    text = json.dumps(fields, sort_keys=True)
    assert agy_receipts._is_bound(text) is bound
    assert bool(json.loads(text).get("result_hash")) is bound  # exactly the decoded meaning


@pytest.mark.parametrize("data,found", [
    (b"plain ascii", False), ("caf\u00e9 and \u2603 are valid UTF-8".encode(), False), (b"caf\xe9", True), (b"\xff", True),
    (b"\x80 at the start", True), (b'an escaped lone surrogate stays text: "\\ud800"', False), ("\ud7ff\ue000".encode(), False),
])
def test_undecodable_bytes_are_found_exactly_as_the_character_loop_found_them(data, found):
    from self_directing_mcp.agy import has_undecodable
    raw = data.decode("utf-8", errors="surrogateescape")
    assert has_undecodable(raw) is found
    assert has_undecodable(raw) is any(0xDC80 <= ord(char) <= 0xDCFF for char in raw)


def test_receipts_decode_when_used():
    rows = [(3, json.dumps({"n": 3})), (1, json.dumps({"n": 1}))]
    receipts = agy_receipts._Receipts(rows)
    assert len(receipts) == 2 and min(receipts) == 1 and receipts.get(2) is None and receipts[3] == {"n": 3}
    assert receipts._decoded == {3: {"n": 3}}  # only what was asked for


def _rows(**by_step):
    return sorted((int(name[1:]), json.dumps({"receipt": value})) for name, value in by_step.items())


@pytest.mark.parametrize("then,indexed,now,affects", [
    (_rows(s4="a", s7="b"), {4, 9}, _rows(s4="a", s7="b"), False),                    # nothing changed
    (_rows(s4="a", s7="b"), {4, 9}, _rows(s4="changed", s7="b"), True),               # an indexed result's receipt
    (_rows(s4="a", s7="b"), {4, 9}, _rows(s7="b"), True),                             # ... removed
    (_rows(s4="a", s7="b"), {4, 9}, _rows(s4="a", s7="b", s9="new"), True),           # a receipt for result 9 appears
    (_rows(s4="a", s7="b"), {4, 9}, _rows(s4="a", s7="b", s12="new"), False),         # a step with no indexed result
    (_rows(s6="a"), {4, 9}, _rows(s2="x", s6="a"), True),                             # first receipt 6 -> 2: result 4
    (_rows(s6="a"), {9}, _rows(s2="x", s6="a"), False),                               # ... but no result in [2, 6)
    ([], {4}, _rows(s2="x"), True),                                                   # the first receipt appears
    ([], {4}, _rows(s9="x"), False),                                                  # ... above every indexed result
    (_rows(s2="x", s6="a"), {4}, _rows(s6="a"), True),                                # the first receipt goes away
])
def test_what_an_index_depends_on_is_exactly_what_can_change_it(then, indexed, now, affects):
    deps = agy_receipts.dependencies(then, indexed)
    assert agy_receipts.affects_index(deps, now) is affects


def test_missing_or_foreign_dependencies_mean_reparse():
    assert agy_receipts.affects_index(None, []) is True
    assert agy_receipts.affects_index({"v": 99, "first": None, "steps": {}}, []) is True
    assert agy_receipts.affects_index({"v": 1, "first": None, "steps": {}}, []) is False


def test_an_old_receipts_database_gains_the_ledger_tables_and_keeps_working(tmp_path):
    path = tmp_path / "agy-receipts.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE receipts (session_id TEXT, step INTEGER, data TEXT NOT NULL, PRIMARY KEY(session_id,step))")
        conn.execute("CREATE TABLE indexed (session_id TEXT PRIMARY KEY, digest TEXT NOT NULL)")
        conn.execute("INSERT INTO receipts VALUES (?,?,?)", (SID, 2, json.dumps({"receipt_id": "r", "path": "/x"})))
    store = agy_receipts.ReceiptStore(tmp_path)
    assert dict(store.read_capture(SID)[0]).keys() == {2}
    with sqlite3.connect(path) as conn:
        assert {row[0] for row in conn.execute("SELECT name FROM sqlite_master")} >= {"scan_state", "scan_results"}


# --- cost: the whole file is read a constant number of times -----------------------------------------------------

def _big_session(session, count):
    """``count`` linked results (one receipt each) in a transcript of a few megabytes, recorded the way hooks do."""
    filler = "synthetic harmless output " * 300
    records, rows, digest, step = [user(0)], [], hashlib.sha256(), 1
    data = b""
    for number in range(count):
        call = json.dumps({"name": "run_command", "args": {"CommandLine": f"echo {number}", "Cwd": "/synthetic"}},
                          sort_keys=True, separators=(",", ":"))
        plan = json.dumps(planner(step, f"echo {number}"), separators=(",", ":")).encode() + b"\n"
        outcome = result(step + 1)
        outcome["content"] += filler
        body = json.dumps(outcome, separators=(",", ":")).encode() + b"\n"
        data += plan
        rows.append((step + 1, call, len(data), hashlib.sha256(data).hexdigest(), agy_receipts.result_digest(outcome)))
        data += body
        step += 2
    first = json.dumps(user(0), separators=(",", ":")).encode() + b"\n"
    offset = len(first)
    session.path.write_bytes(first + data)
    store = agy_receipts.ReceiptStore(session.index)
    store._connect().close()
    with sqlite3.connect(store.path) as conn:
        for number, (result_step, call, length, _digest, result_hash) in enumerate(rows):
            length += offset
            prefix = hashlib.sha256((first + data)[:length]).hexdigest()
            conn.execute("INSERT OR REPLACE INTO receipts VALUES (?,?,?)", (SID, result_step, json.dumps({
                "receipt_id": f"id-{number}", "path": str(session.path.resolve()), "action": call, "prefix_length": length,
                "prefix_hash": prefix, "paired": True, "step_index": result_step, "session_id": SID,
                "preexisting_result": False, "post_error": "", "post_action": call,
                "post_path": str(session.path.resolve()), "result_hash": result_hash}, sort_keys=True)))
    return step


def test_a_full_parse_checks_every_receipt_anchor_without_rereading_the_file(session, hash_meter):
    _big_session(session, 250)
    size = session.path.stat().st_size
    hash_meter.lap()
    info = session.sync()
    linked = [link for link in session.linkages().values() if link == "host_step_receipt"]
    assert len(linked) == 250 and info["new_chunks"] > 0
    # Before: one re-read of the prefix per linked result, about 125 passes over the file. Now the file is hashed by
    # the capture and verified once more as the parse consumes it; what remains is content (the digest of each result
    # record and of each event), not the prefix again.
    assert hash_meter.lap() / size < 5.5


def test_a_hook_hashes_the_file_a_constant_number_of_times(session, hash_meter):
    step = _big_session(session, 250)
    session.engine.upsert_contracts([{"id": "no-wipe", "type": "must_not", "scope": "tool_call", "provider": "agy",
                                      "regex": r"rm\s+-rf"}])
    session.sync()

    def cycle(number):
        nonlocal step
        call = {"name": "run_command", "args": {"CommandLine": f"echo next {number}", "Cwd": "/synthetic"}}
        append(session.path, planner(step, f"echo next {number}"))
        hash_meter.lap()
        agy_hooks.run_hook("PreToolUse", session.payload(step + 1, call), engine=session.engine)
        pre = hash_meter.lap() / session.path.stat().st_size
        append(session.path, result(step + 1))
        hash_meter.lap()
        agy_hooks.run_hook("PostToolUse", session.payload(step + 1, call, ""), engine=session.engine)
        post = hash_meter.lap() / session.path.stat().st_size
        step += 2
        return pre, post

    cycle(0)  # the first check of a contract builds its audit state once
    pre, post = cycle(1)
    # One pass for the receipt (its anchor and the ledger's check), one for the sync's own capture, and small windows
    # (the old code: 250 more, one per linked result, plus a parse of the whole file).
    assert pre < 3.2 and post < 3.2, (pre, post)
    assert session.sync()["reindexed"] is False
