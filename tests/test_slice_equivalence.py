"""The gate for slice commits (roundtable 8, P1/P7): a session synced in slices equals the same session synced in one piece.

For each of the five parsers a synthetic session (hundreds of records: tool call/result pairs, parse errors, unsupported
records, blank lines, invalid UTF-8, an unfinished last line) is indexed once without a deadline (one transaction, as
before) and once with randomized forced stops (``max_records`` caps of 1..N, a cap of exactly one record, byte caps that cut
the capture short), driven through ``sync_session``, ``audit_session`` and ``check_action`` alike, until the engine says it
is complete. The two indexes must hold the same rows. Where they cannot (the publication times, how recently the prefix was
verified in full, the commit log) the difference is asserted, not normalised away.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import pytest

from self_directing_mcp import source_blocks
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from slicing_support import (PROVIDERS, World, forced, integrity_ok, snapshot, sparse_rows)

BIG = 1 << 40   # a deadline so far away that only the forced stops (or the capture cap) cut a sync short


def expected_cursor_hash(path, offset, kind):
    """The cursor hash of the first ``offset`` bytes, recomputed from the file with nothing from the index."""
    data = Path(path).read_bytes()[:offset]
    if kind == "sha":
        return hashlib.sha256(data).hexdigest()
    size = source_blocks.BLOCK_SIZE
    full = len(data) // size
    digests = b"".join(hashlib.sha256(data[i * size:(i + 1) * size]).digest() for i in range(full))
    tail = hashlib.sha256(data[full * size:]).digest() if len(data) % size else None
    return source_blocks.BlockTable.build(offset, data.count(b"\n"), digests, tail).hash


def assert_prefix_invariants(world, engine, previous):
    """The cursor is a record boundary, its hash is the hash of exactly that prefix, no event lies past it."""
    row = engine.store._conn.execute("SELECT byte_offset,prefix_hash FROM session_cursors WHERE provider=? AND session_id=?",
                                     (world.provider, world.sid.lower() if world.provider in ("codex", "agy") else world.sid)
                                     ).fetchone()
    if row is None:
        assert previous == 0
        return 0
    offset, stored = row
    data = world.path.read_bytes()
    assert offset > 0 and data[offset - 1:offset] == b"\n", "the cursor is not the end of a record"
    kind = "sha" if world.provider == "agy" else "blocks"
    assert stored == expected_cursor_hash(world.path, offset, kind)
    last = engine.store._conn.execute("SELECT MAX(byte_start) FROM session_events WHERE provider=? AND session_id=?",
                                      (world.provider, world.sid.lower() if world.provider in ("codex", "agy") else world.sid)
                                      ).fetchone()[0]
    assert last is None or last < offset
    assert offset >= previous
    return offset


def drive(world, engine, rng, *, caps=(1, 1, 2, 3, 7, 19, 40), byte_caps=(None,), routes=("sync", "audit", "check"),
          limit=4000):
    """Call the engine with a fresh forced stop each time until nothing is left to index; returns what every call said."""
    seen, offset = [], 0
    for _ in range(limit):
        engine.slice_policy = forced(rng.choice(caps), cap_bytes=rng.choice(byte_caps))
        route, deadline = rng.choice(routes), time.monotonic() + BIG
        if route == "sync":
            info = engine.sync_session(**world.sync_args(), derived=rng.random() < 0.5, _deadline=deadline)
            coverage, sync = info["coverage"], info["sync"]
            assert info["ok"] is True
        else:
            if route == "audit":
                answer = engine.audit_session(world.sid, provider=world.provider, path=str(world.path), _deadline=deadline)
            else:
                answer = engine.check_action(world.sid, {"tool_name": "bash", "arguments": {"command": "echo hi"}},
                                             provider=world.provider, path=str(world.path),
                                             verify=rng.choice(["fast", "full"]), _deadline=deadline)
            coverage, sync = answer["coverage"], None
            if coverage.get("backlog_bytes", 0):
                # A backlog: never clean, never "clean as of the cursor", and the reason is spelled out.
                assert answer["verdict"] != "clean" and not coverage.get("judged_prefix")
                assert "sync_in_progress" in coverage["issues"] and coverage["complete"] is False
        offset = assert_prefix_invariants(world, engine, offset)
        seen.append((route, coverage.get("backlog_bytes", 0), sync))
        if not coverage.get("backlog_bytes", 0):
            return seen
    raise AssertionError("the sync never completed")


def normalized_audit(answer):
    answer = {k: v for k, v in answer.items() if k not in ("timings_ms", "integrity", "incremental")}
    answer["coverage"] = {k: v for k, v in answer["coverage"].items() if k not in ("synced_at", "backlog_bytes")}
    answer["execution"] = {k: v for k, v in answer["execution"].items() if k not in ("run_id", "elapsed_ms")}
    return answer


def assert_exercised(provider, snap):
    """The fixture really contains what the equivalence is about (an equal pair of empty things proves nothing)."""
    metas = [json.loads(row[5])["meta"] for row in snap.events]
    assert snap.metrics[1] >= 1                                      # parse errors are in there
    assert snap.metrics[2] >= 1 or len(snap.events) < 150            # and, in a session of some size, unsupported records
    if provider == "omp":
        starts = [m for m in metas if m.get("event_type") == "custom"]   # host events that refer to an earlier tool call
        assert sum(not m.get("unsupported_event") for m in starts) >= 5 and sum(bool(m.get("unsupported_event")) for m in starts) >= 3
    if provider == "agy":
        results = Counter(m.get("linkage") for m in metas if m.get("event_type") == "GENERIC")
        assert results["host_step_receipt"] >= 10 and results["native_log_omits_call_id"] >= 5
        assert sum(bool(m.get("restored_fields")) for m in metas) >= 3  # fields restored from the sibling full log
        assert snap.agy[1] and snap.agy[1][0][0]                         # receipt dependencies were stored


def assert_equal_indexes(world, one, sliced, *, calls):
    """Row for row: events (ids and data), cursor, block table, metrics, parser version, AGY dependencies, derived index."""
    a, b = snapshot(one, world.provider, world.sid), snapshot(sliced, world.provider, world.sid)
    assert len(a.events) > 20 and a.events == b.events
    assert_exercised(world.provider, a)
    assert a.cursor == b.cursor and a.cursor[1] > 0
    assert a.metrics == b.metrics and a.versions == b.versions
    # The block table: every column the source determines. ``full_verified_through`` (how far a full scan reached) is the one
    # column that may differ, and it must: a sliced sync verified in full what each slice's capture covered.
    assert a.blocks == b.blocks
    if world.provider == "agy":
        assert a.agy == b.agy   # receipts digest, receipt dependencies, evidence tallies
    # One generation (no slice rewrote the history), and the commit log is the chain of slices ending in the one-shot's row.
    assert a.generation == b.generation == 1
    assert b.last_commit == a.last_commit and len(a.commit_offsets) == 1
    assert b.commit_offsets == sorted(set(b.commit_offsets)) and len(b.commit_offsets) <= calls + 1
    assert sparse_rows(one, world.sid.lower() if world.provider in ("codex", "agy") else world.sid) == \
        sparse_rows(sliced, world.sid.lower() if world.provider in ("codex", "agy") else world.sid)
    assert integrity_ok(one.settings.index_dir) and integrity_ok(sliced.settings.index_dir)


def run_equivalence(tmp_path, monkeypatch, provider, seed, *, count, caps=(1, 1, 2, 3, 7, 19, 40), byte_caps=(None,),
                    routes=("sync", "audit", "check"), **world_options):
    world = World(tmp_path, monkeypatch, provider, seed=seed, count=count, **world_options)
    try:
        one, sliced = world.engine("one-shot"), world.engine("sliced")
        for engine in (one, sliced):
            world.copy_receipts(world.builder, engine)
            world.upsert_contracts(engine)
        info = one.sync_session(**world.sync_args(), derived=True)       # no deadline: the sync of before, one transaction
        assert "sync" not in info and info["ok"] is True
        seen = drive(world, sliced, random.Random(seed * 7919 + 1), caps=caps, byte_caps=byte_caps, routes=routes)
        stops = {sync["stopped_by"] for _route, _backlog, sync in seen if sync is not None}
        # The derived pass of the sliced index, then the comparison.
        sliced.slice_policy = None
        final = sliced.sync_session(**world.sync_args(), derived=True, _deadline=time.monotonic() + BIG)
        assert final["sync"]["state"] == "complete" and final["new_chunks"] == 0
        assert_equal_indexes(world, one, sliced, calls=len(seen))
        # The audit with two contracts: the same answer, from the same events.
        audit_one = one.audit_session(world.sid, provider=provider, path=str(world.path))
        audit_sliced = sliced.audit_session(world.sid, provider=provider, path=str(world.path), _deadline=time.monotonic() + BIG)
        assert not audit_sliced["coverage"].get("backlog_bytes")
        assert normalized_audit(audit_one) == normalized_audit(audit_sliced)
        assert {f["contract_id"] for f in audit_one["findings"]} == {"no-wipe", "ack"}
        return world, seen, stops, snapshot(one, provider, world.sid)
    finally:
        world.close()


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("seed", [1, 2])
def test_randomized_forced_stops_equal_the_one_shot_sync(tmp_path, monkeypatch, provider, seed):
    _world, seen, stops, _snapshot = run_equivalence(tmp_path, monkeypatch, provider, seed, count=110 if provider == "agy" else 240)
    assert len(seen) >= 10 and "record_cap" in stops    # many slices: the stops really cut the parse short


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_cap_of_exactly_one_record_per_slice(tmp_path, monkeypatch, provider):
    _world, seen, stops, _snapshot = run_equivalence(tmp_path, monkeypatch, provider, 5, count=45 if provider != "agy" else 30,
                                          caps=(1,), routes=("sync",))
    assert len(seen) >= 25 and stops == {"record_cap", "eof"}


@pytest.mark.parametrize("provider", PROVIDERS)
def test_byte_caps_that_cut_the_capture_short_equal_the_one_shot_sync(tmp_path, monkeypatch, provider):
    """The capture itself stops at a record boundary below ``cursor + cap``; the parse then reads all of it."""
    _world, seen, stops, _snapshot = run_equivalence(tmp_path, monkeypatch, provider, 3,
                                                     count=110 if provider == "agy" else 200, caps=(None,) * 3 + (11,),
                                                     byte_caps=(500, 1500, 4000), routes=("sync",))
    assert "slice_cap" in stops and len(seen) >= 6


@pytest.mark.parametrize("provider", ["codex", "omp", "claude", "grokbot"])
def test_slices_across_1_mib_block_boundaries_equal_the_one_shot_sync(tmp_path, monkeypatch, provider):
    """About 2.5 MB of records, so the block table has complete blocks and slices end inside later blocks."""
    world, seen, stops, _snapshot = run_equivalence(tmp_path, monkeypatch, provider, 9, count=380, fat=400, caps=(7, 40, 90, 200, None),
                                         byte_caps=(None, 150_000, 600_000))
    assert world.path.stat().st_size > 2 * source_blocks.BLOCK_SIZE and len(seen) >= 6


def test_the_omp_header_may_follow_a_title(tmp_path, monkeypatch):
    run_equivalence(tmp_path, monkeypatch, "omp", 4, count=200, title=True, caps=(2, 3, 9, 30))


def test_a_session_without_an_unfinished_last_line(tmp_path, monkeypatch):
    """No pending bytes at the end: the last slice's cursor is the end of the file."""
    world, seen, _stops, indexed = run_equivalence(tmp_path, monkeypatch, "codex", 6, count=150, partial=False,
                                                   routes=("sync",))
    assert world.path.stat().st_size == indexed.cursor[1]


def test_the_slices_of_a_claude_subagent_transcript_equal_the_one_shot_sync(tmp_path, monkeypatch):
    run_equivalence(tmp_path, monkeypatch, "claude", 8, count=200, subagent=True, caps=(1, 4, 11, 40))


@pytest.mark.xfail(strict=True, reason="Claude parser state across calls: the agent-id consistency check of a subagent file "
                                       "(_File.pinned_agent) starts again in every call, so a slice boundary hides it")
def test_known_limit_a_subagent_file_whose_records_name_different_agents(tmp_path, monkeypatch):
    """Reported, not patched (the parsers are not modified; the same holds for any append, hook or not).

    A Claude Code subagent file pins the ``agentId`` of its first record and rejects a later record that names another
    one ("agent id changed inside one transcript"). That memory lives in the parser call, so a record that opens a slice is
    judged without it. Real files carry one agent id (the one in the file name) on every record and are unaffected: the
    equivalence tests above pass for them. Only a file whose records disagree with each other differs, and then in one
    direction: the slice index keeps a record the one-shot index flags as a parse error."""
    from test_claude_hooks import STEM, SID as CLAUDE, assistant, human, sub_file, text, write
    for name in ("SELF_DIRECT_SESSION_PROVIDER",):
        monkeypatch.delenv(name, raising=False)
    root, records = tmp_path / "projects", []
    for number in range(12):
        first = human(f"question {number}", sub=True)
        first["agentId"] = "ffff0000ffff0000"        # the record names an agent that is not the one of the file
        second = assistant(text(f"answer {number}"), sub=True)       # this one names the file's agent
        records += [first, second]
    path, sid = write(sub_file(root), *records), f"{CLAUDE}_{STEM}"
    texts = {}
    for mode in ("one-shot", "sliced"):
        engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=tmp_path / mode, claude_projects_dir=root,
                                           openrouter_api_key_file=None))
        try:
            if mode == "one-shot":
                engine.sync_session(session_id=sid, provider="claude", path=str(path), embed=False)
            else:
                while True:
                    engine.slice_policy = forced(1)
                    info = engine.sync_session(session_id=sid, provider="claude", path=str(path), embed=False,
                                               _deadline=time.monotonic() + BIG)
                    if info["sync"]["state"] == "complete":
                        break
            texts[mode] = [json.loads(row[5])["text"] for row in snapshot(engine, "claude", sid).events]
        finally:
            engine.close()
    assert texts["one-shot"] == texts["sliced"]


@pytest.mark.parametrize("provider", ["codex", "omp", "claude", "grokbot"])
def test_a_session_that_grows_between_slices_equals_the_one_shot_sync_of_the_final_file(tmp_path, monkeypatch, provider):
    """Cold slices, then appends (one of them ending inside a record), more slices: the mixture of the two things the
    index does with a growing file, against a single sync of the finished file."""
    world = World(tmp_path, monkeypatch, provider, seed=12, count=260, partial=False)
    try:
        data = world.path.read_bytes()
        ends = [i + 1 for i, byte in enumerate(data) if byte == 0x0A]      # the end of every record
        first, second = ends[len(ends) // 3], ends[2 * len(ends) // 3] - 17           # the second cut is inside a record
        one, sliced = world.engine("one-shot"), world.engine("sliced")
        one.sync_session(**world.sync_args(), derived=False)
        expected = snapshot(one, provider, world.sid)
        rng = random.Random(21)
        for stage, size in enumerate((first, second, len(data))):
            world.path.write_bytes(data[:size])
            for _ in range(rng.randint(2, 5) if size < len(data) else 1000):
                sliced.slice_policy = forced(rng.choice([2, 5, 11]), cap_bytes=rng.choice([None, 1500, 6000]))
                info = sliced.sync_session(**world.sync_args(), derived=False, _deadline=time.monotonic() + BIG)
                if info["sync"]["state"] == "complete":
                    break
            if stage < 2:
                assert info["coverage"]["pending_bytes"] >= 0 and info["coverage"]["complete"] is False
        done = snapshot(sliced, provider, world.sid)
        no_mtime = lambda b: b[:2] + b[3:]       # noqa: E731  (this test rewrites the file itself)
        assert (done.events, done.cursor, no_mtime(done.blocks), done.metrics) == \
            (expected.events, expected.cursor, no_mtime(expected.blocks), expected.metrics)
        assert done.generation == expected.generation == 1
    finally:
        world.close()
