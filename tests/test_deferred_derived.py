"""A resident engine (the daemon's) holds the rows that carry no authority and writes them with a later commit.

What is held: the incremental monitor states, the "verified in full at T" stamp of a block table and the AGY "indexed"
marker. Why that may not change any answer is argued in ``index/deferred.py``; these tests are the argument's evidence.

* the answers: against a twin that evaluates every audit in full, a deferring engine answers identically after every
  append, contract edit, rewrite and version change, with the shadow comparison off and on (so a stale held row is a
  mismatch the strict conftest turns into a failure);
* the mechanics: held rows serve the next request, ride the next authority commit, are written by ``flush_derived`` and
  ``close``, are invisible to another process (which then rebuilds, correctly), and a row whose lineage moved on is
  dropped, never written;
* invalidation: a rewrite, a dropped monitor, a replaced block table and a changed evaluator or parser version each make
  the very next request ignore the row they invalidate.

The whole suite also runs in this mode: ``TSUKKOMI_TEST_DEFER_DERIVED=1 python -m pytest``.
"""
from __future__ import annotations

import json
import random
import sqlite3

import pytest

from self_directing_mcp import engine as engine_module
from self_directing_mcp.agy_receipts import ReceiptStore
from self_directing_mcp.audit import incremental as inc
from self_directing_mcp.codex import parse as codex_parse
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index import deferred
from self_directing_mcp.index.deferred import DerivedBuffer
from self_directing_mcp.index.ingest import SessionStore
from test_incremental_engine import RULES, Twin, codex_records, stable
from test_integrity_fast_path import make_world  # noqa: F401  (a fixture: a synthetic Codex session with a clock)
from test_reliability import SID, call, message, write


class DeferringTwin(Twin):
    """``fast`` holds its derived rows; ``full`` evaluates every audit over the whole session."""

    def __init__(self, tmp_path, provider, *, shadow_every=0, **paths):
        super().__init__(tmp_path, provider, **paths)
        self.fast.close()
        base = dict(_env_file=None, local_only=True, openrouter_api_key_file=None, **paths)
        self.fast = SelfDirectEngine(Settings(index_dir=tmp_path / "fast", audit_incremental=True,
                                              audit_shadow_every=shadow_every, **base), defer_derived=True)
        self.index = tmp_path / "fast"
        self.paths = paths

    def other_process(self):
        """A second engine on the same index, as another process would be: it sees only what was written."""
        base = dict(_env_file=None, local_only=True, openrouter_api_key_file=None, **self.paths)
        return SelfDirectEngine(Settings(index_dir=self.index, audit_incremental=True, audit_shadow_every=0, **base))


def make_twin(tmp_path, monkeypatch, shadow_every=0):
    for name in ("SELF_DIRECT_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl"
    write(path, [{"type": "session_meta", "payload": {"id": SID}}])
    return DeferringTwin(tmp_path, "codex", shadow_every=shadow_every, codex_sessions_dir=root), path


@pytest.fixture
def held(tmp_path, monkeypatch):
    twin, path = make_twin(tmp_path, monkeypatch)
    yield twin, path
    twin.close()


def stored_states(engine) -> int:
    return engine.store._conn.execute("SELECT COUNT(*) FROM audit_monitor_state").fetchone()[0]


def database_states(index) -> int:
    """What a second connection (another process) can read."""
    with sqlite3.connect(index / "meta.sqlite") as connection:
        return connection.execute("SELECT COUNT(*) FROM audit_monitor_state").fetchone()[0]


def held_generations(twin) -> set[int]:
    """The generations of the monitor states this engine is holding for the session."""
    return {row["generation"] for row in twin.fast.store.derived.states.get(("codex", SID), {}).values()}


def current_generation(twin) -> int:
    return twin.fast.store.audit_lineage(SID, "codex")["generation"]


def seeded(held, count=30, seed=1):
    twin, path = held
    rng = random.Random(seed)
    numbered = []
    write(path, [message("the user states a rule", "user")] + codex_records(rng, count, numbered), "a")
    twin.upsert(RULES)
    return twin, path, rng, numbered


# --- the mechanics -----------------------------------------------------------------------------------------------

def test_an_audit_holds_its_states_and_the_next_audit_uses_them_without_a_commit(held):
    twin, path, _rng, _numbered = seeded(held)
    first, _ = twin.audit(SID)
    assert first["incremental"]["state"] == "deferred"
    assert stored_states(twin.fast) == 0 and database_states(twin.index) == 0
    assert twin.fast.store.has_pending_derived()
    second, _ = twin.audit(SID)
    assert second["incremental"]["mode"] == "delta" and not second["incremental"]["rebuilt"]
    assert second["incremental"]["events_read"] == 0
    assert database_states(twin.index) == 0     # still nothing written: two audits, no commit for the cache


def test_the_next_authority_commit_carries_the_held_rows(held):
    twin, path, rng, numbered = seeded(held)
    twin.audit(SID)
    assert database_states(twin.index) == 0
    write(path, codex_records(rng, 3, numbered), "a")
    twin.fast.sync_session(session_id=SID, provider="codex", embed=False, derived=False)   # an ordinary append commit
    assert database_states(twin.index) > 0 and not twin.fast.store.has_pending_derived()
    fast, _ = twin.audit(SID)
    assert fast["incremental"]["mode"] == "delta" and not fast["incremental"]["rebuilt"]


def test_flush_derived_writes_held_rows_and_a_second_call_has_nothing_to_do(held):
    twin, _path, _rng, _numbered = seeded(held)
    twin.audit(SID)
    assert twin.fast.flush_derived() > 0
    assert database_states(twin.index) > 0 and not twin.fast.store.has_pending_derived()
    assert twin.fast.flush_derived() == 0


def test_close_writes_held_rows_so_a_restart_continues_from_them(held):
    twin, _path, _rng, _numbered = seeded(held)
    twin.audit(SID)
    assert database_states(twin.index) == 0
    twin.fast.close()
    assert database_states(twin.index) > 0
    successor = twin.other_process()
    try:
        answer = successor.audit_session(SID, provider="codex")
        assert answer["incremental"]["mode"] == "delta" and not answer["incremental"]["rebuilt"]
        assert stable(answer) == stable(twin.full.audit_session(SID, provider="codex"))
    finally:
        successor.close()


def test_another_process_sees_no_held_row_and_rebuilds_to_the_same_answer(held):
    twin, _path, _rng, _numbered = seeded(held)
    twin.audit(SID)
    other = twin.other_process()
    try:
        answer = other.audit_session(SID, provider="codex")
        assert set(answer["incremental"]["rebuilt"].values()) == {"no_state"}
        assert stable(answer) == stable(twin.full.audit_session(SID, provider="codex"))
    finally:
        other.close()


def test_a_flush_drops_a_row_whose_lineage_moved_on_instead_of_writing_it(held):
    twin, path, rng, numbered = seeded(held)
    twin.audit(SID)
    assert twin.fast.store.has_pending_derived()
    # Another process rewrites the session: a new generation, the old commit log is gone.
    lines = path.read_text().splitlines()
    lines[3] = json.dumps(call("deploy deploy tampered", "rewritten"))
    path.write_text("\n".join(lines) + "\n")
    other = twin.other_process()
    try:
        other.sync_session(session_id=SID, provider="codex", embed=False)
    finally:
        other.close()
    twin.fast.flush_derived()
    assert not twin.fast.store.has_pending_derived()
    generations = {row[0] for row in twin.fast.store._conn.execute("SELECT generation FROM audit_monitor_state")}
    current = twin.fast.store._conn.execute("SELECT generation FROM session_generations").fetchone()[0]
    assert generations <= {current}, (generations, current)


def test_a_full_buffer_is_written_without_waiting(held, monkeypatch):
    twin, _path, _rng, _numbered = seeded(held)
    monkeypatch.setattr(deferred, "MAX_ENTRIES", 2)
    twin.audit(SID)
    assert not twin.fast.store.has_pending_derived() and database_states(twin.index) > 0


def test_a_failed_flush_loses_cache_rows_never_the_commit_it_rides(held, monkeypatch):
    twin, path, rng, numbered = seeded(held)
    twin.audit(SID)
    real = SessionStore._trim_state_variants

    def broken(self, provider, session_id):
        raise sqlite3.OperationalError("synthetic failure while writing held rows")

    monkeypatch.setattr(SessionStore, "_trim_state_variants", broken)
    write(path, codex_records(rng, 3, numbered), "a")
    info = twin.fast.sync_session(session_id=SID, provider="codex", embed=False, derived=False)
    assert info["ok"] and info["new_chunks"] > 0                  # the events and the cursor were published
    monkeypatch.setattr(SessionStore, "_trim_state_variants", real)
    fast, _ = twin.audit(SID)                                      # the lost rows are simply rebuilt
    assert stable(fast) == stable(twin.full.audit_session(SID, provider="codex"))


def test_flushing_an_engine_that_defers_nothing_is_a_no_op(tmp_path):
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, openrouter_api_key_file=None,
                                       index_dir=tmp_path / "idx"), defer_derived=False)
    try:
        assert engine.flush_derived() == 0                          # not initialized yet
        engine.audit_status()
        assert engine.flush_derived() == 0 and engine.defer_derived is False
    finally:
        engine.close()


# --- the answers --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("shadow_every", [0, 1])
@pytest.mark.parametrize("seed", range(4))
def test_random_sessions_answer_like_the_full_evaluator_while_rows_are_held(tmp_path, monkeypatch, seed, shadow_every):
    twin, path = make_twin(tmp_path, monkeypatch, shadow_every)
    try:
        rng = random.Random(f"deferred/{seed}")
        numbered = []
        write(path, [message("the user states a rule", "user")] + codex_records(rng, 14, numbered), "a")
        twin.upsert(RULES)
        twin.audit(SID)
        for _ in range(10):
            write(path, codex_records(rng, rng.choice([1, 2, 5, 9]), numbered), "a")
            twin.audit(SID)
            if rng.random() < 0.5:
                twin.check(SID, {"tool_name": "shell", "arguments": {"command": "x deploy"}})
            if rng.random() < 0.3:
                twin.fast.flush_derived()                          # a flush in the middle changes nothing either
        modes = [r["mode"] for r in twin.reports if r]
        assert modes.count("delta") >= 6, modes
    finally:
        twin.close()


def test_the_contract_set_changes_while_rows_are_held(held):
    twin, path, rng, numbered = seeded(held, count=30)
    twin.audit(SID)
    write(path, codex_records(rng, 5, numbered), "a")
    twin.audit(SID)
    anchor = next(c.chunk_id for c in twin.fast.store.list_chunks(SID, "codex") if c.meta.get("role") == "user")
    twin.upsert([{**RULES[0], "regex": "deploy|push", "source_event_id": anchor}])   # a revision bump of a held monitor
    fast, _ = twin.audit(SID)
    assert set(fast["incremental"]["rebuilt"].values()) <= {"contract_revision", "no_state"}
    for engine in (twin.fast, twin.full):
        engine.revoke_contract("no-deploy", source_event_id=anchor)
    write(path, codex_records(rng, 7, numbered), "a")
    fast, _ = twin.audit(SID)
    assert [f["contract_id"] for f in fast["findings"]] == [r["id"] for r in RULES[1:]]
    fast, _ = twin.audit(SID)
    assert fast["incremental"]["mode"] == "delta" and not fast["incremental"]["rebuilt"]


def test_a_rewrite_is_never_judged_from_held_rows(held):
    twin, path, rng, numbered = seeded(held, count=40)
    twin.audit(SID)
    write(path, codex_records(rng, 4, numbered), "a")
    twin.audit(SID)
    assert twin.fast.store.has_pending_derived()
    lines = path.read_text().splitlines()
    lines[10] = json.dumps(call("rm -rf deploy deploy", "tampered"))
    path.write_text("\n".join(lines) + "\n")
    old_generation = current_generation(twin)
    fast, _ = twin.audit(SID)
    assert "source_changed" in fast["coverage"]["issues"] or fast["incremental"]["mode"] == "full"
    assert current_generation(twin) == old_generation + 1
    assert held_generations(twin) <= {current_generation(twin)}      # nothing of the old chain survived in memory
    write(path, codex_records(rng, 4, numbered), "a")
    twin.audit(SID)
    twin.check(SID)


def test_a_changed_evaluator_version_makes_the_next_request_ignore_held_rows(held, monkeypatch):
    twin, _path, _rng, _numbered = seeded(held)
    twin.audit(SID)
    monkeypatch.setattr(inc, "EVALUATOR_VERSION", inc.EVALUATOR_VERSION + 1)
    fast, _ = twin.audit(SID)
    assert set(fast["incremental"]["rebuilt"].values()) == {"evaluator_version"}, fast["incremental"]


def test_a_changed_parser_version_reindexes_and_drops_held_rows(held, monkeypatch):
    twin, path, rng, numbered = seeded(held)
    twin.audit(SID)
    assert twin.fast.store.has_pending_derived()
    monkeypatch.setattr(codex_parse, "PARSER_VERSION", codex_parse.PARSER_VERSION + 1)
    write(path, codex_records(rng, 2, numbered), "a")
    fast, _ = twin.audit(SID)
    assert fast["coverage"]["parser_version"] == codex_parse.PARSER_VERSION
    assert fast["incremental"]["mode"] == "full"                       # a reindex starts a new generation: no old state
    assert held_generations(twin) <= {current_generation(twin)}


def test_dropping_a_monitor_drops_its_held_states(held, monkeypatch):
    twin, _path, _rng, _numbered = seeded(held)
    twin.audit(SID)
    buffer = twin.fast.store.derived
    assert any(monitor == "no-deploy" for (_digest, monitor) in buffer.states[("codex", SID)])
    twin.fast.store.drop_audit_states(SID, "codex", ["no-deploy"])
    assert not any(monitor == "no-deploy" for (_digest, monitor) in buffer.states[("codex", SID)])
    twin.fast.store.record_audit_health(SID, "codex", inc.EVALUATOR_VERSION, ["tests-first"], "error", "synthetic")
    assert not any(monitor == "tests-first" for (_digest, monitor) in buffer.states[("codex", SID)])
    twin.audit(SID)


# --- verification stamps ------------------------------------------------------------------------------------------

@pytest.fixture
def deferring_engines(monkeypatch):
    original = engine_module.SelfDirectEngine.__init__

    def deferring(self, settings=None, **kwargs):
        kwargs.setdefault("defer_derived", True)
        original(self, settings, **kwargs)

    monkeypatch.setattr(engine_module.SelfDirectEngine, "__init__", deferring)


def database_stamp(world):
    with sqlite3.connect(world.engine.settings.index_dir / "meta.sqlite") as connection:
        return connection.execute("SELECT full_verified_at FROM source_blocks").fetchone()[0]


def test_a_full_verification_is_stamped_in_memory_and_written_later(make_world, deferring_engines):
    world = make_world()
    world.audit()                                  # first sync: the table is published with its stamp
    first = world.record().full_verified_at
    assert database_stamp(world) == first
    world.clock.advance(7)
    world.audit()                                  # nothing appended: the stamp alone moves
    stamped = world.record().full_verified_at
    assert stamped != first and stamped == world.clock().isoformat()
    assert database_stamp(world) == first          # not written yet ...
    assert world.check()["integrity"]["full_verified_at"] == stamped   # ... but this process uses it
    world.engine.flush_derived()
    assert database_stamp(world) == stamped


def test_a_stamp_belongs_to_the_table_it_verified(make_world, deferring_engines):
    world = make_world()
    world.audit()
    stale = world.record()
    world.append(call("echo more", "m1"))
    world.sync()                                   # the append replaces the table
    store = world.engine.store
    assert store.mark_full_verified(SID, "codex", stale, at="2026-10-15T00:00:00+00:00", through=0, dev=1, ino=2,
                                    size=3, mtime_ns=4) is False
    assert store.derived.stamps == {}
    assert world.record().full_verified_at != "2026-10-15T00:00:00+00:00"


def test_a_held_stamp_is_dropped_when_a_commit_replaces_the_table(make_world, deferring_engines):
    world = make_world()
    world.audit()
    world.clock.advance(5)
    world.audit()
    assert world.engine.store.derived.stamps
    world.append(call("echo more", "m2"))
    world.sync(verify="fast")                      # a fast check keeps the stamp it was given (it verified nothing)
    assert not world.engine.store.derived.stamps
    assert world.record().full_verified_at == world.clock().isoformat()   # carried into the new table by the draft


def test_a_held_stamp_is_not_applied_to_a_table_another_process_published(make_world, deferring_engines):
    world = make_world()
    world.audit()
    world.clock.advance(7)
    world.audit()
    stamped = world.record().full_verified_at
    assert world.engine.store.derived.stamps
    other = SelfDirectEngine(Settings(codex_sessions_dir=world.root, index_dir=world.engine.settings.index_dir,
                                      local_only=True, _env_file=None))
    try:
        world.append(call("echo from the other process", "o1"))
        other.sync_session(session_id=SID, embed=False, derived=False)       # publishes a new table
        record = world.record()                                               # read through this process's overlay
        assert record.full_verified_at != stamped, "the stamp of the replaced table was applied to the new one"
        assert record.table.covered == world.path.stat().st_size
        assert not world.engine.store.derived.stamps
    finally:
        other.close()


def test_a_held_stamp_does_not_let_another_process_skip_a_full_verification(make_world, deferring_engines):
    world = make_world()
    world.audit()
    world.clock.advance(7)
    world.audit()
    other = SelfDirectEngine(Settings(codex_sessions_dir=world.root, index_dir=world.engine.settings.index_dir,
                                      local_only=True, _env_file=None))
    try:
        other.ensure_ready()
        assert other.store.source_block_record(SID, "codex").full_verified_at != world.clock().isoformat()
    finally:
        other.close()


# --- AGY indexed marker -------------------------------------------------------------------------------------------

def put_receipt(store, session="agy-1", step=1):
    with store._connect() as connection:
        connection.execute("INSERT OR REPLACE INTO receipts VALUES (?,?,?)", (session, step, json.dumps({"receipt_id": "r"})))


def stored_marker(store, session="agy-1"):
    with store._connect() as connection:
        row = connection.execute("SELECT digest FROM indexed WHERE session_id=?", (session,)).fetchone()
    return row[0] if row else None


def test_the_indexed_marker_is_held_read_back_and_written_with_the_next_receipt(tmp_path):
    held = DerivedBuffer()
    store = ReceiptStore(tmp_path, held=held)
    put_receipt(store)
    rows, indexed = store.read_capture("agy-1")
    assert indexed is None
    digest = store.rows_digest(rows)
    store.mark_indexed("agy-1", digest)
    assert stored_marker(store) is None and store.read_capture("agy-1")[1] == digest    # held, and read back
    store.publish_bindings("agy-1", store.read_capture("agy-1"), rows)                   # consistent with the capture
    assert stored_marker(store) is None
    store.record("agy-1", tmp_path / "missing-transcript.jsonl", 2, {"name": "run_command", "args": {}}, "PreToolUse")
    assert stored_marker(store) == digest and not held.markers                            # rode the receipt's commit
    # The new receipt changed the digest: the stored marker now says "not indexed", which is true.
    assert store.read_capture("agy-1")[1] == digest and store.rows_digest(store.read_capture("agy-1")[0]) != digest


def test_a_marker_that_is_already_stored_is_not_held_and_not_rewritten(tmp_path):
    held = DerivedBuffer()
    store = ReceiptStore(tmp_path, held=held)
    put_receipt(store)
    digest = store.rows_digest(store.read_capture("agy-1")[0])
    store.mark_indexed("agy-1", digest)
    assert store.flush_held() == 1 and stored_marker(store) == digest
    store.mark_indexed("agy-1", digest)
    assert not held.markers


def test_a_marker_for_receipts_that_changed_is_refused_as_before(tmp_path):
    held = DerivedBuffer()
    store = ReceiptStore(tmp_path, held=held)
    put_receipt(store)
    stale = store.rows_digest(store.read_capture("agy-1")[0])
    put_receipt(store, step=2)
    with pytest.raises(ValueError, match="changed before indexed marker"):
        store.mark_indexed("agy-1", stale)
    assert not held.markers


def test_an_unchanged_marker_costs_no_commit_even_without_deferral(tmp_path):
    store = ReceiptStore(tmp_path)
    put_receipt(store)
    digest = store.rows_digest(store.read_capture("agy-1")[0])
    store.mark_indexed("agy-1", digest)
    before = store.path.stat()
    store.mark_indexed("agy-1", digest)
    after = store.path.stat()
    assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size) and stored_marker(store) == digest
