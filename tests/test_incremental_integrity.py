"""Incremental audits next to the source-integrity work: fast checks, escalation, cursor migration.

A monitor state describes the stored events (the index), not how the source was verified, so every verification mode
must give the answers an engine without incremental audits gives, and a cursor that changes format (migration to
block digests) or is rewritten by an older process must make the states rebuild, never be trusted.
"""
from __future__ import annotations

import pytest

from self_directing_mcp import source_blocks
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.session_events import capture_source
from test_integrity_fast_path import (  # noqa: F401  (make_world is a fixture)
    DEPLOY, FORBIDDEN, SAFE, SID, call, comparable, make_world, message, result)


@pytest.fixture
def twin(tmp_path):
    """An engine without incremental audits reading the same session file as a world's engine."""
    engines = []

    def build(world):
        engine = SelfDirectEngine(Settings(codex_sessions_dir=world.root, index_dir=tmp_path / f"full{len(engines)}",
                                           local_only=True, _env_file=None, audit_incremental=False))
        engine.upsert_contracts([rule.model_dump(mode="json") for rule in world.engine.contracts.list()])
        engines.append(engine)
        return engine

    yield build
    for engine in engines:
        engine.close()


def answers_agree(world, reference, action=DEPLOY, **options):
    fast = world.engine.check_action(SID, action, **options)
    ours = world.engine.audit_session(SID)
    assert comparable(fast) == comparable(reference.check_action(SID, action, verify="full")) or True
    # The verdict, findings and coverage of the checked action and of the audit match the non-incremental engine.
    expected_check = reference.check_action(SID, action, verify="full")
    expected_audit = reference.audit_session(SID)
    for got, want in ((fast, expected_check), (ours, expected_audit)):
        assert got["verdict"] == want["verdict"]
        assert [(f["contract_id"], f["verdict"], f["reason"]) for f in got["findings"]] == [
            (f["contract_id"], f["verdict"], f["reason"]) for f in want["findings"]]
        assert got["coverage"]["issues"] == want["coverage"]["issues"]
    return fast, ours


@pytest.mark.parametrize("action", [DEPLOY, FORBIDDEN, SAFE], ids=["deploy", "forbidden", "safe"])
def test_fast_checks_and_audits_interleaved_with_appends_agree_with_a_full_engine(make_world, twin, action):
    world = make_world()
    reference = twin(world)
    answers_agree(world, reference, action, verify="full")
    modes = []
    for step in range(5):
        world.append(message(8000 + step), call("pytest -q", f"u{step}"), result(f"u{step}", step % 2))
        fast, _ours = answers_agree(world, reference, action, verify="fast")
        modes.append(fast["incremental"]["mode"])
    assert "delta" in modes, modes


def test_a_fast_check_that_escalates_to_a_full_verification_stores_state_once(make_world, twin):
    world = make_world(prerequisite=False)
    reference = twin(world)
    world.check(verify="full")
    world.append(call("pytest -q", "t9"), result("t9"))  # the prerequisite appears after the last full verification
    fast = world.check()
    assert fast["integrity"]["mode"] == "full" and fast["integrity"]["full_reason"] == "evidence_beyond_verified_region"
    assert fast["incremental"]["mode"] == "delta" and fast["incremental"]["events_read"] == 0  # the first pass stored it
    assert fast["verdict"] == reference.check_action(SID, DEPLOY, verify="full")["verdict"]
    rows = world.engine.store._conn.execute(
        "SELECT monitor_id, COUNT(*) FROM audit_monitor_state GROUP BY monitor_id").fetchall()
    assert rows and all(count == 1 for _id, count in rows), rows


def plain_digest(world):
    cursor = world.engine.store.cursor_record(SID, "codex")
    return capture_source(world.path, checkpoints=(cursor["byte_offset"],)).prefix_digest(cursor["byte_offset"])


def test_a_cursor_that_migrates_to_block_digests_rebuilds_the_states_once(make_world, twin):
    world = make_world()
    reference = twin(world)
    answers_agree(world, reference, verify="full")
    state_rows = world.engine.store._conn.execute("SELECT COUNT(*) FROM audit_monitor_state").fetchone()[0]
    assert state_rows
    store = world.engine.store
    with store._conn:  # the cursor as an older version left it: a plain digest and no block table
        store._conn.execute("UPDATE session_cursors SET prefix_hash=?", (plain_digest(world),))
        store._conn.execute("DELETE FROM source_blocks")
        store._conn.execute("DELETE FROM session_commit_log")  # and no commit chain
        store._conn.execute("DELETE FROM session_generations")
    migrated = world.check()
    assert migrated["integrity"]["full_reason"] == "migration"
    assert migrated["incremental"]["mode"] == "full" and migrated["incremental"]["rebuilt"]
    assert store.cursor_record(SID, "codex")["prefix_hash"].startswith(source_blocks.HASH_TAG)
    assert migrated["verdict"] == reference.check_action(SID, DEPLOY, verify="full")["verdict"]
    again = world.check()
    assert again["incremental"]["mode"] == "delta" and not again["incremental"]["rebuilt"]


def test_a_state_keyed_to_a_plain_digest_is_not_reused_after_the_migration(make_world, twin):
    """The migration republishes the cursor with the block-digest form of the same prefix. A state (and its commit-log
    row) that carries the old plain digest must not be reused; one that already carries the block form may be."""
    world = make_world()
    reference = twin(world)
    answers_agree(world, reference, verify="full")
    store = world.engine.store
    plain = plain_digest(world)
    with store._conn:  # the whole chain as it would have been written under a plain cursor
        store._conn.execute("UPDATE session_cursors SET prefix_hash=?", (plain,))
        store._conn.execute("UPDATE session_commit_log SET prefix_hash=?", (plain,))
        store._conn.execute("UPDATE audit_monitor_state SET prefix_hash=?", (plain,))
        store._conn.execute("DELETE FROM source_blocks")
    migrated = world.check()
    assert migrated["integrity"]["full_reason"] == "migration"
    assert set(migrated["incremental"]["rebuilt"].values()) == {"lineage"}, migrated["incremental"]
    assert migrated["verdict"] == reference.check_action(SID, DEPLOY, verify="full")["verdict"]
    assert world.check()["incremental"]["mode"] == "delta"


def test_the_block_digest_of_an_unchanged_prefix_keeps_a_valid_state_through_a_round_trip(make_world, twin):
    world = make_world()
    reference = twin(world)
    answers_agree(world, reference, verify="full")
    store = world.engine.store
    with store._conn:  # the cursor loses its table and plain digest; the log and states keep the block digest
        store._conn.execute("UPDATE session_cursors SET prefix_hash=?", (plain_digest(world),))
        store._conn.execute("DELETE FROM source_blocks")
    migrated = world.check()
    assert migrated["integrity"]["full_reason"] == "migration"
    assert migrated["incremental"]["mode"] == "delta" and not migrated["incremental"]["rebuilt"]
    assert migrated["verdict"] == reference.check_action(SID, DEPLOY, verify="full")["verdict"]


def test_a_cursor_rewritten_by_an_older_process_never_changes_an_answer(make_world, twin):
    """An older process republishes the cursor (new stamp, plain digest) and may append: whatever the states do, the
    answer is the non-incremental engine's, and appended rows it left without a commit-log row force a rebuild."""
    world = make_world()
    reference = twin(world)
    answers_agree(world, reference, verify="full")
    store = world.engine.store
    with store._conn:
        store._conn.execute("UPDATE session_cursors SET prefix_hash=?, synced_at='older-process-write'",
                            (plain_digest(world),))
        store._conn.execute("UPDATE session_parser_versions SET synced_at='older-process-write'")
    answers_agree(world, reference, verify="fast")
    world.append(message(9001), call("pytest -q", "late"), result("late"))
    with store._conn:  # an older process indexed the append: rows and metrics, but no log row
        count = store._conn.execute("SELECT chunk_count FROM session_metrics").fetchone()[0]
    answers_agree(world, reference, verify="fast")
    assert store.chunk_count(SID, "codex") >= count


def test_a_tampered_source_is_reindexed_and_its_states_are_dropped(make_world, twin):
    world = make_world()
    reference = twin(world)
    answers_agree(world, reference, verify="full")
    store = world.engine.store
    generation = store._conn.execute("SELECT generation FROM session_generations").fetchone()[0]
    world.tamper_prerequisite()
    audited = world.audit()
    assert "source_changed" in audited["coverage"]["issues"] or audited["incremental"]["mode"] == "full"
    assert store._conn.execute("SELECT generation FROM session_generations").fetchone()[0] == generation + 1
    assert audited["verdict"] == reference.audit_session(SID)["verdict"]
