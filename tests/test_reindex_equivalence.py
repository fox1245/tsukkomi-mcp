"""The gate for the staged re-index (roundtable 8, PR3): a session re-indexed through the spool equals the same session
re-indexed in one piece.

For each parser a synthetic session (hundreds of records: tool call/result pairs, parse errors, unsupported records, blank lines,
invalid UTF-8, an unfinished last line) is indexed at an older parser version (or has one word changed, or, for AGY, one hook
receipt), and is then re-indexed twice from the same starting index: once with no deadline (one transaction, as before) and
once in many tiny slices, driven through ``sync_session``, ``audit_session`` and ``check_action`` alike. While the slices are
staged the index is byte for byte the old one and every answer says so; when the last one is in, the two indexes hold the same
rows. Where they cannot (publication times) the difference is asserted, not normalised away.
"""
from __future__ import annotations

import random
import time

import pytest

from reindex_support import (BIG, canonical, commits_one_shot, index_dump, reindex_of, seed_vectors, spool_files,
                             starting_indexes)
from slicing_support import PROVIDERS, World, forced
from test_slice_equivalence import assert_exercised, normalized_audit


def staged_calls(world, engine, before, rng, *, caps, routes=("sync", "audit", "check"), reason, limit=4000):
    """Call the engine with a fresh forced stop each time until the re-index is published. Until then the index is the old
    one, whatever route asked, and the answer says that a re-index is under way."""
    seen, through = [], 0
    for _ in range(limit):
        engine.slice_policy = forced(rng.choice(caps))
        route, deadline = rng.choice(routes), time.monotonic() + BIG
        if route == "sync":
            info = engine.sync_session(**world.sync_args(), derived=rng.random() < 0.5, _deadline=deadline)
            coverage, sync = info["coverage"], info["sync"]
        else:
            if route == "audit":
                answer = engine.audit_session(world.sid, provider=world.provider, path=str(world.path), _deadline=deadline)
            else:
                answer = engine.check_action(world.sid, {"tool_name": "bash", "arguments": {"command": "echo hi"}},
                                             provider=world.provider, path=str(world.path),
                                             verify=rng.choice(["fast", "full"]), _deadline=deadline)
            coverage, sync = answer["coverage"], None
        cursor = engine.store.cursor_record(world.sid.lower() if world.provider in ("codex", "agy") else world.sid,
                                            world.provider)
        published = cursor["synced_at"] != before["session_cursors"][0][5]     # every publication stamps the cursor anew
        waiting = {"sync_in_progress", "reindex_ready_to_commit"} & set(coverage["issues"])
        if not published:
            assert index_dump(engine) == before                            # the old index, to the byte
            assert waiting and coverage["complete"] is False
            assert reason_issue(reason) in coverage["issues"] or reason == "hook_receipts_changed"
            if sync is not None:
                assert sync["state"] == "partial" and sync["indexed_through"] == before["session_cursors"][0][3]
                staged = reindex_of(info)
                assert staged["state"] in ("staging", "ready_to_commit") and staged["staged_through"] >= through
                through = staged["staged_through"]
            else:
                assert answer["verdict"] != "clean" and not coverage.get("judged_prefix")
        seen.append((route, sync))
        if published:
            assert not waiting
            return seen
    raise AssertionError("the re-index was never published")


def reason_issue(reason):
    """The coverage issue that says the index in force is out of date."""
    return {"parser_updated": "parser_outdated", "source_changed": "source_changed"}.get(reason, "")


def run_equivalence(tmp_path, monkeypatch, provider, trigger, seed, *, count, caps=(3, 7, 19, 40), **world_options):
    world = World(tmp_path, monkeypatch, provider, seed=seed, count=count, **world_options)
    try:
        one, spooled = world.engine("one-shot"), world.engine("spooled")
        for engine in (one, spooled):
            world.copy_receipts(world.builder, engine)
            world.upsert_contracts(engine)
        reason = starting_indexes(world, provider, trigger, (one, spooled))
        seed_vectors(spooled)
        before = index_dump(spooled)
        assert index_dump(one)["session_events"] == before["session_events"]
        # The reference: the re-index of before, one transaction.
        info = commits_one_shot(one, world, derived=False)
        assert info["reindexed"] is True and info["reindex_reason"] == reason
        # The same re-index through the spool, in many small slices.
        seen = staged_calls(world, spooled, before, random.Random(seed * 7919 + 1), caps=caps, reason=reason)
        assert len(seen) >= 6 and not spool_files(spooled)
        stops = {sync["stopped_by"] for _route, sync in seen if sync is not None}
        assert "record_cap" in stops
        assert _differences(canonical(one, provider, world.sid), canonical(spooled, provider, world.sid)) == []
        # The derived indexes follow the same way, and the old vectors are still there.
        for engine in (one, spooled):
            engine.slice_policy = None
            done = engine.sync_session(**world.sync_args(), derived=True, _deadline=time.monotonic() + BIG)
            assert done["sync"]["state"] == "complete" and done["new_chunks"] == 0
        first, second = canonical(one, provider, world.sid), canonical(spooled, provider, world.sid)
        assert _differences(first, second) == []
        assert index_dump(spooled)["embedding_cache"] == before["embedding_cache"]
        assert_exercised(provider, _snapshot(one, provider, world))
        audit_one = one.audit_session(world.sid, provider=provider, path=str(world.path))
        audit_spooled = spooled.audit_session(world.sid, provider=provider, path=str(world.path), _deadline=time.monotonic() + BIG)
        assert not audit_spooled["coverage"].get("backlog_bytes")
        assert normalized_audit(audit_one) == normalized_audit(audit_spooled)
        assert {f["contract_id"] for f in audit_one["findings"]} == {"no-wipe", "ack"}
        return world, seen, first
    finally:
        world.close()


def _snapshot(engine, provider, world):
    from slicing_support import snapshot
    return snapshot(engine, provider, world.sid)


def _differences(first, second):
    """The parts of two canonical dumps that differ (none: the indexes hold the same rows)."""
    return [key for key in first if first[key] != second[key]]


@pytest.mark.parametrize("provider", ["codex", "omp", "claude", "agy"])
@pytest.mark.parametrize("seed", [1, 2])
def test_a_parser_upgrade_through_the_spool_equals_the_one_shot_reindex(tmp_path, monkeypatch, provider, seed):
    _world, seen, done = run_equivalence(tmp_path, monkeypatch, provider, "parser", seed,
                                         count=110 if provider == "agy" else 240)
    assert done["generation"] == 2 and len(done["commit_offsets"]) == 1 and done["pending_deletions"] == []


@pytest.mark.parametrize("provider", ["codex", "omp", "claude", "grokbot"])
def test_a_changed_source_through_the_spool_equals_the_one_shot_reindex(tmp_path, monkeypatch, provider):
    """One word changed in the middle: the stored hash no longer matches, one event has a new id (an obsolete one is
    journalled), and a re-index for ``source_changed`` is staged exactly like a parser upgrade. Grok Bot has no parser version,
    so this is its only re-index."""
    _world, seen, done = run_equivalence(tmp_path, monkeypatch, provider, "source", 3, count=240)
    assert len(done["pending_deletions"]) == 0     # the derived pass of the final sync has acknowledged them in both indexes
    assert done["generation"] == 2


def test_changed_hook_receipts_through_the_spool_equal_the_one_shot_reindex(tmp_path, monkeypatch):
    """AGY: a receipt of an indexed result changes after the fact, so the events derived from it are rebuilt from byte 0."""
    _world, seen, done = run_equivalence(tmp_path, monkeypatch, "agy", "receipts", 4, count=110)
    assert done["generation"] == 2


@pytest.mark.parametrize("provider", ["codex", "omp", "claude", "agy", "grokbot"])
def test_a_cap_of_exactly_one_record_per_slice(tmp_path, monkeypatch, provider):
    trigger = "source" if provider == "grokbot" else "parser"
    _world, seen, _done = run_equivalence(tmp_path, monkeypatch, provider, trigger, 5, count=45 if provider != "agy" else 30,
                                          caps=(1,))
    assert len(seen) >= 25


@pytest.mark.parametrize("provider", ["codex", "omp", "claude"])
def test_slices_across_1_mib_block_boundaries_equal_the_one_shot_reindex(tmp_path, monkeypatch, provider):
    """About 2.5 MB of records, so the block table has complete blocks and the staged range ends inside later blocks."""
    from self_directing_mcp import source_blocks
    world, seen, _done = run_equivalence(tmp_path, monkeypatch, provider, "parser", 9, count=380, fat=400,
                                         caps=(5, 15, 40, 90))
    assert world.path.stat().st_size > 2 * source_blocks.BLOCK_SIZE and len(seen) >= 6


def test_the_omp_header_may_follow_a_title(tmp_path, monkeypatch):
    run_equivalence(tmp_path, monkeypatch, "omp", "parser", 4, count=200, title=True, caps=(2, 3, 9, 30))


def test_a_session_without_an_unfinished_last_line(tmp_path, monkeypatch):
    """No pending bytes at the end: the last slice's range ends at the end of the file."""
    world, _seen, done = run_equivalence(tmp_path, monkeypatch, "codex", "parser", 6, count=150, partial=False)
    assert world.path.stat().st_size == done["cursor"][1]


def test_a_claude_subagent_transcript_through_the_spool_equals_the_one_shot_reindex(tmp_path, monkeypatch):
    run_equivalence(tmp_path, monkeypatch, "claude", "parser", 8, count=200, subagent=True, caps=(1, 4, 11, 40))
