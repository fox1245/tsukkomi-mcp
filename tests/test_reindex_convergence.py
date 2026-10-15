"""Repeating a call that cannot finish a re-index makes progress, and the last call publishes it (roundtable 8, PR3).

A session of about 150,000 events, re-indexed after a parser upgrade with per-call budgets of 4 to 8 seconds. Before the spool a
call that could not parse AND commit it inside its budget rolled back and the next one started from byte 0; now every call keeps
its slice, ``reindex.staged_through`` strictly increases, the sync is published in at most ``ceil(M / q) + 2`` calls (M the size
of the source, q the smallest slice that was not the last), and the index equals the one-shot re-index.

What it cannot do is also tested: one transaction publishes the staged rows, so a call whose time left cannot hold it (18
microseconds a row, 1.5 times that to be safe) only reports ``ready_to_commit``; an explicit ``sync_session`` with its larger
budget publishes it.
"""
from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

import pytest

from reindex_support import BIG, canonical, old_parser, reindex_of, spool_files
from self_directing_mcp import journal
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index import ingest
from self_directing_mcp.index.ingest import SlicePolicy
from slicing_support import Tick, World, integrity_ok


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    """One synthetic session of about 150,000 events indexed at the previous parser version; each test copies that index."""
    base = tmp_path_factory.mktemp("reindex-convergence")
    with pytest.MonkeyPatch.context() as monkeypatch:
        world = World(base, monkeypatch, "codex", seed=3, count=125_000, partial=False)
        old = world.engine("old")
        with old_parser("codex"):
            old.sync_session(**world.sync_args(), derived=False)
        old.close()
        world.engines.remove(old)
        # What the re-index must produce: the one-shot re-index of a copy.
        reference = engine_on(world, base / "index-reference", base / "index-old")
        reference.sync_session(**world.sync_args(), derived=False)
        expected = canonical(reference, "codex", world.sid)
        reference.close()
        world.engines.remove(reference)
        shutil.rmtree(base / "index-reference")
        assert len(expected["events"]) > 140_000
        yield world, base, expected
        world.close()


def engine_on(world, index, template):
    shutil.rmtree(index, ignore_errors=True)
    journal.copy_index_dir(template, index)
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, openrouter_api_key_file=None, index_dir=index,
                                       **world.paths))
    world.engines.append(engine)
    return engine


def converge(world, engine, budget, *, policy, limit=80):
    """Repeat the same call with the same budget until the sync is complete; returns what each call answered and how long it took."""
    calls, previous = [], -1
    for number in range(1, limit + 1):
        engine.slice_policy = policy
        started = time.monotonic()
        info = engine.sync_session(**world.sync_args(), derived=False, _deadline=started + budget)
        elapsed = time.monotonic() - started
        sync, staged = info["sync"], reindex_of(info)
        calls.append((elapsed, sync, staged))
        if sync["state"] == "complete":
            break
        if staged["state"] == "ready_to_commit" and len(calls) > 1 and calls[-2][2]["state"] == "ready_to_commit":
            break                                             # it will not get any further with this budget
        if staged["state"] != "ready_to_commit":
            assert staged["staged_through"] > previous        # strictly further every call, until it waits for its commit
        previous = staged["staged_through"]
    return calls


def quantum(calls, size):
    """q: the smallest slice that was not the last, in bytes of source."""
    through = [0] + [staged["staged_through"] for _elapsed, _sync, staged in calls]
    slices = [b - a for a, b in zip(through, through[1:]) if b > a]
    return min(slices[:-1]) if len(slices) > 1 else size


def test_a_150k_event_reindex_converges_in_few_calls_with_a_scripted_clock(big, monkeypatch):
    """Time moves a fixed step per look (64 records), so the slices have the same size on every machine."""
    world, base, expected = big
    monkeypatch.setattr(ingest, "COMMIT_SECONDS_PER_ROW", 1e-9)        # this test is about the slices, not the commit wall
    engine = engine_on(world, base / "index-scripted", base / "index-old")
    size = world.path.stat().st_size
    calls = converge(world, engine, 8.0, policy=SlicePolicy(clock=Tick(0.011)))
    elapsed = [call[0] for call in calls]
    assert calls[-1][1]["state"] == "complete" and reindex_of({"sync": calls[-1][1]})["state"] == "committed"
    q = quantum(calls, size)
    assert q >= 1 << 20                                                # at least the minimum progress of a slice
    assert 5 <= len(calls) <= math.ceil(size / q) + 2
    assert all(e < 8.0 + 0.5 for e in elapsed[:-1])                    # every staging call came back inside its budget
    assert canonical(engine, "codex", world.sid) == expected and not spool_files(engine)
    assert integrity_ok(engine.settings.index_dir)
    print(f"\nscripted clock, 8 s: {len(calls)} calls for {size / 1e6:.1f} MB, q = {q / 1e6:.1f} MB, "
          f"bound {math.ceil(size / q) + 2}, slowest call {max(elapsed):.2f} s")
    engine.close()
    world.engines.remove(engine)


def test_the_same_reindex_on_the_real_clock_with_an_8_second_budget(big):
    world, base, expected = big
    engine = engine_on(world, base / "index-real8", base / "index-old")
    size = world.path.stat().st_size
    calls = converge(world, engine, 8.0, policy=None)
    elapsed = [call[0] for call in calls]
    assert calls[-1][1]["state"] == "complete"
    q = quantum(calls, size)
    assert len(calls) <= math.ceil(size / q) + 2
    assert all(e < 8.0 + 0.5 for e in elapsed)                         # inside the budget, with a stated tolerance
    assert canonical(engine, "codex", world.sid) == expected and not spool_files(engine)
    print(f"\nreal clock, 8 s: {len(calls)} calls ({', '.join(f'{e:.1f}' for e in elapsed)} s) for {size / 1e6:.1f} MB, "
          f"q = {q / 1e6:.1f} MB, bound {math.ceil(size / q) + 2}; states "
          f"{[staged['state'] for _e, _s, staged in calls]}")
    engine.close()
    world.engines.remove(engine)


def test_with_a_4_second_budget_the_commit_is_the_wall_and_an_explicit_sync_gets_past_it(big):
    """150,000 rows are 4 seconds of commit with the margin: a call of 4 seconds stages everything and stops at
    ``ready_to_commit`` (and says so, call after call); a ``sync_session`` of 45 seconds publishes it."""
    world, base, expected = big
    engine = engine_on(world, base / "index-real4", base / "index-old")
    size = world.path.stat().st_size
    calls = converge(world, engine, 4.0, policy=None)
    waiting = calls[-1][2]
    assert waiting["state"] == "ready_to_commit" and waiting["staged_through"] == waiting["target_bytes"]
    assert waiting["rows"] > 140_000 and spool_files(engine)
    assert ingest.COMMIT_SECONDS_PER_ROW * ingest.COMMIT_SAFETY * waiting["rows"] > 0.9 * 4.0       # why it waits
    assert all(call[0] < 4.0 + 0.5 for call in calls)
    again = engine.sync_session(**world.sync_args(), derived=False, _deadline=time.monotonic() + 4.0)
    assert reindex_of(again)["state"] == "ready_to_commit"           # the same call again: no progress to make, and it says so
    started = time.monotonic()
    done = engine.sync_session(**world.sync_args(), derived=False, _deadline=started + 45.0)
    committed = time.monotonic() - started
    assert reindex_of(done)["state"] == "committed" and done["sync"]["state"] == "complete" and done["reindexed"] is True
    assert canonical(engine, "codex", world.sid) == expected and not spool_files(engine)
    print(f"\nreal clock, 4 s: {len(calls)} calls to ready_to_commit ({waiting['rows']} rows), then a 45 s sync_session "
          f"published it in {committed:.1f} s (predicted {ingest.COMMIT_SECONDS_PER_ROW * waiting['rows']:.1f} s)")
    engine.close()
    world.engines.remove(engine)
