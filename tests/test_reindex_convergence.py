"""Repeating a call that cannot finish a re-index makes progress, and the last call publishes it (roundtable 8, PR3).

A session of about 150,000 events, re-indexed after a parser upgrade with per-call budgets of a few seconds. Before the spool a
call that could not parse AND commit it inside its budget rolled back and the next one started from byte 0; now every call keeps
its slice, ``reindex.staged_through`` strictly increases, the sync is published in at most ``ceil(M / q) + 2`` calls (M the size
of the source, q the smallest slice that was not the last), and the index equals the one-shot re-index.

What it cannot do is also tested: one transaction publishes the staged rows, so a call whose time left cannot hold it (18
microseconds a row, 1.5 times that to be safe) only reports ``ready_to_commit``; an explicit ``sync_session`` with its larger
budget publishes it.

Speed. These tests must say the same on a laptop and on a sanitizer build that is four times slower or more, so none of them
has a budget in seconds that was written down on one machine. The scripted-clock test does not look at the wall clock (the hard
deadline of its calls is an hour away). The real-clock tests take their budgets from what the one-shot re-index of this very
session cost here (``Big.one_shot``) and use the commit rate measured there as the prediction (``Big.per_row``; the production
constant is a fixed 18 microseconds, right for the machine it was measured on). A call that the hard deadline stopped all the
same is a hiccup of the machine, not of the code: ``converge`` counts it as a lost response (what the call spooled or committed
stays) and tolerates no more than ``LOST_MAX`` of them.
"""
from __future__ import annotations

import contextlib
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from reindex_support import canonical, old_parser, reindex_of, spool_files
from self_directing_mcp import journal
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index import ingest
from self_directing_mcp.index.ingest import SlicePolicy
from self_directing_mcp.request_control import RequestStopped
from slicing_support import HARD, Tick, World, integrity_ok, scripted_deadline

LOST_MAX = 2                        # calls the hard deadline may stop within one test
TOLERANCE = (1.15, 0.5)             # a call returns inside budget * 1.15 + 0.5 s (the stretch between two looks at the clock)


@dataclass
class Big:
    world: World
    base: Path
    expected: dict          # the index the re-index must produce
    one_shot: float         # seconds the one-shot re-index took here (parse and commit)
    per_row: float          # seconds its commit transaction took here, per row

    @property
    def rows(self) -> int:
        return len(self.expected["events"])


@contextlib.contextmanager
def commit_stopwatch():
    """The seconds the commit transactions of the block take, measured where they run (``SessionStore._commit_spool``)."""
    class Watch:
        seconds = 0.0

    watch, original = Watch(), ingest.SessionStore._commit_spool

    def timed(self, *args, **kwargs):
        started = time.monotonic()
        try:
            return original(self, *args, **kwargs)
        finally:
            watch.seconds += time.monotonic() - started

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ingest.SessionStore, "_commit_spool", timed)
        yield watch


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
        # What the re-index must produce, and what it costs on this machine: the one-shot re-index of a copy.
        reference = engine_on(world, base / "index-reference", base / "index-old")
        with commit_stopwatch() as commit:
            started = time.monotonic()
            reference.sync_session(**world.sync_args(), derived=False)
            one_shot = time.monotonic() - started
        expected = canonical(reference, "codex", world.sid)
        reference.close()
        world.engines.remove(reference)
        shutil.rmtree(base / "index-reference")
        assert len(expected["events"]) > 140_000 and 0 < commit.seconds < one_shot
        yield Big(world, base, expected, one_shot, commit.seconds / len(expected["events"]))
        world.close()


def engine_on(world, index, template):
    shutil.rmtree(index, ignore_errors=True)
    journal.copy_index_dir(template, index)
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, openrouter_api_key_file=None, index_dir=index,
                                       **world.paths))
    world.engines.append(engine)
    return engine


def converge(world, engine, budget, *, policy, limit=80):
    """Repeat the same call, whose hard deadline is ``budget`` seconds away, until the sync is complete or only a commit that
    does not fit is left. Returns what each call that answered returned, as (seconds, sync, reindex), and how many calls the
    hard deadline stopped (``RequestStopped``: no answer, but the slices they spooled or committed are kept)."""
    calls, previous, lost = [], -1, 0
    for _ in range(limit):
        engine.slice_policy = policy
        started = time.monotonic()
        try:
            info = engine.sync_session(**world.sync_args(), derived=False, _deadline=started + budget)
        except RequestStopped as stopped:
            assert stopped.reason == "request_deadline"
            lost += 1
            assert lost <= LOST_MAX, f"{lost} of {len(calls) + lost} calls were stopped by their deadline of {budget:.1f} s"
            continue
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
    return calls, lost


def quantum(calls, size):
    """q: the smallest slice that was not the last, in bytes of source."""
    through = [0] + [staged["staged_through"] for _elapsed, _sync, staged in calls]
    slices = [b - a for a, b in zip(through, through[1:]) if b > a]
    return min(slices[:-1]) if len(slices) > 1 else size


def inside(budget):
    return budget * TOLERANCE[0] + TOLERANCE[1]


def test_a_150k_event_reindex_converges_in_few_calls_with_a_scripted_clock(big, monkeypatch):
    """Time moves a fixed step per look (64 records), so the slices have the same size on every machine. The wall clock is not
    consulted: the hard deadline of every call is an hour away, and ``scripted_deadline`` leaves the soft deadline of a slice
    where a request of 8 seconds has it."""
    world, base, expected = big.world, big.base, big.expected
    scripted_deadline(monkeypatch, 8.0)
    engine = engine_on(world, base / "index-scripted", base / "index-old")
    size = world.path.stat().st_size
    calls, lost = converge(world, engine, HARD, policy=SlicePolicy(clock=Tick(0.011)))
    elapsed = [call[0] for call in calls]
    assert lost == 0
    assert calls[-1][1]["state"] == "complete" and reindex_of({"sync": calls[-1][1]})["state"] == "committed"
    q = quantum(calls, size)
    assert q >= 1 << 20                                                # at least the minimum progress of a slice
    assert 5 <= len(calls) <= math.ceil(size / q) + 2
    assert canonical(engine, "codex", world.sid) == expected and not spool_files(engine)
    assert integrity_ok(engine.settings.index_dir)
    print(f"\nscripted clock, 8 s: {len(calls)} calls for {size / 1e6:.1f} MB, q = {q / 1e6:.1f} MB, "
          f"bound {math.ceil(size / q) + 2}, slowest call {max(elapsed):.2f} s")
    engine.close()
    world.engines.remove(engine)


def test_the_same_reindex_on_the_real_clock_with_a_budget_of_about_the_one_shot_time(big, monkeypatch):
    """The budget is 1.2 times what the one-shot re-index took here (8 seconds on the machine this was written on), or twice
    the predicted commit if that is more; the prediction is the commit rate of this machine."""
    world, base, expected = big.world, big.base, big.expected
    monkeypatch.setattr(ingest, "COMMIT_SECONDS_PER_ROW", big.per_row)
    predicted = ingest.COMMIT_SECONDS_PER_ROW * ingest.COMMIT_SAFETY * big.rows
    budget = max(1.2 * big.one_shot, 2.2 * predicted)
    engine = engine_on(world, base / "index-real", base / "index-old")
    size = world.path.stat().st_size
    calls, lost = converge(world, engine, budget, policy=None)
    elapsed = [call[0] for call in calls]
    assert calls[-1][1]["state"] == "complete"
    q = quantum(calls, size)
    assert len(calls) <= math.ceil(size / q) + 2 + lost
    assert all(e < inside(budget) for e in elapsed)                    # inside the budget, with a stated tolerance
    assert canonical(engine, "codex", world.sid) == expected and not spool_files(engine)
    print(f"\nreal clock, one-shot {big.one_shot:.1f} s, budget {budget:.1f} s: {len(calls)} calls "
          f"({', '.join(f'{e:.1f}' for e in elapsed)} s, {lost} lost) for {size / 1e6:.1f} MB, q = {q / 1e6:.1f} MB, "
          f"bound {math.ceil(size / q) + 2 + lost}; states {[staged['state'] for _e, _s, staged in calls]}")
    engine.close()
    world.engines.remove(engine)


def test_with_a_budget_under_the_commit_the_commit_is_the_wall_and_an_explicit_sync_gets_past_it(big, monkeypatch):
    """The commit of 150,000 rows does not fit a budget of half the one-shot time (4 seconds on the machine this was written
    on): a call stages everything and stops at ``ready_to_commit`` (and says so, call after call); a ``sync_session`` of three
    times the one-shot time publishes it. The prediction is set to three times the budget, so that the wall stands wherever the
    commit is fast, and the explicit call fits it wherever the commit is slow."""
    world, base, expected = big.world, big.base, big.expected
    budget = 0.5 * big.one_shot
    monkeypatch.setattr(ingest, "COMMIT_SECONDS_PER_ROW", 3.0 * budget / (ingest.COMMIT_SAFETY * big.rows))
    engine = engine_on(world, base / "index-wall", base / "index-old")
    size = world.path.stat().st_size
    calls, lost = converge(world, engine, budget, policy=None)
    waiting = calls[-1][2]
    assert waiting["state"] == "ready_to_commit" and waiting["staged_through"] == waiting["target_bytes"]
    assert waiting["rows"] > 140_000 and spool_files(engine)
    assert ingest.COMMIT_SECONDS_PER_ROW * ingest.COMMIT_SAFETY * waiting["rows"] > 0.9 * budget       # why it waits
    assert all(call[0] < inside(budget) for call in calls)
    again = engine.sync_session(**world.sync_args(), derived=False, _deadline=time.monotonic() + budget)
    assert reindex_of(again)["state"] == "ready_to_commit"           # the same call again: no progress to make, and it says so
    started = time.monotonic()
    done = engine.sync_session(**world.sync_args(), derived=False, _deadline=started + 6.0 * budget)
    committed = time.monotonic() - started
    assert reindex_of(done)["state"] == "committed" and done["sync"]["state"] == "complete" and done["reindexed"] is True
    assert canonical(engine, "codex", world.sid) == expected and not spool_files(engine)
    print(f"\nreal clock, budget {budget:.1f} s: {len(calls)} calls to ready_to_commit ({waiting['rows']} rows, {lost} lost), "
          f"then a {6.0 * budget:.0f} s sync_session published it in {committed:.1f} s "
          f"(this machine's commit rate predicts {big.per_row * big.rows:.1f} s)")
    engine.close()
    world.engines.remove(engine)
