"""Slice commits under failure (roundtable 8, P7): a stop at ANY cooperative check, and a process killed at any moment.

* Fault injection. A request is stopped at its N-th ``check_request`` call, for every N from 1 to the number of calls a
  sliced sync makes. After each stop the index must be a valid earlier state: ``PRAGMA integrity_check`` ok, the cursor
  either unchanged or the end of a complete record whose hash is that of exactly that prefix, every earlier event
  intact and none past the cursor, the metrics agreeing with the rows. Then the sync is finished without faults and
  compared with the one-shot index.
* Crash. A subprocess slices a session and is killed with SIGKILL at several moments (while a commit is in flight among
  them); the index must verify and a later sync must complete and equal the one-shot index.
* A stop right after a commit. The request that runs out of time (or is cancelled) at the first check after a slice was
  committed loses its answer and nothing else: the slice stays, and the next call goes on from the cursor.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from self_directing_mcp import journal
from self_directing_mcp.codex import parse as codex_parser
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index import ingest
from self_directing_mcp.request_control import RequestControl, RequestStopped, request_scope
from slicing_support import PROVIDERS, World, forced, integrity_ok, snapshot
from test_slice_convergence import Tripwire
from test_slice_equivalence import BIG, expected_cursor_hash

ROOT = Path(__file__).resolve().parents[1]
CAPS = (3, 5, 7)


class Stopper(RequestControl):
    """A request with a far deadline (so a sync may slice) that is stopped from its ``stop_at``-th cooperative check on.

    The stop is sticky, like a deadline that passed or a cancellation: every later check raises as well. (A single raise
    is not a stop a request can see: some code, like the fallback to the sibling full log of an AGY transcript, treats an
    ``OSError`` as "unavailable", and ``RequestStopped`` is one; it only matters because the next check raises again.)"""

    def __init__(self, stop_at=None):
        super().__init__(deadline=time.monotonic() + BIG)
        self.calls, self.stop_at = 0, stop_at

    def _check(self):
        self.calls += 1
        if self.stop_at is not None and self.calls >= self.stop_at:
            raise RequestStopped("request_cancelled")
        super()._check()


def split_session(world):
    """The session as an index that already holds its first part (a previous index to protect) and the rest to append."""
    data = world.path.read_bytes()
    cut = data.index(b"\n", len(data) // 2) + 1
    world.path.write_bytes(data[:cut])
    return data, cut


def sync_all(world, engine, caps, stopper=None, limit=500):
    """Slice-sync until complete. With a ``stopper`` (a RequestControl) every call of the sequence is one request scope, so
    its check counter runs through all slices. Returns the number of calls."""
    def step(number, **kwargs):
        engine.slice_policy = forced(caps[number % len(caps)])
        return engine.sync_session(**world.sync_args(), derived=False, **kwargs)

    for number in range(limit):
        if stopper is not None:
            with request_scope(stopper):
                info = step(number)
        else:
            info = step(number, _deadline=time.monotonic() + BIG)
        if info["sync"]["state"] == "complete":
            return number + 1
    raise AssertionError("never completed")


def index_facts(engine, world):
    """(cursor offset or None, event rows by id, chunk count in the metrics) straight from the database."""
    conn, sid = engine.store._conn, world.sid.lower() if world.provider in ("codex", "agy") else world.sid
    row = conn.execute("SELECT byte_offset,prefix_hash FROM session_cursors WHERE provider=? AND session_id=?",
                       (world.provider, sid)).fetchone()
    events = {r[0]: r[1] for r in conn.execute("SELECT chunk_id,data FROM session_events WHERE provider=? AND session_id=?",
                                               (world.provider, sid))}
    metrics = conn.execute("SELECT chunk_count FROM session_metrics WHERE provider=? AND session_id=?",
                           (world.provider, sid)).fetchone()
    last = conn.execute("SELECT MAX(byte_start) FROM session_events WHERE provider=? AND session_id=?",
                        (world.provider, sid)).fetchone()[0]
    return (row[0], row[1]) if row else None, events, (metrics[0] if metrics else 0), last


def assert_valid_state(world, engine, before):
    """What must hold after ANY stop: a state an earlier or later complete commit could have left."""
    assert integrity_ok(engine.settings.index_dir)
    cursor, events, count, last = index_facts(engine, world)
    old_cursor, old_events, old_count, _ = before
    assert cursor is not None and cursor[0] >= old_cursor[0]
    data = world.path.read_bytes()
    assert data[cursor[0] - 1:cursor[0]] == b"\n"                                   # the end of a complete record
    kind = "sha" if world.provider == "agy" else "blocks"
    assert cursor[1] == expected_cursor_hash(world.path, cursor[0], kind)         # the hash of exactly that prefix
    assert last is None or last < cursor[0]                                        # no event row beyond the cursor
    assert all(events.get(cid) == data_json for cid, data_json in old_events.items())    # the previous index is intact
    assert count == len(events) >= old_count                                       # the metrics describe the rows
    if cursor == old_cursor:
        assert events == old_events                                                # nothing moved: nothing changed


def build(tmp_path, monkeypatch, provider, seed, count):
    world = World(tmp_path, monkeypatch, provider, seed=seed, count=count)
    full, cut = split_session(world)
    return world, full, cut


def run_faults(tmp_path, monkeypatch, provider, *, count, stride=1, seed=4):
    world, full, cut = build(tmp_path, monkeypatch, provider, seed, count)
    try:
        reference = world.engine("reference")
        world.copy_receipts(world.builder, reference)
        # What the one-shot sync of the whole session produces (the file is restored to its first part afterwards).
        world.path.write_bytes(full)
        reference.sync_session(**world.sync_args(), derived=False)
        expected = snapshot(reference, provider, world.sid)
        world.path.write_bytes(full[:cut])
        # How many cooperative checks a sliced sync of the appended part makes, so that every one can be the last.
        probe = world.engine("probe")
        world.copy_receipts(world.builder, probe)
        probe.sync_session(**world.sync_args(), derived=False)
        world.path.write_bytes(full)
        counter = Stopper()
        calls = sync_all(world, probe, caps=CAPS, stopper=counter)
        total = counter.calls
        assert calls >= 3 and total > 100
        faulted = 0
        for n in range(1, total + 1, stride):
            world.path.write_bytes(full[:cut])
            engine = world.engine(f"fault-{n}")
            world.copy_receipts(world.builder, engine)
            engine.sync_session(**world.sync_args(), derived=False)               # the previous index
            before = index_facts(engine, world)
            world.path.write_bytes(full)
            try:
                sync_all(world, engine, caps=CAPS, stopper=Stopper(stop_at=n))
            except RequestStopped:
                faulted += 1
            assert_valid_state(world, engine, before)
            sync_all(world, engine, caps=CAPS)                                # finish without faults
            done = snapshot(engine, provider, world.sid)
            no_mtime = lambda b: None if b is None else b[:2] + b[3:]       # noqa: E731  (this test rewrites the file)
            assert (done.events, done.cursor, no_mtime(done.blocks), done.metrics) == \
                (expected.events, expected.cursor, no_mtime(expected.blocks), expected.metrics)
            if provider == "agy":
                assert done.agy == expected.agy
            engine.close()
            world.engines.remove(engine)
        assert faulted == len(range(1, total + 1, stride))     # every stop point stops the sync
        return total
    finally:
        world.close()


def test_a_stop_at_every_check_leaves_a_valid_index_codex(tmp_path, monkeypatch):
    assert run_faults(tmp_path, monkeypatch, "codex", count=26) > 150


@pytest.mark.parametrize("provider", [p for p in PROVIDERS if p != "codex"])
def test_a_stop_at_sampled_checks_leaves_a_valid_index(tmp_path, monkeypatch, provider):
    run_faults(tmp_path, monkeypatch, provider, count=26 if provider != "agy" else 16, stride=7)


# --- a process killed at any moment ----------------------------------------------------------------------------------

CHILD = textwrap.dedent('''
    import json, sys, time
    from pathlib import Path
    sys.path[:0] = sys.argv[2:4]                       # the source tree and the tests directory
    from self_directing_mcp.config import Settings
    from self_directing_mcp.engine import SelfDirectEngine
    from slicing_support import forced
    spec = json.loads(Path(sys.argv[1]).read_text())
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, openrouter_api_key_file=None,
                                       index_dir=spec["index"], **spec["paths"]))
    log = open(spec["log"], "a", buffering=1)
    for number in range(100000):
        engine.slice_policy = forced(spec["cap"])
        log.write(f"start {number}\\n")
        info = engine.sync_session(session_id=spec["sid"], provider=spec["provider"], path=spec["path"], embed=False,
                                   derived=False, _deadline=time.monotonic() + 1e9)
        log.write(f"done {number} {info['sync']['indexed_through']}\\n")
        if info["sync"]["state"] == "complete":
            break
''')


def kill_child(tmp_path, world, *, after_start, delay, cap, name):
    """Run the slicing child on a fresh index, kill it ``delay`` seconds after its ``after_start``-th call began."""
    index = tmp_path / f"index-{name}"
    if world.provider == "agy":
        journal.copy_database(Path(world.builder.settings.index_dir) / "agy-receipts.sqlite", _empty(index) / "agy-receipts.sqlite")
    spec = {"index": str(index), "log": str(tmp_path / f"{name}.log"), "cap": cap, "sid": world.sid,
            "provider": world.provider, "path": str(world.path),
            "paths": {key: ([str(p) for p in value] if isinstance(value, list) else str(value))
                      for key, value in world.paths.items()}}
    (tmp_path / f"{name}.json").write_text(json.dumps(spec))
    Path(spec["log"]).write_text("")
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    child = subprocess.Popen([sys.executable, "-I", "-B", str(script), str(tmp_path / f"{name}.json"), str(ROOT / "src"),
                              str(ROOT / "tests")], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and child.poll() is None:
            if f"start {after_start}\n" in Path(spec["log"]).read_text():
                time.sleep(delay)
                break
            time.sleep(0.001)
    finally:
        finished = child.poll() is not None
        if not finished:
            child.kill()                                  # SIGKILL (TerminateProcess on Windows): no cleanup runs
        child.wait(timeout=30)
    return index, finished, child.stderr.read().decode()[-2000:]


def _empty(directory):
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.mark.parametrize("provider,trials", [("codex", 6), ("omp", 2), ("claude", 2), ("grokbot", 2), ("agy", 2)])
def test_a_killed_sliced_sync_leaves_an_index_that_verifies_and_completes(tmp_path, monkeypatch, provider, trials):
    world = World(tmp_path, monkeypatch, provider, seed=11, count=1500 if provider != "agy" else 400)
    try:
        reference = world.engine("reference")
        world.copy_receipts(world.builder, reference)
        reference.sync_session(**world.sync_args(), derived=False)
        expected = snapshot(reference, provider, world.sid)
        no_mtime = lambda b: None if b is None else b[:2] + b[3:]       # noqa: E731
        rng, killed_between_slices, records = random.Random(3), 0, world.path.read_bytes().count(b"\n")
        for trial in range(trials):
            # A kill 0 to 30 ms after a call began must find the sync unfinished (and not only on the machine this was written
            # on): small slices, so that dozens of calls (each a few milliseconds) are left after the aimed one, and an early one.
            cap = rng.choice([10, 20, 40])
            after = rng.randint(2, max(2, min(8, records // cap - 3)))       # a slice that really begins before the end
            index, finished, errors = kill_child(tmp_path, world, after_start=after, delay=rng.uniform(0, 0.03),
                                                 cap=cap, name=f"crash-{trial}")
            assert finished or not errors.strip(), errors            # a child that was killed never reported a failure
            # Open the survivor in this process: a hot journal (a commit that was in flight) is rolled back by SQLite here.
            survivor = SelfDirectEngine(Settings(_env_file=None, local_only=True, openrouter_api_key_file=None,
                                                 index_dir=index, **world.paths))
            world.engines.append(survivor)
            survivor.ensure_ready()
            assert integrity_ok(index)
            cursor, events, count, last = index_facts(survivor, world)
            if cursor is not None:
                assert world.path.read_bytes()[cursor[0] - 1:cursor[0]] == b"\n"
                assert cursor[1] == expected_cursor_hash(world.path, cursor[0], "sha" if provider == "agy" else "blocks")
                assert last is None or last < cursor[0] and count == len(events)
            else:
                assert not events
            sync_all(world, survivor, caps=(50, 200, 1000))                                  # a later sync completes
            done = snapshot(survivor, provider, world.sid)
            assert (done.events, done.cursor, no_mtime(done.blocks), done.metrics) == \
                (expected.events, expected.cursor, no_mtime(expected.blocks), expected.metrics)
            killed_between_slices += cursor is not None and not finished
        assert killed_between_slices >= 1      # a kill landed after a slice was committed and before the sync was done
    finally:
        world.close()


# --- a stop right after a slice was committed ------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["first", "last"])
def test_a_stop_right_after_a_slice_was_committed_loses_the_response_not_the_slice(tmp_path, monkeypatch, which):
    """The commit of a slice is a transaction that returned. A request that runs out of time (or is cancelled) at the very
    next cooperative check gets no answer, but the slice stays published: the cursor moved, its events are in, the index
    verifies, and the next call parses on from that cursor (never from byte 0) to the index the one-shot sync makes.
    ``which``: the first slice of the sync, or the last one (the sync was complete and only its answer is lost)."""
    world = World(tmp_path, monkeypatch, "codex", seed=7, count=400, partial=False)
    try:
        reference = world.engine("reference")
        reference.sync_session(**world.sync_args(), derived=False)
        expected = snapshot(reference, "codex", world.sid)
        size = world.path.stat().st_size
        engine = world.engine("stopped")
        control, parses, committed = Tripwire(), [], []
        commit, parse = ingest.SessionStore.commit_events, codex_parser.parse_session_chunks

        def committing(self, *args, **kwargs):
            commit(self, *args, **kwargs)
            committed.append(kwargs["offset"])
            control.armed = which == "first" or kwargs["offset"] >= size       # from now on every look at the clock is a stop

        def parsing(*args, **kwargs):
            parses.append(kwargs.get("start_byte", 0))
            return parse(*args, **kwargs)

        monkeypatch.setattr(ingest.SessionStore, "commit_events", committing)
        monkeypatch.setattr(codex_parser, "parse_session_chunks", parsing)
        with pytest.raises(RequestStopped) as stopped:
            for _ in range(500):
                engine.slice_policy = forced(60)
                with request_scope(control):
                    engine.sync_session(**world.sync_args(), derived=False)
        assert stopped.value.reason == "request_deadline"
        assert integrity_ok(engine.settings.index_dir)
        kept = snapshot(engine, "codex", world.sid)
        assert kept.events and kept.events == expected.events[:len(kept.events)]            # the slices are in, in order
        assert kept.cursor[1] == committed[-1] and (committed[-1] == size) == (which == "last")
        assert (len(committed) == 1) == (which == "first")
        resumed = len(parses)
        sync_all(world, engine, caps=(100, 250))                                           # the next calls, without a stop
        assert all(start >= kept.cursor[1] for start in parses[resumed:])                  # nothing was parsed twice
        done = snapshot(engine, "codex", world.sid)
        assert (done.events, done.cursor, done.blocks, done.metrics) == \
            (expected.events, expected.cursor, expected.blocks, expected.metrics)
        assert integrity_ok(engine.settings.index_dir)
    finally:
        world.close()
