"""A staged re-index under failure (roundtable 8, PR3): a stop at ANY cooperative check, and a process that dies at each step.

The red line: an interrupted or crashed re-index leaves the old ids, anchors and vectors exactly as they were.

* Fault injection. A request is stopped at its N-th ``check_request`` call, for every N from 1 to the number of calls the
  staged sequence makes. After each stop the index is either the old one, byte for byte, or the re-index published whole; never
  anything in between. An audit of the old index is not clean and still finds the forbidden call it holds. Then the sync is
  finished without faults and equals the one-shot re-index.
* Crash. A subprocess re-indexes in slices and ``os._exit``s at: the row frames written and the trailer not; the trailer
  written and not synced; the spool complete and the commit not started; inside the commit transaction; after the commit and
  before the unlink. The index is verified in a second process each time.
* A stop right after a step. The request that runs out of time (or is cancelled) at the first check after a slice was spooled,
  or after the commit that published the re-index, loses its answer and nothing else.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from reindex_support import BIG, Pair, canonical, index_dump, old_parser, spool_files
from self_directing_mcp import reindex_spool
from self_directing_mcp.codex import parse as codex_parser
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index import ingest
from self_directing_mcp.request_control import RequestStopped, request_scope
from slicing_support import PROVIDERS, forced, integrity_ok
from test_slice_convergence import Tripwire
from test_slice_faults import Stopper

ROOT = Path(__file__).resolve().parents[1]
CAPS = (3, 5, 7)


def staged_until_published(world, engine, caps, stopper=None, limit=500):
    """Slice-sync until the re-index is published. With a ``stopper`` every call of the sequence is one request scope, so its
    check counter runs through all of them. Returns the number of calls."""
    for number in range(limit):
        engine.slice_policy = forced(caps[number % len(caps)])
        if stopper is not None:
            with request_scope(stopper):
                info = engine.sync_session(**world.sync_args(), derived=False)
        else:
            info = engine.sync_session(**world.sync_args(), derived=False, _deadline=time.monotonic() + BIG)
        if info["sync"]["state"] == "complete":
            return number + 1
    raise AssertionError("never published")


def fresh_engine(p, name):
    """An engine with its own index that holds the old index of the session (the one every fault starts from)."""
    engine = p.world.engine(name)
    p.world.copy_receipts(p.world.builder, engine)
    p.world.upsert_contracts(engine)
    with old_parser(p.provider):
        engine.sync_session(**p.world.sync_args(), derived=True)
    engine.store.cache_vectors("synthetic-model", [("text-hash-1", [0.5, 1.0, -0.5]), ("text-hash-2", [0.25, 0.0, 1.0])])
    return engine


def audit_during_the_reindex(p, engine, before, expected, *, commit_fits=True):
    """An audit of the index an interruption left (its own sync stages one record more, or, when everything is staged already,
    publishes the re-index when its commit fits): while the old index is in force the answer is incomplete, never clean, never a
    judged prefix, and a forbidden call in the old events is still a violation; if the audit's sync published, the index is the
    re-index. ``commit_fits=False``: the time left is too short for the commit, so the old index stays in force."""
    from self_directing_mcp.index import ingest
    engine.slice_policy = forced(1)
    deadline = time.monotonic() + (BIG if commit_fits else 60)
    original = ingest.COMMIT_SECONDS_PER_ROW
    ingest.COMMIT_SECONDS_PER_ROW = original if commit_fits else 10.0
    try:
        audit = engine.audit_session(p.sid, provider=p.provider, path=str(p.world.path), _deadline=deadline)
    finally:
        ingest.COMMIT_SECONDS_PER_ROW = original
    coverage = audit["coverage"]
    if {"parser_outdated", "sync_in_progress", "reindex_ready_to_commit"} & set(coverage["issues"]):
        assert coverage["complete"] is False and not coverage.get("judged_prefix") and audit["verdict"] == "violation"
        assert "parser_outdated" in coverage["issues"] and index_dump(engine) == before
        return False
    assert commit_fits and canonical(engine, p.provider, p.sid) == expected       # the audit's own sync published it
    return True


def run_faults(tmp_path, monkeypatch, provider, *, count, stride, seed=4, audit_every=6):
    p = Pair(tmp_path, monkeypatch, provider, "parser", seed=seed, count=count)
    try:
        expected = p.expected()
        assert any("rm -rf" in row[5] for row in p.before["session_events"])         # something forbidden is in the old index
        probe = fresh_engine(p, "probe")
        counter = Stopper()
        calls = staged_until_published(p.world, probe, CAPS, stopper=counter)
        total = counter.calls
        assert calls >= 6 and total > 100
        assert canonical(probe, provider, p.sid) == expected
        faulted = published = 0
        for n in range(1, total + 1, stride):
            engine = fresh_engine(p, f"fault-{n}")
            before = index_dump(engine)
            try:
                staged_until_published(p.world, engine, CAPS, stopper=Stopper(stop_at=n))
            except RequestStopped:
                faulted += 1
            assert integrity_ok(engine.settings.index_dir)
            if index_dump(engine) == before:                        # the re-index was not published: the old index, to the byte
                if n % audit_every == 1:
                    published += audit_during_the_reindex(p, engine, before, expected)
            else:                                                   # published whole, or not at all
                published += 1
                assert canonical(engine, provider, p.sid) == expected
            staged_until_published(p.world, engine, CAPS)             # finish without faults
            assert canonical(engine, provider, p.sid) == expected
            assert not spool_files(engine)
            engine.close()
            p.world.engines.remove(engine)
        assert faulted == len(range(1, total + 1, stride))           # every stop point stops the sync
        assert published >= 1                                       # and the last few come after the commit
        return total
    finally:
        p.close()


def test_a_stop_at_every_check_leaves_the_old_index_or_the_published_one_codex(tmp_path, monkeypatch):
    """Every one of the cooperative checks of the staged sequence (about 570 of them, in 6 calls) is a stop point."""
    assert run_faults(tmp_path, monkeypatch, "codex", count=16, stride=1) > 500


@pytest.mark.parametrize("provider", [name for name in PROVIDERS if name not in ("codex", "grokbot")])
def test_a_stop_at_sampled_checks_leaves_the_old_index_or_the_published_one(tmp_path, monkeypatch, provider):
    run_faults(tmp_path, monkeypatch, provider, count=26 if provider != "agy" else 16, stride=11)


# --- a process that dies at each step ---------------------------------------------------------------------------------

CHILD = textwrap.dedent('''
    import json, os, sys, time
    from pathlib import Path
    sys.path[:0] = sys.argv[2:4]                       # the source tree and the tests directory
    spec = json.loads(Path(sys.argv[1]).read_text())
    from self_directing_mcp import reindex_spool
    from self_directing_mcp.config import Settings
    from self_directing_mcp.engine import SelfDirectEngine
    from self_directing_mcp.index import ingest
    from slicing_support import forced
    point, at = spec["point"], spec["at"]
    Spool, state = reindex_spool.ReindexSpool, {"after_trailer": False}

    def die(*args, **kwargs):
        os._exit(7)                                    # no cleanup runs: a process that is killed

    if point == "rows":                                # the rows of slice ``at`` are written, its trailer is not
        write_trailer = Spool._write_trailer

        def trailer(self, record):
            if record["slice_seq"] >= at:
                die()
            return write_trailer(self, record)

        Spool._write_trailer = trailer
    elif point == "trailer":                           # the trailer of slice ``at`` is written, the file is not synced
        write_trailer, sync = Spool._write_trailer, Spool._sync

        def trailer(self, record):
            write_trailer(self, record)
            state["after_trailer"] = record["slice_seq"] >= at

        def synced(self):
            if state["after_trailer"]:
                die()
            return sync(self)

        Spool._write_trailer, Spool._sync = trailer, synced
    elif point == "complete":                          # everything is staged, the commit is about to start
        ingest.SessionStore.commit_events = die
    elif point == "commit":                            # the commit transaction is open: events, cursor, metrics written
        ingest.SessionStore._record_commit = die
    elif point == "unlink":                            # the commit returned, the spool is not deleted
        Spool.finish = die
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, openrouter_api_key_file=None,
                                       index_dir=spec["index"], **spec["paths"]))
    log = open(spec["log"], "a", buffering=1) if spec.get("log") else None
    for number in range(100000):
        if log:
            log.write(f"start {number}\\n")
        engine.slice_policy = forced(spec["caps"][number % len(spec["caps"])])
        info = engine.sync_session(session_id=spec["sid"], provider=spec["provider"], path=spec["path"], embed=False,
                                   derived=False, _deadline=time.monotonic() + 1e9)
        if info["sync"]["state"] == "complete":
            break
    sys.exit(0)                                        # the kill point was never reached
''')

POINTS = ("rows", "trailer", "complete", "commit", "unlink")


def kill_at(tmp_path, p, engine, point, at=3):
    """Run the re-index of ``engine``'s index in a subprocess that ``os._exit``s at ``point``; returns its exit code."""
    index = engine.settings.index_dir
    engine.close()
    p.world.engines.remove(engine)
    spec = {"index": str(index), "point": point, "at": at, "caps": list(CAPS), "sid": p.sid, "provider": p.provider,
            "path": str(p.world.path),
            "paths": {key: ([str(item) for item in value] if isinstance(value, list) else str(value))
                      for key, value in p.world.paths.items()}}
    (tmp_path / f"{point}.json").write_text(json.dumps(spec))
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    child = subprocess.run([sys.executable, "-I", "-B", str(script), str(tmp_path / f"{point}.json"), str(ROOT / "src"),
                            str(ROOT / "tests")], capture_output=True, timeout=300)
    return child.returncode, child.stderr.decode()[-1500:]


def kill_while_staging(tmp_path, p, engine, *, after_start, delay, name):
    """Run the staging of ``engine``'s index in a subprocess and SIGKILL it ``delay`` seconds after its ``after_start``-th call
    began (so: in the middle of a parse, a spool write or a sync, wherever that lands). Returns (index, finished, stderr)."""
    index = engine.settings.index_dir
    engine.close()
    p.world.engines.remove(engine)
    log = tmp_path / f"{name}.log"
    log.write_text("")
    spec = {"index": str(index), "point": "none", "at": 0, "caps": list(CAPS), "sid": p.sid, "provider": p.provider,
            "path": str(p.world.path), "log": str(log),
            "paths": {key: ([str(item) for item in value] if isinstance(value, list) else str(value))
                      for key, value in p.world.paths.items()}}
    (tmp_path / f"{name}.json").write_text(json.dumps(spec))
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    child = subprocess.Popen([sys.executable, "-I", "-B", str(script), str(tmp_path / f"{name}.json"), str(ROOT / "src"),
                              str(ROOT / "tests")], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and child.poll() is None:
            if f"start {after_start}\n" in log.read_text():
                time.sleep(delay)
                break
            time.sleep(0.001)
    finally:
        finished = child.poll() is not None
        if not finished:
            child.kill()                                  # SIGKILL (TerminateProcess on Windows): no cleanup runs
        child.wait(timeout=30)
    return index, finished, child.stderr.read().decode()[-1500:]


def survivor_of(p, index):
    """Open the index the dead process left, in this one (a hot journal is rolled back by SQLite here)."""
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, openrouter_api_key_file=None, index_dir=index,
                                       **p.world.paths))
    p.world.engines.append(engine)
    engine.ensure_ready()
    return engine


@pytest.mark.parametrize("point", POINTS)
def test_a_process_that_dies_at_each_step_of_a_staged_reindex_leaves_the_old_index(tmp_path, monkeypatch, point):
    p = Pair(tmp_path, monkeypatch, "codex", "parser", seed=11, count=60)
    try:
        expected = p.expected()
        victim = fresh_engine(p, "victim")
        before, index = index_dump(victim), victim.settings.index_dir
        code, errors = kill_at(tmp_path, p, victim, point)
        assert code == 7, errors                                     # the child died at the kill point, not before or after
        survivor = survivor_of(p, index)
        assert integrity_ok(index)
        spools = spool_files(survivor)
        if point == "unlink":
            # The commit returned: the index IS the re-index, and a leftover spool is all that is left of it.
            assert canonical(survivor, "codex", p.sid) == expected and len(spools) == 1
            again = survivor.sync_session(**p.world.sync_args(), derived=False, _deadline=time.monotonic() + BIG)
            assert again["sync"]["state"] == "complete" and "reindex" not in again["sync"] and not spool_files(survivor)
            assert canonical(survivor, "codex", p.sid) == expected
            return
        assert index_dump(survivor) == before                       # ids, anchors and vector cache keys: byte for byte
        assert len(spools) == 1                                      # the staged work is still there
        # An audit that has no time for a commit judges the old index (it stages one more record, or, if all is staged, none).
        assert audit_during_the_reindex(p, survivor, before, expected, commit_fits=False) is False
        assert integrity_ok(index)
        staged_calls = staged_until_published(p.world, survivor, CAPS)      # a later sync converges
        assert staged_calls >= 1
        assert canonical(survivor, "codex", p.sid) == expected and not spool_files(survivor)
    finally:
        p.close()


@pytest.mark.parametrize("provider", ["omp", "claude", "agy", "grokbot"])
def test_a_process_that_dies_inside_the_commit_of_each_provider_leaves_the_old_index(tmp_path, monkeypatch, provider):
    trigger = "source" if provider == "grokbot" else "parser"
    p = Pair(tmp_path, monkeypatch, provider, trigger, seed=13, count=60 if provider != "agy" else 40)
    try:
        expected = p.expected()
        victim = fresh_engine_for(p, "victim")
        before, index = index_dump(victim), victim.settings.index_dir
        code, errors = kill_at(tmp_path, p, victim, "commit")
        assert code == 7, errors
        survivor = survivor_of(p, index)
        assert index_dump(survivor) == before and len(spool_files(survivor)) == 1
        staged_until_published(p.world, survivor, CAPS)
        assert not spool_files(survivor)
        done = canonical(survivor, provider, p.sid)
        if trigger == "source":      # the victim was indexed from the unedited bytes written back: its file has another mtime
            for table in (done, expected):
                table["blocks"] = table["blocks"][:2] + table["blocks"][3:]
        assert done == expected
    finally:
        p.close()


def fresh_engine_for(p, name):
    """``fresh_engine`` for a trigger that is a changed source (Grok Bot has no parser version): the same old index, then the
    word is changed (``Pair`` already changed it for the reference)."""
    if p.trigger == "parser":
        return fresh_engine(p, name)
    # The file was edited by Pair after the others were indexed; index the unedited bytes, then put the edit back in place.
    engine = p.world.engine(name)
    p.world.upsert_contracts(engine)
    edited = p.world.path.read_bytes()
    p.world.path.write_bytes(p.original)
    engine.sync_session(**p.world.sync_args(), derived=True)
    p.world.path.write_bytes(edited)
    engine.store.cache_vectors("synthetic-model", [("text-hash-1", [0.5, 1.0, -0.5])])
    return engine


@pytest.mark.skipif(sys.platform == "win32", reason="the kill is a POSIX SIGKILL in this test")
def test_a_staging_process_killed_at_random_moments_leaves_the_old_index_and_a_spool_that_resumes(tmp_path, monkeypatch):
    """Not at a chosen step: killed 0 to 30 ms into a call that was chosen at random, so the file may end inside a row frame,
    between frames, after a trailer or in the middle of the sync. Whatever it left, the index is the old one and the next
    sync resumes the spool or starts over, and publishes the same re-index."""
    import random
    p = Pair(tmp_path, monkeypatch, "codex", "parser", seed=5, count=400)
    try:
        expected = p.expected()
        rng = random.Random(3)
        spools = []
        for trial in range(4):
            victim = fresh_engine(p, f"victim-{trial}")
            before = index_dump(victim)
            index, finished, errors = kill_while_staging(tmp_path, p, victim, after_start=rng.randint(4, 14),
                                                         delay=rng.uniform(0, 0.03), name=f"kill-{trial}")
            assert not finished, errors                       # still staging when it was killed
            survivor = survivor_of(p, index)
            assert integrity_ok(index) and index_dump(survivor) == before
            spools.append(len(spool_files(survivor)))
            staged_until_published(p.world, survivor, (50, 200, 1000))        # resumes it, or starts over
            assert canonical(survivor, "codex", p.sid) == expected and not spool_files(survivor)
        assert max(spools) == 1                                # a kill did leave a spool behind to resume
    finally:
        p.close()


@pytest.mark.parametrize("point", ["slice", "commit"])
def test_a_stop_right_after_a_spooled_slice_or_the_commit_loses_the_response_only(tmp_path, monkeypatch, point):
    """``slice``: the request is stopped at the first check after a slice was written to the spool; the old index is untouched,
    the spool holds the slice, and the next call goes on from where it ended (it does not parse from byte 0 again).
    ``commit``: it is stopped at the first check after the commit that published the re-index returned; the index IS the
    re-index (only the answer was lost), and the next call finds it complete and removes what is left of the spool."""
    p = Pair(tmp_path, monkeypatch, "codex", "parser", seed=11, count=120)
    try:
        expected = p.expected()
        engine = fresh_engine(p, "stopped")
        before, index = index_dump(engine), engine.settings.index_dir
        control, parses, spooled = Tripwire(), [], []
        append, commit = reindex_spool.ReindexSpool.append_slice, ingest.SessionStore.commit_events
        parse = codex_parser.parse_session_chunks

        def appending(self, rows, end_offset, *args, **kwargs):
            result = append(self, rows, end_offset, *args, **kwargs)
            spooled.append(end_offset)
            control.armed = point == "slice"                 # from now on every look at the clock is a stop
            return result

        def committing(self, *args, **kwargs):
            commit(self, *args, **kwargs)
            control.armed = point == "commit"

        def parsing(*args, **kwargs):
            parses.append(kwargs.get("start_byte", 0))
            return parse(*args, **kwargs)

        monkeypatch.setattr(reindex_spool.ReindexSpool, "append_slice", appending)
        monkeypatch.setattr(ingest.SessionStore, "commit_events", committing)
        monkeypatch.setattr(codex_parser, "parse_session_chunks", parsing)
        with pytest.raises(RequestStopped) as stopped:
            staged_until_published(p.world, engine, CAPS, stopper=control)
        assert stopped.value.reason == "request_deadline"
        assert integrity_ok(index)
        resumed = len(parses)
        if point == "slice":
            assert index_dump(engine) == before and len(spool_files(engine)) == 1 and len(spooled) == 1
            staged_until_published(p.world, engine, CAPS)
            assert parses[resumed] == spooled[0] > 0                    # the next call went on where the stopped slice ended
        else:
            assert canonical(engine, "codex", p.sid) == expected and len(spool_files(engine)) <= 1
            again = engine.sync_session(**p.world.sync_args(), derived=False, _deadline=time.monotonic() + BIG)
            assert again["sync"]["state"] == "complete" and "reindex" not in again["sync"]
        assert canonical(engine, "codex", p.sid) == expected and not spool_files(engine) and integrity_ok(index)
    finally:
        p.close()
