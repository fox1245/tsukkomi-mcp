"""Background full verification for resident servers: scheduling rules and one real end-to-end scan."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import threading
import time

import pytest

from self_directing_mcp.integrity_worker import IntegrityWorker, start_integrity_worker
from self_directing_mcp.request_control import IndexBusy, RequestStopped
from test_integrity_fast_path import DEPLOY, SAFE, SID, World, make_world, tampered_text  # noqa: F401


class Wall:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


class FakeEngine:
    """Just the surface the worker uses."""

    def __init__(self, wall):
        self.wall, self.idle, self.calls, self.raises = wall, 0.0, [], None

    def idle_seconds(self):
        return self.idle

    def verify_integrity(self, session_id, provider=None, path=None, _deadline=None):
        self.calls.append((provider, session_id, path))
        if self.raises is not None:
            raise self.raises
        stamp = datetime.fromtimestamp(self.wall(), timezone.utc).isoformat()
        return {"ok": True, "integrity": {"mode": "full", "full_verified_at": stamp}}


def fast(at):
    return {"mode": "fast", "full_verified_at": datetime.fromtimestamp(at, timezone.utc).isoformat()}


def full(at):
    return {"mode": "full", "full_verified_at": datetime.fromtimestamp(at, timezone.utc).isoformat()}


@pytest.fixture
def rig():
    wall = Wall()
    engine = FakeEngine(wall)
    worker = IntegrityWorker(engine, idle_sec=2.0, interval_sec=300.0, max_scans=1, wall=wall)
    return wall, engine, worker


# ------------------------------------------------------------------------------------------------ scheduling

def test_a_fully_verified_source_is_not_rescanned(rig):
    wall, engine, worker = rig
    worker.note("codex", "s1", "/p", full(wall.now))
    engine.idle = 60
    assert worker.tick() == 0 and engine.calls == []


def test_a_source_checked_only_fast_is_verified_in_full_after_the_idle_delay(rig):
    wall, engine, worker = rig
    worker.note("codex", "s1", "/p", fast(wall.now - 10))
    engine.idle = 1.9  # not idle long enough
    assert worker.tick() == 0
    engine.idle = 2.0
    assert worker.tick() == 1 and engine.calls == [("codex", "s1", "/p")]
    assert worker.snapshot()["owed_full_verification"] == 0 and worker.stats["scans"] == 1
    assert worker.tick() == 0  # verified: nothing more is owed


def test_during_activity_a_scan_starts_in_time_for_the_deadline(rig):
    wall, engine, worker = rig
    verified = wall.now
    worker.note("codex", "s1", "/p", fast(verified))
    engine.idle = 0.0  # never idle
    wall.now = verified + 100
    assert worker.tick() == 0  # far from the 5 minute deadline
    wall.now = verified + 300 - worker.margin_sec + 1
    assert worker.tick() == 1  # shortly before it: do not make a request pay for it inline


def test_scans_run_one_at_a_time_and_at_most_max_scans_per_round(rig):
    wall, engine, worker = rig
    worker.max_scans = 2
    for index in range(5):
        worker.note("codex", f"s{index}", f"/p{index}", fast(wall.now - 10 - index))
    engine.idle = 5
    assert worker.tick() == 2 and len(engine.calls) == 2
    assert worker.tick() == 2 and worker.tick() == 1 and worker.tick() == 0
    assert sorted(call[1] for call in engine.calls) == [f"s{i}" for i in range(5)]


def test_a_fast_check_during_a_scan_keeps_the_source_owed(rig):
    wall, engine, worker = rig
    worker.note("codex", "s1", "/p", fast(wall.now))
    engine.idle = 5
    real = engine.verify_integrity

    def verify_and_meanwhile_a_fast_check(*args, **kwargs):
        worker.note("codex", "s1", "/p", fast(wall.now))  # a request ran its cheap check during the scan
        return real(*args, **kwargs)

    engine.verify_integrity = verify_and_meanwhile_a_fast_check
    assert worker.tick() == 1
    assert worker.snapshot()["owed_full_verification"] == 1  # the new appended bytes are not yet verified in full
    engine.verify_integrity = real
    assert worker.tick() == 1 and worker.snapshot()["owed_full_verification"] == 0


@pytest.mark.parametrize("failure", [IndexBusy(), RequestStopped("request_deadline")])
def test_a_scan_that_a_request_or_the_lifetime_beat_is_deferred_not_failed(rig, failure):
    wall, engine, worker = rig
    worker.note("codex", "s1", "/p", fast(wall.now))
    engine.idle, engine.raises = 5, failure
    assert worker.tick() == 1
    assert worker.stats["deferred"] == 1 and worker.stats["errors"] == 0 and worker.snapshot()["owed_full_verification"] == 1
    engine.raises = None
    assert worker.tick() == 1 and worker.stats["scans"] == 1


def test_an_error_is_counted_visible_and_not_retried_in_a_hot_loop(rig):
    wall, engine, worker = rig
    worker.note("codex", "s1", "/p", fast(wall.now))
    engine.idle, engine.raises = 5, OSError("disk")
    assert worker.tick() == 1
    assert worker.stats["errors"] == 1 and worker.stats["last_error"] == "OSError"
    assert worker.tick() == 0  # backed off
    wall.now += 31
    engine.raises = None
    assert worker.tick() == 1 and worker.stats["scans"] == 1


def test_a_missed_deadline_is_surfaced_as_verification_lag(rig):
    wall, engine, worker = rig
    verified = wall.now
    worker.note("codex", "s1", "/p", fast(verified))
    engine.idle = 5
    wall.now = verified + 300 + 42  # busy for too long: the deadline passed before the scan could run
    assert worker.tick() == 1
    snapshot = worker.snapshot()
    assert snapshot["missed_deadlines"] == 1 and snapshot["max_lag_sec"] == pytest.approx(42.0, abs=0.01)
    assert snapshot["interval_sec"] == 300.0 and snapshot["idle_sec"] == 2.0


def test_the_worker_is_one_daemon_thread_and_stops(rig):
    wall, engine, worker = rig
    worker.poll_sec = 0.01
    worker.note("codex", "s1", "/p", fast(wall.now))
    engine.idle = 5
    before = {t for t in threading.enumerate()}
    worker.start()
    worker.start()  # idempotent
    started = [t for t in threading.enumerate() if t not in before]
    assert len(started) == 1 and started[0].daemon and started[0].name == "tsukkomi-integrity"
    deadline = time.monotonic() + 5
    while worker.stats["scans"] < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert worker.stats["scans"] == 1
    worker.stop()
    assert not started[0].is_alive() and worker.snapshot()["running"] is False


def test_settings_decide_whether_a_server_starts_it(make_world):
    world = make_world()
    world.engine.settings = world.engine.settings.model_copy(update={"integrity_background_verify": False})
    assert start_integrity_worker(world.engine) is None and world.engine._integrity_worker is None
    world.engine.settings = world.engine.settings.model_copy(update={
        "integrity_background_verify": True, "integrity_background_idle_sec": 0.5,
        "integrity_full_verify_interval_sec": 120.0, "integrity_background_max_scans": 3})
    worker = start_integrity_worker(world.engine)
    try:
        assert world.engine._integrity_worker is worker
        assert (worker.idle_sec, worker.interval_sec, worker.max_scans) == (0.5, 120.0, 3)
    finally:
        worker.stop()


def test_only_the_resident_server_starts_a_worker():
    """Hook processes live for one event: nothing in them may start background work."""
    source = Path(__file__).resolve().parents[1] / "src" / "self_directing_mcp"
    users = sorted(path.name for path in source.glob("*.py")
                   if "start_integrity_worker" in path.read_text(encoding="utf-8"))
    assert users == ["integrity_worker.py", "server.py"]
    server = (source / "server.py").read_text(encoding="utf-8")
    assert server.count("start_integrity_worker(_engine)") == 1 and "def main(" in server


# ------------------------------------------------------------------------------------------ the real thing

def test_a_real_scan_finds_a_silent_rewrite_and_stamps_the_table(make_world):
    from self_directing_mcp.integrity_worker import IntegrityWorker as Worker
    world = make_world()
    engine = world.engine
    worker = Worker(engine, idle_sec=0.0, interval_sec=300.0)
    engine.attach_integrity_worker(worker)
    world.check(verify="full")
    world.tamper_filler(12)  # no fast check with this policy compares block 12
    result = world.check(SAFE)  # the request notes the source as owed a full verification
    assert result["integrity"]["mode"] == "fast" and worker.snapshot()["owed_full_verification"] == 1
    assert not tampered_text(world)
    world.clock.advance(5)
    before = world.record().full_verified_at
    assert worker.tick() == 1  # idle (no request running): the background verification reads every block
    assert worker.stats["scans"] == 1 and worker.stats["errors"] == 0
    assert tampered_text(world)  # found, reindexed
    assert worker.snapshot()["owed_full_verification"] == 0
    assert world.record().full_verified_at != before
    status = engine.audit_status()
    assert status["integrity_worker"]["scans"] == 1  # surfaced next to the other status facts
    assert world.check(SAFE)["integrity"]["mode"] == "fast"  # the next fast check rests on that fresh verification


def test_a_worker_never_reopens_an_engine_that_was_closed(make_world):
    world = make_world()
    worker = IntegrityWorker(world.engine, idle_sec=0.0, interval_sec=300.0)
    world.engine.attach_integrity_worker(worker)
    world.check(verify="full")
    world.check(SAFE)  # a fast check: the source is owed a full verification
    assert worker.snapshot()["owed_full_verification"] == 1 and world.engine.accepts_background_work()
    world.engine.close()
    assert not world.engine.accepts_background_work()
    assert worker.tick() == 0 and world.engine.store is None  # nothing was reopened
    assert worker.stats["scans"] == 0 and worker.stats["errors"] == 0


def test_the_engine_reports_when_it_is_busy(make_world):
    world = make_world()
    engine = world.engine
    assert engine.idle_seconds() >= 0.0
    entered, release = threading.Event(), threading.Event()
    real = engine._audit

    def slow(*args, **kwargs):
        entered.set()
        release.wait(5)
        return real(*args, **kwargs)

    engine._audit = slow
    thread = threading.Thread(target=lambda: engine.audit_session(SID))
    thread.start()
    try:
        assert entered.wait(5)
        assert engine.idle_seconds() == 0.0  # a request is running: never idle
    finally:
        release.set()
        thread.join()
    assert engine.idle_seconds() >= 0.0
