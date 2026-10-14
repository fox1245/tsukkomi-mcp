"""Background full verification of session sources for RESIDENT servers.

A fast hook-time check compares only some blocks of a source. This worker closes the gap in the background: after
the engine has been idle for a while (``SELF_DIRECT_INTEGRITY_BACKGROUND_IDLE_SEC``, default 2 s) it verifies every
block of each watched source and stamps the stored table (``full_verified_at`` / ``full_verified_through``), so the
next fast check may rely on a recent full verification. During continuous activity a scan starts shortly before the
5-minute deadline instead. Rules:

* one thread, one scan at a time (``SELF_DIRECT_INTEGRITY_BACKGROUND_MAX_SCANS`` bounds the scans per wake-up);
* it never blocks a request: a scan holds no engine lock while it reads, and a request that arrives first simply
  runs (a fast check, or the full one it needs) on its own;
* a deadline it could not meet is reported, not hidden: ``snapshot()`` (surfaced by ``audit_status``) counts
  ``missed_deadlines`` and the worst lag, and the request that had to verify inline says so in its
  ``integrity.full_reason`` / ``verification_lag_sec``;
* hook processes (Claude Code, AGY) live for one event: they never start this worker.

What it does not do: a source that changes and changes back between two verifications is not detected; no check
here is a statement about the source as it is now, only about when it was last read in full.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import threading
import time

from self_directing_mcp.request_control import IndexBusy, RequestStopped


@dataclass
class _Entry:
    path: str
    dirty: bool = False  # a fast check ran since the last full verification
    noted: int = 0  # increments with every fast note; a scan clears `dirty` only if nothing was noted meanwhile
    deadline: float | None = None  # wall clock: last full verification + interval
    retry_at: float = 0.0


class IntegrityWorker:
    def __init__(self, engine, *, idle_sec: float = 2.0, interval_sec: float = 300.0, max_scans: int = 1,
                 poll_sec: float = 0.25, budget_sec: float = 45.0, wall=time.time):
        self.engine = engine
        self.idle_sec, self.interval_sec, self.max_scans = idle_sec, interval_sec, max(1, max_scans)
        self.poll_sec, self.budget_sec, self._wall = poll_sec, budget_sec, wall
        self.margin_sec = min(30.0, 0.2 * interval_sec)
        self._watch: dict[tuple[str, str], _Entry] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats = {"scans": 0, "deferred": 0, "errors": 0, "missed_deadlines": 0, "max_lag_sec": 0.0,
                      "last_error": None, "last_scan_at": None}

    # -- engine side ----------------------------------------------------------------------------------------
    def note(self, provider: str, session_id: str, path: str, integrity: dict) -> None:
        """Called by the engine after every sync: remember the source and whether a full verification is owed."""
        key = (provider, session_id)
        verified = _parse(integrity.get("full_verified_at"))
        with self._lock:
            entry = self._watch.setdefault(key, _Entry(path))
            entry.path = path
            if verified is not None:
                entry.deadline = verified + self.interval_sec
            if integrity.get("mode") == "fast":
                entry.dirty, entry.noted = True, entry.noted + 1

    def snapshot(self) -> dict:
        """What ``audit_status`` reports: counters and the sources currently owed a full verification."""
        with self._lock:
            owed = sum(1 for entry in self._watch.values() if entry.dirty)
            return {**self.stats, "watched": len(self._watch), "owed_full_verification": owed,
                    "running": self._thread is not None and self._thread.is_alive(),
                    "idle_sec": self.idle_sec, "interval_sec": self.interval_sec}

    # -- lifecycle --------------------------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="tsukkomi-integrity", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.wait(self.poll_sec):
            try:
                self.tick()
            except Exception as exc:  # the worker must outlive any one failure
                self.stats["errors"] += 1
                self.stats["last_error"] = type(exc).__name__

    # -- scheduling -------------------------------------------------------------------------------------------
    def _due(self, entry: _Entry, idle: float, now: float) -> bool:
        if not entry.dirty or now < entry.retry_at:
            return False
        if idle >= self.idle_sec:
            return True
        # Busy: still start in time for the deadline, so that no request has to verify inline.
        return entry.deadline is not None and now >= entry.deadline - self.margin_sec

    def tick(self) -> int:
        """One scheduling round; returns how many scans ran. Public for tests."""
        if not getattr(self.engine, "accepts_background_work", lambda: True)():
            return 0  # closed or closing: never reopen an engine that was shut down
        now = self._wall()
        idle = self.engine.idle_seconds()
        with self._lock:
            due = sorted(((entry.deadline if entry.deadline is not None else 0.0, key) for key, entry
                          in self._watch.items() if self._due(entry, idle, now)))
        ran = 0
        for _deadline, key in due[:self.max_scans]:
            if self._stop.is_set():
                break
            self._scan(key)
            ran += 1
        return ran

    def _scan(self, key: tuple[str, str]) -> None:
        provider, session_id = key
        with self._lock:
            entry = self._watch[key]
            path, noted, deadline = entry.path, entry.noted, entry.deadline
        try:
            info = self.engine.verify_integrity(session_id, provider=provider, path=path,
                                                _deadline=time.monotonic() + self.budget_sec)
        except (IndexBusy, RequestStopped):
            self.stats["deferred"] += 1  # a request or the lifetime got there first; try again next round
            return
        except Exception as exc:
            self.stats["errors"] += 1
            self.stats["last_error"] = type(exc).__name__
            with self._lock:
                entry.retry_at = self._wall() + 30.0  # no hot loop on a persistent failure
            return
        integrity = info.get("integrity") or {}
        finished = self._wall()
        with self._lock:
            if info.get("ok") and integrity.get("mode") == "full":
                entry.dirty = entry.noted != noted
                verified = _parse(integrity.get("full_verified_at"))
                entry.deadline = (verified if verified is not None else finished) + self.interval_sec
                self.stats["scans"] += 1
                self.stats["last_scan_at"] = datetime.fromtimestamp(finished, timezone.utc).isoformat()
            else:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(info.get("error") or "verification_not_full")
                entry.retry_at = finished + 30.0
            if deadline is not None and finished > deadline:
                self.stats["missed_deadlines"] += 1
                self.stats["max_lag_sec"] = round(max(self.stats["max_lag_sec"], finished - deadline), 3)


def _parse(value) -> float | None:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return (moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)).timestamp()


def start_integrity_worker(engine) -> IntegrityWorker | None:
    """Attach and start the worker of a resident server; None when disabled by settings."""
    settings = engine.settings
    if not settings.integrity_background_verify:
        return None
    worker = IntegrityWorker(engine, idle_sec=settings.integrity_background_idle_sec,
                             interval_sec=settings.integrity_full_verify_interval_sec,
                             max_scans=settings.integrity_background_max_scans,
                             budget_sec=settings.request_timeout_sec)
    engine.attach_integrity_worker(worker)
    worker.start()
    return worker
