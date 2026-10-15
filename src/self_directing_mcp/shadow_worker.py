"""Idle comparison of incremental audits against the full evaluator, for RESIDENT engines only.

The adaptive policy (``audit.shadow``) compares only some requests. A request that is not compared is answered from the
stored monitor state without a second opinion. This worker closes part of that gap in the background: when the engine has
been idle for a while (``SELF_DIRECT_AUDIT_SHADOW_IDLE_SEC``, default 2 s) it takes a session that answered requests
without a full comparison since its last match, and compares the latest state of that session with a full evaluation.

Rules:

* it runs in the daemon engine and the MCP stdio server only; hook processes live for one event and never start it;
* at most one pending job per session, one job running at a time (one thread);
* it never blocks a request: a request that arrives while a job runs cancels it (the job is cooperative, like every
  evaluation), and the job holds no engine lease while it evaluates;
* on a difference the contract's incremental path is disabled, the sampling rate returns to 1/4, and the requests answered
  since the last match are recorded as a range that needs review (``audit_status`` reports it).

What it is not: it does not retroactively certify earlier responses. A match says the latest state agrees with the full
evaluator; it says nothing about answers given before it. A mismatch says some of those answers may have been wrong, and
the recorded range names which ones to review. A job that finds nothing to compare (no time, a cancelled job) leaves the
session pending.

A job that was **preempted** by a request (the cancel event of the job was set) is not a failure and is retried as soon as
the engine is idle again. A job that ran out of its own budget, found the index busy, failed, or could not compare
(a time-limited pattern, no time) backs off exponentially (30 s, 60 s, ... 15 minutes) so that a session that cannot be compared
never costs a full evaluation every 30 seconds; a comparison that ran resets it. A session whose contracts no longer have an
incremental path at all (every contract was switched to the full evaluator) has nothing left to verify and is dropped.
Sessions are tried in turn (the one tried least recently first), so one that keeps failing cannot starve the others.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import time

from self_directing_mcp.request_control import IndexBusy, RequestStopped

VARIANTS = ("audit", "check")
BACKOFF_SEC = 30.0          # the first wait after a job that could not compare; it doubles with every failure in a row
BACKOFF_MAX_SEC = 900.0


@dataclass
class _Ledger:
    """Requests of one kind answered without a full comparison since the last match."""
    requests: int = 0
    first_at: str | None = None
    last_at: str | None = None
    first_offset: int | None = None
    last_offset: int | None = None
    spec: object = None            # the proposed action of the latest ``check`` request


@dataclass
class _Entry:
    path: str | None
    ledgers: dict = field(default_factory=lambda: {name: _Ledger() for name in VARIANTS})
    retry_at: float = 0.0          # not before (worker clock): the backoff of a job that could not compare
    failures: int = 0              # jobs in a row that could not compare
    tried_at: float = 0.0          # when a job last started, so that the sessions are tried in turn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ShadowWorker:
    def __init__(self, engine, *, idle_sec: float = 2.0, poll_sec: float = 0.25, budget_sec: float = 45.0,
                 clock=time.monotonic):
        self.engine = engine
        self.idle_sec, self.poll_sec, self.budget_sec, self._clock = idle_sec, poll_sec, budget_sec, clock
        self.started_at = datetime.now(timezone.utc)    # requests answered before this are not in the ledgers
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._job_thread: threading.Thread | None = None
        self._cancel: threading.Event | None = None
        self.stats = {"jobs": 0, "matched": 0, "mismatched": 0, "preempted": 0, "overran": 0, "busy": 0,
                      "not_compared": 0, "dropped": 0, "errors": 0, "last_error": None, "last_job_at": None}

    # -- engine side ----------------------------------------------------------------------------------------
    def note(self, provider, session_id, path, variant, spec, offset, outcome) -> None:
        """After every incremental audit. ``outcome``: ``"match"`` (a comparison of every contract ran and agreed),
        ``"mismatch"`` (it found a difference) or anything else (answered without such a comparison)."""
        key = (provider, session_id)
        with self._lock:
            entry = self._entries.get(key)
            if outcome in ("match", "mismatch"):
                if entry is not None:
                    if outcome == "mismatch":
                        # The difference is recorded (a range for review over both kinds of request, the contracts that
                        # differ now run on the full evaluator): nothing of this session stays pending.
                        del self._entries[key]
                        return
                    entry.ledgers[variant] = _Ledger()
                    entry.failures, entry.retry_at = 0, 0.0
                    if not any(ledger.requests for ledger in entry.ledgers.values()):
                        del self._entries[key]
                return
            if entry is None:
                entry = self._entries[key] = _Entry(path)
            entry.path = path if path is not None else entry.path
            ledger = entry.ledgers[variant]
            at = _now()
            ledger.requests += 1
            ledger.first_at = ledger.first_at or at
            ledger.first_offset = offset if ledger.first_offset is None else ledger.first_offset
            ledger.last_at, ledger.last_offset = at, offset
            if variant == "check":
                ledger.spec = spec

    def since_match(self, provider, session_id) -> dict | None:
        """The requests (of either kind) this process answered since the last match, or ``None`` when none were."""
        with self._lock:
            entry = self._entries.get((provider, session_id))
            ledgers = [l for l in entry.ledgers.values() if l.requests] if entry is not None else []
            if not ledgers:
                return None
            return {"requests": sum(l.requests for l in ledgers), "since_at": min(l.first_at for l in ledgers),
                    "since_offset": min(l.first_offset for l in ledgers),
                    "until_offset": max(l.last_offset for l in ledgers)}

    def ledger_covers(self, since_at) -> bool:
        """Whether this worker was running at ``since_at`` (the time of the last match), so that its ledgers hold every request
        answered since then. After a restart they do not: what came before is known only from the persisted schedule."""
        if not since_at:
            return False
        try:
            moment = datetime.fromisoformat(since_at)
        except ValueError:
            return False
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment >= self.started_at

    def preempt(self) -> None:
        """A request has arrived: cancel the running job (called from the engine for every operation except the worker's)."""
        cancel = self._cancel
        if cancel is not None and threading.current_thread() is not self._job_thread:
            cancel.set()

    def snapshot(self) -> dict:
        """What ``audit_status`` reports."""
        with self._lock:
            pending = sum(1 for entry in self._entries.values() if any(l.requests for l in entry.ledgers.values()))
            unverified = sum(l.requests for entry in self._entries.values() for l in entry.ledgers.values())
            now = self._clock()
            backing_off = sum(1 for entry in self._entries.values() if entry.retry_at > now)
            return {**self.stats, "pending_sessions": pending, "unverified_requests": unverified,
                    "backing_off": backing_off, "running": self._cancel is not None, "idle_sec": self.idle_sec}

    # -- lifecycle --------------------------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="tsukkomi-shadow", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        cancel = self._cancel
        if cancel is not None:
            cancel.set()                      # a closing engine must not wait for a comparison
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        self._job_thread = threading.current_thread()
        while not self._stop.wait(self.poll_sec):
            try:
                self.tick()
            except Exception as exc:          # the worker must outlive any one failure
                self.stats["errors"] += 1
                self.stats["last_error"] = type(exc).__name__

    # -- scheduling -------------------------------------------------------------------------------------------
    def tick(self) -> int:
        """One scheduling round; returns how many comparisons ran. Public for tests (call it from the test's thread)."""
        if not getattr(self.engine, "accepts_background_work", lambda: True)():
            return 0                          # closed or closing: never reopen an engine that was shut down
        if self.engine.idle_seconds() < self.idle_sec:
            return 0
        now = self._clock()
        with self._lock:
            due = [(key, entry) for key, entry in self._entries.items()
                   if now >= entry.retry_at and any(l.requests for l in entry.ledgers.values())]
        if not due:
            return 0
        # In turn: the session tried least recently first, then the one that has waited longest. One that keeps failing
        # backs off and is tried after the others, so it cannot starve them.
        key, entry = min(due, key=lambda item: (item[1].tried_at,
                                                min(l.first_at or "~" for l in item[1].ledgers.values() if l.requests)))
        return self._verify(key, entry)

    def _back_off(self, entry) -> None:
        with self._lock:
            entry.failures += 1
            entry.retry_at = self._clock() + min(BACKOFF_MAX_SEC, BACKOFF_SEC * 2 ** min(entry.failures - 1, 10))

    def _verify(self, key, entry) -> int:
        provider, session_id = key
        ran = 0
        for variant in VARIANTS:
            with self._lock:
                ledger = entry.ledgers[variant]
                if not ledger.requests:
                    continue
                spec, path = ledger.spec, entry.path
            cancel = threading.Event()
            # Published before the idle check, not after it: a request that arrives from here on cancels this job, and one that
            # arrived before is seen by the check (the engine counts a request before it preempts, so on both sides the
            # order is the same and no request is missed).
            self._cancel = cancel
            try:
                if self._stop.is_set() or self.engine.idle_seconds() < self.idle_sec:
                    break                     # a request got there first; nothing started, nothing lost
                if cancel.is_set():
                    self.stats["preempted"] += 1      # one arrived between that check and now
                    break
                if self._job_thread is None:
                    self._job_thread = threading.current_thread()
                self.stats["jobs"] += 1
                entry.tried_at = self._clock()
                info = self.engine.verify_incremental(session_id, provider=provider, path=path, action=spec,
                                                      variant=variant, _deadline=time.monotonic() + self.budget_sec,
                                                      cancel_event=cancel)
            except RequestStopped:
                if cancel.is_set():
                    self.stats["preempted"] += 1      # a request (or the engine closing) got there first: no penalty
                else:
                    self.stats["overran"] += 1        # the job's own budget ran out: do not start it again at once
                    self._back_off(entry)
                break
            except IndexBusy:
                self.stats["busy"] += 1
                self._back_off(entry)
                break
            except Exception as exc:
                self.stats["errors"] += 1
                self.stats["last_error"] = type(exc).__name__
                self._back_off(entry)             # no hot loop on a persistent failure
                break
            finally:
                self._cancel = None
            ran += 1
            self.stats["last_job_at"] = _now()
            if info.get("mismatch"):
                self.stats["mismatched"] += 1
                if info.get("not_recorded"):      # found but not written: the session stays pending, the next job finds it again
                    self._back_off(entry)
                    break
            elif info.get("compared"):
                self.stats["matched"] += 1
            elif info.get("mode") == "full" and not info.get("skipped"):
                # No contract has an incremental path any more, so the audit above was the full evaluator alone: nothing is
                # left to verify here (what was answered before is the business of the mismatch that switched them off).
                self.stats["dropped"] += 1
                with self._lock:
                    entry.ledgers[variant] = _Ledger()
                    if self._entries.get(key) is entry and not any(l.requests for l in entry.ledgers.values()):
                        del self._entries[key]
            else:
                self.stats["not_compared"] += 1   # no time for it, or a time-limited pattern: still pending, later
                self._back_off(entry)
                break
        return ran


def start_shadow_worker(engine) -> ShadowWorker | None:
    """Attach and start the worker of a resident server; None when disabled (``audit_shadow_idle_sec`` 0)."""
    settings = engine.settings
    idle = float(settings.audit_shadow_idle_sec)
    if idle <= 0 or not settings.audit_incremental:
        return None
    worker = ShadowWorker(engine, idle_sec=idle, budget_sec=settings.request_timeout_sec)
    engine.attach_shadow_worker(worker)
    worker.start()
    return worker
