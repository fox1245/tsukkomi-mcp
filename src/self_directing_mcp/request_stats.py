"""Request statistics of this process: how each tool's requests ended, how long they waited and ran, and how many
workers a deadline left running.

One short lock protects plain counters. Nothing here touches the engine, its lock or the index, so ``audit_status`` can
report the numbers while the engine is busy. They live in memory only (a restart starts at zero) and carry no content:
labels come from closed vocabularies (anything else is counted as ``other``) and the rest are integers and floats. No
session id, path, contract text, argument or exception message is ever stored.
"""
from __future__ import annotations

from bisect import bisect_left
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import threading
import time

OTHER = "other"

# The registered MCP tools, plus ``hook`` for the Codex lifecycle tool (its requests are counted as hooks).
TOOLS = (
    "sync_session", "search_history", "audit_session", "check_action", "get_chunk", "list_contracts",
    "revoke_contract", "upsert_contracts", "audit_status", "update_checklist", "get_checklist", "analyze_activity",
    "get_graph_context", "propose_graph_update", "commit_graph_update", "run_neograph_update",
    "graph_next_batch", "graph_submit_batch",
    "workflow_status", "workflow_run_checks", "workflow_classify", "workflow_transition", "workflow_authorize",
    "workflow_execute", "codex_session_hook", "hook",
)
# How a request ended: answered; answered from less evidence than wanted (the sync ran out of its share of the time);
# no answer (a deadline, a busy index, a cancellation inside the engine); the client went away; an unexpected failure.
OUTCOMES = ("ok", "degraded", "incomplete", "client_cancelled", "error")
REASONS = ("request_deadline", "index_busy", "request_cancelled", "operation_timeout", "sync_deadline", OTHER)
WITH_REASON = ("incomplete", "degraded")        # the outcomes that carry a reason
BUCKETS_MS = (50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 45000)    # upper bounds, inclusive; one more bucket: +inf
_MAX_MS = 86_400_000.0                           # a day: keeps a corrupt measurement a finite number


def fold(value, allowed) -> str:
    """``value`` when it is one of the closed ``allowed`` names, else ``other``. Any other type folds too."""
    return value if isinstance(value, str) and value in allowed else OTHER


def tool_label(method) -> str:
    """The closed tool name of a dispatched method (its ``__name__``), or ``other``."""
    return fold(getattr(method, "__name__", None), TOOLS)


def _milliseconds(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if value != value or value < 0 else min(value, _MAX_MS)    # NaN and negative values count as zero


class _Histogram:
    __slots__ = ("counts", "maximum")

    def __init__(self):
        self.counts = [0] * (len(BUCKETS_MS) + 1)
        self.maximum = 0.0

    def add(self, milliseconds: float) -> None:
        self.counts[bisect_left(BUCKETS_MS, milliseconds)] += 1
        self.maximum = max(self.maximum, milliseconds)

    def export(self) -> dict:
        return {"counts": list(self.counts), "max": round(self.maximum, 1)}


class _Tool:
    __slots__ = ("calls", "outcomes", "reasons", "elapsed", "gate_wait")

    def __init__(self):
        self.calls = 0
        self.outcomes = dict.fromkeys(OUTCOMES + (OTHER,), 0)
        self.reasons = {outcome: {} for outcome in WITH_REASON}
        self.elapsed, self.gate_wait = _Histogram(), _Histogram()

    def export(self) -> dict:
        return {"calls": self.calls, "outcomes": dict(self.outcomes),
                "reasons": {outcome: dict(reasons) for outcome, reasons in self.reasons.items()},
                "elapsed_ms": self.elapsed.export(), "gate_wait_ms": self.gate_wait.export()}


class RequestStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._tools: dict[str, _Tool] = {}
            self._abandoned_now = self._abandoned_total = self._late_finish_total = 0
            self._slots = self._in_use = 0
            self._since = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._started = time.monotonic()

    def record(self, tool, outcome, *, reason=None, elapsed_ms=0.0, gate_wait_ms=None) -> None:
        """Count one request, at its response. ``gate_wait_ms`` is None for a call that never queued for a heavy-call
        slot (it bypasses the gate): such a call adds nothing to the wait histogram."""
        tool, outcome = fold(tool, TOOLS), fold(outcome, OUTCOMES)
        reason = fold(reason, REASONS) if outcome in WITH_REASON else None
        elapsed = _milliseconds(elapsed_ms)
        waited = None if gate_wait_ms is None else _milliseconds(gate_wait_ms)
        with self._lock:
            entry = self._tools.get(tool)
            if entry is None:
                entry = self._tools[tool] = _Tool()
            entry.calls += 1
            entry.outcomes[outcome] += 1
            if reason is not None:
                entry.reasons[outcome][reason] = entry.reasons[outcome].get(reason, 0) + 1
            entry.elapsed.add(elapsed)
            if waited is not None:
                entry.gate_wait.add(waited)

    def worker_abandoned(self) -> None:
        """A worker thread is still running although its request was answered ``request_deadline``."""
        with self._lock:
            self._abandoned_now += 1
            self._abandoned_total += 1

    def worker_finished_late(self) -> None:
        """An abandoned worker has ended. It is not a request: only the worker counters hear of it."""
        with self._lock:
            self._abandoned_now = max(0, self._abandoned_now - 1)
            self._late_finish_total += 1

    def gate_acquired(self, slots: int) -> None:
        with self._lock:
            self._slots = slots
            self._in_use += 1

    def gate_released(self) -> None:
        with self._lock:
            self._in_use = max(0, self._in_use - 1)

    def snapshot(self) -> dict:
        """Plain JSON-able data: closed names, integers, floats and the ISO time the statistics began."""
        with self._lock:
            return {
                "since": self._since,
                "uptime_sec": round(time.monotonic() - self._started, 1),
                "buckets_ms": [*BUCKETS_MS, "+inf"],
                "tools": {name: self._tools[name].export() for name in sorted(self._tools)},
                "workers": {"abandoned_now": self._abandoned_now, "abandoned_total": self._abandoned_total,
                            "late_finish_total": self._late_finish_total},
                "gate": {"slots": self._slots, "in_use": self._in_use},
            }


STATS = RequestStats()


class Notes:
    """What the engine noticed while one request ran, for the dispatcher to read at the response."""
    __slots__ = ("degraded",)

    def __init__(self):
        self.degraded = None


_notes: ContextVar[Notes | None] = ContextVar("self_directing_request_notes", default=None)


@contextmanager
def collecting(notes: Notes):
    token = _notes.set(notes)
    try:
        yield notes
    finally:
        _notes.reset(token)


def note_degraded(reason) -> None:
    """The engine answered from less evidence than it wanted (an evidence refresh that ran out of its share of the time).
    Only the first reason of a request is kept; outside a dispatched request there is nobody to tell."""
    notes = _notes.get()
    if notes is not None and notes.degraded is None:
        notes.degraded = fold(reason, REASONS)
