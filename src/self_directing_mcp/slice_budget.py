"""A soft stop for one sync slice: the parse ends at a record boundary and the prefix it read is published.

A source that cannot be synced inside the request budget used to be parsed until the hard deadline, rolled back
and parsed again from byte 0 by the next call, so repeating the call made no progress. A request that may slice
(``index.ingest``) puts a ``SliceBudget`` in scope around the parse. ``session_events.iter_raw_lines`` is the one
place that consults it, before it reads the next record: once the minimum progress is made and the soft deadline has
passed (or a cap is reached) the iterator ends there. Every parser already returns the end of the last record it
consumed, so the cursor, the events, the block table and the metrics of that prefix are published in one
transaction exactly like any other append. The hard deadline (``request_control.check_request``) is untouched.

A request that will judge what it syncs (``audit_session``, ``check_action``) also has to afford the judging of the slice:
it puts a ``Sizing`` in scope (``sizing_scope``) and the slice stops once what it parsed plus what evaluating those
records is expected to cost no longer fits the request (``SliceBudget.cost_budget``). That is a second reason to end
at the same record boundary, decided at the same clock reads: no new reads of the clock, no new kind of stop.

Standard library only: importing this module costs a hook process nothing.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import time

STOP_REASONS = ("soft_deadline", "slice_cap", "record_cap")
CLOCK_EVERY = 64   # records between two reads of the clock: the clock is not read per record


class SliceBudget:
    """What one slice may consume, and (after the parse) why and where it stopped.

    ``start_byte`` is the offset the parse starts at; only the iterator that starts there is subject to the
    budget (a header probe that reads the head of a file is not). ``soft_deadline`` is a ``clock()`` value.
    Minimum progress: a soft deadline is honoured only after ``min_records`` records AND ``min_bytes`` bytes were
    consumed, so a call that arrives with its time spent still moves the cursor. The caps are hard: ``max_bytes``
    (source bytes) and ``max_records`` stop the slice when reached, whatever the minimums say (tests use
    ``max_records`` to force deterministic stops). A slice always holds at least one whole record. (OMP's parse peeks at
    the head of a cold file for its session id with the same iterator; with the production minimums that probe is never
    cut, but a test cap of one record would cut it behind a title line.)

    Cost bound: with a ``cost_budget`` the soft deadline is not the only time limit. A slice that judging will follow
    must leave time to judge itself, and how much that takes grows with the number of records it holds, so the iterator
    also stops once ``(now - started) * commit_factor + records * cost_per_record`` reaches ``cost_budget``. The parse
    speed in that sum is the one measured by this very slice, not a guess. The minimum progress applies to both limits.

    Outputs: ``stopped_by`` (one of ``STOP_REASONS``, ``None`` when the parse ran to the end of what it was given),
    ``bound`` (``"share"`` when the soft deadline ended it, ``"cost"`` when the cost bound did), ``records`` and
    ``consumed`` (the records and source bytes the stopping iterator had yielded) and ``clock_reads``.
    """

    __slots__ = ("start_byte", "soft_deadline", "min_records", "min_bytes", "max_bytes", "max_records", "clock",
                 "stopped_by", "records", "consumed", "clock_reads", "started", "cost_per_record", "cost_budget",
                 "commit_factor", "bound")

    def __init__(self, start_byte=0, *, soft_deadline=None, min_records=256, min_bytes=1 << 20, max_bytes=None,
                 max_records=None, clock=time.monotonic, started=None, cost_per_record=0.0, cost_budget=None,
                 commit_factor=1.0):
        if min_records < 0 or min_bytes < 0 or (max_bytes is not None and max_bytes < 1) \
                or (max_records is not None and max_records < 1) or (cost_budget is not None and started is None):
            raise ValueError("invalid slice budget")
        self.start_byte, self.soft_deadline, self.clock = start_byte, soft_deadline, clock
        self.min_records, self.min_bytes, self.max_bytes, self.max_records = min_records, min_bytes, max_bytes, max_records
        # Cost bound (see ``Sizing``): the slice also ends once the time spent since ``started``, times ``commit_factor``
        # (the commit follows the parse), plus ``cost_per_record`` for every record read, reaches ``cost_budget``.
        self.started, self.cost_per_record, self.cost_budget, self.commit_factor = started, cost_per_record, cost_budget, commit_factor
        self.stopped_by = self.bound = None
        self.records = self.consumed = self.clock_reads = 0

    def look(self, records: int, consumed: int):
        """``(reason, next_look)`` for an iterator about to read the record after ``records`` yielded ones.

        ``reason`` is a stop reason or ``None``; ``next_look`` is the record count at which to ask again, so the
        iterator pays one integer comparison per record, not a call. A slice always holds a whole record: nothing
        stops before the first one.
        """
        if records and self.max_records is not None and records >= self.max_records:
            return "record_cap", records
        if records and self.max_bytes is not None and consumed >= self.max_bytes:
            return "slice_cap", records
        ahead = None                                    # None: nothing left to decide
        if self.soft_deadline is not None:
            floor = max(1, self.min_records)
            if records >= floor and consumed >= self.min_bytes:
                self.clock_reads += 1
                now = self.clock()
                if now >= self.soft_deadline:
                    self.bound = "share"
                    return "soft_deadline", records
                if self.cost_budget is not None and (now - self.started) * self.commit_factor \
                        + records * self.cost_per_record >= self.cost_budget:
                    self.bound = "cost"
                    return "soft_deadline", records
                ahead = records + CLOCK_EVERY
            else:                                       # the minimum progress is not made yet
                ahead = records + CLOCK_EVERY if records >= floor else floor
        if self.max_bytes is not None:
            ahead = records + 1                         # bytes accumulate record by record
        if self.max_records is not None:
            ahead = self.max_records if ahead is None else min(ahead, self.max_records)
        return None, (1 << 62 if ahead is None else max(ahead, records + 1))

    def stop(self, reason: str, records: int, consumed: int) -> None:
        self.stopped_by, self.records, self.consumed = reason, records, consumed


_current: ContextVar[SliceBudget | None] = ContextVar("self_directing_slice_budget", default=None)


def current_slice() -> SliceBudget | None:
    return _current.get()


@contextmanager
def slice_scope(budget: SliceBudget | None):
    """Put ``budget`` in scope for the parse of one slice (``None`` leaves the iterators unbounded)."""
    token = _current.set(budget)
    try:
        yield budget
    finally:
        _current.reset(token)


class Sizing:
    """What the request that syncs also has to afford: judging what the sync publishes.

    ``deadline`` is the request's own deadline (``time.monotonic()``), not the narrower one of its sync, and ``reserve``
    the share of the time that stays for the response. The cost numbers are filled in lazily, by ``resolve``, from
    ``factory()``: ``(eval_per_event, debt_events)`` or ``None`` when nothing is judged after this sync. ``eval_per_event``
    is the expected seconds to evaluate one more event (0 leaves a slice exactly as it was), ``debt_events`` the events
    the index already holds that no monitor state covers yet, which the request has to evaluate first. A request that
    never has anything big to parse never calls the factory (it reads the contracts and the index).
    """

    __slots__ = ("deadline", "reserve", "eval_per_event", "debt_events", "_factory")

    def __init__(self, deadline, factory=None, *, eval_per_event=0.0, debt_events=0, reserve=0.15):
        self.deadline, self.reserve, self._factory = deadline, reserve, factory
        self.eval_per_event, self.debt_events = max(0.0, eval_per_event), max(0, int(debt_events))

    def resolve(self) -> bool:
        """Whether a judging follows (``eval_per_event > 0``); asks the factory once."""
        factory, self._factory = self._factory, None
        if factory is not None:
            numbers = factory()
            if numbers is not None:
                self.eval_per_event, self.debt_events = max(0.0, numbers[0]), max(0, int(numbers[1]))
        return self.eval_per_event > 0


_sizing: ContextVar[Sizing | None] = ContextVar("self_directing_sync_sizing", default=None)


def current_sizing() -> Sizing | None:
    return _sizing.get()


@contextmanager
def sizing_scope(sizing: Sizing | None):
    """Put ``sizing`` in scope for the sync of a request that will judge (``None``: a sync that judges nothing)."""
    token = _sizing.set(sizing)
    try:
        yield sizing
    finally:
        _sizing.reset(token)
