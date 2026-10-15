"""A soft stop for one sync slice: the parse ends at a record boundary and the prefix it read is published.

A source that cannot be synced inside the request budget used to be parsed until the hard deadline, rolled back
and parsed again from byte 0 by the next call, so repeating the call made no progress. A request that may slice
(``index.ingest``) puts a ``SliceBudget`` in scope around the parse. ``session_events.iter_raw_lines`` is the one
place that consults it, before it reads the next record: once the minimum progress is made and the soft deadline has
passed (or a cap is reached) the iterator ends there. Every parser already returns the end of the last record it
consumed, so the cursor, the events, the block table and the metrics of that prefix are published in one
transaction exactly like any other append. The hard deadline (``request_control.check_request``) is untouched.

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

    Outputs: ``stopped_by`` (one of ``STOP_REASONS``, ``None`` when the parse ran to the end of what it was given),
    ``records`` and ``consumed`` (the records and source bytes the stopping iterator had yielded) and
    ``clock_reads``.
    """

    __slots__ = ("start_byte", "soft_deadline", "min_records", "min_bytes", "max_bytes", "max_records", "clock",
                 "stopped_by", "records", "consumed", "clock_reads")

    def __init__(self, start_byte=0, *, soft_deadline=None, min_records=256, min_bytes=1 << 20, max_bytes=None,
                 max_records=None, clock=time.monotonic):
        if min_records < 0 or min_bytes < 0 or (max_bytes is not None and max_bytes < 1) \
                or (max_records is not None and max_records < 1):
            raise ValueError("invalid slice budget")
        self.start_byte, self.soft_deadline, self.clock = start_byte, soft_deadline, clock
        self.min_records, self.min_bytes, self.max_bytes, self.max_records = min_records, min_bytes, max_bytes, max_records
        self.stopped_by = None
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
                if self.clock() >= self.soft_deadline:
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
