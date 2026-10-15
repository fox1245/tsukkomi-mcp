"""Graph update batches: the limits a batch has and the cursor arithmetic of a graph commit.

``run_neograph_update`` cuts a session into batches: at most ``BATCH_MAX_EVENTS`` complete events whose size,
counted by ``batch_event_cost`` on the unmasked event, stays within ``BATCH_MAX_CHARS``. An event is never cut.
``cursor_start``, ``covers_through_order`` and ``through_order_problem`` say where a graph commit may move the cursor.

Everything here is pure: no index, no model, no network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BATCH_MAX_EVENTS = 100
BATCH_MAX_CHARS = 95_000
BATCH_EVENT_OVERHEAD = 80           # what one event costs in the prompt besides its text and its chunk id


def batch_event_cost(frame: dict[str, Any]) -> int:
    """What an unmasked event counts against ``BATCH_MAX_CHARS``."""
    return len(frame["text"]) + len(frame["chunk_id"]) + BATCH_EVENT_OVERHEAD


def cursor_start(cursor: dict[str, int]) -> int:
    """The order of the first event not yet processed. A graph at version 0 has processed nothing, whatever its cursor."""
    return cursor["processed_through_order"] + 1 if cursor["graph_version"] else 0


def covers_through_order(count: int, cursor: dict[str, int]) -> int:
    """The last event order a ``commit_graph_update`` without ``through_order`` moves the cursor to.

    That is the end of the whole session, not the end of the events the proposal was built from.
    """
    return count - 1 if count else cursor["processed_through_order"]


def through_order_problem(through_order: Any, cursor: dict[str, int], count: int) -> str | None:
    """Why ``through_order`` cannot be the new cursor, or None: it must lie in ``[cursor_start, last event]``."""
    if isinstance(through_order, bool) or not isinstance(through_order, int):
        return "through_order must be an integer"
    first, last = cursor_start(cursor), count - 1
    if through_order < first:
        return (f"through_order {through_order} does not advance the cursor: the first event not yet processed "
                f"is {first}")
    if through_order > last:
        return f"through_order {through_order} is beyond the last event of the session ({last})"
    return None


@dataclass(frozen=True)
class BatchRead:
    """One consistent read of the next batch: the graph cursor, the event count and the raw (unmasked) frames."""
    cursor: dict[str, int]          # as stored: processed_through_order, graph_version
    cursor_start: int               # first event not yet processed
    total: int                      # events of the session
    start: int                      # order of the first frame of this batch (cursor_start, or after_order + 1)
    frames: tuple[dict[str, Any], ...]
    blocked: dict[str, Any] | None  # the first event that did not fit the budget (never cut), if the batch stopped there
