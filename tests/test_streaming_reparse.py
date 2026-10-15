"""Full parses spool rows to disk instead of holding every chunk."""
from __future__ import annotations

import json
import sys
import tracemalloc

from test_reliability import SID, message, write, session  # noqa: F401


def settle_the_intern_table():
    """Keep an allocation of the interpreter out of the window the test measures.

    CPython keeps every interned string in one table whose storage doubles when it fills, and pathlib interns each path
    component it sees, so the table grows as the suite runs. A doubling that lands inside the window adds its whole
    allocation (3.8 MB at one point of this suite, over a third of the peak) to the peak of a sync that did not make it.
    Interning until the table has just doubled puts the next doubling far away from any window."""
    tracemalloc.start()
    try:
        previous = tracemalloc.get_traced_memory()[0]
        for number in range(400_000):
            sys.intern(f"settle-the-intern-table-{number}")
            current = tracemalloc.get_traced_memory()[0]
            if current - previous > 500_000:
                return
            previous = current
    finally:
        tracemalloc.stop()


def test_first_sync_of_large_session_streams_rows(session):
    engine, path = session
    line = json.dumps(message("x" * 1000)) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        for i in range(16000):
            stream.write(line.replace("xxx", f"{i:03d}", 1))
    size = path.stat().st_size
    assert size > 15_000_000
    settle_the_intern_table()
    tracemalloc.start()
    try:
        info = engine.sync_session(session_id=SID, embed=False)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert info["ok"] and info["total_chunks"] >= 16000
    assert peak < 0.8 * size, (peak, size)
