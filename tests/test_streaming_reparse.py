"""Full parses spool rows to disk instead of holding every chunk."""
from __future__ import annotations

import json
import tracemalloc

from test_reliability import SID, message, write, session  # noqa: F401


def test_first_sync_of_large_session_streams_rows(session):
    engine, path = session
    line = json.dumps(message("x" * 1000)) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        for i in range(16000):
            stream.write(line.replace("xxx", f"{i:03d}", 1))
    size = path.stat().st_size
    assert size > 15_000_000
    tracemalloc.start()
    try:
        info = engine.sync_session(session_id=SID, embed=False)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert info["ok"] and info["total_chunks"] >= 16000
    assert peak < 0.8 * size, (peak, size)
