"""A Stop checkpoint verifies the transcript in full once, not twice.

Synthetic data only. The checkpoint syncs the history and then audits it; the audit's own refresh verifies the whole source
(a restart of the agent rests on that verification), so the sync before it only has to take in the new records and can
check the source the cheap way (the tail, the latest blocks and a sample against the stored block digests).
"""
from __future__ import annotations

from self_directing_mcp.index import ingest

from test_claude_hooks import ACK, SID, assistant, env, human, main_file, stop, tool_result, use, write  # noqa: F401


def test_a_stop_scans_the_whole_transcript_once_and_checks_the_rest_the_cheap_way(env, monkeypatch):
    engine, path = env
    engine.upsert_contracts([ACK])
    for n in range(40):
        write(path, assistant(use(f"c{n}", "Bash", command="echo " + "x" * 400)), tool_result(f"c{n}", "ok"), mode="a")
    stop(engine, path)                                   # the first checkpoint indexes and verifies everything
    calls = {"full": 0, "fast": 0}
    full, fast = ingest.capture_for_cursor, ingest.capture_source_fast
    monkeypatch.setattr(ingest, "capture_for_cursor", lambda *a, **k: (calls.update(full=calls["full"] + 1), full(*a, **k))[1])
    monkeypatch.setattr(ingest, "capture_source_fast", lambda *a, **k: (calls.update(fast=calls["fast"] + 1), fast(*a, **k))[1])
    write(path, assistant(use("last", "Bash", command="echo more")), tool_result("last", "ok"), mode="a")
    stop(engine, path)
    assert calls == {"full": 1, "fast": 1}, calls
