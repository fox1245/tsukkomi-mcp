"""Lookups of stored events by id must go through the primary key.

Without table statistics SQLite prefers the two equalities on ``(provider, session_id)`` to an id list, which turns
every 500-id batch into a scan of the whole session. A derived pass over one session then costs the square of its
size: 90,000 events took 29 s, and 7 s once the lookups used the key.
"""
from __future__ import annotations

from test_reliability import SID, message, session, write  # noqa: F401


def _plan(store, sql):
    return " ".join(row[3] for row in store._conn.execute("EXPLAIN QUERY PLAN " + sql))


def test_lookups_by_event_id_use_the_primary_key_not_a_scan_of_the_session(session):
    engine, path = session
    write(path, [message(f"event {i}") for i in range(300)], "a")
    assert engine.sync_session(session_id=SID, embed=False)["ok"]
    store = engine.store
    ids = sorted(store.chunk_ids(SID, "codex"))[:5]
    seen = []
    store._conn.set_trace_callback(seen.append)
    try:
        assert store.live_ids(SID, "codex", ids) == set(ids)
        snapshot = store.capture_evidence(SID, "codex", history=False, anchors=ids)
        assert sorted(chunk_id for chunk_id, _ in snapshot.rows) == ids
    finally:
        store._conn.set_trace_callback(None)
    lookups = [sql for sql in seen if "chunk_id IN (" in sql]
    assert len(lookups) == 2, seen
    for sql in lookups:
        plan = _plan(store, sql)
        assert "sqlite_autoindex_session_events_1" in plan, plan
        assert "events_session" not in plan, plan


def test_a_session_scoped_lookup_never_returns_another_sessions_event(session):
    # The ``+`` that steers the planner must not weaken the scope: the id belongs to this session and provider only.
    engine, path = session
    write(path, [message("only here")], "a")
    assert engine.sync_session(session_id=SID, embed=False)["ok"]
    store = engine.store
    ids = sorted(store.chunk_ids(SID, "codex"))
    assert store.live_ids(SID, "codex", ids) == set(ids)
    assert store.live_ids("other-session", "codex", ids) == set()
    assert store.live_ids(SID, "omp", ids) == set()
    assert store.capture_evidence("other-session", "codex", history=False, anchors=ids).rows == ()
    assert store.capture_evidence(SID, "omp", history=False, anchors=ids).rows == ()
