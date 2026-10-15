"""An incremental audit after a small append reads a number of stored rows that does not depend on the session size.

Wall time is not asserted (it varies with the machine); the rows the audit pulls out of SQLite are counted instead,
for a session and for a session four times as large, against the full evaluator reading all of them.
"""
from __future__ import annotations

import pytest

from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from test_reliability import SID, call, message, result, write

# These tests measure the delta path alone (shadow_every=0): no warm-up comparisons after a rebuild.
pytestmark = pytest.mark.usefixtures("no_forced_comparisons")

RULES = [
    {"id": "no-delete", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"},
    {"id": "tests-first", "type": "must", "scope": "tool_call", "regex": "npm test", "before_regex": "deploy",
     "requires_success": True},
    {"id": "linted", "type": "must", "scope": "tool_call", "regex": "eslint", "when": {"tool_call_regex": "npm test"}},
    {"id": "said-summary", "type": "must", "scope": "message", "regex": "summary", "roles": ["assistant"]},
]


def unit(i):
    command = "npm test" if i % 40 == 0 else "eslint ." if i % 97 == 0 else f"echo {i}"
    return [message(f"narration {i}" + (" summary" if i % 500 == 0 else "")), call(command, f"c{i}"),
            result(f"c{i}", 0)]


def session(tmp_path, units, *, incremental):
    root = tmp_path / ("sessions-%d-%s" % (units, incremental))
    root.mkdir()
    path = root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl"
    write(path, [{"type": "session_meta", "payload": {"id": SID}}])
    for start in range(0, units, 500):
        write(path, [record for i in range(start, min(units, start + 500)) for record in unit(i)], "a")
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, codex_sessions_dir=root,
                                       index_dir=tmp_path / ("index-%d-%s" % (units, incremental)),
                                       audit_incremental=incremental, audit_shadow_every=0))
    engine.upsert_contracts(RULES)
    return engine, path


def rows_for_an_append(engine, path, units, appended_units):
    engine.audit_session(SID, provider="codex")  # builds (or, without incremental, just evaluates)
    engine.audit_session(SID, provider="codex")
    write(path, [record for i in range(units, units + appended_units) for record in unit(i)], "a")
    before = engine.store.audit_rows_fetched
    answer = engine.audit_session(SID, provider="codex")
    return engine.store.audit_rows_fetched - before, answer


@pytest.mark.parametrize("units", [1500, 6000])
def test_rows_read_by_an_incremental_audit_do_not_grow_with_the_session(tmp_path, units):
    appended_units = 20  # 60 records
    fast, fast_path = session(tmp_path, units, incremental=True)
    rows, answer = rows_for_an_append(fast, fast_path, units, appended_units)
    assert answer["incremental"]["mode"] == "delta" and answer["incremental"]["events_read"] == 3 * appended_units
    # the appended rows, plus the earlier calls and results that share a call id with an appended call
    assert rows <= 3 * appended_units * 3, (rows, units)
    full, full_path = session(tmp_path, units, incremental=False)
    full_rows, reference = rows_for_an_append(full, full_path, units, appended_units)
    assert full_rows >= 3 * units  # the old path reads the whole session
    assert {k: v for k, v in answer.items() if k in ("verdict", "findings")} == {
        k: v for k, v in reference.items() if k in ("verdict", "findings")}
    fast.close()
    full.close()


def test_an_untouched_session_costs_no_event_rows_at_all(tmp_path):
    engine, path = session(tmp_path, 2000, incremental=True)
    engine.audit_session(SID, provider="codex")
    before = engine.store.audit_rows_fetched
    answer = engine.audit_session(SID, provider="codex")
    assert engine.store.audit_rows_fetched == before and answer["incremental"]["events_read"] == 0
    engine.close()


def vm_steps(connection, run):
    """SQLite virtual-machine work done by ``run`` (progress callbacks every 1000 instructions)."""
    steps = []
    connection.set_progress_handler(lambda: steps.append(1) and 0, 1000)
    try:
        run()
    finally:
        connection.set_progress_handler(None, 0)
    return len(steps)


def test_the_store_reads_for_an_append_are_served_by_indexes(tmp_path):
    engine, _path = session(tmp_path, 3000, incremental=True)  # 9,000 events
    engine.audit_session(SID, provider="codex")
    store = engine.store
    needs = [("tool_call", ("assistant",))]
    whole = vm_steps(store._conn, lambda: store.capture_audit_events(SID, "codex", needs, link_calls=True))
    last = store._conn.execute("SELECT MAX(byte_start) FROM session_events WHERE session_id=?", (SID,)).fetchone()[0]
    with store._read_boundary():
        tail = vm_steps(store._conn, lambda: store.capture_audit_tail(SID, "codex", needs, link_calls=True,
                                                                      from_byte=last - 1000))
        lookup = vm_steps(store._conn, lambda: store.audit_call_events(SID, "codex", ["c1", "c2", "c999"], 10**12))
    assert whole > 100 and tail * 20 < whole and lookup * 50 < whole, (whole, tail, lookup)
    engine.close()
