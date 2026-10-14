"""Incremental audits over a real store: delta reads, persisted state, invalidation and rebuilds.

Every audit goes through ``Harness.audit``, which also evaluates the same stored events with the full evaluator and
fails on any difference in findings, verdict, notes, coverage or graph trace.
"""
from __future__ import annotations

import random

import pytest

from self_directing_mcp.audit import incremental as inc
from self_directing_mcp.index.ingest import SessionStore
from self_directing_mcp.schemas import AuditEvent, ContractRule
from incremental_store import SID, Harness
from incremental_support import Session, contracts, proposed_action

PROVIDERS = ["codex", "claude", "agy", "omp"]


def gate(rules):
    """What the engine keeps of the contracts when a proposed action is checked."""
    return [r for r in rules if r.type != "must" or r.before_regex]


def grow(provider, rng, count, **options):
    session = Session(provider, rng, **options)
    session.extend(count)
    return session


def batches(rng, events):
    """Whole records only: the events of one record (same byte start) are never split across commits."""
    groups = []
    for event in events:
        if groups and groups[-1][0].key[0] == event.key[0]:
            groups[-1].append(event)
        else:
            groups.append([event])
    out, index = [], 0
    while index < len(groups):
        size = rng.choice([1, 1, 2, 3, 5, 9])
        out.append([e for group in groups[index:index + size] for e in group])
        index += size
    return out


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("seed", range(10))
def test_random_appends_match_the_full_evaluator(tmp_path, provider, seed):
    rng = random.Random(f"store/{provider}/{seed}")
    session = grow(provider, rng, 90)
    rules = contracts(rng, provider, session.events, 6, descriptions=0.2)
    harness = Harness(tmp_path, provider)
    reports = []
    for step, batch in enumerate(batches(rng, session.events)):
        harness.commit(batch)
        result, report, _plan = harness.audit(rules)
        reports.append(report)
        if rng.random() < 0.4:
            action = proposed_action(rng)
            harness.audit(gate(rules), planned=action)
        assert result.session_id == SID
    assert reports[0]["mode"] == "full" and set(reports[0]["rebuilt"].values()) <= {"no_state"}
    modes = {r["mode"] for r in reports[1:]}
    assert "delta" in modes or len(reports) < 3, reports
    deferred = [r["fallback"] for r in reports if r.get("fallback")]
    assert not deferred, deferred


store_cap = SessionStore.PROPOSAL_ROWS


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_normal_append_keeps_the_state_and_reads_only_the_appended_rows(tmp_path, provider):
    rng = random.Random(f"delta/{provider}")
    session = grow(provider, rng, 400, deny_mismatch=0.0)
    rules = contracts(rng, provider, session.events, 6)
    harness = Harness(tmp_path, provider)
    batch_list = batches(rng, session.events)
    harness.commit([e for batch in batch_list[:-4] for e in batch])
    _result, first, _ = harness.audit(rules)
    assert first["mode"] == "full"
    for batch in batch_list[-4:]:
        harness.commit(batch)
        before = harness.store.audit_rows_fetched
        _result, report, plan = harness.audit(rules)
        # The harness itself reads the whole session once for the reference; the plan's own reads are what counts.
        fetched = harness.store.audit_rows_fetched - before - len(harness.events)
        assert report["mode"] == "delta" and not report["rebuilt"], report
        assert report["events_read"] == len(batch), report
        # Appended rows, the calls and results they refer to, and (only with a denial) the newest proposals of a tool.
        assert fetched <= 5 * len(batch) + 8 + 2 * store_cap, (fetched, len(batch))


def test_an_audit_with_nothing_appended_reads_no_events(tmp_path):
    rng = random.Random("nothing")
    session = grow("codex", rng, 300)
    rules = contracts(rng, "codex", session.events, 5)
    harness = Harness(tmp_path, "codex")
    harness.commit(session.events)
    harness.audit(rules)
    _result, report, plan = harness.audit(rules)
    assert report["mode"] == "delta" and report["events_read"] == 0 and plan.state_rows() == [] and not report["rebuilt"]


# --- what invalidates a state ---------------------------------------------------------------------------------

def warm(tmp_path, provider="codex", seed=3, events=120, rules=6):
    rng = random.Random(f"warm/{provider}/{seed}")
    session = grow(provider, rng, events + 40)
    rule_list = contracts(rng, provider, session.events, rules)
    harness = Harness(tmp_path, provider)
    harness.commit(session.events[:events])
    harness.audit(rule_list)
    return rng, session, rule_list, harness, events


def test_a_contract_revision_change_rebuilds_that_contract_only(tmp_path):
    rng, session, rules, harness, n = warm(tmp_path)
    changed = rules[2].model_copy(update={"revision": rules[2].revision + 1, "regex": "alpha|beta"})
    rules[2] = changed
    harness.commit(session.events[n:n + 10])
    _r, report, _ = harness.audit(rules)
    assert report["rebuilt"] == {changed.id: "contract_revision"}, report
    _r, report, _ = harness.audit(rules)
    assert report["mode"] == "delta" and not report["rebuilt"], report


def test_an_edited_contract_with_the_same_revision_is_not_trusted(tmp_path):
    rng, session, rules, harness, n = warm(tmp_path)
    rules[1] = rules[1].model_copy(update={"regex": "alpha beta gamma"})
    harness.commit(session.events[n:n + 5])
    _r, report, _ = harness.audit(rules)
    assert report["rebuilt"] == {rules[1].id: "contract_changed"}, report


def test_adding_a_contract_builds_only_the_new_monitor(tmp_path):
    rng, session, rules, harness, n = warm(tmp_path)
    # Same scope and roles as an existing rule: no event needs more text than before.
    extra = rules[0].model_copy(update={"id": "added", "regex": "gamma", "before_regex": None, "requires_success": False,
                                        "when": None, "exception_regex": None, "source_event_id": None,
                                        "applies_from_event_id": None})
    harness.commit(session.events[n:n + 5])
    _r, report, _ = harness.audit([*rules, extra])
    assert report["rebuilt"] == {"added": "no_state"}, report
    _r, report, _ = harness.audit([*rules, extra])
    assert report["mode"] == "delta" and not report["rebuilt"], report


def test_a_contract_that_needs_more_text_rebuilds_every_state_once(tmp_path):
    rng, session, rules, harness, n = warm(tmp_path)
    needs = {(r.scope, tuple(sorted(r.roles))) for r in rules}
    extra = ContractRule(id="added", type="must_not", scope="message", regex="gamma", roles=["developer"])
    assert ("message", ("developer",)) not in needs
    harness.commit(session.events[n:n + 5])
    _r, report, _ = harness.audit([*rules, extra])
    assert report["mode"] == "full" and set(report["rebuilt"].values()) == {"no_state"}, report
    _r, report, _ = harness.audit([*rules, extra])
    assert report["mode"] == "delta" and not report["rebuilt"], report


@pytest.mark.parametrize("kind", ["reparse", "tamper", "path", "parser"])
def test_a_rewrite_of_the_stored_history_invalidates_every_state(tmp_path, kind):
    rng, session, rules, harness, n = warm(tmp_path)
    tail = session.events[n:n + 6]
    if kind == "reparse":
        harness.commit(session.events[:n] + tail, replace=True)
    elif kind == "tamper":
        events = list(session.events[:n])
        victim = events[len(events) // 2]
        events[len(events) // 2] = type(victim)(victim.chunk_id, victim.kind, "gamma deploy rewritten", victim.meta, victim.key)
        harness.commit(events + tail, replace=True)
    elif kind == "path":
        harness.commit(tail, path=tmp_path / "moved.jsonl")
    else:
        harness.commit(tail, parser_version=999)
    _r, report, _ = harness.audit(rules)
    # A rewrite publishes a new generation and drops the old states in the same transaction ("no_state"); a moved
    # path or a new parser leaves them in place to be rejected by their own key.
    expected = {"reparse": "no_state", "tamper": "no_state", "path": "path", "parser": "parser_version"}[kind]
    reasons = set(report["rebuilt"].values())
    assert report["mode"] == "full" and expected in reasons and reasons <= {expected, "no_state"}, report
    _r, report, _ = harness.audit(rules)
    if kind in ("reparse", "tamper"):
        assert report["mode"] == "delta" and not report["rebuilt"], report


def test_an_append_that_overwrites_a_stored_event_is_a_rewrite(tmp_path):
    rng, session, rules, harness, n = warm(tmp_path)
    victim = session.events[3]
    forged = type(victim)(victim.chunk_id, victim.kind, "deploy deploy forged", victim.meta, victim.key)
    harness.commit([forged, *session.events[n:n + 3]])  # not replace: an append that reuses a stored id
    _r, report, _ = harness.audit(rules)
    assert report["mode"] == "full" and set(report["rebuilt"].values()) == {"no_state"}, report


@pytest.mark.parametrize("stamped", [False, True])
def test_a_state_never_outlives_a_foreign_cursor_move(tmp_path, stamped):
    rng, session, rules, harness, n = warm(tmp_path)
    harness.commit(session.events[n:n + 4])
    with harness.store._conn:
        # An older process publishing a cursor leaves no commit-log row behind (and no parser stamp).
        harness.store._conn.execute("UPDATE session_cursors SET synced_at='foreign', prefix_hash='x' WHERE session_id=?", (SID,))
        if stamped:
            harness.store._conn.execute("UPDATE session_parser_versions SET synced_at='foreign' WHERE session_id=?", (SID,))
    _r, report, _ = harness.audit(rules)
    expected = "lineage_unknown" if stamped else "parser_version"
    assert report["mode"] == "full" and set(report["rebuilt"].values()) == {expected}, report


def test_check_action_and_audit_session_keep_separate_states(tmp_path):
    rng, session, rules, harness, n = warm(tmp_path)
    rules = [r for r in rules] + [ContractRule(id="gate", type="must", scope="tool_call", regex="test",
                                               before_regex="deploy", roles=["assistant"])]
    harness.commit(session.events[n:n + 3])
    harness.audit(rules)
    action = proposed_action(rng)
    _r, first, _ = harness.audit(gate(rules), planned=action)
    assert first["mode"] == "full"
    harness.commit(session.events[n + 3:n + 7])
    _r, audit_report, _ = harness.audit(rules)
    _r, action_report, _ = harness.audit(gate(rules), planned=action)
    assert audit_report["mode"] == "delta" and action_report["mode"] == "delta", (audit_report, action_report)
    assert not audit_report["rebuilt"] and not action_report["rebuilt"]


# --- capacity: a summary that cannot hold an input defers to the full evaluator, visibly --------------------

def codex_with(harness, events):
    harness.commit(events)


def test_a_full_exception_store_defers_to_the_reference_and_is_remembered(tmp_path, monkeypatch):
    monkeypatch.setattr(inc, "EXCEPTION_MAX_HITS", 3)
    harness = Harness(tmp_path, "codex")
    rng = random.Random(5)
    session = grow("codex", rng, 200)
    rule = ContractRule(id="ex", type="must_not", scope="any", regex="alpha", exception_regex="beta \\[tool",
                        roles=["assistant", "user", "tool", "runtime"])
    harness.commit(session.events[:60])
    assert sum("alpha" in e.text for e in session.events[:60]) > 3
    _r, report, _ = harness.audit([rule])
    assert report["mode"] == "full"  # the first audit meets the cap while feeding and answers from the reference
    harness.commit(session.events[60:90])
    # The cap is remembered: the next audit says so up front and evaluates in full without retrying.
    _r, report, plan = harness.audit([rule])
    assert plan is None and report["mode"] == "full", report
    assert report["fallback"]["ex"].startswith("overflow: exception hit store is full"), report


def test_a_delta_run_that_meets_a_capacity_limit_asks_for_a_full_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(inc, "EXCEPTION_MAX_HITS", 2)
    harness = Harness(tmp_path, "codex")
    rng = random.Random(9)
    session = grow("codex", rng, 120)
    rule = ContractRule(id="ex", type="must_not", scope="any", regex="alpha|beta", exception_regex="gamma",
                        roles=["assistant", "user", "tool", "runtime"])
    harness.commit([e for e in session.events[:15] if "alpha" not in e.text and "beta" not in e.text])
    harness.audit([rule])
    harness.commit(session.events[15:80])
    with pytest.raises(inc.PlanRetry):
        harness.audit([rule], retry=False)
    _r, report, _ = harness.audit([rule])  # the engine's second pass
    assert report["retry"]["rules"] == ["ex"] and report["fallback"] == {"ex": "fallback"}


def running(step):
    return AuditEvent(f"agy:s:run{step}", "tool_result", "[tool_result] still going",
                      {"role": "tool", "call_id": None, "success": None, "event_type": "GENERIC", "status": "RUNNING",
                       "pending": True, "step_index": step, "linkage": "pending"}, (step * 100, 0))


def test_the_pending_table_is_bounded_and_the_overflow_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(inc, "PENDING_MAX", 3)
    harness = Harness(tmp_path, "agy")
    rule = ContractRule(id="need", type="must", scope="tool_call", regex="alpha", roles=["assistant"])
    reports = []
    for step in range(1, 9):
        harness.commit([running(step)])
        _r, report, _ = harness.audit([rule])  # the harness fails if any answer differs from the full evaluator
        reports.append(report)
    # Three RUNNING results fit; the fourth overflows the table, which is remembered and named, never dropped.
    assert [r["mode"] for r in reports[:3]] == ["full", "delta", "delta"]
    assert reports[3]["retry"]["reason"] == "pending result table is full", reports[3]
    # Remembered: later audits name the overflow up front and no longer attempt an incremental pass.
    assert all("pending result table is full" in str(r["fallback"]) and r["mode"] == "full" and "retry" not in r
               for r in reports[4:]), reports


# --- the lineage -----------------------------------------------------------------------------------------------

def test_the_commit_log_chains_appends_and_restarts_on_a_rewrite(tmp_path):
    rng = random.Random(21)
    harness = Harness(tmp_path, "codex")
    session = grow("codex", rng, 60)
    for batch in batches(rng, session.events)[:5]:
        harness.commit(batch)
    store = harness.store
    with store._read_boundary():
        lineage = store.audit_lineage(SID, "codex")
        generation = lineage["generation"]
        rows = store._conn.execute("SELECT byte_offset FROM session_commit_log WHERE session_id=?", (SID,)).fetchall()
    assert lineage["latest"]["byte_offset"] == lineage["cursor"]["byte_offset"] and len(rows) == 5
    harness.commit(session.events[:10], replace=True)
    with store._read_boundary():
        again = store.audit_lineage(SID, "codex")
        count = store._conn.execute("SELECT COUNT(*) FROM session_commit_log WHERE session_id=?", (SID,)).fetchone()[0]
    assert again["generation"] == generation + 1 and count == 1


def test_a_state_write_is_refused_when_the_index_moved(tmp_path):
    rng, session, rules, harness, n = warm(tmp_path)
    harness.commit(session.events[n:n + 3])
    result, report, plan = harness.audit(rules, save=False)
    harness.commit(session.events[n + 3:n + 6], replace=False)
    harness.store._conn.execute("DELETE FROM session_commit_log WHERE session_id=? AND byte_offset=?", (SID, plan.snapshot["offset"]))
    harness.store._conn.commit()
    assert harness.store.save_audit_states(SID, "codex", plan.digest, plan.state_rows(), plan.snapshot) == "stale_lineage"
