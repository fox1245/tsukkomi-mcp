"""Audits judge the captured prefix of a growing session; rewrites stay unknown."""
from __future__ import annotations

import json

import pytest

from self_directing_mcp import session_events
from self_directing_mcp.audit import runner as audit_runner
from test_append_capture import _AppendOnFirstFstat, _tamper
from test_index_concurrency import pause_first, policy, session  # noqa: F401
from test_reliability import SID, call, write, session as reliability_session  # noqa: F401


def _audit_with_late(engine, path, monkeypatch, late, *, first="echo safe", method="audit_session", args=()):
    write(path, [call(first)], "a")
    size = path.stat().st_size
    with pause_first(monkeypatch, audit_runner, "audit_contract") as (pool, entered, release):
        pending = pool.submit(getattr(engine, method), SID, *args)
        assert entered.wait(5)
        write(path, [call(late, "late")], "a")
        release.set()
        return size, pending.result(5)


def _late_snippets(value):
    return [e["snippet"] for f in value["findings"] for e in f.get("evidence", []) if "DELETE" in e["snippet"]]


def test_growth_during_audit_is_clean_as_of_the_judged_offset(session, monkeypatch):
    engine, path = session
    policy(engine)
    size, value = _audit_with_late(engine, path, monkeypatch, "DELETE")
    appended = path.stat().st_size - size
    assert value["verdict"] == "clean", value
    cov = value["coverage"]
    assert cov["judged_through_byte"] == size and cov["unread_bytes"] == appended
    assert cov["complete"] is False and "unread_bytes" in cov["issues"]
    assert not _late_snippets(value)
    assert "as of byte" in " ".join(value["notes"])
    # The next audit reviews the appended record.
    assert engine.audit_session(SID)["verdict"] == "violation"


def test_violation_in_captured_range_is_reported_despite_growth(session, monkeypatch):
    engine, path = session
    policy(engine)
    size, value = _audit_with_late(engine, path, monkeypatch, "echo more", first="DELETE")
    assert value["verdict"] == "violation", value
    assert value["coverage"]["judged_through_byte"] == size and value["coverage"]["unread_bytes"] > 0


def test_growth_before_capture_is_judged_as_of_offset(session, monkeypatch):
    engine, path = session
    policy(engine)
    write(path, [call("echo safe")], "a")
    size = path.stat().st_size
    monkeypatch.setattr(session_events, "os", _AppendOnFirstFstat(path, (json.dumps(call("DELETE", "l")) + "\n").encode()))
    value = engine.audit_session(SID)
    monkeypatch.undo()
    assert value["verdict"] == "clean", value
    assert value["coverage"]["judged_through_byte"] == size and value["coverage"]["unread_bytes"] > 0


def test_completion_obligation_absence_stays_unknown_with_unread_bytes(session, monkeypatch):
    engine, path = session
    engine.upsert_contracts([{"id": "needs-test", "type": "must", "scope": "tool_call", "regex": "npm test"}])
    _, value = _audit_with_late(engine, path, monkeypatch, "npm test")
    assert value["verdict"] == "unknown", value
    assert value["coverage"]["unread_bytes"] > 0
    assert engine.audit_session(SID)["verdict"] == "clean"


def test_obligation_evidenced_in_prefix_is_clean_as_of_offset(session, monkeypatch):
    engine, path = session
    engine.upsert_contracts([{"id": "needs-test", "type": "must", "scope": "tool_call", "regex": "npm test"}])
    _, value = _audit_with_late(engine, path, monkeypatch, "echo more", first="npm test")
    assert value["verdict"] == "clean", value
    assert value["coverage"]["unread_bytes"] > 0


def test_gate_prerequisite_absence_for_proposed_action_is_unknown_with_growth(session, monkeypatch):
    engine, path = session
    engine.upsert_contracts([{"id": "test-first", "type": "must", "scope": "tool_call", "regex": "npm test",
                              "before_regex": "deploy"}])
    action = {"tool_name": "shell", "arguments": {"command": "deploy now"}}
    _, value = _audit_with_late(engine, path, monkeypatch, "npm test", method="check_action", args=(action,))
    assert value["verdict"] == "unknown", value


@pytest.mark.parametrize("kind", ["overwrite", "overwrite_append", "truncate", "replace", "delete"])
def test_rewrite_during_audit_is_unknown(session, monkeypatch, kind):
    engine, path = session
    policy(engine)
    write(path, [call("echo safe first")], "a")
    with pause_first(monkeypatch, audit_runner, "audit_contract") as (pool, entered, release):
        pending = pool.submit(engine.audit_session, SID)
        assert entered.wait(5)
        _tamper(path, kind)
        release.set()
        value = pending.result(5)
    assert value["verdict"] == "unknown", value
    assert "source_changed_during_evaluation" in value["coverage"]["issues"]
    assert value["coverage"]["complete"] is False and "judged_prefix" not in value["coverage"]


def test_unrelated_incompleteness_is_not_masked_by_growth(session, monkeypatch):
    engine, path = session
    policy(engine)
    with path.open("a") as stream:
        stream.write("{not json\n")
    _, value = _audit_with_late(engine, path, monkeypatch, "echo more")
    assert value["verdict"] == "unknown", value
    assert "parse_errors" in value["coverage"]["issues"]
