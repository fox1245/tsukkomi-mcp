"""A historical prefix is evidence, not proof of a current action prerequisite."""
from __future__ import annotations

import json

import pytest

from self_directing_mcp import session_events
from self_directing_mcp.agy_hooks import _verdict_response
from self_directing_mcp.audit import runner as audit_runner
from self_directing_mcp.codex_hooks import handle_hook
from test_append_capture import _AppendOnFirstFstat, _tamper
from test_index_concurrency import pause_first, session  # noqa: F401
from test_reliability import SID, call, result, write, session as reliability_session  # noqa: F401


ACTION = {"tool_name": "shell", "arguments": {"command": "deploy now"}}


def _gate(engine, path, *, requires_success=True):
    engine.upsert_contracts([{
        "id": "tests-first", "type": "must", "scope": "tool_call",
        "regex": "pytest", "before_regex": "deploy", "requires_success": requires_success,
    }])
    write(path, [call("pytest", "passed"), result("passed"), call("deploy earlier", "old-deploy")], "a")


def _late(status):
    events = [call("pytest", "new-test")]
    if status == "failed":
        events.append(result("new-test", exit_code=1))
    return events


def _with_growth(operation, path, monkeypatch, timing, events):
    size = path.stat().st_size
    if timing == "before_evaluation":
        encoded = "".join(json.dumps(event) + "\n" for event in events).encode()
        with monkeypatch.context() as patch:
            patch.setattr(session_events, "os", _AppendOnFirstFstat(path, encoded))
            value = operation()
    else:
        with pause_first(monkeypatch, audit_runner, "audit_contract") as (pool, entered, release):
            pending = pool.submit(operation)
            assert entered.wait(5)
            write(path, events, "a")
            release.set()
            value = pending.result(5)
    return size, value


def _assert_native_prefix(value, size):
    assert value["execution"]["executor"] == "neograph-engine"
    assert value["execution"]["nodes"] == [
        "select_contracts", "load_evidence", "evaluate_contracts", "aggregate_findings",
    ]
    assert value["action_executed"] is False
    assert value["coverage"]["complete"] is False
    assert value["coverage"]["judged_through_byte"] == size
    assert value["coverage"]["unread_bytes"] > 0
    assert "unread_bytes" in value["coverage"]["issues"]


@pytest.mark.parametrize("timing", ["before_evaluation", "during_evaluation"])
@pytest.mark.parametrize("status", ["failed", "pending"])
def test_current_gate_does_not_certify_old_success_with_unread_tests(session, monkeypatch, timing, status):
    engine, path = session
    _gate(engine, path)
    size, value = _with_growth(lambda: engine.check_action(SID, ACTION), path, monkeypatch, timing, _late(status))
    _assert_native_prefix(value, size)
    assert value["verdict"] == "unknown", value
    assert value["requires_attention"] is True
    assert value["requires_history"] is True
    finding = next(f for f in value["findings"] if f["contract_id"] == "tests-first")
    assert finding["verdict"] == "unknown"
    assert "incomplete" in finding["reason"].lower() or "unread" in finding["reason"].lower()
    assert _verdict_response(value, advisory=False)["decision"] == "deny"
    refreshed = engine.check_action(SID, ACTION)
    assert refreshed["coverage"]["complete"] is True
    assert refreshed["verdict"] == ("violation" if status == "failed" else "unknown")


@pytest.mark.parametrize("timing", ["before_evaluation", "during_evaluation"])
@pytest.mark.parametrize("status", ["failed", "pending"])
def test_retrospective_audit_can_still_judge_successful_historical_prefix(session, monkeypatch, timing, status):
    engine, path = session
    _gate(engine, path)
    size, value = _with_growth(lambda: engine.audit_session(SID), path, monkeypatch, timing, _late(status))
    _assert_native_prefix(value, size)
    assert value["verdict"] == "clean", value
    assert value["requires_attention"] is False
    assert "as of byte" in " ".join(value["notes"])


def test_current_gate_with_complete_successful_history_remains_clean(session):
    engine, path = session
    _gate(engine, path)
    value = engine.check_action(SID, ACTION)
    assert value["verdict"] == "clean"
    assert value["coverage"]["complete"] is True
    assert value["requires_attention"] is False
    assert _verdict_response(value, advisory=False)["decision"] == "allow"


@pytest.mark.parametrize("unrelated_gate", [False, True])
def test_history_independent_action_stays_clean_despite_growth(session, monkeypatch, unrelated_gate):
    engine, path = session
    if unrelated_gate:
        _gate(engine, path)
    engine.upsert_contracts([{"id": "no-delete", "type": "must_not", "scope": "tool_call", "regex": "DELETE"}])
    write(path, [call("echo safe", "safe")], "a")
    _, value = _with_growth(
        lambda: engine.check_action(SID, {"tool_name": "shell", "arguments": "echo safe"}),
        path, monkeypatch, "during_evaluation", _late("failed"),
    )
    assert value["verdict"] == "clean", value
    assert value["requires_history"] is False
    assert value["requires_attention"] is False
    assert _verdict_response(value, advisory=False)["decision"] == "allow"


def test_confirmed_forbidden_action_not_hidden_by_unread_prerequisite(session, monkeypatch):
    engine, path = session
    _gate(engine, path)
    engine.upsert_contracts([{"id": "no-delete", "type": "must_not", "scope": "tool_call", "regex": "DELETE"}])
    _, value = _with_growth(
        lambda: engine.check_action(SID, {"tool_name": "shell", "arguments": "deploy and DELETE"}),
        path, monkeypatch, "during_evaluation", _late("failed"),
    )
    assert value["verdict"] == "violation", value
    assert value["requires_attention"] is True
    findings = {f["contract_id"]: f for f in value["findings"]}
    assert findings["no-delete"]["verdict"] == "violation"
    assert findings["tests-first"]["verdict"] == "unknown"


def test_codex_warns_when_native_current_gate_has_unread_history(session, monkeypatch):
    engine, path = session
    _gate(engine, path)
    _, value = _with_growth(
        lambda: handle_hook(engine, "PreToolUse", SID, str(path), "shell", ACTION["arguments"]),
        path, monkeypatch, "during_evaluation", _late("failed"),
    )
    assert "unknown" in json.dumps(value), value
    # Codex stays advisory: a warning is not an execution or permission decision.
    assert "decision" not in value


@pytest.mark.parametrize("consumer", ["codex_wrapped", "agy_advisory"])
def test_consumers_preserve_native_unknown_without_changing_enforcement_mode(session, monkeypatch, consumer):
    engine, path = session
    _gate(engine, path)
    if consumer == "agy_advisory":
        _, value = _with_growth(lambda: engine.check_action(SID, ACTION), path, monkeypatch,
                                "during_evaluation", _late("failed"))
        assert value["verdict"] == "unknown"
        assert _verdict_response(value, advisory=False)["decision"] == "deny"
        advisory = _verdict_response(value, advisory=True)
        assert advisory["decision"] == "allow" and "ADVISORY" in advisory["reason"]
    else:
        _, value = _with_growth(
            lambda: handle_hook(engine, "PreToolUse", SID, str(path), "functions.exec",
                                'await tools.exec_command({"cmd":"deploy now"});'),
            path, monkeypatch, "during_evaluation", _late("pending"),
        )
        assert "unknown" in json.dumps(value), value
        assert "decision" not in value


@pytest.mark.parametrize("timing", ["before_evaluation", "during_evaluation"])
def test_observed_existence_prerequisite_survives_pure_append(session, monkeypatch, timing):
    engine, path = session
    _gate(engine, path, requires_success=False)
    size, value = _with_growth(lambda: engine.check_action(SID, ACTION), path, monkeypatch, timing, _late("failed"))
    _assert_native_prefix(value, size)
    assert value["requires_history"] is True  # The public field still means any applicable must.
    assert value["verdict"] == "clean", value
    assert value["requires_attention"] is False
    assert value["findings"][0]["verdict"] == "clean"
    assert _verdict_response(value, advisory=False)["decision"] == "allow"


@pytest.mark.parametrize("timing", ["before_evaluation", "during_evaluation"])
def test_mixed_prerequisites_only_downgrade_nonmonotone_success(session, monkeypatch, timing):
    engine, path = session
    _gate(engine, path)
    engine.upsert_contracts([{
        "id": "test-was-run", "type": "must", "scope": "tool_call",
        "regex": "pytest", "before_regex": "deploy", "requires_success": False,
    }])
    _, value = _with_growth(lambda: engine.check_action(SID, ACTION), path, monkeypatch, timing, _late("failed"))
    assert value["verdict"] == "unknown"
    assert value["requires_attention"] is True
    findings = {f["contract_id"]: f for f in value["findings"]}
    assert findings["tests-first"]["verdict"] == "unknown"
    assert findings["test-was-run"]["verdict"] == "clean"


def test_existence_prerequisite_does_not_hide_other_coverage_gaps(session):
    engine, path = session
    _gate(engine, path, requires_success=False)
    with path.open("a") as stream:
        stream.write("{not json}\n")
    value = engine.check_action(SID, ACTION)
    assert value["verdict"] == "unknown"
    assert value["requires_attention"] is True
    assert "parse_errors" in value["coverage"]["issues"]
    assert value["coverage"].get("judged_prefix") is not True
    assert value["findings"][0]["verdict"] == "unknown"


@pytest.mark.parametrize("kind", ["replace", "truncate"])
def test_existence_prerequisite_does_not_hide_source_rewrite(session, monkeypatch, kind):
    engine, path = session
    _gate(engine, path, requires_success=False)
    with pause_first(monkeypatch, audit_runner, "audit_contract") as (pool, entered, release):
        pending = pool.submit(engine.check_action, SID, ACTION)
        assert entered.wait(5)
        _tamper(path, kind)
        release.set()
        value = pending.result(5)
    assert value["verdict"] == "unknown"
    assert value["requires_attention"] is True
    assert "source_changed_during_evaluation" in value["coverage"]["issues"]
    assert value["coverage"].get("judged_prefix") is not True
    assert value["findings"][0]["verdict"] == "unknown"
