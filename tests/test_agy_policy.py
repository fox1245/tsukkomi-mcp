"""AGY enforcement policy agreed in the roundtable: confirmed-only blocking, bounded continue, fail-open errors.

Synthetic data only. Real AGY hook payloads carry ``workspacePaths`` (array) and ``stepIdx`` (nonnegative int).
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from self_directing_mcp import agy_hook_state as state
from self_directing_mcp import agy_hooks
from self_directing_mcp.agy_hooks import (
    _verdict_response, failure_response, handle_pre_invocation, handle_pre_tool_use, handle_stop, run_hook,
)
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine

SID = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"
ROOT = Path(__file__).resolve().parents[1]
MODES = ("advisory", "confirmed_only", "enforced")
SENTINEL = "SENTINEL-payload-text-4f9c"


def transcript_path(root, sid=SID):
    return root / "brain" / sid / ".system_generated" / "logs" / "transcript.jsonl"


def user_input(step=0, text="hi"):
    return {"step_index": step, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE",
            "created_at": "2026-10-01T00:00:00Z", "content": f"<USER_REQUEST>{text}</USER_REQUEST>"}


def assistant(step, text):
    return {"step_index": step, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE",
            "created_at": "2026-10-01T00:00:01Z", "content": text}


def write_history(path, *records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated index and app roots, also visible to hook subprocesses and the stdlib fallbacks."""
    app, index = tmp_path / "app", tmp_path / "index"
    path = transcript_path(app)
    path.parent.mkdir(parents=True)
    write_history(path, user_input())
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(index))
    monkeypatch.setenv("SELF_DIRECT_AGY_APP_DATA_DIRS", json.dumps([str(app)]))
    monkeypatch.setenv("SELF_DIRECT_LOCAL_ONLY", "true")
    monkeypatch.delenv("SELF_DIRECT_AGY_ENFORCEMENT", raising=False)
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=index, agy_app_data_dirs=[app],
                                       openrouter_api_key_file=None))
    yield engine, path
    engine.close()


def set_mode(engine, mode):
    engine.settings.agy_enforcement = mode


def payload(path, event, *, tool="run_command", args=None, step=3, sid=SID, **extra):
    base = {"conversationId": sid, "transcriptPath": str(path), "workspacePaths": ["/synthetic/project"]}
    if event in ("PreToolUse", "PostToolUse"):
        base.update(toolCall={"name": tool, "args": args or {"CommandLine": "echo ok"}}, stepIdx=step)
    if event == "Stop":
        base.update(fullyIdle=True, terminationReason="NO_TOOL_CALL")
    base.update(extra)
    return base


def stop(engine, path, sid=SID, reason="NO_TOOL_CALL", idle=True):
    return handle_stop(engine, sid, {"transcriptPath": str(path), "fullyIdle": idle, "terminationReason": reason})


def call(command):
    return {"name": "run_command", "args": {"CommandLine": command}}


# --- configuration ----------------------------------------------------------------------------------------

def test_default_mode_is_confirmed_only_and_enforced_remains_an_explicit_option(monkeypatch):
    monkeypatch.delenv("SELF_DIRECT_AGY_ENFORCEMENT", raising=False)
    assert Settings(_env_file=None).agy_enforcement == "confirmed_only"
    for mode in MODES:
        monkeypatch.setenv("SELF_DIRECT_AGY_ENFORCEMENT", mode)
        assert Settings(_env_file=None).agy_enforcement == mode
    monkeypatch.setenv("SELF_DIRECT_AGY_ENFORCEMENT", "bogus")
    with pytest.raises(ValueError):
        Settings(_env_file=None)
    assert state.resolve_mode() == "confirmed_only"  # the fallback never trusts an invalid value


# --- PreToolUse: verdict x mode ---------------------------------------------------------------------------

def _violation(engine):
    engine.upsert_contracts([{"id": "c", "type": "must_not", "scope": "tool_call", "provider": "agy", "regex": "forbidden"}])
    return call("run forbidden")


def _suspicious(engine):
    engine.upsert_contracts([{"id": "c", "type": "must_not", "scope": "tool_call", "provider": "agy",
                              "regex": "deploy", "exception_regex": "staging"}])
    return call("deploy staging")


def _unknown(engine, risk="normal"):
    engine.upsert_contracts([{"id": "c", "type": "must_not", "provider": "agy", "risk": risk,
                              "description": "no verifier exists for this rule"}])
    return call("echo ok")


def _clean(engine):
    engine.upsert_contracts([{"id": "c", "type": "must_not", "scope": "tool_call", "provider": "agy", "regex": "forbidden"}])
    return call("echo ok")


SCENARIOS = {
    "violation": _violation,
    "suspicious": _suspicious,
    "unknown": _unknown,
    "unknown_high_risk": lambda engine: _unknown(engine, "high"),
    "clean": _clean,
    "no_contract": lambda engine: call("echo ok"),
}
EXPECTED = {
    "violation": dict(advisory="allow", confirmed_only="deny", enforced="deny"),
    "suspicious": dict(advisory="allow", confirmed_only="force_ask", enforced="force_ask"),
    "unknown": dict(advisory="allow", confirmed_only="allow", enforced="deny"),
    "unknown_high_risk": dict(advisory="allow", confirmed_only="ask", enforced="deny"),
    "clean": dict(advisory="allow", confirmed_only="allow", enforced="allow"),
    "no_contract": dict(advisory="allow", confirmed_only="allow", enforced="allow"),
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_pretooluse_verdict_by_mode(env, scenario, mode):
    engine, path = env
    set_mode(engine, mode)
    action = SCENARIOS[scenario](engine)
    response = handle_pre_tool_use(engine, SID, action, str(path))
    assert response["decision"] == EXPECTED[scenario][mode], response
    if scenario in ("clean", "no_contract"):
        assert "WARNING" not in response.get("reason", "")


def test_unknown_allows_with_a_visible_unverified_warning_not_a_clean_claim(env):
    engine, path = env
    response = handle_pre_tool_use(engine, SID, _unknown(engine), str(path))
    assert response["decision"] == "allow"
    assert "WARNING" in response["reason"] and "not a compliance verdict" in response["reason"]
    assert "[c]" in response["reason"]


def test_unknown_warning_is_injected_once_at_the_next_invocation(env):
    engine, path = env
    handle_pre_tool_use(engine, SID, _unknown(engine), str(path))
    first = handle_pre_invocation(engine, SID, {"transcriptPath": str(path)})
    text = first["injectSteps"][0]["ephemeralMessage"]
    assert "unverified" in text.lower() and "[c]" in text
    second = handle_pre_invocation(engine, SID, {"transcriptPath": str(path)})
    assert "unverified" not in json.dumps(second).lower()


def test_high_risk_only_raises_unknown_for_the_contracts_that_apply_to_the_action(env):
    engine, path = env
    engine.upsert_contracts([
        {"id": "risky", "type": "must_not", "scope": "tool_call", "provider": "agy", "regex": "git push",
         "risk": "high", "exception_regex": "--dry-run"},
        {"id": "prose", "type": "must_not", "provider": "agy", "description": "no verifier"}])
    # the unknown comes from the normal-risk description-only rule; the risky rule does not apply to `ls`
    assert handle_pre_tool_use(engine, SID, call("ls"), str(path))["decision"] == "allow"


def test_suspicious_beside_an_unknown_prerequisite_asks_in_confirmed_only_and_denies_when_enforced(env):
    engine, path = env
    engine.upsert_contracts([
        {"id": "gate", "type": "must_not", "scope": "tool_call", "provider": "agy", "regex": "deploy",
         "exception_regex": "staging"},
        {"id": "tests-first", "provider": "agy", "type": "must", "scope": "tool_call", "regex": "npm test",
         "before_regex": "deploy", "requires_success": True}])
    path.unlink()                         # incomplete history leaves the prerequisite unknown
    action = call("deploy staging")
    assert handle_pre_tool_use(engine, SID, action, str(path))["decision"] == "force_ask"
    set_mode(engine, "enforced")
    assert handle_pre_tool_use(engine, SID, action, str(path))["decision"] == "deny"


def test_verdict_response_is_pure_and_defaults_to_confirmed_only():
    unknown = {"ok": True, "verdict": "unknown", "findings": [], "action_contracts": ["c"]}
    assert _verdict_response(unknown)["decision"] == "allow"
    assert _verdict_response(unknown, high_risk=frozenset({"c"}))["decision"] == "ask"
    assert _verdict_response(unknown, "enforced")["decision"] == "deny"
    assert _verdict_response({"ok": False}, "confirmed_only")["decision"] == "allow"
    advisory = _verdict_response({"ok": True, "verdict": "violation", "findings": []}, "advisory")
    assert advisory["decision"] == "allow" and "ADVISORY" in advisory["reason"]


# --- Stop: bounded continue -------------------------------------------------------------------------------

ACK = {"id": "ack", "type": "must", "scope": "message", "provider": "agy", "regex": "VERIFIED", "auto_continue": True,
       "description": "state VERIFIED when done"}


def test_stop_continues_once_per_session_for_a_confirmed_auto_continue_obligation(env):
    engine, path = env
    engine.upsert_contracts([ACK])
    first = stop(engine, path)
    assert first["decision"] == "continue" and "ack" in first["reason"]
    assert stop(engine, path)["decision"] == "allow"          # another turn: the budget is not reset
    assert stop(engine, path)["decision"] == "allow"
    assert "budget" in stop(engine, path)["reason"].lower()


def test_budget_survives_a_process_restart_and_is_per_conversation(env, tmp_path):
    engine, path = env
    engine.upsert_contracts([ACK])
    other_path = transcript_path(tmp_path / "app", OTHER)
    other_path.parent.mkdir(parents=True)
    write_history(other_path, user_input())
    assert stop(engine, path)["decision"] == "continue"
    restarted = SelfDirectEngine(engine.settings)
    try:
        assert stop(restarted, path)["decision"] == "allow"
        assert stop(restarted, other_path, sid=OTHER)["decision"] == "continue"
    finally:
        restarted.close()


def test_budget_survives_real_process_restarts(env, tmp_path):
    engine, path = env
    engine.upsert_contracts([ACK])
    def run():
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "agy_hook.py"), "--event=Stop"],
                                input=json.dumps(payload(path, "Stop")), capture_output=True, text=True,
                                timeout=60, cwd=tmp_path)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    assert run()["decision"] == "continue"
    assert run()["decision"] == "allow"


def test_same_reason_is_refused_even_when_budget_remains_and_a_new_reason_is_not(env, monkeypatch):
    engine, path = env
    monkeypatch.setattr(state, "CONTINUE_BUDGET", 2)
    engine.upsert_contracts([ACK])
    assert stop(engine, path)["decision"] == "continue"
    assert stop(engine, path)["decision"] == "allow"                     # same reason
    engine.upsert_contracts([dict(ACK, id="ack2", regex="DOUBLE-CHECKED")])
    assert stop(engine, path)["decision"] == "continue"                  # a different reason, within budget
    engine.upsert_contracts([dict(ACK, id="ack3", regex="THIRD")])
    assert stop(engine, path)["decision"] == "allow"                     # budget exhausted


def test_unreadable_or_unwritable_counter_never_continues(env):
    engine, path = env
    engine.upsert_contracts([ACK])
    state_file = Path(engine.settings.index_dir) / state.STATE_FILE
    state_file.mkdir(parents=True)                                       # cannot be opened as a database
    response = stop(engine, path)
    assert response["decision"] == "allow" and "budget" in response["reason"].lower()
    assert state.HookState(engine.settings.index_dir).claim_continue(SID, "k") is False


def test_counter_write_failure_midway_never_continues(env, monkeypatch):
    engine, path = env
    engine.upsert_contracts([ACK])
    real = state.HookState._connect
    def broken(self):
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(state.HookState, "_connect", broken)
    assert stop(engine, path)["decision"] == "allow"
    monkeypatch.setattr(state.HookState, "_connect", real)
    assert stop(engine, path)["decision"] == "continue"                  # nothing was consumed by the failure


def test_unmet_obligation_without_auto_continue_never_continues(env):
    engine, path = env
    engine.upsert_contracts([dict(ACK, auto_continue=False)])
    response = stop(engine, path)
    assert response["decision"] == "allow" and "ack" in response["reason"]


def test_unknown_evidence_never_continues(env):
    engine, path = env
    engine.upsert_contracts([ACK])
    with path.open("a", encoding="utf-8") as stream:
        stream.write("not json\n")                                       # a legacy/parse defect
    response = stop(engine, path)
    assert response["decision"] == "allow"
    assert state.HookState(engine.settings.index_dir).continue_claims(SID) in (0, None)


def test_pending_checklist_and_description_only_rules_do_not_continue(env):
    engine, path = env
    engine.update_checklist(SID, provider="agy", add=[{"text": "Verify integration"}])
    engine.upsert_contracts([{"id": "prose", "type": "must", "provider": "agy", "description": "be thorough",
                              "auto_continue": False}])
    response = stop(engine, path)
    assert response["decision"] == "allow" and "Verify integration" in response["reason"]


def test_unmet_when_makes_an_auto_continue_obligation_not_applicable(env):
    engine, path = env
    engine.upsert_contracts([dict(ACK, when={"tool_call_regex": "docker build"})])
    assert stop(engine, path)["decision"] == "allow"
    assert state.HookState(engine.settings.index_dir).continue_claims(SID) in (0, None)


def test_a_satisfied_obligation_does_not_continue_or_consume_budget(env):
    engine, path = env
    engine.upsert_contracts([ACK])
    write_history(path, user_input(), assistant(1, "VERIFIED"))
    assert stop(engine, path)["decision"] == "allow"
    write_history(path, user_input())
    assert stop(engine, path)["decision"] == "continue"                  # budget was untouched


@pytest.mark.parametrize("reason,idle", [("cancelled", True), ("error", True), ("max_steps_exceeded", True),
                                          ("NO_TOOL_CALL", False)])
def test_abnormal_stops_never_continue_or_consume_budget(env, reason, idle):
    engine, path = env
    engine.upsert_contracts([ACK])
    assert stop(engine, path, reason=reason, idle=idle)["decision"] == "allow"
    assert stop(engine, path)["decision"] == "continue"


def test_advisory_mode_never_continues(env):
    engine, path = env
    engine.upsert_contracts([ACK])
    set_mode(engine, "advisory")
    response = stop(engine, path)
    assert response["decision"] == "allow" and "ADVISORY" in response["reason"]
    set_mode(engine, "confirmed_only")
    assert stop(engine, path)["decision"] == "continue"


@pytest.mark.parametrize("mode", ["confirmed_only", "enforced"])
def test_mode_does_not_change_the_stop_policy(env, mode):
    engine, path = env
    engine.upsert_contracts([ACK])
    set_mode(engine, mode)
    assert stop(engine, path)["decision"] == "continue"
    assert stop(engine, path)["decision"] == "allow"


def test_an_incomplete_workflow_continues_at_most_once(env, monkeypatch):
    engine, path = env
    class Controller:
        def status(self, session_id, provider):
            return {"stage": "implementing", "requirements": [], "evidence": [], "problems": []}
    monkeypatch.setattr("self_directing_mcp.workflow.configured_workflow", lambda: Controller())
    assert stop(engine, path)["decision"] == "continue"
    assert stop(engine, path)["decision"] == "allow"


# --- errors: fail-open in every mode, per event -----------------------------------------------------------

def _raising(*args, **kwargs):
    raise RuntimeError(SENTINEL)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("event", ["PreToolUse", "PostToolUse", "PreInvocation", "Stop"])
@pytest.mark.parametrize("failure", ["handler", "engine", "payload"])
def test_internal_errors_fail_open_with_a_warning_in_every_mode(env, monkeypatch, capsys, event, mode, failure):
    engine, path = env
    set_mode(engine, mode)
    monkeypatch.setenv("SELF_DIRECT_AGY_ENFORCEMENT", mode)
    message = payload(path, event)
    if failure == "handler":
        for name in ("handle_pre_tool_use", "handle_stop", "handle_pre_invocation"):
            monkeypatch.setattr(agy_hooks, name, _raising)
        monkeypatch.setattr(engine, "sync_session", _raising)
        monkeypatch.setattr(engine, "record_agy_hook", _raising)
        response = run_hook(event, message, engine=engine)
    elif failure == "engine":
        monkeypatch.setattr("self_directing_mcp.engine.SelfDirectEngine", _raising)
        response = run_hook(event, message)
    else:
        message["workspacePaths"] = "not-an-array"
        response = run_hook(event, message, engine=engine)
    text = json.dumps(response)
    assert "deny" not in text and "continue" not in text and "force_ask" not in text
    assert SENTINEL not in text + capsys.readouterr().err
    if event == "PostToolUse":
        assert response == {}
    elif event == "PreInvocation":
        assert "do not claim compliance" in response["injectSteps"][0]["ephemeralMessage"]
    else:
        assert response["decision"] == "allow" and "not a compliance verdict" in response["reason"]
        assert ("UNAVAILABLE" if failure == "engine" else "FAILED") in response["reason"]


def test_a_failed_host_receipt_does_not_skip_the_check(env, monkeypatch):
    engine, path = env
    _violation(engine)
    monkeypatch.setattr(engine, "record_agy_hook", _raising)
    response = run_hook("PreToolUse", payload(path, "PreToolUse", args={"CommandLine": "run forbidden"}), engine=engine)
    assert response["decision"] == "deny"                                # a race on receipts cannot hide a violation
    health = state.HookState(engine.settings.index_dir).health()
    assert [(r["kind"], r["event"], r["last_error_class"]) for r in health] == [("check_failed", "PreToolUse", "RuntimeError")]
    clean = run_hook("PreToolUse", payload(path, "PreToolUse", args={"CommandLine": "echo ok"}), engine=engine)
    assert clean["decision"] == "allow" and "receipt" in clean["reason"].lower()


@pytest.mark.parametrize("mode,expected", [("advisory", "allow"), ("confirmed_only", "ask"), ("enforced", "ask")])
def test_error_with_a_matching_high_risk_contract_asks_only_when_not_advisory(env, monkeypatch, mode, expected):
    engine, path = env
    set_mode(engine, mode)
    engine.upsert_contracts([{"id": "no-force-push", "type": "must_not", "scope": "tool_call", "provider": "agy",
                              "regex": "git push --force", "risk": "high"},
                             {"id": "normal-risk", "type": "must_not", "scope": "tool_call", "regex": "curl"}])
    monkeypatch.setattr(agy_hooks, "handle_pre_tool_use", _raising)
    risky = run_hook("PreToolUse", payload(path, "PreToolUse", args={"CommandLine": "git push --force origin x"}), engine=engine)
    assert risky["decision"] == expected
    if expected == "ask":
        assert "no-force-push" in risky["reason"] and "not a compliance verdict" in risky["reason"]
    for command in ("git push origin x", "curl example.invalid"):
        assert run_hook("PreToolUse", payload(path, "PreToolUse", args={"CommandLine": command}),
                        engine=engine)["decision"] == "allow"


def test_error_never_asks_or_denies_when_contracts_are_unreadable(env, monkeypatch):
    engine, path = env
    contracts = engine.settings.resolve_contracts_path()
    contracts.parent.mkdir(parents=True, exist_ok=True)
    contracts.write_text("{not json", encoding="utf-8")
    response = run_hook("PreToolUse", payload(path, "PreToolUse"), engine=engine)
    assert response["decision"] == "allow" and "UNAVAILABLE" in response["reason"]


def test_main_reads_stdin_and_never_raises(env, monkeypatch, capsys):
    engine, path = env
    for stdin, event in [("not json " + SENTINEL, "PreToolUse"), ("[]", "Stop"), ("{}", "PreInvocation"), ("", "PostToolUse")]:
        monkeypatch.setattr(sys, "argv", ["agy_hook.py", f"--event={event}"])
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
        agy_hooks.main()
        out = capsys.readouterr()
        response = json.loads(out.out)
        assert "deny" not in out.out and "continue" not in out.out
        assert SENTINEL not in out.out + out.err
        assert (response == {}) if event == "PostToolUse" else bool(response)


def test_failure_response_is_fail_open_for_every_event():
    assert failure_response("PreToolUse")["decision"] == "allow"
    assert failure_response("PostToolUse") == {}
    assert "injectSteps" in failure_response("PreInvocation")
    assert failure_response("Stop", {"terminationReason": "model_stop", "fullyIdle": True})["decision"] == "allow"
    assert failure_response(None)["decision"] == "allow"


# --- the stdlib fallback in scripts/agy_hook.py -----------------------------------------------------------

def _broken_package(tmp_path, *, with_state):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    wrapper = scripts / "agy_hook.py"
    shutil.copyfile(ROOT / "scripts" / "agy_hook.py", wrapper)
    package = tmp_path / "src" / "self_directing_mcp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "agy_hooks.py").write_text("raise ImportError('native runtime unavailable')\n")
    if with_state:
        shutil.copyfile(ROOT / "src" / "self_directing_mcp" / "agy_hook_state.py", package / "agy_hook_state.py")
    return wrapper


def _run_wrapper(wrapper, event, stdin, cwd, env_extra=None):
    environment = {**os.environ, **(env_extra or {})}
    result = subprocess.run([sys.executable, "-I", str(wrapper), "--event=" + event], input=stdin, cwd=cwd,
                            capture_output=True, text=True, timeout=30, env=environment)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout), result


@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("event,stdin", [
    ("PreToolUse", "broken JSON " + SENTINEL),
    ("Stop", '{"fullyIdle":true,"terminationReason":"model_stop"}'),
    ("Stop", '{"fullyIdle":true,"terminationReason":"NO_TOOL_CALL"}'),
    ("PreInvocation", "{}"),
    ("PostToolUse", "{}"),
])
def test_stdlib_fallback_after_import_failure_is_fail_open(tmp_path, env, event, stdin, mode, with_state):
    base = tmp_path / ("pkg-state" if with_state else "pkg-bare")
    base.mkdir()
    wrapper = _broken_package(base, with_state=with_state)
    response, result = _run_wrapper(wrapper, event, stdin, tmp_path, {"SELF_DIRECT_AGY_ENFORCEMENT": mode})
    text = json.dumps(response)
    assert "deny" not in text and "continue" not in text and "force_ask" not in text
    assert SENTINEL not in result.stdout + result.stderr
    if event == "PostToolUse":
        assert response == {}
    elif event == "PreInvocation":
        assert "injectSteps" in response
    else:
        assert response["decision"] == "allow" and "not a compliance verdict" in response["reason"]
    assert "ImportError" in result.stderr


def test_stdlib_fallback_records_health_and_banners_after_repeats(tmp_path, env):
    engine, path = env
    base = tmp_path / "pkg"
    base.mkdir()
    wrapper = _broken_package(base, with_state=True)
    for count in range(1, 4):
        response, _ = _run_wrapper(wrapper, "PreInvocation", "{}", tmp_path)
        message = response["injectSteps"][0]["ephemeralMessage"]
        assert ("INACTIVE" in message) == (count >= 3), (count, message)
    health = state.HookState(engine.settings.index_dir).health()
    assert [(r["kind"], r["event"], r["consecutive"], r["last_error_class"]) for r in health] == [
        ("audit_unavailable", "PreInvocation", 3, "ImportError")]


def test_stdlib_fallback_asks_for_a_matching_high_risk_contract_without_the_package(tmp_path, env):
    engine, path = env
    engine.upsert_contracts([{"id": "no-force-push", "type": "must_not", "scope": "tool_call", "provider": "agy",
                              "regex": "git push --force", "risk": "high"}])
    base = tmp_path / "pkg"
    base.mkdir()
    wrapper = _broken_package(base, with_state=True)
    risky = payload(path, "PreToolUse", args={"CommandLine": "git push --force origin x"})
    response, _ = _run_wrapper(wrapper, "PreToolUse", json.dumps(risky), tmp_path)
    assert response["decision"] == "ask" and "no-force-push" in response["reason"]
    benign = payload(path, "PreToolUse", args={"CommandLine": "git status"})
    assert _run_wrapper(wrapper, "PreToolUse", json.dumps(benign), tmp_path)[0]["decision"] == "allow"
    advisory = _run_wrapper(wrapper, "PreToolUse", json.dumps(risky), tmp_path,
                            {"SELF_DIRECT_AGY_ENFORCEMENT": "advisory"})[0]
    assert advisory["decision"] == "allow"


# --- hook_health ------------------------------------------------------------------------------------------

def test_hook_health_records_class_counts_and_timestamps_only(env, monkeypatch, capsys):
    engine, path = env
    monkeypatch.setattr(agy_hooks, "handle_pre_tool_use", _raising)
    message = payload(path, "PreToolUse", args={"CommandLine": "echo " + SENTINEL})
    run_hook("PreToolUse", message, engine=engine)
    run_hook("PreToolUse", message, engine=engine)
    [row] = state.HookState(engine.settings.index_dir).health()
    assert (row["kind"], row["event"], row["total"], row["consecutive"], row["last_error_class"]) == (
        "check_failed", "PreToolUse", 2, 2, "RuntimeError")
    assert row["first_at"] <= row["last_at"]
    raw = (Path(engine.settings.index_dir) / state.STATE_FILE).read_bytes()
    assert SENTINEL.encode() not in raw
    assert SENTINEL not in capsys.readouterr().err
    with sqlite3.connect(Path(engine.settings.index_dir) / state.STATE_FILE) as conn:
        columns = {c[1] for table in ("hook_health",) for c in conn.execute(f"PRAGMA table_info({table})")}
    assert not {"verdict", "clean", "payload", "message"} & columns   # an error can never be recorded as clean


def test_banner_appears_at_three_consecutive_failures_and_names_the_kind(env, monkeypatch):
    engine, path = env
    health = state.HookState(engine.settings.index_dir)
    health.record_error(state.CHECK_FAILED, "PreToolUse", "OSError")
    health.record_error(state.CHECK_FAILED, "PreToolUse", "OSError")
    assert health.banner() is None
    assert "tsukkomi" not in json.dumps(handle_pre_invocation(engine, SID, {"transcriptPath": str(path)})).lower()
    health.record_error(state.CHECK_FAILED, "PreToolUse", "OSError")
    banner = health.banner()
    assert "DEGRADED" in banner and "PreToolUse" in banner and "OSError" in banner and "INACTIVE" not in banner
    for _ in range(3):
        health.record_error(state.AUDIT_UNAVAILABLE, "Stop", "ValueError")
    banner = health.banner()
    assert "INACTIVE" in banner and "DEGRADED" in banner
    injected = handle_pre_invocation(engine, SID, {"transcriptPath": str(path)})["injectSteps"][0]["ephemeralMessage"]
    assert "Tsukkomi audit INACTIVE" in injected and "do not claim compliance" in injected


def test_success_resets_the_consecutive_count_and_clears_the_banner(env):
    engine, path = env
    health = state.HookState(engine.settings.index_dir)
    for _ in range(3):
        health.record_error(state.CHECK_FAILED, "Stop", "OSError")
        health.record_error(state.AUDIT_UNAVAILABLE, "PreToolUse", "ImportError")
    assert health.banner()
    health.record_success("Stop")                       # partial failures reset per event
    assert "DEGRADED" not in (health.banner() or "")
    health.record_success("PostToolUse")                # a working runtime clears full unavailability
    assert health.banner() is None
    totals = {(r["kind"], r["event"]): r["total"] for r in health.health()}
    assert totals == {("check_failed", "Stop"): 3, ("audit_unavailable", "PreToolUse"): 3}


def test_run_hook_success_resets_health_and_failures_accumulate_end_to_end(env, monkeypatch):
    engine, path = env
    original = agy_hooks.handle_stop
    monkeypatch.setattr(agy_hooks, "handle_stop", _raising)
    for _ in range(3):
        run_hook("Stop", payload(path, "Stop"), engine=engine)
    injected = run_hook("PreInvocation", payload(path, "PreInvocation"), engine=engine)
    assert "DEGRADED" in injected["injectSteps"][0]["ephemeralMessage"]
    monkeypatch.setattr(agy_hooks, "handle_stop", original)
    assert run_hook("Stop", payload(path, "Stop"), engine=engine)["decision"] == "allow"
    assert state.HookState(engine.settings.index_dir).banner() is None


def test_unusable_health_store_never_breaks_a_hook(env, monkeypatch):
    engine, path = env
    (Path(engine.settings.index_dir) / state.STATE_FILE).mkdir(parents=True)
    monkeypatch.setattr(agy_hooks, "handle_pre_tool_use", _raising)
    response = run_hook("PreToolUse", payload(path, "PreToolUse"), engine=engine)
    assert response["decision"] == "allow" and "not a compliance verdict" in response["reason"]
    health = state.HookState(engine.settings.index_dir)
    assert health.record_error(state.CHECK_FAILED, "Stop", "OSError") is False
    assert health.banner() is None and health.health() == []


def test_warning_queue_is_per_conversation_bounded_and_deduplicated(env):
    engine, _path = env
    health = state.HookState(engine.settings.index_dir)
    for _ in range(3):
        health.queue_warning(SID, "same")
    for index in range(20):
        health.queue_warning(SID, f"w{index}")
    health.queue_warning(OTHER, "other")
    drained = health.drain_warnings(SID)
    assert drained.count("same") <= 1 and len(drained) <= 5
    assert health.drain_warnings(SID) == [] and health.drain_warnings(OTHER) == ["other"]


def test_concurrent_claims_spend_the_budget_exactly_once(env):
    from concurrent.futures import ThreadPoolExecutor
    engine, _path = env
    health = state.HookState(engine.settings.index_dir)
    health.claim_continue(OTHER, "warm-up")           # create the schema once, as an earlier hook would have
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: state.HookState(engine.settings.index_dir).claim_continue(SID, "same"), range(8)))
    assert results.count(True) == 1
    assert health.continue_claims(SID) == 1
