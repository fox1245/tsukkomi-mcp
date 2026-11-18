"""A Claude Code PreToolUse call that a ``risk: high`` contract gates and whose attempt at the resident daemon failed for a
transient cause is judged by the hook's own engine instead of being refused; every other call is the daemon's, exactly as
before, and costs nothing more.

Synthetic data only. The hook script runs in this process with the daemon flag on and the daemon exchange (``client._serve``)
replaced by a recorder that answers or fails as a test says, so nothing here starts a daemon: a call that reaches ``_serve``
was sent to the daemon, a call that reaches ``_new_engine`` was judged locally. The same flows against a real daemon process
are in ``tests/daemon/test_daemon_gated_local.py``; the bound on the whole run is in ``tests/test_hook_deadline.py``.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from self_directing_mcp import claude_hook_state as state
from self_directing_mcp import claude_hooks
from self_directing_mcp.claude_hooks import run_hook
from self_directing_mcp.daemon import client
from self_directing_mcp.request_control import IndexBusy, RequestStopped, TransientHookFailure

from test_claude_hooks import (  # noqa: F401  (env is a fixture)
    assistant, env, human, main_file, payload, permission, set_mode, tool_result, use, write,
)
from test_hook_transient_audit import FORBID, GATE, PUSH, Clock, write_passing_tests

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "claude_hook.py"
STUB = {"systemMessage": "answered by the daemon stub"}
UNGATED = {"command": "ls"}
SATISFIED = {**GATE, "requires_success": True}
TRANSIENT = ["daemon_unavailable", "daemon_timeout", "transient_failure"]

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the resident daemon is Linux only"),
    pytest.mark.skipif((ROOT / ".venv").exists(), reason="a checkout .venv would make the script re-execute this process"),
]


class Seam:
    def __init__(self, engine, path):
        self.engine, self.path = engine, path
        self.attempts, self.local, self.gate_checks = [], [], []
        self.clock = None
        self.result = (dict(STUB), None)            # what the daemon exchange answers: (response, None) or (None, reason)

    def fail(self, reason):
        self.result = (None, reason)


@pytest.fixture
def seam(env, monkeypatch):
    """The daemon switch on, the daemon exchange, the standalone engine factory and the gate recorded."""
    monkeypatch.setenv("SELF_DIRECT_DAEMON", "on")
    monkeypatch.delenv("SELF_DIRECT_HOOK_DEADLINE_SEC", raising=False)
    engine, path = env
    found = Seam(engine, path)

    def serve(host, event, body, argv, environ, python, **kwargs):
        found.attempts.append((host, event))
        return found.result

    real_new_engine, real_rule = claude_hooks._new_engine, state.gating_rule

    def new_engine():
        found.local.append(1)
        return real_new_engine()

    def rule(*args, **kwargs):
        found.gate_checks.append(1)
        return real_rule(*args, **kwargs)
    clock = found.clock = Clock()
    monkeypatch.setattr(client, "_serve", serve)
    monkeypatch.setattr(client, "_monotonic", clock)
    monkeypatch.setattr(client, "_sleep", clock.sleep)
    monkeypatch.setattr(claude_hooks, "_new_engine", new_engine)
    monkeypatch.setattr(state, "gating_rule", rule)
    monkeypatch.setattr(claude_hooks, "spawn_verifier", lambda engine: None)       # no detached process from a test
    return found


def run_script(monkeypatch, capsys, body, event):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), f"--event={event}"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(body)))
    monkeypatch.delenv(client.ENV_LOCAL_BUDGET, raising=False)
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return json.loads(capsys.readouterr().out)


def pre(seam_, tool_input):
    return payload(seam_.path, "PreToolUse", tool_input=tool_input)


# --- the daemon first, and nothing more on that path ------------------------------------------------------------

@pytest.mark.parametrize("tool_input", [PUSH, UNGATED], ids=["gated", "ungated"])
def test_a_call_the_daemon_answers_is_never_inspected_by_the_client(seam, monkeypatch, capsys, tool_input):
    seam.engine.upsert_contracts([SATISFIED])
    response = run_script(monkeypatch, capsys, pre(seam, tool_input), "PreToolUse")
    assert response == STUB
    assert seam.attempts == [("claude", "PreToolUse")] and seam.local == []
    assert seam.gate_checks == []                                    # no contracts file read, no gate module used


# --- the fallback of a failed attempt ---------------------------------------------------------------------------

@pytest.mark.parametrize("reason", TRANSIENT)
def test_a_gated_call_whose_daemon_attempt_failed_transiently_is_judged_by_a_local_engine(seam, monkeypatch, capsys, reason):
    seam.engine.upsert_contracts([SATISFIED])
    write_passing_tests(seam.path)
    seam.fail(reason)
    response = run_script(monkeypatch, capsys, pre(seam, PUSH), "PreToolUse")
    assert seam.attempts == [("claude", "PreToolUse")]               # the daemon was tried once, first
    assert seam.local == [1]
    assert response != STUB and permission(response) == "none" and "UNAVAILABLE" not in json.dumps(response)


@pytest.mark.parametrize("reason", TRANSIENT)
def test_an_ungated_call_whose_daemon_attempt_failed_fails_open_as_before(seam, monkeypatch, capsys, reason):
    seam.engine.upsert_contracts([SATISFIED])
    seam.fail(reason)
    response = run_script(monkeypatch, capsys, pre(seam, UNGATED), "PreToolUse")
    assert seam.local == []
    assert permission(response) == "none" and reason in json.dumps(response)


@pytest.mark.parametrize("reason", ["queue_full", "version_mismatch", "daemon_unavailable:crash-loop",
                                    "daemon_unavailable:restart-loop", "frame_too_large", "internal_error"])
def test_a_failure_that_is_not_transient_is_answered_as_before(seam, monkeypatch, capsys, reason):
    seam.engine.upsert_contracts([SATISFIED])
    seam.fail(reason)
    response = run_script(monkeypatch, capsys, pre(seam, PUSH), "PreToolUse")
    assert seam.local == []
    assert permission(response) == "ask" and reason in json.dumps(response)      # the unchanged prerequisite question


def test_advisory_mode_gates_nothing_so_a_failed_attempt_is_not_judged_locally(seam, monkeypatch, capsys):
    seam.engine.upsert_contracts([SATISFIED])
    monkeypatch.setenv("SELF_DIRECT_CLAUDE_ENFORCEMENT", "advisory")
    seam.fail("daemon_unavailable")
    response = run_script(monkeypatch, capsys, pre(seam, PUSH), "PreToolUse")
    assert seam.local == [] and permission(response) == "none"


@pytest.mark.parametrize("event", ["Stop", "UserPromptSubmit", "PostToolUse", "SessionStart"])
def test_only_a_pre_tool_use_call_can_be_judged_locally(seam, monkeypatch, capsys, event):
    seam.engine.upsert_contracts([SATISFIED])
    seam.fail("daemon_unavailable")
    run_script(monkeypatch, capsys, payload(seam.path, event, tool_input=PUSH), event)
    assert seam.local == []


def test_a_contracts_file_that_cannot_be_read_gates_nothing(seam, monkeypatch, capsys):
    seam.engine.upsert_contracts([SATISFIED])
    seam.engine.settings.resolve_contracts_path().write_text("{ this is not json", encoding="utf-8")
    seam.fail("daemon_unavailable")
    run_script(monkeypatch, capsys, pre(seam, PUSH), "PreToolUse")
    assert seam.local == []


@pytest.mark.parametrize("change", [{"risk": "normal"}, {"enabled": False}, {"tools": ["Write"]}, {"provider": "agy"}],
                         ids=["not-high-risk", "disabled", "other-tool", "other-host"])
def test_a_contract_that_does_not_gate_this_call_leaves_a_failed_attempt_as_before(seam, monkeypatch, capsys, change):
    seam.engine.upsert_contracts([{**SATISFIED, **change}])
    seam.fail("daemon_unavailable")
    run_script(monkeypatch, capsys, pre(seam, PUSH), "PreToolUse")
    assert seam.local == []


def test_a_must_not_contract_gates_the_same_way(seam, monkeypatch, capsys):
    seam.engine.upsert_contracts([FORBID])
    seam.fail("daemon_unavailable")
    response = run_script(monkeypatch, capsys, pre(seam, {"command": "git push --force origin x"}), "PreToolUse")
    assert seam.local == [1] and permission(response) == "deny" and "no-force" in json.dumps(response)


def test_a_local_judgement_that_fails_transiently_too_is_refused_to_the_agent_as_before(seam, monkeypatch, capsys):
    seam.engine.upsert_contracts([SATISFIED])
    monkeypatch.setenv("SELF_DIRECT_CLAUDE_ENFORCEMENT", "agent_review")
    monkeypatch.setenv("SELF_DIRECT_MCP_SERVER_NAME", "tsukkomi")
    seam.fail("daemon_unavailable")
    clock = Clock()
    monkeypatch.setattr(claude_hooks, "_monotonic", clock)
    monkeypatch.setattr(claude_hooks, "_sleep", clock.sleep)

    def busy(*args, **kwargs):
        raise IndexBusy()
    monkeypatch.setattr(claude_hooks, "handle_pre_tool_use", busy)
    response = run_script(monkeypatch, capsys, pre(seam, PUSH), "PreToolUse")
    assert seam.local == [1]
    assert permission(response) == "deny" and "audit could not run" in json.dumps(response)      # the agent's refusal


# --- the daemon reports a failure it did not answer -------------------------------------------------------------

def failing(monkeypatch, error):
    def handler(*args, **kwargs):
        raise error()
    monkeypatch.setattr(claude_hooks, "handle_pre_tool_use", handler)


@pytest.fixture
def clocked(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(claude_hooks, "_monotonic", clock)
    monkeypatch.setattr(claude_hooks, "_sleep", clock.sleep)
    return clock


@pytest.mark.parametrize("error", [IndexBusy, lambda: RequestStopped("request_deadline")],
                         ids=["index-busy", "request-stopped"])
def test_run_hook_hands_a_transient_failure_of_a_gated_call_to_its_caller_when_asked(env, clocked, monkeypatch, error):
    engine, path = env
    engine.upsert_contracts([SATISFIED])
    failing(monkeypatch, error)
    with pytest.raises(TransientHookFailure) as raised:
        run_hook("PreToolUse", payload(path, "PreToolUse", tool_input=PUSH), engine=engine, defer_transient=True)
    assert raised.value.error_class in ("IndexBusy", "RequestStopped")


def test_run_hook_answers_as_before_unless_it_is_asked_to_hand_over(env, clocked, monkeypatch):
    engine, path = env
    engine.upsert_contracts([SATISFIED])
    set_mode(engine, "agent_review")
    failing(monkeypatch, IndexBusy)
    body = payload(path, "PreToolUse", tool_input=PUSH)
    assert permission(run_hook("PreToolUse", body, engine=engine)) == "deny"                       # the agent's refusal
    ungated = payload(path, "PreToolUse", tool_input=UNGATED)
    assert "FAILED" in json.dumps(run_hook("PreToolUse", ungated, engine=engine, defer_transient=True))   # nothing to hand over


def test_a_failure_that_is_not_transient_is_never_handed_over(env, clocked, monkeypatch):
    engine, path = env
    engine.upsert_contracts([SATISFIED])
    failing(monkeypatch, ValueError)
    response = run_hook("PreToolUse", payload(path, "PreToolUse", tool_input=PUSH), engine=engine, defer_transient=True)
    assert permission(response) == "ask"


# --- the answer does not depend on the route --------------------------------------------------------------------

def failed_tests(path):
    write(path, assistant(use("toolu_p1", "Bash", command="pytest -q")), tool_result("toolu_p1", "1 failed", is_error=True),
          mode="a")


def unfinished_tests(path):
    write(path, assistant(use("toolu_p1", "Bash", command="pytest -q")), mode="a")      # a run with no result yet


SCENARIOS = {
    "clean": ([SATISFIED], write_passing_tests, PUSH, "none"),                 # the prerequisite is satisfied in history
    "unverified": ([SATISFIED], unfinished_tests, PUSH, "ask"),                # a run whose result is not known
    "violation": ([SATISFIED], failed_tests, PUSH, "deny"),                    # the last run failed
    "missing": ([SATISFIED], lambda path: None, PUSH, "deny"),                 # the required run never happened
    "forbidden": ([FORBID], lambda path: None, {"command": "git push --force origin x"}, "deny"),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_local_route_gives_the_answer_the_daemon_gives_for_the_same_history(seam, monkeypatch, capsys, name):
    """The daemon runs ``run_hook`` with its engine; the fallback runs the same function with the hook's own."""
    contracts, history, tool_input, expected = SCENARIOS[name]
    seam.engine.upsert_contracts(contracts)
    history(seam.path)
    body = pre(seam, tool_input)
    daemon_answer = run_hook("PreToolUse", body, engine=seam.engine)
    seam.fail("daemon_unavailable")
    routed = run_script(monkeypatch, capsys, body, "PreToolUse")
    assert seam.local == [1]
    assert permission(routed) == expected, routed                             # a scenario cannot pass by judging nothing
    assert routed == daemon_answer


# --- the payload survives the hand-over -------------------------------------------------------------------------

def test_a_large_payload_is_readable_again_by_the_process_that_replaces_this_one():
    code = (
        "import importlib.util, os, sys\n"
        "spec = importlib.util.spec_from_file_location('hook_script', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
        "text = sys.stdin.read()\n"
        "module._reexec_target = lambda: sys.executable\n"
        "module._replay_stdin(text)\n"
        "os.execv(sys.executable, [sys.executable, '-c', 'import sys; sys.stdout.write(sys.stdin.read())'])\n")
    text = json.dumps({"tool_input": {"command": "x" * 300_000}, "n": 1})
    done = subprocess.run([sys.executable, "-c", code, str(SCRIPT)], input=text, capture_output=True, text=True,
                          timeout=60, env={k: v for k, v in os.environ.items()})
    assert done.returncode == 0, done.stderr[-500:]
    assert done.stdout == text
