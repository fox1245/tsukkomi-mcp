"""A Claude Code PreToolUse call that a ``risk: high`` contract gates is judged by the hook's own engine, not sent to the
resident daemon; every other call is served by the daemon as before.

Synthetic data only. The hook script runs in this process with the daemon flag on and the daemon client replaced by a
recorder, so nothing here starts a daemon: a call that reaches the recorder was sent to the daemon, a call that reaches
``_new_engine`` was judged locally. The same flows against a real daemon process and a held index lease are in
``tests/daemon/test_daemon_gated_local.py``.
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

from self_directing_mcp import claude_hooks
from self_directing_mcp.claude_hooks import run_hook
from self_directing_mcp.daemon import client

from test_claude_hooks import (  # noqa: F401  (env is a fixture)
    assistant, env, human, main_file, payload, permission, set_mode, tool_result, use, write,
)
from test_hook_transient_audit import FORBID, GATE, PUSH, write_passing_tests

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "claude_hook.py"
STUB = {"systemMessage": "answered by the daemon stub"}
UNGATED = {"command": "ls"}
SATISFIED = {**GATE, "requires_success": True}

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the resident daemon is Linux only"),
    pytest.mark.skipif((ROOT / ".venv").exists(), reason="a checkout .venv would make the script re-execute this process"),
]


@pytest.fixture
def seam(env, monkeypatch):
    """The daemon switch on, the daemon client and the standalone engine factory replaced by recorders."""
    monkeypatch.setenv("SELF_DIRECT_DAEMON", "on")
    sent, local = [], []

    def serve_hook(host, event, payload, argv, **kwargs):
        sent.append((host, event))
        return dict(STUB), True

    real_new_engine = claude_hooks._new_engine

    def new_engine():
        local.append(1)
        return real_new_engine()
    monkeypatch.setattr(client, "serve_hook", serve_hook)
    monkeypatch.setattr(claude_hooks, "_new_engine", new_engine)
    monkeypatch.setattr(claude_hooks, "spawn_verifier", lambda engine: None)       # no detached process from a test
    return env[0], env[1], sent, local


def run_script(monkeypatch, capsys, body, event):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), f"--event={event}"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(body)))
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return json.loads(capsys.readouterr().out)


# --- the routing decision --------------------------------------------------------------------------------------

def test_a_gated_call_is_judged_by_a_local_engine_and_never_sent_to_the_daemon(seam, monkeypatch, capsys):
    engine, path, sent, local = seam
    engine.upsert_contracts([SATISFIED])
    write_passing_tests(path)
    response = run_script(monkeypatch, capsys, payload(path, "PreToolUse", tool_input=PUSH), "PreToolUse")
    assert sent == [] and local == [1]
    assert response != STUB and permission(response) == "none"           # judged for real: its prerequisite is satisfied


def test_an_ungated_call_is_sent_to_the_daemon_as_before(seam, monkeypatch, capsys):
    engine, path, sent, local = seam
    engine.upsert_contracts([SATISFIED])
    response = run_script(monkeypatch, capsys, payload(path, "PreToolUse", tool_input=UNGATED), "PreToolUse")
    assert sent == [("claude", "PreToolUse")] and local == []
    assert response == STUB


def test_advisory_mode_gates_nothing_so_the_daemon_serves_the_call(seam, monkeypatch, capsys):
    engine, path, sent, local = seam
    engine.upsert_contracts([SATISFIED])
    monkeypatch.setenv("SELF_DIRECT_CLAUDE_ENFORCEMENT", "advisory")
    response = run_script(monkeypatch, capsys, payload(path, "PreToolUse", tool_input=PUSH), "PreToolUse")
    assert sent == [("claude", "PreToolUse")] and local == [] and response == STUB


@pytest.mark.parametrize("event", ["Stop", "UserPromptSubmit", "PostToolUse", "PostToolUseFailure", "SessionStart"])
def test_only_a_pre_tool_use_call_can_be_judged_locally(seam, monkeypatch, capsys, event):
    engine, path, sent, local = seam
    engine.upsert_contracts([SATISFIED])
    run_script(monkeypatch, capsys, payload(path, event, tool_input=PUSH), event)
    assert sent == [("claude", event)] and local == []


def contracts_path(engine):
    return engine.settings.resolve_contracts_path()


def test_a_contracts_file_that_cannot_be_read_gates_nothing(seam, monkeypatch, capsys):
    engine, path, sent, local = seam
    engine.upsert_contracts([SATISFIED])
    contracts_path(engine).write_text("{ this is not json", encoding="utf-8")
    run_script(monkeypatch, capsys, payload(path, "PreToolUse", tool_input=PUSH), "PreToolUse")
    assert sent == [("claude", "PreToolUse")] and local == []


@pytest.mark.parametrize("change", [{"risk": "normal"}, {"enabled": False}, {"tools": ["Write"]},
                                    {"provider": "agy"}, {"before_regex": "kubectl apply"}],
                         ids=["not-high-risk", "disabled", "other-tool", "other-host", "other-trigger"])
def test_a_contract_that_does_not_gate_this_call_leaves_it_to_the_daemon(seam, monkeypatch, capsys, change):
    engine, path, sent, local = seam
    engine.upsert_contracts([{**SATISFIED, **change}])
    run_script(monkeypatch, capsys, payload(path, "PreToolUse", tool_input=PUSH), "PreToolUse")
    assert sent == [("claude", "PreToolUse")] and local == []


def test_a_must_not_contract_gates_the_same_way(seam, monkeypatch, capsys):
    engine, path, sent, local = seam
    engine.upsert_contracts([FORBID])
    run_script(monkeypatch, capsys, payload(path, "PreToolUse", tool_input={"command": "git push --force origin x"}),
               "PreToolUse")
    assert sent == [] and local == [1]


def test_the_routing_decision_is_the_typed_gate_of_the_hook_state(seam):
    engine, path, sent, local = seam
    engine.upsert_contracts([SATISFIED])
    gated, ungated = payload(path, "PreToolUse", tool_input=PUSH), payload(path, "PreToolUse", tool_input=UNGATED)
    assert client.judged_locally("claude", "PreToolUse", gated) is True
    assert client.judged_locally("claude", "PreToolUse", ungated) is False
    assert client.judged_locally("claude", "Stop", gated) is False
    assert client.judged_locally("agy", "PreToolUse", gated) is False
    assert client.judged_locally("claude", "PreToolUse", "not a payload") is False
    assert client.judged_locally("claude", "PreToolUse", {}) is False


# --- the answer does not depend on the route -------------------------------------------------------------------

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
    """The daemon runs ``run_hook`` with its engine; the local route runs the same function with the hook's own."""
    engine, path, sent, local = seam
    contracts, history, tool_input, expected = SCENARIOS[name]
    engine.upsert_contracts(contracts)
    history(path)
    body = payload(path, "PreToolUse", tool_input=tool_input)
    daemon_answer = run_hook("PreToolUse", body, engine=engine)
    routed = run_script(monkeypatch, capsys, body, "PreToolUse")
    assert sent == [] and local == [1]
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
