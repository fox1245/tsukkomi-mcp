"""A gated Claude Code PreToolUse call goes to the daemon first; only when that attempt fails for a transient cause is it
judged by the hook's own engine, inside one bound on the whole run.

Real hook processes, a real daemon process or an in-process daemon, and a held index lease, all under temporary
directories (the daemon fixtures of this folder). The routing logic with the daemon exchange replaced is unit-tested in
``tests/test_hook_gated_local.py`` and the bound in ``tests/test_hook_deadline.py``.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from self_directing_mcp.daemon import client
from self_directing_mcp.index.locking import index_lock
from self_directing_mcp.request_control import TransientHookFailure
import daemon_support as support
from daemon_support import CONTRACTS, World
from test_daemon_hooks import FakeDaemon, silent

# The synthetic session holds a passing test run, so this prerequisite of a deploy is satisfied in history.
GATED = {"id": "tests-before-deploy", "type": "must", "scope": "tool_call", "regex": "pytest", "before_regex": "deploy",
         "requires_success": True, "risk": "high"}
CONTRACTS_GATED = [c for c in CONTRACTS if c["id"] != "tests-before-deploy"] + [GATED]
SHORT = {"SELF_DIRECT_AUDIT_TIMEOUT_SEC": "2", "SELF_DIRECT_LOCK_WAIT_TIMEOUT_SEC": "0.5",
         "SELF_DIRECT_CLAUDE_ENFORCEMENT": "agent_review", "SELF_DIRECT_MCP_SERVER_NAME": "tsukkomi"}


def decision_of(run):
    return run.json.get("hookSpecificOutput", {}).get("permissionDecision", "none")


def hooks_served(runtime):
    return support.daemon_status(runtime)["requests"].get("hooks", 0)       # no key before the first one


def test_a_gated_call_and_an_ungated_one_are_both_served_by_the_daemon_when_it_can(world, runtime):
    index = world.new_index("route", CONTRACTS_GATED)
    try:
        world.start_daemon()
        gated = world.hook("claude", "PreToolUse", "deploy --prod", flag="on", index=index)
        assert gated.code == 0 and decision_of(gated) == "none" and "UNAVAILABLE" not in gated.stdout, gated
        assert hooks_served(runtime) == 1
        ungated = world.hook("claude", "PreToolUse", "ls", flag="on", index=index)
        assert ungated.code == 0 and "UNAVAILABLE" not in ungated.stdout
        assert hooks_served(runtime) == 2
    finally:
        runtime.stop()


@pytest.fixture
def failing_python():
    import shutil
    program = shutil.which("false")
    assert program
    return program


def test_a_gated_call_is_judged_locally_when_no_daemon_can_start(world, runtime, failing_python):
    index = world.new_index("nodaemon", CONTRACTS_GATED)
    gated = world.hook("claude", "PreToolUse", "deploy --prod", flag="on", index=index,
                       SELF_DIRECT_DAEMON_PYTHON=failing_python)
    assert gated.code == 0 and decision_of(gated) == "none", gated           # the prerequisite is satisfied: judged, passed
    assert "UNAVAILABLE" not in gated.stdout and "FAILED" not in gated.stdout
    ungated = world.hook("claude", "PreToolUse", "ls", flag="on", index=index, SELF_DIRECT_DAEMON_PYTHON=failing_python)
    assert "daemon_unavailable" in ungated.stdout                              # an ungated call fails open as before


def hold_lease(index, seconds):
    held = threading.Event()

    def holder():
        with index_lock(index, timeout=5):
            held.set()
            time.sleep(seconds)
    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    assert held.wait(10)
    return thread


def test_a_busy_index_that_outlasts_the_daemons_budget_is_waited_for_by_the_hooks_own_engine(world, runtime):
    """The daemon's judgement gives up after its 2 s (scaled down); the lease is held for 3.5 s. The daemon reports the
    failure instead of refusing the call, and the hook's own engine, which waits longer per acquisition, judges it."""
    index = world.new_index("churn", CONTRACTS_GATED)
    try:
        world.start_daemon()
        warm = world.hook("claude", "PreToolUse", "ls", flag="on", index=index, **SHORT)
        assert "UNAVAILABLE" not in warm.stdout and hooks_served(runtime) == 1
        thread = hold_lease(index, 3.5)
        started = time.monotonic()
        run = world.hook("claude", "PreToolUse", "deploy --prod", flag="on", index=index, **SHORT)
        waited = time.monotonic() - started
        thread.join(10)
        print(f"waited={waited:.2f}s decision={decision_of(run)} stdout={run.stdout[:300]}")
        assert run.code == 0 and decision_of(run) == "none", run
        assert "audit could not run" not in run.stdout and "FAILED" not in run.stdout
        assert waited >= 3.0                                                     # it waited for the holder
        assert hooks_served(runtime) == 1                                        # the daemon failed it; it did not answer
    finally:
        runtime.stop()


def test_the_daemon_reports_a_transient_failure_of_a_gated_call_instead_of_answering_it(world, runtime, inproc, monkeypatch):
    def refuses(host, event, payload, env, argv):
        raise TransientHookFailure("IndexBusy")
    inproc(hook_runner=refuses)
    index = world.new_index("report", CONTRACTS_GATED)
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(index))
    body = world.payload("claude", "PreToolUse", "deploy --prod")
    response, reason = client._serve("claude", "PreToolUse", body, [], dict(__import__("os").environ), None)
    assert response is None and reason == client.R_TRANSIENT
    again, served = client.serve_hook("claude", "PreToolUse", body, [], local_fallback=True)
    assert (again, served) == (None, False)                                       # the caller judges it
    ungated = world.payload("claude", "PreToolUse", "ls")
    answer, served = client.serve_hook("claude", "PreToolUse", ungated, [], local_fallback=True)
    assert served is False and answer is not None and "transient_failure" in json.dumps(answer)    # fails open as before


def test_the_attempt_at_a_daemon_that_never_answers_ends_at_the_bound(world, runtime, monkeypatch):
    index = world.new_index("silent-bound", CONTRACTS_GATED)
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(index))
    fake = FakeDaemon(runtime, silent)
    try:
        body = world.payload("claude", "PreToolUse", "deploy --prod")
        started = time.monotonic()
        response, served = client.serve_hook("claude", "PreToolUse", body, [], local_fallback=True,
                                             deadline=started + 1.0)
        elapsed = time.monotonic() - started
        assert served is False and response is None                           # transient, but no time is left to judge it
        assert elapsed < 2.5, elapsed                                            # not the request budget of 50 s
    finally:
        fake.close()
