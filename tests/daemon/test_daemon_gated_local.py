"""A Claude Code PreToolUse call that a ``risk: high`` contract gates is judged by the hook's own engine, not sent to the daemon.

Real hook processes, a real daemon process and a held index lease, all under temporary directories (the daemon
fixtures of this folder). The routing decision itself is unit-tested in ``tests/test_hook_gated_local.py``.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from self_directing_mcp.index.locking import index_lock
import daemon_support as support
from daemon_support import CONTRACTS, World

# The synthetic session holds a passing test run, so this prerequisite of a deploy is satisfied in history.
GATED = {"id": "tests-before-deploy", "type": "must", "scope": "tool_call", "regex": "pytest", "before_regex": "deploy",
         "requires_success": True, "risk": "high"}
CONTRACTS_GATED = [c for c in CONTRACTS if c["id"] != "tests-before-deploy"] + [GATED]
HOLD_SEC = 3.0


def decision_of(run):
    return run.json.get("hookSpecificOutput", {}).get("permissionDecision", "none")


def hooks_served(runtime):
    return support.daemon_status(runtime)["requests"].get("hooks", 0)       # no key before the first one


def test_a_gated_call_is_not_sent_to_the_daemon_and_an_ungated_one_is(world, runtime):
    index = world.new_index("route", CONTRACTS_GATED)
    try:
        world.start_daemon()
        gated = world.hook("claude", "PreToolUse", "deploy --prod", flag="on", index=index)
        assert gated.code == 0 and decision_of(gated) == "none", gated
        assert "UNAVAILABLE" not in gated.stdout and "FAILED" not in gated.stdout
        assert hooks_served(runtime) == 0                                  # the daemon never saw it
        ungated = world.hook("claude", "PreToolUse", "ls", flag="on", index=index)
        assert ungated.code == 0 and "UNAVAILABLE" not in ungated.stdout
        assert hooks_served(runtime) == 1                                  # an ungated call is served as before
    finally:
        runtime.stop()


def test_advisory_mode_gates_nothing_so_every_call_goes_to_the_daemon(world, runtime):
    index = world.new_index("advisory", CONTRACTS_GATED)
    try:
        world.start_daemon()
        run = world.hook("claude", "PreToolUse", "deploy --prod", flag="on", index=index,
                         SELF_DIRECT_CLAUDE_ENFORCEMENT="advisory")
        assert run.code == 0 and "UNAVAILABLE" not in run.stdout, run
        assert hooks_served(runtime) == 1
    finally:
        runtime.stop()


def hold_lease(index, seconds):
    """Another holder of the index lease (a flock on the index directory), as the engines of other sessions are."""
    held = threading.Event()

    def holder():
        with index_lock(index, timeout=5):
            held.set()
            time.sleep(seconds)
    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    assert held.wait(10)
    return thread


@pytest.mark.parametrize("mode", ["confirmed_only", "agent_review"])
def test_a_gated_call_whose_prerequisite_is_satisfied_passes_after_waiting_for_a_held_lease(world, runtime, mode):
    index = world.new_index("lease", CONTRACTS_GATED)
    try:
        world.start_daemon()
        warm = world.hook("claude", "PreToolUse", "ls", flag="on", index=index)          # the daemon has its engine
        assert "UNAVAILABLE" not in warm.stdout and hooks_served(runtime) == 1
        thread = hold_lease(index, HOLD_SEC)
        started = time.monotonic()
        run = world.hook("claude", "PreToolUse", "deploy --prod", flag="on", index=index,
                         SELF_DIRECT_CLAUDE_ENFORCEMENT=mode, SELF_DIRECT_MCP_SERVER_NAME="tsukkomi")
        waited = time.monotonic() - started
        thread.join(HOLD_SEC + 5)
        print(f"mode={mode} waited={waited:.2f}s decision={decision_of(run)} stdout={run.stdout[:300]}")
        assert run.code == 0 and decision_of(run) == "none", run          # not refused, not asked
        assert "UNAVAILABLE" not in run.stdout and "FAILED" not in run.stdout and "audit could not run" not in run.stdout
        assert waited >= HOLD_SEC - 0.5                                     # it really waited for the holder
        assert hooks_served(runtime) == 1                                   # and the daemon never judged it
    finally:
        runtime.stop()
