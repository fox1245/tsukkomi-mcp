"""One bound on the whole run of a Claude Code pre-tool hook, below the host's timeout for it.

A host that times a pre-tool hook out does not block the call, so a hook that runs past the host's timeout turns a refusal
into a pass. The bound is one setting (``hook_deadline_sec``) whose default is the host's timeout, taken from the table the
installer writes, less a named margin. It covers the attempt at the daemon, the retries and the local judgement together,
and at the bound the answer is the transient failure that is refused to the agent. Synthetic data only; the lease is held
by a thread of this process and the bounds are short, so nothing here waits for real seconds beyond them.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

import pytest

from self_directing_mcp import claude_hooks, hook_timeouts
from self_directing_mcp.config import Settings
from self_directing_mcp.index.locking import index_lock
from self_directing_mcp.request_control import current_request, request_scope
from self_directing_mcp.settings_core import load_hook_settings

from test_claude_hooks import env, payload, permission, set_mode, tool_result, use, write  # noqa: F401
from test_hook_gated_local import SATISFIED, PUSH, pre, run_script, seam  # noqa: F401  (seam is a fixture)
from test_hook_transient_audit import write_passing_tests

ROOT = Path(__file__).resolve().parents[1]
BOUND = 1.0


def installer():
    spec = importlib.util.spec_from_file_location("install_claude_hooks", ROOT / "scripts" / "install_claude_hooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Hold:
    """Another holder of the index lease for ``seconds``."""

    def __init__(self, index, seconds):
        self.index, self.seconds = index, seconds
        self.held = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with index_lock(self.index, timeout=5):
            self.held.set()
            time.sleep(self.seconds)

    def __enter__(self):
        self.thread.start()
        assert self.held.wait(10)
        return self

    def __exit__(self, *exc):
        self.thread.join(self.seconds + 10)


# --- one setting, derived from what the installer writes ---------------------------------------------------------

def test_the_default_bound_is_the_hosts_timeout_less_a_named_margin(env, monkeypatch):
    monkeypatch.delenv("SELF_DIRECT_HOOK_DEADLINE_SEC", raising=False)
    expected = hook_timeouts.HOST_TIMEOUT_SEC["PreToolUse"] - hook_timeouts.DEADLINE_MARGIN_SEC
    assert 0 < expected < hook_timeouts.HOST_TIMEOUT_SEC["PreToolUse"]
    assert load_hook_settings().hook_deadline_sec == expected
    assert Settings(_env_file=None).hook_deadline_sec == expected
    monkeypatch.setenv("SELF_DIRECT_HOOK_DEADLINE_SEC", "7")
    assert load_hook_settings().hook_deadline_sec == 7.0


def test_the_installer_writes_the_table_the_bound_is_derived_from():
    module = installer()
    assert module.HOST_TIMEOUT_SEC is hook_timeouts.HOST_TIMEOUT_SEC
    for event, (_matcher, timeout) in module.EVENT_SPECS.items():
        assert timeout == hook_timeouts.HOST_TIMEOUT_SEC[event]
    block = module.desired_hooks("python", "scripts/claude_hook.py")
    assert block["PreToolUse"][0]["hooks"][0]["timeout"] == hook_timeouts.HOST_TIMEOUT_SEC["PreToolUse"]


# --- the facts: what the engine's lease wait honours ---------------------------------------------------------------

def test_the_engines_lease_wait_honours_a_request_deadline(env, monkeypatch):
    """The wait for the lease ends at the request deadline, not after the hook's own wait (``audit_timeout_sec``)."""
    engine, path = env
    set_mode(engine, "agent_review")
    engine.upsert_contracts([SATISFIED])
    write_passing_tests(path)
    monkeypatch.setenv("SELF_DIRECT_CLAUDE_ENFORCEMENT", "agent_review")
    monkeypatch.setenv("SELF_DIRECT_MCP_SERVER_NAME", "tsukkomi")
    hook_engine = claude_hooks._new_engine()
    try:
        assert hook_engine.settings.lock_wait_timeout_sec >= 8.0            # each acquisition may wait the whole audit budget
        with Hold(engine.settings.index_dir, 4.0):
            started = time.monotonic()
            with request_scope(deadline=started + 0.5):
                response = claude_hooks.run_hook("PreToolUse", pre_payload(path), engine=hook_engine)
            elapsed = time.monotonic() - started
    finally:
        hook_engine.close()
    assert elapsed < 2.0, elapsed
    assert permission(response) == "deny" and "audit could not run" in json.dumps(response)     # the typed transient failure


def pre_payload(path):
    return payload(path, "PreToolUse", tool_input=PUSH)


# --- the standalone path: the whole run ends at the bound -----------------------------------------------------------

def run_main(monkeypatch, capsys, path):
    import io
    monkeypatch.setattr(sys, "argv", ["claude_hook.py", "--event=PreToolUse"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(pre_payload(path))))
    started = time.monotonic()
    claude_hooks.main()
    return json.loads(capsys.readouterr().out), time.monotonic() - started


@pytest.fixture
def agent_review(env, monkeypatch):
    engine, path = env
    monkeypatch.setenv("SELF_DIRECT_CLAUDE_ENFORCEMENT", "agent_review")
    monkeypatch.setenv("SELF_DIRECT_MCP_SERVER_NAME", "tsukkomi")
    monkeypatch.setenv("SELF_DIRECT_HOOK_DEADLINE_SEC", str(BOUND))
    monkeypatch.delenv("SELF_DIRECT_DAEMON", raising=False)
    monkeypatch.setattr(claude_hooks, "spawn_verifier", lambda engine: None)
    engine.upsert_contracts([SATISFIED])
    write_passing_tests(path)
    return engine, path


def test_the_standalone_hook_runs_under_a_request_deadline_at_the_bound(agent_review, monkeypatch, capsys):
    engine, path = agent_review
    seen = []
    real = claude_hooks.handle_pre_tool_use

    def spy(*args, **kwargs):
        control = current_request()
        seen.append(None if control is None else control.deadline)
        return real(*args, **kwargs)
    monkeypatch.setattr(claude_hooks, "handle_pre_tool_use", spy)
    started = time.monotonic()
    run_main(monkeypatch, capsys, path)
    assert seen and seen[0] is not None
    assert started + BOUND - 0.5 <= seen[0] <= time.monotonic() + BOUND


def test_a_lease_held_longer_than_the_bound_is_refused_to_the_agent_before_the_bound_runs_out(agent_review, monkeypatch, capsys):
    engine, path = agent_review
    with Hold(engine.settings.index_dir, 5 * BOUND):
        response, elapsed = run_main(monkeypatch, capsys, path)
    assert elapsed < 2 * BOUND, elapsed                                   # not the hook's 8 s wait for the lease
    assert permission(response) == "deny" and "audit could not run" in json.dumps(response)


def test_a_lease_held_shorter_than_the_bound_is_waited_for_and_the_call_is_judged(agent_review, monkeypatch, capsys):
    engine, path = agent_review
    monkeypatch.setenv("SELF_DIRECT_HOOK_DEADLINE_SEC", str(6 * BOUND))
    with Hold(engine.settings.index_dir, BOUND / 3):
        response, elapsed = run_main(monkeypatch, capsys, path)
    assert permission(response) == "none" and "audit could not run" not in json.dumps(response)
    assert elapsed >= BOUND / 3 - 0.1


# --- the attempt at the daemon counts against the same bound ----------------------------------------------------

def test_what_the_daemon_attempt_spent_is_taken_from_the_local_judgement(seam, monkeypatch, capsys):
    """The attempt used 0.6 of the 1.0 s bound; the lease is held for 5 s: the local judgement ends at what is left."""
    seam.engine.upsert_contracts([SATISFIED])
    write_passing_tests(seam.path)
    monkeypatch.setenv("SELF_DIRECT_CLAUDE_ENFORCEMENT", "agent_review")
    monkeypatch.setenv("SELF_DIRECT_MCP_SERVER_NAME", "tsukkomi")
    monkeypatch.setenv("SELF_DIRECT_HOOK_DEADLINE_SEC", str(BOUND))
    def spend(*args, **kwargs):
        seam.attempts.append(("claude", "PreToolUse"))
        seam.clock.now += 0.6
        return None, "daemon_unavailable"
    monkeypatch.setattr(client_module(), "_serve", spend)
    with Hold(seam.engine.settings.index_dir, 5.0):
        started = time.monotonic()
        response = run_script(monkeypatch, capsys, pre(seam, PUSH), "PreToolUse")
        elapsed = time.monotonic() - started
    assert seam.local == [1]
    assert elapsed < 1.5, elapsed                                          # 0.4 s were left, not 1.0
    assert permission(response) == "deny" and "audit could not run" in json.dumps(response)


def test_an_attempt_that_used_the_whole_bound_leaves_no_time_to_judge_and_is_refused_to_the_agent(seam, monkeypatch, capsys):
    seam.engine.upsert_contracts([SATISFIED])
    write_passing_tests(seam.path)
    monkeypatch.setenv("SELF_DIRECT_CLAUDE_ENFORCEMENT", "agent_review")
    monkeypatch.setenv("SELF_DIRECT_MCP_SERVER_NAME", "tsukkomi")
    monkeypatch.setenv("SELF_DIRECT_HOOK_DEADLINE_SEC", str(BOUND))

    def spend(*args, **kwargs):
        seam.clock.now += 2 * BOUND
        return None, "daemon_timeout"
    monkeypatch.setattr(client_module(), "_serve", spend)
    response = run_script(monkeypatch, capsys, pre(seam, PUSH), "PreToolUse")
    assert seam.local == []
    assert permission(response) == "deny" and "audit could not run" in json.dumps(response)


def client_module():
    from self_directing_mcp.daemon import client
    return client
