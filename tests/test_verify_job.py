"""A hook process leaves its planned shadow comparisons to a detached verifier; answers that need one still get it.

Synthetic data only. The comparison of an incremental audit with a full evaluation is a quality check of the answer
that is already in hand. In a hook (one event, then exit) it used to run in the request and cost seconds on a long
transcript; it now runs in ``verify_job`` after the hook has answered, while a violation, a flip or a high-risk
contract is still compared in the request.
"""
from __future__ import annotations

import fcntl
import subprocess

import pytest

from self_directing_mcp import claude_hooks, verify_job
from self_directing_mcp.audit import incremental

from test_claude_hooks import SID, assistant, env, human, main_file, run, stop, tool_result, use, write  # noqa: F401

RULE = {"id": "no-deploy", "type": "must_not", "provider": "claude", "scope": "tool_call", "regex": "deploy --prod"}


def _grow(engine, path, count=4):
    engine.settings.audit_incremental_min_events = 1                # monitors are kept for a session this small
    for n in range(count):
        write(path, assistant(use(f"c{n}", "Bash", command=f"echo step {n}")), tool_result(f"c{n}", "ok"), mode="a")


def _hook(engine):
    """What the hook process does to its engine, on the production schedule (the suite compares every audit)."""
    engine.settings.audit_shadow_every = "adaptive"
    engine.defer_comparisons = True


def _audit(engine, path):
    return engine.audit_session(SID, provider="claude", path=str(path))


def test_without_the_flag_a_planned_comparison_runs_in_the_request(env):
    engine, path = env
    engine.upsert_contracts([RULE])
    _grow(engine, path)
    report = _audit(engine, path)["incremental"]
    assert report.get("shadow_compared") is True and "shadow_skipped" not in report
    assert engine.deferred_comparisons == {}


def test_a_hook_engine_leaves_the_planned_comparison_to_the_verifier_and_reads_no_reference(env):
    engine, path = env
    engine.upsert_contracts([RULE])
    _grow(engine, path)
    _audit(engine, path)                                              # states exist now: the next audits are deltas
    write(path, assistant(use("c9", "Bash", command="echo more")), tool_result("c9", "ok"), mode="a")
    _hook(engine)
    reads = []
    original = engine.store.capture_audit_events
    engine.store.capture_audit_events = lambda *a, **k: (reads.append(1), original(*a, **k))[1]
    report = _audit(engine, path)["incremental"]
    assert report["shadow_skipped"] == incremental.VERIFIER_SKIP and not report.get("shadow_compared")
    assert reads == []                                                # the whole session was not read for a comparison
    assert engine.deferred_comparisons == {("claude", SID): str(path)}


def test_a_violation_that_only_makes_a_note_is_compared_by_the_verifier_not_in_the_request(env):
    engine, path = env
    engine.upsert_contracts([RULE])
    _grow(engine, path)
    _audit(engine, path)
    write(path, assistant(use("bad", "Bash", command="deploy --prod now")), tool_result("bad", "ok"), mode="a")
    _hook(engine)
    result = _audit(engine, path)
    assert result["verdict"] == "violation"
    assert result["incremental"]["shadow_skipped"] == incremental.VERIFIER_SKIP
    assert not result["incremental"].get("shadow_compared")
    assert engine.deferred_comparisons == {("claude", SID): str(path)}


def test_a_miss_that_can_resume_the_agent_is_still_compared_in_the_request(env):
    engine, path = env
    engine.upsert_contracts([{"id": "ack", "type": "must", "scope": "message", "provider": "claude", "regex": "VERIFIED",
                              "auto_continue": True, "description": "state VERIFIED when done"}])
    _grow(engine, path)
    _audit(engine, path)
    write(path, assistant({"type": "text", "text": "done, nothing stated"}), mode="a")
    _hook(engine)
    result = _audit(engine, path)
    assert result["incremental"].get("shadow_compared") is True       # a block rests on it
    assert set(result["incremental"].get("shadow_scope") or ["ack"]) <= {"ack"}   # only what the block rests on


def test_a_violation_of_a_proposed_action_is_still_compared_in_the_request(env):
    engine, path = env
    engine.upsert_contracts([dict(RULE, id="gate", type="must", before_regex="deploy --prod", regex="tests passed")])
    _grow(engine, path)
    _hook(engine)
    engine.check_action(SID, {"tool_name": "Bash", "arguments": {"command": "echo hi"}}, provider="claude", path=str(path))
    engine.deferred_comparisons.clear()
    write(path, assistant(use("c8", "Bash", command="echo again")), tool_result("c8", "ok"), mode="a")
    action = {"tool_name": "Bash", "arguments": {"command": "deploy --prod"}}
    result = engine.check_action(SID, action, provider="claude", path=str(path))
    assert result["verdict"] == "violation"                            # a deny rests on it
    assert result["incremental"].get("shadow_compared") is True


def test_a_high_risk_contract_that_applies_to_an_unchanged_summary_is_compared_in_the_request_not_left_to_the_verifier(env):
    engine, path = env
    engine.upsert_contracts([dict(RULE, id="gate", type="must", before_regex="deploy --prod", regex="tests passed",
                                  risk="high")])
    engine.settings.audit_incremental_min_events = 1
    write(path, assistant(use("t0", "Bash", command="echo tests passed")), tool_result("t0", "ok"), mode="a")
    _grow(engine, path)
    _hook(engine)
    action = {"tool_name": "Bash", "arguments": {"command": "deploy --prod"}}
    engine.check_action(SID, action, provider="claude", path=str(path))        # builds the state
    write(path, assistant(use("c8", "Bash", command="echo unrelated")), tool_result("c8", "ok"), mode="a")
    engine.deferred_comparisons.clear()
    result = engine.check_action(SID, action, provider="claude", path=str(path))
    assert result["verdict"] == "clean"
    assert result["incremental"].get("shadow_compared") is True       # no answer called for it, the contract's risk did


def test_a_high_risk_contract_that_applies_to_the_action_is_still_compared_in_the_request(env):
    engine, path = env
    engine.upsert_contracts([dict(RULE, id="gate", type="must", scope="tool_call", before_regex="deploy --prod",
                                  regex="echo step 0", risk="high")])
    _grow(engine, path)
    _hook(engine)
    action = {"tool_name": "Bash", "arguments": {"command": "deploy --prod"}}
    result = engine.check_action(SID, action, provider="claude", path=str(path))
    assert result["incremental"].get("shadow_compared") is True


def test_the_setting_that_compares_every_audit_keeps_every_comparison_in_the_request(env):
    engine, path = env
    engine.upsert_contracts([RULE])
    _grow(engine, path)
    _audit(engine, path)
    write(path, assistant(use("c8", "Bash", command="echo again")), tool_result("c8", "ok"), mode="a")
    engine.settings.audit_shadow_every = 1                            # explicit: compare every audit, in the request
    engine.defer_comparisons = True                                   # a hook engine, but the setting is explicit
    report = _audit(engine, path)["incremental"]
    assert report.get("shadow_compared") is True and report.get("shadow_skipped") != incremental.VERIFIER_SKIP
    assert engine.deferred_comparisons == {}


def test_a_hook_does_not_start_another_process_while_a_verifier_runs(env, monkeypatch):
    engine, path = env
    engine.upsert_contracts([RULE])
    _grow(engine, path)
    _audit(engine, path)
    _hook(engine)
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kwargs: started.append(argv))
    write(path, assistant(use("c8", "Bash", command="echo again")), tool_result("c8", "ok"), mode="a")
    index = engine.settings.index_dir
    index.mkdir(parents=True, exist_ok=True)
    with open(verify_job.lock_path(index), "a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert verify_job.running(index) is True
        stop(engine, path)
    assert started == []
    assert verify_job.running(index) is False


def test_the_hook_starts_the_verifier_detached_after_it_has_answered(env, monkeypatch):
    engine, path = env
    engine.upsert_contracts([RULE])
    _grow(engine, path)
    _audit(engine, path)
    _hook(engine)
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kwargs: started.append((argv, kwargs)))
    write(path, assistant(use("c8", "Bash", command="echo again")), tool_result("c8", "ok"), mode="a")
    response = stop(engine, path)
    assert "decision" not in response                                  # the answer did not wait for any comparison
    assert len(started) == 1
    argv, kwargs = started[0]
    assert argv[1:5] == ["-I", "-m", "self_directing_mcp.verify_job", "--provider"] and "claude" in argv and SID in argv
    assert argv[argv.index("--index-dir") + 1] == str(engine.settings.index_dir)
    assert kwargs["start_new_session"] is True and kwargs["stdout"] is subprocess.DEVNULL
    assert engine.deferred_comparisons == {}


def test_a_failure_to_start_the_verifier_does_not_fail_the_hook(env, monkeypatch):
    engine, path = env
    engine.upsert_contracts([RULE])
    _grow(engine, path)
    _audit(engine, path)
    _hook(engine)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("no fork")))
    write(path, assistant(use("c7", "Bash", command="echo again")), tool_result("c7", "ok"), mode="a")
    assert "FAILED" not in str(stop(engine, path))


def test_the_verifier_compares_the_session_and_records_a_match(env):
    engine, path = env
    engine.upsert_contracts([RULE])
    _grow(engine, path)
    _audit(engine, path)
    _hook(engine)
    write(path, assistant(use("c6", "Bash", command="echo again")), tool_result("c6", "ok"), mode="a")
    _audit(engine, path)
    assert engine.deferred_comparisons
    assert verify_job.run("claude", SID, str(path), engine.settings.index_dir) == "compared"


def test_only_one_verifier_runs_per_index(env, monkeypatch):
    engine, path = env
    index = engine.settings.index_dir
    index.mkdir(parents=True, exist_ok=True)
    with open(verify_job.lock_path(index), "a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX)

        def never(*args, **kwargs):
            raise AssertionError("a second verifier must not start an engine")
        monkeypatch.setattr("self_directing_mcp.engine.SelfDirectEngine", never)
        assert verify_job.run("claude", SID, str(path), index) == "busy"
        assert verify_job.main(["--provider", "claude", "--session-id", SID, "--index-dir", str(index)]) == 0


def test_the_verifier_entry_point_reports_only_the_error_class(env, monkeypatch, capsys):
    engine, path = env

    def broken(*args, **kwargs):
        raise RuntimeError("SENTINEL-secret-path")
    monkeypatch.setattr(verify_job, "run", broken)
    assert verify_job.main(["--provider", "claude", "--session-id", SID, "--index-dir", str(engine.settings.index_dir)]) == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == "RuntimeError" and "SENTINEL" not in captured.out + captured.err


def test_a_hook_waits_for_the_index_as_long_as_its_audit_budget_instead_of_answering_busy(env):
    import threading
    import time
    from self_directing_mcp.claude_hooks import _new_engine
    from self_directing_mcp.request_control import IndexBusy
    engine, path = env
    engine.upsert_contracts([RULE])
    hook = _new_engine()
    try:
        assert hook.settings.lock_wait_timeout_sec >= hook.settings.audit_timeout_sec
        hook.settings.audit_timeout_sec = 30.0
        hook.settings.lock_wait_timeout_sec = 30.0               # what the hook engine derives from its budget
        held, release = threading.Event(), threading.Event()

        def hold():
            with engine._publication_guard():                    # the lease another process (a verifier) holds
                held.set()
                release.wait(10)
        thread = threading.Thread(target=hold)
        thread.start()
        held.wait(5)
        threading.Timer(1.5, release.set).start()
        started = time.monotonic()
        result = hook.check_action(SID, {"tool_name": "Bash", "arguments": {"command": "echo hi"}}, provider="claude",
                                   path=str(path))
        assert result["ok"] is True and time.monotonic() - started >= 1.0       # it waited, longer than the old second
        thread.join()
    except IndexBusy:
        raise AssertionError("the hook answered index busy instead of waiting for the lease")
    finally:
        hook.close()


def test_the_wait_before_the_next_verifier_grows_while_a_session_cannot_be_compared(tmp_path):
    from self_directing_mcp.claude_hook_state import ClaudeHookState
    store = ClaudeHookState(tmp_path)
    key = f"claude:{SID}"
    assert store.verifier_ready(key, 30.0, 900.0, now=1000.0)                       # never run
    store.verifier_done(key, False, now=1000.0)
    assert not store.verifier_ready(key, 30.0, 900.0, now=1029.0)
    assert store.verifier_ready(key, 30.0, 900.0, now=1030.0)
    store.verifier_done(key, False, now=1030.0)                                     # twice in a row: 60 s
    assert not store.verifier_ready(key, 30.0, 900.0, now=1089.0) and store.verifier_ready(key, 30.0, 900.0, now=1090.0)
    for _ in range(10):
        store.verifier_done(key, False, now=2000.0)
    assert not store.verifier_ready(key, 30.0, 900.0, now=2899.0) and store.verifier_ready(key, 30.0, 900.0, now=2900.0)
    store.verifier_done(key, True, now=3000.0)                                      # a comparison that ran clears it
    assert store.verifier_ready(key, 30.0, 900.0, now=3000.0)
    assert store.verifier_ready("claude:another-session", 30.0, 900.0, now=2000.0)


def test_a_hook_does_not_start_a_verifier_for_a_session_that_cannot_be_compared_until_its_wait_is_over(env, monkeypatch):
    engine, path = env
    engine.upsert_contracts([RULE])
    _grow(engine, path)
    _audit(engine, path)
    _hook(engine)
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kwargs: started.append(argv))
    from self_directing_mcp.claude_hook_state import ClaudeHookState
    state = ClaudeHookState(engine.settings.index_dir)
    state.verifier_done(f"claude:{SID}", False)                                     # the last one could not compare
    write(path, assistant(use("c8", "Bash", command="echo again")), tool_result("c8", "ok"), mode="a")
    stop(engine, path)
    assert started == []
    state.verifier_done(f"claude:{SID}", True)
    write(path, assistant(use("c9", "Bash", command="echo more")), tool_result("c9", "ok"), mode="a")
    stop(engine, path)
    assert len(started) == 1


def test_the_verifier_records_whether_it_could_compare(env, monkeypatch):
    from self_directing_mcp.claude_hook_state import ClaudeHookState
    engine, path = env
    engine.upsert_contracts([RULE])
    _grow(engine, path)
    _audit(engine, path)
    index = engine.settings.index_dir
    state = ClaudeHookState(index)
    assert verify_job.run("claude", SID, str(path), index) in ("compared", "not_compared")
    from self_directing_mcp.engine import SelfDirectEngine
    monkeypatch.setattr(SelfDirectEngine, "verify_incremental", lambda self, **kwargs: {"compared": False, "mismatch": []})
    assert verify_job.run("claude", SID, str(path), index) == "not_compared"
    assert not state.verifier_ready(f"claude:{SID}", 1e9, 1e9)                      # a run that could not compare waits
    monkeypatch.setattr(SelfDirectEngine, "verify_incremental", lambda self, **kwargs: {"compared": True, "mismatch": []})
    assert verify_job.run("claude", SID, str(path), index) == "compared"
    assert state.verifier_ready(f"claude:{SID}", 1e9, 1e9)
