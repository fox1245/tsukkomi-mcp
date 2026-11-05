"""A contract that is too slow to be re-evaluated in full is not rebuilt on every audit.

Synthetic data only. The incremental monitors keep their state so that an audit folds only the events since the last
one. Two things used to throw a slow contract's state away on every audit of a long session, so that every Stop
rebuilt it from the whole history and ran the full evaluator over it:

* the time allowed for contract patterns in one fold was a flat 2 s per monitor, which a monitor that has to read the
  whole history on a loaded machine can run out of (``audit_fold_budget_sec`` is the budget of the fold now);
* the comparison with the full evaluator, which has the same 2 s per contract, timed out for such a contract, and a
  contract it could not compare lost its state (the comparison of a hook is left to the detached verifier, and the
  idle comparison, which only verifies, no longer drops a state it cannot compare).
"""
from __future__ import annotations

from self_directing_mcp.audit import incremental, patterns

from test_claude_hooks import SID, assistant, env, human, main_file, tool_result, use, write  # noqa: F401

RULE = {"id": "no-deploy", "type": "must_not", "provider": "claude", "scope": "tool_call", "regex": "deploy --prod"}


def _session(engine, path, count=6):
    engine.settings.audit_incremental_min_events = 1
    for n in range(count):
        write(path, assistant(use(f"c{n}", "Bash", command=f"echo step {n}")), tool_result(f"c{n}", "ok"), mode="a")


def _audit(engine, path, **kwargs):
    return engine.audit_session(SID, provider="claude", path=str(path), **kwargs)["incremental"]


def _hook(engine):
    engine.settings.audit_shadow_every = "adaptive"
    engine.defer_comparisons = True


def test_a_contract_too_slow_for_the_full_evaluator_is_built_once_and_then_folded_as_a_delta(env, monkeypatch):
    engine, path = env
    engine.upsert_contracts([RULE])
    _session(engine, path)
    _hook(engine)
    _audit(engine, path)                                              # the first audit builds the states
    monkeypatch.setattr(incremental, "EVALUATOR_VERSION", incremental.EVALUATOR_VERSION + 1)    # every state is stale now
    monkeypatch.setattr(patterns, "CONTRACT_BUDGET_SEC", 0.0)        # a pattern that no flat per-contract budget can hold
    assert _audit(engine, path)["rebuilt"]                            # rebuilt from the whole history, and kept
    write(path, assistant(use("c9", "Bash", command="echo more")), tool_result("c9", "ok"), mode="a")
    again = _audit(engine, path)
    assert again["mode"] == "delta" and again["rebuilt"] == {} and again["events_read"] < 10, again


def test_the_idle_comparison_does_not_drop_the_state_of_a_contract_it_cannot_compare(env, monkeypatch):
    engine, path = env
    engine.upsert_contracts([RULE])
    _session(engine, path)
    engine.settings.audit_shadow_every = "adaptive"
    _audit(engine, path)
    write(path, assistant(use("c8", "Bash", command="echo again")), tool_result("c8", "ok"), mode="a")
    assert _audit(engine, path)["mode"] == "delta"
    monkeypatch.setattr(patterns, "CONTRACT_BUDGET_SEC", 0.0)
    verified = engine.verify_incremental(session_id=SID, provider="claude", path=str(path))
    assert verified["skipped"] == "reference_regex_timeout" and verified["compared"] is False
    monkeypatch.undo()
    write(path, assistant(use("c9", "Bash", command="echo more")), tool_result("c9", "ok"), mode="a")
    after = _audit(engine, path)
    assert after["mode"] == "delta" and after["rebuilt"] == {}, after
