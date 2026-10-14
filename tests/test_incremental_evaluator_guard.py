"""A tripwire for the reference evaluator.

The incremental monitors (``audit/incremental.py``) re-implement what these functions do, in streaming form, and
stored monitor state is valid only for the evaluator version that wrote it. When this test fails, something the
monitors mirror has changed. Do not just update the digest:

1. change the monitors to match (``tests/test_incremental_monitors.py`` and ``tests/test_incremental_store.py`` are
   the differential tests that tell you whether they do);
2. bump ``audit.incremental.EVALUATOR_VERSION``, which discards every stored state and re-enables contracts that a
   shadow mismatch had switched off;
3. then update ``DIGEST`` below.
"""
from __future__ import annotations

import hashlib
import inspect

from self_directing_mcp import agy
from self_directing_mcp.audit import applicability, runner
from self_directing_mcp.retrieve import checks

DIGEST = "ce13195d48995a63"

MIRRORED = [
    runner.audit_contract, runner._Presence, runner._eligible, runner._worse, runner.evidence, runner._key,
    runner.audit_needs, runner.agy_evidence_rules, runner._window, runner._dependencies, runner._wants_attempts,
    runner._reads_results, runner._blocked_proposals, runner._consistent, runner._proposal_args, runner._cut_unknown,
    runner._unknown, runner._absent, runner._unresolved, runner._attempted, runner._attempt_hits,
    runner._hidden_trigger, runner.claude_evidence_rules, runner._cut_results, runner.full_findings,
    applicability.when_state, applicability._is_call_evidence, applicability.not_applicable_reason,
    applicability.when_needs, checks.scope_matches,
]
CONSTANTS = [sorted(agy.RESULT_TYPES), sorted(agy.SUCCESS_TYPES), sorted(map(list, agy.JUDGED_FIELDS)),
             sorted(runner._FINAL_STATUS), sorted(runner._TRUNCATED_CODE.items()), applicability.PROPOSED_CALL_PREFIX]


def digest():
    parts = ["".join(inspect.getsource(function).split()) for function in MIRRORED] + [repr(c) for c in CONSTANTS]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def test_the_reference_evaluator_is_the_one_the_monitors_mirror():
    assert digest() == DIGEST, (
        f"the reference evaluator changed (digest {digest()}). Update the incremental monitors, bump "
        "audit.incremental.EVALUATOR_VERSION, then set DIGEST in tests/test_incremental_evaluator_guard.py.")
