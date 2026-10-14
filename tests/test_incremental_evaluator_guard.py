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
from self_directing_mcp.audit import applicability, patterns, runner, shellstatus
from self_directing_mcp.retrieve import checks

DIGEST = "f325f6d4e3e85056"

MIRRORED = [
    runner.audit_contract, runner._Presence, runner._eligible, runner._worse, runner.evidence, runner._key,
    runner.audit_needs, runner.agy_evidence_rules, runner._window, runner._dependencies, runner._wants_attempts,
    runner._reads_results, runner._blocked_proposals, runner._consistent, runner._proposal_args, runner._cut_unknown,
    runner._unknown, runner._absent, runner._unresolved, runner._attempted, runner._attempt_hits,
    runner._hidden_trigger, runner.claude_evidence_rules, runner._cut_results, runner.full_findings,
    applicability.when_state, applicability._is_call_evidence, applicability.not_applicable_reason,
    applicability.when_needs, checks.scope_matches,
    # contract patterns run on the guarded ``regex`` engine; a timeout makes the contract unknown (never clean)
    runner._audit_contract, runner._post_process, runner._regex_timeout_finding, runner._patterns,
    patterns.compile_contract, patterns.GuardedPattern.search,
    # a success that a masked exit status cannot vouch for is unknown (``success_unverifiable``): one function, both evaluators
    runner.success_unverifiable, shellstatus,
    # a contract anchored to something other than a user message is unknown (``anchor_not_user``)
    runner.is_user_event,
    # a call the host refused is a rejected proposal, never an executed match (Claude Code ``permission_decision``)
    runner.rejected_calls, runner._host_denied, runner._wants_rejections, runner.rejections, runner.rejected_item,
    runner.rejected_clean, runner._event_byte, runner.rejected_proposal_item,
]
CONSTANTS = [sorted(agy.RESULT_TYPES), sorted(agy.SUCCESS_TYPES), sorted(map(list, agy.JUDGED_FIELDS)),
             sorted(runner._FINAL_STATUS), sorted(runner._TRUNCATED_CODE.items()), applicability.PROPOSED_CALL_PREFIX,
             runner.REGEX_TIMEOUT, runner.SUCCESS_UNVERIFIABLE, runner.ANCHOR_NOT_USER,
             runner.REJECTED_REASON, runner.REJECTED_LIMIT, runner.REJECTED_PROPOSAL_REASON]


def digest():
    parts = ["".join(inspect.getsource(function).split()) for function in MIRRORED] + [repr(c) for c in CONSTANTS]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def test_the_reference_evaluator_is_the_one_the_monitors_mirror():
    assert digest() == DIGEST, (
        f"the reference evaluator changed (digest {digest()}). Update the incremental monitors, bump "
        "audit.incremental.EVALUATOR_VERSION, then set DIGEST in tests/test_incremental_evaluator_guard.py.")
