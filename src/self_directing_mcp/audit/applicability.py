"""Contract applicability (``when``) helpers, kept apart from the presence/temporal evaluation.

A ``when`` condition is a regex over tool calls observed earlier in the session. It only scopes ``must``
obligations. Absence of a match is a claim about the whole session, so it is trusted only when the evidence
was complete; otherwise the contract stays unknown (see ``downgrade_unproven_inapplicability``).
"""
from __future__ import annotations

import re

from self_directing_mcp.audit.patterns import compile_contract

from self_directing_mcp.request_control import check_request

# AGY records a model's proposed call as a ``meta`` event with this text prefix until host receipts link it
# to an execution (agy.py). A proposal may be the only trace of a call, so it counts as the condition being
# met: treating it as absent would let an obligation silently vanish.
PROPOSED_CALL_PREFIX = "[proposed_tool_call:"

MET, UNMET, INVALID = "met", "unmet", "invalid"


def when_needs(rules):
    """Extra ``(scope, roles)`` pairs the history reader must keep text for, so ``when`` can see tool calls."""
    if not any(getattr(rule, "mode", "enforced") != "inject_only" and rule.when is not None and rule.regex for rule in rules):
        return []
    return [("tool_call", ("assistant",)), ("any", ("runtime",))]


def _is_call_evidence(event) -> bool:
    return event.kind == "tool_call" or event.text.startswith(PROPOSED_CALL_PREFIX)


def when_state(rule, events) -> str:
    """``met`` (or no condition), ``unmet``, or ``invalid`` for the events inside the rule's activation window."""
    if rule.when is None:
        return MET
    try:
        condition = compile_contract(rule.when.tool_call_regex)   # a timeout propagates to the contract's finding
    except re.error:
        return INVALID
    for event in events:
        check_request()
        if _is_call_evidence(event) and condition.search(event.text):
            return MET
    return UNMET


def not_applicable_reason(rule) -> str:
    return f"not_applicable: the `when` condition (tool call matching /{rule.when.tool_call_regex}/) was not observed in this session"


_STATIC_NOT_APPLICABLE_PREFIXES = (
    "not_applicable: contract mode is inject_only",
    "not_applicable: contract scope (",
)


def downgrade_unproven_inapplicability(findings) -> bool:
    """With incomplete coverage an unmatched condition may sit in unread or unparsed evidence: unknown, not N/A."""
    changed = False
    for item in findings:
        check_request()
        if item.applicability == "not_applicable" and not (
            isinstance(item.reason, str) and item.reason.startswith(_STATIC_NOT_APPLICABLE_PREFIXES)
        ):
            item.verdict, item.applicability = "unknown", "applicable"
            item.reason = "the `when` condition cannot be shown unmet because evidence is incomplete"
            item.basis = (item.basis or "") + "; incomplete coverage prevents proving the condition absent"
            changed = True
    return changed
