"""A high-risk contract asks for confirmation on its own finding, not on another contract's unknown."""
from self_directing_mcp import hook_policy

MODE = "confirmed_only"
GATE, OTHER_GATE, PROSE = "push-needs-check", "deploy-needs-review", "prose-only-rule"


def _finding(contract_id, verdict, reason="r"):
    return {"contract_id": contract_id, "verdict": verdict, "reason": reason}


def _audit(findings, action_contracts, verdict="unknown", ok=True):
    return {"ok": ok, "verdict": verdict, "action_scope": "applicable", "findings": findings,
            "action_contracts": action_contracts}


def test_a_verified_high_risk_contract_does_not_ask_because_another_contract_is_unknown():
    audit = _audit([_finding(GATE, "clean"), _finding(PROSE, "unknown", "description-only contract needs a verifier")],
                   [GATE, PROSE])
    decision, reason, warn = hook_policy.classify(audit, MODE, {GATE})
    assert decision == "allow" and warn is True
    assert PROSE in reason and "unverified" in reason and GATE not in reason


def test_an_unverified_high_risk_contract_asks_and_is_named():
    audit = _audit([_finding(GATE, "unknown", "incomplete current history"), _finding(PROSE, "unknown")], [GATE, PROSE])
    decision, reason, warn = hook_policy.classify(audit, MODE, {GATE})
    assert decision == "ask" and warn is False
    assert reason.startswith(f"High-risk contract [{GATE}] could not be verified")


def test_a_high_risk_contract_without_a_finding_asks():
    audit = _audit([_finding(PROSE, "unknown")], [GATE, PROSE])
    decision, reason, _ = hook_policy.classify(audit, MODE, {GATE})
    assert decision == "ask" and f"[{GATE}]" in reason


def test_only_the_unverified_high_risk_contracts_are_named():
    audit = _audit([_finding(GATE, "clean"), _finding(OTHER_GATE, "unknown"), _finding(PROSE, "unknown")],
                   [GATE, OTHER_GATE, PROSE])
    decision, reason, _ = hook_policy.classify(audit, MODE, {GATE, OTHER_GATE})
    assert decision == "ask" and reason.startswith(f"High-risk contract [{OTHER_GATE}] could not be verified")


def test_a_failed_audit_asks_for_every_high_risk_contract_of_the_action():
    audit = _audit([_finding(GATE, "clean")], [GATE], ok=False)
    decision, reason, _ = hook_policy.classify(audit, MODE, {GATE})
    assert decision == "ask" and f"[{GATE}]" in reason


def test_a_violation_of_a_high_risk_contract_still_denies():
    audit = _audit([_finding(GATE, "violation", "forbidden")], [GATE], verdict="violation")
    assert hook_policy.classify(audit, MODE, {GATE})[0] == "deny"


def test_enforced_mode_still_denies_an_unknown_verdict():
    audit = _audit([_finding(GATE, "clean"), _finding(PROSE, "unknown")], [GATE, PROSE])
    assert hook_policy.classify(audit, "enforced", {GATE})[0] == "deny"
