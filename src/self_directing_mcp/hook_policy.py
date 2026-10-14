"""Host-neutral hook policy shared by the AGY and Claude Code adapters.

One judgement, host-specific enforcement. This module owns the decisions that must not differ between hosts:
how a verdict becomes ``allow`` / ``deny`` / ``force_ask`` / ``ask`` (``classify``), which unmet obligations may
restart a stopped agent (``confirmed_misses``), and the evidence a Stop checkpoint gathers (``checkpoint``).
Adapters only translate the neutral decisions into their host's response and own their persistent state.

Importing it never loads native code; the engine is passed in by the adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, TYPE_CHECKING

from self_directing_mcp.agy_hook_state import AUDIT_UNAVAILABLE
from self_directing_mcp.security.mask import mask_secrets

if TYPE_CHECKING:
    from self_directing_mcp.engine import SelfDirectEngine

NO_ACTION_CONTRACTS = "No applicable action contracts; not a compliance verdict."
NO_OUTSTANDING = "No outstanding obligations; not a compliance verdict."
_RANK = {"clean": 0, "unknown": 1, "suspicious": 2, "violation": 3}


class HookRuntimeError(RuntimeError):
    """The audit itself cannot run (settings, engine or contracts unavailable); the cause is chained."""
    kind = AUDIT_UNAVAILABLE


def short(text: Any, limit: int = 300) -> str:
    return mask_secrets(str(text))[:limit]


def budget_note(unit: str) -> str:
    return (f"Automatic continuation is allowed once per {unit} and is unavailable here "
            "(budget spent, repeated reason, or counter unreadable); not continuing.")


def classify(audit: dict, mode: str, high_risk: frozenset | set = frozenset()) -> tuple[str, str | None, bool]:
    """``(decision, reason, warn)``; ``warn`` marks an allow that rests on unverified evidence."""
    ok = audit.get("ok") is True
    verdict = audit.get("verdict", "unknown") if ok else "unknown"
    if ok and audit.get("action_scope") == "not_applicable":
        return "allow", "No triggering action contract; not a compliance verdict.", False
    findings = audit.get("findings", [])
    if (mode == "enforced" and verdict == "suspicious" and (
            any(f.get("verdict") in ("unknown", "violation") for f in findings
                if f.get("contract_id") in audit.get("action_contracts", []))
            or (audit.get("requires_history") and not audit.get("coverage", {}).get("complete")))):
        verdict = "unknown"  # Strict mode: confirming an exception cannot waive another prerequisite.
    if verdict == "clean":
        return "allow", None, False
    reasons = [f"[{f.get('contract_id')}] {short(f.get('reason', 'Verification unavailable'))}"
               for f in findings if f.get("verdict") != "clean"]
    reason = "; ".join(reasons[:4]) or "Contract verification incomplete; resolve missing evidence."
    if mode == "advisory":
        return "allow", "ADVISORY (not compliance): " + reason, False
    if verdict == "violation":
        return "deny", reason, False
    if verdict == "suspicious":
        return "force_ask", reason, False
    if mode == "enforced":
        return "deny", reason, False
    risky = sorted(set(high_risk) & set(audit.get("action_contracts", [])))
    if risky:
        return "ask", f"High-risk contract [{', '.join(risky)}] could not be verified: " + reason, False
    return "allow", "WARNING - unverified, not a compliance verdict: " + reason, True


def read_obligations(engine: SelfDirectEngine, provider: str, session_id: str, path: str | None) -> dict:
    try:
        return engine.hook_obligations(session_id, provider=provider, path=path)
    except Exception as exc:
        raise HookRuntimeError("contracts or checklist unavailable") from exc


@dataclass(frozen=True)
class Judgement:
    decision: str
    reason: str | None
    warn: bool
    audit: dict

    @property
    def silent(self) -> bool:
        """Nothing to tell the user: clean, or no contract triggered by this action (never a compliance claim)."""
        return self.reason is None or (self.audit.get("ok") is True and self.audit.get("action_scope") == "not_applicable")


def demote_history_findings(audit: dict, obligations: dict, why: str) -> dict:
    """Downgrade confirmed misses of history-dependent (``must``) contracts to unknown.

    For a call whose history this audit cannot see (a subagent's own transcript is a separate session), absence
    of evidence in the inspected transcript is not a confirmed miss. ``must_not`` findings stay as judged.
    """
    types = {rule.id: rule.type for rule in obligations["contracts"]}
    findings, changed = [], False
    for finding in audit.get("findings", []):
        if types.get(finding.get("contract_id")) == "must" and finding.get("verdict") in ("violation", "suspicious"):
            finding, changed = {**finding, "verdict": "unknown", "reason": why}, True
        findings.append(finding)
    if not changed:
        return audit
    remaining = [f["verdict"] for f in findings if f.get("verdict") in ("violation", "suspicious")]
    verdict = max(remaining, key=_RANK.__getitem__) if remaining else "unknown"
    return {**audit, "findings": findings, "verdict": verdict}


def judge_action(engine: SelfDirectEngine, provider: str, session_id: str, path: str | None, tool_name: str,
                 arguments: Any, mode: str, *, adjust: Callable[[dict, dict], dict] | None = None) -> Judgement | None:
    """Judge a proposed tool call. None means no contract applies to this session (not a verdict)."""
    obligations = read_obligations(engine, provider, session_id, path)
    if not obligations["contracts"]:
        return None
    # Hook time: verify the source cheaply (see ``integrity`` in the result); the engine settles it in full when
    # the verdict would rest on anything the cheap check cannot vouch for.
    audit = engine.check_action(session_id=session_id, provider=provider, path=path,
                                action={"tool_name": tool_name, "arguments": arguments}, verify="fast")
    if adjust is not None:
        audit = adjust(audit, obligations)
    high_risk = frozenset(rule.id for rule in obligations["contracts"] if rule.risk == "high")
    decision, reason, warn = classify(audit, mode, high_risk)
    return Judgement(decision, reason, warn, audit)


def reason_key(items: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(items)).encode()).hexdigest()


def confirmed_misses(obligations: dict, audit: dict) -> list:
    """Auto-continue obligations that are confirmed unmet on complete, current evidence; otherwise none.

    Restarting an agent rests on the strongest evidence only: the audit must have verified the source in FULL
    (``integrity.mode == "full"``); a fast or missing integrity report confirms nothing."""
    coverage = audit.get("coverage", {})
    if audit.get("ok") is not True or coverage.get("complete") is not True or coverage.get("issues"):
        return []
    if (audit.get("integrity") or {}).get("mode") != "full":
        return []
    rules = {rule.id: rule for rule in obligations["contracts"]
             if rule.auto_continue and rule.type == "must" and rule.enabled}
    return [(rules[f["contract_id"]], f) for f in audit.get("findings", [])
            if f.get("contract_id") in rules and f.get("verdict") == "violation"
            and f.get("applicability", "applicable") == "applicable"
            and f.get("contract_revision") == rules[f["contract_id"]].revision]


def miss_summary(misses: list) -> str:
    return "; ".join(f"[{rule.id}] {short(rule.description or rule.regex, 160)}" for rule, _f in misses)


def miss_key(misses: list, prefix: str = "") -> str:
    return reason_key([f"{prefix}{rule.id}@{rule.revision}" for rule, _f in misses])


def checkpoint(engine: SelfDirectEngine, provider: str, session_id: str, path: str, obligations: dict,
               label: str) -> tuple[list[str], list]:
    """Refresh history and gather ``(notes, confirmed_misses)`` for a Stop checkpoint. Notes never block."""
    synced = engine.sync_session(session_id=session_id, provider=provider, path=path, embed=False)
    notes = []
    coverage = synced.get("coverage", {})
    # Records the host is still appending are judged next time; any other gap leaves completion unverified.
    if not synced.get("ok") or not (coverage.get("complete") or set(coverage.get("issues", [])) == {"unread_bytes"}):
        notes.append(f"{label} history is incomplete; completion cannot be verified.")
    checklist = engine.get_checklist(session_id, provider=provider)
    pending = [item for item in checklist["items"] if item["status"] not in ("done", "revoked")]
    notes.extend(f"{item['text']} (status={item['status']}; missing verified completion evidence)"
                 for item in pending[:4])
    misses: list = []
    if obligations["contracts"]:
        audit = engine.audit_session(session_id, provider=provider, path=path)
        if audit.get("ok") is not True or audit.get("verdict") != "clean":
            notes.append("Contract checkpoint is " + str(audit.get("verdict", "unknown")) + ".")
        notes.extend(f"[{f['contract_id']}] {short(f.get('reason', ''))}" for f in audit.get("findings", [])
                     if f.get("verdict") != "clean")
        misses = confirmed_misses(obligations, audit)
    return notes, misses
