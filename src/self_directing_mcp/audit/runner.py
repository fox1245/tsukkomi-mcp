from __future__ import annotations

import hashlib
import json
import re

from self_directing_mcp.retrieve.checks import scope_matches
from self_directing_mcp.schemas import AuditResult, Evidence, Finding
from self_directing_mcp.security.mask import mask_secrets
from self_directing_mcp.neograph_runtime import run_stages


def _worse(a, b):
    rank = {"clean": 0, "unknown": 1, "suspicious": 2, "violation": 3}
    return a if rank[a] >= rank[b] else b


def evidence(chunk, detail="regex match"):
    return Evidence(chunk_id=chunk.chunk_id, kind=chunk.kind,
                    snippet=mask_secrets(chunk.text)[:400], match_type="regex", detail=detail)


def _eligible(chunk, rule):
    return scope_matches(chunk, rule.scope) and chunk.meta.get("role") in rule.roles


def action_applies(rule, proposed) -> bool:
    """Determine action scope without mistaking completion obligations for gates."""
    if not _eligible(proposed, rule):
        return False
    if rule.type == "must" and not rule.before_regex:
        return False
    pattern = rule.before_regex if rule.type == "must" else rule.regex
    if not pattern:
        return True
    try:
        return re.search(pattern, proposed.text, re.IGNORECASE | re.MULTILINE) is not None
    except re.error:
        return True


def _presence(rule, matches, prefix):
    if not matches:
        return "violation", "required evidence missing before checkpoint", []
    if not rule.requires_success:
        return "clean", "required regex evidenced", [evidence(matches[-1])]
    unresolved = False
    # A newer failed/pending run invalidates older success at this checkpoint.
    for chunk in matches[-1:]:
        call_id = chunk.meta.get("call_id")
        if not call_id:
            unresolved = True
            continue
        calls = [c for c in prefix if c.kind == "tool_call" and c.meta.get("call_id") == call_id]
        if len(calls) != 1:
            unresolved = True
            continue
        position = next(i for i, c in enumerate(prefix) if c.chunk_id == chunk.chunk_id)
        outputs = [c for c in prefix[position + 1:]
                   if c.kind == "tool_result" and c.meta.get("call_id") == call_id]
        if len(outputs) != 1 or outputs[0].meta.get("success") is None:
            unresolved = True
            continue
        if outputs[0].meta["success"] is True:
            return "clean", "required call has linked successful result", [
                evidence(chunk), evidence(outputs[0], "linked successful tool result")]
    if unresolved:
        return "unknown", "required call has no unambiguous completion status", [evidence(c) for c in matches[-3:]]
    return "violation", "required calls completed without success", [evidence(c) for c in matches[-3:]]


def audit_contract(rule, *, session_id, store, retriever=None, provider=None, proposed=None, chunks=None):
    """Regex checks operate on events. Retrieval ranks are never verification evidence."""
    def finding(verdict, reason, items=None, basis=None):
        return Finding(contract_id=rule.id, contract_revision=rule.revision, severity=rule.severity,
                       verdict=verdict, reason=reason, evidence=items or [], basis=basis)

    if not rule.enabled:
        return finding("clean", "contract disabled")
    chunks = chunks if chunks is not None else store.list_chunks(session_id, provider)
    if (rule.provider and rule.provider != provider) or (rule.session_id and rule.session_id != session_id):
        return finding("clean", "contract outside this provider/session")
    ids = {c.chunk_id: i for i, c in enumerate(chunks)}
    for anchor in (rule.source_event_id, rule.applies_from_event_id):
        if anchor and anchor not in ids:
            return finding("unknown", "contract source or activation event is not in this session")
    if not rule.regex:
        return finding("unknown", "description-only contract needs a verifier; retrieval is not proof")
    try:
        regex = re.compile(rule.regex, re.IGNORECASE | re.MULTILINE)
        trigger = re.compile(rule.before_regex, re.IGNORECASE | re.MULTILINE) if rule.before_regex else None
    except re.error:
        return finding("unknown", "invalid contract regex")
    start = ids[rule.applies_from_event_id] + 1 if rule.applies_from_event_id else 0
    active = chunks[start:]
    if rule.type == "must_not":
        inspected = [proposed] if proposed is not None else active
        hits = [c for c in inspected if _eligible(c, rule) and regex.search(c.text)]
        if not hits:
            return finding("clean", "no forbidden match in inspected events",
                           basis=f"{rule.id}.regex did not match any inspected event")
        exception = None
        if rule.exception_regex:
            try:
                exception = re.compile(rule.exception_regex, re.IGNORECASE | re.MULTILINE)
            except re.error:
                return finding("unknown", "invalid exception_regex",
                               basis=f"{rule.id}.exception_regex failed to compile")
        if exception is not None and exception.search(inspected[0].text if proposed is not None else " ".join(c.text for c in hits)):
            return finding("suspicious", "forbidden match is covered by the authorized exception; confirm scope before proceeding",
                           basis=f"{rule.id}.exception_regex matched the same event(s); user-authorized exception requires confirmation")
        reason = f"forbidden regex matched {len(hits)} event(s)"
        if rule.source_quote:
            reason += f"; user said: {rule.source_quote[:120]}"
        return finding("violation", reason,
                       basis=f"{rule.id}.regex '{rule.regex}' matched prohibited content", items=[evidence(c) for c in hits[:20]])
    if trigger:
        if proposed is not None:
            checkpoints = [(len(active), proposed)] if _eligible(proposed, rule) and trigger.search(proposed.text) else []
        else:
            checkpoints = [(i, c) for i, c in enumerate(active) if _eligible(c, rule) and trigger.search(c.text)]
        if not checkpoints:
            return finding("clean", "no triggering action in inspected events")
        overall, reasons, items = "clean", [], []
        for position, checkpoint in checkpoints:
            prefix = active[:position]
            matches = [c for c in prefix if _eligible(c, rule) and regex.search(c.text)]
            verdict, reason, supporting = _presence(rule, matches, prefix)
            overall = _worse(overall, verdict)
            reasons.append(reason)
            items.extend([evidence(checkpoint, "triggering action"), *supporting])
        return finding(overall, "; ".join(dict.fromkeys(reasons)), items[:20])
    matches = [c for c in active if _eligible(c, rule) and regex.search(c.text)]
    verdict, reason, supporting = _presence(rule, matches, active)
    return finding(verdict, reason, supporting)


def run_audit(*, session_id, contracts, store, retriever=None, provider=None, proposed=None, coverage=None):
    snapshot = hashlib.sha256(json.dumps([c.model_dump(mode="json") for c in contracts],
                                        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    notes = [
        "Detect only. No action was executed or authorized.",
        "Violation/suspicious: stop and report. Unknown: resolve missing evidence; it is not compliance.",
        "Retrieval ranks are discovery signals, never a violation or satisfaction verdict.",
    ]
    active, chunks, findings = [], [], []
    overall = "unknown"
    def select_contracts():
        nonlocal active
        active = [r for r in contracts if r.enabled and (r.provider is None or r.provider == provider)
                  and (r.session_id is None or r.session_id == session_id)]
    def load_evidence():
        nonlocal chunks
        needs_history = any(r.regex and (r.type == "must" or proposed is None) for r in active)
        if needs_history:
            chunks = store.list_chunks(session_id, provider)
        else:
            anchors = {anchor for r in active for anchor in (r.source_event_id, r.applies_from_event_id) if anchor}
            chunks = [c for anchor in anchors if (c := store.get_chunk(anchor)) is not None
                      and c.session_id == session_id and (provider is None or c.provider == provider)]
    def evaluate_contracts():
        nonlocal findings
        findings = [audit_contract(r, session_id=session_id, store=store, provider=provider,
                                   proposed=proposed, chunks=chunks) for r in active]
    def aggregate_findings():
        nonlocal overall
        if coverage and not coverage.get("complete"):
            for rule, item in zip(active, findings):
                if rule.type == "must" and item.verdict == "violation":
                    item.verdict = "unknown"
                    item.reason = "incomplete evidence cannot prove a missing or failed obligation"
                    item.basis = (item.basis or "") + "; incomplete coverage prevents proving absence"
        overall = "clean"
        for item in findings:
            overall = _worse(overall, item.verdict)
        if not active or not store.chunk_count(session_id, provider):
            overall = _worse(overall, "unknown")
            notes.append("No applicable enabled contracts or no indexed events.")
        if coverage and not coverage.get("complete"):
            overall = _worse(overall, "unknown")
            notes.append("Audit coverage incomplete; inspect coverage.issues.")
    execution = run_stages("contract_audit", [
        ("select_contracts", select_contracts), ("load_evidence", load_evidence),
        ("evaluate_contracts", evaluate_contracts), ("aggregate_findings", aggregate_findings),
    ])
    return AuditResult(session_id=session_id, verdict=overall, findings=findings, notes=notes,
                       coverage=coverage or {}, contracts_snapshot=snapshot, execution=execution)
