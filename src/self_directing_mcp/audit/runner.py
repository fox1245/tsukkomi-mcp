from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_left, bisect_right

from self_directing_mcp.agy import JUDGED_FIELDS, RESULT_TYPES, SUCCESS_TYPES
from self_directing_mcp.audit.applicability import (
    INVALID, UNMET, downgrade_unproven_inapplicability, not_applicable_reason, when_needs, when_state,
)
from self_directing_mcp.retrieve.checks import scope_matches
from self_directing_mcp.schemas import AuditResult, Evidence, Finding
from self_directing_mcp.security.mask import mask_secrets
from self_directing_mcp.neograph_runtime import run_stages
from self_directing_mcp.request_control import check_request, request_operation


def _checked(items):
    for item in items:
        check_request()
        yield item
    check_request()


def _worse(a, b):
    rank = {"clean": 0, "unknown": 1, "suspicious": 2, "violation": 3}
    return a if rank[a] >= rank[b] else b


def evidence(chunk, detail="regex match"):
    return Evidence(chunk_id=chunk.chunk_id, kind=chunk.kind,
                    snippet=mask_secrets(chunk.text)[:400], match_type="regex", detail=detail)


def _eligible(chunk, rule):
    check_request()
    return scope_matches(chunk, rule.scope) and chunk.meta.get("role") in rule.roles


def action_applies(rule, proposed) -> bool:
    """Determine action scope without mistaking completion obligations for gates."""
    check_request()
    if not _eligible(proposed, rule):
        return False
    if rule.type == "must" and not rule.before_regex:
        return False
    pattern = rule.before_regex if rule.type == "must" else rule.regex
    if not pattern:
        return True
    try:
        applies = re.search(pattern, proposed.text, re.IGNORECASE | re.MULTILINE) is not None
        check_request()
        return applies
    except re.error:
        return True


def prefix_note(coverage) -> str:
    return (f"Judged as of byte {coverage.get('judged_through_byte', 0)}; "
            f"{coverage.get('unread_bytes', 0)} appended byte(s) were not reviewed.")


def downgrade_unproven_absence(findings, rules) -> bool:
    """Late growth: a missing/failed obligation cannot be proven when bytes are unread. Dict findings."""
    kinds = {rule.id: rule.type for rule in rules}
    changed = False
    for item in findings:
        if kinds.get(item["contract_id"]) == "must" and item["verdict"] == "violation":
            item.update(verdict="unknown", reason="incomplete evidence cannot prove a missing or failed obligation",
                        basis=(item.get("basis") or "") + "; unread appended bytes prevent proving absence")
            changed = True
    return changed


def _key(value):
    try:
        hash(value)
        return value
    except TypeError:
        return json.dumps(value, sort_keys=True, default=str)


def audit_needs(active, proposed):
    """What the history reader must keep text for: ``(scope, roles)`` pairs and whether call linkage is used."""
    rules = [r for r in active if r.regex and (r.type == "must" or proposed is None)]
    needs = [(r.scope, tuple(r.roles)) for r in rules] + when_needs(rules)
    if any(_wants_attempts(r) for r in rules):
        needs.append(("attempt", ("runtime",)))  # host-logged model proposals, see agy_evidence_rules
    return needs, any(r.requires_success for r in rules)


class _Presence:
    """Index prerequisites once; each checkpoint is a binary search plus a bounded evidence slice."""

    def __init__(self, rule, regex, active):
        self.rule, self.active = rule, active
        self.hits, self.calls, self.outputs = [], {}, {}
        for index, chunk in enumerate(_checked(active)):
            if chunk.kind == "tool_call":
                self.calls.setdefault(_key(chunk.meta.get("call_id")), []).append(index)
            elif chunk.kind == "tool_result":
                self.outputs.setdefault(_key(chunk.meta.get("call_id")), []).append(index)
            if _eligible(chunk, rule) and regex.search(chunk.text):
                self.hits.append(index)

    def before(self, position):
        """Verdict for a checkpoint whose prefix is ``active[:position]``."""
        check_request()
        rule, active = self.rule, self.active
        hit_end = bisect_left(self.hits, position)
        if not hit_end:
            return "violation", "required evidence missing before checkpoint", []
        index = self.hits[hit_end - 1]
        chunk = active[index]
        if not rule.requires_success:
            return "clean", "required regex evidenced", [evidence(chunk)]
        # Only the latest matching call decides; a newer failed/pending run invalidates older success.
        # Positions are found by bisection, never by copying the prefix of hits or results.
        unresolved = True
        call_id = chunk.meta.get("call_id")
        if call_id and bisect_left(self.calls.get(_key(call_id), ()), position) == 1:
            outputs = self.outputs.get(_key(call_id), ())
            start, end = bisect_right(outputs, index), bisect_left(outputs, position)
            if end - start == 1 and active[outputs[start]].meta.get("success") is not None:
                output = active[outputs[start]]
                if output.meta["success"] is True:
                    return "clean", "required call has linked successful result", [
                        evidence(chunk), evidence(output, "linked successful tool result")]
                unresolved = False
        items = [evidence(active[i]) for i in self.hits[max(0, hit_end - 3):hit_end]]
        if unresolved:
            return "unknown", "required call has no unambiguous completion status", items
        return "violation", "required calls completed without success", items


def audit_contract(rule, *, session_id, store, retriever=None, provider=None, proposed=None, chunks=None):
    """Regex checks operate on events. Retrieval ranks are never verification evidence."""
    check_request()
    def finding(verdict, reason, items=None, basis=None, applicability="applicable"):
        check_request()
        return Finding(contract_id=rule.id, contract_revision=rule.revision, severity=rule.severity,
                       verdict=verdict, reason=reason, evidence=items or [], basis=basis,
                       applicability=applicability)

    if not rule.enabled:
        return finding("clean", "contract disabled")
    chunks = chunks if chunks is not None else store.list_chunks(session_id, provider)
    if (rule.provider and rule.provider != provider) or (rule.session_id and rule.session_id != session_id):
        return finding("clean", "contract outside this provider/session")
    ids = {c.chunk_id: i for i, c in enumerate(_checked(chunks))}
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
    condition = when_state(rule, active)
    if condition == INVALID:
        return finding("unknown", "invalid `when` regex", basis=f"{rule.id}.when failed to compile")
    if condition == UNMET:
        return finding("clean", not_applicable_reason(rule), basis=f"{rule.id}.when did not match any observed tool call",
                       applicability="not_applicable")
    if rule.type == "must_not":
        inspected = [proposed] if proposed is not None else active
        hits = [c for c in _checked(inspected) if _eligible(c, rule) and regex.search(c.text)]
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
            reason += f"; user said: {mask_secrets(rule.source_quote)[:120]}"
        return finding("violation", reason,
                       basis=f"{rule.id}.regex '{rule.regex}' matched prohibited content", items=[evidence(c) for c in hits[:20]])
    if trigger:
        if proposed is not None:
            checkpoints = [(len(active), proposed)] if _eligible(proposed, rule) and trigger.search(proposed.text) else []
        else:
            checkpoints = [(i, c) for i, c in enumerate(_checked(active)) if _eligible(c, rule) and trigger.search(c.text)]
        if not checkpoints:
            return finding("clean", "no triggering action in inspected events")
        overall, reasons, items = "clean", [], []
        presence = _Presence(rule, regex, active)
        for position, checkpoint in _checked(checkpoints):
            verdict, reason, supporting = presence.before(position)
            overall = _worse(overall, verdict)
            reasons.append(reason)
            items.extend([evidence(checkpoint, "triggering action"), *supporting])
        return finding(overall, "; ".join(dict.fromkeys(reasons)), items[:20])
    verdict, reason, supporting = _Presence(rule, regex, active).before(len(active))
    return finding(verdict, reason, supporting)


@request_operation
def run_audit(*, session_id, contracts, store, retriever=None, provider=None, proposed=None, coverage=None):
    snapshot = hashlib.sha256(json.dumps([c.model_dump(mode="json") for c in _checked(contracts)],
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
        active = [r for r in _checked(contracts) if r.enabled and (r.provider is None or r.provider == provider)
                  and (r.session_id is None or r.session_id == session_id)]
    def load_evidence():
        nonlocal chunks
        needs_history = any(r.regex and (r.type == "must" or proposed is None) for r in active)
        if needs_history:
            chunks = store.list_chunks(session_id, provider)
        else:
            anchors = {anchor for r in active for anchor in (r.source_event_id, r.applies_from_event_id) if anchor}
            chunks = [c for anchor in _checked(anchors) if (c := store.get_chunk(anchor)) is not None
                      and c.session_id == session_id and (provider is None or c.provider == provider)]
    def evaluate_contracts():
        nonlocal findings
        findings = [audit_contract(r, session_id=session_id, store=store, provider=provider,
                                   proposed=proposed, chunks=chunks) for r in _checked(active)]
        # Provider-specific partial-observation passes post-process the findings; other providers have none.
        evidence_rules = _EVIDENCE_RULES.get(provider)
        if evidence_rules is not None:
            evidence_rules(active, findings, chunks, proposed=proposed, coverage=coverage)
    def aggregate_findings():
        nonlocal overall
        if coverage and not coverage.get("complete"):
            for rule, item in _checked(zip(active, findings)):
                if rule.type == "must" and item.verdict == "violation":
                    item.verdict = "unknown"
                    item.reason = "incomplete evidence cannot prove a missing or failed obligation"
                    item.basis = (item.basis or "") + "; incomplete coverage prevents proving absence"
            downgrade_unproven_inapplicability(findings)
        overall = "clean"
        for item in _checked(findings):
            overall = _worse(overall, item.verdict)
        if not active or not store.chunk_count(session_id, provider):
            overall = _worse(overall, "unknown")
            notes.append("No applicable enabled contracts or no indexed events.")
        skipped = [item.contract_id for item in findings if item.applicability == "not_applicable"]
        if skipped:
            notes.append("not_applicable (`when` condition unmet, not dropped): " + ", ".join(skipped))
        if coverage and not coverage.get("complete"):
            if coverage.get("judged_prefix"):
                # Pure append: the verdict is for the captured range; appended bytes are unreviewed.
                notes.append(prefix_note(coverage))
            else:
                overall = _worse(overall, "unknown")
                notes.append("Audit coverage incomplete; inspect coverage.issues.")
    execution = run_stages("contract_audit", [
        ("select_contracts", select_contracts), ("load_evidence", load_evidence),
        ("evaluate_contracts", evaluate_contracts), ("aggregate_findings", aggregate_findings),
    ])
    return AuditResult(session_id=session_id, verdict=overall, findings=findings, notes=notes,
                       coverage=coverage or {}, contracts_snapshot=snapshot, execution=execution)


# --- AGY evidence classes -------------------------------------------------------------------------------
# Truncated fields, pending (RUNNING) results and legacy (pre-receipt) results change what the evidence can
# prove, never what it matched. Everything here post-processes findings and stays out of the contract logic:
#   * a visible match still counts (violation or satisfaction as before);
#   * a contract that would turn clean for lack of a match, or unmet for lack of evidence, while a field it
#     depends on is cut, is unknown with reason ``truncated_<field>``;
#   * a truncated USER_INPUT never grants a permission (exception, user-role satisfaction);
#   * a still-RUNNING result is ``pending_running``; a result with no receipt before the conversation's first one
#     is ``unsupported_legacy`` for obligations that need the executed call (never "unmet", never clean).

_FINAL_STATUS = frozenset({"DONE", "ERROR", "INVALID", "CANCELED"})
_TRUNCATED_CODE = {"content": "truncated_content", "tool_calls": "truncated_tool_calls"}


def _meta(event) -> dict:
    return getattr(event, "meta", None) or {}


def _wants_attempts(rule) -> bool:
    """Rules that also read the host-logged model proposals (the call side of the history)."""
    if rule.type == "must_not":
        return rule.scope in ("tool_call", "any") and "assistant" in rule.roles and "runtime" not in rule.roles
    return rule.scope in ("tool_call", "any")  # a must may be met, or only attempted, by a call


def _reads_results(rule) -> bool:
    """Whether the rule inspects tool-result text (a result in scope for a tool-role rule)."""
    return rule.scope in ("tool_result", "any") and "tool" in set(rule.roles)


def _dependencies(rule) -> set:
    """(step type, field) pairs this rule's verdict can depend on."""
    roles, scope = set(rule.roles), rule.scope
    deps = set()
    if scope in ("message", "any"):
        if "user" in roles:
            deps.add(("USER_INPUT", "content"))
        if "assistant" in roles:
            deps.add(("PLANNER_RESPONSE", "content"))
    if scope in ("tool_call", "any") and roles & {"assistant", "runtime"}:
        deps.add(("PLANNER_RESPONSE", "tool_calls"))
    if _reads_results(rule):
        deps.update((kind, "content") for kind in RESULT_TYPES)  # the CLI's GENERIC and every typed IDE result
    if rule.requires_success:
        deps.update((kind, "content") for kind in SUCCESS_TYPES)  # only results that can carry a success verdict
    return deps


def _proposal_args(text: str) -> dict:
    try:
        value = json.loads(text.split("] ", 1)[1])
    except (IndexError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _consistent(denied_args, args) -> bool:
    """A blocked call matches a proposal when every argument they share is equal (and they share one)."""
    if not isinstance(denied_args, dict) or not args:
        return False
    shared = set(denied_args) & set(args)
    return bool(shared) and all(denied_args[key] == args[key] for key in shared)


def _blocked_proposals(events) -> set:
    """Positions of model proposals the Pre hook demonstrably blocked (non-execution); they are not attempts."""
    waiting: dict = {}
    blocked = set()
    for index, event in enumerate(events):
        meta = _meta(event)
        if meta.get("evidence_origin") == "model_proposal":
            waiting.setdefault(meta.get("tool_name"), []).append((index, _proposal_args(event.text)))
        elif isinstance(meta.get("denied_action"), dict):
            denied = meta["denied_action"]
            queue = waiting.get(denied.get("name"), [])
            for slot in range(len(queue) - 1, -1, -1):
                if _consistent(denied.get("args"), queue[slot][1]):
                    blocked.add(queue.pop(slot)[0])
                    break
    return blocked


def _window(rule, events, ids, proposed):
    """Events a rule judges: after its activation anchor, and before its last triggering action."""
    start = ids[rule.applies_from_event_id] + 1 if rule.applies_from_event_id in ids else 0
    end = len(events)
    if rule.type == "must" and rule.before_regex and proposed is None:
        try:
            trigger = re.compile(rule.before_regex, re.IGNORECASE | re.MULTILINE)
        except re.error:
            return start, end
        positions = [i for i in range(start, end) if _eligible(events[i], rule) and trigger.search(events[i].text)]
        if positions:
            end = positions[-1]
    return start, end


def _unknown(item, code, why) -> None:
    item.verdict = "unknown"
    item.reason = f"{code}: {why}"
    item.basis = (item.basis or "") + f"; {code}"


def _absent(item) -> bool:
    return item.verdict == "violation" and "evidence missing" in item.reason


def _unresolved(item) -> bool:
    return item.verdict == "unknown" and "completion status" in item.reason


def _cut_unknown(rule, item, code, why, *, history=True) -> bool:
    """Partial observation, shared by every provider whose evidence can be cut.

    A cut dependency cannot show a missing match, so a prohibition that would turn clean for lack of a match, or an
    obligation that would turn unmet for lack of evidence, is unknown instead. A visible match is never touched.
    """
    unmet = (history and item.verdict == "clean") if rule.type == "must_not" else (_absent(item) or _unresolved(item))
    if unmet:
        _unknown(item, code, why)
    return unmet


def agy_evidence_rules(rules, findings, events, *, proposed=None, coverage=None) -> None:
    """Apply the partial-observation, pending and legacy rules above to ``findings`` (mutated in place)."""
    events = list(events)
    check_request()
    ids = {event.chunk_id: index for index, event in enumerate(events)}
    last_final = {}
    for index, event in enumerate(events):
        meta = _meta(event)
        if meta.get("event_type") in RESULT_TYPES and meta.get("status") in _FINAL_STATUS:
            last_final[meta.get("step_index")] = index
    history = proposed is None
    blocked = _blocked_proposals(events) if history and any(_wants_attempts(r) for r in rules) else set()
    for rule, item in zip(rules, findings):
        check_request()
        start, end = _window(rule, events, ids, proposed)
        scan = events[start:end]
        deps = _dependencies(rule)
        cut = [(event, field) for event in scan for field in (_meta(event).get("truncated_fields") or ())
               if (_meta(event).get("event_type"), field) in deps and (_meta(event).get("event_type"), field) in JUDGED_FIELDS]
        if rule.type == "must_not":
            if history and item.verdict != "unknown" and _wants_attempts(rule) and rule.regex:
                _attempt_hits(rule, item, events, range(start, end), blocked)
            if cut:
                _cut_unknown(rule, item, _TRUNCATED_CODE.get(cut[0][1], "truncated_" + cut[0][1]),
                             f"a still-truncated {_meta(cut[0][0]).get('event_type')}.{cut[0][1]} may hide a match",
                             history=history)
            if item.verdict == "suspicious" and rule.exception_regex and rule.source_event_id in ids:
                source = _meta(events[ids[rule.source_event_id]])
                if source.get("event_type") == "USER_INPUT" and "content" in (source.get("truncated_fields") or ()):
                    item.verdict = "violation"
                    item.reason = ("truncated_content: the exception comes from a truncated user message and "
                                   "cannot waive the match; " + item.reason)
            continue
        # must
        if item.verdict == "clean" and item.reason.startswith("no triggering action"):
            if rule.before_regex and history:
                code = _hidden_trigger(rule, events, range(start, end), cut, blocked)
                if code:
                    _unknown(item, *code)
            continue
        if item.verdict == "clean":
            supporting = [e.chunk_id for e in item.evidence if e.detail != "triggering action"]
            sources = [_meta(events[ids[c]]) for c in supporting if c in ids]
            if sources and len(sources) == len(supporting) and all(
                    s.get("event_type") == "USER_INPUT" and "content" in (s.get("truncated_fields") or ()) for s in sources):
                _unknown(item, "truncated_content", "the only satisfying evidence is a truncated user message")
            continue
        absent = _absent(item)
        if not (absent or _unresolved(item)):
            continue
        if cut:
            field = cut[0][1]
            _cut_unknown(rule, item, _TRUNCATED_CODE.get(field, "truncated_" + field),
                         f"a still-truncated {_meta(cut[0][0]).get('event_type')}.{field} may hold the missing evidence")
        elif absent and _wants_attempts(rule) and _attempted(rule, events, range(start, end), blocked):
            # Call side: a matching call was proposed, so the obligation may have been met; whether it ran, and
            # how it ended, is what the result side cannot show. A call never proposed is a real absence.
            if any(_meta(e).get("pending") and last_final.get(_meta(e).get("step_index"), -1) < start + i
                   for i, e in enumerate(scan)):
                _unknown(item, "pending_running", "a tool result is still RUNNING, so its outcome is unknown")
            elif any(_meta(e).get("linkage") == "legacy_unlinked" for e in scan):
                _unknown(item, "unsupported_legacy",
                         "results before this conversation's first host receipt cannot be linked to calls")
    if coverage is not None:
        unknown = [f for f in findings if f.verdict == "unknown"]
        legacy = sum(1 for f in unknown if f.reason.startswith("unsupported_legacy"))
        coverage["unknown_breakdown"] = {"current_session": len(unknown) - legacy, "unsupported_legacy": legacy}


# --- Claude Code evidence classes -----------------------------------------------------------------------
# A ``read_truncation_notice`` names (``truncated_result_for``) the tool result the host cut. Such a result is
# partial evidence: a visible match still counts, it never establishes success, and for lack of a match it can
# neither clear a prohibition nor prove an obligation unmet (``truncated_result``). The link is resolved here
# because the notice is a separate record; a streaming parser cannot mark a result it has already emitted.

def _cut_results(events) -> list:
    """Positions of tool results some notice marks as truncated, whatever order notice and result arrive in."""
    targets = {t for e in events if e.kind == "meta" and isinstance(t := _meta(e).get("truncated_result_for"), str) and t}
    if not targets:
        return []
    return [i for i, e in enumerate(events)
            if e.kind == "tool_result" and isinstance(_meta(e).get("call_id"), str) and _meta(e)["call_id"] in targets]


def claude_evidence_rules(rules, findings, events, *, proposed=None, coverage=None) -> None:
    """Apply the cut-result rules above to ``findings`` (mutated in place)."""
    events = list(events)
    check_request()
    cut_at = _cut_results(events)
    if not cut_at:
        return
    ids = {event.chunk_id: index for index, event in enumerate(events)}
    history = proposed is None
    for rule, item in zip(rules, findings):
        check_request()
        reads = _reads_results(rule)
        if not (reads or rule.requires_success):
            continue
        start, end = _window(rule, events, ids, proposed)
        cut = [events[i] for i in cut_at if start <= i < end]
        if not cut:
            continue
        if rule.type == "must" and item.verdict == "clean" and rule.requires_success:
            used = {e.chunk_id for e in item.evidence}
            if any(e.chunk_id in used for e in cut):
                _unknown(item, "truncated_result", "a cut tool result cannot establish success")
        elif reads:
            _cut_unknown(rule, item, "truncated_result",
                         "a cut tool result may hide the match" if rule.type == "must_not"
                         else "a cut tool result may hold the missing evidence", history=history)


_EVIDENCE_RULES = {"agy": agy_evidence_rules, "claude": claude_evidence_rules}


def _attempted(rule, events, positions, blocked) -> bool:
    try:
        regex = re.compile(rule.regex, re.IGNORECASE | re.MULTILINE)
    except re.error:
        return False
    return any(_meta(events[i]).get("evidence_origin") == "model_proposal" and i not in blocked
               and regex.search(events[i].text) for i in positions)


def _attempt_hits(rule, item, events, positions, blocked) -> None:
    """A forbidden call attempt is a violation whatever came back (ERROR, INVALID, no receipt): the host log
    preserves the proposal and its arguments. Proposals the Pre hook blocked never ran and are not counted."""
    try:
        regex = re.compile(rule.regex, re.IGNORECASE | re.MULTILINE)
        exception = re.compile(rule.exception_regex, re.IGNORECASE | re.MULTILINE) if rule.exception_regex else None
    except re.error:
        return
    hits = [events[i] for i in positions if _meta(events[i]).get("evidence_origin") == "model_proposal"
            and i not in blocked and regex.search(events[i].text)]
    if not hits:
        return
    covered = exception is not None and exception.search(" ".join(h.text for h in hits)) is not None
    verdict = "suspicious" if covered else "violation"
    if _worse(item.verdict, verdict) == item.verdict:
        return  # already at least this severe (it was matched on a receipt-linked call)
    reason = f"forbidden regex matched {len(hits)} host-logged call attempt(s); execution is not proven"
    if rule.source_quote:
        reason += f"; user said: {mask_secrets(rule.source_quote)[:120]}"
    item.verdict, item.reason = verdict, reason
    item.basis = f"{rule.id}.regex '{rule.regex}' matched a proposed call; the attempt itself is the violation"
    item.evidence = [evidence(hit, "forbidden call attempt (model proposal)") for hit in hits[:20]]


def _hidden_trigger(rule, events, positions, cut, blocked):
    """(code, why) when a clean "no triggering action" cannot be trusted: calls may be hidden or unlinked."""
    if cut:
        field = cut[0][1]
        return (_TRUNCATED_CODE.get(field, "truncated_" + field),
                f"a still-truncated {_meta(cut[0][0]).get('event_type')}.{field} may hide the triggering call")
    try:
        trigger = re.compile(rule.before_regex, re.IGNORECASE | re.MULTILINE)
    except re.error:
        return None
    window = [events[i] for i in positions]
    if any(_meta(e).get("linkage") == "legacy_unlinked" for e in window):
        if any(_meta(events[i]).get("evidence_origin") == "model_proposal" and i not in blocked
               and trigger.search(events[i].text) for i in positions):
            return ("unsupported_legacy",
                    "a matching call was proposed but its execution cannot be linked to a legacy result")
    return None
