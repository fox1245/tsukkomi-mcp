"""Incremental contract audit: monitors whose state summarizes the already-judged prefix of a session.

``runner.audit_contract`` (plus the provider evidence classes) judges a rule by walking the whole event list.
This module judges the same rule by keeping, per rule, a small summary of the prefix and folding in only the
events that arrived after it. The two must agree on every input; the full evaluator is the reference and wins
whenever they differ (see ``AuditPlan`` and the shadow comparison in ``engine``).

What is summarized, per contract kind (all state is bounded and JSON-serializable):

* anchors (``source_event_id``, ``applies_from_event_id``): whether each was seen and where the activation
  event sits; events before an activation anchor are outside the rule's window; whether the source event is a user
  message (``anchor_not_user`` otherwise);
* ``when``: whether a call-evidence event inside the window matched;
* ``must_not``: the match count, the first 20 matches as evidence and, with an ``exception_regex`` (which the
  reference matches against the join of every hit), the hit texts up to a cap;
* ``must``: the last three regex hits as evidence plus, for ``requires_success``, a tracker of the latest hit's
  call (how many calls share its id, how many results followed, whether the first one succeeded, and whether that
  success is the required command's at all: ``shellstatus``);
* ``before_regex``: each triggering action is judged at its own position (the verdict depends on the prefix
  only), so the worst verdict, the distinct reasons, the first 20 evidence items and the last trigger position
  are kept;
* provider evidence classes: Claude's cut results and AGY's truncated fields, pending results, legacy results,
  host-logged proposals and denials, and the calls a Grok hook blocked (the shared ``hostblocks.HostBlocks`` join,
  persisted whole), as shared "facts" plus a few first-occurrence markers per rule.

A summary that cannot hold an input (a table over its cap, an AGY denial whose proposal lies outside the
look-back window) raises ``NeedFull``: that rule is then evaluated by the reference and the reason is
reported. Nothing is ever dropped silently, and nothing here can turn an unknown into a clean.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from bisect import bisect_left

from self_directing_mcp import contract_paths
from self_directing_mcp.agy import JUDGED_FIELDS, RESULT_TYPES
from self_directing_mcp.audit import hostblocks, runner, shellstatus
from self_directing_mcp.audit import shadow as shadow_module
from self_directing_mcp.audit.patterns import ContractRegexTimeout, compile_contract, contract_budget
from self_directing_mcp.audit import patterns as _patterns
from self_directing_mcp.audit.applicability import PROPOSED_CALL_PREFIX, not_applicable_reason
from self_directing_mcp.eval_cost import COSTS, REFERENCE_PER_EVAL
from self_directing_mcp.request_control import IndexBusy, RequestStopped, check_request, current_request
from self_directing_mcp.schemas import Evidence, Finding
from self_directing_mcp.security.mask import mask_secrets

# Bump when the semantics of any evaluation changes (rules, evidence classes, evidence text); it invalidates every
# stored state and re-enables contracts that a shadow mismatch disabled. (8: the reference evaluator shares its position
# index, call/result tables and rare-evidence lists between rules; results are identical, but the tripwire in
# tests/test_incremental_evaluator_guard.py asks for a bump whenever a mirrored function changes. 9: Grok's blocked
# calls are rejected proposals too (``runner.blocked_calls`` / ``GrokFacts``). 10: the reason and basis of a typed
# ``path_guard`` contract name the guarded path (``contract_paths.forbidden_label`` / ``verifier_basis``), and the
# fingerprint of its saved state includes the regex derived from it; regex contracts judge and read exactly as before.)
EVALUATOR_VERSION = 11
STATE_VERSION = 1

# The full evaluator costs about this many seconds per stored event (measured on a 225k-event session). A shadow run
# is skipped when the request's remaining budget cannot cover it with margin; it must never turn a request that
# the full evaluator alone would have answered in time into a timeout.
SHADOW_SECONDS_PER_EVENT = 30e-6

# A comparison never takes the last part of a request's time budget: this much (at most a quarter of what is left) stays
# for answering from the evaluation that was already made (payload, authority checks, the state write). A forced second pass
# that does not finish within the rest is dropped and the answer of the first pass stands, unverified.
FINISH_RESERVE_SEC = 1.0


def comparison_reserve(remaining: float) -> float:
    return min(FINISH_RESERVE_SEC, 0.25 * max(remaining, 0.0))

EVIDENCE_LIMIT = 20
EXCEPTION_MAX_HITS = 512
EXCEPTION_MAX_CHARS = 262_144
ATTEMPT_LIST = 64
PENDING_MAX = 2048
BLOCKED_MAX = 512
TARGETS_MAX = 4096
CUT_MAX = 2048
REJECTED_MAX = 2048    # refused calls remembered per session
REJ_HITS_MAX = 512     # a rule's matching calls remembered, to subtract the refused ones
REJ_EVIDENCE = 64      # evidence kept for such a rule: enough to refill the 20 shown when refused calls drop out

_SCOPE_KIND = {"tool_call": "tool_call", "tool_result": "tool_result", "message": "turn"}


class NeedFull(Exception):
    """A summary cannot decide this input exactly; the reference evaluator must."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


# Contract patterns run with a time limit (audit/patterns.py). A timeout drops the monitor for the run and the rule is
# judged by the reference evaluator, whose per-contract budget turns it into an unknown finding.
RegexTimeout = ContractRegexTimeout
TIMEOUTS = (ContractRegexTimeout,)


def _compile(pattern):
    """The only place this module compiles a contract pattern (same flags as the reference evaluator).

    A pattern that times out never produces a finding or a stored state here: the monitor is dropped for the run
    (``RuleMonitor.timed_out``), the rule is judged by the full evaluator, and ``AuditPlan.evaluate`` is the one
    place that handles it.
    """
    return compile_contract(pattern)


def _k(value):
    return runner._key(value)


def _tuple(value):
    return None if value is None else tuple(value)


def _truncated_user(meta) -> bool:
    return meta.get("event_type") == "USER_INPUT" and "content" in (meta.get("truncated_fields") or ())


def _ev(event, detail="regex match", flag=False):
    item = {"i": event.chunk_id, "k": event.kind, "s": mask_secrets(event.text)[:400], "d": detail}
    if flag:
        item["tu"] = _truncated_user(event.meta)
    return item


def _evidence(item):
    return Evidence(chunk_id=item["i"], kind=item["k"], snippet=item["s"], match_type="regex", detail=item["d"])


def mask_fingerprint() -> str:
    """Evidence snippets are stored masked; a different masking must not reuse them."""
    from self_directing_mcp.security import mask
    cached = getattr(mask_fingerprint, "_cached", None)
    if cached is None:
        rules = [(hints, pattern.pattern, pattern.flags) for hints, pattern, _replacement in mask._RULES]
        cached = hashlib.sha256(repr(rules).encode()).hexdigest()[:16]
        mask_fingerprint._cached = cached
    from self_directing_mcp.security import mask as module
    return f"{cached}:{len(module._KNOWN)}"


# --- call and proposal views over the events a run reads ---------------------------------------------------------

class Pools:
    """Calls, results and model proposals seen so far, for monitors that must look back.

    ``*_prior`` hold what the store held before the first event of this run (prefetched under the read boundary);
    everything fed in this run is noted as it goes. The evaluation itself never touches the database.
    """

    def __init__(self, track_calls=False, track_proposals=False, calls_prior=None, proposals_prior=None):
        self.track_calls, self.track_proposals = track_calls, track_proposals
        self.calls_prior = calls_prior or {}
        self.proposals_prior = proposals_prior or {}
        self.calls_seen, self.proposals_seen = {}, {}

    def note(self, event):
        meta = event.meta
        if self.track_calls:
            call_id = meta.get("call_id")
            if call_id is not None and (event.kind == "tool_call" or event.kind == "tool_result"):
                self.calls_seen.setdefault(_k(call_id), []).append((event.kind, event.key, event.chunk_id))
        if self.track_proposals and meta.get("evidence_origin") == "model_proposal":
            self.proposals_seen.setdefault(meta.get("tool_name"), []).append((event.key, event.text))

    def call_events(self, call_id, before):
        key = _k(call_id)
        for source in (self.calls_prior.get(key, ()), self.calls_seen.get(key, ())):
            for kind, position, chunk_id in source:
                if position < before:
                    yield kind, position, chunk_id

    def earlier_calls(self, call_id, after, before):
        """Calls sharing ``call_id`` strictly inside ``(after, before)``."""
        return sum(1 for kind, position, _chunk in self.call_events(call_id, before)
                   if kind == "tool_call" and (after is None or position > after))

    def proposals(self, name, before):
        """Proposals of tool ``name`` before ``before``, newest first, as ``(key, text)``."""
        for position, text in reversed(self.proposals_seen.get(name, ())):
            if position < before:
                yield position, text
        yield from self.proposals_prior.get(name, ((), True))[0]

    def proposals_complete(self, name):
        return self.proposals_prior.get(name, ((), True))[1]


# --- shared facts ---------------------------------------------------------------------------------------------------

class Facts:
    """Per-session summaries shared by every rule of a provider's evidence classes."""

    overflow = None
    dirty = False
    needed = False  # whether rule monitors read it

    def feed(self, event, pools):
        return None

    def dump(self):
        return {"overflow": self.overflow}

    def load(self, data):
        self.overflow = data.get("overflow")


class ClaudeFacts(Facts):
    """Tool results that a ``read_truncation_notice`` marks as cut, whatever order notice and result arrive in, and the
    calls the host recorded as refused (a ``deny`` decision on the linked result; see ``runner.rejected_calls``)."""

    needed = True

    def __init__(self):
        self.targets, self.cut = [], []
        self._targets = set()
        self.rejected = {}  # chunk id of a refused tool call -> [chunk id of the result carrying the decision, source]

    def feed(self, event, pools):
        meta = event.meta
        if event.kind == "meta":
            target = meta.get("truncated_result_for")
            if isinstance(target, str) and target and target not in self._targets:
                if len(self.targets) >= TARGETS_MAX:
                    self.overflow = "notice target table is full"
                    return
                self._targets.add(target)
                self.targets.append(target)
                self.dirty = True
                for kind, position, chunk_id in pools.call_events(target, event.key):
                    if kind == "tool_result":
                        self._cut(position, chunk_id)
        if event.kind == "tool_result":
            call_id = meta.get("call_id")
            if isinstance(call_id, str) and call_id in self._targets:
                self._cut(event.key, event.chunk_id)
            if call_id is not None and runner._host_denied(meta.get("permission_decision")):
                calls = [chunk for kind, _position, chunk in pools.call_events(call_id, event.key) if kind == "tool_call"]
                if len(calls) == 1:   # an id shared by several calls vouches for none of them
                    if calls[0] not in self.rejected and len(self.rejected) >= REJECTED_MAX:
                        self.overflow = "refused call table is full"
                        return
                    self.rejected[calls[0]] = [event.chunk_id, runner._denial_source(meta.get("permission_decision"))]
                    self.dirty = True

    def _cut(self, position, chunk_id):
        if len(self.cut) >= CUT_MAX:
            self.overflow = "cut result table is full"
            return
        self.cut.append((position, chunk_id))
        self.dirty = True

    def dump(self):
        return {"overflow": self.overflow, "targets": self.targets, "cut": [[list(p), c] for p, c in self.cut],
                "rejected": self.rejected}

    def load(self, data):
        self.overflow = data.get("overflow")
        self.targets = list(data.get("targets", ()))
        self._targets = set(self.targets)
        self.cut = [(tuple(p), c) for p, c in data.get("cut", ())]
        self.rejected = dict(data.get("rejected", {}))


class GrokFacts(Facts):
    """The calls a Grok hook blocked (``runner.blocked_calls`` for the whole history; the join is the same class).

    The host's note and the failed result of a call meet in whichever order and in whichever run they are fed: the
    class keeps what is still waiting for its other half, bounded, and this persists it as it is."""

    needed = True

    def __init__(self):
        self.blocks = hostblocks.HostBlocks()

    @property
    def rejected(self):
        return self.blocks.rejected     # chunk id of a blocked call -> [chunk id of the host's note, source]

    def feed(self, event, pools):
        self.blocks.feed(event)
        if len(self.blocks.rejected) > REJECTED_MAX:
            self.overflow = "blocked call table is full"

    def dump(self):
        return {"overflow": self.overflow, "blocks": self.blocks.dump()}

    def load(self, data):
        self.overflow = data.get("overflow")
        self.blocks.load(data["blocks"])


class AgyFacts(Facts):
    """RUNNING results still waiting for a final record of their step, and the proposals the Pre hook blocked."""

    needed = True

    def __init__(self, resolve=True):
        # ``resolve``: some rule reads host-logged proposals, so their text is loaded and denials can be matched.
        self.resolve = resolve
        self.unresolved = []  # (key, step_index), in order
        self.blocked = {}     # key -> (tool name, proposal text, key of the denial)
        self.unsure = []      # denials whose proposal was not found in the look-back window: (key, tool name)

    def feed(self, event, pools):
        meta = event.meta
        if meta.get("pending"):
            if len(self.unresolved) >= PENDING_MAX:
                self.overflow = "pending result table is full"
                return
            self.unresolved.append((event.key, meta.get("step_index")))
            self.dirty = True
        if meta.get("event_type") in RESULT_TYPES and meta.get("status") in runner._FINAL_STATUS:
            step = meta.get("step_index")
            before = len(self.unresolved)
            self.unresolved = [item for item in self.unresolved if item[1] != step]
            self.dirty = self.dirty or len(self.unresolved) != before
        denied = meta.get("denied_action")
        if self.resolve and isinstance(denied, dict):
            self._deny(event, denied, pools)

    def _deny(self, event, denied, pools):
        name, args = denied.get("name"), denied.get("args")
        for position, text in pools.proposals(name, event.key):
            if position in self.blocked:
                continue
            if runner._consistent(args, runner._proposal_args(text)):
                if len(self.blocked) >= BLOCKED_MAX:
                    self.overflow = "blocked proposal table is full"
                    return
                self.blocked[position] = (name, text, event.key)
                self.dirty = True
                return
        if not pools.proposals_complete(name):
            # No consistent proposal in the window, and an older one may exist. Only rules that have a matching
            # proposal of their own can be affected (see ``RuleMonitor._post_agy``).
            if len(self.unsure) >= BLOCKED_MAX:
                self.overflow = "unresolved denial table is full"
                return
            self.unsure.append((event.key, name))
            self.dirty = True

    def dump(self):
        return {"overflow": self.overflow, "unresolved": [[list(p), s] for p, s in self.unresolved],
                "blocked": [[list(p), n, t, list(k)] for p, (n, t, k) in self.blocked.items()],
                "unsure": [[list(p), n] for p, n in self.unsure]}

    def load(self, data):
        self.overflow = data.get("overflow")
        self.unresolved = [(tuple(p), s) for p, s in data.get("unresolved", ())]
        self.blocked = {tuple(p): (n, t, tuple(k)) for p, n, t, k in data.get("blocked", ())}
        self.unsure = [(tuple(p), n) for p, n in data.get("unsure", ())]


def make_facts(provider, needs=(), rules=()):
    """The shared summaries ``rules`` need for ``provider``'s evidence classes (a plain ``Facts`` when none)."""
    if provider == "agy":
        return AgyFacts(resolve=any(scope == "attempt" for scope, _roles in needs))
    if provider == "claude" and any(runner._reads_results(r) or r.requires_success or runner._wants_rejections(r)
                                    for r in rules):
        return ClaudeFacts()
    if provider == "grok" and any(runner._wants_rejections(r) for r in rules):
        return GrokFacts()
    return Facts()


# --- rule monitors -------------------------------------------------------------------------------------------------

class RuleMonitor:
    """The summary of one rule over the events fed so far, and the finding the reference would give for them."""

    def __init__(self, rule, provider):
        self.rule, self.provider = rule, provider
        self.agy, self.claude = provider == "agy", provider == "claude"
        self.refusing = provider in runner.REFUSAL_PROVIDERS       # the host's own records can tell a refused call
        self.src_id, self.from_id = rule.source_event_id, rule.applies_from_event_id
        self.roles = frozenset(rule.roles)
        self.kind = _SCOPE_KIND.get(rule.scope)
        self.rs = bool(rule.requires_success)
        self.regex = self.trigger = self.when_re = None
        self.regex_invalid = self.when_invalid = False
        try:
            self.regex = _compile(rule.regex)
            self.trigger = _compile(rule.before_regex) if rule.before_regex else None
        except re.error:
            self.regex = self.trigger = None
            self.regex_invalid = True
        if rule.when is not None:
            try:
                self.when_re = _compile(rule.when.tool_call_regex)
            except re.error:
                self.when_invalid = True
        self.static_unknown = self.regex_invalid or self.when_invalid
        self.is_must_not = rule.type == "must_not"
        self.inert = self.static_unknown  # nothing to fold in: a fixed outcome, or a table that overflowed
        self.deps = runner._dependencies(rule) if self.agy else frozenset()
        self.wants_attempts = self.agy and runner._wants_attempts(rule) and self.regex is not None
        self.track_trigger_proposals = self.agy and rule.type == "must" and self.trigger is not None
        self.wants_rej = self.refusing and runner._wants_rejections(rule) and self.regex is not None
        # state
        self.src = False
        self.src_user = False  # the source event was seen and is a user message
        self.started = self.from_id is None
        self.from_key = None
        self.when_met = False
        self.hits, self.hit_ev, self.exc, self.exc_chars = 0, [], [], 0
        self.exc_ids = []     # chunk id of each ``exc`` text, to drop the refused calls' texts
        self.rej_hits, self.rej_over = [], False   # [chunk id, call id, byte] of the call hits; set when over the cap
        self.hit_n, self.last3, self.lh = 0, [], None
        self.trig_n, self.trig_worst, self.trig_reasons, self.trig_items, self.trig_last = 0, "clean", [], [], None
        self.cut = self.legacy = None
        self.att_n, self.att, self.att_chars, self.att_closed = 0, [], 0, False
        self.trg_n, self.trg = 0, []
        self.src_tu = False
        self.overflow = None
        self.dirty = False
        self.new = False  # built from nothing in this run
        self.timed_out = False  # a pattern timed out on some text: no verdict and no state from this monitor
        if not (self.agy or self.when_re or self.src_id or self.from_id) and not self.static_unknown:
            # No anchors, no `when`, no provider classes: the plain path does the same as ``feed`` with less work.
            self.feed = self._feed_forbidden if self.is_must_not else self._feed_required

    # --- persistence ---

    def dump(self):
        return {
            "src": self.src, "src_user": self.src_user, "started": self.started, "from_key": self.from_key, "when": self.when_met,
            "hits": self.hits, "hit_ev": self.hit_ev, "exc": self.exc, "exc_chars": self.exc_chars,
            "exc_ids": self.exc_ids, "rej_hits": self.rej_hits, "rej_over": self.rej_over,
            "hit_n": self.hit_n, "last3": self.last3, "lh": self.lh,
            "trig_n": self.trig_n, "trig_worst": self.trig_worst, "trig_reasons": self.trig_reasons,
            "trig_items": self.trig_items, "trig_last": self.trig_last,
            "cut": self.cut, "legacy": self.legacy, "att_n": self.att_n, "att": self.att, "att_chars": self.att_chars,
            "att_closed": self.att_closed,
            "trg_n": self.trg_n, "trg": self.trg, "src_tu": self.src_tu, "overflow": self.overflow,
        }

    def load(self, data):
        self.src, self.started, self.when_met = data["src"], data["started"], data["when"]
        self.src_user = data["src_user"]
        self.from_key = _tuple(data["from_key"])
        self.hits, self.hit_ev, self.exc, self.exc_chars = data["hits"], data["hit_ev"], data["exc"], data["exc_chars"]
        self.exc_ids, self.rej_hits, self.rej_over = data["exc_ids"], data["rej_hits"], data["rej_over"]
        self.hit_n, self.last3, self.lh = data["hit_n"], data["last3"], data["lh"]
        self.trig_n, self.trig_worst, self.trig_reasons = data["trig_n"], data["trig_worst"], data["trig_reasons"]
        self.trig_items, self.trig_last = data["trig_items"], _tuple(data["trig_last"])
        cut = data["cut"]
        self.cut = None if cut is None else (tuple(cut[0]), cut[1], cut[2])
        self.legacy = _tuple(data["legacy"])
        self.att_n, self.att_chars, self.att_closed = data["att_n"], data["att_chars"], data["att_closed"]
        self.att = [{"k": tuple(item["k"]), "e": item["e"], "t": item["t"]} for item in data["att"]]
        self.trg_n, self.trg = data["trg_n"], [tuple(key) for key in data["trg"]]
        self.src_tu, self.overflow = data["src_tu"], data["overflow"]
        self.inert = self.static_unknown or bool(self.overflow)

    # --- feeding ---

    def eligible(self, event):
        return ((self.kind is None or event.kind == self.kind)
                and event.meta.get("role") in self.roles
                and runner._tool_matches(event, getattr(self.rule, "tools", None)))

    def _feed_forbidden(self, event, pools):
        if not self.inert and self.eligible(event) and self.regex.search(event.text):
            self._forbidden(event)

    def _feed_required(self, event, pools):
        if self.inert:
            return
        matched = False
        if self.eligible(event):
            text = event.text
            if self.trigger is not None and self.trigger.search(text):
                self._checkpoint(event)
            matched = self.regex.search(text) is not None
        if self.lh is not None:
            self._follow(event)
        if matched:
            self._hit(event, pools)

    def feed(self, event, pools):
        if event.chunk_id == self.src_id:
            self.src = True
            self.src_user = runner.is_user_event(event)
            if self.agy:
                self.src_tu = _truncated_user(event.meta)
        if not self.started:
            if event.chunk_id == self.from_id:
                self.started, self.from_key = True, event.key
            return
        if self.inert:
            return
        text = event.text
        if self.when_re is not None and not self.when_met:
            if (event.kind == "tool_call" or text.startswith(PROPOSED_CALL_PREFIX)) and self.when_re.search(text):
                self.when_met = True
        if self.agy:
            self._track_agy(event)
        eligible = self.eligible(event)
        if self.is_must_not:
            if eligible and self.regex.search(text):
                self._forbidden(event)
            return
        matched = False
        if eligible:
            if self.trigger is not None and self.trigger.search(text):
                self._checkpoint(event)
            matched = self.regex.search(text) is not None
        if self.lh is not None:
            self._follow(event)
        if matched:
            self._hit(event, pools)

    def _forbidden(self, event):
        self.hits += 1
        if len(self.hit_ev) < (REJ_EVIDENCE if self.wants_rej else EVIDENCE_LIMIT):
            self.hit_ev.append(_ev(event, flag=self.agy))
        if self.wants_rej and event.kind == "tool_call" and event.meta.get("call_id") is not None:
            if len(self.rej_hits) < REJ_HITS_MAX:
                self.rej_hits.append([event.chunk_id, event.meta["call_id"], event.key[0] if event.key else None])
            else:
                self.rej_over = True
        if self.rule.exception_regex:
            if len(self.exc) >= EXCEPTION_MAX_HITS or self.exc_chars + len(event.text) > EXCEPTION_MAX_CHARS:
                self.overflow, self.inert = "exception hit store is full", True
                return
            self.exc.append(event.text)
            self.exc_ids.append(event.chunk_id)
            self.exc_chars += len(event.text)

    def _checkpoint(self, event):
        verdict, reason, supporting = self._presence()
        self.trig_n += 1
        self.trig_last = event.key
        self.trig_worst = runner._worse(self.trig_worst, verdict)
        if reason not in self.trig_reasons:
            self.trig_reasons.append(reason)
        if len(self.trig_items) < EVIDENCE_LIMIT:
            self.trig_items.append(_ev(event, "triggering action", self.agy))
            self.trig_items.extend(supporting)
            del self.trig_items[EVIDENCE_LIMIT:]

    def _hit(self, event, pools):
        self.hit_n += 1
        self.last3.append(_ev(event, flag=self.agy))
        del self.last3[:-3]
        if self.rs:
            call_id = event.meta.get("call_id")
            if call_id:
                # [call id, calls sharing it, results after it (capped), result status, success evidence, why the
                # recorded success would not be the required command's (``shellstatus``), or None]
                self.lh = [_k(call_id), pools.earlier_calls(call_id, self.from_key, event.key) + 1, 0, 0, None,
                           shellstatus.unverifiable(event.text, self.regex)]
            else:
                self.lh = None

    def _follow(self, event):
        """The latest hit's call: later calls and results with its id (a result counts once it follows the hit)."""
        call_id = event.meta.get("call_id")
        if call_id is None:
            return
        lh = self.lh
        if event.kind == "tool_call":
            if _k(call_id) == lh[0]:
                lh[1] += 1
        elif event.kind == "tool_result" and _k(call_id) == lh[0]:
            lh[2] = min(lh[2] + 1, 2)
            if lh[2] == 1:
                success = event.meta.get("success")
                lh[3] = 0 if success is None else 1 if success is True else 2
                lh[4] = _ev(event, "linked successful tool result", self.agy) if success is True else None

    def _track_agy(self, event):
        meta = event.meta
        if self.cut is None:
            fields = meta.get("truncated_fields")
            if fields:
                event_type = meta.get("event_type")
                for field in fields:
                    if (event_type, field) in self.deps and (event_type, field) in JUDGED_FIELDS:
                        self.cut = (event.key, event_type, field)
                        break
        if self.legacy is None and meta.get("linkage") == "legacy_unlinked":
            self.legacy = event.key
        if meta.get("evidence_origin") == "model_proposal" and runner._tool_matches(event, getattr(self.rule, "tools", None)):
            text = event.text
            if self.wants_attempts and self.regex.search(text):
                self.att_n += 1
                keep = self.rule.exception_regex is not None
                if not self.att_closed:
                    # The list is always a prefix of the matches: once an entry does not fit, none is added again.
                    if len(self.att) < ATTEMPT_LIST and (not keep or self.att_chars + len(text) <= EXCEPTION_MAX_CHARS):
                        self.att.append({"k": event.key, "e": _ev(event, "forbidden call attempt (model proposal)"),
                                         "t": text if keep else None})
                        self.att_chars += len(text) if keep else 0
                    else:
                        self.att_closed = True
            if self.track_trigger_proposals and self.trigger.search(text):
                self.trg_n += 1
                if len(self.trg) < ATTEMPT_LIST:
                    self.trg.append(event.key)

    # --- the finding ---

    def _make(self, verdict, reason, items=None, basis=None, applicability="applicable", stopped=None):
        rule = self.rule
        return Finding(contract_id=rule.id, contract_revision=rule.revision, severity=rule.severity,
                       verdict=verdict, reason=reason, evidence=[_evidence(i) for i in items or ()], basis=basis,
                       applicability=applicability, proposals_rejected=stopped or [])

    def _presence(self):
        """``_Presence.before`` for the end of the events fed so far."""
        if not self.hit_n:
            return "violation", "required evidence missing before checkpoint", []
        if not self.rs:
            return "clean", "required regex evidenced", [self.last3[-1]]
        lh = self.lh
        unresolved = True
        if lh is not None and lh[1] == 1 and lh[2] == 1 and lh[3] != 0:
            if lh[3] == 1:
                if lh[5]:
                    return "unknown", runner.success_unverifiable(lh[5]), [
                        self.last3[-1], {**lh[4], "d": "linked tool result (success not attributable)"}]
                return "clean", "required call has linked successful result", [self.last3[-1], lh[4]]
            unresolved = False
        if unresolved:
            return "unknown", "required call has no unambiguous completion status", list(self.last3)
        return "violation", "required calls completed without success", list(self.last3)

    def finding(self, proposed, facts):
        check_request()
        if self.overflow:
            raise NeedFull(self.overflow)
        rule = self.rule
        if getattr(rule, "mode", "enforced") == "inject_only":
            return self._make("clean", "not_applicable: contract mode is inject_only (reminder only)",
                              basis=f"{rule.id}.mode is inject_only", applicability="not_applicable")
        if proposed is not None and not self.eligible(proposed):
            return self._make("clean", f"not_applicable: contract scope ({rule.scope}) does not apply to proposed action",
                              basis=f"{rule.id} does not apply to this proposed action", applicability="not_applicable")
        if (self.src_id and not self.src) or (self.from_id and not self.started):
            return self._make("unknown", "contract source or activation event is not in this session")
        if self.src_id and not self.src_user:
            return self._make("unknown", runner.ANCHOR_NOT_USER)
        if self.regex_invalid:
            return self._make("unknown", "invalid contract regex")
        if rule.when is not None:
            if self.when_invalid:
                return self._make("unknown", "invalid `when` regex", basis=f"{rule.id}.when failed to compile")
            if not self.when_met:
                return self._make("clean", not_applicable_reason(rule),
                                  basis=f"{rule.id}.when did not match any observed tool call",
                                  applicability="not_applicable")
        item, tu = self._typed(proposed, facts)
        history = proposed is None
        if self.agy:
            self._post_agy(item, tu, history, facts)
        elif self.claude:
            self._post_claude(item, history, facts)
        return item

    def _refused(self, facts):
        """This rule's matching calls that the host refused (empty unless it recorded some): ``[chunk, call, byte]``."""
        if not (self.wants_rej and self.hits and facts.rejected):
            return []
        if facts.overflow:
            raise NeedFull(facts.overflow)
        if self.rej_over:
            raise NeedFull("matching call table is full")
        return [hit for hit in self.rej_hits if hit[0] in facts.rejected]

    def _typed(self, proposed, facts):
        rule = self.rule
        stopped, texts = [], None
        if self.is_must_not:
            if proposed is not None:
                hit = self.eligible(proposed) and self.regex.search(proposed.text) is not None
                count, evidence, joined = (1, [_ev(proposed, flag=self.agy)], proposed.text) if hit else (0, [], "")
            else:
                if self.hits and rule.exception_regex and (self.hits > len(self.exc)):
                    raise NeedFull("exception hit store is incomplete")
                count, texts = self.hits, self.exc
                refused = self._refused(facts)
                evidence = self.hit_ev
                if refused:
                    gone = {hit[0] for hit in refused}
                    count -= len(refused)
                    evidence = [item for item in evidence if item["i"] not in gone]
                    if count and len(evidence) < min(EVIDENCE_LIMIT, count):
                        raise NeedFull("listed matches do not cover the refused calls")
                    texts = [t for t, chunk in zip(self.exc, self.exc_ids) if chunk not in gone]
                    stopped = [runner.rejection_entry(rule.id, chunk, call_id, facts.rejected[chunk], byte)
                               for chunk, call_id, byte in refused[:runner.REJECTED_LIMIT]]
                evidence, joined = evidence[:EVIDENCE_LIMIT], " ".join(texts)
            if not count:
                if stopped:
                    return self._make("clean", runner.rejected_clean(len(refused)),
                                      basis=f"{rule.id}.{contract_paths.verifier_field(rule)} matched only host-rejected proposal(s), which did not run",
                                      stopped=stopped), {}
                return self._make("clean", "no forbidden match in inspected events",
                                  basis=f"{rule.id}.{contract_paths.verifier_field(rule)} did not match any inspected event"), {}
            exception = None
            if rule.exception_regex:
                try:
                    exception = _compile(rule.exception_regex)
                except re.error:
                    return self._make("unknown", "invalid exception_regex",
                                      basis=f"{rule.id}.exception_regex failed to compile"), {}
            if exception is not None and exception.search(joined):
                return self._make(
                    "suspicious",
                    "forbidden match is covered by the authorized exception; confirm scope before proceeding",
                    basis=f"{rule.id}.exception_regex matched the same event(s); "
                          "user-authorized exception requires confirmation", stopped=stopped), {}
            reason = f"{contract_paths.forbidden_label(rule)} matched {count} event(s)"
            if rule.source_quote:
                reason += f"; user said: {mask_secrets(rule.source_quote)[:120]}"
            return self._make("violation", reason, basis=contract_paths.verifier_basis(rule, "matched prohibited content"),
                              items=evidence, stopped=stopped), {i["i"]: i.get("tu") for i in evidence}
        if self.trigger is not None:
            if proposed is not None:
                if not (self.eligible(proposed) and self.trigger.search(proposed.text)):
                    return self._make("clean", "no triggering action in inspected events"), {}
                verdict, reason, supporting = self._presence()
                items = [_ev(proposed, "triggering action", self.agy), *supporting][:EVIDENCE_LIMIT]
                return self._make(verdict, reason, items), {i["i"]: i.get("tu") for i in items}
            if not self.trig_n:
                return self._make("clean", "no triggering action in inspected events"), {}
            items = self.trig_items[:EVIDENCE_LIMIT]
            return (self._make(self.trig_worst, "; ".join(self.trig_reasons), items),
                    {i["i"]: i.get("tu") for i in items})
        verdict, reason, supporting = self._presence()
        return self._make(verdict, reason, supporting), {i["i"]: i.get("tu") for i in supporting}

    # --- provider evidence classes ---

    def _window_end(self, history):
        """Exclusive end of the rule's window: its last triggering action when judging history."""
        if self.rule.type == "must" and self.rule.before_regex and history:
            return self.trig_last
        return None

    def _post_claude(self, item, history, facts):
        rule = self.rule
        reads = runner._reads_results(rule)
        if not (reads or rule.requires_success):
            return
        if facts.overflow:
            raise NeedFull(facts.overflow)
        end = self._window_end(history)
        cut = [chunk_id for position, chunk_id in facts.cut
               if (self.from_key is None or position > self.from_key) and (end is None or position < end)]
        if not cut:
            return
        if rule.type == "must" and item.verdict == "clean" and rule.requires_success:
            used = {e.chunk_id for e in item.evidence}
            if any(chunk_id in used for chunk_id in cut):
                runner._unknown(item, "truncated_result", "a cut tool result cannot establish success")
        elif reads:
            runner._cut_unknown(rule, item, "truncated_result",
                                "a cut tool result may hide the match" if rule.type == "must_not"
                                else "a cut tool result may hold the missing evidence", history=history)

    def _post_agy(self, item, tu, history, facts):
        rule = self.rule
        if facts.overflow:
            raise NeedFull(facts.overflow)
        end = self._window_end(history)
        cut = self.cut if (self.cut is not None and (end is None or self.cut[0] < end)) else None
        blocked = facts.blocked if history else {}
        if history and facts.unsure and (self.att_n or self.trg_n):
            raise NeedFull("a denial's proposal may lie before the look-back window")
        if rule.type == "must_not":
            if history and item.verdict != "unknown" and self.wants_attempts:
                self._attempt_hits(item, blocked)
            if cut:
                runner._cut_unknown(rule, item, runner._TRUNCATED_CODE.get(cut[2], "truncated_" + cut[2]),
                                    f"a still-truncated {cut[1]}.{cut[2]} may hide a match", history=history)
            if item.verdict == "suspicious" and rule.exception_regex and self.src and self.src_tu:
                item.verdict = "violation"
                item.reason = ("truncated_content: the exception comes from a truncated user message and "
                               "cannot waive the match; " + item.reason)
            return
        if item.verdict == "clean" and item.reason.startswith("no triggering action"):
            if rule.before_regex and history:
                code = self._hidden_trigger(cut, blocked, facts)
                if code:
                    runner._unknown(item, *code)
            return
        if item.verdict == "clean":
            flags = [tu.get(e.chunk_id) for e in item.evidence if e.detail != "triggering action"]
            if flags and all(flags):
                runner._unknown(item, "truncated_content", "the only satisfying evidence is a truncated user message")
            return
        absent = runner._absent(item)
        if not (absent or runner._unresolved(item)):
            return
        if cut:
            field = cut[2]
            runner._cut_unknown(rule, item, runner._TRUNCATED_CODE.get(field, "truncated_" + field),
                                f"a still-truncated {cut[1]}.{field} may hold the missing evidence")
        elif absent and runner._wants_attempts(rule) and self._attempted(history, end, blocked):
            if any(self._after_start(key) and (end is None or key < end) for key, _step in facts.unresolved):
                runner._unknown(item, "pending_running", "a tool result is still RUNNING, so its outcome is unknown")
            elif self.legacy is not None and (end is None or self.legacy < end):
                runner._unknown(item, "unsupported_legacy",
                                "results before this conversation's first host receipt cannot be linked to calls")

    def _after_start(self, key):
        return self.from_key is None or key > self.from_key

    def _attempted(self, history, end, blocked):
        if not history:
            return self.att_n > 0  # a proposed action sees every host-logged proposal, blocked or not
        for entry in self.att:
            if entry["k"] not in blocked:
                return end is None or entry["k"] < end
        if not self.att_closed:
            return False
        raise NeedFull("every listed attempt is blocked and more exist")

    def _attempt_hits(self, item, blocked):
        rule = self.rule
        try:
            exception = _compile(rule.exception_regex) if rule.exception_regex else None
        except re.error:
            return
        refused = sorted((key, name, denied) for key, (name, text, denied) in blocked.items()
                         if self._after_start(key) and self.regex.search(text))
        if refused:   # reported apart, whatever the verdict ends up being
            item.proposals_rejected = [runner.rejected_proposal_item(rule.id, name, key[0], denied[0])
                                       for key, name, denied in refused[:runner.REJECTED_LIMIT]]
        gone = len(refused)
        count = self.att_n - gone
        if count <= 0:
            return
        listed = [entry for entry in self.att if entry["k"] not in blocked]
        if exception is not None and self.att_closed:
            raise NeedFull("attempt text store is incomplete")
        if len(listed) < min(EVIDENCE_LIMIT, count):
            raise NeedFull("listed attempts do not cover the blocked ones")
        covered = exception is not None and exception.search(" ".join(entry["t"] for entry in listed)) is not None
        verdict = "suspicious" if covered else "violation"
        if runner._worse(item.verdict, verdict) == item.verdict:
            return
        reason = f"{contract_paths.forbidden_label(rule)} matched {count} host-logged call attempt(s); execution is not proven"
        if rule.source_quote:
            reason += f"; user said: {mask_secrets(rule.source_quote)[:120]}"
        item.verdict, item.reason = verdict, reason
        item.basis = contract_paths.verifier_basis(rule, "matched a proposed call; the attempt itself is the violation")
        item.evidence = [_evidence(entry["e"]) for entry in listed[:EVIDENCE_LIMIT]]

    def _hidden_trigger(self, cut, blocked, facts):
        if cut:
            field = cut[2]
            return (runner._TRUNCATED_CODE.get(field, "truncated_" + field),
                    f"a still-truncated {cut[1]}.{field} may hide the triggering call")
        if self.legacy is None:
            return None
        for key in self.trg:
            if key not in blocked:
                return ("unsupported_legacy",
                        "a matching call was proposed but its execution cannot be linked to a legacy result")
        if self.trg_n > len(self.trg):
            raise NeedFull("every listed triggering proposal is blocked and more exist")
        return None




# --- keys ------------------------------------------------------------------------------------------------------------

def needs_digest(needs, link_calls, provider) -> str:
    """Which events had text loaded: a monitor's result depends on it only through this."""
    items = sorted({(scope, tuple(sorted(roles))) for scope, roles in needs})
    return hashlib.sha256(json.dumps([items, bool(link_calls), provider]).encode()).hexdigest()[:24]


def rule_fingerprint(rule, provider, parser_version, digest) -> str:
    parts = [EVALUATOR_VERSION, STATE_VERSION, provider, parser_version, mask_fingerprint(), digest,
             rule.model_dump(mode="json")]
    if rule.path_guard is not None:
        # The stored form of a typed contract is its guard, but the monitor runs the regex derived from it under the
        # tool map of this version: a changed map is a changed monitor, so the derived regex is part of the key.
        parts.append(rule.regex)
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def facts_fingerprint(provider, parser_version, digest) -> str:
    payload = json.dumps([EVALUATOR_VERSION, STATE_VERSION, provider, parser_version, digest, "facts"])
    return hashlib.sha256(payload.encode()).hexdigest()


def wants_call_index(active, provider) -> bool:
    """Whether a run may look stored events up by call id (a requires_success tracker, Claude cut results)."""
    return any(r.regex and (r.requires_success or (provider == "claude" and (
        runner._reads_results(r) or runner._wants_rejections(r)))) for r in active)


class PlanRetry(Exception):
    """A delta run met an input its summaries cannot hold; run again with these rules on the full evaluator."""

    def __init__(self, ids, reason, plan=None, health=None, drop=False):
        super().__init__(reason)
        # ``health``: a status to record for these rules ("error"), so a defect is not hit again on every audit.
        # ``drop``: delete these rules' stored states (a pattern timed out: nothing of that run may be reused).
        self.ids, self.reason, self.plan, self.health, self.drop = frozenset(ids), reason, plan, health, drop


def _shadow_affordable(events) -> bool:
    control = current_request()
    if control is None or control.deadline is None:
        return True
    remaining = control.deadline - time.monotonic()
    return remaining - comparison_reserve(remaining) > events * SHADOW_SECONDS_PER_EVENT * 1.5 + 0.5


# A comparison is decided twice, and both decisions can only SKIP more than ``_shadow_affordable`` alone: once before the
# events of the reference are read (``prepare``, from what earlier requests of this process measured) and once after the
# incremental pass, when the speed of that very pass is known (``comparison_shortfall``, from the engine). What is skipped
# is not lost: a resident engine's idle comparison verifies the state (``shadow_worker``), and the policy that decides which
# requests are compared (rates, warm-up) does not change. ``COMPARISON_MARGIN`` is the safety factor on the estimate, and a
# comparison starts only if the estimate fits in ``COMPARISON_ROOM_SHARE`` of the time that is left for it: a verification
# that runs out of that time is cut and was a waste, and the answer it kept waiting was ready long before (measured:
# a comparison started in 27 s of room on a 300,000-event session of heavy contracts took all of it and was cut).
COMPARISON_MARGIN = 1.25
COMPARISON_ROOM_SHARE = 0.5


def comparison_shortfall(events, rate, *, delta=0, defer=False, scale=1.0):
    """Why a comparison does not fit the time left, or ``None`` when it does (or when nothing is known about its cost).

    ``events``: stored events the reference evaluates; ``rate``: seconds one event costs to fold into the monitors (None:
    unknown, no opinion); ``delta``: events the incremental pass still has to fold before the comparison starts (``prepare``
    asks before that pass, the engine after it: 0); ``defer``: a resident engine can verify the state later, which the
    answer then says (``"deferred"``); ``scale``: the share of the contracts the comparison covers (a scoped comparison)."""
    control = current_request()
    if rate is None or control is None or control.deadline is None:
        return None
    left = control.deadline - time.monotonic()
    room = left - comparison_reserve(left) - delta * rate * COMPARISON_MARGIN       # what is left once the fold has been made
    cost = events * rate * REFERENCE_PER_EVAL * scale * COMPARISON_MARGIN
    if cost > room * COMPARISON_ROOM_SHARE:
        return "deferred" if defer else "deadline"
    return None


def _shadow_sample(session_id, offset, active, every) -> bool:
    """The fixed legacy draw (``SELF_DIRECT_AUDIT_SHADOW_EVERY`` set to an integer): about one audit in ``every``."""
    if every <= 0:
        return False
    if every == 1:
        return True
    token = f"{session_id}|{offset}|" + "|".join(f"{r.id}:{r.revision}" for r in active)
    return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % every == 0


def fold(events, monitors, facts, pools, progress=None):
    """Feed ``events`` to every consumer that has not seen them (``monitors``: ``(monitor, offset)`` pairs; ``facts``:
    ``(facts, offset)`` or ``None``); a consumer skips events whose record starts before its offset. ``progress`` (a
    one-item list) receives how many events were folded, at every cooperative check, so a fold that is stopped can say how
    far it got."""
    shared, shared_offset = facts if facts is not None else (None, 0)
    shared_on = shared is not None and not shared.overflow
    feeds = [(monitor.feed, offset, monitor) for monitor, offset in monitors]
    note = pools.note if pools.track_calls or pools.track_proposals else None
    for number, event in enumerate(events):
        if not number & 511:
            check_request()
            if progress is not None:
                progress[0] = number
        start = event.key[0]
        for feed, offset, monitor in feeds:
            if start >= offset:
                try:
                    feed(event, pools)
                except TIMEOUTS:
                    # Marked here, handled in ``AuditPlan.evaluate``; the monitor takes no more events.
                    monitor.timed_out = monitor.inert = True
        if shared_on and start >= shared_offset:
            shared.feed(event, pools)
            shared_on = not shared.overflow
        if note is not None:
            note(event)
    if events:
        last = events[-1].key[0]
        for monitor, offset in monitors:
            monitor.dirty = monitor.dirty or last >= offset
        if shared is not None and last >= shared_offset:
            shared.dirty = True


class AuditPlan:
    """One audit's incremental evaluation: which events were read, which monitors consume them, what to persist.

    Built by ``prepare`` inside the engine's read boundary (database work happens only there); ``load`` and
    ``evaluate`` run as the audit graph's stages with no database access.
    """

    def __init__(self, *, session_id, provider, mode, events, count, history_ids, monitors, offsets, facts,
                 facts_offset, facts_new, pools, routes, rebuilds, support, snapshot, shadow, digest,
                 parser_version, report, reference, comparison=None, schedule=None, schedule_loaded=None,
                 risky=frozenset(), prior=None):
        self.session_id, self.provider, self.mode = session_id, provider, mode
        self.events, self.count, self.reference = events, count, reference
        self.history_ids, self.monitors, self.offsets = history_ids, monitors, offsets
        self.facts, self.facts_offset, self.facts_new = facts, facts_offset, facts_new
        self.pools, self.routes, self.rebuilds = pools, routes, rebuilds
        self.support, self.snapshot, self.shadow = support, snapshot, shadow
        self.digest, self.parser_version, self.report = digest, parser_version, report
        # ``comparison``: what this request is compared on (``audit.shadow.Comparison``); ``schedule``: the
        # ``ShadowState`` it was planned from (``None`` when the setting compares every audit and keeps no state);
        # ``schedule_loaded``: the state as stored, to tell whether it has to be written back.
        self.comparison = comparison if comparison is not None else shadow_module.Comparison()
        self.schedule, self.schedule_loaded = schedule, schedule_loaded
        self.risky, self.prior = risky, prior or {}    # high-risk monitored contracts; their stored summaries
        self.fed = False
        self.overflowed, self.errors, self.timeouts = {}, {}, set()
        # What this pass measured about itself (see ``eval_cost``): seconds to read the events the fold consumes
        # (``load_seconds``, set by ``prepare``), seconds to fold them and how many were folded (``load``).
        self.cost_key, self.load_seconds, self.fold_seconds, self.fold_events = None, 0.0, 0.0, 0

    # --- preparation (inside the read boundary) ---

    @classmethod
    def prepare(cls, store, *, session_id, provider, active, planned, needs, link_calls, shadow_every,
                force_full=frozenset(), report=None, premise_rules=None, force=None, idle=False, cost_key=None,
                defer=False):
        """Decide how to evaluate and read the events needed; ``None`` means evaluate everything the old way.

        ``shadow_every``: ``None`` for the adaptive schedule, else the fixed legacy rate (``audit.shadow``).
        ``premise_rules``: the contracts the verification premise is taken over (default ``active``; the engine passes the
        enabled contracts of the session before the proposed-action filter, so an audit and an action check share one
        schedule). ``force``: comparisons a previous pass of this request found necessary (``shadow.Comparison``).
        ``idle``: the idle comparison of a resident engine, which compares everything. ``cost_key``: the key of this
        contract set in ``eval_cost.COSTS`` (what a comparison is expected to cost, once this process has measured the set);
        ``defer``: a resident engine can verify a skipped comparison later, and the report then says ``deferred``."""
        report = {} if report is None else report
        history = [r for r in active if getattr(r, "mode", "enforced") != "inject_only" and r.regex and (r.type == "must" or planned is None)]
        history_ids = {r.id for r in history}
        lineage = store.audit_lineage(session_id, provider)
        cursor, generation, latest = lineage["cursor"], lineage["generation"], lineage["latest"]
        if cursor is None or not history:
            return None
        parser_version = cursor["parser_version"]  # the parser that produced the stored events
        chain_ok = (generation is not None and latest is not None
                    and latest["byte_offset"] == cursor["byte_offset"] and latest["prefix_hash"] == cursor["prefix_hash"]
                    and latest["synced_at"] == cursor["synced_at"])
        if chain_ok and store.chunk_count(session_id, provider) != latest["event_count"]:
            chain_ok = False  # rows came or went outside a commit: the log no longer describes the stored events
        digest = needs_digest(needs, link_calls, provider)
        states = store.load_audit_states(session_id, provider, digest)
        health = store.audit_health(session_id, provider, EVALUATOR_VERSION)

        def reject(row, fingerprint, revision=None):
            if row is None:
                return "no_state"
            if row["evaluator_version"] != EVALUATOR_VERSION:
                return "evaluator_version"
            if row["parser_version"] != parser_version:
                return "parser_version"
            if revision is not None and row["contract_revision"] != revision:
                return "contract_revision"
            if row["fingerprint"] != fingerprint:
                return "contract_changed"
            if generation is None or row["generation"] != generation:
                return "generation"
            if row["state"].get("overflow"):
                return "overflow"
            if row["path"] != cursor["path"]:
                return "path"
            if not chain_ok:
                return "lineage_unknown"
            if row["byte_offset"] > cursor["byte_offset"]:
                return "ahead_of_index"
            logged = store.lineage_row(session_id, provider, generation, row["byte_offset"])
            if (logged is None or logged["prefix_hash"] != row["prefix_hash"]
                    or logged["event_count"] != row["event_count"]):
                return "lineage"
            return None

        routes, rebuilds, deltas = {}, {}, {}
        counts = {}
        for rule in history:
            counts[rule.id] = counts.get(rule.id, 0) + 1
        for rule in history:
            if rule.id in health:
                routes[rule.id] = "health:" + health[rule.id][0]
            elif rule.id in force_full:
                routes[rule.id] = "fallback"
            elif counts[rule.id] > 1 or rule.id.startswith("#"):
                routes[rule.id] = "contract id is not unique or reserved"
            else:
                row = states.get(rule.id)
                why = reject(row, rule_fingerprint(rule, provider, parser_version, digest), rule.revision)
                if why is None:
                    try:
                        RuleMonitor(rule, provider).load(row["state"])
                    except (KeyError, TypeError, ValueError, IndexError):
                        why = "corrupt_state"  # an unreadable cache is no cache
                if why is None:
                    deltas[rule.id] = row
                elif why == "overflow":
                    routes[rule.id] = "overflow: " + str(row["state"]["overflow"])
                else:
                    rebuilds[rule.id] = why
        facts = make_facts(provider, needs, history)
        facts_row, facts_why = None, None
        if facts.needed:
            row = states.get("#facts")
            facts_why = reject(row, facts_fingerprint(provider, parser_version, digest))
            if facts_why is None:
                try:
                    facts.load(row["state"])
                except (KeyError, TypeError, ValueError, IndexError):
                    facts, facts_why = make_facts(provider, needs, history), "corrupt_state"
                else:
                    facts_row = row
            elif facts_why == "overflow":
                for rule in history:
                    if rule.id not in routes:
                        routes[rule.id] = "facts overflow: " + str(row["state"]["overflow"])
                        deltas.pop(rule.id, None)
                        rebuilds.pop(rule.id, None)
        if not deltas and not rebuilds:
            report.update(mode="full", fallback=dict(routes))
            return None
        facts_rebuild = facts.needed and facts_why not in (None, "overflow")
        offsets = {rule_id: row["byte_offset"] for rule_id, row in deltas.items()}
        facts_offset = facts_row["byte_offset"] if facts_row is not None else 0
        offset = cursor["byte_offset"]
        # --- which comparison this request gets (audit/shadow.py) ---
        comparison = shadow_module.Comparison()
        schedule = schedule_loaded = None
        if shadow_every != 1:           # comparing every audit needs no schedule
            stored = store.load_shadow_state(session_id, provider)
            schedule_loaded = stored.dump() if stored is not None else None
            premise = shadow_module.make_premise(premise_rules if premise_rules is not None else active, parser_version,
                                                 EVALUATOR_VERSION, generation if chain_ok else None)
            schedule, reset = shadow_module.reconcile(stored, premise)
            if reset is not None:
                comparison.add("premise")
                report["shadow_reset"] = reset
            if rebuilds or facts_rebuild:
                schedule = shadow_module.arm_warmup(schedule)
            if schedule.warmup > 0:
                comparison.add("warmup")
        if rebuilds or facts_rebuild:
            comparison.add("rebuild")
        if idle:
            comparison.add("idle")
        if shadow_every is None:
            drawn = shadow_module.sampled(schedule, session_id, provider, offset, None)
        else:
            drawn = _shadow_sample(session_id, offset, active, shadow_every)
        if drawn:
            comparison.add("sampled")
        risky = frozenset(r.id for r in history if r.risk == "high" and r.id not in routes)
        if planned is not None and risky and not comparison.full:
            applies = {r.id for r in history if r.id in risky and runner.action_applies(r, planned)}
            if applies:
                comparison.add("high_risk", applies)
        if force is not None:
            for reason in force.reasons:
                comparison.add(reason, force.scope)
        shadow = comparison.active
        total = store.session_metrics(session_id, provider)[0]
        if shadow and not _shadow_affordable(total):
            shadow = False
            report["shadow_skipped"] = "deferred" if defer else "deadline"
        elif shadow and not idle and force is None and COSTS.observed(cost_key):
            # This process has measured what folding one event costs for this contract set: a comparison that cannot
            # follow the incremental pass in the time left is skipped now, before its events are read. (A request
            # that has nothing measured yet is judged by the estimate above, and again by the engine after the pass. A
            # comparison that an answer called for, ``force``, is the business of the pass that asks for it.)
            building = bool(routes) or bool(rebuilds) or bool(facts_rebuild)
            behind = total if building else max(0, total - min(
                (row["event_count"] for row in [*deltas.values(), *([facts_row] if facts_row is not None else [])]),
                default=total))
            why = comparison_shortfall(total, COSTS.eval_cost(cost_key, 1), delta=behind, defer=defer)
            if why is not None:
                shadow = False
                report["shadow_skipped"] = why

        loading = {"reference": 0.0, "tail": 0.0}      # seconds spent reading events, for the cost model

        def timed(kind, call, *args, **kwargs):
            started = time.perf_counter()
            try:
                return call(*args, **kwargs)
            finally:
                loading[kind] += time.perf_counter() - started

        def tail_mismatch(events, mode):
            """Consumers whose stored prefix no longer holds the number of events it was built from."""
            bad = {}
            checked = {name: row["byte_offset"] for name, row in deltas.items()}
            if facts_row is not None:
                checked["#facts"] = facts_row["byte_offset"]
            rows = {**deltas, "#facts": facts_row}
            starts = [event.key[0] for event in events]  # in order, so a count after an offset is a bisection
            for name, offset in checked.items():
                tail = len(starts) - bisect_left(starts, offset)
                expected = rows[name]["event_count"]
                ok = (tail == latest["event_count"] - expected) if mode == "delta" else (len(events) - tail == expected)
                if not ok:
                    bad[name] = "event_count"
            return bad

        def drop(bad):
            nonlocal facts_row, facts_rebuild
            for name, why in bad.items():
                if name == "#facts":
                    facts_rebuild, facts_row = True, None
                else:
                    rebuilds[name] = why
                    deltas.pop(name)
                    offsets.pop(name)

        full = bool(routes) or bool(rebuilds) or bool(facts_rebuild)
        reference = None
        partial = False          # the reference holds text only for the compared contracts (a scoped comparison)
        if shadow or full:
            if shadow and not comparison.full and not full:
                scoped = [r for r in history if r.id in comparison.scope]
                scoped_needs, scoped_links = runner.audit_needs(scoped, planned)
                snapshot_events = timed("reference", store.capture_audit_events, session_id, provider, scoped_needs,
                                        link_calls=scoped_links)
                partial = True
            else:
                snapshot_events = timed("reference", store.capture_audit_events, session_id, provider, needs,
                                        link_calls=link_calls)
            reference = list(snapshot_events.events)
            if not full:
                bad = tail_mismatch(reference, "full")
                if bad:
                    drop(bad)
                    full = True
                    if "shadow_skipped" not in report:
                        comparison.add("rebuild")
                        shadow = True
                    if partial:       # the monitors about to be rebuilt need the text of every rule
                        snapshot_events = timed("reference", store.capture_audit_events, session_id, provider, needs,
                                                link_calls=link_calls)
                        reference, partial = list(snapshot_events.events), False
        events, mode, count = [], "delta", latest["event_count"] if chain_ok else 0
        if not full:
            from_byte = min([*offsets.values(), *([facts_row["byte_offset"]] if facts_row is not None else [])])
            events = timed("tail", store.capture_audit_tail, session_id, provider, needs, link_calls=link_calls,
                           from_byte=from_byte)
            bad = tail_mismatch(events, "delta")
            if bad:
                drop(bad)
                full = True
                if "shadow_skipped" not in report:
                    comparison.add("rebuild")
                    shadow = True
                if reference is None or partial:
                    snapshot_events = timed("reference", store.capture_audit_events, session_id, provider, needs,
                                            link_calls=link_calls)
                    reference, partial = list(snapshot_events.events), False
        if full:
            events, mode = reference, "full"
            count = latest["event_count"] if chain_ok else snapshot_events.count
            bad = tail_mismatch(events, "full")
            if bad:
                drop(bad)
                if "shadow_skipped" not in report:
                    comparison.add("rebuild")
                    shadow = True
        monitors = {}
        for rule in history:
            if rule.id in routes:
                continue
            monitor = RuleMonitor(rule, provider)
            if rule.id in deltas:
                monitor.load(deltas[rule.id]["state"])
            monitor.new = rule.id not in deltas
            monitors[rule.id] = monitor
            offsets.setdefault(rule.id, 0)
        if facts_row is None:
            facts = make_facts(provider, needs, history)
            facts_offset = 0
        pools = Pools(track_calls=isinstance(facts, ClaudeFacts) or any(m.rs for m in monitors.values()),
                      track_proposals=bool(getattr(facts, "resolve", False)))
        if mode == "delta":
            from_byte = min([*offsets.values(), *([facts_offset] if facts_row is not None else [])])
            if pools.track_calls:
                wanted = set()
                follow = any(m.rs for m in monitors.values())
                for event in events:
                    call_id = event.meta.get("call_id")
                    if follow and event.kind == "tool_call" and call_id:
                        wanted.add(call_id)
                    target = event.meta.get("truncated_result_for")
                    if isinstance(facts, ClaudeFacts) and event.kind == "meta" and isinstance(target, str) and target:
                        wanted.add(target)
                    if (isinstance(facts, ClaudeFacts) and event.kind == "tool_result" and call_id is not None
                            and runner._host_denied(event.meta.get("permission_decision"))):
                        wanted.add(call_id)   # the call a refusal belongs to may be older than this delta
                pools.calls_prior = {_k(key): value for key, value in
                                     store.audit_call_events(session_id, provider, wanted, from_byte).items()}
            if pools.track_proposals:
                names = {event.meta["denied_action"].get("name") for event in events
                         if isinstance(event.meta.get("denied_action"), dict)}
                for name in names:
                    pools.proposals_prior[name] = store.audit_proposals_before(session_id, provider, name, from_byte)
        free = [r for r in active if r.id not in history_ids]
        support = None
        if mode == "delta" and free:
            # Rules that need no history see only their anchor events (as in the non-history audit path).
            anchors = {a for r in free for a in (r.source_event_id, r.applies_from_event_id) if a}
            support = store.anchor_chunks(session_id, provider, anchors)
        snapshot = {"generation": generation if chain_ok else None, "offset": cursor["byte_offset"],
                    "prefix_hash": cursor["prefix_hash"], "synced_at": cursor["synced_at"], "path": cursor["path"],
                    "count": count}
        report.update(mode=mode, shadow=shadow, events_read=len(events), monitors=len(monitors),
                      rebuilt=dict(rebuilds), fallback=dict(routes))
        if shadow:
            report["shadow_reasons"] = list(comparison.reasons)
            if not comparison.full:
                report["shadow_scope"] = sorted(comparison.scope)
        if schedule is not None and shadow_every is None:
            report["shadow_rate"] = shadow_module.rate_text(schedule, None)
        if not shadow:
            comparison = shadow_module.Comparison()
        # The stored summaries of the high-risk monitors as text: a monitor shares its lists with the loaded state and
        # appends to them while it folds, so only a copy taken now can tell afterwards whether the summary changed.
        prior = {rule_id: json.dumps(deltas[rule_id]["state"], separators=(",", ":"))
                 for rule_id in risky if rule_id in deltas and rule_id in monitors}
        if facts_rebuild:
            report["rebuilt"]["#facts"] = facts_why or "event_count"
        plan = cls(session_id=session_id, provider=provider, mode=mode, events=events, count=count,
                   history_ids=history_ids, monitors=monitors, offsets=offsets, facts=facts,
                   facts_offset=facts_offset, facts_new=facts_row is None, pools=pools, routes=routes,
                   rebuilds=rebuilds, support=support, snapshot=snapshot, shadow=shadow, digest=digest,
                   parser_version=parser_version, report=report, reference=reference, comparison=comparison,
                   schedule=schedule, schedule_loaded=schedule_loaded, risky=risky & set(monitors), prior=prior)
        plan.cost_key = cost_key
        plan.load_seconds = loading["tail"] if mode == "delta" else loading["reference"]
        return plan

    # --- comparisons that only the finished answer can call for ---

    def state_key(self):
        """Which stored events, as the cursor identifies them (what a memoized comparison was made on)."""
        snapshot = self.snapshot
        return (snapshot["generation"], snapshot["offset"], snapshot["prefix_hash"], snapshot["count"])

    def summary_changed(self, rule_id) -> bool:
        """Whether this request changed the stored summary of a monitored contract (a new, rebuilt or folded state)."""
        monitor = self.monitors.get(rule_id)
        if monitor is None:
            return False
        if monitor.new or monitor.overflow:
            return True
        if not monitor.dirty:
            return False
        before = self.prior.get(rule_id)
        return before is None or json.dumps(monitor.dump(), separators=(",", ":")) != before

    def late_reasons(self, result, key, applies=None) -> dict:
        """``{reason: contract ids}`` the finished answer calls a comparison for, whatever the plan chose.

        * ``violation``: a monitored contract the answer judges violation or suspicious (a wrong block);
        * ``flip``: a monitored contract that the previous request of this kind judged so and the answer calls clean now
          (a wrongly missed violation; events only append, so a real flip is rare);
        * ``high_risk``: a monitored ``risk: high`` contract whose stored summary this request changed.
        Contracts the plan already compares are left out by the caller; contracts the fast path does not decide (a proposed
        action judged on its own, a rule routed to the full evaluator) never appear here. For an action check
        ``applies(contract_id)`` tells whether the proposed action is in the contract's scope: only those findings can
        block it, so only those are guarded."""
        verdicts = {item.contract_id: item.verdict for item in result.findings}

        def decided(contract_id):       # looked at only for contracts that could matter: a pattern search each
            return contract_id in self.monitors and (applies is None or applies(contract_id))
        out = {"violation": {c for c, v in verdicts.items() if v in shadow_module.FLAGGED and decided(c)}}
        schedule = self.schedule
        out["flip"] = ({c for c in shadow_module.flipped(schedule, key, verdicts) if decided(c)}
                       if schedule is not None else set())
        out["high_risk"] = {c for c in self.risky if c in self.monitors and self.summary_changed(c)}
        return out

    # --- the audit stages (no database) ---

    def load(self, active):
        """Fold the events read into every monitor; nothing else has to be walked."""
        if self.fed:
            return
        self.fed = True
        started, progress = time.perf_counter(), [0]
        try:
            # One pooled budget for the walk (a per-event, per-monitor budget would cost more than the walk). A monitor
            # that runs out is dropped and its rule is judged by the reference evaluator with its own budget.
            with contract_budget(_patterns.CONTRACT_BUDGET_SEC * max(1, len(self.monitors))):
                fold(self.events, [(monitor, self.offsets[monitor.rule.id]) for monitor in self.monitors.values()],
                     (self.facts, self.facts_offset) if self.facts.needed else None, self.pools, progress)
        except (RequestStopped, IndexBusy):
            self._measure(progress[0], started)    # a fold that was stopped still says how fast it was
            raise
        except Exception as exc:
            if self.mode != "full":
                raise PlanRetry(self.monitors, "monitor_error:" + type(exc).__name__, None, "error") from exc
            raise
        self._measure(len(self.events), started)

    def _measure(self, folded, started):
        """Tell the cost model how long ``folded`` events took to read and fold, when every consumer folded the same events
        (a mixed fold, one rebuilt monitor beside up-to-date ones, says nothing about the cost of a whole contract set)."""
        self.fold_seconds, self.fold_events = time.perf_counter() - started, folded
        offsets = {self.offsets[monitor.rule.id] for monitor in self.monitors.values()}
        if self.facts.needed:
            offsets.add(self.facts_offset)
        if len(offsets) == 1 and self.events and folded:
            COSTS.observe_eval(self.cost_key, folded, self.fold_seconds + self.load_seconds * folded / len(self.events))

    def evaluate(self, active, *, proposed, coverage):
        self.load(active)
        findings = [None] * len(active)
        reference, free = [], []
        for index, rule in enumerate(active):
            monitor = self.monitors.get(rule.id)
            if monitor is not None and monitor.timed_out:
                self.timeouts.add(rule.id)
                if self.mode != "full":
                    raise PlanRetry({rule.id}, "regex_timeout", None, None, drop=True)
                reference.append(index)
            elif monitor is not None:
                try:
                    findings[index] = monitor.finding(proposed, self.facts)
                    continue
                except (RequestStopped, IndexBusy):
                    raise
                except TIMEOUTS as exc:
                    monitor.timed_out = True
                    self.timeouts.add(rule.id)
                    if self.mode != "full":
                        raise PlanRetry({rule.id}, "regex_timeout", None, None, drop=True) from exc
                    reference.append(index)
                except NeedFull as exc:
                    monitor.overflow = monitor.overflow or exc.reason
                    if self.mode != "full":
                        raise PlanRetry({rule.id}, exc.reason, self) from exc
                    self.overflowed[rule.id] = exc.reason
                    reference.append(index)
                except Exception as exc:  # a defect in a monitor: answer from the reference and stop trusting it
                    self.errors[rule.id] = "monitor_error:" + type(exc).__name__
                    if self.mode != "full":
                        raise PlanRetry({rule.id}, self.errors[rule.id], None, "error") from exc
                    reference.append(index)
            elif rule.id in self.history_ids:
                reference.append(index)
            else:
                free.append(index)
        if reference and self.mode != "full":
            raise PlanRetry({active[i].id for i in reference}, "rule is routed to the full evaluator", self)
        for group, chunks in ((reference, self.events), (free, self.support if self.mode != "full" else self.events)):
            if not group:
                continue
            rules = [active[i] for i in group]
            rejected = runner.rejections(self.provider, rules, chunks, proposed)
            found = [runner.audit_contract(r, session_id=self.session_id, store=None, provider=self.provider,
                                           proposed=proposed, chunks=chunks, rejected=rejected) for r in rules]
            evidence_rules = runner._EVIDENCE_RULES.get(self.provider)
            if evidence_rules is not None:
                runner._post_process(evidence_rules, rules, found, chunks, proposed=proposed, coverage=None)
            for index, item in zip(group, found):
                findings[index] = item
        if self.provider == "agy" and coverage is not None:
            unknown = [f for f in findings if f.verdict == "unknown"]
            legacy = sum(1 for f in unknown if f.reason.startswith("unsupported_legacy"))
            coverage["unknown_breakdown"] = {"current_session": len(unknown) - legacy, "unsupported_legacy": legacy}
        return findings

    # --- persistence ---

    def overflow_rows(self):
        """Markers for consumers that met a capacity limit: remembered so the next audit goes straight to the
        full evaluator instead of failing the same way."""
        return [row for row in self.state_rows() if row["state"].get("overflow")]

    def state_rows(self):
        """Monitor and facts states to store at the evaluated cursor (only those that advanced)."""
        rows = []
        for rule_id, monitor in self.monitors.items():
            if rule_id in self.errors or monitor.timed_out:
                continue
            if monitor.dirty or monitor.new or monitor.overflow:
                rows.append({"monitor_id": rule_id, "contract_revision": monitor.rule.revision,
                             "fingerprint": rule_fingerprint(monitor.rule, self.provider, self.parser_version, self.digest),
                             "parser_version": self.parser_version, "evaluator_version": EVALUATOR_VERSION,
                             "state": monitor.dump()})
        if self.facts.needed and (self.facts.dirty or self.facts_new or self.facts.overflow):
            rows.append({"monitor_id": "#facts", "contract_revision": 0,
                         "fingerprint": facts_fingerprint(self.provider, self.parser_version, self.digest),
                         "parser_version": self.parser_version, "evaluator_version": EVALUATOR_VERSION,
                         "state": self.facts.dump()})
        return rows


def compare(result, reference, only=None):
    """Contract ids whose finding differs between two audit results, or ``["#audit"]`` for any other difference.

    ``only``: the ids of a scoped comparison; the reference then holds just those contracts and only their findings are
    compared (the verdict, notes and coverage of a partial result are not comparable)."""
    left, right = result.model_dump(mode="json"), reference.model_dump(mode="json")
    if only is not None:
        mine = {item["contract_id"]: item for item in left["findings"] if item["contract_id"] in only}
        theirs = {item["contract_id"]: item for item in right["findings"]}
        return sorted(c for c in only if mine.get(c) != theirs.get(c))
    bad = [a["contract_id"] for a, b in zip(left["findings"], right["findings"]) if a != b]
    if len(left["findings"]) != len(right["findings"]):
        bad = ["#audit"]
    for key in ("verdict", "notes", "coverage", "contracts_snapshot", "session_id"):
        if left[key] != right[key] and not bad:
            bad = ["#audit"]
    steps = lambda r: {k: r["execution"].get(k) for k in ("executor", "graph", "nodes")}  # noqa: E731
    if steps(left) != steps(right) and not bad:
        bad = ["#audit"]
    return bad
