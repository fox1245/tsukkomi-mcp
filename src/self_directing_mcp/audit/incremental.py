"""Incremental contract audit: monitors whose state summarizes the already-judged prefix of a session.

``runner.audit_contract`` (plus the provider evidence classes) judges a rule by walking the whole event list.
This module judges the same rule by keeping, per rule, a small summary of the prefix and folding in only the
events that arrived after it. The two must agree on every input; the full evaluator is the reference and wins
whenever they differ (see ``AuditPlan`` and the shadow comparison in ``engine``).

What is summarized, per contract kind (all state is bounded and JSON-serializable):

* anchors (``source_event_id``, ``applies_from_event_id``): whether each was seen and where the activation
  event sits; events before an activation anchor are outside the rule's window;
* ``when``: whether a call-evidence event inside the window matched;
* ``must_not``: the match count, the first 20 matches as evidence and, with an ``exception_regex`` (which the
  reference matches against the join of every hit), the hit texts up to a cap;
* ``must``: the last three regex hits as evidence plus, for ``requires_success``, a tracker of the latest hit's
  call (how many calls share its id, how many results followed, whether the first one succeeded);
* ``before_regex``: each triggering action is judged at its own position (the verdict depends on the prefix
  only), so the worst verdict, the distinct reasons, the first 20 evidence items and the last trigger position
  are kept;
* provider evidence classes: Claude's cut results and AGY's truncated fields, pending results, legacy results,
  host-logged proposals and denials, as shared "facts" plus a few first-occurrence markers per rule.

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

from self_directing_mcp.agy import JUDGED_FIELDS, RESULT_TYPES
from self_directing_mcp.audit import runner
from self_directing_mcp.audit.patterns import ContractRegexTimeout, compile_contract, contract_budget
from self_directing_mcp.audit import patterns as _patterns
from self_directing_mcp.audit.applicability import PROPOSED_CALL_PREFIX, not_applicable_reason
from self_directing_mcp.request_control import IndexBusy, RequestStopped, check_request, current_request
from self_directing_mcp.schemas import Evidence, Finding
from self_directing_mcp.security.mask import mask_secrets

# Bump when the semantics of any evaluation changes (rules, evidence classes, evidence text); it invalidates every
# stored state and re-enables contracts that a shadow mismatch disabled.
EVALUATOR_VERSION = 1
STATE_VERSION = 1

# The full evaluator costs about this many seconds per stored event (measured on a 225k-event session). A shadow run
# is skipped when the request's remaining budget cannot cover it with margin; it must never turn a request that
# the full evaluator alone would have answered in time into a timeout.
SHADOW_SECONDS_PER_EVENT = 30e-6

EVIDENCE_LIMIT = 20
EXCEPTION_MAX_HITS = 512
EXCEPTION_MAX_CHARS = 262_144
ATTEMPT_LIST = 64
PENDING_MAX = 2048
BLOCKED_MAX = 512
TARGETS_MAX = 4096
CUT_MAX = 2048

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
    """Tool results that a ``read_truncation_notice`` marks as cut, whatever order notice and result arrive in."""

    needed = True

    def __init__(self):
        self.targets, self.cut = [], []
        self._targets = set()

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

    def _cut(self, position, chunk_id):
        if len(self.cut) >= CUT_MAX:
            self.overflow = "cut result table is full"
            return
        self.cut.append((position, chunk_id))
        self.dirty = True

    def dump(self):
        return {"overflow": self.overflow, "targets": self.targets, "cut": [[list(p), c] for p, c in self.cut]}

    def load(self, data):
        self.overflow = data.get("overflow")
        self.targets = list(data.get("targets", ()))
        self._targets = set(self.targets)
        self.cut = [(tuple(p), c) for p, c in data.get("cut", ())]


class AgyFacts(Facts):
    """RUNNING results still waiting for a final record of their step, and the proposals the Pre hook blocked."""

    needed = True

    def __init__(self, resolve=True):
        # ``resolve``: some rule reads host-logged proposals, so their text is loaded and denials can be matched.
        self.resolve = resolve
        self.unresolved = []  # (key, step_index), in order
        self.blocked = {}     # key -> (tool name, proposal text)
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
                self.blocked[position] = (name, text)
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
                "blocked": [[list(p), n, t] for p, (n, t) in self.blocked.items()],
                "unsure": [[list(p), n] for p, n in self.unsure]}

    def load(self, data):
        self.overflow = data.get("overflow")
        self.unresolved = [(tuple(p), s) for p, s in data.get("unresolved", ())]
        self.blocked = {tuple(p): (n, t) for p, n, t in data.get("blocked", ())}
        self.unsure = [(tuple(p), n) for p, n in data.get("unsure", ())]


def make_facts(provider, needs=(), rules=()):
    """The shared summaries ``rules`` need for ``provider``'s evidence classes (a plain ``Facts`` when none)."""
    if provider == "agy":
        return AgyFacts(resolve=any(scope == "attempt" for scope, _roles in needs))
    if provider == "claude" and any(runner._reads_results(r) or r.requires_success for r in rules):
        return ClaudeFacts()
    return Facts()


# --- rule monitors -------------------------------------------------------------------------------------------------

class RuleMonitor:
    """The summary of one rule over the events fed so far, and the finding the reference would give for them."""

    def __init__(self, rule, provider):
        self.rule, self.provider = rule, provider
        self.agy, self.claude = provider == "agy", provider == "claude"
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
        # state
        self.src = False
        self.started = self.from_id is None
        self.from_key = None
        self.when_met = False
        self.hits, self.hit_ev, self.exc, self.exc_chars = 0, [], [], 0
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
            "src": self.src, "started": self.started, "from_key": self.from_key, "when": self.when_met,
            "hits": self.hits, "hit_ev": self.hit_ev, "exc": self.exc, "exc_chars": self.exc_chars,
            "hit_n": self.hit_n, "last3": self.last3, "lh": self.lh,
            "trig_n": self.trig_n, "trig_worst": self.trig_worst, "trig_reasons": self.trig_reasons,
            "trig_items": self.trig_items, "trig_last": self.trig_last,
            "cut": self.cut, "legacy": self.legacy, "att_n": self.att_n, "att": self.att, "att_chars": self.att_chars,
            "att_closed": self.att_closed,
            "trg_n": self.trg_n, "trg": self.trg, "src_tu": self.src_tu, "overflow": self.overflow,
        }

    def load(self, data):
        self.src, self.started, self.when_met = data["src"], data["started"], data["when"]
        self.from_key = _tuple(data["from_key"])
        self.hits, self.hit_ev, self.exc, self.exc_chars = data["hits"], data["hit_ev"], data["exc"], data["exc_chars"]
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
        return (self.kind is None or event.kind == self.kind) and event.meta.get("role") in self.roles

    def _feed_forbidden(self, event, pools):
        if (not self.inert and (self.kind is None or event.kind == self.kind)
                and event.meta.get("role") in self.roles and self.regex.search(event.text)):
            self._forbidden(event)

    def _feed_required(self, event, pools):
        if self.inert:
            return
        matched = False
        if (self.kind is None or event.kind == self.kind) and event.meta.get("role") in self.roles:
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
        if len(self.hit_ev) < EVIDENCE_LIMIT:
            self.hit_ev.append(_ev(event, flag=self.agy))
        if self.rule.exception_regex:
            if len(self.exc) >= EXCEPTION_MAX_HITS or self.exc_chars + len(event.text) > EXCEPTION_MAX_CHARS:
                self.overflow, self.inert = "exception hit store is full", True
                return
            self.exc.append(event.text)
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
                self.lh = [_k(call_id), pools.earlier_calls(call_id, self.from_key, event.key) + 1, 0, 0, None]
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
        if meta.get("evidence_origin") == "model_proposal":
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

    def _make(self, verdict, reason, items=None, basis=None, applicability="applicable"):
        rule = self.rule
        return Finding(contract_id=rule.id, contract_revision=rule.revision, severity=rule.severity,
                       verdict=verdict, reason=reason, evidence=[_evidence(i) for i in items or ()], basis=basis,
                       applicability=applicability)

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
        if (self.src_id and not self.src) or (self.from_id and not self.started):
            return self._make("unknown", "contract source or activation event is not in this session")
        if self.regex_invalid:
            return self._make("unknown", "invalid contract regex")
        if rule.when is not None:
            if self.when_invalid:
                return self._make("unknown", "invalid `when` regex", basis=f"{rule.id}.when failed to compile")
            if not self.when_met:
                return self._make("clean", not_applicable_reason(rule),
                                  basis=f"{rule.id}.when did not match any observed tool call",
                                  applicability="not_applicable")
        item, tu = self._typed(proposed)
        history = proposed is None
        if self.agy:
            self._post_agy(item, tu, history, facts)
        elif self.claude:
            self._post_claude(item, history, facts)
        return item

    def _typed(self, proposed):
        rule = self.rule
        if self.is_must_not:
            if proposed is not None:
                hit = self.eligible(proposed) and self.regex.search(proposed.text) is not None
                count, evidence, joined = (1, [_ev(proposed, flag=self.agy)], proposed.text) if hit else (0, [], "")
            else:
                if self.hits and rule.exception_regex and (self.hits > len(self.exc)):
                    raise NeedFull("exception hit store is incomplete")
                count, evidence, joined = self.hits, self.hit_ev[:EVIDENCE_LIMIT], " ".join(self.exc)
            if not count:
                return self._make("clean", "no forbidden match in inspected events",
                                  basis=f"{rule.id}.regex did not match any inspected event"), {}
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
                          "user-authorized exception requires confirmation"), {}
            reason = f"forbidden regex matched {count} event(s)"
            if rule.source_quote:
                reason += f"; user said: {mask_secrets(rule.source_quote)[:120]}"
            return self._make("violation", reason, basis=f"{rule.id}.regex '{rule.regex}' matched prohibited content",
                              items=evidence), {i["i"]: i.get("tu") for i in evidence}
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
        gone = sum(1 for key, (_name, text) in blocked.items()
                   if self._after_start(key) and self.regex.search(text))
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
        reason = f"forbidden regex matched {count} host-logged call attempt(s); execution is not proven"
        if rule.source_quote:
            reason += f"; user said: {mask_secrets(rule.source_quote)[:120]}"
        item.verdict, item.reason = verdict, reason
        item.basis = f"{rule.id}.regex '{rule.regex}' matched a proposed call; the attempt itself is the violation"
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
    payload = json.dumps([EVALUATOR_VERSION, STATE_VERSION, provider, parser_version, mask_fingerprint(), digest,
                          rule.model_dump(mode="json")], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def facts_fingerprint(provider, parser_version, digest) -> str:
    payload = json.dumps([EVALUATOR_VERSION, STATE_VERSION, provider, parser_version, digest, "facts"])
    return hashlib.sha256(payload.encode()).hexdigest()


def wants_call_index(active, provider) -> bool:
    """Whether a run may look stored events up by call id (a requires_success tracker, Claude cut results)."""
    return any(r.regex and (r.requires_success or (provider == "claude" and runner._reads_results(r)))
               for r in active)


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
    return control.deadline - time.monotonic() > events * SHADOW_SECONDS_PER_EVENT * 1.5 + 0.5


def _shadow_sample(session_id, offset, active, every) -> bool:
    if every <= 0:
        return False
    if every == 1:
        return True
    token = f"{session_id}|{offset}|" + "|".join(f"{r.id}:{r.revision}" for r in active)
    return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % every == 0


def fold(events, monitors, facts, pools):
    """Feed ``events`` to every consumer that has not seen them (``monitors``: ``(monitor, offset)`` pairs; ``facts``:
    ``(facts, offset)`` or ``None``); a consumer skips events whose record starts before its offset."""
    shared, shared_offset = facts if facts is not None else (None, 0)
    shared_on = shared is not None and not shared.overflow
    feeds = [(monitor.feed, offset, monitor) for monitor, offset in monitors]
    note = pools.note if pools.track_calls or pools.track_proposals else None
    for number, event in enumerate(events):
        if not number & 511:
            check_request()
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
                 parser_version, report, reference):
        self.session_id, self.provider, self.mode = session_id, provider, mode
        self.events, self.count, self.reference = events, count, reference
        self.history_ids, self.monitors, self.offsets = history_ids, monitors, offsets
        self.facts, self.facts_offset, self.facts_new = facts, facts_offset, facts_new
        self.pools, self.routes, self.rebuilds = pools, routes, rebuilds
        self.support, self.snapshot, self.shadow = support, snapshot, shadow
        self.digest, self.parser_version, self.report = digest, parser_version, report
        self.fed = False
        self.overflowed, self.errors, self.timeouts = {}, {}, set()

    # --- preparation (inside the read boundary) ---

    @classmethod
    def prepare(cls, store, *, session_id, provider, active, planned, needs, link_calls, shadow_every,
                force_full=frozenset(), report=None):
        """Decide how to evaluate and read the events needed; ``None`` means evaluate everything the old way."""
        report = {} if report is None else report
        history = [r for r in active if r.regex and (r.type == "must" or planned is None)]
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
        sampled = _shadow_sample(session_id, cursor["byte_offset"], active, shadow_every)
        shadow = sampled or bool(rebuilds) or bool(facts_rebuild)
        if shadow and not _shadow_affordable(store.session_metrics(session_id, provider)[0]):
            shadow = False
            report["shadow_skipped"] = "deadline"

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
        if shadow or full:
            snapshot_events = store.capture_audit_events(session_id, provider, needs, link_calls=link_calls)
            reference = list(snapshot_events.events)
            if not full:
                bad = tail_mismatch(reference, "full")
                if bad:
                    drop(bad)
                    full = True
                    shadow = shadow or "shadow_skipped" not in report
        events, mode, count = [], "delta", latest["event_count"] if chain_ok else 0
        if not full:
            from_byte = min([*offsets.values(), *([facts_row["byte_offset"]] if facts_row is not None else [])])
            events = store.capture_audit_tail(session_id, provider, needs, link_calls=link_calls, from_byte=from_byte)
            bad = tail_mismatch(events, "delta")
            if bad:
                drop(bad)
                full = True
                shadow = shadow or "shadow_skipped" not in report
                if reference is None:
                    snapshot_events = store.capture_audit_events(session_id, provider, needs, link_calls=link_calls)
                    reference = list(snapshot_events.events)
        if full:
            events, mode = reference, "full"
            count = latest["event_count"] if chain_ok else snapshot_events.count
            bad = tail_mismatch(events, "full")
            if bad:
                drop(bad)
                shadow = shadow or "shadow_skipped" not in report
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
        if facts_rebuild:
            report["rebuilt"]["#facts"] = facts_why or "event_count"
        return cls(session_id=session_id, provider=provider, mode=mode, events=events, count=count,
                   history_ids=history_ids, monitors=monitors, offsets=offsets, facts=facts,
                   facts_offset=facts_offset, facts_new=facts_row is None, pools=pools, routes=routes,
                   rebuilds=rebuilds, support=support, snapshot=snapshot, shadow=shadow, digest=digest,
                   parser_version=parser_version, report=report, reference=reference)

    # --- the audit stages (no database) ---

    def load(self, active):
        """Fold the events read into every monitor; nothing else has to be walked."""
        if self.fed:
            return
        self.fed = True
        try:
            # One pooled budget for the walk (a per-event, per-monitor budget would cost more than the walk). A monitor
            # that runs out is dropped and its rule is judged by the reference evaluator with its own budget.
            with contract_budget(_patterns.CONTRACT_BUDGET_SEC * max(1, len(self.monitors))):
                fold(self.events, [(monitor, self.offsets[monitor.rule.id]) for monitor in self.monitors.values()],
                     (self.facts, self.facts_offset) if self.facts.needed else None, self.pools)
        except (RequestStopped, IndexBusy):
            raise
        except Exception as exc:
            if self.mode != "full":
                raise PlanRetry(self.monitors, "monitor_error:" + type(exc).__name__, None, "error") from exc
            raise

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
            found = [runner.audit_contract(r, session_id=self.session_id, store=None, provider=self.provider,
                                           proposed=proposed, chunks=chunks) for r in rules]
            evidence_rules = runner._EVIDENCE_RULES.get(self.provider)
            if evidence_rules is not None:
                evidence_rules(rules, found, chunks, proposed=proposed, coverage=None)
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


def compare(result, reference):
    """Contract ids whose finding differs between two audit results, or ``["#audit"]`` for any other difference."""
    left, right = result.model_dump(mode="json"), reference.model_dump(mode="json")
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
