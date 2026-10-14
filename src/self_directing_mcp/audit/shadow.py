"""When the incremental audit is compared with the full evaluator: the schedule, the forced comparisons, the state.

The full evaluator is the reference; ``audit.incremental`` is the fast path that must agree with it. A *shadow
comparison* runs the reference beside the fast path on the same stored events and compares the findings (see
``SelfDirectEngine._audit``). Running it on every request would cost the saving, so most requests are not compared. This
module decides which ones are. It is pure policy and state: no database, no events, no engine.

What the policy does, and what it cannot do
-------------------------------------------

1. **Forced comparisons** (whatever the rate):

   * ``rebuild`` / ``warmup``: after a rebuild of a monitor state, and for the first ``WARMUP_REQUESTS`` compared requests
     after a change of the verification premise (a new session, a changed contract set or revision, a new parser or
     evaluator version, a new session generation, a mismatch);
   * ``violation``: the fast path says ``violation`` or ``suspicious`` for a contract (a wrong block is the costly error);
   * ``flip``: a contract that the previous request of the same kind judged ``violation`` or ``suspicious`` is ``clean``
     now (the costly error in the other direction; events only append, so a real flip is rare);
   * ``high_risk``: a ``risk: high`` contract applies to the proposed action, or its stored summary changed in this request.
     Only that contract is compared (a *scoped* comparison); the existence of a high-risk contract does not make every
     other request compare anything.

2. **Sampling** for everything else. The rate starts at 1/4 and halves (1/8, 1/16, 1/32, 1/64) after every
   ``HALVE_AFTER`` consecutive comparisons that actually ran and matched. It returns to 1/4 only when a premise of the
   verification changes (the same list as above); an ordinary append does not reset it, and neither does a comparison
   that was skipped for lack of time (that is no match either). The draw is a hash of the schedule state and the cursor,
   so it is deterministic and reproducible.

3. ``SELF_DIRECT_AUDIT_SHADOW_EVERY`` set to an integer pins the legacy fixed rate and switches adaptation off: ``1``
   compares every audit, ``0`` only the forced ones, ``N`` about one audit in N. The forced comparisons stay.

What this does not give: a guarantee that every response equals the full evaluation. A wrongly *clean* answer can pass
between two comparisons; the forced comparisons narrow that (a violation that disappears and a high-risk contract are
checked at once) but do not close it. The idle comparison of a resident engine (``shadow_worker``) shortens the time such
an answer stays unnoticed; it never certifies the answers already given, and a mismatch it finds is recorded as a range of
requests that need review.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json

ADAPTIVE = "adaptive"
RATE_DENOMINATORS = (4, 8, 16, 32, 64)     # 1/4 ... 1/64, in order of decreasing rate
HALVE_AFTER = 32                           # consecutive compared-and-matching requests before the rate halves
WARMUP_REQUESTS = 8                        # forced comparisons after a rebuild or a change of premise
FLAG_REQUESTS = 16                         # request kinds whose flagged contracts are remembered
FLAGGED = ("violation", "suspicious")

REASON_ORDER = ("mismatch", "rebuild", "premise", "warmup", "idle", "sampled", "violation", "flip", "high_risk")


def parse_every(value):
    """The setting: ``"adaptive"`` or a non-negative integer. Returns ``None`` for adaptive, else the integer."""
    if value is None or (isinstance(value, str) and value.strip().lower() == ADAPTIVE):
        return None
    return int(value)


def rate_text(state, every) -> str:
    """The sampling rate as ``1/N`` (``0`` and ``1`` for the fixed settings that compare never or always)."""
    if every is None:
        return f"1/{RATE_DENOMINATORS[state.rate_index]}"
    return "never" if every == 0 else "1/1" if every == 1 else f"1/{every}"


@dataclass
class ShadowState:
    """The schedule of one session and provider. Persisted with the other audit states (``index.shadow_state``)."""

    epoch: int = 0                          # bumped on every reset; an older writer never overwrites a newer epoch
    premise: dict = field(default_factory=dict)
    rate_index: int = 0
    streak: int = 0                         # compared-and-matching requests since the rate last changed
    warmup: int = field(default_factory=lambda: WARMUP_REQUESTS)    # forced comparisons still owed
    compared: int = 0                       # full comparisons that ran since the reset; part of the draw
    last_match_at: str | None = None
    last_match_offset: int | None = None
    flags: dict = field(default_factory=dict)       # request key -> contract ids judged violation/suspicious last time
    memo: dict | None = None                # the last forced comparison that matched (see ``memo_covers``)
    reset_reason: str | None = None

    def dump(self) -> dict:
        return {"premise": self.premise, "rate_index": self.rate_index, "streak": self.streak, "warmup": self.warmup,
                "compared": self.compared, "last_match_at": self.last_match_at,
                "last_match_offset": self.last_match_offset, "flags": self.flags, "memo": self.memo,
                "reset_reason": self.reset_reason}

    @classmethod
    def load(cls, epoch: int, data: dict) -> "ShadowState":
        """An unreadable row is the same as no row (the caller treats ``ValueError`` that way)."""
        state = cls(epoch=int(epoch), premise=dict(data["premise"]), rate_index=int(data["rate_index"]),
                    streak=int(data["streak"]), warmup=int(data["warmup"]), compared=int(data["compared"]),
                    last_match_at=data.get("last_match_at"), last_match_offset=data.get("last_match_offset"),
                    flags={str(k): [str(i) for i in v] for k, v in dict(data.get("flags") or {}).items()},
                    memo=data.get("memo"), reset_reason=data.get("reset_reason"))
        if not 0 <= state.rate_index < len(RATE_DENOMINATORS) or min(state.streak, state.warmup, state.compared) < 0:
            raise ValueError("shadow state out of range")
        return state

    def copy(self) -> "ShadowState":
        return replace(self, premise=dict(self.premise), flags={k: list(v) for k, v in self.flags.items()},
                       memo=dict(self.memo) if self.memo else None)


def digest(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def contracts_digest(rules) -> str:
    """The contract set as the verification sees it: ids and revisions of the enabled contracts that apply here."""
    return digest(sorted((r.id, r.revision) for r in rules))


def make_premise(rules, parser_version, evaluator_version, generation) -> dict:
    return {"contracts": contracts_digest(rules), "parser": parser_version, "evaluator": evaluator_version,
            "generation": generation}


def reconcile(state, premise):
    """The state to use for a request: the stored one while its premise holds, else a fresh start at 1/4.

    Returns ``(state, reason)``; ``reason`` names the premise component that changed (``new`` for a session the policy has
    not seen), or is ``None``. The stored object is never modified."""
    if state is None:
        fresh = ShadowState(epoch=1, premise=dict(premise), reset_reason="new")
        return fresh, "new"
    if state.premise == premise:
        return state, None
    changed = next((key for key in premise if state.premise.get(key) != premise[key]), "premise")
    return ShadowState(epoch=state.epoch + 1, premise=dict(premise), reset_reason=changed,
                       flags={k: list(v) for k, v in state.flags.items()}), changed


def sampled(state, session_id, provider, offset, every) -> bool:
    """Whether the draw selects this request (deterministic given the state and the cursor)."""
    if every is None:
        denominator = RATE_DENOMINATORS[state.rate_index]
        token = f"{provider}|{session_id}|{state.epoch}|{state.compared}|{offset}"
    elif every <= 0:
        return False
    elif every == 1:
        return True
    else:
        denominator = every
        token = f"{provider}|{session_id}|{offset}|{state.premise.get('contracts')}"
    return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % denominator == 0


def request_key(planned) -> str:
    """The kind of request, so that 'the previous request' means the previous one of the same kind and action."""
    if planned is None:
        return "audit"
    return "check:" + hashlib.sha256(planned.text.encode("utf-8", "replace")).hexdigest()[:16]


def flipped(state, key, verdicts) -> set:
    """Contracts flagged by the previous request of this kind that the fast path now reports clean."""
    return {contract for contract in state.flags.get(key, ()) if verdicts.get(contract) == "clean"}


def result_digest(result) -> str:
    """Identity of what a comparison covers: the fields ``audit.incremental.compare`` looks at."""
    dumped = result.model_dump(mode="json")
    return digest({k: dumped[k] for k in ("verdict", "findings", "notes", "coverage", "contracts_snapshot", "session_id")})


def memo_covers(state, state_key, key, result) -> bool:
    """True when this very input was already compared with the reference and matched: same stored events, same contracts,
    same request, same incremental answer. The reference is deterministic, so comparing again could not find anything."""
    memo = state.memo
    return bool(memo) and memo.get("state") == list(state_key) and memo.get("request") == key \
        and memo.get("result") == result_digest(result)


def settle(state, *, outcome, full_scope, key, flagged, kept, at, offset, state_key=None, result=None, premise=None):
    """The state after a request.

    ``outcome``: ``"match"`` or ``"mismatch"`` for a comparison that ran, ``"skipped"`` or ``None`` for one that did not.
    ``full_scope``: the comparison covered every contract (only these count towards the schedule).
    ``flagged``: ids the answer judged violation/suspicious; ``kept``: flipped ids whose forced comparison did not run (so
    the next request still sees the flip)."""
    new = state.copy()
    if outcome == "mismatch":
        return ShadowState(epoch=new.epoch + 1, premise=dict(new.premise), reset_reason="mismatch",
                           flags=_flags(new.flags, key, flagged | set(kept)))
    if outcome == "match":
        if full_scope:
            new.compared += 1
            new.warmup = max(0, new.warmup - 1)
            new.streak += 1
            new.last_match_at, new.last_match_offset = at, offset
            if new.streak >= HALVE_AFTER and new.rate_index < len(RATE_DENOMINATORS) - 1:
                new.rate_index, new.streak = new.rate_index + 1, 0
        if state_key is not None and result is not None:
            new.memo = {"state": list(state_key), "request": key, "result": result_digest(result)}
    new.flags = _flags(new.flags, key, flagged | set(kept))
    return new


def _flags(flags, key, flagged) -> dict:
    out = {k: list(v) for k, v in flags.items() if k != key}
    if flagged:
        out[key] = sorted(flagged)                 # re-inserted last: the dict keeps the most recent kinds
    while len(out) > FLAG_REQUESTS:
        del out[next(iter(out))]
    return out


def arm_warmup(state):
    """After a rebuild the next ``WARMUP_REQUESTS`` compared requests are forced (the rate is not reset)."""
    if state.warmup >= WARMUP_REQUESTS:
        return state
    new = state.copy()
    new.warmup = WARMUP_REQUESTS
    return new


@dataclass
class Comparison:
    """What one request will be compared on: ``scope`` is ``None`` (nothing), ``"all"`` or a set of contract ids."""

    scope: object = None
    reasons: list = field(default_factory=list)

    def add(self, reason, scope="all"):
        if reason not in self.reasons:
            self.reasons.append(reason)
            self.reasons.sort(key=REASON_ORDER.index)
        if scope == "all" or self.scope == "all":
            self.scope = "all"
        elif scope:
            self.scope = set(scope) | (self.scope or set())
        return self

    @property
    def active(self) -> bool:
        return self.scope is not None and self.scope != set()

    @property
    def full(self) -> bool:
        return self.scope == "all"

    def covers(self, ids) -> bool:
        return self.full or (self.scope is not None and set(ids) <= self.scope)
