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

2. **Sampling** for everything else. The rate starts at 1/4 and halves (1/8, 1/16, 1/32) after every
   ``HALVE_AFTER`` consecutive comparisons that actually ran and matched; the floor is 1/32. The warm-up comparisons
   count towards the streak, so the 1/4 stage lasts ``HALVE_AFTER - WARMUP_REQUESTS`` sampled matches (4) after the warm-up.
   The rate returns to 1/4 only when a premise of the verification changes (the same list as above); an ordinary append does
   not reset it, and neither does a comparison that was skipped for lack of time (that is no match either). The draw is a
   hash of the schedule state and the cursor, so it is deterministic and reproducible.

3. ``SELF_DIRECT_AUDIT_SHADOW_EVERY`` set to an integer pins the legacy fixed rate and switches adaptation off: ``1``
   compares every audit, ``0`` only the forced ones, ``N`` about one audit in N. The forced comparisons stay.

The three constants below (1/4 down to 1/32, halving after 8 matches, a warm-up of 4) are the chair's decision on issue #18
R2 and deviate on purpose from the roundtable's text (1/4 down to 1/64, halving after 32, a warm-up of 8), on measured
grounds. With the roundtable's constants a session compared more of its requests than the fixed 1 in 16 had until it had run
about 2,000 requests, so no p95 got lower in a session of a few hundred requests. A floor of 1/64 (1.6% of the requests)
buys no percentile, since it is still above the 1% a p99 sees, and it doubles the time a silently missed violation goes
unnoticed; 1/32 (3.1%) is below the 5% a p95 sees, which is the point of an adaptive floor. The numbers are in the README
(``Incremental audits``) and the changelog; ``scripts/measure_shadow_detection.py --simulate`` and ``--share`` reproduce them.

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
RATE_DENOMINATORS = (4, 8, 16, 32)         # 1/4 ... 1/32, in order of decreasing rate; the last is the floor
HALVE_AFTER = 8                            # consecutive compared-and-matching requests before the rate halves
WARMUP_REQUESTS = 4                        # forced comparisons after a rebuild or a change of premise
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
    # The request kinds (keys of ``flags``) this writer changed since it read the state. Not persisted: a writer that saves
    # while another saved since it read must not erase that other writer's flags (see ``merge_flags``).
    touched: frozenset = field(default=frozenset(), repr=False, compare=False)

    def dump(self) -> dict:
        return {"premise": self.premise, "rate_index": self.rate_index, "streak": self.streak, "warmup": self.warmup,
                "compared": self.compared, "last_match_at": self.last_match_at,
                "last_match_offset": self.last_match_offset, "flags": self.flags, "memo": self.memo,
                "reset_reason": self.reset_reason}

    @classmethod
    def load(cls, epoch: int, data: dict) -> "ShadowState":
        """A row that cannot be read (damaged, or written for other constants) raises ``ValueError``, ``TypeError`` or
        ``KeyError``; the caller treats that as ``unreadable``, which a fresh state replaces."""
        if not isinstance(data, dict):
            raise ValueError("shadow state is not an object")
        numbers = [data[name] for name in ("rate_index", "streak", "warmup", "compared")]
        if not all(type(number) is int for number in numbers):
            raise ValueError("shadow state numbers must be integers")
        flags = data.get("flags", {})
        if not isinstance(flags, dict) or not all(isinstance(v, list) and all(isinstance(i, str) for i in v)
                                                  for v in flags.values()):
            raise ValueError("shadow state flags")
        memo = data.get("memo")
        if memo is not None and not _valid_memo(memo):
            raise ValueError("shadow state memo")
        at, offset = data.get("last_match_at"), data.get("last_match_offset")
        if not (at is None or isinstance(at, str)) or not (offset is None or type(offset) is int):
            raise ValueError("shadow state last match")
        state = cls(epoch=int(epoch), premise=dict(data["premise"]), rate_index=numbers[0], streak=numbers[1],
                    warmup=numbers[2], compared=numbers[3], last_match_at=at, last_match_offset=offset,
                    flags={str(k): list(v) for k, v in flags.items()}, memo=memo,
                    reset_reason=data.get("reset_reason"))
        if not 0 <= state.rate_index < len(RATE_DENOMINATORS) or min(state.streak, state.warmup, state.compared) < 0:
            raise ValueError("shadow state out of range")
        return state

    @classmethod
    def unreadable(cls, epoch: int) -> "ShadowState":
        """Stands for a stored row that cannot be read. It keeps the row's epoch, so that the fresh state that replaces it
        wins the epoch guard of the table, and has a premise no request has, so that the next request resets."""
        return cls(epoch=int(epoch), premise={}, reset_reason="unreadable")

    def copy(self) -> "ShadowState":
        return replace(self, premise=dict(self.premise), flags={k: list(v) for k, v in self.flags.items()},
                       memo=dict(self.memo) if self.memo else None)


def _valid_memo(memo) -> bool:
    return (isinstance(memo, dict) and isinstance(memo.get("state"), list) and isinstance(memo.get("request"), str)
            and isinstance(memo.get("result"), str)
            and (memo.get("ids") is None or (isinstance(memo["ids"], list) and all(isinstance(i, str) for i in memo["ids"]))))


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
    changed = ("unreadable" if not state.premise
               else next((key for key in premise if state.premise.get(key) != premise[key]), "premise"))
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


def memo_covers(state, state_key, key, result, ids=None) -> bool:
    """True when this very input was already compared with the reference and matched: same stored events, same request kind,
    same incremental answer, and (``ids``) every contract that is now asked for was among those compared. The reference is
    deterministic, so comparing again could not find anything."""
    memo = state.memo
    if not memo or memo.get("state") != list(state_key) or memo.get("request") != key \
            or memo.get("result") != result_digest(result):
        return False
    return ids is None or set(ids) <= set(memo.get("ids") or ())


def merge_flags(stored, mine, touched) -> dict:
    """The flags to store when a writer saves, given the flags already stored (another writer may have saved since this one
    read the state): the stored flags, with the entries of the request kinds this writer handled replaced by its own. A
    writer that handled none of them (``touched`` empty) keeps its own flags."""
    if not touched:
        return {k: list(v) for k, v in mine.items()}
    out = {k: list(v) for k, v in (stored or {}).items() if k not in touched}
    for key in touched:
        if mine.get(key):
            out[key] = list(mine[key])          # appended last: the dict keeps the most recent kinds
    while len(out) > FLAG_REQUESTS:
        del out[next(iter(out))]
    return out


def settle(state, *, outcome, full_scope, key, flagged, kept, at, offset, state_key=None, result=None, premise=None,
           memo_ids=None):
    """The state after a request.

    ``outcome``: ``"match"`` or ``"mismatch"`` for a comparison that ran, ``"skipped"`` or ``None`` for one that did not.
    ``full_scope``: the comparison covered every contract (only these count towards the schedule).
    ``flagged``: ids the answer judged violation/suspicious; ``kept``: flipped ids whose forced comparison did not run (so
    the next request still sees the flip). ``state_key`` and ``result`` (with ``memo_ids``, the contracts that comparison
    covered) remember a forced comparison that matched, see ``memo_covers``."""
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
            if memo_ids is not None:
                new.memo["ids"] = sorted(memo_ids)
    new.flags = _flags(new.flags, key, flagged | set(kept))
    new.touched = new.touched | {key}               # the flags of this request kind are this writer's to write
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
