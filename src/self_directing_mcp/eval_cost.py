"""What one event costs in this process: the numbers a deadline-bound audit sizes its sync slice and its comparison by.

A request that syncs a large session in slices and then judges it has one budget for both. How much of it a slice may
use depends on what judging that slice will cost, and that depends on the contracts: two regexes cost a few
microseconds per event, eighteen heavy ones fifty. These are moving averages of what was measured, per event, kept in
memory only (a restart starts from the defaults below; a resident server learns within its first requests):

* ``slice``: parsing and committing one event (per provider), the cost of the sync itself;
* ``eval``: loading one new event and folding it into the monitors of a contract set (the incremental evaluation);
* ``reference``: the full evaluator over one stored event (the cost of a comparison with it), per contract set.

Nothing here decides a verdict. The numbers only move a time limit (``slice_budget``, ``index.ingest``) and tell a
comparison whether it fits the time that is left (``audit.incremental``). They are stdlib only: a hook process that
imports the engine imports this too.
"""
from __future__ import annotations

from collections import OrderedDict
import threading

DEFAULT_SLICE_SEC = 40e-6        # parse + commit of one event; the figure PR2 measured on synthetic Codex sessions
DEFAULT_EVAL_SEC = 60e-6         # load + fold of one new event when the contracts need history (floor of the prior)
EVAL_BASE_SEC = 12e-6            # the prior grows with the contracts that need history: base + per contract ...
EVAL_PER_CONTRACT_SEC = 4e-6     # ... measured 1.7 to 3.7 microseconds per contract and event, plus the load
REFERENCE_PER_EVAL = 1.5         # the full evaluator costs about this many folds per event (measured 1.2 to 1.6)
UNMEASURED_MARGIN = 1.5          # a contract set this process has not measured yet is assumed to cost this much more than
                                 # its prior: a slice that is committed cannot be cut again, so a guess that is too low costs
                                 # an evaluation that does not fit, one that is too high only costs a call
MIN_SAMPLE_EVENTS = 2000         # a measurement over fewer events is noise (fixed overheads dominate)
SMOOTHING = 0.3                  # weight of a new measurement in the moving average (the first one replaces the prior)
FLOOR_SEC, CEILING_SEC = 1e-6, 5e-3     # a measurement outside this per-event range is a defect, not a cost
KEEP_KEYS = 64                   # contract sets remembered (least recently used are dropped)


def prior_eval_sec(history_contracts: int) -> float:
    """What folding one event is expected to cost before anything was measured: 60 microseconds, more for many contracts."""
    return max(DEFAULT_EVAL_SEC, EVAL_BASE_SEC + EVAL_PER_CONTRACT_SEC * max(0, history_contracts))


class _Rate:
    """A moving average of seconds per event."""

    __slots__ = ("value", "samples")

    def __init__(self):
        self.value, self.samples = None, 0

    def update(self, seconds_per_event: float) -> None:
        sample = min(CEILING_SEC, max(FLOOR_SEC, seconds_per_event))
        self.value = sample if self.value is None else self.value + SMOOTHING * (sample - self.value)
        self.samples += 1


class CostModel:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._slice: dict[str, _Rate] = {}
            self._eval: OrderedDict[str, _Rate] = OrderedDict()
            self._reference: OrderedDict[str, _Rate] = OrderedDict()

    # --- reading ---------------------------------------------------------------------------------------------------

    def slice_cost(self, provider: str | None = None) -> float:
        """Seconds to parse and commit one event of ``provider`` (measured, else the default)."""
        with self._lock:
            rate = self._slice.get(provider or "")
            return rate.value if rate is not None and rate.value is not None else DEFAULT_SLICE_SEC

    def eval_cost(self, key: str | None, history_contracts: int = 0) -> float:
        """Seconds to load and fold one new event for the contract set ``key``: the moving average once measured, else the
        prior with its margin (``UNMEASURED_MARGIN``); 0 when no contract needs history."""
        if history_contracts <= 0:
            return 0.0
        with self._lock:
            rate = self._eval.get(key) if key is not None else None
            if rate is not None and rate.value is not None:
                self._eval.move_to_end(key)
                return rate.value
        return prior_eval_sec(history_contracts) * UNMEASURED_MARGIN

    def reference_cost(self, key: str | None) -> float | None:
        """Seconds the full evaluator needs per stored event for ``key`` when it was measured, else ``None``."""
        with self._lock:
            rate = self._reference.get(key) if key is not None else None
            return rate.value if rate is not None else None

    def observed(self, key: str | None) -> bool:
        with self._lock:
            return key is not None and key in self._eval

    # --- measuring -------------------------------------------------------------------------------------------------

    def observe_slice(self, provider: str | None, events: int, seconds: float) -> None:
        if events >= MIN_SAMPLE_EVENTS and seconds > 0:
            with self._lock:
                self._slice.setdefault(provider or "", _Rate()).update(seconds / events)

    def observe_eval(self, key: str | None, events: int, seconds: float) -> None:
        self._observe(self._eval, key, events, seconds)

    def observe_reference(self, key: str | None, events: int, seconds: float) -> None:
        self._observe(self._reference, key, events, seconds)

    def _observe(self, table, key, events, seconds) -> None:
        if key is None or events < MIN_SAMPLE_EVENTS or not seconds > 0:
            return
        with self._lock:
            rate = table.get(key)
            if rate is None:
                rate = table[key] = _Rate()
            table.move_to_end(key)
            rate.update(seconds / events)
            while len(table) > KEEP_KEYS:
                table.popitem(last=False)

    def snapshot(self) -> dict:
        """The numbers as ``audit_status`` shows them: microseconds per event, never a session, path or contract text."""
        def micro(value):
            return None if value is None else round(value * 1e6, 1)
        with self._lock:
            return {
                "slice_us": {name: micro(rate.value) for name, rate in sorted(self._slice.items())},
                "defaults_us": {"slice": micro(DEFAULT_SLICE_SEC), "eval": micro(DEFAULT_EVAL_SEC)},
                "eval_sets_measured": len(self._eval),
                "eval_us_median": micro(sorted(r.value for r in self._eval.values())[len(self._eval) // 2])
                if self._eval else None,
            }


COSTS = CostModel()
