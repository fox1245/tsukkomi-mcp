"""Contract regexes, run with a time limit.

Contract patterns come from users and agents. With the standard ``re`` module a pathological pattern can backtrack
for minutes, and it cannot be interrupted: the audit's cooperative deadline never gets a chance to run, and a host
hook that times out lets the tool call proceed unchecked. Contract patterns therefore run on the ``regex`` engine,
which avoids most catastrophic backtracking by itself and supports a per-call timeout.

Syntax is still checked with ``re`` (the documented contract syntax). Two budgets apply:

* each search may take at most ``SEARCH_TIMEOUT_SEC``;
* all searches of one contract inside one evaluation may take at most ``CONTRACT_BUDGET_SEC`` together
  (``contract_budget()``), so a pattern that is merely slow on every event cannot stall an audit either.

Exceeding either raises ``ContractRegexTimeout``. Callers turn it into an ``unknown`` finding for that contract
(``regex_timeout``): a pattern that could not be evaluated proves nothing, so it is neither clean nor a violation.
"""
from __future__ import annotations

from contextlib import contextmanager
import contextvars
import functools
import re
import time

import regex as _regex

SEARCH_TIMEOUT_SEC = 0.25
CONTRACT_BUDGET_SEC = 2.0
_FLAGS = _regex.IGNORECASE | _regex.MULTILINE | _regex.VERSION0
_remaining: contextvars.ContextVar[float | None] = contextvars.ContextVar("contract_regex_budget", default=None)


class ContractRegexTimeout(RuntimeError):
    """A contract pattern exceeded its time limit; the contract is unknown for this evaluation."""

    def __init__(self, pattern: str):
        super().__init__("contract regex exceeded its time limit")
        self.pattern = pattern


class GuardedPattern:
    __slots__ = ("pattern", "_compiled")

    def __init__(self, pattern: str, compiled):
        self.pattern, self._compiled = pattern, compiled

    def search(self, text: str):
        remaining = _remaining.get()
        limit = SEARCH_TIMEOUT_SEC if remaining is None else min(SEARCH_TIMEOUT_SEC, remaining)
        if limit <= 0:
            raise ContractRegexTimeout(self.pattern)
        started = time.perf_counter()
        try:
            return self._compiled.search(text, timeout=limit)
        except TimeoutError:
            raise ContractRegexTimeout(self.pattern) from None
        finally:
            if remaining is not None:
                _remaining.set(_remaining.get() - (time.perf_counter() - started))


@functools.lru_cache(maxsize=1024)
def compile_contract(pattern: str) -> GuardedPattern:
    """Compile a contract pattern (case-insensitive, multiline). Raises ``re.error`` for invalid syntax."""
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)       # the documented syntax gate
    try:
        compiled = _regex.compile(pattern, _FLAGS)
    except _regex.error as exc:                            # accepted by ``re`` but not by ``regex``: still invalid here
        raise re.error(str(exc)) from None
    return GuardedPattern(pattern, compiled)


def search(pattern: str, text: str):
    return compile_contract(pattern).search(text)


@contextmanager
def contract_budget(seconds: float | None = None):
    """Bound the total regex time of one contract evaluation (nested budgets keep the tighter one)."""
    outer = _remaining.get()
    budget = CONTRACT_BUDGET_SEC if seconds is None else seconds
    token = _remaining.set(budget if outer is None else min(outer, budget))
    try:
        yield
    finally:
        _remaining.reset(token)
