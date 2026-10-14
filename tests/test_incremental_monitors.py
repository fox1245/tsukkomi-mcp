"""Incremental monitors against the reference evaluator, on random synthetic sessions.

After every append the monitors (their state round-tripped through JSON, as the store does) must give exactly the
findings ``runner.full_findings`` gives on the same prefix, for the history audit and for a proposed action.
"""
from __future__ import annotations

import json
import random

import pytest

from self_directing_mcp.audit import incremental as inc
from self_directing_mcp.audit import runner
from incremental_support import Session, contracts, proposed_action

PROVIDERS = ["codex", "claude", "agy", "omp"]


def prior_pools(previous, lookback=4096):
    """What the store would prefetch for the events before a run: calls by id and proposals by tool name."""
    calls, proposals = {}, {}
    for event in previous:
        call_id = event.meta.get("call_id")
        if call_id is not None and event.kind in ("tool_call", "tool_result"):
            calls.setdefault(inc._k(call_id), []).append((event.kind, event.key, event.chunk_id))
        if event.meta.get("evidence_origin") == "model_proposal":
            proposals.setdefault(event.meta.get("tool_name"), []).append((event.key, event.text))
    window = previous[-lookback:]
    complete = len(previous) <= lookback
    names = {name for name in proposals}
    pools = inc.Pools(track_calls=True, track_proposals=True, calls_prior=calls)
    for name in names:
        rows = [(e.key, e.text) for e in reversed(window)
                if e.meta.get("evidence_origin") == "model_proposal" and e.meta.get("tool_name") == name]
        pools.proposals_prior[name] = (rows, complete)
    return pools


def reload(monitors, facts, provider, rules, needs):
    by_id = {r.id: r for r in rules}
    fresh = {}
    for rule_id, monitor in monitors.items():
        copy = inc.RuleMonitor(by_id[rule_id], provider)
        copy.load(json.loads(json.dumps(monitor.dump())))
        fresh[rule_id] = copy
    shared = inc.make_facts(provider, needs, rules)
    shared.load(json.loads(json.dumps(facts.dump())))
    return fresh, shared


def needs_for(rules):
    return [("attempt", ("runtime",))] if any(runner._wants_attempts(r) and r.regex for r in rules) else []


def run_case(seed, provider, *, events=70, rules=7, lookback=None):
    rng = random.Random(f"{provider}/{seed}")
    session = Session(provider, rng)
    session.extend(events)
    rule_list = contracts(rng, provider, session.events, rules)
    needs = needs_for(rule_list)
    monitors = {r.id: inc.RuleMonitor(r, provider) for r in rule_list}
    facts = inc.make_facts(provider, needs, rule_list)
    fed, full_needed = 0, set()
    while fed < len(session.events):
        size = rng.choice([1, 2, 3, 5, 8, 13, 30])
        segment = session.events[fed:fed + size]
        pools = prior_pools(session.events[:fed]) if fed else inc.Pools(track_calls=True, track_proposals=True)
        inc.fold(segment, [(m, 0) for m in monitors.values()], (facts, 0) if facts.needed else None, pools)
        fed += len(segment)
        if rng.random() < 0.6:
            monitors, facts = reload(monitors, facts, provider, rule_list, needs)
        prefix = session.events[:fed]
        for proposed in (None, proposed_action(rng)):
            want = runner.full_findings(rule_list, prefix, session_id="s", store=None, provider=provider,
                                        proposed=proposed, coverage={})
            for rule, expected in zip(rule_list, want):
                try:
                    got = monitors[rule.id].finding(proposed, facts)
                except inc.NeedFull as exc:
                    full_needed.add((rule.id, exc.reason))
                    continue
                assert got.model_dump() == expected.model_dump(), (
                    f"{provider} seed={seed} rule={rule.model_dump(exclude_defaults=True)} "
                    f"proposed={proposed is not None} events={fed}\nincremental={got.model_dump()}\n"
                    f"reference={expected.model_dump()}")
    return full_needed


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("block", range(8))
def test_monitors_match_the_reference_after_every_append(provider, block):
    needed = set()
    for seed in range(block * 25, block * 25 + 25):
        needed |= run_case(seed, provider)
    # An input a summary cannot hold is the only reason to defer to the reference; none of these sessions is
    # near a cap, so nothing may be deferred.
    assert not needed, needed
