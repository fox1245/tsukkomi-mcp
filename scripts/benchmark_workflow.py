"""Measure the compiled policy against an independent finite reference model.

This is an adversarial class-input evaluation, NOT a live JEV accuracy benchmark.
No credentials or project contents are read or sent to a remote service.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import statistics
import tempfile
import time

from self_directing_mcp.workflow_lean import FACTS, STAGES, LeanPolicy


def expected(stage, choice, facts):
    current = "verify" if stage == "complete" and not all(facts.values()) else stage
    if choice == 0:
        return False, current
    prerequisites = facts["approved"] and facts["obligations"]
    if choice == 1:
        return (True, "implement" if facts["proofs"] else "formalize") if prerequisites else (False, current)
    if choice == 3:
        return (True, "complete") if current == "verify" and all(facts.values()) else (False, current)
    if current == "requirements":
        return (True, "formalize") if prerequisites else (False, current)
    if current in ("formalize", "implement", "verify") and prerequisites and facts["proofs"] and facts["fresh"]:
        return True, {"formalize": "implement", "implement": "verify", "verify": "verify"}[current]
    return False, current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean", default=os.environ.get("TSUKKOMI_LEAN"))
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="workflow-evaluation-") as temp:
        policy = LeanPolicy(args.cache or Path(temp), lean=args.lean)
        build = policy.build()
        timings = {stage: [] for stage in STAGES}
        positive = negative = false_block = false_allow = mismatched_state = 0
        for stage, choice, values in itertools.product(STAGES, range(4), itertools.product((False, True), repeat=5)):
            facts = dict(zip(FACTS, values, strict=True))
            allow, next_stage = expected(stage, choice, facts)
            started = time.perf_counter()
            actual = policy.evaluate(stage, choice, facts)
            timings[stage].append((time.perf_counter() - started) * 1000)
            if "lean_policy_error" in actual["failed"]:
                raise RuntimeError(actual.get("diagnostics", "native checker unavailable"))
            positive += int(allow)
            negative += int(not allow)
            false_block += int(allow and not actual["allowed"])
            false_allow += int(not allow and actual["allowed"])
            mismatched_state += int(next_stage != actual["next_stage"])
        repaired = 0
        for field in FACTS:
            facts = dict.fromkeys(FACTS, True)
            blocked = policy.evaluate("verify", 3, {**facts, field: False})
            unlocked = policy.evaluate(blocked["next_stage"], 3, facts)
            repaired += int(not blocked["allowed"] and unlocked["allowed"] and unlocked["next_stage"] == "complete")
        report = {
            "evaluation": "synthetic finite policy; arbitrary classes, not live JEV",
            "cases": positive + negative, "valid_requests": positive, "invalid_requests": negative,
            "false_blocks": false_block, "false_allows": false_allow,
            "false_block_rate": false_block / positive, "false_allow_rate": false_allow / negative,
            "state_mismatches": mismatched_state, "recovery_successes": repaired, "recovery_cases": len(FACTS),
            "recovery_success_rate": repaired / len(FACTS), "policy_hash": build["policy_hash"],
            "stage_latency_ms": {stage: {"median": round(statistics.median(samples), 3),
                                         "max": round(max(samples), 3)} for stage, samples in timings.items()},
        }
        print(json.dumps(report, sort_keys=True, indent=2))
        if false_block or false_allow or mismatched_state or repaired != len(FACTS):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
