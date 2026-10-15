#!/usr/bin/env python3
"""Fault injection for the shadow comparison: how many requests pass before a defect of the incremental path is noticed.

A defect is injected into the incremental path of an engine that audits a growing synthetic session, and the requests are
counted until a comparison with the full evaluator finds it (``incremental.incremental_mismatch``). Everything is synthetic
(a temporary index and session per trial, a fixed list of contracts); nothing reads a user's sessions or settings.

Scenarios (each trial is one seed: another session id, other event texts, hence other hash draws):

* ``missed``: a monitor silently ignores forbidden events, so the answer is ``clean`` where the full evaluator says
  ``violation``. The forbidden call is the request ``--violation-at`` (0-based). Nothing but the sampled comparison can
  notice this in the first request that carries it; ``requests`` counts from that request (1 = noticed in it), ``wrong``
  is how many wrong answers were returned before it was noticed.
* ``wrong_block``: a monitor reports ``violation`` where the full evaluator says ``clean`` (the reverse defect).
* ``flip``: a violation is reported correctly, then the stored monitor state is corrupted so that the next request says
  ``clean`` for a contract the previous request flagged.
* ``high_risk``: a ``risk: high`` obligation (tests before deploy) whose monitor wrongly says ``clean`` when the proposed
  action is a deploy.

``--every`` is the schedule: ``adaptive`` (the new default), or an integer (``16`` is the previous fixed rate). With
``--floor`` the adaptive schedule starts at its floor (1/32) as after a long clean streak, by writing that state. Use
``--src`` to run the same script against another checkout (the previous fixed schedule lives in the previous code).

    python scripts/measure_shadow_detection.py --scenario missed --every adaptive [--floor] [--seeds 300] [--out r.json]

``--simulate`` skips the engine and plays the schedule alone (the policy functions of ``audit/shadow.py`` with the same
hash draws, and the previous fixed draw for an integer ``--every``) for the ``missed`` scenario, which is what makes
thousands of seeds affordable; ``--share`` prints how large a share of a clean session's requests the schedule (``--every``)
compares after 10, 100, 1,000 ... requests; ``--session`` prints, for sessions of 100 to 3,000 requests, the comparisons a
clean session makes and the mean number of requests a missed violation stays unnoticed in it. ``--constants
4,8,16,32,64/32/8`` (denominators / halve after / warm-up) replays them with other constants, for instance the roundtable's
original ones, which the chair replaced by 4,8,16,32/8/4 (issue #18 R2).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random
import shutil
import statistics
import sys
import tempfile
import uuid

HERE = Path(__file__).resolve().parent

CONTRACTS = [
    {"id": "no-deploy", "type": "must_not", "scope": "tool_call", "regex": "deploy", "roles": ["assistant"]},
    {"id": "no-wipe", "type": "must_not", "scope": "tool_call", "regex": "rm -rf /", "roles": ["assistant"]},
    {"id": "no-secret", "type": "must_not", "scope": "tool_result", "regex": "BEGIN PRIVATE", "roles": ["tool"]},
    {"id": "tests-first", "type": "must", "scope": "tool_call", "regex": "pytest", "before_regex": "deploy",
     "requires_success": True},
    {"id": "said-done", "type": "must", "scope": "message", "regex": "all done", "roles": ["assistant"]},
]
HIGH_RISK = {**CONTRACTS[3], "id": "tests-first", "risk": "high"}
WORDS = ["alpha", "beta", "gamma", "build", "lint", "status", "show", "list", "echo", "fetch"]


def message(text, role="assistant"):
    return {"type": "response_item", "payload": {"type": "message", "role": role, "content": text}}


def call(command, call_id):
    return {"type": "response_item", "payload": {"type": "function_call", "name": "shell", "call_id": call_id,
                                                 "arguments": {"command": command}}}


def output(call_id):
    return {"type": "response_item", "payload": {"type": "function_call_output", "call_id": call_id,
                                                 "output": json.dumps({"exit_code": 0, "output": "fine"})}}


def write(path, events, mode="a"):
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        for event in events:
            stream.write(json.dumps(event) + "\n")


class Trial:
    def __init__(self, scratch: Path, seed: str, every, contracts):
        from self_directing_mcp.config import Settings
        from self_directing_mcp.engine import SelfDirectEngine
        self.rng = random.Random(seed)
        self.sid = str(uuid.UUID(int=self.rng.getrandbits(128), version=4))
        root = scratch / "sessions"
        root.mkdir()
        self.path = root / f"rollout-2026-09-09T01-00-00-{self.sid}.jsonl"
        write(self.path, [{"type": "session_meta", "payload": {"id": self.sid}}], "w")
        self.engine = SelfDirectEngine(Settings(
            _env_file=None, local_only=True, openrouter_api_key_file=None, codex_sessions_dir=root,
            index_dir=scratch / "index", audit_shadow_every=every, audit_incremental_min_events=0,
            audit_shadow_idle_sec=0))
        self.engine.upsert_contracts(contracts)
        self.serial = 0
        self.append(self.rng.randint(20, 30))

    def words(self):
        return " ".join(self.rng.choice(WORDS) for _ in range(self.rng.randint(1, 3)))

    def append(self, count=1, extra=()):
        records = []
        for _ in range(count):
            self.serial += 1
            if self.rng.random() < 0.5:
                records += [call(self.words(), f"c{self.serial}"), output(f"c{self.serial}")]
            else:
                records.append(message(self.words()))
        write(self.path, [*records, *extra])

    def audit(self):
        return self.engine.audit_session(self.sid, provider="codex")

    def check(self, command):
        return self.engine.check_action(self.sid, {"tool_name": "shell", "arguments": {"command": command}},
                                        provider="codex")

    def seed_floor(self):
        """Start the adaptive schedule at its floor, as after a long clean streak (needs the new code)."""
        from self_directing_mcp.audit import shadow
        state = self.engine.store.load_shadow_state(self.sid, "codex")
        floor = len(shadow.RATE_DENOMINATORS) - 1
        state.rate_index, state.warmup = floor, 0
        state.streak, state.compared = self.rng.randrange(shadow.HALVE_AFTER), self.rng.randrange(10_000)
        self.engine.store.save_shadow_state(self.sid, "codex", state)
        # a row the schedule cannot read would silently start the trial at 1/4 instead of at the floor
        assert self.engine.store.load_shadow_state(self.sid, "codex").rate_index == floor

    def close(self):
        self.engine.close()


def verdict_of(response, contract_id):
    return next(item["verdict"] for item in response["findings"] if item["contract_id"] == contract_id)


def noticed(response):
    return bool((response.get("incremental") or {}).get("incremental_mismatch"))


def patch_missed():
    from self_directing_mcp.audit.incremental import RuleMonitor
    real = RuleMonitor._forbidden

    def broken(self, event):
        if self.rule.id != "no-deploy":
            real(self, event)
    RuleMonitor._forbidden = broken
    return lambda: setattr(RuleMonitor, "_forbidden", real)


ARMED = {"on": False}      # the injected defect only acts while this is set (the scenario arms it when it is due)


def patch_finding(target, actual, reported):
    """Make the monitor of ``target`` report ``reported`` when it computed ``actual`` (nothing else changes)."""
    from self_directing_mcp.audit.incremental import RuleMonitor
    real = RuleMonitor.finding

    def faulty(self, proposed, facts):
        item = real(self, proposed, facts)
        if ARMED["on"] and self.rule.id == target and item.verdict == actual:
            return item.model_copy(update={"verdict": reported, "reason": "injected", "evidence": []})
        return item
    RuleMonitor.finding = faulty
    return lambda: setattr(RuleMonitor, "finding", real)


def corrupt_stored(engine, monitor_id, **fields):
    engine.flush_derived()
    connection = engine.store._conn
    with connection:
        for row_id, state in connection.execute("SELECT rowid, state FROM audit_monitor_state WHERE monitor_id=?",
                                                (monitor_id,)).fetchall():
            data = json.loads(state)
            data.update(fields)
            connection.execute("UPDATE audit_monitor_state SET state=? WHERE rowid=?", (json.dumps(data), row_id))


def run_trial(scenario, seed, *, every, floor, violation_at, cap, tmp):
    scratch = Path(tempfile.mkdtemp(prefix="detect-", dir=tmp))
    restore = None
    trial = None
    try:
        contracts = [*CONTRACTS[:3], HIGH_RISK, CONTRACTS[4]] if scenario == "high_risk" else CONTRACTS
        trial = Trial(scratch, f"{scenario}/{seed}", every, contracts)
        if scenario == "missed":
            restore = patch_missed()
        elif scenario == "wrong_block":
            restore = patch_finding("no-deploy", "clean", "violation")
        elif scenario == "high_risk":
            restore = patch_finding("tests-first", "violation", "clean")
        wrong = 0
        if floor:
            trial.append(1)
            trial.audit()
            trial.seed_floor()
        if scenario == "missed":
            # The forbidden call arrives at request ``violation_at``; every earlier request is clean and correct.
            for index in range(cap):
                forbidden = [call("deploy now", f"bad{index}")] if index == violation_at else []
                trial.append(trial.rng.randint(1, 3), forbidden)
                response = trial.audit()
                if index < violation_at:
                    continue
                if noticed(response):
                    return {"requests": index - violation_at + 1, "wrong": wrong}
                wrong += 1
            return {"requests": None, "wrong": wrong}
        if scenario == "wrong_block":
            for _ in range(violation_at):       # correct requests first, so that the schedule is past its first requests
                trial.append(trial.rng.randint(1, 3))
                trial.audit()
            ARMED["on"] = True
            for index in range(cap):
                trial.append(trial.rng.randint(1, 3))
                response = trial.audit()
                if noticed(response):
                    return {"requests": index + 1, "wrong": wrong}
                wrong += 1          # a block that the full evaluator would not give
            return {"requests": None, "wrong": wrong}
        if scenario == "flip":
            trial.append(1, [call("deploy now", "bad0")])
            first = trial.audit()
            assert verdict_of(first, "no-deploy") == "violation"
            corrupt_stored(trial.engine, "no-deploy", hits=0, hit_ev=[], hit_n=0)
            for index in range(cap):
                trial.append(trial.rng.randint(1, 3))
                response = trial.audit()
                if noticed(response):
                    return {"requests": index + 1, "wrong": wrong}
                wrong += 1          # clean for a contract the previous request flagged
            return {"requests": None, "wrong": wrong}
        if scenario == "high_risk":
            for _ in range(violation_at):       # unrelated action checks: the high-risk contract does not apply
                trial.append(1)
                trial.check(trial.words())
            ARMED["on"] = True
            for index in range(cap):
                trial.append(1)
                response = trial.check("deploy now")
                if noticed(response):
                    return {"requests": index + 1, "wrong": wrong}
                wrong += 1          # "tests-first" clean although no test ran
            return {"requests": None, "wrong": wrong}
        raise SystemExit(f"unknown scenario {scenario}")
    finally:
        ARMED["on"] = False
        if restore:
            restore()
        if trial:
            trial.close()
        shutil.rmtree(scratch, ignore_errors=True)


def compared_requests(seed, every, floor=False, label="simulate"):
    """Endless iterator over the requests of a clean synthetic session: ``True`` when the schedule compares the request.

    ``every`` is ``"adaptive"`` (the policy functions of ``audit/shadow.py`` with their hash draws) or an integer, the
    previous fixed schedule (a rebuild at the first request, then a hash draw per cursor)."""
    import hashlib
    from self_directing_mcp.audit import shadow
    rng = random.Random(f"{label}/{seed}")
    sid = str(uuid.UUID(int=rng.getrandbits(128), version=4))
    state, _ = shadow.reconcile(None, {"contracts": "c", "parser": 3, "evaluator": 8, "generation": 1})
    if floor:
        state.rate_index, state.warmup = len(shadow.RATE_DENOMINATORS) - 1, 0
        state.streak, state.compared = rng.randrange(shadow.HALVE_AFTER), rng.randrange(10_000)
    offset = 0
    for index in itertools.count():
        offset += rng.randint(100, 600)
        if every == "adaptive":
            compared = state.warmup > 0 or shadow.sampled(state, sid, "codex", offset, None)
            if compared:
                state = shadow.settle(state, outcome="match", full_scope=True, key="audit", flagged=set(), kept=set(),
                                      at="t", offset=offset)
        else:
            compared = index == 0 or int(hashlib.sha256(f"{sid}|{offset}|c:1".encode()).hexdigest()[:8], 16) % every == 0
        yield compared


def simulate_missed(seed, every, floor, violation_at, cap=20_000):
    """The ``missed`` scenario without an engine: requests until a comparison covers the request that carries the violation."""
    for index, compared in enumerate(itertools.islice(compared_requests(seed, every, floor), cap)):
        if compared and index >= violation_at:
            return {"requests": index - violation_at + 1, "wrong": index - violation_at}
    return {"requests": None, "wrong": cap - violation_at}


def share_of_requests(seeds, every="adaptive", checkpoints=(10, 32, 100, 300, 1000, 2000, 3000, 5000)):
    """Mean share (percent) of a clean session's requests that the schedule compared, after each number of requests."""
    import statistics
    runs = []
    for seed in range(seeds):
        seen, compared = {}, 0
        for index, drawn in enumerate(itertools.islice(compared_requests(seed, every, label="share"), max(checkpoints)), 1):
            compared += drawn
            if index in checkpoints:
                seen[index] = compared / index
        runs.append(seen)
    return {str(c): round(100 * statistics.mean(run[c] for run in runs), 1) for c in checkpoints}


def session_cost(seeds, every="adaptive", lengths=(100, 300, 1000, 3000)):
    """Per session length: the mean number of comparisons in a clean session, and the mean number of requests a missed
    violation stays unnoticed (it starts at ten evenly spread times of the session; the session's end caps the wait)."""
    import statistics
    out = {}
    for length in lengths:
        counts, waits = [], []
        for seed in range(seeds):
            drawn = list(itertools.islice(compared_requests(seed, every, label="session"), length))
            counts.append(sum(drawn))
            for k in range(10):
                start = int((k + 0.5) * length / 10)
                after = next((i for i in range(start, length) if drawn[i]), None)
                waits.append(after - start + 1 if after is not None else length - start)
        out[str(length)] = {"comparisons": round(statistics.mean(counts), 1), "unnoticed": round(statistics.mean(waits), 1)}
    return out


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def summarize(results):
    done = [r for r in results if r["requests"] is not None]
    out = {"seeds": len(results), "undetected": len(results) - len(done)}
    if done:
        requests, wrong = [r["requests"] for r in done], [r["wrong"] for r in done]
        out.update(median=statistics.median(requests), p95=percentile(requests, 0.95), max=max(requests),
                   mean=round(statistics.mean(requests), 1), wrong_median=statistics.median(wrong),
                   wrong_p95=percentile(wrong, 0.95), wrong_max=max(wrong),
                   first_request=sum(1 for value in requests if value == 1))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", choices=["missed", "wrong_block", "flip", "high_risk"], default="missed")
    parser.add_argument("--every", default="adaptive", help="adaptive, or an integer (the fixed rate 1/N; 0 never)")
    parser.add_argument("--floor", action="store_true", help="start the adaptive schedule at its floor (1/32)")
    parser.add_argument("--violation-at", type=int, default=9,
                        help="missed: the request carrying the forbidden call (default 9, just after the warm-up); "
                             "high_risk: how many unrelated checks come first")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--cap", type=int, default=3000, help="requests after which a trial is given up as undetected")
    parser.add_argument("--simulate", action="store_true",
                        help="play the schedule alone (scenario missed only), no engine: thousands of seeds are cheap")
    parser.add_argument("--share", action="store_true",
                        help="print the share of a clean session's requests the schedule compares, and stop")
    parser.add_argument("--session", action="store_true",
                        help="print the comparisons per clean session and the mean time a missed violation stays unnoticed")
    parser.add_argument("--constants", help="replace the schedule's constants for this run, as DENOMINATORS/HALVE_AFTER/WARMUP, "
                                            "e.g. 4,8,16,32,64/32/8 (the roundtable's original ones)")
    parser.add_argument("--src", type=Path, default=HERE.parent / "src", help="the src directory to import from")
    parser.add_argument("--tmpdir", help="directory for the temporary indexes (default: the system's)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(args.src))
    every = args.every if args.every == "adaptive" else int(args.every)
    if args.constants:
        from self_directing_mcp.audit import shadow
        denominators, halve, warmup = args.constants.split("/")
        shadow.RATE_DENOMINATORS = tuple(int(item) for item in denominators.split(","))
        shadow.HALVE_AFTER, shadow.WARMUP_REQUESTS = int(halve), int(warmup)
    if args.share:
        print(json.dumps({"every": args.every, "share_percent_by_requests": share_of_requests(args.seeds, every)}))
        return 0
    if args.session:
        print(json.dumps({"every": args.every, "per_session_length": session_cost(args.seeds, every)}))
        return 0
    if args.simulate:
        results = [simulate_missed(seed, every, args.floor, args.violation_at)
                   for seed in range(args.first_seed, args.first_seed + args.seeds)]
        print(json.dumps({"simulated": True, "every": args.every, "floor": args.floor, "violation_at": args.violation_at,
                          **summarize(results)}))
        return 0
    results = [run_trial(args.scenario, seed, every=every, floor=args.floor, violation_at=args.violation_at,
                         cap=args.cap, tmp=args.tmpdir)
               for seed in range(args.first_seed, args.first_seed + args.seeds)]
    summary = summarize(results)
    print(json.dumps({"scenario": args.scenario, "every": args.every, "floor": args.floor,
                      "violation_at": args.violation_at, "src": str(args.src), **summary}))
    if args.out:
        args.out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "summary": summary,
                                        "results": results}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
