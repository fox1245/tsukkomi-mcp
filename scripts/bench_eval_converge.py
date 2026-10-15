"""Does an audit of a big session converge? Repeated calls under a per-call budget, on synthetic Codex sessions. Nothing real is read.

``bench_slicing.py`` measures the SYNC of a session that does not fit its request budget. This measures what follows it: the
JUDGING. A hook (8 seconds) or a tool (45 seconds) asks for an audit; the session is too big to be synced and judged in one call;
the call is repeated. For each combination of session size, contract set, budget and starting point the script calls
``audit_session`` until the answer is complete and reports

  K          calls it took (a call that the deadline cut, ``request_deadline``, counts and is marked ``x``)
  calls      the elapsed seconds of every call; every one must stay inside the budget (plus 0.25 s: a stop is noticed at the
             next cooperative check)
  answer     whether the final answer equals the unbounded one-shot answer (verdict, findings with their evidence, coverage)

Starting points:
  cold       nothing is indexed
  synced     the whole session is indexed, in one commit, and no monitor state exists: the state right after a build with a
             new evaluator version is deployed (no commit boundary inside the session to save a state at)
  cold+idle, synced+idle
             the same in a resident server: the idle comparison (``shadow_worker``) is attached and gets its turn, as if the
             engine had been quiet for a while, after every call that did not complete. It is the engine's own worker, run
             on the spot (``tick``), and its time is not part of any call.

Contract sets (all of them need history, so judging a slice costs something):
  2          two prohibitions on tool calls
  6          two prohibitions, an order rule with a required success, a required success, and one prohibition and one obligation
             on the assistant's messages
  18         the six with heavier patterns, three times over: a user with many rules (judging an event costs about as much as
             parsing and committing it)

    python scripts/bench_eval_converge.py [--events 150000,300000] [--sets 2,6,18] [--budgets 8,45]
                                          [--scenarios cold,synced] [--limit 14] [--detail] [--json FILE] [--dir DIR]

The cost model of a fresh server process is reset before every run (``eval_cost.COSTS``, when this checkout has it), so each run
is the first request of a freshly started server. The script also runs on a checkout without it, which is how the "before" numbers
of a change are taken. Everything lives in one temporary directory that is deleted at the end (give --dir to choose its parent);
the per-user TOML configuration is replaced by an empty file. Results are printed, never written into the repository.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SID = "11111111-2222-3333-4444-555555555555"
TOLERANCE = 0.25        # seconds a call may run past its budget: the cooperative checks of a loop are 25 ms apart at worst
WORDS = ("alpha beta gamma delta epsilon zeta theta iota kappa lambda sigma omega build test deploy review lint format "
         "merge plan note check retry cache index stream record slice commit").split()
SPECIAL = ["pytest -q", "make build", "deploy now", "git status", "ls -la"]

TWO = [
    {"id": "no-wipe", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf", "severity": "high"},
    {"id": "no-pipe-sh", "type": "must_not", "scope": "tool_call", "regex": r"curl\s.*\|\s*sh", "severity": "high"},
]
SIX = TWO + [
    {"id": "test-before-deploy", "type": "must", "scope": "tool_call", "regex": r"pytest", "before_regex": r"deploy now",
     "requires_success": True},
    {"id": "build-passed", "type": "must", "scope": "tool_call", "regex": r"make build", "requires_success": True},
    {"id": "no-oops", "type": "must_not", "scope": "message", "regex": r"OOPS-FORBIDDEN", "roles": ["assistant"]},
    {"id": "ack-finished", "type": "must", "scope": "message", "regex": r"FINISHED", "roles": ["assistant"]},
]
HEAVY_SIX = [
    {"id": "no-secrets", "type": "must_not", "scope": "any", "roles": ["assistant", "tool", "user"],
     "regex": r"(?i)\b(?:password|passwd|secret|token|api[_-]?key|private[_-]?key)\s*[:=]\s*\S{8,}"},
    {"id": "no-wipe", "type": "must_not", "scope": "tool_call",
     "regex": r"(?i)\b(?:rm|del|rmdir|unlink|shred)\b\s+(?:-[a-z]+\s+)*(?:/|~|\*|\.\.)"},
    {"id": "test-before-deploy", "type": "must", "scope": "tool_call", "regex": r"(?i)\b(?:pytest|unittest|nosetests|tox)\b",
     "before_regex": r"(?i)\b(?:deploy|release|publish|ship)\b", "requires_success": True},
    {"id": "build-passed", "type": "must", "scope": "tool_call",
     "regex": r"(?i)(?:make|npm|cargo|gradle)\s+(?:run\s+)?(?:build|compile|package)", "requires_success": True},
    {"id": "no-refusal-drop", "type": "must_not", "scope": "message", "roles": ["assistant"],
     "regex": r"(?i)\b(?:as an ai|i cannot|i'm sorry|unable to comply)\b.{0,80}\b(?:delete|drop|destroy)\b"},
    {"id": "ack-finished", "type": "must", "scope": "message", "roles": ["assistant"],
     "regex": r"(?i)\b(?:finished|completed?)\b\W*$"},
]


def eighteen():
    out = []
    for copy in range(3):
        for rule in HEAVY_SIX:
            item = dict(rule, id=f"{rule['id']}-v{copy}")
            if copy:                                         # another pattern text (so the compiled patterns differ), same cost
                for key in ("regex", "before_regex"):
                    if key in item:
                        item[key] = item[key] + f"(?#v{copy})"
            out.append(item)
    return out


SETS = {"2": TWO, "6": SIX, "18": eighteen()}


def build_session(path: Path, events: int, seed: int = 7) -> int:
    """A Codex rollout of ``events`` events: messages, shell calls with their outputs, agent messages; ``rm -rf`` now and then,
    a few ``pytest``, ``make build`` and ``deploy now`` calls, and a final ``FINISHED`` from the assistant."""
    rng = random.Random(seed)

    def say(low, high):
        return " ".join(rng.choice(WORDS) for _ in range(rng.randint(low, high)))

    lines, produced, serial = [json.dumps({"type": "session_meta", "payload": {"id": SID}})], 1, 0
    while produced < events:
        pick = rng.random()
        if pick < 0.30:
            lines.append(json.dumps({"type": "response_item", "payload": {
                "type": "message", "role": rng.choice(["user", "assistant"]), "content": say(14, 34)}}))
            produced += 1
        elif pick < 0.76:
            serial += 1
            roll = rng.random()
            command = (rng.choice(SPECIAL) if roll < 0.04 else "rm -rf /tmp/scratch-build" if roll < 0.0402
                       else "echo " + say(3, 10))
            lines.append(json.dumps({"type": "response_item", "payload": {
                "type": "function_call", "name": "shell", "call_id": f"call-{serial}", "arguments": {"command": command}}}))
            lines.append(json.dumps({"type": "response_item", "payload": {
                "type": "function_call_output", "call_id": f"call-{serial}",
                "output": json.dumps({"exit_code": 0 if rng.random() < 0.9 else 1, "output": say(16, 44)})}}))
            produced += 2
        else:
            lines.append(json.dumps({"type": "event_msg", "payload": {"type": "agent_message", "message": say(10, 30)}}))
            produced += 1
    lines.append(json.dumps({"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                                                  "content": "FINISHED"}}))
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ("\n".join(lines) + "\n").encode()
    path.write_bytes(data)
    return len(data)


def brief(payload):
    """What a complete answer must agree on whatever path led to it."""
    return {"verdict": payload["verdict"],
            "findings": [(f["contract_id"], f["verdict"], f["reason"], [e["chunk_id"] for e in f.get("evidence", [])])
                         for f in payload["findings"]],
            "complete": payload["coverage"]["complete"], "issues": list(payload["coverage"]["issues"])}


def make_engine(base: Path, name: str, root: Path, rules, **settings):
    from self_directing_mcp.config import Settings
    from self_directing_mcp.engine import SelfDirectEngine
    options = dict(_env_file=None, local_only=True, openrouter_api_key_file=None, codex_sessions_dir=root,
                   index_dir=base / f"index-{name}")
    options.update(settings)
    engine = SelfDirectEngine(Settings(**options))
    engine.upsert_contracts([dict(rule, provider="codex") for rule in rules])
    return engine


def fresh_server():
    """The cost model of a freshly started server (a checkout without one has nothing to reset)."""
    try:
        from self_directing_mcp.eval_cost import COSTS
    except ImportError:
        return
    COSTS.reset()


def one_shot(base, root, path, rules):
    """The unbounded answer: the full evaluator over a session synced with no deadline."""
    engine = make_engine(base, "one-shot", root, rules, audit_incremental=False)
    engine.sync_session(session_id=SID, provider="codex", embed=False, derived=False)
    answer = engine.audit_session(SID, provider="codex", path=str(path))
    engine.close()
    shutil.rmtree(base / "index-one-shot", ignore_errors=True)
    return answer


def converge(base, root, path, rules, budget, scenario, expected, limit, name):
    engine = make_engine(base, name, root, rules)
    fresh_server()
    start, _, resident = scenario.partition("+")
    worker = None
    if resident:
        from self_directing_mcp.shadow_worker import ShadowWorker
        worker = ShadowWorker(engine, idle_sec=0.0, poll_sec=0.01)
        engine.attach_shadow_worker(worker)
    if start == "synced":
        engine.sync_session(session_id=SID, provider="codex", embed=False, derived=False)
    calls, last, idle_jobs = [], None, 0
    for _ in range(limit):
        began = time.perf_counter()
        try:
            last = engine.audit_session(SID, provider="codex", path=str(path), _deadline=time.monotonic() + budget)
            outcome = "ok"
        except Exception as exc:                                          # noqa: BLE001 - reported, not hidden
            if type(exc).__name__ != "RequestStopped":
                raise
            last, outcome = None, "x"
        elapsed = time.perf_counter() - began
        coverage = last["coverage"] if last else None
        calls.append({"elapsed": round(elapsed, 2), "outcome": outcome,
                      "complete": bool(coverage and coverage["complete"] and not coverage["issues"]),
                      "byte_offset": coverage.get("byte_offset") if coverage else None,
                      "sync_limited_by": ((last.get("sync") or {}).get("cost") or {}).get("limited_by") if last else None,
                      "shadow_skipped": (last.get("incremental") or {}).get("shadow_skipped") if last else None})
        if calls[-1]["complete"]:
            break
        if worker is not None:
            idle_jobs += worker.tick()                                       # a quiet moment: the idle comparison runs
    engine.close()
    shutil.rmtree(base / f"index-{name}", ignore_errors=True)
    done = bool(calls) and calls[-1]["complete"]
    return {"scenario": scenario, "budget": budget, "calls": calls, "k": len(calls), "converged": done, "idle_jobs": idle_jobs,
            "lost": sum(1 for c in calls if c["outcome"] == "x"),
            "max_call": max(c["elapsed"] for c in calls), "total": round(sum(c["elapsed"] for c in calls), 1),
            "inside_budget": all(c["elapsed"] <= budget + TOLERANCE for c in calls),
            "answer_equals_one_shot": bool(done and last is not None and brief(last) == brief(expected))}


def table(rows, detail):
    print("\n| events | contracts | budget | start | K | lost | per-call seconds | max | inside budget | = one-shot |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        seconds = " ".join(f"{c['elapsed']:g}{'x' if c['outcome'] == 'x' else ''}" for c in row["calls"])
        k = str(row["k"]) if row["converged"] else f"{row['k']}+ (no answer)"
        if row["idle_jobs"]:
            k += f" (+{row['idle_jobs']} idle)"
        print(f"| {row['events']:,} | {row['contracts']} | {row['budget']:g} s | {row['scenario']} | {k} | {row['lost']} | "
              f"{seconds} | {row['max_call']:g} | {'yes' if row['inside_budget'] else 'NO'} | "
              f"{'yes' if row['answer_equals_one_shot'] else 'no'} |")
    if detail:
        for row in rows:
            print(f"\n{row['events']:,} events, {row['contracts']} contracts, {row['budget']:g} s, {row['scenario']}:")
            for number, call in enumerate(row["calls"], 1):
                print(f"  call {number}: {call['elapsed']:g} s {call['outcome']} complete={call['complete']} "
                      f"offset={call['byte_offset']} limited_by={call['sync_limited_by']} skipped={call['shadow_skipped']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events", default="150000,300000")
    parser.add_argument("--sets", default="2,6,18")
    parser.add_argument("--budgets", default="8,45")
    parser.add_argument("--scenarios", default="cold,synced", help="cold, synced, cold+idle, synced+idle")
    parser.add_argument("--limit", type=int, default=14, help="stop after this many calls without a complete answer")
    parser.add_argument("--detail", action="store_true", help="print every call")
    parser.add_argument("--json", help="also write the results to this file (outside the repository)")
    parser.add_argument("--dir", help="parent of the temporary directory")
    args = parser.parse_args()
    base = Path(tempfile.mkdtemp(prefix="bench-eval-", dir=args.dir))
    config = base / "config.toml"
    config.write_text("", encoding="utf-8")
    os.environ["SELF_DIRECT_CONFIG_FILE"] = str(config)
    os.environ["SELF_DIRECT_DENSE_BACKEND"] = "numpy"
    os.environ["HOME"] = os.environ["USERPROFILE"] = str(base / "home")
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    results = {"python": sys.version.split()[0], "cpus": os.cpu_count(), "load_1m": load, "runs": []}
    try:
        for events in [int(x) for x in args.events.split(",")]:
            root = base / f"sessions-{events}"
            path = root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl"
            size = build_session(path, events)
            print(f"\n## {events:,} events, {size / 1e6:.1f} MB")
            rows = []
            for name in args.sets.split(","):
                rules = SETS[name]
                expected = one_shot(base, root, path, rules)
                for budget in [float(x) for x in args.budgets.split(",")]:
                    for scenario in args.scenarios.split(","):
                        row = converge(base, root, path, rules, budget, scenario, expected, args.limit,
                                       f"{name}-{int(budget)}-{scenario}")
                        row.update(events=events, contracts=len(rules))
                        rows.append(row)
            table(rows, args.detail)
            results["runs"].extend(rows)
            shutil.rmtree(root, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
