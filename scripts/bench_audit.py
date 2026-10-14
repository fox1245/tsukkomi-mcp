"""Incremental audit benchmark on a large synthetic Codex session (all data is generated here, nothing is read from
a real session or index).

    python scripts/bench_audit.py --work-dir /absolute/scratch --events 225000        # about 300 MB
    python scripts/bench_audit.py --work-dir ... --baseline-src /path/to/older/src     # also measure an older tree

The session mixes assistant narration, tool calls and tool results. Each measurement runs in a fresh child process
with ``PYTHONPATH`` pointing at the tree under test, so imports never mix. For every tree it reports, for
``audit_session`` and ``check_action``:

* ``unchanged``: nothing appended since the previous call (median of several calls);
* ``+100`` / ``+1000``: after appending that many records to the file.

Wall and CPU seconds are for the whole call, ``sync`` is the engine's own source verification and ``evaluation`` the
rest (reads, evaluation, state). ``evaluation_cpu`` is CPU spent after the sync phase. A tree without
``audit_incremental`` is measured as it is; the tree under test is also measured with ``audit_incremental=false``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

SID = "11111111-2222-3333-4444-555555555555"
FILE = f"rollout-2026-09-09T01-00-00-{SID}.jsonl"
CONTRACTS = [
    {"id": "no-recursive-delete", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf\s+/", "roles": ["assistant"]},
    {"id": "tests-before-deploy", "type": "must", "scope": "tool_call", "regex": "npm test", "before_regex": "deploy now",
     "requires_success": True},
    {"id": "linted", "type": "must", "scope": "tool_call", "regex": "eslint"},
    {"id": "no-leaked-output", "type": "must_not", "scope": "tool_result", "regex": "LEAKED_SECRET", "roles": ["tool"]},
    {"id": "summarized", "type": "must", "scope": "message", "regex": "summary", "roles": ["assistant"]},
    {"id": "release-after-publish", "type": "must", "scope": "tool_call", "regex": "release",
     "when": {"tool_call_regex": "npm publish"}},
]
DEPLOY = {"tool_name": "shell", "arguments": {"command": "deploy now"}}
FILLER = ("synthetic harmless transcript text for benchmarking the audit; " * 60)


def records(first, count):
    """``count`` synthetic records numbered from ``first``: narration, a tool call, its result, repeating."""
    for index in range(first, first + count):
        slot = index % 3
        if slot == 0:
            yield {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                                        "content": f"{index} narration {FILLER[:1100]}" + (
                                                            " summary" if index % 997 == 0 else "")}}
        elif slot == 1:
            command = ("npm test" if index % 50 == 1 else "eslint src" if index % 211 == 1 else f"echo step {index}")
            yield {"type": "response_item", "payload": {"type": "function_call", "name": "shell",
                                                        "call_id": f"call_{index}", "arguments": {"command": command}}}
        else:
            yield {"type": "response_item", "payload": {"type": "function_call_output", "call_id": f"call_{index - 1}",
                                                        "output": json.dumps({"exit_code": 0,
                                                                              "output": f"{index} {FILLER[:2400]}"})}}


def generate(directory, events):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FILE
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"type": "session_meta", "payload": {"id": SID}}) + "\n")
        for record in records(0, events):
            stream.write(json.dumps(record) + "\n")
    return path


def append(path, first, count):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for record in records(first, count):
            stream.write(json.dumps(record) + "\n")


def child(args):
    """Measure one tree. Prints one JSON document."""
    from self_directing_mcp.config import Settings
    from self_directing_mcp.engine import SelfDirectEngine
    sessions = Path(args.work_dir) / "sessions"
    settings = dict(codex_sessions_dir=sessions, index_dir=Path(args.index), local_only=True, _env_file=None,
                    audit_incremental=bool(args.incremental), audit_shadow_every=args.shadow,
                    seed_example_contracts=False)
    engine = SelfDirectEngine(Settings(**settings))
    path = sessions / FILE
    engine.upsert_contracts(CONTRACTS)

    def timed(method, *call):
        wall, cpu = time.perf_counter(), time.process_time()
        payload = getattr(engine, method)(*call)
        wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
        timings = payload.get("timings_ms", {})
        shape = json.dumps([payload["verdict"], payload["findings"], payload["notes"]], sort_keys=True, default=str)
        return {"wall": wall, "cpu": cpu, "sync_ms": timings.get("sync"), "evaluation_ms": timings.get("evaluation"),
                "verdict": payload["verdict"], "incremental": (payload.get("incremental") or {}).get("mode"),
                "answer": hashlib.sha256(shape.encode()).hexdigest()[:12]}

    def run(method, *call):
        return lambda: timed(method, *call)

    started = time.perf_counter()
    first = timed("audit_session", SID)
    report = {"cold_first_audit": first, "cold_seconds": round(time.perf_counter() - started, 2), "calls": {}}
    if args.index_only:
        print(json.dumps(report), flush=True)
        engine.close()
        return
    total = args.events
    for label, appended in (("unchanged", 0), ("+100", 100), ("+1000", 1000)):
        for method, call in (("audit_session", (SID,)), ("check_action", (SID, DEPLOY))):
            samples = []
            for repeat in range(args.repeats):
                if appended:
                    append(path, total, appended)
                    total += appended
                samples.append(run(method, *call)())
            ordered = sorted(samples, key=lambda s: s["wall"])
            median = ordered[len(ordered) // 2]
            report["calls"][f"{method}/{label}"] = {
                "wall_s": round(median["wall"], 4), "cpu_s": round(median["cpu"], 4),
                "sync_ms": median["sync_ms"], "evaluation_ms": median["evaluation_ms"],
                "evaluation_cpu_s_est": round(max(0.0, median["cpu"] - (median["sync_ms"] or 0) / 1000.0), 4),
                "verdict": median["verdict"], "mode": median["incremental"],
                "answers": sorted({sample["answer"] for sample in samples})}
    report["index_mb"] = round(sum(f.stat().st_size for f in Path(args.index).iterdir() if f.is_file()) / 1e6, 1)
    print(json.dumps(report), flush=True)
    engine.close()


def measure(label, src, args, index, incremental, shadow=0, index_only=False):
    env = dict(os.environ, PYTHONPATH=str(src), SELF_DIRECT_CONFIG_FILE=str(Path(args.work_dir) / "empty.toml"),
               OPENBLAS_NUM_THREADS="1")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", "--work-dir", args.work_dir, "--index", str(index),
               "--events", str(args.events), "--incremental", str(incremental), "--shadow", str(shadow),
               "--repeats", str(args.repeats), *(["--index-only"] if index_only else [])]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=3600)
    if result.returncode:
        raise SystemExit(f"{label} failed:\n{result.stderr[-2000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", required=True, help="absolute scratch directory (sessions and indexes are written here)")
    parser.add_argument("--events", type=int, default=225_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baseline-src", help="src directory of an older tree to measure as the 'before'")
    parser.add_argument("--only", help="comma separated run labels to measure (default: all)")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--index-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--index")
    parser.add_argument("--incremental", type=int, default=1)
    parser.add_argument("--shadow", type=int, default=0)
    args = parser.parse_args()
    if args.child:
        return child(args)
    work = Path(args.work_dir)
    if not work.is_absolute():
        raise SystemExit("--work-dir must be absolute")
    work.mkdir(parents=True, exist_ok=True)
    (work / "empty.toml").write_text("")
    sessions = work / "sessions"
    session = sessions / FILE
    if not session.exists() or session.stat().st_size < args.events * 900:
        print(f"generating {args.events} synthetic records ...", flush=True)
        generate(sessions, args.events)
    print(f"session: {session.stat().st_size / 1e6:.0f} MB, {args.events + 1} records", flush=True)
    here = Path(__file__).resolve().parents[1] / "src"
    out = {"session_mb": round(session.stat().st_size / 1e6), "records": args.events + 1}
    # Warm one index, then copy it so every tree starts from the same stored events.
    base_index = work / "index-warm"
    if not base_index.exists():
        print("building the index ...", flush=True)
        measure("warmup", here, args, base_index, 1, index_only=True)
    import shutil
    sys.path.insert(0, str(here))
    from self_directing_mcp.journal import copy_index_dir
    runs = [("incremental", here, 1, 0)]
    if args.baseline_src:
        runs.append(("baseline-tree", Path(args.baseline_src), 1, 0))
    runs.append(("same-tree-incremental-off", here, 0, 0))
    snapshot = work / "session-snapshot.jsonl"  # the file as the warm index knows it
    if not snapshot.exists():
        shutil.copy(session, snapshot)
    for label, src, incremental, shadow in runs:
        if args.only and label not in args.only.split(","):
            continue
        index = work / f"index-{label}"
        if index.exists():
            shutil.rmtree(index)
        copy_index_dir(base_index, index)   # databases through the backup API: a plain copy would lose what is in a -wal
        shutil.copy(snapshot, session)
        print(f"measuring {label} ...", flush=True)
        out[label] = measure(label, src, args, index, incremental, shadow)
    print(json.dumps(out, indent=2))
    labels = [label for label in out if isinstance(out[label], dict) and "calls" in out[label]]
    for key in out[labels[0]]["calls"] if labels else ():
        answers = {label: out[label]["calls"][key]["answers"] for label in labels}
        # Appends differ between trees only in timing, never in content, so every tree must give the same answers.
        if len({json.dumps(v) for v in answers.values()}) > 1:
            print(f"WARNING: trees disagree on {key}: {answers}", file=sys.stderr)


if __name__ == "__main__":
    main()
