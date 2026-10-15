"""Cost of one AGY hook cycle against transcript size (all data is generated here; nothing real is read).

    python scripts/bench_agy_hooks.py --work-dir /absolute/scratch                      # 0.1, 1, 3 and 10 MB
    python scripts/bench_agy_hooks.py --work-dir ... --baseline-src /path/to/older/src  # also an older tree

Each size gets a synthetic display transcript (planner steps with a tool call, GENERIC results) and one linked host
receipt per result, written under the layout the AGY code expects (``brain/<id>/.system_generated/logs``). The index
is warmed once per tree, then each cycle appends a few KB and runs the real hook entry point in a fresh process:

* ``pre``: a planner step is appended, then the PreToolUse hook (host receipt, then the contract check);
* ``post``: the result is appended, then the PostToolUse hook (host receipt, then the history sync).

Wall and CPU seconds are for the whole hook process (interpreter start and imports included; ``floor`` is a hook
process that only imports the package, so the size-dependent part is the difference). After the cycles the child
verifies the index (every result linked to its call by a receipt) so a fast but failing-open hook cannot pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

SID = "33333333-4444-4555-8666-777777777777"
CONTRACTS = [
    {"id": "no-root-delete", "type": "must_not", "scope": "tool_call", "provider": "agy", "regex": r"rm\s+-rf\s+/"},
    {"id": "tests-before-deploy", "type": "must", "scope": "tool_call", "provider": "agy", "regex": "npm test",
     "before_regex": "deploy now", "requires_success": True},
]
WORDS = ("alpha beta gamma delta epsilon zeta theta kappa lambda sigma omega index buffer cursor record parser "
         "stream digest prefix offset layout result status planner receipt anchor session ledger").split()


def _text(rng, chars):
    out, size = [], 0
    while size < chars:
        word = rng.choice(WORDS)
        out.append(word)
        size += len(word) + 1
    return " ".join(out)


def planner(step, n, rng):
    return {"step_index": step, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE",
            "created_at": "2026-10-01T00:00:01Z", "content": f"Step {n}. " + _text(rng, 900),
            "tool_calls": [{"name": "run_command", "args": {
                "CommandLine": json.dumps(f"echo step {n}"), "Cwd": json.dumps("/synthetic/project"),
                "WaitMsBeforeAsync": "5000", "toolAction": json.dumps("Running"), "toolSummary": json.dumps("Echo")}}]}


def result(step, n, rng):
    return {"step_index": step, "source": "MODEL", "type": "GENERIC", "status": "DONE",
            "created_at": "2026-10-01T00:00:02Z",
            "content": "Created At: 2026-10-01T00:00:02Z\nCompleted At: 2026-10-01T00:00:03Z\n\n"
                       f"The command exited with code 0.\nOutput:\nstep {n} " + _text(rng, 1500) + "\n"}


def tool_call(n):
    return {"name": "run_command", "args": {"CommandLine": f"echo step {n}", "Cwd": "/synthetic/project"}}


def line(record):
    return (json.dumps(record, separators=(",", ":")) + "\n").encode()


def transcript_path(root):
    return root / "app" / "brain" / SID / ".system_generated" / "logs" / "transcript.jsonl"


def generate(root, megabytes):
    """The transcript, plus the receipts a host would have recorded for every result (as rows of the receipt table)."""
    from self_directing_mcp.agy_receipts import result_digest
    path = transcript_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    digest, written, rows, step, n = hashlib.sha256(), 0, [], 1, 0
    first = line({"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE",
                  "created_at": "2026-10-01T00:00:00Z", "content": "<USER_REQUEST>Work through the tasks.</USER_REQUEST>"})
    with path.open("wb") as stream:
        stream.write(first)
        digest.update(first)
        written += len(first)
        while written < megabytes * 1_000_000:
            call, outcome = line(planner(step, n, rng)), result(step + 1, n, rng)
            stream.write(call)
            digest.update(call)
            written += len(call)
            action = json.dumps(tool_call(n), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            rows.append((step + 1, json.dumps({
                "receipt_id": str(uuid.UUID(int=rng.getrandbits(128), version=4)), "path": str(path.resolve()),
                "action": action, "prefix_length": written, "prefix_hash": digest.hexdigest(), "paired": True,
                "step_index": step + 1, "session_id": SID, "preexisting_result": False, "post_error": "",
                "post_action": action, "post_path": str(path.resolve()), "result_hash": result_digest(outcome)},
                sort_keys=True)))
            body = line(outcome)
            stream.write(body)
            digest.update(body)
            written += len(body)
            step, n = step + 2, n + 1
    (root / "receipts.json").write_text(json.dumps(rows))
    (root / "meta.json").write_text(json.dumps({"next_step": step, "n": n, "size": written, "receipts": len(rows)}))


# A hook process that starts the engine and does nothing else: the part of a hook that does not depend on the transcript.
FLOOR = ("from self_directing_mcp.config import hook_settings; from self_directing_mcp.engine import SelfDirectEngine; "
         "SelfDirectEngine(hook_settings()).close()")


def settings_file(work, index, app):
    config = work / "config.toml"
    config.write_text(f'[paths]\nindex_dir = "{index}"\nagy_app_data_dirs = ["{app}"]\n')
    return config


def hook_env(src, config):
    env = dict(os.environ, PYTHONPATH=str(src), SELF_DIRECT_CONFIG_FILE=str(config), SELF_DIRECT_LOCAL_ONLY="true",
               OPENBLAS_NUM_THREADS="1")
    for key in [key for key in env if key.startswith("SELF_DIRECT_") and key not in
                ("SELF_DIRECT_CONFIG_FILE", "SELF_DIRECT_LOCAL_ONLY")]:
        del env[key]
    return env


def child_warm(args):
    """Seed the receipts, upsert the contracts and index the transcript once, as a resident server would have."""
    import sqlite3
    from self_directing_mcp.agy_receipts import ReceiptStore
    from self_directing_mcp.config import Settings
    from self_directing_mcp.engine import SelfDirectEngine
    root, index = Path(args.root), Path(args.index)
    app = root / "app"
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=index, agy_app_data_dirs=[app],
                                       seed_example_contracts=False))
    try:
        here = str(transcript_path(root).resolve())  # the receipts name the file they were generated beside
        rows = []
        for step, data in json.loads((root / "receipts.json").read_text()):
            row = json.loads(data)
            row.update(path=here, post_path=here)
            rows.append((SID, step, json.dumps(row, sort_keys=True)))
        with ReceiptStore(index)._connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO receipts VALUES (?,?,?)", rows)
        engine.upsert_contracts(CONTRACTS)
        started = time.perf_counter()
        info = engine.sync_session(session_id=SID, provider="agy", path=str(transcript_path(root)), embed=False)
        print(json.dumps({"seconds": round(time.perf_counter() - started, 2), "ok": info.get("ok"),
                          "chunks": info.get("total_chunks"), "reindex_reason": info.get("reindex_reason")}))
    finally:
        engine.close()


def child_verify(args):
    """How the index stands: results linked to their call by a receipt, and whether a fresh sync is a no-op."""
    from self_directing_mcp.config import Settings
    from self_directing_mcp.engine import SelfDirectEngine
    root, index = Path(args.root), Path(args.index)
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=index, agy_app_data_dirs=[root / "app"],
                                       seed_example_contracts=False))
    try:
        info = engine.sync_session(session_id=SID, provider="agy", path=str(transcript_path(root)), embed=False)
        chunks = engine.store.list_chunks(SID, "agy")
        results = [c for c in chunks if c.kind == "tool_result"]
        linked = [c for c in results if c.meta.get("linkage") == "host_step_receipt"]
        print(json.dumps({"ok": info.get("ok"), "complete": info["coverage"]["complete"], "chunks": len(chunks),
                          "results": len(results), "linked": len(linked), "resync_new_chunks": info["new_chunks"],
                          "resync_reindexed": info["reindexed"]}))
    finally:
        engine.close()


def run_process(command, env, stdin=None):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    done = subprocess.run(command, env=env, input=stdin, capture_output=True, text=True, timeout=1800)
    wall = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    return done, wall, cpu


def hook_command(event):
    code = ("import sys; sys.argv = ['agy_hook', '--event=%s']; "
            "from self_directing_mcp.agy_hooks import main; main()" % event)
    return [sys.executable, "-B", "-c", code]


def self_command(*extra):
    return [sys.executable, "-B", str(Path(__file__).resolve()), *extra]


def measure(label, src, root, work, repeats):
    """One tree at one size: warm an index, then ``repeats`` cycles of append, Pre hook, append, Post hook."""
    index = work / f"index-{label}"
    app = root / "app"
    if index.exists():
        shutil.rmtree(index)
    config = settings_file(work, index, app)
    env = hook_env(src, config)
    warm, _wall, _cpu = run_process(self_command("--child", "warm", "--root", str(root), "--index", str(index)), env)
    if warm.returncode:
        raise SystemExit(f"{label}: warm-up failed\n{warm.stderr[-2000:]}")
    warmed = json.loads(warm.stdout.strip().splitlines()[-1])
    meta = json.loads((root / "meta.json").read_text())
    path, step, n, rng = transcript_path(root), meta["next_step"], meta["n"], random.Random(11)
    floor = []
    for _ in range(3):
        done, wall, cpu = run_process([sys.executable, "-B", "-c", FLOOR], env)
        floor.append((wall, cpu))
    samples = {"pre": [], "post": []}
    problems = []
    for _ in range(repeats):
        with path.open("ab") as stream:
            stream.write(line(planner(step, n, rng)))
        payload = {"conversationId": SID, "transcriptPath": str(path), "workspacePaths": ["/synthetic/project"],
                   "stepIdx": step + 1, "toolCall": tool_call(n)}
        for event, extra in (("PreToolUse", {}), ("PostToolUse", {"error": ""})):
            if event == "PostToolUse":
                with path.open("ab") as stream:
                    stream.write(line(result(step + 1, n, rng)))
            done, wall, cpu = run_process(hook_command(event), env, json.dumps({**payload, **extra}))
            samples["pre" if event == "PreToolUse" else "post"].append((wall, cpu))
            text = done.stdout + done.stderr
            if done.returncode or "unavailable" in text or "not recorded" in text or "failed" in text.lower():
                problems.append(f"{event}: {text.strip()[:200]}")
        step, n = step + 2, n + 1
    verified, _wall, _cpu = run_process(self_command("--child", "verify", "--root", str(root), "--index", str(index)), env)
    state = json.loads(verified.stdout.strip().splitlines()[-1]) if verified.stdout.strip() else {"error": verified.stderr[-500:]}
    state.update(expected_results=n, hook_problems=problems[:3], warm_seconds=warmed["seconds"])

    def median(values):
        return {"wall_s": round(statistics.median(v[0] for v in values), 4),
                "cpu_s": round(statistics.median(v[1] for v in values), 4)}
    return {"floor": median(floor), "pre": median(samples["pre"]), "post": median(samples["post"]), "state": state}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", help="absolute scratch directory (transcripts and indexes are written here)")
    parser.add_argument("--sizes", default="0.1,1,3,10", help="transcript sizes in MB")
    parser.add_argument("--repeats", type=int, default=5, help="append/hook cycles per measurement (median reported)")
    parser.add_argument("--baseline-src", help="src directory of an older tree to measure as the 'before'")
    parser.add_argument("--trees", help="comma separated subset of: this-tree, baseline (default: all that are available)")
    parser.add_argument("--child", choices=("warm", "verify"))
    parser.add_argument("--root")
    parser.add_argument("--index")
    args = parser.parse_args()
    if args.child:
        return {"warm": child_warm, "verify": child_verify}[args.child](args)
    if not args.work_dir or not Path(args.work_dir).is_absolute():
        raise SystemExit("--work-dir must be an absolute path")
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(here))
    from self_directing_mcp.journal import copy_index_dir
    trees = [("this-tree", here)] + ([("baseline", Path(args.baseline_src))] if args.baseline_src else [])
    if args.trees:
        trees = [tree for tree in trees if tree[0] in args.trees.split(",")]
    report = {}
    for size in args.sizes.split(","):
        pristine = work / f"pristine-{size}MB"
        if not (pristine / "meta.json").exists():
            if pristine.exists():
                shutil.rmtree(pristine)
            print(f"generating {size} MB ...", flush=True)
            generate(pristine, float(size))
        for label, src in trees:
            root = work / f"run-{label}-{size}MB"
            if root.exists():
                shutil.rmtree(root)
            copy_index_dir(pristine, root)       # databases through the backup API, never a bare file copy
            print(f"measuring {label} at {size} MB ...", flush=True)
            report.setdefault(label, {})[size] = measure(f"{label}-{size}", src, root, work, args.repeats)
    print(json.dumps(report, indent=2))
    print("\n| tree | MB | floor wall/cpu | pre wall/cpu | post wall/cpu | linked/results |")
    print("|---|---|---|---|---|---|")
    for label, sizes in report.items():
        for size, row in sizes.items():
            cell = lambda m: f"{m['wall_s']:.3f} / {m['cpu_s']:.3f}"
            state = row["state"]
            print(f"| {label} | {size} | {cell(row['floor'])} | {cell(row['pre'])} | {cell(row['post'])} | "
                  f"{state.get('linked')}/{state.get('expected_results')} |")


if __name__ == "__main__":
    main()
