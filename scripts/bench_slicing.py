"""What slice commits cost and what they buy, on synthetic Codex sessions. Nothing real is read.

A sync that does not fit its request budget used to be parsed until the hard deadline, rolled back, and started again from
zero by the next call. With slicing (SELF_DIRECT_SYNC_SLICING=auto, the default) each call publishes the records it
parsed. This script measures, for sessions of 100,000 and 300,000 events and per-call budgets of 1, 4 and 8 seconds:

  baseline        a one-shot sync with no deadline right before each sliced run (the cost to beat), and four consecutive
                  budgeted calls with slicing off (what used to happen: no progress)
  convergence     with slicing on: calls until the sync is complete, each call's wall time (it must stay inside its
                  budget), total CPU against the one-shot sync, the time spent in the capture and the parse and the
                  commit of each slice
  capture         how the capture time grows with the number of slices, from a sweep with forced slice counts and a
                  least-squares line through (slices, capture seconds); the capture cap (SELF_DIRECT_SYNC_SLICE_BYTES)
                  is what keeps one slice's capture from growing with the file
  overhead        slicing on against slicing off on a session that fits its budget, so no stop ever happens: the cost of
                  the code that is only there to be able to stop (target: 2 percent or less), interleaved runs, median CPU

    python scripts/bench_slicing.py [--events 100000,300000] [--budgets 1,4,8] [--overhead-events 40000]
                                    [--overhead-runs 9] [--sweep-events 100000] [--json FILE] [--dir DIR]
    python scripts/bench_slicing.py --reindex [--events 100000,300000,500000] [--reindex-budget 8] [--json FILE] [--dir DIR]

``--reindex`` measures the re-index of a session after a parser upgrade (``reindex_spool``) instead: for each size, the cold
sync of the session and the first audit after it; the one-shot re-index (no deadline, one transaction) with its peak memory and
extra disk; the staged re-index (budgeted calls, each in a fresh process, until the commit is published) with the time of every
slice (capture, validation of the spool, parse, spool write, the commit) against the prediction ``18 us * rows``, the peak
memory of the call that commits against the one-shot re-index, the extra disk it needed at its peak (spool, database growth,
journal, the temporary file of a slice), and the first audit after it against the first audit after the cold sync.

Everything lives in one temporary directory that is deleted at the end (give --dir to choose its parent). The per-user
TOML configuration is replaced by an empty file. Results are printed, never written into the repository.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SID = "11111111-2222-3333-4444-555555555555"
WORDS = ("alpha beta gamma delta epsilon zeta theta iota kappa lambda sigma omega build test deploy review lint format "
         "merge plan note check retry cache index stream record slice commit").split()


def build_session(path: Path, events: int, seed: int = 7) -> int:
    """A Codex rollout of ``events`` events (messages, tool calls with their outputs, agent messages), about 215 bytes each."""
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
            cid = f"call-{serial}"
            lines.append(json.dumps({"type": "response_item", "payload": {
                "type": "function_call", "name": "shell", "call_id": cid, "arguments": {"command": "echo " + say(3, 10)}}}))
            lines.append(json.dumps({"type": "response_item", "payload": {
                "type": "function_call_output", "call_id": cid,
                "output": json.dumps({"exit_code": 0, "output": say(16, 44)})}}))
            produced += 2
        else:
            lines.append(json.dumps({"type": "event_msg", "payload": {"type": "agent_message", "message": say(10, 30)}}))
            produced += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ("\n".join(lines) + "\n").encode()
    path.write_bytes(data)
    return len(data)


class Run:
    """One index directory and engine for the session, with timers around the parts of a sync."""

    def __init__(self, base: Path, name: str, root: Path, slicing: str, policy=None):
        from self_directing_mcp import session_events
        from self_directing_mcp.codex import parse as codex_parse
        from self_directing_mcp.config import Settings
        from self_directing_mcp.engine import SelfDirectEngine
        from self_directing_mcp.index import ingest
        self.laps = {"capture": 0.0, "parse": 0.0, "commit": 0.0}
        self.engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, openrouter_api_key_file=None,
                                                codex_sessions_dir=root, index_dir=base / f"index-{name}",
                                                sync_slicing=slicing))
        if policy is not None:
            self.engine.slice_policy = policy
        self._undo = []

        def timed(owner, attribute, key):
            original = getattr(owner, attribute)

            def wrapper(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    self.laps[key] += time.perf_counter() - started
            setattr(owner, attribute, wrapper)
            self._undo.append(lambda: setattr(owner, attribute, original))

        timed(ingest, "capture_source", "capture")
        timed(ingest, "capture_source_fast", "capture")
        timed(codex_parse, "parse_session_chunks", "parse")
        timed(ingest.SessionStore, "commit_events", "commit")
        self.session_events = session_events

    def sync(self, root: Path, budget: float | None, verify: str = "full"):
        """One ``sync_session``; returns (info or None when the request ran out, wall seconds, cpu seconds, laps)."""
        for key in self.laps:
            self.laps[key] = 0.0
        wall, cpu = time.perf_counter(), time.process_time()
        info = None
        try:
            kwargs = {} if budget is None else {"_deadline": time.monotonic() + budget}
            info = self.engine.sync_session(session_id=SID, provider="codex", embed=False, derived=False, verify=verify,
                                            **kwargs)
        except Exception as exc:                                           # noqa: BLE001 - reported, not hidden
            if type(exc).__name__ != "RequestStopped":
                raise
        return info, time.perf_counter() - wall, time.process_time() - cpu, dict(self.laps)

    def cursor(self):
        record = self.engine.store.cursor_record(SID, "codex")
        return record["byte_offset"] if record else None

    def close(self):
        for undo in reversed(self._undo):
            undo()
        self.engine.close()


def stats(values):
    return {"min": round(min(values), 3), "median": round(statistics.median(values), 3), "max": round(max(values), 3)}


def one_shot(base, root, name="oneshot"):
    run = Run(base, name, root, "auto")
    info, wall, cpu, laps = run.sync(root, None)
    run.close()
    return {"wall": round(wall, 2), "cpu": round(cpu, 2), "events": info["total_chunks"], "laps": {k: round(v, 2) for k, v in laps.items()}}


def old_behaviour(base, root, budget, tries=4):
    """Slicing off: consecutive calls with the same budget (the cursor after each; ``None`` is no progress at all)."""
    run = Run(base, f"off-{budget}", root, "off")
    cursors, walls = [], []
    for _ in range(tries):
        info, wall, _cpu, _laps = run.sync(root, budget)
        cursors.append(run.cursor())
        walls.append(round(wall, 2))
        if info is not None:
            break
    run.close()
    return {"cursor_after_each_call": cursors, "wall_seconds": walls}


def converge(base, root, size, budget, cpu_reference, limit=400):
    run = Run(base, f"auto-{budget}", root, "auto")
    calls, previous = [], 0
    for number in range(1, limit + 1):
        info, wall, cpu, laps = run.sync(root, budget)
        if info is None:
            calls.append({"call": number, "failed": True, "wall": round(wall, 2)})
            break
        sync = info["sync"]
        calls.append({"call": number, "wall": round(wall, 3), "cpu": round(cpu, 3), "indexed": sync["indexed_through"],
                      "progress": sync["progress"], "state": sync["state"], "stopped_by": sync["stopped_by"],
                      **{k: round(v, 3) for k, v in laps.items()}})
        assert sync["indexed_through"] > previous or sync["state"] == "complete", "no progress"
        previous = sync["indexed_through"]
        if sync["state"] == "complete":
            break
    run.close()
    done = [c for c in calls if not c.get("failed")]
    total_cpu = sum(c["cpu"] for c in done)
    return {"budget": budget, "calls": len(calls), "complete": bool(done) and done[-1]["state"] == "complete",
            "wall_per_call": stats([c["wall"] for c in done]), "inside_budget": all(c["wall"] <= budget for c in done),
            "total_cpu": round(total_cpu, 2), "cpu_vs_one_shot": round(total_cpu / cpu_reference, 3),
            "capture_total": round(sum(c["capture"] for c in done), 3), "parse_total": round(sum(c["parse"] for c in done), 2),
            "commit_total": round(sum(c["commit"] for c in done), 2),
            "slice_bytes": stats([b - a for a, b in zip([0] + [c["indexed"] for c in done], [c["indexed"] for c in done])])
            if len(done) > 1 else None,
            "detail": calls}


def forced_slices(base, root, slices, size):
    """A sync cut into about ``slices`` equal pieces by a record cap (no time involved): the capture time of each."""
    from self_directing_mcp.index.ingest import SlicePolicy
    records = size // 215
    cap = max(1, math.ceil(records / slices))
    run = Run(base, f"sweep-{slices}", root, "auto", SlicePolicy(min_records=1, min_bytes=0, max_records=cap))
    calls = captured = 0.0
    while True:
        info, wall, _cpu, laps = run.sync(root, 3600.0)
        calls += 1
        captured += laps["capture"]
        if info["sync"]["state"] == "complete":
            break
    run.close()
    return int(calls), captured


def linear_fit(points):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in points) / sxx
    intercept = my - slope * mx
    total = sum((y - my) ** 2 for y in ys) or 1.0
    residual = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
    return slope, intercept, 1 - residual / total


def overhead(base, root, runs):
    """Slicing on (a finite deadline that is never reached) against slicing off, interleaved, a fresh index for each run."""
    cpu = {"off": [], "auto": []}
    for number in range(runs):
        for mode in (("off", "auto") if number % 2 == 0 else ("auto", "off")):
            run = Run(base, f"overhead-{mode}-{number}", root, mode)
            info, _wall, seconds, _laps = run.sync(root, 3600.0)
            assert info is not None and ("sync" in info) == (mode == "auto") and info["coverage"]["byte_offset"] > 0
            run.close()
            shutil.rmtree(base / f"index-overhead-{mode}-{number}", ignore_errors=True)
            cpu[mode].append(seconds)
    off, auto = statistics.median(cpu["off"]), statistics.median(cpu["auto"])
    return {"runs": runs, "off_cpu_median": round(off, 3), "auto_cpu_median": round(auto, 3),
            "overhead_percent": round((auto / off - 1) * 100, 2), "off_all": [round(x, 3) for x in cpu["off"]],
            "auto_all": [round(x, 3) for x in cpu["auto"]]}


# --- re-index (reindex_spool) ------------------------------------------------------------------------------------------

CONTRACTS = [   # the six contracts of scripts/bench_audit.py: regex, an ordering, a success requirement, message scope
    {"id": "no-recursive-delete", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf\s+/", "roles": ["assistant"]},
    {"id": "tests-before-deploy", "type": "must", "scope": "tool_call", "regex": "npm test", "before_regex": "deploy now",
     "requires_success": True},
    {"id": "linted", "type": "must", "scope": "tool_call", "regex": "eslint"},
    {"id": "no-leaked-output", "type": "must_not", "scope": "tool_result", "regex": "LEAKED_SECRET", "roles": ["tool"]},
    {"id": "summarized", "type": "must", "scope": "message", "regex": "summary", "roles": ["assistant"]},
    {"id": "release-after-publish", "type": "must", "scope": "tool_call", "regex": "release",
     "when": {"tool_call_regex": "npm publish"}},
]


def _status():
    """(current, peak) resident set in KiB, from /proc (Linux); zeros elsewhere."""
    values = {"VmRSS": 0, "VmHWM": 0}
    try:
        with open("/proc/self/status") as stream:
            for line in stream:
                name, _, rest = line.partition(":")
                if name in values:
                    values[name] = int(rest.split()[0])
    except OSError:
        pass
    return values["VmRSS"], values["VmHWM"]


def _reset_peak() -> None:
    try:
        with open("/proc/self/clear_refs", "w") as stream:
            stream.write("5")                   # resets the peak resident set (VmHWM) of this process
    except OSError:
        pass


def _unlinked_bytes(*roots) -> int:
    """Bytes of files under ``roots`` that were deleted while this process still has them open (the temporary file of a
    ``RowSpool`` is one: a directory walk cannot see it). Linux only."""
    total = 0
    try:
        for entry in os.listdir("/proc/self/fd"):
            try:
                target = os.readlink(f"/proc/self/fd/{entry}")
                if target.endswith(" (deleted)") and any(target.startswith(str(root)) for root in roots):
                    total += os.fstat(int(entry)).st_size
            except (OSError, ValueError):
                pass
    except OSError:
        pass
    return total


def _tree_bytes(*roots) -> int:
    total = _unlinked_bytes(*roots)
    for root in roots:
        for directory, _names, files in os.walk(root):
            for name in files:
                try:
                    total += os.stat(os.path.join(directory, name)).st_size
                except OSError:
                    pass
    return total


class DiskSampler(threading.Thread):
    """The most bytes the index directory and the temporary directory held at any moment (polled every 10 ms)."""

    def __init__(self, *roots):
        super().__init__(daemon=True)
        self.roots, self.peak, self.stopping = roots, 0, threading.Event()

    def run(self):
        while not self.stopping.is_set():
            self.peak = max(self.peak, _tree_bytes(*self.roots))
            time.sleep(0.01)

    def finish(self) -> int:
        self.stopping.set()
        self.join()
        self.peak = max(self.peak, _tree_bytes(*self.roots))
        return self.peak


def call_in_child(spec: dict) -> dict:
    """One call in a fresh process (so memory and the cost of opening a spool are those of a new process)."""
    done = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", json.dumps(spec)], capture_output=True,
                          text=True, check=False)
    if done.returncode:
        raise RuntimeError(done.stderr[-2000:])
    return json.loads(done.stdout.strip().splitlines()[-1])


def child_call(spec: dict) -> None:
    """The child: one ``sync_session`` or ``audit_session`` against ``spec['index']``, timed. Prints one JSON document."""
    from self_directing_mcp import reindex_spool
    from self_directing_mcp.codex import parse as codex_parse
    from self_directing_mcp.config import Settings
    from self_directing_mcp.engine import SelfDirectEngine
    from self_directing_mcp.index import ingest
    tempfile.tempdir = spec["tmp"]                       # the temporary file of a slice is part of its disk footprint
    if spec.get("old_parser"):
        codex_parse.PARSER_VERSION -= 1                  # the index this builds is out of date for the code that follows
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, openrouter_api_key_file=None,
                                       codex_sessions_dir=spec["sessions"], index_dir=spec["index"],
                                       sync_slicing="auto", seed_example_contracts=False))
    laps, counts = {"capture": 0.0, "parse": 0.0, "open": 0.0, "persist": 0.0, "rowsource": 0.0, "commit": 0.0}, {}

    def timed(owner, attribute, key, rows=None):
        method = getattr(owner, attribute)

        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                return method(*args, **kwargs)
            finally:
                laps[key] += time.perf_counter() - started
                if rows is not None:
                    counts[key] = rows(args, kwargs)
        setattr(owner, attribute, wrapper)

    timed(ingest, "capture_source", "capture")
    timed(ingest, "capture_source_fast", "capture")
    timed(codex_parse, "parse_session_chunks", "parse")
    original_open = reindex_spool.ReindexSpool.open.__func__

    def timed_open(cls, *args, **kwargs):
        started = time.perf_counter()
        try:
            return original_open(cls, *args, **kwargs)
        finally:
            laps["open"] += time.perf_counter() - started

    reindex_spool.ReindexSpool.open = classmethod(timed_open)
    timed(reindex_spool.ReindexSpool, "append_slice", "persist")
    timed(reindex_spool.ReindexSpool, "as_row_source", "rowsource")
    timed(ingest.SessionStore, "commit_events", "commit",
          rows=lambda args, kwargs: getattr(kwargs.get("prepared_rows"), "count", 0))
    index = Path(spec["index"])
    sessions = Path(spec["sessions"])
    path = next(sessions.glob("*.jsonl"))
    if spec["kind"] == "audit":
        engine.upsert_contracts(CONTRACTS)
    before_disk = _tree_bytes(index, spec["tmp"])
    baseline, _peak = _status()
    _reset_peak()
    sampler = DiskSampler(index, spec["tmp"])
    sampler.start()
    wall, cpu = time.perf_counter(), time.process_time()
    result = {}
    if spec["kind"] == "audit":
        answer = engine.audit_session(SID, provider="codex", path=str(path))
        result.update(verdict=answer["verdict"], mode=(answer.get("incremental") or {}).get("mode"))
    else:
        kwargs = {} if spec.get("budget") is None else {"_deadline": time.monotonic() + spec["budget"]}
        info = engine.sync_session(session_id=SID, provider="codex", path=str(path), embed=False, derived=False, **kwargs)
        sync = info.get("sync") or {}
        staged = sync.get("reindex") or {}
        result.update(state=sync.get("state", "complete"), reindexed=info["reindexed"], reindex=staged.get("state"),
                      staged_through=staged.get("staged_through"), rows=staged.get("rows"),
                      target_bytes=staged.get("target_bytes"), total_chunks=info["total_chunks"])
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    peak_disk = sampler.finish()
    now, peak = _status()
    spool = sorted(index.glob("reindex/*.spool"))
    result.update(wall=round(wall, 3), cpu=round(cpu, 3), laps={k: round(v, 3) for k, v in laps.items()},
                  commit_rows=counts.get("commit", 0), rss_start_mb=round(baseline / 1024, 1),
                  rss_peak_mb=round(peak / 1024, 1), rss_extra_mb=round((peak - baseline) / 1024, 1),
                  disk_before_mb=round(before_disk / 1e6, 1), disk_peak_mb=round(peak_disk / 1e6, 1),
                  disk_extra_peak_mb=round((peak_disk - before_disk) / 1e6, 1),
                  spool_mb=round(sum(f.stat().st_size for f in spool) / 1e6, 1),
                  database_mb=round(sum((index / name).stat().st_size for name in ("meta.sqlite",) if (index / name).exists()) / 1e6, 1))
    engine.close()
    print(json.dumps(result), flush=True)


def reindex_one(base: Path, events: int, budget: float) -> dict:
    """Cold sync, one-shot re-index and staged re-index of one synthetic session, each from the same starting index."""
    root = base / f"sessions-reindex-{events}"
    size = build_session(root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl", events)
    tmp = base / "tmp"
    tmp.mkdir(exist_ok=True)
    spec = dict(sessions=str(root), tmp=str(tmp), budget=None)

    def fresh(name, template=None):
        target = base / f"index-{name}"
        shutil.rmtree(target, ignore_errors=True)
        if template is None:
            target.mkdir()
        else:
            shutil.copytree(base / f"index-{template}", target)
        return str(target)

    def first_audit(index, name, repeats=3):
        """The first audit after a sync, in a fresh process, on a copy of the index each time; the median of ``repeats``."""
        walls = []
        for number in range(repeats):
            copy = fresh(f"{name}-audit-{number}", name)
            walls.append(call_in_child(dict(spec, kind="audit", index=copy))["wall"])
            shutil.rmtree(copy, ignore_errors=True)
        return {"wall": statistics.median(walls), "all": walls}

    out = {"events": events, "megabytes": round(size / 1e6, 1)}
    # The cold sync of the session at the CURRENT parser, then the first audit after it: what the re-index is compared with.
    index = fresh("cold")
    out["cold_sync"] = call_in_child(dict(spec, kind="sync", index=index))
    out["cold_audit"] = first_audit(index, "cold")
    shutil.rmtree(index, ignore_errors=True)
    # The out-of-date index every re-index starts from (never audited: a copy is used each time).
    old = fresh("old")
    out["old_sync"] = call_in_child(dict(spec, kind="sync", index=old, old_parser=True))
    # One transaction, as before.
    index = fresh("oneshot", "old")
    out["one_shot"] = call_in_child(dict(spec, kind="sync", index=index))
    out["one_shot_audit"] = first_audit(index, "oneshot")
    shutil.rmtree(index, ignore_errors=True)
    # Staged: the same call, with a budget, until the commit is published.
    index = fresh("staged", "old")
    calls = []
    for number in range(1, 200):
        answer = call_in_child(dict(spec, kind="sync", index=index, budget=budget))
        calls.append(answer)
        if answer["state"] == "complete":
            break
        if answer["reindex"] == "ready_to_commit" and len(calls) > 1 and calls[-2]["reindex"] == "ready_to_commit":
            break
    out["staged_calls"], out["staged_budget"] = calls, budget
    if calls[-1]["state"] != "complete":     # the commit does not fit this budget: the explicit sync_session (45 s) gets it
        calls.append(call_in_child(dict(spec, kind="sync", index=index, budget=45.0)))
        out["explicit"] = calls[-1]
    out["staged_audit"] = first_audit(index, "staged")
    shutil.rmtree(index, ignore_errors=True)
    shutil.rmtree(old, ignore_errors=True)
    shutil.rmtree(root, ignore_errors=True)
    return out


def print_reindex(item: dict) -> None:
    calls = item["staged_calls"]          # the explicit sync_session, when there was one, is the last of them
    one = item["one_shot"]
    last = calls[-1]
    print(f"\n## re-index of {item['events']:,} events ({item['megabytes']} MB), calls of {item['staged_budget']:g} s")
    cold = item["cold_sync"]
    print(f"\ncold sync {cold['wall']} s wall / {cold['cpu']} s CPU; one-shot re-index {one['wall']} s wall / {one['cpu']} s CPU "
          f"(commit {one['laps']['commit']} s for {one['commit_rows']} rows = "
          f"{one['laps']['commit'] / max(1, one['commit_rows']) * 1e6:.1f} us/row; peak memory +{one['rss_extra_mb']} MB, "
          f"peak extra disk {one['disk_peak_mb'] - one['disk_before_mb']:.1f} MB over the old index)")
    print("\n| call | wall | state | staged / target MB | rows | capture | spool open+verify | parse | spool write | row index | commit | memory +MB | disk peak MB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for number, call in enumerate(calls, 1):
        laps = call["laps"]
        staged = "-" if call.get("staged_through") is None else f"{call['staged_through'] / 1e6:.1f} / {call['target_bytes'] / 1e6:.1f}"
        state = call.get("reindex") or call["state"]
        print(f"| {number}{' (explicit, 45 s)' if call is item.get('explicit') else ''} | {call['wall']} | {state} | {staged} | "
              f"{call.get('rows') or call['commit_rows'] or '-'} | {laps['capture']} | {laps['open']} | {laps['parse']} | "
              f"{laps['persist']} | {laps['rowsource']} | {laps['commit']} | {call['rss_extra_mb']} | {call['disk_peak_mb']} |")
    commit_call = next(call for call in reversed(calls) if call["laps"]["commit"])
    predicted = 18e-6 * commit_call["commit_rows"]
    old_mb = calls[0]["disk_before_mb"]
    peak_mb = max(call["disk_peak_mb"] for call in calls)
    print(f"\ncommit {commit_call['laps']['commit']} s for {commit_call['commit_rows']} rows against the prediction 18 us * rows = "
          f"{predicted:.2f} s ({commit_call['laps']['commit'] / max(predicted, 1e-9):.2f}x; the safety factor is 1.5); the call that "
          f"commits peaked at +{commit_call['rss_extra_mb']} MB of memory ({commit_call['rss_peak_mb']} MB resident) against "
          f"+{one['rss_extra_mb']} MB ({one['rss_peak_mb']} MB resident) for the one-shot re-index.")
    print(f"disk: the spool held {max(call['spool_mb'] for call in calls)} MB at most ({max(call['spool_mb'] for call in calls) / item['megabytes']:.1f} "
          f"bytes per source byte); the largest disk footprint (index and temporary files) was {peak_mb} MB, {peak_mb - old_mb:.1f} MB over "
          f"the old index of {old_mb} MB (one-shot: {one['disk_peak_mb'] - one['disk_before_mb']:.1f} MB over); the database "
          f"{item['old_sync']['database_mb']} -> {commit_call['database_mb']} MB.")
    print(f"in total the staged re-index took {sum(c['wall'] for c in calls):.2f} s wall / {sum(c['cpu'] for c in calls):.2f} s CPU in "
          f"{len(calls)} calls, against {one['wall']} s / {one['cpu']} s for the one-shot re-index; spool writes "
          f"{sum(c['laps']['persist'] for c in calls):.2f} s, spool validation {sum(c['laps']['open'] for c in calls):.2f} s, "
          f"captures {sum(c['laps']['capture'] for c in calls):.2f} s.")
    print(f"first audit after (median of 3, a fresh process each): cold sync {item['cold_audit']['wall']} s "
          f"{item['cold_audit']['all']}, one-shot re-index {item['one_shot_audit']['wall']} s {item['one_shot_audit']['all']}, "
          f"staged re-index {item['staged_audit']['wall']} s {item['staged_audit']['all']} "
          f"(acceptance: the staged one is not worse than the cold one).")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events", default="100000,300000")
    parser.add_argument("--budgets", default="1,4,8")
    parser.add_argument("--overhead-events", type=int, default=40000)
    parser.add_argument("--overhead-runs", type=int, default=9)
    parser.add_argument("--sweep-events", type=int, default=100000)
    parser.add_argument("--json", help="also write the results to this file (outside the repository)")
    parser.add_argument("--dir", help="parent of the temporary directory")
    parser.add_argument("--reindex", action="store_true", help="measure the re-index (reindex_spool) instead")
    parser.add_argument("--reindex-budget", type=float, default=8.0, help="seconds each staged call of --reindex may take")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        os.environ["SELF_DIRECT_CONFIG_FILE"] = os.path.join(json.loads(args.child)["tmp"], "config.toml")
        os.environ["SELF_DIRECT_DENSE_BACKEND"] = "numpy"
        child_call(json.loads(args.child))
        return
    base = Path(tempfile.mkdtemp(prefix="bench-slicing-", dir=args.dir))
    config = base / "config.toml"
    config.write_text("", encoding="utf-8")
    os.environ["SELF_DIRECT_CONFIG_FILE"] = str(config)
    os.environ["SELF_DIRECT_DENSE_BACKEND"] = "numpy"
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None     # not on Windows
    results = {"python": sys.version.split()[0], "cpus": os.cpu_count(), "load_1m": load, "sessions": []}
    if args.reindex:
        (base / "tmp").mkdir(exist_ok=True)
        (base / "tmp" / "config.toml").write_text("", encoding="utf-8")
        results["reindex"] = []
        try:
            for events in [int(x) for x in args.events.split(",")]:
                item = reindex_one(base, events, args.reindex_budget)
                results["reindex"].append(item)
                print_reindex(item)
        finally:
            shutil.rmtree(base, ignore_errors=True)
        if args.json:
            Path(args.json).write_text(json.dumps(results, indent=1))
        return
    try:
        for events in [int(x) for x in args.events.split(",")]:
            root = base / f"sessions-{events}"
            size = build_session(root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl", events)
            entry = {"events": events, "megabytes": round(size / 1e6, 1), "budgets": []}
            for budget in [float(x) for x in args.budgets.split(",")]:
                # The one-shot sync the sliced run is compared with is measured right before it: the machine is not
                # quiet, and a baseline taken minutes earlier makes the ratio a measurement of the load.
                baseline = one_shot(base, root, name=f"oneshot-{budget}")
                shutil.rmtree(base / f"index-oneshot-{budget}", ignore_errors=True)
                entry["budgets"].append({"one_shot": baseline, "old": old_behaviour(base, root, budget),
                                         "sliced": converge(base, root, size, budget, baseline["cpu"])})
                shutil.rmtree(base / f"index-auto-{budget}", ignore_errors=True)
                shutil.rmtree(base / f"index-off-{budget}", ignore_errors=True)
            results["sessions"].append(entry)
            print_session(entry)
            shutil.rmtree(root, ignore_errors=True)
        root = base / "sessions-sweep"
        size = build_session(root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl", args.sweep_events)
        points = []
        for slices in (1, 2, 4, 8, 16, 32, 64):
            count, captured = forced_slices(base, root, slices, size)
            points.append((count, captured))
            shutil.rmtree(base / f"index-sweep-{slices}", ignore_errors=True)
        slope, intercept, r2 = linear_fit(points)
        results["capture_sweep"] = {"events": args.sweep_events, "points": [(c, round(s, 4)) for c, s in points],
                                    "seconds_per_slice": round(slope, 5), "intercept": round(intercept, 4), "r2": round(r2, 4)}
        print_sweep(results["capture_sweep"])
        shutil.rmtree(root, ignore_errors=True)
        root = base / "sessions-fit"
        build_session(root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl", args.overhead_events)
        results["overhead"] = overhead(base, root, args.overhead_runs)
        print_overhead(results["overhead"])
    finally:
        shutil.rmtree(base, ignore_errors=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


def print_session(entry):
    print(f"\n## {entry['events']:,} events, {entry['megabytes']} MB")
    print("\n| budget | one-shot sync (wall / CPU, just before) | slicing off: cursor after 4 calls | slicing on: calls | "
          "wall per call (min/med/max) | inside budget | total CPU (x one-shot) | capture / parse / commit in total "
          "| slice size MB (min/med/max) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for item in entry["budgets"]:
        shot, old, new = item["one_shot"], item["old"], item["sliced"]
        sizes = new["slice_bytes"]
        mb = "-" if not sizes else "/".join(f"{sizes[k] / 1e6:.1f}" for k in ("min", "median", "max"))
        print(f"| {new['budget']:g} s | {shot['wall']} / {shot['cpu']} s | {old['cursor_after_each_call']} | {new['calls']} | "
              f"{new['wall_per_call']['min']}/{new['wall_per_call']['median']}/{new['wall_per_call']['max']} s | "
              f"{'yes' if new['inside_budget'] else 'NO'} | {new['total_cpu']} s ({new['cpu_vs_one_shot']}) | "
              f"{new['capture_total']} / {new['parse_total']} / {new['commit_total']} s | {mb} |")


def print_sweep(sweep):
    print(f"\n## capture time against the number of slices ({sweep['events']:,} events, forced equal slices)")
    print("\n| slices | capture seconds in total |")
    print("|---|---|")
    for count, seconds in sweep["points"]:
        print(f"| {count} | {seconds} |")
    print(f"\nline: {sweep['seconds_per_slice']} s per slice + {sweep['intercept']} s, R^2 = {sweep['r2']}")


def print_overhead(item):
    print(f"\n## overhead of slicing that never stops (median CPU of {item['runs']} interleaved runs)")
    print(f"\noff {item['off_cpu_median']} s, auto {item['auto_cpu_median']} s: {item['overhead_percent']} percent "
          f"(target: 2 percent or less)")


if __name__ == "__main__":
    main()
