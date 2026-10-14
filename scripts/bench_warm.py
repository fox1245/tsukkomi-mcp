#!/usr/bin/env python3
"""Benchmark of the WARM judgement path: one long-lived engine, repeated hook judgements. Synthetic data only.

``bench_hooks.py`` measures fresh hook processes and ``bench_daemon.py`` measures a daemon through its socket. This
script measures what remains once the process and its imports are gone: the engine's own work per request, in one
process, with one ``SelfDirectEngine`` kept between requests (what the daemon does). It reuses the synthetic sessions,
contracts, payloads and sandbox of ``bench_hooks.py`` and runs under an isolated HOME, index and config file; it
never reads a user's sessions, indexes or credentials.

Per cell (host x event x bytes appended before each request) it reports the wall time p50 / p95 / min of
``claude_hooks.run_hook`` / ``agy_hooks.run_hook``, and how many durable SQLite commits a request made (an ``fsync``
each: on this class of machine 10 to 25 ms, which is most of what is left once the CPU work is gone). With
``--profile`` it also prints a ``cProfile`` of one cell; with ``--spans`` the inclusive wall time of the stages the
judgement is made of (per request, so the numbers add up to the request time where the spans do not nest).

    python scripts/bench_warm.py [--runs 40] [--contracts 10] [--session-mb 1] [--defer]
                                 [--shadow-every N] [--min-events N] [--tmpdir DIR]
                                 [--profile claude:Stop:4096 [--top 40]] [--spans claude:Stop:4096]
                                 [--out report.json] [--markdown report.md]

``--defer`` builds the engine the way the daemon does (``defer_derived=True``: monitor states, verification stamps and
the AGY indexed marker are held and written with the next authority commit). ``--tmpdir /dev/shm/...`` puts the
index on a RAM disk, which takes the disk's ``fsync`` out and leaves the CPU cost. Run both to separate the two.
"""
from __future__ import annotations

import argparse
import cProfile
import functools
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import pstats
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))     # this checkout, whatever is installed
spec = importlib.util.spec_from_file_location("bench_hooks_for_warm", HERE / "bench_hooks.py")
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)

SESSION_CLAUDE = "aaaaaaaa-1111-4111-8111-aaaaaaaa0001"
SESSION_AGY = "11111111-1111-4111-8111-111111111111"
CELLS = (("claude", "PreToolUse"), ("claude", "Stop"), ("agy", "PreToolUse"), ("agy", "Stop"))

# --- durable commits ---------------------------------------------------------------------------------------------
COMMITS = {"n": 0}


def count_commits() -> None:
    """Count the COMMITs that wrote something (a commit with no change does not sync), on every connection opened
    from now on, through SQLite's own statement trace."""
    real = sqlite3.connect

    def connect(*args, **kwargs):
        connection = real(*args, **kwargs)
        seen = {"changes": 0}

        def trace(statement):
            head = statement.lstrip()[:6].upper()
            if head.startswith(("COMMIT", "END")):
                if connection.total_changes > seen["changes"]:
                    COMMITS["n"] += 1
                seen["changes"] = connection.total_changes

        try:
            connection.set_trace_callback(trace)
        except sqlite3.Error:
            pass
        return connection

    sqlite3.connect = connect


# --- wall-clock spans ----------------------------------------------------------------------------------------------
SPANS: dict[str, list] = {}


def span(owner, name, label=None) -> None:
    function = getattr(owner, name, None)
    if function is None:                               # an older checkout does not have every function
        return
    label = label or f"{getattr(owner, '__name__', owner.__class__.__name__)}.{name}"

    @functools.wraps(function)
    def timed(*args, **kwargs):
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            entry = SPANS.setdefault(label, [0, 0.0])
            entry[0] += 1
            entry[1] += time.perf_counter() - started

    setattr(owner, name, timed)


def install_spans() -> None:
    from self_directing_mcp import agy_receipts, engine as engine_module, hook_policy
    from self_directing_mcp import neograph_runtime
    from self_directing_mcp.audit import incremental, runner
    from self_directing_mcp.index import ingest, sparse
    engine = engine_module.SelfDirectEngine
    for name in ("hook_obligations", "_refresh_evidence", "_sync_session", "get_checklist", "flush_derived"):
        span(engine, name)
    store = ingest.SessionStore
    for name in ("capture_audit_events", "capture_audit_tail", "state_token", "cursor_record", "source_block_record",
                 "save_audit_states", "mark_full_verified", "commit_events", "load_audit_states", "capture_evidence",
                 "chunk_ids", "live_ids", "flush_derived"):
        span(store, name)
    span(ingest, "ingest_session", "ingest.ingest_session")
    span(ingest, "capture_for_cursor")
    span(engine_module, "run_audit", "engine.run_audit")
    span(incremental.AuditPlan, "prepare")
    span(incremental.AuditPlan, "evaluate")
    span(incremental, "compare")
    span(runner, "full_findings")
    span(neograph_runtime, "run_stages")
    span(hook_policy, "judge_action")
    span(hook_policy, "checkpoint")
    for name in ("record", "mark_indexed", "authority_token", "read_capture", "prepare_bindings", "publish_bindings"):
        span(agy_receipts.ReceiptStore, name)
    span(sparse.SparseIndex, "upsert_many")
    span(sparse.SparseIndex, "present")


def report_spans(requests: int, key: str, top: int = 24) -> None:
    print(f"  spans per request for {key} (inclusive ms [calls]; spans nest, so they do not add up):")
    for label, (calls, total) in sorted(SPANS.items(), key=lambda item: -item[1][1])[:top]:
        print(f"    {total / requests * 1000:8.2f} ms [{calls / requests:5.1f}]  {label}")


# --- the lab -------------------------------------------------------------------------------------------------------
def loadavg() -> tuple:
    return os.getloadavg() if hasattr(os, "getloadavg") else (float("nan"),) * 3


def trim_to_record(path: Path, size: int) -> None:
    """Cut the file back to the last complete record at or before ``size`` bytes (the generator writes whole chunks of
    64 steps, so a small session cannot be asked for otherwise)."""
    data = path.read_bytes()
    if len(data) > size:
        path.write_bytes(data[:data.rfind(b"\n", 0, size) + 1])


class Lab:
    def __init__(self, args) -> None:
        self.args = args
        self.root = Path(tempfile.mkdtemp(prefix="bench-warm-", dir=args.tmpdir))
        self.sandbox = bench.Sandbox(self.root, HERE.parent, sys.executable)
        index = self.sandbox.index_dir("warm")
        contracts = index / "contracts.json"
        bench.write_contracts(contracts, args.contracts)
        config = self.sandbox.write_config(index, contracts)
        os.environ.update(HOME=str(self.sandbox.home), SELF_DIRECT_CONFIG_FILE=str(config), SELF_DIRECT_LOCAL_ONLY="true")
        for name in list(os.environ):
            if name.startswith("SELF_DIRECT_") and name not in ("SELF_DIRECT_CONFIG_FILE", "SELF_DIRECT_LOCAL_ONLY"):
                del os.environ[name]
        os.environ.pop("CODEX_SESSIONS_DIR", None)
        if args.shadow_every is not None:
            os.environ["SELF_DIRECT_AUDIT_SHADOW_EVERY"] = str(args.shadow_every)
        if args.min_events is not None:
            os.environ["SELF_DIRECT_AUDIT_INCREMENTAL_MIN_EVENTS"] = str(args.min_events)
        from self_directing_mcp import agy_hooks, claude_hooks
        from self_directing_mcp.engine import SelfDirectEngine
        from self_directing_mcp.settings_core import load_hook_settings
        self.hooks = {"claude": claude_hooks, "agy": agy_hooks}
        self.settings = load_hook_settings()
        # An older checkout has no ``defer_derived``: the script still runs against it (a before/after comparison).
        deferring = args.defer and "defer_derived" in inspect.signature(SelfDirectEngine.__init__).parameters
        self.engine = SelfDirectEngine(self.settings, **({"defer_derived": True} if deferring else {}))
        self.deferring = deferring
        size = int(args.session_mb * 1_000_000)
        self.sessions = {
            "claude": (bench.Transcript("claude", bench.claude_path(self.sandbox.projects, SESSION_CLAUDE), SESSION_CLAUDE),
                       SESSION_CLAUDE, bench.claude_payload),
            "agy": (bench.Transcript("agy", bench.agy_path(self.sandbox.agy_app, SESSION_AGY), SESSION_AGY),
                    SESSION_AGY, bench.agy_payload)}
        for transcript, _sid, _payload in self.sessions.values():
            transcript.create(size)
            if args.exact_size:
                trim_to_record(transcript.path, size)

    def request(self, host: str, event: str, append: int):
        transcript, sid, payload_of = self.sessions[host]
        transcript.append(append)
        payload = payload_of(event, sid, transcript.path, transcript.last_step + 1)
        started = time.perf_counter()
        response = self.hooks[host].run_hook(event, payload, engine=self.engine)
        return (time.perf_counter() - started) * 1000, response

    def events(self, host: str) -> int:
        _transcript, sid, _payload = self.sessions[host]
        provider = host
        return self.engine.store.session_metrics(sid.lower() if host == "agy" else sid, provider)[0]

    def close(self) -> None:
        self.engine.close()
        shutil.rmtree(self.root, ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--contracts", type=int, default=10)
    parser.add_argument("--session-mb", type=float, default=1.0)
    parser.add_argument("--exact-size", action="store_true",
                        help="cut the generated session back to --session-mb at a record boundary (for sessions of "
                             "tens or hundreds of events; the generator writes 64 steps at a time)")
    parser.add_argument("--append-bytes", default="0,4096", help="bytes appended before each request, comma list")
    parser.add_argument("--defer", action="store_true", help="build the engine as the daemon does (defer_derived=True)")
    parser.add_argument("--shadow-every", type=int, help="SELF_DIRECT_AUDIT_SHADOW_EVERY (default: the setting's default)")
    parser.add_argument("--min-events", type=int, help="SELF_DIRECT_AUDIT_INCREMENTAL_MIN_EVENTS")
    parser.add_argument("--tmpdir", help="directory for the index and the sessions (e.g. a RAM disk)")
    parser.add_argument("--only", default="", help="comma list of host:event cells (default: all four)")
    parser.add_argument("--profile", default="", help="host:event:bytes, e.g. claude:Stop:4096")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--spans", default="", help="host:event:bytes (printed), or 'all' (kept in the JSON report)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    appends = [int(value) for value in args.append_bytes.split(",")]
    count_commits()
    lab = Lab(args)
    load_start = loadavg()
    spans_installed = False
    report: dict = {"machine": bench.machine_info(sys.executable), "runs": args.runs, "contracts": args.contracts,
                    "session_mb": args.session_mb, "defer": lab.deferring, "shadow_every": lab.settings.audit_shadow_every,
                    "min_events": lab.settings.audit_incremental_min_events, "cells": {}}
    try:
        for host, event in CELLS:
            if args.only and f"{host}:{event}" not in args.only.split(","):
                continue
            for _ in range(3):                                   # the first requests build the index and warm the caches
                lab.request(host, event, 0)
            report.setdefault("events", {})[f"{host}:{event}"] = lab.events(host)   # the session at the start of this cell
            for append in appends:
                key = f"{host}:{event}:{append}"
                profiler = cProfile.Profile() if args.profile == key else None
                spanned = bool(args.spans) and args.spans in ("all", key)
                if spanned:
                    if not spans_installed:
                        install_spans()
                        spans_installed = True
                    SPANS.clear()
                COMMITS["n"] = 0
                samples = []
                for _ in range(args.runs):
                    if profiler is not None:
                        profiler.enable()
                    elapsed, _response = lab.request(host, event, append)
                    if profiler is not None:
                        profiler.disable()
                    samples.append(elapsed)
                cell = {"p50": round(statistics.median(samples), 2), "p95": round(bench.percentile(samples, 0.95), 2),
                        "min": round(min(samples), 2), "commits_per_request": round(COMMITS["n"] / args.runs, 2)}
                report["cells"][key] = cell
                print(f"{key:26s} p50 {cell['p50']:7.1f}  p95 {cell['p95']:7.1f}  min {cell['min']:7.1f} ms   "
                      f"durable commits/request {cell['commits_per_request']:4.1f}", flush=True)
                if profiler is not None:
                    for order in ("cumulative", "tottime"):
                        stream = io.StringIO()
                        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(order).print_stats(args.top)
                        print(stream.getvalue())
                if spanned:
                    report.setdefault("spans", {})[key] = {label: [round(calls / args.runs, 2), round(total / args.runs * 1000, 3)]
                                                           for label, (calls, total) in SPANS.items()}
                    if args.spans != "all":
                        report_spans(args.runs, key)
        incremental_on = lab.settings.audit_incremental
        verdicts = {cell: ("incremental" if incremental_on and count >= lab.settings.audit_incremental_min_events
                           else "full evaluation") for cell, count in report.get("events", {}).items()}
        print(f"stored events at the start of each cell: {report.get('events')}; "
              f"SELF_DIRECT_AUDIT_INCREMENTAL_MIN_EVENTS={lab.settings.audit_incremental_min_events} -> {verdicts}; "
              f"shadow every {lab.settings.audit_shadow_every}; engine {'deferring' if lab.deferring else 'synchronous'}; "
              f"load average {load_start[0]:.2f} -> {loadavg()[0]:.2f}")
        report["load_average"] = [round(value, 2) for value in (*load_start, *loadavg())]
        report["modes"] = verdicts
    finally:
        lab.close()
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown:
        lines = ["| cell | p50 ms | p95 ms | min ms | durable commits / request |", "|---|---|---|---|---|"]
        lines += [f"| {key} | {c['p50']} | {c['p95']} | {c['min']} | {c['commits_per_request']} |"
                  for key, c in report["cells"].items()]
        args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
