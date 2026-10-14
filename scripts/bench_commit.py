#!/usr/bin/env python3
"""Commit latency and crash survival of the authority databases under three journal modes. Synthetic data only.

The question (issue #18, item R1): does write-ahead logging with ``synchronous=FULL`` make the commits of the session
index (``meta.sqlite``) and the AGY receipts database (``agy-receipts.sqlite``) clearly faster than the rollback
journal with ``synchronous=FULL``? Only a clear win (the roundtable's rule: at least 25 % lower p50 on the append
commit on ext4) justifies WAL as a default. This script measures on the real code paths, not on a toy table:

* ``append`` : ``index.ingest.ingest_session`` of a synthetic Codex transcript that grew by a few KB, the way a hook
  syncs it (``derived=False``, no embedding). Reported: the whole call, ``SessionStore.commit_events`` (events, cursor,
  block table and audit bookkeeping in one transaction) and the ``commit()`` call inside it.
* ``receipt``: ``agy_receipts.ReceiptStore.record`` (``PreToolUse`` / ``PostToolUse`` of a synthetic AGY transcript).
* ``hookstate``: ``agy_hook_state.HookState.record_error``. This database is NOT an authority record and is not
  switched to WAL by the package; it is measured only to show what WAL would have bought there.

Two process models for each, because the default install runs a fresh process per hook while the experimental daemon
keeps the engine (and its connections) open:

* ``resident``: one store object and its connections stay open across samples (the daemon).
* ``fresh``   : every sample opens the store, does one operation and closes it (a command hook). In WAL mode the last
  connection to close checkpoints the log, so this is where a hidden cost would show.

Three modes, sampled round-robin in random order so slow drift hits all of them equally:

* ``delete_full`` : rollback journal (DELETE), ``synchronous=FULL``  (the behaviour before WAL)
* ``wal_full``    : write-ahead log, ``synchronous=FULL``            (the candidate; what ``journal.py`` implements)
* ``wal_keep``    : EXPERIMENT, not in the package: ``wal_full`` where a connection sets
  ``SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE`` and a fresh store does not checkpoint when it closes, to see whether skipping the
  second round of ``fsync`` at the end of a short-lived process pays. It does not: the log grows to the autocheckpoint
  threshold and every next process must recover it. Kept here so the question has an answer.
* ``wal_normal``  : write-ahead log, ``synchronous=NORMAL``          (REFERENCE ONLY, NOT SHIPPABLE: after power loss it
  can lose commits that were acknowledged; this package never sets it for authority records)

File systems: the scratch directory (ext4 on the development machine), ``/dev/shm`` (tmpfs: the CPU-only floor) and an
optional ``--slow-dir`` (the project's 9p/drvfs-backed checkout on WSL). On the slow directory only ``delete_full``
runs: ``journal.choose_journal`` denies WAL there, and the script does not override that. The file system type that was
found is printed with every cell.

``--kill`` adds the crash test: a writer process loops over real commits (an index append and a receipt record) and
prints an id after each one returns; the script sends it ``SIGKILL`` after a random number of acknowledgements, then
checks ``PRAGMA integrity_check`` and that every acknowledged id is present. That shows survival of a killed process,
which every mode passes by design; it says nothing about power loss or an operating-system crash (data still in the
page cache, which is what ``synchronous`` is about, survives a killed process in every mode).

    python scripts/bench_commit.py [--samples 200] [--warmup 10] [--kill 50] [--tmpdir DIR] [--shm DIR]
                                   [--slow-dir DIR] [--only append,receipt,hookstate] [--out report.json]
                                   [--markdown report.md]

Windows could not be measured by the author; the Windows branch of ``journal.choose_journal`` is covered by tests that
fake the Win32 calls and by the Windows CI job.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))          # this checkout, whatever is installed
spec = importlib.util.spec_from_file_location("bench_hooks_for_commit", HERE / "bench_hooks.py")
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)

from self_directing_mcp import journal                                      # noqa: E402
from self_directing_mcp.agy_hook_state import HookState                     # noqa: E402
from self_directing_mcp.agy_receipts import ReceiptStore                    # noqa: E402
from self_directing_mcp.index import ingest                                 # noqa: E402
from self_directing_mcp.index.deferred import DerivedBuffer                 # noqa: E402
from self_directing_mcp.index.sparse import SparseIndex                     # noqa: E402
from self_directing_mcp.schemas import Chunk                                # noqa: E402

MODES = ("delete_full", "wal_full", "wal_keep", "wal_normal")
NOT_SHIPPABLE = "wal_normal"
POLICY = {"delete_full": "delete", "wal_full": "auto", "wal_keep": "auto", "wal_normal": "auto"}
POLICY["wal_keep"] = "auto"
SID = "33333333-2222-3333-4444-555555555501"
AGY_SID = "11111111-1111-4111-8111-111111111101"
APPEND_BYTES = 3000                  # about two steps of the synthetic transcript, a typical hook's worth

# --- instrumentation: timings of the real calls, taken from outside them -------------------------------------------
RECORD: dict[str, list] = {"commit": [], "events": []}
STATE = {"normal": False, "keep": False}   # applied to connections opened while set (reference / experiment cells only)


class TimedConnection(sqlite3.Connection):
    """Times the explicit ``commit()`` calls the stores make (the durable part of a write)."""

    def commit(self):
        pending = self.in_transaction
        started = time.perf_counter()
        try:
            return super().commit()
        finally:
            if pending:
                RECORD["commit"].append((time.perf_counter() - started) * 1000)


def install() -> None:
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        kwargs.setdefault("factory", TimedConnection)
        connection = real_connect(*args, **kwargs)
        if STATE["normal"]:
            connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    sqlite3.connect = connect
    real_apply = journal.apply_journal

    def apply(connection, path, policy=None, **kwargs):
        state = real_apply(connection, path, policy, **kwargs)
        if STATE["normal"]:
            connection.execute("PRAGMA synchronous=NORMAL")      # after the package set FULL: reference cell only
        if STATE["keep"] and state.mode == "wal":
            connection.setconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True)     # experiment only
        return state

    journal.apply_journal = apply
    real_commit_events = ingest.SessionStore.commit_events

    def commit_events(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return real_commit_events(self, *args, **kwargs)
        finally:
            RECORD["events"].append((time.perf_counter() - started) * 1000)

    ingest.SessionStore.commit_events = commit_events


@contextlib.contextmanager
def mode_flags(mode: str):
    STATE["normal"], STATE["keep"] = mode == "wal_normal", mode == "wal_keep"
    try:
        yield
    finally:
        STATE["normal"] = STATE["keep"] = False


def timed(call):
    RECORD["commit"].clear()
    RECORD["events"].clear()
    started = time.perf_counter()
    call()
    total = (time.perf_counter() - started) * 1000
    return {"op_ms": total, "commit_ms": sum(RECORD["commit"]), "events_ms": sum(RECORD["events"]),
            "commits": len(RECORD["commit"])}


# --- fixtures (one per mode and operation) -----------------------------------------------------------------------
class Append:
    def __init__(self, root: Path, mode: str, fresh: bool) -> None:
        self.mode, self.fresh, self.root = mode, fresh, root
        self.index = root / "index"
        self.index.mkdir(parents=True)
        path = bench.codex_path(root / "sessions", SID)
        self.transcript = bench.Transcript("codex", path, SID)
        self.transcript.create(30_000)
        self.sparse = SparseIndex(self.index / "sparse.sqlite")
        self.store = None
        with mode_flags(mode):
            if not fresh:
                self.store = ingest.SessionStore(self.index / "meta.sqlite", journal=POLICY[mode])
                self._check(self.store)
        self.sample(record=False)

    def _check(self, store) -> None:
        if (store.journal.mode == "wal") != (self.mode != "delete_full"):
            raise RuntimeError(f"{self.mode}: the database is in {store.journal.mode} mode ({store.journal.choice.reason})")

    def _sync(self, store) -> None:
        ingest.ingest_session(path=self.transcript.path, session_id=SID, store=store, sparse=self.sparse, dense=None,
                              embedder=None, provider="codex", embed=False, derived=False)

    def sample(self, record: bool = True):
        self.transcript.append(APPEND_BYTES)
        with mode_flags(self.mode):
            if self.fresh:
                def operation():
                    store = ingest.SessionStore(self.index / "meta.sqlite", journal=POLICY[self.mode])
                    try:
                        self._sync(store)
                    finally:
                        if self.mode == "wal_keep":
                            store._conn.close()          # the experiment: no checkpoint at the end of the process
                        else:
                            store.close()
                return timed(operation)
            return timed(lambda: self._sync(self.store))

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
        self.sparse._conn.close()


class Receipt:
    def __init__(self, root: Path, mode: str, fresh: bool) -> None:
        self.mode, self.fresh, self.root = mode, fresh, root
        self.index = root / "index"
        self.index.mkdir(parents=True)
        self.transcript = bench.Transcript("agy", bench.agy_path(root / "agy", AGY_SID), AGY_SID)
        self.transcript.create(40_000)
        self.store = ReceiptStore(self.index, held=None if fresh else DerivedBuffer(), journal=POLICY[mode])
        self.step = None
        self.phase = "PreToolUse"
        with mode_flags(mode):
            self.sample(record=False)
            self.sample(record=False)
            if not fresh:
                self.store.authority_token(AGY_SID)           # opens the persistent read-only connection
        wal = (self.index / "agy-receipts.sqlite-wal").exists()
        state = self.store.journal_report()
        if (state["mode"] == "wal") != (mode != "delete_full"):
            raise RuntimeError(f"{mode}: receipts database is in {state['mode']} mode ({state['reason']}, wal file {wal})")

    def sample(self, record: bool = True):
        if self.phase == "PreToolUse":
            self.step = self.transcript.last_step
        call = {"name": "run_command", "args": {"CommandLine": json.dumps("echo step-%d" % self.step)}}
        event = self.phase
        with mode_flags(self.mode):
            out = timed(lambda: self.store.record(AGY_SID, self.transcript.path, self.step, call, event,
                                                  error="" if event == "PostToolUse" else None))
        if event == "PreToolUse":
            self.transcript.append(1)
            self.phase = "PostToolUse"
        else:
            self.phase = "PreToolUse"
        return out

    def close(self) -> None:
        ReceiptStore.release(self.index)


class HookStateOp:
    """Not an authority record and not switched by the package: WAL is set by hand to show what it would buy."""

    def __init__(self, root: Path, mode: str, fresh: bool) -> None:
        self.mode, self.fresh = mode, fresh
        self.index = root / "index"
        self.index.mkdir(parents=True)
        self.state = HookState(self.index)
        self.state.record_error("check_failed", "PreToolUse", "Warmup")
        if mode != "delete_full":
            with contextlib.closing(sqlite3.connect(self.state.path)) as connection:
                connection.execute("PRAGMA journal_mode=WAL")      # the package never does this to hook state
        self.n = 0

    def sample(self, record: bool = True):
        self.n += 1
        with mode_flags(self.mode):
            return timed(lambda: self.state.record_error("check_failed", "PreToolUse", f"E{self.n % 5}"))

    def close(self) -> None:
        pass


OPERATIONS = {
    "append": (Append, (("resident", False), ("fresh", True))),
    "receipt": (Receipt, (("resident", False), ("fresh", True))),
    "hookstate": (HookStateOp, (("fresh", True),)),
}


# --- statistics --------------------------------------------------------------------------------------------------
def percentile(values: list, q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(values: list) -> dict:
    return {"n": len(values), "p50": percentile(values, 0.50), "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99), "min": min(values), "max": max(values)}


def loadavg() -> list:
    return [round(x, 2) for x in os.getloadavg()] if hasattr(os, "getloadavg") else []


def filesystem(path: Path) -> dict:
    """The file system type that holds ``path``, as the package's own probe reads it."""
    choice = journal.choose_journal(path)
    return {"choice": choice.mode, "reason": choice.reason}


def run_group(label: str, directory: Path, operation: str, kind: str, fresh: bool, modes, samples: int, warmup: int,
              rng: random.Random) -> dict:
    cls = OPERATIONS[operation][0]
    root = Path(tempfile.mkdtemp(prefix=f"bench-commit-{operation}-{kind}-", dir=directory))
    fixtures = {}
    result = {"fs": label, "operation": operation, "model": kind, "load_start": loadavg(), "modes": {}}
    try:
        for mode in modes:
            fixtures[mode] = cls(root / mode, mode, fresh)
        for round_number in range(warmup + samples):
            order = list(modes)
            rng.shuffle(order)
            for mode in order:
                row = fixtures[mode].sample()
                if round_number >= warmup:
                    for key, value in row.items():
                        result["modes"].setdefault(mode, {}).setdefault(key, []).append(value)
        result["load_end"] = loadavg()
        for mode, columns in result["modes"].items():
            result["modes"][mode] = {key: summarize(values) for key, values in columns.items()
                                     if key != "commits" or True}
            result["modes"][mode]["commits_per_op"] = sum(columns["commits"]) / len(columns["commits"])
    finally:
        for fixture in fixtures.values():
            with contextlib.suppress(Exception):
                fixture.close()
        shutil.rmtree(root, ignore_errors=True)
    return result


# --- the crash test ----------------------------------------------------------------------------------------------
def worker(mode: str, index: Path, run: int) -> None:
    """Commit forever; print an id once its commit has returned (flushed, so the parent sees only acknowledged ones).

    Three index appends for every receipt, so that most of the time is spent inside a commit."""
    install_worker(mode)
    store = ingest.SessionStore(index / "meta.sqlite", journal=POLICY[mode])
    receipts = ReceiptStore(index, journal=POLICY[mode])
    source = index / "kill-source.jsonl"
    transcript = index / "kill-transcript.jsonl"
    n = 0
    while True:
        for _ in range(3):
            n += 1
            base = run * 1_000_000 + n * 10
            chunks = [Chunk(chunk_id=f"k{run}-{n}-{j}", session_id="kill-test", provider="codex", kind="turn",
                            text=f"synthetic event {run}-{n}-{j}", content_hash=f"h{run}-{n}-{j}",
                            byte_start=base + j, byte_end=base + j + 1, line_start=1, line_end=1,
                            meta={"sub_index": 0}) for j in range(3)]
            store.commit_events(chunks, session_id="kill-test", provider="codex", path=source, offset=base + 3,
                                digest="d" * 64)
            print(f"A {run}-{n}", flush=True)
        step = run * 100_000 + n
        receipts.record("kill-session", transcript, step,
                        {"name": "run_command", "args": {"CommandLine": f'"echo {n}"'}}, "PreToolUse")
        print(f"R {step}", flush=True)


def install_worker(mode: str) -> None:
    STATE["normal"], STATE["keep"] = mode == NOT_SHIPPABLE, mode == "wal_keep"
    install()


def kill_test(directory: Path, mode: str, runs: int, rng: random.Random) -> dict:
    root = Path(tempfile.mkdtemp(prefix=f"bench-kill-{mode}-", dir=directory))
    index = root / "index"
    index.mkdir()
    (index / "kill-source.jsonl").write_text("synthetic\n")
    (index / "kill-transcript.jsonl").write_text(bench.agy_head())
    acked_events, acked_steps = [], []
    outcome = {"mode": mode, "runs": runs, "acknowledged_appends": 0, "acknowledged_receipts": 0, "lost_appends": [],
               "lost_receipts": [], "integrity_failures": [], "reopen_failures": [], "killed_with_wal_file": 0,
               "killed_with_hot_journal": 0, "committed_not_acknowledged": 0}
    try:
        for run in range(1, runs + 1):
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", mode, str(index),
                                        str(run)], stdout=subprocess.PIPE, text=True,
                                       env={**os.environ, "PYTHONPATH": str(HERE.parent / "src")})
            wanted = rng.randint(2, 60)
            seen = 0
            for line in process.stdout:
                kind, _, ident = line.strip().partition(" ")
                (acked_events if kind == "A" else acked_steps).append(ident)
                seen += 1
                if seen >= wanted:
                    break
            time.sleep(rng.uniform(0, 0.04))             # land anywhere in the next commit, not just right after an ack
            process.send_signal(signal.SIGKILL)          # no cleanup, no atexit, no flush of anything buffered in-process
            process.wait()
            process.stdout.close()
            meta, receipts = index / "meta.sqlite", index / "agy-receipts.sqlite"
            outcome["killed_with_wal_file"] += int(any(Path(str(p) + "-wal").exists() and Path(str(p) + "-wal").stat().st_size
                                                       for p in (meta, receipts)))
            outcome["killed_with_hot_journal"] += int(any(Path(str(p) + "-journal").exists() for p in (meta, receipts)))
            check_after_kill(index, run, acked_events, acked_steps, outcome, mode)
        outcome["acknowledged_appends"], outcome["acknowledged_receipts"] = len(acked_events), len(acked_steps)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return outcome


def check_after_kill(index: Path, run: int, acked_events: list, acked_steps: list, outcome: dict, mode: str) -> None:
    with contextlib.closing(sqlite3.connect(index / "meta.sqlite")) as connection:
        verdict = connection.execute("PRAGMA integrity_check").fetchall()
        if verdict != [("ok",)]:
            outcome["integrity_failures"].append((run, "meta", verdict[:2]))
        present = {row[0] for row in connection.execute("SELECT chunk_id FROM session_events WHERE chunk_id LIKE 'k%'")}
    with contextlib.closing(sqlite3.connect(index / "agy-receipts.sqlite")) as connection:
        verdict = connection.execute("PRAGMA integrity_check").fetchall()
        if verdict != [("ok",)]:
            outcome["integrity_failures"].append((run, "receipts", verdict[:2]))
        steps = {row[0] for row in connection.execute("SELECT step FROM receipts WHERE session_id='kill-session'")}
    outcome["lost_appends"] = [ident for ident in acked_events
                               if not all(f"k{ident}-{j}" in present for j in range(3))]
    outcome["lost_receipts"] = [ident for ident in acked_steps if int(ident) not in steps]
    mine = {c for c in present if c.startswith(f"k{run}-") and c.endswith("-0")}
    outcome["committed_not_acknowledged"] += len(mine - {f"k{i}-0" for i in acked_events if i.startswith(f"{run}-")})
    try:                                                  # the package's own classes open it and see the same rows
        store = ingest.SessionStore(index / "meta.sqlite", journal=POLICY[mode])
        store.close()
        ReceiptStore(index, journal=POLICY[mode]).read_capture("kill-session")
    except Exception as exc:                              # noqa: BLE001 - reported, not hidden
        outcome["reopen_failures"].append((run, type(exc).__name__))


# --- report ------------------------------------------------------------------------------------------------------
def render(results: list, kills: list, meta: dict) -> str:
    lines = [f"commit latency, {meta['samples']} samples per cell (after {meta['warmup']} warm-up), interleaved; ms",
             f"python {meta['python']}, sqlite {meta['sqlite']}, load average at start {meta['load_start']}", ""]
    for group in results:
        lines.append(f"### {group['fs']} ({group['fs_probe']['choice']}: {group['fs_probe']['reason']}) - "
                     f"{group['operation']} / {group['model']}   load {group['load_start']} -> {group.get('load_end')}")
        lines.append("")
        lines.append("| metric | mode | p50 | p95 | p99 | max | commits/op |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for metric in ("op_ms", "events_ms", "commit_ms"):
            for mode in MODES:
                column = group["modes"].get(mode)
                if column is None or (metric == "events_ms" and group["operation"] != "append"):
                    continue
                if metric == "commit_ms" and group["operation"] == "hookstate":
                    continue
                s = column[metric]
                tag = " (reference only, not shippable)" if mode == NOT_SHIPPABLE else ""
                lines.append(f"| {metric.replace('_ms', '')} | {mode}{tag} | {s['p50']:.2f} | {s['p95']:.2f} | "
                             f"{s['p99']:.2f} | {s['max']:.2f} | {column['commits_per_op']:.1f} |")
        lines.append("")
    lines.append("### ratios of p50 (variant / delete_full; below 1.00 is faster)")
    lines.append("")
    lines.append("| fs | operation | model | variant | whole op | commit_events | commit() |")
    lines.append("|---|---|---|---|---:|---:|---:|")
    for group in results:
        delete = group["modes"].get("delete_full")
        if delete is None:
            continue
        for variant in ("wal_full", "wal_keep"):
            wal = group["modes"].get(variant)
            if wal is None:
                continue

            def ratio(metric):
                return f"{wal[metric]['p50'] / delete[metric]['p50']:.2f}" if delete[metric]["p50"] else "n/a"
            events = ratio("events_ms") if group["operation"] == "append" else "-"
            commit = ratio("commit_ms") if group["operation"] != "hookstate" else "-"
            lines.append(f"| {group['fs']} | {group['operation']} | {group['model']} | {variant} | {ratio('op_ms')} | "
                         f"{events} | {commit} |")
    lines.append("")
    if kills:
        lines.append("### kill -9 test (process kill, NOT power loss)")
        lines.append("")
        lines.append("| mode | kills | acked appends | acked receipts | lost appends | lost receipts | integrity failures | "
                     "reopen failures | committed, not acknowledged | kills with non-empty -wal | kills with hot -journal |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for k in kills:
            lines.append(f"| {k['mode']} | {k['runs']} | {k['acknowledged_appends']} | {k['acknowledged_receipts']} | "
                         f"{len(k['lost_appends'])} | {len(k['lost_receipts'])} | {len(k['integrity_failures'])} | "
                         f"{len(k['reopen_failures'])} | {k['committed_not_acknowledged']} | {k['killed_with_wal_file']} | "
                         f"{k['killed_with_hot_journal']} |")
        lines.append("")
        lines.append("A killed process keeps everything the operating system already holds, so this passes in every mode "
                     "by design. It is not a test of power-loss durability, which is what `synchronous=FULL` versus "
                     "`NORMAL` decides.")
    return "\n".join(lines)


def decision(results: list) -> str:
    """The roundtable's rule for making WAL the default: p50 of the append commit on ext4 at least 25 % below
    DELETE+FULL on the real paths.

    "The append commit" is measured twice, because a short-lived process pays for the write-ahead log when it closes:
    ``commit_events`` (the transaction itself) and the whole append operation (open, commit, close; what a command hook
    experiences). Default-on needs both to meet the threshold in both process models. ``wal_keep`` is the experiment
    that skips the close-time checkpoint; it is not in the package, so it never counts."""
    lines = ["decision rule (append on ext4, p50 reduction of wal_full against delete_full; default-on needs >= 25 % for "
             "the commit AND for the whole operation, in both process models):"]
    verdict = []
    for group in results:
        if group["fs"] != "ext4" or group["operation"] != "append":
            continue
        delete = group["modes"].get("delete_full")
        for variant in ("wal_full", "wal_keep"):
            wal = group["modes"].get(variant)
            if not delete or not wal:
                continue
            for metric in ("events_ms", "op_ms"):
                gain = 1 - wal[metric]["p50"] / delete[metric]["p50"]
                counts = variant == "wal_full"
                if counts:
                    verdict.append(gain >= 0.25)
                lines.append(f"  {group['model']:9s} {variant:9s} {'commit' if metric == 'events_ms' else 'whole op':9s} "
                             f"{gain * 100:6.1f} %  {'meets' if gain >= 0.25 else 'does NOT meet'}"
                             f"{'' if counts else '   (experiment, not counted)'}")
    lines.append("RULE: " + ("MET, WAL may be the default" if verdict and all(verdict) else
                             "NOT MET, WAL stays opt-in (SELF_DIRECT_SQLITE_JOURNAL=auto)"))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--kill", type=int, default=0, help="kill -9 runs per mode (0: skip the crash test)")
    parser.add_argument("--tmpdir", type=Path, default=None, help="the ext4 (or other local) scratch directory")
    parser.add_argument("--shm", type=Path, default=Path("/dev/shm"), help="a tmpfs directory (skipped if missing)")
    parser.add_argument("--slow-dir", type=Path, default=None, help="a 9p/drvfs/NFS directory: DELETE only")
    parser.add_argument("--only", default="append,receipt,hookstate")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--worker", nargs=3, metavar=("MODE", "INDEX", "RUN"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args.worker[0], Path(args.worker[1]), int(args.worker[2]))
        return 0
    install()
    rng = random.Random(args.seed)
    targets = [("ext4", Path(tempfile.mkdtemp(prefix="bench-commit-", dir=args.tmpdir)), MODES)]
    if args.shm and args.shm.is_dir():
        targets.append(("tmpfs", Path(tempfile.mkdtemp(prefix="bench-commit-", dir=args.shm)), MODES))
    if args.slow_dir:
        args.slow_dir.mkdir(parents=True, exist_ok=True)
        targets.append(("slow", Path(tempfile.mkdtemp(prefix="bench-commit-", dir=args.slow_dir)), ("delete_full",)))
    meta = {"samples": args.samples, "warmup": args.warmup, "python": sys.version.split()[0],
            "sqlite": sqlite3.sqlite_version, "load_start": loadavg()}
    results, kills = [], []
    try:
        for label, directory, modes in targets:
            probe = filesystem(directory)
            print(f"== {label}: {directory} -> {probe}", file=sys.stderr)
            for operation in args.only.split(","):
                for kind, fresh in OPERATIONS[operation][1]:
                    print(f"   {operation}/{kind} ...", file=sys.stderr, flush=True)
                    cell_modes = tuple(m for m in modes if not (operation == "hookstate" and m == "wal_keep"))
                    group = run_group(label, directory, operation, kind, fresh, cell_modes, args.samples, args.warmup, rng)
                    group["fs_probe"] = probe
                    results.append(group)
        if args.kill:
            for mode in MODES:
                print(f"== kill test {mode} x {args.kill}", file=sys.stderr, flush=True)
                kills.append(kill_test(targets[0][1], mode, args.kill, rng))
    finally:
        for _label, directory, _modes in targets:
            shutil.rmtree(directory, ignore_errors=True)
    text = render(results, kills, meta) + "\n\n" + decision(results)
    print(text)
    if args.markdown:
        args.markdown.write_text(text + "\n", encoding="utf-8")
    if args.out:
        args.out.write_text(json.dumps({"meta": meta, "results": results, "kills": kills}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
