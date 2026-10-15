#!/usr/bin/env python3
"""Benchmark of the resident daemon (experimental): cold hook processes through a warm daemon versus standalone,
and idle MCP proxies plus one daemon versus the same number of standalone servers. Linux only. Synthetic data only.

It reuses the synthetic sessions, sandbox and process-measurement helpers of ``scripts/bench_hooks.py`` and runs under an
isolated HOME, index, config file and ``XDG_RUNTIME_DIR``; it never reads a user's sessions, indexes, credentials or
runtime directory, and it stops every daemon it starts.

Measured, separately:

* ``hooks``: per host, per event, ``--runs`` fresh hook processes (the real ``scripts/claude_hook.py`` /
  ``scripts/agy_hook.py``) alternately standalone (flag unset) and through a warm daemon (``SELF_DIRECT_DAEMON=on``),
  on separate identically built indexes of the same synthetic session (``--session-mb``, default 1). Wall time, CPU
  and peak RSS of the client process come from the OS (``wait4``); the daemon's own processing time per hook comes from
  the durations in its log (event names, sizes and durations only). Every run must judge: a fail-open answer
  invalidates the cell.
* ``daemon``: the daemon's RSS/PSS after those hooks and after the MCP sessions below.
* ``servers``: ``--servers`` (12) idle MCP sessions after the handshake and one tool call, as standalone servers
  (flag unset) and as relays plus one daemon (flag on), all on one shared index as in production.

Targets are the roundtable's provisional numbers (``TARGETS``); the report states each result against its target and
says "met" or "NOT met"; nothing is gated.

    python scripts/bench_daemon.py [--runs 30] [--servers 12] [--out report.json] [--markdown report.md]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bench_hooks_for_daemon", HERE / "bench_hooks.py")
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)

TARGETS = {
    "hook_wall_p95_ms": ("New-process hook wall time through the daemon, p95", 40.0, "ms"),
    "daemon_side_p95_ms": ("Daemon-side processing time, p95", 15.0, "ms"),
    "daemon_overhead_p95_ms": ("Daemon-side processing time without the judgement, p95", 15.0, "ms"),
    "daemon_rss_mb": ("Daemon RSS after use", 120.0, "MB"),
    "proxy_rss_mb": ("MCP relay RSS (each)", 15.0, "MB"),
    "total_rss_mb": ("12 relays + 1 daemon, total RSS", 300.0, "MB"),
}
SESSION_CLAUDE = "aaaaaaaa-1111-4111-8111-aaaaaaaa0001"
SESSION_AGY = "11111111-1111-4111-8111-111111111111"


class Lab:
    def __init__(self, root: Path, repo: Path, python: str, session_mb: float, contracts: int) -> None:
        self.root, self.repo, self.python = root, repo, python
        self.sandbox = bench.Sandbox(root, repo, python)
        self.xdg = root / "xdg"
        self.xdg.mkdir(mode=0o700)
        os.chmod(self.xdg, 0o700)
        self.session_bytes = int(session_mb * 1_000_000)
        self.contracts = contracts
        self.daemon_proc: subprocess.Popen | None = None
        self.claude = bench.Transcript("claude", bench.claude_path(self.sandbox.projects, SESSION_CLAUDE), SESSION_CLAUDE)
        self.agy = bench.Transcript("agy", bench.agy_path(self.sandbox.agy_app, SESSION_AGY), SESSION_AGY)
        self.claude.create(self.session_bytes)
        self.agy.create(self.session_bytes)

    @property
    def runtime_dir(self) -> Path:
        return self.xdg / "tsukkomi"

    def index(self, name: str) -> tuple[Path, Path]:
        index = self.sandbox.index_dir(name)
        contracts = index / "contracts.json"
        bench.write_contracts(contracts, self.contracts)
        return index, self.sandbox.write_config(index, contracts)

    def env(self, config: Path, daemon: bool, **extra) -> dict:
        env = self.sandbox.env(config, SELF_DIRECT_LOCAL_ONLY="true", XDG_RUNTIME_DIR=str(self.xdg), **extra)
        if daemon:
            env["SELF_DIRECT_DAEMON"] = "on"
        return env

    def start_daemon(self, config: Path) -> int:
        env = {k: v for k, v in self.sandbox.env(config).items() if k in ("PATH", "HOME", "PYTHONPATH", "LANG", "TMPDIR")}
        env["XDG_RUNTIME_DIR"] = str(self.xdg)
        self.daemon_proc = subprocess.Popen([self.python, "-m", "self_directing_mcp.daemon"], env=env,
                                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL, start_new_session=True, cwd=str(self.root))
        pid_file = self.runtime_dir / "daemon.pid"
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.daemon_proc.poll() is not None:
                raise RuntimeError("the daemon exited at startup")
            if (self.runtime_dir / "daemon.sock").exists() and pid_file.exists():
                return int(pid_file.read_text())
            time.sleep(0.02)
        raise RuntimeError("the daemon did not start")

    def stop_spawned_daemon(self) -> None:
        """A daemon that a relay or a hook started (not ``daemon_proc``): stop it through its own pid file, so that
        removing the runtime directory never leaves one running."""
        try:
            pid = int((self.runtime_dir / "daemon.pid").read_text())
            with open(f"/proc/{pid}/cmdline", "rb") as stream:
                if b"self_directing_mcp.daemon" not in stream.read():
                    return
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 20
            while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
        except (OSError, ValueError):
            pass

    def stop_daemon(self) -> None:
        self.stop_spawned_daemon()
        if self.daemon_proc is None:
            return
        try:
            self.daemon_proc.send_signal(signal.SIGTERM)       # a graceful drain: socket and pid file are removed
            self.daemon_proc.wait(timeout=20)
        except Exception:
            self.daemon_proc.kill()
            self.daemon_proc.wait()
        self.daemon_proc = None

    def daemon_log_ms(self, event: str, field: str = "ms") -> list[float]:
        """Durations the daemon logged for this event: ``ms`` is the whole request (queueing, settings, judgement,
        serialization), ``run_ms`` the part inside the lane thread (settings and judgement)."""
        try:
            lines = (self.runtime_dir / "daemon.log").read_text().splitlines()
        except OSError:
            return []
        records = (json.loads(line) for line in lines)
        return [r[field] for r in records if r.get("ev") == "hook" and r.get("event") == event and field in r]


def measure_hooks(lab: Lab, args, log=print) -> dict:
    report: dict = {}
    for host, transcript, sid in (("claude", lab.claude, SESSION_CLAUDE), ("agy", lab.agy, SESSION_AGY)):
        events = [e for e in args.events.split(",") if e in (bench.CLAUDE_EVENTS if host == "claude" else bench.AGY_EVENTS)]
        index_s, config_s = lab.index(f"{host}-standalone")
        index_d, config_d = lab.index(f"{host}-daemon")
        sides = {"standalone": (index_s, lab.env(config_s, False)), "daemon": (index_d, lab.env(config_d, True))}
        if lab.daemon_proc is None:
            lab.start_daemon(config_d)
        report[host] = {}
        payload_of = bench.claude_payload if host == "claude" else bench.agy_payload
        for event, append in [(e, a) for e in events for a in args.append_list]:
            key = event if append == args.append_list[-1] and len(args.append_list) == 1 else f"{event} (+{append} B)"
            samples = {name: {"wall": [], "cpu": [], "rss": [], "labels": {}, "failed": 0} for name in sides}
            body = json.dumps(payload_of(event, sid, transcript.path, transcript.last_step + 1)).encode()
            for name, (index, env) in sides.items():                      # one discarded warm-up per side
                bench.run_process(bench.hook_argv(lab.sandbox, host, event, index), body, env, lab.root, 120)
            for run in range(args.runs):
                transcript.append(append)
                body = json.dumps(payload_of(event, sid, transcript.path, transcript.last_step + 1)).encode()
                for name in (("standalone", "daemon") if run % 2 == 0 else ("daemon", "standalone")):
                    index, env = sides[name]
                    result = bench.run_process(bench.hook_argv(lab.sandbox, host, event, index), body, env, lab.root, 120)
                    bag = samples[name]
                    bag["wall"].append(result["wall_s"] * 1000)
                    bag["cpu"].append((result["cpu_s"] or 0) * 1000)
                    bag["rss"].append((result["peak_rss"] or 0) / 1048576)
                    verdict = bench.judgement(host, event, result["stdout"], result["stderr"])
                    bag["labels"][verdict["label"]] = bag["labels"].get(verdict["label"], 0) + 1
                    bag["failed"] += verdict["failed_open"] or result["returncode"] != 0
            side_ms = lab.daemon_log_ms(event)[-args.runs:]
            run_ms = lab.daemon_log_ms(event, "run_ms")[-args.runs:]
            overhead_ms = [a - b for a, b in zip(side_ms, run_ms)]
            cell = {name: {"wall_ms": bench.summarize(bag["wall"]), "cpu_ms": bench.summarize(bag["cpu"]),
                           "peak_rss_mb": bench.summarize(bag["rss"], 1), "decisions": bag["labels"],
                           "failed_open": bag["failed"]} for name, bag in samples.items()}
            for name, bag in samples.items():
                cell[name]["wall_ms"]["p99"] = round(bench.percentile(bag["wall"], 0.99), 1)
            cell["daemon"]["daemon_side_ms"] = bench.summarize(side_ms, 2)
            cell["daemon"]["daemon_side_ms"]["p99"] = round(bench.percentile(side_ms, 0.99), 2) if side_ms else None
            cell["daemon"]["daemon_run_ms"] = bench.summarize(run_ms, 2)
            cell["daemon"]["daemon_overhead_ms"] = bench.summarize(overhead_ms, 2)
            report[host][key] = cell
            log(f"  {host:6s} {key:22s} standalone wall p50/p95/p99 {cell['standalone']['wall_ms']['p50']:7.1f}/"
                f"{cell['standalone']['wall_ms']['p95']:7.1f}/{cell['standalone']['wall_ms']['p99']:7.1f} ms | daemon wall "
                f"p50/p95/p99 {cell['daemon']['wall_ms']['p50']:6.1f}/{cell['daemon']['wall_ms']['p95']:6.1f}/"
                f"{cell['daemon']['wall_ms']['p99']:6.1f} ms, daemon-side p50/p95/p99 {cell['daemon']['daemon_side_ms']['p50']}/"
                f"{cell['daemon']['daemon_side_ms']['p95']}/{cell['daemon']['daemon_side_ms']['p99']} ms, client rss max "
                f"{cell['daemon']['peak_rss_mb']['max']} MB "
                f"{samples['daemon']['labels']}{'  FAILED OPEN' if samples['daemon']['failed'] else ''}")
    return report


class McpProc:
    """One MCP stdio process (standalone server or relay) driven by raw JSON-RPC."""

    def __init__(self, lab: Lab, config: Path, daemon: bool, cwd: Path) -> None:
        self.proc = subprocess.Popen([lab.python, "-m", "self_directing_mcp"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=str(cwd),
                                     env=lab.env(config, daemon, SELF_DIRECT_INTEGRITY_BACKGROUND_VERIFY="true"))
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for raw in self.proc.stdout:
            self.lines.put(raw)

    def send(self, message: dict) -> None:
        self.proc.stdin.write((json.dumps(message) + "\n").encode())
        self.proc.stdin.flush()

    def wait_for(self, request_id: int, timeout: float = 120.0) -> dict:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                message = json.loads(self.lines.get(timeout=0.2))
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise RuntimeError("the MCP process exited early")
                continue
            if message.get("id") == request_id:
                return message
        raise TimeoutError(f"no answer to request {request_id}")

    def start(self) -> None:
        self.send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "bench", "version": "0"}}})
        self.wait_for(1)
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.wait_for(2)

    def use(self) -> None:
        self.send({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "audit_status", "arguments": {}}})
        self.wait_for(3)

    def stop(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def memory_row(pid: int) -> dict:
    row = bench.proc_status(pid)
    return {k: round(v / 1048576, 1) for k, v in row.items() if k in ("rss", "peak_rss", "pss")}


def measure_servers(lab: Lab, args, log=print) -> dict:
    index, config = lab.index("servers-shared")                   # one shared index, as production sessions use
    cwd = lab.root / "server-cwd"
    cwd.mkdir(exist_ok=True)
    out: dict = {"count": args.servers}
    for label, daemon in (("standalone", False), ("proxied", True)):
        if daemon:
            lab.stop_daemon()
            shutil.rmtree(lab.runtime_dir, ignore_errors=True)
        procs = [McpProc(lab, config, daemon, cwd) for _ in range(args.servers)]
        try:
            for proc in procs:
                proc.start()
            time.sleep(1.0)
            after_init = [memory_row(p.proc.pid) for p in procs]
            for proc in procs:
                proc.use()
            time.sleep(1.5)
            after_use = [memory_row(p.proc.pid) for p in procs]
            block = {"each_rss_mb": [r.get("rss") for r in after_use],
                     "rss_mb_total": round(sum(r.get("rss", 0) for r in after_use), 1),
                     "pss_mb_total": round(sum(r.get("pss", 0) for r in after_use), 1),
                     "rss_mb_max": max(r.get("rss", 0) for r in after_use),
                     "after_init_rss_mb_total": round(sum(r.get("rss", 0) for r in after_init), 1)}
            if daemon:
                pid = int((lab.runtime_dir / "daemon.pid").read_text())
                resident = memory_row(pid)
                block["daemon"] = resident
                block["total_with_daemon_rss_mb"] = round(block["rss_mb_total"] + resident.get("rss", 0), 1)
                block["total_with_daemon_pss_mb"] = round(block["pss_mb_total"] + resident.get("pss", 0), 1)
            out[label] = block
            log(f"  {label:10s} {args.servers} sessions: rss total {block['rss_mb_total']} MB (max {block['rss_mb_max']}), "
                f"pss total {block['pss_mb_total']} MB" + (f", daemon rss {block['daemon'].get('rss')} MB, total with "
                f"daemon {block['total_with_daemon_rss_mb']} MB (pss {block['total_with_daemon_pss_mb']})" if daemon else ""))
        finally:
            for proc in procs:
                proc.stop()
    return out


def evaluate(report: dict) -> list[dict]:
    rows = []

    def add(key, value, note="", label_suffix=""):
        label, target, unit = TARGETS[key]
        label += label_suffix
        rows.append({"metric": label, "target": f"<= {target:g} {unit}", "measured": None if value is None else f"{value:g} {unit}",
                     "status": "n/a" if value is None else ("met" if value <= target else "NOT met"), "note": note})

    hooks = report.get("hooks", {})
    names = {"claude": "Claude Code", "agy": "AGY"}
    for host, events in hooks.items():
        for key, cell in events.items():
            if key.split(" ")[0] not in ("PreToolUse", "UserPromptSubmit"):
                continue
            d = cell["daemon"]
            suffix = f" [{names[host]} {key}]"
            add("hook_wall_p95_ms", d["wall_ms"]["p95"], "the whole new process, this machine", suffix)
            add("daemon_side_p95_ms", d["daemon_side_ms"]["p95"], "from the daemon's log; includes the judgement", suffix)
            add("daemon_overhead_p95_ms", d["daemon_overhead_ms"]["p95"], "request time minus the lane thread's time", suffix)
    servers = report.get("servers", {})
    proxied = servers.get("proxied")
    if report.get("daemon_after_hooks"):
        add("daemon_rss_mb", report["daemon_after_hooks"].get("rss"), "after the hook runs above (two engines)", " [after hooks]")
    if proxied:
        add("daemon_rss_mb", proxied["daemon"].get("rss"), f"after {servers['count']} MCP sessions' first calls (one engine)", f" [{servers['count']} sessions]")
        add("proxy_rss_mb", proxied["rss_mb_max"], "largest of the relays after one call")
        add("total_rss_mb", proxied["total_with_daemon_rss_mb"], "RSS counts shared libraries once per process")
    return rows


def markdown(report: dict) -> str:
    lines = ["# Resident daemon benchmark", "", f"Machine: {report['machine'].get('cpu_model', '?')}, "
             f"{report['machine'].get('cpu_count')} CPUs, Python {report['machine'].get('python')}, "
             f"load average at start {report['machine'].get('loadavg_start')}.", "",
             "## Cold hook processes (wall time, ms)", "",
             "| host | event | standalone p50 | standalone p95 | daemon p50 | daemon p95 | daemon-side p50 | daemon-side p95 | client peak RSS (MB) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for host, events in report.get("hooks", {}).items():
        for event, cell in events.items():
            s, d = cell["standalone"], cell["daemon"]
            lines.append(f"| {host} | {event} | {s['wall_ms']['p50']} | {s['wall_ms']['p95']} | {d['wall_ms']['p50']} | "
                         f"{d['wall_ms']['p95']} | {d['daemon_side_ms']['p50']} | {d['daemon_side_ms']['p95']} | "
                         f"{d['peak_rss_mb']['max']} |")
    lines += ["", "Daemon-side time is the whole request in the daemon (queueing, settings, judgement); the judgement "
              "(`run_ms`, the engine's own work: ingest of appended bytes, audit) dominates it:", "",
              "| host | event | judgement (run) p50 | run p95 | rest of the daemon's request p50 | p95 |",
              "|---|---|---|---|---|---|"]
    for host, events in report.get("hooks", {}).items():
        for event, cell in events.items():
            d = cell["daemon"]
            lines.append(f"| {host} | {event} | {d['daemon_run_ms']['p50']} | {d['daemon_run_ms']['p95']} | "
                         f"{d['daemon_overhead_ms']['p50']} | {d['daemon_overhead_ms']['p95']} |")
    servers = report.get("servers")
    if servers:
        lines += ["", f"## {servers['count']} idle MCP sessions", "", "| mode | total RSS (MB) | total PSS (MB) | largest process RSS (MB) |",
                  "|---|---|---|---|"]
        for label in ("standalone", "proxied"):
            block = servers.get(label)
            if block:
                total = block.get("total_with_daemon_rss_mb", block["rss_mb_total"])
                pss = block.get("total_with_daemon_pss_mb", block["pss_mb_total"])
                lines.append(f"| {label}{' (relays + daemon)' if label == 'proxied' else ''} | {total} | {pss} | "
                             f"{max(block['rss_mb_max'], (block.get('daemon') or {}).get('rss', 0))} |")
    lines += ["", "## Against the roundtable's provisional targets", "", "| metric | target | measured | status | note |",
              "|---|---|---|---|---|"]
    for row in report["evaluation"]:
        lines.append(f"| {row['metric']} | {row['target']} | {row['measured']} | {row['status']} | {row['note']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", type=Path, default=HERE.parent)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--servers", type=int, default=12)
    parser.add_argument("--session-mb", type=float, default=1.0)
    parser.add_argument("--contracts", type=int, default=10)
    parser.add_argument("--append-bytes", default="0,4096",
                        help="bytes appended to the session before each run, comma list (0 = nothing new to ingest)")
    parser.add_argument("--events", default="PreToolUse,UserPromptSubmit,Stop")
    parser.add_argument("--only", default="hooks,servers")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        print("the daemon is Linux only in this release", file=sys.stderr)
        return 2
    args.append_list = [int(v) for v in str(args.append_bytes).split(",")]
    only = set(args.only.split(","))
    root = Path(tempfile.mkdtemp(prefix="bench-daemon-"))
    lab = Lab(root, args.repo.resolve(), args.python, args.session_mb, args.contracts)
    report: dict = {"machine": bench.machine_info(args.python), "runs": args.runs}
    try:
        if "hooks" in only:
            print("cold hooks, standalone vs daemon (interleaved)")
            report["hooks"] = measure_hooks(lab, args)
            if lab.daemon_proc is not None:
                pid = int((lab.runtime_dir / "daemon.pid").read_text())
                report["daemon_after_hooks"] = memory_row(pid)
                print(f"  daemon after hooks: {report['daemon_after_hooks']}")
        if "servers" in only:
            print(f"{args.servers} idle MCP sessions")
            report["servers"] = measure_servers(lab, args)
        report["evaluation"] = evaluate(report)
    finally:
        lab.stop_daemon()
        shutil.rmtree(root, ignore_errors=True)
    text = markdown(report)
    print(text)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
