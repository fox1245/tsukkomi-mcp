#!/usr/bin/env python3
"""Long-running memory soak through the MCP tool entry points. Offline: FakeEmbedder and synthetic sessions.

One process and one event loop, like a long-lived server. Three phases:

* ``steady``: unchanged sessions and every tool, repeatedly. Retained Python heap, file descriptors and threads
  must not grow.
* ``growth``: the same sessions keep growing and are re-embedded. Retained heap must not scale with history size.
* ``churn``: a new session every cycle. Per-session in-memory state (hook states, single-flight entries) must
  stay bounded.

Each phase warms up first and is then measured with ``tracemalloc`` (net retained allocations since the end of the
warm-up) over two consecutive windows of equal length. A leak grows in both windows; a cache that fills once, or a
dict that resizes, shows up as a single step in one of them. So a phase fails only when *both* windows grow beyond
the limit. The churn warm-up exceeds the hook-state LRU (256 sessions) so bounded per-session caches are full before
measuring. RSS is reported but only loosely checked, because allocator fragmentation makes it noisy on shared
runners.

Known limit: CPython 3.12 builds where interned strings are immortal (for example Ubuntu 24.04's 3.12.3)
keep every unique path component that ``pathlib`` interns, about 150 bytes per new session path. That is bounded by
the number of sessions, not by requests; newer 3.12 releases and 3.13 release them.

    python scripts/memory_soak.py --check --report soak.json
    python scripts/memory_soak.py --dense-backend sqlite-vector --vector-path /path/to/vector.so --check
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import tracemalloc
import uuid

KIB = 1024
MIB = 1024 * KIB
# Per-window growth limits. A phase fails when both of its windows exceed the limit. Calibrated on Linux with
# CPython 3.12 and the official sqlite-vector build at several times the observed noise; a leak of one event's text
# (2 KiB) per cycle over a window exceeds each of them at the default cycle count.
LIMITS = {"steady": 96 * KIB, "growth": 128 * KIB, "churn": 128 * KIB}
CHURN_WARMUP = 300
RSS_LIMIT = 192 * MIB               # whole run, after warm-up; loose by design
# asyncio's default executor starts worker threads on demand up to this cap and keeps them: not a leak.
EXECUTOR_CAP = min(32, (os.cpu_count() or 1) + 4)
BULK = "lorem ipsum dolor sit amet " * 80   # about 2 KiB per synthetic event


def _status(field: str) -> int | None:
    try:
        with open("/proc/self/status", encoding="ascii") as stream:
            for line in stream:
                if line.startswith(field + ":"):
                    value = line.split()[1]
                    return int(value) * KIB if field.startswith("Vm") else int(value)
    except OSError:
        return None
    return None


def _fds() -> int | None:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return None


def sample() -> dict:
    gc.collect()
    return {"traced": tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else None,
            "rss": _status("VmRSS"), "fds": _fds(), "native_threads": _status("Threads"),
            "python_threads": sum(not t.name.startswith("asyncio_") for t in threading.enumerate()),
            "executor_threads": sum(t.name.startswith("asyncio_") for t in threading.enumerate())}


# --- synthetic sessions -----------------------------------------------------------------------------------

def codex_events(n: int, start: int = 0) -> list[dict]:
    events = []
    for i in range(start, start + n):
        events.append({"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                                              "content": f"step {i} {BULK}"}})
        events.append({"type": "response_item", "payload": {"type": "function_call", "name": "shell",
                                                              "call_id": f"c{i}",
                                                              "arguments": {"command": f"echo {i}"}}})
        events.append({"type": "response_item", "payload": {"type": "function_call_output", "call_id": f"c{i}",
                                                              "output": json.dumps({"exit_code": 0, "output": "ok"})}})
    return events


def claude_events(sid: str, n: int, start: int = 0) -> list[dict]:
    events = []
    for i in range(start, start + n):
        common = {"sessionId": sid, "parentUuid": None, "cwd": "/synthetic/project", "isSidechain": False,
                  "timestamp": "2026-10-01T00:00:00.000Z"}
        events.append({**common, "type": "user", "uuid": str(uuid.uuid4()), "origin": {"kind": "human"},
                       "message": {"role": "user", "content": f"request {i}"}})
        events.append({**common, "type": "assistant", "uuid": str(uuid.uuid4()),
                       "message": {"id": f"msg_{i}", "role": "assistant", "model": "claude-synthetic",
                                   "stop_reason": "end_turn", "content": [{"type": "text", "text": f"{i} {BULK}"}]}})
    return events


def append(path: Path, events: list[dict]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for event in events:
            stream.write(json.dumps(event) + "\n")


class Soak:
    def __init__(self, root: Path, dense_backend: str, vector_path: str | None):
        self.root = root
        # The numpy backend is the offline test index and keeps every vector in RAM by design, which would hide
        # a leak behind expected growth. Without the native library the soak indexes locally (embed=False).
        self.embed = dense_backend == "sqlite-vector"
        self.codex_root = root / "codex"
        self.claude_root = root / "claude"
        self.codex_root.mkdir()
        (root / "empty.toml").write_text("", encoding="utf-8")
        os.environ.update({
            "SELF_DIRECT_INDEX_DIR": str(root / "index"), "SELF_DIRECT_USE_FAKE_EMBEDDER": "true",
            "SELF_DIRECT_DENSE_BACKEND": dense_backend, "SELF_DIRECT_CODEX_SESSIONS_DIR": str(self.codex_root),
            "SELF_DIRECT_CLAUDE_PROJECTS_DIR": str(self.claude_root),
            "SELF_DIRECT_CONFIG_FILE": str(root / "empty.toml"), "SELF_DIRECT_SEED_EXAMPLE_CONTRACTS": "false",
        })
        for name in ("SELF_DIRECT_OPENROUTER_API_KEY_FILE", "SELF_DIRECT_LOCAL_ONLY", "SELF_DIRECT_SESSION_PROVIDER"):
            os.environ.pop(name, None)
        if vector_path:
            os.environ["SELF_DIRECT_SQLITE_VECTOR_PATH"] = vector_path
        os.chdir(root)                       # no stray .env is ever read
        from self_directing_mcp import claude_hooks, server
        self.server, self.claude_hooks = server, claude_hooks
        self.loop = asyncio.new_event_loop()
        self.codex_sid = str(uuid.uuid4())
        self.codex_path = self.new_codex(self.codex_sid, 30)
        self.claude_sid = str(uuid.uuid4())
        self.claude_path = self.claude_root / "-synthetic-project" / f"{self.claude_sid}.jsonl"
        self.claude_path.parent.mkdir(parents=True)
        append(self.claude_path, claude_events(self.claude_sid, 30))
        self.grown = 30
        self.tool("upsert_contracts", contracts=[
            {"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"},
            {"id": "tests-first", "type": "must", "scope": "tool_call", "regex": "npm test",
             "before_regex": "deploy", "requires_success": True},
            {"id": "report", "type": "must", "scope": "message", "regex": "VERIFIED", "provider": "claude"},
        ])
        for sid, provider in ((self.codex_sid, "codex"), (self.claude_sid, "claude")):
            self.tool("update_checklist", session_id=sid, provider=provider, add=[{"text": "finish the task"}])

    def new_codex(self, sid: str, events: int) -> Path:
        path = self.codex_root / f"rollout-2026-10-01T00-00-00-{sid}.jsonl"
        append(path, [{"type": "session_meta", "payload": {"id": sid}}] + codex_events(events))
        return path

    def tool(self, name: str, **arguments):
        result = self.loop.run_until_complete(self.server.mcp.call_tool(name, arguments))
        content, structured = result if isinstance(result, tuple) else (result, None)
        value = structured if structured is not None else json.loads(content[0].text)
        if isinstance(value, dict) and value.get("ok") is False and value.get("error") not in (None, "not_found"):
            raise RuntimeError(f"{name} failed: {value.get('error')}: {value.get('message')}")
        return value

    def claude_hook(self, event: str, **extra):
        payload = {"session_id": self.claude_sid, "transcript_path": str(self.claude_path), "cwd": "/synthetic/project",
                   "hook_event_name": event, **extra}
        return self.claude_hooks.run_hook(event, payload, engine=self.server._engine)

    def exercise(self, codex_sid: str, codex_path: Path) -> None:
        action = {"tool_name": "shell", "arguments": {"command": "ls -la"}}
        self.tool("codex_session_hook", event_name="UserPromptSubmit", session_id=codex_sid,
                  transcript_path=str(codex_path), prompt="continue")
        self.tool("codex_session_hook", event_name="PreToolUse", session_id=codex_sid,
                  transcript_path=str(codex_path), tool_name="shell", tool_input={"command": "ls"})
        self.tool("check_action", session_id=codex_sid, action=action, provider="codex")
        self.tool("audit_session", session_id=codex_sid, provider="codex")
        self.tool("codex_session_hook", event_name="Stop", session_id=codex_sid, transcript_path=str(codex_path))

    def steady(self) -> None:
        self.exercise(self.codex_sid, self.codex_path)
        self.claude_hook("PreToolUse", tool_name="Bash", tool_input={"command": "ls"})
        self.claude_hook("Stop", stop_hook_active=False)
        self.tool("check_action", session_id=self.claude_sid, provider="claude",
                  action={"tool_name": "Bash", "arguments": {"command": "rm -rf build"}})
        self.tool("search_history", query="step lorem", mode="sparse", top_k=5)
        if self.embed:
            self.tool("search_history", query="step lorem", mode="hybrid", top_k=5)
        self.tool("get_checklist", session_id=self.codex_sid, provider="codex")
        self.tool("audit_status")
        self.tool("list_contracts")

    def growth(self) -> None:
        append(self.codex_path, codex_events(2, start=self.grown))
        append(self.claude_path, claude_events(self.claude_sid, 2, start=self.grown))
        self.grown += 2
        self.tool("sync_session", session_id=self.codex_sid, provider="codex", embed=self.embed)
        self.tool("sync_session", session_id=self.claude_sid, provider="claude", embed=self.embed)
        self.steady()

    def churn(self) -> None:
        sid = str(uuid.uuid4())
        path = self.new_codex(sid, 10)
        self.tool("sync_session", session_id=sid, provider="codex", embed=self.embed)
        self.exercise(sid, path)


def run_phase(soak: Soak, name: str, cycles: int, warmup: int) -> dict:
    step = getattr(soak, name)
    if name == "churn":
        warmup = max(warmup, CHURN_WARMUP)
    for _ in range(warmup):
        step()
    tracemalloc.start()
    started, first = time.monotonic(), sample()
    samples, middle = [first], None
    for cycle in range(1, cycles + 1):
        step()
        if cycle == cycles // 2:
            middle = sample()
            samples.append(middle)
        elif cycle % max(1, cycles // 10) == 0:
            samples.append(sample())
    last = sample()
    tracemalloc.stop()
    windows = [middle["traced"] - first["traced"], last["traced"] - middle["traced"]]
    report = {"phase": name, "cycles": cycles, "warmup": warmup, "seconds": round(time.monotonic() - started, 1),
              "retained": last["traced"], "window_growth": windows, "limit": LIMITS[name],
              "fds": [first["fds"], last["fds"]], "native_threads": [first["native_threads"], last["native_threads"]],
              "python_threads": [first["python_threads"], last["python_threads"]],
              "executor_threads": [first["executor_threads"], last["executor_threads"]],
              "rss": [first["rss"], last["rss"]], "samples": samples}
    problems = []
    if min(windows) > LIMITS[name]:
        problems.append(f"retained heap grew in both windows: {windows[0] / KIB:.0f} and {windows[1] / KIB:.0f} KiB "
                        f"(limit {LIMITS[name] / KIB:.0f} each)")
    if first["fds"] is not None and last["fds"] > first["fds"]:
        problems.append(f"open file descriptors grew {first['fds']} -> {last['fds']}")
    if last["python_threads"] > first["python_threads"]:
        problems.append(f"Python threads grew {first['python_threads']} -> {last['python_threads']}")
    if last["executor_threads"] > EXECUTOR_CAP:
        problems.append(f"{last['executor_threads']} executor threads exceed the default cap {EXECUTOR_CAP}")
    report["problems"] = problems
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cycles", type=int, default=160, help="measured cycles per phase")
    parser.add_argument("--warmup", type=int, default=30, help="unmeasured cycles before each phase")
    parser.add_argument("--phases", default="steady,growth,churn")
    parser.add_argument("--dense-backend", default="numpy", choices=("numpy", "sqlite-vector"))
    parser.add_argument("--vector-path", help="sqlite-vector library (required with --dense-backend sqlite-vector)")
    parser.add_argument("--report", type=Path, help="write the JSON report here")
    parser.add_argument("--check", action="store_true", help="exit 1 when a limit is exceeded")
    args = parser.parse_args(argv)
    if args.dense_backend == "sqlite-vector" and not args.vector_path:
        parser.error("--vector-path is required with --dense-backend sqlite-vector")
    # The soak changes into its temporary directory, so caller-relative paths are resolved first.
    report_path = args.report.resolve() if args.report else None
    vector_path = str(Path(args.vector_path).resolve()) if args.vector_path else None
    origin = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="tsukkomi-soak-") as directory:
        soak = Soak(Path(directory), args.dense_backend, vector_path)
        baseline = sample()
        phases = [run_phase(soak, name, args.cycles, args.warmup) for name in args.phases.split(",")]
        final = sample()
        soak.loop.close()
        soak.server._engine.close()
        os.chdir(origin)
    problems = [f"{p['phase']}: {problem}" for p in phases for problem in p["problems"]]
    if baseline["rss"] is not None and final["rss"] - baseline["rss"] > RSS_LIMIT:
        problems.append(f"RSS grew {(final['rss'] - baseline['rss']) / MIB:.0f} MiB over the run "
                        f"(limit {RSS_LIMIT / MIB:.0f})")
    report = {"python": sys.version.split()[0], "dense_backend": args.dense_backend, "baseline": baseline,
              "final": final, "phases": phases, "problems": problems}
    if report_path:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for phase in phases:
        first_window, second_window = phase["window_growth"]
        print(f"{phase['phase']:>6}: {phase['cycles']} cycles in {phase['seconds']}s, retained "
              f"{phase['retained'] / KIB:.0f} KiB, windows {first_window / KIB:+.0f}/{second_window / KIB:+.0f} KiB "
              f"(limit {phase['limit'] / KIB:.0f}), fds {phase['fds']}, threads {phase['python_threads']} (+executor {phase['executor_threads'][1]})")
    if baseline["rss"] is not None:
        print(f"   rss: {baseline['rss'] / MIB:.0f} -> {final['rss'] / MIB:.0f} MiB")
    for problem in problems:
        print("FAIL", problem)
    return 1 if args.check and problems else 0


if __name__ == "__main__":
    sys.exit(main())
