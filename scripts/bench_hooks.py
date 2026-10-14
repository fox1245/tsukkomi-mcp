#!/usr/bin/env python3
"""Reproducible hook-cost benchmark: cold command hooks, warm resident-server calls, idle-server memory.

Everything is synthetic. The benchmark runs under an isolated HOME, index and config file; it never reads a user's
sessions, indexes or credentials. See README "Testing and verification" for usage.

What is measured, separately (the numbers are never mixed):

* ``cold``: one fresh Python process per hook event, through the real ``scripts/claude_hook.py`` and
  ``scripts/agy_hook.py`` entry points with synthetic payloads (Claude Code: PreToolUse, PostToolUse,
  UserPromptSubmit, SessionStart, Stop; AGY: PreToolUse, PostToolUse, PreInvocation, Stop). The page cache is warm;
  "cold" means a new interpreter that imports the package and judges once. Wall time, user+sys CPU and peak RSS come
  from the operating system (``os.wait4`` on POSIX; ``GetProcessTimes`` / ``GetProcessMemoryInfo`` on Windows).
  The checked-in installers register ``python scripts/*_hook.py``. A ``.venv`` re-exec (one extra interpreter start)
  is not included: the benchmark runs the interpreter it is given directly.
* ``warm``: calls through the FastMCP dispatcher of a long-lived server process (``codex_session_hook``,
  ``check_action``, ``audit_session``), one asyncio loop, one engine. ``first_call`` is the index build; steady
  state is reported separately for "nothing appended" and "after appending N bytes".
* ``idle_servers``: N real ``python -m self_directing_mcp`` stdio servers started with an isolated index, after the
  MCP handshake (``after_init``) and after one tool call that initializes the engine (``after_use``). Linux reads
  /proc (VmRSS, VmHWM, Pss); Windows reads the working set through psapi.

Fixed conditions (flags, defaults in parentheses): contract counts (10,100), each a rotating mix of must_not,
must, must+before_regex and must+before_regex+requires_success; small session size (1 MB); large session
(``--large``, 300 MB, opt-in, never in CI); appended bytes per step (4096); concurrent sessions in the burst (4);
idle servers (12); repetitions (20 after 1 discarded warm-up).

Targets and CI thresholds live in ``TARGETS`` below (one place). A CI threshold is 1.5x the target. Rows marked
``gated: False`` are reported with their target but never fail ``--check``; they depend on other in-flight work
(fast integrity path, incremental audit).

Usage::

    python scripts/bench_hooks.py --out bench.json --markdown bench.md [--check]
    python scripts/bench_hooks.py --large --large-mb 300 --out bench-large.json
    python scripts/bench_hooks.py --compare before.json after.json
    python scripts/bench_hooks.py --repo /path/to/other/checkout ...      # benchmark another checkout
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import datetime
import json
import math
import os
from pathlib import Path
import platform
import queue
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

IS_WINDOWS = os.name == "nt"
HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parent
CI_FACTOR = 1.5          # default CI threshold = target x 1.5 (``--ci-factor`` widens it on a slower machine)

# --- targets (one place) ---------------------------------------------------------------------------------------
# Source: the 3-vendor roundtable minutes, item Q3. ``stat`` says how the measured value is derived from the
# report. ``gated`` is the decision for THIS branch: True = ``--check`` fails above target * CI factor (1.5).
#
# Gating decision for the cold hooks. This branch removes the import cost of the hook path (the light settings
# reader, lazy NumPy, no HTTP stack, no parser for bookkeeping events). The events below spend their time in that
# floor, so their numbers are what the branch can honestly be held to. Every other event also runs a judgement
# whose cost grows with the session (PreToolUse: current-action capture; Stop: sync + audit over the whole history,
# evaluating every contract; AGY: the transcript is re-parsed on every call). Those wait for the fast integrity
# path and the incremental audit, so they are reported against the same target and not gated yet.
GATED_COLD_EVENTS = {"claude": ("PostToolUse", "UserPromptSubmit", "SessionStart")}
NOT_GATED_COLD = ("reported, not gated yet: PreToolUse/Stop evaluate every contract over the whole session (needs the "
                  "fast integrity path and the incremental audit) and AGY re-parses its transcript on every call")
TARGETS = [
    {"id": "cold_hook_wall_p95_ms", "label": "Cold hook wall, p95 (gated events)", "unit": "ms", "target": 400,
     "gated": True, "stat": "max over the gated events of p95 wall (small session, every contract count)",
     "note": "phase 1 gate; gated events: Claude Code PostToolUse, UserPromptSubmit, SessionStart"},
    {"id": "cold_hook_wall_p95_all_ms", "label": "Cold hook wall, p95 (worst of all events)", "unit": "ms", "target": 400,
     "gated": False, "stat": "max over every cold event of p95 wall (small session)", "note": NOT_GATED_COLD},
    {"id": "cold_hook_wall_p50_ms", "label": "Cold hook wall, median (worst of all events)", "unit": "ms", "target": 150,
     "gated": False, "stat": "max over every cold event of p50 wall (small session)",
     "note": "phase 1 goal, not a gate: " + NOT_GATED_COLD},
    {"id": "cold_hook_cpu_s", "label": "Cold hook CPU (user+sys), p95 (gated events)", "unit": "s", "target": 0.3,
     "gated": True, "stat": "max over the gated events of p95 CPU (small session)", "note": "phase 1"},
    {"id": "cold_hook_cpu_all_s", "label": "Cold hook CPU (user+sys), p95 (worst of all events)", "unit": "s",
     "target": 0.3, "gated": False, "stat": "max over every cold event of p95 CPU (small session)",
     "note": NOT_GATED_COLD},
    {"id": "warm_check_action_unchanged_cpu_ms", "label": "check_action, large session, nothing appended, CPU",
     "unit": "ms", "target": 100, "gated": False, "stat": "p50 CPU, large session (--large only)",
     "note": "reported, not gated yet: needs the fast integrity path"},
    {"id": "warm_incremental_audit_cpu_ms", "label": "audit_session after <=100 appended events, CPU",
     "unit": "ms", "target": 20, "gated": False, "stat": "p50 CPU after append (small session)",
     "note": "reported, not gated yet: needs incremental audit"},
    {"id": "idle_server_rss_mb", "label": "Idle server RSS (after one tool call)", "unit": "MB", "target": 90,
     "gated": True, "stat": "max over servers of RSS after_use", "note": "target 70 MB desired"},
]
# Informational only (phase 2 entry conditions, Q3): never gated here.
PHASE2_TRIGGERS = {"total_rss_mb": 500, "turn_hook_latency_s": 2.0}

# --- identifiers and synthetic content --------------------------------------------------------------------------
CLAUDE_SID = "aaaaaaaa-1111-4111-8111-aaaaaaaa{:04d}"
AGY_SID = "11111111-1111-4111-8111-1111111111{:02d}"
CODEX_SID = "33333333-2222-3333-4444-5555555555{:02d}"
PROJECT = "-synthetic-bench"
FILLER = ("synthetic bench output: nothing here is real, all lines are generated filler about build steps, test "
          "collection, linting and packaging so the transcript has realistic bulk ")
ACTION_COMMAND = "deploy --prod"          # matches the before_regex contracts: a real prerequisite judgement


def _tmpl_contracts(count: int) -> list[dict]:
    """A rotating mix of the four judgement shapes; ids are stable so runs are comparable."""
    must_not = [r"rm\s+-rf\s+/(?:\s|$)", r"git\s+push\s+--force", r"curl\s+[^|]*\|\s*(?:sudo\s+)?sh",
                r"DROP\s+TABLE", r"chmod\s+777", r"mkfs\.", r"--no-verify", r"npm\s+publish"]
    contracts = []
    for index in range(count):
        kind = index % 4
        n = index // 4
        base = {"id": f"bench-{index:03d}", "scope": "tool_call", "description": f"synthetic bench contract {index}"}
        if kind == 0:      # prohibition: judged on every proposed action and over history
            contracts.append({**base, "type": "must_not", "regex": must_not[n % len(must_not)] + f"(?# {n})"})
        elif kind == 1:    # completion obligation: judged at Stop
            contracts.append({**base, "type": "must", "regex": r"pytest|unit tests ran" + f"(?# {n})"})
        elif kind == 2:    # precondition that never applies to the benchmark action (applicability check only)
            contracts.append({**base, "type": "must", "regex": r"pytest", "before_regex": f"release-stage-{n}"})
        else:              # precondition that applies to the benchmark action and needs a successful prior call
            contracts.append({**base, "type": "must", "regex": r"pytest", "before_regex": r"deploy",
                              "requires_success": True})
    return contracts


def write_contracts(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"contracts": _tmpl_contracts(count)}), encoding="utf-8")


def _command(i: int) -> str:
    if i % 41 == 0:
        return "pytest -q tests"
    return ("echo step-%d" % i, "git status --short", "ls -la src", "python -m compileall -q src")[i % 4]


def claude_pair(sid: str, n: int, i: int) -> str:
    """Two JSONL lines: an assistant tool_use and its successful tool_result (one 'step')."""
    def env(kind, k, **extra):
        record = {"type": kind, "uuid": f"00000000-0000-4000-8000-{k:012d}", "parentUuid": None, "sessionId": sid,
                  "timestamp": "2026-10-01T00:00:00.000Z", "cwd": "/synthetic/project", "isSidechain": False,
                  "userType": "external", "version": "9.9.9"}
        record.update(extra)
        return json.dumps(record)
    cid = f"toolu_{sid[-4:]}_{i}"
    use = env("assistant", 2 * n, requestId="req_synthetic", message={
        "id": f"msg_{i}", "role": "assistant", "model": "claude-synthetic", "stop_reason": "tool_use",
        "stop_sequence": None, "content": [
            {"type": "text", "text": f"Step {i}. " + FILLER * 2},
            {"type": "tool_use", "id": cid, "name": "Bash", "input": {"command": _command(i)}}]})
    result = env("user", 2 * n + 1, sourceToolAssistantUUID="00000000-0000-4000-8000-0000000000aa", message={
        "role": "user", "content": [{"type": "tool_result", "tool_use_id": cid, "is_error": False,
                                     "content": f"ok {i}. " + FILLER * 3}]})
    return use + "\n" + result + "\n"


def claude_head(sid: str) -> str:
    return json.dumps({"type": "user", "uuid": "00000000-0000-4000-8000-000000000001", "parentUuid": None,
                       "sessionId": sid, "timestamp": "2026-10-01T00:00:00.000Z", "cwd": "/synthetic/project",
                       "isSidechain": False, "userType": "external", "version": "9.9.9",
                       "message": {"role": "user", "content": "run the unit tests, then deploy"},
                       "origin": {"kind": "human"}, "promptId": "prompt-1"}) + "\n"


def agy_pair(step: int) -> str:
    """AGY display transcript: a planner step with one proposed call and its RUN_COMMAND result."""
    call = {"step_index": step, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE",
            "created_at": "2026-10-01T00:00:01Z", "content": f"Step {step}. " + FILLER,
            "tool_calls": [{"name": "run_command", "args": {"CommandLine": json.dumps(_command(step)),
                                                            "Cwd": json.dumps("/synthetic/project")}}]}
    result = {"step_index": step + 1, "source": "MODEL", "type": "RUN_COMMAND", "status": "DONE",
              "created_at": "2026-10-01T00:00:02Z", "exit_code": 0, "content": f"ok {step}. " + FILLER * 2}
    return json.dumps(call) + "\n" + json.dumps(result) + "\n"


def agy_head() -> str:
    return json.dumps({"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE",
                       "created_at": "2026-10-01T00:00:00Z",
                       "content": "<USER_REQUEST>run the unit tests, then deploy</USER_REQUEST>"}) + "\n"


def codex_pair(i: int) -> str:
    cid = f"call-{i}"
    call = {"type": "response_item", "payload": {"type": "function_call", "name": "shell", "call_id": cid,
                                                 "arguments": {"command": _command(i)}}}
    out = {"type": "response_item", "payload": {"type": "function_call_output", "call_id": cid,
                                                "output": json.dumps({"exit_code": 0, "output": "ok " + FILLER * 3})}}
    msg = {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                                "content": f"Step {i}. " + FILLER * 2}}
    return json.dumps(msg) + "\n" + json.dumps(call) + "\n" + json.dumps(out) + "\n"


class Transcript:
    """One synthetic session file that can grow by whole records."""

    def __init__(self, kind: str, path: Path, sid: str) -> None:
        self.kind, self.path, self.sid, self.i = kind, path, sid, 0

    def _record(self) -> str:
        self.i += 1
        if self.kind == "claude":
            return claude_pair(self.sid, self.i, self.i)
        if self.kind == "agy":
            return agy_pair(2 * self.i - 1)
        return codex_pair(self.i)

    def create(self, target_bytes: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        head = {"claude": lambda: claude_head(self.sid), "agy": agy_head,
                "codex": lambda: json.dumps({"type": "session_meta", "payload": {"id": self.sid}}) + "\n"}[self.kind]()
        written = 0
        with self.path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(head)
            written += len(head)
            while written < target_bytes:
                chunk = "".join(self._record() for _ in range(64))
                stream.write(chunk)
                written += len(chunk)

    def append(self, nbytes: int) -> int:
        if nbytes <= 0:
            return 0
        written, parts = 0, []
        while written < nbytes:
            part = self._record()
            parts.append(part)
            written += len(part)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write("".join(parts))
        return written

    @property
    def last_step(self) -> int:
        return 2 * self.i + 1 if self.kind == "agy" else self.i


def agy_path(app: Path, sid: str) -> Path:
    return app / "brain" / sid / ".system_generated" / "logs" / "transcript.jsonl"


def claude_path(projects: Path, sid: str) -> Path:
    return projects / PROJECT / f"{sid}.jsonl"


def codex_path(sessions: Path, sid: str) -> Path:
    return sessions / f"rollout-2026-09-09T01-00-00-{sid}.jsonl"


# --- machine and process measurement ----------------------------------------------------------------------------
if IS_WINDOWS:
    from ctypes import wintypes

    class _Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _k32.GetCurrentProcess.restype = wintypes.HANDLE          # the pseudo-handle is pointer sized: truncating it
    _k32.GetCurrentProcess.argtypes = []                      # makes psapi return zeros
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    _psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
    PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_VM_READ = 0x1000, 0x10

    def _filetime(value) -> float:
        return ((value.dwHighDateTime << 32) | value.dwLowDateTime) / 1e7

    def win_memory(handle) -> dict:
        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        if not _psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return {}
        return {"rss": counters.WorkingSetSize, "peak_rss": counters.PeakWorkingSetSize,
                "private": counters.PrivateUsage}

    def win_cpu(handle) -> float:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not _k32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
            return float("nan")
        return _filetime(times[2]) + _filetime(times[3])

    def win_open(pid: int):
        return _k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, pid)


def self_cpu() -> float:
    """CPU seconds (user+sys, all threads) used by this process so far."""
    if IS_WINDOWS:
        return win_cpu(_k32.GetCurrentProcess())
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def self_memory() -> dict:
    if IS_WINDOWS:
        return win_memory(_k32.GetCurrentProcess())
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = peak if sys.platform == "darwin" else peak * 1024
    return {"rss": proc_status(os.getpid()).get("rss", peak), "peak_rss": peak}


def proc_status(pid: int) -> dict:
    """Linux: VmRSS/VmHWM from /proc/<pid>/status, Pss from smaps_rollup, CPU from stat. Empty elsewhere."""
    out: dict = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                out["rss"] = int(line.split()[1]) * 1024
            elif line.startswith("VmHWM:"):
                out["peak_rss"] = int(line.split()[1]) * 1024
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                out["pss"] = int(line.split()[1]) * 1024
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        out["cpu_s"] = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        pass
    return out


def process_memory(proc: subprocess.Popen) -> dict:
    if IS_WINDOWS:
        handle = win_open(proc.pid)
        if not handle:
            return {}
        try:
            return {**win_memory(handle), "cpu_s": win_cpu(handle)}
        finally:
            _k32.CloseHandle(handle)
    return proc_status(proc.pid)


def machine_info(python: str) -> dict:
    info = {"platform": platform.platform(), "system": platform.system(), "machine": platform.machine(),
            "release": platform.release(), "cpu_count": os.cpu_count(), "python": sys.version.split()[0],
            "bench_python": sys.version.split()[0], "interpreter": python}
    if sys.platform.startswith("linux"):
        info["wsl"] = "microsoft" in platform.release().lower()
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    info["ram_mb"] = int(line.split()[1]) // 1024
                    break
        except OSError:
            pass
    elif IS_WINDOWS:
        info["cpu_model"] = os.environ.get("PROCESSOR_IDENTIFIER", "")
    if hasattr(os, "getloadavg"):
        info["loadavg_start"] = [round(v, 2) for v in os.getloadavg()]
    code = ("import sys,json,importlib.metadata as m\n"
            "d={'python':sys.version.split()[0]}\n"
            "for n in ('numpy','pydantic','pydantic-settings','mcp','neograph-engine','httpx'):\n"
            "    try: d[n]=m.version(n)\n"
            "    except Exception: d[n]=None\n"
            "print(json.dumps(d))\n")
    try:
        done = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=60)
        info["child"] = json.loads(done.stdout)
        info["python"] = info["child"]["python"]
    except Exception:
        pass
    return info


# --- statistics ------------------------------------------------------------------------------------------------
def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def summarize(values: list[float], digits: int = 1) -> dict:
    return {"n": len(values), "p50": round(statistics.median(values), digits) if values else None,
            "p95": round(percentile(values, 0.95), digits) if values else None,
            "max": round(max(values), digits) if values else None}


# --- isolated environment ---------------------------------------------------------------------------------------
def toml_string(value) -> str:
    return json.dumps(str(value))


class Sandbox:
    """Directories, config file and environment shared by every child process of one benchmark run."""

    def __init__(self, root: Path, repo: Path, python: str) -> None:
        self.root, self.repo, self.python = root, repo, python
        self.home = root / "home"
        self.projects = root / "claude-projects"
        self.agy_app = root / "agy-app"
        self.sessions = root / "codex-sessions"
        self.home.mkdir(parents=True, exist_ok=True)
        self.config = root / "config.toml"

    def index_dir(self, name: str) -> Path:
        path = self.root / "index" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_config(self, index: Path, contracts: Path | None = None) -> Path:
        lines = ["[paths]", f"index_dir = {toml_string(index)}",
                 f"claude_projects_dir = {toml_string(self.projects)}",
                 f"agy_app_data_dirs = [{toml_string(self.agy_app)}]",
                 f"codex_sessions_dir = {toml_string(self.sessions)}"]
        if contracts is not None:
            lines.append(f"contracts_path = {toml_string(contracts)}")
        path = self.root / f"config-{index.name}.toml"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def env(self, config: Path | None = None, **extra: str) -> dict:
        keep = {k: v for k, v in os.environ.items() if k.upper() in (
            "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL",
            "LD_LIBRARY_PATH", "VIRTUAL_ENV", "PROCESSOR_IDENTIFIER", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE")}
        # Bytecode caches stay enabled: a compiled tree is the normal state of an installed package.
        keep.update(HOME=str(self.home), USERPROFILE=str(self.home), PYTHONPATH=str(self.repo / "src"),
                    SELF_DIRECT_CONFIG_FILE=str(config or self.config))
        keep.update(extra)
        return keep


# --- cold hook processes -----------------------------------------------------------------------------------------
def claude_payload(event: str, sid: str, path: Path, step: int) -> dict:
    base = {"session_id": sid, "transcript_path": str(path), "cwd": "/synthetic/project", "hook_event_name": event}
    if event in ("PreToolUse", "PostToolUse"):
        base.update(tool_name="Bash", tool_input={"command": ACTION_COMMAND if event == "PreToolUse" else _command(step)})
    if event == "PostToolUse":
        base["tool_response"] = {"stdout": "ok", "stderr": "", "interrupted": False}
    if event == "Stop":
        base["stop_hook_active"] = False
    if event == "UserPromptSubmit":
        base["prompt"] = "continue please"
    if event == "SessionStart":
        base["source"] = "startup"
    return base


def agy_payload(event: str, sid: str, path: Path, step: int) -> dict:
    base = {"conversationId": sid, "transcriptPath": str(path), "workspacePaths": ["/synthetic/project"]}
    if event in ("PreToolUse", "PostToolUse"):
        base.update(toolCall={"name": "run_command", "args": {"CommandLine": ACTION_COMMAND if event == "PreToolUse"
                                                              else _command(step)}}, stepIdx=step)
    if event == "Stop":
        base.update(terminationReason="model_stop", fullyIdle=True)
    return base


CLAUDE_EVENTS = ("PreToolUse", "PostToolUse", "UserPromptSubmit", "SessionStart", "Stop")
AGY_EVENTS = ("PreToolUse", "PostToolUse", "PreInvocation", "Stop")


def run_process(argv: list[str], stdin_bytes: bytes, env: dict, cwd: Path, timeout: float) -> dict:
    """Run one process to completion; wall, user+sys CPU and peak RSS from the OS (not from Python)."""
    scratch = Path(tempfile.mkdtemp(prefix="bench-proc-", dir=cwd))
    try:
        (scratch / "in").write_bytes(stdin_bytes)
        with (scratch / "in").open("rb") as fin, (scratch / "out").open("wb") as fout, \
                (scratch / "err").open("wb") as ferr:
            start = time.perf_counter()
            proc = subprocess.Popen(argv, stdin=fin, stdout=fout, stderr=ferr, env=env, cwd=str(cwd))
            peak, cpu = None, None
            try:
                if IS_WINDOWS:
                    proc.wait(timeout=timeout)
                    wall = time.perf_counter() - start
                    handle = int(proc._handle)          # still open: Popen closes it with the object
                    memory = win_memory(wintypes.HANDLE(handle))
                    cpu, peak = win_cpu(wintypes.HANDLE(handle)), memory.get("peak_rss")
                else:
                    deadline = time.monotonic() + timeout
                    while True:
                        pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
                        if pid:
                            break
                        if time.monotonic() > deadline:
                            proc.kill()
                            os.wait4(proc.pid, 0)
                            raise subprocess.TimeoutExpired(argv, timeout)
                        time.sleep(0.0005)
                    wall = time.perf_counter() - start
                    proc.returncode = os.waitstatus_to_exitcode(status)
                    cpu = usage.ru_utime + usage.ru_stime
                    peak = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
            except subprocess.TimeoutExpired:
                proc.kill()
                raise
        return {"wall_s": wall, "cpu_s": cpu, "peak_rss": peak, "returncode": proc.returncode,
                "stdout": (scratch / "out").read_bytes().decode("utf-8", "replace"),
                "stderr": (scratch / "err").read_bytes().decode("utf-8", "replace")}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def hook_argv(sandbox: Sandbox, host: str, event: str, index: Path, mode: str = "local-only") -> list[str]:
    script = sandbox.repo / "scripts" / ("claude_hook.py" if host == "claude" else "agy_hook.py")
    argv = [sandbox.python, str(script), f"--event={event}"]
    if host == "claude":
        argv += ["--local-only"] if mode == "local-only" else []
        argv += [f"--index-dir={index}"]
    return argv


def hook_env(sandbox: Sandbox, config: Path, mode: str) -> dict:
    """``local-only``: the engine never builds an embedder or dense index (the Claude installer's ``--local-only``;
    AGY has no flag, so its hook environment sets the variable). ``default``: the offline stand-in for a normal
    install: fake embedder and the NumPy dense backend, so every hook process also initializes the dense side."""
    if mode == "local-only":
        return sandbox.env(config, SELF_DIRECT_LOCAL_ONLY="true")
    return sandbox.env(config, SELF_DIRECT_USE_FAKE_EMBEDDER="true", SELF_DIRECT_DENSE_BACKEND="numpy")


def judgement(host: str, event: str, out: str, err: str) -> dict:
    """What the hook said, in one word, and whether it failed open. A benchmark of fail-open hooks is invalid."""
    try:
        response = json.loads(out.strip().splitlines()[-1]) if out.strip() else {}
    except (ValueError, IndexError):
        response = {"unparsable": True}
    text = json.dumps(response)
    failed_open = "UNAVAILABLE" in text or "FAILED" in text or "UNAVAILABLE" in err or "check FAILED" in err
    if host == "claude":
        decision = (response.get("hookSpecificOutput", {}).get("permissionDecision")
                    or ("block" if response.get("decision") == "block" else None))
        label = decision or ("context" if "hookSpecificOutput" in response else
                             "message" if "systemMessage" in response else "empty")
    else:
        label = response.get("decision") or ("inject" if "injectSteps" in response else "empty")
    return {"label": label, "failed_open": bool(failed_open)}


class Side:
    """One checkout under test: its own sandbox (repo), index directory, config and environment."""

    def __init__(self, sandbox: Sandbox, label: str, host: str, contracts: int, size_label: str, mode: str) -> None:
        self.sandbox, self.label, self.host, self.mode = sandbox, label, host, mode
        self.index = sandbox.index_dir(f"{host}-{contracts}-{size_label}-{label}")
        contracts_file = self.index / "contracts.json"
        write_contracts(contracts_file, contracts)
        self.env = hook_env(sandbox, sandbox.write_config(self.index, contracts_file), mode)

    def argv(self, event: str) -> list[str]:
        return hook_argv(self.sandbox, self.host, event, self.index, self.mode)


def cold_cell(sandbox: Sandbox, host: str, contracts: int, size_label: str, size_bytes: int, args, *,
              sid_index: int = 0, versus: Sandbox | None = None, log=print) -> dict:
    """All events of one host for one (contracts, size) condition, each event ``--runs`` fresh processes.

    With ``versus`` (a second checkout) the two are measured interleaved on the same transcript and payloads, the
    order alternating each repetition, so that load from other processes on the machine hits both alike.
    """
    sid = CLAUDE_SID.format(sid_index) if host == "claude" else AGY_SID.format(sid_index)
    path = claude_path(sandbox.projects, sid) if host == "claude" else agy_path(sandbox.agy_app, sid)
    transcript = Transcript(host, path, sid)
    started = time.perf_counter()
    transcript.create(size_bytes)
    initial_bytes = path.stat().st_size
    log(f"  {host} {size_label}: generated {initial_bytes / 1e6:.1f} MB in {time.perf_counter() - started:.1f}s")
    sides = [Side(sandbox, "main", host, contracts, size_label, args.hook_mode)]
    if versus is not None:
        sides.append(Side(versus, "versus", host, contracts, size_label, args.hook_mode))
    events = CLAUDE_EVENTS if host == "claude" else AGY_EVENTS
    payload_of = claude_payload if host == "claude" else agy_payload
    runs = args.runs if size_label == "small" else args.large_repeat
    # The first judgement of a session indexes it; it is measured once, apart from the steady state.
    first: dict = {side.label: {} for side in sides}
    for event in events:
        for side in sides:
            body = json.dumps(payload_of(event, sid, path, transcript.last_step + 1)).encode()
            warm_up = run_process(side.argv(event), body, side.env, sandbox.root, timeout=args.hook_timeout)
            first[side.label][event] = {"wall_s": round(warm_up["wall_s"], 3), "cpu_s": round(warm_up["cpu_s"] or 0, 3),
                                        **judgement(host, event, warm_up["stdout"], warm_up["stderr"])}
    cells: dict = {side.label: {} for side in sides}
    for event in events:
        samples = {side.label: {"wall": [], "cpu": [], "rss": [], "labels": {}, "failed": 0} for side in sides}
        for run in range(runs):
            transcript.append(args.append_bytes)
            body = json.dumps(payload_of(event, sid, path, transcript.last_step + 1)).encode()
            for side in (sides if run % 2 == 0 else sides[::-1]):
                result = run_process(side.argv(event), body, side.env, sandbox.root, timeout=args.hook_timeout)
                bag = samples[side.label]
                bag["wall"].append(result["wall_s"] * 1000)
                bag["cpu"].append((result["cpu_s"] or 0) * 1000)
                bag["rss"].append((result["peak_rss"] or 0) / 1048576)
                verdict = judgement(host, event, result["stdout"], result["stderr"])
                bag["labels"][verdict["label"]] = bag["labels"].get(verdict["label"], 0) + 1
                bag["failed"] += verdict["failed_open"] or result["returncode"] != 0
        for side in sides:
            bag = samples[side.label]
            cells[side.label][event] = {"wall_ms": summarize(bag["wall"]), "cpu_ms": summarize(bag["cpu"]),
                                        "peak_rss_mb": summarize(bag["rss"]), "decisions": bag["labels"],
                                        "failed_open": bag["failed"], "first_run": first[side.label][event]}
            row = cells[side.label][event]
            log(f"    {host:6s} {event:17s} {side.label:6s} wall p50/p95 {row['wall_ms']['p50']:7.1f}/"
                f"{row['wall_ms']['p95']:7.1f} ms  cpu p50/p95 {row['cpu_ms']['p50']:7.1f}/{row['cpu_ms']['p95']:7.1f} ms  "
                f"rss {row['peak_rss_mb']['max']:5.1f} MB  {bag['labels']}"
                f"{'  FAILED OPEN x%d' % bag['failed'] if bag['failed'] else ''}")
    result = {"host": host, "contracts": contracts, "session": size_label,
              "session_bytes": initial_bytes, "final_session_bytes": path.stat().st_size,
              "append_bytes_per_step": args.append_bytes, "events": cells["main"]}
    if versus is not None:
        result["versus"] = {"label": args.versus_label, "repo": str(versus.repo), "events": cells["versus"]}
    return result


def burst_cell(sandbox: Sandbox, contracts: int, args, log=print) -> dict:
    """N concurrent sessions: every session fires one hook at the same moment, repeated; shared index."""
    n = args.sessions
    results = {}
    for host in ("claude", "agy"):
        index = sandbox.index_dir(f"burst-{host}-{contracts}")
        contracts_file = index / "contracts.json"
        write_contracts(contracts_file, contracts)
        env = hook_env(sandbox, sandbox.write_config(index, contracts_file), args.hook_mode)
        transcripts = []
        for k in range(n):
            sid = (CLAUDE_SID if host == "claude" else AGY_SID).format(50 + k)
            path = claude_path(sandbox.projects, sid) if host == "claude" else agy_path(sandbox.agy_app, sid)
            transcript = Transcript(host, path, sid)
            transcript.create(args.session_mb * 1_000_000)
            transcripts.append(transcript)
        for event in ("PreToolUse", "Stop"):
            argv = hook_argv(sandbox, host, event, index, args.hook_mode)
            payload_of = claude_payload if host == "claude" else agy_payload
            for t in transcripts:                         # index every session once, outside the measurement
                run_process(argv, json.dumps(payload_of(event, t.sid, t.path, t.last_step + 1)).encode(), env,
                            sandbox.root, args.hook_timeout)
            walls, cpus, spans, failed = [], [], [], 0
            for _ in range(max(3, args.runs // 4)):
                jobs = []
                for t in transcripts:
                    t.append(args.append_bytes)
                    jobs.append(json.dumps(payload_of(event, t.sid, t.path, t.last_step + 1)).encode())
                slots: list = [None] * n
                def go(k, body, t=None):
                    slots[k] = run_process(argv, body, env, sandbox.root, args.hook_timeout)
                threads = [threading.Thread(target=go, args=(k, body)) for k, body in enumerate(jobs)]
                begin = time.perf_counter()
                for th in threads:
                    th.start()
                for th in threads:
                    th.join()
                spans.append((time.perf_counter() - begin) * 1000)
                for slot in slots:
                    walls.append(slot["wall_s"] * 1000)
                    cpus.append((slot["cpu_s"] or 0) * 1000)
                    failed += judgement(host, event, slot["stdout"], slot["stderr"])["failed_open"]
            results[f"{host}.{event}"] = {"wall_ms": summarize(walls), "cpu_ms": summarize(cpus),
                                          "burst_makespan_ms": summarize(spans), "failed_open": failed}
            log(f"    burst x{n} {host:6s} {event:12s} wall p50/p95 {results[f'{host}.{event}']['wall_ms']['p50']:7.1f}/"
                f"{results[f'{host}.{event}']['wall_ms']['p95']:7.1f} ms  makespan p95 "
                f"{results[f'{host}.{event}']['burst_makespan_ms']['p95']:7.1f} ms")
    return {"contracts": contracts, "sessions": n, "session_bytes": args.session_mb * 1_000_000, "results": results}


# --- warm resident-server calls ---------------------------------------------------------------------------------
def _payload(result):
    """The tool's JSON object from whatever FastMCP.call_tool returns in this mcp version."""
    if isinstance(result, tuple):
        result = result[-1] if isinstance(result[-1], dict) else result[0]
    if isinstance(result, dict):
        return result.get("result", result) if set(result) == {"result"} else result
    for block in result or ():
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except ValueError:
                return {"raw": text[:80]}
    return {}


async def _timed(coro_factory, repeat: int, before=None) -> dict:
    walls, cpus, verdicts = [], [], {}
    for _ in range(repeat):
        if before is not None:
            before()
        cpu0, t0 = self_cpu(), time.perf_counter()
        result = _payload(await coro_factory())
        walls.append((time.perf_counter() - t0) * 1000)
        cpus.append((self_cpu() - cpu0) * 1000)
        label = result.get("verdict") or ("incomplete" if result.get("ok") is False else
                                          ("empty" if not result else "context"))
        if isinstance(result, dict) and result.get("hookSpecificOutput"):
            label = "context"
        if isinstance(result, dict) and result.get("systemMessage"):
            label = "context"
        verdicts[label] = verdicts.get(label, 0) + 1
    return {"wall_ms": summarize(walls), "cpu_ms": summarize(cpus), "verdicts": verdicts}


def warm_worker(spec_path: str) -> None:
    """Child process: import the server, drive it through FastMCP's dispatcher, print one JSON document."""
    spec = json.loads(Path(spec_path).read_text())
    t_cpu, t_wall = self_cpu(), time.perf_counter()
    from self_directing_mcp import server          # noqa: E402  (after the environment is set)
    imported = {"wall_ms": round((time.perf_counter() - t_wall) * 1000, 1),
                "cpu_ms": round((self_cpu() - t_cpu) * 1000, 1)}
    transcript = Transcript("codex", Path(spec["path"]), spec["sid"])
    transcript.i = spec["records"]
    session, path = spec["sid"], spec["path"]
    action = {"tool_name": "shell", "arguments": {"command": ACTION_COMMAND}}
    out: dict = {"server_import": imported, "scenarios": {}}

    async def call(name, arguments):
        return await server.mcp.call_tool(name, arguments)

    def check():
        return call("check_action", {"session_id": session, "action": action, "provider": "codex", "path": path})

    def audit():
        return call("audit_session", {"session_id": session, "provider": "codex", "path": path})

    def hook(event):
        args = {"event_name": event, "session_id": session, "transcript_path": path}
        if event == "PreToolUse":
            args.update(tool_name="shell", tool_input={"command": ACTION_COMMAND})
        if event == "UserPromptSubmit":
            args["prompt"] = "continue please"
        return call("codex_session_hook", args)

    def grow():
        transcript.append(spec["append_bytes"])

    async def run() -> None:
        scenarios = out["scenarios"]
        n = spec["repeat"]
        cpu0, t0 = self_cpu(), time.perf_counter()
        first = _payload(await call("check_action", {"session_id": session, "action": action, "provider": "codex",
                                                     "path": path, "timeout_ms": spec["first_timeout_ms"]}))
        scenarios["first_call_check_action"] = {"wall_ms": round((time.perf_counter() - t0) * 1000, 1),
                                                "cpu_ms": round((self_cpu() - cpu0) * 1000, 1),
                                                "verdict": first.get("verdict")}
        scenarios["check_action_nothing_appended"] = await _timed(check, n)
        scenarios["check_action_after_append"] = await _timed(check, n, grow)
        scenarios["audit_session_nothing_appended"] = await _timed(audit, n)
        scenarios["audit_session_after_append"] = await _timed(audit, n, grow)
        for event in ("PreToolUse", "UserPromptSubmit", "Stop"):
            scenarios[f"codex_session_hook.{event}"] = await _timed(lambda e=event: hook(e), n, grow)
        scenarios["codex_session_hook.PreToolUse_nothing_appended"] = await _timed(lambda: hook("PreToolUse"), n)

    asyncio.run(run())
    memory = self_memory()
    out["peak_rss_mb"] = round(memory.get("peak_rss", 0) / 1048576, 1)
    out["rss_mb"] = round(memory.get("rss", 0) / 1048576, 1)
    out["modules"] = {name: (name in sys.modules) for name in ("numpy", "mcp", "pydantic", "pydantic_settings",
                                                              "neograph_engine", "httpx")}
    print("\n@@BENCH@@" + json.dumps(out))


def warm_cell(sandbox: Sandbox, contracts: int, size_label: str, size_bytes: int, args, log=print) -> dict:
    index = sandbox.index_dir(f"warm-{contracts}-{size_label}")
    contracts_file = index / "contracts.json"
    write_contracts(contracts_file, contracts)
    config = sandbox.write_config(index, contracts_file)
    sid = CODEX_SID.format(contracts % 100 if size_label == "small" else 99)
    path = codex_path(sandbox.sessions, sid)
    transcript = Transcript("codex", path, sid)
    began = time.perf_counter()
    transcript.create(size_bytes)
    log(f"  codex {size_label}: generated {path.stat().st_size / 1e6:.1f} MB in {time.perf_counter() - began:.1f}s")
    spec = {"path": str(path), "sid": sid, "records": transcript.i, "append_bytes": args.append_bytes,
            "initial_bytes": path.stat().st_size,
            "repeat": args.warm_repeat if size_label == "small" else args.large_repeat,
            "first_timeout_ms": 1_800_000}
    spec_path = sandbox.root / f"warm-{contracts}-{size_label}.json"
    spec_path.write_text(json.dumps(spec))
    env = sandbox.env(config, SELF_DIRECT_LOCAL_ONLY="true", SELF_DIRECT_MAX_TOOL_TIMEOUT_SEC="1800",
                      SELF_DIRECT_AUDIT_TIMEOUT_SEC="60", SELF_DIRECT_REQUEST_TIMEOUT_SEC="60")
    done = subprocess.run([sandbox.python, str(Path(__file__).resolve()), "--warm-worker", str(spec_path)],
                          env=env, cwd=str(sandbox.root), capture_output=True, text=True,
                          timeout=args.warm_timeout)
    marker = done.stdout.rfind("@@BENCH@@")
    if done.returncode != 0 or marker < 0:
        raise RuntimeError("warm worker failed: " + (done.stderr or done.stdout)[-600:])
    report = json.loads(done.stdout[marker + len("@@BENCH@@"):])
    report.update(contracts=contracts, session=size_label, session_bytes=spec["initial_bytes"],
                  append_bytes_per_step=args.append_bytes)
    for name, row in report["scenarios"].items():
        if "wall_ms" in row and isinstance(row["wall_ms"], dict):
            log(f"    warm {name:46s} wall p50/p95 {row['wall_ms']['p50']:9.1f}/{row['wall_ms']['p95']:9.1f} ms  "
                f"cpu p50 {row['cpu_ms']['p50']:9.1f} ms  {row['verdicts']}")
        else:
            log(f"    warm {name:46s} wall {row['wall_ms']:9.1f} ms  cpu {row['cpu_ms']:9.1f} ms  {row.get('verdict')}")
    return report


# --- idle servers -----------------------------------------------------------------------------------------------
class StdioServer:
    def __init__(self, sandbox: Sandbox, number: int, profile: str) -> None:
        index = sandbox.index_dir(f"server-{number}")
        config = sandbox.root / f"server-{number}.toml"
        config.write_text("[paths]\n" f"index_dir = {toml_string(index)}\n"
                          f"claude_projects_dir = {toml_string(sandbox.projects)}\n"
                          f"codex_sessions_dir = {toml_string(sandbox.sessions)}\n", encoding="utf-8")
        extra = {"SELF_DIRECT_LOCAL_ONLY": "true"} if profile == "local-only" else {
            "SELF_DIRECT_USE_FAKE_EMBEDDER": "true", "SELF_DIRECT_DENSE_BACKEND": "numpy"}
        self.number = number
        self.proc = subprocess.Popen([sandbox.python, "-m", "self_directing_mcp"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=str(sandbox.root),
                                     env=sandbox.env(config, **extra))
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.started = time.perf_counter()
        self.start_cpu_s: float | None = None
        self.ready_s: float | None = None

    def _pump(self) -> None:
        for raw in self.proc.stdout:
            self.lines.put(raw)

    def send(self, message: dict) -> None:
        self.proc.stdin.write((json.dumps(message) + "\n").encode())
        self.proc.stdin.flush()

    def wait_for(self, request_id: int, timeout: float = 60.0) -> dict:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                raw = self.lines.get(timeout=0.2)
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise RuntimeError("server exited early")
                continue
            try:
                message = json.loads(raw)
            except ValueError:
                continue
            if message.get("id") == request_id:
                return message
        raise TimeoutError(f"no response to request {request_id}")

    def handshake(self) -> None:
        self.send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "bench", "version": "0"}}})
        self.wait_for(1)
        self.ready_s = time.perf_counter() - self.started
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.wait_for(2)
        self.start_cpu_s = process_memory(self.proc).get("cpu_s")

    def use(self) -> None:
        self.send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                   "params": {"name": "audit_status", "arguments": {}}})
        self.wait_for(3, timeout=120)

    def stop(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def idle_servers(sandbox: Sandbox, count: int, profile: str, log=print) -> dict:
    servers = [StdioServer(sandbox, k, profile) for k in range(count)]
    try:
        for server in servers:
            server.handshake()
        time.sleep(1.0)
        after_init = [process_memory(s.proc) for s in servers]
        for server in servers:
            server.use()
        time.sleep(1.0)
        after_use = [process_memory(s.proc) for s in servers]
    finally:
        for server in servers:
            server.stop()

    def block(rows):
        rss = [r.get("rss", 0) / 1048576 for r in rows]
        peak = [r.get("peak_rss", 0) / 1048576 for r in rows]
        pss = [r["pss"] / 1048576 for r in rows if "pss" in r]
        out = {"rss_mb_each": [round(v, 1) for v in rss], "rss_mb_min": round(min(rss), 1),
               "rss_mb_median": round(statistics.median(rss), 1), "rss_mb_max": round(max(rss), 1),
               "rss_mb_total": round(sum(rss), 1), "peak_rss_mb_max": round(max(peak), 1)}
        if pss:
            out["pss_mb_total"] = round(sum(pss), 1)
        if IS_WINDOWS:
            out["private_mb_total"] = round(sum(r.get("private", 0) for r in rows) / 1048576, 1)
        return out

    report = {"count": count, "profile": profile, "after_init": block(after_init), "after_use": block(after_use),
              "server_ready_s": summarize([s.ready_s for s in servers if s.ready_s is not None], 2),
              "server_start_cpu_s": summarize([s.start_cpu_s for s in servers if s.start_cpu_s is not None], 2),
              "note": "RSS sums count shared libraries once per process; pss_mb_total (Linux) divides them fairly."}
    log(f"    {count} idle servers ({profile}): after_init total {report['after_init']['rss_mb_total']} MB "
        f"(each {report['after_init']['rss_mb_min']}-{report['after_init']['rss_mb_max']}), after_use total "
        f"{report['after_use']['rss_mb_total']} MB (each {report['after_use']['rss_mb_min']}-"
        f"{report['after_use']['rss_mb_max']}), start CPU p50 {report['server_start_cpu_s']['p50']} s")
    return report


# --- derived metrics, gates, rendering ---------------------------------------------------------------------------
def derive(report: dict) -> dict:
    values: dict[str, float | None] = {}
    small = [c for c in report.get("cold", []) if c["session"] == "small"]
    cells = [(c["host"], event, row) for c in small for event, row in c["events"].items()]
    gated = [(h, e, row) for h, e, row in cells if e in GATED_COLD_EVENTS.get(h, ())]
    if cells:
        values["cold_hook_wall_p95_all_ms"] = max(row["wall_ms"]["p95"] for _h, _e, row in cells)
        values["cold_hook_wall_p50_ms"] = max(row["wall_ms"]["p50"] for _h, _e, row in cells)
        values["cold_hook_cpu_all_s"] = round(max(row["cpu_ms"]["p95"] for _h, _e, row in cells) / 1000, 3)
    if gated:
        values["cold_hook_wall_p95_ms"] = max(row["wall_ms"]["p95"] for _h, _e, row in gated)
        values["cold_hook_cpu_s"] = round(max(row["cpu_ms"]["p95"] for _h, _e, row in gated) / 1000, 3)
    warm = report.get("warm", [])
    large = [w for w in warm if w["session"] == "large"]
    if large:
        values["warm_check_action_unchanged_cpu_ms"] = large[0]["scenarios"]["check_action_nothing_appended"]["cpu_ms"]["p50"]
    small_warm = [w for w in warm if w["session"] == "small"]
    if small_warm:
        values["warm_incremental_audit_cpu_ms"] = max(
            w["scenarios"]["audit_session_after_append"]["cpu_ms"]["p50"] for w in small_warm)
    idle = report.get("idle_servers")
    if idle:
        values["idle_server_rss_mb"] = idle["after_use"]["rss_mb_max"]
    return values


def evaluate(report: dict, factor: float = CI_FACTOR) -> list[dict]:
    values = derive(report)
    rows = []
    for target in TARGETS:
        value = values.get(target["id"])
        threshold = round(target["target"] * factor, 3)
        if value is None:
            status = "not measured"
        elif not target["gated"]:
            status = "reported, not gated yet" + (" (over target)" if value > target["target"] else " (within target)")
        elif value > threshold:
            status = "FAIL"
        elif value > target["target"]:
            status = "pass (above target, below CI threshold)"
        else:
            status = "pass (within target)"
        rows.append({**{k: target[k] for k in ("id", "label", "unit", "target", "gated", "note")},
                     "ci_threshold": threshold, "value": value, "status": status})
    # Fail-open hooks make the numbers meaningless: they are a failure whatever the thresholds say.
    failed_open = sum(row["failed_open"] for c in report.get("cold", []) for row in c["events"].values())
    failed_open += sum(r["failed_open"] for b in report.get("burst", []) for r in b["results"].values())
    rows.append({"id": "hooks_failed_open", "label": "Hooks that failed open or exited non-zero", "unit": "count",
                 "target": 0, "gated": True, "note": "the benchmark is invalid when a hook cannot judge",
                 "ci_threshold": 0, "value": failed_open, "status": "FAIL" if failed_open else "pass (within target)"})
    return rows


def markdown(report: dict) -> str:
    m = report["machine"]
    lines = ["# Hook cost benchmark", "",
             f"- Machine: {m.get('platform')} ({m.get('machine')}, {m.get('cpu_count')} CPUs"
             f"{', ' + m['cpu_model'] if m.get('cpu_model') else ''}{', WSL' if m.get('wsl') else ''})",
             f"- Python {m.get('python')}; " + ", ".join(f"{k} {v}" for k, v in (m.get("child") or {}).items()
                                                       if k != "python" and v),
             f"- Checkout: {report['repo'].get('describe', '?')}"
             + (f"; load average at start/end {m.get('loadavg_start')} / {m.get('loadavg_end')}"
                if m.get("loadavg_start") else ""),
             f"- Conditions: {json.dumps(report['conditions'])}", ""]
    if report.get("cold"):
        lines += ["## Cold command hooks (fresh process per event)", "",
                  "| host | event | contracts | session | wall p50 | wall p95 | wall max | CPU p50 | CPU p95 | CPU max | peak RSS max | decisions |",
                  "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for cell in report["cold"]:
            for event, row in cell["events"].items():
                w, c = row["wall_ms"], row["cpu_ms"]
                decisions = ", ".join(f"{k}:{v}" for k, v in row["decisions"].items()) + (
                    f" FAILED-OPEN:{row['failed_open']}" if row["failed_open"] else "")
                lines.append(f"| {cell['host']} | {event} | {cell['contracts']} | {cell['session']} "
                             f"({cell['session_bytes'] / 1e6:.0f} MB) | {w['p50']:.0f} | {w['p95']:.0f} | {w['max']:.0f} "
                             f"| {c['p50']:.0f} | {c['p95']:.0f} | {c['max']:.0f} | {row['peak_rss_mb']['max']:.0f} | {decisions} |")
        lines += ["", "Wall and CPU in ms, peak RSS in MB. First run of each event (index build) is in the JSON "
                  "under `first_run`.", ""]
    paired = [c for c in report.get("cold", []) if c.get("versus")]
    if paired:
        other = paired[0]["versus"]["label"]
        lines += [f"## Cold hooks interleaved: this checkout vs {other}", "",
                  "Same transcript and payload for both, alternating order each repetition (cancels machine load).", "",
                  f"| host | event | contracts | wall p50 {other} | wall p50 this | ratio | wall p95 {other} | wall p95 this | "
                  f"CPU p50 {other} | CPU p50 this | ratio | CPU p95 {other} | CPU p95 this |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for cell in paired:
            for event, row in cell["events"].items():
                old_row = cell["versus"]["events"][event]
                wp, cp = (old_row["wall_ms"], row["wall_ms"]), (old_row["cpu_ms"], row["cpu_ms"])
                lines.append(
                    f"| {cell['host']} | {event} | {cell['contracts']} | {wp[0]['p50']:.0f} | {wp[1]['p50']:.0f} | "
                    f"{wp[1]['p50'] / wp[0]['p50']:.2f} | {wp[0]['p95']:.0f} | {wp[1]['p95']:.0f} | {cp[0]['p50']:.0f} | "
                    f"{cp[1]['p50']:.0f} | {cp[1]['p50'] / cp[0]['p50']:.2f} | {cp[0]['p95']:.0f} | {cp[1]['p95']:.0f} |")
        lines.append("")
    if report.get("burst"):
        lines += ["## Concurrent sessions (every session fires at once)", "",
                  "| process | contracts | sessions | wall p50 | wall p95 | wall max | makespan p95 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for burst in report["burst"]:
            for name, row in burst["results"].items():
                lines.append(f"| {name} | {burst['contracts']} | {burst['sessions']} | {row['wall_ms']['p50']:.0f} | "
                             f"{row['wall_ms']['p95']:.0f} | {row['wall_ms']['max']:.0f} | {row['burst_makespan_ms']['p95']:.0f} |")
        lines.append("")
    if report.get("warm"):
        lines += ["## Warm resident-server calls (one process, FastMCP dispatcher)", "",
                  "| session | contracts | call | wall p50 | wall p95 | wall max | CPU p50 | CPU p95 | results |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---|"]
        for cell in report["warm"]:
            tag = f"{cell['session']} ({cell['session_bytes'] / 1e6:.0f} MB)"
            imp = cell["server_import"]
            lines.append(f"| {tag} | {cell['contracts']} | server import (once) | {imp['wall_ms']:.0f} | | | {imp['cpu_ms']:.0f} | | |")
            for name, row in cell["scenarios"].items():
                if isinstance(row["wall_ms"], dict):
                    w, c = row["wall_ms"], row["cpu_ms"]
                    lines.append(f"| {tag} | {cell['contracts']} | {name} | {w['p50']:.1f} | {w['p95']:.1f} | {w['max']:.1f} | "
                                 f"{c['p50']:.1f} | {c['p95']:.1f} | {row['verdicts']} |")
                else:
                    lines.append(f"| {tag} | {cell['contracts']} | {name} (once) | {row['wall_ms']:.0f} | | | {row['cpu_ms']:.0f} | | {row.get('verdict')} |")
        lines += ["", "Wall and CPU in ms; CPU is user+sys of the whole process (all threads).", ""]
    idle = report.get("idle_servers")
    if idle:
        lines += [f"## {idle['count']} idle stdio servers (profile {idle['profile']})", "",
                  "| point | each (MB) | median | max | total RSS | total PSS | peak RSS max |", "|---|---|---:|---:|---:|---:|---:|"]
        for point in ("after_init", "after_use"):
            b = idle[point]
            lines.append(f"| {point} | {b['rss_mb_min']}-{b['rss_mb_max']} | {b['rss_mb_median']} | {b['rss_mb_max']} | "
                         f"{b['rss_mb_total']} | {b.get('pss_mb_total', 'n/a')} | {b['peak_rss_mb_max']} |")
        lines += ["", f"Server ready p50 {idle['server_ready_s']['p50']} s, CPU until ready p50 "
                  f"{idle['server_start_cpu_s']['p50']} s. {idle['note']}", ""]
        total = idle["after_use"]["rss_mb_total"]
        if total > PHASE2_TRIGGERS["total_rss_mb"]:
            lines += [f"Phase-2 entry condition (summed RSS > {PHASE2_TRIGGERS['total_rss_mb']} MB across hosts) is "
                      f"met by this {idle['count']}-server total ({total} MB).", ""]
    lines += ["## Targets and CI thresholds", "",
              f"CI threshold = {report.get('ci_factor', CI_FACTOR)}x target. `gated: no` rows are reported with their "
              "target and never fail the check.", "",
              "| metric | measured | target | CI threshold | gated | status |", "|---|---:|---:|---:|---|---|"]
    for row in report["gates"]:
        value = "n/a" if row["value"] is None else f"{row['value']} {row['unit']}"
        lines.append(f"| {row['label']} | {value} | {row['target']} {row['unit']} | {row['ci_threshold']} {row['unit']} | "
                     f"{'yes' if row['gated'] else 'no'} | {row['status']} |")
    lines.append("")
    return "\n".join(lines)


def compare(before: dict, after: dict) -> str:
    lines = ["# Before / after", "",
             f"- before: {before['repo'].get('describe')} ({before['machine'].get('platform')})",
             f"- after: {after['repo'].get('describe')} ({after['machine'].get('platform')})", ""]

    def key(c):
        return (c["host"], c["contracts"], c["session"])
    base = {key(c): c for c in before.get("cold", [])}
    lines += ["## Cold hooks (ms)", "", "| host | event | contracts | wall p50 before | after | wall p95 before | after | CPU p50 before | after | CPU p95 before | after |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in after.get("cold", []):
        old = base.get(key(cell))
        for event, row in cell["events"].items():
            o = old["events"].get(event) if old else None
            def pair(metric, stat, o=o, row=row):
                return (f"{o[metric][stat]:.0f}" if o else "n/a", f"{row[metric][stat]:.0f}")
            wp50, wp95, cp50, cp95 = pair("wall_ms", "p50"), pair("wall_ms", "p95"), pair("cpu_ms", "p50"), pair("cpu_ms", "p95")
            lines.append(f"| {cell['host']} | {event} | {cell['contracts']} | {wp50[0]} | {wp50[1]} | {wp95[0]} | {wp95[1]} | "
                         f"{cp50[0]} | {cp50[1]} | {cp95[0]} | {cp95[1]} |")
    base_w = {(w["contracts"], w["session"]): w for w in before.get("warm", [])}
    lines += ["", "## Warm calls (ms)", "", "| session | contracts | call | wall p50 before | after | CPU p50 before | after |", "|---|---:|---|---:|---:|---:|---:|"]
    for cell in after.get("warm", []):
        old = base_w.get((cell["contracts"], cell["session"]))
        lines.append(f"| {cell['session']} | {cell['contracts']} | server import (once) | "
                     f"{old['server_import']['wall_ms'] if old else 'n/a'} | {cell['server_import']['wall_ms']} | "
                     f"{old['server_import']['cpu_ms'] if old else 'n/a'} | {cell['server_import']['cpu_ms']} |")
        for name, row in cell["scenarios"].items():
            if not isinstance(row["wall_ms"], dict):
                continue
            o = old["scenarios"].get(name) if old else None
            lines.append(f"| {cell['session']} | {cell['contracts']} | {name} | {o['wall_ms']['p50'] if o else 'n/a'} | "
                         f"{row['wall_ms']['p50']} | {o['cpu_ms']['p50'] if o else 'n/a'} | {row['cpu_ms']['p50']} |")
    bi, ai = before.get("idle_servers"), after.get("idle_servers")
    if ai:
        lines += ["", f"## {ai['count']} idle servers (MB)", "", "| point | each before | each after | total RSS before | total RSS after | total PSS before | total PSS after |", "|---|---|---|---:|---:|---:|---:|"]
        for point in ("after_init", "after_use"):
            o, a = (bi or {}).get(point), ai[point]
            lines.append(f"| {point} | {o['rss_mb_min'] if o else 'n/a'}-{o['rss_mb_max'] if o else ''} | {a['rss_mb_min']}-{a['rss_mb_max']} | "
                         f"{o['rss_mb_total'] if o else 'n/a'} | {a['rss_mb_total']} | {o.get('pss_mb_total', 'n/a') if o else 'n/a'} | {a.get('pss_mb_total', 'n/a')} |")
    return "\n".join(lines) + "\n"


def git_describe(repo: Path) -> dict:
    try:
        rev = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "src", "scripts"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return {"path": str(repo), "rev": rev, "describe": rev + ("+modified" if dirty else "")}
    except Exception:
        return {"path": str(repo), "describe": "unknown"}


# --- main -------------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--warm-worker", metavar="SPEC", help=argparse.SUPPRESS)
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="print a before/after table from two reports")
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="checkout whose scripts/ and src/ are measured")
    p.add_argument("--label", help="name of the measured checkout in the report (default: git revision)")
    p.add_argument("--versus", type=Path, help="a second checkout: cold hooks are measured interleaved against it "
                   "(same transcript, same payloads, alternating order), which cancels load from other processes")
    p.add_argument("--versus-label", default="versus", help="name of the --versus checkout in the report")
    p.add_argument("--python", default=sys.executable, help="interpreter used for hooks, workers and servers")
    p.add_argument("--workdir", type=Path, help="keep synthetic data here (default: a temporary directory)")
    p.add_argument("--out", type=Path, help="write the JSON report here")
    p.add_argument("--markdown", type=Path, help="write the markdown report here")
    p.add_argument("--only", default="cold,burst,warm,servers", help="comma list of: cold, burst, warm, servers")
    p.add_argument("--contracts", default="10,100", help="contract counts (rotating must_not/must/before_regex/requires_success)")
    p.add_argument("--session-mb", type=int, default=1, help="small synthetic session size in MB (default 1)")
    p.add_argument("--large", action="store_true", help="also run the large session (opt-in; never in CI)")
    p.add_argument("--large-mb", type=int, default=300, help="large session size in MB (default 300)")
    p.add_argument("--large-hosts", default="codex,claude", help="hosts for the large session: codex (warm), claude, agy (cold)")
    p.add_argument("--large-contracts", type=int, default=10)
    p.add_argument("--append-bytes", type=int, default=4096, help="bytes appended to the transcript before each cold run/warm step")
    p.add_argument("--runs", type=int, default=20, help="cold repetitions per event (after one discarded warm-up)")
    p.add_argument("--warm-repeat", type=int, default=15)
    p.add_argument("--large-repeat", type=int, default=5, help="repetitions per call/event on the large session")
    p.add_argument("--sessions", type=int, default=4, help="concurrent sessions in the burst")
    p.add_argument("--servers", type=int, default=12, help="idle stdio servers")
    p.add_argument("--server-profile", choices=("local-only", "fake-numpy"), default="local-only")
    p.add_argument("--hook-mode", choices=("local-only", "default"), default="local-only",
                   help="engine mode of the hook processes: local-only (default) or default (fake embedder + NumPy dense)")
    p.add_argument("--hook-timeout", type=float, default=900.0)
    p.add_argument("--warm-timeout", type=float, default=3600.0)
    p.add_argument("--check", action="store_true", help="exit 1 when a gated metric exceeds its CI threshold")
    p.add_argument("--ci-factor", type=float, default=CI_FACTOR,
                   help="CI threshold = target x this factor (default 1.5, the agreed value; widen only on a "
                        "machine that is slower than the reference AMD Ryzen 7 5800X)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.warm_worker:
        warm_worker(args.warm_worker)
        return 0
    if args.compare:
        before, after = (json.loads(Path(p).read_text()) for p in args.compare)
        print(compare(before, after))
        return 0
    only = set(args.only.split(","))
    counts = [int(x) for x in args.contracts.split(",") if x]
    log = lambda *a: print(*a, flush=True)               # noqa: E731
    keep = args.workdir is not None
    root = args.workdir or Path(tempfile.mkdtemp(prefix="tsukkomi-bench-"))
    root.mkdir(parents=True, exist_ok=True)
    sandbox = Sandbox(root.resolve(), args.repo.resolve(), args.python)
    versus = Sandbox(root.resolve(), args.versus.resolve(), args.python) if args.versus else None
    report: dict = {"schema": 1, "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "machine": machine_info(args.python), "repo": git_describe(sandbox.repo),
                    "conditions": {"contracts": counts, "small_session_mb": args.session_mb,
                                   "large": bool(args.large), "large_mb": args.large_mb if args.large else None,
                                   "append_bytes_per_step": args.append_bytes, "cold_runs": args.runs,
                                   "warm_repeat": args.warm_repeat, "concurrent_sessions": args.sessions,
                                   "idle_servers": args.servers, "server_profile": args.server_profile,
                                   "hook_mode": args.hook_mode,
                                   "note": "synthetic data, isolated HOME/index/config, warm page cache"},
                    "targets": TARGETS, "ci_factor": args.ci_factor, "cold": [], "burst": [], "warm": []}
    if args.label:
        report["repo"]["describe"] = args.label
    if versus is not None:
        report["versus"] = {**git_describe(versus.repo), "label": args.versus_label}
    try:
        small = args.session_mb * 1_000_000
        if "cold" in only:
            log("cold hooks (fresh process per event)")
            for n in counts:
                for host in ("claude", "agy"):
                    report["cold"].append(cold_cell(sandbox, host, n, "small", small, args, versus=versus, log=log))
            if args.large:
                for host in ("claude", "agy"):
                    if host in args.large_hosts.split(","):
                        log("large cold session (opt-in)")
                        report["cold"].append(cold_cell(sandbox, host, args.large_contracts, "large",
                                                        args.large_mb * 1_000_000, args, sid_index=9, versus=versus,
                                                        log=log))
        if "burst" in only:
            log("concurrent sessions")
            for n in counts[-1:]:
                report["burst"].append(burst_cell(sandbox, n, args, log=log))
        if "warm" in only:
            log("warm resident-server calls")
            for n in counts:
                report["warm"].append(warm_cell(sandbox, n, "small", small, args, log=log))
            if args.large and "codex" in args.large_hosts.split(","):
                report["warm"].append(warm_cell(sandbox, args.large_contracts, "large", args.large_mb * 1_000_000,
                                                args, log=log))
        if "servers" in only:
            log("idle servers")
            report["idle_servers"] = idle_servers(sandbox, args.servers, args.server_profile, log=log)
        report["gates"] = evaluate(report, args.ci_factor)
        if hasattr(os, "getloadavg"):
            report["machine"]["loadavg_end"] = [round(v, 2) for v in os.getloadavg()]
        text = markdown(report)
        if args.out:
            args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        if args.markdown:
            args.markdown.write_text(text, encoding="utf-8")
        print("\n" + text)
        failed = [row for row in report["gates"] if row["status"] == "FAIL"]
        for row in failed:
            print(f"GATE FAILED: {row['label']}: {row['value']} {row['unit']} > {row['ci_threshold']} {row['unit']}",
                  file=sys.stderr)
        return 1 if (args.check and failed) else 0
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
