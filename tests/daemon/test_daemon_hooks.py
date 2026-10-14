"""Command hooks through the daemon: parity with the standalone hook, failure behaviour, versions, import budget.

Every scenario runs the real hook scripts (``scripts/claude_hook.py`` / ``scripts/agy_hook.py``) as fresh processes,
once standalone (flag unset) and once with ``SELF_DIRECT_DAEMON=on`` against a daemon of its own, on separate but
identically built indexes of the same synthetic sessions and contracts. Synthetic data only.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from self_directing_mcp import __version__
from self_directing_mcp.daemon import client, core, protocol as P, rundir, served
import daemon_support as support
from daemon_support import AGY_SID, CONTRACTS, HIGH_RISK, SID, UNMET_AUTO_CONTINUE, World

# --- parity -----------------------------------------------------------------------------------------------------------

PARITY = [
    # host, event, command, contracts, what the response must be (so that a scenario cannot pass by judging nothing)
    ("claude", "PreToolUse", "rm -rf /", None, "deny"),
    ("claude", "PreToolUse", "deploy --prod", None, "pass"),         # prerequisite (a successful pytest) satisfied
    ("claude", "Stop", None, None, "pass"),
    ("claude", "Stop", None, [*CONTRACTS, UNMET_AUTO_CONTINUE], "block"),
    ("claude", "SessionStart", None, None, "context"),
    ("claude", "UserPromptSubmit", None, None, "pass"),
    ("claude", "PostToolUse", None, None, "pass"),
    ("agy", "PreToolUse", "rm -rf /", None, "deny"),
    ("agy", "PreToolUse", "deploy --prod", None, "pass"),
    ("agy", "Stop", None, None, "pass"),
    ("agy", "Stop", None, [*CONTRACTS, UNMET_AUTO_CONTINUE], "continue"),
    ("agy", "PreInvocation", None, None, "inject"),
    ("agy", "PostToolUse", None, None, "pass"),
]


def check_shape(host, expect, response):
    text = json.dumps(response)
    assert "UNAVAILABLE" not in text and "FAILED" not in text, f"failed open instead of judging: {text[:300]}"
    if host == "claude":
        decision = response.get("hookSpecificOutput", {}).get("permissionDecision")
        blocked = response.get("decision") == "block"
    else:
        decision = response.get("decision")
        blocked = decision in ("continue", "force_ask", "ask")
    if expect == "deny":
        assert decision == "deny"
    elif expect == "pass":
        assert decision in (None, "allow") and not blocked
    elif expect == "block":
        assert blocked
    elif expect == "continue":
        assert decision == "continue"
    elif expect == "context":
        assert "Contract [" in text
    elif expect == "inject":
        assert "injectSteps" in response and "Contract [" in text


@pytest.mark.parametrize("host,event,command,contracts,expect", PARITY,
                         ids=[f"{h}-{e}-{'-'.join((c or 'none').split()[:1])}-{'unmet' if k else 'base'}"
                              for h, e, c, k, _x in PARITY])
def test_the_daemon_and_the_standalone_hook_print_identical_bytes(world, runtime, host, event, command, contracts,
                                                                  expect):
    standalone_index = world.new_index("standalone", contracts)
    daemon_index = world.new_index("daemon", contracts)
    standalone = world.hook(host, event, command, flag=None, index=standalone_index)
    assert standalone.code == 0, standalone
    check_shape(host, expect, standalone.json)

    world.start_daemon()
    served_run = world.hook(host, event, command, flag="on", index=daemon_index)
    assert served_run.code == 0, served_run
    assert served_run.stdout == standalone.stdout, (served_run, standalone)      # byte for byte, newline included
    assert served_run.stderr == standalone.stderr
    top = set(served_run.json)
    assert "served_by" not in top and "served_by" not in served_run.stdout       # host schemas get no extra key

    status = support.daemon_status(runtime)
    assert status["requests"]["hooks"] == 1                                        # it really was the daemon
    runtime.stop()
    state_file = served.STATE_FILES[host]
    rows = served.read(daemon_index, host)
    assert [(r["event"], r["served_by"], r["total"]) for r in rows] == [(event, "daemon", 1)], rows
    assert (daemon_index / state_file).exists()
    assert served.read(standalone_index, host) == []                              # a standalone hook records nothing


def test_standalone_mode_is_untouched_when_the_flag_is_unset_or_off(world, runtime):
    base = world.hook("claude", "PreToolUse", "rm -rf /", flag=None, index=world.new_index("a"))
    check_shape("claude", "deny", base.json)
    off = world.hook("claude", "PreToolUse", "rm -rf /", flag="off", index=world.new_index("b"))
    assert off.stdout == base.stdout and off.stderr == base.stderr
    assert not runtime.dir.exists()                                                # no daemon was ever involved


@pytest.mark.parametrize("value", ["garbage", "ON", "On", "", "true", "1"])
def test_an_invalid_flag_is_a_settings_error_like_any_other_and_never_starts_a_daemon(world, runtime, value):
    run = world.hook("claude", "PreToolUse", "rm -rf /", flag=value, index=world.new_index("c"))
    assert run.code == 0 and "UNAVAILABLE" in run.stdout and "ValueError" in run.stdout     # visible, fail-open
    assert not runtime.dir.exists()


def test_a_cold_start_spawns_the_daemon_and_the_hook_is_answered(world, runtime):
    """No daemon is running: the first Stop-class hook starts one (budget: seconds) and is judged by it."""
    index = world.new_index("cold")
    run = world.hook("claude", "Stop", flag="on", index=index)
    assert run.code == 0 and "UNAVAILABLE" not in run.stdout, run
    status = support.daemon_status(runtime)
    assert status["requests"]["hooks"] == 1 and status["pid"] != os.getpid()
    spawns = json.loads((runtime.dir / rundir.SPAWNS_NAME).read_text())
    assert len(spawns) == 1 and spawns[0]["pid"] == status["pid"]
    again = world.hook("claude", "Stop", flag="on", index=index)
    assert again.code == 0 and support.daemon_status(runtime)["requests"]["hooks"] == 2
    assert len(json.loads((runtime.dir / rundir.SPAWNS_NAME).read_text())) == 1   # reused, not respawned
    runtime.stop()


# --- concurrency --------------------------------------------------------------------------------------------------------

def test_four_sessions_with_parallel_hooks_get_the_same_answers_as_standalone(runtime, tmp_path_factory):
    """Claude Code runs parallel tool calls, so its hooks overlap freely (every job twice, 16 at a time). An AGY
    conversation's steps are sequential (each has its own step index and a receipt the next judgement sees), so
    its calls keep their order; the four conversations run in parallel."""
    claude_sids = [f"aaaaaaaa-1111-4111-8111-aaaaaaaa{n:04d}" for n in range(1, 5)]
    agy_sids = [f"11111111-1111-4111-8111-1111111111{n:02d}" for n in range(1, 5)]
    world = World(tmp_path_factory.mktemp("burst"), runtime, claude_sids=claude_sids, agy_sids=agy_sids)
    claude_jobs = [("claude", event, command, sid) for sid in claude_sids
                   for event, command in (("PreToolUse", "rm -rf /"), ("PreToolUse", "deploy --prod"),
                                          ("PreToolUse", "ls"), ("Stop", None))]
    agy_sequences = {sid: [("agy", "PreToolUse", "rm -rf /", sid), ("agy", "PreToolUse", "deploy --prod", sid),
                           ("agy", "PreToolUse", "ls", sid), ("agy", "Stop", None, sid)] for sid in agy_sids}

    def sequential(jobs, index, flag, start=0):
        return [world.hook(job[0], job[1], job[2], flag=flag, sid=job[3], index=index, step_offset=start + n)
                for n, job in enumerate(jobs)]

    standalone_index = world.new_index("standalone")
    expected_claude = dict(zip(claude_jobs, sequential(claude_jobs, standalone_index, None)))
    expected_agy = {sid: sequential(jobs, standalone_index, None) for sid, jobs in agy_sequences.items()}

    daemon_index = world.new_index("daemon")
    world.start_daemon()
    numbered = list(enumerate(claude_jobs * 2))

    def run_claude(item):
        n, job = item
        return job, world.hook(job[0], job[1], job[2], flag="on", sid=job[3], index=daemon_index, step_offset=n)

    with ThreadPoolExecutor(max_workers=16) as pool:
        claude_future = pool.map(run_claude, numbered)
        agy_futures = {sid: pool.submit(sequential, jobs, daemon_index, "on") for sid, jobs in agy_sequences.items()}
        claude_results = list(claude_future)
        agy_results = {sid: future.result() for sid, future in agy_futures.items()}
    for job, got in claude_results:
        assert got.code == 0, got
        assert got.stdout == expected_claude[job].stdout, (job, got, expected_claude[job])
    for sid, results in agy_results.items():
        for got, want, job in zip(results, expected_agy[sid], agy_sequences[sid]):
            assert got.code == 0 and got.stdout == want.stdout, (job, got, want)
    status = support.daemon_status(runtime)
    assert status["requests"]["hooks"] == len(claude_jobs) * 2 + sum(len(j) for j in agy_sequences.values())
    assert status["engines"] == 1 and "queue_full" not in status["requests"]
    runtime.stop()


# --- the daemon cannot start ----------------------------------------------------------------------------------------------

@pytest.fixture
def failing_python(tmp_path):
    import shutil as sh
    program = sh.which("false")
    assert program, "needs a `false` executable"
    return program


@pytest.mark.parametrize("host", ["claude", "agy"])
def test_when_the_daemon_cannot_start_the_hook_fails_open_within_its_budget(world, runtime, failing_python, host):
    index = world.new_index("nostart")
    run = world.hook(host, "PreToolUse", "rm -rf /", flag="on", index=index, SELF_DIRECT_DAEMON_PYTHON=failing_python)
    assert run.code == 0
    assert run.ms < (client.AVAILABILITY_FAST_SEC + 0.7) * 1000, run.ms          # budget plus a small margin
    response = run.json
    text = json.dumps(response)
    assert "daemon_unavailable" in text and "fail-open" in text
    assert "deny" not in text and "ask" not in json.dumps(response.get("decision", ""))
    assert "UNAVAILABLE" in text                                                  # a visible warning, never a silence
    # the failure is recorded in the hook health store like any other unavailability
    from self_directing_mcp import agy_hook_state as shared, claude_hook_state as claude_state
    store = claude_state.ClaudeHookState(index) if host == "claude" else shared.HookState(index)
    rows = store.health()
    assert [(r["kind"], r["event"], r["last_error_class"]) for r in rows] == [
        ("audit_unavailable", "PreToolUse", "daemon_unavailable")]


@pytest.mark.parametrize("host,expect_key", [("claude", "hookSpecificOutput"), ("agy", "decision")])
def test_a_matching_high_risk_contract_asks_when_the_daemon_is_unavailable(world, runtime, failing_python, host,
                                                                           expect_key):
    index = world.new_index("risk", [*CONTRACTS, HIGH_RISK])
    asked = world.hook(host, "PreToolUse", "deploy --prod", flag="on", index=index,
                       SELF_DIRECT_DAEMON_PYTHON=failing_python)
    response = asked.json
    decision = (response["hookSpecificOutput"]["permissionDecision"] if host == "claude" else response["decision"])
    reason = (response["hookSpecificOutput"]["permissionDecisionReason"] if host == "claude" else response["reason"])
    assert decision == "ask" and "no-prod-deploy" in reason and "daemon_unavailable" in reason
    other = world.hook(host, "PreToolUse", "ls", flag="on", index=index, SELF_DIRECT_DAEMON_PYTHON=failing_python)
    assert "ask" not in json.dumps(other.json) and "daemon_unavailable" in json.dumps(other.json)
    advisory = world.hook(host, "PreToolUse", "deploy --prod", flag="on", index=index,
                          SELF_DIRECT_DAEMON_PYTHON=failing_python,
                          **{"SELF_DIRECT_CLAUDE_ENFORCEMENT" if host == "claude" else "SELF_DIRECT_AGY_ENFORCEMENT":
                             "advisory"})
    assert "ask" not in json.dumps(advisory.json).replace("fail-open", "")        # advisory mode never asks


RUNNER = r"""
import io, json, runpy, sys
host, event, payload = sys.argv[1], sys.argv[2], sys.argv[3]
script = sys.argv[4]
sys.argv = [script, "--event=" + event, *json.loads(sys.argv[5])]
sys.stdin = io.StringIO(payload)
captured, real = io.StringIO(), sys.stdout
sys.stdout = captured
baseline = set(sys.modules)
try:
    runpy.run_path(script, run_name="__main__")
finally:
    sys.stdout = real
print(json.dumps({"response": captured.getvalue(), "loaded": sorted(set(sys.modules) - baseline),
                  "stdlib": sorted(sys.stdlib_module_names)}))
"""

HEAVY = ("numpy", "pydantic", "pydantic_core", "pydantic_settings", "neograph_engine", "mcp", "httpx", "anyio", "dotenv",
         "regex", "starlette", "zstandard")
HEAVY_OURS = ("self_directing_mcp.engine", "self_directing_mcp.config", "self_directing_mcp.settings_core",
              "self_directing_mcp.schemas", "self_directing_mcp.server", "self_directing_mcp.claude_hooks",
              "self_directing_mcp.agy_hooks", "self_directing_mcp.claude", "self_directing_mcp.agy",
              "self_directing_mcp.hook_policy", "self_directing_mcp.security.mask")


def run_in_probe(world, host, event, command, index, flag="on", **extra_env):
    body = world.payload(host, event, command)
    script = str(support.SCRIPTS / ("claude_hook.py" if host == "claude" else "agy_hook.py"))
    argv = ["--local-only", f"--index-dir={index}"] if host == "claude" else []
    env = world.env(flag, SELF_DIRECT_INDEX_DIR=str(index), **extra_env)
    done = subprocess.run([support.PY, "-c", RUNNER, host, event, json.dumps(body), script, json.dumps(argv)],
                          env=env, capture_output=True, text=True, timeout=120, cwd=world.root)
    assert done.returncode == 0, done.stderr[-800:]
    return json.loads(done.stdout.strip().splitlines()[-1]), done.stderr


def third_party(report):
    stdlib = set(report["stdlib"])
    return sorted({name.split(".")[0] for name in report["loaded"]
                   if name.split(".")[0] not in stdlib and name.split(".")[0] != "self_directing_mcp"
                   and not name.startswith("_sysconfigdata")})


@pytest.mark.parametrize("host", ["claude", "agy"])
def test_the_client_against_a_live_daemon_imports_only_the_standard_library(world, runtime, host):
    index = world.new_index("budget")
    world.start_daemon()
    report, _stderr = run_in_probe(world, host, "PreToolUse", "rm -rf /", index)
    response = json.loads(report["response"])
    check_shape(host, "deny", response)                                            # judged by the daemon
    loaded = set(report["loaded"])
    top = {name.split(".")[0] for name in loaded}
    assert not set(HEAVY) & top, sorted(set(HEAVY) & top)
    assert not set(HEAVY_OURS) & loaded, sorted(set(HEAVY_OURS) & loaded)
    assert third_party(report) == [], third_party(report)
    ours = sorted(name for name in loaded if name.startswith("self_directing_mcp"))
    assert set(ours) <= {"self_directing_mcp", "self_directing_mcp.daemon", "self_directing_mcp.daemon.client",
                         "self_directing_mcp.daemon.protocol", "self_directing_mcp.daemon.rundir"}, ours
    runtime.stop()


@pytest.mark.parametrize("host", ["claude", "agy"])
def test_the_engine_is_never_imported_when_the_daemon_cannot_start(world, runtime, failing_python, host):
    index = world.new_index("noengine")
    report, stderr = run_in_probe(world, host, "PreToolUse", "rm -rf /", index,
                                  SELF_DIRECT_DAEMON_PYTHON=failing_python)
    assert "daemon_unavailable" in report["response"]
    loaded = set(report["loaded"])
    top = {name.split(".")[0] for name in loaded}
    assert not set(HEAVY) & top, sorted(set(HEAVY) & top)
    assert not (set(HEAVY_OURS) - {"self_directing_mcp.security.mask"}) & loaded, sorted(set(HEAVY_OURS) & loaded)
    assert "self_directing_mcp.engine" not in loaded and third_party(report) == []


def test_a_slow_to_start_daemon_costs_the_fast_events_only_their_budget_and_stop_its_longer_one(
        world, runtime, tmp_path):
    stub = support.make_stub(tmp_path, seconds=6)
    index = world.new_index("slow")
    fast = world.hook("claude", "PreToolUse", "ls", flag="on", index=index, SELF_DIRECT_DAEMON_PYTHON=str(stub))
    assert client.AVAILABILITY_FAST_SEC * 1000 <= fast.ms < (client.AVAILABILITY_FAST_SEC + 0.8) * 1000, fast
    assert "daemon_unavailable" in fast.stdout
    runtime.stop(timeout=1)                                                        # the stub is a test process
    runtime.kill_leftovers()
    slow = world.hook("claude", "Stop", flag="on", index=index, SELF_DIRECT_DAEMON_PYTHON=str(stub))
    assert client.AVAILABILITY_SLOW_SEC * 1000 <= slow.ms < (client.AVAILABILITY_SLOW_SEC + 1.2) * 1000, slow
    assert "daemon_unavailable" in slow.stdout
    runtime.kill_leftovers()


def test_the_availability_budgets_are_named_constants():
    assert client.AVAILABILITY_FAST_SEC == 0.3 and client.AVAILABILITY_SLOW_SEC == 3.0
    assert {"PreToolUse", "PostToolUse", "UserPromptSubmit", "PreInvocation"} <= set(client.FAST_EVENTS)
    assert {"Stop", "SubagentStop", "SessionStart"} <= set(client.SLOW_EVENTS)
    assert client.availability_budget("PreToolUse") == 0.3 and client.availability_budget("Stop") == 3.0
    assert client.availability_budget(None) == 0.3


# --- crash-loop guard -----------------------------------------------------------------------------------------------------

def test_more_than_three_spawns_in_five_minutes_stops_spawning_and_warns(world, runtime, failing_python):
    index = world.new_index("loop")
    env = {"SELF_DIRECT_DAEMON_PYTHON": failing_python}
    runs = [world.hook("claude", "PreToolUse", "ls", flag="on", index=index, **env) for _ in range(7)]
    spawns = json.loads((runtime.dir / rundir.SPAWNS_NAME).read_text())
    assert len(spawns) == client.CRASH_LOOP_SPAWNS + 1                              # "more than 3", then no more
    for run in runs[:4]:
        assert "crash-loop" not in run.stderr
    for run in runs[4:]:
        assert "crash-loop" in run.stderr
        assert "daemon_unavailable:crash-loop" in run.stdout and "fail-open" in run.stdout
        assert run.ms < 1500
    # Old entries age out: the window is five minutes.
    for record in spawns:
        record["t"] -= client.CRASH_LOOP_WINDOW_SEC + 5
    (runtime.dir / rundir.SPAWNS_NAME).write_text(json.dumps(spawns))
    again = world.hook("claude", "PreToolUse", "ls", flag="on", index=index, **env)
    assert "crash-loop" not in again.stderr
    assert len(json.loads((runtime.dir / rundir.SPAWNS_NAME).read_text())) == 1


def test_a_daemon_that_is_still_starting_is_waited_for_not_started_twice(world, runtime, tmp_path):
    stub = support.make_stub(tmp_path, seconds=5)
    index = world.new_index("grace")
    env = {"SELF_DIRECT_DAEMON_PYTHON": str(stub)}
    first = world.hook("claude", "PreToolUse", "ls", flag="on", index=index, **env)
    second = world.hook("claude", "PreToolUse", "ls", flag="on", index=index, **env)
    third = world.hook("agy", "PreToolUse", "ls", flag="on", index=index, **env)
    assert all("daemon_unavailable" in r.stdout for r in (first, second, third))
    assert len(json.loads((runtime.dir / rundir.SPAWNS_NAME).read_text())) == 1     # one start, two waits
    runtime.kill_leftovers()


# --- in-process client behaviour against a daemon in this process ---------------------------------------------------------

class Hold:
    def __init__(self, hold=True):
        self.event = threading.Event()
        if not hold:
            self.event.set()
        self.calls = []

    def __call__(self, host, event, payload, env, argv):
        self.calls.append(event)
        self.event.wait(30)
        return {"systemMessage": "stub"}


def claude_payload(world, command="ls"):
    return world.payload("claude", "PreToolUse", command)


def test_queue_full_makes_the_client_fail_open_with_that_reason(world, runtime, inproc, monkeypatch):
    hold = Hold()
    daemon = inproc(core.DaemonConfig(session_bound=1), hook_runner=hold)
    payload = claude_payload(world)
    first = threading.Thread(target=client.serve_hook, args=("claude", "PreToolUse", payload, []))
    first.start()
    deadline = time.monotonic() + 10
    while daemon.daemon.inflight < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    index = world.new_index("qf")
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(index))
    response, served_by_daemon = client.serve_hook("claude", "PreToolUse", payload, [])
    assert served_by_daemon is False
    text = json.dumps(response)
    assert "queue_full" in text and "fail-open" in text and "permissionDecision" not in text
    hold.event.set()
    first.join(10)
    ok, from_daemon = client.serve_hook("claude", "PreToolUse", payload, [])
    assert from_daemon is True and ok == {"systemMessage": "stub"}


def test_an_older_client_gets_version_mismatch_and_a_newer_one_replaces_the_daemon(world, runtime, inproc, monkeypatch,
                                                                                    tmp_path):
    hold = Hold(hold=False)
    daemon = inproc(hook_runner=hold, version="9.0.0")                              # a daemon newer than this client
    index = world.new_index("ver")
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(index))
    payload = claude_payload(world)
    response, served_by_daemon = client.serve_hook("claude", "PreToolUse", payload, [])
    assert served_by_daemon is False and "version_mismatch" in json.dumps(response)
    assert daemon.running and not daemon.daemon.draining and not hold.calls         # an older client never replaces it
    daemon.stop()

    old = inproc(hook_runner=hold, version="0.0.1")                                 # a daemon older than this client
    stub = support.make_stub(tmp_path, seconds=4)
    monkeypatch.setenv("SELF_DIRECT_DAEMON_PYTHON", str(stub))                      # the successor will not come up
    response, served_by_daemon = client.serve_hook("claude", "PreToolUse", payload, [])
    assert served_by_daemon is False and "daemon_unavailable" in json.dumps(response)   # switchover: fail open
    assert old.wait_exit(10)                                                         # the old daemon drained and left
    assert not hold.calls                                                            # ... without judging anything
    runtime.kill_leftovers()


def test_a_newer_client_starts_a_real_successor_that_takes_over(world, runtime, inproc, monkeypatch):
    hold = Hold(hold=False)
    old = inproc(hook_runner=hold, version="0.0.1")
    index = world.new_index("handover")
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(index))
    monkeypatch.setenv("SELF_DIRECT_LOCAL_ONLY", "true")
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(world.config))
    monkeypatch.setenv("SELF_DIRECT_DAEMON_PYTHON", support.PY)
    payload = world.payload("claude", "Stop")
    response, served_by_daemon = client.serve_hook("claude", "Stop", payload, ["--local-only", f"--index-dir={index}"])
    assert served_by_daemon is True, response                                       # Stop's budget covers the handover
    assert old.wait_exit(10) and not hold.calls
    status = support.daemon_status(runtime)
    assert status["pid"] != os.getpid() and status["requests"]["hooks"] == 1
    assert len(json.loads((runtime.dir / rundir.SPAWNS_NAME).read_text())) == 1
    runtime.stop()


class FakeDaemon:
    """A socket that speaks just enough protocol to misbehave in one way."""

    def __init__(self, runtime, behaviour):
        self.behaviour = behaviour
        area = rundir.open_runtime_dir(runtime.environ())
        self.area = area
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(area.address())
        self.server.listen(4)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            try:
                P.read_frame(conn)
                P.write_frame(conn, {"op": "hello", "ok": True, "proto": P.PROTO, "package_version": __version__,
                                     "relation": "same", "draining": False})
                P.read_frame(conn)
                self.behaviour(conn)
            except Exception:
                pass
            finally:
                if self.behaviour.__name__ != "silent":
                    conn.close()

    def close(self):
        self.server.close()
        self.area.unlink(rundir.SOCKET_NAME)
        self.area.close()


def die(conn):
    pass                                                 # the daemon "crashes" after reading the request


def silent(conn):
    time.sleep(30)


def test_a_daemon_that_dies_mid_request_means_fail_open_not_a_hang(world, runtime, monkeypatch):
    index = world.new_index("died")
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(index))
    fake = FakeDaemon(runtime, die)
    try:
        started = time.monotonic()
        response, from_daemon = client.serve_hook("claude", "PreToolUse", claude_payload(world), [])
        assert from_daemon is False and "daemon_unavailable" in json.dumps(response)
        assert time.monotonic() - started < 2
    finally:
        fake.close()


def test_a_daemon_that_never_answers_is_cut_off_at_the_request_budget(world, runtime, monkeypatch):
    index = world.new_index("silent")
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(index))
    monkeypatch.setenv("SELF_DIRECT_REQUEST_TIMEOUT_SEC", "0.3")
    monkeypatch.setattr(client, "RESPONSE_MARGIN_SEC", 0.2)
    fake = FakeDaemon(runtime, silent)
    try:
        started = time.monotonic()
        response, from_daemon = client.serve_hook("claude", "PreToolUse", claude_payload(world), [])
        elapsed = time.monotonic() - started
        assert from_daemon is False and "daemon_timeout" in json.dumps(response)
        assert 0.4 < elapsed < 3
    finally:
        fake.close()


def test_an_unsafe_runtime_directory_fails_open_with_the_reason_on_stderr(world, runtime, monkeypatch, capsys):
    runtime.dir.mkdir(mode=0o755)
    os.chmod(runtime.dir, 0o755)
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(world.new_index("unsafe")))
    response, from_daemon = client.serve_hook("claude", "PreToolUse", claude_payload(world), [])
    assert from_daemon is False and "daemon_unavailable" in json.dumps(response)
    assert "must be exactly 0700" in capsys.readouterr().err


def test_an_oversized_payload_fails_open_before_anything_is_started(world, runtime, monkeypatch):
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(world.new_index("big")))
    payload = claude_payload(world)
    payload["tool_input"] = {"command": "x" * (P.MAX_FRAME_BYTES + 10)}
    response, from_daemon = client.serve_hook("claude", "PreToolUse", payload, [])
    assert from_daemon is False and "frame_too_large" in json.dumps(response)
    assert not runtime.dir.exists()


# --- flag, platform, constants ----------------------------------------------------------------------------------------------

def test_flag_states():
    assert client.flag_state({}) == "off"
    assert client.flag_state({"SELF_DIRECT_DAEMON": "on"}) == "on"
    assert client.flag_state({"self_direct_daemon": "off"}) == "off"
    for bad in ("ON", "true", "1", "", " on", "auto"):
        assert client.flag_state({"SELF_DIRECT_DAEMON": bad}) == "invalid"
    assert client.use_daemon({"SELF_DIRECT_DAEMON": "off"}) is False
    assert client.use_daemon({"SELF_DIRECT_DAEMON": "bogus"}) is False


def test_on_an_unsupported_platform_the_flag_behaves_as_off_and_says_so_once(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "platform", "win32")
    environ = {"SELF_DIRECT_DAEMON": "on", "HOME": str(tmp_path)}
    assert client.use_daemon(environ) is False
    first = capsys.readouterr().err
    assert "ignored on this platform" in first and "standalone" in first
    assert client.use_daemon(environ) is False
    assert capsys.readouterr().err == ""                                             # once, not on every hook
    monkeypatch.setattr(sys, "platform", "linux")
    assert client.use_daemon(environ) is True


def test_the_client_constants_agree_with_the_settings():
    from self_directing_mcp import settings_core
    assert client.DEFAULT_REQUEST_TIMEOUT_SEC == settings_core.DEFAULT_REQUEST_TIMEOUT_SEC
    assert core.DEFAULT_REQUEST_TIMEOUT_SEC == settings_core.DEFAULT_REQUEST_TIMEOUT_SEC
    assert client.response_timeout({}) == settings_core.DEFAULT_REQUEST_TIMEOUT_SEC + client.RESPONSE_MARGIN_SEC
    assert client.response_timeout({"SELF_DIRECT_REQUEST_TIMEOUT_SEC": "100"}) == 100 + client.RESPONSE_MARGIN_SEC
    assert client.response_timeout({"SELF_DIRECT_REQUEST_TIMEOUT_SEC": "-3"}) == settings_core.DEFAULT_REQUEST_TIMEOUT_SEC + client.RESPONSE_MARGIN_SEC


def test_the_spawned_daemon_environment_carries_no_settings_and_no_credentials():
    environ = {"PATH": "/bin", "HOME": "/h", "XDG_RUNTIME_DIR": "/run/u", "SELF_DIRECT_INDEX_DIR": "/idx",
               "OPENROUTER_API_KEY": "sk-synthetic", "SELF_DIRECT_OPENROUTER_API_KEY_FILE": "/k", "LC_ALL": "C",
               "SELF_DIRECT_DAEMON_IDLE_EXIT_SEC": "60", "AWS_SECRET_ACCESS_KEY": "x", "TSUKKOMI_WORKFLOW_STATE": "/s"}
    env = client.daemon_environment(environ)
    assert env["PATH"] == "/bin" and env["XDG_RUNTIME_DIR"] == "/run/u" and env["LC_ALL"] == "C"
    assert env["SELF_DIRECT_DAEMON_IDLE_EXIT_SEC"] == "60"
    for name in ("SELF_DIRECT_INDEX_DIR", "OPENROUTER_API_KEY", "SELF_DIRECT_OPENROUTER_API_KEY_FILE",
                 "AWS_SECRET_ACCESS_KEY", "TSUKKOMI_WORKFLOW_STATE"):
        assert name not in env
    assert os.path.isdir(os.path.join(env["PYTHONPATH"].split(os.pathsep)[0], "self_directing_mcp"))


def test_no_credential_value_ever_reaches_the_daemon_process(world, runtime):
    """A key in the client's environment is not forwarded; the daemon's own logs and status never contain it."""
    secret = "sk-synthetic-NOT-A-REAL-KEY-0000"
    index = world.new_index("nokey")
    world.start_daemon()
    run = world.hook("claude", "PreToolUse", "rm -rf /", flag="on", index=index, OPENROUTER_API_KEY=secret,
                     SELF_DIRECT_OPENROUTER_API_KEY=secret, SELF_DIRECT_SOME_TOKEN=secret)
    assert "deny" in run.stdout
    status_text = json.dumps(support.daemon_status(runtime))
    log_text = (runtime.dir / rundir.LOG_NAME).read_text()
    assert secret not in status_text and secret not in log_text
    pid = json.loads(status_text)["pid"]
    environ = open(f"/proc/{pid}/environ", "rb").read().decode("utf-8", "replace")       # our own test daemon only
    assert secret not in environ
    runtime.stop()
