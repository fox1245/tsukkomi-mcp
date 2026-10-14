"""MCP through the daemon: ``self-directing-mcp`` as a stdio relay to the same FastMCP tool functions.

Real MCP clients (the SDK's ``stdio_client`` / ``ClientSession``) talk to real relay processes, once standalone
(flag unset) and once with ``SELF_DIRECT_DAEMON=on``; the answers must be equal except for the ``served_by`` stamp the
daemon adds to tool results. Synthetic data only.
"""
from __future__ import annotations

import asyncio
import json
import os
import select
import subprocess
import sys
import threading
import time

import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from self_directing_mcp.daemon import core, mcp_session, protocol as P, relay, rundir
import daemon_support as support
from daemon_support import SID, World


def mcp_env(world, index, flag, **extra):
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(support.ROOT / "src"),
           "PYTHONDONTWRITEBYTECODE": "1", **world.runtime.environ(),
           "SELF_DIRECT_CONFIG_FILE": str(world.config), "SELF_DIRECT_LOCAL_ONLY": "true",
           "SELF_DIRECT_INDEX_DIR": str(index), "SELF_DIRECT_INTEGRITY_BACKGROUND_VERIFY": "false"}
    if flag is not None:
        env["SELF_DIRECT_DAEMON"] = flag
    env.update(extra)
    return env


def server_params(world, index, flag, args=("-m", "self_directing_mcp"), **extra):
    return StdioServerParameters(command=sys.executable, args=list(args), env=mcp_env(world, index, flag, **extra),
                                 cwd=str(world.root))


def decode(result):
    assert not result.isError, result
    return json.loads(result.content[0].text)


def strip_served_by(value):
    if isinstance(value, dict):
        return {k: strip_served_by(v) for k, v in value.items() if k != "served_by"}
    if isinstance(value, list):
        return [strip_served_by(v) for v in value]
    return value


def brief(result):
    """The parts of a check/audit result that must not differ between a standalone server and the daemon."""
    return {"ok": result.get("ok"), "verdict": result.get("verdict"), "action_scope": result.get("action_scope"),
            "coverage_complete": (result.get("coverage") or {}).get("complete"),
            "contracts_snapshot": result.get("contracts_snapshot"), "action_executed": result.get("action_executed"),
            "findings": [{k: f.get(k) for k in ("contract_id", "verdict", "reason", "applicability")}
                         for f in result.get("findings", [])]}


async def exercise(world, params):
    transcript = world.claude[SID].path
    out = {}
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            init = await session.initialize()
            out["initialize"] = init.model_dump(mode="json", by_alias=True, exclude_none=True)
            out["tools"] = [t.model_dump(mode="json", by_alias=True, exclude_none=True)
                            for t in (await session.list_tools()).tools]

            async def call(name, arguments):
                return decode(await session.call_tool(name, arguments))

            def action(command):
                return {"session_id": SID, "provider": "claude", "path": str(transcript),
                        "action": {"tool_name": "Bash", "arguments": {"command": command}}}

            out["deny"] = await call("check_action", action("rm -rf /"))
            out["allow"] = await call("check_action", action("deploy --prod"))
            out["quiet"] = await call("check_action", action("ls"))
            out["audit"] = await call("audit_session", {"session_id": SID, "provider": "claude",
                                                        "path": str(transcript)})
            out["upsert"] = await call("upsert_contracts", {"contracts": [
                {"id": "no-force-push", "type": "must_not", "scope": "tool_call", "regex": r"push\s+--force"}]})
            out["list"] = await call("list_contracts", {})
            out["after"] = await call("check_action", action("git push --force"))
            out["status"] = await call("audit_status", {"session_id": SID, "provider": "claude"})
    return out


def normalized(run):
    return {"initialize": run["initialize"], "tools": run["tools"],
            "deny": brief(run["deny"]), "allow": brief(run["allow"]), "quiet": brief(run["quiet"]),
            "audit": brief(run["audit"]), "after": brief(run["after"]),
            "list": strip_served_by(run["list"]), "upsert": strip_served_by(run["upsert"]),
            "status": {k: run["status"].get(k) for k in ("ok", "degraded", "dense_backend", "local_only")}}


def test_initialize_tools_check_action_and_audit_through_the_proxy_equal_standalone(world, runtime):
    world = world
    standalone = asyncio.run(asyncio.wait_for(exercise(world, server_params(world, world.new_index("s"), None)), 120))
    proxied = asyncio.run(asyncio.wait_for(exercise(world, server_params(world, world.new_index("d"), "on")), 120))
    assert normalized(proxied) == normalized(standalone)
    assert standalone["deny"]["verdict"] == "violation" and standalone["quiet"]["verdict"] == "clean"
    assert proxied["deny"]["verdict"] == "violation" and proxied["after"]["verdict"] == "violation"
    assert [c["id"] for c in proxied["list"]["contracts"]] == [c["id"] for c in standalone["list"]["contracts"]]
    # The daemon stamps what it served; a standalone server's results are unchanged.
    for name in ("deny", "allow", "quiet", "audit", "list", "upsert", "after", "status"):
        assert proxied[name].get("served_by") == "daemon", name
        assert "served_by" not in standalone[name], name
    status = support.daemon_status(runtime)
    assert status["requests"]["mcp_sessions"] == 1 and status["engines"] == 1
    runtime.stop()


def test_two_proxies_with_different_index_dirs_get_different_engines(world, runtime):
    world = world
    first, second = world.new_index("one"), world.new_index("two")
    support.start_daemon_process(runtime)

    async def scenario():
        async with stdio_client(server_params(world, first, "on")) as (r1, w1), \
                stdio_client(server_params(world, second, "on")) as (r2, w2), \
                stdio_client(server_params(world, first, "on")) as (r3, w3):
            async with ClientSession(r1, w1) as one, ClientSession(r2, w2) as two, ClientSession(r3, w3) as third:
                for session in (one, two, third):
                    await session.initialize()
                await one.call_tool("upsert_contracts", {"contracts": [
                    {"id": "only-in-one", "type": "must_not", "scope": "tool_call", "regex": "zzz-one"}]})
                ids = {}
                for name, session in (("one", one), ("two", two), ("third", third)):
                    ids[name] = sorted(c["id"] for c in decode(await session.call_tool("list_contracts", {}))["contracts"])
                during = support.daemon_status(runtime)
                return ids, during

    ids, during = asyncio.run(asyncio.wait_for(scenario(), 120))
    assert "only-in-one" in ids["one"] and "only-in-one" not in ids["two"]      # separate indexes, separate engines
    assert ids["third"] == ids["one"]                                          # the same index shares one engine
    assert during["engines"] == 2 and during["requests"]["mcp_sessions"] == 3
    assert (first / "contracts.json").exists() and (second / "contracts.json").exists()
    runtime.stop()


def test_a_dotenv_in_the_clients_working_directory_is_honoured_by_the_daemon_for_that_session(world, runtime):
    """The standalone server reads ``.env`` in its working directory; the daemon reads the *client's*."""
    world = world
    project = world.root / "project-with-env"
    project.mkdir()
    (project / ".env").write_text("SELF_DIRECT_SEARCH_TOP_K=3\n", encoding="utf-8")
    other = world.root / "project-without-env"
    other.mkdir()

    async def top_k(cwd):
        params = server_params(world, world.new_index("env"), "on")
        params.cwd = str(cwd)
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                return decode(await session.call_tool("audit_status", {}))

    asyncio.run(asyncio.wait_for(top_k(project), 120))
    asyncio.run(asyncio.wait_for(top_k(other), 120))
    assert support.daemon_status(runtime)["engines"] == 2                       # different resolved settings, two engines
    runtime.stop()


# --- --omp-bridge -------------------------------------------------------------------------------------------------

def raw_initialize(proc, request_id=1):
    request = {"jsonrpc": "2.0", "id": request_id, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
    proc.stdin.write((json.dumps(request) + "\n").encode())
    proc.stdin.flush()
    ready, _, _ = select.select([proc.stdout], [], [], 60)
    assert ready, "no answer to initialize"
    return json.loads(proc.stdout.readline())


@pytest.mark.parametrize("failure", ["no_key_file", "missing_key_file", "no_native_library"])
def test_a_failed_omp_bridge_validation_is_an_mcp_error_never_a_silent_fallback(world, runtime, tmp_path, failure):
    world = world
    extra = {"SELF_DIRECT_LOCAL_ONLY": "false"}
    if failure == "missing_key_file":
        extra["SELF_DIRECT_OPENROUTER_API_KEY_FILE"] = str(tmp_path / "absent.env")
    if failure == "no_native_library":
        key = tmp_path / "key.env"
        key.write_text("OPENROUTER_API_KEY=sk-synthetic-not-real\n")
        extra["SELF_DIRECT_OPENROUTER_API_KEY_FILE"] = str(key)
        extra["SELF_DIRECT_SQLITE_VECTOR_PATH"] = str(tmp_path / "missing-vector.so")
    results = {}
    for flag in (None, "on"):
        proc = subprocess.Popen([sys.executable, "-m", "self_directing_mcp", "--omp-bridge"],
                                env=mcp_env(world, world.new_index(f"omp-{flag}"), flag, **extra), cwd=str(world.root),
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            reply = raw_initialize(proc) if flag == "on" else None
            code = proc.wait(timeout=30)
            results[flag] = (reply, code, proc.stderr.read().decode())
        finally:
            if proc.poll() is None:
                proc.kill()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                stream.close()
    standalone_reply, standalone_code, standalone_err = results[None]
    reply, code, stderr = results["on"]
    assert standalone_code == 2 and "OMP bridge configuration is unavailable or invalid" in standalone_err
    assert code == 2 and "OMP bridge configuration is unavailable or invalid" in stderr   # the same message and code
    assert reply["id"] == 1 and reply["error"]["code"] == -32603                           # ... and an MCP error too
    assert "OMP bridge configuration is unavailable or invalid" in reply["error"]["message"]
    assert "sk-synthetic" not in json.dumps(reply) + stderr
    assert support.daemon_status(runtime)["requests"]["mcp_refused"] == 1
    runtime.stop()


def test_a_valid_local_only_omp_bridge_runs_through_the_proxy(world, runtime, tmp_path):
    world = world
    history = tmp_path / "omp-history"
    history.mkdir()
    sid = "bridge-session"
    (history / "session.jsonl").write_text(
        json.dumps({"type": "session", "id": sid}) + "\n" + json.dumps({"type": "message", "message": {
            "role": "user", "content": [{"type": "text", "text": "Do not delete anything."}]}}) + "\n",
        encoding="utf-8")
    index = world.new_index("omp", [])
    params = server_params(world, index, "on", args=("-m", "self_directing_mcp", "--omp-bridge"),
                           SELF_DIRECT_OMP_SESSIONS_DIR=str(history), SELF_DIRECT_SESSION_PROVIDER="codex")

    async def scenario():
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                decode(await session.call_tool("upsert_contracts", {"contracts": [
                    {"id": "no-delete", "type": "must_not", "scope": "tool_call", "provider": "omp",
                     "regex": r"rm\s+-rf"}]}))
                result = decode(await session.call_tool("check_action", {
                    "session_id": sid, "action": {"tool_name": "shell", "arguments": {"command": "rm -rf /x"}}}))
                status = decode(await session.call_tool("audit_status", {"session_id": sid}))
                return result, status

    result, status = asyncio.run(asyncio.wait_for(scenario(), 120))
    assert result["verdict"] == "violation" and result["coverage"]["complete"] is True
    assert status["session"]["chunk_count"] == 1 and status["dense_backend"] == "disabled"
    runtime.stop()


# --- the relay itself ---------------------------------------------------------------------------------------------

PROBE = r"""
import json, sys
from self_directing_mcp import entry
from self_directing_mcp.daemon import relay
relay.pipe = lambda sock: (sock.close(), 0)[1]
baseline = set(sys.modules)
try:
    entry.main([])
except SystemExit as exc:
    code = exc.code
print(json.dumps({"code": code, "loaded": sorted(set(sys.modules) - baseline), "stdlib": sorted(sys.stdlib_module_names)}))
"""


def test_the_relay_imports_only_the_standard_library_and_stays_small(world, runtime):
    world = world
    support.start_daemon_process(runtime)
    env = mcp_env(world, world.new_index("probe"), "on")
    done = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=120,
                          cwd=str(world.root))
    assert done.returncode == 0, done.stderr[-600:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report["code"] == 0
    stdlib = set(report["stdlib"])
    loaded = set(report["loaded"])
    third_party = {n.split(".")[0] for n in loaded if n.split(".")[0] not in stdlib and
                   n.split(".")[0] != "self_directing_mcp" and not n.startswith("_sysconfigdata")}
    assert third_party == set(), third_party
    heavy = {"numpy", "pydantic", "pydantic_settings", "mcp", "anyio", "httpx", "neograph_engine", "dotenv"}
    assert not heavy & {n.split(".")[0] for n in loaded}
    assert not {"self_directing_mcp.server", "self_directing_mcp.engine", "self_directing_mcp.config",
                "self_directing_mcp.schemas", "self_directing_mcp.settings_core"} & loaded
    runtime.stop()


def test_without_the_daemon_the_relay_fails_with_an_mcp_error_and_does_not_start_a_server(world, runtime, tmp_path):
    world = world
    stub = support.make_stub(tmp_path, seconds=30)
    proc = subprocess.Popen([sys.executable, "-m", "self_directing_mcp"], cwd=str(world.root),
                            env=mcp_env(world, world.new_index("fail"), "on", SELF_DIRECT_DAEMON_PYTHON=str(stub)),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    relay_budget = relay.AVAILABILITY_SEC
    try:
        started = time.monotonic()
        reply = raw_initialize(proc)
        code = proc.wait(timeout=relay_budget + 20)
        elapsed = time.monotonic() - started
        stderr = proc.stderr.read().decode()
    finally:
        if proc.poll() is None:
            proc.kill()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            stream.close()
        runtime.kill_leftovers()
    assert code == 1 and reply["error"]["code"] == -32603 and "daemon_unavailable" in reply["error"]["message"]
    assert "daemon_unavailable" in stderr
    assert relay_budget - 2 < elapsed < relay_budget + 10                       # bounded wait, then a clear failure


def test_a_relay_can_be_stopped_by_closing_stdin_and_the_daemon_outlives_it(world, runtime):
    world = world
    proc = subprocess.Popen([sys.executable, "-m", "self_directing_mcp"], cwd=str(world.root),
                            env=mcp_env(world, world.new_index("close"), "on"),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        assert "result" in raw_initialize(proc)
        proc.stdin.close()
        assert proc.wait(timeout=20) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
    status = support.daemon_status(runtime)
    assert status["pid"] != os.getpid() and status["connections"] == 1             # only this status query
    runtime.stop()


# --- lanes, drain -------------------------------------------------------------------------------------------------

def test_hook_requests_are_not_queued_behind_saturated_heavy_work(runtime, inproc):
    release = threading.Event()
    daemon = inproc(core.DaemonConfig(heavy_workers=2), hook_runner=lambda *a: {"systemMessage": "stub"})
    assert daemon.daemon._hook_pool is not daemon.daemon._heavy_pool
    loop = daemon.daemon._loop
    for _ in range(2):                                              # fill the heavy lane with blocked work
        loop.call_soon_threadsafe(lambda: loop.run_in_executor(None, release.wait, 30))
    time.sleep(0.2)
    sock, _ = daemon.connect()
    started = time.monotonic()
    P.write_frame(sock, {"op": "hook", "host": "claude", "event": "PreToolUse", "payload": {"session_id": "s"},
                         "env": {}, "argv": []})
    answer = P.read_frame(sock)
    elapsed = time.monotonic() - started
    sock.close()
    release.set()
    assert answer["ok"] is True and elapsed < 1.0, elapsed


class SlowEngine:
    def __init__(self, settings, gate):
        self.settings, self.gate = settings, gate

    def audit_status(self, **kwargs):
        self.gate.wait(30)
        return {"ok": True, "slow": True}

    def close(self):
        pass


def test_a_draining_daemon_finishes_running_mcp_requests_and_refuses_new_ones(runtime, inproc):
    gate = threading.Event()
    daemon = inproc(hook_runner=lambda *a: {}, engine_factory=lambda settings: SlowEngine(settings, gate),
                    mcp_server=mcp_session.McpHandler())
    sock, _ = daemon.connect()
    P.write_frame(sock, {"op": "mcp", "env": {"SELF_DIRECT_LOCAL_ONLY": "true",
                                              "SELF_DIRECT_INDEX_DIR": str(runtime.root / "mcp-index")},
                         "cwd": str(runtime.root), "mode": "default"})
    assert P.read_frame(sock)["ok"] is True
    reader = sock.makefile("rb")

    def send(message):
        sock.sendall((json.dumps(message) + "\n").encode())

    send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}})
    assert json.loads(reader.readline())["id"] == 1
    send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    send({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "audit_status", "arguments": {}}})
    deadline = time.monotonic() + 10
    while daemon.daemon.inflight < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert daemon.daemon.inflight == 1
    daemon.daemon._loop.call_soon_threadsafe(daemon.daemon.begin_drain, "test")
    time.sleep(0.2)
    send({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
    refused = json.loads(reader.readline())
    assert refused["id"] == 3 and refused["error"]["code"] == -32603 and "restarting" in refused["error"]["message"]
    assert daemon.running
    gate.set()
    finished = json.loads(reader.readline())
    assert finished["id"] == 2 and json.loads(finished["result"]["content"][0]["text"])["slow"] is True
    assert daemon.wait_exit(10)
    assert reader.readline() == b""                                  # the connection closed after the last answer
    sock.close()
