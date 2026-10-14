"""The daemon process: handshake, authentication, bounds, versions, idle exit, memory drain, logging, engines.

These tests run a real ``Daemon`` on its own event loop inside the test process (private runtime directory) with a
stub hook runner where the judgement itself is irrelevant, and with the real runner where it is the point.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time

import pytest

from self_directing_mcp import __version__
from self_directing_mcp.daemon import core, envview, protocol as P, rundir
import daemon_support as support


class Gate:
    """A hook runner that answers ``{"systemMessage": "stub"}`` once released (immediately when not held)."""

    def __init__(self, hold: bool = False) -> None:
        self.event = threading.Event()
        if not hold:
            self.event.set()
        self.calls: list[tuple] = []

    def __call__(self, host, event, payload, env, argv):
        self.calls.append((host, event, payload, env, argv))
        assert self.event.wait(30), "the test never released the stub hook"
        return {"systemMessage": "stub"}


def hook_frame(session="s1", host="claude", event="PreToolUse", payload=None):
    payload = payload if payload is not None else {"session_id": session}
    return {"op": P.OP_HOOK, "host": host, "event": event, "payload": payload, "env": {}, "argv": []}


def request(daemon, frame, *, version=None, proto=None):
    """One connection, hello, one frame; returns ``(hello_reply, answer)``."""
    sock, hello = daemon.connect(version=version, proto=proto)
    try:
        P.write_frame(sock, frame)
        return hello, P.read_frame(sock)
    finally:
        sock.close()


def in_thread(fn, *args):
    box = {}

    def run():
        try:
            box["value"] = fn(*args)
        except BaseException as exc:                    # noqa: BLE001
            box["error"] = exc
    thread = threading.Thread(target=run)
    thread.start()
    thread.box = box
    return thread


def wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# --- handshake --------------------------------------------------------------------------------------------------

def test_hello_reports_the_versions_and_facts_a_client_needs(inproc):
    daemon = inproc(hook_runner=Gate())
    assert daemon.daemon._ready.wait(60)
    sock, hello = daemon.connect()
    sock.close()
    assert hello["op"] == "hello" and hello["ok"] is True
    assert hello["proto"] == P.PROTO and hello["package_version"] == __version__ and hello["relation"] == "same"
    assert hello["pid"] == os.getpid() and hello["draining"] is False
    assert {"hook", "mcp", "drain", "status"} <= set(hello["features"])
    assert set(hello["parser_versions"]) == {"codex", "omp", "claude", "agy"}
    assert all(isinstance(v, int) for v in hello["parser_versions"].values())
    assert isinstance(hello["evaluator_version"], int)
    assert set(hello["index_schema"]) == {"incremental_state", "analysis", "graph", "contracts"}


def test_the_first_frame_must_be_hello_and_a_bad_peer_does_not_hurt_the_daemon(inproc):
    daemon = inproc(hook_runner=Gate())
    sock, _ = daemon.connect(hello=False)
    P.write_frame(sock, hook_frame())                          # no hello first
    with pytest.raises((P.ConnectionClosed, ConnectionError, P.ProtocolError)):
        P.read_frame(sock)
    sock.close()
    assert request(daemon, hook_frame())[1]["ok"] is True      # and the daemon is fine
    assert wait_for(lambda: daemon.daemon.stats["protocol_errors"] >= 1)


@pytest.mark.parametrize("header", [struct.pack(">I", P.MAX_FRAME_BYTES + 1), struct.pack(">I", 0),
                                    struct.pack(">I", 0xFFFFFFFF), struct.pack(">I", 12) + b"not json!!!!!"])
def test_hostile_frames_close_that_connection_only(inproc, header):
    daemon = inproc(hook_runner=Gate())
    sock, hello = daemon.connect()
    sock.sendall(header)
    sock.settimeout(5)
    with pytest.raises((P.ConnectionClosed, ConnectionError, P.ProtocolError, OSError)):
        P.read_frame(sock)
    sock.close()
    assert request(daemon, hook_frame())[1]["ok"] is True
    assert wait_for(lambda: daemon.daemon.stats["protocol_errors"] >= 1)


def test_a_silent_connection_is_dropped_after_the_hello_timeout(inproc, monkeypatch):
    monkeypatch.setattr(core, "HELLO_TIMEOUT_SEC", 0.3)
    daemon = inproc(hook_runner=Gate())
    sock, _ = daemon.connect(hello=False)
    sock.settimeout(5)
    with pytest.raises((P.ConnectionClosed, ConnectionError, OSError)):
        P.read_frame(sock)
    sock.close()


# --- authentication -----------------------------------------------------------------------------------------------

def test_a_peer_with_another_user_id_is_rejected_and_never_reaches_the_runner(inproc, monkeypatch):
    gate = Gate()
    daemon = inproc(hook_runner=gate)
    assert request(daemon, hook_frame())[1]["ok"] is True       # control: this peer is accepted
    real = socket.socket.getsockopt

    def foreign(self, level, option, *args):
        if level == socket.SOL_SOCKET and option == socket.SO_PEERCRED:
            return struct.pack("3i", os.getpid(), os.geteuid() + 1, os.getegid())     # somebody else's uid
        return real(self, level, option, *args)

    monkeypatch.setattr(socket.socket, "getsockopt", foreign)
    sock, _ = daemon.connect(hello=False)
    try:
        P.write_frame(sock, P.hello_request("test", __version__))
        with pytest.raises((P.ConnectionClosed, ConnectionError, OSError)):
            P.read_frame(sock)                                   # closed without a word
    finally:
        sock.close()
    assert wait_for(lambda: daemon.daemon.stats["rejected_peer"] == 1)
    assert len(gate.calls) == 1                                  # only the control request ever ran
    monkeypatch.undo()
    assert request(daemon, hook_frame())[1]["ok"] is True        # the daemon went on serving its own user


def test_a_peer_whose_credentials_cannot_be_read_is_rejected(inproc, monkeypatch):
    daemon = inproc(hook_runner=Gate())
    real = socket.socket.getsockopt

    def broken(self, level, option, *args):
        if level == socket.SOL_SOCKET and option == socket.SO_PEERCRED:
            raise OSError("no credentials")
        return real(self, level, option, *args)

    monkeypatch.setattr(socket.socket, "getsockopt", broken)
    sock, _ = daemon.connect(hello=False)
    try:
        P.write_frame(sock, P.hello_request("test", __version__))
        with pytest.raises((P.ConnectionClosed, ConnectionError, OSError)):
            P.read_frame(sock)
    finally:
        sock.close()


def test_the_socket_is_private_and_only_unix(inproc):
    daemon = inproc(hook_runner=Gate())
    path = os.path.join(daemon.runtime.path, rundir.SOCKET_NAME)
    import stat
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    sockets = [s for s in (daemon.daemon._server.sockets or [])]
    assert sockets and all(s.family == socket.AF_UNIX for s in sockets)


# --- bounds ---------------------------------------------------------------------------------------------------------

def test_requests_beyond_the_per_session_bound_answer_unknown_queue_full(inproc):
    gate = Gate(hold=True)
    daemon = inproc(core.DaemonConfig(session_bound=2), hook_runner=gate)
    first = [in_thread(request, daemon, hook_frame("busy")) for _ in range(2)]
    assert wait_for(lambda: daemon.daemon.inflight_by_session["busy"] == 2)
    _hello, over = request(daemon, hook_frame("busy"))
    assert over == {"op": "result", "ok": False, "status": "unknown", "reason": "queue_full"}
    other = in_thread(request, daemon, hook_frame("quiet"))      # another session is not affected
    assert wait_for(lambda: daemon.daemon.inflight == 3)
    gate.event.set()
    for thread in (*first, other):
        thread.join(10)
        assert thread.box["value"][1]["ok"] is True
    assert request(daemon, hook_frame("busy"))[1]["ok"] is True  # the bound frees up
    assert daemon.daemon.stats["queue_full"] == 1
    assert daemon.daemon.inflight == 0 and not daemon.daemon.inflight_by_session


def test_the_total_bound_applies_across_sessions(inproc):
    gate = Gate(hold=True)
    daemon = inproc(core.DaemonConfig(session_bound=5, total_bound=3), hook_runner=gate)
    running = [in_thread(request, daemon, hook_frame(f"s{i}")) for i in range(3)]
    assert wait_for(lambda: daemon.daemon.inflight == 3)
    assert request(daemon, hook_frame("s9"))[1]["reason"] == "queue_full"
    gate.event.set()
    for thread in running:
        thread.join(10)
        assert thread.box["value"][1]["ok"] is True


def test_a_runner_failure_is_reported_as_internal_error_without_its_message(inproc):
    def broken(host, event, payload, env, argv):
        raise RuntimeError("secret-looking text that must not travel: sk-synthetic-0000")

    daemon = inproc(hook_runner=broken)
    _hello, answer = request(daemon, hook_frame())
    assert answer == {"op": "result", "ok": False, "status": "unknown", "reason": "internal_error"}
    assert "sk-synthetic" not in json.dumps(answer) and "sk-synthetic" not in (
        (daemon.runtime_area.dir / rundir.LOG_NAME).read_text())


@pytest.mark.parametrize("frame", [
    {"op": "hook", "host": "codex", "event": "PreToolUse", "payload": {}},
    {"op": "hook", "host": "claude", "event": 5, "payload": {}},
    {"op": "hook", "host": "claude", "event": "Stop"},
    {"op": "hook", "host": "claude", "event": "Stop", "payload": {}, "env": "x"},
    {"op": "hook", "host": "claude", "event": "Stop", "payload": {}, "argv": [1]},
    {"op": "nonsense"},
])
def test_malformed_requests_are_answered_bad_request(inproc, frame):
    gate = Gate()
    daemon = inproc(hook_runner=gate)
    _hello, answer = request(daemon, frame)
    assert answer["ok"] is False and answer["reason"] == "bad_request" and not gate.calls


# --- versions ---------------------------------------------------------------------------------------------------------

def test_an_older_client_is_told_version_mismatch_and_cannot_replace_or_drain_the_daemon(inproc):
    gate = Gate()
    daemon = inproc(hook_runner=gate, version="0.6.0")
    for version, proto in (("0.5.9", None), ("0.0.1", None), ("0.5.0", P.PROTO + 1)):     # older, older, mixed
        hello, answer = request(daemon, hook_frame(), version=version, proto=proto)
        assert hello["relation"] in ("older", "mixed")
        assert answer == {"op": "result", "ok": False, "status": "unknown", "reason": "version_mismatch"}
        _hello, drain = request(daemon, {"op": "drain"}, version=version, proto=proto)
        assert drain["ok"] is False and drain["reason"] == "version_mismatch"
    assert daemon.running and not daemon.daemon.draining and not gate.calls
    assert request(daemon, hook_frame(), version="0.6.0")[1]["ok"] is True


@pytest.mark.parametrize("version,proto", [("0.7.0", None), ("0.6.1", None), ("0.6.0", P.PROTO + 1)])
def test_a_newer_client_is_refused_service_and_may_drain_the_daemon(inproc, version, proto):
    gate = Gate()
    daemon = inproc(hook_runner=gate, version="0.6.0")
    sock, hello = daemon.connect(version=version, proto=proto)
    assert hello["relation"] == "newer"
    P.write_frame(sock, hook_frame())
    assert P.read_frame(sock)["reason"] == "version_mismatch"      # never judged by the older daemon
    P.write_frame(sock, {"op": "drain"})
    assert P.read_frame(sock) == {"op": "drain", "ok": True, "draining": True}
    sock.close()
    assert daemon.wait_exit(10)
    assert not gate.calls
    assert not daemon.runtime.exists(rundir.SOCKET_NAME) and not daemon.runtime.exists(rundir.PID_NAME)


def test_a_drain_lets_in_flight_requests_finish_and_refuses_new_ones(inproc):
    gate = Gate(hold=True)
    daemon = inproc(hook_runner=gate, version="0.6.0")
    running = in_thread(request, daemon, hook_frame("a"))
    assert wait_for(lambda: daemon.daemon.inflight == 1)
    held, _ = daemon.connect(version="0.6.0")                       # an existing connection
    request(daemon, {"op": "drain"}, version="0.7.0")
    assert wait_for(lambda: daemon.daemon.draining)
    assert not daemon.runtime.exists(rundir.SOCKET_NAME)             # no new clients find this daemon any more
    with pytest.raises(OSError):
        daemon.connect()
    P.write_frame(held, hook_frame("b"))
    assert P.read_frame(held)["reason"] == "draining"                # nothing new starts on an old connection
    held.close()
    assert daemon.running                                            # a request is still running
    gate.event.set()
    running.join(10)
    assert running.box["value"][1]["ok"] is True                     # and it was answered
    assert daemon.wait_exit(10)
    assert len(gate.calls) == 1


def test_force_drain_is_how_the_cli_stops_a_daemon(inproc):
    daemon = inproc(hook_runner=Gate())
    _hello, answer = request(daemon, {"op": "drain", "force": True})
    assert answer["ok"] is True and daemon.wait_exit(10)


# --- idle exit --------------------------------------------------------------------------------------------------------

def test_the_daemon_exits_after_the_idle_time_and_cleans_up(inproc):
    daemon = inproc(core.DaemonConfig(idle_exit_sec=0.5), hook_runner=Gate())
    assert request(daemon, hook_frame())[1]["ok"] is True
    began = time.monotonic()
    assert daemon.wait_exit(10)
    assert 0.4 <= time.monotonic() - began < 5
    area = daemon.runtime_area.dir
    assert not (area / rundir.SOCKET_NAME).exists() and not (area / rundir.PID_NAME).exists()
    events = support.daemon_log(daemon.runtime_area)
    assert [e["ev"] for e in events][-1] == "stopped" and events[-1]["reason"] == "idle"


def test_an_open_connection_or_a_running_request_keeps_the_daemon_alive(inproc):
    gate = Gate(hold=True)
    daemon = inproc(core.DaemonConfig(idle_exit_sec=0.4), hook_runner=gate)
    held, _ = daemon.connect()                                       # connected, saying nothing
    time.sleep(1.2)
    assert daemon.running
    running = in_thread(request, daemon, hook_frame())
    assert wait_for(lambda: daemon.daemon.inflight == 1)
    held.close()
    time.sleep(1.2)
    assert daemon.running                                            # the request is in flight
    gate.event.set()
    running.join(10)
    assert daemon.wait_exit(10)                                      # idle again: now it leaves


def test_idle_exit_checkpoints_counters_and_closes_engines(inproc, tmp_path):
    closed = []

    class Engine:
        def close(self):
            closed.append(1)

    daemon = inproc(core.DaemonConfig(idle_exit_sec=0.3), hook_runner=Gate(), engine_factory=lambda s: Engine())
    settings = type("S", (), {"model_dump": lambda self: {"index_dir": "x", "openrouter_api_key": None}})()
    daemon.daemon.engines.get(settings)
    daemon.daemon.ledger.add("claude", tmp_path / "idx", "PreToolUse")
    assert daemon.wait_exit(10)
    assert closed == [1]
    from self_directing_mcp.daemon import served
    assert served.read(tmp_path / "idx", "claude")[0]["served_by"] == "daemon"


# --- memory cap -------------------------------------------------------------------------------------------------------

def test_past_the_memory_cap_the_daemon_stops_accepting_finishes_in_flight_work_and_exits(inproc, monkeypatch):
    rss = {"value": 10.0}
    monkeypatch.setattr(core, "read_rss_mb", lambda: rss["value"])
    gate = Gate(hold=True)
    daemon = inproc(core.DaemonConfig(max_rss_mb=100), hook_runner=gate)
    slow = in_thread(request, daemon, hook_frame("slow"))
    assert wait_for(lambda: daemon.daemon.inflight == 1)
    rss["value"] = 500.0
    fast_gate = threading.Thread(target=lambda: (time.sleep(0.3), gate.event.set()))
    # A second request finishes while the first is still running: its completion notices the memory.
    daemon.daemon._hook_runner = lambda *a: {"systemMessage": "quick"}
    answer = request(daemon, hook_frame("quick"))[1]
    assert answer["ok"] is True                                       # the request that crossed the cap is answered
    assert wait_for(lambda: daemon.daemon.draining) and daemon.daemon.drain_reason == "rss"
    assert not daemon.runtime.exists(rundir.SOCKET_NAME)             # new clients now start a successor
    with pytest.raises(OSError):
        daemon.connect()
    assert daemon.running and slow.is_alive()                        # the older request is still being served
    fast_gate.start()
    slow.join(10)
    assert slow.box["value"][1] == {"op": "result", "ok": True, "response": {"systemMessage": "stub"},
                                    "served_by": "daemon", "ms": slow.box["value"][1]["ms"]}
    assert daemon.wait_exit(10)
    fast_gate.join()
    assert support.daemon_log(daemon.runtime_area)[-1]["reason"] == "rss"


def test_under_the_cap_nothing_drains(inproc, monkeypatch):
    monkeypatch.setattr(core, "read_rss_mb", lambda: 50.0)
    daemon = inproc(core.DaemonConfig(max_rss_mb=100), hook_runner=Gate())
    for _ in range(3):
        assert request(daemon, hook_frame())[1]["ok"] is True
    assert not daemon.daemon.draining


def test_the_real_resident_set_is_readable_and_plausible():
    assert 5 < core.read_rss_mb() < 5000


# --- configuration, lock, stale socket ------------------------------------------------------------------------------------

def test_the_configuration_comes_from_the_environment_and_rejects_nonsense():
    assert core.DaemonConfig.from_environment({}).idle_exit_sec == 900
    assert core.DaemonConfig.from_environment({}).max_rss_mb == 400
    config = core.DaemonConfig.from_environment({"SELF_DIRECT_DAEMON_IDLE_EXIT_SEC": "12.5",
                                                 "SELF_DIRECT_DAEMON_MAX_RSS_MB": "256",
                                                 "SELF_DIRECT_DAEMON_SESSION_BOUND": "3"})
    assert (config.idle_exit_sec, config.max_rss_mb, config.session_bound) == (12.5, 256, 3)
    for name in ("SELF_DIRECT_DAEMON_IDLE_EXIT_SEC", "SELF_DIRECT_DAEMON_MAX_RSS_MB", "SELF_DIRECT_DAEMON_SESSION_BOUND"):
        for bad in ("abc", "0", "-1", "nan", "inf", ""):
            with pytest.raises(ValueError):
                core.DaemonConfig.from_environment({name: bad})


def test_a_second_daemon_cannot_take_the_lock_and_a_stale_socket_is_replaced(inproc, runtime):
    first = inproc(hook_runner=Gate())
    with rundir.open_runtime_dir(runtime.environ()) as other:
        assert core.acquire_instance_lock(other, wait_sec=0.2) is None
    first.stop()
    # A crashed daemon leaves its socket file behind; the next one owns the lock and replaces it.
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(runtime.dir / rundir.SOCKET_NAME))
    stale.close()
    assert (runtime.dir / rundir.SOCKET_NAME).exists()
    second = inproc(hook_runner=Gate())
    assert request(second, hook_frame())[1]["ok"] is True


def test_a_waiting_successor_gets_the_lock_when_its_predecessor_leaves(inproc, runtime):
    first = inproc(hook_runner=Gate())
    result = {}

    def successor():
        with rundir.open_runtime_dir(runtime.environ()) as other:
            fd = core.acquire_instance_lock(other, wait_sec=10)
            result["fd"] = fd
            if fd is not None:
                os.close(fd)

    thread = threading.Thread(target=successor)
    thread.start()
    time.sleep(0.3)
    assert "fd" not in result                                         # still waiting
    first.stop()
    thread.join(10)
    assert result["fd"] is not None


# --- logging ----------------------------------------------------------------------------------------------------------

def test_the_log_carries_event_names_sizes_and_durations_but_never_request_text(inproc):
    marker = "UNIQUE-MARKER-text-4c1d9f"
    seen = []

    def runner(host, event, payload, env, argv):
        seen.append(payload)
        return {"systemMessage": marker + " in the response"}

    daemon = inproc(hook_runner=runner)
    frame = hook_frame("sess-1", payload={"session_id": "sess-1", "tool_input": {"command": marker + " rm -rf"},
                                          "transcript_path": "/srv/example/private/" + marker})
    frame["env"] = {"SELF_DIRECT_INDEX_DIR": "/private/" + marker}
    assert request(daemon, frame)[1]["ok"] is True
    daemon.stop()
    text = (daemon.runtime_area.dir / rundir.LOG_NAME).read_text()
    assert marker not in text and "/private" not in text and "example" not in text and "rm -rf" not in text
    records = [json.loads(line) for line in text.splitlines()]
    assert {r["ev"] for r in records} >= {"started", "hook", "stopped"}
    for record in records:
        for key, value in record.items():
            assert isinstance(value, (int, float, str, bool)) and (not isinstance(value, str) or len(value) <= 64)


def test_the_log_function_drops_free_text_and_rotates(tmp_path):
    area = tmp_path / "run"
    area.mkdir(mode=0o700)
    os.chmod(area, 0o700)
    with rundir.open_runtime_dir({"XDG_RUNTIME_DIR": str(tmp_path)}) as runtime:
        log = core.DaemonLog(runtime)
        log.event("x", ok="fine", free="two words", longish="a" * 200, path="/etc/passwd", number=3, flag=True,
                  nested={"a": 1}, error="ValueError")
        text = open(os.path.join(runtime.path, rundir.LOG_NAME)).read()
        record = json.loads(text)
        assert record["ok"] == "fine" and record["number"] == 3 and record["flag"] is True
        assert record["error"] == "ValueError"
        assert not {"free", "longish", "path", "nested"} & set(record)
        for _ in range(4000):
            log.event("filler", n=1, a="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        assert os.path.exists(os.path.join(runtime.path, rundir.LOG_NAME + ".1"))
        assert os.path.getsize(os.path.join(runtime.path, rundir.LOG_NAME)) <= core.LOG_ROTATE_BYTES + 200
        log.close()


# --- per-request environment and engines -------------------------------------------------------------------------------

def test_the_environment_view_isolates_requests_from_each_other_and_from_the_daemon(monkeypatch, overlay_environment):
    monkeypatch.setenv("SELF_DIRECT_DAEMON_OWN", "daemon-value")      # the daemon's own value of a managed name
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-synthetic-ambient")
    results = {}
    barrier = threading.Barrier(2)

    def request_with(name, value):
        with envview.request_environment({"SELF_DIRECT_INDEX_DIR": value, "HOME": "/h/" + name}):
            barrier.wait(5)
            os.environ["SELF_DIRECT_SET_IN_REQUEST"] = name
            time.sleep(0.05)
            results[name] = (os.environ.get("SELF_DIRECT_INDEX_DIR"), os.environ.get("SELF_DIRECT_DAEMON_OWN"),
                             os.environ.get("OPENROUTER_API_KEY"), os.environ.get("SELF_DIRECT_SET_IN_REQUEST"),
                             os.path.expanduser("~"), os.environ.get("PATH") is not None,
                             "SELF_DIRECT_INDEX_DIR" in os.environ and len(dict(os.environ)) > 1)

    threads = [threading.Thread(target=request_with, args=(n, f"/idx/{n}")) for n in ("one", "two")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert results["one"] == ("/idx/one", None, None, "one", "/h/one", True, True)
    assert results["two"] == ("/idx/two", None, None, "two", "/h/two", True, True)
    # Outside any request the daemon's own environment is what it always was.
    assert os.environ.get("SELF_DIRECT_DAEMON_OWN") == "daemon-value" and os.environ.get("SELF_DIRECT_INDEX_DIR") is None
    assert os.environ.get("SELF_DIRECT_SET_IN_REQUEST") is None


def test_one_engine_per_resolved_settings_key_with_the_real_runner(inproc, world, overlay_environment):
    daemon = inproc()
    assert daemon.daemon._ready.wait(120)
    second_index = world.root / "index-two"
    second_index.mkdir()
    (second_index / "contracts.json").write_text(json.dumps({"contracts": support.CONTRACTS}))
    env = P.forwarded_environment(world.env("on"))

    def ask(index, command):
        frame = {"op": P.OP_HOOK, "host": "claude", "event": "PreToolUse",
                 "payload": world.payload("claude", "PreToolUse", command), "env": env,
                 "argv": ["--local-only", f"--index-dir={index}"]}
        return request(daemon, frame)[1]

    first, again = ask(world.index, "rm -rf /"), ask(world.index, "rm -rf /")
    assert first["ok"] and again["ok"] and first["response"] == again["response"]
    assert first["response"]["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert len(daemon.daemon.engines) == 1
    other = ask(second_index, "rm -rf /")
    assert other["response"]["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert len(daemon.daemon.engines) == 2
    ask(second_index, "ls")
    assert len(daemon.daemon.engines) == 2                            # reused, not recreated
    assert (second_index / "meta.sqlite").exists() and (world.index / "meta.sqlite").exists()


def test_an_unusable_configuration_fails_open_exactly_as_a_standalone_hook_does(inproc, world, overlay_environment):
    daemon = inproc()
    env = P.forwarded_environment(world.env("on", SELF_DIRECT_CLAUDE_ENFORCEMENT="bogus-mode"))
    frame = {"op": P.OP_HOOK, "host": "claude", "event": "PreToolUse",
             "payload": world.payload("claude", "PreToolUse", "ls"), "env": env, "argv": []}
    _hello, answer = request(daemon, frame)
    assert answer["ok"] is True and "UNAVAILABLE" in json.dumps(answer["response"])
    assert len(daemon.daemon.engines) == 0                            # no engine was created for broken settings
