"""Code identity: a daemon serves a client only when both run the same code, not merely the same version string.

The package version names a release, not the files: a second installation of the same version (an older copy in another
virtual environment, a checkout on another branch) started a daemon that every hook of the first then reached, and the
hooks failed. The hello exchange now carries a digest of the package's source files. Equal protocol and version with a
different or absent identity is never ``same``: the daemon does not judge, does not drain, and the client judges with its own
code (a Claude Code hook) or fails open visibly (AGY).

Synthetic data only; daemons live in temporary runtime directories (see ``conftest.py``).
"""
from __future__ import annotations

import json
import socket
import threading

import pytest

from self_directing_mcp import __version__
from self_directing_mcp.daemon import client, identity, protocol as P, relay
import daemon_support as support
from test_daemon_core import Gate, hook_frame, request
from test_daemon_hooks import Hold, claude_payload

OTHER = "f" * 64


def package(tmp_path, files):
    """A package directory holding ``files`` (relative POSIX path -> bytes)."""
    root = tmp_path / "pkg"
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return str(root)


# --- the identity ---------------------------------------------------------------------------------------------------

def test_the_identity_is_a_digest_of_the_source_files_and_follows_every_byte(tmp_path):
    base = {"a.py": b"x = 1\n", "sub/b.py": b"y = 2\n"}
    first = identity.digest_of(package(tmp_path, base))
    assert isinstance(first, str) and len(first) == 64
    assert identity.digest_of(package(tmp_path, base)) == first                        # same files, same identity
    for changed in ({**base, "a.py": b"x = 2\n"},                                      # a byte of content
                    {"a2.py": base["a.py"], "sub/b.py": base["sub/b.py"]},             # a file renamed
                    {**base, "c.py": b""},                                              # a file added
                    {"a.py": base["a.py"]}):                                            # a file removed
        assert identity.digest_of(package(tmp_path / str(len(str(changed))), changed)) != first


def test_the_framing_cannot_be_confused_by_moving_bytes_between_a_name_and_a_content(tmp_path):
    one = identity.digest_of(package(tmp_path / "one", {"a.py": b"bc", "ab.py": b"c"}))
    two = identity.digest_of(package(tmp_path / "two", {"a.py": b"b", "abc.py": b"c"}))
    three = identity.digest_of(package(tmp_path / "three", {"a.pyb": b"", "c.py": b"c"}))
    assert len({one, two, three}) == 3


def test_only_python_source_counts_and_the_bytecode_cache_does_not(tmp_path):
    base = {"a.py": b"x = 1\n"}
    first = identity.digest_of(package(tmp_path / "base", base))
    with_noise = {**base, "notes.txt": b"hello", "data.json": b"{}", "__pycache__/a.cpython-312.pyc": b"\0\1",
                  "sub/__pycache__/b.pyc": b"zz"}
    assert identity.digest_of(package(tmp_path / "noise", with_noise)) == first


def test_the_order_of_the_files_does_not_depend_on_the_directory_listing(tmp_path):
    names = ["z.py", "a/b.py", "a.py", "B.py", "é.py"]
    first = identity.digest_of(package(tmp_path / "one", {name: name.encode() for name in names}))
    second = identity.digest_of(package(tmp_path / "two", {name: name.encode() for name in reversed(names)}))
    assert first == second


def test_a_package_that_cannot_be_read_has_no_identity_and_no_identity_equals_nothing(tmp_path):
    assert identity.digest_of(str(tmp_path / "missing")) is None
    assert identity.digest_of(package(tmp_path, {"notes.txt": b"no source here"})) is None
    assert identity.same_code(None, None) is False
    assert identity.same_code("a", None) is False and identity.same_code(None, "a") is False
    assert identity.same_code("a", "b") is False and identity.same_code("a", "a") is True


def test_the_process_identity_is_the_digest_of_its_own_package_and_is_computed_once(monkeypatch):
    own = identity.digest_of(identity.PACKAGE_ROOT)
    assert own is not None and identity.code_identity() == own
    monkeypatch.setattr(identity, "digest_of", lambda root: pytest.fail("computed again"))
    assert identity.code_identity() == own


def test_the_files_the_identity_covers_are_those_of_the_package_the_client_imports():
    import self_directing_mcp
    import os
    assert identity.PACKAGE_ROOT == os.path.dirname(os.path.abspath(self_directing_mcp.__file__))


# --- the protocol ---------------------------------------------------------------------------------------------------

def test_the_hello_request_carries_the_identity_when_there_is_one():
    assert P.hello_request("hook", "1.2.3", code_identity="abc")["code_identity"] == "abc"
    assert "code_identity" not in P.hello_request("hook", "1.2.3")
    assert "code_identity" not in P.hello_request("hook", "1.2.3", code_identity=None)


@pytest.mark.parametrize("value,expected", [("abc", "abc"), ("", None), (None, None), (7, None), (["a"], None)])
def test_an_identity_in_a_frame_counts_only_as_a_non_empty_string(value, expected):
    assert P.identity_of({"code_identity": value}) == expected
    assert P.identity_of({}) is None


@pytest.mark.parametrize("relation,own,other,expected", [
    ("same", "a", "a", "same"),
    ("same", "a", "b", "code"),
    ("same", "a", None, "code"),                  # a peer from before identities existed
    ("same", None, "a", "code"),
    ("same", None, None, "code"),                 # nobody can name its code: never the same
    ("newer", "a", "b", "newer"),                 # a different version decides first, as before
    ("older", "a", "a", "older"),
    ("mixed", "a", None, "mixed"),
])
def test_equal_version_with_other_or_absent_code_is_not_same(relation, own, other, expected):
    assert P.relation_with_code(relation, own, other) == expected


# --- the daemon ------------------------------------------------------------------------------------------------------

def test_the_hello_answer_names_the_daemons_code(inproc):
    default = inproc()
    _sock, hello = default.connect()
    _sock.close()
    assert hello["code_identity"] == identity.code_identity()


def test_a_client_of_other_code_is_not_judged_and_cannot_drain_the_daemon(inproc):
    gate = Gate()
    daemon = inproc(hook_runner=gate, code_identity=OTHER)
    hello, answer = request(daemon, hook_frame(), code_identity=None)               # a client that sends no identity
    assert hello["relation"] == "code"
    assert answer == {"op": "result", "ok": False, "status": "unknown", "reason": "code_mismatch"}
    _hello, drain = request(daemon, {"op": "drain"}, code_identity=None)
    assert drain["ok"] is False and drain["reason"] == "version_mismatch"
    assert daemon.running and not daemon.daemon.draining and not gate.calls


def test_a_client_with_other_code_gets_the_same_answer_for_a_session(inproc):
    daemon = inproc(hook_runner=Gate(), code_identity=OTHER)
    sock, hello = daemon.connect(code_identity="1" * 64)
    try:
        assert hello["relation"] == "code"
        P.write_frame(sock, {"op": P.OP_MCP, "env": {}, "cwd": "/", "mode": "plain"})
        assert P.read_frame(sock)["reason"] == "code_mismatch"
    finally:
        sock.close()


def test_a_client_of_the_same_code_is_served(inproc):
    gate = Gate()
    daemon = inproc(hook_runner=gate, code_identity=OTHER)
    sock, hello = daemon.connect(code_identity=OTHER)
    try:
        assert hello["relation"] == "same"
        P.write_frame(sock, hook_frame())
        assert P.read_frame(sock)["ok"] is True
    finally:
        sock.close()
    assert len(gate.calls) == 1


def test_a_daemon_that_cannot_name_its_code_serves_nobody(inproc):
    gate = Gate()
    daemon = inproc(hook_runner=gate, code_identity=None)
    sock, hello = daemon.connect(code_identity=OTHER)
    try:
        assert hello["relation"] == "code" and hello.get("code_identity") is None
        P.write_frame(sock, hook_frame())
        assert P.read_frame(sock)["reason"] == "code_mismatch"
    finally:
        sock.close()
    assert not gate.calls


def test_a_newer_client_still_drains_the_daemon_whatever_its_code(inproc):
    daemon = inproc(hook_runner=Gate(), code_identity=OTHER)
    sock, hello = daemon.connect(version="99.0.0", code_identity="1" * 64)
    try:
        assert hello["relation"] == "newer"
    finally:
        sock.close()
    _hello, drain = request(daemon, {"op": "drain"}, version="99.0.0", code_identity="1" * 64)
    assert drain["ok"] is True and daemon.wait_exit(10)


def test_the_log_names_a_mismatch_by_its_class_once_and_the_counter_counts_every_one(inproc, runtime):
    daemon = inproc(hook_runner=Gate(), code_identity=OTHER)
    for _ in range(3):
        request(daemon, hook_frame(), code_identity=None)                               # absent
    sock, _hello = daemon.connect(code_identity="1" * 64)                           # different
    sock.close()
    sock, _hello = daemon.connect(code_identity="2" * 64)                           # different, said once already
    sock.close()
    records = [r for r in support.daemon_log(runtime) if r["ev"] == "code_mismatch"]
    assert sorted(r["client"] for r in records) == ["absent", "different"]
    text = json.dumps(records)
    assert OTHER not in text and "1" * 64 not in text and "2" * 64 not in text      # no identity value, no path, no payload
    assert all(set(r) <= {"t", "ev", "client"} for r in records)
    assert support.daemon_status(runtime)["requests"]["code_mismatch"] == 3         # the hook requests it refused


# --- a hook client ---------------------------------------------------------------------------------------------------

def test_a_client_of_other_code_does_not_use_the_daemon_and_does_not_ask_it_to_drain(world, runtime, inproc, monkeypatch):
    hold = Hold(hold=False)
    daemon = inproc(hook_runner=hold, code_identity=OTHER)
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(world.new_index("other")))
    payload = claude_payload(world, "ls")
    response, reason = client._serve("claude", "PreToolUse", payload, [], dict(__import__("os").environ), None)
    assert response is None and reason == client.R_CODE_MISMATCH == "code_mismatch"
    for event in ("PreToolUse", "PostToolUse", "Stop", "UserPromptSubmit"):          # the client judges them all itself
        assert client.serve_hook("claude", event, payload, [], local_fallback=True) == (None, False)
    assert daemon.running and not daemon.daemon.draining and not hold.calls          # nothing judged, nobody asked to leave


def test_without_a_local_path_the_call_fails_open_and_names_the_mismatch(world, runtime, inproc, monkeypatch):
    hold = Hold(hold=False)
    daemon = inproc(hook_runner=hold, code_identity=OTHER)
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(world.new_index("agy")))
    body = world.payload("agy", "PreToolUse", "ls")
    response, served = client.serve_hook("agy", "PreToolUse", body, [])
    assert served is False and "code_mismatch" in json.dumps(response)
    assert daemon.running and not daemon.daemon.draining and not hold.calls


def test_a_client_of_the_same_code_is_served_by_the_daemon(world, runtime, inproc, monkeypatch):
    hold = Hold(hold=False)
    inproc(hook_runner=hold, code_identity=OTHER)
    monkeypatch.setattr(client, "own_identity", lambda: OTHER)
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(world.new_index("same")))
    response, served = client.serve_hook("claude", "PreToolUse", claude_payload(world, "ls"), [], local_fallback=True)
    assert served is True and response == {"systemMessage": "stub"} and hold.calls == ["PreToolUse"]


def test_a_client_that_cannot_name_its_code_is_never_served(world, runtime, inproc, monkeypatch):
    hold = Hold(hold=False)
    inproc(hook_runner=hold, code_identity=OTHER)
    monkeypatch.setattr(client, "own_identity", lambda: None)
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(world.new_index("noid")))
    assert client.serve_hook("claude", "Stop", claude_payload(world), [], local_fallback=True) == (None, False)
    assert not hold.calls


class OldDaemon:
    """A daemon from before identities: it answers ``same`` and judges the request, as the daemon of the incident did."""

    def __init__(self, runtime):
        from self_directing_mcp.daemon import rundir
        self.area = rundir.open_runtime_dir(runtime.environ())
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.area.address())
        self.server.listen(4)
        self.judged = []
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
                self.judged.append(P.read_frame(conn).get("event"))
                P.write_frame(conn, {"op": "result", "ok": True, "response": {"systemMessage": "from the old daemon"}})
            except Exception:
                pass
            finally:
                conn.close()

    def close(self):
        from self_directing_mcp.daemon import rundir
        self.server.close()
        self.area.unlink(rundir.SOCKET_NAME)
        self.area.close()


def test_the_answer_of_a_daemon_that_sent_no_identity_is_not_used(world, runtime, monkeypatch):
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(world.new_index("old")))
    old = OldDaemon(runtime)
    try:
        payload = claude_payload(world, "ls")
        assert client.serve_hook("claude", "Stop", payload, [], local_fallback=True) == (None, False)
        response, served = client.serve_hook("agy", "PreToolUse", world.payload("agy", "PreToolUse", "ls"), [])
        assert served is False and "from the old daemon" not in json.dumps(response)
        assert "code_mismatch" in json.dumps(response)
    finally:
        old.close()


# --- a session -------------------------------------------------------------------------------------------------------

def handshake_against(hello_fields):
    """``relay._handshake`` against a peer that answers the hello with ``hello_fields`` (and nothing else)."""
    mine, theirs = socket.socketpair()
    mine.settimeout(5)
    theirs.settimeout(5)

    def peer():
        try:
            P.read_frame(theirs)
            P.write_frame(theirs, {"op": "hello", "ok": True, "proto": P.PROTO, "package_version": __version__,
                                   "draining": False, **hello_fields})
        except Exception:
            pass
    thread = threading.Thread(target=peer, daemon=True)
    thread.start()
    try:
        return relay._handshake(mine, {"op": P.OP_MCP, "env": {}, "cwd": "/", "mode": "plain"})
    finally:
        thread.join(5)
        theirs.close()


@pytest.mark.parametrize("fields", [{"relation": "code", "code_identity": OTHER},          # a daemon that noticed
                                    {"relation": "same", "code_identity": OTHER},          # one that says same, other code
                                    {"relation": "same"}])                                  # one from before identities
def test_a_session_on_other_code_is_refused_by_name_and_asks_nobody_to_leave(fields, monkeypatch):
    monkeypatch.setattr(client, "own_identity", lambda: "1" * 64)
    with pytest.raises(relay.RelayError) as excinfo:
        handshake_against(fields)
    message = excinfo.value.message
    assert "code_mismatch" in message and "--stop" in message and "newer than this client" not in message


# --- the real hook script --------------------------------------------------------------------------------------------

def test_a_real_claude_hook_judges_for_itself_when_the_daemon_runs_other_code(world, runtime, inproc):
    hold = Hold(hold=False)
    inproc(hook_runner=hold, code_identity=OTHER)
    index = world.new_index("script")
    on = world.hook("claude", "PreToolUse", "rm -rf /", flag="on", index=index)
    assert on.code == 0 and not hold.calls
    assert on.json.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"       # judged: the contract fired
    assert "code_mismatch" not in on.stdout and "UNAVAILABLE" not in on.stdout
    for event in ("Stop", "UserPromptSubmit"):
        run = world.hook("claude", event, flag="on", index=index)
        assert run.code == 0 and not hold.calls and "code_mismatch" not in run.stdout and "UNAVAILABLE" not in run.stdout, run
    agy = world.hook("agy", "PreToolUse", "ls", flag="on", index=index)
    assert "code_mismatch" in agy.stdout and not hold.calls                                   # AGY fails open, visibly
