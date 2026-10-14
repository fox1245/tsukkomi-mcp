"""Framing, version comparison and the forwarded environment of the daemon protocol."""
from __future__ import annotations

import asyncio
import json
import socket
import struct
import threading

import pytest

from self_directing_mcp.daemon import protocol as P


def pair():
    left, right = socket.socketpair()
    left.settimeout(5)
    right.settimeout(5)
    return left, right


def send_in_background(sock, chunks):
    """A socket pair buffers little: senders of large or many-piece frames run beside the reader."""
    def run():
        for chunk in chunks:
            sock.sendall(chunk)
    thread = threading.Thread(target=run)
    thread.start()
    return thread


# --- framing ----------------------------------------------------------------------------------------------------

def test_a_frame_round_trips_including_non_ascii_text_and_lone_surrogates():
    left, right = pair()
    message = {"op": "hook", "text": "한국어 ☃ tab\t quote\" \ud800 end", "n": [1, 2.5, None, True]}
    P.write_frame(left, message)
    assert P.read_frame(right) == message
    left.close()
    right.close()


def test_a_frame_split_across_many_small_writes_is_reassembled():
    left, right = pair()
    wire = P.encode_frame({"op": "x", "blob": "y" * 5000})
    sender = send_in_background(left, [wire[index:index + 7] for index in range(0, len(wire), 7)])
    assert P.read_frame(right)["blob"] == "y" * 5000
    sender.join()
    left.close()
    right.close()


def test_two_frames_in_one_segment_are_read_one_at_a_time():
    left, right = pair()
    left.sendall(P.encode_frame({"op": "a"}) + P.encode_frame({"op": "b"}))
    assert [P.read_frame(right)["op"], P.read_frame(right)["op"]] == ["a", "b"]
    left.close()
    right.close()


def test_a_large_frame_below_the_limit_round_trips():
    left, right = pair()
    left.settimeout(30)
    right.settimeout(30)
    body = {"op": "hook", "payload": {"text": "z" * (3 * 1024 * 1024)}}
    sender = threading.Thread(target=P.write_frame, args=(left, body))
    sender.start()
    assert P.read_frame(right) == body
    sender.join()
    left.close()
    right.close()


@pytest.mark.parametrize("length", [0, P.MAX_FRAME_BYTES + 1, 0xFFFFFFFF, 2 ** 31])
def test_an_out_of_range_length_is_refused_before_any_body_is_read(length):
    left, right = pair()
    left.sendall(struct.pack(">I", length))                      # no body follows: nothing may wait for one
    with pytest.raises(P.ProtocolError):
        P.read_frame(right)
    left.close()
    right.close()


@pytest.mark.parametrize("body", [b"not json", b"\xff\xfe\xfa", b"[1, 2]", b"3", b'"s"', b"null", b"{", b"{\"a\": "])
def test_an_invalid_or_non_object_body_is_refused(body):
    left, right = pair()
    left.sendall(struct.pack(">I", len(body)) + body)
    with pytest.raises(P.ProtocolError) as info:
        P.read_frame(right)
    assert body.decode("latin-1") not in str(info.value)          # the message never echoes frame content
    left.close()
    right.close()


def test_a_deeply_nested_body_is_a_protocol_error_not_a_crash():
    left, right = pair()
    body = b"[" * 200_000 + b"]" * 200_000
    sender = send_in_background(left, [struct.pack(">I", len(body)) + body])
    with pytest.raises(P.ProtocolError):
        P.read_frame(right)
    sender.join()
    left.close()
    right.close()


def test_a_close_between_frames_is_a_clean_close_and_inside_a_frame_is_an_error():
    left, right = pair()
    left.close()
    with pytest.raises(P.ConnectionClosed):
        P.read_frame(right)
    right.close()
    left, right = pair()
    left.sendall(struct.pack(">I", 10) + b"{\"a\"")
    left.close()
    with pytest.raises(P.ProtocolError):
        P.read_frame(right)
    right.close()
    left, right = pair()
    left.sendall(b"\x00\x00")                                    # half a header
    left.close()
    with pytest.raises(P.ProtocolError):
        P.read_frame(right)
    right.close()


def test_encoding_refuses_a_frame_above_the_limit():
    with pytest.raises(P.ProtocolError):
        P.encode_frame({"op": "x", "blob": "a" * 2000}, max_bytes=1000)


def test_the_async_reader_applies_the_same_rules():
    async def scenario(data: bytes):
        reader = asyncio.StreamReader()
        reader.feed_data(data)
        reader.feed_eof()
        return await P.read_frame_async(reader)

    assert asyncio.run(scenario(P.encode_frame({"op": "ok"}))) == {"op": "ok"}
    for bad in (struct.pack(">I", 0), struct.pack(">I", P.MAX_FRAME_BYTES + 1), struct.pack(">I", 5) + b"abc",
                struct.pack(">I", 2) + b"[]", b"\x00\x00"):
        with pytest.raises(P.ProtocolError):
            asyncio.run(scenario(bad))
    with pytest.raises(P.ConnectionClosed):
        asyncio.run(scenario(b""))


# --- versions ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("client,daemon,expected", [
    ((1, (0, 6, 0)), (1, (0, 6, 0)), "same"),
    ((1, (0, 7, 0)), (1, (0, 6, 0)), "newer"),
    ((2, (0, 6, 0)), (1, (0, 6, 0)), "newer"),
    ((1, (0, 6, 1)), (1, (0, 6, 0)), "newer"),
    ((1, (0, 5, 9)), (1, (0, 6, 0)), "older"),
    ((0, (0, 6, 0)), (1, (0, 6, 0)), "older"),
    ((2, (0, 5, 0)), (1, (0, 6, 0)), "mixed"),                  # newer proto, older package: never replaces
    ((1, (1, 0)), (1, (0, 9, 9)), "newer"),
])
def test_version_relations(client, daemon, expected):
    assert P.compare(client, daemon) == expected


def test_versions_parse_numerically_not_textually():
    assert P.parse_version("0.10.0") > P.parse_version("0.9.0")
    assert P.parse_version("0.6.0rc1") == (0, 6, 0)
    assert P.parse_version("garbage") == (0,) and P.parse_version("") == (0,)
    assert P.versions_of({"proto": "x", "package_version": "0.6.0"}) == (0, (0, 6, 0))


# --- the forwarded environment -------------------------------------------------------------------------------------

def test_only_the_relevant_environment_travels_and_no_credential_does():
    environ = {"SELF_DIRECT_INDEX_DIR": "/idx", "self_direct_local_only": "true", "TSUKKOMI_WORKFLOW_STATE": "/s",
               "HOME": "/home/u", "CODEX_SESSIONS_DIR": "/c", "CLAUDE_CONFIG_DIR": "/cc",
               "OPENROUTER_API_KEY": "sk-synthetic-secret", "SELF_DIRECT_OPENROUTER_API_KEY": "sk-also-secret",
               "SELF_DIRECT_OPENROUTER_API_KEY_FILE": "/keys/.env", "SELF_DIRECT_SOME_TOKEN": "t",
               "SELF_DIRECT_DB_PASSWORD": "p", "PATH": "/bin", "AWS_SECRET_ACCESS_KEY": "no", "LANG": "C"}
    sent = P.forwarded_environment(environ)
    assert sent == {"SELF_DIRECT_INDEX_DIR": "/idx", "self_direct_local_only": "true",
                    "TSUKKOMI_WORKFLOW_STATE": "/s", "HOME": "/home/u", "CODEX_SESSIONS_DIR": "/c",
                    "CLAUDE_CONFIG_DIR": "/cc", "SELF_DIRECT_OPENROUTER_API_KEY_FILE": "/keys/.env"}
    assert "sk-synthetic-secret" not in json.dumps(sent) and "sk-also-secret" not in json.dumps(sent)


def test_the_daemon_ignores_whatever_a_client_adds_beyond_the_whitelist():
    env, argv = P.validate_environment(
        {"SELF_DIRECT_INDEX_DIR": "/i", "PATH": "/evil", "OPENROUTER_API_KEY": "k", "SELF_DIRECT_API_KEY": "k",
         "LD_PRELOAD": "/x"}, ["--local-only", "--event=PreToolUse", "--index-dir=/i", "--weird"])
    assert env == {"SELF_DIRECT_INDEX_DIR": "/i"}
    assert argv == ["--local-only", "--index-dir=/i"]


@pytest.mark.parametrize("env,argv", [("x", []), ({"a": 1}, []), ({}, "x"), ({}, [1]), ({"a\0": "b"}, []),
                                       ({"HOME": "x" * (P.MAX_ENV_VALUE + 1)}, [])])
def test_malformed_requests_are_refused(env, argv):
    with pytest.raises(P.ProtocolError):
        P.validate_environment(env, argv)


def test_claude_flags_do_what_the_standalone_hook_does_with_them():
    assert P.apply_claude_flags(["--local-only", "--index-dir=/abs/idx"], {}) == {
        "SELF_DIRECT_LOCAL_ONLY": "true", "SELF_DIRECT_INDEX_DIR": "/abs/idx"}
    assert P.apply_claude_flags(["--index-dir=relative"], {"X": "1"}) == {"X": "1"}     # only absolute paths count
    assert P.apply_claude_flags([], {"SELF_DIRECT_INDEX_DIR": "/e"}) == {"SELF_DIRECT_INDEX_DIR": "/e"}


def test_the_session_key_is_the_hosts_session_id_and_bounded():
    assert P.session_key("claude", {"session_id": "abc"}) == "abc"
    assert P.session_key("agy", {"conversationId": "xyz"}) == "xyz"
    assert P.session_key("claude", {"session_id": "s" * 500}) == "s" * 80
    assert P.session_key("claude", None) == P.session_key("agy", {}) == "?"


def test_the_client_side_modules_import_only_the_standard_library():
    import ast
    import pathlib
    import sys
    stdlib = set(sys.stdlib_module_names)
    package = pathlib.Path(P.__file__).parent
    for name in ("protocol.py", "rundir.py", "client.py", "relay.py"):
        path = package / name
        if not path.exists():
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                       [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else [])
            for module in modules:
                top = module.split(".")[0]
                assert top in stdlib or top == "self_directing_mcp", f"{name} imports {module}"
                if top == "self_directing_mcp":
                    assert module in ("self_directing_mcp", "self_directing_mcp.daemon") or \
                        module.startswith("self_directing_mcp.daemon.") or \
                        module in ("self_directing_mcp.claude_hook_state", "self_directing_mcp.agy_hook_state"), \
                        f"{name} imports {module}"
