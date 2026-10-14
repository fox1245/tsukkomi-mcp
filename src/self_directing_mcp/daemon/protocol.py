"""Wire protocol between the daemon and its clients. Standard library only (clients import it).

Framing: a 4-byte big-endian length followed by that many bytes of UTF-8 JSON (one object). A length of zero, a
length above ``MAX_FRAME_BYTES`` or a body that is not a JSON object is a ``ProtocolError``; the length is checked
before anything is allocated.

Operations (``op``):

* ``hello``  -> ``{proto, package_version, index_schema, parser_versions, evaluator_version, features, pid}``.
  Always the first frame on a connection; its layout is stable across protocol versions.
* ``hook``   -> ``{host, event, payload, env, argv}``; the answer carries the host's hook response unchanged.
* ``drain``  -> a NEWER client asks an older daemon to finish its in-flight requests and exit.
* ``status`` -> counters (no request or response content).
* ``mcp``    -> ``{env, cwd, mode}``; after the ``ok`` answer the connection carries newline-delimited JSON-RPC
  (the MCP stdio framing) in both directions until either side closes it.

Failures the daemon reports are ``{"ok": false, "status": "unknown", "reason": <name>}``: ``queue_full``,
``version_mismatch``, ``draining``, ``bad_request``, ``internal_error``. A hook client turns every one of them into
its host's fail-open response; none of them is a compliance verdict.

The environment a client forwards is a whitelist (``forwarded_environment``): ``SELF_DIRECT_*``, ``TSUKKOMI_*``,
``HOME`` and the two session-root variables the settings read. Credentials are never forwarded: names that look
like keys, tokens, secrets or passwords are dropped unless they name a FILE or PATH (the key-file setting), and
``OPENROUTER_API_KEY`` is never on the list. The daemon reads a key file itself, exactly as a standalone process does.
"""
from __future__ import annotations

import json
import os
import re
import struct

PROTO = 1
MAX_FRAME_BYTES = 16 * 1024 * 1024
HEADER = struct.Struct(">I")

OP_HELLO, OP_HOOK, OP_DRAIN, OP_STATUS, OP_MCP = "hello", "hook", "drain", "status", "mcp"
HOSTS = ("claude", "agy")

REASON_QUEUE_FULL = "queue_full"
REASON_VERSION_MISMATCH = "version_mismatch"
REASON_DRAINING = "draining"
REASON_BAD_REQUEST = "bad_request"
REASON_INTERNAL = "internal_error"

FORWARD_PREFIXES = ("SELF_DIRECT_", "TSUKKOMI_")
FORWARD_NAMES = frozenset({"HOME", "USERPROFILE", "CODEX_SESSIONS_DIR", "CLAUDE_CONFIG_DIR"})
NEVER_FORWARD = frozenset({"OPENROUTER_API_KEY"})
SECRET_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")
PATH_SUFFIXES = ("_FILE", "_PATH", "_DIR")
MAX_ENV_ENTRIES = 512
MAX_ENV_VALUE = 32 * 1024
MAX_ARGV = 64
MAX_ARG = 4096


class ProtocolError(ValueError):
    """A frame that violates the framing rules. The message never contains frame content."""


class ConnectionClosed(EOFError):
    """The peer closed the connection at a frame boundary (or before sending one)."""


# --- framing ------------------------------------------------------------------------------------------------------

def encode_frame(obj: dict, *, max_bytes: int = MAX_FRAME_BYTES) -> bytes:
    # ASCII escapes round-trip every str, including the lone surrogates a hook payload can legally carry.
    body = json.dumps(obj, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    if not body or len(body) > max_bytes:
        raise ProtocolError("frame too large")
    return HEADER.pack(len(body)) + body


def check_length(length: int, max_bytes: int = MAX_FRAME_BYTES) -> int:
    if length <= 0 or length > max_bytes:
        raise ProtocolError("frame length out of range")
    return length


def decode_body(body: bytes) -> dict:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ProtocolError("frame body is not valid JSON") from None
    if not isinstance(value, dict):
        raise ProtocolError("frame body is not a JSON object")
    return value


def _read_exact(sock, count: int, *, what: str) -> bytes:
    chunks, remaining = [], count
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            if remaining == count and what == "header":
                raise ConnectionClosed("connection closed")
            raise ProtocolError("connection closed inside a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(sock, *, max_bytes: int = MAX_FRAME_BYTES) -> dict:
    """Blocking read of one frame from a socket (honours the socket's timeout)."""
    (length,) = HEADER.unpack(_read_exact(sock, HEADER.size, what="header"))
    return decode_body(_read_exact(sock, check_length(length, max_bytes), what="body"))


def write_frame(sock, obj: dict, *, max_bytes: int = MAX_FRAME_BYTES) -> None:
    sock.sendall(encode_frame(obj, max_bytes=max_bytes))


async def read_frame_async(reader, *, max_bytes: int = MAX_FRAME_BYTES) -> dict:
    """The same for an ``asyncio.StreamReader`` (duck-typed: this module never imports asyncio)."""
    try:
        header = await reader.readexactly(HEADER.size)
    except EOFError as exc:
        if getattr(exc, "partial", b"x"):
            raise ProtocolError("connection closed inside a frame") from None
        raise ConnectionClosed("connection closed") from None
    (length,) = HEADER.unpack(header)
    try:
        return decode_body(await reader.readexactly(check_length(length, max_bytes)))
    except EOFError:
        raise ProtocolError("connection closed inside a frame") from None


async def write_frame_async(writer, obj: dict, *, max_bytes: int = MAX_FRAME_BYTES) -> None:
    writer.write(encode_frame(obj, max_bytes=max_bytes))
    await writer.drain()


# --- versions -----------------------------------------------------------------------------------------------------

def parse_version(text) -> tuple[int, ...]:
    """``"0.6.0"`` -> ``(0, 6, 0)``; a pre-release or build suffix is ignored. Unparseable text sorts lowest."""
    parts = []
    for piece in str(text).split("."):
        match = re.match(r"\d+", piece)
        if match is None:
            break
        parts.append(int(match.group()))
    return tuple(parts) or (0,)


def compare(client: tuple[int, tuple[int, ...]], daemon: tuple[int, tuple[int, ...]]) -> str:
    """``same``, ``newer`` (the client is newer in proto or package version and older in neither), ``older`` or
    ``mixed`` (one is newer and the other older). Only ``newer`` may replace a running daemon."""
    ahead = client[0] > daemon[0] or client[1] > daemon[1]
    behind = client[0] < daemon[0] or client[1] < daemon[1]
    if ahead and behind:
        return "mixed"
    return "newer" if ahead else "older" if behind else "same"


def hello_request(role: str, package_version: str, *, proto: int = PROTO) -> dict:
    return {"op": OP_HELLO, "proto": proto, "package_version": package_version, "role": role}


def versions_of(frame: dict) -> tuple[int, tuple[int, ...]]:
    proto = frame.get("proto")
    return (proto if type(proto) is int else 0), parse_version(frame.get("package_version", "0"))


# --- the forwarded environment ------------------------------------------------------------------------------------

def _looks_secret(upper: str) -> bool:
    return any(marker in upper for marker in SECRET_MARKERS) and not upper.endswith(PATH_SUFFIXES)


def is_managed(name: str) -> bool:
    """Names whose value, inside the daemon, comes from the request and never from the daemon's own environment."""
    upper = name.upper()
    return upper.startswith(FORWARD_PREFIXES) or upper in FORWARD_NAMES or upper in NEVER_FORWARD


def forwarded_environment(environ) -> dict[str, str]:
    """The client's relevant environment, credentials removed."""
    sent: dict[str, str] = {}
    for name, value in environ.items():
        upper = name.upper()
        if upper in NEVER_FORWARD or not isinstance(value, str) or len(value) > MAX_ENV_VALUE:
            continue
        if not (upper.startswith(FORWARD_PREFIXES) or upper in FORWARD_NAMES) or _looks_secret(upper):
            continue
        sent[name] = value
    return sent


def forwarded_argv(argv) -> list[str]:
    """Only the hook flags that change settings travel; the event name travels separately."""
    return [arg for arg in argv if isinstance(arg, str) and len(arg) <= MAX_ARG
            and (arg == "--local-only" or arg.startswith("--index-dir="))][:MAX_ARGV]


def validate_environment(env, argv) -> tuple[dict[str, str], list[str]]:
    """Daemon side: a request's environment and flags, re-checked. Raises ``ProtocolError``."""
    if not isinstance(env, dict) or len(env) > MAX_ENV_ENTRIES:
        raise ProtocolError("env must be an object")
    clean: dict[str, str] = {}
    for name, value in env.items():
        if not isinstance(name, str) or not isinstance(value, str) or len(value) > MAX_ENV_VALUE or "\0" in name + value:
            raise ProtocolError("env entries must be strings")
        if not is_managed(name) or name.upper() in NEVER_FORWARD or _looks_secret(name.upper()):
            continue                         # whatever else a client sends is ignored, never trusted
        clean[name] = value
    if not isinstance(argv, list) or len(argv) > MAX_ARGV or any(
            not isinstance(a, str) or len(a) > MAX_ARG for a in argv):
        raise ProtocolError("argv must be a list of strings")
    return clean, [a for a in argv if a == "--local-only" or a.startswith("--index-dir=")]


def apply_claude_flags(argv, env: dict[str, str]) -> dict[str, str]:
    """What ``claude_hooks.main`` does with its flags, applied to a request's environment (AGY ignores them)."""
    env = dict(env)
    if "--local-only" in argv:
        env["SELF_DIRECT_LOCAL_ONLY"] = "true"
    for arg in argv:
        if arg.startswith("--index-dir=") and os.path.isabs(arg.split("=", 1)[1]):
            env["SELF_DIRECT_INDEX_DIR"] = arg.split("=", 1)[1]
    return env


def session_key(host: str, payload) -> str:
    """The key requests are bounded by: the host's session id (or ``?``); never anything longer than 80 characters."""
    name = "session_id" if host == "claude" else "conversationId"
    value = payload.get(name) if isinstance(payload, dict) else None
    return value[:80] if isinstance(value, str) and value else "?"
