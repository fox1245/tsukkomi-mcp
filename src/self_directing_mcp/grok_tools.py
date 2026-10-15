"""Grok CLI tool names and argument keys, rendered with the Claude-canonical vocabulary. Standard library only.

Contracts are written against the text ``[tool_call:<tool_name>] <json args>``, and the contracts users already
hold name Claude Code's tools (``Bash``, ``Read``, ``Edit``) and keys (``command``, ``file_path``). Grok emits its
own names and keys (``run_terminal_command``, ``read_file`` with ``target_file``). Nothing is matched against the
native spelling: this module is the one place where a Grok call becomes the call a contract sees, and
``tool_payload.tool_call_parts`` applies it for provider ``grok`` in the parser (indexed history), in the engine
(the proposed action a hook asks about) and in the hook-side pre-checks, so the two sides cannot disagree.

The native name and arguments are kept beside the canonical text (``meta.native_tool_name`` /
``meta.native_arguments``) and named in the hook's deny reason, so a user reading a finding sees what Grok ran.

Rules, all of them pinned by ``tests/test_grok_tools.py``:

* A row renames the tool and, per row, some argument keys. A key is renamed only when the canonical key is not
  already present in the same call; nothing is ever dropped or overwritten silently.
* ``run_terminal_command.description`` is model-written prose about the call (the recorded fixtures show
  ``"description": "Print probe-ok marker"``): it is kept out of the call text like Claude's ``Bash`` description
  (``tool_payload``), and recorded in ``meta.tool_prose``.
* An MCP tool is ``<server>__<tool>`` in Grok and ``mcp__<server>__<tool>`` in Claude Code; the canonical form is the
  Claude one. A name that already starts with ``mcp__`` is canonical and passes through, which also makes every
  conversion idempotent (a hook may pass a canonical call to an engine that converts again).
* Anything else passes through with its native name and arguments.

Bump ``TOOL_MAP_VERSION`` with every change of a row, and ``grok.PARSER_VERSION`` with it: the indexed call text
changes, so an indexed Grok session must be reparsed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

TOOL_MAP_VERSION = 1

NATIVE_ARGUMENT_STRING_CAP = 1000        # longest string kept per value of ``meta.native_arguments``
NATIVE_ARGUMENT_CAP = 8000               # larger serialized arguments keep only their key names


@dataclass(frozen=True)
class ToolRow:
    native: str
    canonical: str
    keys: tuple[tuple[str, str], ...] = ()          # (native argument key, canonical argument key)
    prose: frozenset[str] = frozenset()             # argument keys that are model-written prose about the call
    source: str = ""                                # where the row was verified


# Order is the order of the README table.
TOOL_ROWS: tuple[ToolRow, ...] = (
    ToolRow("run_terminal_command", "Bash", (("is_background", "run_in_background"),), frozenset({"description"}),
            "recorded hook payloads and updates.jsonl: command, description; is_background in tool_call_update"),
    ToolRow("read_file", "Read", (("target_file", "file_path"), ("path", "file_path")), frozenset(),
            "recorded hook payloads: target_file, limit"),
    ToolRow("search_replace", "Edit", (), frozenset(),
            "recorded hook payloads: file_path, old_string, new_string (the same names Claude Code's Edit uses)"),
    ToolRow("write", "Write", (("target_file", "file_path"), ("path", "file_path"), ("contents", "content")), frozenset(),
            "named by Grok's hook documentation; its arguments are NOT in the recorded fixtures (verify with a live run)"),
    ToolRow("grep", "Grep", (), frozenset(),
            "recorded hook payloads: pattern, path (Claude Code's Grep uses the same names)"),
    ToolRow("list_dir", "LS", (("target_directory", "path"),), frozenset(),
            "recorded hook payloads: target_directory; LS is Claude Code's directory listing (its Glob needs a pattern)"),
    ToolRow("web_search", "WebSearch", (), frozenset(), "named by Grok's hook documentation; arguments pass through"),
    ToolRow("web_fetch", "WebFetch", (), frozenset(), "named by Grok's hook documentation; arguments pass through"),
    ToolRow("spawn_subagent", "Task", (), frozenset(), "named by Grok's hook documentation; arguments pass through"),
)
_BY_NATIVE = {row.native: row for row in TOOL_ROWS}
_CANONICAL = frozenset(row.canonical for row in TOOL_ROWS)

MCP_PREFIX = "mcp__"
_MCP_NATIVE = re.compile(r"[^\W_][\w.-]*__[\w.-]+\Z")      # <server>__<tool>; neither part empty


@dataclass(frozen=True)
class CanonicalCall:
    name: Any                       # the Claude-canonical name, or the native name when nothing maps
    arguments: Any                  # the canonical arguments (anything but a dictionary passes through untouched)
    native_name: Any
    prose: dict[str, str]           # model-written prose fields, kept out of ``arguments``
    mapped: bool                    # the call was renamed or its keys were; False for a pass-through


def _mcp_name(name: str) -> str | None:
    if name.startswith(MCP_PREFIX) or name in _BY_NATIVE or not _MCP_NATIVE.fullmatch(name):
        return None
    return MCP_PREFIX + name


def canonical_call(tool_name: Any, arguments: Any) -> CanonicalCall:
    """The call a contract sees. Pure; never raises."""
    if not isinstance(tool_name, str):
        return CanonicalCall(tool_name, arguments, tool_name, {}, False)
    row = _BY_NATIVE.get(tool_name)
    if row is None:
        mcp = _mcp_name(tool_name)
        return CanonicalCall(mcp or tool_name, arguments, tool_name, {}, mcp is not None)
    if not isinstance(arguments, dict):
        return CanonicalCall(row.canonical, arguments, tool_name, {}, True)
    renames = dict(row.keys)
    prose = {key: value for key, value in arguments.items() if key in row.prose and isinstance(value, str)}
    canonical: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in prose:
            continue
        target = renames.get(key)
        canonical[target if target is not None and target not in arguments and target not in canonical else key] = value
    return CanonicalCall(row.canonical, canonical, tool_name, prose, True)


def native_meta(call: CanonicalCall, arguments: Any) -> dict[str, Any]:
    """The metadata that keeps what Grok actually sent: name, arguments (capped), and the map version."""
    if not call.mapped:
        return {}
    meta: dict[str, Any] = {"native_tool_name": call.native_name, "grok_tool_map": TOOL_MAP_VERSION}
    meta["native_arguments"] = _capped(arguments)
    return meta


def _capped(arguments: Any) -> Any:
    if isinstance(arguments, dict):
        shown = {str(key): (value[:NATIVE_ARGUMENT_STRING_CAP] + "..." if isinstance(value, str)
                            and len(value) > NATIVE_ARGUMENT_STRING_CAP else value) for key, value in arguments.items()}
    else:
        shown = arguments
    try:
        size = len(json.dumps(shown, ensure_ascii=False, sort_keys=True, default=str))
    except (TypeError, ValueError):
        size = NATIVE_ARGUMENT_CAP + 1
    if size > NATIVE_ARGUMENT_CAP:
        keys = sorted(str(key) for key in arguments) if isinstance(arguments, dict) else []
        return {"truncated": True, "keys": keys[:50]}
    return shown


def describe(native_name: Any, canonical_name: Any) -> str:
    """Short phrase for a deny reason: which Grok tool a finding is about."""
    return f"Grok tool {native_name}" if native_name == canonical_name else f"Grok tool {native_name} (matched as {canonical_name})"
