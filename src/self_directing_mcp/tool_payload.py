"""The text of a tool call that contracts match: its payload, never the model's own commentary about it.

Some hosts let the model add a free-text field to a tool call (Claude Code's ``description`` on ``Bash``, AGY's
``toolAction`` / ``toolSummary``). That field is prose the model wrote about the call; the call itself is the
other fields. A prohibition such as ``sqlite3`` must not fire on a careful command only because its prose says
"without connecting to it with sqlite3", so the prose is kept out of the text a contract is matched against.

One function builds that text, and every writer uses it: the parsers (indexed history), the engine (the proposed
action a hook asks about) and the hook-side pre-checks. The two sides cannot disagree about what a call contains.

Only the fields named below are prose, and only where the prose nature is confirmed from the host's schema or
from a recorded fixture. An unclassified field stays payload: it is matched exactly as before. Fields that are
suspected but unconfirmed are listed in ``UNCONFIRMED`` and are deliberately not excluded.

Grok CLI calls are rendered with Claude Code's tool names and argument keys (``grok_tools``): a contract written
for ``[tool_call:Bash] {"command": ...}`` matches Grok's ``run_terminal_command`` too. That conversion happens
here, for provider ``grok`` only, so every writer of the text applies it and the native name stays in the metadata.

This module has no heavy imports; hook processes load it.
"""
from __future__ import annotations

import json
from typing import Any

# Longest prose kept per field (in meta, for a later ``scope: prose``); the call text never contains it.
PROSE_CAP = 4000

# provider -> (field names whose string value is model-written prose, predicate on the tool name).
# Claude Code: the built-in tools that take a ``description`` (Bash, Monitor, Task/Agent, ...) document it as a
# short model-written label for the call. An MCP tool (``mcp__server__tool``) defines its own schema, so a
# ``description`` there may be real payload (an issue or event body): it is not touched.
# AGY: ``toolAction`` and ``toolSummary`` appear beside the payload fields of ``run_command`` in the recorded
# fixtures (``"toolAction": "Running tests"``, ``"toolSummary": "Execute tests"``): UI labels the model writes.
PROSE_FIELDS: dict[str, tuple[frozenset[str], Any]] = {
    "claude": (frozenset({"description"}), lambda name: not str(name).startswith("mcp__")),
    "agy": (frozenset({"toolAction", "toolSummary"}), lambda name: True),
}

# Fields that look like prose but are NOT confirmed from this repository's fixtures or code, so they stay payload.
UNCONFIRMED = {
    "agy": ("Description", "Instruction", "Explanation"),
    "codex": ("justification",),
    "omp": ("_i", "intent"),
}


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def split_prose(provider: str | None, tool_name: Any, arguments: Any) -> tuple[Any, dict[str, str]]:
    """``(payload, prose)``: the arguments without this host's confirmed prose fields, and those fields.

    Only a dictionary has fields, and only a string value counts as prose; any other shape is payload, untouched.
    """
    known = PROSE_FIELDS.get(provider or "")
    if known is None or not isinstance(arguments, dict) or not known[1](tool_name):
        return arguments, {}
    prose = {key: value for key, value in arguments.items() if key in known[0] and isinstance(value, str)}
    if not prose:
        return arguments, {}
    return {key: value for key, value in arguments.items() if key not in prose}, prose


def tool_call_parts(provider: str | None, tool_name: Any, arguments: Any, *, label: str = "tool_call") -> tuple[str, dict]:
    """``(text, meta)`` of a tool call: ``[<label>:<name>] <payload>`` and ``{"tool_prose": {...}}`` when it had prose.

    For Grok the name and keys are the Claude-canonical ones and ``meta`` also names the native call
    (``native_tool_name``, ``native_arguments``, ``grok_tool_map``)."""
    if provider == "grok":
        from self_directing_mcp import grok_tools
        call = grok_tools.canonical_call(tool_name, arguments)
        meta = grok_tools.native_meta(call, arguments)
        if call.prose:
            meta["tool_prose"] = {key: value[:PROSE_CAP] for key, value in call.prose.items()}
        return f"[{label}:{call.name}] {as_text(call.arguments)}", meta
    payload, prose = split_prose(provider, tool_name, arguments)
    meta = {"tool_prose": {key: value[:PROSE_CAP] for key, value in prose.items()}} if prose else {}
    return f"[{label}:{tool_name}] {as_text(payload)}", meta


def tool_call_text(provider: str | None, tool_name: Any, arguments: Any, *, label: str = "tool_call") -> str:
    return tool_call_parts(provider, tool_name, arguments, label=label)[0]
