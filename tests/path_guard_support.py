"""Tool-call texts as each host's own parser renders them, shared by the ``path_guard`` tests.

A neutral ``Call`` says what the model did (read a file, ran a shell command, wrote a file, patched a file, searched a
path, mentioned a path in a field that is not a path); ``HOSTS`` map it to the host's tool name and argument keys, taken from
the repository's own fixtures and tests (see ``contract_paths.TOOL_ENTRIES`` for the sources). ``render`` writes ONE
synthetic transcript per host and returns the text of every ``tool_call`` event the real parser produced, so a lowered
regex is always tried against the text a parser (and therefore a hook) really builds. Synthetic data only.
"""
from __future__ import annotations

from collections import namedtuple
import json
from pathlib import Path

from self_directing_mcp import agy, grok, omp
from self_directing_mcp.codex.parse import parse_session_chunks as codex_parse
from grok_support import call as grok_call, session_file as grok_session_file, user as grok_user, write as grok_write
from test_agy_parser_v6 import make as make_agy, planner, user as agy_user
from test_claude_provider import assistant, human, main_file, parse as parse_claude, use
from test_claude_provider import write as write_claude

Call = namedtuple("Call", "kind value extra", defaults=(None,))

READ, SHELL, WRITE, EDIT, SEARCH, LIST, PATCH, PROSE = ("read", "shell", "write", "edit", "search", "list", "patch",
                                                          "prose")
HOSTS = ("claude", "agy", "omp", "codex", "grok")

CODEX_SID = "11111111-2222-3333-4444-555555555555"


def patch_text(path: str, verb: str = "Update") -> str:
    return f"*** Begin Patch\n*** {verb} File: {path}\n@@\n-old\n+new\n*** End Patch"


# kind -> (tool name, arguments) for each host. ``value`` is the path or the command.
def claude_call(call: Call):
    value = call.value
    return {READ: ("Read", {"file_path": value}), SHELL: ("Bash", {"command": value, "description": "run it"}),
            WRITE: ("Write", {"file_path": value, "content": "x"}),
            EDIT: ("Edit", {"file_path": value, "old_string": "a", "new_string": "b"}),
            SEARCH: ("Grep", {"pattern": "SECRET", "path": value}), LIST: ("LS", {"path": value}),
            PROSE: ("Write", {"file_path": "/work/notes.txt", "content": f"see {value}"})}.get(call.kind)


def agy_call(call: Call):
    value = call.value
    return {READ: ("view_file", {"AbsolutePath": value}),
            SHELL: ("run_command", {"CommandLine": value, "Cwd": "/work", "toolAction": "Running", "toolSummary": "s"}),
            WRITE: ("write_to_file", {"TargetFile": value, "CodeContent": "x"}),
            EDIT: ("replace_file_content", {"TargetFile": value}),
            LIST: ("list_dir", {"DirectoryPath": value}),
            PROSE: ("write_to_file", {"TargetFile": "/work/notes.txt", "CodeContent": f"see {value}"})}.get(call.kind)


def omp_call(call: Call):
    value = call.value
    return {READ: ("read", {"path": value}), SHELL: ("bash", {"command": value}),
            WRITE: ("write", {"path": value, "content": "x"}), EDIT: ("edit", {"path": value}),
            PROSE: ("write", {"path": "/work/notes.txt", "content": f"see {value}"})}.get(call.kind)


def codex_call(call: Call):
    value = call.value
    return {SHELL: ("shell", {"command": ["bash", "-lc", value], "workdir": "/work"}),
            PATCH: ("apply_patch", patch_text(value, call.extra or "Update")),
            PROSE: ("apply_patch", patch_text("/work/notes.txt") + f"\n+see {value}")}.get(call.kind)


def grok_native_call(call: Call):
    value = call.value
    return {READ: ("read_file", {"target_file": value}),
            SHELL: ("run_terminal_command", {"command": value, "description": "run it"}),
            WRITE: ("write", {"target_file": value, "contents": "x"}),
            EDIT: ("search_replace", {"file_path": value, "old_string": "a", "new_string": "b"}),
            SEARCH: ("grep", {"pattern": "SECRET", "path": value}), LIST: ("list_dir", {"target_directory": value}),
            PROSE: ("write", {"target_file": "/work/notes.txt", "contents": f"see {value}"})}.get(call.kind)


MAPPERS = {"claude": claude_call, "agy": agy_call, "omp": omp_call, "codex": codex_call, "grok": grok_native_call}


def supported(host: str, call: Call) -> bool:
    return MAPPERS[host](call) is not None


def _claude(tmp: Path, items):
    records = [human("go")] + [assistant(use(f"toolu_{i}", name, **args)) for i, (name, args) in enumerate(items)]
    file = write_claude(main_file(tmp), *records)
    return [c.text for c in parse_claude(file) if c.kind == "tool_call"]


def _agy(tmp: Path, items):
    rows = [agy_user(0, "go")] + [planner(i + 1, "", [(name, args)]) for i, (name, args) in enumerate(items)]
    file = make_agy(tmp, rows)
    return [c.text for c in agy.parse_session_chunks(file)[0] if c.text.startswith("[proposed_tool_call:")]


def _omp(tmp: Path, items):
    tmp.mkdir(parents=True, exist_ok=True)
    file = tmp / "omp-session.jsonl"
    records = [{"type": "session", "id": "omp-session"}] + [
        {"type": "message", "message": {"role": "assistant", "content": [
            {"type": "toolCall", "id": f"call-{i}", "name": name, "arguments": args}]}}
        for i, (name, args) in enumerate(items)]
    file.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return [c.text for c in omp.parse_session_chunks(file)[0] if c.kind == "tool_call"]


def _codex(tmp: Path, items):
    tmp.mkdir(parents=True, exist_ok=True)
    file = tmp / "rollout.jsonl"
    records = [{"type": "session_meta", "payload": {"id": CODEX_SID}}]
    for i, (name, args) in enumerate(items):
        if name == "apply_patch":            # Codex stores a patch as a custom tool call whose input is the patch text
            records.append({"type": "response_item", "payload": {"type": "custom_tool_call", "name": name,
                                                                  "call_id": f"c{i}", "input": args}})
        else:
            records.append({"type": "response_item", "payload": {"type": "function_call", "name": name,
                                                                  "call_id": f"c{i}", "arguments": args}})
    file.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return [c.text for c in codex_parse(file, session_id=CODEX_SID)[0] if c.kind == "tool_call"]


def _grok(tmp: Path, items):
    records = [grok_user("go")] + [grok_call(f"call-{i}", name, args) for i, (name, args) in enumerate(items)]
    file = grok_write(grok_session_file(tmp), *records)
    return [c.text for c in grok.parse_session_chunks(file)[0] if c.kind == "tool_call"]


_RENDER = {"claude": _claude, "agy": _agy, "omp": _omp, "codex": _codex, "grok": _grok}


def render(tmp: Path, host: str, calls) -> list[str | None]:
    """The parser's text for each call, ``None`` where the host has no such tool (so the lists line up)."""
    wanted = [(i, MAPPERS[host](c)) for i, c in enumerate(calls)]
    items = [item for _i, item in wanted if item is not None]
    texts = iter(_RENDER[host](tmp / host, items) if items else [])
    out: list[str | None] = []
    for _i, item in wanted:
        out.append(next(texts) if item is not None else None)
    return out
