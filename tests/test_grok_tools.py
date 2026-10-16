"""Grok tool names and argument keys are rendered in Claude Code's vocabulary at the matching boundary.

Every row of the map is pinned here, with the argument shapes of the recorded fixtures (``tests/fixtures/grok``).
Synthetic data only.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from self_directing_mcp import grok_tools
from self_directing_mcp.tool_payload import tool_call_parts, tool_call_text

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "grok"

# (native name, native arguments, canonical name, canonical arguments, prose)
ROWS = [
    ("run_terminal_command", {"command": "echo probe-ok", "description": "Print probe-ok marker"},
     "Bash", {"command": "echo probe-ok"}, {"description": "Print probe-ok marker"}),
    ("run_terminal_command", {"command": "sleep 9", "is_background": True},
     "Bash", {"command": "sleep 9", "run_in_background": True}, {}),
    ("read_file", {"target_file": "/work/proj/notes.txt", "limit": 5},
     "Read", {"file_path": "/work/proj/notes.txt", "limit": 5}, {}),
    ("read_file", {"path": "/work/proj/notes.txt", "offset": 3},
     "Read", {"file_path": "/work/proj/notes.txt", "offset": 3}, {}),
    ("search_replace", {"file_path": "app.py", "old_string": "a", "new_string": "b"},
     "Edit", {"file_path": "app.py", "old_string": "a", "new_string": "b"}, {}),
    ("write", {"file_path": "out.txt", "content": "hello"}, "Write", {"file_path": "out.txt", "content": "hello"}, {}),
    ("write", {"target_file": "out.txt", "contents": "hello"}, "Write", {"file_path": "out.txt", "content": "hello"}, {}),
    ("grep", {"pattern": "greet", "path": "app.py"}, "Grep", {"pattern": "greet", "path": "app.py"}, {}),
    ("list_dir", {"target_directory": "."}, "LS", {"path": "."}, {}),
    ("web_search", {"query": "weather"}, "WebSearch", {"query": "weather"}, {}),
    ("web_fetch", {"url": "https://example.com/x"}, "WebFetch", {"url": "https://example.com/x"}, {}),
    ("spawn_subagent", {"prompt": "find it", "subagent_type": "explore"}, "Task",
     {"prompt": "find it", "subagent_type": "explore"}, {}),
    ("linear__save_issue", {"title": "t"}, "mcp__linear__save_issue", {"title": "t"}, {}),
    ("my-server__do_it", {"x": 1}, "mcp__my-server__do_it", {"x": 1}, {}),
]


@pytest.mark.parametrize("native,arguments,name,canonical,prose", ROWS, ids=[f"{r[0]}-{i}" for i, r in enumerate(ROWS)])
def test_every_row_of_the_map(native, arguments, name, canonical, prose):
    call = grok_tools.canonical_call(native, arguments)
    assert (call.name, call.arguments, call.prose) == (name, canonical, prose)
    assert call.native_name == native and call.mapped is True


@pytest.mark.parametrize("native,arguments,name,canonical,prose", ROWS, ids=[f"{r[0]}-{i}" for i, r in enumerate(ROWS)])
def test_the_text_is_what_a_contract_for_claude_matches(native, arguments, name, canonical, prose):
    text, meta = tool_call_parts("grok", native, arguments)
    assert text == f"[tool_call:{name}] " + json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    assert meta["native_tool_name"] == native and meta["grok_tool_map"] == grok_tools.TOOL_MAP_VERSION
    assert meta["native_arguments"] == arguments
    assert meta.get("tool_prose", {}) == prose
    # the text Claude Code's own call renders for the same canonical call, its prose excluded the same way
    assert text == tool_call_text("claude", name, {**canonical, **prose})


def test_map_documents_every_row_and_its_version():
    names = [row.native for row in grok_tools.TOOL_ROWS]
    assert names == ["run_terminal_command", "read_file", "search_replace", "write", "grep", "list_dir", "web_search",
                     "web_fetch", "spawn_subagent"]
    assert len(set(names)) == len(names) and len({row.canonical for row in grok_tools.TOOL_ROWS}) == len(names)
    assert isinstance(grok_tools.TOOL_MAP_VERSION, int) and grok_tools.TOOL_MAP_VERSION >= 1
    assert all(row.source for row in grok_tools.TOOL_ROWS)


@pytest.mark.parametrize("name", ["todo_write", "schedule", "use_tool", "Bash", "mcp__srv__tool", "x", "__x", "x__", "_",
                                  "a b__c", "", "read-file"])
def test_unknown_names_pass_through_untouched(name):
    arguments = {"a": 1, "description": "kept: only run_terminal_command has prose"}
    call = grok_tools.canonical_call(name, arguments)
    assert (call.name, call.arguments, call.mapped, call.prose) == (name, arguments, False, {})
    text, meta = tool_call_parts("grok", name, arguments)
    assert text == f"[tool_call:{name}] " + json.dumps(arguments, sort_keys=True) and meta == {}


def test_a_canonical_call_converts_to_itself_so_hook_and_parser_agree():
    for native, arguments, name, canonical, _prose in ROWS:
        once = grok_tools.canonical_call(native, arguments)
        twice = grok_tools.canonical_call(once.name, once.arguments)
        assert (twice.name, twice.arguments) == (once.name, once.arguments)


def test_a_key_is_renamed_only_when_the_canonical_key_is_free():
    # both spellings present: nothing is overwritten and nothing is dropped
    call = grok_tools.canonical_call("read_file", {"target_file": "/a", "file_path": "/b"})
    assert call.arguments == {"target_file": "/a", "file_path": "/b"}
    call = grok_tools.canonical_call("read_file", {"target_file": "/a", "path": "/b"})
    assert call.arguments == {"file_path": "/a", "path": "/b"}


def test_prose_is_kept_out_of_the_text_and_in_the_metadata_only_for_the_shell():
    text, meta = tool_call_parts("grok", "run_terminal_command",
                                 {"command": "ls", "description": "without connecting to it with sqlite3"})
    assert "sqlite3" not in text and meta["tool_prose"] == {"description": "without connecting to it with sqlite3"}
    text, _meta = tool_call_parts("grok", "spawn_subagent", {"description": "uses sqlite3"})
    assert "sqlite3" in text                      # unconfirmed prose stays payload: matched, the cautious side


def test_non_dictionary_arguments_and_non_string_names_never_raise():
    assert grok_tools.canonical_call("read_file", "raw").arguments == "raw"
    assert grok_tools.canonical_call("run_terminal_command", None).name == "Bash"
    assert grok_tools.canonical_call(None, {"a": 1}).name is None
    assert grok_tools.canonical_call(5, 5).mapped is False


def test_native_arguments_are_capped_not_dropped_silently():
    big = {"command": "x" * 5000, "description": "d"}
    meta = grok_tools.native_meta(grok_tools.canonical_call("run_terminal_command", big), big)
    assert meta["native_arguments"]["command"].endswith("...") and len(meta["native_arguments"]["command"]) < 1100
    huge = {f"k{i}": "y" * 900 for i in range(20)}
    meta = grok_tools.native_meta(grok_tools.canonical_call("run_terminal_command", huge), huge)
    assert meta["native_arguments"]["truncated"] is True and "k0" in meta["native_arguments"]["keys"]
    assert grok_tools.native_meta(grok_tools.canonical_call("todo_write", {"a": 1}), {"a": 1}) == {}


def test_describe_names_the_grok_tool_a_finding_is_about():
    assert grok_tools.describe("run_terminal_command", "Bash") == "Grok tool run_terminal_command (matched as Bash)"
    assert grok_tools.describe("todo_write", "todo_write") == "Grok tool todo_write"


def test_the_recorded_tool_calls_render_as_claude_calls():
    seen = {}
    for name in ("updates_session2.jsonl", "updates_session3.jsonl"):
        for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines():
            update = json.loads(line)["params"]["update"]
            if update["sessionUpdate"] == "tool_call":
                seen[update["title"]] = tool_call_parts("grok", update["title"], update["rawInput"])[0]
    assert seen == {
        "read_file": '[tool_call:Read] {"file_path": "/work/proj/secret.env"}',
        "run_terminal_command": '[tool_call:Bash] {"command": "echo probe-ok"}',
        "list_dir": '[tool_call:LS] {"path": "."}',
        "grep": '[tool_call:Grep] {"path": "app.py", "pattern": "greet"}',
        "search_replace": '[tool_call:Edit] {"file_path": "app.py", "new_string": "return \\"hi \\" + name", '
                          '"old_string": "return \\"hello \\" + name"}',
    }


def test_map_module_stays_standard_library_only():
    source = Path(grok_tools.__file__).read_text(encoding="utf-8")
    imports = set(re.findall(r"^(?:from|import) ([\w.]+)", source, re.MULTILINE))
    assert imports <= {"__future__", "dataclasses", "json", "re", "typing"}
