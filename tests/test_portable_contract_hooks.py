"""The portable-contract tools in the hook adapters, and the hook import budget.

Hooks stay advisory and local: the Codex adapter exempts only proven reads (``list_contract_candidates`` is one, because
it reads files and writes nothing; ``adopt_contracts`` and ``export_contracts`` change state like ``upsert_contracts``), the
Claude Code and AGY adapters exempt no tool by name at all, and no hook process ever imports the bundle module.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from self_directing_mcp import agy_hooks, claude_hooks, codex_hooks, server
from self_directing_mcp.claude_hooks import run_hook
from semantic_support import Claude, SID, bash, human, tool_result

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "mcp__self-directing-mcp__"

# Every tool the server registers, and whether the Codex adapter treats it as a read that needs no contract check.
# A new tool must be added here on purpose: this table is what "classified" means.
CODEX_READS = {"sync_session": True, "search_history": True, "audit_session": True, "check_action": True,
               "get_chunk": True, "list_contracts": True, "list_contract_candidates": True, "get_checklist": True,
               "audit_status": True}


def registered():
    return sorted(tool.name for tool in asyncio.run(server.mcp.list_tools()))


def test_every_registered_tool_is_classified_by_the_codex_adapter_and_the_new_ones_as_documented():
    for name in registered():
        expected = CODEX_READS.get(name, False)
        assert codex_hooks._read_only(PREFIX + name, {}) is expected, name
        assert codex_hooks._read_only("functions." + PREFIX + name, {}) is expected, name
    assert codex_hooks._read_only(PREFIX + "list_contract_candidates", {"project_dir": "/x"}) is True
    for mutating in ("upsert_contracts", "revoke_contract", "adopt_contracts", "export_contracts"):
        assert codex_hooks._read_only(PREFIX + mutating, {}) is False, mutating
    assert {"list_contract_candidates", "adopt_contracts", "export_contracts"} <= set(registered())


def test_a_read_that_a_contract_restricts_is_still_checked_so_classification_is_not_an_exemption():
    class Rule:
        scope, roles, type, regex, before_regex = "tool_call", ["assistant"], "must_not", "list_contract_candidates", None
    assert codex_hooks._read_only(PREFIX + "list_contract_candidates", {}) is True
    assert codex_hooks._read_has_constraint([Rule()], PREFIX + "list_contract_candidates", {}) is True


def test_the_claude_code_and_agy_adapters_have_no_tool_name_table():
    for module in (claude_hooks, agy_hooks):
        source = inspect.getsource(module)
        assert "_READ_TOOLS" not in source and "_SELF_READS" not in source and "_read_only" not in source


@pytest.fixture
def session(tmp_path):
    world = Claude(tmp_path)
    world.write(human("never adopt or export contracts without asking"), bash("t1", "ls"), tool_result("t1", "file"))
    world.sync()
    yield world
    world.close()


@pytest.mark.parametrize("tool", ["adopt_contracts", "export_contracts", "list_contract_candidates"])
def test_claude_code_judges_the_new_tools_like_any_other_tool_a_named_prohibition_denies_them(session, tool):
    session.upsert({"id": "no-tool", "type": "must_not", "scope": "tool_call", "regex": tool})
    payload = {"session_id": SID, "transcript_path": str(session.path), "cwd": "/synthetic/project",
               "hook_event_name": "PreToolUse", "tool_name": PREFIX + tool,
               "tool_input": {"bundle_id": "0123456789ab", "source_event_id": "claude:x:1"}}
    response = run_hook("PreToolUse", payload, engine=session.engine)
    assert response["hookSpecificOutput"]["permissionDecision"] == "deny", response


def test_claude_code_does_not_block_the_new_tools_when_no_contract_names_them(session):
    session.upsert({"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"})
    payload = {"session_id": SID, "transcript_path": str(session.path), "cwd": "/synthetic/project",
               "hook_event_name": "PreToolUse", "tool_name": PREFIX + "adopt_contracts",
               "tool_input": {"bundle_id": "0123456789ab", "source_event_id": "claude:x:1"}}
    assert run_hook("PreToolUse", payload, engine=session.engine).get("hookSpecificOutput", {}).get(
        "permissionDecision", "none") == "none"


def run_python(code: str, tmp_path: Path) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SELF_DIRECT_", "CODEX_", "OPENROUTER"))}
    env.update(PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1", HOME=str(tmp_path),
               SELF_DIRECT_CONFIG_FILE=str(tmp_path / "empty.toml"), SELF_DIRECT_INDEX_DIR=str(tmp_path / "index"))
    (tmp_path / "empty.toml").write_text("", encoding="utf-8")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def test_the_bundle_module_is_loaded_only_by_the_tools_that_need_it(tmp_path):
    code = """
import json, sys
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp import codex_hooks
name = "self_directing_mcp.contract_bundles"
engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, openrouter_api_key_file=None))
engine.upsert_contracts([{"id": "a", "type": "must_not", "regex": "x"}])
engine.list_contracts()
engine.hook_obligations("s", provider="claude")
engine.audit_session("s", provider="claude")
before = name in sys.modules
engine.list_contract_candidates()
print(json.dumps([before, name in sys.modules]))
engine.close()
"""
    assert json.loads(run_python(code, tmp_path)) == [False, True]


def test_a_command_hook_never_loads_the_bundle_module(tmp_path):
    code = """
import io, json, sys
sys.stdin = io.StringIO(json.dumps({"session_id": "s", "transcript_path": "/nonexistent.jsonl", "cwd": "/x",
                                    "hook_event_name": "UserPromptSubmit", "prompt": "hello"}))
sys.argv = ["hook", "--event=UserPromptSubmit"]
from self_directing_mcp.claude_hooks import main
real, sys.stdout = sys.stdout, io.StringIO()
try:
    main()
finally:
    sys.stdout = real
print(json.dumps("self_directing_mcp.contract_bundles" in sys.modules))
"""
    assert run_python(code, tmp_path).endswith("false")
