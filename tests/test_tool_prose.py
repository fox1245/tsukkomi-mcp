"""B6(a): a model-written prose field is not part of the tool-call payload a contract matches.

Claude Code's ``Bash`` input carries a ``description`` the model writes about the call. A prohibition such as
``sqlite3`` used to fire on ``ls -l data/app.sqlite3`` because that prose said "without connecting to it with
sqlite3". The prose is now kept out of the call text (and kept in ``meta.tool_prose``); the proposed action a hook
asks about is built by the same function, so history and proposal cannot disagree.
"""
from __future__ import annotations

import json

import pytest

from self_directing_mcp import agy, agy_hook_state, claude
from self_directing_mcp.claude_hooks import run_hook
from self_directing_mcp.index.ingest import current_parser_version
from self_directing_mcp.tool_payload import split_prose, tool_call_parts, tool_call_text
from semantic_support import SID, Claude, assistant, bash, finding, human, tool_result, use

FORBIDDEN = {"id": "no-sqlite", "type": "must_not", "scope": "tool_call", "regex": r"\bsqlite3\s"}
CAREFUL = "ls -l data/app.sqlite3"
PROSE = "Check the file size without connecting to it with sqlite3 at all"


@pytest.fixture
def session(tmp_path):
    world = Claude(tmp_path)
    yield world
    world.close()


# ---------------------------------------------------------------- the helper

def test_claude_description_is_prose_and_command_is_payload():
    payload, prose = split_prose("claude", "Bash", {"command": CAREFUL, "description": PROSE, "timeout": 5})
    assert payload == {"command": CAREFUL, "timeout": 5} and prose == {"description": PROSE}
    text_, meta = tool_call_parts("claude", "Bash", {"command": CAREFUL, "description": PROSE})
    assert text_ == '[tool_call:Bash] {"command": "ls -l data/app.sqlite3"}'
    assert meta == {"tool_prose": {"description": PROSE}}


def test_other_claude_built_in_tools_with_a_description_are_covered():
    for name in ("Task", "Agent", "Monitor", "TaskCreate"):
        assert split_prose("claude", name, {"prompt": "x", "description": "y"})[0] == {"prompt": "x"}


def test_an_mcp_tool_keeps_its_own_description_field_as_payload():
    """An MCP tool defines its own schema: a ``description`` there may be the real content (an issue body)."""
    arguments = {"title": "t", "description": "touches sqlite3 here"}
    assert split_prose("claude", "mcp__tracker__create_issue", arguments) == (arguments, {})


def test_unclassified_fields_stay_payload():
    arguments = {"command": "x", "reason": "because sqlite3", "note": "n", "description": 7}
    # A non-string ``description`` is not prose either: only a string value is excluded.
    assert split_prose("claude", "Bash", arguments) == (arguments, {})
    assert tool_call_text("claude", "Bash", arguments) == "[tool_call:Bash] " + json.dumps(arguments, sort_keys=True)


def test_agy_tool_labels_are_prose_and_nothing_else_is():
    arguments = {"CommandLine": "npm test", "Cwd": "/x", "toolAction": "Running tests", "toolSummary": "Execute tests",
                 "Description": "unconfirmed", "Instruction": "unconfirmed"}
    payload, prose = split_prose("agy", "run_command", arguments)
    assert prose == {"toolAction": "Running tests", "toolSummary": "Execute tests"}
    # Fields that are only suspected to be prose stay payload until a fixture or the code confirms them.
    assert set(payload) == {"CommandLine", "Cwd", "Description", "Instruction"}


@pytest.mark.parametrize("provider", ["codex", "omp", "grokbot", None])
def test_hosts_without_a_confirmed_prose_field_are_untouched(provider):
    arguments = {"command": "x", "description": "d", "justification": "j", "_i": "i", "toolAction": "a"}
    assert split_prose(provider, "shell", arguments) == (arguments, {})


def test_a_string_argument_is_never_split():
    assert tool_call_text("claude", "Bash", "ls description") == "[tool_call:Bash] ls description"


def test_prose_in_meta_is_capped():
    _text, meta = tool_call_parts("claude", "Bash", {"command": "x", "description": "p" * 100_000})
    assert len(meta["tool_prose"]["description"]) == 4000


# ---------------------------------------------------------------- the false positive, end to end

def history(command, description):
    return [human("be careful with the database files"), bash("t1", command, description), tool_result("t1")]


def test_the_prose_is_kept_out_of_the_indexed_call_text_and_stored_in_meta(session):
    session.write(*history(CAREFUL, PROSE))
    session.sync()
    (call,) = [c for c in session.chunks() if c.kind == "tool_call"]
    assert call.text == '[tool_call:Bash] {"command": "ls -l data/app.sqlite3"}'
    assert call.meta["tool_prose"] == {"description": PROSE}


def test_the_careful_command_is_no_longer_a_violation_through_its_description(session):
    session.write(*history(CAREFUL, PROSE))
    session.upsert(FORBIDDEN)
    audit = session.audit()
    assert finding(audit, "no-sqlite")["verdict"] == "clean", audit["findings"]
    assert audit["verdict"] == "clean"


def test_a_violation_inside_the_command_is_still_caught_when_the_description_is_innocent(session):
    session.write(*history("sqlite3 data/app.sqlite3 .tables", "List the tables"))
    session.upsert(FORBIDDEN)
    item = finding(session.audit(), "no-sqlite")
    assert item["verdict"] == "violation"
    assert "sqlite3 data/app.sqlite3" in item["evidence"][0]["snippet"]


def test_a_violation_in_the_command_is_caught_even_when_the_description_says_otherwise(session):
    session.write(*history("sqlite3 data/app.sqlite3 .tables", "Look at the file without sqlite"))
    session.upsert(FORBIDDEN)
    assert finding(session.audit(), "no-sqlite")["verdict"] == "violation"


def test_an_mcp_tool_description_is_still_matched(session):
    session.write(human("x"), assistant(use("m1", "mcp__tracker__create_issue", title="t",
                                            description="run sqlite3 against it")), tool_result("m1"))
    session.upsert(FORBIDDEN)
    assert finding(session.audit(), "no-sqlite")["verdict"] == "violation"


# ---------------------------------------------------------------- parser version and reparse

def test_the_parser_version_was_bumped_and_a_session_indexed_before_is_reparsed(session, monkeypatch):
    assert claude.PARSER_VERSION >= 3 and current_parser_version("claude") == claude.PARSER_VERSION
    session.write(*history(CAREFUL, PROSE))
    session.upsert(FORBIDDEN)
    with monkeypatch.context() as old:
        # What parser v2 produced: the call text carried the whole input, description included.
        old.setattr(claude, "PARSER_VERSION", 2)
        old.setattr(claude, "tool_call_parts", lambda provider, name, arguments: (
            f"[tool_call:{name}] " + json.dumps(arguments, ensure_ascii=False, sort_keys=True), {}))
        first = session.sync()
        assert first["parser_version"] == 2
        assert finding(session.audit(), "no-sqlite")["verdict"] == "violation"   # the old false positive
    second = session.sync()
    assert second["reindexed"] and second["reindex_reason"] == "parser_updated"
    assert second["parser_version"] == claude.PARSER_VERSION
    assert finding(session.audit(), "no-sqlite")["verdict"] == "clean"


def test_the_agy_parser_version_was_bumped():
    assert agy.PARSER_VERSION >= 8 and current_parser_version("agy") == agy.PARSER_VERSION


# ---------------------------------------------------------------- the proposed action (hooks) follows the same rule

def pre_tool_use(session, tool_input, tool="Bash"):
    payload = {"session_id": SID, "transcript_path": str(session.path), "cwd": "/synthetic/project",
               "hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": tool_input}
    return run_hook("PreToolUse", payload, engine=session.engine)


def decision(response):
    return response.get("hookSpecificOutput", {}).get("permissionDecision", "none")


def test_a_hook_proposal_ignores_the_description_exactly_like_history(session):
    session.write(human("hi"))
    session.upsert(FORBIDDEN)
    careful = pre_tool_use(session, {"command": CAREFUL, "description": PROSE})
    assert decision(careful) == "none", careful
    real = pre_tool_use(session, {"command": "sqlite3 data/app.sqlite3 .tables", "description": "List the tables"})
    assert decision(real) == "deny", real


def test_indexed_and_proposed_text_are_identical_for_the_same_call(session):
    """One shared builder: the same input renders to the same text on both sides."""
    arguments = {"command": CAREFUL, "description": PROSE, "timeout": 120000}
    session.write(human("hi"), assistant(use("t1", "Bash", **arguments)), tool_result("t1"))
    session.upsert({"id": "any-bash", "type": "must_not", "scope": "tool_call", "regex": "ls -l"})
    session.sync()
    (indexed,) = [c for c in session.chunks() if c.kind == "tool_call"]
    snippets = {
        "history": finding(session.audit(), "any-bash")["evidence"][0]["snippet"],
        "proposal": finding(session.check("Bash", arguments), "any-bash")["evidence"][0]["snippet"],
    }
    assert snippets["history"] == snippets["proposal"] == indexed.text
    assert PROSE not in snippets["proposal"] and tool_call_text("claude", "Bash", arguments) == indexed.text


def test_the_fail_open_high_risk_check_uses_the_same_text(tmp_path):
    """``agy_hook_state.high_risk_match`` answers when the runtime cannot load; it must not see the prose either."""
    contracts = tmp_path / "contracts.json"
    contracts.write_text(json.dumps({"contracts": [
        {"id": "no-sqlite", "type": "must_not", "scope": "tool_call", "regex": r"\bsqlite3\s", "risk": "high",
         "provider": "claude"}]}), encoding="utf-8")
    careful = {"name": "Bash", "args": {"command": CAREFUL, "description": PROSE}}
    real = {"name": "Bash", "args": {"command": "sqlite3 data/app.sqlite3 .tables", "description": "List"}}
    assert agy_hook_state.high_risk_match(contracts, SID, careful, "claude") is None
    assert agy_hook_state.high_risk_match(contracts, SID, real, "claude") == "no-sqlite"


# ---------------------------------------------------------------- AGY: the labels in the recorded fixture

AGY_SID = "11111111-1111-4111-8111-111111111111"


def agy_world(tmp_path):
    from self_directing_mcp.config import Settings
    from self_directing_mcp.engine import SelfDirectEngine
    root = tmp_path / "app"
    path = root / "brain" / AGY_SID / ".system_generated" / "logs" / "transcript.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE",
         "created_at": "2026-10-01T00:00:00Z", "content": "<USER_REQUEST>hello</USER_REQUEST>"},
        {"step_index": 1, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE",
         "created_at": "2026-10-01T00:00:01Z", "tool_calls": [{"name": "run_command", "args": {
             "CommandLine": json.dumps("ls -l data"), "toolAction": json.dumps("Avoiding sqlite3"),
             "toolSummary": json.dumps("List data without sqlite3 ")}}]},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=tmp_path / "index",
                                       agy_app_data_dirs=[root], openrouter_api_key_file=None))
    return engine, path


def test_agy_labels_are_kept_out_of_the_proposal_text_and_the_proposed_action(tmp_path):
    engine, path = agy_world(tmp_path)
    try:
        engine.sync_session(session_id=AGY_SID, provider="agy", path=str(path), embed=False)
        (proposal,) = [c for c in engine.store.list_chunks(AGY_SID, "agy")
                       if c.meta.get("evidence_origin") == "model_proposal"]
        assert proposal.text == '[proposed_tool_call:run_command] {"CommandLine": "ls -l data"}'
        assert proposal.meta["tool_prose"] == {"toolAction": "Avoiding sqlite3", "toolSummary": "List data without sqlite3 "}
        engine.upsert_contracts([{"id": "no-sqlite", "provider": "agy", "type": "must_not", "scope": "tool_call",
                                  "regex": r"sqlite3\b"}])
        careful = {"tool_name": "run_command", "arguments": {
            "CommandLine": "ls -l data", "toolAction": "Avoiding sqlite3", "toolSummary": "no sqlite3 here"}}
        result = engine.check_action(AGY_SID, careful, provider="agy", path=str(path))
        assert finding(result, "no-sqlite")["verdict"] == "clean"
        real = {"tool_name": "run_command", "arguments": {"CommandLine": "sqlite3 a.db", "toolAction": "Querying"}}
        assert finding(engine.check_action(AGY_SID, real, provider="agy", path=str(path)),
                       "no-sqlite")["verdict"] == "violation"
    finally:
        engine.close()
