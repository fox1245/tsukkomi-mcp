"""``path_guard`` through the real evaluators: hooks, the history audit and the incremental evaluator.

A ``path_guard`` contract is stored typed and lowered to a regex when it is loaded; this file proves that every place a
contract is evaluated sees it: the Claude Code, AGY and Grok hooks (deny the forbidden file-read tool call and the shell
``cat``, allow the neighbour, name the contract and the path in the reason), the Codex hook and the OMP check, the history
audit by the reference evaluator and by the incremental evaluator (same verdicts, same text), the high-risk hook fast path
that reads ``contracts.json`` without the runtime, the context lines, ``list_contracts`` and the upsert response.

Synthetic data only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_directing_mcp import agy_hook_state, contract_paths, grok_hook_state
from self_directing_mcp.agy_hooks import handle_pre_invocation, handle_pre_tool_use
from self_directing_mcp.claude_hooks import run_hook as claude_hook
from self_directing_mcp.codex_hooks import handle_hook as codex_hook
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.grok_hooks import run_hook as grok_hook
from self_directing_mcp.schemas import ContractRule
from self_directing_mcp.tool_payload import tool_call_text
from grok_support import SID as GROK_SID, hook_payload as grok_payload, session_file as grok_file, user as grok_user
from grok_support import write as grok_write
from semantic_support import SID as CLAUDE_SID, Claude, assistant, call, event, evaluate, human, text, use, user_turn
from test_agy_parser_v6 import SID as AGY_SID, make as make_agy, user as agy_user
from test_incremental_engine import Twin
from test_reliability import SID as CODEX_SID, session as codex_session, write as write_codex  # noqa: F401
from test_portable_contracts import world  # noqa: F401  (two devices with their own indexes)

GUARD = {"id": "no-dotenv", "type": "must_not", "severity": "high", "description": "Never read the secrets in .env.",
         "path_guard": {"paths": [".env"], "except": [".env.example"], "modes": "any"}}
READ_ONLY = {**GUARD, "path_guard": {**GUARD["path_guard"], "modes": "read"}}


def denied_reason(response):
    return response.get("reason") or response.get("hookSpecificOutput", {}).get("permissionDecisionReason", "")


# ------------------------------------------------------------------------------------------------- Claude Code

@pytest.fixture
def claude(tmp_path):
    session = Claude(tmp_path)
    session.write(human("never read the secrets in .env"))
    yield session
    session.close()


def claude_pre(claude, tool, arguments):
    payload = {"session_id": CLAUDE_SID, "transcript_path": str(claude.path), "cwd": "/work",
               "hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": arguments}
    return claude_hook("PreToolUse", payload, engine=claude.engine)


def claude_decision(response):
    return response.get("hookSpecificOutput", {}).get("permissionDecision", "none")


def test_claude_hook_denies_the_file_read_tool_and_the_shell_reader_and_allows_the_neighbours(claude):
    claude.upsert(GUARD)
    for tool, arguments in (("Read", {"file_path": "/work/.env"}), ("Bash", {"command": "cat /work/.env", "description": "x"}),
                            ("Write", {"file_path": "/work/.env", "content": "A=1"}),
                            ("Edit", {"file_path": "/work/.env", "old_string": "a", "new_string": "b"}),
                            ("Grep", {"pattern": "KEY", "path": "/work/.env"})):
        response = claude_pre(claude, tool, arguments)
        assert claude_decision(response) == "deny", (tool, response)
        reason = denied_reason(response)
        assert "[no-dotenv]" in reason and ".env" in reason and "forbidden path" in reason, reason
        assert "tool_call:" not in reason and "(?:" not in reason                    # the guard, not the generated regex
    for tool, arguments in (("Read", {"file_path": "/work/.env.example"}), ("Read", {"file_path": "/work/.environment"}),
                            ("Read", {"file_path": "/work/src/app.py"}), ("Bash", {"command": "cat /work/my.env"}),
                            ("Bash", {"command": "git commit -m 'update .env.example'"}),
                            ("Write", {"file_path": "/work/notes.md", "content": "see .env"})):
        assert claude_decision(claude_pre(claude, tool, arguments)) == "none", (tool, arguments)


def test_claude_mode_read_leaves_a_write_alone(claude):
    claude.upsert(READ_ONLY)
    assert claude_decision(claude_pre(claude, "Read", {"file_path": "/work/.env"})) == "deny"
    assert claude_decision(claude_pre(claude, "Bash", {"command": "head -n 3 .env"})) == "deny"
    assert claude_decision(claude_pre(claude, "Write", {"file_path": "/work/.env", "content": "A=1"})) == "none"
    assert claude_decision(claude_pre(claude, "Bash", {"command": "echo A=1 > .env"})) == "none"


def test_a_typed_contract_is_stored_typed_and_listed_typed(claude):
    response = claude.upsert(GUARD)
    (stored,) = response["contracts"]
    assert stored["path_guard"] == GUARD["path_guard"] and "regex" not in stored and stored["scope"] == "tool_call"
    listed = claude.engine.list_contracts()["contracts"][0]
    assert listed["path_guard"] == GUARD["path_guard"] and "regex" not in listed
    raw = claude.engine.settings.resolve_contracts_path().read_text(encoding="utf-8")
    assert '"path_guard"' in raw and '"regex"' not in raw and "tool_call:" not in raw
    # in memory every evaluator sees an ordinary regex
    (rule,) = claude.engine.hook_obligations(CLAUDE_SID, provider="claude")["contracts"]
    assert rule.regex and rule.path_guard.paths == [".env"]


def test_the_upsert_response_suggests_a_path_guard_for_a_shell_scoped_regex_and_changes_nothing(claude):
    shell_scoped = {"id": "no-dotenv", "type": "must_not", "scope": "tool_call", "regex": r"(cat|head|tail|less)\s+\S*\.env",
                    "description": "Never print the secrets in .env."}
    response = claude.upsert(shell_scoped)
    assert response["contracts"][0]["regex"] == shell_scoped["regex"] and "path_guard" not in response["contracts"][0]
    (entry,) = [c for c in response["fragility"]["contracts"] if c["contract_id"] == "no-dotenv"]
    suggestion = entry["suggested_path_guard"]
    assert suggestion["path_guard"] == {"paths": [".env"], "except": [], "modes": "read"}
    ready = suggestion["contract"]
    assert ready["id"] == "no-dotenv" and ready["path_guard"] == suggestion["path_guard"] and "regex" not in ready
    assert "nothing was changed" in suggestion["note"].lower()
    assert ContractRule.model_validate(ready).regex                                  # it can be re-submitted as it is
    # the stored contract is still the regex, and the file-read tool is still not denied by it
    assert claude.engine.list_contracts()["contracts"][0]["regex"] == shell_scoped["regex"]
    assert claude_decision(claude_pre(claude, "Read", {"file_path": "/work/.env"})) == "none"
    # a regex that is not shell-scoped, or one that names no path, gets none
    plain = claude.upsert({"id": "plain", "type": "must_not", "scope": "tool_call", "regex": r"\.env(?![\w.])"})
    assert not [c for c in plain["fragility"]["contracts"] if "suggested_path_guard" in c]
    # re-submitting the suggestion replaces the regex: a changed verifier, so it needs a user anchor
    held = claude.upsert(ready)
    assert held["pending"][0]["reasons"] == ["regex changed"]


def test_a_typed_contract_is_not_analyzed_by_the_fragility_report(claude):
    response = claude.upsert({**GUARD, "path_guard": {"paths": [".env"], "modes": "read"}})
    assert response["fragility"]["contracts"] == [] and response["fragility"]["analyzed"] == 0


def test_the_context_lines_show_the_guard_not_the_generated_regex(claude):
    claude.upsert({**GUARD, "description": ""})
    from self_directing_mcp.claude_hooks import handle_session_start
    from self_directing_mcp.hook_policy import read_obligations
    obligations = read_obligations(claude.engine, "claude", CLAUDE_SID, str(claude.path))
    assert contract_paths.verifier_text(obligations["contracts"][0]) == "path_guard .env (except .env.example); mode any"
    context = handle_session_start(claude.engine, type("Call", (), {"session_id": CLAUDE_SID, "transcript_path": str(claude.path),
                                                                    "event": "SessionStart"})())
    line = json.dumps(context)
    assert "no-dotenv" in line and "path_guard .env" in line and "tool_call:" not in line and "(?:" not in line


# ---------------------------------------------------------------------------------------------------------- AGY

@pytest.fixture
def agy_engine(tmp_path):
    root = tmp_path / "app"
    path = make_agy(root, [agy_user(0, "never read the secrets in .env")])
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=tmp_path / "index",
                                       agy_app_data_dirs=[root], openrouter_api_key_file=None))
    yield engine, path
    engine.close()


def test_agy_hook_denies_view_file_run_command_and_write_to_file_and_allows_the_neighbour(agy_engine):
    engine, path = agy_engine
    engine.upsert_contracts([GUARD])
    for name, args in (("view_file", {"AbsolutePath": "/work/.env"}),
                       ("run_command", {"CommandLine": "cat /work/.env", "Cwd": "/work", "toolAction": "Reading"}),
                       ("write_to_file", {"TargetFile": "/work/.env", "CodeContent": "A=1"}),
                       ("replace_file_content", {"TargetFile": "/work/.env"}),
                       ("list_dir", {"DirectoryPath": "/work/.env"})):
        response = handle_pre_tool_use(engine, AGY_SID, {"name": name, "args": args}, str(path))
        assert response["decision"] == "deny", (name, response)
        assert "[no-dotenv]" in response["reason"] and "forbidden path .env" in response["reason"]
    for name, args in (("view_file", {"AbsolutePath": "/work/.env.example"}), ("view_file", {"AbsolutePath": "/work/a.py"}),
                       ("run_command", {"CommandLine": "ls /work", "Cwd": "/work.env"}),
                       ("run_command", {"CommandLine": "ls", "toolAction": "Reading the .env file"}),
                       ("write_to_file", {"TargetFile": "/work/n.md", "CodeContent": "see .env"})):
        assert handle_pre_tool_use(engine, AGY_SID, {"name": name, "args": args}, str(path))["decision"] == "allow", (name, args)


def test_agy_pre_invocation_context_names_the_guard(agy_engine):
    engine, path = agy_engine
    engine.upsert_contracts([{**GUARD, "description": ""}])
    line = handle_pre_invocation(engine, AGY_SID, {"transcriptPath": str(path)})["injectSteps"][0]["ephemeralMessage"]
    assert "[no-dotenv]" in line and "path_guard .env" in line and "(?:" not in line


# ---------------------------------------------------------------------------------------------------------- Grok

@pytest.fixture
def grok_engine(tmp_path, monkeypatch):
    sessions = tmp_path / "sessions"
    path = grok_write(grok_file(sessions), grok_user("never read the secrets in .env"))
    for name in ("SELF_DIRECT_GROK_ENFORCEMENT", "SELF_DIRECT_CONTRACTS_PATH", "SELF_DIRECT_SESSION_PROVIDER", "GROK_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("SELF_DIRECT_GROK_SESSIONS_DIR", str(sessions))
    monkeypatch.setenv("SELF_DIRECT_LOCAL_ONLY", "true")
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=tmp_path / "index",
                                       grok_sessions_dir=sessions, openrouter_api_key_file=None))
    yield engine, path
    engine.close()


def test_grok_hook_matches_its_own_tools_through_claudes_names(grok_engine):
    engine, path = grok_engine
    engine.upsert_contracts([GUARD])
    forbidden = (("read_file", {"target_file": "/work/proj/.env", "limit": 5}),
                 ("run_terminal_command", {"command": "cat /work/proj/.env", "description": "Print it"}),
                 ("write", {"target_file": "/work/proj/.env", "contents": "A=1"}),
                 ("search_replace", {"file_path": "/work/proj/.env", "old_string": "a", "new_string": "b"}),
                 ("grep", {"pattern": "KEY", "path": "/work/proj/.env"}))
    for tool, arguments in forbidden:
        response = grok_hook("PreToolUse", grok_payload("PreToolUse", path, tool=tool, tool_input=arguments), engine=engine)
        assert response["decision"] == "deny", (tool, response)
        assert "[no-dotenv]" in response["reason"] and "forbidden path .env" in response["reason"], response
    for tool, arguments in (("read_file", {"target_file": "/work/proj/.env.example"}),
                            ("read_file", {"target_file": "/work/proj/notes.txt"}),
                            ("run_terminal_command", {"command": "echo probe-ok", "description": "mentions .env"})):
        response = grok_hook("PreToolUse", grok_payload("PreToolUse", path, tool=tool, tool_input=arguments), engine=engine)
        assert response["decision"] == "allow" and "[no-dotenv]" not in json.dumps(response), (tool, response)


def test_a_high_risk_typed_contract_is_matched_by_the_hook_fast_path_without_the_runtime(tmp_path):
    contracts = tmp_path / "contracts.json"
    typed = {"id": "no-dotenv", "type": "must_not", "scope": "tool_call", "risk": "high",
             "path_guard": {"paths": [".env"], "except": [".env.example"], "modes": "any"}}
    contracts.write_text(json.dumps({"contracts": [typed]}), encoding="utf-8")        # no regex key at all
    gate = lambda name, args, provider="agy": agy_hook_state.high_risk_match(
        str(contracts), AGY_SID, {"name": name, "args": args}, provider)
    assert gate("view_file", {"AbsolutePath": "/work/.env"}) == "no-dotenv"
    assert gate("run_command", {"CommandLine": "cat .env"}) == "no-dotenv"
    assert gate("view_file", {"AbsolutePath": "/work/.env.example"}) is None           # not "applies to every call"
    assert gate("view_file", {"AbsolutePath": "/work/a.py"}) is None
    assert gate("Read", {"file_path": "/work/.env"}, "claude") == "no-dotenv"
    assert gate("read_file", {"target_file": "/work/.env"}, "grok") == "no-dotenv"
    assert grok_hook_state.high_risk_match is agy_hook_state.high_risk_match             # one function for every host
    # a stored guard the loader would refuse still asks (cannot be decided: it applies)
    contracts.write_text(json.dumps({"contracts": [{**typed, "path_guard": {"paths": []}}]}), encoding="utf-8")
    assert gate("view_file", {"AbsolutePath": "/work/a.py"}) == "no-dotenv"
    # a plain regex contract is read exactly as before
    contracts.write_text(json.dumps({"contracts": [{**typed, "path_guard": None, "regex": "rm -rf"}]}), encoding="utf-8")
    assert gate("run_command", {"CommandLine": "rm -rf /x"}) == "no-dotenv" and gate("run_command", {"CommandLine": "ls"}) is None


# --------------------------------------------------------------------------------------------- Codex and OMP

def test_codex_hook_and_check_action_flag_the_shell_reader_and_the_patch(codex_session):
    engine, path = codex_session
    engine.upsert_contracts([GUARD])
    forbidden = (("shell", {"command": ["bash", "-lc", "cat .env"], "workdir": "/work"}),
                 ("exec_command", {"cmd": "head -n 3 .env"}),
                 ("apply_patch", "*** Begin Patch\n*** Update File: sub/.env\n@@\n-a\n+b\n*** End Patch"))
    for tool, arguments in forbidden:
        verdict = engine.check_action(CODEX_SID, {"tool_name": tool, "arguments": arguments}, provider="codex", path=str(path))
        assert verdict["verdict"] == "violation", (tool, verdict)
        (finding,) = [f for f in verdict["findings"] if f["contract_id"] == "no-dotenv"]
        assert "forbidden path .env" in finding["reason"] and "path_guard" in finding["basis"]
        assert "(?:" not in finding["basis"] and "(?:" not in finding["reason"]
    clean = engine.check_action(CODEX_SID, {"tool_name": "shell", "arguments": {"command": ["bash", "-lc", "cat .env.example"]}},
                                provider="codex", path=str(path))
    assert clean["verdict"] == "clean"
    # the hook reports it as data and never blocks; a proven read is no longer exempt once a restriction names it
    flagged = codex_hook(engine, "PreToolUse", CODEX_SID, str(path), "shell", {"cmd": "rg KEY .env"})
    context = flagged["hookSpecificOutput"]["additionalContext"]
    assert "no-dotenv" in context and "violation" in context and "decision" not in flagged
    assert codex_hook(engine, "PreToolUse", CODEX_SID, str(path), "shell", {"cmd": "rg KEY src"}) == {}


def test_omp_check_action_flags_read_and_bash_and_not_a_write_in_mode_read(tmp_path):
    root = tmp_path / "sessions"
    root.mkdir()
    file = root / "session.jsonl"
    file.write_text(json.dumps({"type": "session", "id": "session-1"}) + "\n" + json.dumps({
        "type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "never read .env"}]}}) + "\n",
        encoding="utf-8")
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=tmp_path / "index", omp_sessions_dir=root,
                                       openrouter_api_key_file=None, use_fake_embedder=False))
    try:
        engine.upsert_contracts([READ_ONLY])

        def check(tool, arguments):
            return engine.check_action("session-1", {"tool_name": tool, "arguments": arguments}, provider="omp",
                                       path=str(file))["verdict"]

        assert check("read", {"path": "/work/.env"}) == "violation"
        assert check("bash", {"command": "cat /work/.env"}) == "violation"
        assert check("write", {"path": "/work/.env", "content": "A=1"}) == "clean"
        assert check("read", {"path": "/work/.env.example"}) == "clean"
    finally:
        engine.close()


# -------------------------------------------------------- the history audit: reference and incremental evaluator

HISTORY = [
    ("Read", {"file_path": "/work/src/app.py"}, False), ("Read", {"file_path": "/work/.env"}, True),
    ("Bash", {"command": "ls -la"}, False), ("Bash", {"command": "cat /work/.env.example"}, False),
    ("Bash", {"command": "head -n 2 /work/.env"}, True), ("Write", {"file_path": "/work/.env", "content": "A=1"}, True),
    ("Read", {"file_path": "/work/.environment"}, False),
]


def history_events(items):
    events = [user_turn("never read the secrets in .env")]
    for index, (tool, arguments, _bad) in enumerate(items):
        events.append(event("tool_call", tool_call_text("claude", tool, arguments), "assistant", call_id=f"c{index}",
                            tool_name=tool))
    return events


@pytest.mark.parametrize("mode", ["any", "read"])
def test_reference_and_incremental_evaluators_give_the_same_finding_for_every_prefix(mode):
    rule = {**GUARD, "path_guard": {**GUARD["path_guard"], "modes": mode}}
    for count in range(1, len(HISTORY) + 1):
        items = HISTORY[:count]
        finding = evaluate([rule], history_events(items))[0]            # evaluate() fails if the two evaluators differ
        wanted = [item for item in items if item[2] and (mode == "any" or item[0] != "Write")]
        assert finding.verdict == ("violation" if wanted else "clean"), (mode, count, finding)
        if wanted:
            assert finding.reason.startswith("forbidden path .env (except .env.example); mode " + mode)
            assert finding.basis.startswith("no-dotenv.path_guard .env") and "(?:" not in finding.basis
            assert len(finding.evidence) == len(wanted)
        else:
            assert finding.basis == "no-dotenv.path_guard did not match any inspected event"


PROPOSALS = {
    "claude": (("Read", {"file_path": "/work/.env"}), ("Read", {"file_path": "/work/.env.example"})),
    "agy": (("view_file", {"AbsolutePath": "/work/.env"}), ("view_file", {"AbsolutePath": "/work/.env.example"})),
    "omp": (("read", {"path": "/work/.env"}), ("read", {"path": "/work/.env.example"})),
    "codex": (("shell", {"command": ["bash", "-lc", "cat .env"]}), ("shell", {"command": ["bash", "-lc", "cat .env.example"]})),
    "grok": (("read_file", {"target_file": "/work/.env"}), ("read_file", {"target_file": "/work/.env.example"})),
}


@pytest.mark.parametrize("host", sorted(PROPOSALS))
def test_both_evaluators_agree_on_a_proposed_action_of_every_host(host):
    events = [user_turn("never read the secrets in .env")]
    for (tool, arguments), verdict in zip(PROPOSALS[host], ("violation", "clean")):
        proposed = event("tool_call", tool_call_text(host, tool, arguments), "assistant", tool_name=tool, proposed=True)
        finding = evaluate([GUARD], events, host, proposed=proposed)[0]
        assert finding.verdict == verdict, (host, tool, finding)
        if verdict == "violation":
            assert finding.reason.startswith("forbidden path .env") and "path_guard" in finding.basis


def twin_with_session(tmp_path, monkeypatch):
    from test_claude_provider import SID as SESSION, main_file
    for name in ("CLAUDE_CONFIG_DIR", "SELF_DIRECT_CLAUDE_PROJECTS_DIR"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "projects"
    twin = Twin(tmp_path, "claude", claude_projects_dir=root)
    return twin, main_file(root), SESSION


def test_claude_history_audit_is_the_same_incrementally_and_in_full(tmp_path, monkeypatch):
    from test_claude_provider import human as human_turn, tool_result, use as use_tool, write as write_claude
    twin, path, session = twin_with_session(tmp_path, monkeypatch)
    try:
        write_claude(path, human_turn("never read the secrets in .env"), assistant(use_tool("t0", "Bash", command="ls")),
                     tool_result("t0", "ok", block_extra={"is_error": False}))
        twin.upsert([{**GUARD, "provider": "claude"}, {**READ_ONLY, "id": "reads-only", "provider": "claude"}])
        twin.fast.settings.audit_incremental_min_events = 1
        first, _ = twin.audit(session)
        assert {f["contract_id"]: f["verdict"] for f in first["findings"]} == {"no-dotenv": "clean", "reads-only": "clean"}
        verdicts = []
        steps = [
            [assistant(use_tool("t1", "Read", file_path="/work/.env.example")), tool_result("t1", "x")],
            [assistant(use_tool("t2", "Write", file_path="/work/.env", content="A=1")), tool_result("t2", "wrote")],
            [assistant(use_tool("t3", "Bash", command="cat /work/.env")), tool_result("t3", "SECRET=1")],
            [assistant(use_tool("t4", "Read", file_path="/work/.env")), tool_result("t4", "SECRET=1")],
        ]
        for records in steps:
            write_claude(path, *records, mode="a")
            fast, _full = twin.audit(session)                           # Twin fails the test if the two payloads differ
            verdicts.append({f["contract_id"]: f["verdict"] for f in fast["findings"]})
            twin.check(session, {"tool_name": "Read", "arguments": {"file_path": "/work/.env"}})
            twin.check(session, {"tool_name": "Read", "arguments": {"file_path": "/work/a.py"}})
        assert verdicts == [{"no-dotenv": "clean", "reads-only": "clean"},
                            {"no-dotenv": "violation", "reads-only": "clean"},
                            {"no-dotenv": "violation", "reads-only": "violation"},
                            {"no-dotenv": "violation", "reads-only": "violation"}]
        finding = next(f for f in fast["findings"] if f["contract_id"] == "no-dotenv")
        assert finding["reason"].startswith("forbidden path .env") and "(?:" not in finding["basis"]
        assert [r["mode"] for r in twin.reports if r].count("delta") >= 3, twin.reports
        check = twin.fast.check_action(session, {"tool_name": "Read", "arguments": {"file_path": "/work/.env"}},
                                       provider="claude", path=str(path))
        assert check["verdict"] == "violation" and check["findings"][0]["contract_id"] in ("no-dotenv", "reads-only")
    finally:
        twin.close()


def test_agy_history_audit_is_the_same_incrementally_and_in_full(tmp_path):
    from test_agy_parser_v6 import append, generic, planner
    root = tmp_path / "app"
    twin = Twin(tmp_path, "agy", agy_app_data_dirs=[root])
    try:
        path = make_agy(root, [agy_user(0, "never read the secrets in .env"), planner(1, "", [("run_command", {"CommandLine": "ls"})]),
                               generic(2, "ok")])
        twin.upsert([{**GUARD, "provider": "agy"}])
        twin.fast.settings.audit_incremental_min_events = 1
        assert twin.audit(AGY_SID)[0]["verdict"] == "clean"
        step, verdicts = 3, []
        for name, args in (("view_file", {"AbsolutePath": "/work/.env.example"}), ("view_file", {"AbsolutePath": "/work/.env"}),
                           ("run_command", {"CommandLine": "ls"})):
            append(path, planner(step, "", [(name, args)]), generic(step + 1, "done"))
            step += 2
            verdicts.append(twin.audit(AGY_SID)[0]["verdict"])
            twin.check(AGY_SID, {"tool_name": "view_file", "arguments": {"AbsolutePath": "/work/.env"}})
        assert verdicts == ["clean", "violation", "violation"]
    finally:
        twin.close()


def test_codex_history_audit_is_the_same_incrementally_and_in_full(tmp_path, monkeypatch):
    for name in ("SELF_DIRECT_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / f"rollout-2026-09-09T01-00-00-{CODEX_SID}.jsonl"
    write_codex(path, [{"type": "session_meta", "payload": {"id": CODEX_SID}}])
    twin = Twin(tmp_path, "codex", codex_sessions_dir=root)
    try:
        twin.upsert([{**GUARD, "provider": "codex"}])
        twin.fast.settings.audit_incremental_min_events = 1
        patch = "*** Begin Patch\n*** Update File: .env\n@@\n-a\n+b\n*** End Patch"
        verdicts = []
        for record in ({"type": "response_item", "payload": {"type": "function_call", "name": "shell", "call_id": "a",
                                                              "arguments": {"command": ["bash", "-lc", "ls"]}}},
                       {"type": "response_item", "payload": {"type": "custom_tool_call", "name": "apply_patch",
                                                              "call_id": "b", "input": patch}},
                       {"type": "response_item", "payload": {"type": "function_call", "name": "shell", "call_id": "c",
                                                              "arguments": {"command": ["cat", ".env"]}}}):
            write_codex(path, [record], "a")
            verdicts.append(twin.audit(CODEX_SID)[0]["verdict"])
        assert verdicts == ["clean", "violation", "violation"]
    finally:
        twin.close()


# --------------------------------------------------------------------------- weakening through the real engine

def test_narrowing_a_stored_guard_needs_a_user_anchor_and_widening_does_not(claude):
    claude.upsert(GUARD)
    claude.append(human("please let me read .env.local now"))
    claude.sync()
    anchor = next(c.chunk_id for c in claude.chunks() if c.meta.get("role") == "user" and "please let me" in c.text)
    narrower = {**GUARD, "path_guard": {"paths": [".env"], "except": [".env.example", ".env.local"], "modes": "any"}}
    held = claude.upsert(narrower)
    assert held["pending"][0]["reasons"] == ["path_guard.except added"] and "NOT APPLIED" in held["nudge"].upper()
    assert claude.engine.list_contracts()["contracts"][0]["path_guard"]["except"] == [".env.example"]      # still in force
    applied = claude.upsert({**narrower, "source_event_id": anchor})
    assert "pending" not in applied and applied["contracts"][0]["revision"] == 2
    assert claude.engine.list_contracts()["contracts"][0]["path_guard"]["except"] == [".env.example", ".env.local"]
    wider = {**GUARD, "path_guard": {"paths": [".env", "*.pem"], "except": [], "modes": "any"}}
    assert "pending" not in claude.upsert(wider)                                  # tightening needs no anchor
    assert claude.engine.list_contracts()["contracts"][0]["path_guard"]["paths"] == [".env", "*.pem"]


# ------------------------------------------------------------- bundles: export, adopt, and a changed tool map

def test_a_typed_contract_travels_in_a_bundle_and_is_enforced_after_adoption(world, monkeypatch):
    from test_portable_contracts import BUNDLE, SID_A, SID_B, write_bundle
    from self_directing_mcp import contract_bundles as cb
    from semantic_support import bash, tool_result  # noqa: F401  (the Device helpers use them)
    owner, adopter = world.device("a"), world.device("b")
    project = world.project("repo")
    owner.engine.upsert_contracts([GUARD])
    written = owner.engine.export_contracts(BUNDLE, project_dir=str(project))
    assert written["written"] is True and written["exported"] == ["no-dotenv"]
    on_disk = json.loads((project / BUNDLE).read_text(encoding="utf-8"))["contracts"][0]
    assert on_disk["path_guard"] == GUARD["path_guard"] and "regex" not in on_disk and "tool_call:" not in json.dumps(on_disk)
    candidate = adopter.engine.list_contract_candidates(project_dir=str(project))["candidates"][0]
    (listed,) = candidate["contracts"]
    assert listed["status"] == "new" and listed["path_guard"] == GUARD["path_guard"] and "regex" not in listed
    anchor = adopter.say(SID_B, "yes, adopt the team contracts")
    result = adopter.engine.adopt_contracts(candidate["bundle_id"], anchor, project_dir=str(project))
    assert result["ok"] and result["counts"]["adopted"] == 1
    stored = adopter.contracts()["no-dotenv"]
    assert stored["path_guard"] == GUARD["path_guard"] and "regex" not in stored and stored["origin"]["adopted_by_event_id"] == anchor
    for tool, arguments, verdict in (("Read", {"file_path": "/work/.env"}, "violation"), ("Bash", {"command": "cat .env"}, "violation"),
                                     ("Read", {"file_path": "/work/.env.example"}, "clean")):
        checked = adopter.engine.check_action(SID_B, {"tool_name": tool, "arguments": arguments}, provider="claude",
                                              path=str(adopter.path(SID_B)))
        assert checked["verdict"] == verdict, (tool, checked)
    # a later tool map changes the derived regex and nothing that identifies the contract: still "adopted", same pin
    extra = contract_paths.ToolEntry("omp", "cat_file", contract_paths.FILE_READ, ("filename",), False, "test")
    monkeypatch.setattr(contract_paths, "TOOL_MAP_VERSION", 2)
    monkeypatch.setattr(contract_paths, "TOOL_ENTRIES", contract_paths.TOOL_ENTRIES + (extra,))
    contract_paths._lower.cache_clear()
    try:
        again = adopter.engine.list_contract_candidates(project_dir=str(project))["candidates"][0]
        assert again["bundle_id"] == candidate["bundle_id"] and again["contracts"][0]["status"] == "adopted"
    finally:
        monkeypatch.undo()
        contract_paths._lower.cache_clear()
    # a candidate that narrows the adopted guard is a weakening; replacing it needs replace_ids and is reported as one
    narrower = {"id": "no-dotenv", "type": "must_not", "scope": "tool_call", "severity": "high", "description": GUARD["description"],
                "path_guard": {"paths": [".env"], "except": [".env.example", ".env.local"], "modes": "read"}}
    write_bundle(project, narrower)
    changed = adopter.engine.list_contract_candidates(project_dir=str(project))["candidates"][0]
    assert changed["contracts"][0]["status"] == "changed_since_adoption"
    assert {item["field"] for item in changed["contracts"][0]["differences"]} == {"path_guard"}
    refused = adopter.engine.adopt_contracts(changed["bundle_id"], anchor, project_dir=str(project))
    assert refused["counts"]["conflicts"] == 1 and adopter.contracts()["no-dotenv"]["path_guard"]["modes"] == "any"
    replaced = adopter.engine.adopt_contracts(changed["bundle_id"], anchor, replace_ids=["no-dotenv"], project_dir=str(project))
    assert replaced["replaced"][0]["weakening"] is True
    assert set(replaced["replaced"][0]["reasons"]) == {"path_guard.except added", "path_guard.modes narrowed (any -> read)"}


# ------------------------------------------------------------- saved monitor state, and a file that cannot be loaded

def test_the_monitor_fingerprint_of_a_typed_contract_follows_the_tool_map_and_a_regex_contract_is_unchanged(monkeypatch):
    from self_directing_mcp.audit import incremental
    typed = ContractRule.model_validate(READ_ONLY)            # mode read: the later tool map adds a read tool to its regex
    plain = ContractRule(id="p", type="must_not", scope="tool_call", regex=r"rm\s+-rf")

    def prints():
        return [incremental.rule_fingerprint(r, "claude", 3, "digest") for r in (typed, plain)]

    before = prints()
    assert before == prints()
    extra = contract_paths.ToolEntry("omp", "cat_file", contract_paths.FILE_READ, ("filename",), False, "test")
    monkeypatch.setattr(contract_paths, "TOOL_MAP_VERSION", 2)
    monkeypatch.setattr(contract_paths, "TOOL_ENTRIES", contract_paths.TOOL_ENTRIES + (extra,))
    contract_paths._lower.cache_clear()
    try:
        later_rule = ContractRule.model_validate(READ_ONLY)
        assert later_rule.model_dump(mode="json") == typed.model_dump(mode="json")      # the stored form did not move
        assert incremental.rule_fingerprint(later_rule, "claude", 3, "digest") != before[0]   # the monitor must be rebuilt
        assert incremental.rule_fingerprint(plain, "claude", 3, "digest") == before[1]        # a regex contract keeps its state
    finally:
        monkeypatch.undo()
        contract_paths._lower.cache_clear()


def test_a_contracts_file_with_a_bad_typed_contract_is_reported_not_silently_dropped(claude):
    claude.upsert(GUARD)
    path = claude.engine.settings.resolve_contracts_path()
    document = json.loads(path.read_text(encoding="utf-8"))
    document["contracts"][0]["path_guard"]["modes"] = "write"                  # a file edited by hand, or written by a later version
    path.write_text(json.dumps(document), encoding="utf-8")
    audit = claude.audit()
    assert "contracts_corrupt" in audit["coverage"]["issues"] and audit["coverage"]["complete"] is False
    with pytest.raises(Exception, match="contracts or checklist unavailable"):
        from self_directing_mcp.hook_policy import read_obligations
        read_obligations(claude.engine, "claude", CLAUDE_SID, str(claude.path))
