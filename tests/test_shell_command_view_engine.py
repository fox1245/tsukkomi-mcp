"""The command view reaches every evaluator: a proposed call, the audit of the history, and the hook pre-checks.

Synthetic data only. A contract about commands is judged on the command view in the full evaluator and in the
incremental one alike (the shadow comparison of the test suite would report a difference as a mismatch), and the
stdlib pre-check of a failing hook agrees with them.
"""
from __future__ import annotations

from self_directing_mcp import claude_hooks

from test_claude_hooks import (  # noqa: F401
    SID, _raising, assistant, env, human, main_file, permission, run, tool_result, use, write,
)

GATED = "gh pr create"
HEREDOC = f"cat > notes.md <<'EOF'\nthen run {GATED}\nEOF"
CONTRACT = {"id": "no-pr", "type": "must_not", "provider": "claude", "regex": GATED, "tools": ["Bash"],
            "risk": "high"}


def _proposed(engine, path, command, tool="Bash"):
    return engine.check_action(SID, {"tool_name": tool, "arguments": {"command": command}}, provider="claude",
                               path=str(path))["verdict"]


def test_a_proposed_command_is_judged_on_its_command_view(env):
    engine, path = env
    engine.upsert_contracts([CONTRACT])
    assert _proposed(engine, path, GATED) == "violation"
    assert _proposed(engine, path, f"bash -c '{GATED}'") == "violation"
    assert _proposed(engine, path, HEREDOC) == "clean"
    assert _proposed(engine, path, f"grep -E '{GATED}' log") == "clean"


def test_a_contract_that_says_target_text_sees_the_whole_text(env):
    engine, path = env
    engine.upsert_contracts([dict(CONTRACT, id="text-contract", target="text")])
    assert _proposed(engine, path, HEREDOC) == "violation"


def test_a_contract_without_tools_is_matched_against_the_whole_text_as_before(env):
    engine, path = env
    engine.upsert_contracts([{"id": "any-tool", "type": "must_not", "provider": "claude", "scope": "tool_call",
                              "regex": GATED}])
    assert _proposed(engine, path, HEREDOC) == "violation"


def test_a_contract_without_tools_that_asks_for_the_command_view_gets_it(env):
    engine, path = env
    engine.upsert_contracts([{"id": "any-tool", "type": "must_not", "provider": "claude", "scope": "tool_call",
                              "regex": GATED, "target": "command"}])
    assert _proposed(engine, path, HEREDOC) == "clean"
    assert _proposed(engine, path, f"bash -c '{GATED}'") == "violation"


def test_switching_a_stored_contract_to_the_command_view_needs_a_user_anchor(env):
    engine, path = env
    engine.upsert_contracts([{"id": "any-tool", "type": "must_not", "provider": "claude", "scope": "tool_call",
                              "regex": GATED}])
    answer = engine.upsert_contracts([{"id": "any-tool", "type": "must_not", "provider": "claude", "scope": "tool_call",
                                       "regex": GATED, "target": "command"}])
    assert "NOT APPLIED" in str(answer)
    assert _proposed(engine, path, HEREDOC) == "violation"                 # the stored revision is still the one enforced


def test_the_audit_of_the_history_judges_the_same_way_in_the_full_and_the_incremental_evaluator(env):
    engine, path = env
    engine.upsert_contracts([CONTRACT])
    write(path, assistant(use("t1", "Bash", command=HEREDOC)), tool_result("t1", "ok"),
          assistant(use("t2", "Bash", command=f"echo '{GATED}' | cat")), tool_result("t2", "ok"), mode="a")
    clean = engine.audit_session(SID, provider="claude", path=str(path))
    assert clean["verdict"] == "clean", clean["findings"]
    assert not (clean.get("incremental") or {}).get("incremental_mismatch")
    write(path, assistant(use("t3", "Bash", command=f"cd /tmp && {GATED}")), tool_result("t3", "ok"), mode="a")
    hit = engine.audit_session(SID, provider="claude", path=str(path))
    assert hit["verdict"] == "violation"
    assert not (hit.get("incremental") or {}).get("incremental_mismatch")


def test_a_failing_hook_refuses_a_command_and_not_a_text_that_names_it(env, monkeypatch):
    engine, path = env
    engine.upsert_contracts([CONTRACT])
    monkeypatch.setattr(claude_hooks, "handle_pre_tool_use", _raising)
    assert permission(run(engine, path, "PreToolUse", tool_input={"command": f"bash -c '{GATED}'"})) == "deny"
    assert permission(run(engine, path, "PreToolUse", tool_input={"command": HEREDOC})) == "none"


def test_matching_less_text_is_a_weakening_and_matching_more_is_not():
    from self_directing_mcp.audit.contracts import loosening_reasons
    from self_directing_mcp.schemas import ContractRule
    text_contract = ContractRule(id="c", type="must_not", scope="tool_call", regex=GATED)
    command_contract = ContractRule(id="c", type="must_not", scope="tool_call", regex=GATED, target="command")
    reason = "matched against the command view of shell calls instead of the whole text (target)"
    assert reason in loosening_reasons(text_contract, command_contract)                  # whole text -> command view
    assert loosening_reasons(command_contract, text_contract) == []                      # command view -> whole text
    bash = ContractRule(id="c", type="must_not", scope="tool_call", regex=GATED, tools=["Bash"])
    both = ContractRule(id="c", type="must_not", scope="tool_call", regex=GATED, tools=["Bash", "Write"])
    assert reason in loosening_reasons(both, bash)                                       # derived target changes
    assert loosening_reasons(bash, ContractRule(**{**bash.model_dump(), "target": "text"})) == []
    pinned_text = ContractRule(**{**bash.model_dump(), "target": "text"})
    assert reason in loosening_reasons(pinned_text, bash)                                # unset -> derived command view
