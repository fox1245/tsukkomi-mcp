"""Regressions for the findings of the scripted evaluation harness: contract regex time limits, Codex hook noise
for actions no contract covers, credential names the masker missed, and unverifiable high-risk contracts."""
from __future__ import annotations

import time

import pytest

from self_directing_mcp import codex_hooks
from self_directing_mcp.audit import patterns
from self_directing_mcp.audit.runner import REGEX_TIMEOUT
from self_directing_mcp.retrieve.checks import regex_search
from self_directing_mcp.security.mask import mask_secrets
from test_reliability import SID, call, message, result, write, session  # noqa: F401

PATHOLOGICAL = r"(a|a)+$"                      # exponential even for the regex engine's own safeguards
POISON = "a" * 30 + "!"


@pytest.fixture
def tight(monkeypatch):
    monkeypatch.setattr(patterns, "SEARCH_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(patterns, "CONTRACT_BUDGET_SEC", 0.3)


def test_a_pathological_contract_regex_is_unknown_and_other_contracts_still_judge(session, tight):
    engine, path = session
    write(path, [message(POISON), call("rm -rf build", "c1"), result("c1")], "a")
    engine.upsert_contracts([{"id": "slow", "type": "must_not", "scope": "any", "regex": PATHOLOGICAL},
                             {"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"}])
    started = time.monotonic()
    audit = engine.audit_session(SID)
    assert time.monotonic() - started < 5
    by_id = {f["contract_id"]: f for f in audit["findings"]}
    assert by_id["slow"]["verdict"] == "unknown" and by_id["slow"]["reason"] == REGEX_TIMEOUT
    assert by_id["no-rm"]["verdict"] == "violation"
    assert audit["verdict"] == "violation"


def test_check_action_with_a_pathological_regex_is_unknown_not_clean(session, tight):
    engine, path = session
    write(path, [message("hello")], "a")
    engine.upsert_contracts([{"id": "slow", "type": "must_not", "scope": "tool_call", "regex": PATHOLOGICAL}])
    started = time.monotonic()
    audit = engine.check_action(SID, {"tool_name": "shell", "arguments": {"command": POISON}})
    assert time.monotonic() - started < 5
    assert audit["verdict"] == "unknown"
    assert [f["reason"] for f in audit["findings"]] == [REGEX_TIMEOUT]


def test_a_slow_when_condition_is_unknown(session, tight):
    engine, path = session
    write(path, [call(POISON, "c1"), result("c1")], "a")
    engine.upsert_contracts([{"id": "gated", "type": "must", "scope": "message", "regex": "VERIFIED",
                              "when": {"tool_call_regex": PATHOLOGICAL}}])
    finding = engine.audit_session(SID)["findings"][0]
    assert finding["verdict"] == "unknown" and finding["reason"] == REGEX_TIMEOUT


def test_the_contract_budget_bounds_a_pattern_that_is_slow_on_every_event(session, monkeypatch):
    engine, path = session
    write(path, [message(POISON) for _ in range(40)], "a")
    engine.upsert_contracts([{"id": "slow", "type": "must_not", "scope": "any", "regex": PATHOLOGICAL}])
    monkeypatch.setattr(patterns, "SEARCH_TIMEOUT_SEC", 0.2)       # each event alone stays under this ...
    monkeypatch.setattr(patterns, "CONTRACT_BUDGET_SEC", 0.5)      # ... but all of them together do not
    started = time.monotonic()
    finding = engine.audit_session(SID)["findings"][0]
    assert time.monotonic() - started < 3
    assert finding["verdict"] == "unknown"


def test_regex_search_reports_a_timeout_instead_of_partial_results(session, tight):
    engine, path = session
    write(path, [message(POISON)], "a")
    engine.sync_session(session_id=SID, embed=False)
    with pytest.raises(ValueError, match="regex_timeout"):
        regex_search(engine.store, PATHOLOGICAL, session_id=SID)


def test_patterns_keep_re_syntax_and_semantics():
    with pytest.raises(Exception):
        patterns.compile_contract("(unclosed")
    pattern = patterns.compile_contract(r"^git\s+push\b.*--force")
    assert pattern.search("x\nGIT push origin --force") is not None      # IGNORECASE | MULTILINE as before
    assert pattern.search("git pushed --force") is None


def test_codex_pretooluse_is_silent_when_no_contract_applies_to_the_action(session):
    engine, path = session
    write(path, [message("hello")], "a")
    engine.upsert_contracts([{"id": "verify", "type": "must", "scope": "tool_call", "regex": "pytest",
                              "auto_continue": True}])
    out = codex_hooks.handle_hook(engine, "PreToolUse", SID, str(path), "exec_command", {"cmd": "git status"})
    assert out == {}


def test_codex_pretooluse_still_reports_an_applicable_violation(session):
    engine, path = session
    write(path, [message("hello")], "a")
    engine.upsert_contracts([{"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"}])
    out = codex_hooks.handle_hook(engine, "PreToolUse", SID, str(path), "exec_command", {"cmd": "rm -rf build"})
    assert "violation" in out["hookSpecificOutput"]["additionalContext"]


@pytest.mark.parametrize("name", ["DB_PASS", "DB_PWD", "MYSQL_PWD", "PASSPHRASE", "SESSION_KEY", "JWT_KEY", "HMAC_KEY",
                                  "AUTH_KEY", "STRIPE_KEY", "ACCESS_KEY_ID", "INTERNAL_SALT", "SALT", "PEPPER",
                                  "REDIS_AUTH", "NPM_AUTH", "BASIC_AUTH"])
@pytest.mark.parametrize("form", ["{n}={v}", "export {n}={v}", '{n}: "{v}"', '{{"{n}": "{v}"}}'])
def test_common_credential_variable_names_are_masked(name, form):
    value = "Zq8" + "VnTb7LmX9Kd2PaR5WcY8H"
    masked = mask_secrets(form.format(n=name, v=value))
    assert value not in masked and name in masked


def test_the_working_directory_variable_stays_readable():
    assert mask_secrets("PWD=/srv/app") == "PWD=/srv/app"


def test_an_unverifiable_high_risk_contract_is_rejected(session):
    engine, _path = session
    with pytest.raises(ValueError, match="high-risk must_not needs a regex"):
        engine.upsert_contracts([{"id": "careful", "type": "must_not", "description": "be careful", "risk": "high"}])
    engine.upsert_contracts([{"id": "gated", "type": "must", "description": "review the plan", "risk": "high",
                              "when": {"tool_call_regex": "terraform apply"}}])
    engine.upsert_contracts([{"id": "rm", "type": "must_not", "regex": r"rm\s+-rf", "risk": "high"}])


def test_an_invalid_exception_regex_is_rejected_at_upsert(session):
    engine, _path = session
    with pytest.raises(Exception):
        engine.upsert_contracts([{"id": "x", "type": "must_not", "regex": "deploy", "exception_regex": "(bad"}])
