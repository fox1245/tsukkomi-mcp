"""B6(b): conservative coverage under format drift.

A host can start writing a record this parser does not know. Such a record is kept visible and makes coverage
incomplete (``unsupported_events``); it may hold the forbidden call or the commit an order rule waits for. So, for a
history audit:

* an absence proof (``must_not`` with no match, a ``before_regex`` rule with no triggering action) is ``unknown``, and
  so is the overall verdict;
* an obligation that rests on missing evidence is ``unknown``, and a proposed action's prerequisite cannot be
  established by incomplete history (``before_regex`` / ``requires_success``);
* a violation confirmed from the evidence that was parsed stays a violation, in the findings and overall;
* the same session without the unknown record gives the ordinary verdicts.
"""
from __future__ import annotations

import json

import pytest

from semantic_support import Claude, bash, env_record, finding, human, tool_result
from test_reliability import SID as CODEX_SID, call as codex_call, message, result as codex_result, session as codex_session, write as codex_write
from test_agy_parser_v6 import SID as AGY_SID, append, env as agy_env, finding as agy_finding, hook, make, planner, user as agy_user  # noqa: F401
from test_agy_parser_v7 import rec

NO_RM = {"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"}
NO_LS = {"id": "no-ls", "type": "must_not", "scope": "tool_call", "regex": r"\bls\b"}
GATE = {"id": "gate", "type": "must", "scope": "tool_call", "regex": "pytest", "before_regex": "git commit",
        "requires_success": True}
UNMET = {"id": "unmet", "type": "must", "scope": "tool_call", "regex": r"\bmake\b"}
ACTION = {"tool_name": "Bash", "arguments": {"command": "git commit -m change"}}
UNSUPPORTED_ABSENCE = "incomplete evidence cannot prove the absence of a forbidden match"


def unknown_record(sid):
    return env_record("future-record-type", sid=sid, payload="a record this parser has never seen")


@pytest.fixture
def claude(tmp_path):
    world = Claude(tmp_path)
    yield world
    world.close()


def drifted(world, *, drift=True, ran_ls=False, tests_pass=True):
    records = [human("run the tests, then commit"), bash("t1", "pytest -q"), tool_result("t1", "3 passed",
                                                                                        is_error=not tests_pass)]
    if drift:
        records.append(unknown_record("aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"))
    if ran_ls:
        records += [bash("t2", "ls"), tool_result("t2", "a b")]
    world.write(*records)
    world.upsert(NO_RM, NO_LS, GATE, UNMET)


def verdicts(audit):
    return {f["contract_id"]: f["verdict"] for f in audit["findings"]}


# ---------------------------------------------------------------- Claude Code

def test_control_without_the_unknown_record_the_verdicts_are_ordinary(claude):
    drifted(claude, drift=False)
    audit = claude.audit()
    assert audit["coverage"]["complete"] is True
    assert verdicts(audit) == {"no-rm": "clean", "no-ls": "clean", "gate": "clean", "unmet": "violation"}


def test_an_unknown_record_makes_every_absence_proof_unknown(claude):
    drifted(claude)
    audit = claude.audit()
    assert audit["coverage"]["complete"] is False and audit["coverage"]["issues"] == ["unsupported_events"]
    found = verdicts(audit)
    assert found["no-rm"] == "unknown" and found["no-ls"] == "unknown"        # nothing matched in what was parsed
    assert found["gate"] == "unknown"                                          # "no commit yet" is an absence proof
    assert found["unmet"] == "unknown"                                         # a missing obligation is not proven
    assert audit["verdict"] == "unknown"
    assert finding(audit, "no-rm")["reason"] == UNSUPPORTED_ABSENCE


def test_a_confirmed_violation_elsewhere_stays_a_violation(claude):
    drifted(claude, ran_ls=True)
    audit = claude.audit()
    assert verdicts(audit)["no-ls"] == "violation" and audit["verdict"] == "violation"
    assert verdicts(audit)["no-rm"] == "unknown"                              # ...without vouching for the others


def test_a_proposed_action_cannot_establish_its_prerequisite_from_incomplete_history(claude):
    drifted(claude)
    checked = claude.check("Bash", ACTION["arguments"])
    gate = finding(checked, "gate")
    assert gate["verdict"] == "unknown" and "incomplete" in gate["reason"]
    assert checked["verdict"] == "unknown" and checked["coverage"]["complete"] is False


def test_the_prerequisite_is_established_when_the_history_is_complete(claude):
    drifted(claude, drift=False)
    assert finding(claude.check("Bash", ACTION["arguments"]), "gate")["verdict"] == "clean"


def test_a_failed_prerequisite_is_unknown_not_a_violation_while_history_is_incomplete(claude):
    """The failed run may have been repeated successfully in the record this parser cannot read."""
    drifted(claude, tests_pass=False)
    assert finding(claude.check("Bash", ACTION["arguments"]), "gate")["verdict"] in ("unknown", "violation")
    audit = claude.audit()
    assert verdicts(audit)["gate"] == "unknown"                                # no commit yet: an unproven absence


def test_a_proposed_forbidden_action_is_a_violation_whatever_the_history_coverage(claude):
    drifted(claude)
    checked = claude.check("Bash", {"command": "rm -rf /data"})
    assert finding(checked, "no-rm")["verdict"] == "violation" and checked["verdict"] == "violation"


def test_a_benign_proposed_action_is_not_called_clean_overall_under_drift(claude):
    drifted(claude)
    checked = claude.check("Bash", {"command": "echo hello"})
    assert checked["verdict"] == "unknown" and checked["coverage"]["issues"] == ["unsupported_events"]


def test_a_parse_error_is_incomplete_coverage_like_an_unsupported_record(claude):
    drifted(claude, drift=False)
    with claude.path.open("a", encoding="utf-8") as stream:
        stream.write("{this is not json}\n")
    audit = claude.audit()
    assert audit["coverage"]["complete"] is False and verdicts(audit)["no-rm"] == "unknown"
    assert audit["verdict"] == "unknown"


def test_pure_growth_is_still_judged_as_of_its_captured_offset(claude):
    """Not drift: unread appended bytes are the documented ``judged_prefix`` case and stay ``clean`` as of the offset."""
    drifted(claude, drift=False)
    claude.sync()
    with claude.path.open("ab") as stream:
        stream.write(b'{"type": "assistant", "uuid"')           # a record the host has not finished writing
    audit = claude.audit()
    assert audit["coverage"]["judged_prefix"] is True and verdicts(audit)["no-rm"] == "clean"


# ---------------------------------------------------------------- Codex

@pytest.fixture
def codex(codex_session):
    engine, path = codex_session
    try:
        yield engine, path
    finally:
        engine.close()


def codex_drifted(codex, *, drift):
    engine, path = codex
    events = [message("run the tests, then commit", "user"), codex_call("pytest -q", "a"), codex_result("a")]
    if drift:
        events.append({"type": "future_event", "payload": {"anything": "at all"}})
    codex_write(path, events, "a")
    engine.upsert_contracts([NO_RM, NO_LS, GATE, UNMET])


def test_codex_drift_makes_absences_unknown_and_keeps_a_confirmed_violation(codex):
    engine, path = codex
    codex_drifted(codex, drift=False)
    assert verdicts(engine.audit_session(CODEX_SID)) == {"no-rm": "clean", "no-ls": "clean", "gate": "clean",
                                                          "unmet": "violation"}
    codex_write(path, [{"type": "future_event", "payload": {"anything": "at all"}}], "a")
    audit = engine.audit_session(CODEX_SID)
    assert audit["coverage"]["complete"] is False
    assert verdicts(audit) == {"no-rm": "unknown", "no-ls": "unknown", "gate": "unknown", "unmet": "unknown"}
    codex_write(path, [codex_call("ls", "b"), codex_result("b")], "a")
    violated = engine.audit_session(CODEX_SID)
    assert verdicts(violated)["no-ls"] == "violation" and violated["verdict"] == "violation"
    checked = engine.check_action(CODEX_SID, {"tool_name": "shell", "arguments": {"command": "git commit -m x"}})
    assert finding(checked, "gate")["verdict"] == "unknown"


# ---------------------------------------------------------------- AGY

def test_agy_drift_keeps_the_same_rules(agy_env):
    engine, root = agy_env
    engine.upsert_contracts([{**NO_RM, "provider": "agy"}, {**UNMET, "provider": "agy"}])
    path = make(root, [agy_user(0, "go"), planner(1, "", [("run_command", {"CommandLine": "ls"})])])
    clean = agy_finding(engine.audit_session(AGY_SID, provider="agy", path=str(path)), "no-rm")
    assert clean["verdict"] in ("clean", "unknown")           # the control: whatever AGY's own evidence rules say
    append(path, rec(2, "FUTURE_TOOL", "MODEL", "DONE", content="a step type this parser does not know"))
    audit = engine.audit_session(AGY_SID, provider="agy", path=str(path))
    assert audit["coverage"]["complete"] is False
    assert agy_finding(audit, "no-rm")["verdict"] == "unknown" and agy_finding(audit, "unmet")["verdict"] == "unknown"
    assert audit["verdict"] == "unknown"
    append(path, planner(3, "", [("run_command", {"CommandLine": "rm -rf /data"})]))
    attempt = engine.audit_session(AGY_SID, provider="agy", path=str(path))
    assert agy_finding(attempt, "no-rm")["verdict"] == "violation" and attempt["verdict"] == "violation"
