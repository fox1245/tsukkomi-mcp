"""B3: a masked exit status is not success evidence.

A ``requires_success`` contract ("commit only after a successful pytest") accepted any matching command whose
recorded result was a success. A host records one status per command line, so ``pytest -q || true``, ``pytest | tee
log`` (no pipefail), ``pytest; exit 0`` and ``pytest --collect-only`` all "succeed" without the tests having
passed. Such a result is no longer success evidence: the finding is ``unknown`` (``success_unverifiable``), never
clean and never a violation. A result recorded as failed is still a confirmed failure, and the command itself is
never blocked. Every case runs through both evaluators (``semantic_support.evaluate`` compares them).
"""
from __future__ import annotations

import json

import pytest

from self_directing_mcp.audit import runner, shellstatus
from self_directing_mcp.audit.patterns import compile_contract
from self_directing_mcp.claude_hooks import run_hook
from semantic_support import (
    SID, Claude, assistant, bash, call, evaluate, finding, human, one, result, tool_result, use, user_turn,
)

GATE = {"id": "tests-first", "type": "must", "scope": "tool_call", "regex": "pytest", "before_regex": "git commit",
        "requires_success": True}
DONE = {"id": "tests-ran", "type": "must", "scope": "tool_call", "regex": "pytest", "requires_success": True}
COMMIT = "git commit -m change"

MASKED = [
    ("pytest -q || true", "or_masks_failure"),
    ("pytest -q || :", "or_masks_failure"),
    ("pytest -q || echo tests failed", "or_masks_failure"),
    ("pytest -q || { echo bad; true; }", "or_masks_failure"),
    ("pytest -q && echo ok || true", "or_masks_failure"),
    ("pytest -q; exit 0", "followed_by_command"),
    ("pytest -q ; true", "followed_by_command"),
    ("pytest -q\necho done", "followed_by_command"),
    ("pytest -q &", "background"),
    ("pytest -q | tee log", "pipeline_without_pipefail"),
    ("pytest -q 2>&1 | tail -5", "pipeline_without_pipefail"),
    ("pytest -q |& tee log", "pipeline_without_pipefail"),
    ("set -o pipefail; set +o pipefail; pytest | tee log", "pipeline_without_pipefail"),
    ("pytest --collect-only", "collect_only"),
    ("python -m pytest tests --co -q", "collect_only"),
    ("(cd tests && pytest -q) || true", "or_masks_failure"),
    ("bash -c 'pytest -q || true'", "or_masks_failure"),
    ("bash -lc \"pytest -q | tee log\"", "pipeline_without_pipefail"),
    ("sudo -n sh -c 'pytest; true'", "followed_by_command"),
]

VERIFIED = [
    "pytest -q",
    "python -m pytest tests -x --cov",
    "cd repo && pytest -q",
    "pytest -q && echo done",
    "pytest -q 2>&1",
    "pytest -q > out.log 2>&1",
    "FOO=1 pytest -q",
    "set +e; pytest -q",                            # `set +e` alone is not a masking tail
    "set -o pipefail; pytest -q | tee log",
    "set -euo pipefail\npytest -q | tee log",
    "bash -o pipefail -c 'pytest -q | tee log'",
    "bash -lc 'set -o pipefail; pytest -q | tee log'",
    "cat input | pytest -",                         # the matched command is the last stage
    "pytest -q || exit 1",                          # a failing branch keeps the failure
    "pytest -q || { echo bad; exit 1; }",
    "pytest -q || false",
    "pytest -q; exit $?",
    "pytest -q ;",
    "pytest -q\n",
    "pytest -q # || true",                          # a comment
    'pytest -q -k "not slow || fast"',              # `||` inside quotes is no operator
    "pytest -q -k 'a | b'",
    "pytest -k 'not co'",
    "pytest -q \\\n  --maxfail=1",
    "echo \"a || true\" && pytest -q",
    "(cd tests && pytest -q)",
    "bash -c 'pytest -q'",
]


# ---------------------------------------------------------------- the lexer and the analysis

def text_of(command, key="command"):
    return "[tool_call:Bash] " + json.dumps({key: command})


@pytest.mark.parametrize("command,why", MASKED)
def test_masking_forms_are_unverifiable(command, why):
    assert shellstatus.unverifiable(text_of(command), compile_contract("pytest")) == why


@pytest.mark.parametrize("command", VERIFIED)
def test_plain_forms_and_quoted_false_friends_stay_verifiable(command):
    assert shellstatus.unverifiable(text_of(command), compile_contract("pytest")) is None


def test_the_matched_command_is_located_not_the_whole_line():
    rx = compile_contract("pytest")
    # A mask that follows an unrelated command is not this command's mask; the last matched command decides.
    assert shellstatus.unverifiable(text_of("echo start || true; pytest -q"), rx) is None
    assert shellstatus.unverifiable(text_of("pytest -q --collect-only; pytest -q"), rx) is None
    assert shellstatus.unverifiable(text_of("pytest -q; pytest -q | tee log"), rx) == "pipeline_without_pipefail"


def test_the_quoted_false_friend_of_the_task_when_the_matched_command_is_echo():
    rx = compile_contract("echo")
    assert shellstatus.unverifiable(text_of('echo "a || true"'), rx) is None
    assert shellstatus.unverifiable(text_of("echo a || true"), rx) == "or_masks_failure"
    assert shellstatus.unverifiable(text_of("echo 'x; exit 0'"), rx) is None


def test_argv_and_codex_style_arguments_are_read_like_a_command_line():
    rx = compile_contract("pytest")
    argv = "[tool_call:shell] " + json.dumps({"command": ["bash", "-lc", "pytest -q || true"], "workdir": "/w"})
    assert shellstatus.unverifiable(argv, rx) == "or_masks_failure"
    plain = "[tool_call:shell] " + json.dumps({"command": ["pytest", "--collect-only"]})
    assert shellstatus.unverifiable(plain, rx) == "collect_only"
    assert shellstatus.unverifiable("[tool_call:shell] " + json.dumps({"command": ["pytest", "-q"]}), rx) is None
    # A plain-text argument (not JSON) is the command line itself.
    assert shellstatus.unverifiable("[tool_call:shell] pytest -q || true", rx) == "or_masks_failure"


def test_a_command_the_host_ran_in_the_background_reports_only_that_it_started():
    rx = compile_contract("pytest")
    started = "[tool_call:Bash] " + json.dumps({"command": "pytest -q", "run_in_background": True})
    assert shellstatus.unverifiable(started, rx) == "background"
    foreground = "[tool_call:Bash] " + json.dumps({"command": "pytest -q", "run_in_background": False})
    assert shellstatus.unverifiable(foreground, rx) is None


def test_an_unlexable_line_and_an_oversized_one_are_handled_conservatively():
    rx = compile_contract("pytest")
    assert shellstatus.unverifiable(text_of("pytest -q 'unterminated || true"), rx) is None   # not a runnable command
    assert shellstatus.unverifiable(text_of("pytest " + "x" * shellstatus.MAX_CHARS), rx) == "too_long_to_analyze"


# ---------------------------------------------------------------- both evaluators

PROVIDERS = ["claude", "codex", "agy", "omp"]


def history(command, outcome, call_id="c1", name="Bash"):
    events = [user_turn(), call(command, call_id, name=name)]
    if outcome is not None:
        events.append(result(call_id, outcome))
    return events


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("command,why", MASKED)
def test_a_masked_success_is_unknown_not_clean_and_not_a_violation(provider, command, why):
    item = one(GATE, history(command, True) + [call(COMMIT, "c2")], provider)
    assert item.verdict == "unknown"
    assert item.reason.startswith("success_unverifiable:") and shellstatus.REASONS[why] in item.reason
    assert not item.reason.startswith("required evidence missing")
    assert [e.detail for e in item.evidence][-1] == "linked tool result (success not attributable)"


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("command", VERIFIED)
def test_an_attributable_success_is_still_clean(provider, command):
    item = one(GATE, history(command, True) + [call(COMMIT, "c2")], provider)
    assert item.verdict == "clean" and item.reason == "required call has linked successful result"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_recorded_failure_stays_a_confirmed_failure_even_when_the_line_masks(provider):
    """The host recorded a failed result: that is a fact whatever the command line looks like."""
    item = one(GATE, history("pytest -q || true", False) + [call(COMMIT, "c2")], provider)
    assert item.verdict == "violation" and item.reason == "required calls completed without success"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_masked_call_with_no_result_is_unresolved_as_before(provider):
    item = one(GATE, history("pytest -q || true", None) + [call(COMMIT, "c2")], provider)
    assert item.verdict == "unknown" and "completion status" in item.reason


@pytest.mark.parametrize("provider", PROVIDERS)
def test_no_matching_call_at_all_is_still_a_missing_prerequisite(provider):
    item = one(GATE, [user_turn(), call("ls", "c1")] + [call(COMMIT, "c2")], provider)
    assert item.verdict == "violation" and item.reason == "required evidence missing before checkpoint"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_the_latest_matching_call_decides_a_newer_masked_run_is_not_a_success(provider):
    events = (history("pytest -q", True, "c1") + history("pytest -q | tee log", True, "c2")[1:]
              + [call(COMMIT, "c3")])
    item = one(GATE, events, provider)
    assert item.verdict == "unknown" and item.reason.startswith("success_unverifiable")
    reverse = history("pytest -q | tee log", True, "c1") + history("pytest -q", True, "c2")[1:] + [call(COMMIT, "c3")]
    assert one(GATE, reverse, provider).verdict == "clean"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_completion_obligation_without_before_regex_follows_the_same_rule(provider):
    assert one(DONE, history("pytest -q || true", True), provider).verdict == "unknown"
    assert one(DONE, history("pytest -q", True), provider).verdict == "clean"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_proposed_action_sees_the_same_prerequisite_judgement(provider):
    from semantic_support import event
    proposed = event("tool_call", "[tool_call:Bash] " + json.dumps({"command": COMMIT}), "assistant",
                     tool_name="Bash", proposed=True)
    masked = one(GATE, history("pytest -q || true", True), provider, proposed=proposed)
    assert masked.verdict == "unknown" and masked.reason.startswith("success_unverifiable")
    assert one(GATE, history("pytest -q", True), provider, proposed=proposed).verdict == "clean"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_rule_that_matches_echo_is_not_fooled_by_a_quoted_operator(provider):
    rule = {**DONE, "id": "said", "regex": "echo"}
    assert one(rule, history('echo "a || true"', True), provider).verdict == "clean"
    assert one(rule, history("echo a || true", True), provider).verdict == "unknown"


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("block", range(4))
def test_random_sessions_of_every_form_are_judged_identically_by_both_evaluators(provider, block):
    import random
    commands = [command for command, _why in MASKED] + VERIFIED + ["ls", "git status", "make test"]
    for seed in range(block * 15, block * 15 + 15):
        rng = random.Random(f"mask/{provider}/{seed}")
        events, numbered = [user_turn()], 0
        for _ in range(rng.randint(2, 9)):
            numbered += 1
            events.append(call(rng.choice(commands), f"k{numbered}"))
            if rng.random() < 0.85:
                events.append(result(f"k{numbered}", rng.choice([True, True, False, None])))
            if rng.random() < 0.2:
                events.append(call(COMMIT, f"g{numbered}"))
        evaluate([GATE, DONE, {**DONE, "id": "any", "regex": "make|ls|git"}], events, provider)


# ---------------------------------------------------------------- end to end: indexed history, the hook never blocks

@pytest.fixture
def session(tmp_path):
    world = Claude(tmp_path)
    yield world
    world.close()


def commit_gate(session, command, *, is_error=False):
    session.write(human("run the tests before you commit"), bash("t1", command), tool_result("t1", "3 passed",
                                                                                           is_error=is_error))
    session.upsert(GATE)
    return session.check("Bash", {"command": COMMIT})


def test_end_to_end_a_masked_pytest_does_not_satisfy_the_commit_gate(session):
    verdict = commit_gate(session, "pytest -q || true")
    item = finding(verdict, "tests-first")
    assert verdict["verdict"] == "unknown" and item["verdict"] == "unknown"
    assert item["reason"].startswith("success_unverifiable")


def test_end_to_end_a_plain_pytest_satisfies_it(session):
    assert finding(commit_gate(session, "pytest -q"), "tests-first")["verdict"] == "clean"


@pytest.mark.parametrize("command,verdict", [("set -o pipefail; pytest -q | tee log", "clean"),
                                             ("pytest -q | tee log", "unknown")])
def test_end_to_end_pipefail_decides_whether_a_pipeline_counts(session, command, verdict):
    assert finding(commit_gate(session, command), "tests-first")["verdict"] == verdict


def test_end_to_end_a_failed_masked_run_is_a_failure(session):
    assert finding(commit_gate(session, "pytest -q || true", is_error=True), "tests-first")["verdict"] == "violation"


def test_the_hook_does_not_block_the_command_on_an_unverifiable_prerequisite(session):
    session.write(human("run the tests before you commit"), bash("t1", "pytest -q | tee log"), tool_result("t1"))
    session.upsert(GATE)
    payload = {"session_id": SID, "transcript_path": str(session.path), "cwd": "/synthetic/project",
               "hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": COMMIT}}
    response = run_hook("PreToolUse", payload, engine=session.engine)
    assert response.get("hookSpecificOutput", {}).get("permissionDecision", "none") == "none", response
    assert "success_unverifiable" in response["systemMessage"]      # visible to the user, but only a warning
