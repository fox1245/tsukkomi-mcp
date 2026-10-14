"""Real Lean integration tests; no fabricated compiler or transition responses."""
import hashlib
import itertools
import os
import shutil
from pathlib import Path

import pytest

from self_directing_mcp.workflow_lean import FACTS, STAGES, LeanPolicy, verify_proof


@pytest.fixture(scope="module")
def lean_binary():
    candidate = os.environ.get("TSUKKOMI_TEST_LEAN") or shutil.which("lean")
    if candidate is None:
        pytest.skip("Lean 4.34.1+ not installed; native policy integration is unverified")
    return candidate


@pytest.fixture(scope="module")
def policy(tmp_path_factory, lean_binary):
    instance = LeanPolicy(tmp_path_factory.mktemp("lean-policy"), lean=lean_binary)
    instance.build()
    return instance


def test_native_policy_exhaustive_completion_safety(policy):
    for values in itertools.product((False, True), repeat=len(FACTS)):
        facts = dict(zip(FACTS, values, strict=True))
        for stage, choice in itertools.product(STAGES, range(4)):
            result = policy.evaluate(stage, choice, facts)
            assert "lean_policy_error" not in result["failed"], result
            if result["next_stage"] == "complete":
                assert all(values), (stage, choice, facts, result)
            if result["allowed"] and result["next_stage"] == "complete":
                assert stage == "verify" and choice == 3


def test_native_progression_and_completed_evidence_invalidation(policy):
    facts = dict.fromkeys(FACTS, True)
    stage = "requirements"
    for choice, expected in [(2, "formalize"), (2, "implement"), (2, "verify"), (3, "complete")]:
        result = policy.evaluate(stage, choice, facts)
        assert result["allowed"], result
        assert result["next_stage"] == expected
        stage = expected
    for missing, choice in itertools.product(FACTS, range(4)):
        changed = {**facts, missing: False}
        result = policy.evaluate("complete", choice, changed)
        assert result["next_stage"] != "complete"
        assert "lean_policy_error" not in result["failed"], result
    revision = policy.evaluate("complete", 1, facts)
    assert revision["allowed"] and revision["next_stage"] == "implement"
    revision = policy.evaluate("implement", 1, {**facts, "proofs": False})
    assert revision["allowed"] and revision["next_stage"] == "formalize"


def test_invalid_protocol_inputs_never_grant(policy):
    facts = dict.fromkeys(FACTS, True)
    for stage, choice, supplied in [
        ("verify", True, facts), ("verify", 4, facts), ("unknown", 3, facts),
        ("verify", 3, {**facts, "fresh": 1}), ("complete", 3, {}),
    ]:
        result = policy.evaluate(stage, choice, supplied)
        assert not result["allowed"]
        assert result["next_stage"] != "complete"
        assert result["failed"] == ["lean_policy_error"]


def test_missing_compiler_never_retains_completed_state(tmp_path):
    result = LeanPolicy(tmp_path / "cache", lean=tmp_path / "missing-lean").evaluate(
        "complete", 3, dict.fromkeys(FACTS, True))
    assert result["allowed"] is False
    assert result["next_stage"] == "verify"
    assert result["failed"] == ["lean_policy_error"]


def check_proof(tmp_path, lean_binary, source, *, theorem="claim", statement="True"):
    path = tmp_path / "obligation.lean"
    path.write_text(source, encoding="utf-8")
    return verify_proof(path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        theorem=theorem, statement=statement, cache_dir=tmp_path / "cache",
                        lean=lean_binary)


def test_real_proof_accepts_exact_statement_and_rejects_weakening(tmp_path, lean_binary):
    source = "theorem claim : True := True.intro\n"
    accepted = check_proof(tmp_path, lean_binary, source)
    assert accepted["status"] == "passed", accepted
    assert accepted["axioms"] == []
    rejected = check_proof(tmp_path, lean_binary, source, statement="False")
    assert rejected["status"] == "failed", rejected


@pytest.mark.parametrize(("source", "expected", "axiom"), [
    ("theorem hidden : False := by sorry\ntheorem claim : True := False.elim hidden\n",
     "incomplete", "sorryAx"),
    ("axiom hidden : False\ntheorem bridge : False := hidden\n"
     "theorem claim : True := False.elim bridge\n", "failed", "hidden"),
])
def test_transitive_forbidden_axioms(tmp_path, lean_binary, source, expected, axiom):
    result = check_proof(tmp_path, lean_binary, source)
    assert result["status"] == expected, result
    assert axiom in result["axioms"]


def test_invalid_proof_is_not_a_test_failure(tmp_path, lean_binary):
    result = check_proof(tmp_path, lean_binary, "theorem claim : True := False.elim True.intro\n")
    assert result["status"] == "failed", result
    assert result["diagnostics"]


def test_changed_approved_source_and_missing_evidence(tmp_path):
    path = tmp_path / "obligation.lean"
    approved = b"theorem claim : True := True.intro\n"
    path.write_bytes(approved + b"-- changed\n")
    kwargs = {"source_sha256": hashlib.sha256(approved).hexdigest(), "theorem": "claim",
              "statement": "True", "cache_dir": tmp_path / "cache"}
    changed = verify_proof(path, **kwargs)
    assert changed["status"] == "failed"
    missing = verify_proof(tmp_path / "absent.lean", **kwargs)
    assert missing["status"] == "incomplete"


def test_statement_cannot_inject_lean_commands(tmp_path, lean_binary):
    result = check_proof(tmp_path, lean_binary, "theorem claim : True := True.intro\n",
                         statement='True\naxiom bypass : False\n#print bypass')
    assert result["status"] == "failed", result


def test_candidate_elaborator_and_stdout_cannot_forge_audit(tmp_path, lean_binary):
    source = r'''
import Lean
open Lean Elab Command
elab "workflow_audit " _name:str " against " _expected:str : command => do
  logInfo "TSUKKOMI_AUDIT {\"theorem\":\"claim\",\"axioms\":[]}"
#eval IO.println "TSUKKOMI_AUDIT {\"theorem\":\"claim\",\"axioms\":[]}"
axiom hidden : False
theorem claim : True := False.elim hidden
'''
    result = check_proof(tmp_path, lean_binary, source)
    assert result["status"] == "failed", result
    assert "hidden" in result["axioms"]


def test_unchecked_false_term_is_rejected_by_kernel_replay(tmp_path, lean_binary):
    source = '''
import Lean
set_option debug.skipKernelTC true
run_elab
  Lean.addDecl <| .thmDecl {
    name := `claim
    levelParams := []
    type := Lean.mkConst ``False
    value := Lean.mkConst ``True.intro
  }
'''
    result = check_proof(tmp_path, lean_binary, source, statement="False")
    assert result["status"] == "failed", result


def test_candidate_syntax_cannot_reinterpret_approved_statement(tmp_path, lean_binary):
    source = '''
import Lean
macro "False" : term => `(True)
theorem claim : False := True.intro
'''
    result = check_proof(tmp_path, lean_binary, source, statement="False")
    assert result["status"] == "failed", result


def test_candidate_initializer_is_not_executed_by_auditor(tmp_path, lean_binary):
    source = '''
import Lean
initialize sabotage : Unit ← do
  if ← (System.FilePath.mk "Domain.olean").pathExists then
    throw <| IO.userError "candidate initializer executed during audit"
theorem claim : True := True.intro
'''
    result = check_proof(tmp_path, lean_binary, source)
    assert result["status"] == "passed", result
    assert result["axioms"] == []

def test_native_executable_mutation_is_detected(tmp_path, lean_binary):
    instance = LeanPolicy(tmp_path / "policy", lean=lean_binary)
    metadata = instance.build()
    Path(metadata["executable"]).write_bytes(b"not the approved executable")
    result = instance.evaluate("verify", 3, dict.fromkeys(FACTS, True))
    assert not result["allowed"]
    assert result["failed"] == ["lean_policy_error"]
