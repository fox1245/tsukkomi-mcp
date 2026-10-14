"""Contract schema extensions: risk, auto_continue and the optional ``when`` applicability condition.

Everything here is synthetic. Existing contracts.json files must load and be written back unchanged.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from self_directing_mcp.audit.contracts import ContractStore
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.schemas import ContractRule, ContractsDocument
from semantic_support import codex_user_anchor

ROOT = Path(__file__).resolve().parents[1]
SID = "11111111-2222-3333-4444-555555555555"
AGY_SID = "11111111-1111-4111-8111-111111111111"
LEGACY = {"id": "legacy", "type": "must_not", "scope": "tool_call", "severity": "high", "regex": "rm -rf",
          "enabled": True, "revision": 1, "roles": ["assistant"]}


def _message(text, role="assistant"):
    return {"type": "response_item", "payload": {"type": "message", "role": role, "content": text}}


def _call(command, call_id="call-1"):
    return {"type": "response_item", "payload": {"type": "function_call", "name": "shell", "call_id": call_id,
                                                 "arguments": {"command": command}}}


def _write(path, events, mode="w"):
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        for event in events:
            stream.write(json.dumps(event) + "\n")


@pytest.fixture
def session(tmp_path, monkeypatch):
    for name in ("SELF_DIRECT_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl"
    _write(path, [{"type": "session_meta", "payload": {"id": SID}}])
    engine = SelfDirectEngine(Settings(codex_sessions_dir=root, index_dir=tmp_path / "index",
                                       use_fake_embedder=True, local_only=True, _env_file=None))
    yield engine, path
    engine.close()


# --- schema -----------------------------------------------------------------------------------------------

def test_legacy_contract_loads_with_defaults_and_serializes_unchanged():
    rule = ContractRule.model_validate(LEGACY)
    assert (rule.risk, rule.auto_continue, rule.when) == ("normal", False, None)
    dumped = rule.model_dump(mode="json")
    # Defaults are omitted so files written by this version stay readable by older processes
    # (extra="forbid") and the contract snapshot hash of unchanged contracts does not move.
    assert not {"risk", "auto_continue", "when"} & set(dumped)
    assert json.loads(rule.model_dump_json()) == dumped


def test_new_fields_serialize_only_when_set():
    rule = ContractRule(id="r", type="must", regex="x", risk="high", auto_continue=True,
                        when={"tool_call_regex": "docker"})
    dumped = rule.model_dump(mode="json")
    assert dumped["risk"] == "high" and dumped["auto_continue"] is True
    assert dumped["when"] == {"tool_call_regex": "docker"}
    assert ContractRule.model_validate_json(rule.model_dump_json()) == rule


def test_json_schema_exposes_the_new_fields():
    props = ContractRule.model_json_schema()["properties"]
    assert props["risk"]["enum"] == ["normal", "high"]
    assert {"auto_continue", "when"} <= set(props)


@pytest.mark.parametrize("fields", [
    {"type": "must_not", "regex": "x", "auto_continue": True},               # only completion obligations
    {"type": "must", "auto_continue": True},                                  # needs a verifier
    {"type": "must", "regex": "x", "before_regex": "y", "auto_continue": True},  # gates are not completions
    {"type": "must_not", "regex": "x", "when": {"tool_call_regex": "y"}},     # when scopes must obligations
    {"type": "must", "regex": "x", "when": {"tool_call_regex": ""}},
    {"type": "must", "regex": "x", "when": {"tool_call_regex": "y", "extra": 1}},
    {"type": "must", "regex": "x", "when": "docker"},
    {"type": "must", "regex": "x", "risk": "extreme"},
])
def test_invalid_combinations_are_rejected(fields):
    with pytest.raises(ValidationError):
        ContractRule(id="bad", **fields)


def test_upsert_list_revoke_round_trip_preserves_the_new_fields(session):
    engine, path = session
    anchor = codex_user_anchor(engine, path, SID, "change the release condition and drop the gate")
    engine.upsert_contracts([
        {"id": "gate", "type": "must_not", "scope": "tool_call", "regex": "git push --force", "risk": "high"},
        {"id": "done", "type": "must", "scope": "message", "regex": "VERIFIED", "auto_continue": True,
         "when": {"tool_call_regex": "deploy"}}])
    listed = {c["id"]: c for c in engine.list_contracts()["contracts"]}
    assert listed["gate"]["risk"] == "high" and "auto_continue" not in listed["gate"]
    assert listed["done"]["auto_continue"] is True and listed["done"]["when"] == {"tool_call_regex": "deploy"}
    # A new process reads the same file.
    reloaded = SelfDirectEngine(engine.settings)
    try:
        assert {c["id"]: c for c in reloaded.list_contracts()["contracts"]} == listed
        reloaded.upsert_contracts([{"id": "done", "type": "must", "scope": "message", "regex": "VERIFIED",
                                    "auto_continue": True, "when": {"tool_call_regex": "release"},
                                    "source_event_id": anchor}])
        document = ContractsDocument.model_validate_json(engine.settings.resolve_contracts_path().read_bytes())
        current = {c.id: c for c in document.contracts}["done"]
        assert current.revision == 2 and current.when.tool_call_regex == "release"
        assert document.history["done"][0].when.tool_call_regex == "deploy"
    finally:
        reloaded.close()
    engine.revoke_contract("gate", source_event_id=anchor)
    assert engine.list_contracts()["contracts"][0]["risk"] == "high"  # revoking keeps the declaration


def test_rewriting_a_legacy_file_adds_no_new_keys(session):
    engine, _path = session
    contracts = engine.settings.resolve_contracts_path()
    contracts.parent.mkdir(parents=True, exist_ok=True)
    contracts.write_text(json.dumps({"schema_version": "1.0", "contracts": [LEGACY]}), encoding="utf-8")
    engine.upsert_contracts([{"id": "other", "type": "must_not", "regex": "x"}])
    written = json.loads(contracts.read_text(encoding="utf-8"))
    assert all(not {"risk", "auto_continue", "when"} & set(rule) for rule in written["contracts"])


def test_upsert_rejects_an_uncompilable_when_without_saving(session):
    engine, _path = session
    with pytest.raises(re.error):
        engine.upsert_contracts([{"id": "bad", "type": "must", "scope": "message", "regex": "x",
                                  "when": {"tool_call_regex": "("}}])
    assert engine.list_contracts()["contracts"] == []


# --- when -------------------------------------------------------------------------------------------------

def _finding(audit, contract_id):
    return next(f for f in audit["findings"] if f["contract_id"] == contract_id)


def _tests_rule(**extra):
    return {"id": "ran-tests", "type": "must", "scope": "tool_call", "regex": "pytest", **extra}


def test_unmet_when_reports_not_applicable_with_a_reason_and_is_never_dropped(session):
    engine, path = session
    _write(path, [_call("echo hello")], "a")
    engine.upsert_contracts([_tests_rule(when={"tool_call_regex": "docker build"})])
    audit = engine.audit_session(SID)
    finding = _finding(audit, "ran-tests")
    assert (finding["verdict"], finding["applicability"]) == ("clean", "not_applicable")
    assert "when" in finding["reason"] and "docker build" in finding["reason"]
    assert audit["verdict"] == "clean" and audit["coverage"]["complete"] is True
    assert any("not_applicable" in note for note in audit["notes"])


def test_without_when_the_same_obligation_is_evaluated_as_before(session):
    engine, path = session
    _write(path, [_call("echo hello")], "a")
    engine.upsert_contracts([_tests_rule()])
    finding = _finding(engine.audit_session(SID), "ran-tests")
    assert finding["verdict"] == "violation" and finding["applicability"] == "applicable"


def test_met_when_evaluates_exactly_as_today(session):
    engine, path = session
    _write(path, [_call("docker build .", "c1")], "a")
    engine.upsert_contracts([_tests_rule(when={"tool_call_regex": "docker build"})])
    finding = _finding(engine.audit_session(SID), "ran-tests")
    assert finding["verdict"] == "violation" and finding["applicability"] == "applicable"
    _write(path, [_call("pytest -q", "c2")], "a")
    finding = _finding(engine.audit_session(SID), "ran-tests")
    assert finding["verdict"] == "clean" and finding["applicability"] == "applicable"


def test_when_inspects_tool_calls_even_if_the_obligation_only_reads_messages(session):
    """The audit keeps event text only for what a contract can inspect; ``when`` must widen that."""
    engine, path = session
    _write(path, [_call("deploy --prod", "c1"), _message("done")], "a")
    engine.upsert_contracts([{"id": "say-verified", "type": "must", "scope": "message", "regex": "VERIFIED",
                              "when": {"tool_call_regex": "deploy"}}])
    finding = _finding(engine.audit_session(SID), "say-verified")
    assert finding["applicability"] == "applicable" and finding["verdict"] == "violation"


def test_when_matching_a_user_message_is_not_a_tool_call(session):
    engine, path = session
    _write(path, [_message("please deploy later", "user"), _call("echo hello")], "a")
    engine.upsert_contracts([_tests_rule(when={"tool_call_regex": "deploy"})])
    assert _finding(engine.audit_session(SID), "ran-tests")["applicability"] == "not_applicable"


def test_unmet_when_with_incomplete_coverage_is_unknown_not_not_applicable(session):
    engine, path = session
    _write(path, [_call("echo hello")], "a")
    with path.open("a", encoding="utf-8") as stream:
        stream.write("not json\n")  # a record we could not read may hold the condition
    engine.upsert_contracts([_tests_rule(when={"tool_call_regex": "docker build"})])
    audit = engine.audit_session(SID)
    finding = _finding(audit, "ran-tests")
    assert finding["verdict"] == "unknown" and finding["applicability"] == "applicable"
    assert audit["verdict"] == "unknown"


def test_invalid_when_regex_in_a_hand_edited_file_is_unknown(session):
    engine, path = session
    _write(path, [_call("echo hello")], "a")
    engine.upsert_contracts([_tests_rule()])
    contracts = engine.settings.resolve_contracts_path()
    document = json.loads(contracts.read_text(encoding="utf-8"))
    document["contracts"][0]["when"] = {"tool_call_regex": "("}
    contracts.write_text(json.dumps(document), encoding="utf-8")
    finding = _finding(engine.audit_session(SID), "ran-tests")
    assert finding["verdict"] == "unknown" and "when" in finding["reason"]


def test_description_only_obligation_stays_unknown_whatever_its_condition(session):
    engine, path = session
    _write(path, [_call("echo hello")], "a")
    engine.upsert_contracts([{"id": "prose", "type": "must", "description": "use uv for installs",
                              "when": {"tool_call_regex": "pip install"}}])
    # No verifier exists, so unmet-when must not silently become a verdict about the prose.
    finding = _finding(engine.audit_session(SID), "prose")
    assert finding["verdict"] == "unknown"


def test_agy_model_proposals_count_as_the_condition_being_met(tmp_path):
    """AGY records proposals without receipts as meta events; they must not make an obligation vanish."""
    fixture = ROOT / "fixtures" / "sample_agy_session" / "transcript.jsonl"
    app = tmp_path / "app"
    path = app / "brain" / AGY_SID / ".system_generated" / "logs" / "transcript.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(fixture.read_bytes())
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=tmp_path / "index",
                                       agy_app_data_dirs=[app], openrouter_api_key_file=None))
    try:
        engine.upsert_contracts([{"id": "say-verified", "provider": "agy", "type": "must", "scope": "message",
                                  "regex": "VERIFIED", "when": {"tool_call_regex": "npm test"}},
                                 {"id": "other", "provider": "agy", "type": "must", "scope": "message",
                                  "regex": "VERIFIED", "when": {"tool_call_regex": "kubectl apply"}}])
        audit = engine.audit_session(AGY_SID, provider="agy", path=str(path))
        assert audit["coverage"]["complete"] is True
        assert _finding(audit, "say-verified")["applicability"] == "applicable"
        assert _finding(audit, "other")["applicability"] == "not_applicable"
    finally:
        engine.close()


# --- high-risk presets ------------------------------------------------------------------------------------

PRESETS = ROOT / "src" / "self_directing_mcp" / "resources" / "high_risk_presets.json"

POSITIVE = {
    "preset-no-recursive-delete": ["rm -rf build", "rm -fr /", "rm -r -f x", "rm --recursive dir",
                                   "Remove-Item -Recurse -Force C:\\x", "rmdir /s /q out",
                                   "python -c 'import shutil; shutil.rmtree(p)'", "find . -name x -delete"],
    "preset-no-write-outside-workspace": ["echo x > /etc/hosts", "cp a.txt /usr/local/bin/a", "tee -a ~/.bashrc",
                                          "mv out ../../other", "sed -i s/a/b/ /etc/fstab",
                                          {"TargetFile": "/etc/passwd"}, "dd if=a of=/boot/x"],
    "preset-no-force-push": ["git push --force origin main", "git push -f", "git push origin +main",
                             "git push --force-with-lease origin x", "git push -uf origin x"],
    "preset-no-branch-delete": ["git branch -D feature", "git branch -d old", "git branch --delete old",
                                "git push origin --delete feature", "git push origin :feature"],
}
NEGATIVE = {
    "preset-no-recursive-delete": ["rm file.txt", "rm -f file.txt", "ls -R", "git rm -r --cached x", "echo rmdir"],
    "preset-no-write-outside-workspace": ["echo x > build/out.txt", "cp a.txt b.txt", "ls /etc", "cat /etc/hosts",
                                          "echo done 2>/dev/null"],
    "preset-no-force-push": ["git push origin main", "git push -u origin feature-fix",
                             "git push origin HEAD:refs/heads/x"],
    "preset-no-branch-delete": ["git branch --list", "git branch -a", "git branch feature", "git push origin main",
                                "git push origin HEAD:refs/heads/x"],
}


def _presets():
    return ContractsDocument.model_validate_json(PRESETS.read_bytes()).contracts


def test_presets_are_a_small_set_of_disabled_high_risk_contracts():
    rules = _presets()
    assert 3 <= len(rules) <= 5 and len({r.id for r in rules}) == len(rules)
    for rule in rules:
        assert rule.enabled is False and rule.risk == "high" and rule.type == "must_not"
        assert rule.scope == "tool_call" and not rule.auto_continue and rule.description
        re.compile(rule.regex)
    assert set(POSITIVE) == {r.id for r in rules}


def _tool_text(sample):
    arguments = sample if isinstance(sample, dict) else {"CommandLine": sample}
    return "[tool_call:run_command] " + json.dumps(arguments, ensure_ascii=False, sort_keys=True)


@pytest.mark.parametrize("rule_id", sorted(POSITIVE))
def test_preset_regexes_match_destructive_forms_and_not_benign_ones(rule_id):
    rule = next(r for r in _presets() if r.id == rule_id)
    pattern = re.compile(rule.regex, re.IGNORECASE | re.MULTILINE)
    for sample in POSITIVE[rule_id]:
        assert pattern.search(_tool_text(sample)), sample
    for sample in NEGATIVE[rule_id]:
        assert not pattern.search(_tool_text(sample)), sample


def _engine(tmp_path, **settings):
    return SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=tmp_path / "index",
                                     openrouter_api_key_file=None, **settings))


def test_presets_are_not_seeded_by_default(tmp_path):
    engine = _engine(tmp_path)
    try:
        engine.ensure_ready()
        assert engine.list_contracts()["contracts"] == []
        assert Settings(_env_file=None).seed_example_contracts is False
    finally:
        engine.close()


def test_seeding_adds_presets_disabled_and_never_resets_a_user_choice(tmp_path):
    engine = _engine(tmp_path, seed_example_contracts=True)
    try:
        engine.ensure_ready()
        listed = {c["id"]: c for c in engine.list_contracts()["contracts"]}
        preset_ids = {r.id for r in _presets()}
        assert preset_ids <= set(listed)
        assert all(listed[i]["enabled"] is False and listed[i]["risk"] == "high" for i in preset_ids)
        assert not any(c.id in preset_ids for c in engine.hook_obligations(SID)["contracts"])
        user = dict(listed["preset-no-force-push"], enabled=True)
        engine.upsert_contracts([ContractRule.model_validate(user)])
    finally:
        engine.close()
    again = _engine(tmp_path, seed_example_contracts=True)
    try:
        again.ensure_ready()
        listed = [c for c in again.list_contracts()["contracts"] if c["id"] in preset_ids]
        assert len(listed) == len(preset_ids)
        assert {c["id"]: c["enabled"] for c in listed}["preset-no-force-push"] is True
    finally:
        again.close()


def test_seed_missing_is_idempotent_and_leaves_existing_contracts_alone(tmp_path):
    store = ContractStore(tmp_path / "contracts.json")
    store.upsert([ContractRule(id="mine", type="must_not", regex="x")])
    assert store.seed_missing(PRESETS) == len(_presets())
    assert store.seed_missing(PRESETS) == 0
    assert "mine" in {r.id for r in store.list()}


def test_when_also_scopes_a_gating_prerequisite_checked_against_a_proposed_action(session):
    engine, path = session
    _write(path, [_call("echo hello")], "a")
    engine.upsert_contracts([{"id": "tests-first", "type": "must", "scope": "tool_call", "regex": "pytest",
                              "before_regex": "release", "requires_success": True,
                              "when": {"tool_call_regex": "docker build"}}])
    action = {"tool_name": "shell", "arguments": {"command": "release now"}}
    unmet = engine.check_action(SID, action)
    assert (unmet["findings"][0]["verdict"], unmet["findings"][0]["applicability"]) == ("clean", "not_applicable")
    _write(path, [_call("docker build .", "c1")], "a")
    met = engine.check_action(SID, action)
    assert (met["verdict"], met["findings"][0]["applicability"]) == ("violation", "applicable")


def test_unmet_when_on_a_gate_is_unknown_for_a_proposed_action_when_coverage_is_incomplete(session):
    engine, path = session
    _write(path, [_call("echo hello")], "a")
    with path.open("a", encoding="utf-8") as stream:
        stream.write("not json\n")
    engine.upsert_contracts([{"id": "tests-first", "type": "must", "scope": "tool_call", "regex": "pytest",
                              "before_regex": "release", "when": {"tool_call_regex": "docker build"}}])
    result = engine.check_action(SID, {"tool_name": "shell", "arguments": {"command": "release now"}})
    assert result["coverage"]["complete"] is False
    assert (result["verdict"], result["findings"][0]["verdict"], result["findings"][0]["applicability"]) == (
        "unknown", "unknown", "applicable")
