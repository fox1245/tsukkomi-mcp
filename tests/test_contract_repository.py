"""``ContractRepository``: the operations the engine performs on its contracts, as a protocol the file store satisfies.

The protocol is extracted from what ``SelfDirectEngine`` actually calls (a test below reads the engine's source and
compares), so a future backend implements exactly that and nothing is promised that the engine does not use. The same
conformance checks run against the file store and against a small in-memory implementation that lives only here: that
the engine works through the second one is the evidence the protocol is complete. There is no remote implementation,
and the command hooks still read the local contracts file directly.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
import typing
from pathlib import Path

import pytest

from self_directing_mcp import engine as engine_module
from self_directing_mcp.audit.contracts import ContractRepository, ContractStore
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.schemas import ContractRule, ContractsDocument

RM = ContractRule(id="no-rm", type="must_not", scope="tool_call", regex=r"rm\s+-rf")
TESTS = ContractRule(id="tests-ran", type="must", scope="tool_call", regex="pytest")
MEMBERS = {"capture", "current_token", "publish_document", "prepare_upsert", "prepare_revoke", "seed_from",
           "seed_missing"}


class MemoryRepository:
    """A repository that keeps its one document in memory: what a non-file backend has to provide."""

    def __init__(self):
        self._raw: bytes | None = None

    def capture(self):
        return (None, "missing") if self._raw is None else (self._raw, "present:" + hashlib.sha256(self._raw).hexdigest())

    def current_token(self):
        return self.capture()[1]

    def publish_document(self, document, payload, expected_token=None):
        if expected_token is not None and self.current_token() != expected_token:
            raise ValueError("stale contract preparation")
        self._raw = payload.encode("utf-8")

    prepare_upsert = staticmethod(ContractStore.prepare_upsert)
    prepare_revoke = staticmethod(ContractStore.prepare_revoke)

    def seed_from(self, seed_path):
        if self._raw is not None and ContractsDocument.model_validate_json(self._raw).contracts:
            return 0
        document = ContractsDocument.model_validate(json.loads(Path(seed_path).read_text(encoding="utf-8")))
        self.publish_document(document, document.model_dump_json(indent=2))
        return len(document.contracts)

    def seed_missing(self, preset_path):
        document = ContractsDocument.model_validate_json(self._raw) if self._raw is not None else ContractsDocument()
        presets = ContractsDocument.model_validate(json.loads(Path(preset_path).read_text(encoding="utf-8"))).contracts
        known = {rule.id for rule in document.contracts}
        added = [rule for rule in presets if rule.id not in known]
        if added:
            document.contracts.extend(added)
            self.publish_document(document, document.model_dump_json(indent=2))
        return len(added)


FACTORIES = {"file": lambda tmp_path: ContractStore(tmp_path / "contracts.json"), "memory": lambda tmp_path: MemoryRepository()}


@pytest.fixture(params=sorted(FACTORIES))
def repository(request, tmp_path):
    return FACTORIES[request.param](tmp_path)


def publish(repo, *rules, expected=None):
    document = ContractsDocument()
    repo.prepare_upsert(document, list(rules))
    repo.publish_document(document, document.model_dump_json(indent=2), expected)
    return document


# ---------------------------------------------------------------------------------------------- the protocol itself

def test_the_protocol_is_exactly_what_the_engine_uses_today():
    source = inspect.getsource(engine_module)
    used = set(re.findall(r"self\.contracts\.(\w+)", source))
    assert used == MEMBERS, "the engine's calls on self.contracts and ContractRepository must stay the same set"
    declared = {name for name, member in vars(ContractRepository).items()
                if callable(member) and not name.startswith("_")}
    assert declared == MEMBERS


def test_the_file_store_satisfies_the_protocol_and_so_does_the_in_memory_double(tmp_path):
    assert isinstance(ContractStore(tmp_path / "c.json"), ContractRepository)
    assert isinstance(MemoryRepository(), ContractRepository)
    assert not isinstance(object(), ContractRepository)
    assert getattr(ContractRepository, "_is_runtime_protocol", False) and typing.Protocol in ContractRepository.__mro__


def test_the_engine_is_typed_against_the_protocol_not_the_file_store():
    assert "ContractRepository" in inspect.getsource(SelfDirectEngine.__init__)


def test_there_is_no_remote_implementation():
    package = Path(inspect.getsourcefile(ContractStore)).parent.parent
    names = {path.name for path in package.rglob("*.py")}
    assert not {name for name in names if "postgres" in name or "remote_contract" in name}


# ------------------------------------------------------------------------------------------------- the conformance

def test_an_empty_repository_captures_no_document_with_a_stable_token(repository):
    assert repository.capture() == (None, "missing") and repository.current_token() == "missing"


def test_publishing_replaces_the_document_and_changes_the_token_only_with_the_content(repository):
    first = publish(repository, RM)
    raw, token = repository.capture()
    assert ContractsDocument.model_validate_json(raw).contracts[0].id == "no-rm" and token != "missing"
    assert repository.current_token() == token
    repository.publish_document(first, first.model_dump_json(indent=2))              # the same bytes again
    assert repository.capture() == (raw, token)
    publish(repository, RM, TESTS)
    assert repository.current_token() != token


def test_a_stale_preparation_is_refused_and_publishes_nothing(repository):
    publish(repository, RM)
    raw, token = repository.capture()
    publish(repository, RM, TESTS, expected=token)                                       # current token: published
    after, _ = repository.capture()
    assert after != raw
    document = ContractsDocument()
    repository.prepare_upsert(document, [TESTS])
    with pytest.raises(ValueError, match="stale|changed"):
        repository.publish_document(document, document.model_dump_json(indent=2), token)   # the token is old now
    assert repository.capture()[0] == after


def test_prepare_upsert_edits_the_given_document_only_and_counts_revisions(repository):
    publish(repository, RM)
    before = repository.capture()
    raw, _ = before
    document = ContractsDocument.model_validate_json(raw)
    saved = repository.prepare_upsert(document, [RM.model_copy(update={"regex": "rm"}), TESTS])
    assert [(rule.id, rule.revision) for rule in saved] == [("no-rm", 2), ("tests-ran", 1)]
    assert [rule.regex for rule in document.history["no-rm"]] == [r"rm\s+-rf"]
    assert repository.capture() == before                                              # nothing was published


def test_prepare_revoke_disables_keeps_the_history_and_says_when_there_is_nothing(repository):
    publish(repository, RM)
    document = ContractsDocument.model_validate_json(repository.capture()[0])
    disabled = repository.prepare_revoke(document, "no-rm")
    assert disabled is not None and disabled.enabled is False and document.history["no-rm"][0].enabled is True
    assert repository.prepare_revoke(document, "ghost") is None


def test_seeding_adds_what_is_absent_and_never_alters_an_existing_contract(repository, tmp_path):
    presets = tmp_path / "presets.json"
    presets.write_text(ContractsDocument(contracts=[RM.model_copy(update={"enabled": False}), TESTS]).model_dump_json(),
                       encoding="utf-8")
    publish(repository, RM)                                                              # an enabled no-rm already exists
    assert repository.seed_from(presets) == 0                                           # not empty: nothing seeded
    assert repository.seed_missing(presets) == 1
    stored = {r.id: r for r in ContractsDocument.model_validate_json(repository.capture()[0]).contracts}
    assert stored["no-rm"].enabled is True and "tests-ran" in stored


# ---------------------------------------------------------------------------------- the engine over any repository

def test_the_engine_runs_upsert_revoke_list_and_check_through_the_in_memory_repository(tmp_path):
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=tmp_path / "ix",
                                       openrouter_api_key_file=None))
    try:
        engine.ensure_ready()
        engine.contracts = MemoryRepository()
        engine.upsert_contracts([{"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"}])
        assert [c["id"] for c in engine.list_contracts()["contracts"]] == ["no-rm"]
        held = engine.revoke_contract("no-rm")                                         # no user anchor: parked, in force
        assert held["status"] == "pending" and engine.list_contracts()["contracts"][0]["enabled"] is True
        assert engine.audit_status()["contract_count"] == 1
        assert not (tmp_path / "ix" / "contracts.json").exists()                       # the file store was not involved
    finally:
        engine.close()
