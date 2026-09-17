"""Issue #2 regression: prohibition memory, pre-action checks, exceptions, revoke."""
from __future__ import annotations

from pathlib import Path

import pytest

from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine


@pytest.fixture
def engine(tmp_path: Path) -> SelfDirectEngine:
    settings = Settings(use_fake_embedder=True, index_dir=tmp_path / "index",
                        contracts_path=tmp_path / "index" / "contracts.json")
    return SelfDirectEngine(settings=settings)


def test_source_quote_survives_revision(engine):
    try:
        engine.upsert_contracts([{"id": "no-prod-db", "type": "must_not", "regex": "prod-db",
                                  "source_quote": "운영 DB는 수정하지 마"}])
        engine.upsert_contracts([{"id": "no-prod-db", "type": "must_not", "regex": "prod|production",
                                  "description": "broadened"}])
        contracts = engine.list_contracts()["contracts"]
        rule = next(c for c in contracts if c["id"] == "no-prod-db")
        assert rule["source_quote"] == "운영 DB는 수정하지 마"
        assert rule["revision"] == 2
    finally:
        engine.close()


def test_exception_regex_lowers_violation_to_suspicious():
    from self_directing_mcp.audit.runner import audit_contract
    from self_directing_mcp.schemas import Chunk, ContractRule

    rule = {"id": "no-delete", "type": "must_not", "regex": r"rm\s+-rf", "exception_regex": "tmp/rebuild"}
    rule = ContractRule.model_validate(rule)
    chunk = Chunk(chunk_id="c1", session_id="s", kind="tool_call", provider="codex",
                  text="[tool_call:shell] rm -rf tmp/rebuild/old", content_hash="h",
                  meta={"role": "assistant"})
    result = audit_contract(rule, session_id="s", store=None, provider="codex",
                            proposed=chunk, chunks=[chunk])
    assert result.verdict == "suspicious"
    assert "exception" in (result.basis or "")


def test_violation_reason_includes_source_quote_and_basis():
    from self_directing_mcp.audit.runner import audit_contract
    from self_directing_mcp.schemas import Chunk, ContractRule

    rule = ContractRule.model_validate({"id": "no-prod", "type": "must_not", "regex": "prod-db",
                                        "source_quote": "운영 DB는 수정하지 마"})
    chunk = Chunk(chunk_id="c1", session_id="s", kind="tool_call", provider="codex",
                  text="[tool_call:shell] migrate prod-db", content_hash="h",
                  meta={"role": "assistant"})
    result = audit_contract(rule, session_id="s", store=None, provider="codex",
                            proposed=chunk, chunks=[chunk])
    assert result.verdict == "violation"
    assert "운영 DB는 수정하지 마" in result.reason
    assert "no-prod" in (result.basis or "")


def test_revoke_disables_rule_and_keeps_history(engine):
    try:
        engine.upsert_contracts([{"id": "temp", "type": "must_not", "regex": "TEMP"}])
        result = engine.revoke_contract("temp")
        assert result["ok"] and result["contract"]["enabled"] is False
        # Disabled rule no longer triggers.
        audit = engine.check_action("missing-session", {"tool_name": "shell", "arguments": "rm TEMP"})
        assert audit["verdict"] in ("unknown", "clean")
        missing = engine.revoke_contract("nonexistent")
        assert missing["ok"] is False
    finally:
        engine.close()

