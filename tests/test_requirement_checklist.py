"""Issue #3 regression: requirement checklist tracking with evidence gate."""
from __future__ import annotations

from pathlib import Path

import pytest

from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine


SID = "aaaaaaaa-1111-2222-3333-444444444444"


@pytest.fixture
def engine(tmp_path: Path) -> SelfDirectEngine:
    settings = Settings(use_fake_embedder=True, index_dir=tmp_path / "index",
                        contracts_path=tmp_path / "index" / "contracts.json")
    return SelfDirectEngine(settings=settings)


def test_add_splits_requirements_and_preserves_source(engine):
    try:
        result = engine.update_checklist(SID, add=[
            {"text": "한국어 보고서 작성", "source_message": "한국어 보고서를 만들어줘"},
            {"text": "출처 포함"},
            {"text": "PDF로 저장"},
            {"text": "열리는지 확인"},
        ])
        items = result["added"]
        assert [i["item_id"] for i in items] == ["req-001", "req-002", "req-003", "req-004"]
        assert items[0]["source_message"] == "한국어 보고서를 만들어줘"
        assert all(i["status"] == "open" for i in items)
        assert len(result["remaining"]) == 4
    finally:
        engine.close()


def test_done_claim_without_evidence_stays_pending(engine):
    try:
        engine.update_checklist(SID, add=[{"text": "PDF 저장"}])
        result = engine.update_checklist(SID, update=[
            {"item_id": "req-001", "status": "done"}])
        updated = result["updated"][0]
        assert updated["status"] == "pending_verification"
        assert updated["verified"] is False
        # With evidence the same claim completes.
        result = engine.update_checklist(SID, update=[
            {"item_id": "req-001", "status": "done", "evidence_chunk_ids": ["chunk-9"]},
        ])
        assert result["updated"][0]["status"] == "done"
        assert result["updated"][0]["verified"] is True
        assert result["remaining"] == []
    finally:
        engine.close()


def test_blocked_items_report_reason(engine):
    try:
        engine.update_checklist(SID, add=[{"text": "사용자 확인 대기"}])
        result = engine.update_checklist(SID, update=[
            {"item_id": "req-001", "status": "blocked", "blocked_reason": "사용자 답변 대기"}])
        item = result["updated"][0]
        assert item["status"] == "blocked"
        assert item["blocked_reason"] == "사용자 답변 대기"
        assert len(result["remaining"]) == 1
    finally:
        engine.close()


def test_unknown_item_id_is_reported_not_silent(engine):
    try:
        result = engine.update_checklist(SID, update=[{"item_id": "req-999", "status": "done"}])
        assert result["updated"][0]["ok"] is False
    finally:
        engine.close()


def test_checklist_survives_reopen_after_compaction(tmp_path):
    settings = Settings(use_fake_embedder=True, index_dir=tmp_path / "index",
                        contracts_path=tmp_path / "index" / "contracts.json")
    engine = SelfDirectEngine(settings=settings)
    try:
        engine.update_checklist(SID, add=[{"text": "요구 A"}, {"text": "요구 B"}])
    finally:
        engine.close()
    # New engine instance = simulated restart/compaction.
    engine2 = SelfDirectEngine(settings=settings)
    try:
        items = engine2.get_checklist(SID)["items"]
        assert [i["text"] for i in items] == ["요구 A", "요구 B"]
        assert items[0]["history"][0]["change"] == "added"
    finally:
        engine2.close()
