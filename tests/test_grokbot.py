from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.grokbot.discover import (
    PathTraversalError,
    ensure_under_root,
    find_session_by_id,
    resolve_session_path,
    session_id_from_path,
)
from self_directing_mcp.grokbot.parse import parse_session_chunks, peek_session_id

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
GROK_ROOT = FIXTURES / "sample_grokbot_session"
GROK_SESSION_ID = "15c5c6e5-f1db-4492-add5-4d6d2ab5600c"
GROK_FILE = GROK_ROOT / GROK_SESSION_ID / f"{GROK_SESSION_ID}.jsonl"


@pytest.fixture
def grok_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SelfDirectEngine:
    monkeypatch.setenv("SELF_DIRECT_USE_FAKE_EMBEDDER", "true")
    monkeypatch.setenv("SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR", str(GROK_ROOT))
    monkeypatch.setenv("SELF_DIRECT_SESSION_PROVIDER", "grokbot")
    settings = Settings(
        grokbot_transcripts_dir=str(GROK_ROOT),
        session_provider="grokbot",
        index_dir=tmp_path / "index",
        use_fake_embedder=True,
        contracts_path=tmp_path / "index" / "contracts.json",
    )
    seed = FIXTURES / "contracts" / "examples.json"
    (tmp_path / "index").mkdir(parents=True, exist_ok=True)
    (tmp_path / "index" / "contracts.json").write_text(
        seed.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return SelfDirectEngine(settings=settings)


def test_discover_and_parse_grokbot_fixture():
    assert GROK_FILE.exists()
    assert session_id_from_path(GROK_FILE) == GROK_SESSION_ID
    assert peek_session_id(GROK_FILE) == GROK_SESSION_ID
    found = find_session_by_id(GROK_ROOT, GROK_SESSION_ID)
    assert found is not None
    assert found.resolve() == GROK_FILE.resolve()

    chunks, offset, sid = parse_session_chunks(GROK_FILE)
    assert sid == GROK_SESSION_ID
    assert offset > 0
    assert 8 <= len(chunks) <= 40
    kinds = {c.kind for c in chunks}
    assert "tool_call" in kinds
    assert "turn" in kinds or "meta" in kinds
    assert any("Shell" in c.text or "WebSearch" in c.text for c in chunks)
    assert any("curl" in c.text.lower() for c in chunks)


def test_grokbot_sync_and_search_finds_tool_name(grok_engine: SelfDirectEngine):
    r = grok_engine.sync_session(session_id=GROK_SESSION_ID, provider="grokbot")
    assert r["ok"] is True
    assert r["provider"] == "grokbot"
    assert r["embedded"] >= 1

    res = grok_engine.search_history(
        "Shell WebSearch",
        session_id=GROK_SESSION_ID,
        mode="hybrid",
        top_k=10,
        provider="grokbot",
    )
    assert res["ok"] is True
    assert len(res["hits"]) >= 1
    blob = json.dumps(res["hits"])
    assert "Shell" in blob or "WebSearch" in blob or "shell" in blob.lower()


def test_grokbot_path_traversal_rejected(grok_engine: SelfDirectEngine, tmp_path: Path):
    outsider = tmp_path / "outside.jsonl"
    outsider.write_text(
        json.dumps({"role": "user", "message": {"content": [{"type": "text", "text": "x"}]}})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PathTraversalError):
        ensure_under_root(outsider, GROK_ROOT)
    res = grok_engine.sync_session(path=str(outsider), provider="grokbot")
    assert res["ok"] is False
    assert res["error"] == "path_traversal"


def test_grokbot_resolve_by_id():
    p = resolve_session_path(GROK_ROOT, session_id=GROK_SESSION_ID)
    assert p.resolve() == GROK_FILE.resolve()


def test_grokbot_audit_catches_curl(grok_engine: SelfDirectEngine):
    grok_engine.sync_session(session_id=GROK_SESSION_ID, provider="grokbot")
    audit = grok_engine.audit_session(GROK_SESSION_ID, provider="grokbot")
    assert audit["ok"] is True
    findings = {f["contract_id"]: f for f in audit["findings"]}
    assert "no-curl-exfil" in findings
    assert findings["no-curl-exfil"]["verdict"] == "violation"
