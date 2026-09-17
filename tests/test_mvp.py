from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_directing_mcp.codex.discover import (
    PathTraversalError,
    ensure_under_root,
    find_session_by_id,
    resolve_session_path,
    session_id_from_filename,
)
from self_directing_mcp.codex.parse import parse_session_chunks, peek_session_id
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.schemas import ContractRule
from self_directing_mcp.security.mask import mask_secrets

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SAMPLE_ROOT = FIXTURES / "sample_session"
SESSION_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
SAMPLE_FILE = (
    SAMPLE_ROOT
    / "2026"
    / "09"
    / "09"
    / f"rollout-2026-09-09T01-00-00-{SESSION_ID}.jsonl"
)


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SelfDirectEngine:
    monkeypatch.setenv("SELF_DIRECT_USE_FAKE_EMBEDDER", "true")
    monkeypatch.setenv("SELF_DIRECT_CODEX_SESSIONS_DIR", str(SAMPLE_ROOT))
    settings = Settings(
        codex_sessions_dir=SAMPLE_ROOT,
        index_dir=tmp_path / "index",
        use_fake_embedder=True,
        contracts_path=tmp_path / "index" / "contracts.json",
    )
    # Seed contracts into index dir from fixtures
    seed = FIXTURES / "contracts" / "examples.json"
    (tmp_path / "index").mkdir(parents=True, exist_ok=True)
    (tmp_path / "index" / "contracts.json").write_text(seed.read_text(encoding="utf-8"), encoding="utf-8")
    return SelfDirectEngine(settings=settings)


def test_discover_and_parse_synthetic_session():
    assert SAMPLE_FILE.exists()
    assert session_id_from_filename(SAMPLE_FILE) == SESSION_ID
    assert peek_session_id(SAMPLE_FILE) == SESSION_ID
    found = find_session_by_id(SAMPLE_ROOT, SESSION_ID)
    assert found is not None
    assert found.resolve() == SAMPLE_FILE.resolve()

    chunks, offset, sid = parse_session_chunks(SAMPLE_FILE)
    assert sid == SESSION_ID
    assert offset > 0
    assert len(chunks) >= 5
    kinds = {c.kind for c in chunks}
    assert "tool_call" in kinds
    assert "meta" in kinds or "policy" in kinds
    assert any("curl" in c.text for c in chunks)


def test_incremental_sync_only_embeds_new_hashes(engine: SelfDirectEngine):
    r1 = engine.sync_session(session_id=SESSION_ID)
    assert r1["ok"] is True
    assert r1["embedded"] >= 1
    first_embedded = r1["embedded"]
    total1 = r1["total_chunks"]

    r2 = engine.sync_session(session_id=SESSION_ID)
    assert r2["ok"] is True
    assert r2["embedded"] == 0
    assert r2["new_chunks"] == 0
    assert r2["total_chunks"] == total1
    assert first_embedded == total1 or total1 >= first_embedded


def test_hybrid_search_finds_known_tool_call(engine: SelfDirectEngine):
    engine.sync_session(session_id=SESSION_ID)
    res = engine.search_history(
        "shell curl api_key",
        session_id=SESSION_ID,
        mode="hybrid",
        top_k=10,
    )
    assert res["ok"] is True
    assert len(res["hits"]) >= 1
    blob = json.dumps(res["hits"]).lower()
    assert "curl" in blob or "shell" in blob or "api_key" in blob


def test_regex_must_not_catches_curl_api_key(engine: SelfDirectEngine):
    engine.sync_session(session_id=SESSION_ID)
    audit = engine.audit_session(SESSION_ID)
    assert audit["ok"] is True
    findings = {f["contract_id"]: f for f in audit["findings"]}
    assert "no-curl-exfil" in findings
    assert findings["no-curl-exfil"]["verdict"] == "violation"
    assert any(e["match_type"] == "regex" for e in findings["no-curl-exfil"]["evidence"])
    assert audit["verdict"] == "violation"


def test_unrelated_semantic_hit_alone_not_violation(engine: SelfDirectEngine):
    engine.sync_session(session_id=SESSION_ID)
    # Description-only must_not about gardening — hybrid may or may not hit;
    # must NEVER be violation without regex evidence.
    engine.upsert_contracts(
        [
            ContractRule(
                id="semantic-only-example",
                type="must_not",
                scope="any",
                severity="medium",
                description="Description-only contract about unrelated gardening topics",
                search_query="planting roses in the garden watering flowers",
                regex=None,
                enabled=True,
            ).model_dump()
        ]
    )
    audit = engine.audit_session(SESSION_ID)
    findings = {f["contract_id"]: f for f in audit["findings"]}
    sem = findings["semantic-only-example"]
    assert sem["verdict"] in ("clean", "suspicious", "unknown")
    assert sem["verdict"] != "violation"


def test_path_outside_sessions_root_rejected(engine: SelfDirectEngine, tmp_path: Path):
    outsider = tmp_path / "outside.jsonl"
    outsider.write_text('{"type":"session_meta","payload":{"id":"x"}}\n', encoding="utf-8")
    with pytest.raises(PathTraversalError):
        ensure_under_root(outsider, SAMPLE_ROOT)
    res = engine.sync_session(path=str(outsider))
    assert res["ok"] is False
    assert res["error"] == "path_traversal"


def test_secret_masking_in_get_chunk_and_audit(engine: SelfDirectEngine):
    engine.sync_session(session_id=SESSION_ID)
    # get_chunk masking
    chunks = engine.store.list_chunks(SESSION_ID)  # type: ignore[union-attr]
    secretish = [c for c in chunks if "sk-" in c.text or "Bearer" in c.text]
    assert secretish, "fixture should contain secrets"
    got = engine.get_chunk(secretish[0].chunk_id)
    assert got["found"] is True
    text = got["chunk"]["text"]
    assert "sk-SECRETKEY1234567890" not in text
    assert "sk-***" in text or "***" in text

    audit = engine.audit_session(SESSION_ID)
    dumped = json.dumps(audit)
    assert "sk-SECRETKEY1234567890" not in dumped
    assert "tok_live_ABCDEFG_should_mask" not in dumped


def test_mask_secrets_unit():
    s = mask_secrets("key sk-ABCDEFGHIJK and Bearer abc.def.ghi and api_key=supersecret")
    assert "sk-***" in s
    assert "Bearer ***" in s
    assert "api_key=***" in s
    assert "supersecret" not in s


def test_resolve_session_path_by_id():
    p = resolve_session_path(SAMPLE_ROOT, session_id=SESSION_ID)
    assert p.resolve() == SAMPLE_FILE.resolve()


def test_audit_status(engine: SelfDirectEngine):
    engine.sync_session(session_id=SESSION_ID)
    st = engine.audit_status(session_id=SESSION_ID)
    assert st["ok"] is True
    assert st["session"]["chunk_count"] >= 1
    assert "dense_numpy_fallback" in st["degraded_flags"] or st["dense_backend"] in (
        "numpy",
        "sqlite-vector",
    )
