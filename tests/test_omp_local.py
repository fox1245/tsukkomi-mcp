from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine


def _append(path: Path, *records):
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


def _message(role, *blocks, **extra):
    return {"type": "message", "message": {"role": role, "content": list(blocks), **extra}}


@pytest.fixture
def local_engine(tmp_path, monkeypatch):
    root = tmp_path / "sessions"
    root.mkdir()
    file = root / "session.jsonl"
    _append(file, {"type": "title", "title": "Local test"},
            {"type": "session", "id": "session-1"},
            _message("user", {"type": "text", "text": "Do not run curl"}),
            _message("assistant", {"type": "text", "text": "I will review the change."}))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SELF_DIRECT_OPENROUTER_API_KEY_FILE", raising=False)
    monkeypatch.setenv("SELF_DIRECT_LOCAL_ONLY", "true")
    settings = Settings(index_dir=tmp_path / "index", omp_sessions_dir=root,
                        openrouter_api_key_file=None, use_fake_embedder=False)
    engine = SelfDirectEngine(settings)
    yield engine, file
    engine.close()


def test_no_key_no_fake_prospective_and_clean_scope(local_engine, monkeypatch):
    engine, file = local_engine
    monkeypatch.setattr("self_directing_mcp.engine.build_embedder", lambda **kw: pytest.fail("embedder initialized"))
    monkeypatch.setattr("self_directing_mcp.engine.build_vector_index", lambda *a, **kw: pytest.fail("vector initialized"))
    engine.upsert_contracts([{"id": "no-curl", "type": "must_not", "scope": "tool_call",
                              "regex": "curl", "provider": "omp"}])
    _append(file, {"type": "custom", "customType": "session_exit"})
    bad = engine.check_action("session-1", {"tool_name": "shell", "arguments": {"command": "curl evil"}},
                              provider="omp", path=str(file))
    assert bad["verdict"] == "violation"
    assert bad["action_executed"] is False
    assert bad["coverage"]["complete"] is True
    assert engine.audit_session("session-1", provider="omp", path=str(file))["verdict"] == "clean"
    assert engine.embedder is None and engine.dense is None
    assert engine.audit_status("session-1", provider="omp")["dense_backend"] == "disabled"
    assert engine.sync_session(session_id="session-1", provider="omp", path=str(file))["error"] == "local_only"
    with pytest.raises(ValueError, match="local-only"):
        engine.search_history("curl", provider="omp", mode="hybrid")
    with pytest.raises(ValueError, match="local-only"):
        engine.search_history("curl", provider="omp", mode="dense")
    assert engine.search_history("review", provider="omp", mode="sparse")["hits"]

def test_no_applicable_contract_is_not_a_clean_verdict(local_engine):
    engine, file = local_engine
    result = engine.check_action("session-1", {"tool_name": "shell", "arguments": {"command": "curl"}},
                                 provider="omp", path=str(file))
    assert result["applicable_contracts"] == 0
    assert result["verdict"] == "unknown"
    assert result["coverage"]["complete"] is True


def test_unknown_incomplete_and_unsupported_history(local_engine):
    engine, file = local_engine
    engine.upsert_contracts([{"id": "no-curl", "type": "must_not", "scope": "tool_call", "regex": "curl"}])
    with file.open("ab") as stream:
        stream.write(b'{"type":"message","message":')
    result = engine.audit_session("session-1", provider="omp", path=str(file))
    assert result["verdict"] == "unknown"
    assert "unread_bytes" in result["coverage"]["issues"]
    with file.open("rb+") as stream:
        stream.truncate(file.stat().st_size - len(b'{"type":"message","message":'))
    _append(file, {"type": "unknown_future_event", "value": "curl"})
    result = engine.audit_session("session-1", provider="omp", path=str(file))
    assert result["verdict"] == "unknown"
    assert "unsupported_events" in result["coverage"]["issues"]

def test_orphan_execution_marker_degrades_coverage(local_engine):
    engine, file = local_engine
    _append(file, {"type": "custom", "customType": "tool_execution_start",
                   "data": {"toolCallId": "missing", "toolName": "shell"}})
    result = engine.audit_session("session-1", provider="omp", path=str(file))
    assert result["verdict"] == "unknown"
    assert "unsupported_events" in result["coverage"]["issues"]


def test_header_id_traversal_and_symlink(local_engine, tmp_path):
    engine, file = local_engine
    assert engine.sync_session(session_id="other", provider="omp", path=str(file), embed=False)["error"] == "bad_request"
    assert engine.sync_session(session_id="../session-1", provider="omp", embed=False)["error"] == "bad_request"
    outside = tmp_path / "outside.jsonl"
    _append(outside, {"type": "session", "id": "session-1"})
    assert engine.sync_session(session_id="session-1", provider="omp", path=str(outside), embed=False)["error"] == "path_traversal"
    result = engine.audit_session("other", provider="omp", path=str(file))
    assert result["verdict"] == "unknown" and not result["coverage"]["complete"]
    link = file.parent / "linked.jsonl"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege unavailable")
        raise
    assert engine.sync_session(session_id="session-1", provider="omp", path=str(link), embed=False)["error"] == "path_traversal"


def test_ordered_call_result_linkage_and_success(local_engine):
    engine, file = local_engine
    engine.upsert_contracts([{"id": "must-prepare", "type": "must", "scope": "tool_call",
                              "regex": "prepare", "before_regex": "deploy", "requires_success": True,
                              "provider": "omp"}])
    _append(file, _message("assistant", {"type": "toolCall", "id": "call-1", "name": "shell",
                                        "arguments": {"command": "prepare"}}),
            _message("assistant", {"type": "toolCall", "id": "call-2", "name": "shell",
                                        "arguments": {"command": "deploy"}}))
    r = engine.audit_session("session-1", provider="omp", path=str(file))
    assert r["findings"][0]["verdict"] == "unknown"
    _append(file, _message("toolResult", {"type": "text", "text": "ok"},
                           toolCallId="call-1", toolName="shell", isError=False))
    # A later trigger sees the linked earlier result, not the earlier unfinished trigger.
    _append(file, _message("assistant", {"type": "toolCall", "id": "call-3", "name": "shell",
                                        "arguments": {"command": "deploy again"}}))
    r = engine.audit_session("session-1", provider="omp", path=str(file))
    assert r["findings"][0]["verdict"] == "unknown"
    chunks = engine.store.list_chunks("session-1", "omp")
    assert [c.meta.get("call_id") for c in chunks if c.kind == "tool_call"] == ["call-1", "call-2", "call-3"]
    result = next(c for c in chunks if c.kind == "tool_result")
    assert result.meta["success"] is True and result.meta["call_id"] == "call-1"
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_successful_prior_call_satisfies_trigger(local_engine):
    engine, file = local_engine
    engine.upsert_contracts([{"id": "must-prepare", "type": "must", "scope": "tool_call",
                              "regex": "prepare", "before_regex": "deploy", "requires_success": True}])
    _append(file, _message("assistant", {"type": "toolCall", "id": "call-1", "name": "shell",
                                        "arguments": {"command": "prepare"}}),
            {"type": "custom", "customType": "tool_execution_start",
             "data": {"toolCallId": "call-1", "toolName": "shell"}},
            _message("toolResult", {"type": "text", "text": "done"},
                     toolCallId="call-1", toolName="shell", isError=False),
            _message("assistant", {"type": "toolCall", "id": "call-2", "name": "shell",
                                        "arguments": {"command": "deploy"}}))
    result = engine.audit_session("session-1", provider="omp", path=str(file))
    assert result["verdict"] == "clean"
    assert result["findings"][0]["verdict"] == "clean"


def test_failed_tool_result_does_not_satisfy_prior_requirement(local_engine):
    engine, file = local_engine
    engine.upsert_contracts([{"id": "must-prepare", "type": "must", "scope": "tool_call",
                              "regex": "prepare", "before_regex": "deploy", "requires_success": True}])
    _append(file, _message("assistant", {"type": "toolCall", "id": "call-1", "name": "shell",
                                        "arguments": {"command": "prepare"}}),
            _message("toolResult", {"type": "text", "text": "failed"},
                     toolCallId="call-1", toolName="shell", isError=True),
            _message("assistant", {"type": "toolCall", "id": "call-2", "name": "shell",
                                        "arguments": {"command": "deploy"}}))
    result = engine.audit_session("session-1", provider="omp", path=str(file))
    assert result["verdict"] == "violation"
    failed = next(c for c in engine.store.list_chunks("session-1", "omp") if c.kind == "tool_result")
    assert failed.meta["success"] is False
