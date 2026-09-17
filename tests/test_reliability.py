from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_directing_mcp.codex.parse import parse_session_chunks as codex_parse
from self_directing_mcp.grokbot.parse import parse_session_chunks as grok_parse
from self_directing_mcp.config import Settings
from self_directing_mcp.embed.embedder import FakeEmbedder
from self_directing_mcp.engine import SelfDirectEngine

SID = "11111111-2222-3333-4444-555555555555"


def message(text, role="assistant"):
    return {"type": "response_item", "payload": {"type": "message", "role": role, "content": text}}


def call(command, call_id="call-1"):
    return {"type": "response_item", "payload": {"type": "function_call", "name": "shell", "call_id": call_id, "arguments": {"command": command}}}


def result(call_id, exit_code=0):
    return {"type": "response_item", "payload": {"type": "function_call_output", "call_id": call_id, "output": json.dumps({"exit_code": exit_code, "output": "test output"})}}


def grok_message(text):
    return {"role": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


def write(path, events, mode="w"):
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        for event in events:
            stream.write(json.dumps(event) + "\n")


@pytest.fixture
def session(tmp_path, monkeypatch):
    for name in ("SELF_DIRECT_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER", "SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl"
    write(path, [{"type": "session_meta", "payload": {"id": SID}}])
    engine = SelfDirectEngine(Settings(codex_sessions_dir=root, index_dir=tmp_path / "index", use_fake_embedder=True, _env_file=None))
    yield engine, path


@pytest.mark.parametrize("parser,event", [(codex_parse, message), (grok_parse, grok_message)])
def test_first_append_and_partial_line(parser, event, tmp_path):
    path = tmp_path / "events.jsonl"
    write(path, [event("initial")])
    _, offset, _ = parser(path, session_id=SID)
    write(path, [event("first"), event("second")], "a")
    chunks, offset, _ = parser(path, session_id=SID, start_byte=offset)
    assert len(chunks) == 2
    assert "first" in chunks[0].text
    encoded = (json.dumps(event("partial")) + "\n").encode()
    with path.open("ab") as stream:
        stream.write(encoded[:12])
    chunks, incomplete_offset, _ = parser(path, session_id=SID, start_byte=offset)
    assert chunks == []
    assert incomplete_offset == offset
    with path.open("ab") as stream:
        stream.write(encoded[12:])
    chunks, _, _ = parser(path, session_id=SID, start_byte=incomplete_offset)
    assert len(chunks) == 1 and "partial" in chunks[0].text


def test_long_grok_command_preserved(tmp_path):
    path = tmp_path / "events.jsonl"
    command = "echo " + "x" * 400 + " && rm -rf /fictional-test-path"
    write(path, [{"role": "assistant", "message": {"content": [{"type": "tool_use", "id": "c1", "name": "Shell", "input": {"command": command}}]}}])
    chunks, _, _ = grok_parse(path, session_id=SID)
    assert command in chunks[0].text
    assert chunks[0].meta["call_id"] == "c1"


class RecordingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.documents = []

    def embed_documents(self, texts):
        self.documents.extend(texts)
        return super().embed_documents(texts)


def test_events_preserved_and_embeddings_reused(session):
    engine, path = session
    write(path, [call("echo repeat", "a"), call("echo repeat", "b"), call("echo repeat", "c")], "a")
    engine.ensure_ready()
    spy = RecordingEmbedder()
    engine.embedder = spy
    first = engine.sync_session(session_id=SID)
    events = [c for c in engine.store.list_chunks(SID) if c.kind == "tool_call"]
    assert len(events) == 3
    assert len({c.chunk_id for c in events}) == 3
    assert first["embedded"] < first["new_chunks"]
    count = len(spy.documents)
    ids = [c.chunk_id for c in events]
    assert engine.sync_session(session_id=SID)["embedded"] == 0
    assert len(spy.documents) == count
    assert [c.chunk_id for c in engine.store.list_chunks(SID) if c.kind == "tool_call"] == ids
    write(path, [call("echo repeat", "d")], "a")
    assert engine.sync_session(session_id=SID)["embedded"] == 0
    assert len([c for c in engine.store.list_chunks(SID) if c.kind == "tool_call"]) == 4


def test_no_auto_seed_and_no_rank_based_stop(session):
    engine, path = session
    write(path, [call("echo hello")], "a")
    engine.sync_session(session_id=SID)
    assert engine.list_contracts()["contracts"] == []
    assert engine.audit_session(SID)["verdict"] == "unknown"
    engine.upsert_contracts([{"id": "no-delete", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf", "description": "do not delete files"}])
    assert engine.audit_session(SID)["verdict"] == "clean"
    engine.upsert_contracts([{"id": "garden", "type": "must_not", "description": "planting roses in the garden"}])
    findings = {f["contract_id"]: f for f in engine.audit_session(SID)["findings"]}
    assert findings["garden"]["verdict"] == "unknown"
    assert findings["no-delete"]["verdict"] == "clean"


def test_external_documents_redacted_local_evidence_retained(session):
    engine, path = session
    token = "sk-FAKE_REVIEW_SENTINEL_123456789"
    write(path, [call(f"echo {token}")], "a")
    engine.ensure_ready()
    spy = RecordingEmbedder()
    engine.embedder = spy
    engine.sync_session(session_id=SID)
    assert all(token not in t for t in spy.documents)
    chunk = next(c for c in engine.store.list_chunks(SID) if c.kind == "tool_call")
    assert token in chunk.text
    assert token not in json.dumps(engine.get_chunk(chunk.chunk_id))


def test_action_checks_refresh_and_do_not_execute_or_store(session):
    engine, path = session
    engine.upsert_contracts([{"id": "no-delete", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"}])
    engine.sync_session(session_id=SID)
    before = engine.store.chunk_count(SID)
    checked = engine.check_action(SID, {"tool_name": "shell", "arguments": {"command": "rm -rf /fictional-test-path"}})
    assert checked["verdict"] == "violation"
    assert checked["action_executed"] is False
    assert engine.store.chunk_count(SID) == before
    assert checked["coverage"]["complete"] is True
    assert checked["contracts_snapshot"]
    write(path, [call("rm -rf /fictional-test-path")], "a")
    assert engine.audit_session(SID)["verdict"] == "violation"


def test_roles_and_activation_anchor(session):
    engine, path = session
    write(path, [call("rm -rf /fictional-test-path", "old"), message("new policy", "user")], "a")
    engine.sync_session(session_id=SID)
    anchor = engine.store.list_chunks(SID)[-1].chunk_id
    engine.upsert_contracts([{"id": "new-no-delete", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf", "source_event_id": anchor, "applies_from_event_id": anchor}])
    assert engine.audit_session(SID)["verdict"] == "clean"
    write(path, [call("rm -rf /fictional-test-path", "new")], "a")
    assert engine.audit_session(SID)["verdict"] == "violation"


def test_user_text_does_not_satisfy_assistant_obligation(session):
    engine, path = session
    write(path, [message("say VERIFIED", "user")], "a")
    engine.upsert_contracts([{"id": "ack", "type": "must", "scope": "message", "regex": "VERIFIED", "roles": ["assistant"]}])
    assert engine.audit_session(SID)["verdict"] == "violation"
    write(path, [message("VERIFIED", "assistant")], "a")
    assert engine.audit_session(SID)["verdict"] == "clean"


@pytest.mark.parametrize("exit_code,expected", [(0, "clean"), (1, "violation"), (None, "unknown")])
def test_successful_test_required_before_deploy(session, exit_code, expected):
    engine, path = session
    events = [call("pytest", "tests")]
    if exit_code is not None:
        events.append(result("tests", exit_code))
    write(path, events, "a")
    engine.upsert_contracts([{"id": "test-before-deploy", "type": "must", "scope": "tool_call", "regex": "pytest", "before_regex": "deploy", "requires_success": True}])
    checked = engine.check_action(SID, {"tool_name": "shell", "arguments": {"command": "deploy"}})
    assert checked["verdict"] == expected


def test_later_test_does_not_fix_earlier_deploy(session):
    engine, path = session
    write(path, [call("deploy", "deploy"), call("pytest", "tests"), result("tests")], "a")
    engine.upsert_contracts([{"id": "test-before-deploy", "type": "must", "scope": "tool_call", "regex": "pytest", "before_regex": "deploy", "requires_success": True}])
    assert engine.audit_session(SID)["verdict"] == "violation"


def test_missing_anchor_and_incomplete_coverage_are_unknown(session):
    engine, path = session
    engine.upsert_contracts([{"id": "no-delete", "type": "must_not", "regex": "DELETE", "applies_from_event_id": "missing"}])
    assert engine.audit_session(SID)["verdict"] == "unknown"
    engine.upsert_contracts([{"id": "no-delete", "type": "must_not", "regex": "DELETE"}])
    with path.open("ab") as stream:
        stream.write(b'{"partial":')
    audit = engine.audit_session(SID)
    assert audit["verdict"] == "unknown"
    assert audit["coverage"]["complete"] is False


def test_malformed_line_never_yields_clean(session):
    engine, path = session
    engine.upsert_contracts([{"id": "no-delete", "type": "must_not", "regex": "DELETE"}])
    with path.open("ab") as stream:
        stream.write(b'{broken json}\n')
    audit = engine.audit_session(SID)
    assert audit["verdict"] == "unknown"
    assert audit["coverage"]["parse_error_count"] == 1


def test_rewrite_reindexes_and_invalidates_old_events(session):
    engine, path = session
    write(path, [call("old command")], "a")
    engine.sync_session(session_id=SID)
    write(path, [{"type": "session_meta", "payload": {"id": SID}}, call("new command")])
    synced = engine.sync_session(session_id=SID)
    assert synced["reindexed"] is True
    text = " ".join(c.text for c in engine.store.list_chunks(SID))
    assert "old command" not in text and "new command" in text


def test_tool_output_is_not_a_call_and_preserves_result(session):
    engine, path = session
    write(path, [call("pytest", "tests"), result("tests")], "a")
    engine.sync_session(session_id=SID)
    chunks = engine.store.list_chunks(SID)
    assert sum(c.kind == "tool_call" for c in chunks) == 1
    output = next(c for c in chunks if c.kind == "tool_result")
    assert output.meta["call_id"] == "tests"
    assert output.meta["success"] is True


def test_embedding_outage_does_not_block_local_audit(session):
    engine, path = session
    write(path, [call("rm -rf /fictional-test-path")], "a")
    engine.ensure_ready()
    class BrokenEmbedder(RecordingEmbedder):
        def embed_documents(self, texts):
            raise RuntimeError("synthetic network failure")
    engine.embedder = BrokenEmbedder()
    sync = engine.sync_session(session_id=SID)
    assert sync["embedding_error"] == "RuntimeError"
    assert engine.store.chunk_count(SID) == 2
    engine.upsert_contracts([{"id": "no-delete", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"}])
    assert engine.audit_session(SID)["verdict"] == "violation"
    engine.embedder = RecordingEmbedder()
    assert engine.sync_session(session_id=SID)["embedded"] == 2
    assert engine.dense.count() == 2


def test_dense_filter_happens_before_top_k(session):
    engine, path = session
    write(path, [call("TARGET_UNIQUE alpha")], "a")
    engine.sync_session(session_id=SID)
    other_sid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    other = path.parent / f"rollout-2026-09-09T01-00-00-{other_sid}.jsonl"
    write(other, [{"type": "session_meta", "payload": {"id": other_sid}}] +
          [call("other" + str(i) + " TARGET_UNIQUE alpha") for i in range(40)])
    engine.sync_session(session_id=other_sid)
    hits = engine.search_history("TARGET_UNIQUE alpha", session_id=SID, mode="dense", top_k=1)["hits"]
    assert len(hits) == 1
    assert engine.get_chunk(hits[0]["chunk_id"])["chunk"]["session_id"] == SID


def test_same_session_id_across_providers_is_isolated(session, tmp_path):
    engine, path = session
    write(path, [call("CODEX_ONLY")], "a")
    engine.sync_session(session_id=SID)
    grok_root = tmp_path / "grok"
    directory = grok_root / SID
    directory.mkdir(parents=True)
    grok_file = directory / f"{SID}.jsonl"
    write(grok_file, [grok_message("GROK_ONLY")])
    engine.settings.grokbot_transcripts_dir = str(grok_root)
    engine.sync_session(session_id=SID, provider="grokbot")
    codex = engine.search_history("ONLY", mode="regex", session_id=SID, provider="codex")["hits"]
    grok = engine.search_history("ONLY", mode="regex", session_id=SID, provider="grokbot")["hits"]
    assert len(codex) == len(grok) == 1
    assert "CODEX_ONLY" in codex[0]["snippet"] and "GROK_ONLY" in grok[0]["snippet"]
    assert engine.audit_status(SID, provider="grokbot")["session"]["chunk_count"] == 1


def test_legacy_tables_are_retained_and_replayed(session):
    engine, path = session
    import sqlite3
    directory = Path(engine.settings.index_dir)
    directory.mkdir()
    with sqlite3.connect(directory / "meta.sqlite") as db:
        db.execute("CREATE TABLE chunks (chunk_id TEXT, text TEXT)")
        db.execute("INSERT INTO chunks VALUES ('old-id', 'retained evidence')")
    write(path, [call("repeat", "a"), call("repeat", "b")], "a")
    engine.sync_session(session_id=SID)
    assert engine.audit_status()["legacy_chunks_retained"] == 1
    assert engine.store.chunk_count(SID) == 3
    with sqlite3.connect(directory / "meta.sqlite") as db:
        assert db.execute("SELECT text FROM chunks").fetchone()[0] == "retained evidence"


def test_concurrent_engine_calls_are_serialized(session):
    from concurrent.futures import ThreadPoolExecutor
    engine, path = session
    write(path, [call("echo hello")], "a")
    engine.upsert_contracts([{"id": "no-delete", "type": "must_not", "regex": "DELETE"}])
    def work(i):
        if i % 2:
            return engine.sync_session(session_id=SID)["ok"]
        return engine.audit_session(SID)["verdict"] == "clean"
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(work, range(12)))
    assert engine.store.chunk_count(SID) == 2


def test_contract_revisions_and_snapshot_change(session):
    engine, path = session
    engine.upsert_contracts([{"id": "r", "type": "must_not", "regex": "first"}])
    first = engine.audit_session(SID)
    engine.upsert_contracts([{"id": "r", "type": "must_not", "regex": "second"}])
    second = engine.audit_session(SID)
    assert first["contracts_snapshot"] != second["contracts_snapshot"]
    assert second["findings"][0]["contract_revision"] == 2


def test_latest_test_failure_overrides_earlier_success(session):
    engine, path = session
    write(path, [call("pytest", "a"), result("a"), call("pytest", "b"), result("b", 1)], "a")
    engine.upsert_contracts([{"id": "test-before-deploy", "type": "must", "scope": "tool_call",
                              "regex": "pytest", "before_regex": "deploy", "requires_success": True}])
    assert engine.check_action(SID, {"tool_name": "shell", "arguments": "deploy"})["verdict"] == "violation"


@pytest.mark.parametrize("text", [
    '{"api_key": "synthetic_plain_token_123456"}',
    '{"password": "synthetic_plain_token_123456"}',
    'Authorization: Bearer synthetic_plain_token_123456',
])
def test_structured_secrets_masked_before_embedding(session, text):
    engine, path = session
    write(path, [message(text)], "a")
    engine.ensure_ready()
    spy = RecordingEmbedder()
    engine.embedder = spy
    engine.sync_session(session_id=SID)
    assert all("synthetic_plain_token_123456" not in text for text in spy.documents)


def test_external_query_redacted_before_embedder(session):
    engine, path = session
    engine.sync_session(session_id=SID)
    class QuerySpy(RecordingEmbedder):
        def embed_queries(self, texts):
            assert all("FAKE_QUERY_TOKEN_123456" not in text for text in texts)
            return super().embed_queries(texts)
    engine.retriever.embedder = QuerySpy()
    result = engine.search_history("sk-FAKE_QUERY_TOKEN_123456", session_id=SID)
    assert "FAKE_QUERY_TOKEN_123456" not in result["query"]


def test_incomplete_log_cannot_prove_missing_obligation(session):
    engine, path = session
    engine.upsert_contracts([{"id": "ack", "type": "must", "scope": "message", "regex": "ACK"}])
    with path.open("ab") as stream:
        stream.write(b'{"partial":')
    audit = engine.audit_session(SID)
    assert audit["verdict"] == "unknown"
    assert audit["findings"][0]["verdict"] == "unknown"


def test_tool_completion_metadata_rewrite_invalidates_event_id(session):
    engine, path = session
    write(path, [call("pytest", "a")], "a")
    engine.sync_session(session_id=SID)
    old = engine.store.list_chunks(SID)[-1].chunk_id
    write(path, [{"type": "session_meta", "payload": {"id": SID}}, call("pytest", "b")])
    engine.sync_session(session_id=SID)
    assert engine.store.list_chunks(SID)[-1].chunk_id != old
