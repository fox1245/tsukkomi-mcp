"""Codex parser v3: multi-agent messaging, tool search and new-style session headers (synthetic)."""
from __future__ import annotations

import json

import pytest

from self_directing_mcp.codex import parse
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine

SID = "11111111-2222-3333-4444-555555555555"
PARENT = "99999999-2222-3333-4444-555555555555"
SECRETS = ("ENCSECRET", "BASESECRET", "DYNSECRET", "CREATORSECRET", "TOOLSLISTSECRET", "GITSECRET",
           "HISTSECRET", "CTXSECRET", "ROOTSSECRET")


def write(path, events):
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


@pytest.fixture
def session(tmp_path):
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / f"rollout-{SID}.jsonl"
    engine = SelfDirectEngine(Settings(codex_sessions_dir=root, index_dir=tmp_path / "index",
                                      use_fake_embedder=True, _env_file=None))
    try:
        yield engine, path
    finally:
        engine.close()


def header(sid=SID, **over):
    payload = {"id": sid, "session_id": sid, "timestamp": "2026-01-01T00:00:00Z", "cwd": "/work",
               "originator": "codex_cli", "cli_version": "1.2.3", "model_provider": "openai",
               "base_instructions": {"text": "BASESECRET"}, "context_window": {"w": "CTXSECRET"},
               "history_mode": "paginated", **over}
    return {"type": "session_meta", "payload": {k: v for k, v in payload.items() if v is not ...}}


def user(text):
    return {"type": "response_item", "payload": {"type": "message", "role": "user", "content": text}}


def agent_msg(content=None, *, top=None, **over):
    content = [{"type": "input_text", "text": "zzforbidden zzsatisfy hello"}] if content is None else content
    payload = {"type": "agent_message", "id": "m1", "author": "/root", "recipient": "/root/worker",
               "content": content,
               "internal_chat_message_metadata_passthrough": {"create_time": "t", "turn_id": "u"}, **over}
    return {"timestamp": "2026-01-01T00:00:01Z", "type": "response_item", "ordinal": 3,
            "payload": {k: v for k, v in payload.items() if v is not ...}, **(top or {})}


def search_call(**over):
    payload = {"type": "tool_search_call", "id": "ts1", "call_id": "c1", "status": "completed",
               "execution": "client", "arguments": {"query": "find the formatter", "limit": 5},
               "internal_chat_message_metadata_passthrough": {"turn_id": "u"}, **over}
    return {"type": "response_item", "payload": {k: v for k, v in payload.items() if v is not ...}}


def search_out(**over):
    payload = {"type": "tool_search_output", "id": "ts2", "call_id": "c1", "status": "completed",
               "execution": "client", "tools": [{"name": "TOOLSLISTSECRET"}],
               "internal_chat_message_metadata_passthrough": {"turn_id": "u"}, **over}
    return {"type": "response_item", "payload": {k: v for k, v in payload.items() if v is not ...}}


def iac(**over):
    return {"type": "inter_agent_communication_metadata", "payload": {"trigger_turn": False, **over}}


FULL_HEADER = header(thread_source="subagent", agent_nickname="n", agent_path="/root/a", multi_agent_version="2",
                     parent_thread_id=PARENT, forked_from_id=PARENT, creator_account_id="CREATORSECRET",
                     creator_user_id="CREATORSECRET", git={"branch": "GITSECRET"}, history_base={"h": "HISTSECRET"},
                     dynamic_tools=[{"name": "DYNSECRET"}], runtime_workspace_roots=["ROOTSSECRET"],
                     source={"kind": "vscode"}, subagent_history_start_ordinal=7)

VALID = [
    iac(), iac(trigger_turn=True),
    agent_msg(), agent_msg([]), agent_msg([{"type": "encrypted_content", "encrypted_content": "ENCSECRET"}]),
    agent_msg([{"type": "input_text", "text": "a"}, {"type": "encrypted_content", "encrypted_content": "ENCSECRET"}]),
    agent_msg(top={"metadata": {"client_authored": False, "user_input_order": None}}),
    agent_msg(top={"metadata": {"client_authored": False, "user_input_order": 4}}),
    agent_msg(internal_chat_message_metadata_passthrough=...), agent_msg(internal_chat_message_metadata_passthrough={}),
    agent_msg(author="/root", recipient="/root"),
    search_call(), search_call(arguments={"query": "q"}), search_out(),
    header(), FULL_HEADER, header(source="cli"), header(thread_source="user"),
]


def sync(engine):
    return engine.sync_session(session_id=SID, embed=False)


@pytest.mark.parametrize("record", VALID)
def test_valid_shapes_are_supported_and_complete(session, record):
    engine, path = session
    write(path, [record] if record["type"] == "session_meta" else [header(), record])
    result = sync(engine)
    assert result["coverage"]["complete"] is True, result["coverage"]
    chunks = engine.store.list_chunks(SID)
    assert not any(c.meta.get("unsupported_event") or c.meta.get("parse_error") for c in chunks)
    blob = " ".join(c.text + json.dumps(c.meta, default=str) for c in chunks)
    assert not [s for s in SECRETS if s in blob]


def test_runtime_classification(session):
    engine, path = session
    write(path, [header(), iac(), agent_msg([{"type": "input_text", "text": "hi"},
                                              {"type": "encrypted_content", "encrypted_content": "ENCSECRET"}]),
                 search_call(), search_out()])
    sync(engine)
    _hdr, meta_chunk, msg, call, out = engine.store.list_chunks(SID)
    for chunk in (meta_chunk, msg, call, out):
        assert chunk.kind == "meta" and chunk.meta["role"] == "runtime"
    assert meta_chunk.meta["evidence_origin"] == "inter_agent_metadata" and "ENCSECRET" not in meta_chunk.text
    assert msg.meta["evidence_origin"] == "agent_message"
    assert msg.text.startswith("[agent_message:/root->/root/worker] hi") and "[encrypted_content]" in msg.text
    assert call.meta["evidence_origin"] == out.meta["evidence_origin"] == "tool_search"
    assert "find the formatter" in call.text and "TOOLSLISTSECRET" not in out.text


@pytest.mark.parametrize("record", [
    iac(trigger_turn="no"), iac(trigger_turn=1), iac(extra=1), {"type": "inter_agent_communication_metadata", "payload": {}},
    {"type": "inter_agent_communication_metadata", "payload": []},
    agent_msg(extra=1), agent_msg(author=5), agent_msg(recipient=None), agent_msg(author=...),
    agent_msg(recipient=...), agent_msg(content="plain"), agent_msg(content=...),
    agent_msg([{"type": "output_text", "text": "x"}]),
    agent_msg([{"type": "input_text", "text": 5}]),
    agent_msg([{"type": "input_text", "text": "x", "extra": 1}]),
    agent_msg([{"type": "encrypted_content", "encrypted_content": 5}]),
    agent_msg([{"type": "encrypted_content"}]), agent_msg(["bare"]),
    agent_msg(internal_chat_message_metadata_passthrough=[]),
    agent_msg(internal_chat_message_metadata_passthrough={"extra": 1}),
    agent_msg(top={"metadata": {"client_authored": True, "user_input_order": None}}),
    agent_msg(top={"metadata": {"client_authored": "no"}}),
    agent_msg(top={"metadata": {"client_authored": False, "user_input_order": "1"}}),
    agent_msg(top={"metadata": {"client_authored": False, "extra": 1}}),
    agent_msg(top={"metadata": "x"}), agent_msg(top={"ordinal": "3"}), agent_msg(top={"extra": 1}),
    search_call(extra=1), search_call(arguments={"query": 5}), search_call(arguments={"query": "q", "x": 1}),
    search_call(arguments={"limit": 1}), search_call(arguments="q"), search_call(call_id=...),
    search_call(status=5), search_call(execution=None),
    search_call(internal_chat_message_metadata_passthrough={"extra": 1}),
    search_out(extra=1), search_out(tools="x"), search_out(tools=...), search_out(call_id=5),
    search_out(internal_chat_message_metadata_passthrough={"extra": 1}),
])
def test_malformed_shapes_stay_unsupported(session, record):
    engine, path = session
    write(path, [header(), record])
    result = sync(engine)
    assert result["coverage"]["complete"] is False
    assert not [c for c in engine.store.list_chunks(SID)[1:] if c.kind == "turn"]


@pytest.mark.parametrize("fields", [
    {"cwd": 5}, {"originator": None}, {"cli_version": 1}, {"model_provider": []}, {"timestamp": 5},
    {"base_instructions": "str"}, {"context_window": []}, {"history_mode": 1},
    {"base_instructions": ...}, {"context_window": ...}, {"history_mode": ...}, {"cwd": ...}, {"session_id": ...},
    {"thread_source": 5}, {"agent_nickname": 5}, {"agent_path": []}, {"multi_agent_version": 2},
    {"parent_thread_id": 5}, {"forked_from_id": {}}, {"creator_account_id": 5}, {"creator_user_id": []},
    {"git": "x"}, {"history_base": []}, {"dynamic_tools": {}}, {"runtime_workspace_roots": "x"},
    {"source": 5}, {"source": []}, {"subagent_history_start_ordinal": "1"},
    {"subagent_history_start_ordinal": True},
])
def test_malformed_new_style_session_meta_is_unsupported(session, fields):
    engine, path = session
    write(path, [header(**fields)])
    result = sync(engine)
    assert result["coverage"]["complete"] is False
    assert engine.store.list_chunks(SID)[0].meta["unsupported_event"] is True


def test_session_meta_never_indexes_sensitive_or_unknown_keys(session):
    engine, path = session
    write(path, [header(**{**FULL_HEADER["payload"], "future_key": "FUTURESECRET"})])
    assert sync(engine)["coverage"]["complete"] is True
    (chunk,) = engine.store.list_chunks(SID)
    assert SID in chunk.text and "/work" in chunk.text
    assert not [s for s in (*SECRETS, "FUTURESECRET") if s in chunk.text + json.dumps(chunk.meta, default=str)]


def test_session_id_resolution_is_unchanged(session, tmp_path):
    engine, path = session
    write(path, [header(sid=SID, session_id=PARENT)])
    # The header id (not session_id) is the identity, exactly as for legacy headers.
    assert parse.peek_session_id(path) == SID
    assert sync(engine)["coverage"]["complete"] is True
    other = tmp_path / "sessions" / "rollout-other.jsonl"
    write(other, [header(sid=PARENT, session_id=SID)])
    assert parse.peek_session_id(other) == PARENT
    chunks, _, sid = parse.parse_session_chunks(other)
    assert sid == PARENT and not chunks[0].meta.get("unsupported_event")
    # A first-line header for another session is still rejected.
    chunks, _, _ = parse.parse_session_chunks(path, session_id=PARENT)
    assert chunks[0].meta.get("parse_error")


def test_inherited_parent_header_after_start_is_runtime_meta(session):
    engine, path = session
    write(path, [header(thread_source="subagent", parent_thread_id=PARENT, subagent_history_start_ordinal=2),
                 header(sid=PARENT)])
    result = sync(engine)
    assert result["coverage"]["complete"] is True
    first, inherited = engine.store.list_chunks(SID)
    assert first.meta["evidence_origin"] == "session_meta"
    assert inherited.kind == "meta" and inherited.meta["role"] == "runtime"
    assert inherited.meta["evidence_origin"] == "inherited_session_meta"


def test_foreign_header_stays_a_parse_error_unless_new_style_and_not_first(session):
    engine, path = session
    legacy_foreign = {"type": "session_meta", "payload": {"id": PARENT}}
    write(path, [header(), legacy_foreign, header(sid=PARENT, cwd=5)])
    chunks, _, _ = parse.parse_session_chunks(path, session_id=SID)
    assert [bool(c.meta.get("parse_error")) for c in chunks] == [False, True, True]


def test_agent_messages_and_tool_search_never_satisfy_or_trigger_user_contracts(session):
    engine, path = session
    engine.upsert_contracts([
        {"id": "no-zz", "type": "must_not", "scope": "message", "regex": "zzforbidden", "roles": ["user"]},
        {"id": "need-zz", "type": "must", "scope": "message", "regex": "zzsatisfy", "roles": ["user"]},
        {"id": "no-formatter", "type": "must_not", "scope": "message", "regex": "find the formatter",
         "roles": ["user"]},
    ])
    write(path, [header(), agent_msg(), search_call(), user("Please proceed")])
    audit = engine.audit_session(SID)
    found = {f["contract_id"]: f for f in audit["findings"]}
    assert audit["coverage"]["complete"] is True
    assert found["no-zz"]["verdict"] == found["no-formatter"]["verdict"] == "clean"
    assert found["need-zz"]["verdict"] == "violation" and not found["need-zz"]["evidence"]
    write(path, [header(), agent_msg(), search_call(), user("Please proceed zzforbidden zzsatisfy")])
    audit = engine.audit_session(SID)
    found = {f["contract_id"]: f for f in audit["findings"]}
    assert found["no-zz"]["verdict"] == "violation"
