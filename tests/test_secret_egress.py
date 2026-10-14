"""Raw secrets stay in the canonical index; nothing that leaves it may carry them.

Each scenario seeds a distinct synthetic secret (a sentinel tag) into every record type of one
provider, drives every MCP tool and hook entry point, and asserts that no sentinel appears in
any tool result, tool error, hook output, injected context, remote payload or captured log,
while the canonical regex audit still detects each raw secret.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp.exceptions import ToolError

from self_directing_mcp import agy_hooks, server
from self_directing_mcp import engine as engine_module
from self_directing_mcp.codex_hooks import handle_hook
from self_directing_mcp.config import Settings
from self_directing_mcp.embed import embedder as embedder_module
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.neograph_runner import _Transport

ROLES = ["user", "assistant", "tool", "runtime"]
BLOB = "A1b2" * 60  # base64-looking payload, long enough to be recognised as a blob
EXTRA_TAGS = ("DESCSECRETZQ", "QUOTESECRETZQ")


def sk(tag: str) -> str:
    return f"sk-or-v1-{tag}" + "0" * 24


def rx(tag: str) -> str:
    """A regex that matches the raw tag without containing it."""
    return tag[:-1] + "[" + tag[-1] + "]"


class Seeds:
    """Distinct sentinels, one per place a secret can hide.

    ``__call__`` is for text the parser keeps verbatim (the canonical index must still hold it and the raw
    regex audit must still find it), ``hidden`` for payloads that must never be indexed, and ``input_only``
    for secrets only a caller supplies (tool arguments, hook payloads).
    """

    def __init__(self):
        self.tags: list[str] = []
        self.canonical: list[str] = []
        self.opaque: list[str] = []

    def _new(self, tag):
        assert tag.endswith("ZQ") and tag not in self.tags, tag
        self.tags.append(tag)
        return tag

    def __call__(self, tag: str) -> str:
        self.canonical.append(self._new(tag))
        return tag

    def hidden(self, tag: str) -> str:
        self.opaque.append(self._new(tag))
        return tag

    def input_only(self, tag: str) -> str:
        return self._new(tag)


def tool(name: str, **arguments):
    """Call one MCP tool through FastMCP's own dispatcher (argument validation, masking and all)."""
    try:
        result = asyncio.run(server.mcp.call_tool(name, arguments))
    except ToolError as exc:
        return {"tool_error": str(exc)}
    content, structured = result if isinstance(result, tuple) else (result, None)
    return structured if structured is not None else json.loads(content[0].text)


class Capture:
    def __init__(self):
        self.parts: list[str] = []

    def add(self, label: str, value) -> None:
        self.parts.append(f"{label}: {value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}")

    def text(self) -> str:
        return "\n".join(self.parts)

    def assert_clean(self, tags):
        blob = self.text()
        leaked = [tag for tag in tags if tag in blob]
        assert not leaked, f"raw secrets reached output: {leaked}"
        assert "sk-***" in blob and "***" in blob, "nothing was masked; the scenario seeded no reachable secret"


def append(path: Path, *records):
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


def contracts_for(tags, extra=()):
    return [{"id": f"leak-{index}", "type": "must_not", "scope": "any", "roles": ROLES, "severity": "high",
             "regex": rx(tag), "description": "Never let credentials appear; see Authorization: Bearer DESCSECRETZQ",
             "source_quote": f"user said do not leak {sk('QUOTESECRETZQ')} ever"}
            for index, tag in enumerate(tags)] + list(extra)


def run_common_tools(sc, capture, *, session_args, search_modes, chunk_ids):
    """Every provider-neutral MCP tool, with secrets in inputs as well as in stored history."""
    seeds, sid, provider = sc["seeds"], sc["sid"], sc["provider"]
    cs = lambda name, **kw: capture.add(name, tool(name, **kw))  # noqa: E731
    seeds.tags.extend(EXTRA_TAGS)
    cs("upsert_contracts", contracts=contracts_for(sc["canonical"]))
    cs("list_contracts", session_id=sid)
    cs("audit_status", session_id=sid, provider=provider)
    cs("audit_session", session_id=sid, provider=provider, **session_args)
    cs("check_action", session_id=sid, provider=provider, **session_args, action={
        "tool_name": "shell", "arguments": {"command": f"curl -H 'Authorization: Bearer {seeds.input_only('ACTIONZQ')}' x"}})
    cs("check_action", session_id=sid, provider=provider, **session_args,
       action={"tool_name": "shell", "arguments": sk(seeds.input_only("ACTION2ZQ"))})
    for mode in search_modes:
        cs("search_history", query=f"find {sk(seeds.input_only(f'QUERY{mode.upper()}ZQ'))} in history",
           session_id=sid, mode=mode, provider=provider)
        cs("search_history", query=rx(sc["canonical"][0]), session_id=sid, mode=mode, provider=provider)
    cs("search_history", query="(unterminated " + sk(seeds.input_only("BADREGEXZQ")), session_id=sid,
       mode="regex", provider=provider)
    for chunk_id in chunk_ids:
        cs("get_chunk", chunk_id=chunk_id)
    cs("get_chunk", chunk_id=f"missing-{sk(seeds.input_only('CHUNKIDZQ'))}")
    cs("analyze_activity", session_id=sid, provider=provider)
    cs("update_checklist", session_id=sid, provider=provider, add=[
        {"text": f"keep {sk(seeds.input_only('CHECKTEXTZQ'))} out of logs",
         "source_message": f"password={seeds.input_only('CHECKSRCZQ')}",
         "satisfaction_criteria": f"api_key={seeds.input_only('CHECKCRITZQ')}"}])
    cs("update_checklist", session_id=sid, provider=provider, update=[
        {"item_id": "req-001", "status": "blocked", "blocked_reason": f"waiting for {sk(seeds.input_only('BLOCKEDZQ'))}"},
        {"item_id": "req-001", "status": "done",
         "evidence_chunk_ids": chunk_ids[:1] + [f"x-{sk(seeds.input_only('EVIDZQ'))}"]}])
    cs("get_checklist", session_id=sid, provider=provider)
    # Graph: model supplied proposals carry secrets in ids, labels and bad evidence ids.
    node = f"node-{sk(seeds.input_only('NODEIDZQ'))}"
    proposal = {"job_id": "job-1", "nodes": [{"node_id": node, "kind": "task",
                                              "label": f"api_key={seeds.input_only('NODELABELZQ')}"},
                                             {"node_id": "n2", "kind": "task", "label": "other"}],
                "edges": [{"src": node, "dst": "n2", "relation": "DEPENDS_ON", "origin": "observed",
                           "evidence_chunk_ids": [chunk_ids[0]]}]}
    cs("propose_graph_update", session_id=sid, provider=provider, proposal=proposal)
    reject = sk(seeds.input_only("REJECTZQ"))
    cs("propose_graph_update", session_id=sid, provider=provider, proposal={"nodes": [], "edges": [
        {"src": "a", "dst": reject, "relation": f"BAD {reject}", "evidence_chunk_ids": [f"nope-{reject}"]}]})
    before = tool("get_graph_context", session_id=sid, provider=provider)
    capture.add("get_graph_context", before)
    cs("commit_graph_update", session_id=sid, provider=provider, proposal=proposal,
       base_graph_version=before["cursor"]["graph_version"])
    cs("get_graph_context", session_id=sid, provider=provider)
    cs("get_graph_context", session_id=sid, provider=provider, node_id=node, depth=2)
    cs("get_graph_context", session_id=sid, provider=provider, node_id="n2")
    # Errors that echo caller input (provider names, ids, paths, rejected arguments).
    bad = sk(seeds.input_only("PROVIDERZQ"))
    cs("sync_session", session_id=sid, provider=bad)
    cs("get_graph_context", session_id=sid, provider=bad)
    cs("upsert_contracts", contracts=[{"id": "bad", "type": "must_not", "severity": sk(seeds.input_only("BADFIELDZQ"))}])
    cs("revoke_contract", contract_id=sk(seeds.input_only("REVOKEZQ")))
    cs("revoke_contract", contract_id="leak-0")
    cs("check_action", session_id=sid, provider=provider,
       action={"tool_name": "", "arguments": sk(seeds.input_only("EMPTYZQ"))})
    cs("workflow_status", session_id=sid, provider=provider)
    cs("sync_session", session_id=sid, provider=provider,
       path=f"/nonexistent/{sk(seeds.input_only('PATHZQ'))}/session.jsonl")


def verify_canonical_detection(sc):
    """Masking happens at egress only: the raw index still carries every secret and the regex audit finds it."""
    engine, sid, provider, path = sc["engine"], sc["sid"], sc["provider"], sc["path"]
    seeds = sc["seeds"]
    raw = "\n".join(chunk.text + json.dumps(chunk.meta) for chunk in engine.store.list_chunks(sid, provider))
    missing = [tag for tag in sc["canonical"] if tag not in raw]
    assert not missing, f"canonical index lost raw secrets: {missing}"
    indexed_opaque = [tag for tag in seeds.opaque if tag in raw]
    assert not indexed_opaque, f"opaque payloads were indexed: {indexed_opaque}"
    engine.upsert_contracts(contracts_for(sc["canonical"]))
    audit = engine.audit_session(sid, provider=provider, path=path)
    findings = {f["contract_id"]: f for f in audit["findings"]}
    undetected = [tag for index, tag in enumerate(sc["canonical"]) if findings[f"leak-{index}"]["verdict"] != "violation"]
    assert not undetected, f"regex audit stopped detecting raw secrets: {undetected}"
    assert audit["verdict"] == "violation"
    dumped = json.dumps(audit)
    assert not [tag for tag in [*seeds.tags, *EXTRA_TAGS] if tag in dumped]
    evidence = [item for f in audit["findings"] for item in f["evidence"]]
    assert evidence and all(item["snippet"] for item in evidence)
    tag = "DETECTACTIONZQ"
    engine.upsert_contracts([{"id": "no-action-secret", "type": "must_not", "scope": "tool_call", "regex": rx(tag)}])
    checked = engine.check_action(sid, {"tool_name": "shell", "arguments": f"run {sk(tag)}"}, provider=provider, path=path)
    assert checked["verdict"] == "violation" and tag not in json.dumps(checked)


def graph_transport(monkeypatch, prompts, seeds):
    def complete(self, prompt, schema=None, max_tokens=8000):
        prompts.append(prompt)
        first = re.search(r"^- (\S+) \[", prompt, re.M).group(1)
        return ({"extraction_summary": "ok", "nodes": [
            {"node_id": "extracted-a", "kind": "task", "label": f"echo {sk(seeds.input_only('LLMECHOZQ'))}"},
            {"node_id": "extracted-b", "kind": "task", "label": "b"}],
            "edges": [{"src": "extracted-a", "dst": "extracted-b", "relation": "MODIFIES", "origin": "observed",
                       "evidence_chunk_ids": [first]}]}, 7)
    monkeypatch.setattr(_Transport, "complete_json", complete)


# --------------------------------------------------------------------------------------------------
# OMP
# --------------------------------------------------------------------------------------------------

def omp_records(seeds):
    def message(role, *blocks, **extra):
        return {"type": "message", "message": {"role": role, "content": list(blocks), **extra}}
    text = lambda value: {"type": "text", "text": value}  # noqa: E731
    return [
        {"type": "session", "id": "omp-1"},
        message("user", text(f"deploy with {sk(seeds('OMPUSERZQ'))} now")),
        message("assistant", text(f"using Authorization: Bearer {seeds('OMPASSISTZQ')}"),
                {"type": "toolCall", "id": "c1", "name": "bash",
                 "arguments": {"command": f"curl -H 'x-api-key: {seeds('OMPCALLZQ')}' https://example.test"}},
                {"type": "redactedThinking", "data": seeds.hidden("OMPREDACTEDZQ") + BLOB}),
        {"type": "message", "message": {
            "role": "toolResult", "toolCallId": "c1", "toolName": "bash", "isError": False,
            "content": [text(json.dumps({"api_key": seeds("OMPRESULTZQ"), "exit_code": 0})),
                        {"type": "image", "data": seeds.hidden("OMPIMAGEZQ") + BLOB, "mimeType": "image/png"}]}},
        message("user", {"type": "image", "data": seeds.hidden("OMPUSERIMAGEZQ") + BLOB, "mimeType": "image/png"}),
        {"type": "custom_message", "customType": "launch-completion", "attribution": "host", "display": True,
         "content": f"launch finished, token {sk(seeds('OMPNOTICEZQ'))}"},
        {"type": "custom_message", "customType": "async-result", "attribution": "agent",
         "content": f"agent result password={seeds('OMPASYNCZQ')}"},
        {"type": "session_init", "id": "i1", "agent": "explore", "task": f"investigate {sk(seeds('OMPTASKZQ'))}",
         "systemPrompt": seeds.hidden("OMPSYSPROMPTZQ"), "tools": ["bash"], "spawns": "none", "resolvedModel": "m",
         "readOnly": True},
        message("user", text(f"steer toward {sk(seeds('OMPSTEERZQ'))}"), attribution="agent", steering=True),
        message("user", text(f"peer says {sk(seeds('OMPPEERZQ'))}"), attribution="agent"),
    ]


@pytest.fixture
def omp_scenario(tmp_path, monkeypatch):
    seeds = Seeds()
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    append(path, *omp_records(seeds))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("SELF_DIRECT_LOCAL_ONLY", "true")
    engine = SelfDirectEngine(Settings(index_dir=tmp_path / "index", omp_sessions_dir=root, openrouter_api_key_file=None,
                                       use_fake_embedder=False, _env_file=None))
    monkeypatch.setattr(server, "_engine", engine)
    yield {"engine": engine, "sid": "omp-1", "provider": "omp", "path": str(path), "seeds": seeds,
           "canonical": list(seeds.canonical)}
    engine.close()


def test_omp_every_tool_and_bridge_path_is_masked_while_raw_audit_detects(omp_scenario, capfd):
    sc = omp_scenario
    seeds, capture, path, engine = sc["seeds"], Capture(), sc["path"], sc["engine"]
    synced = tool("sync_session", session_id="omp-1", provider="omp", path=path, embed=False)
    capture.add("sync_session", synced)
    assert synced["ok"]
    chunks = engine.store.list_chunks("omp-1", "omp")
    run_common_tools(sc, capture, session_args={"path": path}, search_modes=("regex", "sparse", "hybrid"),
                     chunk_ids=[c.chunk_id for c in chunks if any(tag in c.text for tag in seeds.canonical)])
    capture.add("get_chunk-all", [tool("get_chunk", chunk_id=c.chunk_id) for c in chunks])
    capture.add("run_neograph_update", tool("run_neograph_update", session_id="omp-1", provider="omp"))
    capture.add("sync_embed", tool("sync_session", session_id="omp-1", provider="omp", path=path, embed=True))
    assert "local_only" in capture.text()
    captured = capfd.readouterr()  # native runtimes write to the real file descriptors
    capture.add("process-stdout", captured.out)
    capture.add("process-stderr", captured.err)
    capture.assert_clean(seeds.tags)
    verify_canonical_detection(sc)


def test_sparse_index_holds_masked_text_so_search_is_not_a_secret_oracle(omp_scenario):
    sc = omp_scenario
    engine, seeds = sc["engine"], sc["seeds"]
    assert tool("sync_session", session_id="omp-1", provider="omp", path=sc["path"], embed=False)["ok"]
    assert tool("search_history", query="steer", session_id="omp-1", provider="omp", mode="sparse")["hits"]
    stored = " ".join(row[0] for row in engine.sparse._conn.execute("SELECT text FROM chunks_fts"))
    assert "sk-***" in stored and not [tag for tag in seeds.tags if tag in stored]
    # The raw token is indexed nowhere, so querying for it reveals nothing about the history.
    probe = sk("OMPUSERZQ").rsplit("-", 1)[-1]
    assert any(probe in chunk.text for chunk in engine.store.list_chunks("omp-1", "omp"))
    assert tool("search_history", query=probe, session_id="omp-1", provider="omp", mode="sparse")["hits"] == []


# --------------------------------------------------------------------------------------------------
# Codex (remote embeddings, remote graph extraction, lifecycle hooks)
# --------------------------------------------------------------------------------------------------

CODEX_SID = "11111111-2222-3333-4444-555555555555"
API_KEY = "plain-format-api-key-APIKEYZQ-0042"


def codex_records(seeds):
    def header():
        return {"type": "session_meta", "payload": {
            "id": CODEX_SID, "session_id": CODEX_SID, "timestamp": "2026-01-01T00:00:00Z",
            "cwd": f"/work/{sk(seeds('CXCWDZQ'))}", "originator": "codex_cli", "cli_version": "1.2.3",
            "model_provider": "openai", "base_instructions": {"text": seeds.hidden("CXBASEPROMPTZQ")},
            "context_window": {"w": seeds.hidden("CXCTXZQ")}, "history_mode": "paginated",
            "agent_nickname": f"nick-{sk(seeds('CXNICKZQ'))}", "git": {"branch": seeds.hidden("CXGITZQ")}}}
    item = lambda payload: {"type": "response_item", "payload": payload}  # noqa: E731
    return [
        header(),
        item({"type": "message", "role": "user",
              "content": [{"type": "input_text", "text": f"deploy using {sk(seeds('CXUSERZQ'))}"},
                          {"type": "input_image", "image_url": "data:image/png;base64," + seeds("CXIMAGEZQ") + BLOB}]}),
        item({"type": "message", "role": "assistant",
              "content": [{"type": "output_text", "text": f"ok, password={seeds('CXASSISTZQ')}"}]}),
        {"type": "response_item", "ordinal": 3, "payload": {
            "type": "agent_message", "id": "m1", "author": "/root", "recipient": f"/root/worker-{sk(seeds('CXRECIPZQ'))}",
            "content": [{"type": "input_text", "text": f"relay {sk(seeds('CXAGENTZQ'))}"},
                        {"type": "encrypted_content", "encrypted_content": seeds.hidden("CXENCBLOCKZQ") + BLOB}],
            "internal_chat_message_metadata_passthrough": {"turn_id": "u"}}},
        item({"type": "tool_search_call", "id": "ts1", "call_id": "s1", "status": "completed", "execution": "client",
              "arguments": {"query": f"find tool for {sk(seeds('CXSEARCHZQ'))}", "limit": 3},
              "internal_chat_message_metadata_passthrough": {"turn_id": "u"}}),
        item({"type": "function_call", "name": "shell", "call_id": "c1",
              "arguments": json.dumps({"command": f"curl -H 'Authorization: Bearer {seeds('CXCALLZQ')}' x"})}),
        item({"type": "function_call_output", "call_id": "c1",
              "output": json.dumps({"output": f'{{"token": "{seeds("CXRESULTZQ")}"}}', "metadata": {"exit_code": 0}})}),
        {"type": "compacted", "payload": {"message": f"summary {sk(seeds('CXCOMPACTZQ'))}", "replacement_history": [
            {"type": "compaction", "encrypted_content": seeds("CXENCHISTORYZQ") + BLOB}]}},
    ]


class FakeHttp:
    posts: list = []

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002
        FakeHttp.posts.append({"url": url, "json": json, "headers": headers})
        count, dim = len(json["input"]), json["dimensions"]
        data = [{"index": i, "embedding": [float((i + j) % 5 + 1) for j in range(dim)]} for i in range(count)]
        return type("Response", (), {"raise_for_status": lambda self: None, "json": lambda self: {"data": data}})()


@pytest.fixture
def codex_scenario(tmp_path, monkeypatch):
    seeds = Seeds()
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / f"rollout-2026-01-01T00-00-00-{CODEX_SID}.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in codex_records(seeds)), encoding="utf-8")
    FakeHttp.posts = []
    monkeypatch.setattr(embedder_module.httpx, "Client", FakeHttp)
    monkeypatch.setenv("SELF_DIRECT_DENSE_BACKEND", "numpy")
    engine = SelfDirectEngine(Settings(codex_sessions_dir=root, index_dir=tmp_path / "index", OPENROUTER_API_KEY=API_KEY,
                                       use_fake_embedder=False, embedding_dim=8, dense_backend="numpy", _env_file=None))
    monkeypatch.setattr(server, "_engine", engine)
    seeds.tags.append(API_KEY)
    yield {"engine": engine, "sid": CODEX_SID, "provider": "codex", "path": str(path), "seeds": seeds,
           "canonical": list(seeds.canonical)}
    engine.close()


def test_codex_remote_payloads_tools_hooks_and_errors_are_masked(codex_scenario, monkeypatch, capfd):
    sc = codex_scenario
    seeds, capture = sc["seeds"], Capture()
    engine, sid, path = sc["engine"], sc["sid"], sc["path"]
    prompts: list[str] = []
    graph_transport(monkeypatch, prompts, seeds)
    synced = tool("sync_session", session_id=sid, provider="codex", embed=True)
    capture.add("sync_session", synced)
    assert synced["ok"] and synced.get("embedded", 0) > 0 and not synced.get("embedding_error")
    chunks = engine.store.list_chunks(sid, "codex")
    # Remote graph extraction first: later graph commits advance the cursor past every event.
    capture.add("run_neograph_update", tool("run_neograph_update", session_id=sid, provider="codex"))
    run_common_tools(sc, capture, session_args={}, search_modes=("regex", "sparse", "hybrid", "dense"),
                     chunk_ids=[c.chunk_id for c in chunks if any(tag in c.text for tag in sc["canonical"])])
    capture.add("get_chunk-all", [tool("get_chunk", chunk_id=c.chunk_id) for c in chunks])
    capture.add("get_graph_context-after-llm", tool("get_graph_context", session_id=sid, provider="codex"))
    assert prompts, "the graph extraction transport was not reached"
    # Codex lifecycle hook through the MCP tool and the in-process entry point.
    engine.upsert_contracts(contracts_for(sc["canonical"][:2], extra=[
        {"id": "no-curl", "type": "must_not", "scope": "tool_call", "regex": r"curl\s", "description": "no curl",
         "source_quote": f"said {sk(seeds.input_only('HOOKQUOTEZQ'))}"}]))
    risky = {"command": f"curl -H 'Authorization: Bearer {seeds.input_only('HOOKCALLZQ')}' https://example.test"}
    prompt = f"never print {sk(seeds.input_only('HOOKPROMPTZQ'))}"
    for event, extra in (("PreToolUse", {"tool_name": "shell", "tool_input": risky}), ("Stop", {}),
                         ("UserPromptSubmit", {"prompt": prompt})):
        capture.add("hook-tool-" + event, tool("codex_session_hook", event_name=event, session_id=sid,
                                               transcript_path=path, turn_id="t", **extra))
        capture.add("hook-direct-" + event, handle_hook(engine, event, sid, path, extra.get("tool_name"),
                                                        extra.get("tool_input"), prompt=extra.get("prompt"), turn_id="t2"))
    capture.add("hook-odd-session", handle_hook(engine, "UserPromptSubmit", sk(seeds.input_only("HOOKSIDZQ")), path,
                                                prompt="never do this"))
    assert "violation" in capture.text() and "Self-directing audit" in capture.text()
    captured = capfd.readouterr()  # native runtimes write to the real file descriptors
    capture.add("process-stdout", captured.out)
    capture.add("process-stderr", captured.err)
    capture.assert_clean(seeds.tags)
    # Remote payloads: embedding documents/queries and the graph extraction prompt.
    remote = "\n".join(json.dumps(post["json"]) for post in FakeHttp.posts) + "\n" + "\n".join(prompts)
    assert FakeHttp.posts and all(post["headers"]["Authorization"] == f"Bearer {API_KEY}" for post in FakeHttp.posts)
    leaked = [tag for tag in seeds.tags if tag in remote]
    assert not leaked, f"raw secrets reached a remote payload: {leaked}"
    assert sum(len(post["json"]["input"]) for post in FakeHttp.posts) >= 5
    assert "sk-***" in remote
    verify_canonical_detection(sc)


# --------------------------------------------------------------------------------------------------
# AGY (the hook process entry point)
# --------------------------------------------------------------------------------------------------

AGY_SID = "11111111-1111-4111-8111-111111111111"
AGY_SID2 = "33333333-3333-4333-8333-333333333333"


def agy_steps(seeds, prefix, *, with_generic):
    step = lambda n, source, kind, **extra: {"step_index": n, "source": source, "type": kind, "status": "DONE",  # noqa: E731
                                              "created_at": f"2026-10-01T00:00:0{n}Z", **extra}
    records = [
        step(0, "USER_EXPLICIT", "USER_INPUT", content=f"<USER_REQUEST>use {sk(seeds(prefix + 'USERZQ'))}</USER_REQUEST>"),
        step(1, "SYSTEM_SDK", "EPHEMERAL_MESSAGE", content=f"ctx password={seeds(prefix + 'EPHEMERALZQ')}"),
        step(2, "MODEL", "PLANNER_RESPONSE", content=f"reply Authorization: Bearer {seeds(prefix + 'ASSISTZQ')}",
             tool_calls=[{"name": "run_command", "args": {"CommandLine": json.dumps(f"curl {sk(seeds(prefix + 'CALLZQ'))}")}}]),
    ]
    if with_generic:
        records += [
            step(3, "SYSTEM", "SYSTEM_MESSAGE", content=f"host says {sk(seeds(prefix + 'SYSTEMZQ'))}"),
            step(4, "MODEL", "GENERIC", content="Created At: x\nCompleted At: y\n\nThe command exited with code 0.\nOutput:\n"
                                                f'{{"secret": "{seeds(prefix + "RESULTZQ")}"}}\n')]
    return records


class FakeWorkflow:
    def __init__(self, tag):
        self.tag = tag

    def is_completion_action(self, action):
        return False

    def status(self, session_id, provider="agy", **_):
        return {"ok": True, "stage": "verify", "requirements": ["req-1"], "evidence": [],
                "problems": [f"approval drift token={self.tag}"]}


@pytest.fixture
def agy_scenario(tmp_path, monkeypatch):
    seeds = Seeds()
    app = tmp_path / "app"
    paths = {}
    for sid, prefix, generic in ((AGY_SID, "AG", True), (AGY_SID2, "AH", False)):
        path = app / "brain" / sid / ".system_generated" / "logs" / "transcript.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in agy_steps(seeds, prefix, with_generic=generic)),
                        encoding="utf-8")
        paths[sid] = str(path)
    settings = Settings(_env_file=None, local_only=True, index_dir=tmp_path / "index", agy_app_data_dirs=[app],
                        openrouter_api_key_file=None)
    engine = SelfDirectEngine(settings)
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(engine_module, "SelfDirectEngine", lambda: SelfDirectEngine(settings))
    yield {"engine": engine, "sid": AGY_SID, "provider": "agy", "path": paths[AGY_SID], "path2": paths[AGY_SID2],
           "seeds": seeds, "canonical": [tag for tag in seeds.canonical if tag.startswith("AG")]}
    engine.close()


def run_agy_hook(monkeypatch, capsys, event, payload):
    monkeypatch.setattr(sys, "argv", ["agy_hook", f"--event={event}"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    agy_hooks.main()
    captured = capsys.readouterr()
    return captured.out + "\n" + captured.err


def test_agy_hook_main_masks_injected_context_reasons_and_stderr(agy_scenario, monkeypatch, capsys):
    sc = agy_scenario
    seeds, capture, engine, path = sc["seeds"], Capture(), sc["engine"], sc["path"]
    assert tool("sync_session", session_id=AGY_SID, provider="agy", path=path, embed=False)["ok"]
    chunks = engine.store.list_chunks(AGY_SID, "agy")
    run_common_tools(sc, capture, session_args={"path": path}, search_modes=("regex", "sparse"),
                     chunk_ids=[c.chunk_id for c in chunks if any(tag in c.text for tag in sc["canonical"])])
    # A second, fully readable conversation: graph relations, contracts and a pending item are quoted back
    # into PreInvocation context, where nothing raw may be injected.
    assert tool("sync_session", session_id=AGY_SID2, provider="agy", path=sc["path2"], embed=False)["ok"]
    first = engine.store.list_chunks(AGY_SID2, "agy")[0].chunk_id
    edge = {"job_id": "agy-job", "nodes": [{"node_id": f"svc-{sk(seeds.input_only('AGEDGEZQ'))}", "kind": "task", "label": "a"},
                                           {"node_id": "db", "kind": "task", "label": "b"}],
            "edges": [{"src": f"svc-{sk('AGEDGEZQ')}", "dst": "db", "relation": "DEPENDS_ON", "origin": "observed",
                       "evidence_chunk_ids": [first]}]}
    assert tool("commit_graph_update", session_id=AGY_SID2, provider="agy", proposal=edge,
                base_graph_version=0)["ok"] is True
    engine.upsert_contracts(contracts_for(sc["canonical"][:3], extra=[
        {"id": "no-curl", "provider": "agy", "type": "must_not", "scope": "tool_call", "regex": r"curl\s",
         "description": f"Never run curl with {sk(seeds.input_only('AGDESCZQ'))}",
         "source_quote": f"said {sk(seeds.input_only('AGQUOTEZQ'))}"}]))
    for sid in (AGY_SID, AGY_SID2):
        engine.update_checklist(sid, provider="agy", add=[{"text": f"finish without {sk(seeds.input_only('AGPEND' + sid[:1].upper() + 'ZQ'))}"}])
    workflow = FakeWorkflow(seeds.input_only("AGWORKFLOWZQ"))
    monkeypatch.setattr("self_directing_mcp.workflow.configured_workflow", lambda: workflow)
    base = {"conversationId": AGY_SID, "transcriptPath": path, "workspacePaths": []}
    base2 = {"conversationId": AGY_SID2, "transcriptPath": sc["path2"], "workspacePaths": []}
    outputs = [
        run_agy_hook(monkeypatch, capsys, "PreInvocation", base),
        run_agy_hook(monkeypatch, capsys, "PreInvocation", base2),
        run_agy_hook(monkeypatch, capsys, "PreToolUse", {**base, "stepIdx": 5, "toolCall": {
            "name": "run_command", "args": {"CommandLine": f"curl Authorization: Bearer {seeds.input_only('AGHOOKCALLZQ')}"}}}),
        run_agy_hook(monkeypatch, capsys, "Stop", {**base, "fullyIdle": True, "terminationReason": "model_stop"}),
        run_agy_hook(monkeypatch, capsys, "Stop", {**base2, "fullyIdle": True, "terminationReason": "model_stop"}),
        run_agy_hook(monkeypatch, capsys, "PostToolUse", {**base, "stepIdx": 5, "toolCall": {
            "name": "run_command", "args": {"CommandLine": "ls"}}, "error": f"failed {sk(seeds.input_only('AGPOSTZQ'))}"}),
    ]
    joined = "\n".join(outputs)
    assert "Tsukkomi AGY context" in joined and "Contract [" in joined and "Pending:" in joined, joined
    assert "Context only:" in joined and "-[DEPENDS_ON]->" in joined  # graph relation lines were injected
    assert "Approved workflow" in joined and '"decision"' in joined and "user said" in joined
    for text in outputs:
        capture.add("agy-hook", text)
    capture.assert_clean(seeds.tags)
    verify_canonical_detection(sc)


# --------------------------------------------------------------------------------------------------
# Workflow tools, rule round-trips, duplicate sources, logs, real stdio transport
# --------------------------------------------------------------------------------------------------

class FakeController:
    """Stands in for the host-owned workflow controller; returns secret-bearing observations."""

    def __init__(self, seeds):
        self.seeds = seeds

    def _result(self, name):
        return {"ok": True, "method": name,
                "reason": f"approval note password={self.seeds.input_only('WF' + name.upper() + 'ZQ')}",
                "history": [{"payload": {"stdout": f"Authorization: Bearer {self.seeds.input_only('WFHIST' + name.upper() + 'ZQ')}"}}],
                "grant": "Zm9vYmFy-single-use_grant.0123456789", "classification_id": "0123456789abcdef"}

    def status(self, *args, **kwargs):
        return self._result("status")

    def run_checks(self, *args, **kwargs):
        return self._result("checks")

    def classify(self, *args, **kwargs):
        return self._result("classify")

    def transition(self, *args, **kwargs):
        return self._result("transition")

    def authorize(self, *args, **kwargs):
        return self._result("authorize")

    def execute(self, *args, **kwargs):
        raise ValueError(f"runner failed with {sk(self.seeds.input_only('WFERRORZQ'))}")


def test_workflow_tool_outputs_and_errors_are_masked_but_grants_survive(monkeypatch):
    seeds, capture = Seeds(), Capture()
    controller = FakeController(seeds)
    monkeypatch.setattr("self_directing_mcp.workflow.configured_workflow", lambda: controller)
    command = {"tool_name": "workflow_command", "arguments": {"argv": ["true"], "cwd": "."}}
    results = [tool("workflow_status", session_id="s1"), tool("workflow_run_checks", session_id="s1"),
               tool("workflow_classify", session_id="s1"),
               tool("workflow_transition", session_id="s1", classification_id="0123456789abcdef"),
               tool("workflow_authorize", session_id="s1", classification_id="0123456789abcdef", action=command),
               tool("workflow_execute", session_id="s1", grant="g", action=command)]
    for result in results:
        capture.add("workflow", result)
    capture.assert_clean(seeds.tags)
    assert results[4]["grant"] == "Zm9vYmFy-single-use_grant.0123456789", "the single-use grant must round-trip exactly"
    assert results[0]["reason"] == "approval note password=***"
    assert results[5]["tool_error"].endswith("runner failed with sk-***")


def test_contract_rule_logic_round_trips_while_text_fields_are_masked(tmp_path, monkeypatch):
    engine = SelfDirectEngine(Settings(index_dir=tmp_path / "index", openrouter_api_key_file=None, local_only=True,
                                       _env_file=None))
    monkeypatch.setattr(server, "_engine", engine)
    try:
        rule = {"id": "rt", "type": "must_not", "scope": "any", "regex": r"password=\S+",
                "description": f"watch {sk('RTDESCZQ')}", "source_quote": "password=RTQUOTEZQ"}
        saved = tool("upsert_contracts", contracts=[rule])["contracts"][0]
        listed = tool("list_contracts")["contracts"][0]
        for view in (saved, listed):
            assert view["regex"] == r"password=\S+", "rule regex must not be rewritten by masking"
            assert "RTDESCZQ" not in json.dumps(view) and "RTQUOTEZQ" not in json.dumps(view)
        stored = json.loads(Path(engine.settings.resolve_contracts_path()).read_text())["contracts"][0]
        assert "RTDESCZQ" in stored["description"] and "RTQUOTEZQ" in stored["source_quote"], "storage stays raw"
        assert tool("upsert_contracts", contracts=[listed])["contracts"][0]["regex"] == r"password=\S+"
    finally:
        engine.close()


def test_workflow_cli_output_and_errors_are_masked_but_the_grant_is_handed_over(monkeypatch, capsys):
    from self_directing_mcp import workflow, workflow_cli

    seeds = Seeds()
    results = {"authorize": {"ok": True, "grant": "Zm9vYmFy-single-use_grant.0123456789",
                             "reason": f"note password={seeds.input_only('CLIOKZQ')}"}}

    class Controller:
        def __init__(self, *args, **kwargs):
            pass

        def authorize(self, **kwargs):
            return results["authorize"]

        def status(self, **kwargs):
            raise ValueError(f"cannot read {sk(seeds.input_only('CLIERRZQ'))}")

    monkeypatch.setattr(workflow, "WorkflowController", Controller)
    action = Path("action.json")
    base = ["tsukkomi-workflow", "{}", "--approval", "/approved", "--state", "/state", "--session", "s1"]
    monkeypatch.setattr(sys, "argv", [*base[:1], "authorize", *base[2:], "--classification", "c1", "--action", str(action)])
    monkeypatch.setattr(Path, "read_text", lambda self, encoding=None: "{}")
    workflow_cli.main()
    out = capsys.readouterr()
    assert json.loads(out.out)["grant"] == "Zm9vYmFy-single-use_grant.0123456789"
    assert "CLIOKZQ" not in out.out + out.err and "password=***" in out.out
    monkeypatch.setattr(sys, "argv", [*base[:1], "status", *base[2:]])
    with pytest.raises(SystemExit):
        workflow_cli.main()
    out = capsys.readouterr()
    assert "CLIERRZQ" not in out.out + out.err and "sk-***" in out.err


def test_duplicate_source_error_paths_and_messages_are_masked(tmp_path, monkeypatch):
    root = tmp_path / "sessions"
    first = root / sk("DUPDIRAZQ") / "a.jsonl"
    second = root / "password=DUPDIRBZQ" / "b.jsonl"
    for path, text in ((first, "first history"), (second, "different history")):
        path.parent.mkdir(parents=True)
        append(path, {"type": "session", "id": "dup-1"},
               {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": text}]}})
    monkeypatch.setenv("SELF_DIRECT_LOCAL_ONLY", "true")
    engine = SelfDirectEngine(Settings(index_dir=tmp_path / "index", omp_sessions_dir=root, openrouter_api_key_file=None,
                                       _env_file=None))
    monkeypatch.setattr(server, "_engine", engine)
    try:
        assert tool("sync_session", session_id="dup-1", provider="omp", path=str(first), embed=False)["ok"]
        conflict = tool("sync_session", session_id="dup-1", provider="omp", path=str(second), embed=False)
        audit = tool("audit_session", session_id="dup-1", provider="omp", path=str(second))
        raw = engine.sync_session(session_id="dup-1", provider="omp", path=str(second), embed=False)
        assert conflict["error"] == "ambiguous_session_source" and raw["error"] == "ambiguous_session_source"
        assert "DUPDIR" in json.dumps(raw), "the engine result itself stays raw; masking is at the tool boundary"
        text = json.dumps([conflict, audit])
        assert "DUPDIRAZQ" not in text and "DUPDIRBZQ" not in text and "sk-***" in text
    finally:
        engine.close()


def test_masking_precedes_truncation_so_no_secret_prefix_survives(omp_scenario):
    """A slice taken before masking can leave an unmaskable fragment (``sk-or-v1-O``) of a real key."""
    sc = omp_scenario
    engine = sc["engine"]
    append(Path(sc["path"]), {"type": "message", "message": {"role": "user", "content": [
        {"type": "text", "text": "do not " + "x" * 127 + " " + sk("CUTLINEZQ") + " tail"}]}})
    assert tool("sync_session", session_id="omp-1", provider="omp", path=sc["path"], embed=False)["ok"]
    snippet = tool("analyze_activity", session_id="omp-1", provider="omp")["behavior_around_instructions"][-1]["snippet"]
    assert "sk-or" not in snippet and "sk-***" in snippet, snippet
    engine.upsert_contracts([{"id": "quoted", "type": "must_not", "scope": "any", "roles": ROLES, "regex": rx("CUTLINEZQ"),
                              "source_quote": "y" * 104 + " " + sk("CUTQUOTEZQ") + " ever"}])
    reason = next(f for f in tool("audit_session", session_id="omp-1", provider="omp", path=sc["path"])["findings"]
                  if f["contract_id"] == "quoted")["reason"]
    assert "sk-or" not in reason and "sk-***" in reason, reason


def test_log_records_and_tracebacks_are_masked():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handler.addFilter(server._MaskingFilter())
    logger = logging.getLogger("tsukkomi.masking.test")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        secret = sk("LOGEXCZQ")  # a traceback quotes source lines, so the raw tag must not be written in them
        logger.error("failed for %s and password=%s", sk("LOGARGZQ"), "LOGKVZQ")
        try:
            raise RuntimeError("boom " + secret)
        except RuntimeError:
            logger.exception("request failed")
        logger.error("plain Authorization: Bearer LOGBEARERZQ", stack_info=True)
    finally:
        logger.removeHandler(handler)
    text = stream.getvalue()
    assert "RuntimeError" in text and "sk-***" in text
    assert not [tag for tag in ("LOGARGZQ", "LOGKVZQ", "LOGEXCZQ", "LOGBEARERZQ") if tag in text]


def test_root_handlers_receive_the_masking_filter():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        server._mask_logging()
        logging.getLogger("tsukkomi.root.test").error("leak %s", sk("ROOTLOGZQ"))
    finally:
        root.removeHandler(handler)
    assert "ROOTLOGZQ" not in stream.getvalue() and "sk-***" in stream.getvalue()


def stdio_env(tmp_path):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("SELF_DIRECT_") and key not in ("OPENROUTER_API_KEY", "CODEX_SESSIONS_DIR")}
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")
    env.update(SELF_DIRECT_CONFIG_FILE=str(config), PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
               PYTHONDONTWRITEBYTECODE="1", SELF_DIRECT_LOCAL_ONLY="true",
               SELF_DIRECT_OMP_SESSIONS_DIR=str(tmp_path / "history"), SELF_DIRECT_INDEX_DIR=str(tmp_path / "index"))
    return env


def test_real_omp_bridge_process_masks_results_errors_and_stderr(tmp_path):
    """The stdio server an OMP extension spawns: protocol-level results, isError text and stderr."""
    seeds = Seeds()
    history = tmp_path / "history"
    history.mkdir()
    session = history / "session.jsonl"
    append(session, *omp_records(seeds))
    seeds.tags.extend(EXTRA_TAGS)
    stderr = (tmp_path / "server.stderr").open("w+")
    texts: list[str] = []

    async def scenario():
        params = StdioServerParameters(command=sys.executable, args=["-m", "self_directing_mcp.server", "--omp-bridge"],
                                       env=stdio_env(tmp_path), cwd=str(tmp_path))
        async with stdio_client(params, errlog=stderr) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()

                async def call(name, **arguments):
                    response = await client.call_tool(name, arguments)
                    texts.extend(part.text for part in response.content if hasattr(part, "text"))
                    texts.append(json.dumps(response.structuredContent))
                    return response

                await call("upsert_contracts", contracts=contracts_for(seeds.canonical))
                audit = await call("audit_session", session_id="omp-1", provider="omp", path=str(session))
                assert json.loads(audit.content[0].text)["verdict"] == "violation"
                check = await call("check_action", session_id="omp-1", provider="omp", path=str(session), action={
                    "tool_name": "bash", "arguments": {"command": f"echo {sk(seeds.input_only('BRIDGEACTIONZQ'))}"}})
                assert json.loads(check.content[0].text)["action_executed"] is False
                bad = await call("get_graph_context", session_id="omp-1", provider=sk(seeds.input_only("BRIDGEPROVIDERZQ")))
                assert bad.isError
                invalid = await call("check_action", session_id="omp-1", provider="omp", action={
                    "tool_name": "x", "arguments": {}, "unexpected": sk(seeds.input_only("BRIDGEFIELDZQ"))})
                assert invalid.isError
                unknown = await call("no_such_tool_" + sk(seeds.input_only("BRIDGETOOLZQ")))
                assert unknown.isError
                await call("search_history", query="x", session_id="omp-1", provider="omp", mode="hybrid")

    try:
        asyncio.run(asyncio.wait_for(scenario(), timeout=60))
    finally:
        stderr.seek(0)
        err = stderr.read()
        stderr.close()
    blob = "\n".join(texts) + "\n" + err
    assert "sk-***" in blob
    leaked = [tag for tag in seeds.tags if tag in blob]
    assert not leaked, f"raw secrets reached the protocol or stderr: {leaked}"
