"""NeoGraph-executed GraphRAG pipeline (issue #5 deep integration)."""
from __future__ import annotations

from pathlib import Path

import pytest

from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.schemas import Chunk


SID = "ngtest000-1111-2222-3333-444444444444"


def test_neograph_missing_or_no_key_reports_clearly(tmp_path):
    from self_directing_mcp.engine import SelfDirectEngine

    settings = Settings(use_fake_embedder=True, index_dir=tmp_path / "index",
                        contracts_path=tmp_path / "index" / "contracts.json")
    engine = SelfDirectEngine(settings=settings)
    try:
        engine.ensure_ready()
        result = engine.run_neograph_update(SID)
        # Either the key is missing or neograph is missing; both are clear errors.
        if not result["ok"]:
            assert result["error"] in ("no_api_key", "neograph_missing", "no_events")
    finally:
        engine.close()


def test_topology_definition_compiles(tmp_path):
    """The NeoGraph topology itself must compile with registered node types."""
    ng = pytest.importorskip("neograph_engine")
    from self_directing_mcp.neograph_runner import run_update_pipeline

    # Register node types without running LLM: compile-only check.
    import neograph_engine as ng2

    class DummyNode(ng2.GraphNode):
        def get_name(self):
            return "dummy"

        def run(self, input):
            return []

    ng2.NodeFactory.register_type("ng_extract", lambda name, config, ctx: DummyNode())
    definition = {
        "schema_version": ng2.TOPOLOGY_SCHEMA_VERSION,
        "name": "compile_check",
        "channels": {"events": {"reducer": "overwrite"}},
        "nodes": {"e": {"type": "ng_extract"}},
        "edges": [{"from": ng2.START_NODE, "to": "e"}, {"from": "e", "to": ng2.END_NODE}],
    }
    engine = ng2.GraphEngine.compile(definition, ng2.NodeContext())
    assert engine is not None


def test_transport_schema_defaults(tmp_path):
    from self_directing_mcp.neograph_runner import _Transport

    t = _Transport("test-key", "test-model")
    assert t.schema["required"] == ["nodes", "edges", "extraction_summary"]
    edge_props = t.schema["properties"]["edges"]["items"]["properties"]
    assert set(edge_props) == {"src", "dst", "relation", "origin", "evidence_chunk_ids"}


def test_real_native_pipeline_executes_and_checkpoints_all_stages(tmp_path, monkeypatch):
    import neograph_engine as ng
    from self_directing_mcp.neograph_runner import run_update_pipeline, _Transport
    proposals = {'nodes': [{'node_id':'a'}, {'node_id':'b'}], 'edges': [
        {'src':'a','dst':'b','relation':'IMPLEMENTS','evidence_chunk_ids':['event-1']}],
        'extraction_summary': 'one relation'}
    seen = []
    def complete(self, prompt):
        assert 'full evidence at the end' in prompt
        return proposals, 3
    monkeypatch.setattr(_Transport, 'complete_json', complete)
    def on_proposal(stage, proposal):
        seen.append(stage)
        return {'ok': True} if stage == 'validate' else {'ok': True, 'status': 'applied'}
    result = run_update_pipeline(session_id=SID, provider='codex',
        events=[{'chunk_id':'event-1','kind':'turn','text':'x' * 250 + ' full evidence at the end'}],
        api_key='unit-test', model='unit-test', checkpoint_dir=tmp_path, on_proposal=on_proposal)
    assert seen == ['validate', 'commit']
    assert result['executor'] == 'neograph-engine'
    assert result['execution_trace'] == ['e','v','a']
    store = ng.SqliteCheckpointStore(str(tmp_path / 'neograph_checkpoints.sqlite'))
    assert store.load_latest(result['thread_id']) is not None


def test_cancel_after_extraction_never_commits(tmp_path, monkeypatch):
    from threading import Event
    from self_directing_mcp.neograph_runner import run_update_pipeline, _Transport
    cancelled, commits = Event(), []
    def complete(self, prompt):
        cancelled.set()
        return {'nodes': [], 'edges': []}, 1
    monkeypatch.setattr(_Transport, 'complete_json', complete)
    with pytest.raises((RuntimeError, TimeoutError), match='cancelled|expired'):
        run_update_pipeline(session_id=SID, provider='codex', events=[], api_key='unit-test',
            model='unit-test', checkpoint_dir=tmp_path, cancel_event=cancelled,
            on_proposal=lambda *args: commits.append(args))
    assert not commits


def test_audit_uses_native_graph_without_promoting_description_to_clean(tmp_path):
    from self_directing_mcp.audit.runner import run_audit
    from self_directing_mcp.schemas import ContractRule
    class Store:
        def chunk_count(self, *args): return 1
    result = run_audit(session_id=SID, provider='codex', store=Store(),
        contracts=[ContractRule(id='natural',type='must',description='Use the intended architecture')],
        coverage={'complete': True})
    assert result.verdict == 'unknown'
    assert result.execution['executor'] == 'neograph-engine'
    assert result.execution['nodes'] == ['select_contracts','load_evidence','evaluate_contracts','aggregate_findings']


def test_empty_ai_graph_retains_cursor_and_pending_events(tmp_path, monkeypatch):
    from self_directing_mcp.neograph_runner import _Transport
    monkeypatch.setenv('OPENROUTER_API_KEY', 'unit-test-not-a-real-key')
    monkeypatch.setattr(_Transport, 'complete_json', lambda *a, **k: (
        {'nodes': [], 'edges': [], 'extraction_summary': 'No relations found'}, 5))
    engine = SelfDirectEngine(Settings(use_fake_embedder=True, index_dir=tmp_path/'index',
                                       contracts_path=tmp_path/'contracts.json'))
    try:
        engine.ensure_ready()
        engine.store.commit_events([
            Chunk(chunk_id='event-1',session_id=SID,provider='codex',kind='turn',
                  text='src/audit.py IMPLEMENTS R-audit',content_hash='test-content')
        ], session_id=SID, provider='codex', path=tmp_path/'s.jsonl', offset=10,digest='test')
        before = engine.graph.cursor(SID, 'codex')
        for _ in range(2):
            result = engine.run_neograph_update(SID)
            assert result['ok'] is False
            assert result['apply_result']['status'] == 'rejected'
            assert result['validation']['problems'] == ['empty_graph_requires_review']
            assert result['remaining_events'] == 1
            assert result['graph_cursor_after'] == before
            assert engine.graph.cursor(SID, 'codex') == before
    finally:
        engine.close()


def test_graph_extraction_masks_session_secrets_before_remote_prompt(tmp_path, monkeypatch):
    from self_directing_mcp.neograph_runner import _Transport

    prompts = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-not-a-real-key")

    def complete(self, prompt):
        prompts.append(prompt)
        return {"nodes": [], "edges": [], "extraction_summary": "No relations found"}, 5

    monkeypatch.setattr(_Transport, "complete_json", complete)
    engine = SelfDirectEngine(Settings(use_fake_embedder=True, index_dir=tmp_path / "index",
                                       contracts_path=tmp_path / "contracts.json"))
    try:
        engine.ensure_ready()
        engine.store.commit_events([
            Chunk(chunk_id="event-1", session_id=SID, provider="omp", kind="meta",
                  text="[omp_event:custom_message:async-result] OPENROUTER_API_KEY=unit-test-private-token",
                  content_hash="test-content",
                  meta={"role": "runtime", "error_text": "Bearer unit-test-private-token"})
        ], session_id=SID, provider="omp", path=tmp_path / "session.jsonl", offset=10, digest="test")
        engine.run_neograph_update(SID, provider="omp")
        assert len(prompts) == 1
        assert "unit-test-private-token" not in prompts[0]
        assert "OPENROUTER_API_KEY=***" in prompts[0]
        assert "event-1" in prompts[0]
    finally:
        engine.close()


def test_graph_settings_toml_catalog_validation_and_audit_status(tmp_path, monkeypatch):
    from self_directing_mcp import neograph_runner, settings_core
    from self_directing_mcp.config import hook_settings

    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[paths]\nembedding_api_key_file = "emb.key"\n'
        '[graph]\nmodel = "qwen/qwen3-30b-a3b"\nmax_output_tokens = 6000\nreasoning_effort = "medium"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(cfg))
    for reader in (hook_settings, settings_core.load_hook_settings):
        s = reader()
        assert s.graph_model == "qwen/qwen3-30b-a3b"
        assert s.graph_max_output_tokens == 6000
        assert s.graph_reasoning_effort == "medium"
        assert s.embedding_api_key_file == (tmp_path / "emb.key").resolve()

    fake_catalog = {
        "qwen/qwen3-30b-a3b": {"id": "qwen/qwen3-30b-a3b", "context_length": 65536, "supported_parameters": ["response_format", "reasoning"]},
        "bad/no-json-model": {"id": "bad/no-json-model", "context_length": 32768, "supported_parameters": ["temperature"]},
    }
    neograph_runner.set_model_catalog_cache(fake_catalog)
    try:
        info = neograph_runner.validate_graph_model("qwen/qwen3-30b-a3b")
        assert info["ok"] is True
        assert info["context_length"] == 65536
        bad_info = neograph_runner.validate_graph_model("bad/no-json-model")
        assert bad_info["ok"] is False and bad_info["error"] == "unsupported_graph_model"
        missing_info = neograph_runner.validate_graph_model("unknown/missing-model")
        assert missing_info["ok"] is False and missing_info["error"] == "invalid_graph_model"
    finally:
        neograph_runner.clear_model_catalog_cache()


def test_graph_batch_trimming_and_specific_failure_reasons(tmp_path, monkeypatch):
    from self_directing_mcp import neograph_runner
    from self_directing_mcp.neograph_runner import _Transport

    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-not-a-real-key")
    neograph_runner.set_model_catalog_cache({
        neograph_runner.DEFAULT_GRAPH_MODEL: {
            "id": neograph_runner.DEFAULT_GRAPH_MODEL,
            "context_length": 1800,
            "supported_parameters": ["response_format", "reasoning"],
        }
    })
    prompts = []
    def complete_ok(self, prompt):
        prompts.append(prompt)
        return {
            "nodes": [
                {"node_id": "file:src/a.py", "kind": "File", "label": "src/a.py"},
                {"node_id": "req:R-1", "kind": "Requirement", "label": "R-1"},
            ],
            "edges": [
                {
                    "src": "file:src/a.py",
                    "dst": "req:R-1",
                    "relation": "IMPLEMENTS",
                    "origin": "observed",
                    "evidence_chunk_ids": ["ev-1"],
                }
            ],
            "extraction_summary": "extracted file relation",
        }, 12

    monkeypatch.setattr(_Transport, "complete_json", complete_ok)
    engine = SelfDirectEngine(Settings(
        use_fake_embedder=True,
        index_dir=tmp_path / "index",
        contracts_path=tmp_path / "contracts.json",
        graph_max_output_tokens=1000,
        mask_pii="email",
    ))
    try:
        engine.ensure_ready()
        status = engine.audit_status()
        assert status["graph_model"] == neograph_runner.DEFAULT_GRAPH_MODEL
        assert status["graph_max_output_tokens"] == 1000
        assert status["graph_reasoning_effort"] == "low"
        assert isinstance(status["graph_input_budget_tokens"], int) and status["graph_input_budget_tokens"] > 0

        engine.store.commit_events([
            Chunk(chunk_id="ev-1", session_id=SID, provider="codex", kind="turn",
                  text="contact dev@example.invalid src/a.py IMPLEMENTS R-1 " + ("alpha " * 90), content_hash="h1"),
            Chunk(chunk_id="ev-2", session_id=SID, provider="codex", kind="turn",
                  text="second event src/a.py IMPLEMENTS R-1 " + ("beta " * 90), content_hash="h2"),
        ], session_id=SID, provider="codex", path=tmp_path / "s.jsonl", offset=20, digest="d1")

        r1 = engine.run_neograph_update(SID)
        assert r1["ok"] is True
        assert r1["batch_events"] == 1
        assert r1["remaining_events"] == 1
        assert "dev@example.invalid" not in prompts[0]
        assert "[MASKED_EMAIL]" in prompts[0]

        # Drain ev-2 with matching evidence_chunk_ids
        def complete_ok_2(self, prompt):
            prompts.append(prompt)
            return {
                "nodes": [
                    {"node_id": "file:src/a.py", "kind": "File", "label": "src/a.py"},
                    {"node_id": "req:R-1", "kind": "Requirement", "label": "R-1"},
                ],
                "edges": [
                    {
                        "src": "file:src/a.py",
                        "dst": "req:R-1",
                        "relation": "IMPLEMENTS",
                        "origin": "observed",
                        "evidence_chunk_ids": ["ev-2"],
                    }
                ],
                "extraction_summary": "extracted second event",
            }, 12
        monkeypatch.setattr(_Transport, "complete_json", complete_ok_2)
        r2 = engine.run_neograph_update(SID)
        assert r2["ok"] is True
        assert r2["batch_events"] == 1
        assert r2["remaining_events"] == 0
        assert len(prompts) == 2
        assert "second event" in prompts[1]

        # Oversized single event -> graph_event_too_large with needed_tokens and budget_tokens
        engine.store.commit_events([
            Chunk(chunk_id="ev-huge", session_id=SID, provider="codex", kind="turn",
                  text="가" * 5000, content_hash="h3"),
        ], session_id=SID, provider="codex", path=tmp_path / "s.jsonl", offset=40, digest="d2")
        r3 = engine.run_neograph_update(SID)
        assert r3["ok"] is False
        assert r3["error"] == "graph_event_too_large"
        assert r3["chunk_id"] == "ev-huge"
        assert r3["needed_tokens"] > r3["budget_tokens"]
    finally:
        engine.close()
        neograph_runner.clear_model_catalog_cache()


def test_graph_transport_distinguishes_completion_not_stopped_and_invalid_json(tmp_path, monkeypatch):
    from self_directing_mcp import neograph_runner
    from self_directing_mcp.neograph_runner import _Transport

    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-not-a-real-key")
    neograph_runner.clear_model_catalog_cache()

    def fail_length(self, prompt):
        raise ValueError("completion_not_stopped: finish_reason=length")

    monkeypatch.setattr(_Transport, "complete_json", fail_length)
    engine = SelfDirectEngine(Settings(
        use_fake_embedder=True,
        index_dir=tmp_path / "index",
        contracts_path=tmp_path / "contracts.json",
    ))
    try:
        engine.ensure_ready()
        engine.store.commit_events([
            Chunk(chunk_id="ev-len", session_id=SID, provider="codex", kind="turn",
                  text="src/x.py IMPLEMENTS R-1", content_hash="hx"),
        ], session_id=SID, provider="codex", path=tmp_path / "s.jsonl", offset=10, digest="dx")
        res = engine.run_neograph_update(SID)
        assert res["ok"] is False
        assert any("completion_not_stopped" in p for p in res["validation"]["problems"])

        def fail_json(self, prompt):
            raise ValueError("invalid_json: Expecting value")

        monkeypatch.setattr(_Transport, "complete_json", fail_json)
        res2 = engine.run_neograph_update(SID)
        assert res2["ok"] is False
        assert any("invalid_json" in p for p in res2["validation"]["problems"])
    finally:
        engine.close()


def test_explicit_mask_pii_none_overrides_env_and_strict_catalog_checks(tmp_path, monkeypatch):
    from self_directing_mcp import neograph_runner
    from self_directing_mcp.neograph_runner import _Transport

    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-not-a-real-key")
    monkeypatch.setenv("SELF_DIRECT_MASK_PII", "email,phone")

    # 1. Catalog missing when fetch=True returns catalog_unavailable
    monkeypatch.setattr(neograph_runner, "fetch_model_catalog", lambda *a, **k: None)
    neograph_runner.clear_model_catalog_cache()
    unavail = neograph_runner.validate_graph_model("deepseek/deepseek-v4-flash-0731", fetch=True)
    assert unavail["ok"] is False and unavail["error"] == "catalog_unavailable"

    # 2. structured_outputs alone without response_format is rejected; non-list supported_parameters is rejected
    neograph_runner.set_model_catalog_cache({
        "m/struct-only": {"id": "m/struct-only", "context_length": 32768, "supported_parameters": ["structured_outputs"]},
        "m/no-list": {"id": "m/no-list", "context_length": 32768},
        "m/valid": {"id": "m/valid", "context_length": 4096, "supported_parameters": ["response_format"]},
    })
    try:
        assert neograph_runner.validate_graph_model("m/struct-only")["error"] == "unsupported_graph_model"
        assert neograph_runner.validate_graph_model("m/no-list")["error"] == "unsupported_graph_model"
        # max_output_tokens exceeding context_length is rejected and budget is 0 (no 256 floor)
        over_ctx = neograph_runner.validate_graph_model("m/valid", max_output_tokens=4000)
        assert over_ctx["ok"] is False and over_ctx["error"] == "invalid_graph_max_output_tokens"
        assert neograph_runner.compute_input_token_budget("m/valid", max_output_tokens=4000) == 0

        # 3. Settings(mask_pii="none") overrides SELF_DIRECT_MASK_PII=email,phone on the graph path
        prompts = []
        def complete_ok(self, prompt):
            prompts.append(prompt)
            return {
                "nodes": [
                    {"node_id": "file:src/b.py", "kind": "File", "label": "src/b.py"},
                    {"node_id": "req:R-2", "kind": "Requirement", "label": "R-2"},
                ],
                "edges": [
                    {
                        "src": "file:src/b.py",
                        "dst": "req:R-2",
                        "relation": "IMPLEMENTS",
                        "origin": "observed",
                        "evidence_chunk_ids": ["ev-nomask"],
                    }
                ],
                "extraction_summary": "ok",
            }, 10
        monkeypatch.setattr(_Transport, "complete_json", complete_ok)
        engine = SelfDirectEngine(Settings(
            use_fake_embedder=True,
            index_dir=tmp_path / "index",
            contracts_path=tmp_path / "contracts.json",
            graph_model="m/valid",
            graph_max_output_tokens=1000,
            mask_pii="none",
        ))
        try:
            engine.ensure_ready()
            engine.store.commit_events([
                Chunk(chunk_id="ev-nomask", session_id=SID, provider="codex", kind="turn",
                      text="keep bob@example.invalid and 010-1234-5678. src/b.py IMPLEMENTS R-2", content_hash="hnm"),
            ], session_id=SID, provider="codex", path=tmp_path / "s.jsonl", offset=10, digest="dnm")
            res = engine.run_neograph_update(SID)
            assert res["ok"] is True
            assert "bob@example.invalid" in prompts[0]
            assert "010-1234-5678." in prompts[0]
        finally:
            engine.close()
    finally:
        neograph_runner.clear_model_catalog_cache()
