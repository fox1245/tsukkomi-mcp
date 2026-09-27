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
