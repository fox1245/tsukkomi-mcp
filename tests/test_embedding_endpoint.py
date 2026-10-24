"""The embedding endpoint is a setting: any OpenAI-compatible ``/embeddings`` server works, OpenRouter or a local one.

``SELF_DIRECT_EMBEDDING_BASE_URL`` (``Settings.embedding_base_url``) names the server, ``SELF_DIRECT_EMBEDDING_MODEL`` and
``SELF_DIRECT_EMBEDDING_DIM`` the model and the vector size, and the API key is whatever string the server accepts. This test
runs the real embedder client against a stub server on the loopback interface (no model, no network beyond 127.0.0.1) and
checks what the client sends and that the engine indexes and searches with the vectors it gets back. It proves the wiring,
not the quality of any model: the stub hashes character bigrams, so the order of its dense hits means nothing.
All text is synthetic.
"""
from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from test_reliability import SID, message, write

DIM = 48
TEXTS = ["검색 색인을 다시 만들었다", "english filler about nothing", "집에서 쉬었다", "배포 계약을 확인했다"]


def stub_vector(text: str, dim: int) -> list[float]:
    """A deterministic embedding: the hashed character bigrams of the text, normalized."""
    vec = np.zeros(dim)
    flat = "".join(text.split())
    for i in range(max(1, len(flat) - 1)):
        vec[int.from_bytes(hashlib.sha256(flat[i:i + 2].encode()).digest()[:4], "big") % dim] += 1.0
    norm = float(np.linalg.norm(vec)) or 1.0
    return (vec / norm).tolist()


class Stub:
    def __init__(self):
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):          # keep the test output quiet
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                stub.requests.append({"path": self.path, "authorization": self.headers.get("Authorization"), "body": body})
                if self.path != "/v1/embeddings":
                    self.send_error(404)
                    return
                dim = int(body.get("dimensions") or DIM)
                data = [{"index": i, "embedding": stub_vector(text, dim)} for i, text in enumerate(body["input"])]
                payload = json.dumps({"data": data}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def local_endpoint(tmp_path, monkeypatch):
    for name in ("SELF_DIRECT_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER",
                 "SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR", "SELF_DIRECT_SPARSE_FORMAT", "SELF_DIRECT_OPENROUTER_API_KEY_FILE",
                 "SELF_DIRECT_EMBEDDING_BASE_URL", "SELF_DIRECT_USE_FAKE_EMBEDDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "local-server-needs-no-key")          # any string: the stub ignores it
    root = tmp_path / "sessions"
    root.mkdir()
    write(root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl", [{"type": "session_meta", "payload": {"id": SID}}] + [message(t) for t in TEXTS])
    with Stub() as stub:
        engine = SelfDirectEngine(Settings(codex_sessions_dir=root, index_dir=tmp_path / "index", use_fake_embedder=False,
                                           local_only=False, dense_backend="numpy", embedding_base_url=stub.base_url,
                                           embedding_model="local-test-model", embedding_dim=DIM, _env_file=None))
        engine.ensure_ready()
        try:
            yield engine, stub
        finally:
            engine.close()


def test_the_engine_embeds_and_searches_through_a_local_endpoint(local_endpoint):
    engine, stub = local_endpoint
    assert engine.sync_session(session_id=SID, embed=True)["ok"]
    assert stub.requests, "the client never called the configured endpoint"
    first = stub.requests[0]
    assert first["path"] == "/v1/embeddings"
    assert first["authorization"] is None
    assert first["body"]["model"] == "local-test-model" and first["body"]["dimensions"] == DIM
    assert all(isinstance(text, str) for request in stub.requests for text in request["body"]["input"])
    status = engine.audit_status(session_id=SID)
    assert status["dense_count"] > 0 and status["embedder_degraded"] is False
    for mode in ("dense", "hybrid"):
        hits = engine.search_history("색인을 다시", session_id=SID, mode=mode)["hits"]
        assert hits, mode
        if mode == "dense":
            assert all(hit["dense_rank"] is not None for hit in hits)
        else:
            assert any(hit["sparse_rank"] is not None for hit in hits)          # the Korean sparse channel works beside it


def test_a_query_is_embedded_by_the_same_endpoint(local_endpoint):
    engine, stub = local_endpoint
    assert engine.sync_session(session_id=SID, embed=True)["ok"]
    before = len(stub.requests)
    assert engine.search_history("배포 계약", session_id=SID, mode="dense")["hits"]
    assert len(stub.requests) > before, "the query vector did not come from the configured endpoint"


def test_local_endpoint_uses_dedicated_embedding_api_key_file_and_masks_pii(tmp_path, monkeypatch):
    from self_directing_mcp.config import Settings
    from self_directing_mcp.engine import SelfDirectEngine

    key_file = tmp_path / "emb.key"
    key_file.write_text("EMBEDDING_API_KEY=dedicated-local-emb-key\n", encoding="utf-8")
    root = tmp_path / "sessions"
    root.mkdir()
    raw_text = "Contact alice.dev@example.invalid. Call +82-10-1234-5678. On 2026-10-24 v0.14.1"
    write(
        root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl",
        [{"type": "session_meta", "payload": {"id": SID}}, message(raw_text)],
    )
    with Stub() as stub:
        engine = SelfDirectEngine(Settings(
            codex_sessions_dir=root,
            use_fake_embedder=False,
            local_only=False,
            dense_backend="numpy",
            index_dir=tmp_path / "index",
            contracts_path=tmp_path / "contracts.json",
            embedding_base_url=stub.base_url,
            embedding_model="local-test-model",
            embedding_dim=DIM,
            openrouter_api_key="openrouter-secret-must-not-leak",
            embedding_api_key_file=key_file,
            mask_pii="email,phone",
            _env_file=None,
        ))
        try:
            engine.ensure_ready()
            res = engine.sync_session(session_id=SID, embed=True)
            assert res["ok"] is True
            assert stub.requests
            req = stub.requests[0]
            assert req["authorization"] == "Bearer dedicated-local-emb-key"
            sent_inputs = " ".join(req["body"]["input"])
            assert "alice.dev@example.invalid" not in sent_inputs
            assert "+82-10-1234-5678" not in sent_inputs
            assert "[MASKED_EMAIL]." in sent_inputs and "[MASKED_PHONE]." in sent_inputs
            assert "2026-10-24" in sent_inputs and "v0.14.1" in sent_inputs
        finally:
            engine.close()


def test_shared_dotenv_with_only_openrouter_key_never_leaks_to_local_endpoint(tmp_path):
    from self_directing_mcp.config import Settings

    shared_env = tmp_path / "shared.env"
    shared_env.write_text("OPENROUTER_API_KEY=or-secret-never-send-to-local\n", encoding="utf-8")
    s_local = Settings(
        embedding_base_url="http://127.0.0.1:11434/v1",
        embedding_api_key_file=shared_env,
        _env_file=None,
    )
    assert s_local.resolve_embedding_api_key() is None

    s_or = Settings(
        embedding_base_url="https://openrouter.ai/api/v1",
        embedding_api_key_file=shared_env,
        _env_file=None,
    )
    assert s_or.resolve_embedding_api_key() == "or-secret-never-send-to-local"
