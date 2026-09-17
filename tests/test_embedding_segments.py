import numpy as np

from self_directing_mcp.embed.embedder import OpenRouterEmbedder
from self_directing_mcp.security.mask import mask_secrets


def test_long_documents_preserve_all_sanitized_text_and_aggregate(monkeypatch):
    embedder = OpenRouterEmbedder(api_key="test", dim=2)
    embedder.document_chunk_chars = 20
    embedder.request_batch_size = 2
    # The token spans the first boundary; redaction must precede splitting.
    original = "한글 🙂 prefix " + "sk-TEST_SECRET_SENTINEL_123456" + " 뒤쪽" * 30
    sent = []
    def call(parts):
        assert len(parts) <= 2 and all(len(part) <= 20 for part in parts)
        sent.extend(parts)
        return [np.array([1., 0.]) if "prefix" in part else np.array([0., 1.]) for part in parts]
    monkeypatch.setattr(embedder, "_call", call)
    vectors = embedder.embed_documents([original, "short"])
    assert "".join(sent[:-1]) == mask_secrets(original)
    assert sent[-1] == "short"
    assert "TEST_SECRET_SENTINEL" not in "".join(sent)
    assert len(vectors) == 2 and all(np.isclose(np.linalg.norm(v), 1) for v in vectors)
    expected = np.array([len(sent[0]), sum(len(part) for part in sent[1:-1])], dtype=float)
    assert np.allclose(vectors[0], expected / np.linalg.norm(expected))
    assert np.allclose(vectors[1], [0, 1])


def test_document_encoding_is_in_cache_identity():
    first = OpenRouterEmbedder(api_key="test", dim=2)
    second = OpenRouterEmbedder(api_key="test", dim=2)
    second.document_chunk_chars = 100
    assert first.cache_key != second.cache_key
