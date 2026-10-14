"""Stress the native sqlite-vector extension the way the server uses it: batches, deletes, filtered and read-only
searches, reopen, and concurrent readers beside a writer. Runs only with SQLITE_VECTOR_TEST_PATH; the sanitizer CI
points it at an ASan/UBSan or TSan build of the official extension, so memory and thread errors in the native
code surface here."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading

import numpy as np
import pytest

from self_directing_mcp.index.vector import build_vector_index

pytestmark = pytest.mark.skipif(not os.environ.get("SQLITE_VECTOR_TEST_PATH"),
                                reason="Set SQLITE_VECTOR_TEST_PATH to exercise the native extension")


def _native(tmp_path, dim):
    dense, backend = build_vector_index(tmp_path, dim=dim, extension_path=Path(os.environ["SQLITE_VECTOR_TEST_PATH"]))
    assert backend == "sqlite-vector"
    return dense


@pytest.mark.parametrize("dim", [1, 3, 17, 384, 1536])
def test_batches_deletes_filters_and_reopen_stay_exact(tmp_path, dim):
    rng = np.random.default_rng(dim)
    dense = _native(tmp_path, dim)
    vectors = {}
    try:
        for batch in range(6):
            pairs = [(f"v{batch}-{i}", rng.standard_normal(dim)) for i in range(1 + batch * 37)]
            dense.upsert_many(pairs)
            vectors.update(pairs)
            doomed = [key for key in list(vectors)[::5]]
            dense.delete_ids(doomed)
            for key in doomed:
                vectors.pop(key)
            query = rng.standard_normal(dim)
            cosine = {key: float(np.dot(v, query) / (np.linalg.norm(v) * np.linalg.norm(query)))
                      for key, v in vectors.items()}
            best = sorted(cosine.values(), reverse=True)[:10]
            got = dense.search(query, 10)
            # Scores, not ids: equal cosines (always the case at dim 1) have no defined order.
            assert [score for _key, score in got] == pytest.approx(best, abs=1e-5)
            assert all(score == pytest.approx(cosine[key], abs=1e-5) for key, score in got)
            allowed = set(list(vectors)[::3])
            assert {key for key, _ in dense.search(query, 50, allowed_ids=allowed)} <= allowed
            assert [score for _key, score in dense.search_readonly(query, 10)] == pytest.approx(best, abs=1e-5)
        assert dense.count() == len(vectors)
    finally:
        dense.close()
    reopened = _native(tmp_path, dim)
    try:
        assert reopened.count() == len(vectors) and reopened.present(list(vectors)) == set(vectors)
    finally:
        reopened.close()


def test_concurrent_engine_searches_beside_syncs(tmp_path):
    """The production pattern: several request threads share one engine, which serializes every use of the
    native store through its publication guard (NativeVectorStore itself is not thread-safe)."""
    from self_directing_mcp.config import Settings
    from self_directing_mcp.engine import SelfDirectEngine
    sid = "11111111-2222-3333-4444-555555555555"
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / f"rollout-2026-10-01T00-00-00-{sid}.jsonl"

    def append(start, n):
        with path.open("a", encoding="utf-8") as stream:
            if start == 0:
                stream.write(json.dumps({"type": "session_meta", "payload": {"id": sid}}) + "\n")
            for i in range(start, start + n):
                stream.write(json.dumps({"type": "response_item", "payload": {
                    "type": "message", "role": "assistant", "content": f"progress note {i} alpha beta"}}) + "\n")

    append(0, 20)
    engine = SelfDirectEngine(Settings(_env_file=None, codex_sessions_dir=root, index_dir=tmp_path / "index",
                                       use_fake_embedder=True, dense_backend="sqlite-vector",
                                       sqlite_vector_path=Path(os.environ["SQLITE_VECTOR_TEST_PATH"]),
                                       lock_wait_timeout_sec=120, request_timeout_sec=300))
    assert engine.sync_session(session_id=sid, embed=True)["ok"]
    errors, stop = [], threading.Event()

    def read(mode):
        try:
            while not stop.is_set():
                result = engine.search_history("progress note", session_id=sid, mode=mode, top_k=5)
                assert result.get("ok", True) is not False and result["hits"], result
        except Exception as exc:          # surfaced in the main thread
            errors.append(exc)

    readers = [threading.Thread(target=read, args=(mode,)) for mode in ("dense", "hybrid", "dense", "hybrid")]
    for reader in readers:
        reader.start()
    try:
        for round_ in range(15):
            append(20 + round_ * 5, 5)
            assert engine.sync_session(session_id=sid, embed=True)["ok"]
    finally:
        stop.set()
        for reader in readers:
            reader.join(timeout=120)
        engine.close()
    assert not errors, errors[:1]
    assert not any(reader.is_alive() for reader in readers)
