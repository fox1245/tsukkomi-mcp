"""Seeded synthetic vectors and an exact-cosine oracle for comparing dense backends (shared by the pgvector integration,
parity and benchmark code). Everything here is synthetic."""
from __future__ import annotations

import numpy as np


def normalized_vectors(count: int, dim: int, seed: int) -> np.ndarray:
    """``count`` seeded unit vectors of float32 (the form an embedder hands the index)."""
    matrix = np.random.default_rng(seed).standard_normal((count, dim)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix


def make_ids(count: int, prefix: str = "c") -> list[str]:
    return [f"{prefix}{n:07d}" for n in range(count)]


def exact_cosine(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine of every row with ``query`` in float64 (the reference the backends are compared with)."""
    m = np.asarray(matrix, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)
    denominator = np.linalg.norm(m, axis=1) * np.linalg.norm(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = (m @ q) / denominator
    return np.where(denominator == 0, 0.0, scores)


def assert_same_topk(got, ids: list[str], scores: np.ndarray, k: int, *, allowed=None, tolerance: float = 1e-5,
                     label: str = "") -> None:
    """``got`` (a list of ``(id, score)``) is a correct exact top-``k`` of ``ids`` scored by ``scores``.

    Correct means: as many hits as the population allows, no id twice, every id allowed, every reported score within
    ``tolerance`` of the reference score of that id, and the reference scores of the hits equal the reference top-``k``
    scores rank by rank within ``tolerance`` (so hits may differ from the reference ids only inside a group of equal
    scores, which have no defined order). With no near-ties around the top ``k`` the ids must be the reference ids in
    the reference order."""
    index_of = {cid: n for n, cid in enumerate(ids)}
    pool = [n for n, cid in enumerate(ids) if allowed is None or cid in allowed]
    ranking = sorted(pool, key=lambda n: (-scores[n], ids[n]))
    expected = ranking[:k]
    where = f"{label}: " if label else ""
    assert len(got) == len(expected), f"{where}{len(got)} hits, expected {len(expected)}"
    assert len({cid for cid, _ in got}) == len(got), f"{where}an id is returned twice"
    for cid, score in got:
        assert allowed is None or cid in allowed, f"{where}{cid} is not allowed"
        assert abs(score - scores[index_of[cid]]) <= tolerance, (
            f"{where}score of {cid}: {score!r} vs reference {scores[index_of[cid]]!r}")
    reference_scores = [scores[n] for n in expected]
    hit_scores = [scores[index_of[cid]] for cid, _ in got]
    assert all(abs(a - b) <= tolerance for a, b in zip(hit_scores, reference_scores)), (
        f"{where}the hits are not a top-{k}: {hit_scores} vs {reference_scores}")
    near_tie = any(abs(scores[a] - scores[b]) <= 2 * tolerance for a, b in zip(ranking[:k + 1], ranking[1:k + 2]))
    if not near_tie:
        assert [cid for cid, _ in got] == [ids[n] for n in expected], f"{where}ids differ without a tie"
