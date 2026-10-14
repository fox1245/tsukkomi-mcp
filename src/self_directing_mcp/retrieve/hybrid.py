from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from self_directing_mcp.embed.embedder import Embedder
from self_directing_mcp.index.sparse import SparseIndex
from self_directing_mcp.index.vector import VectorIndex
from self_directing_mcp.schemas import SearchHit
from self_directing_mcp.request_control import check_request, request_operation


@dataclass
class RRFResult:
    chunk_id: str
    ranking_score: float
    dense_rank: int | None
    sparse_rank: int | None
    dense_score: float | None = None
    sparse_score: float | None = None


def reciprocal_rank_fusion(
    dense: Sequence[tuple[str, float]],
    sparse: Sequence[tuple[str, float, int]],
    *,
    k: int = 60,
) -> list[RRFResult]:
    dense_rank: dict[str, int] = {}
    dense_score: dict[str, float] = {}
    for i, (cid, score) in enumerate(dense, start=1):
        check_request()
        if cid not in dense_rank:
            dense_rank[cid] = i
            dense_score[cid] = score

    sparse_rank: dict[str, int] = {}
    sparse_score: dict[str, float] = {}
    for cid, score, rank in sparse:
        check_request()
        if cid not in sparse_rank:
            sparse_rank[cid] = rank
            sparse_score[cid] = score

    fused: list[RRFResult] = []
    for cid in set(dense_rank) | set(sparse_rank):
        check_request()
        score = 0.0
        dr = dense_rank.get(cid)
        sr = sparse_rank.get(cid)
        if dr is not None:
            score += 1.0 / (k + dr)
        if sr is not None:
            score += 1.0 / (k + sr)
        fused.append(
            RRFResult(
                chunk_id=cid,
                ranking_score=score,
                dense_rank=dr,
                sparse_rank=sr,
                dense_score=dense_score.get(cid),
                sparse_score=sparse_score.get(cid),
            )
        )
    fused.sort(
        key=lambda r: (
            -r.ranking_score,
            r.dense_rank if r.dense_rank is not None else 10**9,
            r.chunk_id,
        )
    )
    return fused


class HybridRetriever:
    def __init__(
        self,
        *,
        sparse: SparseIndex,
        dense: VectorIndex,
        embedder: Embedder,
        rrf_k: int = 60,
        retrieve_top_k: int = 20,
        store=None,
    ) -> None:
        self.sparse = sparse
        self.dense = dense
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.retrieve_top_k = retrieve_top_k
        self.store = store

    @request_operation
    def prepare_query(self, query: str, mode: str = "hybrid") -> np.ndarray | None:
        """Prepare remote work before the caller acquires its storage lease."""
        if mode == "sparse":
            return None
        from self_directing_mcp.security.mask import mask_secrets
        return self.embedder.embed_queries([mask_secrets(query)])[0]

    @request_operation
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        session_id: str | None = None,
        mode: str = "hybrid",
        provider: str | None = None,
        query_vector: np.ndarray | None = None,
    ) -> list[RRFResult]:
        top_k = top_k if top_k is not None else self.retrieve_top_k
        channel_k = max(self.retrieve_top_k, top_k)
        allowed = set(self.store.chunk_ids(session_id, provider)) if self.store else None
        check_request()
        from self_directing_mcp.security.mask import mask_secrets
        query = mask_secrets(query)

        if mode == "sparse":
            sparse_hits = self.sparse.search(query, top_k=channel_k, session_id=session_id, allowed_ids=allowed)
            return [
                RRFResult(chunk_id=cid, ranking_score=score, dense_rank=None, sparse_rank=rank, sparse_score=score)
                for cid, score, rank in sparse_hits[:top_k]
            ]

        if mode == "dense":
            qvec = query_vector if query_vector is not None else self.prepare_query(query, mode)
            dense_hits = self.dense.search(qvec, top_k=channel_k, allowed_ids=allowed)
            return [
                RRFResult(chunk_id=cid, ranking_score=score, dense_rank=i, sparse_rank=None, dense_score=score)
                for i, (cid, score) in enumerate(dense_hits[:top_k], start=1)
            ]

        # hybrid (default)
        sparse_hits = self.sparse.search(query, top_k=channel_k, session_id=session_id, allowed_ids=allowed)
        qvec = query_vector if query_vector is not None else self.prepare_query(query, mode)
        dense_hits = self.dense.search(qvec, top_k=channel_k, allowed_ids=allowed)
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=self.rrf_k)
        return fused[: max(top_k, 1)]


def hits_to_schema(results: list[RRFResult], get_chunk) -> list[SearchHit]:
    from self_directing_mcp.security.mask import mask_secrets

    out: list[SearchHit] = []
    for r in results:
        check_request()
        chunk = get_chunk(r.chunk_id)
        snippet = mask_secrets(chunk.text)[:500] if chunk else None
        out.append(
            SearchHit(
                chunk_id=r.chunk_id,
                ranking_score=r.ranking_score,
                dense_rank=r.dense_rank,
                sparse_rank=r.sparse_rank,
                dense_score=r.dense_score,
                sparse_score=r.sparse_score,
                kind=chunk.kind if chunk else None,
                snippet=snippet,
                timestamp=chunk.timestamp if chunk else None,
            )
        )
    return out
