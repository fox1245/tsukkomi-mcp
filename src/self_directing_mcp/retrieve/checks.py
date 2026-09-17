"""Retrieval helpers and regex search utilities."""

from __future__ import annotations

import re

from self_directing_mcp.index.ingest import SessionStore
from self_directing_mcp.schemas import Chunk, SearchHit
from self_directing_mcp.security.mask import mask_secrets


def regex_search(
    store: SessionStore,
    pattern: str,
    *,
    session_id: str | None = None,
    top_k: int = 20,
    kind: str | None = None,
    provider: str | None = None,
) -> list[SearchHit]:
    try:
        cre = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    except re.error as e:
        raise ValueError(f"invalid regex: {e}") from e
    hits: list[SearchHit] = []
    for chunk in store.list_chunks(session_id, provider):
        if kind and chunk.kind != kind:
            continue
        if cre.search(chunk.text):
            hits.append(
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    ranking_score=1.0,
                    kind=chunk.kind,
                    snippet=mask_secrets(chunk.text)[:500],
                    timestamp=chunk.timestamp,
                )
            )
        if len(hits) >= top_k:
            break
    return hits


def scope_matches(chunk: Chunk, scope: str) -> bool:
    if scope == "any":
        return True
    if scope == "tool_call":
        return chunk.kind == "tool_call"
    if scope == "tool_result":
        return chunk.kind == "tool_result"
    if scope == "message":
        return chunk.kind == "turn"
    return True
