"""Retrieval helpers and regex search utilities."""

from __future__ import annotations

import re

from self_directing_mcp.audit.patterns import ContractRegexTimeout, compile_contract

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
        cre = compile_contract(pattern)
    except re.error as e:
        raise ValueError(f"invalid regex: {e}") from e
    hits: list[SearchHit] = []
    for chunk in store.list_chunks(session_id, provider):
        if kind and chunk.kind != kind:
            continue
        try:
            matched = cre.search(chunk.text)
        except ContractRegexTimeout:
            # Partial results would read as "searched everything"; report the limit instead.
            raise ValueError("regex_timeout: the pattern exceeded its time limit; simplify it") from None
        if matched:
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
