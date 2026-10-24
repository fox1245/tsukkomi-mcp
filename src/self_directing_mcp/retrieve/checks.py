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
    if session_id is not None:
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
                        session_id=chunk.session_id,
                        ranking_score=1.0,
                        kind=chunk.kind,
                        snippet=mask_secrets(chunk.text)[:500],
                        timestamp=chunk.timestamp,
                    )
                )
            if len(hits) >= top_k:
                break
        return hits

    # Multi-session (family or global) regex search: cap per-session matches at top_k and interleave
    # round-robin across sessions so one verbose session never starves later subagent sessions.
    buckets: dict[str, list[SearchHit]] = {}
    for chunk in store.list_chunks(None, provider):
        if kind and chunk.kind != kind:
            continue
        bucket = buckets.get(chunk.session_id)
        if bucket is not None and len(bucket) >= top_k:
            continue
        try:
            matched = cre.search(chunk.text)
        except ContractRegexTimeout:
            raise ValueError("regex_timeout: the pattern exceeded its time limit; simplify it") from None
        if matched:
            buckets.setdefault(chunk.session_id, []).append(
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    session_id=chunk.session_id,
                    ranking_score=1.0,
                    kind=chunk.kind,
                    snippet=mask_secrets(chunk.text)[:500],
                    timestamp=chunk.timestamp,
                )
            )
    interleaved: list[SearchHit] = []
    queues = list(buckets.values())
    idx = 0
    while queues and len(interleaved) < top_k:
        next_queues = []
        for q in queues:
            if idx < len(q):
                interleaved.append(q[idx])
                if len(interleaved) >= top_k:
                    break
                if idx + 1 < len(q):
                    next_queues.append(q)
        queues = next_queues
        idx += 1
    return interleaved


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
