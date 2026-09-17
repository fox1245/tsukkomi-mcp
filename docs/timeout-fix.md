# Large-session timeout correction (v0.4)

Baseline: 90b74de (v0.3.0). Actual local diagnostics found approximately 21,000 indexed events in a roughly 92 MB session and a locked sparse database. No real transcript content is used in this repository's tests or benchmarks.

## Requirements and acceptance

| ID | Requirement | Check |
|---|---|---|
| T1 | Unchanged sync must not rewrite FTS | SQLite total_changes remains unchanged |
| T2 | New events only, with interrupted-write repair | One appended event updates one sparse document; missing FTS entries recover without new input |
| T3 | One hook sync | PreToolUse invokes sync once with the supplied validated path |
| T4 | Fast identity lookup | Persist chunk_id -> FTS rowid with a primary key; preserve existing FTS rowids when migrating |
| T5 | Lightweight coverage and descriptive checks | Incremental metrics; no full Chunk deserialization for description-only audits |
| T6 | Bounded contention and responsive transport | One-second combined local/file-lock wait; async MCP wrappers offload synchronous work; expired queued work does not start |
| T7 | Preserve verdict/evidence meaning | Partial evidence stays unknown; invalid supplied paths never fall back to clean; original regression tests still pass |
| T8 | Large concurrent regression | 20,000 4 KiB synthetic messages plus a small session: warm check <5 s, full audit <8 s, concurrent verified check including bounded retries <5 s |

The client MCP timeout remains 60 seconds and hook timeout remains 20 seconds. Internal local audit response budget defaults to 8 seconds; other tool responses default to 45 seconds. A busy/deadline response reports unknown, not clean, with incomplete coverage. An already-running indexing operation may finish bookkeeping after a response deadline; cancellation does not execute any proposed external action. New queued operations check their deadline before entering the engine.

Full-file fingerprints are still checked to detect transcript changes. This fix removes repeated FTS rewrites and unnecessary deserialization rather than weakening source verification. The FTS rowid map and session metrics are additive local migrations; existing evidence and contracts are preserved. Restart pre-v0.4 server processes before using the upgraded shared index.

## Local measured comparison

Same synthetic 83,908,974-byte transcript and indexed event set:

- v0.3 check_action: did not complete within 65 seconds; isolated baseline subprocess was terminated.
- v0.4 warm check_action: 0.336 s.
- v0.4 full audit_session: 1.566 s.
- Two processes / six checks: maximum 0.933 s.
- Cold sync of both sessions: 12.106 s.
- Unchanged FTS writes: zero.

These are bounded synthetic measurements on the development PC, not universal latency guarantees. The large-session gate is included in CI on Windows and Linux.

The concurrent gate exercises the public async MCP wrapper. It records explicit busy/unknown responses separately and allows at most three attempts per logical check, retaining the five-second end-to-end limit. Unknown outcomes unrelated to contention fail immediately; only a verified clean response counts as success. A Windows CI run exposed that the earlier direct-engine benchmark bypassed this documented busy-response handling.

Commands: python -m pytest -q; python scripts/benchmark_large.py; python -m build.
