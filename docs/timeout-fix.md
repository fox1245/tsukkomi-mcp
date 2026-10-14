# Shared-index concurrency and request deadlines

Current correction: [issue #3](https://github.com/fox1245/tsukkomi-mcp/issues/3), investigated against production source `7618028`. All reproductions use synthetic sources, isolated indexes and the real local engine/native graph. No live model call is needed to verify these boundaries.

## Protected state and execution boundaries

The engine retains its process-local lock and cross-process `.engine.lock` for initialization, scoped raw captures and publication. It no longer retains that lease across source parsing/hashing or detached audit evaluation. The source preparation owns a stable byte generation; commit compares exact cursor/parser/metrics/receipt inputs so delayed work cannot rewind newer state.

Deterministic audits refresh canonical metadata only. Their targeted snapshot contains immutable scoped raw event rows, a true event count, coverage, and detached rule content. JSON decoding, regex/temporal evaluation and request-local native graph stages run after the capture transaction/lease ends. Metadata, source observations, rule content and receipt authority are revalidated before publishing a current result. Changed/unavailable authority produces unknown/incomplete coverage; captured findings and any snapshot verdict remain evidence, not current compliance.

Explicit sync preserves delta FTS writes, content-addressed cache reuse and metadata-first interrupted repair. Sparse/hybrid search repairs a deferred FTS view from canonical metadata before querying. Derived preparation and remote work occur outside the canonical writer boundary; actual FTS/vector/NumPy persistence still needs guarded publication. Those writes and full raw captures are not claimed constant-time.

AGY metadata publication also records the receipt digest used with the cursor timestamp in `session_receipt_versions`. The existing six-column cursor schema remains intact. Receipt indexed markers are published conditionally afterward; interruption is replayable, and an older writer's unmatched cursor timestamp invalidates the stamp. This is an additive startup schema change, not a journal-mode migration.

## Request stopping and errors

The unchanged default combined process/file-lease wait is 1 second, audit response budget 8 seconds, ordinary budget 45 seconds, and client-provided tool cap 600 seconds. Per-call `timeout_ms` does not extend the separate writer-lock cap.

Ordinary requests/hooks share an absolute monotonic deadline and cancellation control with nested operations and native callbacks. Parsing, row/batch work, evaluation, and final publication check that control. Cancellation and publication admission are ordered by a tiny request-local gate released before I/O. An already-admitted commit may physically finish afterward; no response asserts rollback or thread termination.

| Error | Meaning |
| --- | --- |
| `index_busy` | A live request exhausted the engine/file writer-lease wait |
| `request_deadline` | The request's response/execution deadline expired |
| `request_cancelled` | Its cancellation state was observed |
| `operation_timeout` | A worker raised a separate, unclassified timeout |

These responses are unknown/incomplete, not clean. Unrelated OS/SQLite errors remain failures; sparse search does not turn them into empty successful results. Native callbacks retain original stop exceptions. The owned workflow dispatch still records actual started process outcomes and cleanup receipts; request stopping must not erase them.

Cooperative checks cannot preempt a single regex/JSON/native computation, SQLite statement/commit, HTTP call or filesystem syscall already in progress. They prevent later phases/publications after abandonment is observed. Genuine initialization, raw capture, FTS/vector mutation or external writer contention can still exhaust the configured wait.

## Executed verification

The bounded before/after diagnostic paused actual parser/evaluator boundaries, not mocked responses:

| Scenario | Before | After |
| --- | --- | --- |
| Resume parser after a 50 ms expired response | Two events and cursor published later | No events/cursor published; `request_deadline` |
| Independent status during paused pure audit evaluation | Writer lease failed after about 1 second | Actual status completed while evaluation stayed paused |
| Remove an already-loaded rule file | Cached rules yielded clean | Unknown, no cached rule acceptance |

Permanent regressions cover snapshot rule add/upsert/revoke/loss/corruption, source changes, scoped completion evidence, request isolation, stale preparations, AGY receipt generations, derived repair and distinct failure classes. Actual MCP stdio smoke exercised 16 concurrent clean requests, successful prerequisite evidence, a newer failed prerequisite, and a forbidden proposal without executing proposed actions.

The native queue PoC reports writer lease and evaluation activity separately and checks overlapping real native evaluations plus independent metadata progress. Its deliberate writer-hold fixture demonstrates residual serialization, not a universal throughput bound. It uses private temporary endpoint credentials, naturally shuts down its services and removes scratch state.

```bash
python -m pytest -q tests/test_index_concurrency.py tests/test_timeout_regression.py tests/test_tool_timeout_arg.py tests/test_mcp_transport.py
python scripts/poc_shared_index_queue.py run --neograph-root ../NeoGraph --workers 4
python -m build
```

These are local deep checks. CI remains only the explicit lightweight transport/JEV/session-path smoke files and distribution build; no performance, large-session, native compilation or paid-model gate is added.

## Deployment and historical measurements

Every process sharing an index must use the updated code from its actual configured source/package and be restarted through its owner's normal deployment procedure. Updating a checkout or passing a temporary-index PoC does not update running servers. This work does not terminate deployment processes or rewrite their indexes.

Historical v0.3/v0.4 comparison: a synthetic 83,908,974-byte transcript reported a v0.4 warm action check of 0.336 s, full audit of 1.566 s and cold sync of 12.106 s. Those measurements were made before this snapshot/request-control correction and are not current performance guarantees or CI gates.
