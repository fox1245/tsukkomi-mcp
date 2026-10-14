# Shared-index concurrency and request deadlines

Current correction: [issue #3](https://github.com/fox1245/tsukkomi-mcp/issues/3), investigated against production source `7618028` and shipped in 0.6.0. The 0.6.0 follow-ups ([#4](https://github.com/fox1245/tsukkomi-mcp/issues/4), [#5](https://github.com/fox1245/tsukkomi-mcp/issues/5), [#6](https://github.com/fox1245/tsukkomi-mcp/issues/6), [#8](https://github.com/fox1245/tsukkomi-mcp/issues/8), [#10](https://github.com/fox1245/tsukkomi-mcp/issues/10)) are summarized at the end of this page. All reproductions use synthetic sources, isolated indexes and the real local engine/native graph. No live model call is needed to verify these boundaries.

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
| `request_deadline` | The request's response/execution deadline expired, including while it waited for a heavy-call slot; a call that expired in that queue never started |
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

These are local deep checks. CI runs the lightweight transport/JEV/session-path smoke files with a distribution build and, since 0.6.0, a separate Linux `memory-audit-regressions` job of small offline tests (memory, prefix freshness, append capture, single-flight, concurrency and deadlines). No performance, large-session, native compilation or paid-model gate is added.

## 0.6.0 follow-ups

These build on the snapshot and request-control work above.

- **Heavy-call gate ([#5](https://github.com/fox1245/tsukkomi-mcp/issues/5)).** Sync, search, audit, `check_action`, checklist, graph, activity and hook calls run in worker threads, at most `SELF_DIRECT_MAX_CONCURRENT_CALLS` (default 4) at a time. Waiting for a slot spends the request's own deadline. A slot is released when the worker thread really finishes, not when its caller gives up. `list_contracts`, `upsert_contracts`, `revoke_contract`, `get_chunk`, `audit_status` and `workflow_*` calls bypass the gate. After the last running heavy call that took at least 0.5 s (at most once every 5 s), glibc Linux processes call `malloc_trim(0)`.
- **Single-flight and streaming.** Capture, parse and publication for one (index, provider, session id) run one at a time in a process, so parallel hooks no longer parse the same session repeatedly. Source capture hashes the file without retaining its bytes and tolerates appended records (`unread_bytes`). Reparses spool rows to disk and publish them in batches. Cross-process safety is still the metadata state-token compare-and-swap at publication, and a capture older than a cursor another process already published is retaken.
- **Audit evaluation of growing sessions ([#4](https://github.com/fox1245/tsukkomi-mcp/issues/4), [#8](https://github.com/fox1245/tsukkomi-mcp/issues/8)).** The final source check re-hashes the captured prefix outside the lock and re-checks only identity, size and the last 64 KiB under it. Pure growth is judged as of the captured offset; any other change is `source_changed_during_evaluation` and `unknown`. See the README's "Coverage and verdicts".
- **`audit_status`.** The sparse count reads the `chunk_rows` mirror table instead of scanning the FTS5 table under the shared lock. The status call itself bypasses the gate.
- **Checklist conflicts ([#6](https://github.com/fox1245/tsukkomi-mcp/issues/6)).** Plain checklist edits run inside one publication guard. An update that needs evidence evaluation retries three times, then returns `checklist_conflict` without saving.
- **Measured effect.** On a 100 MB synthetic OMP session with 8 concurrent threads, peak RSS fell from 2,304 MB to 193 MB and the first sync no longer hit `index_busy` (PR #7; table in the README).

## Deployment and historical measurements

Every process sharing an index must use the updated code from its actual configured source/package and be restarted through its owner's normal deployment procedure. Updating a checkout or passing a temporary-index PoC does not update running servers. This work does not terminate deployment processes or rewrite their indexes. A parser-version bump (OMP 8, Codex 3) then reparses each session lazily on its next sync.

Historical v0.3/v0.4 comparison: a synthetic 83,908,974-byte transcript reported a v0.4 warm action check of 0.336 s, full audit of 1.566 s and cold sync of 12.106 s. Those measurements were made before this snapshot/request-control correction and are not current performance guarantees or CI gates.
