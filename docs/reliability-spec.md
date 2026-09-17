# Session audit reliability (v0.2)

Source baseline: `ea39649`. Scope: detection and evidence; no action execution or automatic blocking.

| ID | Requirement | Acceptance |
|---|---|---|
| R1 | Lossless incremental JSONL | Both providers retain the first appended event, wait for complete lines, surface malformed lines, and retain long commands. |
| R2 | Event identity independent of embedding identity | Repeated calls remain separate, stable events; unchanged/same-content embeddings are reused. Provider/session boundaries are preserved. |
| R3 | Honest verdicts | Rank-only retrieval never causes suspicious/violation. Description-only, empty contracts, incomplete coverage and missing anchors yield unknown. Example rules are opt-in. |
| R4 | Contextual contracts | Source event, provider/session, roles and activation anchor are explicit. A must rule can require matching evidence before each trigger and a linked successful tool result. |
| R5 | Prospective audit | check_action syncs history and checks the proposed call without executing or recording it as an executed event. Returns coverage and contract snapshot. |
| R6 | Redacted external inputs | Mask before embedding documents and queries; preserve local evidence. No real sessions or live embeddings in tests. |
| R7 | Delivery | Regression tests, original tests, MCP smoke test, wheel build and documented limitations. Integrate reviewed diff into source working tree without moving its branch. |

The active event schema is separate from legacy v0.1 tables. Old metadata remains available in SQLite for recovery; syncing replays source JSONL into the new schema. Contracts are preserved. The next sync can rebuild a changed/truncated source session. Audits must never call stale/incomplete data clean.

Runtime: serialize engine operations with a reentrant lock and a local cross-process file lease; refresh in-memory vectors when another process updates them. Embedding errors must not prevent local regex auditing. External API output is validated before cache insertion. Index caches can be repaired by replaying events; they are not the audit source of truth.

Performance gate: 1,000 synthetic events sync + audit under 15 seconds locally, no embedding calls on unchanged second sync, no per-event rewriting of the vector archive. Record observed timings, not a production throughput claim.

Out of scope: a general shell parser, semantic entailment judge, authorization enforcement, human identity attestation, automatic rule extraction, remote deployment.
