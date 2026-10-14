# Tsukkomi MCP (ツッコミ)

> A deterministic governance and evidence server for autonomous coding agents, version 0.6.0.
> The distribution is `tsukkomi-mcp`; the Python package is `self_directing_mcp`.

## Overview

Prompt instructions alone do not keep an agent from context drift, false completion reports or a forbidden command. Tsukkomi MCP indexes the agent's own session history on the local machine, checks proposed actions and finished work against explicit, user-registered contracts, tracks requirement checklists that need linked evidence, and optionally maintains a session knowledge graph.

Every check returns one of four verdicts: `clean`, `violation`, `suspicious` or `unknown`. `unknown` means the evidence needed for a verdict is missing or has changed, and it is never reported as `clean`. The one incomplete input that can still yield `clean` is a pure append to a growing session, which is judged as of the captured offset and reports that offset (see [Coverage and verdicts](#coverage-and-verdicts)). Whether a verdict blocks anything depends on the host (see [Supported hosts](#supported-hosts-and-parser-versions)). Rules are matched with regular expressions and structured evidence, not with an LLM judge.

```
                  ┌──────────────────────────────────────────────┐
                  │          Autonomous Agent Client             │
                  │ (Claude Code / AGY / Codex / OMP / Grok Bot) │
                  └──────────────────────┬───────────────────────┘
                                         │
                      Lifecycle hooks (host dependent) or MCP calls
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                           Self-Directing MCP Engine                            │
├───────────────────────┬────────────────────────┬───────────────────────────────┤
│  Invariant Contracts  │   Verifiable Evidence  │      Session GraphRAG         │
│  - must / must_not    │   Checklist Gate       │  - Entity dependencies        │
│  - Session isolation  │   - No done without    │  - NeoGraph engine            │
│  - Regex + temporal   │     linked evidence    │  - Context injection (AGY)    │
├───────────────────────┴────────────────────────┴───────────────────────────────┤
│                 Local session index (SQLite, FTS5, sqlite-vector)              │
│     - Cosine vector search  - FTS5 BM25 and regex  - Known-secret masking      │
└────────────────────────────────────────────────────────────────────────────────┘
```

Session evidence stays on the local machine. Remote calls are limited to embeddings, GraphRAG extraction and the opt-in workflow classifier, all sent to OpenRouter. The canonical index keeps raw text so audits see the truth; everything that leaves it (tool results, errors, hook output, logs, remote payloads) is masked for known secret patterns, and the workflow classifier refuses context in which a secret pattern is found. See the [masking contract](SECURITY.md#masking-contract), [SECURITY.md](SECURITY.md) and [Verification scope](#verification-scope).

## What is new in 0.6.0

Version 0.6.0 (see [CHANGELOG.md](CHANGELOG.md)) mainly fixes memory growth, append handling and concurrency in long-running servers. It also adds a Claude Code integration, a shared hook policy and an egress masking contract.

- **Dependency:** `neograph-engine==0.14.1` (was 0.12.1). `zstandard>=0.23` is new on Python before 3.14, for compressed Codex sessions.
- **Compressed Codex sessions.** Codex CLI 0.156 compresses older rollouts in place to `rollout-*.jsonl.zst`. They are discovered, synced, audited and searched like plain rollouts, streamed through a bounded decoder and never held in memory. See [Compressed Codex sessions](#compressed-codex-sessions).
- **Appending sessions sync normally ([#4](https://github.com/fox1245/tsukkomi-mcp/issues/4)).** Source capture streams a SHA-256 over the file up to its length at the first measurement and keeps no bytes. Records appended afterwards are reported as `unread_bytes` instead of failing the sync.
- **Audits judge a growing session as of the captured offset.** Results carry `judged_through_byte`, `unread_bytes` and `judged_prefix`; absence-based `must` obligations stay `unknown` while bytes are unread. See [Coverage and verdicts](#coverage-and-verdicts).
- **`check_action` does not trust older history for a proposed action ([#8](https://github.com/fox1245/tsukkomi-mcp/issues/8)).** A prerequisite that requires success stays `unknown` while required history is unread, because a newer failed run may be in the unread part.
- **Bounded memory ([#5](https://github.com/fox1245/tsukkomi-mcp/issues/5)).** Single-flight sync per session, streaming reparse, per-session id queries, float32 vectors, a cap on concurrent heavy calls (`SELF_DIRECT_MAX_CONCURRENT_CALLS`, default 4) and `malloc_trim` on glibc. See [Memory and concurrency](#memory-concurrency-and-deadlines).
- **Concurrent checklist updates ([#6](https://github.com/fox1245/tsukkomi-mcp/issues/6)).** `update_checklist` serializes plain edits and returns a `checklist_conflict` status instead of raising or overwriting.
- **Duplicate session sources ([#10](https://github.com/fox1245/tsukkomi-mcp/issues/10)).** Copies and continuations of an indexed session are recognised instead of reparsed on every sync (`alias_of`, `ambiguous_session_source`).
- **Parsers.** OMP parser v8 (host notices, images, steering attribution, `session_init`, `model_usage`, `branch_summary`, `redactedThinking`; [#9](https://github.com/fox1245/tsukkomi-mcp/issues/9)) and Codex parser v3 (multi-agent messages, tool search, new `session_meta` fields). Each bump reparses lazily, one session at a time.
- **CI.** A new Linux `memory-audit-regressions` job runs small offline memory, freshness and concurrency tests next to the existing smoke job.
- **Claude Code ([docs/claude-installation.md](docs/claude-installation.md)).** A `claude` provider (parser version 2) for Claude Code session and subagent transcripts, and opt-in command hooks installed by `scripts/install_claude_hooks.py` for `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `Stop` and `SubagentStop`. By default (`SELF_DIRECT_CLAUDE_ENFORCEMENT=confirmed_only`) they deny only a confirmed `violation` and ask on `suspicious`; they never answer `allow`, `SubagentStop` only reports, and every internal error fails open with a visible warning.
- **One hook policy for AGY and Claude Code ([docs/agy-installation.md](docs/agy-installation.md)).** The AGY hook default is now `confirmed_only` (it was `enforced`): `unknown` passes with a visible warning unless the contract is marked `risk: "high"`, a `Stop` continues the agent at most once per conversation and only for a confirmed `auto_continue` miss, and hook errors fail open with a `hook_health` banner. `enforced` stays an explicit opt-in, and `advisory` never gates an approved workflow either.
- **AGY parser version 7.** Truncated fields fall back to the same step of `transcript_full.jsonl`; what stays cut is `unknown`, never `clean`. `RUNNING`, `ERROR`, `INVALID`, `CHECKPOINT` and media records are handled, and results from before a conversation's first hook receipt are classified as `legacy_unlinked`. Version 7 adds the Antigravity IDE's typed result steps (`RUN_COMMAND` with its host `exit_code`, `VIEW_FILE`, `CODE_ACTION`, `MCP_TOOL`, `LIST_DIRECTORY`) and context-loading steps (`CONVERSATION_HISTORY`, `KNOWLEDGE_ARTIFACTS`), and accepts a host-written run of NUL bytes in front of an otherwise valid record.
- **Contract fields `risk`, `auto_continue` and `when`**, and four disabled high-risk presets. See [Contract fields](#contract-fields-risk-auto_continue-and-when).
- **Masking contract.** Every egress path is masked, the sparse text index is masked for new rows, and a logging filter covers server logs. See the [masking contract](SECURITY.md#masking-contract).
- **Also in this release since 0.5.0:** the shared-index snapshot and request-deadline work ([#3](https://github.com/fox1245/tsukkomi-mcp/issues/3)), the AGY evidence workflow and approved-workflow layer, user path configuration in TOML, and a default index directory outside the checkout. Read [Upgrading to 0.6.0](#upgrading-to-060) before restarting existing servers.

---

## Key capabilities

### 1. Invariant contracts (`upsert_contracts`)
- **Prohibitions (`must_not`)** match a regex against proposed or recorded actions (`decision: deny` on hosts that enforce).
- **Obligations (`must`)** require matching evidence before a gated action (`before_regex`) or at a completion checkpoint, optionally with a successful linked tool result (`requires_success: true`).
- AGY returns `force_ask` for suspicious exceptions so a cached Always Allow grant does not replace confirmation; Claude Code, which has no force-ask, returns `ask`.
- Contracts are global or scoped to one session. A description-only contract stays `unknown`, and search rankings are never violation evidence.

#### Contract fields: `risk`, `auto_continue` and `when`

Three optional fields refine how hosts act on a contract. They never change what a regex matched. At their defaults they are omitted from the stored contract, so existing `contracts.json` files load and are rewritten unchanged and older processes keep reading files that do not use them.

| Field | Values | Applies to | Effect |
|---|---|---|---|
| `risk` | `normal` (default) or `high` | any contract | Declares that the actions the contract applies to are high-risk. When such an action cannot be verified (`unknown`), the AGY and Claude Code hooks ask the user instead of silently allowing it (`confirmed_only`), including after an internal hook error unless the mode is `advisory`. It does not change the verdict. Codex hooks are advisory and the OMP extension blocks only confirmed findings, so `risk` does not change their behavior. |
| `when` | `{"tool_call_regex": "<regex>"}` | `must` only | The obligation applies only if some tool call observed earlier in the same session matches the regex (case-insensitive, multiline). Otherwise the finding is reported as `clean` with `applicability: "not_applicable"` and a reason, never dropped. While coverage is incomplete, an unmet condition becomes `unknown` because the call may be in unread evidence. An AGY model proposal counts as the condition being met. |
| `auto_continue` | `true` or `false` (default) | `must` with a `regex` and no `before_regex` | A confirmed, fully observed miss of this completion obligation may resume a stopping agent: once per conversation on AGY (`continue`) and once per session on Claude Code (`block`). Unknown evidence, partial coverage and hook errors never resume it. Any other contract shape is rejected when it is saved. |

A high-risk `must_not` must have a `regex` naming the actions it covers: a description-only one would apply to every tool call and could never be verified, so every call would be asked (a denial in headless runs). `upsert_contracts` rejects it.

Contract regexes (`regex`, `before_regex`, `exception_regex`, `when.tool_call_regex`) and `search_history` regex queries run on the `regex` engine with a time limit: 0.25 s per search and 2 s per contract per evaluation. A pattern that exceeds it makes only that contract `unknown` (`regex_timeout`), never clean or a violation, so a pathological pattern cannot stall a hook past the host's timeout. The syntax is still Python `re` syntax.

Set these only when the user asked for them. The high-risk presets in [`src/self_directing_mcp/resources/high_risk_presets.json`](src/self_directing_mcp/resources/high_risk_presets.json) are four example `must_not` contracts with `risk: "high"`: `preset-no-recursive-delete`, `preset-no-write-outside-workspace` (a path-pattern approximation to replace with your own boundary), `preset-no-force-push` and `preset-no-branch-delete`. They ship with `enabled: false` and exist in the contract store only when `SELF_DIRECT_SEED_EXAMPLE_CONTRACTS=true`, where they are added if their id is missing and are never altered or re-enabled. The owner enables the ones they want.

### 2. Checklist with evidence gating (`update_checklist`)
- Track user requirements with provenance and completion criteria.
- A completion claim without evidence is downgraded to `pending_verification`. An item becomes `done` (`verified=true`) only when its `evidence_chunk_ids` resolve to successful tool results linked to their calls **and** session coverage is complete. While appended bytes are unread, coverage is incomplete, so the item stays `pending_verification` until a later call sees the caught-up history.
- Concurrent updates do not lose each other's edits. An update that needs evidence evaluation is retried up to three times; if the checklist or evidence keeps changing, the call returns `{"ok": false, "status": "checklist_conflict", "action_executed": false}` and saves nothing.

### 3. Session GraphRAG and context injection (`get_graph_context`)
- Entity dependency graph (nodes, relations, cursors) in SQLite, executed through the **NeoGraph** engine.
- Typed relations: `IMPLEMENTS`, `DEPENDS_ON`, `MODIFIES`, `PRODUCES`, `CHECKS_VERSION`, `EVIDENCED_BY`, `SUPERSEDES`.
- Graph updates are validated against session history and applied idempotently. Updates require an explicit `run_neograph_update` call; lifecycle hooks never send session text to a model.
- In Google Antigravity, active graph relationships are delivered into agent context before each tool invocation.

### 4. Hybrid history search
- Dense cosine similarity (`sqlite-vector`, native), SQLite FTS5 BM25 and regex, fused with reciprocal-rank fusion.
- **Secret masking:** known credential patterns are masked before embedding, in sparse search text and in everything returned to the client. This is not a guarantee that arbitrary personal data or every secret format is removed; see the [masking contract](SECURITY.md#masking-contract).

---

## Supported hosts and parser versions

| Host | Provider id | Evidence source | Lifecycle integration | Blocks actions? | Parser version |
|---|---|---|---|---|---|
| Google Antigravity (AGY) | `agy` | `brain/<conversation-id>/.system_generated/logs/transcript*.jsonl` plus hook receipts | Stdio hooks: `PreToolUse`, `PostToolUse`, `PreInvocation`, `Stop` | By default (`confirmed_only`) only a confirmed `violation` (`deny`) and `suspicious` (`force_ask`); `unknown` passes with a warning, or asks for a `risk: "high"` contract. `advisory` and `enforced` are explicit. See [docs/agy-installation.md](docs/agy-installation.md) | 7 |
| Claude Code | `claude` | `<projects>/<project>/<session-uuid>.jsonl` plus `<session-uuid>/subagents/agent-*.jsonl` | Opt-in command hooks from `scripts/install_claude_hooks.py`: `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `Stop`, `SubagentStop` | By default (`confirmed_only`) only a confirmed `violation` (`deny`) and `suspicious` (`ask`); never an explicit `allow`. A `Stop` blocks at most once per session for a confirmed `auto_continue` miss. `SubagentStop` never blocks. See [docs/claude-installation.md](docs/claude-installation.md) | 2 |
| OpenAI Codex CLI | `codex` | Rollout JSONL in the Codex sessions directory, plain or zstd-compressed (`rollout-*.jsonl.zst`) | Three `mcp_tool` hooks installed by `scripts/install_codex.py`: `UserPromptSubmit`, `PreToolUse`, `Stop` | No. Hooks are advisory | 3 |
| OMP | `omp` | OMP session JSONL | Opt-in extension `extensions/tsukkomi-omp.ts` | Only confirmed `violation`/`suspicious` on model-issued tools. `unknown` and transport failures never block | 8 |
| Cursor / Grok Bot | `grokbot` | `agent-transcripts` roots | None; call the MCP tools manually | No | 1 (unversioned) |

A parser version is recorded per session. When a provider's version is raised, each already-indexed session is reparsed from byte 0 on its next sync or audit, one session at a time. Unchanged event ids, anchors and cached vectors survive, and no embedding API is called when `embed=false`. A running MCP process only picks up a new parser after it is restarted.

### AGY parser (version 7)

Fields the host cut (`truncated_fields`) are restored from the same step of the sibling `transcript_full.jsonl` when it matches; otherwise the visible text is kept, marked as truncated, and any verdict that depends on the cut field is `unknown` with a `truncated_<field>` reason. A match that is visible still counts. `RUNNING` results are pending (`pending_running`), `ERROR`, `INVALID` and `CANCELED` results are never success, `CHECKPOINT` and `ERROR_MESSAGE` records are runtime context, and media payloads become placeholders. Results from before the conversation's first hook receipt are `legacy_unlinked` and leave dependent obligations `unknown` (`unsupported_legacy`); coverage reports `receipts_since`, `legacy_unlinked_count` and `result_dependent_complete`. Details are in [docs/agy-installation.md](docs/agy-installation.md#parser-version-6-truncated-fields-result-statuses-and-legacy-results).

Version 7 reads two more shapes. The IDE writes a tool result as a step of its own type, `RUN_COMMAND`, `VIEW_FILE`, `CODE_ACTION`, `MCP_TOOL` or `LIST_DIRECTORY`, and the parser treats it exactly like the CLI's `GENERIC` result: it is linked to its call only through a host receipt, truncated content falls back to `transcript_full.jsonl` or leaves dependent verdicts `unknown`, and `RUNNING` is pending. A linked `RUN_COMMAND` with status `DONE` and host `exit_code` 0 is success evidence, a nonzero `exit_code` is a failure, and an unlinked result is never success. `CONVERSATION_HISTORY` and `KNOWLEDGE_ARTIFACTS` are runtime metadata. A contiguous run of NUL bytes at the start of a line is stripped when the rest is a valid record (`host_padding_bytes`, counted in `agy_evidence.host_padded_records`); any other NUL use, and any other unknown step type, stays a parse error or an unsupported event. See [docs/agy-installation.md](docs/agy-installation.md#parser-version-7-ide-result-steps-context-steps-and-host-padding).

Hook receipts cost what was appended, not the size of the transcript. A receipt hashes the file once and parses only the records after a prefix it has verified; a sync checks every receipt anchor in the pass that hashes the file; and a receipt reindexes the conversation only when it can change an event that is already indexed. On a synthetic 10 MB transcript with a linked receipt per result, one `PreToolUse` hook (receipt and contract check) fell from 20.5 s to 0.51 s and a `PostToolUse` hook (receipt and sync) from 20.4 s to 0.58 s, against 0.38 s and 0.36 s at 0.1 MB (whole hook processes, wall time). The mechanisms and what they fall back to are in [AGY hook cost](docs/agy-installation.md#hook-cost-what-a-hook-reads); `scripts/bench_agy_hooks.py` reproduces the table.

### Claude Code parser (version 2)

Only a record the host stamped `origin.kind == "human"` (or, in a main transcript, a legacy prompt from before origin stamps, or a human `queued_command` prompt) is a user turn. Task notifications, peer and coordinator messages, injected meta messages, compaction summaries, attachments, hook output, tool results and a subagent's task prompt are runtime evidence. Unknown record, attachment, block and origin shapes are unsupported events that keep coverage incomplete. Opaque payloads (thinking, images, hook output) are placeholders. A subagent transcript is its own session, `<session-uuid>_agent-<agent-id>`. Version 2 accepts the `system` record `scheduled_task_fire` (a scheduled task fired) as runtime metadata: its `prompt` is automation text, not something the user said, so it is validated, never indexed, and cannot satisfy or violate a user-role contract; every other unknown system subtype stays unsupported. See [docs/claude-installation.md](docs/claude-installation.md#9-how-the-claude-parser-decides-who-is-the-user).

### OMP parser (version 8)

Runtime records are indexed as metadata or `runtime`-role events. They are never user instructions and never proof that a tool ran.

- Host notices, `mode_change`, `ttsr_injection`, developer messages, `session_init`, `model_usage` and `branch_summary` are runtime metadata. Only the parent's task text of `session_init` is searchable; its prompt, tools and schema are not indexed.
- Image blocks become `[image:<mime-type>]` placeholders and `thinking`/`redactedThinking` blocks become placeholders; base64 and opaque reasoning payloads are never indexed, including inside blocks of an unsupported shape (only their key names are kept).
- **Attribution:** among steering messages only `attribution: "user"` is a user turn. Any user-role message with `attribution: "agent"`, with or without steering metadata, is runtime text and cannot satisfy or violate a user-role contract.
- Partial steering metadata (`steering` or `liveSteered` without the full valid shape) and unknown attribution values are rejected as parse errors, which makes coverage incomplete.
- Unknown event shapes and unmatched execution-start markers still make coverage incomplete. `service_tier_change` is metadata only when `serviceTier` is `null` or a map from the known providers (`openai`, `anthropic`, `google`) to known tiers.

### Codex parser (version 3)

- `response_item` `agent_message` (multi-agent messaging) and `inter_agent_communication_metadata` are runtime metadata, neither user authority nor tool evidence. `encrypted_content` blocks become a placeholder.
- `tool_search_call` / `tool_search_output` are runtime metadata. The search query is indexed; the returned `tools` list is not.
- `session_meta` headers, including the newer fields, are shape- and type-checked, and only a fixed subset of identifying fields is indexed. `base_instructions`, `dynamic_tools`, `context_window`, `git`, `history_base`, account ids and workspace roots are never indexed. The older behaviour of indexing a legacy header's whole payload is gone.
- A complete new-style `session_meta` for another thread in the middle of a sub-agent file (an inherited parent header) is accepted as inherited metadata. A first-line id mismatch is still an error.
- For these newly supported record types, an envelope with `metadata.client_authored: true` is treated as unsupported because its meaning is unclear.
- Earlier support remains: `world_state`, `token_usage_record`, `compacted` and native `web_search_call` records. Unknown event types stay visible as unsupported.

### Compressed Codex sessions

Codex CLI 0.156 compresses older rollouts in place: `sessions/YYYY/MM/DD/rollout-<timestamp>-<uuid>.jsonl.zst`, one zstd frame that declares its content size. A forked or sub-agent rollout may carry a second `_<uuid>` suffix. Session discovery, `session_id` lookup and explicit `path` arguments accept `rollout-*.jsonl.zst` under the configured Codex root with the same root, symlink and traversal checks as `.jsonl`. The session id comes from the `session_meta` header, as for plain files; a file name with two different UUIDs never names a session by itself. While a plain rollout and its compressed twin both exist, a lookup by id returns the plain (live) one.

- **Offsets address the decompressed stream.** Event ids, anchors, `byte_offset`, `judged_through_byte` and the stored prefix hash are exactly what the plain file with the same bytes yields, so Codex compressing a rollout in place, or a resume decompressing it and appending, is the same session: nothing is reparsed, no event id, anchor or cached vector changes, and no duplicate session or `ambiguous_session_source` appears. If the compressed file holds more records than the plain file had when it was indexed, only the new part is parsed. Plain and compressed copies that exist together follow the [duplicate source](#duplicate-session-sources) rules.
- **A compressed file is an immutable snapshot.** There is no growing tail: coverage is complete when the stream has been read to its end (`unread_bytes` stays absent unless the file ends with an unterminated record, which is never parsed). A changed compressed file is a rewrite like any other: identical decompressed content (for example recompressed at another level) is no change, a changed earlier record is reindexed (`source_changed`), and more records after an identical prefix are a continuation. Growth of the compressed file itself is never treated as pure append.
- **Streaming only.** The decoder reads in bounded blocks; a cursor in the middle of a compressed source is reached by decoding and discarding from the first byte, so memory does not depend on the session size. On a 54.5 MB synthetic session (28 MB compressed) a full sync peaks at the same traced memory as the identical plain file, 16 MB, and an audit-style refresh at 3.5 MB; a 60 MB stream that compresses to 6 KB is decoded with a 2.4 MB peak. Source observation validates a compressed snapshot by the SHA-256 of the compressed bytes (no second decode); each audit or status check of an indexed compressed session decodes the file once.
- **Strict frames and visible errors.** A compressed source must be exactly one zstd frame that declares its content size, without a dictionary and with a window of at most 128 MiB (zstd's own default). Anything else, and every unreadable file, is an explicit error with a code, never a skipped file or a short read: `zstd_corrupt`, `zstd_truncated`, `zstd_size_mismatch` (more or fewer bytes than the header declares), `zstd_unsupported_frame`, `zstd_too_large` and `zstd_unavailable`. `sync_session` returns `ok: false` with that `error`, nothing is indexed or changed, and an audit is `unknown` with the code in `coverage.issues`. A source that was indexed and later goes bad reports the code in coverage instead of `source_unavailable_or_truncated`.
- **Size limit.** `SELF_DIRECT_MAX_DECOMPRESSED_SESSION_BYTES` (default 4 GiB, about ten times the largest observed session) bounds the size the frame header declares; a larger file is refused as `zstd_too_large` before any byte is decoded, and a stream that disagrees with its header stops at the declared size. A plain `.jsonl` has no such limit.
- **Decoder.** Python 3.14+ uses the standard library `compression.zstd`; older interpreters use the `zstandard` package (`zstandard>=0.23`, installed with the project). Without either, a `.zst` source reports `zstd_unavailable` and plain `.jsonl` sessions keep working. The Codex parser stays at version 3: the parsed records and event ids are unchanged, so existing indexes are not reparsed.

---

## MCP tools reference

| Category | Tool | Description |
|---|---|---|
| **Governance & Contracts** | `upsert_contracts` | Create or update explicit invariant rules (`must` / `must_not`). |
| | `list_contracts` | Inspect active contracts for the current session and global scope. |
| | `revoke_contract` | Disable a contract while preserving history. |
| | `check_action` | Pre-flight test of a proposed action against current contracts. `verify="fast"` selects the hook-time source verification; the result's `integrity` says what was compared. |
| | `audit_session` | Audit executed session history against active contracts (always a full source verification). |
| | `audit_status` | Health check: vector extension, index schema, degraded flags, per-session coverage. |
| **Checklist & Evidence** | `update_checklist` | Update requirements; enforces evidence gating for completion. |
| | `get_checklist` | Inspect checklist status, history and linked evidence. |
| **GraphRAG & Architecture** | `get_graph_context` | Query entity dependencies and the active knowledge graph. |
| | `propose_graph_update` | Validate node/edge additions with evidence chunk requirements. |
| | `commit_graph_update` | Atomically commit validated graph updates with cursor progression. |
| | `run_neograph_update` | Run the LLM GraphRAG extraction pipeline through NeoGraph. |
| **Search & Analytics** | `sync_session` | Incremental evidence sync for Codex, Grok Bot, OMP, AGY and Claude Code transcripts. |
| | `search_history` | Hybrid retrieval (dense vector + BM25 + regex) over history. |
| | `get_chunk` | Retrieve a raw evidence chunk with secret masking. |
| | `analyze_activity` | SQLite time-series analysis: retry gaps, error bursts, stalled items. |
| **Hooks** | `codex_session_hook` | Adapter for Codex lifecycle events and local audit context. |
| **Approved Workflows** | `workflow_status` | Current state, requirements, versions and observed verification records. |
| | `workflow_run_checks` | Run owner-approved test commands and kernel-check approved Lean statements. |
| | `workflow_classify` | Request a versioned JEV Choice using approved synthetic context. |
| | `workflow_transition` | Evaluate the current classification through the compiled Lean transition function. |
| | `workflow_authorize` | Issue a single-use, action/version/state-bound execution grant. |
| | `workflow_execute` | Recheck and consume the grant, run the exact argv, then record success or failure. |

---

## Quickstart

### Prerequisites
- Python 3.12+ and `uv` (recommended) or `venv`
- The official `sqlite-vector` 1.1.0 native library (the setup script below downloads and verifies it)
- An OpenRouter key for the default embedding and retrieval path; not needed in explicit local-only mode

### 1. Clone and install

```bash
git clone https://github.com/fox1245/tsukkomi-mcp.git ~/.mcp-servers/tsukkomi-mcp
cd ~/.mcp-servers/tsukkomi-mcp

uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Provision sqlite-vector

```bash
python scripts/setup_sqlite_vector.py --output-dir /absolute/path/to/native
```

The script pins release 1.1.0, verifies the official archive's SHA-256 and checks native cosine search. Without `--output-dir` it writes to `.native/` in the current directory. Set `SELF_DIRECT_SQLITE_VECTOR_PATH` (or `sqlite_vector_path` in TOML) to the resulting `vector.so`, `vector.dll` or `vector.dylib`. With no setting, the runtime only looks for `native/vector.*` inside the source checkout. If the library cannot be loaded, engine initialization fails with an error; it does not fall back to NumPy. NumPy (`SELF_DIRECT_DENSE_BACKEND=numpy`) and `SELF_DIRECT_USE_FAKE_EMBEDDER=true` are explicit offline and test options.

### 3. Configure

Settings come from, in order of precedence: explicit `Settings` arguments, environment variables, the TOML path file, a `.env` in the **current working directory**, then defaults. A server started by an MCP host usually has an unrelated working directory, so do not depend on a `.env` there; use explicit `SELF_DIRECT_*` variables in the host's server entry or the TOML file below. Lifecycle hooks (AGY and Claude Code) never read a working-directory `.env`: they use the environment and the TOML only, because that file is content the audited agent can write.

#### User path configuration (TOML)

Use `~/.config/tsukkomi-mcp/config.toml` on any supported platform, or set `SELF_DIRECT_CONFIG_FILE` in the process environment to select another file. Copy and edit [config.example.toml](config.example.toml); the edited file belongs outside the repository and agent-writable workspaces.

```toml
[paths]
index_dir = "~/private-tsukkomi/index"
omp_sessions_dir = "~/.omp/agent/sessions"
openrouter_api_key_file = "../credentials/openrouter.env"
sqlite_vector_path = "native/vector.so"
```

The nine supported keys are `index_dir`, `codex_sessions_dir`, `omp_sessions_dir`, `claude_projects_dir`, `grokbot_transcripts_dir`, `agy_app_data_dirs`, `contracts_path`, `sqlite_vector_path` and `openrouter_api_key_file`. AGY roots use a string array; Grokbot accepts its existing delimiter-separated string or an array of individual paths. Empty arrays disable those optional roots.

TOML paths expand `~`; relative paths resolve against the canonical TOML file's directory, not the working directory or a symlink's directory. `${ENV_VAR}` interpolation is not performed. The selector `SELF_DIRECT_CONFIG_FILE` is environment-only, so a `.env` cannot silently select another TOML file. An absent default file keeps the defaults. An explicitly selected missing file, malformed TOML, an unknown section or key, or an invalid path value fails instead of silently selecting another index. Settings capture the selected sources, so restart affected servers and extensions after changing them.

#### Environment settings

All settings use the `SELF_DIRECT_` prefix except the key itself. The names below are the runtime fields in `src/self_directing_mcp/config.py`; [.env.example](.env.example) lists the commonly used ones.

| Setting | Default | Meaning |
|---|---|---|
| `SELF_DIRECT_INDEX_DIR` | `~/.self_direct_index` | Private index directory (events, FTS, vectors, contracts, checklists). |
| `SELF_DIRECT_CONTRACTS_PATH` | `<index_dir>/contracts.json` | Contracts file. |
| `SELF_DIRECT_SESSION_PROVIDER` | `codex` | Provider used when a tool call omits `provider`: `codex`, `grokbot`, `omp`, `agy` or `claude`. |
| `SELF_DIRECT_CODEX_SESSIONS_DIR`, `CODEX_SESSIONS_DIR` | `~/.codex/sessions` | Codex session root. |
| `SELF_DIRECT_OMP_SESSIONS_DIR` | `~/.omp/sessions` | OMP session root. |
| `SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR` | empty (disabled) | Cursor / Grok Bot transcript roots. |
| `SELF_DIRECT_AGY_APP_DATA_DIRS` | `~/.gemini/antigravity`, `-cli`, `-ide` | JSON array of AGY app-data directories. |
| `SELF_DIRECT_CLAUDE_PROJECTS_DIR` | `$CLAUDE_CONFIG_DIR/projects` if `CLAUDE_CONFIG_DIR` is set, else `~/.claude/projects` | Claude Code projects root (TOML `claude_projects_dir`). The environment variable wins over TOML, and TOML over the default. |
| `SELF_DIRECT_AGY_ENFORCEMENT` | `confirmed_only` | AGY hook policy: `confirmed_only`, `advisory` (never blocks) or `enforced` (also denies an unverifiable `PreToolUse`). Hook errors fail open in every mode. |
| `SELF_DIRECT_CLAUDE_ENFORCEMENT` | `confirmed_only` | Claude Code hook policy, same values, independent of the AGY setting. `scripts/install_claude_hooks.py` writes it into the Claude Code settings `env`. |
| `SELF_DIRECT_USE_FAKE_EMBEDDER` | `false` | Offline/test embedder; never a silent fallback. |
| `SELF_DIRECT_LOCAL_ONLY` | `false` | Explicit offline mode: no embedder or dense index; embedding, dense/hybrid search and LLM graph updates are rejected. |
| `SELF_DIRECT_DENSE_BACKEND` | `sqlite-vector` | `sqlite-vector` or `numpy`. |
| `SELF_DIRECT_SQLITE_VECTOR_PATH` | `native/vector.*` in the checkout, if present | Native library path. |
| `SELF_DIRECT_OPENROUTER_API_KEY_FILE` | unset | Authoritative dotenv file with `OPENROUTER_API_KEY`. |
| `SELF_DIRECT_EMBEDDING_MODEL`, `SELF_DIRECT_EMBEDDING_DIM`, `SELF_DIRECT_EMBEDDING_BASE_URL` | `qwen/qwen3-embedding-8b`, `1024`, `https://openrouter.ai/api/v1` | Embedding model, dimension and endpoint. |
| `SELF_DIRECT_LOCK_WAIT_TIMEOUT_SEC` | `1` | Combined process/file writer-lease wait. |
| `SELF_DIRECT_AUDIT_TIMEOUT_SEC` | `8` | Response budget for audits and hooks. |
| `SELF_DIRECT_REQUEST_TIMEOUT_SEC` | `45` | Response budget for ordinary calls. |
| `SELF_DIRECT_GRAPH_UPDATE_TIMEOUT_SEC` | `300` | Budget for `run_neograph_update`. |
| `SELF_DIRECT_MAX_TOOL_TIMEOUT_SEC` | `600` | Upper bound for a client-supplied `timeout_ms`. |
| `SELF_DIRECT_MAX_CONCURRENT_CALLS` | `4` | **New in 0.6.0.** Heavy tool calls that may run at once (minimum 1); excess calls wait within their own deadline. |
| `SELF_DIRECT_INTEGRITY_FAST_RECENT_BLOCKS`, `SELF_DIRECT_INTEGRITY_FAST_SAMPLE_BLOCKS` | `4`, `8` | Fast (hook-time) source verification: the latest complete 1 MiB blocks and the random sample compared with their stored digests. `0` for either disables that part (the tail block is always compared). See [Source integrity](#source-integrity-full-and-fast-verification). |
| `SELF_DIRECT_INTEGRITY_FULL_VERIFY_INTERVAL_SEC` | `300` | A fast check is allowed only this long after the last verification of every block; then the next check is full. `0`: always full. |
| `SELF_DIRECT_INTEGRITY_BACKGROUND_VERIFY`, `SELF_DIRECT_INTEGRITY_BACKGROUND_IDLE_SEC`, `SELF_DIRECT_INTEGRITY_BACKGROUND_MAX_SCANS` | `true`, `2`, `1` | Resident servers verify watched sources in full on one background thread after this much idleness; at most this many scans per wake-up. Hook processes never do background work. |
| `SELF_DIRECT_MAX_DECOMPRESSED_SESSION_BYTES` | `4294967296` | **New in 0.6.0.** Largest decompressed size a `.jsonl.zst` Codex session may declare; larger files are refused as `zstd_too_large`. |
| `SELF_DIRECT_RETRIEVE_TOP_K`, `SELF_DIRECT_SEARCH_TOP_K`, `SELF_DIRECT_RRF_K` | `20`, `10`, `60` | Retrieval tuning. |
| `SELF_DIRECT_SEED_EXAMPLE_CONTRACTS` | `false` | Seed disabled demonstration contracts. |

#### Credentials

Credential values are not TOML options. A key file selected in TOML or by `SELF_DIRECT_OPENROUTER_API_KEY_FILE` is authoritative: a missing, unreadable or keyless file is an error, never a fallback to another key. Without a selected key file, `OPENROUTER_API_KEY` follows the precedence above (explicit argument, environment, working-directory `.env`). Keep the key file outside the repository and agent-writable workspaces. Workflow sandboxes reject mounts that expose the selected TOML, a host dotenv or the selected credential file.

---

## Platform configuration

### Google Antigravity (AGY)

Configure `tsukkomi-mcp` in your `mcp_config.json`:

```json
{
  "mcpServers": {
    "tsukkomi-mcp": {
      "command": "~/.mcp-servers/tsukkomi-mcp/.venv/bin/python",
      "args": ["-m", "self_directing_mcp"]
    }
  }
}
```

To enable lifecycle interception in AGY, register `scripts/agy_hook.py` as your workspace hook command. By default only confirmed findings gate a tool call. See [docs/agy-installation.md](docs/agy-installation.md) for the hook file, enforcement modes and host boundaries.

### OpenAI Codex CLI

Install advisory lifecycle hooks and register the MCP server in one command:

```bash
python scripts/install_codex.py --trust-hooks
python scripts/install_codex.py --verify
```

The installer registers explicit `SELF_DIRECT_INDEX_DIR`, session-root, backend and key-file settings in the Codex MCP entry, so it is not affected by the default index location. See [docs/codex-installation.md](docs/codex-installation.md).

### Claude Code

Register the MCP server and, if the user wants lifecycle controls, review and then install the opt-in hooks:

```bash
claude mcp add --scope user tsukkomi-mcp -- /absolute/path/to/venv/bin/python -m self_directing_mcp
python scripts/install_claude_hooks.py --print --python /absolute/path/to/venv/bin/python
python scripts/install_claude_hooks.py --apply --settings /absolute/path/to/settings.json --python /absolute/path/to/venv/bin/python
```

`--print` is the default and changes nothing; `--apply` merges into the settings file you name, keeps unrelated hooks, backs the file up when it changes and replaces it atomically. Add `--index-dir /absolute/path/to/index` to point the hooks at the index whose contracts should apply (the one the MCP server writes to); hooks never read a project `.env`. See [docs/claude-installation.md](docs/claude-installation.md) for the policy table, the Stop budget, subagents, fail-open behavior and verification steps, and [docs/claude-hooks.example.json](docs/claude-hooks.example.json) for the complete hooks block.

### OMP (OpenRouter-backed retrieval, deterministic contract hooks)

Register this server in OMP's **user** `mcp.json` and opt in to `extensions/tsukkomi-omp.ts` as an OMP user extension. Configure both processes with the same private TOML file (or environment overrides) for the index, OMP session root and authorized credential/vector references. The extension starts Python in `--omp-bridge` mode, so the shared Python settings source owns validation; TOML-only paths need not be duplicated in the extension environment.

```text
SELF_DIRECT_LOCAL_ONLY=false
SELF_DIRECT_CONFIG_FILE=<private TOML path>  # omit to use the default user file
TSUKKOMI_OMP_MONITOR=1
TSUKKOMI_OMP_PYTHON=<Python with tsukkomi-mcp dependencies>
TSUKKOMI_OMP_SOURCE=<source checkout>/src  # omit when installed in that Python
```

Interpreter/source bootstrap and `SELF_DIRECT_LOCAL_ONLY` remain environment options, not `[paths]` keys. Remote bridge startup requires a usable explicitly selected key file and an existing native-library path; it does not make an API call or eagerly initialize the index. Environment paths already present in managed registrations override TOML until the owner removes or changes those entries.

OpenRouter-backed hybrid/dense history search, `sync_session(embed=True)` and NeoGraph updates are available by default; embedding uploads sanitized session chunks, so use an authorized key and session root. The hook's `check_action` and `audit_session` stay deterministic and refresh JSONL without embedding; they do **not** call OpenRouter on every tool or turn. Explicit offline-only mode is `SELF_DIRECT_LOCAL_ONLY=true`, which disables those remote functions. NeoGraph extraction also masks known secret patterns in session events before sending a prompt to OpenRouter; keep the original session files and local index private.

Keep machine-specific values in local OMP configuration or a private extension copy, not this repository. Start a **new persisted OMP session**; `--no-session` cannot provide history. Register only user-stated, machine-checkable contracts with `upsert_contracts` (`provider: "omp"` and the actual OMP session ID for session scope). The extension checks pending model-issued tools through local MCP before execution, blocks confirmed `violation`/`suspicious`, and re-audits after results and turns. `unknown` and transport failure are reported but do **not** block; zero applicable contracts is not `clean`. Direct shell or host actions outside OMP's tool pipeline, skipped hooks, in-memory sessions and a result not yet flushed to disk are not covered.

The extension also registers the read-only `tsukkomi_session_context` tool. Call it before creating a session-scoped contract or checklist item: its `session_id` comes from the active OMP session manager. Never infer a session ID from `PI_SESSION_FILE` or a shell variable, which can belong to a parent OMP process. If no persisted session is available, do not silently create a global contract. `AGENTS.md` instructions alone are not audit contracts: register the user's explicit, machine-checkable prohibitions; broader subjective mistakes are not automatically judged by an LLM.

### Cursor / Grok Bot

Set `SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR` (or `grokbot_transcripts_dir` in TOML) and pass `provider="grokbot"` to the MCP tools. There is no lifecycle hook for this host in this repository.

---

## Coverage and verdicts

`sync_session`, `audit_session` and `check_action` report a `coverage` object (`audit_status` reports the same data under `session` when given a session id). It holds the indexed source `path`, `byte_offset` (how far the index has parsed), `file_size`, `chunk_count`, `parser_version`, an `issues` list and `complete`, which is true only when `issues` is empty. The same responses carry an `integrity` object that says how the source was verified for that request and when it was last verified in full ([Source integrity](#source-integrity-full-and-fast-verification)).

| Verdict | Meaning |
|---|---|
| `violation` | A contract is broken by evidence in the judged history or by the proposed action. |
| `suspicious` | An exception that needs confirmation. |
| `unknown` | The evidence needed to decide is missing, incomplete or changed. Resolve the gap; it is not permission. |
| `clean` | No problem was found **in the returned evidence and contract snapshot**. It is not a new authorization. |

Audit and `check_action` results also carry `requires_attention`, which is `true` whenever the verdict is not `clean`.

### Growing sessions

Agents append to their own session file while they work, so an audit usually sees a file that is still growing. The audit judges the bytes captured at the start and says so.

| Field (in `coverage`) | Meaning |
|---|---|
| `judged_through_byte` | Offset up to which the verdict was computed. |
| `unread_bytes` | Bytes after that offset (appended records, or an unfinished final line) that were not reviewed. Also listed in `issues` as `unread_bytes`. |
| `judged_prefix` | `true` when appended, unread bytes are the only coverage gap. |
| `complete` | `false` while any bytes are unread. |

The result's `notes` then contain `Judged as of byte N; M appended byte(s) were not reviewed.` Within that prefix:

- A `must_not` violation seen in the captured range is reported as `violation`.
- Otherwise the verdict is `clean` **as of that offset**. Appended records are judged by the next audit.
- A `must` obligation that would be a violation only because evidence is absent, or failed, becomes `unknown` while bytes are unread. An obligation already evidenced inside the prefix stays `clean` as of the offset.
- Any other gap (parse errors, unsupported events, an outdated parser, a missing source) is not a pure append. The verdict follows the normal incomplete-coverage rules and `judged_prefix` is absent.

A rewrite of earlier bytes, truncation, file replacement or deletion is never judged as growth. It stays `unknown` (`source_changed`, `source_unavailable_or_truncated`, `source_changed_during_evaluation`). If metadata, contracts or AGY receipts change while an audit is evaluating, the result is `unknown` with the earlier result kept as `snapshot_verdict`.

### `check_action` and current prerequisites

A proposed action runs now, so unread records come before it, not after it. When a contract with a gated prerequisite (`must` with `before_regex`) applies to the proposed action and the history is incomplete:

- With `requires_success: true`, a `clean` finding becomes `unknown` and the overall verdict is raised to at least `unknown`, because a newer failed or pending run may be in the unread part ([#8](https://github.com/fox1245/tsukkomi-mcp/issues/8)).
- A presence-only prerequisite (`requires_success: false`) stays `clean` while the only gap is appended bytes, because an observed earlier run is not undone by later records. For any other coverage gap it becomes `unknown`.
- A `must_not` violation already found is not hidden by this rule.

### Duplicate session sources

Several files can carry the same session id, for example when sub-agent session files are copied under another parent folder. Codex, OMP and Grok Bot compare the stored prefix hash with the requested file using streaming hashes, without loading either file. AGY is excluded because its hook receipts bind to an exact path, so a changed AGY path is still reindexed.

| Relation of the requested file to the indexed one | Result |
|---|---|
| Identical copy | Not reparsed. The response has `alias_of` (indexed path) and `coverage.indexed_path` / `coverage.requested_path`; the cursor stays on the indexed file. |
| Continuation (starts with the indexed prefix and is longer) | Only the part after the cursor is parsed and the cursor path moves to the longer file. |
| Strict prefix of the indexed file | The cursor is kept. Coverage reports the indexed source, adds the issue `requested_source_is_prefix_of_indexed` and sets `complete=false`, because the index holds events the requested file lacks. |
| Diverged (neither file is a prefix of the other) | The index is preserved. The call fails with `error: "ambiguous_session_source"` (with `indexed_path` and `requested_path`) and an audit is `unknown`, with `ambiguous_session_source` in `coverage.issues`. |
| Indexed file gone, requested file continues it | The new path is adopted without reparsing. If it does not continue the indexed history, the session is reindexed (`source_changed`). |

A rewrite or truncation of the same path is still reparsed or rejected as before. A plain rollout and its zstd-compressed twin (`.jsonl` and `.jsonl.zst`) are compared by their decompressed bytes, so every row above applies to that pair too; see [Compressed Codex sessions](#compressed-codex-sessions).

### Incremental audits

An audit used to re-read and re-evaluate every stored event of the session, so its cost grew with the session even when
nothing had been appended. Audits now keep a small **monitor state** per contract and fold in only the events that
arrived after it. The answer is the same: verdicts, findings, evidence, notes, `coverage` (including
`judged_through_byte`, `unread_bytes` and every unknown reason) and the NeoGraph stage trace are identical to the full
evaluation, which remains the reference. The audit still runs through the native `contract_audit` graph; only what its
`load_evidence` and `evaluate_contracts` stages do changes.

- **What is stored.** `meta.sqlite` gains derived tables (`audit_monitor_state`, `session_commit_log`,
  `session_generations`, `audit_incremental_health`) and, once a `requires_success` contract or a Claude Code result
  contract is audited, an index on tool-call ids. Nothing in them is authority: the events and the cursor are. A missing,
  unreadable or disagreeing row only means the contract is evaluated in full again.
- **Per contract kind** the state holds: anchors and the activation point (`applies_from_event_id`); whether a
  `when` condition was seen; for `must_not` the match count, the first 20 matches and (with an `exception_regex`, which
  is matched against the join of all matches) the match texts; for `must` the last three regex hits and, for
  `requires_success`, a per-call tracker (how many calls share the latest hit's id, how many results followed and whether
  the first succeeded); for `before_regex` the worst verdict, the distinct reasons, the first 20 evidence items and the
  last trigger position, because each triggering action is judged at its own position from the prefix alone. Provider
  evidence classes keep Claude Code's cut results and AGY's truncated fields, still-`RUNNING` results, legacy results,
  host-logged proposals and denied proposals.
- **A normal append does not invalidate state.** A state is reused only if all of this holds, otherwise the contract is
  rebuilt by a full pass: same contract (revision **and** content), parser version, evaluator version and `needs` (which
  events had their text loaded); same index generation and cursor path; the stored `(byte_offset, prefix_hash,
  event_count)` is a row of the session's commit log, and the log's newest row is the current cursor; and the number of
  events after the stored offset is what the cursor's count implies. A reindex, a rewrite (`source_changed`), a parser
  bump, a moved or copied source (`alias`/path change), an append that overwrites a stored event, or a cursor written by an
  older server process each start a new generation or fail one of those checks. State and the evaluated cursor are written
  in one transaction, after the index commit they depend on, and never over a newer state.
- **Limits are explicit.** Tables are bounded (hit texts for an `exception_regex`, pending results, blocked proposals, cut
  results, notice targets). A run that would exceed one reports it, evaluates that contract with the full evaluator
  and remembers the limit, so the next audit goes straight there. Where state cannot be summarized faithfully the contract
  stays on full evaluation: a `must_not` whose `exception_regex` has more match text than the cap; an AGY denial whose
  proposal lies beyond the 4,096-event look-back window for a contract that has its own matching proposals; any contract
  disabled by a shadow mismatch (below). Adding a contract that needs text of events no other contract reads changes the
  `needs` and rebuilds every state once. An AGY conversation rebuilds when a hook receipt changes an event that is already
  indexed (see [AGY hook cost](docs/agy-installation.md#hook-cost-what-a-hook-reads)); an ordinary receipt is an append.
- **Shadow verification.** About every `SELF_DIRECT_AUDIT_SHADOW_EVERY`th audit (default 16; `1` compares every audit,
  `0` only after state was rebuilt) also runs the full evaluator on the same events and compares findings, verdict, notes,
  coverage and trace. On a difference the **full result is returned**, `incremental.incremental_mismatch` names the contracts, a row with status
  `incremental_mismatch` is written to `audit_incremental_health` and that contract is evaluated in full in that session
  until the evaluator version changes. The comparison is skipped (`incremental.shadow_skipped: "deadline"`) when the request's
  remaining time budget could not cover the full evaluation. The test suite compares every audit and fails on any
  difference.
- **Small sessions.** A session with fewer than `SELF_DIRECT_AUDIT_INCREMENTAL_MIN_EVENTS` stored events (default 500)
  is evaluated in full and no state is kept (`incremental.mode: "full"`, `below_min_events`). Below that size one full
  evaluation costs less than storing states, rebuilding them and shadow-checking the rebuild; on the scripted evaluation
  harness (short sessions) keeping state roughly doubled the Codex `Stop` latency. `0` keeps state for every session.
- **Reporting.** Results carry an `incremental` object: `mode` (`delta` reads only appended events, `full` reads all),
  `events_read`, `rebuilt` (contract to reason), `fallback` (contracts on full evaluation and why), `shadow`,
  `shadow_compared`, `incremental_mismatch`, `retry` and `state`. `SELF_DIRECT_AUDIT_INCREMENTAL=false` switches the whole feature off.
- **With source verification.** A state describes the stored events, not how the source was verified: it is written
  before the fast-verification gate and is the same whether the request verified `fast` or `full`. If a later full
  verification finds a rewrite, the reindex starts a new generation and deletes the states. A cursor that migrates from a
  plain digest to block digests, or that an older process rewrote, no longer matches the commit log, so states keyed to
  the old digest are rebuilt and never reused.
- **With contract time limits.** Contract patterns run on the time-limited engine. A monitor whose pattern times out
  takes no more events and is dropped for that run; its contract is judged (`unknown`, never `clean`) by the full
  evaluator, its stored state is deleted and nothing is marked unhealthy. When a shadow run's full evaluation itself
  times out for a contract, the full (`unknown`) answer is returned, the contract's state is dropped, and it is reported as
  `incremental.shadow_skipped: "reference_regex_timeout"`, not as a mismatch.
- **What it does not do.** It does not make an unknown clean and does not change coverage: unread bytes, parse errors,
  unsupported events and rewrites are judged by the unchanged ingestion and coverage code, and a state describes the index,
  not the source file. The source verification (hashing) that precedes every audit is a separate cost; see
  `scripts/bench_audit.py` for a before/after measurement on a synthetic session of any size.

### Source capture

Sync captures at most the length measured at the first `fstat`, in 1 MiB blocks. It keeps only small hash state: the SHA-256 of every complete 1 MiB block (the [block digests](#source-integrity-full-and-fast-verification)), and for a source whose cursor is a plain digest (`.jsonl.zst`, AGY, or an index written before this version) SHA-256 states at the cursor, at the last complete record and at the end. No file bytes are retained, so memory does not scale with the session size.

- **One full scan of the prefix per request.** The capture pass is the scan. Every later stage of the same request (the two ingestion stages, coverage, the audit's own stage check) re-reads a bounded amount instead of hashing the prefix again: the file identity, `size >= captured length` and the SHA-256 of the last 64 KiB as it was streamed during the pass. Before this version one `check_action` hashed the whole prefix five times (measured: 5.0 passes on a 300 MB session; now 1.0). `validate_prefix()` is that bounded re-check; `validate_prefix(locked=True)` is the same without any prefix re-read, for use under the shared publication lock; `validate_prefix(full=True)` re-reads the whole prefix and is used only when asked for or when the cheap signals demand it (below). A mismatch, once detected, stays failed.
- **Cheap signals escalate, they never excuse.** Without growth, an unchanged size with a moved modification time means the file was written (or touched). The unlocked check then settles it with a full re-read: identical bytes pass and the new time becomes the baseline; different bytes fail. Equal stamps prove nothing; they are never a reason to skip a check that would otherwise be made.
- **A change during preparation is retried, then reported as before.** If a later stage finds that the file was rewritten, truncated or replaced between capture and publication, the observation is discarded and the file is captured again, in full, at most three times in all. If it still does not hold still, the call fails with `source changed during preparation` and nothing is published, as before.
- Growth after the first measurement is not an error. An overwrite, truncation or replacement during the pass is. If the file grew while it was being hashed, the bounded check compares the last 64 KiB with the bytes that were streamed; this no longer costs a second scan.
- `validate()` compares metadata only (device, inode, size, mtime, ctime). Equal stamps do not prove equal content.
- `stat` and `fstat` can give `st_ctime` different meaning on some runtimes (for example 9p/drvfs), so full stamps are compared only within the same API and identity, size and mtime across them. On Windows `st_ctime` is the creation time and is never used as evidence of change. When either side reports `st_ino` 0, which some Windows filesystems do, the inode comparison is skipped and size, mtime and the hash decide.
- A `.jsonl.zst` source is an immutable snapshot: its capture is one streaming decode, and its later checks re-read and hash the compressed file (which is small), so any change to it is a rewrite.
- **Limits, stated precisely.** Capture and validation are not an atomic snapshot. A rewrite that leaves the identity, the size, the last 64 KiB and the modification time unchanged and lies outside the last 64 KiB, made after the capture pass of the same request, is accepted by that request's later stages. So is a rewrite outside the last 64 KiB made while the file also grew (the modification time then moves legitimately). The next full scan (the next sync's capture, or a full verification) compares the stored cursor hash and reindexes. This is the same residual window the lock-time check always had; it now starts at the end of the capture pass for every later stage.

### Source integrity: full and fast verification

Re-hashing a 300 MB session on every hook made the cost of a tool-call check proportional to the size of the session file. A cursor of a plain file therefore keeps, next to the byte offset, the **block digests** of the indexed prefix: the SHA-256 of every complete 1 MiB block, the digest of the partial last block, the newline count and a root over all of it (`blocks1:<root>` is the cursor's `prefix_hash`). They are built while the file is read anyway and extended, never recomputed, as the cursor advances: a sync hashes only the new complete blocks and the partial tail. The table lives in the index metadata (`source_blocks`, created by a schema migration on first open; existing data is untouched) and is bound to exactly one cursor publication. A table that is missing, stale (a cursor written by an older process), damaged (its root does not add up) or for another path is never trusted: the next request is a full verification that rebuilds it.

Every `sync_session`, `check_action` and `audit_session` response, and the context a Codex hook emits, carries an `integrity` object:

```json
"integrity": {
  "mode": "fast",
  "full_verified_through_byte": 314572800,
  "full_verified_at": "2026-10-15T12:00:00.123456+00:00",
  "source_generation": "g1:830-269c96:314572800:5b0e8e1b98536aaf",
  "index_generation": "i1:93d3c13bb79981cf",
  "fast_checked": {"tail": true, "recent": [309, 310, 311, 312], "sampled": [3, 8, 41, 77, 90, 134, 201, 250]}
}
```

| Field | Meaning |
|---|---|
| `mode` | How **this request** checked the source: `full` (every block of the prefix was read and compared) or `fast` (see below). |
| `full_verified_through_byte`, `full_verified_at` | The last verification of **every** block of the indexed prefix: through which byte, and when. They describe a **past event**. They are not a statement about the source as it is now, and a `fast` response must never be read as a guarantee about the whole current source. |
| `source_generation` | Names the content this request observed: file identity, covered length and a digest of what was read. It changes with the bytes, not with the clock. |
| `index_generation` | Names one publication of the session's index state; it changes with every cursor write. |
| `fast_checked` | (`fast` only) The blocks that were compared: the tail, the latest blocks and the random sample. |
| `full_reason` | (`full` only) Why the verification was full beyond the caller asking for it. `first_sync` (nothing indexed yet) and `legacy_source` (AGY, `.jsonl.zst`) can appear on any request; the others only on a request that asked for `fast`: `migration`, `parser_updated`, `path_changed`, `no_record`, `interval_elapsed`, `identity_changed`, `shrunk`, `same_size_modified`, `block_mismatch`, `tail_mismatch`, or `evidence_beyond_verified_region`. `interval_elapsed` also carries `verification_lag_sec`: how far past its deadline the verification was. |
| `reindex_reason` | (when the index was rebuilt) Why; `source_changed` means the stored digests no longer matched the source and the index was rebuilt from the source as it is now. |

**Which operations verify how.**

| Operation | Source verification |
|---|---|
| `audit_session` | Always `full`. There is no fast variant. |
| `check_action`, `sync_session` | `full` by default. `verify="fast"` is for hook-time callers: the Codex `PreToolUse` hook, the Claude Code and AGY hook adapters and the OMP extension pass it. It is never chosen implicitly. |
| Stop checkpoints, `auto_continue` decisions | `full`. A restart of the agent needs `integrity.mode == "full"` on a complete, current audit; a fast or missing `integrity` confirms nothing. |
| A `fast` request when the last full verification is older than `SELF_DIRECT_INTEGRITY_FULL_VERIFY_INTERVAL_SEC` (default 300) | Becomes `full` (`full_reason: "interval_elapsed"`). `0` makes every check full. |
| Resident servers (Codex, OMP, the stdio server) | One background thread verifies sources in full after `SELF_DIRECT_INTEGRITY_BACKGROUND_IDLE_SEC` (default 2) of idleness, and starts early enough to beat the interval during continuous activity. It holds no engine lock while reading, runs one scan at a time (at most `SELF_DIRECT_INTEGRITY_BACKGROUND_MAX_SCANS` per wake-up), never blocks a request and reports a deadline it could not meet (`audit_status` shows `integrity_worker`: scans, deferrals, errors, `missed_deadlines`, `max_lag_sec`). Short-lived hook processes (Claude Code, AGY) never do background work. |
| AGY sources, `.jsonl.zst` sources | Always `full` (`full_reason: "legacy_source"`): AGY hook receipts bind to plain prefix digests, and a compressed snapshot is validated by the digest of its compressed bytes. |

**What a fast check does.** Fast verification compares, against the stored digests: the file identity and size (it may only grow); the partial tail block; the latest `SELF_DIRECT_INTEGRITY_FAST_RECENT_BLOCKS` blocks (default 4); and `SELF_DIRECT_INTEGRITY_FAST_SAMPLE_BLOCKS` (default 8) random other blocks, drawn from fresh system randomness for every call. It then parses only the bytes appended after the cursor, each of them hashed block by block while consumed. No other block is read. Cost is a few MiB plus the appended bytes, whatever the size of the file.

Measured with `scripts/bench_integrity.py` on a synthetic 314.6 MB Codex session of 75,000 events (WSL2, ext4, Python 3.12; median of three runs; "passes" is the bytes through SHA-256 divided by the file size):

| Operation | Before: CPU (passes) | After: CPU (passes) |
|---|---|---|
| `check_action`, nothing appended, prohibition contracts, hook-time (`fast`) | 1.13 s (5.00) | 0.035 s (0.04) |
| the same after 5 events were appended | 1.27 s (5.00) | 0.11 s (0.04) |
| the same with a prerequisite gate (evaluates the indexed history) | 1.85 s (5.00) | 0.60 s (0.04); 0.02 s of it verifies the source, the rest evaluates 75,000 events |
| `audit_session` (full verification) | 1.69 s (5.00) | 1.02 s (1.00); 0.40 s of it verifies the source |
| first sync (authority only) | 22.2 s (6.03) | 21.4 s (3.03) |

The cost of the gate row is the evaluation of the indexed history (it grows with the number of events, not with the bytes of the file); making that incremental is separate work.

Cheap signals only ever escalate to full: a different file identity (a new inode), a smaller file, an unchanged size with a moved modification time, a block or tail digest that does not match. A mismatch is `source_changed` and a reindex, exactly as a mismatch of the whole-prefix hash always was; it is a content finding, never a runtime failure to fail open on. `reindex_reason: "source_changed"` is reported as before, and a stale index is never allowed to certify the new bytes: extending the table copies the digests of blocks that were not compared, so a rewrite the fast check missed is still found by the next full verification.

**What a fast verdict may rely on.** Contract evaluation is the same code and the same indexed history as in a full check; nothing is cached across calls. A fast verdict adds three rules: a `clean` finding that rests on a satisfied prerequisite is accepted only when every event it cites lies inside the region that was last verified in full (it then carries `verified_through`); otherwise the request is settled in full before it answers (`full_reason: "evidence_beyond_verified_region"`); and a fast verdict never turns `unknown` or a coverage gap (`unread_bytes`, parse errors, unsupported events) into a satisfied obligation. A fast answer is labelled `fast` and is never presented as `full`.

**Residual risks, stated precisely.**

- A rewrite of a block that no fast check compared (it is not in the tail, not among the latest blocks and was not drawn) is not noticed by fast checks until a full verification reads it: at the latest `SELF_DIRECT_INTEGRITY_FULL_VERIFY_INTERVAL_SEC` after the previous one, at the next `audit_session`, Stop checkpoint or background scan, whichever comes first. Until then a fast verdict is computed from the index as it was verified, and says `fast`.
- A source that is rewritten and restored between two verifications (every block compared is, again, what it was) is not detected, and no claim to the contrary is made.
- The file identity, size and modification time that the table remembers are advisory hints. They are not compared across machines as a guarantee (a Windows and a WSL process can see different `st_dev`/`st_ino` for one file); a difference only costs a full verification. File-system notifications are not used as evidence.
- The residual window of a request's later stages is the one described under [Source capture](#source-capture).

**Compatibility.** The cursor's `prefix_hash` is a plain SHA-256 until a source gets a table: the first full request after this version migrates it in place, without a reindex (`full_reason: "migration"`). A process of an older version that meets a block cursor does not understand it, treats it as a changed source and reindexes once; restart every process that shares an index after upgrading (see [Upgrading to 0.6.0](#upgrading-to-060)). A plain rollout and its `.jsonl.zst` twin still share one cursor: the cursor's hash form follows the source form.

---

## Memory, concurrency and deadlines

Long-running servers sharing one index had grown to many gigabytes in a real deployment ([#5](https://github.com/fox1245/tsukkomi-mcp/issues/5)). The causes were whole-file captures per request, concurrent reparses of the same session and heap that was freed but not returned. 0.6.0 changes:

- **No retained source bytes** (see Source capture).
- **Single-flight per (index, provider, session id).** Concurrent hooks for one session wait for one capture/parse/publish, then find the cursor current. Copies of one session id in several folders share the same flight. Across processes, a state-token compare-and-swap still decides which writer publishes, and a stale capture is retaken instead of rewinding a cursor that another process already moved forward.
- **Streaming reparse.** Parsed rows go through a disk-backed spool (`RowSpool`, memory below 1 MiB) and are published in batches inside one transaction.
- **Per-session id queries.** Sparse and dense catch-up query only the synced session's ids, in batches. Candidate vectors are held as float32 (the optional NumPy backend still stores float64).
- **Slim audit loading.** An audit pages in only the rows its contracts inspect, 512 at a time, and prerequisite search is linear rather than quadratic.
- **Bounded heavy calls.** At most `SELF_DIRECT_MAX_CONCURRENT_CALLS` (default 4) heavy calls run at once: sync, search, audit, `check_action`, checklist, graph, activity and hook calls. A request waits for a slot only within its own deadline; one that expires while queued returns `request_deadline` without starting work. A slot is held until the worker thread really finishes, even if its caller timed out. `list_contracts`, `upsert_contracts`, `revoke_contract`, `get_chunk`, `audit_status` and the `workflow_*` tools bypass the limit.
- **`malloc_trim` on glibc.** When the last running heavy call that took at least 0.5 s finishes (at most once per 5 s), the server calls `malloc_trim(0)`. It is skipped on Windows, macOS and musl, and failures are ignored.
- **`audit_status` no longer scans FTS.** The `sparse_count` reads the `chunk_rows` mirror table instead of counting the FTS5 table while holding the shared lock.

### Measurements

Measured by the maintainer; the real-session figures use private corpora that are not in this repository.

| Case | Before | After |
|---|---|---|
| 100 MB synthetic OMP session, `check_action` and `audit_session` repeated 3 times, 1 thread: peak RSS (VmHWM) | 705 MB | 134 MB |
| Same, 4 concurrent threads | 1,750 MB; first sync `index_busy` after 48.7 s | 198 MB; first sync clean in 12.1 s |
| Same, 8 concurrent threads | 2,304 MB; first sync `index_busy` after 48.0 s | 193 MB; first sync clean in 14.9 s |
| Idle RSS after `gc` (1, 4, 8 threads) | 505, 926, 941 MB | 104, 144, 151 MB |
| First sync of the 100 MB session, 1 thread | 10.4 s | 10.8 s |
| Three processes calling `audit_status` 5 times each on a 365k-chunk Codex index | 2.1 s for the sparse count under the lock | 15 of 15 succeeded, no `index_busy`, 0.045 s per call |
| 733 real OMP sessions synced three times in a row | 178 s per pass (repeated reparse of duplicate session ids) | 7 s per pass, 0 reparses, 27 duplicate pairs recognised as aliases |
| 571 real Codex sessions reparsed with parser v3 | v2: unsupported events in 401 sessions, parse errors in 121 | 0 unsupported, 0 parse errors; peak working set 225 MB in 876 s |

The issue's 19.9 GB observation came from one process after about 3.5 days with a growing ~310 MB session. The roughly 7 GB figure for that case in #5 is an extrapolation from the 100 MB synthetic run, and the PR did not measure a 310 MB live session. Memory under such a session has to be confirmed by observation after deployment.

### Deadlines and errors

The [shared-index fix (#3)](https://github.com/fox1245/tsukkomi-mcp/issues/3) separates stable source preparation, guarded publication and detached audit evaluation. Audit snapshots contain only scoped authoritative event rows and counts plus detached rule content, and deterministic checks do not repair FTS or restore embedding caches. Explicit sync still repairs derived indexes, and sparse/hybrid search materializes deferred canonical evidence before querying.

Audit evaluation and source hashing or parsing do not hold the shared writer lease. Before a result is published, metadata, source observations, rules and AGY receipts are revalidated; changed authority makes the result incomplete (`unknown`) while keeping the captured findings. Startup may add the AGY receipt-version side table without rewriting the six-column cursor schema or changing SQLite journal modes.

Requests share an absolute deadline and cancellation control through nested calls and native callbacks. Non-authorizing responses distinguish `index_busy`, `request_deadline`, `request_cancelled` and `operation_timeout`; unrelated SQLite failures are not converted into successful empty searches. A stopped request cannot begin a later publication, but an admitted commit or one already-running blocking call may finish. This is cooperative stopping, not forced thread termination or proof of rollback.

Default combined writer-lock wait (1 second), audit budget (8 seconds), ordinary budget (45 seconds) and explicit-tool cap (600 seconds) are unchanged. Real writes, initialization, raw captures and derived persistent mutations remain guarded and can still contend. See [implementation and verification boundaries](docs/timeout-fix.md).

---

## Upgrading to 0.6.0

1. **Restart every process that uses the index.** Stop and restart all MCP servers and extensions that share an index (Codex, OMP, AGY, any manual `python -m self_directing_mcp`) from the new package. A running process keeps the old code and old parser, and a process on an older version can write cursors that a newer one must then reparse. Updating a checkout or a wheel does not update processes that are already running.
2. **Parser bumps reparse lazily.** The OMP parser is now version 8 (0.5.0 had 2) and the Codex parser version 3 (was 2). The AGY parser is version 7 and the new Claude Code parser is version 2. Each session is reparsed once, on its next sync or audit, within the bounded-memory path. There is no upfront migration step and no storm of reparses.
3. **Existing indexes are preserved.** Events, anchors, model-specific vector caches, contracts and checklists are kept. Unchanged event ids and cached vectors survive a reparse. Obsolete entries are cleaned through a durable journal, so an interrupted upgrade resumes.
4. **Check the index location.** The default index is `~/.self_direct_index` unless `SELF_DIRECT_INDEX_DIR` or `index_dir` in TOML overrides it. Older versions defaulted to a `.self_direct_index` inside the checkout. An install that relied on that default must set `SELF_DIRECT_INDEX_DIR` (or TOML `index_dir`) to the existing absolute directory before restarting; nothing is moved or copied automatically. The Codex installer and managed OMP registrations already pass an explicit path.
5. **Check how the key file is selected.** A checkout `.env` is no longer treated as an authoritative key file. Select it explicitly with `SELF_DIRECT_OPENROUTER_API_KEY_FILE` or TOML `openrouter_api_key_file`, or supply `OPENROUTER_API_KEY` through the environment or a `.env` in the server's working directory.
6. **Reinstall dependencies.** `neograph-engine` must be exactly 0.14.1, and Python before 3.14 now needs `zstandard>=0.23` to read compressed Codex sessions (`.jsonl.zst`). `pip install -e ".[dev]"` or `scripts/install_codex.py` takes care of both; without the package, plain sessions work and compressed ones report `zstd_unavailable`.
7. **AGY hook policy.** The default is now `confirmed_only`. An install that relied on the earlier `enforced` default (an unverifiable `PreToolUse` denied, unverified items continuing a `Stop`, an unavailable runtime denying `PreToolUse`) must set `SELF_DIRECT_AGY_ENFORCEMENT=enforced` explicitly, and even then hook errors fail open.
8. **Sparse index masking.** Sparse (FTS) rows written before 0.6.0 hold raw text. Delete `sparse.sqlite` while the servers are stopped to have it rebuilt, masked, from the canonical store; see the [masking contract](SECURITY.md#masking-contract).
9. **Source integrity tables.** The first open of an index adds the `source_blocks` table (a schema migration that touches nothing else), and each session's cursor migrates to a block root on its first full sync after the upgrade, without a reindex. A process on an older version that meets a migrated cursor reindexes that session once, so restart every process that shares the index (step 1). See [Source integrity](#source-integrity-full-and-fast-verification).
10. **Re-verify.** Run `python scripts/install_codex.py --verify` for Codex and `audit_status` for any host. Confirm `neograph_version`, `embedder`, `dense_backend` and that `degraded_flags` is empty.

---

## Approved workflows: PRD, JEV, Lean, and NeoGraph

The optional workflow layer separates owner-approved original requirements from a derived Markdown wiki. It progresses through `requirements → formalize → implement → verify → complete`, with explicit revision paths. JEV requests a transition; it does not supply proof or test evidence. The runtime calls the **compiled Lean function**, not a separately maintained Python gate.

Requirements, code (including additions/deletions), specification, tests, policy, and class mapping have content identities. Changes invalidate classifications, verification records, and grants. Required obligations and obligations activated by actual code changes are independent of the classifier. Explicit `depends_on` and `conflicts_with` links are checked; contradictions in unrestricted natural-language prose still need owner review. Changing the PRD, expected tests, any Lean source/definitions/proof body, specification, policy, or approval manifest requires fresh owner approval. A live file hash cannot substitute for its captured approved identity.

### Owner setup

Install Lean **4.34.1 or newer**, including its matching `leanc` and `leanchecker`, and put `lean` on `PATH` (or set `TSUKKOMI_LEAN`). Workflow execution requires **Linux bubblewrap with unprivileged namespaces**; configure `TSUKKOMI_BWRAP` if it is not on `PATH`. Missing isolation fails closed, with no unsandboxed fallback. Compilation and policy audits are cached by source/toolchain identity, not repeated at every hook. The packaged policy proves one-step and arbitrary-sequence completion safety; its three theorems require no axioms. Domain proofs permit only `propext`, `Classical.choice`, and `Quot.sound`; transitive `sorryAx`, custom axioms, and mismatched theorem types are rejected.

Create the real PRD, implementation, test, specification, and proof files before approving a manifest. For example:

```json
{
  "schema_version": 1,
  "prd": "PRD.md",
  "scope": ["src/**/*.py"],
  "class_mapping_version": "1",
  "confidence_threshold": 0.7,
  "external_context": "synthetic",
  "protected_paths": ["release.py"],
  "requirements": [{
    "id": "REQ-RETRY",
    "text": "Retrying an existing request must not create another order.",
    "required": true,
    "status": "approved",
    "source": {
      "path": "PRD.md",
      "section": "Retry",
      "quote": "A repeated request ID creates one order."
    },
    "code": ["src/orders.py"],
    "specs": ["spec/order.md"],
    "tests": ["retry"],
    "proofs": ["retry"],
    "depends_on": [],
    "conflicts_with": [],
    "exceptions": []
  }],
  "tests": {
    "retry": {
      "argv": ["{python}", "-m", "pytest", "tests/test_retry.py", "-q"],
      "paths": ["tests/test_retry.py"],
      "timeout_sec": 60
    }
  },
  "proofs": {
    "retry": {
      "path": "proofs/Retry.lean",
      "theorem": "Order.retry",
      "statement": "∀ (s : Order.State) (r : Nat), Order.insert (Order.insert s r) r = Order.insert s r"
    }
  },
  "completion_actions": [
    {
      "tool_name": "workflow_command",
      "argument_pattern": "^\\{\"argv\":\\[\"\\{python\\}\",\"release\\.py\"\\],\"cwd\":\"\\.\"\\}$"
    },
    {"tool_name": "run_command", "argument_pattern": "release\\.py"}
  ]
}
```

The PRD origin uses an exact, unique Markdown heading and a verbatim quote under that heading. All referenced paths are project-relative; escaping paths and tracked symlinks are rejected. Test/proof IDs must resolve, and mandatory requirements need both kinds of verification.

From the directory containing `project/`, an owner—not the agent—runs:

```bash
tsukkomi-workflow approve --root project --manifest requirements.json --output trusted/approval
tsukkomi-workflow wiki --approval trusted/approval --output derived-wiki
```

Approval creation is exclusive and is deliberately **not** exposed as an MCP tool. A revised approval goes into a new directory. Configure the MCP server and AGY hook process with the same absolute, host-owned paths:

```text
TSUKKOMI_WORKFLOW_APPROVAL=<read-only approved snapshot outside the workspace>
TSUKKOMI_WORKFLOW_STATE=<private host-writable state outside the workspace>
TSUKKOMI_LEAN=<trusted lean executable>
TSUKKOMI_BWRAP=<trusted bubblewrap executable>
SELF_DIRECT_OPENROUTER_API_KEY_FILE=<authorized external dotenv containing OPENROUTER_API_KEY>
```

Keep approval, checker, hook configuration, Lean toolchain/cache, and evidence storage outside the agent's write authority. Checksums and file modes do not protect against an unrestricted process running as the same OS user. Use the host's sandbox/read-only mounts or a separate service account. Approval and workflow state must be separate directories.

### Verification and execution

Run `workflow_run_checks` to observe real subprocess exit codes and independently checked Lean proofs. Tests run against a fresh, readonly materialized snapshot of approved verification files and current tracked implementation—not the live project with untracked `conftest.py`, `sitecustomize.py`, or generated caches. Source-glob selection cannot silently authorize mutable harness/configuration files. The sandbox has a private writable temporary directory, no network or host credentials/state, and an allowlisted environment. Tests must write temporary artifacts there rather than modify the readonly snapshot. The runner does not accept model-supplied pass flags, replacement commands, or expected results. Source versions are checked before and after verification; interruption leaves nonpassing evidence.

Candidate Lean compilation is sandboxed. `leanchecker` replays the candidate against hash-pinned, readonly core imports; a source-pinned native auditor separately replays candidate declarations and checks the exact approved type and transitive axioms without loading candidate elaborator extensions. Unchanged trusted core libraries are not reproved for every request. Candidate stdout is not an audit report. Approved statements use trusted core syntax and fully qualified names, not candidate-defined notation or typeclass registration. Native `.olean` deserialization still assumes structurally valid artifacts: this is not the stronger comparator-plus-independent-external-checker guarantee described in [Lean's proof-validation guidance](https://lean-lang.org/doc/reference/latest/ValidatingProofs/).

`workflow_classify` calls [OpenRouter's documented System One route](https://openrouter.ai/docs/guides/community/typesafe-sdk), `https://openrouter.ai/api/v1/systemone`, with pinned `typesafe/jev-1.13` and an OpenRouter bearer key. It accepts only that exact response identity or its canonical release identity, `typesafe/jev-1.13-20260917`, and retains the returned identity. Each classification makes at most one request: no retry, redirect, replay, provider fallback, or model alias. The fixed mapping is `0=InsufficientEvidence`, `1=NeedsRevision`, `2=ReadyForVerification`, `3=ReadyForCompletion`; no Score rounding is used. The input hash binds the actual request, rubric, mapping, and model. Approved requirement origins, actual bounded code diffs, state, and verification summaries are the only context sent. Oversized or non-text changes are not silently truncated.

Remote classification is disabled unless the owner explicitly sets `external_context: "synthetic"`. This release's experiments are synthetic; do not label private production data synthetic to bypass consent. Classification reuses the server's shared OpenRouter settings: `SELF_DIRECT_OPENROUTER_API_KEY_FILE` selects an authoritative dotenv file containing `OPENROUTER_API_KEY`, read without interpolation. A missing, unreadable, or keyless configured file fails closed rather than falling back to an environment key. With no configured file, the shared `OPENROUTER_API_KEY` setting/environment is used. No separate TypeSafe key is needed or accepted, and other providers' keys are never substituted. Keep the credential file outside the workspace and runtime mounts; the child sandbox excludes it and strips host keys from its environment. A missing key, timeout, invalid distribution, unknown class/model, or low confidence is a nonauthorizing result. The example confidence threshold is an operator setting, **not** an empirically validated accuracy guarantee.

Use a fresh classification ID for each `workflow_transition`. A completion choice reports readiness while remaining in `verify`; it cannot skip the owned completion action. After reaching `verify` with current mandatory evidence, `workflow_authorize` accepts an action such as:

```json
{"tool_name":"workflow_command","arguments":{"argv":["{python}","release.py"],"cwd":"."}}
```

Pass the returned grant and the **identical action** to `workflow_execute`. It rechecks state, versions, classification, cancellation, and Lean policy immediately before consuming the grant. Only successful owned execution sets `complete`. Started, succeeded, failed, and uncertain outcomes are distinct; a crash after consumption cannot replay the grant. A timeout after launch never claims that no action ran. The local completion sandbox permits writes to the actual approved project, but not private host state/credentials or network access. No shell interpolation is performed, and cancellation/timeout terminates the sandbox process group and descendants.

The AGY and Claude Code hooks reject direct mapped completion actions and direct the agent to the owned executor; on a normal stop they continue an incomplete workflow once (the same once-per-conversation or once-per-session budget as `auto_continue`) and provide requirements, evidence, and permitted recovery actions. In `advisory` mode they neither reject nor continue. Cancellation, error stops, and pauses for background work are not restarted. Reads, repairs, and checks remain available unless a separate explicit contract restricts them. **Unmapped tools, direct shell access outside registered hooks, and the advisory Codex adapter are not an enforcement boundary.**

Lean proves the policy/model under its stated assumptions; it does not prove arbitrary Python/TypeScript implementations equivalent to that model. Real tests supply implementation evidence only for exercised cases. The workflow is separate from session GraphRAG extraction and does not require an LLM to decide whether recorded tests passed.

### Reproducible synthetic policy evaluation

```bash
TSUKKOMI_TEST_LEAN=lean python -m pytest -q
python scripts/benchmark_workflow.py --lean lean
```

The benchmark compares 640 finite stage/class/evidence combinations with an independent reference, measures false allowances/blocks and per-stage latency, and exercises correction of each missing mandatory fact. It is an adversarial **policy** evaluation, not live JEV accuracy or a whole-host coverage claim. Live JEV validation requires an OpenRouter key; offline transport fixtures never count as live service evidence. A real low-confidence Choice remains nonauthorizing: the proposed label and observed probabilities are retained, but completion stays locked.

---

## Testing and verification

The suite covers parsers, host receipt linkage, restricted reads, temporal obligations, append capture and prefix judgement, duplicate session sources, single-flight and concurrency, workflow version/grant invalidation, proof rejection, path isolation and MCP transport. Tests use synthetic sessions and offline embeddings. Lean integration tests, Windows-only path tests and bubblewrap tests report a skip when the toolchain or platform is unavailable, and symlink tests report a skip with a reason where Windows denies symlink creation (`WinError 1314`, Developer Mode off).

```bash
python -m pytest -q
python -m build
python scripts/benchmark.py
python scripts/bench_hooks.py --out bench-hooks.json --markdown bench-hooks.md   # hook cost, see below
```

### CI

GitHub Actions runs two workflows. Superseded runs for the same ref are cancelled, and no API keys or paid model calls are used.

The `CI` workflow runs three jobs on every push and pull request:

- **`test`** (Ubuntu and Windows, Python 3.12, 10 minutes): the explicit MCP transport, Jev boundary and Windows session-path smoke tests and the compressed Codex session tests (without their two 50 MB allocation probes), then builds the source distribution and wheel.
- **`memory-audit-regressions`** (Ubuntu, Python 3.12, 15 minutes): small offline regression files for, among others, audit memory, prefix judgement, current-action prerequisites, append and source capture, single-flight, index and server concurrency, deadlines, checklist conflicts, quiet hooks, AGY and Claude Code hooks, the Claude Code installer and provider, and hook gating. It also runs the hook import budget and settings parity tests ([Hook import budget](#hook-import-budget)) and the BLAS thread test. It runs with the fake embedder and the NumPy backend and deselects the one test that measures allocation of a large capture. These tests check ratios on small inputs; they do not certify live-service memory.
- **`bench`** (Ubuntu, Python 3.12, 10 minutes): `scripts/bench_hooks.py --check` on small synthetic sessions. It fails when a gated metric exceeds 1.5 times its target and uploads the JSON report and the markdown table as an artifact ([Hook cost benchmark](#hook-cost-benchmark)). The 300 MB session is never run in CI.

The `Memory` workflow (pull requests to `main`, `main`, nightly and on demand) is described in [Memory verification](#memory-verification).

Real Lean proof checks and the performance, large-session and concurrency benchmarks are local deep checks, not CI gates. A green CI run does not establish full workflow or native-proof coverage or performance guarantees. Live model validation needs a separate model and budget approval.

### Hook import budget

A Claude Code or AGY hook is a fresh Python process for every tool call, so every module it imports is paid for on every call. The hook path is kept deliberately small and a test enforces it.

`tests/test_hook_import_budget.py` runs **representative judgements**, not bare imports, in subprocesses: a `PreToolUse` that denies (a confirmed violation) and one that passes (a prerequisite satisfied from the transcript), `Stop` with its sync and audit, `SessionStart`, `PreInvocation` and the bookkeeping events, for Claude Code and AGY. Afterwards `sys.modules` must not contain a forbidden module, and the only third-party packages loaded must be the documented exceptions below. A new import that breaks this fails the test with the module's name.

Never loaded on the local-only hook path. The cost is the cumulative import time that the module added to a hook (`python -X importtime`, median of 13 fresh processes, AMD Ryzen 7 5800X under WSL2, Python 3.12; the tracing itself adds roughly 15 percent):

| Module | Why hooks do not need it | Cost when loaded |
|---|---|---|
| `numpy` | Dense vectors only, and local-only hooks never embed. `self_directing_mcp._runtime.np` is the single loading point and a lazy proxy: NumPy loads, with one BLAS thread and the environment restored, at the first attribute access. | 37 ms and about 15 MB (with the default BLAS threads about 1 CPU second more) |
| `pydantic_settings` (and `dotenv`) | The hooks read their settings with `settings_core.load_hook_settings()`: the same sources, precedence and rejections as `config.hook_settings()` (environment, `CODEX_SESSIONS_DIR`, the user TOML `[paths]`, defaults; never a working-directory `.env`), standard library only. The defaults, the TOML source and the accessors live in `settings_core` and are shared with `config.Settings`, so they cannot drift; `tests/test_hook_settings_parity.py` compares the two readers on about 850 environment and TOML combinations, including the ones both must reject. | 28 ms |
| `httpx` (with `idna` and `zstandard`) | The OpenRouter client. `embed.embedder`, `workflow_jev` and the graph runner import it, `dotenv` and `urllib.request` only where a request is made. | 28 ms (`urllib.request`: 4 ms) |
| `mcp` (with `anyio`, `starlette`) | The server transport; hooks are plain processes. | not loaded before this change either; guarded against regressions |
| `self_directing_mcp.config`, `.server` | Server-side settings and tool registration. | included above |

Claude Code's bookkeeping events (`UserPromptSubmit`, `PostToolUse`, `PostToolUseFailure`) load none of pydantic, the engine or the transcript parser: they validate the session id with `claude_ids` and deliver queued warnings.

Documented exceptions, loaded by every hook that judges contracts:

| Module | Reason | Measured cost |
|---|---|---|
| `pydantic` (with `pydantic_core`, `annotated_types`, `typing_extensions`) | `schemas.py` defines the contract, chunk and audit models, and the contracts file and every parsed chunk are validated through them. Replacing them is a rewrite, not an import fix. | about 75 ms cumulative: pydantic 16 ms, `importlib.metadata` 15 ms (pydantic's plugin loader) and the models themselves |
| `regex` | Contract patterns come from users and agents. They run on this engine because a search can be given a time limit (`audit/patterns.py`: 0.25 s per search, 2 s per contract, `regex_timeout` unknown beyond that), which the standard `re` module cannot interrupt. | about 10 ms |
| `neograph_engine` (with `certifi`) | Project rule ([AGENTS.md](AGENTS.md)): audit judgements run through the NeoGraph native graph, with no direct executor fallback. Kept as a documented exception; the roundtable proposed forbidding it and left the rule to the owner. | about 40 ms: the native extension 21 ms and `asyncio` 15 ms |

To re-measure, run a hook with `-X importtime` (`python -X importtime scripts/claude_hook.py --event=PreToolUse --local-only < payload.json 2> importtime.txt`; the columns are self and cumulative microseconds). Change the budget only by editing `FORBIDDEN` or `ALLOWED_THIRD_PARTY` in the test together with this section, with the module, the reason and the measured cost.

### Hook cost benchmark

`scripts/bench_hooks.py` measures what the hooks cost. It uses synthetic sessions, contracts and payloads under an isolated `HOME`, index directory and config file, and never reads a user's sessions, indexes or credentials. Cold, warm and server numbers are reported separately and never mixed.

```bash
python scripts/bench_hooks.py --out bench-hooks.json --markdown bench-hooks.md      # small sessions, about 8 minutes
python scripts/bench_hooks.py --check                                                # exit 1 above a gated CI threshold
python scripts/bench_hooks.py --large --large-mb 300 --out bench-large.json         # opt-in 300 MB session, never in CI
python scripts/bench_hooks.py --only cold --versus /path/to/older/checkout          # interleaved A/B of two checkouts
python scripts/bench_hooks.py --compare before.json after.json                      # before/after tables from two reports
python scripts/bench_hooks.py --python /path/to/venv/bin/python --hook-mode default # other interpreter / engine mode
```

What is measured:

- **Cold command hooks** (`cold`): a fresh Python process per event through the real `scripts/claude_hook.py` (`PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `SessionStart`, `Stop`) and `scripts/agy_hook.py` (`PreToolUse`, `PostToolUse`, `PreInvocation`, `Stop`) with synthetic payloads, plus a burst of N concurrent sessions (`burst`). The page cache is warm; "cold" means a new interpreter that imports the package and judges once. The transcript grows by whole records between runs, so each hook syncs a small increment. The first run of each event, which builds the index, is recorded separately (`first_run`). A `.venv` re-exec (one more interpreter start) is not included. `--hook-mode local-only` (default) runs the engine without an embedder or dense index, as `--local-only` does for Claude Code; `default` uses the offline fake embedder and the NumPy backend, so each hook also initializes the dense side.
- **Warm resident-server calls** (`warm`): one long-lived process and one asyncio loop through FastMCP's dispatcher: `codex_session_hook` (`PreToolUse`, `UserPromptSubmit`, `Stop`), `check_action` and `audit_session`, each with nothing appended and after appending bytes. The index build of the first call is reported on its own.
- **Idle servers** (`idle_servers`): N real `python -m self_directing_mcp` stdio servers with an isolated index, after the MCP handshake (`after_init`) and after one tool call that initializes the engine (`after_use`). Linux reads `/proc` (`VmRSS`, `VmHWM`, `Pss`); Windows reads the working set, peak working set and private bytes through `GetProcessMemoryInfo`. A sum of RSS counts shared libraries once per process; `pss_mb_total` divides them fairly.

Every row gives p50, p95 and maximum wall time, CPU time (user plus system, from the operating system: `os.wait4` for hook processes, `getrusage` or `GetProcessTimes` in a server) and peak RSS per process, together with the machine, OS, Python and package versions and the load average. A hook that fails open or exits non-zero invalidates the run (`hooks_failed_open`). On a shared machine use `--versus`: the two checkouts see the same transcript and payloads in alternating order, so load from other processes hits both alike.

Fixed conditions (flags, defaults in parentheses): contract counts (`--contracts 10,100`, each a rotating mix of `must_not`, `must`, `must` with `before_regex` and `must` with `before_regex` and `requires_success`), small session size (`--session-mb 1`), large session (`--large`, `--large-mb 300`, opt-in), bytes appended per step (`--append-bytes 4096`), concurrent sessions in the burst (`--sessions 4`), idle servers (`--servers 12`, `--server-profile local-only|fake-numpy`), repetitions (`--runs 20` after one discarded warm-up, `--warm-repeat 15`, `--large-repeat 5`).

Targets and CI thresholds live in one table, `TARGETS` in the script (also written into every report). A CI threshold is `--ci-factor` (default 1.5) times the target. Rows that depend on other work are reported against their target and marked "reported, not gated yet":

| Metric | Target | CI threshold | Gated here |
|---|---:|---:|---|
| Cold hook wall, p95 | 400 ms | 600 ms | yes, for Claude Code `PostToolUse` and `UserPromptSubmit`, which fire after every tool call and every prompt |
| Cold hook CPU (user+sys), p95 | 0.3 s | 0.45 s | yes, for the same events |
| Cold hook wall, p95 and CPU, every event | 400 ms, 0.3 s | 600 ms, 0.45 s | no: `PreToolUse` and `Stop` evaluate every contract over the whole session and AGY re-parses its transcript on every call; they wait for the fast integrity path and the incremental audit |
| Cold hook wall, median (goal) | 150 ms | 225 ms | no |
| `check_action`, large session, nothing appended, CPU | 100 ms | 150 ms | no: needs the fast integrity path |
| `audit_session` after at most 100 appended events, CPU | 20 ms | 30 ms | no: needs the incremental audit |
| Idle server RSS | 90 MB (70 MB desired) | 135 MB | yes |

Phase 2 (a resident daemon) becomes due if the summed RSS across hosts exceeds 500 MB, a turn's accumulated hook latency exceeds 2 s, or the phase 1 gates are missed; the report states when the idle-server total passes 500 MB. The thresholds are absolute: on a slower machine (a hosted CI runner, Windows) widen `--ci-factor` rather than the targets, and report Linux and Windows separately.

Measured on 2026-10-15, Linux (WSL2) on an AMD Ryzen 7 5800X with 16 logical CPUs, Python 3.12.14, 1 MB synthetic sessions, 10 contracts, 4 KB appended per step, 20 interleaved runs per event (`--versus`), the machine shared with other work (load average about 6). Wall and CPU in milliseconds, before (commit `00cb90e`) → after this change:

| host | event | wall p50 | wall p95 | CPU p50 | CPU p95 | peak RSS |
|---|---|---|---|---|---|---|
| claude | PreToolUse | 407 → 281 | 445 → 309 | 385 → 263 | 424 → 294 | 83 → 67 MB |
| claude | PostToolUse (gated) | 189 → 52 | 254 → 60 | 188 → 52 | 252 → 59 | 35 → 25 MB |
| claude | UserPromptSubmit (gated) | 185 → 50 | 205 → 62 | 185 → 50 | 205 → 62 | 35 → 25 MB |
| claude | SessionStart | 316 → 208 | 344 → 247 | 315 → 208 | 344 → 246 | 73 → 56 MB |
| claude | Stop | 598 → 410 | 766 → 474 | 551 → 365 | 718 → 406 | 90 → 74 MB |
| agy | PreToolUse | 742 → 646 | 854 → 754 | 654 → 551 | 765 → 623 | 86 → 70 MB |
| agy | PostToolUse | 780 → 664 | 898 → 748 | 646 → 538 | 743 → 606 | 82 → 65 MB |
| agy | PreInvocation | 420 → 321 | 531 → 349 | 372 → 268 | 476 → 295 | 78 → 62 MB |
| agy | Stop | 647 → 539 | 740 → 589 | 570 → 452 | 655 → 496 | 94 → 77 MB |

With 100 contracts the same events move by about the same amount (`PostToolUse` 190 → 53, `UserPromptSubmit` 193 → 54, `SessionStart` 328 → 212, `PreToolUse` 443 → 345, `Stop` 669 → 558 ms at p50); AGY `Stop` is still 931 ms and AGY `PreToolUse` 854 ms, because both evaluate the contracts over the whole transcript (`audit_session` after appending: about 550 ms CPU for 100 contracts against the 20 ms target). Warm calls and the first-call index build did not change (the work is the same): `check_action` with nothing appended 40 ms, `audit_session` 80 ms, `codex_session_hook` `Stop` about 150 ms at 10 contracts. Twelve idle local-only servers: 1073 → 949 MB total RSS (89.4 → 79.1 MB each, 637 → 570 MB PSS), because a local-only server no longer loads NumPy. The roundtable's own measurement of an idle OMP server was 86 MB and of a Codex server 79 MB; a server with an embedder still loads NumPy at startup.

Large session (`--large --skip-small`, opt-in), same machine: a 300 MB synthetic Codex rollout of about 200,000 events of 1.5 KB (the roundtable measured 314 MB and 225,000 events), one warm server process with a local-only engine. With one contract the numbers match the roundtable's measurement:

| call (CPU, p50 of 5) | 1 `must_not` contract | 10-contract mix, before | 10-contract mix, after |
|---|---:|---:|---:|
| first call (builds the index, once) | 26.20 s | 29.39 s | 29.37 s |
| `check_action`, nothing appended | 1.02 s | 4.71 s | 5.01 s |
| `check_action`, 4 KB appended | 1.32 s | 4.82 s | 5.11 s |
| `audit_session`, nothing appended | 2.69 s | 7.40 s | 7.27 s |
| `audit_session`, 4 KB appended | 3.05 s | 7.64 s | 7.65 s |
| `codex_session_hook` `PreToolUse` | 1.06 s | 4.61 s | 4.63 s |
| `codex_session_hook` `Stop` | 10.73 s | 15.18 s | 17.23 s |

Against the targets (`check_action` with nothing appended 100 ms, `audit_session` for a few appended events 20 ms) these are the work of the fast integrity path and the incremental audit, not of this change: before and after agree, which is why they are reported and not gated. With the 10-contract mix `check_action` costs 5 s because the `before_regex` obligations are evaluated over the whole history on every call. `codex_session_hook` `Stop` costs 10.7 s on this session: its evidence stamp reads and parses every line of the transcript before the audit runs.

Windows, reported separately (Windows 11, Python 3.13, the same machine, 12 interleaved runs per event, 10 contracts; before → after, median wall in ms): Claude Code `PostToolUse` 361 → 141, `UserPromptSubmit` 358 → 144, `SessionStart` 543 → 359, `PreToolUse` 625 → 473, `Stop` 716 → 552; AGY events 11 to 28 percent faster. Windows is slower in absolute terms (a hook's process start and imports cost about 1.3 to 2.7 times as much), so the events that are not gated can exceed their 1.5 times threshold there (`SessionStart` reaches 0.56 s CPU at p95); the gated `PostToolUse` and `UserPromptSubmit` stay at about 0.2 s p95 and pass with the default factor. Twelve idle local-only servers: 966 → 877 MB working set (80.6 → 73.3 MB each). A venv's `python.exe` is a launcher that starts the real interpreter, so the benchmark runs the process tree in a job object and reads the interpreter's own peak working set; Windows CPU times have the 15.6 ms resolution of the operating system clock. The roundtable measured the 314 MB session on Windows at commit `00cb90e` (local-only engine): server import 1.12 s, first `check_action` 44.8 s, `check_action` with nothing appended 1.50 to 1.58 s CPU, `audit_session` 4.22 s.

### Memory verification

tsukkomi is Python, but it runs native code: neograph-engine (the audit and GraphRAG graphs) and the sqlite-vector extension (dense search). After the 0.5.x memory incident the project checks both the native code and the long-running server, modelled on NeoGraph's sanitizer CI. `.github/workflows/memory.yml` runs three jobs:

| Job | What runs | Fails on |
|---|---|---|
| `asan` | neograph-engine and sqlite-vector rebuilt with AddressSanitizer + UndefinedBehaviorSanitizer; the whole suite with the runtime preloaded, `PYTHONMALLOC=malloc` and the native vector backend | any ASan or UBSan report; a direct leak allocated by neograph or sqlite-vector code |
| `tsan` | the same libraries rebuilt with ThreadSanitizer; the whole suite under `setarch -R` | any data race or deadlock report |
| `soak` | `scripts/memory_soak.py` with the official sqlite-vector release: one long-lived process drives the MCP tool entry points through steady, growing and churning sessions | retained Python heap growing in both measured windows of a phase, or growing file descriptors or threads |

How the sanitizer jobs work (`scripts/verify_memory.py`, standard library only, Linux and GCC):

- `native` downloads the neograph-engine sdist pinned in `pyproject.toml` and checks its SHA-256 against both PyPI and the digest pinned in the script, clones sqlite-vector `1.1.0` and checks that it resolves to the pinned commit, and builds both with the profile's flags. The builds are cached by these pins; a cold build of neograph-engine takes about 45 minutes on a hosted runner.
- `canary` builds a small library with one deliberate defect per check (heap overflow, signed overflow, a leak, a data race) and requires the sanitizer to report each. A run without reports therefore means the sanitizer was live, not that it was missing.
- `test` refuses to run unless the installed neograph module links the sanitizer runtime, so the release wheel can never be tested by mistake. It then runs the suite and scans the sanitizer logs and the test output.
- LeakSanitizer does not run at interpreter exit, where CPython leaves thousands of objects unfreed by design. `tests/conftest.py` runs LSan's recoverable check when the pytest session finishes, while live Python objects are still reachable. Each direct leak is attributed to the first frame above the allocator: neograph or sqlite-vector code fails the run, while CPython and third-party leaks are counted and printed.
- Under TSan, `/bin/sh` itself crashes with the runtime preloaded, so the two installer tests that start hook commands through a shell are deselected there.

Run them locally:

```bash
python scripts/verify_memory.py native --profile asan --out build/memory/asan
python -m pip install build/memory/asan/wheels/neograph_engine-*.whl && python -m pip install -e ".[dev]"
python scripts/verify_memory.py canary --profile asan --out build/memory/asan
python scripts/verify_memory.py test --profile asan --out build/memory/asan    # same for --profile tsan, in its own venv
python scripts/memory_soak.py --check --dense-backend sqlite-vector --vector-path /absolute/path/to/vector.so
```

Use a separate virtual environment per profile, because the sanitizer build replaces the release wheel. `TSUKKOMI_TEST_DENSE_BACKEND=sqlite-vector` with `SQLITE_VECTOR_TEST_PATH` runs any suite against a native vector library. Known limit: CPython 3.12 builds where interned strings are immortal (Ubuntu 24.04's 3.12.3, for example) keep each path component that `pathlib` interns, about 150 bytes per distinct session path. This is bounded by the number of sessions; newer 3.12 releases and 3.13 release them.

### Shared-index queue PoC

The opt-in `scripts/poc_shared_index_queue.py` experiment connects separate client processes to one authenticated IPv4 loopback service. It uses the real NeoGraph `RequestQueue` through a small C++/ctypes bridge and runs real `SelfDirectEngine` operations against a temporary synthetic index. It does not replace the deployed MCP transport, touch existing indexes, load API keys or add a CI gate.

Requirements: a POSIX host, an installed GCC-compatible C++20 compiler, the project Python dependencies, and a NeoGraph source checkout containing `include/neograph/util/request_queue.h` and `deps/concurrentqueue.h`. No dependencies are downloaded by the PoC.

```bash
python scripts/poc_shared_index_queue.py run --neograph-root ../NeoGraph --workers 4
```

The runner compiles only the bridge, compares 16 independent clients with one worker and the selected multiworker count, and checks real clean/violation verdicts, bounded rejection, queued expiry, disconnect cancellation, running deadlines and shutdown settlement. It separately reports `writer_lease_peak`, actual native audit `evaluation_peak`, metadata progress during evaluation and `residual_writer_contention`. Deliberate writer/native-stage holds are labeled diagnostic fixtures, not production latency. Wall throughput includes client startup and IPC and has no performance pass/fail threshold.

For an interactive experiment, use `serve --neograph-root ../NeoGraph --workers 1 --capacity 32`; its readiness line gives a private endpoint-file path. The `client --endpoint-file <path> --request '<JSON>'` mode supports `check_action`, `sync_session` with `embed=false`, `audit_session` and `audit_status` for the synthetic session. `{"op":"stop"}` closes the queue, settles accepted futures and shuts the service down naturally. Endpoint credentials and compiled libraries are temporary; do not commit an endpoint file.

A queued request that expires never enters the engine. An already-started request can temporarily outlive its response deadline: it reports `still_running`/unknown, then settles at a cooperative checkpoint or returns an actual admitted-work outcome. Multiple workers can overlap detached evaluation, but cannot bypass a genuinely held writer lease. The PoC never claims forced cancellation of a running syscall.

### Verification scope

Test results and staged secret-pattern scans are bounded evidence, not certification that arbitrary secrets or every host bypass are absent. Inspect matched staged lines before committing, keep credentials in an external authorized dotenv file, and never commit raw personal transcripts or local runtime state.

---

## Security and changelog

See [SECURITY.md](SECURITY.md) for supported versions and how to report a vulnerability, and [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

MIT License. See [LICENSE](LICENSE) for details.
