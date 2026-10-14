# Tsukkomi MCP (ツッコミ)

> A deterministic governance and evidence server for autonomous coding agents, version 0.6.0.
> The distribution is `tsukkomi-mcp`; the Python package is `self_directing_mcp`.

## Overview

Prompt instructions alone do not keep an agent from context drift, false completion reports or a forbidden command. Tsukkomi MCP indexes the agent's own session history on the local machine, checks proposed actions and finished work against explicit, user-registered contracts, tracks requirement checklists that need linked evidence, and optionally maintains a session knowledge graph.

Every check returns one of four verdicts: `clean`, `violation`, `suspicious` or `unknown`. `unknown` means the evidence needed for a verdict is missing or has changed, and it is never reported as `clean`. The one incomplete input that can still yield `clean` is a pure append to a growing session, which is judged as of the captured offset and reports that offset (see [Coverage and verdicts](#coverage-and-verdicts)). Whether a verdict blocks anything depends on the host (see [Supported hosts](#supported-hosts-and-parser-versions)). Rules are matched with regular expressions and structured evidence, not with an LLM judge.

```
                  ┌──────────────────────────────────────────────┐
                  │          Autonomous Agent Client             │
                  │  (Antigravity / Codex / OMP / Cursor, Grok)  │
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

Session evidence stays on the local machine. Remote calls are limited to embeddings, GraphRAG extraction and the opt-in workflow classifier, all sent to OpenRouter. Known secret patterns are masked before embedding and graph extraction, and the workflow classifier refuses context in which a secret pattern is found. See [SECURITY.md](SECURITY.md) and [Verification scope](#verification-scope).

## What is new in 0.6.0

Version 0.6.0 (see [CHANGELOG.md](CHANGELOG.md)) mainly fixes memory growth, append handling and concurrency in long-running servers.

- **Dependency:** `neograph-engine==0.14.1` (was 0.12.1).
- **Appending sessions sync normally ([#4](https://github.com/fox1245/tsukkomi-mcp/issues/4)).** Source capture streams a SHA-256 over the file up to its length at the first measurement and keeps no bytes. Records appended afterwards are reported as `unread_bytes` instead of failing the sync.
- **Audits judge a growing session as of the captured offset.** Results carry `judged_through_byte`, `unread_bytes` and `judged_prefix`; absence-based `must` obligations stay `unknown` while bytes are unread. See [Coverage and verdicts](#coverage-and-verdicts).
- **`check_action` does not trust older history for a proposed action ([#8](https://github.com/fox1245/tsukkomi-mcp/issues/8)).** A prerequisite that requires success stays `unknown` while required history is unread, because a newer failed run may be in the unread part.
- **Bounded memory ([#5](https://github.com/fox1245/tsukkomi-mcp/issues/5)).** Single-flight sync per session, streaming reparse, per-session id queries, float32 vectors, a cap on concurrent heavy calls (`SELF_DIRECT_MAX_CONCURRENT_CALLS`, default 4) and `malloc_trim` on glibc. See [Memory and concurrency](#memory-concurrency-and-deadlines).
- **Concurrent checklist updates ([#6](https://github.com/fox1245/tsukkomi-mcp/issues/6)).** `update_checklist` serializes plain edits and returns a `checklist_conflict` status instead of raising or overwriting.
- **Duplicate session sources ([#10](https://github.com/fox1245/tsukkomi-mcp/issues/10)).** Copies and continuations of an indexed session are recognised instead of reparsed on every sync (`alias_of`, `ambiguous_session_source`).
- **Parsers.** OMP parser v8 (host notices, images, steering attribution, `session_init`, `model_usage`, `branch_summary`, `redactedThinking`; [#9](https://github.com/fox1245/tsukkomi-mcp/issues/9)) and Codex parser v3 (multi-agent messages, tool search, new `session_meta` fields). Each bump reparses lazily, one session at a time.
- **CI.** A new Linux `memory-audit-regressions` job runs small offline memory, freshness and concurrency tests next to the existing smoke job.
- **Also in this release since 0.5.0:** the shared-index snapshot and request-deadline work ([#3](https://github.com/fox1245/tsukkomi-mcp/issues/3)), the AGY evidence workflow and approved-workflow layer, user path configuration in TOML, and a default index directory outside the checkout. Read [Upgrading to 0.6.0](#upgrading-to-060) before restarting existing servers.

---

## Key capabilities

### 1. Invariant contracts (`upsert_contracts`)
- **Prohibitions (`must_not`)** match a regex against proposed or recorded actions (`decision: deny` on hosts that enforce).
- **Obligations (`must`)** require matching evidence before a gated action (`before_regex`) or at a completion checkpoint, optionally with a successful linked tool result (`requires_success: true`).
- AGY returns `force_ask` for suspicious exceptions so a cached Always Allow grant does not replace confirmation.
- Contracts are global or scoped to one session. A description-only contract stays `unknown`, and search rankings are never violation evidence.

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
- **Secret masking:** known credential patterns are masked before embedding or retrieval. This is not a guarantee that arbitrary personal data or every secret format is removed.

---

## Supported hosts and parser versions

| Host | Provider id | Evidence source | Lifecycle integration | Blocks actions? | Parser version |
|---|---|---|---|---|---|
| Google Antigravity (AGY) | `agy` | `brain/<conversation-id>/.system_generated/logs/transcript*.jsonl` plus hook receipts | Stdio hooks: `PreToolUse`, `PostToolUse`, `PreInvocation`, `Stop` | Yes in the default `enforced` mode; `advisory` is explicit. See [docs/agy-installation.md](docs/agy-installation.md) | 5 |
| OpenAI Codex CLI | `codex` | Rollout JSONL in the Codex sessions directory | Three `mcp_tool` hooks installed by `scripts/install_codex.py`: `UserPromptSubmit`, `PreToolUse`, `Stop` | No. Hooks are advisory | 3 |
| OMP | `omp` | OMP session JSONL | Opt-in extension `extensions/tsukkomi-omp.ts` | Only confirmed `violation`/`suspicious` on model-issued tools. `unknown` and transport failures never block | 8 |
| Cursor / Grok Bot | `grokbot` | `agent-transcripts` roots | None; call the MCP tools manually | No | 1 (unversioned) |

A parser version is recorded per session. When a provider's version is raised, each already-indexed session is reparsed from byte 0 on its next sync or audit, one session at a time. Unchanged event ids, anchors and cached vectors survive, and no embedding API is called when `embed=false`. A running MCP process only picks up a new parser after it is restarted.

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

---

## MCP tools reference

| Category | Tool | Description |
|---|---|---|
| **Governance & Contracts** | `upsert_contracts` | Create or update explicit invariant rules (`must` / `must_not`). |
| | `list_contracts` | Inspect active contracts for the current session and global scope. |
| | `revoke_contract` | Disable a contract while preserving history. |
| | `check_action` | Pre-flight test of a proposed action against current contracts. |
| | `audit_session` | Audit executed session history against active contracts. |
| | `audit_status` | Health check: vector extension, index schema, degraded flags, per-session coverage. |
| **Checklist & Evidence** | `update_checklist` | Update requirements; enforces evidence gating for completion. |
| | `get_checklist` | Inspect checklist status, history and linked evidence. |
| **GraphRAG & Architecture** | `get_graph_context` | Query entity dependencies and the active knowledge graph. |
| | `propose_graph_update` | Validate node/edge additions with evidence chunk requirements. |
| | `commit_graph_update` | Atomically commit validated graph updates with cursor progression. |
| | `run_neograph_update` | Run the LLM GraphRAG extraction pipeline through NeoGraph. |
| **Search & Analytics** | `sync_session` | Incremental evidence sync for Codex, Grok Bot, OMP and AGY transcripts. |
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

Settings come from, in order of precedence: explicit `Settings` arguments, environment variables, the TOML path file, a `.env` in the **current working directory**, then defaults. A server started by an MCP host usually has an unrelated working directory, so do not depend on a `.env` there; use explicit `SELF_DIRECT_*` variables in the host's server entry or the TOML file below.

#### User path configuration (TOML)

Use `~/.config/tsukkomi-mcp/config.toml` on any supported platform, or set `SELF_DIRECT_CONFIG_FILE` in the process environment to select another file. Copy and edit [config.example.toml](config.example.toml); the edited file belongs outside the repository and agent-writable workspaces.

```toml
[paths]
index_dir = "~/private-tsukkomi/index"
omp_sessions_dir = "~/.omp/agent/sessions"
openrouter_api_key_file = "../credentials/openrouter.env"
sqlite_vector_path = "native/vector.so"
```

The eight supported keys are `index_dir`, `codex_sessions_dir`, `omp_sessions_dir`, `grokbot_transcripts_dir`, `agy_app_data_dirs`, `contracts_path`, `sqlite_vector_path` and `openrouter_api_key_file`. AGY roots use a string array; Grokbot accepts its existing delimiter-separated string or an array of individual paths. Empty arrays disable those optional roots.

TOML paths expand `~`; relative paths resolve against the canonical TOML file's directory, not the working directory or a symlink's directory. `${ENV_VAR}` interpolation is not performed. The selector `SELF_DIRECT_CONFIG_FILE` is environment-only, so a `.env` cannot silently select another TOML file. An absent default file keeps the defaults. An explicitly selected missing file, malformed TOML, an unknown section or key, or an invalid path value fails instead of silently selecting another index. Settings capture the selected sources, so restart affected servers and extensions after changing them.

#### Environment settings

All settings use the `SELF_DIRECT_` prefix except the key itself. The names below are the runtime fields in `src/self_directing_mcp/config.py`; [.env.example](.env.example) lists the commonly used ones.

| Setting | Default | Meaning |
|---|---|---|
| `SELF_DIRECT_INDEX_DIR` | `~/.self_direct_index` | Private index directory (events, FTS, vectors, contracts, checklists). |
| `SELF_DIRECT_CONTRACTS_PATH` | `<index_dir>/contracts.json` | Contracts file. |
| `SELF_DIRECT_SESSION_PROVIDER` | `codex` | Provider used when a tool call omits `provider`: `codex`, `grokbot`, `omp` or `agy`. |
| `SELF_DIRECT_CODEX_SESSIONS_DIR`, `CODEX_SESSIONS_DIR` | `~/.codex/sessions` | Codex session root. |
| `SELF_DIRECT_OMP_SESSIONS_DIR` | `~/.omp/sessions` | OMP session root. |
| `SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR` | empty (disabled) | Cursor / Grok Bot transcript roots. |
| `SELF_DIRECT_AGY_APP_DATA_DIRS` | `~/.gemini/antigravity`, `-cli`, `-ide` | JSON array of AGY app-data directories. |
| `SELF_DIRECT_AGY_ENFORCEMENT` | `enforced` | `enforced` or `advisory`. |
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

To enable deterministic pre-tool lifecycle interception in AGY, register `scripts/agy_hook.py` as your workspace hook command. See [docs/agy-installation.md](docs/agy-installation.md) for the hook file, enforcement modes and host boundaries.

### OpenAI Codex CLI

Install advisory lifecycle hooks and register the MCP server in one command:

```bash
python scripts/install_codex.py --trust-hooks
python scripts/install_codex.py --verify
```

The installer registers explicit `SELF_DIRECT_INDEX_DIR`, session-root, backend and key-file settings in the Codex MCP entry, so it is not affected by the default index location. See [docs/codex-installation.md](docs/codex-installation.md).

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

`sync_session`, `audit_session` and `check_action` report a `coverage` object (`audit_status` reports the same data under `session` when given a session id). It holds the indexed source `path`, `byte_offset` (how far the index has parsed), `file_size`, `chunk_count`, `parser_version`, an `issues` list and `complete`, which is true only when `issues` is empty.

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

A rewrite or truncation of the same path is still reparsed or rejected as before.

### Source capture

Sync captures at most the length measured at the first `fstat`, in 1 MiB blocks, keeping only SHA-256 states at the cursor, at the last complete record and at the end. No file bytes are retained, so memory does not scale with the session size.

- Growth after the first measurement is not an error. An overwrite, truncation or replacement during the pass is.
- `validate_prefix()` accepts pure growth after re-hashing the captured prefix; it runs outside the shared publication lock. Under the lock, only identity, size and a SHA-256 of the last 64 KiB are re-checked. A mismatch, once detected, stays failed.
- `validate()` compares metadata only (device, inode, size, mtime, ctime). Equal stamps do not prove equal content.
- `stat` and `fstat` can give `st_ctime` different meaning on some runtimes (for example 9p/drvfs), so full stamps are compared only within the same API and identity, size and mtime across them. When either side reports `st_ino` 0, which some Windows filesystems do, the inode comparison is skipped and size, mtime and the hash decide.
- Limits: capture and validation are not an atomic snapshot. A rewrite outside the final 64 KiB between the last full hash and the lock is accepted at that moment, and the next sync reindexes because the stored cursor's prefix hash no longer matches. On Windows `st_ctime` is the creation time, so a same-size rewrite that restores mtime is invisible to `stat` and is caught by the same prefix-hash check on the next sync.

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
2. **Parser bumps reparse lazily.** The OMP parser is now version 8 (0.5.0 had 2) and the Codex parser version 3 (was 2). Each session is reparsed once, on its next sync or audit, within the bounded-memory path. There is no upfront migration step and no storm of reparses.
3. **Existing indexes are preserved.** Events, anchors, model-specific vector caches, contracts and checklists are kept. Unchanged event ids and cached vectors survive a reparse. Obsolete entries are cleaned through a durable journal, so an interrupted upgrade resumes.
4. **Check the index location.** The default index is `~/.self_direct_index` unless `SELF_DIRECT_INDEX_DIR` or `index_dir` in TOML overrides it. Older versions defaulted to a `.self_direct_index` inside the checkout. An install that relied on that default must set `SELF_DIRECT_INDEX_DIR` (or TOML `index_dir`) to the existing absolute directory before restarting; nothing is moved or copied automatically. The Codex installer and managed OMP registrations already pass an explicit path.
5. **Check how the key file is selected.** A checkout `.env` is no longer treated as an authoritative key file. Select it explicitly with `SELF_DIRECT_OPENROUTER_API_KEY_FILE` or TOML `openrouter_api_key_file`, or supply `OPENROUTER_API_KEY` through the environment or a `.env` in the server's working directory.
6. **Reinstall dependencies.** `neograph-engine` must be exactly 0.14.1. `pip install -e ".[dev]"` or `scripts/install_codex.py` takes care of it.
7. **Re-verify.** Run `python scripts/install_codex.py --verify` for Codex and `audit_status` for any host. Confirm `neograph_version`, `embedder`, `dense_backend` and that `degraded_flags` is empty.

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

AGY rejects direct mapped completion actions and directs the agent to the owned executor; its normal Stop hook holds an incomplete workflow and provides requirements, evidence, and permitted recovery actions. Cancellation, error stops, and pauses for background work are not restarted. Reads, repairs, and checks remain available unless a separate explicit contract restricts them. **Unmapped tools, direct shell access outside registered hooks, and the advisory Codex adapter are not an enforcement boundary.**

Lean proves the policy/model under its stated assumptions; it does not prove arbitrary Python/TypeScript implementations equivalent to that model. Real tests supply implementation evidence only for exercised cases. The workflow is separate from session GraphRAG extraction and does not require an LLM to decide whether recorded tests passed.

### Reproducible synthetic policy evaluation

```bash
TSUKKOMI_TEST_LEAN=lean python -m pytest -q
python scripts/benchmark_workflow.py --lean lean
```

The benchmark compares 640 finite stage/class/evidence combinations with an independent reference, measures false allowances/blocks and per-stage latency, and exercises correction of each missing mandatory fact. It is an adversarial **policy** evaluation, not live JEV accuracy or a whole-host coverage claim. Live JEV validation requires an OpenRouter key; offline transport fixtures never count as live service evidence. A real low-confidence Choice remains nonauthorizing: the proposed label and observed probabilities are retained, but completion stays locked.

---

## Testing and verification

The suite covers parsers, host receipt linkage, restricted reads, temporal obligations, append capture and prefix judgement, duplicate session sources, single-flight and concurrency, workflow version/grant invalidation, proof rejection, path isolation and MCP transport. Tests use synthetic sessions and offline embeddings. Lean integration tests, Windows-only path tests and bubblewrap tests report a skip when the toolchain or platform is unavailable.

```bash
python -m pytest -q
python -m build
python scripts/benchmark.py
```

### CI

GitHub Actions runs two jobs on pushes and pull requests. Superseded runs for the same ref are cancelled, and no API keys or paid model calls are used.

- **`test`** (Ubuntu and Windows, Python 3.12, 10 minutes): the explicit MCP transport, Jev boundary and Windows session-path smoke tests, then builds the source distribution and wheel.
- **`memory-audit-regressions`** (Ubuntu, Python 3.12, 15 minutes): small offline regression files for, among others, audit memory, prefix judgement, current-action prerequisites, append and source capture, single-flight, index and server concurrency, deadlines, checklist conflicts, quiet hooks, AGY hooks and hook gating. It runs with the fake embedder and the NumPy backend and deselects the one test that measures allocation of a large capture. These tests check ratios on small inputs; they do not certify live-service memory.

The full suite, real Lean proof checks, native vector tests and all performance, large-session and concurrency benchmarks are local deep checks, not CI gates. A green CI run does not establish full workflow or native-proof coverage or performance guarantees. Live model validation needs a separate model and budget approval.

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
