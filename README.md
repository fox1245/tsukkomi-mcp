# Tsukkomi MCP (ツッコミ)

> Deterministic governance, verifiable checklists, and session GraphRAG for autonomous coding agents (`v0.11.0`).
> Distribution: `tsukkomi-mcp` · Python package: `self_directing_mcp`

## Overview

Prompt instructions alone do not prevent context drift, premature completion claims, or forbidden tool calls. **Tsukkomi MCP** indexes local agent session transcripts, evaluates proposed actions and completed turns against deterministic contracts, enforces evidence-gated checklists, and maintains a session knowledge graph.

Every check returns one of four verdicts:
- `clean` — all applicable contracts are satisfied by verified session evidence.
- `violation` — a forbidden action (`must_not`) or broken ordering rule was observed.
- `suspicious` — a heuristic or partial hazard indicator requires inspection.
- `unknown` — required evidence is missing, unindexed, or unreadable (`unknown` is never treated as `clean`).

```
                  ┌──────────────────────────────────────────────┐
                  │          Autonomous Agent Client             │
                  │ (Claude Code / AGY / Codex / OMP / Grok CLI) │
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
│  - Regex + path_guard │     linked evidence    │  - Context injection (AGY)    │
├───────────────────────┴────────────────────────┴───────────────────────────────┤
│                 Local session index (SQLite, FTS5, sqlite-vector)              │
│     - Cosine vector search  - FTS5 BM25 and regex  - Known-secret masking      │
└────────────────────────────────────────────────────────────────────────────────┘
```

Session transcripts remain on the local machine. Outbound payloads (tool outputs, hook messages, logs, and remote embedding/extraction requests) are masked for known secret and PII patterns; see [SECURITY.md](SECURITY.md) and [docs/reference.md](docs/reference.md#verification-scope).

## Quickstart

### Prerequisites

- **Python 3.12+**
- **`neograph-engine==0.14.1`** (installed automatically with the package)
- **`sqlite-vector`** native extension (for the default local vector backend) or PostgreSQL with `pgvector`

### 1. Clone and install

```bash
git clone https://github.com/fox1245/tsukkomi-mcp.git
cd tsukkomi-mcp
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Provision sqlite-vector

```bash
python scripts/setup_sqlite_vector.py
```

### 3. Configure

#### User path configuration (TOML)

Create `~/.config/tsukkomi-mcp/config.toml` (or set `SELF_DIRECT_CONFIG_FILE`) from [config.example.toml](config.example.toml) so the MCP server and lifecycle hooks share the same index and contract paths:

```toml
[paths]
index_dir = ".self-directing-index"
contracts_path = ".self-directing-index/contracts.json"
sqlite_vector_path = "/absolute/path/to/vector.so"
openrouter_api_key_file = "~/.config/tsukkomi-mcp/openrouter.key"
```

#### Environment settings

Copy [.env.example](.env.example) to `.env` when using environment overrides. Key settings:

| Setting | Default | Purpose |
| :--- | :--- | :--- |
| `SELF_DIRECT_INDEX_DIR` | `.self-directing-index` | Local SQLite/FTS5/vector index directory |
| `SELF_DIRECT_DENSE_BACKEND` | `sqlite-vector` | Dense backend (`sqlite-vector`, `pgvector`, or `numpy`) |
| `SELF_DIRECT_EMBEDDING_BASE_URL` | OpenRouter API | OpenAI-compatible `/v1/embeddings` endpoint (e.g. local TEI or OpenRouter) |
| `SELF_DIRECT_EMBEDDING_MODEL` | `qwen/qwen3-embedding-8b` | Embedding model identifier |
| `SELF_DIRECT_LOCAL_ONLY` | `false` | Disable remote embedding and GraphRAG network calls when `true` |
| `SELF_DIRECT_PROJECT_CONTRACTS` | `candidates` | Project contract bundle handling (`candidates` or `off`) |
| `SELF_DIRECT_CONTRACT_BUNDLE_PATHS` | `.tsukkomi/contracts.json` | Relative path(s) searched for portable contract bundles |

See [docs/reference.md](docs/reference.md#environment-settings) for the full configuration table (deadlines, incremental shadow audits, resident daemon, and CJK sparse indexing).

#### Credentials

Store API keys in a user-only file (`chmod 600 ~/.config/tsukkomi-mcp/openrouter.key`) referenced by `openrouter_api_key_file` or `SELF_DIRECT_OPENROUTER_API_KEY_FILE`. Use `SELF_DIRECT_EMBEDDING_API_KEY` (or `SELF_DIRECT_EMBEDDING_API_KEY_FILE`) when authenticating against a separate embedding server.

## Platform configuration

Each supported host has a dedicated installation guide and hook template under `docs/`:

| Host | Parser | Hook / Setup Command | Guide & Example |
| :--- | :--- | :--- | :--- |
| **Google Antigravity (AGY)** | v9 | `scripts/agy_hook.py` | [docs/agy-installation.md](docs/agy-installation.md) · [example](docs/agy-hooks.example.json) |
| **OpenAI Codex CLI** | v3 (`.jsonl` / `.zst`) | `tsukkomi-install-codex` | [docs/codex-installation.md](docs/codex-installation.md) · [example](docs/codex-hooks.example.json) |
| **Claude Code** | v4 | `python scripts/install_claude_hooks.py` | [docs/claude-installation.md](docs/claude-installation.md) · [example](docs/claude-hooks.example.json) |
| **Grok CLI** | v2 | `python scripts/install_grok_hooks.py` | [docs/grok-installation.md](docs/grok-installation.md) · [example](docs/grok-hooks.example.json) |
| **OMP** | v8 | `extensions/tsukkomi-omp.ts` (`--omp-bridge`) | [docs/reference.md](docs/reference.md#omp-openrouter-backed-retrieval-deterministic-contract-hooks) |
| **Cursor / Grok Bot** | — | Set `grokbot_transcripts_dir` (`provider="grokbot"`) | [docs/reference.md](docs/reference.md#cursor--grok-bot) |

## Key capabilities

### 1. Invariant contracts (`upsert_contracts`)

Register deterministic `must` and `must_not` rules scoped to a session or globally (`session_id: null`). Contracts match tool calls, results, or assistant messages via time-bounded regular expressions or typed path guards (`path_guard`). Note: A contract about a file must also match the calls of the host's file-read tools (Claude Code Read, AGY view_file, OMP read), whose text is `[tool_call:<name>] {json}` and not a shell command, so name the file itself rather than only shell readers such as cat; the fragility report in the response lists such spellings.

#### Contract fields: `risk`, `auto_continue` and `when`

- `risk` (`"low"` | `"high"`): when `"high"`, an unverifiable (`unknown`) pre-tool check escalates to host confirmation (`ask`) instead of passing silently.
- `auto_continue` (`bool`): controls whether a `Stop` hook may automatically resume the agent on a violation.
- `when`: activates a conditional obligation only after a matching trigger event is observed in the session.
- `enforcement` (`"enforce"` | `"inject_only"`): `"inject_only"` surfaces contract reminders in prompt hooks without failing `audit_session` or `check_action`.

For typed path contracts (`path_guard`), fragility reports, and portable bundles (`SELF_DIRECT_PROJECT_CONTRACTS` / `SELF_DIRECT_CONTRACT_BUNDLE_PATHS`), see [docs/reference.md](docs/reference.md#1-invariant-contracts-upsert_contracts).

### 2. Checklist with evidence gating (`update_checklist`)

Track multi-step engineering requirements per session. Moving an item to `done` requires `evidence_chunk_ids` that resolve to real indexed events in the same session.

### 3. Session GraphRAG and context injection (`get_graph_context`)

Build and query a session knowledge graph (`run_neograph_update`, `propose_graph_update`, `commit_graph_update`, or host-side `graph_next_batch` / `graph_submit_batch`). Detailed batch limits and text validation rules are documented in [docs/reference.md](docs/reference.md#host-side-graphrag-extraction).

### 4. Hybrid history search

`search_history` combines dense cosine similarity (`sqlite-vector` or `pgvector`) and FTS5 BM25 sparse search (including CJK trigram/prefix indexing) via Reciprocal Rank Fusion (RRF).

## MCP tools reference

| Category | Tool | Description |
|---|---|---|
| **Governance & Contracts** | `upsert_contracts` | Create or update explicit invariant rules (`must` / `must_not`). A `must_not` about a file can carry a typed [`path_guard`](docs/reference.md#typed-path-contracts-path_guard) instead of a regex. The response adds an informational [fragility report](docs/reference.md#contract-fragility-report-informational): equivalent spellings the saved regexes miss, and a `suggested_path_guard` for a shell-scoped regex that names a path. |
| | `list_contracts` | Inspect active contracts for the current session and global scope, plus any `pending` weakening awaiting a user anchor. |
| | `list_contract_candidates` | Read-only. Contract bundle files this device can see, as candidates with a status per contract; nothing is enforced ([Sharing contracts across devices](docs/reference.md#sharing-contracts-across-devices)). |
| | `adopt_contracts` | Adopt a candidate bundle as this device's own contracts. Needs `source_event_id`, a user message of this device's index; the bundle's hash must match; conflicts are not adopted unless named in `replace_ids`. |
| | `export_contracts` | Write this device's global contracts as a portable bundle, to the project or a configured personal file, atomically and with a fixed mode. |
| | `revoke_contract` | Disable a contract while preserving history. Takes `source_event_id` (the user's message that asked for it); without one the request is held as pending and the contract stays in force. |
| | `check_action` | Pre-flight test of a proposed action against current contracts. `verify="fast"` selects the hook-time source verification; the result's `integrity` says what was compared. |
| | `audit_session` | Audit executed session history against active contracts (always a full source verification). |
| | `audit_status` | Health check: vector extension, index schema, degraded flags, per-session coverage, and `ops`, the [request statistics](docs/reference.md#timeouts-and-what-a-deadline-leaves-behind) of the process (answered even while the engine is busy). |
| | `job_status` | Poll live progress (`stage`, `bytes_done`, `chunks_done`, `embedded_done`, `elapsed_ms`, `eta_ms`) for an in-flight session sync or background embedding job (`#51`). |
| | `test_contract` | Pre-save dry-run harness for candidate contract rules: evaluates positive/negative `cases`, synthesizes canonical tool calls, checks per-case `max_ms`, and reports baseline diffs/regressions (`#57`). |
| | `grant_once` | Create a scoped, bounded one-time grant for a `must_not` contract anchored to an indexed user approval message (`source_event_id`) and verified against Canonical English (`#56`). |
| **Checklist & Evidence** | `update_checklist` | Update requirements; enforces evidence gating for completion. |
| | `get_checklist` | Inspect checklist status, history and linked evidence. |
| **GraphRAG & Architecture** | `get_graph_context` | Query entity dependencies and the active knowledge graph. |
| | `propose_graph_update` | Validate node/edge additions with evidence chunk requirements. |
| | `commit_graph_update` | Atomically commit validated graph updates with cursor progression. |
| | `run_neograph_update` | Run the LLM GraphRAG extraction pipeline through NeoGraph. |
| | `graph_next_batch` | Next batch of events for the **calling agent** to extract (no model call, key use or network access here); `after_order` looks ahead, `max_chars` and `max_events` size the result. See [Host-side GraphRAG extraction](docs/reference.md#host-side-graphrag-extraction). |
| | `graph_submit_batch` | Validate and commit the agent's extraction of one batch, in order and idempotently; a batch without relations needs a recorded reason. |
| **Search & Analytics** | `sync_session` | Incremental evidence sync for Codex, Grok Bot, OMP, AGY and Claude Code transcripts. A session too large for the request budget is indexed in slices: the result's `sync` object says how far it got ([Large sessions and deadlines](docs/reference.md#large-sessions-and-deadlines)). |
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

## Documentation

| Document | Contents |
| :--- | :--- |
| [docs/reference.md](docs/reference.md) | Full technical reference: contract semantics (`path_guard`, fragility, portable bundles), parser versions, GraphRAG batch rules, dense/sparse index internals, incremental audits, resident daemon, workflows, and benchmarks |
| [docs/local-embeddings.md](docs/local-embeddings.md) | Running local embedding servers (TEI, vLLM, Ollama) with Tsukkomi MCP |
| [docs/reliability-spec.md](docs/reliability-spec.md) | Reliability and fault-handling specification |
| [docs/timeout-fix.md](docs/timeout-fix.md) | Bounded slice deadlines and timeout architecture |
| [CHANGELOG.md](CHANGELOG.md) | Release history and upgrade notes (`0.6.0` – `0.11.0`) |
| [SECURITY.md](SECURITY.md) | Secret/PII egress masking contract and security policy |

## License

MIT — see [LICENSE](LICENSE).
