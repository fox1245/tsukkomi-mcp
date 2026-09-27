# 💥 Tsukkomi MCP (ツッコミ)

> **The Deterministic "No-BS" Governance & Tsukkomi Harness for Autonomous AI Agents**  
> *When your AI coding agent tries to delete files, leak secrets, write fake "done" reports, or touch unauthorized repos, Tsukkomi immediately slaps it down.*  
> *Enforce hard invariant contracts, verifiable evidence gates, and session GraphRAG across Google Antigravity (AGY), OpenAI Codex, and Cursor/Grok Bot.*

---

## Overview

Autonomous coding agents are powerful, but prompt-based instructions alone are vulnerable to hallucination, context drift, false completion claims, and unintended destructive actions.

**Tsukkomi MCP** is an authoritative, deterministic governance harness. It runs alongside your agent, intercepting tool calls at the lifecycle hook level, checking actions against explicit invariant contracts, maintaining verifiable evidence checklists, and injecting live architectural knowledge graphs into agent context.

```
                  ┌──────────────────────────────────────────────┐
                  │          Autonomous Agent Client             │
                  │   (Google Antigravity / Codex / Cursor)      │
                  └──────────────────────┬───────────────────────┘
                                         │
                               Lifecycle Hooks (PreToolUse / Stop)
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                           Self-Directing MCP Engine                            │
├───────────────────────┬────────────────────────┬───────────────────────────────┤
│  Invariant Contracts  │   Verifiable Evidence  │      Session GraphRAG         │
│  - must / must_not    │   Checklist Gate       │  - Entity dependencies        │
│  - Hard deny / ask    │   - No done without    │  - NeoGraph engine            │
│  - Session isolation  │     linked evidence    │  - Context injection          │
├───────────────────────┴────────────────────────┴───────────────────────────────┤
│                 Privacy-Preserving Local Index & SQLite-Vector                 │
│      - Cosine vector search (native AVX2)  - FTS5 BM25  - Secret Masking       │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Capabilities

### 1. 🛑 Deterministic Invariant Contracts (`upsert_contracts`)
- Define explicit, non-negotiable rules for your agent:
  - **Prohibitions (`must_not`)**: Intercept dangerous commands, unapproved pushes, or directory escapes before execution (`decision: deny`).
  - **Mandatory Prerequisites (`must`)**: Enforce testing before deployment (`requires_success: true`).
  - **Confirmation Prompts**: Convert risky actions into interactive user confirmations (`decision: ask`).
- Contracts can be **global** or strictly **scoped per session**.

### 2. 📋 Verifiable Checklist with Evidence Gating (`update_checklist`)
- Track user requirements with provenance and strict completion criteria.
- **Evidence Gating**: An agent cannot mark an item as `done` simply by claiming "I did it." The harness automatically downgrades unbacked completion claims to `pending_verification`.
- Only when verifiable `evidence_chunk_ids` (tool results, test artifacts, subagent logs) are attached can a requirement transition to `done (verified=True)`.

### 3. 🕸️ Session-Scoped GraphRAG & Context Injection (`get_graph_context`)
- Maintain an active entity dependency graph (nodes, relations, and cursors) powered by SQLite and the **NeoGraph** execution engine.
- Supported strictly-typed relations: `IMPLEMENTS`, `DEPENDS_ON`, `MODIFIES`, `PRODUCES`, `CHECKS_VERSION`, `EVIDENCED_BY`, `SUPERSEDES`.
- Graph updates are validated against session history and applied idempotently.
- In Google Antigravity, active graph relationships are delivered into agent context before each tool invocation.

### 4. 🔒 Multi-Runtime Lifecycle Hooks
- **Google Antigravity (AGY)**: High-performance stdio hook adapter (`scripts/agy_hook.py`) supporting `PreToolUse` contract interception and `Stop` evidence checklist verification.
- **OpenAI Codex**: Native lifecycle hook installer (`scripts/install_codex.py`) with advisory UserPromptSubmit, PreToolUse, and Stop hooks.
- **Cursor / Grok Bot**: Dynamic transcript detection and multi-root parsing.

### 5. 🔍 Privacy-Preserving Hybrid Search
- Dense cosine similarity using native `sqlite-vector` (AVX2-accelerated C extension) combined with SQLite FTS5 BM25 and Regex.
- **Automatic Secret Masking**: API keys, bearer tokens, and private credentials are automatically masked before any embedding or text retrieval.

---

## 🛠️ MCP Tools Reference (17 Tools)

| Category | Tool | Description |
|---|---|---|
| **Governance & Contracts** | `upsert_contracts` | Create/update explicit invariant rules (`must` / `must_not`). |
| | `list_contracts` | Inspect active contracts for current session and global scope. |
| | `revoke_contract` | Safely disable a contract while preserving history. |
| | `check_action` | Pre-flight test a proposed action against current contracts. |
| | `audit_session` | Audit executed session history against active invariant rules. |
| | `audit_status` | Health check: vector extension, index schema, and degraded flags. |
| **Checklist & Evidence** | `update_checklist` | Update requirements; enforces evidence gating for completion. |
| | `get_checklist` | Inspect requirement checklist status, history, and linked evidence. |
| **GraphRAG & Architecture** | `get_graph_context` | Query entity dependencies and active knowledge graph. |
| | `propose_graph_update`| Validate node/edge additions with evidence chunk requirements. |
| | `commit_graph_update` | Atomically commit validated graph updates with cursor progression. |
| | `run_neograph_update` | Execute live LLM GraphRAG extraction pipeline via NeoGraph. |
| **Search & Analytics** | `sync_session` | Incremental evidence sync across Codex/Cursor transcripts. |
| | `search_history` | Hybrid retrieval (Dense vector + BM25 + Regex) over history. |
| | `get_chunk` | Retrieve raw evidence chunk with automatic secret masking. |
| | `analyze_activity` | SQLite time-series analysis: retry gaps, error bursts, stalled items. |
| **Hooks** | `codex_session_hook` | Adapter for Codex lifecycle events and local audit context. |

---

## 🚀 Quickstart

### Prerequisites
- Python 3.12+
- `uv` (recommended) or `venv`
- `sqlite-vector` native extension (optional for dense embeddings; setup script included)

### 1. Clone & Install

```bash
git clone https://github.com/fox1245/tsukkomi-mcp.git ~/.mcp-servers/tsukkomi-mcp
cd ~/.mcp-servers/tsukkomi-mcp

uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Setup SQLite-Vector (Optional, for Dense Search)

```bash
python scripts/setup_sqlite_vector.py
```

Downloads the platform-appropriate binary release verified with SHA-256 checksums into `.native/`.

---

## 🔌 Platform Configuration

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

To enable deterministic pre-tool lifecycle interception in AGY, register `scripts/agy_hook.py` as your workspace hook command.

### OpenAI Codex CLI

Install advisory lifecycle hooks and register the MCP server in one command:

```bash
python scripts/install_codex.py --trust-hooks
python scripts/install_codex.py --verify
```

### OMP (OpenRouter-backed retrieval, deterministic contract hooks)

Register this server in OMP's **user** `mcp.json` and opt in to `extensions/tsukkomi-omp.ts` as an OMP user extension. Configure both the MCP entry and the extension's process with the same private index, OMP session root, and authorized credential/vector references:

```text
SELF_DIRECT_LOCAL_ONLY=false
SELF_DIRECT_OPENROUTER_API_KEY_FILE=<authorized Codex/OpenRouter dotenv path>
SELF_DIRECT_SQLITE_VECTOR_PATH=<installed native vector library path>
SELF_DIRECT_OMP_SESSIONS_DIR=<OMP user session root>
SELF_DIRECT_INDEX_DIR=<private writable index>
TSUKKOMI_OMP_MONITOR=1
TSUKKOMI_OMP_PYTHON=<Python with tsukkomi-mcp dependencies>
TSUKKOMI_OMP_SOURCE=<source checkout>/src  # omit when installed in that Python
```

OpenRouter-backed hybrid/dense history search, `sync_session(embed=True)`, and NeoGraph updates are available by default; embedding uploads sanitized session chunks, so use an authorized key and session root. The hook's `check_action`/`audit_session` stay deterministic and refresh JSONL without embedding; they do **not** call OpenRouter on every tool/turn. Explicit offline-only mode is `SELF_DIRECT_LOCAL_ONLY=true`, which disables those remote functions.

Keep machine-specific values in local OMP configuration or a private extension copy, not this repository. Start a **new persisted OMP session**; `--no-session` cannot provide history. Register only user-stated, machine-checkable contracts with `upsert_contracts` (`provider: "omp"` and the actual OMP session ID for session scope). The extension checks pending model-issued tools through local MCP before execution, blocks confirmed `violation`/`suspicious`, and re-audits after results and turns. `unknown` and transport failure are reported but do **not** block; zero applicable contracts is not `clean`. The OMP JSONL parser treats unrecognized events as incomplete coverage. Direct shell/host actions outside OMP's tool pipeline, skipped hooks, in-memory sessions, and a result not yet flushed to disk are not covered.

Known OMP presentation changes, compaction records, agent notifications, empty messages, and UI metadata are indexed without treating them as assistant tool actions. Execution-start markers are matched against earlier indexed tool calls even when the session is synced between the call and marker. Unknown event shapes and unmatched markers still make coverage incomplete; restart the OMP MCP process after a parser upgrade so the parser version triggers a full local reindex.

The extension also registers the read-only `tsukkomi_session_context` tool. Call it before creating a session-scoped contract or checklist item: its `session_id` comes from the active OMP session manager. Never infer a session ID from `PI_SESSION_FILE` or a shell variable, which can belong to a parent OMP process. If no persisted session is available, do not silently create a global contract. `AGENTS.md` instructions alone are not audit contracts: register the user's explicit, machine-checkable prohibitions; broader subjective mistakes are not automatically judged by an LLM.

---

## 🧪 Testing & Audit Certification

The repository includes a comprehensive test suite covering parsers, streaming appends, redaction, temporal obligations, Windows UNC traversal defense, and MCP transport:

```bash
pytest -q
python scripts/benchmark.py
```

### 🏅 LLM-as-a-Judge Audit Passed (100%)
All 37 commits in this repository have been independently reviewed and certified safe by **9 LLM-as-a-judge subagents** across 3 batches ($X = N \times Y \times Z$ formula):
- Zero secret / credential leaks
- Zero unresolved hardcoded personal paths
- Zero destructive shell scripts
- Full memory & path-traversal boundary guarantees

---

## License

MIT License. See [LICENSE](LICENSE) for details.
