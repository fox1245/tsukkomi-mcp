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
      "args": ["-m", "self_directing_mcp"],
      "env": {
        "OPENROUTER_API_KEY": "your-api-key-here"
      }
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
