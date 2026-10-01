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
- **Confirmation Prompts**: AGY returns `force_ask` for suspicious exceptions so a cached Always Allow grant does not replace confirmation.
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
- **Google Antigravity (AGY)**: Stdio hooks cover every tool, local transcript synchronization, host-observed Pre/Post execution receipts, normal-stop checklist enforcement, and pre-invocation context. Unknown verification denies in enforced mode; advisory mode must be explicitly selected. See [installation and host boundaries](docs/agy-installation.md).
- **OpenAI Codex**: Native lifecycle hook installer (`scripts/install_codex.py`) with advisory UserPromptSubmit, PreToolUse, and Stop hooks.
- **Cursor / Grok Bot**: Dynamic transcript detection and multi-root parsing.

### 5. 🔍 Privacy-Preserving Hybrid Search
- Dense cosine similarity using native `sqlite-vector` (AVX2-accelerated C extension) combined with SQLite FTS5 BM25 and Regex.
- **Secret Masking**: Known credential patterns are masked before embedding or retrieval. This is not a guarantee that arbitrary personal data or every secret format is removed.

---

## MCP Tools Reference

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
| **Search & Analytics** | `sync_session` | Incremental evidence sync across Codex, Grok Bot, OMP, and AGY transcripts. |
| | `search_history` | Hybrid retrieval (Dense vector + BM25 + Regex) over history. |
| | `get_chunk` | Retrieve raw evidence chunk with automatic secret masking. |
| | `analyze_activity` | SQLite time-series analysis: retry gaps, error bursts, stalled items. |
| **Hooks** | `codex_session_hook` | Adapter for Codex lifecycle events and local audit context. |
| **Approved Workflows** | `workflow_status` | Current state, requirements, versions, and observed verification records. |
| | `workflow_run_checks` | Execute owner-approved test commands and kernel-check approved Lean statements. |
| | `workflow_classify` | Request a versioned JEV Choice using approved synthetic context. |
| | `workflow_transition` | Evaluate the current classification through the compiled Lean transition function. |
| | `workflow_authorize` | Issue a single-use, action/version/state-bound execution grant. |
| | `workflow_execute` | Recheck and consume the grant, run the exact argv, then record success or failure. |

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
TSUKKOMI_TYPESAFE_KEY_FILE=<authorized external dotenv containing TYPESAFE_API_KEY>
```

Keep approval, checker, hook configuration, Lean toolchain/cache, and evidence storage outside the agent's write authority. Checksums and file modes do not protect against an unrestricted process running as the same OS user. Use the host's sandbox/read-only mounts or a separate service account. Approval and workflow state must be separate directories.

### Verification and execution

Run `workflow_run_checks` to observe real subprocess exit codes and independently checked Lean proofs. Tests run against a fresh, readonly materialized snapshot of approved verification files and current tracked implementation—not the live project with untracked `conftest.py`, `sitecustomize.py`, or generated caches. Source-glob selection cannot silently authorize mutable harness/configuration files. The sandbox has a private writable temporary directory, no network or host credentials/state, and an allowlisted environment. Tests must write temporary artifacts there rather than modify the readonly snapshot. The runner does not accept model-supplied pass flags, replacement commands, or expected results. Source versions are checked before and after verification; interruption leaves nonpassing evidence.

Candidate Lean compilation is sandboxed. `leanchecker` replays the candidate against hash-pinned, readonly core imports; a source-pinned native auditor separately replays candidate declarations and checks the exact approved type and transitive axioms without loading candidate elaborator extensions. Unchanged trusted core libraries are not reproved for every request. Candidate stdout is not an audit report. Approved statements use trusted core syntax and fully qualified names, not candidate-defined notation or typeclass registration. Native `.olean` deserialization still assumes structurally valid artifacts: this is not the stronger comparator-plus-independent-external-checker guarantee described in [Lean's proof-validation guidance](https://lean-lang.org/doc/reference/latest/ValidatingProofs/).

`workflow_classify` calls the documented TypeSafe Choice endpoint with pinned `jev-1.13.0`. The fixed mapping is `0=InsufficientEvidence`, `1=NeedsRevision`, `2=ReadyForVerification`, `3=ReadyForCompletion`; no Score rounding is used. The input hash binds the actual request, rubric, mapping, and model. Approved requirement origins, actual bounded code diffs, state, and verification summaries are the only context sent. Oversized or non-text changes are not silently truncated.

Remote classification is disabled unless the owner explicitly sets `external_context: "synthetic"`. This release's experiments are synthetic; do not label private production data synthetic to bypass consent. `TYPESAFE_API_KEY` may alternatively be supplied in the process environment. A missing key, timeout, invalid distribution, unknown class/model, or low confidence is a nonauthorizing result. Other providers' keys are never substituted. The example confidence threshold is an operator setting, **not** an empirically validated accuracy guarantee.

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

The benchmark compares 640 finite stage/class/evidence combinations with an independent reference, measures false allowances/blocks and per-stage latency, and exercises correction of each missing mandatory fact. It is an adversarial **policy** evaluation, not live JEV accuracy or a whole-host coverage claim. Live JEV validation requires a TypeSafe key; offline transport fixtures never count as live service evidence.


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
NeoGraph extraction also masks known secret patterns in session events before sending a prompt to OpenRouter; keep the original session files and local index private.

Keep machine-specific values in local OMP configuration or a private extension copy, not this repository. Start a **new persisted OMP session**; `--no-session` cannot provide history. Register only user-stated, machine-checkable contracts with `upsert_contracts` (`provider: "omp"` and the actual OMP session ID for session scope). The extension checks pending model-issued tools through local MCP before execution, blocks confirmed `violation`/`suspicious`, and re-audits after results and turns. `unknown` and transport failure are reported but do **not** block; zero applicable contracts is not `clean`. The OMP JSONL parser treats unrecognized events as incomplete coverage. Direct shell/host actions outside OMP's tool pipeline, skipped hooks, in-memory sessions, and a result not yet flushed to disk are not covered.

Known OMP presentation changes, compaction records, agent notifications, empty messages, and UI metadata are indexed without treating them as assistant tool actions. Execution-start markers are matched against earlier indexed tool calls even when the session is synced between the call and marker. Unknown event shapes and unmatched markers still make coverage incomplete; restart the OMP MCP process after a parser upgrade so the parser version triggers a full local reindex.

The extension also registers the read-only `tsukkomi_session_context` tool. Call it before creating a session-scoped contract or checklist item: its `session_id` comes from the active OMP session manager. Never infer a session ID from `PI_SESSION_FILE` or a shell variable, which can belong to a parent OMP process. If no persisted session is available, do not silently create a global contract. `AGENTS.md` instructions alone are not audit contracts: register the user's explicit, machine-checkable prohibitions; broader subjective mistakes are not automatically judged by an LLM.

---

## Testing and verification

The suite covers parsers, host receipt linkage, restricted reads, temporal obligations, workflow version/grant invalidation, proof rejection, path isolation, and MCP transport. Lean integration tests report a skip when the native toolchain is unavailable; install it to exercise the new workflow layer.

```bash
pytest -q
python scripts/benchmark.py
```

### Verification scope

Test results and staged secret-pattern scans are bounded evidence, not certification that arbitrary secrets or every host bypass are absent. Inspect matched staged lines before committing, keep credentials in an external authorized dotenv file, and never commit raw personal transcripts or local runtime state.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
