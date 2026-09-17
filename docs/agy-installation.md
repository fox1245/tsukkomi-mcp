# Antigravity (AGY) Integration & Lifecycle Hooks

This document details how to integrate `self-directing-mcp` with Google Antigravity (`agy`), enable automatic lifecycle guardrails via `hooks.json`, and enforce self-auditing behavior in agent prompts.

---

## 1. Overview

While OpenAI Codex uses internal MCP tool hooks (`codex_session_hook`), Antigravity (`agy`) supports **stdio-based lifecycle hooks** (`hooks.json`) where shell commands receive structured JSON on `stdin` and return execution decisions on `stdout`.

By pairing `self-directing-mcp` with AGY's lifecycle hooks:
- **`PreToolUse`** automatically runs `check_action` on high-risk tools (`run_command`, `write_to_file`, `replace_file_content`). If a contract is violated, the action is **hard-blocked** (`{"decision": "deny"}`) before reaching the system shell.
- **`Stop`** prevents the agent from prematurely claiming victory if required checklist items lack verifiable evidence (`{"decision": "continue"}`).
- **`PreInvocation`** keeps active invariants visible without bloating prompt context.

---

## 2. MCP Server Configuration

Register the server in `~/.gemini/config/mcp_config.json`:

```json
{
  "mcpServers": {
    "self-directing-mcp": {
      "command": "python",
      "args": [
        "-m",
        "self_directing_mcp"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```
*(Note: `SELF_DIRECT_INDEX_DIR`, `SELF_DIRECT_SQLITE_VECTOR_PATH`, and `SELF_DIRECT_OPENROUTER_API_KEY_FILE` automatically resolve relative to the repository via `pathlib.Path` defaults, but can be explicitly overridden in `env` if needed).*

---

## 3. Lifecycle Hooks Configuration (`hooks.json`)

Place `hooks.json` in your workspace customization directory (`.agents/hooks.json`):

```json
{
  "self-directing-guard": {
    "enabled": true,
    "PreToolUse": [
      {
        "matcher": "run_command|write_to_file|replace_file_content",
        "hooks": [
          {
            "type": "command",
            "command": "~/.mcp-servers/self-directing-mcp/scripts/agy_hook.py --event PreToolUse",
            "timeout": 10
          }
        ]
      }
    ],
    "Stop": [
      {
        "type": "command",
        "command": "~/.mcp-servers/self-directing-mcp/scripts/agy_hook.py --event Stop",
        "timeout": 10
      }
    ]
  }
}
```

### Hook Actions:
- **`PreToolUse`**:
  - `violation` -> `{"decision": "deny", "reason": "..."}`: Hard aborts execution.
  - `suspicious` -> `{"decision": "ask", "reason": "..."}`: Prompts user for confirmation.
  - `clean` / read-only -> `{"decision": "allow"}`: Proceeds immediately.
- **`Stop`**:
  - Pending/unverified checklist items -> `{"decision": "continue", "reason": "..."}`: Keeps agent in loop.
  - All verified -> `{}`: Allows turn completion.

---

## 4. Agent Rules Enforcement (`AGENTS.md`)

Add the following to your workspace `AGENTS.md` (or `GEMINI.md`) to guide Gemini's attention:

```markdown
## Self-Directing MCP Invariants

1. Absolute Contract Authority:
   - Always honor verdicts returned by `check_action` and `audit_session`.
   - Never ignore a `violation` or `suspicious` decision. Stop the affected action immediately.
   - `unknown` represents missing evidence, not permission.

2. Verifiable Milestones:
   - Track user requirements with `update_checklist`.
   - Never mark a requirement as completed without citing verified evidence (`evidence_chunk_ids`).
```
