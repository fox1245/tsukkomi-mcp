---
name: self-directing-mcp-setup
description: Complete guide and workflow for installing, configuring, and verifying self-directing-mcp in Google Antigravity (agy) and agentic environments. Covers Python 3.12+ virtualenv setup via uv, sqlite-vector native AVX2 extension provisioning, OpenRouter API key management, and MCP client registration.
---

# Self-Directing MCP Setup Guide

This skill provides step-by-step instructions to install, configure, verify, and operate `self-directing-mcp` within Google Antigravity (`agy`), Codex, and compatible MCP environments.

## Overview

`self-directing-mcp` is an MCP server designed to audit and constrain agent behavior against explicit rules, obligations, and session histories. It provides 16 specialized tools:
- **Contract & Action Auditing**: `check_action`, `audit_session`, `upsert_contracts`, `list_contracts`, `revoke_contract`
- **Checklists & Activity Tracking**: `update_checklist`, `get_checklist`, `analyze_activity`
- **Session History & Retrieval**: `search_history`, `get_chunk`, `sync_session`, `audit_status`
- **GraphRAG & Knowledge Topology**: `run_neograph_update`, `propose_graph_update`, `commit_graph_update`

---

## Prerequisites

- **OS**: Linux (x86_64, glibc), macOS, or Windows
- **Python**: 3.12+ (Python 3.12.3+ recommended)
- **Package Manager**: `uv` (fast, reliable dependency management)
- **SQLite**: `sqlite3` CLI (3.50+), `sqlite-utils`, and SQLite extension loading enabled
- **API Key**: `OPENROUTER_API_KEY` (optional for local regex audit; required for dense semantic retrieval and GraphRAG)

---

## Step-by-Step Installation

### 1. Repository Placement
Clone or place the repository into the standard MCP servers directory:
```bash
mkdir -p ~/.mcp-servers
# If cloning fresh:
gh repo clone fox1245/self-directing-mcp ~/.mcp-servers/self-directing-mcp
cd ~/.mcp-servers/self-directing-mcp
```

### 2. Virtual Environment Setup with `uv`
Create a clean Python 3.12 virtual environment and install dependencies:
```bash
uv venv --python 3.12 .venv
uv pip install -e . --python .venv/bin/python
```

Key installed packages:
- `neograph-engine==0.12.1` (C++ graph agent execution engine)
- `mcp>=1.0,<2` (Model Context Protocol SDK / FastMCP)
- `pydantic>=2.0` & `pydantic-settings`
- `numpy>=1.26`
- `httpx` & `python-dotenv`

### 3. Native `sqlite-vector` Extension Provisioning
Dense vector search requires the official `sqliteai/sqlite-vector` v1.1.0 release library. Download and verify the AVX2 checksum-pinned binary:
```bash
.venv/bin/python scripts/setup_sqlite_vector.py --output-dir native
```
This produces:
- `native/vector.so` (Linux) / `vector.dll` (Windows) / `vector.dylib` (macOS)
- `native/sqlite-vector-installation.json` (metadata verification record)

Smoke test verification:
```bash
sqlite3 :memory: ".load native/vector.so" "SELECT vector_version(), vector_backend();"
# Expected output: 1.1.0 | AVX2
```

### 4. SQLite CLI Tools Installation
Install official SQLite CLI utilities and `sqlite-utils` for database inspection:
```bash
# Install sqlite-utils CLI
uv tool install sqlite-utils

# Download and install official sqlite3 binaries into ~/.local/bin if not present:
python3 -c "
import urllib.request, zipfile, io, os, hashlib
url = 'https://www.sqlite.org/2026/sqlite-tools-linux-x64-3530400.zip'
data = urllib.request.urlopen(url).read()
assert hashlib.sha3_256(data).hexdigest() == '6eeb57e8f2aef7687f9f016a980992cf2799c8c07a87c5e21495530f91915047'
dest = os.path.expanduser('~/.local/bin')
with zipfile.ZipFile(io.BytesIO(data)) as z:
    for item in ('sqlite3', 'sqldiff', 'sqlite3_analyzer'):
        target = os.path.join(dest, item)
        with open(target, 'wb') as f:
            f.write(z.read(f'sqlite-tools-linux-x64-3530400/{item}'))
        os.chmod(target, 0o755)
"
```

### 5. Environment & API Key Configuration
Create a `.env` file in the project root:
```env
# OpenRouter API Key (sk-or-v1-...)
OPENROUTER_API_KEY=

SELF_DIRECT_USE_FAKE_EMBEDDER=false
SELF_DIRECT_DENSE_BACKEND=sqlite-vector
# Note: SELF_DIRECT_SQLITE_VECTOR_PATH, SELF_DIRECT_INDEX_DIR, and OPENROUTER_API_KEY_FILE
# automatically resolve dynamically relative to the repository via pathlib.
```
> [!NOTE]
> If `OPENROUTER_API_KEY` is not provided, set `SELF_DIRECT_USE_FAKE_EMBEDDER=true` to enable offline bag-of-tokens embedding and local regex contracts.

### 6. Antigravity (`agy`) MCP Registration
Register `self-directing-mcp` in `~/.gemini/config/mcp_config.json`:

```json
{
  "mcpServers": {
    "self-directing-mcp": {
      "command": "~/.mcp-servers/self-directing-mcp/.venv/bin/python",
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

---

## Verification & Health Check

### 1. Test Suite Execution
```bash
cd ~/.mcp-servers/self-directing-mcp
uv pip install pytest --python .venv/bin/python
.venv/bin/pytest tests -q
# Expected: 172 passed, 2 skipped
```

### 2. MCP stdio Handshake Test
```bash
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}}}' | .venv/bin/python -m self_directing_mcp
```
The server will respond with JSON capabilities containing `self-directing-mcp` server info.

### 3. Engine Readiness Test
```bash
.venv/bin/python -c "
from self_directing_mcp.engine import SelfDirectEngine
engine = SelfDirectEngine()
engine.ensure_ready()
print('Engine ready:', type(engine.embedder).__name__, '| Dense:', engine.dense_backend)
"
```

---

## Usage Patterns in Antigravity (`agy`)

### Registering Constraints (`upsert_contracts`)
Define strict prohibitions and required preconditions:
```json
{
  "contracts": [
    {
      "id": "prevent-rm-rf",
      "provider": "agy",
      "type": "must_not",
      "scope": "tool_call",
      "regex": "rm\\s+-rf\\s+[/~]",
      "description": "Prevent dangerous recursive deletion of root or home"
    },
    {
      "id": "tests-before-deploy",
      "provider": "agy",
      "type": "must",
      "scope": "tool_call",
      "regex": "\\bpytest\\b",
      "before_regex": "\\bdeploy\\b",
      "requires_success": true,
      "description": "Require successful pytest before deploy"
    }
  ]
}
```

### Pre-execution Checking (`check_action`)
Before executing high-risk shell commands or tool steps, verify the proposed action:
```json
{
  "session_id": "11111111-1111-4111-8111-111111111111",
  "provider": "agy",
  "path": "/absolute/app-data/brain/11111111-1111-4111-8111-111111111111/.system_generated/logs/transcript.jsonl",
  "action": {
    "tool_name": "run_command",
    "arguments": {"CommandLine": "rm -rf /"}
  }
}
```
If the verdict is `violation` or `suspicious`, **immediately abort the tool call** and report to the user.

### Lifecycle Hooks Automation (`.agents/hooks.json`)

Use the complete `docs/agy-hooks.example.json`, replacing its executable and
checkout paths with absolute installed paths. Preserve unrelated hooks.
Register exactly `PreToolUse`, `PostToolUse`, `PreInvocation`, and `Stop`, with
`matcher: "*"` for both tool events. This covers restricted reads and
`multi_replace_file_content`; never reinstate command-first-word or internal-name
exemptions. Pass the event explicitly as `--event=EVENT`.

- `PreToolUse`: applicable contracts gate every tool; violation/unknown -> deny,
  suspicious -> force_ask (ignores Always Allow), clean -> allow. No contracts
  permits independent actions without asserting compliance.
- `PostToolUse`: local transcript refresh only; returns `{}`.
- `PreInvocation`: provider-scoped contract/checklist/graph context injection.
- `Stop`: pending, failed, or unknown obligations -> continue only for fully-idle
  model_stop or CLI 1.2.9 NO_TOOL_CALL. Cancellation, error, and background work never restart.

All AGY sync/check/audit/context calls must select `provider: "agy"` and use the
host's actual UUID and transcriptPath. Official transcript roots are
`~/.gemini/{antigravity,antigravity-cli,antigravity-ide}/brain/<UUID>/.system_generated/logs/transcript.jsonl`.
CLI 1.2.9 passes `transcript_full.jsonl` in that same directory; both are accepted.
Native logs omit call IDs. Separate Pre/Post host receipts bind exact step IDs,
tool identity, transcript path, and prior-history hash to explicit native command
exit evidence. Keep the receipt/index directory, hooks, and contracts outside
agent write authority; same-user unrestricted access is not tamper isolation.
`SELF_DIRECT_AGY_APP_DATA_DIRS` accepts a JSON array of app-data roots for isolated
installs. `SELF_DIRECT_AGY_ENFORCEMENT=advisory` is an explicit non-enforcing mode,
not evidence of compliance. Default enforced mode does not auto-allow missing
history, broken JSON, or unavailable native verification.

Follow `docs/agy-installation.md` for the full policy and host verification
checklist. Run synthetic forbidden/read checks, failed and successful prerequisite
transitions, checklist Stop, context injection, and cancellation in the actual
host; verify force_ask with cached permissions. Do not claim actual-host coverage
from offline tests. Keep keys, private histories, and indexes out of commits.

