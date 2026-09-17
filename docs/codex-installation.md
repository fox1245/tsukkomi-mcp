# Codex installation and lifecycle hooks

Tested with Codex CLI 0.153.4 and Python 3.13 on Windows. Python 3.12+ is required. Registration relies on the installed Codex app-server accepting MCP tool hooks and exposing their trust metadata. An older client may need an update; the verifier must not claim success when that capability is missing.

## Complete installation

Clone the repository or open an existing checkout, read AGENTS.md and the installer/hook code, then run:

```bash
python scripts/install_codex.py --trust-hooks
python scripts/install_codex.py --verify
```

Use python3 or py -3.13 where appropriate. The script prefers uv when installed and otherwise uses venv/pip. It installs a regular package in a managed environment, while its configured credential file and native library must remain at their registered absolute paths.

Options:

| Option | Behavior |
|---|---|
| --codex-home PATH | Select a non-default Codex home |
| --codex PATH | Use a specific Codex executable |
| --trust-hooks | Trust only this package's three exact reviewed hook definitions |
| --verify | Check existing installation without reinstalling |
| --dry-run | Show intended paths without writing files |
| --env-file PATH | Authorized dotenv key source (default: checkout .env) |
| --vector-archive PATH | Previously downloaded official vector ZIP; SHA-256 verified |
| --offline | Use cached Python packages and an already installed DLL or --vector-archive |

Without --trust-hooks, hooks can be registered but remain pending review. Inspect them with /hooks in Codex. Do not use a global bypass flag to make verification pass.

## Registration targets

The installer changes only its MCP entry, the hooks feature flag, one guidance block and its three hook definitions/trust entries. Existing hook positions are preserved. Files are backed up under CODEX_HOME/backups/self-directing-mcp-<time>-<id> before registration.

Guidance is merged into the effective global file: AGENTS.md, or an existing nonempty AGENTS.override.md. Other guidance is retained. The configuration writer is Codex's local app-server API, preserving unrelated TOML entries and comments. Hooks use hooks.json; hooks.state in config.toml holds their trust records.

Default installation requires a real API key in the authorized .env file, fake=false, and the official sqlite-vector native library. Set --env-file to reuse a different authorized key file. Hooks/audits still refresh locally without embedding; remote re-embedding requires user consent. New contracts are empty until the agent registers actual user requirements.

## Manual MCP registration

Use the actual installed environment path:

```toml
[mcp_servers.self-directing-mcp]
command = "/absolute/codex-home/mcp-servers/self-directing-mcp/venv/bin/python"
args = ["-m", "self_directing_mcp"]
enabled = true
startup_timeout_sec = 30
tool_timeout_sec = 60

[mcp_servers.self-directing-mcp.env]
SELF_DIRECT_INDEX_DIR = "/absolute/codex-home/mcp-servers/self-directing-mcp/index"
SELF_DIRECT_CODEX_SESSIONS_DIR = "/absolute/codex-home/sessions"
SELF_DIRECT_USE_FAKE_EMBEDDER = "false"
SELF_DIRECT_DENSE_BACKEND = "sqlite-vector"
SELF_DIRECT_SQLITE_VECTOR_PATH = "/absolute/path/to/native/vector.so"
SELF_DIRECT_OPENROUTER_API_KEY_FILE = "/absolute/path/to/authorized/.env"
SELF_DIRECT_SEED_EXAMPLE_CONTRACTS = "false"

[features]
hooks = true
```

Windows uses venv\Scripts\python.exe. Merge into existing tables; do not duplicate an existing [features] table. The normal installer handles this.

## Manual hook registration

Merge [codex-hooks.example.json](codex-hooks.example.json) into CODEX_HOME/hooks.json. Keep existing entries and their order. The handlers call self-directing-mcp.codex_session_hook and receive structured values from Codex:

- UserPromptSubmit passes the actual prompt and supplies session information only when an explicit constraint signal needs registration. Ordinary questions stay quiet.
- PreToolUse passes the pending tool name/input for a prospective check.
- Stop verifies changed work and remaining obligations, deduplicating unchanged checkpoints without continuing or terminating the turn. A no-contract hook skips index initialization and does not claim compliance.

Read the resulting definitions, then trust them with /hooks. The installer can perform that narrow trust step after an explicit --trust-hooks request: it uses each currentHash returned by hooks/list and writes only that hook's trusted_hash. It never invents a hash or trusts other handlers.

Missing MCP connections do not cause the hook system to reconnect the server. Some hosted/specialized tools do not emit PreToolUse. Manual check_action/audit_session remains available. Hook output is advisory and does not become a new authorization.

## Manual AGENTS guidance

Copy the section under "Guidance installed for everyday use" from the repository AGENTS.md into your effective global AGENTS file, or run the installer to maintain its marker-delimited block.

The guidance asks the agent to register the user's actual scoped requirements, inspect planned actions, retrieve source evidence and distinguish unknown from clean. It also discourages redundant checks when a fresh hook result already covers the action. Reading AGENTS.md describes the procedure; an agent or a human must actually run the installer to apply it.

## Verification and recovery

The verification command checks:

1. The configured executable exists and the server is enabled.
2. The effective global guidance and three hook definitions match the installed version.
3. Codex loads the three handlers and reports each trust state.
4. The configured stdio server initializes, exposes the required tools, and returns a valid hook output shape.

"registered" and "trusted" are separate fields. Start a new Codex session/restart to activate changes in clients that already started. The installer does not interrupt running work.

On failure, inspect the reported step and the timestamped backup. Restore only files that need recovery after comparing against newer edits. The installer refuses concurrent file replacement and malformed/duplicate owned markers instead of deleting unrelated settings. It never modifies auth.json.

For updates, rerun the same installation command. New hook definitions require review/trust again. Global AGENTS or project instructions can override one another; report which effective file was changed rather than assuming an ignored file will load.

## Sources

Runtime hook behavior and review requirements follow the [official hooks documentation](https://learn.chatgpt.com/docs/hooks). MCP registration follows the [official MCP guide](https://learn.chatgpt.com/docs/extend/mcp?surface=cli). Effective guidance selection follows the [official AGENTS.md guide](https://learn.chatgpt.com/docs/agent-configuration/agents-md). The app-server request schemas were generated from the locally installed Codex CLI.
