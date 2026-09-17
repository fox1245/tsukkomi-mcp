# Client integration

Run `python scripts/install_codex.py --trust-hooks` for complete integration. It installs the MCP, global guidance and three advisory lifecycle hooks. Registering the MCP alone does not install those hooks.

1. Install the package and register its absolute venv Python executable using [AGENTS.md](AGENTS.md).
2. Register only explicit user-added/changed/revoked constraints using the real provider/session ID. Questions do not require registration or a separate sync.
3. For actions affected by active constraints, use check_action with the intended tool and arguments. A fresh matching hook check avoids a second manual check.
4. At meaningful completion checkpoints, call audit_session when obligations need verification; both audit tools refresh local JSONL without remote embeddings. No-contract hooks are quiet and not a clean verdict.
5. On violation/suspicious, stop and report. On unknown, inspect coverage and missing evidence before claiming compliance. Other independent work can proceed under the user's instructions.
6. Recheck when the action, history, or contracts change. Findings describe the returned snapshot and are not execution authorization.

For Cursor/Grok Bot, set SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR and pass provider="grokbot". The same workflow applies.

See [README.md](README.md) for temporal contracts, exact verdict semantics, migration, and limitations.

See [Codex installation](docs/codex-installation.md) for repeatable installation, manual hook JSON and trust verification.

