## Summary

On very large sessions (a single JSONL rollout here exceeded 100 MB with ~26k chunks), `audit_session`, `check_action`, and `sync_session` calls are failing with a fixed 60-second timeout (`timed out awaiting tools/call after 60s`). The hooks populate the contract and index layers, but the audit never reaches a verdict, leaving coverage `unknown` instead of `clean`. The agent has no way to tell the MCP to wait longer.

## Observed behavior

- `audit_status` reports `coverage_complete: false` and `issues: ["unread_bytes", "unsupported_events"]`.
- `audit_session` / `check_action` / `upsert_contracts` / `sync_session` intermittently fail with a 60-second client timeout against this large session, while small sessions complete immediately.
- This makes it impossible to get a verified audit verdict for the very sessions that most need it.

## Request

Allow the agent (the MCP client) to specify a timeout on the long-running tools so it can accommodate large sessions instead of being hard-capped at 60s.

Suggested options (pick what fits best):
- Accept an optional `timeout_ms` argument on `sync_session`, `audit_session`, `check_action`, and `search_history`.
- Alternatively, add a config/session-scoped `default_audit_timeout_ms`.
- Support incremental/long-poll style behavior so a single bounded call can return as soon as the incremental index is ready, mirroring the existing incremental audit work.

## Why it matters

- The audit hooks are a REQUIRED gate before risky actions; on large sessions they currently cannot finish, so compliance stays `unknown` rather than verified.
- Agents should be able to trade latency for completeness on the tools they actually need to wait on.

## Acceptance criteria

- A client can pass an explicit timeout larger than 60s and receive a `clean`/`violation`/`suspicious`/`unknown` verdict on a large session.
- Default behavior (no argument) remains backward compatible (current timeout).
- `sync_session(embed=False)` still keeps session text local when no remote embedding is authorized.

## Context

Encountered while using Self-directing MCP as the audit hook for the AgentX/NeoGraph session. Server logs and index remain intact; only the verification verdict is blocked by the fixed timeout.
