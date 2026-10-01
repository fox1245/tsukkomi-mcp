# Antigravity (AGY) Integration & Lifecycle Hooks

This document details how to integrate `self-directing-mcp` with Google Antigravity (`agy`), enable automatic lifecycle guardrails via `hooks.json`, and enforce self-auditing behavior in agent prompts.

---

## 1. Overview

While OpenAI Codex uses internal MCP tool hooks (`codex_session_hook`), Antigravity (`agy`) supports **stdio-based lifecycle hooks** (`hooks.json`) where shell commands receive structured JSON on `stdin` and return execution decisions on `stdout`.

The adapter uses provider `agy` and the host's `conversationId` and `transcriptPath`
for local evidence. It never searches Codex history for an AGY conversation.
- `PreToolUse` checks all tools with applicable contracts, including restricted reads and `multi_replace_file_content`.
- `PostToolUse` refreshes local history; hook payloads are not substituted for transcript evidence.
- `Stop` checks contracts and checklist evidence on normal fully-idle `model_stop` / `NO_TOOL_CALL` only.
- `PreInvocation` injects active contracts, pending checklist items, and stored graph context.

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

Use the complete [example JSON](agy-hooks.example.json) in `.agents/hooks.json`
or `~/.gemini/config/hooks.json`. Replace the example Python executable and checkout
path with absolute paths to the installed environment; quote paths with spaces.
Preserve unrelated hooks. The identical event set used by the setup skill is
`PreToolUse`, `PostToolUse`, `PreInvocation`, and `Stop`. Both tool matchers MUST
remain `"*"`: reads can be restricted, and tool names/command first words do not
prove absence of side effects. Check the host's loaded hooks with `/hooks`.

### Enforcement and failures

The default `SELF_DIRECT_AGY_ENFORCEMENT=enforced` maps:

| Event / outcome | Response |
| --- | --- |
| PreToolUse, confirmed violation | `deny` |
| PreToolUse, exception requiring confirmation (`suspicious`) | `force_ask` (ignores Always Allow) |
| PreToolUse, enforced unknown / unavailable verification | `deny` |
| PreToolUse, clean | `allow`, limited to inspected contracts/history |
| PreToolUse, no applicable action contracts | `allow`, explicitly not a compliance verdict |
| Stop, normal fully-idle model stop with pending/unknown/failed obligations | `continue` |
| Stop, cancellation, error, step limit, or active background work | `allow`; never restart cancellation |
| PreInvocation failure | Ephemeral warning, not a permission decision |
| PostToolUse | `{}`; failures are diagnosed on stderr, never fabricated successful results |

Explicit `SELF_DIRECT_AGY_ENFORCEMENT=advisory` allows actions/stops while reporting
non-clean findings as advisory, not compliance. Invalid input, broken imports,
native runtime failures, and corrupt configuration are not converted to permission:
before policy can be loaded, PreToolUse fails closed even if advisory was intended.
Malformed Stop metadata does not restart the host, since it cannot establish a
normal model stop. Unsupported events are rejected. Always register the event
explicitly (`--event=PreToolUse`, etc.); tool payload shape cannot distinguish pre/post.

Temporal `must` rules use `regex` for the required predecessor and `before_regex`
for the gated action. A `must` without `before_regex` is a Stop/checkpoint obligation,
not a barrier preventing the test needed to satisfy it. Action applicability is
computed from the actual rule, role, scope, tool, and arguments—not a command-verb
allowlist. An exception cannot waive an independently unknown prerequisite.

### Transcript roots and evidence

Only `<app_data_dir>/brain/<conversation-UUID>/.system_generated/logs/transcript.jsonl`
and `transcript_full.jsonl` in that same directory are accepted. The official docs
name the former; actual CLI 1.2.9 hook payloads name the latter. Defaults are
`~/.gemini/antigravity`, `~/.gemini/antigravity-cli`, and `~/.gemini/antigravity-ide`.
For isolated installs set `SELF_DIRECT_AGY_APP_DATA_DIRS`
to a JSON array of absolute app-data directories, not `brain` directories.
Traversal, symlink redirection/escape, mismatching UUIDs, and ambiguous discovery
are rejected. A valid path may precede the first host flush; missing evidence
remains unknown, never successful. Unknown records and partial final JSONL records
keep coverage incomplete. Success requires linked observed result evidence, not
assistant prose or a requested tool call. Keep private histories and keys outside
the repository and use a private index.

The parser is grounded in isolated synthetic Antigravity CLI 1.2.9 runs:
`USER_INPUT`, `PLANNER_RESPONSE`, `SYSTEM_SDK/EPHEMERAL_MESSAGE`, and `GENERIC`
command results with an explicit runtime exit-code prefix. Display arguments
are JSON-encoded strings; full-log arguments are ordinary JSON values.
Both log formats omit tool call IDs and result references. Therefore the hook
stores separate host receipts in `SELF_DIRECT_INDEX_DIR/agy-receipts.sqlite`.
The native transcript is never modified or replaced with hook payloads.

Pre/Post receipts bind the conversation UUID, actual execution `stepIdx`, exact
canonical tool name/arguments, transcript path, and hash of complete native
records preceding that execution step. The first completed native result is also
hash-bound immutably, including when its persistence follows PostToolUse. Rewriting
that result cannot reuse an old successful receipt. Paired receipts join only the
native result with the exact `step_index`; no adjacency inference is used.
Success requires both a matching Post receipt without an error and an explicit
successful native command exit (`Output:` and observed `Stdout:` layouts).
Missing, mismatched, stale, duplicate, or unlinked receipts cannot establish
success. Receipt revisions trigger local reparsing even if transcript bytes have
not changed. Installing hooks does not retroactively manufacture receipts.

A prefix-bound Pre receipt plus the native structured hook-denial error is retained
as nonexecution metadata. It is not a successful tool result, and does not poison
later recovery. Denial-looking output without that host observation remains unknown.
Checklist completion IDs must resolve to current, same-provider/session, linked
successful tool results; a supplied `verified: true` or invented ID is not evidence.
This validates execution provenance, not arbitrary natural-language satisfaction;
use owner-approved workflows for requirement-specific test/model binding.

Use a **host-controlled private index directory outside agent write authority**
for enforced deployments, and protect hook code, contracts, and configuration
likewise. Same-OS-user unrestricted file access is not a security boundary.
Receipts are not accepted through an MCP/model-facing API.

CLI 1.2.9 emits `NO_TOOL_CALL` for normal Stop; the documented `model_stop` is
also accepted. All other termination reasons pause/stop without restarting.
Other native record formats remain unsupported until observed and validated.

For manual MCP calls pass `provider: "agy"`, the real `session_id`, and `path`
on sync/check/audit operations. Codex remains the shared API default.

### Host verification checklist

With isolated synthetic workspace/history, verify forbidden commands and restricted
reads are blocked, a failed prerequisite remains blocked, a successful prerequisite
unlocks its dependent action, unverified checklist work continues a normal Stop,
cancel still stops, and PreInvocation context appears. Verify `force_ask` with a
cached Always Allow grant. Offline tests do not establish host registration,
cached-grant behavior, or UI behavior. Never claim these without observing them.
The official contract is [Google's hooks documentation](https://antigravity.google/docs/hooks).

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
