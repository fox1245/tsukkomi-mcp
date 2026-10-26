# Antigravity (AGY) Integration & Lifecycle Hooks

This document details how to integrate `self-directing-mcp` with Google Antigravity (`agy`), enable automatic lifecycle guardrails via `hooks.json`, and enforce self-auditing behavior in agent prompts.

---

## 1. Overview

While OpenAI Codex uses internal MCP tool hooks (`codex_session_hook`), Antigravity (`agy`) supports **stdio-based lifecycle hooks** (`hooks.json`) where shell commands receive structured JSON on `stdin` and return execution decisions on `stdout`.

The adapter uses provider `agy` and the host's `conversationId` and `transcriptPath`
for local evidence. It never searches Codex history for an AGY conversation.
- `PreToolUse` checks all tools with applicable contracts, including restricted reads and `multi_replace_file_content`.
- `PostToolUse` refreshes local history; hook payloads are not substituted for transcript evidence.
- `Stop` checks contracts and checklist evidence on normal fully-idle `model_stop` / `NO_TOOL_CALL` only. It continues the agent only for a confirmed, fully observed miss of an `auto_continue` contract, once per conversation.
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
The `python` in the entry must be the interpreter of an environment where this package is installed. Its dependencies include the exact `neograph-engine==0.14.1`; a missing native NeoGraph runtime is an error, not a fallback.

Configure paths and credentials with explicit `SELF_DIRECT_*` variables in the server entry's `env`, or with `~/.config/tsukkomi-mcp/config.toml` and a `[paths]` table (set `SELF_DIRECT_CONFIG_FILE` in the server environment to select another file). A `.env` is read only from the server process's working directory, which the host chooses, so do not rely on it. See [config.example.toml](../config.example.toml) and the [path source contract](../README.md#user-path-configuration-toml). Relative TOML paths use the configuration file's directory; explicit arguments/environment override them.

Lifecycle hooks do not read a `.env` at all. Hosts run them inside the project being edited, and that file is content the audited agent can write, so the hook process builds its settings from its own environment and the user TOML only (`settings_core.load_hook_settings()`, a standard-library twin of `config.hook_settings()` that avoids importing pydantic-settings; a parity test keeps them identical). Set the index directory, the contracts path and `SELF_DIRECT_AGY_ENFORCEMENT` for the hooks through the environment of the hook command or in the TOML, and keep them consistent with the MCP server that holds the contracts.

Without an override, the index is `~/.self_direct_index`, not a directory inside the installed package or checkout (older versions defaulted to one inside the checkout). Preserve an existing/managed index by selecting its absolute `index_dir` in TOML or `SELF_DIRECT_INDEX_DIR`. No stored evidence or rules are automatically migrated.

Select `sqlite_vector_path` and `openrouter_api_key_file` in TOML or their `SELF_DIRECT_` environment equivalents. Only the authorized key file's path belongs in TOML, not the key value. Missing or keyless selected files still fail closed. Keep the TOML and credentials outside writable workflow workspaces.

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

The default `SELF_DIRECT_AGY_ENFORCEMENT=confirmed_only` gates only confirmed findings. `advisory` never blocks, and `enforced` is the strict opt-in that also denies an unverifiable `PreToolUse`. The mode is independent of `SELF_DIRECT_CLAUDE_ENFORCEMENT`. The verdict-to-decision table is shared with the [Claude Code adapter](claude-installation.md#4-what-the-hooks-decide) (`src/self_directing_mcp/hook_policy.py`); only the host's way of expressing a decision differs.

| PreToolUse outcome | `advisory` | `confirmed_only` (default) | `enforced` |
| --- | --- | --- | --- |
| Confirmed `violation` | `allow`, `ADVISORY (not compliance)` reason | `deny` | `deny` |
| `suspicious` (an exception that needs confirmation) | `allow`, advisory reason | `force_ask` (ignores Always Allow) | `force_ask` |
| `unknown` or unavailable verification | `allow`, advisory reason | `allow` with a visible `WARNING - unverified, not a compliance verdict` reason | `deny` |
| `unknown`, and an applicable contract has `risk: "high"` | `allow`, advisory reason | `ask` | `deny` |
| `clean` | `allow` | `allow`, limited to the inspected contracts and history | `allow` |
| No applicable action contract | `allow`, reason says this is not a compliance verdict | same | same |

`suspicious` beside an `unknown` prerequisite stays `force_ask` in `confirmed_only` and becomes `deny` in `enforced`, because confirming an exception cannot waive another unverified prerequisite in strict mode. An `unknown` that is allowed in `confirmed_only` is also queued and shown once as an ephemeral message at the next `PreInvocation`, so the model sees it even though the host does not show an allow's reason to it. An owner-approved workflow adds one rule: a direct completion action that the workflow maps to its own executor is denied in `confirmed_only` and `enforced`. In `advisory` it is allowed with an advisory reason; advisory never gates workflows either.

| Other events | Response |
| --- | --- |
| `Stop`, a normal fully-idle `model_stop` / `NO_TOOL_CALL`, with an `auto_continue` obligation confirmed unmet on complete, current evidence | `continue`, once per conversation |
| `Stop`, an incomplete owner-approved workflow | `continue`, sharing the same once-per-conversation budget; `allow` in `advisory` |
| `Stop`, anything else on a normal stop (unknown evidence, an unmet obligation without `auto_continue`, pending checklist items, incomplete history, budget spent) | `allow`, with a note that completion was not verified |
| `Stop`, cancellation, error, step limit, or active background work | `allow`; never restart cancellation |
| `PreInvocation` | Ephemeral message with the health banner, queued warnings, workflow status, up to 8 contracts, up to 8 pending items and up to 8 graph relations (all marked as data, each on one line of its own: a control, invisible or bidirectional character in a stored description, item or id is shown as a backslash escape with its code point, so stored text cannot start a line that reads like another rule). When it fails: an ephemeral warning, not a permission decision |
| `PostToolUse` | `{}`; refreshes local history. Failures are diagnosed on stderr and queued as a warning, never fabricated as successful results |

The stop policy is the same in `confirmed_only` and `enforced`; `advisory` never continues.

**The `continue` budget.** At most one continuation per conversation, persisted in `<index_dir>/agy-hook-state.sqlite` so turns and process restarts never reset it. The same reason (contract ids and revisions) is never used twice, and an unreadable or unwritable counter means "do not continue". Unknown evidence, legacy defects (see below), pending checklist items, description-only contracts and hook errors never continue.

**Failures fail open.** Invalid input, a malformed or unsupported event, a broken import, a native runtime failure, an unreadable index and corrupt configuration produce a visible warning and never a block, in every mode. `PreToolUse` answers `allow` with the warning, except that when the mode is not `advisory` and an enabled `risk: "high"` contract applies to the call (read with the standard library, so it works without the package) the answer is `ask`. `Stop` answers `allow`, `PostToolUse` answers `{}`, and `PreInvocation` skips the injection and says so. A failed host-receipt write does not skip the check; the response notes the missing receipt and later success evidence for that call may stay unlinked. `scripts/agy_hook.py` has a standard-library fallback that does the same after an import failure. Malformed Stop metadata cannot establish a normal model stop, so it never restarts the host. Always register the event explicitly (`--event=PreToolUse`, etc.); tool payload shape cannot distinguish pre/post.

**`hook_health` and the banner.** Each outcome updates a table in `<index_dir>/agy-hook-state.sqlite` that stores error classes, counts and timestamps, never payload text. It separates `audit_unavailable` (nothing can be checked) from `check_failed` (one event's check failed). After three consecutive failures, `PreInvocation` adds a `Tsukkomi audit INACTIVE` or `Tsukkomi audit DEGRADED` banner. A working call ends the streak. Claude Code keeps its own state file, so the two hosts never share a budget or a banner.

Temporal `must` rules use `regex` for the required predecessor and `before_regex`
for the gated action. A `must` without `before_regex` is a Stop/checkpoint obligation,
not a barrier preventing the test needed to satisfy it. Action applicability is
computed from the actual rule, role, scope, tool, and arguments—not a command-verb
allowlist. An exception cannot waive an independently unknown prerequisite. The contract fields `risk`, `auto_continue` and `when` are described in the [README](../README.md#contract-fields-risk-auto_continue-and-when).

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
command results with an explicit runtime exit-code prefix. Parser version 7 also
reads the IDE's typed result steps (see [parser version 7](#parser-version-7-ide-result-steps-context-steps-and-host-padding)). Display arguments
are JSON-encoded strings; full-log arguments are ordinary JSON values.
Both log formats omit tool call IDs and result references. Therefore the hook
stores separate host receipts in `SELF_DIRECT_INDEX_DIR/agy-receipts.sqlite`.
The native transcript is never modified or replaced with hook payloads.
The receipts database is an authority record: copy it with SQLite's backup API, not as a bare file,
whenever `SELF_DIRECT_SQLITE_JOURNAL=auto` may have left commits in its `-wal` file (see
[SQLite journal mode, index files and backups](reference.md#sqlite-journal-mode-index-files-and-backups)).

Pre/Post receipts bind the conversation UUID, actual execution `stepIdx`, exact
canonical tool name/arguments, transcript path, and hash of complete native
records preceding that execution step. The first completed native result is also
hash-bound immutably, including when its persistence follows PostToolUse. Rewriting
that result cannot reuse an old successful receipt. Paired receipts join only the
native result with the exact `step_index`; no adjacency inference is used.
Success requires both a matching Post receipt without an error and an explicit
successful native command exit (`Output:` and observed `Stdout:` layouts).
Missing, mismatched, stale, duplicate, or unlinked receipts cannot establish
success. A receipt revision that can change an event that is already indexed triggers
local reparsing even if transcript bytes have not changed (see
[Hook cost](#hook-cost-what-a-hook-reads)); any other receipt is indexed with the
records that follow it. Installing hooks does not retroactively manufacture receipts.

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

### Hook cost: what a hook reads

A hook costs what was appended since the last one, not the size of the transcript. Three mechanisms
make that so, and each one falls back to the whole-file parse the hooks always did whenever it cannot prove itself.

1. **One pass, from a verified prefix.** A receipt hashes the file once and parses only the records after a prefix it
   already knows. For each conversation and file, `agy-receipts.sqlite` keeps the length of the last scanned prefix and
   the SHA-256 of exactly those bytes, the file identity, the highest step number, whether any record lacked an integer
   `step_index`, and the digest of every completed result (`scan_state`, `scan_results`). This ledger is a cache of a
   parse, never evidence. It is used only after the SHA-256 of that prefix was recomputed from the file in the pass that
   hashes it for the current hook (a hash state is kept at the stored length, so the check costs nothing extra), and only
   for the same path and file identity. A step the transcript has already reached (a repeated or late hook: the position
   matters), a result numbered at or below the stored maximum, an unreadable line, a different file or any mismatch parses
   the whole file. A failed parse stores nothing, and writes are compare-and-set on the stored length and digest in the
   transaction of the receipt that needed them. What a receipt records is unchanged: the same fields and the same plain
   SHA-256 `prefix_hash`.
2. **Anchors in the capture pass.** A sync hashes the file once with a state at the `prefix_length` of every receipt, so
   checking the anchor of each linked result needs no further read; it used to re-read the whole prefix once per linked
   result. The anchor is always the plain SHA-256 of the file's bytes as they are now. AGY sources are verified in
   full, never with block digests or samples (`integrity.full_reason: "legacy_source"`), and no cache or cheap
   observation can turn a mismatch into a link.
3. **A receipt reindexes a conversation only when it can change an indexed event.** The parser reads receipts in two
   ways only: the receipt of the step of each result record it turned into events, and the conversation's first
   receipt, which decides `legacy_unlinked`. The index stores, with the cursor and in the same transaction
   (`session_receipt_deps`), the digest of the receipt of every result step it parsed and the first receipt. A sync
   reparses from byte 0, with the existing `reindex_reason: "hook_receipts_changed"`, when the receipt of an indexed
   result step changed (added, removed or modified), or when the first receipt moved past an indexed result. A receipt for
   a step that has no result yet (the usual `PreToolUse`), and a `PostToolUse` that arrives before its result is
   indexed, are plain appends; the cursor is still republished so the stored receipt digest follows the receipts. A
   `PostToolUse` that pairs a result a sync already indexed as unlinked reparses, as does an index without stored
   dependencies (written before this version, or by an older process): once, on its next receipt change.

Measured with `scripts/bench_agy_hooks.py` (WSL2, ext4, Python 3.12; synthetic transcripts with a linked receipt per result; a few KB appended
before each hook; each hook a fresh process, interpreter start and imports included; median of five cycles; wall / CPU
seconds, and a process that only starts the engine costs about 0.3 / 0.3):

| Transcript | `PreToolUse` before | after | `PostToolUse` before | after |
|---|---|---|---|---|
| 0.1 MB | 0.44 / 0.35 | 0.38 / 0.33 | 0.42 / 0.34 | 0.36 / 0.31 |
| 1 MB | 0.89 / 0.77 | 0.45 / 0.36 | 0.84 / 0.75 | 0.41 / 0.33 |
| 3 MB | 2.83 / 2.68 | 0.45 / 0.40 | 2.74 / 2.60 | 0.44 / 0.39 |
| 10 MB | 20.5 / 20.1 | 0.51 / 0.45 | 20.4 / 20.0 | 0.58 / 0.52 |

What still grows with the file: two SHA-256 passes per hook (one for the receipt, one for the sync's capture;
about 14 ms per 10 MB each), and per-receipt work done in C (reading the receipts to check them, comparing the stored
dependencies): about 0.1 s for 3,400 receipts. The first hook after an upgrade, or after the transcript was
replaced, parses the whole file once.

### Parser version 6: truncated fields, result statuses and legacy results

The display log (`transcript.jsonl`) can cut a long field and list it in the record's `truncated_fields`. Parser version 6 handles that explicitly instead of treating the visible prefix as the whole value. Raising the parser version reparses each already-indexed AGY session once, on its next sync or audit.

**Fallback to `transcript_full.jsonl`.** For `USER_INPUT.content`, `PLANNER_RESPONSE.content`, `PLANNER_RESPONSE.tool_calls` and `GENERIC.content` (and `content` of the IDE's typed results), the parser looks up the same `step_index` in the sibling `transcript_full.jsonl` in the same directory (a symlink or a file of another conversation is refused). The full log is indexed in one streaming pass that keeps, per complete line, only the step, its byte range and a short digest, so memory grows with the number of records, not the file size; nothing about the full log is persisted. A restore must pass these checks: the record at that step has the same type, source and status; the field is complete and longer than the visible text and starts with the same text (for `tool_calls`, with the same call names); and the file still has the identity and digest it was indexed with. When it passes, the field is restored within the same step: the chunk carries the full text, `restored_fields`, a `full_ref` (byte range and digest in the full log) and, when no other judged field of that record is still cut, `source: "full"`. The restored field has no truncation marker.

**When the fallback fails.** The chunk keeps its visible text, keeps the field in `truncated_fields`, has `source: "display"` and records `fallback: {reason, retryable, full_size}`. The reasons are `full_missing`, `full_unavailable`, `full_changed`, `full_missing_step`, `full_duplicate_step`, `full_step_mismatch`, `full_truncated`, `full_field_missing`, `full_field_invalid`, `full_prefix_mismatch`, `full_not_longer` and `source_is_full` (the log being parsed is itself `transcript_full.jsonl`, so there is no sibling). Only transient failures are retryable: the full log is missing, has not reached that step yet, changed or could not be read. A retryable failure is retried once for each change of the full log's size (`reindex_reason: truncated_fallback_retry`); a permanent failure is never retried on every growth.

`thinking`, `CHECKPOINT`, `SYSTEM_MESSAGE` and `EPHEMERAL_MESSAGE` text is context, not evidence. Truncation of those fields is counted as `ignored` and never changes a verdict.

**Partial-observation rules.** A field that is still cut changes what the evidence can prove, never what it matched:

- A visible match still counts. A forbidden command in the visible part of a call is a `violation`.
- A contract that would be `clean` only for lack of a match, or unmet only for lack of evidence, while a field it depends on is still cut, is `unknown` with the reason `truncated_content` or `truncated_tool_calls` (in general `truncated_<field>`).
- A truncated `USER_INPUT` never grants a permission: it cannot extend an exception (the finding stays a `violation` with that reason) and cannot satisfy a user-role requirement.
- A truncated tool result never establishes success (`success_withheld: truncated_content`), but it does not hide a failure.

`coverage.agy_evidence.truncation` counts `restored_from_full`, `unrestored` (judged fields still cut) and `ignored` fields by `<type>.<field>`.

**Statuses and record types.** A `GENERIC` result (or one of the IDE's typed results) has a `status` of `DONE`, `RUNNING`, `ERROR`, `INVALID` or `CANCELED`; a recorded status outside that set is an unsupported event (a missing status is accepted).

| Record | Handling |
| --- | --- |
| `RUNNING` result | Pending: `success` is unset and `pending: true`. An obligation that depends on it is `unknown` with the reason `pending_running` until a final record for that step arrives. |
| `ERROR`, `INVALID`, `CANCELED` result | Never success evidence. The attempt remains: a forbidden command that was proposed is still a violation. An attempt that the pre-tool hook blocked (`tool call denied by pre-tool hook:` plus the anchored Pre receipt) is recorded as non-execution and is not counted as a violation. |
| `CHECKPOINT` | A context projection (`[checkpoint]`, runtime role): never a new user instruction and never observed execution. |
| `ERROR_MESSAGE` | Runtime text (`[host_error_message]`), not part of the default judgement. |
| Media on a result | One placeholder per item, `[media #N type=<mime> chars=<n> sha256=<12 hex>]`, and `media_count` in the metadata. The payload is never indexed; data URIs and long base64 runs elsewhere in text become `[blob omitted: N chars]`. |

**Legacy results.** The native logs omit call ids, so a result is linked to its call only through a host receipt. A tool result (`GENERIC` or a typed IDE result) without a receipt that comes before the conversation's first receipt (or in a conversation that has none) is `legacy_unlinked`. It is a known limitation of that history, not a current-session defect: it does not make `coverage.complete` false. A result without a receipt after the first receipt stays an unsupported-event defect. Coverage reports:

| Field | Meaning |
| --- | --- |
| `receipts_since` | `{step_index, byte_offset}` of the conversation's first receipt, or `null` if it has none. |
| `legacy_unlinked_count` | Number of legacy results. |
| `result_dependent_complete` | `false` while any result is legacy; evidence that depends on results is not complete. |
| `unknown_breakdown` | Count of `unknown` findings split into `current_session` and `unsupported_legacy`. |
| `agy_evidence` | Counts of legacy and running results, media placeholders, host-padded records (`host_padded_records`, parser version 7) and truncation by class. |

In an audit, an obligation that needs the executed call or its result and can only rest on legacy results is `unknown` with the reason `unsupported_legacy`: never unmet and never clean. Judgement of the call side (a forbidden command proposed by the model) is unaffected. Hooks add receipts from the moment they are installed; they do not create them retroactively.

For manual MCP calls pass `provider: "agy"`, the real `session_id`, and `path`
on sync/check/audit operations. Codex remains the shared API default.

### Parser version 7: IDE result steps, context steps and host padding

Version 7 adds three things. Raising the parser version reparses each already-indexed AGY session once, on its next sync or audit; event ids, anchors and cached vectors of unchanged events survive, and `embed=false` calls no embedding API. A running MCP process or hook process only picks the new parser up after a restart.

**IDE typed result steps.** The Antigravity IDE (`antigravity-ide` app data) writes the result of a tool call as a step whose type names the tool, where the CLI writes `GENERIC`. These five types are tool results with the same evidence rules as `GENERIC`:

| Step type | Carries | Notes |
| --- | --- | --- |
| `RUN_COMMAND` | `content`, `exit_code` | The only typed result that can be success evidence (below). |
| `VIEW_FILE` | `content`, `truncated_fields` | Output of reading a file. |
| `CODE_ACTION` | `content` | Result of an edit. |
| `MCP_TOOL` | `content` | Result of an MCP tool. |
| `LIST_DIRECTORY` | `content` | Result of a directory listing. |

- **Linkage.** No call ids exist, so a result is linked to its call only through a host receipt (`linkage: host_step_receipt`, with the call emitted as a `tool_call` event). A result without a receipt is `legacy_unlinked` when it comes before the conversation's first receipt (or in a conversation that has none) and a current-session defect (`native_log_omits_call_id`, an unsupported event) after it, exactly as for `GENERIC`. Hook receipts bind to a typed result the same way: a result that already existed when the Pre hook ran, or that was rewritten afterwards, cannot be linked. The chunk metadata keeps the step type as `result_type` (the `GENERIC` chunk is unchanged), and `result_type` is part of the slim event metadata audits see.
- **Truncation.** `content` falls back to the same step of `transcript_full.jsonl` under the checks above; what stays cut is marked, counted in `coverage.agy_evidence.truncation` (`VIEW_FILE.content`, `RUN_COMMAND.content`, ...) and leaves a dependent verdict `unknown` with the reason `truncated_content`. A visible match still counts. A cut result that cannot carry success (`VIEW_FILE`, `CODE_ACTION`, `MCP_TOOL`, `LIST_DIRECTORY`) does not affect a success obligation.
- **Status.** `RUNNING` is pending (`pending_running` until a final record for that step arrives; a command that is still running when the session ends stays `unknown`), `ERROR`, `INVALID` and `CANCELED` are never success, and an unrecognized status is an unsupported event.
- **`RUN_COMMAND` exit codes.** `exit_code` is the host-recorded execution status, read from the structured field and never from output text, and not at all while the command is `RUNNING`. A `DONE` command with `exit_code` 0 that is linked to its call by a host receipt (a `run_command` call) is success evidence wherever a `GENERIC` success would be: it satisfies `requires_success` and checklist evidence. A nonzero `exit_code` is a failure (`success: false`, recorded as `exit_code` in the metadata) whether or not the result is linked; that can only make a verdict worse. An unlinked result is never success (`success_withheld: unlinked`). Success is also withheld, as for `GENERIC`, when the output is still cut (`truncated_content`) and is `false` when the Post hook reported an error. A linked `DONE` `RUN_COMMAND` without an `exit_code` is an unsupported event; a present `exit_code` that is not an integer is a parse error.
- **Other typed results** can be linked to their call but are never proof that the tool met its objective: their `success` stays unset (a failed status gives `false`).
- A typed result is accepted only from `source: "MODEL"`. A typed result from another source, or with a status outside the known set, stays an unsupported event.

**Context steps.** `CONVERSATION_HISTORY` and `KNOWLEDGE_ARTIFACTS` (source `SYSTEM`, no content) record that the IDE loaded context. They are runtime metadata (`role: runtime`, `evidence_origin: conversation_history` or `knowledge_artifacts`): never a user instruction, never observed execution, and not unsupported. Any other unknown step type stays `unsupported_event`, as before.

**Host padding.** One CLI transcript contained a line made of a contiguous run of NUL bytes immediately followed by a complete JSON record, between two consecutive steps. The parser strips a *leading* NUL run from a line; if the remainder parses as a valid record, the record is accepted and its chunks carry `host_padding_bytes` (the run length; the byte range and anchors still cover the host's own bytes). `coverage.agy_evidence.host_padded_records` counts padded lines once each. The same reader is used for `transcript_full.jsonl` (the fallback) and for hook-receipt anchoring. Nothing else changes: a line that is only NULs, NULs followed by something that is not a valid record, NULs after or inside a record, and a padding run that is not at the very start of the line all stay parse errors that make coverage incomplete. Neighbouring step indexes never excuse a bad line, so a missing step stays visible.

### Parser version 8: model-written tool labels

Version 8 keeps the model's own labels for a tool call out of the call text that contracts match. A call's arguments can carry `toolAction` and `toolSummary` (the recorded fixture has `"toolAction": "Running tests"` and `"toolSummary": "Execute tests"` beside `CommandLine`): short UI labels the model writes, not the command. They are removed from the text of a host-receipt `tool_call` and of a model proposal and kept in the event's `meta.tool_prose`; the proposed action a `PreToolUse` hook asks about is built by the same function (`tool_payload.py`), and so is the text the standard-library fallback matches against `risk: "high"` contracts. Every other argument stays payload, and only a string value counts as prose. `Description`, `Instruction` and `Explanation` style arguments are not excluded: nothing in the code or the fixtures confirms that they are prose. Raising the parser version reparses each already-indexed AGY session once, on its next sync or audit; restart running processes to load it.

### Host verification checklist

With isolated synthetic workspace/history, verify forbidden commands and restricted
reads are blocked, a failed prerequisite remains blocked, a successful prerequisite
unlocks its dependent action, an `unknown` call is allowed with a warning that
appears at the next PreInvocation (and asks when a `risk: "high"` contract applies),
a confirmed unmet `auto_continue` obligation continues a normal Stop once and not a
second time, unverified checklist work only produces a note, cancel still stops, and
PreInvocation context appears. Verify `force_ask` with a cached Always Allow grant. Offline tests do not establish host registration,
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
