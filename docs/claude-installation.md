# Claude Code integration and lifecycle hooks

This document explains how to connect `tsukkomi-mcp` to Claude Code: the MCP server, the opt-in command hooks installed by `scripts/install_claude_hooks.py`, what those hooks may block, and how to check that everything works. Nothing here is installed by default. Reading or cloning the repository does not change any Claude Code setting.

---

## 1. What the integration consists of

| Part | What it does |
|---|---|
| Provider `claude` (parser version 3) | Reads Claude Code session transcripts, `<projects>/<project>/<session-uuid>.jsonl`, and subagent transcripts, `<projects>/<project>/<session-uuid>/subagents/agent-<agent-id>.jsonl`, as evidence. See [section 9](#9-how-the-claude-parser-decides-who-is-the-user). |
| MCP server | The same `python -m self_directing_mcp` server used for every host. The agent calls it with `provider: "claude"`. See [section 2](#2-register-the-mcp-server). |
| Command hooks (opt-in) | `scripts/claude_hook.py` is registered for seven Claude Code events by `scripts/install_claude_hooks.py`. It judges proposed tool calls and stop checkpoints with the same contracts. See [sections 3 and 4](#3-install-the-hooks). |

The hooks are separate from the MCP connection. Each hook invocation is a short-lived process that opens the local index, reads the transcript Claude Code names in the hook payload and answers on stdout. Hooks never call a model and never embed text (`embed=false`).

By default the hooks block only confirmed findings: a confirmed `violation` is denied and a `suspicious` finding asks the user. Everything else passes through with a visible warning. See the [policy table](#4-what-the-hooks-decide).

---

## 2. Register the MCP server

Install the package into a Python 3.12+ environment first (see the [README quickstart](../README.md#quickstart)), then register the server with Claude Code. A user-scope registration with an absolute interpreter looks like this:

```bash
claude mcp add --scope user tsukkomi-mcp -- /absolute/path/to/venv/bin/python -m self_directing_mcp
```

Check `claude mcp add --help` for the flags of your Claude Code version (scope and environment options have changed between versions) and `claude mcp list` afterwards. Tool names appear to the agent as `mcp__tsukkomi-mcp__<tool>`.

Contracts are written by the MCP server and read by the hooks, so both must use the same index. Keep the settings that must agree in the shared TOML file `~/.config/tsukkomi-mcp/config.toml` (or the file named by `SELF_DIRECT_CONFIG_FILE`) instead of in the server entry: the index directory, the contracts file and the projects root must resolve to the same values in both processes. See [config.example.toml](../config.example.toml) and the [README path contract](../README.md#user-path-configuration-toml). The relevant keys are `index_dir`, `contracts_path` and `claude_projects_dir`. If the hooks must use a specific index regardless of the environment, install them with `--index-dir` (see [section 3](#where-the-hooks-read-their-settings)).

For the default OpenRouter-backed retrieval, also provide `SELF_DIRECT_OPENROUTER_API_KEY_FILE` and `SELF_DIRECT_SQLITE_VECTOR_PATH` (TOML keys `openrouter_api_key_file` and `sqlite_vector_path`) as described in the README. For a credential-free setup, set `SELF_DIRECT_LOCAL_ONLY=true` in the server's environment; embedding, dense and hybrid search and LLM graph updates are then rejected rather than replaced by a fallback.

Calls from the agent must name the host: pass `provider: "claude"` and the real session UUID. A session-scoped contract needs the UUID that is the file name of the transcript (`<session-uuid>.jsonl`); omit `session_id` to register a global contract that applies to every Claude Code session. A subagent transcript is its own session with the id `<session-uuid>_agent-<agent-id>`. Setting `SELF_DIRECT_SESSION_PROVIDER=claude` in a Claude-only server entry changes the default provider; the shared default remains `codex`.

---

## 3. Install the hooks

`scripts/install_claude_hooks.py` uses only the Python standard library. Its default action prints what it would add and changes nothing. There is no default settings path: the file to modify is always named explicitly.

```bash
# 1. Review the exact hooks block and env (changes nothing).
python scripts/install_claude_hooks.py --print --python /absolute/path/to/venv/bin/python

# 2. Merge it into the settings file you choose.
python scripts/install_claude_hooks.py --apply --settings /absolute/path/to/settings.json \
    --python /absolute/path/to/venv/bin/python

# 3. Remove exactly what the installer owns.
python scripts/install_claude_hooks.py --uninstall --settings /absolute/path/to/settings.json
```

Apply only when the user asked for the hooks. The complete `--print` output with placeholder paths is in [claude-hooks.example.json](claude-hooks.example.json). Use whichever settings file the user chose; the installer assumes no location and edits only the file it is given.

### Options

| Option | Meaning |
|---|---|
| `--print` | Default. Print `{"hooks": ..., "env": ...}` as JSON and write nothing. |
| `--apply` | Merge into `--settings PATH`. Refused if the hook script file does not exist. |
| `--uninstall` | Remove the owned handlers and the owned `env` key from `--settings PATH`. |
| `--settings PATH` | The settings JSON file. Required with `--apply` and `--uninstall`. |
| `--mode {confirmed_only,advisory,enforced}` | Value written to `env.SELF_DIRECT_CLAUDE_ENFORCEMENT`. Default `confirmed_only`. |
| `--events A,B,...` | Comma-separated subset of the seven events. Default all. |
| `--python PATH` | Interpreter that runs the hook command. Default: the interpreter running the installer. |
| `--script PATH` | Path of `scripts/claude_hook.py`. Default: the copy next to the installer. |
| `--index-dir PATH` | Absolute index directory, baked into every hook command as `--index-dir=PATH`. A relative path is rejected (exit status 2). See [below](#where-the-hooks-read-their-settings). |
| `--local-only` | Append `--local-only` to each hook command. See below. |

### Events

| Event | Matcher | Timeout (s) | What the hook does |
|---|---|---|---|
| `SessionStart` | none | 30 | Injects active contracts (up to 8) and pending checklist items (up to 8) as context, marked as data. Also delivers queued warnings and the health banner. |
| `UserPromptSubmit` | none | 30 | Delivers queued warnings and the health banner. Never reads the prompt. |
| `PreToolUse` | `*` | 60 | Judges the proposed tool call. May deny or ask. |
| `PostToolUse` | `*` | 30 | Delivers queued warnings. Records nothing. |
| `PostToolUseFailure` | `*` | 30 | Same as `PostToolUse`. A failed tool call is never success evidence. |
| `Stop` | none | 120 | Checks outstanding obligations. May continue the agent once. |
| `SubagentStop` | none | 120 | Judges the subagent's own transcript. Reports only. |

The tool matchers stay `*`. Reads can be restricted by a contract, and a tool name does not prove the absence of side effects, so the hook has no read exemption and no tool-name exemption.

### How the installer edits the file

- **Merge, not overwrite.** Unrelated hooks, their order, unrelated settings and other handlers sharing a matcher group are preserved. A missing settings file is created. Rerunning is idempotent: an unchanged result reports `"written": false` and writes nothing.
- **Ownership.** A handler is owned when its command is a `command` handler containing `--event=` and a `scripts/claude_hook.py` path token. A differently named foreign script is never touched. Reapplying updates an owned handler in place, collapses duplicate owned handlers to one, and removes owned handlers of events no longer selected by `--events`.
- **Refusal.** A settings file that is not a JSON object, is unreadable, or has a malformed `hooks` entry for an event the installer manages is an error, and nothing is written.
- **Backup only on change.** Before writing a changed file, the installer copies it to `<settings-file>.tsukkomi-backup-<UTC timestamp>` next to it (with a counter suffix if that name exists). A newly created file, or an unchanged result, makes no backup.
- **Atomic write.** The new content goes to a temporary file in the same directory and replaces the target with `os.replace`. File mode and the original indentation are kept. A symlinked settings file is written through, not replaced.
- **The `env` key.** `--apply` always sets `env.SELF_DIRECT_CLAUDE_ENFORCEMENT` to the `--mode` value (default `confirmed_only`), so rerunning without `--mode` resets an earlier choice. `--uninstall` removes that key.
- **Interpreter and checkout.** `scripts/claude_hook.py` starts with whatever `--python` names. If a `.venv` exists in the checkout and the running interpreter is not that one, the script re-executes itself with the checkout's `.venv` interpreter. If the package cannot be imported at all, a standard-library fallback answers fail-open and records `hook_health` (see [section 7](#7-failures-fail-open)).

### Where the hooks read their settings

Hosts run lifecycle hooks inside the project being edited. A `.env` in that directory is content the audited agent can write, so the hooks never read it: they build their settings from the process environment and the user TOML file only (`config.hook_settings()`), the same sources the standard-library fallback uses. Otherwise a project `.env` could point a hook at an empty index or at `advisory` mode and turn a confirmed violation into an allow. The MCP server is different: it still reads a `.env` in its own working directory, at the lowest precedence.

So the index directory, the contracts path and the enforcement mode of the hooks must come from one of these:

- The user TOML (`index_dir`, `contracts_path`, `claude_projects_dir`), shared with the MCP server. This is the simplest way to keep both on one index.
- The environment the hook command inherits from Claude Code, for example the settings `env` block. The installer writes `SELF_DIRECT_CLAUDE_ENFORCEMENT` there.
- `--index-dir /absolute/path/to/index`. The installer adds `--index-dir=/absolute/path/to/index` to every hook command, and the hook entry point (and its standard-library fallback) sets `SELF_DIRECT_INDEX_DIR` from it, overriding the inherited environment. Contracts, the continue budget, queued warnings and `hook_health` then live in that index. This is the recommended way to point the hooks at the index whose contracts should apply, that is, the index of the MCP server the user registers contracts with.

The installer does not write `SELF_DIRECT_INDEX_DIR` into the settings `env`, for the same reason it does not write `SELF_DIRECT_LOCAL_ONLY`: that block is session-wide and would also reach every MCP server Claude Code starts.

### Why `--local-only` is a flag and `SELF_DIRECT_LOCAL_ONLY` is not in `env`

Hooks never embed, but a hook process that builds its engine in the default mode still needs the key file and the native vector library to initialize. `--local-only` makes the hook process set `SELF_DIRECT_LOCAL_ONLY=true` for itself, so it needs neither. The installer does not write that variable into the settings `env` block because that block is session-wide: the same variable could reach other processes Claude Code starts, such as the MCP server, and silently disable its embeddings and dense search. The flag is limited to the hook command.

---

## 4. What the hooks decide

One judgement, host-specific enforcement. The verdict-to-decision table is shared with the AGY adapter in `src/self_directing_mcp/hook_policy.py`; this document and the [AGY document](agy-installation.md#enforcement-and-failures) differ only in how each host expresses a decision. The mode comes from `SELF_DIRECT_CLAUDE_ENFORCEMENT` and is independent of `SELF_DIRECT_AGY_ENFORCEMENT`.

| Mode | Meaning |
|---|---|
| `confirmed_only` (default) | Only confirmed findings gate: `violation` denies, `suspicious` asks. Unverifiable calls pass with a warning unless a `risk: "high"` contract applies. |
| `advisory` | Never denies or asks. Non-clean findings appear as `ADVISORY (not compliance)` messages. |
| `enforced` | Strict opt-in. Also denies an unverifiable PreToolUse. |

### PreToolUse

"None" means the hook gives no permission decision, so Claude Code's own permission flow decides.

| Verdict for the proposed call | `advisory` | `confirmed_only` | `enforced` |
|---|---|---|---|
| `violation` (confirmed) | none, advisory message | `deny` | `deny` |
| `suspicious` (concrete evidence, such as an exception that needs confirmation) | none, advisory message | `ask` | `ask` |
| `unknown`, no high-risk contract applies | none, advisory message | none, with a visible unverified warning | `deny` |
| `unknown`, and an applicable contract has `risk: "high"` | none, advisory message | `ask` | `deny` |
| `clean` | none | none, silent | none, silent |
| No applicable contract for this call | none | none, silent | none, silent |

Claude Code has no equivalent of AGY's `force_ask`: `ask` is the strongest confirmation available. A `suspicious` finding that sits beside an `unknown` prerequisite stays `ask` in `confirmed_only`, because confirming an exception cannot waive another unverified prerequisite in strict mode; `enforced` turns that combination into `deny`.

The optional owner-approved workflow layer adds one more rule: a direct completion action that the approved workflow maps to its own executor is denied in `confirmed_only` and `enforced`, and passes with an advisory message in `advisory`.

### Stop, SubagentStop and errors

| Case | `advisory` | `confirmed_only` and `enforced` |
|---|---|---|
| `Stop`, an `auto_continue` obligation confirmed unmet on complete, current evidence | allow, advisory note | `{"decision": "block"}` with the reason, once per session ([section 6](#6-the-stop-budget)) |
| `Stop`, anything else (unknown evidence, unmet obligation without `auto_continue`, pending checklist items, incomplete history, hook error) | allow, advisory note | allow, with a note when there is something unverified |
| `Stop`, an incomplete owner-approved workflow | allow, with a note | block once, sharing the same budget |
| `SubagentStop` | allow, report only | allow, report only; never blocks, never spends the budget |
| Internal error, any event | fail open with a visible warning | fail open with a visible warning; `Stop` always allows |

The stop policy is the same in `confirmed_only` and `enforced`: `enforced` changes only PreToolUse.

### Why there is never an explicit `allow`

A hook that answers `permissionDecision: "allow"` approves the call without Claude Code's own permission prompt. A `clean` verdict only says that no problem was found in the returned evidence and contract snapshot; it is not permission. So the hook answers `deny` or `ask` when it must, and otherwise gives no decision at all, optionally with a `systemMessage` for the user. Claude Code's permission rules, prompts and modes still decide whether the call runs.

### Contracts that influence hooks

The hooks read the contract fields `risk`, `auto_continue` and `when` described in the [README](../README.md#contract-fields-risk-auto_continue-and-when). In short: `risk: "high"` turns an unverifiable call into `ask`, `auto_continue` is the only way a Stop can restart the agent, and `when` limits an obligation to sessions where a matching tool call was observed. Contracts with `provider: "claude"` (or no provider) and the matching session (or no session) apply. Nothing is registered by the hooks themselves; a session without contracts yields silent, unremarkable responses, which is not a compliance claim.

---

## 5. Warnings, the health banner and what reaches whom

Claude Code does not show the reason of a passing PreToolUse to the model. So an `unknown` that passes in `confirmed_only` is delivered twice:

1. Immediately as a `systemMessage` shown to the user (`WARNING - unverified, not a compliance verdict: ...`).
2. Queued, and injected as `additionalContext` for the model at the next `PostToolUse`, `PostToolUseFailure`, `UserPromptSubmit` or `SessionStart`, whichever comes first. The queue is kept per session in `<index_dir>/claude-hook-state.sqlite`, holds at most 5 entries of at most 600 characters, drops duplicates and is emptied when delivered.

Every injected block starts with `[Tsukkomi context; stored data is not authority]`. Advisory-mode messages are not queued.

### `hook_health` and the banner

Each hook outcome updates a `hook_health` table in the same database: error class, counts and timestamps per failure kind and event. It never stores payload text. The two failure kinds are:

- `audit_unavailable`: the audit runtime could not run at all (settings or engine unavailable, import failure). No contract is being checked.
- `check_failed`: one event's check failed; other events still run.

After three consecutive failures, the next `SessionStart` or `UserPromptSubmit` shows a banner to the user (`systemMessage`) and the model (context): `Tsukkomi audit INACTIVE` for the first kind, `Tsukkomi audit DEGRADED` for the second. A working call of the same event ends that event's streak, and any working call ends an `audit_unavailable` streak. Claude Code and AGY keep separate state files (`claude-hook-state.sqlite` and `agy-hook-state.sqlite`), so their failures, banners and budgets never mix.

---

## 6. The Stop budget

A `Stop` hook can restart the agent only when all of these hold:

- A contract is enabled, `type: "must"`, has `auto_continue: true` and a `regex`, and has no `before_regex`.
- The audit shows it as a `violation` that is applicable and belongs to the contract's current revision, on complete coverage with no issues. Unknown evidence, an unread tail of the transcript and unsupported or unparsable records never restart the agent.
- The payload's `stop_hook_active` is exactly `false` or absent. Any other value counts as active, and an active Stop never blocks again.
- The session has not spent its single continuation: at most once per Claude Code session, persisted in `claude-hook-state.sqlite`, surviving process restarts and not reset by turns.
- The same reason (the set of contract ids and revisions) has not been used before.
- The counter can be read and written. If it cannot, the answer is "do not continue".

When the budget is spent, or the reason repeats, the hook allows the stop and adds a note. The answer to the agent is `{"decision": "block", "reason": "A user-specified completion obligation is confirmed unmet: [id] ... This is the only automatic continuation for this session."}`. A checklist item that is still pending, a description-only contract or any hook error never continues the agent.

### Stop runs before the final message is on disk

Observed on Claude Code 2.1.299 (payload key names only): when `Stop` runs, the last assistant message is not yet in the transcript file. `Stop` and `SubagentStop` carry `last_assistant_message` and `stop_hook_active`. An audit of the file alone would call a `message`-scoped obligation unmet even if the final message satisfies it, and a block must rest on observed evidence only.

The adapter therefore defers only misses of contracts whose `scope` is `message` or `any`, until the host's copy of the final message appears in the last 256 KiB of the transcript (the first 80 characters, compared in both JSON-escaped forms). If it has not appeared, those misses are dropped for this stop and the note `Claude Code's transcript does not contain the final assistant message yet` is added; no budget is spent. Misses of `tool_call` and `tool_result` obligations are judged normally, because earlier calls and results are already on disk. If the payload carries no usable `last_assistant_message` (an older version, or an empty value), nothing can be compared and the audit stands on its own coverage checks.

---

## 7. Failures fail open

The hook always exits with code 0 and prints one JSON object. Exit code 2, which would block, is never used. An internal error (malformed stdin, an invalid session id or transcript path, an unavailable runtime, a corrupt contracts file, a handler exception) produces a warning and no block:

| Event | Response on error |
|---|---|
| `PreToolUse` | `systemMessage` with the warning and no decision. Only when the mode is not `advisory` and an enabled `risk: "high"` contract applies to the call (read with the standard library, without the package), the answer is `ask` instead. Unreadable contracts never ask or deny. |
| `Stop`, `SubagentStop`, `PostToolUse`, `PostToolUseFailure` | `systemMessage` with the warning. |
| `SessionStart`, `UserPromptSubmit` | Context stating `Contract context was not injected; do not claim compliance.` plus the warning and, when due, the banner. |

The warning names the error class only (`Tsukkomi check FAILED (ValueError): ... fail-open, not a compliance verdict.` or `Tsukkomi audit UNAVAILABLE (...)`), never the message, and the raw stdin is never echoed. All hook output, including contract text and history excerpts, passes through the secret masking described in [SECURITY.md](../SECURITY.md#masking-contract).

Fail-open also covers the cases the hooks cannot see: hosted tools that emit no `PreToolUse` event, out-of-band shell access, and transcript records not yet flushed. A skipped or failed hook is neither `clean` nor authorization.

---

## 8. Subagents

- **PreToolUse inside a subagent** carries `agent_id` and `agent_type` but no `agent_transcript_path`. The audit reads the main session's transcript, which does not contain the subagent's own history. Findings of history-dependent `must` obligations (a prerequisite, for example) are therefore demoted to `unknown`; `must_not` findings stand. Subagent calls are otherwise judged like any other call.
- **SubagentStop** carries `agent_id`, `agent_transcript_path` (`<projects>/<project>/<session-uuid>/subagents/agent-<agent_id>.jsonl`), `last_assistant_message` and `stop_hook_active`. The adapter judges that file as its own session, `<session-uuid>_agent-<agent_id>`. If the payload has no path, it derives the same path from the main transcript's directory. A path that does not belong to the stated session is an error and fails open. A transcript that cannot be located yields an allow with a note.
- **SubagentStop reports but never blocks.** A global obligation the user set for their own session is not a task the subagent was given, so it cannot justify restarting it. A confirmed miss is reported with the note `A subagent's stop is reported, never blocked.`, and subagent stops never spend the session's continuation. Contracts scoped to the parent session do not apply to the subagent's own transcript; global Claude Code contracts do.

---

## 9. How the Claude parser decides who is the user

Parser version 3 (`src/self_directing_mcp/claude.py`) treats authority as something the host recorded, never something the text claims. A user turn is only:

- a record the host stamped `origin.kind == "human"` that is not a meta, compaction or tool-result-carrying message;
- in a main transcript only, a legacy prompt from before origin stamps existed (no `origin` at all), marked `attribution: "inferred_legacy"`;
- a `queued_command` attachment that has `origin.kind == "human"`, `commandMode == "prompt"` and is not `isMeta`. A queued command with any other origin or mode is runtime evidence. A human origin on a meta or compaction record, or on a queued command that is not a prompt, is a contradiction and is reported as unsupported.

Everything else is runtime evidence and can neither grant nor revoke a user permission: task notifications, peer and coordinator messages, injected meta messages, compaction summaries, transcript-only messages, text riding with tool results, attachments, hook output, tool results, synthetic or API-error assistant text, scheduled-task announcements, and a subagent's task prompt (the parent agent's instruction, not the human's).

**Model-written tool prose (parser version 3).** Claude Code's `Bash` input (and that of the other built-in tools that take one) carries a `description`: a label the model writes about the call. It is not part of what the call does, so the parser removes it from the indexed call text (`[tool_call:Bash] {"command": "..."}`) and keeps it in `meta.tool_prose`. The `PreToolUse` hook's proposed action goes through the same builder, so a careful `ls -l data/app.sqlite3` is no longer matched by a `sqlite3` prohibition because its description mentions it, while `sqlite3 data/app.sqlite3` in `command` still is, whatever the description says. An MCP tool (`mcp__server__tool`) defines its own schema; its `description` argument may be real content and stays payload, like every field that is not a confirmed prose field. Raising the parser version from 2 to 3 reparses each indexed session once, on its next sync or audit.

**Scheduled tasks (parser version 2).** When a scheduled task (a cron entry) fires, Claude Code writes a `system` record with `subtype: "scheduled_task_fire"`; its `prompt` is the scheduled automation text. The parser records it as runtime metadata (`evidence_origin: "scheduled_task_fire"`, `role: "runtime"`) and, like every other `system` payload, validates `prompt` and `content` as text and replaces them by a placeholder: the scheduled prompt is not indexed and is never a user turn, so it cannot satisfy or violate a user-role contract. Only short identifiers reach the index and the metadata (`task_id`, `task_kind`, `cron_kind` and a `cron` expression made of digits and cron punctuation); an odd identifier is dropped. A payload field that is not text, and every other system subtype (including look-alikes such as `scheduled_task_fired`), stays an unsupported event that keeps coverage incomplete. Raising the parser version from 1 to 2 reparses each indexed Claude Code session once, on its next sync or audit; event ids, anchors and cached vectors of unchanged events survive.

Shapes are validated strictly. An unknown record type, attachment type, content block or origin kind is reported as an unsupported event and keeps coverage incomplete; unknown never becomes `clean`. Thinking blocks, signatures, base64 images, hook output and file snapshots are replaced by placeholders and are not indexed. A tool result is linked to its call by `tool_use_id` and is success evidence only when it is not an error. A `read_truncation_notice` marks the result it names as cut: a visible match in it still counts, but it never establishes success, it cannot clear a prohibition, and it does not make an obligation unmet. Those cases are `unknown` with reason `truncated_result`.

The hooks follow the same rule. `UserPromptSubmit` fires for typed prompts and also for scheduled, loop, subagent-report and peer messages, so it is never authority: the hook does not read the prompt, and a payload field such as `prompt` is neither indexed nor echoed. Authority comes only from the transcript.

### Session roots and `CLAUDE_CONFIG_DIR`

The default projects root is `~/.claude/projects`. When `CLAUDE_CONFIG_DIR` is set in the process environment, the default becomes `$CLAUDE_CONFIG_DIR/projects`. The precedence is `SELF_DIRECT_CLAUDE_PROJECTS_DIR`, then the TOML key `claude_projects_dir`, then that default. The hook processes and the MCP server inherit Claude Code's environment, but verify that both resolve to the same root.

A transcript is accepted only if it is an absolute path without `..`, inside the configured root, in one of the two official layouts above, with no symlink or junction below the root. Anything else is an error: for a hook that means a fail-open warning, so a wrong root shows up as repeated `check_failed` health entries and then the banner.

---

## 10. Verify the installation

1. **Print before applying.** `python scripts/install_claude_hooks.py --print` shows the exact hooks and `env`. After `--apply`, running it again reports `"written": false` and an empty `changes` list.
2. **Check the hooks Claude Code loaded.** Use `/hooks` in a new Claude Code session; settings changes may need a new session to take effect.
3. **Run the hook by hand against a synthetic session.** This exercises the real script without Claude Code. Use a scratch directory, never a real index:

   ```bash
   DEMO=$(mktemp -d)
   SID=11111111-2222-4333-8444-555555555555
   TRANSCRIPT=$DEMO/projects/-demo/$SID.jsonl
   mkdir -p "$DEMO/projects/-demo" "$DEMO/index"
   printf '%s\n' '{"type":"user","uuid":"00000000-0000-4000-8000-000000000001","parentUuid":null,"sessionId":"'$SID'","timestamp":"2026-10-01T00:00:01.000Z","cwd":"/demo","isSidechain":false,"origin":{"kind":"human"},"message":{"role":"user","content":"hello"}}' > "$TRANSCRIPT"
   printf '%s\n' '{"schema_version":"1.0","contracts":[{"id":"no-force-push","type":"must_not","scope":"tool_call","provider":"claude","regex":"git push --force"}]}' > "$DEMO/index/contracts.json"
   export SELF_DIRECT_INDEX_DIR="$DEMO/index" SELF_DIRECT_CLAUDE_PROJECTS_DIR="$DEMO/projects"
   printf '%s' '{"hook_event_name":"PreToolUse","session_id":"'$SID'","transcript_path":"'"$TRANSCRIPT"'","tool_name":"Bash","tool_input":{"command":"git push --force origin main"}}' \
     | /absolute/path/to/venv/bin/python scripts/claude_hook.py --event=PreToolUse --local-only
   ```

   Expected: a `permissionDecision` of `deny` in `hookSpecificOutput`. With `"command":"ls"` the output is `{}` (no decision). Malformed stdin (for example `echo '{}'`) yields a `systemMessage` that starts with `Tsukkomi check FAILED` and exit code 0.
4. **Check the parser.** In a real session, call `sync_session` with `provider: "claude"` and the session UUID, then `audit_status`. Expect `parser_version: 2` and `coverage.complete: true` (a transcript still being written may report `unread_bytes`, which is a pure append and not a defect). `unsupported_events` or `parse_errors` mean a record shape the parser does not accept.
5. **Check the configured process, not just the files.** `audit_status` reports the embedder, the dense backend and degraded flags of the server Claude Code started. File existence is not verification.
6. **Look at hook health.** `<index_dir>/claude-hook-state.sqlite`, table `hook_health`, lists error classes and counts only. A growing `consecutive` value for an event means that check is failing open.
7. **Uninstall check.** `--uninstall --settings PATH` removes only the owned handlers and the `SELF_DIRECT_CLAUDE_ENFORCEMENT` key; a repeat reports `"written": false`.

Offline tests do not establish how a particular Claude Code version renders `systemMessage`, whether it honours `ask` in every permission mode, or how it behaves headless. Do not claim those without observing them. In particular, in a headless `claude -p` run an `ask` cannot be answered interactively and the tool call fails, which is effectively a `deny`. A `suspicious` finding (or an `unknown` under a `risk: "high"` contract) therefore blocks an unattended run. Use `advisory` when that is not wanted.

The hooks are a guardrail, not a security boundary. A user or process with write access to the index, the contracts file, the hook script or the Claude Code settings can change what they do, and Claude Code itself decides whether a hook decision is honoured. See [SECURITY.md](../SECURITY.md) for what is in and out of scope.
