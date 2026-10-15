# Grok CLI integration and lifecycle hooks

This document explains how to connect `tsukkomi-mcp` to the Grok CLI (xAI; developed against version 1.0.52): the session provider, the opt-in command hooks installed by `scripts/install_grok_hooks.py`, what those hooks may block, and how to check that everything works. Nothing here is installed by default. Reading or cloning the repository does not change any Grok setting, and none of this repository's code ever runs the `grok` binary.

---

## 1. What the integration consists of

| Part | What it does |
|---|---|
| Provider `grok` (parser version 1) | Reads `updates.jsonl`, the ACP session update stream Grok appends to, under `<sessions dir>/<url-encoded cwd>/<session id>/`. Nothing else in a session directory is read. See [section 6](#6-how-the-grok-parser-decides-what-happened). |
| MCP server | The same `python -m self_directing_mcp` server used for every host. The agent calls it with `provider: "grok"`. See [section 2](#2-register-the-mcp-server). |
| Command hooks (opt-in) | `scripts/grok_hook.py` is registered for six Grok events by `scripts/install_grok_hooks.py`, in one file, `<grok home>/hooks/tsukkomi.json`. It judges proposed tool calls and stop checkpoints with the same contracts. See [sections 3 and 4](#3-install-the-hooks). |
| Tool-name map | Contracts written for Claude Code's tool names and argument keys match Grok's calls, because a Grok call is rendered in that vocabulary at the matching boundary. See [section 5](#5-the-tool-name-map). |

The hooks are separate from the MCP connection. Each hook invocation is a short-lived process that opens the local index, reads the transcript Grok names in the hook payload and answers on stdout. Hooks never call a model, never embed text (`embed=false`) and never send session text anywhere. There is no resident-daemon mode for Grok.

By default the hooks block only confirmed findings: a confirmed `violation` is denied and a `suspicious` finding asks the user. Everything else is allowed with a warning the model receives. See the [policy table](#4-what-the-hooks-decide).

---

## 2. Register the MCP server

Install the package into a Python 3.12+ environment first (see the [README quickstart](../README.md#quickstart)), then register the server with Grok in the way your Grok version documents for MCP servers (user level, with an absolute interpreter): `/absolute/path/to/venv/bin/python -m self_directing_mcp`.

Contracts are written by the MCP server and read by the hooks, so both must use the same index. Keep the settings that must agree in the shared TOML file `~/.config/tsukkomi-mcp/config.toml` (or the file named by `SELF_DIRECT_CONFIG_FILE`): the index directory, the contracts file and the sessions root must resolve to the same values in both processes. The relevant keys are `index_dir`, `contracts_path` and `grok_sessions_dir`. If the hooks must use a specific index regardless of the environment, install them with `--index-dir`.

The sessions root is `SELF_DIRECT_GROK_SESSIONS_DIR`, else TOML `grok_sessions_dir`, else `$GROK_HOME/sessions` when `GROK_HOME` is set, else `~/.grok/sessions`. Calls from the agent must name the host: pass `provider: "grok"` and the real session id, a UUID (the name of the session directory). Omit `session_id` to register a global contract that applies to every Grok session.

---

## 3. Install the hooks

`scripts/install_grok_hooks.py` uses only the Python standard library. Its default action prints what it would write and changes nothing.

```bash
# 1. Review the exact hook file (changes nothing).
python scripts/install_grok_hooks.py --print --python /absolute/path/to/venv/bin/python

# 2. Write <grok home>/hooks/tsukkomi.json.
python scripts/install_grok_hooks.py --apply --python /absolute/path/to/venv/bin/python --local-only \
    --index-dir /absolute/path/to/index

# 3. Check the file and run the registered command once on synthetic data.
python scripts/install_grok_hooks.py --verify --python /absolute/path/to/venv/bin/python --local-only \
    --index-dir /absolute/path/to/index

# 4. Remove exactly what the installer wrote.
python scripts/install_grok_hooks.py --uninstall
```

Apply only when the user asked for the hooks. The complete `--print` output with placeholder paths is in [grok-hooks.example.json](grok-hooks.example.json). A global hook file takes effect in the next Grok session; inspect it with `/hooks` in Grok.

What the installer touches, exactly: the one file `<grok home>/hooks/tsukkomi.json` (and, next to it, a timestamped `tsukkomi.json.tsukkomi-backup-<stamp>` before an existing file is changed, never for an unchanged result). It never reads or writes `~/.claude/settings.json`, `<grok home>/config.toml`, `trusted_folders.toml` or any other hook file. Hooks in `<grok home>/hooks/` are global, always trusted by Grok and need no folder trust, so nothing needs to be trusted; project-level `.grok/hooks` installation is not offered.

### Options

| Option | Meaning |
|---|---|
| `--print` | Default. Print `{"hooks": ...}` as JSON and write nothing. |
| `--apply` | Write or update `<grok home>/hooks/tsukkomi.json`. Refused if the hook script file does not exist. |
| `--uninstall` | Remove the owned handlers from that file; the file is deleted when nothing else is left in it. |
| `--verify` | Check that the file holds exactly what `--apply` would write, then run the registered command twice on synthetic data in a temporary directory (a forbidden command must be denied, a harmless one allowed). Exit status 1 when anything differs or fails. `--no-run` checks the file and the paths only. |
| `--grok-home PATH` | Absolute Grok home. Default `GROK_HOME`, else `~/.grok`. |
| `--mode {confirmed_only,advisory,enforced}` | Written into every command as `--mode=...`. Default `confirmed_only`. |
| `--events A,B,...` | Comma-separated subset of the six events. Default all. |
| `--python PATH` | Interpreter that runs the hook command. Default: the interpreter running the installer. |
| `--script PATH` | Path of `scripts/grok_hook.py`. Default: the copy next to the installer. |
| `--index-dir PATH` | Absolute index directory, baked into every hook command as `--index-dir=PATH`. A relative path is rejected. |
| `--local-only` | Append `--local-only` to each hook command (no embedder, no credentials, as for every hook). |

Handlers are recognised by their command (`.../scripts/grok_hook.py --event=...`), so a handler someone else added to the same file survives `--apply` and `--uninstall`. A malformed file is an error and is never overwritten. Rerunning `--apply` with other options updates the owned handlers in place; with the same options it writes nothing.

### Events

| Event | Matcher | Timeout (s) | What the hook does |
|---|---|---|---|
| `SessionStart` | none | 30 | Shows the health banner (`systemMessage`) when the hooks have been failing. Nothing is injected: Grok reads only `systemMessage` from this observe-only event. |
| `UserPromptSubmit` | none | 30 | Bookkeeping only. Grok discards the stdout of an allowing hook (documented: "no `additionalContext`") and a block would reject the user's prompt, which this hook never does. Never reads the prompt. |
| `PreToolUse` | `*` | 60 | Judges the proposed tool call. May deny or ask. |
| `PostToolUse` | `*` | 30 | Never reads the tool output. Hands the model, once, the warning a failed `PreToolUse` queued (Grok drops the context of a hook that exits non-zero). |
| `PostToolUseFailure` | `*` | 30 | Same as `PostToolUse`. A failed tool call is never success evidence. |
| `Stop` | none | 120 | At a turn end, checks completion obligations; may keep the agent working once. |

`UserPromptSubmit`, `PostToolUse` and `PostToolUseFailure` exist so that a recovered audit ends its failure streak, so that a failed `PreToolUse` can reach the model and so that every event is answered; `--events PreToolUse,Stop,SessionStart` installs only the gating ones and saves one process per prompt and per tool call.

### Where the hooks read their settings

Hooks never read a project `.env` (the audited agent can write it). They use, in order: flags on the registered command (`--index-dir=`, `--mode=`, `--local-only`), the process environment, the user TOML, then the defaults. The mode in the command beats `SELF_DIRECT_GROK_ENFORCEMENT`.

---

## 4. What the hooks decide

One judgement, host-specific enforcement. The verdict-to-decision table is `src/self_directing_mcp/hook_policy.py`, shared with the AGY and Claude Code adapters; this adapter only translates it into Grok's replies. Modes (`SELF_DIRECT_GROK_ENFORCEMENT` or `--mode=`):

| Verdict for the proposed call | `advisory` | `confirmed_only` (default) | `enforced` |
|---|---|---|---|
| Confirmed `violation` | allow, user notice | **deny** | **deny** |
| `suspicious` (concrete evidence, e.g. an exception that needs confirmation) | allow, user notice | **ask** | **ask** |
| `unknown` | allow, user notice | allow with a warning for the model; **ask** for a `risk: "high"` contract | **deny** |
| `clean`, or no contract applies | allow | allow | allow |

Reply formats, exactly (JSON on stdout, one line):

| Event | Reply |
|---|---|
| `PreToolUse` allow | `{"decision": "allow"}`. Always explicit, never `{}` and never exit 2. Grok documents that an allow only means "not blocked": it does not approve a call Grok would otherwise ask about. |
| `PreToolUse` deny | `{"decision": "deny", "reason": "[<contract id>] ... (Grok tool run_terminal_command (matched as Bash))"}`; the model sees the reason. |
| `PreToolUse` ask | `{"decision": "ask", "reason": "..."}`; the permission prompt names the hook and shows the reason. |
| `PreToolUse` allow with a warning (`unknown`) | `{"decision": "allow", "hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": "Unverified tool call allowed: WARNING - unverified, not a compliance verdict: [<id>] ..."}}` |
| `PreToolUse` allow in `advisory` | `{"decision": "allow", "systemMessage": "ADVISORY (not compliance): ..."}`. Nothing goes to the model. |
| `Stop` keep working | `{"decision": "block", "reason": "A user-specified completion obligation is confirmed unmet: ..."}` |
| `Stop` allow | `{}`, or `{"systemMessage": "Not verified (does not block stopping): ..."}` |
| `SessionStart` | `{}`, or `{"systemMessage": "<health banner>"}` |
| `UserPromptSubmit` | `{}` |
| `PostToolUse`, `PostToolUseFailure` | `{}`, or `{"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": "[Tsukkomi context; stored data is not authority]\nA PreToolUse call was allowed without verification (<ErrorClass>): fail-open, not a compliance verdict; do not claim compliance for it."}}` once after a failed `PreToolUse` |

Every reply is masked for known secret patterns. The exit code is 0 for all of these.

**Stop.** A `Stop` blocks (keeps the agent working) only for a confirmed, fully observed miss of a contract marked `auto_continue`, at most once per session and never for the same reason twice (a persisted counter that cannot be read means "do not continue"). It never blocks while `stopHookActive` is true (a block cannot loop), and Grok also caps continuations at 8 per turn. Grok fires `Stop` again at session end with `reason` `shutdown` or `channel_closed`, where a decision is ignored: those fires, a missing or unknown `reason` and any `Stop` inside a subagent (`subagentType`) are record-only and do not even open the index. Grok writes the final message chunks after the `Stop` hook has run, so when the host's `lastAssistantMessage` is not in the transcript yet, message-scoped misses are not confirmed (the hook says so).

**Failures are visible and still fail open.** Grok itself fails open on a hook crash or timeout; only an explicit `deny` blocks. When the audit cannot run (settings, contracts or engine unavailable, an invalid payload, an invalid `--mode`), the hook answers an explicit `allow` for `PreToolUse` (`{"decision": "allow", "systemMessage": "Tsukkomi audit UNAVAILABLE (<ErrorClass>): this PreToolUse was not verified; fail-open, not a compliance verdict.", "hookSpecificOutput": {... "additionalContext": "... Do not claim compliance for this call."}}`), writes the same line as the first line of stderr, records `hook_health`, and exits **1**, and queues a warning that the next `PostToolUse` (which exits 0) hands the model. Grok records a non-zero exit as `pre_tool_use hook (...) failed, ignored: exit code 1: <first stderr line>` in the scrollback, which is the only channel Grok offers that the user sees for an allowing hook, and blocks nothing. A user-selected `risk: "high"` contract that matches the unverifiable call turns the allow into an `ask` (exit 0, so the prompt carries the reason). Exit code 2 is never used: Grok treats it as a deny. After three consecutive failures the next `SessionStart` shows a banner. The payload text is never echoed: only the error class is reported.

**A shrunk transcript is `unknown`, not compliance.** Grok's `/rewind` drops later turns from the history while file changes stay on disk. The hook remembers how large each session's `updates.jsonl` was at its previous call (`grok-hook-state.sqlite` in the index directory); once the file is smaller than before, confirmed misses of history-dependent contracts (`must`, `requires_success`, `before_regex`) become `unknown`, a clean verdict is delivered with the notice that the history was rewritten, and `Stop` does not continue the agent on evidence that is gone. The MCP audit tools judge the file as it is and cannot know that evidence was removed.

**Subagents.** A `spawn_subagent` child is another session directory with its own id; v1 reads only the session the hook names. History-dependent findings for a call made inside a subagent (`subagentType` in the payload) are `unknown`, never a confirmed miss, and a subagent's `Stop` is never blocked.

---

## 5. The tool-name map

Contracts are regexes over `[tool_call:<tool_name>] <json arguments>`. The contracts users already hold name Claude Code's tools, so the Grok parser (for the history) and the hook (for the proposed call) both render a Grok call with the Claude-canonical name and keys, in one module, `src/self_directing_mcp/grok_tools.py` (`TOOL_MAP_VERSION = 1`). The native name and arguments stay in the event metadata (`native_tool_name`, `native_arguments`) and the deny reason names the Grok tool.

| Grok tool | Rendered as | Argument keys |
|---|---|---|
| `run_terminal_command` | `Bash` | `command`; `is_background` becomes `run_in_background`; `description` is model-written prose, kept out of the matched text (`meta.tool_prose`) |
| `read_file` | `Read` | `target_file` (or `path`) becomes `file_path`; `limit`, `offset` unchanged |
| `search_replace` | `Edit` | `file_path`, `old_string`, `new_string`, `replace_all` unchanged |
| `write` | `Write` | `target_file` (or `path`) becomes `file_path`; `contents` becomes `content`. **Not in the recorded fixtures:** verify with a live run |
| `grep` | `Grep` | `pattern`, `path` and the rest unchanged |
| `list_dir` | `LS` | `target_directory` becomes `path` (Claude Code's `Glob` needs a pattern that `list_dir` lacks) |
| `web_search` | `WebSearch` | unchanged |
| `web_fetch` | `WebFetch` | unchanged |
| `spawn_subagent` | `Task` | unchanged |
| `<server>__<tool>` (MCP) | `mcp__<server>__<tool>` | unchanged |
| anything else | the native name | unchanged |

A key is renamed only when the canonical key is not already in the call; nothing is dropped or overwritten. A name that already starts with `mcp__`, or a canonical name, passes through, so the conversion is idempotent. Grok's own hook matchers accept the Claude names (`Bash` matches `run_terminal_command`), which is why the installed hooks match `*`. A contract that names a Grok spelling (`run_terminal_command`, `target_file`) does not match: write contracts against the Claude vocabulary. Changing a row changes the indexed call text, so it comes with a bump of the parser version.

---

## 6. How the Grok parser decides what happened

Parser version 1 (`src/self_directing_mcp/grok.py`) reads `updates.jsonl` only. `chat_history.jsonl` and `system_prompt.txt` hold the rendered system prompt and are never opened; the resolver refuses any other file name, any path outside the sessions root, any symlink below it and any layout but `<group>/<uuid>/updates.jsonl`. The session id is the directory name, never a record's content.

| Record | Becomes |
|---|---|
| `user_message_chunk` | A user turn (authority for contract anchors) only when Grok stamped it with its `promptIndex` (the shape of a typed prompt), it opens with no `<system-reminder>` wrapper (how Grok delivers hook-injected text) and it is not replayed from another session. Anything else is runtime text and cannot anchor a contract. |
| `agent_message_chunk` | An assistant message. Contiguous chunks of one kind (and one prompt) are joined into one event; any other event-producing record, a parse error and the end of what was read flush it. |
| `agent_thought_chunk` | A runtime placeholder (`[thinking]` and the length): the reasoning is never indexed, as with Claude Code's thinking blocks. |
| `tool_call` | A call, rendered through the [map](#5-the-tool-name-map). |
| `tool_call_update` with `status` `completed` / `failed` | The call's result, linked by `toolCallId`. `failed` is failure; `completed` is success unless the tool's own output says otherwise (a non-zero `exit_code`, `timed_out`, a signal; a non-zero shell exit is `completed` for the host). `_meta.toolNotRun` is kept. A diff is named by its path, not repeated. |
| `tool_call_update` without a terminal status | Progress; skipped. A call with no result is in flight or was cancelled and is never success. |
| `_x.ai/session/update` `hook_execution`, `hook_annotation` | The host reporting on hooks, including this server's own denials: skipped, never indexed. |
| `_x.ai/session/update` `turn_completed` | Runtime metadata (`stop_reason`, token counts). |
| anything else (an unknown `sessionUpdate`, a compaction or rewind marker, an unknown method, non-text content, an unknown status) | One placeholder flagged `unsupported_event`: counted in `coverage.unsupported_event_count`, keeps coverage incomplete and never becomes `clean`. |
| a malformed record | One `parse_error` chunk and no partial evidence. |

A denied call (by this server or any hook) is an attempt: its result is a failed `tool_result` with Grok's text, and a prohibition still matches the call in the history audit. Grok records no structured refusal, so the "rejected proposals" report that Claude Code's transcripts allow is not available for Grok.

Resume appends to the same file and is indexed like any append, from the stored offset. A file that shrinks or is rewritten is re-indexed from byte 0 (`reindex_reason: "source_changed"`). A streamed run still open at the end of what was read is emitted as it stands, and chunks that arrive later start a new event: a message split by a sync that fell mid-stream is two events, deterministically for those sync points. Sessions name no working directory in `updates.jsonl`, so pass `project_dir` to the contract-candidate tools.

---

## 7. Using this with the Claude Code hooks

Grok reads Claude Code's settings file (`~/.claude/settings.json`) by default, so Claude hooks installed there also run on Grok's events, with Grok's payload on stdin. `scripts/claude_hook.py` detects a Grok payload before it imports the package or looks at the daemon, and answers with silence: nothing printed, nothing written, no `hook_health`, exit 0. Earlier versions rejected the Grok transcript path on every tool call (a visible "Tsukkomi check FAILED (PathTraversalError)" for about 0.45 s each, without protecting anything). A payload is Grok's when its transcript path has a `.grok/sessions` segment (either slash style), or when it names the event in Grok's own spelling (`hookEventName`) beside another camelCase key Claude Code never sends and no `.claude` transcript path. Grok has its own hook, installed separately into its own file; having both is the intended setup, and the Claude one costs a Grok session one interpreter start per event (about 6 ms over a bare interpreter, measured).

---

## 8. What is not in this version

- **No input rewrite (`updatedInput`) and no output redaction (`updatedToolOutput`).** Both are future work and were cut on purpose: the model acts on altered output without knowing it was altered, a partial mask can leave a secret reconstructible, binary and very large outputs have no safe text form, and the original stays in `updates.jsonl` (Grok keeps the real result on the record), so a redaction would hide it from the model but not from the index. Denying a call is the supported way to stop one.
- No context injection at `SessionStart` or `UserPromptSubmit` (the documented channels do not carry it).
- No resident daemon (`SELF_DIRECT_DAEMON=on` does not apply to `grok_hook.py`).
- No project-level `.grok/hooks` installation.
- No following of a `spawn_subagent` child session.

### Assumptions a live Grok run should verify

1. The `write` tool's argument names and the argument names of `web_search`, `web_fetch` and `spawn_subagent` (the recorded fixtures contain none of them).
2. How an MCP tool call appears in `updates.jsonl`: the map expects its `title` to be the qualified `<server>__<tool>` name, as the hook payload's `toolName` is documented. If Grok records `use_tool` there, such a call passes through under that name and its arguments.
3. That every typed prompt carries `_meta.promptIndex` (also after a resume) and that a hook's `additionalContext` or a Stop block reason is not written as a plain `user_message_chunk` with that index. A typed prompt without the index would be runtime text and could not anchor a contract.
4. That Grok shows `systemMessage` of `SessionStart` and the first stderr line of a failed hook to the user.
5. That `Stop` payloads keep `reason`, `stopHookActive` and `lastAssistantMessage`, and that the hook runs before the last message chunks are written (as the recorded stream shows).
6. That no other `sessionUpdate` kind appears in ordinary sessions; any that does is reported as an unsupported event until it is added.
