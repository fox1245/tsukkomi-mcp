# Tsukkomi MCP (ツッコミ)

> A deterministic governance and evidence server for autonomous coding agents, version 0.7.0.
> The distribution is `tsukkomi-mcp`; the Python package is `self_directing_mcp`.

## Overview

Prompt instructions alone do not keep an agent from context drift, false completion reports or a forbidden command. Tsukkomi MCP indexes the agent's own session history on the local machine, checks proposed actions and finished work against explicit, user-registered contracts, tracks requirement checklists that need linked evidence, and optionally maintains a session knowledge graph.

Every check returns one of four verdicts: `clean`, `violation`, `suspicious` or `unknown`. `unknown` means the evidence needed for a verdict is missing or has changed, and it is never reported as `clean`. The one incomplete input that can still yield `clean` is a pure append to a growing session, which is judged as of the captured offset and reports that offset (see [Coverage and verdicts](#coverage-and-verdicts)). Whether a verdict blocks anything depends on the host (see [Supported hosts](#supported-hosts-and-parser-versions)). Rules are matched with regular expressions and structured evidence, not with an LLM judge.

```
                  ┌──────────────────────────────────────────────┐
                  │          Autonomous Agent Client             │
                  │ (Claude Code / AGY / Codex / OMP / Grok Bot) │
                  └──────────────────────┬───────────────────────┘
                                         │
                      Lifecycle hooks (host dependent) or MCP calls
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                           Self-Directing MCP Engine                            │
├───────────────────────┬────────────────────────┬───────────────────────────────┤
│  Invariant Contracts  │   Verifiable Evidence  │      Session GraphRAG         │
│  - must / must_not    │   Checklist Gate       │  - Entity dependencies        │
│  - Session isolation  │   - No done without    │  - NeoGraph engine            │
│  - Regex + temporal   │     linked evidence    │  - Context injection (AGY)    │
├───────────────────────┴────────────────────────┴───────────────────────────────┤
│                 Local session index (SQLite, FTS5, sqlite-vector)              │
│     - Cosine vector search  - FTS5 BM25 and regex  - Known-secret masking      │
└────────────────────────────────────────────────────────────────────────────────┘
```

Session evidence stays on the local machine. Remote calls are limited to embeddings, GraphRAG extraction and the opt-in workflow classifier, all sent to OpenRouter. The canonical index keeps raw text so audits see the truth; everything that leaves it (tool results, errors, hook output, logs, remote payloads) is masked for known secret patterns, and the workflow classifier refuses context in which a secret pattern is found. See the [masking contract](SECURITY.md#masking-contract), [SECURITY.md](SECURITY.md) and [Verification scope](#verification-scope).

## What is new in 0.7.0

Version 0.7.0 (see [CHANGELOG.md](CHANGELOG.md)) mainly makes big sessions converge under the request deadlines. It was found by reproducing the report that timeouts also happen without an external model; the design record is [#26](https://github.com/fox1245/tsukkomi-mcp/issues/26).

- **A sync that cannot finish keeps its progress.** A request whose budget (8 s for hooks, `audit_session` and `check_action`; 45 s for the other tools) is too short for a cold or appended source stops at a record boundary and publishes the prefix it parsed; the next call continues. A 300,000-event synthetic session that never converged now converges in 5 calls. See "Large sessions and deadlines".
- **A re-index keeps its progress without touching the index that exists.** A parser-version bump, a changed source or changed AGY receipts are staged in a spool next to the index and published by one ordinary commit; an interrupted re-index leaves the old events, anchors and vectors as they were.
- **An audit of a big session converges.** A slice is sized by what judging it costs, a comparison never takes the answer with it, and a resident server finishes what a deadline cut.
- **A sync that runs out of time no longer takes the judgement with it.** The stored index is judged with `sync_deadline` or `sync_in_progress` in the coverage issues: nothing is `clean`, a confirmed `must_not` match is still a `violation`.
- **`audit_status.ops` and `audit_status.evaluation`** say how each tool's requests ended and how far a session's monitor states lag, even while the engine is busy.
- **Host-side GraphRAG extraction** (`graph_next_batch`, `graph_submit_batch`, `SELF_DIRECT_GRAPH_EXTRACTOR=host`) and an O(batch) read of a graph batch.
- **Incremental contract audits, source integrity and the adaptive shadow comparison** (see their sections), an opt-in resident daemon (`SELF_DIRECT_DAEMON=on`) and an opt-in write-ahead journal (`SELF_DIRECT_SQLITE_JOURNAL=auto`).
- **Two quadratic costs removed.** SQLite planned `chunk_id IN (...)` lookups as a scan of the whole session; a derived pass over 150,000 events went from 64 s to 7.7 s and appending 100,000 events to a session of 100,000 from 12.7 s to 3.4 s.

---

## What is new in 0.6.0

Version 0.6.0 (see [CHANGELOG.md](CHANGELOG.md)) mainly fixes memory growth, append handling and concurrency in long-running servers. It also adds a Claude Code integration, a shared hook policy and an egress masking contract.

- **Dependency:** `neograph-engine==0.14.1` (was 0.12.1). `zstandard>=0.23` is new on Python before 3.14, for compressed Codex sessions.
- **Compressed Codex sessions.** Codex CLI 0.156 compresses older rollouts in place to `rollout-*.jsonl.zst`. They are discovered, synced, audited and searched like plain rollouts, streamed through a bounded decoder and never held in memory. See [Compressed Codex sessions](#compressed-codex-sessions).
- **Appending sessions sync normally ([#4](https://github.com/fox1245/tsukkomi-mcp/issues/4)).** Source capture streams a SHA-256 over the file up to its length at the first measurement and keeps no bytes. Records appended afterwards are reported as `unread_bytes` instead of failing the sync.
- **Audits judge a growing session as of the captured offset.** Results carry `judged_through_byte`, `unread_bytes` and `judged_prefix`; absence-based `must` obligations stay `unknown` while bytes are unread. See [Coverage and verdicts](#coverage-and-verdicts).
- **`check_action` does not trust older history for a proposed action ([#8](https://github.com/fox1245/tsukkomi-mcp/issues/8)).** A prerequisite that requires success stays `unknown` while required history is unread, because a newer failed run may be in the unread part.
- **Bounded memory ([#5](https://github.com/fox1245/tsukkomi-mcp/issues/5)).** Single-flight sync per session, streaming reparse, per-session id queries, float32 vectors, a cap on concurrent heavy calls (`SELF_DIRECT_MAX_CONCURRENT_CALLS`, default 4) and `malloc_trim` on glibc. See [Memory and concurrency](#memory-concurrency-and-deadlines).
- **Concurrent checklist updates ([#6](https://github.com/fox1245/tsukkomi-mcp/issues/6)).** `update_checklist` serializes plain edits and returns a `checklist_conflict` status instead of raising or overwriting.
- **Duplicate session sources ([#10](https://github.com/fox1245/tsukkomi-mcp/issues/10)).** Copies and continuations of an indexed session are recognised instead of reparsed on every sync (`alias_of`, `ambiguous_session_source`).
- **Parsers.** OMP parser v8 (host notices, images, steering attribution, `session_init`, `model_usage`, `branch_summary`, `redactedThinking`; [#9](https://github.com/fox1245/tsukkomi-mcp/issues/9)) and Codex parser v3 (multi-agent messages, tool search, new `session_meta` fields). Each bump reparses lazily, one session at a time.
- **CI.** A new Linux `memory-audit-regressions` job runs small offline memory, freshness and concurrency tests next to the existing smoke job.
- **Claude Code ([docs/claude-installation.md](docs/claude-installation.md)).** A `claude` provider (parser version 2) for Claude Code session and subagent transcripts, and opt-in command hooks installed by `scripts/install_claude_hooks.py` for `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `Stop` and `SubagentStop`. By default (`SELF_DIRECT_CLAUDE_ENFORCEMENT=confirmed_only`) they deny only a confirmed `violation` and ask on `suspicious`; they never answer `allow`, `SubagentStop` only reports, and every internal error fails open with a visible warning.
- **One hook policy for AGY and Claude Code ([docs/agy-installation.md](docs/agy-installation.md)).** The AGY hook default is now `confirmed_only` (it was `enforced`): `unknown` passes with a visible warning unless the contract is marked `risk: "high"`, a `Stop` continues the agent at most once per conversation and only for a confirmed `auto_continue` miss, and hook errors fail open with a `hook_health` banner. `enforced` stays an explicit opt-in, and `advisory` never gates an approved workflow either.
- **AGY parser version 7.** Truncated fields fall back to the same step of `transcript_full.jsonl`; what stays cut is `unknown`, never `clean`. `RUNNING`, `ERROR`, `INVALID`, `CHECKPOINT` and media records are handled, and results from before a conversation's first hook receipt are classified as `legacy_unlinked`. Version 7 adds the Antigravity IDE's typed result steps (`RUN_COMMAND` with its host `exit_code`, `VIEW_FILE`, `CODE_ACTION`, `MCP_TOOL`, `LIST_DIRECTORY`) and context-loading steps (`CONVERSATION_HISTORY`, `KNOWLEDGE_ARTIFACTS`), and accepts a host-written run of NUL bytes in front of an otherwise valid record.
- **Contract fields `risk`, `auto_continue` and `when`**, and four disabled high-risk presets. See [Contract fields](#contract-fields-risk-auto_continue-and-when).
- **Masking contract.** Every egress path is masked, the sparse text index is masked for new rows, and a logging filter covers server logs. See the [masking contract](SECURITY.md#masking-contract).
- **Also in this release since 0.5.0:** the shared-index snapshot and request-deadline work ([#3](https://github.com/fox1245/tsukkomi-mcp/issues/3)), the AGY evidence workflow and approved-workflow layer, user path configuration in TOML, and a default index directory outside the checkout. Read [Upgrading to 0.6.0](#upgrading-to-060) before restarting existing servers.

---

## Key capabilities

### 1. Invariant contracts (`upsert_contracts`)
- **Prohibitions (`must_not`)** match a regex against proposed or recorded actions (`decision: deny` on hosts that enforce).
- **Obligations (`must`)** require matching evidence before a gated action (`before_regex`) or at a completion checkpoint, optionally with a successful linked tool result (`requires_success: true`).
- **A contract's origin must be a user message.** `source_event_id` names the history event a contract was derived from, and only the user can be the origin of one: a tool result, runtime text or injected context cannot. In an audit, a `source_event_id` that resolves to anything but a user-role event (`meta.role == "user"`) makes that contract's findings `unknown` with the reason `anchor_not_user` (reference evaluator and monitors alike); at upsert time such an anchor is rejected with an error when the event is already indexed, and accepted when it cannot be resolved yet (the session is not indexed), leaving the audit to decide. What counts as a user message is each parser's own classification: Claude Code `origin.kind == "human"` (plus a legacy prompt from before origin stamps), AGY `USER_INPUT` from `USER_EXPLICIT`, OMP a user-role message not attributed to an agent. Codex cannot tell a typed prompt from user-role context injected into the model's input, since it records both as user turns, so both are accepted. `applies_from_event_id` may still be any event. The upsert response reports each anchor check (`anchor_checks`); it is not stored on the contract.
- **Weakening a contract needs a user anchor.** Disabling a contract, or replacing it with one that may enforce less (a different regex, an added or changed `exception_regex`, a different `before_regex`, a narrower scope, fewer roles, an added `when`, a narrower provider or session, dropping `requires_success`, lowering `risk`, dropping `auto_continue`), must carry a `source_event_id` that resolves to a user message. Without one the proposed revision is stored under `pending` in the contracts file and listed by `list_contracts`, the previous revision stays active (hooks and audits keep enforcing it), and the response says `NOT APPLIED`. Repeating the change with a user anchor applies it and clears the pending entry. Strictly tightening edits (a new verifier, an added role, a wider scope, removing an exception, enabling) need no anchor, and moving `applies_from_event_id` is not judged. The `pending` key is omitted from the file while empty.
- **A call the host refused is a rejected proposal, not an executed violation.** When a hook or the user's permission prompt denies a tool call, the transcript still holds the call. Claude Code records its decision on the record of the linked tool result: a top-level `permissionDecision` object (kept as `meta.permission_decision`) beside `toolDenialKind`, with an error result. A decision of `{"decision": "reject", "source": "hook", ...}` (whatever its `source`: hook, user, rule, ...) marks its call as refused, provided that call is the only one with that id; the older forms `"deny"` and `{"behavior": "deny"}` are kept. This shape was verified against Claude Code 2.1.299 with a synthetic probe (a `PreToolUse` hook answering `deny` for `Bash`). Any other value (`accept`, `allow`, an unknown or missing decision) is an attempt. A `must_not` match in a refused call is not an executed match: it is reported separately in the finding's `proposals_rejected` (`contract_id`, `event_id`, `call_id`, `reason`, `denied_by` (the decision's `source`, or `host`), `denial_event_id`, `at_byte`; omitted while empty; at most 20), the verdict comes from the remaining evidence (`clean` with the reason `no executed forbidden match ...` when every match was refused), and an `exception_regex` is matched only against what ran. Only the host's own record counts: a result that merely says error, or a decision of `allow` or `ask`, is still an attempt. AGY already treated a model proposal that the Pre hook denied (`host_rejected_proposal`) as non-execution; it is now also listed in `proposals_rejected` (`tool_name`, `at_byte`, `denied_at_byte`). Codex, OMP and Cursor / Grok Bot have no host-recorded denial that their parsers can read reliably (Codex hooks never block; the OMP extension's block leaves no marker the parser knows), so a blocked call there stays an attempt. A `must` obligation and a rule that reads tool output (`scope` `tool_result`, or `any` with the `tool` role) are unchanged: a result that echoes the command still matches. The Claude Code and AGY `Stop` notes list rejected proposals on their own line ("Rejected by the host, not executed ..."), apart from "Not verified" and from any executed violation.
- **`requires_success` needs a success that belongs to the required command.** A host records one exit status per command line. `pytest -q || true`, `|| :`, `pytest; exit 0` (any later command that replaces the status), a command sent to the background (`&`, or Claude Code's `run_in_background`), `pytest | tee log` without `set -o pipefail` earlier in the same line, and `pytest --collect-only` / `--co` all end in a recorded success without the tests having passed, so that success is not evidence: the finding is `unknown` with the reason `success_unverifiable` (never `clean`, and never a violation). A result recorded as failed is still a confirmed failure (`violation`), a call with no linked result keeps its `no unambiguous completion status` reason, and the command itself is never blocked. The check is a small quote-aware parse of the line (`audit/shellstatus.py`, also reading `bash -c '...'`, `( ... )` and `{ ...; }`): an operator inside quotes, `$( )` or a comment is not an operator, `set +e` alone changes nothing, `|| exit 1` and `; exit $?` keep a failure, and the last command the contract's regex matched is the one whose status counts. The latest matching call still decides, so a newer masked run supersedes an older clean one. Both evaluators call the same function.
- AGY returns `force_ask` for suspicious exceptions so a cached Always Allow grant does not replace confirmation; Claude Code, which has no force-ask, returns `ask`.
- Contracts are global or scoped to one session. A description-only contract stays `unknown`, and search rankings are never violation evidence.
- **A tool-call contract matches the payload, not the model's commentary.** Some hosts let the model write a free-text field about a call: Claude Code's `description` on `Bash` (and on the other built-in tools that take one), AGY's `toolAction` and `toolSummary`. That prose is excluded from the call text a contract is matched against, in indexed history and in the proposed action a hook asks about alike (one builder, `tool_payload.py`), so `ls -l data/app.sqlite3` is no longer a `sqlite3` violation because its description says "without connecting to it with sqlite3". Only confirmed prose fields are excluded and only when their value is a string; every other field stays payload, an MCP tool's own `description` argument included, and a violation inside `command` is caught whatever the description says. The prose is kept in the event's `meta.tool_prose`. Matching it with an explicit `scope: "prose"` (at most `suspicious`) is not implemented yet. Fields suspected but not confirmed (AGY `Description`/`Instruction`, Codex `justification`, OMP `_i`) are deliberately not excluded.

#### Contract fields: `risk`, `auto_continue` and `when`

Three optional fields refine how hosts act on a contract. They never change what a regex matched. At their defaults they are omitted from the stored contract, so existing `contracts.json` files load and are rewritten unchanged and older processes keep reading files that do not use them.

| Field | Values | Applies to | Effect |
|---|---|---|---|
| `risk` | `normal` (default) or `high` | any contract | Declares that the actions the contract applies to are high-risk. When such an action cannot be verified (`unknown`), the AGY and Claude Code hooks ask the user instead of silently allowing it (`confirmed_only`), including after an internal hook error unless the mode is `advisory`. It does not change the verdict. Codex hooks are advisory and the OMP extension blocks only confirmed findings, so `risk` does not change their behavior. |
| `when` | `{"tool_call_regex": "<regex>"}` | `must` only | The obligation applies only if some tool call observed earlier in the same session matches the regex (case-insensitive, multiline). Otherwise the finding is reported as `clean` with `applicability: "not_applicable"` and a reason, never dropped. While coverage is incomplete, an unmet condition becomes `unknown` because the call may be in unread evidence. An AGY model proposal counts as the condition being met. |
| `auto_continue` | `true` or `false` (default) | `must` with a `regex` and no `before_regex` | A confirmed, fully observed miss of this completion obligation may resume a stopping agent: once per conversation on AGY (`continue`) and once per session on Claude Code (`block`). Unknown evidence, partial coverage and hook errors never resume it. Any other contract shape is rejected when it is saved. |

A high-risk `must_not` must have a `regex` naming the actions it covers: a description-only one would apply to every tool call and could never be verified, so every call would be asked (a denial in headless runs). `upsert_contracts` rejects it.

Contract regexes (`regex`, `before_regex`, `exception_regex`, `when.tool_call_regex`) and `search_history` regex queries run on the `regex` engine with a time limit: 0.25 s per search and 2 s per contract per evaluation. A pattern that exceeds it makes only that contract `unknown` (`regex_timeout`), never clean or a violation, so a pathological pattern cannot stall a hook past the host's timeout. The syntax is still Python `re` syntax.

Set these only when the user asked for them. The high-risk presets in [`src/self_directing_mcp/resources/high_risk_presets.json`](src/self_directing_mcp/resources/high_risk_presets.json) are four example `must_not` contracts with `risk: "high"`: `preset-no-recursive-delete`, `preset-no-write-outside-workspace` (a path-pattern approximation to replace with your own boundary), `preset-no-force-push` and `preset-no-branch-delete`. They ship with `enabled: false` and exist in the contract store only when `SELF_DIRECT_SEED_EXAMPLE_CONTRACTS=true`, where they are added if their id is missing and are never altered or re-enabled. The owner enables the ones they want.

#### Contract fragility report (informational)

A regex over command text is easy to bypass with an equivalent spelling (`rm -r -f` for `rm -rf`), channel (`python3 -c "shutil.rmtree(...)"`) or lexical trick (a backslash-newline, `busybox rm`, a GNU option abbreviation). The `upsert_contracts` response therefore ends with a `fragility` section that says which of those spellings the regex you just saved does **not** match.

- **How it is chosen.** The server holds a versioned library of equivalent spellings ([`resources/contract_variants.json`](src/self_directing_mcp/resources/contract_variants.json)) in ten families: `recursive_delete`, `force_push`, `discard_worktree`, `world_writable`, `remote_script_exec`, `read_secret_file`, `stage_all`, `db_cli_access`, `skip_tests` and `tls_verify_off`. A family is reported only when the regex matches one of its canonical (standard) spellings; the section then lists that family's other spellings the regex misses, with their platform, technique and preconditions. Families the regex is not about are not mentioned. For a `must_not` contract the `regex` is analyzed; for a `must` it is the gated action (`before_regex`).
- **Informational only.** It never blocks a registration, never edits or broadens a regex and never changes a verdict: findings still come from the saved regex alone. Use it as a prompt to widen the regex deliberately, if you want to. `exception_regex` and `when` are not analyzed.
- **Unverified candidates.** Every library entry has `verified: false`: equivalence was not verified by execution, and the response says so (`"verified": false` and a note). A listed spelling may behave differently on your platform or tool version; read its preconditions.
- **Deterministic and offline.** No model is involved. The report names the `library_version` that produced it. Each spelling is matched as plain command text with the contract regex engine (case-insensitive, multiline, the same time limits). A regex that exceeds them produces an explicit `"status": "regex_timeout"` entry for that contract, never a failure of the save.
- **Bounded.** At most 8 spellings per family (`missed_total` counts all misses, `omitted` the unlisted ones; a capped list takes one spelling per technique in turn), 4 families per contract (`families_omitted`), 20 contracts per call (`contracts_not_analyzed`) and about 3 s per call (`skipped_time_budget`). A contract without such a pattern, or whose regex matches every listed spelling, is omitted.
- **Hook paths never load it.** Only `upsert_contracts` imports the library, after the contracts are saved. If the library cannot be read, the response carries `"fragility": {"status": "unavailable", ...}` and the contracts are still saved.

Abridged example for `{"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": "rm\\s+-rf"}`:

```json
"fragility": {
  "library_version": "1.0.0", "informational": true, "verified": false, "analyzed": 1,
  "contracts": [{"contract_id": "no-rm", "field": "regex", "status": "ok", "families_omitted": 0, "families": [{
    "family": "recursive_delete", "matched_canonical": ["rm -rf build"], "missed_total": 19, "omitted": 11,
    "missed_variants": [
      {"id": "recursive_delete.split_flags", "text": "rm -r -f build", "platform": "posix",
       "technique": "option_split", "verified": false},
      {"id": "recursive_delete.python_rmtree", "text": "python3 -c \"import shutil; shutil.rmtree('build')\"",
       "platform": "python", "technique": "interpreter", "verified": false, "preconditions": ["python3 on PATH"]}]}]}]}
```

### 2. Checklist with evidence gating (`update_checklist`)
- Track user requirements with provenance and completion criteria.
- A completion claim without evidence is downgraded to `pending_verification`. An item becomes `done` (`verified=true`) only when its `evidence_chunk_ids` resolve to successful tool results linked to their calls **and** session coverage is complete. While appended bytes are unread, coverage is incomplete, so the item stays `pending_verification` until a later call sees the caught-up history.
- Concurrent updates do not lose each other's edits. An update that needs evidence evaluation is retried up to three times; if the checklist or evidence keeps changing, the call returns `{"ok": false, "status": "checklist_conflict", "action_executed": false}` and saves nothing.

### 3. Session GraphRAG and context injection (`get_graph_context`)
- Entity dependency graph (nodes, relations, cursors) in SQLite, executed through the **NeoGraph** engine.
- Typed relations: `IMPLEMENTS`, `DEPENDS_ON`, `MODIFIES`, `PRODUCES`, `CHECKS_VERSION`, `EVIDENCED_BY`, `SUPERSEDES`.
- Graph updates are validated against session history and applied idempotently. Updates require an explicit call, `run_neograph_update` or `graph_next_batch` plus `graph_submit_batch`; lifecycle hooks never send session text to a model.
- The extraction can also be done by the calling agent instead of an OpenRouter model: [Host-side GraphRAG extraction](#host-side-graphrag-extraction).
- In Google Antigravity, active graph relationships are delivered into agent context before each tool invocation.

### 4. Hybrid history search
- Dense cosine similarity (`sqlite-vector`, native, by default; NumPy or PostgreSQL with pgvector when selected: see [Dense backends](#dense-backends)), SQLite FTS5 BM25 and regex, fused with reciprocal-rank fusion.
- **Secret masking:** known credential patterns are masked before embedding, in sparse search text and in everything returned to the client. This is not a guarantee that arbitrary personal data or every secret format is removed; see the [masking contract](SECURITY.md#masking-contract).

---

## Supported hosts and parser versions

| Host | Provider id | Evidence source | Lifecycle integration | Blocks actions? | Parser version |
|---|---|---|---|---|---|
| Google Antigravity (AGY) | `agy` | `brain/<conversation-id>/.system_generated/logs/transcript*.jsonl` plus hook receipts | Stdio hooks: `PreToolUse`, `PostToolUse`, `PreInvocation`, `Stop` | By default (`confirmed_only`) only a confirmed `violation` (`deny`) and `suspicious` (`force_ask`); `unknown` passes with a warning, or asks for a `risk: "high"` contract. `advisory` and `enforced` are explicit. See [docs/agy-installation.md](docs/agy-installation.md) | 8 |
| Claude Code | `claude` | `<projects>/<project>/<session-uuid>.jsonl` plus `<session-uuid>/subagents/agent-*.jsonl` | Opt-in command hooks from `scripts/install_claude_hooks.py`: `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `Stop`, `SubagentStop` | By default (`confirmed_only`) only a confirmed `violation` (`deny`) and `suspicious` (`ask`); never an explicit `allow`. A `Stop` blocks at most once per session for a confirmed `auto_continue` miss. `SubagentStop` never blocks. See [docs/claude-installation.md](docs/claude-installation.md) | 3 |
| OpenAI Codex CLI | `codex` | Rollout JSONL in the Codex sessions directory, plain or zstd-compressed (`rollout-*.jsonl.zst`) | Three `mcp_tool` hooks installed by `scripts/install_codex.py`: `UserPromptSubmit`, `PreToolUse`, `Stop` | No. Hooks are advisory | 3 |
| OMP | `omp` | OMP session JSONL | Opt-in extension `extensions/tsukkomi-omp.ts` | Only confirmed `violation`/`suspicious` on model-issued tools. `unknown` and transport failures never block | 8 |
| Cursor / Grok Bot | `grokbot` | `agent-transcripts` roots | None; call the MCP tools manually | No | 1 (unversioned) |

A parser version is recorded per session. When a provider's version is raised, each already-indexed session is reparsed from byte 0 on its next sync or audit, one session at a time. Unchanged event ids, anchors and cached vectors survive, and no embedding API is called when `embed=false`. A running MCP process only picks up a new parser after it is restarted.

### AGY parser (version 8)

Fields the host cut (`truncated_fields`) are restored from the same step of the sibling `transcript_full.jsonl` when it matches; otherwise the visible text is kept, marked as truncated, and any verdict that depends on the cut field is `unknown` with a `truncated_<field>` reason. A match that is visible still counts. `RUNNING` results are pending (`pending_running`), `ERROR`, `INVALID` and `CANCELED` results are never success, `CHECKPOINT` and `ERROR_MESSAGE` records are runtime context, and media payloads become placeholders. Results from before the conversation's first hook receipt are `legacy_unlinked` and leave dependent obligations `unknown` (`unsupported_legacy`); coverage reports `receipts_since`, `legacy_unlinked_count` and `result_dependent_complete`. Details are in [docs/agy-installation.md](docs/agy-installation.md#parser-version-6-truncated-fields-result-statuses-and-legacy-results).

Version 8 keeps the model-written `toolAction` and `toolSummary` labels of a tool call (they sit beside `CommandLine` in the recorded fixtures) out of the call text, for hook receipts and for model proposals, and records them in `meta.tool_prose`; every other argument stays payload.

Version 7 reads two more shapes. The IDE writes a tool result as a step of its own type, `RUN_COMMAND`, `VIEW_FILE`, `CODE_ACTION`, `MCP_TOOL` or `LIST_DIRECTORY`, and the parser treats it exactly like the CLI's `GENERIC` result: it is linked to its call only through a host receipt, truncated content falls back to `transcript_full.jsonl` or leaves dependent verdicts `unknown`, and `RUNNING` is pending. A linked `RUN_COMMAND` with status `DONE` and host `exit_code` 0 is success evidence, a nonzero `exit_code` is a failure, and an unlinked result is never success. `CONVERSATION_HISTORY` and `KNOWLEDGE_ARTIFACTS` are runtime metadata. A contiguous run of NUL bytes at the start of a line is stripped when the rest is a valid record (`host_padding_bytes`, counted in `agy_evidence.host_padded_records`); any other NUL use, and any other unknown step type, stays a parse error or an unsupported event. See [docs/agy-installation.md](docs/agy-installation.md#parser-version-7-ide-result-steps-context-steps-and-host-padding).

Hook receipts cost what was appended, not the size of the transcript. A receipt hashes the file once and parses only the records after a prefix it has verified; a sync checks every receipt anchor in the pass that hashes the file; and a receipt reindexes the conversation only when it can change an event that is already indexed. On a synthetic 10 MB transcript with a linked receipt per result, one `PreToolUse` hook (receipt and contract check) fell from 20.5 s to 0.51 s and a `PostToolUse` hook (receipt and sync) from 20.4 s to 0.58 s, against 0.38 s and 0.36 s at 0.1 MB (whole hook processes, wall time). The mechanisms and what they fall back to are in [AGY hook cost](docs/agy-installation.md#hook-cost-what-a-hook-reads); `scripts/bench_agy_hooks.py` reproduces the table.

### Claude Code parser (version 3)

Only a record the host stamped `origin.kind == "human"` (or, in a main transcript, a legacy prompt from before origin stamps, or a human `queued_command` prompt) is a user turn. Task notifications, peer and coordinator messages, injected meta messages, compaction summaries, attachments, hook output, tool results and a subagent's task prompt are runtime evidence. Unknown record, attachment, block and origin shapes are unsupported events that keep coverage incomplete. Opaque payloads (thinking, images, hook output) are placeholders. A subagent transcript is its own session, `<session-uuid>_agent-<agent-id>`. Version 3 keeps the model-written `description` of a built-in tool call (`Bash`, `Task`, `Monitor`, ...) out of the call text and records it in `meta.tool_prose`; the call text contains only the other input fields, exactly like the proposed action a `PreToolUse` hook asks about. An MCP tool's own `description` argument is payload. Version 2 accepts the `system` record `scheduled_task_fire` (a scheduled task fired) as runtime metadata: its `prompt` is automation text, not something the user said, so it is validated, never indexed, and cannot satisfy or violate a user-role contract; every other unknown system subtype stays unsupported. See [docs/claude-installation.md](docs/claude-installation.md#9-how-the-claude-parser-decides-who-is-the-user).

### OMP parser (version 8)

Runtime records are indexed as metadata or `runtime`-role events. They are never user instructions and never proof that a tool ran.

- Host notices, `mode_change`, `ttsr_injection`, developer messages, `session_init`, `model_usage` and `branch_summary` are runtime metadata. Only the parent's task text of `session_init` is searchable; its prompt, tools and schema are not indexed.
- Image blocks become `[image:<mime-type>]` placeholders and `thinking`/`redactedThinking` blocks become placeholders; base64 and opaque reasoning payloads are never indexed, including inside blocks of an unsupported shape (only their key names are kept).
- **Attribution:** among steering messages only `attribution: "user"` is a user turn. Any user-role message with `attribution: "agent"`, with or without steering metadata, is runtime text and cannot satisfy or violate a user-role contract.
- Partial steering metadata (`steering` or `liveSteered` without the full valid shape) and unknown attribution values are rejected as parse errors, which makes coverage incomplete.
- Unknown event shapes and unmatched execution-start markers still make coverage incomplete. `service_tier_change` is metadata only when `serviceTier` is `null` or a map from the known providers (`openai`, `anthropic`, `google`) to known tiers.

### Codex parser (version 3)

- `response_item` `agent_message` (multi-agent messaging) and `inter_agent_communication_metadata` are runtime metadata, neither user authority nor tool evidence. `encrypted_content` blocks become a placeholder.
- `tool_search_call` / `tool_search_output` are runtime metadata. The search query is indexed; the returned `tools` list is not.
- `session_meta` headers, including the newer fields, are shape- and type-checked, and only a fixed subset of identifying fields is indexed. `base_instructions`, `dynamic_tools`, `context_window`, `git`, `history_base`, account ids and workspace roots are never indexed. The older behaviour of indexing a legacy header's whole payload is gone.
- A complete new-style `session_meta` for another thread in the middle of a sub-agent file (an inherited parent header) is accepted as inherited metadata. A first-line id mismatch is still an error.
- For these newly supported record types, an envelope with `metadata.client_authored: true` is treated as unsupported because its meaning is unclear.
- Earlier support remains: `world_state`, `token_usage_record`, `compacted` and native `web_search_call` records. Unknown event types stay visible as unsupported.

### Compressed Codex sessions

Codex CLI 0.156 compresses older rollouts in place: `sessions/YYYY/MM/DD/rollout-<timestamp>-<uuid>.jsonl.zst`, one zstd frame that declares its content size. A forked or sub-agent rollout may carry a second `_<uuid>` suffix. Session discovery, `session_id` lookup and explicit `path` arguments accept `rollout-*.jsonl.zst` under the configured Codex root with the same root, symlink and traversal checks as `.jsonl`. The session id comes from the `session_meta` header, as for plain files; a file name with two different UUIDs never names a session by itself. While a plain rollout and its compressed twin both exist, a lookup by id returns the plain (live) one.

- **Offsets address the decompressed stream.** Event ids, anchors, `byte_offset`, `judged_through_byte` and the stored prefix hash are exactly what the plain file with the same bytes yields, so Codex compressing a rollout in place, or a resume decompressing it and appending, is the same session: nothing is reparsed, no event id, anchor or cached vector changes, and no duplicate session or `ambiguous_session_source` appears. If the compressed file holds more records than the plain file had when it was indexed, only the new part is parsed. Plain and compressed copies that exist together follow the [duplicate source](#duplicate-session-sources) rules.
- **A compressed file is an immutable snapshot.** There is no growing tail: coverage is complete when the stream has been read to its end (`unread_bytes` stays absent unless the file ends with an unterminated record, which is never parsed). A changed compressed file is a rewrite like any other: identical decompressed content (for example recompressed at another level) is no change, a changed earlier record is reindexed (`source_changed`), and more records after an identical prefix are a continuation. Growth of the compressed file itself is never treated as pure append.
- **Streaming only.** The decoder reads in bounded blocks; a cursor in the middle of a compressed source is reached by decoding and discarding from the first byte, so memory does not depend on the session size. On a 54.5 MB synthetic session (28 MB compressed) a full sync peaks at the same traced memory as the identical plain file, 16 MB, and an audit-style refresh at 3.5 MB; a 60 MB stream that compresses to 6 KB is decoded with a 2.4 MB peak. Source observation validates a compressed snapshot by the SHA-256 of the compressed bytes (no second decode); each audit or status check of an indexed compressed session decodes the file once.
- **Strict frames and visible errors.** A compressed source must be exactly one zstd frame that declares its content size, without a dictionary and with a window of at most 128 MiB (zstd's own default). Anything else, and every unreadable file, is an explicit error with a code, never a skipped file or a short read: `zstd_corrupt`, `zstd_truncated`, `zstd_size_mismatch` (more or fewer bytes than the header declares), `zstd_unsupported_frame`, `zstd_too_large` and `zstd_unavailable`. `sync_session` returns `ok: false` with that `error`, nothing is indexed or changed, and an audit is `unknown` with the code in `coverage.issues`. A source that was indexed and later goes bad reports the code in coverage instead of `source_unavailable_or_truncated`.
- **Size limit.** `SELF_DIRECT_MAX_DECOMPRESSED_SESSION_BYTES` (default 4 GiB, about ten times the largest observed session) bounds the size the frame header declares; a larger file is refused as `zstd_too_large` before any byte is decoded, and a stream that disagrees with its header stops at the declared size. A plain `.jsonl` has no such limit.
- **Decoder.** Python 3.14+ uses the standard library `compression.zstd`; older interpreters use the `zstandard` package (`zstandard>=0.23`, installed with the project). Without either, a `.zst` source reports `zstd_unavailable` and plain `.jsonl` sessions keep working. The Codex parser stays at version 3: the parsed records and event ids are unchanged, so existing indexes are not reparsed.

---

## MCP tools reference

| Category | Tool | Description |
|---|---|---|
| **Governance & Contracts** | `upsert_contracts` | Create or update explicit invariant rules (`must` / `must_not`). The response adds an informational [fragility report](#contract-fragility-report-informational): equivalent spellings the saved regexes miss. |
| | `list_contracts` | Inspect active contracts for the current session and global scope, plus any `pending` weakening awaiting a user anchor. |
| | `revoke_contract` | Disable a contract while preserving history. Takes `source_event_id` (the user's message that asked for it); without one the request is held as pending and the contract stays in force. |
| | `check_action` | Pre-flight test of a proposed action against current contracts. `verify="fast"` selects the hook-time source verification; the result's `integrity` says what was compared. |
| | `audit_session` | Audit executed session history against active contracts (always a full source verification). |
| | `audit_status` | Health check: vector extension, index schema, degraded flags, per-session coverage, and `ops`, the [request statistics](#timeouts-and-what-a-deadline-leaves-behind) of the process (answered even while the engine is busy). |
| **Checklist & Evidence** | `update_checklist` | Update requirements; enforces evidence gating for completion. |
| | `get_checklist` | Inspect checklist status, history and linked evidence. |
| **GraphRAG & Architecture** | `get_graph_context` | Query entity dependencies and the active knowledge graph. |
| | `propose_graph_update` | Validate node/edge additions with evidence chunk requirements. |
| | `commit_graph_update` | Atomically commit validated graph updates with cursor progression. |
| | `run_neograph_update` | Run the LLM GraphRAG extraction pipeline through NeoGraph. |
| | `graph_next_batch` | Next batch of events for the **calling agent** to extract (no model call, key use or network access here); `after_order` looks ahead, `max_chars` and `max_events` size the result. See [Host-side GraphRAG extraction](#host-side-graphrag-extraction). |
| | `graph_submit_batch` | Validate and commit the agent's extraction of one batch, in order and idempotently; a batch without relations needs a recorded reason. |
| **Search & Analytics** | `sync_session` | Incremental evidence sync for Codex, Grok Bot, OMP, AGY and Claude Code transcripts. A session too large for the request budget is indexed in slices: the result's `sync` object says how far it got ([Large sessions and deadlines](#large-sessions-and-deadlines)). |
| | `search_history` | Hybrid retrieval (dense vector + BM25 + regex) over history. |
| | `get_chunk` | Retrieve a raw evidence chunk with secret masking. |
| | `analyze_activity` | SQLite time-series analysis: retry gaps, error bursts, stalled items. |
| **Hooks** | `codex_session_hook` | Adapter for Codex lifecycle events and local audit context. |
| **Approved Workflows** | `workflow_status` | Current state, requirements, versions and observed verification records. |
| | `workflow_run_checks` | Run owner-approved test commands and kernel-check approved Lean statements. |
| | `workflow_classify` | Request a versioned JEV Choice using approved synthetic context. |
| | `workflow_transition` | Evaluate the current classification through the compiled Lean transition function. |
| | `workflow_authorize` | Issue a single-use, action/version/state-bound execution grant. |
| | `workflow_execute` | Recheck and consume the grant, run the exact argv, then record success or failure. |

### Host-side GraphRAG extraction

`run_neograph_update` sends one batch per call to an OpenRouter model. Two tools let the calling agent (or a restricted subagent) do the extraction instead: they make no model call and no network call and do not use the OpenRouter key (a server that is not in local-only mode still needs its embedder configuration to start, as for every tool). They work with `SELF_DIRECT_LOCAL_ONLY=true` and with either `SELF_DIRECT_GRAPH_EXTRACTOR` value, like `propose_graph_update` and `commit_graph_update`, which make no model or network call either. `SELF_DIRECT_GRAPH_EXTRACTOR=host` also turns `run_neograph_update` off, so this server sends no session text to a model for graph extraction.

1. `graph_next_batch(session_id)` returns the next batch: complete events, masked exactly as `run_neograph_update` masks them, up to 100 complete events and 24,000 characters by default. `max_events` and `max_chars` ask for less, or for more up to 100 events and 95,000 characters, the limits `run_neograph_update` always uses. An event over the limit is `graph_event_too_large` (with the `needed_max_chars` to ask for, unless it exceeds 95,000 characters) and is never truncated. The result carries a `batch_id`, `remaining_events`, the graph cursor, and the extractor instructions, proposal schema, limits and allowed relations. `status: "up_to_date"` ends the loop; a session without indexed events is `no_events`.
2. The agent extracts nodes and edges whose `evidence_chunk_ids` are `chunk_id`s of that batch.
3. `graph_submit_batch(session_id, batch_id, proposal, extractor=...)` recomputes the batch from the index (`batch_changed` if its events are not what the id was made from; `invalid_batch_id` for an id `graph_next_batch` could not have written), requires it to be the next one (`out_of_order`, `stale_batch`), checks the proposal's shape and size caps (500 nodes, 1,000 edges, 100 evidence ids per edge, 500 characters per string, 200 per id, 2,000 for the summary), that ids, kinds, labels and edge ends are single-line text, that every edge cites only events of this batch, and the existing evidence rules (`rejected`), then commits with the same graph-version and evidence checks as every graph write. The job id is the batch id, so repeating a submit returns `already_applied`; if another session or provider has already recorded a job under that id the answer is `job_id_in_use` and nothing is written.

A batch is never skipped silently. A proposal without edges is rejected (`empty_graph_requires_review`, cursor retained): submit the relations the events state. When they really state none, call `graph_submit_batch` without a proposal and with `no_graph_facts_reason` (up to 500 characters); the cursor moves past the batch (`skipped_no_graph_facts`) and the reason and the `extractor` label are recorded in the job. A skip stores no nodes.

**Result size.** A full 95,000-character batch is about 100 KB of JSON (25,000 to 35,000 tokens), more than a host may let one tool result carry (Claude Code caps MCP tool output at about 25,000 tokens by default, Codex has a tool-output token limit). A host that shortens a result would show the agent part of a batch while a submit, a skip or a no-facts record covers all of it. So the default is 24,000 characters, and a complete result ends with `"end_of_batch": true` as its last key, with a `truncation_hint` near the start: if yours does not end with it, call again with a smaller `max_chars`, and never submit a batch you did not read completely. The server checks which events a batch names, not what the agent read; that rule is the agent's.

**Parallel extraction.** `graph_next_batch(session_id, after_order=k)` starts the batch at event `k + 1` and moves nothing, so several batches can be planned from one cursor and extracted at the same time (for example one subagent each). Commits stay in order: a batch submitted before its predecessors returns `out_of_order` with `expected_start_order`; submit the earlier ones, then this one again. A batch is the events it names, so events appended to a live session meanwhile do not invalidate it.

**Untrusted data.** Event text is copied from a transcript and can contain instructions aimed at whoever reads it. Run the extraction in an agent that has only these two tools (no shell, file or network access) and have it call nothing but `graph_submit_batch` with the result. Both tools accept any `session_id` and `provider`, so the server cannot restrict a subagent to one session: which sessions a subagent may extract is decided by how it is configured and instructed. What the server bounds is each call: a submit can add at most 500 nodes and 1,000 edges, every edge backed by events of its own batch, and move the cursor only over that batch. The `extractor` label is free text (control characters removed, 80 characters, default `host:unspecified`) kept for provenance; nothing trusts it.

**Single-line text.** Ids, kinds, labels and edge ends are stored as given and reach agent context line by line (the AGY hook prints each relation), so a proposal whose node ids, kinds, labels or edge ends contain an unsafe character is rejected, not edited (`rejected`; the problem names the code points, `U+2060` for one, and does not repeat the text). Unsafe, by Unicode category: control characters (Cc: newline, carriage return, tab, escape and the rest), the line and paragraph separators U+2028 and U+2029 (Zl, Zp), lone surrogates (Cs) and every format character (Cf) except three. That covers the characters that reorder or hide text: the bidirectional marks, embeddings and isolates U+061C, U+200E, U+200F, U+202A to U+202E and U+2066 to U+2069, the zero-width space U+200B, the word joiner and the invisible operators U+2060 to U+2064, the byte order mark U+FEFF, the interlinear annotation characters U+FFF9 to U+FFFB and the tag characters (U+E0001, U+E0020 to U+E007F). The other format characters, such as the Arabic number signs, are rejected with them, so the rule stays closed for characters added later. Three format characters are allowed because they only join or hyphenate and cannot hide or reorder anything: the zero-width non-joiner U+200C, the zero-width joiner U+200D and the soft hyphen U+00AD. Everything else is accepted, unassigned and private-use code points included: Hangul and the other scripts, Persian words with their zero-width non-joiner, emoji sequences joined by U+200D, Indic conjuncts. The extractor prompt and the `text_rules` of `graph_next_batch` tell the model this rule. The AGY context also prints every stored relation, contract description and checklist item on one line, showing such characters as escapes (rows stored before the check included).

**Cross-vendor.** Extracting a session with a different vendor's model discloses that session to that vendor. By default extract only the sessions of the host you are running in.

**Batch-range commits.** `commit_graph_update` without `through_order` moves the cursor to the end of the whole session, even when the proposal was built from some of the events; `propose_graph_update` and `commit_graph_update` report that position as `covers_through_order` (also when they answer with a conflict). Pass `through_order`, the order of the last event the proposal covers (from the first unprocessed event to the last event, otherwise `invalid_through_order` and nothing is stored), to advance only that far. It is also the explicit way past an event too large for a batch: read it with `get_chunk` and commit the relations you find with `through_order` set to its order. A commit without a `job_id` is named by a digest of provider, session, the event order the commit moves the cursor to (`through_order`, or the last event of the session as it stands when none is given) and the proposal: repeating the identical call at the same place is `already_applied`; any other commit applies, the same proposal after the session has grown included.

**Limits.** The cursor is an event order, not an event identity. A session rewritten shorter and then grown again can skip events: with 10 events and the cursor at 9, a rewrite to 5 events reads as `up_to_date`, and after 10 more events are appended the orders 5 to 9 are never extracted. `run_neograph_update` behaves the same way. Job ids are global, not per session: a caller who has seen a batch id can claim it first, through another session's `commit_graph_update` with that `job_id`, and so wedge that batch's host path (`graph_submit_batch` then answers `job_id_in_use`, explicit and never a false `already_applied`, and the other session's row is not touched) until the batch is committed with the documented `through_order` commit, which needs no job id.

---

## Quickstart

### Prerequisites
- Python 3.12+ and `uv` (recommended) or `venv`
- The official `sqlite-vector` 1.1.0 native library (the setup script below downloads and verifies it)
- An OpenRouter key for the default embedding and retrieval path; not needed in explicit local-only mode
- Optional: `pip install 'tsukkomi-mcp[pgvector]'` and a PostgreSQL server with the pgvector extension, only to keep the dense index there ([Dense backends](#dense-backends))

### 1. Clone and install

```bash
git clone https://github.com/fox1245/tsukkomi-mcp.git ~/.mcp-servers/tsukkomi-mcp
cd ~/.mcp-servers/tsukkomi-mcp

uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Provision sqlite-vector

```bash
python scripts/setup_sqlite_vector.py --output-dir /absolute/path/to/native
```

The script pins release 1.1.0, verifies the official archive's SHA-256 and checks native cosine search. Without `--output-dir` it writes to `.native/` in the current directory. Set `SELF_DIRECT_SQLITE_VECTOR_PATH` (or `sqlite_vector_path` in TOML) to the resulting `vector.so`, `vector.dll` or `vector.dylib`. With no setting, the runtime only looks for `native/vector.*` inside the source checkout. If the library cannot be loaded, engine initialization fails with an error; it does not fall back to NumPy. NumPy (`SELF_DIRECT_DENSE_BACKEND=numpy`) and `SELF_DIRECT_USE_FAKE_EMBEDDER=true` are explicit offline and test options.

### 3. Configure

Settings come from, in order of precedence: explicit `Settings` arguments, environment variables, the TOML path file, a `.env` in the **current working directory**, then defaults. A server started by an MCP host usually has an unrelated working directory, so do not depend on a `.env` there; use explicit `SELF_DIRECT_*` variables in the host's server entry or the TOML file below. Lifecycle hooks (AGY and Claude Code) never read a working-directory `.env`: they use the environment and the TOML only, because that file is content the audited agent can write.

#### User path configuration (TOML)

Use `~/.config/tsukkomi-mcp/config.toml` on any supported platform, or set `SELF_DIRECT_CONFIG_FILE` in the process environment to select another file. Copy and edit [config.example.toml](config.example.toml); the edited file belongs outside the repository and agent-writable workspaces.

```toml
[paths]
index_dir = "~/private-tsukkomi/index"
omp_sessions_dir = "~/.omp/agent/sessions"
openrouter_api_key_file = "../credentials/openrouter.env"
sqlite_vector_path = "native/vector.so"
```

The nine supported keys are `index_dir`, `codex_sessions_dir`, `omp_sessions_dir`, `claude_projects_dir`, `grokbot_transcripts_dir`, `agy_app_data_dirs`, `contracts_path`, `sqlite_vector_path` and `openrouter_api_key_file`. AGY roots use a string array; Grokbot accepts its existing delimiter-separated string or an array of individual paths. Empty arrays disable those optional roots.

TOML paths expand `~`; relative paths resolve against the canonical TOML file's directory, not the working directory or a symlink's directory. `${ENV_VAR}` interpolation is not performed. The selector `SELF_DIRECT_CONFIG_FILE` is environment-only, so a `.env` cannot silently select another TOML file. An absent default file keeps the defaults. An explicitly selected missing file, malformed TOML, an unknown section or key, or an invalid path value fails instead of silently selecting another index. Settings capture the selected sources, so restart affected servers and extensions after changing them.

#### Environment settings

All settings use the `SELF_DIRECT_` prefix except the key itself. The names below are the runtime fields in `src/self_directing_mcp/config.py`; [.env.example](.env.example) lists the commonly used ones.

| Setting | Default | Meaning |
|---|---|---|
| `SELF_DIRECT_INDEX_DIR` | `~/.self_direct_index` | Private index directory (events, FTS, vectors, contracts, checklists). |
| `SELF_DIRECT_CONTRACTS_PATH` | `<index_dir>/contracts.json` | Contracts file. |
| `SELF_DIRECT_SESSION_PROVIDER` | `codex` | Provider used when a tool call omits `provider`: `codex`, `grokbot`, `omp`, `agy` or `claude`. |
| `SELF_DIRECT_CODEX_SESSIONS_DIR`, `CODEX_SESSIONS_DIR` | `~/.codex/sessions` | Codex session root. |
| `SELF_DIRECT_OMP_SESSIONS_DIR` | `~/.omp/sessions` | OMP session root. |
| `SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR` | empty (disabled) | Cursor / Grok Bot transcript roots. |
| `SELF_DIRECT_AGY_APP_DATA_DIRS` | `~/.gemini/antigravity`, `-cli`, `-ide` | JSON array of AGY app-data directories. |
| `SELF_DIRECT_CLAUDE_PROJECTS_DIR` | `$CLAUDE_CONFIG_DIR/projects` if `CLAUDE_CONFIG_DIR` is set, else `~/.claude/projects` | Claude Code projects root (TOML `claude_projects_dir`). The environment variable wins over TOML, and TOML over the default. |
| `SELF_DIRECT_AGY_ENFORCEMENT` | `confirmed_only` | AGY hook policy: `confirmed_only`, `advisory` (never blocks) or `enforced` (also denies an unverifiable `PreToolUse`). Hook errors fail open in every mode. |
| `SELF_DIRECT_CLAUDE_ENFORCEMENT` | `confirmed_only` | Claude Code hook policy, same values, independent of the AGY setting. `scripts/install_claude_hooks.py` writes it into the Claude Code settings `env`. |
| `SELF_DIRECT_USE_FAKE_EMBEDDER` | `false` | Offline/test embedder; never a silent fallback. |
| `SELF_DIRECT_LOCAL_ONLY` | `false` | Explicit offline mode: no embedder or dense index; embedding, dense/hybrid search and LLM graph updates are rejected. |
| `SELF_DIRECT_DENSE_BACKEND` | `sqlite-vector` | `sqlite-vector`, `numpy` (explicit offline option) or `pgvector` (opt-in extra; the vectors go to a PostgreSQL server, see [Dense backends](#dense-backends)). Never substituted for one another. |
| `SELF_DIRECT_PGVECTOR_DSN_FILE` | unset | `pgvector` only: a file (TOML `pgvector_dsn_file`) holding one libpq connection string, `host=... dbname=...` or a `postgresql://` URI. It is a secret: the file must not be readable by group or others (`chmod 600`), its content is never logged or echoed, and a connection string in an environment variable (`SELF_DIRECT_PGVECTOR_DSN`) is refused. Unset: only the standard libpq environment (`PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`, `PGPASSFILE`, `PGSERVICE`, `PGSSLMODE`) selects the server. |
| `SELF_DIRECT_PGVECTOR_SCHEMA` | `tsukkomi` | `pgvector` only: the PostgreSQL schema that holds `chunk_vectors` and `tsukkomi_meta`; ASCII letters, digits and underscore, at most 63, not `pg_*`. One schema per index directory. |
| `SELF_DIRECT_PGVECTOR_HNSW` | `false` | `pgvector` only: also keep an approximate (HNSW) index. Off by default: the exact scan is the reference. See [HNSW](#hnsw-optional-approximate-search). |
| `SELF_DIRECT_SQLITE_VECTOR_PATH` | `native/vector.*` in the checkout, if present | Native library path. |
| `SELF_DIRECT_OPENROUTER_API_KEY_FILE` | unset | Authoritative dotenv file with `OPENROUTER_API_KEY`. |
| `SELF_DIRECT_EMBEDDING_MODEL`, `SELF_DIRECT_EMBEDDING_DIM`, `SELF_DIRECT_EMBEDDING_BASE_URL` | `qwen/qwen3-embedding-8b`, `1024`, `https://openrouter.ai/api/v1` | Embedding model, dimension and endpoint. |
| `SELF_DIRECT_LOCK_WAIT_TIMEOUT_SEC` | `1` | Combined process/file writer-lease wait. |
| `SELF_DIRECT_AUDIT_TIMEOUT_SEC` | `8` | Response budget for audits and hooks. |
| `SELF_DIRECT_REQUEST_TIMEOUT_SEC` | `45` | Response budget for ordinary calls. |
| `SELF_DIRECT_GRAPH_UPDATE_TIMEOUT_SEC` | `300` | Budget for `run_neograph_update`. |
| `SELF_DIRECT_GRAPH_EXTRACTOR` | `openrouter` | Who extracts for `run_neograph_update`: `openrouter` (an OpenRouter model) or `host`, which disables that tool (`extractor_disabled`; the key and the network are not touched). `graph_next_batch` and `graph_submit_batch` work with either value, also with `SELF_DIRECT_LOCAL_ONLY=true`. An unknown value is an error. See [Host-side GraphRAG extraction](#host-side-graphrag-extraction). |
| `SELF_DIRECT_MAX_TOOL_TIMEOUT_SEC` | `600` | Upper bound for a client-supplied `timeout_ms`. |
| `SELF_DIRECT_MAX_CONCURRENT_CALLS` | `4` | **New in 0.6.0.** Heavy tool calls that may run at once (minimum 1); excess calls wait within their own deadline. |
| `SELF_DIRECT_INTEGRITY_FAST_RECENT_BLOCKS`, `SELF_DIRECT_INTEGRITY_FAST_SAMPLE_BLOCKS` | `4`, `8` | Fast (hook-time) source verification: the latest complete 1 MiB blocks and the random sample compared with their stored digests. `0` for either disables that part (the tail block is always compared). See [Source integrity](#source-integrity-full-and-fast-verification). |
| `SELF_DIRECT_INTEGRITY_FULL_VERIFY_INTERVAL_SEC` | `300` | A fast check is allowed only this long after the last verification of every block; then the next check is full. `0`: always full. |
| `SELF_DIRECT_INTEGRITY_BACKGROUND_VERIFY`, `SELF_DIRECT_INTEGRITY_BACKGROUND_IDLE_SEC`, `SELF_DIRECT_INTEGRITY_BACKGROUND_MAX_SCANS` | `true`, `2`, `1` | Resident servers verify watched sources in full on one background thread after this much idleness; at most this many scans per wake-up. Hook processes never do background work. |
| `SELF_DIRECT_AUDIT_SHADOW_EVERY`, `SELF_DIRECT_AUDIT_SHADOW_IDLE_SEC` | `adaptive`, `2` | How incremental audits are compared with the full evaluator: `adaptive` (rate 1/4 down to 1/32 plus forced comparisons) or an integer that pins the fixed rate (`1` every audit, `0` only the forced comparisons, `N` about one in `N`); resident servers compare the latest state of a session after this many idle seconds (`0` off). See [Incremental audits](#incremental-audits). |
| `SELF_DIRECT_MAX_DECOMPRESSED_SESSION_BYTES` | `4294967296` | **New in 0.6.0.** Largest decompressed size a `.jsonl.zst` Codex session may declare; larger files are refused as `zstd_too_large`. |
| `SELF_DIRECT_RETRIEVE_TOP_K`, `SELF_DIRECT_SEARCH_TOP_K`, `SELF_DIRECT_RRF_K` | `20`, `10`, `60` | Retrieval tuning. |
| `SELF_DIRECT_SEED_EXAMPLE_CONTRACTS` | `false` | Seed disabled demonstration contracts. |
| `SELF_DIRECT_DAEMON` | `off` | **Experimental.** `on` makes hooks and the MCP entry point clients of the resident daemon (Linux only; read from the process environment by the entry points). See [Resident daemon](#resident-daemon-experimental-off-by-default), which also lists the daemon's own `SELF_DIRECT_DAEMON_*` limits. |
| `SELF_DIRECT_SQLITE_JOURNAL` | `delete` | Journal mode of the authority databases `meta.sqlite` and `agy-receipts.sqlite`: `delete` (the rollback journal) or `auto` (write-ahead logging on a verified local file system, the rollback journal elsewhere). Always `synchronous=FULL`. Faster for resident processes only; see [SQLite journal mode](#sqlite-journal-mode-index-files-and-backups). |
| `SELF_DIRECT_SYNC_SLICING` | `auto` | A sync that cannot finish inside its request budget publishes the records it parsed (a slice) and the next call continues from there. `auto`: requests with a finite deadline, plain session files; appends are published in slices, and a re-index is staged in a spool and published by one commit, never in pieces over an existing index; `off`: the all-or-nothing sync of before. See [Large sessions and deadlines](#large-sessions-and-deadlines). |
| `SELF_DIRECT_SYNC_SLICE_BYTES` | `67108864` | The most a slice observes (hashes and parses) past the indexed offset, to the end of a record (a slice always holds at least one whole record). Plain session files only. |

#### Credentials

Credential values are not TOML options. A key file selected in TOML or by `SELF_DIRECT_OPENROUTER_API_KEY_FILE` is authoritative: a missing, unreadable or keyless file is an error, never a fallback to another key. Without a selected key file, `OPENROUTER_API_KEY` follows the precedence above (explicit argument, environment, working-directory `.env`). Keep the key file outside the repository and agent-writable workspaces. Workflow sandboxes reject mounts that expose the selected TOML, a host dotenv or the selected credential file.

---

## Platform configuration

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

To enable lifecycle interception in AGY, register `scripts/agy_hook.py` as your workspace hook command. By default only confirmed findings gate a tool call. See [docs/agy-installation.md](docs/agy-installation.md) for the hook file, enforcement modes and host boundaries.

### OpenAI Codex CLI

Install advisory lifecycle hooks and register the MCP server in one command:

```bash
python scripts/install_codex.py --trust-hooks
python scripts/install_codex.py --verify
```

The installer registers explicit `SELF_DIRECT_INDEX_DIR`, session-root, backend and key-file settings in the Codex MCP entry, so it is not affected by the default index location. See [docs/codex-installation.md](docs/codex-installation.md).

### Claude Code

Register the MCP server and, if the user wants lifecycle controls, review and then install the opt-in hooks:

```bash
claude mcp add --scope user tsukkomi-mcp -- /absolute/path/to/venv/bin/python -m self_directing_mcp
python scripts/install_claude_hooks.py --print --python /absolute/path/to/venv/bin/python
python scripts/install_claude_hooks.py --apply --settings /absolute/path/to/settings.json --python /absolute/path/to/venv/bin/python
```

`--print` is the default and changes nothing; `--apply` merges into the settings file you name, keeps unrelated hooks, backs the file up when it changes and replaces it atomically. Add `--index-dir /absolute/path/to/index` to point the hooks at the index whose contracts should apply (the one the MCP server writes to); hooks never read a project `.env`. See [docs/claude-installation.md](docs/claude-installation.md) for the policy table, the Stop budget, subagents, fail-open behavior and verification steps, and [docs/claude-hooks.example.json](docs/claude-hooks.example.json) for the complete hooks block.

### OMP (OpenRouter-backed retrieval, deterministic contract hooks)

Register this server in OMP's **user** `mcp.json` and opt in to `extensions/tsukkomi-omp.ts` as an OMP user extension. Configure both processes with the same private TOML file (or environment overrides) for the index, OMP session root and authorized credential/vector references. The extension starts Python in `--omp-bridge` mode, so the shared Python settings source owns validation; TOML-only paths need not be duplicated in the extension environment.

```text
SELF_DIRECT_LOCAL_ONLY=false
SELF_DIRECT_CONFIG_FILE=<private TOML path>  # omit to use the default user file
TSUKKOMI_OMP_MONITOR=1
TSUKKOMI_OMP_PYTHON=<Python with tsukkomi-mcp dependencies>
TSUKKOMI_OMP_SOURCE=<source checkout>/src  # omit when installed in that Python
```

Interpreter/source bootstrap and `SELF_DIRECT_LOCAL_ONLY` remain environment options, not `[paths]` keys. Remote bridge startup requires a usable explicitly selected key file and an existing native-library path; it does not make an API call or eagerly initialize the index. Environment paths already present in managed registrations override TOML until the owner removes or changes those entries.

OpenRouter-backed hybrid/dense history search, `sync_session(embed=True)` and NeoGraph updates are available by default; embedding uploads sanitized session chunks, so use an authorized key and session root. The hook's `check_action` and `audit_session` stay deterministic and refresh JSONL without embedding; they do **not** call OpenRouter on every tool or turn. Explicit offline-only mode is `SELF_DIRECT_LOCAL_ONLY=true`, which disables those remote functions. NeoGraph extraction also masks known secret patterns in session events before sending a prompt to OpenRouter; keep the original session files and local index private.

Keep machine-specific values in local OMP configuration or a private extension copy, not this repository. Start a **new persisted OMP session**; `--no-session` cannot provide history. Register only user-stated, machine-checkable contracts with `upsert_contracts` (`provider: "omp"` and the actual OMP session ID for session scope). The extension checks pending model-issued tools through local MCP before execution, blocks confirmed `violation`/`suspicious`, and re-audits after results and turns. `unknown` and transport failure are reported but do **not** block; zero applicable contracts is not `clean`. Direct shell or host actions outside OMP's tool pipeline, skipped hooks, in-memory sessions and a result not yet flushed to disk are not covered.

The extension also registers the read-only `tsukkomi_session_context` tool. Call it before creating a session-scoped contract or checklist item: its `session_id` comes from the active OMP session manager. Never infer a session ID from `PI_SESSION_FILE` or a shell variable, which can belong to a parent OMP process. If no persisted session is available, do not silently create a global contract. `AGENTS.md` instructions alone are not audit contracts: register the user's explicit, machine-checkable prohibitions; broader subjective mistakes are not automatically judged by an LLM.

### Cursor / Grok Bot

Set `SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR` (or `grokbot_transcripts_dir` in TOML) and pass `provider="grokbot"` to the MCP tools. There is no lifecycle hook for this host in this repository.

## Dense backends

The dense half of `search_history` sits behind one interface (`index/vector.py::VectorIndex`). The canonical event store, the sparse (FTS5) index, the contracts and the embedding cache are always local SQLite files; only the vectors can live somewhere else. A backend is chosen with `SELF_DIRECT_DENSE_BACKEND` and is **never replaced by another one**: when the selected backend cannot work, the call fails with an error that names the repair, and `audit_status` says so.

| Backend | Select | What it is | Use it when |
|---|---|---|---|
| `sqlite-vector` (default) | `sqlite-vector` | The official sqlite-vector 1.1.0 native library: an exact cosine scan over a local file (`vectors-v2-*/dense.sqlite`). | Every normal install. Nothing to operate, nothing leaves the machine. |
| `numpy` | `numpy` | An explicit offline and test option: a float64 matrix in memory, persisted as `dense_numpy.npz`. It scores a dot product, which is a cosine only for unit vectors (every embedder here returns unit vectors). | Tests, and machines that cannot load the native library. Not for large indexes: each write rewrites the file and each search copies the matrix. |
| `pgvector` | `pgvector` | PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension: an exact cosine scan with the id filter pushed into SQL, and an optional HNSW index. An opt-in extra. | The vectors should live on a database server you operate: more vectors than you want in a laptop file, one place for them, session-scoped searches that must stay fast as the index grows. Not needed otherwise. |

All three score `1 - cosine distance` (1 is identical, a zero vector scores 0), apply `allowed_ids` before the top-k, and break exact ties by id in byte order (NumPy: arbitrary inside a tie). `tests/test_dense_backend_parity.py` holds pgvector and sqlite-vector to NumPy's exact scan on the same seeded vectors, filters and write sequences: identical top-k except inside groups of exactly equal scores, scores within 1e-6 (float32 arithmetic; the largest difference seen was 2e-7).

### pgvector

```bash
pip install 'tsukkomi-mcp[pgvector]'          # psycopg 3; imported at the first query, never by hooks or the default install
# once, by a database administrator, in the target database (this process never installs it):
#   CREATE EXTENSION vector;
(umask 077; printf '%s\n' 'host=db.example.org port=5432 dbname=tsukkomi user=tsukkomi password=<password> sslmode=verify-full' > /absolute/path/pgvector.dsn)
export SELF_DIRECT_DENSE_BACKEND=pgvector
export SELF_DIRECT_PGVECTOR_DSN_FILE=/absolute/path/pgvector.dsn
```

- **The connection string** is read only from the file named by `SELF_DIRECT_PGVECTOR_DSN_FILE` (one line, `host=... dbname=...` or a `postgresql://` URI; on POSIX not readable or writable by group or others) and from the standard libpq environment (`PGHOST`, `PGPASSFILE`, `PGSERVICE`...). It is never accepted as a setting value, never logged, never echoed: errors show `host:port/database`, the server's primary message with any password removed (also URI-encoded), and nothing else. `audit_status` shows `host`, `port`, `database` and `schema`, never the user or any option.
- **TLS.** A server that is not this machine (anything but `localhost`, a loopback address or a socket directory) is reached only with `sslmode=verify-full` and a root certificate that verifies its name; `disable`, `allow`, `prefer` (libpq's default), `require` and `verify-ca` are refused before a connection is opened, so no credential is sent in the clear. A connection that only names a `service` hides its host, so it is forced to `verify-full`. A loopback server (a local container, a socket) needs no TLS.
- **What leaves the machine.** Chunk ids and embedding vectors (float32, `SELF_DIRECT_EMBEDDING_DIM` numbers per chunk). Never session text, contracts or the canonical store. Vectors are **not anonymous**: an embedding can be partly inverted to recover the text it stands for, so treat them like the text. Selecting `pgvector` is therefore an explicit opt-in to that data flow, and a real remote server needs the owner's approval before it is first used; try it with synthetic data and a local container (`tests/pgvector_support.py` starts one). `SELF_DIRECT_LOCAL_ONLY=true` still forbids any dense index.
- **Never on the hook path.** Hooks and audits do not use the dense index at all; the command hooks publish the canonical events and leave the vectors to the next `sync_session` of the server, so a database outage cannot delay or hide an audit. Only `sync_session` (vector upkeep) and `search_history` (`dense` and `hybrid` modes) talk to the server.
- **Failure is an error, never a fallback.** An unreachable server, bad credentials, a missing `vector` extension, a dimension or embedding-policy mismatch each raise an error that names the repair (the DSN file, reachability, `CREATE EXTENSION vector;`, another `SELF_DIRECT_PGVECTOR_SCHEMA`). `sync_session` and dense or hybrid `search_history` return `error: dense_backend_unavailable` (the canonical events of a sync were already published and the journal keeps the vector work it owes; the next sync resumes). `sparse` and `regex` searches and all audits keep working. `audit_status` reports `dense_backend: "pgvector"`, `dense_count` (`null` when the server cannot answer), `dense_error`, the degraded flag `dense_backend_unavailable`, and `dense_info`: `host`, `port`, `database`, `schema`, `dimension`, `extension_version`, `server_version`, `tls`, `loopback`.
- **Time budget.** Every statement runs with `statement_timeout` and `lock_timeout` equal to the time the request has left, so a slow or blocked statement ends with the request's own `request_deadline`; a connection attempt takes at most 10 seconds or the time left. A peer that vanishes is noticed through TCP keepalives (idle, about a minute) and `tcp_user_timeout` (30 s with data in flight). One connection per index object, opened lazily, validated on connect, replaced once if it drops (every operation is idempotent), closed with the engine. A write call is one transaction that commits only if the request is still live, so a stopped request leaves nothing half-applied.
- **The schema** (`SELF_DIRECT_PGVECTOR_SCHEMA`, default `tsukkomi`), created under an advisory lock on first use by a role that may `CREATE` in the database, or by an administrator beforehand: `chunk_vectors(chunk_id text COLLATE "C" PRIMARY KEY, embedding vector(N) NOT NULL)` and `tsukkomi_meta(key, value)`, which records `schema_version`, `dim` (the vector dimension) and `policy` (the prefix of the embedder's cache key, which names the model, dimension and encoding). A process whose dimension or policy differs gets an explicit error instead of mixed vectors. Use **one schema per index directory**: the table is keyed by chunk id only. Up to 1700 dimensions the vector column is stored inline (`SET STORAGE PLAIN`): about 46 % more disk than pgvector's default out-of-line storage for 2.4 times faster scans (50,000 x 1024: 392 MB and 58 ms against 268 MB and 141 ms); above that the default is kept.
- **Migrating** is not an in-place conversion: change the setting, and the next `sync_session` refills the new index from the local embedding cache (`dense_restored`; no embedding call for text that is already cached). The old `vectors-v2-*` files stay untouched, and so does the PostgreSQL schema when you go back (reverting the setting is the whole downgrade).
- **Versions.** Tested with PostgreSQL 17.11, pgvector 0.8.7 and psycopg 3.3 on Linux. Exact search needs only the `vector` type and `<=>`; HNSW needs pgvector 0.5.0 or newer. Other PostgreSQL and pgvector versions, a remote server, TLS to a real server and Windows are not verified (Windows needs only the optional extra; the default install does not).

#### HNSW (optional approximate search)

`SELF_DIRECT_PGVECTOR_HNSW=true` creates `chunk_vectors_embedding_hnsw` (cosine, pgvector's defaults `m=16`, `ef_construction=64`; at most 2000 dimensions) and serves searches **without** an id filter with `hnsw.ef_search = max(100, 4 x top_k)`. It is off by default because the exact scan is the reference and because of what it costs: the index is as large as the table, every write maintains it, and it is approximate. Searches with an id filter are always exact, and **`search_history` always passes one** (the session's ids, or every id of the provider), so with the current engine the index is built and kept but not used: turn it on only when calling the backend directly. A large table's index takes minutes to build; build it outside a request with `PgVectorIndex(...).build_hnsw_index(maintenance_work_mem='2GB', timeout_ms=0)` (a build inside a request that runs out of time says so and names this command). The recall trade-off below is measured on clustered synthetic vectors; vectors with no neighbourhood structure (uniform random in 1024 dimensions) are the worst case, with recall@20 of 0.03 at 100,000 vectors and `ef_search=40`.

### pgvector numbers

Synthetic unit float32 vectors of dimension 1024 (uniform random for the comparison, 500 Gaussian clusters for HNSW recall), top_k = 20, 30 searches per cell after 2 warm-ups, p50 / p95 in milliseconds, one client on one machine (16 logical CPUs, WSL2, shared with other work: repeated runs differed by up to 35 %, so read the ratios, not the digits). pgvector: PostgreSQL 17.11 and pgvector 0.8.7 in a loopback container with a tmpfs data directory, `shared_buffers=512MB`, PostgreSQL's default of 2 parallel workers per scan, inline vector storage, batches of 2000 rows per call. sqlite-vector 1.1.0 on a local file; NumPy in memory (loaded by one call, because every call re-stacks the whole matrix). Ids are 65 characters long, like the engine's chunk ids. The filters are 500 stored ids and 50,000 ids (ids that are not stored pad the set when N is below 50,000). `scripts/bench_pgvector.py` reproduces everything; the 300,000-vector runs needed a larger tmpfs (8 GB) and, for HNSW, `--shm 6g --maintenance-work-mem 4GB`.

| N | backend | upsert rows/s | `present` of 500 | search, no filter | 500 ids | 50,000 ids | size |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1,000 | pgvector | 6,570 | 2.5 | 1.3 / 1.6 | 1.8 / 2.6 | 1.5 / 1.8 | 8 MB |
| | sqlite-vector | 47,943 | 0.45 | 1.6 / 1.8 | 2.1 / 2.2 | 43 / 51 | 4 MB |
| | NumPy | 311,380 | 0.02 | 4.0 / 8.1 | 4.0 / 4.1 | 4.4 / 8.1 | 8 MB |
| 10,000 | pgvector | 9,325 | 2.5 | 8.6 / 10.0 | 2.2 / 3.1 | 9.1 / 12.3 | 81 MB |
| | sqlite-vector | 64,158 | 0.47 | 18.2 / 25.7 | 20.2 / 22.7 | 59 / 76 | 45 MB |
| | NumPy | 106,060 | 0.15 | 16.5 / 27.3 | 4.0 / 13.0 | 16.9 / 22.6 | 78 MB |
| 100,000 | pgvector | 9,733 | 2.4 | 78 / 85 | 2.5 / 4.6 | 80 / 93 | 810 MB |
| | sqlite-vector | 59,424 | 1.2 | 195 / 212 | 214 / 244 | 266 / 294 | 448 MB |
| | NumPy | 96,826 | 1.2 | 142 / 166 | 17.2 / 34.8 | 88 / 117 | 781 MB |
| 300,000 | pgvector | 9,739 | 4.4 | 248 / 277 | 4.0 / 5.2 | 229 / 265 | 2,429 MB |
| | sqlite-vector | 51,361 | 1.7 | 646 / 711 | 682 / 787 | 727 / 773 | 1,343 MB |
| | NumPy | 70,276 | 4.0 | 417 / 438 | 43.7 / 64.9 | 142 / 168 | 2,344 MB |

Where pgvector is **slower**: writes (about 9,000 rows/s, a sixth of sqlite-vector's: one transaction per batch, and a 4 kB vector fills a whole 8 kB page), `present` (a round trip: 2.4 to 4.4 ms against 0.5 to 1.7), disk (810 MB against 448 MB at 100,000, because inline storage wastes half of each page; pgvector's default storage is 32 % smaller but scans 2.4 times slower), the first query of a process (a validation round trip), a 50,000-id filter on a large table when the filter names less than a quarter of it (the ids are sent and looked up: 229 ms at 300,000 against NumPy's 142), and everything on a small index: below about 10,000 vectors the local backends answer first. Where it **wins**: the exact scan once the index is large (2.5 times faster than sqlite-vector at 100,000 vectors and at 300,000, 1.7 to 1.8 times faster than NumPy: pgvector's SIMD distance and parallel scan against a single-threaded one; with `max_parallel_workers_per_gather=0` expect less), a session-scoped search (500 ids: 2 to 4 ms at every size, an index lookup per id, where sqlite-vector scans everything and filters afterwards, 20 to 682 ms, and NumPy copies the matrix, 4 to 44 ms), a search over most of the index (a filter naming a quarter of the table or more is applied after one unfiltered scan, so no ids are sent), and concurrent readers and writers.

How a filter is sent matters more than it looks. A 50,000-id filter of 65-character ids is 3.4 MB (split by the server in 44 ms as one joined text parameter, which is what is sent; with 8-character ids a `text[]` parameter took 42 ms against 12 ms joined, because psycopg dumps an array element by element). A hashed `= ANY` scan with such ids was 4 times slower than the scan alone (290 against 72 ms at 100,000 rows) and slower than looking the ids up by primary key (146 ms for 50,000 ids), so the planner's choice is left alone. A filter naming a quarter of the table or more therefore sends nothing: the top rows of the whole table are fetched, the ids that are not allowed are dropped here, and the fetch is widened (4 times as many rows, at most three scans) until enough allowed rows are in hand; the answer is exact, and after three scans that were not enough the ids are sent after all.

HNSW on clustered vectors (build with 2 parallel workers; `ef_search` is the search width; p50 latency at `ef_search=100`, the default here, against the exact scan):

| N | build | index size | recall@20 at `ef_search` 40 / 100 / 200 / 400 | HNSW p50 | exact p50 |
|---:|---:|---:|---|---:|---:|
| 10,000 | 5.0 s | 78 MB | 0.989 / 0.993 / 0.998 / 1.000 | 3.2 ms | 11.1 ms |
| 100,000 | 30 s | 781 MB | 0.918 / 0.998 / 0.998 / 0.999 | 1.7 ms | 92.9 ms |
| 300,000 | 89 s | 2,344 MB | 0.797 / 0.894 / 0.956 / 0.980 | 1.8 ms | 227.9 ms |

**A LAN adds round trips, not computation.** A search or `present` costs about one round trip; a write batch pipelines its rows. Measured through a loopback proxy that delays each direction (10,000 vectors, p50): 0.25 ms each way (a 0.5 ms round trip) adds about 0.5 ms to `present` and nothing measurable to a search; 1 ms each way (2 ms round trip) adds 2.3 ms to `present` and about 1.5 ms to a search, and lowers bulk upserts from 6,700 to 6,000 rows/s (bandwidth was not limited: 4 kB per vector is 30,000 rows/s per gigabit). A filter sent as ids is 3.4 MB per 50,000 ids, 27 ms on a gigabit link.

These are one machine's numbers on synthetic data. They do not include concurrent load, a cold cache or a remote server, and loopback understates every network cost.

---

## Coverage and verdicts

`sync_session`, `audit_session` and `check_action` report a `coverage` object (`audit_status` reports the same data under `session` when given a session id). It holds the indexed source `path`, `byte_offset` (how far the index has parsed), `file_size`, `chunk_count`, `parser_version`, an `issues` list and `complete`, which is true only when `issues` is empty. While a sync is still in progress it adds `backlog_bytes` and the issue `sync_in_progress` ([Large sessions and deadlines](#large-sessions-and-deadlines)). The same responses carry an `integrity` object that says how the source was verified for that request and when it was last verified in full ([Source integrity](#source-integrity-full-and-fast-verification)).

| Verdict | Meaning |
|---|---|
| `violation` | A contract is broken by evidence in the judged history or by the proposed action. |
| `suspicious` | An exception that needs confirmation. |
| `unknown` | The evidence needed to decide is missing, incomplete or changed. Resolve the gap; it is not permission. |
| `clean` | No problem was found **in the returned evidence and contract snapshot**. It is not a new authorization. |

Audit and `check_action` results also carry `requires_attention`, which is `true` whenever the verdict is not `clean`.

### Format drift: unsupported records

A record the parser does not know (an unsupported record type or step, a parse error) is kept visible and makes `coverage.complete` false (`unsupported_events`, `parse_errors`). It may hold the forbidden call or the commit an order rule waits for, so a history audit does not vouch for the records it could not read:

- A `must_not` with no match and a `before_regex` rule with no triggering action are `unknown` (reasons `incomplete evidence cannot prove the absence of a forbidden match` / `... that no triggering action occurred`), and so is the overall verdict.
- A `must` obligation that rests on missing evidence is `unknown` (as before), and a proposed action's `before_regex` / `requires_success` prerequisite cannot be established by incomplete history.
- A violation confirmed from the parsed evidence stays a violation, in the findings and overall; a proposed action that matches a prohibition is a violation whatever the history coverage; a `clean` that rests on positive evidence (a satisfied prerequisite) is not touched.
- Appended, unread bytes are not drift: they keep the `judged_prefix` behaviour below.

### Growing sessions

Agents append to their own session file while they work, so an audit usually sees a file that is still growing. The audit judges the bytes captured at the start and says so.

| Field (in `coverage`) | Meaning |
|---|---|
| `judged_through_byte` | Offset up to which the verdict was computed. |
| `unread_bytes` | Bytes after that offset (appended records, or an unfinished final line) that were not reviewed. Also listed in `issues` as `unread_bytes`. |
| `judged_prefix` | `true` when appended, unread bytes are the only coverage gap. |
| `complete` | `false` while any bytes are unread. |

The result's `notes` then contain `Judged as of byte N; M appended byte(s) were not reviewed.` Within that prefix:

- A `must_not` violation seen in the captured range is reported as `violation`.
- Otherwise the verdict is `clean` **as of that offset**. Appended records are judged by the next audit.
- A `must` obligation that would be a violation only because evidence is absent, or failed, becomes `unknown` while bytes are unread. An obligation already evidenced inside the prefix stays `clean` as of the offset.
- Any other gap (parse errors, unsupported events, an outdated parser, a missing source) is not a pure append. The verdict follows the normal incomplete-coverage rules and `judged_prefix` is absent.

A rewrite of earlier bytes, truncation, file replacement or deletion is never judged as growth. It stays `unknown` (`source_changed`, `source_unavailable_or_truncated`, `source_changed_during_evaluation`). If metadata, contracts or AGY receipts change while an audit is evaluating, the result is `unknown` with the earlier result kept as `snapshot_verdict`.

### `check_action` and current prerequisites

A proposed action runs now, so unread records come before it, not after it. When a contract with a gated prerequisite (`must` with `before_regex`) applies to the proposed action and the history is incomplete:

- With `requires_success: true`, a `clean` finding becomes `unknown` and the overall verdict is raised to at least `unknown`, because a newer failed or pending run may be in the unread part ([#8](https://github.com/fox1245/tsukkomi-mcp/issues/8)).
- A presence-only prerequisite (`requires_success: false`) stays `clean` while the only gap is appended bytes, because an observed earlier run is not undone by later records. For any other coverage gap it becomes `unknown`.
- A `must_not` violation already found is not hidden by this rule.

### Duplicate session sources

Several files can carry the same session id, for example when sub-agent session files are copied under another parent folder. Codex, OMP and Grok Bot compare the stored prefix hash with the requested file using streaming hashes, without loading either file. AGY is excluded because its hook receipts bind to an exact path, so a changed AGY path is still reindexed.

| Relation of the requested file to the indexed one | Result |
|---|---|
| Identical copy | Not reparsed. The response has `alias_of` (indexed path) and `coverage.indexed_path` / `coverage.requested_path`; the cursor stays on the indexed file. |
| Continuation (starts with the indexed prefix and is longer) | Only the part after the cursor is parsed and the cursor path moves to the longer file. |
| Strict prefix of the indexed file | The cursor is kept. Coverage reports the indexed source, adds the issue `requested_source_is_prefix_of_indexed` and sets `complete=false`, because the index holds events the requested file lacks. |
| Diverged (neither file is a prefix of the other) | The index is preserved. The call fails with `error: "ambiguous_session_source"` (with `indexed_path` and `requested_path`) and an audit is `unknown`, with `ambiguous_session_source` in `coverage.issues`. |
| Indexed file gone, requested file continues it | The new path is adopted without reparsing. If it does not continue the indexed history, the session is reindexed (`source_changed`). |

A rewrite or truncation of the same path is still reparsed or rejected as before. A plain rollout and its zstd-compressed twin (`.jsonl` and `.jsonl.zst`) are compared by their decompressed bytes, so every row above applies to that pair too; see [Compressed Codex sessions](#compressed-codex-sessions).

### Incremental audits

An audit used to re-read and re-evaluate every stored event of the session, so its cost grew with the session even when
nothing had been appended. Audits now keep a small **monitor state** per contract and fold in only the events that
arrived after it. The answer is the same: verdicts, findings, evidence, notes, `coverage` (including
`judged_through_byte`, `unread_bytes` and every unknown reason) and the NeoGraph stage trace are identical to the full
evaluation, which remains the reference. The audit still runs through the native `contract_audit` graph; only what its
`load_evidence` and `evaluate_contracts` stages do changes.

- **What is stored.** `meta.sqlite` gains derived tables (`audit_monitor_state`, `session_commit_log`,
  `session_generations`, `audit_incremental_health`, `audit_shadow_state`, `audit_review_ranges`) and, once a `requires_success` contract or a Claude Code result
  contract is audited, an index on tool-call ids. Nothing in them is authority: the events and the cursor are. A missing,
  unreadable or disagreeing row only means the contract is evaluated in full again.
- **Per contract kind** the state holds: anchors and the activation point (`applies_from_event_id`); whether a
  `when` condition was seen; for `must_not` the match count, the first 20 matches and (with an `exception_regex`, which
  is matched against the join of all matches) the match texts and, for Claude Code, its matching tool calls (up to 64 evidence items and 512 call ids, so calls the host refused can be subtracted; over a cap the rule goes to the full evaluator); for `must` the last three regex hits and, for
  `requires_success`, a per-call tracker (how many calls share the latest hit's id, how many results followed, whether
  the first succeeded and whether that success is the required command's at all, see `success_unverifiable`); for `before_regex` the worst verdict, the distinct reasons, the first 20 evidence items and the
  last trigger position, because each triggering action is judged at its own position from the prefix alone. Provider
  evidence classes keep Claude Code's cut results and refused calls and AGY's truncated fields, still-`RUNNING` results, legacy results,
  host-logged proposals and denied proposals.
- **A normal append does not invalidate state.** A state is reused only if all of this holds, otherwise the contract is
  rebuilt by a full pass: same contract (revision **and** content), parser version, evaluator version and `needs` (which
  events had their text loaded); same index generation and cursor path; the stored `(byte_offset, prefix_hash,
  event_count)` is a row of the session's commit log, and the log's newest row is the current cursor; and the number of
  events after the stored offset is what the cursor's count implies. A reindex, a rewrite (`source_changed`), a parser
  bump, a moved or copied source (`alias`/path change), an append that overwrites a stored event, or a cursor written by an
  older server process each start a new generation or fail one of those checks. State and the evaluated cursor are written
  in one transaction, after the index commit they depend on, and never over a newer state.
- **Limits are explicit.** Tables are bounded (hit texts for an `exception_regex`, pending results, blocked proposals, cut
  results, notice targets). A run that would exceed one reports it, evaluates that contract with the full evaluator
  and remembers the limit, so the next audit goes straight there. Where state cannot be summarized faithfully the contract
  stays on full evaluation: a `must_not` whose `exception_regex` has more match text than the cap; an AGY denial whose
  proposal lies beyond the 4,096-event look-back window for a contract that has its own matching proposals; any contract
  disabled by a shadow mismatch (below). Adding a contract that needs text of events no other contract reads changes the
  `needs` and rebuilds every state once. An AGY conversation rebuilds when a hook receipt changes an event that is already
  indexed (see [AGY hook cost](docs/agy-installation.md#hook-cost-what-a-hook-reads)); an ordinary receipt is an append.
- **Shadow verification is a sample with forced exceptions, not a proof.** Some requests also run the full evaluator on the
  same stored events and compare it with the incremental result: the findings and, when every contract is compared, the
  verdict, notes, coverage and trace. On a difference the **full result is returned**, `incremental.incremental_mismatch`
  names the contracts, a row with status `incremental_mismatch` is written to `audit_incremental_health`, that contract is
  evaluated in full in that session until the evaluator version changes, and the schedule below starts again at 1/4. A
  comparison is skipped (`incremental.shadow_skipped: "deadline"`, or `"deferred"` where a resident engine's idle comparison will make it)
  when the request's remaining time budget could not cover the full evaluation, and one that starts and cannot finish in the room it has leaves the
  incremental answer standing, unverified (see [Large sessions and deadlines](#large-sessions-and-deadlines)). Which requests are compared (`incremental.shadow_reasons` says why):
  - **Forced, whatever the rate.** `rebuild`, `premise`, `warmup`: a state was rebuilt in this request, and for the first 4 compared
    requests after a rebuild or a change of premise (below), the session is compared in full. `violation`: the incremental
    answer holds a `violation` or `suspicious` finding of a contract it decides (a wrong block is the costly error; for an action check
    only the contracts that apply to the action count, since only those can block it). `flip`: a contract that the previous request
    of the same kind and action (an audit, or the same proposed action) judged `violation` or `suspicious` is `clean` now (events only
    append, so a real flip is rare, and a wrongly missed violation is the costly error). `high_risk`: a `risk: high` contract applies to the proposed
    action, or this request changed its stored summary. The last three compare **only the contracts that call for it**
    (`incremental.shadow_scope`: a reference evaluation of those contracts alone), so a high-risk contract in the session
    makes no unrelated request compare anything and does not make the other contracts dearer; a scoped comparison does not count
    towards the rate below, and a scoped mismatch is settled by a full comparison in the same request. These three are not repeated for
    an input that was already compared and matched (the same stored events, contracts, request and incremental answer, with every
    contract asked for among those compared: `shadow_skipped: "verified_before"`): the reference is deterministic, so a repeat
    could find nothing. A comparison that did not run (no time) is not remembered as done. A violation, a flip or a changed
    high-risk summary is only known once the first pass has answered, so the comparison it calls for is a second pass over the same
    request. That pass gets the time that is left minus a reserve (a quarter of it, at most one second), and no comparison is started
    unless its estimated cost fits in what is left after that reserve; if the pass cannot finish, because that time is up or the index
    is busy, the answer of the first pass is returned unverified (`shadow_skipped: "deadline"` or
    `"busy"`), the violation stays remembered for the next request of that kind, and no answer is lost. A request that was cancelled,
    or is past its own deadline, is still an error.
  - **Sampled, for everything else, at an adaptive rate.** The rate starts at 1/4 and halves (1/8, 1/16, 1/32) after
    every 8 consecutive comparisons that actually ran, covered every contract and matched; the floor is 1/32. The 4 comparisons of the
    warm-up count towards that streak, so the 1/4 stage of a new session is the 4 sampled matches that follow them. The rate returns to 1/4
    only when a premise of the verification changes (`incremental.shadow_reset` names it): the contract set or one contract's
    revision, the parser or evaluator version, the session generation (a reindex or a rewrite) or a mismatch. A normal append does not
    reset it. A comparison that was skipped for lack of time is no match and does not interrupt the streak. The draw is a hash of
    the schedule state and the cursor, so it is deterministic and reproducible. The schedule is kept per session and provider in
    `audit_shadow_state`, written with the monitor states (a resident engine holds it with them), so a restarted process continues
    it; a row that cannot be read (damaged, written for other constants, or with any invalid field, the remembered comparison
    included) only means "start again at 1/4 with the warm-up", which compares more, never less. Parallel requests cannot undo a
    reset: an older epoch never replaces a newer one, in the table or in the memory a resident engine holds (and a reset another
    process stored wins over the older schedule a resident engine still holds); two writers of the same epoch keep each other's
    remembered violations, each replacing only the entries of the request kinds it handled. The action-check and the audit variants keep monitor states of their own, so the first request of
    each variant is a rebuild and owes the warm-up again.
  - **These constants are not the roundtable's, on purpose.** Issue #18 R2 asked for 1/4 down to 1/64, halving after 32 matches and a
    warm-up of 8. The review chair changed that to the constants above after measuring the policy functions themselves
    (`scripts/measure_shadow_detection.py --share`, `--session` and `--simulate`, 300 seeds and 20,000 draws of a synthetic session;
    `--constants 4,8,16,32,64/32/8` replays the original ones). With the original constants a session compared more of its requests than
    the fixed 1 in 16 until it had run about 2,000 requests, so no p95 got lower in a session of a few hundred requests. Share of the
    requests compared after 10 / 100 / 300 / 1,000 / 2,000 / 5,000 requests: the fixed 1 in 16 (plus its rebuild at the first request)
    15.3% / 7.2% / 6.6% / 6.4% / 6.3% / 6.3%, the original
    constants 85.1% / 30.5% / 18.8% / 10.0% / 6.5% / 3.5%, the constants as built
    54.8% / 16.8% / 8.9% / 4.9% / 4.0% / 3.5%. Comparisons in a session of 300 / 1,000 / 3,000
    requests, and the mean number of requests a missed violation stays unnoticed in it (starting at ten times spread over the session):
    1 in 16 19 / 63 / 188 and 15 / 16 / 16; original constants
    57 / 100 / 146 and 6 / 14 / 38; as built 27 / 49 / 112 and
    15 / 26 / 30. A floor of 1/64 (1.6% of the requests) buys no percentile: it is still above the 1% a p99 sees, and it doubles
    the time a silently missed violation goes unnoticed at the floor (45 / 188 / 627 requests, against 22 / 95 / 269 at 1/32); 1/32 (3.1%) is
    below the 5% a p95 sees, which is the point of an adaptive floor. The constants are `audit.shadow.RATE_DENOMINATORS`, `HALVE_AFTER` and
    `WARMUP_REQUESTS`.
  - **Settings.** `SELF_DIRECT_AUDIT_SHADOW_EVERY` is `adaptive` by default. An integer pins the previous fixed behaviour and switches
    adaptation off, the forced comparisons stay: `1` compares every audit (the test suite does), `0` only the forced comparisons,
    `N` about one audit in `N` (`16` was the previous default). `audit_status` reports the policy, the schedule of a session
    (`rate`, `streak`, `forced_comparisons_left`, `last_match_*`, `reset_reason`), the contracts whose incremental path is switched off and
    the ranges that need review under `incremental_verification`.
  - **Idle comparison (resident engines only).** The daemon's engines and the stdio MCP server (never a hook process) compare the latest state of a
    session with a full evaluation once the engine has been idle for `SELF_DIRECT_AUDIT_SHADOW_IDLE_SEC` seconds (default 2; `0` switches
    it off), if the session answered requests without a full comparison since its last match (a comparison skipped for lack of time is such a
    request, `"deferred"`), or if an audit of it was cut by its deadline before its monitor states existed: the job audits the session with the budget of
    a background job and keeps the states (see [Large sessions and deadlines](#large-sessions-and-deadlines)). One job is pending per session and one runs at a
    time; it never blocks a request (a request that arrives cancels the running job at its next cooperative check, and the session stays
    pending). On a mismatch the contract's incremental path is disabled, the rate returns to 1/4 and the requests answered since the
    last match are recorded as a range that **needs review** (`audit_review_ranges`, reported by `audit_status`; a mismatch found by
    a request records one too). The three changes are one transaction, all or none, and not undone by a cancellation that arrives
    meanwhile; if the index cannot be written the answer is still the full evaluator's and the report says
    `incremental.mismatch_not_recorded`, the session stays pending and the next comparison finds the difference again. The range
    starts at the persisted last match, so it is recorded after a restart too; the count of requests in doubt (of both kinds) is
    known only to a resident engine that was running at the last match, and the detail of the range says when it is not. A job that a
    request cancelled is not a failure and is retried as soon as the engine is idle; a job that ran out of its own budget, found the
    index busy, failed or could not compare backs off for 30 seconds, doubling to 15 minutes, and sessions are tried in turn, so one
    that keeps failing cannot starve the others; a session whose contracts all run on the full evaluator has nothing left to verify and is
    dropped. The worker never reopens an engine that was closed. The job reads the events under the index's file lock like a request
    (while it reads them, not while it evaluates), and only requests to its own engine cancel it: a request of *another process*
    that arrives meanwhile (a hook process that does not use the daemon) waits up to `SELF_DIRECT_LOCK_WAIT_TIMEOUT_SEC` (1 second) and is
    then answered `index_busy`, which a hook treats as unknown. It does not retroactively guarantee earlier responses: a match says
    the latest state agrees with the full evaluator, nothing about the answers given before it.
  - **What this does not promise.** Not every response equals the full evaluation. An answer that is wrongly **clean** passes between
    two comparisons unnoticed; a violation, a violation that turns clean and a high-risk contract are checked in the request that
    carries them, a silent miss is not. Fault injection (`scripts/measure_shadow_detection.py`: a monitor that ignores forbidden
    events, in a synthetic Codex session with five contracts; requests counted from the one that carries the violation,
    1 = noticed in it) gives, as median / p95 / max requests until the defect was noticed (the wrong answers returned before that
    are one fewer). Engine runs use 200 seeds; `--simulate` plays the schedule alone for 20,000 draws, which is what the tails need:

    | a violation the monitor misses (answers `clean`) | previous fixed 1/16 | adaptive, engine | adaptive, `--simulate` |
    |---|---|---|---|
    | in the warm-up (request 3) | not applicable | 1 / 1 / 1 | 1 / 1 / 1 |
    | just after it (request 9) | 11.5 / 53 / 92 | 3 / 12 / 36 | 3 / 11 / 64 (1/16: 11 / 47 / 138) |
    | request 40 | 10.5 / 47 / 123 | 5 / 23 / 47 | 6 / 23 / 73 |
    | at the floor (1/32) | not applicable | 19.5 / 77 / 155 | 22 / 95 / 269 |

    | other defects, noticed in the request that carries them | previous fixed 1/16 | adaptive, cold start (20 seeds) | adaptive, at the floor (20 seeds) |
    |---|---|---|---|
    | a block the full evaluator would not give (`violation` for `clean`) | 12 / 46 / 92 | 1 / 1 / 1 | 1 / 1 / 1 |
    | a violation that turns clean | 10 / 41 / 88 | 1 / 1 / 1 | 1 / 1 / 1 |
    | a high-risk obligation that wrongly says `clean` | 12 / 52 / 113 | 1 / 1 / 1 | 1 / 1 / 1 |

    (The previous-1/16 column is the earlier run of the same harness against the commit this work started from, 200 seeds; the
    fixed schedule is the same code in the merged tree. `tests/test_shadow_detection.py` repeats the forced rows at the floor and
    with `SELF_DIRECT_AUDIT_SHADOW_EVERY=0`.) The adaptive schedule trades a longer worst case for a lower steady cost: after a long
    clean streak a silently missed violation is noticed after 22 / 95 / 269 requests (median / p95 / max) where 1 in 16 needs
    11 / 47 / 138, and every request in between returns the wrong `clean`. It still compares *more* than 1 in 16 did in a young
    session (54.8% of the first 10 requests, 16.8% of the first 100, 8.9% of the first 300) and falls below the 6.25% of 1/16
    after about 600 requests (4.9% over 1,000, 4.0% over 2,000, 3.5% over 5,000; 3.1% at the floor), where the original
    constants needed about 2,000.
- **Small sessions.** A session with fewer than `SELF_DIRECT_AUDIT_INCREMENTAL_MIN_EVENTS` stored events (default 500)
  is evaluated in full and no state is kept (`incremental.mode: "full"`, `below_min_events`). Below that size one full
  evaluation costs less than storing states, rebuilding them and shadow-checking the rebuild; on the scripted evaluation
  harness (short sessions) keeping state roughly doubled the Codex `Stop` latency. `0` keeps state for every session.
  A resident engine that holds its state writes no commit for it, which lowers the break-even to about 250
  events; the default is unchanged (see [Warm judgement benchmark](#warm-judgement-benchmark)).
- **Reporting.** Results carry an `incremental` object: `mode` (`delta` reads only appended events, `full` reads all),
  `events_read`, `rebuilt` (contract to reason), `fallback` (contracts on full evaluation and why), `shadow`,
  `shadow_reasons`, `shadow_scope`, `shadow_rate`, `shadow_reset`, `shadow_compared`, `incremental_mismatch`, `retry` and `state` (`saved`; `unchanged`; `deferred` when a resident engine holds the write, see [Held writes](#lifecycle); or why it was not written). `SELF_DIRECT_AUDIT_INCREMENTAL=false` switches the whole feature off.
- **With source verification.** A state describes the stored events, not how the source was verified: it is written
  before the fast-verification gate and is the same whether the request verified `fast` or `full`. If a later full
  verification finds a rewrite, the reindex starts a new generation and deletes the states. A cursor that migrates from a
  plain digest to block digests, or that an older process rewrote, no longer matches the commit log, so states keyed to
  the old digest are rebuilt and never reused.
- **With contract time limits.** Contract patterns run on the time-limited engine. A monitor whose pattern times out
  takes no more events and is dropped for that run; its contract is judged (`unknown`, never `clean`) by the full
  evaluator, its stored state is deleted and nothing is marked unhealthy. When a shadow run's full evaluation itself
  times out for a contract, the full (`unknown`) answer is returned, the contract's state is dropped, and it is reported as
  `incremental.shadow_skipped: "reference_regex_timeout"`, not as a mismatch.
- **What it does not do.** It does not make an unknown clean and does not change coverage: unread bytes, parse errors,
  unsupported events and rewrites are judged by the unchanged ingestion and coverage code, and a state describes the index,
  not the source file. The source verification (hashing) that precedes every audit is a separate cost; see
  `scripts/bench_audit.py` for a before/after measurement on a synthetic session of any size.

### Source capture

Sync captures at most the length measured at the first `fstat`, in 1 MiB blocks. It keeps only small hash state: the SHA-256 of every complete 1 MiB block (the [block digests](#source-integrity-full-and-fast-verification)), and for a source whose cursor is a plain digest (`.jsonl.zst`, AGY, or an index written before this version) SHA-256 states at the cursor, at the last complete record and at the end. No file bytes are retained, so memory does not scale with the session size.

- **One full scan of the prefix per request.** The capture pass is the scan. Every later stage of the same request (the two ingestion stages, coverage, the audit's own stage check) re-reads a bounded amount instead of hashing the prefix again: the file identity, `size >= captured length` and the SHA-256 of the last 64 KiB as it was streamed during the pass. Before this version one `check_action` hashed the whole prefix five times (measured: 5.0 passes on a 300 MB session; now 1.0). `validate_prefix()` is that bounded re-check; `validate_prefix(locked=True)` is the same without any prefix re-read, for use under the shared publication lock; `validate_prefix(full=True)` re-reads the whole prefix and is used only when asked for or when the cheap signals demand it (below). A mismatch, once detected, stays failed.
- **Cheap signals escalate, they never excuse.** Without growth, an unchanged size with a moved modification time means the file was written (or touched). The unlocked check then settles it with a full re-read: identical bytes pass and the new time becomes the baseline; different bytes fail. Equal stamps prove nothing; they are never a reason to skip a check that would otherwise be made.
- **A change during preparation is retried, then reported as before.** If a later stage finds that the file was rewritten, truncated or replaced between capture and publication, the observation is discarded and the file is captured again, in full, at most three times in all. If it still does not hold still, the call fails with `source changed during preparation` and nothing is published, as before.
- Growth after the first measurement is not an error. An overwrite, truncation or replacement during the pass is. If the file grew while it was being hashed, the bounded check compares the last 64 KiB with the bytes that were streamed; this no longer costs a second scan.
- `validate()` compares metadata only (device, inode, size, mtime, ctime). Equal stamps do not prove equal content.
- `stat` and `fstat` can give `st_ctime` different meaning on some runtimes (for example 9p/drvfs), so full stamps are compared only within the same API and identity, size and mtime across them. On Windows `st_ctime` is the creation time and is never used as evidence of change. When either side reports `st_ino` 0, which some Windows filesystems do, the inode comparison is skipped and size, mtime and the hash decide.
- A `.jsonl.zst` source is an immutable snapshot: its capture is one streaming decode, and its later checks re-read and hash the compressed file (which is small), so any change to it is a rewrite.
- **Limits, stated precisely.** Capture and validation are not an atomic snapshot. A rewrite that leaves the identity, the size, the last 64 KiB and the modification time unchanged and lies outside the last 64 KiB, made after the capture pass of the same request, is accepted by that request's later stages. So is a rewrite outside the last 64 KiB made while the file also grew (the modification time then moves legitimately). The next full scan (the next sync's capture, or a full verification) compares the stored cursor hash and reindexes. This is the same residual window the lock-time check always had; it now starts at the end of the capture pass for every later stage.

### Source integrity: full and fast verification

Re-hashing a 300 MB session on every hook made the cost of a tool-call check proportional to the size of the session file. A cursor of a plain file therefore keeps, next to the byte offset, the **block digests** of the indexed prefix: the SHA-256 of every complete 1 MiB block, the digest of the partial last block, the newline count and a root over all of it (`blocks1:<root>` is the cursor's `prefix_hash`). They are built while the file is read anyway and extended, never recomputed, as the cursor advances: a sync hashes only the new complete blocks and the partial tail. The table lives in the index metadata (`source_blocks`, created by a schema migration on first open; existing data is untouched) and is bound to exactly one cursor publication. A table that is missing, stale (a cursor written by an older process), damaged (its root does not add up) or for another path is never trusted: the next request is a full verification that rebuilds it.

Every `sync_session`, `check_action` and `audit_session` response, and the context a Codex hook emits, carries an `integrity` object:

```json
"integrity": {
  "mode": "fast",
  "full_verified_through_byte": 314572800,
  "full_verified_at": "2026-10-15T12:00:00.123456+00:00",
  "source_generation": "g1:830-269c96:314572800:5b0e8e1b98536aaf",
  "index_generation": "i1:93d3c13bb79981cf",
  "fast_checked": {"tail": true, "recent": [309, 310, 311, 312], "sampled": [3, 8, 41, 77, 90, 134, 201, 250]}
}
```

| Field | Meaning |
|---|---|
| `mode` | How **this request** checked the source: `full` (every block of the prefix was read and compared) or `fast` (see below). |
| `full_verified_through_byte`, `full_verified_at` | The last verification of **every** block of the indexed prefix: through which byte, and when. They describe a **past event**. They are not a statement about the source as it is now, and a `fast` response must never be read as a guarantee about the whole current source. |
| `source_generation` | Names the content this request observed: file identity, covered length and a digest of what was read. It changes with the bytes, not with the clock. |
| `index_generation` | Names one publication of the session's index state; it changes with every cursor write. |
| `fast_checked` | (`fast` only) The blocks that were compared: the tail, the latest blocks and the random sample. |
| `full_reason` | (`full` only) Why the verification was full beyond the caller asking for it. `first_sync` (nothing indexed yet) and `legacy_source` (AGY, `.jsonl.zst`) can appear on any request; the others only on a request that asked for `fast`: `migration`, `parser_updated`, `path_changed`, `no_record`, `interval_elapsed`, `identity_changed`, `shrunk`, `same_size_modified`, `block_mismatch`, `tail_mismatch`, or `evidence_beyond_verified_region`. `interval_elapsed` also carries `verification_lag_sec`: how far past its deadline the verification was. |
| `reindex_reason` | (when the index was rebuilt) Why; `source_changed` means the stored digests no longer matched the source and the index was rebuilt from the source as it is now. |

**Which operations verify how.**

| Operation | Source verification |
|---|---|
| `audit_session` | Always `full`. There is no fast variant. |
| `check_action`, `sync_session` | `full` by default. `verify="fast"` is for hook-time callers: the Codex `PreToolUse` hook, the Claude Code and AGY hook adapters and the OMP extension pass it. It is never chosen implicitly. |
| Stop checkpoints, `auto_continue` decisions | `full`. A restart of the agent needs `integrity.mode == "full"` on a complete, current audit; a fast or missing `integrity` confirms nothing. |
| A `fast` request when the last full verification is older than `SELF_DIRECT_INTEGRITY_FULL_VERIFY_INTERVAL_SEC` (default 300) | Becomes `full` (`full_reason: "interval_elapsed"`). `0` makes every check full. |
| Resident servers (Codex, OMP, the stdio server) | One background thread verifies sources in full after `SELF_DIRECT_INTEGRITY_BACKGROUND_IDLE_SEC` (default 2) of idleness, and starts early enough to beat the interval during continuous activity. It holds no engine lock while reading, runs one scan at a time (at most `SELF_DIRECT_INTEGRITY_BACKGROUND_MAX_SCANS` per wake-up), never blocks a request and reports a deadline it could not meet (`audit_status` shows `integrity_worker`: scans, deferrals, errors, `missed_deadlines`, `max_lag_sec`). Short-lived hook processes (Claude Code, AGY) never do background work. |
| AGY sources, `.jsonl.zst` sources | Always `full` (`full_reason: "legacy_source"`): AGY hook receipts bind to plain prefix digests, and a compressed snapshot is validated by the digest of its compressed bytes. |

**What a fast check does.** Fast verification compares, against the stored digests: the file identity and size (it may only grow); the partial tail block; the latest `SELF_DIRECT_INTEGRITY_FAST_RECENT_BLOCKS` blocks (default 4); and `SELF_DIRECT_INTEGRITY_FAST_SAMPLE_BLOCKS` (default 8) random other blocks, drawn from fresh system randomness for every call. It then parses only the bytes appended after the cursor, each of them hashed block by block while consumed. No other block is read. Cost is a few MiB plus the appended bytes, whatever the size of the file.

Measured with `scripts/bench_integrity.py` on a synthetic 314.6 MB Codex session of 75,000 events (WSL2, ext4, Python 3.12; median of three runs; "passes" is the bytes through SHA-256 divided by the file size):

| Operation | Before: CPU (passes) | After: CPU (passes) |
|---|---|---|
| `check_action`, nothing appended, prohibition contracts, hook-time (`fast`) | 1.13 s (5.00) | 0.035 s (0.04) |
| the same after 5 events were appended | 1.27 s (5.00) | 0.11 s (0.04) |
| the same with a prerequisite gate (evaluates the indexed history) | 1.85 s (5.00) | 0.60 s (0.04); 0.02 s of it verifies the source, the rest evaluates 75,000 events |
| `audit_session` (full verification) | 1.69 s (5.00) | 1.02 s (1.00); 0.40 s of it verifies the source |
| first sync (authority only) | 22.2 s (6.03) | 21.4 s (3.03) |

The cost of the gate row is the evaluation of the indexed history (it grows with the number of events, not with the bytes of the file); making that incremental is separate work.

Cheap signals only ever escalate to full: a different file identity (a new inode), a smaller file, an unchanged size with a moved modification time, a block or tail digest that does not match. A mismatch is `source_changed` and a reindex, exactly as a mismatch of the whole-prefix hash always was; it is a content finding, never a runtime failure to fail open on. `reindex_reason: "source_changed"` is reported as before, and a stale index is never allowed to certify the new bytes: extending the table copies the digests of blocks that were not compared, so a rewrite the fast check missed is still found by the next full verification.

**What a fast verdict may rely on.** Contract evaluation is the same code and the same indexed history as in a full check; nothing is cached across calls. A fast verdict adds three rules: a `clean` finding that rests on a satisfied prerequisite is accepted only when every event it cites lies inside the region that was last verified in full (it then carries `verified_through`); otherwise the request is settled in full before it answers (`full_reason: "evidence_beyond_verified_region"`); and a fast verdict never turns `unknown` or a coverage gap (`unread_bytes`, parse errors, unsupported events) into a satisfied obligation. A fast answer is labelled `fast` and is never presented as `full`.

**Residual risks, stated precisely.**

- A rewrite of a block that no fast check compared (it is not in the tail, not among the latest blocks and was not drawn) is not noticed by fast checks until a full verification reads it: at the latest `SELF_DIRECT_INTEGRITY_FULL_VERIFY_INTERVAL_SEC` after the previous one, at the next `audit_session`, Stop checkpoint or background scan, whichever comes first. Until then a fast verdict is computed from the index as it was verified, and says `fast`.
- A source that is rewritten and restored between two verifications (every block compared is, again, what it was) is not detected, and no claim to the contrary is made.
- The file identity, size and modification time that the table remembers are advisory hints. They are not compared across machines as a guarantee (a Windows and a WSL process can see different `st_dev`/`st_ino` for one file); a difference only costs a full verification. File-system notifications are not used as evidence.
- The residual window of a request's later stages is the one described under [Source capture](#source-capture).

**Compatibility.** The cursor's `prefix_hash` is a plain SHA-256 until a source gets a table: the first full request after this version migrates it in place, without a reindex (`full_reason: "migration"`). A process of an older version that meets a block cursor does not understand it, treats it as a changed source and reindexes once; restart every process that shares an index after upgrading (see [Upgrading to 0.6.0](#upgrading-to-060)). A plain rollout and its `.jsonl.zst` twin still share one cursor: the cursor's hash form follows the source form.

### SQLite journal mode, index files and backups

The index directory (`SELF_DIRECT_INDEX_DIR`) holds these SQLite files and one JSON file:

| File | What it is | Journal mode |
|---|---|---|
| `meta.sqlite` | **Authority.** Events, cursors, and for AGY the digest of the receipts each session was indexed with. Also derived audit state. | `SELF_DIRECT_SQLITE_JOURNAL` |
| `agy-receipts.sqlite` | **Authority.** AGY host hook receipts and the receipts ledger. | `SELF_DIRECT_SQLITE_JOURNAL` |
| `sparse.sqlite`, `graph.sqlite`, `dense.sqlite` / `vectors-v2-*`, `neograph_checkpoints.sqlite`, `workflow.sqlite` | Derived indexes, GraphRAG and checkpoints. | rollback journal, never changed |
| `claude-hook-state.sqlite`, `agy-hook-state.sqlite` | Hook health and the continue budget. Not authority records; they use the standard library only so that hooks can still report when the runtime cannot import. | rollback journal, never changed |
| `contracts.json`, `checklists/` | Own publication protocol (atomic replace). Not SQLite. | not applicable |

**`SELF_DIRECT_SQLITE_JOURNAL`** is `delete` (the default: the rollback journal, as before this setting existed) or `auto` (write-ahead logging, "WAL", for `meta.sqlite` and `agy-receipts.sqlite` on a file system that is positively identified as local, the rollback journal everywhere else). Both policies use `synchronous=FULL`, set explicitly on every connection. `synchronous=NORMAL` is never used for events, cursors or receipts: in WAL mode it can lose the last acknowledged commits at power loss, and `scripts/bench_commit.py` measures it for reference only.

**Why `auto` is not the default.** WAL makes the commit itself cheaper (one `fsync` of an appended log instead of the journal file, the database and the journal's deletion), but a command hook is a new process that opens the database, commits once and closes it, and SQLite folds the log into the database when the last connection closes: the saving moves to the end of the process. Measured on ext4 (the tables below; 300 interleaved samples per cell), a resident engine (the experimental daemon, the stdio server) is clearly faster, and a fresh process is not: the append transaction (`commit_events`) has a p50 of 21.0 ms with the rollback journal and 6.8 ms with WAL in a resident engine (-68 %; the whole append operation -56 %), but 25.3 ms against 20.4 ms in a fresh process (-20 %) and the whole operation, which is what a command hook experiences, goes from 29.2 ms to 32.2 ms (+10 %). The roundtable's rule asked for at least 25 % lower p50 on the append commit on ext4 on the real paths; the standalone command hook, the default install, does not meet it, so the default stays `delete` and the chooser, the reporting and the WAL-aware copy helpers ship switched off. `auto` is for resident processes: set it together with `SELF_DIRECT_DAEMON=on`, or for a long-running stdio server. Skipping the fold at process exit was tried and is worse (the log grows to the autocheckpoint size and every next process has to recover it; see the benchmark below).

**What `auto` does.**

- **Which file systems.** `journal.choose_journal(path)` returns `JournalChoice(mode, reason)`. On Linux it reads the mount that holds the (symlink-resolved) directory from `/proc/self/mountinfo` after an `os.statvfs` check. Allowed: ext2/3/4, xfs, btrfs, tmpfs, f2fs, zfs, and an `overlay` only when the mount holding its `upperdir` can be shown (same device, one of those types). Denied: `9p` (WSL drive mounts, also reported as `drvfs`), `nfs*`, `cifs`, `smb*`, `fuse*` and `virtiofs`. Anything else (`ntfs3`, `vfat`, an unrecognised type, an unreadable mount table, a read-only file system, another operating system) is `unknown`. On Windows the volume must be a fixed drive (`GetDriveTypeW` is `DRIVE_FIXED`); UNC paths, `\\?\UNC\`, device paths and mapped network drives are refused. The Windows branch is verified by tests that fake the Win32 calls and by the Windows CI job; it could not be measured. **Unknown means DELETE.** That is the old behaviour, not a statement that DELETE is safe on that file system.
- **Reporting.** `audit_status` returns `journal` (policy, `synchronous: "FULL"`, and per database the actual mode, the wanted mode, the reason, a `pending` reason when they differ, and `fallback`) and `journal_fallback`, a list of the databases that use the rollback journal because the file system is `denied:*` or `unknown:*` under `auto` (empty under `delete`, where the reason is `setting:delete`). The same fact is logged once per directory and reason as `journal_fallback ...` at INFO.
- **When the mode changes.** Only when a connection is opened (`apply_journal`), never inside a transaction, and never by waiting longer than 50 ms for another connection. A refused switch (another connection, possibly an older process, holds the file: `SQLITE_BUSY`) keeps the existing mode, is recorded in `pending` and is tried again at the next open (every receipts connection; the next process for the index; for a resident engine, once it has been idle for 30 s or more, from its idle flush). It never fails a request. Switching back with `delete` (or on a file system that is no longer allowed) works the same way, in the other direction. The mode is stored in the database file, so every later connection finds it.
- **Older processes.** A release that knows nothing about this setting opens a WAL database with plain `sqlite3`, reads and writes it, and sees every committed row (`tests/test_journal.py` does exactly that with a second plain connection). Two processes may even use different policies on one index; the file's mode is whatever was set last.
- **Checkpoints.** SQLite's automatic checkpoint (1000 pages) is unchanged. A clean `engine.close()` folds the log of both databases into the files (`PRAGMA wal_checkpoint(TRUNCATE)` with a 250 ms bound for readers), and so does a resident daemon when its idle exit closes its engines. A process that dies without closing leaves a log that the next connection recovers; the kill test below shows nothing acknowledged is lost.

**Copying, moving, backing up and deleting index files.** With `auto` a committed row can live only in `meta.sqlite-wal`. Copying `meta.sqlite` by itself then loses the most recent commits, and a stale `-wal` next to a replaced database file would be replayed into it. Therefore:

- Copy a live index with SQLite's online backup API, never with `cp`: `journal.copy_database(src, dst)` / `journal.copy_index_dir(src, dst)` (every `*.sqlite` / `*.db` through `sqlite3.Connection.backup`, sidecars skipped, other files copied as is), or the `sqlite3` shell's `.backup`. `scripts/bench_audit.py`, `scripts/bench_agy_hooks.py`, `scripts/benchmark_large.py` and the AGY equivalence test use them. A directory can also be copied after every process using it has closed cleanly (its logs are then empty), or after `PRAGMA wal_checkpoint(TRUNCATE)`.
- Delete or replace a database together with its `-wal`, `-shm` and `-journal` files (`journal.remove_database(path)`). The installers (`scripts/install_codex.py`, `scripts/install_claude_hooks.py`) back up configuration files only and never touch the index.

**Receipt and event pairing (AGY).** The receipts live in their own database file, so an event commit and a receipt commit are two commits. They are tied by content, not by a request id: with every cursor the index stores the SHA-256 of the receipt rows the events were derived from (`session_receipt_versions.digest`) and the receipts those events depend on (`session_receipt_deps`), and every status, sync and judgement compares that digest with the digest of the receipts database as it stands. A receipt committed without the events commit that follows it, or an events commit whose receipt commit is missing, differ in that digest and report `receipts_not_indexed` (the coverage is incomplete, so a verdict that depends on it is `unknown`) until a sync brings the two together. Receipts are written before the events that cite them, each with `synchronous=FULL`, so the order survives a crash in either journal mode. A receipt for a step whose result is not indexed yet is still a plain append. Nothing about what a receipt may prove changed; `tests/test_journal_engine.py` pins both directions.

---

## Resident daemon (experimental, off by default)

In production a coding agent's integrations add up to many processes: Claude Code and AGY start a fresh Python process for **every hook event**, each OMP session starts its own stdio MCP server, and AGY starts one per session (16 processes and about 1.3 GB in the case that prompted this, [#15](https://github.com/fox1245/tsukkomi-mcp/issues/15)). The resident daemon replaces them with **one process per user and OS environment**; hooks and MCP servers become thin clients of it. It is a stage-1 skeleton behind a flag: Linux (including WSL) only, off by default, and judged by the same code as the standalone processes.

```
Claude Code / AGY hook event ─ scripts/*_hook.py (stdlib client) ┐
OMP / AGY MCP server ───────── self-directing-mcp (stdlib relay) ┼─ Unix socket ─ python -m self_directing_mcp.daemon
                                                                 ┘                 ├─ hook lane: claude_hooks.run_hook / agy_hooks.run_hook
                                                                                   ├─ heavy lane: the FastMCP tool functions of server.py
                                                                                   └─ one SelfDirectEngine (+ integrity and shadow workers) per resolved-settings key
```

### Turning it on, and off

Set `SELF_DIRECT_DAEMON=on` in the environment of the hook command (the installers' `env`) and of the MCP server entry. The entry points read it from the **process environment** before importing anything else (a `.env` or the TOML cannot switch it; `config.Settings` and the light hook reader validate the same two values, `off` and `on`, identically, and an invalid value is a settings error like any other). `off`, the default, is the explicit **standalone** mode: nothing about hooks or servers changes. On a platform other than Linux `on` behaves as `off` and says so once on stderr (a marker file in `~/.cache/tsukkomi/notices/` keeps later processes quiet).

`self-directing-mcp` and `tsukkomi-mcp` now point at `self_directing_mcp.entry:main`, which decides before importing the MCP SDK. Installations whose console script still points at `server:main` get the relay too (`server.main` delegates), at the cost of the imports. `python -m self_directing_mcp.daemon --status` prints the running daemon's counters (no request content), `--stop` asks it to drain and exit.

### Threat model

The authentication boundary is **the operating-system user**, and nothing finer.

- The runtime directory is `$XDG_RUNTIME_DIR/tsukkomi/` when that variable names a usable directory (yours, not group- or world-writable), else `~/.cache/tsukkomi/run/`. It is created mode 0700 and **verified on every use**: a real directory (never a symbolic link), owned by the current user, mode exactly 0700; otherwise the daemon and the clients refuse it with a clear error and nothing is created or reused. The verified directory is held open as a file descriptor, files are opened relative to it and a long path is reached through `/proc/self/fd/<n>/`, so the check and the use cannot be separated by a swapped path component.
- The socket (mode 0600, no TCP of any kind) lives in that directory, and every accepted connection's `SO_PEERCRED` user id must equal the daemon's; another uid is closed without a word.
- **Processes of the same user are not isolated from each other**: any of them can connect, run hooks and MCP sessions as that user, and drain the daemon. That is no weaker than their access to the index and the contracts file today, and no stronger.
- No credential travels. A client forwards a whitelist of its environment (`SELF_DIRECT_*`, `TSUKKOMI_*`, `HOME`, `CODEX_SESSIONS_DIR`, `CLAUDE_CONFIG_DIR`), never `OPENROUTER_API_KEY` or any name that looks like a key, token, secret or password (the key-file *path* setting is fine); the daemon re-checks what it receives. The daemon reads a key file itself, exactly as a standalone process does, so use `SELF_DIRECT_OPENROUTER_API_KEY_FILE`: an `OPENROUTER_API_KEY` in a client's environment is not forwarded and is ignored by the daemon, which starts with a scrubbed environment and never uses an ambient key. For an MCP session the daemon also reads `.env` in the client's working directory, as the standalone server would.
- `daemon.log` (mode 0600, rotated at 256 KiB) holds event names, sizes, durations and error classes. Request and response bodies, payloads, paths and exception messages are never logged, and a log field that is not a short identifier is dropped.
- A frame is a 4-byte length plus UTF-8 JSON, at most 16 MiB; a zero or oversized length, a body that is not a JSON object, and a connection that stays silent after connecting are protocol errors that close that connection only.

### Lifecycle

- **Start.** The first client that finds no daemon takes a non-blocking lock, starts one detached (own session, stdio on `/dev/null`) and waits for its socket for a bounded time (`AVAILABILITY_FAST_SEC` 0.3 s for `PreToolUse`, `PostToolUse`, `UserPromptSubmit` and `PreInvocation`; `AVAILABILITY_SLOW_SEC` 3 s for `Stop`, `SubagentStop` and `SessionStart`; an MCP relay waits up to 15 s). The interpreter is the checkout's `.venv` Python, else the client's own; `SELF_DIRECT_DAEMON_PYTHON` overrides it. The daemon's environment carries no settings: every request brings its own.
- **One instance.** A lock file (`flock`) is held for the daemon's lifetime; a second daemon exits at once, and a successor started while its predecessor drains waits up to 10 s for the lock. Never two writers.
- **Leaving.** After `SELF_DIRECT_DAEMON_IDLE_EXIT_SEC` (900) with no connection and no request; after a request when its resident set exceeds `SELF_DIRECT_DAEMON_MAX_RSS_MB` (400): it stops accepting, finishes what is running and exits, and the next client starts a new one; on SIGTERM/SIGINT; and when a newer client asks (below). Each exit flushes its counters, closes its engines and removes the socket and the pid file.
- **Judging.** Hook requests run on their own thread pool and MCP tool calls on another, so a long `sync_session` never queues a `PreToolUse`. Requests are bounded per session (`SELF_DIRECT_DAEMON_SESSION_BOUND`, 16) and in total (128); beyond the bound the answer is `unknown(queue_full)`. Each request's environment is a per-request view of `os.environ` (managed names come from the request alone), settings are resolved with `settings_core.load_hook_settings` (hooks) or `Settings` (MCP; `--omp-bridge` validation through the same `server.omp_bridge_settings`), and engines are created lazily, one per resolved-settings key and reused, each with one resident integrity worker and one idle shadow-comparison worker (see [Incremental audits](#incremental-audits)). An engine serializes its own operations exactly as in a standalone process.
- **Held writes.** A hook that appends events and judges them used to make up to five synchronous SQLite commits, an `fsync` each (10 to 25 ms on the benchmark machine; the CPU work of the whole judgement is 4 to 21 ms, and was 4 to 75 ms before the `Stop` hooks stopped reading every stored event for an empty checklist and writing the full-text index): the events and the cursor, the incremental monitor states, the "verified in full at T" stamp of the block table, and for AGY the receipt and the receipts-indexed marker. The daemon's engines (`SelfDirectEngine(defer_derived=True)`) keep the authority commits synchronous (the events and cursor, the AGY receipt) and **hold the four rows that carry no authority**: the monitor states, the verification stamp, the AGY indexed marker and the shadow-comparison schedule (which carries its own premise and epoch) are validated against the commit log, the generation, the cursor and the block table whenever they are read, and a missing, stale or lost one only ever means "rebuild from the events" or "verify in full again", never `clean`. A held row is read back in the same process through the merge rule the database applies (the newer `(generation, byte_offset)` wins; a stamp applies only to the exact table, `synced_at` and `root`, it was made for) and is written in the transaction of the next authority commit of that database, after the daemon has been idle for 1 s, at shutdown, or when 2048 rows or 16 MiB are held. A rewrite or reindex of a session, a dropped or demoted monitor and a replaced block table remove the rows they invalidate at once; a row whose lineage moved on by flush time is dropped instead of written. Another process that shares the index (a standalone hook) sees only what was written, so it rebuilds, which is correct and costs only time. A crash loses a cache. `incremental.state` reports `deferred` for a held write. `SELF_DIRECT_DAEMON_DEFER_DERIVED=off` makes the daemon write everything synchronously, as a standalone process does. Shadow comparison, source verification and coverage are not touched: the whole test suite also passes with every engine deferring (`TSUKKOMI_TEST_DEFER_DERIVED=1 python -m pytest`).
- **Receipts read once per change.** An AGY request read all the receipts of the conversation a dozen times (the authority tokens that detect a change between two phases, the capture of a sync, the checkpoint list). A daemon engine keeps one read-only connection to the receipts database and re-reads a conversation's receipts only after SQLite's own change witness (`PRAGMA data_version`, which differs exactly when another connection committed) moved; the file's identity is compared with the path on every use, so a deleted, replaced or recreated database is never served from the old connection. A write that bypasses SQLite moves no witness (the index store's row-count cache makes the same assumption).

### Failure behaviour: never an in-process fallback

With the flag on, a hook's whole answer comes from the daemon or from a visible fail-open. If the daemon cannot be reached or started the client **never imports the engine and never judges itself**; it answers through `claude_hook_state.fail_open` / `agy_hook_state.fail_open`: no deny, a visible warning with the reason, the failure recorded in `hook_health`, and **`ask` when a contract marked `risk: high` matches the proposed action** (the same policy as every other unavailability).

| Situation | Reason in the warning |
|---|---|
| Daemon unreachable after the availability budget; the daemon died mid-request; the spawned interpreter exited | `daemon_unavailable` |
| More than 3 daemon starts within 5 minutes (recorded in the runtime directory; a daemon that is still starting is waited for, not started twice) | `daemon_unavailable:crash-loop`, also a stderr warning |
| The runtime directory is unsafe (symlink, wrong owner or mode) | `daemon_unavailable:runtime_dir`, reason on stderr |
| No answer within the request budget (`SELF_DIRECT_REQUEST_TIMEOUT_SEC`, default 45 s, plus 5 s) | `daemon_timeout` |
| A session beyond its bound | `queue_full` |
| The daemon runs another version (see below) | `version_mismatch` |
| A payload above the 16 MiB frame limit | `frame_too_large` |

The availability budget is not the judgement deadline: once connected, a request has the normal hook budget. An MCP relay behaves the same way: a failure becomes a JSON-RPC error for the host's first request (`initialize`) plus a non-zero exit (2 for a configuration the daemon rejected, such as an `--omp-bridge` without its key file or native library; 1 otherwise), never a silent standalone server. If the daemon goes away while a session is running (memory cap, version switch, crash), the relay answers the requests that were in flight with a JSON-RPC error, reconnects (starting a successor), replays the host's `initialize` handshake and carries on; five recoveries within a minute end the process.

### Versions

`hello` reports `proto`, `package_version`, `index_schema`, `parser_versions`, `evaluator_version`, `features` and `pid`. A client whose package version or proto is **newer** than the daemon's asks it to drain (it stops accepting, finishes in-flight requests, closes its engines and exits) and starts the successor; **an older client never replaces the daemon**: it gets `unknown(version_mismatch)` and its hook fails open. During any switchover other hooks fail open (`daemon_unavailable`) rather than being judged twice. The version is the package version: two checkouts with the same number share one daemon, so after switching branches without a version bump run `python -m self_directing_mcp.daemon --stop`.

### Observability

MCP tool results the daemon served carry `served_by: "daemon"` (a standalone server's results are unchanged, so existing consumers see no new key). Host hook JSON gets **no** extra key; where a hook was served is recorded in the hook state database that already holds `hook_health` (`<index_dir>/claude-hook-state.sqlite`, `agy-hook-state.sqlite`), in a `served_by` table of counters per event, written at most every 30 s and at exit; a database without `daemon` rows was never served by it. Failures a client could not hand to the daemon are `hook_health` rows as usual.

### Settings

| Setting | Default | Meaning |
|---|---|---|
| `SELF_DIRECT_DAEMON` | `off` | `on`: hooks and the MCP entry point are clients of the daemon (Linux). |
| `SELF_DIRECT_DAEMON_IDLE_EXIT_SEC` | `900` | Daemon-side: leave after this long with no connection and no request. |
| `SELF_DIRECT_DAEMON_MAX_RSS_MB` | `400` | Daemon-side: drain and exit after a request when the resident set exceeds this. |
| `SELF_DIRECT_DAEMON_SESSION_BOUND` | `16` | Daemon-side: queued plus running hook requests per session before `queue_full`. |
| `SELF_DIRECT_DAEMON_DEFER_DERIVED` | `on` | Daemon-side: hold the rows that carry no authority (monitor states, verification stamps, the AGY indexed marker, the shadow-comparison schedule) and write them with the next authority commit, when idle and at exit. `off`: write each synchronously, as a standalone process does. See "Held writes". |
| `SELF_DIRECT_DAEMON_PYTHON` | the `.venv` Python, else the client's | Client-side: interpreter that starts the daemon. |

The `SELF_DIRECT_DAEMON_*` limits are read from the daemon's own environment (clients pass them on when they start it); a malformed value stops the daemon with an error instead of using a default.

### Limits of this stage

Linux and WSL only; one daemon version at a time; the `.env`-in-working-directory behaviour of an MCP server depends on the client's directory; every client of a user must see the same `XDG_RUNTIME_DIR` (or all fall back to the cache directory), or they would start separate daemons (index locking keeps that safe, not efficient); a request that exceeds its budget leaves its worker thread running until its cooperative deadline; the judgement is the standalone judgement (same verdicts, same verification, same shadow comparison), made cheaper by what a resident process can keep (see [Held writes](#lifecycle)), and what is still slow is listed in the measurements below (one synchronous commit per appended request, the shadow comparison, and for AGY the receipts commit); a hook arriving while the daemon is restarting fails open instead of waiting for it. Daemon-side `served_by` for hooks is a counter, not a per-call record.

### Measurements

Measured with `scripts/bench_daemon.py --runs 30` on synthetic data (AMD Ryzen 7 5800X under WSL2, ext4, Python 3.12, other work running on the machine: load average 2.0 to 3.1 during the runs; 1 MB sessions, 10 contracts, local-only, 30 interleaved runs per cell after a discarded warm-up). The "daemon" columns are the real hook script as a fresh process talking to a warm daemon; the standalone columns are the same script with the flag unset on a separate, identically built index. A cell that shows `before → after` compares the daemon commit this warm-path work started from with the result of it, on the same machine and benchmark.

| Event (new process, wall ms) | standalone p50 / p95 | through the daemon p50 / p95 | judgement inside the daemon p50 / p95 | rest of the daemon's request p50 / p95 |
|---|---|---|---|---|
| Claude Code `UserPromptSubmit` | 51 / 56 | **21 / 23** | 0.7 / 0.8 | 0.2 / 0.3 |
| Claude Code `PreToolUse` (nothing appended) | 251 / 282 → 248 / 281 | 53 / 66 → **47 / 58** | 32 / 44 → **25 / 35** | 0.2 / 0.3 |
| Claude Code `PreToolUse` (4 KB appended) | 248 / 285 → 255 / 278 | 56 / 86 → **45 / 66** | 35 / 65 → **24 / 45** | 0.2 / 0.3 |
| Claude Code `Stop` (nothing appended) | 272 / 306 → 252 / 290 | 76 / 93 → **30 / 33** | 55 / 72 → **8.9 / 9.7** | 0.2 / 0.2 |
| Claude Code `Stop` (4 KB appended) | 316 / 352 → 274 / 315 | 117 / 137 → **48 / 81** | 95 / 116 → **27 / 60** | 0.2 / 0.3 |
| AGY `PreToolUse` (nothing appended) | 267 / 295 → 263 / 293 | 72 / 80 → **58 / 67** | 50 / 57 → **37 / 40** | 0.2 / 0.3 |
| AGY `PreToolUse` (4 KB appended) | 294 / 326 → 300 / 324 | 97 / 106 → **79 / 90** | 76 / 83 → **59 / 68** | 0.2 / 0.3 |
| AGY `Stop` (nothing appended) | 295 / 327 → 229 / 278 | 101 / 115 → **35 / 42** | 79 / 91 → **14 / 16** | 0.2 / 0.3 |
| AGY `Stop` (4 KB appended) | 355 / 392 → 261 / 301 | 154 / 180 → **54 / 61** | 133 / 159 → **33 / 38** | 0.2 / 0.3 |

What the daemon removes is the process cost: interpreter start, imports, settings, engine and index setup (about 190 ms of a 250 ms Claude Code `PreToolUse`). The thin client alone costs about 20 ms end to end (12.6 ms of it is the bare interpreter start; the client adds the standard-library imports and one socket round trip), its peak resident set is about 14 MB (a bare interpreter: 11 MB; a standalone judging hook: 68 to 77 MB), and the daemon's own handling beyond the judgement is about 0.25 ms. What is left is the judgement, and the warm-path work cut it from 32 to 133 ms to 9 to 59 ms by removing commits and repeated work, never checks. Profiled with `scripts/bench_warm.py` (below), what remains of a judgement is: one synchronous `fsync`-ed commit for every request that appends events (10 to 15 ms on this machine; AGY makes a second one for the host receipt, and a third when a result binds to its receipt), 4 to 21 ms of CPU, and the **shadow comparison**. Under the fixed 1-in-16 sample these tables were measured with, `Claude Code PreToolUse (nothing appended)` is a request whose offset the deterministic sample selects; with nothing appended every request is at that offset, so this cell measures the comparison (a full evaluation of the same events) in every request: 25 ms, against 4 ms without it (`SELF_DIRECT_AUDIT_SHADOW_EVERY=0`). Because 1 in 16 is more than the 5% a p95 leaves out, the p95 of every cell with appended events is the cost of a request that was shadowed, which is a full evaluation more: with 10 contracts 17 ms on Claude Code `PreToolUse`, 42 ms on `Stop` and 64 ms on AGY `Stop` (its evidence classes make the full evaluation dearer); with 100 contracts 270 ms on Claude Code `Stop` and 360 ms on AGY `Stop`. Which of the 30 requests of a cell are shadowed is chance, so the p95 of those cells moves by 10 to 15 ms between two runs (a second run of this benchmark gave `Stop` with 4 KB appended 29 / 73 ms instead of 27 / 60). That was the fixed sample. The default is now adaptive (see [Incremental audits](#incremental-audits)), and [Shadow comparison cost](#shadow-comparison-cost) repeats these cells before and after.

Idle MCP sessions (12, one shared index, after the handshake and one tool call): twelve standalone servers use 971 MB resident (588 MB proportional); twelve relays plus one daemon use **251 MB** (119 MB proportional): each relay 13.7 to 14.4 MB, the daemon 83.5 MB. After the hook runs above the daemon, holding two engines, was at 99.9 MB. (Measured before the warm-path work; it holds a few more kilobytes per session and is not expected to change.)

Against the roundtable's provisional targets: relay at most 15 MB, met (14.4 MB); daemon at most 120 MB, met (83.5 to 99.9 MB); 12 sessions at most 300 MB in total, met (251 MB); the new-process hook p95 of at most 40 ms is **met for events without a judgement** (`UserPromptSubmit`: 23 ms) and **not met** for the judging events (Claude Code `PreToolUse` 58 / 66 ms, `Stop` 33 / 81 ms; AGY 42 to 90 ms); the daemon-side processing p95 of at most 15 ms is **met for the daemon's own work** (0.3 ms beyond the judgement, 1 ms for `UserPromptSubmit`) and for a warm judgement that appends nothing and is not shadowed (Claude Code `PreToolUse` 4 ms, `Stop` 9.7 ms, AGY `Stop` 16 ms), and **not met** wherever a request appends events or is shadowed (one commit is 10 to 15 ms before any CPU; the shadow comparison is a full evaluation). A `Stop` with 100 contracts (`bench_warm.py --contracts 100`, 60 requests per cell) is 15 / 18 ms at p50 / p95 with nothing appended and 40 / 51 ms with 4 KB appended on Claude Code (112 / 136 ms before), 57 / 66 ms on AGY with the shadow sample off, against the target of 80 ms at p95, **met** except in a window that holds a shadowed request (310 ms on Claude Code and 420 ms on AGY with 100 contracts when every request is shadowed). Two things would still move these numbers: write-ahead logging for `meta.sqlite` and the receipts database, now implemented as the opt-in `SELF_DIRECT_SQLITE_JOURNAL=auto` and measured in the [commit benchmark](#commit-benchmark) (a resident engine's append transaction is 68 % cheaper at p50, a fresh process's whole append is 10 % slower, which is why it is not the default), and a faster or rarer shadow comparison (a weaker check, deliberately not done).

---

## Memory, concurrency and deadlines

Long-running servers sharing one index had grown to many gigabytes in a real deployment ([#5](https://github.com/fox1245/tsukkomi-mcp/issues/5)). The causes were whole-file captures per request, concurrent reparses of the same session and heap that was freed but not returned. 0.6.0 changes:

- **No retained source bytes** (see Source capture).
- **Single-flight per (index, provider, session id).** Concurrent hooks for one session wait for one capture/parse/publish, then find the cursor current. Copies of one session id in several folders share the same flight. Across processes, a state-token compare-and-swap still decides which writer publishes, and a stale capture is retaken instead of rewinding a cursor that another process already moved forward.
- **Streaming reparse.** Parsed rows go through a disk-backed spool (`RowSpool`, memory below 1 MiB) and are published in batches inside one transaction.
- **Per-session id queries.** Sparse and dense catch-up query only the synced session's ids, in batches. Candidate vectors are held as float32 (the optional NumPy backend still stores float64).
- **Slim audit loading.** An audit pages in only the rows its contracts inspect, 512 at a time, and prerequisite search is linear rather than quadratic.
- **Bounded heavy calls.** At most `SELF_DIRECT_MAX_CONCURRENT_CALLS` (default 4) heavy calls run at once: sync, search, audit, `check_action`, checklist, graph, activity and hook calls. A request waits for a slot only within its own deadline; one that expires while queued returns `request_deadline` without starting work. A slot is held until the worker thread really finishes, even if its caller timed out. `list_contracts`, `upsert_contracts`, `revoke_contract`, `get_chunk`, `audit_status` and the `workflow_*` tools bypass the limit.
- **`malloc_trim` on glibc.** When the last running heavy call that took at least 0.5 s finishes (at most once per 5 s), the server calls `malloc_trim(0)`. It is skipped on Windows, macOS and musl, and failures are ignored.
- **`audit_status` no longer scans FTS.** The `sparse_count` reads the `chunk_rows` mirror table instead of counting the FTS5 table while holding the shared lock.

### Measurements

Measured by the maintainer; the real-session figures use private corpora that are not in this repository.

| Case | Before | After |
|---|---|---|
| 100 MB synthetic OMP session, `check_action` and `audit_session` repeated 3 times, 1 thread: peak RSS (VmHWM) | 705 MB | 134 MB |
| Same, 4 concurrent threads | 1,750 MB; first sync `index_busy` after 48.7 s | 198 MB; first sync clean in 12.1 s |
| Same, 8 concurrent threads | 2,304 MB; first sync `index_busy` after 48.0 s | 193 MB; first sync clean in 14.9 s |
| Idle RSS after `gc` (1, 4, 8 threads) | 505, 926, 941 MB | 104, 144, 151 MB |
| First sync of the 100 MB session, 1 thread | 10.4 s | 10.8 s |
| Three processes calling `audit_status` 5 times each on a 365k-chunk Codex index | 2.1 s for the sparse count under the lock | 15 of 15 succeeded, no `index_busy`, 0.045 s per call |
| 733 real OMP sessions synced three times in a row | 178 s per pass (repeated reparse of duplicate session ids) | 7 s per pass, 0 reparses, 27 duplicate pairs recognised as aliases |
| 571 real Codex sessions reparsed with parser v3 | v2: unsupported events in 401 sessions, parse errors in 121 | 0 unsupported, 0 parse errors; peak working set 225 MB in 876 s |

The issue's 19.9 GB observation came from one process after about 3.5 days with a growing ~310 MB session. The roughly 7 GB figure for that case in #5 is an extrapolation from the 100 MB synthetic run, and the PR did not measure a 310 MB live session. Memory under such a session has to be confirmed by observation after deployment.

### Deadlines and errors

The [shared-index fix (#3)](https://github.com/fox1245/tsukkomi-mcp/issues/3) separates stable source preparation, guarded publication and detached audit evaluation. Audit snapshots contain only scoped authoritative event rows and counts plus detached rule content, and deterministic checks do not repair FTS or restore embedding caches. Explicit sync still repairs derived indexes, and sparse/hybrid search materializes deferred canonical evidence before querying.

Audit evaluation and source hashing or parsing do not hold the shared writer lease. Before a result is published, metadata, source observations, rules and AGY receipts are revalidated; changed authority makes the result incomplete (`unknown`) while keeping the captured findings. Startup may add the AGY receipt-version side table without rewriting the six-column cursor schema or changing SQLite journal modes.

Requests share an absolute deadline and cancellation control through nested calls and native callbacks. Non-authorizing responses distinguish `index_busy`, `request_deadline`, `request_cancelled` and `operation_timeout`; unrelated SQLite failures are not converted into successful empty searches. A stopped request cannot begin a later publication, but an admitted commit or one already-running blocking call may finish. This is cooperative stopping, not forced thread termination or proof of rollback.

Default combined writer-lock wait (1 second), audit budget (8 seconds), ordinary budget (45 seconds) and explicit-tool cap (600 seconds) are unchanged. Real writes, initialization, raw captures and derived persistent mutations remain guarded and can still contend. See [implementation and verification boundaries](docs/timeout-fix.md).

### Timeouts and what a deadline leaves behind

A request has one budget: 8 seconds (`SELF_DIRECT_AUDIT_TIMEOUT_SEC`) for hooks and audit-class calls such as `audit_session` and `check_action`, 45 seconds (`SELF_DIRECT_REQUEST_TIMEOUT_SEC`) for the other tools, or the caller's `timeout_ms`. An answer that is not ready within it is `request_deadline`. A sync that the hard deadline stops before its commit leaves the index as it was. A sync of a plain session file stops earlier, at its soft deadline, and commits the prefix it parsed ([Large sessions and deadlines](#large-sessions-and-deadlines)), so repeating the call makes progress; only a sync that cannot keep a prefix keeps none. The worker thread of a request that was answered `request_deadline` runs on until its next cooperative check and holds its heavy-call slot until it ends. What the request itself can still answer is decided by the split below.

**The 60/40 split.** `check_action` and `audit_session` refresh the session's evidence (the sync) before they judge. When the request has a deadline, the sync may use 60 % of the time left (`SYNC_BUDGET_SHARE` in `engine.py`) and judging keeps the rest. If the sync does not finish within its share and the request itself is still live, the stored index is judged, the way a sync that failed for another reason always was ([#26](https://github.com/fox1245/tsukkomi-mcp/issues/26)):

- `coverage.complete` is `false` and `coverage.issues` contains `sync_deadline` (`integrity.reason` too, with `verification: "unavailable"`). The extra issue also keeps `judged_prefix` from being set, so a `clean` "as of the cursor" cannot happen.
- A proposed action that matches a `must_not` regex is a `violation`, and so is a `must_not` match in the stored events of an `audit_session`. Everything else is `unknown` at best, never `clean`. A `must` obligation that rests on missing evidence is `unknown`. With nothing stored, `No applicable enabled contracts or no indexed events.` applies.
- The request's own stop is not absorbed. A deadline that passes, or a client that cancels, while the sync or the judging runs still ends the request as `request_deadline` or `request_cancelled`, and a busy index still ends it as `index_busy`.
- `sync_session` has no fallback and reports its own deadline. `update_checklist` and `get_graph_context` need the whole refresh and keep the whole budget, and so does the idle comparison of a resident engine.
- A sync of a plain session file keeps what it parsed: it stops at its soft deadline, which falls inside its share, and commits the prefix. A session whose sync needs more than 60 % of the budget is therefore judged on the committed prefix with the issue `sync_in_progress` (a `must_not` match in it is still a `violation`; nothing is `clean`) until enough calls have finished it, and then judged whole, where an all-or-nothing sync was `sync_deadline` on every call (see [Large sessions and deadlines](#large-sessions-and-deadlines)). `sync_deadline` remains for a sync that cannot keep its progress: a `.jsonl.zst` source, a record larger than the share, `SELF_DIRECT_SYNC_SLICING=off`, or a re-index that cannot be staged (no room on the index volume). A re-index of a plain file keeps its progress in a spool ([Re-indexing](#re-indexing)): the audit judges the previous index meanwhile, with the issue `sync_in_progress`, or `reindex_ready_to_commit` once everything is staged and only the commit is left.

An answer that did not complete carries `timings_ms` (`gate_wait`, the wait for a heavy-call slot, and `elapsed`, the whole request, in milliseconds) and `budget_ms`, so the time can be placed in the queue or in the work. The reasons remain an open set: a client must not reject one it does not know.

**`audit_status.ops`.** `audit_status` attaches the request statistics of the process to every answer as `ops`, an `index_busy` or `request_deadline` answer included: they sit behind their own short lock and never wait for the engine. They are in memory only (a restart starts at zero), hold no session id, path, contract text, argument or error message, and use closed vocabularies; any other name is counted as `other`.

| Field | Content |
|---|---|
| `since`, `uptime_sec` | When the counting began (ISO time, UTC) and for how long. |
| `buckets_ms` | The inclusive upper bounds of the histogram buckets: 50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 45000 and `"+inf"`. |
| `tools.<name>` | Only for tools that were called. Names: the registered tool names, `hook` (the Codex hook tool) and `other`. |
| `calls`, `outcomes` | Requests, each counted once at its response, and how they ended: `ok`; `degraded` (answered from less evidence than wanted: the stored index after `sync_deadline`, or a session whose sync is still in progress, `sync_in_progress`, or whose staged re-index only waits for its commit, `reindex_ready_to_commit`); `incomplete` (no answer); `client_cancelled`; `error`; `other`. |
| `reasons` | For `incomplete` and `degraded`: `request_deadline`, `index_busy`, `request_cancelled`, `operation_timeout`, `sync_deadline`, `sync_in_progress`, `reindex_ready_to_commit`, `other`. |
| `elapsed_ms`, `gate_wait_ms` | `counts` per bucket (one more than there are bounds) and `max`. The wait is recorded only for calls that queue for a heavy-call slot. |
| `workers` | `abandoned_now`: worker threads still running after their request was answered `request_deadline`; `abandoned_total`; `late_finish_total`. The end of such a worker is not a request. |
| `gate` | `slots` (`SELF_DIRECT_MAX_CONCURRENT_CALLS`) and `in_use` (slots held, abandoned workers included). |

Counted: the MCP tool calls, with the Codex hook tool as `hook`. Not counted: the Claude Code and AGY command hooks, which do not pass the MCP dispatcher, and the `workflow_*` tools. A resident daemon counts across all of its connections.

### Large sessions and deadlines

A source that could not be synced inside the request budget (8 seconds for hooks, `audit_session` and `check_action`, 45 seconds for the other tools) used to be parsed until the hard deadline, rolled back, and started again from zero by the next call: repeating the call only used the slot and the CPU again. On a synthetic Codex session of 300,000 events (72 MB, about 10 seconds in one piece on a loaded development machine), four consecutive syncs with a budget of 1, 4 or 8 seconds each left no cursor and no event.

**What a slice is.** A request that has a finite deadline and syncs a plain session file may stop parsing at a record boundary when its soft deadline passes, and publishes the prefix it parsed in one transaction, exactly like any other append: the events, the cursor, the block table and the metrics together, so the index is always a valid earlier state and a crash or a stop at any point keeps either the previous index or the new prefix. The next call continues from the cursor. Nothing else about the index changes: event ids, anchors, cached vectors, the block table and the metrics are the same as for a sync in one piece, and the commit log is the chain of the slices ending in the row of the single commit (`tests/test_slice_equivalence.py` compares the two row for row, for all five parsers, and an audit of each).

- **Soft deadline.** `now + 0.5 * remaining / 1.3` (parse and commit use about half of the budget and at least a quarter remains for the evaluation and the response; never less than 0.25 s after the start). For `audit_session` and `check_action`, `remaining` is the sync's 60 % share of the request (the 60/40 split above), so an 8-second audit parses for about 1.85 s. The parse always makes a minimum of progress even when its time is already spent: 256 records **and** 1 MiB. The hard deadline is untouched. For `audit_session` and `check_action`, which also judge what they sync, this is only the upper bound: see "A slice that will be judged" below.
- **Capture cap.** `SELF_DIRECT_SYNC_SLICE_BYTES` (default 64 MiB) bounds how much of a source that is far from indexed one slice observes: the capture ends at the last record inside `cursor + cap`, or at the first record end after it when none ends inside. The observation then behaves exactly like one made when the file was shorter.
- **Only additions.** A slice is published only where it adds to what the index holds. A parse that would replace events that exist (a parser upgrade, a source whose stored hash no longer matches, changed AGY receipts, another path) is never published in pieces: the index is not touched until ONE commit replaces them, so a stop or a crash leaves the previous index, its ids, anchors and vectors, untouched. What such a parse produces waits in a spool in the meantime ([Re-indexing](#re-indexing)). A capture that was cut short and then turns out to need a re-index is taken again whole. `.jsonl.zst` sources are never cut either (a forward seek in a compressed stream decodes the prefix again), and never staged.
- **What was consumed is verified.** At a stop the iterator checks what it read against the capture before the prefix can be published: a stop inside a block is compared with the capture's digest of the whole block, so a byte that changed between the capture and the parse is an error, not evidence.

**What a call says.** `sync_session` gets a `sync` object (calls with a deadline only; `SELF_DIRECT_SYNC_SLICING=off` and requests without a deadline answer exactly as before):

| Field | Meaning |
|---|---|
| `state` | `partial` (the index holds a prefix of what was observed, or a re-index is staged and not yet published) or `complete`. |
| `indexed_through` | The cursor: the byte offset up to which events are committed (while a re-index is staged, the cursor of the previous index). |
| `pending_bytes` | Bytes of the file after the cursor (as `coverage.pending_bytes`). |
| `target_bytes` | How far this sync meant to go: the last record boundary it observed, or the size of the file when the capture was cut short. |
| `progress` | `indexed_through / target_bytes`: the committed share, never what was parsed but not yet committed. While a re-index is staged it is the staged share (`reindex.staged_through / target_bytes`), because nothing of the new index is committed yet. |
| `retry_after_ms` | A hint (500) while `partial`; `null` when complete. Advisory: nothing in this package retries by itself. |
| `stopped_by` | `soft_deadline`, `slice_cap` (the capture cap) or `eof` (also for a staged re-index that is complete and waits for its commit). |
| `reindex` | Only for a sync that is a re-index of a plain file with slicing on: see [Re-indexing](#re-indexing). |
| `cost` | What sized the slice. `slice_us` and `eval_us`: the expected microseconds to parse and commit one event and to judge one (`eval_us` is 0 when nothing is judged after this sync, as for `sync_session`, and for a slice that is staged for a re-index: nothing of what it stages is read until it is published); `debt_events`: events already indexed that no monitor state covers; `limited_by`: `share` (the soft deadline above), `cost` (the cost bound, below) or `null` (neither: the end of the file, a forced stop). A re-index that is staged has both `reindex` and `cost`. |

`coverage` then has `backlog_bytes` (records the capture saw that the index does not hold yet) and the issue `sync_in_progress` next to `unread_bytes`, and `complete` is false. Because `sync_in_progress` is a coverage issue, `judged_prefix` is never set while a backlog exists: **a partial prefix is never `clean`, not even "as of the cursor"** (`judged_prefix` stays for pure append). An `audit_session` or `check_action` that meets a backlog syncs one more slice itself and then judges what is committed: a forbidden match in the prefix is a `violation`; a `must_not` that merely found no match, an obligation and a proposed action's prerequisite are `unknown`; a proposed action that matches a prohibition is a `violation` whatever the backlog. Call again to make progress; every call moves the cursor further (`tests/test_slice_convergence.py`: 50,000 events in a handful of calls on a scripted clock, and 30,000 on the real clock with a budget that is a share of the one-shot time on the machine that runs the test, each call inside its budget).

**A slice that will be judged is sized by what judging it costs.** `audit_session` and `check_action` do not only sync: they judge what they synced in the same request, and judging an event can cost as much as parsing and committing it (eighteen heavy contracts: about 52 microseconds an event against 40). With the share alone a slice could be synced that its own request could not judge, and an evaluation that the deadline cut saved nothing: the next call repeated it, and the audit stopped converging although the sync did (measured below). For these two tools the share above is therefore only the upper bound. (A slice that is staged for a re-index, [Re-indexing](#re-indexing), is not sized this way: nothing of what it stages is judged until it is published, and the audit that follows reads the previous index.) The parse also ends, at the same record boundary and at the same clock reads (nothing reads the clock more often), once `time spent so far * 1.3 + records read * the expected seconds to judge one record` reaches 85 % of the time the request has left, less the judging of the events the index already holds without a monitor state (`debt_events`: a request that meets such a backlog has to fold it before it can answer). The parse speed in that sum is the one this slice shows, so a wrong guess about the parse cannot break it, and the minimum progress above still holds.

What judging an event costs is a moving average of what this process measured for the contract set (`eval_cost.py`; smoothing 0.3, the first measurement replaces the prior, and a fold that the deadline stopped counts too). Before a set was measured it is 60 microseconds, more for many contracts (12 plus 4 for each one that needs history, when that is more), and half as much again: a slice that is committed cannot be cut again, so a guess that is too low costs an evaluation that does not fit, and one that is too high only costs a call. It is nothing when no contract needs history (a `check_action` against prohibitions judges nothing that the sync publishes), and with `SELF_DIRECT_AUDIT_INCREMENTAL=false` the slices are those of the share alone. A resident server learns within its first requests. The numbers a partial answer was sized by are in its `sync` object (`sync.cost`); `audit_status` has an `evaluation` object with the model and, for a session, `events_without_state`: how many of its events no monitor state covers.

**A comparison with the full evaluator never takes the answer with it.** The first requests of a session are compared with the full evaluator (the warm-up of [Incremental audits](#incremental-audits)), and that comparison used to run in the same pass as the answer, under the same deadline: on a heavy contract set the first call of a cold session spent on it the room that its answer needed, ended as `request_deadline` and saved nothing. Now the incremental answer comes first. A planned comparison is skipped when it cannot follow it: before its events are read, from what this process has measured of the contract set, and again after the pass, from the speed that very pass folded at (a full evaluation costs about 1.5 folds an event, with a margin of 25 %); the answer says `incremental.shadow_skipped: "deadline"`. These checks can only skip more than the estimate that was there before, never less. One that starts and does not finish in the room it has (the request's time less the reserve for the answer) is cut, and the incremental answer stands, unverified, with its states saved. Where a resident engine's idle comparison exists the skip says `"deferred"` instead: the comparison is owed, not forgotten (the warm-up count does not move), and the idle comparison makes it and counts for it. Which requests are compared, the rates and the warm-up count do not change, and a comparison that an answer called for (a violation, a flip, a high-risk contract) keeps its own second pass and its own fallback.

**What the first audit of a big session costs, and what converging takes.** Measured on synthetic Codex sessions with `scripts/bench_eval_converge.py` (one development machine under ordinary load; every run is the first request of a freshly started server). Per 150,000 events (36 MB), in seconds, for the same audit with two, six and eighteen heavy contracts:

| step | 2 contracts | 6 contracts | 18 heavy |
|---|---|---|---|
| sync (parse and commit), once | 5.4 | 5.4 | 5.4 |
| read the events the audit needs | 1.0 | 1.0 | 0.9 |
| fold them into the monitors | 0.4 | 1.4 | 7.0 |
| the call index (`requires_success` rules), once | | 0.2 | 0.2 |
| the full evaluator, i.e. one comparison | 0.4 | 1.6 | 9.9 |
| saving the monitor states | < 0.01 | < 0.01 | < 0.01 |

The first audit of a cold session is the sync plus the read, the fold and, while the warm-up lasts, a comparison; each of the next four audits of the session read nothing new and still ran one full comparison. Calls to a complete answer equal to the unbounded one (K; `x` = ended by the deadline, the seconds of each call in brackets):

| 18 heavy contracts | before | after |
|---|---|---|
| 150k events, 8 s, cold | none in 14 (all `x`) | **3** (4.1, 6.5, 4.7) |
| 300k events, 8 s, cold | none in 14 (all `x`) | **6** (3.9, 6.2, 6.4, 5.9, 6.0, 1.1) |
| 150k, 45 s, cold | 1 (24.1) | 1 (25.2) |
| 300k, 45 s, cold | 2 (44.8, 23.2) | 2 (23.6, 6.5) |
| 300k, 45 s, synced, no state | 1 (40.6) | 1 (16.3) |
| 150k, 8 s, synced, no state | none in 14 (all `x`) | 12 (11 `x`: a fold of 7 to 8 s against 8 s) |
| 300k, 8 s, synced, no state | none in 14 (all `x`) | none in 14 (a fold of 15 s against 8 s) |
| the two rows above in a resident server | | **2** (`x`, then 0.1 s), after one idle job |

For two and for six contracts judging is cheap, nothing is cut before or after, and K and the seconds of every call are the same (3 calls at 150k and 5 at 300k with an 8-second budget, none longer than 4.2 s). Only the eighteen-contract rows are new. The slices of the eighteen-contract rows are shorter than those of six (`sync.cost.limited_by` says `cost`), and a comparison that the speed of the pass says cannot finish is skipped instead of taking the 45-second calls to 40 seconds.

**One writer per session.** Requests for one session still queue behind a single flight. A request that may slice no longer waits for the leader until its own deadline: it waits at most `min(1 s, a quarter of its time left)` (for `audit_session` and `check_action`, of the sync's 60 % share), so a heavy-call slot is never held idle longer than that, and then judges on what is committed with the issue `sync_in_progress`. The statistics count such an answer, and a `sync_session` that ended partway, as `degraded` with the reason `sync_in_progress`, and a `sync_session` that met a busy session as `incomplete` with the same reason. A direct `sync_session` that meets a busy session is answered with an incomplete response (`ok: false`, `verdict: unknown`, `error: "sync_in_progress"`); a deadline that really ran out still says `request_deadline`. **Reasons and issues are an open set:** every client in this repository (the OMP extension, the Codex hook, the AGY and Claude Code adapters) shows them as opaque strings and reads an unknown one as "unverified" (`tests/test_slice_adapters.py`), and a client outside it should do the same.

**Limits, stated precisely.**

- The guarantee of progress is conditional: one record larger than the whole budget, or I/O that stalls, can still overrun the hard deadline, and the answer is then `unknown`, as before. Nothing is skipped or truncated.
- A re-index is staged, not sliced into the index: its last step is one transaction whose size grows with the session, so a session of more than roughly 150,000 events needs an explicit `sync_session` to publish it from a hook-sized budget, and one of more than about 1.5 million events a `timeout_ms` above the default of 45 seconds ([Re-indexing](#re-indexing)); until it is published, audits report `parser_outdated` (or `source_changed`) and judge the previous events.
- Evaluation is sized, not sliced. Each slice is judged by the call that syncs it and its monitor states are saved at its commit, so a cold session is judged slice by slice. A session that is already indexed but has NO monitor state (a build with a new evaluator version was deployed after the index was made in one piece) has to be folded by one request: there is no commit boundary inside it to save a state at, and a state at any other offset would change the lineage rules (a state is valid only at a logged commit of its session). A re-index that is published starts a new generation of the session, which drops its monitor states, so the first audit after it is such a session too (the commit is started when its own time fits; it does not look at what the audit that follows will cost). That fold costs, per 150,000 events, 0.4 s for two contracts, 1.4 s for six and 7.0 s for eighteen heavy ones, plus 1 s to read the events; when it does not fit the budget every call ends `request_deadline` and keeps nothing (a fold that finishes just before the deadline does keep its states, and the next call is instant; near that edge the number of calls is luck). Three ways out: `audit_session` with a larger `timeout_ms` (up to `SELF_DIRECT_MAX_TOOL_TIMEOUT_SEC`) builds the states once; a resident server notes the stopped audit to its idle comparison (`SELF_DIRECT_AUDIT_SHADOW_IDLE_SEC`, default 2 s), which audits the session with the budget of a background job and keeps the states (a request that arrives cancels it, so it needs a quiet moment as long as the build); and `audit_status.evaluation.session.events_without_state` says how far behind the states are. Folding in chunks between commit boundaries would not help sessions that grow the way live ones do: the commit log keeps the last 256 commits, and for a session that grew by a hook sync after every six events they span the last 7.6 % of 20,000 events.
- The parsers are not modified, so what a parser remembers from record to record is per call: a slice boundary is an append boundary. The only such memory that matters is the Claude Code subagent file's check that all of its records name one agent (`_File.pinned_agent`): a file whose records disagree with each other keeps a record in slices that a single pass rejects (pinned as a strict `xfail` in `tests/test_slice_equivalence.py`; files whose records carry the agent id of their file name are unaffected). OMP's lookup of earlier tool calls reads every stored call of the session on each append (about 10 microseconds per stored call), which a sliced sync repeats once per slice.
- Only requests with a finite deadline slice: the Claude Code and AGY command hooks, which run without one (unless they go through the resident daemon, which gives every request a deadline), still sync in one piece within their host's hook timeout.

`scripts/bench_slicing.py` measures all of this on synthetic sessions: calls to converge and per-call wall time for several budgets, total CPU against a one-shot sync, the capture time against the number of slices, and the cost of the stop checks when no stop happens. With `--reindex` it measures the re-index instead: the time of every staged slice (capture, validation of the spool, parse, spool write) and of the commit against the prediction, the peak memory of the call that commits against a one-shot re-index, the extra disk it needs at its peak and the first audit after it against the first audit after a cold sync.

#### Re-indexing

A parser upgrade (the AGY parser moved to version 8 and the Claude Code parser to 3, so every session of those providers is reparsed on its first sync), a source whose stored hash no longer matches, changed AGY receipts and a retried truncated fallback all make a sync parse the session from byte 0 and replace its events in one transaction. For a session that does not fit the request budget that was all or nothing, and repeating the call parsed it again every time. A request with a finite deadline that syncs a plain file now **stages** such a re-index and publishes it with one commit, so those parser bumps converge on a large session too ([#26](https://github.com/fox1245/tsukkomi-mcp/issues/26), roundtable 8, P2):

- **What is staged, and where.** Each slice of the parse is appended to `<index dir>/reindex/<provider>-<session key>.<target parser version>.spool` (the session key is a hash of the id: no session id is a file name). The rows are the ones the commit will insert (same ids, same JSON), with their flags and tallies and what the next slice's parse needs from this one (OMP: the earliest byte of every tool call; AGY: the receipts the events were derived from). A slice ends with a trailer frame, written after its rows reached the disk (`fsync`), that names the offset it reached and the hash of the source prefix up to it. The progress is the last valid trailer; a torn tail after it is cut off the next time the file is opened, and a slice that was already staged is never staged twice. A re-index that is parsed whole in one call and whose commit fits creates no file at all.
- **The index is not touched.** Until the commit the index holds the previous events, anchors, cached vectors, cursor and block table, byte for byte. Audits and action checks keep judging the previous events: the coverage keeps `parser_outdated` (or `source_changed`), adds `sync_in_progress` while the staging goes on and `reindex_ready_to_commit` when everything is staged and only the commit is left, `complete` is false and `judged_prefix` is never set, so **nothing is `clean`** meanwhile. A forbidden call in the previous events is still a `violation`; an obligation is `unknown`. The sync answers `state: "partial"`, `indexed_through` is the previous cursor, `reindexed` is `false` and `reindex_reason` says why a re-index is under way.
- **One commit.** When the staged range reaches the end of what the capture observed, the sync verifies `[0, target)` (the capture of this very request scanned the whole prefix, and every staged slice was bound to it by the hash of the range it ended at) and commits **once**, with the ordinary `replace`: the events, cursor, block table, metrics, generation and AGY receipt bookkeeping are exactly what a one-shot re-index writes (`tests/test_reindex_equivalence.py` compares the two row for row, for all five parsers and an audit of each), and the spool is deleted after the commit returns.
- **The commit is not started when it would not fit.** It is one transaction that cannot be cut; a transaction that starts and is rolled back by the deadline only wastes the budget. It starts only if `COMMIT_SECONDS_PER_ROW * rows * COMMIT_SAFETY` (18 microseconds a row, measured, times 1.5) is at most 90 % of the time the request has left. Otherwise the answer is `reindex.state: "ready_to_commit"`, the spool is kept, nothing is reported as done, and the next call that has the time publishes it. An explicit `sync_session` (45 seconds, or its `timeout_ms` up to `SELF_DIRECT_MAX_TOOL_TIMEOUT_SEC`) takes the same path and finishes it; its `nudge` says what to do. **The wall:** a hook's sync (60 % of 8 seconds) can publish at most about 160,000 events, fewer once the capture and the checks of the call have taken their share, and a `sync_session` of 45 seconds about 1.5 million; a larger session stays `ready_to_commit` in hooks, and audits judge the previous index with the issue `reindex_ready_to_commit`, until an explicit `sync_session` publishes it.
- **The `reindex` object** of `sync`: `{state, staged_through, target_bytes, rows, reason}`. `state` is `staging`; `ready_to_commit` (`reason: "commit_exceeds_budget"`); `committed`; `discarded` (a spool that no longer matched was thrown away and the staging started over from byte 0 in the same request; `reason` says why); or `disabled` (no spool for this request, `reason`: `disk`, `unsafe` or `unavailable`: the sync is the all-or-nothing re-index of before). When several apply the order is `disabled`, `committed`, `ready_to_commit`, `discarded`, `staging`.
- **What discards a spool** (the spool is deleted; the index never is): a different spool format, target or base parser version (`target_version`, `base_version`; `orphan` when the index already stores the target: another process finished); a cursor rewritten since the spool started (`index_changed`); another source file, a changed head (the first 64 KiB), a source that ends before the staged range, a changed prefix hash at the staged offset (`source_identity`, `source_changed`, `source_shrunk`, `prefix_changed`); AGY hook receipts that changed in a way that affects an event already staged (`receipts_changed`; a receipt for a step that is not staged yet does not restart a session that is still being written); a frame that does not verify (`corrupt`, found when the file is first read by a process, and again by the commit, which rolls back and discards); an unreadable header (`unreadable`); and the room of the index volume (`disk`). The evaluator version is not part of the binding: a spool holds parser output only. A spool whose target version is not the one this code would reindex to, or whose header cannot be read, is deleted by any sync of its session; so is one that a crash left behind after its commit. A spool of another session is never touched.
- **A second copy of the text.** A spool holds the rows of the session as the index will, so it is in the same trust domain as the index: the directory is created 0700 and the file 0600 (POSIX; Windows ignores the mode bits), a symlink or a directory of another user is refused, nothing but the sync reads it, nothing sends it anywhere and no row text is logged. It is deleted when its commit succeeds or when anything it was built on changes. A spool of a session that is never synced again stays until that session is, and is safe to delete by hand: it is only parser output that can be made again.
- **Disk.** No spool is started when the free space of the index volume is below 9 times the size of the source (twice the measured 4.2 spool bytes per source byte; the commit then grows the database by about as much again while the spool still exists). A spool that is already there is discarded when the room runs out, giving its space back.
- **One writer.** The spool is locked with `fcntl.flock` while a process has it open. On Windows it is `msvcrt.locking` of one byte far beyond the data: the Windows smoke job takes and releases it, but a second process meeting it is tested on POSIX only, and the lock has not been run on Windows by hand. The header, the trailers, the checksums, the binding to the cursor stamp and the per-session flight stay what protects a spool where the lock does not.
- **Limits.** Evaluation is not staged. The staged range is verified against each request's own capture (one pass over the source per call, so the capture time grows with the number of calls, not with the number of rows already staged), and the first call a process makes on a spool re-reads it to verify every checksum. Staging is for plain files (`SELF_DIRECT_SYNC_SLICING=auto`, a finite deadline); `.jsonl.zst` sources, `SELF_DIRECT_SYNC_SLICING=off` and requests without a deadline keep the all-or-nothing re-index.

`tests/test_reindex_spool.py`, `test_reindex_equivalence.py`, `test_reindex_staging.py`, `test_reindex_faults.py` (a stop at every cooperative check of a Codex re-index and at a sample of the others, a process that is killed after the row frames, after the trailer, with the spool complete, inside the commit transaction and after the commit, and a request that is stopped right after a spooled slice or after the commit) and `test_reindex_convergence.py` (a 150,000-event session with budgets of about 4 and 8 seconds on the development machine; on any other they are multiples of the time the one-shot re-index took there, so that a sanitizer build, which is three to four times slower, gives the same answer) pin all of this on synthetic sessions.

`scripts/bench_eval_converge.py` measures what follows the sync, the judging: repeated `audit_session` calls under a budget of 8 or 45 seconds on synthetic sessions of 150,000 and 300,000 events, for a cold session and for one that is indexed but has no monitor state, with two, six and eighteen contracts (`--tool check` repeats `check_action` for a gated action instead, and the scenarios `cold+idle` and `synced+idle` give a resident server's idle comparison its turn after every call that did not complete); it reports the calls needed (K), the seconds of every call (a call the deadline cut is marked `x`), whether every call stayed inside its budget, and whether the final answer equals the unbounded one-shot answer, finding for finding. It resets the cost model before every run (the first request of a freshly started server) and also runs on a checkout without it, which is how the numbers of a change are compared with those before it. `tests/test_eval_converges.py` runs the same questions in virtual time (a price in seconds for parsing a record, folding an event, running the full evaluator; the deadlines read that clock), so that the calls, the stops and the answers do not depend on the machine. `tests/test_eval_reindex.py` pins how this fits with a staged re-index: a staged slice never reads the sizing numbers, the appended slice of the same session afterwards is cost-limited, and a request that stops while it stages is not noted to the idle comparison (scripted clocks, no wall clock).

---

## Upgrading to 0.7.0

1. **Restart every process that uses the index** (Codex, OMP, AGY, Claude Code, any manual `python -m self_directing_mcp`). A running process keeps the old code in memory.
2. **Reinstall the package, do not copy files.** `regex>=2024.11.6` is a new dependency, and the console entry points now point at `self_directing_mcp.entry:main`. An editable checkout needs `pip install -e .` again; a wheel needs `--reinstall`.
3. **Parser and evaluator bumps are lazy.** The AGY parser is version 8 and the Claude Code parser version 3 (Codex 3 and OMP 8 did not change): a session of those providers is re-parsed on its first touch, and a re-index that does not fit one request is staged in `<index dir>/reindex/` (a spool with the session text, mode 0600, deleted when done) and published by one commit; `sync_session` with a larger `timeout_ms` finishes one that is ready. The evaluator version is 8: the first audit of a session builds its monitor states again, and `audit_status.evaluation` shows the lag.
4. **Existing indexes are preserved.** No migration beyond new tables created on open. Take a copy of the index directory before the first start if the index is irreplaceable.
5. **New defaults to know.** `SELF_DIRECT_SYNC_SLICING=auto` (a request with a deadline publishes partial progress; `off` restores the all-or-nothing sync) and `SELF_DIRECT_SYNC_SLICE_BYTES=67108864`. The daemon and the write-ahead journal stay off.
6. **Re-verify.** `audit_status` on every host: `ops` and `evaluation` are present, `degraded_flags` is empty. For Codex run `python scripts/install_codex.py --verify`.

---

## Upgrading to 0.6.0

1. **Restart every process that uses the index.** Stop and restart all MCP servers and extensions that share an index (Codex, OMP, AGY, any manual `python -m self_directing_mcp`) from the new package. A running process keeps the old code and old parser, and a process on an older version can write cursors that a newer one must then reparse. Updating a checkout or a wheel does not update processes that are already running.
2. **Parser bumps reparse lazily.** The OMP parser is now version 8 (0.5.0 had 2) and the Codex parser version 3 (was 2). The AGY parser is version 7 and the new Claude Code parser is version 2. Each session is reparsed once, on its next sync or audit, within the bounded-memory path. There is no upfront migration step and no storm of reparses.
3. **Existing indexes are preserved.** Events, anchors, model-specific vector caches, contracts and checklists are kept. Unchanged event ids and cached vectors survive a reparse. Obsolete entries are cleaned through a durable journal, so an interrupted upgrade resumes.
4. **Check the index location.** The default index is `~/.self_direct_index` unless `SELF_DIRECT_INDEX_DIR` or `index_dir` in TOML overrides it. Older versions defaulted to a `.self_direct_index` inside the checkout. An install that relied on that default must set `SELF_DIRECT_INDEX_DIR` (or TOML `index_dir`) to the existing absolute directory before restarting; nothing is moved or copied automatically. The Codex installer and managed OMP registrations already pass an explicit path.
5. **Check how the key file is selected.** A checkout `.env` is no longer treated as an authoritative key file. Select it explicitly with `SELF_DIRECT_OPENROUTER_API_KEY_FILE` or TOML `openrouter_api_key_file`, or supply `OPENROUTER_API_KEY` through the environment or a `.env` in the server's working directory.
6. **Reinstall dependencies.** `neograph-engine` must be exactly 0.14.1, and Python before 3.14 now needs `zstandard>=0.23` to read compressed Codex sessions (`.jsonl.zst`). `pip install -e ".[dev]"` or `scripts/install_codex.py` takes care of both; without the package, plain sessions work and compressed ones report `zstd_unavailable`.
7. **AGY hook policy.** The default is now `confirmed_only`. An install that relied on the earlier `enforced` default (an unverifiable `PreToolUse` denied, unverified items continuing a `Stop`, an unavailable runtime denying `PreToolUse`) must set `SELF_DIRECT_AGY_ENFORCEMENT=enforced` explicitly, and even then hook errors fail open.
8. **Sparse index masking.** Sparse (FTS) rows written before 0.6.0 hold raw text. Delete `sparse.sqlite` while the servers are stopped to have it rebuilt, masked, from the canonical store; see the [masking contract](SECURITY.md#masking-contract).
9. **Source integrity tables.** The first open of an index adds the `source_blocks` table (a schema migration that touches nothing else), and each session's cursor migrates to a block root on its first full sync after the upgrade, without a reindex. A process on an older version that meets a migrated cursor reindexes that session once, so restart every process that shares the index (step 1). See [Source integrity](#source-integrity-full-and-fast-verification).
10. **Re-verify.** Run `python scripts/install_codex.py --verify` for Codex and `audit_status` for any host. Confirm `neograph_version`, `embedder`, `dense_backend` and that `degraded_flags` is empty.

---

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
SELF_DIRECT_OPENROUTER_API_KEY_FILE=<authorized external dotenv containing OPENROUTER_API_KEY>
```

Keep approval, checker, hook configuration, Lean toolchain/cache, and evidence storage outside the agent's write authority. Checksums and file modes do not protect against an unrestricted process running as the same OS user. Use the host's sandbox/read-only mounts or a separate service account. Approval and workflow state must be separate directories.

### Verification and execution

Run `workflow_run_checks` to observe real subprocess exit codes and independently checked Lean proofs. Tests run against a fresh, readonly materialized snapshot of approved verification files and current tracked implementation—not the live project with untracked `conftest.py`, `sitecustomize.py`, or generated caches. Source-glob selection cannot silently authorize mutable harness/configuration files. The sandbox has a private writable temporary directory, no network or host credentials/state, and an allowlisted environment. Tests must write temporary artifacts there rather than modify the readonly snapshot. The runner does not accept model-supplied pass flags, replacement commands, or expected results. Source versions are checked before and after verification; interruption leaves nonpassing evidence.

Candidate Lean compilation is sandboxed. `leanchecker` replays the candidate against hash-pinned, readonly core imports; a source-pinned native auditor separately replays candidate declarations and checks the exact approved type and transitive axioms without loading candidate elaborator extensions. Unchanged trusted core libraries are not reproved for every request. Candidate stdout is not an audit report. Approved statements use trusted core syntax and fully qualified names, not candidate-defined notation or typeclass registration. Native `.olean` deserialization still assumes structurally valid artifacts: this is not the stronger comparator-plus-independent-external-checker guarantee described in [Lean's proof-validation guidance](https://lean-lang.org/doc/reference/latest/ValidatingProofs/).

`workflow_classify` calls [OpenRouter's documented System One route](https://openrouter.ai/docs/guides/community/typesafe-sdk), `https://openrouter.ai/api/v1/systemone`, with pinned `typesafe/jev-1.13` and an OpenRouter bearer key. It accepts only that exact response identity or its canonical release identity, `typesafe/jev-1.13-20260917`, and retains the returned identity. Each classification makes at most one request: no retry, redirect, replay, provider fallback, or model alias. The fixed mapping is `0=InsufficientEvidence`, `1=NeedsRevision`, `2=ReadyForVerification`, `3=ReadyForCompletion`; no Score rounding is used. The input hash binds the actual request, rubric, mapping, and model. Approved requirement origins, actual bounded code diffs, state, and verification summaries are the only context sent. Oversized or non-text changes are not silently truncated.

Remote classification is disabled unless the owner explicitly sets `external_context: "synthetic"`. This release's experiments are synthetic; do not label private production data synthetic to bypass consent. Classification reuses the server's shared OpenRouter settings: `SELF_DIRECT_OPENROUTER_API_KEY_FILE` selects an authoritative dotenv file containing `OPENROUTER_API_KEY`, read without interpolation. A missing, unreadable, or keyless configured file fails closed rather than falling back to an environment key. With no configured file, the shared `OPENROUTER_API_KEY` setting/environment is used. No separate TypeSafe key is needed or accepted, and other providers' keys are never substituted. Keep the credential file outside the workspace and runtime mounts; the child sandbox excludes it and strips host keys from its environment. A missing key, timeout, invalid distribution, unknown class/model, or low confidence is a nonauthorizing result. The example confidence threshold is an operator setting, **not** an empirically validated accuracy guarantee.

Use a fresh classification ID for each `workflow_transition`. A completion choice reports readiness while remaining in `verify`; it cannot skip the owned completion action. After reaching `verify` with current mandatory evidence, `workflow_authorize` accepts an action such as:

```json
{"tool_name":"workflow_command","arguments":{"argv":["{python}","release.py"],"cwd":"."}}
```

Pass the returned grant and the **identical action** to `workflow_execute`. It rechecks state, versions, classification, cancellation, and Lean policy immediately before consuming the grant. Only successful owned execution sets `complete`. Started, succeeded, failed, and uncertain outcomes are distinct; a crash after consumption cannot replay the grant. A timeout after launch never claims that no action ran. The local completion sandbox permits writes to the actual approved project, but not private host state/credentials or network access. No shell interpolation is performed, and cancellation/timeout terminates the sandbox process group and descendants.

The AGY and Claude Code hooks reject direct mapped completion actions and direct the agent to the owned executor; on a normal stop they continue an incomplete workflow once (the same once-per-conversation or once-per-session budget as `auto_continue`) and provide requirements, evidence, and permitted recovery actions. In `advisory` mode they neither reject nor continue. Cancellation, error stops, and pauses for background work are not restarted. Reads, repairs, and checks remain available unless a separate explicit contract restricts them. **Unmapped tools, direct shell access outside registered hooks, and the advisory Codex adapter are not an enforcement boundary.**

Lean proves the policy/model under its stated assumptions; it does not prove arbitrary Python/TypeScript implementations equivalent to that model. Real tests supply implementation evidence only for exercised cases. The workflow is separate from session GraphRAG extraction and does not require an LLM to decide whether recorded tests passed.

### Reproducible synthetic policy evaluation

```bash
TSUKKOMI_TEST_LEAN=lean python -m pytest -q
python scripts/benchmark_workflow.py --lean lean
```

The benchmark compares 640 finite stage/class/evidence combinations with an independent reference, measures false allowances/blocks and per-stage latency, and exercises correction of each missing mandatory fact. It is an adversarial **policy** evaluation, not live JEV accuracy or a whole-host coverage claim. Live JEV validation requires an OpenRouter key; offline transport fixtures never count as live service evidence. A real low-confidence Choice remains nonauthorizing: the proposed label and observed probabilities are retained, but completion stays locked.

---

## Testing and verification

The suite covers parsers, host receipt linkage, restricted reads, temporal obligations, append capture and prefix judgement, duplicate session sources, single-flight and concurrency, workflow version/grant invalidation, proof rejection, path isolation and MCP transport, the adaptive shadow schedule (`tests/test_shadow_schedule.py`) and the fault injection that measures how long a defect of the incremental path goes unnoticed (`tests/test_shadow_detection.py`). The suite compares every audit with the full evaluator (`SELF_DIRECT_AUDIT_SHADOW_EVERY=1`). Tests use synthetic sessions and offline embeddings. Lean integration tests, Windows-only path tests and bubblewrap tests report a skip when the toolchain or platform is unavailable, and symlink tests report a skip with a reason where Windows denies symlink creation (`WinError 1314`, Developer Mode off).

```bash
python -m pytest -q
python -m build
python scripts/benchmark.py
python scripts/bench_hooks.py --out bench-hooks.json --markdown bench-hooks.md   # hook cost, see below
python scripts/bench_daemon.py --out bench-daemon.json --markdown bench-daemon.md  # resident daemon, Linux (experimental)
python scripts/bench_warm.py --defer --spans claude:Stop:4096                      # the warm judgement of one engine, in-process
python scripts/measure_shadow_detection.py --scenario missed --every adaptive --floor  # fault injection: requests until a defect of the incremental path is noticed
python scripts/bench_commit.py --samples 300 --kill 50 --tmpdir /path/on/ext4  # commit latency and crash survival by journal mode
python scripts/bench_pgvector.py --sizes 1000,10000,100000 --sqlite-vector /abs/vector.so --out /abs/pgvector.json  # pgvector against sqlite-vector and NumPy, see "pgvector numbers"
pip install -e ".[dev,pgvector]" && TSUKKOMI_REQUIRE_PGVECTOR=1 python -m pytest -q tests/test_pgvector_integration.py tests/test_pgvector_engine.py tests/test_dense_backend_parity.py  # needs Docker and the pgvector/pgvector:pg17 image
TSUKKOMI_TEST_DEFER_DERIVED=1 python -m pytest -q                                 # the whole suite with every engine holding its derived writes
```

The resident daemon's tests live in `tests/daemon/` (Linux only; other platforms skip them): framing and directory checks, peer credentials, bounds, version switching, idle exit and the memory drain, byte-identical hook output through the daemon and standalone (Claude Code and AGY: deny, allow with a satisfied prerequisite, `Stop`), concurrency, every fail-open path, the client import budget, MCP parity through the relay, and a fixture that fails any test that leaves a daemon process running (and the session, if one survives).

### CI

GitHub Actions runs two workflows. Superseded runs for the same ref are cancelled, and no API keys or paid model calls are used.

The `CI` workflow runs four jobs on every push and pull request:

- **`test`** (Ubuntu and Windows, Python 3.12, 10 minutes): the explicit MCP transport, Jev boundary and Windows session-path smoke tests and the compressed Codex session tests (without their two 50 MB allocation probes), then builds the source distribution and wheel.
- **`memory-audit-regressions`** (Ubuntu, Python 3.12, 15 minutes): small offline regression files for, among others, audit memory, prefix judgement, current-action prerequisites, append and source capture, single-flight, index and server concurrency, deadlines, request statistics and degraded evaluation, checklist conflicts, quiet hooks, AGY and Claude Code hooks, the Claude Code installer and provider, and hook gating. It also runs the hook import budget and settings parity tests ([Hook import budget](#hook-import-budget)) and the BLAS thread test. It runs with the fake embedder and the NumPy backend and deselects the one test that measures allocation of a large capture. These tests check ratios on small inputs; they do not certify live-service memory.
- **`pgvector`** (Ubuntu, Python 3.12, 20 minutes; a failure fails the workflow): the optional pgvector backend against a `pgvector/pgvector:pg17` service container, with `TSUKKOMI_REQUIRE_PGVECTOR=1` (a missing server is a failure, never a skip) and the official sqlite-vector library, running `tests/test_pgvector_unit.py`, `test_pgvector_config.py`, `test_pgvector_integration.py`, `test_pgvector_engine.py` and `test_dense_backend_parity.py`. The first two (a fake driver, no psycopg, no Docker) also run in the `test` and `memory-audit-regressions` jobs, so the smoke job never needs either. The container's password belongs to that throwaway service and protects nothing.
- **`bench`** (Ubuntu, Python 3.12, 10 minutes): `scripts/bench_hooks.py --check` on small synthetic sessions. It fails when a gated metric exceeds 1.5 times its target and uploads the JSON report and the markdown table as an artifact ([Hook cost benchmark](#hook-cost-benchmark)). The 300 MB session is never run in CI.

The `Memory` workflow (pull requests to `main`, `main`, nightly and on demand) is described in [Memory verification](#memory-verification).

Real Lean proof checks and the performance, large-session and concurrency benchmarks are local deep checks, not CI gates. A green CI run does not establish full workflow or native-proof coverage or performance guarantees. Live model validation needs a separate model and budget approval.

### Hook import budget

A Claude Code or AGY hook is a fresh Python process for every tool call, so every module it imports is paid for on every call. The hook path is kept deliberately small and a test enforces it.

`tests/test_hook_import_budget.py` runs **representative judgements**, not bare imports, in subprocesses: a `PreToolUse` that denies (a confirmed violation) and one that passes (a prerequisite satisfied from the transcript), `Stop` with its sync and audit, `SessionStart`, `PreInvocation` and the bookkeeping events, for Claude Code and AGY. Afterwards `sys.modules` must not contain a forbidden module, and the only third-party packages loaded must be the documented exceptions below. A new import that breaks this fails the test with the module's name.

Never loaded on the local-only hook path. The cost is the cumulative import time that the module added to a hook (`python -X importtime`, median of 13 fresh processes, AMD Ryzen 7 5800X under WSL2, Python 3.12; the tracing itself adds roughly 15 percent):

| Module | Why hooks do not need it | Cost when loaded |
|---|---|---|
| `numpy` | Dense vectors only, and local-only hooks never embed. `self_directing_mcp._runtime.np` is the single loading point and a lazy proxy: NumPy loads, with one BLAS thread and the environment restored, at the first attribute access. | 37 ms and about 15 MB (with the default BLAS threads about 1 CPU second more) |
| `pydantic_settings` (and `dotenv`) | The hooks read their settings with `settings_core.load_hook_settings()`: the same sources, precedence and rejections as `config.hook_settings()` (environment, `CODEX_SESSIONS_DIR`, the user TOML `[paths]`, defaults; never a working-directory `.env`), standard library only. The defaults, the TOML source and the accessors live in `settings_core` and are shared with `config.Settings`, so they cannot drift; `tests/test_hook_settings_parity.py` compares the two readers on about 850 environment and TOML combinations, including the ones both must reject. | 28 ms |
| `httpx` (with `idna` and `zstandard`) | The OpenRouter client. `embed.embedder`, `workflow_jev` and the graph runner import it, `dotenv` and `urllib.request` only where a request is made. | 28 ms (`urllib.request`: 4 ms) |
| `mcp` (with `anyio`, `starlette`) | The server transport; hooks are plain processes. | not loaded before this change either; guarded against regressions |
| `self_directing_mcp.config`, `.server` | Server-side settings and tool registration. | included above |

Claude Code's bookkeeping events (`UserPromptSubmit`, `PostToolUse`, `PostToolUseFailure`) load none of pydantic, the engine or the transcript parser: they validate the session id with `claude_ids` and deliver queued warnings.

Documented exceptions, loaded by every hook that judges contracts:

| Module | Reason | Measured cost |
|---|---|---|
| `pydantic` (with `pydantic_core`, `annotated_types`, `typing_extensions`) | `schemas.py` defines the contract, chunk and audit models, and the contracts file and every parsed chunk are validated through them. Replacing them is a rewrite, not an import fix. | about 75 ms cumulative: pydantic 16 ms, `importlib.metadata` 15 ms (pydantic's plugin loader) and the models themselves |
| `regex` | Contract patterns come from users and agents. They run on this engine because a search can be given a time limit (`audit/patterns.py`: 0.25 s per search, 2 s per contract, `regex_timeout` unknown beyond that), which the standard `re` module cannot interrupt. | about 10 ms |
| `neograph_engine` (with `certifi`) | Project rule ([AGENTS.md](AGENTS.md)): audit judgements run through the NeoGraph native graph, with no direct executor fallback. Kept as a documented exception; the roundtable proposed forbidding it and left the rule to the owner. | about 40 ms: the native extension 21 ms and `asyncio` 15 ms |

To re-measure, run a hook with `-X importtime` (`python -X importtime scripts/claude_hook.py --event=PreToolUse --local-only < payload.json 2> importtime.txt`; the columns are self and cumulative microseconds). Change the budget only by editing `FORBIDDEN` or `ALLOWED_THIRD_PARTY` in the test together with this section, with the module, the reason and the measured cost.

### Hook cost benchmark

`scripts/bench_hooks.py` measures what the hooks cost. It uses synthetic sessions, contracts and payloads under an isolated `HOME`, index directory and config file, and never reads a user's sessions, indexes or credentials. Cold, warm and server numbers are reported separately and never mixed.

```bash
python scripts/bench_hooks.py --out bench-hooks.json --markdown bench-hooks.md      # small sessions, about 8 minutes
python scripts/bench_hooks.py --check                                                # exit 1 above a gated CI threshold
python scripts/bench_hooks.py --large --large-mb 300 --out bench-large.json         # opt-in 300 MB session, never in CI
python scripts/bench_hooks.py --only cold --versus /path/to/older/checkout          # interleaved A/B of two checkouts
python scripts/bench_hooks.py --compare before.json after.json                      # before/after tables from two reports
python scripts/bench_hooks.py --python /path/to/venv/bin/python --hook-mode default # other interpreter / engine mode
```

What is measured:

- **Cold command hooks** (`cold`): a fresh Python process per event through the real `scripts/claude_hook.py` (`PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `SessionStart`, `Stop`) and `scripts/agy_hook.py` (`PreToolUse`, `PostToolUse`, `PreInvocation`, `Stop`) with synthetic payloads, plus a burst of N concurrent sessions (`burst`). The page cache is warm; "cold" means a new interpreter that imports the package and judges once. The transcript grows by whole records between runs, so each hook syncs a small increment. The first run of each event, which builds the index, is recorded separately (`first_run`). A `.venv` re-exec (one more interpreter start) is not included. `--hook-mode local-only` (default) runs the engine without an embedder or dense index, as `--local-only` does for Claude Code; `default` uses the offline fake embedder and the NumPy backend, so each hook also initializes the dense side.
- **Warm resident-server calls** (`warm`): one long-lived process and one asyncio loop through FastMCP's dispatcher: `codex_session_hook` (`PreToolUse`, `UserPromptSubmit`, `Stop`), `check_action` and `audit_session`, each with nothing appended and after appending bytes. The index build of the first call is reported on its own.
- **Idle servers** (`idle_servers`): N real `python -m self_directing_mcp` stdio servers with an isolated index, after the MCP handshake (`after_init`) and after one tool call that initializes the engine (`after_use`). Linux reads `/proc` (`VmRSS`, `VmHWM`, `Pss`); Windows reads the working set, peak working set and private bytes through `GetProcessMemoryInfo`. A sum of RSS counts shared libraries once per process; `pss_mb_total` divides them fairly.

Every row gives p50, p95 and maximum wall time, CPU time (user plus system, from the operating system: `os.wait4` for hook processes, `getrusage` or `GetProcessTimes` in a server) and peak RSS per process, together with the machine, OS, Python and package versions and the load average. A hook that fails open or exits non-zero invalidates the run (`hooks_failed_open`). On a shared machine use `--versus`: the two checkouts see the same transcript and payloads in alternating order, so load from other processes hits both alike.

Fixed conditions (flags, defaults in parentheses): contract counts (`--contracts 10,100`, each a rotating mix of `must_not`, `must`, `must` with `before_regex` and `must` with `before_regex` and `requires_success`), small session size (`--session-mb 1`), large session (`--large`, `--large-mb 300`, opt-in), bytes appended per step (`--append-bytes 4096`), concurrent sessions in the burst (`--sessions 4`), idle servers (`--servers 12`, `--server-profile local-only|fake-numpy`), repetitions (`--runs 20` after one discarded warm-up, `--warm-repeat 15`, `--large-repeat 5`).

Targets and CI thresholds live in one table, `TARGETS` in the script (also written into every report). A CI threshold is `--ci-factor` (default 1.5) times the target. Rows that depend on other work are reported against their target and marked "reported, not gated yet":

| Metric | Target | CI threshold | Gated here |
|---|---:|---:|---|
| Cold hook wall, p95 | 400 ms | 600 ms | yes, for Claude Code `PostToolUse` and `UserPromptSubmit`, which fire after every tool call and every prompt |
| Cold hook CPU (user+sys), p95 | 0.3 s | 0.45 s | yes, for the same events |
| Cold hook wall, p95 and CPU, every event | 400 ms, 0.3 s | 600 ms, 0.45 s | no: `PreToolUse` and `Stop` evaluate every contract over the whole session and AGY re-parses its transcript on every call; they wait for the fast integrity path and the incremental audit |
| Cold hook wall, median (goal) | 150 ms | 225 ms | no |
| `check_action`, large session, nothing appended, CPU | 100 ms | 150 ms | no: needs the fast integrity path |
| `audit_session` after at most 100 appended events, CPU | 20 ms | 30 ms | no: needs the incremental audit |
| Idle server RSS | 90 MB (70 MB desired) | 135 MB | yes |

Phase 2 (a resident daemon) becomes due if the summed RSS across hosts exceeds 500 MB, a turn's accumulated hook latency exceeds 2 s, or the phase 1 gates are missed; the report states when the idle-server total passes 500 MB. The thresholds are absolute: on a slower machine (a hosted CI runner, Windows) widen `--ci-factor` rather than the targets, and report Linux and Windows separately.

Measured on 2026-10-15, Linux (WSL2) on an AMD Ryzen 7 5800X with 16 logical CPUs, Python 3.12.14, 1 MB synthetic sessions, 10 contracts, 4 KB appended per step, 20 interleaved runs per event (`--versus`), the machine shared with other work (load average about 6). Wall and CPU in milliseconds, before (commit `00cb90e`) → after this change:

| host | event | wall p50 | wall p95 | CPU p50 | CPU p95 | peak RSS |
|---|---|---|---|---|---|---|
| claude | PreToolUse | 407 → 281 | 445 → 309 | 385 → 263 | 424 → 294 | 83 → 67 MB |
| claude | PostToolUse (gated) | 189 → 52 | 254 → 60 | 188 → 52 | 252 → 59 | 35 → 25 MB |
| claude | UserPromptSubmit (gated) | 185 → 50 | 205 → 62 | 185 → 50 | 205 → 62 | 35 → 25 MB |
| claude | SessionStart | 316 → 208 | 344 → 247 | 315 → 208 | 344 → 246 | 73 → 56 MB |
| claude | Stop | 598 → 410 | 766 → 474 | 551 → 365 | 718 → 406 | 90 → 74 MB |
| agy | PreToolUse | 742 → 646 | 854 → 754 | 654 → 551 | 765 → 623 | 86 → 70 MB |
| agy | PostToolUse | 780 → 664 | 898 → 748 | 646 → 538 | 743 → 606 | 82 → 65 MB |
| agy | PreInvocation | 420 → 321 | 531 → 349 | 372 → 268 | 476 → 295 | 78 → 62 MB |
| agy | Stop | 647 → 539 | 740 → 589 | 570 → 452 | 655 → 496 | 94 → 77 MB |

With 100 contracts the same events move by about the same amount (`PostToolUse` 190 → 53, `UserPromptSubmit` 193 → 54, `SessionStart` 328 → 212, `PreToolUse` 443 → 345, `Stop` 669 → 558 ms at p50); AGY `Stop` is still 931 ms and AGY `PreToolUse` 854 ms, because both evaluate the contracts over the whole transcript (`audit_session` after appending: about 550 ms CPU for 100 contracts against the 20 ms target). Warm calls and the first-call index build did not change (the work is the same): `check_action` with nothing appended 40 ms, `audit_session` 80 ms, `codex_session_hook` `Stop` about 150 ms at 10 contracts. Twelve idle local-only servers: 1073 → 949 MB total RSS (89.4 → 79.1 MB each, 637 → 570 MB PSS), because a local-only server no longer loads NumPy. The roundtable's own measurement of an idle OMP server was 86 MB and of a Codex server 79 MB; a server with an embedder still loads NumPy at startup.

Large session (`--large --skip-small`, opt-in), same machine: a 300 MB synthetic Codex rollout of about 200,000 events of 1.5 KB (the roundtable measured 314 MB and 225,000 events), one warm server process with a local-only engine. With one contract the numbers match the roundtable's measurement:

| call (CPU, p50 of 5) | 1 `must_not` contract | 10-contract mix, before | 10-contract mix, after |
|---|---:|---:|---:|
| first call (builds the index, once) | 26.20 s | 29.39 s | 29.37 s |
| `check_action`, nothing appended | 1.02 s | 4.71 s | 5.01 s |
| `check_action`, 4 KB appended | 1.32 s | 4.82 s | 5.11 s |
| `audit_session`, nothing appended | 2.69 s | 7.40 s | 7.27 s |
| `audit_session`, 4 KB appended | 3.05 s | 7.64 s | 7.65 s |
| `codex_session_hook` `PreToolUse` | 1.06 s | 4.61 s | 4.63 s |
| `codex_session_hook` `Stop` | 10.73 s | 15.18 s | 17.23 s |

Against the targets (`check_action` with nothing appended 100 ms, `audit_session` for a few appended events 20 ms) these are the work of the fast integrity path and the incremental audit, not of this change: before and after agree, which is why they are reported and not gated. With the 10-contract mix `check_action` costs 5 s because the `before_regex` obligations are evaluated over the whole history on every call. `codex_session_hook` `Stop` costs 10.7 s on this session: its evidence stamp reads and parses every line of the transcript before the audit runs.

Windows, reported separately (Windows 11, Python 3.13, the same machine, 12 interleaved runs per event, 10 contracts; before → after, median wall in ms): Claude Code `PostToolUse` 361 → 141, `UserPromptSubmit` 358 → 144, `SessionStart` 543 → 359, `PreToolUse` 625 → 473, `Stop` 716 → 552; AGY events 11 to 28 percent faster. Windows is slower in absolute terms (a hook's process start and imports cost about 1.3 to 2.7 times as much), so the events that are not gated can exceed their 1.5 times threshold there (`SessionStart` reaches 0.56 s CPU at p95); the gated `PostToolUse` and `UserPromptSubmit` stay at about 0.2 s p95 and pass with the default factor. Twelve idle local-only servers: 966 → 877 MB working set (80.6 → 73.3 MB each). A venv's `python.exe` is a launcher that starts the real interpreter, so the benchmark runs the process tree in a job object and reads the interpreter's own peak working set; Windows CPU times have the 15.6 ms resolution of the operating system clock. The roundtable measured the 314 MB session on Windows at commit `00cb90e` (local-only engine): server import 1.12 s, first `check_action` 44.8 s, `check_action` with nothing appended 1.50 to 1.58 s CPU, `audit_session` 4.22 s.

### Warm judgement benchmark

`scripts/bench_warm.py` measures what is left of a hook judgement once the process and its imports are gone (the daemon case): one `SelfDirectEngine` kept between requests, 1 MB synthetic Claude Code and AGY sessions, 10 contracts, `claude_hooks.run_hook` / `agy_hooks.run_hook` called repeatedly with nothing appended and with 4 KB appended before each request, under an isolated `HOME`, index and config file. Per cell it prints p50 / p95 / minimum wall time and the **durable SQLite commits per request** (counted through SQLite's statement trace: a commit that wrote something is an `fsync`, 10 to 25 ms on the benchmark machine and most of what a warm judgement costs once the CPU work is gone). `--defer` builds the engine as the daemon does; `--tmpdir /dev/shm/...` puts the index on a RAM disk, which removes the `fsync` and leaves the CPU cost; `--profile host:event:bytes` prints a `cProfile`; `--spans host:event:bytes` (or `all`) times the stages the judgement is made of, per request; `--exact-size` cuts the session to `--session-mb` at a record boundary (tens or hundreds of events); `--shadow-every adaptive|N`, `--min-events N` and `--warmup N` (the untimed requests before a cell, default 3) set the incremental-audit settings and the warm-up; each cell also reports p99 and how many shadow comparisons its requests carried. It runs against an older checkout too (`--defer` is then ignored), which is how the before/after below was made.

```bash
python scripts/bench_warm.py --defer --runs 50                                  # the daemon's engine
python scripts/bench_warm.py --defer --tmpdir /dev/shm/warm --runs 50           # the same without the disk's fsync
python scripts/bench_warm.py --defer --spans claude:Stop:4096 --only claude:Stop # where one cell's time goes
```

Median over three interleaved rounds of 50 requests (before = the commit this work started from; machine and conditions as in [Measurements](#measurements) above, load average 2.7 to 3.8; p50 / p95 ms and durable commits per request):

| Cell | before | after, synchronous engine | after, daemon engine (`--defer`) |
|---|---|---|---|
| Claude Code `PreToolUse`, nothing appended | 23.9 / 34.0, 0 | 21.2 / 33.0, 0 | 21.3 / 31.3, 0 |
| Claude Code `PreToolUse`, 4 KB appended | 35.8 / 56.0, 2.1 | 34.2 / 53.9, 2.1 | **22.2 / 42.6**, 1 |
| Claude Code `Stop`, nothing appended | 59.5 / 69.5, 2 | 34.0 / 37.7, 2 | **9.4 / 12.0**, 0 |
| Claude Code `Stop`, 4 KB appended | 101.6 / 119.2, 4 | 53.9 / 60.2, 3 | **28.1 / 32.7**, 1 |
| AGY `PreToolUse`, nothing appended | 54.7 / 60.1, 3 | 48.4 / 53.0, 3 | **37.2 / 45.4**, 2 |
| AGY `PreToolUse`, 4 KB appended | 87.7 / 97.3, 5 | 86.9 / 96.5, 5 | **60.7 / 68.1**, 3 |
| AGY `Stop`, nothing appended | 83.8 / 100.7, 2 | 16.4 / 21.1, 0 | **14.9 / 20.6**, 0 |
| AGY `Stop`, 4 KB appended | 146.9 / 252.6, 5 | 51.4 / 132.2, 2 | **35.5 / 117.8**, 1 |

With the shadow comparison off (`--shadow-every 0`, under the fixed schedule these tables were measured with), which isolates the steady state, the same cells are 4 / 5, 34 / 49, 57 / 69, 98 / 110, 52 / 57, 81 / 90, 84 / 104 and 144 / 172 ms before and 3.7 / 4.3, 22 / 27, 9.5 / 12, 28 / 31, 35 / 41, 57 / 64, 15 / 19 and 35 / 42 ms after (daemon engine). Of what is left, the CPU work is 4 to 21 ms (measure it with `--tmpdir` on a RAM disk; it was 4 to 75 ms before) and the rest is the commits in the table: one for the events and cursor, and for AGY the host receipt and the binding of results to receipts. The p95 of the cells that append events is a shadowed request (see [Measurements](#measurements)). Under that fixed schedule the first two cells' p50 is the shadow comparison too: that synthetic session's offset is one the deterministic sample selects, and with nothing appended every request is at it.

Where the time went in the slowest cells (per-request inclusive spans of `--spans`, p50 ms; spans nest, so the commit is inside the sync):

| Cell | total before → after | sync (capture, parse, commit) | of which events commit | derived writes (states, stamps, marker, FTS) | checklist read |
|---|---|---|---|---|---|
| Claude Code `Stop`, 4 KB | 100 → 29 | 65 → 26 | 16 → 17 | 40 → 0.2 | 20 → 0.2 |
| Claude Code `Stop`, nothing appended | 57 → 9.3 | 38 → 6.9 | 0 | 23 → 0.1 | 18 → 0.2 |
| AGY `Stop`, 4 KB | 132 → 35 | 84 → 30 | 17 → 17 | 47 → 0.9 | 33 → 0.2 |
| AGY `PreToolUse`, 4 KB | 77 → 53 | 46 → 34 | 17 → 16 | 22 → 0.6 | 0 |

AGY costs grow with the number of receipts in the conversation (every request read them all a dozen times). With 1,200 realistic receipts (4,800 events, RAM disk) an AGY `Stop` took 37.5 ms before the receipts reader and 21.4 ms after it, a `PreToolUse` 33 and 25 ms.

**`SELF_DIRECT_AUDIT_INCREMENTAL_MIN_EVENTS` (default 500) and resident engines.** The benchmark sessions hold 1,729 (Claude Code) and 3,265 (AGY) events, so every request of the tables above keeps state and is judged incrementally; `bench_warm.py` prints this for each cell. The default exists because on a one-shot hook the monitor state costs a commit and a rebuild plus its shadow comparison. A resident engine that holds the state pays no commit, so the break-even moves down, and measuring it (`--exact-size`, 512 bytes appended per request, state kept against none, 20 requests per cell, p50 ms, Claude Code, RAM disk / real disk) gives:

| stored events | `PreToolUse` kept / none, resident | `Stop` kept / none, resident | `PreToolUse` kept / none, one-shot | `Stop` kept / none, one-shot |
|---|---|---|---|---|
| 31 | 6.2 / 4.5 RAM, 18.9 / 16.7 disk | 6.8 / 7.6, 20.5 / 20.0 | 4.4 / 5.0, 31.3 / 20.0 | 6.4 / 7.9, 44.9 / 35.2 |
| 124 | 5.0 / 5.5, 18.3 / 18.9 | 6.8 / 9.9, 20.5 / 24.0 | 4.6 / 5.6, 32.1 / 20.5 | 6.7 / 10.3, 47.6 / 38.8 |
| 496 | 5.3 / 8.8, 19.5 / 23.2 | 7.2 / 17.2, 22.1 / 32.0 | 4.8 / 9.1, 29.3 / 23.2 | 7.4 / 17.2, 41.9 / 42.7 |
| 1,857 | 5.3 / 23.9, 20.4 / 37.1 | 8.9 / 48.4, 24.2 / 60.0 | 5.5 / 22.3, 31.7 / 36.5 | 9.3 / 49.0, 48.0 / 71.8 |

For a resident engine keeping state costs the same as keeping none up to about 250 events (for `Stop` it is cheaper from the start) and is cheaper beyond; on a one-shot engine the extra commit makes it dearer until about 500 events (`Stop`) and 1,500 (`PreToolUse`). The default of 500 is therefore right for `Stop` on both, a little conservative for a resident engine between 250 and 500 events (a few milliseconds at most) and a little low for a one-shot `PreToolUse`; it was left unchanged.

### Shadow comparison cost

A shadow comparison runs the full evaluator beside the incremental one (see [Incremental audits](#incremental-audits)). It is the largest cost the warm path has left, and the requests that carry one are the p95 and p99 of a cell. The tables repeat the cells above with `scripts/bench_warm.py --runs 30` (10 contracts, 1 MB sessions), **before** (`main` at the merge of PR #21: the fixed 1 in 16, journal mode at its default `delete`) and **after** (the adaptive schedule with the constants above), as the median of three interleaved rounds. Each entry is `p50 / p95 / p99 ms [shadow comparisons among the cell's 30 requests]`. *Default schedule* is the benchmark's usual 3 untimed requests per cell; *past the warm-up* is `--warmup 12`. Every run was started alone, one after the other, with no test suite or mutation run on the machine; the load average was 2.7 to 4.7 during them (other work was running; the tables further up were measured at 2.7 to 3.8), and before and after ran in the same rounds, interleaved.

**Daemon engine (`--defer`), load average 3.2 to 4.2:**

| Cell | before: fixed 1/16 | after, default schedule | after, past the warm-up |
|---|---|---|---|
| Claude Code `PreToolUse`, nothing appended | 26.8 / 42.7 / 45 [30] | 4.6 / 6.1 / 31 [1] | 5.7 / 7.3 / 8 [0] |
| Claude Code `PreToolUse`, 4 KB appended | 14.9 / 38.9 / 39 [3] | 16.5 / 39.0 / 46 [7] | 16.3 / 39.2 / 42 [7] |
| Claude Code `Stop`, nothing appended | 12.8 / 15.3 / 19 [0] | 11.5 / 17.1 / 54 [1] | 11.1 / 14.1 / 15 [0] |
| Claude Code `Stop`, 4 KB appended | 22.6 / 28.8 / 62 [1] | 23.2 / 79.2 / 82 [5] | 21.2 / 74.1 / 82 [5] |
| AGY `PreToolUse`, nothing appended | 24.4 / 28.4 / 29 [0] | 22.6 / 31.8 / 80 [1] | 20.4 / 25.2 / 26 [0] |
| AGY `PreToolUse`, 4 KB appended | 36.0 / 43.8 / 71 [1] | 37.8 / 83.6 / 88 [4] | 34.6 / 71.4 / 87 [4] |
| AGY `Stop`, nothing appended | 17.8 / 26.3 / 28 [0] | 20.6 / 27.7 / 121 [1] | 17.6 / 26.9 / 28 [0] |
| AGY `Stop`, 4 KB appended | 29.7 / 36.9 / 37 [0] | 30.5 / 120.2 / 136 [3] | 29.8 / 123.3 / 133 [3] |

**Synchronous engine (a one-shot hook process's), load average 2.7 to 4.7:**

| Cell | before: fixed 1/16 | after, default schedule | after, past the warm-up |
|---|---|---|---|
| Claude Code `PreToolUse`, nothing appended | 26.5 / 38.9 / 40 [30] | 5.1 / 6.9 / 52 [1] | 4.9 / 6.1 / 6 [0] |
| Claude Code `PreToolUse`, 4 KB appended | 20.1 / 45.4 / 55 [3] | 20.1 / 43.0 / 46 [7] | 22.4 / 52.9 / 60 [7] |
| Claude Code `Stop`, nothing appended | 22.7 / 33.3 / 43 [0] | 20.3 / 25.1 / 79 [1] | 19.9 / 22.7 / 25 [0] |
| Claude Code `Stop`, 4 KB appended | 32.0 / 60.2 / 93 [1] | 33.5 / 92.8 / 104 [5] | 29.4 / 90.2 / 116 [5] |
| AGY `PreToolUse`, nothing appended | 29.9 / 33.3 / 33 [0] | 26.8 / 35.3 / 80 [1] | 27.3 / 32.0 / 37 [0] |
| AGY `PreToolUse`, 4 KB appended | 44.9 / 64.2 / 90 [1] | 41.6 / 86.2 / 90 [4] | 41.8 / 86.1 / 94 [4] |
| AGY `Stop`, nothing appended | 19.4 / 24.1 / 31 [0] | 18.4 / 33.5 / 102 [1] | 18.8 / 23.2 / 24 [0] |
| AGY `Stop`, 4 KB appended | 42.0 / 55.6 / 57 [0] | 36.9 / 123.5 / 132 [3] | 37.5 / 108.7 / 140 [3] |

What the tables show, and what they do not:

- **Nothing appended: the cell the fixed sample hit on every request is fixed, and the others do not change.** The fixed schedule compared *every* request of `Claude Code PreToolUse (nothing appended)` (the cell's offset is one the sample selects, and with nothing appended every request is at it). The adaptive draw depends on the schedule state as well as the cursor, so a repeated request is not repeatedly selected: the cell goes from 26.8 to 4.6 ms at p50 and from 42.7 to 6.1 ms at p95 (daemon engine). In the default column each nothing-appended cell still holds 1 comparison among its 30 requests, the last of the 4 warm-up comparisons that the 3 untimed requests did not use up (it is the cell's p99); past the warm-up there is none.
- **Cells that append events carry more comparisons than 1 in 16 did, in a session this young, and their p95 and p99 are a full evaluation.** A new session is compared at 1/4 until 8 comparisons have matched (the 4 of the warm-up count), at 1/8 for the next 8, and so on, so 3 to 7 of a cell's 30 requests are compared where 1 in 16 compared 0 to 3. The p95 and p99 of such a cell are a request that ran a full evaluation, as under 1 in 16 when it happened to compare one, and **higher where 1 in 16 happened to compare none** (daemon engine, p95 in ms: AGY `Stop` with 4 KB appended 36.9 → 120.2, Claude Code `Stop` with 4 KB 28.8 → 79.2, AGY `PreToolUse` with 4 KB 43.8 → 83.6). The p50 of the other cells differs by -5.1 to +2.8 ms between before and after, in both directions, which is the noise of these runs (a median of 30 requests, with other work on the machine). These sessions have no violation, flip or high-risk contract, so no forced comparison of that kind appears here; a request that has one adds a scoped comparison of just those contracts.
- **Against the constants this replaced** (the first measurement of this schedule, 1/4 down to 1/64, halving after 32, a warm-up of 8; load 5.2 to 8.0, so only the counts compare): 5 comparisons among the 30 requests of every nothing-appended cell in the default column and 4 to 9 in the appended ones; now 1 and 3 to 7.
- **A long window shows what the schedule is for.** The share of requests compared falls below 5%, which is what a p95 sees, after about 900 requests, and to 3.1% at the floor. Claude Code `Stop` with 4 KB appended, 1,500 requests in one session, daemon engine, median of three interleaved rounds (load average 3.2 to 5.3). The p99 is a compared request on both sides (4 to 6% of the requests are compared, so the 15 slowest of 1,500 are all comparisons):

| 1,500 requests of one session | p50 / p95 / p99 ms | compared |
|---|---|---|
| before: fixed 1/16 | 43.7 / 200.1 / 362 | 96 (6.4%) |
| after: adaptive | 41.2 / 74.7 / 311 | 62 (4.1%) |

- **A resident engine writes no commit for the schedule** (`--defer`: the schedule is held with the monitor states). A synchronous engine writes it with the states when a request writes states, and in a commit of its own otherwise: the one compared request of a nothing-appended cell, which is why those cells show 0.03 more durable commits per request (1 in 30) in the default column and none past the warm-up.

**What a violation that stands costs.** A violation (and a flip, and a changed high-risk summary) is called for by the finished answer, so the comparison it owes is a second, scoped pass over the same request, and it is owed again by every request that appends an event while the violation stands: the stored events changed, so the remembered comparison does not apply (an identical repeat is not compared twice, and a `PreToolUse` check does not evaluate prohibitions over history). The same cells with one more contract that the benchmark sessions violate from their first event on (`scripts/bench_warm.py`, `--defer`, 60 requests per cell after a warm-up of 12, median of three interleaved rounds, each started alone, load average 2.2 to 3.1; each entry is `p50 / p95 / p99 ms [shadow comparisons]`):

| `Stop` with 4 KB appended, daemon engine, 60 requests | no violation (p50 / p95 / p99 ms) | a contract violated from the first event, as built | the same with the second pass left out |
|---|---|---|---|
| Claude Code session, 1,729 events | 20.0 / 60.3 / 74 [11] | 41.8 / 66.4 / 77 [60] | 20.5 / 64.6 / 84 [11] |
| AGY session, 3,265 events | 24.3 / 99.0 / 109 [9] | 76.5 / 104.7 / 131 [60] | 26.5 / 105.0 / 119 [9] |

The pass adds about 21 ms (Claude Code) and 50 ms (AGY) to the p50 of such a request, growing with the size of the session, and nothing to a repeat of the same request (9.6 against 9.0 ms without the violation, nothing appended). It stays: a state that missed events is what the pass is for, and a violation is not monotone (a refused call is subtracted later, a `must` can be met afterwards), so skipping it while a violation stands would leave a wrong block unnoticed until a sampled comparison. The third column is that skipping, measured, not offered.

The resident daemon through `scripts/bench_daemon.py --runs 30 --only hooks` (cold hook processes through a warm daemon; one run per side, load average at the start 2.99 before and 3.83 after; the cells of a host share one session and one daemon engine, so the first `Stop` cells build the audit variant's states in the middle of the run). Wall time of the whole new process and the daemon-side time, p50 / p95 / p99 ms:

| Event (new process) | through the daemon, before: wall p50 / p95 / p99 ms | after | daemon-side before: p50 / p95 / p99 ms | after |
|---|---|---|---|---|
| Claude Code `PreToolUse (+0 B)` | 55.8 / 66.2 / 67.0 | 29.9 / 55.2 / 58.7 | 28.25 / 36.11 / 39.99 | 6.57 / 32.6 / 33.27 |
| Claude Code `PreToolUse (+4096 B)` | 39.4 / 57.4 / 70.8 | 43.8 / 76.7 / 88.4 | 15.39 / 36.21 / 49.19 | 17.77 / 52.56 / 52.64 |
| Claude Code `UserPromptSubmit (+0 B)` | 24.0 / 28.1 / 29.9 | 23.9 / 29.9 / 36.6 | 1.12 / 1.48 / 1.5 | 1.08 / 1.58 / 2.14 |
| Claude Code `UserPromptSubmit (+4096 B)` | 25.1 / 29.7 / 38.4 | 25.6 / 29.3 / 36.2 | 1.21 / 1.55 / 1.65 | 1.13 / 1.61 / 2.39 |
| Claude Code `Stop (+0 B)` | 36.5 / 47.6 / 51.0 | 40.0 / 106.6 / 109.8 | 11.66 / 15.34 / 21.86 | 13.88 / 81.89 / 86.03 |
| Claude Code `Stop (+4096 B)` | 43.8 / 80.4 / 102.6 | 47.9 / 118.6 / 119.4 | 20.48 / 51.77 / 80.0 | 22.62 / 92.9 / 95.92 |
| AGY `PreToolUse (+0 B)` | 45.5 / 53.0 / 56.4 | 48.7 / 84.3 / 93.4 | 22.91 / 27.39 / 28.16 | 24.38 / 60.19 / 67.21 |
| AGY `PreToolUse (+4096 B)` | 56.6 / 86.0 / 123.8 | 60.9 / 106.7 / 108.6 | 33.97 / 62.78 / 98.54 | 35.54 / 83.0 / 84.32 |
| AGY `Stop (+0 B)` | 43.8 / 58.3 / 60.7 | 43.0 / 141.8 / 148.8 | 19.38 / 28.59 / 28.78 | 18.32 / 118.44 / 119.8 |
| AGY `Stop (+4096 B)` | 57.6 / 77.1 / 80.9 | 55.4 / 135.6 / 159.6 | 32.29 / 43.08 / 47.42 | 30.55 / 113.34 / 138.0 |

The same pattern as above, one run per side, so the p95 and p99 of 30 requests move by tens of milliseconds between runs; the number of comparisons, not the milliseconds, is what the schedule decides. The cell the fixed sample compared in every request (Claude Code `PreToolUse`, nothing appended) falls from 55.8 to 29.9 ms wall at p50 (p95 66.2 → 55.2) and from 28.2 to 6.6 ms daemon-side at p50; the cells that append events, and `Stop` and AGY generally, now carry the warm-up and the sample of a young session in their p95 and p99 (Claude Code `Stop`, nothing appended: daemon-side p95 15.3 → 81.9 ms; AGY `Stop` with 4 KB appended: 43.1 → 113.3 ms).

### Commit benchmark

`scripts/bench_commit.py` measures the commit latency of the two authority databases on the real code paths, under three journal modes, and runs a crash test. It is the measurement behind the default of [`SELF_DIRECT_SQLITE_JOURNAL`](#sqlite-journal-mode-index-files-and-backups).

```bash
python scripts/bench_commit.py --samples 300 --warmup 15 --kill 50 --tmpdir /path/on/ext4 --shm /dev/shm \
    --slow-dir /path/on/9p --out report.json --markdown report.md
```

- **Operations.** *append*: `ingest_session` of a synthetic Codex transcript that grew by 3 KB, the way a hook syncs it (`derived=False`, no embedding), timed as a whole, as `SessionStore.commit_events` (events, cursor, block table and audit bookkeeping in one transaction) and as the `commit()` call inside it. *receipt*: `ReceiptStore.record` (`PreToolUse` and `PostToolUse` of a synthetic AGY transcript). *hookstate*: `HookState.record_error`, which is not an authority record and is not switched; WAL is set on it by hand to show what it would buy.
- **Process models.** *resident*: one store and its connections stay open (the daemon, the stdio server). *fresh*: every sample opens the store, does one operation and closes it (a command hook; with WAL the last connection to close folds the log into the database).
- **Modes**, sampled in random order round-robin so drift hits all of them: `delete_full` (today), `wal_full` (the candidate), `wal_normal` (**reference only, not shippable**: after power loss it can lose acknowledged commits) and `wal_keep` (an experiment that is not in the package: WAL without the fold at close).
- **File systems.** The scratch directory (ext4), `/dev/shm` (tmpfs, the CPU-only floor) and the project checkout on WSL (9p/drvfs, which `choose_journal` denies: DELETE only, WAL is never enabled there).

Median / p95 / p99 in ms, 300 samples per cell after 15 warm-up samples, one machine (AMD Ryzen 7 5800X, WSL2 ext4 on a virtual disk, Linux 6.18, SQLite 3.53.1, Python 3.12), 1-minute load average 5.4 to 6.1 during the ext4 cells (5-minute 4.8 to 5.2; the machine was busy with other work, and the modes are interleaved so it hits all of them alike). Windows could not be measured.

| ext4, append | measure | DELETE + FULL (today) | WAL + FULL | WAL + NORMAL (reference only) | WAL + FULL vs today, p50 |
|---|---|---|---|---|---|
| resident | whole operation | 24.5 / 31.6 / 36.2 | 10.7 / 15.6 / 20.1 | 5.2 / 10.2 / 24.3 | -56 % |
| resident | `commit_events` | 21.0 / 28.2 / 31.0 | 6.8 / 10.3 / 15.7 | 1.8 / 4.5 / 21.2 | -68 % |
| resident | `commit()` | 18.6 / 26.6 / 29.9 | 4.9 / 9.0 / 13.9 | 0.1 / 0.2 / 18.1 | -74 % |
| fresh | whole operation | 29.2 / 37.8 / 43.0 | 32.2 / 39.6 / 42.5 | 32.1 / 39.4 / 44.9 | **+10 %** |
| fresh | `commit_events` | 25.3 / 32.2 / 38.0 | 20.4 / 25.8 / 27.5 | 12.5 / 17.3 / 19.9 | **-20 %** |
| fresh | `commit()` | 23.2 / 29.8 / 34.5 | 18.3 / 23.1 / 24.7 | 10.6 / 15.2 / 17.8 | -21 % |

| ext4, receipt | measure | DELETE + FULL | WAL + FULL | WAL + NORMAL (reference only) | WAL + FULL vs today, p50 |
|---|---|---|---|---|---|
| resident | whole `record()` | 24.6 / 31.9 / 33.6 | 12.0 / 16.7 / 18.4 | 1.6 / 2.4 / 2.8 | -51 % |
| resident | `commit()` | 22.7 / 30.1 / 31.7 | 10.4 / 15.2 / 17.1 | 0.0 / 0.1 / 0.2 | -54 % |
| fresh | whole `record()` | 22.8 / 27.3 / 31.2 | 25.2 / 33.1 / 38.4 | 25.6 / 30.9 / 33.0 | **+10 %** |
| fresh | `commit()` | 20.9 / 25.3 / 29.6 | 15.0 / 20.7 / 23.3 | 9.7 / 12.6 / 14.2 | -28 % |

| other file systems | operation | model | measure | DELETE + FULL | WAL + FULL |
|---|---|---|---|---|---|
| tmpfs | append | resident | whole / `commit_events` | 4.1 / 8.5 / 10.6 and 1.7 / 3.2 / 4.1 | 3.9 / 8.3 / 10.6 and 1.6 / 3.2 / 3.8 |
| tmpfs | append | fresh | whole / `commit_events` | 4.8 / 7.7 / 8.3 and 1.7 / 2.9 / 3.3 | 4.8 / 7.2 / 8.4 and 1.7 / 2.7 / 3.2 |
| tmpfs | receipt | resident / fresh | whole | 1.2 / 1.7 / 1.9 and 1.2 / 1.8 / 2.0 | 1.1 / 1.5 / 1.7 and 1.4 / 1.9 / 2.1 |
| 9p (`denied:9p`) | append | resident / fresh | whole | 187.7 / 286.6 / 326.7 and 206.7 / 288.3 / 306.4 | not run: denied |
| 9p (`denied:9p`) | append | resident / fresh | `commit_events` | 144.7 / 230.9 / 269.3 and 144.8 / 220.8 / 234.0 | not run: denied |
| 9p (`denied:9p`) | receipt | resident / fresh | whole | 97.5 / 116.6 / 126.5 and 155.0 / 176.8 / 187.5 | not run: denied |

What the tables say. (1) With the process kept open WAL halves the cost of an append or a receipt and the synchronous commit stops being the dominant cost; `NORMAL` would halve what is left again, which is why it is tempting and why it is excluded. (2) With a fresh process per operation the transaction is 20 to 28 % cheaper, but opening and closing the database costs it back (a new log file, recovery or the fold at close), and the whole operation is 10 % slower; on tmpfs, where there is no `fsync` to save, WAL is neutral. (3) Keeping the log at the end of a short-lived process (`wal_keep`) is not a way out: the log grows to the 1000-page autocheckpoint and every next process recovers it (ext4 whole append 37.4 ms against 32.2 ms with the fold; tmpfs `commit_events` 2.05 times slower). (4) On the 9p mount of this checkout one append takes about 190 to 210 ms with the rollback journal, seven to eight times the ext4 figure and the reason to keep the index on a local disk; WAL is denied there (it needs shared memory in a `-shm` file that network and virtual file systems do not keep coherent) and is not measured. (5) The hook-state database in WAL (set by hand) is 15 % slower for a fresh process; it stays as it is.

**Crash test (`--kill 50`).** A writer process loops over real commits (three index appends and one receipt record per round, each printing its id after the commit returned), the script sends `SIGKILL` after a random number of acknowledgements and a random 0 to 40 ms, then runs `PRAGMA integrity_check` on both databases, checks that every acknowledged id is present, and reopens them through the package's own classes. 50 kills per mode on ext4:

| mode | acknowledged appends / receipts | lost | integrity failures | reopen failures | committed but not acknowledged | killed with a hot journal / a non-empty `-wal` |
|---|---|---|---|---|---|---|
| `delete_full` | 1244 / 389 | 0 | 0 | 0 | 37 | 47 / 0 |
| `wal_full` | 1257 / 396 | 0 | 0 | 0 | 119 | 0 / 50 |
| `wal_keep` (experiment) | 1138 / 350 | 0 | 0 | 0 | 163 | 0 / 50 |
| `wal_normal` (reference) | 1103 / 339 | 0 | 0 | 0 | 145 | 0 / 50 |

No acknowledged commit was lost in any mode. That is **process-kill survival, not power-loss durability**: a killed process leaves everything the operating system already holds in its page cache, so every mode passes by construction. What `synchronous=FULL` buys over `NORMAL` is the case this test cannot show, and the reason the package never sets `NORMAL`.

### Memory verification

tsukkomi is Python, but it runs native code: neograph-engine (the audit and GraphRAG graphs) and the sqlite-vector extension (dense search). After the 0.5.x memory incident the project checks both the native code and the long-running server, modelled on NeoGraph's sanitizer CI. `.github/workflows/memory.yml` runs three jobs:

| Job | What runs | Fails on |
|---|---|---|
| `asan` | neograph-engine and sqlite-vector rebuilt with AddressSanitizer + UndefinedBehaviorSanitizer; the whole suite with the runtime preloaded, `PYTHONMALLOC=malloc` and the native vector backend | any ASan or UBSan report; a direct leak allocated by neograph or sqlite-vector code |
| `tsan` | the same libraries rebuilt with ThreadSanitizer; the whole suite under `setarch -R` | any data race or deadlock report |
| `soak` | `scripts/memory_soak.py` with the official sqlite-vector release: one long-lived process drives the MCP tool entry points through steady, growing and churning sessions | retained Python heap growing in both measured windows of a phase, or growing file descriptors or threads |

How the sanitizer jobs work (`scripts/verify_memory.py`, standard library only, Linux and GCC):

- `native` downloads the neograph-engine sdist pinned in `pyproject.toml` and checks its SHA-256 against both PyPI and the digest pinned in the script, clones sqlite-vector `1.1.0` and checks that it resolves to the pinned commit, and builds both with the profile's flags. The builds are cached by these pins; a cold build of neograph-engine takes about 45 minutes on a hosted runner.
- `canary` builds a small library with one deliberate defect per check (heap overflow, signed overflow, a leak, a data race) and requires the sanitizer to report each. A run without reports therefore means the sanitizer was live, not that it was missing.
- `test` refuses to run unless the installed neograph module links the sanitizer runtime, so the release wheel can never be tested by mistake. It then runs the suite and scans the sanitizer logs and the test output.
- LeakSanitizer does not run at interpreter exit, where CPython leaves thousands of objects unfreed by design. `tests/conftest.py` runs LSan's recoverable check when the pytest session finishes, while live Python objects are still reachable. Each direct leak is attributed to the first frame above the allocator: neograph or sqlite-vector code fails the run, while CPython and third-party leaks are counted and printed.
- Under TSan, `/bin/sh` itself crashes with the runtime preloaded, so the two installer tests that start hook commands through a shell are deselected there.

Run them locally:

```bash
python scripts/verify_memory.py native --profile asan --out build/memory/asan
python -m pip install build/memory/asan/wheels/neograph_engine-*.whl && python -m pip install -e ".[dev]"
python scripts/verify_memory.py canary --profile asan --out build/memory/asan
python scripts/verify_memory.py test --profile asan --out build/memory/asan    # same for --profile tsan, in its own venv
python scripts/memory_soak.py --check --dense-backend sqlite-vector --vector-path /absolute/path/to/vector.so
```

Use a separate virtual environment per profile, because the sanitizer build replaces the release wheel. `TSUKKOMI_TEST_DENSE_BACKEND=sqlite-vector` with `SQLITE_VECTOR_TEST_PATH` runs any suite against a native vector library. Known limit: CPython 3.12 builds where interned strings are immortal (Ubuntu 24.04's 3.12.3, for example) keep each path component that `pathlib` interns, about 150 bytes per distinct session path. This is bounded by the number of sessions; newer 3.12 releases and 3.13 release them.

Daemon tests under the sanitizers: the daemon, the relays and the hook scripts start in a clean environment on purpose (that keeps credentials out of the daemon), so a child process of a sanitizer run would load the instrumented NeoGraph extension without the preloaded runtime and die at import. `tests/daemon/test_daemon_hooks.py` and `test_daemon_mcp.py` therefore skip the tests that start real processes under the ASan and TSan profiles (60 of 161 daemon tests); the in-process daemon tests, which run the daemon's own threads, sockets and drain logic inside the sanitized interpreter, still run, and the ordinary Linux job runs all of them. The in-process daemon tests also never apply the production resident-memory cap to the pytest process itself.

Known third-party exemption: sqlite-vector 1.1.0 computes `1 << 31` on an `int` in `cpu_supports_avx512()` (`src/distance-cpu.c`), which UBSan reports as `shift-base` on CPUs that reach that AVX-512 probe (GitHub runners do; a Zen 3 workstation does not). The ASan build of that one library is compiled with `-fno-sanitize=shift-base`; NeoGraph and the canaries keep the full `-fsanitize=undefined` set. The sanitizer suites run pytest without output capture (`-s`), so a native report reaches the job log.

### Shared-index queue PoC

The opt-in `scripts/poc_shared_index_queue.py` experiment connects separate client processes to one authenticated IPv4 loopback service. It uses the real NeoGraph `RequestQueue` through a small C++/ctypes bridge and runs real `SelfDirectEngine` operations against a temporary synthetic index. It does not replace the deployed MCP transport, touch existing indexes, load API keys or add a CI gate.

Requirements: a POSIX host, an installed GCC-compatible C++20 compiler, the project Python dependencies, and a NeoGraph source checkout containing `include/neograph/util/request_queue.h` and `deps/concurrentqueue.h`. No dependencies are downloaded by the PoC.

```bash
python scripts/poc_shared_index_queue.py run --neograph-root ../NeoGraph --workers 4
```

The runner compiles only the bridge, compares 16 independent clients with one worker and the selected multiworker count, and checks real clean/violation verdicts, bounded rejection, queued expiry, disconnect cancellation, running deadlines and shutdown settlement. It separately reports `writer_lease_peak`, actual native audit `evaluation_peak`, metadata progress during evaluation and `residual_writer_contention`. Deliberate writer/native-stage holds are labeled diagnostic fixtures, not production latency. Wall throughput includes client startup and IPC and has no performance pass/fail threshold.

For an interactive experiment, use `serve --neograph-root ../NeoGraph --workers 1 --capacity 32`; its readiness line gives a private endpoint-file path. The `client --endpoint-file <path> --request '<JSON>'` mode supports `check_action`, `sync_session` with `embed=false`, `audit_session` and `audit_status` for the synthetic session. `{"op":"stop"}` closes the queue, settles accepted futures and shuts the service down naturally. Endpoint credentials and compiled libraries are temporary; do not commit an endpoint file.

A queued request that expires never enters the engine. An already-started request can temporarily outlive its response deadline: it reports `still_running`/unknown, then settles at a cooperative checkpoint or returns an actual admitted-work outcome. Multiple workers can overlap detached evaluation, but cannot bypass a genuinely held writer lease. The PoC never claims forced cancellation of a running syscall.

### Verification scope

Test results and staged secret-pattern scans are bounded evidence, not certification that arbitrary secrets or every host bypass are absent. Inspect matched staged lines before committing, keep credentials in an external authorized dotenv file, and never commit raw personal transcripts or local runtime state.

---

## Security and changelog

See [SECURITY.md](SECURITY.md) for supported versions and how to report a vulnerability, and [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

MIT License. See [LICENSE](LICENSE) for details.
