# Changelog

Notable changes to tsukkomi-mcp (Python package `self_directing_mcp`). Versions follow `pyproject.toml`. Releases before 0.6.0 are not itemized here; see the git history.

## 0.6.0 - 2026-10-14

Merged in [PR #7](https://github.com/fox1245/tsukkomi-mcp/pull/7) and [PR #11](https://github.com/fox1245/tsukkomi-mcp/pull/11). PR #7 also carried the unmerged commits of the AGY and workflow branch. Read "Upgrading to 0.6.0" in the [README](README.md) before restarting existing servers: restart every process that shares an index.

### Upgrade-relevant changes

- `neograph-engine` is pinned to `0.14.1` (was `0.12.1`).
- The default index directory is `~/.self_direct_index` (was `.self_direct_index` inside the checkout). Installs that relied on the old default must set `SELF_DIRECT_INDEX_DIR` or TOML `index_dir`; nothing is moved automatically.
- A checkout `.env` is no longer selected as an authoritative key file. Use `SELF_DIRECT_OPENROUTER_API_KEY_FILE` or TOML `openrouter_api_key_file`. A plain `.env` is read only from the server's working directory.
- Parser versions: OMP 2 to 8, Codex 2 to 3. Each indexed session is reparsed lazily on its next sync; existing indexes, vector caches, contracts and checklists are preserved.

### Added

- `SELF_DIRECT_MAX_CONCURRENT_CALLS` (default 4) bounds concurrently running heavy tool calls ([#5](https://github.com/fox1245/tsukkomi-mcp/issues/5)).
- Coverage fields `judged_through_byte`, `unread_bytes` and `judged_prefix` for audits of growing sessions ([#4](https://github.com/fox1245/tsukkomi-mcp/issues/4)).
- `checklist_conflict` status from `update_checklist` ([#6](https://github.com/fox1245/tsukkomi-mcp/issues/6)).
- Duplicate session sources: `alias_of`, `coverage.indexed_path` and `coverage.requested_path`, the `requested_source_is_prefix_of_indexed` issue and the `ambiguous_session_source` error ([#10](https://github.com/fox1245/tsukkomi-mcp/issues/10)).
- Linux CI job `memory-audit-regressions` with small offline memory, freshness and concurrency tests ([#8](https://github.com/fox1245/tsukkomi-mcp/issues/8)).
- AGY evidence linkage and enforcement with host receipts (part of [#1](https://github.com/fox1245/tsukkomi-mcp/issues/1)) and a first implementation of the approved-workflow layer, the `workflow_*` tools and the `tsukkomi-workflow` command ([#2](https://github.com/fox1245/tsukkomi-mcp/issues/2)). Both issues remain open.
- User path configuration in `~/.config/tsukkomi-mcp/config.toml` (or `SELF_DIRECT_CONFIG_FILE`) and a shared OMP bridge bootstrap (`--omp-bridge`).
- OMP extension tool `tsukkomi_session_context`, which returns the actual persisted OMP session id.
- `CHANGELOG.md`, and a project-specific `SECURITY.md`.

### Changed

- Source capture streams a SHA-256 over the file up to its first-measured length and keeps no bytes. `validate_prefix()` tolerates pure growth after re-hashing the prefix; `validate()` is metadata only. `stat` and `fstat` stamps are compared within the same API, and an `st_ino` of 0 (some Windows filesystems) skips the inode comparison ([#4](https://github.com/fox1245/tsukkomi-mcp/issues/4), [#5](https://github.com/fox1245/tsukkomi-mcp/issues/5)).
- Audits judge a growing session as of the captured offset and say so in `coverage` and `notes`. `must` obligations that rest on absent or failed evidence become `unknown` while bytes are unread. Rewrites, truncation, replacement and deletion stay `unknown`. `requires_attention` follows the verdict ([#4](https://github.com/fox1245/tsukkomi-mcp/issues/4)).
- `check_action` keeps a proposed action's `requires_success` prerequisites `unknown` while required history is unread ([#8](https://github.com/fox1245/tsukkomi-mcp/issues/8)).
- Memory and concurrency: single-flight per (index, provider, session id), streaming reparse through a disk spool with batched publication, per-session id queries, float32 sync vectors, slimmer audit row loading, and `malloc_trim` on glibc after heavy calls ([#5](https://github.com/fox1245/tsukkomi-mcp/issues/5), [#10](https://github.com/fox1245/tsukkomi-mcp/issues/10)). On a 100 MB synthetic OMP session with 8 concurrent threads, peak RSS fell from 2,304 MB to 193 MB ([PR #7](https://github.com/fox1245/tsukkomi-mcp/pull/7)).
- `audit_status` counts sparse documents from `chunk_rows` instead of scanning the FTS5 table ([#5](https://github.com/fox1245/tsukkomi-mcp/issues/5)).
- Shared-index snapshots and cooperative request deadlines; `index_busy`, `request_deadline`, `request_cancelled` and `operation_timeout` are distinct results ([#3](https://github.com/fox1245/tsukkomi-mcp/issues/3)).
- OMP parser v8 accepts host notices, image placeholders, steering records, `session_init`, `model_usage`, `branch_summary` and `redactedThinking` as runtime records. Only `attribution: "user"` steering is a user turn; agent-attributed user-role messages are runtime text ([#9](https://github.com/fox1245/tsukkomi-mcp/issues/9)).
- Codex parser v3 accepts `agent_message`, `inter_agent_communication_metadata`, `tool_search_call`/`tool_search_output`, the newer `session_meta` fields and inherited parent headers. `base_instructions`, `dynamic_tools`, account ids and tool lists are never indexed ([PR #11](https://github.com/fox1245/tsukkomi-mcp/pull/11)).
- OMP history retrieval uses OpenRouter by default; offline mode is the explicit `SELF_DIRECT_LOCAL_ONLY=true`. Session evidence is masked before remote graph extraction, and workflow classification uses the shared OpenRouter key settings.

### Fixed

- Sync no longer fails with `source changed during preparation` when records are appended during capture ([#4](https://github.com/fox1245/tsukkomi-mcp/issues/4)).
- Equal file stamps no longer skip content verification, and a detected mismatch stays failed ([#8](https://github.com/fox1245/tsukkomi-mcp/issues/8)).
- A stale capture can no longer rewind a cursor that another process already published ([PR #7](https://github.com/fox1245/tsukkomi-mcp/pull/7)).
- Unbounded memory growth in long-running servers ([#5](https://github.com/fox1245/tsukkomi-mcp/issues/5)).
- Concurrent `update_checklist` and contract updates no longer fail with stale-preparation errors ([#6](https://github.com/fox1245/tsukkomi-mcp/issues/6)).
- Agent-attributed messages with partial steering metadata were treated as user turns, and payloads of unsupported image and `redactedThinking` blocks were indexed ([#9](https://github.com/fox1245/tsukkomi-mcp/issues/9)).
- Files with the same session id in different folders caused a full reparse on every sync ([#10](https://github.com/fox1245/tsukkomi-mcp/issues/10)).
- Real Codex sessions no longer report most records as unsupported or as parse errors; in a 571-session check, v2 showed unsupported events in 401 sessions and parse errors in 121, and v3 showed none ([PR #11](https://github.com/fox1245/tsukkomi-mcp/pull/11)).

### Known limitations

- Capture and validation are not an atomic snapshot. A rewrite outside the last 64 KiB between the final full hash and the publication lock is accepted at that moment; the next sync reindexes because the cursor's prefix hash no longer matches.
- On Windows `st_ctime` is the creation time, so a same-size rewrite that restores `mtime` is invisible to `stat`; the next sync catches it through the cursor prefix hash.
- `clean` for a growing session means clean as of `judged_through_byte`. Records appended during the audit are judged by the next audit.
- Memory with a live ~310 MB session was not measured before release.
