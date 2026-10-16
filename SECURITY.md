# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.6.x   | Yes       |
| < 0.6   | No        |

Security fixes are made on the latest 0.6.x release. Upgrade before reporting an issue found in an older version, and read the upgrade notes in the [README](README.md#upgrading-to-060).

## Reporting a Vulnerability

Please report suspected vulnerabilities privately through GitHub: open the repository's **Security** tab and choose **Report a vulnerability** (GitHub private vulnerability reporting). This creates a private advisory visible only to you and the maintainers.

If the problem has no exploit detail or sensitive data, such as a documentation error or a hardening suggestion, you may open a normal [issue](https://github.com/fox1245/tsukkomi-mcp/issues) instead.

Include the version, the affected host (Codex, Claude Code, OMP, AGY or Grok Bot), and a minimal reproduction using synthetic session data. Do not attach real session transcripts, API keys, account identifiers or private index files. If a secret was exposed, revoke it first.

This is a small project maintained on a best-effort basis; no response time is guaranteed. Discussion happens on the advisory or issue. A confirmed vulnerability is fixed on the latest supported release where feasible and noted in the [changelog](CHANGELOG.md).

## Scope

The project's data-handling model is part of what this policy covers:

- Session transcripts, the index and contracts stay on the local machine. Remote calls are limited to embeddings, GraphRAG extraction and the opt-in workflow classifier, all sent to OpenRouter and only with a key the owner configured, and, only when the owner selects the opt-in `pgvector` dense backend, chunk ids and embedding vectors (never text) sent to the PostgreSQL server named in a private connection-string file (see [Dense backends](README.md#dense-backends); vectors are not anonymous, a server that is not this machine needs `sslmode=verify-full`, and a real remote server needs the owner's approval). With `SELF_DIRECT_LOCAL_ONLY=true`, embeddings and LLM graph updates are rejected, and workflow classification stays off unless the workflow owner approves remote synthetic context.
- Known secret patterns are masked before embedding and graph extraction and in everything the server and its hooks return, and the workflow classifier refuses context in which a secret pattern is found. Masking is pattern-based; it does not guarantee that every secret format or all personal data is removed. See the [masking contract](#masking-contract).

In scope, for example: leaking a credential or session text outside the documented remote calls, a bypass of secret masking for a known pattern, a project-controlled file (such as a `.env` in the working directory) that changes the index, contracts or mode a lifecycle hook reads, an escape from the configured session roots or sandbox, a way for an agent to forge host receipts or workflow approvals through the MCP API, and a verdict reported as `clean` when the evidence is changed or unread.

Out of scope, for example:

- An attacker with write access as the same OS user to the index, contracts, hook configuration, approval directories or credential files. Those must be protected by the host.
- Actions the hooks do not see: hosted tools that emit no `PreToolUse` event, out-of-band shell access, in-memory sessions and records not yet written to disk. The Codex hooks are advisory by design and never block. The AGY, Claude Code and Grok CLI hooks block only what their documented mode says (by default a confirmed `violation`, and `suspicious` as a confirmation request) and fail open on their own errors.
- Vulnerabilities in dependencies or services (NeoGraph, the MCP SDK, sqlite-vector, OpenRouter, Lean, bubblewrap); report those upstream.

## Hook configuration

Hosts run lifecycle hooks inside the project being edited, so anything in the working directory is content the audited agent can write. The AGY, Claude Code and Grok CLI hooks therefore build their settings from the process environment and the user TOML only and never read a working-directory `.env`; otherwise a project `.env` could point a hook at an empty index or at `advisory` mode and turn a confirmed violation into an allow. The MCP server keeps reading `./.env` at the lowest precedence. The hook's index, contracts path and enforcement mode must be set through its environment, the user TOML or the installer's `--index-dir`. A user or process with write access to those, to the hook script or to the host settings remains out of scope (see above).

**Grok CLI session files.** The `grok` provider reads exactly one file per session, `<sessions dir>/<url-encoded cwd>/<session id>/updates.jsonl`, under the configured root (`SELF_DIRECT_GROK_SESSIONS_DIR`, default `$GROK_HOME/sessions` or `~/.grok/sessions`). The resolver refuses every other file name (`chat_history.jsonl` and `system_prompt.txt` hold the rendered system prompt and are never opened), a path outside the root, `..`, and any symlink or junction below the root. The agent's reasoning (`agent_thought_chunk`) is never indexed, and the host's `hook_execution` / `hook_annotation` records are skipped. `scripts/install_grok_hooks.py` writes only `<grok home>/hooks/tsukkomi.json`; it never touches `~/.claude/settings.json`, `config.toml` or `trusted_folders.toml` and never runs `grok`. A Claude Code hook that receives a Grok payload (Grok loads Claude's settings file) answers silently and writes no state.

## Masking contract

The index is a local evidence store, and the audit needs the truth, so what is stored and what leaves are treated differently.

**Stored raw.** The canonical event store (`meta.sqlite`), the original session files, the checklist and contract files keep text as the host or caller wrote it, so a regular expression audit can see a secret that a session contains. Anyone who can read the index directory can read those secrets. Keep the index private and outside agent-writable workspaces. While a re-index is staged, the same raw rows are also in `<index dir>/reindex/*.spool` (a 0600 file in a 0700 directory on POSIX; deleted when the re-index is published or discarded; see [Re-indexing](README.md#re-indexing)): it is part of the index directory for everything said here.

**Masked on every way out.** Text passes through `security/mask.py` (`mask_secrets` for strings, `mask_value` for JSON-like values) before it leaves the evidence store:

- MCP tool results and MCP tool error text, including evidence snippets and `get_chunk` output.
- Output of the Codex, AGY, Claude Code and Grok CLI hooks, including injected context and fail-open warnings.
- Server log records (messages, formatted tracebacks and stack information) through a logging filter on the root logger's handlers, and the traceback printed if the server crashes. A record whose masking itself fails is replaced by `log record withheld (masking failed)`. Error output of the Codex installer and the workflow command is masked too.
- Text sent to OpenRouter: embedding inputs (masked before a long event is split, so a secret crossing a segment boundary is still caught), search queries, GraphRAG extraction prompts and the opt-in workflow classifier context, which is refused outright if a secret pattern is found.
- The sparse (FTS5) search text, for rows written since 0.6.0 (both formats: see below).

**What the sparse index file holds.** The sparse (FTS5) index is derived data, rebuilt from the canonical store, and everything it indexes passes through `mask_secrets` first (the same masking in both formats). Two files exist (`SELF_DIRECT_SPARSE_FORMAT`, README "Sparse index formats"):

- `sparse-v2.sqlite` (the compact format, the default): **stores no text**. It holds the FTS5 inverted index of the masked text (the lower-case, stemmed terms with their positions and the document sizes), the chunk ids, the session ids and the kind of each chunk. There is no copy of any event, masked or raw. An inverted index with positions can in principle be used to reconstruct the stemmed, case-folded token sequence of a document (no punctuation, no original spelling), which is the same information the legacy file held in addition to the text itself, so the compact file holds strictly less than the legacy one and nothing that the legacy file did not hold.
- `sparse.sqlite` (the legacy format, used with `SELF_DIRECT_SPARSE_FORMAT=legacy` or by an older version): stores the masked text of every event next to its index, for rows written by 0.6.0 and later. Rows written by an earlier version stay raw until the file is rebuilt.

The program never deletes, truncates or rewrites `sparse.sqlite`; with the compact format it does not even open it for anything but a read-only row count in `audit_status`. **Delete it yourself** once the compact index is complete and no process uses the legacy format any more: stop every process that uses the index, delete `<index_dir>/sparse.sqlite` (and a leftover `sparse.sqlite-journal`), and start the servers again. This also removes any raw rows that a pre-0.6.0 legacy file still holds. To rebuild the compact file from the canonical store (for instance after deleting it), delete `<index_dir>/sparse-v2.sqlite` the same way or run `python -m self_directing_mcp.maintenance sparse-rebuild`; each session's documents are re-added, masked, on that session's next `sync_session` or audit refresh, on demand when a `sparse` or `hybrid` `search_history` covers it, or all at once by the rebuild command. Nothing else is lost; the canonical store and the vector index are not touched.

**Registered secrets.** `register_secret` gives the process an exact value to replace with `***` whatever its format. The OpenRouter key is registered when an embedder, the GraphRAG transport or the workflow classifier loads it (values shorter than 8 characters are ignored). Registration is per process: a process that never loads the key, such as a hook started with `--local-only`, relies on the patterns alone.

**Patterns.** Private key blocks, data URIs and long base64 runs (replaced by `[base64 data omitted]` or a blob placeholder), `encrypted_content` and thinking-signature fields, credentials inside URLs, `Authorization`, `Cookie` and `Bearer` values, common provider token formats (OpenAI-style `sk-`, Anthropic, OpenRouter, Stripe, GitHub, GitLab, Slack, AWS key ids, Google API and OAuth, Hugging Face, npm, PyPI, JWT), and key/value pairs whose key names an API key, secret, password, token or credential in JSON, `.env`, YAML, URL query strings and `--password value` style flags. There is no bare-hex heuristic, so commit SHAs, content hashes and UUIDs are left alone.

**What masking does not cover.**

- Key/value pairs are masked by the name of the key: besides `password`, `token`, `secret`, `api_key`, `credentials` and the `*_access_key`/`*_private_key` family, any `*_PASS`, `*_PWD`, `PASSPHRASE`, `*_KEY` (including `*_KEY_ID`), `*_SALT`, `PEPPER` and `*_AUTH` name is masked in env, YAML, JSON and query-string form. A bare `PWD` (the working directory) stays readable. Over-masking a non-secret `*_KEY` value hides only output text; contracts always match the raw record.
- It is pattern-based best effort. A secret in a format the patterns do not know, a password in prose with no recognizable key name, and personal data (names, e-mail addresses, paths, proprietary code) are not removed. Harmless text that looks like a secret may be masked.
- Contract rule fields `regex`, `before_regex` and `exception_regex`, and the single-use workflow `grant`, are returned unmasked so that they round-trip exactly. Never put a secret in a rule regex. Other contract text (`description`, `source_quote`) is stored as typed and masked only on the way out.
- Text a caller sends in (contracts, checklist items, tool arguments) is not masked before it is stored.
- The host keeps its own unmasked transcript, and masking cannot change what the host shows the model or the user.
- Rows written to a legacy `sparse.sqlite` before 0.6.0 are raw until that file is deleted or rebuilt, as above. The compact format never held them.
- The regular expression audit runs on raw text by design; its findings quote masked snippets.
- Log handlers added after the server has started are not covered until the filter is applied to them.
