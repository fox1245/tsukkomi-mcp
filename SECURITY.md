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

- Session transcripts, the index and contracts stay on the local machine. Remote calls are limited to embeddings, GraphRAG extraction and the opt-in workflow classifier, all sent to OpenRouter and only with a key the owner configured. With `SELF_DIRECT_LOCAL_ONLY=true`, embeddings and LLM graph updates are rejected, and workflow classification stays off unless the workflow owner approves remote synthetic context.
- Known secret patterns are masked before embedding and graph extraction and in everything the server and its hooks return, and the workflow classifier refuses context in which a secret pattern is found. Masking is pattern-based; it does not guarantee that every secret format or all personal data is removed. See the [masking contract](#masking-contract).

In scope, for example: leaking a credential or session text outside the documented remote calls, a bypass of secret masking for a known pattern, an escape from the configured session roots or sandbox, a way for an agent to forge host receipts or workflow approvals through the MCP API, and a verdict reported as `clean` when the evidence is changed or unread.

Out of scope, for example:

- An attacker with write access as the same OS user to the index, contracts, hook configuration, approval directories or credential files. Those must be protected by the host.
- Actions the hooks do not see: hosted tools that emit no `PreToolUse` event, out-of-band shell access, in-memory sessions and records not yet written to disk. The Codex hooks are advisory by design and never block. The AGY and Claude Code hooks block only what their documented mode says (by default a confirmed `violation`, and `suspicious` as a confirmation request) and fail open on their own errors.
- Vulnerabilities in dependencies or services (NeoGraph, the MCP SDK, sqlite-vector, OpenRouter, Lean, bubblewrap); report those upstream.

## Masking contract

The index is a local evidence store, and the audit needs the truth, so what is stored and what leaves are treated differently.

**Stored raw.** The canonical event store (`meta.sqlite`), the original session files, the checklist and contract files keep text as the host or caller wrote it, so a regular expression audit can see a secret that a session contains. Anyone who can read the index directory can read those secrets. Keep the index private and outside agent-writable workspaces.

**Masked on every way out.** Text passes through `security/mask.py` (`mask_secrets` for strings, `mask_value` for JSON-like values) before it leaves the evidence store:

- MCP tool results and MCP tool error text, including evidence snippets and `get_chunk` output.
- Output of the Codex, AGY and Claude Code hooks, including injected context and fail-open warnings.
- Server log records (messages, formatted tracebacks and stack information) through a logging filter on the root logger's handlers, and the traceback printed if the server crashes. A record whose masking itself fails is replaced by `log record withheld (masking failed)`. Error output of the Codex installer and the workflow command is masked too.
- Text sent to OpenRouter: embedding inputs (masked before a long event is split, so a secret crossing a segment boundary is still caught), search queries, GraphRAG extraction prompts and the opt-in workflow classifier context, which is refused outright if a secret pattern is found.
- The sparse (FTS5) search text, for rows written since 0.6.0.

**Sparse index and older rows.** `sparse.sqlite` stores masked text for rows written by 0.6.0 and later. Rows written by an earlier version stay raw until the file is rebuilt. To rebuild: stop every process that uses the index, delete `<index_dir>/sparse.sqlite`, and start the servers again. The file is recreated empty and each session's documents are re-added, masked, from the canonical store on that session's next `sync_session` or audit refresh, or on demand when a `sparse` or `hybrid` `search_history` covers it. Nothing else is lost; the canonical store and the vector index are not touched.

**Registered secrets.** `register_secret` gives the process an exact value to replace with `***` whatever its format. The OpenRouter key is registered when an embedder, the GraphRAG transport or the workflow classifier loads it (values shorter than 8 characters are ignored). Registration is per process: a process that never loads the key, such as a hook started with `--local-only`, relies on the patterns alone.

**Patterns.** Private key blocks, data URIs and long base64 runs (replaced by `[base64 data omitted]` or a blob placeholder), `encrypted_content` and thinking-signature fields, credentials inside URLs, `Authorization`, `Cookie` and `Bearer` values, common provider token formats (OpenAI-style `sk-`, Anthropic, OpenRouter, Stripe, GitHub, GitLab, Slack, AWS key ids, Google API and OAuth, Hugging Face, npm, PyPI, JWT), and key/value pairs whose key names an API key, secret, password, token or credential in JSON, `.env`, YAML, URL query strings and `--password value` style flags. There is no bare-hex heuristic, so commit SHAs, content hashes and UUIDs are left alone.

**What masking does not cover.**

- It is pattern-based best effort. A secret in a format the patterns do not know, a password in prose with no recognizable key name, and personal data (names, e-mail addresses, paths, proprietary code) are not removed. Harmless text that looks like a secret may be masked.
- Contract rule fields `regex`, `before_regex` and `exception_regex`, and the single-use workflow `grant`, are returned unmasked so that they round-trip exactly. Never put a secret in a rule regex. Other contract text (`description`, `source_quote`) is stored as typed and masked only on the way out.
- Text a caller sends in (contracts, checklist items, tool arguments) is not masked before it is stored.
- The host keeps its own unmasked transcript, and masking cannot change what the host shows the model or the user.
- Rows written to the sparse index before 0.6.0 are raw until the file is rebuilt, as above.
- The regular expression audit runs on raw text by design; its findings quote masked snippets.
- Log handlers added after the server has started are not covered until the filter is applied to them.
