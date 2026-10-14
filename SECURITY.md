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

Include the version, the affected host (Codex, OMP, AGY or Grok Bot), and a minimal reproduction using synthetic session data. Do not attach real session transcripts, API keys, account identifiers or private index files. If a secret was exposed, revoke it first.

This is a small project maintained on a best-effort basis; no response time is guaranteed. Discussion happens on the advisory or issue. A confirmed vulnerability is fixed on the latest supported release where feasible and noted in the [changelog](CHANGELOG.md).

## Scope

The project's data-handling model is part of what this policy covers:

- Session transcripts, the index and contracts stay on the local machine. Remote calls are limited to embeddings, GraphRAG extraction and the opt-in workflow classifier, all sent to OpenRouter and only with a key the owner configured. With `SELF_DIRECT_LOCAL_ONLY=true`, embeddings and LLM graph updates are rejected, and workflow classification stays off unless the workflow owner approves remote synthetic context.
- Known secret patterns are masked before embedding and graph extraction, and the workflow classifier refuses context in which a secret pattern is found. Masking is pattern-based; it does not guarantee that every secret format or all personal data is removed.

In scope, for example: leaking a credential or session text outside the documented remote calls, a bypass of secret masking for a known pattern, an escape from the configured session roots or sandbox, a way for an agent to forge host receipts or workflow approvals through the MCP API, and a verdict reported as `clean` when the evidence is changed or unread.

Out of scope, for example:

- An attacker with write access as the same OS user to the index, contracts, hook configuration, approval directories or credential files. Those must be protected by the host.
- Actions the hooks do not see: hosted tools that emit no `PreToolUse` event, out-of-band shell access, in-memory sessions and records not yet written to disk. The Codex hooks are advisory by design and never block.
- Vulnerabilities in dependencies or services (NeoGraph, the MCP SDK, sqlite-vector, OpenRouter, Lean, bubblewrap); report those upstream.
