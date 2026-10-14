"""Best-effort secret and blob masking for everything that leaves the local evidence store.

The canonical index keeps raw session text so regex audits see the truth. Remote embedding
inputs, remote graph extraction, MCP results, hook output, injected context and error text
pass through ``mask_secrets`` / ``mask_value`` instead. Masking is pattern based defense in
depth, never proof that text is free of credentials.

Every pattern is linear or explicitly bounded (quoted strings use unrolled loops, block scans are
capped), leaves its own replacement unchanged when applied again, and uses no bare hex heuristic,
so commit SHAs, content hashes and UUIDs are untouched.
"""
from __future__ import annotations

import re

# Rules run in order. ``hints`` are lowercase substrings: when none occurs in the lowered text the
# rule cannot match and is skipped, so ordinary text costs a few C-speed substring scans.
_RULES: list[tuple[tuple[str, ...], re.Pattern[str], object]] = []


def _rule(hints, pattern, replacement):
    # ASCII mode keeps case-insensitive search fast and treats CJK text next to a token as a boundary.
    _RULES.append((tuple(hints), re.compile(pattern, re.ASCII), replacement))


# --- Structural blocks -------------------------------------------------------------------------
_PEM = r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"
_PEM_MASK = "[PRIVATE KEY BLOCK MASKED]"
_rule(("-----end",), _PEM + r"[\s\S]{0,16384}?-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----", _PEM_MASK)
# A truncated block (no END marker): drop the header and the base64 armor body that follows it.
_rule(("-----begin",), _PEM + r"(?:\\[nr]|[\r\n]|[A-Za-z0-9+/=]){0,16384}", _PEM_MASK)

# Inline binary payloads. Image and encrypted content never belongs in model-visible output.
_rule(("base64",), r"(?i)data:[a-z0-9.+/-]{1,80};(?:[a-z0-9=._-]+;)*base64,[A-Za-z0-9+/_=-]{16,}",
      "[base64 data omitted]")
# A quoted string body without nested quantifiers: unrolled loop, linear even when unterminated.
_QUOTED = r""""[^"\\]*(?:\\.[^"\\]*)*"|'[^'\\]*(?:\\.[^'\\]*)*'|\\+["'][^"'\\]*\\+["']"""
_rule(("encrypted", "redacted", "thinking"),
      r"(?i)((?:encrypted[_-]?content|redacted[_-]?thinking|thinking[_-]?signature)"
      r"""(?:\\*["'])?[ \t]*[:=][ \t]*)(?:""" + _QUOTED + r"""|[^\s,;}\]]+)""",
      r'\1"[encrypted content omitted]"')


def _blob(match: re.Match[str]) -> str:
    body = match.group(0)
    # Random base64 this long mixes cases and digits; plain runs ("AAAA...", hex dumps) are not blobs.
    if body.isalpha() or body.isdigit() or body.islower() or body.isupper():
        return body
    return f"[base64 blob omitted: {len(body)} chars]"


# Long unbroken base64 runs (images, ciphertext). No substring hint exists, so this rule always runs.
_RULES.append(((), re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{256,}={0,2}(?![A-Za-z0-9+/=])", re.ASCII), _blob))

# --- Credentials embedded in URLs and headers ----------------------------------------------------
_rule(("://",), r"(?i)\b([a-z][a-z0-9+.-]{1,20}://)([^\s/?#@:]*):([^\s/?#@]+)@", r"\1\2:***@")
_rule(("authorization",),
      r"""(?i)(\bauthorization(?:\\*["'])?[ \t]*[:=][ \t]*(?:\\*["'])?)"""
      r"""((?:bearer|basic|digest|negotiate|ntlm|token|apikey|aws4-hmac-sha256)[ \t]+)?[^\s"',;}\]\\]+""",
      r"\1\2***")
_rule(("cookie",),
      r"""(?i)(\b(?:set-)?cookie(?:\\*["'])?[ \t]*:[ \t]*(?:\\*["'])?)[^\r\n"'\\]+""", r"\1***")
_rule(("bearer",), r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*", "Bearer ***")

# --- Provider token formats (value based; the prefix is kept so the kind of leak stays visible) --
# ``(?<![A-Za-z0-9])`` rather than ``\b``: a token glued to an underscore (``key_sk-...``) is still a token.
_rule(("sk-",), r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_\-]{8,}\b", "sk-***")  # OpenAI sk-/sk-proj-, OpenRouter sk-or-v1-, Anthropic sk-ant-
_rule(("_live_", "_test_"), r"(?<![A-Za-z0-9])((?:sk|rk)_(?:live|test)_)[A-Za-z0-9]{8,}\b", r"\1***")  # Stripe
_rule(("whsec_",), r"(?<![A-Za-z0-9])(whsec_)[A-Za-z0-9]{16,}\b", r"\1***")
_rule(("ghp_", "gho_", "ghu_", "ghs_", "ghr_"), r"(?<![A-Za-z0-9])(gh[pousr]_)[A-Za-z0-9]{20,}\b", r"\1***")  # GitHub
_rule(("github_pat_",), r"(?<![A-Za-z0-9])(github_pat_)[A-Za-z0-9_]{20,}\b", r"\1***")
_rule(tuple(f"gl{kind}-" for kind in ("pat", "ptt", "rt", "dt", "oas", "cbt", "imt", "ft", "soat", "agent")),
      r"(?<![A-Za-z0-9])(gl(?:pat|ptt|rt|dt|oas|cbt|imt|ft|soat|agent)-)[A-Za-z0-9_.-]{16,}", r"\1***")  # GitLab
_rule(("xox", "xapp-"), r"(?<![A-Za-z0-9])(xox[abeprs]-|xapp-)[A-Za-z0-9-]{8,}", r"\1***")  # Slack
_rule(("akia", "asia", "agpa", "aida", "aipa", "anpa", "anva", "aroa"),
      r"(?<![A-Za-z0-9])((?:AKIA|ASIA|AGPA|AIDA|AIPA|ANPA|ANVA|AROA))[A-Z0-9]{16}\b", r"\1***")  # AWS key ids
_rule(("aiza",), r"(?<![A-Za-z0-9])(AIza)[0-9A-Za-z_-]{30,}", r"\1***")  # Google API keys
_rule(("ya29.", "gocspx-"), r"(?<![A-Za-z0-9])(ya29\.|GOCSPX-)[0-9A-Za-z_-]{20,}", r"\1***")  # Google OAuth
_rule(("hf_", "api_org_"), r"(?<![A-Za-z0-9])(hf_|api_org_)[A-Za-z0-9]{20,}\b", r"\1***")  # Hugging Face
_rule(("npm_",), r"(?<![A-Za-z0-9])(npm_)[A-Za-z0-9]{30,}\b", r"\1***")
_rule(("pypi-",), r"(?<![A-Za-z0-9])(pypi-)AgEIcHlwaS5vcmc[A-Za-z0-9_-]{20,}", r"\1***")
_rule(("eyj",), r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*", "eyJ***")  # JWT

# --- Key/value pairs: JSON, env/.env, YAML, URL query strings, CLI assignments -------------------
# A key stem need not start at a word boundary (DB_PASSWORD, client_secret, X-Api-Key, SessionToken).
# Plural ``tokens`` and ``token_type`` do not match: a quote or separator must follow the stem.
# One rule per stem: each starts with a rare first character so a scan skips quickly through ordinary text.
_STEMS = (
    (("api_key", "api-key", "apikey", "api_secret", "api-secret", "apisecret"), r"api[_-]?(?:key|secret)"),
    (("secret",), r"secret(?:[_-]?(?:access[_-]?)?key)?"),
    (("private",), r"private[_-]?key"),
    (("signing", "encryption", "master", "access"), r"(?:signing|encryption|master|access)[_-]?key"),
    (("passw",), r"passw(?:or)?d"),
    (("token",), r"token"),
    (("credential",), r"credentials?"),
    # Common .env names the stems above miss: DB_PASS, MYSQL_PWD, PASSPHRASE, JWT_KEY, ACCESS_KEY_ID, *_SALT, REDIS_AUTH.
    # ``_pwd``/``_auth``/``_key`` need a preceding name part, so a bare PWD (the working directory) stays readable.
    # Over-masking a non-secret ``*_KEY`` value only hides text from output; contracts still match the raw record.
    (("_pass", "-pass", "passphrase"), r"(?:(?<=[a-z0-9])[_-]pass(?:word|wd|phrase)?|passphrase)"),
    (("_pwd", "-pwd"), r"(?<=[a-z0-9])[_-]pwd"),
    (("_key", "-key"), r"(?<=[a-z0-9])[_-]key(?:[_-]?id)?"),
    (("salt", "pepper"), r"(?:salt|pepper)"),
    (("_auth", "-auth"), r"(?<=[a-z0-9])[_-]auth"),
)
# URL query strings end a value at ``&`` (and ``#``), so unrelated parameters stay readable.
_QUERY = r"(?i)([?&;][\w.-]{0,40}(?:%s)=)[^&\s#\"'\\]*"
# The value is one of: a double-quoted, single-quoted or backslash-escaped-quoted string (quotes are kept so
# the syntax survives), or a bare token that stops before any quote. Already masked values are left alone.
_KEY_VALUES = (r"(?i)((?:%s)(?:\\*[\"'])?[ \t]*[:=][ \t]*)"
               r"""(?:(?P<dq>")[^"\\]*(?:\\.[^"\\]*)*"|(?P<sq>')[^'\\]*(?:\\.[^'\\]*)*'"""
               r"""|(?P<eq>\\+["'])[^"'\\]*\\+["']|(?!\*\*\*)[^\s\u00a0\u3000,;}\]"']+"""
               r"""|(?P<uq>\\*["'])[^\s\u00a0\u3000,;}\]"']*)""")  # unterminated quote: mask what follows
for _hints, _stem in _STEMS:
    _rule(_hints, _QUERY % _stem, r"\1***")
for _hints, _stem in _STEMS:
    _rule(_hints, _KEY_VALUES % _stem, r"\1\g<dq>\g<sq>\g<eq>\g<uq>***\g<dq>\g<sq>\g<eq>")
# ``--password hunter2`` style flags (space separated).
_rule(("--",), r"(?i)(--(?:[a-z0-9]+-)*(?:password|passwd|token|secret|api-?key)[ \t]+)(?!-)(?!\*\*\*)\S+", r"\1***")


_KNOWN: tuple[str, ...] = ()  # exact credentials this process holds; replaced whatever their format


def register_secret(value) -> None:
    """Remember a credential the process uses (e.g. the OpenRouter key) so it can never be echoed."""
    global _KNOWN
    if isinstance(value, str) and len(value.strip()) >= 8 and value.strip() not in _KNOWN:
        _KNOWN += (value.strip(),)


def mask_secrets(text: str) -> str:
    if not text:
        return text
    out = text
    for known in _KNOWN:
        if known in out:
            out = out.replace(known, "***")
    lowered = out.lower()
    for hints, pattern, replacement in _RULES:
        if hints and not any(hint in lowered for hint in hints):
            continue
        masked = pattern.sub(replacement, out)
        if masked is not out and masked != out:
            out, lowered = masked, masked.lower()
    return out


def mask_value(value, *, skip_keys=frozenset()):
    """Mask every string (and mapping key) inside a JSON-like value.

    ``skip_keys`` names mapping keys whose values are returned unchanged, for fields that
    must round-trip exactly (rule regexes, single-use grants).
    """
    if isinstance(value, str):
        return mask_secrets(value)
    if isinstance(value, dict):
        return {mask_secrets(key) if isinstance(key, str) else key:
                item if key in skip_keys else mask_value(item, skip_keys=skip_keys)
                for key, item in value.items()}
    if isinstance(value, list):
        return [mask_value(item, skip_keys=skip_keys) for item in value]
    if isinstance(value, tuple):
        return tuple(mask_value(item, skip_keys=skip_keys) for item in value)
    return value
