"""Best-effort secret and blob masking for everything that leaves the local evidence store.

The canonical index keeps raw session text so regex audits see the truth. Remote embedding
inputs, remote graph extraction, MCP results, hook output, injected context and error text
pass through ``mask_secrets`` / ``mask_value`` instead. Masking is pattern based defense in
depth, never proof that text is free of credentials.

Every pattern is linear or explicitly bounded (no nested unbounded quantifiers), leaves its
own replacement unchanged when applied again, and uses no bare hex heuristic, so commit SHAs,
content hashes and UUIDs are untouched.
"""
from __future__ import annotations

import re

# Rules run in order. ``hints`` are lowercase substrings: when none occurs in the lowered text the
# rule cannot match and is skipped, so ordinary text costs a few C-speed substring scans.
_RULES: list[tuple[tuple[str, ...], re.Pattern[str], object]] = []


def _rule(hints, pattern, replacement):
    _RULES.append((tuple(hints), re.compile(pattern), replacement))


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
      lambda m: m.group(1) + '"[encrypted content omitted]"')
_BLOB = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{256,}={0,2}(?![A-Za-z0-9+/=])")


def _blob(match: re.Match[str]) -> str:
    body = match.group(0)
    # Random base64 this long mixes cases and digits; plain runs ("AAAA...", hex dumps) are not blobs.
    if body.isalpha() or body.isdigit() or body.islower() or body.isupper():
        return body
    return f"[base64 blob omitted: {len(body)} chars]"


_RULES.append(((), _BLOB, _blob))

# --- Credentials embedded in URLs and headers ----------------------------------------------------
_rule(("://",), r"(?i)\b([a-z][a-z0-9+.-]{1,20}://)([^\s/?#@:]*):([^\s/?#@]+)@", r"\1\2:***@")
_rule(("authorization",),
      r"""(?i)(\bauthorization(?:\\*["'])?[ \t]*[:=][ \t]*(?:\\*["'])?)"""
      r"""((?:bearer|basic|digest|negotiate|ntlm|token|apikey|aws4-hmac-sha256)[ \t]+)?[^\s"',;}\]\\]+""",
      lambda m: m.group(1) + (m.group(2) or "") + "***")
_rule(("cookie",),
      r"""(?i)(\b(?:set-)?cookie(?:\\*["'])?[ \t]*:[ \t]*(?:\\*["'])?)[^\r\n"'\\]+""", r"\1***")
_rule(("bearer",), r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*", "Bearer ***")

# --- Provider token formats (value based; the prefix is kept so the kind of leak stays visible) --
_rule(("sk-",), r"\bsk-[A-Za-z0-9_\-]{8,}\b", "sk-***")  # OpenAI sk-/sk-proj-, OpenRouter sk-or-v1-, Anthropic sk-ant-
_rule(("_live_", "_test_"), r"\b((?:sk|rk)_(?:live|test)_)[A-Za-z0-9]{8,}\b", r"\1***")  # Stripe
_rule(("whsec_",), r"\b(whsec_)[A-Za-z0-9]{16,}\b", r"\1***")
_rule(("ghp_", "gho_", "ghu_", "ghs_", "ghr_"), r"\b(gh[pousr]_)[A-Za-z0-9]{20,}\b", r"\1***")  # GitHub
_rule(("github_pat_",), r"\b(github_pat_)[A-Za-z0-9_]{20,}\b", r"\1***")
_rule(tuple(f"gl{kind}-" for kind in ("pat", "ptt", "rt", "dt", "oas", "cbt", "imt", "ft", "soat", "agent")),
      r"\b(gl(?:pat|ptt|rt|dt|oas|cbt|imt|ft|soat|agent)-)[A-Za-z0-9_.-]{16,}", r"\1***")  # GitLab
_rule(("xox", "xapp-"), r"\b(xox[abeprs]-|xapp-)[A-Za-z0-9-]{8,}", r"\1***")  # Slack
_rule(("akia", "asia", "agpa", "aida", "aipa", "anpa", "anva", "aroa"),
      r"\b((?:AKIA|ASIA|AGPA|AIDA|AIPA|ANPA|ANVA|AROA))[A-Z0-9]{16}\b", r"\1***")  # AWS key ids
_rule(("aiza",), r"\b(AIza)[0-9A-Za-z_-]{30,}", r"\1***")  # Google API keys
_rule(("ya29.", "gocspx-"), r"\b(ya29\.|GOCSPX-)[0-9A-Za-z_-]{20,}", r"\1***")  # Google OAuth
_rule(("hf_", "api_org_"), r"\b(hf_|api_org_)[A-Za-z0-9]{20,}\b", r"\1***")  # Hugging Face
_rule(("npm_",), r"\b(npm_)[A-Za-z0-9]{30,}\b", r"\1***")
_rule(("pypi-",), r"\b(pypi-)AgEIcHlwaS5vcmc[A-Za-z0-9_-]{20,}", r"\1***")
_rule(("eyj",), r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*", "eyJ***")  # JWT

# --- Key/value pairs: JSON, env/.env, YAML, URL query strings, CLI assignments -------------------
# A key stem need not start at a word boundary (DB_PASSWORD, client_secret, X-Api-Key, SessionToken).
# Plural ``tokens`` and ``token_type`` do not match: a quote or separator must follow the stem.
_STEM = (r"api[_-]?key|api[_-]?secret|secret[_-]?(?:access[_-]?)?key|private[_-]?key|"
         r"(?:signing|encryption|master|access)[_-]?key|passw(?:or)?d|passwd|secret|token|credentials?")
# URL query strings end a value at ``&`` (and ``#``), so unrelated parameters stay readable.
_rule(("key", "secret", "passw", "token", "credential"),
      r"(?i)([?&;][\w.-]{0,40}(?:" + _STEM + r")=)[^&\s#\"'\\]*", r"\1***")
_KEY_VALUES = re.compile(r"(?i)((?:" + _STEM + r""")(?:\\*["'])?[ \t]*[:=][ \t]*)(""" + _QUOTED + r"""|[^\s,;}\]]+)""")
_QUOTE_PAIR = re.compile(r"""(\\*["'])[\s\S]*?(\\*["'])""")
_TRAILING_QUOTES = re.compile(r"""["']+\Z""")


def _mask_key_value(match):
    value = match.group(2)
    if value.startswith("***"):  # already masked (a query rule ran first); the rest is not secret
        return match.group(0)
    if len(value) >= 2 and value.lstrip("\\")[:1] in ('"', "'"):
        quotes = _QUOTE_PAIR.fullmatch(value)
        if quotes:
            return match.group(1) + quotes.group(1) + "***" + quotes.group(2)
    # A bare value may carry the closing quote of an enclosing shell word: keep it.
    tail = _TRAILING_QUOTES.search(value)
    return match.group(1) + "***" + (tail.group(0) if tail else "")


_RULES.append((("key", "secret", "passw", "token", "credential"), _KEY_VALUES, _mask_key_value))
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


def mask_error(exc: BaseException) -> str:
    """Client-safe description of an exception: its type and a masked message."""
    detail = mask_secrets(str(exc))
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
