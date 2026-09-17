from __future__ import annotations

import re

# Common secret-ish patterns to mask in snippets returned to agents.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b"), "sk-***"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*"), "Bearer ***"),
    (re.compile(r'(?i)\bapi[_-]?key\s*[=:]\s*["\']?[A-Za-z0-9_\-]{6,}["\']?'), "api_key=***"),
    (re.compile(r'(?i)\bauthorization\s*[=:]\s*["\']?Bearer\s+[^\s"\']+'), "Authorization=Bearer ***"),
    (re.compile(r"(?i)\bOPENROUTER_API_KEY\s*[=:]\s*\S+"), "OPENROUTER_API_KEY=***"),
]



# Preserve syntax while replacing quoted JSON values and shell-style assignments.
_KEY_VALUES = re.compile(
    r"""(?ix)(["']?\b(?:api[_-]?key|openrouter_api_key|access_token|refresh_token|password|passwd|secret|authorization)\b["']?\s*[:=]\s*)
    ("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;}\]]+)"""
)


def _mask_key_value(match):
    value = match.group(2)
    replacement = value[0] + "***" + value[0] if value[:1] in ('"', "'") else "***"
    return match.group(1) + replacement


def mask_secrets(text: str) -> str:
    if not text:
        return text
    out = text
    for pat, repl in _PATTERNS:
        out = pat.sub(repl, out)
    return _KEY_VALUES.sub(_mask_key_value, out)


def mask_value(value):
    if isinstance(value, str):
        return mask_secrets(value)
    if isinstance(value, dict):
        return {mask_secrets(str(key)): mask_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [mask_value(item) for item in value]
    return value
