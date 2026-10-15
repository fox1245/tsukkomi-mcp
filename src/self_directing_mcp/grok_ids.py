"""Grok CLI session identity and payload recognition, standard library only.

The hook adapters validate ``session_id`` before they know whether an event needs the transcript parser, and the
Claude Code hook must recognise a Grok payload without importing anything heavy. Keeping the grammar here lets the
bookkeeping events answer without importing ``grok`` (and, through it, the pydantic models). ``grok`` re-exports
these names.

Layout (Grok CLI 1.0.x, ``docs/17-sessions.md`` of its user guide)::

    <sessions dir>/<url-encoded cwd>/<session id>/updates.jsonl

The group directory is the URL-encoded working directory (a slug plus hash, with the path in a ``.cwd`` file, when
the encoding exceeds 255 bytes), so it is never decoded or trusted: the session id comes from the directory that
holds ``updates.jsonl``, never from record contents. The session id is a UUID (v7 when Grok generates it; ``-s``
accepts any UUID). Only ``updates.jsonl`` is ever read: ``chat_history.jsonl`` and ``system_prompt.txt`` hold the
rendered system prompt and are not evidence.
"""
from __future__ import annotations

import re
from typing import Any

SESSION_FILE = "updates.jsonl"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_SEPARATORS = re.compile(r"[\\/]+")

# The envelope keys only Grok writes (Claude Code's payload is snake_case throughout).
_ENVELOPE_KEYS = ("workspaceRoot", "sessionId", "toolUseId", "permissionMode")


def valid_id(session_id: Any) -> bool:
    return isinstance(session_id, str) and UUID.fullmatch(session_id) is not None


def _segments(path: str) -> list[str]:
    return [part for part in _SEPARATORS.split(path) if part]


def under_grok_sessions(path: Any) -> bool:
    """True when ``path`` has a ``.grok`` segment directly followed by ``sessions`` (either slash style, any case)."""
    if not isinstance(path, str):
        return False
    parts = [part.lower() for part in _segments(path)]
    return any(left == ".grok" and right == "sessions" for left, right in zip(parts, parts[1:]))


def _transcript(payload: dict) -> str | None:
    for key in ("transcript_path", "transcriptPath"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def is_grok_payload(payload: Any) -> bool:
    """Whether a hook payload was sent by Grok CLI.

    Grok loads Claude Code's settings file by default, so a Claude hook installed there is also run on Grok's
    events, with Grok's payload on stdin. That payload is a superset of Claude Code's: it keeps the snake_case keys
    and adds the camelCase ones. Two independent signs, either of which suffices:

    * the transcript path lies under a ``.grok/sessions`` directory (SessionStart and UserPromptSubmit carry none);
    * the envelope names the event in Grok's own spelling (``hookEventName``, a string; Claude Code only writes
      ``hook_event_name``) together with another camelCase key Claude Code never sends, and no transcript path
      that names Claude Code's configuration directory.

    Only top-level keys, which the host writes, are consulted; tool input and output are never looked at.
    """
    if not isinstance(payload, dict):
        return False
    path = _transcript(payload)
    if path is not None and under_grok_sessions(path):
        return True
    if not isinstance(payload.get("hookEventName"), str):
        return False
    if path is not None and any(part.lower() == ".claude" for part in _segments(path)):
        return False
    return any(isinstance(payload.get(key), str) for key in _ENVELOPE_KEYS)
