"""Claude Code session-id grammar, standard library only.

The hook adapter validates ``session_id`` before it knows whether the event needs the transcript parser. Keeping the
grammar here lets the bookkeeping events (``UserPromptSubmit``, ``PostToolUse``) answer without importing
``claude`` and, through it, the pydantic models. ``claude`` re-exports these names.

Session ids come from the transcript's location, never from record contents:

    <project>/<uuid>.jsonl                          -> "<uuid>"
    <project>/<uuid>/subagents/<stem>.jsonl         -> "<uuid>_<stem>"   (child of session <uuid>)
"""
from __future__ import annotations

import re

SUBAGENTS = "subagents"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def session_parts(session_id) -> tuple[str, str | None] | None:
    """``(session uuid, subagent stem or None)`` for a canonical Claude Code session id, else None."""
    if not isinstance(session_id, str):
        return None
    if UUID.fullmatch(session_id):
        return session_id, None
    if len(session_id) > 37 and session_id[36] == "_" and UUID.fullmatch(session_id[:36]) \
            and STEM.fullmatch(session_id[37:]):
        return session_id[:36], session_id[37:]
    return None


def valid_id(session_id) -> bool:
    return session_parts(session_id) is not None
