"""The host's timeout for each Claude Code hook event, in seconds, and the bound the tool keeps below it.

The installer (``scripts/install_claude_hooks.py``) writes ``HOST_TIMEOUT_SEC`` into the host's settings, and the hook's own
bound on its whole run is derived from the same table, so the two cannot drift. A host that times a pre-tool hook out does
not block the call, so a hook must answer before that, never after it. Standard library only.
"""
from __future__ import annotations

HOST_TIMEOUT_SEC = {
    "SessionStart": 30,
    "UserPromptSubmit": 30,
    "PreToolUse": 60,
    "PostToolUse": 30,
    "PostToolUseFailure": 30,
    "Stop": 120,
    "SubagentStop": 120,
}
# What the hook keeps back from the host's timeout: the interpreter start-up before the bound is first read, the answer
# that is printed after it, and the scheduling of a loaded machine.
DEADLINE_MARGIN_SEC = 10.0


def default_deadline_sec(event: str = "PreToolUse") -> float:
    """The default bound on the whole run of a hook for ``event``: the host's timeout less the margin."""
    return HOST_TIMEOUT_SEC[event] - DEADLINE_MARGIN_SEC
