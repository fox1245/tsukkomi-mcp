"""Standard-library state and fail-open responses for the Claude Code hook adapter.

Like ``agy_hook_state`` this module must stay importable when the rest of the runtime is not (missing native
extension, broken dependency): ``scripts/claude_hook.py`` uses it to answer, record ``hook_health`` and honour
user-selected high-risk contracts after an import failure. It imports only the standard library and the equally
dependency-free ``agy_hook_state``, whose ``HookState`` (continue budget, ``hook_health``, pending warnings) is
the shared store. Claude Code uses its own database file with the same schema, so its failures, banner and
continue budget never mix with AGY's (both hosts name events such as ``PreToolUse``).

Claude Code response shapes (command hooks, exit code 0 with JSON on stdout):

* PreToolUse: ``hookSpecificOutput.permissionDecision`` ``deny`` / ``ask`` only. A hook ``allow`` would approve the
  call without Claude Code's own permission prompt, which a clean verdict must never do ("clean" is not
  permission), so "allow" is the absence of a decision, optionally with a ``systemMessage`` for the user.
* UserPromptSubmit / SessionStart / PostToolUse / PostToolUseFailure: ``hookSpecificOutput.additionalContext``.
* Stop / SubagentStop: ``{"decision": "block", "reason": ...}`` restarts the agent; omitting it allows the stop.
* Exit code 2 is never used: an internal error must not block anything.
"""
from __future__ import annotations

import os
import sys

from self_directing_mcp.agy_hook_state import (  # noqa: F401  (re-exported for the fallback script)
    AUDIT_UNAVAILABLE, CHECK_FAILED, DEFAULT_MODE, MODES, HookState, high_risk_match, resolve_contracts_path,
    resolve_index_dir, warning_text,
)

STATE_FILE = "claude-hook-state.sqlite"
ENV_MODE = "SELF_DIRECT_CLAUDE_ENFORCEMENT"
EVENTS = ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PostToolUseFailure",
          "Stop", "SubagentStop")
CONTEXT_EVENTS = ("SessionStart", "UserPromptSubmit")
STOP_EVENTS = ("Stop", "SubagentStop")


def resolve_mode(environ=None) -> str:
    value = (environ if environ is not None else os.environ).get(ENV_MODE)
    return value if value in MODES else DEFAULT_MODE


class ClaudeHookState(HookState):
    def __init__(self, index_dir) -> None:
        super().__init__(index_dir, STATE_FILE)


def _context(event: str, text: str, *, notice: str | None = None) -> dict:
    response = {"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}
    if notice:
        response["systemMessage"] = notice
    return response


def fail_open_response(event, reason: str, *, ask: str | None = None, banner: str | None = None) -> dict:
    """Never deny, never block. PreToolUse asks only when a user-selected high-risk contract applies."""
    if event in CONTEXT_EVENTS:
        text = reason + " Contract context was not injected; do not claim compliance."
        return _context(event, text + (("\n" + banner) if banner else ""), notice=reason)
    if event == "PreToolUse" and ask:
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "ask",
                                       "permissionDecisionReason": reason + " " + ask}}
    return {"systemMessage": reason}


def fail_open(event, payload, error_class: str, kind: str = CHECK_FAILED, *, mode: str | None = None,
              index_dir=None, contracts_path=None) -> dict:
    """The response for any internal hook error, after recording it. Stores and prints the class name only."""
    mode = mode if mode in MODES else resolve_mode()
    reason = warning_text(event, error_class, kind)
    recorded, banner, ask = False, None, None
    try:
        health = ClaudeHookState(index_dir if index_dir is not None else resolve_index_dir())
        recorded = health.record_error(kind, event or "unknown", error_class)
        if event in CONTEXT_EVENTS:
            banner = health.banner()
    except Exception:
        pass
    if event == "PreToolUse" and mode != "advisory" and isinstance(payload, dict):
        try:
            matched = high_risk_match(contracts_path if contracts_path is not None else resolve_contracts_path(),
                                      payload.get("session_id"),
                                      {"name": payload.get("tool_name"), "args": payload.get("tool_input")},
                                      provider="claude")
        except Exception:
            matched = None
        if matched:
            ask = f"High-risk contract [{matched}] applies to this action and could not be verified; confirm manually."
    print(reason + ("" if recorded else " (hook health not recorded)"), file=sys.stderr)
    return fail_open_response(event, reason, ask=ask, banner=banner)
