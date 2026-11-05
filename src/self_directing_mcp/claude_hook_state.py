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


NOTICE_DEGRADED = "hook_degraded"        # the kind of the one-per-session degraded-hook notice (``claim_notice``)
_REPEATING_EVENTS = ("PreToolUse", "PostToolUse", "PostToolUseFailure", "Stop", "SubagentStop")


def fail_open_response(event, reason: str, *, refuse: str | None = None, banner: str | None = None,
                       quiet: bool = False) -> dict:
    """Never block a stop and never deny by accident. PreToolUse is refused only when ``refuse`` says that a
    user-selected high-risk contract applies to this very action and could not be checked.

    ``quiet``: the same degraded-hook notice was already shown in this session; a refusal is still said every time."""
    if event in CONTEXT_EVENTS:
        text = reason + " Contract context was not injected; do not claim compliance."
        return _context(event, text + (("\n" + banner) if banner else ""), notice=reason)
    if event == "PreToolUse" and refuse:
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                       "permissionDecisionReason": reason + " " + refuse}}
    return {} if quiet else {"systemMessage": reason}


def fail_open(event, payload, error_class: str, kind: str = CHECK_FAILED, *, mode: str | None = None,
              index_dir=None, contracts_path=None) -> dict:
    """The response for any internal hook error, after recording it. Stores and prints the class name only.

    An action that matches the trigger of an enabled ``risk: high`` contract is refused (the audit that would have
    judged it did not run, and a safety hook must not wave through what its contract is about to gate); every other
    action passes with a notice. The notice is said once per session and failure kind, because the same degraded
    state repeated on every tool call tells nothing new; the health banner and ``hook_health`` keep counting."""
    mode = mode if mode in MODES else resolve_mode()
    reason = warning_text(event, error_class, kind)
    recorded, banner, quiet = False, None, False
    session = payload.get("session_id") if isinstance(payload, dict) else None
    try:
        health = ClaudeHookState(index_dir if index_dir is not None else resolve_index_dir())
        recorded = health.record_error(kind, event or "unknown", error_class)
        if event in CONTEXT_EVENTS:
            banner = health.banner()
        elif event in _REPEATING_EVENTS and isinstance(session, str) and session:
            first = health.claim_notice(session, f"{NOTICE_DEGRADED}:{event}", f"{kind}:{error_class}")
            quiet = not first
            if first and event != "Stop" and event != "SubagentStop":
                # The model never sees a hook's systemMessage; the next PostToolUse / UserPromptSubmit carries this.
                health.queue_warning(session, reason)
    except Exception:
        pass
    refuse = None
    if event == "PreToolUse" and mode != "advisory" and isinstance(payload, dict):
        try:
            matched = high_risk_match(contracts_path if contracts_path is not None else resolve_contracts_path(),
                                      payload.get("session_id"),
                                      {"name": payload.get("tool_name"), "args": payload.get("tool_input")},
                                      provider="claude")
        except Exception:
            matched = None
        if matched:
            refuse = (f"Refused: high-risk contract [{matched}] applies to this action and could not be verified. "
                      "Fix the audit (see the error class above) or confirm the action with the owner and retry.")
    print(reason + ("" if recorded else " (hook health not recorded)"), file=sys.stderr)
    return fail_open_response(event, reason, refuse=refuse, banner=banner, quiet=quiet)
