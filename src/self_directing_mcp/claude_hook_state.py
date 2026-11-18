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
    AUDIT_UNAVAILABLE, CHECK_FAILED, DEFAULT_MODE, MODES, HookState, high_risk_hit, high_risk_match, high_risk_rule,
    resolve_contracts_path, resolve_index_dir, warning_text,
)

STATE_FILE = "claude-hook-state.sqlite"
ENV_MODE = "SELF_DIRECT_CLAUDE_ENFORCEMENT"
# Claude Code also has ``agent_review`` (``settings_core.CLAUDE_ENFORCEMENT_MODES``). After an import failure no review
# can be checked, so the fallback treats it as ``confirmed_only``: it refuses or asks for a high-risk contract's trigger.
AGENT_REVIEW = "agent_review"
CLAUDE_MODES = MODES + (AGENT_REVIEW,)
# The kind of the ``hook_health`` row that counts the calls refused to the agent because the audit could not run.
AGENT_RETRY_REFUSED = "agent_retry_refused"
EVENTS = ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PostToolUseFailure",
          "Stop", "SubagentStop")
CONTEXT_EVENTS = ("SessionStart", "UserPromptSubmit")
STOP_EVENTS = ("Stop", "SubagentStop")


def resolve_mode(environ=None) -> str:
    value = (environ if environ is not None else os.environ).get(ENV_MODE)
    return value if value in CLAUDE_MODES else DEFAULT_MODE


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


# What the agent reads when a call is refused because the audit could not run (``agent_review``, transient cause): it is a
# refusal for the agent to handle, not a question for the user and not a finding, so there is nothing to review.
AGENT_RETRY_TEXT = (
    "Tsukkomi audit could not run ({error_class}), so nothing was verified for this call: high-risk contract [{contract}] "
    "asks for a prerequisite of this action and, without the audit, it is neither satisfied nor violated. This call is "
    "refused so that you, the agent, handle it; nobody else is asked, and there is no finding to review, so no "
    "review_action call applies. The cause is usually transient (the index is busy, the daemon is restarting or slow): "
    "wait briefly and retry the identical call. If it is refused again, look at the daemon's health with the "
    "audit_status tool of the server that holds these contracts (it answers while the index is busy; see its ops) before "
    "anything else, and do not run the action by another route.")


def fail_open_response(event, reason: str, *, refuse: str | None = None, ask: str | None = None,
                       banner: str | None = None, quiet: bool = False, retry: str | None = None) -> dict:
    """Never block a stop and never deny by accident. PreToolUse is refused only when ``refuse`` says that the trigger of
    a user-selected high-risk ``must_not`` contract matches this very action and could not be checked; ``ask`` (a ``must``
    prerequisite that history may already satisfy) asks the user instead, unless ``retry`` (``agent_review`` and a transient
    cause) refuses it to the agent with that text, which replaces ``reason``.

    ``quiet``: the same degraded-hook notice was already shown in this session; a refusal is still said every time."""
    if event in CONTEXT_EVENTS:
        text = reason + " Contract context was not injected; do not claim compliance."
        return _context(event, text + (("\n" + banner) if banner else ""), notice=reason)
    if event == "PreToolUse" and retry:
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                       "permissionDecisionReason": retry}}
    if event == "PreToolUse" and (refuse or ask):
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny" if refuse else "ask",
                                       "permissionDecisionReason": reason + " " + (refuse or ask)}}
    return {} if quiet else {"systemMessage": reason}


def gating_rule(event, payload, *, mode: str | None = None, contracts_path=None) -> tuple[str, str, str] | None:
    """``(id, type, on_unverified)`` of the enabled ``risk: high`` contract that gates this PreToolUse call, or None.
    Nothing gates a call in ``advisory`` mode, and an unreadable contracts file gates nothing (it is unknown, never a guess)."""
    mode = mode if mode in CLAUDE_MODES else resolve_mode()
    if event != "PreToolUse" or mode == "advisory" or not isinstance(payload, dict):
        return None
    try:
        return high_risk_rule(contracts_path if contracts_path is not None else resolve_contracts_path(),
                              payload.get("session_id"),
                              {"name": payload.get("tool_name"), "args": payload.get("tool_input")}, provider="claude")
    except Exception:
        return None


def fail_open(event, payload, error_class: str, kind: str = CHECK_FAILED, *, mode: str | None = None,
              index_dir=None, contracts_path=None, transient: bool = False) -> dict:
    """The response for any internal hook error, after recording it. Stores and prints the class name only.

    An action that matches the trigger of an enabled ``risk: high`` contract is refused (the audit that would have
    judged it did not run, and a safety hook must not wave through what its contract is about to gate); every other
    action passes with a notice. The notice is said once per session and failure kind, because the same degraded
    state repeated on every tool call tells nothing new; the health banner and ``hook_health`` keep counting.

    A ``must`` prerequisite is asked of the user, with one exception: in ``agent_review`` mode, when the failure is
    ``transient`` (a condition of the tool itself that a retry may clear: the caller says so from the typed error or the
    daemon's reason) and the contract leaves unverified calls to the agent (``on_unverified``, the default), the call is
    refused to the agent with ``AGENT_RETRY_TEXT`` and counted under ``AGENT_RETRY_REFUSED`` in ``hook_health``, beside
    the failure itself. A ``must_not`` trigger is refused as before, in every mode but ``advisory``."""
    mode = mode if mode in CLAUDE_MODES else resolve_mode()
    reason = warning_text(event, error_class, kind)
    rule = gating_rule(event, payload, mode=mode, contracts_path=contracts_path)
    refuse = ask = retry = None
    if rule and rule[1] != "must":
        refuse = (f"Refused: high-risk contract [{rule[0]}] forbids what this action matches and could not be verified. "
                  "Fix the audit (see the error class above) or confirm the action with the user and retry.")
    elif rule and mode == AGENT_REVIEW and transient and rule[2] == "agent":
        retry = AGENT_RETRY_TEXT.format(error_class=error_class, contract=rule[0])
    elif rule:
        ask = (f"High-risk contract [{rule[0]}] asks for a prerequisite of this action that could not be checked; "
               "confirm manually.")
    recorded, banner, quiet = False, None, False
    session = payload.get("session_id") if isinstance(payload, dict) else None
    try:
        health = ClaudeHookState(index_dir if index_dir is not None else resolve_index_dir())
        recorded = health.record_error(kind, event or "unknown", error_class)
        if retry:
            health.record_error(AGENT_RETRY_REFUSED, event, error_class)      # the agent is told directly: no notice
        elif event in CONTEXT_EVENTS:
            banner = health.banner()
        elif event in _REPEATING_EVENTS and isinstance(session, str) and session:
            first = health.claim_notice(session, f"{NOTICE_DEGRADED}:{event}", f"{kind}:{error_class}")
            quiet = not first
            if first and event != "Stop" and event != "SubagentStop":
                # The model never sees a hook's systemMessage; the next PostToolUse / UserPromptSubmit carries this.
                health.queue_warning(session, reason)
    except Exception:
        pass
    print(reason + ("" if recorded else " (hook health not recorded)"), file=sys.stderr)
    return fail_open_response(event, reason, refuse=refuse, ask=ask, banner=banner, quiet=quiet, retry=retry)
