"""Standard-library state and fail-open responses for the Grok CLI hook adapter.

Like ``claude_hook_state`` this module must stay importable when the rest of the runtime is not (missing native
extension, broken dependency): ``scripts/grok_hook.py`` uses it to answer, record ``hook_health`` and honour
user-selected high-risk contracts after an import failure. It imports only the standard library and the equally
dependency-free ``agy_hook_state``, whose ``HookState`` (continue budget, ``hook_health``) is the shared store. Grok
uses its own database file with the same schema, so its failures, banner and continue budget never mix with another
host's.

Grok reply shapes (command hooks, ``docs/10-hooks.md`` of its user guide), all JSON on stdout:

* PreToolUse: ``{"decision": "allow" | "deny" | "ask", "reason": ...}``. The word ``allow`` only means "not blocked":
  it never approves a call Grok would otherwise ask the user about, so unlike Claude Code's hooks this adapter always
  answers explicitly and never prints ``{}``. ``hookSpecificOutput.additionalContext`` tells the model something.
* PostToolUse / PostToolUseFailure: ``hookSpecificOutput.additionalContext`` (a note for the model).
* Stop: ``{"decision": "block", "reason": ...}`` keeps the agent working; anything else lets it stop.
* SessionStart and UserPromptSubmit: Grok reads only a top-level ``systemMessage`` from an observe-only event and
  discards the stdout of an allowing UserPromptSubmit, so neither can inject contract context. Nothing is injected.
* A failure is visible and still fail-open: the reply allows, the first line of stderr names the failure, and the
  exit code is 1 (Grok records "exit code 1: <first stderr line>" in the scrollback and blocks nothing). The one
  exception is a high-risk ask, which exits 0 so the prompt carries the reason. Exit code 2 is never used: it denies.
"""
from __future__ import annotations

import os
import sys

from self_directing_mcp.agy_hook_state import (  # noqa: F401  (re-exported for the fallback script)
    AUDIT_UNAVAILABLE, CHECK_FAILED, DEFAULT_MODE, MODES, HookState, high_risk_match, resolve_contracts_path,
    resolve_index_dir, warning_text,
)

STATE_FILE = "grok-hook-state.sqlite"
ENV_MODE = "SELF_DIRECT_GROK_ENFORCEMENT"
EVENTS = ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PostToolUseFailure", "Stop")
PATH_EVENTS = ("PreToolUse", "Stop")             # events that read the transcript
LIGHT_EVENTS = ("SessionStart", "UserPromptSubmit", "PostToolUse", "PostToolUseFailure")   # bookkeeping only; no engine
# The event name in Grok's own spelling (``hookEventName``), which a payload carries beside Claude's (``hook_event_name``).
NATIVE_EVENT = {"SessionStart": "session_start", "UserPromptSubmit": "user_prompt_submit", "PreToolUse": "pre_tool_use",
                "PostToolUse": "post_tool_use", "PostToolUseFailure": "post_tool_use_failure", "Stop": "stop"}
FAILURE_EXIT = 1


def resolve_mode(environ=None) -> str:
    value = (environ if environ is not None else os.environ).get(ENV_MODE)
    return value if value in MODES else DEFAULT_MODE


class GrokHookState(HookState):
    def __init__(self, index_dir) -> None:
        super().__init__(index_dir, STATE_FILE)


def _tool_context(event: str, text: str) -> dict:
    return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}


def fail_open_response(event, reason: str, *, ask: str | None = None, banner: str | None = None) -> dict:
    """Never deny, never block. PreToolUse asks only when a user-selected high-risk contract applies."""
    if event == "PreToolUse":
        if ask:
            return {"decision": "ask", "reason": reason + " " + ask}
        return {"decision": "allow", "systemMessage": reason,
                **_tool_context(event, reason + " Do not claim compliance for this call.")}
    if event == "SessionStart":
        return {"systemMessage": reason + (("\n" + banner) if banner else "")}
    return {"systemMessage": reason}


def fail_open(event, payload, error_class: str, kind: str = CHECK_FAILED, *, mode: str | None = None,
              index_dir=None, contracts_path=None) -> dict:
    """The response for any internal hook error, after recording it. Stores and prints the class name only."""
    mode = mode if mode in MODES else resolve_mode()
    reason = warning_text(event, error_class, kind)
    recorded, banner, ask = False, None, None
    try:
        health = GrokHookState(index_dir if index_dir is not None else resolve_index_dir())
        recorded = health.record_error(kind, event or "unknown", error_class)
        if event == "SessionStart":
            banner = health.banner()
    except Exception:
        pass
    if event == "PreToolUse" and mode != "advisory" and isinstance(payload, dict):
        try:
            name = payload.get("tool_name", payload.get("toolName"))
            args = payload.get("tool_input", payload.get("toolInput"))
            matched = high_risk_match(contracts_path if contracts_path is not None else resolve_contracts_path(),
                                      payload.get("session_id", payload.get("sessionId")),
                                      {"name": name, "args": args}, provider="grok")
        except Exception:
            matched = None
        if matched:
            ask = f"High-risk contract [{matched}] applies to this action and could not be verified; confirm manually."
    print(reason + ("" if recorded else " (hook health not recorded)"), file=sys.stderr)
    return fail_open_response(event, reason, ask=ask, banner=banner)


def exit_code(response: dict) -> int:
    """0 for a judgement, :data:`FAILURE_EXIT` for a fail-open reply (so Grok shows the failure), except an ``ask``."""
    return 0 if response.get("decision") == "ask" else FAILURE_EXIT
