"""Grok CLI lifecycle controls (command hooks). Importing this module does not load native code.

One judgement, host-specific enforcement: the verdict logic is ``hook_policy`` (shared with the AGY and Claude Code
adapters); this module only reads Grok's hook payloads and writes its replies. Default ``confirmed_only``
(``SELF_DIRECT_GROK_ENFORCEMENT``, or ``--mode=`` on the registered command):

* PreToolUse: a confirmed ``violation`` denies; ``suspicious`` asks; ``unknown`` allows with a warning that tells
  the model the call was not verified, unless a contract the user marked ``risk: high`` applies (then ask); clean and
  not-applicable allow silently. Every reply is explicit (``{"decision": "allow"}`` at least): Grok documents that an
  allow only means "not blocked" and never skips its own permission prompt. ``advisory`` never denies or asks;
  ``enforced`` also denies ``unknown``. The call is judged as the Claude-canonical call (``grok_tools``), so the
  contracts users wrote for other hosts apply; the deny names the Grok tool that was matched.
* Stop: block (keep the agent working) only for a confirmed, fully observed miss of a contract marked
  ``auto_continue``, at most once per session, never for the same reason twice, never when the persisted counter is
  unreadable, never while ``stopHookActive`` and never when the fire is not a turn end (``reason`` is not
  ``end_turn``: Grok also fires Stop at session shutdown, where a decision is ignored).
* SessionStart / UserPromptSubmit / PostToolUse / PostToolUseFailure: bookkeeping only. Grok discards the stdout of
  an allowing UserPromptSubmit and reads only ``systemMessage`` from the observe-only SessionStart, so no contract
  context is injected anywhere; the only output is the health banner (``systemMessage`` at SessionStart).
* Internal errors fail open with a visible warning in every mode and are recorded in ``hook_health``; Stop always
  allows. Exit code 2 is never used.

No input rewrite (``updatedInput``) and no output redaction (``updatedToolOutput``): see the README.

Every reply passes through ``mask_value``. The payload's prompt and tool output are untrusted data and are never read.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import sys
from typing import Any, TYPE_CHECKING

from self_directing_mcp import grok_hook_state as hook_state
from self_directing_mcp import grok_ids, grok_tools, hook_policy
from self_directing_mcp.agy_hook_state import AUDIT_UNAVAILABLE, CHECK_FAILED
from self_directing_mcp.grok_hook_state import EVENTS, LIGHT_EVENTS, NATIVE_EVENT, GrokHookState
from self_directing_mcp.hook_policy import HookRuntimeError
from self_directing_mcp.security.mask import mask_value

if TYPE_CHECKING:
    from self_directing_mcp.engine import SelfDirectEngine

PROVIDER = "grok"
BUDGET_NOTE = hook_policy.budget_note("session")
TRANSCRIPT_LAGS = ("Grok's transcript does not contain the final assistant message yet; "
                   "message obligations cannot be verified (does not block stopping).")
SUBAGENT_HISTORY = ("A subagent's own history is not consulted for this call; "
                    "history-dependent obligations stay unverified.")
REWRITTEN = ("The Grok transcript of this session shrank (a rewind, reset or compaction rewrote its history): what it no "
             "longer holds cannot be judged, so history-dependent contracts are unverified and the absence of an old "
             "call is not compliance.")
_FINAL_MESSAGE_SCOPES = ("message", "any")     # what the unflushed final assistant text could still satisfy
TURN_END = "end_turn"
_TAIL_BYTES = 262144
_CLIPPED = re.compile(r"\s*\u2026 \[\+\d+ chars\]\s*\Z")        # "... [+N chars]", how Grok marks a clipped free-text field


@dataclass(frozen=True)
class HookCall:
    event: str
    session_id: str
    transcript_path: str | None
    tool_name: str | None = None
    tool_input: Any = None
    stop_hook_active: bool = True     # only an explicit false (or absence) permits a continue
    stop_reason: str | None = None    # Stop only: end_turn (a turn end), shutdown / channel_closed (session end)
    last_message: str | None = None   # the host's own copy of the final assistant text (Stop)
    subagent_type: str | None = None  # set when the event fired inside a subagent's session


def failure_response(event: str | None, payload: Any = None, *, reason: str = "Hook verification unavailable") -> dict:
    """Fail-open response for an event; ``payload`` is accepted for symmetry and never inspected."""
    return hook_state.fail_open_response(event, reason)


def _mode(source: Any) -> str:
    """``source`` is an engine or its settings."""
    settings = getattr(source, "settings", source)
    return settings.grok_enforcement


# --- payload ----------------------------------------------------------------------------------------------

def _pair(payload: dict, snake: str, camel: str) -> Any:
    """A field Grok sends under both spellings. The two must agree; a payload that contradicts itself is invalid."""
    first, second = payload.get(snake), payload.get(camel)
    if first is not None and second is not None and first != second:
        raise ValueError(f"{snake} and {camel} disagree")
    return first if first is not None else second


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _validate_payload(payload: Any, event: str | None) -> HookCall:
    """Validate only what the event uses; every other field is tolerated and never trusted."""
    if event not in EVENTS or not isinstance(payload, dict):
        raise ValueError("unsupported hook event or payload")
    declared = payload.get("hook_event_name")
    if declared is not None and declared != event:
        raise ValueError("hook_event_name does not match the registered event")
    native = payload.get("hookEventName")
    if native is not None and native != NATIVE_EVENT[event]:
        raise ValueError("hookEventName does not match the registered event")
    session_id = _pair(payload, "session_id", "sessionId")
    if not grok_ids.valid_id(session_id):
        raise ValueError("session_id must be a Grok session UUID")
    path = _text(_pair(payload, "transcript_path", "transcriptPath"))
    if event == "PreToolUse" and path is None:
        raise ValueError("transcript_path is required")
    name = arguments = None
    if event == "PreToolUse":
        name, arguments = _pair(payload, "tool_name", "toolName"), _pair(payload, "tool_input", "toolInput")
        if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
            raise ValueError("tool_name and an object tool_input are required")
    # An absent flag means "not continuing from a Stop hook"; anything other than a plain false counts as active.
    flags = [payload[key] for key in ("stopHookActive", "stop_hook_active") if key in payload]
    return HookCall(event, session_id, path, name, arguments, any(flag is not False for flag in flags),
                    _text(payload.get("reason")), _text(_pair(payload, "last_assistant_message", "lastAssistantMessage")),
                    _text(payload.get("subagentType")))


def _validated_path(engine: SelfDirectEngine, session_id: str, path: str) -> str:
    from self_directing_mcp import grok
    return str(grok.resolve_session_path(engine.settings.resolve_grok_sessions_dir(),
                                         session_id=session_id, path=path, require_exists=False))


def _assistant_text(path: str) -> str | None:
    """The concatenated agent message text in the tail of ``updates.jsonl``; None when the file cannot be read."""
    try:
        with open(path, "rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - _TAIL_BYTES))
            tail = stream.read().decode("utf-8", "replace")
    except OSError:
        return None
    parts = []
    for line in tail.splitlines():
        try:
            update = json.loads(line)["params"]["update"]
            if update["sessionUpdate"] == "agent_message_chunk":
                parts.append(update["content"]["text"])
        except (ValueError, KeyError, TypeError):
            continue          # a cut first line, a record of another shape: not part of the message
    return "".join(parts)


def _transcript_lags(path: str, last_message: str | None) -> bool:
    """True when the host reports a final assistant message that the transcript file does not contain yet.

    Grok runs a Stop hook before the final records are written (the recorded streams show the hook's own report
    ahead of the last message chunk). An audit of that file would call a satisfied obligation unmet, and a block must
    rest on observed evidence only. The message arrives as streamed deltas, so the comparison joins the agent
    message records of the file's tail instead of searching for one string. Without the host's copy of the message
    nothing can be compared and the audit stands on its own coverage checks.
    """
    message = _CLIPPED.sub("", last_message or "").strip()      # Grok clips a long message and marks the cut
    if not message:
        return False
    seen = _assistant_text(path)
    if seen is None:
        return False                          # the engine reports an unreadable transcript as incomplete coverage
    return message not in seen


# --- handlers (host-neutral decisions; ``render`` produces Grok's JSON) ---------------------------------------

def _rewritten(engine: SelfDirectEngine, call: HookCall, path: str) -> dict | None:
    """Whether this session's ``updates.jsonl`` was ever seen to shrink (see ``GrokHookState.observe_transcript``)."""
    try:
        size = os.stat(path).st_size
    except OSError:
        return None                      # the engine reports an unreadable transcript as incomplete coverage
    return GrokHookState(engine.settings.index_dir).observe_transcript(call.session_id, size)


def _native_hint(call: HookCall) -> str:
    canonical = grok_tools.canonical_call(call.tool_name, call.tool_input)
    return " (" + grok_tools.describe(call.tool_name, canonical.name) + ")" if canonical.mapped else ""


def handle_pre_tool_use(engine: SelfDirectEngine, call: HookCall, path: str | None = None) -> dict[str, Any]:
    # No read-command classifier or tool-name exemption: reads can be restricted, shell flags can mutate,
    # and names containing this server are not authority. The engine renders the call in Claude's vocabulary
    # (``tool_payload``), exactly as the parser does for the history.
    from self_directing_mcp.workflow import guard_host_action
    canonical = grok_tools.canonical_call(call.tool_name, call.tool_input)
    workflow = guard_host_action(call.session_id, {"tool_name": canonical.name, "arguments": canonical.arguments}, PROVIDER)
    if workflow is not None:
        if _mode(engine) == "advisory":      # advisory never denies, whatever the source of the finding
            return {"decision": "allow", "notice": "ADVISORY (not compliance): " + workflow["reason"]}
        return {"decision": workflow["decision"], "reason": workflow["reason"]}
    rewritten = _rewritten(engine, call, path) is not None if path else False
    adjust = None
    if rewritten or call.subagent_type is not None:
        def adjust(audit, obligations):
            # A confirmed miss of a history-dependent contract rests on evidence that is gone (or never seen): unknown.
            if rewritten:
                audit = hook_policy.demote_history_findings(audit, obligations, REWRITTEN)
            if call.subagent_type is not None:
                audit = hook_policy.demote_history_findings(audit, obligations, SUBAGENT_HISTORY)
            return audit
    judged = hook_policy.judge_action(engine, PROVIDER, call.session_id, path, call.tool_name, call.tool_input,
                                      _mode(engine), adjust=adjust)
    if judged is None:
        return {"decision": "allow"}
    if judged.silent:
        # No contract objected, but on a rewritten history that is not a compliance verdict: say so to the model.
        return {"decision": "allow", "context": REWRITTEN} if rewritten and _mode(engine) != "advisory" else {"decision": "allow"}
    reason = judged.reason + _native_hint(call)
    if judged.warn:
        # The model is told right away (PreToolUse additionalContext); the user has no channel on an allow.
        return {"decision": "allow", "context": "Unverified tool call allowed: " + reason}
    if judged.decision == "allow":
        return {"decision": "allow", "notice": reason}          # advisory: reported, never delivered to the model
    return {"decision": judged.decision, "reason": reason}


def _claim(engine: SelfDirectEngine, call: HookCall, key: str) -> bool:
    return GrokHookState(engine.settings.index_dir).claim_continue(call.session_id, key)


def handle_stop(engine: SelfDirectEngine, call: HookCall, path: str) -> dict[str, Any]:
    if call.stop_hook_active:
        return {"decision": "allow"}            # already continuing because of a Stop hook: never again
    sid = call.session_id
    mode = _mode(engine)
    from self_directing_mcp.workflow import configured_workflow
    workflow = configured_workflow()
    if workflow is not None:
        state = workflow.status(sid, PROVIDER)
        if state["stage"] != "complete":
            details = {"stage": state["stage"], "requirements": state["requirements"],
                       "evidence": state["evidence"], "problems": state["problems"],
                       "allowed_next_actions": ["revise_implementation", "workflow_run_checks", "workflow_classify"]}
            note = "Workflow completion is unverified: " + json.dumps(details)
            # An owner-approved workflow is a user-specified completion obligation, but the cap still applies.
            if mode != "advisory" and _claim(engine, call, hook_policy.reason_key([f"workflow:{state['stage']}"])):
                return {"decision": "continue", "reason": note}
            return {"decision": "allow", "notice": note + ("" if mode == "advisory" else " " + BUDGET_NOTE)}
    obligations = hook_policy.read_obligations(engine, PROVIDER, sid, path)
    if not obligations["contracts"] and not obligations["has_checklist"]:
        return {"decision": "allow"}
    notes, misses = hook_policy.checkpoint(engine, PROVIDER, sid, path, obligations, "Grok")
    if _rewritten(engine, call, path) is not None:
        # Evidence that is gone confirms nothing: no continuation, and a clean checkpoint is not compliance.
        misses = []
        notes.append(REWRITTEN)
    if any(rule.scope in _FINAL_MESSAGE_SCOPES for rule, _f in misses) and _transcript_lags(path, call.last_message):
        # Earlier tool calls and results are on disk; only the final text can still satisfy these obligations.
        misses = [(rule, f) for rule, f in misses if rule.scope not in _FINAL_MESSAGE_SCOPES]
        notes.append(TRANSCRIPT_LAGS)
    if mode == "advisory":
        return ({"decision": "allow", "notice": hook_policy.with_notes(notes, "ADVISORY (not compliance): ")}
                if notes else {"decision": "allow"})
    if misses:
        key = hook_policy.miss_key(misses)
        if _claim(engine, call, key):
            return {"decision": "continue", "reason":
                    "A user-specified completion obligation is confirmed unmet: " + hook_policy.miss_summary(misses)
                    + ". This is the only automatic continuation for this session."}
        notes.append(BUDGET_NOTE)
    if notes:
        return {"decision": "allow", "notice": hook_policy.with_notes(notes, "Not verified (does not block stopping): ")}
    return {"decision": "allow"}


def handle_light(settings: Any, call: HookCall) -> dict[str, Any]:
    """SessionStart / UserPromptSubmit / PostToolUse / PostToolUseFailure / a Stop that is not a turn end.

    Nothing in the payload is read: the prompt and tool responses are untrusted data, a failed tool result is never
    success evidence because history comes only from the transcript's linked records, and Grok would discard any
    context anyway. SessionStart shows the health banner (``systemMessage``, the one field Grok reads there).
    """
    if call.event == "SessionStart":
        banner = GrokHookState(settings.index_dir).banner()
        if banner:
            return {"decision": "allow", "notice": banner}
    return {"decision": "allow"}


# --- Grok replies -------------------------------------------------------------------------------------------

def render(event: str | None, reply: dict[str, Any]) -> dict[str, Any]:
    """Translate a host-neutral reply (``decision`` / ``reason`` / ``context`` / ``notice``) into Grok's JSON."""
    decision, reason = reply.get("decision", "allow"), reply.get("reason")
    out: dict[str, Any] = {}
    if event == "PreToolUse":
        if decision == "deny":
            out = {"decision": "deny", "reason": reason or "A tsukkomi contract applies to this action."}
        elif decision in ("ask", "force_ask"):
            out = {"decision": "ask", "reason": reason or "A tsukkomi contract applies to this action."}
        else:
            out = {"decision": "allow"}          # explicit, never `{}`: an allow only means "not blocked"
            if reply.get("context"):
                out["hookSpecificOutput"] = {"hookEventName": event, "additionalContext": reply["context"]}
    elif event == "Stop" and decision == "continue":
        out = {"decision": "block", "reason": reason or "A confirmed completion obligation is unmet."}
    if reply.get("notice") and decision not in ("deny", "ask", "force_ask", "continue"):
        out["systemMessage"] = reply["notice"]
    return out


# --- dispatch ----------------------------------------------------------------------------------------------

def _new_settings():
    from self_directing_mcp.settings_core import load_hook_settings
    try:
        return load_hook_settings()
    except Exception as exc:
        raise HookRuntimeError("settings unavailable") from exc


def _new_engine():
    from self_directing_mcp.engine import SelfDirectEngine
    from self_directing_mcp.settings_core import load_hook_settings
    try:
        return SelfDirectEngine(load_hook_settings())
    except Exception as exc:
        raise HookRuntimeError("settings or engine unavailable") from exc


def _gates(call: HookCall) -> bool:
    """A Stop that may keep the agent working: a main-agent turn end. Shutdown fires are record-only."""
    return call.stop_reason == TURN_END and call.subagent_type is None


def _dispatch(payload: Any, event: str | None, engine: SelfDirectEngine | None, context: dict[str, Any]) -> dict[str, Any]:
    """The host-neutral reply; ``context`` receives settings-derived paths as soon as known."""
    call = _validate_payload(payload, event)
    if event in LIGHT_EVENTS or (event == "Stop" and not _gates(call)):
        settings = engine.settings if engine is not None else _new_settings()
        context.update(mode=settings.grok_enforcement, index_dir=settings.index_dir,
                       contracts_path=settings.resolve_contracts_path())
        return handle_light(settings, call)
    owned = engine is None
    engine = engine if engine is not None else _new_engine()
    try:
        context.update(mode=engine.settings.grok_enforcement, index_dir=engine.settings.index_dir,
                       contracts_path=engine.settings.resolve_contracts_path())
        if call.transcript_path is None:
            raise ValueError("transcript_path is required")
        path = _validated_path(engine, call.session_id, call.transcript_path)
        if event == "PreToolUse":
            return handle_pre_tool_use(engine, call, path)
        return handle_stop(engine, call, path)
    finally:
        if owned:
            engine.close()


def dispatch_grok_hook(payload: dict[str, Any], event_name: str | None = None) -> dict[str, Any]:
    """Grok's reply for a payload. Raises on invalid input or errors; ``run_hook`` never does."""
    return render(event_name, _dispatch(payload, event_name, None, {}))


def _run(event: str | None, payload: Any, engine: SelfDirectEngine | None = None) -> tuple[dict[str, Any], bool]:
    """``(masked Grok reply, failed)``: the reply for every input, never an exception."""
    context: dict[str, Any] = {}
    try:
        response = render(event, _dispatch(payload, event, engine, context))
    except Exception as exc:
        # Messages can contain transcript data or credentials; only the class name leaves this function.
        unavailable = isinstance(exc, HookRuntimeError)
        cause = exc.__cause__ if unavailable and exc.__cause__ is not None else exc
        return mask_value(hook_state.fail_open(
            event, payload, type(cause).__name__, AUDIT_UNAVAILABLE if unavailable else CHECK_FAILED,
            mode=context.get("mode"), index_dir=context.get("index_dir"), contracts_path=context.get("contracts_path"))), True
    if "index_dir" in context:
        GrokHookState(context["index_dir"]).record_success(event)
    return mask_value(response), False   # contract text, reasons and history excerpts may quote secrets


def run_hook(event: str | None, payload: Any, *, engine: SelfDirectEngine | None = None) -> dict[str, Any]:
    """The hook's whole contract: a masked Grok reply for every input, never an exception."""
    return _run(event, payload, engine)[0]


def exit_status(response: dict[str, Any], failed: bool) -> int:
    """0 for a judgement. A fail-open reply exits 1 so that Grok records the failure; a high-risk ask exits 0."""
    return hook_state.exit_code(response) if failed else 0


def _event_from_argv(argv: list[str]) -> str | None:
    return next((arg.split("=", 1)[1] if arg.startswith("--event=") else arg
                 for arg in argv if arg.startswith("--event=") or arg in EVENTS), None)


def _event_from_payload(payload: Any) -> str | None:
    name = payload.get("hook_event_name") if isinstance(payload, dict) else None
    return name if name in EVENTS else None


def apply_flags(argv: list[str]) -> str | None:
    """Apply the registered command's flags to the environment; returns an error class name for an invalid one."""
    import os
    if "--local-only" in argv:
        # Hooks never embed. This keeps the credential-free engine local without touching the MCP server's own env.
        os.environ["SELF_DIRECT_LOCAL_ONLY"] = "true"
    error = None
    for arg in argv:
        # Named at registration because hooks never read a project .env (see config.hook_settings).
        if arg.startswith("--index-dir=") and os.path.isabs(arg.split("=", 1)[1]):
            os.environ["SELF_DIRECT_INDEX_DIR"] = arg.split("=", 1)[1]
        elif arg.startswith("--mode="):
            mode = arg.split("=", 1)[1]
            if mode in hook_state.MODES:
                os.environ[hook_state.ENV_MODE] = mode
            else:
                error = "InvalidMode"
    return error


def main() -> int:
    argv = sys.argv[1:]
    flag_error = apply_flags(argv)
    event = _event_from_argv(argv)
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        # Only the error class is reported; the raw stdin payload never reaches diagnostics.
        response, failed = mask_value(hook_state.fail_open(event, None, type(exc).__name__, CHECK_FAILED)), True
    else:
        if flag_error is not None:
            response, failed = mask_value(hook_state.fail_open(
                event or _event_from_payload(payload), payload, flag_error, AUDIT_UNAVAILABLE)), True
        else:
            response, failed = _run(event or _event_from_payload(payload), payload)
    json.dump(mask_value(response), sys.stdout)
    sys.stdout.write("\n")
    return exit_status(response, failed)


if __name__ == "__main__":
    sys.exit(main())
