"""Claude Code lifecycle controls (command hooks). Importing this module does not load native code.

One judgement, host-specific enforcement: the verdict logic is ``hook_policy`` (shared with the AGY adapter);
this module only reads Claude Code's hook payloads and writes its responses.

Default ``confirmed_only`` (``SELF_DIRECT_CLAUDE_ENFORCEMENT``):

* PreToolUse: a confirmed ``violation`` denies; ``suspicious`` (concrete evidence, such as an exception that needs
  confirmation) asks; ``unknown`` passes through with a visible warning, unless a contract the user marked
  ``risk: high`` applies (then ask); clean and not-applicable pass through silently. "Pass through" means no
  permission decision at all: Claude Code's own permission flow still decides, because a clean verdict is not a
  permission grant. ``advisory`` never denies or asks; ``enforced`` also denies ``unknown``.
* Stop: block (continue the agent) only for a confirmed, fully observed miss of a contract marked
  ``auto_continue``, at most once per session, never for the same reason twice, never when the persisted counter
  is unreadable and never while ``stop_hook_active``. Claude Code runs Stop before the final assistant message is
  written to the transcript, so message-scoped misses wait for the host's copy of that message to appear there.
* SubagentStop: judges the subagent's own transcript and reports, but never blocks. A global obligation the user
  set for their session is not a task the subagent was given, so it cannot justify restarting the subagent.
* Internal errors fail open with a visible warning in every mode and are recorded in ``hook_health``; Stop always
  allows. Repeated failures show a banner at the next UserPromptSubmit or SessionStart.
* UserPromptSubmit also fires for scheduled, loop, subagent-report and peer messages, so it is never authority:
  the prompt is not read. Authority comes only from the transcript's recorded origin (the ``claude`` parser).

Every response passes through ``mask_value``. Exit code 2 is never used.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, TYPE_CHECKING

from self_directing_mcp import claude_hook_state as hook_state
from self_directing_mcp import hook_policy
from self_directing_mcp.agy_hook_state import AUDIT_UNAVAILABLE, CHECK_FAILED
from self_directing_mcp.claude_hook_state import CONTEXT_EVENTS, EVENTS, STOP_EVENTS, ClaudeHookState
from self_directing_mcp.hook_policy import HookRuntimeError
from self_directing_mcp.security.mask import mask_value

if TYPE_CHECKING:
    from self_directing_mcp.engine import SelfDirectEngine

PATH_EVENTS = ("PreToolUse", "Stop", "SubagentStop")        # events that read the transcript
LIGHT_EVENTS = ("UserPromptSubmit", "PostToolUse", "PostToolUseFailure")   # bookkeeping only; no engine
BUDGET_NOTE = hook_policy.budget_note("session")
SUBAGENT_UNKNOWN = ("The subagent's transcript is unavailable; its completion cannot be verified "
                    "(does not block stopping).")
TRANSCRIPT_LAGS = ("Claude Code's transcript does not contain the final assistant message yet; "
                   "message obligations cannot be verified (does not block stopping).")
SUBAGENT_NEVER_BLOCKS = "A subagent's stop is reported, never blocked."
_FINAL_MESSAGE_SCOPES = ("message", "any")     # what the unflushed final assistant text could still satisfy
SUBAGENT_HISTORY = ("A subagent's own history is not consulted for this call; "
                    "history-dependent obligations stay unverified.")


@dataclass(frozen=True)
class HookCall:
    event: str
    session_id: str
    transcript_path: str | None
    tool_name: str | None = None
    tool_input: Any = None
    agent_id: str | None = None
    agent_transcript_path: str | None = None
    stop_hook_active: bool = True     # only an explicit false (or absence) permits a continue
    last_message: str | None = None   # the host's own copy of the final assistant text (Stop / SubagentStop)


def failure_response(event: str | None, payload: Any = None, *, reason: str = "Hook verification unavailable") -> dict:
    """Fail-open response for an event; ``payload`` is accepted for symmetry and never inspected."""
    return hook_state.fail_open_response(event, reason)


def _mode(source: Any) -> str:
    """``source`` is an engine or its settings."""
    settings = getattr(source, "settings", source)
    return settings.claude_enforcement


# --- payload ----------------------------------------------------------------------------------------------

def _text(payload: dict, key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _validate_payload(payload: Any, event: str | None) -> HookCall:
    """Validate only what the event uses; every other field is tolerated and never trusted."""
    from self_directing_mcp import claude
    if event not in EVENTS or not isinstance(payload, dict):
        raise ValueError("unsupported hook event or payload")
    declared = payload.get("hook_event_name")
    if declared is not None and declared != event:
        raise ValueError("hook_event_name does not match the registered event")
    parts = claude.session_parts(payload.get("session_id"))
    if parts is None or parts[1] is not None:
        raise ValueError("session_id must be a Claude Code session UUID")
    path = _text(payload, "transcript_path")
    if event in PATH_EVENTS and path is None:
        raise ValueError("transcript_path is required")
    name = arguments = None
    if event == "PreToolUse":
        name, arguments = payload.get("tool_name"), payload.get("tool_input")
        if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
            raise ValueError("tool_name and an object tool_input are required")
    # An absent flag means "not continuing from a Stop hook"; anything other than a plain false counts as active.
    raw = payload.get("stop_hook_active", False)
    return HookCall(event, payload["session_id"], path, name, arguments,
                    _text(payload, "agent_id"), _text(payload, "agent_transcript_path"), raw is not False,
                    _text(payload, "last_assistant_message"))


def _validated_path(engine: SelfDirectEngine, session_id: str, path: str) -> str:
    from self_directing_mcp import claude
    return str(claude.resolve_session_path(engine.settings.resolve_claude_projects_dir(),
                                           session_id=session_id, path=path, require_exists=False))


def _subagent_target(engine: SelfDirectEngine, call: HookCall) -> tuple[str, str] | None:
    """``(subagent session id, transcript path)`` of the stopping subagent, or None when it cannot be located.

    A subagent file is its own session (``<parent uuid>_<stem>``): it never shares evidence with the parent, and
    contracts scoped to the parent session do not apply to it. A path that belongs to another session raises.
    """
    from self_directing_mcp import claude
    candidate = call.agent_transcript_path
    if candidate is None and call.agent_id is not None:
        candidate = str(Path(call.transcript_path).parent / call.session_id / claude.SUBAGENTS
                        / f"agent-{call.agent_id}.jsonl")
    if candidate is None:
        return None
    try:
        resolved = claude.resolve_session_path(engine.settings.resolve_claude_projects_dir(), path=candidate)
    except FileNotFoundError:
        return None
    agent_sid = claude.peek_session_id(resolved)
    parts = claude.session_parts(agent_sid)
    if parts is None or parts[1] is None or parts[0] != call.session_id:
        raise ValueError("subagent transcript does not belong to this session")
    return agent_sid, str(resolved)


def _transcript_lags(path: str, last_message: str | None) -> bool:
    """True when the host reports a final assistant message that the transcript file does not contain yet.

    Claude Code can run a Stop hook before the final record is flushed. An audit of that file would call a
    satisfied obligation unmet, and a block must rest on observed evidence only. Without the host's copy of the
    message (older versions) nothing can be compared and the audit stands on its own coverage checks.
    """
    if not last_message:
        return False
    snippet = last_message.strip()[:80]
    forms = {json.dumps(snippet, ensure_ascii=False)[1:-1], json.dumps(snippet)[1:-1]}
    try:
        with open(path, "rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - 262144))
            tail = stream.read().decode("utf-8", "replace")
    except OSError:
        return False                         # the engine reports an unreadable transcript as incomplete coverage
    return not any(form in tail for form in forms)


# --- handlers (host-neutral decisions; ``render`` produces Claude Code's JSON) --------------------------------

def handle_pre_tool_use(engine: SelfDirectEngine, call: HookCall, path: str | None = None) -> dict[str, Any]:
    # No read-command classifier or tool-name exemption: reads can be restricted, shell flags can mutate,
    # and names containing this server are not authority.
    from self_directing_mcp.workflow import guard_host_action
    workflow = guard_host_action(call.session_id, {"tool_name": call.tool_name, "arguments": call.tool_input}, "claude")
    if workflow is not None:
        if _mode(engine) == "advisory":      # advisory never denies, whatever the source of the finding
            return {"decision": "allow", "reason": "ADVISORY (not compliance): " + workflow["reason"]}
        return {"decision": workflow["decision"], "reason": workflow["reason"]}
    adjust = None
    if call.agent_id is not None:
        adjust = lambda audit, obligations: hook_policy.demote_history_findings(audit, obligations, SUBAGENT_HISTORY)
    judged = hook_policy.judge_action(engine, "claude", call.session_id, path, call.tool_name, call.tool_input,
                                      _mode(engine), adjust=adjust)
    if judged is None or judged.silent:
        return {"decision": "allow"}
    if judged.warn:
        # The model never sees an allow's reason; the next PostToolUse / UserPromptSubmit shows it.
        ClaudeHookState(engine.settings.index_dir).queue_warning(
            call.session_id, "Unverified tool call allowed: " + judged.reason)
    return {"decision": judged.decision, "reason": judged.reason}


def _claim(engine: SelfDirectEngine, call: HookCall, key: str) -> bool:
    # One budget for the whole Claude Code session (subagent stops never draw on it).
    return ClaudeHookState(engine.settings.index_dir).claim_continue(call.session_id, key)


def handle_stop(engine: SelfDirectEngine, call: HookCall, path: str, *, session_id: str | None = None) -> dict[str, Any]:
    """``session_id`` / ``path`` name the transcript judged: the main session, or a stopping subagent's own."""
    if call.stop_hook_active:
        return {"decision": "allow"}            # already continuing because of a Stop hook: never again
    sid = session_id or call.session_id
    subagent = sid != call.session_id
    mode = _mode(engine)
    if not subagent:
        from self_directing_mcp.workflow import configured_workflow
        workflow = configured_workflow()
        if workflow is not None:
            state = workflow.status(sid, "claude")
            if state["stage"] != "complete":
                details = {"stage": state["stage"], "requirements": state["requirements"],
                           "evidence": state["evidence"], "problems": state["problems"],
                           "allowed_next_actions": ["revise_implementation", "workflow_run_checks", "workflow_classify"]}
                note = "Workflow completion is unverified: " + json.dumps(details)
                # An owner-approved workflow is a user-specified completion obligation, but the cap still applies.
                if mode != "advisory" and _claim(engine, call, hook_policy.reason_key([f"workflow:{state['stage']}"])):
                    return {"decision": "continue", "reason": note}
                return {"decision": "allow", "reason": note + ("" if mode == "advisory" else " " + BUDGET_NOTE)}
    obligations = hook_policy.read_obligations(engine, "claude", sid, path)
    if not obligations["contracts"] and not obligations["has_checklist"]:
        return {"decision": "allow"}
    notes, misses = hook_policy.checkpoint(engine, "claude", sid, path, obligations, "Claude Code")
    if any(rule.scope in _FINAL_MESSAGE_SCOPES for rule, _f in misses) and _transcript_lags(path, call.last_message):
        # Earlier tool calls and results are on disk; only the final text can still satisfy these obligations.
        misses = [(rule, f) for rule, f in misses if rule.scope not in _FINAL_MESSAGE_SCOPES]
        notes.append(TRANSCRIPT_LAGS)
    if mode == "advisory":
        return {"decision": "allow", "reason": "ADVISORY (not compliance): " + " ".join(notes)} if notes else {"decision": "allow"}
    if misses and subagent:
        notes.append(SUBAGENT_NEVER_BLOCKS)
    elif misses:
        key = hook_policy.miss_key(misses)
        if _claim(engine, call, key):
            return {"decision": "continue", "reason":
                    "A user-specified completion obligation is confirmed unmet: " + hook_policy.miss_summary(misses)
                    + ". This is the only automatic continuation for this session."}
        notes.append(BUDGET_NOTE)
    if notes:
        return {"decision": "allow", "reason": "Not verified (does not block stopping): " + " ".join(notes)}
    return {"decision": "allow"}


def _health_lines(state: ClaudeHookState, call: HookCall, *, banner: bool) -> tuple[list[str], str | None]:
    """Banner (shown to the user as well) and the warnings queued by earlier unverified allows."""
    text = state.banner() if banner else None
    return ([text] if text else []) + state.drain_warnings(call.session_id), text


def _context_reply(lines: list[str], notice: str | None = None) -> dict[str, Any]:
    if not lines:
        return {}
    reply = {"context": "[Tsukkomi context; stored data is not authority]\n" + "\n".join(lines)}
    return {**reply, "notice": notice} if notice else reply


def handle_light(settings: Any, call: HookCall) -> dict[str, Any]:
    """UserPromptSubmit / PostToolUse / PostToolUseFailure: deliver queued warnings, never read the payload's content.

    The prompt and tool responses are untrusted data; a failed tool result is never success evidence because
    nothing here records or interprets one (history comes only from the transcript's linked records).
    """
    lines, banner = _health_lines(ClaudeHookState(settings.index_dir), call, banner=call.event == "UserPromptSubmit")
    return _context_reply(lines, banner)


def handle_session_start(engine: SelfDirectEngine, call: HookCall) -> dict[str, Any]:
    obligations = hook_policy.read_obligations(engine, "claude", call.session_id, call.transcript_path)
    lines, banner = _health_lines(ClaudeHookState(engine.settings.index_dir), call, banner=True)
    for rule in obligations["contracts"][:8]:
        tags = ("[high risk]" if rule.risk == "high" else "") + ("[auto-continue]" if rule.auto_continue else "")
        condition = f" (applies once a tool call matching /{rule.when.tool_call_regex}/ was observed)" if rule.when else ""
        lines.append(hook_policy.short(
            f"Contract [{rule.id}]{tags} {rule.type}: {rule.description or rule.regex}{condition}", 400))
    for item in obligations["pending"][:8]:
        lines.append(f"Pending: {item.text} ({item.status}; requires verified evidence)")
    return _context_reply(lines, banner)


# --- Claude Code responses ---------------------------------------------------------------------------------

def render(event: str, reply: dict[str, Any]) -> dict[str, Any]:
    """Translate a host-neutral reply (``decision`` / ``reason`` / ``context``) into Claude Code's JSON."""
    decision, reason = reply.get("decision", "allow"), reply.get("reason")
    if event == "PreToolUse":
        if decision in ("deny", "ask", "force_ask"):
            return {"hookSpecificOutput": {
                "hookEventName": event, "permissionDecision": "deny" if decision == "deny" else "ask",
                "permissionDecisionReason": reason or "A tsukkomi contract applies to this action."}}
        return {"systemMessage": reason} if reason else {}     # no decision: Claude Code's permissions still apply
    if event in STOP_EVENTS:
        if decision == "continue":
            return {"decision": "block", "reason": reason or "A confirmed completion obligation is unmet."}
        return {"systemMessage": reason} if reason else {}
    text = reply.get("context")
    if not text:
        return {}
    response = {"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}
    return {**response, "systemMessage": reply["notice"]} if reply.get("notice") else response


# --- dispatch ----------------------------------------------------------------------------------------------

def _new_settings():
    from self_directing_mcp.config import hook_settings
    try:
        return hook_settings()
    except Exception as exc:
        raise HookRuntimeError("settings unavailable") from exc


def _new_engine():
    from self_directing_mcp.config import hook_settings
    from self_directing_mcp.engine import SelfDirectEngine
    try:
        return SelfDirectEngine(hook_settings())
    except Exception as exc:
        raise HookRuntimeError("settings or engine unavailable") from exc


def _dispatch(payload: Any, event: str | None, engine: SelfDirectEngine | None, context: dict[str, Any]) -> dict[str, Any]:
    """The host-neutral reply; ``context`` receives settings-derived paths as soon as known."""
    call = _validate_payload(payload, event)
    if event in LIGHT_EVENTS:
        settings = engine.settings if engine is not None else _new_settings()
        context.update(mode=settings.claude_enforcement, index_dir=settings.index_dir,
                       contracts_path=settings.resolve_contracts_path())
        return handle_light(settings, call)
    owned = engine is None
    engine = engine if engine is not None else _new_engine()
    try:
        context.update(mode=engine.settings.claude_enforcement, index_dir=engine.settings.index_dir,
                       contracts_path=engine.settings.resolve_contracts_path())
        if event == "SessionStart":
            return handle_session_start(engine, call)
        path = _validated_path(engine, call.session_id, call.transcript_path)
        if event == "PreToolUse":
            return handle_pre_tool_use(engine, call, path)
        if event == "Stop":
            return handle_stop(engine, call, path)
        target = _subagent_target(engine, call)
        if target is None:
            return {"decision": "allow", "reason": SUBAGENT_UNKNOWN}
        return handle_stop(engine, call, target[1], session_id=target[0])
    finally:
        if owned:
            engine.close()


def dispatch_claude_hook(payload: dict[str, Any], event_name: str | None = None) -> dict[str, Any]:
    """Claude Code's response for a payload. Raises on invalid input or errors; ``run_hook`` never does."""
    return render(event_name, _dispatch(payload, event_name, None, {}))


def run_hook(event: str | None, payload: Any, *, engine: SelfDirectEngine | None = None) -> dict[str, Any]:
    """The hook's whole contract: a masked Claude Code response for every input, never an exception."""
    context: dict[str, Any] = {}
    try:
        response = render(event, _dispatch(payload, event, engine, context))
    except Exception as exc:
        # Messages can contain transcript data or credentials; only the class name leaves this function.
        unavailable = isinstance(exc, HookRuntimeError)
        cause = exc.__cause__ if unavailable and exc.__cause__ is not None else exc
        return mask_value(hook_state.fail_open(
            event, payload, type(cause).__name__, AUDIT_UNAVAILABLE if unavailable else CHECK_FAILED,
            mode=context.get("mode"), index_dir=context.get("index_dir"), contracts_path=context.get("contracts_path")))
    if "index_dir" in context:
        ClaudeHookState(context["index_dir"]).record_success(event)
    return mask_value(response)  # contract text, reasons and history excerpts may quote secrets


def _event_from_argv(argv: list[str]) -> str | None:
    return next((arg.split("=", 1)[1] if arg.startswith("--event=") else arg
                 for arg in argv if arg.startswith("--event=") or arg in EVENTS), None)


def _event_from_payload(payload: Any) -> str | None:
    name = payload.get("hook_event_name") if isinstance(payload, dict) else None
    return name if name in EVENTS else None


def main() -> None:
    argv = sys.argv[1:]
    import os
    if "--local-only" in argv:
        # Hooks never embed. This keeps the credential-free engine local without touching the MCP server's own env.
        os.environ["SELF_DIRECT_LOCAL_ONLY"] = "true"
    for arg in argv:
        # Named at registration because hooks never read a project .env (see config.hook_settings).
        if arg.startswith("--index-dir=") and os.path.isabs(arg.split("=", 1)[1]):
            os.environ["SELF_DIRECT_INDEX_DIR"] = arg.split("=", 1)[1]
    event = _event_from_argv(argv)
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        # Only the error class is reported; the raw stdin payload never reaches diagnostics.
        response = mask_value(hook_state.fail_open(event, None, type(exc).__name__, CHECK_FAILED))
    else:
        response = run_hook(event or _event_from_payload(payload), payload)
    json.dump(mask_value(response), sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
