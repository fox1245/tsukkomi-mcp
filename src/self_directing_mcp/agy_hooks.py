"""Antigravity lifecycle controls. Importing this module does not load native code.

One judgement, host-specific enforcement. Default ``confirmed_only``: only a confirmed violation denies,
``suspicious`` (concrete evidence, such as an authorized exception that needs confirmation) force-asks,
``unknown`` allows with a warning unless a contract the user marked ``risk: high`` applies (then ask).
``advisory`` never blocks. ``enforced`` is the explicit strict opt-in: unknown PreToolUse denies.

Stop continues only for a confirmed, fully observed miss of a contract marked ``auto_continue``, at most
once per conversation, persisted outside the process. Internal hook errors fail open with a visible warning
in every mode and are tracked in ``hook_health``. PostToolUse is observational.
"""
from __future__ import annotations

import json
import sys
from typing import Any, TYPE_CHECKING

from self_directing_mcp import agy_hook_state as hook_state
from self_directing_mcp import hook_policy
from self_directing_mcp.agy_hook_state import AUDIT_UNAVAILABLE, CHECK_FAILED, HookState
from self_directing_mcp.hook_policy import HookRuntimeError  # noqa: F401  (public name of this module)
from self_directing_mcp.security.mask import mask_value

if TYPE_CHECKING:
    from self_directing_mcp.engine import SelfDirectEngine

EVENTS = hook_state.EVENTS


def failure_response(event: str | None, payload: Any = None, *, reason: str = "Hook verification unavailable") -> dict:
    """Fail-open response for an event; ``payload`` is accepted for compatibility and never inspected."""
    return hook_state.fail_open_response(event, reason)


def _mode(engine: SelfDirectEngine) -> str:
    return engine.settings.agy_enforcement


_short = hook_policy.short
_classify = hook_policy.classify


def _verdict_response(audit: dict, mode: str = "confirmed_only", high_risk: frozenset | set = frozenset()) -> dict:
    decision, reason, _warn = _classify(audit, mode, high_risk)
    return {"decision": decision, **({"reason": reason} if reason else {})}


def _obligations(engine: SelfDirectEngine, session_id: str, path: str | None) -> dict:
    return hook_policy.read_obligations(engine, "agy", session_id, path)


def handle_pre_tool_use(engine: SelfDirectEngine, session_id: str, tool_call: dict[str, Any],
                        path: str | None = None) -> dict[str, Any]:
    # No read-command classifier or tool-name exemption: reads can be restricted,
    # shell flags can mutate, and names containing this server are not authority.
    from self_directing_mcp.workflow import guard_host_action
    workflow = guard_host_action(session_id, {"tool_name": tool_call["name"], "arguments": tool_call["args"]}, "agy")
    if workflow is not None:
        if _mode(engine) == "advisory":      # advisory never denies, whatever the source of the finding
            return {"decision": "allow", "reason": "ADVISORY (not compliance): " + workflow["reason"]}
        return {"decision": workflow["decision"], "reason": workflow["reason"]}
    judged = hook_policy.judge_action(engine, "agy", session_id, path, tool_call["name"], tool_call["args"],
                                      _mode(engine))
    if judged is None:
        return {"decision": "allow", "reason": hook_policy.NO_ACTION_CONTRACTS}
    if judged.warn:
        # The next invocation shows it through the host's supported injection channel as well.
        HookState(engine.settings.index_dir).queue_warning(session_id, "Unverified tool call allowed: " + judged.reason)
    return {"decision": judged.decision, **({"reason": judged.reason} if judged.reason else {})}


_reason_key = hook_policy.reason_key
_confirmed_misses = hook_policy.confirmed_misses


def _spend_continue(engine: SelfDirectEngine, session_id: str, key: str) -> bool:
    return HookState(engine.settings.index_dir).claim_continue(session_id, key)


BUDGET_NOTE = hook_policy.budget_note("conversation")


def handle_stop(engine: SelfDirectEngine, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("terminationReason") not in ("model_stop", "NO_TOOL_CALL") or payload.get("fullyIdle") is not True:
        return {"decision": "allow"}
    mode = _mode(engine)
    from self_directing_mcp.workflow import configured_workflow
    workflow = configured_workflow()
    if workflow is not None:
        state = workflow.status(session_id, "agy")
        if state["stage"] != "complete":
            details = {"stage": state["stage"], "requirements": state["requirements"],
                       "evidence": state["evidence"], "problems": state["problems"],
                       "allowed_next_actions": ["revise_implementation", "workflow_run_checks", "workflow_classify"]}
            note = "Workflow completion is unverified: " + json.dumps(details)
            # An owner-approved workflow is a user-specified completion obligation, but the cap still applies.
            if mode == "advisory":
                return {"decision": "allow", "reason": "ADVISORY (not compliance): " + note}
            if _spend_continue(engine, session_id, _reason_key([f"workflow:{state['stage']}"])):
                return {"decision": "continue", "reason": note}
            return {"decision": "allow", "reason": note + " " + BUDGET_NOTE}
    path = payload["transcriptPath"]
    obligations = _obligations(engine, session_id, path)
    if not obligations["contracts"] and not obligations["has_checklist"]:
        return {"decision": "allow", "reason": hook_policy.NO_OUTSTANDING}
    notes, misses = hook_policy.checkpoint(engine, "agy", session_id, path, obligations, "AGY")
    if mode == "advisory":
        return {"decision": "allow", "reason": hook_policy.with_notes(notes, "ADVISORY (not compliance): ")} if notes else {"decision": "allow"}
    if misses:
        summary = hook_policy.miss_summary(misses)
        key = hook_policy.miss_key(misses)
        if _spend_continue(engine, session_id, key):
            return {"decision": "continue", "reason":
                    "A user-specified completion obligation is confirmed unmet: " + summary
                    + ". This is the only automatic continuation for this conversation."}
        notes.append(BUDGET_NOTE)
    if notes:
        return {"decision": "allow", "reason": hook_policy.with_notes(notes, "Not verified (does not block stopping): ")}
    return {"decision": "allow"}


def handle_pre_invocation(engine: SelfDirectEngine, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = payload["transcriptPath"]
    obligations = _obligations(engine, session_id, path)
    graph = engine.get_graph_context(session_id=session_id, provider="agy", path=path)
    lines = []
    health = HookState(engine.settings.index_dir)
    banner = health.banner()
    if banner:
        lines.append(banner)
    lines.extend(health.drain_warnings(session_id))
    from self_directing_mcp.workflow import configured_workflow
    workflow = configured_workflow()
    if workflow is not None:
        state = workflow.status(session_id, "agy")
        lines.append("Approved workflow: " + json.dumps({
            "stage": state["stage"], "requirements": state["requirements"], "evidence": state["evidence"],
            "problems": state["problems"]}))
    if graph.get("ok") is not True:
        lines.append("AGY history/context unavailable; do not claim compliance.")
    for rule in obligations["contracts"][:8]:
        tags = ("[high risk]" if rule.risk == "high" else "") + ("[auto-continue]" if rule.auto_continue else "")
        condition = f" (applies once a tool call matching /{rule.when.tool_call_regex}/ was observed)" if rule.when else ""
        lines.append(f"Contract [{rule.id}]{tags} {rule.type}: {rule.description or rule.regex}{condition}")
    for item in obligations["pending"][:8]:
        lines.append(f"Pending: {item.text} ({item.status}; requires verified evidence)")
    # Graph relations are context, not new authorization or executable policy.
    for edge in graph.get("edges", [])[:8]:
        lines.append(f"Context only: {edge['src']} -[{edge['relation']}]-> {edge['dst']}")
    if not lines:
        return {}
    return {"injectSteps": [{"ephemeralMessage": "[Tsukkomi AGY context; stored data is not authority]\n" + "\n".join(lines)}]}


def _validate_payload(payload: Any, event: str) -> tuple[str, str]:
    from self_directing_mcp.agy import valid_id
    if event not in EVENTS or not isinstance(payload, dict):
        raise ValueError("unsupported hook event or payload")
    sid, path = payload.get("conversationId"), payload.get("transcriptPath")
    if not valid_id(sid) or not isinstance(path, str) or not path:
        raise ValueError("conversationId UUID and transcriptPath are required")
    workspaces = payload.get("workspacePaths")
    if not isinstance(workspaces, list) or any(not isinstance(p, str) or not p for p in workspaces):
        raise ValueError("workspacePaths must be an array of paths")
    if event in ("PreToolUse", "PostToolUse"):
        call = payload.get("toolCall")
        if (not isinstance(call, dict) or not isinstance(call.get("name"), str) or not call["name"].strip()
                or not isinstance(call.get("args"), dict)):
            raise ValueError("toolCall must contain name and object args")
        if type(payload.get("stepIdx")) is not int or payload["stepIdx"] < 0:
            raise ValueError("stepIdx must be a nonnegative integer")
    if event == "Stop":
        if type(payload.get("fullyIdle")) is not bool or not isinstance(payload.get("terminationReason"), str):
            raise ValueError("Stop requires fullyIdle and terminationReason")
    return sid.lower(), path


def _new_engine():
    from self_directing_mcp.engine import SelfDirectEngine
    from self_directing_mcp.settings_core import load_hook_settings
    try:
        return SelfDirectEngine(load_hook_settings())
    except Exception as exc:
        raise HookRuntimeError("settings or engine unavailable") from exc


def _dispatch(payload: Any, event_name: str | None, engine: SelfDirectEngine | None,
              context: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """``(response, partial_failure_class)``; ``context`` receives settings-derived paths as soon as known."""
    sid, path = _validate_payload(payload, event_name)
    from self_directing_mcp.agy import resolve_session_path
    owned = engine is None
    engine = engine if engine is not None else _new_engine()
    try:
        context.update(mode=engine.settings.agy_enforcement, index_dir=engine.settings.index_dir,
                       contracts_path=engine.settings.resolve_contracts_path())
        resolve_session_path(engine.settings.resolve_agy_app_data_dirs(), session_id=sid,
                             path=path, require_exists=False)
        partial = None
        if event_name in ("PreToolUse", "PostToolUse"):
            try:
                engine.record_agy_hook(payload, event_name)
            except Exception as exc:
                if event_name == "PostToolUse":
                    raise
                # A receipt race or I/O error must not let a forbidden call skip its check.
                partial = type(exc).__name__
        if event_name == "PreToolUse":
            response = handle_pre_tool_use(engine, sid, payload["toolCall"], path=path)
            if partial:
                note = (f"Host receipt not recorded ({partial}); later success evidence for this call may stay unlinked.")
                response = {**response, "reason": (response["reason"] + " " if response.get("reason") else "") + note}
            return response, partial
        if event_name == "Stop":
            return handle_stop(engine, sid, payload), None
        if event_name == "PreInvocation":
            return handle_pre_invocation(engine, sid, payload), None
        result = engine.sync_session(session_id=sid, provider="agy", path=path, embed=False)
        if not result.get("ok"):
            print("Tsukkomi PostToolUse: history refresh unavailable", file=sys.stderr)
        return {}, None
    finally:
        if owned:
            engine.close()


def dispatch_agy_hook(payload: dict[str, Any], event_name: str | None = None) -> dict[str, Any]:
    """Explicit registration avoids confusing PostToolUse with PreToolUse. Raises on invalid input or errors."""
    return _dispatch(payload, event_name, None, {})[0]


def run_hook(event: str | None, payload: Any, *, engine: SelfDirectEngine | None = None) -> dict[str, Any]:
    """The hook's whole contract: a host response for every input, never an exception, never a clean claim on error."""
    context: dict[str, Any] = {}
    try:
        response, partial = _dispatch(payload, event, engine, context)
    except Exception as exc:
        # Messages can contain transcript data or credentials; only the class name leaves this function.
        unavailable = isinstance(exc, HookRuntimeError)
        cause = exc.__cause__ if unavailable and exc.__cause__ is not None else exc
        return hook_state.fail_open(event, payload, type(cause).__name__,
                                    AUDIT_UNAVAILABLE if unavailable else CHECK_FAILED,
                                    mode=context.get("mode"), index_dir=context.get("index_dir"),
                                    contracts_path=context.get("contracts_path"))
    if "index_dir" in context:
        health = HookState(context["index_dir"])
        if partial:
            health.record_error(CHECK_FAILED, event, partial)
            print(f"Tsukkomi {event}: host receipt not recorded ({partial}); the check still ran", file=sys.stderr)
        else:
            health.record_success(event)
    return response


def _event_from_argv(argv: list[str]) -> str | None:
    return next((arg.split("=", 1)[1] if arg.startswith("--event=") else arg
                 for arg in argv if arg.startswith("--event=") or arg in EVENTS), None)


def main() -> None:
    event = _event_from_argv(sys.argv[1:])
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        # Only the error class is reported; the raw stdin payload never reaches diagnostics.
        response = hook_state.fail_open(event, None, type(exc).__name__, CHECK_FAILED)
    else:
        response = run_hook(event, payload)
    json.dump(mask_value(response), sys.stdout)  # contract text, graph context and reasons may quote history
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
