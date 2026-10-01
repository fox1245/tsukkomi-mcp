"""Antigravity lifecycle controls. Importing this module does not load native code.

Only explicit advisory mode relaxes enforcement. Hook failures are not compliance.
PostToolUse is observational; Stop only resumes a normal, fully-idle model stop.
"""
from __future__ import annotations

import json
import sys
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from self_directing_mcp.engine import SelfDirectEngine

EVENTS = ("PreToolUse", "PostToolUse", "PreInvocation", "Stop")


def failure_response(event: str | None, payload: Any = None, *, reason: str = "Hook verification unavailable") -> dict:
    if event == "PostToolUse":
        return {}
    if event == "PreInvocation":
        return {"injectSteps": [{"ephemeralMessage": reason + "; do not claim compliance."}]}
    if event == "Stop":
        # Never turn cancellation, runtime error, or a background-task pause into
        # a restart loop. Missing/malformed metadata is not a normal model stop.
        resume = isinstance(payload, dict) and payload.get("terminationReason") in ("model_stop", "NO_TOOL_CALL") and payload.get("fullyIdle") is True
        return {"decision": "continue" if resume else "allow", "reason": reason}
    return {"decision": "deny", "reason": reason}


def _advisory(engine: SelfDirectEngine) -> bool:
    return engine.settings.agy_enforcement == "advisory"


def _verdict_response(audit: dict, advisory: bool) -> dict:
    verdict = audit.get("verdict", "unknown")
    if audit.get("ok") is not True:
        verdict = "unknown"
    if audit.get("ok") is True and audit.get("action_scope") == "not_applicable":
        return {"decision": "allow", "reason": "No triggering action contract; not a compliance verdict."}
    if verdict == "suspicious" and (
            any(f.get("verdict") in ("unknown", "violation") for f in audit.get("findings", [])
                if f.get("contract_id") in audit.get("action_contracts", []))
            or (audit.get("requires_history") and not audit.get("coverage", {}).get("complete"))):
        verdict = "unknown"  # Confirming an exception cannot waive another prerequisite.
    if verdict == "clean":
        return {"decision": "allow"}
    reasons = [f"[{f.get('contract_id')}] {f.get('reason', 'Verification unavailable')}"
               for f in audit.get("findings", []) if f.get("verdict") != "clean"]
    reason = "; ".join(reasons[:4]) or "Contract verification incomplete; resolve missing evidence."
    if advisory:
        return {"decision": "allow", "reason": "ADVISORY (not compliance): " + reason}
    return {"decision": "force_ask" if verdict == "suspicious" else "deny", "reason": reason}




def handle_pre_tool_use(engine: SelfDirectEngine, session_id: str, tool_call: dict[str, Any],
                        path: str | None = None) -> dict[str, Any]:
    # No read-command classifier or tool-name exemption: reads can be restricted,
    # shell flags can mutate, and names containing this server are not authority.
    from self_directing_mcp.workflow import guard_host_action
    workflow = guard_host_action(session_id, {"tool_name": tool_call["name"], "arguments": tool_call["args"]}, "agy")
    if workflow is not None:
        return {"decision": workflow["decision"], "reason": workflow["reason"]}
    obligations = engine.hook_obligations(session_id, provider="agy", path=path)
    if not obligations["contracts"]:
        return {"decision": "allow", "reason": "No applicable action contracts; not a compliance verdict."}
    audit = engine.check_action(session_id=session_id, provider="agy", path=path,
                                action={"tool_name": tool_call["name"], "arguments": tool_call["args"]})
    return _verdict_response(audit, _advisory(engine))


def handle_stop(engine: SelfDirectEngine, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("terminationReason") not in ("model_stop", "NO_TOOL_CALL") or payload.get("fullyIdle") is not True:
        return {"decision": "allow"}
    from self_directing_mcp.workflow import configured_workflow
    workflow = configured_workflow()
    if workflow is not None:
        state = workflow.status(session_id, "agy")
        if state["stage"] != "complete":
            details = {"stage": state["stage"], "requirements": state["requirements"],
                       "evidence": state["evidence"], "problems": state["problems"],
                       "allowed_next_actions": ["revise_implementation", "workflow_run_checks", "workflow_classify"]}
            return {"decision": "continue", "reason": "Workflow completion is unverified: " + json.dumps(details)}
    path = payload["transcriptPath"]
    obligations = engine.hook_obligations(session_id, provider="agy", path=path)
    if not obligations["contracts"] and not obligations["has_checklist"]:
        return {"decision": "allow", "reason": "No outstanding obligations; not a compliance verdict."}
    synced = engine.sync_session(session_id=session_id, provider="agy", path=path, embed=False)
    reasons = []
    if not synced.get("ok") or not synced.get("coverage", {}).get("complete"):
        reasons.append("AGY history is incomplete; completion cannot be verified.")
    checklist = engine.get_checklist(session_id, provider="agy")
    pending = [item for item in checklist["items"] if item["status"] not in ("done", "revoked")]
    reasons.extend(f"{item['text']} (status={item['status']}; missing verified completion evidence)"
                   for item in pending[:4])
    if obligations["contracts"]:
        audit = engine.audit_session(session_id, provider="agy", path=path)
        if audit.get("ok") is not True or audit.get("verdict") != "clean":
            reasons.append("Contract checkpoint is " + str(audit.get("verdict", "unknown")) + ".")
    if reasons:
        return {"decision": "allow" if _advisory(engine) else "continue",
                "reason": ("ADVISORY (not compliance): " if _advisory(engine) else "") + " ".join(reasons)}
    return {"decision": "allow"}


def handle_pre_invocation(engine: SelfDirectEngine, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = payload["transcriptPath"]
    obligations = engine.hook_obligations(session_id, provider="agy", path=path)
    graph = engine.get_graph_context(session_id=session_id, provider="agy", path=path)
    lines = []
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
        lines.append(f"Contract [{rule.id}] {rule.type}: {rule.description or rule.regex}")
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


def dispatch_agy_hook(payload: dict[str, Any], event_name: str | None = None) -> dict[str, Any]:
    # Explicit registration avoids confusing PostToolUse with PreToolUse.
    sid, path = _validate_payload(payload, event_name)
    from self_directing_mcp.agy import resolve_session_path
    from self_directing_mcp.engine import SelfDirectEngine
    engine = SelfDirectEngine()
    try:
        resolve_session_path(engine.settings.resolve_agy_app_data_dirs(), session_id=sid,
                             path=path, require_exists=False)
        if event_name in ("PreToolUse", "PostToolUse"):
            engine.record_agy_hook(payload, event_name)
        if event_name == "PreToolUse":
            return handle_pre_tool_use(engine, sid, payload["toolCall"], path=path)
        if event_name == "Stop":
            return handle_stop(engine, sid, payload)
        if event_name == "PreInvocation":
            return handle_pre_invocation(engine, sid, payload)
        result = engine.sync_session(session_id=sid, provider="agy", path=path, embed=False)
        if not result.get("ok"):
            print("Tsukkomi PostToolUse: history refresh unavailable", file=sys.stderr)
        return {}
    finally:
        engine.close()


def main() -> None:
    event = next((arg.split("=", 1)[1] if arg.startswith("--event=") else arg
                  for arg in sys.argv[1:] if arg.startswith("--event=") or arg in EVENTS), None)
    payload = None
    try:
        payload = json.load(sys.stdin)
        response = dispatch_agy_hook(payload, event_name=event)
    except Exception as exc:
        # Exception messages can contain transcript data or credentials. Emit only
        # the error class; diagnostics must never leak the raw stdin payload.
        reason = f"Tsukkomi verification unavailable ({type(exc).__name__})"
        print(reason, file=sys.stderr)
        response = failure_response(event, payload, reason=reason)
    json.dump(response, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
