"""Antigravity (AGY) lifecycle hook adapter for self-directing-mcp.

Implements the AGY JSON-over-stdio contract for:
- PreToolUse: checks tool calls against active contracts (deny/ask/allow).
- Stop: blocks premature agent termination if checklist obligations are pending.
- PreInvocation: optionally injects ephemeral contract reminders.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

from self_directing_mcp.config import get_settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.session_events import as_text

# Auto-detect repo .env file if key environment variables are not set
_repo_env = Path(__file__).resolve().parent.parent.parent / ".env"
if _repo_env.is_file() and "SELF_DIRECT_OPENROUTER_API_KEY_FILE" not in os.environ and "OPENROUTER_API_KEY" not in os.environ:
    os.environ["SELF_DIRECT_OPENROUTER_API_KEY_FILE"] = str(_repo_env)

# Auto-detect native/vector.so if path is not explicitly set
for ext_name in ("vector.so", "vector.dll", "vector.dylib"):
    _cand = Path(__file__).resolve().parent.parent.parent / "native" / ext_name
    if _cand.is_file() and "SELF_DIRECT_SQLITE_VECTOR_PATH" not in os.environ:
        os.environ["SELF_DIRECT_SQLITE_VECTOR_PATH"] = str(_cand)
        break

AGY_READ_TOOLS = {
    "view_file",
    "list_dir",
    "grep_search",
    "find_by_name",
    "read_resource",
    "list_resources",
    "read_url_content",
    "search_web",
}

AGY_READ_COMMAND_VERBS = {
    "ls", "dir", "pwd", "whoami", "cat", "head", "tail", "grep", "rg",
    "find", "which", "echo", "true", "test"
}


def is_read_only_tool_call(tool_name: str, args: dict[str, Any] | Any) -> bool:
    """Fast-path check for read-only tools to avoid lock overhead."""
    name = (tool_name or "").lower()
    if name in AGY_READ_TOOLS:
        return True
    if name == "run_command" and isinstance(args, dict):
        cmd = args.get("CommandLine", "").strip()
        if not cmd or any(c in cmd for c in "\n\r;|&<>`$(){}!"):
            return False
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            return False
        if not tokens:
            return True
        verb = tokens[0].lower()
        if verb in AGY_READ_COMMAND_VERBS:
            return True
        if verb == "git" and len(tokens) > 1 and tokens[1].lower() in {"status", "diff", "log", "branch", "show"}:
            return True
    return False


def _extract_target_entities(tool_name: str, args: dict[str, Any]) -> list[str]:
    """Extract candidate files or entity names from tool call arguments."""
    entities: list[str] = []
    if not isinstance(args, dict):
        return entities
    for key in ("TargetFile", "TargetDirectory", "AbsolutePath", "SearchDirectory", "file_path", "path"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            entities.append(val.strip())
            base = Path(val.strip()).name
            if base and base not in entities:
                entities.append(base)
    cmd = args.get("CommandLine")
    if isinstance(cmd, str) and cmd.strip():
        tokens = re.findall(r'[a-zA-Z0-9_\-\./]+\.[a-zA-Z0-9]+', cmd)
        for t in tokens:
            entities.append(t)
            base = Path(t).name
            if base and base not in entities:
                entities.append(base)
    return entities


def handle_pre_tool_use(engine: SelfDirectEngine, session_id: str, tool_call: dict[str, Any]) -> dict[str, Any]:
    tool_name = tool_call.get("name", "")
    args = tool_call.get("args", {})

    # Auto-allow internal self-directing-mcp tool calls to prevent recursion
    if "self_directing_mcp" in tool_name.lower() or "self-directing-mcp" in tool_name.lower():
        return {"decision": "allow"}

    # Fast path for obvious read-only actions
    if is_read_only_tool_call(tool_name, args):
        return {"decision": "allow"}

    # Check GraphRAG relations for target entities
    dep_notices: list[str] = []
    try:
        engine.ensure_ready()
        targets = _extract_target_entities(tool_name, args)
        if targets and hasattr(engine, "graph") and engine.graph:
            for tgt in targets[:3]:
                sub = engine.graph.neighbors(session_id, "codex", tgt, depth=1)
                for e in sub.get("edges", []):
                    if e.get("src") == tgt:
                        dep_notices.append(f"{tgt} -[{e['relation']}]-> {e['dst']}")
                    elif e.get("dst") == tgt:
                        dep_notices.append(f"{e['src']} -[{e['relation']}]-> {tgt}")
    except Exception:
        pass

    # Evaluate against active contracts
    audit = engine.check_action(
        session_id=session_id,
        action={"tool_name": tool_name, "arguments": args},
    )

    verdict = audit.get("verdict", "unknown")
    findings = [f for f in audit.get("findings", []) if f.get("verdict") != "clean"]

    dep_suffix = f" (GraphRAG dependencies: {'; '.join(dep_notices[:2])})" if dep_notices else ""

    if verdict == "violation":
        reasons = [f"[{f.get('contract_id')}] {f.get('reason', 'Rule violated')}" for f in findings]
        reason_str = "; ".join(reasons) if reasons else "Violated active contract rule."
        return {
            "decision": "deny",
            "reason": f"Self-directing MCP Contract Violation: {reason_str}{dep_suffix}"
        }

    if verdict == "suspicious":
        reasons = [f"[{f.get('contract_id')}] {f.get('reason', 'Suspicious action')}" for f in findings]
        reason_str = "; ".join(reasons) if reasons else "Action flagged as suspicious."
        return {
            "decision": "ask",
            "reason": f"Self-directing MCP Contract Warning: {reason_str}{dep_suffix}"
        }

    if dep_notices:
        return {
            "decision": "allow",
            "reason": f"Self-directing MCP GraphRAG Note: {'; '.join(dep_notices[:2])}"
        }

    return {"decision": "allow"}


def handle_stop(engine: SelfDirectEngine, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    # If background tasks or subagents are still actively running, allow pausing to await their completion
    if not payload.get("fullyIdle", True):
        return {}

    obligations = engine.hook_obligations(session_id)
    pending_items = obligations.get("pending", [])

    if pending_items:
        reasons = [f"- {i.text} (status={i.status})" for i in pending_items[:3]]
        return {
            "decision": "continue",
            "reason": (
                "Self-directing MCP: Cannot terminate loop. The following checklist items lack verified completion evidence:\n"
                + "\n".join(reasons)
                + "\nPlease complete or verify these requirements with evidence before stopping."
            )
        }

    return {}


def handle_pre_invocation(engine: SelfDirectEngine, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    obligations = engine.hook_obligations(session_id)
    contracts = obligations.get("contracts", [])
    pending = obligations.get("pending", [])

    # Query GraphRAG knowledge graph
    graph_edges: list[dict[str, Any]] = []
    graph_nodes: list[dict[str, Any]] = []
    try:
        graph_ctx = engine.get_graph_context(session_id=session_id)
        graph_edges = graph_ctx.get("edges", [])
        graph_nodes = graph_ctx.get("nodes", [])
    except Exception:
        pass

    if not contracts and not pending and not graph_edges and not graph_nodes:
        return {}

    lines = ["[Self-Directing MCP: Active Context & GraphRAG Guard]"]

    if graph_edges:
        lines.append(f"Knowledge Graph Relationships ({len(graph_nodes)} nodes, {len(graph_edges)} relations):")
        for e in graph_edges[:8]:
            lines.append(f"  • {e['src']} -[{e['relation']}]-> {e['dst']} (origin={e.get('origin', 'inferred')})")
    elif graph_nodes:
        node_labels = [f"{n['node_id']} ({n.get('kind', 'concept')})" for n in graph_nodes[:6]]
        lines.append("Active Graph Entities: " + ", ".join(node_labels))

    if pending:
        lines.append(f"Pending Checklist Obligations ({len(pending)}):")
        for item in pending[:5]:
            lines.append(f"  • {item.text} (status={item.status})")

    if contracts:
        lines.append(f"Active Invariant Contracts ({len(contracts)} rules enforced):")
        for c in contracts[:4]:
            cid = getattr(c, "id", getattr(c, "contract_id", "rule"))
            ctype = getattr(c, "type", getattr(c, "rule_type", "rule"))
            cdesc = getattr(c, "description", getattr(c, "regex", ""))
            lines.append(f"  • [{cid}] {ctype}: {cdesc}")

    lines.append("All file modifications and commands are audited against these relations and contracts.")
    msg = "\n".join(lines)

    return {
        "injectSteps": [
            {
                "ephemeralMessage": msg
            }
        ]
    }


def dispatch_agy_hook(payload: dict[str, Any], event_name: str | None = None) -> dict[str, Any]:
    session_id = str(payload.get("conversationId") or "default-agy-session")

    # Detect event type if not explicitly supplied
    if not event_name:
        if "toolCall" in payload:
            event_name = "PreToolUse"
        elif "terminationReason" in payload or "executionNum" in payload:
            event_name = "Stop"
        elif "invocationNum" in payload:
            event_name = "PreInvocation"
        else:
            event_name = "PreToolUse"

    engine = SelfDirectEngine()

    if event_name == "PreToolUse":
        tool_call = payload.get("toolCall") or {}
        return handle_pre_tool_use(engine, session_id, tool_call)
    elif event_name == "Stop":
        return handle_stop(engine, session_id, payload)
    elif event_name == "PreInvocation":
        return handle_pre_invocation(engine, session_id, payload)

    return {}


def main() -> None:
    try:
        raw_input = sys.stdin.read()
        payload = json.loads(raw_input) if raw_input.strip() else {}
    except Exception as exc:
        # Fallback to allow if stdin JSON parsing fails
        json.dump({"decision": "allow", "reason": f"Hook error parsing stdin: {exc}"}, sys.stdout)
        return

    # Check optional CLI argument for event
    event_arg = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--event="):
                event_arg = arg.split("=", 1)[1]
            elif arg in ("PreToolUse", "Stop", "PreInvocation", "PostToolUse", "PostInvocation"):
                event_arg = arg

    try:
        response = dispatch_agy_hook(payload, event_name=event_arg)
    except Exception as exc:
        # Never crash the agent loop on unhandled exception; return warning
        response = {"decision": "ask", "reason": f"Self-directing MCP hook error: {type(exc).__name__}: {exc}"}

    json.dump(response, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
