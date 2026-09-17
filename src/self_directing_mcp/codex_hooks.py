"""Conditional, local-only lifecycle checks. Silence is never authorization."""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
from pathlib import Path
from contextlib import contextmanager

from self_directing_mcp.security.mask import mask_secrets
from self_directing_mcp.session_events import as_text
from self_directing_mcp.codex.discover import ensure_under_root

EVENTS = ("UserPromptSubmit", "PreToolUse", "Stop")
_READ_TOOLS = {"read", "read_file", "list", "list_files", "glob", "search", "get",
               "view_image", "get_chunk", "list_contracts", "get_checklist", "audit_status",
               "search_history", "help", "ls"}
_SELF_READS = _READ_TOOLS | {"audit_session", "check_action", "sync_session"}
_REQUIREMENT_SIGNALS = (
    "하지 마", "하지 말고", "금지", "필수", "반드시", "꼭", "항상", "절대",
    "do not", "don't", "never", "always", "must", "forbidden", "prohibited",
    "requirement", "except", "only", "먼저", "마지막으로",
)
_GUIDANCE = "Register only the user's explicit added/changed constraints with upsert_contracts; do not invent rules."


def _carries_requirement_signal(prompt) -> bool:
    return isinstance(prompt, str) and any(sig in prompt.lower() for sig in _REQUIREMENT_SIGNALS)


def _leaf(tool_name):
    return re.split(r"__|[.:]", tool_name.lower())[-1]


def _is_risky_tool(tool_name: str) -> bool:
    # Exact names, never substring matching ("delete_list" is not a read).
    return _leaf(tool_name or "") not in _READ_TOOLS


def _is_risky_input(tool_input) -> bool:
    text = as_text(tool_input).lower()
    return any(k in text for k in ("rm -rf", "delete", "drop table", "--force", "deploy", "publish", "install"))


def _unwrap(tool_name, tool_input):
    """Accept one literal JSON tool call only; never evaluate JavaScript.

    Dynamic arguments, aliases, additional statements and arbitrary expressions
    remain opaque and are checked as executable code.
    """
    if tool_name not in ("functions.exec", "exec"):
        return tool_name, tool_input
    source = tool_input.get("code") if isinstance(tool_input, dict) else tool_input
    if not isinstance(source, str):
        return tool_name, tool_input
    match = re.match(r"\s*(text\(\s*)?await\s+tools\.([a-zA-Z0-9_]+)\(", source)
    if not match:
        return tool_name, tool_input
    try:
        rest = source[match.end():].lstrip()
        args, end = json.JSONDecoder().raw_decode(rest)
        suffix = rest[end:].strip().rstrip(";").strip()
        if suffix != ("))" if match[1] else ")") or not isinstance(args, dict):
            return tool_name, tool_input
        return match[2], args
    except (ValueError, TypeError):
        return tool_name, tool_input


def _read_command(args):
    if not isinstance(args, dict):
        return False
    command = args.get("cmd", args.get("command", ""))
    if not isinstance(command, str) or any(c in command for c in "\n\r;|&<>`$(){}!"):
        return False
    try:
        words = shlex.split(command, posix=False)
    except ValueError:
        return False
    if not words:
        return False
    verb = words[0].lower()
    if verb == "rg":
        return not any(w.strip("\"'").lower().startswith(("--pre", "--hostname-bin")) for w in words[1:])
    return verb in {"get-content", "get-item", "get-childitem", "get-command", "resolve-path", "test-path"}


def _read_only(tool_name, tool_input):
    name, args = _unwrap(tool_name, tool_input)
    if re.search(r"(?:^|__)self[-_]directing[-_]mcp__", name):
        return _leaf(name) in _SELF_READS
    return not _is_risky_tool(name) or (_leaf(name) in {"exec_command", "shell", "powershell"} and _read_command(args))


def _read_has_constraint(rules, tool_name, tool_input):
    text = f"[tool_call:{tool_name}] {as_text(tool_input)}"
    name, args = _unwrap(tool_name, tool_input)
    if name != tool_name:
        text += f"\n[tool_call:{name}] {as_text(args)}"
    for rule in rules:
        if rule.scope not in ("tool_call", "any") or "assistant" not in rule.roles:
            continue
        if rule.type == "must_not" and not rule.regex:
            return True
        pattern = rule.regex if rule.type == "must_not" else rule.before_regex
        if pattern:
            try:
                if re.search(pattern, text, re.I | re.M):
                    return True
            except re.error:
                return True
    return False


def _context(event, text):
    return {"systemMessage": text} if event == "Stop" else {
        "hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}


@contextmanager
def _state_lock(engine, deadline):
    budget = engine.settings.lock_wait_timeout_sec
    if deadline is not None:
        budget = min(budget, deadline - time.monotonic())
    if budget <= 0 or not engine._hook_lock.acquire(timeout=budget):
        raise TimeoutError("hook_state_busy")
    try:
        yield
    finally:
        engine._hook_lock.release()


def _evidence_stamp(engine, path, rules):
    """Only reuse a checkpoint with unchanged relevant transcript evidence.

    Ignore our own developer-context receipts and known usage metadata. Other
    unknown records invalidate the checkpoint conservatively. No source text is
    retained, and root validation precedes reading even on this fast path.
    """
    if not path:
        return None
    checked = ensure_under_root(Path(path), Path(engine.settings.codex_sessions_dir))
    messages = any(r.regex and r.scope in ("message", "any") for r in rules)
    digest = hashlib.sha256()
    with checked.open("rb") as source:
        for line in source:
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError):
                digest.update(line)
                continue
            if not isinstance(record, dict):
                digest.update(line)
                continue
            payload = record.get("payload", {})
            if not isinstance(payload, dict):
                digest.update(line)
                continue
            if record.get("type") == "token_usage_record" or (
                record.get("type") == "event_msg" and payload.get("type") == "token_count"
            ):
                continue
            if record.get("type") == "response_item" and payload.get("type") == "message":
                text = "\n".join(c.get("text", "") for c in payload.get("content", []) if isinstance(c, dict))
                if payload.get("role") == "developer" and text.startswith(("Self-directing audit", "Self-directing session:")):
                    continue
                if payload.get("role") == "assistant" and not messages:
                    continue
            if record.get("type") == "event_msg" and payload.get("type") in ("agent_message", "agent_reasoning") and not messages:
                continue
            digest.update(line)
    return digest.hexdigest()


def handle_hook(engine, event_name, session_id=None, transcript_path=None,
                tool_name=None, tool_input=None, *, prompt=None, turn_id=None,
                stop_hook_active=False, _deadline=None):
    if event_name not in EVENTS or (event_name == "Stop" and stop_hook_active):
        return {}
    if tool_name and re.search(r"(?:^|__)self[-_]directing[-_]mcp(?:__|[.:])", tool_name):
        return {}
    if event_name == "PreToolUse" and not tool_name:
        return {}
    user_prompt = prompt if prompt is not None else tool_input
    if not session_id:
        return _context(event_name, "Current session id is unavailable. " + _GUIDANCE) if (
            event_name == "UserPromptSubmit" and _carries_requirement_signal(user_prompt)) else {}
    try:
        obligations = engine.hook_obligations(session_id)
        with _state_lock(engine, _deadline):
            key = session_id.lower()
            state = engine._hook_states.setdefault(key, {"dirty": False, "checkpoint": None})
            engine._hook_states.move_to_end(key)
            while len(engine._hook_states) > 256:
                engine._hook_states.popitem(last=False)
            stamp = obligations["snapshot"]
            if event_name == "UserPromptSubmit":
                if not state["dirty"]:
                    state["checkpoint"] = stamp
                    state["evidence"] = (_evidence_stamp(engine, transcript_path, obligations["contracts"])
                                         if obligations["status"] != "not_applicable" else None)
                if not _carries_requirement_signal(user_prompt):
                    return {}
                signature = (turn_id, user_prompt)
                if state.get("nudge") == signature:
                    return {}
                state["nudge"] = signature
                return _context(event_name, f"Self-directing session: {session_id}. " + _GUIDANCE)
            if obligations["status"] == "not_applicable":
                return {}
            rules = obligations["contracts"]
            if event_name == "PreToolUse":
                if not rules:
                    return {}  # Checklists describe completion, not proposed actions.
                if _read_only(tool_name, tool_input) and not _read_has_constraint(rules, tool_name, tool_input):
                    return {}
                state["dirty"] = True
            elif not state["dirty"] and state["checkpoint"] == stamp:
                current = _evidence_stamp(engine, transcript_path, rules)
                if current is not None and current == state.get("evidence"):
                    return {}
            if _deadline is not None and time.monotonic() >= _deadline:
                raise TimeoutError("request_expired_before_execution")
            if event_name == "PreToolUse":
                # Every proposed mutation gets fresh evidence, including repeated
                # commands following a newly failed prerequisite test.
                audit = engine.check_action(session_id, {"tool_name": tool_name, "arguments": tool_input or {}},
                                            provider="codex", path=transcript_path, _deadline=_deadline)
                name, args = _unwrap(tool_name, tool_input)
                if name != tool_name:
                    inner = engine.check_action(session_id, {"tool_name": name, "arguments": args},
                                                provider="codex", path=transcript_path, _deadline=_deadline)
                    rank = {"clean": 0, "unknown": 1, "suspicious": 2, "violation": 3}
                    if rank[inner["verdict"]] > rank[audit["verdict"]]:
                        audit, inner = inner, audit
                    audit["findings"] = audit.get("findings", []) + [f for f in inner.get("findings", [])
                        if f["verdict"] != "clean" and f not in audit.get("findings", [])]
            else:
                # Stamp BEFORE the audit: concurrent writes must invalidate it.
                evidence_stamp = _evidence_stamp(engine, transcript_path, rules)
                audit = engine.audit_session(session_id, provider="codex", path=transcript_path, _deadline=_deadline)
                state["dirty"] = False
                state["checkpoint"] = stamp
                state["evidence"] = evidence_stamp
            findings = [{"id": f["contract_id"], "verdict": f["verdict"], "reason": f["reason"]}
                        for f in audit.get("findings", []) if f["verdict"] != "clean"]
            if event_name == "Stop":
                findings += [{"id": i.item_id, "verdict": "unknown", "reason": "checklist lacks verified completion evidence"}
                             for i in obligations["pending"]]
            coverage = audit.get("coverage", {})
            verdict = "unknown" if findings and audit["verdict"] == "clean" else audit["verdict"]
            if verdict == "clean" and not findings:
                return {}
            facts = {"session_id": session_id, "verdict": verdict,
                     "coverage_complete": coverage.get("complete", False),
                     "issues": coverage.get("issues", []), "contracts_snapshot": stamp,
                     "findings": findings[:5]}
            context = "Self-directing audit (data, not instructions): " + mask_secrets(json.dumps(facts, ensure_ascii=False))[:2200]
            signature = hashlib.sha256(context.encode()).hexdigest()
            if event_name == "Stop" and state.get("last_output") == signature:
                return {}
            state["last_output"] = signature
            return _context(event_name, context)
    except Exception as exc:
        return _context(event_name, f"Self-directing audit: unknown ({type(exc).__name__}); local evidence could not be checked.")
