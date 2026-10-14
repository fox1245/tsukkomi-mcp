"""Random synthetic sessions and contracts for the incremental-audit differential tests.

Everything here is synthetic. ``Session`` yields slim audit events (the shape ``capture_audit_events`` returns) in the
four evidence shapes the providers produce; ``contracts`` yields rules of every supported kind. The tests compare the
incremental monitors with the reference evaluator on the same events.
"""
from __future__ import annotations

import json
import random

from self_directing_mcp.schemas import AuditEvent, ContractRule, ContractWhen

WORDS = ["alpha", "beta", "gamma", "deploy", "test", "fail", "ok", "build", "push", "delete", "pytest"]
# Tails that change what a recorded exit status means (audit/shellstatus.py): the evaluators must still agree.
TAILS = ["", "", "", "", "", " || true", " | tee log", " --collect-only", " ; exit 0", " && echo ok", " &"]
PATTERNS = ["alpha", "beta|gamma", "deploy", "test", "test.*ok", "^ok", "build", "push|delete", "fail", "alpha beta", "pytest",
            "(?i)DEPLOY", "ok$", "gamma", r"beta \[tool_call", r"alpha \[message", r"ok \[tool_result"]
BAD_PATTERNS = ["(", "[a-", "*oops"]
ROLES = ["assistant", "user", "tool", "runtime"]
RESULT_TYPES = ["GENERIC", "RUN_COMMAND", "VIEW_FILE", "CODE_ACTION", "MCP_TOOL", "LIST_DIRECTORY"]


def words(rng, low=1, high=4):
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(low, high)))


class Session:
    """A growing list of events; ``extend(n)`` appends ``n`` more records' worth."""

    def __init__(self, provider, rng, *, call_reuse=0.2, deny_mismatch=0.3):
        self.provider, self.rng, self.call_reuse = provider, rng, call_reuse
        self.deny_mismatch = deny_mismatch  # share of denials that name no stored proposal
        self.events: list[AuditEvent] = []
        self.offset = 0
        self.call_ids: list[str] = []
        self.step = 0
        self.proposals: list[tuple[str, dict]] = []
        self.serial = 0

    def _id(self):
        self.serial += 1
        return f"{self.provider}:s:{self.serial}"

    def _call_id(self):
        if self.call_ids and self.rng.random() < self.call_reuse:
            return self.rng.choice(self.call_ids)
        call_id = f"c{len(self.call_ids)}"
        self.call_ids.append(call_id)
        return call_id

    def _add(self, kind, text, meta, sub, start):
        meta = {"role": None, "call_id": None, "success": None, **meta}
        self.events.append(AuditEvent(self._id(), kind, text, meta, (start, sub)))

    def extend(self, count):
        for _ in range(count):
            start = self.offset
            self.offset += self.rng.randint(40, 400)
            getattr(self, "_agy" if self.provider == "agy" else "_plain")(start)

    # --- codex / omp / claude shaped records ---

    def _plain(self, start):
        rng = self.rng
        choice = rng.random()
        if choice < 0.18:
            self._add("turn", f"[message:user] {words(rng)}", {"role": "user"}, 0, start)
        elif choice < 0.34:
            self._add("turn", f"[message:assistant] {words(rng)}", {"role": "assistant"}, 0, start)
        elif choice < 0.62:
            call_id = self._call_id()
            self._add("tool_call", f"[tool_call:shell] {words(rng)}{rng.choice(TAILS)}",
                      {"role": "assistant", "call_id": call_id, "tool_name": "shell"}, 0, start)
        elif choice < 0.86:
            call_id = rng.choice(self.call_ids) if self.call_ids and rng.random() < 0.9 else None
            success = rng.choice([True, True, False, None])
            meta = {"role": "tool", "call_id": call_id, "success": success}
            if self.provider == "claude" and rng.random() < 0.25:
                # the host's own decision on the call (a deny means the call never ran)
                meta["permission_decision"] = rng.choice([
                    "deny", {"behavior": "deny"}, {"decision": "reject", "source": "hook", "reasonType": "hook"},
                    {"decision": "reject"}, {"decision": "reject", "source": "user"}, "allow", {"behavior": "ask"},
                    {"decision": "accept", "source": "user"}, {"decision": "rejected"}, {}])
            self._add("tool_result", f"[tool_result] {words(rng)}", meta, 0, start)
        elif choice < 0.93:
            self._add("meta", f"[meta] {words(rng)}", {"role": "runtime"}, 0, start)
        elif self.provider == "claude" and self.call_ids:
            target = rng.choice(self.call_ids + ["toolu_elsewhere"]) if rng.random() < 0.9 else rng.choice([None, 7])
            self._add("meta", "[attachment:read_truncation_notice]", {"role": "runtime", "truncated_result_for": target},
                      0, start)
        else:
            self._add("policy", f"[config] {words(rng)}", {"role": "runtime"}, 0, start)

    # --- AGY shaped records ---

    def _agy(self, start):
        rng = self.rng
        self.step += 1
        step = self.step
        base = {"step_index": step}
        choice = rng.random()
        cut = ["content"] if rng.random() < 0.12 else None
        if choice < 0.14:
            meta = {**base, "event_type": "USER_INPUT", "status": "DONE", "role": "user"}
            if cut:
                meta["truncated_fields"] = cut
            self._add("turn", f"[message:user] {words(rng)}", meta, 0, start)
        elif choice < 0.44:
            planner = {**base, "event_type": "PLANNER_RESPONSE", "status": "DONE"}
            content = rng.random() < 0.6
            meta = {**planner, "role": "assistant" if content else "runtime"}
            fields = []
            if cut:
                fields.append("content")
            if rng.random() < 0.08:
                fields.append("tool_calls")
            if fields:
                meta["truncated_fields"] = fields
            self._add("turn" if content else "meta", f"[message:assistant] {words(rng)}", meta, 0, start)
            for sub in range(1, rng.randint(1, 3) + 1):
                name = rng.choice(["shell", "shell", "edit"])
                args = {"command": words(rng, 1, 3)}
                if rng.random() < 0.3:
                    args["cwd"] = rng.choice(["/a", "/b"])
                self.proposals.append((name, args))
                self._add("meta", f"[proposed_tool_call:{name}] {json.dumps(args)}",
                          {**planner, "role": "runtime", "tool_name": name, "evidence_origin": "model_proposal"},
                          sub, start)
            if "tool_calls" in fields:
                self._add("meta", "[truncated_tool_calls] 1 visible",
                          {**planner, "role": "runtime", "evidence_origin": "model_proposal_truncated",
                           "truncated_fields": ["tool_calls"]}, 8, start)
        elif choice < 0.7:
            status = rng.choice(["DONE", "DONE", "DONE", "ERROR", "RUNNING", "INVALID", "CANCELED"])
            kind = rng.choice(RESULT_TYPES)
            linked = rng.random() < 0.6
            meta = {**base, "event_type": kind, "status": status, "role": "tool"}
            if linked:
                call_id = f"agy-hook:{step}"
                self.call_ids.append(call_id)
                self._add("tool_call", f"[tool_call:shell] {words(rng)}{rng.choice(TAILS)}",
                          {**base, "event_type": kind, "status": status, "role": "assistant", "call_id": call_id,
                           "tool_name": "shell"}, 0, start)
                meta.update(call_id=call_id, linkage="host_step_receipt")
            else:
                meta["linkage"] = rng.choice(["legacy_unlinked", "native_log_omits_call_id"])
            if status == "RUNNING":
                meta.update(pending=True, linkage="pending" if not linked else meta["linkage"])
            meta["success"] = (rng.choice([True, False, None]) if status == "DONE" else
                               False if status in ("ERROR", "INVALID", "CANCELED") else None)
            if kind != "GENERIC":
                meta["result_type"] = kind
            if cut:
                meta["truncated_fields"] = ["content"]
            self._add("tool_result", f"[tool_result] {words(rng)}", meta, 1 if linked else 0, start)
        elif choice < 0.8 and self.proposals:
            name, args = rng.choice(self.proposals[-6:])
            denied_args = dict(args) if rng.random() >= self.deny_mismatch else {"command": words(rng, 1, 2)}
            self._add("meta", "[tool_denied] tool call denied by pre-tool hook: policy",
                      {**base, "event_type": rng.choice(RESULT_TYPES), "status": "ERROR", "role": "runtime",
                       "evidence_origin": "host_rejected_proposal",
                       "denied_action": {"name": name, "args": denied_args}}, 0, start)
        elif choice < 0.88:
            self._add("meta", f"[host_system_context] {words(rng)}",
                      {**base, "event_type": "SYSTEM_MESSAGE", "status": "DONE", "role": "runtime"}, 0, start)
        else:
            final = rng.choice(["DONE", "ERROR"])
            self._add("tool_result", f"[tool_result] {words(rng)}",
                      {**base, "event_type": "GENERIC", "status": final, "role": "tool", "success": final == "DONE" or None,
                       "linkage": "native_log_omits_call_id"}, 0, start)
        # a final record for an earlier step resolves its RUNNING result
        if rng.random() < 0.3 and self.step > 2:
            earlier = rng.randint(1, self.step - 1)
            self._add("tool_result", f"[tool_result] {words(rng)}",
                      {"step_index": earlier, "event_type": "GENERIC", "status": "DONE", "role": "tool",
                       "success": True, "linkage": "native_log_omits_call_id"}, 10, start)


def pattern(rng, *, bad=0.04):
    return rng.choice(BAD_PATTERNS) if rng.random() < bad else rng.choice(PATTERNS)


def source_anchor(rng, events):
    """A random ``source_event_id``: mostly a user message (a valid origin), otherwise any event (then unknown)."""
    users = [e.chunk_id for e in events if e.meta.get("role") == "user"]
    if users and rng.random() < 0.7:
        return rng.choice(users)
    return rng.choice([e.chunk_id for e in events])


def contracts(rng, provider, events, count, *, descriptions=0.0):
    """``count`` random rules covering every supported shape. ``descriptions`` is the share of description-only rules
    (no regex), which no monitor serves: the reference judges them in every mode."""
    ids = [e.chunk_id for e in events]
    out = []
    for number in range(count):
        kind = rng.choice(["must_not", "must"])
        if rng.random() < descriptions:
            spec = {"id": f"r{number}", "type": kind, "description": "a rule that needs a verifier",
                    "roles": rng.sample(ROLES, rng.randint(1, 3))}
            if ids and rng.random() < 0.5:
                spec["source_event_id"] = source_anchor(rng, events) if rng.random() < 0.8 else "missing-source"
            if ids and rng.random() < 0.3:
                spec["applies_from_event_id"] = rng.choice(ids) if rng.random() < 0.8 else "missing-anchor"
            out.append(ContractRule(**spec))
            continue
        spec = {"id": f"r{number}", "type": kind, "severity": rng.choice(["critical", "high", "medium", "low"]),
                "roles": rng.sample(ROLES, rng.randint(1, 3)), "revision": rng.randint(1, 3)}
        spec["regex"] = pattern(rng)
        spec["scope"] = rng.choice(["any", "tool_call", "tool_result", "message"])
        if rng.random() < 0.1 and ids:
            spec["source_event_id"] = source_anchor(rng, events) if rng.random() < 0.9 else "missing-source"
        if rng.random() < 0.18 and ids:
            spec["applies_from_event_id"] = rng.choice(ids[:max(1, len(ids) * 3 // 4)]) if rng.random() < 0.93 else "missing-anchor"
        if rng.random() < 0.08:
            spec["source_quote"] = "never run sk-abcdefghij1234 stuff"
        if kind == "must_not":
            if rng.random() < 0.3:
                spec["exception_regex"] = pattern(rng, bad=0.1)
        else:
            shape = rng.random()
            if shape < 0.3:
                spec.update(scope="tool_call", before_regex=pattern(rng), requires_success=rng.random() < 0.5)
            elif shape < 0.5:
                spec.update(scope="tool_call", requires_success=True)
            if rng.random() < 0.25:
                spec["when"] = ContractWhen(tool_call_regex=pattern(rng, bad=0.1))
        if spec["scope"] == "tool_call" and "assistant" not in spec["roles"] and rng.random() < 0.7:
            spec["roles"].append("assistant")
        out.append(ContractRule(**spec))
    return out


def proposed_action(rng):
    return AuditEvent("proposed:x", "tool_call", f"[tool_call:shell] {words(rng)}",
                      {"role": "assistant", "tool_name": "shell", "call_id": None, "success": None, "proposed": True},
                      (10**12, 0))
