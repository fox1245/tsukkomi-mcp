"""Synthetic Claude Code sessions and evaluator-pair helpers for the stage-1 semantics tests.

Everything here is synthetic. ``Claude`` writes a session transcript and indexes it; ``evaluate`` runs one set of
rules over a list of slim events through BOTH evaluators (the reference ``runner.full_findings`` and the incremental
monitors) and fails if they differ, so a test states each semantic once and covers both implementations.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from self_directing_mcp.audit import incremental as inc
from self_directing_mcp.audit import runner
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.schemas import AuditEvent, ContractRule

SID = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"
PROJECT = "-synthetic-project"
_counter = itertools.count(1)


# ---------------------------------------------------------------- Claude Code transcript records

def env_record(kind, *, sid=SID, **extra):
    n = next(_counter)
    record = {"type": kind, "uuid": f"00000000-0000-4000-8000-{n:012d}", "parentUuid": None, "sessionId": sid,
              "timestamp": f"2026-10-01T00:{(n // 60) % 60:02d}:{n % 60:02d}.000Z", "cwd": "/synthetic/project",
              "isSidechain": False, "userType": "external", "version": "9.9.9"}
    record.update(extra)
    return record


def human(text, **extra):
    return env_record("user", message={"role": "user", "content": text}, origin={"kind": "human"},
                      promptId="prompt-1", **extra)


def injected(text, **extra):
    """A user-role record the host injected (meta): runtime evidence, never the user."""
    return env_record("user", message={"role": "user", "content": text}, isMeta=True, **extra)


def assistant(*blocks, **extra):
    message = {"id": "msg_synthetic_1", "role": "assistant", "model": "claude-synthetic", "content": list(blocks),
               "stop_reason": "end_turn", "stop_sequence": None}
    return env_record("assistant", message=message, requestId="req_synthetic_1", **extra)


def text(value):
    return {"type": "text", "text": value}


def use(call_id, name, **arguments):
    return {"type": "tool_use", "id": call_id, "name": name, "input": arguments}


def bash(call_id, command, description=None, **extra):
    arguments = {"command": command, **({"description": description} if description is not None else {})}
    return assistant(use(call_id, "Bash", **arguments), **extra)


def tool_result(call_id, content="ok", *, is_error=False, **extra):
    block = {"type": "tool_result", "tool_use_id": call_id, "content": content, "is_error": is_error}
    return env_record("user", message={"role": "user", "content": [block]},
                      sourceToolAssistantUUID="00000000-0000-4000-8000-0000000000aa", **extra)


def denied_result(call_id, decision="deny", content="blocked by a hook"):
    """The host's record of a refused call: the result carries the permission decision."""
    return tool_result(call_id, content, is_error=True, permissionDecision=decision)


# The record Claude Code 2.1.299 wrote when a PreToolUse hook answered ``deny`` for a Bash call (reproduced by hand from
# a synthetic probe; the field names and values are the recorded ones, the text is invented).
HOOK_REJECT = {"decision": "reject", "source": "hook", "reasonType": "hook"}


def hook_denied_result(call_id, reason="never rm -rf", decision=None, kind="permission-rule"):
    """The tool-result record of a call refused by a hook, as Claude Code 2.1.299 records it: the host's decision is a
    top-level ``permissionDecision`` object beside ``toolDenialKind``, and the block is an error result."""
    extra = {"permissionDecision": HOOK_REJECT if decision is None else decision}
    if kind is not None:
        extra["toolDenialKind"] = kind
    return tool_result(call_id, f"PreToolUse:Bash hook error: {reason}", is_error=True, **extra)


def write(path, *records, mode="w"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    return path


class Claude:
    """A Claude Code projects root, one session file and an engine over a private index."""

    def __init__(self, tmp_path, **settings):
        self.root = Path(tmp_path) / "projects"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / PROJECT / f"{SID}.jsonl"
        self.engine = SelfDirectEngine(Settings(
            _env_file=None, local_only=True, index_dir=Path(tmp_path) / "index", claude_projects_dir=self.root,
            openrouter_api_key_file=None, **settings))

    def write(self, *records, mode="w"):
        write(self.path, *records, mode=mode)
        return self

    def append(self, *records):
        return self.write(*records, mode="a")

    def sync(self):
        return self.engine.sync_session(session_id=SID, provider="claude", path=str(self.path), embed=False)

    def chunks(self):
        return self.engine.store.list_chunks(SID, "claude")

    def upsert(self, *rules):
        return self.engine.upsert_contracts([dict(rule) for rule in rules])

    def audit(self):
        return self.engine.audit_session(SID, provider="claude", path=str(self.path))

    def check(self, tool_name, arguments):
        return self.engine.check_action(SID, {"tool_name": tool_name, "arguments": arguments},
                                        provider="claude", path=str(self.path))

    def close(self):
        self.engine.close()


def finding(result, contract_id):
    return next(item for item in result["findings"] if item["contract_id"] == contract_id)


# ---------------------------------------------------------------- both evaluators on slim events

_serial = itertools.count(1)


def event(kind, text_, role, *, call_id=None, success=None, timestamp=None, **meta):
    """A slim audit event as ``capture_audit_events`` produces it; the key is its place in the session.
    ``timestamp`` is kept the way ``capture_audit_events(..., times=True)`` keeps it."""
    number = next(_serial)
    return AuditEvent(f"claude:s:{number}", kind, text_, {"role": role, "call_id": call_id, "success": success, **meta},
                      (number * 100, 0), timestamp=timestamp)


def user_turn(text_="please do it", **meta):
    return event("turn", f"[message:user] {text_}", "user", **meta)


def call(command, call_id, *, name="Bash", **meta):
    return event("tool_call", f"[tool_call:{name}] " + json.dumps({"command": command}), "assistant",
                 call_id=call_id, tool_name=name, **meta)


def result(call_id, success, text_="done", **meta):
    return event("tool_result", f"[tool_result] {text_}", "tool", call_id=call_id, success=success, **meta)


def evaluate(rules, events, provider="claude", *, proposed=None, chunk_sizes=(None, 1, 3), anchors=None):
    """Findings of ``rules`` over ``events`` by the reference; every monitor feed must agree exactly.

    ``chunk_sizes``: how the events are fed to the monitors (all at once, one by one, in threes), and each time the
    state is round-tripped through JSON between feeds as the store does. ``anchors``: ``{event id: audit.anchors.Anchor}``
    for the anchors of global rules that live in another session (the engine resolves them from the index).
    """
    rules = [ContractRule(**rule) if isinstance(rule, dict) else rule for rule in rules]
    expected = runner.full_findings(rules, list(events), session_id="s", store=None, provider=provider,
                                    proposed=proposed, coverage={}, anchors=anchors)
    needs, _link = runner.audit_needs(rules, proposed)
    for size in chunk_sizes:
        monitors = {r.id: inc.RuleMonitor(r, provider, anchors) for r in rules}
        facts = inc.make_facts(provider, needs, rules)
        step = size or max(1, len(events))
        for start in range(0, len(events), step):
            segment = list(events[start:start + step])
            pools = _prior(events[:start])
            inc.fold(segment, [(m, 0) for m in monitors.values()], (facts, 0) if facts.needed else None, pools)
            monitors, facts = _reload(monitors, facts, provider, rules, needs, anchors)
        for rule, want in zip(rules, expected):
            got = monitors[rule.id].finding(proposed, facts)
            assert got.model_dump() == want.model_dump(), (
                f"monitor and reference differ for {rule.id} (feed={size})\nmonitor  ={got.model_dump()}\n"
                f"reference={want.model_dump()}")
    return expected


def _prior(previous, lookback=4096):
    """What the store would prefetch for the events before a feed: calls by id and proposals by tool name."""
    calls, proposals = {}, {}
    for item in previous:
        call_id = item.meta.get("call_id")
        if call_id is not None and item.kind in ("tool_call", "tool_result"):
            calls.setdefault(inc._k(call_id), []).append((item.kind, item.key, item.chunk_id))
        if item.meta.get("evidence_origin") == "model_proposal":
            proposals.setdefault(item.meta.get("tool_name"), []).append((item.key, item.text))
    pools = inc.Pools(track_calls=True, track_proposals=True, calls_prior=calls)
    complete = len(previous) <= lookback
    for name in proposals:
        pools.proposals_prior[name] = ([(e.key, e.text) for e in reversed(previous[-lookback:])
                                        if e.meta.get("evidence_origin") == "model_proposal"
                                        and e.meta.get("tool_name") == name], complete)
    return pools


def _reload(monitors, facts, provider, rules, needs, anchors=None):
    by_id = {r.id: r for r in rules}
    fresh = {}
    for rule_id, monitor in monitors.items():
        copy = inc.RuleMonitor(by_id[rule_id], provider, anchors)
        copy.load(json.loads(json.dumps(monitor.dump())))
        fresh[rule_id] = copy
    shared = inc.make_facts(provider, needs, rules)
    shared.load(json.loads(json.dumps(facts.dump())))
    return fresh, shared


def one(rule, events, provider="claude", **kwargs):
    """The single finding of one rule, with both evaluators checked against each other."""
    return evaluate([rule], events, provider, **kwargs)[0]


# ---------------------------------------------------------------- a user anchor in a Codex rollout

def codex_user_anchor(engine, path, sid, text="please make that change to the rule", provider="codex"):
    """Append a user message to a Codex rollout, index it and return its event id: a valid contract anchor."""
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"type": "response_item", "payload": {
            "type": "message", "role": "user", "content": text}}) + "\n")
    engine.sync_session(session_id=sid, provider=provider, path=str(path), embed=False)
    return next(c.chunk_id for c in reversed(engine.store.list_chunks(sid, provider)) if c.meta.get("role") == "user")
