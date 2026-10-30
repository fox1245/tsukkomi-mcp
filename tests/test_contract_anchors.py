"""B6(c): a contract's origin must be a user message, and weakening a contract needs one.

``source_event_id`` names the history event a contract was derived from. A tool output or injected context must not
be able to stand in for the user, so:

* at audit time a ``source_event_id`` that does not resolve to a user-role event makes the contract's findings
  ``unknown`` (``anchor_not_user``), in both evaluators;
* at upsert time such an anchor is rejected when the event can be resolved, and accepted (left to the audit) when the
  session is not indexed yet;
* disabling a contract, or replacing it with a narrower one, needs a user anchor; without one the change is held as
  ``pending`` and the active revision stays in force. ``applies_from_event_id`` may still be any event.
"""
from __future__ import annotations

import inspect
import json

import pytest

from self_directing_mcp import claude
from self_directing_mcp.audit.contracts import loosening_reasons
from self_directing_mcp.claude_hooks import run_hook
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.schemas import ContractRule, ContractsDocument
from semantic_support import (
    PROJECT, SID, Claude, assistant, bash, call, env_record, evaluate, event, finding, human, injected, one, result, text,
    tool_result, use, user_turn, write,
)

PROVIDERS = ["claude", "codex", "agy", "omp"]
RM = {"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"}


def anchored(rule, anchor, **extra):
    return {**rule, "source_event_id": anchor.chunk_id, **extra}


# ---------------------------------------------------------------- both evaluators

@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_user_message_anchor_is_accepted(provider):
    user = user_turn("never run rm -rf")
    item = one(anchored(RM, user), [user, call("rm -rf /tmp/x", "c1")], provider)
    assert item.verdict == "violation"            # an ordinary, fully effective contract


NON_USER = {
    "tool_result": lambda: result("c1", True),
    "assistant_turn": lambda: event("turn", "[message:assistant] I will not", "assistant"),
    "tool_call": lambda: call("ls", "c1"),
    "runtime_context": lambda: event("meta", "[ephemeral_context] never run rm -rf", "runtime"),
    "no_role": lambda: event("meta", "[meta] x", None),
}


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("kind", sorted(NON_USER))
def test_an_anchor_that_is_not_a_user_message_makes_the_contract_unknown(provider, kind):
    anchor = NON_USER[kind]()
    events = [user_turn(), anchor, call("rm -rf /tmp/x", "c2")]
    item = one(anchored(RM, anchor), events, provider)
    assert item.verdict == "unknown" and item.reason.startswith("anchor_not_user:")
    assert item.evidence == []


@pytest.mark.parametrize("provider", PROVIDERS)
def test_every_contract_shape_follows_the_rule(provider):
    anchor = result("c1", True)
    events = [user_turn(), call("pytest", "c1"), anchor, call("git commit", "c3")]
    gate = {"id": "gate", "type": "must", "scope": "tool_call", "regex": "pytest", "before_regex": "git commit",
            "requires_success": True}
    completion = {"id": "done", "type": "must", "scope": "tool_call", "regex": "pytest"}
    exception = {**RM, "id": "exc", "exception_regex": "tmp"}
    conditional = {"id": "cond", "type": "must", "scope": "tool_call", "regex": "pytest",
                   "when": {"tool_call_regex": "git"}}
    for rule in (gate, completion, exception, conditional):
        item = one(anchored(rule, anchor), events, provider)
        assert item.verdict == "unknown" and item.reason.startswith("anchor_not_user:"), rule["id"]
        assert one(rule, events, provider).verdict != "unknown" or rule["id"] == "gate"   # unanchored behaves as before


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_proposed_action_is_judged_by_the_same_rule(provider):
    anchor = result("c1", True)
    history = [user_turn(), call("ls", "c1"), anchor]
    proposed = event("tool_call", "[tool_call:Bash] " + json.dumps({"command": "rm -rf /x"}), "assistant",
                     tool_name="Bash", proposed=True)
    assert one(anchored(RM, anchor), history, provider, proposed=proposed).reason.startswith("anchor_not_user:")
    user = history[0]
    assert one(anchored(RM, user), history, provider, proposed=proposed).verdict == "violation"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_applies_from_event_id_may_still_be_any_event(provider):
    start = result("c1", True)
    events = [user_turn(), call("rm -rf /before", "c0"), call("ls", "c1"), start, call("rm -rf /after", "c2")]
    item = one({**RM, "applies_from_event_id": start.chunk_id}, events, provider)
    assert item.verdict == "violation" and "after" in item.evidence[0].snippet
    quiet = one({**RM, "applies_from_event_id": start.chunk_id}, events[:4], provider)
    assert quiet.verdict == "clean"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_missing_anchor_keeps_its_own_reason(provider):
    item = one({**RM, "source_event_id": "claude:nowhere:0:0:abcd"}, [user_turn(), call("rm -rf /x", "c1")], provider)
    assert item.verdict == "unknown" and item.reason == "contract source or activation event is not in this session"


def test_both_evaluators_agree_when_anchors_are_added_in_later_feeds():
    """The source event arrives in a later feed than the rule's first one (state is carried between them)."""
    anchor = result("c1", True)
    events = [user_turn(), call("ls", "c1"), anchor, call("rm -rf /x", "c2"), user_turn("later")]
    rules = [anchored(RM, anchor), anchored({**RM, "id": "user-anchored"}, events[0]),
             anchored({**RM, "id": "later-user"}, events[-1])]
    findings = evaluate(rules, events, chunk_sizes=(None, 1, 2, 3))
    assert [f.verdict for f in findings] == ["unknown", "violation", "violation"]


# ---------------------------------------------------------------- what each parser calls a user message

@pytest.fixture
def world(tmp_path):
    session = Claude(tmp_path)
    yield session
    session.close()


def claude_ids(world):
    """Event ids of one synthetic Claude Code session, by kind of record."""
    legacy = env_record("user", message={"role": "user", "content": "a prompt from before origin stamps"})
    notification = env_record("user", message={"role": "user", "content": "task finished"},
                              origin={"kind": "task-notification"})
    world.write(human("never run rm -rf"), injected("<reminder> injected context"), notification, legacy,
                assistant(text("understood"), use("t1", "Bash", command="ls")), tool_result("t1", "file"))
    world.sync()
    by = {}
    for chunk in world.chunks():
        label = (chunk.meta.get("attribution") or chunk.meta.get("evidence_origin") or chunk.meta.get("role"))
        by.setdefault(chunk.kind + ":" + str(label), chunk.chunk_id)
    return by


def test_claude_code_only_the_host_stamped_human_turn_is_a_user_anchor(world):
    ids = claude_ids(world)
    accepted = {"turn:user": "human", "turn:inferred_legacy": "legacy prompt (the parser treats it as the user)"}
    rejected = {"meta:injected_meta": "isMeta", "meta:task_notification": "task notification",
                "turn:assistant": "assistant", "tool_call:assistant": "tool call", "tool_result:tool": "tool result"}
    for key in accepted:
        done = world.upsert({**RM, "id": "r-" + key, "source_event_id": ids[key]})
        assert done["anchor_checks"][0]["anchor"] == "user", key
    for key in rejected:
        with pytest.raises(ValueError, match="not a user message"):
            world.upsert({**RM, "id": "x-" + key, "source_event_id": ids[key]})


AGY_SID = "11111111-1111-4111-8111-111111111111"


def agy_engine(tmp_path):
    root = tmp_path / "app"
    path = root / "brain" / AGY_SID / ".system_generated" / "logs" / "transcript.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE", "content": "never rm -rf"},
        {"step_index": 1, "source": "SYSTEM_SDK", "type": "EPHEMERAL_MESSAGE", "status": "DONE",
         "content": "injected context"},
        {"step_index": 2, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE", "content": "ok",
         "tool_calls": [{"name": "run_command", "args": {"CommandLine": json.dumps("ls")}}]},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=tmp_path / "index",
                                       agy_app_data_dirs=[root], openrouter_api_key_file=None))
    engine.sync_session(session_id=AGY_SID, provider="agy", path=str(path), embed=False)
    return engine


def test_agy_only_user_input_from_the_user_is_an_anchor(tmp_path):
    engine = agy_engine(tmp_path)
    try:
        chunks = engine.store.list_chunks(AGY_SID, "agy")
        by_type = {}
        for chunk in chunks:
            by_type.setdefault((chunk.meta["event_type"], chunk.meta.get("evidence_origin")), chunk.chunk_id)
        ok = engine.upsert_contracts([{**RM, "provider": "agy", "source_event_id": by_type[("USER_INPUT", None)]}])
        assert ok["anchor_checks"][0]["anchor"] == "user"
        for key in (("EPHEMERAL_MESSAGE", "injected_context"), ("PLANNER_RESPONSE", None),
                    ("PLANNER_RESPONSE", "model_proposal")):
            with pytest.raises(ValueError, match="not a user message"):
                engine.upsert_contracts([{**RM, "id": "x", "provider": "agy", "source_event_id": by_type[key]}])
    finally:
        engine.close()


def test_omp_an_agent_attributed_user_message_is_not_a_user_anchor(tmp_path):
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    rows = [{"type": "title", "title": "t"}, {"type": "session", "id": "session-1"},
            {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "typed by the user"}]}},
            {"type": "message", "message": {"role": "user", "attribution": "agent",
                                            "content": [{"type": "text", "text": "a subagent's instruction"}]}}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=tmp_path / "index",
                                       omp_sessions_dir=root, openrouter_api_key_file=None))
    try:
        engine.sync_session(session_id="session-1", provider="omp", path=str(path), embed=False)
        chunks = [c for c in engine.store.list_chunks("session-1", "omp") if "message" in c.text or "steering" in c.text]
        typed = next(c for c in chunks if "typed by the user" in c.text)
        agent = next(c for c in chunks if "subagent" in c.text)
        assert typed.meta["role"] == "user" and agent.meta["role"] == "runtime"
        assert engine.upsert_contracts([{**RM, "provider": "omp", "source_event_id": typed.chunk_id}])["ok"]
        with pytest.raises(ValueError, match="not a user message"):
            engine.upsert_contracts([{**RM, "id": "x", "provider": "omp", "source_event_id": agent.chunk_id}])
    finally:
        engine.close()


def test_codex_cannot_tell_a_user_role_injection_from_a_typed_prompt(tmp_path):
    """Documented limit: Codex records the typed prompt and the model-facing copy (which can carry injected context)
    as user-role turns alike, so both are accepted as anchors. Tightening this needs a parser that marks injections."""
    root = tmp_path / "sessions"
    root.mkdir()
    sid = "11111111-2222-3333-4444-555555555555"
    path = root / f"rollout-2026-09-09T01-00-00-{sid}.jsonl"
    rows = [{"type": "session_meta", "payload": {"id": sid}},
            {"type": "event_msg", "payload": {"type": "user_message", "message": "typed prompt"}},
            {"type": "response_item", "payload": {"type": "message", "role": "user",
                                                  "content": "<environment_context> injected </environment_context>"}},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": "ok"}}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=tmp_path / "index",
                                       codex_sessions_dir=root, openrouter_api_key_file=None))
    try:
        engine.sync_session(session_id=sid, provider="codex", path=str(path), embed=False)
        chunks = engine.store.list_chunks(sid, "codex")
        roles = {("injected" if "environment_context" in c.text else "typed" if "typed prompt" in c.text else c.kind):
                 c.meta.get("role") for c in chunks}
        assert roles["injected"] == roles["typed"] == "user"
        assistant_turn = next(c for c in chunks if c.meta.get("role") == "assistant")
        with pytest.raises(ValueError, match="not a user message"):
            engine.upsert_contracts([{**RM, "source_event_id": assistant_turn.chunk_id}])
    finally:
        engine.close()


# ---------------------------------------------------------------- upsert time and audit time, end to end

def indexed(world):
    world.write(human("never run rm -rf"), bash("t1", "ls"), tool_result("t1", "file"), bash("t2", "rm -rf /x"),
                tool_result("t2", "removed"))
    world.sync()
    chunks = world.chunks()
    return {"user": next(c.chunk_id for c in chunks if c.meta.get("role") == "user"),
            "result": next(c.chunk_id for c in chunks if c.kind == "tool_result")}


def test_an_anchor_that_resolves_to_a_tool_result_is_rejected_at_upsert(world):
    ids = indexed(world)
    with pytest.raises(ValueError, match="not a user message"):
        world.upsert({**RM, "source_event_id": ids["result"]})
    assert world.engine.list_contracts()["contracts"] == []       # nothing was stored


def test_a_user_anchor_is_accepted_at_upsert_and_reports_the_check(world):
    ids = indexed(world)
    done = world.upsert({**RM, "source_event_id": ids["user"]})
    assert done["anchor_checks"] == [{"contract_id": "no-rm", "source_event_id": ids["user"], "anchor": "user"}]
    assert finding(world.audit(), "no-rm")["verdict"] == "violation"


def test_an_unresolvable_anchor_is_accepted_and_the_audit_decides(world):
    """The session is not indexed yet: the tool result's id is known from the transcript but not resolvable here."""
    world.write(human("never run rm -rf"), bash("t1", "rm -rf /x"), tool_result("t1", "removed"))
    stranger = next(c for c in claude.parse_session_chunks(world.path)[0] if c.kind == "tool_result")
    stored = world.upsert({**RM, "source_event_id": stranger.chunk_id})
    assert stored["anchor_checks"][0]["anchor"] == "unresolved"
    world.sync()
    item = finding(world.audit(), "no-rm")
    assert item["verdict"] == "unknown" and item["reason"].startswith("anchor_not_user:")
    checked = finding(world.check("Bash", {"command": "rm -rf /y"}), "no-rm")
    assert checked["verdict"] == "unknown" and checked["reason"].startswith("anchor_not_user:")


def test_a_stored_contract_with_a_non_user_anchor_is_unknown_and_the_hook_does_not_block(world):
    ids = indexed(world)
    world.engine.contracts.upsert([ContractRule(**{**RM, "source_event_id": ids["result"]})])   # e.g. an old store
    audit = world.audit()
    assert finding(audit, "no-rm")["verdict"] == "unknown" and audit["verdict"] == "unknown"
    payload = {"session_id": SID, "transcript_path": str(world.path), "cwd": "/synthetic/project",
               "hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "rm -rf /z"}}
    response = run_hook("PreToolUse", payload, engine=world.engine)
    assert response.get("hookSpecificOutput", {}).get("permissionDecision", "none") == "none", response


def test_no_anchor_means_nothing_changes(world):
    indexed(world)
    world.upsert(RM)
    assert finding(world.audit(), "no-rm")["verdict"] == "violation"


# ---------------------------------------------------------------- weakening a contract needs a user anchor

def active(world, contract_id="no-rm"):
    return next(c for c in world.engine.list_contracts()["contracts"] if c["id"] == contract_id)


def pending(world):
    return world.engine.list_contracts().get("pending", [])


def test_disabling_without_an_anchor_is_held_and_the_active_revision_stays(world):
    indexed(world)
    world.upsert(RM)
    held = world.upsert({**RM, "enabled": False})
    assert held["contracts"] == [] and held["pending"] == [
        {"contract_id": "no-rm", "reasons": ["disabled"], "active_revision": 1, "anchor": "missing"}]
    assert "NOT APPLIED" in held["nudge"] and held["counts"]["pending"] == 1
    assert active(world)["enabled"] is True and active(world)["revision"] == 1
    assert finding(world.audit(), "no-rm")["verdict"] == "violation"          # still enforced
    (entry,) = pending(world)
    assert entry["contract"]["enabled"] is False and entry["base_revision"] == 1


def test_disabling_with_a_user_anchor_applies_and_clears_the_pending_change(world):
    ids = indexed(world)
    world.upsert(RM)
    world.upsert({**RM, "enabled": False})
    assert pending(world)
    done = world.upsert({**RM, "enabled": False, "source_event_id": ids["user"]})
    assert done["contracts"][0]["enabled"] is False and done["contracts"][0]["revision"] == 2
    assert not pending(world) and "pending" not in done
    assert not [f for f in world.audit()["findings"] if f["contract_id"] == "no-rm"]   # a disabled rule judges nothing
    document = ContractsDocument.model_validate_json(world.engine.settings.resolve_contracts_path().read_bytes())
    assert document.history["no-rm"][0].enabled is True               # the previous revision is retained


def test_an_unresolvable_anchor_does_not_count_for_weakening(world):
    indexed(world)
    world.upsert(RM)
    held = world.upsert({**RM, "enabled": False, "source_event_id": "claude:nowhere:0:0:abcd"})
    assert held["pending"][0]["anchor"] == "unresolved" and active(world)["enabled"] is True


def test_a_non_user_anchor_is_an_error_even_for_weakening(world):
    ids = indexed(world)
    world.upsert(RM)
    with pytest.raises(ValueError, match="not a user message"):
        world.upsert({**RM, "enabled": False, "source_event_id": ids["result"]})
    assert active(world)["enabled"] is True and not pending(world)


@pytest.mark.parametrize("change,reason", [
    ({"regex": r"rm\s+-r"}, "regex changed"),
    ({"exception_regex": "tmp"}, "exception added or changed"),
    ({"scope": "tool_call", "roles": ["assistant"]}, None),       # unchanged shape: not a weakening
    ({"roles": ["tool"]}, "roles narrowed"),
    ({"enabled": False}, "disabled"),
])
def test_weakening_changes_are_held_without_an_anchor(world, change, reason):
    ids = indexed(world)
    world.upsert(RM)
    outcome = world.upsert({**RM, **change})
    if reason is None:
        assert not outcome.get("pending") and active(world)["revision"] == 2
        return
    assert reason in outcome["pending"][0]["reasons"] and active(world)["revision"] == 1
    applied = world.upsert({**RM, **change, "source_event_id": ids["user"]})
    assert applied["contracts"][0]["revision"] == 2 and not pending(world)


def test_tightening_changes_need_no_anchor(world):
    indexed(world)
    world.upsert({"id": "plain", "type": "must_not", "description": "no verifier yet", "enabled": False})
    for number, change in enumerate(({"roles": ["assistant", "tool"]},     # one more role
                                     {"scope": "any"},                    # wider scope
                                     {"severity": "critical"}, {"description": "reworded"})):
        world.upsert({**RM, "id": f"t{number}"})
        out = world.upsert({**RM, "id": f"t{number}", **change})
        assert out["contracts"] and "pending" not in out, change
    world.upsert({"id": "plain", "type": "must_not", "description": "now enabled", "enabled": True})
    out = world.upsert({"id": "plain", "type": "must_not", "regex": "x", "description": "now has a verifier"})
    assert "pending" not in out and out["contracts"][0]["revision"] == 3


def test_removing_an_exception_and_enabling_are_tightening(world):
    ids = indexed(world)
    world.upsert({**RM, "exception_regex": "tmp"})
    assert "pending" not in world.upsert(RM)
    world.upsert({**RM, "enabled": False, "source_event_id": ids["user"]})
    assert "pending" not in world.upsert({**RM, "enabled": True})


def test_a_batch_applies_what_it_may_and_holds_the_rest(world):
    indexed(world)
    world.upsert(RM, {"id": "other", "type": "must_not", "regex": "curl"})
    out = world.upsert({**RM, "enabled": False}, {"id": "other", "type": "must_not", "regex": "curl",
                                                   "severity": "high"}, {"id": "brand-new", "type": "must_not",
                                                                         "regex": "dd if="})
    assert [c["id"] for c in out["contracts"]] == ["other", "brand-new"]
    assert [p["contract_id"] for p in out["pending"]] == ["no-rm"]
    assert active(world)["enabled"] is True


def test_revoke_without_an_anchor_is_held_and_with_one_it_disables(world):
    ids = indexed(world)
    world.upsert(RM)
    held = world.engine.revoke_contract("no-rm")
    assert held["status"] == "pending" and held["contract"]["enabled"] is True and "NOT APPLIED" in held["nudge"]
    assert active(world)["enabled"] is True and pending(world)
    assert finding(world.audit(), "no-rm")["verdict"] == "violation"
    done = world.engine.revoke_contract("no-rm", source_event_id=ids["user"])
    assert done["ok"] and done["status"] == "disabled" and done["contract"]["enabled"] is False
    assert not pending(world)
    with pytest.raises(ValueError, match="not a user message"):
        world.engine.revoke_contract("no-rm", source_event_id=ids["result"])
    assert world.engine.revoke_contract("missing", source_event_id=ids["user"])["error"] == "not_found"


def test_the_hook_keeps_using_the_active_revision_while_a_disable_is_pending(world):
    indexed(world)
    world.upsert(RM)
    world.engine.revoke_contract("no-rm")
    payload = {"session_id": SID, "transcript_path": str(world.path), "cwd": "/synthetic/project",
               "hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "rm -rf /z"}}
    response = run_hook("PreToolUse", payload, engine=world.engine)
    assert response["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_pending_changes_persist_across_processes_and_leave_unaffected_files_unchanged(world):
    indexed(world)
    world.upsert(RM)
    path = world.engine.settings.resolve_contracts_path()
    assert "pending" not in json.loads(path.read_text(encoding="utf-8"))          # nothing pending: no new key
    world.upsert({**RM, "enabled": False})
    assert "no-rm" in json.loads(path.read_text(encoding="utf-8"))["pending"]
    other = SelfDirectEngine(world.engine.settings)
    try:
        assert other.list_contracts()["pending"][0]["contract"]["id"] == "no-rm"
        assert other.list_contracts()["contracts"][0]["enabled"] is True
    finally:
        other.close()


# ---------------------------------------------------------------- the weakening table

BASE = ContractRule(id="c", type="must_not", scope="tool_call", regex="a", roles=["assistant", "tool"])


@pytest.mark.parametrize("change,expected", [
    ({"enabled": False}, ["disabled"]),
    ({"regex": "b"}, ["regex changed"]),
    ({"exception_regex": "x"}, ["exception added or changed"]),
    ({"scope": "message"}, ["scope narrowed"]),
    ({"scope": "any"}, []),
    ({"roles": ["assistant"]}, ["roles narrowed"]),
    ({"roles": ["assistant", "tool", "runtime"]}, []),
    ({"provider": "claude"}, ["narrowed to one provider"]),
    ({"session_id": "s"}, ["narrowed to one session"]),
    ({"severity": "low", "description": "d"}, []),
    ({"regex": None, "description": "d"}, ["regex changed"]),
])
def test_loosening_reasons(change, expected):
    new = ContractRule(**{**BASE.model_dump(), **change})
    assert loosening_reasons(BASE, new) == expected


def test_loosening_reasons_for_the_other_contract_fields():
    gate = ContractRule(id="g", type="must", scope="tool_call", regex="pytest", before_regex="commit",
                        requires_success=True, risk="high", auto_continue=False, when={"tool_call_regex": "x"})
    weaker = lambda **c: loosening_reasons(gate, ContractRule(**{**gate.model_dump(), **c}))   # noqa: E731
    assert weaker(requires_success=False) == ["success requirement removed"]
    assert weaker(risk="normal") == ["risk lowered"]
    assert weaker(before_regex="push") == ["before_regex changed"]
    assert weaker(when={"tool_call_regex": "y"}) == ["applicability condition (when) added or changed"]
    assert weaker(applies_from_event_id="e") == []                  # "from now on" is how a revision starts
    assert loosening_reasons(ContractRule(id="p", type="must_not", description="d", enabled=False),
                             ContractRule(id="p", type="must_not", regex="x", enabled=False)) == []


def test_the_revoke_tool_takes_the_anchor():
    from self_directing_mcp import server
    assert "source_event_id" in inspect.signature(server.revoke_contract).parameters


# ---------------------------------------------------------------- a global contract anchored in another session
#
# A global contract (no session id) is judged in every session, but its source and activation event usually belong to
# the session it was written in. The engine resolves such an anchor from the whole index once per build and hands it to
# both evaluators (``audit.anchors``); a session contract never takes its anchor from another session.

from self_directing_mcp.audit import runner                                                      # noqa: E402
from self_directing_mcp.audit.anchors import Anchor, parse_time                                # noqa: E402
from self_directing_mcp.audit import incremental as inc                                         # noqa: E402

DEPLOY = {"id": "no-deploy", "type": "must_not", "scope": "tool_call", "regex": r"terraform\s+apply"}
OTHER = "claude:other-session:0:0:abcd"          # an event of a session that is not the one judged
NOT_HERE = "contract source or activation event is not in this session"


def foreign(user=True, at=None, event_id=OTHER, cut_user=False):
    return {event_id: Anchor(event_id, user, cut_user, parse_time(at))}


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_global_contract_takes_a_user_source_from_another_session(provider):
    events = [user_turn(), call("terraform apply -auto-approve", "c1")]
    rule = {**DEPLOY, "source_event_id": OTHER}
    item = one(rule, events, provider, anchors=foreign())
    assert item.verdict == "violation" and "terraform apply" in item.evidence[0].snippet
    proposed = event("tool_call", "[tool_call:Bash] " + json.dumps({"command": "terraform apply"}), "assistant",
                     tool_name="Bash", proposed=True)
    assert one(rule, [user_turn()], provider, proposed=proposed, anchors=foreign()).verdict == "violation"
    assert one(rule, [user_turn(), call("ls", "c1")], provider, anchors=foreign()).verdict == "clean"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_global_contract_whose_source_in_another_session_is_not_a_user_message_is_unknown(provider):
    events = [user_turn(), call("terraform apply", "c1")]
    item = one({**DEPLOY, "source_event_id": OTHER}, events, provider, anchors=foreign(user=False))
    assert item.verdict == "unknown" and item.reason.startswith("anchor_not_user:") and item.evidence == []


@pytest.mark.parametrize("provider", PROVIDERS)
def test_an_anchor_that_is_in_no_session_of_the_index_stays_unknown(provider):
    events = [user_turn(), call("terraform apply", "c1")]
    for extra in ({"source_event_id": OTHER}, {"applies_from_event_id": OTHER},
                  {"source_event_id": OTHER, "applies_from_event_id": "claude:elsewhere:0:0:ef01"}):
        for anchors in (None, {}, foreign(event_id="claude:unrelated:0:0:2345")):
            item = one({**DEPLOY, **extra}, events, provider, anchors=anchors)
            assert item.verdict == "unknown" and item.reason == NOT_HERE, (extra, anchors)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_session_contract_never_takes_its_anchor_from_another_session(provider):
    events = [user_turn(), call("terraform apply", "c1")]
    mine = {**DEPLOY, "source_event_id": OTHER, "session_id": "s"}      # the audit judges session "s"
    item = one(mine, events, provider, anchors=foreign())
    assert item.verdict == "unknown" and item.reason == NOT_HERE
    # a contract of another session is not selected by the engine at all; the reference says so for itself
    (elsewhere,) = runner.full_findings([ContractRule(**{**mine, "session_id": "some-other-session"})], events,
                                        session_id="s", store=None, provider=provider, proposed=None, coverage={},
                                        anchors=foreign())
    assert elsewhere.verdict == "clean" and elsewhere.reason == "contract outside this provider/session"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_an_anchor_in_the_judged_session_is_unchanged_whatever_the_map_says(provider):
    user = user_turn()
    events = [user, call("terraform apply", "c1")]
    # the map speaks about an id that this session also holds: the session's own event decides
    claimed = foreign(user=False, event_id=user.chunk_id)
    assert one({**DEPLOY, "source_event_id": user.chunk_id}, events, provider, anchors=claimed).verdict == "violation"
    tool = result("c1", True)
    mixed = [user, call("ls", "c1"), tool, call("terraform apply", "c2")]
    claimed = foreign(user=True, event_id=tool.chunk_id)
    assert one({**DEPLOY, "source_event_id": tool.chunk_id}, mixed, provider, anchors=claimed).reason.startswith(
        "anchor_not_user:")


T0 = "2026-10-01T12:00:00Z"


def stamped(builder, at, *args, **kwargs):
    item = builder(*args, **kwargs)
    item.timestamp = at
    return item


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_global_contract_that_starts_in_another_session_judges_the_events_from_that_time(provider):
    before = stamped(call, "2026-10-01T11:59:59Z", "terraform apply before", "c1")
    after = stamped(call, "2026-10-01T12:00:01+00:00", "terraform apply after", "c2")
    events = [stamped(user_turn, "2026-10-01T11:00:00Z"), before, after]
    rule = {**DEPLOY, "applies_from_event_id": OTHER}
    item = one(rule, events, provider, anchors=foreign(at=T0))
    assert item.verdict == "violation" and len(item.evidence) == 1 and "apply after" in item.evidence[0].snippet
    assert one(rule, events[:2], provider, anchors=foreign(at=T0)).verdict == "clean"      # only an earlier match
    # an offset spelling of an instant orders like Z
    late = stamped(call, "2026-10-01T20:59:59+09:00", "terraform apply late", "c3")
    assert one(rule, [events[0], before, late], provider, anchors=foreign(at=T0)).verdict == "clean"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_when_the_time_cannot_be_compared_the_whole_session_is_judged(provider):
    applies = [stamped(call, at, "terraform apply x", f"c{n}") for n, at in
               enumerate(("2026-10-01T11:00:00Z", None, "not a time", "2026-10-01T11:00:00"))]   # the last: no offset
    rule = {**DEPLOY, "applies_from_event_id": OTHER}
    # the first event is before the time and is not judged; those without a comparable time are (judge more, never less)
    item = one(rule, [stamped(user_turn, "2026-10-01T10:00:00Z"), *applies], provider, anchors=foreign(at=T0))
    assert item.verdict == "violation" and len(item.evidence) == 3
    # the anchor itself has no comparable time: nothing is cut
    for anchor_time in (None, "garbage", "2026-10-01T12:00:00"):
        item = one(rule, [stamped(call, "2000-01-01T00:00:00Z", "terraform apply old", "c1")], provider,
                   anchors=foreign(at=anchor_time))
        assert item.verdict == "violation", anchor_time


@pytest.mark.parametrize("provider", PROVIDERS)
def test_an_obligation_that_starts_in_another_session_counts_only_evidence_from_that_time(provider):
    gate = {"id": "gate", "type": "must", "scope": "tool_call", "regex": "pytest", "before_regex": "git commit",
            "requires_success": True, "applies_from_event_id": OTHER}
    old = stamped(call, "2026-10-01T10:00:00Z", "pytest", "c1")
    old_ok = stamped(result, "2026-10-01T10:00:01Z", "c1", True)
    commit = stamped(call, "2026-10-01T13:00:00Z", "git commit -m x", "c2")
    new = stamped(call, "2026-10-01T12:30:00Z", "pytest", "c3")
    new_ok = stamped(result, "2026-10-01T12:30:01Z", "c3", True)
    stale = one(gate, [old, old_ok, commit], provider, anchors=foreign(at=T0))
    assert stale.verdict == "violation"                    # a run from before the activation does not count
    fresh = one(gate, [old, old_ok, new, new_ok, commit], provider, anchors=foreign(at=T0))
    assert fresh.verdict == "clean"
    assert one(gate, [old, old_ok, commit], provider, anchors=foreign(at=None)).verdict == "clean"


def test_both_evaluators_agree_when_a_foreign_window_opens_in_a_later_feed():
    events = [stamped(call, "2026-10-01T11:00:00Z", "terraform apply 1", "c1"),
              stamped(call, "2026-10-01T11:30:00Z", "ls", "c2"),
              stamped(call, "2026-10-01T12:30:00Z", "terraform apply 2", "c3"),
              stamped(call, "2026-10-01T12:40:00Z", "ls", "c4")]
    rules = [{**DEPLOY, "applies_from_event_id": OTHER}, {**DEPLOY, "id": "src", "source_event_id": OTHER},
             {**DEPLOY, "id": "both", "source_event_id": OTHER, "applies_from_event_id": OTHER}]
    findings = evaluate(rules, events, chunk_sizes=(None, 1, 2, 3), anchors=foreign(at=T0))
    assert [f.verdict for f in findings] == ["violation"] * 3
    assert [len(f.evidence) for f in findings] == [1, 2, 1]


def test_a_stored_monitor_state_is_keyed_by_the_foreign_anchor_it_was_built_under():
    rule = ContractRule(**{**DEPLOY, "source_event_id": OTHER, "applies_from_event_id": OTHER})
    plain = ContractRule(**DEPLOY)
    keys = [inc.rule_fingerprint(rule, "claude", 1, "d", anchors) for anchors in
            (None, {}, foreign(), foreign(user=False), foreign(at=T0), foreign(at="2026-10-01T12:00:01Z"))]
    assert keys[0] == keys[1] and len(set(keys[1:])) == 5       # none, user, other, two times
    mine = ContractRule(**{**DEPLOY, "source_event_id": OTHER, "session_id": "s"})
    assert (inc.rule_fingerprint(mine, "claude", 1, "d", foreign()) == inc.rule_fingerprint(mine, "claude", 1, "d", None))
    assert inc.rule_fingerprint(plain, "claude", 1, "d", foreign()) == inc.rule_fingerprint(plain, "claude", 1, "d")


# ---------------------------------------------------------------- the same through the engine: two indexed sessions

SID_B = "bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb"
APPLY = "terraform apply -auto-approve"


@pytest.fixture
def pair(tmp_path):
    """Session A (the contract's origin) and session B in one index; the engine audits B with incremental state."""
    world = Claude(tmp_path, audit_incremental_min_events=0, audit_shadow_every=1)
    path_b = world.root / PROJECT / f"{SID_B}.jsonl"

    class Pair:
        a = world

        def write_b(self, *records, mode="w"):
            write(path_b, *records, mode=mode)
            world.engine.sync_session(session_id=SID_B, provider="claude", path=str(path_b), embed=False)

        def audit_b(self):
            return world.engine.audit_session(SID_B, provider="claude", path=str(path_b))

        def check_b(self, command):
            return world.engine.check_action(SID_B, {"tool_name": "Bash", "arguments": {"command": command}},
                                             provider="claude", path=str(path_b))

        def hook_b(self, command):
            payload = {"session_id": SID_B, "transcript_path": str(path_b), "cwd": "/synthetic/project",
                       "hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": command}}
            return run_hook("PreToolUse", payload, engine=world.engine)

    yield Pair()
    world.close()


def in_b(builder, *args, **extra):
    return builder(*args, sessionId=SID_B, **extra)


def prepare_a(pair):
    """Session A: a user message, a tool call and its result; returns their event ids."""
    pair.a.write(human("never apply terraform"), bash("t1", "ls"), tool_result("t1", "file"))
    pair.a.sync()
    chunks = pair.a.chunks()
    return {"user": next(c.chunk_id for c in chunks if c.meta.get("role") == "user"),
            "result": next(c.chunk_id for c in chunks if c.kind == "tool_result"),
            "call": next(c.chunk_id for c in chunks if c.kind == "tool_call")}


def test_a_global_contract_anchored_in_session_a_blocks_the_same_action_in_session_b(pair):
    ids = prepare_a(pair)
    pair.a.upsert({**DEPLOY, "source_event_id": ids["user"]})
    assert finding(pair.a.check("Bash", {"command": APPLY}), "no-deploy")["verdict"] == "violation"
    pair.write_b(in_b(human, "carry on"), in_b(bash, "b1", "ls"), in_b(tool_result, "b1", "file"))
    assert finding(pair.check_b(APPLY), "no-deploy")["verdict"] == "violation"
    assert finding(pair.check_b("ls -la"), "no-deploy")["verdict"] == "clean"
    response = pair.hook_b(APPLY)
    assert response["hookSpecificOutput"]["permissionDecision"] == "deny", response
    # the history audit: nothing applied yet, then a call is appended and the stored state moves on
    first = pair.audit_b()
    assert finding(first, "no-deploy")["verdict"] == "clean" and "incremental_mismatch" not in first["incremental"]
    pair.write_b(in_b(bash, "b2", APPLY), mode="a")
    second = pair.audit_b()
    assert finding(second, "no-deploy")["verdict"] == "violation"
    assert "incremental_mismatch" not in second["incremental"] and second["incremental"]["mode"] == "delta"


def test_a_non_user_anchor_in_session_a_is_not_a_source_in_session_b(pair):
    ids = prepare_a(pair)
    pair.a.engine.contracts.upsert([ContractRule(**{**DEPLOY, "source_event_id": ids["result"]})])   # an old store
    pair.write_b(in_b(human, "carry on"), in_b(bash, "b1", APPLY))
    for outcome in (pair.check_b(APPLY), pair.audit_b()):
        item = finding(outcome, "no-deploy")
        assert item["verdict"] == "unknown" and item["reason"].startswith("anchor_not_user:")
    assert pair.hook_b(APPLY).get("hookSpecificOutput", {}).get("permissionDecision", "none") == "none"


def test_an_anchor_in_no_indexed_session_is_unknown_in_session_b(pair):
    prepare_a(pair)
    pair.a.engine.contracts.upsert([ContractRule(**{**DEPLOY, "source_event_id": "claude:nowhere:0:0:abcd"})])
    pair.write_b(in_b(human, "carry on"), in_b(bash, "b1", APPLY))
    for outcome in (pair.check_b(APPLY), pair.audit_b()):
        item = finding(outcome, "no-deploy")
        assert item["verdict"] == "unknown" and item["reason"] == NOT_HERE


def test_a_session_contract_anchored_in_session_a_is_not_applied_in_session_b(pair):
    ids = prepare_a(pair)
    pair.a.upsert({**DEPLOY, "source_event_id": ids["user"], "session_id": SID})
    pair.write_b(in_b(human, "carry on"), in_b(bash, "b1", APPLY))
    assert not [f for f in pair.audit_b()["findings"] if f["contract_id"] == "no-deploy"]
    assert not [f for f in pair.check_b(APPLY)["findings"] if f["contract_id"] == "no-deploy"]


def test_a_global_contract_that_starts_in_session_a_judges_session_b_from_that_time(pair):
    ids = prepare_a(pair)
    started = next(c for c in pair.a.chunks() if c.chunk_id == ids["call"]).timestamp
    assert parse_time(started) is not None
    pair.a.engine.contracts.upsert([ContractRule(**{**DEPLOY, "applies_from_event_id": ids["call"]})])
    old, new = "2020-01-01T00:00:00.000Z", "2099-01-01T00:00:00.000Z"
    pair.write_b(in_b(human, "carry on", timestamp=old), in_b(bash, "b1", APPLY + " old", timestamp=old))
    quiet = pair.audit_b()
    assert finding(quiet, "no-deploy")["verdict"] == "clean" and "incremental_mismatch" not in quiet["incremental"]
    pair.write_b(in_b(bash, "b2", APPLY + " new", timestamp=new), mode="a")
    loud = pair.audit_b()
    item = finding(loud, "no-deploy")
    assert item["verdict"] == "violation" and len(item["evidence"]) == 1 and "apply" in item["evidence"][0]["snippet"]
    assert "-auto-approve new" in item["evidence"][0]["snippet"] and "incremental_mismatch" not in loud["incremental"]
