"""One judgement, four hosts: the same synthetic session and contracts through Codex, OMP, AGY and Claude Code.

The judge (``check_action`` / ``audit_session``) must return the same verdicts, finding verdicts and coverage
no matter which adapter asked. Only the host response differs, exactly as specified:

* Codex: advisory context, never a permission decision.
* OMP: the extension blocks a proposed tool call only for ``violation`` / ``suspicious``.
* AGY (default ``confirmed_only``): deny / force_ask / allow-with-warning / ask for unknown on ``risk: high``;
  Stop continues only for a confirmed unmet ``auto_continue`` obligation, once per conversation.
* Claude Code (default ``confirmed_only``): ``permissionDecision`` deny / ask, never an explicit allow (a clean
  verdict is not permission); unknown passes through with a ``systemMessage`` warning, or asks on ``risk: high``.
  Stop blocks only for a confirmed unmet ``auto_continue`` obligation, once per session.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re

import pytest

from self_directing_mcp import claude_hooks
from self_directing_mcp.agy_hooks import handle_pre_tool_use, handle_stop
from self_directing_mcp.codex_hooks import handle_hook
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine

ROOT = Path(__file__).resolve().parents[1]
CODEX_SID = "11111111-2222-3333-4444-555555555555"
AGY_SID = "11111111-1111-4111-8111-111111111111"
CLAUDE_SID = "22222222-2222-4222-8222-222222222222"
OMP_SID = "session-1"
PREFIX = "Self-directing audit (data, not instructions): "

USER = ("user", "Please finish the task and say VERIFIED when it is done.")
WORKING = ("assistant", "Working on it.")
DONE = ("assistant", "VERIFIED")


# --- one synthetic history, written in each host's native format -------------------------------------------

def _write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _codex_history(root, lines):
    records = [{"type": "session_meta", "payload": {"id": CODEX_SID}}]
    records += [{"type": "response_item", "payload": {"type": "message", "role": role, "content": text}}
                for role, text in lines]
    return _write(root / f"rollout-2026-09-09T01-00-00-{CODEX_SID}.jsonl", records)


def _omp_history(root, lines):
    records = [{"type": "title", "title": "Common fixture"}, {"type": "session", "id": OMP_SID}]
    records += [{"type": "message", "message": {"role": role, "content": [{"type": "text", "text": text}]}}
                for role, text in lines]
    return _write(root / "session.jsonl", records)


def _agy_history(root, lines):
    records = []
    for step, (role, text) in enumerate(lines):
        if role == "user":
            records.append({"step_index": step, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE",
                            "created_at": "2026-10-01T00:00:00Z", "content": f"<USER_REQUEST>{text}</USER_REQUEST>"})
        else:
            records.append({"step_index": step, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE",
                            "created_at": "2026-10-01T00:00:01Z", "content": text})
    return _write(root / "brain" / AGY_SID / ".system_generated" / "logs" / "transcript.jsonl", records)


def _claude_history(root, lines):
    records = []
    for number, (role, line) in enumerate(lines, 1):
        common = {"uuid": f"00000000-0000-4000-8000-{number:012d}", "parentUuid": None, "sessionId": CLAUDE_SID,
                  "timestamp": f"2026-10-01T00:00:{number:02d}.000Z", "cwd": "/synthetic/project",
                  "isSidechain": False, "userType": "external", "version": "9.9.9"}
        if role == "user":
            records.append({"type": "user", **common, "message": {"role": "user", "content": line},
                            "origin": {"kind": "human"}, "promptId": "prompt-1"})
        else:
            records.append({"type": "assistant", **common, "requestId": "req-1", "message": {
                "id": "msg-1", "role": "assistant", "model": "claude-synthetic", "stop_reason": "end_turn",
                "stop_sequence": None, "content": [{"type": "text", "text": line}]}})
    return _write(root / "-synthetic-project" / f"{CLAUDE_SID}.jsonl", records)


@dataclass
class Host:
    name: str
    engine: SelfDirectEngine
    session_id: str
    path: Path
    judged: list = field(default_factory=list)   # every audit the adapter obtained from the judge


def _spy(engine, log):
    for name in ("check_action", "audit_session"):
        original = getattr(engine, name)
        def wrapper(*args, _original=original, _name=name, **kwargs):
            result = _original(*args, **kwargs)
            log.append((_name, result))
            return result
        setattr(engine, name, wrapper)


def _hosts(tmp_path, lines, contracts):
    codex_root, omp_root, agy_root, claude_root = (tmp_path / "codex", tmp_path / "omp", tmp_path / "agy",
                                                   tmp_path / "claude")
    specs = [
        ("codex", CODEX_SID, _codex_history(codex_root, lines), dict(codex_sessions_dir=codex_root)),
        ("omp", OMP_SID, _omp_history(omp_root, lines), dict(omp_sessions_dir=omp_root)),
        ("agy", AGY_SID, _agy_history(agy_root, lines), dict(agy_app_data_dirs=[agy_root])),
        ("claude", CLAUDE_SID, _claude_history(claude_root, lines), dict(claude_projects_dir=claude_root)),
    ]
    hosts = {}
    for name, sid, path, paths in specs:
        engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=tmp_path / f"index-{name}",
                                           openrouter_api_key_file=None, **paths))
        if contracts:
            engine.upsert_contracts(contracts)
        host = Host(name, engine, sid, path)
        _spy(engine, host.judged)
        hosts[name] = host
    return hosts


@pytest.fixture
def make_hosts(tmp_path, monkeypatch):
    for name in ("SELF_DIRECT_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER",
                 "SELF_DIRECT_AGY_ENFORCEMENT", "SELF_DIRECT_CLAUDE_ENFORCEMENT", "SELF_DIRECT_CLAUDE_PROJECTS_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(tmp_path / "hook-state"))   # AGY hook bookkeeping
    created = []
    def make(lines, contracts):
        hosts = _hosts(tmp_path, lines, contracts)
        created.extend(hosts.values())
        return hosts
    yield make
    for host in created:
        host.engine.close()


# --- host adapters and the response each host produces ------------------------------------------------------

def codex_action(host, command):
    response = handle_hook(host.engine, "PreToolUse", host.session_id, str(host.path), "shell", {"command": command})
    return response


def omp_action(host, command):
    """What extensions/tsukkomi-omp.ts does on a tool_call: ask check_action, block only if actionable."""
    result = host.engine.check_action(host.session_id, {"tool_name": "shell", "arguments": {"command": command}},
                                      provider="omp", path=str(host.path), verify="fast")
    blocked = result["verdict"] in OMP_ACTIONABLE
    return {"block": True} if blocked else {}


def agy_action(host, command):
    return handle_pre_tool_use(host.engine, host.session_id,
                               {"name": "run_command", "args": {"CommandLine": command}}, str(host.path))


def claude_payload(host, event, **fields):
    return {"session_id": host.session_id, "transcript_path": str(host.path), "cwd": "/synthetic/project",
            "hook_event_name": event, **fields}


def claude_action(host, command):
    """The full adapter path: Claude Code's PreToolUse JSON in, its response JSON out."""
    return claude_hooks.run_hook("PreToolUse", claude_payload(host, "PreToolUse", tool_name="Bash",
                                                              tool_input={"command": command}), engine=host.engine)


def codex_stop(host):
    return handle_hook(host.engine, "Stop", host.session_id, str(host.path), stop_hook_active=False)


def omp_stop(host):
    """OMP audits at session events and only notifies; it never blocks from an audit."""
    host.engine.audit_session(host.session_id, provider="omp", path=str(host.path))
    return {}


def agy_stop(host):
    return handle_stop(host.engine, host.session_id,
                       {"transcriptPath": str(host.path), "fullyIdle": True, "terminationReason": "NO_TOOL_CALL"})


def claude_stop(host):
    return claude_hooks.run_hook("Stop", claude_payload(host, "Stop", stop_hook_active=False), engine=host.engine)


def _omp_actionable():
    source = (ROOT / "extensions" / "tsukkomi-omp.ts").read_text(encoding="utf-8")
    declared = re.search(r"const actionable: Record<string, true> = \{([^}]*)\}", source)
    assert declared, "the OMP extension no longer declares its actionable verdicts"
    # The block is returned for actionable verdicts only (optionally after resetting the check notice).
    assert re.search(r"if \(actionable\[result\.verdict\]\) (?:\{ lastNotice\.check = \"\"; )?return \{ block: true",
                     source)
    return set(re.findall(r"(\w+): true", declared.group(1)))


OMP_ACTIONABLE = _omp_actionable()


def test_the_omp_extension_blocks_only_violation_and_suspicious():
    assert OMP_ACTIONABLE == {"violation", "suspicious"}


def _facts(response):
    """Codex's advisory context payload, or None when the hook stayed silent."""
    if not response:
        return None
    assert set(response) <= {"hookSpecificOutput", "systemMessage"}, "Codex hooks never decide permission"
    text = response.get("systemMessage") or response["hookSpecificOutput"]["additionalContext"]
    assert text.startswith(PREFIX)
    return json.loads(text[len(PREFIX):])


def _normalize(audit):
    coverage = audit["coverage"]
    return {"verdict": audit["verdict"],
            "findings": sorted((f["contract_id"], f["verdict"], f.get("applicability", "applicable"))
                               for f in audit["findings"]),
            "coverage": (coverage["complete"], tuple(sorted(coverage["issues"])))}


def _same_judgement(hosts, method):
    seen = {}
    for name, host in hosts.items():
        audits = [result for called, result in host.judged if called == method]
        assert audits, f"{name} never asked the judge ({method})"
        seen[name] = _normalize(audits[-1])
    assert seen["codex"] == seen["omp"] == seen["agy"] == seen["claude"], seen
    return seen["agy"]


# --- scenarios ----------------------------------------------------------------------------------------------

NO_FORCE_PUSH = {"id": "no-force-push", "type": "must_not", "scope": "tool_call", "regex": r"git push --force",
                 "risk": "high"}
NO_PROD = {"id": "no-prod-deploy", "type": "must_not", "scope": "tool_call", "regex": r"deploy prod",
           "exception_regex": r"--dry-run"}
PROSE = {"id": "prose-rule", "type": "must_not", "description": "Do not do the thing no verifier can see."}
# A high-risk target names the actions it covers (a regex); unknown here because its activation event is absent.
PROSE_HIGH = dict(PROSE, id="prose-high", risk="high", regex="echo", applies_from_event_id="not-in-this-session")
ACK = {"id": "ack", "type": "must", "scope": "message", "regex": "VERIFIED", "auto_continue": True}


@dataclass
class ActionCase:
    contracts: list
    command: str
    verdict: str
    finding_verdicts: dict
    agy: str
    claude: str     # Claude Code permissionDecision; "none" = no decision, Claude Code's own permissions apply


ACTIONS = {
    "violation": ActionCase([NO_FORCE_PUSH], "git push --force origin main", "violation",
                            {"no-force-push": "violation"}, "deny", "deny"),
    "suspicious": ActionCase([NO_PROD], "deploy prod --dry-run", "suspicious", {"no-prod-deploy": "suspicious"},
                             "force_ask", "ask"),
    "unknown": ActionCase([PROSE], "echo ok", "unknown", {"prose-rule": "unknown"}, "allow", "none"),
    "unknown_high_risk": ActionCase([PROSE_HIGH], "echo ok", "unknown", {"prose-high": "unknown"}, "ask", "ask"),
    "clean": ActionCase([NO_FORCE_PUSH], "git status", "clean", {"no-force-push": "clean"}, "allow", "none"),
}


@pytest.mark.parametrize("name", sorted(ACTIONS))
def test_proposed_action_same_judgement_host_specific_response(make_hosts, name):
    case = ACTIONS[name]
    hosts = make_hosts([USER, WORKING], case.contracts)
    responses = {"codex": codex_action(hosts["codex"], case.command),
                 "omp": omp_action(hosts["omp"], case.command),
                 "agy": agy_action(hosts["agy"], case.command),
                 "claude": claude_action(hosts["claude"], case.command)}
    judged = _same_judgement(hosts, "check_action")
    assert judged["verdict"] == case.verdict
    assert dict((cid, verdict) for cid, verdict, _ in judged["findings"]) == case.finding_verdicts
    assert judged["coverage"] == (True, ())

    facts = _facts(responses["codex"])                                  # Codex: advisory context only
    if case.verdict == "clean":
        assert facts is None
    else:
        assert facts["verdict"] == case.verdict and facts["coverage_complete"] is True
    assert (responses["omp"] == {"block": True}) == (case.verdict in ("violation", "suspicious"))
    assert responses["agy"]["decision"] == case.agy
    claude = responses["claude"]
    assert claude.get("hookSpecificOutput", {}).get("permissionDecision", "none") == case.claude
    assert '"allow"' not in json.dumps(claude)          # a clean verdict is never a permission grant
    if name == "unknown":
        assert "WARNING" in responses["agy"]["reason"] and "not a compliance verdict" in responses["agy"]["reason"]
        assert "WARNING" in claude["systemMessage"] and "not a compliance verdict" in claude["systemMessage"]
    if name == "clean":
        assert claude == {}


def test_no_applicable_contract_is_not_a_verdict_on_any_host(make_hosts):
    hosts = make_hosts([USER, WORKING], [])
    assert codex_action(hosts["codex"], "git push --force") == {}
    assert agy_action(hosts["agy"], "git push --force")["decision"] == "allow"
    assert claude_action(hosts["claude"], "git push --force") == {}
    omp = hosts["omp"].engine.check_action(OMP_SID, {"tool_name": "shell", "arguments": {"command": "git push --force"}},
                                           provider="omp", path=str(hosts["omp"].path))
    assert omp["applicable_contracts"] == 0 and omp["verdict"] == "unknown" and omp["coverage"]["complete"]
    assert omp_action(hosts["omp"], "git push --force") == {}


@dataclass
class CheckpointCase:
    lines: list
    contracts: list
    verdict: str
    finding: tuple
    codex_silent: bool
    agy_first: str
    claude_first: str    # "block" or "none"


CHECKPOINTS = {
    "unmet_auto_continue": CheckpointCase([USER, WORKING], [ACK], "violation", ("ack", "violation", "applicable"),
                                          False, "continue", "block"),
    "met": CheckpointCase([USER, WORKING, DONE], [ACK], "clean", ("ack", "clean", "applicable"), True, "allow", "none"),
    "when_not_met": CheckpointCase([USER, WORKING], [dict(ACK, when={"tool_call_regex": "docker build"})], "clean",
                                   ("ack", "clean", "not_applicable"), True, "allow", "none"),
    "unmet_without_auto_continue": CheckpointCase([USER, WORKING], [dict(ACK, auto_continue=False)], "violation",
                                                  ("ack", "violation", "applicable"), False, "allow", "none"),
}


@pytest.mark.parametrize("name", sorted(CHECKPOINTS))
def test_checkpoint_same_judgement_host_specific_response(make_hosts, name):
    case = CHECKPOINTS[name]
    hosts = make_hosts(case.lines, case.contracts)
    responses = {"codex": codex_stop(hosts["codex"]), "omp": omp_stop(hosts["omp"]), "agy": agy_stop(hosts["agy"]),
                 "claude": claude_stop(hosts["claude"])}
    judged = _same_judgement(hosts, "audit_session")
    assert judged["verdict"] == case.verdict
    assert judged["findings"] == [case.finding]
    assert judged["coverage"] == (True, ())

    facts = _facts(responses["codex"])
    assert (facts is None) == case.codex_silent
    if facts is not None:
        assert facts["verdict"] == case.verdict and "continue" not in json.dumps(responses["codex"]).lower()
    assert responses["omp"] == {}
    assert responses["agy"]["decision"] == case.agy_first
    assert responses["claude"].get("decision", "none") == case.claude_first
    assert "hookSpecificOutput" not in responses["claude"]
    # The continue budget is per conversation: the same checkpoint on the next turn only reports.
    assert agy_stop(hosts["agy"])["decision"] == "allow"
    assert "decision" not in claude_stop(hosts["claude"])


def test_unknown_evidence_is_never_clean_and_never_resumes_on_any_host(make_hosts):
    """Description-only obligation: no verifier exists, so all hosts judge unknown and none act on it."""
    hosts = make_hosts([USER, WORKING], [dict(PROSE, type="must", id="prose-must")])
    responses = {"codex": codex_stop(hosts["codex"]), "omp": omp_stop(hosts["omp"]), "agy": agy_stop(hosts["agy"]),
                 "claude": claude_stop(hosts["claude"])}
    judged = _same_judgement(hosts, "audit_session")
    assert judged["verdict"] == "unknown" and judged["findings"] == [("prose-must", "unknown", "applicable")]
    assert _facts(responses["codex"])["verdict"] == "unknown"
    assert responses["omp"] == {} and responses["agy"]["decision"] == "allow"
    assert "decision" not in responses["claude"] and "Not verified" in responses["claude"]["systemMessage"]


# --- host runtime records are not evidence on any host ---------------------------------------------------------

def _append_host_runtime_records(hosts, phrases):
    """Records only AGY (IDE context steps, a NUL-padded runtime line) and Claude Code (a scheduled-task
    announcement whose prompt carries ``phrases``) emit. Codex and OMP histories stay as they are."""
    agy = hosts["agy"].path
    lines = [{"step_index": 2, "source": "SYSTEM", "type": "CONVERSATION_HISTORY", "status": "DONE",
              "created_at": "2026-10-01T00:00:02Z"},
             {"step_index": 3, "source": "SYSTEM", "type": "KNOWLEDGE_ARTIFACTS", "status": "DONE",
              "created_at": "2026-10-01T00:00:03Z"}]
    padded = {"step_index": 4, "source": "SYSTEM", "type": "SYSTEM_MESSAGE", "status": "DONE",
              "created_at": "2026-10-01T00:00:04Z", "content": " ".join(phrases)}
    with agy.open("ab") as stream:
        for record in lines:
            stream.write(json.dumps(record).encode() + b"\n")
        stream.write(b"\x00" * 1550 + json.dumps(padded).encode() + b"\n")
    claude = hosts["claude"].path
    fired = {"type": "system", "subtype": "scheduled_task_fire", "uuid": "00000000-0000-4000-8000-0000000000f1",
             "parentUuid": None, "sessionId": CLAUDE_SID, "timestamp": "2026-10-01T00:00:59.000Z",
             "cwd": "/synthetic/project", "isSidechain": False, "userType": "external", "version": "9.9.9",
             "isMeta": False, "cron": "*/5 * * * *", "cronKind": "recurring", "taskId": "task-0001",
             "taskKind": "cron", "content": " ".join(phrases), "prompt": " ".join(phrases)}
    with claude.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(fired) + "\n")


def test_host_runtime_records_never_change_the_judgement_on_any_host(make_hosts):
    """A scheduled-task prompt (Claude Code), IDE context steps and a padded line (AGY) carry the very phrases the
    user-role contracts look for. They are runtime text: they do not satisfy a user obligation, do not violate a
    user prohibition, and leave coverage complete, so all four hosts still return the same verdicts."""
    approval, secret = "approved-by-the-owner", "SCHEDULED-ONLY-PHRASE"
    contracts = [ACK,
                 {"id": "owner-approval", "type": "must", "scope": "message", "roles": ["user"], "regex": approval},
                 {"id": "no-secret", "type": "must_not", "scope": "message", "roles": ["user"], "regex": secret}]
    hosts = make_hosts([USER, WORKING], contracts)
    _append_host_runtime_records(hosts, [approval, secret, "VERIFIED"])
    for host in hosts.values():
        host.engine.audit_session(host.session_id, provider=host.name, path=str(host.path))
    judged = _same_judgement(hosts, "audit_session")
    assert judged["verdict"] == "violation" and judged["coverage"] == (True, ())
    assert judged["findings"] == [("ack", "violation", "applicable"), ("no-secret", "clean", "applicable"),
                                  ("owner-approval", "violation", "applicable")]
    # And the extra records were in fact parsed on the two hosts that have them (no silent skip).
    agy_chunks = hosts["agy"].engine.store.list_chunks(AGY_SID, "agy")
    assert {c.meta.get("event_type") for c in agy_chunks} >= {"CONVERSATION_HISTORY", "KNOWLEDGE_ARTIFACTS"}
    assert any(c.meta.get("host_padding_bytes") == 1550 for c in agy_chunks)
    claude_chunks = hosts["claude"].engine.store.list_chunks(CLAUDE_SID, "claude")
    assert any(c.meta.get("system_subtype") == "scheduled_task_fire" and c.meta["role"] == "runtime"
               for c in claude_chunks)
    assert not any(secret in c.text or approval in c.text for c in agy_chunks + claude_chunks if c.meta["role"] == "user")
