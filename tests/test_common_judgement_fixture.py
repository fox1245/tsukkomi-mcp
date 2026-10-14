"""One judgement, three hosts: the same synthetic session and contracts through Codex, OMP and AGY.

The judge (``check_action`` / ``audit_session``) must return the same verdicts, finding verdicts and coverage
no matter which adapter asked. Only the host response differs, exactly as specified:

* Codex: advisory context, never a permission decision.
* OMP: the extension blocks a proposed tool call only for ``violation`` / ``suspicious``.
* AGY (default ``confirmed_only``): deny / force_ask / allow-with-warning / ask for unknown on ``risk: high``;
  Stop continues only for a confirmed unmet ``auto_continue`` obligation, once per conversation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re

import pytest

from self_directing_mcp.agy_hooks import handle_pre_tool_use, handle_stop
from self_directing_mcp.codex_hooks import handle_hook
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine

ROOT = Path(__file__).resolve().parents[1]
CODEX_SID = "11111111-2222-3333-4444-555555555555"
AGY_SID = "11111111-1111-4111-8111-111111111111"
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
    codex_root, omp_root, agy_root = (tmp_path / "codex", tmp_path / "omp", tmp_path / "agy")
    specs = [
        ("codex", CODEX_SID, _codex_history(codex_root, lines), dict(codex_sessions_dir=codex_root)),
        ("omp", OMP_SID, _omp_history(omp_root, lines), dict(omp_sessions_dir=omp_root)),
        ("agy", AGY_SID, _agy_history(agy_root, lines), dict(agy_app_data_dirs=[agy_root])),
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
                 "SELF_DIRECT_AGY_ENFORCEMENT"):
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
                                      provider="omp", path=str(host.path))
    blocked = result["verdict"] in OMP_ACTIONABLE
    return {"block": True} if blocked else {}


def agy_action(host, command):
    return handle_pre_tool_use(host.engine, host.session_id,
                               {"name": "run_command", "args": {"CommandLine": command}}, str(host.path))


def codex_stop(host):
    return handle_hook(host.engine, "Stop", host.session_id, str(host.path), stop_hook_active=False)


def omp_stop(host):
    """OMP audits at session events and only notifies; it never blocks from an audit."""
    host.engine.audit_session(host.session_id, provider="omp", path=str(host.path))
    return {}


def agy_stop(host):
    return handle_stop(host.engine, host.session_id,
                       {"transcriptPath": str(host.path), "fullyIdle": True, "terminationReason": "NO_TOOL_CALL"})


def _omp_actionable():
    source = (ROOT / "extensions" / "tsukkomi-omp.ts").read_text(encoding="utf-8")
    declared = re.search(r"const actionable: Record<string, true> = \{([^}]*)\}", source)
    assert declared, "the OMP extension no longer declares its actionable verdicts"
    assert re.search(r"if \(actionable\[result\.verdict\]\) return \{ block: true", source)
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
    assert seen["codex"] == seen["omp"] == seen["agy"], seen
    return seen["agy"]


# --- scenarios ----------------------------------------------------------------------------------------------

NO_FORCE_PUSH = {"id": "no-force-push", "type": "must_not", "scope": "tool_call", "regex": r"git push --force",
                 "risk": "high"}
NO_PROD = {"id": "no-prod-deploy", "type": "must_not", "scope": "tool_call", "regex": r"deploy prod",
           "exception_regex": r"--dry-run"}
PROSE = {"id": "prose-rule", "type": "must_not", "description": "Do not do the thing no verifier can see."}
PROSE_HIGH = dict(PROSE, id="prose-high", risk="high")
ACK = {"id": "ack", "type": "must", "scope": "message", "regex": "VERIFIED", "auto_continue": True}


@dataclass
class ActionCase:
    contracts: list
    command: str
    verdict: str
    finding_verdicts: dict
    agy: str


ACTIONS = {
    "violation": ActionCase([NO_FORCE_PUSH], "git push --force origin main", "violation",
                            {"no-force-push": "violation"}, "deny"),
    "suspicious": ActionCase([NO_PROD], "deploy prod --dry-run", "suspicious", {"no-prod-deploy": "suspicious"},
                             "force_ask"),
    "unknown": ActionCase([PROSE], "echo ok", "unknown", {"prose-rule": "unknown"}, "allow"),
    "unknown_high_risk": ActionCase([PROSE_HIGH], "echo ok", "unknown", {"prose-high": "unknown"}, "ask"),
    "clean": ActionCase([NO_FORCE_PUSH], "git status", "clean", {"no-force-push": "clean"}, "allow"),
}


@pytest.mark.parametrize("name", sorted(ACTIONS))
def test_proposed_action_same_judgement_host_specific_response(make_hosts, name):
    case = ACTIONS[name]
    hosts = make_hosts([USER, WORKING], case.contracts)
    responses = {"codex": codex_action(hosts["codex"], case.command),
                 "omp": omp_action(hosts["omp"], case.command),
                 "agy": agy_action(hosts["agy"], case.command)}
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
    if name == "unknown":
        assert "WARNING" in responses["agy"]["reason"] and "not a compliance verdict" in responses["agy"]["reason"]


def test_no_applicable_contract_is_not_a_verdict_on_any_host(make_hosts):
    hosts = make_hosts([USER, WORKING], [])
    assert codex_action(hosts["codex"], "git push --force") == {}
    assert agy_action(hosts["agy"], "git push --force")["decision"] == "allow"
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


CHECKPOINTS = {
    "unmet_auto_continue": CheckpointCase([USER, WORKING], [ACK], "violation", ("ack", "violation", "applicable"),
                                          False, "continue"),
    "met": CheckpointCase([USER, WORKING, DONE], [ACK], "clean", ("ack", "clean", "applicable"), True, "allow"),
    "when_not_met": CheckpointCase([USER, WORKING], [dict(ACK, when={"tool_call_regex": "docker build"})], "clean",
                                   ("ack", "clean", "not_applicable"), True, "allow"),
    "unmet_without_auto_continue": CheckpointCase([USER, WORKING], [dict(ACK, auto_continue=False)], "violation",
                                                  ("ack", "violation", "applicable"), False, "allow"),
}


@pytest.mark.parametrize("name", sorted(CHECKPOINTS))
def test_checkpoint_same_judgement_host_specific_response(make_hosts, name):
    case = CHECKPOINTS[name]
    hosts = make_hosts(case.lines, case.contracts)
    responses = {"codex": codex_stop(hosts["codex"]), "omp": omp_stop(hosts["omp"]), "agy": agy_stop(hosts["agy"])}
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
    # The continue budget is per conversation: the same checkpoint on the next turn only reports.
    assert agy_stop(hosts["agy"])["decision"] == "allow"


def test_unknown_evidence_is_never_clean_and_never_resumes_on_any_host(make_hosts):
    """Description-only obligation: no verifier exists, so all hosts judge unknown and none act on it."""
    hosts = make_hosts([USER, WORKING], [dict(PROSE, type="must", id="prose-must")])
    responses = {"codex": codex_stop(hosts["codex"]), "omp": omp_stop(hosts["omp"]), "agy": agy_stop(hosts["agy"])}
    judged = _same_judgement(hosts, "audit_session")
    assert judged["verdict"] == "unknown" and judged["findings"] == [("prose-must", "unknown", "applicable")]
    assert _facts(responses["codex"])["verdict"] == "unknown"
    assert responses["omp"] == {} and responses["agy"]["decision"] == "allow"
