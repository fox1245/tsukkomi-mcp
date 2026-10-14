"""End to end: an incremental engine against a twin that evaluates every audit in full.

Two engines read the same growing session file into separate indexes. One audits incrementally (shadow comparison
off, so the delta path alone answers); the twin has ``audit_incremental=false``. After every append both answer
``audit_session`` and ``check_action`` and the whole payload must be identical (findings, verdict, notes, coverage,
graph trace), including parse errors, unsupported events, rewrites, compressed sources and duplicate sources.
"""
from __future__ import annotations

import copy
import json
import random
import re
import shutil

import pytest

from self_directing_mcp.audit import incremental as inc
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from incremental_support import contracts, proposed_action, words
from test_reliability import SID, call, message, result, write

ACTION = {"tool_name": "shell", "arguments": {"command": "deploy now"}}


def stable(payload):
    out = copy.deepcopy(payload)
    for key in ("timings_ms", "incremental"):
        out.pop(key, None)
    out["execution"] = {key: out["execution"].get(key) for key in ("executor", "graph", "nodes")}
    out["coverage"].pop("synced_at", None)  # the moment each twin wrote its own cursor
    # AGY host receipts carry a random id per twin, which is part of an event id and of the receipt digest.
    out["coverage"].pop("receipt_digest", None)
    for finding in out["findings"]:
        for item in finding["evidence"]:
            item["chunk_id"] = re.sub(r":[0-9a-f]{16}$", "", item["chunk_id"])
    return out


class Twin:
    def __init__(self, tmp_path, provider, **paths):
        base = dict(_env_file=None, local_only=True, openrouter_api_key_file=None, **paths)
        self.fast = SelfDirectEngine(Settings(index_dir=tmp_path / "fast", audit_incremental=True,
                                              audit_shadow_every=0, **base))
        self.full = SelfDirectEngine(Settings(index_dir=tmp_path / "full", audit_incremental=False, **base))
        self.provider, self.reports = provider, []

    def upsert(self, rules):
        for engine in (self.fast, self.full):
            engine.upsert_contracts([r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in rules])

    def both(self, method, *args, **kwargs):
        fast = getattr(self.fast, method)(*args, **kwargs)
        full = getattr(self.full, method)(*args, **kwargs)
        self.reports.append(fast.get("incremental"))
        assert stable(fast) == stable(full), (fast.get("incremental"), fast["findings"], full["findings"])
        return fast, full

    def audit(self, sid, **kwargs):
        return self.both("audit_session", sid, provider=self.provider, **kwargs)

    def check(self, sid, action=ACTION, **kwargs):
        return self.both("check_action", sid, action, provider=self.provider, **kwargs)

    def close(self):
        self.fast.close()
        self.full.close()


@pytest.fixture
def codex(tmp_path, monkeypatch):
    for name in ("SELF_DIRECT_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl"
    write(path, [{"type": "session_meta", "payload": {"id": SID}}])
    twin = Twin(tmp_path, "codex", codex_sessions_dir=root)
    yield twin, path
    twin.close()


def codex_records(rng, count, numbered):
    out = []
    for _ in range(count):
        choice = rng.random()
        if choice < 0.2:
            out.append(message(words(rng), rng.choice(["user", "assistant"])))
        elif choice < 0.55:
            numbered.append(f"c{len(numbered)}")
            out.append(call(words(rng), numbered[-1]))
        elif choice < 0.9 and numbered:
            out.append(result(rng.choice(numbered), rng.choice([0, 0, 1, 2])))
        else:
            out.append({"type": "turn_context", "payload": {"note": words(rng)}})
    return out


RULES = [
    {"id": "no-deploy", "type": "must_not", "scope": "tool_call", "regex": "deploy", "roles": ["assistant"]},
    {"id": "tests-first", "type": "must", "scope": "tool_call", "regex": "test", "before_regex": "deploy",
     "requires_success": True},
    {"id": "tests-ran", "type": "must", "scope": "tool_call", "regex": "build", "when": {"tool_call_regex": "push"}},
    {"id": "no-secret-output", "type": "must_not", "scope": "tool_result", "regex": "fail", "roles": ["tool"],
     "exception_regex": "ok"},
    {"id": "said-ok", "type": "must", "scope": "message", "regex": "ok", "roles": ["assistant"]},
]


@pytest.mark.parametrize("seed", range(12))
def test_random_codex_sessions_answer_identically_after_every_append(codex, seed):
    twin, path = codex
    rng = random.Random(f"engine/{seed}")
    numbered = []
    write(path, codex_records(rng, 12, numbered), "a")
    twin.upsert(RULES)
    twin.audit(SID)
    ids = [c.chunk_id for c in twin.fast.store.list_chunks(SID, "codex")]
    twin.upsert([{"id": "anchored", "type": "must_not", "scope": "tool_call", "regex": "push", "roles": ["assistant"],
                  "applies_from_event_id": rng.choice(ids[:6]), "source_event_id": rng.choice(ids)}])
    for _ in range(14):
        write(path, codex_records(rng, rng.choice([1, 2, 5, 11]), numbered), "a")
        twin.audit(SID)
        if rng.random() < 0.5:
            twin.check(SID, {"tool_name": "shell", "arguments": {"command": words(rng) + " deploy"}})
        if rng.random() < 0.08:
            with path.open("a") as stream:
                stream.write("{not json\n" if rng.random() < 0.5 else json.dumps({"type": "future_event"}) + "\n")
    modes = [r["mode"] for r in twin.reports if r]
    assert modes.count("delta") >= 8, modes


def test_the_contract_set_changes_while_the_session_grows(codex):
    twin, path = codex
    rng = random.Random(4)
    numbered = []
    write(path, codex_records(rng, 30, numbered), "a")
    twin.upsert(RULES[:2])
    twin.audit(SID)
    write(path, codex_records(rng, 5, numbered), "a")
    twin.audit(SID)
    # revision bump of one contract, a new contract, a revoked contract
    twin.upsert([{**RULES[0], "regex": "deploy|push"}])
    fast, _ = twin.audit(SID)
    assert set(fast["incremental"]["rebuilt"].values()) <= {"contract_revision", "no_state"}
    twin.upsert([RULES[3]])
    twin.audit(SID)
    for engine in (twin.fast, twin.full):
        engine.revoke_contract("no-deploy")
    write(path, codex_records(rng, 7, numbered), "a")
    fast, _ = twin.audit(SID)
    assert [f["contract_id"] for f in fast["findings"]] == ["tests-first", "no-secret-output"]
    fast, _ = twin.audit(SID)
    assert fast["incremental"]["mode"] == "delta" and not fast["incremental"]["rebuilt"]


def test_a_rewrite_of_earlier_bytes_is_never_judged_from_the_old_state(codex):
    twin, path = codex
    rng = random.Random(8)
    numbered = []
    write(path, codex_records(rng, 40, numbered), "a")
    twin.upsert(RULES)
    twin.audit(SID)
    write(path, codex_records(rng, 4, numbered), "a")
    twin.audit(SID)
    lines = path.read_text().splitlines()
    lines[10] = json.dumps(call("rm -rf deploy deploy", "tampered"))
    path.write_text("\n".join(lines) + "\n")
    fast, _ = twin.audit(SID)
    assert "source_changed" in fast["coverage"]["issues"] or fast["incremental"]["mode"] == "full"
    write(path, codex_records(rng, 4, numbered), "a")
    twin.audit(SID)
    twin.check(SID)


def test_parse_errors_unsupported_events_and_unread_bytes(codex):
    twin, path = codex
    rng = random.Random(6)
    numbered = []
    write(path, codex_records(rng, 20, numbered), "a")
    twin.upsert(RULES)
    twin.audit(SID)
    with path.open("a") as stream:
        stream.write("{broken\n")
    fast, _ = twin.audit(SID)
    assert "parse_errors" in fast["coverage"]["issues"]
    with path.open("a") as stream:
        stream.write(json.dumps({"type": "totally_new_event"}) + "\n")
        stream.write('{"type": "response_item", "payload": {"type": "mess')  # an unfinished record
    fast, _ = twin.audit(SID)
    assert {"unsupported_events", "unread_bytes"} <= set(fast["coverage"]["issues"])
    twin.check(SID)


def test_a_duplicate_source_is_judged_like_the_indexed_one(codex, tmp_path):
    twin, path = codex
    rng = random.Random(2)
    numbered = []
    write(path, codex_records(rng, 25, numbered), "a")
    twin.upsert(RULES)
    twin.audit(SID)
    copy_path = path.parent / f"rollout-2026-09-09T02-00-00-{SID}.jsonl"
    shutil.copy(path, copy_path)
    fast, _ = twin.audit(SID, path=str(copy_path))
    assert fast["coverage"].get("indexed_path") or fast["coverage"]["complete"]
    write(path, codex_records(rng, 3, numbered), "a")
    twin.audit(SID)
    twin.audit(SID, path=str(copy_path))


def test_a_compressed_rollout_is_audited_incrementally(codex, tmp_path):
    zstandard = pytest.importorskip("zstandard")
    twin, path = codex
    rng = random.Random(3)
    numbered = []
    write(path, codex_records(rng, 30, numbered), "a")
    twin.upsert(RULES)
    twin.audit(SID)
    packed = path.parent / (path.name + ".zst")
    packed.write_bytes(zstandard.ZstdCompressor().compress(path.read_bytes()))
    path.unlink()
    fast, _ = twin.audit(SID, path=str(packed))
    twin.audit(SID, path=str(packed))
    twin.check(SID, path=str(packed))


def test_an_untouched_session_is_answered_without_reading_events(codex):
    twin, path = codex
    rng = random.Random(1)
    numbered = []
    write(path, codex_records(rng, 40, numbered), "a")
    twin.upsert(RULES)
    twin.audit(SID)
    twin.audit(SID)
    fast, _ = twin.audit(SID)
    assert fast["incremental"]["mode"] == "delta" and fast["incremental"]["events_read"] == 0


# --- shadow verification, mismatches and the health table -----------------------------------------------------

def corrupt(engine, field, value):
    connection = engine.store._conn
    with connection:
        for row_id, state in connection.execute("SELECT rowid, state FROM audit_monitor_state WHERE monitor_id != '#facts'"
                                                ).fetchall():
            data = json.loads(state)
            data[field] = value
            connection.execute("UPDATE audit_monitor_state SET state=? WHERE rowid=?", (json.dumps(data), row_id))


@pytest.mark.allow_incremental_mismatch
def test_a_shadow_mismatch_returns_the_full_result_and_disables_that_contract(codex, tmp_path, monkeypatch):
    twin, path = codex
    write(path, [call(f"echo step {i}", f"b{i}") for i in range(30)], "a")
    rule = {"id": "no-deploy", "type": "must_not", "scope": "tool_call", "regex": "deploy", "roles": ["assistant"]}
    twin.upsert([rule])
    write(path, [call("echo hello", "keep")], "a")
    twin.audit(SID)
    assert twin.fast.audit_session(SID, provider="codex")["verdict"] == "clean"
    # Poison the cache so that it says a forbidden call was seen.
    corrupt(twin.fast, "hits", 7)
    twin.fast.settings.audit_shadow_every = 1
    write(path, [call("echo again", "next")], "a")
    answer = twin.fast.audit_session(SID, provider="codex")
    reference = twin.full.audit_session(SID, provider="codex")
    assert stable(answer) == stable(reference) and answer["verdict"] == "clean"
    assert answer["incremental"]["mismatch"] == ["no-deploy"] and answer["incremental"]["shadow_compared"]
    health = twin.fast.store._conn.execute("SELECT status FROM audit_incremental_health WHERE monitor_id='no-deploy'").fetchall()
    assert [row[0] for row in health] == ["mismatch"]
    assert twin.fast.store._conn.execute("SELECT COUNT(*) FROM audit_monitor_state WHERE monitor_id='no-deploy'").fetchone()[0] == 0
    # From now on that contract is evaluated in full, in this process and after a restart.
    twin.fast.settings.audit_shadow_every = 0
    write(path, [call("echo more", "more")], "a")
    answer = twin.fast.audit_session(SID, provider="codex")
    assert answer["incremental"]["fallback"]["no-deploy"] == "health:mismatch"
    assert stable(answer) == stable(twin.full.audit_session(SID, provider="codex"))
    twin.fast.close()
    reopened = SelfDirectEngine(twin.fast.settings)
    try:
        write(path, [call("echo once more", "again2")], "a")
        answer = reopened.audit_session(SID, provider="codex")
        assert answer["incremental"]["fallback"]["no-deploy"] == "health:mismatch"
        # A new evaluator version is a new chance.
        monkeypatch.setattr(inc, "EVALUATOR_VERSION", inc.EVALUATOR_VERSION + 1)
        write(path, [call("echo and again", "again3")], "a")
        answer = reopened.audit_session(SID, provider="codex")
        assert not answer["incremental"]["fallback"] and answer["incremental"]["mode"] == "full"
        write(path, [call("echo last", "again4")], "a")
        answer = reopened.audit_session(SID, provider="codex")
        assert answer["incremental"]["mode"] == "delta"
    finally:
        reopened.close()


def test_an_audit_samples_the_shadow_comparison_and_says_so(codex):
    twin, path = codex
    rng = random.Random(10)
    numbered = []
    write(path, codex_records(rng, 20, numbered), "a")
    twin.upsert(RULES[:2])
    twin.fast.settings.audit_shadow_every = 1
    first = twin.fast.audit_session(SID, provider="codex")
    assert first["incremental"]["shadow"] is True and first["incremental"]["shadow_compared"] is True
    twin.fast.settings.audit_shadow_every = 0
    write(path, codex_records(rng, 3, numbered), "a")
    second = twin.fast.audit_session(SID, provider="codex")
    assert second["incremental"]["shadow"] is False and "shadow_compared" not in second["incremental"]


def test_incremental_audits_can_be_switched_off(codex):
    twin, path = codex
    rng = random.Random(11)
    numbered = []
    write(path, codex_records(rng, 20, numbered), "a")
    twin.upsert(RULES[:2])
    twin.fast.settings.audit_incremental = False
    payload = twin.fast.audit_session(SID, provider="codex")
    assert "incremental" not in payload
    assert twin.fast.store._conn.execute("SELECT COUNT(*) FROM audit_monitor_state").fetchone()[0] == 0


# --- the other providers --------------------------------------------------------------------------------------

def test_claude_sessions_with_cut_tool_results(tmp_path, monkeypatch):
    from test_claude_provider import (PROJECT, SID as CLAUDE, assistant, attachment, human, main_file, tool_result,
                                      use, write as write_claude)
    for name in ("CLAUDE_CONFIG_DIR", "SELF_DIRECT_CLAUDE_PROJECTS_DIR"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "projects"
    path = main_file(root)
    twin = Twin(tmp_path, "claude", claude_projects_dir=root)
    try:
        write_claude(path, human("ship it"), assistant(use("t0", "Bash", command="npm test")),
                     tool_result("t0", "ran", block_extra={"is_error": False}))
        twin.upsert([
            {"id": "gate", "type": "must", "scope": "tool_call", "regex": "npm test", "before_regex": "deploy",
             "requires_success": True, "provider": "claude"},
            {"id": "no-secret", "type": "must_not", "scope": "tool_result", "regex": "LEAKED", "roles": ["tool"],
             "provider": "claude"},
            {"id": "need-ok", "type": "must", "scope": "tool_result", "regex": "BUILD OK", "roles": ["tool"],
             "provider": "claude"},
            {"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": "rm -rf", "roles": ["assistant"],
             "provider": "claude"}])
        twin.audit(CLAUDE)
        twin.check(CLAUDE, {"tool_name": "Bash", "arguments": {"command": "deploy now"}})
        steps = [
            [assistant(use("t1", "Read", file_path="/s/a.txt")), tool_result("t1", "head only",
                                                                            block_extra={"is_error": False})],
            [attachment("read_truncation_notice", banner="cut", toolUseID="t1")],
            [assistant(use("t2", "Bash", command="npm test")), tool_result("t2", "BUILD OK",
                                                                           block_extra={"is_error": False})],
            [attachment("read_truncation_notice", banner="cut", toolUseID="t0")],  # a notice for an older result
            [assistant(use("t3", "Bash", command="rm -rf build")), tool_result("t3", "LEAKED", block_extra={"is_error": True})],
            [assistant(use("t4", "Bash", command="npm test")), tool_result("t4", "Exit code 1", block_extra={"is_error": True})],
        ]
        for records in steps:
            write_claude(path, *records, mode="a")
            twin.audit(CLAUDE)
            twin.check(CLAUDE, {"tool_name": "Bash", "arguments": {"command": "deploy now"}})
        assert [r["mode"] for r in twin.reports if r].count("delta") >= 6
    finally:
        twin.close()


def test_omp_sessions(tmp_path, monkeypatch):
    from test_omp_local import _append, _message
    root = tmp_path / "sessions"
    root.mkdir()
    file = root / "session.jsonl"
    _append(file, {"type": "title", "title": "t"}, {"type": "session", "id": "session-1"},
            _message("user", {"type": "text", "text": "Run tests before deploy"}))
    twin = Twin(tmp_path, "omp", omp_sessions_dir=root)
    try:
        twin.upsert([
            {"id": "gate", "type": "must", "scope": "tool_call", "regex": "npm test", "before_regex": "deploy",
             "requires_success": True, "provider": "omp"},
            {"id": "no-curl", "type": "must_not", "scope": "tool_call", "regex": "curl", "provider": "omp"},
            {"id": "said", "type": "must", "scope": "message", "regex": "done", "roles": ["assistant"], "provider": "omp"}])
        twin.audit("session-1", path=str(file))
        def tool(call_id, command):
            return _message("assistant", {"type": "toolCall", "id": call_id, "name": "shell",
                                          "arguments": {"command": command}})
        def output(call_id, text, error=False):
            return {"type": "message", "message": {"role": "toolResult", "toolCallId": call_id, "toolName": "shell",
                                                   "content": [{"type": "text", "text": text}], "isError": error}}
        steps = [[tool("c1", "npm test"), output("c1", "ok")], [tool("c2", "curl http://x")],
                 [output("c2", "boom", True)], [tool("c3", "npm test"), output("c3", "failed", True)],
                 [_message("assistant", {"type": "text", "text": "all done"})], [tool("c1", "npm test")]]
        for records in steps:
            _append(file, *records)
            twin.audit("session-1", path=str(file))
            twin.check("session-1", {"tool_name": "shell", "arguments": {"command": "deploy now"}}, path=str(file))
        assert [r["mode"] for r in twin.reports if r].count("delta") >= 6
    finally:
        twin.close()


def test_agy_sessions_with_pending_truncated_and_legacy_results(tmp_path):
    from test_agy_parser_v6 import SID as AGY, append, generic, hook, make, planner, user
    root = tmp_path / "app"
    twin = Twin(tmp_path, "agy", agy_app_data_dirs=[root])
    try:
        path = make(root, [user(0, "Run npm test before deploy."),
                           generic(1, "old output", truncated_fields=["content"])])  # before any receipt: legacy
        twin.upsert([
            {"id": "tests-first", "provider": "agy", "type": "must", "scope": "tool_call", "regex": "npm test",
             "before_regex": "deploy", "requires_success": True},
            {"id": "no-rm", "provider": "agy", "type": "must_not", "scope": "tool_call", "regex": "rm -rf"},
            {"id": "no-curl", "provider": "agy", "type": "must_not", "scope": "message", "regex": "curl", "roles": ["user"]},
            {"id": "ran-tests", "provider": "agy", "type": "must", "scope": "tool_call", "regex": "npm test"}])
        twin.audit(AGY)
        deploy = {"tool_name": "run_command", "arguments": {"CommandLine": "deploy"}}
        twin.check(AGY, deploy)
        step = 2
        script = [
            ("npm test", "DONE", None),
            ("rm -rf build", "DONE", None),
            ("npm test", "RUNNING", None),
            ("npm test", "DONE", "again"),
            ("deploy", "ERROR", None),
        ]
        for command, status, _note in script:
            append(path, planner(step, "", [("run_command", {"CommandLine": command})]))
            for engine in (twin.fast, twin.full):
                engine.record_agy_hook(hook(path, step + 1, command), "PreToolUse")
            append(path, generic(step + 1, "tests finished", status=status))
            for engine in (twin.fast, twin.full):
                engine.record_agy_hook(hook(path, step + 1, command), "PostToolUse")
            step += 2
            twin.audit(AGY)
            twin.check(AGY, deploy)
        # A hook receipt changes the receipt digest, which reindexes the conversation (a new generation), so the
        # steps above rebuild. Appends that bring no receipt are incremental again.
        for extra in ([user(step, "keep going", truncated_fields=["content"])],
                      [planner(step + 1, "thinking aloud", None, truncated_fields=["content"])],
                      [user(step + 2, "curl the staging box")]):
            append(path, *extra)
            twin.audit(AGY)
            twin.check(AGY, deploy)
            step += 3
        modes = [r["mode"] for r in twin.reports[-6:] if r]
        assert modes.count("delta") >= 4, twin.reports[-6:]
    finally:
        twin.close()


def test_a_shadow_run_is_skipped_when_the_request_cannot_afford_it(codex, monkeypatch):
    import time
    twin, path = codex
    rng = random.Random(12)
    numbered = []
    write(path, codex_records(rng, 25, numbered), "a")
    twin.upsert(RULES)
    twin.fast.settings.audit_shadow_every = 1
    monkeypatch.setattr(inc, "SHADOW_SECONDS_PER_EVENT", 100.0)
    answer = twin.fast.audit_session(SID, provider="codex", _deadline=time.monotonic() + 30)
    assert answer["incremental"]["shadow"] is False and answer["incremental"]["shadow_skipped"] == "deadline"
    assert stable(answer) == stable(twin.full.audit_session(SID, provider="codex"))
    # No deadline, nothing to protect: the comparison runs.
    write(path, codex_records(rng, 3, numbered), "a")
    answer = twin.fast.audit_session(SID, provider="codex")
    assert answer["incremental"]["shadow"] is True and answer["incremental"]["shadow_compared"] is True
