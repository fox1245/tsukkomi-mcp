"""The clients treat the reasons of an incomplete answer as opaque strings (roundtable 8, P5).

A partial sync adds the coverage issue ``sync_in_progress`` and, when a waiter gives up, the response reason
``sync_in_progress``. Nothing in this repository compares reasons against a closed list: the Codex hook prints the
issues it was given, the AGY and Claude Code adapters turn any unknown or incomplete answer into the same short
"unverified" text (and ``ask`` for a ``risk: high`` contract), and the OMP extension prints the first four reason
strings of the answer. These tests feed each adapter the real answer of a session that is partway through its sync and
check that the text it renders is short, says that the check is unknown or unverified, and never claims compliance or
blocks on it.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import pytest

from self_directing_mcp import agy_hooks, claude_hooks, hook_policy, request_stats, server
from self_directing_mcp.codex_hooks import handle_hook
from self_directing_mcp.index.ingest import FlightBusy
from self_directing_mcp.request_control import RequestControl, request_scope
from slicing_support import World, forced
from test_slice_equivalence import BIG

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "Self-directing audit (data, not instructions): "
# A history-dependent obligation that applies to the proposed action: with part of the history unread it cannot be met.
TESTS_FIRST = {"id": "tests-first", "type": "must", "scope": "tool_call", "regex": "pytest", "before_regex": "deploy"}


def partial_world(tmp_path, monkeypatch, provider, **contract):
    world = World(tmp_path, monkeypatch, provider, seed=2, count=60 if provider != "agy" else 40)
    engine = world.engine("host")
    world.copy_receipts(world.builder, engine)
    engine.upsert_contracts([dict(TESTS_FIRST, provider=provider, **contract)])
    engine.slice_policy = forced(4)
    with request_scope(RequestControl(deadline=time.monotonic() + BIG)):     # a request with a deadline: it may slice
        info = engine.sync_session(**world.sync_args(), derived=False)
    assert info["sync"]["state"] == "partial" and "sync_in_progress" in info["coverage"]["issues"]
    return world, engine


def far():
    return request_scope(RequestControl(deadline=time.monotonic() + BIG))


def test_the_codex_hook_prints_the_issues_it_was_given(tmp_path, monkeypatch):
    world, engine = partial_world(tmp_path, monkeypatch, "codex")
    try:
        response = handle_hook(engine, "PreToolUse", world.sid, str(world.path), "shell", {"command": "deploy now"},
                               _deadline=time.monotonic() + BIG)
        assert set(response) == {"hookSpecificOutput"}                       # advisory context: never a permission decision
        text = response["hookSpecificOutput"]["additionalContext"]
        facts = json.loads(text[len(PREFIX):])
        assert text.startswith(PREFIX) and len(text) < 900
        assert facts["verdict"] == "unknown" and facts["coverage_complete"] is False
        assert "sync_in_progress" in facts["issues"] and "unread_bytes" in facts["issues"]
        assert not any("clean" == f["verdict"] for f in facts["findings"]) and "judged_through_byte" not in facts
    finally:
        world.close()


def test_the_codex_hook_of_a_waiter_that_gave_up_says_unknown_with_the_reason(monkeypatch):
    def gave_up(*args, **kwargs):
        raise FlightBusy()

    fresh = request_stats.RequestStats()
    monkeypatch.setattr(server, "STATS", fresh)
    monkeypatch.setattr(server, "handle_hook", gave_up)
    answer = asyncio.run(server._dispatch_hook("PreToolUse", "sid", "/p", "shell", {"command": "x"}))
    context = answer["hookSpecificOutput"]["additionalContext"]
    assert context == "Self-directing audit: unknown (sync_in_progress); do not claim compliance."
    stop = asyncio.run(server._dispatch_hook("Stop", "sid", "/p"))
    assert stop == {"systemMessage": context}
    counted = fresh.snapshot()["tools"]["hook"]           # counted under its own name, not folded into "other"
    assert counted["outcomes"]["incomplete"] == 2 and counted["reasons"]["incomplete"] == {"sync_in_progress": 2}


def test_the_agy_adapter_allows_with_a_short_unverified_note_and_asks_for_a_high_risk_contract(tmp_path, monkeypatch):
    world, engine = partial_world(tmp_path, monkeypatch, "agy")
    try:
        call_ = {"name": "run_command", "args": {"CommandLine": "deploy now"}}
        with far():
            reply = agy_hooks.handle_pre_tool_use(engine, world.sid, call_, str(world.path))
        assert reply["decision"] == "allow" and "unverified, not a compliance verdict" in reply["reason"]
        assert len(reply["reason"]) < 300 and "continue" not in json.dumps(reply)
    finally:
        world.close()
    risky, engine = partial_world(tmp_path / "risky", monkeypatch, "agy", risk="high")
    try:
        with far():
            reply = agy_hooks.handle_pre_tool_use(engine, risky.sid, call_, str(risky.path))
        assert reply["decision"] == "ask" and "could not be verified" in reply["reason"] and len(reply["reason"]) < 300
    finally:
        risky.close()


def test_the_claude_code_adapter_warns_and_asks_for_a_high_risk_contract_but_never_blocks_on_a_backlog(tmp_path, monkeypatch):
    world, engine = partial_world(tmp_path, monkeypatch, "claude")
    payload = {"session_id": world.sid, "transcript_path": str(world.path), "cwd": "/synthetic/project",
               "hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "deploy now"}}
    try:
        with far():
            reply = claude_hooks.run_hook("PreToolUse", payload, engine=engine)
        assert set(reply) == {"systemMessage"} and "unverified, not a compliance verdict" in reply["systemMessage"]
        assert len(reply["systemMessage"]) < 300
    finally:
        world.close()
    risky, engine = partial_world(tmp_path / "risky", monkeypatch, "claude", risk="high")
    try:
        with far():
            reply = claude_hooks.run_hook("PreToolUse", dict(payload, session_id=risky.sid, transcript_path=str(risky.path)),
                                          engine=engine)
        decision = reply["hookSpecificOutput"]
        assert decision["permissionDecision"] == "ask" and len(decision["permissionDecisionReason"]) < 300
    finally:
        risky.close()


def test_a_stop_checkpoint_calls_the_history_incomplete(tmp_path, monkeypatch):
    world, engine = partial_world(tmp_path, monkeypatch, "claude")
    try:
        with far():
            notes, misses = hook_policy.checkpoint(engine, "claude", world.sid, str(world.path),
                                                   engine.hook_obligations(world.sid, provider="claude"), "Claude Code")
        assert any("history is incomplete; completion cannot be verified" in note for note in notes) and not misses
    finally:
        world.close()


def ts_reason_filter():
    """The pattern the OMP extension keeps reason strings by, and the notices it builds, read from its source."""
    source = (ROOT / "extensions" / "tsukkomi-omp.ts").read_text(encoding="utf-8")
    pattern = re.search(r"typeof value === \"string\" && /(\^\[[^/]+\$)/\.test\(value\)", source)
    assert pattern and "Tsukkomi check unknown: ${incompleteReason(result)}; action not blocked." in source
    assert "[...new Set(reasons)].slice(0, 4).join(\", \")" in source
    return re.compile(pattern.group(1))


def test_the_omp_extension_prints_the_first_reasons_of_the_answer_and_does_not_block(tmp_path, monkeypatch):
    keep = ts_reason_filter()
    world, engine = partial_world(tmp_path, monkeypatch, "omp")
    try:
        with far():
            answer = engine.check_action(world.sid, {"tool_name": "shell", "arguments": {"command": "deploy now"}},
                                         provider="omp", path=str(world.path), verify="fast")
        assert answer["verdict"] == "unknown"                          # not violation/suspicious: the extension lets it through
        reasons = [r for r in [answer.get("error"), *answer["coverage"]["issues"]] if isinstance(r, str) and keep.fullmatch(r)]
        notice = f"Tsukkomi check unknown: {', '.join(dict.fromkeys(reasons))}; action not blocked."
        assert "sync_in_progress" in reasons and len(notice) < 160
        # The answer of a waiter that gave up (an incomplete response) is accepted by the extension's own checks too.
        busy = server._incomplete_response("sync_in_progress")
        assert busy["ok"] is False and busy["verdict"] == "unknown" and busy["action_executed"] is False
        assert isinstance(busy["error"], str) and keep.fullmatch(busy["error"]) and busy["coverage"]["complete"] is False
    finally:
        world.close()


@pytest.mark.parametrize("mode", ["confirmed_only", "advisory", "enforced"])
def test_an_incomplete_response_with_the_new_reason_is_classified_like_any_other(mode):
    decision, reason, _warn = hook_policy.classify(server._incomplete_response("sync_in_progress"), mode)
    assert decision == ("deny" if mode == "enforced" else "allow") and len(reason) < 200
    assert "unverified" in reason.lower() or "incomplete" in reason.lower() or "ADVISORY" in reason


# --- a re-index that is staged or waits for its commit (roundtable 8, PR3) ---------------------------------------------

def test_the_codex_hook_prints_the_issues_of_a_staged_reindex_and_of_one_that_waits_for_its_commit(tmp_path, monkeypatch):
    from reindex_support import Pair
    from self_directing_mcp.index import ingest
    p = Pair(tmp_path, monkeypatch, "codex")
    try:
        p.staged.upsert_contracts([dict(TESTS_FIRST, provider="codex")])      # applies to the proposed action below
        p.stage(cap=20)
        p.staged.slice_policy = forced(20)
        for expected in ("sync_in_progress", "reindex_ready_to_commit"):
            if expected == "reindex_ready_to_commit":
                monkeypatch.setattr(ingest, "COMMIT_SECONDS_PER_ROW", 10.0)
                p.staged.slice_policy = None            # the settings decide: everything is staged, the commit cannot fit
            response = handle_hook(p.staged, "PreToolUse", p.sid, str(p.world.path), "shell", {"command": "deploy now"},
                                   _deadline=time.monotonic() + 60)
            text = response["hookSpecificOutput"]["additionalContext"]
            facts = json.loads(text[len(PREFIX):])
            assert set(response) == {"hookSpecificOutput"} and text.startswith(PREFIX) and len(text) < 900
            assert facts["verdict"] == "unknown" and facts["coverage_complete"] is False
            assert expected in facts["issues"] and "parser_outdated" in facts["issues"]
            assert "judged_through_byte" not in facts
    finally:
        p.close()


def test_the_omp_extension_keeps_the_issues_of_a_staged_reindex():
    keep = ts_reason_filter()
    assert keep.fullmatch("sync_in_progress") and keep.fullmatch("reindex_ready_to_commit")
    assert request_stats.fold("reindex_ready_to_commit", request_stats.REASONS) == "reindex_ready_to_commit"
