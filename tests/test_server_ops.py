"""Request statistics at the server: one count per request, taken at its response; ``audit_status.ops``; the timings an
incomplete answer carries; and that nothing but closed names and numbers is ever recorded."""
from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from test_reliability import session, SID
from self_directing_mcp import engine as engine_module
from self_directing_mcp import request_stats, server
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.request_control import IndexBusy, RequestStopped, check_request
from self_directing_mcp.schemas import ProposedAction

OLD_KEYS = {"ok", "verdict", "error", "coverage", "action_executed", "requires_attention", "nudge"}
NO_DELETE = {"id": "no-delete", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf",
             "description": "Never delete user files."}
LEAK = "sk-LIVE-SECRET-1234567890abcdef /home/someone/private 11111111-2222-3333-4444-555555555555"


@pytest.fixture(autouse=True)
def stats(monkeypatch):
    fresh = request_stats.RequestStats()
    monkeypatch.setattr(server, "STATS", fresh)
    monkeypatch.setattr(server._engine.settings, "max_concurrent_calls", 2)
    monkeypatch.setattr(server, "_libc", lambda: None)
    return fresh


def tool(name, hold=0.0, raises=None, result=None):
    """A stand-in engine method that plays the tool called ``name``."""
    def method(*args, _deadline=None, cancel_event=None, **kwargs):
        time.sleep(hold)
        if raises is not None:
            raise raises
        return {"ok": True} if result is None else result
    method.__name__ = name
    return method


def only(mapping):
    return {key: value for key, value in mapping.items() if value}


def counted(snapshot, name):
    entry = snapshot["tools"][name]
    return entry["calls"], only(entry["outcomes"]), {outcome: group for outcome, group in entry["reasons"].items() if group}


async def until(condition, seconds=5.0):
    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        await asyncio.sleep(0.02)


def use_engine(monkeypatch, engine):
    monkeypatch.setattr(server, "_engine", engine)
    return engine


def stuck(monkeypatch, share=0.05):
    def wait_for_stop(self, **kwargs):
        while True:
            check_request()
            time.sleep(0.005)

    monkeypatch.setattr(SelfDirectEngine, "_sync_session", wait_for_stop)
    monkeypatch.setattr(engine_module, "SYNC_BUDGET_SHARE", share)


# --- what is counted, and when -------------------------------------------------------------------------------------

def test_the_closed_tool_names_are_the_registered_tools_plus_hook():
    registered = {registered.name for registered in asyncio.run(server.mcp.list_tools())}
    assert set(request_stats.TOOLS) == registered | {"hook"}


def test_a_completed_call_is_one_ok_with_its_slot_wait_and_elapsed_time(stats):
    async def scenario():
        response = await server._dispatch(tool("check_action", hold=0.05), budget_override=10)
        await asyncio.sleep(0.05)
        return response, stats.snapshot()

    response, snapshot = asyncio.run(scenario())
    assert response == {"ok": True}
    assert counted(snapshot, "check_action") == (1, {"ok": 1}, {})
    entry = snapshot["tools"]["check_action"]
    assert sum(entry["elapsed_ms"]["counts"]) == 1 and entry["elapsed_ms"]["max"] >= 40
    assert sum(entry["gate_wait_ms"]["counts"]) == 1
    assert snapshot["gate"] == {"slots": 2, "in_use": 0} and snapshot["workers"]["abandoned_total"] == 0


def test_a_call_that_bypasses_the_gate_adds_no_slot_wait(stats):
    asyncio.run(server._dispatch(tool("get_chunk"), heavy=False, budget_override=10))
    entry = stats.snapshot()["tools"]["get_chunk"]
    assert entry["calls"] == 1 and sum(entry["elapsed_ms"]["counts"]) == 1
    assert sum(entry["gate_wait_ms"]["counts"]) == 0
    assert stats.snapshot()["gate"]["slots"] == 0           # the gate was never touched


def test_the_gate_gauge_is_visible_while_a_call_runs(stats):
    seen = {}

    def spy(*args, _deadline=None, cancel_event=None, **kwargs):
        seen.update(stats.snapshot()["gate"])
        return {"ok": True}

    spy.__name__ = "search_history"
    asyncio.run(server._dispatch(spy, budget_override=10))
    assert seen == {"slots": 2, "in_use": 1}


def test_a_deadline_stop_is_one_incomplete_and_the_late_worker_is_counted_apart(stats):
    async def scenario():
        response = await server._dispatch(tool("check_action", hold=0.8), budget_override=0.1)
        during = stats.snapshot()
        await until(lambda: stats.snapshot()["workers"]["abandoned_now"] == 0)
        return response, during, stats.snapshot()

    response, during, after = asyncio.run(scenario())
    assert response["error"] == "request_deadline" and response["verdict"] == "unknown"
    assert counted(during, "check_action") == (1, {"incomplete": 1}, {"incomplete": {"request_deadline": 1}})
    assert during["workers"] == {"abandoned_now": 1, "abandoned_total": 1, "late_finish_total": 0}
    assert during["gate"]["in_use"] == 1                    # the slot is held until the worker really ends
    assert after["workers"] == {"abandoned_now": 0, "abandoned_total": 1, "late_finish_total": 1}
    assert after["gate"]["in_use"] == 0
    assert after["tools"] == during["tools"]                # the worker's end is not a second request


def test_a_light_call_that_overruns_is_abandoned_without_holding_a_slot(stats):
    async def scenario():
        response = await server._dispatch(tool("audit_status", hold=0.8), heavy=False, budget_override=0.1)
        during = stats.snapshot()
        await until(lambda: stats.snapshot()["workers"]["abandoned_now"] == 0)
        return response, during, stats.snapshot()

    response, during, after = asyncio.run(scenario())
    assert response["error"] == "request_deadline"
    assert during["workers"]["abandoned_now"] == 1 and during["gate"]["in_use"] == 0
    assert after["workers"] == {"abandoned_now": 0, "abandoned_total": 1, "late_finish_total": 1}
    assert counted(after, "audit_status")[0] == 1


def test_a_request_that_expires_in_the_queue_never_started_and_leaves_no_worker(stats):
    async def scenario():
        running = [asyncio.create_task(server._dispatch(tool("check_action", hold=0.5), budget_override=10))
                   for _ in range(2)]
        await asyncio.sleep(0.1)
        late = await server._dispatch(tool("audit_session"), budget_override=0.2)
        await asyncio.gather(*running)
        await asyncio.sleep(0.05)
        return late, stats.snapshot()

    late, snapshot = asyncio.run(scenario())
    assert late["error"] == "request_deadline"
    assert counted(snapshot, "audit_session") == (1, {"incomplete": 1}, {"incomplete": {"request_deadline": 1}})
    assert snapshot["workers"] == {"abandoned_now": 0, "abandoned_total": 0, "late_finish_total": 0}
    assert snapshot["tools"]["audit_session"]["gate_wait_ms"]["max"] >= 150     # the wait is measured though it ended in a stop
    assert snapshot["gate"] == {"slots": 2, "in_use": 0}


@pytest.mark.parametrize("error,reason", [
    (IndexBusy(), "index_busy"),
    (RequestStopped("request_deadline"), "request_deadline"),
    (RequestStopped("request_cancelled"), "request_cancelled"),
    (TimeoutError(LEAK), "operation_timeout"),
])
def test_an_incomplete_answer_keeps_its_reason(stats, error, reason):
    response = asyncio.run(server._dispatch(tool("audit_session", raises=error), budget_override=10))
    assert response["error"] == reason and response["coverage"] == {"complete": False, "issues": [reason]}
    assert counted(stats.snapshot(), "audit_session") == (1, {"incomplete": 1}, {"incomplete": {reason: 1}})
    assert LEAK not in json.dumps(stats.snapshot())


def test_a_client_cancellation_is_counted_and_still_raised(stats):
    async def scenario():
        task = asyncio.create_task(server._dispatch(tool("audit_session", hold=0.4), budget_override=10))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await until(lambda: stats.snapshot()["gate"]["in_use"] == 0)
        return stats.snapshot()

    snapshot = asyncio.run(scenario())
    assert counted(snapshot, "audit_session") == (1, {"client_cancelled": 1}, {})
    assert snapshot["workers"]["abandoned_total"] == 0      # a departed client is not a deadline: nothing is abandoned


def test_a_client_cancellation_while_queued_is_counted_with_its_wait(stats):
    async def scenario():
        running = [asyncio.create_task(server._dispatch(tool("check_action", hold=0.6), budget_override=10))
                   for _ in range(2)]
        await asyncio.sleep(0.1)
        queued = asyncio.create_task(server._dispatch(tool("audit_session"), budget_override=10))
        await asyncio.sleep(0.2)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        await asyncio.gather(*running)
        await asyncio.sleep(0.05)
        return stats.snapshot()

    snapshot = asyncio.run(scenario())
    assert counted(snapshot, "audit_session") == (1, {"client_cancelled": 1}, {})
    assert snapshot["tools"]["audit_session"]["gate_wait_ms"]["max"] >= 150
    assert snapshot["gate"]["in_use"] == 0


def test_an_unexpected_failure_is_counted_as_an_error_and_raised_as_before(stats):
    with pytest.raises(ValueError) as failed:
        asyncio.run(server._dispatch(tool("search_history", raises=ValueError(LEAK)), budget_override=10))
    assert str(failed.value) == LEAK
    assert counted(stats.snapshot(), "search_history") == (1, {"error": 1}, {})
    assert "SECRET" not in json.dumps(stats.snapshot()) and "someone" not in json.dumps(stats.snapshot())


@pytest.mark.parametrize("kind", ["ok", "deadline", "busy", "operation", "error", "cancel"])
def test_every_request_is_counted_exactly_once(stats, kind):
    errors = {"busy": IndexBusy(), "operation": TimeoutError("x"), "error": RuntimeError("x")}

    async def scenario():
        if kind == "deadline":
            await server._dispatch(tool("check_action", hold=0.3), budget_override=0.1)
        elif kind == "cancel":
            task = asyncio.create_task(server._dispatch(tool("check_action", hold=0.3), budget_override=10))
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif kind == "error":
            with pytest.raises(RuntimeError):
                await server._dispatch(tool("check_action", raises=errors[kind]), budget_override=10)
        else:
            await server._dispatch(tool("check_action", raises=errors.get(kind)), budget_override=10)
        await asyncio.sleep(0.6)                            # let any late worker end: it must not add a request
        return stats.snapshot()

    snapshot = asyncio.run(scenario())
    assert sum(entry["calls"] for entry in snapshot["tools"].values()) == 1
    assert sum(sum(entry["outcomes"].values()) for entry in snapshot["tools"].values()) == 1


def test_names_that_are_not_tools_and_messages_never_reach_the_statistics(stats):
    async def scenario():
        await server._dispatch(tool(LEAK), budget_override=10)
        await server._dispatch(tool(LEAK, raises=TimeoutError(LEAK)), budget_override=10)
        await server._dispatch(tool("sk-" + "A" * 4000, raises=IndexBusy()), budget_override=10)

    asyncio.run(scenario())
    snapshot = stats.snapshot()
    assert list(snapshot["tools"]) == ["other"] and snapshot["tools"]["other"]["calls"] == 3
    rendered = json.dumps(snapshot)
    for fragment in ("sk-", "SECRET", "someone", "11111111", "AAAA"):
        assert fragment not in rendered
    assert len(rendered) < 6000


@pytest.mark.parametrize("coverage,outcome", [
    ({"complete": False, "issues": ["sync_deadline"]}, "degraded"),
    ({"complete": False, "issues": ["unread_bytes", "sync_deadline"]}, "degraded"),
    ({"complete": False, "issues": ["unread_bytes"]}, "ok"),
    ({"complete": True, "issues": []}, "ok"),
    ({"issues": "sync_deadline"}, "ok"),                    # not a list of issues
    ({"issues": None}, "ok"),
    (None, "ok"),
    ("sync_deadline", "ok"),
], ids=["only", "with another issue", "another issue", "complete", "a string", "none", "no coverage", "a string coverage"])
def test_a_result_whose_coverage_names_sync_deadline_is_counted_degraded(stats, coverage, outcome):
    answer = {"ok": True, "verdict": "unknown", "coverage": coverage}
    assert asyncio.run(server._dispatch(tool("audit_session", result=answer), budget_override=10)) == answer
    assert counted(stats.snapshot(), "audit_session") == (
        1, {outcome: 1}, {"degraded": {"sync_deadline": 1}} if outcome == "degraded" else {})


def test_a_stand_in_without_a_name_is_other(stats):
    class Nameless:
        def __call__(self, *args, _deadline=None, cancel_event=None, **kwargs):
            return {"ok": True}

    asyncio.run(server._dispatch(Nameless(), budget_override=10))
    assert list(stats.snapshot()["tools"]) == ["other"]


# --- the answer of a request that did not complete ------------------------------------------------------------------

def test_the_incomplete_answer_is_additive_and_bare_calls_keep_the_old_shape():
    bare = server._incomplete_response("request_deadline")
    assert set(bare) == OLD_KEYS and bare["error"] == "request_deadline"
    trace = server._Trace("check_action", 8.0)
    trace.gate_wait = 0.0125
    extended = server._incomplete_response("request_deadline", trace)
    assert set(extended) == OLD_KEYS | {"timings_ms", "budget_ms"}
    assert {key: extended[key] for key in OLD_KEYS} == bare
    assert extended["budget_ms"] == 8000 and set(extended["timings_ms"]) == {"gate_wait", "elapsed"}
    assert extended["timings_ms"]["gate_wait"] == 12.5 and extended["timings_ms"]["elapsed"] >= 0


def test_a_queue_expiry_says_the_time_went_to_waiting_for_a_slot(stats):
    async def scenario():
        running = [asyncio.create_task(server._dispatch(tool("check_action", hold=0.6), budget_override=10))
                   for _ in range(2)]
        await asyncio.sleep(0.1)
        late = await server._dispatch(tool("audit_session"), budget_override=0.25)
        await asyncio.gather(*running)
        return late

    late = asyncio.run(scenario())
    assert late["error"] == "request_deadline" and late["budget_ms"] == 250
    waited, elapsed = late["timings_ms"]["gate_wait"], late["timings_ms"]["elapsed"]
    assert 200 <= waited <= elapsed < 5000


def test_an_overrun_says_the_time_went_to_the_work_not_the_queue(stats):
    response = asyncio.run(server._dispatch(tool("check_action", hold=0.4), budget_override=0.15))
    assert response["error"] == "request_deadline" and response["budget_ms"] == 150
    assert response["timings_ms"]["gate_wait"] < 100
    assert 140 <= response["timings_ms"]["elapsed"] < 5000


def test_a_call_that_bypasses_the_gate_reports_no_wait(stats):
    response = asyncio.run(server._dispatch(tool("audit_status", raises=IndexBusy()), heavy=False, budget_override=3))
    assert response["error"] == "index_busy" and response["timings_ms"]["gate_wait"] == 0.0
    assert response["budget_ms"] == 3000


# --- audit_status.ops ------------------------------------------------------------------------------------------------

def test_audit_status_includes_the_statistics_and_keeps_its_other_output(session, monkeypatch, stats):
    engine, _ = session
    use_engine(monkeypatch, engine)
    asyncio.run(server._dispatch(tool("check_action"), budget_override=10))
    status = asyncio.run(server.audit_status())
    assert status["ok"] is True
    assert set(status) - {"ops"} == set(engine.audit_status())
    ops = status["ops"]
    assert counted(ops, "check_action") == (1, {"ok": 1}, {})
    assert counted(ops, "audit_status")[:2] == (1, {"ok": 1})        # this very call is in its own snapshot
    assert ops["since"] == stats.snapshot()["since"] and ops["gate"]["in_use"] == 0
    json.dumps(status, allow_nan=False)


@pytest.mark.parametrize("timeout_ms,error", [(None, "index_busy"), (50, "request_deadline")])
def test_ops_is_attached_to_the_incomplete_answer_while_the_engine_lock_is_held(session, monkeypatch, stats,
                                                                                timeout_ms, error):
    engine, _ = session
    use_engine(monkeypatch, engine)
    engine.settings.lock_wait_timeout_sec = 0.1
    held, release = threading.Event(), threading.Event()

    def hold():
        with engine._lock:
            held.set()
            release.wait(10)

    thread = threading.Thread(target=hold)
    thread.start()
    assert held.wait(5)
    try:
        status = asyncio.run(server.audit_status(timeout_ms=timeout_ms))
    finally:
        release.set()
        thread.join(5)
    assert status["error"] == error and status["verdict"] == "unknown" and status["coverage"]["complete"] is False
    assert counted(status["ops"], "audit_status") == (1, {"incomplete": 1}, {"incomplete": {error: 1}})
    assert {"timings_ms", "budget_ms"} <= set(status)
    assert status["ops"]["gate"]["in_use"] == 0
    json.dumps(status, allow_nan=False)


def test_ops_does_not_depend_on_what_the_engine_answers(monkeypatch, stats):
    class Refusing:
        settings = server._engine.settings

        def audit_status(self, **kwargs):
            raise IndexBusy()

    use_engine(monkeypatch, Refusing())
    status = asyncio.run(server.audit_status())
    assert status["error"] == "index_busy" and counted(status["ops"], "audit_status")[0] == 1


# --- degraded answers, and the hook tool ----------------------------------------------------------------------------

def test_a_check_whose_sync_ran_out_of_time_is_counted_degraded_not_ok(session, monkeypatch, stats):
    engine, _ = session
    use_engine(monkeypatch, engine)
    engine.upsert_contracts([NO_DELETE])
    stuck(monkeypatch)
    action = ProposedAction(tool_name="shell", arguments={"command": "rm -rf /fictional-path"})
    response = asyncio.run(server.check_action(SID, action, timeout_ms=8000))
    assert response["verdict"] == "violation" and "sync_deadline" in response["coverage"]["issues"]
    assert counted(stats.snapshot(), "check_action") == (1, {"degraded": 1}, {"degraded": {"sync_deadline": 1}})
    harmless = ProposedAction(tool_name="shell", arguments={"command": "echo harmless"})
    response = asyncio.run(server.check_action(SID, harmless, timeout_ms=8000))
    assert response["verdict"] == "unknown"
    audited = asyncio.run(server.audit_session(SID, timeout_ms=8000))
    assert audited["verdict"] == "unknown" and "sync_deadline" in audited["coverage"]["issues"]
    snapshot = stats.snapshot()
    assert counted(snapshot, "check_action")[1] == {"degraded": 2}
    assert counted(snapshot, "audit_session") == (1, {"degraded": 1}, {"degraded": {"sync_deadline": 1}})


def test_a_hook_whose_sync_ran_out_of_time_still_reports_the_violation_and_is_counted_degraded(session, monkeypatch, stats):
    engine, path = session
    use_engine(monkeypatch, engine)
    engine.settings.audit_timeout_sec = 8.0
    engine.upsert_contracts([NO_DELETE])
    stuck(monkeypatch)
    answer = asyncio.run(server.codex_session_hook(
        event_name="PreToolUse", session_id=SID, transcript_path=str(path), tool_name="shell",
        tool_input={"command": "rm -rf /fictional-path"}))
    context = answer["hookSpecificOutput"]["additionalContext"]
    assert '"verdict": "violation"' in context and "sync_deadline" in context
    assert counted(stats.snapshot(), "hook") == (1, {"degraded": 1}, {"degraded": {"sync_deadline": 1}})


@pytest.mark.parametrize("event", ["PreToolUse", "Stop"])
def test_a_hook_that_cannot_finish_keeps_its_advisory_message_and_is_counted_incomplete(monkeypatch, stats, event):
    def busy(*args, **kwargs):
        raise IndexBusy()

    monkeypatch.setattr(server, "handle_hook", busy)
    answer = asyncio.run(server._dispatch_hook(event, SID, None, "shell", {}))
    text = "Self-directing audit: unknown (index_busy); do not claim compliance."
    assert answer == ({"systemMessage": text} if event == "Stop" else
                      {"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}})
    assert counted(stats.snapshot(), "hook") == (1, {"incomplete": 1}, {"incomplete": {"index_busy": 1}})


def test_the_requests_deadline_during_the_fallback_is_still_the_incomplete_answer(session, monkeypatch, stats):
    engine, _ = session
    use_engine(monkeypatch, engine)
    engine.upsert_contracts([NO_DELETE])
    stuck(monkeypatch, share=0.2)
    original = engine_module.coverage_from_capture

    def slow_coverage(*args, **kwargs):
        time.sleep(1.0)             # the sync ended at 0.2 s of 1 s; this overruns the request, which was answered at 1 s
        check_request()
        return original(*args, **kwargs)

    monkeypatch.setattr(engine_module, "coverage_from_capture", slow_coverage)
    action = ProposedAction(tool_name="shell", arguments={"command": "rm -rf /fictional-path"})

    async def scenario():
        response = await server.check_action(SID, action, timeout_ms=1000)
        during = stats.snapshot()
        await until(lambda: stats.snapshot()["workers"]["abandoned_now"] == 0)
        return response, during, stats.snapshot()

    response, during, after = asyncio.run(scenario())
    assert response["error"] == "request_deadline" and response["verdict"] == "unknown"
    assert counted(after, "check_action") == (1, {"incomplete": 1}, {"incomplete": {"request_deadline": 1}})
    assert during["workers"]["abandoned_now"] == 1
    assert after["workers"] == {"abandoned_now": 0, "abandoned_total": 1, "late_finish_total": 1}
    assert after["gate"]["in_use"] == 0
