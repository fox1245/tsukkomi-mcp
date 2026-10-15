"""Request statistics: closed vocabularies, fixed buckets, the shape of the snapshot, thread safety, and no content.

Pure unit tests of ``self_directing_mcp.request_stats``; the wiring into the server is in ``test_server_ops.py``.
"""
from __future__ import annotations

from datetime import datetime
import json
import threading

import pytest

from self_directing_mcp import request_stats
from self_directing_mcp.request_stats import BUCKETS_MS, OUTCOMES, REASONS, TOOLS, Notes, RequestStats

HOSTILE = [
    "sk-LIVE-SECRET-1234567890abcdef",
    "/home/someone/.ssh/id_rsa",
    "11111111-2222-3333-4444-555555555555",
    "rm -rf / --no-preserve-root",
    "line one\nline two\x00",
    "contract text: never touch the production database",
    "x" * 5000,
]


@pytest.fixture
def stats():
    return RequestStats()


def only(mapping):
    """The non-zero entries of a counter mapping."""
    return {key: value for key, value in mapping.items() if value}


def leaves(value):
    """Every scalar of a JSON-like value, asserting that it is plain data with string keys."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str)
            yield from leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from leaves(item)
    else:
        assert value is None or isinstance(value, (str, int, float, bool)), type(value)
        yield value


def test_the_vocabularies_are_the_documented_closed_sets():
    assert OUTCOMES == ("ok", "degraded", "incomplete", "client_cancelled", "error")
    assert REASONS == ("request_deadline", "index_busy", "request_cancelled", "operation_timeout", "sync_deadline", "other")
    assert BUCKETS_MS == (50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 45000)
    assert "hook" in TOOLS and "check_action" in TOOLS and "audit_status" in TOOLS
    assert len(set(TOOLS)) == len(TOOLS)


def test_requests_are_counted_per_tool_outcome_and_reason(stats):
    stats.record("check_action", "ok", elapsed_ms=12)
    stats.record("check_action", "incomplete", reason="request_deadline", elapsed_ms=8000)
    stats.record("check_action", "incomplete", reason="request_deadline", elapsed_ms=8001)
    stats.record("check_action", "incomplete", reason="index_busy", elapsed_ms=1000)
    stats.record("check_action", "degraded", reason="sync_deadline", elapsed_ms=7500)
    stats.record("audit_session", "client_cancelled", elapsed_ms=30)
    stats.record("hook", "error", elapsed_ms=3)
    tools = stats.snapshot()["tools"]
    assert sorted(tools) == ["audit_session", "check_action", "hook"]
    check = tools["check_action"]
    assert check["calls"] == 5
    assert only(check["outcomes"]) == {"ok": 1, "incomplete": 3, "degraded": 1}
    assert check["reasons"] == {"incomplete": {"request_deadline": 2, "index_busy": 1},
                                "degraded": {"sync_deadline": 1}}
    assert only(tools["audit_session"]["outcomes"]) == {"client_cancelled": 1}
    assert only(tools["hook"]["outcomes"]) == {"error": 1}
    assert tools["hook"]["reasons"] == {"incomplete": {}, "degraded": {}}
    assert sum(only(check["outcomes"]).values()) == check["calls"]


def test_only_incomplete_and_degraded_requests_carry_a_reason(stats):
    stats.record("check_action", "ok", reason="request_deadline")
    stats.record("check_action", "error", reason="index_busy")
    stats.record("check_action", "client_cancelled", reason="request_cancelled")
    stats.record("check_action", "incomplete")                      # a missing reason is counted as other
    stats.record("check_action", "degraded", reason="no such reason")
    check = stats.snapshot()["tools"]["check_action"]
    assert check["calls"] == 5
    assert check["reasons"] == {"incomplete": {"other": 1}, "degraded": {"other": 1}}


@pytest.mark.parametrize("milliseconds,bucket", [
    (0, 0), (0.4, 0), (50, 0), (50.0001, 1), (100, 1), (100.5, 2), (250, 2), (500, 3), (1000, 4), (1000.1, 5),
    (2000, 5), (4000, 6), (8000, 7), (8000.5, 8), (16000, 8), (45000, 9), (45000.5, 10), (10 ** 9, 10),
])
def test_histogram_buckets_have_inclusive_upper_bounds(stats, milliseconds, bucket):
    stats.record("sync_session", "ok", elapsed_ms=milliseconds, gate_wait_ms=milliseconds)
    entry = stats.snapshot()["tools"]["sync_session"]
    for name in ("elapsed_ms", "gate_wait_ms"):
        counts = entry[name]["counts"]
        assert len(counts) == len(BUCKETS_MS) + 1
        assert counts[bucket] == 1 and sum(counts) == 1, (name, counts)


def test_the_maximum_is_kept_and_a_call_that_bypasses_the_gate_adds_no_wait(stats):
    for elapsed in (10, 300, 20):
        stats.record("audit_status", "ok", elapsed_ms=elapsed)                  # no gate_wait_ms: it never queued
    entry = stats.snapshot()["tools"]["audit_status"]
    assert entry["elapsed_ms"]["max"] == 300.0
    assert sum(entry["gate_wait_ms"]["counts"]) == 0 and entry["gate_wait_ms"]["max"] == 0.0
    stats.record("audit_status", "ok", elapsed_ms=1, gate_wait_ms=0)            # a measured wait of nothing is still a wait
    assert sum(stats.snapshot()["tools"]["audit_status"]["gate_wait_ms"]["counts"]) == 1


@pytest.mark.parametrize("value", [float("nan"), -5, None, "slow", object()])
def test_unusable_durations_count_as_zero(stats, value):
    stats.record("check_action", "ok", elapsed_ms=value, gate_wait_ms=value if value is not None else 0)
    entry = stats.snapshot()["tools"]["check_action"]
    assert entry["elapsed_ms"]["counts"][0] == 1 and entry["elapsed_ms"]["max"] == 0.0


def test_an_infinite_duration_stays_a_finite_number(stats):
    stats.record("check_action", "ok", elapsed_ms=float("inf"))
    entry = stats.snapshot()["tools"]["check_action"]
    assert entry["elapsed_ms"]["counts"][-1] == 1
    json.dumps(stats.snapshot(), allow_nan=False)


@pytest.mark.parametrize("name", HOSTILE + [None, 5, b"check_action", ("check_action",), ""])
def test_unknown_names_fold_into_other(stats, name):
    stats.record(name, name, reason=name, elapsed_ms=1)
    stats.record("check_action", "incomplete", reason=name)
    stats.record("check_action", "degraded", reason=name)
    snapshot = stats.snapshot()
    assert sorted(snapshot["tools"]) == ["check_action", "other"]
    assert snapshot["tools"]["other"]["calls"] == 1
    assert only(snapshot["tools"]["other"]["outcomes"]) == {"other": 1}
    assert snapshot["tools"]["check_action"]["reasons"] == {"incomplete": {"other": 1}, "degraded": {"other": 1}}


def test_hostile_strings_never_reach_the_snapshot(stats):
    for text in HOSTILE:
        stats.record(text, "incomplete", reason=text, elapsed_ms=2, gate_wait_ms=1)
        stats.record("check_action", text, reason=text)
        stats.record("check_action", "degraded", reason=text)
    snapshot = stats.snapshot()
    rendered = json.dumps(snapshot)
    for text in HOSTILE:
        assert text not in rendered and text.strip()[:12] not in rendered
    labels = set()
    for name, entry in snapshot["tools"].items():
        labels |= {name, *entry["outcomes"], *(r for group in entry["reasons"].values() for r in group)}
    assert labels <= set(TOOLS) | set(OUTCOMES) | set(REASONS) | {"other", "incomplete", "degraded"}


def test_the_snapshot_is_plain_json_data_of_a_fixed_shape(stats):
    stats.record("check_action", "degraded", reason="sync_deadline", elapsed_ms=5000, gate_wait_ms=3)
    stats.worker_abandoned()
    stats.gate_acquired(4)
    snapshot = stats.snapshot()
    assert sorted(snapshot) == ["buckets_ms", "gate", "since", "tools", "uptime_sec", "workers"]
    assert snapshot["buckets_ms"] == [50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 45000, "+inf"]
    assert snapshot["workers"] == {"abandoned_now": 1, "abandoned_total": 1, "late_finish_total": 0}
    assert snapshot["gate"] == {"slots": 4, "in_use": 1}
    entry = snapshot["tools"]["check_action"]
    assert sorted(entry) == ["calls", "elapsed_ms", "gate_wait_ms", "outcomes", "reasons"]
    assert sorted(entry["elapsed_ms"]) == ["counts", "max"]
    assert sorted(entry["outcomes"]) == sorted([*OUTCOMES, "other"])
    assert sorted(entry["reasons"]) == ["degraded", "incomplete"]
    assert snapshot["uptime_sec"] >= 0
    assert datetime.fromisoformat(snapshot["since"]).tzinfo is not None
    list(leaves(snapshot))
    assert json.loads(json.dumps(snapshot, allow_nan=False)) == snapshot


def test_a_fresh_snapshot_is_empty_and_is_a_copy(stats):
    snapshot = stats.snapshot()
    assert snapshot["tools"] == {}
    assert snapshot["workers"] == {"abandoned_now": 0, "abandoned_total": 0, "late_finish_total": 0}
    assert snapshot["gate"] == {"slots": 0, "in_use": 0}
    stats.record("check_action", "ok")
    assert snapshot["tools"] == {}                          # an earlier snapshot is not changed by later requests
    snapshot["tools"]["check_action"] = "tampered"
    assert stats.snapshot()["tools"]["check_action"]["calls"] == 1


def test_abandoned_workers_are_a_gauge_with_totals_and_never_negative(stats):
    stats.worker_abandoned()
    stats.worker_abandoned()
    stats.worker_finished_late()
    assert stats.snapshot()["workers"] == {"abandoned_now": 1, "abandoned_total": 2, "late_finish_total": 1}
    stats.worker_finished_late()
    stats.worker_finished_late()
    assert stats.snapshot()["workers"] == {"abandoned_now": 0, "abandoned_total": 2, "late_finish_total": 3}
    assert stats.snapshot()["tools"] == {}                  # the end of a worker is not a request


def test_the_gate_reports_its_limit_and_the_slots_in_use(stats):
    stats.gate_acquired(4)
    stats.gate_acquired(4)
    stats.gate_released()
    assert stats.snapshot()["gate"] == {"slots": 4, "in_use": 1}
    stats.gate_released()
    stats.gate_released()
    assert stats.snapshot()["gate"] == {"slots": 4, "in_use": 0}


def test_reset_starts_again(stats):
    stats.record("check_action", "ok")
    stats.worker_abandoned()
    stats.gate_acquired(2)
    before = stats.snapshot()["since"]
    stats.reset()
    snapshot = stats.snapshot()
    assert snapshot["tools"] == {} and snapshot["gate"] == {"slots": 0, "in_use": 0}
    assert snapshot["workers"]["abandoned_total"] == 0
    assert datetime.fromisoformat(snapshot["since"]) >= datetime.fromisoformat(before)


def test_concurrent_writers_and_readers_stay_consistent(stats):
    threads, rounds = 8, 400
    start, failures = threading.Barrier(threads + 1), []

    def writer(index):
        start.wait()
        for step in range(rounds):
            tool = TOOLS[(index + step) % len(TOOLS)]
            outcome = OUTCOMES[step % len(OUTCOMES)]
            stats.record(tool, outcome, reason=REASONS[step % len(REASONS)], elapsed_ms=step % 60000,
                         gate_wait_ms=None if step % 3 == 0 else step % 90)
            stats.gate_acquired(4)
            stats.worker_abandoned()
            stats.worker_finished_late()
            stats.gate_released()

    def reader():
        start.wait()
        for _ in range(rounds):
            snapshot = stats.snapshot()
            for entry in snapshot["tools"].values():
                if sum(entry["outcomes"].values()) != entry["calls"] or sum(entry["elapsed_ms"]["counts"]) != entry["calls"]:
                    failures.append(entry)
            assert snapshot["workers"]["abandoned_now"] >= 0 and snapshot["gate"]["in_use"] >= 0

    pool = [threading.Thread(target=writer, args=(index,)) for index in range(threads)] + [threading.Thread(target=reader)]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join(30)
        assert not thread.is_alive()
    assert not failures
    snapshot = stats.snapshot()
    assert sum(entry["calls"] for entry in snapshot["tools"].values()) == threads * rounds
    assert snapshot["workers"] == {"abandoned_now": 0, "abandoned_total": threads * rounds,
                                   "late_finish_total": threads * rounds}
    assert snapshot["gate"]["in_use"] == 0


def named(name):
    def method():
        pass
    method.__name__ = name
    return method


def test_a_method_is_labelled_by_its_name_when_that_name_is_a_tool():
    class Instance:
        def __call__(self):
            pass

    assert request_stats.tool_label(named("check_action")) == "check_action"
    assert request_stats.tool_label(named("not_a_tool")) == "other"
    assert request_stats.tool_label(named("sk-LIVE-SECRET-1234567890abcdef")) == "other"
    assert request_stats.tool_label(lambda: None) == "other"
    assert request_stats.tool_label(Instance()) == "other"                 # no __name__ at all
    assert request_stats.tool_label(named("hook")) == "hook"

    class Odd:
        __name__ = 5

    assert request_stats.tool_label(Odd()) == "other"                      # a name that is not even a string
    assert request_stats.fold(None, TOOLS) == "other" and request_stats.fold("hook", TOOLS) == "hook"


def test_a_note_reaches_only_the_request_collecting_it():
    request_stats.note_degraded("sync_deadline")                       # nobody is collecting: nothing to tell, no error
    notes = Notes()
    assert notes.degraded is None
    with request_stats.collecting(notes):
        request_stats.note_degraded("sync_deadline")
        request_stats.note_degraded("index_busy")                       # the first reason of a request is kept
    assert notes.degraded == "sync_deadline"
    request_stats.note_degraded("sync_deadline")                       # collection ended with the request
    other = Notes()
    with request_stats.collecting(other):
        request_stats.note_degraded("not a reason")
    assert other.degraded == "other" and notes.degraded == "sync_deadline"


def test_notes_follow_a_worker_thread_that_was_given_the_collector():
    notes = Notes()

    def worker():
        with request_stats.collecting(notes):
            request_stats.note_degraded("sync_deadline")

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(10)
    assert notes.degraded == "sync_deadline"
