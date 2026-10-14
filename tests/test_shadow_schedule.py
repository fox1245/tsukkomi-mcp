"""The adaptive schedule of shadow comparisons (``audit/shadow.py``) and how the engine applies it.

Everything is synthetic. The policy functions are tested directly; the engine tests drive a real engine over a growing
codex-shaped session and read ``incremental`` (the report in every audit payload) and ``audit_status``. The conftest sets
``SELF_DIRECT_AUDIT_SHADOW_EVERY=1`` for the suite; every engine here passes the setting it is testing explicitly.
"""
from __future__ import annotations

import json
import random
import threading
import time

import pytest

from self_directing_mcp.audit import incremental as inc
from self_directing_mcp.audit import shadow
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.request_control import check_request
from self_directing_mcp.shadow_worker import ShadowWorker, start_shadow_worker
from test_reliability import SID, call, message, write

HEAD = [{"type": "session_meta", "payload": {"id": SID}}]
RULES = [
    {"id": "no-deploy", "type": "must_not", "scope": "tool_call", "regex": "deploy", "roles": ["assistant"]},
    {"id": "no-wipe", "type": "must_not", "scope": "tool_call", "regex": "rm -rf /", "roles": ["assistant"]},
    {"id": "said-done", "type": "must", "scope": "message", "regex": "all done", "roles": ["assistant"]},
]
HIGH = {"id": "tests-first", "type": "must", "scope": "tool_call", "regex": "pytest", "before_regex": "deploy",
        "risk": "high"}


class Lab:
    """A codex session that grows by one record per request, and an engine auditing it with a chosen schedule."""

    def __init__(self, tmp_path, *, every="adaptive", rules=RULES, defer=False, idle_sec=0.0, name="a"):
        self.tmp = tmp_path
        self.root = tmp_path / ("sessions-" + name)
        self.root.mkdir(exist_ok=True)
        self.path = self.root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl"
        write(self.path, HEAD + [message("hello", "user"), message("all done")], "w")
        self.index = tmp_path / ("index-" + name)
        self.every, self.defer, self.idle_sec = every, defer, idle_sec
        self.engine = self.open()
        self.engine.upsert_contracts(rules)
        self.serial = 0

    def open(self):
        settings = Settings(_env_file=None, local_only=True, openrouter_api_key_file=None, codex_sessions_dir=self.root,
                            index_dir=self.index, audit_shadow_every=self.every, audit_incremental_min_events=0,
                            audit_shadow_idle_sec=self.idle_sec)
        return SelfDirectEngine(settings, defer_derived=self.defer)

    def append(self, *records):
        self.serial += 1
        write(self.path, list(records) or [call(f"echo step {self.serial}", f"c{self.serial}")], "a")

    def audit(self, *records):
        self.append(*records)
        return self.engine.audit_session(SID, provider="codex")

    def check(self, command, *records):
        self.append(*records)
        return self.engine.check_action(SID, {"tool_name": "shell", "arguments": {"command": command}}, provider="codex")

    def report(self, payload):
        return payload["incremental"]

    def status(self):
        return self.engine.audit_status(SID, provider="codex")["incremental_verification"]

    def reopen(self):
        self.engine.close()
        self.engine = self.open()
        return self.engine

    def close(self):
        self.engine.close()


@pytest.fixture
def lab(tmp_path):
    made = []

    def make(**kwargs):
        made.append(Lab(tmp_path, **kwargs))
        return made[-1]
    yield make
    for item in made:
        item.close()


@pytest.fixture
def draw(monkeypatch):
    """Control the sampled draw: ``draw.value = True/False`` for every request, or a callable ``(offset) -> bool``."""
    class Draw:
        value = False

    state = Draw()
    monkeypatch.setattr(shadow, "sampled", lambda *a, **k: state.value(a[3]) if callable(state.value) else state.value)
    return state


# --- the policy functions ------------------------------------------------------------------------------------------

def fresh(**premise):
    state, reason = shadow.reconcile(None, {"contracts": "c", "parser": 3, "evaluator": 8, "generation": 1, **premise})
    assert reason == "new"
    return state


def match(state, *, full=True, key="audit", offset=1):
    return shadow.settle(state, outcome="match", full_scope=full, key=key, flagged=set(), kept=set(), at="t",
                         offset=offset)


def test_a_new_session_starts_at_one_in_four_with_the_warmup_owed():
    state = fresh()
    assert shadow.RATE_DENOMINATORS[state.rate_index] == 4 and state.warmup == shadow.WARMUP_REQUESTS == 8
    assert shadow.rate_text(state, None) == "1/4"


def test_the_rate_halves_after_every_32_matching_comparisons_down_to_the_floor():
    state, seen = fresh(), []
    for number in range(1, 32 * 6 + 1):
        state = match(state)
        if number in (31, 32, 63, 64, 95, 96, 127, 128, 159, 160, 192):
            seen.append((number, shadow.rate_text(state, None)))
    assert seen == [(31, "1/4"), (32, "1/8"), (63, "1/8"), (64, "1/16"), (95, "1/16"), (96, "1/32"), (127, "1/32"),
                    (128, "1/64"), (159, "1/64"), (160, "1/64"), (192, "1/64")]       # the floor stays the floor
    assert shadow.RATE_DENOMINATORS == (4, 8, 16, 32, 64)


@pytest.mark.parametrize("component,value", [("contracts", "other"), ("parser", 4), ("evaluator", 9), ("generation", 2)])
def test_each_change_of_premise_resets_the_rate_and_the_warmup(component, value):
    state = fresh()
    for _ in range(40):
        state = match(state)
    assert state.rate_index == 1 and state.warmup == 0
    same, reason = shadow.reconcile(state, dict(state.premise))
    assert reason is None and same is state                       # nothing changed: nothing resets (an append never does)
    reset, reason = shadow.reconcile(state, {**state.premise, component: value})
    assert reason == component and reset.rate_index == 0 and reset.streak == 0
    assert reset.warmup == shadow.WARMUP_REQUESTS and reset.epoch == state.epoch + 1
    assert state.rate_index == 1                                  # the stored object was not touched


def test_a_mismatch_resets_the_rate_and_keeps_the_flags():
    state = fresh()
    for _ in range(40):
        state = match(state)
    state = shadow.settle(state, outcome="match", full_scope=True, key="audit", flagged={"a"}, kept=set(), at="t", offset=1)
    after = shadow.settle(state, outcome="mismatch", full_scope=True, key="audit", flagged={"a"}, kept=set(), at="t", offset=2)
    assert after.rate_index == 0 and after.streak == 0 and after.warmup == shadow.WARMUP_REQUESTS
    assert after.epoch == state.epoch + 1 and after.reset_reason == "mismatch" and after.flags == {"audit": ["a"]}
    assert after.last_match_offset is None


def test_a_comparison_that_did_not_run_or_covered_one_contract_is_no_match():
    state = fresh()
    for _ in range(10):
        state = shadow.settle(state, outcome=None, full_scope=False, key="audit", flagged=set(), kept=set(), at="t", offset=1)
        state = match(state, full=False)          # a scoped comparison matched, but it did not look at everything
    assert (state.streak, state.warmup, state.compared, state.rate_index) == (0, shadow.WARMUP_REQUESTS, 0, 0)
    state = match(state)
    assert (state.streak, state.warmup, state.compared) == (1, shadow.WARMUP_REQUESTS - 1, 1)


def test_the_draw_is_deterministic_and_follows_the_rate():
    state = fresh()
    first = [shadow.sampled(state, "s", "codex", offset, None) for offset in range(4000)]
    assert first == [shadow.sampled(state, "s", "codex", offset, None) for offset in range(4000)]
    assert 0.2 < sum(first) / len(first) < 0.3                     # 1/4
    for index, denominator in enumerate(shadow.RATE_DENOMINATORS):
        state.rate_index = index
        hits = sum(shadow.sampled(state, "s", "codex", offset, None) for offset in range(20_000))
        assert abs(hits / 20_000 - 1 / denominator) < 0.5 / denominator + 0.003, denominator
    other = fresh()
    other.compared = 1                                             # the draw depends on the state, not just the cursor
    assert [shadow.sampled(other, "s", "codex", o, None) for o in range(200)] != first[:200]


def test_the_fixed_settings_keep_their_meaning():
    state = fresh()
    assert shadow.parse_every("adaptive") is None and shadow.parse_every(" ADAPTIVE ") is None
    assert shadow.parse_every(0) == 0 and shadow.parse_every("16") == 16 and shadow.parse_every(1) == 1
    assert all(shadow.sampled(state, "s", "codex", offset, 1) for offset in range(50))
    assert not any(shadow.sampled(state, "s", "codex", offset, 0) for offset in range(500))
    share = sum(shadow.sampled(state, "s", "codex", offset, 16) for offset in range(8000)) / 8000
    assert 0.04 < share < 0.09                                      # about 1 in 16
    assert [shadow.rate_text(state, value) for value in (0, 1, 16)] == ["never", "1/1", "1/16"]


def test_flags_remember_the_previous_request_of_the_same_kind_only():
    state = fresh()
    state = shadow.settle(state, outcome=None, full_scope=False, key="check:a", flagged={"x"}, kept=set(), at="t", offset=1)
    assert shadow.flipped(state, "check:a", {"x": "clean"}) == {"x"}
    assert shadow.flipped(state, "check:b", {"x": "clean"}) == set()          # another action
    assert shadow.flipped(state, "check:a", {"x": "violation"}) == set()
    state = shadow.settle(state, outcome=None, full_scope=False, key="check:a", flagged=set(), kept=set(), at="t", offset=2)
    assert state.flags == {}
    for number in range(40):
        state = shadow.settle(state, outcome=None, full_scope=False, key=f"k{number}", flagged={"x"}, kept=set(), at="t", offset=3)
    assert len(state.flags) == shadow.FLAG_REQUESTS


def test_a_state_round_trips_and_an_unreadable_one_is_refused():
    state = fresh()
    for _ in range(40):
        state = match(state)
    again = shadow.ShadowState.load(state.epoch, json.loads(json.dumps(state.dump())))
    assert again == state
    for broken in ({"premise": {}, "rate_index": 9, "streak": 0, "warmup": 0, "compared": 0},
                   {"premise": {}, "rate_index": 0, "streak": -1, "warmup": 0, "compared": 0}, {}):
        with pytest.raises((ValueError, KeyError)):
            shadow.ShadowState.load(1, broken)


# --- the schedule in an engine ---------------------------------------------------------------------------------------

def test_the_first_requests_are_compared_then_the_draw_decides(lab, draw):
    box = lab(every="adaptive")
    shown = []
    for _ in range(12):
        report = box.report(box.audit())
        shown.append((bool(report.get("shadow_compared")), report.get("shadow_reasons")))
    assert [compared for compared, _ in shown] == [True] * 8 + [False] * 4
    assert "rebuild" in shown[0][1] and "premise" in shown[0][1] and all("warmup" in why for _, why in shown[:8])
    assert box.status()["session"]["forced_comparisons_left"] == 0


def test_the_rate_halves_in_an_engine_resets_on_each_premise_and_not_on_an_append(lab, draw, monkeypatch):
    draw.value = True
    box = lab(every="adaptive")
    for _ in range(31):
        box.audit()
    assert box.status()["session"]["rate"] == "1/4"
    box.audit()
    assert box.status()["session"]["rate"] == "1/8" and box.status()["session"]["streak"] == 0
    draw.value = False
    for _ in range(5):                                      # plain appends, uncompared: nothing resets
        report = box.report(box.audit())
        assert "shadow_reset" not in report
    assert box.status()["session"]["rate"] == "1/8"
    # a new contract
    box.engine.upsert_contracts([{"id": "added", "type": "must_not", "scope": "tool_call", "regex": "zzz", "roles": ["assistant"]}])
    report = box.report(box.audit())
    assert report["shadow_reset"] == "contracts" and box.status()["session"]["rate"] == "1/4"
    assert "premise" in report["shadow_reasons"] and box.status()["session"]["forced_comparisons_left"] == 7
    # a changed revision of one contract
    draw.value = True
    for _ in range(32):
        box.audit()
    assert box.status()["session"]["rate"] == "1/8"
    box.engine.upsert_contracts([{"id": "added", "type": "must_not", "scope": "tool_call", "regex": "zzz", "roles": ["assistant"], "severity": "high"}])
    assert box.report(box.audit())["shadow_reset"] == "contracts"
    # a new evaluator version
    draw.value = True
    for _ in range(32):
        box.audit()
    assert box.status()["session"]["rate"] == "1/8"
    monkeypatch.setattr(inc, "EVALUATOR_VERSION", inc.EVALUATOR_VERSION + 1)
    assert box.report(box.audit())["shadow_reset"] == "evaluator" and box.status()["session"]["rate"] == "1/4"


def test_a_new_parser_or_generation_resets_the_rate(lab, draw, monkeypatch):
    from self_directing_mcp.codex import parse
    draw.value = True
    box = lab(every="adaptive")
    for _ in range(32):
        box.audit()
    assert box.status()["session"]["rate"] == "1/8"
    monkeypatch.setattr(parse, "PARSER_VERSION", parse.PARSER_VERSION + 1)
    report = box.report(box.audit())
    assert report["shadow_reset"] in ("parser", "generation") and box.status()["session"]["rate"] == "1/4"
    for _ in range(32):
        box.audit()
    assert box.status()["session"]["rate"] == "1/8"
    # an edit of earlier bytes is a rewrite: the session gets a new generation
    lines = box.path.read_text(encoding="utf-8").splitlines()
    lines[1] = json.dumps(message("hello, edited", "user"))
    box.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = box.report(box.audit())
    assert report["shadow_reset"] == "generation" and box.status()["session"]["rate"] == "1/4"


def test_a_comparison_skipped_for_lack_of_time_is_no_match(lab, draw, monkeypatch):
    draw.value = True
    box = lab(every="adaptive")
    for _ in range(3):
        box.audit()
    before = box.status()["session"]
    assert before["streak"] == 3
    real = inc._shadow_affordable
    monkeypatch.setattr(inc, "_shadow_affordable", lambda events: False)
    for _ in range(5):
        report = box.report(box.audit())
        assert report["shadow"] is False and report["shadow_skipped"] == "deadline"
    after = box.status()["session"]
    assert (after["streak"], after["comparisons"], after["forced_comparisons_left"]) == (
        before["streak"], before["comparisons"], before["forced_comparisons_left"])
    monkeypatch.setattr(inc, "_shadow_affordable", real)
    box.audit()
    assert box.status()["session"]["streak"] == 4


@pytest.mark.parametrize("defer", [False, True])
def test_the_schedule_survives_a_restart(lab, draw, defer):
    draw.value = True
    box = lab(every="adaptive", defer=defer)
    for _ in range(35):
        box.audit()
    expected = box.status()["session"]
    assert expected["rate"] == "1/8" and expected["streak"] == 3
    box.reopen()                                        # close writes what a deferring engine held
    assert box.status()["session"] == expected
    box.audit()
    assert box.status()["session"]["streak"] == 4 and box.status()["session"]["rate"] == "1/8"


def test_an_older_epoch_never_overwrites_a_newer_one(lab, draw):
    draw.value = True
    box = lab(every="adaptive")
    for _ in range(3):
        box.audit()
    store = box.engine.store
    newer = store.load_shadow_state(SID, "codex")
    older = newer.copy()
    older.epoch, older.rate_index = newer.epoch - 1, 3
    store.save_shadow_state(SID, "codex", older)
    assert store.load_shadow_state(SID, "codex").rate_index == newer.rate_index


def test_every_set_to_an_integer_pins_the_legacy_rate(lab, draw):
    draw.value = False
    box = lab(every=16)
    for _ in range(40):
        report = box.report(box.audit())
        assert "shadow_rate" not in report
    session = box.status()
    assert session["policy"].startswith("fixed") and session["session"]["rate"] == "1/16"
    never = lab(every=0, name="b")
    first = never.report(never.audit())
    assert first["shadow_compared"] and "rebuild" in first["shadow_reasons"]    # forced comparisons still run
    every = lab(every=1, name="c")
    for _ in range(4):
        assert every.report(every.audit())["shadow_compared"]
    assert every.engine.store.load_shadow_state(SID, "codex") is None            # comparing everything keeps no schedule


# --- the forced comparisons -------------------------------------------------------------------------------------------

def settle_in(box, count=8, checks=False):
    """Use up the warm-up comparisons. With ``checks`` the action-check variant is built and warmed up as well (it keeps
    monitor states of its own, so its first request is a rebuild, which owes the warm-up again)."""
    for _ in range(count):
        box.audit()
    if checks:
        for _ in range(count + 1):
            box.check("ls")
    assert box.status()["session"]["forced_comparisons_left"] == 0


def test_a_rebuild_forces_a_comparison_and_the_warmup_again(lab, draw):
    box = lab(every="adaptive")
    settle_in(box)
    assert "shadow_compared" not in box.report(box.audit())
    box.engine.flush_derived()
    with box.engine.store._conn:
        box.engine.store._conn.execute("DELETE FROM audit_monitor_state")
    report = box.report(box.audit())
    assert report["shadow_compared"] and "rebuild" in report["shadow_reasons"]
    assert box.status()["session"]["forced_comparisons_left"] == 7
    assert box.report(box.audit())["shadow_reasons"] == ["warmup"]


def test_a_violation_forces_a_comparison_of_that_contract_only(lab, draw):
    box = lab(every="adaptive")
    settle_in(box)
    report = box.report(box.audit(call("deploy now", "bad")))
    assert report["shadow_compared"] and report["shadow_reasons"] == ["violation"]
    assert report["shadow_scope"] == ["no-deploy"]                  # the other contracts were not re-evaluated in full
    payload = box.engine.audit_session(SID, provider="codex")
    assert payload["verdict"] == "violation" and payload["incremental"]["shadow_skipped"] == "verified_before"


def test_a_clean_answer_without_a_flag_is_not_compared(lab, draw):
    box = lab(every="adaptive")
    settle_in(box)
    for _ in range(5):
        report = box.report(box.audit())
        assert report["shadow"] is False and "shadow_reasons" not in report


@pytest.mark.allow_incremental_mismatch
def test_a_violation_that_turns_clean_is_compared_at_once(lab, draw):
    box = lab(every="adaptive")
    settle_in(box)
    assert box.audit(call("deploy now", "bad"))["verdict"] == "violation"
    box.engine.flush_derived()
    connection = box.engine.store._conn
    with connection:
        for row, state in connection.execute("SELECT rowid, state FROM audit_monitor_state WHERE monitor_id='no-deploy'").fetchall():
            data = json.loads(state)
            data.update(hits=0, hit_ev=[], hit_n=0)
            connection.execute("UPDATE audit_monitor_state SET state=? WHERE rowid=?", (json.dumps(data), row))
    box.engine.close()
    box.engine = box.open()
    payload = box.audit()
    report = payload["incremental"]
    assert report["incremental_mismatch"] == ["no-deploy"]          # found by the flip rule, with the draw off
    finding = next(f for f in payload["findings"] if f["contract_id"] == "no-deploy")
    assert finding["verdict"] == "violation"                       # the full evaluator's answer was returned
    assert box.status()["session"]["rate"] == "1/4" and box.status()["needs_review"] is not None


def test_a_high_risk_contract_is_compared_only_when_it_applies_or_changed(lab, draw):
    box = lab(every="adaptive", rules=[*RULES, HIGH])
    settle_in(box, checks=True)
    # Nothing touches the contract: neither an unrelated audit nor an unrelated action check compares anything.
    assert box.report(box.audit())["shadow"] is False
    unrelated = box.report(box.check("ls -la"))
    assert unrelated["shadow"] is False and "shadow_reasons" not in unrelated
    # An appended record that changes its summary compares it (only it); an unrelated one does not.
    changed = box.report(box.audit(call("pytest -q", "t1")))
    assert changed["shadow_reasons"] == ["high_risk"] and changed["shadow_scope"] == ["tests-first"]
    assert box.report(box.audit())["shadow"] is False
    # An action the contract applies to compares that contract, and only that one.
    applies = box.report(box.check("deploy now"))
    assert applies["shadow_reasons"] == ["high_risk"] and applies["shadow_scope"] == ["tests-first"]
    # The same contract at normal risk: neither the record nor the action compares anything.
    plain = lab(every="adaptive", rules=[*RULES, {**HIGH, "risk": "normal"}], name="b")
    settle_in(plain, checks=True)
    assert plain.report(plain.audit(call("pytest -q", "t1")))["shadow"] is False
    assert plain.report(plain.check("deploy now"))["shadow"] is False


def test_a_high_risk_contract_in_the_session_does_not_slow_unrelated_requests(lab, draw):
    box = lab(every="adaptive", rules=[*RULES, HIGH])
    settle_in(box, checks=True)
    reports = [box.report(box.check("echo fine")) for _ in range(10)]
    assert not any(report["shadow"] for report in reports)


@pytest.mark.allow_incremental_mismatch
def test_a_scoped_mismatch_returns_the_full_result_and_disables_the_contract(lab, draw, monkeypatch):
    box = lab(every="adaptive", rules=[*RULES, HIGH])
    settle_in(box, checks=True)
    real = inc.RuleMonitor.finding

    def broken(self, proposed, facts):
        item = real(self, proposed, facts)
        if self.rule.id == "tests-first" and item.verdict == "violation":
            return item.model_copy(update={"verdict": "clean", "reason": "injected", "evidence": []})
        return item
    monkeypatch.setattr(inc.RuleMonitor, "finding", broken)
    payload = box.check("deploy now")
    assert payload["incremental"]["incremental_mismatch"] == ["tests-first"]
    assert next(f for f in payload["findings"] if f["contract_id"] == "tests-first")["verdict"] == "violation"
    again = box.check("deploy now")
    assert again["incremental"]["fallback"]["tests-first"] == "health:incremental_mismatch"
    status = box.status()
    assert [row["contract_id"] for row in status["disabled_contracts"]] == ["tests-first"]
    assert status["session"]["rate"] == "1/4" and status["session"]["reset_reason"] == "mismatch"


# --- the status report -------------------------------------------------------------------------------------------------

def test_audit_status_reports_the_policy_and_never_calls_it_a_guarantee(lab, draw):
    box = lab(every="adaptive")
    box.audit()
    status = box.engine.audit_status()["incremental_verification"]
    assert status["policy"] == "adaptive" and status["idle_comparison"] is None and "session" not in status
    assert "not a proof" in status["guarantee"]
    session = box.status()["session"]
    assert session["rate"] == "1/4" and session["comparisons"] == 1 and session["last_match_offset"] is not None


# --- differential: the adaptive schedule never changes an answer --------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_adaptive_answers_equal_the_full_evaluator_on_random_sessions(tmp_path, seed):
    rng = random.Random(f"adaptive/{seed}")
    box = Lab(tmp_path, every="adaptive", rules=[*RULES, HIGH])
    twin = SelfDirectEngine(Settings(_env_file=None, local_only=True, openrouter_api_key_file=None,
                                     codex_sessions_dir=box.root, index_dir=tmp_path / "twin", audit_incremental=False))
    twin.upsert_contracts([*RULES, HIGH])
    try:
        for step in range(70):
            words = rng.choice(["deploy now", "pytest -q", "echo x", "rm -rf /tmp/x", "ls", "all done"])
            record = message(words) if words == "all done" else call(words, f"r{step}")
            box.append(record)
            if rng.random() < 0.3:
                action = {"tool_name": "shell", "arguments": {"command": rng.choice(["deploy now", "ls", "pytest"])}}
                fast, full = (e.check_action(SID, action, provider="codex") for e in (box.engine, twin))
            else:
                fast, full = (e.audit_session(SID, provider="codex") for e in (box.engine, twin))
            assert (fast["verdict"], [(f["contract_id"], f["verdict"], f["reason"]) for f in fast["findings"]]) == (
                full["verdict"], [(f["contract_id"], f["verdict"], f["reason"]) for f in full["findings"]]), step
    finally:
        twin.close()
        box.close()


# --- the idle comparison --------------------------------------------------------------------------------------------------

def worker_for(box, **kwargs):
    worker = ShadowWorker(box.engine, idle_sec=0.0, poll_sec=0.01, **kwargs)
    box.engine.attach_shadow_worker(worker)
    return worker


def test_hook_processes_and_a_disabled_setting_never_start_the_idle_comparison(lab, tmp_path, monkeypatch):
    from self_directing_mcp import agy_hooks, claude_hooks
    monkeypatch.setenv("HOME", str(tmp_path / "home"))           # nothing of a real user's is read or created
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    for module in (claude_hooks, agy_hooks):
        engine = module._new_engine()
        try:
            assert engine._shadow_worker is None
        finally:
            engine.close()
    box = lab(every="adaptive", idle_sec=0.0)
    assert start_shadow_worker(box.engine) is None and box.engine._shadow_worker is None
    resident = lab(every="adaptive", idle_sec=2.0, name="b")
    worker = start_shadow_worker(resident.engine)
    try:
        assert worker is not None and resident.engine._shadow_worker is worker and worker.idle_sec == 2.0
    finally:
        worker.stop()


def test_an_idle_engine_compares_the_latest_state_of_a_session_that_skipped_comparisons(lab, draw):
    box = lab(every="adaptive")
    worker = worker_for(box)
    settle_in(box)
    for _ in range(6):
        assert box.report(box.audit())["shadow"] is False
    assert worker.snapshot()["pending_sessions"] == 1 and worker.snapshot()["unverified_requests"] == 6
    before = box.status()["session"]["comparisons"]
    assert worker.tick() == 1
    stats = worker.snapshot()
    assert stats["matched"] == 1 and stats["pending_sessions"] == 0 and stats["unverified_requests"] == 0
    assert box.status()["session"]["comparisons"] == before + 1
    assert worker.tick() == 0                                       # nothing left to verify


def test_the_idle_comparison_waits_for_the_engine_to_be_idle(lab, draw):
    box = lab(every="adaptive")
    worker = ShadowWorker(box.engine, idle_sec=3600.0)
    box.engine.attach_shadow_worker(worker)
    settle_in(box)
    box.audit()
    assert worker.snapshot()["pending_sessions"] == 1 and worker.tick() == 0


def test_a_comparison_that_could_not_run_leaves_the_session_pending_and_is_retried_later(lab, draw, monkeypatch):
    box = lab(every="adaptive")
    now = [1000.0]
    worker = ShadowWorker(box.engine, idle_sec=0.0, clock=lambda: now[0])       # a clock the test moves
    box.engine.attach_shadow_worker(worker)
    settle_in(box)
    box.audit()
    real = inc._shadow_affordable
    monkeypatch.setattr(inc, "_shadow_affordable", lambda events: False)       # no time for a full evaluation
    assert worker.tick() == 1 and worker.snapshot()["not_compared"] == 1
    assert worker.snapshot()["pending_sessions"] == 1
    assert worker.tick() == 0                                                  # not before the retry time
    monkeypatch.setattr(inc, "_shadow_affordable", real)
    now[0] += 29
    assert worker.tick() == 0
    now[0] += 2
    assert worker.tick() == 1 and worker.snapshot()["matched"] == 1 and worker.snapshot()["pending_sessions"] == 0


def test_one_pending_job_per_session_and_one_running_at_a_time(lab, draw, monkeypatch):
    first = lab(every="adaptive", rules=[*RULES, HIGH])
    worker = worker_for(first)
    settle_in(first, checks=True)
    for _ in range(5):
        first.audit()
        first.check("echo hi")
    assert worker.snapshot()["pending_sessions"] == 1               # five audits and five checks, still one session
    for name in ("x", "y"):                                          # two more sessions
        worker.note("codex", "session-" + name, None, "audit", None, 1, "unverified")
    assert worker.snapshot()["pending_sessions"] == 3
    running, peak, jobs = [], [0], []
    real = first.engine.verify_incremental

    def watched(session_id, **kwargs):
        running.append(1)
        peak[0] = max(peak[0], len(running))
        jobs.append((session_id, kwargs.get("variant")))
        time.sleep(0.02)
        try:
            if session_id == SID:
                return real(session_id, **kwargs)
            worker.note("codex", session_id, None, kwargs.get("variant"), None, 1, "match")    # what a real match does
            return {"ok": True, "compared": True, "mismatch": [], "skipped": None}
        finally:
            running.pop()
    monkeypatch.setattr(first.engine, "verify_incremental", watched)
    worker.start()
    try:
        deadline = time.monotonic() + 20
        while worker.snapshot()["pending_sessions"] and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        worker.stop()
    assert peak[0] == 1 and worker.snapshot()["pending_sessions"] == 0
    assert sorted(session for session, _ in jobs if session != SID) == ["session-x", "session-y"]     # once each
    assert sorted(v for session, v in jobs if session == SID) == ["audit", "check"]


def test_the_idle_comparison_never_blocks_a_request(lab, draw, monkeypatch):
    """A request that arrives while the comparison runs is answered at once, and the comparison stops."""
    box = lab(every="adaptive")
    worker = worker_for(box, budget_sec=30.0)
    settle_in(box)
    box.audit()
    started, release = threading.Event(), threading.Event()
    from self_directing_mcp import engine as engine_module
    real = engine_module.run_audit
    seen = {"worker": threading.current_thread()}

    def slow(**kwargs):
        if threading.current_thread() is not threading.main_thread() and kwargs.get("plan") is None:
            started.set()                                     # the full evaluation of the idle comparison
            while not release.is_set():
                check_request()                                # cooperative, like every evaluation
                time.sleep(0.005)
        return real(**kwargs)
    monkeypatch.setattr(engine_module, "run_audit", slow)
    outcome = {}
    thread = threading.Thread(target=lambda: outcome.update(ran=worker.tick()), name="idle-job")
    worker._job_thread = thread                               # the thread the test runs the job on is the worker's
    thread.start()
    assert started.wait(10)
    began = time.monotonic()
    answer = box.audit()                                      # a request from the main thread, while the job runs
    elapsed = time.monotonic() - began
    thread.join(10)
    release.set()
    assert not thread.is_alive() and answer["incremental"]["mode"] == "delta"
    assert elapsed < 5 and worker.snapshot()["preempted"] == 1 and worker.snapshot()["matched"] == 0
    assert worker.snapshot()["pending_sessions"] == 1          # the session stays pending for the next quiet moment


@pytest.mark.allow_incremental_mismatch
def test_a_mismatch_found_while_idle_disables_the_contract_resets_the_rate_and_records_a_range_for_review(lab, draw, monkeypatch):
    box = lab(every="adaptive")
    worker = worker_for(box)
    settle_in(box)
    for _ in range(4):
        box.audit()
    last_match = box.status()["session"]["last_match_offset"]
    # a defect enters the incremental path: a forbidden call is ignored, so the answers stay clean
    real = inc.RuleMonitor._forbidden

    def blind(self, event):
        if self.rule.id != "no-deploy":
            real(self, event)
    monkeypatch.setattr(inc.RuleMonitor, "_forbidden", blind)
    for number in range(3):
        payload = box.audit(call("deploy now", f"bad{number}"))
        assert next(f for f in payload["findings"] if f["contract_id"] == "no-deploy")["verdict"] == "clean"   # wrong, unnoticed
    assert worker.snapshot()["unverified_requests"] == 7
    assert worker.tick() == 1
    assert worker.snapshot()["mismatched"] == 1
    status = box.status()
    assert [row["contract_id"] for row in status["disabled_contracts"]] == ["no-deploy"]
    assert status["session"]["rate"] == "1/4" and status["session"]["reset_reason"] == "mismatch"
    (review,) = status["needs_review"]
    assert review["source"] == "idle" and review["requests"] == 7 and review["contracts"] == ["no-deploy"]
    assert review["since_offset"] == last_match and review["until_offset"] > review["since_offset"]
    assert worker.snapshot()["pending_sessions"] == 0
    # From now on the contract is judged by the full evaluator: the next answer is right.
    payload = box.audit()
    assert next(f for f in payload["findings"] if f["contract_id"] == "no-deploy")["verdict"] == "violation"


@pytest.mark.allow_incremental_mismatch
def test_a_sampled_mismatch_also_records_the_requests_it_leaves_in_doubt(lab, draw, monkeypatch):
    box = lab(every="adaptive")
    worker = worker_for(box)
    settle_in(box)
    real = inc.RuleMonitor._forbidden
    monkeypatch.setattr(inc.RuleMonitor, "_forbidden",
                        lambda self, event: None if self.rule.id == "no-deploy" else real(self, event))
    box.audit(call("deploy now", "bad"))
    box.audit()
    draw.value = True
    payload = box.audit()
    assert payload["incremental"]["incremental_mismatch"] == ["no-deploy"]
    (review,) = box.status()["needs_review"]
    assert review["source"] == "sampled" and review["requests"] == 2          # the two unverified answers before it


def test_a_deferring_engine_holds_the_schedule_until_the_next_commit(lab, draw):
    draw.value = True
    box = lab(every="adaptive", defer=True)
    box.audit()
    assert box.engine.store.derived.shadow_for("codex", SID) is not None
    with box.engine.store._conn:
        stored = box.engine.store._conn.execute("SELECT COUNT(*) FROM audit_shadow_state").fetchone()[0]
    assert stored == 0                                                         # held, not written
    box.engine.flush_derived()
    assert box.engine.store._conn.execute("SELECT COUNT(*) FROM audit_shadow_state").fetchone()[0] == 1
