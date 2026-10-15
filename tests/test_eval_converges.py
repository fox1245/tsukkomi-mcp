"""An audit of a big session converges: the evaluation keeps its progress, not only the sync (roundtable 8, PR4).

PR2 made a sync that cannot finish keep what it parsed. The judging that follows was still all or nothing under the same
deadline: an evaluation the deadline cut saved nothing, so the next call repeated it, and a heavy contract set made the audit
stop converging although the sync did. These tests run the real engine, index and monitors on a small synthetic session in
VIRTUAL time (``convergence_support.Virtual``): parsing a record, folding an event or running the full evaluator is charged a
price, and the request deadline, the soft deadline of a slice and the room left for a comparison all read that clock. Same calls,
same stops, same answers on every machine. The prices are the figures measured on the real engine times about 50, and the session
holds 3,000 events, so it behaves like one of about 150,000.

* Cold: nothing is indexed. Repeated calls reach a complete answer, each inside its budget, equal to the unbounded one-shot answer,
  in three regimes: judging is cheaper than syncing (LIGHT), about as expensive (HEAVY: an 18-contract set), and more than a prior
  for four contracts predicts (XHEAVY).
* A comparison never takes the answer with it: it is skipped when the speed of the pass that just ran says it cannot finish, and cut
  when it does not finish after all; a resident engine's idle comparison verifies what was skipped.
* Synced but never evaluated: one call builds the states when it can; when it cannot, the limit is stated and a resident engine's
  idle comparison builds them.
"""
from __future__ import annotations

import pytest

from convergence_support import CONTRACTS, DEPLOY, SID, Session, Virtual, brief
from self_directing_mcp import eval_cost
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.eval_cost import COSTS
from self_directing_mcp.index.ingest import SlicePolicy
from self_directing_mcp.shadow_worker import ShadowWorker

EVENTS = 3000
BUDGET = 8.0            # the hook budget, in virtual seconds
LIGHT = {}                                       # folding 10 microseconds an event (real figures), the full evaluator 11
HEAVY = dict(fold=2.3e-3, full=3.3e-3)           # 46 and 66 microseconds: eighteen heavy contracts, the regime of the report
XHEAVY = dict(fold=4.5e-3, full=6.5e-3)          # about 90 microseconds: more than the prior for four contracts expects


@pytest.fixture
def session(tmp_path):
    made = Session(tmp_path, EVENTS)
    try:
        yield made
    finally:
        made.close()


def policy(virtual):
    return SlicePolicy(mode="auto", clock=virtual.read, min_records=16, min_bytes=0)


def one_shot(session, action=None, name="one-shot", contracts=None):
    """The unbounded answer: a sync and an audit with no deadline, nothing sliced, nothing cut. Measured by a model of its own."""
    engine = session.engine(name, contracts=contracts)
    engine.sync_session(session_id=SID, provider="codex", path=str(session.path), embed=False, derived=False)
    kwargs = dict(provider="codex", path=str(session.path))
    answer = engine.audit_session(SID, **kwargs) if action is None else engine.check_action(SID, action, **kwargs)
    COSTS.reset()                    # the process of the real engine has not measured anything yet
    return answer


def stops(calls):
    return [call for call in calls if call.stopped]


def lag(engine):
    """``audit_status.evaluation.session``: how many events the monitor states do not cover, and how far they reach."""
    return engine.audit_status(SID, provider="codex")["evaluation"]["session"]


def assert_converged(calls, expected, budget=BUDGET):
    detail = [(round(c.elapsed, 2), c.stopped) for c in calls]
    assert calls[-1].complete, detail
    assert not stops(calls), detail                                                     # no call lost its answer
    assert all(call.elapsed <= budget for call in calls), detail                        # every call inside its budget
    assert brief(calls[-1].payload) == brief(expected)                                  # the one-shot answer, finding for finding


# --- cold --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rules", [CONTRACTS[:2], CONTRACTS], ids=["two-regexes", "five-contracts"])
@pytest.mark.parametrize("prices", [LIGHT, HEAVY, XHEAVY], ids=["light", "heavy", "xheavy"])
def test_a_cold_session_converges_to_the_one_shot_answer(session, monkeypatch, prices, rules):
    virtual = Virtual(monkeypatch, **prices)
    expected = one_shot(session, contracts=rules)
    engine = session.engine("cold", policy=policy(virtual), contracts=rules)
    calls, saved = [], []
    while len(calls) < 30:
        calls.append(virtual.audit(engine, session, budget=BUDGET))
        saved.append(lag(engine)["states_through_byte"])
        if calls[-1].complete:
            break
    assert_converged(calls, expected)
    assert 2 <= len(calls) <= 4                                                         # K, the bound this test reports
    offsets = [call.payload["coverage"]["byte_offset"] for call in calls]
    assert offsets == sorted(set(offsets))                                              # the index moves further with every call
    assert saved == sorted(set(saved)) and saved[-1] == expected["coverage"]["byte_offset"]   # and so do the saved states
    for call in calls[:-1]:                                                             # an unfinished call says what sized its slice
        sync = call.payload["sync"]
        assert sync["state"] == "partial" and sync["cost"]["limited_by"] in ("share", "cost")
        assert sync["cost"]["eval_us"] > 0 and sync["cost"]["slice_us"] > 0
    assert "sync" not in calls[-1].payload                                              # a finished sync adds nothing to coverage


def test_the_check_of_a_gated_action_converges_too(session, monkeypatch):
    virtual = Virtual(monkeypatch, **HEAVY)
    expected = one_shot(session, DEPLOY)
    engine = session.engine("cold", policy=policy(virtual))
    calls = virtual.converge(engine, session, budget=BUDGET, action=DEPLOY)
    assert_converged(calls, expected)
    assert all(call.payload["sync"]["cost"]["eval_us"] > 0 for call in calls[:-1])      # the order rule needs history: sized by it


def test_a_check_that_needs_no_history_is_not_sized_by_judging(session, monkeypatch):
    """Only a prohibition applies to the proposed action: nothing the sync publishes has to be judged, so the share alone sizes it."""
    virtual = Virtual(monkeypatch, **XHEAVY)
    harmless = {"tool_name": "shell", "arguments": {"command": "echo hello"}}
    engine = session.engine("cold", policy=policy(virtual), contracts=CONTRACTS[:2])          # the two prohibitions
    calls = virtual.converge(engine, session, budget=BUDGET, action=harmless)
    assert not stops(calls) and all(call.elapsed <= BUDGET for call in calls)
    assert all(call.payload["sync"]["cost"]["eval_us"] == 0 for call in calls[:-1])
    assert all(call.payload["sync"]["cost"]["limited_by"] == "share" for call in calls[:-1])


def test_without_the_cost_bound_the_same_regime_loses_calls(session, monkeypatch):
    """The control: with PR2's share alone (the slice is not sized by what judging it costs) the heavy regime cannot be judged
    inside the budget, so the first calls end as ``request_deadline`` although the sync keeps moving."""
    virtual = Virtual(monkeypatch, **XHEAVY)
    monkeypatch.setattr(SelfDirectEngine, "_sizing", lambda self, *a, **k: None)
    engine = session.engine("cold", policy=policy(virtual))
    calls = virtual.converge(engine, session, budget=BUDGET, limit=6)
    assert stops(calls) and calls[0].stopped == "request_deadline"


def test_a_session_judged_in_45_seconds_needs_fewer_calls(session, monkeypatch):
    virtual = Virtual(monkeypatch, **HEAVY)
    expected = one_shot(session)
    engine = session.engine("cold", policy=policy(virtual))
    calls = virtual.converge(engine, session, budget=45.0)
    assert_converged(calls, expected, budget=45.0)
    assert len(calls) <= 2


# --- a comparison never takes the answer with it ----------------------------------------------------------------------------

def report(call):
    return call.payload["incremental"]


def owed(engine):
    """How many forced (warm-up) comparisons the session still owes, as ``audit_status`` shows it."""
    return engine.audit_status(SID, provider="codex")["incremental_verification"]["session"]["forced_comparisons_left"]


def test_a_comparison_that_fits_still_runs(session, monkeypatch):
    """No regression of the verification: where judging is cheap the first slice is compared with the full evaluator as before."""
    virtual = Virtual(monkeypatch, **LIGHT)
    engine = session.engine("cold", policy=policy(virtual))
    first = virtual.audit(engine, session, budget=BUDGET)
    assert report(first)["shadow_compared"] is True and "shadow_skipped" not in report(first)
    assert virtual.spent["full"] > 0 and owed(engine) == 3                        # one of the four forced comparisons was made


def test_a_comparison_the_speed_of_the_pass_says_cannot_finish_is_not_started(session, monkeypatch):
    virtual = Virtual(monkeypatch, **HEAVY)
    engine = session.engine("cold", policy=policy(virtual))
    first = virtual.audit(engine, session, budget=BUDGET)
    assert first.stopped is None and first.payload["verdict"] in ("violation", "unknown")     # the answer was given
    assert report(first)["shadow"] is False and report(first)["shadow_skipped"] == "deadline"
    assert "shadow_compared" not in report(first) and virtual.spent["full"] == 0              # the full evaluator never ran
    assert report(first)["state"] == "saved"                                                  # the progress was kept
    assert owed(engine) == 4                                                                  # and the comparison is still owed


def test_once_the_speed_is_known_the_reference_is_not_even_read(session, monkeypatch):
    """The second call knows what folding costs (this process measured it): a comparison that cannot follow the pass is skipped
    before the events of its reference are read, so the call reads only the events it folds."""
    virtual = Virtual(monkeypatch, **HEAVY)
    engine = session.engine("cold", policy=policy(virtual))
    virtual.audit(engine, session, budget=BUDGET)
    read_before = virtual.counts["load"]
    second = virtual.audit(engine, session, budget=BUDGET)
    assert report(second)["shadow_skipped"] == "deadline" and report(second)["mode"] == "delta"
    assert virtual.counts["load"] - read_before == report(second)["events_read"]             # no second read of the whole session


def test_a_comparison_that_does_not_finish_after_all_leaves_the_answer_standing(session, monkeypatch):
    """The estimate rests on the speed of the fold; a full evaluator that is far slower than that is cut by the room the
    comparison has, not by the request: the call returns the incremental answer, unverified, with its states saved."""
    virtual = Virtual(monkeypatch, fold=0.5e-3, full=12e-3)
    engine = session.engine("cold", policy=policy(virtual))
    first = virtual.audit(engine, session, budget=BUDGET)
    assert first.stopped is None and first.elapsed <= BUDGET
    assert report(first)["shadow_skipped"] == "deadline" and report(first)["shadow"] is False
    assert "shadow_compared" not in report(first) and virtual.spent["full"] > 0              # it was started, and cut
    assert report(first)["state"] == "saved" and owed(engine) == 4
    expected = one_shot(session)
    calls = [first, *virtual.converge(engine, session, budget=BUDGET)]
    assert_converged(calls, expected)


def test_a_comparison_that_was_declined_is_not_asked_for_again_by_a_second_pass(session, monkeypatch):
    """The answer is a violation, which calls for a second pass that compares it. When the first pass has just declined the
    planned comparison because it does not fit, a second pass would fold everything again and have less time for it."""
    virtual = Virtual(monkeypatch, **XHEAVY)
    engine = session.engine("cold", policy=policy(virtual))
    first = virtual.audit(engine, session, budget=45.0)
    assert first.complete and first.payload["verdict"] == "violation"
    assert report(first)["shadow_skipped"] == "deadline" and "shadow_compared" not in report(first)
    assert virtual.counts["fold"] <= 3100 and virtual.spent["full"] == 0          # folded once, never compared
    assert first.elapsed < 25                                                     # and not made to wait for a second pass


def test_a_resident_engine_defers_a_skipped_comparison_to_its_idle_comparison(session, monkeypatch):
    virtual = Virtual(monkeypatch, **HEAVY)
    engine = session.engine("cold", policy=policy(virtual))
    worker = ShadowWorker(engine, idle_sec=0.0, poll_sec=0.01)
    engine.attach_shadow_worker(worker)
    calls = virtual.converge(engine, session, budget=BUDGET)
    assert calls[-1].complete and not stops(calls)
    assert all(report(call).get("shadow_skipped") == "deferred" for call in calls)           # every comparison went to the worker
    assert worker.snapshot()["pending_sessions"] == 1 and owed(engine) == 4
    assert worker.tick() == 1                                                                # the engine is idle: it compares now
    snapshot = worker.snapshot()
    assert snapshot["matched"] == 1 and snapshot["pending_sessions"] == 0
    assert owed(engine) == 3                                                                 # and that counted as a forced comparison


def test_without_a_resident_engine_the_skipped_comparison_is_not_called_deferred(session, monkeypatch):
    virtual = Virtual(monkeypatch, **HEAVY)
    engine = session.engine("cold", policy=policy(virtual))
    assert report(virtual.audit(engine, session, budget=BUDGET))["shadow_skipped"] == "deadline"      # nobody will verify it later


# --- synced, but never evaluated ----------------------------------------------------------------------------------------------

def synced_engine(session, virtual, name="synced", **kwargs):
    """An engine whose index holds the whole session, committed in one piece, and no monitor state: the state right after a build
    with a new evaluator version is deployed, which has no commit boundary inside the session to save a state at."""
    engine = session.engine(name, policy=policy(virtual), **kwargs)
    engine.sync_session(session_id=SID, provider="codex", path=str(session.path), embed=False, derived=False)
    return engine


def test_one_call_builds_the_states_when_it_can(session, monkeypatch):
    virtual = Virtual(monkeypatch, fold=2.0e-3, full=3.0e-3)            # 3,000 events at 2.35 ms: 7 s of the 8
    expected = one_shot(session)
    engine = synced_engine(session, virtual)
    assert lag(engine)["events_without_state"] == lag(engine)["events_indexed"] > 2900
    first = virtual.audit(engine, session, budget=BUDGET)
    assert first.complete and brief(first.payload) == brief(expected) and first.elapsed <= BUDGET
    assert lag(engine)["events_without_state"] == 0 and lag(engine)["states_through_byte"] == expected["coverage"]["byte_offset"]
    assert virtual.audit(engine, session, budget=BUDGET).elapsed < 0.5                    # and the next one only reads what is new


def test_a_build_that_does_not_fit_one_request_is_stated_not_hidden(session, monkeypatch):
    """The limit: with nothing between the states and the end of the index, one request has to fold all of it. Repeating the call
    cannot help (the status says how far behind the states are); a larger budget, the explicit escape, can."""
    virtual = Virtual(monkeypatch, **XHEAVY)                              # 14.5 s of folding against 8
    expected = one_shot(session)
    engine = synced_engine(session, virtual)
    calls = [virtual.audit(engine, session, budget=BUDGET) for _ in range(3)]
    assert [call.stopped for call in calls] == ["request_deadline"] * 3
    assert all(call.elapsed <= BUDGET + 0.1 for call in calls)           # stopped at the deadline, not after it
    assert lag(engine)["events_without_state"] == lag(engine)["events_indexed"] and lag(engine)["states_through_byte"] is None
    escape = virtual.audit(engine, session, budget=45.0)                  # audit_session(timeout_ms=45000)
    assert escape.complete and brief(escape.payload) == brief(expected)
    assert lag(engine)["events_without_state"] == 0
    assert virtual.audit(engine, session, budget=BUDGET).complete         # and the hook budget does for what follows


def test_a_resident_engine_builds_the_states_while_it_is_idle(session, monkeypatch):
    virtual = Virtual(monkeypatch, **XHEAVY)
    expected = one_shot(session)
    engine = synced_engine(session, virtual)
    worker = ShadowWorker(engine, idle_sec=0.0, poll_sec=0.01)
    engine.attach_shadow_worker(worker)
    first = virtual.audit(engine, session, budget=BUDGET)
    assert first.stopped == "request_deadline"
    assert worker.snapshot()["pending_sessions"] == 1                     # the stopped evaluation was noted, not forgotten
    assert worker.tick() == 1 and worker.snapshot()["matched"] == 1       # idle: an audit with the large budget of a background job
    assert lag(engine)["events_without_state"] == 0
    second = virtual.audit(engine, session, budget=BUDGET)
    assert second.complete and brief(second.payload) == brief(expected) and second.elapsed < 0.5


def test_a_stopped_evaluation_is_not_noted_when_the_client_went_away(session, monkeypatch):
    """Only a deadline is worth finishing in the background; a cancelled request is nobody's business any more."""
    virtual = Virtual(monkeypatch, **XHEAVY)
    engine = synced_engine(session, virtual)
    worker = ShadowWorker(engine, idle_sec=0.0, poll_sec=0.01)
    engine.attach_shadow_worker(worker)
    import threading
    from self_directing_mcp.request_control import RequestControl, RequestStopped, request_scope
    control = RequestControl(deadline=virtual.read() + BUDGET, cancel_event=threading.Event())
    original = virtual.charge

    def cancelling(kind, events):
        original(kind, events)
        if kind == "fold" and virtual.spent["fold"] > 1.0:
            control.cancel("request_cancelled")

    virtual.charge = cancelling
    with pytest.raises(RequestStopped) as stopped:
        with request_scope(control):
            engine.audit_session(SID, provider="codex", path=str(session.path))
    assert stopped.value.reason == "request_cancelled" and worker.snapshot()["pending_sessions"] == 0


# --- what an index already owes -----------------------------------------------------------------------------------------------

def half_indexed_session(tmp_path, monkeypatch, virtual, *, ignore_debt=False):
    """Half of a session is indexed and no state covers it (the states of an older evaluator, say) and the other half was
    appended since: the engine, the session and the one-shot answer for the whole file."""
    half = Session(tmp_path, 1500)
    engine = half.engine("debt", policy=policy(virtual))
    engine.sync_session(session_id=SID, provider="codex", path=str(half.path), embed=False, derived=False)
    half.append(1500)
    reference = half.engine("reference")
    reference.sync_session(session_id=SID, provider="codex", path=str(half.path), embed=False, derived=False)
    expected = reference.audit_session(SID, provider="codex", path=str(half.path))
    COSTS.reset()
    if ignore_debt:
        real = SelfDirectEngine._sizing_numbers
        monkeypatch.setattr(SelfDirectEngine, "_sizing_numbers",
                            lambda self, *a: (lambda numbers: numbers and (numbers[0], 0))(real(self, *a)))
    return half, engine, expected


def test_the_slice_leaves_room_for_the_events_that_no_state_covers(tmp_path, monkeypatch):
    """The call has to fold the first half before it can answer, so the slice it syncs is cut by that, and the call still fits."""
    virtual = Virtual(monkeypatch, **HEAVY)
    half, engine, expected = half_indexed_session(tmp_path, monkeypatch, virtual)
    try:
        calls = virtual.converge(engine, half, budget=BUDGET)
        assert_converged(calls, expected)
        cost = calls[0].payload["sync"]["cost"]
        assert cost["debt_events"] > 1400 and cost["limited_by"] == "cost"
        assert calls[0].payload["coverage"]["byte_offset"] < 2 * half.path.stat().st_size // 3     # a short slice: little of the half
    finally:
        half.close()


def test_without_the_debt_in_the_bound_the_same_call_does_not_fit(tmp_path, monkeypatch):
    """The control: sized as if the first half were not waiting, the slice is too long to be judged with it and the call is lost."""
    virtual = Virtual(monkeypatch, **HEAVY)
    half, engine, _expected = half_indexed_session(tmp_path, monkeypatch, virtual, ignore_debt=True)
    try:
        first = virtual.audit(engine, half, budget=BUDGET)
        assert first.stopped == "request_deadline"
    finally:
        half.close()


# --- the comparison's arithmetic ----------------------------------------------------------------------------------------------

def test_the_shortfall_of_a_comparison_is_decided_by_the_room_after_the_pass(monkeypatch):
    from self_directing_mcp.audit import incremental
    from self_directing_mcp.request_control import RequestControl, request_scope
    now = [100.0]
    monkeypatch.setattr(incremental, "time", type("T", (), {"monotonic": staticmethod(lambda: now[0])}))
    short = incremental.comparison_shortfall
    assert short(10_000, 1e-3) is None                                    # no request: no opinion
    with request_scope(RequestControl(deadline=108.0)):
        assert short(10_000, None) is None                                # nothing known about the cost: no opinion
        # 8 s left, 1 s kept back for the answer: 7 s of room, of which a comparison may use half. It costs 1.5 folds an
        # event, with a margin of 25 %: 1.875 ms an event at a fold of 1 ms.
        assert short(1_800, 1e-3) is None                                 # 1,800 x 1.875 ms = 3.4 s <= 3.5 s
        assert short(2_000, 1e-3) == "deadline"                           # 2,000 x 1.875 ms = 3.75 s
        assert short(2_000, 1e-3, defer=True) == "deferred"               # a resident engine will verify it instead
        assert short(3_800, 1e-3, scale=0.5) == "deadline"                # a comparison of half the contracts costs half:
        assert short(3_700, 1e-3, scale=0.5) is None                      # 3,700 x 0.94 ms = 3.5 s
        assert short(0, 1e-3, delta=1_000) is None
        # The fold still to come comes off the room first: 2,000 events at 1 ms, with the margin, 2.5 s; 4.5 s are left, 2.25 s usable.
        assert short(1_100, 1e-3, delta=2_000) is None and short(1_300, 1e-3, delta=2_000) == "deadline"
        now[0] += 4.0                                                     # time passes: 4 s left, 0.97 s of it kept back
        assert short(1_800, 1e-3) == "deadline"


# --- what a call says, and what a small session does not pay ----------------------------------------------------------------

def test_a_sync_session_says_the_numbers_that_sized_its_slice(session, monkeypatch):
    virtual = Virtual(monkeypatch, **HEAVY)
    engine = session.engine("direct", policy=policy(virtual))
    info = engine.sync_session(session_id=SID, provider="codex", path=str(session.path), embed=False, derived=False,
                               _deadline=virtual.read() + BUDGET)
    sync = info["sync"]
    assert sync["state"] == "partial"
    # Nothing is judged after a direct sync, so the share of the request alone sizes the slice.
    assert sync["cost"] == {"slice_us": 2000.0, "eval_us": 0.0, "debt_events": 0, "limited_by": "share"}


def test_audit_status_shows_the_cost_model_and_how_far_the_states_lag(session, monkeypatch):
    virtual = Virtual(monkeypatch, **HEAVY)
    engine = session.engine("status", policy=policy(virtual))
    before = engine.audit_status(SID, provider="codex")["evaluation"]
    assert before["session"] == {"events_indexed": 0, "events_without_state": 0, "states_through_byte": None}
    assert before["cost"]["eval_sets_measured"] == 0 and before["cost"]["defaults_us"]["eval"] == pytest.approx(
        eval_cost.DEFAULT_EVAL_SEC * 1e6, rel=0.01)
    virtual.converge(engine, session, budget=BUDGET)
    after = engine.audit_status(SID, provider="codex")["evaluation"]
    assert after["cost"]["eval_sets_measured"] == 1 and after["cost"]["slice_us"]["codex"] > 0
    assert after["session"]["events_without_state"] == 0 and after["session"]["states_through_byte"] > 0
    assert "evaluation" in engine.audit_status()                                    # and without a session: the model alone
    assert engine.audit_status()["evaluation"]["session"] is None


def test_a_small_session_never_pays_for_the_cost_bound(tmp_path, monkeypatch):
    """Real time, the shipped slice policy: a session far below the minimum progress of a slice is never cut, so the contracts
    are not read for sizing, nothing is measured, and a complete answer carries no ``sync`` object."""
    import time
    small = Session(tmp_path, 300)
    try:
        engine = small.engine("warm")
        asked = []
        real = SelfDirectEngine._sizing_numbers
        monkeypatch.setattr(SelfDirectEngine, "_sizing_numbers", lambda self, *a: asked.append(a) or real(self, *a))
        for _ in range(3):
            # Real time: nothing here is cut by a clock (the session is far below the minimum progress of a slice), the deadline
            # only has to be far enough for a slow machine.
            answer = engine.audit_session(SID, provider="codex", path=str(small.path), _deadline=time.monotonic() + 600)
            assert answer["coverage"]["complete"] and "sync" not in answer
            small.append(20)
        assert asked == []                                                         # the contracts were not read for the bound
        assert COSTS.snapshot()["eval_sets_measured"] == 0 and COSTS.snapshot()["slice_us"] == {}
    finally:
        small.close()


def test_without_incremental_audits_the_slices_are_those_of_the_share(session, monkeypatch):
    """Every audit then evaluates the whole session again whatever is synced: there is no delta to size a slice by."""
    virtual = Virtual(monkeypatch, **LIGHT)
    engine = session.engine("full", policy=policy(virtual), audit_incremental=False)
    first = virtual.audit(engine, session, budget=BUDGET)
    assert first.payload["sync"]["cost"] == {"slice_us": 2000.0, "eval_us": 0.0, "debt_events": 0, "limited_by": "share"}


def test_numbers_that_cannot_be_read_never_fail_the_sync_they_size(session, monkeypatch):
    import sqlite3
    virtual = Virtual(monkeypatch, **LIGHT)
    engine = session.engine("unreadable", policy=policy(virtual))

    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("no such table")

    monkeypatch.setattr(type(engine.store), "audit_debt", broken)
    first = virtual.audit(engine, session, budget=BUDGET)
    assert first.stopped is None and first.payload["sync"]["cost"]["debt_events"] == 0     # sized without the debt, nothing failed
    assert first.payload["sync"]["cost"]["eval_us"] > 0


def test_a_resident_engine_that_holds_its_states_in_memory_still_reports_them(session, monkeypatch):
    """The daemon's engines write their monitor states with the next commit; the lag and the debt count the held ones too."""
    virtual = Virtual(monkeypatch, **LIGHT)
    engine = session.engine("held", policy=policy(virtual), defer_derived=True)
    calls = virtual.converge(engine, session, budget=BUDGET)
    assert calls[-1].complete and engine.store.has_pending_derived()                 # the states are still only in memory
    assert lag(engine)["events_without_state"] == 0 and lag(engine)["states_through_byte"] > 0
    assert engine.store.audit_debt(SID, "codex") == 0


def test_only_a_fold_that_every_monitor_shared_says_what_a_contract_set_costs(session, monkeypatch):
    """A contract added to a session that is judged already is folded from the start while the others are up to date: that fold
    says nothing about what the whole set costs per event, so the cost model does not learn from it."""
    virtual = Virtual(monkeypatch, **LIGHT)
    engine = session.engine("mixed", policy=policy(virtual), contracts=CONTRACTS[:2])
    virtual.converge(engine, session, budget=BUDGET)
    measured = COSTS.snapshot()
    assert measured["eval_sets_measured"] == 1
    # One more prohibition on tool calls: it reads what the others read, so only its own monitor is rebuilt, from the start.
    engine.upsert_contracts([{"id": "no-sudo", "type": "must_not", "scope": "tool_call", "regex": r"sudo\s", "provider": "codex"}])
    answer = virtual.audit(engine, session, budget=BUDGET)
    assert answer.complete and answer.payload["incremental"]["rebuilt"]              # only its monitor was folded over everything
    assert COSTS.snapshot() == measured                                              # and the model did not move


def test_states_of_an_earlier_generation_do_not_count_as_covering_the_events(tmp_path, monkeypatch):
    """A reindex starts a new generation of the session and drops the states of the old one: every event is owed again."""
    from self_directing_mcp.codex import parse as codex_parser
    small = Session(tmp_path, 400)
    try:
        engine = small.engine("generations")
        engine.audit_session(SID, provider="codex", path=str(small.path))
        assert lag(engine)["events_without_state"] == 0 and lag(engine)["states_through_byte"] > 0
        monkeypatch.setattr(codex_parser, "PARSER_VERSION", codex_parser.PARSER_VERSION + 1)
        info = engine.sync_session(session_id=SID, provider="codex", path=str(small.path), embed=False, derived=False)
        assert info["reindexed"] is True
        owed = lag(engine)
        assert owed["events_without_state"] == owed["events_indexed"] > 350 and owed["states_through_byte"] is None
    finally:
        small.close()
