"""The cost model (``eval_cost``) and the cost bound of a sync slice (``slice_budget``).

What one event costs to parse and commit, to fold into the monitors and to compare is measured in this process and kept as a
moving average; a request that will also judge what it syncs sizes its slice by it. These tests pin the numbers (defaults,
priors, the average), the arithmetic of the bound with a clock the test moves, and the two rules that keep it safe: the share of
the request stays the upper bound, and a parse never stops before the minimum progress.
"""
from __future__ import annotations

import threading

import pytest

from self_directing_mcp import eval_cost
from self_directing_mcp.eval_cost import CostModel
from self_directing_mcp.slice_budget import CLOCK_EVERY, Sizing, SliceBudget, current_sizing, sizing_scope


@pytest.fixture
def model():
    return CostModel()


# --- the defaults ----------------------------------------------------------------------------------------------------------

def test_the_defaults_are_the_figures_of_the_design(model):
    assert model.slice_cost("codex") == pytest.approx(40e-6)          # parse + commit of one event, before anything was measured
    assert model.eval_cost("any", 0) == 0.0                           # no contract needs history: nothing is judged after a sync
    assert model.eval_cost(None, 0) == 0.0
    # With history: 60 microseconds is the floor of the prior, the margin of an unmeasured set is on top of it.
    assert model.eval_cost("any", 3) == pytest.approx(60e-6 * eval_cost.UNMEASURED_MARGIN)
    assert model.reference_cost("any") is None and not model.observed("any")


def test_the_prior_grows_with_the_number_of_contracts_that_need_history(model):
    few, many = eval_cost.prior_eval_sec(2), eval_cost.prior_eval_sec(40)
    assert few == pytest.approx(60e-6)                               # the floor
    assert many == pytest.approx(eval_cost.EVAL_BASE_SEC + 40 * eval_cost.EVAL_PER_CONTRACT_SEC) and many > 2 * few
    assert eval_cost.prior_eval_sec(0) == pytest.approx(60e-6)


# --- the moving average ----------------------------------------------------------------------------------------------------

def test_the_first_measurement_replaces_the_prior_and_later_ones_are_averaged(model):
    model.observe_eval("set", 10_000, 0.5)                           # 50 microseconds per event
    assert model.observed("set") and model.eval_cost("set", 5) == pytest.approx(50e-6)
    model.observe_eval("set", 10_000, 1.0)                           # 100 microseconds: the average moves by SMOOTHING
    assert model.eval_cost("set", 5) == pytest.approx(50e-6 + eval_cost.SMOOTHING * 50e-6)
    for _ in range(40):
        model.observe_eval("set", 10_000, 1.0)
    assert model.eval_cost("set", 5) == pytest.approx(100e-6, rel=0.01)  # it converges on a steady cost
    model.observe_slice("codex", 5000, 0.25)
    assert model.slice_cost("codex") == pytest.approx(50e-6) and model.slice_cost("omp") == pytest.approx(40e-6)  # per provider


def test_a_measurement_over_few_events_or_no_time_is_noise(model):
    model.observe_eval("set", eval_cost.MIN_SAMPLE_EVENTS - 1, 5.0)
    model.observe_eval("set", 10_000, 0.0)
    model.observe_eval("set", 10_000, -1.0)
    model.observe_eval(None, 10_000, 1.0)
    model.observe_slice("codex", 10, 5.0)
    assert not model.observed("set") and model.slice_cost("codex") == pytest.approx(40e-6)


def test_an_absurd_measurement_is_clamped_not_believed(model):
    model.observe_eval("set", 10_000, 1e-9)                          # a nanosecond for ten thousand events
    assert model.eval_cost("set", 1) == pytest.approx(eval_cost.FLOOR_SEC)
    model.observe_eval("other", 10_000, 1e9)
    assert model.eval_cost("other", 1) == pytest.approx(eval_cost.CEILING_SEC)


def test_only_the_most_recent_contract_sets_are_remembered(model, monkeypatch):
    monkeypatch.setattr(eval_cost, "KEEP_KEYS", 3)
    for n in range(6):
        model.observe_eval(f"set-{n}", 10_000, 0.1 * (n + 1))
    assert not model.observed("set-0") and not model.observed("set-2") and model.observed("set-3") and model.observed("set-5")


def test_the_snapshot_holds_numbers_and_no_name_of_a_session_a_path_or_a_contract(model):
    model.observe_eval("codex:5f3a-secret-contract-digest", 10_000, 0.5)
    model.observe_slice("codex", 10_000, 0.4)
    snapshot = model.snapshot()
    assert snapshot["slice_us"] == {"codex": 40.0} and snapshot["eval_us_median"] == 50.0 and snapshot["eval_sets_measured"] == 1
    assert snapshot["defaults_us"] == {"slice": 40.0, "eval": 60.0}
    assert "secret-contract" not in repr(snapshot)


def test_the_model_is_safe_under_threads(model):
    def hammer():
        for _ in range(200):
            model.observe_eval("set", 5000, 0.2)
            model.observe_slice("codex", 5000, 0.2)
            model.eval_cost("set", 3)
            model.snapshot()

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert model.eval_cost("set", 3) == pytest.approx(40e-6)


# --- the lazy numbers of a request -----------------------------------------------------------------------------------------

def test_sizing_reads_its_numbers_once_and_only_when_asked():
    asked = []
    sizing = Sizing(100.0, lambda: asked.append(1) or (2e-3, 500))
    assert asked == [] and sizing.eval_per_event == 0.0              # nothing is read until a slice needs it
    assert sizing.resolve() is True and sizing.resolve() is True
    assert asked == [1] and sizing.eval_per_event == 2e-3 and sizing.debt_events == 500


def test_a_request_that_judges_nothing_resolves_to_no_cost():
    assert Sizing(100.0, lambda: None).resolve() is False            # only prohibitions on a proposed action: no history
    assert Sizing(100.0).resolve() is False
    clamped = Sizing(100.0, lambda: (-1.0, -5))
    assert clamped.resolve() is False and clamped.debt_events == 0


def test_the_scope_puts_the_sizing_in_view_and_takes_it_out():
    sizing = Sizing(100.0)
    assert current_sizing() is None
    with sizing_scope(sizing):
        assert current_sizing() is sizing
    assert current_sizing() is None


# --- the bound of a slice --------------------------------------------------------------------------------------------------

class Moving:
    """A clock the test moves by hand."""

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def look_until_stop(budget, clock, per_record, limit=100_000):
    """Feed records of ``per_record`` seconds each to the budget the way ``iter_raw_lines`` does; returns the records read."""
    records, next_look = 0, 0
    while records < limit:
        if records >= next_look:
            reason, next_look = budget.look(records, records * 200)
            if reason is not None:
                budget.stop(reason, records, records * 200)
                return records
        records += 1
        clock.now += per_record
    return None


def cost_budget(clock, **kwargs):
    options = dict(soft_deadline=clock.now + 1000.0, min_records=1, min_bytes=0, clock=clock, started=clock.now,
                   commit_factor=1.3)
    options.update(kwargs)
    return SliceBudget(0, **options)


def test_a_slice_ends_when_parsing_committing_and_judging_it_no_longer_fit():
    clock = Moving()
    # 10 ms to parse a record (13 ms with its commit) and 20 ms to judge one: 33 ms a record; 6.6 s of room: 200 records.
    budget = cost_budget(clock, cost_per_record=0.02, cost_budget=6.6)
    records = look_until_stop(budget, clock, 0.01)
    assert budget.stopped_by == "soft_deadline" and budget.bound == "cost"
    assert 200 <= records <= 200 + CLOCK_EVERY                       # the bound is looked at every CLOCK_EVERY records


def test_the_parse_speed_in_the_bound_is_the_one_the_slice_shows():
    slow, quick = Moving(), Moving()
    stopped = {}
    for name, clock, per_record in (("slow", slow, 0.02), ("quick", quick, 0.002)):
        budget = cost_budget(clock, cost_per_record=0.02, cost_budget=6.6)
        stopped[name] = look_until_stop(budget, clock, per_record)
    assert stopped["quick"] > stopped["slow"]                        # cheaper to parse: more records fit beside the judging
    # 2 ms parse (2.6 with the commit) + 20 ms judging per record: 6.6 s hold 292 records; 20 ms + 26 ms: 143.
    assert 292 <= stopped["quick"] <= 292 + CLOCK_EVERY and 143 <= stopped["slow"] <= 143 + CLOCK_EVERY


def test_the_share_stays_the_upper_bound():
    clock = Moving()
    # A bound far looser than the soft deadline never decides: the soft deadline (1 s: 100 records) does.
    budget = cost_budget(clock, soft_deadline=clock.now + 1.0, cost_per_record=1e-6, cost_budget=1e6)
    records = look_until_stop(budget, clock, 0.01)
    assert budget.bound == "share" and 100 <= records <= 100 + CLOCK_EVERY


def test_the_minimum_progress_holds_even_when_the_bound_is_spent():
    clock = Moving()
    budget = cost_budget(clock, min_records=500, min_bytes=500 * 200, cost_per_record=0.02, cost_budget=0.0)
    records = look_until_stop(budget, clock, 0.001)
    assert budget.bound == "cost" and 500 <= records <= 500 + CLOCK_EVERY       # not before 500 records, then at the next look


def test_without_a_cost_budget_nothing_changes():
    clock = Moving()
    plain = SliceBudget(0, soft_deadline=clock.now + 1.0, min_records=1, min_bytes=0, clock=clock)
    records = look_until_stop(plain, clock, 0.01)
    assert plain.bound == "share" and plain.cost_budget is None and 100 <= records <= 100 + CLOCK_EVERY


def test_a_cost_budget_needs_a_start():
    with pytest.raises(ValueError):
        SliceBudget(0, soft_deadline=1.0, cost_budget=1.0)
