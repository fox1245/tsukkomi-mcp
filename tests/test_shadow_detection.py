"""Fault injection: how fast a defect of the incremental path is noticed under the comparison schedule.

``scripts/measure_shadow_detection.py`` injects a defect into an engine that audits a growing synthetic session and counts
the requests until a comparison with the full evaluator finds it. These tests run its scenarios on a few seeds (the
script itself reports the distribution over hundreds). The forced comparisons must notice a wrong block, a violation that
turns clean and a high-risk contract in the request that carries them; a silently missed violation is noticed by sampling
only, so the tests bound the delay and check that the answers returned meanwhile are counted, never hidden.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import statistics

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("measure_shadow_detection", ROOT / "scripts" / "measure_shadow_detection.py")
measure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure)

pytestmark = pytest.mark.allow_incremental_mismatch      # the defects are the point: the suite's strict comparison would raise


def trials(scenario, seeds, tmp_path, **kwargs):
    return [measure.run_trial(scenario, seed, cap=1500, tmp=str(tmp_path), **kwargs) for seed in seeds]


@pytest.mark.parametrize("scenario,extra", [("wrong_block", {"violation_at": 12}), ("flip", {"violation_at": 0}),
                                            ("high_risk", {"violation_at": 12})])
def test_the_forced_comparisons_notice_these_defects_in_the_request_that_carries_them(tmp_path, scenario, extra):
    found = trials(scenario, range(4), tmp_path, every="adaptive", floor=False, **extra)
    assert all(item == {"requests": 1, "wrong": 0} for item in found), found


@pytest.mark.parametrize("scenario,extra", [("wrong_block", {"violation_at": 12}), ("flip", {"violation_at": 0}),
                                            ("high_risk", {"violation_at": 12})])
def test_the_forced_comparisons_do_not_depend_on_the_rate(tmp_path, scenario, extra):
    """The same defects, with the schedule at its floor and with the sampling switched off (``every=0``)."""
    for every, floor in (("adaptive", True), (0, False)):
        found = trials(scenario, range(3), tmp_path, every=every, floor=floor, **extra)
        assert all(item == {"requests": 1, "wrong": 0} for item in found), (every, floor, found)


def test_a_missed_violation_inside_the_warmup_is_noticed_at_once(tmp_path):
    found = trials("missed", range(4), tmp_path, every="adaptive", floor=False, violation_at=3)
    assert all(item == {"requests": 1, "wrong": 0} for item in found), found


def test_a_missed_violation_is_noticed_by_sampling_and_the_wrong_answers_are_counted(tmp_path):
    cold = trials("missed", range(6), tmp_path, every="adaptive", floor=False, violation_at=9)
    floor = trials("missed", range(6), tmp_path, every="adaptive", floor=True, violation_at=0)
    fixed = trials("missed", range(6), tmp_path, every=16, floor=False, violation_at=9)
    for group in (cold, floor, fixed):
        assert all(item["requests"] is not None for item in group), group
        assert all(item["wrong"] == item["requests"] - 1 for item in group)         # every earlier answer was wrong
    # The trade the policy makes, in numbers: at the floor (1/32) the defect stays unnoticed longer than under 1/16.
    assert statistics.mean(item["requests"] for item in floor) > statistics.mean(item["requests"] for item in fixed)


def test_with_only_the_forced_comparisons_a_silent_miss_is_never_noticed(tmp_path):
    """Said plainly: ``every=0`` is the forced comparisons and nothing else, and a silently missed violation is not one."""
    found = measure.run_trial("missed", 1, every=0, floor=False, violation_at=20, cap=80, tmp=str(tmp_path))
    assert found["requests"] is None and found["wrong"] == 80 - 20


def test_the_simulation_of_the_schedule_agrees_with_the_numbers_the_documentation_quotes():
    """``--simulate`` and ``--share`` play the schedule alone; the README quotes their output. The draw is a hash, so the
    numbers are deterministic: the windows below are a few percent wide around them (a change of a constant, of the floor or
    of the draw moves them out; the expected medians of a geometric wait are ln 2 times the interval: 22 at 1/32, 11 at 1/16)."""
    floor = [measure.simulate_missed(seed, "adaptive", True, 0)["requests"] for seed in range(600)]
    fixed = [measure.simulate_missed(seed, 16, False, 9)["requests"] for seed in range(600)]
    cold = [measure.simulate_missed(seed, "adaptive", False, 3)["requests"] for seed in range(100)]
    after_warmup = [measure.simulate_missed(seed, "adaptive", False, 9)["requests"] for seed in range(600)]
    assert 19 <= statistics.median(floor) <= 23 and 28 <= statistics.mean(floor) <= 34 and max(floor) <= 260     # 1/32 at the floor
    assert 9 <= statistics.median(fixed) <= 13 and 13 <= statistics.mean(fixed) <= 18                            # 1/16 before
    assert 2 <= statistics.median(after_warmup) <= 4 and statistics.mean(after_warmup) < 6                       # 1/4 just after the warm-up
    assert cold == [1] * 100                                   # inside the warm-up the first request is compared
    share = measure.share_of_requests(20)
    assert 52 <= share["10"] <= 58 and 15 <= share["100"] <= 19 and 8 <= share["300"] <= 10 and 4.4 <= share["1000"] <= 5.6
    assert 3.7 <= share["2000"] <= 4.5 and 3.3 <= share["5000"] <= 3.9          # below the 5% a p95 sees after about 1,000 requests
    old = measure.share_of_requests(20, 16)
    assert 15 <= old["10"] <= 18 and 6.2 <= old["1000"] <= 6.8 and 6.0 <= old["5000"] <= 6.6       # 1 in 16, plus the first request
    cost = measure.session_cost(10, "adaptive", (300,))["300"]
    previous = measure.session_cost(10, 16, (300,))["300"]
    assert 24 <= cost["comparisons"] <= 28 and 17 <= previous["comparisons"] <= 21     # a young session compares more than 1 in 16 did
    assert 11 <= cost["unnoticed"] <= 17 and 12 <= previous["unnoticed"] <= 18
