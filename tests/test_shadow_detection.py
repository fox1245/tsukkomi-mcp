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
    # The trade the policy makes, in numbers: at the floor (1/64) the defect stays unnoticed longer than under 1/16.
    assert statistics.mean(item["requests"] for item in floor) > statistics.mean(item["requests"] for item in fixed)


def test_with_only_the_forced_comparisons_a_silent_miss_is_never_noticed(tmp_path):
    """Said plainly: ``every=0`` is the forced comparisons and nothing else, and a silently missed violation is not one."""
    found = measure.run_trial("missed", 1, every=0, floor=False, violation_at=20, cap=80, tmp=str(tmp_path))
    assert found["requests"] is None and found["wrong"] == 80 - 20


def test_the_simulation_of_the_schedule_agrees_with_the_numbers_the_documentation_quotes():
    """``--simulate`` and ``--share`` play the schedule alone; the README quotes their output."""
    floor = [measure.simulate_missed(seed, "adaptive", True, 0)["requests"] for seed in range(600)]
    fixed = [measure.simulate_missed(seed, 16, False, 9)["requests"] for seed in range(600)]
    cold = [measure.simulate_missed(seed, "adaptive", False, 3)["requests"] for seed in range(100)]
    assert 30 <= statistics.median(floor) <= 60 and 8 <= statistics.median(fixed) <= 14
    assert cold == [1] * 100                                   # inside the warm-up the first request is compared
    share = measure.share_of_requests(20)
    assert share["10"] > 80 and share["100"] > 25 and share["2000"] < 8 and share["5000"] < 5
