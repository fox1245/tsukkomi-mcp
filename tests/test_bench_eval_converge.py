"""scripts/bench_eval_converge.py runs end to end on a tiny synthetic session and reports what a reader needs to trust it."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Real time, a tiny session: every call is far inside the budget. The budget is the longest one the benchmark offers (45 s) so that
# a runner that is many times slower than usual (a sanitizer build is 3 to 4 times slower, a loaded runner more) still has room: the
# test is about what the report says, not about how fast the machine is.


@pytest.mark.parametrize("tool", ["audit", "check"])
def test_reports_every_combination_with_its_calls_and_checks_the_answer(tmp_path, tool):
    report = tmp_path / "report.json"
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "bench_eval_converge.py"), "--events", "2500", "--sets", "2,18",
         "--budgets", "45", "--scenarios", "cold,synced", "--tool", tool, "--dir", str(tmp_path), "--json", str(report)],
        capture_output=True, text=True, timeout=240, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    runs = json.loads(report.read_text())["runs"]
    assert [(run["contracts"], run["scenario"]) for run in runs] == [(2, "cold"), (2, "synced"), (18, "cold"), (18, "synced")]
    for run in runs:
        assert run["tool"] == tool
        assert run["converged"] and run["answer_equals_one_shot"] and run["inside_budget"] and run["lost"] == 0
        assert run["k"] == len(run["calls"]) >= 1 and run["calls"][-1]["complete"]
    assert "| events | contracts | budget | start | K |" in done.stdout
    assert not list(tmp_path.glob("bench-eval-*")), "the temporary directory is deleted at the end"


def test_a_resident_engine_is_a_scenario_of_its_own(tmp_path):
    report = tmp_path / "report.json"
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "bench_eval_converge.py"), "--events", "2500", "--sets", "6",
         "--budgets", "45", "--scenarios", "cold+idle,synced+idle", "--dir", str(tmp_path), "--json", str(report)],
        capture_output=True, text=True, timeout=240, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    runs = json.loads(report.read_text())["runs"]
    assert [run["scenario"] for run in runs] == ["cold+idle", "synced+idle"]
    assert all(run["converged"] and run["answer_equals_one_shot"] for run in runs)
