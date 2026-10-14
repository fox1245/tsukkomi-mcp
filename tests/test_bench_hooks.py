"""scripts/bench_hooks.py: thresholds, gating decisions and a tiny end-to-end run, all on synthetic data."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench_hooks.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_hooks_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cell(host, event, wall_p95, cpu_p95, *, failed_open=0, session="small"):
    row = {"wall_ms": {"p50": wall_p95 / 2, "p95": wall_p95, "max": wall_p95},
           "cpu_ms": {"p50": cpu_p95 / 2, "p95": cpu_p95, "max": cpu_p95},
           "peak_rss_mb": {"max": 50}, "decisions": {"empty": 1}, "failed_open": failed_open}
    return {"host": host, "contracts": 10, "session": session, "session_bytes": 1, "events": {event: row}}


def status(rows, metric):
    return next(row["status"] for row in rows if row["id"] == metric)


def test_targets_carry_the_minutes_numbers_and_the_ci_factor_is_one_and_a_half(bench):
    by_id = {t["id"]: t for t in bench.TARGETS}
    assert by_id["cold_hook_wall_p95_ms"]["target"] == 400 and by_id["cold_hook_wall_p50_ms"]["target"] == 150
    assert by_id["cold_hook_wall_p95_all_ms"]["target"] == 400
    assert by_id["cold_hook_cpu_s"]["target"] == 0.3 and by_id["cold_hook_cpu_all_s"]["target"] == 0.3
    assert by_id["warm_check_action_unchanged_cpu_ms"]["target"] == 100
    assert by_id["warm_incremental_audit_cpu_ms"]["target"] == 20
    assert by_id["idle_server_rss_mb"]["target"] == 90
    assert bench.CI_FACTOR == 1.5


def test_work_that_depends_on_other_branches_is_reported_but_never_gated(bench):
    gated = {t["id"] for t in bench.TARGETS if t["gated"]}
    assert gated == {"cold_hook_wall_p95_ms", "cold_hook_cpu_s", "idle_server_rss_mb"}
    assert bench.GATED_COLD_EVENTS == {"claude": ("PostToolUse", "UserPromptSubmit")}
    report = {"cold": [], "burst": [], "warm": [{
        "session": "large", "contracts": 10, "scenarios": {
            "check_action_nothing_appended": {"cpu_ms": {"p50": 10_000, "p95": 10_000}}}}]}
    row = status(bench.evaluate(report), "warm_check_action_unchanged_cpu_ms")
    assert row.startswith("reported, not gated yet") and "over target" in row


def test_a_gated_metric_passes_below_the_threshold_and_fails_above_it(bench):
    ok = bench.evaluate({"cold": [cell("claude", "PostToolUse", 350, 250)]})
    assert status(ok, "cold_hook_wall_p95_ms").startswith("pass (within")
    between = bench.evaluate({"cold": [cell("claude", "UserPromptSubmit", 500, 400)]})
    assert status(between, "cold_hook_wall_p95_ms").startswith("pass (above target")
    assert status(between, "cold_hook_cpu_s").startswith("pass (above target")
    over = bench.evaluate({"cold": [cell("claude", "UserPromptSubmit", 601, 451)]})
    assert status(over, "cold_hook_wall_p95_ms") == "FAIL" and status(over, "cold_hook_cpu_s") == "FAIL"
    assert status(bench.evaluate({"cold": [cell("claude", "UserPromptSubmit", 601, 451)]}, 2.0),
                  "cold_hook_wall_p95_ms").startswith("pass")                    # --ci-factor widens the threshold


def test_only_the_gated_events_can_fail_the_cold_gates(bench):
    report = {"cold": [cell("claude", "PostToolUse", 100, 100), cell("agy", "Stop", 1700, 900),
                       cell("claude", "Stop", 900, 800), cell("claude", "PreToolUse", 800, 700),
                       cell("claude", "SessionStart", 900, 800),
                       cell("claude", "Stop", 9000, 9000, session="large")]}
    rows = bench.evaluate(report)
    assert status(rows, "cold_hook_wall_p95_ms").startswith("pass") and status(rows, "cold_hook_cpu_s").startswith("pass")
    assert next(r for r in rows if r["id"] == "cold_hook_wall_p95_ms")["value"] == 100
    everything = next(r for r in rows if r["id"] == "cold_hook_wall_p95_all_ms")
    assert everything["value"] == 1700 and everything["status"].startswith("reported, not gated yet (over target)")
    assert status(rows, "cold_hook_cpu_all_s").startswith("reported, not gated yet")
    assert not any(r["status"] == "FAIL" for r in rows)


def test_hooks_that_fail_open_invalidate_the_run(bench):
    rows = bench.evaluate({"cold": [cell("agy", "Stop", 100, 100, failed_open=2)]})
    assert status(rows, "hooks_failed_open") == "FAIL"
    assert status(bench.evaluate({"cold": [cell("agy", "Stop", 100, 100)]}), "hooks_failed_open").startswith("pass")


def test_percentiles_use_nearest_rank(bench):
    assert bench.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95) == 10
    assert bench.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.5) == 5
    assert bench.summarize([5.0, 1.0, 3.0]) == {"n": 3, "p50": 3.0, "p95": 5.0, "max": 5.0}


def test_the_contract_mix_has_the_four_judgement_shapes_and_the_requested_count(bench):
    for count in (10, 100):
        contracts = bench._tmpl_contracts(count)
        assert len(contracts) == count and len({c["id"] for c in contracts}) == count
        kinds = {(c["type"], bool(c.get("before_regex")), bool(c.get("requires_success"))) for c in contracts}
        assert kinds == {("must_not", False, False), ("must", False, False), ("must", True, False), ("must", True, True)}


def test_synthetic_transcripts_grow_by_whole_records_and_reach_the_requested_size(bench, tmp_path):
    for kind, path in (("claude", tmp_path / "c.jsonl"), ("agy", tmp_path / "a.jsonl"), ("codex", tmp_path / "x.jsonl")):
        transcript = bench.Transcript(kind, path, "aaaaaaaa-1111-4111-8111-aaaaaaaa0001")
        transcript.create(50_000)
        assert path.stat().st_size >= 50_000
        before = path.stat().st_size
        assert transcript.append(2000) >= 2000 and transcript.append(0) == 0
        data = path.read_bytes()
        assert data.endswith(b"\n") and len(data) > before
        assert all(json.loads(line) for line in data.splitlines())


def test_compare_renders_before_and_after_rows(bench):
    before = {"repo": {"describe": "old"}, "machine": {"platform": "p"}, "cold": [cell("claude", "Stop", 400, 300)]}
    after = {"repo": {"describe": "new"}, "machine": {"platform": "p"}, "cold": [cell("claude", "Stop", 200, 100)]}
    text = bench.compare(before, after)
    assert "| claude | Stop | 10 | 200 | 100 |" in text


# The benchmark gives its hook processes a clean environment, so under the sanitizer profiles of
# scripts/verify_memory.py they would load the instrumented native extension without the preloaded runtime. Timings
# under a sanitizer mean nothing anyway; the CI bench job runs this end to end.
@pytest.mark.skipif(any(name in os.environ.get("LD_PRELOAD", "") for name in ("libasan", "libtsan")),
                    reason="hook timings are meaningless under a sanitizer runtime")
def test_a_small_end_to_end_run_produces_a_valid_report_and_no_fail_open(tmp_path):
    report_path, table_path = tmp_path / "bench.json", tmp_path / "bench.md"
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--contracts", "4", "--runs", "1", "--warm-repeat", "2", "--servers", "1",
         "--sessions", "2", "--session-mb", "1", "--only", "cold,warm,servers", "--out", str(report_path),
         "--markdown", str(table_path), "--workdir", str(tmp_path / "work")],
        capture_output=True, text=True, timeout=600, cwd=tmp_path)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    report = json.loads(report_path.read_text())
    assert {c["host"] for c in report["cold"]} == {"claude", "agy"}
    assert {e for c in report["cold"] if c["host"] == "claude" for e in c["events"]} == set(
        ("PreToolUse", "PostToolUse", "UserPromptSubmit", "SessionStart", "Stop"))
    assert {e for c in report["cold"] if c["host"] == "agy" for e in c["events"]} == set(
        ("PreToolUse", "PostToolUse", "PreInvocation", "Stop"))
    assert all(row["failed_open"] == 0 for c in report["cold"] for row in c["events"].values())
    assert all(row["cpu_ms"]["p50"] > 0 for c in report["cold"] for row in c["events"].values())
    scenarios = report["warm"][0]["scenarios"]
    assert {"check_action_nothing_appended", "audit_session_after_append", "codex_session_hook.PreToolUse"} <= set(scenarios)
    assert report["idle_servers"]["after_use"]["rss_mb_max"] > 10
    assert report["machine"]["python"] and report["gates"]
    table = table_path.read_text()
    assert "Cold command hooks" in table and "Targets and CI thresholds" in table
