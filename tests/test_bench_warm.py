"""scripts/bench_warm.py runs end to end on a tiny synthetic session and tells deferred commits from synchronous ones."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CELLS = ("claude:Stop:0", "claude:Stop:2048", "agy:PreToolUse:0", "agy:PreToolUse:2048")


def run(tmp_path, *extra):
    report = tmp_path / "report.json"
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "bench_warm.py"), "--runs", "3", "--session-mb", "0.05",
         "--only", "claude:Stop,agy:PreToolUse", "--append-bytes", "0,2048", "--tmpdir", str(tmp_path),
         "--out", str(report), *extra],
        capture_output=True, text=True, timeout=240, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(report.read_text()), done.stdout


def test_reports_every_requested_cell_with_its_commits(tmp_path):
    report, text = run(tmp_path)
    assert set(report["cells"]) == set(CELLS)
    for cell in report["cells"].values():
        assert 0 < cell["min"] <= cell["p50"] <= cell["p95"] and cell["commits_per_request"] >= 0
    assert "SELF_DIRECT_AUDIT_INCREMENTAL_MIN_EVENTS=" in text and "load average" in text
    assert report["defer"] is False


def test_a_deferring_engine_commits_less_than_a_synchronous_one(tmp_path):
    (tmp_path / "plain").mkdir()
    (tmp_path / "deferring").mkdir()
    plain, _ = run(tmp_path / "plain")
    deferring, _ = run(tmp_path / "deferring", "--defer")
    assert deferring["defer"] is True
    assert deferring["cells"]["claude:Stop:0"]["commits_per_request"] == 0
    assert plain["cells"]["claude:Stop:0"]["commits_per_request"] > 0         # the verification stamps
    for cell in CELLS:
        assert deferring["cells"][cell]["commits_per_request"] <= plain["cells"][cell]["commits_per_request"], cell
