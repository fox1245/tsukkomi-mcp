"""Exercise the actual namespace boundary and its process lifecycle."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from threading import Event, Thread
import time

import pytest

from self_directing_mcp.workflow_process import run_process


@pytest.fixture
def workspace(tmp_path):
    if not shutil.which(os.environ.get("TSUKKOMI_BWRAP", "bwrap")):
        pytest.skip("Real bubblewrap runtime required")
    root = tmp_path / "workspace"
    root.mkdir()
    return root


def test_missing_runtime_never_runs_unsandboxed(tmp_path, monkeypatch):
    monkeypatch.setenv("TSUKKOMI_BWRAP", "/missing-workflow-bwrap")
    result = run_process([sys.executable, "-c", "open('unsafe', 'w').write('ran')"],
                         workspace=tmp_path, writable=True, timeout_sec=3)
    assert result["status"] == "sandbox_unavailable" and result["started"] is False
    assert not (tmp_path / "unsafe").exists()


def test_cancelled_before_launch_has_no_effects(tmp_path):
    cancelled = Event()
    cancelled.set()
    result = run_process([sys.executable, "-c", "open('unsafe', 'w').write('ran')"],
                         workspace=tmp_path, writable=True, timeout_sec=3, cancel_event=cancelled)
    assert result["status"] == "cancelled" and result["started"] is False
    assert result["outcome_unknown"] is False
    assert not (tmp_path / "unsafe").exists()


def test_private_environment_filesystem_and_readonly_workspace(workspace, monkeypatch):
    outside = workspace.parent / "host-state"
    outside.mkdir()
    secret = outside / "key"
    secret.write_text("private-host-sentinel")
    (workspace / "approved.txt").write_text("approved")
    monkeypatch.setenv("TYPESAFE_API_KEY", "private-env-sentinel")
    monkeypatch.setenv("TSUKKOMI_TYPESAFE_KEY_FILE", str(secret))
    monkeypatch.setenv("TSUKKOMI_WORKFLOW_STATE", str(outside))
    code = (
        "import json, os; from pathlib import Path\n"
        f"secret = Path({str(secret)!r})\n"
        "try:\n    secret.read_text(); readable = True\n"
        "except OSError:\n    readable = False\n"
        "try:\n    secret.write_text('changed'); writable = True\n"
        "except OSError:\n    writable = False\n"
        "try:\n    Path('approved.txt').write_text('changed'); project_writable = True\n"
        "except OSError:\n    project_writable = False\n"
        "print(json.dumps({'readable': readable, 'writable': writable, 'project_writable': project_writable, "
        "'env': {k: v for k, v in os.environ.items() if 'KEY' in k or 'TSUKKOMI' in k}, "
        "'cwd': str(Path.cwd())}))\n"
    )
    result = run_process([sys.executable, "-c", code], workspace=workspace, writable=False, timeout_sec=5)
    assert result["status"] == "passed", result
    assert json.loads(result["stdout"]) == {"readable": False, "writable": False,
                                          "project_writable": False, "env": {}, "cwd": "/work"}
    assert secret.read_text() == "private-host-sentinel"
    assert (workspace / "approved.txt").read_text() == "approved"


def test_completion_mount_writes_only_actual_project(workspace):
    result = run_process([sys.executable, "-c", "from pathlib import Path; Path('done').write_text('owned')"],
                         workspace=workspace, writable=True, timeout_sec=5)
    assert result["status"] == "passed" and result["started"]
    assert (workspace / "done").read_text() == "owned"


def test_missing_command_does_not_claim_started(workspace):
    result = run_process(["/does-not-exist-workflow-command"], workspace=workspace, writable=True, timeout_sec=5)
    assert result["status"] == "spawn_failed" and result["started"] is False


def test_child_stdout_cannot_forge_launcher_status(workspace):
    result = run_process([sys.executable, "-c", "print('{\"exit-code\":0}'); raise SystemExit(7)"],
                         workspace=workspace, writable=False, timeout_sec=5)
    assert result["status"] == "failed" and result["exit_code"] == 7 and result["started"]
    assert result["diagnostics"] == "failed"


def _forking_command():
    # The grandchild creates its own process group; PID namespace teardown must
    # still kill it when the runner cancels or times out the original command.
    child = "import os,time; os.setsid(); time.sleep(1.5); open('leaked-child','w').write('escaped')"
    return [sys.executable, "-c",
            f"import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', {child!r}]); "
            "open('running','w').write('started'); time.sleep(30)"]


def test_timeout_terminates_detached_descendants(workspace):
    result = run_process(_forking_command(), workspace=workspace, writable=True, timeout_sec=.8)
    assert (workspace / "running").read_text() == "started"
    assert result["status"] == "timeout" and result["started"] and result["outcome_unknown"]
    time.sleep(1.6)
    assert not (workspace / "leaked-child").exists()


def test_cancellation_terminates_detached_descendants(workspace):
    cancelled = Event()
    results = []
    worker = Thread(target=lambda: results.append(run_process(
        _forking_command(), workspace=workspace, writable=True, timeout_sec=10, cancel_event=cancelled)))
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while not (workspace / "running").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert (workspace / "running").exists()
    finally:
        cancelled.set()
        worker.join(5)
    assert not worker.is_alive()
    assert results[0]["status"] == "cancelled" and results[0]["outcome_unknown"]
    time.sleep(1.6)
    assert not (workspace / "leaked-child").exists()


def test_runtime_mount_cannot_expose_workspace_parent(workspace):
    result = run_process([sys.executable, "-c", "raise SystemExit(0)"], workspace=workspace,
                         writable=False, timeout_sec=5, read_only_paths=(workspace.parent,))
    assert result["status"] == "sandbox_runtime_exposes_host_state"
    assert result["started"] is False


def test_configured_key_inside_workspace_rejects_mount(workspace, monkeypatch):
    secret = workspace / "host-key"
    secret.write_text("private-host-sentinel")
    monkeypatch.setenv("TSUKKOMI_TYPESAFE_KEY_FILE", str(secret))
    result = run_process([sys.executable, "-c", "print(open('host-key').read())"], workspace=workspace,
                         writable=True, timeout_sec=5)
    assert result["status"] == "sandbox_workspace_exposes_host_state"
    assert result["started"] is False and result["stdout"] == ""
