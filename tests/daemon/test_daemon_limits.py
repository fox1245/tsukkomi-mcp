"""A daemon's limits belong to the daemon: they come from its own configuration, not from the client that started it.

Two clients start the same daemon on one machine: the hook commands (which may export ``SELF_DIRECT_DAEMON_*`` inline) and the
MCP relay (whose environment is the host's registration of the server). Before the ``[daemon]`` table of the user
configuration existed, the limit of a daemon was whatever the client that happened to find none running had in its environment:
the default of 400 MB for one, a chosen value for the other, so a daemon left on its memory limit every minute or two whenever the
relay started it. The environment still wins for whoever sets it on purpose (tests, one-off runs); the file is what every
client's daemon has in common.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from self_directing_mcp import settings_core
from self_directing_mcp.daemon import client, core, rundir
import daemon_support as support


def write_config(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_the_daemon_table_of_the_user_configuration_sets_the_limits(tmp_path, monkeypatch):
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(write_config(tmp_path / "c.toml", (
        "[daemon]\nmax_rss_mb = 1500\nidle_exit_sec = 60.5\nsession_bound = 4\ndefer_derived = \"off\"\n"))))
    config = core.DaemonConfig.from_environment()
    assert (config.max_rss_mb, config.idle_exit_sec, config.session_bound, config.defer_derived) == (1500, 60.5, 4, False)


def test_without_a_table_the_defaults_stay_and_an_explicit_environment_never_reads_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(write_config(tmp_path / "c.toml", "[daemon]\nmax_rss_mb = 1500\n")))
    assert core.DaemonConfig.from_environment({}).max_rss_mb == 400           # a mapping given by the caller is the whole input
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(write_config(tmp_path / "d.toml", "[paths]\n")))
    assert core.DaemonConfig.from_environment().max_rss_mb == 400
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(tmp_path / "missing.toml"))
    assert core.DaemonConfig.from_environment().max_rss_mb == 400              # no file, no table


def test_the_environment_beats_the_file_and_the_file_beats_the_default(tmp_path):
    path = write_config(tmp_path / "c.toml", "[daemon]\nmax_rss_mb = 1500\nsession_bound = 4\n")
    config = core.DaemonConfig.from_environment({"SELF_DIRECT_CONFIG_FILE": str(path),
                                                 "SELF_DIRECT_DAEMON_MAX_RSS_MB": "256"})
    assert (config.max_rss_mb, config.session_bound, config.idle_exit_sec) == (256, 4, 900)


@pytest.mark.parametrize("body", [
    "[daemon]\nmax_rss_mb = 0\n", "[daemon]\nmax_rss_mb = -5\n", "[daemon]\nmax_rss_mb = \"2048\"\n",
    "[daemon]\nmax_rss_mb = true\n", "[daemon]\nmax_rss_mb = inf\n", "[daemon]\nmax_rss_mb = nan\n",
    "[daemon]\nsession_bound = 2.5\n", "[daemon]\ndefer_derived = \"yes\"\n", "[daemon]\nmax_rss = 2048\n",
    "daemon = 5\n", "[daemon]\nmax_rss_mb = 1e999\n",
])
def test_a_malformed_daemon_table_is_an_error_not_a_default(tmp_path, body):
    path = write_config(tmp_path / "c.toml", body)
    with pytest.raises(ValueError):
        core.DaemonConfig.from_environment({"SELF_DIRECT_CONFIG_FILE": str(path)})


def test_a_daemon_table_does_not_stop_the_command_hooks_from_reading_their_settings(tmp_path, monkeypatch):
    path = write_config(tmp_path / "c.toml", f"[paths]\nindex_dir = {json.dumps(str(tmp_path / 'idx'))}\n"
                                              "[daemon]\nmax_rss_mb = 1500\n")
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(path))
    assert str(settings_core.load_hook_settings().index_dir) == str(tmp_path / "idx")


def test_a_daemon_started_by_a_client_with_no_limit_in_its_environment_runs_on_the_configured_limit(runtime):
    """The relay's environment carries no ``SELF_DIRECT_DAEMON_*``: its daemon used to get the default of 400 MB."""
    write_config(runtime.home / ".config" / "tsukkomi-mcp" / "config.toml", "[daemon]\nmax_rss_mb = 1500\n")
    relay_like = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.environ.get("PYTHONPATH", ""), **runtime.environ()}
    hook_like = {**relay_like, "SELF_DIRECT_DAEMON_MAX_RSS_MB": "1500"}
    limits = []
    for environ in (relay_like, hook_like):
        process = subprocess.Popen(client.spawn_command(environ, support.PY), env=client.daemon_environment(environ),
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=True, cwd=str(runtime.root))
        runtime.extra_pids.add(process.pid)
        support.STARTED.add(process.pid)
        deadline = time.monotonic() + 30
        status = None
        while status is None and time.monotonic() < deadline:
            status = support.daemon_status(runtime) if (runtime.dir / rundir.SOCKET_NAME).exists() else None
            time.sleep(0.05)
        assert status is not None and status["pid"] == process.pid, status
        limits.append(status["max_rss_mb"])
        runtime.stop()
        assert process.wait(timeout=20) == 0
    assert limits == [1500, 1500]


def test_the_started_event_names_the_limit_the_daemon_runs_on(runtime, inproc):
    from self_directing_mcp.daemon import core as daemon_core
    daemon = inproc(daemon_core.DaemonConfig(max_rss_mb=777))
    started = [row for row in support.daemon_log(runtime) if row.get("ev") == "started"]
    assert started and started[-1]["max_rss_mb"] == 777
    assert daemon.daemon.snapshot()["max_rss_mb"] == 777
