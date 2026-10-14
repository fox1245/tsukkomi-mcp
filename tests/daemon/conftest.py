"""Fixtures for the daemon tests, including the no-stray-process guard."""
from __future__ import annotations

import os
import sys

import pytest

import daemon_support as support

_RUNTIMES: list[support.Runtime] = []
# The daemon is Linux only in this release (``SELF_DIRECT_DAEMON=on`` is ``off`` elsewhere); these modules exercise it.
PORTABLE = {"test_daemon_protocol.py"}


# Under the ASan and TSan profiles of scripts/verify_memory.py the interpreter runs with the sanitizer runtime
# preloaded. The daemon, the relays and the hook scripts start in a clean environment on purpose (that is what keeps
# credentials out of the daemon), so their child processes would load the instrumented NeoGraph extension without the
# runtime and die at import; timings under a sanitizer mean nothing either. The tests that start real processes are
# therefore skipped there; the in-process daemon tests (the daemon's own threads and sockets) still run.
SANITIZED = any(name in os.environ.get("LD_PRELOAD", "") for name in ("libasan", "libtsan", "libubsan"))
REAL_PROCESS_MODULES = {"test_daemon_hooks.py", "test_daemon_mcp.py"}


def pytest_collection_modifyitems(config, items):
    if SANITIZED:
        skip_real = pytest.mark.skip(reason="starts real processes; the sanitizer runtime is not inherited by their "
                                            "clean environment and timings are meaningless under a sanitizer")
        for item in items:
            names = set(getattr(item, "fixturenames", ()))
            in_process_only = "inproc" in names and "world" not in names
            if os.path.basename(str(item.fspath)) in REAL_PROCESS_MODULES and not in_process_only:
                item.add_marker(skip_real)
    if sys.platform.startswith("linux"):
        return
    skip = pytest.mark.skip(reason="the resident daemon is Linux only")
    for item in items:
        name = os.path.basename(str(item.fspath))
        if "tests/daemon" in str(item.fspath).replace(os.sep, "/") and name not in PORTABLE:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_runtime_variables(monkeypatch):
    """Whatever a developer's shell exports, no test inherits a daemon flag, a limit or a real runtime directory."""
    for name in list(os.environ):
        if name.upper().startswith(("SELF_DIRECT_DAEMON", "XDG_RUNTIME_DIR")):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """A private runtime area. At the end of the test no daemon or spawned process of it may be alive: a daemon a
    test left running is stopped (so the suite stays clean) and the test FAILS for leaving it."""
    area = support.Runtime(tmp_path)
    _RUNTIMES.append(area)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(area.xdg))
    monkeypatch.setenv("HOME", str(area.home))
    yield area
    support.STARTED.update(area.pids())
    left = area.alive()
    if left:
        area.stop(timeout=5)
        left = area.kill_leftovers()
        pytest.fail(f"the test left processes running: {sorted(left)}")


@pytest.fixture
def inproc(runtime):
    """Factory for in-process daemons; they are stopped at teardown and must stop."""
    started: list[support.InProcessDaemon] = []

    def make(config=None, **kwargs) -> support.InProcessDaemon:
        daemon = support.InProcessDaemon(runtime, config, **kwargs).start()
        started.append(daemon)
        return daemon

    yield make
    for daemon in started:
        daemon.stop()


@pytest.fixture
def world(tmp_path_factory, runtime):
    return support.World(tmp_path_factory.mktemp("world"), runtime)


@pytest.fixture
def overlay_environment():
    """The daemon replaces ``os.environ`` with a per-request view; undo it so no other test sees the replacement."""
    from self_directing_mcp.daemon import envview
    envview.install()
    yield
    envview.uninstall()


def pytest_sessionfinish(session, exitstatus):
    candidates = {pid for area in _RUNTIMES for pid in area.pids()} | support.STARTED
    strays = {pid for pid in candidates if support.pid_alive(pid) and support.is_test_process(pid)}
    if strays:                       # reported, never signalled here: the per-test fixture already tried to stop them
        print(f"\nSTRAY DAEMON PROCESSES SURVIVED THE SUITE: {sorted(strays)}")
        session.exitstatus = 1
