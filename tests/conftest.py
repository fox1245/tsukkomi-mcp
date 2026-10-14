

import os

import pytest

WINDOWS_PRIVILEGE_NOT_HELD = 1314


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Report symlink tests as skipped, not failed, where Windows denies symlink creation."""
    outcome = yield
    error = outcome.excinfo[1] if outcome.excinfo else None
    if isinstance(error, OSError) and getattr(error, "winerror", None) == WINDOWS_PRIVILEGE_NOT_HELD:
        outcome.force_exception(pytest.skip.Exception(
            "creating symlinks needs Windows Developer Mode or administrator rights", _use_item_location=True))


@pytest.fixture(autouse=True)
def explicit_test_backend(tmp_path, monkeypatch):
    # Offline default: NumPy. The sanitizer CI sets TSUKKOMI_TEST_DENSE_BACKEND=sqlite-vector together with
    # SQLITE_VECTOR_TEST_PATH (a sanitizer build of the official extension), so every engine the suite creates
    # without an explicit backend loads the native library.
    if os.environ.get("TSUKKOMI_TEST_DENSE_BACKEND") == "sqlite-vector":
        monkeypatch.setenv("SELF_DIRECT_DENSE_BACKEND", "sqlite-vector")
        monkeypatch.setenv("SELF_DIRECT_SQLITE_VECTOR_PATH", os.environ["SQLITE_VECTOR_TEST_PATH"])
    else:
        monkeypatch.setenv("SELF_DIRECT_DENSE_BACKEND", "numpy")
    monkeypatch.delenv("SELF_DIRECT_OPENROUTER_API_KEY_FILE", raising=False)
    # Never load a developer's per-user TOML while running isolated regressions.
    config_file = tmp_path / "isolated-config.toml"
    config_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(config_file))


def pytest_sessionfinish(session, exitstatus):
    """Under the ASan profile of scripts/verify_memory.py, ask LeakSanitizer for unreachable blocks while the
    interpreter is still alive. Live Python objects are reachable then, so only genuine leaks are reported (the
    exit-time check would also list everything CPython does not free at shutdown)."""
    report = os.environ.get("TSUKKOMI_LSAN_RESULT")
    if not report:
        return
    import ctypes
    import gc
    gc.collect()
    leaked = ctypes.CDLL(None).__lsan_do_recoverable_leak_check()
    with open(report, "w", encoding="utf-8") as stream:
        stream.write(str(leaked))
