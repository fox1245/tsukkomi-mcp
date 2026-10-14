

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
    monkeypatch.setenv("SELF_DIRECT_DENSE_BACKEND", "numpy")
    monkeypatch.delenv("SELF_DIRECT_OPENROUTER_API_KEY_FILE", raising=False)
    # Never load a developer's per-user TOML while running isolated regressions.
    config_file = tmp_path / "isolated-config.toml"
    config_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(config_file))
