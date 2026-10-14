

import pytest

@pytest.fixture(autouse=True)
def explicit_test_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("SELF_DIRECT_DENSE_BACKEND", "numpy")
    monkeypatch.delenv("SELF_DIRECT_OPENROUTER_API_KEY_FILE", raising=False)
    # Never load a developer's per-user TOML while running isolated regressions.
    config_file = tmp_path / "isolated-config.toml"
    config_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(config_file))
