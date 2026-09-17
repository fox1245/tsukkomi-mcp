

import pytest

@pytest.fixture(autouse=True)
def explicit_test_backend(monkeypatch):
    monkeypatch.setenv("SELF_DIRECT_DENSE_BACKEND", "numpy")
    monkeypatch.delenv("SELF_DIRECT_OPENROUTER_API_KEY_FILE", raising=False)
