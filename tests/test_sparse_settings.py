"""SELF_DIRECT_SPARSE_FORMAT: default, validation (an error, never a silent default), the server and the hook reader agreeing,
and the setting reaching the engine. The two readers are compared on many inputs in tests/test_hook_settings_parity.py."""
from __future__ import annotations

import os

import pytest

from self_directing_mcp import settings_core
from self_directing_mcp.config import Settings, hook_settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index.sparse import compact_support
from self_directing_mcp.settings_core import load_hook_settings


@pytest.fixture
def clean(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.upper().startswith(("SELF_DIRECT_", "OPENROUTER_")):
            monkeypatch.delenv(name, raising=False)
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(config))
    monkeypatch.chdir(tmp_path)
    return monkeypatch


def test_the_default_is_auto_in_both_readers(clean):
    for settings in (Settings(_env_file=None), hook_settings(), load_hook_settings()):
        assert settings.sparse_format == "auto"
    assert settings_core.DEFAULT_SPARSE_FORMAT == "auto" and settings_core.SPARSE_FORMATS == ("auto", "legacy", "compact")


@pytest.mark.parametrize("value", ["auto", "legacy", "compact"])
def test_each_format_is_read_by_both_readers(clean, value):
    clean.setenv("SELF_DIRECT_SPARSE_FORMAT", value)
    assert Settings(_env_file=None).sparse_format == hook_settings().sparse_format == load_hook_settings().sparse_format == value


@pytest.mark.parametrize("value", ["Auto", "COMPACT", "", "v2", "sparse-v2", "true", "0", " legacy", "compact\n"])
def test_an_unknown_format_is_an_error_in_both_readers_not_a_default(clean, value):
    clean.setenv("SELF_DIRECT_SPARSE_FORMAT", value)
    with pytest.raises(ValueError):
        Settings(_env_file=None)
    with pytest.raises(ValueError):
        load_hook_settings()


SUPPORTED = compact_support() is None


@pytest.mark.parametrize("value,expected", [
    ("legacy", "legacy"),
    pytest.param("compact", "compact", marks=pytest.mark.skipif(not SUPPORTED, reason="this SQLite cannot use the compact format")),
    ("auto", "compact" if SUPPORTED else "legacy")])
def test_the_setting_decides_which_file_the_engine_opens(clean, tmp_path, value, expected):
    engine = SelfDirectEngine(Settings(index_dir=tmp_path / "index", sparse_format=value, use_fake_embedder=True,
                                       dense_backend="numpy", _env_file=None))
    try:
        engine.ensure_ready()
        assert engine.sparse.format == expected
        assert engine.sparse.db_path.name == ("sparse-v2.sqlite" if expected == "compact" else "sparse.sqlite")
        assert [p.name for p in (tmp_path / "index").glob("sparse*.sqlite")] == [engine.sparse.db_path.name]   # only the one in use
    finally:
        engine.close()


def test_the_hook_reader_does_not_import_the_sparse_index_or_the_maintenance_command():
    """The setting is plain data in ``settings_core``: reading it must not pull the index code into a hook process
    (tests/test_hook_import_budget.py checks the whole hook path)."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    code = ("import sys; from self_directing_mcp import settings_core; settings_core.load_hook_settings(); "
            "bad = [m for m in ('self_directing_mcp.index.sparse', 'self_directing_mcp.maintenance', 'self_directing_mcp.engine', 'sqlite3') "
            "if m in sys.modules]; print(bad)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root, timeout=60,
                          env={**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
    assert done.returncode == 0, done.stderr[-1000:]
    assert done.stdout.strip() == "[]", done.stdout
