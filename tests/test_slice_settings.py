"""SELF_DIRECT_SYNC_SLICING and SELF_DIRECT_SYNC_SLICE_BYTES: defaults, validation (errors, never silent defaults), the
server and the hook reader agreeing, and the settings reaching the sync (roundtable 8, P8). The two readers are compared
on hundreds of inputs in tests/test_hook_settings_parity.py; this is the part that is specific to slicing."""
from __future__ import annotations

import time

import pytest

from self_directing_mcp import settings_core
from self_directing_mcp.config import Settings, hook_settings
from self_directing_mcp.index.ingest import SlicePolicy
from self_directing_mcp.settings_core import load_hook_settings
from slicing_support import World

MIB = 1024 * 1024


@pytest.fixture
def clean(monkeypatch, tmp_path):
    import os
    for name in list(os.environ):
        if name.upper().startswith(("SELF_DIRECT_", "OPENROUTER_")):
            monkeypatch.delenv(name, raising=False)
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(config))
    monkeypatch.chdir(tmp_path)
    return monkeypatch


def test_the_defaults_are_auto_and_64_mib(clean):
    for settings in (Settings(_env_file=None), hook_settings(), load_hook_settings()):
        assert settings.sync_slicing == "auto" and settings.sync_slice_bytes == 64 * MIB
    assert settings_core.DEFAULT_SYNC_SLICING == "auto" and settings_core.DEFAULT_SYNC_SLICE_BYTES == 64 * MIB
    policy = SlicePolicy.from_settings(Settings(_env_file=None))
    assert (policy.mode, policy.cap_bytes, policy.min_records, policy.min_bytes) == ("auto", 64 * MIB, 256, MIB)
    assert policy.max_records is None and policy.max_bytes is None


@pytest.mark.parametrize("value,expected", [("auto", "auto"), ("off", "off")])
def test_the_mode_is_read_by_both_readers(clean, value, expected):
    clean.setenv("SELF_DIRECT_SYNC_SLICING", value)
    assert Settings(_env_file=None).sync_slicing == hook_settings().sync_slicing == load_hook_settings().sync_slicing == expected
    assert SlicePolicy.from_settings(load_hook_settings()).mode == expected


@pytest.mark.parametrize("value", ["on", "Auto", "OFF", "", "slice", "true", "0", " off"])
def test_an_unknown_mode_is_an_error_in_both_readers_not_a_default(clean, value):
    clean.setenv("SELF_DIRECT_SYNC_SLICING", value)
    with pytest.raises(ValueError):
        Settings(_env_file=None)
    with pytest.raises(ValueError):
        load_hook_settings()


@pytest.mark.parametrize("value,expected", [("4096", 4096), ("1", 1), (" 1_000 ", 1000), ("4.0", 4), ("1073741824", 1 << 30)])
def test_the_cap_is_a_positive_integer(clean, value, expected):
    clean.setenv("SELF_DIRECT_SYNC_SLICE_BYTES", value)
    assert Settings(_env_file=None).sync_slice_bytes == load_hook_settings().sync_slice_bytes == expected
    assert SlicePolicy.from_settings(Settings(_env_file=None)).cap_bytes == expected


@pytest.mark.parametrize("value", ["0", "-1", "abc", "", "4.5", "1e3", "64MiB", "0x10"])
def test_a_bad_cap_is_an_error_in_both_readers_not_a_default(clean, value):
    clean.setenv("SELF_DIRECT_SYNC_SLICE_BYTES", value)
    with pytest.raises(ValueError):
        Settings(_env_file=None)
    with pytest.raises(ValueError):
        load_hook_settings()


def test_a_policy_cannot_be_built_from_nonsense():
    for options in ({"mode": "on"}, {"cap_bytes": 0}, {"cap_bytes": -5}):
        with pytest.raises(ValueError):
            SlicePolicy(**options)


def test_the_settings_reach_the_sync(tmp_path, monkeypatch):
    """No policy injected: the engine builds it from its settings, a 4,000-byte cap cuts the capture, ``off`` does not."""
    world = World(tmp_path, monkeypatch, "codex", seed=4, count=200, partial=False)
    try:
        size = world.path.stat().st_size
        engine = world.engine("capped", sync_slice_bytes=4000)
        assert engine.slice_policy is None and engine._slice_policy().cap_bytes == 4000
        info = engine.sync_session(**world.sync_args(), derived=False, _deadline=time.monotonic() + 60)
        assert info["sync"]["state"] == "partial" and info["sync"]["stopped_by"] == "slice_cap"
        assert 0 < info["sync"]["indexed_through"] <= 4000 + 400 and info["sync"]["target_bytes"] == size
        off = world.engine("off", sync_slicing="off", sync_slice_bytes=4000)
        done = off.sync_session(**world.sync_args(), derived=False, _deadline=time.monotonic() + 60)
        assert "sync" not in done and done["coverage"]["byte_offset"] == size and done["coverage"]["complete"] is False
    finally:
        world.close()
