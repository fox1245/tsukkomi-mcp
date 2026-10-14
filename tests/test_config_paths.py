"""User-selected path sources, consumer boundaries, and fail-closed TOML loading."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import pytest
from pydantic_settings import SettingsError

from self_directing_mcp import config
from self_directing_mcp.config import Settings, get_settings
from self_directing_mcp.engine import SelfDirectEngine


PATH_FIELDS = (
    "index_dir", "codex_sessions_dir", "omp_sessions_dir",
    "grokbot_transcripts_dir", "agy_app_data_dirs", "contracts_path",
    "sqlite_vector_path", "openrouter_api_key_file",
)
ARRAY_FIELDS = {"grokbot_transcripts_dir", "agy_app_data_dirs"}
CONFIG_ERRORS = (ValueError, OSError, SettingsError)
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(autouse=True)
def isolated_path_sources(explicit_test_backend, tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("SELF_DIRECT_") and name != "SELF_DIRECT_CONFIG_FILE":
            monkeypatch.delenv(name)
    monkeypatch.delenv("CODEX_SESSIONS_DIR", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "_repo_root", lambda: tmp_path)


def _select(monkeypatch, path: Path, **paths) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[paths]\n" + "".join(
        f"{name} = {json.dumps(value)}\n" for name, value in paths.items()
    ), encoding="utf-8")
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(path))
    return path


def _resolved(settings: Settings, name: str):
    resolvers = {
        "codex_sessions_dir": settings.resolve_sessions_dir,
        "omp_sessions_dir": settings.resolve_omp_sessions_dir,
        "grokbot_transcripts_dir": settings.resolve_grokbot_transcripts_dirs,
        "agy_app_data_dirs": settings.resolve_agy_app_data_dirs,
        "contracts_path": settings.resolve_contracts_path,
    }
    return resolvers[name]() if name in resolvers else getattr(settings, name)


def _home(monkeypatch, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(path))
    monkeypatch.setenv("USERPROFILE", str(path))


def test_optional_default_discovery_and_explicit_alternate(tmp_path, monkeypatch):
    home = tmp_path / "owner"
    _home(monkeypatch, home)
    monkeypatch.delenv("SELF_DIRECT_CONFIG_FILE")
    default = home / ".config" / "tsukkomi-mcp" / "config.toml"
    assert config.get_config_path() == default
    assert Settings(_env_file=None).index_dir == home / ".self_direct_index"
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True))
    try:
        engine.ensure_ready()
        assert (home / ".self_direct_index" / "meta.sqlite").is_file()
        assert not (tmp_path / ".self_direct_index").exists()
    finally:
        engine.close()
    _select(monkeypatch, default, index_dir="default-cache")
    monkeypatch.delenv("SELF_DIRECT_CONFIG_FILE")
    assert Settings(_env_file=None).index_dir == default.parent / "default-cache"
    alternate = _select(monkeypatch, home / "alternate.toml", index_dir="alternate-cache")
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", "~/alternate.toml")
    assert config.get_config_path() == alternate
    assert Settings(_env_file=None).index_dir == home / "alternate-cache"


def test_relative_paths_use_canonical_toml_parent_not_symlink_parent(tmp_path, monkeypatch):
    selected = _select(monkeypatch, tmp_path / "private" / "settings.toml", index_dir="cache")
    links = tmp_path / "links"
    links.mkdir()
    alias = links / "selected.toml"
    alias.symlink_to(selected)
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(alias))
    assert config.get_config_path() == selected
    assert Settings(_env_file=None).index_dir == selected.parent / "cache"


@pytest.mark.parametrize("name", PATH_FIELDS)
def test_toml_paths_remain_config_relative_after_cwd_changes(tmp_path, monkeypatch, name):
    relative = "../chosen-data"
    value = [relative] if name in ARRAY_FIELDS else relative
    selected = _select(monkeypatch, tmp_path / "configuration" / "selected.toml", **{name: value})
    expected = tmp_path / "chosen-data"
    if name in ARRAY_FIELDS:
        expected = [expected]
    before = Settings(_env_file=None)
    other = tmp_path / "unrelated-working-directory"
    other.mkdir()
    monkeypatch.chdir(other)
    assert _resolved(before, name) == expected
    assert _resolved(Settings(_env_file=None), name) == expected
    assert config.get_config_path() == selected


@pytest.mark.parametrize("name", PATH_FIELDS)
def test_toml_paths_expand_tilde(tmp_path, monkeypatch, name):
    home = tmp_path / "owner"
    _home(monkeypatch, home)
    value = ["~/chosen-data"] if name in ARRAY_FIELDS else "~/chosen-data"
    _select(monkeypatch, tmp_path / "configuration" / "selected.toml", **{name: value})
    expected = home / "chosen-data"
    assert _resolved(Settings(_env_file=None), name) == ([expected] if name in ARRAY_FIELDS else expected)


def test_grok_string_uses_existing_multi_root_parser(tmp_path, monkeypatch):
    _select(monkeypatch, tmp_path / "settings.toml", grokbot_transcripts_dir="one;two,three")
    assert Settings(_env_file=None).resolve_grokbot_transcripts_dirs() == [
        tmp_path / name for name in ("one", "two", "three")
    ]


def test_grok_array_preserves_delimiters_in_individual_paths(tmp_path, monkeypatch):
    roots = ["one, literal", "two; literal", "three: literal"]
    _select(monkeypatch, tmp_path / "settings.toml", grokbot_transcripts_dir=roots)
    assert Settings(_env_file=None).resolve_grokbot_transcripts_dirs() == [tmp_path / root for root in roots]


@pytest.mark.parametrize("name", sorted(ARRAY_FIELDS))
def test_empty_toml_arrays_disable_source_roots(tmp_path, monkeypatch, name):
    _select(monkeypatch, tmp_path / "settings.toml", **{name: []})
    assert _resolved(Settings(_env_file=None), name) == []


def test_toml_does_not_interpolate_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIVATE_ROOT", str(tmp_path / "unexpected"))
    _select(monkeypatch, tmp_path / "settings.toml", index_dir="${PRIVATE_ROOT}/cache")
    assert Settings(_env_file=None).index_dir == tmp_path / "${PRIVATE_ROOT}" / "cache"


@pytest.mark.parametrize("name", PATH_FIELDS)
def test_explicit_then_environment_then_toml_then_dotenv(tmp_path, monkeypatch, name):
    roots = {source: tmp_path / source for source in ("explicit", "environment", "toml", "dotenv")}
    def value(source):
        path = str(roots[source])
        return [path] if name in ARRAY_FIELDS else path
    _select(monkeypatch, tmp_path / "settings.toml", **{name: value("toml")})
    env_name = "SELF_DIRECT_" + name.upper()
    dotenv_value = json.dumps(value("dotenv")) if name == "agy_app_data_dirs" else str(roots["dotenv"])
    (tmp_path / ".env").write_text(f"{env_name}='{dotenv_value}'\n", encoding="utf-8")
    env_value = json.dumps(value("environment")) if name == "agy_app_data_dirs" else str(roots["environment"])
    monkeypatch.setenv(env_name, env_value)
    explicit = Settings(**{name: value("explicit")})
    def expected(source):
        return [roots[source]] if name in ARRAY_FIELDS else roots[source]
    assert _resolved(explicit, name) == expected("explicit")
    assert _resolved(Settings(), name) == expected("environment")
    monkeypatch.delenv(env_name)
    assert _resolved(explicit, name) == expected("explicit")
    assert _resolved(Settings(), name) == expected("toml")
    _select(monkeypatch, tmp_path / "settings.toml")
    assert _resolved(Settings(), name) == expected("dotenv")


def test_legacy_codex_alias_precedence_and_settings_snapshot(tmp_path, monkeypatch):
    _select(monkeypatch, tmp_path / "settings.toml", codex_sessions_dir="toml")
    (tmp_path / ".env").write_text(f"SELF_DIRECT_CODEX_SESSIONS_DIR={tmp_path / 'dotenv'}\n")
    monkeypatch.setenv("CODEX_SESSIONS_DIR", str(tmp_path / "legacy"))
    legacy = Settings()
    assert legacy.resolve_sessions_dir() == tmp_path / "legacy"
    monkeypatch.setenv("SELF_DIRECT_CODEX_SESSIONS_DIR", str(tmp_path / "prefixed"))
    assert Settings().resolve_sessions_dir() == tmp_path / "prefixed"
    explicit = Settings(codex_sessions_dir=tmp_path / "explicit")
    assert explicit.resolve_sessions_dir() == tmp_path / "explicit"
    assert legacy.resolve_sessions_dir() == tmp_path / "legacy"
    monkeypatch.delenv("SELF_DIRECT_CODEX_SESSIONS_DIR")
    monkeypatch.delenv("CODEX_SESSIONS_DIR")
    assert Settings().resolve_sessions_dir() == tmp_path / "toml"
    _select(monkeypatch, tmp_path / "settings.toml")
    assert Settings().resolve_sessions_dir() == tmp_path / "dotenv"


def test_explicit_provider_is_not_overridden_by_live_environment(monkeypatch):
    monkeypatch.setenv("SELF_DIRECT_SESSION_PROVIDER", "codex")
    settings = Settings(_env_file=None, session_provider="omp")
    monkeypatch.setenv("SELF_DIRECT_SESSION_PROVIDER", "agy")
    assert settings.resolve_session_provider() == "omp"


def test_dotenv_cannot_select_a_toml_file(tmp_path, monkeypatch):
    other = tmp_path / "unselected.toml"
    other.write_text('[paths]\nindex_dir="unselected-cache"\n')
    home = tmp_path / "owner"
    _home(monkeypatch, home)
    monkeypatch.delenv("SELF_DIRECT_CONFIG_FILE")
    (tmp_path / ".env").write_text(f"SELF_DIRECT_CONFIG_FILE={other}\n")
    assert Settings().index_dir == home / ".self_direct_index"
    assert config.get_config_path() == home / ".config" / "tsukkomi-mcp" / "config.toml"


@pytest.mark.parametrize("document", [
    '[paths\nindex_dir="cache"',
    '[unexpected]\nindex_dir="cache"',
    '[paths]\nunknown_path="cache"',
    'paths="cache"',
    '[paths]\nindex_dir=42',
    '[paths]\nindex_dir=true',
    '[paths]\nindex_dir=[]',
    '[paths]\nindex_dir=""',
    '[paths]\nindex_dir="   "',
    '[paths]\ngrokbot_transcripts_dir=42',
    '[paths]\ngrokbot_transcripts_dir=["valid", 42]',
    '[paths]\ngrokbot_transcripts_dir=[""]',
    '[paths]\nagy_app_data_dirs="not-an-array"',
    '[paths]\nagy_app_data_dirs=["valid", false]',
    '[paths]\nagy_app_data_dirs=[""]',
    '[paths]\nopenrouter_api_key="credential-value-not-a-path"',
    'OPENROUTER_API_KEY="credential-value-not-a-path"',
])
def test_invalid_selected_toml_fails_even_with_explicit_path_override(tmp_path, monkeypatch, document):
    selected = tmp_path / "invalid.toml"
    selected.write_text(document, encoding="utf-8")
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(selected))
    with pytest.raises(CONFIG_ERRORS):
        Settings(_env_file=None, index_dir=tmp_path / "explicit")


@pytest.mark.parametrize("selection", ["missing.toml", "directory"])
def test_explicit_selector_must_name_an_existing_file(tmp_path, monkeypatch, selection):
    path = tmp_path / selection
    if selection == "directory":
        path.mkdir()
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(path))
    with pytest.raises(CONFIG_ERRORS):
        Settings(_env_file=None)


def test_malformed_default_file_does_not_silently_fall_back(tmp_path, monkeypatch):
    home = tmp_path / "owner"
    _home(monkeypatch, home)
    default = home / ".config" / "tsukkomi-mcp" / "config.toml"
    _select(monkeypatch, default)
    default.write_text('[paths]\nindex_dir = [broken')
    monkeypatch.delenv("SELF_DIRECT_CONFIG_FILE")
    with pytest.raises(CONFIG_ERRORS):
        Settings(_env_file=None)


def test_toml_key_file_is_authoritative_without_exporting_its_value(tmp_path, monkeypatch):
    key_file = tmp_path / "credentials" / "authorized.env"
    key_file.parent.mkdir()
    key_file.write_text("OPENROUTER_API_KEY=toml-file-key\n")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=ordinary-dotenv-key\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-key")
    _select(monkeypatch, tmp_path / "configuration" / "settings.toml",
            openrouter_api_key_file="../credentials/authorized.env")
    assert get_settings().resolve_api_key() == "toml-file-key"
    assert os.environ["OPENROUTER_API_KEY"] == "environment-key"
    key_file.unlink()
    with pytest.raises(ValueError):
        get_settings().resolve_api_key()


@pytest.mark.parametrize("provider", ["codex", "omp", "grokbot", "agy"])
def test_engine_uses_toml_source_index_and_rule_paths(tmp_path, monkeypatch, provider):
    sid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    root = tmp_path / "configuration" / "approved-source"
    if provider == "codex":
        source = FIXTURES / "sample_session" / "2026" / "09" / "09" / f"rollout-2026-09-09T01-00-00-{sid}.jsonl"
        relative = source.relative_to(FIXTURES / "sample_session")
    elif provider == "grokbot":
        sid = "15c5c6e5-f1db-4492-add5-4d6d2ab5600c"
        relative = Path(sid) / f"{sid}.jsonl"
        source = FIXTURES / "sample_grokbot_session" / relative
    elif provider == "agy":
        sid = "11111111-1111-4111-8111-111111111111"
        relative = Path("brain") / sid / ".system_generated" / "logs" / "transcript.jsonl"
        source = FIXTURES / "sample_agy_session" / "transcript.jsonl"
    else:
        sid = "toml-omp-session"
        relative = Path("session.jsonl")
        source = None
    session = root / relative
    session.parent.mkdir(parents=True)
    if source:
        shutil.copyfile(source, session)
    else:
        records = [{"type": "session", "id": sid},
                   {"type": "message", "message": {"role": "user", "content": [
                       {"type": "text", "text": "selected-root-evidence"}]}}]
        session.write_text("".join(json.dumps(record) + "\n" for record in records))
    root_field = {"codex": "codex_sessions_dir", "omp": "omp_sessions_dir",
                  "grokbot": "grokbot_transcripts_dir", "agy": "agy_app_data_dirs"}[provider]
    root_value = ["approved-source"] if root_field in ARRAY_FIELDS else "approved-source"
    selected = _select(monkeypatch, tmp_path / "configuration" / "settings.toml",
                       index_dir="selected-index", contracts_path="rules/selected.json",
                       **{root_field: root_value})
    rules = selected.parent / "rules" / "selected.json"
    rules.parent.mkdir()
    rules.write_text(json.dumps({"contracts": [{"id": "selected-rule", "type": "must_not",
                     "scope": "tool_call", "regex": "toml-forbidden-command", "provider": provider}]}))
    elsewhere = tmp_path / "other-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    engine = SelfDirectEngine(get_settings(_env_file=None, local_only=True, session_provider=provider))
    try:
        sync = engine.sync_session(session_id=sid, provider=provider, embed=False)
        assert sync["ok"] is True, sync
        assert {chunk.provider for chunk in engine.store.list_chunks(sid, provider)} == {provider}
        result = engine.check_action(sid, {"tool_name": "shell", "arguments": {
            "command": "toml-forbidden-command"}}, provider=provider)
        assert result["verdict"] == "violation", result
        assert "selected-rule" in {finding["contract_id"] for finding in result["findings"]}
        assert (selected.parent / "selected-index" / "meta.sqlite").is_file()
        assert not (elsewhere / "selected-index").exists()
        outside = tmp_path / "outside-source" / relative
        outside.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(session, outside)
        rejected = engine.sync_session(session_id=sid, provider=provider, path=str(outside), embed=False)
        assert rejected["error"] == "path_traversal", rejected
        engine.upsert_contracts([{"id": "persisted-rule", "type": "must_not", "regex": "other-command"}])
        assert "persisted-rule" in {rule["id"] for rule in json.loads(rules.read_text())["contracts"]}
    finally:
        engine.close()
