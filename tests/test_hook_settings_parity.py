"""The command hooks' light settings reader and ``config.hook_settings()`` must agree on every input.

``hook_settings()`` (pydantic-settings) is the reference; ``settings_core.load_hook_settings()`` is what the hook
processes actually run, because importing pydantic-settings costs about 30 ms per tool call. Both must read the same
sources in the same order (environment, ``CODEX_SESSIONS_DIR``, the user TOML ``[paths]``, defaults; never a working
directory ``.env``) and both must reject exactly the same inputs: an invalid value raises ``ValueError`` in each,
which the hooks turn into a visible fail-open warning.

Synthetic values only; nothing here reads a real user configuration.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import typing

import pytest

from self_directing_mcp import settings_core
from self_directing_mcp.config import Settings, hook_settings
from self_directing_mcp.settings_core import FIELDS, load_hook_settings

ENV_PREFIXES = ("SELF_DIRECT_", "CODEX_", "CLAUDE_", "OPENROUTER_")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A clean environment, an isolated home, a decoy project ``.env`` and an empty user TOML."""
    for name in list(os.environ):
        if name.upper().startswith(ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(config))
    # The server would honour this file; hooks must not (the audited agent can write it).
    (project / ".env").write_text("SELF_DIRECT_INDEX_DIR=/decoy\nSELF_DIRECT_CLAUDE_ENFORCEMENT=advisory\n"
                                  "SELF_DIRECT_LOCAL_ONLY=true\nOPENROUTER_API_KEY=decoy-key\n", encoding="utf-8")
    monkeypatch.chdir(project)
    return {"tmp": tmp_path, "home": home, "config": config, "monkeypatch": monkeypatch}


def outcome(factory):
    try:
        return ("ok", factory().model_dump())
    except ValueError as exc:                       # pydantic's ValidationError and our errors are ValueErrors
        return ("error", None)


def agree(case: str = "") -> None:
    reference, light = outcome(hook_settings), outcome(load_hook_settings)
    assert light == reference, f"{case}\n light:     {light}\n reference: {reference}"


def put(world, **env):
    for name, value in env.items():
        world["monkeypatch"].setenv(name, value)


def toml(world, text):
    world["config"].write_text(text, encoding="utf-8")


# --- defaults and the field table ---------------------------------------------------------------------------------

def test_defaults_agree(world):
    agree("defaults")
    assert outcome(load_hook_settings)[0] == "ok"


def test_the_field_table_covers_exactly_the_settings_model():
    assert [spec.name for spec in FIELDS] == list(Settings.model_fields)


def test_the_shadow_comparison_settings_default_to_the_adaptive_schedule(world):
    for settings in (hook_settings(), load_hook_settings()):
        assert settings.audit_shadow_every == "adaptive" and settings.audit_shadow_idle_sec == 2.0


@pytest.mark.parametrize("text,expected", [("adaptive", "adaptive"), ("  ADAPTIVE\t", "adaptive"), ("0", 0), ("1", 1),
                                            ("16", 16), (" 4 ", 4), ("1_0", 10)])
def test_the_shadow_every_spellings_mean_the_same_to_both_readers(world, text, expected):
    put(world, SELF_DIRECT_AUDIT_SHADOW_EVERY=text)
    agree(text)
    assert hook_settings().audit_shadow_every == expected == load_hook_settings().audit_shadow_every


@pytest.mark.parametrize("text", ["-1", "fixed", "", "1.5", "adaptive 1"])
def test_the_shadow_every_values_both_readers_reject(world, text):
    put(world, SELF_DIRECT_AUDIT_SHADOW_EVERY=text)
    assert outcome(hook_settings)[0] == "error" and outcome(load_hook_settings)[0] == "error"


def test_literal_choices_match_the_model():
    for name, choices in (("agy_enforcement", settings_core.ENFORCEMENT_MODES),
                          ("claude_enforcement", settings_core.ENFORCEMENT_MODES),
                          ("session_provider", settings_core.SESSION_PROVIDERS),
                          ("dense_backend", settings_core.DENSE_BACKENDS),
                          ("graph_extractor", settings_core.GRAPH_EXTRACTORS),
                          ("daemon", settings_core.DAEMON_MODES),
                          ("sqlite_journal", settings_core.SQLITE_JOURNAL_MODES),
                          ("sync_slicing", settings_core.SYNC_SLICING_MODES)):
        assert typing.get_args(Settings.model_fields[name].annotation) == choices, name


def test_the_graph_extractor_setting_is_read_the_same_way_and_unknown_values_are_errors(world):
    assert hook_settings().graph_extractor == load_hook_settings().graph_extractor == "openrouter"
    put(world, SELF_DIRECT_GRAPH_EXTRACTOR="host")
    assert hook_settings().graph_extractor == load_hook_settings().graph_extractor == "host"
    world["monkeypatch"].delenv("SELF_DIRECT_GRAPH_EXTRACTOR")
    put(world, self_direct_graph_extractor="host")                  # names are case-insensitive, like every other setting
    assert hook_settings().graph_extractor == load_hook_settings().graph_extractor == "host"
    world["monkeypatch"].delenv("self_direct_graph_extractor")
    assert hook_settings().graph_extractor == load_hook_settings().graph_extractor == "openrouter"
    for bad in ("hosted", "Host", "openrouter,host", ""):
        put(world, SELF_DIRECT_GRAPH_EXTRACTOR=bad)
        assert outcome(hook_settings)[0] == "error" and outcome(load_hook_settings)[0] == "error", bad
        with pytest.raises(ValueError, match="openrouter.*host"):
            Settings(_env_file=None)


def test_the_pgvector_settings_default_off_and_read_the_same_way(world):
    for settings in (hook_settings(), load_hook_settings()):
        assert settings.dense_backend == "sqlite-vector" and settings.pgvector_dsn_file is None
        assert settings.pgvector_schema == "tsukkomi" and settings.pgvector_hnsw is False
        assert settings.resolve_pgvector_dsn() == ""
    put(world, SELF_DIRECT_DENSE_BACKEND="pgvector", SELF_DIRECT_PGVECTOR_SCHEMA="Archive_2", SELF_DIRECT_PGVECTOR_HNSW="true",
        SELF_DIRECT_PGVECTOR_DSN_FILE=str(world["tmp"] / "dsn.txt"))
    agree("pgvector settings")
    for settings in (hook_settings(), load_hook_settings()):
        assert (settings.dense_backend, settings.pgvector_schema, settings.pgvector_hnsw) == ("pgvector", "Archive_2", True)
        assert settings.pgvector_dsn_file == world["tmp"] / "dsn.txt"


def test_the_hook_reader_never_reads_the_working_directory_env_file(world):
    settings = load_hook_settings()
    assert settings.index_dir != Path("/decoy") and settings.claude_enforcement == "confirmed_only"
    assert settings.local_only is False and settings.openrouter_api_key is None
    assert Settings().index_dir == Path("/decoy")        # the server does read it: the hook difference is deliberate


def test_the_light_settings_hide_the_api_key_in_repr(world):
    put(world, OPENROUTER_API_KEY="sk-synthetic-not-a-real-key")
    assert "sk-synthetic" not in repr(load_hook_settings())


def test_the_hook_adapters_run_on_the_light_reader(world):
    from self_directing_mcp import agy_hooks, claude_hooks
    assert isinstance(claude_hooks._new_settings(), settings_core.LightSettings)
    for module in (claude_hooks, agy_hooks):
        engine = module._new_engine()
        try:
            assert isinstance(engine.settings, settings_core.LightSettings)
        finally:
            engine.close()


def test_the_accessors_agree(world):
    put(world, SELF_DIRECT_INDEX_DIR=str(world["tmp"] / "i"), SELF_DIRECT_AGY_APP_DATA_DIRS=json.dumps(["a", "b"]),
        SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR=json.dumps([str(world["tmp"] / "g")]))
    ref, light = hook_settings(), load_hook_settings()
    assert light.resolve_contracts_path() == ref.resolve_contracts_path()
    assert light.resolve_agy_app_data_dirs() == ref.resolve_agy_app_data_dirs()
    assert light.resolve_grokbot_transcripts_dirs() == ref.resolve_grokbot_transcripts_dirs()
    assert light.resolve_claude_projects_dir() == ref.resolve_claude_projects_dir()
    assert light.resolve_sessions_dir() == ref.resolve_sessions_dir()
    assert light.resolve_omp_sessions_dir() == ref.resolve_omp_sessions_dir()
    assert light.resolve_session_provider() == ref.resolve_session_provider()


# --- one environment variable at a time: every field, valid and invalid values ----------------------------------------

BOOLS = ["true", "True", "TRUE", "false", "False", "1", "0", "yes", "no", "on", "off", "t", "f", "y", "n", "",
         " true", "true ", "maybe", "2", "tru", "yes ", "Y", "N", "enabled", "none"]
INTS = ["4", "004", "+4", "-4", " 4 ", "4.0", "4.5", "1e2", "1_0", "\u0663", "", "abc", "0x10", "nan", "inf", "0", "1",
        "-1", "2147483648", "9" * 30, "1.0e0", "+0", "00", " ", "4\n", "١٢", "1__0", "_1", "1_", "1_0_0", "-1_0", "-_1",
        "1_0.0", "1.0_0", "1_.0", "10.00", "1.", ".0", "\x851", "\u20281", "\x1c1", "\u200b1", "\ufeff1", "1 0",
        "9999999999999999999999999.0", "-0.0"]
FLOATS = ["1", "1.5", ".5", "5.", "1e3", "1E-3", "+2", "-1", "0", "0.0", "-0.0", "inf", "-inf", "nan", "NaN", "infinity",
          "Infinity", "+inf", "1_0.5", " 2 ", "", "abc", "1e400", "1e-400", "0x1p3", "1,5", "1e", "e5", "-.5", "+.5e1",
          ".", "1__0", "2\n", "\u0661", "1_0e3", "1e1_0", "1_e3", "1e_3", "5_.", "-_1", "in_f", "in_finity", "na_n",
          "i_nf_", "1e1__0", "1_", "_1", "\x851.5", "\x1c1.5", "\u200b1.5", "1.e3", ".e3", "+nan", "-Infinity"]
PATHS = ["/abs/dir", "relative/dir", "~/x", "", ".", "..", "a b", "/a/../b", "C:\\x\\y", "x" * 300, "//double"]
LISTS = ['["/a", "/b"]', "[]", '"/a"', '{"a": 1}', "not json", '["/a", 5]', "[1]", "", "null", "[null]", "true", "5",
         '["~/a", "rel"]', '[""]', ' ["/a"] ', "['/a']"]
STR_OR_LISTS = ["/a;/b", '["/a"]', '"quoted"', "[1]", "5", "true", "null", '{"a": 1}', "", "[]", "/a:/b", "C:\\g",
                '["/a", "/b"]', "  ", '["a", 1]', "1.5", "false"]
CHOICES = {"agy_enforcement": ["confirmed_only", "advisory", "enforced", "Advisory", "", " advisory", "bogus", "ENFORCED"],
           "claude_enforcement": ["confirmed_only", "advisory", "enforced", "Enforced", "", "advisory ", "bogus"],
           "session_provider": ["codex", "grokbot", "omp", "agy", "claude", "Codex", "", "cursor", "claude "],
           "dense_backend": ["sqlite-vector", "numpy", "pgvector", "NumPy", "PGVECTOR", "Pgvector", "", "sqlite_vector", "faiss",
                             "pg-vector", "pgvector ", "postgres"],
           "graph_extractor": ["openrouter", "host", "Host", "HOST", "OpenRouter", "OPENROUTER", "", " host", "host ",
                               "open_router", "open-router", "local", "none", "true", "1", "host\n"],
           "daemon": ["off", "on", "On", "ON", "", " on", "on ", "auto", "true", "1", "false", "Off"],
           "sqlite_journal": ["auto", "delete", "Auto", "DELETE", "", " auto", "delete ", "wal", "off", "true", "truncate"],
           "sync_slicing": ["auto", "off", "Auto", "OFF", "Off", "", " off", "auto ", "on", "false", "0", "true", "slice",
                           "off\n"]}
# SELF_DIRECT_AUDIT_SHADOW_EVERY: a non-negative integer, or "adaptive" in any case with surrounding whitespace.
SHADOW_EVERY = [*INTS, "adaptive", "Adaptive", "ADAPTIVE", " adaptive ", "adaptive\n", "\tadaptive", "adaptiv", "adaptive!",
                "adapt ive", "fixed", "auto", "none", "-1", "adaptive0", "0adaptive", "\u0430daptive", "adaptive\u200b"]
KIND_VALUES = {"bool": BOOLS, "int": INTS, "float": FLOATS, "path": PATHS, "list": LISTS, "mixed": STR_OR_LISTS,
               "shadow_every": SHADOW_EVERY}
FIELD_KINDS = {
    "codex_sessions_dir": "path", "omp_sessions_dir": "path", "claude_projects_dir": "path", "index_dir": "path",
    "contracts_path": "path", "sqlite_vector_path": "path", "openrouter_api_key_file": "path", "pgvector_dsn_file": "path",
    "agy_app_data_dirs": "list", "grokbot_transcripts_dir": "mixed",
    "use_fake_embedder": "bool", "local_only": "bool", "seed_example_contracts": "bool", "pgvector_hnsw": "bool",
    "embedding_dim": "int", "max_concurrent_calls": "int", "max_decompressed_session_bytes": "int", "rrf_k": "int",
    "retrieve_top_k": "int", "search_top_k": "int",
    "lock_wait_timeout_sec": "float", "audit_timeout_sec": "float", "request_timeout_sec": "float",
    "graph_update_timeout_sec": "float", "max_tool_timeout_sec": "float",
    "integrity_fast_recent_blocks": "int", "integrity_fast_sample_blocks": "int",
    "integrity_full_verify_interval_sec": "float", "integrity_background_verify": "bool",
    "integrity_background_idle_sec": "float", "integrity_background_max_scans": "int",
    "audit_incremental": "bool", "audit_shadow_every": "shadow_every", "audit_shadow_idle_sec": "float",
    "audit_incremental_min_events": "int", "sync_slice_bytes": "int",
}
# SELF_DIRECT_PGVECTOR_SCHEMA: ASCII letters, digits, underscore; at most 63; no leading digit; not pg_* or information_schema.
PGVECTOR_SCHEMAS = ["tsukkomi", "a_b1", "Tsukkomi", "_x", "x", "public", "x" * 63, "x" * 64, "1abc", "", " tsukkomi", "tsukkomi ",
                    "x y", "a;b", 'a"b', "a-b", "a$b", "schema\u00e9", "pg_catalog", "PG_x", "pgx", "information_schema",
                    "Information_Schema", "tsukkomi\n", "a.b", "\u0661a"]
FREE_TEXT = {"pgvector_schema": PGVECTOR_SCHEMAS, "embedding_model": ["m", "", " spaced model ", "a/b:c"], "embedding_base_url": ["http://x", "", "u r l"],
             "openrouter_api_key": ["k", "", " k ", "sk-synthetic"]}


def single_cases():
    for name, kind in FIELD_KINDS.items():
        for value in KIND_VALUES[kind]:
            yield name, value
    for name, values in CHOICES.items():
        for value in values:
            yield name, value
    for name, values in FREE_TEXT.items():
        for value in values:
            yield name, value


@pytest.mark.parametrize("name,value", list(single_cases()),
                         ids=[f"{n}={v!r}"[:60] for n, v in single_cases()])
def test_each_environment_variable_agrees(world, name, value):
    variable = "OPENROUTER_API_KEY" if name == "openrouter_api_key" else "SELF_DIRECT_" + name.upper()
    put(world, **{variable: value})
    agree(f"{variable}={value!r}")


@pytest.mark.parametrize("name", PGVECTOR_SCHEMAS)
def test_the_pgvector_schema_rule_is_one_function(world, name):
    put(world, SELF_DIRECT_PGVECTOR_SCHEMA=name)
    try:
        settings_core.check_pgvector_schema(name)
        accepted = True
    except ValueError:
        accepted = False
    assert (outcome(hook_settings)[0] == "ok") is accepted and (outcome(load_hook_settings)[0] == "ok") is accepted


def test_every_field_is_exercised_by_the_single_variable_cases():
    covered = {name for name, _value in single_cases()}
    assert covered == set(Settings.model_fields)


@pytest.mark.parametrize("variable", ["self_direct_index_dir", "Self_Direct_Local_Only", "openrouter_api_key",
                                      "SELF_DIRECT_openrouter_api_key", "SELF_DIRECT_OPENROUTER_API_KEY"])
def test_environment_names_are_case_insensitive_and_the_key_has_only_its_alias(world, variable):
    put(world, **{variable: "true" if "only" in variable.lower() else str(world["tmp"] / "x")})
    agree(variable)


# --- precedence between the sources ----------------------------------------------------------------------------------

def test_environment_beats_codex_variable_beats_toml_beats_default(world):
    sessions_toml, sessions_codex, sessions_env = (world["tmp"] / n for n in ("toml", "codex", "env"))
    toml(world, f"[paths]\ncodex_sessions_dir = {json.dumps(str(sessions_toml))}\n")
    agree("toml only")
    assert load_hook_settings().codex_sessions_dir == sessions_toml.resolve()
    put(world, CODEX_SESSIONS_DIR=str(sessions_codex))
    agree("codex over toml")
    assert load_hook_settings().codex_sessions_dir == sessions_codex
    put(world, SELF_DIRECT_CODEX_SESSIONS_DIR=str(sessions_env))
    agree("env over codex")
    assert load_hook_settings().codex_sessions_dir == sessions_env


def test_environment_beats_toml_for_every_path_field(world):
    toml(world, "[paths]\n" + "\n".join([
        'index_dir = "ti"', 'contracts_path = "tc.json"', 'claude_projects_dir = "tp"', 'omp_sessions_dir = "to"',
        'agy_app_data_dirs = ["ta", "tb"]', 'grokbot_transcripts_dir = ["tg1", "tg2"]',
        'sqlite_vector_path = "tv.so"', 'openrouter_api_key_file = "tk.env"', 'pgvector_dsn_file = "tp.dsn"']) + "\n")
    agree("toml only")
    put(world, SELF_DIRECT_INDEX_DIR=str(world["tmp"] / "ei"), SELF_DIRECT_AGY_APP_DATA_DIRS='["/ea"]',
        SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR="/eg", SELF_DIRECT_SQLITE_VECTOR_PATH="/ev.so",
        SELF_DIRECT_PGVECTOR_DSN_FILE="/ep.dsn")
    agree("env over toml")


def test_toml_relative_paths_resolve_against_the_toml_directory_and_tilde_expands(world):
    nested = world["tmp"] / "nested"
    nested.mkdir()
    other = nested / "other.toml"
    other.write_text('[paths]\nindex_dir = "idx"\ncontracts_path = "~/c.json"\nagy_app_data_dirs = ["a", "~/b"]\n',
                     encoding="utf-8")
    put(world, SELF_DIRECT_CONFIG_FILE=str(other))
    agree("relative toml")
    assert load_hook_settings().index_dir == (nested / "idx").resolve()


def test_toml_default_location_under_home_is_used_without_an_explicit_file(world):
    world["monkeypatch"].delenv("SELF_DIRECT_CONFIG_FILE")
    agree("no toml at the default location")
    default = world["home"] / ".config" / "tsukkomi-mcp" / "config.toml"
    default.parent.mkdir(parents=True)
    default.write_text('[paths]\nindex_dir = "from-default"\n', encoding="utf-8")
    agree("toml at the default location")
    assert load_hook_settings().index_dir == (default.parent / "from-default").resolve()


@pytest.mark.parametrize("text", [
    "", "[paths]\n", '[paths]\nindex_dir = "x"\n', '[paths]\nindex_dir = ""\n', "[paths]\nindex_dir = 5\n",
    '[paths]\nunknown = "x"\n', '[other]\nindex_dir = "x"\n', '[paths]\nindex_dir = "x"\n[extra]\na = 1\n',
    "not toml at all [", '[paths]\nagy_app_data_dirs = "not-a-list"\n', '[paths]\nagy_app_data_dirs = [""]\n',
    '[paths]\nagy_app_data_dirs = [1]\n', '[paths]\ngrokbot_transcripts_dir = "a;b"\n',
    '[paths]\ngrokbot_transcripts_dir = ""\n', '[paths]\ngrokbot_transcripts_dir = 5\n',
    '[paths]\ngrokbot_transcripts_dir = ["a", "b"]\n', 'paths = "x"\n', "[paths]\ncontracts_path = []\n",
    '[paths]\nopenrouter_api_key_file = "k.env"\nsqlite_vector_path = "v.so"\n',
])
def test_toml_documents_agree_including_invalid_ones(world, text):
    toml(world, text)
    agree(text)


def test_a_missing_explicit_toml_file_is_an_error_in_both(world):
    put(world, SELF_DIRECT_CONFIG_FILE=str(world["tmp"] / "does-not-exist.toml"))
    assert outcome(hook_settings)[0] == "error" and outcome(load_hook_settings)[0] == "error"


def test_an_invalid_toml_is_an_error_even_when_the_environment_would_override_it(world):
    toml(world, "not toml [")
    put(world, SELF_DIRECT_INDEX_DIR=str(world["tmp"] / "i"))
    assert outcome(hook_settings)[0] == "error" and outcome(load_hook_settings)[0] == "error"


def test_the_native_vector_library_default_follows_the_checkout(world, tmp_path):
    library = tmp_path / "native"
    library.mkdir()
    (library / "vector.so").write_bytes(b"")
    world["monkeypatch"].setattr("self_directing_mcp.config._repo_root", lambda: tmp_path)
    world["monkeypatch"].setattr(settings_core, "repo_root", lambda: tmp_path)
    agree("library next to the checkout")
    assert load_hook_settings().sqlite_vector_path == (library / "vector.so").resolve()


# --- many combinations -------------------------------------------------------------------------------------------------

def random_world(rng: random.Random, world) -> str:
    """Set a random mix of environment variables and a random TOML; returns a description for failure messages."""
    parts = []
    names = rng.sample(sorted(FIELD_KINDS) + sorted(CHOICES) + sorted(FREE_TEXT), rng.randint(0, 9))
    for name in names:
        if name in FIELD_KINDS:
            pool = KIND_VALUES[FIELD_KINDS[name]]
            # Mostly well-formed values so that the combination is exercised, with an occasional bad one.
            value = rng.choice(pool)
        elif name in CHOICES:
            value = rng.choice(CHOICES[name])
        else:
            value = rng.choice(FREE_TEXT[name])
        variable = "OPENROUTER_API_KEY" if name == "openrouter_api_key" else "SELF_DIRECT_" + name.upper()
        if rng.random() < 0.15:
            variable = variable.lower()
        world["monkeypatch"].setenv(variable, value)
        parts.append(f"{variable}={value!r}")
    if rng.random() < 0.3:
        world["monkeypatch"].setenv("CODEX_SESSIONS_DIR", rng.choice(PATHS))
        parts.append("CODEX_SESSIONS_DIR")
    if rng.random() < 0.3:
        world["monkeypatch"].setenv("CLAUDE_CONFIG_DIR", rng.choice(PATHS))
        parts.append("CLAUDE_CONFIG_DIR")
    keys = rng.sample(["index_dir", "contracts_path", "claude_projects_dir", "omp_sessions_dir", "codex_sessions_dir",
                       "sqlite_vector_path", "openrouter_api_key_file", "pgvector_dsn_file", "agy_app_data_dirs",
                       "grokbot_transcripts_dir"], rng.randint(0, 6))
    lines = ["[paths]"]
    for key in keys:
        if key == "agy_app_data_dirs":
            lines.append(f"{key} = {json.dumps(rng.sample(['a', '/b', '~/c', 'd/e'], rng.randint(0, 3)))}")
        elif key == "grokbot_transcripts_dir":
            lines.append(f"{key} = {json.dumps(rng.choice(['x;y', ['p', '/q'], '/solo']))}")
        else:
            lines.append(f"{key} = {json.dumps(rng.choice(['rel', '/abs', '~/t', 'a/../b']))}")
    toml(world, "\n".join(lines) + "\n")
    parts.append(f"toml keys {keys}")
    return "; ".join(parts)


@pytest.mark.parametrize("seed", range(40))
def test_random_combinations_agree(world, seed):
    rng = random.Random(seed)
    for _ in range(8):
        for name in list(os.environ):
            if name.upper().startswith(ENV_PREFIXES) and name.upper() != "SELF_DIRECT_CONFIG_FILE":
                world["monkeypatch"].delenv(name, raising=False)
        description = random_world(rng, world)
        agree(description)
