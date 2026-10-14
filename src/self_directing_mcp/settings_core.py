"""Settings sources and defaults that need neither pydantic nor pydantic-settings.

This is the single source of truth shared by the pydantic ``config.Settings`` model (the server) and the light
reader the command hooks use (``load_hook_settings``). A hook process starts fresh for every tool call, so loading
pydantic-settings just to read a handful of environment variables and one TOML file is wasted time.

What lives here, so that the two readers cannot drift:

* the defaults (constants and default factories), the allowed values and numeric bounds of every field;
* the user TOML ``[paths]`` source and the ``CODEX_SESSIONS_DIR`` source;
* the accessor methods (``resolve_*``) that both settings objects share (``SettingsAccessors``);
* ``FIELDS``, the declarative field table the light reader is driven by. ``config.Settings`` declares the same
  fields with pydantic types, and ``tests/test_hook_settings_parity.py`` compares the two on many environment/TOML
  combinations, including invalid ones (both must raise).

Precedence, identical to ``Settings`` (highest first): environment (``SELF_DIRECT_<FIELD>``; ``OPENROUTER_API_KEY``
for the key; names are case-insensitive), ``CODEX_SESSIONS_DIR`` (only for the Codex sessions directory), the user
TOML ``[paths]`` (``SELF_DIRECT_CONFIG_FILE`` or ``~/.config/tsukkomi-mcp/config.toml``), then the defaults. A ``.env``
in the working directory is never read by the light reader, exactly as ``config.hook_settings()`` documents: hosts run
hooks inside the project being edited, and that file is content the audited agent can write.

Importing this module loads only the standard library.
"""
from __future__ import annotations

import json
import math
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from self_directing_mcp.compressed_source import DEFAULT_MAX_DECOMPRESSED_BYTES

ENV_PREFIX = "SELF_DIRECT_"
API_KEY_ENV = "OPENROUTER_API_KEY"

ENFORCEMENT_MODES = ("confirmed_only", "advisory", "enforced")
SESSION_PROVIDERS = ("codex", "grokbot", "omp", "agy", "claude")
DENSE_BACKENDS = ("sqlite-vector", "numpy")
DAEMON_MODES = ("off", "on")

DEFAULT_ENFORCEMENT = "confirmed_only"
DEFAULT_SESSION_PROVIDER = "codex"
DEFAULT_DENSE_BACKEND = "sqlite-vector"
DEFAULT_DAEMON = "off"
DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_EMBEDDING_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LOCK_WAIT_TIMEOUT_SEC = 1.0
DEFAULT_AUDIT_TIMEOUT_SEC = 8.0
DEFAULT_REQUEST_TIMEOUT_SEC = 45.0
DEFAULT_GRAPH_UPDATE_TIMEOUT_SEC = 300.0
DEFAULT_MAX_TOOL_TIMEOUT_SEC = 600.0
DEFAULT_MAX_CONCURRENT_CALLS = 4
DEFAULT_INTEGRITY_FAST_RECENT_BLOCKS = 4
DEFAULT_INTEGRITY_FAST_SAMPLE_BLOCKS = 8
DEFAULT_INTEGRITY_FULL_VERIFY_INTERVAL_SEC = 300.0
DEFAULT_INTEGRITY_BACKGROUND_IDLE_SEC = 2.0
DEFAULT_INTEGRITY_BACKGROUND_MAX_SCANS = 1
DEFAULT_AUDIT_SHADOW_EVERY = "adaptive"
DEFAULT_AUDIT_SHADOW_IDLE_SEC = 2.0
DEFAULT_AUDIT_INCREMENTAL_MIN_EVENTS = 500
DEFAULT_RRF_K = 60
DEFAULT_RETRIEVE_TOP_K = 20
DEFAULT_SEARCH_TOP_K = 10
AGY_APP_DATA_NAMES = ("antigravity", "antigravity-cli", "antigravity-ide")

PATH_FIELDS = frozenset({
    "index_dir", "codex_sessions_dir", "omp_sessions_dir", "grokbot_transcripts_dir", "claude_projects_dir",
    "agy_app_data_dirs", "contracts_path", "sqlite_vector_path", "openrouter_api_key_file",
})


# --- default factories --------------------------------------------------------------------------------------

def get_config_path() -> Path:
    selected = os.environ.get("SELF_DIRECT_CONFIG_FILE")
    return Path(selected or Path.home() / ".config" / "tsukkomi-mcp" / "config.toml").expanduser().resolve()


def repo_root() -> Path:
    """The checkout a source install runs from (two levels above this package's directory)."""
    return Path(__file__).resolve().parents[2]


def default_codex_sessions_dir() -> Path:
    # Windows: %USERPROFILE%\.codex\sessions ; Linux/macOS: ~/.codex/sessions
    override = os.environ.get("SELF_DIRECT_CODEX_SESSIONS_DIR") or os.environ.get("CODEX_SESSIONS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".codex" / "sessions"


def default_grokbot_transcripts_dir() -> str:
    """Empty by default; set SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR to enable."""
    return os.environ.get("SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR") or ""


def default_omp_sessions_dir() -> Path:
    return Path(os.environ.get("SELF_DIRECT_OMP_SESSIONS_DIR") or Path.home() / ".omp" / "sessions")


def default_claude_projects_dir() -> Path:
    # Claude Code keeps its projects under CLAUDE_CONFIG_DIR when set; the MCP server and hooks inherit it.
    configured = os.environ.get("SELF_DIRECT_CLAUDE_PROJECTS_DIR")
    if configured:
        return Path(configured)
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "projects"


def default_agy_app_data_dirs() -> list[Path]:
    return [Path.home() / ".gemini" / name for name in AGY_APP_DATA_NAMES]


def default_index_dir() -> Path:
    override = os.environ.get("SELF_DIRECT_INDEX_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".self_direct_index"


def default_sqlite_vector_path(root: Path | None = None) -> Path | None:
    override = os.environ.get("SELF_DIRECT_SQLITE_VECTOR_PATH")
    if override:
        return Path(override).expanduser().resolve()
    repo = root if root is not None else repo_root()
    # scripts/setup_sqlite_vector.py defaults to .native/; older installs used native/.
    for directory in ("native", ".native"):
        for ext in ("vector.so", "vector.dll", "vector.dylib"):
            cand = repo / directory / ext
            if cand.is_file():
                return cand.resolve()
    return None


# --- sources shared by both readers ---------------------------------------------------------------------------

def toml_path(value: object, directory: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("TOML paths must be nonempty strings")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else directory / path).resolve()


def toml_path_settings() -> dict[str, object]:
    """The user TOML ``[paths]`` table as resolved ``Path`` values (raises ``ValueError`` when unusable)."""
    source = get_config_path()
    try:
        with source.open("rb") as stream:
            document = tomllib.load(stream)
    except FileNotFoundError:
        if os.environ.get("SELF_DIRECT_CONFIG_FILE"):
            raise ValueError("Configured TOML path file does not exist") from None
        return {}
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        raise ValueError("Unable to read valid TOML path configuration") from None
    if set(document) - {"paths"}:
        raise ValueError("TOML path configuration only supports [paths]")
    paths = document.get("paths", {})
    if not isinstance(paths, dict) or set(paths) - PATH_FIELDS:
        raise ValueError("Invalid or unknown TOML path setting")
    resolved: dict[str, object] = {}
    for name, value in paths.items():
        if name == "grokbot_transcripts_dir":
            from self_directing_mcp.grokbot.discover import parse_transcripts_dirs
            if isinstance(value, str):
                if not value.strip():
                    raise ValueError("TOML paths must be nonempty strings")
                values = [str(path) for path in parse_transcripts_dirs(value)]
            elif isinstance(value, list):
                values = value
            else:
                raise ValueError("TOML Grokbot paths must be a string or array")
            resolved[name] = [toml_path(item, source.parent) for item in values]
        elif name == "agy_app_data_dirs":
            if not isinstance(value, list):
                raise ValueError("TOML AGY paths must be an array")
            resolved[name] = [toml_path(item, source.parent) for item in value]
        else:
            resolved[name] = toml_path(value, source.parent)
    return resolved


def codex_environment_settings() -> dict[str, object]:
    value = os.environ.get("CODEX_SESSIONS_DIR")
    return {"codex_sessions_dir": value} if value else {}


# --- accessors shared by both settings objects -------------------------------------------------------------------

class SettingsAccessors:
    """The ``resolve_*`` methods of ``Settings``; mixed into the pydantic model and into ``LightSettings``."""

    def resolve_api_key(self) -> str | None:
        if self.openrouter_api_key_file is not None:
            from dotenv import dotenv_values
            source = self.openrouter_api_key_file.expanduser().resolve()
            if not source.is_file():
                raise ValueError("Configured OpenRouter key file does not exist")
            key = dotenv_values(source, interpolate=False).get("OPENROUTER_API_KEY")
            if not key or not key.strip():
                raise ValueError("Configured OpenRouter key file has no OPENROUTER_API_KEY")
            return key.strip()
        return self.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")

    def resolve_sessions_dir(self) -> Path:
        return Path(self.codex_sessions_dir)

    def resolve_omp_sessions_dir(self) -> Path:
        return Path(self.omp_sessions_dir)

    def resolve_claude_projects_dir(self) -> Path:
        return Path(self.claude_projects_dir)

    def resolve_agy_app_data_dirs(self) -> list[Path]:
        return [Path(root).expanduser().resolve() for root in self.agy_app_data_dirs]

    def resolve_grokbot_transcripts_dirs(self) -> list[Path]:
        from self_directing_mcp.grokbot.discover import parse_transcripts_dirs

        return parse_transcripts_dirs(self.grokbot_transcripts_dir)

    def resolve_session_provider(self) -> str:
        return self.session_provider

    def resolve_contracts_path(self) -> Path:
        if self.contracts_path:
            return Path(self.contracts_path)
        return Path(self.index_dir) / "contracts.json"


# --- the light reader ---------------------------------------------------------------------------------------------
# Value coercion mirrors what pydantic accepts for these field types when they come from the environment (a string):
# the differential tests in tests/test_hook_settings_parity.py pin the agreement, including the strings it rejects.

_BOOL_TRUE = frozenset({"1", "on", "t", "true", "y", "yes"})
_BOOL_FALSE = frozenset({"0", "off", "f", "false", "n", "no"})
# pydantic trims Unicode White_Space (not U+001C..U+001F, U+200B or U+FEFF) before parsing a number, and accepts
# single underscores between digits. Booleans and choices are not trimmed.
_WHITESPACE = ("\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009"
               "\u200a\u2028\u2029\u202f\u205f\u3000")
_INT = re.compile(r"[+-]?[0-9]+(?:_[0-9]+)*(?:\.0+)?")           # a whole number, optionally spelled with ".0"
_FLOAT = re.compile(r"[+-]?(?:(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|inf|infinity|nan)", re.IGNORECASE)


def _bad(name: str, what: str) -> ValueError:
    return ValueError(f"invalid {what} for {name}")


def _to_bool(name: str, text: str) -> bool:
    lowered = text.lower()
    if lowered in _BOOL_TRUE:
        return True
    if lowered in _BOOL_FALSE:
        return False
    raise _bad(name, "boolean")


def _to_int(name: str, text: str, minimum: int | None = None) -> int:
    stripped = text.strip(_WHITESPACE)
    if not _INT.fullmatch(stripped):
        raise _bad(name, "integer")
    value = int(stripped.split(".", 1)[0].replace("_", ""))
    if minimum is not None and value < minimum:
        raise _bad(name, "integer")
    return value


def _to_float(name: str, text: str, *, positive: bool = False, finite: bool = False) -> float:
    stripped = text.strip(_WHITESPACE)
    if not _FLOAT.fullmatch(stripped):
        # A single underscore between characters is dropped; a leading, trailing or doubled one is an error.
        if "_" in stripped and not stripped.startswith("_") and not stripped.endswith("_") and "__" not in stripped:
            stripped = stripped.replace("_", "")
        if not _FLOAT.fullmatch(stripped):
            raise _bad(name, "number")
    value = float(stripped)
    if finite and not math.isfinite(value):
        raise _bad(name, "number")
    if positive and not value > 0:
        raise _bad(name, "number")
    return value


def _to_nonnegative_finite(name: str, text: str) -> float:
    value = _to_float(name, text, finite=True)
    if not value >= 0:
        raise _bad(name, "number")
    return value


def _to_shadow_every(name: str, text: str):
    """``adaptive`` (any case, surrounding whitespace allowed) or a non-negative integer."""
    if text.strip(_WHITESPACE).lower() == "adaptive":
        return "adaptive"
    return _to_int(name, text, 0)


def _literal(choices: tuple[str, ...]) -> Callable[[str, str], str]:
    def convert(name: str, text: str) -> str:
        if text not in choices:
            raise _bad(name, "choice")
        return text
    return convert


def _path(name: str, text: str) -> Path:
    return Path(text)


def _json_value(text: str) -> Any:
    return json.loads(text)


def _path_list(name: str, text: str) -> list[Path] | None:
    """A JSON list of paths; JSON ``null`` means "not provided" (the next source decides), as in pydantic-settings."""
    try:
        value = _json_value(text)
    except ValueError:
        raise _bad(name, "JSON list") from None
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise _bad(name, "list of paths")
    return [Path(item) for item in value]


def _str_or_path_list(name: str, text: str):
    """``grokbot_transcripts_dir``: a plain string, or a JSON list of paths."""
    try:
        value = _json_value(text)
    except ValueError:
        return text
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [Path(item) for item in value]
    raise _bad(name, "path string or list")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    parse: Callable[[str, str], Any]            # (field name, environment text) -> value
    default: Callable[[], Any]                  # evaluated when no source provides the field
    alias: str | None = None                    # environment variable name without prefix (the API key)


def _static(value):
    return lambda: value


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("codex_sessions_dir", _path, default_codex_sessions_dir),
    FieldSpec("grokbot_transcripts_dir", _str_or_path_list, default_grokbot_transcripts_dir),
    FieldSpec("omp_sessions_dir", _path, default_omp_sessions_dir),
    FieldSpec("claude_projects_dir", _path, default_claude_projects_dir),
    FieldSpec("agy_app_data_dirs", _path_list, default_agy_app_data_dirs),
    FieldSpec("agy_enforcement", _literal(ENFORCEMENT_MODES), _static(DEFAULT_ENFORCEMENT)),
    FieldSpec("claude_enforcement", _literal(ENFORCEMENT_MODES), _static(DEFAULT_ENFORCEMENT)),
    FieldSpec("session_provider", _literal(SESSION_PROVIDERS), _static(DEFAULT_SESSION_PROVIDER)),
    FieldSpec("index_dir", _path, default_index_dir),
    FieldSpec("contracts_path", _path, _static(None)),
    FieldSpec("openrouter_api_key", lambda name, text: text, _static(None), alias=API_KEY_ENV),
    FieldSpec("embedding_model", lambda name, text: text, _static(DEFAULT_EMBEDDING_MODEL)),
    FieldSpec("embedding_dim", lambda name, text: _to_int(name, text), _static(DEFAULT_EMBEDDING_DIM)),
    FieldSpec("embedding_base_url", lambda name, text: text, _static(DEFAULT_EMBEDDING_BASE_URL)),
    FieldSpec("use_fake_embedder", lambda name, text: _to_bool(name, text), _static(False)),
    FieldSpec("local_only", lambda name, text: _to_bool(name, text), _static(False)),
    FieldSpec("dense_backend", _literal(DENSE_BACKENDS), _static(DEFAULT_DENSE_BACKEND)),
    FieldSpec("sqlite_vector_path", _path, default_sqlite_vector_path),
    FieldSpec("openrouter_api_key_file", _path, _static(None)),
    FieldSpec("seed_example_contracts", lambda name, text: _to_bool(name, text), _static(False)),
    FieldSpec("lock_wait_timeout_sec", lambda name, text: _to_float(name, text, positive=True),
              _static(DEFAULT_LOCK_WAIT_TIMEOUT_SEC)),
    FieldSpec("audit_timeout_sec", lambda name, text: _to_float(name, text, positive=True),
              _static(DEFAULT_AUDIT_TIMEOUT_SEC)),
    FieldSpec("request_timeout_sec", lambda name, text: _to_float(name, text, positive=True),
              _static(DEFAULT_REQUEST_TIMEOUT_SEC)),
    FieldSpec("graph_update_timeout_sec", lambda name, text: _to_float(name, text, positive=True, finite=True),
              _static(DEFAULT_GRAPH_UPDATE_TIMEOUT_SEC)),
    FieldSpec("max_tool_timeout_sec", lambda name, text: _to_float(name, text, positive=True),
              _static(DEFAULT_MAX_TOOL_TIMEOUT_SEC)),
    FieldSpec("max_concurrent_calls", lambda name, text: _to_int(name, text, 1), _static(DEFAULT_MAX_CONCURRENT_CALLS)),
    FieldSpec("max_decompressed_session_bytes", lambda name, text: _to_int(name, text, 1),
              _static(DEFAULT_MAX_DECOMPRESSED_BYTES)),
    FieldSpec("integrity_fast_recent_blocks", lambda name, text: _to_int(name, text, 0),
              _static(DEFAULT_INTEGRITY_FAST_RECENT_BLOCKS)),
    FieldSpec("integrity_fast_sample_blocks", lambda name, text: _to_int(name, text, 0),
              _static(DEFAULT_INTEGRITY_FAST_SAMPLE_BLOCKS)),
    FieldSpec("integrity_full_verify_interval_sec", _to_nonnegative_finite,
              _static(DEFAULT_INTEGRITY_FULL_VERIFY_INTERVAL_SEC)),
    FieldSpec("integrity_background_verify", lambda name, text: _to_bool(name, text), _static(True)),
    FieldSpec("integrity_background_idle_sec", lambda name, text: _to_float(name, text, positive=True, finite=True),
              _static(DEFAULT_INTEGRITY_BACKGROUND_IDLE_SEC)),
    FieldSpec("integrity_background_max_scans", lambda name, text: _to_int(name, text, 1),
              _static(DEFAULT_INTEGRITY_BACKGROUND_MAX_SCANS)),
    FieldSpec("audit_incremental", lambda name, text: _to_bool(name, text), _static(True)),
    FieldSpec("audit_shadow_every", _to_shadow_every, _static(DEFAULT_AUDIT_SHADOW_EVERY)),
    FieldSpec("audit_shadow_idle_sec", _to_nonnegative_finite, _static(DEFAULT_AUDIT_SHADOW_IDLE_SEC)),
    FieldSpec("audit_incremental_min_events", lambda name, text: _to_int(name, text, 0),
              _static(DEFAULT_AUDIT_INCREMENTAL_MIN_EVENTS)),
    FieldSpec("rrf_k", lambda name, text: _to_int(name, text), _static(DEFAULT_RRF_K)),
    FieldSpec("retrieve_top_k", lambda name, text: _to_int(name, text), _static(DEFAULT_RETRIEVE_TOP_K)),
    FieldSpec("search_top_k", lambda name, text: _to_int(name, text), _static(DEFAULT_SEARCH_TOP_K)),
    FieldSpec("daemon", _literal(DAEMON_MODES), _static(DEFAULT_DAEMON)),
)
FIELD_NAMES = tuple(spec.name for spec in FIELDS)


class LightSettings(SettingsAccessors):
    """Plain attribute bag with every ``Settings`` field and accessor; no validation framework involved.

    Mutable on purpose, like ``Settings``: callers (and tests) set ``claude_enforcement`` and friends directly.
    """

    def __init__(self, **values: Any) -> None:
        unknown = set(values) - set(FIELD_NAMES)
        if unknown:
            raise TypeError(f"unknown setting: {sorted(unknown)[0]}")
        self.__dict__.update(values)

    def model_dump(self) -> dict[str, Any]:
        return {name: self.__dict__[name] for name in FIELD_NAMES}

    def __repr__(self) -> str:
        shown = {name: ("***" if name == "openrouter_api_key" and value else value)
                 for name, value in self.model_dump().items()}
        return "LightSettings(" + ", ".join(f"{k}={v!r}" for k, v in shown.items()) + ")"


def load_hook_settings(environ: dict[str, str] | None = None) -> LightSettings:
    """Hook settings: environment, ``CODEX_SESSIONS_DIR``, the user TOML ``[paths]``, then defaults; never a ``.env``.

    Same sources, same precedence and same errors as ``config.hook_settings()`` (``ValueError`` for anything
    ``Settings`` would reject), without importing pydantic-settings. ``environ`` defaults to ``os.environ``.
    """
    env = {key.lower(): value for key, value in (os.environ if environ is None else environ).items()}
    toml = toml_path_settings()                      # read (and validated) whatever the environment provides
    codex_env = codex_environment_settings()
    values: dict[str, Any] = {}
    for spec in FIELDS:
        key = spec.alias.lower() if spec.alias else (ENV_PREFIX + spec.name).lower()
        value = spec.parse(spec.name, env[key]) if key in env else None      # None: not provided by the environment
        if value is None and spec.name in codex_env:
            value = spec.parse(spec.name, codex_env[spec.name])
        if value is None and spec.name in toml:
            value = toml[spec.name]
        values[spec.name] = value if value is not None else spec.default()
    return LightSettings(**values)
