from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Callable, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


_PATH_FIELDS = frozenset({
    "index_dir", "codex_sessions_dir", "omp_sessions_dir", "grokbot_transcripts_dir",
    "agy_app_data_dirs", "contracts_path", "sqlite_vector_path", "openrouter_api_key_file",
})


def get_config_path() -> Path:
    selected = os.environ.get("SELF_DIRECT_CONFIG_FILE")
    return Path(selected or Path.home() / ".config" / "tsukkomi-mcp" / "config.toml").expanduser().resolve()


def _toml_path(value: object, directory: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("TOML paths must be nonempty strings")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else directory / path).resolve()


def _toml_path_settings() -> dict[str, object]:
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
    if not isinstance(paths, dict) or set(paths) - _PATH_FIELDS:
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
            resolved[name] = [_toml_path(item, source.parent) for item in values]
        elif name == "agy_app_data_dirs":
            if not isinstance(value, list):
                raise ValueError("TOML AGY paths must be an array")
            resolved[name] = [_toml_path(item, source.parent) for item in value]
        else:
            resolved[name] = _toml_path(value, source.parent)
    return resolved


def _codex_environment_settings() -> dict[str, object]:
    value = os.environ.get("CODEX_SESSIONS_DIR")
    return {"codex_sessions_dir": value} if value else {}


def _default_codex_sessions_dir() -> Path:
    # Windows: %USERPROFILE%\.codex\sessions ; Linux/macOS: ~/.codex/sessions
    override = os.environ.get("SELF_DIRECT_CODEX_SESSIONS_DIR") or os.environ.get("CODEX_SESSIONS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".codex" / "sessions"


def _default_grokbot_transcripts_dir() -> str:
    """Empty by default; set SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR to enable."""
    return os.environ.get("SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR") or ""


def _default_omp_sessions_dir() -> Path:
    return Path(os.environ.get("SELF_DIRECT_OMP_SESSIONS_DIR") or Path.home() / ".omp" / "sessions")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_index_dir() -> Path:
    override = os.environ.get("SELF_DIRECT_INDEX_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".self_direct_index"


def _default_sqlite_vector_path() -> Path | None:
    override = os.environ.get("SELF_DIRECT_SQLITE_VECTOR_PATH")
    if override:
        return Path(override).expanduser().resolve()
    repo = _repo_root()
    for ext in ("vector.so", "vector.dll", "vector.dylib"):
        cand = repo / "native" / ext
        if cand.is_file():
            return cand.resolve()
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SELF_DIRECT_",
        env_file=".env",
        extra="ignore",
    )

    codex_sessions_dir: Path = Field(default_factory=_default_codex_sessions_dir)
    grokbot_transcripts_dir: str | list[Path] = Field(default_factory=_default_grokbot_transcripts_dir)
    omp_sessions_dir: Path = Field(default_factory=_default_omp_sessions_dir)
    agy_app_data_dirs: list[Path] = Field(default_factory=lambda: [
        Path.home() / ".gemini" / name
        for name in ("antigravity", "antigravity-cli", "antigravity-ide")
    ])
    agy_enforcement: Literal["enforced", "advisory"] = "enforced"
    session_provider: Literal["codex", "grokbot", "omp", "agy"] = "codex"
    index_dir: Path = Field(default_factory=_default_index_dir)
    contracts_path: Path | None = Field(
        default=None,
        description="Optional JSON contracts file; default index_dir/contracts.json",
    )
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    embedding_model: str = "qwen/qwen3-embedding-8b"
    embedding_dim: int = 1024
    embedding_base_url: str = "https://openrouter.ai/api/v1"
    use_fake_embedder: bool = False
    local_only: bool = False
    dense_backend: Literal["sqlite-vector", "numpy"] = "sqlite-vector"
    sqlite_vector_path: Path | None = Field(default_factory=_default_sqlite_vector_path)
    openrouter_api_key_file: Path | None = Field(
        default=None,
        description="Authorized dotenv file containing OPENROUTER_API_KEY; never log its contents",
    )
    seed_example_contracts: bool = False
    lock_wait_timeout_sec: float = Field(default=1.0, gt=0)
    audit_timeout_sec: float = Field(default=8.0, gt=0)
    request_timeout_sec: float = Field(default=45.0, gt=0)
    graph_update_timeout_sec: float = Field(default=300.0, gt=0, allow_inf_nan=False)
    max_tool_timeout_sec: float = Field(
        default=600.0,
        gt=0,
        description="Upper bound for client-provided timeout_ms on long-running tools.",
    )
    rrf_k: int = 60
    retrieve_top_k: int = 20
    search_top_k: int = 10

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource | Callable[[], dict[str, object]], ...]:
        return (init_settings, env_settings, _codex_environment_settings,
                _toml_path_settings, dotenv_settings, file_secret_settings)

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

    def resolve_agy_app_data_dirs(self) -> list[Path]:
        return [Path(root).expanduser().resolve() for root in self.agy_app_data_dirs]

    def resolve_grokbot_transcripts_dirs(self) -> list[Path]:
        from self_directing_mcp.grokbot.discover import parse_transcripts_dirs

        return parse_transcripts_dirs(self.grokbot_transcripts_dir)

    def resolve_session_provider(self) -> Literal["codex", "grokbot", "omp", "agy"]:
        return self.session_provider

    def resolve_contracts_path(self) -> Path:
        if self.contracts_path:
            return Path(self.contracts_path)
        return Path(self.index_dir) / "contracts.json"


def get_settings(**overrides) -> Settings:
    key_file = os.environ.get("SELF_DIRECT_OPENROUTER_API_KEY_FILE")
    if key_file and Path(key_file).is_file():
        from dotenv import load_dotenv
        load_dotenv(key_file)
    return Settings(**overrides)
