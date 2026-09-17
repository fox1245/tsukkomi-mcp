from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_codex_sessions_dir() -> Path:
    # Windows: %USERPROFILE%\.codex\sessions ; Linux/macOS: ~/.codex/sessions
    override = os.environ.get("SELF_DIRECT_CODEX_SESSIONS_DIR") or os.environ.get("CODEX_SESSIONS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".codex" / "sessions"


def _default_grokbot_transcripts_dir() -> str:
    """Empty by default; set SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR to enable."""
    return os.environ.get("SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR") or ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_index_dir() -> Path:
    override = os.environ.get("SELF_DIRECT_INDEX_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return _repo_root() / ".self_direct_index"


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


def _default_openrouter_api_key_file() -> Path | None:
    override = os.environ.get("SELF_DIRECT_OPENROUTER_API_KEY_FILE")
    if override:
        return Path(override).expanduser().resolve()
    env_file = _repo_root() / ".env"
    if env_file.is_file():
        return env_file.resolve()
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SELF_DIRECT_",
        env_file=".env",
        extra="ignore",
    )

    codex_sessions_dir: Path = Field(default_factory=_default_codex_sessions_dir)
    grokbot_transcripts_dir: str = Field(default_factory=_default_grokbot_transcripts_dir)
    session_provider: Literal["codex", "grokbot"] = "codex"
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
    dense_backend: Literal["sqlite-vector", "numpy"] = "sqlite-vector"
    sqlite_vector_path: Path | None = Field(default_factory=_default_sqlite_vector_path)
    openrouter_api_key_file: Path | None = Field(
        default_factory=_default_openrouter_api_key_file,
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
        # Re-check env each call so CODEX_SESSIONS_DIR works without prefix
        override = os.environ.get("SELF_DIRECT_CODEX_SESSIONS_DIR") or os.environ.get("CODEX_SESSIONS_DIR")
        if override:
            return Path(override)
        return Path(self.codex_sessions_dir)

    def resolve_grokbot_transcripts_dirs(self) -> list[Path]:
        from self_directing_mcp.grokbot.discover import parse_transcripts_dirs

        override = os.environ.get("SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR")
        raw = override if override is not None else self.grokbot_transcripts_dir
        return parse_transcripts_dirs(raw)

    def resolve_session_provider(self) -> Literal["codex", "grokbot"]:
        override = (os.environ.get("SELF_DIRECT_SESSION_PROVIDER") or "").strip().lower()
        if override in ("codex", "grokbot"):
            return override  # type: ignore[return-value]
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
