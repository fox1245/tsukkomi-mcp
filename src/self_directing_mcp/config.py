from __future__ import annotations

from pathlib import Path
from typing import Annotated, Callable, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, PydanticBaseSettingsSource, SettingsConfigDict

from self_directing_mcp import settings_core
from self_directing_mcp.settings_core import SettingsAccessors, get_config_path  # noqa: F401  (public name here)

# The sources and defaults below live in ``settings_core`` (standard library only), shared with the light reader
# the command hooks use; these names stay importable from here for existing callers.
_PATH_FIELDS = settings_core.PATH_FIELDS
_toml_path = settings_core.toml_path
_toml_path_settings = settings_core.toml_path_settings
_codex_environment_settings = settings_core.codex_environment_settings
_default_codex_sessions_dir = settings_core.default_codex_sessions_dir
_default_grokbot_transcripts_dir = settings_core.default_grokbot_transcripts_dir
_default_omp_sessions_dir = settings_core.default_omp_sessions_dir
_default_claude_projects_dir = settings_core.default_claude_projects_dir
_default_index_dir = settings_core.default_index_dir


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_sqlite_vector_path() -> Path | None:
    return settings_core.default_sqlite_vector_path(_repo_root())    # tests relocate the checkout through _repo_root


class Settings(SettingsAccessors, BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SELF_DIRECT_",
        env_file=".env",
        extra="ignore",
    )

    codex_sessions_dir: Path = Field(default_factory=_default_codex_sessions_dir)
    grokbot_transcripts_dir: str | list[Path] = Field(default_factory=_default_grokbot_transcripts_dir)
    omp_sessions_dir: Path = Field(default_factory=_default_omp_sessions_dir)
    claude_projects_dir: Path = Field(default_factory=_default_claude_projects_dir)
    agy_app_data_dirs: list[Path] = Field(default_factory=settings_core.default_agy_app_data_dirs)
    agy_enforcement: Literal["confirmed_only", "advisory", "enforced"] = settings_core.DEFAULT_ENFORCEMENT
    claude_enforcement: Literal["confirmed_only", "advisory", "enforced"] = settings_core.DEFAULT_ENFORCEMENT
    session_provider: Literal["codex", "grokbot", "omp", "agy", "claude"] = settings_core.DEFAULT_SESSION_PROVIDER
    index_dir: Path = Field(default_factory=_default_index_dir)
    contracts_path: Path | None = Field(
        default=None,
        description="Optional JSON contracts file; default index_dir/contracts.json",
    )
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    embedding_model: str = settings_core.DEFAULT_EMBEDDING_MODEL
    embedding_dim: int = settings_core.DEFAULT_EMBEDDING_DIM
    embedding_base_url: str = settings_core.DEFAULT_EMBEDDING_BASE_URL
    use_fake_embedder: bool = False
    local_only: bool = False
    dense_backend: Literal["sqlite-vector", "numpy"] = settings_core.DEFAULT_DENSE_BACKEND
    sqlite_vector_path: Path | None = Field(default_factory=_default_sqlite_vector_path)
    openrouter_api_key_file: Path | None = Field(
        default=None,
        description="Authorized dotenv file containing OPENROUTER_API_KEY; never log its contents",
    )
    seed_example_contracts: bool = False
    lock_wait_timeout_sec: float = Field(default=settings_core.DEFAULT_LOCK_WAIT_TIMEOUT_SEC, gt=0)
    audit_timeout_sec: float = Field(default=settings_core.DEFAULT_AUDIT_TIMEOUT_SEC, gt=0)
    request_timeout_sec: float = Field(default=settings_core.DEFAULT_REQUEST_TIMEOUT_SEC, gt=0)
    graph_update_timeout_sec: float = Field(default=settings_core.DEFAULT_GRAPH_UPDATE_TIMEOUT_SEC, gt=0,
                                            allow_inf_nan=False)
    graph_extractor: Literal["openrouter", "host"] = Field(
        default=settings_core.DEFAULT_GRAPH_EXTRACTOR,
        description="Who extracts for run_neograph_update: 'openrouter' (default) calls an OpenRouter model; 'host' "
                    "disables that tool (error extractor_disabled, no key and no network touched) so that only the "
                    "calling agent extracts, through graph_next_batch and graph_submit_batch, which work in both "
                    "modes. SELF_DIRECT_GRAPH_EXTRACTOR.",
    )
    max_tool_timeout_sec: float = Field(
        default=settings_core.DEFAULT_MAX_TOOL_TIMEOUT_SEC,
        gt=0,
        description="Upper bound for client-provided timeout_ms on long-running tools.",
    )
    max_concurrent_calls: int = Field(
        default=settings_core.DEFAULT_MAX_CONCURRENT_CALLS,
        ge=1,
        description="Max heavy tool calls (sync/search/audit/checklist/hooks) running at once; "
                    "excess requests queue within their own deadline. SELF_DIRECT_MAX_CONCURRENT_CALLS.",
    )
    max_decompressed_session_bytes: int = Field(
        default=settings_core.DEFAULT_MAX_DECOMPRESSED_BYTES,
        ge=1,
        description="Upper bound on the declared decompressed size of a zstd-compressed Codex session "
                    "(rollout-*.jsonl.zst). A larger file is reported as zstd_too_large and not decoded. "
                    "SELF_DIRECT_MAX_DECOMPRESSED_SESSION_BYTES.",
    )
    integrity_fast_recent_blocks: int = Field(
        default=settings_core.DEFAULT_INTEGRITY_FAST_RECENT_BLOCKS, ge=0,
        description="Fast (hook-time) source check: the latest complete 1 MiB blocks always compared with their "
                    "stored digests. SELF_DIRECT_INTEGRITY_FAST_RECENT_BLOCKS.",
    )
    integrity_fast_sample_blocks: int = Field(
        default=settings_core.DEFAULT_INTEGRITY_FAST_SAMPLE_BLOCKS, ge=0,
        description="Fast source check: random other blocks compared, drawn fresh for every call. "
                    "SELF_DIRECT_INTEGRITY_FAST_SAMPLE_BLOCKS.",
    )
    integrity_full_verify_interval_sec: float = Field(
        default=settings_core.DEFAULT_INTEGRITY_FULL_VERIFY_INTERVAL_SEC, ge=0, allow_inf_nan=False,
        description="A fast check is allowed only this long after the last full verification of every block; "
                    "after that the next check is full (0: always full). SELF_DIRECT_INTEGRITY_FULL_VERIFY_INTERVAL_SEC.",
    )
    integrity_background_verify: bool = Field(
        default=True,
        description="Resident MCP servers re-verify watched sources in full on one background thread while idle. "
                    "Short-lived hook processes never do. SELF_DIRECT_INTEGRITY_BACKGROUND_VERIFY.",
    )
    integrity_background_idle_sec: float = Field(
        default=settings_core.DEFAULT_INTEGRITY_BACKGROUND_IDLE_SEC, gt=0, allow_inf_nan=False,
        description="Idle time before the background verification starts. SELF_DIRECT_INTEGRITY_BACKGROUND_IDLE_SEC.",
    )
    integrity_background_max_scans: int = Field(
        default=settings_core.DEFAULT_INTEGRITY_BACKGROUND_MAX_SCANS, ge=1,
        description="Most background full verifications started per wake-up (they run one at a time, never "
                    "concurrently). SELF_DIRECT_INTEGRITY_BACKGROUND_MAX_SCANS.",
    )
    audit_incremental: bool = Field(
        default=True,
        description="Evaluate contracts incrementally from stored per-contract monitor state (see README, 'Incremental "
                    "audits'). false evaluates every audit over the whole session. SELF_DIRECT_AUDIT_INCREMENTAL.",
    )
    audit_shadow_every: Annotated[int, Field(ge=0)] | Literal["adaptive"] = Field(
        default=settings_core.DEFAULT_AUDIT_SHADOW_EVERY,
        description="How often the full evaluator also runs beside the incremental one and the two are compared (see "
                    "README, 'Incremental audits'). 'adaptive' (default): the rate starts at 1/4 and halves after every "
                    "8 consecutive matching comparisons down to 1/32, and returns to 1/4 when the contracts, the parser "
                    "or evaluator version, the session generation or a mismatch change the premise; comparisons are also "
                    "forced after a rebuild and for the next 4 compared requests, for a violation/suspicious finding, "
                    "when a violation turns clean and for a high-risk contract. An integer pins the fixed legacy rate and switches adaptation off: 1 compares "
                    "every audit, 0 only the forced comparisons, N about one audit in N. A mismatch returns the full "
                    "result and disables incremental evaluation of that contract in that session until the evaluator "
                    "version changes. SELF_DIRECT_AUDIT_SHADOW_EVERY.",
    )
    audit_shadow_idle_sec: float = Field(
        default=settings_core.DEFAULT_AUDIT_SHADOW_IDLE_SEC, ge=0, allow_inf_nan=False,
        description="Resident servers (the daemon, the MCP stdio server) compare the latest incremental state of a "
                    "session that answered requests without a comparison against a full evaluation after this many idle "
                    "seconds (0 switches it off). Hook processes never do. SELF_DIRECT_AUDIT_SHADOW_IDLE_SEC.",
    )
    audit_incremental_min_events: int = Field(
        default=settings_core.DEFAULT_AUDIT_INCREMENTAL_MIN_EVENTS,
        ge=0,
        description="Sessions with fewer stored events are evaluated in full without keeping monitor state: below "
                    "this size a full evaluation costs less than storing, rebuilding and shadow-checking states. "
                    "0 keeps state for every session. SELF_DIRECT_AUDIT_INCREMENTAL_MIN_EVENTS.",
    )
    rrf_k: int = settings_core.DEFAULT_RRF_K
    retrieve_top_k: int = settings_core.DEFAULT_RETRIEVE_TOP_K
    search_top_k: int = settings_core.DEFAULT_SEARCH_TOP_K
    daemon: Literal["off", "on"] = Field(
        default=settings_core.DEFAULT_DAEMON,
        description="Resident daemon (experimental, Linux only): 'on' makes command hooks and the MCP entry point "
                    "thin clients of one per-user process; 'off' (default) is the standalone mode. The entry points "
                    "read it from the process environment before importing anything else. SELF_DIRECT_DAEMON.",
    )
    sqlite_journal: Literal["auto", "delete"] = Field(
        default=settings_core.DEFAULT_SQLITE_JOURNAL,
        description="Journal mode of the authority databases (meta.sqlite, agy-receipts.sqlite). 'auto': write-ahead "
                    "logging on a verified local file system (ext2/3/4, xfs, btrfs, tmpfs, f2fs, zfs, a local overlay; a "
                    "fixed Windows drive), the rollback journal everywhere else; 'delete': the rollback journal everywhere. "
                    "Always synchronous=FULL. SELF_DIRECT_SQLITE_JOURNAL.",
    )
    sync_slicing: Literal["auto", "off"] = Field(
        default=settings_core.DEFAULT_SYNC_SLICING,
        description="A sync that cannot finish inside its request budget publishes the records it parsed (a slice, ending "
                    "at a record boundary) and the next call continues from there. 'auto': for requests with a finite "
                    "deadline and plain session files (a re-index is staged in a spool and published by one commit, never "
                    "in pieces over an existing index); 'off': the all-or-nothing sync of before. "
                    "SELF_DIRECT_SYNC_SLICING.",
    )
    sync_slice_bytes: int = Field(
        default=settings_core.DEFAULT_SYNC_SLICE_BYTES, ge=1,
        description="A slice observes (hashes, parses) at most this many bytes of a plain session file past the "
                    "indexed offset, to the end of a record (at least one whole record). Bounds the capture of a source "
                    "that is far from indexed. SELF_DIRECT_SYNC_SLICE_BYTES.",
    )

    project_contracts: Literal["candidates", "off"] = Field(
        default=settings_core.DEFAULT_PROJECT_CONTRACTS,
        description="'candidates' (default): list_contract_candidates and audit_status find the project's "
                    ".tsukkomi/contracts.json (from the project directory up to the repository root) as candidates; "
                    "nothing is enforced until adopt_contracts adopts it with a user message of this device. 'off': "
                    "the project bundle is not looked for. Personal bundles are unaffected. "
                    "SELF_DIRECT_PROJECT_CONTRACTS.",
    )
    contract_bundle_paths: Annotated[list[Path], NoDecode] = Field(
        default_factory=list,
        description="Personal contract bundle files (absolute paths separated by the platform path separator in the "
                    "environment; an array in the TOML [paths] table), also the only places besides the project "
                    "directory that export_contracts may write. Candidates only, like the project bundle. "
                    "SELF_DIRECT_CONTRACT_BUNDLE_PATHS.",
    )

    @field_validator("contract_bundle_paths", mode="before")
    @classmethod
    def _bundle_path_list(cls, value):
        # The environment spelling is one string; the TOML source and callers pass lists. Either way: absolute paths.
        if isinstance(value, str):
            return settings_core.parse_bundle_paths("contract_bundle_paths", value)
        if isinstance(value, (list, tuple)):
            paths = [Path(item).expanduser() if isinstance(item, (str, Path)) else item for item in value]
            if any(not isinstance(item, Path) or not item.is_absolute() for item in paths):
                raise ValueError("invalid absolute path for contract_bundle_paths")
            return paths
        return value

    @field_validator("audit_shadow_every", mode="before")
    @classmethod
    def _adaptive_spelling(cls, value):
        # 'adaptive' in any case and with surrounding whitespace, like the light reader (settings_core).
        if isinstance(value, str) and value.strip(settings_core._WHITESPACE).lower() == "adaptive":
            return "adaptive"
        return value

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


def hook_settings() -> Settings:
    """Settings for lifecycle hooks: environment and the user TOML only, never a ``.env`` in the working directory.

    Hosts run hooks in the project being worked on, so a ``.env`` there is content the audited agent can write.
    Reading it would let that agent point the hook at an empty index or another mode and silence its own checks.
    The hook state modules resolve the same keys from the same two sources.

    This is the reference implementation. The command hooks themselves read ``settings_core.load_hook_settings()``,
    which applies the same sources, precedence and errors without importing pydantic-settings (about 30 ms per hook
    process); ``tests/test_hook_settings_parity.py`` keeps the two in agreement.
    """
    return Settings(_env_file=None)


def get_settings(**overrides) -> Settings:
    # The key file is read by resolve_api_key() for OPENROUTER_API_KEY only. It is never loaded into
    # os.environ, where subprocesses and native hosts would inherit every variable it holds.
    return Settings(**overrides)
