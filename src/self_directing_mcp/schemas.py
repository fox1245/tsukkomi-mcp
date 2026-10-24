from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_serializer, model_validator

from self_directing_mcp import contract_paths


SCHEMA_VERSION = "1.0"

ChunkKind = Literal["turn", "tool_call", "tool_result", "policy", "meta"]
ContractRuleType = Literal["must", "must_not"]
ContractScope = Literal["tool_call", "tool_result", "message", "any"]
Severity = Literal["critical", "high", "medium", "low"]
ContractRisk = Literal["normal", "high"]
ContractMode = Literal["enforced", "inject_only"]
Applicability = Literal["applicable", "not_applicable"]
Verdict = Literal["violation", "suspicious", "clean", "unknown"]
SearchMode = Literal["hybrid", "regex", "sparse", "dense"]


class Chunk(BaseModel):
    chunk_id: str
    session_id: str
    provider: Literal["codex", "grokbot", "omp", "agy", "claude", "grok"] = "codex"
    kind: ChunkKind
    text: str
    content_hash: str
    timestamp: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    byte_start: int | None = None
    byte_end: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class AuditEvent:
    """Read-only slim event for audits: identity, kind, role/call linkage and (only if needed) text.

    ``text`` is empty for events no contract can inspect; nothing else of the stored chunk is kept.
    """
    __slots__ = ("chunk_id", "kind", "text", "meta", "key")

    def __init__(self, chunk_id, kind, text, meta, key=None):
        # ``key`` is the event's place in the session order, ``(byte_start, sub_index)``; incremental monitors
        # compare keys, never row numbers. Events built without a store carry ``None``.
        self.chunk_id, self.kind, self.text, self.meta, self.key = chunk_id, kind, text, meta, key


class ContractWhen(BaseModel):
    """Optional applicability condition for a ``must`` obligation.

    The obligation applies only if some tool call observed earlier in the same session matches
    ``tool_call_regex`` (case-insensitive, multiline). A contract whose condition is not met is reported
    as ``not_applicable`` with this reason; it is never dropped. Without a condition nothing changes.
    """
    model_config = ConfigDict(extra="forbid")
    tool_call_regex: str = Field(
        min_length=1,
        description="Regex over tool calls observed in this session; the obligation applies only after one matched.",
    )


# Fields added after 1.0. They are omitted from serialized contracts while at their default so files written
# by this version stay loadable by older processes (extra="forbid") and snapshots of unchanged contracts hold.
_OPTIONAL_DEFAULTS = {"risk": "normal", "auto_continue": False, "when": None, "origin": None, "path_guard": None, "mode": "enforced", "tools": None}


class ContractOrigin(BaseModel):
    """Where an adopted contract came from. Written only by ``adopt_contracts``; informational, never authority.

    The contract was loaded from a portable bundle file and became an ordinary stored contract when the user adopted
    it on this device: ``adopted_by_event_id`` is the local user message that did so (checked against this device's
    index when the adoption happened). ``bundle_path`` is a relative or labelled path, never an absolute one. The
    ``claimed_*`` values were written in the file, by whoever wrote it: they are not verified and not an anchor.
    """
    model_config = ConfigDict(extra="forbid")
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256 of the bundle file's bytes.")
    bundle_label: str = Field(max_length=80)
    origin_kind: Literal["project", "personal"]
    bundle_path: str | None = Field(default=None, max_length=300)
    adopted_by_event_id: str = Field(min_length=1, max_length=400)
    claimed_origin_event_id: str | None = Field(default=None, max_length=400)
    claimed_quote: str | None = Field(default=None, max_length=500)

    @model_serializer(mode="wrap")
    def _omit_empty_claims(self, handler):
        data = handler(self)
        for name in ("bundle_path", "claimed_origin_event_id", "claimed_quote"):
            if data.get(name) is None:
                data.pop(name, None)
        return data


class PathGuard(BaseModel):
    """A typed ``must_not`` verifier that names PATHS instead of a regex (see ``contract_paths`` for the semantics).

    Stored, hashed, exported and compared in this form. The contract regex every evaluator runs is derived from it
    when the contract is loaded, through the versioned per-host tool map, and is never written back.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    paths: list[str] = Field(
        description="Paths or globs the contract guards: 1 to 16 entries of at most 256 characters. A bare name "
                    "('.env') matches that file name in any directory; an entry that starts with '/', '~/' or a drive "
                    "letter is anchored ('~/' also matches $HOME and the home directories); '*' stays inside a path "
                    "segment, '**' crosses segments, '?' and '[...]' work as in a shell; forward and back slashes are "
                    "equivalent; the match is case-insensitive.")
    except_: list[str] = Field(
        default_factory=list, alias="except",
        description="Paths or globs that are NOT guarded although 'paths' matches them, for example '.env.example'.")
    modes: Literal["any", "read"] = Field(
        default="any",
        description="'any' (default): the path appears as a path argument of any tool call (a read, a write, a patch, "
                    "a shell word). 'read': the path is read (a file-read tool, or a reader inside a shell command). "
                    "'write' is not supported in this version.")

    @field_validator("paths", mode="before")
    @classmethod
    def _check_paths(cls, value):
        return list(_checked(contract_paths.check_entries, "paths", value, required=True))

    @field_validator("except_", mode="before")
    @classmethod
    def _check_except(cls, value):
        return list(_checked(contract_paths.check_entries, "except", value, required=False))

    @field_validator("modes", mode="before")
    @classmethod
    def _check_modes(cls, value):
        return _checked(contract_paths.check_modes, value)

    @model_serializer(mode="plain")
    def _typed_form(self):
        return contract_paths.as_dict(self)


def _checked(check, *args, **kwargs):
    """Run a ``contract_paths`` check; its error already names the field."""
    try:
        return check(*args, **kwargs)
    except contract_paths.PathGuardError as exc:
        raise ValueError(str(exc)) from None


class ContractRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(description="Stable unique contract id used for updates (revision increments).")
    type: ContractRuleType = Field(
        description="'must' = required user requirement; 'must_not' = prohibited action/effect."
    )
    scope: ContractScope = Field(
        default="any",
        description="Where the rule applies: tool_call, tool_result, message or any.",
    )
    severity: Severity = Field(default="medium", description="critical|high|medium|low.")
    description: str = Field(
        default="",
        description="Human-readable meaning. Description-only rules audit as unknown until a verifier exists.",
    )
    regex: str | None = Field(
        default=None,
        description="Primary verifier regex over scoped text. Required for temporal rules. Omit it when "
                    "path_guard names the paths: the regex is then derived from path_guard and never stored.",
    )
    path_guard: PathGuard | None = Field(
        default=None,
        description="Typed alternative to regex for a must_not contract about files (scope tool_call): names the "
                    "paths, and the server matches them through every host's tool vocabulary (Claude Code Read, "
                    "AGY view_file, OMP read, shell readers such as cat, patches). {paths, except, modes}. "
                    "path_guard and regex cannot be combined.",
    )
    # Hybrid auxiliary: description/query used for semantic+sparse support (never sole evidence for violation)
    search_query: str | None = Field(
        default=None,
        description="Auxiliary hybrid retrieval query; never sole evidence for violation.",
    )
    enabled: bool = True
    revision: int = Field(default=1, ge=1, description="Set on update; do not send manually.")
    provider: Literal["codex", "grokbot", "omp", "agy", "claude", "grok"] | None = Field(
        default=None, description="Limit to one provider; omit for all."
    )
    session_id: str | None = Field(
        default=None,
        description="Session-scoped rule id. Null means a global rule that applies to every audited session.",
    )
    source_event_id: str | None = Field(
        default=None, description="History event this rule was derived from."
    )
    applies_from_event_id: str | None = Field(
        default=None, description="Exclusive anchor: rule applies to events after this id."
    )
    roles: list[str] = Field(
        default_factory=lambda: ["assistant"],
        description="Roles whose events the rule applies to. Must be non-empty.",
    )
    before_regex: str | None = Field(
        default=None,
        description="Temporal guard for must tool_call rules: this action must appear earlier.",
    )
    requires_success: bool = Field(
        default=False,
        description="For must tool_call rules: require a successful (non-error) earlier call.",
    )
    source_quote: str | None = Field(
        default=None,
        description="User's original wording this rule was derived from; preserved across context compaction.",
    )
    exception_regex: str | None = Field(
        default=None,
        description="User-authorized exception pattern for must_not rules; a match lowers violation to suspicious pending confirmation.",
    )
    risk: ContractRisk = Field(
        default="normal",
        description="'high' declares that actions this contract applies to are high-risk. Hosts may ask the user "
                    "instead of silently allowing when such an action cannot be verified.",
    )
    auto_continue: bool = Field(
        default=False,
        description="For must completion obligations only: a confirmed, fully observed miss may resume a stopping "
                    "agent once per session. Unknown evidence never resumes.",
    )
    when: ContractWhen | None = Field(
        default=None,
        description="Optional applicability condition for must obligations; see ContractWhen. "
                    "Unmet conditions report not_applicable with a reason.",
    )
    origin: ContractOrigin | None = Field(
        default=None,
        description="Read-only: set by adopt_contracts when the contract came from a portable bundle file. "
                    "upsert_contracts rejects an origin that the stored contract does not already have.",
    )
    mode: ContractMode = Field(
        default="enforced",
        description="'enforced' (default): evaluated by audit_session and check_action. 'inject_only': injected at "
                    "SessionStart/PreInvocation as a reminder, never evaluated as a violation or unknown.",
    )
    tools: list[str] | None = Field(
        default=None,
        description="Optional list of tool names (case-insensitive, e.g. ['Bash', 'run_command']) that a tool_call/any "
                    "contract applies to. Tool calls for other tools are ignored / not_applicable.",
    )

    @model_serializer(mode="wrap")
    def _omit_default_extensions(self, handler):
        data = handler(self)
        for name, default in _OPTIONAL_DEFAULTS.items():
            if name in data and data[name] == default:
                del data[name]
        if data.get("path_guard") is not None:
            data.pop("regex", None)          # derived from path_guard when loaded: the typed form is what is stored
        return data

    @model_validator(mode="before")
    @classmethod
    def _typed_inputs(cls, data):
        if isinstance(data, dict):
            if data.get("path_guard") is not None:
                if data.get("regex") is not None:
                    raise ValueError("path_guard and regex cannot both be set on one contract: path_guard is the single "
                                     "source of truth and the regex is derived from it")
                if "scope" not in data:
                    data = {**data, "scope": "tool_call"}
            elif data.get("tools") is not None and "scope" not in data:
                data = {**data, "scope": "tool_call"}
        return data

    @model_validator(mode="after")
    def lower_path_guard(self):
        """The one place a typed guard becomes the regex every evaluator reads (see ``contract_paths``)."""
        if self.path_guard is None:
            return self
        if self.type != "must_not":
            raise ValueError("path_guard applies only to must_not contracts (a must contract, or one with before_regex, "
                             "requires_success or auto_continue, takes a regex)")
        if self.scope != "tool_call":
            raise ValueError(f"path_guard needs scope 'tool_call' (it is matched against the text of tool calls), "
                             f"got {self.scope!r}")
        try:
            self.regex = contract_paths.lower(self.path_guard.paths, self.path_guard.except_, self.path_guard.modes)
        except contract_paths.PathGuardError as exc:
            raise ValueError(f"path_guard.{exc}") from None
        return self

    # https://docs.pydantic.dev/latest/concepts/validators/ (Pydantic 2)
    @model_validator(mode="after")
    def validate_temporal_shape(self):
        if self.before_regex and (self.type != "must" or self.scope != "tool_call" or not self.regex):
            raise ValueError("before_regex requires a must tool_call rule with regex")
        if self.requires_success and (self.type != "must" or self.scope != "tool_call" or not self.regex):
            raise ValueError("requires_success requires a must tool_call rule with regex")
        if not self.roles:
            raise ValueError("roles must explicitly name at least one role")
        if self.exception_regex and self.type != "must_not":
            raise ValueError("exception_regex applies only to must_not rules")
        if self.auto_continue and (self.type != "must" or not self.regex or self.before_regex):
            raise ValueError("auto_continue requires a must completion obligation with regex and no before_regex")
        if self.when is not None and self.type != "must":
            raise ValueError("when applies only to must rules")
        if self.tools is not None:
            if self.scope not in ("tool_call", "any"):
                raise ValueError("tools filter requires scope 'tool_call' or 'any'")
            if not (1 <= len(self.tools) <= 32):
                raise ValueError("tools must contain between 1 and 32 entries")
            cleaned_tools: list[str] = []
            seen_tools: set[str] = set()
            for item in self.tools:
                if not isinstance(item, str) or not item.strip() or len(item.strip()) > 80:
                    raise ValueError("tools entries must be non-empty strings up to 80 chars")
                norm = item.strip()
                if norm.lower() not in seen_tools:
                    seen_tools.add(norm.lower())
                    cleaned_tools.append(norm)
            self.tools = cleaned_tools
        return self


class ProposedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] | str


class PendingChange(BaseModel):
    """A weakening of an active contract that no user message vouches for; the active revision stays in force.

    An upsert that disables a contract or replaces it with a narrower one (see ``audit.contracts.loosening_reasons``)
    needs a ``source_event_id`` that resolves to a user message. Without one the proposed revision is parked here and
    the previous revision keeps being enforced. A later upsert (or revoke) of the same id that carries a user anchor
    applies the change and clears this entry.
    """
    contract: ContractRule
    reasons: list[str]
    source_event_id: str | None = None
    anchor_status: Literal["missing", "unresolved"] = "missing"
    base_revision: int = 1


class ContractsDocument(BaseModel):
    schema_version: str = SCHEMA_VERSION
    contracts: list[ContractRule] = Field(default_factory=list)
    history: dict[str, list[ContractRule]] = Field(default_factory=dict)
    # Weakenings awaiting a user anchor, by contract id. Omitted while empty: files that never held one are unchanged
    # and older processes (which ignore the key) keep reading the others.
    pending: dict[str, PendingChange] = Field(default_factory=dict)

    @model_serializer(mode="wrap")
    def _omit_empty_pending(self, handler):
        data = handler(self)
        if not data.get("pending"):
            data.pop("pending", None)
        return data


class Evidence(BaseModel):
    chunk_id: str
    kind: ChunkKind | str
    snippet: str
    match_type: Literal["regex", "sparse", "dense", "hybrid", "rule"]
    score: float | None = None
    detail: str | None = None


class Finding(BaseModel):
    contract_id: str
    severity: Severity
    verdict: Verdict
    reason: str
    basis: str | None = Field(
        default=None,
        description="Which rule condition matched or which evidence is missing; empty for legacy findings.",
    )
    evidence: list[Evidence] = Field(default_factory=list)
    contract_revision: int = 1
    applicability: Applicability = Field(
        default="applicable",
        description="not_applicable: the contract's `when` condition is not met (verdict stays clean); "
                    "the contract is reported, never dropped.",
    )
    proposals_rejected: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Calls matching a must_not contract that the host recorded as refused (a deny decision linked to "
                    "the call): proposals, not executed actions. They never make the verdict a violation; omitted "
                    "while empty.",
    )

    @model_serializer(mode="wrap")
    def _omit_empty_rejections(self, handler):
        data = handler(self)
        if not data.get("proposals_rejected"):
            data.pop("proposals_rejected", None)
        return data


class AuditResult(BaseModel):
    execution: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    session_id: str
    verdict: Verdict
    findings: list[Finding] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    coverage: dict[str, Any] = Field(default_factory=dict)
    contracts_snapshot: str | None = None


class SearchHit(BaseModel):
    chunk_id: str
    session_id: str | None = None
    ranking_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    kind: str | None = None
    snippet: str | None = None
    timestamp: str | None = None
