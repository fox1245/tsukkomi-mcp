from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict, model_serializer, model_validator


SCHEMA_VERSION = "1.0"

ChunkKind = Literal["turn", "tool_call", "tool_result", "policy", "meta"]
ContractRuleType = Literal["must", "must_not"]
ContractScope = Literal["tool_call", "tool_result", "message", "any"]
Severity = Literal["critical", "high", "medium", "low"]
ContractRisk = Literal["normal", "high"]
Applicability = Literal["applicable", "not_applicable"]
Verdict = Literal["violation", "suspicious", "clean", "unknown"]
SearchMode = Literal["hybrid", "regex", "sparse", "dense"]


class Chunk(BaseModel):
    chunk_id: str
    session_id: str
    provider: Literal["codex", "grokbot", "omp", "agy", "claude"] = "codex"
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
_OPTIONAL_DEFAULTS = {"risk": "normal", "auto_continue": False, "when": None, "origin": None}


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
        description="Primary verifier regex over scoped text. Required for temporal rules.",
    )
    # Hybrid auxiliary: description/query used for semantic+sparse support (never sole evidence for violation)
    search_query: str | None = Field(
        default=None,
        description="Auxiliary hybrid retrieval query; never sole evidence for violation.",
    )
    enabled: bool = True
    revision: int = Field(default=1, ge=1, description="Set on update; do not send manually.")
    provider: Literal["codex", "grokbot", "omp", "agy", "claude"] | None = Field(
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

    @model_serializer(mode="wrap")
    def _omit_default_extensions(self, handler):
        data = handler(self)
        for name, default in _OPTIONAL_DEFAULTS.items():
            if name in data and data[name] == default:
                del data[name]
        return data

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
    ranking_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    kind: str | None = None
    snippet: str | None = None
    timestamp: str | None = None
