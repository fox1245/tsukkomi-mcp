from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict, model_validator


SCHEMA_VERSION = "1.0"

ChunkKind = Literal["turn", "tool_call", "tool_result", "policy", "meta"]
ContractRuleType = Literal["must", "must_not"]
ContractScope = Literal["tool_call", "tool_result", "message", "any"]
Severity = Literal["critical", "high", "medium", "low"]
Verdict = Literal["violation", "suspicious", "clean", "unknown"]
SearchMode = Literal["hybrid", "regex", "sparse", "dense"]


class Chunk(BaseModel):
    chunk_id: str
    session_id: str
    provider: Literal["codex", "grokbot", "omp"] = "codex"
    kind: ChunkKind
    text: str
    content_hash: str
    timestamp: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    byte_start: int | None = None
    byte_end: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


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
    provider: Literal["codex", "grokbot", "omp"] | None = Field(
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
        return self


class ProposedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] | str


class ContractsDocument(BaseModel):
    schema_version: str = SCHEMA_VERSION
    contracts: list[ContractRule] = Field(default_factory=list)
    history: dict[str, list[ContractRule]] = Field(default_factory=dict)


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
