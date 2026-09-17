"""User requirement checklist: per-requirement tracking with provenance.

Requirements are split into independently verifiable items, each keeping the
user's original wording and source message. Statuses distinguish planned,
in-progress, awaiting-verification, done (evidence-backed) and blocked.
Completion claims alone never mark an item done; evidence is required.
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class RequirementStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING_VERIFICATION = "pending_verification"
    DONE = "done"
    BLOCKED = "blocked"
    REVOKED = "revoked"


Status = Literal["open", "in_progress", "pending_verification", "done", "blocked", "revoked"]


class RequirementItem(BaseModel):
    item_id: str = Field(description="Stable checklist item id (req-001 style).")
    text: str = Field(description="Verbatim user requirement fragment this item tracks.")
    source_event_id: str | None = Field(default=None, description="History event the requirement came from.")
    source_message: str | None = Field(default=None, description="Masked snippet of the user's original message.")
    status: Status = "open"
    satisfaction_criteria: str | None = Field(default=None, description="What counts as satisfied.")
    evidence_chunk_ids: list[str] = Field(default_factory=list, description="Chunks proving the requirement was met.")
    verified: bool = Field(default=False, description="True only when evidence was linked and checked.")
    blocked_reason: str | None = Field(default=None)
    history: list[dict[str, Any]] = Field(default_factory=list, description="Status/provenance change log.")


class ChecklistDocument(BaseModel):
    schema_version: str = "1.0"
    session_id: str
    items: list[RequirementItem] = Field(default_factory=list)

    def next_item_id(self) -> str:
        return f"req-{len(self.items) + 1:03d}"


class ChecklistStore:
    """Per-session JSON checklist stored under the local index dir."""

    def __init__(self, index_dir: Path) -> None:
        self.dir = Path(index_dir) / "checklists"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id)[:120]
        return self.dir / f"{safe}.json"

    def load(self, session_id: str) -> ChecklistDocument:
        path = self._path(session_id)
        if path.exists():
            return ChecklistDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))
        return ChecklistDocument(session_id=session_id)

    def save(self, doc: ChecklistDocument) -> None:
        path = self._path(doc.session_id)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)


INCOMPLETE_NOTE = (
    "Checklist items without verified evidence must not be claimed complete. "
    "Attach evidence_chunk_ids from tool results or tests before marking done."
)
