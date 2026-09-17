from __future__ import annotations

import json
from pathlib import Path

from self_directing_mcp.schemas import ContractRule, ContractsDocument


GLOBAL_CONTRACT_MEANING = (
    "session_id is null; the contract is global and applies to every audited session"
)


def visible_contracts(rules: list[ContractRule], session_id: str | None = None) -> list[ContractRule]:
    if session_id is None:
        return [rule for rule in rules if rule.session_id is None]
    return [rule for rule in rules if rule.session_id is None or rule.session_id == session_id]


def contract_scope(session_id: str | None) -> str:
    return "session_and_global" if session_id else "global_only"


def scope_counts(selected: list[ContractRule], store_count: int) -> dict[str, int]:
    return {
        "returned": len(selected),
        "session": sum(1 for rule in selected if rule.session_id is not None),
        "global": sum(1 for rule in selected if rule.session_id is None),
        "store": store_count,
    }


class ContractStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._doc = ContractsDocument()
        if self.path.exists():
            self.load()

    def load(self) -> ContractsDocument:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._doc = ContractsDocument.model_validate(data)
        return self._doc

    def save(self) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            self._doc.model_dump_json(indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def list(self) -> list[ContractRule]:
        return list(self._doc.contracts)

    def load_if_changed(self) -> None:
        if self.path.exists():
            self.load()

    def upsert(self, rules: list[ContractRule]) -> list[ContractRule]:
        by_id = {c.id: c for c in self._doc.contracts}
        saved: list[ContractRule] = []
        for r in rules:
            if r.id in by_id:
                previous = by_id[r.id]
                r = r.model_copy(update={"revision": previous.revision + 1})
                # Preserve the user's original wording unless the update provides one.
                if not r.source_quote and previous.source_quote:
                    r = r.model_copy(update={"source_quote": previous.source_quote})
                self._doc.history.setdefault(r.id, []).append(previous)
            by_id[r.id] = r
            saved.append(r)
        self._doc.contracts = list(by_id.values())
        self.save()
        return saved

    def revoke(self, rule_id: str) -> ContractRule | None:
        """Disable a rule; history and source quote remain for audit."""
        by_id = {c.id: c for c in self._doc.contracts}
        rule = by_id.get(rule_id)
        if rule is None:
            return None
        disabled = rule.model_copy(update={"enabled": False})
        self._doc.history.setdefault(rule_id, []).append(rule)
        by_id[rule_id] = disabled
        self._doc.contracts = list(by_id.values())
        self.save()
        return disabled

    def seed_from(self, seed_path: Path) -> int:
        """Load contracts from a seed JSON if store is empty."""
        if self._doc.contracts:
            return 0
        seed_path = Path(seed_path)
        if not seed_path.exists():
            return 0
        data = json.loads(seed_path.read_text(encoding="utf-8"))
        doc = ContractsDocument.model_validate(data)
        self._doc = doc
        self.save()
        return len(doc.contracts)
