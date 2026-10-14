from __future__ import annotations

import json
import hashlib
from pathlib import Path

from self_directing_mcp.schemas import ContractRule, ContractsDocument
from self_directing_mcp.request_control import check_request, publication


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

    def capture(self) -> tuple[bytes | None, str]:
        check_request()
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return None, "missing"
        return raw, "present:" + hashlib.sha256(raw).hexdigest()

    def current_token(self) -> str:
        return self.capture()[1]

    def load(self) -> ContractsDocument:
        raw, _token = self.capture()
        # Never retain previously valid authority when the file is lost/corrupt.
        self._doc = ContractsDocument()
        if raw is not None:
            self._doc = ContractsDocument.model_validate_json(raw)
        return self._doc

    def save(self) -> None:
        self.publish_document(self._doc, self._doc.model_dump_json(indent=2))

    def publish_document(self, document, payload, expected_token=None):
        check_request()
        if expected_token is not None and self.current_token() != expected_token:
            raise ValueError("stale contract preparation")
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temporary.write_text(payload, encoding="utf-8")
            with publication():
                if expected_token is not None and self.current_token() != expected_token:
                    raise ValueError("contract authority changed before publication")
                check_request()
                temporary.replace(self.path)
            self._doc = document
        finally:
            temporary.unlink(missing_ok=True)

    def list(self) -> list[ContractRule]:
        return [rule.model_copy(deep=True) for rule in self._doc.contracts]

    def load_if_changed(self) -> None:
        self.load()

    @staticmethod
    def prepare_upsert(document, rules):
        by_id = {c.id: c for c in document.contracts}
        saved = []
        for original in rules:
            check_request()
            r = original.model_copy(deep=True)
            if r.id in by_id:
                previous = by_id[r.id]
                r = r.model_copy(update={"revision": previous.revision + 1})
                if not r.source_quote and previous.source_quote:
                    r = r.model_copy(update={"source_quote": previous.source_quote})
                document.history.setdefault(r.id, []).append(previous)
            by_id[r.id] = r
            saved.append(r)
        document.contracts = list(by_id.values())
        return saved

    def upsert(self, rules: list[ContractRule]) -> list[ContractRule]:
        document = self._doc.model_copy(deep=True)
        saved = self.prepare_upsert(document, rules)
        self.publish_document(document, document.model_dump_json(indent=2))
        return [rule.model_copy(deep=True) for rule in saved]

    @staticmethod
    def prepare_revoke(document, rule_id):
        by_id = {c.id: c for c in document.contracts}
        rule = by_id.get(rule_id)
        if rule is None:
            return None
        disabled = rule.model_copy(update={"enabled": False}, deep=True)
        document.history.setdefault(rule_id, []).append(rule)
        by_id[rule_id] = disabled
        document.contracts = list(by_id.values())
        return disabled

    def revoke(self, rule_id: str) -> ContractRule | None:
        """Disable a rule; history and source quote remain for audit."""
        document = self._doc.model_copy(deep=True)
        disabled = self.prepare_revoke(document, rule_id)
        if disabled is not None:
            self.publish_document(document, document.model_dump_json(indent=2))
        return disabled.model_copy(deep=True) if disabled is not None else None

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
