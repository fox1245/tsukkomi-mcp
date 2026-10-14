from __future__ import annotations

from collections import OrderedDict
import json
import hashlib
from pathlib import Path
import threading

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


def loosening_reasons(previous: ContractRule, new: ContractRule) -> list[str]:
    """Why replacing the active ``previous`` by ``new`` may enforce less; ``[]`` when it cannot.

    Whether a regex got narrower is undecidable here, so any change of what a contract matches counts, as does every
    field that restricts where or when it applies (moving ``applies_from_event_id`` is not judged: it is how a new
    revision says "from now on"). Only strictly tightening edits (a new verifier for a description-only
    contract, an added role, widening the scope to ``any``, adding an obligation flag, removing an exception, enabling)
    pass without a user anchor. A contract that is not enabled enforces nothing, so editing it weakens nothing.
    """
    if not previous.enabled:
        return []
    reasons = []
    if not new.enabled:
        reasons.append("disabled")
    if new.type != previous.type:
        reasons.append("type changed")
    if previous.regex and new.regex != previous.regex:
        reasons.append("regex changed")
    if new.before_regex != previous.before_regex:
        reasons.append("before_regex changed")
    if new.exception_regex and new.exception_regex != previous.exception_regex:
        reasons.append("exception added or changed")
    if new.scope != previous.scope and new.scope != "any":
        reasons.append("scope narrowed")
    if set(previous.roles) - set(new.roles):
        reasons.append("roles narrowed")
    if new.when is not None and new.when != previous.when:
        reasons.append("applicability condition (when) added or changed")
    if new.provider is not None and new.provider != previous.provider:
        reasons.append("narrowed to one provider")
    if new.session_id is not None and new.session_id != previous.session_id:
        reasons.append("narrowed to one session")
    if previous.requires_success and not new.requires_success:
        reasons.append("success requirement removed")
    if previous.risk == "high" and new.risk != "high":
        reasons.append("risk lowered")
    if previous.auto_continue and not new.auto_continue:
        reasons.append("auto_continue removed")
    return reasons


class ContentMemo:
    """A small thread-safe LRU whose KEY IS THE CONTENT its values were computed from.

    Used for derivations of a file's exact bytes (a validated contracts document, the obligations read out of one):
    a changed byte is a different key, so a stale entry can never be returned, and nothing needs invalidating. Errors
    are not cached (``put`` is only called with a finished value). Values are shared between callers and must be
    treated as read-only; whoever edits a document validates a private copy of its own.
    """

    def __init__(self, size: int) -> None:
        self._size, self._items, self._lock = size, OrderedDict(), threading.Lock()

    def get(self, key):
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def put(self, key, value) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self._size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


MEMO_MAX_BYTES = 1024 * 1024   # a larger contracts file is validated every time rather than held (bounds what the memos keep)
_DOCUMENTS = ContentMemo(8)


def parse_contracts(raw: bytes | None) -> ContractsDocument:
    """The validated contracts document of these exact bytes (an empty one for ``None``).

    Read-only callers only: the document is shared by everyone who reads the same bytes, so a reader must not modify
    it or hand its rules to code that does. ``ContractStore.load`` and the upsert/revoke paths, which edit, validate
    their own copy. Invalid bytes raise ``ValueError`` on every call (failures are not remembered).
    """
    if raw is None:
        return ContractsDocument()
    memoize = len(raw) <= MEMO_MAX_BYTES
    document = _DOCUMENTS.get(raw) if memoize else None
    if document is None:
        document = ContractsDocument.model_validate_json(raw)
        if memoize:
            _DOCUMENTS.put(raw, document)
    return document


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

    def seed_missing(self, preset_path: Path) -> int:
        """Add preset contracts whose ids are absent; never alter or re-enable an existing contract.

        Presets are shipped disabled, so adding them changes no verdict until the owner enables one.
        """
        preset_path = Path(preset_path)
        if not preset_path.exists():
            return 0
        presets = ContractsDocument.model_validate(json.loads(preset_path.read_text(encoding="utf-8"))).contracts
        known = {rule.id for rule in self._doc.contracts}
        added = [rule for rule in presets if rule.id not in known]
        if added:
            self._doc.contracts.extend(added)
            self.save()
        return len(added)
