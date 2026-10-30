"""Contract anchors that live in another session.

A contract's ``source_event_id`` (the user message it came from) and ``applies_from_event_id`` (where it starts to
apply) name events of the index. A contract of one session finds them among that session's events. A global contract
(``session_id`` unset) is judged in every session, so its anchors usually belong to the session it was written in:
registration resolves them across the whole index (``SelfDirectEngine._anchor_status``), and so does the audit, once per
anchor and build, with ``resolve``. Both evaluators then take the resolved ``Anchor`` instead of looking for the event
among the session's own events: the reference (``runner._audit_contract``) and the incremental monitor
(``incremental.RuleMonitor``) read the same facts and apply the same rule, so they agree.

The facts of an anchor are fixed by its event (a stored event never changes), except that an event can appear in the
index later; ``fingerprint`` is part of a monitor's stored-state key for that reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from self_directing_mcp.request_control import check_request


def is_user_event(event) -> bool:
    """A user message: what the parser marked ``role == "user"`` (never runtime, injected or tool text)."""
    return (getattr(event, "meta", None) or {}).get("role") == "user"


def anchor_status(chunk) -> tuple[str, str]:
    """``(status, detail)`` of a resolved anchor event: ``user`` (a user message) or ``other``.

    An anchor is the event a contract was derived from, so it must be the user's own message: a tool result, runtime
    text or injected context cannot grant or revoke authority, whatever role the host rendered it with.
    """
    if is_user_event(chunk):
        return "user", "user message"
    return "other", f"a {chunk.kind} event with role {(chunk.meta or {}).get('role')!r}"


def truncated_user(meta) -> bool:
    """An AGY user message whose content the host cut."""
    return meta.get("event_type") == "USER_INPUT" and "content" in (meta.get("truncated_fields") or ())


def parse_time(value) -> datetime | None:
    """The instant a host timestamp names, or ``None`` when it cannot be compared (missing, not ISO 8601, no offset)."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None


@dataclass(frozen=True)
class Anchor:
    """An anchor event of another session: whether it is a user message, whether its content was cut, and when it was."""
    event_id: str
    user: bool
    cut_user: bool
    at: datetime | None

    def key(self) -> list:
        return [self.event_id, self.user, self.cut_user, self.at.isoformat() if self.at is not None else None]


def in_window(at: datetime, event) -> bool:
    """Whether ``event`` is not before ``at``. An event whose time cannot be compared is inside (judge more, not less)."""
    moment = parse_time(getattr(event, "timestamp", None))
    return moment is None or moment >= at


def foreign_anchors(rule, anchors) -> dict:
    """The resolved anchors that apply to ``rule``: only a global contract takes its anchors from another session."""
    if not anchors or rule.session_id is not None:
        return {}
    return {event_id: anchors[event_id] for event_id in (rule.source_event_id, rule.applies_from_event_id)
            if event_id and event_id in anchors}


def fingerprint(rule, anchors) -> list | None:
    """What a stored monitor state of ``rule`` depends on besides the rule itself, or ``None`` when nothing."""
    found = foreign_anchors(rule, anchors)
    return [found[event_id].key() for event_id in sorted(found)] or None


def wants_times(rules, anchors) -> bool:
    """Whether some rule's window starts at a foreign time: only then do events have to carry their timestamps."""
    for rule in rules:
        found = foreign_anchors(rule, anchors).get(rule.applies_from_event_id) if rule.applies_from_event_id else None
        if found is not None and found.at is not None:
            return True
    return False


def resolve(store, rules, session_id, provider) -> dict:
    """``{event id: Anchor}`` for the anchors of the global ``rules`` that belong to another session.

    One lookup per anchor, with the store's ``get_chunk`` (what registration uses). An anchor that is in this session, or
    in no session of the index, is left out: the evaluators judge those as before. ``store`` must be the session store
    and the call must be inside the engine's read boundary.
    """
    found: dict[str, Anchor] = {}
    wanted = dict.fromkeys(event_id for rule in rules if rule.session_id is None
                           for event_id in (rule.source_event_id, rule.applies_from_event_id) if event_id)
    for event_id in wanted:
        check_request()
        chunk = store.get_chunk(event_id)
        if chunk is None or (chunk.session_id == session_id and chunk.provider == provider):
            continue
        found[event_id] = Anchor(event_id, is_user_event(chunk), truncated_user(chunk.meta or {}),
                                 parse_time(chunk.timestamp))
    return found
