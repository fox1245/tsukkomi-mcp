"""Agent reviews of unverified high-risk calls (``SELF_DIRECT_CLAUDE_ENFORCEMENT=agent_review``). Standard library only.

In ``confirmed_only`` a proposed call whose ``risk: high`` contract cannot be verified asks the user. In ``agent_review``
the hook refuses that call once and tells the agent how to judge it; the agent records its judgement with the MCP tool
``review_action``, and the retried call runs once on a ``proceed`` review (an ``escalate`` review asks the user). This
module is the store both sides share and the few pure functions they must agree on:

* ``action_digest``: the identity of a call, a SHA-256 over its tool name and its payload (``tool_payload.digest_payload``:
  only fields declared prose for that exact tool, such as ``Bash``'s ``description`` in Claude Code, are left out).
* ``refusals``: every refusal the hook issued (a random ``refusal_id``, the session, the digest, the contracts and their
  revisions, the finding, when it expires). A review can only name one of these, so a call that was never refused cannot
  be reviewed.
* ``reviews``: at most one per refusal: the decision, the agent's reason (masked) and a digest of it, the evidence ids,
  when it expires, and when the hook consumed it or rejected it. A review is consumed once.

The store lives in ``<index_dir>/action-reviews.sqlite``, never in a transcript. It never holds a call's arguments: only
their digest and a short masked rendering. A review never changes a finding: the finding stays ``unknown``, and the hook's
note and this store say that the call ran on the agent's decision.

``CLOCK`` is the time source (seconds since the epoch); tests replace it instead of sleeping.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import time
from typing import Any

from self_directing_mcp.tool_payload import digest_payload

STORE_FILE = "action-reviews.sqlite"
DECISIONS = ("proceed", "escalate")
REVIEW_TOOL = "review_action"              # this server's tool; a host names it ``mcp__<server>__review_action``
REVIEW_TOOL_PARAMS = frozenset({
    "session_id", "refusal_id", "action_digest", "contract_ids", "decision", "reason", "evidence_event_ids",
})
RESERVED_FILE = Path(__file__).resolve().parent / "resources" / "agent_review_reserved.txt"
RESERVED_MAX_BYTES = 8192                   # the guidance is a short list shown to the agent
REASON_MAX_CHARS = 4000
SHOWN_REASON_CHARS = 1000
MAX_IDS = 32

CLOCK = time.time


def now() -> float:
    return CLOCK()


def action_digest(provider: str, tool_name: Any, arguments: Any) -> str:
    """The identity of a proposed call: its tool name and payload in canonical JSON (exact-tool prose fields left out)."""
    payload = digest_payload(provider, tool_name, arguments)
    canonical = json.dumps({"tool_name": tool_name, "arguments": payload}, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def reason_digest(reason: str) -> str:
    return hashlib.sha256(reason.encode("utf-8")).hexdigest()


def is_review_tool(tool_name: Any, server_name: str | None = None) -> bool:
    """Whether a host tool name is an MCP server's ``review_action`` (``mcp__<server>__review_action``;
    ``mcp__<server_name>__review_action`` when ``server_name`` is set)."""
    if not isinstance(tool_name, str) or not tool_name.startswith("mcp__"):
        return False
    parts = tool_name.split("__")
    if not (len(parts) >= 3 and all(parts[1:-1]) and parts[-1] == REVIEW_TOOL):
        return False
    if server_name is not None:
        return isinstance(server_name, str) and bool(server_name) and tool_name == f"mcp__{server_name}__{REVIEW_TOOL}"
    return True


def permission_rejected(decision: Any) -> bool:
    """Whether a tool result's ``permission_decision`` metadata records a host or hook rejection."""
    if isinstance(decision, str):
        return decision in ("deny", "reject")
    return isinstance(decision, dict) and (
        decision.get("decision") in ("reject", "deny") or decision.get("behavior") in ("deny", "reject")
    )


def result_matches(text: str, review: dict) -> bool:
    """Whether a stored ``tool_result`` text is a succeeded ``review_action`` reply naming ``review['review_id']``."""
    prefix = "[tool_result] "
    review_id = review.get("review_id")
    if not isinstance(text, str) or not text.startswith(prefix) or not isinstance(review_id, str) or not review_id:
        return False
    try:
        payload = json.loads(text[len(prefix):])
    except ValueError:
        return False
    if not isinstance(payload, dict) or payload.get("ok") is False or "error" in payload:
        return False
    if payload.get("review_id") != review_id:
        return False
    if "refusal_id" in payload and payload["refusal_id"] != review.get("refusal_id"):
        return False
    return True


def call_arguments(text: str, tool_name: str) -> dict | None:
    """The arguments of a stored tool call (``[tool_call:<name>] <json>``), or None when the text is not one."""
    prefix = f"[tool_call:{tool_name}] "
    if not isinstance(text, str) or not text.startswith(prefix):
        return None
    try:
        value = json.loads(text[len(prefix):])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def call_matches(arguments: dict, review: dict) -> bool:
    """Whether a stored ``review_action`` call's arguments are the ones that recorded ``review`` (its refusal's session,
    id, digest and contracts, its decision and, through its digest, its reason)."""
    ids = arguments.get("contract_ids")
    if isinstance(ids, str):              # a client that sent the list as JSON text
        try:
            ids = json.loads(ids)
        except ValueError:
            return False
    reason = arguments.get("reason")
    return (arguments.get("session_id") == review["session_id"] and arguments.get("refusal_id") == review["refusal_id"]
            and arguments.get("action_digest") == review["action_digest"] and arguments.get("decision") == review["decision"]
            and isinstance(reason, str) and reason_digest(reason) == review["reason_digest"]
            and isinstance(ids, list) and all(isinstance(item, str) for item in ids)
            and sorted(set(ids)) == sorted(cid for cid, _rev in review["contracts"]))


def reserved_classes(configured: Path | None) -> tuple[list[str], str | None]:
    """``(entries, note)``: the actions the user reserved for themselves, as guidance for the agent.

    Data the user edits: ``configured`` (``SELF_DIRECT_AGENT_REVIEW_RESERVED_FILE``) or the packaged default. One entry
    per line; blank lines and lines that start with ``#`` are skipped. It is shown to the agent and never matched against a
    call. A configured file that cannot be read is said in ``note`` and the packaged default is shown instead."""
    note = None
    if configured is not None:
        entries = _read_entries(Path(configured))
        if entries is not None:
            return entries, None
        note = "the configured list of reserved actions could not be read; the packaged default is shown"
    entries = _read_entries(RESERVED_FILE)
    return (entries or []), note


def _read_entries(path: Path) -> list[str] | None:
    try:
        with path.open("rb") as stream:
            raw = stream.read(RESERVED_MAX_BYTES + 1)
        text = raw[:RESERVED_MAX_BYTES].decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def _pairs(raw: str) -> list[tuple[str, int]]:
    return [(str(cid), int(rev)) for cid, rev in json.loads(raw)]


def covers(reviewed: list[tuple[str, int]], current: list[tuple[str, int]]) -> bool:
    """Whether a review of ``reviewed`` (contract id, revision) covers every contract unverified now."""
    return set(current) <= set(reviewed)


class ReviewStore:
    """The refusals the hook issued and the reviews recorded for them. Raises ``sqlite3.Error``/``OSError`` on storage
    failures: callers decide what an unusable store means (the hook asks the user, the tool reports an error)."""

    def __init__(self, index_dir) -> None:
        self.path = Path(index_dir) / STORE_FILE

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=2.0, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS refusals (
                    refusal_id TEXT PRIMARY KEY, host TEXT NOT NULL, session_id TEXT NOT NULL, agent_id TEXT NOT NULL,
                    action_digest TEXT NOT NULL, tool_name TEXT NOT NULL, action_preview TEXT NOT NULL,
                    contracts TEXT NOT NULL, finding TEXT NOT NULL, source_offset INTEGER NOT NULL,
                    ttl_sec REAL NOT NULL, issued_at REAL NOT NULL, expires_at REAL NOT NULL,
                    review_call_passes INTEGER NOT NULL DEFAULT 0);
                CREATE INDEX IF NOT EXISTS refusals_action ON refusals(host, session_id, agent_id, action_digest);
                CREATE TABLE IF NOT EXISTS reviews (
                    review_id TEXT PRIMARY KEY, refusal_id TEXT NOT NULL UNIQUE, decision TEXT NOT NULL,
                    reason TEXT NOT NULL, reason_digest TEXT NOT NULL, evidence TEXT NOT NULL,
                    recorded_at REAL NOT NULL, expires_at REAL NOT NULL,
                    consumed_at REAL, origin_event_id TEXT, rejected_at REAL, rejected_reason TEXT);""")
        except BaseException:
            conn.close()
            raise
        return conn

    # --- the hook ---------------------------------------------------------------------------------------------------

    def refuse(self, *, host: str, session_id: str, agent_id: str | None, digest: str, tool_name: str, preview: str,
               contracts: list[tuple[str, int]], finding: str, source_offset: int, ttl_sec: float) -> dict:
        """Record a refusal-for-review and return it. The id is random: nobody can name it before the hook issued it."""
        issued = now()
        row = {"refusal_id": secrets.token_hex(12), "host": host, "session_id": session_id, "agent_id": agent_id or "",
               "action_digest": digest, "tool_name": tool_name, "action_preview": preview,
               "contracts": json.dumps([[cid, rev] for cid, rev in contracts]), "finding": finding,
               "source_offset": int(source_offset), "ttl_sec": float(ttl_sec), "issued_at": issued,
               "expires_at": issued + float(ttl_sec), "review_call_passes": 0}
        with closing(self._connect()) as conn:
            conn.execute("INSERT INTO refusals VALUES (:refusal_id,:host,:session_id,:agent_id,:action_digest,:tool_name,"
                         ":action_preview,:contracts,:finding,:source_offset,:ttl_sec,:issued_at,:expires_at,"
                         ":review_call_passes)", row)
        return {**row, "contracts": _pairs(row["contracts"])}

    def candidates(self, *, host: str, session_id: str, agent_id: str | None, digest: str,
                   contracts: list[tuple[str, int]]) -> list[dict]:
        """Unused, unexpired reviews of refusals of this very call (same host, session, agent and digest) that cover
        every contract unverified now, newest first."""
        with closing(self._connect()) as conn:
            stamp = now()
            rows = conn.execute(
                "SELECT r.*, f.contracts AS contracts, f.source_offset AS source_offset, f.session_id AS session_id,"
                " f.action_digest AS action_digest"
                " FROM reviews r JOIN refusals f ON f.refusal_id = r.refusal_id"
                " WHERE f.host=? AND f.session_id=? AND f.agent_id=? AND f.action_digest=?"
                " AND f.issued_at <= ? AND r.recorded_at <= ?"
                " AND r.consumed_at IS NULL AND r.rejected_at IS NULL AND r.expires_at > ?"
                " ORDER BY r.recorded_at DESC, r.rowid DESC",
                (host, session_id, agent_id or "", digest, stamp, stamp, stamp)).fetchall()
        found = []
        for row in rows:
            item = {**dict(row), "contracts": _pairs(row["contracts"])}
            if covers(item["contracts"], contracts):
                found.append(item)
        return found

    def consume(self, review_id: str, origin_event_id: str) -> bool:
        """Spend a review once. False when it was used, rejected, expired or recorded in the future."""
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                stamp = now()
                changed = conn.execute(
                    "UPDATE reviews SET consumed_at=?, origin_event_id=? WHERE review_id=? AND consumed_at IS NULL"
                    " AND rejected_at IS NULL AND recorded_at <= ? AND expires_at > ?"
                    " AND EXISTS (SELECT 1 FROM refusals f WHERE f.refusal_id = reviews.refusal_id AND f.issued_at <= ?)",
                    (stamp, origin_event_id, review_id, stamp, stamp, stamp)).rowcount
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return changed == 1

    def reject(self, review_id: str, why: str) -> None:
        """Mark a review unusable (its call is not a call of the refused agent's own turn); kept for the audit trail."""
        with closing(self._connect()) as conn:
            conn.execute("UPDATE reviews SET rejected_at=?, rejected_reason=? WHERE review_id=? AND consumed_at IS NULL"
                         " AND rejected_at IS NULL", (now(), why[:200], review_id))

    def open_refusal_for(self, *, host: str, session_id: str, agent_id: str | None, tool_name: Any,
                         arguments: Any, server_name: str | None = None) -> dict | None:
        """The open refusal a proposed ``review_action`` call records a review for, counted as a pass; else None.

        The call must target the review tool (``mcp__<server_name>__review_action`` when ``server_name`` is set), carry
        only ``review_action`` parameters, and name this session, the hook's own random ``refusal_id``, the refused
        call's ``action_digest``, the refusal's exact ``contract_ids`` and a valid ``decision`` while that refusal has
        no review and has not expired."""
        if not is_review_tool(tool_name, server_name) or not isinstance(arguments, dict):
            return None
        if not set(arguments) <= REVIEW_TOOL_PARAMS or arguments.get("session_id") != session_id:
            return None
        refusal_id, digest = arguments.get("refusal_id"), arguments.get("action_digest")
        decision, contract_ids = arguments.get("decision"), arguments.get("contract_ids")
        if (not isinstance(refusal_id, str) or not refusal_id or not isinstance(digest, str) or not digest
                or not isinstance(decision, str) or decision not in DECISIONS
                or not isinstance(contract_ids, list)
                or any(not isinstance(item, str) or not item for item in contract_ids)):
            return None
        if "reason" in arguments and not isinstance(arguments["reason"], str):
            return None
        evidence = arguments.get("evidence_event_ids")
        if evidence is not None and (not isinstance(evidence, list)
                                     or any(not isinstance(item, str) or not item for item in evidence)):
            return None
        contracts: list[tuple[str, int]] = []
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                stamp = now()
                row = conn.execute(
                    "SELECT f.* FROM refusals f LEFT JOIN reviews r ON r.refusal_id = f.refusal_id"
                    " WHERE f.refusal_id=? AND f.host=? AND f.session_id=? AND f.agent_id=? AND f.action_digest=?"
                    " AND f.issued_at <= ? AND f.expires_at > ? AND r.review_id IS NULL",
                    (refusal_id, host, session_id, agent_id or "", digest, stamp, stamp)).fetchone()
                if row is not None:
                    contracts = _pairs(row["contracts"])
                    if sorted(contract_ids) != sorted(cid for cid, _rev in contracts):
                        row = None
                    else:
                        conn.execute("UPDATE refusals SET review_call_passes = review_call_passes + 1"
                                     " WHERE refusal_id=?", (refusal_id,))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        if row is None:
            return None
        return {**dict(row), "contracts": contracts, "review_call_passes": int(row["review_call_passes"]) + 1}

    # --- the review tool ------------------------------------------------------------------------------------------------

    def refusal(self, refusal_id: str) -> dict | None:
        if not self.path.exists():
            return None
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT f.*, r.review_id AS review_id FROM refusals f"
                               " LEFT JOIN reviews r ON r.refusal_id = f.refusal_id WHERE f.refusal_id=?",
                               (refusal_id,)).fetchone()
        return None if row is None else {**dict(row), "contracts": _pairs(row["contracts"])}

    def record_review(self, *, refusal_id: str, decision: str, reason_shown: str, reason_sha: str,
                      evidence: list[str]) -> dict | str:
        """Record the one review of a refusal; returns the row, or an error code when the refusal is gone, expired or
        already reviewed. Its expiry is the refusal's ``ttl_sec`` (the hook's setting) after it is recorded."""
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                refusal = conn.execute("SELECT * FROM refusals WHERE refusal_id=?", (refusal_id,)).fetchone()
                if refusal is None:
                    conn.execute("ROLLBACK")
                    return "refusal_not_found"
                recorded = now()
                if refusal["expires_at"] <= recorded:
                    conn.execute("ROLLBACK")
                    return "refusal_expired"
                if conn.execute("SELECT 1 FROM reviews WHERE refusal_id=?", (refusal_id,)).fetchone():
                    conn.execute("ROLLBACK")
                    return "refusal_already_reviewed"
                row = {"review_id": secrets.token_hex(12), "refusal_id": refusal_id, "decision": decision,
                       "reason": reason_shown, "reason_digest": reason_sha, "evidence": json.dumps(evidence),
                       "recorded_at": recorded, "expires_at": recorded + float(refusal["ttl_sec"])}
                conn.execute("INSERT INTO reviews (review_id, refusal_id, decision, reason, reason_digest, evidence,"
                             " recorded_at, expires_at) VALUES (:review_id,:refusal_id,:decision,:reason,:reason_digest,"
                             ":evidence,:recorded_at,:expires_at)", row)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return row

    # --- the audit trail ------------------------------------------------------------------------------------------------

    def listing(self, *, session_id: str | None = None, limit: int = 50) -> dict:
        """Every refusal with its review (newest first, at most ``limit``) and counts over the whole store."""
        if not self.path.exists():
            return {"counts": _counts([], now()), "items": []}
        where, params = ("WHERE f.session_id=?", [session_id]) if session_id else ("", [])
        with closing(self._connect()) as conn:
            rows = [dict(row) for row in conn.execute(
                "SELECT f.*, r.review_id, r.decision, r.reason, r.evidence, r.recorded_at, r.expires_at AS review_expires_at,"
                " r.consumed_at, r.origin_event_id, r.rejected_at, r.rejected_reason"
                f" FROM refusals f LEFT JOIN reviews r ON r.refusal_id = f.refusal_id {where}"
                " ORDER BY f.issued_at DESC, f.rowid DESC", params)]
        stamp = now()
        return {"counts": _counts(rows, stamp), "items": [_item(row, stamp) for row in rows[:max(1, int(limit))]]}


def _iso(epoch: float | None) -> str | None:
    return None if epoch is None else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _counts(rows: list[dict], stamp: float) -> dict:
    reviewed = [row for row in rows if row.get("review_id")]
    return {
        "refusals": len(rows),
        "refusals_open": sum(1 for row in rows if not row.get("review_id") and row["expires_at"] > stamp),
        "refusals_expired_unreviewed": sum(1 for row in rows if not row.get("review_id") and row["expires_at"] <= stamp),
        "reviews": {decision: sum(1 for row in reviewed if row["decision"] == decision) for decision in DECISIONS},
        "consumed": {decision: sum(1 for row in reviewed if row["decision"] == decision and row["consumed_at"] is not None)
                     for decision in DECISIONS},
        "rejected": sum(1 for row in reviewed if row["rejected_at"] is not None),
        "expired_unused": sum(1 for row in reviewed if row["consumed_at"] is None and row["rejected_at"] is None
                              and row["review_expires_at"] <= stamp),
        "review_call_passes": sum(int(row.get("review_call_passes") or 0) for row in rows),
    }


def _item(row: dict, stamp: float) -> dict:
    review = None
    if row.get("review_id"):
        state = ("consumed" if row["consumed_at"] is not None else "rejected" if row["rejected_at"] is not None
                 else "expired" if row["review_expires_at"] <= stamp else "unused")
        review = {"review_id": row["review_id"], "decision": row["decision"], "reason": row["reason"],
                  "evidence_event_ids": json.loads(row["evidence"] or "[]"), "recorded_at": _iso(row["recorded_at"]),
                  "expires_at": _iso(row["review_expires_at"]), "state": state, "consumed_at": _iso(row["consumed_at"]),
                  "origin_event_id": row["origin_event_id"], "rejected_at": _iso(row["rejected_at"]),
                  "rejected_reason": row["rejected_reason"]}
    return {"refusal_id": row["refusal_id"], "host": row["host"], "session_id": row["session_id"],
            "agent_id": row["agent_id"] or None, "issued_at": _iso(row["issued_at"]),
            "expires_at": _iso(row["expires_at"]),
            "contracts": [{"id": cid, "revision": rev} for cid, rev in _pairs(row["contracts"])],
            "finding": row["finding"], "action_digest": row["action_digest"], "tool_name": row["tool_name"],
            "action": row["action_preview"], "review_call_passes": int(row.get("review_call_passes") or 0),
            "review": review}
