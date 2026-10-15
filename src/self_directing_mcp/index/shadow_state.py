"""SQL for the shadow-comparison schedule and the review ranges (see ``audit.shadow``).

Kept apart from ``ingest`` so the store only has to call these functions with its connection. Both tables are derived
data in the sense of ``audit_monitor_state``: losing a row loses a schedule (the policy starts again at 1/4 with its
warm-up, which compares more, never less) or a report, never an answer.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3

SCHEMA = """
            -- The adaptive shadow-comparison schedule of a session (audit/shadow.py): one small JSON row. ``epoch`` is
            -- bumped on every reset and an older epoch never overwrites a newer one.
            CREATE TABLE IF NOT EXISTS audit_shadow_state (
                provider TEXT NOT NULL, session_id TEXT NOT NULL, epoch INTEGER NOT NULL, state TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(provider,session_id)
            );
            -- Ranges of requests that were answered from incremental state between two comparisons when a comparison
            -- then found a difference. Advice for a reviewer, never a verdict.
            CREATE TABLE IF NOT EXISTS audit_review_ranges (
                id INTEGER PRIMARY KEY AUTOINCREMENT, provider TEXT NOT NULL, session_id TEXT NOT NULL,
                detected_at TEXT NOT NULL, source TEXT NOT NULL, since_at TEXT, since_offset INTEGER,
                until_offset INTEGER, requests INTEGER, contracts TEXT NOT NULL, detail TEXT NOT NULL
            );
"""

KEEP_RANGES = 20

_UPSERT = (
    "INSERT INTO audit_shadow_state VALUES (?,?,?,?,?)"
    " ON CONFLICT(provider,session_id) DO UPDATE SET epoch=excluded.epoch, state=excluded.state,"
    " updated_at=excluded.updated_at WHERE audit_shadow_state.epoch <= excluded.epoch")


def load(conn, provider, session_id):
    """``(epoch, state dict)``; ``None`` when there is no row (or it cannot be read at all). The dict is ``None`` when the
    row's text is not a JSON object: its epoch is still known, so that a fresh state can supersede it."""
    try:
        row = conn.execute("SELECT epoch,state FROM audit_shadow_state WHERE provider=? AND session_id=?",
                           (provider, session_id)).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    try:
        data = json.loads(row[1])
    except ValueError:
        return row[0], None
    return row[0], data if isinstance(data, dict) else None


def write(conn, provider, session_id, epoch, state, now=None):
    """Inside an open write transaction."""
    conn.execute(_UPSERT, (provider, session_id, epoch, json.dumps(state, separators=(",", ":")),
                           now or datetime.now(timezone.utc).isoformat()))


def add_range(conn, provider, session_id, *, source, since_at, since_offset, until_offset, requests, contracts, detail,
              now=None):
    """Inside an open write transaction."""
    conn.execute(
        "INSERT INTO audit_review_ranges (provider,session_id,detected_at,source,since_at,since_offset,until_offset,"
        "requests,contracts,detail) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (provider, session_id, now or datetime.now(timezone.utc).isoformat(), source, since_at, since_offset,
         until_offset, requests, json.dumps(sorted(contracts)), detail[:2000]))
    conn.execute(
        "DELETE FROM audit_review_ranges WHERE provider=? AND session_id=? AND id NOT IN ("
        " SELECT id FROM audit_review_ranges WHERE provider=? AND session_id=? ORDER BY id DESC LIMIT ?)",
        (provider, session_id, provider, session_id, KEEP_RANGES))


def ranges(conn, provider, session_id):
    try:
        rows = conn.execute(
            "SELECT detected_at,source,since_at,since_offset,until_offset,requests,contracts,detail"
            " FROM audit_review_ranges WHERE provider=? AND session_id=? ORDER BY id DESC",
            (provider, session_id)).fetchall()
    except sqlite3.Error:
        return []
    return [{"detected_at": r[0], "source": r[1], "since_at": r[2], "since_offset": r[3], "until_offset": r[4],
             "requests": r[5], "contracts": json.loads(r[6]), "detail": r[7]} for r in rows]
