"""Activity time-series analysis over the existing SQLite event store.

Pure SQL (window functions) over session_events: retry patterns, failure
rates per window, stalled checklist items, pre/post-instruction behavior,
post-test modifications and fix->reverify sequences. Observations only —
temporal adjacency never proves cause; completeness flags accompany results.
"""
from __future__ import annotations

from typing import Any


ANALYSIS_SCHEMA_VERSION = 1


def analyze_activity(store, *, session_id: str, provider: str,
                     window_minutes: int = 60) -> dict[str, Any]:
    """Return activity analytics with original-evidence ids and completeness."""
    rows = store.event_frames(session_id, provider)
    notes = ["Observations only; time order does not prove cause or violation."]
    if not rows:
        notes.append("No indexed events for this session; run sync_session first.")
        return {"ok": True, "schema_version": ANALYSIS_SCHEMA_VERSION,
                "session_id": session_id, "provider": provider,
                "event_count": 0, "data_complete": False, "notes": notes,
                "repeated_errors": [], "failure_rate_by_window": [],
                "stalled_requirements": [], "behavior_around_instructions": [],
                "post_test_modifications": [], "fix_reverify_sequences": []}
    # Chunk ids carry order; timestamps may be missing and are never fabricated.
    missing_time = sum(1 for r in rows if not r["timestamp"])
    if missing_time:
        notes.append(f"{missing_time} event(s) lack timestamps; window stats cover only timestamped events.")
    window_seconds = window_minutes * 60
    return {
        "ok": True,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "session_id": session_id,
        "provider": provider,
        "event_count": len(rows),
        "data_complete": missing_time == 0,
        "notes": notes,
        "repeated_errors": _repeated_errors(rows),
        "failure_rate_by_window": _failure_rate(rows, window_seconds),
        "stalled_requirements": _stalled_requirements(store, session_id, provider),
        "behavior_around_instructions": _around_instructions(rows),
        "post_test_modifications": _post_test_modifications(rows),
        "fix_reverify_sequences": _fix_reverify(rows),
    }


def _repeated_errors(rows) -> list[dict[str, Any]]:
    """Same error signature repeated, with retry intervals in event positions."""
    errors = [r for r in rows if r["kind"] == "tool_result" and r["success"] is False]
    by_signature: dict[str, list[dict[str, Any]]] = {}
    for r in errors:
        signature = (r["tool_name"] or "") + ":" + (r["error_text"] or "")[:120]
        by_signature.setdefault(signature, []).append(r)
    out = []
    for signature, group in by_signature.items():
        if len(group) < 2:
            continue
        gaps = [group[i]["order"] - group[i - 1]["order"] for i in range(1, len(group))]
        out.append({"signature": signature[:200], "count": len(group),
                    "retry_gaps_in_events": gaps,
                    "first_event": group[0]["chunk_id"], "last_event": group[-1]["chunk_id"],
                    "evidence_chunk_ids": [r["chunk_id"] for r in group[:20]]})
    out.sort(key=lambda item: -item["count"])
    return out[:10]


def _failure_rate(rows, window_seconds) -> list[dict[str, Any]]:
    """Failure rate per time window for timestamped tool results."""
    results = [r for r in rows if r["kind"] == "tool_result" and r["timestamp"]]
    if not results:
        return []
    try:
        from datetime import datetime
        parsed = []
        for r in results:
            try:
                ts = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
                parsed.append((ts.timestamp(), r["success"] is False, r))
            except ValueError:
                continue
    except Exception:
        return []
    if not parsed:
        return []
    start = min(t for t, _, _ in parsed)
    buckets: dict[int, dict[str, int]] = {}
    for t, failed, _ in parsed:
        key = int((t - start) // window_seconds)
        slot = buckets.setdefault(key, {"total": 0, "failures": 0})
        slot["total"] += 1
        slot["failures"] += int(failed)
    return [{"window_start_offset_sec": key * window_seconds,
             "tool_results": slot["total"],
             "failures": slot["failures"],
             "failure_rate": round(slot["failures"] / slot["total"], 4)}
            for key, slot in sorted(buckets.items())]


def _stalled_requirements(store, session_id, provider) -> list[dict[str, Any]]:
    """Checklist items long open or without new evidence (from checklist store)."""
    stalled = []
    checklist_path = store.checklist_items(session_id, provider)
    for item in checklist_path:
        if item.get("status") in ("done", "revoked"):
            continue
        is_stalled = item.get("status") in ("open", "in_progress") and not item.get("evidence_chunk_ids")
        if is_stalled or item.get("status") == "blocked":
            stalled_item = {"item_id": item["item_id"], "text": item["text"][:200],
                            "status": item["status"]}
            if item.get("blocked_reason"):
                stalled_item["blocked_reason"] = item["blocked_reason"]
            stalled_item["note"] = "no evidence attached yet" if not item.get("evidence_chunk_ids") else item.get("blocked_reason")
            stalled.append(stalled_item)
    return stalled


def _around_instructions(rows) -> list[dict[str, Any]]:
    """Events immediately before/after instruction-type turns (add/modify/revoke)."""
    out = []
    for i, r in enumerate(rows):
        text = (r["text"] or "").lower()
        if not any(k in text for k in ("하지 마", "금지", "하지 말고", "do not", "never", "must not")):
            continue
        before = [rows[j]["chunk_id"] for j in range(max(0, i - 3), i)]
        after = [rows[j]["chunk_id"] for j in range(i + 1, min(i + 4, len(rows)))]
        out.append({"instruction_event": r["chunk_id"], "snippet": (r["text"] or "")[:160],
                    "events_before": before, "events_after": after if (after := [rows[j]["chunk_id"] for j in range(i + 1, min(i + 4, len(rows)))]) else []})
    return out[:20]


def _post_test_modifications(rows) -> list[dict[str, Any]]:
    """File edits after a successful test pass: re-verification may be needed."""
    out = []
    test_ok_positions = [i for i, r in enumerate(rows)
                         if r["kind"] == "tool_result" and r["success"] is True
                         and any(k in (r["text"] or "").lower() for k in ("pytest", "test", "verify"))]
    for pos in test_ok_positions:
        after = rows[pos + 1:pos + 11]
        edits = [r for r in after if r["kind"] == "tool_call"
                 and any(k in (r["text"] or "").lower() for k in ("write", "edit", "patch", "apply"))]
        if edits:
            out.append({"after_test_event": rows[pos]["chunk_id"],
                        "suspicious_edits": [r["chunk_id"] for r in edits[:5]],
                        "note": "edits followed a passing test; re-verification may be required"})
    return out[:10]


def _fix_reverify(rows) -> list[dict[str, Any]]:
    """Fail -> fix -> reverify sequences with event ids as evidence."""
    out = []
    failures = [i for i, r in enumerate(rows)
                if r["kind"] == "tool_result" and r["success"] is False]
    for f in failures[:20]:
        after = rows[f + 1:f + 15]
        fix = next((r for r in after if r["kind"] == "tool_call"), None)
        reverify = next((r for r in after if r["kind"] == "tool_result" and r["success"] is True), None)
        entry = {"failed_event": rows[f]["chunk_id"], "error": (rows[f]["error_text"] or "")[:120]}
        if fix:
            entry["fix_attempt"] = fix["chunk_id"]
        if reverify:
            entry["reverified_ok"] = reverify["chunk_id"]
        elif fix:
            entry["note"] = "fix attempted but no subsequent success observed"
        out.append(entry)
    return out
