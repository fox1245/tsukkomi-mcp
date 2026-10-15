"""A session store fed in batches, and an audit that runs exactly the way the engine runs one.

``Harness.audit`` plans an incremental evaluation inside the store's read boundary, evaluates it through the audit
graph, evaluates the same events with the full evaluator, compares the two and stores the new state. It is the
engine's flow without the source file, so tests can commit, rewrite and tamper with the stored events directly.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from self_directing_mcp.audit import incremental as inc
from self_directing_mcp.audit.runner import audit_needs, run_audit
from self_directing_mcp.index.ingest import AuditEventSnapshot, SessionStore, current_parser_version
from self_directing_mcp.schemas import Chunk

SID = "s-incremental"


def chunk_of(provider, event):
    meta = dict(event.meta)
    meta["sub_index"] = event.key[1]
    return Chunk(chunk_id=event.chunk_id, session_id=SID, provider=provider, kind=event.kind, text=event.text,
                 content_hash=hashlib.sha256(event.text.encode()).hexdigest(), byte_start=event.key[0],
                 byte_end=event.key[0] + 1, line_start=1, line_end=1, meta=meta)


class Harness:
    def __init__(self, tmp_path, provider):
        self.provider = provider
        self.store = SessionStore(tmp_path / "meta.sqlite")
        self.source = tmp_path / "source.jsonl"
        self.source.write_text("synthetic\n")
        self.events: list = []
        self.committed = 0
        self.offset = 0

    # --- the store ---

    def commit(self, events, *, replace=False, path=None, parser_version=None):
        """Publish ``events`` (the next ones of the session) as one append, or the whole session when ``replace``."""
        if replace:
            self.events = list(events)
            batch = self.events
        else:
            self.events = self.events + list(events)
            batch = list(events)
        self.committed = len(self.events)
        self.offset = max([e.key[0] for e in self.events], default=0) + 1
        digest = hashlib.sha256(f"{self.offset}|{len(self.events)}|{self.events[-1].chunk_id if self.events else ''}"
                                .encode()).hexdigest()
        self.store.commit_events([chunk_of(self.provider, e) for e in batch], session_id=SID, provider=self.provider,
                                 path=path or self.source, offset=self.offset, digest=digest, replace=replace,
                                 parser_version=parser_version)

    # --- audits ---

    def audit(self, rules, *, proposed=None, shadow_every=0, force_full=frozenset(), save=True, planned=None,
              retry=True):
        """One audit, the way the engine runs it (a delta run that meets a capacity limit is repeated in full).

        Returns ``(result, report, plan)`` after asserting the incremental and full results agree. ``retry=False``
        lets the ``PlanRetry`` through, to test the request itself.
        """
        try:
            return self._audit(rules, shadow_every, force_full, save, planned)
        except inc.PlanRetry as again:
            if not retry:
                raise
            if save and again.plan is not None:
                rows = again.plan.overflow_rows()
                if rows:
                    self.store.save_audit_states(SID, self.provider, again.plan.digest, rows, again.plan.snapshot)
            result, report, plan = self._audit(rules, shadow_every, again.ids, save, planned)
            report["retry"] = {"rules": sorted(again.ids), "reason": again.reason}
            return result, report, plan

    def _audit(self, rules, shadow_every, force_full, save, planned):
        store, provider = self.store, self.provider
        active = [r for r in rules if r.enabled]
        needs, links = audit_needs(active, planned)
        if inc.wants_call_index(active, provider):
            store.ensure_call_index()
        report = {}
        history = any(r.regex and (r.type == "must" or planned is None) for r in active)
        anchors = {a for r in active for a in (r.source_event_id, r.applies_from_event_id) if a}

        def capture():
            if history:
                return store.capture_audit_events(SID, provider, needs, link_calls=links)
            return store.capture_evidence(SID, provider, history=False, anchors=anchors)

        with store._read_boundary():
            plan = None
            if history:
                plan = inc.AuditPlan.prepare(store, session_id=SID, provider=provider, active=active, planned=planned,
                                             needs=needs, link_calls=links,
                                             shadow_every=shadow_every, force_full=force_full, report=report)
            evidence = (AuditEventSnapshot(SID, provider, tuple(plan.events), plan.count) if plan is not None
                        else capture())
            reference = capture()
        coverage = {"complete": True, "issues": []}
        result = run_audit(session_id=SID, contracts=rules, store=evidence, provider=provider, proposed=planned,
                           coverage=coverage, plan=plan)
        expected = run_audit(session_id=SID, contracts=rules, store=reference, provider=provider, proposed=planned,
                             coverage={"complete": True, "issues": []})
        differing = inc.compare(result, expected)
        assert not differing, (differing, report, [f.model_dump() for f in result.findings],
                               [f.model_dump() for f in expected.findings])
        if plan is not None and save:
            rows = plan.state_rows()
            report["state"] = store.save_audit_states(SID, provider, plan.digest, rows, plan.snapshot) if rows else "unchanged"
            if plan.schedule is not None:     # the engine keeps the comparison schedule next to the states
                store.save_shadow_state(SID, provider, plan.schedule)
        return result, report, plan
