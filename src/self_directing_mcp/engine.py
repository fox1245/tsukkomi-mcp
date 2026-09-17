from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from functools import wraps
from threading import RLock
from collections import OrderedDict
import hashlib
import json
import re
import time

from self_directing_mcp.session_events import as_text, make_chunk

from self_directing_mcp.audit.contracts import (
    GLOBAL_CONTRACT_MEANING,
    ContractStore,
    contract_scope,
    scope_counts,
    visible_contracts,
)
from self_directing_mcp.audit.runner import run_audit
from self_directing_mcp.checklist import (
    INCOMPLETE_NOTE,
    ChecklistStore,
    ChecklistDocument,
    RequirementItem,
    RequirementStatus,
)
from self_directing_mcp.codex import discover as codex_discover
from self_directing_mcp.grokbot import discover as grokbot_discover
from self_directing_mcp.config import Settings, get_settings
from self_directing_mcp.embed.embedder import build_embedder
from self_directing_mcp.index.ingest import SessionStore, ingest_session
from self_directing_mcp.index.sparse import SparseIndex
from self_directing_mcp.index.vector import build_vector_index
from self_directing_mcp.index.locking import index_lock
from self_directing_mcp.retrieve.checks import regex_search
from self_directing_mcp.retrieve.hybrid import HybridRetriever, hits_to_schema
from self_directing_mcp.schemas import ContractRule, ContractsDocument, ProposedAction
from self_directing_mcp.security.mask import mask_secrets, mask_value
from self_directing_mcp.timeseries import analyze_activity
from self_directing_mcp.graphrag import GraphStore, validate_proposal
from self_directing_mcp.neograph_runner import run_update_pipeline
from self_directing_mcp.neograph_runtime import ng, run_stages, graph_operation


def serialized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        deadline = kwargs.pop("_deadline", None)
        started = time.monotonic()
        wait_until = min(deadline or float("inf"), started + self.settings.lock_wait_timeout_sec)
        remaining = wait_until - started
        if remaining <= 0 or not self._lock.acquire(timeout=remaining):
            raise TimeoutError("local_index_busy")
        try:
            if self._operation_depth:
                return method(self, *args, **kwargs)
            remaining = wait_until - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("local_index_busy")
            with index_lock(Path(self.settings.index_dir), timeout=remaining):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("request_expired_before_execution")
                self._operation_depth += 1
                try:
                    if self.dense is not None:
                        self.dense.refresh()
                    return method(self, *args, **kwargs)
                finally:
                    self._operation_depth -= 1
        finally:
            self._lock.release()
    return wrapped


class SelfDirectEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self._lock = RLock()
        self._operation_depth = 0
        self._hook_lock = RLock()
        self._hook_states = OrderedDict()
        self.settings = settings or get_settings()
        self.store: SessionStore | None = None
        self.sparse: SparseIndex | None = None
        self.dense = None
        self.dense_backend = "numpy"
        self.embedder = None
        self.embedder_degraded = False
        self.retriever: HybridRetriever | None = None
        self.contracts: ContractStore | None = None
        self.checklists: ChecklistStore | None = None
        self.graph: GraphStore | None = None
        self.neograph = None
        self._ready = False

    def hook_obligations(self, session_id: str) -> dict[str, Any]:
        """Read atomically replaced rule files without initializing indexes.

        Missing obligations are not a compliance verdict. Corrupt files raise,
        so unavailable rules cannot be mistaken for an empty rule set.
        """
        sid = session_id.lower()
        path = self.settings.resolve_contracts_path()
        doc = ContractsDocument.model_validate_json(path.read_bytes()) if path.exists() else ContractsDocument()
        rules = [r for r in visible_contracts(doc.contracts, sid)
                 if r.enabled and r.provider in (None, "codex")]
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sid)[:120]
        path = Path(self.settings.index_dir) / "checklists" / f"{safe}.json"
        checklist = ChecklistDocument.model_validate_json(path.read_bytes()) if path.exists() else ChecklistDocument(session_id=sid)
        pending = [item for item in checklist.items if item.status != "revoked"
                   and (item.status != "done" or not item.verified or not item.evidence_chunk_ids)]
        payload = {"contracts": [r.model_dump(mode="json") for r in rules],
                   "pending": [i.model_dump(mode="json") for i in pending]}
        return {"status": "applicable" if rules or pending else "not_applicable",
                "contracts": rules, "pending": pending,
                "snapshot": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}

    @serialized
    def ensure_ready(self) -> None:
        if self._ready:
            return
        s = self.settings
        index_dir = Path(s.index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        self.embedder, self.embedder_degraded = build_embedder(
            api_key=s.resolve_api_key(),
            use_fake=s.use_fake_embedder,
            model=s.embedding_model,
            dim=s.embedding_dim,
            base_url=s.embedding_base_url,
        )
        self.dense, self.dense_backend = build_vector_index(
            index_dir / ("vectors-v2-" + self.embedder.cache_key[:16]), dim=s.embedding_dim,
            backend=s.dense_backend, extension_path=s.sqlite_vector_path)
        self.sparse = SparseIndex(index_dir / "sparse.sqlite")
        self.store = SessionStore(index_dir / "meta.sqlite")
        self.contracts = ContractStore(s.resolve_contracts_path())
        self.checklists = ChecklistStore(index_dir)
        self.graph = GraphStore(index_dir / "graph.sqlite")
        self.neograph = ng
        # Explicit opt-in, including wheel installations.
        seed = Path(__file__).resolve().parent / "resources" / "examples.json"
        if s.seed_example_contracts:
            self.contracts.seed_from(seed)

        self.retriever = HybridRetriever(
            sparse=self.sparse,
            dense=self.dense,
            embedder=self.embedder,
            rrf_k=s.rrf_k,
            retrieve_top_k=s.retrieve_top_k,
            store=self.store,
        )
        self._ready = True

    def _resolve_provider(
        self,
        provider: str | None,
        *,
        path: str | None = None,
    ) -> Literal["codex", "grokbot"]:
        """Explicit arg > settings/env > light auto-detect from path layout."""
        if provider:
            p = provider.strip().lower()
            if p in ("codex", "grokbot"):
                return p  # type: ignore[return-value]
            raise ValueError(f"unknown provider: {provider!r} (use codex|grokbot)")
        # Auto-detect: path under a configured grokbot root, or <id>/<id>.jsonl layout
        if path:
            try:
                path_r = Path(path).expanduser().resolve()
            except OSError:
                path_r = Path(path)
            for root in self.settings.resolve_grokbot_transcripts_dirs():
                try:
                    path_r.relative_to(Path(root).resolve())
                    return "grokbot"
                except ValueError:
                    continue
            # Layout hint: parent name == stem (Grok Bot / Cursor transcripts)
            if path_r.parent.name == path_r.stem and path_r.suffix.lower() == ".jsonl":
                return "grokbot"
        return self.settings.resolve_session_provider()

    @serialized
    def sync_session(
        self,
        *,
        session_id: str | None = None,
        path: str | None = None,
        provider: str | None = None,
        embed: bool = True,
    ) -> dict[str, Any]:
        self.ensure_ready()
        assert self.store and self.sparse and self.dense and self.embedder
        try:
            resolved_provider = self._resolve_provider(provider, path=path)
        except ValueError as e:
            return {"ok": False, "error": "bad_provider", "message": str(e)}

        try:
            if resolved_provider == "grokbot":
                roots = self.settings.resolve_grokbot_transcripts_dirs()
                resolved = grokbot_discover.resolve_session_path(
                    roots, session_id=session_id, path=path
                )
            else:
                sessions_root = self.settings.resolve_sessions_dir()
                resolved = codex_discover.resolve_session_path(
                    sessions_root, session_id=session_id, path=path
                )
        except (codex_discover.PathTraversalError, grokbot_discover.PathTraversalError) as e:
            return {"ok": False, "error": "path_traversal", "message": str(e)}
        except FileNotFoundError as e:
            return {"ok": False, "error": "not_found", "message": str(e)}
        except ValueError as e:
            return {"ok": False, "error": "bad_request", "message": str(e)}

        info = ingest_session(
            path=resolved,
            session_id=session_id,
            store=self.store,
            sparse=self.sparse,
            dense=self.dense,
            embedder=self.embedder,
            provider=resolved_provider,
            embed=embed,
        )
        info["ok"] = True
        info["dense_backend"] = self.dense_backend
        info["embedder"] = type(self.embedder).__name__
        info["embedder_degraded"] = self.embedder_degraded
        info["nudge"] = (
            "NUDGE: After sync_session, call audit_session before risky shell/network/"
            "file-destructive actions."
        )
        return info

    @serialized
    def search_history(
        self,
        query: str,
        *,
        session_id: str | None = None,
        mode: str = "hybrid",
        top_k: int | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
        self.ensure_ready()
        assert self.store and self.retriever
        provider = self._resolve_provider(provider)
        if session_id and provider == "codex":
            session_id = session_id.lower()
        k = top_k if top_k is not None else self.settings.search_top_k
        if mode not in ("hybrid", "regex", "sparse", "dense") or not 1 <= k <= 100:
            raise ValueError("mode must be hybrid|regex|sparse|dense and top_k must be 1..100")
        if mode == "regex":
            hits = regex_search(self.store, query, session_id=session_id, top_k=k, provider=provider)
        else:
            results = self.retriever.retrieve(
                query, top_k=k, session_id=session_id, mode=mode, provider=provider
            )
            hits = hits_to_schema(results, self.store.get_chunk)
        return {
            "ok": True,
            "mode": mode,
            "query": mask_secrets(query),
            "session_id": session_id,
            "provider": (provider or self.settings.resolve_session_provider()),
            "hits": [h.model_dump(mode="json") for h in hits],
            "nudge": (
                "NUDGE: Do not treat retrieval alone as compliance. "
                "Call audit_session — contracts/regex are authoritative."
            ),
        }

    def _audit(self, session_id, provider, proposed=None, path=None):
        started = time.perf_counter()
        self.ensure_ready()
        provider = self._resolve_provider(provider)
        sid = session_id.lower() if provider == "codex" else session_id
        # Always refresh local evidence. Regex audit never depends on remote embeddings.
        record = self.store.cursor_record(sid, provider)
        try:
            synced = self.sync_session(session_id=sid, path=path or (record["path"] if record else None),
                                       provider=provider, embed=False)
            coverage = synced.get("coverage") or self.store.session_status(sid, provider)
            if not synced.get("ok"):
                coverage["complete"] = False
                coverage["issues"].append(synced.get("error", "sync_failed"))
        except (OSError, ValueError) as exc:
            coverage = self.store.session_status(sid, provider)
            coverage["complete"] = False
            coverage["issues"].append("sync_failed:" + type(exc).__name__)
        sync_finished = time.perf_counter()
        self.contracts.load_if_changed()
        planned = None
        if proposed is not None:
            action = ProposedAction.model_validate(proposed)
            planned = make_chunk(provider, sid, "tool_call",
                                 f"[tool_call:{action.tool_name}] {as_text(action.arguments)}",
                                 {"role": "assistant", "tool_name": action.tool_name, "proposed": True},
                                 0, coverage.get("byte_offset", 0), coverage.get("byte_offset", 0))
            planned.chunk_id = "proposed:" + planned.chunk_id
        result = run_audit(session_id=sid, contracts=self.contracts.list(), store=self.store,
                           provider=provider, proposed=planned, coverage=coverage)
        payload = result.model_dump(mode="json")
        payload.update(ok=True, provider=provider, action_executed=False,
                       requires_attention=result.verdict != "clean")
        payload["timings_ms"] = {
            "sync": round((sync_finished - started) * 1000, 2),
            "evaluation": round((time.perf_counter() - sync_finished) * 1000, 2)}
        payload["nudge"] = (
            "Detect only. violation/suspicious: STOP and report; unknown: resolve missing evidence. "
            "clean is limited to the reported contracts and history snapshot, not execution authorization."
        )
        payload["global_meaning"] = GLOBAL_CONTRACT_MEANING
        payload["contract_scope"] = {
            "session_id": sid,
            "scope": contract_scope(sid),
        }
        return payload

    @serialized
    def audit_session(self, session_id: str, provider: str | None = None, path: str | None = None) -> dict[str, Any]:
        return self._audit(session_id, provider, path=path)

    @serialized
    def check_action(self, session_id: str, action: dict[str, Any], provider: str | None = None, path: str | None = None):
        """Check a proposed tool call against refreshed history; never execute or persist it."""
        return self._audit(session_id, provider, proposed=action, path=path)

    @serialized
    def get_chunk(self, chunk_id: str) -> dict[str, Any]:
        self.ensure_ready()
        assert self.store
        chunk = self.store.get_chunk(chunk_id)
        if not chunk:
            return {"ok": False, "found": False, "chunk_id": chunk_id}
        data = chunk.model_dump(mode="json")
        data = mask_value(data)
        return {"ok": True, "found": True, "chunk": data}

    @serialized
    def list_contracts(self, session_id: str | None = None) -> dict[str, Any]:
        self.ensure_ready()
        assert self.contracts
        self.contracts.load_if_changed()
        stored = self.contracts.list()
        selected = visible_contracts(stored, session_id)
        return {
            "ok": True,
            "session_id": session_id,
            "scope": contract_scope(session_id),
            "global_meaning": GLOBAL_CONTRACT_MEANING,
            "contracts": [c.model_dump(mode="json") for c in selected],
            "counts": scope_counts(selected, len(stored)),
        }

    @serialized
    def revoke_contract(self, contract_id: str) -> dict[str, Any]:
        """Disable a rule while keeping its history and source quote for audit."""
        self.ensure_ready()
        assert self.contracts
        self.contracts.load_if_changed()
        disabled = self.contracts.revoke(contract_id)
        if disabled is None:
            return {"ok": False, "error": "not_found", "contract_id": contract_id}
        return {"ok": True, "contract": disabled.model_dump(mode="json"),
                "nudge": "Contract disabled. History retained; re-enable via upsert_contracts with the same id."}

    @serialized
    def upsert_contracts(self, contracts: list[dict[str, Any] | ContractRule]) -> dict[str, Any]:
        self.ensure_ready()
        assert self.contracts
        rules: list[ContractRule] = []
        for c in contracts:
            if isinstance(c, ContractRule):
                rules.append(c)
            else:
                rules.append(ContractRule.model_validate(c))
        for rule in rules:
            for pattern in (rule.regex, rule.before_regex):
                if pattern:
                    re.compile(pattern)
        self.contracts.load_if_changed()
        updated = self.contracts.upsert(rules)
        stored = self.contracts.list()
        return {
            "ok": True,
            "returned": "upserted",
            "global_meaning": GLOBAL_CONTRACT_MEANING,
            "contracts": [c.model_dump(mode="json") for c in updated],
            "counts": {
                "upserted": len(updated),
                "store": len(stored),
            },
        }

    @serialized
    def update_checklist(
        self,
        session_id: str,
        provider: str | None = None,
        add: list[dict[str, Any]] | None = None,
        update: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Track user requirements as checklist items with provenance.

        add: [{text, source_event_id?, source_message?, satisfaction_criteria?}]
        update: [{item_id, status?, evidence_chunk_ids?, blocked_reason?, verified?}]
        A completion claim without evidence stays pending_verification, not done.
        """
        self.ensure_ready()
        assert self.checklists
        resolved_provider = self._resolve_provider(provider)
        sid = session_id.lower() if resolved_provider == "codex" else session_id
        doc = self.checklists.load(sid)
        added = []
        for item in add or []:
            entry = RequirementItem(
                item_id=doc.next_item_id(),
                text=item.get("text", ""),
                source_event_id=item.get("source_event_id"),
                source_message=item.get("source_message"),
                satisfaction_criteria=item.get("satisfaction_criteria"),
            )
            entry.history.append({"change": "added", "status": entry.status})
            doc.items.append(entry)
            added.append(entry.model_dump(mode="json"))
        updated = []
        for change in update or []:
            item = next((i for i in doc.items if i.item_id == change.get("item_id")), None)
            if item is None:
                updated.append({"ok": False, "error": "not_found", "item_id": change.get("item_id")})
                continue
            before = item.status
            if "evidence_chunk_ids" in change:
                item.evidence_chunk_ids = list(change["evidence_chunk_ids"])
            if change.get("status") == RequirementStatus.DONE.value:
                if item.evidence_chunk_ids:
                    item.status = RequirementStatus.DONE
                    item.verified = bool(change.get("verified", True))
                else:
                    # Completion claim without evidence never marks done.
                    item.status = RequirementStatus.PENDING_VERIFICATION
                    item.verified = False
            elif change.get("status"):
                item.status = change["status"]
            if change.get("blocked_reason"):
                item.blocked_reason = change["blocked_reason"]
            item.history.append({"change": "updated", "from": before, "to": item.status,
                                 "evidence": len(item.evidence_chunk_ids)})
            updated.append(item.model_dump(mode="json"))
        self.checklists.save(doc)
        remaining = [i for i in doc.items
                     if i.status not in (RequirementStatus.DONE, RequirementStatus.REVOKED)]
        return {
            "ok": True,
            "added": added,
            "updated": updated,
            "remaining": [i.model_dump(mode="json") for i in remaining],
            "nudge": INCOMPLETE_NOTE if remaining else None,
        }

    @serialized
    def get_checklist(self, session_id: str, provider: str | None = None) -> dict[str, Any]:
        """Restore requirements and remaining items after compaction."""
        self.ensure_ready()
        assert self.checklists
        resolved_provider = self._resolve_provider(provider)
        sid = session_id.lower() if resolved_provider == "codex" else session_id
        doc = self.checklists.load(sid)
        return {"ok": True, "session_id": sid, "items": [i.model_dump(mode="json") for i in doc.items]}

    @serialized
    def analyze_activity(self, session_id: str, provider: str | None = None,
                         window_minutes: int = 60) -> dict[str, Any]:
        """SQLite-based activity time-series analysis with evidence ids."""
        self.ensure_ready()
        assert self.store
        resolved_provider = self._resolve_provider(provider)
        sid = session_id.lower() if resolved_provider == "codex" else session_id
        return analyze_activity(self.store, session_id=sid, provider=resolved_provider,
                                window_minutes=max(1, min(window_minutes, 1440)))

    @serialized
    def get_graph_context(
        self,
        session_id: str,
        node_id: str | None = None,
        provider: str | None = None,
        depth: int = 1,
    ) -> dict[str, Any]:
        """Fetch GraphRAG context: entity neighbor subgraph or overall session graph."""
        self.ensure_ready()
        assert self.graph is not None
        resolved_provider = self._resolve_provider(provider)
        sid = session_id.lower() if resolved_provider == "codex" else session_id

        if node_id:
            res = self.graph.neighbors(sid, resolved_provider, node_id, depth=depth)
            if not res.get("edges"):
                matches = self.graph.find_nodes_like(node_id, sid, resolved_provider)
                if matches:
                    matched_id = matches[0]["node_id"]
                    res = self.graph.neighbors(sid, resolved_provider, matched_id, depth=depth)
            return {
                "ok": True,
                "session_id": sid,
                "provider": resolved_provider,
                "node_id": node_id,
                "subgraph": res,
                "cursor": self.graph.cursor(sid, resolved_provider),
            }
        else:
            graph_data = self.graph.get_session_graph(sid, resolved_provider)
            return {
                "ok": True,
                "session_id": sid,
                "provider": resolved_provider,
                "nodes": graph_data["nodes"],
                "edges": graph_data["edges"],
                "cursor": self.graph.cursor(sid, resolved_provider),
            }

    @serialized
    @graph_operation("validate_graph_proposal")
    def propose_graph_update(self, session_id: str, proposal: dict[str, Any],
                             provider: str | None = None) -> dict[str, Any]:
        """Validate a node/edge proposal against indexed evidence; nothing stored here."""
        self.ensure_ready()
        assert self.store and self.graph
        resolved_provider = self._resolve_provider(provider)
        sid = session_id.lower() if resolved_provider == "codex" else session_id
        verdict = validate_proposal(proposal, store=self.store, session_id=sid, provider=resolved_provider)
        cursor = self.graph.cursor(sid, resolved_provider)
        return {"ok": verdict["ok"], **verdict, "cursor": cursor,
                "nudge": None if verdict["ok"] else "Fix problems and resubmit; nothing was stored."}

    @serialized
    @graph_operation("commit_graph_proposal")
    def commit_graph_update(self, session_id: str, proposal: dict[str, Any],
                            base_graph_version: int, provider: str | None = None) -> dict[str, Any]:
        """Apply a validated proposal atomically and advance the cursor idempotently."""
        self.ensure_ready()
        assert self.store and self.graph
        resolved_provider = self._resolve_provider(provider)
        sid = session_id.lower() if resolved_provider == "codex" else session_id
        existing = self.graph.find_job(proposal.get("job_id", "") or "adhoc")
        if existing and existing["status"] == "applied":
            # Idempotent replay: same job id never double-applies.
            return {"ok": True, "status": "already_applied", "job": existing}
        cursor = self.graph.cursor(sid, resolved_provider)
        if base_graph_version != cursor["graph_version"]:
            # Stale base: the caller read an outdated graph state. Do not apply.
            return {"ok": False, "status": "version_conflict",
                    "expected_base_graph_version": cursor["graph_version"],
                    "provided_base_graph_version": base_graph_version,
                    "nudge": "Re-read the current graph and rebuild the proposal against it."}
        verdict = validate_proposal(proposal, store=self.store, session_id=sid, provider=resolved_provider)
        if not verdict["ok"]:
            return {"ok": False, "status": "rejected", **verdict}
        frames = self.store.event_frames(sid, resolved_provider)
        through = len(frames) - 1 if frames else cursor["processed_through_order"]
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        job = {"job_id": proposal.get("job_id", "adhoc"), "provider": resolved_provider,
               "session_id": sid, "range_start": cursor["processed_through_order"],
               "range_end": through, "input_digest": str(hash(json.dumps(proposal, sort_keys=True))),
               "base_graph_version": base_graph_version, "status": "pending",
               "created_at": now, "updated_at": now}
        result = self.graph.apply_proposal(proposal, session_id=sid, provider=resolved_provider,
                                           base_version=base_graph_version, through_order=through)
        job["status"] = result["status"]
        job["detail"] = result.get("detail")
        self.graph.record_job(job)
        return result

    @serialized
    def _prepare_neograph_update(self, session_id, provider=None):
        self.ensure_ready()
        resolved = self._resolve_provider(provider)
        sid = session_id.lower() if resolved == "codex" else session_id
        return sid, resolved, self.store.event_frames(sid, resolved), self.graph.cursor(sid, resolved)

    @serialized
    def _commit_neograph_snapshot(self, sid, provider, proposal, base_version, through,
                                  cancel_event=None):
        if cancel_event is not None and cancel_event.is_set():
            raise TimeoutError("graph_update_cancelled_before_commit")
        if self.graph.cursor(sid, provider)["graph_version"] != base_version:
            return {"ok": False, "status": "version_conflict"}
        verdict = validate_proposal(proposal, store=self.store, session_id=sid, provider=provider)
        if not verdict["ok"]:
            return {"ok": False, "status": "rejected", **verdict}
        return self.graph.apply_proposal(proposal, session_id=sid, provider=provider,
                                         base_version=base_version, through_order=through)

    def run_neograph_update(self, session_id: str, provider: str | None = None,
                            *, _deadline=None, cancel_event=None) -> dict[str, Any]:
        """Run the full update pipeline as a NeoGraph topology with the live LLM.

        Requires OPENROUTER_API_KEY and the neograph-engine package. Steps:
        extract (real LLM) -> validate (MCP evidence checks) -> apply (atomic
        write + cursor). Each run leaves a durable checkpoint; failures stay
        inspectable via thread_id graphrag:<provider>:<session>.
        """
        sid, resolved_provider, frames, cursor = self._prepare_neograph_update(
            session_id, provider, _deadline=_deadline)
        api_key = self.settings.resolve_api_key() or ""
        if not api_key:
            return {"ok": False, "error": "no_api_key",
                    "message": "Set OPENROUTER_API_KEY or SELF_DIRECT_OPENROUTER_API_KEY_FILE to run the LLM-driven update."}
        if self.neograph is None:
            return {"ok": False, "error": "neograph_missing",
                    "message": "pip install neograph-engine first."}
        if not frames:
            return {"ok": False, "error": "no_events",
                    "message": "Nothing to extract; sync_session first."}
        start = cursor["processed_through_order"] + 1 if cursor["graph_version"] else 0
        pending = frames[start:]
        if not pending:
            return {"ok": True, "status": "up_to_date", "graph_cursor_after": cursor}
        batch, size = [], 0
        for frame in pending[:100]:
            cost = len(frame["text"]) + len(frame["chunk_id"]) + 80
            if size + cost > 95000:
                break
            batch.append(frame)
            size += cost
        if not batch:
            return {"ok": False, "error": "graph_event_too_large",
                    "chunk_id": pending[0]["chunk_id"], "graph_cursor_after": cursor,
                    "message": "The next complete event exceeds the graph input budget; no text was truncated."}
        proposal_result: dict[str, Any] = {}

        def on_proposal(step: str, proposal: Any) -> dict[str, Any]:
            if (_deadline is not None and time.monotonic() >= _deadline) or (
                    cancel_event is not None and cancel_event.is_set()):
                raise TimeoutError("graph_update_expired_before_write")
            if step == "validate":
                verdict = self.propose_graph_update(sid, proposal, provider=resolved_provider,
                                                      _deadline=_deadline)
                proposal_result["verdict"] = verdict
                return verdict
            verdict = proposal_result.get("verdict", {"ok": False})
            if not verdict.get("ok"):
                return {"status": "rejected", "problems": verdict.get("problems", [])}
            through = batch[-1]["order"]
            return self._commit_neograph_snapshot(sid, resolved_provider, proposal,
                cursor["graph_version"], through, cancel_event=cancel_event, _deadline=_deadline)

        pipeline = run_update_pipeline(
            session_id=sid, provider=resolved_provider, events=batch,
            api_key=api_key, model="deepseek/deepseek-v4-flash-0731",
            checkpoint_dir=Path(self.settings.index_dir), on_proposal=on_proposal,
            deadline=_deadline, cancel_event=cancel_event)
        pipeline["graph_cursor_after"] = self.graph.cursor(sid, resolved_provider)
        pipeline["batch_events"] = len(batch)
        applied = pipeline.get("apply_result", {}).get("status") == "applied"
        pipeline["ok"] = applied
        pipeline["remaining_events"] = len(pending) - (len(batch) if applied else 0)
        return pipeline

    @serialized
    def audit_status(self, session_id: str | None = None, provider: str | None = None) -> dict[str, Any]:
        self.ensure_ready()
        assert self.store and self.sparse
        s = self.settings
        status: dict[str, Any] = {
            "ok": True,
            "orchestration_backend": "neograph-engine",
            "neograph_version": ng.__version__,
            "sessions_dir": str(s.resolve_sessions_dir()),
            "grokbot_transcripts_dirs": [str(p) for p in s.resolve_grokbot_transcripts_dirs()],
            "session_provider": s.resolve_session_provider(),
            "index_dir": str(Path(s.index_dir).resolve()),
            "dense_backend": self.dense_backend,
            "dense_count": self.dense.count(),
            "vector_extension_version": getattr(self.dense, "extension_version", None),
            "vector_compute_backend": getattr(self.dense, "compute_backend", None),
            "embedder": type(self.embedder).__name__ if self.embedder else None,
            "embedder_degraded": self.embedder_degraded,
            "degraded_flags": [],
            "total_chunks": self.store.chunk_count(),
            "legacy_chunks_retained": self.store.legacy_count(),
            "index_schema": 2,
            "sparse_count": self.sparse.count(),
            "contract_count": len(self.contracts.list()) if self.contracts else 0,
        }
        if self.embedder_degraded:
            status["degraded_flags"].append("embedder_fake_fallback")
        if self.dense_backend == "numpy":
            status["degraded_flags"].append("dense_numpy_fallback")
        if session_id:
            status["session"] = self.store.session_status(session_id.lower() if self._resolve_provider(provider) == "codex" else session_id, self._resolve_provider(provider))
        status["nudge"] = (
            "NUDGE: Call sync_session at session start; audit_session before risky actions."
        )
        return status

    def close(self) -> None:
        """Release SQLite handles, including on Windows."""
        # Closing process-local handles neither reads nor writes shared index state.
        with self._lock:
            for component in (self.store, self.sparse, self.dense, self.graph):
                connection = getattr(component, "_conn", None)
                if connection is not None:
                    connection.close()
            self.store = self.sparse = self.dense = self.retriever = self.contracts = None
            self.graph = self.neograph = None
            self._ready = False
