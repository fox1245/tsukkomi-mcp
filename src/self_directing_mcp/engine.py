from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from functools import wraps
from threading import Condition, RLock
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
import copy
import hashlib
import json
import sqlite3
import re
import time

from self_directing_mcp.compressed_source import CompressedSourceError
from self_directing_mcp.session_events import as_text, make_chunk

from self_directing_mcp.audit.contracts import (
    GLOBAL_CONTRACT_MEANING,
    ContractStore,
    contract_scope,
    scope_counts,
    visible_contracts,
)
from self_directing_mcp.audit import incremental
from self_directing_mcp.audit.patterns import compile_contract
from self_directing_mcp.audit.runner import _worse, action_applies, audit_needs, downgrade_unproven_absence, prefix_note, run_audit
from self_directing_mcp.checklist import (
    INCOMPLETE_NOTE,
    ChecklistStore,
    ChecklistDocument,
    RequirementItem,
    RequirementStatus,
)
from self_directing_mcp.codex import discover as codex_discover
from self_directing_mcp.grokbot import discover as grokbot_discover
from self_directing_mcp import omp, agy, claude
from self_directing_mcp.config import Settings, get_settings
from self_directing_mcp.embed.embedder import build_embedder
from self_directing_mcp.index.ingest import (
    AmbiguousSessionSource, AuditEventSnapshot, IntegrityPolicy, SessionStore, capture_for_cursor, coverage_from_capture,
    ingest_session,
)
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
from self_directing_mcp.neograph_runtime import ng, graph_operation
from self_directing_mcp.request_control import (
    IndexBusy, RequestStopped, check_request, current_request, publication, request_operation, request_scope,
)


def serialized(method):
    @request_operation
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._publication_guard():
            return method(self, *args, **kwargs)
    return wrapped


def detached_operation(method):
    @request_operation
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._components():
            return method(self, *args, **kwargs)
    return wrapped

def workflow_operation(method):
    """Own component lifetime without discarding observed workflow cleanup outcomes."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with request_scope(deadline=kwargs.get("_deadline"), cancel_event=kwargs.get("cancel_event")):
            with self._components():
                return method(self, *args, **kwargs)
    return wrapped



def _mark_growth(coverage, observation) -> bool:
    """Record the judged offset and unread bytes. ``judged_prefix`` means pure append is the only gap."""
    judged = coverage.get("byte_offset")
    if judged is None:
        return False
    unread = coverage.get("pending_bytes", 0)
    if observation is not None:
        try:
            unread = max(unread, observation.current_size() - judged)
        except OSError:
            pass
    unread = max(0, unread)
    coverage["judged_through_byte"], coverage["unread_bytes"] = judged, unread
    if unread:
        if "unread_bytes" not in coverage["issues"]:
            coverage["issues"].append("unread_bytes")
        coverage["complete"] = False
        coverage["judged_prefix"] = set(coverage["issues"]) == {"unread_bytes"}
    return bool(unread)


VERIFY_MODES = ("full", "fast")


def _check_verify(verify):
    if verify not in VERIFY_MODES:
        raise ValueError("verify must be 'full' or 'fast'")
    return verify


def _event_start(chunk_id):
    """Byte offset a chunk id encodes (``provider:session:start:sub:digest``); None when it cannot be read."""
    try:
        return int(chunk_id.rsplit(":", 3)[1])
    except (AttributeError, IndexError, ValueError):
        return None


def _fast_gate(payload, integrity):
    """Why a fast verdict may not stand, or None.

    A fast check compared only some blocks of the prefix. A clean finding that rests on historical evidence (a
    satisfied prerequisite) is accepted only when every cited event lies inside the region that was last verified
    in FULL (``full_verified_through_byte``). Such findings carry ``verified_through``. Anything else is settled by a
    full verification before it is reported; a fast verdict never turns missing or unverified evidence into a
    satisfied obligation.
    """
    through = integrity.get("full_verified_through_byte")
    for finding in payload.get("findings", []):
        if finding.get("verdict") != "clean" or finding.get("applicability", "applicable") != "applicable":
            continue
        cited = [item["chunk_id"] for item in finding.get("evidence", [])
                 if item.get("detail") != "triggering action" and not str(item["chunk_id"]).startswith("proposed:")]
        if not cited:
            continue
        starts = [_event_start(chunk_id) for chunk_id in cited]
        if through is None or any(start is None or start >= through for start in starts):
            return "evidence_beyond_verified_region"
        finding["verified_through"] = through
    return None


CHECKLIST_EVALUATION_ATTEMPTS = 3


class ChecklistConflict(ValueError):
    """Checklist file or evidence moved between capture and publication."""


class SelfDirectEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self._lock = RLock()
        self._lifetime = Condition(self._lock)
        self._active_operations = 0
        self._closing = False
        self._operation_depth = 0
        self._hook_lock = RLock()
        self._hook_states = OrderedDict()
        self._last_activity = time.monotonic()
        self._integrity_worker = None  # set only by a resident server (see integrity_worker)
        self.integrity_policy: IntegrityPolicy | None = None  # tests may inject a clock / randomness
        self.settings = settings or get_settings()
        self.store: SessionStore | None = None
        self.sparse: SparseIndex | None = None
        self.dense = None
        self.dense_backend = "disabled"
        self.embedder = None
        self.embedder_degraded = False
        self.retriever: HybridRetriever | None = None
        self.contracts: ContractStore | None = None
        self.checklists: ChecklistStore | None = None
        self.graph: GraphStore | None = None
        self.neograph = None
        self._ready = False

    @contextmanager
    def _publication_guard(self):
        """Combined bounded local/file lease for capture, lifetime and publication."""
        check_request()
        started = time.monotonic()
        control = current_request()
        wait_until = min(control.deadline if control and control.deadline is not None else float("inf"),
                         started + self.settings.lock_wait_timeout_sec)
        remaining = wait_until - started
        if remaining <= 0 or not self._lock.acquire(timeout=max(0, remaining)):
            check_request()
            raise IndexBusy()
        try:
            check_request()
            if self._operation_depth:
                yield
                return
            remaining = wait_until - time.monotonic()
            if remaining <= 0:
                check_request()
                raise IndexBusy()
            with index_lock(Path(self.settings.index_dir), timeout=remaining):
                check_request()
                self._operation_depth += 1
                try:
                    yield
                finally:
                    self._operation_depth -= 1
        finally:
            self._lock.release()

    @contextmanager
    def _components(self):
        # Refcount lifetime without retaining either publication lock over CPU/I/O.
        with self._publication_guard():
            if self._closing:
                raise IndexBusy()
            self.ensure_ready()
            self._active_operations += 1
        try:
            yield
        finally:
            with self._lifetime:
                self._active_operations -= 1
                self._last_activity = time.monotonic()
                self._lifetime.notify_all()

    def _policy(self) -> IntegrityPolicy:
        return self.integrity_policy or IntegrityPolicy.from_settings(self.settings)

    def attach_integrity_worker(self, worker) -> None:
        self._integrity_worker = worker

    def accepts_background_work(self) -> bool:
        """False while closed or closing: a background worker must never reopen an engine that was shut down."""
        return self._ready and not self._closing

    def idle_seconds(self) -> float:
        """0 while any request is running, else the time since the last one finished."""
        return 0.0 if self._active_operations else time.monotonic() - self._last_activity

    def hook_obligations(self, session_id: str, provider: str = "codex", path: str | None = None) -> dict[str, Any]:
        """Read atomically replaced rule files without initializing indexes.

        Missing obligations are not a compliance verdict. Corrupt files raise,
        so unavailable rules cannot be mistaken for an empty rule set.
        """
        provider = self._resolve_provider(provider)
        sid = session_id.lower() if provider in ("codex", "agy") else session_id
        if provider == "agy" and path is not None:
            agy.resolve_session_path(self.settings.resolve_agy_app_data_dirs(),
                                     session_id=sid, path=path, require_exists=False)
        path = self.settings.resolve_contracts_path()
        doc = ContractsDocument.model_validate_json(path.read_bytes()) if path.exists() else ContractsDocument()
        rules = [r for r in visible_contracts(doc.contracts, sid)
                 if r.enabled and r.provider in (None, provider)]
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sid)[:120]
        path = Path(self.settings.index_dir) / "checklists" / f"{safe}.json"
        checklist = ChecklistDocument.model_validate_json(path.read_bytes()) if path.exists() else ChecklistDocument(session_id=sid)
        pending = [item for item in checklist.items if item.status != "revoked"
                   and (item.status != "done" or not item.verified or not item.evidence_chunk_ids)]
        payload = {"contracts": [r.model_dump(mode="json") for r in rules],
                   "pending": [i.model_dump(mode="json") for i in pending]}
        return {"status": "applicable" if rules or pending else "not_applicable",
                "contracts": rules, "pending": pending, "has_checklist": bool(checklist.items),
                "snapshot": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}

    @serialized
    def ensure_ready(self) -> None:
        if self._ready:
            return
        try:
            with publication():
                self._initialize()
        except BaseException:
            self.close()
            raise

    def _initialize(self):
        s = self.settings
        index_dir = Path(s.index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        if not s.local_only:
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
        self.store = SessionStore(index_dir / "meta.sqlite", max_decompressed_bytes=s.max_decompressed_session_bytes)
        self.contracts = ContractStore(s.resolve_contracts_path())
        self.checklists = ChecklistStore(index_dir)
        self.graph = GraphStore(index_dir / "graph.sqlite")
        self.neograph = ng
        # Explicit opt-in, including wheel installations.
        seed = Path(__file__).resolve().parent / "resources" / "examples.json"
        if s.seed_example_contracts:
            self.contracts.seed_from(seed)
            # Disabled high-risk presets; the owner enables the ones they want.
            self.contracts.seed_missing(seed.with_name("high_risk_presets.json"))

        if not s.local_only:
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
    ) -> Literal["codex", "grokbot", "omp", "agy", "claude"]:
        """Explicit arg > settings/env > light auto-detect from path layout."""
        if provider:
            p = provider.strip().lower()
            if p in ("codex", "grokbot", "omp", "agy", "claude"):
                return p  # type: ignore[return-value]
            raise ValueError(f"unknown provider: {provider!r} (use codex|grokbot|omp|agy|claude)")
        # Auto-detect: path under a configured grokbot root, or <id>/<id>.jsonl layout
        if path:
            try:
                path_r = Path(path).expanduser().resolve()
            except OSError:
                path_r = Path(path)
            if agy.peek_session_id(path_r) and any(
                path_r.is_relative_to(root) for root in self.settings.resolve_agy_app_data_dirs()
            ):
                return "agy"
            if claude.peek_session_id(path_r) and path_r.is_relative_to(
                    self.settings.resolve_claude_projects_dir().expanduser().resolve()):
                return "claude"
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

    @detached_operation
    def sync_session(
        self,
        *,
        session_id: str | None = None,
        path: str | None = None,
        provider: str | None = None,
        embed: bool = True,
        verify: str = "full",
    ) -> dict[str, Any]:
        info = self._sync_session(session_id=session_id, path=path, provider=provider, embed=embed,
                                  verify=_check_verify(verify))
        return {key: value for key, value in info.items() if not key.startswith("_")}

    @detached_operation
    def verify_integrity(self, session_id: str, provider: str | None = None, path: str | None = None) -> dict[str, Any]:
        """Full verification of one session's source (the background worker's work; also callable directly)."""
        info = self._sync_session(session_id=session_id, path=path, provider=provider, embed=False, derived=False,
                                  verify="full", watch=False)
        return {key: value for key, value in info.items() if not key.startswith("_")}

    def _sync_session(self, *, session_id=None, path=None, provider=None, embed=True, derived=True, verify="full",
                      watch=True):
        assert self.store and self.sparse
        if self.settings.local_only and embed:
            return {"ok": False, "error": "local_only", "message": "embed=True is unavailable in local-only mode"}
        try:
            resolved_provider = self._resolve_provider(provider, path=path)
        except ValueError as e:
            return {"ok": False, "error": "bad_provider", "message": str(e)}

        try:
            if resolved_provider == "agy":
                resolved = agy.resolve_session_path(
                    self.settings.resolve_agy_app_data_dirs(), session_id=session_id, path=path
                )
            elif resolved_provider == "claude":
                resolved = claude.resolve_session_path(
                    self.settings.resolve_claude_projects_dir(), session_id=session_id, path=path
                )
            elif resolved_provider == "omp":
                resolved = omp.resolve_session_path(
                    self.settings.resolve_omp_sessions_dir(), session_id=session_id, path=path
                )
            elif resolved_provider == "grokbot":
                roots = self.settings.resolve_grokbot_transcripts_dirs()
                resolved = grokbot_discover.resolve_session_path(
                    roots, session_id=session_id, path=path
                )
            else:
                sessions_root = self.settings.resolve_sessions_dir()
                resolved = codex_discover.resolve_session_path(
                    sessions_root, session_id=session_id, path=path
                )
        except (codex_discover.PathTraversalError, grokbot_discover.PathTraversalError, omp.PathTraversalError,
                agy.PathTraversalError, claude.PathTraversalError) as e:
            return {"ok": False, "error": "path_traversal", "message": str(e)}
        except FileNotFoundError as e:
            return {"ok": False, "error": "not_found", "message": str(e)}
        except CompressedSourceError as e:
            return {"ok": False, "error": e.code, "message": str(e), "provider": resolved_provider}
        except ValueError as e:
            return {"ok": False, "error": "bad_request", "message": str(e)}

        try:
            info = ingest_session(
                path=resolved,
                session_id=session_id,
                store=self.store,
                sparse=self.sparse,
                dense=self.dense,
                embedder=self.embedder,
                provider=resolved_provider,
                embed=embed,
                derived=derived,
                guard=self._publication_guard,
                # A local-only process (a hook) may share a server's index: leave vector cleanup journalled for it.
                retain_dense_deletions=self.dense is None and any(Path(self.settings.index_dir).glob("vectors-v2-*")),
                verify=verify,
                policy=self._policy(),
            )
        except AmbiguousSessionSource as exc:
            return {"ok": False, "error": "ambiguous_session_source", "message": str(exc), "provider": resolved_provider,
                    "session_id": exc.session_id, "indexed_path": exc.indexed_path, "requested_path": exc.requested_path}
        except CompressedSourceError as exc:
            # An unreadable .zst source (missing module, corrupt or truncated frame, size mismatch, over the limit):
            # visible, nothing indexed or published, coverage stays incomplete.
            return {"ok": False, "error": exc.code, "message": str(exc), "provider": resolved_provider,
                    "path": str(resolved)}
        info["ok"] = True
        worker = self._integrity_worker
        if watch and worker is not None:
            worker.note(resolved_provider, info["session_id"], str(resolved), info["integrity"])
        info["dense_backend"] = self.dense_backend
        info["embedder"] = type(self.embedder).__name__ if self.embedder else None
        info["embedder_degraded"] = self.embedder_degraded
        info["nudge"] = (
            "NUDGE: After sync_session, call audit_session before risky shell/network/"
            "file-destructive actions."
        )
        return info

    @request_operation
    def record_agy_hook(self, payload: dict[str, Any], event: str) -> None:
        """Record host hook observations separately from native transcript bytes."""
        from self_directing_mcp.agy_receipts import ReceiptStore
        sid = payload["conversationId"].lower()
        path = agy.resolve_session_path(self.settings.resolve_agy_app_data_dirs(),
                                        session_id=sid, path=payload["transcriptPath"],
                                        require_exists=False)
        ReceiptStore(Path(self.settings.index_dir)).record(
            sid, path, payload["stepIdx"], payload["toolCall"], event,
            error=payload.get("error", "") if event == "PostToolUse" else None,
            guard=self._publication_guard)

    @detached_operation
    def search_history(
        self,
        query: str,
        *,
        session_id: str | None = None,
        mode: str = "hybrid",
        top_k: int | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
        prepared_query = None
        assert self.store and self.sparse
        provider = self._resolve_provider(provider)
        if session_id and provider in ("codex", "agy"):
            session_id = session_id.lower()
        k = top_k if top_k is not None else self.settings.search_top_k
        if mode not in ("hybrid", "regex", "sparse", "dense") or not 1 <= k <= 100:
            raise ValueError("mode must be hybrid|regex|sparse|dense and top_k must be 1..100")
        if self.settings.local_only and mode in ("hybrid", "dense"):
            raise ValueError("dense/hybrid search unavailable in local-only mode")
        if mode in ("sparse", "hybrid"):
            self._repair_sparse_for_search(session_id, provider)
        if mode == "regex":
            with self._publication_guard():
                snapshot = self.store.capture_evidence(session_id, provider)
            hits = regex_search(snapshot, query, session_id=session_id, top_k=k, provider=provider)
        else:
            if not self.settings.local_only:
                prepared_query = self.retriever.prepare_query(query, mode=mode)
            with self._publication_guard():
                if self.dense is not None and mode in ("dense", "hybrid"):
                    self.dense.refresh()
                if mode == "sparse" and self.settings.local_only:
                    from self_directing_mcp.retrieve.hybrid import RRFResult
                    snapshot = self.store.capture_evidence(session_id, provider)
                    allowed = {cid for cid, _raw in snapshot.rows}
                    rows = self.sparse.search(mask_secrets(query), top_k=k, session_id=session_id, allowed_ids=allowed)
                    results = [RRFResult(chunk_id=cid, ranking_score=score, dense_rank=None,
                                         sparse_rank=rank, sparse_score=score) for cid, score, rank in rows]
                else:
                    results = self.retriever.retrieve(query, top_k=k, session_id=session_id, mode=mode,
                                                      provider=provider, query_vector=prepared_query)
                    snapshot = self.store.capture_evidence(session_id, provider, history=False,
                                                          anchors=(row.chunk_id for row in results))
            hits = hits_to_schema(results, snapshot.get_chunk)
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

    def _repair_sparse_for_search(self, session_id, provider):
        """Materialize deferred canonical evidence before searching its derived view."""
        with self._publication_guard():
            allowed = self.store.chunk_ids(session_id, provider)
            # A session search probes only its own ids; a global search needs every id, so one scan is cheaper.
            missing = allowed - (self.sparse.present(allowed) if session_id else self.sparse.ids())
            if not missing:
                return
            snapshot = self.store.capture_evidence(session_id, provider, history=False, anchors=missing)
        documents = snapshot.list_chunks(session_id, provider)
        with self._publication_guard():
            current = self.store.chunk_ids(session_id, provider)
            documents = [chunk for chunk in documents if chunk.chunk_id in current]
            if documents:
                self.sparse.upsert_many(documents)

    def _refresh_evidence(self, sid, provider, path=None, verify="full"):
        """Refresh only authority; return detached coverage and its exact inputs."""
        with self._publication_guard():
            record = self.store.cursor_record(sid, provider)
        failed = {}
        try:
            info = self._sync_session(session_id=sid, provider=provider, embed=False,
                                      path=path if path is not None else record["path"] if record else None,
                                      derived=False, verify=verify)
            if info.get("ok"):
                return info
            failure, failed = info.get("error", "sync_failed"), info
        except (RequestStopped, IndexBusy):
            raise
        except (OSError, ValueError) as exc:
            failure = "sync_failed:" + type(exc).__name__
        with self._publication_guard(), self.store._read_boundary():
            record = self.store.cursor_record(sid, provider)
            metrics = self.store.session_metrics(sid, provider)
            token = self.store.state_token(sid, provider)
        coverage = coverage_from_capture(sid, provider, record, metrics, max_bytes=self.store.max_decompressed_bytes)
        coverage["complete"] = False
        coverage["issues"].append(failure)
        coverage.update({key: failed[key] for key in ("indexed_path", "requested_path") if key in failed})
        return {"ok": False, "coverage": coverage, "_metadata_token": token,
                "_source_observation": None, "_receipt_token": None,
                "integrity": self._integrity_unavailable(verify, failure)}

    @staticmethod
    def _integrity_unavailable(verify, failure):
        """The source could not be verified for this request (the refresh failed); nothing is claimed about it."""
        return {"mode": verify, "full_verified_through_byte": None, "full_verified_at": None,
                "source_generation": None, "index_generation": None, "verification": "unavailable",
                "reason": failure}

    def _audit(self, session_id, provider, proposed=None, path=None, verify="full", escalated=None):
        started = time.perf_counter()
        provider = self._resolve_provider(provider)
        sid = session_id.lower() if provider in ("codex", "agy") else session_id
        refreshed = self._refresh_evidence(sid, provider, path, verify)
        integrity = dict(refreshed.get("integrity") or {})
        if escalated:
            integrity["full_reason"] = escalated
        coverage = dict(refreshed["coverage"])
        coverage["issues"] = list(coverage["issues"])
        sync_finished = time.perf_counter()
        with self._publication_guard():
            raw_contracts, contract_token = self.contracts.capture()
        try:
            document = ContractsDocument.model_validate_json(raw_contracts) if raw_contracts is not None else ContractsDocument()
            contracts = document.contracts
        except ValueError:
            contracts = []
            coverage["complete"] = False
            coverage["issues"].append("contracts_corrupt")
        planned = None
        if proposed is not None:
            action = ProposedAction.model_validate(proposed)
            planned = make_chunk(provider, sid, "tool_call",
                                 f"[tool_call:{action.tool_name}] {as_text(action.arguments)}",
                                 {"role": "assistant", "tool_name": action.tool_name, "proposed": True},
                                 0, coverage.get("byte_offset", 0), coverage.get("byte_offset", 0))
            planned.chunk_id = "proposed:" + planned.chunk_id
            # Preserve the public contracts_snapshot filtering/hash semantics.
            contracts = [rule for rule in contracts if rule.type != "must" or rule.before_regex]
        active = [rule for rule in contracts if rule.enabled and rule.provider in (None, provider)
                  and rule.session_id in (None, sid)]
        history = any(rule.regex and (rule.type == "must" or planned is None) for rule in active)
        anchors = {anchor for rule in active for anchor in (rule.source_event_id, rule.applies_from_event_id) if anchor}
        observation = refreshed["_source_observation"]
        report, plan, force_full = {}, None, frozenset()
        use_plan = history and self.settings.audit_incremental
        if use_plan and incremental.wants_call_index(active, provider):
            try:
                with self._publication_guard():
                    self.store.ensure_call_index()
            except (sqlite3.Error, RuntimeError, OSError) as exc:
                use_plan = False
                report.update(mode="full", fallback={"*": "call_index_unavailable:" + type(exc).__name__})
        for attempt in range(2):
            with self._publication_guard(), self.store._read_boundary():
                metadata_token = self.store.state_token(sid, provider)
                if metadata_token != refreshed["_metadata_token"] and "metadata_changed_before_capture" not in coverage["issues"]:
                    coverage["complete"] = False
                    coverage["issues"].append("metadata_changed_before_capture")
                plan = None
                if history:
                    # Only events some contract can inspect keep their text; rows are paged, never held as JSON.
                    needs, link_calls = audit_needs(active, planned)
                    if use_plan:
                        report = {key: value for key, value in report.items() if key == "retry"}
                        try:
                            plan = incremental.AuditPlan.prepare(
                                self.store, session_id=sid, provider=provider, active=active, planned=planned,
                                needs=needs, link_calls=link_calls, shadow_every=self.settings.audit_shadow_every,
                                force_full=force_full, report=report)
                        except (RequestStopped, IndexBusy):
                            raise
                        except Exception as exc:  # the cache could not even be read: evaluate everything in full
                            plan = None
                            report.update(mode="full", fallback={"*": "prepare_failed:" + type(exc).__name__})
                    if plan is not None:
                        evidence = AuditEventSnapshot(sid, provider, tuple(plan.events), plan.count)
                    else:
                        evidence = self.store.capture_audit_events(sid, provider, needs, link_calls=link_calls)
                else:
                    evidence = self.store.capture_evidence(sid, provider, history=False, anchors=anchors)
                receipt_token = self._receipt_token(sid, provider)
                if (provider == "agy" and receipt_token != refreshed["_receipt_token"]
                        and "receipts_changed_before_capture" not in coverage["issues"]):
                    coverage["complete"] = False
                    coverage["issues"].append("receipts_changed_before_capture")
            _mark_growth(coverage, observation)
            reference = None
            if plan is not None and plan.shadow:
                reference = AuditEventSnapshot(sid, provider, tuple(plan.reference), len(plan.reference))
                reference_coverage = copy.deepcopy(coverage)
            try:
                # The actual native stages, model decoding and regex evaluation own no
                # engine lease, shared SQLite handle or open read transaction.
                result = run_audit(session_id=sid, contracts=contracts, store=evidence,
                                   provider=provider, proposed=planned, coverage=coverage, plan=plan)
            except incremental.PlanRetry as retry:
                # A summary could not hold some input: those rules run on the full evaluator in a second pass, and
                # the limit is remembered so later audits do not fail the same way.
                force_full = retry.ids
                report = {"retry": {"rules": sorted(retry.ids), "reason": retry.reason}}
                try:
                    with self._publication_guard():
                        if retry.plan is not None:
                            self.store.save_audit_states(sid, provider, retry.plan.digest,
                                                         retry.plan.overflow_rows(), retry.plan.snapshot)
                        if retry.drop:
                            self.store.drop_audit_states(sid, provider, sorted(retry.ids))
                        if retry.health:
                            self.store.record_audit_health(sid, provider, incremental.EVALUATOR_VERSION,
                                                           [*sorted(retry.ids), "#facts"], retry.health, retry.reason)
                except (sqlite3.Error, OSError):
                    pass
                if attempt:
                    raise
                continue
            break
        mismatch = []
        if reference is not None:
            # Shadow: the full evaluator over the same events. It is the answer whenever the two differ.
            full = run_audit(session_id=sid, contracts=contracts, store=reference, provider=provider,
                             proposed=planned, coverage=reference_coverage)
            mismatch = incremental.compare(result, full)
            if mismatch:
                result = full
                ids = sorted({item.contract_id for item in full.findings} & set(plan.monitors)
                             if mismatch == ["#audit"] else set(mismatch))
                report["incremental_mismatch"] = ids
                try:
                    with self._publication_guard():
                        self.store.record_audit_health(
                            sid, provider, incremental.EVALUATOR_VERSION, [*ids, "#facts"], "incremental_mismatch",
                            "incremental result differed from the full evaluator at byte "
                            f"{plan.snapshot['offset']}")
                except (sqlite3.Error, OSError):
                    pass
            report["shadow_compared"] = True
        payload = result.model_dump(mode="json")
        payload.update(ok=True, provider=provider, action_executed=False,
                       applicable_contracts=len(active), requires_attention=result.verdict != "clean")
        if planned is not None:
            action_rules = [rule for rule in active if action_applies(rule, planned)]
            payload["action_scope"] = "applicable" if action_rules else "not_applicable"
            payload["action_contracts"] = [rule.id for rule in action_rules]
            payload["requires_history"] = any(rule.type == "must" for rule in action_rules)
        payload["timings_ms"] = {
            "sync": round((sync_finished - started) * 1000, 2),
            "evaluation": round((time.perf_counter() - sync_finished) * 1000, 2)}
        if report:
            payload["incremental"] = report
        payload["nudge"] = (
            "Detect only. violation/suspicious: STOP and report; unknown: resolve missing evidence. "
            "clean is limited to the reported contracts and history snapshot, not execution authorization.")
        payload["global_meaning"] = GLOBAL_CONTRACT_MEANING
        payload["contract_scope"] = {"session_id": sid, "scope": contract_scope(sid)}
        # Keep the full result: a transient read error does not latch a mismatch, so a later
        # successful bounded tail check must not certify this read.
        source_ok = observation is None or observation.validate_prefix()  # bounded stage check (identity, size, tail; see validate_prefix), off the shared lock
        with self._publication_guard():
            stale = []
            tokens = (
                ("metadata", metadata_token, lambda: self.store.state_token(sid, provider)),
                ("contracts", contract_token, self.contracts.current_token),
                ("receipts", receipt_token, lambda: self._receipt_token(sid, provider)),
            )
            for label, expected, read_token in tokens:
                try:
                    if read_token() != expected:
                        stale.append(label + "_changed_during_evaluation")
                except (RequestStopped, IndexBusy):
                    raise
                except (OSError, ValueError, sqlite3.DatabaseError) as exc:
                    stale.append(label + "_authority_unavailable:" + type(exc).__name__)
            known = bool(payload["coverage"].get("judged_prefix"))
            grown = False
            if observation is not None:
                # Pure growth is judged as of the captured offset; any rewrite, truncation, replacement or
                # deletion fails the prefix check and stays unknown.
                if not source_ok or not observation.validate_prefix(locked=True):
                    stale.append("source_changed_during_evaluation")
                else:
                    grown = _mark_growth(payload["coverage"], observation)
            if grown and not known and payload["coverage"].get("judged_prefix") and not stale:
                if downgrade_unproven_absence(payload["findings"], active):
                    payload["verdict"] = "clean"
                    for item in payload["findings"]:
                        payload["verdict"] = _worse(payload["verdict"], item["verdict"])
                payload["notes"].append(prefix_note(payload["coverage"]))
            if stale:
                payload["coverage"].pop("judged_prefix", None)
                payload["snapshot_verdict"] = payload["verdict"]
                payload.update(verdict="unknown", requires_attention=True)
                payload["coverage"]["complete"] = False
                payload["coverage"]["issues"].extend(stale)
                payload["notes"].append("Authority changed or became unavailable; findings describe the captured snapshot, not current compliance.")
            if planned is not None and payload["requires_history"] and not payload["coverage"].get("complete"):
                # A retrospective audit may be clean as of its offset, but a proposed action runs now: unread
                # records precede it. Pure append cannot erase an observed run, yet a newer failed or pending
                # run can invalidate an older success; any other coverage gap invalidates both.
                gate_ids = {rule.id for rule in action_rules if rule.type == "must"
                            and (rule.requires_success or not payload["coverage"].get("judged_prefix"))}
                for item in payload["findings"]:
                    if item["contract_id"] in gate_ids and item["verdict"] == "clean":
                        item.update(verdict="unknown", applicability="applicable",
                                    reason="incomplete current history cannot establish the proposed action's prerequisite",
                                    basis=(item.get("basis") or "") + "; captured prerequisite evidence does not establish current compliance")
                if gate_ids:
                    payload["verdict"] = _worse(payload["verdict"], "unknown")
                    payload["notes"].append("Current action prerequisites remain unverified while their required evidence is incomplete.")
            payload["requires_attention"] = payload["verdict"] != "clean"
            payload["integrity"] = integrity
            if plan is not None and not mismatch:
                # The monitor state is tied to the index commit it evaluated, not to how the source was verified: if a
                # later full verification finds a rewrite, the reindex starts a new generation and drops these states.
                try:
                    if plan.timeouts:
                        self.store.drop_audit_states(sid, provider, sorted(plan.timeouts))
                    rows = plan.state_rows()
                    report["state"] = (self.store.save_audit_states(sid, provider, plan.digest, rows, plan.snapshot)
                                       if rows else "unchanged")
                except (sqlite3.Error, OSError) as exc:
                    report["state"] = "write_failed:" + type(exc).__name__  # a cache; the next audit rebuilds it
            reason = _fast_gate(payload, integrity) if integrity.get("mode") == "fast" else None
            if reason is None:
                with publication():
                    return payload
        # Leave the guard before the long work: the fast verdict rested on evidence only a full scan can vouch for.
        return self._audit(session_id, provider, proposed=proposed, path=path, verify="full", escalated=reason)

    def _receipt_token(self, sid, provider):
        if provider != "agy":
            return None
        from self_directing_mcp.agy_receipts import ReceiptStore
        receipts = ReceiptStore(Path(self.settings.index_dir))
        return receipts.authority_token(sid)

    @detached_operation
    def audit_session(self, session_id: str, provider: str | None = None, path: str | None = None) -> dict[str, Any]:
        """Always a full verification of the source (``integrity.mode == "full"``); there is no fast variant."""
        return self._audit(session_id, provider, path=path, verify="full")

    @detached_operation
    def check_action(self, session_id: str, action: dict[str, Any], provider: str | None = None, path: str | None = None,
                     verify: str = "full"):
        """Check a proposed tool call against refreshed history; never execute or persist it.

        ``verify="fast"`` is for hook-time checks: the source is verified by comparing its tail, latest blocks and a
        random sample with the stored block digests instead of a scan of everything (see ``integrity`` in the
        result, and the README's "Source integrity"). It is never used implicitly.
        """
        return self._audit(session_id, provider, proposed=action, path=path, verify=_check_verify(verify))

    @detached_operation
    def get_chunk(self, chunk_id: str) -> dict[str, Any]:
        assert self.store
        with self._publication_guard():
            snapshot = self.store.capture_evidence(history=False, anchors=(chunk_id,))
        chunk = snapshot.get_chunk(chunk_id)
        if not chunk:
            return {"ok": False, "found": False, "chunk_id": chunk_id}
        data = chunk.model_dump(mode="json")
        data = mask_value(data)
        return {"ok": True, "found": True, "chunk": data}

    @detached_operation
    def list_contracts(self, session_id: str | None = None) -> dict[str, Any]:
        assert self.contracts
        with self._publication_guard():
            raw, _token = self.contracts.capture()
        stored = (ContractsDocument.model_validate_json(raw).contracts if raw is not None else [])
        selected = visible_contracts(stored, session_id)
        return {
            "ok": True,
            "session_id": session_id,
            "scope": contract_scope(session_id),
            "global_meaning": GLOBAL_CONTRACT_MEANING,
            "contracts": [c.model_dump(mode="json") for c in selected],
            "counts": scope_counts(selected, len(stored)),
        }

    @detached_operation
    def revoke_contract(self, contract_id: str) -> dict[str, Any]:
        """Disable a rule while keeping its history and source quote for audit."""
        assert self.contracts
        with self._publication_guard():
            raw, token = self.contracts.capture()
            document = ContractsDocument.model_validate_json(raw) if raw is not None else ContractsDocument()
            disabled = self.contracts.prepare_revoke(document, contract_id)
            if disabled is None:
                return {"ok": False, "error": "not_found", "contract_id": contract_id}
            payload = document.model_dump_json(indent=2)
            self.contracts.publish_document(document, payload, token)
        return {"ok": True, "contract": disabled.model_dump(mode="json"),
                "nudge": "Contract disabled. History retained; re-enable via upsert_contracts with the same id."}

    @detached_operation
    def upsert_contracts(self, contracts: list[dict[str, Any] | ContractRule]) -> dict[str, Any]:
        assert self.contracts
        rules: list[ContractRule] = []
        for c in contracts:
            if isinstance(c, ContractRule):
                rules.append(c)
            else:
                rules.append(ContractRule.model_validate(c))
        for rule in rules:
            for pattern in (rule.regex, rule.before_regex, rule.exception_regex,
                            rule.when.tool_call_regex if rule.when else None):
                if pattern:
                    compile_contract(pattern)   # the syntax gate of both engines (raises re.error)
            if rule.risk == "high" and rule.type == "must_not" and not rule.regex:
                # Applies to every action and can never be verified: every tool call would be unknown, hence asked
                # (a denial when headless). A high-risk target must name the actions it covers.
                raise ValueError(f"contract {rule.id!r}: a high-risk must_not needs a regex that names the actions it "
                                 "covers; without one it would ask on every tool call")
        with self._publication_guard():
            raw, token = self.contracts.capture()
            document = ContractsDocument.model_validate_json(raw) if raw is not None else ContractsDocument()
            updated = self.contracts.prepare_upsert(document, rules)
            payload = document.model_dump_json(indent=2)
            self.contracts.publish_document(document, payload, token)
        return {
            "ok": True,
            "returned": "upserted",
            "global_meaning": GLOBAL_CONTRACT_MEANING,
            "contracts": [c.model_dump(mode="json") for c in updated],
            "counts": {
                "upserted": len(updated),
                "store": len(document.contracts),
            },
        }

    def _verified_checklist_evidence(self, evidence_ids, chunks, complete) -> bool:
        """Evidence IDs are claims until current, linked runtime results resolve."""
        if not evidence_ids or not complete:
            return False
        positions = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        for identifier in evidence_ids:
            result = by_id.get(identifier)
            if result is None or result.kind != "tool_result" or result.meta.get("success") is not True:
                return False
            call_id = result.meta.get("call_id")
            calls = [chunk for chunk in chunks if chunk.kind == "tool_call" and chunk.meta.get("call_id") == call_id]
            outputs = [chunk for chunk in chunks if chunk.kind == "tool_result" and chunk.meta.get("call_id") == call_id]
            if (not call_id or len(calls) != 1 or len(outputs) != 1
                    or positions[calls[0].chunk_id] >= positions[result.chunk_id]):
                return False
        return True

    def _capture_checklist(self, sid):
        with self._publication_guard():
            path = self.checklists._path(sid)
            try:
                raw = path.read_bytes()
            except FileNotFoundError:
                raw = None
        return (ChecklistDocument.model_validate_json(raw) if raw is not None else ChecklistDocument(session_id=sid)), raw

    def _evidence_current(self, sid, provider, refreshed):
        observation = refreshed.get("_source_observation")
        return (self.store.state_token(sid, provider) == refreshed["_metadata_token"]
                and observation is not None and observation.validate_prefix(locked=True)
                and self._receipt_token(sid, provider) == refreshed.get("_receipt_token"))

    def _save_checklist(self, doc, original, provider, refreshed=None):
        payload = doc.model_dump_json(indent=2)
        source_ok = True
        if refreshed is not None and refreshed.get("_source_observation") is not None:
            source_ok = refreshed["_source_observation"].validate_prefix()  # bounded stage check (identity, size, tail; see validate_prefix), off the shared lock
        with self._publication_guard():
            path = self.checklists._path(doc.session_id)
            try:
                current = path.read_bytes()
            except FileNotFoundError:
                current = None
            if current != original:
                raise ChecklistConflict("stale checklist preparation")
            if refreshed is not None and (not source_ok or not self._evidence_current(doc.session_id, provider, refreshed)):
                raise ChecklistConflict("checklist evidence changed during evaluation")
            temporary = path.with_suffix(path.suffix + ".tmp")
            try:
                temporary.write_text(payload, encoding="utf-8")
                with publication():
                    if refreshed is not None and (not source_ok or not self._evidence_current(doc.session_id, provider, refreshed)):
                        raise ChecklistConflict("checklist evidence changed before publication")
                    check_request()
                    temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)

    @detached_operation
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
        assert self.checklists
        resolved_provider = self._resolve_provider(provider)
        sid = session_id.lower() if resolved_provider in ("codex", "agy") else session_id
        needs_evidence = any(change.get("status") == RequirementStatus.DONE.value or "evidence_chunk_ids" in change
                             for change in update or [])
        if not needs_evidence:
            with self._publication_guard():
                doc, original = self._capture_checklist(sid)
                added, updated = self._apply_checklist_changes(doc, add, update, None, [])
                self._save_checklist(doc, original, resolved_provider)
        else:
            for _attempt in range(CHECKLIST_EVALUATION_ATTEMPTS):
                doc, original = self._capture_checklist(sid)
                refreshed = self._refresh_evidence(sid, resolved_provider)
                with self._publication_guard(), self.store._read_boundary():
                    evidence = self.store.capture_evidence(sid, resolved_provider)
                    if self.store.state_token(sid, resolved_provider) != refreshed["_metadata_token"]:
                        refreshed["coverage"]["complete"] = False
                evidence_chunks = evidence.list_chunks(sid, resolved_provider)
                added, updated = self._apply_checklist_changes(doc, add, update, refreshed, evidence_chunks)
                proof = refreshed if any(item.get("verified") and item.get("status") == RequirementStatus.DONE
                                         for item in updated) else None
                try:
                    self._save_checklist(doc, original, resolved_provider, proof)
                    break
                except ChecklistConflict:
                    continue
            else:
                return {"ok": False, "status": "checklist_conflict", "action_executed": False,
                        "attempts": CHECKLIST_EVALUATION_ATTEMPTS,
                        "message": "Checklist or evidence changed during every evaluation attempt; nothing was saved. Retry."}
        remaining = [i for i in doc.items
                     if i.status not in (RequirementStatus.DONE, RequirementStatus.REVOKED)]
        return {
            "ok": True,
            "added": added,
            "updated": updated,
            "remaining": [i.model_dump(mode="json") for i in remaining],
            "nudge": INCOMPLETE_NOTE if remaining else None,
        }

    def _apply_checklist_changes(self, doc, add, update, refreshed, evidence_chunks):
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
            if change.get("status") == RequirementStatus.DONE.value or (
                    "evidence_chunk_ids" in change and item.status == RequirementStatus.DONE):
                verified = (refreshed is not None and refreshed.get("ok") is True
                            and change.get("verified") is not False
                            and self._verified_checklist_evidence(item.evidence_chunk_ids, evidence_chunks,
                                                                  refreshed["coverage"]["complete"]))
                item.status = RequirementStatus.DONE if verified else RequirementStatus.PENDING_VERIFICATION
                item.verified = verified
            elif change.get("status"):
                item.status = change["status"]
            if change.get("blocked_reason"):
                item.blocked_reason = change["blocked_reason"]
            item.history.append({"change": "updated", "from": before, "to": item.status,
                                 "evidence": len(item.evidence_chunk_ids)})
            updated.append(item.model_dump(mode="json"))
        return added, updated

    @detached_operation
    def get_checklist(self, session_id: str, provider: str | None = None) -> dict[str, Any]:
        """Restore requirements and remaining items after compaction."""
        assert self.checklists
        resolved_provider = self._resolve_provider(provider)
        sid = session_id.lower() if resolved_provider in ("codex", "agy") else session_id
        doc, _original = self._capture_checklist(sid)
        with self._publication_guard(), self.store._read_boundary():
            record = self.store.cursor_record(sid, resolved_provider)
            metrics = self.store.session_metrics(sid, resolved_provider)
            metadata_token = self.store.state_token(sid, resolved_provider)
            receipt_token = self._receipt_token(sid, resolved_provider)
            evidence = self.store.capture_evidence(sid, resolved_provider)
        try:
            observation = (capture_for_cursor(resolved_provider, Path(record["path"]), record,
                                              (record["byte_offset"],), self.store.max_decompressed_bytes)
                           if record else None)
        except RequestStopped:
            raise
        except (OSError, ValueError):
            observation = None
        coverage = coverage_from_capture(sid, resolved_provider, record, metrics, observation,
                                         max_bytes=self.store.max_decompressed_bytes)
        if resolved_provider == "agy" and record and record.get("receipt_digest") != receipt_token[2]:
            coverage["complete"] = False
            coverage["issues"].append("receipts_not_indexed")
        chunks = evidence.list_chunks(sid, resolved_provider)
        for item in doc.items:
            if item.status == RequirementStatus.DONE and (
                    not item.verified or not self._verified_checklist_evidence(item.evidence_chunk_ids, chunks,
                                                                               coverage["complete"])):
                item.status = RequirementStatus.PENDING_VERIFICATION
                item.verified = False
        source_ok = observation is not None and observation.validate_prefix()  # bounded stage check (identity, size, tail; see validate_prefix), off the shared lock
        with self._publication_guard():
            state = {"_metadata_token": metadata_token, "_source_observation": observation,
                     "_receipt_token": receipt_token}
            if not source_ok or not self._evidence_current(sid, resolved_provider, state):
                for item in doc.items:
                    if item.status == RequirementStatus.DONE:
                        item.status, item.verified = RequirementStatus.PENDING_VERIFICATION, False
        return {"ok": True, "session_id": sid, "items": [i.model_dump(mode="json") for i in doc.items]}

    @detached_operation
    def analyze_activity(self, session_id: str, provider: str | None = None,
                         window_minutes: int = 60) -> dict[str, Any]:
        """SQLite-based activity time-series analysis with evidence ids."""
        assert self.store
        resolved_provider = self._resolve_provider(provider)
        sid = session_id.lower() if resolved_provider in ("codex", "agy") else session_id
        with self._publication_guard():
            snapshot = self.store.capture_evidence(sid, resolved_provider)
            try:
                checklist_raw = self.checklists._path(sid).read_bytes()
            except FileNotFoundError:
                checklist_raw = None
        snapshot = replace(snapshot, checklist_raw=checklist_raw)
        return analyze_activity(snapshot, session_id=sid, provider=resolved_provider,
                                window_minutes=max(1, min(window_minutes, 1440)))

    @detached_operation
    def get_graph_context(
        self,
        session_id: str,
        node_id: str | None = None,
        provider: str | None = None,
        depth: int = 1,
        path: str | None = None,
    ) -> dict[str, Any]:
        """Fetch GraphRAG context: entity neighbor subgraph or overall session graph."""
        assert self.graph is not None
        resolved_provider = self._resolve_provider(provider)
        sid = session_id.lower() if resolved_provider in ("codex", "agy") else session_id
        if path is not None:
            synced = self._refresh_evidence(sid, resolved_provider, path)
            if not synced.get("ok") or not synced.get("coverage", {}).get("complete"):
                return {"ok": False, "error": synced.get("error", "incomplete_history"), "nodes": [], "edges": []}

        source_ok = True
        if path is not None and synced.get("_source_observation") is not None:
            source_ok = synced["_source_observation"].validate_prefix()  # bounded stage check (identity, size, tail; see validate_prefix), off the shared lock
        with self._publication_guard():
            if path is not None and (not source_ok or not self._evidence_current(sid, resolved_provider, synced)):
                return {"ok": False, "error": "incomplete_history", "nodes": [], "edges": []}
            cursor = self.graph.cursor(sid, resolved_provider)
            if node_id:
                res = self.graph.neighbors(sid, resolved_provider, node_id, depth=depth)
                if not res.get("edges"):
                    matches = self.graph.find_nodes_like(node_id, sid, resolved_provider)
                    if matches:
                        res = self.graph.neighbors(sid, resolved_provider, matches[0]["node_id"], depth=depth)
                return {"ok": True, "session_id": sid, "provider": resolved_provider,
                        "node_id": node_id, "subgraph": res, "cursor": cursor}
            graph_data = self.graph.get_session_graph(sid, resolved_provider)
            return {"ok": True, "session_id": sid, "provider": resolved_provider,
                    "nodes": graph_data["nodes"], "edges": graph_data["edges"], "cursor": cursor}

    @detached_operation
    @graph_operation("validate_graph_proposal")
    def propose_graph_update(self, session_id: str, proposal: dict[str, Any],
                             provider: str | None = None) -> dict[str, Any]:
        """Validate a node/edge proposal against indexed evidence; nothing stored here."""
        assert self.store and self.graph
        resolved_provider = self._resolve_provider(provider)
        sid = session_id.lower() if resolved_provider in ("codex", "agy") else session_id
        anchors = {cid for edge in proposal.get("edges", []) for cid in edge.get("evidence_chunk_ids", [])}
        with self._publication_guard(), self.store._read_boundary():
            snapshot = self.store.capture_evidence(sid, resolved_provider, history=False, anchors=anchors)
            metadata_token = self.store.state_token(sid, resolved_provider)
            cursor = self.graph.cursor(sid, resolved_provider)
        verdict = validate_proposal(proposal, store=snapshot, session_id=sid, provider=resolved_provider)
        with self._publication_guard():
            if self.store.state_token(sid, resolved_provider) != metadata_token:
                return {"ok": False, "status": "evidence_conflict", "cursor": cursor}
        return {"ok": verdict["ok"], **verdict, "cursor": cursor,
                "nudge": None if verdict["ok"] else "Fix problems and resubmit; nothing was stored."}

    @detached_operation
    @graph_operation("commit_graph_proposal")
    def commit_graph_update(self, session_id: str, proposal: dict[str, Any],
                            base_graph_version: int, provider: str | None = None) -> dict[str, Any]:
        """Validate detached evidence, then CAS the graph and metadata generations."""
        resolved_provider = self._resolve_provider(provider)
        sid = session_id.lower() if resolved_provider in ("codex", "agy") else session_id
        anchors = {cid for edge in proposal.get("edges", []) for cid in edge.get("evidence_chunk_ids", [])}
        with self._publication_guard(), self.store._read_boundary():
            existing = self.graph.find_job(proposal.get("job_id", "") or "adhoc")
            if existing and existing["status"] == "applied":
                return {"ok": True, "status": "already_applied", "job": existing}
            cursor = self.graph.cursor(sid, resolved_provider)
            metadata_token = self.store.state_token(sid, resolved_provider)
            snapshot = self.store.capture_evidence(sid, resolved_provider, history=False, anchors=anchors)
        if base_graph_version != cursor["graph_version"]:
            return {"ok": False, "status": "version_conflict",
                    "expected_base_graph_version": cursor["graph_version"],
                    "provided_base_graph_version": base_graph_version,
                    "nudge": "Re-read the current graph and rebuild the proposal against it."}
        verdict = validate_proposal(proposal, store=snapshot, session_id=sid, provider=resolved_provider)
        if not verdict["ok"]:
            return {"ok": False, "status": "rejected", **verdict}
        through = snapshot.count - 1 if snapshot.count else cursor["processed_through_order"]
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        job = {"job_id": proposal.get("job_id", "adhoc"), "provider": resolved_provider,
               "session_id": sid, "range_start": cursor["processed_through_order"],
               "range_end": through, "input_digest": str(hash(json.dumps(proposal, sort_keys=True))),
               "base_graph_version": base_graph_version, "status": "pending",
               "created_at": now, "updated_at": now}
        with self._publication_guard():
            existing = self.graph.find_job(proposal.get("job_id", "") or "adhoc")
            if existing and existing["status"] == "applied":
                return {"ok": True, "status": "already_applied", "job": existing}
            current_version = self.graph.cursor(sid, resolved_provider)["graph_version"]
            if current_version != base_graph_version:
                return {"ok": False, "status": "version_conflict",
                        "expected_base_graph_version": current_version,
                        "provided_base_graph_version": base_graph_version,
                        "nudge": "Re-read the current graph and rebuild the proposal against it."}
            if self.store.state_token(sid, resolved_provider) != metadata_token:
                return {"ok": False, "status": "evidence_conflict"}
            result = self.graph.apply_proposal(proposal, session_id=sid, provider=resolved_provider,
                                               base_version=base_graph_version, through_order=through)
            job["status"], job["detail"] = result["status"], result.get("detail")
            self.graph.record_job(job)
            return result

    @detached_operation
    def _prepare_neograph_update(self, session_id, provider=None):
        resolved = self._resolve_provider(provider)
        sid = session_id.lower() if resolved in ("codex", "agy") else session_id
        with self._publication_guard():
            snapshot = self.store.capture_evidence(sid, resolved)
            cursor = self.graph.cursor(sid, resolved)
        return sid, resolved, snapshot.event_frames(sid, resolved), cursor

    @workflow_operation
    def _commit_neograph_snapshot(self, sid, provider, proposal, base_version, through,
                                  *, _deadline=None, cancel_event=None):
        anchors = {cid for edge in proposal.get("edges", []) for cid in edge.get("evidence_chunk_ids", [])}
        with self._publication_guard(), self.store._read_boundary():
            if self.graph.cursor(sid, provider)["graph_version"] != base_version:
                return {"ok": False, "status": "version_conflict"}
            metadata_token = self.store.state_token(sid, provider)
            snapshot = self.store.capture_evidence(sid, provider, history=False, anchors=anchors)
        verdict = validate_proposal(proposal, store=snapshot, session_id=sid, provider=provider)
        if not verdict["ok"]:
            return {"ok": False, "status": "rejected", **verdict}
        with self._publication_guard():
            if self.graph.cursor(sid, provider)["graph_version"] != base_version:
                return {"ok": False, "status": "version_conflict"}
            if self.store.state_token(sid, provider) != metadata_token:
                return {"ok": False, "status": "evidence_conflict"}
            return self.graph.apply_proposal(proposal, session_id=sid, provider=provider,
                                             base_version=base_version, through_order=through)

    @workflow_operation
    def run_neograph_update(self, session_id: str, provider: str | None = None,
                            *, _deadline=None, cancel_event=None) -> dict[str, Any]:
        """Run the full update pipeline as a NeoGraph topology with the live LLM.

        Requires OPENROUTER_API_KEY and the neograph-engine package. Steps:
        extract (real LLM) -> validate (MCP evidence checks) -> apply (atomic
        write + cursor). Each run leaves a durable checkpoint; failures stay
        inspectable via thread_id graphrag:<provider>:<session>.
        """
        if self.settings.local_only:
            return {"ok": False, "error": "local_only",
                    "message": "LLM graph updates are unavailable in local-only mode."}
        sid, resolved_provider, frames, cursor = self._prepare_neograph_update(
            session_id, provider, _deadline=_deadline, cancel_event=cancel_event)
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
            batch.append(mask_value(frame))
            size += cost
        if not batch:
            return {"ok": False, "error": "graph_event_too_large",
                    "chunk_id": pending[0]["chunk_id"], "graph_cursor_after": cursor,
                    "message": "The next complete event exceeds the graph input budget; no text was truncated."}
        proposal_result: dict[str, Any] = {}

        def on_proposal(step: str, proposal: Any) -> dict[str, Any]:
            check_request()
            if step == "validate":
                verdict = self.propose_graph_update(sid, proposal, provider=resolved_provider,
                                                      _deadline=_deadline, cancel_event=cancel_event)
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
            _deadline=_deadline, cancel_event=cancel_event)
        # Preserve actual completed workflow receipts even after response expiry.
        with self._lock:
            pipeline["graph_cursor_after"] = self.graph.observed_cursor(sid, resolved_provider)
        pipeline["batch_events"] = len(batch)
        applied = pipeline.get("apply_result", {}).get("status") == "applied"
        pipeline["ok"] = applied
        pipeline["remaining_events"] = len(pending) - (len(batch) if applied else 0)
        return pipeline

    @detached_operation
    def audit_status(self, session_id: str | None = None, provider: str | None = None) -> dict[str, Any]:
        resolved = self._resolve_provider(provider)
        sid = session_id.lower() if session_id and resolved in ("codex", "agy") else session_id
        status, raw_contracts, record, metrics, metadata_token, receipt_token = self._capture_status(sid, resolved)
        status["contract_count"] = len(ContractsDocument.model_validate_json(raw_contracts).contracts) if raw_contracts is not None else 0
        if sid:
            status["session"] = coverage_from_capture(sid, resolved, record, metrics,
                                                      max_bytes=self.store.max_decompressed_bytes)
            if resolved == "agy" and record and record.get("receipt_digest") != receipt_token[2]:
                status["session"]["complete"] = False
                status["session"]["issues"].append("receipts_not_indexed")
            with self._publication_guard():
                if self.store.state_token(sid, resolved) != metadata_token:
                    status["session"]["complete"] = False
                    status["session"]["issues"].append("metadata_changed_during_status")
                if self._receipt_token(sid, resolved) != receipt_token:
                    status["session"]["complete"] = False
                    status["session"]["issues"].append("receipts_changed_during_status")
        return status

    @serialized
    def _capture_status(self, session_id=None, provider=None):
        self.ensure_ready()
        assert self.store and self.sparse
        s = self.settings
        if self.dense is not None:
            self.dense.refresh()
        status: dict[str, Any] = {
            "ok": True,
            "orchestration_backend": "neograph-engine",
            "neograph_version": ng.__version__,
            "sessions_dir": str(s.resolve_sessions_dir()),
            "grokbot_transcripts_dirs": [str(p) for p in s.resolve_grokbot_transcripts_dirs()],
            "omp_sessions_dir": str(s.resolve_omp_sessions_dir()),
            "claude_projects_dir": str(s.resolve_claude_projects_dir()),
            "session_provider": s.resolve_session_provider(),
            "index_dir": str(Path(s.index_dir).resolve()),
            "dense_backend": self.dense_backend,
            "dense_count": self.dense.count() if self.dense is not None else 0,
            "vector_extension_version": getattr(self.dense, "extension_version", None),
            "vector_compute_backend": getattr(self.dense, "compute_backend", None),
            "embedder": type(self.embedder).__name__ if self.embedder else None,
            "embedder_degraded": self.embedder_degraded,
            "degraded_flags": [],
            "total_chunks": self.store.chunk_count(),
            "legacy_chunks_retained": self.store.legacy_count(),
            "index_schema": 2,
            "sparse_count": self.sparse.count(),
        }
        worker = self._integrity_worker
        status["integrity_worker"] = worker.snapshot() if worker is not None else None
        if self.embedder_degraded:
            status["degraded_flags"].append("embedder_fake_fallback")
        if self.dense_backend == "numpy":
            status["degraded_flags"].append("dense_numpy_fallback")
        raw_contracts, _contract_token = self.contracts.capture()
        with self.store._read_boundary():
            record = self.store.cursor_record(session_id, provider) if session_id else None
            metrics = self.store.session_metrics(session_id, provider) if session_id else None
            metadata_token = self.store.state_token(session_id, provider) if session_id else None
            receipt_token = self._receipt_token(session_id, provider) if session_id else None
        status["nudge"] = (
            "NUDGE: Call sync_session at session start; audit_session before risky actions."
        )
        return status, raw_contracts, record, metrics, metadata_token, receipt_token

    def close(self) -> None:
        """Wait for request-owned snapshots/preparation, then release local handles."""
        with self._lifetime:
            self._closing = True
            try:
                while self._active_operations:
                    self._lifetime.wait()
                for component in (self.store, self.sparse, self.dense, self.graph):
                    connection = getattr(component, "_conn", None)
                    if connection is not None:
                        connection.close()
                self.store = self.sparse = self.dense = self.retriever = self.contracts = None
                self.graph = self.neograph = None
                self._ready = False
            finally:
                self._closing = False
                self._lifetime.notify_all()
