from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING
from functools import wraps
from threading import Condition, RLock
from collections import OrderedDict
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from datetime import datetime, timezone
import copy
import hashlib
import json
import sqlite3
import re
import time

from self_directing_mcp.compressed_source import CompressedSourceError
from self_directing_mcp.session_events import make_chunk
from self_directing_mcp.tool_payload import tool_call_parts

from self_directing_mcp.audit.contracts import (
    GLOBAL_CONTRACT_MEANING,
    MEMO_MAX_BYTES,
    ContentMemo,
    ContractStore,
    contract_scope,
    loosening_reasons,
    parse_contracts,
    scope_counts,
    visible_contracts,
)
from self_directing_mcp.audit import incremental
from self_directing_mcp.audit import shadow as shadow_policy
from self_directing_mcp.audit.patterns import compile_contract
from self_directing_mcp import eval_cost
from self_directing_mcp.eval_cost import COSTS
from self_directing_mcp.slice_budget import Sizing, sizing_scope
from self_directing_mcp.audit.runner import (
    REGEX_TIMEOUT, _worse, action_applies, audit_needs, downgrade_unproven_absence, prefix_note, run_audit,
)
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
from self_directing_mcp.embed.embedder import build_embedder
from self_directing_mcp.index.ingest import (
    AmbiguousSessionSource, AuditEventSnapshot, FlightBusy, IntegrityPolicy, SessionStore, SlicePolicy,
    capture_for_cursor, coverage_from_capture, ingest_session,
)
from self_directing_mcp.index.sparse import SparseIndex, count_sparse_file, open_sparse_index, sparse_file
from self_directing_mcp.index.vector import build_vector_index
from self_directing_mcp.index.locking import index_lock
from self_directing_mcp.retrieve.checks import regex_search
from self_directing_mcp.retrieve.hybrid import HybridRetriever, hits_to_schema
from self_directing_mcp.schemas import ContractRule, ContractsDocument, PendingChange, ProposedAction
from self_directing_mcp.security.mask import mask_secrets, mask_value
from self_directing_mcp.timeseries import analyze_activity
from self_directing_mcp import graph_batch
from self_directing_mcp.graphrag import GraphStore, validate_proposal
from self_directing_mcp.neograph_runner import run_update_pipeline
from self_directing_mcp.neograph_runtime import ng, graph_operation
from self_directing_mcp.request_control import (
    IndexBusy, RequestStopped, check_request, current_request, publication, request_operation, request_scope,
)
from self_directing_mcp import request_stats

if TYPE_CHECKING:       # the pydantic Settings model is imported lazily: command hooks pass a light settings object
    from self_directing_mcp.config import Settings


_OBLIGATIONS = ContentMemo(16)


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

def background_operation(method):
    """A background worker's operation: it never reopens an engine that was closed (``detached_operation`` would)."""
    @request_operation
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._components(reopen=False):
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

# A deadline-bound audit or action check gives its evidence refresh this share of the time it has left and keeps the rest
# for judging; see ``_refresh_evidence``.
SYNC_BUDGET_SHARE = 0.6
SPARSE_REPAIR_BATCH = 500     # documents per transaction when a search or the rebuild command materializes the sparse index


def _request_over(control) -> bool:
    """Whether the request's own stop (its deadline, a cancellation) has fired, as opposed to a narrower budget in it."""
    try:
        control.check()
    except RequestStopped:
        return True
    return False


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
PENDING_NUDGE = ("NOT APPLIED: this change would weaken an active contract and no user message vouches for it "
                 "(source_event_id missing or not found). The previous revision stays in force and the change is held "
                 "as pending. Repeat it with source_event_id set to the user's own message that asked for it.")


class ChecklistConflict(ValueError):
    """Checklist file or evidence moved between capture and publication."""


class SelfDirectEngine:
    def __init__(self, settings: Settings | None = None, *, defer_derived: bool = False) -> None:
        """``defer_derived``: a resident process (the daemon) holds the rows that carry no authority (monitor states,
        verification stamps, the AGY indexed marker) in memory and writes them with the next authority commit, at idle
        and at ``close`` instead of committing each synchronously; see ``index.deferred``. Off for every other use."""
        self._lock = RLock()
        self._lifetime = Condition(self._lock)
        self._active_operations = 0
        self._closing = False
        self._operation_depth = 0
        self._hook_lock = RLock()
        self._hook_states = OrderedDict()
        self._last_activity = time.monotonic()
        self._integrity_worker = None  # set only by a resident server (see integrity_worker)
        self._shadow_worker = None     # likewise: the idle comparison of incremental audits (see shadow_worker)
        self.integrity_policy: IntegrityPolicy | None = None  # tests may inject a clock / randomness
        self.slice_policy: SlicePolicy | None = None  # tests may inject forced stops / a clock (default: the settings)
        if settings is None:
            from self_directing_mcp.config import get_settings    # the server's pydantic settings; hooks pass their own
            settings = get_settings()
        self.settings = settings
        self.defer_derived = defer_derived
        self.store: SessionStore | None = None
        self.sparse: SparseIndex | None = None
        self.sparse_resolution: dict[str, Any] = {}     # how SELF_DIRECT_SPARSE_FORMAT was resolved (open_sparse_index)
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
    def _components(self, reopen=True):
        # Refcount lifetime without retaining either publication lock over CPU/I/O. ``reopen=False``: a background worker's
        # operation, which must not initialize an engine that was closed.
        with self._publication_guard():
            if self._closing or not (reopen or self._ready):
                raise IndexBusy()
            self.ensure_ready()
            self._active_operations += 1
        try:
            worker = self._shadow_worker
            if worker is not None:
                # A request outranks the idle comparison: it stops at its next cooperative check. After the count above, so
                # that a job that publishes its cancel event and then asks whether the engine is idle sees this request, and a
                # job that published it earlier is cancelled here (done in this order on both sides, no request is missed).
                worker.preempt()
            yield
        finally:
            with self._lifetime:
                self._active_operations -= 1
                self._last_activity = time.monotonic()
                self._lifetime.notify_all()

    def _policy(self) -> IntegrityPolicy:
        return self.integrity_policy or IntegrityPolicy.from_settings(self.settings)

    def _slice_policy(self) -> SlicePolicy:
        return self.slice_policy or SlicePolicy.from_settings(self.settings)

    def attach_integrity_worker(self, worker) -> None:
        self._integrity_worker = worker

    def attach_shadow_worker(self, worker) -> None:
        self._shadow_worker = worker

    def flush_derived(self) -> int:
        """Write the derived rows a ``defer_derived`` engine is holding (the daemon calls it when idle). Cheap and lock-free
        when nothing is held; never raises for a busy index or a stopped request (the rows stay held for the next call)."""
        store = self.store
        if store is None or not self._ready:
            return 0
        pending = store.journal.pending is not None and store.journal.retry_due()
        if not pending and not store.has_pending_derived():
            return 0
        try:
            with self._publication_guard():
                if self.store is not store or not self._ready:
                    return 0                  # closed while this waited for the lease: close wrote what was held
                if pending:
                    store.retry_journal()     # a refused journal-mode switch is tried again at a quiet point
                if not store.has_pending_derived():
                    return 0
                return store.flush_derived() + self._receipts(held=store.derived).flush_held()
        except (IndexBusy, RequestStopped):
            return 0

    def _journal_policy(self) -> str | None:
        """``auto`` or ``delete`` (SELF_DIRECT_SQLITE_JOURNAL); None leaves the package default to the stores."""
        return getattr(self.settings, "sqlite_journal", None)

    def _receipts(self, **kwargs):
        from self_directing_mcp.agy_receipts import ReceiptStore
        return ReceiptStore(Path(self.settings.index_dir), journal=self._journal_policy(), **kwargs)

    def _held_rows(self):
        """The buffer a ``defer_derived`` engine holds its derived rows in, or None (nothing is deferred)."""
        store = self.store
        return store.derived if self.defer_derived and store is not None else None

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
        raw = path.read_bytes() if path.exists() else None
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sid)[:120]
        path = Path(self.settings.index_dir) / "checklists" / f"{safe}.json"
        checklist_raw = path.read_bytes() if path.exists() else None
        # Everything below is a function of these four values, so they are the key: a changed contract file, a changed
        # checklist, another session or provider is another entry, and an unchanged one costs no parsing, no
        # serialization and no hashing (this runs on every hook).
        key = (raw, checklist_raw, sid, provider)
        memoize = len(raw or b"") + len(checklist_raw or b"") <= MEMO_MAX_BYTES
        found = _OBLIGATIONS.get(key) if memoize else None
        if found is None:
            doc = parse_contracts(raw)
            rules = [r for r in visible_contracts(doc.contracts, sid)
                     if r.enabled and r.provider in (None, provider)]
            checklist = (ChecklistDocument.model_validate_json(checklist_raw) if checklist_raw is not None
                         else ChecklistDocument(session_id=sid))
            pending = [item for item in checklist.items if item.status != "revoked"
                       and (item.status != "done" or not item.verified or not item.evidence_chunk_ids)]
            payload = {"contracts": [r.model_dump(mode="json") for r in rules],
                       "pending": [i.model_dump(mode="json") for i in pending]}
            found = {"status": "applicable" if rules or pending else "not_applicable",
                     "contracts": rules, "pending": pending, "has_checklist": bool(checklist.items),
                     "snapshot": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}
            if memoize:
                _OBLIGATIONS.put(key, found)
        # A fresh dict with fresh lists each time; the rules and items inside are shared and read-only.
        return {**found, "contracts": list(found["contracts"]), "pending": list(found["pending"])}

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
        self.sparse, self.sparse_resolution = open_sparse_index(index_dir, getattr(s, "sparse_format", "auto"))
        self.store = SessionStore(index_dir / "meta.sqlite", max_decompressed_bytes=s.max_decompressed_session_bytes,
                                  defer_derived=self.defer_derived, journal=self._journal_policy())
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
        derived: bool = True,
    ) -> dict[str, Any]:
        """Index the appended events. ``derived=False`` publishes only the canonical events and the cursor (what every
        audit reads); the sparse and dense search indexes are then left to the next ``derived`` sync or search, which
        materialize whatever they lack (``_repair_sparse_for_search``). Host hooks use it: they judge, they do not search."""
        info = self._sync_session(session_id=session_id, path=path, provider=provider, embed=embed,
                                  derived=derived, verify=_check_verify(verify))
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
                slicing=self._slice_policy(),
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
        sync = info.get("sync")
        reindex = (sync or {}).get("reindex")
        if sync is not None and sync["state"] == "partial" and reindex is not None:
            # The old index is still the one in force; the new one is being staged beside it (README, "Re-indexing").
            if reindex["state"] == "ready_to_commit":
                info["nudge"] = (
                    f"NUDGE: The re-index is staged ({reindex['rows']} events) but its commit does not fit the time this call "
                    "had left; call sync_session again (a larger timeout_ms gives the commit more room) to publish it. "
                    "Until then audits and action checks judge the previous index and cannot be clean "
                    "(coverage.issues has reindex_ready_to_commit)."
                )
            else:
                info["nudge"] = (
                    f"NUDGE: The re-index is staged through {reindex['staged_through']} of {reindex['target_bytes']} bytes "
                    f"({reindex['rows']} events); call sync_session again to continue (retry_after_ms="
                    f"{sync['retry_after_ms']} is a hint, nothing retries by itself). The previous index stays in force until "
                    "it is published, so audits and action checks cannot be clean meanwhile (coverage.issues has "
                    "sync_in_progress)."
                )
        elif sync is not None and sync["state"] == "partial":
            info["nudge"] = (
                f"NUDGE: Sync is partial ({sync['indexed_through']} of {sync['target_bytes']} bytes indexed, stopped by "
                f"{sync['stopped_by']}); call sync_session again to continue (retry_after_ms={sync['retry_after_ms']} is "
                "a hint, nothing retries by itself). Until it completes, audits and action checks cannot be clean "
                "(coverage.issues has sync_in_progress)."
            )
        return info

    @request_operation
    def record_agy_hook(self, payload: dict[str, Any], event: str) -> None:
        """Record host hook observations separately from native transcript bytes."""
        sid = payload["conversationId"].lower()
        path = agy.resolve_session_path(self.settings.resolve_agy_app_data_dirs(),
                                        session_id=sid, path=payload["transcriptPath"],
                                        require_exists=False)
        self._receipts(held=self._held_rows()).record(
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
        """Materialize deferred canonical evidence before searching its derived view.

        The missing documents are written in bounded batches, each committed on its own, so a repair that the request
        deadline interrupts keeps what it wrote and the next call continues (a repeat of a one-transaction repair would
        make no progress). The search never answers from a partial index: a repair that does not finish ends the
        request (``request_deadline``)."""
        with self._publication_guard():
            allowed = self.store.chunk_ids(session_id, provider)
            # A session search probes only its own ids; a global search needs every id, so one scan is cheaper.
            missing = allowed - (self.sparse.present(allowed) if session_id else self.sparse.ids())
        if missing:
            self._fill_sparse(session_id, provider, sorted(missing))

    def _fill_sparse(self, session_id, provider, ids, *, batch=SPARSE_REPAIR_BATCH, progress=None):
        """Write the sparse documents of the canonical events ``ids``, ``batch`` at a time. Returns how many were written."""
        written = 0
        for start in range(0, len(ids), batch):
            check_request()
            with self._publication_guard():
                snapshot = self.store.capture_evidence(session_id, provider, history=False, anchors=ids[start:start + batch])
            documents = snapshot.list_chunks(session_id, provider)
            with self._publication_guard():
                # An event that a concurrent reindex removed meanwhile is not indexed (its deletion is journalled).
                groups: dict[tuple, list[str]] = {}
                for chunk in documents:
                    groups.setdefault((chunk.provider, chunk.session_id), []).append(chunk.chunk_id)
                live: set[str] = set()
                for (chunk_provider, chunk_session), chunk_ids in groups.items():
                    live |= self.store.live_ids(chunk_session, chunk_provider, chunk_ids)
                documents = [chunk for chunk in documents if chunk.chunk_id in live]
                if documents:
                    self.sparse.upsert_many(documents)
            written += len(documents)
            if progress is not None:
                progress(start + len(ids[start:start + batch]), len(ids), written)
        return written

    @detached_operation
    def rebuild_sparse(self, *, budget_sec: float | None = None, batch: int = SPARSE_REPAIR_BATCH, progress=None) -> dict[str, Any]:
        """Materialize every canonical event the active sparse index lacks, outside a request (``python -m
        self_directing_mcp.maintenance sparse-rebuild``). Bounded (``budget_sec``), resumable and safe to interrupt: each
        batch is its own transaction and only missing events are written, so running it again continues where it stopped
        and a finished index is left unchanged. It converts a legacy index to the compact format without reading or
        touching the legacy file; the canonical store is the only source."""
        started = time.monotonic()
        deadline = started + budget_sec if budget_sec is not None else None
        resolution = dict(self.sparse_resolution)
        info: dict[str, Any] = {"ok": True, "format": self.sparse.format, "setting": resolution.get("setting"),
                                "file": str(self.sparse.db_path), "state": "complete", "stopped_by": None}
        if self.sparse.format != "compact":
            info.update(ok=False, error="sparse_format_not_compact", state="unchanged",
                        reason=resolution.get("reason") or "the active sparse format is legacy (SELF_DIRECT_SPARSE_FORMAT=legacy); there is nothing to convert")
            return info
        with self._publication_guard():
            canonical = self.store.chunk_ids(None, None)
            missing = sorted(canonical - self.sparse.ids())
        info.update(canonical=len(canonical), already_indexed=len(canonical) - len(missing), written=0, remaining=len(missing))
        for start in range(0, len(missing), batch):
            if deadline is not None and time.monotonic() >= deadline:
                info.update(state="partial", stopped_by="time_budget")
                break
            piece = missing[start:start + batch]
            try:
                info["written"] += self._fill_sparse(None, None, piece, batch=len(piece))
            except IndexBusy:
                info.update(state="partial", stopped_by="index_busy")
                break
            info["remaining"] = len(missing) - (start + len(piece))
            if progress is not None:
                progress(info["already_indexed"] + start + len(piece), len(canonical), info["written"])
        info["seconds"] = round(time.monotonic() - started, 2)
        if info["state"] == "complete":
            info["statistics"] = self.sparse.verify_statistics()
        return info

    def _refresh_evidence(self, sid, provider, path=None, verify="full", *, sync_share=None, sizing=None):
        """Refresh only authority; return detached coverage and its exact inputs.

        ``sync_share``: when the request has a deadline, the sync may use that share of the time left, so that judging
        keeps the rest. A sync that runs out of its share while the request itself is still live is handled like any
        other failed sync: the stored index is judged, ``sync_deadline`` joins the coverage issues (nothing is clean),
        and a confirmed ``must_not`` match is still a violation. A stop of the request (its deadline, a cancellation)
        and a busy index propagate as before.

        ``sizing`` (``slice_budget.Sizing``): the request will also judge what the sync publishes, so a slice is sized
        to leave the time that judging takes (see ``_sizing`` and the cost bound of ``SliceBudget``)."""
        with self._publication_guard():
            record = self.store.cursor_record(sid, provider)
        failed = {}
        request = current_request()
        limit = None
        if sync_share is not None and request is not None and request.deadline is not None:
            now = time.monotonic()
            limit = now + sync_share * max(0.0, request.deadline - now)
        try:
            with request_scope(deadline=limit) if limit is not None else nullcontext(), sizing_scope(sizing):
                info = self._sync_session(session_id=sid, provider=provider, embed=False,
                                          path=path if path is not None else record["path"] if record else None,
                                          derived=False, verify=verify)
            if info.get("ok"):
                return info
            failure, failed = info.get("error", "sync_failed"), info
        except FlightBusy as busy:
            failure = busy.reason   # another request is syncing this session: judge on what it has committed so far
            request_stats.note_degraded(failure)
        except IndexBusy:
            raise
        except RequestStopped as stopped:
            # The share's end and the request's own stop both arrive as RequestStopped; only the request's is final.
            if limit is None or stopped.reason != "request_deadline" or _request_over(request):
                raise
            failure = "sync_deadline"
            request_stats.note_degraded(failure)
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

    @staticmethod
    def _cost_key(provider, history):
        """The key of a contract set in the cost model: the provider and the ids and revisions of the rules that need history."""
        return provider + ":" + shadow_policy.contracts_digest(history)

    def _sizing(self, sid, provider, proposed, idle):
        """The cost numbers that size the sync slice of a request that will also judge it, or ``None``.

        Only a request with a deadline slices, and the idle comparison of a resident engine has its own large budget. The
        numbers are read lazily, when a slice with something big to parse is about to start, so the usual small append
        pays nothing for them. Without incremental audits (``SELF_DIRECT_AUDIT_INCREMENTAL=false``) every audit evaluates
        the whole session again whatever is synced, so there is nothing to size: the slices are those of the share alone."""
        control = current_request()
        if idle or control is None or control.deadline is None or not self.settings.audit_incremental:
            return None
        return Sizing(control.deadline, lambda: self._sizing_numbers(sid, provider, proposed))

    def _sizing_numbers(self, sid, provider, proposed):
        """``(seconds to evaluate one more event, events no monitor state covers)`` for this request, or ``None`` when no
        contract needs history (a check of a proposed action against prohibitions judges nothing the sync publishes)."""
        try:
            with self._publication_guard():
                raw, _token = self.contracts.capture()
            rules = parse_contracts(raw).contracts
        except ValueError:
            return None
        active = [rule for rule in rules if rule.enabled and rule.provider in (None, provider)
                  and rule.session_id in (None, sid)]
        if proposed is not None:
            active = [rule for rule in active if rule.type != "must" or rule.before_regex]
        history = [rule for rule in active if rule.regex and (rule.type == "must" or proposed is None)]
        if not history:
            return None
        try:
            with self._publication_guard():
                debt = self.store.audit_debt(sid, provider)
        except (RequestStopped, IndexBusy):
            raise
        except (sqlite3.Error, OSError):
            debt = 0            # a number that sizes a slice must never fail the sync it sizes
        return COSTS.eval_cost(self._cost_key(provider, history), len(history)), debt

    def _comparison_skip(self, plan, events, cost_key, scale=1.0):
        """Why the comparison that is due after the incremental pass should not start (``"deadline"``, or ``"deferred"``
        when a resident engine's idle comparison will verify the state instead), or ``None``. The best number there is
        for the cost of a fold is the speed of the pass that just ran; without one, what this process measured earlier;
        without either there is no opinion and the scope of the comparison is the only protection."""
        if plan.fold_events >= eval_cost.MIN_SAMPLE_EVENTS:
            rate = plan.fold_seconds / plan.fold_events
        else:
            rate = COSTS.eval_cost(cost_key, 1) if COSTS.observed(cost_key) else None
        return incremental.comparison_shortfall(events, rate, defer=self._shadow_worker is not None, scale=scale)

    @staticmethod
    def _comparison_scope():
        """The time share of a pass that only verifies an answer already in hand: it ends ``comparison_reserve`` before the
        request's own deadline, so a verification that does not finish leaves time to return the answer it was verifying."""
        control = current_request()
        if control is None or control.deadline is None:
            return nullcontext()
        remaining = control.deadline - time.monotonic()
        return request_scope(deadline=control.deadline - incremental.comparison_reserve(remaining))

    def _audit(self, session_id, provider, proposed=None, path=None, verify="full", escalated=None, idle=False):
        started = time.perf_counter()
        provider = self._resolve_provider(provider)
        sid = session_id.lower() if provider in ("codex", "agy") else session_id
        # The idle comparison is not a request: a refresh it cannot finish still ends it (and it backs off).
        refreshed = self._refresh_evidence(sid, provider, path, verify, sync_share=None if idle else SYNC_BUDGET_SHARE,
                                           sizing=self._sizing(sid, provider, proposed, idle))
        integrity = dict(refreshed.get("integrity") or {})
        if escalated:
            integrity["full_reason"] = escalated
        coverage = dict(refreshed["coverage"])
        coverage["issues"] = list(coverage["issues"])
        sync_finished = time.perf_counter()
        with self._publication_guard():
            raw_contracts, contract_token = self.contracts.capture()
        try:
            contracts = parse_contracts(raw_contracts).contracts   # shared and read-only: filtered below, never edited
        except ValueError:
            contracts = []
            coverage["complete"] = False
            coverage["issues"].append("contracts_corrupt")
        # The verification premise is taken over the contracts of this session before the proposed-action filter below, so
        # an audit and an action check of one session share one comparison schedule.
        premise_rules = [rule for rule in contracts if rule.enabled and rule.provider in (None, provider)
                         and rule.session_id in (None, sid)]
        planned = None
        if proposed is not None:
            action = ProposedAction.model_validate(proposed)
            # The same text builder as the parsers: a host's model-written prose field is not part of what the
            # contracts match, in history and in the proposed action alike (see ``tool_payload``).
            text, prose = tool_call_parts(provider, action.tool_name, action.arguments)
            planned = make_chunk(provider, sid, "tool_call", text,
                                 {"role": "assistant", "tool_name": action.tool_name, "proposed": True, **prose},
                                 0, coverage.get("byte_offset", 0), coverage.get("byte_offset", 0))
            planned.chunk_id = "proposed:" + planned.chunk_id
            # Preserve the public contracts_snapshot filtering/hash semantics.
            contracts = [rule for rule in contracts if rule.type != "must" or rule.before_regex]
        active = [rule for rule in contracts if rule.enabled and rule.provider in (None, provider)
                  and rule.session_id in (None, sid)]
        history_rules = [rule for rule in active if rule.regex and (rule.type == "must" or planned is None)]
        history = bool(history_rules)
        cost_key = self._cost_key(provider, history_rules) if history else None
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
        variant = "audit" if planned is None else "check"
        request_key = shadow_policy.request_key(planned)
        by_id = {rule.id: rule for rule in active}
        applies = None if planned is None else (lambda contract_id: contract_id in by_id
                                                and action_applies(by_id[contract_id], planned))
        every = shadow_policy.parse_every(self.settings.audit_shadow_every)    # None: adaptive; else the fixed legacy rate
        force, retried, late, fallback = None, False, {}, None
        passes = ExitStack()
        try:
            for attempt in range(4):    # normal pass; a retry; a pass the finished answer asked to compare; a full comparison
                passes.close()          # the reserve of the pass before it ends with that pass
                if fallback is not None:
                    passes.enter_context(self._comparison_scope())
                with self._publication_guard(), self.store._read_boundary():
                    metadata_token = self.store.state_token(sid, provider)
                    if metadata_token != refreshed["_metadata_token"] and "metadata_changed_before_capture" not in coverage["issues"]:
                        coverage["complete"] = False
                        coverage["issues"].append("metadata_changed_before_capture")
                    plan = None
                    if history:
                        # Only events some contract can inspect keep their text; rows are paged, never held as JSON.
                        needs, link_calls = audit_needs(active, planned)
                        minimum = self.settings.audit_incremental_min_events
                        if use_plan and minimum and self.store.session_metrics(sid, provider)[0] < minimum:
                            # A short session: one full evaluation costs less than keeping, rebuilding and
                            # shadow-checking monitor states. Nothing is stored; the answer is the reference one.
                            use_plan = False
                            report.update(mode="full", below_min_events=minimum)
                        if use_plan:
                            report = {key: value for key, value in report.items()
                                      if key in ("retry", "incremental_mismatch", "mismatch_not_recorded")}
                            try:
                                plan = incremental.AuditPlan.prepare(
                                    self.store, session_id=sid, provider=provider, active=active, planned=planned,
                                    needs=needs, link_calls=link_calls, shadow_every=every, force_full=force_full,
                                    report=report, premise_rules=premise_rules, force=force, idle=idle,
                                    cost_key=cost_key, defer=self._shadow_worker is not None and not idle)
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
                    if retried:
                        raise
                    retried = True
                    continue
                mismatch, scoped = [], plan is not None and plan.shadow and not plan.comparison.full
                if reference is not None:
                    # Shadow: the full evaluator over the same events (for a scoped comparison, over the compared contracts
                    # alone). It is the answer whenever the two differ.
                    subject = [rule for rule in contracts if rule.id in plan.comparison.scope] if scoped else contracts
                    # The incremental answer is in hand. A comparison that was planned (``force`` is None: not one that an
                    # answer called for, which has its own pass and its own fallback below) runs in the time that is left
                    # after it, and never in the part that gives it: it does not start when the speed of the pass that just
                    # ran says it cannot finish, and one that does not finish leaves the answer standing, unverified (the
                    # rule of a forced second pass). A stop of the request itself is not absorbed.
                    planned_pass = force is None
                    full, skipped = None, None if idle or not planned_pass else self._comparison_skip(
                        plan, len(plan.reference), cost_key, scale=len(subject) / max(1, len(contracts)))
                    if skipped is None:
                        try:
                            with self._comparison_scope() if planned_pass else nullcontext():
                                full = run_audit(session_id=sid, contracts=subject, store=reference, provider=provider,
                                                 proposed=planned, coverage=reference_coverage)
                        except RequestStopped:
                            if not planned_pass or _request_over(current_request()):
                                raise
                            skipped = "deadline"
                    if skipped is not None:
                        report["shadow_skipped"], report["shadow"] = skipped, False
                        report.pop("shadow_reasons", None)
                        report.pop("shadow_scope", None)
                        plan.shadow, plan.comparison = False, shadow_policy.Comparison()
                        reference = None
                if reference is not None:
                    timed_out = sorted({item.contract_id for item in full.findings if item.reason == REGEX_TIMEOUT})
                    if timed_out:
                        # A time limit is not a result: the full evaluator could not finish these contracts in its budget, so
                        # its (unknown) answer stands and the incremental state of those contracts is not trusted either.
                        # This is not a mismatch of the incremental path and disables nothing. (A scoped reference holds
                        # only some contracts: the incremental answer stands, unverified.)
                        if not scoped:
                            result = full
                        report["shadow_skipped"] = "reference_regex_timeout"
                        report["regex_timeout"] = timed_out
                        for contract_id in timed_out:
                            if contract_id in plan.monitors:
                                plan.monitors[contract_id].timed_out = True
                                plan.timeouts.add(contract_id)
                        report["shadow"] = False
                        reference = None
                    else:
                        mismatch = incremental.compare(result, full, only=plan.comparison.scope if scoped else None)
                    if mismatch:
                        ids = (sorted({item.contract_id for item in full.findings} & set(plan.monitors))
                               if mismatch == ["#audit"] else sorted(set(mismatch)))
                        report["incremental_mismatch"] = ids
                        flagged = {item.contract_id for item in full.findings
                                   if item.verdict in shadow_policy.FLAGGED and item.contract_id in plan.monitors
                                   and (applies is None or applies(item.contract_id))}
                        # The incremental answers so far are known to be wrong: there is nothing to fall back to. Recording
                        # the difference is not part of the verification, so it does not run in its time share.
                        fallback = None
                        passes.close()
                        self._record_mismatch(sid, provider, plan, ids, variant, request_key, idle, flagged, report)
                        if scoped:
                            # A partial reference cannot be the answer: the contracts that differ now run on the full
                            # evaluator (their health row), and a full comparison decides this request.
                            force = shadow_policy.Comparison().add("mismatch")
                            continue
                        result = full
                    if reference is not None:
                        report["shadow_compared"] = True
                if plan is not None and not mismatch:
                    # Comparisons that only the finished answer can call for: a violation (a wrong block), a violation that
                    # turned clean (a missed one), a high-risk contract whose summary changed.
                    late = plan.late_reasons(result, request_key, applies)
                    wanted = {reason: ids for reason, ids in late.items() if ids and not plan.comparison.covers(ids)}
                    if wanted and plan.schedule is not None and not idle:
                        wanted_ids = set().union(*wanted.values())
                        if shadow_policy.memo_covers(plan.schedule, plan.state_key(), request_key, result, wanted_ids):
                            report["shadow_skipped"] = "verified_before"
                            late = {}
                        elif (attempt < 3 and report.get("shadow_skipped") not in ("deadline", "deferred")
                              and incremental._shadow_affordable(plan.count)):
                            # (A planned comparison that this pass just declined for lack of room is not asked for again:
                            # a second pass would repeat the fold and have even less time for it.)
                            force = shadow_policy.Comparison()
                            for reason, ids in wanted.items():
                                force.add(reason, ids)
                            # This answer stands, unverified, if the forced pass cannot finish (see the handler below).
                            fallback = (result, copy.deepcopy(result.coverage), plan, dict(report), metadata_token,
                                        receipt_token, late)
                            continue
                        else:
                            report.setdefault("shadow_skipped", "deadline")
                break
            else:
                raise RuntimeError("audit passes exhausted")
        except (RequestStopped, IndexBusy) as stopped:
            passes.close()                  # leave the time share of the forced pass before anything else runs
            if fallback is None:
                worker = self._shadow_worker
                if (worker is not None and not idle and use_plan and history and isinstance(stopped, RequestStopped)
                        and stopped.reason == "request_deadline"):
                    # An evaluation the deadline cut saves nothing in this request, so the next one starts again. A resident
                    # engine does not have to wait for that: its idle comparison audits the session with the large budget of a
                    # background job (it never delays a request: the next request cancels it) and keeps the monitor states.
                    worker.note(provider, sid, coverage.get("path"), variant, proposed, coverage.get("byte_offset", 0),
                                "unverified")
                raise
            # The pass that was to verify the answer could not finish (its share of the time budget ran out, or the index was
            # busy): the answer of the pass before it stands, unverified. A verification that ran out of time must never turn
            # into a lost answer. A cancelled request, or one past its own deadline, has no answer to give: it raises here.
            check_request()
            result, saved_coverage, plan, report, metadata_token, receipt_token, late = fallback
            result.coverage = saved_coverage    # the pass after it may have added issues to the lists the result shared
            report["shadow_skipped"] = "busy" if isinstance(stopped, IndexBusy) else "deadline"
            mismatch = []
        finally:
            passes.close()
        ran = bool(report.get("shadow_compared"))
        schedule_after = None
        if plan is not None and plan.schedule is not None and not mismatch:
            # What this request tells the schedule: a comparison of every contract that ran and matched counts towards
            # the rate; contracts the answer flags are remembered for the next request of this kind; a violation that
            # turned clean and was not compared stays remembered (the next request meets the flip again).
            flips = late.get("flip", set())
            unverified = set() if (ran and plan.comparison.covers(flips)) else flips
            flagged = {item.contract_id for item in result.findings if item.verdict in shadow_policy.FLAGGED
                       and item.contract_id in plan.monitors and (applies is None or applies(item.contract_id))}
            # A compared input is remembered only when every contract a late reason asked for was compared: a scoped
            # comparison of one contract (a high-risk one) says nothing about a violation of another.
            late_ids = set().union(*late.values())
            memoized = ran and bool(late_ids) and plan.comparison.covers(late_ids)
            schedule_after = shadow_policy.settle(
                plan.schedule, outcome="match" if ran else None, full_scope=ran and plan.comparison.full,
                key=request_key, flagged=flagged, kept=unverified, at=datetime.now(timezone.utc).isoformat(),
                offset=plan.snapshot["offset"], state_key=plan.state_key() if memoized else None,
                result=result if memoized else None, memo_ids=late_ids if memoized else None)
            if plan.schedule_loaded is not None and schedule_after.dump() == plan.schedule_loaded:
                schedule_after = None       # nothing to write back
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
        sync = refreshed.get("sync")
        if sync is not None and sync["state"] != "complete":
            payload["sync"] = sync       # how far the index got and what sized the slice; a finished sync adds nothing to coverage
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
                    report["state"] = (self.store.save_audit_states(sid, provider, plan.digest, rows, plan.snapshot,
                                                                    shadow=schedule_after)
                                       if rows else "unchanged")
                    if not rows and schedule_after is not None:
                        self.store.save_shadow_state(sid, provider, schedule_after)
                except (sqlite3.Error, OSError) as exc:
                    report["state"] = "write_failed:" + type(exc).__name__  # a cache; the next audit rebuilds it
            reason = _fast_gate(payload, integrity) if integrity.get("mode") == "fast" else None
            if reason is None:
                worker = self._shadow_worker
                if worker is not None and plan is not None and not mismatch:
                    matched = ran and plan.comparison.full
                    if matched or not idle:      # the idle comparison itself is not a request that went unverified
                        worker.note(provider, sid, plan.snapshot.get("path"), variant, proposed,
                                    plan.snapshot["offset"], "match" if matched else "unverified")
                with publication():
                    return payload
        # Leave the guard before the long work: the fast verdict rested on evidence only a full scan can vouch for.
        return self._audit(session_id, provider, proposed=proposed, path=path, verify="full", escalated=reason, idle=idle)

    def _record_mismatch(self, sid, provider, plan, ids, variant, request_key, idle, flagged=frozenset(), report=None):
        """A comparison found a difference: disable the incremental path of the contracts that differ, reset the schedule
        to 1/4 and record the range of earlier requests that were answered without a comparison (a range for review,
        never a claim about what those answers were). All of it is one transaction, so a contract is never left disabled
        without the reset and the range that go with it; if it cannot be written the answer still stands (the full
        evaluator's) and the report says so."""
        offset = plan.snapshot["offset"]
        worker = self._shadow_worker
        schedule = plan.schedule
        reasons = plan.comparison.reasons
        source = "idle" if idle else "sampled" if reasons == ["sampled"] else "forced:" + ",".join(
            reason for reason in reasons if reason != "sampled")
        review, after = None, None
        if schedule is not None:
            after = shadow_policy.settle(schedule, outcome="mismatch", full_scope=True, key=request_key,
                                         flagged=set(flagged), kept=set(), at=datetime.now(timezone.utc).isoformat(),
                                         offset=offset)
            since = worker.since_match(provider, sid) if worker is not None else None
            # A resident engine counts the requests it answered since the last match, if it was running at that time; for what
            # came before (a restart, a hook process) only the persisted cursor of that match is left.
            counted = worker is not None and worker.ledger_covers(schedule.last_match_at)
            if since is not None or (not counted and schedule.last_match_offset != offset):
                since_at = (since["since_at"] if counted and since else None) or schedule.last_match_at
                detail = "answers given from incremental state since the last matching comparison"
                if not counted:
                    detail += "; requests answered before this process started are not in the count"
                review = {"source": source, "since_at": since_at or (since or {}).get("since_at"),
                          "since_offset": schedule.last_match_offset, "until_offset": offset,
                          "requests": since["requests"] if since else None, "contracts": ids, "detail": detail}
        try:
            with self._publication_guard():
                self.store.record_incremental_mismatch(
                    sid, provider, incremental.EVALUATOR_VERSION, [*ids, "#facts"],
                    f"incremental result differed from the full evaluator at byte {offset}", after, review)
        except (RequestStopped, IndexBusy):
            raise
        except (sqlite3.Error, OSError) as exc:
            if report is not None:
                report["mismatch_not_recorded"] = type(exc).__name__
            return
        if worker is not None:
            worker.note(provider, sid, plan.snapshot.get("path"), variant, None, offset, "mismatch")

    def _receipt_token(self, sid, provider):
        if provider != "agy":
            return None
        return self._receipts().authority_token(sid)

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

    @background_operation
    def verify_incremental(self, session_id: str, provider: str | None = None, path: str | None = None,
                           action: dict[str, Any] | None = None, variant: str = "audit") -> dict[str, Any]:
        """The idle comparison of a resident engine (``shadow_worker``): audit the latest state of a session and compare
        it with a full evaluation, whatever the schedule says. The answer is discarded; only the outcome is returned.
        It is not a request of the host's and it never certifies answers already given."""
        payload = self._audit(session_id, provider, proposed=action if variant == "check" else None, path=path,
                              verify="fast", idle=True)
        report = payload.get("incremental") or {}
        return {"ok": True, "compared": bool(report.get("shadow_compared")),
                "mismatch": list(report.get("incremental_mismatch") or ()), "skipped": report.get("shadow_skipped"),
                "mode": report.get("mode"), "not_recorded": report.get("mismatch_not_recorded")}

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
        document = ContractsDocument.model_validate_json(raw) if raw is not None else ContractsDocument()
        stored = document.contracts
        selected = visible_contracts(stored, session_id)
        result = {
            "ok": True,
            "session_id": session_id,
            "scope": contract_scope(session_id),
            "global_meaning": GLOBAL_CONTRACT_MEANING,
            "contracts": [c.model_dump(mode="json") for c in selected],
            "counts": scope_counts(selected, len(stored)),
        }
        shown = [change for change in document.pending.values()
                 if change.contract.session_id is None or change.contract.session_id == session_id]
        if shown:
            result["pending"] = [change.model_dump(mode="json") for change in shown]
            result["counts"]["pending"] = len(shown)
        return result

    def _anchor_status(self, event_ids) -> dict[str, tuple[str, str]]:
        """``{event id: (status, detail)}``: ``user`` (a user message), ``other`` or ``unresolved`` (not indexed here).

        An anchor is the event a contract was derived from, so it must be the user's own message: a tool result,
        runtime text or injected context cannot grant or revoke authority, whatever role the host rendered it with.
        The parsers mark a user turn with ``meta.role == "user"`` (Claude Code: ``origin.kind == "human"``; AGY:
        ``USER_INPUT`` from ``USER_EXPLICIT``; OMP: a user-role message not attributed to an agent). Codex records the
        typed prompt and the model-facing copy (which may also carry injected context) both as user-role turns and
        cannot tell them apart.
        """
        wanted = tuple(dict.fromkeys(event_id for event_id in event_ids if event_id))
        found: dict[str, tuple[str, str]] = {}
        if not wanted:
            return found
        with self._publication_guard(), self.store._read_boundary():
            for event_id in wanted:
                check_request()
                chunk = self.store.get_chunk(event_id)
                if chunk is None:
                    found[event_id] = ("unresolved", "not in the index")
                elif chunk.meta.get("role") == "user":
                    found[event_id] = ("user", "user message")
                else:
                    found[event_id] = ("other", f"a {chunk.kind} event with role {chunk.meta.get('role')!r}")
        return found

    @staticmethod
    def _reject_non_user_anchors(label, anchors, event_id) -> None:
        status, detail = anchors.get(event_id, ("unresolved", ""))
        if status == "other":
            raise ValueError(f"{label}: source_event_id {event_id!r} is {detail}, not a user message. A contract anchor "
                             "must be the user's own message; a tool result, runtime or injected text cannot be one.")

    @detached_operation
    def revoke_contract(self, contract_id: str, source_event_id: str | None = None) -> dict[str, Any]:
        """Disable a rule while keeping its history and source quote for audit.

        Disabling weakens enforcement, so it needs ``source_event_id``: a user message that asked for it. Without a
        user anchor the request is parked as a pending change and the contract stays in force.
        """
        assert self.contracts
        anchors = self._anchor_status([source_event_id])
        self._reject_non_user_anchors(f"revoke {contract_id!r}", anchors, source_event_id)
        status = anchors.get(source_event_id, ("missing", ""))[0] if source_event_id else "missing"
        with self._publication_guard():
            raw, token = self.contracts.capture()
            document = ContractsDocument.model_validate_json(raw) if raw is not None else ContractsDocument()
            current = next((rule for rule in document.contracts if rule.id == contract_id), None)
            if current is None:
                return {"ok": False, "error": "not_found", "contract_id": contract_id}
            if current.enabled and status != "user":
                document.pending[contract_id] = PendingChange(
                    contract=current.model_copy(update={"enabled": False}, deep=True), reasons=["disabled"],
                    source_event_id=source_event_id, anchor_status="unresolved" if source_event_id else "missing",
                    base_revision=current.revision)
                self.contracts.publish_document(document, document.model_dump_json(indent=2), token)
                return {"ok": True, "status": "pending", "contract": current.model_dump(mode="json"),
                        "reasons": ["disabled"], "nudge": PENDING_NUDGE}
            disabled = self.contracts.prepare_revoke(document, contract_id)
            document.pending.pop(contract_id, None)
            payload = document.model_dump_json(indent=2)
            self.contracts.publish_document(document, payload, token)
        return {"ok": True, "status": "disabled", "contract": disabled.model_dump(mode="json"),
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
        # An anchor that resolves to something other than a user message is rejected; one that cannot be resolved
        # (the session is not indexed yet) is accepted and left to the audit (``anchor_not_user`` there).
        anchors = self._anchor_status([rule.source_event_id for rule in rules])
        for rule in rules:
            self._reject_non_user_anchors(f"contract {rule.id!r}", anchors, rule.source_event_id)
        with self._publication_guard():
            raw, token = self.contracts.capture()
            document = ContractsDocument.model_validate_json(raw) if raw is not None else ContractsDocument()
            active = {rule.id: rule for rule in document.contracts}
            apply, held = [], []
            for rule in rules:
                previous = active.get(rule.id)
                reasons = loosening_reasons(previous, rule) if previous is not None else []
                if reasons and anchors.get(rule.source_event_id, ("", ""))[0] != "user":
                    held.append((rule, reasons, previous))     # weakening needs a user anchor: not applied
                    continue
                apply.append(rule)
                active[rule.id] = rule
            updated = self.contracts.prepare_upsert(document, apply)
            for rule in apply:
                document.pending.pop(rule.id, None)
            for rule, reasons, previous in held:
                document.pending[rule.id] = PendingChange(
                    contract=rule, reasons=reasons, source_event_id=rule.source_event_id,
                    anchor_status="unresolved" if rule.source_event_id else "missing", base_revision=previous.revision)
            payload = document.model_dump_json(indent=2)
            self.contracts.publish_document(document, payload, token)
        try:
            # Informational report of equivalent spellings the regexes miss. Imported here and nowhere else, so
            # hook paths never load the variant library; it runs after the save and can only add this section.
            from self_directing_mcp.audit.variants import fragility_report
            fragility = fragility_report(updated)
        except Exception as exc:
            fragility = {"informational": True, "status": "unavailable", "error": type(exc).__name__}
        result = {
            "ok": True,
            "returned": "upserted",
            "global_meaning": GLOBAL_CONTRACT_MEANING,
            "contracts": [c.model_dump(mode="json") for c in updated],
            "counts": {
                "upserted": len(updated),
                "store": len(document.contracts),
            },
            "fragility": fragility,
        }
        checks = [{"contract_id": rule.id, "source_event_id": rule.source_event_id,
                   "anchor": anchors[rule.source_event_id][0]} for rule in rules if rule.source_event_id]
        if checks:
            result["anchor_checks"] = checks
        if held:
            result["pending"] = [{"contract_id": rule.id, "reasons": reasons, "active_revision": previous.revision,
                                  "anchor": "unresolved" if rule.source_event_id else "missing"}
                                 for rule, reasons, previous in held]
            result["counts"]["pending"] = len(held)
            result["nudge"] = PENDING_NUDGE
        return result

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
        if not any(item.status == RequirementStatus.DONE for item in doc.items):
            # The evidence below can only ever demote a ``done`` item (to pending_verification); no other field of the
            # reply depends on it. A checklist without a ``done`` item therefore needs neither the scan of every
            # stored event (parsing each one) nor a second hash of the whole source, and the reply is the same.
            return {"ok": True, "session_id": sid, "items": [i.model_dump(mode="json") for i in doc.items]}
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
        """Validate a node/edge proposal against indexed evidence; nothing stored here.

        ``covers_through_order`` in the result is the event order a ``commit_graph_update`` WITHOUT ``through_order``
        would move the cursor to: the end of the whole session, even if the proposal was built from fewer events.
        """
        assert self.store and self.graph
        resolved_provider = self._resolve_provider(provider)
        sid = session_id.lower() if resolved_provider in ("codex", "agy") else session_id
        anchors = {cid for edge in proposal.get("edges", []) for cid in edge.get("evidence_chunk_ids", [])}
        with self._publication_guard(), self.store._read_boundary():
            snapshot = self.store.capture_evidence(sid, resolved_provider, history=False, anchors=anchors)
            metadata_token = self.store.state_token(sid, resolved_provider)
            cursor = self.graph.cursor(sid, resolved_provider)
        verdict = validate_proposal(proposal, store=snapshot, session_id=sid, provider=resolved_provider)
        covers = graph_batch.covers_through_order(snapshot.count, cursor)
        with self._publication_guard():
            if self.store.state_token(sid, resolved_provider) != metadata_token:
                return {"ok": False, "status": "evidence_conflict", "cursor": cursor, "covers_through_order": covers}
        return {"ok": verdict["ok"], **verdict, "cursor": cursor, "covers_through_order": covers,
                "nudge": None if verdict["ok"] else "Fix problems and resubmit; nothing was stored."}

    @detached_operation
    @graph_operation("commit_graph_proposal")
    def commit_graph_update(self, session_id: str, proposal: dict[str, Any], base_graph_version: int,
                            provider: str | None = None, through_order: int | None = None) -> dict[str, Any]:
        """Validate detached evidence, then CAS the graph and metadata generations.

        WITHOUT ``through_order`` the cursor moves to the END OF THE SESSION (``covers_through_order`` in the result),
        even when the proposal was built from fewer events: every event after the ones the proposal covers is then
        marked processed without ever being extracted. ``through_order`` is the order of the last event the proposal
        covers; it must lie between the first event not yet processed and the last event of the session, and the
        cursor then moves only that far (``invalid_through_order`` otherwise; nothing is stored).

        A proposal without a ``job_id`` is named by a digest of provider, session, the event order the commit moves
        the cursor to and the proposal, so repeating the identical call is ``already_applied`` and any other commit
        applies, one after the session has grown included. A ``job_id`` that another session or provider recorded is
        ``job_id_in_use``.
        """
        resolved_provider = self._resolve_provider(provider)
        sid = session_id.lower() if resolved_provider in ("codex", "agy") else session_id
        anchors = {cid for edge in proposal.get("edges", []) for cid in edge.get("evidence_chunk_ids", [])}
        with self._publication_guard(), self.store._read_boundary():
            cursor = self.graph.cursor(sid, resolved_provider)
            count = self.store.chunk_count(sid, resolved_provider)
            covers = graph_batch.covers_through_order(count, cursor)
            through = covers if through_order is None else through_order        # where this commit moves the cursor
            job_id = proposal.get("job_id") or graph_batch.adhoc_job_id(resolved_provider, sid, through, proposal)
            held, existing = self._job_state(job_id, sid, resolved_provider)
            if held == "applied":
                return {"ok": True, "status": "already_applied", "job": existing, "covers_through_order": covers}
            if held == "foreign":
                return self._job_id_in_use(covers)
            metadata_token = self.store.state_token(sid, resolved_provider)
            snapshot = self.store.capture_evidence(sid, resolved_provider, history=False, anchors=anchors)
        if base_graph_version != cursor["graph_version"]:
            return {"ok": False, "status": "version_conflict",
                    "expected_base_graph_version": cursor["graph_version"],
                    "provided_base_graph_version": base_graph_version, "covers_through_order": covers,
                    "nudge": "Re-read the current graph and rebuild the proposal against it."}
        if through_order is not None:
            problem = graph_batch.through_order_problem(through_order, cursor, count)
            if problem:
                return {"ok": False, "status": "invalid_through_order", "problem": problem,
                        "first_unprocessed_order": graph_batch.cursor_start(cursor), "last_order": count - 1,
                        "covers_through_order": covers,
                        "nudge": "Nothing was stored. Pass the order of the last event the proposal covers, between "
                                 "first_unprocessed_order and last_order, or omit it to cover the whole session."}
        verdict = validate_proposal(proposal, store=snapshot, session_id=sid, provider=resolved_provider)
        if not verdict["ok"]:
            return {"ok": False, "status": "rejected", **verdict, "covers_through_order": covers}
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        job = {"job_id": job_id, "provider": resolved_provider,
               "session_id": sid, "range_start": cursor["processed_through_order"],
               "range_end": through, "input_digest": str(hash(json.dumps(proposal, sort_keys=True))),
               "base_graph_version": base_graph_version, "status": "pending",
               "created_at": now, "updated_at": now}
        with self._publication_guard():
            held, existing = self._job_state(job_id, sid, resolved_provider)
            if held == "applied":
                return {"ok": True, "status": "already_applied", "job": existing, "covers_through_order": covers}
            if held == "foreign":
                return self._job_id_in_use(covers)
            current_version = self.graph.cursor(sid, resolved_provider)["graph_version"]
            if current_version != base_graph_version:
                return {"ok": False, "status": "version_conflict",
                        "expected_base_graph_version": current_version,
                        "provided_base_graph_version": base_graph_version, "covers_through_order": covers,
                        "nudge": "Re-read the current graph and rebuild the proposal against it."}
            if self.store.state_token(sid, resolved_provider) != metadata_token:
                return {"ok": False, "status": "evidence_conflict", "covers_through_order": covers}
            result = self.graph.apply_proposal({**proposal, "job_id": job_id}, session_id=sid,
                                               provider=resolved_provider, base_version=base_graph_version,
                                               through_order=through)
            if result["status"] in ("applied", "unclear"):      # a refused write leaves the job table as it was
                job["status"], job["detail"] = result["status"], result.get("detail")
                self.graph.record_job(job)
            return {**result, "covers_through_order": covers}

    @staticmethod
    def _job_id_in_use(covers: int) -> dict[str, Any]:
        return {"ok": False, "status": "job_id_in_use", "covers_through_order": covers,
                "nudge": "This job_id is recorded for another session or provider. Nothing was stored; use a job_id "
                         "of your own, or none."}

    def _read_graph_batch(self, sid: str, provider: str, after_order: int | None = None, *,
                          max_chars: int = graph_batch.BATCH_MAX_CHARS,
                          max_events: int = graph_batch.BATCH_MAX_EVENTS) -> graph_batch.BatchRead:
        """The graph cursor, the event count and the next batch of raw frames, read at one point of the index.

        The batch starts at the first event not yet processed, or right after ``after_order``, and ends at
        ``max_events`` events or ``max_chars`` of ``batch_event_cost`` (``run_neograph_update`` always uses the
        maximum a batch can have). Only the rows of the batch are read and parsed; what precedes them is stepped over
        in the index (``event_frames_range``), so a batch costs what it holds, not what the session holds. Whether
        ``after_order`` is acceptable is the caller's to judge.
        """
        with self._publication_guard(), self.store._read_boundary():
            cursor = self.graph.cursor(sid, provider)
            first = graph_batch.cursor_start(cursor)
            total = self.store.chunk_count(sid, provider)
            start = first if after_order is None else after_order + 1
            frames, blocked = [], None
            if first <= start < total:
                frames, blocked = self.store.event_frames_range(
                    sid, provider, start, max_events, max_chars, cost=graph_batch.batch_event_cost)
        return graph_batch.BatchRead(cursor, first, total, start, tuple(frames), blocked)

    @detached_operation
    def _prepare_neograph_update(self, session_id, provider=None):
        resolved = self._resolve_provider(provider)
        sid = session_id.lower() if resolved in ("codex", "agy") else session_id
        return sid, resolved, self._read_graph_batch(sid, resolved)

    def _publish_graph_proposal(self, sid, provider, proposal, base_version, through, metadata_token, **job):
        """The second half of every validated graph write, under the lease: the graph version and the evidence state
        must still be those the proposal was validated against (``version_conflict`` / ``evidence_conflict``), then
        nodes, edges and cursor (and, with ``job_*`` arguments, the job record) are applied in one transaction. A
        proposal that names a ``job_id`` already applied for this session reports ``already_applied`` instead, and
        one that another session or provider recorded is ``job_id_in_use``."""
        with self._publication_guard():
            job_id = proposal.get("job_id")
            held = self._job_state(job_id, sid, provider)[0] if job_id else None
            if held == "applied":
                return {"ok": True, "status": "already_applied"}
            if held == "foreign":
                return {"ok": False, "status": "job_id_in_use"}
            if self.graph.cursor(sid, provider)["graph_version"] != base_version:
                return {"ok": False, "status": "version_conflict"}
            if self.store.state_token(sid, provider) != metadata_token:
                return {"ok": False, "status": "evidence_conflict"}
            return self.graph.apply_proposal(proposal, session_id=sid, provider=provider,
                                             base_version=base_version, through_order=through, **job)

    def _job_state(self, job_id, sid, provider):
        """``(state, job)`` for a job id. An id is global but its row belongs to one session: the state is ``"applied"``
        when this session's job was applied, ``"foreign"`` when the id is recorded for another session or provider
        (never to be treated as a replay or written over), else None."""
        job = self.graph.find_job(job_id)
        if job is None:
            return None, None
        if job["session_id"] != sid or job["provider"] != provider:
            return "foreign", job
        return ("applied" if job["status"] == "applied" else None), job

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
        return self._publish_graph_proposal(sid, provider, proposal, base_version, through, metadata_token)

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
                    "message": "LLM graph updates are unavailable in local-only mode. graph_next_batch and "
                               "graph_submit_batch (extraction by the calling agent, no model call here) work."}
        if self.settings.graph_extractor == "host":
            return {"ok": False, "error": "extractor_disabled",
                    "message": "SELF_DIRECT_GRAPH_EXTRACTOR=host: this server makes no model call for graph "
                               "updates. Fetch the next batch with graph_next_batch and commit the calling agent's "
                               "extraction with graph_submit_batch."}
        sid, resolved_provider, read = self._prepare_neograph_update(
            session_id, provider, _deadline=_deadline, cancel_event=cancel_event)
        cursor = read.cursor
        api_key = self.settings.resolve_api_key() or ""
        if not api_key:
            return {"ok": False, "error": "no_api_key",
                    "message": "Set OPENROUTER_API_KEY or SELF_DIRECT_OPENROUTER_API_KEY_FILE to run the LLM-driven update."}
        if self.neograph is None:
            return {"ok": False, "error": "neograph_missing",
                    "message": "pip install neograph-engine first."}
        if not read.total:
            return {"ok": False, "error": "no_events",
                    "message": "Nothing to extract; sync_session first."}
        if read.start >= read.total or (not read.frames and read.blocked is None):
            return {"ok": True, "status": "up_to_date", "graph_cursor_after": cursor}
        pending = read.total - read.start
        batch = [mask_value(frame) for frame in read.frames]
        if not batch:
            return {"ok": False, "error": "graph_event_too_large",
                    "chunk_id": read.blocked["chunk_id"], "graph_cursor_after": cursor,
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
        pipeline["remaining_events"] = pending - (len(batch) if applied else 0)
        return pipeline

    @detached_operation
    def graph_next_batch(self, session_id: str, provider: str | None = None, after_order: int | None = None,
                         max_chars: int | None = None, max_events: int | None = None) -> dict[str, Any]:
        """The next batch of session events for the calling agent to extract (``SELF_DIRECT_GRAPH_EXTRACTOR=host``
        or not; no model call, no key, no network). The events are masked as ``run_neograph_update`` masks them; the
        result carries the extractor prompt, the proposal schema and a ``batch_id`` for ``graph_submit_batch``.
        ``after_order`` starts the batch right after that event order (a lookahead, to extract several batches in
        parallel and commit them in order); it may not lie behind the cursor.

        The batch ends at ``max_events`` events (default 100) or ``max_chars`` of event size (default 24,000), never
        above the 100 events and 95,000 characters ``run_neograph_update`` cuts at: a full batch is more than many
        hosts let one tool result carry, and a host that shortens the result must not leave the agent believing it
        read a whole batch. A complete result ends with ``"end_of_batch": true``."""
        assert self.store and self.graph
        resolved_provider = self._resolve_provider(provider)
        sid = session_id.lower() if resolved_provider in ("codex", "agy") else session_id
        if after_order is not None and (isinstance(after_order, bool) or not isinstance(after_order, int)):
            return {"ok": False, "error": "invalid_after_order", "message": "after_order must be an integer."}
        chars, events_limit, problem = graph_batch.batch_limits(max_chars, max_events)
        if problem:
            return {"ok": False, "error": "invalid_batch_limit", "message": problem}
        read = self._read_graph_batch(sid, resolved_provider, after_order, max_chars=chars, max_events=events_limit)
        if not read.total:
            return {"ok": False, "error": "no_events", "message": "Nothing to extract; sync_session first."}
        if read.start < read.cursor_start:
            return {"ok": False, "error": "invalid_after_order", "cursor": read.cursor,
                    "minimum_after_order": read.cursor_start - 1,
                    "message": "after_order lies behind the cursor: those events are already processed."}
        whole = {"session_id": sid, "provider": resolved_provider, "cursor": read.cursor, "total_events": read.total}
        if read.start >= read.total or (not read.frames and read.blocked is None):
            return {"ok": True, "status": "up_to_date", **whole, "events": [], "event_count": 0, "remaining_events": 0}
        if not read.frames:
            needed = graph_batch.batch_event_cost(read.blocked)
            if chars < needed <= graph_batch.BATCH_MAX_CHARS:     # this call's own limit was the obstacle
                return {"ok": False, "error": "graph_event_too_large", "chunk_id": read.blocked["chunk_id"],
                        "order": read.start, **whole, "max_chars": chars, "needed_max_chars": needed,
                        "message": f"The next complete event needs max_chars of at least {needed} (this call allowed "
                                   f"{chars}); call again with that. No text was truncated."}
            return {"ok": False, "error": "graph_event_too_large", "chunk_id": read.blocked["chunk_id"],
                    "order": read.start, **whole,
                    "message": "The next complete event exceeds the graph input budget; no text was truncated."}
        events = [mask_value(frame) for frame in read.frames]
        end = events[-1]["order"]
        # The notice and the instructions come before the events: the reader has them before the untrusted text.
        # end_of_batch comes last: a result a host cut short visibly lacks it.
        return {"ok": True, "status": "batch", **whole, "truncation_hint": graph_batch.TRUNCATION_HINT,
                "untrusted_data_notice": graph_batch.UNTRUSTED_DATA_NOTICE,
                "batch_id": graph_batch.make_batch_id(resolved_provider, sid, events),
                "range": {"start_order": read.start, "end_order": end},
                "event_count": len(events), "remaining_events": read.total - (end + 1),
                "max_chars": chars, "max_events": events_limit,
                "extraction": graph_batch.extraction_block(), "events": events, "end_of_batch": True}

    def _graph_progress(self, sid: str, provider: str, after: dict[str, int] | None = None) -> dict[str, int]:
        """The graph's version, the last processed event order and the events still waiting after it. By default the
        stored cursor; ``after`` is the cursor a commit of this very call produced, which stays what that call did
        when other batches are committed meanwhile."""
        with self._publication_guard(), self.store._read_boundary():
            cursor = after or self.graph.cursor(sid, provider)
            total = self.store.chunk_count(sid, provider)
        return {"graph_version": cursor["graph_version"], "processed_through_order": cursor["processed_through_order"],
                "remaining_events": max(0, total - graph_batch.cursor_start(cursor))}

    @detached_operation
    @graph_operation("submit_graph_batch")
    def graph_submit_batch(self, session_id: str, batch_id: str, proposal: dict[str, Any] | None = None,
                           no_graph_facts_reason: str | None = None, extractor: str | None = None,
                           provider: str | None = None) -> dict[str, Any]:
        """Validate and commit the calling agent's extraction of one batch (see ``graph_next_batch``).

        Nothing is stored unless the batch is current: ``batch_id`` is recomputed from the index (``batch_changed``),
        the batch must start at the first unprocessed event (``out_of_order``: submit the earlier batches first, then
        this one again; ``stale_batch`` / ``already_applied`` when the cursor is past it), and the proposal must
        match the published schema and its caps (at most 100 evidence ids per edge, single-line ids and labels),
        cite only events of this batch and pass the evidence rules (``rejected``). A proposal without edges is
        rejected (``empty_graph_requires_review``, cursor retained) unless ``no_graph_facts_reason`` records why the
        batch holds none: then the cursor moves past the batch (``skipped_no_graph_facts``). The commit uses the same
        version and evidence checks as every graph write; the job id is the batch id, so a repeated submit reports
        ``already_applied``, and an id another session or provider recorded is ``job_id_in_use``."""
        assert self.store and self.graph
        resolved_provider = self._resolve_provider(provider)
        sid = session_id.lower() if resolved_provider in ("codex", "agy") else session_id
        parsed = graph_batch.parse_batch_id(batch_id)
        if parsed is None:
            return {"ok": False, "status": "invalid_batch_id",
                    "nudge": "Pass the batch_id exactly as graph_next_batch returned it."}
        start, end, digest = parsed
        label, reason, text_problems = graph_batch.submission_text(extractor, no_graph_facts_reason)
        shape_ok = proposal is not None and not graph_batch.proposal_problems(proposal)
        anchors = {cid for edge in proposal["edges"] for cid in edge["evidence_chunk_ids"]} if shape_ok else set()
        with self._publication_guard(), self.store._read_boundary():
            held = self._job_state(batch_id, sid, resolved_provider)[0]
            if held != "applied":
                cursor = self.graph.cursor(sid, resolved_provider)
                # Exactly the range the id names, within the size a batch can have (which bounds the work done here).
                frames, _blocked = self.store.event_frames_range(
                    sid, resolved_provider, start, end - start + 1, graph_batch.BATCH_MAX_CHARS,
                    cost=graph_batch.batch_event_cost)
                metadata_token = self.store.state_token(sid, resolved_provider)
                snapshot = self.store.capture_evidence(sid, resolved_provider, history=False, anchors=anchors)
        if held == "applied":
            return {"ok": True, "status": "already_applied", **self._graph_progress(sid, resolved_provider)}
        events = [mask_value(frame) for frame in frames]
        full_digest = (graph_batch.batch_digest(resolved_provider, sid, start, end, events)
                       if len(events) == end - start + 1 else None)
        if full_digest is None or full_digest[:24] != digest:
            return {"ok": False, "status": "batch_changed",
                    "nudge": "The events of this range are not what this batch_id was made from. Nothing was "
                             "stored; call graph_next_batch again."}
        if held == "foreign":           # the id is this session's batch, but another session's job row holds the name
            return self._batch_id_in_use()
        first = graph_batch.cursor_start(cursor)
        if start > first:
            return {"ok": False, "status": "out_of_order", "expected_start_order": first,
                    "batch_start_order": start,
                    "nudge": "Earlier events are not processed yet. Submit the batches before this one, from "
                             "expected_start_order on, then submit this one again."}
        if start < first:
            return {"ok": False, "status": "stale_batch", "expected_start_order": first,
                    "nudge": "The cursor is already past this batch, which was not applied. Call graph_next_batch."}
        kind, problems = graph_batch.classify_submission(proposal, reason, text_problems,
                                                         {event["chunk_id"] for event in events})
        if kind == "apply":
            problems = validate_proposal(proposal, store=snapshot, session_id=sid, provider=resolved_provider)["problems"]
        if kind == "empty":
            return {"ok": False, "status": "rejected", "problems": ["empty_graph_requires_review"],
                    "extraction_summary": proposal.get("extraction_summary") if proposal is not None else None,
                    "message": "No relations given; cursor retained. Review the batch before retrying: submit the "
                               "relations it states, or record why it states none with no_graph_facts_reason."}
        if problems:
            return {"ok": False, "status": "rejected", "problems": problems,
                    "nudge": "Fix the problems and resubmit; nothing was stored and the cursor is unchanged."}
        skipped = kind == "skip"
        detail = {"source": "graph_submit_batch", "extractor": label,
                  "outcome": "skipped_no_graph_facts" if skipped else "applied"}
        if skipped:
            detail["reason"] = reason
            stored = {"nodes": [], "edges": []}                      # a skip records the batch; it stores no graph facts
        else:
            detail.update(nodes=len(proposal["nodes"]), edges=len(proposal["edges"]))
            stored = {"nodes": proposal["nodes"], "edges": proposal["edges"]}
        result = self._publish_graph_proposal(
            sid, resolved_provider, {**stored, "job_id": batch_id}, cursor["graph_version"], end, metadata_token,
            job_start=start, job_digest=full_digest, job_detail=json.dumps(detail, sort_keys=True))
        if result.get("status") == "applied":
            ignored = len(proposal["nodes"]) if skipped and proposal is not None else 0
            made = {"graph_version": result["graph_version"], "processed_through_order": result["processed_through_order"]}
            return {"ok": True, "status": "skipped_no_graph_facts" if skipped else "applied",
                    **({"nodes_ignored": ignored} if ignored else {}),
                    **self._graph_progress(sid, resolved_provider, after=made)}
        if result.get("status") == "already_applied":
            return {"ok": True, "status": "already_applied", **self._graph_progress(sid, resolved_provider)}
        if result.get("status") == "job_id_in_use":
            return self._batch_id_in_use()
        return {**result, "nudge": "Nothing was applied. Call graph_next_batch again and extract from its result."}

    @staticmethod
    def _batch_id_in_use() -> dict[str, Any]:
        return {"ok": False, "status": "job_id_in_use",
                "nudge": "Another session or provider already recorded a job under this batch id, so this batch cannot "
                         "be recorded and nothing was stored. Record it with propose_graph_update and "
                         "commit_graph_update(through_order=...)."}

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
        status["incremental_verification"] = self._verification_status(sid, resolved)
        status["evaluation"] = self._evaluation_status(sid, resolved)
        return status

    def _evaluation_status(self, sid, provider) -> dict[str, Any]:
        """What a deadline-bound audit sizes its sync by, and how far the monitor states of a session lag behind its events
        (``events_without_state``: what the next incremental audit must fold; a state belongs to one contract set, so this
        is the lag of the newest). A session whose lag is more than one request can fold is brought up to date by an audit
        with a larger ``timeout_ms`` (up to ``max_tool_timeout_sec``), see the README, "Large sessions and deadlines"."""
        info: dict[str, Any] = {"cost": COSTS.snapshot(), "session": None}
        if sid:
            with self._publication_guard():
                total, covered, through = self.store.audit_progress(sid, provider)
            info["session"] = {"events_indexed": total, "events_without_state": max(0, total - covered),
                               "states_through_byte": through}
        return info

    def _verification_status(self, sid, provider) -> dict[str, Any]:
        """How incremental audits are compared with the full evaluator, and what a session's record says (see README,
        'Incremental audits'): the setting, the idle comparison, and for a session its schedule, the contracts whose
        incremental path is switched off and the ranges of requests that need review. None of it is a verdict."""
        every = shadow_policy.parse_every(self.settings.audit_shadow_every)
        worker = self._shadow_worker
        info: dict[str, Any] = {
            "policy": "adaptive" if every is None else f"fixed (SELF_DIRECT_AUDIT_SHADOW_EVERY={every})",
            "forced_after_premise_change": shadow_policy.WARMUP_REQUESTS,
            "idle_comparison": worker.snapshot() if worker is not None else None,
            "guarantee": "a sample, not a proof: answers between comparisons were not compared; needs_review lists "
                         "ranges that a later comparison put in doubt"}
        if sid:
            with self._publication_guard():
                report = self.store.shadow_report(sid, provider, incremental.EVALUATOR_VERSION)
            state = report["state"]
            info["session"] = None if state is None else {
                "rate": shadow_policy.rate_text(state, every), "streak": state.streak, "forced_comparisons_left": state.warmup,
                "comparisons": state.compared, "last_match_at": state.last_match_at,
                "last_match_offset": state.last_match_offset, "reset_reason": state.reset_reason}
            info["disabled_contracts"] = report["disabled"]
            info["needs_review"] = report["needs_review"]
        return info

    def _sparse_status(self) -> dict[str, Any]:
        """``audit_status`` fields of the sparse index: its format and how the setting was resolved, both files with their
        sizes (so the owner sees what can be deleted once the compact index is complete) and the document count of each
        (the one that is not in use is read through a read-only connection and is never modified)."""
        index_dir = Path(self.settings.index_dir)
        active = self.sparse.format
        files, counts = [], {}
        for fmt in ("legacy", "compact"):
            path = sparse_file(index_dir, fmt)
            size = sum(p.stat().st_size for p in (path, Path(f"{path}-journal"), Path(f"{path}-wal")) if p.exists())
            exists = path.exists()
            files.append({"name": path.name, "format": fmt, "active": fmt == active, "exists": exists,
                          "bytes": size if exists else None})
            counts[fmt] = self.sparse.count() if fmt == active else (count_sparse_file(path) if exists else None)
        return {"sparse_format": active, "sparse_format_setting": self.sparse_resolution.get("setting"),
                "sparse_format_reason": self.sparse_resolution.get("reason"), "sparse_files": files, "sparse_counts": counts}

    def _journal_status(self):
        """``(journal, journal_fallback)`` for ``audit_status``: the journal mode of each authority database, and the
        databases that use the rollback journal because their file system is denied or unknown (a fallback is the old
        behaviour, not a statement that the file system is safe)."""
        databases = {"meta.sqlite": self.store.journal_report(), "agy-receipts.sqlite": self._receipts().journal_report()}
        report = {"policy": self.store.journal_policy, "synchronous": "FULL", "databases": databases}
        fallback = [{"database": name, "reason": entry["reason"]} for name, entry in databases.items()
                    if entry["fallback"]]
        return report, fallback

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
            **self._sparse_status(),
        }
        status["journal"], status["journal_fallback"] = self._journal_status()
        worker = self._integrity_worker
        status["integrity_worker"] = worker.snapshot() if worker is not None else None
        if self.sparse_resolution.get("reason"):
            status["degraded_flags"].append("sparse_legacy_fallback")       # auto asked for compact and this SQLite cannot
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
                if self.store is not None and self._ready and self.store.has_pending_derived():
                    try:
                        with self._publication_guard():
                            self.store.flush_derived()
                            self._receipts(held=self.store.derived).flush_held()
                    except Exception:
                        pass    # held rows are a cache; closing must not fail because they could not be written
                if self.defer_derived:
                    from self_directing_mcp.agy_receipts import ReceiptStore
                    ReceiptStore.release(Path(self.settings.index_dir))     # the persistent receipts reader, if any
                if self._ready:
                    self._receipts().checkpoint()       # fold a write-ahead log into the file; a no-op without one
                if self.store is not None:
                    self.store.close()                  # checkpoints meta.sqlite's log (if any), then closes
                for component in (self.sparse, self.dense, self.graph):
                    connection = getattr(component, "_conn", None)
                    if connection is not None:
                        connection.close()
                self.store = self.sparse = self.dense = self.retriever = self.contracts = None
                self.graph = self.neograph = None
                self._ready = False
            finally:
                self._closing = False
                self._lifetime.notify_all()
