from __future__ import annotations

from typing import Any
import argparse
import asyncio
import ctypes
import logging
import sys
import time
import traceback
from threading import Event

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import TypeAdapter, ValidationError, create_model

from self_directing_mcp.config import get_settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.schemas import ContractRule, ProposedAction
from self_directing_mcp.codex_hooks import handle_hook
from self_directing_mcp.security.mask import mask_secrets, mask_value
from self_directing_mcp.request_control import (
    IndexBusy, RequestControl, RequestStopped, check_request, request_scope,
)

HOOK_INSTRUCTIONS = (
    "Local session history and explicit constraint checks, used when relevant. "
    "Register only user-stated constraints; check applicable actions and completion obligations. "
    "Questions and ordinary reads need no ritual sync/audit. Audits refresh local evidence themselves. "
    "Violation/suspicious: stop the affected action. Unknown is unverified. "
    "A skipped hook is not clean or authorization. Session text stays local unless remote retrieval is authorized."
)



class _MaskedFastMCP(FastMCP):
    """Tool failures embed argument values and exception text; mask them before the client sees them."""

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        try:
            return await super().call_tool(name, arguments)
        except ToolError as exc:
            raise ToolError(mask_secrets(str(exc))) from None


mcp = _MaskedFastMCP(
    "self-directing-mcp",
    instructions=HOOK_INSTRUCTIONS,
)

# Rule logic must round-trip exactly (list -> edit -> upsert) and single-use grants are credentials the
# caller must replay; every other string leaving a tool is masked.
_RULE_KEYS = frozenset({"regex", "before_regex", "exception_regex"})
_GRANT_KEYS = frozenset({"grant"})


class _MaskingFilter(logging.Filter):
    """Mask log messages and formatted tracebacks on every handler that receives them."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg, record.args = mask_secrets(record.getMessage()), None
            if record.exc_info and not record.exc_text:
                record.exc_text = mask_secrets("".join(traceback.format_exception(*record.exc_info)))
            if record.stack_info:
                record.stack_info = mask_secrets(record.stack_info)
        except Exception:
            record.msg, record.args, record.exc_info = "log record withheld (masking failed)", None, None
        return True


def _mask_logging() -> None:
    mask = next((f for f in logging.getLogger().filters if isinstance(f, _MaskingFilter)), None)
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, _MaskingFilter) for f in handler.filters):
            handler.addFilter(mask or _MaskingFilter())


_mask_logging()

_engine = SelfDirectEngine()

_contract_list_adapter = TypeAdapter(list[ContractRule])


def _format_contract_errors(contracts, exc: ValidationError) -> str:
    """index, id, field path and fix hint; never the full raw contract."""
    lines = []
    for err in exc.errors():
        loc = err.get("loc", ())
        idx = loc[0] if loc and isinstance(loc[0], int) else None
        field_path = ".".join(str(p) for p in loc[1:]) or "value"
        contract_id = None
        if idx is not None and idx < len(contracts):
            raw = contracts[idx]
            if isinstance(raw, dict):
                contract_id = raw.get("id")
            elif isinstance(raw, ContractRule):
                contract_id = raw.id
        target = f"contracts[{idx}]" if idx is not None else "contracts"
        if contract_id:
            target += f" (id={contract_id!r})"
        msg = err.get("msg", "invalid value")
        value = err.get("input")
        if isinstance(value, (str, int, float, bool)):
            # Scalar field values only; never echo a whole contract object.
            msg += f"; got {value!r}"
        lines.append(f"{target}.{field_path}: {msg}")
    return "Invalid contracts input; nothing was saved. " + "; ".join(lines)

def _incomplete_response(reason):
    return {"ok": False, "verdict": "unknown", "error": reason,
            "coverage": {"complete": False, "issues": [reason]},
            "action_executed": False, "requires_attention": True,
            "nudge": "Local work did not complete; this is not compliance or proof of rollback."}


def _collect_worker_exception(worker):
    if not worker.cancelled():
        worker.exception()


# Heavy work since the last trim, summed over calls: many short calls (hook checks) fragment the heap as much as
# one long call, so the threshold applies to the total, not to a single call.
TRIM_MIN_CALL_SEC = 0.5
TRIM_MIN_INTERVAL_SEC = 5.0
_last_trim = 0.0
_busy_since_trim = 0.0


def _libc():
    try:
        return ctypes.CDLL("libc.so.6")
    except OSError:
        return None


def _trim_heap():
    """Return freed glibc heap to the OS; Linux/glibc only, failures ignored."""
    if not sys.platform.startswith("linux"):
        return
    try:
        trim = getattr(_libc(), "malloc_trim", None)
        if trim is not None:
            trim(0)
    except Exception:
        pass


class _HeavyGate:
    """Loop-local semaphore for heavy engine work. Slots are released only when the
    worker thread really finishes, so abandoned (deadline-expired) workers still count."""

    def __init__(self):
        self._loop = self._limit = self._sem = None
        self.running = 0

    async def acquire(self, deadline):
        loop, limit = asyncio.get_running_loop(), max(1, _engine.settings.max_concurrent_calls)
        if self._sem is None or self._loop is not loop or self._limit != limit:
            self._loop, self._limit, self._sem, self.running = loop, limit, asyncio.Semaphore(limit), 0
        sem = self._sem
        try:
            await asyncio.wait_for(sem.acquire(), max(0.0, deadline - time.monotonic()))
        except TimeoutError:
            raise RequestStopped("request_deadline") from None
        self.running += 1
        return sem

    def release(self, sem, started):
        sem.release()
        if sem is not self._sem:
            return
        self.running -= 1
        global _last_trim, _busy_since_trim
        now = time.monotonic()
        _busy_since_trim += now - started
        if (self.running == 0 and _busy_since_trim >= TRIM_MIN_CALL_SEC
                and now - _last_trim >= TRIM_MIN_INTERVAL_SEC):
            _last_trim, _busy_since_trim = now, 0.0
            self._loop.run_in_executor(None, _trim_heap)


_gate = _HeavyGate()


async def _await_request(work, control, budget, heavy=True):
    # Keep the actual thread task alive to collect its eventual exception.
    slot = await _gate.acquire(control.deadline) if heavy else None
    started = time.monotonic()
    try:
        worker = asyncio.create_task(asyncio.to_thread(work))
    except BaseException:
        if slot is not None:
            _gate.release(slot, started)
        raise
    worker.add_done_callback(_collect_worker_exception)
    if slot is not None:
        worker.add_done_callback(lambda _w: _gate.release(slot, started))
    try:
        done, _ = await asyncio.wait({worker}, timeout=max(0.0, control.deadline - time.monotonic()))
        if not done:
            control.cancel("request_deadline")
            raise RequestStopped("request_deadline")
        return worker.result()
    except asyncio.CancelledError:
        control.cancel()
        raise


async def _dispatch(method, *args, audit=False, **kwargs):
    """Bound the response and share cooperative stop state with its worker."""
    budget = kwargs.pop("budget_override", None)
    heavy = kwargs.pop("heavy", True)
    keep = kwargs.pop("keep_keys", frozenset())
    if budget is None:
        budget = _engine.settings.audit_timeout_sec if audit else _engine.settings.request_timeout_sec
    control = RequestControl(deadline=time.monotonic() + budget)

    def work():
        with request_scope(control):
            check_request()
            try:
                result = method(*args, _deadline=control.deadline,
                                cancel_event=control.cancel_event, **kwargs)
            except (IndexBusy, RequestStopped):
                raise
            except Exception:
                check_request()
                raise
            check_request()
            return result

    try:
        return mask_value(await _await_request(work, control, budget, heavy), skip_keys=keep)
    except (IndexBusy, RequestStopped) as exc:
        return _incomplete_response(exc.reason)
    except TimeoutError:
        return _incomplete_response("operation_timeout")


def _resolve_timeout_sec(timeout_ms: int | None, *, audit: bool) -> float:
    """Client-supplied budget; default unchanged. Capped by max_tool_timeout_sec."""
    settings = _engine.settings
    default = settings.audit_timeout_sec if audit else settings.request_timeout_sec
    if timeout_ms is None:
        return default
    if timeout_ms <= 0:
        raise ToolError("timeout_ms must be a positive integer (milliseconds).")
    requested = timeout_ms / 1000.0
    if requested > settings.max_tool_timeout_sec:
        raise ToolError(
            f"timeout_ms {timeout_ms} exceeds max_tool_timeout_sec "
            f"{settings.max_tool_timeout_sec:g} (SELF_DIRECT_MAX_TOOL_TIMEOUT_SEC)."
        )
    return requested


async def _dispatch_hook(*args, **kwargs):
    budget = _engine.settings.audit_timeout_sec
    control = RequestControl(deadline=time.monotonic() + budget)
    def work():
        with request_scope(control):
            check_request()
            result = handle_hook(_engine, *args, _deadline=control.deadline,
                                 cancel_event=control.cancel_event, **kwargs)
            check_request()
            return result
    try:
        return mask_value(await _await_request(work, control, budget))
    except (IndexBusy, RequestStopped, TimeoutError) as exc:
        reason = exc.reason if isinstance(exc, (IndexBusy, RequestStopped)) else "operation_timeout"
        event = args[0]
        context = f"Self-directing audit: unknown ({reason}); do not claim compliance."
        return ({"systemMessage": context} if event == "Stop" else
                {"hookSpecificOutput": {"hookEventName": event, "additionalContext": context}})



@mcp.tool()
async def codex_session_hook(event_name: str, session_id: str | None = None,
                       transcript_path: str | None = None, tool_name: str | None = None,
                       tool_input: Any = None, turn_id: str | None = None,
                        stop_hook_active: bool = False, prompt: str | None = None) -> dict[str, Any]:
    """Codex lifecycle adapter: local audit context only; never block, execute, or restart.

    Used by UserPromptSubmit, PreToolUse and Stop mcp_tool hooks. Event values come
    from Codex, not model guesses. It does not embed or send session text remotely.
    """
    return await _dispatch_hook(event_name, session_id, transcript_path, tool_name,
                                 tool_input, turn_id=turn_id, stop_hook_active=stop_hook_active, prompt=prompt)


@mcp.tool()
async def sync_session(
    session_id: str | None = None,
    path: str | None = None,
    provider: str | None = None,
    embed: bool = True,
    timeout_ms: int | None = None,
) -> dict[str, Any]:
    """Index/update local session history when retrieval needs fresh evidence.

    Audit/check_action refresh automatically; do not sync separately before them.
    Provide session_id or path under the configured root.
    provider: "codex" (default), "grokbot", "omp" (OMP JSONL), "agy" or "claude" (Claude Code transcripts).
    When omitted, uses SELF_DIRECT_SESSION_PROVIDER or auto-detects from path layout.
    Every event occurrence is retained; only new sanitized text hashes are embedded.
    Set embed=False for local indexing. SELF_DIRECT_LOCAL_ONLY=true disables
    embeddings and dense indexes entirely; embed=True then returns an error.
    check_action/audit_session always refresh without embedding. Example contracts are disabled by default.
    timeout_ms: optional per-call budget in milliseconds for large sessions
    (overrides the default request timeout, capped by max_tool_timeout_sec).
    """
    budget = _resolve_timeout_sec(timeout_ms, audit=False)
    return await _dispatch(_engine.sync_session, session_id=session_id, path=path, provider=provider, embed=embed, budget_override=budget)


@mcp.tool()
async def search_history(
    query: str,
    session_id: str | None = None,
    mode: str = "hybrid",
    top_k: int | None = None,
    provider: str | None = None,
    timeout_ms: int | None = None,
) -> dict[str, Any]:
    """Search indexed session history (hybrid|regex|sparse|dense).

    Use when relevant evidence is missing. Do NOT treat retrieval
    alone as compliance proof — call audit_session; contracts/regex are authoritative.
    top_k defaults from settings (search_top_k). Never treat top-1 alone as a violation.
    provider filters the index (codex|grokbot|omp|agy|claude). Local-only supports regex/sparse, not dense/hybrid.
    """
    budget = _resolve_timeout_sec(timeout_ms, audit=False)
    return await _dispatch(_engine.search_history,
        query, session_id=session_id, mode=mode, top_k=top_k, provider=provider,
        budget_override=budget
    )


@mcp.tool()
async def audit_session(session_id: str, provider: str | None = None, path: str | None = None,
                        timeout_ms: int | None = None) -> dict[str, Any]:
    """Verify explicit contracts at a relevant completion checkpoint (detect only).

    Use check_action for proposed actions. No automatic audit is needed for every
    turn or question. Verdicts: violation|suspicious|clean|unknown.
    If violation or suspicious: STOP, report findings, do not proceed with the risky action.
    Rules/regex are primary; hybrid is auxiliary only.
    Refreshes local JSONL first and returns coverage and a contracts snapshot.
    Unknown means evidence/verification is missing; it must not be called compliance.
    provider filters the audit (codex|grokbot|omp|agy|claude).
    timeout_ms: optional per-call budget in milliseconds for large sessions.
    """
    budget = _resolve_timeout_sec(timeout_ms, audit=True)
    return await _dispatch(_engine.audit_session, session_id, provider=provider, path=path, audit=True, budget_override=budget)


@mcp.tool()
async def check_action(session_id: str, action: ProposedAction, provider: str | None = None, path: str | None = None,
                       timeout_ms: int | None = None) -> dict[str, Any]:
    """Check a proposed tool call against current contracts and refreshed local history.

    action contains tool_name and arguments (object or string). Supports must_not
    rules and must prerequisites such as successful tests before deploy.
    Never executes the action or persists it as an executed event. A clean result
    is limited to the returned coverage and contracts snapshot; it is not permission.
    """
    budget = _resolve_timeout_sec(timeout_ms, audit=True)
    return await _dispatch(_engine.check_action, session_id, action.model_dump(), provider=provider, path=path, audit=True, budget_override=budget)


@mcp.tool()
async def get_chunk(chunk_id: str) -> dict[str, Any]:
    """Fetch a chunk by id. Secrets in text are masked (sk-, Bearer, api_key=)."""
    return await _dispatch(_engine.get_chunk, chunk_id, heavy=False)


@mcp.tool()
async def list_contracts(session_id: str | None = None) -> dict[str, Any]:
    """List contracts for the current session plus explicit global rules.

    session_id selects that session's rules together with global rules
    (session_id is null). Omit session_id to list only global rules.
    Other sessions' contracts are not returned. This is a view over the
    shared store, not a per-session store split.
    """
    return await _dispatch(_engine.list_contracts, heavy=False, keep_keys=_RULE_KEYS, session_id=session_id)


@mcp.tool()
async def revoke_contract(contract_id: str) -> dict[str, Any]:
    """Disable a contract by id. History and the user's source quote are retained."""
    return await _dispatch(_engine.revoke_contract, contract_id, heavy=False, keep_keys=_RULE_KEYS)


@mcp.tool()
async def upsert_contracts(contracts: list[dict[str, Any]]) -> dict[str, Any]:
    """Create/update explicit contracts; this is not an authorization grant.

    Each contract requires id and type. type="must" is a required user
    requirement; type="must_not" is a prohibited action/effect.
    Example: [{"id": "no-delete", "type": "must_not", "scope": "tool_call",
    "regex": "rm\\s+-rf", "description": "Never delete user files."}]
    Optional fields: provider, session_id, roles (default assistant), source_event_id,
    applies_from_event_id (exclusive), before_regex, requires_success. before_regex
    and requires_success apply to must tool_call rules with regex. Revisions increment
    on update. Description-only rules return unknown until a verifier exists.
    Set risk="high", auto_continue=true or when={"tool_call_regex": ...} only when
    the user asked for that: risk="high" marks actions the user named high-risk (a
    host may ask instead of allowing when verification fails); auto_continue=true
    (must with regex only) lets a confirmed, fully observed miss resume a stopping
    agent once per session; when limits a must obligation to sessions where a
    matching tool call was observed (otherwise it is reported not_applicable).
    The response contains only the contracts from this call after save,
    not the full store. session_id=null means a global rule that applies
    to every audited session. Use list_contracts(session_id=...) to inspect
    the current session plus globals.
    """
    try:
        validated = _contract_list_adapter.validate_python(contracts)
    except ValidationError as exc:
        raise ToolError(_format_contract_errors(contracts, exc))
    return await _dispatch(_engine.upsert_contracts, validated, heavy=False, keep_keys=_RULE_KEYS)


# Replace the auto-generated input schema with a typed one that exposes
# required fields, enum values and field descriptions, while runtime inputs
# stay dicts so validation errors can be formatted here instead of by FastMCP.
_upsert_contract_tool = mcp._tool_manager._tools.get("upsert_contracts")
if _upsert_contract_tool is not None:
    _ContractsInputModel = create_model(
        "upsert_contractsArguments",
        contracts=(list[ContractRule], ...),
    )
    _upsert_contract_tool.parameters = _ContractsInputModel.model_json_schema(by_alias=True)


@mcp.tool()
async def audit_status(session_id: str | None = None, provider: str | None = None,
                       timeout_ms: int | None = None) -> dict[str, Any]:
    """Index health: cursor, chunk counts, degraded flags (numpy/fake embedder).

    Call when diagnosing sync gaps. Prefer sync_session + audit_session for the hook workflow.
    """
    budget = _resolve_timeout_sec(timeout_ms, audit=True)
    return await _dispatch(_engine.audit_status, heavy=False, session_id=session_id, provider=provider, audit=True, budget_override=budget)


@mcp.tool()
async def update_checklist(
    session_id: str,
    provider: str | None = None,
    add: list[dict[str, Any]] | None = None,
    update: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Track user requirements as verifiable checklist items with provenance.

    add: one item per independently verifiable requirement, keeping the user's
    original wording (text) and source. update: set status or attach
    evidence_chunk_ids. A done claim without evidence stays pending_verification.
    Items persist locally per session and survive context compaction.
    """
    return await _dispatch(_engine.update_checklist, session_id, provider=provider, add=add, update=update)


@mcp.tool()
async def get_checklist(session_id: str, provider: str | None = None) -> dict[str, Any]:
    """Read the requirement checklist: statuses, evidence links and history."""
    return await _dispatch(_engine.get_checklist, session_id, provider=provider)


@mcp.tool()
async def analyze_activity(session_id: str, provider: str | None = None,
                           window_minutes: int = 60) -> dict[str, Any]:
    """SQLite time-series analysis of session activity (no external DB).

    Returns repeated errors with retry gaps, failure rates per time window,
    stalled checklist items, behavior around user instructions, post-test
    modifications and fix->reverify sequences — each with evidence chunk ids
    and a data-completeness flag. Observations only; time order never proves
    cause. Timestamps are used as-is; missing ones are reported, not fabricated.
    """
    return await _dispatch(_engine.analyze_activity, session_id, provider=provider,
                           window_minutes=window_minutes, audit=True)


@mcp.tool()
async def get_graph_context(session_id: str, node_id: str | None = None,
                            provider: str | None = None, depth: int = 1,
                            path: str | None = None) -> dict[str, Any]:
    """Query the GraphRAG knowledge graph: nodes, edges, and entity dependencies.

    If node_id is provided, returns the entity's neighbor subgraph up to depth.
    If node_id is omitted, returns the active session knowledge graph.
    """
    return await _dispatch(_engine.get_graph_context, session_id, node_id=node_id,
                           provider=provider, depth=depth, path=path, audit=True)


@mcp.tool()
async def propose_graph_update(session_id: str, proposal: dict[str, Any],
                               provider: str | None = None) -> dict[str, Any]:
    """Validate a knowledge-graph node/edge proposal before storing.

    Edges require ALLOWED relations (IMPLEMENTS, DEPENDS_ON, MODIFIES, PRODUCES,
    CHECKS_VERSION, EVIDENCED_BY, SUPERSEDES) and evidence_chunk_ids present in
    the indexed session. origin marks observed facts vs agent-inferred relations.
    Nothing is stored during validation.
    """
    return await _dispatch(_engine.propose_graph_update, session_id, proposal, provider=provider, audit=True)


@mcp.tool()
async def commit_graph_update(session_id: str, proposal: dict[str, Any],
                              base_graph_version: int, provider: str | None = None) -> dict[str, Any]:
    """Apply a validated graph proposal atomically and advance the processing cursor.

    Idempotent by job_id: replaying an applied job returns already_applied.
    Rejected proposals store nothing and never advance the cursor. A failed
    write reports status=unclear; the range is not advanced and recovery is
    re-run. Existing search/audit paths keep working when the graph lags.
    """
    return await _dispatch(_engine.commit_graph_update, session_id, proposal, base_graph_version, provider=provider, audit=True)


@mcp.tool()
async def run_neograph_update(session_id: str, provider: str | None = None,
                               timeout_ms: int | None = None) -> dict[str, Any]:
    """Run the full GraphRAG update as a NeoGraph topology with the live LLM.

    Topology: extract (DeepSeek via OpenRouter) -> validate (MCP evidence
    checks) -> apply (atomic write + cursor advance). Requires OPENROUTER_API_KEY
    and neograph-engine. Each run leaves a durable NeoGraph checkpoint so a
    failed or interrupted run remains inspectable per thread.
    """
    budget = (_resolve_timeout_sec(timeout_ms, audit=False) if timeout_ms is not None else
              min(_engine.settings.graph_update_timeout_sec, _engine.settings.max_tool_timeout_sec))
    return await _dispatch(_engine.run_neograph_update, session_id, provider=provider,
                           budget_override=budget)


async def _dispatch_workflow(method: str, *args, timeout_ms=None, **kwargs):
    from self_directing_mcp.workflow import configured_workflow
    budget = _resolve_timeout_sec(timeout_ms, audit=False)
    deadline = time.monotonic() + budget
    abandoned, entered = Event(), Event()

    def invoke():
        if abandoned.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("workflow request expired before execution")
        entered.set()
        controller = configured_workflow()
        if controller is None:
            return {"ok": False, "error": "workflow_not_configured", "action_executed": False}
        return getattr(controller, method)(*args, _deadline=deadline, cancel_event=abandoned, **kwargs)

    worker = asyncio.create_task(asyncio.to_thread(invoke))
    try:
        done, _ = await asyncio.wait({worker}, timeout=budget)
        if done:
            return mask_value(worker.result(), skip_keys=_GRANT_KEYS)
        abandoned.set()
        # A process already running must be killed and its observed result kept.
        # A worker still waiting on a host lock must never launch after expiry.
        done, _ = await asyncio.wait({worker}, timeout=.25)
        if done:
            return mask_value(worker.result(), skip_keys=_GRANT_KEYS)
        may_have_run = method == "execute" and entered.is_set()
        return {"ok": False, "error": "workflow_deadline", "outcome": "outcome_unknown" if may_have_run else "not_started",
                "outcome_unknown": may_have_run, "started": None if may_have_run else False,
                "action_executed": None if may_have_run else False, "requires_attention": True}
    except asyncio.CancelledError:
        abandoned.set()
        raise
    finally:
        # Retrieve a late worker exception without cancelling its cleanup or
        # pretending that cancelling an asyncio Future terminates a process.
        worker.add_done_callback(_collect_worker_exception)


@mcp.tool()
async def workflow_status(session_id: str, provider: str = "agy") -> dict[str, Any]:
    """Read the host-approved workflow, current obligations and observed evidence."""
    return await _dispatch_workflow("status", session_id, provider=provider)


@mcp.tool()
async def workflow_run_checks(session_id: str, provider: str = "agy", kind: str = "all",
                              timeout_ms: int | None = None) -> dict[str, Any]:
    """Run owner-approved real tests/proofs; bind observations to current versions.

    No model-supplied pass flag, command, expected result or proof can be submitted.
    """
    return await _dispatch_workflow("run_checks", session_id, provider=provider,
                                    kind=kind, timeout_ms=timeout_ms)


@mcp.tool()
async def workflow_classify(session_id: str, provider: str = "agy") -> dict[str, Any]:
    """Request JEV Choice using explicitly approved synthetic requirement context.

    The host reuses OPENROUTER_API_KEY / SELF_DIRECT_OPENROUTER_API_KEY_FILE for
    pinned typesafe/jev-1.13 on OpenRouter. Errors and low confidence never
    authorize a transition. Each classification makes at most one remote call;
    no transcript is sent.
    """
    return await _dispatch_workflow("classify", session_id, provider=provider)


@mcp.tool()
async def workflow_transition(session_id: str, classification_id: str,
                               provider: str = "agy") -> dict[str, Any]:
    """Evaluate a current classification through the compiled Lean policy."""
    return await _dispatch_workflow("transition", session_id, classification_id, provider=provider)


@mcp.tool()
async def workflow_authorize(session_id: str, classification_id: str, action: dict[str, Any],
                              provider: str = "agy") -> dict[str, Any]:
    """Mint a single-use grant bound to exact workflow_command argv/cwd and versions.

    Requires current proof/test observations. This does not execute the action.
    Owner approval is deliberately not an MCP tool.
    """
    return await _dispatch_workflow("authorize", session_id, classification_id, action, provider=provider)


@mcp.tool()
async def workflow_execute(session_id: str, grant: str, action: dict[str, Any],
                            provider: str = "agy", timeout_ms: int | None = None) -> dict[str, Any]:
    """Recheck and consume a grant, execute its exact argv, and observe the outcome.

    Changed arguments, targets, state, policy or evidence invalidate the grant.
    """
    return await _dispatch_workflow("execute", session_id, grant, action, provider=provider,
                                    timeout_ms=timeout_ms)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Tsukkomi MCP stdio server.")
    parser.add_argument("--omp-bridge", action="store_true",
                        help="Use OMP settings and validate private remote sources before startup.")
    args = parser.parse_args(argv)
    if args.omp_bridge:
        try:
            settings = get_settings(session_provider="omp")
            if not settings.local_only:
                key_file = settings.openrouter_api_key_file
                native_path = settings.sqlite_vector_path
                if key_file is None or not key_file.expanduser().is_file():
                    raise ValueError("Missing selected key file")
                if native_path is None or not native_path.expanduser().is_file():
                    raise ValueError("Missing selected native library")
                settings.resolve_api_key()
        except (OSError, ValueError):
            parser.error("OMP bridge configuration is unavailable or invalid")
        # Bind the shared source to the singleton used by every MCP handler.
        # Indexes, native loading, and remote clients remain lazy.
        _engine.settings = settings
    if not _engine.settings.local_only:
        # Dense retrieval and embeddings need NumPy. Load it now, while the process is single-threaded, so the
        # one-BLAS-thread setup of _runtime never races a worker thread that spawns a child. A local-only server
        # never touches a dense vector, so it never loads NumPy at all (about 25 MB of resident memory).
        from self_directing_mcp._runtime import load_numpy
        load_numpy()
    _mask_logging()
    try:
        mcp.run(transport="stdio")
    except Exception:
        # A crash traceback can quote request content; emit it masked.
        print(mask_secrets(traceback.format_exc()), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
