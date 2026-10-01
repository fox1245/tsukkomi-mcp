from __future__ import annotations

from typing import Any
import asyncio
import time
from threading import Event

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import TypeAdapter, ValidationError, create_model

from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.schemas import ContractRule, ProposedAction
from self_directing_mcp.codex_hooks import handle_hook

HOOK_INSTRUCTIONS = (
    "Local session history and explicit constraint checks, used when relevant. "
    "Register only user-stated constraints; check applicable actions and completion obligations. "
    "Questions and ordinary reads need no ritual sync/audit. Audits refresh local evidence themselves. "
    "Violation/suspicious: stop the affected action. Unknown is unverified. "
    "A skipped hook is not clean or authorization. Session text stays local unless remote retrieval is authorized."
)

mcp = FastMCP(
    "self-directing-mcp",
    instructions=HOOK_INSTRUCTIONS,
)

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

async def _dispatch(method, *args, audit=False, **kwargs):
    """Keep JSON-RPC responsive; queued calls expire before touching the engine."""
    budget = kwargs.pop("budget_override", None)
    cooperative = kwargs.pop("cooperative", False)
    if budget is None:
        budget = _engine.settings.audit_timeout_sec if audit else _engine.settings.request_timeout_sec
    deadline = time.monotonic() + budget
    abandoned = Event()

    def work():
        if abandoned.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("request_expired_before_execution")
        if cooperative:
            return method(*args, _deadline=deadline, cancel_event=abandoned, **kwargs)
        return method(*args, _deadline=deadline, **kwargs)

    try:
        return await asyncio.wait_for(asyncio.to_thread(work), timeout=budget)
    except TimeoutError:
        abandoned.set()
        return {"ok": False, "verdict": "unknown", "error": "index_busy_or_deadline",
                "coverage": {"complete": False, "issues": ["index_busy_or_deadline"]},
                "action_executed": False, "requires_attention": True,
                "nudge": "Local audit could not complete within its budget; this is not compliance."}
    except asyncio.CancelledError:
        abandoned.set()
        raise


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
    # handle_hook uses bounded engine operations and returns hook-specific context.
    deadline = time.monotonic() + _engine.settings.audit_timeout_sec
    abandoned = Event()
    def work():
        if abandoned.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("request_expired_before_execution")
        return handle_hook(_engine, *args, _deadline=deadline, **kwargs)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(work),
            timeout=_engine.settings.audit_timeout_sec)
    except TimeoutError:
        abandoned.set()
        event = args[0]
        context = "Self-directing audit: unknown (local audit deadline); do not claim compliance."
        return ({"systemMessage": context} if event == "Stop" else
                {"hookSpecificOutput": {"hookEventName": event, "additionalContext": context}})
    except asyncio.CancelledError:
        abandoned.set()
        raise



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
    provider: "codex" (default), "grokbot", "omp" (OMP JSONL), or "agy".
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
    provider filters the index (codex|grokbot|omp|agy). Local-only supports regex/sparse, not dense/hybrid.
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
    provider filters the audit (codex|grokbot|omp|agy).
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
    return await _dispatch(_engine.get_chunk, chunk_id)


@mcp.tool()
async def list_contracts(session_id: str | None = None) -> dict[str, Any]:
    """List contracts for the current session plus explicit global rules.

    session_id selects that session's rules together with global rules
    (session_id is null). Omit session_id to list only global rules.
    Other sessions' contracts are not returned. This is a view over the
    shared store, not a per-session store split.
    """
    return await _dispatch(_engine.list_contracts, session_id=session_id)


@mcp.tool()
async def revoke_contract(contract_id: str) -> dict[str, Any]:
    """Disable a contract by id. History and the user's source quote are retained."""
    return await _dispatch(_engine.revoke_contract, contract_id)


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
    The response contains only the contracts from this call after save,
    not the full store. session_id=null means a global rule that applies
    to every audited session. Use list_contracts(session_id=...) to inspect
    the current session plus globals.
    """
    try:
        validated = _contract_list_adapter.validate_python(contracts)
    except ValidationError as exc:
        raise ToolError(_format_contract_errors(contracts, exc))
    return await _dispatch(_engine.upsert_contracts, validated)


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
    return await _dispatch(_engine.audit_status, session_id=session_id, provider=provider, audit=True, budget_override=budget)


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
                           budget_override=budget, cooperative=True)


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
        return await asyncio.wait_for(asyncio.shield(worker), timeout=budget)
    except TimeoutError:
        abandoned.set()
        # A process already running must be killed and its observed result kept.
        # A worker still waiting on a host lock must never launch after expiry.
        try:
            return await asyncio.wait_for(asyncio.shield(worker), timeout=.25)
        except TimeoutError:
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
        worker.add_done_callback(lambda finished: finished.exception() if not finished.cancelled() else None)


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

    The host supplies TYPESAFE_API_KEY or TSUKKOMI_TYPESAFE_KEY_FILE. Errors and
    low confidence never authorize a transition. No transcript is sent.
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


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
