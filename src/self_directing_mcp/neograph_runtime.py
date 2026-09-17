"""Required native NeoGraph execution for host-owned, request-local stages."""
from __future__ import annotations

import threading
import time
import uuid
from functools import wraps
from collections.abc import Callable, Sequence

try:
    import neograph_engine as ng
except (ImportError, OSError) as exc:
    raise RuntimeError(
        "NeoGraph native runtime is required. Install neograph-engine==0.12.1 "
        "in this MCP's configured Python environment; no direct executor fallback."
    ) from exc

_pending: dict[str, dict[str, Callable]] = {}
_guard = threading.RLock()
_TYPE = "self_directing_host_stage_v1"


class _Stage(ng.GraphNode):
    def __init__(self, name, callback):
        super().__init__()
        self.name, self.callback = name, callback

    def get_name(self):
        return self.name

    def run(self, input):
        self.callback()
        return [ng.ChannelWrite("completed", [self.name])]


def _factory(name, config, ctx):
    with _guard:
        callback = _pending[ctx.extra_config["invocation"]][name]
    return _Stage(name, callback)


ng.NodeFactory.register_type(_TYPE, _factory)


def graph_operation(name):
    """Route a host-owned operation through a native graph, with honest receipts."""
    def decorate(method):
        @wraps(method)
        def wrapped(*args, **kwargs):
            result = None
            def invoke():
                nonlocal result
                result = method(*args, **kwargs)
            execution = run_stages(name, [(name, invoke)])
            return {**result, "executor": "neograph-engine", "execution": execution}
        return wrapped
    return decorate


def run_stages(name: str, stages: Sequence[tuple[str, Callable]]) -> dict:
    """Execute each stage in the C++ graph; errors abort downstream execution.

    Callback state stays request-local, never in a global factory or durable
    checkpoint. These short graphs restart from fresh inputs after interruption;
    they do not advertise recovery of Python closures or external side effects.
    """
    names = [stage for stage, _ in stages]
    if not names or len(set(names)) != len(names):
        raise ValueError("NeoGraph stages must have unique, nonempty names")
    invocation = uuid.uuid4().hex
    ctx = ng.NodeContext()
    ctx.extra_config = {"invocation": invocation}
    definition = {
        "schema_version": ng.TOPOLOGY_SCHEMA_VERSION,
        "name": name,
        "channels": {"completed": {"reducer": "append"}},
        "nodes": {stage: {"type": _TYPE} for stage in names},
        "edges": [{"from": left, "to": right} for left, right in
                  zip([ng.START_NODE, *names], [*names, ng.END_NODE])],
    }
    with _guard:
        _pending[invocation] = dict(stages)
    try:
        engine = ng.GraphEngine.compile(definition, ctx)
    finally:
        with _guard:
            _pending.pop(invocation, None)
    # Host callbacks use request-owned Python locks and SQLite connections.
    # Execute sequentially on the calling thread, not native fan-out workers.
    engine.set_worker_count(1)
    started = time.monotonic()
    result = engine.run(ng.RunConfig(thread_id=invocation, input={},
                                   resume_if_exists=False, max_steps=len(names) + 2))
    completed = result.output.get("channels", {}).get("completed", {}).get("value", [])
    trace = list(result.execution_trace or [])
    if completed != names or trace != names:
        raise RuntimeError("NeoGraph did not complete the required stage sequence")
    return {"executor": "neograph-engine", "version": ng.__version__,
            "run_id": invocation, "graph": name, "nodes": trace,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2)}
