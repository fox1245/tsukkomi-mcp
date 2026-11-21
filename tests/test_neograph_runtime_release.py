"""A host stage that fails must not keep the work of its call alive.

``run_stages`` runs the stages of one operation in a native graph. A failing stage's exception used to be kept in a
closure cell that the native node (and so the native engine, which Python's cyclic collector cannot see into) held on to,
while the exception's traceback held the frame of ``run_stages`` and with it that engine: a cycle through native memory
that no collection breaks. Everything the failed call's frames could reach stayed alive for the life of the process: for
an audit of a long session that is the whole list of its events, every time a request stopped one (the resident daemon's
idle comparison is stopped by the next request, many times an hour). The daemon grew by about 25 MB per such stop on a
60 MB session until it left on its memory limit.
"""
from __future__ import annotations

import gc
import threading
import weakref

import pytest

from self_directing_mcp.audit import runner as audit_runner
from self_directing_mcp.index.ingest import AuditEventSnapshot
from self_directing_mcp.neograph_runtime import run_stages
from self_directing_mcp.request_control import RequestStopped, request_scope
from self_directing_mcp.schemas import AuditEvent, ContractRule


class Payload:
    """Something a stage's callback reaches, as an audit's callbacks reach their event lists."""


def call(make_failure):
    payload = Payload()
    seen = weakref.ref(payload)

    def work():
        payload.touched = True
        if make_failure is not None:
            raise make_failure()

    try:
        run_stages("release_probe", [("only", work)])
    except BaseException as exc:                      # noqa: BLE001 - the caller of an operation gets it and moves on
        assert make_failure is not None and type(exc) is type(make_failure())
    return seen


@pytest.mark.parametrize("make_failure", [None, lambda: ValueError("boom"), lambda: RequestStopped("request_cancelled")],
                         ids=["no failure", "an error", "a stop"])
def test_nothing_a_stage_reached_outlives_its_call(make_failure):
    seen = [call(make_failure) for _ in range(5)]
    gc.collect()
    assert [ref() for ref in seen] == [None] * 5


def test_the_failure_still_reaches_the_caller_unchanged():
    error = ValueError("boom")
    with pytest.raises(ValueError) as caught:
        run_stages("release_probe", [("first", lambda: None), ("second", lambda: (_ for _ in ()).throw(error))])
    assert caught.value is error


def live_events() -> int:
    return sum(1 for item in gc.get_objects() if type(item) is AuditEvent)


def test_a_stopped_audit_leaves_none_of_its_events_alive():
    """What the resident daemon's idle comparison does when a request arrives: the audit is stopped part way."""
    rule = ContractRule(id="no-delete", type="must_not", scope="tool_call", regex="DELETE")
    cancel = threading.Event()

    class StoppedWhileLoading(AuditEventSnapshot):
        def list_chunks(self, session_id=None, provider=None):
            cancel.set()                        # the stop arrives while the evidence is being loaded
            return super().list_chunks(session_id, provider)

    def stopped_audit():
        cancel.clear()
        events = tuple(AuditEvent(f"c{n}", "tool_call", f"echo {n}", {"role": "assistant"}, (n, 0)) for n in range(2000))
        store = StoppedWhileLoading("s", "claude", events, len(events))
        with request_scope(cancel_event=cancel):
            try:
                audit_runner.run_audit(session_id="s", contracts=[rule], store=store, provider="claude")
            except RequestStopped:
                return
        raise AssertionError("the audit was not stopped")

    stopped_audit()                             # warm-up: imports, compiled patterns
    gc.collect()
    before = live_events()
    for _ in range(3):
        stopped_audit()
    gc.collect()
    assert live_events() == before
