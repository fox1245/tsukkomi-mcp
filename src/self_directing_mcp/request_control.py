"""Request-local cooperative stopping and ordered publication admission.

Admission is a small state decision, not a lock held over I/O. Work admitted
before cancellation may finish afterward; this module does not undo commits.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
import threading
import time


class RequestStopped(TimeoutError):
    def __init__(self, reason: str = "request_cancelled"):
        if reason not in {"request_deadline", "request_cancelled"}:
            raise ValueError("Invalid request stop reason")
        self.reason = reason
        super().__init__(reason)


class IndexBusy(TimeoutError):
    reason = "index_busy"

    def __init__(self):
        super().__init__(self.reason)


class RequestControl:
    def __init__(self, deadline=None, cancel_event=None):
        self.deadline = deadline
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self._gate = threading.RLock()
        self._reason = None
        self._parents = ()

    def _check(self):
        for parent in self._parents:
            parent._check()
        if self._reason is not None:
            raise RequestStopped(self._reason)
        if self.cancel_event.is_set():
            raise RequestStopped("request_cancelled")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise RequestStopped("request_deadline")

    def check(self):
        with self._gate:
            self._check()

    def _lineage(self):
        controls = []
        pending = [self]
        while pending:
            control = pending.pop()
            if control not in controls:
                controls.append(control)
                pending.extend(control._parents)
        return sorted(controls, key=id)

    def cancel(self, reason="request_cancelled"):
        if reason not in {"request_deadline", "request_cancelled"}:
            raise ValueError("Invalid request stop reason")
        controls = self._lineage()
        with ExitStack() as gates:
            for control in controls:
                gates.enter_context(control._gate)
            for control in controls:
                if control.cancel_event is self.cancel_event and control._reason is None:
                    control._reason = reason
            self.cancel_event.set()

    @contextmanager
    def publication(self):
        with ExitStack() as gates:
            for control in self._lineage():
                gates.enter_context(control._gate)
            self._check()
        # Cancellation can proceed independently of the admitted I/O.
        yield


_active: ContextVar[RequestControl | None] = ContextVar("self_directing_request", default=None)


def current_request() -> RequestControl | None:
    return _active.get()


def check_request() -> None:
    control = current_request()
    if control is not None:
        control.check()


@contextmanager
def request_scope(control=None, *, deadline=None, cancel_event=None):
    parent = current_request()
    sources = tuple(dict.fromkeys(c for c in (parent, control) if c is not None))
    base = control if control is not None else parent
    deadlines = [c.deadline for c in sources if c.deadline is not None]
    if deadline is not None:
        deadlines.append(deadline)
    effective_deadline = min(deadlines) if deadlines else None
    if base is None:
        if deadline is not None or cancel_event is not None:
            control = RequestControl(deadline, cancel_event)
    elif (len(sources) > 1 or effective_deadline != base.deadline or
          (cancel_event is not None and cancel_event is not base.cancel_event)):
        # Keep all cancellation sources live; child budgets cannot extend them.
        control = RequestControl(effective_deadline,
                                 cancel_event if cancel_event is not None else base.cancel_event)
        control._parents = sources
    else:
        control = base
    token = _active.set(control)
    try:
        yield control
    finally:
        _active.reset(token)


def request_operation(method):
    """Bind canonical entry inputs without acquiring an engine/index lease."""
    parameters = inspect.signature(method).parameters
    @wraps(method)
    def wrapped(*args, **kwargs):
        deadline = kwargs.pop("_deadline", None)
        cancel_event = kwargs.pop("cancel_event", None)
        with request_scope(deadline=deadline, cancel_event=cancel_event) as control:
            check_request()
            if "_deadline" in parameters:
                kwargs["_deadline"] = control.deadline if control is not None else deadline
            if "cancel_event" in parameters:
                kwargs["cancel_event"] = control.cancel_event if control is not None else cancel_event
            try:
                result = method(*args, **kwargs)
            except (RequestStopped, IndexBusy):
                raise
            except Exception:
                check_request()
                raise
            check_request()
            return result
    return wrapped


@contextmanager
def publication():
    control = current_request()
    if control is None:
        yield
    else:
        with control.publication():
            yield
