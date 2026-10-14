"""Same-session syncs coalesce into one parse; waiters keep their own deadlines."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from self_directing_mcp.codex import parse as codex_parser
from self_directing_mcp.request_control import RequestStopped
from test_reliability import SID, call, message, write, session  # noqa: F401


def count_parses(monkeypatch, delay=0.0):
    calls = []
    original = codex_parser.parse_session_chunks

    def counted(*args, **kwargs):
        calls.append(kwargs.get("start_byte", 0))
        if delay:
            time.sleep(delay)
        return original(*args, **kwargs)

    monkeypatch.setattr(codex_parser, "parse_session_chunks", counted)
    return calls


def run_parallel(fn, n=8):
    barrier = threading.Barrier(n)

    def worker():
        barrier.wait(5)
        return fn()

    with ThreadPoolExecutor(n) as pool:
        return [f.result(30) for f in [pool.submit(worker) for _ in range(n)]]


def test_concurrent_syncs_parse_once(session, monkeypatch):
    engine, path = session
    write(path, [message(f"event {i}") for i in range(50)], "a")
    calls = count_parses(monkeypatch, delay=0.2)
    results = run_parallel(lambda: engine.sync_session(session_id=SID, embed=False))
    assert len(calls) == 1, calls
    assert all(r["ok"] for r in results)
    assert {r["total_chunks"] for r in results} == {engine.store.chunk_count(SID)}
    assert len({r["byte_offset"] for r in results}) == 1


def test_concurrent_parser_bump_reparses_once(session, monkeypatch):
    engine, path = session
    write(path, [message(f"event {i}") for i in range(20)], "a")
    with monkeypatch.context() as legacy:
        legacy.setattr(codex_parser, "PARSER_VERSION", 1)
        engine.sync_session(session_id=SID, embed=False)
    calls = count_parses(monkeypatch, delay=0.2)
    results = run_parallel(lambda: engine.sync_session(session_id=SID, embed=False))
    assert calls == [0], calls
    assert sum(r["reindexed"] for r in results) == 1


def test_waiter_deadline_and_leader_cancel_do_not_poison(session, monkeypatch):
    engine, path = session
    write(path, [message("hello")], "a")
    entered, release = threading.Event(), threading.Event()
    original = codex_parser.parse_session_chunks

    def slow(*args, **kwargs):
        entered.set()
        release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(codex_parser, "parse_session_chunks", slow)
    cancel = threading.Event()
    with ThreadPoolExecutor(3) as pool:
        leader = pool.submit(engine.sync_session, session_id=SID, embed=False, cancel_event=cancel)
        assert entered.wait(5)
        started = time.monotonic()
        with pytest.raises(RequestStopped) as stopped:
            engine.sync_session(session_id=SID, embed=False, _deadline=time.monotonic() + 0.3)
        assert stopped.value.reason == "request_deadline" and time.monotonic() - started < 3
        waiter = pool.submit(engine.sync_session, session_id=SID, embed=False)
        time.sleep(0.2)
        cancel.set()
        release.set()
        with pytest.raises(RequestStopped):
            leader.result(10)
        info = waiter.result(10)
    assert info["ok"] and engine.store.chunk_count(SID) == 1
