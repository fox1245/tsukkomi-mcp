"""Issue #5: bounded heavy-call concurrency and allocator trim."""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from self_directing_mcp import server


@pytest.fixture(autouse=True)
def limits(monkeypatch):
    monkeypatch.setattr(server._engine.settings, "max_concurrent_calls", 2)
    monkeypatch.setattr(server, "_libc", lambda: None)


class Spy:
    def __init__(self, hold=0.15):
        self.lock, self.running, self.peak, self.started, self.hold = threading.Lock(), 0, 0, 0, hold

    def __call__(self, *args, _deadline=None, cancel_event=None, **kwargs):
        with self.lock:
            self.running += 1
            self.started += 1
            self.peak = max(self.peak, self.running)
        time.sleep(self.hold)
        with self.lock:
            self.running -= 1
        return {"ok": True}


def test_heavy_calls_honor_limit():
    spy = Spy()

    async def scenario():
        return await asyncio.gather(*(server._dispatch(spy, budget_override=10) for _ in range(6)))

    results = asyncio.run(scenario())
    assert all(r == {"ok": True} for r in results)
    assert spy.peak == 2 and spy.started == 6


def test_queued_request_past_deadline_never_starts():
    spy = Spy(hold=0.6)

    async def scenario():
        running = [asyncio.create_task(server._dispatch(spy, budget_override=10)) for _ in range(2)]
        await asyncio.sleep(0.1)
        late = await server._dispatch(spy, budget_override=0.15)
        await asyncio.gather(*running)
        return late

    late = asyncio.run(scenario())
    assert late["error"] == "request_deadline" and late["verdict"] == "unknown"
    assert spy.started == 2


def test_light_calls_bypass_gate():
    spy, light = Spy(hold=0.5), Spy(hold=0)

    async def scenario():
        running = [asyncio.create_task(server._dispatch(spy, budget_override=10)) for _ in range(2)]
        await asyncio.sleep(0.1)
        began = time.monotonic()
        result = await server._dispatch(light, heavy=False, budget_override=10)
        elapsed = time.monotonic() - began
        await asyncio.gather(*running)
        return result, elapsed

    result, elapsed = asyncio.run(scenario())
    assert result == {"ok": True} and elapsed < 0.3


def test_slot_held_until_abandoned_worker_finishes():
    spy = Spy(hold=0.5)

    async def scenario():
        first = await server._dispatch(spy, budget_override=0.1)
        assert first["error"] == "request_deadline"
        return await asyncio.gather(*(server._dispatch(spy, budget_override=10) for _ in range(3)))

    asyncio.run(scenario())
    assert spy.peak <= 2


def _trim_env(monkeypatch, platform):
    calls = []
    monkeypatch.setattr(server, "TRIM_MIN_CALL_SEC", 0)
    monkeypatch.setattr(server, "TRIM_MIN_INTERVAL_SEC", 0)
    monkeypatch.setattr(server, "_last_trim", 0.0)
    monkeypatch.setattr(server, "_busy_since_trim", 0.0)
    monkeypatch.setattr(server.sys, "platform", platform)

    class Libc:
        def malloc_trim(self, n):
            calls.append(n)
            return 1

    monkeypatch.setattr(server, "_libc", lambda: Libc())
    return calls


def test_malloc_trim_after_heavy_call_on_linux(monkeypatch):
    calls = _trim_env(monkeypatch, "linux")

    async def scenario():
        await server._dispatch(Spy(hold=0), budget_override=10)
        await asyncio.sleep(0.2)

    asyncio.run(scenario())
    assert calls == [0]


def test_malloc_trim_skipped_off_linux_and_for_light_calls(monkeypatch):
    for platform in ("win32", "darwin"):
        calls = _trim_env(monkeypatch, platform)
        asyncio.run(server._dispatch(Spy(hold=0), budget_override=10))
        assert calls == []
    calls = _trim_env(monkeypatch, "linux")

    async def scenario():
        await server._dispatch(Spy(hold=0), heavy=False, budget_override=10)
        await asyncio.sleep(0.2)

    asyncio.run(scenario())
    assert calls == []


def test_many_short_heavy_calls_trim_once_their_total_reaches_the_threshold(monkeypatch):
    # Each call is far below the threshold on its own; hook-driven servers mostly see calls like these.
    calls = _trim_env(monkeypatch, "linux")
    monkeypatch.setattr(server, "TRIM_MIN_CALL_SEC", 0.1)

    async def scenario():
        for _ in range(3):
            await server._dispatch(Spy(hold=0.02), budget_override=10)
        await asyncio.sleep(0.1)
        assert calls == []                         # 0.06 s of work so far
        for _ in range(3):
            await server._dispatch(Spy(hold=0.02), budget_override=10)
        await asyncio.sleep(0.2)

    asyncio.run(scenario())
    assert calls == [0]
    assert server._busy_since_trim < 0.1          # the total restarts after a trim


def test_trim_interval_still_bounds_how_often_short_calls_trim(monkeypatch):
    calls = _trim_env(monkeypatch, "linux")
    monkeypatch.setattr(server, "TRIM_MIN_CALL_SEC", 0.01)
    monkeypatch.setattr(server, "TRIM_MIN_INTERVAL_SEC", 60)

    async def scenario():
        for _ in range(5):
            await server._dispatch(Spy(hold=0.02), budget_override=10)
        await asyncio.sleep(0.2)

    asyncio.run(scenario())
    assert calls == [0]


def test_malloc_trim_failure_is_ignored(monkeypatch):
    _trim_env(monkeypatch, "linux")

    class Broken:
        def malloc_trim(self, n):
            raise OSError("boom")

    monkeypatch.setattr(server, "_libc", lambda: Broken())
    server._trim_heap()
    monkeypatch.setattr(server, "_libc", lambda: object())
    server._trim_heap()
