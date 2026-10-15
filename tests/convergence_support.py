"""Virtual time for the engine: what an event costs is charged in virtual seconds, so that a test can say "parsing a record
takes 1.5 ms, folding an event into the monitors 2.3 ms" and get the same calls, stops and answers on any machine.

The engine, the request control, the ingest and the incremental audit all read ``time.monotonic()`` (the deadline of a request,
the soft deadline of a slice, the room left for a comparison). ``Virtual`` replaces the ``time`` those modules see by one whose
clock moves only when work is charged, and meters the real work where it happens:

* parsing: one charge per record the Codex parser reads (the consumer of ``iter_raw_lines``, so the clock the slice budget reads
  has moved when it looks);
* committing: one charge per row, before the commit (a deadline that passes meanwhile stops it, as the real one does);
* reading events for an audit: one charge per event (``capture_audit_events``, ``capture_audit_tail``);
* folding events into the monitors (``incremental.fold``), in chunks with a cooperative check between them, as the real loop
  does;
* the full evaluator (a comparison, or an audit without monitors): one charge per event, in the same chunks.

The chunks are ``PAGE`` events, not the real 512: the prices are about 50 times the real ones, so a chunk of 512 would be 50 times
as long as a real one, and a deadline would be noticed 50 times as late.

The data and the logic are the real ones. Only the clock and its price list are not.
"""
from __future__ import annotations

import time as real_time
from dataclasses import dataclass
from pathlib import Path

from self_directing_mcp import eval_cost, engine as engine_module, request_control, shadow_worker
from self_directing_mcp.audit import incremental
from self_directing_mcp.codex import parse as codex_parse
from self_directing_mcp.index import ingest
from self_directing_mcp.request_control import RequestStopped, check_request


PAGE = 10                # events between two cooperative checks (the real loops check every 512, at 1/50 of the price)


class _Time:
    """The ``time`` module as the engine sees it: ``monotonic`` and ``perf_counter`` are virtual, the rest is real."""

    def __init__(self, clock):
        self.monotonic = self.perf_counter = clock.read

    def __getattr__(self, name):
        return getattr(real_time, name)


@dataclass
class Call:
    """What one audit call did: its virtual duration, and its answer or the stop that ended it."""
    elapsed: float
    payload: dict | None
    stopped: str | None

    @property
    def complete(self) -> bool:
        return self.payload is not None and self.payload["coverage"]["complete"] and not self.payload["coverage"]["issues"]


class Virtual:
    """One virtual clock and its price list (seconds of virtual time per event). Install it before the engines are made."""

    def __init__(self, monkeypatch, *, parse=1.5e-3, commit=0.5e-3, load=0.35e-3, fold=0.5e-3, full=0.55e-3,
                 scale_model=True):
        self.now = real_time.monotonic()
        self.price = dict(parse=parse, commit=commit, load=load, fold=fold, full=full)
        self.spent = dict.fromkeys(self.price, 0.0)
        self.counts = dict.fromkeys(self.price, 0)
        proxy = _Time(self)
        for module in (request_control, engine_module, ingest, incremental, shadow_worker):
            monkeypatch.setattr(module, "time", proxy)
        if scale_model:
            # The cost model's defaults are real seconds (40 microseconds to parse and commit an event). The prices above are
            # the same figures times about 50, so that a session of 3,000 events behaves like one of 150,000: the model's
            # priors and its smallest measurement scale with them.
            factor = (parse + commit) / 40e-6
            monkeypatch.setattr(eval_cost, "DEFAULT_SLICE_SEC", eval_cost.DEFAULT_SLICE_SEC * factor)
            monkeypatch.setattr(eval_cost, "DEFAULT_EVAL_SEC", eval_cost.DEFAULT_EVAL_SEC * factor)
            monkeypatch.setattr(eval_cost, "EVAL_BASE_SEC", eval_cost.EVAL_BASE_SEC * factor)
            monkeypatch.setattr(eval_cost, "EVAL_PER_CONTRACT_SEC", eval_cost.EVAL_PER_CONTRACT_SEC * factor)
            monkeypatch.setattr(eval_cost, "MIN_SAMPLE_EVENTS", max(10, int(eval_cost.MIN_SAMPLE_EVENTS / factor)))
            monkeypatch.setattr(incremental, "SHADOW_SECONDS_PER_EVENT", incremental.SHADOW_SECONDS_PER_EVENT * factor)
        eval_cost.COSTS.reset()
        self._meter(monkeypatch)

    # --- the clock ---
    def read(self) -> float:
        return self.now

    def charge(self, kind: str, events: int) -> None:
        if events > 0:
            seconds = self.price[kind] * events
            self.now += seconds
            self.spent[kind] += seconds
            self.counts[kind] += events

    def work(self, kind: str, events: int) -> None:
        """Charge ``events`` in pages with a cooperative check after each, like a loop that checks its request as it goes."""
        for start in range(0, events, PAGE):
            self.charge(kind, min(PAGE, events - start))
            check_request()

    # --- the meters ---
    def _meter(self, monkeypatch) -> None:
        real_lines = codex_parse.iter_raw_lines

        def lines(path, *, start_byte=0):
            for item in real_lines(path, start_byte=start_byte):
                self.charge("parse", 1)
                yield item

        monkeypatch.setattr(codex_parse, "iter_raw_lines", lines)

        real_commit = ingest.SessionStore.commit_events

        def commit(store, events, **kwargs):
            rows = kwargs.get("prepared_rows")
            self.charge("commit", rows.count if rows is not None else len(events))
            return real_commit(store, events, **kwargs)

        monkeypatch.setattr(ingest.SessionStore, "commit_events", commit)

        real_events = ingest.SessionStore.capture_audit_events
        real_tail = ingest.SessionStore.capture_audit_tail

        def events(store, *args, **kwargs):
            snapshot = real_events(store, *args, **kwargs)
            self.work("load", len(snapshot.events))
            return snapshot

        def tail(store, *args, **kwargs):
            rows = real_tail(store, *args, **kwargs)
            self.work("load", len(rows))
            return rows

        monkeypatch.setattr(ingest.SessionStore, "capture_audit_events", events)
        monkeypatch.setattr(ingest.SessionStore, "capture_audit_tail", tail)

        real_fold = incremental.fold

        def fold(rows, monitors, facts, pools, progress=None):
            for start in range(0, len(rows), PAGE):
                chunk = rows[start:start + PAGE]
                self.charge("fold", len(chunk))
                check_request()
                if progress is not None:
                    progress[0] = start
                real_fold(chunk, monitors, facts, pools)
            if progress is not None:
                progress[0] = len(rows)

        monkeypatch.setattr(incremental, "fold", fold)

        real_run = engine_module.run_audit

        def run_audit(**kwargs):
            store = kwargs.get("store")
            if kwargs.get("plan") is None and isinstance(store, ingest.AuditEventSnapshot):
                self.work("full", len(store.events))
            return real_run(**kwargs)

        monkeypatch.setattr(engine_module, "run_audit", run_audit)

    # --- driving ---
    def audit(self, engine, world, *, budget, action=None) -> Call:
        """One ``audit_session`` (or ``check_action`` with ``action``) with a virtual budget of ``budget`` seconds."""
        began = self.now
        kwargs = dict(provider=world.provider, path=str(world.path), _deadline=self.now + budget)
        try:
            payload = (engine.audit_session(world.sid, **kwargs) if action is None
                       else engine.check_action(world.sid, action, **kwargs))
            return Call(self.now - began, payload, None)
        except RequestStopped as stopped:
            return Call(self.now - began, None, stopped.reason)

    def converge(self, engine, world, *, budget, action=None, limit=30):
        """Repeat the call until the answer is complete or ``limit`` calls were made; returns every call."""
        calls = []
        while len(calls) < limit:
            calls.append(self.audit(engine, world, budget=budget, action=action))
            if calls[-1].complete:
                break
        return calls


def brief(payload: dict) -> dict:
    """What a complete answer must agree on whatever the path that led to it: verdict, findings with their evidence, coverage."""
    return {"verdict": payload["verdict"],
            "findings": [(f["contract_id"], f["verdict"], f["reason"], [e["chunk_id"] for e in f.get("evidence", [])])
                         for f in payload["findings"]],
            "complete": payload["coverage"]["complete"], "issues": list(payload["coverage"]["issues"]),
            "byte_offset": payload["coverage"].get("byte_offset")}


# --- a synthetic session and the engines that audit it ---------------------------------------------------------------------

import json
import random

from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine

SID = "11111111-2222-3333-4444-555555555555"
WORDS = ("alpha beta gamma delta epsilon zeta theta iota kappa lambda sigma omega build test deploy review lint format "
         "merge plan note check retry cache index stream record slice commit").split()
SPECIAL = ["pytest -q", "make build", "deploy now", "git status", "ls -la"]

# Five contracts that all need history: a prohibition that is broken now and then, one that never is, an obligation that
# needs a successful call, and one on the assistant's messages.
CONTRACTS = [
    {"id": "no-wipe", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"},
    {"id": "no-pipe-sh", "type": "must_not", "scope": "tool_call", "regex": r"curl\s.*\|\s*sh"},
    {"id": "build-passed", "type": "must", "scope": "tool_call", "regex": "make build", "requires_success": True},
    {"id": "ack-finished", "type": "must", "scope": "message", "regex": "FINISHED", "roles": ["assistant"]},
    {"id": "tests-before-deploy", "type": "must", "scope": "tool_call", "regex": "pytest", "before_regex": "deploy now"},
]
DEPLOY = {"tool_name": "shell", "arguments": {"command": "deploy now"}}      # a proposed action that the order rule gates


class Session:
    """A Codex rollout of about ``events`` events (messages, tool calls with their results, agent messages) with no record the
    parser cannot read, so that a finished sync is a complete one. ``rm -rf`` appears a few times, ``FINISHED`` once at the end.
    ``append`` adds more of the same, ending in another ``FINISHED``."""

    def __init__(self, tmp_path: Path, events: int, seed: int = 7):
        self.provider, self.sid = "codex", SID
        self.tmp = Path(tmp_path)
        self.root = self.tmp / "sessions"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl"
        self.rng, self.serial = random.Random(seed), 0
        self.engines: list[SelfDirectEngine] = []
        self.path.write_text("\n".join(self._lines(events, header=True)) + "\n", encoding="utf-8")

    def append(self, events: int) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write("\n".join(self._lines(events, header=False)) + "\n")

    def _lines(self, events: int, *, header: bool) -> list[str]:
        rng = self.rng

        def say(low, high):
            return " ".join(rng.choice(WORDS) for _ in range(rng.randint(low, high)))

        lines = [json.dumps({"type": "session_meta", "payload": {"id": SID}})] if header else []
        produced = len(lines)
        while produced < events:
            pick = rng.random()
            if pick < 0.30:
                lines.append(json.dumps({"type": "response_item", "payload": {
                    "type": "message", "role": rng.choice(["user", "assistant"]), "content": say(14, 34)}}))
                produced += 1
            elif pick < 0.76:
                self.serial += 1
                roll = rng.random()
                command = (rng.choice(SPECIAL) if roll < 0.04 else "rm -rf /tmp/scratch" if roll < 0.0415
                           else "echo " + say(3, 10))
                lines.append(json.dumps({"type": "response_item", "payload": {
                    "type": "function_call", "name": "shell", "call_id": f"call-{self.serial}",
                    "arguments": {"command": command}}}))
                lines.append(json.dumps({"type": "response_item", "payload": {
                    "type": "function_call_output", "call_id": f"call-{self.serial}",
                    "output": json.dumps({"exit_code": 0 if rng.random() < 0.9 else 1, "output": say(16, 44)})}}))
                produced += 2
            else:
                lines.append(json.dumps({"type": "event_msg", "payload": {"type": "agent_message", "message": say(10, 30)}}))
                produced += 1
        lines.append(json.dumps({"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                                                      "content": "FINISHED"}}))
        return lines

    def engine(self, name: str, *, policy=None, contracts=None, defer_derived=False, **settings) -> SelfDirectEngine:
        """An engine of its own (local-only, an index directory of its own) whose slice policy is ``policy`` (a test's clock and
        minimum progress). Adaptive comparisons, as shipped, and monitor states for every session size."""
        options = dict(_env_file=None, local_only=True, openrouter_api_key_file=None, codex_sessions_dir=self.root,
                       index_dir=self.tmp / f"index-{name}", audit_shadow_every="adaptive", audit_incremental_min_events=0)
        options.update(settings)
        engine = SelfDirectEngine(Settings(**options), defer_derived=defer_derived)
        if policy is not None:
            engine.slice_policy = policy
        engine.upsert_contracts([dict(c, provider="codex") for c in (CONTRACTS if contracts is None else contracts)])
        self.engines.append(engine)
        return engine

    def close(self) -> None:
        for engine in self.engines:
            engine.close()
