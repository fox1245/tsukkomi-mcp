"""An AGY index kept current by appends, syncs and hooks equals a fresh full reindex of the same files.

Hooks used to force a full reparse of the transcript at every receipt change; now an append is indexed from the
cursor and a reparse happens only when a receipt change can alter an event that is already indexed. The
only acceptable difference is time, so these tests build the same session two ways:

* the *incremental* index: transcript appends (planner steps with tool calls, results with several statuses, typed
  IDE results, context and unsupported records, truncated fields with the sibling ``transcript_full.jsonl``
  appearing early, late or never) interleaved at random with PreToolUse / PostToolUse receipts (matching,
  mismatched, repeated, late, for a step that already has a result), syncs, checks and audits;
* a *fresh* index of the final transcript, built from a copy of the final receipts.

Every chunk (id, text, meta incl. ``linkage``, ``legacy_unlinked``, truncation and fallback fields, byte ranges),
the AGY evidence stats, the coverage and the audit verdicts must be identical.

One thing is deliberately not generated: a truncated field whose fallback to ``transcript_full.jsonl`` fails *for good*
while the full log keeps growing (a step missing from it that is below its highest step). That failure records the size
of the full log at the moment it was evaluated (``fallback.full_size``, part of the event id), and nothing re-evaluates it
because the failure is permanent; so an index kept by plain appends, hooks or not, differs from a fresh reindex in that
one field once the full log has grown. Before this change every hook happened to reparse and refresh it. Failures that
can be retried (the full log missing, or not yet at that step) and every success are deterministic and generated.

Two independent references guard what the module under test could get wrong together with itself: the receipts a
hook records are compared with the old whole-file algorithm (``reference_*`` below, copied from the previous
implementation), and the set of results the index links to a call is compared with the anchor rule restated on raw
bytes (plain SHA-256 of the file prefix).
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import sqlite3

import pytest

from self_directing_mcp import agy_hooks, agy_receipts
from self_directing_mcp.agy import RESULT_TYPES
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine

SID = "44444444-4444-4444-8444-444444444444"
CONTRACTS = [
    {"id": "no-wipe", "type": "must_not", "scope": "tool_call", "provider": "agy", "regex": r"rm\s+-rf"},
    {"id": "tests-first", "type": "must", "scope": "tool_call", "provider": "agy", "regex": "npm test",
     "before_regex": "deploy", "requires_success": True},
    {"id": "lint-ran", "type": "must", "scope": "tool_call", "provider": "agy", "regex": "eslint"},
]
COMPLETED = ("DONE", "ERROR", "INVALID", "CANCELED")


# --- the previous whole-file algorithm, as the independent reference --------------------------------------------

def reference_lines(path):
    data = path.read_bytes()
    position = 0
    while True:
        newline = data.find(b"\n", position)
        if newline < 0:
            return
        yield position, newline + 1, data[position:newline + 1].decode("utf-8", errors="surrogateescape")
        position = newline + 1


def reference_prefix(path, step):
    """The old ``_prefix``: parse from byte 0 until the first record numbered ``step`` or more."""
    offset = 0
    for _start, end, raw in reference_lines(path):
        row = json.loads(raw.lstrip("\x00"))
        index = row.get("step_index")
        if type(index) is not int:
            raise ValueError("cannot anchor receipt to unknown transcript layout")
        if index >= step:
            break
        offset = end
    return offset, hashlib.sha256(path.read_bytes()[:offset]).hexdigest()


def reference_completed(path):
    """The old ``_completed_results``."""
    results = {}
    for _start, _end, raw in reference_lines(path):
        record = json.loads(raw.lstrip("\x00"))
        if (record.get("type") in RESULT_TYPES and record.get("source") == "MODEL"
                and record.get("status") in COMPLETED and type(record.get("step_index")) is int):
            if record["step_index"] in results:
                raise ValueError("ambiguous duplicate native execution step")
            results[record["step_index"]] = agy_receipts.result_digest(record)
    return results


def reference_linked(row, record, start, path):
    """The anchor rule on raw bytes: same file, a receipt of this very result, and the plain SHA-256 of its prefix."""
    if (not row or row.get("preexisting_result") or row.get("path") != str(path)
            or row.get("result_hash") != agy_receipts.result_digest(record)):
        return False
    length = row.get("prefix_length", 0)
    return bool(length) and length <= start and hashlib.sha256(path.read_bytes()[:length]).hexdigest() == row.get("prefix_hash")


# --- a synthetic AGY session ------------------------------------------------------------------------------------

class World:
    def __init__(self, tmp_path, seed):
        self.rng = random.Random(seed)
        self.tmp = tmp_path
        self.app = tmp_path / "app"
        self.path = self.app / "brain" / SID / ".system_generated" / "logs" / "transcript.jsonl"
        self.full = self.path.parent / "transcript_full.jsonl"
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b"")
        self.index = tmp_path / "inc"
        self.engine = self.make_engine(self.index)
        self.engine.upsert_contracts(CONTRACTS)
        self.step = 0
        self.pending = []     # calls the planner issued that have no result record yet: dicts
        self.issued = []      # every call, for hooks that arrive late or twice
        self.partial = b""    # an unfinished record at the end of the file
        self.full_queue = []  # sibling records not written yet, in step order
        self.results = set()  # steps that already have a record
        self.number = 0       # operations so far
        self.hooked = False   # whether a PreToolUse has been seen
        self.hooks_from = self.rng.choice([0, 0, 12, 30])  # hooks installed part-way through a conversation

    def make_engine(self, index):
        return SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=index, agy_app_data_dirs=[self.app],
                                         openrouter_api_key_file=None))

    # -- the transcript --
    def append(self, record, *, newline=True):
        with self.path.open("ab") as stream:
            if self.partial:  # finish the record a previous step left open
                stream.write(self.partial)
                self.partial = b""
            stream.write(json.dumps(record, separators=(",", ":")).encode() + (b"\n" if newline else b""))

    def stamp(self, record):
        record.setdefault("created_at", "2026-10-01T00:00:00Z")
        return record

    def user(self):
        self.append(self.stamp({"step_index": self.step, "source": "USER_EXPLICIT", "type": "USER_INPUT",
                                "status": "DONE", "content": f"<USER_REQUEST>task {self.step}</USER_REQUEST>"}))
        self.step += 1

    def context(self):
        kind = self.rng.choice(["CHECKPOINT", "EPHEMERAL_MESSAGE", "SYSTEM_MESSAGE", "ERROR_MESSAGE",
                                "CONVERSATION_HISTORY", "FUTURE_STEP"])
        source = {"EPHEMERAL_MESSAGE": "SYSTEM_SDK", "FUTURE_STEP": "MODEL"}.get(kind, "SYSTEM")
        self.append(self.stamp({"step_index": self.step, "source": source, "type": kind, "status": "DONE",
                                "content": f"context {self.step}"}))
        self.step += 1

    def drain(self):
        """Every call issued so far gets its result record before the next planner step, as a host writes them."""
        while self.pending:
            self.result()

    def planner(self):
        self.drain()
        commands = self.rng.choice([["npm test"], ["deploy now"], ["eslint src"], ["ls", "pwd"], ["echo hi", "npm test", "ls"],
                                    ["rm -rf build"]])
        calls = [{"name": self.rng.choice(["run_command", "run_command", "view_file", "mcp_tool"]),
                  "args": {"CommandLine": command, "Cwd": "/synthetic"}} for command in commands]
        record = self.stamp({"step_index": self.step, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE",
                             "content": self.rng.choice(["", f"plan {self.step}", f"plan {self.step} " + "x" * 80]),
                             "tool_calls": [{"name": call["name"], "args": {key: json.dumps(value)
                                                                           for key, value in call["args"].items()}}
                                            for call in calls]})
        sibling = None
        mode = self.rng.choice(["none", "none", "content", "calls"])
        if mode != "none":
            sibling = dict(record)
            if mode == "content":
                sibling["content"] = record["content"] + " " + "FULL-TEXT " * 30
                record["truncated_fields"] = ["content"]
                record["content"] = record["content"][:20] or "plan"
                sibling["content"] = record["content"] + sibling["content"][len(record["content"]):]
            else:
                record["truncated_fields"] = ["tool_calls"]
                sibling["tool_calls"] = [{"name": call["name"], "args": call["args"]} for call in calls] + [
                    {"name": "run_command", "args": {"CommandLine": "extra hidden call"}}]
                record["tool_calls"] = record["tool_calls"][:max(1, len(calls) - 1)]
        if sibling is not None:
            self.full_queue.append({k: v for k, v in sibling.items() if k != "truncated_fields"})
            if self.rng.random() < 0.5:
                self.flush_full()
        self.append(record)
        first = self.step + 1
        self.step = first + len(calls)
        for offset, call in enumerate(calls):
            issued = {"step": first + offset, "name": call["name"], "args": dict(call["args"]), "status": None}
            self.pending.append(issued)
            self.issued.append(issued)

    def flush_full(self):
        if self.full_queue:
            with self.full.open("ab") as stream:
                for record in self.full_queue:
                    stream.write(json.dumps(record, separators=(",", ":")).encode() + b"\n")
            self.full_queue = []

    def result(self):
        if not self.pending:
            return
        call = self.pending.pop(0)
        if call["status"] is None and self.rng.random() < 0.06:
            return  # a call whose result never reaches the transcript (cancelled upstream): a hole in the steps
        kind = self.rng.choice(["GENERIC", "GENERIC", "RUN_COMMAND", "VIEW_FILE", "MCP_TOOL"])
        status = self.rng.choices(["DONE", "ERROR", "RUNNING", "INVALID", "CANCELED", "WEIRD"],
                                  [70, 8, 8, 3, 3, 2])[0]
        code = self.rng.choice([0, 0, 0, 1])
        content = (f"Created At: t\nCompleted At: t\n\nThe command exited with code {code}.\nOutput:\nout {call['step']}\n"
                   if kind == "GENERIC" else f"typed {call['step']}")
        record = self.stamp({"step_index": call["step"], "source": "MODEL", "type": kind, "status": status,
                             "content": content})
        if kind == "RUN_COMMAND":
            record["exit_code"] = code
        if status == "ERROR" and self.rng.random() < 0.5:
            record["error"] = "tool call denied by pre-tool hook: policy"
        if self.rng.random() < 0.12 and kind != "RUN_COMMAND" and status not in ("RUNNING", "WEIRD"):
            # Cut in the display log; the full log holds the whole text (it may arrive later, or not at all).
            self.full_queue.append(dict(record, content=content + " FULL-OUTPUT" * 20))
            record["truncated_fields"] = ["content"]
        if status == "RUNNING":
            self.pending.insert(0, dict(call, status="running"))  # it finishes later with a second record
        self.append(record)
        self.results.add(call["step"])

    def odd(self):
        what = self.rng.choice(["layout", "partial"])
        if what == "layout" and self.rng.random() < 0.5:
            self.append({"source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE", "content": "no step"})
        else:
            record = json.dumps({"step_index": self.step, "source": "SYSTEM", "type": "CHECKPOINT", "status": "DONE",
                                 "content": "half"}, separators=(",", ":")).encode() + b"\n"
            self.step += 1
            with self.path.open("ab") as stream:
                if self.partial:
                    stream.write(self.partial)
                self.partial = record[len(record) // 2:]
                stream.write(record[:len(record) // 2])

    # -- hooks --
    def payload(self, call, step=None, name=None, args=None, error=None):
        body = {"conversationId": SID, "transcriptPath": str(self.path), "workspacePaths": ["/synthetic"],
                "stepIdx": call["step"] if step is None else step,
                "toolCall": {"name": name or call["name"], "args": args if args is not None else call["args"]}}
        if error is not None:
            body["error"] = error
        return body

    def hook(self, event, payload):
        """Run one receipt operation, through the engine or the whole hook, and check it against the reference."""
        store = agy_receipts.ReceiptStore(self.index)
        before = dict(store.read_capture(SID)[0])
        step = payload["stepIdx"]
        try:
            expected = self.expected_row(event, payload, before.get(step))
        except ValueError:
            expected = ValueError
        whole_hook = self.rng.random() < 0.4
        failed = None
        if whole_hook:
            assert isinstance(agy_hooks.run_hook(event, payload, engine=self.engine), dict)  # fail-open: never raises
        else:
            try:
                self.engine.record_agy_hook(payload, event)
            except Exception as exc:  # noqa: BLE001 - compared with the reference below
                failed = exc
        after = dict(store.read_capture(SID)[0])
        if expected is ValueError:
            assert after.get(step) == before.get(step)  # nothing is recorded for a layout that cannot be read
            assert failed is None or isinstance(failed, ValueError)
            return
        assert failed is None, failed
        actual = json.loads(after[step])
        recorded_id, expected_id = actual.pop("receipt_id", None), expected.pop("receipt_id", None)
        assert (recorded_id == expected_id) if event == "PostToolUse" else bool(recorded_id)
        if whole_hook and expected_id and not expected.get("result_hash"):
            # The sync that follows the receipt binds a receipt whose result is in the transcript.
            bound = reference_completed(self.path).get(step)
            if bound is not None:
                expected["result_hash"] = bound
        assert actual == expected

    def expected_row(self, event, payload, original):
        """What the old implementation would have stored for this hook, from the file as it is now."""
        action = json.dumps(payload["toolCall"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        step = payload["stepIdx"]
        if event == "PreToolUse":
            length, digest = reference_prefix(self.path, step)
            return {"receipt_id": "x", "path": str(self.path), "action": action, "prefix_length": length,
                    "prefix_hash": digest, "paired": False, "step_index": step, "session_id": SID,
                    "preexisting_result": step in reference_completed(self.path)}
        row = json.loads(original) if original else {}
        error = payload.get("error", "")
        paired = (row.get("action") == action and row.get("path") == str(self.path) and row.get("paired") is False
                  and not row.get("preexisting_result") and isinstance(error, str))
        row.update(paired=paired, post_error=error, post_action=action, post_path=str(self.path))
        if paired and not row.get("result_hash"):
            observed = reference_completed(self.path).get(step)
            if observed is not None:
                row["result_hash"] = observed
        return row

    def recent(self):
        """A call a hook may concern: usually a recent one, now and then any (hooks arrive late, or are replayed)."""
        return self.rng.choice(self.issued if self.rng.random() < 0.15 else self.issued[-4:])

    def pre(self):
        if not self.issued or self.number < self.hooks_from:
            return
        call = self.recent()
        variant = self.rng.choice(["plain", "plain", "plain", "again", "wrong_step", "wrong_args", "stray"])
        if not self.hooked and self.rng.random() < 0.5:
            variant = "stray"  # the first receipt of a conversation that has results below it already
        self.hooked = True
        if variant == "stray":
            # A receipt for any earlier step: it may become the first receipt of the conversation, below results
            # that are already indexed as predating the receipts.
            self.hook("PreToolUse", self.payload(call, step=self.rng.randrange(0, max(1, self.step))))
        elif variant == "wrong_step":
            self.hook("PreToolUse", self.payload(call, step=max(0, call["step"] + self.rng.choice([-1, 1, 3]))))
        elif variant == "wrong_args":
            self.hook("PreToolUse", self.payload(call, args={"CommandLine": "something else"}))
        else:
            self.hook("PreToolUse", self.payload(call))
            if variant == "again":
                self.hook("PreToolUse", self.payload(call))

    def post(self):
        if not self.issued or self.number < self.hooks_from:
            return
        call = self.recent()
        variant = self.rng.choice(["plain", "plain", "plain", "error", "twice", "wrong_args"])
        if variant == "wrong_args":
            self.hook("PostToolUse", self.payload(call, args={"CommandLine": "other"}, error=""))
        elif variant == "error":
            self.hook("PostToolUse", self.payload(call, error="runtime failure"))
        else:
            self.hook("PostToolUse", self.payload(call, error=""))
            if variant == "twice":
                self.hook("PostToolUse", self.payload(call, error=""))

    # -- reads that sync --
    def assert_linkage(self):
        """Right after a sync the index tells, for every result record, what the receipts of this moment say."""
        receipts = {step: json.loads(data) for step, data in
                    agy_receipts.ReceiptStore(self.index).read_capture(SID)[0]}
        first = min(receipts) if receipts else None
        chunks = self.engine.store.list_chunks(SID, "agy")
        by_start = {}
        for chunk in chunks:
            by_start.setdefault(chunk.byte_start, []).append(chunk)
        for start, _end, raw in reference_lines(self.path):
            record = json.loads(raw)
            status = record.get("status")
            if (record.get("type") not in RESULT_TYPES or record.get("source") != "MODEL" or
                    status not in (None, "DONE", "RUNNING", "ERROR", "INVALID", "CANCELED") or start not in by_start):
                continue
            row = receipts.get(record["step_index"])
            anchored = reference_linked(row, record, start, self.path)
            denied = (anchored and not row.get("paired") and status == "ERROR"
                      and str(record.get("error", "")).startswith("tool call denied by pre-tool hook: "))
            here = by_start[start]
            if denied:
                assert any(c.meta.get("evidence_origin") == "host_rejected_proposal" for c in here), start
                continue
            result = next(c for c in here if c.kind == "tool_result")
            if anchored and row.get("paired"):
                expected = "host_step_receipt"
            elif status == "RUNNING":
                expected = "pending"
            elif first is None or record["step_index"] < first:
                expected = "legacy_unlinked"
            else:
                expected = "native_log_omits_call_id"
            assert result.meta["linkage"] == expected, (start, record["step_index"], expected, result.meta["linkage"])

    def sync(self):
        self.engine.sync_session(session_id=SID, provider="agy", path=str(self.path), embed=False)
        self.assert_linkage()

    def check(self):
        self.engine.check_action(SID, {"tool_name": "run_command", "arguments": {"CommandLine": "deploy now"}},
                                 provider="agy", path=str(self.path), verify="fast")
        self.assert_linkage()

    def audit(self):
        self.engine.audit_session(SID, provider="agy", path=str(self.path))
        self.assert_linkage()

    OPS = [("user", 2), ("context", 3), ("planner", 8), ("result", 9), ("pre", 8), ("post", 8), ("sync", 5), ("check", 3),
           ("audit", 1), ("flush_full", 2), ("odd", 1)]

    def step_once(self):
        name = self.rng.choices([op for op, _ in self.OPS], [weight for _, weight in self.OPS])[0]
        getattr(self, name)()
        self.number += 1


def snapshot(engine, world):
    """Everything the index says about the session, minus what is only a publication timestamp."""
    info = engine.sync_session(session_id=SID, provider="agy", path=str(world.path), embed=False)
    coverage = {key: value for key, value in info["coverage"].items() if key != "synced_at"}
    chunks = sorted(c.model_dump_json() for c in engine.store.list_chunks(SID, "agy"))
    stats = engine.store.agy_evidence_stats(SID)
    audit = engine.audit_session(SID, provider="agy", path=str(world.path))
    verdict = {key: audit.get(key) for key in ("verdict", "findings", "notes")}
    verdict["coverage"] = {key: value for key, value in audit["coverage"].items() if key != "synced_at"}
    assert "incremental_mismatch" not in (audit.get("incremental") or {})
    return {"chunks": chunks, "stats": stats, "coverage": coverage, "audit": verdict}


def fresh_snapshot(world, name):
    """The same final files, indexed from nothing: a copy of the receipts as they now stand, no ledger, no cursor."""
    index = world.tmp / name
    index.mkdir()
    shutil.copy(world.index / "agy-receipts.sqlite", index / "agy-receipts.sqlite")
    with sqlite3.connect(index / "agy-receipts.sqlite") as conn:
        conn.execute("DELETE FROM scan_state")
        conn.execute("DELETE FROM scan_results")
        conn.execute("DELETE FROM indexed")
    engine = world.make_engine(index)
    try:
        engine.upsert_contracts(CONTRACTS)
        return snapshot(engine, world), engine.store.list_chunks(SID, "agy")
    finally:
        engine.close()


def linked_starts(world, receipts):
    """Where a result record is tied to its call, by the anchor rule on raw bytes."""
    rows = {step: json.loads(data) for step, data in receipts}
    found = set()
    for start, _end, raw in reference_lines(world.path):
        record = json.loads(raw)
        if (type(record.get("step_index")) is int and record.get("type") in RESULT_TYPES
                and record.get("source") == "MODEL" and record.get("status") in (None, "DONE", "RUNNING", "ERROR",
                                                                                  "INVALID", "CANCELED")):
            row = rows.get(record["step_index"])
            if row and row.get("paired") and reference_linked(row, record, start, world.path):
                found.add(start)
    return found


def assert_bindings(world, receipts):
    """After a sync, every receipt of this file holds the digest of its completed result, and no other."""
    completed = reference_completed(world.path)
    for step, data in receipts:
        row = json.loads(data)
        if row.get("receipt_id") and row.get("path") == str(world.path):
            assert row.get("result_hash") == completed.get(step), (step, row)


SEEDS = list(range(20)) + [101, 2024, 31337]


@pytest.fixture(autouse=True)
def no_fsync(monkeypatch):
    """Thousands of small commits per case: without fsync the cases take a tenth of the time, and nothing here
    tests durability."""
    real = sqlite3.connect

    def connect(*args, **kwargs):
        conn = real(*args, **kwargs)
        if not kwargs.get("uri"):
            conn.execute("PRAGMA synchronous=OFF")
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect)


@pytest.mark.parametrize("seed", SEEDS)
def test_incremental_index_equals_a_fresh_reindex(tmp_path, seed):
    world = World(tmp_path, seed)
    try:
        steps = 70 + seed % 5 * 10
        marks = {steps // 3, 2 * steps // 3}
        for number in range(steps):
            world.step_once()
            if number in marks:
                # Mid-session too: the cursor, the ledger and the dependencies must agree with a rebuild at any point.
                world.sync()
                middle, _ = fresh_snapshot(world, f"mid-{number}")
                assert snapshot(world.engine, world)["chunks"] == middle["chunks"]
        world.flush_full()
        world.sync()
        final = snapshot(world.engine, world)
        fresh, chunks = fresh_snapshot(world, "fresh")
        assert final["chunks"] == fresh["chunks"]
        assert final["stats"] == fresh["stats"]
        assert final["coverage"] == fresh["coverage"]
        assert final["audit"] == fresh["audit"]
        receipts = agy_receipts.ReceiptStore(world.index).read_capture(SID)[0]
        assert_bindings(world, receipts)
        indexed = {c.byte_start for c in chunks if c.kind == "tool_result" and c.meta.get("linkage") == "host_step_receipt"}
        assert indexed == linked_starts(world, receipts)
    finally:
        world.engine.close()
