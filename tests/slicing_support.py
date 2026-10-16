"""Synthetic sessions and helpers for the slice-commit tests (roundtable 8). Everything here is synthetic.

``World`` builds one session of one provider (Codex, OMP, AGY, Claude Code or Grok Bot) as a list of raw lines: ordinary
records, tool call/result pairs, parse errors, unsupported records, blank lines, a line with invalid UTF-8 and (when asked
for) a final unfinished line. It makes engines for it with an index of their own, so the same session can be indexed
once in one piece and once in slices and the two compared.
"""
from __future__ import annotations

import json
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from self_directing_mcp import journal
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index import ingest
from self_directing_mcp.index.ingest import SlicePolicy
from test_claude_hooks import SID as CLAUDE_SID, STEM, assistant, human, text, tool_result, use
from test_reliability import SID as CODEX_SID, call, message, result

OMP_SID = "session-1"
AGY_SID = "44444444-4444-4444-8444-444444444444"
GROK_SID = "00000000-0000-4000-8000-0000000000a1"
PROVIDERS = ("codex", "omp", "agy", "claude", "grokbot")
SIDS = {"codex": CODEX_SID, "omp": OMP_SID, "agy": AGY_SID, "claude": CLAUDE_SID, "grokbot": GROK_SID}

# Two contracts every provider can be audited with: a prohibition some commands break, and an obligation.
CONTRACTS = [
    {"id": "no-wipe", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"},
    {"id": "ack", "type": "must", "scope": "message", "regex": "FINISHED", "roles": ["assistant"]},
]

WORDS = ("alpha beta gamma delta build test deploy review lint format merge plan note check retry cache index "
         "stream record slice commit").split()
COMMANDS = ("echo hello", "ls -la", "pytest -q", "git status", "rm -rf build", "deploy now", "npm test", "cat notes.txt",
            "grep -r alpha .", "make all")


def line(record) -> bytes:
    return (json.dumps(record, separators=(",", ":")) + "\n").encode()


def noise(rng) -> bytes:
    """A line no parser can use: invalid JSON, a JSON array, a blank, a blank with spaces, invalid UTF-8."""
    return rng.choice([b'{"type": broken\n', b"[1, 2, 3]\n", b"\n", b"   \n", b'{"type":"x","payload":"caf\xe9"}\n',
                       b'"just a string"\n'])


def sentence(rng, low=3, high=14) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(low, high)))


# --- per-provider line builders: ``units(rng, count)`` yields lists of lines ------------------------------------------

def codex_units(rng, count, fat=1):
    say = lambda low=3, high=14: sentence(rng, low * fat, high * fat)   # noqa: E731
    yield [line({"type": "session_meta", "payload": {"id": CODEX_SID}})]
    serial = 0
    for _ in range(count):
        pick = rng.random()
        if pick < 0.22:
            yield [line(message(say(), rng.choice(["user", "assistant"])))]
        elif pick < 0.50:
            serial += 1
            cid = f"call-{serial}"
            yield [line(call(rng.choice(COMMANDS), cid)), line(result(cid, rng.choice([0, 0, 1])))]
        elif pick < 0.60:
            yield [line({"type": "event_msg", "payload": {"type": "agent_message", "message": say()}})]
        elif pick < 0.65:
            serial += 1
            yield [line({"type": "response_item", "payload": {"type": "web_search_call", "id": f"ws-{serial}",
                                                                "status": rng.choice(["completed", "in_progress", "failed"]),
                                                                "action": {"type": "search", "query": say(2, 4)}}})]
        elif pick < 0.69:
            yield [line({"type": rng.choice(["token_usage_record", "world_state"]),
                         "payload": {"response_id": f"r{serial}", "turn_id": f"t{serial}", "full": False}})]
        elif pick < 0.71:
            yield [line({"type": "compacted", "payload": {"replacement_history": [], "window_id": serial}})]
        elif pick < 0.74:
            yield [line({"type": "unheard_of_event", "payload": {"x": say(1, 3)}})]
        elif pick < 0.76:
            yield [line({"type": "session_meta", "payload": {"id": CODEX_SID}})]   # a repeated header of the same session
        else:
            yield [noise(rng)]
    yield [line(message("FINISHED", "assistant"))]


def omp_units(rng, count, *, title=False, fat=1):
    say = lambda low=3, high=14: sentence(rng, low * fat, high * fat)   # noqa: E731
    if title:
        yield [line({"type": "title", "title": "Synthetic"})]
    yield [line({"type": "session", "id": OMP_SID})]
    serial = 0
    open_calls = []
    for _ in range(count):
        pick = rng.random()
        if pick < 0.20:
            yield [line({"type": "message", "message": {"role": rng.choice(["user", "assistant"]),
                                                          "content": [{"type": "text", "text": say()}]}})]
        elif pick < 0.42:
            serial += 1
            cid = f"omp-call-{serial}"
            open_calls.append(cid)
            yield [line({"type": "message", "message": {"role": "assistant", "content": [
                {"type": "text", "text": say(2, 5)},
                {"type": "toolCall", "id": cid, "name": "bash", "arguments": {"command": rng.choice(COMMANDS)}}]}})]
        elif pick < 0.62 and open_calls:
            cid = open_calls.pop(rng.randrange(len(open_calls)))
            yield [line({"type": "message", "message": {"role": "toolResult", "toolCallId": cid, "toolName": "bash",
                                                          "isError": rng.random() < 0.2,
                                                          "content": [{"type": "text", "text": say()}]}})]
        elif pick < 0.74:
            # A host event that refers to a call; it is metadata only when that call was seen EARLIER (maybe in another slice).
            cid = f"omp-call-{rng.randint(1, max(1, serial))}"
            yield [line({"type": "custom", "customType": "tool_execution_start",
                         "data": {"toolCallId": cid, "toolName": rng.choice(["bash", "bash", "other"])}})]
        elif pick < 0.80:
            yield [line({"type": "message", "message": {"role": "assistant",
                                                          "content": [{"type": "thinking", "thinking": say()}]}})]
        elif pick < 0.84:
            yield [line({"type": rng.choice(["model_change", "thinking_level_change"]), "id": f"e{serial}"})]
        elif pick < 0.87:
            yield [line({"type": "message", "message": {"role": "user", "attribution": "agent", "steering": True,
                                                          "content": say()}})]
        elif pick < 0.89:
            yield [line({"type": "mystery_record", "id": f"m{serial}"})]
        else:
            yield [noise(rng)]
    yield [line({"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "FINISHED"}]}})]


def claude_units(rng, count, *, sub=False, sid=CLAUDE_SID, fat=1):
    say = lambda low=3, high=14: sentence(rng, low * fat, high * fat)   # noqa: E731
    serial = 0
    for _ in range(count):
        pick = rng.random()
        if pick < 0.20:
            yield [line(human(say(), sub=sub, sessionId=sid))]
        elif pick < 0.34:
            yield [line(assistant(text(say()), sub=sub, sessionId=sid))]
        elif pick < 0.62:
            serial += 1
            cid = f"toolu_{serial:06d}"
            yield [line(assistant(text(say(2, 5)), use(cid, "Bash", command=rng.choice(COMMANDS)),
                                  sub=sub, sessionId=sid)),
                   line(tool_result(cid, say(), is_error=rng.random() < 0.15, sub=sub, sessionId=sid))]
        elif pick < 0.72:
            yield [line({"type": "queue-operation", "operation": "enqueue", "timestamp": "2026-10-01T00:00:00.000Z",
                         "sessionId": sid, "content": say()})]
        elif pick < 0.78:
            yield [line({"type": "permission-mode", "permissionMode": "default", "sessionId": sid})]
        elif pick < 0.82:
            yield [line({"type": "never_seen_before", "sessionId": sid, "uuid": f"u{serial}"})]
        else:
            yield [noise(rng)]
    yield [line(assistant(text("FINISHED"), sub=sub, sessionId=sid))]


def grok_units(rng, count, fat=1):
    say = lambda low=3, high=14: sentence(rng, low * fat, high * fat)   # noqa: E731
    serial = 0
    for _ in range(count):
        pick = rng.random()
        if pick < 0.25:
            yield [line({"role": rng.choice(["user", "assistant"]),
                         "message": {"content": [{"type": "text", "text": say()}]}})]
        elif pick < 0.55:
            serial += 1
            cid = f"gc{serial}"
            yield [line({"role": "assistant", "message": {"content": [
                {"type": "text", "text": say(2, 5)},
                {"type": "tool_use", "id": cid, "name": "Shell", "input": {"command": rng.choice(COMMANDS)}}]}}),
                   line({"role": "tool", "message": {"content": [{"type": "tool_result", "tool_use_id": cid,
                                                                  "content": say()}]}})]
        elif pick < 0.62:
            yield [line({"note": say()})]                       # neither role nor message: a meta record
        elif pick < 0.66:
            yield [line({"role": "assistant", "message": {"content": [{"type": "image_url", "url": "x"}]}})]
        else:
            yield [noise(rng)]
    yield [line({"role": "assistant", "message": {"content": [{"type": "text", "text": "FINISHED"}]}})]


PARTIAL = {"codex": b'{"type":"response_item","payload":{"type":"message","role":"assistant","content":"unfinis',
           "omp": b'{"type":"message","message":{"role":"assistant","content":[{"type":"te',
           "agy": b'{"step_index":99999,"source":"MODEL","type":"PLANNER_RESPONSE","status":"DO',
           "claude": b'{"type":"assistant","uuid":"00000000-0000-4000-8000-0000000fffff","mess',
           "grokbot": b'{"role":"assistant","message":{"content":[{"type":"te'}


# --- AGY: a transcript written the way the host writes it, with hook receipts recorded as the steps happen --------------

def agy_build(world: "World", rng, count: int, *, receipts: bool = True) -> None:
    """Planner steps with run_command calls, their results, context steps, truncated fields with a sibling full log, and
    PreToolUse/PostToolUse receipts recorded through ``world.builder`` while the file grows (a receipt anchors to the file
    as it is when the hook runs, so this order matters). The receipts database is the builder's; engines copy it."""
    path, step = world.path, 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    full = path.with_name("transcript_full.jsonl")
    full_records = []
    late_hooks_off = [False]

    def put(record, raw=None):
        with path.open("ab") as stream:
            stream.write(raw if raw is not None else line(record))

    def stamp(record):
        record.setdefault("created_at", "2026-10-01T00:00:00Z")
        return record

    def hook(event, number, command, error=None):
        if not receipts or late_hooks_off[0]:
            return
        payload = {"conversationId": AGY_SID, "transcriptPath": str(path), "workspacePaths": ["/synthetic"],
                   "stepIdx": number, "toolCall": {"name": "run_command", "args": {"CommandLine": command, "Cwd": "/synthetic"}}}
        if error is not None:
            payload["error"] = error
        world.builder.record_agy_hook(payload, event)

    hooks_until = int(count * 0.7)
    for number_of_units in range(count):
        # Hooks record while the transcript is readable. A line that is not a JSON object (a blank, a torn write) makes
        # every receipt scan fail (receipts are never anchored to a layout that cannot be read), so such lines come only
        # in the last part of the session, after the receipts were bound to their results and no hook runs any more.
        late = number_of_units >= hooks_until
        if number_of_units == hooks_until and receipts:
            world.builder.sync_session(session_id=AGY_SID, provider="agy", path=str(path), embed=False)   # binds receipts
        late_hooks_off[0] = late
        if late and rng.random() < 0.15:
            put(None, raw=noise(rng))
            continue
        pick = rng.random()
        if not late and 0.22 <= pick < 0.28:
            pick = 0.5
        if pick < 0.14:
            put(stamp({"step_index": step, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE",
                       "content": f"<USER_REQUEST>{sentence(rng)}</USER_REQUEST>"}))
            step += 1
        elif pick < 0.22:
            kind = rng.choice(["CHECKPOINT", "EPHEMERAL_MESSAGE", "SYSTEM_MESSAGE", "ERROR_MESSAGE", "CONVERSATION_HISTORY"])
            put(stamp({"step_index": step, "source": "SYSTEM_SDK" if kind == "EPHEMERAL_MESSAGE" else "SYSTEM",
                       "type": kind, "status": "DONE", "content": f"context {sentence(rng, 1, 4)}"}))
            step += 1
        elif pick < 0.28:
            put(None, raw=noise(rng))
        elif pick < 0.31:
            put(stamp({"step_index": step, "source": "MODEL", "type": "FUTURE_STEP", "status": "DONE"}))
            step += 1
        elif pick < 0.34:
            body = line(stamp({"step_index": step, "source": "SYSTEM", "type": "SYSTEM_MESSAGE", "status": "DONE",
                               "content": "padded " + sentence(rng, 1, 3)}))
            put(None, raw=b"\x00" * rng.randint(3, 40) + body)   # host padding in front of a valid record
            step += 1
        else:
            commands = [rng.choice(COMMANDS) for _ in range(rng.randint(1, 3))]
            content = rng.choice(["", f"plan {sentence(rng)}", f"plan {sentence(rng)}"])
            record = stamp({"step_index": step, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE",
                            "content": content,
                            "tool_calls": [{"name": "run_command",
                                            "args": {"CommandLine": json.dumps(c), "Cwd": json.dumps("/synthetic")}}
                                           for c in commands]})
            if content and rng.random() < 0.25:     # cut in the display log; the full log holds the whole text
                sibling = dict(record, content=content + " FULL-TEXT" * 20)
                record["truncated_fields"] = ["content"]
                record["content"] = content[:8] or "plan"
                full_records.append({k: v for k, v in sibling.items() if k != "truncated_fields"})
            put(record)
            first = step + 1
            step = first + len(commands)
            for offset, command in enumerate(commands):
                number = first + offset
                status = rng.choices(["DONE", "ERROR", "RUNNING"], [80, 12, 8])[0]
                if rng.random() < 0.8 and status != "RUNNING":   # a result that never completes would stay unbound
                    hook("PreToolUse", number, command)
                code = rng.choice([0, 0, 1])
                result_record = stamp({"step_index": number, "source": "MODEL", "type": "GENERIC", "status": status,
                                       "content": (f"Created At: t\nCompleted At: t\n\nThe command exited with code "
                                                   f"{code}.\nOutput:\nout {number}\n")})
                if status == "ERROR" and rng.random() < 0.5:
                    result_record["error"] = "tool call denied by pre-tool hook: policy"
                if rng.random() < 0.1 and status == "DONE":
                    full_records.append(dict(result_record, content=result_record["content"] + "FULL-OUTPUT" * 10))
                    result_record["truncated_fields"] = ["content"]
                put(result_record)
                if rng.random() < 0.8:
                    hook("PostToolUse", number, command, error="" if code == 0 else "failed")
    put(stamp({"step_index": step, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE", "content": "FINISHED"}))
    if full_records:
        with full.open("wb") as stream:
            for record in full_records:
                stream.write(line(record))


# --- the world --------------------------------------------------------------------------------------------------------

@dataclass
class Snapshot:
    """Everything an index says about one session that must not depend on how it was synced."""
    events: list
    cursor: tuple
    blocks: tuple | None
    metrics: tuple
    versions: tuple
    agy: tuple | None
    generation: int | None
    last_commit: tuple | None
    commit_offsets: list


class World:
    """One synthetic session of one provider on disk, and the engines that index it.

    ``lines`` holds the complete records; ``partial`` is the unfinished last line (written without a newline) or None.
    """

    def __init__(self, tmp_path: Path, monkeypatch, provider: str, *, seed: int = 0, count: int = 300,
                 partial: bool = True, title: bool = False, receipts: bool = True, subagent: bool = False,
                 fat: int = 1):
        for name in ("SELF_DIRECT_SESSION_PROVIDER", "SELF_DIRECT_CONTRACTS_PATH", "SELF_DIRECT_CODEX_SESSIONS_DIR",
                     "CODEX_SESSIONS_DIR", "SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR", "SELF_DIRECT_OMP_SESSIONS_DIR"):
            monkeypatch.delenv(name, raising=False)
        self.provider, self.sid, self.tmp = provider, SIDS[provider], Path(tmp_path)
        self.rng = random.Random(seed)
        self.engines: list[SelfDirectEngine] = []
        self.root = self.tmp / "sessions"
        self.root.mkdir(parents=True, exist_ok=True)
        self.sid_for_engine = self.sid
        if provider == "codex":
            self.path = self.root / f"rollout-2026-09-09T01-00-00-{CODEX_SID}.jsonl"
            self.paths = {"codex_sessions_dir": self.root}
        elif provider == "omp":
            self.path = self.root / "session.jsonl"
            self.paths = {"omp_sessions_dir": self.root}
        elif provider == "agy":
            self.path = self.root / "brain" / AGY_SID / ".system_generated" / "logs" / "transcript.jsonl"
            self.paths = {"agy_app_data_dirs": [self.root]}
        elif provider == "claude":
            self.sid = f"{CLAUDE_SID}_{STEM}" if subagent else CLAUDE_SID
            self.path = (self.root / "-synthetic-project" / CLAUDE_SID / "subagents" / f"{STEM}.jsonl" if subagent
                         else self.root / "-synthetic-project" / f"{CLAUDE_SID}.jsonl")
            self.paths = {"claude_projects_dir": self.root}
        else:
            self.path = self.root / GROK_SID / f"{GROK_SID}.jsonl"
            self.paths = {"grokbot_transcripts_dir": str(self.root)}
        self.sid_for_engine = self.sid
        self.builder = None
        if provider == "agy":
            self.builder = self.engine("builder")
            agy_build(self, self.rng, count, receipts=receipts)
            self.lines = []
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            build = {"codex": lambda r, n: codex_units(r, n, fat), "omp": lambda r, n: omp_units(r, n, title=title, fat=fat),
                     "claude": lambda r, n: claude_units(r, n, sub=subagent, sid=CLAUDE_SID, fat=fat),
                     "grokbot": lambda r, n: grok_units(r, n, fat)}[provider]
            self.lines = [raw for unit in build(self.rng, count) for raw in unit]
            self.path.write_bytes(b"".join(self.lines))
        if partial:
            with self.path.open("ab") as stream:
                stream.write(PARTIAL[provider])

    def engine(self, name: str, **settings) -> SelfDirectEngine:
        """An engine with its own index directory ``name`` (a local-only one: no embedder is created or needed)."""
        engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=self.tmp / f"index-{name}",
                                           openrouter_api_key_file=None, **self.paths, **settings))
        self.engines.append(engine)
        return engine

    def copy_receipts(self, source: SelfDirectEngine, target: SelfDirectEngine) -> None:
        """AGY: the receipts the builder recorded, copied into another index directory before that index is used."""
        if self.provider != "agy":
            return
        Path(target.settings.index_dir).mkdir(parents=True, exist_ok=True)
        journal.copy_database(Path(source.settings.index_dir) / "agy-receipts.sqlite",
                              Path(target.settings.index_dir) / "agy-receipts.sqlite")

    def close(self) -> None:
        for engine in self.engines:
            engine.close()

    def sync_args(self) -> dict:
        return dict(session_id=self.sid, provider=self.provider, path=str(self.path), embed=False)

    def upsert_contracts(self, engine) -> None:
        engine.upsert_contracts([dict(c, provider=self.provider) for c in CONTRACTS])


def snapshot(engine: SelfDirectEngine, provider: str, sid: str) -> Snapshot:
    """The authority rows of one session, read straight from the index (timestamps left out: they are publication times)."""
    sid_key = sid.lower() if provider in ("codex", "agy") else sid
    conn = engine.store._conn
    q = lambda sql, *args: conn.execute(sql, args).fetchall()   # noqa: E731
    events = [tuple(r) for r in q("SELECT chunk_id,provider,session_id,byte_start,sub_index,data FROM session_events"
                                  " WHERE provider=? AND session_id=? ORDER BY byte_start,sub_index,chunk_id", provider, sid_key)]
    cursor = q("SELECT path,byte_offset,prefix_hash FROM session_cursors WHERE provider=? AND session_id=?", provider, sid_key)
    blocks = q("SELECT path,size,mtime_ns,block_size,covered,lines,digests,tail,root FROM source_blocks"
               " WHERE provider=? AND session_id=?", provider, sid_key)
    metrics = q("SELECT chunk_count,parse_errors,unsupported_events FROM session_metrics WHERE provider=? AND session_id=?",
                provider, sid_key)
    versions = q("SELECT parser_version FROM session_parser_versions WHERE provider=? AND session_id=?", provider, sid_key)
    agy = None
    if provider == "agy":
        agy = (q("SELECT digest FROM session_receipt_versions WHERE provider=? AND session_id=?", provider, sid_key),
               q("SELECT deps FROM session_receipt_deps WHERE provider=? AND session_id=?", provider, sid_key),
               q("SELECT stats FROM session_agy_stats WHERE session_id=?", sid_key))
        agy = tuple(tuple(tuple(r) for r in part) for part in agy)
    generation = q("SELECT generation FROM session_generations WHERE provider=? AND session_id=?", provider, sid_key)
    log = q("SELECT generation,byte_offset,prefix_hash,event_count FROM session_commit_log WHERE provider=? AND session_id=?"
            " ORDER BY byte_offset", provider, sid_key)
    return Snapshot(events=events, cursor=tuple(cursor[0]) if cursor else (), blocks=tuple(blocks[0]) if blocks else None,
                    metrics=tuple(metrics[0]) if metrics else (), versions=tuple(versions[0]) if versions else (),
                    agy=agy, generation=generation[0][0] if generation else None,
                    last_commit=tuple(log[-1]) if log else None, commit_offsets=[r[1] for r in log])


def sparse_rows(engine: SelfDirectEngine, sid: str) -> list:
    """The full-text index rows of one session (the derived index), in a stable order. The legacy format stores the text;
    the compact format stores none, so its rows carry the document's token count (what FTS5 recorded) in its place."""
    conn = engine.sparse._conn
    from self_directing_mcp.index.sparse import CompactSparseIndex
    if isinstance(engine.sparse, CompactSparseIndex):               # the compact and CJK formats store no text
        from self_directing_mcp.index.sparse import _read_varints
        return [(cid, session, kind, _read_varints(size)[0]) for cid, session, kind, size in conn.execute(
            "SELECT r.chunk_id, s.session_id, k.kind, d.sz FROM chunk_rows r JOIN sparse_sessions s USING (session_no) "
            "JOIN sparse_kinds k USING (kind_no) JOIN chunks_fts_docsize d ON d.id = r.rowid WHERE s.session_id=? "
            "ORDER BY r.chunk_id", (sid,)).fetchall()]
    return [tuple(r) for r in conn.execute("SELECT chunk_id,session_id,kind,text FROM chunks_fts WHERE session_id=?"
                                           " ORDER BY chunk_id", (sid,)).fetchall()]


def integrity_ok(index_dir: Path) -> bool:
    """``PRAGMA integrity_check`` of the metadata database, opened on its own."""
    with sqlite3.connect(str(Path(index_dir) / "meta.sqlite")) as conn:
        return conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]


def forced(max_records=None, *, cap_bytes=None, **kwargs) -> SlicePolicy:
    """A slice policy whose stops are decided by counts, not by time: the minimum progress is waived."""
    options = dict(mode="auto", min_records=1, min_bytes=0, max_records=max_records)
    if cap_bytes is not None:
        options["cap_bytes"] = cap_bytes
    options.update(kwargs)
    return SlicePolicy(**options)


class Tick:
    """A clock that moves ``step`` seconds every time it is read. The soft deadline of a slice then passes after a fixed
    number of looks (64 records each), whatever the speed of the machine."""

    def __init__(self, step: float = 0.05):
        self.now, self.step = 0.0, step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


HARD = 3600.0      # seconds: a hard deadline that no run of a test gets anywhere near


def scripted_deadline(monkeypatch, nominal):
    """The hard deadline of a call whose slices are cut by a scripted clock (``Tick``), and the soft one such a call has.

    A slice's soft deadline is a share of the time the request has left on the REAL clock (``ingest.SYNC_SHARE``), so with a
    real deadline of ``nominal`` seconds a slow machine (a sanitizer build is three to four times slower, a loaded runner
    more) can run out of the hard deadline before the script has cut the slice, and the test fails on the speed of the
    machine, not on what it checks. Here the hard deadline is an hour away (``HARD``) and ``SYNC_SHARE`` is scaled by
    ``nominal / HARD``: the soft deadline is the one a request of ``nominal`` seconds has (0.5 * nominal / 1.3 seconds of
    scripted time) and no wall clock decides anything. Returns the function that gives the deadline of one call."""
    monkeypatch.setattr(ingest, "SYNC_SHARE", ingest.SYNC_SHARE * nominal / HARD)
    return lambda: time.monotonic() + HARD
