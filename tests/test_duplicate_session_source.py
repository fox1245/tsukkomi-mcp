"""Several files may claim one (provider, session id): copies and continuations are adopted, never reparsed.

Synthetic sessions only. Every provider except AGY (whose hook receipts bind to an exact path) shares the rule.
"""
from __future__ import annotations

import threading
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import pytest

from self_directing_mcp import omp
from self_directing_mcp.codex import parse as codex_parser
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index import ingest
from test_agy_transcript import SID as AGY_SID, session as agy_session  # noqa: F401
from test_omp_local import _message
from test_reliability import SID, message, write

OMP_SID = "session-1"
PREFIX = "requested_source_is_prefix_of_indexed"


class World:
    """One engine, one session id, any number of files that claim it."""

    def __init__(self, provider, engine, root):
        self.provider, self.engine, self.root = provider, engine, root
        self.sid = SID if provider == "codex" else OMP_SID
        self.parser = codex_parser if provider == "codex" else omp
        self.parses = []

    def event(self, text):
        if self.provider == "codex":
            return message(text)
        return _message("assistant", {"type": "text", "text": text})

    def header(self):
        if self.provider == "codex":
            return {"type": "session_meta", "payload": {"id": SID}}
        return {"type": "session", "id": OMP_SID}

    def records(self, start, stop, *, diverge_at=None, size=0):
        pad = "x" * size
        return [self.event(("other" if diverge_at is not None and i >= diverge_at else "event") + f" {i} {pad}")
                for i in range(start, stop)]

    def file(self, name, count, **options):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, [self.header(), *self.records(0, count, **options)])
        return path.resolve()

    def grow(self, path, start, stop):
        write(path, self.records(start, stop), "a")

    def spy(self, monkeypatch):
        original = self.parser.parse_session_chunks

        def counted(*args, **kwargs):
            self.parses.append(kwargs.get("start_byte", 0))
            return original(*args, **kwargs)

        monkeypatch.setattr(self.parser, "parse_session_chunks", counted)
        return original

    def sync(self, path):
        return self.engine.sync_session(session_id=self.sid, path=str(path), provider=self.provider, embed=False)

    def sync_authority_only(self, path):
        """The audit refresh: no derived (sparse/dense) repair, so only source and metadata work is measured."""
        return self.engine._sync_session(session_id=self.sid, path=str(path), provider=self.provider, embed=False,
                                         derived=False)

    def audit(self, path):
        return self.engine.audit_session(self.sid, provider=self.provider, path=str(path))

    def cursor(self):
        return self.engine.store.cursor_record(self.sid, self.provider)

    def chunks(self):
        return self.engine.store.chunk_count(self.sid, self.provider)

    def state(self):
        return self.engine.store.state_token(self.sid, self.provider)

    def contract(self):
        self.engine.upsert_contracts([{"id": "no-curl", "type": "must_not", "scope": "tool_call",
                                       "regex": "curl", "provider": self.provider}])


@pytest.fixture(params=["codex", "omp"])
def world(request, tmp_path, monkeypatch):
    for name in ("SELF_DIRECT_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER",
                 "SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR", "OPENROUTER_API_KEY", "SELF_DIRECT_OPENROUTER_API_KEY_FILE",
                 "SELF_DIRECT_OMP_SESSIONS_DIR"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "sessions"
    root.mkdir()
    if request.param == "codex":
        settings = Settings(codex_sessions_dir=root, index_dir=tmp_path / "index", use_fake_embedder=True, _env_file=None)
    else:
        monkeypatch.setenv("SELF_DIRECT_LOCAL_ONLY", "true")
        settings = Settings(index_dir=tmp_path / "index", omp_sessions_dir=root,
                            openrouter_api_key_file=None, use_fake_embedder=False)
    engine = SelfDirectEngine(settings)
    try:
        yield World(request.param, engine, root)
    finally:
        engine.close()


def test_identical_copies_alternate_without_reparse(world, monkeypatch):
    world.spy(monkeypatch)
    first, second = world.file("one/s.jsonl", 6), world.file("two/s.jsonl", 6)
    assert first.read_bytes() == second.read_bytes()
    results = [world.sync(path) for path in (first, second) * 3]
    assert world.parses == [0]
    assert world.chunks() > 0 and {r["total_chunks"] for r in results} == {world.chunks()}
    for result in results:
        assert result["ok"] and not result["reindexed"] and result["reindex_reason"] is None
        assert "source_changed" not in result["coverage"]["issues"] and result["coverage"]["complete"]
    assert world.cursor()["path"] == str(first)  # the first source stays; an equal copy never moves the cursor
    assert "alias_of" not in results[0]
    for result in results[1::2]:
        assert result["alias_of"] == str(first) and result["path"] == str(second)
        assert result["coverage"]["indexed_path"] == str(first) and result["coverage"]["requested_path"] == str(second)


def test_continuation_parses_only_new_events_and_never_rewinds(world, monkeypatch):
    world.spy(monkeypatch)
    short, long = world.file("one/s.jsonl", 4), world.file("two/s.jsonl", 9)
    world.sync(short)
    before = world.chunks()
    info = world.sync(long)
    assert world.parses == [0, short.stat().st_size]  # only the appended events were parsed
    assert info["ok"] and not info["reindexed"] and info["new_chunks"] == world.chunks() - before > 0
    cursor = world.cursor()
    assert cursor["path"] == str(long) and cursor["byte_offset"] == long.stat().st_size
    state = world.state()
    for path in (short, long, short, long, short):
        info = world.sync(path)
        assert info["ok"] and not info["reindexed"] and info["reindex_reason"] is None
        assert "source_changed" not in info["coverage"]["issues"]
        if path == short:
            assert info["coverage"]["indexed_path"] == str(long) and PREFIX in info["coverage"]["issues"]
        else:
            assert info["coverage"]["complete"] and "alias_of" not in info
    assert world.parses == [0, short.stat().st_size] and world.state() == state  # nothing rewound or rewritten


def test_long_first_then_short_reports_the_indexed_source(world, monkeypatch):
    world.spy(monkeypatch)
    world.contract()
    long, short = world.file("one/s.jsonl", 9), world.file("two/s.jsonl", 4)
    world.sync(long)
    state = world.state()
    info = world.sync(short)
    assert world.parses == [0] and world.state() == state
    assert info["ok"] and not info["reindexed"] and info["alias_of"] == str(long) and info["path"] == str(short)
    coverage = info["coverage"]
    assert coverage["indexed_path"] == str(long) and coverage["requested_path"] == str(short)
    assert coverage["byte_offset"] == long.stat().st_size
    # The index holds events the requested file does not contain: complete would be a false claim for it.
    assert PREFIX in coverage["issues"] and coverage["complete"] is False
    assert "source_changed" not in coverage["issues"]
    assert world.audit(short)["verdict"] == "unknown"
    assert world.audit(long)["verdict"] == "clean"
    assert world.state() == state


def test_copy_cut_inside_a_record_is_still_a_prefix(world, monkeypatch):
    world.spy(monkeypatch)
    long = world.file("one/s.jsonl", 8)
    cut = world.root / "two" / "s.jsonl"
    cut.parent.mkdir()
    cut.write_bytes(long.read_bytes()[:long.stat().st_size // 2 + 7])
    assert not cut.read_bytes().endswith(b"\n")
    world.sync(long)
    info = world.sync(cut)
    assert info["ok"] and not info["reindexed"] and info["alias_of"] == str(long) and world.parses == [0]


def test_copies_found_by_path_alone_share_the_session(world, monkeypatch):
    world.spy(monkeypatch)
    first, second = world.file("one/s.jsonl", 5), world.file("two/s.jsonl", 5)
    for path in (first, second, first, second):
        info = world.engine.sync_session(path=str(path), provider=world.provider, embed=False)
        assert info["ok"] and not info["reindexed"] and info["session_id"] == world.sid
    assert world.parses == [0]


def test_alias_request_still_follows_growth_of_the_indexed_source(world, monkeypatch):
    world.spy(monkeypatch)
    long, short = world.file("one/s.jsonl", 6), world.file("two/s.jsonl", 3)
    world.sync(long)
    size = long.stat().st_size
    world.grow(long, 6, 8)
    info = world.sync(short)  # an alias request still reports (and catches up) the indexed source it stands for
    assert world.parses == [0, size] and not info["reindexed"]
    assert world.cursor()["byte_offset"] == long.stat().st_size and info["coverage"]["indexed_path"] == str(long)


@pytest.mark.parametrize("variant", ["same_size", "longer", "shorter"])
def test_diverged_files_are_ambiguous_and_leave_the_index_alone(world, monkeypatch, variant):
    world.spy(monkeypatch)
    world.contract()
    indexed = world.file("one/s.jsonl", 6)
    rival = world.file("two/s.jsonl", {"same_size": 6, "longer": 9, "shorter": 3}[variant], diverge_at=2)
    if variant == "same_size":
        assert rival.stat().st_size == indexed.stat().st_size
    world.sync(indexed)
    state, cursor = world.state(), world.cursor()
    info = world.sync(rival)
    assert info["ok"] is False and info["error"] == "ambiguous_session_source"
    assert info["indexed_path"] == str(indexed) and info["requested_path"] == str(rival)
    assert world.parses == [0] and world.state() == state and world.cursor() == cursor
    audit = world.audit(rival)
    assert audit["verdict"] == "unknown" and audit["coverage"]["complete"] is False
    assert "ambiguous_session_source" in audit["coverage"]["issues"]
    assert world.state() == state
    again = world.sync(indexed)  # the indexed source is untouched and still current
    assert again["ok"] and not again["reindexed"] and world.audit(indexed)["verdict"] == "clean"
    assert world.parses == [0]


def test_moved_file_is_adopted_without_reindex(world, monkeypatch):
    world.spy(monkeypatch)
    old = world.file("one/s.jsonl", 5)
    world.sync(old)
    new = world.file("two/s.jsonl", 5)
    old.unlink()
    info = world.sync(new)
    assert info["ok"] and not info["reindexed"] and info["reindex_reason"] is None and "alias_of" not in info
    assert world.parses == [0] and world.cursor()["path"] == str(new)
    assert info["coverage"]["complete"] and world.sync(new)["coverage"]["complete"]


def test_moved_file_that_also_grew_continues_from_the_cursor(world, monkeypatch):
    world.spy(monkeypatch)
    old = world.file("one/s.jsonl", 5)
    world.sync(old)
    size = old.stat().st_size
    new = world.file("two/s.jsonl", 8)
    old.unlink()
    info = world.sync(new)
    assert world.parses == [0, size] and not info["reindexed"] and world.cursor()["path"] == str(new)
    assert world.cursor()["byte_offset"] == new.stat().st_size


@pytest.mark.parametrize("count,diverge_at", [(5, 2), (3, None)])
def test_unrelated_replacement_of_a_vanished_source_still_reindexes(world, monkeypatch, count, diverge_at):
    world.spy(monkeypatch)
    old = world.file("one/s.jsonl", 5)
    world.sync(old)
    new = world.file("two/s.jsonl", count, diverge_at=diverge_at)  # (3, None) is shorter than the indexed history
    old.unlink()
    info = world.sync(new)
    assert info["ok"] and info["reindexed"] and info["reindex_reason"] == "source_changed"
    assert world.cursor()["path"] == str(new)


def test_same_path_rewrite_and_truncation_still_reindex(world):
    path = world.file("one/s.jsonl", 6)
    world.sync(path)
    path.write_bytes(path.read_bytes().replace(b"event 1 ", b"event 9 "))  # same size, earlier bytes changed
    info = world.sync(path)
    assert info["ok"] and info["reindexed"] and info["reindex_reason"] == "source_changed"
    kept = path.read_bytes().splitlines(keepends=True)[:3]
    path.write_bytes(b"".join(kept))  # truncated well below the cursor
    info = world.sync(path)
    assert info["ok"] and info["reindexed"] and info["reindex_reason"] == "source_changed"
    assert world.cursor()["byte_offset"] == path.stat().st_size and "alias_of" not in info


def test_parser_update_on_a_copy_reparses_once(world, monkeypatch):
    first, second = world.file("one/s.jsonl", 5), world.file("two/s.jsonl", 5)
    with monkeypatch.context() as legacy:
        legacy.setattr(world.parser, "PARSER_VERSION", 1)
        world.sync(first)
    world.spy(monkeypatch)
    results = [world.sync(path) for path in (second, first, second, first)]
    assert world.parses == [0] and results[0]["reindexed"] and results[0]["reindex_reason"] == "parser_updated"
    assert not any(result["reindexed"] for result in results[1:])


def test_paths_of_one_session_are_single_flighted(world, monkeypatch):
    original = world.spy(monkeypatch)
    short, long = world.file("one/s.jsonl", 4), world.file("two/s.jsonl", 9)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(world.sync, (short, long) * 8))
    assert all(r["ok"] for r in results)
    assert len(world.parses) <= 2 and world.parses[0] == 0  # first source, then at most the continuation
    expected = len(original(long, session_id=world.sid)[0])
    assert world.chunks() == expected and world.cursor()["path"] == str(long)
    assert world.cursor()["byte_offset"] == long.stat().st_size


def test_stale_preparation_from_another_path_cannot_publish(world, monkeypatch):
    """Another process shares no flight: a preparation made against an older cursor must lose, not overwrite."""
    monkeypatch.setattr(ingest, "session_flight", lambda key: nullcontext())
    original = world.spy(monkeypatch)
    short, long, middle = (world.file("one/s.jsonl", 4), world.file("two/s.jsonl", 9), world.file("three/s.jsonl", 7))
    world.sync(short)
    counted, entered, release, armed = world.parser.parse_session_chunks, threading.Event(), threading.Event(), [True]

    def gated(*args, **kwargs):
        if armed.pop() if armed else False:
            entered.set()
            release.wait(10)
        return counted(*args, **kwargs)

    monkeypatch.setattr(world.parser, "parse_session_chunks", gated)
    with ThreadPoolExecutor(1) as pool:
        slow = pool.submit(world.sync, long)  # prepared against the cursor on ``short``
        assert entered.wait(5)
        assert world.sync(middle)["ok"]  # another source publishes first
        published = world.state()
        release.set()
        with pytest.raises(ValueError, match="stale session preparation"):
            slow.result(10)
    assert world.state() == published and world.cursor()["path"] == str(middle)
    assert world.chunks() == len(original(middle, session_id=world.sid)[0])
    assert world.sync(long)["ok"] and world.cursor()["path"] == str(long)  # a fresh sync continues from ``middle``
    assert world.chunks() == len(original(long, session_id=world.sid)[0])


def test_cas_still_guards_paths_that_race_across_processes(world, monkeypatch):
    """Another process shares no flight: the state token must keep the loser from publishing a stale view."""
    monkeypatch.setattr(ingest, "session_flight", lambda key: nullcontext())
    original = world.spy(monkeypatch)
    short, long = world.file("one/s.jsonl", 4), world.file("two/s.jsonl", 9)
    outcomes = []

    def attempt(path):
        try:
            outcomes.append(world.sync(path)["ok"])
        except ValueError as exc:
            assert "stale session preparation" in str(exc) or "source changed during preparation" in str(exc), exc
            outcomes.append(False)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(attempt, (short, long) * 8))
    assert any(outcomes)
    assert world.sync(short)["ok"] and world.sync(long)["ok"]  # settled: one more pass of each is quiet
    expected = len(original(long, session_id=world.sid)[0])
    assert world.chunks() == expected and world.cursor()["path"] == str(long)
    assert world.cursor()["byte_offset"] == long.stat().st_size


def test_alias_checks_stream_instead_of_loading_files(world):
    big, copy, short = (world.file("one/s.jsonl", 40, size=400_000), world.file("two/s.jsonl", 40, size=400_000),
                        world.file("three/s.jsonl", 10, size=400_000))
    size = big.stat().st_size
    assert size > 15_000_000
    world.sync(big)
    peaks = []
    for path in (copy, short):
        tracemalloc.start()
        try:
            info = world.sync_authority_only(path)
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        assert info["ok"] and not info["reindexed"]
    assert max(peaks) < 0.4 * size, (peaks, size)


def test_agy_keeps_path_identity_because_receipts_bind_to_the_exact_file(agy_session):
    engine, path = agy_session
    mirror = path.with_name("transcript_full.jsonl")
    mirror.write_bytes(path.read_bytes())
    engine.sync_session(session_id=AGY_SID, provider="agy", path=str(path), embed=False)
    info = engine.sync_session(session_id=AGY_SID, provider="agy", path=str(mirror), embed=False)
    assert info["ok"] and info["reindexed"] and info["reindex_reason"] == "source_changed" and "alias_of" not in info
    assert engine.store.cursor_record(AGY_SID, "agy")["path"] == str(mirror.resolve())
