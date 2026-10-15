"""Helpers for the staged re-index tests (roundtable 8, PR3). Everything here is synthetic.

A re-index needs an index that exists and is out of date: ``old_parser`` makes the first sync of a session run with a parser
one version older (the cursor then carries that version), and ``edit_source`` changes one byte of the session file in place,
which makes the stored hash stop matching. ``canonical`` is everything an index says about one session that must not depend on
how it was re-indexed; ``index_dump`` is the same index byte for byte.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import time
from pathlib import Path

from slicing_support import forced, snapshot, sparse_rows

BIG = 1 << 40            # a deadline so far away that only the forced stops cut a sync short

PARSERS = {"codex": "self_directing_mcp.codex.parse", "omp": "self_directing_mcp.omp", "agy": "self_directing_mcp.agy",
           "claude": "self_directing_mcp.claude"}


@contextlib.contextmanager
def old_parser(provider, back=1):
    """Within the block ``provider``'s parser is ``back`` versions older than it is: what is synced in it is out of date
    afterwards. Grok Bot has no parser version; use ``edit_source`` for it."""
    module = importlib.import_module(PARSERS[provider])
    current = module.PARSER_VERSION
    module.PARSER_VERSION = current - back
    try:
        yield current
    finally:
        module.PARSER_VERSION = current


def edit_source(path: Path, after: float = 0.4, needle: bytes = b"alpha", replacement: bytes = b"alphb") -> int:
    """Change one word in place (same size, valid JSON still): the offset of the change. Which makes the stored hash of the
    source stop matching, so the next sync is a re-index for ``source_changed``."""
    data = bytearray(path.read_bytes())
    at = data.index(needle, int(len(data) * after))
    data[at:at + len(needle)] = replacement
    path.write_bytes(bytes(data))
    return at


def spool_files(engine) -> list[Path]:
    directory = Path(engine.settings.index_dir) / "reindex"
    return sorted(directory.glob("*.spool")) if directory.exists() else []


def reindex_of(info):
    return (info.get("sync") or {}).get("reindex")


TABLES = ("session_events", "session_cursors", "session_parser_versions", "session_metrics", "source_blocks",
          "session_receipt_versions", "session_receipt_deps", "session_agy_stats", "pending_index_deletions",
          "session_generations", "session_commit_log", "embedding_cache")


def index_dump(engine) -> dict:
    """Every row of the tables a re-index may touch (these tests keep one session in an index): the old index, byte for byte."""
    conn = engine.store._conn
    return {table: sorted(tuple(row) for row in conn.execute(f"SELECT * FROM {table}")) for table in TABLES}


def seed_vectors(engine, count=3) -> None:
    """Cached vectors of a model: a re-index must never touch the vector cache."""
    engine.store.cache_vectors("synthetic-model", [(f"text-hash-{n}", [0.25 * n, 1.0, -0.5]) for n in range(count)])


def canonical(engine, provider, sid) -> dict:
    """What an index says about a session that must not depend on HOW it was re-indexed: events (ids and data), cursor,
    block table, metrics, parser version, AGY dependencies and tallies, the ids the re-index made obsolete, the shape of the
    generations (a re-index is one new generation, one commit-log row) and the full-text rows."""
    snap = snapshot(engine, provider, sid)
    key = sid.lower() if provider in ("codex", "agy") else sid
    pending = sorted(row[0] for row in engine.store._conn.execute(
        "SELECT chunk_id FROM pending_index_deletions WHERE provider=? AND session_id=?", (provider, key)))
    return {"events": snap.events, "cursor": snap.cursor, "blocks": snap.blocks, "metrics": snap.metrics,
            "versions": snap.versions, "agy": snap.agy, "pending_deletions": pending,
            "generation": snap.generation, "commit": snap.last_commit, "commit_offsets": snap.commit_offsets,
            "sparse": sparse_rows(engine, key)}


def commits_one_shot(engine, world, **kwargs):
    """The reference: a sync with no deadline is the re-index of before, in one piece."""
    info = engine.sync_session(**world.sync_args(), **kwargs)
    assert "sync" not in info and info["ok"] is True
    return info


def starting_indexes(world, provider, trigger, engines):
    """Index the session in every engine so that a re-index is due, by the trigger: an older parser, a changed word, or (AGY)
    a hook receipt that arrived after its result was indexed. Returns the reason the next sync must give."""
    if trigger == "parser":
        with old_parser(provider):
            for engine in engines:
                engine.sync_session(**world.sync_args(), derived=True)
        return "parser_updated"
    for engine in engines:
        engine.sync_session(**world.sync_args(), derived=True)
    if trigger == "source":
        edit_source(world.path)
        return "source_changed"
    assert trigger == "receipts"
    from self_directing_mcp.agy_receipts import ReceiptStore
    for engine in engines:
        rows = {step: json.loads(data) for step, data in ReceiptStore(engine.settings.index_dir).read_capture(world.sid)[0]}
        step, row = next((s, r) for s, r in sorted(rows.items()) if r.get("paired") and r.get("result_hash"))
        engine.record_agy_hook({"conversationId": world.sid, "transcriptPath": str(world.path), "workspacePaths": ["/synthetic"],
                                "stepIdx": step, "toolCall": json.loads(row["action"]), "error": "reported later"},
                               "PostToolUse")
    return "hook_receipts_changed"


class Pair:
    """A session and two engines that start from the same out-of-date index: ``ref`` is re-indexed in one piece (when asked),
    ``staged`` is the one a test works on. ``before`` is the old index of ``staged``, byte for byte."""

    def __init__(self, tmp_path, monkeypatch, provider="codex", trigger="parser", *, seed=3, count=240, **world_options):
        from slicing_support import World
        self.provider, self.trigger = provider, trigger
        self.world = World(tmp_path, monkeypatch, provider, seed=seed, count=count, **world_options)
        self.ref, self.staged = self.world.engine("reference"), self.world.engine("staged")
        for engine in (self.ref, self.staged):
            self.world.copy_receipts(self.world.builder, engine)
            self.world.upsert_contracts(engine)
        self.original = self.world.path.read_bytes()      # the bytes every engine is indexed from (a source trigger edits them)
        self.reason = starting_indexes(self.world, provider, trigger, (self.ref, self.staged))
        seed_vectors(self.staged)
        self.before = index_dump(self.staged)
        self._expected = None

    @property
    def sid(self):
        return self.world.sid

    @property
    def key(self):
        return self.world.sid.lower() if self.provider in ("codex", "agy") else self.world.sid

    def expected(self):
        """The index the re-index must produce: the reference engine's, re-indexed in one piece now (once)."""
        if self._expected is None:
            commits_one_shot(self.ref, self.world, derived=False)
            self._expected = canonical(self.ref, self.provider, self.sid)
        return self._expected

    def stage(self, cap=30, calls=1, **kwargs):
        """``calls`` syncs that each stage ``cap`` records; returns the last answer."""
        info = None
        for _ in range(calls):
            self.staged.slice_policy = forced(cap)
            info = self.staged.sync_session(**self.world.sync_args(), derived=False, _deadline=time.monotonic() + BIG, **kwargs)
        return info

    def finish(self, **kwargs):
        """Let the settings decide (a far deadline): the re-index is published in this call."""
        self.staged.slice_policy = None
        return self.staged.sync_session(**self.world.sync_args(), derived=False, _deadline=time.monotonic() + BIG, **kwargs)

    def intact(self) -> bool:
        return index_dump(self.staged) == self.before

    def close(self):
        self.world.close()
