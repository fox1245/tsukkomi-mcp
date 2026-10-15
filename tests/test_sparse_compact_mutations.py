"""Mutation checks of ``test_sparse_compact.py``: each broken variant of ``CompactSparseIndex`` below is a defect the parity
checks must catch. The control run (no mutation) passes the same scenario, so a failure is the mutation's."""
from __future__ import annotations

import sqlite3

import pytest

from self_directing_mcp.index import sparse as sparse_module
from self_directing_mcp.index.sparse import CompactSparseIndex
from self_directing_mcp.security.mask import mask_secrets
from sparse_support import Doc, SECRET, assert_same_results, corpus, pair, queries

original = {name: getattr(CompactSparseIndex, name) for name in ("search", "_drop", "_add", "clear", "upsert_many", "delete_ids")}
pytestmark = pytest.mark.skipif(sparse_module.compact_support() is not None,       # also warms the cached feature probe before anything is broken
                                reason=f"this SQLite cannot use the compact format: {sparse_module.compact_support()}")


def scenario(tmp_path):
    """Build, delete, re-upsert, repeat an id, reuse freed rowids, clear and refill; check ids, statistics, masking and results
    against the legacy index after every step."""
    docs = corpus(220) + [Doc("codex:s1:secret:0:0", "s1", "tool_result", f"needle haystack {SECRET}")]
    legacy, compact = pair(tmp_path, docs)

    def same(what):
        assert assert_same_results(legacy, compact, queries(docs), what) > 40, what
        assert legacy.ids() == compact.ids(), what
        assert compact.verify_statistics()["ok"], what

    def both(method, *args):
        getattr(legacy, method)(*args)
        getattr(compact, method)(*args)

    same("build")
    assert compact.search(SECRET.rsplit("-", 1)[-1]) == []                        # masking before tokenizing
    other = CompactSparseIndex(compact.db_path)                                    # a second writer is locked out from the start
    other._conn.execute("PRAGMA busy_timeout=20")
    try:
        with compact._write():
            try:
                other.delete_ids([docs[0].chunk_id])
                locked = False
            except sqlite3.OperationalError:
                locked = True
        assert locked, "a second writer got in while a write transaction was open"
    finally:
        other._conn.close()
    both("delete_ids", [d.chunk_id for d in docs[::6]])
    same("deletes")
    both("upsert_many", [d._replace(text=d.text + " zeta rewritten") for d in docs[:50:2]])
    same("re-upserts")
    both("upsert_many", [Doc("codex:s2:rep:0:0", "s2", "turn", "first beta"), docs[5], Doc("codex:s2:rep:0:0", "s2", "turn", "second gamma")])
    same("a repeated id in a batch")
    tail = sorted(legacy.ids())[-12:]
    both("delete_ids", tail)
    both("upsert_many", [Doc(cid + "x", "s3", "turn", "reused rowid alpha") for cid in tail])
    same("rowid reuse")
    both("clear")
    both("upsert_many", docs[:90])
    same("clear and refill")


def test_the_scenario_passes_without_a_mutation(tmp_path):
    scenario(tmp_path)


def stale_statistics(monkeypatch):
    monkeypatch.setattr(sparse_module, "_retire_statistics", lambda conn, fts, rowids: None)


def missed_fts_delete(monkeypatch):
    def drop(self, rowids):
        for rowid in rowids:
            self._conn.execute("DELETE FROM chunk_rows WHERE rowid=?", (rowid,))    # the FTS document stays behind
    monkeypatch.setattr(CompactSparseIndex, "_drop", drop)


def missed_delete(monkeypatch):
    monkeypatch.setattr(CompactSparseIndex, "delete_ids", lambda self, ids: None)


def old_version_kept(monkeypatch):
    monkeypatch.setattr(CompactSparseIndex, "_drop", lambda self, rowids: None)


def wrong_rowid_mapping(monkeypatch):
    def add(self, chunk_id, session_id, kind, text, local):
        masked = mask_secrets(text)
        session_no = self._intern("sparse_sessions", "session_id", session_id, local, create=True)
        kind_no = self._intern("sparse_kinds", "kind", kind, local, create=True)
        rowid = self._conn.execute("INSERT INTO chunk_rows(chunk_id, session_no, kind_no) VALUES (?,?,?)",
                                   (chunk_id, session_no, kind_no)).lastrowid
        self._conn.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?,?)", (rowid + 100000, masked))   # points nowhere
    monkeypatch.setattr(CompactSparseIndex, "_add", add)


def masking_skipped(monkeypatch):
    def add(self, chunk_id, session_id, kind, text, local):
        session_no = self._intern("sparse_sessions", "session_id", session_id, local, create=True)
        kind_no = self._intern("sparse_kinds", "kind", kind, local, create=True)
        rowid = self._conn.execute("INSERT INTO chunk_rows(chunk_id, session_no, kind_no) VALUES (?,?,?)",
                                   (chunk_id, session_no, kind_no)).lastrowid
        self._conn.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?,?)", (rowid, text))              # raw text
    monkeypatch.setattr(CompactSparseIndex, "_add", add)


def session_filter_ignored(monkeypatch):
    monkeypatch.setattr(CompactSparseIndex, "search", lambda self, query, top_k=20, *, session_id=None, allowed_ids=None:
                        original["search"](self, query, top_k, session_id=None, allowed_ids=allowed_ids))


def allowed_ids_ignored(monkeypatch):
    monkeypatch.setattr(CompactSparseIndex, "search", lambda self, query, top_k=20, *, session_id=None, allowed_ids=None:
                        original["search"](self, query, top_k, session_id=session_id, allowed_ids=None))


def top_k_ignored(monkeypatch):
    monkeypatch.setattr(CompactSparseIndex, "search", lambda self, query, top_k=20, *, session_id=None, allowed_ids=None:
                        original["search"](self, query, top_k + 7, session_id=session_id, allowed_ids=allowed_ids))


def ties_reversed(monkeypatch):
    def search(self, query, top_k=20, *, session_id=None, allowed_ids=None):
        hits = original["search"](self, query, top_k, session_id=session_id, allowed_ids=allowed_ids)
        out, group = [], []
        for hit in hits:
            if group and hit[1] != group[0][1]:
                out += group[::-1]
                group = []
            group.append(hit)
        out += group[::-1]
        return [(cid, score, rank) for rank, (cid, score, _old) in enumerate(out, start=1)]
    monkeypatch.setattr(CompactSparseIndex, "search", search)


def clear_keeps_statistics(monkeypatch):
    def clear(self):
        with self._write():
            self._conn.execute("DELETE FROM chunks_fts")          # leaves the FTS5 statistics of the deleted rows in place
            self._conn.execute("DELETE FROM chunk_rows")
    monkeypatch.setattr(CompactSparseIndex, "clear", clear)


def deferred_write_lock(monkeypatch):
    from contextlib import contextmanager
    from self_directing_mcp.request_control import publication

    @contextmanager
    def write(self):
        self._conn.execute("BEGIN")                  # the lock is taken at the first write, after the statistics were read
        try:
            yield
            with publication():
                self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
    monkeypatch.setattr(CompactSparseIndex, "_write", write)


def first_of_a_repeated_id_wins(monkeypatch):
    def upsert_many(self, chunks):
        latest = {}
        for chunk in chunks:
            latest.setdefault(chunk.chunk_id, chunk)
        with self._write():
            self._drop(self._retire(list(latest)))
            local = {}
            for chunk in latest.values():
                self._add(chunk.chunk_id, chunk.session_id, chunk.kind, chunk.text, local)
    monkeypatch.setattr(CompactSparseIndex, "upsert_many", upsert_many)


MUTATIONS = [stale_statistics, missed_fts_delete, missed_delete, old_version_kept, wrong_rowid_mapping, masking_skipped,
             session_filter_ignored, allowed_ids_ignored, top_k_ignored, ties_reversed, clear_keeps_statistics,
             first_of_a_repeated_id_wins, deferred_write_lock]


@pytest.mark.parametrize("mutation", MUTATIONS, ids=lambda m: m.__name__)
def test_every_mutation_is_caught(mutation, tmp_path, monkeypatch):
    mutation(monkeypatch)
    with pytest.raises((AssertionError, sqlite3.Error)):
        scenario(tmp_path)
