"""Orphans of the compact sparse index: documents the canonical store no longer holds. ``rebuild_sparse`` (the ``sparse-rebuild``
command) and a global ``sparse``/``hybrid`` search delete them in bounded batches, never a document the store holds, with the bm25
statistics kept exact, and never write the legacy ``sparse.sqlite``. Synthetic data only. The mutation checks at the end show that
the scenario fails for a deletion that removes a live id, skips the statistics correction, ignores a budget or touches the legacy
file."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_directing_mcp import engine as engine_module
from self_directing_mcp import maintenance
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index import sparse as sparse_module
from self_directing_mcp.index.sparse import CompactSparseIndex, SparseIndex
from self_directing_mcp.request_control import IndexBusy, RequestStopped
from sparse_support import Doc, assert_same_results, build, corpus, queries, run
from test_reliability import SID, grok_message, message, write
from test_sparse_formats import make, stat_key

pytestmark = pytest.mark.skipif(sparse_module.compact_support() is not None,
                                reason=f"this SQLite cannot use the compact format: {sparse_module.compact_support()}")

QUERIES = ("alpha", "alpha beta", "pre-commit gamma", "filler text uniqueword3", "event_number7 alpha", "beta gamma uniqueword11")


def orphan_docs(count, seed=1):
    """Synthetic documents the store never held. Their ids sort before every real id ("0" < "1" < "c"), so a test knows which
    delete batch a real id lands in."""
    return [Doc(f"codex:0-orphan-{i:05d}:{i}:0:{i:016x}", f"orphan-{i % 3}", doc.kind, doc.text)
            for i, doc in enumerate(corpus(count, seed=seed))]


def world(tmp_path, monkeypatch, *, events=24, orphans=11, legacy=True, grok=False):
    """An engine whose compact index holds every canonical event plus ``orphans`` extra documents. With ``legacy`` a
    ``sparse.sqlite`` made by the previous format sits next to it; ``assert_legacy()`` proves it is byte-identical."""
    index = tmp_path / "index"
    if legacy:
        old, _ = make(tmp_path, monkeypatch, events=events, sparse_format="legacy")
        try:
            old.sync_session(session_id=SID, embed=False)
        finally:
            old.close()
    engine, path = make(tmp_path, monkeypatch, events=events)
    engine.sync_session(session_id=SID, embed=False)
    if grok:
        root = tmp_path / "grok"
        directory = root / SID
        directory.mkdir(parents=True)
        write(directory / f"{SID}.jsonl", [grok_message(f"alpha beta grokbot_only{i} filler text") for i in range(9)])
        engine.settings.grokbot_transcripts_dir = str(root)
        assert engine.sync_session(session_id=SID, provider="grokbot", embed=False)["ok"]
    docs = orphan_docs(orphans)
    if docs:
        engine.sparse.upsert_many(docs)
    legacy_path = index / "sparse.sqlite"
    before = stat_key(legacy_path) if legacy else None

    def assert_legacy():
        if legacy:
            assert stat_key(legacy_path) == before and not (index / "sparse.sqlite-journal").exists(), "the legacy file changed"
        else:
            assert not legacy_path.exists(), "a legacy file appeared"

    return SimpleNamespace(engine=engine, path=path, index=index, orphans={d.chunk_id for d in docs}, assert_legacy=assert_legacy,
                           canonical=engine.store.chunk_ids(None, None), root=tmp_path)


def close(w):
    w.engine.close()


def reach(engine, query, **kw):
    return [(h["chunk_id"], h["sparse_rank"], h["sparse_score"]) for h in engine.search_history(query, mode="sparse", **kw)["hits"]]


def forget_row(engine, chunk_id):
    """Take one event out of the canonical store only and return a function that puts it back (an event that arrives)."""
    conn = engine.store._conn
    row = conn.execute("SELECT * FROM session_events WHERE chunk_id=?", (chunk_id,)).fetchone()
    with conn:
        conn.execute("DELETE FROM session_events WHERE chunk_id=?", (chunk_id,))

    def restore():
        with conn:
            conn.execute("INSERT INTO session_events VALUES (" + ",".join("?" * len(row)) + ")", tuple(row))
    return restore


# ---------------------------------------------------------------------------------------------------------------------
# the index: counting deletes, and the parity with an index that never held the orphans
# ---------------------------------------------------------------------------------------------------------------------
def test_delete_ids_reports_what_it_removed(tmp_path):
    docs = corpus(60)
    index = build(tmp_path / "c.sqlite", docs, CompactSparseIndex)
    ids = [d.chunk_id for d in docs[:10]]
    assert index.delete_ids(ids + ["codex:unknown", ids[0]]) == 10                # an unknown id and a repeat count nothing
    assert index.delete_ids(ids) == 0 and index.delete_ids([]) == 0
    assert index.count() == 50 and index.verify_statistics()["ok"]


def test_the_store_answers_which_ids_it_holds(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, events=700, orphans=0, legacy=False)
    try:
        held = sorted(w.canonical)
        assert len(held) > 600                                                      # more than one lookup batch
        assert w.engine.store.existing_ids(held + ["codex:nowhere", "grokbot:nowhere"]) == w.canonical
        assert w.engine.store.existing_ids(["codex:nowhere"]) == set() and w.engine.store.existing_ids([]) == set()
    finally:
        close(w)


def test_deleting_orphans_gives_the_results_of_an_index_that_never_held_them(tmp_path):
    docs = corpus(320, seed=4)
    orphans = docs[::4]
    kept = [d for i, d in enumerate(docs) if i % 4]
    held, clean = build(tmp_path / "held.sqlite", docs, CompactSparseIndex), build(tmp_path / "clean.sqlite", kept, CompactSparseIndex)
    allowed = {d.chunk_id for d in kept}
    probe = [q for q in queries(kept) if q.get("allowed_ids") is None or q["allowed_ids"] <= allowed]
    perturbed = sum(a != b for a, b in zip(run(clean, probe), run(held, probe)))
    assert perturbed >= 10, "the orphans must change some scores, or this comparison proves nothing"
    gone = sorted(d.chunk_id for d in orphans)
    for start in range(0, len(gone), 37):
        assert held.delete_ids(gone[start:start + 37]) == len(gone[start:start + 37])
    assert held.ids() == clean.ids() == allowed
    assert assert_same_results(clean, held, probe, "after the orphans were deleted") > 40         # ids, ranks and scores to 1e-12
    assert held.verify_statistics()["ok"] and held.verify_statistics()["documents"] == len(kept)
    # the multi-token queries are in that set; spell one out
    multi = "alpha beta gamma run"
    assert held.search(multi, 20) == clean.search(multi, 20) and held.search(multi, 20)


# ---------------------------------------------------------------------------------------------------------------------
# the rebuild
# ---------------------------------------------------------------------------------------------------------------------
def test_rebuild_removes_the_orphans_keeps_the_events_and_leaves_the_legacy_file_alone(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, orphans=17)
    try:
        assert w.engine.sparse.ids() == w.canonical | w.orphans
        assert w.engine.audit_status()["sparse_extra_ids"] == 17
        w.assert_legacy()
        info = w.engine.rebuild_sparse()
        assert info["state"] == "complete" and info["written"] == 0 and info["remaining"] == 0
        assert (info["orphans"], info["orphans_removed"], info["orphans_spared"], info["orphans_remaining"]) == (17, 17, 0, 0)
        assert info["canonical"] == len(w.canonical) and info["already_indexed"] == len(w.canonical)
        assert info["statistics"]["ok"] and info["statistics"]["documents"] == len(w.canonical)
        assert w.engine.sparse.ids() == w.canonical and w.engine.sparse.verify_statistics()["ok"]
        assert w.engine.audit_status()["sparse_extra_ids"] == 0
        assert w.engine.search_history("alpha", session_id=SID, mode="sparse")["hits"]
        w.assert_legacy()
    finally:
        close(w)
    w.assert_legacy()


def test_rebuild_after_the_engine_scores_like_an_index_that_never_held_the_orphans(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    dirty = world(tmp_path / "a", monkeypatch, orphans=40, legacy=False)
    clean = world(tmp_path / "b", monkeypatch, orphans=0, legacy=False)
    try:
        assert dirty.canonical == clean.canonical
        before = [dirty.engine.sparse.search(q, 20, allowed_ids=dirty.canonical) for q in QUERIES]
        reference = [clean.engine.sparse.search(q, 20, allowed_ids=clean.canonical) for q in QUERIES]
        assert any(a != b for a, b in zip(before, reference)), "the orphans must move some score, or this proves nothing"
        assert dirty.engine.rebuild_sparse()["orphans_removed"] == 40
        after = [dirty.engine.sparse.search(q, 20, allowed_ids=dirty.canonical) for q in QUERIES]
        assert after == reference and all(after)
        for q in QUERIES:
            assert reach(dirty.engine, q, session_id=SID) == reach(clean.engine, q, session_id=SID)
        assert dirty.engine.sparse.verify_statistics() == clean.engine.sparse.verify_statistics()
    finally:
        close(dirty)
        close(clean)


def test_the_rebuild_takes_both_sets_from_one_snapshot_under_the_publication_guard(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, orphans=9)
    try:
        depth, seen = [], []
        real = w.engine._publication_guard

        @contextmanager
        def guard():
            with real():
                depth.append(1)
                try:
                    yield
                finally:
                    depth.pop()

        monkeypatch.setattr(w.engine, "_publication_guard", guard)
        for owner, name in ((w.engine.store, "chunk_ids"), (w.engine.sparse, "ids"), (w.engine.store, "existing_ids"),
                            (w.engine.sparse, "delete_ids")):
            def wrapper(*args, _real=getattr(owner, name), _name=name, **kwargs):
                seen.append((_name, bool(depth)))
                return _real(*args, **kwargs)
            monkeypatch.setattr(owner, name, wrapper)
        assert w.engine.rebuild_sparse(batch=4)["orphans_removed"] == 9
        assert {name for name, _under in seen} == {"chunk_ids", "ids", "existing_ids", "delete_ids"}
        assert all(under for _name, under in seen), seen
    finally:
        close(w)


def test_an_event_that_arrives_during_the_rebuild_is_not_removed(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, orphans=14, grok=True)
    try:
        live = sorted(w.canonical)[-1]                       # sorts after every orphan: its delete batch comes last
        restore = forget_row(w.engine, live)                 # at the snapshot it is in the index and not in the store
        assert live in w.engine.sparse.ids() and live not in w.engine.store.chunk_ids(None, None)
        batches, restored = [], []

        def arrives(examined, total, removed):
            batches.append(examined)
            if not restored:
                restore()
                restored.append(examined)

        info = w.engine.rebuild_sparse(batch=4, orphan_progress=arrives)
        assert restored == [4] and batches == [4, 8, 12, 15]
        assert (info["orphans"], info["orphans_removed"], info["orphans_spared"], info["orphans_remaining"]) == (15, 14, 1, 0)
        assert w.engine.sparse.ids() == w.canonical and live in w.engine.sparse.ids()
        assert info["statistics"]["ok"] and info["statistics"]["documents"] == len(w.canonical)
        w.assert_legacy()
    finally:
        close(w)


def test_a_rebuild_with_a_tiny_budget_resumes_and_the_removals_add_up(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, orphans=23)
    try:
        info = w.engine.rebuild_sparse(budget_sec=0.0)                      # an exhausted budget deletes nothing
        assert (info["state"], info["stopped_by"], info["orphans_removed"], info["orphans_remaining"]) == ("partial", "time_budget", 0, 23)
        assert w.engine.sparse.ids() == w.canonical | w.orphans and "statistics" not in info

        removed, runs, seen = 0, [], []
        while True:
            info = w.engine.rebuild_sparse(budget_sec=0.15, batch=4, orphan_progress=lambda *a: (seen.append(a), time.sleep(0.25)))
            runs.append(info)
            removed += info["orphans_removed"]
            assert info["orphans_removed"] <= 4 * 2 and w.engine.sparse.verify_statistics()["documents"] == w.engine.sparse.count()
            if info["state"] == "complete":
                break
            assert info["stopped_by"] == "time_budget" and info["orphans_remaining"] > 0 and info["orphans"] == 23 - (removed - info["orphans_removed"])
            assert len(runs) < 20
        assert removed == 23 and len(runs) >= 3                              # several runs, each keeping its batches
        assert runs[-1]["statistics"]["ok"] and w.engine.sparse.ids() == w.canonical
        assert seen and all(total <= 23 for _done, total, _removed in seen)
        again = w.engine.rebuild_sparse()
        assert (again["orphans"], again["orphans_removed"], again["written"]) == (0, 0, 0)
        w.assert_legacy()
    finally:
        close(w)


def test_a_rebuild_interrupted_between_batches_keeps_them_and_resumes(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, orphans=30)
    try:
        seen = []

        def interrupt(examined, total, removed):
            seen.append((examined, total, removed))
            if len(seen) == 2:
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            w.engine.rebuild_sparse(batch=5, orphan_progress=interrupt)
        assert seen == [(5, 30, 5), (10, 30, 10)] and len(w.engine.sparse.ids() - w.canonical) == 20   # finished batches are kept
        info = w.engine.rebuild_sparse(batch=5)
        assert (info["orphans"], info["orphans_removed"], info["orphans_remaining"]) == (20, 20, 0)
        assert w.engine.sparse.ids() == w.canonical and info["statistics"]["ok"]
        w.assert_legacy()
    finally:
        close(w)


def test_the_orphans_wait_while_events_are_still_missing_and_the_budget_covers_both(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, events=20, orphans=10)
    try:
        w.engine.sparse.delete_ids(sorted(w.canonical)[:12])                  # events the index lacks, orphans it should not hold
        info = w.engine.rebuild_sparse(budget_sec=0.15, batch=5, progress=lambda *a: time.sleep(0.25))
        assert info["state"] == "partial" and info["written"] == 5 and info["remaining"] == 7
        assert (info["orphans_removed"], info["orphans_remaining"]) == (0, 10)    # no deletion before the events are all written
        info = w.engine.rebuild_sparse(batch=5)
        assert (info["written"], info["orphans_removed"], info["state"]) == (7, 10, "complete")
        assert w.engine.sparse.ids() == w.canonical and info["statistics"]["ok"]
    finally:
        close(w)


def test_a_finished_index_is_left_alone_by_a_second_run(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, orphans=8)
    try:
        assert w.engine.rebuild_sparse()["orphans_removed"] == 8

        def forbidden(*args, **kwargs):
            raise AssertionError("a finished index must not be written")

        monkeypatch.setattr(w.engine.sparse, "delete_ids", forbidden)
        monkeypatch.setattr(w.engine.sparse, "upsert_many", forbidden)
        monkeypatch.setattr(w.engine.sparse, "_write", forbidden)
        before = stat_key(w.engine.sparse.db_path)
        info = w.engine.rebuild_sparse()
        assert (info["state"], info["written"], info["remaining"], info["orphans"], info["orphans_removed"],
                info["orphans_remaining"]) == ("complete", 0, 0, 0, 0, 0)
        w.engine.sparse._conn.close()
        assert stat_key(w.engine.sparse.db_path) == before
    finally:
        close(w)


def test_a_busy_index_stops_the_deletion_and_the_next_run_finishes(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, orphans=12)
    try:
        real, calls = w.engine._prune_sparse_batch, []

        def busy(ids):
            calls.append(len(ids))
            if len(calls) == 3:
                raise IndexBusy()
            return real(ids)

        monkeypatch.setattr(w.engine, "_prune_sparse_batch", busy)
        info = w.engine.rebuild_sparse(batch=4)
        assert (info["state"], info["stopped_by"], info["orphans_removed"], info["orphans_remaining"]) == ("partial", "index_busy", 8, 4)
        monkeypatch.undo()
        info = w.engine.rebuild_sparse(batch=4)
        assert info["state"] == "complete" and info["orphans_removed"] == 4 and w.engine.sparse.ids() == w.canonical
    finally:
        close(w)


# ---------------------------------------------------------------------------------------------------------------------
# the search repair
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["sparse", "hybrid"])
def test_a_global_search_removes_the_orphans_before_it_answers(tmp_path, monkeypatch, mode):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    dirty = world(tmp_path / "a", monkeypatch, orphans=25)
    clean = world(tmp_path / "b", monkeypatch, orphans=0, legacy=False)
    try:
        assert dirty.engine.sparse.ids() == dirty.canonical | dirty.orphans
        for q in QUERIES:
            found = dirty.engine.search_history(q, mode=mode)["hits"]
            assert [(h["chunk_id"], h["sparse_rank"], h["sparse_score"]) for h in found] == \
                   [(h["chunk_id"], h["sparse_rank"], h["sparse_score"]) for h in clean.engine.search_history(q, mode=mode)["hits"]]
        assert dirty.engine.sparse.ids() == dirty.canonical and dirty.engine.sparse.verify_statistics()["ok"]
        assert dirty.engine.audit_status()["sparse_extra_ids"] == 0
        dirty.assert_legacy()
    finally:
        close(dirty)
        close(clean)


def test_a_search_of_one_session_leaves_the_orphans_to_the_global_search_and_the_rebuild(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, orphans=13)
    try:
        assert w.engine.search_history("alpha", session_id=SID, mode="sparse")["hits"]
        assert w.engine.sparse.ids() == w.canonical | w.orphans                # the compact index cannot list one session cheaply
        assert w.engine.search_history("alpha", mode="sparse")["hits"]
        assert w.engine.sparse.ids() == w.canonical
    finally:
        close(w)


def test_a_search_never_deletes_a_document_of_another_provider(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, orphans=6, grok=True)
    try:
        grok = {cid for cid in w.canonical if cid.startswith("grokbot:")}
        assert grok and grok < w.canonical and w.engine.sparse.ids() == w.canonical | w.orphans
        assert w.engine.search_history("alpha", mode="sparse")["hits"]           # provider codex, all sessions
        assert w.engine.sparse.ids() == w.canonical                              # the orphans went, the grokbot documents stayed
        assert w.engine.search_history("alpha", mode="sparse", provider="grokbot")["hits"]
        assert w.engine.rebuild_sparse()["orphans"] == 0
    finally:
        close(w)


def test_a_consistent_index_costs_a_search_nothing_extra(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, orphans=0, grok=True)
    try:
        calls = []

        def count(owner, name):
            real = getattr(owner, name)

            def wrapper(*args, **kwargs):
                calls.append((name, args))
                return real(*args, **kwargs)
            monkeypatch.setattr(owner, name, wrapper)

        count(w.engine.store, "chunk_ids")
        count(w.engine.store, "existing_ids")
        count(w.engine.sparse, "delete_ids")
        monkeypatch.setattr(w.engine, "_prune_sparse_for_search", lambda ids: calls.append(("prune", ids)))
        assert w.engine.search_history("alpha", mode="sparse")["hits"]
        assert not [name for name, _args in calls if name != "chunk_ids"]
        assert ("chunk_ids", (None, None)) not in calls          # the store-wide id scan is made only when the counts show extras
        assert ("chunk_ids", (None, "codex")) in calls           # what the repair always asked: the ids of its scope
    finally:
        close(w)


def test_a_search_deletes_only_within_its_housekeeping_budget_and_still_answers(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, orphans=40)
    try:
        monkeypatch.setattr(engine_module, "SPARSE_PRUNE_SEARCH_SEC", 0.0)             # a spent budget: the answer is the same
        assert w.engine.search_history("alpha", mode="sparse")["hits"]
        assert w.engine.sparse.ids() == w.canonical | w.orphans
        monkeypatch.setattr(engine_module, "SPARSE_PRUNE_SEARCH_SEC", 0.12)
        monkeypatch.setattr(engine_module, "SPARSE_PRUNE_BATCH", 4)
        real = w.engine._prune_sparse_batch

        def slow(ids):
            time.sleep(0.05)
            return real(ids)

        monkeypatch.setattr(w.engine, "_prune_sparse_batch", slow)
        assert w.engine.search_history("alpha", mode="sparse")["hits"]
        left = len(w.engine.sparse.ids() - w.canonical)
        assert 0 < left < 40, left                                                      # some batches, not all: the budget ended it
        monkeypatch.undo()
        assert w.engine.search_history("alpha", mode="sparse")["hits"]
        assert w.engine.sparse.ids() == w.canonical
    finally:
        close(w)


def test_a_busy_index_does_not_fail_the_search_but_a_stop_does_and_keeps_its_batches(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, orphans=20)
    try:
        monkeypatch.setattr(engine_module, "SPARSE_PRUNE_BATCH", 4)
        real, calls = w.engine._prune_sparse_batch, []

        def flaky(ids):
            calls.append(1)
            if len(calls) == 3:
                raise RequestStopped("request_deadline")
            return real(ids)

        monkeypatch.setattr(w.engine, "_prune_sparse_batch", flaky)
        with pytest.raises(RequestStopped):
            w.engine.search_history("alpha", mode="sparse")
        assert len(w.engine.sparse.ids() - w.canonical) == 12                          # two batches of four were committed

        def busy(ids):
            raise IndexBusy()

        monkeypatch.setattr(w.engine, "_prune_sparse_batch", busy)
        assert w.engine.search_history("alpha", mode="sparse")["hits"]                  # housekeeping only: the answer stands
        assert len(w.engine.sparse.ids() - w.canonical) == 12
        monkeypatch.undo()
        assert w.engine.search_history("alpha", mode="sparse")["hits"]
        assert w.engine.sparse.ids() == w.canonical
    finally:
        close(w)


def test_an_event_that_arrives_before_a_search_deletes_its_batch_is_kept(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, orphans=9)
    try:
        live = sorted(w.canonical)[-1]
        restore = forget_row(w.engine, live)
        real = w.engine._prune_sparse_for_search

        def racing(ids):
            assert live in ids                                    # the snapshot had it as an extra
            restore()                                             # the event is published between the snapshot and the delete
            return real(ids)

        monkeypatch.setattr(w.engine, "_prune_sparse_for_search", racing)
        assert w.engine.search_history("alpha", mode="sparse")["hits"]
        assert w.engine.sparse.ids() == w.canonical and live in w.engine.sparse.ids()
    finally:
        close(w)


# ---------------------------------------------------------------------------------------------------------------------
# audit_status
# ---------------------------------------------------------------------------------------------------------------------
def test_audit_status_counts_the_extra_documents_from_counts_it_already_has(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, orphans=7)
    try:
        def scan(*args, **kwargs):
            raise AssertionError("audit_status must not scan the index for this")

        monkeypatch.setattr(w.engine.sparse, "ids", scan)
        status = w.engine.audit_status()
        assert status["sparse_extra_ids"] == 7 == status["sparse_counts"]["compact"] - status["total_chunks"]
        assert status["sparse_count"] == status["total_chunks"] + 7
        monkeypatch.undo()
        w.engine.rebuild_sparse()
        assert w.engine.audit_status()["sparse_extra_ids"] == 0
        w.engine.sparse.clear()                                                           # an index that lacks events: never negative
        status = w.engine.audit_status()
        assert status["sparse_extra_ids"] == 0 and status["sparse_counts"]["compact"] == 0
    finally:
        close(w)


def test_audit_status_has_no_extra_count_without_a_compact_file(tmp_path, monkeypatch):
    engine, _ = make(tmp_path, monkeypatch, sparse_format="legacy")
    try:
        engine.sync_session(session_id=SID, embed=False)
        assert engine.audit_status()["sparse_extra_ids"] is None and not (tmp_path / "index" / "sparse-v2.sqlite").exists()
    finally:
        engine.close()


# ---------------------------------------------------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------------------------------------------------
def command_env(w, monkeypatch):
    monkeypatch.setenv("SELF_DIRECT_CODEX_SESSIONS_DIR", str(w.root / "sessions"))
    monkeypatch.setenv("SELF_DIRECT_USE_FAKE_EMBEDDER", "true")
    return ["sparse-rebuild", "--index-dir", str(w.index)]


def test_the_rebuild_command_removes_the_orphans_and_reports_them(tmp_path, monkeypatch, capsys):
    w = world(tmp_path, monkeypatch, orphans=19)
    close(w)
    argv = command_env(w, monkeypatch)
    assert maintenance.main([*argv, "--budget-sec", "0"]) == maintenance.EXIT_PARTIAL
    out, err = capsys.readouterr()
    info = json.loads(out)
    assert (info["stopped_by"], info["orphans_removed"], info["orphans_remaining"]) == ("time_budget", 0, 19)
    assert "19 extra documents left" in err and "run the command again" in err

    assert maintenance.main([*argv, "--batch", "5"]) == 0
    out, err = capsys.readouterr()
    info = json.loads(out)
    assert info["state"] == "complete" and info["orphans"] == info["orphans_removed"] == 19 and info["orphans_remaining"] == 0
    assert info["written"] == 0 and info["statistics"]["ok"]
    assert f"19/19 extra documents" in err and "19 removed this run" in err                  # progress goes to stderr
    w.assert_legacy()

    key = stat_key(w.index / "sparse-v2.sqlite")
    assert maintenance.main(argv) == 0
    info = json.loads(capsys.readouterr()[0])
    assert (info["orphans"], info["orphans_removed"], info["written"]) == (0, 0, 0)
    assert stat_key(w.index / "sparse-v2.sqlite")[0] == key[0]
    w.assert_legacy()


def test_a_rebuild_killed_between_deletions_leaves_a_sound_index_and_resumes(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, orphans=1500)
    close(w)
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1",
           "SELF_DIRECT_CODEX_SESSIONS_DIR": str(tmp_path / "sessions"), "SELF_DIRECT_USE_FAKE_EMBEDDER": "true",
           "SELF_DIRECT_DENSE_BACKEND": "numpy"}
    command = [sys.executable, "-m", "self_directing_mcp.maintenance", "sparse-rebuild", "--index-dir", str(w.index), "--batch", "1"]
    process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert "extra documents" in process.stderr.readline()            # the first deletion is committed before this line
        process.kill()                                                    # no cleanup, no engine.close(): a crash
    finally:
        process.communicate(timeout=60)
    compact = CompactSparseIndex(w.index / "sparse-v2.sqlite")
    try:
        left = compact.ids()
        assert w.canonical <= left and len(left - w.canonical) <= 1500    # nothing real was lost, however far it got
        assert compact._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok" and compact.verify_statistics()["ok"]
    finally:
        compact._conn.close()
    again = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=300)
    assert again.returncode == 0, again.stderr[-1500:]
    info = json.loads(again.stdout)
    assert info["orphans"] == info["orphans_removed"] == len(left - w.canonical) and info["statistics"]["ok"]
    w.assert_legacy()
    finished = CompactSparseIndex(w.index / "sparse-v2.sqlite")
    try:
        assert finished.ids() == w.canonical
    finally:
        finished._conn.close()


# ---------------------------------------------------------------------------------------------------------------------
# mutation checks: the scenario below must fail for each broken variant of the deletion
# ---------------------------------------------------------------------------------------------------------------------
original = {name: getattr(SelfDirectEngine, name) for name in ("rebuild_sparse", "_prune_sparse_batch", "_prune_sparse_for_search")}


def scenario(tmp_path, monkeypatch):
    """A canonical event that comes back during the rebuild, a second provider, the legacy file, budgets, and the statistics."""
    w = world(tmp_path, monkeypatch, events=20, orphans=14, grok=True)
    try:
        everything = w.canonical | w.orphans
        info = w.engine.rebuild_sparse(budget_sec=0.0)                                       # a budget of nothing deletes nothing
        assert info["orphans_removed"] == 0 and w.engine.sparse.ids() == everything, "the budget was ignored"
        monkeypatch.setattr(engine_module, "SPARSE_PRUNE_SEARCH_SEC", 0.0)
        assert w.engine.search_history("alpha", mode="sparse")["hits"]                       # a spent search budget deletes nothing
        assert w.engine.sparse.ids() == everything, "the search budget was ignored"
        monkeypatch.setattr(engine_module, "SPARSE_PRUNE_SEARCH_SEC", 2.0)

        live = sorted(w.canonical)[-1]
        restore = forget_row(w.engine, live)
        restored = []
        info = w.engine.rebuild_sparse(batch=4, orphan_progress=lambda *a: restored or (restore(), restored.append(1)))
        assert live in w.engine.sparse.ids(), "a live id was removed"
        assert w.engine.sparse.ids() == w.canonical, "the orphans were not all deleted"
        assert (info["orphans_removed"], info["orphans_spared"]) == (14, 1)
        assert info["statistics"]["ok"], "the statistics were not corrected"
        assert w.engine.sparse.verify_statistics()["ok"]
        w.assert_legacy()

        w.engine.sparse.upsert_many(orphan_docs(6, seed=2))                                  # a global search of provider codex
        assert w.engine.search_history("alpha", mode="sparse")["hits"]
        assert w.engine.sparse.ids() == w.canonical, "a document of another provider was removed, or the orphans stayed"
        assert w.engine.sparse.verify_statistics()["ok"], "the statistics were not corrected"
        w.assert_legacy()
        again = w.engine.rebuild_sparse()
        assert (again["orphans"], again["written"]) == (0, 0)
    finally:
        close(w)


def test_the_scenario_passes_without_a_mutation(tmp_path, monkeypatch):
    scenario(tmp_path, monkeypatch)


def removes_a_live_id(monkeypatch):
    def batch(self, ids):
        with self._publication_guard():
            return self.sparse.delete_ids(list(ids)), 0                  # no second look at the store
    monkeypatch.setattr(SelfDirectEngine, "_prune_sparse_batch", batch)


def removes_another_providers_documents(monkeypatch):
    # The extras diffed against the searched provider's ids (every other provider's documents look extra) and no second look
    # at the store; either defence alone would still protect them.
    monkeypatch.setattr(SelfDirectEngine, "_sparse_extra_ids",
                        lambda self, indexed, pending: sorted(indexed - self.store.chunk_ids(None, "codex")))
    removes_a_live_id(monkeypatch)


def skips_the_statistics_correction(monkeypatch):
    monkeypatch.setattr(sparse_module, "_retire_statistics", lambda conn, fts, rowids: None)


def ignores_the_rebuild_budget(monkeypatch):
    monkeypatch.setattr(SelfDirectEngine, "rebuild_sparse",
                        lambda self, *, budget_sec=None, **kwargs: original["rebuild_sparse"](self, budget_sec=None, **kwargs))


def ignores_the_search_budget(monkeypatch):
    def prune(self, ids):
        for start in range(0, len(ids), engine_module.SPARSE_PRUNE_BATCH):
            self._prune_sparse_batch(ids[start:start + engine_module.SPARSE_PRUNE_BATCH])
    monkeypatch.setattr(SelfDirectEngine, "_prune_sparse_for_search", prune)


def touches_the_legacy_file(monkeypatch):
    def batch(self, ids):
        out = original["_prune_sparse_batch"](self, ids)
        legacy = SparseIndex(Path(self.settings.index_dir) / "sparse.sqlite")
        try:
            legacy.upsert(ids[0], SID, "turn", "written into the legacy file")
        finally:
            legacy._conn.close()
        return out
    monkeypatch.setattr(SelfDirectEngine, "_prune_sparse_batch", batch)


def deletes_nothing(monkeypatch):
    monkeypatch.setattr(CompactSparseIndex, "delete_ids", lambda self, ids: 0)


MUTATIONS = [removes_a_live_id, removes_another_providers_documents, skips_the_statistics_correction, ignores_the_rebuild_budget,
             ignores_the_search_budget, touches_the_legacy_file, deletes_nothing]


@pytest.mark.parametrize("mutation", MUTATIONS, ids=lambda m: m.__name__)
def test_every_mutation_is_caught(mutation, tmp_path, monkeypatch):
    mutation(monkeypatch)
    with pytest.raises((AssertionError, sqlite3.Error)):
        scenario(tmp_path, monkeypatch)
