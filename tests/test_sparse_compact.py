"""The compact sparse format (contentless FTS5, ``sparse-v2.sqlite``) returns what the legacy format returns: the same ranked
ids and the same bm25 scores for every kind of query, after every kind of write. The file holds no text and the legacy
file is never read or written by it. ``test_sparse_compact_mutations.py`` shows these checks fail for broken variants."""
from __future__ import annotations

import random
import sqlite3

import pytest

from self_directing_mcp.index import sparse as sparse_module
from self_directing_mcp.index.sparse import (
    CompactSparseIndex, SparseFormatError, SparseIndex, compact_support, count_sparse_file, open_sparse_index, sparse_file,
)
from sparse_support import Doc, SECRET, assert_same_results, build, corpus, pair, queries, sha256


def both(legacy, compact, method, *args, **kwargs):
    getattr(legacy, method)(*args, **kwargs)
    getattr(compact, method)(*args, **kwargs)


def check(legacy, compact, docs, what, minimum=40):
    found = assert_same_results(legacy, compact, queries(docs), what)
    assert found >= minimum, f"{what}: only {found} queries found anything, the comparison is too weak"
    assert compact.verify_statistics()["ok"], (what, compact.verify_statistics())
    assert legacy.count() == compact.count() and legacy.ids() == compact.ids()
    return found


def test_a_plain_build_gives_the_same_results(tmp_path):
    docs = corpus()
    legacy, compact = pair(tmp_path, docs)
    assert check(legacy, compact, docs, "build") > 100
    assert compact.count() == len(docs) and compact.format == "compact" and legacy.format == "legacy"


def test_the_batch_size_of_the_build_does_not_matter(tmp_path):
    docs = corpus(150)
    legacy = build(tmp_path / "l.sqlite", docs, SparseIndex, 7)
    for size in (1, 33, 500):
        compact = build(tmp_path / f"c{size}.sqlite", docs, CompactSparseIndex, size)
        check(legacy, compact, docs, f"batches of {size}", minimum=20)


def test_deletes_and_reupserts_keep_the_results_identical(tmp_path):
    docs = corpus()
    legacy, compact = pair(tmp_path, docs)
    gone = [d.chunk_id for d in docs[::7]] + ["codex:no-such-chunk"]
    both(legacy, compact, "delete_ids", gone)
    check(legacy, compact, docs, "after deleting every 7th")
    changed = [d._replace(text=d.text + " zeta zeta rewritten") for d in docs[:60:3]]
    both(legacy, compact, "upsert_many", changed + [d for d in docs[::7][:20]])      # edits and the deleted ones back
    check(legacy, compact, docs, "after re-upserting")
    both(legacy, compact, "upsert", docs[1].chunk_id, "s9", "meta", "single upsert with a new session and kind alpha")
    both(legacy, compact, "delete_ids", [])
    check(legacy, compact, docs, "after a single upsert")
    assert legacy.count("s9") == compact.count("s9") == 1


def test_random_operation_sequences_stay_identical(tmp_path):
    rng = random.Random(77)
    docs = corpus(250, seed=2)
    legacy, compact = pair(tmp_path, docs[:150])
    live = {d.chunk_id: d for d in docs[:150]}
    spare = docs[150:]
    for step in range(25):
        action = rng.choice(("delete", "upsert", "add", "rewrite", "clear-tail"))
        if action == "delete" and live:
            ids = rng.sample(sorted(live), min(len(live), rng.randrange(1, 15)))
            both(legacy, compact, "delete_ids", ids)
            for cid in ids:
                live.pop(cid)
        elif action == "add" and spare:
            batch = [spare.pop() for _ in range(min(len(spare), rng.randrange(1, 12)))]
            both(legacy, compact, "upsert_many", batch)
            live.update({d.chunk_id: d for d in batch})
        elif action in ("upsert", "rewrite") and live:
            picked = [live[c] for c in rng.sample(sorted(live), min(len(live), rng.randrange(1, 10)))]
            picked = [d._replace(text=d.text + " rewritten " + rng.choice(("alpha", "error", "queue"))) for d in picked]
            both(legacy, compact, "upsert_many", picked)
            live.update({d.chunk_id: d for d in picked})
        elif action == "clear-tail" and live:
            tail = sorted(live)[-8:]                      # the highest rowids go away; the next writes reuse them
            both(legacy, compact, "delete_ids", tail)
            for cid in tail:
                live.pop(cid)
        assert legacy.ids() == compact.ids() == set(live), (step, action)
        assert compact.verify_statistics()["ok"], (step, action)
        if step % 5 == 4:
            check(legacy, compact, docs, f"step {step} {action}", minimum=10)


def test_equal_scores_rank_in_write_order_like_the_legacy_index(tmp_path):
    docs = [Doc(f"codex:s{i % 3}:{i}:0:{i:016x}", f"s{i % 3}", "turn", "ok" if i % 4 else "ok done") for i in range(300)]
    legacy, compact = pair(tmp_path, docs, size=25)
    tie_queries = [{"q": q, "top_k": k, **extra} for q in ("ok", "done", "ok done") for k in (1, 5, 20, 100, 300)
                   for extra in ({}, {"session_id": "s1"}, {"allowed_ids": {d.chunk_id for d in docs[::2]}})]
    assert assert_same_results(legacy, compact, tie_queries, "ties") == len(tie_queries)
    first = [hit[0] for hit in compact.search("ok", 300)]
    assert first[:5] == [d.chunk_id for d in docs if d.text == "ok"][:5]       # ascending write order among the equal scores
    both(legacy, compact, "delete_ids", [d.chunk_id for d in docs[-30:]])
    both(legacy, compact, "upsert_many", [d._replace(chunk_id=d.chunk_id + "x") for d in docs[-30:]])    # reuses the freed rowids
    assert assert_same_results(legacy, compact, tie_queries, "ties after rowid reuse") == len(tie_queries)


def test_a_repeated_id_in_one_batch_keeps_its_last_text(tmp_path):
    docs = corpus(60)
    legacy, compact = pair(tmp_path, docs)
    twice = [Doc("codex:s1:rep:0:0", "s1", "turn", "first version alpha"), docs[3], Doc("codex:s1:rep:0:0", "s1", "turn", "second version beta"),
             docs[3]._replace(text="gamma gamma")]
    both(legacy, compact, "upsert_many", twice)
    check(legacy, compact, docs, "repeated ids", minimum=10)
    assert [h[0] for h in compact.search("first")] == []
    assert [h[0] for h in compact.search("second")] == ["codex:s1:rep:0:0"]
    assert compact.verify_statistics()["ok"]


def test_empty_index_clear_and_refill(tmp_path):
    docs = corpus(120)
    legacy, compact = pair(tmp_path, docs[:0])
    assert compact.count() == 0 and compact.ids() == set() and compact.present(["x"]) == set()
    assert compact.search("alpha") == [] == legacy.search("alpha")
    assert compact.verify_statistics() == {"ok": True, "documents": 0, "tokens": 0, "recorded": [0, 0], "mapped": 0}
    both(legacy, compact, "delete_ids", ["nothing"])
    both(legacy, compact, "upsert_many", [])
    both(legacy, compact, "upsert_many", docs)
    check(legacy, compact, docs, "filled", minimum=30)
    both(legacy, compact, "clear")
    assert compact.count() == 0 and compact.search("alpha") == [] == legacy.search("alpha")
    assert compact.verify_statistics()["ok"] and compact.verify_statistics()["tokens"] == 0
    both(legacy, compact, "upsert_many", docs[:80])
    check(legacy, compact, docs[:80], "refilled", minimum=20)


def test_counts_ids_and_presence_match(tmp_path):
    docs = corpus(120)
    legacy, compact = pair(tmp_path, docs)
    assert compact.count() == legacy.count() == 120
    for sid in ("s0", "s3", "none"):
        assert compact.count(sid) == legacy.count(sid)
    probe = [d.chunk_id for d in docs[:10]] + ["codex:absent"]
    assert compact.present(probe) == legacy.present(probe) == {d.chunk_id for d in docs[:10]}
    assert compact.ids() == legacy.ids()


def test_the_secret_is_masked_before_it_is_tokenized_and_no_text_is_stored(tmp_path):
    phrase = "needle in the haystack sentence of several plain words"
    docs = corpus(40) + [Doc("codex:s1:secret:0:0", "s1", "tool_result", f"{phrase} authorization sk-{'Qx7' * 12} and {SECRET}")]
    legacy, compact = pair(tmp_path, docs)
    tail = SECRET.rsplit("-", 1)[-1]
    for index in (legacy, compact):
        assert index.search(tail) == [], index.format                    # the raw secret is searchable nowhere
        assert [h[0] for h in index.search("haystack")] == ["codex:s1:secret:0:0"]
        assert index.search("sk")                                        # the masked marker is what was indexed
    assert compact._conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE text IS NOT NULL").fetchone()[0] == 0
    assert not [name for (name,) in compact._conn.execute("SELECT name FROM sqlite_master") if "content" in name and not name.endswith("_config")]
    compact._conn.commit()
    for path, expect in ((legacy.db_path, True), (compact.db_path, False)):
        raw = path.read_bytes()
        assert (phrase.encode() in raw) is expect, path.name              # the legacy file keeps the text, the compact one does not
        assert SECRET.rsplit("-", 1)[-1].encode() not in raw


def test_the_format_marker_and_foreign_files(tmp_path):
    docs = corpus(30)
    legacy, compact = pair(tmp_path, docs)
    assert compact.meta() == {"format": "compact-1", "sqlite_version": sqlite3.sqlite_version}
    compact._conn.close()
    reopened = CompactSparseIndex(compact.db_path)                       # reopening an existing file
    assert reopened.count() == 30 and reopened.meta()["format"] == "compact-1"
    legacy._conn.close()
    before = sha256(legacy.db_path)
    with pytest.raises(SparseFormatError, match="another format"):
        CompactSparseIndex(legacy.db_path)                               # a legacy file is refused, not converted
    assert sha256(legacy.db_path) == before
    wrong = tmp_path / "other.sqlite"
    with sqlite3.connect(wrong) as conn:
        conn.execute("CREATE TABLE sparse_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID")
        conn.execute("INSERT INTO sparse_meta VALUES ('format', 'compact-9')")
    conn.close()
    with pytest.raises(SparseFormatError, match="compact-9"):
        CompactSparseIndex(wrong)


def test_an_allow_list_search_that_gives_up_lazy_lookups_answers_the_same(tmp_path, monkeypatch):
    docs = corpus(300)
    legacy, compact = pair(tmp_path, docs)
    with_allowed = [q for q in queries(docs) if q.get("allowed_ids") is not None and q.get("session_id") is None]
    assert len(with_allowed) >= 10
    assert assert_same_results(legacy, compact, with_allowed, "lazy") >= 5
    monkeypatch.setattr(sparse_module, "LAZY_LOOKUPS", 2)           # every such search falls back to the joined query
    assert assert_same_results(legacy, compact, with_allowed, "joined fallback") >= 5
    monkeypatch.setattr(sparse_module, "LAZY_LOOKUPS", 40)          # some finish lazily, some fall back
    assert assert_same_results(legacy, compact, with_allowed, "mixed") >= 5


def test_the_search_is_driven_by_the_fts_scan(tmp_path):
    docs = corpus(80)
    _legacy, compact = pair(tmp_path, docs)
    plan = " | ".join(row[3] for row in compact._conn.execute(
        "EXPLAIN QUERY PLAN SELECT r.chunk_id AS chunk_id, bm25(chunks_fts) AS score FROM chunks_fts CROSS JOIN chunk_rows AS r "
        "ON r.rowid = chunks_fts.rowid WHERE chunks_fts MATCH ? AND r.session_no = ? ORDER BY score, chunks_fts.rowid LIMIT ?",
        ('"alpha"', 1, 5)))
    assert plan.index("chunks_fts") < plan.index("SEARCH r USING INTEGER PRIMARY KEY"), plan


def test_the_compact_file_is_much_smaller(tmp_path):
    """A size regression guard: the same realistic documents take less than 55 % of the legacy file (about 30-35 % measured
    on the synthetic Codex-like and OMP-like corpora of scripts/bench_sparse.py)."""
    rng = random.Random(4)
    vocabulary = [f"{rng.choice('abcdefghij')}{rng.choice('klmnopqrst')}{rng.choice('uvwxyz')}{rng.randrange(10000)}" for _ in range(4000)]
    docs = [Doc(f"codex:s{i % 6}:{i * 1000}:0:{i:016x}", f"s{i % 6}", "tool_result",
                "\n".join(" ".join(rng.choices(vocabulary, k=rng.randrange(4, 14))) + f" /home/user/proj/{rng.choice(vocabulary)}.py"
                          for _ in range(rng.randrange(20, 70))))
            for i in range(500)]
    legacy, compact = pair(tmp_path, docs, size=100)
    legacy._conn.close()
    compact._conn.close()
    legacy_bytes, compact_bytes = legacy.db_path.stat().st_size, compact.db_path.stat().st_size
    assert compact_bytes < 0.55 * legacy_bytes, (compact_bytes, legacy_bytes)


# --- the feature probe: no silent substitution --------------------------------------------------------------------------

def test_an_old_sqlite_is_reported_with_the_repair_and_nothing_is_created(tmp_path, monkeypatch):
    monkeypatch.setattr(sparse_module, "sqlite_version_info", lambda: (3, 42, 1))
    reason = compact_support()
    assert "3.42.1" in reason and "3.43" in reason and "SELF_DIRECT_SPARSE_FORMAT=legacy" in reason and "newer" in reason
    target = tmp_path / "idx" / "sparse-v2.sqlite"
    with pytest.raises(SparseFormatError, match="3.43"):
        CompactSparseIndex(target)
    assert not target.exists()                                                     # an unsupported SQLite leaves no file behind
    with pytest.raises(SparseFormatError, match="SELF_DIRECT_SPARSE_FORMAT=legacy"):
        open_sparse_index(tmp_path / "idx", "compact")
    assert not target.exists()


def test_a_sqlite_that_cannot_keep_the_statistics_exact_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(sparse_module, "_statistics_probe", lambda runtime: ("OperationalError: table may not be modified", False))
    assert "may not be modified" in compact_support()
    with pytest.raises(SparseFormatError, match="may not be modified"):
        open_sparse_index(tmp_path, "compact")
    index, resolution = open_sparse_index(tmp_path, "auto")
    assert index.format == "legacy" and "may not be modified" in resolution["reason"]


def test_the_probe_proves_the_statistics_mechanism_on_this_sqlite():
    sparse_module._statistics_probe.cache_clear()
    reason, needs_patch = sparse_module._statistics_probe(sqlite3.sqlite_version)
    assert reason is None
    assert needs_patch is True or needs_patch is False                              # True on SQLite up to at least 3.53


def test_open_sparse_index_resolves_the_setting(tmp_path, monkeypatch):
    compact, resolution = open_sparse_index(tmp_path, "auto")
    assert (compact.format, resolution) == ("compact", {"setting": "auto", "format": "compact", "reason": None})
    assert compact.db_path == sparse_file(tmp_path, "compact") == tmp_path / "sparse-v2.sqlite"
    assert not sparse_file(tmp_path, "legacy").exists()                              # the other file is not even created
    legacy, resolution = open_sparse_index(tmp_path, "legacy")
    assert (legacy.format, resolution) == ("legacy", {"setting": "legacy", "format": "legacy", "reason": None})
    assert legacy.db_path == tmp_path / "sparse.sqlite"
    forced, resolution = open_sparse_index(tmp_path, "compact")
    assert forced.format == "compact"
    monkeypatch.setattr(sparse_module, "sqlite_version_info", lambda: (3, 40, 0))
    fallback, resolution = open_sparse_index(tmp_path, "auto")
    assert fallback.format == "legacy" and resolution["setting"] == "auto" and "3.40.0" in resolution["reason"]
    with pytest.raises(ValueError, match="auto, legacy or compact"):
        open_sparse_index(tmp_path, "v2")


def test_count_sparse_file_reads_without_creating_or_changing(tmp_path):
    docs = corpus(25)
    legacy, compact = pair(tmp_path, docs)
    legacy._conn.close()
    before = sha256(legacy.db_path)
    assert count_sparse_file(legacy.db_path) == 25 and count_sparse_file(compact.db_path) == 25
    assert sha256(legacy.db_path) == before
    missing = tmp_path / "absent.sqlite"
    assert count_sparse_file(missing) is None and not missing.exists()
    junk = tmp_path / "junk.sqlite"
    junk.write_bytes(b"not a database" * 10)
    assert count_sparse_file(junk) is None and junk.read_bytes() == b"not a database" * 10
