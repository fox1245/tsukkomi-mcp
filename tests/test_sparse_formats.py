"""Sparse index formats in the engine: the compact format is the default, it fills lazily like any missing derived index, the
legacy ``sparse.sqlite`` is never read for results, written, truncated or deleted, a rebuild is bounded and resumable, the
old-SQLite path is an explicit error or a visible fallback, and ``audit_status`` shows both files."""
from __future__ import annotations

import json
import os
import sqlite3
import time

import pytest

from self_directing_mcp import engine as engine_module
from self_directing_mcp import maintenance
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index import sparse as sparse_module
from self_directing_mcp.index.sparse import SparseFormatError, SparseIndex
from sparse_support import sha256
from test_reliability import SID, message, write

WORDS = ["alpha", "beta", "gamma"]


def make(tmp_path, monkeypatch, *, events=14, ready=True, **settings):
    for name in ("SELF_DIRECT_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER", "SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR",
                 "SELF_DIRECT_SPARSE_FORMAT"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "sessions"
    root.mkdir(exist_ok=True)
    path = root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl"
    if not path.exists():
        write(path, [{"type": "session_meta", "payload": {"id": SID}}]
              + [message(f"{WORDS[i % 3]} pre-commit event_number{i} uniqueword{i} " + "filler text " * (i % 5)) for i in range(events)])
    engine = SelfDirectEngine(Settings(codex_sessions_dir=root, index_dir=tmp_path / "index", use_fake_embedder=True,
                                       _env_file=None, **settings))
    if ready:
        engine.ensure_ready()
    return engine, path


def stat_key(path):
    info = path.stat()
    return sha256(path), info.st_size, info.st_mtime_ns


def test_the_default_is_the_compact_format_in_its_own_file(tmp_path, monkeypatch):
    engine, _ = make(tmp_path, monkeypatch)
    try:
        assert engine.sync_session(session_id=SID, embed=False)["ok"]
        index = tmp_path / "index"
        assert engine.sparse.format == "compact" and engine.sparse.db_path == index / "sparse-v2.sqlite"
        assert (index / "sparse-v2.sqlite").exists() and not (index / "sparse.sqlite").exists()
        assert engine.sparse.count() == engine.store.chunk_count() > 10
        assert engine.search_history("alpha", session_id=SID, mode="sparse")["hits"]
    finally:
        engine.close()


def test_a_legacy_index_is_never_touched_and_never_answers(tmp_path, monkeypatch):
    legacy, _ = make(tmp_path, monkeypatch, sparse_format="legacy")
    try:
        legacy.sync_session(session_id=SID, embed=False)
        legacy.sparse.upsert("codex:ghost:0:0:0", SID, "turn", "legacyonlyword appears nowhere in the canonical store")
        total = legacy.store.chunk_count()
        assert legacy.sparse.format == "legacy" and legacy.sparse.count() == total + 1
        assert legacy.search_history("legacyonlyword", session_id=SID, mode="sparse")["hits"] == []   # not canonical: filtered
    finally:
        legacy.close()
    old = tmp_path / "index" / "sparse.sqlite"
    before = stat_key(old)

    engine, path = make(tmp_path, monkeypatch)                      # the default: compact, same index directory
    try:
        assert engine.sparse.format == "compact" and engine.sparse.count() == 0
        assert stat_key(old) == before
        status = engine.audit_status()
        assert status["sparse_counts"] == {"legacy": total + 1, "compact": 0} and stat_key(old) == before
        # Until the session is in the new file nothing is answered from the old one: the repair path runs first.
        found = engine.search_history("alpha", session_id=SID, mode="sparse")
        assert found["hits"] and engine.sparse.count() == total
        assert engine.search_history("legacyonlyword", session_id=SID, mode="sparse")["hits"] == []
        assert stat_key(old) == before
        write(path, [message("delta appended after the conversion")], "a")
        assert engine.sync_session(session_id=SID, embed=False)["ok"]
        assert engine.search_history("conversion", session_id=SID, mode="sparse")["hits"]
        engine.sparse.clear()
        assert engine.search_history("alpha", session_id=SID, mode="sparse")["hits"]          # repaired again, from the store
        assert engine.rebuild_sparse()["ok"]
        assert engine.audit_status()["sparse_counts"]["legacy"] == total + 1
        assert stat_key(old) == before
    finally:
        engine.close()
    assert stat_key(old) == before


def test_audit_status_shows_the_format_both_files_and_both_counts(tmp_path, monkeypatch):
    legacy, _ = make(tmp_path, monkeypatch, sparse_format="legacy")
    try:
        legacy.sync_session(session_id=SID, embed=False)
        status = legacy.audit_status()
        assert (status["sparse_format"], status["sparse_format_setting"], status["sparse_format_reason"]) == ("legacy", "legacy", None)
        assert "sparse_legacy_fallback" not in status["degraded_flags"]                      # chosen, not a fallback
        assert status["sparse_counts"]["compact"] is None                                    # no such file, and none is created
        assert not (tmp_path / "index" / "sparse-v2.sqlite").exists()
    finally:
        legacy.close()
    engine, _ = make(tmp_path, monkeypatch)
    try:
        engine.sync_session(session_id=SID, embed=False)
        status = engine.audit_status()
        index = tmp_path / "index"
        assert status["sparse_format"] == "compact" and status["sparse_format_setting"] == "auto" and status["sparse_format_reason"] is None
        assert "sparse_legacy_fallback" not in status["degraded_flags"]
        files = {f["name"]: f for f in status["sparse_files"]}
        assert set(files) == {"sparse.sqlite", "sparse-v2.sqlite"}
        assert files["sparse.sqlite"] == {"name": "sparse.sqlite", "format": "legacy", "active": False, "exists": True,
                                          "bytes": (index / "sparse.sqlite").stat().st_size}
        assert files["sparse-v2.sqlite"]["active"] and files["sparse-v2.sqlite"]["bytes"] == (index / "sparse-v2.sqlite").stat().st_size
        assert status["sparse_count"] == status["sparse_counts"]["compact"] == status["sparse_counts"]["legacy"] == status["total_chunks"]
    finally:
        engine.close()


def test_an_old_sqlite_is_an_error_for_compact_and_a_visible_fallback_for_auto(tmp_path, monkeypatch):
    monkeypatch.setattr(sparse_module, "sqlite_version_info", lambda: (3, 42, 0))
    engine, _ = make(tmp_path, monkeypatch, ready=False, sparse_format="compact")
    with pytest.raises(SparseFormatError, match=r"3\.43.*SELF_DIRECT_SPARSE_FORMAT=legacy"):
        engine.sync_session(session_id=SID, embed=False)
    assert not (tmp_path / "index" / "sparse-v2.sqlite").exists()
    engine, _ = make(tmp_path, monkeypatch)
    try:
        engine.sync_session(session_id=SID, embed=False)
        status = engine.audit_status()
        assert status["sparse_format"] == "legacy" and status["sparse_format_setting"] == "auto"
        assert "3.43" in status["sparse_format_reason"] and "sparse_legacy_fallback" in status["degraded_flags"]
        assert (tmp_path / "index" / "sparse.sqlite").exists() and not (tmp_path / "index" / "sparse-v2.sqlite").exists()
        assert engine.search_history("alpha", session_id=SID, mode="sparse")["hits"]
        assert engine.rebuild_sparse()["error"] == "sparse_format_not_compact"
    finally:
        engine.close()


def test_a_repair_that_is_interrupted_keeps_its_batches_and_the_next_search_finishes(tmp_path, monkeypatch):
    engine, _ = make(tmp_path, monkeypatch, events=20)
    try:
        engine.sync_session(session_id=SID, embed=False, derived=False)
        total = engine.store.chunk_count()
        assert engine.sparse.count() == 0 and total >= 20
        monkeypatch.setattr(engine_module, "SPARSE_REPAIR_BATCH", 4)
        calls, real = [], engine.sparse.upsert_many

        def flaky(chunks):
            calls.append(len(list(chunks)) if not isinstance(chunks, list) else len(chunks))
            if len(calls) == 3:
                raise RuntimeError("interrupted")
            return real(chunks)

        monkeypatch.setattr(engine.sparse, "upsert_many", flaky)
        with pytest.raises(RuntimeError, match="interrupted"):
            engine.search_history("alpha", session_id=SID, mode="sparse")           # never a partial answer
        assert engine.sparse.count() == 8                                           # the two finished batches were committed
        monkeypatch.undo()
        assert engine.search_history("alpha", session_id=SID, mode="sparse")["hits"]
        assert engine.sparse.count() == total and engine.sparse.verify_statistics()["ok"]
    finally:
        engine.close()


def test_rebuild_is_bounded_resumable_and_idempotent(tmp_path, monkeypatch):
    engine, _ = make(tmp_path, monkeypatch, events=23)
    try:
        engine.sync_session(session_id=SID, embed=False, derived=False)
        total = engine.store.chunk_count()
        assert engine.sparse.count() == 0

        seen = []

        def interrupt(done, of, written):
            seen.append((done, of, written))
            if len(seen) == 2:
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            engine.rebuild_sparse(batch=5, progress=interrupt)
        assert seen == [(5, total, 5), (10, total, 10)] and engine.sparse.count() == 10        # finished batches are kept

        info = engine.rebuild_sparse(budget_sec=0.0)                                           # an exhausted budget writes nothing
        assert (info["state"], info["stopped_by"], info["written"], info["remaining"]) == ("partial", "time_budget", 0, total - 10)

        info = engine.rebuild_sparse(budget_sec=0.05, batch=4, progress=lambda *a: time.sleep(0.08))
        assert (info["state"], info["stopped_by"], info["written"]) == ("partial", "time_budget", 4)
        assert info["already_indexed"] == 10 and engine.sparse.count() == 14

        info = engine.rebuild_sparse(batch=5)
        assert info["state"] == "complete" and info["written"] == total - 14 and info["already_indexed"] == 14
        assert info["statistics"]["ok"] and engine.sparse.count() == total
        before = stat_key(engine.sparse.db_path)
        info = engine.rebuild_sparse()
        assert (info["state"], info["written"], info["remaining"]) == ("complete", 0, 0)       # a finished index is left alone
        engine.sparse._conn.close()
        assert sha256(engine.sparse.db_path) == before[0]
    finally:
        engine.close()


def test_both_formats_answer_alike_through_the_engine(tmp_path, monkeypatch):
    results = {}
    for fmt in ("legacy", "compact"):
        monkeypatch.setenv("SELF_DIRECT_SPARSE_FORMAT", fmt)
        root = tmp_path / "sessions"
        root.mkdir(exist_ok=True)
        path = root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl"
        if not path.exists():
            write(path, [{"type": "session_meta", "payload": {"id": SID}}]
                  + [message(f"{WORDS[i % 3]} pre-commit event_number{i} " + ("ok" if i % 4 == 0 else "uniqueword%d filler " % i * (1 + i % 5))) for i in range(60)])
        engine = SelfDirectEngine(Settings(codex_sessions_dir=root, index_dir=tmp_path / f"index-{fmt}", use_fake_embedder=True, _env_file=None))
        try:
            engine.sync_session(session_id=SID, embed=True)
            assert engine.sparse.format == fmt
            out = []
            for query in ("alpha", "pre-commit", "event_number7 beta", "ok", "uniqueword9", "nothing-matches-this"):
                for mode in ("sparse", "hybrid"):
                    for k in (3, 20):
                        hits = engine.search_history(query, session_id=SID, mode=mode, top_k=k)["hits"]
                        out.append((query, mode, k, [(h["chunk_id"], h["sparse_rank"], h["sparse_score"], h["dense_rank"]) for h in hits]))
            results[fmt] = out
        finally:
            engine.close()
    assert results["legacy"] == results["compact"]
    assert sum(1 for *_x, hits in results["compact"] if hits) >= 15


def test_the_rebuild_command(tmp_path, monkeypatch, capsys):
    engine, _ = make(tmp_path, monkeypatch, events=18)
    try:
        engine.sync_session(session_id=SID, embed=False, derived=False)
        total = engine.store.chunk_count()
    finally:
        engine.close()
    monkeypatch.setenv("SELF_DIRECT_CODEX_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("SELF_DIRECT_USE_FAKE_EMBEDDER", "true")
    index = tmp_path / "index"
    argv = ["sparse-rebuild", "--index-dir", str(index)]

    assert maintenance.main([*argv, "--budget-sec", "0"]) == maintenance.EXIT_PARTIAL
    out, err = capsys.readouterr()
    assert json.loads(out)["stopped_by"] == "time_budget" and "run the command again" in err

    assert maintenance.main([*argv, "--batch", "5"]) == 0
    out, err = capsys.readouterr()
    info = json.loads(out)
    assert info["state"] == "complete" and info["written"] == total and info["statistics"]["ok"]
    assert f"{total}/{total} events" in err and "sparse-rebuild:" in err                      # progress goes to stderr
    assert not (index / "sparse.sqlite").exists()                                              # nothing legacy was created

    key = stat_key(index / "sparse-v2.sqlite")
    assert maintenance.main(argv) == 0
    assert json.loads(capsys.readouterr()[0])["written"] == 0 and stat_key(index / "sparse-v2.sqlite")[0] == key[0]

    assert maintenance.main([*argv, "--format", "legacy"]) == maintenance.EXIT_FAILED
    assert "legacy" in capsys.readouterr()[1]
    assert not (index / "sparse.sqlite").exists()
    with pytest.raises(SystemExit):
        maintenance.main(["sparse-rebuild", "--batch", "0"])


def test_the_rebuild_command_reports_an_old_sqlite(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sparse_module, "sqlite_version_info", lambda: (3, 40, 0))
    monkeypatch.setenv("SELF_DIRECT_USE_FAKE_EMBEDDER", "true")
    code = maintenance.main(["sparse-rebuild", "--index-dir", str(tmp_path / "idx"), "--format", "compact"])
    assert code == maintenance.EXIT_FAILED and "SELF_DIRECT_SPARSE_FORMAT=legacy" in capsys.readouterr()[1]
    assert not (tmp_path / "idx" / "sparse-v2.sqlite").exists()
