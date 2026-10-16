"""The CJK sparse format in the engine: which file ``auto`` picks and why, the states ``ready`` and ``building``, a tokenization
this version does not know, the old SQLite, the rebuild (``sparse-rebuild --format cjk``: resumable, a catch-up pass, ready only
when the counts agree), ``audit_status``, the old files staying byte-identical, and a compound's inner word found through the
real engine. The format itself (pre-tokenizer, query builder, round trips, statistics) is in ``test_sparse_cjk.py``."""
from __future__ import annotations

import json
import sqlite3

import pytest

from self_directing_mcp import engine as engine_module
from self_directing_mcp import maintenance
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index import sparse as sparse_module
from self_directing_mcp.index.sparse import (
    CJK_FILE, CjkSparseIndex, CompactSparseIndex, SparseFormatError, SparseIndex, choose_auto_format, look_at_sparse_file, open_sparse_index,
)
from sparse_support import Doc, sha256
from test_reliability import SID, message, write
from test_sparse_formats import stat_key

pytestmark = pytest.mark.skipif(sparse_module.compact_support() is not None,
                                reason=f"this SQLite cannot use the compact format: {sparse_module.compact_support()}")

TEXTS = ["원탁회의를 열었다", "v2색인을 다시 만들었다", "검색 색인 점검 끝", "plain english words only", "日本語のテキストを検索", "내 집에 갔다"]
QUERIES = ("회의", "색인", "검색", "english", "テキスト", "집", "원탁회의를", "index")


def events(count):
    return TEXTS + [f"이벤트 {i} 회의록 검색어{i} 색인 filler" for i in range(count - len(TEXTS))]


def world(tmp_path, monkeypatch, *, name="", count=14, build=False, ready=True, **settings):
    """An engine over a small synthetic Korean session in ``tmp_path/name``. The format is whatever ``settings`` says."""
    for env in ("SELF_DIRECT_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER", "SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR",
                "SELF_DIRECT_SPARSE_FORMAT"):
        monkeypatch.delenv(env, raising=False)
    base = tmp_path / name
    root = base / "sessions"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl"
    if not path.exists():
        write(path, [{"type": "session_meta", "payload": {"id": SID}}] + [message(text) for text in events(count)])
    engine = SelfDirectEngine(Settings(codex_sessions_dir=root, index_dir=base / "index", use_fake_embedder=True, _env_file=None, **settings),
                              sparse_build=build)
    if ready:
        engine.ensure_ready()
    return engine, path, base / "index"


def seeded(directory, **made):
    """Sparse files made directly: ``legacy``, ``compact``, ``cjk`` (ready), ``building`` (state building) with the given texts."""
    docs = [Doc(f"codex:s:{i}:0:{i:016x}", "s", "turn", text) for i, text in enumerate(TEXTS)]
    directory.mkdir(parents=True, exist_ok=True)
    for kind, count in made.items():
        index = {"legacy": lambda: SparseIndex(directory / "sparse.sqlite"),
                 "compact": lambda: CompactSparseIndex(directory / "sparse-v2.sqlite"),
                 "cjk": lambda: CjkSparseIndex(directory / CJK_FILE),
                 "building": lambda: CjkSparseIndex(directory / CJK_FILE, build=True)}[kind]()
        if count:
            index.upsert_many(docs[:count])
        index._conn.commit()
        index._conn.close()


# ---------------------------------------------------------------------------------------------------- what auto picks
def test_auto_on_a_fresh_install_creates_the_cjk_file(tmp_path, monkeypatch):
    engine, _path, index = world(tmp_path, monkeypatch, sparse_format="auto")
    try:
        assert engine.sync_session(session_id=SID, embed=False)["ok"]
        assert engine.sparse.format == "cjk" and engine.sparse.db_path == index / "sparse-v3.sqlite"
        assert engine.sparse.meta() == {"format": "compact-cjk-1", "tokenization": "cjk-bigram-1", "state": "ready", "sqlite_version": sqlite3.sqlite_version}
        assert [p.name for p in index.glob("sparse*.sqlite")] == ["sparse-v3.sqlite"]            # no other sparse file is even created
        status = engine.audit_status()
        assert (status["sparse_format"], status["sparse_format_setting"], status["sparse_format_reason"]) == ("cjk", "auto", None)
        assert "no sparse file holds data" in status["sparse_format_choice"] and status["sparse_cjk_hint"] is None
        assert not [flag for flag in status["degraded_flags"] if flag.startswith("sparse")]
    finally:
        engine.close()


def test_auto_picks_the_ready_cjk_file_even_next_to_older_files(tmp_path):
    seeded(tmp_path, legacy=3, compact=4, cjk=2)
    fmt, details = choose_auto_format(tmp_path)
    assert fmt == "cjk" and "sparse-v3.sqlite is ready" in details["choice"] and details["cjk_problem"] is False
    index, resolution = open_sparse_index(tmp_path, "auto")
    assert (index.format, resolution["format"], resolution["reason"]) == ("cjk", "cjk", None)


@pytest.mark.parametrize("older", [{"compact": 4}, {"legacy": 3}, {"legacy": 3, "compact": 2}])
def test_auto_never_starts_the_cjk_format_over_an_index_that_holds_data(tmp_path, older):
    """A v2 or legacy file with documents is kept (a legacy-only index moves to the compact file, as in 0.8.0); sparse-v3.sqlite
    is not created, and the hint names the command that builds it."""
    seeded(tmp_path, **older)
    before = {p.name: stat_key(p) for p in tmp_path.glob("sparse*.sqlite")}
    index, resolution = open_sparse_index(tmp_path, "auto")
    index._conn.close()
    assert index.format == "compact" and resolution["setting"] == "auto" and resolution["reason"] is None
    assert not (tmp_path / CJK_FILE).exists()
    assert "already holds data" in resolution["choice"] and "--format cjk" in resolution["cjk_hint"] and resolution["cjk_problem"] is False
    for name, key in before.items():
        if name != "sparse-v2.sqlite" or "compact" in older:
            assert stat_key(tmp_path / name) == key, name                       # opening the file in use does not even rewrite it


def test_empty_older_files_do_not_hold_the_cjk_format_back(tmp_path):
    seeded(tmp_path, legacy=0, compact=0)                                       # engines ran, nothing was indexed yet
    index, resolution = open_sparse_index(tmp_path, "auto")
    index._conn.close()
    assert index.format == "cjk" and resolution["cjk_hint"] is None


def test_an_unreadable_older_file_counts_as_holding_data(tmp_path):
    (tmp_path / "sparse-v2.sqlite").write_bytes(b"this is not a database" * 20)
    fmt, details = choose_auto_format(tmp_path)
    assert fmt == "compact" and "already holds data" in details["choice"]       # never guess: the cjk format is not started


def test_a_building_file_is_not_used_by_auto_and_the_fallback_is_visible(tmp_path):
    seeded(tmp_path, building=2, compact=3)
    before = stat_key(tmp_path / CJK_FILE)
    fmt, details = choose_auto_format(tmp_path)
    assert fmt == "compact" and details["cjk_problem"] is True
    assert "not finished" in details["choice"] and "sparse-rebuild --format cjk" in details["cjk_hint"]
    assert choose_auto_format(tmp_path, build=True)[0] == "cjk"                  # the rebuild command continues it
    index, resolution = open_sparse_index(tmp_path, "auto")
    index._conn.close()
    assert index.format == "compact" and resolution["cjk_problem"] is True
    assert stat_key(tmp_path / CJK_FILE) == before                               # looked at read-only, never touched


def test_explicit_cjk_refuses_a_building_file_and_names_the_repair(tmp_path):
    seeded(tmp_path, building=2)
    before = stat_key(tmp_path / CJK_FILE)
    with pytest.raises(SparseFormatError, match=r"state 'building'.*sparse-rebuild --format cjk.*SELF_DIRECT_SPARSE_FORMAT=compact"):
        open_sparse_index(tmp_path, "cjk")
    with pytest.raises(SparseFormatError, match="not finished"):
        CjkSparseIndex(tmp_path / CJK_FILE)
    assert stat_key(tmp_path / CJK_FILE) == before
    index = CjkSparseIndex(tmp_path / CJK_FILE, build=True)                      # only the rebuild command opens it
    try:
        assert index.state() == "building" and index.count() == 2
    finally:
        index._conn.close()


def test_a_tokenization_this_version_does_not_know_is_never_answered_from(tmp_path):
    seeded(tmp_path, cjk=3, compact=2)
    with sqlite3.connect(tmp_path / CJK_FILE) as conn:
        conn.execute("UPDATE sparse_meta SET value='cjk-trigram-9' WHERE key='tokenization'")
    conn.close()
    before = stat_key(tmp_path / CJK_FILE)
    with pytest.raises(SparseFormatError, match=r"tokenization 'cjk-trigram-9'.*does not know"):
        open_sparse_index(tmp_path, "cjk")
    with pytest.raises(SparseFormatError, match="does not know"):
        CjkSparseIndex(tmp_path / CJK_FILE, build=True)                           # not even the rebuild converts one tokenization into another
    fmt, details = choose_auto_format(tmp_path)
    assert fmt == "compact" and details["cjk_problem"] and "cjk-trigram-9" in details["choice"]
    assert stat_key(tmp_path / CJK_FILE) == before


def test_a_file_of_another_format_marker_is_left_untouched(tmp_path):
    seeded(tmp_path, cjk=2)
    with sqlite3.connect(tmp_path / CJK_FILE) as conn:
        conn.execute("UPDATE sparse_meta SET value='compact-cjk-2' WHERE key='format'")
    conn.close()
    before = stat_key(tmp_path / CJK_FILE)
    with pytest.raises(SparseFormatError, match="compact-cjk-2"):
        open_sparse_index(tmp_path, "cjk")
    assert choose_auto_format(tmp_path)[0] == "compact"
    assert stat_key(tmp_path / CJK_FILE) == before
    # and the compact class refuses a CJK file, as the CJK class refuses a compact one
    seeded(tmp_path / "other", cjk=1, compact=1)
    with pytest.raises(SparseFormatError, match="not a compact-1"):
        CompactSparseIndex(tmp_path / "other" / CJK_FILE)
    with pytest.raises(SparseFormatError, match="not a compact-cjk-1"):
        CjkSparseIndex(tmp_path / "other" / "sparse-v2.sqlite")


def test_an_old_sqlite_is_an_error_for_cjk_and_a_visible_fallback_for_auto(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sparse_module, "sqlite_version_info", lambda: (3, 42, 0))
    with pytest.raises(SparseFormatError, match=r"3\.43.*SELF_DIRECT_SPARSE_FORMAT=legacy"):
        open_sparse_index(tmp_path / "a", "cjk")
    assert not (tmp_path / "a" / CJK_FILE).exists()
    engine, _path, index = world(tmp_path, monkeypatch, name="b", sparse_format="auto")
    try:
        status = engine.audit_status()
        assert status["sparse_format"] == "legacy" and "3.43" in status["sparse_format_reason"] and "sparse_legacy_fallback" in status["degraded_flags"]
        assert "cannot use the compact formats" in status["sparse_format_choice"]
        assert not (index / CJK_FILE).exists() and (index / "sparse.sqlite").exists()
    finally:
        engine.close()
    code = maintenance.main(["sparse-rebuild", "--index-dir", str(tmp_path / "c"), "--format", "cjk"])
    assert code == maintenance.EXIT_FAILED and "SELF_DIRECT_SPARSE_FORMAT=legacy" in capsys.readouterr()[1]
    assert not (tmp_path / "c" / CJK_FILE).exists()


def test_the_engine_refuses_a_building_file_and_auto_falls_back_visibly(tmp_path, monkeypatch):
    engine, _path, index = world(tmp_path, monkeypatch, sparse_format="cjk", build=True)
    try:
        engine.sync_session(session_id=SID, embed=False, derived=False)
        assert engine.rebuild_sparse(batch=4, budget_sec=0.0)["state"] == "partial"      # the file is new: state building
        assert look_at_sparse_file(index / CJK_FILE)["state"] == "building"
    finally:
        engine.close()
    before = stat_key(index / CJK_FILE)
    explicit, _p, _i = world(tmp_path, monkeypatch, sparse_format="cjk", ready=False)
    with pytest.raises(SparseFormatError, match="sparse-rebuild --format cjk"):
        explicit.sync_session(session_id=SID, embed=False)
    auto, _p, _i = world(tmp_path, monkeypatch, sparse_format="auto")
    try:
        status = auto.audit_status()
        assert status["sparse_format"] == "compact" and "sparse_cjk_unavailable" in status["degraded_flags"]
        assert "not finished" in status["sparse_format_choice"] and "--format cjk" in status["sparse_cjk_hint"]
        assert auto.search_history("회의", session_id=SID, mode="sparse")["hits"] is not None      # answered from sparse-v2.sqlite
        files = {f["name"]: f for f in status["sparse_files"]}
        assert files["sparse-v3.sqlite"]["state"] == "building" and files["sparse-v3.sqlite"]["active"] is False
    finally:
        auto.close()
    assert stat_key(index / CJK_FILE) == before


# ---------------------------------------------------------------------------------------------------- the rebuild
def build_engine(tmp_path, monkeypatch, **kwargs):
    engine, path, index = world(tmp_path, monkeypatch, sparse_format="cjk", build=True, **kwargs)
    engine.sync_session(session_id=SID, embed=False, derived=False)                 # the canonical store only; the index is built below
    return engine, path, index


def snapshot(index_path, queries=QUERIES):
    sparse = CjkSparseIndex(index_path)
    try:
        return {q: sparse.search(q, 50) for q in queries}, sparse.verify_statistics()
    finally:
        sparse._conn.close()


def test_a_rebuild_is_building_until_it_finishes_and_a_killed_one_resumes_to_the_same_index(tmp_path, monkeypatch):
    engine, _path, index = build_engine(tmp_path, monkeypatch, count=23)
    try:
        total = engine.store.chunk_count()
        assert engine.sparse.meta()["state"] == "building" and engine.sparse.count() == 0
        seen = []

        def killed(done, of, written):
            seen.append((done, of, written, look_at_sparse_file(index / CJK_FILE)["state"]))
            if len(seen) == 2:
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            engine.rebuild_sparse(batch=5, progress=killed)
        assert seen == [(5, total, 5, "building"), (10, total, 10, "building")]
        assert look_at_sparse_file(index / CJK_FILE) == {"documents": 10, "format": "compact-cjk-1", "tokenization": "cjk-bigram-1", "state": "building"}
        info = engine.rebuild_sparse(batch=5)
        assert info["state"] == "complete" and info["already_indexed"] == 10 and info["written"] == total - 10
        assert info["index_state"] == "ready" and info["tokenization"] == "cjk-bigram-1" and info["statistics"]["ok"]
        assert info["catch_up_written"] == 0 and engine.sparse.count() == total
        before = stat_key(index / CJK_FILE)
        again = engine.rebuild_sparse()
        assert (again["state"], again["written"], again["remaining"], again["index_state"]) == ("complete", 0, 0, "ready")
        assert "catch_up_written" not in again
        engine.sparse._conn.close()
        assert sha256(index / CJK_FILE) == before[0]                                # a finished index is left alone
    finally:
        engine.close()
    resumed = snapshot(index / CJK_FILE)
    reference, _p, ref_index = build_engine(tmp_path, monkeypatch, name="ref", count=23)
    try:
        assert reference.rebuild_sparse(batch=5)["index_state"] == "ready"
    finally:
        reference.close()
    assert snapshot(ref_index / CJK_FILE) == resumed                                # same ids, ranks, scores and statistics
    assert sum(map(bool, resumed[0].values())) >= 6


def test_the_rebuild_stays_building_when_the_budget_runs_out(tmp_path, monkeypatch):
    engine, _path, index = build_engine(tmp_path, monkeypatch)
    try:
        info = engine.rebuild_sparse(budget_sec=0.0)
        assert (info["state"], info["stopped_by"], info["written"], info["index_state"]) == ("partial", "time_budget", 0, "building")
        assert look_at_sparse_file(index / CJK_FILE)["state"] == "building"
    finally:
        engine.close()


def test_finish_build_is_ready_only_when_the_count_matches(tmp_path):
    seeded(tmp_path, building=3)
    index = CjkSparseIndex(tmp_path / CJK_FILE, build=True)
    try:
        assert index.finish_build(4) is False and index.state() == "building"
        assert index.finish_build(3) is True and index.state() == "ready"
        key = stat_key(tmp_path / CJK_FILE)
        index.set_state("ready")                                                 # nothing is written when it already holds the value
        index._conn.commit()
        assert stat_key(tmp_path / CJK_FILE)[0] == key[0]
        with pytest.raises(ValueError):
            index.set_state("done")
    finally:
        index._conn.close()


def test_a_catch_up_pass_picks_up_events_published_while_the_batches_ran(tmp_path, monkeypatch):
    engine, path, index = build_engine(tmp_path, monkeypatch, count=16)
    try:
        fired = []

        def publish_more(done, total, written):
            if fired:
                return
            fired.append(1)
            write(path, [message(f"늦게 온 이벤트 {i} 회의 마무리") for i in range(3)], "a")
            other, _p, _i = world(tmp_path, monkeypatch, sparse_format="compact")      # another process publishes into the same store
            try:
                assert other.sync_session(session_id=SID, embed=False, derived=False)["ok"]
            finally:
                other.close()

        total = engine.store.chunk_count()
        info = engine.rebuild_sparse(batch=5, progress=publish_more)
        assert fired and engine.store.chunk_count() == total + 3
        assert info["state"] == "complete" and info["catch_up_written"] == 3 and info["index_state"] == "ready"
        assert engine.sparse.count() == engine.store.chunk_count() and engine.sparse.verify_statistics()["ok"]
        assert {h[0] for h in engine.sparse.search("마무리", 50)} and engine.sparse.state() == "ready"
    finally:
        engine.close()


def test_the_index_is_not_ready_while_the_counts_disagree(tmp_path, monkeypatch):
    engine, _path, index = build_engine(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(engine_module, "SPARSE_CATCH_UP_ROUNDS", 2)
        monkeypatch.setattr(CjkSparseIndex, "finish_build", lambda self, documents: False)     # the store always moved on
        info = engine.rebuild_sparse(batch=4)
        assert (info["state"], info["stopped_by"], info["index_state"]) == ("partial", "catch_up", "building")
        assert look_at_sparse_file(index / CJK_FILE)["state"] == "building"
        monkeypatch.undo()
    finally:
        engine.close()


def test_a_ready_file_is_topped_up_and_never_demoted_to_building(tmp_path, monkeypatch):
    engine, _path, index = world(tmp_path, monkeypatch, count=20, sparse_format="cjk")      # lazily filled: ready from the start
    try:
        engine.sync_session(session_id=SID, embed=False, derived=False)
        assert engine.sparse.state() == "ready" and engine.sparse.count() == 0
        states = []
        info = engine.rebuild_sparse(batch=6, progress=lambda *a: states.append(look_at_sparse_file(index / CJK_FILE)["state"]))
        assert states and set(states) == {"ready"} and info["index_state"] == "ready" and info["written"] == engine.store.chunk_count()
    finally:
        engine.close()


def test_orphans_of_the_cjk_index_are_deleted_like_those_of_the_compact_one(tmp_path, monkeypatch):
    engine, _path, _index = world(tmp_path, monkeypatch, sparse_format="cjk")
    try:
        engine.sync_session(session_id=SID, embed=False)
        total = engine.store.chunk_count()
        engine.sparse.upsert("codex:0-ghost:0:0:0", "ghost", "turn", "유령 회의 문서")
        status = engine.audit_status()
        assert status["sparse_extra_ids"] == 1 and status["sparse_counts"]["cjk"] == total + 1
        info = engine.rebuild_sparse()
        assert (info["orphans"], info["orphans_removed"], info["state"]) == (1, 1, "complete") and info["statistics"]["ok"]
        assert engine.sparse.count() == total
        engine.sparse.upsert("codex:0-ghost:0:0:1", "ghost", "turn", "유령 회의 문서")
        assert engine.search_history("유령", mode="sparse")["hits"] == []            # not canonical: never an answer ...
        assert engine.sparse.count() == total                                          # ... and a search over all sessions prunes it
    finally:
        engine.close()


def command_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SELF_DIRECT_CODEX_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("SELF_DIRECT_USE_FAKE_EMBEDDER", "true")
    return ["sparse-rebuild", "--index-dir", str(tmp_path / "index")]


def test_the_rebuild_command_builds_the_cjk_file(tmp_path, monkeypatch, capsys):
    engine, _path, index = world(tmp_path, monkeypatch, count=18, sparse_format="compact")
    try:
        engine.sync_session(session_id=SID, embed=False, derived=False)
        total = engine.store.chunk_count()
    finally:
        engine.close()
    argv = command_env(tmp_path, monkeypatch)
    old = stat_key(index / "sparse-v2.sqlite")                                         # an empty v2 file: engines ran, nothing indexed

    assert maintenance.main([*argv, "--format", "cjk", "--budget-sec", "0"]) == maintenance.EXIT_PARTIAL
    out, err = capsys.readouterr()
    assert json.loads(out)["stopped_by"] == "time_budget" and json.loads(out)["index_state"] == "building" and "run the command again" in err
    assert look_at_sparse_file(index / CJK_FILE)["state"] == "building"

    assert maintenance.main([*argv, "--format", "cjk", "--batch", "5"]) == 0
    out, err = capsys.readouterr()
    info = json.loads(out)
    assert info["state"] == "complete" and info["written"] == total and info["index_state"] == "ready" and info["statistics"]["ok"]
    assert (info["format"], info["tokenization"], info["file"]) == ("cjk", "cjk-bigram-1", str(index / CJK_FILE))
    assert f"{total}/{total} events" in err
    assert not (index / "sparse.sqlite").exists() and stat_key(index / "sparse-v2.sqlite") == old        # the other files are not touched

    key = stat_key(index / CJK_FILE)
    assert maintenance.main([*argv, "--format", "cjk"]) == 0
    assert json.loads(capsys.readouterr()[0])["written"] == 0 and stat_key(index / CJK_FILE)[0] == key[0]
    assert maintenance.main([*argv, "--format", "compact"]) == 0                     # the compact file can be built too; the CJK one stays
    assert json.loads(capsys.readouterr()[0])["written"] == total and stat_key(index / CJK_FILE) == key

    auto = open_sparse_index(index, "auto")[0]
    try:
        assert auto.format == "cjk" and auto.count() == total
    finally:
        auto._conn.close()


def test_a_plain_sparse_rebuild_finishes_an_interrupted_cjk_build(tmp_path, monkeypatch, capsys):
    engine, _path, index = world(tmp_path, monkeypatch, count=18, sparse_format="compact")
    try:
        engine.sync_session(session_id=SID, embed=False, derived=False)
        total = engine.store.chunk_count()
    finally:
        engine.close()
    argv = command_env(tmp_path, monkeypatch)
    real, calls = SelfDirectEngine._fill_sparse, []

    def killed(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt                                                   # no cleanup, as when the process is killed
        return real(self, *args, **kwargs)

    monkeypatch.setattr(SelfDirectEngine, "_fill_sparse", killed)
    assert maintenance.main([*argv, "--format", "cjk", "--batch", "5"]) == maintenance.EXIT_INTERRUPTED
    monkeypatch.setattr(SelfDirectEngine, "_fill_sparse", real)
    capsys.readouterr()
    state = look_at_sparse_file(index / CJK_FILE)
    assert state["state"] == "building" and 0 < state["documents"] < total
    assert maintenance.main(argv) == 0                                                # no --format: auto continues the build in progress
    info = json.loads(capsys.readouterr()[0])
    assert info["format"] == "cjk" and info["index_state"] == "ready" and info["already_indexed"] + info["written"] == total


# ---------------------------------------------------------------------------------------------------- audit_status and the old files
def test_audit_status_describes_every_file(tmp_path, monkeypatch):
    seeded(tmp_path / "index", legacy=3, compact=4)
    engine, _path, index = world(tmp_path, monkeypatch, sparse_format="cjk")
    try:
        engine.sync_session(session_id=SID, embed=False)
        status = engine.audit_status()
        files = {f["name"]: f for f in status["sparse_files"]}
        assert set(files) == {"sparse.sqlite", "sparse-v2.sqlite", "sparse-v3.sqlite"}
        v3 = files["sparse-v3.sqlite"]
        assert v3 == {"name": "sparse-v3.sqlite", "path": str(index / "sparse-v3.sqlite"), "format": "cjk", "active": True, "exists": True,
                      "bytes": (index / "sparse-v3.sqlite").stat().st_size, "tokenization": "cjk-bigram-1", "state": "ready",
                      "documents": status["total_chunks"]}
        assert files["sparse-v2.sqlite"]["documents"] == 4 and files["sparse.sqlite"]["documents"] == 3
        assert files["sparse-v2.sqlite"]["tokenization"] is None and files["sparse-v2.sqlite"]["active"] is False
        assert files["sparse-v2.sqlite"]["state"] is None and files["sparse-v2.sqlite"]["bytes"] > 0
        assert status["sparse_counts"] == {"legacy": 3, "compact": 4, "cjk": status["total_chunks"]} and status["sparse_count"] == status["total_chunks"]
        assert status["sparse_format"] == "cjk" and status["sparse_format_setting"] == "cjk" and status["sparse_format_choice"] == "SELF_DIRECT_SPARSE_FORMAT=cjk"
        assert status["sparse_extra_ids"] == 0
    finally:
        engine.close()


def test_the_older_files_are_byte_identical_after_every_operation(tmp_path, monkeypatch):
    seeded(tmp_path / "index", legacy=3, compact=4)
    index = tmp_path / "index"
    before = {p.name: stat_key(p) for p in index.glob("sparse*.sqlite")}
    assert set(before) == {"sparse.sqlite", "sparse-v2.sqlite"}
    engine, _path, _i = world(tmp_path, monkeypatch, sparse_format="cjk")
    try:
        engine.sync_session(session_id=SID, embed=False)
        engine.audit_status()
        for query in QUERIES:
            engine.search_history(query, session_id=SID, mode="sparse")
        engine.search_history("회의", mode="sparse")
        engine.sparse.clear()
        assert engine.rebuild_sparse(batch=4)["state"] == "complete"
        engine.sparse.delete_ids(sorted(engine.sparse.ids())[:3])
        assert engine.search_history("색인", session_id=SID, mode="sparse")["hits"]
    finally:
        engine.close()
    assert {p.name: stat_key(p) for p in index.glob("sparse*.sqlite") if p.name != "sparse-v3.sqlite"} == before


# ---------------------------------------------------------------------------------------------------- the engine finds what the prefix fix cannot
@pytest.mark.parametrize("local_only", [True, False])
def test_search_history_finds_the_inner_word_of_a_compound(tmp_path, monkeypatch, local_only):
    engine, _path, _index = world(tmp_path, monkeypatch, sparse_format="auto", local_only=local_only)
    try:
        assert engine.sync_session(session_id=SID, embed=not local_only)["ok"]
        assert engine.sparse.format == "cjk"
        hits = engine.search_history("회의", session_id=SID, mode="sparse")["hits"]
        by_regex = engine.search_history("회의", session_id=SID, mode="regex")["hits"]
        assert hits and {h["chunk_id"] for h in hits} == {h["chunk_id"] for h in by_regex}      # exactly what a substring search finds
        assert any("원탁회의를" in h["snippet"] for h in hits)                                    # inside the compound
        glued = engine.search_history("색인", session_id=SID, mode="sparse")["hits"]
        assert any("v2색인을" in h["snippet"] for h in glued)                                    # inside v2색인을
        assert [h["sparse_rank"] for h in glued] == list(range(1, len(glued) + 1))
        assert engine.search_history("テキスト", session_id=SID, mode="sparse")["hits"]
        assert engine.search_history("english", session_id=SID, mode="sparse")["hits"]
        assert engine.search_history("없는말없는말", session_id=SID, mode="sparse")["hits"] == []
        if not local_only:
            hybrid = engine.search_history("회의", session_id=SID, mode="hybrid")["hits"]
            assert any(h["sparse_rank"] for h in hybrid)
    finally:
        engine.close()


def test_the_compact_format_cannot_find_it(tmp_path, monkeypatch):
    """The comparison that motivates the format: on the compact file the same search misses the compound."""
    engine, _path, _index = world(tmp_path, monkeypatch, sparse_format="compact", local_only=True)
    try:
        engine.sync_session(session_id=SID, embed=False)
        compact = {h["chunk_id"] for h in engine.search_history("회의", session_id=SID, mode="sparse")["hits"]}
        regex = {h["chunk_id"] for h in engine.search_history("회의", session_id=SID, mode="regex")["hits"]}
        assert compact < regex                                                        # strictly fewer: 원탁회의를 is missing
    finally:
        engine.close()
