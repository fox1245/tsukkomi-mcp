"""The engine with ``SELF_DIRECT_DENSE_BACKEND=pgvector`` against a real pgvector server (synthetic sessions, the offline
test embedder of the existing suite). Skipped with a reason without a server; a failure with TSUKKOMI_REQUIRE_PGVECTOR=1."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("psycopg", reason="psycopg 3 is not installed (pip install 'psycopg[binary]>=3.2')")

from pgvector_support import TcpProxy
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index.pgvector import PgVectorIndex

pytestmark = pytest.mark.pgvector
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SAMPLE_ROOT = FIXTURES / "sample_session"
SAMPLE_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
SID = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"


def settings(tmp_path, *, dsn_file=None, schema=None, backend="pgvector", **options):
    return Settings(_env_file=None, index_dir=tmp_path / "index", codex_sessions_dir=SAMPLE_ROOT, claude_projects_dir=tmp_path / "projects",
                    use_fake_embedder=True, dense_backend=backend, pgvector_dsn_file=dsn_file, pgvector_schema=schema or "tsukkomi",
                    contracts_path=tmp_path / "index" / "contracts.json", **options)


@pytest.fixture
def world(tmp_path, pgvector_server, pg_schema):
    dsn_file = pgvector_server.write_dsn_file(tmp_path)
    engines = []

    def engine(*, local_only=False, backend="pgvector", directory=None, **options):
        base = directory or tmp_path
        made = SelfDirectEngine(settings(base, dsn_file=dsn_file, schema=pg_schema, backend=backend, local_only=local_only, **options))
        engines.append(made)
        return made
    yield SimpleNamespace(engine=engine, dsn_file=dsn_file, server=pgvector_server, schema=pg_schema, tmp=tmp_path)
    for made in engines:
        made.close()



def table_ids(world):
    return {row[0] for row in world.server.execute(f'SELECT chunk_id FROM "{world.schema}".chunk_vectors')}


def record(n, role, text):
    base = {"type": role, "uuid": f"00000000-0000-4000-8000-{n:012d}", "parentUuid": None, "sessionId": SID,
            "timestamp": f"2026-10-01T00:00:{n:02d}.000Z", "cwd": "/synthetic/project", "isSidechain": False}
    if role == "user":
        base.update(message={"role": "user", "content": text}, origin={"kind": "human"})
    else:
        base.update(message={"id": f"msg_{n}", "role": "assistant", "model": "claude-synthetic",
                             "content": [{"type": "text", "text": text}], "stop_reason": "end_turn"})
    return base


def write(path, *texts, append_from=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    start = append_from or 0
    lines = "".join(json.dumps(record(n, "user" if n % 2 == 0 else "assistant", text)) + "\n"
                    for n, text in enumerate(texts, start=start))
    if append_from is None:
        path.write_text(lines, encoding="utf-8")
    else:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(lines)


def ranking(hits):
    """The hits as score groups: ``[(score, {ids})]``; ids inside one group of equal scores have no defined order."""
    groups: list[tuple[float, set]] = []
    for hit in hits:
        score = hit["dense_score"]
        if groups and abs(groups[-1][0] - score) <= 1e-6:
            groups[-1][1].add(hit["chunk_id"])
        else:
            groups.append((score, {hit["chunk_id"]}))
    return groups


def test_a_session_is_synced_searched_and_reported_through_pgvector(world):
    engine = world.engine()
    synced = engine.sync_session(session_id=SAMPLE_ID)
    assert synced["ok"] is True and synced["dense_backend"] == "pgvector" and synced["embedded"] >= 1
    assert table_ids(world) == set(engine.store.chunk_ids(SAMPLE_ID, "codex")) and len(table_ids(world)) == synced["total_chunks"]
    again = engine.sync_session(session_id=SAMPLE_ID)
    assert again["ok"] and again["embedded"] == 0 and again["new_chunks"] == 0 and again["dense_restored"] == 0
    dense = engine.search_history("shell curl api_key", session_id=SAMPLE_ID, mode="dense", top_k=5)
    hybrid = engine.search_history("shell curl api_key", session_id=SAMPLE_ID, mode="hybrid", top_k=5)
    assert dense["ok"] and hybrid["ok"] and dense["hits"] and hybrid["hits"]
    assert all(hit["dense_score"] is not None for hit in dense["hits"])
    status = engine.audit_status()
    assert status["dense_backend"] == "pgvector" and status["dense_count"] == len(table_ids(world))
    assert status["degraded_flags"] == [] and "dense_error" not in status
    info = status["dense_info"]
    assert info["backend"] == "pgvector" and info["host"] == world.server.host and info["port"] == world.server.port
    assert info["database"] == world.server.dbname and info["schema"] == world.schema and info["dimension"] == 1024
    assert info["extension_version"] and info["server_version"] and info["loopback"] is True and info["tls"] is False
    assert status["vector_extension_version"] == info["extension_version"]
    assert world.server.password not in json.dumps(status) and world.server.user not in info.values()
    assert engine.audit_session(SAMPLE_ID)["ok"] is True            # an audit does not use the dense index at all


def test_dense_search_returns_what_the_numpy_engine_returns(world, tmp_path):
    pg = world.engine()
    numpy_engine = world.engine(backend="numpy", directory=tmp_path / "numpy-side")
    for engine in (pg, numpy_engine):
        assert engine.sync_session(session_id=SAMPLE_ID)["ok"]
    for query in ("shell curl api_key", "delete file rm", "token secret password", "function call tool"):
        for session in (SAMPLE_ID, None):
            a = pg.search_history(query, session_id=session, mode="dense", top_k=10)
            b = numpy_engine.search_history(query, session_id=session, mode="dense", top_k=10)
            assert a["ok"] and b["ok"]
            ga, gb = ranking(a["hits"]), ranking(b["hits"])
            assert len(ga) == len(gb) and all(abs(x[0] - y[0]) <= 1e-6 for x, y in zip(ga, gb))
            assert [x[1] for x in ga[:-1]] == [y[1] for y in gb[:-1]]        # all but a group the top_k may cut through


def test_rewriting_a_session_deletes_the_obsolete_vectors_and_a_hook_leaves_the_cleanup_to_the_server(world):
    path = world.tmp / "projects" / "-synthetic-project" / f"{SID}.jsonl"
    write(path, "first request alpha", "first answer bravo", "second request charlie")
    server = world.engine()
    assert server.sync_session(session_id=SID, provider="claude", path=str(path), embed=True)["ok"]
    before = set(server.store.chunk_ids(SID, "claude"))
    assert table_ids(world) == before and len(before) >= 3
    server.close()

    write(path, "rewritten request delta", "rewritten answer echo")                 # not an append: a full reparse
    hook = world.engine(local_only=True)                                            # what a lifecycle hook runs
    try:
        assert hook.dense is None
        assert hook.sync_session(session_id=SID, provider="claude", path=str(path), embed=False)["ok"]
        obsolete = before - set(hook.store.chunk_ids(SID, "claude"))
        assert obsolete and set(hook.store.pending_deletions(SID, "claude")) >= obsolete
        assert table_ids(world) == before                                           # the hook never touched the server
    finally:
        hook.close()

    server = world.engine()
    assert server.sync_session(session_id=SID, provider="claude", path=str(path), embed=True)["ok"]
    current = set(server.store.chunk_ids(SID, "claude"))
    assert table_ids(world) == current and not (obsolete & table_ids(world))
    assert server.store.pending_deletions(SID, "claude") == []
    found = server.search_history("rewritten request delta", session_id=SID, provider="claude", mode="dense", top_k=3)
    assert found["ok"] and {hit["chunk_id"] for hit in found["hits"]} <= current


def test_a_second_engine_sees_what_the_first_indexed_and_a_wrong_dimension_is_reported(world, tmp_path):
    first = world.engine()
    assert first.sync_session(session_id=SAMPLE_ID)["ok"]
    expected = len(table_ids(world))
    second = world.engine(directory=tmp_path / "other-machine")
    assert second.audit_status()["dense_count"] == expected                         # one dense index, two processes
    other = world.engine(directory=tmp_path / "small", embedding_dim=8)
    other_status = other.audit_status()
    assert other_status["dense_count"] is None and "dense_backend_unavailable" in other_status["degraded_flags"]
    assert "holds 1024-dimensional vectors but this process uses 8" in other_status["dense_error"]
    assert other.sync_session(session_id=SAMPLE_ID)["error"] == "dense_backend_unavailable"
    assert len(table_ids(world)) == expected                                        # nothing was written


def test_an_unreachable_server_is_reported_and_no_other_backend_is_used(world, tmp_path):
    closed = tmp_path / "closed-dsn"
    closed.write_text(f"host=127.0.0.1 port=1 dbname=x user=u password={world.server.password} connect_timeout=2\n")
    closed.chmod(0o600)
    engine = SelfDirectEngine(settings(tmp_path, dsn_file=closed, schema=world.schema))
    try:
        synced = engine.sync_session(session_id=SAMPLE_ID)
        assert synced["ok"] is False and synced["error"] == "dense_backend_unavailable"
        assert "127.0.0.1:1/x" in synced["message"] and world.server.password not in json.dumps(synced)
        assert engine.store.chunk_count() > 0                                       # the canonical events were published
        for mode in ("dense", "hybrid"):
            assert engine.search_history("curl", session_id=SAMPLE_ID, mode=mode)["error"] == "dense_backend_unavailable"
        assert engine.search_history("curl", session_id=SAMPLE_ID, mode="sparse")["ok"] is True
        assert engine.audit_session(SAMPLE_ID)["ok"] is True                        # audits never need the dense index
        status = engine.audit_status()
        assert status["dense_backend"] == "pgvector" and status["dense_count"] is None and "dense_error" in status
        assert "dense_backend_unavailable" in status["degraded_flags"]
        assert "dense_numpy_fallback" not in status["degraded_flags"] and world.server.password not in json.dumps(status)
        assert isinstance(engine.dense, PgVectorIndex)
        assert not list((tmp_path / "index").glob("**/dense_numpy.npz")) and not list((tmp_path / "index").glob("**/dense.sqlite"))
    finally:
        engine.close()


def test_a_server_that_goes_away_between_syncs_loses_nothing_and_the_next_sync_catches_up(world, tmp_path):
    proxy = TcpProxy(world.server.host, world.server.port)
    dsn = tmp_path / "via-proxy"
    dsn.write_text(world.server.conninfo(port=proxy.port, connect_timeout=2) + "\n")
    dsn.chmod(0o600)
    path = tmp_path / "projects" / "-synthetic-project" / f"{SID}.jsonl"
    write(path, "first request alpha", "first answer bravo")
    engine = SelfDirectEngine(settings(tmp_path, dsn_file=dsn, schema=world.schema))
    try:
        assert engine.sync_session(session_id=SID, provider="claude", path=str(path), embed=True)["ok"]
        stored = len(table_ids(world))
        proxy.stop()
        write(path, "second request charlie", "second answer delta", append_from=2)
        down = engine.sync_session(session_id=SID, provider="claude", path=str(path), embed=True)
        assert down["ok"] is False and down["error"] == "dense_backend_unavailable"
        assert "SELF_DIRECT_PGVECTOR_DSN_FILE" in down["message"] and world.server.password not in json.dumps(down)
        assert len(engine.store.chunk_ids(SID, "claude")) > stored                  # the new events are indexed locally
        assert len(table_ids(world)) == stored                                      # and owed to the server
        proxy.start()
        caught_up = engine.sync_session(session_id=SID, provider="claude", path=str(path), embed=True)
        assert caught_up["ok"] is True
        assert table_ids(world) == set(engine.store.chunk_ids(SID, "claude"))
        assert engine.audit_status()["degraded_flags"] == []
    finally:
        engine.close()
        proxy.stop()


def test_the_derived_free_sync_of_a_hook_never_connects(world, tmp_path):
    closed = tmp_path / "closed-dsn"
    closed.write_text("host=127.0.0.1 port=1 dbname=x connect_timeout=2\n")
    closed.chmod(0o600)
    engine = SelfDirectEngine(settings(tmp_path, dsn_file=closed, schema=world.schema))
    try:
        result = engine.sync_session(session_id=SAMPLE_ID, embed=False, derived=False)
        assert result["ok"] is True
        assert engine.audit_session(SAMPLE_ID)["ok"] is True
        assert engine.dense.describe()["connected"] is False
    finally:
        engine.close()
