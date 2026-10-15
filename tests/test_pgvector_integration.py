"""The pgvector backend against a real pgvector server (a throwaway container, see ``pgvector_support``).

Skipped, with the reason, when Docker or the image is not available; a failure when TSUKKOMI_REQUIRE_PGVECTOR=1. Every
test works in a schema of its own and the data is synthetic."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

pytest.importorskip("psycopg", reason="psycopg 3 is not installed (pip install 'psycopg[binary]>=3.2')")

from dense_support import assert_same_topk, exact_cosine, make_ids, normalized_vectors
from pgvector_support import TcpProxy
from self_directing_mcp.index.pgvector import (
    PLAIN_STORAGE_MAX_DIM, PRESENT_BATCH, UPSERT_BATCH, PgVectorConfigError, PgVectorError, PgVectorIndex,
    PgVectorUnavailable)
from self_directing_mcp.request_control import RequestStopped, request_scope

pytestmark = pytest.mark.pgvector
SRC = str(Path(__file__).resolve().parents[1] / "src")


@pytest.fixture
def make_index(pgvector_server, pg_schema):
    made: list[PgVectorIndex] = []

    def make(dim: int = 16, *, schema: str | None = None, dbname: str | None = None, policy: str | None = "policy-1",
             conninfo: str | None = None, **options) -> PgVectorIndex:
        index = PgVectorIndex(conninfo or pgvector_server.conninfo(dbname), dim=dim, schema=schema or pg_schema,
                              policy=policy, **options)
        made.append(index)
        return index
    yield make
    for index in made:
        index.close()


def table(schema: str) -> str:
    return f'"{schema}"."chunk_vectors"'


def fill(index, count, dim, seed=0, prefix="c"):
    matrix = normalized_vectors(count, dim, seed)
    ids = make_ids(count, prefix)
    for start in range(0, count, 1000):
        index.upsert_many(list(zip(ids[start:start + 1000], matrix[start:start + 1000])))
    return ids, matrix


# --- the round trip ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [8, 1024])
def test_a_round_trip_matches_the_exact_cosine_reference(make_index, dim):
    index = make_index(dim)
    ids, matrix = fill(index, 400, dim, seed=dim)
    assert index.count() == 400 and index.ids() == set(ids) and index.refresh() is None
    for seed in range(5):
        query = normalized_vectors(1, dim, 100 + seed)[0]
        assert_same_topk(index.search(query, 20), ids, exact_cosine(matrix, query), 20, label=f"dim {dim}")
    best = int(np.argmax(exact_cosine(matrix, matrix[7])))
    hits = index.search(matrix[7], 3)
    assert hits[0][0] == ids[best] and hits[0][1] == pytest.approx(1.0, abs=1e-6)


def test_scores_have_the_sign_and_scale_of_a_cosine_similarity(make_index):
    index = make_index(4)
    index.upsert_many([("same", [1, 0, 0, 0]), ("orthogonal", [0, 1, 0, 0]), ("opposite", [-1, 0, 0, 0]),
                       ("scaled", [5, 0, 0, 0])])
    scores = dict(index.search(np.array([1, 0, 0, 0], dtype=np.float32), 10))
    assert scores["same"] == pytest.approx(1.0, abs=1e-6) and scores["scaled"] == pytest.approx(1.0, abs=1e-6)
    assert scores["orthogonal"] == pytest.approx(0.0, abs=1e-6) and scores["opposite"] == pytest.approx(-1.0, abs=1e-6)


def test_the_zero_vector_scores_zero_and_ties_break_by_id_in_byte_order(make_index):
    index = make_index(4)
    same = [0.5, 0.5, 0.5, 0.5]
    index.upsert_many([("b", same), ("B", same), ("é", same), ("a", same), ("zero", [0, 0, 0, 0])])
    hits = index.search(np.array(same, dtype=np.float32), 10)
    assert [cid for cid, _ in hits] == ["B", "a", "b", "é", "zero"]            # byte order, like SQLite's BINARY
    assert hits[-1][1] == 0.0 and all(np.isfinite(score) for _cid, score in hits)
    assert [score for _cid, score in index.search(np.zeros(4, dtype=np.float32), 10)] == [0.0] * 5


def test_float32_transport_and_validation(make_index):
    index = make_index(4)
    index.upsert("a", np.array([1, 2, 3, 4], dtype=np.float64))
    with pytest.raises(ValueError):
        index.upsert("b", [1, 2, 3])
    with pytest.raises(ValueError):
        index.upsert("c", [1, 2, 3, float("nan")])
    with pytest.raises(ValueError):
        index.search([1, 2, 3], 3)
    assert index.ids() == {"a"}


# --- writes -------------------------------------------------------------------------------------------------------------

def test_upsert_is_idempotent_overwrites_and_does_not_rewrite_an_identical_row(make_index, pgvector_server, pg_schema):
    index = make_index(8)
    vectors = normalized_vectors(3, 8, 1)
    index.upsert_many([("a", vectors[0]), ("b", vectors[1])])

    def versions():
        return dict(pgvector_server.execute(f"SELECT chunk_id, xmin::text || ':' || ctid::text FROM {table(pg_schema)}"))
    before = versions()
    index.upsert_many([("a", vectors[0]), ("b", vectors[1])])
    assert index.count() == 2 and versions() == before          # the same call twice changes nothing, not even a row version
    index.upsert("a", vectors[2])                               # overwrite
    after = versions()
    assert index.count() == 2 and after["a"] != before["a"] and after["b"] == before["b"]
    assert index.search(vectors[2], 1)[0][0] == "a" and index.search(vectors[2], 1)[0][1] == pytest.approx(1.0, abs=1e-6)


def test_a_batch_larger_than_the_wire_batch_is_one_transaction(make_index, pgvector_server, pg_schema):
    index = make_index(8)
    ids = make_ids(3 * UPSERT_BATCH + 11)
    index.upsert_many(list(zip(ids, normalized_vectors(len(ids), 8, 2))))
    xids = {row[0] for row in pgvector_server.execute(f"SELECT xmin::text FROM {table(pg_schema)}")}
    assert index.count() == len(ids) and len(xids) == 1


def test_delete_present_ids_count_and_clear(make_index):
    index = make_index(8)
    count = 2 * PRESENT_BATCH + 300
    ids, matrix = fill(index, count, 8, seed=3)
    assert index.count() == count
    unknown = [f"missing-{n}" for n in range(3000)]
    assert index.present(ids + unknown) == set(ids) and index.present(unknown) == set() and index.present([]) == set()
    doomed = ids[::2]
    index.delete_ids(doomed + unknown)                          # absent ids are not an error
    assert index.count() == count - len(doomed) and index.present(doomed) == set()
    assert index.ids() == set(ids[1::2])
    index.delete_ids([])
    index.clear()
    assert index.count() == 0 and index.ids() == set() and index.search(matrix[0], 5) == []
    index.upsert("again", matrix[0])
    assert index.ids() == {"again"}


def test_unicode_and_extreme_ids_round_trip(make_index):
    index = make_index(4)
    ids = ["plain", "한글-日本語", "\U0001f600" * 5, "a" * 256, "\U0001d538" * 256, "with space", "q'uote\"s;--"]
    index.upsert_many([(cid, normalized_vectors(1, 4, n)[0]) for n, cid in enumerate(ids)])
    assert index.ids() == set(ids) and index.present(ids) == set(ids)
    index.delete_ids(ids[1:3])
    assert index.ids() == set(ids) - set(ids[1:3])


def test_storage_matches_the_dimension_and_a_full_size_row_fits_a_page(make_index, pgvector_server):
    for dim, storage in ((1024, "p"), (PLAIN_STORAGE_MAX_DIM, "p"), (PLAIN_STORAGE_MAX_DIM + 1, "e"), (1800, "e")):
        schema = "t_store_%d" % dim
        pgvector_server.drop_schema(schema)
        try:
            index = PgVectorIndex(pgvector_server.conninfo(), dim=dim, schema=schema)
            widest = "\U0001d538" * 256                      # 1 kB of UTF-8, the longest id there is
            index.upsert_many([(widest, normalized_vectors(1, dim, 1)[0]), ("x", normalized_vectors(1, dim, 2)[0])])
            assert index.count() == 2 and index.present([widest]) == {widest}
            kind = pgvector_server.execute("SELECT attstorage FROM pg_attribute WHERE attrelid = %s::regclass "
                                           "AND attname = 'embedding'", (table(schema),))[0][0]
            assert kind == storage, dim
            index.close()
        finally:
            pgvector_server.drop_schema(schema)


# --- allowed_ids: the filter is applied before the limit -----------------------------------------------------------------

def test_the_filter_is_applied_before_the_top_k_for_empty_one_and_many_ids(make_index, pgvector_server, pg_schema):
    index = make_index(32)
    ids, matrix = fill(index, 6000, 32, seed=5)
    pgvector_server.execute(f"ANALYZE {table(pg_schema)}")
    query = normalized_vectors(1, 32, 77)[0]
    scores = exact_cosine(matrix, query)
    assert index.search(query, 10, set()) == []
    only = {ids[1234]}
    assert [cid for cid, _ in index.search(query, 10, only)] == [ids[1234]]
    rng = np.random.default_rng(9)
    # the best matches overall are excluded: the answer is the best of the rest, still k long
    best_overall = {ids[n] for n in np.argsort(-scores)[:50]}
    outside = set(ids) - best_overall
    assert_same_topk(index.search(query, 20, outside), ids, scores, 20, allowed=outside, label="all but the best 50")
    for size in (1, 7, 500, 2000, 4500, 6000):                  # lookups below a third of the table, one scan above it
        allowed = {ids[n] for n in rng.choice(len(ids), size=size, replace=False)}
        assert_same_topk(index.search(query, 20, allowed), ids, scores, 20, allowed=allowed, label=f"{size} ids")
    padded = {ids[n] for n in range(0, 6000, 2)} | {f"not-in-the-table-{n}" for n in range(5000)}
    assert_same_topk(index.search(query, 20, padded), ids, scores, 20, allowed=padded, label="ids that are not stored")
    assert len(index.search(query, 10_000)) == 6000 and index.search(query, 0) == [] and index.search(query, -1) == []


def test_ids_that_contain_the_filter_separator_are_filtered_exactly(make_index):
    index = make_index(4)
    ids = ["plain", "a\x1fb", "\x1f", "tail\x1f", "\x1fhead", "other"]
    vectors = normalized_vectors(len(ids), 4, 12)
    index.upsert_many(list(zip(ids, vectors)))
    for allowed in ({"a\x1fb"}, {"\x1f"}, {"tail\x1f", "plain"}, {"\x1fhead", "a\x1fb", "plain"}, set(ids), {"plain", "nope\x1f"}):
        got = {cid for cid, _ in index.search(vectors[0], 10, allowed)}
        assert got == allowed & set(ids), allowed


def test_a_plan_shows_neither_the_query_vector_nor_the_ids(make_index, pgvector_server, pg_schema):
    index = make_index(16)
    ids, matrix = fill(index, 200, 16, seed=1)
    plan = "\n".join(index.explain(matrix[0], 5, set(ids[:50])))
    assert "Limit" in plan and "chunk_vectors" in plan
    assert ids[0] not in plan and "[0." not in plan and "[-0." not in plan and "{c0" not in plan


def test_a_very_large_filter_is_answered_exactly(make_index, pgvector_server, pg_schema):
    index = make_index(16)
    ids, matrix = fill(index, 20_000, 16, seed=6)
    pgvector_server.execute(f"ANALYZE {table(pg_schema)}")
    query = normalized_vectors(1, 16, 8)[0]
    scores = exact_cosine(matrix, query)
    allowed = {ids[n] for n in range(0, 20_000, 4)} | {f"x{n}" for n in range(50_000)}      # 55 000 ids, 5000 of them stored
    assert_same_topk(index.search(query, 20, allowed), ids, scores, 20, allowed=allowed)


# --- the schema ---------------------------------------------------------------------------------------------------------

def test_describe_reports_the_server_and_nothing_secret(make_index, pgvector_server, pg_schema):
    index = make_index(8)
    assert index.describe()["connected"] is False
    index.count()
    info = index.describe()
    assert info["host"] == pgvector_server.host and info["port"] == pgvector_server.port
    assert info["database"] == pgvector_server.dbname and info["schema"] == pg_schema and info["dimension"] == 8
    assert info["extension_version"] == pgvector_server.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")[0][0]
    assert info["server_version"].split(".")[0].isdigit() and info["loopback"] is True and info["connected"] is True
    assert pgvector_server.password not in repr(info) + repr(index) and pgvector_server.user not in info.values()


def test_the_connection_has_its_guards_set(make_index):
    index = make_index(8)
    with request_scope(deadline=time.monotonic() + 30):
        index.count()
        shown = index._conn.execute("SHOW statement_timeout").fetchone()[0]
    assert shown.endswith("ms") or shown.endswith("s")
    assert index._conn.execute("SHOW idle_in_transaction_session_timeout").fetchone()[0] in ("2min", "120s", "120000ms")
    assert index._conn.execute("SHOW application_name").fetchone()[0] == "tsukkomi-mcp"


def test_the_schema_is_created_twice_and_concurrently_without_error(make_index, pgvector_server, pg_schema):
    first = make_index(8)
    first.count()
    second = make_index(8)
    second.count()
    assert first.ids() == second.ids() == set()
    fresh = pg_schema + "_c"
    pgvector_server.drop_schema(fresh)
    try:
        barrier = threading.Barrier(8)

        def open_one(n):
            index = PgVectorIndex(pgvector_server.conninfo(), dim=8, schema=fresh, policy="p")
            try:
                barrier.wait(timeout=20)
                index.upsert(f"id{n}", normalized_vectors(1, 8, n)[0])
                return index.count()
            finally:
                index.close()
        with ThreadPoolExecutor(8) as pool:
            counts = list(pool.map(open_one, range(8)))
        assert max(counts) == 8
        meta = dict(pgvector_server.execute(f'SELECT key, value FROM "{fresh}".tsukkomi_meta'))
        assert meta == {"schema_version": "1", "dim": "8", "vector_type": "vector", "policy": "p"}
        assert pgvector_server.execute("SELECT count(*) FROM pg_tables WHERE schemaname = %s", (fresh,))[0][0] == 2
    finally:
        pgvector_server.drop_schema(fresh)


def test_a_dimension_or_policy_mismatch_is_an_explicit_error_and_nothing_is_changed(make_index, pgvector_server, pg_schema):
    first = make_index(8, policy="model-a")
    first.upsert("a", normalized_vectors(1, 8, 1)[0])
    with pytest.raises(PgVectorConfigError, match=r"8-dimensional vectors but this process uses 16"):
        make_index(16, policy="model-a").count()
    with pytest.raises(PgVectorConfigError, match="model-a.*model-b"):
        make_index(8, policy="model-b").count()
    assert make_index(8, policy=None).ids() == {"a"}                      # a process with no policy reads what is there
    assert first.count() == 1
    assert dict(pgvector_server.execute(f'SELECT key, value FROM "{pg_schema}".tsukkomi_meta'))["policy"] == "model-a"


def test_a_table_of_another_shape_without_metadata_is_refused(make_index, pgvector_server, pg_schema):
    pgvector_server.execute(f'CREATE SCHEMA "{pg_schema}"')
    pgvector_server.execute(f"CREATE TABLE {table(pg_schema)} (chunk_id text PRIMARY KEY, embedding vector(8) NOT NULL)")
    with pytest.raises(PgVectorConfigError, match="exists with another shape"):
        make_index(16).count()
    assert make_index(8).count() == 0                                     # the right shape is adopted and recorded
    assert dict(pgvector_server.execute(f'SELECT key, value FROM "{pg_schema}".tsukkomi_meta'))["dim"] == "8"


def test_a_vanished_table_is_re_created_and_a_vanished_schema_too(make_index, pgvector_server, pg_schema):
    index = make_index(8)
    index.upsert("a", normalized_vectors(1, 8, 1)[0])
    pgvector_server.execute(f"DROP TABLE {table(pg_schema)}")
    assert index.count() == 0
    index.upsert("b", normalized_vectors(1, 8, 2)[0])
    pgvector_server.drop_schema(pg_schema)
    assert index.ids() == set()
    index.upsert("c", normalized_vectors(1, 8, 3)[0])
    assert index.ids() == {"c"}


def test_a_schema_version_from_another_release_is_refused(make_index, pgvector_server, pg_schema):
    make_index(8).count()
    pgvector_server.execute(f"UPDATE \"{pg_schema}\".tsukkomi_meta SET value = '2' WHERE key = 'schema_version'")
    with pytest.raises(PgVectorConfigError, match="schema version '2'"):
        make_index(8).count()


def test_a_database_without_the_extension_names_the_repair(make_index, pgvector_server):
    name = "noext_" + os.urandom(4).hex()
    pgvector_server.new_database(name)
    try:
        index = make_index(8, dbname=name, schema="s1")
        with pytest.raises(PgVectorConfigError, match="CREATE EXTENSION vector") as caught:
            index.count()
        assert pgvector_server.password not in str(caught.value) and name in str(caught.value)
        assert pgvector_server.execute("SELECT count(*) FROM pg_namespace WHERE nspname = 's1'", dbname=name)[0][0] == 0
    finally:
        pgvector_server.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def test_a_role_without_create_gets_the_permission_hint_and_a_granted_one_works(make_index, pgvector_server, pg_schema):
    role = "tsk_low_" + os.urandom(4).hex()                   # hex only: safe to put in a statement
    password = "pw-" + os.urandom(8).hex()
    pgvector_server.execute(f"CREATE ROLE {role} LOGIN PASSWORD '{password}'")
    try:
        low = f"host={pgvector_server.host} port={pgvector_server.port} dbname={pgvector_server.dbname} user={role} password={password}"
        with pytest.raises(PgVectorError, match="permission denied.*needs CREATE on the database") as caught:
            make_index(8, conninfo=low).count()
        assert password not in str(caught.value)
        pgvector_server.execute(f'GRANT CREATE ON DATABASE "{pgvector_server.dbname}" TO {role}')
        index = make_index(8, conninfo=low)
        index.upsert("a", normalized_vectors(1, 8, 1)[0])
        assert index.ids() == {"a"}
        index.close()
    finally:
        pgvector_server.drop_schema(pg_schema)
        pgvector_server.execute(f"DROP OWNED BY {role}")
        pgvector_server.execute(f"DROP ROLE {role}")


def test_bad_credentials_and_an_unreachable_port_are_unavailable_errors_without_secrets(make_index, pgvector_server):
    wrong = "wrong-" + os.urandom(6).hex()
    with pytest.raises(PgVectorUnavailable) as caught:
        make_index(8, conninfo=pgvector_server.conninfo(password=wrong)).count()
    text = str(caught.value)
    assert wrong not in text and pgvector_server.password not in text
    assert f"{pgvector_server.host}:{pgvector_server.port}/{pgvector_server.dbname}" in text and "SELF_DIRECT_PGVECTOR_DSN_FILE" in text
    with pytest.raises(PgVectorUnavailable, match="Cannot connect to the pgvector server 127.0.0.1:1/") as caught:
        make_index(8, conninfo=pgvector_server.conninfo(port=1, connect_timeout=2)).count()
    assert pgvector_server.password not in str(caught.value) and "No other dense backend is used" in str(caught.value)


# --- connection loss, outages and the time budget -----------------------------------------------------------------------

def test_a_terminated_backend_is_replaced_once_and_the_operation_repeated(make_index, pgvector_server):
    index = make_index(8)
    ids, matrix = fill(index, 50, 8, seed=1)
    first_pid = index._conn.info.backend_pid
    pgvector_server.execute("SELECT pg_terminate_backend(%s)", (first_pid,))
    time.sleep(0.2)
    assert index.count() == 50 and index._conn.info.backend_pid != first_pid
    pgvector_server.execute("SELECT pg_terminate_backend(%s)", (index._conn.info.backend_pid,))
    time.sleep(0.2)
    index.upsert("late", matrix[0])                                      # a write on a dead connection: repeated whole
    assert index.count() == 51


def test_a_backend_terminated_between_batches_leaves_the_write_whole_and_applied_once(make_index, pgvector_server):
    index = make_index(8)
    index.count()
    original, calls = index._arm, {"n": 0}

    def arm(connection, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:                                              # the start of the second batch
            pgvector_server.execute("SELECT pg_terminate_backend(%s)", (connection.info.backend_pid,))
            time.sleep(0.3)
        return original(connection, *args, **kwargs)
    index._arm = arm
    ids = make_ids(3 * UPSERT_BATCH)
    index.upsert_many(list(zip(ids, normalized_vectors(len(ids), 8, 4))))
    assert index.count() == len(ids) and index.ids() == set(ids)


def test_a_server_that_goes_away_is_an_explicit_error_with_the_repair_and_comes_back(pgvector_server, pg_schema):
    proxy = TcpProxy(pgvector_server.host, pgvector_server.port)
    index = PgVectorIndex(pgvector_server.conninfo(port=proxy.port, connect_timeout=3), dim=8, schema=pg_schema, policy="p")
    try:
        index.upsert("a", normalized_vectors(1, 8, 1)[0])
        proxy.stop()
        for call in (lambda: index.count(), lambda: index.search(normalized_vectors(1, 8, 2)[0], 3),
                     lambda: index.upsert("b", normalized_vectors(1, 8, 3)[0]), lambda: index.present(["a"])):
            with pytest.raises(PgVectorUnavailable) as caught:
                call()
            text = str(caught.value)
            assert f"127.0.0.1:{proxy.port}/" in text and pgvector_server.password not in text
            assert "SELF_DIRECT_PGVECTOR_DSN_FILE" in text or "unreachable" in text
            assert "no other dense backend" in text.lower()
        proxy.start()
        assert index.ids() == {"a"}                                      # recovered: the failed write did not land
        index.upsert("b", normalized_vectors(1, 8, 3)[0])
        assert index.count() == 2
    finally:
        index.close()
        proxy.stop()


def test_a_request_deadline_ends_a_statement_blocked_on_a_lock_and_applies_nothing(make_index, pgvector_server, pg_schema):
    index = make_index(8)
    ids, matrix = fill(index, 100, 8, seed=2)
    blocker = pgvector_server.connect()
    blocker.autocommit = False
    blocker.execute(f"LOCK TABLE {table(pg_schema)} IN ACCESS EXCLUSIVE MODE")
    try:
        for call in (lambda: index.search(matrix[0], 5), lambda: index.count(), lambda: index.present(ids[:5]),
                     lambda: index.upsert_many([("new", matrix[0])]), lambda: index.delete_ids(ids[:5]), lambda: index.clear()):
            started = time.monotonic()
            with request_scope(deadline=started + 1.0), pytest.raises(RequestStopped) as caught:
                call()
            assert caught.value.reason == "request_deadline" and 0.8 < time.monotonic() - started < 4.0
    finally:
        blocker.rollback()
        blocker.close()
    assert index.count() == 100 and index.present(["new"]) == set() and index.present(ids[:5]) == set(ids[:5])


def test_a_cancel_between_batches_leaves_no_partial_batch(make_index):
    index = make_index(8)
    base_ids, base = fill(index, 10, 8, seed=1, prefix="base")
    event = threading.Event()
    original, calls = index._arm, {"n": 0}

    def arm(connection, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:                                              # while the second batch is being sent
            event.set()
        return original(connection, *args, **kwargs)
    index._arm = arm
    ids = make_ids(3 * UPSERT_BATCH, "new")
    with request_scope(cancel_event=event), pytest.raises(RequestStopped) as caught:
        index.upsert_many(list(zip(ids, normalized_vectors(len(ids), 8, 4))))
    assert caught.value.reason == "request_cancelled" and calls["n"] == 2
    assert index.ids() == set(base_ids)                                  # not even the batch already sent is visible
    calls["n"] = 0
    event.clear()
    doomed = base_ids + [f"zz-{n}" for n in range(2 * PRESENT_BATCH)]      # sorts after the stored ids: three delete batches
    with request_scope(cancel_event=event), pytest.raises(RequestStopped):
        index.delete_ids(doomed)
    index._arm = original
    assert index.present(base_ids) == set(base_ids) and index.count() == 10    # the first delete batch was rolled back
    index.delete_ids(doomed)
    assert index.count() == 0


def test_a_deadline_that_has_passed_sends_nothing(make_index):
    index = make_index(8)
    index.count()
    with request_scope(deadline=time.monotonic() - 1):
        with pytest.raises(RequestStopped):
            index.upsert("a", normalized_vectors(1, 8, 1)[0])
    assert index.ids() == set()


def test_two_threads_share_one_index_object(make_index):
    index = make_index(8)
    errors = []

    def work(seed):
        try:
            for n in range(10):
                index.upsert(f"t{seed}-{n}", normalized_vectors(1, 8, seed * 100 + n)[0])
                assert index.search(normalized_vectors(1, 8, n)[0], 3)
                index.present([f"t{seed}-{n}"])
        except BaseException as exc:
            errors.append(exc)
    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    assert errors == [] and index.count() == 40


# --- another process on the same index ---------------------------------------------------------------------------------

CHILD = """
import os, sys
import numpy as np
from self_directing_mcp.index.pgvector import PgVectorIndex
from self_directing_mcp.settings_core import read_pgvector_dsn_file
from pathlib import Path
index = PgVectorIndex(read_pgvector_dsn_file(Path(os.environ["CHILD_DSN_FILE"])), dim=16, schema=os.environ["CHILD_SCHEMA"], policy="p")
rng = np.random.default_rng(int(os.environ["CHILD_SEED"]))
for round_number in range(25):
    pairs = [(f"child-{round_number}-{n}", rng.standard_normal(16)) for n in range(40)]
    pairs += [(f"shared-{n}", rng.standard_normal(16)) for n in range(30)]
    index.upsert_many(pairs)
    index.search(rng.standard_normal(16), 5)
    if round_number % 5 == 4:
        index.delete_ids([f"child-{round_number - 4}-{n}" for n in range(0, 40, 2)])
index.close()
print("child done")
"""


def test_another_process_writes_the_same_index_concurrently(make_index, pgvector_server, pg_schema, tmp_path):
    dsn_file = pgvector_server.write_dsn_file(tmp_path)
    parent = make_index(16, policy="p")
    parent.count()
    environment = {**os.environ, "PYTHONPATH": SRC, "CHILD_DSN_FILE": str(dsn_file), "CHILD_SCHEMA": pg_schema, "CHILD_SEED": "5",
                   "PYTHONDONTWRITEBYTECODE": "1"}
    child = subprocess.Popen([sys.executable, "-c", CHILD], env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    rng = np.random.default_rng(6)
    for round_number in range(25):
        pairs = [(f"parent-{round_number}-{n}", rng.standard_normal(16)) for n in range(40)]
        pairs += [(f"shared-{n}", rng.standard_normal(16)) for n in range(30)]
        parent.upsert_many(pairs)
        assert parent.search(rng.standard_normal(16), 5)
    out, err = child.communicate(timeout=180)
    assert child.returncode == 0 and "child done" in out and err.strip() == "", err
    expected_parent = {f"parent-{r}-{n}" for r in range(25) for n in range(40)}
    expected_child = {f"child-{r}-{n}" for r in range(25) for n in range(40)}
    deleted = {f"child-{r}-{n}" for r in range(0, 21, 5) for n in range(0, 40, 2)}
    ids = parent.ids()
    assert expected_parent <= ids and ids == expected_parent | (expected_child - deleted) | {f"shared-{n}" for n in range(30)}
    assert parent.count() == len(ids)


# --- the approximate index -------------------------------------------------------------------------------------------------

def test_hnsw_serves_unfiltered_searches_with_good_recall_and_filtered_searches_stay_exact(make_index, pgvector_server, pg_schema):
    index = make_index(32, hnsw=True)
    ids, matrix = fill(index, 3000, 32, seed=11)
    pgvector_server.execute(f"ANALYZE {table(pg_schema)}")
    assert pgvector_server.execute("SELECT count(*) FROM pg_indexes WHERE schemaname = %s AND indexname = 'chunk_vectors_embedding_hnsw'",
                                   (pg_schema,))[0][0] == 1
    recalls = []
    for seed in range(20):
        query = normalized_vectors(1, 32, 500 + seed)[0]
        scores = exact_cosine(matrix, query)
        want = {ids[n] for n in np.argsort(-scores)[:10]}
        recalls.append(len(want & {cid for cid, _ in index.search(query, 10)}) / 10)
    assert np.mean(recalls) >= 0.85
    plan = "\n".join(index.explain(normalized_vectors(1, 32, 1)[0], 10))
    assert "chunk_vectors_embedding_hnsw" in plan, plan                   # the unfiltered search is served by the index
    allowed_plan = "\n".join(index.explain(normalized_vectors(1, 32, 1)[0], 10, {ids[n] for n in range(0, 3000, 3)}))
    assert "hnsw" not in allowed_plan, allowed_plan                       # a filtered search never uses it
    query = normalized_vectors(1, 32, 999)[0]
    scores = exact_cosine(matrix, query)
    allowed = {ids[n] for n in range(0, 3000, 3)}
    assert_same_topk(index.search(query, 15, allowed), ids, scores, 15, allowed=allowed, label="filtered, hnsw on")
    assert index.describe()["hnsw"] is True


def test_an_hnsw_index_can_be_built_on_demand_for_an_existing_table(make_index, pgvector_server, pg_schema):
    index = make_index(16)
    fill(index, 500, 16, seed=3)
    index.build_hnsw_index(maintenance_work_mem="64MB", timeout_ms=0)
    names = [row[0] for row in pgvector_server.execute("SELECT indexname FROM pg_indexes WHERE schemaname = %s", (pg_schema,))]
    assert "chunk_vectors_embedding_hnsw" in names
    with_hnsw = make_index(16, hnsw=True)                                 # an existing index is adopted, not rebuilt
    assert with_hnsw.count() == 500


# --- a real stop, last: the container is restarted afterwards ------------------------------------------------------------

def test_a_stopped_server_gives_the_explicit_error_and_a_restarted_one_works(pgvector_server, pg_schema):
    if not pgvector_server.managed:
        pytest.skip("the server of TSUKKOMI_PGVECTOR_TEST_DSN_FILE cannot be stopped by the tests (the proxy test covers the outage)")
    index = PgVectorIndex(pgvector_server.conninfo(connect_timeout=3), dim=8, schema=pg_schema, policy="p")
    try:
        index.upsert("a", normalized_vectors(1, 8, 1)[0])
        password = pgvector_server.password
        pgvector_server.stop()
        for call in (lambda: index.count(), lambda: index.upsert("b", normalized_vectors(1, 8, 2)[0]),
                     lambda: index.search(normalized_vectors(1, 8, 3)[0], 3)):
            with pytest.raises(PgVectorUnavailable) as caught:
                call()
            assert password not in str(caught.value) and "no other dense backend" in str(caught.value).lower()
    finally:
        index.close()
    pgvector_server.restart()
    fresh = PgVectorIndex(pgvector_server.conninfo(), dim=8, schema=pg_schema, policy="p")
    try:
        assert fresh.count() == 0                                         # a new server: the old data went with it
        fresh.upsert("a", normalized_vectors(1, 8, 1)[0])
        assert fresh.ids() == {"a"}
    finally:
        fresh.close()
