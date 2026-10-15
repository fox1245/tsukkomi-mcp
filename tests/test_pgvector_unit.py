"""The pgvector backend against a fake driver (``pgvector_fakes``): SQL shapes, batching, validation, the time budget,
cancellation, reconnect, error text and the TLS rule. No psycopg, no server, no Docker: this runs in every environment.
Behaviour against a real server is in ``test_pgvector_integration.py``."""
from __future__ import annotations

import json
import struct
import threading
import time
import traceback
from urllib.parse import quote

import numpy as np
import pytest

from pgvector_fakes import FakeDbError, FakeDriver, FakeInfo, FakeServer
from self_directing_mcp.index import pgvector
from self_directing_mcp.index.pgvector import (
    MAX_ID_CHARS, NO_DEADLINE_TIMEOUT_MS, PRESENT_BATCH, UPSERT_BATCH, PgVectorConfigError, PgVectorError,
    PgVectorIndex, PgVectorUnavailable, check_transport)
from self_directing_mcp.request_control import RequestControl, RequestStopped, request_scope

PASSWORD = "S3cr3t-PW-4711"
DSN = f"host=127.0.0.1 port=5433 dbname=appdb user=app password={PASSWORD}"


def make(server=None, *, dim=4, dsn=DSN, info=None, **options):
    driver = FakeDriver(server if server is not None else FakeServer(dim=dim), info=info)
    index = PgVectorIndex(dsn, dim=dim, schema=options.pop("schema", "tsukkomi"), driver=driver, **options)
    return index, driver, driver.server


def unit(dim=4, seed=0):
    vector = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def arm_statements(server):
    return server.statements("set_config('statement_timeout'")


# --- construction -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [0, -1, 16001, True, 1.5, "4"])
def test_a_bad_dimension_is_refused(dim):
    with pytest.raises(ValueError, match="dimensions"):
        PgVectorIndex(DSN, dim=dim, driver=FakeDriver())


def test_hnsw_refuses_more_than_2000_dimensions_and_policy_is_checked():
    with pytest.raises(PgVectorConfigError, match="2000"):
        PgVectorIndex(DSN, dim=2001, hnsw=True, driver=FakeDriver())
    for policy in ("", "x" * 129, "bad\npolicy", 5):
        with pytest.raises(ValueError, match="policy"):
            PgVectorIndex(DSN, dim=4, policy=policy, driver=FakeDriver())


def test_without_a_dsn_or_libpq_environment_the_error_names_the_settings(monkeypatch):
    for name in pgvector.CONNECTION_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(PgVectorConfigError, match="SELF_DIRECT_PGVECTOR_DSN_FILE"):
        PgVectorIndex("", dim=4, driver=FakeDriver())
    monkeypatch.setenv("PGHOST", "127.0.0.1")
    PgVectorIndex("", dim=4, driver=FakeDriver())           # the libpq environment alone is enough


def test_a_missing_driver_module_gives_an_install_hint(monkeypatch):
    monkeypatch.setattr(pgvector.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(PgVectorConfigError, match=r"tsukkomi-mcp\[pgvector\]"):
        PgVectorIndex(DSN, dim=4)


def test_construction_and_refresh_never_connect_and_the_first_query_does():
    index, driver, _server = make()
    index.refresh()
    assert driver.connections == [] and index.describe()["connected"] is False
    assert index.count() == 0
    assert len(driver.connections) == 1 and index.describe()["connected"] is True
    index.close()
    index.close()                                           # idempotent
    assert driver.connections[0].closed and index.describe()["connected"] is False


def test_repr_and_describe_hold_no_credential():
    index, _driver, _server = make(info=FakeInfo(port=5433))
    index.count()
    text = repr(index) + json.dumps(index.describe())
    assert PASSWORD not in text and "app" not in index.describe().values()
    assert index.describe()["host"] == "127.0.0.1" and index.describe()["port"] == 5433


# --- vectors and ids reach SQL only when valid ----------------------------------------------------------------------------

def test_vectors_travel_as_float32_big_endian_with_pgvectors_header():
    index, _driver, server = make()
    index.upsert("a", np.array([1, 2, 3, 4], dtype=np.float64))
    assert server.rows["a"].payload == struct.pack(">HH", 4, 0) + struct.pack(">4f", 1, 2, 3, 4)
    index.upsert_many([("b", [0.5, 0.25, 0.125, 0.0625])])
    assert server.rows["b"].array().tolist() == [0.5, 0.25, 0.125, 0.0625]


@pytest.mark.parametrize("vector", [[1, 2, 3], [[1, 2, 3, 4]], [1, 2, 3, float("nan")], [1, 2, float("inf"), 4],
                                    [1, 2, 3, 4, 5], "abcd"])
def test_an_invalid_vector_never_reaches_the_server(vector):
    index, driver, server = make()
    with pytest.raises((ValueError, TypeError)):
        index.upsert_many([("fine", unit()), ("bad", vector)])
    with pytest.raises((ValueError, TypeError)):
        index.search(vector, 3)
    assert driver.connections == [] and server.log == []


@pytest.mark.parametrize("bad", ["", None, 7, b"x", "a\x00b", "x" * (MAX_ID_CHARS + 1), "\ud800"])
def test_an_id_that_cannot_be_stored_is_an_error_for_writes_and_absent_for_reads(bad):
    index, driver, server = make()
    with pytest.raises(ValueError, match="chunk id"):
        index.upsert(bad, unit())
    index.upsert("good", unit())
    assert index.present([bad, "good"]) == {"good"}
    assert [cid for cid, _ in index.search(unit(), 5, {bad, "good"})] == ["good"]
    index.delete_ids([bad])
    assert set(server.rows) == {"good"}
    before = len(server.log)
    assert index.present([bad]) == set() and index.search(unit(), 5, {bad}) == []
    index.delete_ids([bad])
    assert len(server.log) == before                          # nothing storable: no statement at all


def test_a_duplicate_id_in_one_call_keeps_the_last_vector_and_an_empty_call_does_nothing():
    index, driver, server = make()
    index.upsert_many([])
    assert driver.connections == []
    first, last = unit(seed=1), unit(seed=2)
    index.upsert_many([("a", first), ("a", last)])
    assert np.allclose(server.rows["a"].array(), last, atol=1e-7)


# --- SQL shapes and batching ---------------------------------------------------------------------------------------------

def test_an_upsert_is_one_transaction_of_batches_and_never_inlines_values():
    index, driver, server = make()
    index.count()
    server.log.clear()
    base = driver.connections[0].commits
    pairs = [(f"id-{n:05d}", unit(seed=n)) for n in range(2 * UPSERT_BATCH + 7)]
    index.upsert_many(pairs)
    inserts = [entry for entry in server.log if entry[0].startswith('INSERT INTO "tsukkomi"."chunk_vectors"')]
    assert [len(rows) for _sql, rows in inserts] == [UPSERT_BATCH, UPSERT_BATCH, 7]
    sql = inserts[0][0]
    assert "ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding" in sql and "%b::vector" in sql
    assert "id-00000" not in sql and "IS DISTINCT FROM" in sql
    assert all(rows == sorted(rows, key=lambda row: row[0]) for _sql, rows in inserts)     # one lock order for every writer
    connection = driver.connections[0]
    assert connection.commits == base + 1 and connection.autocommit is False
    assert [entry[0] for entry in server.log].count("COMMIT") == 1 and server.log[-1][0] == "COMMIT"
    assert len(arm_statements(server)) == 3                    # the time left is set again for every batch
    assert all(entry[1][1] is True for entry in arm_statements(server))          # SET LOCAL inside the transaction
    assert len(server.rows) == len(pairs)


def test_reads_are_autocommit_statements_with_the_time_left_set_for_the_session():
    index, driver, server = make()
    index.upsert("a", unit())
    index.count()
    assert driver.connections[0].autocommit is True
    assert arm_statements(server)[-1][1][1] is False


def test_present_and_delete_run_in_batches_inside_the_call():
    index, driver, server = make()
    ids = [f"id-{n:05d}" for n in range(2 * PRESENT_BATCH + 5)]
    index.upsert_many([(cid, unit(seed=n)) for n, cid in enumerate(ids[:300])])
    base = driver.connections[0].commits
    assert index.present(ids) == set(ids[:300])
    selects = [entry for entry in server.log if entry[0].startswith("SELECT chunk_id FROM") and "ANY" in entry[0]]
    assert [len(entry[1][0]) for entry in selects] == [PRESENT_BATCH, PRESENT_BATCH, 5]
    index.delete_ids(ids)
    deletes = [entry for entry in server.log if entry[0].startswith("DELETE FROM")]
    assert [len(entry[1][0]) for entry in deletes] == [PRESENT_BATCH, PRESENT_BATCH, 5]
    assert driver.connections[0].commits == base + 1 and server.rows == {}      # the whole delete is one transaction
    assert index.present([]) == set()


def test_ids_count_and_clear():
    index, driver, server = make()
    index.upsert_many([("b", unit(seed=1)), ("a", unit(seed=2))])
    assert index.ids() == {"a", "b"} and index.count() == 2
    index.clear()
    assert index.count() == 0 and any(entry[0].startswith("TRUNCATE") for entry in server.log)


def test_search_filters_before_the_limit_and_breaks_ties_by_id_in_byte_order():
    index, driver, server = make()
    index.upsert_many([(f"c{n}", unit(seed=n)) for n in range(6)])
    index.search(unit(seed=3), 4, {"c1", "c2", "c3"})
    sql, params = [entry for entry in server.log if entry[0].startswith("SELECT chunk_id, 1.0 -")][-1]
    assert sql.index("WHERE chunk_id = ANY(string_to_array(%(blob)s::text, chr(31)) || %(odd)b::text[])") < sql.index("ORDER BY") < sql.index("LIMIT")
    assert 'ORDER BY embedding <=> %(q)b::vector, chunk_id COLLATE "C" LIMIT' in sql
    assert sorted(params["blob"].split("\x1f")) == ["c1", "c2", "c3"] and params["odd"] == [] and params["k"] == 4
    assert "c1" not in sql and "c2" not in sql                       # ids travel as parameters, never inside the text
    hits = index.search(unit(seed=3), 4, {"c1", "c2", "c3"})
    assert [cid for cid, _ in hits][0] == "c3" and hits[0][1] == pytest.approx(1.0, abs=1e-6)
    assert all(isinstance(score, float) for _cid, score in hits)


def test_search_without_work_to_do_sends_nothing():
    index, driver, server = make()
    assert index.search(unit(), 0) == [] and index.search(unit(), -3) == []
    assert index.search(unit(), 5, set()) == [] and index.search(unit(), 5, frozenset()) == []
    assert driver.connections == []


def test_a_zero_vector_scores_zero_like_the_other_backends():
    index, driver, server = make()
    index.upsert_many([("zero", np.zeros(4, dtype=np.float32)), ("a", unit())])
    scores = dict(index.search(unit(), 5))
    assert scores["zero"] == 0.0 and np.isfinite(list(scores.values())).all()


def test_the_planner_is_pushed_to_scan_when_the_filter_is_a_large_part_of_the_table():
    index, driver, server = make(FakeServer(dim=4, reltuples=90))
    index.upsert_many([(f"c{n}", unit(seed=n)) for n in range(9)])
    index.search(unit(), 3, {"c1", "c2"})
    assert arm_statements(server)[-1][1][4] == "on"           # 2 of 90: look the ids up
    index.search(unit(), 3, {f"c{n}" for n in range(9)} | {f"x{n}" for n in range(20)})
    assert arm_statements(server)[-1][1][4] == "on"           # 29 of 90: still looked up
    index.search(unit(), 3, {f"c{n}" for n in range(9)} | {f"x{n}" for n in range(21)})
    assert arm_statements(server)[-1][1][4] == "off"          # 30 of 90 (a third): one scan with a hashed filter
    index2, _d2, server2 = make(FakeServer(dim=4))            # never analyzed: the planner decides alone
    index2.search(unit(), 3, {"c1"})
    assert arm_statements(server2)[-1][1][4] == "on"


def test_hnsw_serves_only_unfiltered_searches():
    index, driver, server = make(FakeServer(dim=4, has_index=True), hnsw=True)
    index.upsert_many([(f"c{n}", unit(seed=n)) for n in range(4)])
    index.search(unit(), 3)
    sql = [entry[0] for entry in server.log if entry[0].startswith("SELECT chunk_id, 1.0 -")][-1]
    assert "chunk_id COLLATE" not in sql and any("hnsw.ef_search" in entry[0] for entry in server.log)
    server.log.clear()
    index.search(unit(), 3, {"c1", "c2"})
    sql = [entry[0] for entry in server.log if entry[0].startswith("SELECT chunk_id, 1.0 -")][-1]
    assert 'chunk_id COLLATE "C"' in sql and not any("hnsw.ef_search" in entry[0] for entry in server.log)


# --- the schema ---------------------------------------------------------------------------------------------------------

def test_creating_the_schema_is_one_locked_transaction_that_records_dimension_and_policy():
    server = FakeServer(dim=1024, has_schema=False, has_meta=False, has_table=False)
    index, driver, _ = make(server, dim=1024, policy="abc123")
    index.count()
    statements = [entry[0] for entry in server.log]
    first_ddl = next(i for i, sql in enumerate(statements) if "pg_advisory_xact_lock" in sql)
    assert first_ddl < next(i for i, sql in enumerate(statements) if sql.startswith("CREATE SCHEMA IF NOT EXISTS"))
    create_table = next(sql for sql in statements if sql.startswith('CREATE TABLE IF NOT EXISTS "tsukkomi"."chunk_vectors"'))
    assert 'chunk_id text COLLATE "C" PRIMARY KEY' in create_table and "vector(1024)" in create_table
    assert any("SET STORAGE PLAIN" in sql for sql in statements)
    assert server.meta == {"schema_version": "1", "dim": "1024", "vector_type": "vector", "policy": "abc123"}
    assert not any("hnsw" in sql.lower() for sql in statements)
    assert driver.connections[0].commits == 1


def test_a_large_dimension_keeps_pgvectors_default_storage_and_hnsw_is_created_only_on_request():
    server = FakeServer(dim=1800, has_schema=False, has_meta=False, has_table=False)
    index, driver, _ = make(server, dim=1800, hnsw=True)
    index.count()
    statements = [entry[0] for entry in server.log]
    assert not any("SET STORAGE" in sql for sql in statements)
    assert any(sql.startswith('CREATE INDEX IF NOT EXISTS "chunk_vectors_embedding_hnsw"') and "vector_cosine_ops" in sql
               for sql in statements)


def test_a_dimension_mismatch_is_an_explicit_error_naming_both_dimensions_and_the_repair():
    index, driver, server = make(FakeServer(dim=8), dim=4)
    with pytest.raises(PgVectorConfigError, match=r"8-dimensional.*uses 4.*SELF_DIRECT_PGVECTOR_SCHEMA"):
        index.count()
    assert driver.connections[0].closed and len(driver.connections) == 1          # no retry of a configuration error


def test_a_policy_or_schema_version_mismatch_is_refused_and_an_unrecorded_policy_is_claimed():
    server = FakeServer(dim=4, meta={"schema_version": "1", "dim": "4", "policy": "old-model"})
    index, _driver, _ = make(server, policy="new-model")
    with pytest.raises(PgVectorConfigError, match="old-model.*new-model"):
        index.count()
    server = FakeServer(dim=4, meta={"schema_version": "2", "dim": "4"})
    with pytest.raises(PgVectorConfigError, match="schema version '2'"):
        make(server)[0].count()
    server = FakeServer(dim=4)                                  # built by a process that had no policy to record
    make(server, policy="p1")[0].count()
    assert server.meta["policy"] == "p1"
    with pytest.raises(PgVectorConfigError, match="p1.*p2"):
        make(server, policy="p2")[0].count()
    make(server)[0].count()                                     # a process without a policy accepts what is there


def test_a_missing_extension_names_create_extension_and_is_never_created_by_this_process():
    index, driver, server = make(FakeServer(dim=4, extension_installed=False))
    with pytest.raises(PgVectorConfigError, match="CREATE EXTENSION vector"):
        index.count()
    assert not any("CREATE EXTENSION" in entry[0] for entry in server.log)
    assert driver.connections[0].closed


def test_hnsw_needs_a_pgvector_that_has_it():
    index, _driver, _ = make(FakeServer(dim=4, extension="0.4.4"), hnsw=True)
    with pytest.raises(PgVectorConfigError, match=r"0\.5\.0"):
        index.count()
    make(FakeServer(dim=4, extension="0.4.4"))[0].count()       # exact search needs nothing newer


def test_a_vanished_table_is_re_created_on_the_next_connection():
    index, driver, server = make()
    index.upsert("a", unit())
    server.has_table = False
    driver.connections[0].fail_on.append(("SELECT count(*)", FakeDbError("undefined", "relation does not exist", "42P01")))
    assert index.count() == 1
    assert len(driver.connections) == 2
    assert any(entry[0].startswith('CREATE TABLE IF NOT EXISTS "tsukkomi"."chunk_vectors"') for entry in server.log)


def test_the_schema_name_is_validated_and_always_quoted():
    with pytest.raises(ValueError):
        PgVectorIndex(DSN, dim=4, schema='a"; DROP TABLE x; --', driver=FakeDriver())
    index, _driver, server = make(schema="My_Schema1")
    index.count()
    assert any('"My_Schema1"."chunk_vectors"' in entry[0] for entry in server.log)


# --- the time budget, cancellation and the transaction -------------------------------------------------------------------

def test_the_statement_timeout_is_what_the_request_has_left():
    index, driver, server = make()
    with request_scope(deadline=time.monotonic() + 5.0):
        index.upsert("a", unit())
        index.count()
    values = [int(entry[1][0]) for entry in arm_statements(server)]
    assert all(0 < value <= 5000 for value in values) and min(values) >= 3000
    assert arm_statements(server)[-1][1][0] == arm_statements(server)[-1][1][2]        # lock_timeout is the same budget
    index.count()                                               # no request: the fixed ceiling
    assert arm_statements(server)[-1][1][0] == str(NO_DEADLINE_TIMEOUT_MS)


def test_the_timeout_is_clamped_to_what_postgres_accepts_and_is_at_least_one_millisecond():
    index, _driver, _server = make()
    with request_scope(deadline=time.monotonic() + 10 ** 9):
        assert index._timeout_ms() == 2_147_483_647
    with request_scope(deadline=time.monotonic() + 0.0004):
        assert index._timeout_ms() >= 1
    assert index._timeout_ms(0) == 0                             # an explicit 0 disables the limit (index builds)


def test_an_expired_or_cancelled_request_sends_nothing():
    index, driver, server = make()
    with request_scope(deadline=time.monotonic() - 1):
        for call in (lambda: index.count(), lambda: index.upsert("a", unit()), lambda: index.search(unit(), 2),
                     lambda: index.present(["a"]), lambda: index.delete_ids(["a"]), lambda: index.clear()):
            with pytest.raises(RequestStopped):
                call()
    event = threading.Event()
    event.set()
    with request_scope(cancel_event=event), pytest.raises(RequestStopped) as caught:
        index.count()
    assert caught.value.reason == "request_cancelled" and driver.connections == []


def test_a_cancel_between_batches_stops_the_write_and_leaves_no_half_applied_batch():
    index, driver, server = make()
    index.count()
    base = driver.connections[0].commits
    event = threading.Event()
    inserted = []

    def after(connection, sql):
        if sql.startswith("INSERT INTO"):
            inserted.append(1)
            event.set()                                         # the stop arrives while batch one is applied
    server.after_hook = after
    with request_scope(cancel_event=event), pytest.raises(RequestStopped) as caught:
        index.upsert_many([(f"c{n:04d}", unit(seed=n)) for n in range(3 * UPSERT_BATCH)])
    assert caught.value.reason == "request_cancelled" and len(inserted) == 1
    assert driver.connections[0].commits == base and driver.connections[0].rollbacks >= 1
    assert server.rows == {}                                    # batch one was rolled back with the rest


def test_a_hard_deadline_after_the_last_batch_never_commits():
    index, driver, server = make()
    index.count()
    base = driver.connections[0].commits
    control = RequestControl(deadline=time.monotonic() + 60)

    def after(connection, sql):
        if sql.startswith("INSERT INTO"):
            control.cancel("request_deadline")                  # the deadline hits just before the commit
    server.after_hook = after
    with request_scope(control), pytest.raises(RequestStopped) as caught:
        index.upsert_many([("a", unit()), ("b", unit(seed=1))])
    assert caught.value.reason == "request_deadline" and driver.connections[0].commits == base and server.rows == {}


def test_a_statement_timeout_from_the_server_becomes_the_requests_own_deadline_or_a_clear_error():
    index, driver, server = make()
    index.count()
    driver.connections[0].fail_on.append(("SELECT count(*)", FakeDbError("timeout", "canceling statement", "57014")))
    with pytest.raises(PgVectorError, match="exceeded its time limit"):
        index.count()
    with request_scope(deadline=time.monotonic() - 0.001), pytest.raises(RequestStopped) as caught:
        index._translate(FakeDbError("timeout"), "timeout")
    assert caught.value.reason == "request_deadline"             # the request's own deadline wins over the database's text


# --- connection loss, retry and errors -----------------------------------------------------------------------------------

def test_a_dropped_connection_is_replaced_once_and_the_operation_repeated():
    index, driver, server = make()
    index.upsert_many([("a", unit()), ("b", unit(seed=1))])
    driver.connections[0].fail_on.append(("SELECT count(*)", FakeDbError("connection", "server closed the connection")))
    assert index.count() == 2
    assert len(driver.connections) == 2 and driver.connections[0].closed and not driver.connections[1].closed


def test_a_write_interrupted_by_a_lost_connection_is_repeated_whole_and_is_idempotent():
    index, driver, server = make()
    index.count()
    driver.connections[0].fail_on.append(("INSERT INTO", FakeDbError("connection", "terminating connection")))
    pairs = [(f"c{n}", unit(seed=n)) for n in range(5)]
    index.upsert_many(pairs)
    assert set(server.rows) == {cid for cid, _ in pairs} and len(driver.connections) == 2


def test_a_second_connection_failure_is_an_unavailable_error_with_the_repair_and_no_fallback():
    index, driver, server = make()
    index.count()
    driver.connections[0].fail_on.append(("SELECT count(*)", FakeDbError("connection", "server closed the connection")))
    driver.connect_failures.append(FakeDbError("connection", "connection refused"))
    with pytest.raises(PgVectorUnavailable) as caught:
        index.count()
    message = str(caught.value)
    assert "127.0.0.1:5433/appdb" in message and "SELF_DIRECT_PGVECTOR_DSN_FILE" in message and "reachable" in message
    assert "no other dense backend" in message.lower()
    assert len(driver.connect_calls) == 2                       # one reconnect, not a loop


def test_a_server_that_drops_every_connection_gives_the_unavailable_error_after_one_retry():
    index, driver, server = make()
    index.count()
    driver.connections[0].fail_on.append(("SELECT count(*)", FakeDbError("connection", "gone")))
    original = driver.connect

    def failing(conninfo, **params):
        connection = original(conninfo, **params)
        connection.fail_on.append(("SELECT count(*)", FakeDbError("connection", "gone again")))
        return connection
    driver.connect = failing
    with pytest.raises(PgVectorUnavailable, match="became unreachable.*no other dense backend"):
        index.count()
    assert len(driver.connections) == 2


def test_a_conflict_with_another_writer_is_retried_once():
    index, driver, server = make()
    index.count()
    driver.connections[0].fail_on.append(("INSERT INTO", FakeDbError("transient", "deadlock detected", "40P01")))
    index.upsert("a", unit())
    assert set(server.rows) == {"a"} and len(driver.connections) == 1
    driver.connections[0].fail_on.extend([("INSERT INTO", FakeDbError("transient", "deadlock detected", "40P01")),
                                          ("INSERT INTO", FakeDbError("transient", "deadlock detected", "40P01"))])
    with pytest.raises(PgVectorError, match="conflicted with another writer twice"):
        index.upsert("b", unit())


def test_an_error_that_is_not_the_databases_is_not_swallowed_or_retried():
    index, driver, server = make()
    index.count()
    driver.connections[0].fail_on.append(("SELECT count(*)", KeyError("not a database error")))
    with pytest.raises(KeyError):
        index.count()
    assert len(driver.connections) == 1


def test_a_database_error_names_the_endpoint_and_a_permission_error_names_the_repair():
    index, driver, server = make()
    index.count()
    driver.connections[0].fail_on.append(("SELECT count(*)", FakeDbError("database", "permission denied for table", "42501")))
    with pytest.raises(PgVectorError, match=r"127\.0\.0\.1:5433/appdb.*permission denied.*CREATE on the database"):
        index.count()


@pytest.mark.parametrize("dsn", [DSN, "postgresql://app:p%40ss%3Aw%2Frd@127.0.0.1:5433/appdb",
                                 "host=127.0.0.1 password=p@ss:w/rd"])
def test_no_error_message_trace_or_description_ever_contains_the_password(dsn):
    secret = PASSWORD if dsn == DSN else "p@ss:w/rd"
    leaking = f'FATAL: password authentication failed for "{secret}" and {quote(secret, safe="")} in {dsn}'
    index, driver, server = make(dsn=dsn)
    driver.connect_failures.append(FakeDbError("connection", leaking))
    with pytest.raises(PgVectorUnavailable) as caught:
        index.count()
    rendered = "".join(traceback.format_exception(caught.value)) + str(caught.value) + repr(index) + json.dumps(index.describe())
    assert secret not in rendered and quote(secret, safe="") not in rendered and dsn not in rendered
    index.count()
    driver.connections[0].fail_on.append(("SELECT count(*)", FakeDbError("database", leaking, "XX000")))
    with pytest.raises(PgVectorError) as caught:
        index.count()
    rendered = "".join(traceback.format_exception(caught.value)) + str(caught.value)
    assert secret not in rendered and quote(secret, safe="") not in rendered and dsn not in rendered


def test_the_libpq_password_variable_is_scrubbed_too(monkeypatch):
    monkeypatch.setenv("PGPASSWORD", "env-secret-9")
    index, driver, server = make(dsn="host=127.0.0.1 dbname=appdb")
    driver.connect_failures.append(FakeDbError("connection", "auth failed with env-secret-9"))
    with pytest.raises(PgVectorUnavailable) as caught:
        index.count()
    assert "env-secret-9" not in str(caught.value) and "***" in str(caught.value)


def test_an_unparseable_connection_string_is_reported_without_its_text():
    index, driver, _server = make(dsn="this-is-not-a-dsn-but-looks-like-it-has-secret-KEY")
    with pytest.raises(PgVectorConfigError) as caught:
        index.count()
    assert "secret-KEY" not in str(caught.value) and "not a valid libpq connection string" in str(caught.value)
    assert driver.connections == []


def test_two_threads_never_interleave_statements_on_the_one_connection():
    index, driver, server = make()
    server.delay = 0.002
    errors = []

    def work(seed):
        try:
            for n in range(5):
                index.upsert(f"t{seed}-{n}", unit(seed=n))
                index.search(unit(seed=n), 3)
        except BaseException as exc:
            errors.append(exc)
    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    assert errors == [] and server.max_active == 1 and len(driver.connections) == 1 and len(server.rows) == 20


# --- the TLS rule -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("parsed, environ, allowed", [
    ({"host": "127.0.0.1"}, {}, True), ({"host": "localhost"}, {}, True), ({"host": "::1"}, {}, True),
    ({"host": "/var/run/postgresql"}, {}, True), ({}, {}, True), ({"host": ""}, {}, True), ({"host": "127.0.0.1,::1"}, {}, True),
    ({"host": "127.0.0.1", "sslmode": "disable"}, {}, True),
    ({"host": "db.example.org"}, {}, False), ({"host": "10.1.2.3", "sslmode": "prefer"}, {}, False),
    ({"host": "10.1.2.3", "sslmode": "disable"}, {}, False), ({"host": "10.1.2.3", "sslmode": "require"}, {}, False),
    ({"host": "10.1.2.3", "sslmode": "verify-ca"}, {}, False), ({"host": "10.1.2.3", "sslmode": "allow"}, {}, False),
    ({"host": "10.1.2.3", "sslmode": "verify-full"}, {}, True),
    ({"host": "10.1.2.3"}, {"PGSSLMODE": "verify-full"}, True), ({"host": "10.1.2.3"}, {"PGSSLMODE": "require"}, False),
    ({"host": "10.1.2.3", "sslmode": "verify-full"}, {"PGSSLMODE": "disable"}, True),    # the string wins over the variable
    ({}, {"PGHOST": "db.example.org"}, False), ({}, {"PGHOST": "db.example.org", "PGSSLMODE": "verify-full"}, True),
    ({"host": "127.0.0.1"}, {"PGHOST": "db.example.org"}, True),                         # the string wins over the variable
    ({"host": "127.0.0.1", "hostaddr": "10.9.9.9"}, {}, False),
    ({"host": "127.0.0.1,db.example.org"}, {}, False), ({"host": "db.example.org,127.0.0.1", "sslmode": "verify-full"}, {}, True),
    ({"hostaddr": "192.168.1.5"}, {}, False), ({}, {"PGHOSTADDR": "192.168.1.5"}, False),
    ({"service": "prod", "sslmode": "disable"}, {}, False), ({"service": "prod"}, {}, True),   # forced to verify-full instead
    ({"service": "prod", "sslmode": "verify-full"}, {}, True), ({}, {"PGSERVICE": "prod", "PGSSLMODE": "require"}, False),
    ({"service": "prod", "host": "127.0.0.1"}, {}, True),
])
def test_a_server_that_is_not_this_machine_requires_verify_full(parsed, environ, allowed):
    if allowed:
        check_transport(parsed, environ)
    else:
        with pytest.raises(PgVectorConfigError, match="verify-full"):
            check_transport(parsed, environ)


def test_the_tls_rule_is_applied_before_anything_is_connected_and_leaks_nothing_but_the_mode():
    index, driver, _server = make(dsn=f"host=10.1.2.3 dbname=appdb user=app password={PASSWORD} sslmode=prefer")
    with pytest.raises(PgVectorConfigError) as caught:
        index.count()
    assert driver.connect_calls == [] and PASSWORD not in str(caught.value) and "sslmode=prefer" in str(caught.value)
    index, driver, _server = make(dsn="host=10.1.2.3 dbname=appdb sslmode=verify-full", info=FakeInfo("10.1.2.3", ssl=True,
                                                                                                    sslmode="verify-full"))
    assert index.count() == 0 and driver.connect_calls[0][1].get("sslmode") is None


def test_a_service_without_a_visible_host_is_connected_with_verify_full():
    index, driver, _server = make(dsn="service=prod dbname=appdb")
    index.count()
    assert driver.connect_calls[0][1]["sslmode"] == "verify-full"
    index, driver, _server = make(dsn="service=prod host=127.0.0.1")
    index.count()
    assert "sslmode" not in driver.connect_calls[0][1]


def test_the_connection_libpq_made_is_checked_too():
    insecure = FakeInfo("10.1.2.3", ssl=False)
    index, driver, _server = make(dsn="service=prod", info=insecure)
    with pytest.raises(PgVectorConfigError, match="not a verified TLS connection"):
        index.count()
    assert driver.connections[0].closed and index._conn is None
    weak = FakeInfo("10.1.2.3", ssl=True, sslmode="require")
    with pytest.raises(PgVectorConfigError, match="verified TLS"):
        make(dsn="service=prod", info=weak)[0].count()
    good = FakeInfo("10.1.2.3", ssl=True, sslmode="verify-full")
    index, _driver, _server = make(dsn="service=prod", info=good)
    assert index.count() == 0 and index.describe()["tls"] is True and index.describe()["loopback"] is False
    index, _driver, _server = make(info=FakeInfo("/var/run/postgresql"))
    index.count()
    assert index.describe()["loopback"] is True and index.describe()["tls"] is False


def test_a_filter_travels_as_one_text_parameter_and_separator_ids_as_an_array():
    index, driver, server = make()
    odd = "has\x1fseparator"
    index.upsert_many([("a", unit(seed=1)), ("b", unit(seed=2)), (odd, unit(seed=3))])
    hits = index.search(unit(seed=3), 5, {"a", odd})
    assert {cid for cid, _ in hits} == {"a", odd}
    _sql, params = [entry for entry in server.log if entry[0].startswith("SELECT chunk_id, 1.0 -")][-1]
    assert params["blob"] == "a" and params["odd"] == [odd]
    assert PgVectorIndex._filter_parameters(["x", "y", "z"]) == ("x\x1fy\x1fz", [])
    assert PgVectorIndex._filter_parameters([]) == ("", [])
    assert PgVectorIndex._filter_parameters(["a\x1fb"]) == ("", ["a\x1fb"])


@pytest.mark.parametrize("ids, kept", [
    ({"a", "b"}, {"a", "b"}), (["a", "a", "b"], {"a", "b"}), ((n for n in ["x", "y"]), {"x", "y"}),
    ({"a", None, 3, b"x", "", "ok", "a\x00b", "\ud800", "x" * (MAX_ID_CHARS + 1), "y" * MAX_ID_CHARS}, {"a", "ok", "y" * MAX_ID_CHARS}),
    (frozenset({"p"}), {"p"}), (set(), set()),
])
def test_lookup_ids_keeps_exactly_the_ids_that_could_be_stored(ids, kept):
    found = PgVectorIndex._lookup_ids(ids)
    assert set(found) == kept and len(found) == len(kept)
