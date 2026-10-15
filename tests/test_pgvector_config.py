"""The pgvector settings, the connection-string file rules, the builder and the engine's use of the backend that need no
server: a fake driver stands in for psycopg (``pgvector_fakes``). Nothing here needs psycopg or Docker."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from pgvector_fakes import FakeDbError, FakeDriver, FakeServer
from self_directing_mcp import settings_core
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.hook_policy import wants_derived_work
from self_directing_mcp.index import pgvector
from self_directing_mcp.index.vector import DenseBackendError, build_vector_index
from self_directing_mcp.settings_core import read_pgvector_dsn_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SAMPLE_ROOT = FIXTURES / "sample_session"
SESSION_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
SECRET = "SENTINEL-password-0451"
POSIX = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")


def dsn_file(tmp_path, text, mode=0o600, name="dsn.txt"):
    path = tmp_path / name
    path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    os.chmod(path, mode)
    return path


# --- the connection-string file -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    (f"host=127.0.0.1 dbname=x password={SECRET}\n", f"host=127.0.0.1 dbname=x password={SECRET}"),
    (f"postgresql://u:{SECRET}@127.0.0.1/x", f"postgresql://u:{SECRET}@127.0.0.1/x"),
    ("﻿host=127.0.0.1\r\n", "host=127.0.0.1"), ("  host=127.0.0.1  \n\n\n", "host=127.0.0.1"),
])
def test_a_private_single_line_file_is_read(tmp_path, text, expected):
    assert read_pgvector_dsn_file(dsn_file(tmp_path, text)) == expected


@pytest.mark.parametrize("content", [
    "", "\n\n", f"host=a password={SECRET}\nhost=b\n", f"host=a password={SECRET}\x00\n", b"host=\xff\xfe password=" + SECRET.encode(),
    f"host=a password={SECRET} " + "x" * 5000,
])
def test_a_file_that_is_not_one_connection_string_is_refused_without_echoing_it(tmp_path, content):
    with pytest.raises(ValueError) as caught:
        read_pgvector_dsn_file(dsn_file(tmp_path, content))
    assert SECRET not in str(caught.value) and str(tmp_path) not in str(caught.value)


@POSIX
@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o660, 0o666, 0o602, 0o620, 0o750])
def test_a_file_readable_or_writable_by_group_or_others_is_refused(tmp_path, mode):
    path = dsn_file(tmp_path, f"host=127.0.0.1 password={SECRET}", mode)
    with pytest.raises(ValueError, match="chmod 600") as caught:
        read_pgvector_dsn_file(path)
    assert SECRET not in str(caught.value) and str(path) not in str(caught.value)


@POSIX
@pytest.mark.parametrize("mode", [0o600, 0o400, 0o700])
def test_an_owner_only_file_is_accepted(tmp_path, mode):
    assert read_pgvector_dsn_file(dsn_file(tmp_path, "host=127.0.0.1", mode)) == "host=127.0.0.1"


@POSIX
def test_a_missing_file_a_directory_and_a_fifo_are_refused_and_never_block(tmp_path):
    with pytest.raises(ValueError, match="cannot be opened"):
        read_pgvector_dsn_file(tmp_path / "absent")
    with pytest.raises(ValueError, match="regular file"):
        read_pgvector_dsn_file(tmp_path)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo, 0o600)
    with pytest.raises(ValueError, match="regular file"):
        read_pgvector_dsn_file(fifo)


@POSIX
def test_a_symlink_to_a_private_file_is_followed_and_to_a_shared_one_is_refused(tmp_path):
    private = dsn_file(tmp_path, "host=127.0.0.1", 0o600, "real.txt")
    link = tmp_path / "link.txt"
    link.symlink_to(private)
    assert read_pgvector_dsn_file(link) == "host=127.0.0.1"
    os.chmod(private, 0o644)
    with pytest.raises(ValueError, match="chmod 600"):
        read_pgvector_dsn_file(link)


def test_the_setting_resolves_the_file_or_the_libpq_environment_alone(tmp_path, monkeypatch):
    monkeypatch.delenv("SELF_DIRECT_PGVECTOR_DSN", raising=False)
    assert Settings(_env_file=None).resolve_pgvector_dsn() == ""
    path = dsn_file(tmp_path, "host=127.0.0.1")
    assert Settings(_env_file=None, pgvector_dsn_file=path).resolve_pgvector_dsn() == "host=127.0.0.1"
    monkeypatch.setenv("SELF_DIRECT_PGVECTOR_DSN_FILE", str(path))
    assert Settings(_env_file=None).resolve_pgvector_dsn() == "host=127.0.0.1"


def test_a_connection_string_in_an_environment_variable_is_refused_by_name(monkeypatch):
    monkeypatch.setenv("SELF_DIRECT_PGVECTOR_DSN", f"postgresql://u:{SECRET}@127.0.0.1/x")
    with pytest.raises(ValueError, match="SELF_DIRECT_PGVECTOR_DSN_FILE") as caught:
        Settings(_env_file=None).resolve_pgvector_dsn()
    assert SECRET not in str(caught.value)


def test_the_pgvector_settings_defaults_and_validation(monkeypatch):
    settings = Settings(_env_file=None, dense_backend="pgvector")
    assert (settings.pgvector_schema, settings.pgvector_hnsw, settings.pgvector_dsn_file) == ("tsukkomi", False, None)
    for bad in ("1abc", "a b", "pg_catalog", "x" * 64, ""):
        with pytest.raises(ValueError):
            Settings(_env_file=None, pgvector_schema=bad)
    with pytest.raises(ValueError):
        Settings(_env_file=None, dense_backend="postgres")
    monkeypatch.setenv("SELF_DIRECT_DENSE_BACKEND", "pgvector")
    monkeypatch.setenv("SELF_DIRECT_PGVECTOR_HNSW", "true")
    assert Settings(_env_file=None).dense_backend == "pgvector" and Settings(_env_file=None).pgvector_hnsw is True


def test_the_default_backend_of_a_fresh_environment_is_sqlite_vector(monkeypatch):
    monkeypatch.delenv("SELF_DIRECT_DENSE_BACKEND", raising=False)
    assert Settings(_env_file=None).dense_backend == settings_core.DEFAULT_DENSE_BACKEND == "sqlite-vector"


# --- the builder and the import rule --------------------------------------------------------------------------------------

def test_the_builder_returns_the_pgvector_index_and_names_it(tmp_path, monkeypatch):
    monkeypatch.setattr(pgvector, "PsycopgDriver", FakeDriver)
    dense, name = build_vector_index(tmp_path / "vectors-v2-abc", dim=4, backend="pgvector", dsn="host=127.0.0.1 dbname=x",
                                     schema="s1", hnsw=False, policy="abc")
    assert name == "pgvector" and isinstance(dense, pgvector.PgVectorIndex) and dense.remote is True
    assert (dense.schema, dense.dim, dense.policy) == ("s1", 4, "abc")
    assert (tmp_path / "vectors-v2-abc").is_dir()           # what tells a local-only process that vector cleanup is owed
    with pytest.raises(ValueError, match="Unknown dense backend"):
        build_vector_index(tmp_path, backend="postgres")


def test_the_other_backends_are_not_remote_and_describe_nothing(tmp_path):
    dense, name = build_vector_index(tmp_path, dim=4, backend="numpy")
    assert name == "numpy" and dense.remote is False and dense.describe() == {}


def run_python(code: str) -> str:
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=environment, timeout=120)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def test_neither_the_package_nor_the_backend_module_imports_psycopg():
    out = run_python("import sys, self_directing_mcp.engine, self_directing_mcp.index.vector as v;"
                     "print('psycopg' in sys.modules, 'self_directing_mcp.index.pgvector' in sys.modules);"
                     "import self_directing_mcp.index.pgvector;"
                     "print('psycopg' in sys.modules)")
    assert out.splitlines() == ["False False", "False"]


@pytest.mark.skipif(importlib.util.find_spec("psycopg") is None, reason="psycopg is not installed")
def test_building_the_index_does_not_import_psycopg_before_the_first_query():
    out = run_python("import sys, tempfile\n"
                     "from self_directing_mcp.index.vector import build_vector_index\n"
                     "dense, name = build_vector_index(tempfile.mkdtemp(), dim=4, backend='pgvector', dsn='host=127.0.0.1')\n"
                     "print(name, 'psycopg' in sys.modules)")
    assert out == "pgvector False"


# --- the engine: what is local, what is remote -----------------------------------------------------------------------------

def settings_for(tmp_path, **options):
    return Settings(_env_file=None, codex_sessions_dir=SAMPLE_ROOT, index_dir=tmp_path / "index", use_fake_embedder=True,
                    contracts_path=tmp_path / "index" / "contracts.json", **options)


@pytest.fixture
def fake_server(monkeypatch):
    """The engine's pgvector backend over a fake driver sharing one in-memory 'database'."""
    server = FakeServer(dim=1024)
    driver = FakeDriver(server)
    monkeypatch.setattr(pgvector, "PsycopgDriver", lambda: driver)
    return server, driver


def pg_settings(tmp_path, **options):
    path = dsn_file(tmp_path, f"host=127.0.0.1 port=5433 dbname=appdb user=app password={SECRET}")
    return settings_for(tmp_path, dense_backend="pgvector", pgvector_dsn_file=path, **options)


def test_the_hook_path_never_asks_for_derived_work_when_the_dense_index_is_a_server(tmp_path):
    for backend, expected in (("sqlite-vector", True), ("numpy", True), ("pgvector", False)):
        engine = types.SimpleNamespace(settings=settings_for(tmp_path, dense_backend=backend))
        assert wants_derived_work(engine) is expected
    assert wants_derived_work(types.SimpleNamespace(settings=settings_for(tmp_path, local_only=True,
                                                                          dense_backend="pgvector"))) is False


def test_a_derived_free_sync_never_touches_the_server(tmp_path, fake_server):
    server, driver = fake_server
    engine = SelfDirectEngine(pg_settings(tmp_path))
    try:
        result = engine.sync_session(session_id=SESSION_ID, embed=False, derived=False)
        assert result["ok"] is True and result["dense_backend"] == "pgvector"
        assert driver.connect_calls == [] and server.log == []        # the audit path is local: no connection, no query
    finally:
        engine.close()


def test_local_only_mode_builds_no_dense_index_and_never_reads_the_connection_file(tmp_path, fake_server):
    server, driver = fake_server
    engine = SelfDirectEngine(settings_for(tmp_path, local_only=True, dense_backend="pgvector",
                                           pgvector_dsn_file=tmp_path / "absent-and-never-read"))
    try:
        engine.ensure_ready()
        assert engine.dense is None and engine.dense_backend == "disabled"
        with pytest.raises(ValueError, match="local-only"):
            engine.search_history("curl", mode="dense")
        assert driver.connect_calls == []
    finally:
        engine.close()


def test_a_bad_connection_file_stops_the_engine_before_anything_is_connected(tmp_path, fake_server):
    server, driver = fake_server
    path = dsn_file(tmp_path, f"host=127.0.0.1 password={SECRET}", 0o644 if os.name == "posix" else 0o600)
    engine = SelfDirectEngine(settings_for(tmp_path, dense_backend="pgvector", pgvector_dsn_file=path))
    if os.name == "posix":
        with pytest.raises(ValueError, match="chmod 600") as caught:
            engine.ensure_ready()
        assert SECRET not in str(caught.value)
    assert driver.connect_calls == []


def test_the_engine_syncs_searches_and_reports_a_server_it_never_replaces(tmp_path, fake_server):
    server, driver = fake_server
    engine = SelfDirectEngine(pg_settings(tmp_path))
    try:
        synced = engine.sync_session(session_id=SESSION_ID)
        assert synced["ok"] is True and synced["dense_backend"] == "pgvector" and synced["embedded"] >= 1
        assert len(server.rows) == synced["total_chunks"]
        found = engine.search_history("shell curl api_key", session_id=SESSION_ID, mode="hybrid", top_k=5)
        assert found["ok"] is True and found["hits"]
        status = engine.audit_status()
        assert status["dense_backend"] == "pgvector" and status["dense_count"] == len(server.rows)
        assert status["dense_info"]["host"] == "127.0.0.1" and status["dense_info"]["database"] == "appdb"
        assert SECRET not in repr(status) and "dense_numpy_fallback" not in status["degraded_flags"]
        assert status["degraded_flags"] == []                               # honest: nothing is degraded or substituted
    finally:
        engine.close()


def test_an_unreachable_server_is_an_error_everywhere_and_never_a_numpy_index(tmp_path, fake_server):
    server, driver = fake_server
    driver.connect_failures.extend(FakeDbError("connection", f"connection refused (password {SECRET})") for _ in range(20))
    engine = SelfDirectEngine(pg_settings(tmp_path))
    try:
        synced = engine.sync_session(session_id=SESSION_ID)
        assert synced["ok"] is False and synced["error"] == "dense_backend_unavailable" and synced["dense_backend"] == "pgvector"
        assert "SELF_DIRECT_PGVECTOR_DSN_FILE" in synced["message"] and SECRET not in repr(synced)
        assert engine.store.chunk_count() > 0                              # the canonical events were published first
        found = engine.search_history("curl", session_id=SESSION_ID, mode="hybrid")
        assert found["ok"] is False and found["error"] == "dense_backend_unavailable"
        found = engine.search_history("curl", session_id=SESSION_ID, mode="dense")
        assert found["ok"] is False and found["error"] == "dense_backend_unavailable"
        sparse = engine.search_history("curl", session_id=SESSION_ID, mode="sparse")
        assert sparse["ok"] is True                                         # modes that never use the dense index still work
        status = engine.audit_status()
        assert status["dense_backend"] == "pgvector" and status["dense_count"] is None
        assert "dense_backend_unavailable" in status["degraded_flags"] and "dense_numpy_fallback" not in status["degraded_flags"]
        assert "SELF_DIRECT_PGVECTOR_DSN_FILE" in status["dense_error"] and SECRET not in repr(status)
        assert isinstance(engine.dense, pgvector.PgVectorIndex) and engine.dense_backend == "pgvector"
    finally:
        engine.close()
    assert driver.connections == []
