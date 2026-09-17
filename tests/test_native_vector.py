"""Native integration and strict runtime selection regression tests."""
import os
from pathlib import Path

import numpy as np
import pytest

from self_directing_mcp.config import Settings
from self_directing_mcp.embed.embedder import build_embedder, OpenRouterEmbedder, FakeEmbedder
from self_directing_mcp.index.vector import build_vector_index


def test_missing_key_does_not_silently_use_fake():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        build_embedder(api_key=None, use_fake=False)
    assert isinstance(build_embedder(api_key="test", use_fake=False)[0], OpenRouterEmbedder)
    assert isinstance(build_embedder(api_key=None, use_fake=True)[0], FakeEmbedder)


def test_shared_key_file_is_authoritative(tmp_path, monkeypatch):
    key_file=tmp_path/"shared.env"
    key_file.write_text("OPENROUTER_API_KEY=shared-test-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "different-test-key")
    settings=Settings(_env_file=None, openrouter_api_key_file=key_file)
    assert settings.resolve_api_key()=="shared-test-key"
    key_file.write_text("OTHER_VALUE=empty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no OPENROUTER_API_KEY"):
        settings.resolve_api_key()
    key_file.unlink()
    with pytest.raises(ValueError, match="does not exist"):
        settings.resolve_api_key()


def test_default_backend_is_native(monkeypatch):
    monkeypatch.delenv("SELF_DIRECT_DENSE_BACKEND", raising=False)
    monkeypatch.delenv("SELF_DIRECT_USE_FAKE_EMBEDDER", raising=False)
    settings=Settings(_env_file=None)
    assert settings.dense_backend=="sqlite-vector"
    assert settings.use_fake_embedder is False


def test_engine_close_releases_graph_database(tmp_path):
    import sqlite3
    from self_directing_mcp.engine import SelfDirectEngine
    engine = SelfDirectEngine(Settings(_env_file=None, index_dir=tmp_path, use_fake_embedder=True))
    engine.ensure_ready()
    connection = engine.graph._conn
    engine.close()
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
    (tmp_path / "graph.sqlite").unlink()


def test_invalid_native_path_fails_without_numpy(tmp_path):
    with pytest.raises(RuntimeError, match="sqliteai/sqlite-vector"):
        build_vector_index(tmp_path, dim=2, extension_path=tmp_path/"missing-vector.dll")
    assert not (tmp_path/"dense_numpy.npz").exists()
    # A failed load closes the SQLite connection, including on Windows.
    (tmp_path/"dense.sqlite").unlink()


@pytest.fixture
def native_path():
    path=os.environ.get("SQLITE_VECTOR_TEST_PATH")
    if not path:
        pytest.skip("Set SQLITE_VECTOR_TEST_PATH to exercise the actual release library")
    assert Path(path).is_file()
    return Path(path)


def test_native_cosine_persistence_filter_and_readonly(tmp_path, native_path):
    dense,backend=build_vector_index(tmp_path,dim=3,extension_path=native_path)
    assert backend=="sqlite-vector"
    assert dense.extension_version=="1.1.0"
    # Non-unit vectors prove that cosine (not raw dot product) is computed.
    dense.upsert_many([("b",np.array([2.,0.,0.])),("a",np.array([1.,0.,0.])),
                       ("c",np.array([0.,3.,0.])),("d",np.array([-1.,0.,0.]))])
    query=np.array([4.,0.,0.])
    statements=[]
    dense._conn.set_trace_callback(statements.append)
    result=dense.search(query,4)
    assert [i for i,s in result]==["a","b","c","d"]
    assert [s for i,s in result]==pytest.approx([1,1,0,-1],abs=1e-6)
    assert any("vector_full_scan" in sql for sql in statements)
    assert dense.search(query,1,allowed_ids={"c","d"})==[("c",0.)]
    assert dense.search(query,2,allowed_ids=set())==[]
    assert dense.search_readonly(query,4)==result
    dense.close()
    dense,_=build_vector_index(tmp_path,dim=3,extension_path=native_path)
    try:
        assert dense.count()==4
        assert dense.search(query,4)==result
        dense.delete_ids(["a","b"])
        assert dense.search_readonly(query,1)==[("c",0.)]
        with pytest.raises(ValueError):
            dense.upsert_many([("valid",np.ones(3)),("bad",np.ones(2))])
        assert dense.count()==2  # Batch validation cannot partially write.
        dense.clear()
        assert dense.count()==0
    finally:
        dense.close()
