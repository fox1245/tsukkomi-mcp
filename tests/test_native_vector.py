"""Native integration and strict runtime selection regression tests."""
import os
from pathlib import Path

import numpy as np
import pytest

from self_directing_mcp import config
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.config import Settings
from self_directing_mcp.embed.embedder import build_embedder, OpenRouterEmbedder, FakeEmbedder
from self_directing_mcp.index.vector import build_vector_index


def test_missing_key_does_not_silently_use_fake():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        build_embedder(api_key=None, use_fake=False)
    assert isinstance(build_embedder(api_key="test", use_fake=False)[0], OpenRouterEmbedder)
    assert isinstance(build_embedder(api_key=None, use_fake=True)[0], FakeEmbedder)


@pytest.mark.parametrize("source", ["argument", "environment"])
def test_shared_key_file_is_authoritative(tmp_path, monkeypatch, source):
    key_file=tmp_path/"shared.env"
    key_file.write_text("OPENROUTER_API_KEY=shared-test-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "different-test-key")
    if source == "environment":
        monkeypatch.setenv("SELF_DIRECT_OPENROUTER_API_KEY_FILE", str(key_file))
        settings = Settings(_env_file=None)
    else:
        settings = Settings(_env_file=None, openrouter_api_key_file=key_file)
    assert settings.resolve_api_key()=="shared-test-key"
    key_file.write_text("OTHER_VALUE=empty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no OPENROUTER_API_KEY"):
        settings.resolve_api_key()
    key_file.unlink()
    with pytest.raises(ValueError, match="does not exist"):
        settings.resolve_api_key()


@pytest.fixture
def isolated_configuration_paths(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    installation = tmp_path / "installation"
    installation.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("SELF_DIRECT_INDEX_DIR", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(config, "__file__", str(
        installation / "src" / "self_directing_mcp" / "config.py"))
    monkeypatch.chdir(installation)
    return home, installation


@pytest.mark.parametrize("source", ["argument", "environment", "dotenv"])
def test_implicit_dotenv_preserves_key_precedence(isolated_configuration_paths, monkeypatch, source):
    _, installation = isolated_configuration_paths
    kwargs = {}
    expected = source + "-fixture-key"
    content = "OTHER_SETTING=fixture\n"
    if source == "dotenv":
        content = "OPENROUTER_API_KEY=" + expected + "\n"
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", "environment-fixture-key")
        if source == "argument":
            kwargs["OPENROUTER_API_KEY"] = expected
    (installation / ".env").write_text(content, encoding="utf-8")
    assert Settings(**kwargs).resolve_api_key() == expected


def test_keyless_dotenv_does_not_block_fake_engine(isolated_configuration_paths):
    _, installation = isolated_configuration_paths
    (installation / ".env").write_text("OTHER_SETTING=fixture\n", encoding="utf-8")
    engine = SelfDirectEngine(Settings(use_fake_embedder=True, dense_backend="numpy"))
    try:
        engine.upsert_contracts([{"id": "fixture-rule", "type": "must_not", "regex": "forbidden"}])
        assert [rule["id"] for rule in engine.list_contracts()["contracts"]] == ["fixture-rule"]
    finally:
        engine.close()


def test_default_index_survives_install_relocation_and_isolates_users(
        isolated_configuration_paths, tmp_path, monkeypatch):
    _, installation = isolated_configuration_paths
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True))
    try:
        engine.upsert_contracts([{"id": "persistent-rule", "type": "must_not", "regex": "forbidden"}])
    finally:
        engine.close()
    monkeypatch.setattr(config, "__file__", str(
        installation / "replacement" / "site-packages" / "self_directing_mcp" / "config.py"))
    reopened = SelfDirectEngine(Settings(_env_file=None, local_only=True))
    try:
        assert [rule["id"] for rule in reopened.list_contracts()["contracts"]] == ["persistent-rule"]
    finally:
        reopened.close()
    other_home = tmp_path / "other-home"
    other_home.mkdir()
    monkeypatch.setenv("HOME", str(other_home))
    monkeypatch.setenv("USERPROFILE", str(other_home))
    other_user = SelfDirectEngine(Settings(_env_file=None, local_only=True))
    try:
        assert other_user.list_contracts()["contracts"] == []
    finally:
        other_user.close()


def test_engine_close_releases_graph_database(tmp_path):
    import sqlite3
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
    result=dense.search(query,4)
    assert [i for i,s in result]==["a","b","c","d"]
    assert [s for i,s in result]==pytest.approx([1,1,0,-1],abs=1e-6)
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
