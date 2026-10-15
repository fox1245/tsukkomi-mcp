"""The journal setting through the engine: status reporting, close/idle checkpoints, short-lived hook engines, and the
receipt/event pairing that makes a half-finished pair of commits visible. Synthetic data only.

The pairing needs no new marker. The events database stores, with every cursor it publishes, the SHA-256 of the
receipt rows the events were derived from (``session_receipt_versions.digest``) and the receipts they depend on
(``session_receipt_deps``). Every judgement compares that digest with the digest of the receipts database as it stands
(``receipts_not_indexed`` makes the coverage incomplete, which a verdict reports as ``unknown``). A receipt committed
without the events commit that follows it, and an events commit whose receipt commit is missing, both differ in that
digest, in either journal mode.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

from self_directing_mcp import journal, settings_core
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index.ingest import SessionStore
from self_directing_mcp.journal import JournalChoice
from test_agy_ledger import SID as AGY_SID, Session, append, blocks, planner
from test_journal import run_and_die
from test_reliability import SID, message, write as write_codex


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setattr(journal, "choose_journal_cached", lambda path: JournalChoice("wal", "local:test"))


@pytest.fixture
def denied(monkeypatch):
    monkeypatch.setattr(journal, "choose_journal_cached", lambda path: JournalChoice("delete", "denied:9p"))


def codex_engine(tmp_path, monkeypatch, **extra):
    for name in ("SELF_DIRECT_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "sessions"
    root.mkdir(exist_ok=True)
    path = root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl"
    write_codex(path, [{"type": "session_meta", "payload": {"id": SID}}, message("hello")])
    settings = Settings(_env_file=None, local_only=True, openrouter_api_key_file=None, codex_sessions_dir=root,
                        index_dir=tmp_path / "index", **{"sqlite_journal": "auto", **extra})
    return settings, path


def wal_size(index: Path, name="meta.sqlite") -> int:
    wal = Path(str(index / name) + "-wal")
    return wal.stat().st_size if wal.exists() else 0


# --- the setting --------------------------------------------------------------------------------------------------

def test_the_setting_is_validated_and_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("SELF_DIRECT_SQLITE_JOURNAL", "delete")
    assert Settings(_env_file=None).sqlite_journal == "delete"
    assert settings_core.load_hook_settings().sqlite_journal == "delete"
    monkeypatch.setenv("SELF_DIRECT_SQLITE_JOURNAL", "wal")
    with pytest.raises(ValueError):
        Settings(_env_file=None)
    with pytest.raises(ValueError):
        settings_core.load_hook_settings()
    monkeypatch.delenv("SELF_DIRECT_SQLITE_JOURNAL")
    assert Settings(_env_file=None).sqlite_journal == settings_core.DEFAULT_SQLITE_JOURNAL == journal.DEFAULT_POLICY


# --- audit_status -------------------------------------------------------------------------------------------------

def test_audit_status_reports_the_journal_of_both_databases(tmp_path, monkeypatch, local):
    settings, _path = codex_engine(tmp_path, monkeypatch)
    engine = SelfDirectEngine(settings)
    try:
        status = engine.audit_status()
        assert status["journal_fallback"] == []
        report = status["journal"]
        assert report["policy"] == "auto" and report["synchronous"] == "FULL"
        meta = report["databases"]["meta.sqlite"]
        assert (meta["mode"], meta["wanted"], meta["reason"], meta["pending"], meta["fallback"]) == \
            ("wal", "wal", "local:test", None, False)
        receipts = report["databases"]["agy-receipts.sqlite"]
        assert receipts["mode"] is None and receipts["wanted"] == "wal"          # no receipts yet: nothing created
        assert not (tmp_path / "index" / "agy-receipts.sqlite").exists()
    finally:
        engine.close()


def test_audit_status_names_a_fallback_and_the_log_says_so(tmp_path, monkeypatch, denied, caplog):
    monkeypatch.setattr(journal, "_LOGGED", set())
    settings, _path = codex_engine(tmp_path, monkeypatch)
    engine = SelfDirectEngine(settings)
    try:
        with caplog.at_level(logging.INFO, logger="self_directing_mcp.journal"):
            engine.ensure_ready()
            status = engine.audit_status()
        assert status["journal_fallback"] == [{"database": "meta.sqlite", "reason": "denied:9p"},
                                              {"database": "agy-receipts.sqlite", "reason": "denied:9p"}]
        assert status["journal"]["databases"]["meta.sqlite"]["mode"] == "delete"
        assert wal_size(tmp_path / "index") == 0
        messages = [record.getMessage() for record in caplog.records if "journal_fallback" in record.getMessage()]
        assert len(messages) == 1 and "denied:9p" in messages[0]            # once per directory and reason
    finally:
        engine.close()


def test_the_delete_setting_is_not_a_fallback(tmp_path, monkeypatch, local):
    settings, _path = codex_engine(tmp_path, monkeypatch, sqlite_journal="delete")
    engine = SelfDirectEngine(settings)
    try:
        status = engine.audit_status()
        assert status["journal_fallback"] == []
        meta = status["journal"]["databases"]["meta.sqlite"]
        assert (meta["mode"], meta["reason"], meta["fallback"]) == ("delete", "setting:delete", False)
    finally:
        engine.close()


def test_an_unrecognised_file_system_is_reported_as_unknown_with_the_detail(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "choose_journal_cached", lambda path: JournalChoice("delete", "unknown:ntfs3"))
    settings, _path = codex_engine(tmp_path, monkeypatch)
    engine = SelfDirectEngine(settings)
    try:
        assert engine.audit_status()["journal_fallback"][0] == {"database": "meta.sqlite", "reason": "unknown:ntfs3"}
    finally:
        engine.close()


# --- close, idle and short-lived engines --------------------------------------------------------------------------

def test_a_resident_engine_checkpoints_when_it_closes(tmp_path, monkeypatch, local):
    settings, path = codex_engine(tmp_path, monkeypatch)
    engine = SelfDirectEngine(settings)
    info = engine.sync_session(session_id=SID, path=str(path), embed=False)
    assert info["ok"] and info["new_chunks"] >= 1
    index = tmp_path / "index"
    assert wal_size(index) > 0 or engine.store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    engine.close()
    assert wal_size(index) == 0
    with closing(sqlite3.connect(index / "meta.sqlite")) as connection:       # the main file alone has the events
        assert connection.execute("SELECT COUNT(*) FROM session_events").fetchone()[0] >= 1


def test_a_hook_killed_before_it_closes_leaves_a_log_that_the_next_process_recovers(tmp_path, monkeypatch, local):
    settings, path = codex_engine(tmp_path, monkeypatch)
    index = tmp_path / "index"
    run_and_die(f"""
from pathlib import Path
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
settings = Settings(_env_file=None, local_only=True, openrouter_api_key_file=None, sqlite_journal='auto',
                    codex_sessions_dir=Path({str(path.parent)!r}), index_dir=Path({str(index)!r}))
engine = SelfDirectEngine(settings)
assert engine.sync_session(session_id={SID!r}, path={str(path)!r}, embed=False)['ok']
""")
    assert wal_size(index) > 0
    with closing(sqlite3.connect(index / "meta.sqlite")) as plain:           # an older process: plain sqlite3
        assert plain.execute("SELECT COUNT(*) FROM session_events").fetchone()[0] >= 1
        assert plain.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    write_codex(path, [message("second")], "a")
    server = SelfDirectEngine(settings)
    info = server.sync_session(session_id=SID, path=str(path), embed=False)
    assert info["ok"] and info["reindexed"] is False and info["new_chunks"] == 1
    server.close()
    assert wal_size(index) == 0                                  # a clean close folded the log


@pytest.mark.skipif(sys.platform != "linux", reason="the resident daemon is Linux only")
def test_the_daemons_exit_closes_its_engines_and_folds_the_log(tmp_path, monkeypatch, local):
    """Idle exit, drain and shutdown all end in ``Engines.close_all``; a resident engine's log must not outlive it."""
    from self_directing_mcp.daemon.core import Engines
    settings, path = codex_engine(tmp_path, monkeypatch)
    engines = Engines()
    engine = engines.get(settings)
    assert engine.sync_session(session_id=SID, path=str(path), embed=False)["ok"]
    index = tmp_path / "index"
    assert wal_size(index) > 0                                   # committed, in the log, the engine still open
    engines.close_all()
    assert wal_size(index) == 0
    with closing(sqlite3.connect(index / "meta.sqlite")) as plain:
        assert plain.execute("SELECT COUNT(*) FROM session_events").fetchone()[0] >= 1


def test_the_idle_flush_retries_a_refused_switch_and_never_fails(tmp_path, monkeypatch, local):
    settings, path = codex_engine(tmp_path, monkeypatch)
    SessionStore(tmp_path / "index" / "meta.sqlite", journal="delete").close()        # an index still in DELETE mode
    holder = sqlite3.connect(tmp_path / "index" / "meta.sqlite")
    holder.execute("BEGIN")
    holder.execute("SELECT COUNT(*) FROM session_events").fetchall()                  # another process reads
    engine = SelfDirectEngine(settings, defer_derived=True)
    try:
        engine.ensure_ready()
        assert (engine.store.journal.mode, engine.store.journal.pending) == ("delete", "busy")
    finally:
        holder.rollback()
        holder.close()
    assert engine.sync_session(session_id=SID, path=str(path), embed=False)["ok"]     # served in DELETE mode meanwhile
    engine.flush_derived()                                       # idle, too soon after the refusal: nothing happens
    assert engine.store.journal.pending == "busy"
    engine.store.journal.last_attempt -= 3600
    engine.flush_derived()                                       # what the daemon calls when it has been idle
    assert engine.store.journal.pending is None and engine.store.journal.mode == "wal"
    assert engine.audit_status()["journal"]["databases"]["meta.sqlite"]["mode"] == "wal"
    engine.close()


# --- receipt / event pairing ----------------------------------------------------------------------------------------

@pytest.fixture
def agy(tmp_path, monkeypatch, local):
    monkeypatch.setenv("SELF_DIRECT_SQLITE_JOURNAL", "auto")
    world = Session(tmp_path)
    yield world
    world.engine.close()


def issues(world):
    return world.engine.audit_status(AGY_SID, provider="agy")["session"]


def test_both_databases_are_in_wal_mode_for_agy(agy):
    blocks(agy, 2)
    agy.sync()
    report = agy.engine.audit_status(AGY_SID, provider="agy")["journal"]["databases"]
    assert report["meta.sqlite"]["mode"] == "wal" and report["agy-receipts.sqlite"]["mode"] == "wal"


def test_a_receipt_committed_without_its_events_commit_is_visible_until_the_events_follow(agy):
    step = blocks(agy, 2)
    agy.sync()
    assert "receipts_not_indexed" not in issues(agy)["issues"] and issues(agy)["complete"] is True
    append(agy.path, planner(step))
    agy.pre(step + 1)                                        # the receipt commit lands; the events commit has not
    session = issues(agy)
    assert "receipts_not_indexed" in session["issues"] and session["complete"] is False
    agy.sync()                                               # the events commit follows (a plain append: no reparse)
    session = issues(agy)
    assert "receipts_not_indexed" not in session["issues"]


def test_an_events_commit_whose_receipt_commit_is_missing_is_visible_too(agy):
    step = blocks(agy, 3)
    agy.sync()
    assert "receipts_not_indexed" not in issues(agy)["issues"]
    with closing(sqlite3.connect(agy.receipts.path)) as connection:         # the receipt rows as an older commit left them
        connection.execute("DELETE FROM receipts WHERE step=?", (step - 1,))
        connection.commit()
    session = issues(agy)
    assert "receipts_not_indexed" in session["issues"] and session["complete"] is False


def test_the_digest_that_ties_the_two_databases_is_stored_with_the_cursor(agy):
    blocks(agy, 2)
    info = agy.sync()
    token = agy.receipts.authority_token(AGY_SID)
    assert info["coverage"]["receipt_digest"] == token[2]
    with closing(sqlite3.connect(agy.database)) as connection:
        stored = connection.execute("SELECT digest FROM session_receipt_versions WHERE session_id=?",
                                    (AGY_SID,)).fetchone()
        deps = connection.execute("SELECT deps FROM session_receipt_deps WHERE session_id=?", (AGY_SID,)).fetchone()
    assert stored == (token[2],) and json.loads(deps[0])["steps"]
