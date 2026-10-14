"""When write-ahead logging is allowed, how a connection switches, and the WAL-aware copy helpers. Synthetic data only.

* ``choose_journal``: a table of fabricated ``/proc/self/mountinfo`` samples and fake Win32 answers (the real mount table
  of the machine running the tests is never consulted by the table tests);
* ``apply_journal`` / the stores: the mode is applied on open, a second plain ``sqlite3`` connection (what an older
  release is) reads and writes the same database, a refused switch keeps the old mode and fails nothing, ``close``
  checkpoints;
* ``copy_database`` / ``copy_index_dir``: a copy of a database that still has committed pages only in its ``-wal`` loses
  nothing.
"""
from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import types
from contextlib import closing
from pathlib import Path

import pytest

from self_directing_mcp import journal
from self_directing_mcp.agy_receipts import ReceiptStore
from self_directing_mcp.index.ingest import SessionStore
from self_directing_mcp.journal import JournalChoice
from self_directing_mcp.schemas import Chunk


# --- fabricated hosts -------------------------------------------------------------------------------------------

def mountinfo(*mounts):
    """``/proc/self/mountinfo`` text for ``(mount point, fstype[, source[, super options]])`` tuples, in order; mount
    number n has device ``n:0``."""
    lines = []
    for number, (point, fstype, *rest) in enumerate(mounts, start=1):
        source = rest[0] if rest else "none"
        options = rest[1] if len(rest) > 1 else "rw"
        lines.append(f"{number + 20} 1 {number}:0 / {point} rw,relatime shared:{number} - {fstype} {source} {options}")
    return "\n".join(lines) + "\n"


class FakeHost(journal._Host):
    def __init__(self, text=None, *, read_only=False, devices=None, statvfs_error=None, links=None):
        self.text, self.read_only, self.devices = text, read_only, devices or {}
        self.statvfs_error, self.links = statvfs_error, links or {}

    def realpath(self, path):
        for source, target in self.links.items():
            if path == source or path.startswith(source + "/"):
                return target + path[len(source):]
        return path

    def isdir(self, path):
        return not path.endswith(".sqlite")          # the database file itself; its directory is what is probed

    def statvfs(self, path):
        if self.statvfs_error:
            raise self.statvfs_error
        return types.SimpleNamespace(f_flag=1 if self.read_only else 0)

    def device(self, path):
        return self.devices.get(path)

    def mountinfo(self):
        if self.text is None:
            raise FileNotFoundError("/proc/self/mountinfo")
        return self.text


def linux(text, path="/data/index/meta.sqlite", **kwargs):
    return journal.choose_journal(path, _host=FakeHost(text, **kwargs), _system="Linux")


@pytest.mark.parametrize("fstype", ["ext2", "ext3", "ext4", "xfs", "btrfs", "tmpfs", "f2fs", "zfs"])
def test_local_file_systems_allow_wal(fstype):
    choice = linux(mountinfo(("/", "ext4"), ("/data", fstype)))
    assert choice == JournalChoice("wal", f"local:{fstype}") and not choice.fallback


@pytest.mark.parametrize("fstype", ["9p", "drvfs", "nfs", "nfs4", "cifs", "smb3", "smbfs", "fuse", "fuse.sshfs",
                                    "fuseblk", "fuse.gcsfuse", "virtiofs", "NFS4"])
def test_network_and_virtual_file_systems_are_denied(fstype):
    choice = linux(mountinfo(("/", "ext4"), ("/data", fstype)))
    assert choice.mode == "delete" and choice.reason == f"denied:{fstype.lower()}" and choice.fallback


@pytest.mark.parametrize("fstype", ["ntfs3", "ntfs", "vfat", "exfat", "reiserfs", "bcachefs", "somethingnew", "ramfs"])
def test_unlisted_file_systems_are_unknown_not_local(fstype):
    choice = linux(mountinfo(("/", "ext4"), ("/data", fstype)))
    assert choice.mode == "delete" and choice.reason == f"unknown:{fstype}" and choice.fallback


def test_the_wsl_drive_mount_is_denied_as_9p():
    text = ("82 67 8:48 / / rw,relatime - ext4 /dev/sdd rw,discard,errors=remount-ro,data=ordered\n"
            "139 82 0:77 / /mnt/d rw,noatime - 9p D:\\134 rw,aname=drvfs;path=D:\\;uid=1000;gid=1000;"
            "symlinkroot=/mnt/,cache=0x5,access=client,msize=65536,trans=fd,rfd=6,wfd=6\n")
    assert linux(text, "/mnt/d/work/repo/.index/meta.sqlite") == JournalChoice("delete", "denied:9p")
    assert linux(text, "/home/user/.self_direct_index/meta.sqlite") == JournalChoice("wal", "local:ext4")


def test_the_longest_mount_point_decides():
    text = mountinfo(("/", "ext4"), ("/data", "nfs"), ("/data/fast", "xfs"), ("/databases", "tmpfs"))
    assert linux(text, "/data/x/meta.sqlite").reason == "denied:nfs"
    assert linux(text, "/data/fast/x/meta.sqlite").reason == "local:xfs"
    assert linux(text, "/databases/meta.sqlite").reason == "local:tmpfs"      # not a prefix match on the name
    assert linux(text, "/other/meta.sqlite").reason == "local:ext4"


def test_a_mount_stacked_over_another_shadows_it():
    text = mountinfo(("/", "ext4"), ("/data", "ext4"), ("/data", "9p"))
    assert linux(text).reason == "denied:9p"
    devices = {"/data/index": (2, 0)}            # the directory's device names the lower /data mount
    assert linux(text, devices=devices).reason == "local:ext4"


def test_mount_points_with_escaped_spaces_are_unescaped():
    text = mountinfo(("/", "ext4"), ("/my\\040data", "nfs"))
    assert linux(text, "/my data/index/meta.sqlite").reason == "denied:nfs"


def test_symlinks_are_resolved_before_the_mount_is_looked_up():
    text = mountinfo(("/", "ext4"), ("/net", "nfs4"))
    assert linux(text, "/home/u/index/meta.sqlite", links={"/home/u/index": "/net/u/index"}).reason == "denied:nfs4"


def test_missing_unreadable_or_garbage_mount_tables_mean_delete():
    assert linux(None) == JournalChoice("delete", "unknown:mountinfo-FileNotFoundError")
    assert linux("") == JournalChoice("delete", "unknown:mountinfo-empty")
    assert linux("this is not a mount table\nneither is this\n") == JournalChoice("delete", "unknown:mountinfo-empty")
    assert linux("1 2 3:4 / /data rw\n") == JournalChoice("delete", "unknown:mountinfo-empty")   # no ' - ' separator
    assert linux(mountinfo(("/elsewhere", "ext4"))) == JournalChoice("delete", "unknown:no-mount")


def test_a_read_only_file_system_or_a_failed_statvfs_means_delete():
    text = mountinfo(("/", "ext4"))
    assert linux(text, read_only=True) == JournalChoice("delete", "unknown:read-only")
    assert linux(text, statvfs_error=PermissionError()) == JournalChoice("delete", "unknown:statvfs-PermissionError")


def test_a_probe_that_raises_is_unknown_not_an_error():
    class Broken(FakeHost):
        def realpath(self, path):
            raise RuntimeError("boom")

    choice = journal.choose_journal("/data/x", _host=Broken(mountinfo(("/", "ext4"))), _system="Linux")
    assert choice == JournalChoice("delete", "unknown:probe-RuntimeError")


def overlay(upper_fs, *, options="lowerdir=/l,upperdir=/up/upper,workdir=/up/work", devices=None):
    text = mountinfo(("/", "overlay", "none", options), ("/up", upper_fs) if upper_fs else ("/zz", "ext4"))
    return linux(text, "/var/index/meta.sqlite", devices={"/up/upper": (2, 0)} if devices is None else devices)


def test_an_overlay_is_local_only_when_its_upper_directory_is_shown_to_be():
    assert overlay("ext4") == JournalChoice("wal", "local:overlay(upper=ext4)")
    assert overlay("xfs") == JournalChoice("wal", "local:overlay(upper=xfs)")
    assert overlay("9p") == JournalChoice("delete", "denied:overlay(upper=9p)")
    assert overlay("ntfs3") == JournalChoice("delete", "unknown:overlay(upper=ntfs3)")


def test_an_overlay_whose_upper_directory_cannot_be_verified_is_unknown():
    assert overlay("ext4", devices={}).reason == "unknown:overlay-upper-unverifiable"          # not visible from here
    assert overlay("ext4", devices={"/up/upper": (9, 9)}).reason == "unknown:overlay-upper-unverifiable"   # other device
    assert overlay(None).reason == "unknown:overlay-upper-unverifiable"
    assert overlay("ext4", options="lowerdir=/l").reason == "unknown:overlay-without-upperdir"
    nested = mountinfo(("/", "overlay", "none", "lowerdir=/l,upperdir=/up/upper"),
                       ("/up", "overlay", "none", "lowerdir=/m,upperdir=/x"))
    assert linux(nested, "/var/m.sqlite", devices={"/up/upper": (2, 0)}).reason == "unknown:overlay-upper-unverifiable"


def test_other_platforms_are_unknown():
    assert journal.choose_journal("/x", _system="darwin") == JournalChoice("delete", "unknown:platform-darwin")


def test_the_real_machine_gets_an_answer_and_a_missing_path_is_not_an_error(tmp_path):
    for path in (tmp_path / "meta.sqlite", tmp_path / "not" / "yet" / "meta.sqlite", tmp_path):
        choice = journal.choose_journal(path)
        assert choice.mode in ("wal", "delete") and choice.reason.split(":")[0] in ("local", "denied", "unknown")
        assert choice.mode == "delete" or choice.reason.startswith("local:")


# --- Windows (fabricated Win32 answers; the real calls are covered by the Windows CI job only) -------------------

class FakeWindows(journal._Host):
    def __init__(self, kinds, root="C:\\", resolved=None, error=None):
        self.kinds, self.root, self.resolved, self.error = kinds, root, resolved, error
        self.asked = []

    def realpath(self, path):
        return self.resolved or path

    def volume_root(self, path):
        return self.root

    def drive_type(self, root):
        if self.error:
            raise self.error
        self.asked.append(root)
        return self.kinds.get(root[:2].upper(), 0)


def windows(path, kinds=None, **kwargs):
    host = FakeWindows({"C:": 3, "D:": 3, "E:": 2, "Z:": 4} if kinds is None else kinds, **kwargs)
    return journal.choose_journal(path, _host=host, _system="Windows"), host


@pytest.mark.parametrize("path,root,expected", [
    ("C:\\Users\\u\\index\\meta.sqlite", "C:\\", JournalChoice("wal", "local:windows-fixed")),
    ("D:\\idx\\meta.sqlite", "D:\\", JournalChoice("wal", "local:windows-fixed")),
    ("\\\\?\\C:\\Users\\u\\meta.sqlite", "C:\\", JournalChoice("wal", "local:windows-fixed")),
    ("E:\\idx\\meta.sqlite", "E:\\", JournalChoice("delete", "unknown:windows-removable")),
    ("Z:\\idx\\meta.sqlite", "Z:\\", JournalChoice("delete", "denied:windows-remote")),
])
def test_windows_drive_types(path, root, expected):
    choice, host = windows(path, root=root)
    assert choice == expected and host.asked == [root]


@pytest.mark.parametrize("path,reason", [
    ("\\\\server\\share\\idx\\meta.sqlite", "denied:windows-unc"),
    ("\\\\?\\UNC\\server\\share\\idx\\meta.sqlite", "denied:windows-unc"),
    ("\\\\?\\unc\\server\\share\\idx", "denied:windows-unc"),
    ("\\\\.\\C:\\meta.sqlite", "denied:windows-device"),
    ("\\\\.\\PhysicalDrive0", "denied:windows-device"),
    ("relative\\no-drive", "unknown:windows-no-drive-letter"),
])
def test_windows_unc_device_and_driveless_paths_are_refused_before_any_drive_query(path, reason):
    choice, host = windows(path)
    assert choice == JournalChoice("delete", reason) and host.asked == []


def test_a_mapped_drive_that_resolves_to_unc_is_refused():
    choice, host = windows("Z:\\idx\\meta.sqlite", resolved="\\\\?\\UNC\\nas\\share\\idx\\meta.sqlite")
    assert choice == JournalChoice("delete", "denied:windows-unc") and host.asked == []


def test_a_failing_win32_call_is_unknown():
    choice, _ = windows("C:\\x\\meta.sqlite", error=OSError("no api"))
    assert choice == JournalChoice("delete", "unknown:windows-api-OSError")
    assert windows("C:\\x\\meta.sqlite", {})[0] == JournalChoice("delete", "unknown:windows-unknown-type")


def test_the_real_ctypes_wrappers_pass_the_documented_arguments(monkeypatch):
    """``_Host.volume_root`` / ``drive_type`` against a stand-in for ``ctypes.windll.kernel32``."""
    import ctypes
    calls = []

    class Function:
        def __init__(self, name, result):
            self.name, self.result = name, result

        def __call__(self, *args):
            calls.append((self.name, args[0]))
            if self.name == "GetVolumePathNameW":
                args[1].value = "D:\\"
            return self.result

    kernel32 = types.SimpleNamespace(GetDriveTypeW=Function("GetDriveTypeW", 3),
                                     GetVolumePathNameW=Function("GetVolumePathNameW", 1))
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(kernel32=kernel32), raising=False)
    host = journal._Host()
    assert host.volume_root("D:\\idx\\meta.sqlite") == "D:\\"
    assert host.drive_type("D:\\") == 3
    assert calls == [("GetVolumePathNameW", "D:\\idx\\meta.sqlite"), ("GetDriveTypeW", "D:\\")]
    kernel32.GetVolumePathNameW.result = 0
    assert host.volume_root("D:\\idx") is None
    choice = journal.choose_journal("D:\\idx\\meta.sqlite", _host=types.SimpleNamespace(
        realpath=lambda p: p, volume_root=lambda p: None, drive_type=host.drive_type), _system="Windows")
    assert choice == JournalChoice("wal", "local:windows-fixed")        # falls back to the drive root of the path


@pytest.mark.skipif(sys.platform != "win32", reason="the real Win32 calls; verified by the Windows CI job only")
def test_the_real_windows_calls_accept_a_fixed_drive_and_refuse_a_unc_path(tmp_path):
    choice = journal.choose_journal(tmp_path / "meta.sqlite")
    assert choice in (JournalChoice("wal", "local:windows-fixed"), JournalChoice("delete", "unknown:windows-removable"),
                      JournalChoice("delete", "denied:windows-remote"), JournalChoice("delete", "unknown:windows-ramdisk"))
    assert journal.choose_journal("\\\\localhost\\c$\\x\\meta.sqlite") == JournalChoice("delete", "denied:windows-unc")
    assert journal.choose_journal("\\\\?\\UNC\\localhost\\c$\\x") == JournalChoice("delete", "denied:windows-unc")


# --- the setting ----------------------------------------------------------------------------------------------

def test_the_policy_is_validated():
    assert journal.validate_policy("auto") == "auto" and journal.validate_policy("delete") == "delete"
    for bad in ("wal", "AUTO", "", "off"):
        with pytest.raises(ValueError):
            journal.validate_policy(bad)


@pytest.fixture
def local(monkeypatch):
    """Every directory is a verified local file system (the real answer depends on where the suite runs)."""
    monkeypatch.setattr(journal, "choose_journal_cached", lambda path: JournalChoice("wal", "local:test"))


@pytest.fixture
def denied(monkeypatch):
    monkeypatch.setattr(journal, "choose_journal_cached", lambda path: JournalChoice("delete", "denied:9p"))


def mode_of(path):
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute("PRAGMA journal_mode").fetchone()[0]


# --- apply_journal ----------------------------------------------------------------------------------------------

def test_wal_is_applied_with_synchronous_full(tmp_path, local):
    database = tmp_path / "db.sqlite"
    with closing(sqlite3.connect(database)) as connection:
        state = journal.apply_journal(connection, database, "auto")
        assert (state.mode, state.pending, state.choice.reason) == ("wal", None, "local:test")
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2        # FULL, whatever the build defaults to
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert mode_of(database) == "wal"                                           # persistent: the next open is already WAL


def test_delete_policy_is_the_old_behaviour_and_switches_a_wal_database_back(tmp_path, local):
    database = tmp_path / "db.sqlite"
    with closing(sqlite3.connect(database)) as connection:
        journal.apply_journal(connection, database, "auto")
    with closing(sqlite3.connect(database)) as connection:
        state = journal.apply_journal(connection, database, "delete")
        assert (state.mode, state.pending, state.choice.reason, state.fallback) == ("delete", None, "setting:delete", False)
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    assert mode_of(database) == "delete"


def test_a_denied_file_system_gets_the_rollback_journal_and_reports_the_fallback(tmp_path, denied, caplog):
    database = tmp_path / "db.sqlite"
    with caplog.at_level(logging.INFO, logger="self_directing_mcp.journal"):
        with closing(sqlite3.connect(database)) as connection:
            state = journal.apply_journal(connection, database, "auto")
            journal.apply_journal(connection, database, "auto")
    assert (state.mode, state.fallback, state.choice.reason) == ("delete", True, "denied:9p")
    assert state.report()["fallback"] is True
    assert sum("journal_fallback" in record.getMessage() for record in caplog.records) == 1     # once per process and reason
    assert "denied:9p" in caplog.text


def test_a_database_already_in_wal_on_a_denied_file_system_is_moved_back_when_it_can_be(tmp_path, denied):
    database = tmp_path / "db.sqlite"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    with closing(sqlite3.connect(database)) as connection:
        state = journal.apply_journal(connection, database, "auto")
    assert state.mode == "delete" and state.pending is None and mode_of(database) == "delete"


def test_a_switch_refused_by_another_connection_keeps_the_mode_and_is_retried(tmp_path, local):
    database = tmp_path / "db.sqlite"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("CREATE TABLE t (a)")
    holder = sqlite3.connect(database)
    holder.execute("BEGIN")
    holder.execute("SELECT * FROM t").fetchall()               # a read transaction: the file is share-locked
    try:
        connection = sqlite3.connect(database)
        state = journal.apply_journal(connection, database, "auto")
        assert (state.mode, state.pending) == ("delete", "busy")        # kept, recorded, nothing raised
        assert connection.execute("SELECT COUNT(*) FROM t").fetchone() == (0,)
    finally:
        holder.rollback()
        holder.close()
    journal.reconcile(connection, state)                         # the "later open"
    assert (state.mode, state.pending) == ("wal", None)
    connection.close()


def test_the_mode_is_never_switched_inside_a_transaction(tmp_path, local):
    database = tmp_path / "db.sqlite"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("CREATE TABLE t (a)")
        connection.commit()
        connection.execute("INSERT INTO t VALUES (1)")           # implicit BEGIN
        assert connection.in_transaction
        state = journal.apply_journal(connection, database, "auto")
        assert (state.mode, state.pending) == ("delete", "in-transaction")
        connection.commit()
        journal.reconcile(connection, state)
        assert (state.mode, state.pending) == ("wal", None)


def test_synchronous_full_is_enforced_on_every_open(tmp_path, local):
    database = tmp_path / "db.sqlite"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("PRAGMA synchronous=OFF")
        journal.apply_journal(connection, database, "delete")
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2


def test_an_old_process_is_a_plain_sqlite_connection_and_nothing_breaks(tmp_path, local):
    """An older release opens the file with plain sqlite3 and no knowledge of this module."""
    database = tmp_path / "db.sqlite"
    with closing(sqlite3.connect(database)) as new:
        journal.apply_journal(new, database, "auto")
        new.execute("CREATE TABLE t (a INTEGER PRIMARY KEY, who TEXT)")
        new.execute("INSERT INTO t (who) VALUES ('new')")
        new.commit()
        with closing(sqlite3.connect(database)) as old:             # the second, plain connection
            assert old.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert old.execute("SELECT who FROM t").fetchall() == [("new",)]
            old.execute("INSERT INTO t (who) VALUES ('old')")
            old.commit()
            assert new.execute("SELECT who FROM t ORDER BY a").fetchall() == [("new",), ("old",)]
            new.execute("INSERT INTO t (who) VALUES ('new again')")
            new.commit()
            assert old.execute("SELECT COUNT(*) FROM t").fetchone() == (3,)
            assert old.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    with closing(sqlite3.connect(database)) as later:
        assert later.execute("SELECT COUNT(*) FROM t").fetchone() == (3,)


def test_checkpoint_folds_the_log_into_the_file(tmp_path, local):
    database = tmp_path / "db.sqlite"
    connection = sqlite3.connect(database)
    journal.apply_journal(connection, database, "auto")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute("CREATE TABLE t (a)")
    connection.executemany("INSERT INTO t VALUES (?)", [(n,) for n in range(200)])
    connection.commit()
    wal = Path(str(database) + "-wal")
    assert wal.stat().st_size > 0
    assert journal.checkpoint(connection) is True
    assert wal.stat().st_size == 0
    connection.close()
    assert journal.checkpoint(sqlite3.connect(tmp_path / "plain.sqlite")) is True      # a rollback-journal database: no-op


def test_a_checkpoint_blocked_by_a_reader_reports_false_and_never_raises(tmp_path, local):
    database = tmp_path / "db.sqlite"
    writer = sqlite3.connect(database)
    journal.apply_journal(writer, database, "auto")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("CREATE TABLE t (a)")
    writer.commit()
    reader = sqlite3.connect(database)
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM t").fetchall()
    writer.execute("INSERT INTO t VALUES (1)")
    writer.commit()
    assert journal.checkpoint(writer) is False
    reader.rollback()
    assert journal.checkpoint(writer) is True
    reader.close()
    writer.close()


def test_checkpoint_file_is_cheap_without_a_log(tmp_path):
    assert journal.checkpoint_file(tmp_path / "missing.sqlite") is True
    plain = tmp_path / "plain.sqlite"
    sqlite3.connect(plain).close()
    assert journal.checkpoint_file(plain) is True


# --- SessionStore -----------------------------------------------------------------------------------------------

def chunk(n, session="s-journal"):
    return Chunk(chunk_id=f"c{n}", session_id=session, provider="codex", kind="turn", text=f"event {n}",
                 content_hash=f"h{n}", byte_start=n * 10, byte_end=n * 10 + 1, line_start=1, line_end=1,
                 meta={"sub_index": 0})


def commit(store, n, source):
    store.commit_events([chunk(n)], session_id="s-journal", provider="codex", path=source, offset=n * 10 + 1,
                        digest=f"{n:064d}")


def run_and_die(code: str) -> None:
    """Run ``code`` in another interpreter that ends with ``os._exit``: no ``close``, no ``atexit``, nothing flushed
    by the interpreter, as when a host kills a hook that overran its time limit."""
    program = ("import os\nfrom self_directing_mcp import journal\n"
               "from self_directing_mcp.journal import JournalChoice\n"
               "journal.choose_journal_cached = lambda path: JournalChoice('wal', 'local:test')\n"
               + code + "\nos._exit(0)\n")
    env = {**os.environ, "PYTHONPATH": str(Path(journal.__file__).resolve().parents[1])}
    done = subprocess.run([sys.executable, "-c", program], env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_text("synthetic\n")
    return path


def test_the_session_store_uses_wal_checkpoints_on_close_and_survives_a_copy_of_the_file(tmp_path, local, source):
    database = tmp_path / "meta.sqlite"
    store = SessionStore(database, journal="auto")
    assert (store.journal.mode, store.journal_report()["reason"]) == ("wal", "local:test")
    for n in range(1, 6):
        commit(store, n, source)
    assert Path(str(database) + "-wal").stat().st_size > 0
    store.close()
    assert not Path(str(database) + "-wal").exists() or Path(str(database) + "-wal").stat().st_size == 0
    copy = tmp_path / "copy.sqlite"
    shutil.copy(database, copy)                     # only the main file: nothing is lost because close checkpointed
    with closing(sqlite3.connect(copy)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM session_events").fetchone() == (5,)


def test_a_process_that_dies_without_closing_leaves_a_log_and_the_next_store_recovers_it(tmp_path, local, source):
    database = tmp_path / "meta.sqlite"
    run_and_die(f"""
from pathlib import Path
from self_directing_mcp.index.ingest import SessionStore
from self_directing_mcp.schemas import Chunk
store = SessionStore(Path({str(database)!r}), journal='auto')
for n in range(1, 4):
    store.commit_events([Chunk(chunk_id=f'c{{n}}', session_id='s-journal', provider='codex', kind='turn',
                               text='event', content_hash=f'h{{n}}', byte_start=n * 10, byte_end=n * 10 + 1,
                               meta={{'sub_index': 0}})],
                        session_id='s-journal', provider='codex', path=Path({str(source)!r}), offset=n * 10 + 1,
                        digest=f'{{n:064d}}')
""")
    assert Path(str(database) + "-wal").stat().st_size > 0              # the commits are in the log, not the main file
    with closing(sqlite3.connect(database)) as plain:                    # an older process recovers them
        assert plain.execute("SELECT COUNT(*) FROM session_events").fetchone() == (3,)
        assert plain.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    run_and_die(f"""
from pathlib import Path
from self_directing_mcp.index.ingest import SessionStore
store = SessionStore(Path({str(database)!r}), journal='auto')
assert store.chunk_count('s-journal', 'codex') == 3
""")
    reopened = SessionStore(database, journal="auto")
    assert reopened.chunk_count("s-journal", "codex") == 3
    commit(reopened, 4, source)
    reopened.close()                                                     # a clean close folds the log
    assert not Path(str(database) + "-wal").exists() or Path(str(database) + "-wal").stat().st_size == 0
    with closing(sqlite3.connect(database)) as plain:
        assert plain.execute("SELECT COUNT(*) FROM session_events").fetchone() == (4,)


def test_a_copy_of_the_main_file_alone_loses_logged_commits_and_the_backup_helpers_do_not(tmp_path, local, source):
    database = tmp_path / "index" / "meta.sqlite"
    SessionStore(database, journal="delete").close()          # the tables are in the main file; the commits will not be
    store = SessionStore(database, journal="auto")
    for n in range(1, 4):
        commit(store, n, source)
    assert Path(str(database) + "-wal").stat().st_size > 0              # committed, not yet checkpointed; store still open
    naive = tmp_path / "naive.sqlite"
    shutil.copy(database, naive)                                          # the hazard
    with closing(sqlite3.connect(naive)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM session_events").fetchone() == (0,)
    safe = tmp_path / "safe.sqlite"
    journal.copy_database(database, safe)
    with closing(sqlite3.connect(safe)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM session_events").fetchone() == (3,)
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    assert not Path(str(safe) + "-wal").exists()
    store.close()


def test_copy_index_dir_copies_databases_through_the_backup_api_and_skips_sidecars(tmp_path, local, source):
    index = tmp_path / "index"
    store = SessionStore(index / "meta.sqlite", journal="auto")
    for n in range(1, 4):
        commit(store, n, source)
    (index / "contracts.json").write_text("{}")
    (index / "nested").mkdir()
    (index / "nested" / "note.txt").write_text("kept")
    sqlite3.connect(index / "other.db").close()
    copy = tmp_path / "copy"
    journal.copy_index_dir(index, copy)
    assert sorted(path.name for path in copy.iterdir()) == ["contracts.json", "meta.sqlite", "nested", "other.db"]
    assert (copy / "nested" / "note.txt").read_text() == "kept"
    with closing(sqlite3.connect(copy / "meta.sqlite")) as connection:
        assert connection.execute("SELECT COUNT(*) FROM session_events").fetchone() == (3,)
    store.close()


def test_remove_database_removes_the_sidecars_too(tmp_path):
    database = tmp_path / "x.sqlite"
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(str(database) + suffix).write_text("x")
    journal.remove_database(database)
    assert not any(Path(str(database) + suffix).exists() for suffix in ("", "-wal", "-shm", "-journal"))
    journal.remove_database(database)                                    # idempotent


def test_a_store_opened_while_another_connection_blocks_the_switch_still_serves_requests(tmp_path, local, source):
    database = tmp_path / "meta.sqlite"
    SessionStore(database, journal="delete").close()
    holder = sqlite3.connect(database)
    holder.execute("BEGIN")
    holder.execute("SELECT COUNT(*) FROM session_events").fetchall()
    store = SessionStore(database, journal="auto")
    try:
        assert (store.journal.mode, store.journal.pending) == ("delete", "busy")
        assert store.journal_report()["pending"] == "busy" and store.journal_report()["wanted"] == "wal"
        assert store.chunk_count("s-journal", "codex") == 0                       # reads work
    finally:
        holder.rollback()
        holder.close()
    commit(store, 1, source)                                                       # the request succeeds in DELETE mode
    assert store.retry_journal() is False                                          # rate-limited: too soon
    store.journal.last_attempt -= 3600
    assert store.retry_journal() is True and store.journal.mode == "wal"
    commit(store, 2, source)
    assert store.chunk_count("s-journal", "codex") == 2
    store.close()


def test_retry_journal_leaves_an_open_transaction_alone(tmp_path, local, source):
    database = tmp_path / "meta.sqlite"
    SessionStore(database, journal="delete").close()
    holder = sqlite3.connect(database)
    holder.execute("BEGIN")
    holder.execute("SELECT COUNT(*) FROM session_events").fetchall()
    store = SessionStore(database, journal="auto")
    holder.rollback()
    holder.close()
    store.journal.last_attempt -= 3600
    store._conn.execute("BEGIN")
    store._conn.execute("SELECT 1").fetchall()
    assert store.retry_journal() is False and store.journal.pending == "in-transaction"
    store._conn.rollback()
    store.journal.last_attempt -= 3600
    assert store.retry_journal() is True
    store.close()


def test_the_default_policy_comes_from_the_package(tmp_path, local, monkeypatch, source):
    monkeypatch.setattr(journal, "DEFAULT_POLICY", "auto")
    store = SessionStore(tmp_path / "a.sqlite")
    assert store.journal.mode == "wal"
    store.close()
    monkeypatch.setattr(journal, "DEFAULT_POLICY", "delete")
    other = SessionStore(tmp_path / "b.sqlite")
    assert other.journal.mode == "delete"
    other.close()
    with pytest.raises(ValueError):
        SessionStore(tmp_path / "c.sqlite", journal="wal")


def test_closing_twice_is_harmless(tmp_path, local):
    store = SessionStore(tmp_path / "meta.sqlite", journal="auto")
    store.close()
    store.close()


# --- ReceiptStore -----------------------------------------------------------------------------------------------

CALL = {"name": "run_command", "args": {"CommandLine": "echo hi"}}
SESSION = "55555555-5555-4555-8555-555555555555"


def test_receipts_use_wal_and_an_old_process_can_read_and_write_them(tmp_path, local):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    store = ReceiptStore(tmp_path / "index", journal="auto")
    store.record(SESSION, transcript, 1, CALL, "PreToolUse")
    assert store.journal_report()["mode"] == "wal" and store.journal_report()["pending"] is None
    with closing(sqlite3.connect(store.path)) as old:                             # a plain connection
        assert old.execute("SELECT COUNT(*) FROM receipts").fetchone() == (1,)
        old.execute("INSERT INTO indexed VALUES ('x', 'd')")
        old.commit()
    store.record(SESSION, transcript, 2, CALL, "PreToolUse")
    assert [step for step, _ in store.read_capture(SESSION)[0]] == [1, 2]
    assert store.checkpoint() is True


def test_the_persistent_reader_sees_commits_made_through_the_log(tmp_path, local):
    from self_directing_mcp.index.deferred import DerivedBuffer
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    index = tmp_path / "index"
    resident = ReceiptStore(index, held=DerivedBuffer(), journal="auto")
    other_process = ReceiptStore(index, journal="auto")
    try:
        resident.record(SESSION, transcript, 1, CALL, "PreToolUse")
        first = resident.authority_token(SESSION)
        assert resident.authority_token(SESSION) == first                          # served from the reader's values
        other_process.record(SESSION, transcript, 2, CALL, "PreToolUse")
        assert Path(str(index / "agy-receipts.sqlite") + "-wal").stat().st_size > 0
        second = resident.authority_token(SESSION)
        assert second != first and second[0] == "present"                         # data_version moved with the commit
        assert [step for step, _ in resident.read_capture(SESSION)[0]] == [1, 2]
    finally:
        ReceiptStore.release(index)


def test_receipts_report_their_journal_without_creating_the_file(tmp_path, local):
    report = ReceiptStore(tmp_path / "none", journal="auto").journal_report()
    assert report["mode"] is None and report["wanted"] == "wal" and report["pending"] is None
    assert not (tmp_path / "none").exists()


def test_receipts_in_delete_policy_stay_in_delete(tmp_path, local):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    store = ReceiptStore(tmp_path / "index", journal="delete")
    store.record(SESSION, transcript, 1, CALL, "PreToolUse")
    report = store.journal_report()
    assert (report["mode"], report["reason"], report["fallback"]) == ("delete", "setting:delete", False)


def test_the_hook_state_modules_stay_standard_library_only_and_outside_the_switch():
    """They are not authority records and must import when the rest of the runtime does not (the fallback script copies
    them alone); WAL is deliberately not wired into them."""
    for name in ("agy_hook_state.py", "claude_hook_state.py"):
        text = (Path(journal.__file__).parent / name).read_text(encoding="utf-8")
        assert "self_directing_mcp.journal" not in text and "import journal" not in text
