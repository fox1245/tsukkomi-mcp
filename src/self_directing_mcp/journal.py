"""SQLite journal mode for the authority databases: when write-ahead logging is allowed, and the WAL-aware helpers.

Standard library only (the hook path must stay light; see ``tests/test_hook_import_budget.py``).

Two databases hold authority records and are the only ones this module configures: the session index
(``meta.sqlite``: events, cursors, receipts digests) and the AGY receipts database (``agy-receipts.sqlite``). Both use
``synchronous=FULL`` in every mode. ``synchronous=NORMAL`` is never set here: in WAL mode it can lose the last
acknowledged commits at power loss (``scripts/bench_commit.py`` measures it for reference only). The contracts file
has its own publication protocol and the hook-state databases are not authority records; neither is touched.

The default policy is ``delete`` (the rollback journal, as before): ``scripts/bench_commit.py`` shows write-ahead
logging cutting the append transaction of a resident engine by two thirds, but a command hook, a new process that opens,
commits once and closes, does not get faster end to end (README, "SQLite journal mode, index files and backups").
``auto`` is the opt-in. Under ``auto`` write-ahead logging is used only on a file system that is positively identified
as local:

* ``choose_journal(path)`` returns ``JournalChoice(mode, reason)``. ``mode`` is ``"wal"`` or ``"delete"``. ``reason``
  names the evidence: ``local:ext4``, ``denied:9p``, ``unknown:<detail>``, ``setting:delete``.
* **Unknown means DELETE.** The fallback is the old behaviour; it is not a proof that DELETE is safe on that file
  system, only that WAL (shared memory in a ``-shm`` file) is not known to work there.
* Linux: ``os.statvfs`` plus ``/proc/self/mountinfo`` for the mount that holds the path. Local types are ext2/3/4,
  xfs, btrfs, tmpfs, f2fs, zfs, and overlay only when its upper directory resolves to a mount of one of those.
  ``9p`` (also WSL's ``drvfs``), ``nfs*``, ``cifs``, ``smb*``, ``fuse*`` and ``virtiofs`` are denied; anything else,
  and anything that cannot be determined, is unknown. FUSE is never treated as local, whatever it mounts: a local
  FUSE file system cannot be told from a network one with the standard library.
* Windows: the volume holding the (resolved) path must be a fixed drive (``GetDriveTypeW`` is ``DRIVE_FIXED``);
  UNC paths, ``\\\\?\\UNC\\``, device paths and mapped network drives are refused. This branch is verified by the
  Windows CI job only; it could not be measured by the benchmark.
* Any other platform is unknown.

The mode changes only when a connection is opened, never inside a transaction (``apply_journal``). A switch that is
refused (``SQLITE_BUSY``: another connection, possibly an older process, holds the file) keeps the existing mode,
is recorded in ``JournalState.pending`` and is tried again at the next open; it never fails a request. Older
releases of this package read and write a WAL database normally (any SQLite since 3.7 does).

A copied ``.sqlite`` file without its ``-wal`` loses recent commits. ``copy_database`` / ``copy_index_dir`` use
SQLite's online backup API, which sees the committed WAL content; ``checkpoint`` folds the WAL into the main file.
"""
from __future__ import annotations

import ntpath
import os
import re
import sqlite3
import sys
import threading
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from self_directing_mcp import settings_core

# ``auto``: WAL on a verified local file system, DELETE elsewhere. ``delete``: DELETE + FULL everywhere (the pre-WAL
# behaviour). The values and the default live in ``settings_core`` with every other setting, which the command hooks
# read without importing this module (and ``sqlite3`` with it).
POLICIES = settings_core.SQLITE_JOURNAL_MODES
DEFAULT_POLICY = settings_core.DEFAULT_SQLITE_JOURNAL
SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
DATABASE_SUFFIXES = (".sqlite", ".db")

LOCAL_FILESYSTEMS = frozenset({"ext2", "ext3", "ext4", "xfs", "btrfs", "tmpfs", "f2fs", "zfs"})
_DENIED_EXACT = frozenset({"9p", "drvfs", "virtiofs"})
_DENIED_PREFIXES = ("nfs", "smb", "cifs", "fuse")
_SWITCH_BUSY_MS = 50          # how long a mode switch may wait for another connection before it is deferred
_CHECKPOINT_BUSY_MS = 250     # how long a checkpoint may wait for readers
_CACHE_SECONDS = 30.0         # a mount table does not change under a running process in any way that matters here
_RETRY_SECONDS = 30.0         # a resident connection retries a refused switch no more often than this


@dataclass(frozen=True)
class JournalChoice:
    mode: str       # "wal" or "delete"
    reason: str     # "local:ext4", "denied:9p", "unknown:<detail>", "setting:delete"

    @property
    def fallback(self) -> bool:
        """DELETE because the file system is denied or unknown (not because the setting asked for it)."""
        return self.mode == "delete" and self.reason.startswith(("denied:", "unknown:"))


def validate_policy(value: str) -> str:
    if value not in POLICIES:
        raise ValueError(f"unknown SQLite journal policy: {value!r} (use {'|'.join(POLICIES)})")
    return value


# --- Linux ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Mount:
    dev: tuple[int, int]
    root: str
    point: str
    fstype: str
    source: str
    options: str          # the super-block options (``rw,lowerdir=...``)


def _unescape(text: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), text)


def parse_mountinfo(text: str) -> list[Mount]:
    """The mounts of a ``/proc/self/mountinfo`` text, in file order; lines that do not parse are skipped."""
    mounts = []
    for line in text.splitlines():
        parts = line.split(" ")
        try:
            separator = parts.index("-", 6)
            major, minor = parts[2].split(":")
            mounts.append(Mount(dev=(int(major), int(minor)), root=_unescape(parts[3]), point=_unescape(parts[4]),
                                fstype=parts[separator + 1].lower(), source=_unescape(parts[separator + 2]),
                                options=parts[separator + 3] if len(parts) > separator + 3 else ""))
        except (ValueError, IndexError):
            continue
    return mounts


def _contains(point: str, path: str) -> bool:
    return point == "/" or path == point or path.startswith(point.rstrip("/") + "/")


def _mount_for(mounts: list[Mount], path: str, dev: tuple[int, int] | None) -> Mount | None:
    """The mount that holds ``path``: the longest mount point; among stacked mounts at one point the topmost (last in
    the file), preferring one whose device is the path's device."""
    containing = [mount for mount in mounts if _contains(mount.point, path)]
    if not containing:
        return None
    longest = max(len(mount.point) for mount in containing)
    deepest = [mount for mount in containing if len(mount.point) == longest]
    matching = [mount for mount in deepest if dev is not None and mount.dev == dev]
    return (matching or deepest)[-1]


def _classify(fstype: str) -> str:
    """``local``, ``denied`` or ``unknown`` for a file system type name."""
    if fstype in LOCAL_FILESYSTEMS:
        return "local"
    if fstype in _DENIED_EXACT or fstype.startswith(_DENIED_PREFIXES):
        return "denied"
    return "unknown"


class _Host:
    """The operating-system facts ``choose_journal`` reads; tests substitute fabricated ones."""

    def realpath(self, path: str) -> str:
        return os.path.realpath(path)

    def isdir(self, path: str) -> bool:
        return os.path.isdir(path)

    def statvfs(self, path: str):
        return os.statvfs(path)

    def device(self, path: str) -> tuple[int, int] | None:
        try:
            value = os.stat(path).st_dev
        except OSError:
            return None
        return os.major(value), os.minor(value)

    def mountinfo(self) -> str:
        with open("/proc/self/mountinfo", encoding="utf-8", errors="replace") as stream:
            return stream.read()

    # Windows (only reached when ``system`` is "Windows")
    def volume_root(self, path: str) -> str | None:
        import ctypes
        from ctypes import wintypes
        buffer = ctypes.create_unicode_buffer(32768)
        function = ctypes.windll.kernel32.GetVolumePathNameW
        function.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        function.restype = wintypes.BOOL
        return buffer.value if function(path, buffer, len(buffer)) else None

    def drive_type(self, root: str) -> int:
        import ctypes
        from ctypes import wintypes
        function = ctypes.windll.kernel32.GetDriveTypeW
        function.argtypes = [wintypes.LPCWSTR]
        function.restype = wintypes.UINT
        return function(root)


_HOST = _Host()
DRIVE_FIXED = 3
_DRIVE_NAMES = {0: "unknown-type", 1: "no-root", 2: "removable", 3: "fixed", 4: "remote", 5: "cdrom", 6: "ramdisk"}


def _existing_directory(path: str, host) -> str:
    """The nearest existing directory at or above ``path`` (the database file itself may not exist yet)."""
    current = host.realpath(path)
    while not host.isdir(current):
        parent = os.path.dirname(current)
        if not parent or parent == current:
            break
        current = parent
    return current


def _choose_linux(path: str, host) -> JournalChoice:
    directory = _existing_directory(path, host)
    try:
        stat = host.statvfs(directory)
    except OSError as exc:
        return JournalChoice("delete", f"unknown:statvfs-{type(exc).__name__}")
    if getattr(stat, "f_flag", 0) & getattr(os, "ST_RDONLY", 1):
        return JournalChoice("delete", "unknown:read-only")
    try:
        mounts = parse_mountinfo(host.mountinfo())
    except OSError as exc:
        return JournalChoice("delete", f"unknown:mountinfo-{type(exc).__name__}")
    if not mounts:
        return JournalChoice("delete", "unknown:mountinfo-empty")
    mount = _mount_for(mounts, directory, host.device(directory))
    if mount is None:
        return JournalChoice("delete", "unknown:no-mount")
    if mount.fstype == "overlay":
        return _choose_overlay(mount, mounts, host)
    kind = _classify(mount.fstype)
    return JournalChoice("wal" if kind == "local" else "delete", f"{kind}:{mount.fstype}")


def _choose_overlay(mount: Mount, mounts: list[Mount], host) -> JournalChoice:
    """An overlay is as local as the file system under its upper directory, if that can be shown."""
    upper = next((_unescape(option[len("upperdir="):]) for option in mount.options.split(",")
                  if option.startswith("upperdir=")), None)
    if not upper:
        return JournalChoice("delete", "unknown:overlay-without-upperdir")
    dev = host.device(upper)
    if dev is None:
        return JournalChoice("delete", "unknown:overlay-upper-unverifiable")
    under = _mount_for(mounts, upper, dev)
    if under is None or under.dev != dev or under.fstype == "overlay":
        return JournalChoice("delete", "unknown:overlay-upper-unverifiable")
    kind = _classify(under.fstype)
    if kind == "local":
        return JournalChoice("wal", f"local:overlay(upper={under.fstype})")
    return JournalChoice("delete", f"{kind}:overlay(upper={under.fstype})")


# --- Windows ----------------------------------------------------------------------------------------------------

def _choose_windows(path: str, host) -> JournalChoice:
    try:
        resolved = host.realpath(path)
    except OSError as exc:
        return JournalChoice("delete", f"unknown:windows-resolve-{type(exc).__name__}")
    # Extended-length prefixes name the same paths; UNC and device forms are refused whichever way they are spelled.
    upper = resolved.upper()
    if upper.startswith("\\\\?\\UNC\\") or upper.startswith("\\\\?\\GLOBALROOT"):
        return JournalChoice("delete", "denied:windows-unc")
    if upper.startswith("\\\\.\\") or upper.startswith("\\??\\"):
        return JournalChoice("delete", "denied:windows-device")
    if upper.startswith("\\\\?\\"):
        resolved = resolved[4:]
    drive, _rest = ntpath.splitdrive(resolved)
    if resolved.startswith("\\\\") or drive.startswith("\\\\"):
        return JournalChoice("delete", "denied:windows-unc")
    if len(drive) != 2 or drive[1] != ":":
        return JournalChoice("delete", "unknown:windows-no-drive-letter")
    try:
        root = host.volume_root(resolved) or drive + "\\"
        kind = host.drive_type(root)
    except (OSError, AttributeError) as exc:
        return JournalChoice("delete", f"unknown:windows-api-{type(exc).__name__}")
    if kind == DRIVE_FIXED:
        return JournalChoice("wal", "local:windows-fixed")
    name = _DRIVE_NAMES.get(kind, str(kind))
    return JournalChoice("delete", ("denied:windows-" if kind == 4 else "unknown:windows-") + name)


def choose_journal(path, *, _host=None, _system: str | None = None) -> JournalChoice:
    """Whether the database at ``path`` may use write-ahead logging; see the module docstring. Never raises.

    ``_host`` and ``_system`` exist so tests can supply a fabricated mount table or drive API."""
    system = _system or ("Windows" if sys.platform == "win32" else "Linux" if sys.platform.startswith("linux")
                         else sys.platform)
    host = _host or _HOST
    try:
        text = os.fspath(path)
        if system == "Windows":
            return _choose_windows(text, host)
        if system == "Linux":
            return _choose_linux(text, host)
        return JournalChoice("delete", f"unknown:platform-{system}")
    except Exception as exc:          # a probe must never decide a request
        return JournalChoice("delete", f"unknown:probe-{type(exc).__name__}")


_CACHE: dict[str, tuple[float, JournalChoice]] = {}
_CACHE_LOCK = threading.Lock()


def choose_journal_cached(path) -> JournalChoice:
    """``choose_journal`` for the directory of ``path``, remembered for a short time (a connection opens often)."""
    key = os.path.dirname(os.path.abspath(os.fspath(path)))
    now = time.monotonic()
    with _CACHE_LOCK:
        held = _CACHE.get(key)
        if held is not None and now - held[0] < _CACHE_SECONDS:
            return held[1]
    choice = choose_journal(key)
    with _CACHE_LOCK:
        _CACHE[key] = (now, choice)
    return choice


# --- applying a choice to a connection --------------------------------------------------------------------------

@dataclass
class JournalState:
    """What one connection knows about its database's journal: the policy, the choice, the mode it is in."""
    policy: str
    choice: JournalChoice
    path: str = ""
    mode: str | None = None          # the mode the database is actually in after the last attempt
    pending: str | None = None       # why the wanted mode is not reached: "busy", "in-transaction", "error:..."
    attempts: int = 0
    last_attempt: float = field(default=0.0, repr=False)

    @property
    def fallback(self) -> bool:
        return self.policy == "auto" and self.choice.fallback

    def report(self) -> dict[str, Any]:
        return {"policy": self.policy, "mode": self.mode, "wanted": self.choice.mode, "reason": self.choice.reason,
                "pending": self.pending, "synchronous": "FULL", "fallback": self.fallback}

    def retry_due(self) -> bool:
        return self.pending is not None and time.monotonic() - self.last_attempt >= _RETRY_SECONDS


_LOGGED: set[tuple[str, str]] = set()


def _log_fallback(state: JournalState) -> None:
    key = (os.path.dirname(state.path), state.choice.reason)
    if not state.fallback or key in _LOGGED:
        return
    _LOGGED.add(key)
    import logging
    logging.getLogger(__name__).info(
        "journal_fallback: %s uses DELETE journaling (%s); write-ahead logging needs a verified local file system",
        os.path.basename(state.path), state.choice.reason)


def _current_mode(conn: sqlite3.Connection) -> str:
    return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()


def _switch(conn: sqlite3.Connection, wanted: str) -> str | None:
    """Ask for ``wanted``; None on success, else why not. Waits only briefly for other connections."""
    timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    conn.execute(f"PRAGMA busy_timeout={_SWITCH_BUSY_MS}")
    try:
        row = conn.execute(f"PRAGMA journal_mode={wanted.upper()}").fetchone()
    except sqlite3.OperationalError as exc:
        return "busy" if "lock" in str(exc).lower() or "busy" in str(exc).lower() else "error:OperationalError"
    finally:
        try:
            conn.execute(f"PRAGMA busy_timeout={int(timeout)}")
        except sqlite3.Error:
            pass
    return None if row and str(row[0]).lower() == wanted else "busy"


def reconcile(conn: sqlite3.Connection, state: JournalState) -> JournalState:
    """Bring ``conn``'s database to ``state.choice.mode`` if it is not there. Never raises; never switches inside a
    transaction. ``synchronous=FULL`` is set on every call: a WAL build's compiled-in default must not apply."""
    state.attempts += 1
    state.last_attempt = time.monotonic()
    try:
        current = _current_mode(conn)
        wanted = state.choice.mode
        state.pending = None
        if current not in ("memory", wanted) and (wanted == "wal" or current == "wal"):
            if conn.in_transaction:
                state.pending = "in-transaction"
            else:
                state.pending = _switch(conn, wanted)
                current = _current_mode(conn)
        state.mode = current
        if not conn.in_transaction:                  # SQLite refuses to change the sync level inside one
            conn.execute("PRAGMA synchronous=FULL")
    except sqlite3.Error as exc:
        state.pending = "error:" + type(exc).__name__
    return state


def apply_journal(conn: sqlite3.Connection, path, policy: str | None = None) -> JournalState:
    """Choose and apply the journal mode of the database ``conn`` was just opened on (call it before any transaction)."""
    policy = validate_policy(DEFAULT_POLICY if policy is None else policy)
    choice = JournalChoice("delete", "setting:delete") if policy == "delete" else choose_journal_cached(path)
    state = JournalState(policy=policy, choice=choice, path=os.fspath(path))
    reconcile(conn, state)
    _log_fallback(state)
    return state


def inspect_journal(path, policy: str | None = None) -> dict[str, Any]:
    """The journal report of a database file without keeping a connection: the actual mode (None when the file does
    not exist), what the policy and file system call for, and whether they differ."""
    policy = validate_policy(DEFAULT_POLICY if policy is None else policy)
    choice = JournalChoice("delete", "setting:delete") if policy == "delete" else choose_journal_cached(path)
    state = JournalState(policy=policy, choice=choice, path=os.fspath(path))
    try:
        if os.path.exists(path):
            with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
                state.mode = _current_mode(conn)
    except (sqlite3.Error, OSError):
        state.mode = None
    if state.mode is not None and state.mode != choice.mode and state.mode != "memory":
        state.pending = "differs"
    return state.report()


def checkpoint(conn: sqlite3.Connection, mode: str = "TRUNCATE") -> bool:
    """Fold the WAL into the main file (a no-op outside WAL mode). True when it completed; False is not an error:
    a reader kept it from finishing, and the next connection or checkpoint will."""
    try:
        if conn.in_transaction or _current_mode(conn) != "wal":
            return True
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        conn.execute(f"PRAGMA busy_timeout={_CHECKPOINT_BUSY_MS}")
        try:
            busy = conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()[0]
        finally:
            conn.execute(f"PRAGMA busy_timeout={int(timeout)}")
        return not busy
    except sqlite3.Error:
        return False


def checkpoint_file(path) -> bool:
    """``checkpoint`` for a database nobody here holds open; cheap when no ``-wal`` file exists."""
    if not os.path.exists(os.fspath(path) + "-wal"):
        return True
    try:
        with closing(sqlite3.connect(os.fspath(path))) as conn:
            return checkpoint(conn)
    except sqlite3.Error:
        return False


# --- copying and removing databases -----------------------------------------------------------------------------

def sidecars(path) -> list[Path]:
    """The ``-wal``, ``-shm`` and ``-journal`` companions of a database file (existing or not)."""
    return [Path(os.fspath(path) + suffix) for suffix in SIDECAR_SUFFIXES]


def copy_database(source, destination) -> None:
    """A consistent copy of a SQLite database that includes everything committed, whatever its journal mode.

    Copying the ``.sqlite`` file alone loses the commits that still sit in ``-wal``; this uses the online backup API,
    which reads through the WAL. The destination is replaced."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    remove_database(destination)
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=rw", uri=True)) as src, \
            closing(sqlite3.connect(destination)) as dst:
        src.backup(dst)
        dst.execute("PRAGMA journal_mode=DELETE")   # a standalone copy: no sidecars of its own


def remove_database(path) -> None:
    """Delete a database file and its sidecars (a leftover ``-wal`` next to a new file would be replayed into it)."""
    for target in (Path(path), *sidecars(path)):
        try:
            target.unlink()
        except FileNotFoundError:
            pass


def copy_index_dir(source, destination, *, copy_function: Callable[..., Any] | None = None) -> None:
    """``copytree`` for an index directory: databases through ``copy_database`` (never their sidecars), every other
    file as is. For a directory no process is writing, or one that is being written."""
    source, destination = Path(source), Path(destination)
    if copy_function is None:
        import shutil
        copy_function = shutil.copy2
    destination.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        if entry.name.endswith(SIDECAR_SUFFIXES):
            continue
        target = destination / entry.name
        if entry.is_dir():
            copy_index_dir(entry, target, copy_function=copy_function)
        elif entry.name.endswith(DATABASE_SUFFIXES):
            copy_database(entry, target)
        else:
            copy_function(entry, target)
