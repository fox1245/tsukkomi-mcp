"""A parse-ahead spool: the rows of a re-index that did not fit one request wait here (#26, roundtable 8, P2).

A parser upgrade, a changed source or changed AGY receipts make a sync parse a session from byte 0 and replace its stored
events in one transaction. For a session that does not fit the request budget that is all or nothing: every call parses
until its deadline, rolls back and starts again. With a spool each call keeps what it parsed (``append_slice``), the call
that has all of it publishes the rows with ONE ordinary ``replace`` commit (``index.ingest``), and the index database is not
touched until then: an interrupted or crashed re-index leaves the old ids, anchors and vectors exactly as they were.

A spool is a second copy of session text. It lives in ``<index dir>/reindex`` (0700 on POSIX), the file is 0600, nothing but
the sync reads it, nothing sends it anywhere, and it is deleted when its commit succeeded or when anything it was built on
changed. Standard library only (plus the request's cooperative stop).

File: a HEADER frame, then slices. A slice is its row frames (``R``), an index frame (``X``: the ids, flags and tallies of its
rows, which the commit needs before it streams them), a meta frame (``M``: what the next slice's parse needs from this one)
and a TRAILER frame (``T``: ``slice_seq``, ``start``, ``source_offset_end``, ``rows``, ``rows_total``, ``prefix_hash``). A
frame is ``[length][crc32][payload]`` (the crc covers the length too); the first payload byte is its kind. The rows are written
and ``fsync``-ed first and the trailer after them, so a trailer on disk names rows that are on disk. The progress is the last
valid trailer; whatever follows it is a torn tail and is cut off the next time the file is opened.

Single writer: ``fcntl.flock`` (POSIX) or ``msvcrt.locking`` of one byte far beyond the data (Windows; not exercised in this
repository's own runs, see the README), held while the file is open. The per-session flight of ``index.ingest`` already keeps
one process's threads apart. A spool that cannot be validated is discarded, never repaired; the index is never touched.
"""
from __future__ import annotations

import errno
import hashlib
import itertools
import json
import os
import shutil
import stat
import struct
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from self_directing_mcp.request_control import check_request

try:
    import fcntl
except ImportError:                 # Windows
    fcntl = None
try:
    import msvcrt
except ImportError:                 # POSIX
    msvcrt = None

SPOOL_FORMAT = 1
DIRECTORY = "reindex"
SUFFIX = ".spool"
HEAD_BYTES = 64 * 1024              # the head of the source whose hash binds a spool to it
BLOCK_BYTES = 1 << 20               # about this much row text per frame
INDEX_CHUNK = 16384                 # ids (and flags, tallies) per index frame: what a commit decodes at a time
# Free space wanted on the index volume, per source byte, before a spool is started: twice the measured 4.2 spool bytes per
# source byte (the commit then grows the database by about as much again while the spool still exists).
DISK_FACTOR = 2 * 4.5
_LOCK_OFFSET = 1 << 40              # Windows: the byte that is locked, far beyond the data (a locked byte cannot be read)

_FRAME = struct.Struct("<II")       # payload length, crc32 of the length and the payload
_LEN = struct.Struct("<I")
_ROWS = struct.Struct("<I")
HEADER, ROWS, INDEX, META, TRAILER = b"H", b"R", b"X", b"M", b"T"
_SMALL = (HEADER, META, TRAILER)    # read whole when the file is scanned

_BINARY = getattr(os, "O_BINARY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)

# Reasons a spool is thrown away (``ReindexSpool.discard_reason``): a closed set, shown in the ``reindex`` object of a sync.
DISCARD_REASONS = ("format", "session", "target_version", "base_version", "orphan", "index_changed", "source_identity",
                   "source_changed", "source_shrunk", "prefix_changed", "receipts_changed", "corrupt", "unreadable",
                   "disk")


class SpoolError(ValueError):
    """The spool cannot be used (a frame that does not verify, a slice that does not continue the staged range)."""


class SpoolBusy(SpoolError):
    """Another process holds the spool of this session."""


class SpoolUnsafe(SpoolError):
    """The re-index directory or file is not a private plain one (a symlink, another user's)."""


def session_key(provider: str, session_id: str) -> str:
    """A file-name safe key of a session: any id is accepted and no id appears in a file name."""
    return hashlib.sha256(f"{provider}\0{session_id}".encode()).hexdigest()[:32]


def spool_name(provider: str, session_id: str, target_version: int) -> str:
    return f"{provider}-{session_key(provider, session_id)}.{target_version}{SUFFIX}"


def directory_of(index_dir) -> Path:
    return Path(index_dir) / DIRECTORY


def ensure_directory(index_dir) -> Path:
    """The private directory of the spools; a symlink or a directory of another user is refused, never followed."""
    path = directory_of(index_dir)
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
        info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise SpoolUnsafe("the re-index directory is not a plain directory")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise SpoolUnsafe("the re-index directory belongs to another user")
    if os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o077:
        os.chmod(path, 0o700)
    return path


def free_space(path) -> int:
    return shutil.disk_usage(path).free


def has_room(index_dir, source_bytes: int) -> bool:
    """Whether the index volume has the room a spool of ``source_bytes`` of source needs. A volume that cannot be asked
    gives no reason to refuse."""
    try:
        return free_space(index_dir) >= DISK_FACTOR * source_bytes
    except OSError:
        return True


@dataclass(frozen=True)
class SpoolBinding:
    """What a spool is built on. A spool survives a request only while every part of this still holds.

    ``base_version``/``base_stamp``: the parser version and the publication stamp of the cursor the re-index replaces, as they
    were when the spool started; a cursor written by anything else since means the index it was staged against is gone.
    ``receipt_digest`` is the digest of the AGY hook receipts the rows were parsed with (None elsewhere). The source is named
    by device and inode where the platform has them (else by path), its size at the start and a hash of its first 64 KiB.
    The evaluator version is deliberately not part of this: a spool holds parser output only.
    """

    provider: str
    session_id: str
    target_version: int
    base_version: int
    base_stamp: str | None
    hash_kind: str                  # "blocks" / "sha": the form of the prefix hashes
    dev: int
    ino: int
    path: str
    size: int
    head_length: int
    head_hash: str
    receipt_digest: str | None = None

    def header(self) -> dict:
        return {"format": SPOOL_FORMAT, "provider": self.provider, "session_id": self.session_id,
                "target_version": self.target_version, "base_version": self.base_version,
                "base_stamp": self.base_stamp, "hash_kind": self.hash_kind, "receipt_digest": self.receipt_digest,
                "source": {"dev": self.dev, "ino": self.ino, "path": self.path, "size": self.size,
                           "head_length": self.head_length, "head_hash": self.head_hash},
                "created": datetime.now(timezone.utc).isoformat()}


# --- frames -----------------------------------------------------------------------------------------------------------

def _crc(length: int, payload) -> int:
    return zlib.crc32(payload, zlib.crc32(_LEN.pack(length)))


def _write_all(fd: int, data) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _stream_crc(stream, remaining: int, running: int, buffer: bytearray):
    """The crc after ``remaining`` more bytes of ``stream`` (None when the file ends first). One reused buffer."""
    view = memoryview(buffer)
    while remaining:
        count = stream.readinto(view[:min(remaining, len(buffer))])
        if not count:
            return None
        running = zlib.crc32(view[:count], running)
        remaining -= count
    return running


@dataclass
class _Slice:
    rows: int = 0
    row_frames: list = field(default_factory=list)      # (offset, rows) of its R frames
    index_frames: list = field(default_factory=list)    # offsets of its X frames
    meta: dict | None = None


@dataclass
class _Scan:
    """What the frames of a spool file say: the header and the slices that a trailer closed."""

    header: dict | None = None
    reason: str | None = None           # why the file is unusable ("unreadable": no header; "corrupt")
    valid_end: int = 0                  # the end of the last trailer (of the header when there is none)
    slices: int = 0
    rows: int = 0
    staged_through: int = 0
    prefix_hash: str | None = None
    row_frames: list = field(default_factory=list)
    index_frames: list = field(default_factory=list)
    calls: dict = field(default_factory=dict)           # OMP: (call id, tool name) -> earliest byte of a staged call
    deps_steps: dict = field(default_factory=dict)      # AGY: result step -> digest of the receipt its events came from
    deps_first: int | None = None
    receipt_digest: str | None = None


def _json(payload):
    try:
        value = json.loads(payload[1:])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _scan(path, *, verify: bool, meta: bool = True) -> _Scan:
    """Walk the frames of ``path``. ``verify``: check the crc of every frame (else only of header, meta and trailer frames,
    and the walk only follows the lengths). A frame that does not fit the file, or a file that ends inside a header, is a torn
    tail and ends the walk quietly; a frame that fits but does not verify is a torn tail too unless another whole frame
    follows it, which a crash cannot do: then the file is ``corrupt``."""
    scan = _Scan()
    buffer = bytearray(1 << 20)
    pending = _Slice()
    with open(path, "rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        position = 0
        while position < size:
            check_request()
            stream.seek(position)
            head = stream.read(_FRAME.size)
            if len(head) < _FRAME.size:
                break
            length, crc = _FRAME.unpack(head)
            end = position + _FRAME.size + length
            if length < 1 or end > size:
                break
            lead = stream.read(min(length, 5))
            kind = lead[:1]
            if kind in _SMALL:
                payload = lead + stream.read(length - len(lead))
                good = len(payload) == length and _crc(length, payload) == crc
            else:
                payload = None
                good = True
                if verify:
                    good = _stream_crc(stream, length - len(lead), zlib.crc32(lead, zlib.crc32(_LEN.pack(length))),
                                       buffer) == crc
            if not good:
                if _another_frame_follows(stream, end, size):
                    scan.reason = "corrupt"
                break
            if scan.header is None:
                header = _json(payload) if kind == HEADER else None
                if header is None:
                    scan.reason = "unreadable"
                    break
                scan.header, scan.valid_end = header, end
            elif kind == ROWS:
                if len(lead) < 5:
                    scan.reason = "corrupt"
                    break
                (count,) = _ROWS.unpack(lead[1:5])
                pending.rows += count
                pending.row_frames.append((position, count))
            elif kind == INDEX:
                pending.index_frames.append(position)
            elif kind == META:
                pending.meta = _json(payload)
                if pending.meta is None:
                    scan.reason = "corrupt"
                    break
            elif kind == TRAILER:
                if not _close_slice(scan, pending, _json(payload), meta):
                    scan.reason = "corrupt"
                    break
                scan.valid_end, pending = end, _Slice()
            else:
                scan.reason = "corrupt"      # a kind this version never writes
                break
            position = end
    if scan.header is None and scan.reason is None:
        scan.reason = "unreadable"
    return scan


def _another_frame_follows(stream, position: int, size: int) -> bool:
    if position + _FRAME.size > size:
        return False
    stream.seek(position)
    length, _crc_unused = _FRAME.unpack(stream.read(_FRAME.size))
    return 1 <= length and position + _FRAME.size + length <= size


def _close_slice(scan: _Scan, pending: _Slice, trailer: dict | None, meta: bool) -> bool:
    """A trailer closes the slice read since the last one: the slice joins the staged range if the trailer says what the
    frames say (sequence, start, row counts)."""
    if trailer is None:
        return False
    end, digest = trailer.get("source_offset_end"), trailer.get("prefix_hash")
    if (trailer.get("slice_seq") != scan.slices + 1 or trailer.get("start") != scan.staged_through
            or trailer.get("rows") != pending.rows or trailer.get("rows_total") != scan.rows + pending.rows
            or type(end) is not int or end < scan.staged_through or not isinstance(digest, str)):
        return False
    scan.slices += 1
    scan.rows += pending.rows
    scan.staged_through, scan.prefix_hash = end, digest
    scan.row_frames.extend(pending.row_frames)
    scan.index_frames.extend(pending.index_frames)
    if meta and pending.meta is not None:
        for call_id, name, byte in pending.meta.get("calls") or ():
            key = (call_id, name)
            scan.calls[key] = min(scan.calls.get(key, byte), byte)
        deps = pending.meta.get("deps")
        if isinstance(deps, dict):
            scan.deps_steps.update({int(step): digest for step, digest in (deps.get("steps") or {}).items()})
            scan.deps_first = deps.get("first")
        scan.receipt_digest = pending.meta.get("receipt_digest", scan.receipt_digest)
    return True


# --- the lock ---------------------------------------------------------------------------------------------------------

def _lock(fd: int) -> None:
    """Take the writer lock of an open spool, or raise ``SpoolBusy``. Never waits."""
    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                raise SpoolBusy("another process is writing this spool") from None
            raise
    elif msvcrt is not None:
        try:
            os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise SpoolBusy("another process is writing this spool") from None
            # Any other failure means this file system cannot lock: the header/trailer validation still holds.
        finally:
            os.lseek(fd, 0, os.SEEK_SET)


def _unlink(path) -> bool:
    try:
        os.unlink(path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


# A spool this process has verified (or written) is not read in full again while its file is as it left it. The commit
# verifies every frame as it streams them in any case, so this only saves the re-read of everything staged by each request.
_verified: dict[str, tuple] = {}


def _stamp(info) -> tuple:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def forget_verified() -> None:
    """Drop what this process remembers about the spools it has verified (a restart, for the tests)."""
    _verified.clear()


class ReindexSpool:
    """The staged rows of one session's re-index. See the module docstring.

    ``ReindexSpool.open`` finds a spool of this session and target version and validates it against ``binding`` and the source
    of this request; one that fails is discarded (the object then describes an empty spool: ``valid`` false,
    ``discard_reason`` says why). Nothing is created on disk before the first ``append_slice``.
    """

    def __init__(self, directory: Path, binding: SpoolBinding):
        self.binding = binding
        self.directory = directory
        self.path = directory / spool_name(binding.provider, binding.session_id, binding.target_version)
        self._fd: int | None = None
        self._end = 0
        self.valid = False                      # a spool of this session exists and is bound to this request
        self.discard_reason: str | None = None  # why a spool that was found has been thrown away
        self.staged_through = 0                 # the source offset the staged rows reach
        self.rows = 0
        self.slices = 0
        self.prefix_hash: str | None = None     # hash of [0, staged_through) when the last slice was staged
        self.calls: dict = {}
        self.deps_steps: dict = {}
        self.deps_first: int | None = None
        self.receipt_digest: str | None = None
        self._row_frames: list = []
        self._index_frames: list = []

    # -- opening -------------------------------------------------------------------------------------------------------

    @classmethod
    def peek(cls, index_dir, provider: str, session_id: str, target_version: int) -> int | None:
        """The offset a spool of this session has staged through, as a hint only (nothing is validated, locked or removed),
        or None. A sync uses it to ask its capture for the hash state at that offset."""
        path = directory_of(index_dir) / spool_name(provider, session_id, target_version)
        try:
            scan = _scan(path, verify=False, meta=False)
        except OSError:
            return None
        return scan.staged_through or None if scan.header is not None and scan.reason is None else None

    @classmethod
    def discard_session(cls, index_dir, provider: str, session_id: str, *, keep_target: int | None = None) -> int:
        """Remove the spools of one session (those of other target versions only, with ``keep_target``). A spool of
        another session is never touched. Never raises; returns how many files were removed."""
        prefix = f"{provider}-{session_key(provider, session_id)}."
        removed = 0
        try:
            names = os.listdir(directory_of(index_dir))
        except OSError:
            return 0
        for name in names:
            if not (name.startswith(prefix) and name.endswith(SUFFIX)):
                continue
            if keep_target is not None and name == spool_name(provider, session_id, keep_target):
                continue
            path = directory_of(index_dir) / name
            _verified.pop(str(path), None)
            removed += _unlink(path)
        return removed

    @classmethod
    def open(cls, index_dir, binding: SpoolBinding, *, source_length: int, head_hash, cursor_hash) -> "ReindexSpool":
        """The spool of ``binding``'s session for its target version, validated against this request's source.

        ``source_length``: the bytes the request's capture observed. ``head_hash(n)``: the hash of the first ``n`` bytes of the
        source; ``cursor_hash(offset)``: the prefix hash of ``[0, offset)`` in the form ``binding.hash_kind`` names. Raises
        ``SpoolBusy`` when another process holds the spool and ``SpoolUnsafe`` when the directory is not private.
        """
        directory = ensure_directory(index_dir)
        # A spool for another target version (the parser moved on again) cannot be used and is not looked at.
        other = cls.discard_session(index_dir, binding.provider, binding.session_id, keep_target=binding.target_version)
        spool = cls(directory, binding)
        spool._attach(source_length, head_hash, cursor_hash)
        if other and spool.discard_reason is None and not spool.valid:
            spool.discard_reason = "target_version"
        return spool

    def _attach(self, source_length, head_hash, cursor_hash) -> None:
        if os.path.islink(self.path):
            raise SpoolUnsafe("the spool is a symlink")
        try:
            fd = os.open(self.path, os.O_RDWR | _BINARY | _NOFOLLOW | _CLOEXEC)
        except FileNotFoundError:
            return
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise SpoolUnsafe("the spool is a symlink") from None
            raise
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
                raise SpoolUnsafe("the spool is not a regular file of this user")
            _lock(fd)
            if os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o077:
                os.fchmod(fd, 0o600)
            self._fd = fd
            if not self._still_at_path(info):   # removed or replaced while we took the lock: there is nothing to resume
                self._release()
                return
            if info.st_size == 0:               # created, and the process stopped before its header: nothing was staged
                self._release()
                _unlink(self.path)
                return
            reason = self._validate(info, source_length, head_hash, cursor_hash)
        except BaseException:
            self._release()
            raise
        if reason is not None:
            self.discard(reason)

    def _still_at_path(self, info) -> bool:
        try:
            now = os.stat(self.path)
        except OSError:
            return False
        return not info.st_ino or not now.st_ino or (now.st_dev, now.st_ino) == (info.st_dev, info.st_ino)

    def _validate(self, info, source_length, head_hash, cursor_hash) -> str | None:
        """``None`` when the file is a spool of this request, else the reason it is not. Cheap checks first."""
        key = str(self.path)
        full = _verified.get(key) != _stamp(info)
        scan = _scan(self.path, verify=full)
        if scan.reason is not None:
            return scan.reason
        header, binding = scan.header, self.binding
        source = header.get("source")
        if header.get("format") != SPOOL_FORMAT or not isinstance(source, dict) or header.get("hash_kind") != binding.hash_kind:
            return "format"
        if header.get("provider") != binding.provider or header.get("session_id") != binding.session_id:
            return "session"
        if header.get("target_version") != binding.target_version:
            return "target_version"
        base = header.get("base_version")
        if base != binding.base_version:
            # The spool started as a parser upgrade and the index now stores its target (or more): another process finished.
            if type(base) is int and base < binding.target_version <= binding.base_version:
                return "orphan"
            return "base_version"
        if header.get("base_stamp") != binding.base_stamp:
            return "index_changed"
        if source.get("ino") and binding.ino and (source.get("dev"), source.get("ino")) != (binding.dev, binding.ino):
            return "source_identity"
        if not (source.get("ino") and binding.ino) and source.get("path") != binding.path:
            return "source_identity"
        if source_length < scan.staged_through:
            return "source_shrunk"
        head = source.get("head_length")
        if type(head) is not int or head < 0 or head > source_length or head_hash(head) != source.get("head_hash"):
            return "source_changed"
        if scan.staged_through and cursor_hash(scan.staged_through) != scan.prefix_hash:
            return "prefix_changed"
        if scan.valid_end < info.st_size:       # a torn tail: cut it so the next slice is appended after the last trailer
            os.ftruncate(self._fd, scan.valid_end)
        self._take(scan)
        self._remember()
        return None

    def _take(self, scan: _Scan) -> None:
        self._end, self.valid = scan.valid_end, True
        self.staged_through, self.rows, self.slices = scan.staged_through, scan.rows, scan.slices
        self.prefix_hash, self.calls = scan.prefix_hash, scan.calls
        self.deps_steps, self.deps_first = scan.deps_steps, scan.deps_first
        # The receipts the staged rows were parsed with: those of the last slice, or (no slice yet) of the start.
        self.receipt_digest = scan.receipt_digest if scan.slices else scan.header.get("receipt_digest")
        self._row_frames, self._index_frames = scan.row_frames, scan.index_frames

    def _remember(self) -> None:
        try:
            _verified[str(self.path)] = _stamp(os.fstat(self._fd))
        except OSError:
            _verified.pop(str(self.path), None)

    # -- writing -------------------------------------------------------------------------------------------------------

    def _ensure_file(self) -> None:
        if self._fd is not None:
            return
        if os.path.islink(self.path):
            raise SpoolUnsafe("the spool is a symlink")
        try:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL | _BINARY | _NOFOLLOW | _CLOEXEC, 0o600)
        except FileExistsError:
            raise SpoolBusy("another process created this spool") from None
        try:
            _lock(fd)
            self._fd = fd
            self._end = 0
            self._write_frame(HEADER + json.dumps(self.binding.header()).encode())
            self._sync()
            self._sync_directory()
        except BaseException:
            self._release()
            _unlink(self.path)
            raise
        self.valid, self.receipt_digest = True, self.binding.receipt_digest
        self._remember()

    def _write_frame(self, payload: bytes) -> int:
        length = len(payload)
        offset = self._end
        _write_all(self._fd, _FRAME.pack(length, _crc(length, payload)))
        _write_all(self._fd, payload)
        self._end += _FRAME.size + length
        return offset

    def _sync(self) -> None:
        os.fsync(self._fd)

    def _sync_directory(self) -> None:
        if os.name == "nt":
            return
        try:
            fd = os.open(self.directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def _write_rows(self, rows) -> tuple[list, int, list]:
        """Write the row frames of a slice; returns ``(frames, count, ids)``."""
        frames, count = [], 0
        if hasattr(rows, "blocks"):                 # a RowSpool: its text is already the lines of the file
            source, ids = rows.blocks(BLOCK_BYTES), list(rows.ids)
        else:
            source, ids = self._encode(rows)
        for data, number in source:
            check_request()
            frames.append((self._write_frame(ROWS + _ROWS.pack(number) + data), number))
            count += number
        if hasattr(rows, "blocks") and rows.count != count:
            raise SpoolError("the rows of a slice do not match their count")
        return frames, count, ids

    @staticmethod
    def _encode(rows):
        ids = []

        def lines():
            pending, size = [], 0
            for row in rows:
                ids.append(row[0])
                line = (json.dumps(list(row)) + "\n").encode()
                pending.append(line)
                size += len(line)
                if size >= BLOCK_BYTES:
                    yield b"".join(pending), len(pending)
                    pending, size = [], 0
            if pending:
                yield b"".join(pending), len(pending)

        return lines(), ids

    @staticmethod
    def _index_parts(ids, flags, tallies):
        """The ids, flags and tallies of a slice in parts of ``INDEX_CHUNK``: no part is a document of the size of the slice."""
        ids, flags, tallies = iter(ids), iter(flags.items()), iter(tallies.items())
        while True:
            part = {"ids": list(itertools.islice(ids, INDEX_CHUNK)),
                    "flags": {cid: [bool(a), bool(b)] for cid, (a, b) in itertools.islice(flags, INDEX_CHUNK)},
                    "tallies": {cid: [[key, number] for key, number in pairs]
                                for cid, pairs in itertools.islice(tallies, INDEX_CHUNK)}}
            if not (part["ids"] or part["flags"] or part["tallies"]):
                return
            yield part

    def _write_trailer(self, trailer: dict) -> None:
        self._write_frame(TRAILER + json.dumps(trailer).encode())

    def append_slice(self, rows, end_offset: int, prefix_hash: str, *, start_offset: int | None = None, flags=None,
                     tallies=None, calls=None, deps=None, receipt_digest: str | None = None) -> bool:
        """Stage the rows a parse produced from ``start_offset`` to ``end_offset`` and make the progress durable.

        ``rows``: a ``RowSpool`` (its text is copied as it is) or an iterable of row tuples
        ``(chunk_id, provider, session_id, byte_start, sub_index, data)``, then ``flags`` and ``tallies`` give what a
        ``RowSpool`` keeps beside its rows. ``prefix_hash``: the hash of ``[0, end_offset)`` of the source. ``calls`` (OMP,
        ``{(call id, tool name): earliest byte}``) and ``deps`` (AGY, the dependencies of the events on the receipts) are what the
        next slice's parse needs from this one. A slice that ends where the staged range already ends is a re-run and writes
        nothing (returns False): a slice is never staged twice. One that does not start at the end of the staged range is an
        error. The rows reach the disk before the trailer that names them.
        """
        if end_offset <= self.staged_through:
            return False
        if start_offset is not None and start_offset != self.staged_through:
            raise SpoolError("a slice must continue where the staged range ends")
        self._ensure_file()
        flags = rows.flags if flags is None and hasattr(rows, "flags") else (flags or {})
        tallies = rows.tallies if tallies is None and hasattr(rows, "tallies") else (tallies or {})
        calls = getattr(rows, "calls", None) if calls is None else calls
        begin = self._end
        os.lseek(self._fd, begin, os.SEEK_SET)
        try:
            frames, count, ids = self._write_rows(rows)
            index_at = []
            for part in self._index_parts(ids, flags, tallies):
                check_request()
                index_at.append(self._write_frame(INDEX + json.dumps(part).encode()))
            meta = {"calls": [[call_id, name, byte] for (call_id, name), byte in (calls or {}).items()],
                    "deps": deps, "receipt_digest": receipt_digest}
            self._write_frame(META + json.dumps(meta).encode())
            self._sync()                    # the rows are on disk before a trailer says they are
            self._write_trailer({"slice_seq": self.slices + 1, "start": self.staged_through,
                                 "source_offset_end": end_offset, "rows": count, "rows_total": self.rows + count,
                                 "prefix_hash": prefix_hash})
            self._sync()
        except BaseException:
            self._truncate(begin)
            raise
        self.slices += 1
        self.rows += count
        self.staged_through, self.prefix_hash = end_offset, prefix_hash
        self._row_frames.extend(frames)
        self._index_frames.extend(index_at)
        for (call_id, name), byte in (calls or {}).items():
            self.calls[(call_id, name)] = min(self.calls.get((call_id, name), byte), byte)
        if isinstance(deps, dict):
            self.deps_steps.update({int(step): digest for step, digest in (deps.get("steps") or {}).items()})
            self.deps_first = deps.get("first")
        if receipt_digest is not None:
            self.receipt_digest = receipt_digest
        self._remember()
        return True

    def _truncate(self, length: int) -> None:
        """Drop what an unfinished slice wrote (best effort: the next open cuts a torn tail anyway)."""
        try:
            os.ftruncate(self._fd, length)
        except OSError:
            pass
        self._end = length

    # -- reading and ending --------------------------------------------------------------------------------------------

    def as_row_source(self) -> "StagedRows":
        """The staged rows as the ``prepared_rows`` of a commit (the interface of ``index.ingest.RowSpool``)."""
        if not self.valid or self._fd is None:
            raise SpoolError("there is no staged spool")
        return StagedRows(self.path, list(self._row_frames), list(self._index_frames), self.rows)

    def discard(self, reason: str) -> None:
        """Throw the spool away (the index is never touched). The object then describes an empty spool."""
        self.discard_reason = reason
        self._release()
        _verified.pop(str(self.path), None)
        _unlink(self.path)
        self.valid, self.staged_through, self.rows, self.slices, self.prefix_hash = False, 0, 0, 0, None
        self.calls, self.deps_steps, self.deps_first, self.receipt_digest = {}, {}, None, None
        self._row_frames, self._index_frames, self._end = [], [], 0

    def finish(self) -> None:
        """The re-index is published: the spool has done its job and is deleted."""
        self._release()
        _verified.pop(str(self.path), None)
        _unlink(self.path)
        self.valid = False

    def abort(self) -> None:
        """A request that stopped inside a slice: cut what it wrote after the last trailer and let go of the lock."""
        if self._fd is not None:
            self._truncate(self._end)
        self._release()

    def close(self) -> None:
        self._release()

    def _release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is not None:
            try:
                os.close(fd)        # releases the lock
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        # The lock is the file descriptor: a spool nobody closed must not keep its session locked for this process's life.
        try:
            self._release()
        except Exception:
            pass

    @property
    def size(self) -> int:
        """Bytes of the spool file that belong to the staged range."""
        return self._end


class StagedRows:
    """The rows of a spool, read back for a commit: ``count``, ``ids``, ``flags``, ``tallies`` and ``batches()``, the
    interface of ``index.ingest.RowSpool``. The rows are streamed (a block at a time), never loaded whole; every frame is
    verified as it is read, and a frame that does not verify raises ``SpoolError`` (the commit then rolls back)."""

    def __init__(self, path, row_frames, index_frames, count: int):
        self.path, self._row_frames, self.count = path, row_frames, count
        self.ids, self.flags, self.tallies = set(), {}, {}
        with open(path, "rb") as stream:
            for offset in index_frames:
                check_request()
                index = _json(self._frame(stream, offset))
                if index is None:
                    raise SpoolError("an index frame of the spool is unreadable")
                self.ids.update(index["ids"])
                self.flags.update({cid: (a, b) for cid, (a, b) in index["flags"].items()})
                self.tallies.update({cid: [(key, number) for key, number in pairs] for cid, pairs in index["tallies"].items()})

    @staticmethod
    def _frame(stream, offset: int) -> bytes:
        stream.seek(offset)
        head = stream.read(_FRAME.size)
        if len(head) < _FRAME.size:
            raise SpoolError("the spool ends inside a frame")
        length, crc = _FRAME.unpack(head)
        payload = stream.read(length)
        if len(payload) != length or _crc(length, payload) != crc:
            raise SpoolError("a frame of the spool does not verify")
        return payload

    def __len__(self):
        return self.count

    def batches(self, size: int = 256):
        batch = []
        with open(self.path, "rb") as stream:
            for offset, number in self._row_frames:
                check_request()
                payload = self._frame(stream, offset)
                lines = payload[5:].split(b"\n")
                if len(lines) != number + 1 or lines[-1]:
                    raise SpoolError("a row frame of the spool does not hold the rows it counts")
                for line in lines[:-1]:
                    batch.append(tuple(json.loads(line)))
                    if len(batch) == size:
                        yield batch
                        batch = []
        if batch:
            yield batch

    def close(self) -> None:
        """Nothing to release: each ``batches()`` opens and closes its own handle."""
