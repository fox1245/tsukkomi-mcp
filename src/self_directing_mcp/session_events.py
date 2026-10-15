"""Shared lossless JSONL and event helpers. Raw evidence stays local."""
from __future__ import annotations

import bisect
import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
import os
import json
import re
from pathlib import Path
from typing import Any, Iterator

from self_directing_mcp import source_blocks
from self_directing_mcp.compressed_source import DEFAULT_MAX_DECOMPRESSED_BYTES, is_compressed, open_source
from self_directing_mcp.schemas import Chunk
from self_directing_mcp.request_control import check_request
from self_directing_mcp.slice_budget import current_slice
from self_directing_mcp.source_blocks import BlockScan, FastFallback
from self_directing_mcp.tool_payload import as_text  # noqa: F401  (shared with the hook-side text builders)


def source_stamp(path: Path) -> tuple[int, ...]:
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _stamp(stat) -> tuple[int, ...]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


_BLOCK = 1024 * 1024
_TAIL_WINDOW = 64 * 1024  # bytes before ``length`` that later stages re-read instead of the whole prefix
_SLICE_SCAN = 64 * 1024  # bytes read per step when looking for the record boundary a limited capture ends at


class SourceChanged(ValueError):
    """The source was observed to change (rewritten, truncated, replaced) while it was prepared or read.

    A ``ValueError`` so every existing handler keeps working. ``ingest_session`` discards the observation and
    retries a bounded number of times before reporting it.
    """


class SourceObservation:
    """One captured prefix generation of an append-only session file.

    The file may keep growing. Only bytes ``[0, length)`` were observed; ``complete_end`` is the last record
    boundary inside them. No file bytes are retained: the observation keeps device/inode, a streaming
    SHA-256 state per requested checkpoint (with its line count), the 1 MiB block digests of the prefix when
    they were requested, and the digest of the last bytes (the tail window).

    One request performs ONE full scan of the prefix: the capture pass itself. Every later check of the
    observation re-reads only a bounded amount (see ``validate_prefix``); none re-hashes the whole prefix.
    The scan's result is bound to the file's identity, size and ``generation``: a later check that finds a
    different file, a smaller file, a different tail or (without growth) a moved modification time discards it.
    A fast observation (``mode == "fast"``) never scanned the whole prefix: it compared the tail, the latest
    blocks and a random sample with the stored block digests and read only what was appended.
    """

    def __init__(self, path, stamp, length, complete_end, full_digest, checkpoints, tail_digest=None):
        self.path = path
        self.stamp = stamp
        self.length = length
        self.complete_end = complete_end
        self._full_digest = full_digest  # SHA-256 of [0, length); None when only block digests were collected
        self._checkpoints = checkpoints  # offset -> (sha256 state, newline count)
        self._tail_digest = tail_digest  # SHA-256 of the last <= _TAIL_WINDOW bytes of [0, length), as streamed
        self._prefix_ok = True  # A detected identity/size/content mismatch does not heal
        self._mtime_ns = stamp[3]  # the modification time the scan is bound to (see validate_prefix)
        # What validate_prefix() re-checks in the file itself. A plain file may keep growing past it; see
        # CompressedObservation for the immutable variant.
        self._guard_length, self._guard_digest, self._tolerates_growth = length, full_digest, True
        self.blocks: BlockScan | None = None  # block digests collected by the pass (see source_blocks)
        self.mode = "full"  # "full": the whole prefix was scanned; "fast": see the class docstring
        self.full_reason: str | None = None  # why a full scan was done when a fast check was possible
        self.fast_checked: dict | None = None  # {"tail": bool, "recent": [..], "sampled": [..]} for mode "fast"
        self.scanned_at: str | None = None  # when the scan finished (set by the index layer, UTC ISO-8601)
        self.lag_sec: float | None = None  # how far past its deadline a forced full scan was
        self.limited_from: int | None = None  # the size the file had when this capture was cut short (see _slice_length)

    @property
    def identity(self) -> tuple[int, int]:
        return self.stamp[0], self.stamp[1]

    @property
    def target_end(self) -> int:
        """How far a sync of this source goes: the last record boundary observed, or, when the capture was cut short
        (``limited_from``), the size of the file it was cut from (a trailing unfinished record included)."""
        return self.complete_end if self.limited_from is None else max(self.complete_end, self.limited_from)

    @property
    def generation(self) -> str:
        """Names the observed content: identity of the file, covered length and a digest of what was read."""
        table = self.blocks.table_at(self.complete_end) if self.blocks is not None else None
        content = table.root if table is not None else (self._full_digest or "")
        return f"g1:{self.stamp[0]:x}-{self.stamp[1]:x}:{self.complete_end}:{content[:16]}"

    def _open(self):
        return self.path.open("rb")

    def _size_ok(self, size):
        return size >= self._guard_length if self._tolerates_growth else size == self._guard_length

    def _checkpoint(self, offset):
        found = self._checkpoints.get(offset)
        if found is not None:
            return found
        if offset < 0 or offset > self.length:
            raise SourceChanged("source truncated during read")
        digest, lines = hashlib.sha256(), 0
        with self._open() as stream:
            if not same_identity(_identity(os.fstat(stream.fileno())), self.identity):
                raise SourceChanged("source replaced during read")
            remaining = offset
            while remaining:
                check_request()
                block = stream.read(min(remaining, _BLOCK))
                if not block:
                    raise SourceChanged("source truncated during read")
                digest.update(block)
                lines += block.count(b"\n")
                remaining -= len(block)
        self._checkpoints[offset] = (digest, lines)
        return digest, lines

    def prefix_digest(self, length: int) -> str:
        """Plain SHA-256 of ``[0, length)``. Free at a requested checkpoint; anywhere else a re-read."""
        if length < 0 or length > self.length:
            raise SourceChanged("source truncated during read")
        return self._checkpoint(length)[0].hexdigest()

    def line_state(self, offset: int):
        """Hash state (a copy) and line count of ``[0, offset)``."""
        digest, lines = self._checkpoint(offset)
        return digest.copy(), lines

    def table_at(self, offset: int):
        """The block table of ``[0, offset)`` (see ``source_blocks``); only for observations that collected one."""
        scan = self.blocks
        if scan is None:
            raise ValueError("block digests were not collected for this observation")
        table = scan.table_at(offset)
        if table is None:
            self._observe(offset)
            table = scan.table_at(offset)
        if table is None:
            raise ValueError("offset was not observed")
        return table

    def mark_at(self, offset: int):
        """``(digest of the partial block ending at offset | None, newline count of [0, offset))``."""
        if offset not in self.blocks.marks:
            self._observe(offset)
        try:
            return self.blocks.marks[offset]
        except KeyError:
            raise ValueError("offset was not observed") from None

    def _observe(self, offset: int) -> None:
        """Compute the mark of an offset the pass did not stop at (rare): re-read one block's head."""
        scan, size = self.blocks, self.blocks.block_size
        block = offset // size
        if offset < scan.first_block * size or offset > self.length or block - scan.first_block > len(scan.block_lines):
            return
        with self._open() as stream:
            if not same_identity(_identity(os.fstat(stream.fileno())), self.identity):
                raise SourceChanged("source replaced during read")
            stream.seek(block * size)
            data = stream.read(offset - block * size)
        if len(data) != offset - block * size:
            raise SourceChanged("source truncated during read")
        lines = scan.origin + sum(scan.block_lines[:block - scan.first_block]) + data.count(b"\n")
        scan.marks[offset] = (hashlib.sha256(data).digest() if data else None, lines)

    def cursor_hash(self, offset: int, kind: str = "sha") -> str:
        """What a cursor stores for the prefix ``[0, offset)``: the plain SHA-256 or the block root (``kind``)."""
        return self.table_at(offset).hash if kind == "blocks" else self.prefix_digest(offset)

    def consume_guard(self, start: int):
        """Verifies, byte by byte, that what a parser consumes from ``start`` is what the scan observed."""
        have_sha = start in self._checkpoints or self.blocks is None
        return _ShaGuard(self, start) if have_sha else _BlockGuard(self, start)

    def validate(self) -> bool:
        """Compare identity, size and timestamps only; this does not verify content equality."""
        check_request()
        try:
            return source_stamp(self.path) == self.stamp
        except OSError:
            return False

    def current_size(self) -> int:
        return self.path.stat().st_size

    def validate_prefix(self, *, locked: bool = False, full: bool = False) -> bool:
        """Is the file still the one the scan saw? Bounded: this never re-hashes the whole prefix.

        The capture pass is the request's one full scan. Later stages re-check, without trusting stamps alone:
        the file identity, ``size >= captured length`` (an immutable ``.zst`` snapshot: exactly its size) and the
        digest of the tail window, as streamed during the scan. Without growth, an unchanged size with a moved
        modification time means the file was written; the unlocked check then settles it with a full re-read
        (identical bytes, e.g. a ``touch``, pass; the new time becomes the baseline). ``locked=True`` is for use
        under the publication guard: identity, size and tail only, never a re-read of the prefix.
        ``full=True`` forces the whole-prefix re-read. A detected mismatch stays failed even if the original
        bytes return.

        Residual risk, stated precisely: a rewrite that leaves identity, size, the tail window and the
        modification time unchanged and lies outside the tail window, made after the capture pass of the same
        request, is accepted at that moment. The next full scan (the next sync's capture, or a full
        verification) compares the stored cursor hash and reindexes. ``.zst`` snapshots are different: any
        change to the compressed file is a rewrite, and their digest check re-reads the (compressed) file.
        """
        check_request()
        if not self._prefix_ok:
            return False
        if full or not self._tolerates_growth:
            return self._rescan()
        try:
            with self.path.open("rb") as stream:
                before = os.fstat(stream.fileno())
                if not same_identity(_identity(before), self.identity) or not self._size_ok(before.st_size):
                    return self._fail()
                if self._tail_digest is None:
                    return False
                window = min(self._guard_length, _TAIL_WINDOW)
                stream.seek(self._guard_length - window)
                tail = stream.read(window)
        except OSError:
            return False
        if len(tail) != window or hashlib.sha256(tail).hexdigest() != self._tail_digest:
            return self._fail()
        if not locked and before.st_size == self._guard_length and before.st_mtime_ns != self._mtime_ns:
            return self._rescan()
        return True

    def _rescan(self) -> bool:
        """Re-read the whole guarded region and compare it with what the scan saw."""
        try:
            with self.path.open("rb") as stream:
                before = os.fstat(stream.fileno())
                if not same_identity(_identity(before), self.identity) or not self._size_ok(before.st_size):
                    return self._fail()
                if self._guard_digest is not None:
                    same = self._reread_sha(stream)
                elif self.blocks is not None and self.blocks.first_block == 0:
                    same = self._reread_blocks(stream)
                else:
                    return self._fail()  # a fast observation holds no whole-prefix digest to compare with
                if same is None:
                    return self._fail()
                after = os.fstat(stream.fileno())
        except OSError:
            return False
        if not same or not same_identity(_identity(after), self.identity) or not self._size_ok(after.st_size):
            return self._fail()
        if after.st_size == self._guard_length:
            self._mtime_ns = after.st_mtime_ns
        return True

    def _reread_sha(self, stream):
        digest, remaining = hashlib.sha256(), self._guard_length
        buffer = bytearray(min(remaining, _BLOCK))  # one reused buffer: no per-block bytes objects
        view = memoryview(buffer)
        while remaining:
            check_request()
            count = stream.readinto(view[:min(remaining, len(buffer))])
            if not count:
                return None
            digest.update(view[:count])
            remaining -= count
        return digest.hexdigest() == self._guard_digest

    def _reread_blocks(self, stream):
        """Compare the guarded region with the block digests of the scan; one reused buffer, no per-block objects."""
        scan, size = self.blocks, self.blocks.block_size
        buffer = bytearray(size)
        view = memoryview(buffer)

        def fill(count):
            filled = 0
            while filled < count:
                check_request()
                got = stream.readinto(view[filled:count])
                if not got:
                    return False  # early EOF: the file is shorter than what was observed
                filled += got
            return True

        for expected in scan.digests:
            if not fill(size):
                return None
            if hashlib.sha256(view).digest() != expected:
                return False
        remaining = self.length - len(scan.digests) * size
        if remaining:
            if not fill(remaining):
                return None
            return hashlib.sha256(view[:remaining]).digest() == scan.marks.get(self.length, (None, 0))[0]
        return True

    def _fail(self) -> bool:
        self._prefix_ok = False
        return False


class CompressedObservation(SourceObservation):
    """One captured ``.zst`` source: an immutable snapshot whose offsets address the decompressed stream.

    ``length``, the line states and ``prefix_digest`` come from one streaming decode and are exactly what a
    plain file with the same bytes yields, so cursors move between the two forms. Decoding is deterministic,
    so the SHA-256 of the COMPRESSED bytes stands in for the decompressed content when the snapshot is
    validated (no decoder pass); any growth, truncation or rewrite of the compressed file is a rewrite, never
    pure growth. The decoded stream has no unread tail to wait for: ``current_size()`` is its fixed length.
    """

    def __init__(self, path, stamp, length, complete_end, full_digest, checkpoints, tail_digest, *,
                 raw_size, raw_digest, max_bytes):
        super().__init__(path, stamp, length, complete_end, full_digest, checkpoints, tail_digest)
        self.raw_size, self.raw_digest, self.max_bytes = raw_size, raw_digest, max_bytes
        self._guard_length, self._guard_digest, self._tolerates_growth = raw_size, raw_digest, False

    @property
    def generation(self) -> str:
        return f"g1:{self.stamp[0]:x}-{self.stamp[1]:x}:{self.raw_size}:{self.raw_digest[:16]}"

    def _open(self):
        return open_source(self.path, max_bytes=self.max_bytes)

    def current_size(self):
        self.path.stat()  # a vanished file is an error here, as for a plain source
        return self.length


def _identity(stat) -> tuple[int, int]:
    return stat.st_dev, stat.st_ino


def same_identity(left, right) -> bool:
    """(st_dev, st_ino) pairs from different stat APIs. Some Windows filesystems report st_ino == 0 on one side
    (or on both); that side carries no identity, so the pair is skipped and size/mtime/hash decide. Two
    non-zero inodes are always compared."""
    return left[1] == 0 or right[1] == 0 or tuple(left) == tuple(right)


class _Pass:
    """Result of one streaming pass: running SHA-256 (optional), record boundary, states, block scan, tail."""

    __slots__ = ("digest", "boundary", "states", "scan", "tail")

    def __init__(self, digest, boundary, states, scan, tail):
        self.digest, self.boundary, self.states, self.scan, self.tail = digest, boundary, states, scan, tail


def _hash_prefix(stream, length, checkpoints, *, sha=True, scan=None, start=0, min_boundary=0, tail_window=0):
    """Stream ``[start, length)`` once.

    ``sha``: keep the running SHA-256 of ``[0, length)`` (then ``start`` is 0), with a state at every requested
    checkpoint and at the last record boundary. ``scan``: a ``BlockScan`` that receives the digest of every
    complete block (``start`` is block aligned) and a mark at ``start``, at every checkpoint, at ``length`` and
    at the last record boundary after ``min_boundary``. ``tail_window``: keep the last bytes streamed.
    States and marks are small; memory is O(1) in the file size.
    """
    assert not (sha and start), "a running SHA-256 starts at byte 0"
    size = scan.block_size if scan is not None else _BLOCK
    wanted = sorted({o for o in checkpoints if start < o <= length})
    wanted_set = set(wanted)
    marked = wanted_set | {length}
    digest = hashlib.sha256() if sha else None
    block_hash = hashlib.sha256() if scan is not None else None
    lines = block_start_lines = 0
    position, boundary, tail = start, min_boundary, b""
    states = {0: (digest.copy(), 0)} if sha else {}
    boundary_state, boundary_mark = states.get(0), None
    if scan is not None:
        scan.marks[start] = (None, 0)
    if start:
        stream.seek(start)
    while position < length:
        check_request()
        data = stream.read(min(size, length - position))
        if not data:
            raise SourceChanged("source truncated during read")
        end = position + len(data)
        newline = data.rfind(b"\n")
        splits = {o - position for o in wanted[bisect.bisect_right(wanted, position):bisect.bisect_right(wanted, end)]}
        if newline >= 0 and position + newline + 1 > min_boundary:
            splits.add(newline + 1)
        if scan is not None:
            next_block = (position // size + 1) * size
            while next_block <= end:
                splits.add(next_block - position)
                next_block += size
        cursor = 0
        for split in sorted(splits | {len(data)}):
            piece = data if cursor == 0 and split == len(data) else data[cursor:split]
            lines += piece.count(b"\n")
            cursor = split
            absolute = position + split
            if sha:
                digest.update(piece)
                if absolute in wanted_set:
                    states[absolute] = (digest.copy(), lines)
            at_boundary = newline >= 0 and split == newline + 1 and absolute > min_boundary
            if scan is not None:
                block_hash.update(piece)
                aligned = absolute % size == 0
                partial = None if aligned else (block_hash.copy().digest()
                                                if absolute in marked or at_boundary else None)
                if aligned:
                    scan.digests.append(block_hash.digest())
                    scan.block_lines.append(lines - block_start_lines)
                    block_start_lines, block_hash = lines, hashlib.sha256()
                if absolute in marked:
                    scan.marks[absolute] = (partial, lines)
                if absolute in wanted_set and not aligned:
                    scan.states[absolute] = block_hash.copy()
                if at_boundary:
                    boundary_mark = (partial, lines)
            if at_boundary:
                boundary = absolute
                boundary_state = (digest.copy(), lines) if sha else None
        position = end
        if tail_window:
            tail = (tail + data)[-tail_window:]
    if scan is not None and boundary_mark is not None:
        scan.marks.setdefault(boundary, boundary_mark)
    if sha:
        states.setdefault(length, (digest.copy(), lines))
        states.setdefault(boundary, boundary_state)
    return _Pass(digest.hexdigest() if sha else None, boundary, states, scan, tail)


class _ShaGuard:
    """Consumed bytes are hashed from the SHA-256 state at ``start``; the end must match the observed prefix."""

    def __init__(self, observation, start):
        self.observation = observation
        self._digest, self.lines = observation.line_state(start)

    def update(self, raw):
        self._digest.update(raw)

    def finish(self, position):
        if self._digest.hexdigest() != self.observation.prefix_digest(position):
            raise SourceChanged("source changed during read")


class _BlockGuard:
    """Consumed bytes are hashed block by block; every block must equal the digest the scan observed.

    The block that contains ``start`` begins before it: its earlier bytes are read back once to seed the block
    hash, so the whole block is verified, not just the consumed part.
    """

    def __init__(self, observation, start):
        scan = observation.blocks
        self.observation, self.scan, self.size = observation, scan, scan.block_size
        self.position, self.index = start, start // self.size
        self.hasher = hashlib.sha256()
        self.lines = observation.mark_at(start)[1]
        head = start - self.index * self.size
        state = scan.states.get(start)
        if head and state is not None:
            self.hasher = state.copy()  # the scan hashed the head of this block already
        elif head:
            with observation._open() as stream:
                if not same_identity(_identity(os.fstat(stream.fileno())), observation.identity):
                    raise SourceChanged("source replaced during read")
                stream.seek(self.index * self.size)
                data = stream.read(head)
            if len(data) != head:
                raise SourceChanged("source truncated during read")
            self.hasher.update(data)

    def update(self, raw):
        view = memoryview(raw)
        while len(view):
            room = (self.index + 1) * self.size - self.position
            piece = view[:room]
            self.hasher.update(piece)
            self.position += len(piece)
            view = view[len(piece):]
            if self.position == (self.index + 1) * self.size:
                if self.hasher.digest() != self.scan.block_digest(self.index):
                    raise SourceChanged("source changed during read")
                self.index, self.hasher = self.index + 1, hashlib.sha256()

    def finish(self, position):
        if position != self.position:
            raise SourceChanged("source changed during read")
        expected = self.observation.mark_at(position)[0]
        if expected is not None and self.hasher.digest() != expected:
            raise SourceChanged("source changed during read")


def _capture_compressed(path: Path, checkpoints, max_bytes, sha=True, blocks=False) -> CompressedObservation:
    """One streaming decode of a ``.zst`` source: stream digest, line states, record boundary and the digest of
    the compressed bytes. A compressed file is immutable, so any change during the pass is a rewrite."""
    limit = DEFAULT_MAX_DECOMPRESSED_BYTES if max_bytes is None else max_bytes
    with open_source(path, max_bytes=limit, tap=True) as stream:
        reader = stream.raw
        before = os.fstat(stream.fileno())
        length = reader.declared
        scan = BlockScan(None, 0, source_blocks.BLOCK_SIZE) if blocks else None
        result = _hash_prefix(stream, length, checkpoints, sha=sha, scan=scan)
        reader.verify_complete()  # exactly one frame, ending exactly at the declared size
        reader.drain()
        after = os.fstat(stream.fileno())
        raw_size, raw_digest, tail = reader.tap.count, reader.tap.digest.hexdigest(), reader.tap.tail()
    path_stamp = source_stamp(path)
    if (_stamp(after) != _stamp(before) or after.st_size != raw_size or not same_identity(path_stamp[:2], _identity(after))
            or path_stamp[2] != raw_size or path_stamp[3] != after.st_mtime_ns):
        raise SourceChanged("source changed during preparation")
    observation = CompressedObservation(path, path_stamp, length, result.boundary, result.digest, result.states,
                                        hashlib.sha256(tail).hexdigest(), raw_size=raw_size, raw_digest=raw_digest,
                                        max_bytes=limit)
    observation.blocks = scan
    return observation


def _check_pass_consistency(observation, before, after, path_stamp, length):
    """Shared end-of-pass checks of a plain-file capture (full or fast)."""
    if (not same_identity(_identity(after), _identity(before)) or after.st_size < length
            or not same_identity(path_stamp[:2], _identity(after)) or path_stamp[2] < after.st_size
            or (path_stamp[2] == after.st_size and path_stamp[3] != after.st_mtime_ns)):
        raise SourceChanged("source changed during preparation")
    if after.st_size == length and path_stamp[2] == length:
        if _stamp(after) != _stamp(before):
            raise SourceChanged("source changed during preparation")  # same size, new mtime: rewritten in place
    else:
        # Grew while hashing: appended bytes are fine. A rewrite of the tail window during the pass is not; the
        # bounded check compares it with the bytes that were streamed (the pass is the request's only full scan).
        if not observation.validate_prefix():
            raise SourceChanged("source changed during preparation")


def _slice_length(stream, size: int, floor: int, cap: int) -> tuple[int, int | None]:
    """``(length, cut_from)``: how much of a ``size``-byte plain file one slice observes.

    A source that is indexed up to ``floor`` (a record boundary: the cursor) is observed at most up to ``floor +
    cap``, rounded down to the end of the last record inside that bound. A slice always holds one whole record: when
    none ends inside the bound, the observation ends at the first record end after it. The result behaves exactly
    like a capture of a file that was that short (``validate_prefix`` already tolerates growth); ``cut_from`` is the
    size the file really had. A file that fits, or whose rest is one unfinished record, is observed whole (``None``).
    The stream is left at byte 0.
    """
    wanted = floor + cap
    if size <= wanted:
        return size, None
    end, position = None, wanted
    while end is None and position > floor:             # the last newline in [floor, wanted)
        check_request()
        begin = max(floor, position - _SLICE_SCAN)
        stream.seek(begin)
        found = stream.read(position - begin).rfind(b"\n")
        if found >= 0:
            end = begin + found + 1
        position = begin
    position = wanted
    while end is None and position < size:              # none inside it: the first newline after it
        check_request()
        stream.seek(position)
        data = stream.read(min(_BLOCK, size - position))
        if not data:
            raise SourceChanged("source truncated during read")
        found = data.find(b"\n")
        if found >= 0:
            end = position + found + 1
        position += len(data)
    stream.seek(0)
    if end is None or end >= size:
        return size, None
    return end, size


def capture_source(path: Path, *, checkpoints=(), max_bytes=None, sha=True, blocks=False,
                   window=None) -> SourceObservation:
    """Observe the first ``size`` bytes of a possibly growing source in one streaming pass.

    Bytes are hashed, not retained. Growth after the initial ``fstat`` is not an error; an overwrite,
    truncation or replacement during the pass is. A ``.zst`` source is decoded as a stream and observed as an
    immutable snapshot (``max_bytes`` bounds its declared decompressed size); see ``CompressedObservation``.
    ``sha`` keeps the running SHA-256 (plain cursors); ``blocks`` collects the 1 MiB block digests
    (``source_blocks``) in the same pass. At least one of them must be requested.

    ``window=(floor, cap)``: a plain source longer than ``floor + cap`` is observed only up to the end of the last
    record inside that bound (``_slice_length``); ``limited_from`` of the observation says so. A ``.zst`` source is
    never cut.
    """
    check_request()
    if not (sha or blocks):
        raise ValueError("a capture needs a plain digest, block digests or both")
    path = Path(path).resolve()
    if is_compressed(path):
        return _capture_compressed(path, checkpoints, max_bytes, sha, blocks)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        length, cut_from = (before.st_size, None) if window is None else _slice_length(stream, before.st_size, *window)
        scan = BlockScan(None, 0, source_blocks.BLOCK_SIZE) if blocks else None
        result = _hash_prefix(stream, length, checkpoints, sha=sha, scan=scan, tail_window=_TAIL_WINDOW)
        after = os.fstat(stream.fileno())
    path_stamp = source_stamp(path)
    # Some runtimes (e.g. 9p/drvfs) give stat and fstat different ctime semantics. Compare full stamps
    # within each API, and only identity, size and mtime across them. validate() later compares stat to
    # stat, so the observation keeps the path stamp.
    observation = SourceObservation(path, path_stamp, length, result.boundary, result.digest, result.states,
                                    hashlib.sha256(result.tail).hexdigest())
    observation.blocks, observation.limited_from = scan, cut_from
    _check_pass_consistency(observation, before, after, path_stamp, length)
    return observation


def capture_source_fast(path: Path, record, *, recent: int, samples: int, rng=None, cap=None) -> SourceObservation:
    """Observe a source without scanning its whole prefix, using the block table stored for it.

    Compares, in order: the file's identity, that it did not shrink, that a same-size file did not get a new
    modification time (advisory signals, a difference means "do this in full"), the latest ``recent`` blocks and
    ``samples`` random blocks with their stored digests, and the partial tail block; then reads only what was
    appended after the table's offset. The result is an observation with ``mode == "fast"``: it vouches for the
    bytes that were compared and read, and for nothing else. Raises ``FastFallback`` whenever it cannot.

    ``cap``: read at most that many bytes past the table's offset, up to the end of a record (``_slice_length``).
    """
    check_request()
    path = Path(path).resolve()
    table, size = record.table, record.table.block_size
    if size != source_blocks.BLOCK_SIZE or not table.consistent():
        raise FastFallback("record_unusable")
    rng = rng if rng is not None else source_blocks.new_rng()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        length = before.st_size
        if not same_identity(_identity(before), (record.dev, record.ino)):
            raise FastFallback("identity_changed")
        if length < table.covered:
            raise FastFallback("shrunk")
        if length == record.size and before.st_mtime_ns != record.mtime_ns:
            raise FastFallback("same_size_modified")
        cut_from = None
        if cap is not None:
            length, cut_from = _slice_length(stream, length, table.covered, cap)
        picked = source_blocks.choose_blocks(table.nfull, recent, samples, rng)
        for index in picked:
            check_request()
            stream.seek(index * size)
            data = stream.read(size)
            if len(data) != size or hashlib.sha256(data).digest() != table.digest(index):
                raise FastFallback("block_mismatch", index)
        scan = BlockScan(table, table.nfull, size)
        result = _hash_prefix(stream, length, (table.covered,), sha=False, scan=scan, start=table.tail_start,
                              min_boundary=table.covered, tail_window=_TAIL_WINDOW)
        covered_mark = scan.marks.get(table.covered)
        if covered_mark is None or covered_mark[0] != table.tail:
            raise FastFallback("tail_mismatch", table.nfull)
        scan.origin = table.lines - covered_mark[1]  # marks and block counts were relative to the tail block
        scan.marks = {offset: (partial, lines + scan.origin) for offset, (partial, lines) in scan.marks.items()}
        tail = result.tail
        window = min(length, _TAIL_WINDOW)
        if len(tail) < window:  # the scanned region is shorter than the window: the rest lies in earlier blocks
            stream.seek(length - window)
            tail = stream.read(window - len(tail)) + tail
        after = os.fstat(stream.fileno())
    path_stamp = source_stamp(path)
    observation = SourceObservation(path, path_stamp, length, result.boundary, None, {},
                                    hashlib.sha256(tail).hexdigest() if len(tail) == window else None)
    observation.blocks, observation.mode, observation.limited_from = scan, "fast", cut_from
    observation.fast_checked = {"tail": True, "recent": [i for i in picked if i >= table.nfull - max(0, recent)],
                                "sampled": [i for i in picked if i < table.nfull - max(0, recent)]}
    _check_pass_consistency(observation, before, after, path_stamp, length)
    return observation


_source_observation: ContextVar[SourceObservation | None] = ContextVar("source_observation", default=None)


def observed_source(path: Path) -> SourceObservation | None:
    observation = _source_observation.get()
    if observation is None:
        return None
    candidate = Path(path)
    # ``observation.path`` is resolved: an equal path needs no filesystem lookup (one per result record otherwise).
    return observation if candidate == observation.path or candidate.resolve() == observation.path else None


@contextmanager
def source_scope(observation: SourceObservation):
    token = _source_observation.set(observation)
    try:
        yield observation
    finally:
        _source_observation.reset(token)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# A byte that is not valid UTF-8 decodes (``errors="surrogateescape"``, see ``iter_raw_lines``) to a lone surrogate in
# U+DC80..U+DCFF. One compiled scan in C replaces a Python generator that visited every character of every record.
_ESCAPED_BYTE = re.compile("[\udc80-\udcff]")


def has_invalid_utf8(raw: str) -> bool:
    """True when ``raw`` (a record decoded with ``surrogateescape``) holds a byte that was not valid UTF-8."""
    return _ESCAPED_BYTE.search(raw) is not None


def iter_raw_lines(path: Path, *, start_byte: int = 0) -> Iterator[tuple[int, int, int, str]]:
    """Cursors point to the start of a record. Leave unfinished final records unread.

    Under a ``source_scope`` the read is bounded to the observed prefix and every consumed byte is hashed
    against the capture (against the SHA-256 state, or block by block when the capture holds block digests), so
    a rewrite between capture and parse is an error, not silent wrong evidence.

    Under a ``slice_scope`` (``slice_budget``) the iterator that starts at the slice's offset may end early: before
    it reads a record, once the budget says so, it stops at that record boundary. What it consumed is verified
    against the capture exactly like a read to the end (``guard.finish``), so a prefix is never published unverified.
    A compressed source is never cut short.
    """
    check_request()
    if start_byte < 0:
        raise ValueError("negative byte cursor")
    observation = observed_source(path)
    limit = observation.complete_end if observation else None
    if observation is not None and start_byte > observation.complete_end:
        raise ValueError("cursor is not on a complete record boundary")
    size_limit = getattr(observation, "max_bytes", None)
    budget = current_slice()
    if budget is not None and (start_byte != budget.start_byte or size_limit is not None):
        budget = None  # only the parse loop of the slice is bounded: not a probe of the head, nor a .zst decoder
    # A compressed source is read through a seekable decoder: forward seeks decode and discard, backward seeks
    # restart, so the cursor logic below is the same for both forms. A plain path opens as before.
    with (open_source(path) if size_limit is None else open_source(path, max_bytes=size_limit)) as stream:
        if observation is not None and not same_identity(_identity(os.fstat(stream.fileno())), observation.identity):
            raise SourceChanged("source replaced during read")
        guard = None
        if start_byte:
            stream.seek(start_byte - 1)
            if stream.read(1) != b"\n":
                raise ValueError("cursor is not on a complete record boundary")
        if observation is not None:
            guard = observation.consume_guard(start_byte)
            line_no = guard.lines
        else:
            line_no = 0
            stream.seek(0)
            remaining = start_byte
            while remaining:
                check_request()
                block = stream.read(min(remaining, _BLOCK))
                if len(block) == 0:
                    raise ValueError("cursor is not on a complete record boundary")
                line_no += block.count(b"\n")
                remaining -= len(block)
        stream.seek(start_byte)
        position = start_byte
        records, next_look, cut = 0, 0, False
        while True:
            check_request()
            if limit is not None and position >= limit:
                break
            if budget is not None and records >= next_look:
                reason, next_look = budget.look(records, position - start_byte)
                if reason is not None:
                    budget.stop(reason, records, position - start_byte)
                    cut = True
                    break
            raw = stream.readline() if limit is None else stream.readline(limit - position)
            if not raw or not raw.endswith(b"\n"):
                break
            start, position = position, position + len(raw)
            line_no += 1
            records += 1
            if guard is not None:
                guard.update(raw)
            # Invalid UTF-8 remains detectable instead of silently altering evidence.
            yield line_no, start, position, raw.decode("utf-8", errors="surrogateescape")
    if observation is not None and guard is not None and (cut or position == observation.complete_end):
        guard.finish(position)


def make_chunk(provider, sid, kind, text, meta, line, start, end, sub=0, timestamp=None):
    digest = content_hash(f"{kind}|{text}")
    # Include content digest to invalidate anchors after a source rewrite at the same offset.
    event_digest = content_hash(f"{digest}|{as_text(meta)}|{timestamp}")
    event_id = f"{provider}:{sid}:{start}:{sub}:{event_digest[:16]}"
    return Chunk(chunk_id=event_id, session_id=sid, provider=provider, kind=kind,
                 text=text, content_hash=digest, timestamp=timestamp,
                 line_start=line, line_end=line, byte_start=start, byte_end=end,
                 meta={**meta, "provider": provider, "sub_index": sub})


def parse_error(provider, sid, line, start, end, reason):
    return make_chunk(provider, sid, "meta", f"[parse_error] {reason}",
                      {"parse_error": True}, line, start, end)


def result_success(payload: Any) -> bool | None:
    """Use structured status / explicit runtime exit status, never prose such as 'passed'."""
    if isinstance(payload, str):
        try:
            return result_success(json.loads(payload))
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"(?im)^\s*(?:Process exited with code|Process exit code|Exit code)\s*:?\s*(-?\d+)\s*$", payload)
            return int(match.group(1)) == 0 if match else None
    if isinstance(payload, dict):
        for key in ("exit_code", "exitCode"):
            value = payload.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value == 0
        if payload.get("is_error") is True:
            return False
        for key in ("output", "content"):
            if key in payload:
                status = result_success(payload[key])
                if status is not None:
                    return status
    if isinstance(payload, list):
        statuses = [result_success(block.get("text")) for block in payload if isinstance(block, dict)]
        if False in statuses:
            return False
        if True in statuses:
            return True
    return None
