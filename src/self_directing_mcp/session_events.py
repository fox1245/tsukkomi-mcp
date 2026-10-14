"""Shared lossless JSONL and event helpers. Raw evidence stays local."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
import os
import json
import re
from pathlib import Path
from typing import Any, Iterator

from self_directing_mcp.schemas import Chunk
from self_directing_mcp.request_control import check_request


def source_stamp(path: Path) -> tuple[int, ...]:
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _stamp(stat) -> tuple[int, ...]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


_BLOCK = 1024 * 1024
_TAIL_WINDOW = 64 * 1024  # bytes before ``length`` re-hashed under the publication lock instead of the whole prefix


class SourceObservation:
    """One captured prefix generation of an append-only session file.

    The file may keep growing. Only bytes ``[0, length)`` were observed; ``complete_end`` is the last record
    boundary inside them. No file bytes are retained: the observation keeps device/inode, a streaming
    SHA-256 state per requested checkpoint (with its line count) and the digest of the whole prefix.
    ``validate()`` is the strict "file untouched since capture" test; ``validate_prefix()`` accepts pure
    growth after verifying the captured prefix still hashes identically. ``validate_prefix(locked=True)`` is the
    bounded variant for use under the publication guard (see there).
    """

    def __init__(self, path, stamp, length, complete_end, full_digest, checkpoints, tail_digest=None):
        self.path = path
        self.stamp = stamp
        self.length = length
        self.complete_end = complete_end
        self._full_digest = full_digest
        self._checkpoints = checkpoints  # offset -> (sha256 state, newline count)
        self._verified_stamp = stamp
        self._tail_digest = tail_digest  # SHA-256 of the last <= _TAIL_WINDOW bytes of [0, length)
        self._prefix_ok = True  # False once a full verification failed; the prefix does not heal

    @property
    def identity(self) -> tuple[int, int]:
        return self.stamp[0], self.stamp[1]

    def _checkpoint(self, offset):
        found = self._checkpoints.get(offset)
        if found is not None:
            return found
        if offset < 0 or offset > self.length:
            raise ValueError("source truncated during read")
        digest, lines = hashlib.sha256(), 0
        with self.path.open("rb") as stream:
            if not same_identity(_identity(os.fstat(stream.fileno())), self.identity):
                raise ValueError("source replaced during read")
            remaining = offset
            while remaining:
                check_request()
                block = stream.read(min(remaining, _BLOCK))
                if not block:
                    raise ValueError("source truncated during read")
                digest.update(block)
                lines += block.count(b"\n")
                remaining -= len(block)
        self._checkpoints[offset] = (digest, lines)
        return digest, lines

    def prefix_digest(self, length: int) -> str:
        if length < 0 or length > self.length:
            raise ValueError("source truncated during read")
        return self._checkpoint(length)[0].hexdigest()

    def line_state(self, offset: int):
        """Hash state (a copy) and line count of ``[0, offset)``."""
        digest, lines = self._checkpoint(offset)
        return digest.copy(), lines

    def validate(self) -> bool:
        """Strict: the file's identity, size and timestamps are exactly the captured ones."""
        check_request()
        try:
            return source_stamp(self.path) == self.stamp
        except OSError:
            return False

    def current_size(self) -> int:
        return self.path.stat().st_size

    def validate_prefix(self, *, locked: bool = False) -> bool:
        """Growth tolerant: same file, and the captured prefix is byte-for-byte what was hashed.

        The default re-reads and hashes the whole prefix; call it OUTSIDE the publication guard. With
        ``locked=True`` (inside the guard) only identity, size >= captured length and the digest of a bounded
        tail window are checked, and a failed earlier full check still fails. Residual risk: a rewrite
        outside the tail window between the last full check and the lock is accepted at that moment; the next
        sync compares the stored cursor ``prefix_hash`` and reindexes.
        """
        check_request()
        if locked:
            return self._validate_tail()
        try:
            with self.path.open("rb") as stream:
                before = os.fstat(stream.fileno())
                if not same_identity(_identity(before), self.identity) or before.st_size < self.length:
                    return self._fail()
                if _stamp(before) == self._verified_stamp:
                    return True
                digest, remaining = hashlib.sha256(), self.length
                while remaining:
                    check_request()
                    block = stream.read(min(remaining, _BLOCK))
                    if not block:
                        return self._fail()
                    digest.update(block)
                    remaining -= len(block)
                after = os.fstat(stream.fileno())
        except OSError:
            return False
        if digest.hexdigest() != self._full_digest or not same_identity(_identity(after), self.identity) or after.st_size < self.length:
            return self._fail()
        if _stamp(after) == _stamp(before):
            self._verified_stamp = _stamp(after)
        return True

    def _fail(self) -> bool:
        self._prefix_ok = False
        return False

    def _validate_tail(self) -> bool:
        if not self._prefix_ok:
            return False
        try:
            with self.path.open("rb") as stream:
                before = os.fstat(stream.fileno())
                if not same_identity(_identity(before), self.identity) or before.st_size < self.length:
                    return self._fail()
                if _stamp(before) == self._verified_stamp:
                    return True
                if self._tail_digest is None:
                    return False
                window = min(self.length, _TAIL_WINDOW)
                stream.seek(self.length - window)
                tail = stream.read(window)
        except OSError:
            return False
        if len(tail) != window or hashlib.sha256(tail).hexdigest() != self._tail_digest:
            return self._fail()
        return True


def _identity(stat) -> tuple[int, int]:
    return stat.st_dev, stat.st_ino


def same_identity(left, right) -> bool:
    """(st_dev, st_ino) pairs from different stat APIs. Some Windows filesystems report st_ino == 0 on one side
    (or on both); that side carries no identity, so the pair is skipped and size/mtime/hash decide. Two
    non-zero inodes are always compared."""
    return left[1] == 0 or right[1] == 0 or tuple(left) == tuple(right)


def _hash_prefix(stream, length, checkpoints):
    """Stream ``[0, length)`` once. Returns (full digest, last record boundary, {offset: (state, lines)}).

    States are kept for every requested checkpoint and for the last record boundary; memory is O(1) in
    the file size apart from those small hash states.
    """
    wanted = sorted({o for o in checkpoints if 0 < o <= length})
    digest, lines, position, boundary = hashlib.sha256(), 0, 0, 0
    states = {0: (digest.copy(), 0)}
    boundary_state = states[0]
    while position < length:
        check_request()
        block = stream.read(min(_BLOCK, length - position))
        if not block:
            raise ValueError("source truncated during read")
        newline = block.rfind(b"\n")
        splits = {o - position for o in wanted if position < o <= position + len(block)}
        if newline >= 0:
            splits.add(newline + 1)
        cursor = 0
        for split in sorted(splits | {len(block)}):
            piece = block[cursor:split]
            digest.update(piece)
            lines += piece.count(b"\n")
            cursor = split
            absolute = position + split
            if absolute in wanted:
                states[absolute] = (digest.copy(), lines)
            if newline >= 0 and split == newline + 1:
                boundary, boundary_state = absolute, (digest.copy(), lines)
        position += len(block)
    states.setdefault(length, (digest.copy(), lines))
    states.setdefault(boundary, boundary_state)
    return digest.hexdigest(), boundary, states


def capture_source(path: Path, *, checkpoints=()) -> SourceObservation:
    """Observe the first ``size`` bytes of a possibly growing source in one streaming pass.

    Bytes are hashed, not retained. Growth after the initial ``fstat`` is not an error; an overwrite,
    truncation or replacement during the pass is.
    """
    check_request()
    path = Path(path).resolve()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        length = before.st_size
        full_digest, boundary, states = _hash_prefix(stream, length, checkpoints)
        window = min(length, _TAIL_WINDOW)
        stream.seek(length - window)
        tail = stream.read(window)
        tail_digest = hashlib.sha256(tail).hexdigest() if len(tail) == window else None
        after = os.fstat(stream.fileno())
    path_stamp = source_stamp(path)
    # Some runtimes (e.g. 9p/drvfs) give stat and fstat different ctime semantics. Compare full stamps
    # within each API, and only identity, size and mtime across them. validate() later compares stat to
    # stat, so the observation keeps the path stamp.
    if (not same_identity(_identity(after), _identity(before)) or after.st_size < length
            or not same_identity(path_stamp[:2], _identity(after)) or path_stamp[2] < after.st_size
            or (path_stamp[2] == after.st_size and path_stamp[3] != after.st_mtime_ns)):
        raise ValueError("source changed during preparation")
    observation = SourceObservation(path, path_stamp, length, boundary, full_digest, states, tail_digest)
    observation._verified_stamp = _stamp(after)
    if after.st_size == length and path_stamp[2] == length:
        if _stamp(after) != _stamp(before):
            raise ValueError("source changed during preparation")  # same size, new mtime: rewritten in place
    else:
        # Grew while hashing: appended bytes are fine, a rewritten prefix is not.
        observation._verified_stamp = None
        if not observation.validate_prefix():
            raise ValueError("source changed during preparation")
    return observation


_source_observation: ContextVar[SourceObservation | None] = ContextVar("source_observation", default=None)


def observed_source(path: Path) -> SourceObservation | None:
    observation = _source_observation.get()
    return observation if observation is not None and Path(path).resolve() == observation.path else None


@contextmanager
def source_scope(observation: SourceObservation):
    token = _source_observation.set(observation)
    try:
        yield observation
    finally:
        _source_observation.reset(token)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_raw_lines(path: Path, *, start_byte: int = 0) -> Iterator[tuple[int, int, int, str]]:
    """Cursors point to the start of a record. Leave unfinished final records unread.

    Under a ``source_scope`` the read is bounded to the observed prefix and every consumed byte is hashed
    against the capture, so a rewrite between capture and parse is an error, not silent wrong evidence.
    """
    check_request()
    if start_byte < 0:
        raise ValueError("negative byte cursor")
    observation = observed_source(path)
    limit = observation.complete_end if observation else None
    if observation is not None and start_byte > observation.complete_end:
        raise ValueError("cursor is not on a complete record boundary")
    with Path(path).open("rb") as stream:
        if observation is not None and not same_identity(_identity(os.fstat(stream.fileno())), observation.identity):
            raise ValueError("source replaced during read")
        check_digest = None
        if start_byte:
            stream.seek(start_byte - 1)
            if stream.read(1) != b"\n":
                raise ValueError("cursor is not on a complete record boundary")
        if observation is not None:
            check_digest, line_no = observation.line_state(start_byte)
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
        while True:
            check_request()
            if limit is not None and position >= limit:
                break
            raw = stream.readline() if limit is None else stream.readline(limit - position)
            if not raw or not raw.endswith(b"\n"):
                break
            start, position = position, position + len(raw)
            line_no += 1
            if check_digest is not None:
                check_digest.update(raw)
            # Invalid UTF-8 remains detectable instead of silently altering evidence.
            yield line_no, start, position, raw.decode("utf-8", errors="surrogateescape")
    if observation is not None and position == observation.complete_end and check_digest is not None:
        if check_digest.hexdigest() != observation.prefix_digest(position):
            raise ValueError("source changed during read")


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
