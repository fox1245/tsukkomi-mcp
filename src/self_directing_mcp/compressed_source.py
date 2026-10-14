"""Streaming access to zstd-compressed session files (Codex ``rollout-*.jsonl.zst``).

A compressed session is an immutable snapshot of a plain rollout. Its byte offsets address the DECOMPRESSED
stream, so cursors, anchors and prefix hashes stay valid when a plain rollout is compressed in place (or the
reverse) and the bytes are the same.

Nothing is ever materialized. ``open_source`` returns a buffered, seekable reader over a bounded decoder:
forward seeks decode and discard in fixed blocks, backward seeks restart from the first byte. A source is
refused, with a visible ``CompressedSourceError`` code, unless it is exactly one frame that declares its
content size: the declared size bounds the work before any byte is decoded (``max_bytes``), and a decoder
that produces more or fewer bytes, ends early, or finds corruption is an error, never a short read. The
decoder's window is capped at zstd's own default (128 MiB).

Backend: the standard library ``compression.zstd`` on Python 3.14+, otherwise the ``zstandard`` package.
``.jsonl`` sessions never import either.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

from self_directing_mcp.request_control import check_request

SUFFIX = ".zst"
# A compressed file can expand by orders of magnitude, so its decompressed size is capped (a plain file needs
# no cap: its size is what is on disk). 4 GiB is about ten times the largest observed Codex session (hundreds
# of MB); a pass over it stays within tens of seconds of CPU and every pass honors the request deadline.
DEFAULT_MAX_DECOMPRESSED_BYTES = 4 * 1024**3
WINDOW_LOG_MAX = 27  # zstd's default decoder limit: a 128 MiB window
_MAGIC = b"\x28\xb5\x2f\xfd"
_HEADER_PEEK = 64  # a frame header is at most 18 bytes
_BUFFER = 256 * 1024
_SKIP_BLOCK = 1024 * 1024
_TAIL_WINDOW = 64 * 1024
_UNKNOWN_SIZE = 2**63  # zstd reports an unknown content size as 2**64 - 1

# Tried in this order. Tests may narrow it to exercise one backend.
_PREFERENCE = ("stdlib", "zstandard")


class CompressedSourceError(ValueError):
    """A compressed session file cannot be read faithfully. ``code`` names the coverage issue / error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def is_compressed(path) -> bool:
    return Path(path).suffix.lower() == SUFFIX


class _Stdlib:
    name = "stdlib"

    def __init__(self, module):
        self._zstd = module
        self.errors = (module.ZstdError, EOFError)

    def frame(self, header):
        info = self._zstd.get_frame_info(header)
        return info.decompressed_size, None, info.dictionary_id

    def reader(self, source):
        options = {self._zstd.DecompressionParameter.window_log_max: WINDOW_LOG_MAX}
        return self._zstd.ZstdFile(source, "rb", options=options)


class _Zstandard:
    name = "zstandard"

    def __init__(self, module):
        self._zstd = module
        self.errors = (module.ZstdError, EOFError)

    def frame(self, header):
        params = self._zstd.get_frame_parameters(header)
        size = params.content_size
        return (None if size < 0 or size >= _UNKNOWN_SIZE else size), params.window_size, params.dict_id

    def reader(self, source):
        decompressor = self._zstd.ZstdDecompressor(max_window_size=1 << WINDOW_LOG_MAX)
        return decompressor.stream_reader(source, read_across_frames=True, closefd=False)


def _backend():
    for name in _PREFERENCE:
        try:
            if name == "stdlib":
                from compression import zstd
                return _Stdlib(zstd)
            if name == "zstandard":
                import zstandard
                return _Zstandard(zstandard)
        except ImportError:
            continue
    raise CompressedSourceError(
        "zstd_unavailable",
        "reading .zst sessions needs Python 3.14+ (compression.zstd) or the 'zstandard' package; "
        "install it, or decompress the file next to the original name")


class _Tap:
    """Pass-through that hashes every compressed byte the decoder consumes and keeps the last window."""

    def __init__(self, source):
        self._source = source
        self.digest = hashlib.sha256()
        self.count = 0
        self._tail = bytearray()

    def read(self, size=-1):
        data = self._source.read(size)
        self.digest.update(data)
        self.count += len(data)
        self._tail += data
        if len(self._tail) > _TAIL_WINDOW:
            del self._tail[:len(self._tail) - _TAIL_WINDOW]
        return data

    def tail(self) -> bytes:
        return bytes(self._tail)


class _Reader(io.RawIOBase):
    """Decompressed bytes of one single-frame file. Offsets are decompressed offsets."""

    def __init__(self, file, backend, declared, tap=None):
        super().__init__()
        self._file, self._backend, self.declared, self.tap = file, backend, declared, tap
        self._source = tap if tap is not None else file
        self._pos = 0
        self._complete = False
        self._decoder = None  # close() must work if the decoder cannot be built
        self._decoder = backend.reader(self._source)

    def readable(self):
        return True

    def seekable(self):
        return True

    def fileno(self):
        return self._file.fileno()

    def tell(self):
        return self._pos

    def _fail(self, exc):
        code = "zstd_truncated" if isinstance(exc, EOFError) else "zstd_corrupt"
        return CompressedSourceError(code, f"zstd data cannot be decoded ({code[5:]}): {exc}")

    def readinto(self, buffer):
        view = memoryview(buffer).cast("B")
        want = min(len(view), self.declared - self._pos)
        if want <= 0:
            if len(view):
                self.verify_complete()
            return 0
        try:
            count = self._decoder.readinto(view[:want])
        except self._backend.errors as exc:
            raise self._fail(exc) from None
        if not count:
            raise CompressedSourceError(
                "zstd_truncated", f"decompressed stream ended at {self._pos} of {self.declared} declared bytes")
        self._pos += count
        return count

    def verify_complete(self):
        """Once all declared bytes are out: the frame must end here (this also checks its checksum, if any)."""
        if self._complete:
            return
        if self._pos != self.declared:
            raise CompressedSourceError("zstd_truncated", "decompressed stream was not read to its declared size")
        try:
            extra = self._decoder.read(1)
        except self._backend.errors as exc:
            raise self._fail(exc) from None
        if extra:
            raise CompressedSourceError(
                "zstd_size_mismatch", f"decompressed data exceeds the {self.declared} bytes the frame declares")
        self._complete = True

    def drain(self):
        """Consume the compressed file to its end so a tap has seen every byte."""
        while self._source.read(_SKIP_BLOCK):
            check_request()

    def _restart(self):
        if self.tap is not None:
            raise ValueError("a single-pass compressed read cannot move backwards")
        self._decoder.close()
        self._file.seek(0)
        self._pos, self._complete = 0, False
        self._decoder = self._backend.reader(self._source)

    def seek(self, offset, whence=io.SEEK_SET):
        target = {io.SEEK_SET: offset, io.SEEK_CUR: self._pos + offset, io.SEEK_END: self.declared + offset}[whence]
        if target < 0:
            raise ValueError("negative seek position")
        if target > self.declared:
            raise ValueError("seek beyond the end of the decompressed stream")
        if target < self._pos:
            self._restart()
        scratch = bytearray(min(_SKIP_BLOCK, max(target - self._pos, 1)))
        while self._pos < target:
            check_request()
            if not self.readinto(memoryview(scratch)[:min(len(scratch), target - self._pos)]):
                break
        return self._pos

    def close(self):
        if not self.closed:
            try:
                if self._decoder is not None:
                    self._decoder.close()
            finally:
                try:
                    self._file.close()
                finally:
                    super().close()


def _admit(header, backend, max_bytes):
    """Validate one frame header; return its declared decompressed size."""
    if len(header) < len(_MAGIC) and _MAGIC.startswith(header):
        raise CompressedSourceError("zstd_truncated", "compressed file ends inside the frame header")
    if header[:len(_MAGIC)] != _MAGIC:
        raise CompressedSourceError("zstd_corrupt", "file does not start with a zstd frame")
    try:
        declared, window, dictionary = backend.frame(header)
    except backend.errors as exc:
        if len(header) < _HEADER_PEEK:  # a short file cannot hold a whole frame header
            raise CompressedSourceError("zstd_truncated", "compressed file ends inside the frame header") from None
        raise CompressedSourceError("zstd_corrupt", f"the zstd frame header is invalid: {exc}") from None
    if declared is None:
        raise CompressedSourceError(
            "zstd_unsupported_frame", "the zstd frame does not declare its content size, so its length cannot be verified")
    if dictionary:
        raise CompressedSourceError("zstd_unsupported_frame", "the zstd frame requires a dictionary")
    if window is not None and window > 1 << WINDOW_LOG_MAX:
        raise CompressedSourceError("zstd_unsupported_frame", f"the zstd frame needs a window larger than {1 << WINDOW_LOG_MAX} bytes")
    if max_bytes is not None and declared > max_bytes:
        raise CompressedSourceError(
            "zstd_too_large", f"the session declares {declared} decompressed bytes; the limit is {max_bytes} "
                              "(SELF_DIRECT_MAX_DECOMPRESSED_SESSION_BYTES)")
    return declared


def open_source(path, *, max_bytes=DEFAULT_MAX_DECOMPRESSED_BYTES, tap=False):
    """Binary stream of a session source: the file itself, or the decompressed bytes of a ``.zst`` file.

    ``max_bytes`` bounds a compressed source's declared size (``None``: no bound, for readers of a few lines).
    ``tap`` (compressed only) makes ``stream.raw.tap`` hash the compressed bytes consumed; that pass is forward
    only. ``stream.raw.declared`` is the decompressed length.
    """
    path = Path(path)
    if not is_compressed(path):
        return path.open("rb")
    backend = _backend()
    file = path.open("rb")
    try:
        declared = _admit(file.read(_HEADER_PEEK), backend, max_bytes)
        file.seek(0)
        raw = _Reader(file, backend, declared, _Tap(file) if tap else None)
    except BaseException:
        file.close()
        raise
    return io.BufferedReader(raw, buffer_size=_BUFFER)
