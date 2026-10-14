"""Reading zstd-compressed session files as streams: the decoder and source capture.

Synthetic data only. These tests need no engine, so they also run on interpreters where the full
dependency set is unavailable. Offsets of a compressed source address the decompressed stream.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import tracemalloc

import pytest

from self_directing_mcp import compressed_source
from self_directing_mcp.compressed_source import CompressedSourceError, open_source
from self_directing_mcp.session_events import capture_source, iter_raw_lines, source_scope

SID = "11111111-2222-3333-4444-555555555555"
OTHER = "99999999-8888-7777-6666-555555555555"


def message(text, role="assistant"):
    return {"type": "response_item", "payload": {"type": "message", "role": role, "content": text}}


def _importable(name):
    try:
        if name == "stdlib":
            from compression import zstd  # noqa: F401
        else:
            import zstandard  # noqa: F401
    except ImportError:
        return False
    return True


BACKENDS = [name for name in ("stdlib", "zstandard") if _importable(name)]
pytestmark = pytest.mark.skipif(not BACKENDS, reason="needs Python 3.14+ or the zstandard package")


def make_frame(data, *, content_size=True, level=None, checksum=False):
    """One zstd frame of ``data``. The compressor is whichever module is installed."""
    if _importable("stdlib"):
        from compression import zstd
        options = {zstd.CompressionParameter.checksum_flag: int(checksum)}
        if level is not None:
            options[zstd.CompressionParameter.compression_level] = level
        if not content_size:
            options[zstd.CompressionParameter.content_size_flag] = 0
        return zstd.compress(data, options=options)
    import zstandard
    return zstandard.ZstdCompressor(level=level or 3, write_content_size=content_size, write_checksum=checksum).compress(data)


def patch_declared_size(frame, size):
    """The same frame with another Frame_Content_Size in its header."""
    descriptor = frame[4]
    flag, single, dictionary = descriptor >> 6, (descriptor >> 5) & 1, descriptor & 3
    offset = 5 + (0 if single else 1) + (0, 1, 2, 4)[dictionary]
    width = {0: 1 if single else 0, 1: 2, 2: 4, 3: 8}[flag]
    assert width, "frame does not carry a content size"
    if flag == 1:
        size -= 256
    return frame[:offset] + struct.pack({1: "<B", 2: "<H", 4: "<I", 8: "<Q"}[width], size) + frame[offset + width:]


@pytest.fixture(params=BACKENDS)
def backend(request, monkeypatch):
    monkeypatch.setattr(compressed_source, "_PREFERENCE", (request.param,))
    return request.param


def jsonl(count, *, start=0, pad=0, header=True, tail=b""):
    records = ([{"type": "session_meta", "payload": {"id": SID}}] if header else [])
    records += [message(f"event {i} " + "x" * pad) for i in range(start, count)]
    return "".join(json.dumps(record) + "\n" for record in records).encode() + tail


def lines_of(path, **kwargs):
    with source_scope(capture_source(path)):
        return list(iter_raw_lines(path, **kwargs))


# ---------------------------------------------------------------- the stream reader

def test_stream_matches_the_plain_bytes_for_reads_lines_and_seeks(tmp_path, backend):
    data = jsonl(3000, pad=40)
    path = tmp_path / "s.jsonl.zst"
    path.write_bytes(make_frame(data))
    with open_source(path) as stream:
        assert stream.raw.declared == len(data) and stream.seekable()
        assert stream.read(100) == data[:100] and stream.tell() == 100
        stream.seek(len(data) - 50)  # forward: decode and discard
        assert stream.read() == data[-50:]
        stream.seek(7)  # backward: restart from the first byte
        assert stream.readline() == data[7:data.index(b"\n", 7) + 1]
        assert stream.seek(-10, 2) == len(data) - 10 and stream.read() == data[-10:]
        stream.seek(0)
        assert b"".join(stream.readlines()) == data
        with pytest.raises(ValueError):
            stream.seek(len(data) + 1)
    plain = tmp_path / "s.jsonl"
    plain.write_bytes(data)
    assert lines_of(path) == lines_of(plain)
    assert lines_of(path, start_byte=lines_of(plain)[500][2]) == lines_of(plain)[501:]


def test_capture_equals_a_plain_file_with_the_same_bytes(tmp_path, backend):
    data = jsonl(800, pad=30, tail=b'{"partial"')
    plain, packed = tmp_path / "a.jsonl", tmp_path / "a.jsonl.zst"
    plain.write_bytes(data)
    packed.write_bytes(make_frame(data))
    for checkpoints in ((), (777,)):
        first, second = capture_source(plain, checkpoints=checkpoints), capture_source(packed, checkpoints=checkpoints)
        assert (first.length, first.complete_end) == (second.length, second.complete_end) == (len(data), len(data) - 10)
        assert first.prefix_digest(first.complete_end) == second.prefix_digest(second.complete_end)
        assert first.prefix_digest(777) == second.prefix_digest(777) == hashlib.sha256(data[:777]).hexdigest()
        assert first.line_state(777)[1] == second.line_state(777)[1]
        assert second.validate() and second.validate_prefix() and second.validate_prefix(locked=True)
        assert second.current_size() == len(data)  # a decoded stream has no growing tail


def test_a_compressed_snapshot_is_immutable(tmp_path, backend):
    data = jsonl(300)
    path = tmp_path / "a.jsonl.zst"
    path.write_bytes(make_frame(data))
    observation = capture_source(path)
    with path.open("ab") as stream:
        stream.write(b"\0")  # growth of a compressed file is a rewrite, never pure append
    assert not observation.validate_prefix() and not observation.validate_prefix(locked=True)
    other = tmp_path / "b.jsonl.zst"
    other.write_bytes(make_frame(data))
    observation = capture_source(other)
    raw = bytearray(other.read_bytes())
    raw[len(raw) // 2] ^= 0x01  # same size, same stamps are not trusted: the compressed bytes are re-hashed
    stamp = other.stat()
    other.write_bytes(bytes(raw))
    os.utime(other, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert not observation.validate_prefix()


def test_corrupt_truncated_and_foreign_files_are_visible_errors(tmp_path, backend):
    data = jsonl(2000, pad=60)
    frame = make_frame(data)
    scrambled = frame[:12] + bytes(range(256)) * 12  # a valid header over a body that is not a zstd block sequence
    sealed = bytearray(make_frame(data, checksum=True))  # zstd verifies a checksum the writer chose to add
    sealed[len(sealed) // 2] ^= 0x01
    cases = {
        "zstd_corrupt": [scrambled, bytes(sealed), b"this is not zstd at all" * 20, frame + b"trailing junk"],
        "zstd_truncated": [frame[:-9], frame[:len(frame) // 2], frame[:3], b""],
        "zstd_unsupported_frame": [make_frame(data, content_size=False)],
        "zstd_size_mismatch": [frame + make_frame(data)],  # a second frame: more bytes than the frame declares
    }
    for code, blobs in cases.items():
        for blob in blobs:
            path = tmp_path / "bad.jsonl.zst"
            path.write_bytes(blob)
            with pytest.raises(CompressedSourceError) as caught:
                capture_source(path)
            assert caught.value.code == code, (code, caught.value.code, len(blob))


@pytest.mark.parametrize("delta", [-7, 7])
def test_a_declared_size_that_disagrees_with_the_data_is_an_error(tmp_path, backend, delta):
    data = jsonl(2000, pad=60)
    path = tmp_path / "lying.jsonl.zst"
    path.write_bytes(patch_declared_size(make_frame(data), len(data) + delta))
    with pytest.raises(CompressedSourceError) as caught:
        capture_source(path)
    assert caught.value.code in {"zstd_corrupt", "zstd_size_mismatch", "zstd_truncated"}


def test_a_session_over_the_limit_is_refused_before_any_decoding(tmp_path, backend, monkeypatch):
    data = jsonl(500)
    path = tmp_path / "big.jsonl.zst"
    path.write_bytes(make_frame(data))

    def no_decoder(*args, **kwargs):
        raise AssertionError("a refused frame must not start a decoder")

    monkeypatch.setattr(compressed_source._Reader, "__init__", no_decoder)
    with pytest.raises(CompressedSourceError) as caught:
        capture_source(path, max_bytes=len(data) - 1)
    assert caught.value.code == "zstd_too_large"
    monkeypatch.undo()
    assert capture_source(path, max_bytes=len(data)).length == len(data)


def test_the_decoder_streams_a_highly_compressible_stream_in_bounded_memory(tmp_path):
    line = (json.dumps(message("z" * 900)) + "\n").encode()
    data = line * 60000  # 50+ MB that compresses to almost nothing: the worst case for a buffering reader
    path = tmp_path / "zeros.jsonl.zst"
    path.write_bytes(make_frame(data))
    assert len(data) > 50 * 1024 * 1024 and path.stat().st_size < len(data) // 100
    tracemalloc.start()
    try:
        observation = capture_source(path, checkpoints=(len(data) // 2 // len(line) * len(line),))
        with source_scope(observation):
            count = sum(1 for _ in iter_raw_lines(path, start_byte=len(line) * 30000))
        with open_source(path) as stream:
            stream.seek(len(data) - 1)
            stream.seek(5)
            stream.read(10)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert observation.length == len(data) and count == 30000
    print(f"zstd decoder: {len(data) / 1e6:.1f} MB decoded from {path.stat().st_size / 1e3:.0f} KB, peak traced {peak / 1e6:.2f} MB")
    assert peak < 8 * 1024 * 1024, peak  # the stdlib reader (3.14) buffers a little more than zstandard
