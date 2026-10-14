"""Block digests of a session source: the scan, the stored table, and what a fast check may and may not vouch for.

Synthetic bytes only. A tiny block size keeps the files small; one test uses the real 1 MiB size.
"""
from __future__ import annotations

import hashlib
import os
import random

import pytest

from self_directing_mcp import session_events, source_blocks
from self_directing_mcp.session_events import (
    SourceChanged, capture_source, capture_source_fast, iter_raw_lines, source_scope,
)
from self_directing_mcp.source_blocks import BlockRecord, BlockTable, FastFallback, choose_blocks

BLOCK = 256


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch):
    monkeypatch.setattr(source_blocks, "BLOCK_SIZE", BLOCK)


def records(count, *, seed=1, widths=(5, 90)):
    """Newline terminated records of varying width (they straddle block boundaries in every way)."""
    rng = random.Random(seed)
    return [(f"{i:05d}:" + "x" * rng.randint(*widths) + "\n").encode() for i in range(count)]


def reference(data, offset, block=BLOCK):
    """The table of ``data[:offset]`` computed the obvious way, independent of the streaming scan."""
    prefix = data[:offset]
    nfull = offset // block
    digests = b"".join(hashlib.sha256(prefix[i * block:(i + 1) * block]).digest() for i in range(nfull))
    tail = hashlib.sha256(prefix[nfull * block:]).digest() if offset % block else None
    return BlockTable.build(offset, prefix.count(b"\n"), digests, tail, block)


def record_for(path, observation, *, at="2026-10-14T00:00:00+00:00"):
    stat = path.stat()
    table = observation.table_at(observation.complete_end)
    return BlockRecord("codex", "sid", str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, table,
                       "published", at, table.covered)


@pytest.fixture
def source(tmp_path):
    return tmp_path / "s.jsonl"


# ----------------------------------------------------------------------------------------- the full scan

@pytest.mark.parametrize("count", [0, 1, 3, 12, 40, 120])
@pytest.mark.parametrize("partial", [b"", b'{"unfinished"', b"x" * 700])
def test_full_scan_matches_an_independent_computation(source, count, partial):
    data = b"".join(records(count)) + partial
    source.write_bytes(data)
    observation = capture_source(source, sha=False, blocks=True)
    boundary = len(b"".join(records(count)))
    assert observation.length == len(data) and observation.complete_end == boundary
    assert observation.table_at(boundary) == reference(data, boundary)
    assert observation.table_at(len(data)) == reference(data, len(data))
    assert observation.cursor_hash(boundary, "blocks") == reference(data, boundary).hash


def test_requested_checkpoints_are_observed_without_rereading(source):
    data = b"".join(records(60))
    source.write_bytes(data)
    cut = data.index(b"\n", 3000) + 1
    observation = capture_source(source, checkpoints=(cut,), sha=True, blocks=True)
    assert observation.table_at(cut) == reference(data, cut)
    assert observation.prefix_digest(cut) == hashlib.sha256(data[:cut]).hexdigest()  # both digests, one pass


def test_an_unrequested_offset_is_computed_lazily_and_correctly(source):
    data = b"".join(records(60))
    source.write_bytes(data)
    observation = capture_source(source, sha=False, blocks=True)
    for offset in (0, BLOCK, BLOCK + 17, 5 * BLOCK - 1, observation.length):
        assert observation.table_at(offset) == reference(data, offset)


def test_a_capture_needs_some_digest():
    with pytest.raises(ValueError):
        capture_source(__file__, sha=False, blocks=False)


# ------------------------------------------------------------------------------- extension is not recomputation

@pytest.mark.parametrize("base", [0, 200, BLOCK, 3 * BLOCK + 40, 6 * BLOCK])
@pytest.mark.parametrize("extra", [0, 30, BLOCK, 2 * BLOCK + 70, 9 * BLOCK])
@pytest.mark.parametrize("partial", [b"", b'{"unfinished"'])
def test_a_fast_extension_equals_a_full_scan_of_the_same_bytes(source, base, extra, partial):
    rows = records(400, seed=base + extra)
    data0, position = b"", 0
    while len(data0) < base:
        data0 += rows[position]
        position += 1
    source.write_bytes(data0)
    full = capture_source(source, sha=False, blocks=True)
    stored = record_for(source, full)
    stat0 = source.stat()
    data1 = data0
    while len(data1) - len(data0) < extra:
        data1 += rows[position]
        position += 1
    source.write_bytes(data1 + partial)
    os.utime(source, ns=(stat0.st_atime_ns, stat0.st_mtime_ns))  # a same-size rewrite is not what this test is about
    fast = capture_source_fast(source, stored, recent=2, samples=2, rng=random.Random(5))
    scratch = capture_source(source, sha=False, blocks=True)
    assert fast.mode == "fast" and fast.length == scratch.length == len(data1) + len(partial)
    assert fast.complete_end == scratch.complete_end == len(data1)
    assert fast.table_at(fast.complete_end) == scratch.table_at(scratch.complete_end) == reference(data1, len(data1))
    assert fast.generation == scratch.generation


def test_the_extension_never_rereads_the_verified_prefix(source, hash_meter):
    """Cost is the checked blocks plus the appended bytes: bytes hashed stay far below the file size."""
    data = b"".join(records(20000))
    source.write_bytes(data)
    stored = record_for(source, capture_source(source, sha=False, blocks=True))
    source.write_bytes(data + b"".join(records(5, seed=9)))
    hash_meter.lap()
    fast = capture_source_fast(source, stored, recent=2, samples=3, rng=random.Random(2))
    assert fast.mode == "fast"
    # 2 + 3 compared blocks, the tail block and the appended records, the tail window (twice: baseline and the
    # bounded re-check) and the root over the stored digests (32 bytes per block; tiny blocks here): not the prefix.
    budget = 8 * BLOCK + 2 * session_events._TAIL_WINDOW + 2000 + 2 * len(stored.table.digests)
    assert hash_meter.lap() <= budget < len(data) / 2, (budget, len(data))


# ------------------------------------------------------------------------------------ what a fast check sees

def stored_for(source, *, rows=3000):
    data = b"".join(records(rows))
    source.write_bytes(data)
    full = capture_source(source, sha=False, blocks=True)
    return data, record_for(source, full)


def rewrite(source, data, index, *, restore_mtime=True):
    """Flip one byte at ``index`` keeping the size; optionally put the modification time back."""
    stat = source.stat()
    changed = bytearray(data)
    changed[index] ^= 0x01
    source.write_bytes(bytes(changed))
    if restore_mtime:
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_choose_blocks_covers_the_latest_and_a_random_sample():
    rng = random.Random(3)
    picked = choose_blocks(100, 4, 8, rng)
    assert picked == sorted(set(picked)) and len(picked) == 12
    assert {96, 97, 98, 99} <= set(picked) and all(0 <= i < 100 for i in picked)
    assert len(set(picked) - {96, 97, 98, 99}) == 8
    assert choose_blocks(3, 4, 8, rng) == [0, 1, 2]  # fewer blocks than the budget: all of them
    assert choose_blocks(0, 4, 8, rng) == [] and choose_blocks(10, 0, 0, rng) == []
    assert choose_blocks(10, 2, 100, rng) == list(range(10))
    # Fresh randomness per call: two calls with the system generator are (overwhelmingly) not the same sample.
    draws = {tuple(choose_blocks(1000, 0, 8, source_blocks.new_rng())) for _ in range(5)}
    assert len(draws) > 1


class _Sample:
    """A generator that 'randomly' picks exactly the given blocks."""

    def __init__(self, *picked):
        self.picked = list(picked)

    def sample(self, _population, count):
        return self.picked[:count]


@pytest.mark.parametrize("where", ["recent", "sampled", "tail"])
def test_a_fast_check_detects_a_same_size_rewrite_of_a_block_it_compares(source, where):
    data, stored = stored_for(source)
    table = stored.table
    block = {"recent": table.nfull - 1, "sampled": 2, "tail": table.nfull}[where]
    rewrite(source, data, block * BLOCK + 7)
    with pytest.raises(FastFallback) as info:
        capture_source_fast(source, stored, recent=1, samples=1 if where == "sampled" else 0, rng=_Sample(2))
    assert info.value.reason == ("tail_mismatch" if where == "tail" else "block_mismatch")
    assert info.value.block == block


def test_a_fast_check_does_not_see_a_rewrite_of_a_block_it_does_not_compare(source):
    """The stated limit of the fast path: only the compared blocks are vouched for."""
    data, stored = stored_for(source)
    rewrite(source, data, 3 * BLOCK + 5)  # a middle block, modification time restored
    fast = capture_source_fast(source, stored, recent=1, samples=0)
    assert fast.mode == "fast" and fast.fast_checked["sampled"] == [] and fast.fast_checked["recent"] == [stored.table.nfull - 1]
    # ...and a full scan does see it: the root of the prefix no longer matches the stored table.
    full = capture_source(source, sha=False, blocks=True)
    assert full.table_at(stored.table.covered).hash != stored.table.hash


def test_a_modified_same_size_file_is_settled_in_full_even_if_no_compared_block_changed(source):
    data, stored = stored_for(source)
    rewrite(source, data, 3 * BLOCK + 5, restore_mtime=False)
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    with pytest.raises(FastFallback) as info:
        capture_source_fast(source, stored, recent=1, samples=0)
    assert info.value.reason == "same_size_modified"


def test_a_shrunk_or_replaced_source_falls_back(source):
    data, stored = stored_for(source)
    source.write_bytes(data[:len(data) // 2])
    with pytest.raises(FastFallback) as info:
        capture_source_fast(source, stored, recent=1, samples=1)
    assert info.value.reason == "shrunk"
    replacement = source.with_name("replacement")
    replacement.write_bytes(data)
    os.replace(replacement, source)
    with pytest.raises(FastFallback) as info:
        capture_source_fast(source, stored, recent=1, samples=1)
    assert info.value.reason == "identity_changed"


def test_growth_with_a_rewrite_is_detected_when_the_block_is_compared(source):
    data, stored = stored_for(source)
    rewrite(source, data, stored.table.nfull * BLOCK + 1)
    with source.open("ab") as stream:
        stream.write(b"".join(records(10, seed=4)))
    with pytest.raises(FastFallback) as info:
        capture_source_fast(source, stored, recent=1, samples=0)
    assert info.value.reason == "tail_mismatch"


def test_a_table_that_does_not_add_up_is_never_used(source):
    data, stored = stored_for(source)
    table = stored.table
    damaged = [
        BlockTable(table.covered, table.lines + 1, table.digests, table.tail, table.root, table.block_size),
        BlockTable(table.covered, table.lines, table.digests[:-1], table.tail, table.root, table.block_size),
        BlockTable(table.covered, table.lines, table.digests, None, table.root, table.block_size),
        BlockTable(table.covered, table.lines, b"\0" * len(table.digests), table.tail, table.root, table.block_size),
    ]
    for bad in damaged:
        assert not bad.consistent()
        with pytest.raises(FastFallback):
            capture_source_fast(source, BlockRecord(**{**stored.__dict__, "table": bad}), recent=1, samples=1)
    assert table.consistent()


def test_a_record_is_bound_to_exactly_one_cursor_publication(source):
    _data, stored = stored_for(source)
    cursor = {"synced_at": "published", "path": str(source), "byte_offset": stored.table.covered,
              "prefix_hash": stored.table.hash}
    assert stored.bound_to(cursor)
    for field, value in (("synced_at", "other"), ("path", "/elsewhere"), ("byte_offset", 1),
                         ("prefix_hash", "0" * 64)):
        assert not stored.bound_to({**cursor, field: value}), field
    assert not stored.bound_to(None)


# -------------------------------------------------------- bytes consumed by a parser are the bytes that were scanned

@pytest.mark.parametrize("mode", ["full", "fast"])
def test_a_rewrite_between_capture_and_parse_is_caught_block_by_block(source, mode):
    base = b"".join(records(150))
    source.write_bytes(base)
    first = capture_source(source, sha=False, blocks=True)
    stored = record_for(source, first)
    appended = b"".join(records(40, seed=8))
    source.write_bytes(base + appended)
    observation = (capture_source_fast(source, stored, recent=1, samples=0) if mode == "fast"
                   else capture_source(source, checkpoints=(stored.table.covered,), sha=False, blocks=True))
    start = stored.table.covered if mode == "fast" else 0
    with source_scope(observation):
        assert sum(1 for _ in iter_raw_lines(source, start_byte=start)) == (40 if mode == "fast" else 190)
    stat = source.stat()
    changed = bytearray(base + appended)
    changed[len(base) + 20] ^= 0x02  # inside the records the parser consumes
    source.write_bytes(bytes(changed))
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with source_scope(observation), pytest.raises(SourceChanged):
        list(iter_raw_lines(source, start_byte=start))


def test_the_head_of_the_first_block_is_prefix_not_consumed_evidence(source):
    """Bytes before the cursor were verified by the scan and are not parsed again; a later rewrite of them is the
    next scan's business (it compares the stored table), not the parse's."""
    base = b"".join(records(150))
    source.write_bytes(base)
    stored = record_for(source, capture_source(source, sha=False, blocks=True))
    source.write_bytes(base + b"".join(records(40, seed=8)))
    fast = capture_source_fast(source, stored, recent=1, samples=0)
    stat = source.stat()
    changed = bytearray(source.read_bytes())
    changed[stored.table.tail_start + 1] ^= 0x02
    source.write_bytes(bytes(changed))
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with source_scope(fast):
        assert sum(1 for _ in iter_raw_lines(source, start_byte=stored.table.covered)) == 40
    assert capture_source(source, sha=False, blocks=True).table_at(stored.table.covered).hash != stored.table.hash


def test_line_numbers_continue_from_the_stored_table(source):
    base = b"".join(records(150))
    source.write_bytes(base)
    stored = record_for(source, capture_source(source, sha=False, blocks=True))
    source.write_bytes(base + b"".join(records(5, seed=8)))
    fast = capture_source_fast(source, stored, recent=1, samples=0)
    with source_scope(fast):
        numbers = [line for line, *_rest in iter_raw_lines(source, start_byte=stored.table.covered)]
    assert numbers == [151, 152, 153, 154, 155]


# ------------------------------------------------------------------------------------------ bounded later stages

def test_later_stages_never_rehash_the_prefix_and_the_pass_is_the_only_scan(source, hash_meter):
    data = b"".join(records(2000))
    source.write_bytes(data)
    observation = capture_source(source, sha=True, blocks=False)
    assert len(data) <= hash_meter.lap() <= len(data) + session_events._TAIL_WINDOW  # one pass (+ the tail window)
    for _ in range(5):
        assert observation.validate_prefix() and observation.validate_prefix(locked=True)
    assert hash_meter.lap() <= 10 * session_events._TAIL_WINDOW  # bounded re-checks, never the prefix
    assert observation.validate_prefix(full=True)
    assert hash_meter.lap() >= len(data)  # the explicit full re-read is still available


def test_a_moved_modification_time_without_growth_is_settled_by_a_reread(source):
    data = b"".join(records(3000))
    source.write_bytes(data)
    observation = capture_source(source)
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 3_000_000_000))  # touched, same bytes
    assert observation.validate_prefix() is True
    assert observation.validate_prefix() is True  # the new time is the baseline now
    changed = bytearray(data)
    changed[100] ^= 1  # outside the tail window; modification time moves
    source.write_bytes(bytes(changed))
    assert observation.validate_prefix(locked=True) is True  # the bounded variant cannot see it ...
    assert observation.validate_prefix() is False  # ... the re-read the moved time triggers can
    assert observation.validate_prefix(locked=True) is False  # and a mismatch does not heal


def test_the_documented_residual_window_is_exactly_this(source):
    """A rewrite outside the tail window that restores size and modification time, after the capture pass of the
    same request, is not seen by later stages of that request (a full re-read still sees it)."""
    data = b"".join(records(3000))
    source.write_bytes(data)
    observation = capture_source(source)
    stat = source.stat()
    changed = bytearray(data)
    changed[100] ^= 1
    source.write_bytes(bytes(changed))
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert observation.validate_prefix() is True and observation.validate_prefix(locked=True) is True
    assert observation.validate_prefix(full=True) is False


def test_the_real_block_size_is_one_mebibyte(monkeypatch):
    monkeypatch.undo()
    assert source_blocks.BLOCK_SIZE == 1024 * 1024
    assert source_blocks.HASH_TAG == "blocks1:" and source_blocks.is_block_hash("blocks1:" + "0" * 64)
    assert not source_blocks.is_block_hash("0" * 64)
