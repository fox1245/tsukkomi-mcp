"""The two pieces slicing is made of, on their own: a capture cut short at a record boundary, and the soft stop of the parse.

* ``capture_source(window=(floor, cap))`` / ``capture_source_fast(cap=...)`` observe a plain file only up to the end of the
  last record inside ``floor + cap`` (at least one whole record) and behave exactly like a capture of a file that was that
  short.
* ``iter_raw_lines`` under a ``slice_scope`` ends at a record boundary once the minimum progress is made and the soft
  deadline has passed, or a cap is reached, and verifies what it consumed against the capture before it lets the prefix
  be published (a byte changed between capture and parse is an error even when the stop falls inside a block).
"""
from __future__ import annotations

import hashlib
import json

import pytest

from self_directing_mcp import source_blocks
from self_directing_mcp.session_events import (
    SourceChanged, capture_source, capture_source_fast, iter_raw_lines, source_scope,
)
from self_directing_mcp.slice_budget import SliceBudget, slice_scope
from self_directing_mcp.source_blocks import BlockRecord

BLOCK = source_blocks.BLOCK_SIZE


def record(number, size=60):
    return json.dumps({"n": number, "pad": "x" * size}).encode() + b"\n"


def make(path, count, size=60, tail=b""):
    data = b"".join(record(n, size) for n in range(count)) + tail
    path.write_bytes(data)
    return data


def boundaries(data):
    out, position = [], 0
    while True:
        found = data.find(b"\n", position)
        if found < 0:
            return out
        position = found + 1
        out.append(position)


# --- the capture cut short -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("cap", [1, 70, 500, 4096, 5000, 12345])
def test_a_cut_capture_ends_at_the_last_record_inside_the_bound_and_holds_one_record_at_least(tmp_path, cap):
    data = make(tmp_path / "s.jsonl", 400, tail=b'{"unfinished"')
    ends = boundaries(data)
    observation = capture_source(tmp_path / "s.jsonl", window=(0, cap), sha=True, blocks=True)
    inside = [e for e in ends if e <= cap]
    expected = inside[-1] if inside else ends[0]            # no record ends inside the bound: the first record is the slice
    assert observation.length == observation.complete_end == expected
    assert observation.limited_from == len(data) and observation.target_end == len(data)
    assert observation.prefix_digest(expected) == hashlib.sha256(data[:expected]).hexdigest()


def test_a_cut_capture_behaves_exactly_like_the_capture_of_a_file_that_was_that_short(tmp_path):
    data = make(tmp_path / "long.jsonl", 40_000)                    # about 2.7 MB: two complete blocks and a partial one
    cut = [e for e in boundaries(data) if e <= 1_700_000][-1]
    (tmp_path / "short.jsonl").write_bytes(data[:cut])
    cut_capture = capture_source(tmp_path / "long.jsonl", window=(0, 1_700_000), sha=True, blocks=True)
    short_capture = capture_source(tmp_path / "short.jsonl", sha=True, blocks=True)
    assert (cut_capture.length, cut_capture.complete_end) == (short_capture.length, short_capture.complete_end) == (cut, cut)
    assert cut_capture.prefix_digest(cut) == short_capture.prefix_digest(cut)
    assert cut_capture.table_at(cut) == short_capture.table_at(cut)
    assert cut_capture.blocks.digests == short_capture.blocks.digests
    assert cut_capture._tail_digest == short_capture._tail_digest and short_capture.limited_from is None
    assert cut_capture.validate_prefix() and cut_capture.validate_prefix(locked=True)     # the file is "longer": growth


def test_a_file_that_fits_or_whose_rest_is_one_unfinished_record_is_captured_whole(tmp_path):
    data = make(tmp_path / "s.jsonl", 100, tail=b'{"unfinished record that goes on and on')
    for window in [(0, len(data)), (0, len(data) + 5000), (len(boundaries(data)) and boundaries(data)[-1], 10)]:
        observation = capture_source(tmp_path / "s.jsonl", window=window, sha=True, blocks=True)
        assert observation.length == len(data) and observation.limited_from is None
        assert observation.complete_end == boundaries(data)[-1] and observation.target_end == observation.complete_end


def test_a_record_longer_than_the_bound_is_the_slice(tmp_path):
    big = json.dumps({"pad": "y" * 5000}).encode() + b"\n"
    data = record(0) + record(1) + big + record(3) + record(4)
    (tmp_path / "s.jsonl").write_bytes(data)
    first = len(record(0))
    observation = capture_source(tmp_path / "s.jsonl", window=(first, 100), sha=True, blocks=True)
    assert observation.complete_end == first + len(record(1))      # the bound (100 bytes) ends inside the big record
    again = capture_source(tmp_path / "s.jsonl", window=(first + len(record(1)), 100), sha=True, blocks=True)
    assert again.complete_end == first + len(record(1)) + len(big)  # no newline inside the bound: up to the first one after it


def test_the_cap_never_cuts_a_record_in_two_and_slices_tile_the_file(tmp_path):
    data = make(tmp_path / "s.jsonl", 600)
    position, ends = 0, []
    while position < len(data):
        observation = capture_source(tmp_path / "s.jsonl", window=(position, 3333), sha=True, blocks=True)
        assert observation.complete_end > position and data[observation.complete_end - 1:observation.complete_end] == b"\n"
        position = observation.complete_end
        ends.append(position)
    assert position == len(data) and len(ends) >= 10


def test_a_cut_fast_capture_reads_at_most_the_cap_past_the_stored_table(tmp_path):
    data = make(tmp_path / "s.jsonl", 40_000)
    cursor = [e for e in boundaries(data) if e <= 1_500_000][-1]
    full = capture_source(tmp_path / "s.jsonl", checkpoints=(cursor,), sha=False, blocks=True)
    # The stored table of a cursor at ``cursor`` (what a previous slice published).
    table = full.table_at(cursor)
    stat = (tmp_path / "s.jsonl").stat()
    stored = BlockRecord("codex", "sid", str(tmp_path / "s.jsonl"), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
                         table, "t", None, None)
    fast = capture_source_fast(tmp_path / "s.jsonl", stored, recent=2, samples=2, cap=200_000)
    expected = [e for e in boundaries(data) if e <= cursor + 200_000][-1]
    assert fast.mode == "fast" and fast.length == fast.complete_end == expected and fast.limited_from == len(data)
    reference = capture_source(tmp_path / "s.jsonl", window=(cursor, 200_000), checkpoints=(cursor,), sha=False, blocks=True)
    assert fast.table_at(expected) == reference.table_at(expected)
    assert fast.validate_prefix()


def test_a_changed_byte_inside_a_cut_capture_is_still_detected(tmp_path):
    data = make(tmp_path / "s.jsonl", 3000)
    observation = capture_source(tmp_path / "s.jsonl", window=(0, 60_000), sha=True, blocks=True)
    assert observation.validate_prefix()
    tampered = bytearray(data)
    tampered[observation.length - 10] ^= 1                       # inside the tail window of the capture
    (tmp_path / "s.jsonl").write_bytes(bytes(tampered))
    assert observation.validate_prefix() is False


# --- the soft stop of the parse ---------------------------------------------------------------------------------------

def run(path, observation, budget, start=0):
    """Consume ``iter_raw_lines`` under a source scope and a slice scope; returns (records, position)."""
    records, position = 0, start
    with source_scope(observation), slice_scope(budget):
        for _line, _start, end, _raw in iter_raw_lines(path, start_byte=start):
            records, position = records + 1, end
    return records, position


class Fixed:
    """A clock that always says the same time and counts how often it was read."""

    def __init__(self, now):
        self.now, self.reads = now, 0

    def __call__(self):
        self.reads += 1
        return self.now


def test_a_soft_deadline_that_already_passed_still_makes_the_minimum_progress(tmp_path):
    data = make(tmp_path / "s.jsonl", 30_000)                               # 100-byte records, 3 MB
    observation = capture_source(tmp_path / "s.jsonl", sha=True, blocks=True)
    clock = Fixed(10.0)
    budget = SliceBudget(0, soft_deadline=1.0, clock=clock)                  # the production minimums, long past due
    records, position = run(tmp_path / "s.jsonl", observation, budget)
    assert budget.stopped_by == "soft_deadline" and position < observation.complete_end
    assert position >= 1 << 20 and records >= 256                           # a whole MiB and 256 records, no less
    assert position - (1 << 20) < 64 * 100 + 100                            # and no more than one look (64 records) beyond
    assert data[position - 1:position] == b"\n" and budget.records == records and budget.consumed == position


def test_the_minimum_is_records_and_bytes_not_either(tmp_path):
    make(tmp_path / "s.jsonl", 600, size=20_000)                            # 20 KB records: 1 MiB is 52 records
    observation = capture_source(tmp_path / "s.jsonl", sha=True, blocks=True)
    budget = SliceBudget(0, soft_deadline=0.0, clock=Fixed(5.0))
    records, _position = run(tmp_path / "s.jsonl", observation, budget)
    assert budget.stopped_by == "soft_deadline" and 256 <= records < 256 + 64


def test_caps_stop_a_slice_whatever_the_minimums_say(tmp_path):
    make(tmp_path / "s.jsonl", 1000)
    observation = capture_source(tmp_path / "s.jsonl", sha=True, blocks=True)
    budget = SliceBudget(0, soft_deadline=None, max_records=7)
    assert run(tmp_path / "s.jsonl", observation, budget)[0] == 7 and budget.stopped_by == "record_cap"
    budget = SliceBudget(0, max_bytes=1000)
    records, position = run(tmp_path / "s.jsonl", observation, budget)
    assert budget.stopped_by == "slice_cap" and position >= 1000 and position - len(record(0)) < 1000
    budget = SliceBudget(0, max_records=1)                                  # a slice always holds a whole record
    assert run(tmp_path / "s.jsonl", observation, budget)[0] == 1
    with pytest.raises(ValueError):
        SliceBudget(0, max_records=0)


def test_without_a_stop_the_parse_reads_to_the_end_and_says_nothing_stopped_it(tmp_path):
    make(tmp_path / "s.jsonl", 300)
    observation = capture_source(tmp_path / "s.jsonl", sha=True, blocks=True)
    budget = SliceBudget(0, soft_deadline=None)
    records, position = run(tmp_path / "s.jsonl", observation, budget)
    assert records == 300 and position == observation.complete_end and budget.stopped_by is None


def test_the_clock_is_read_every_64_records_and_only_after_the_minimum(tmp_path):
    make(tmp_path / "s.jsonl", 20_000)
    observation = capture_source(tmp_path / "s.jsonl", sha=True, blocks=True)
    clock = Fixed(0.0)
    budget = SliceBudget(0, soft_deadline=1e9, min_records=10, min_bytes=0, clock=clock)
    records, _ = run(tmp_path / "s.jsonl", observation, budget)
    assert records == 20_000 and clock.reads <= 20_000 // 64 + 1 and clock.reads > 20_000 // 64 - 3
    quiet = Fixed(0.0)
    budget = SliceBudget(0, soft_deadline=1e9, clock=quiet)                 # production minimums: 1 MiB first
    run(tmp_path / "s.jsonl", observation, budget)
    assert 0 < quiet.reads < clock.reads


def test_only_the_iterator_that_starts_at_the_slice_offset_is_bounded(tmp_path):
    data = make(tmp_path / "s.jsonl", 300)
    observation = capture_source(tmp_path / "s.jsonl", sha=True, blocks=True)
    budget = SliceBudget(boundaries(data)[99], max_records=3)
    with source_scope(observation), slice_scope(budget):
        assert len(list(iter_raw_lines(tmp_path / "s.jsonl"))) == 300      # a probe from byte 0: not the slice
        assert len(list(iter_raw_lines(tmp_path / "s.jsonl", start_byte=budget.start_byte))) == 3


def test_a_compressed_source_is_never_cut_short(tmp_path):
    from test_zstd_stream import BACKENDS, make_frame
    if not BACKENDS:
        pytest.skip("needs Python 3.14+ or the zstandard package")
    data = make(tmp_path / "plain.jsonl", 800)
    packed = tmp_path / "s.jsonl.zst"
    packed.write_bytes(make_frame(data))
    observation = capture_source(packed, sha=True, blocks=False)
    budget = SliceBudget(0, max_records=5)
    assert run(packed, observation, budget)[0] == 800 and budget.stopped_by is None


# --- what a stop inside a block verifies -------------------------------------------------------------------------------

def stops_inside_second_block(data):
    """How many records to consume so that the parse stops about 30,000 bytes into the second 1 MiB block."""
    return len([end for end in boundaries(data) if end <= BLOCK + 30_000])


def tamper(path, offset):
    data = bytearray(path.read_bytes())
    data[offset] = (data[offset] + 1) % 256 if data[offset] != 0x0A else 0x20
    path.write_bytes(bytes(data))


@pytest.mark.parametrize("sha", [False, True])
def test_a_byte_changed_between_capture_and_parse_is_caught_when_the_stop_is_inside_a_block(tmp_path, sha):
    data = make(tmp_path / "s.jsonl", 30_000)                              # three blocks
    observation = capture_source(tmp_path / "s.jsonl", sha=sha, blocks=not sha)
    count = stops_inside_second_block(data)
    # A byte of the part of the second block that the parse consumes, written after the capture and before the parse.
    tamper(tmp_path / "s.jsonl", BLOCK + 5000)
    with pytest.raises(SourceChanged):
        run(tmp_path / "s.jsonl", observation, SliceBudget(0, max_records=count))


@pytest.mark.parametrize("sha", [False, True])
def test_a_change_in_the_rest_of_the_block_after_the_stop_is_caught_too(tmp_path, sha):
    """The whole block is compared with what the capture saw: the unparsed rest of it must not have moved either."""
    data = make(tmp_path / "s.jsonl", 30_000)
    observation = capture_source(tmp_path / "s.jsonl", sha=sha, blocks=not sha)
    count = stops_inside_second_block(data)
    tamper(tmp_path / "s.jsonl", BLOCK + 900_000)                         # far behind the stop, still in the second block
    with pytest.raises(SourceChanged):
        run(tmp_path / "s.jsonl", observation, SliceBudget(0, max_records=count))


@pytest.mark.parametrize("sha", [False, True])
def test_an_untouched_source_is_verified_at_a_stop_inside_a_block(tmp_path, sha):
    data = make(tmp_path / "s.jsonl", 30_000)
    observation = capture_source(tmp_path / "s.jsonl", sha=sha, blocks=not sha)
    count = stops_inside_second_block(data)
    budget = SliceBudget(0, max_records=count)
    records, position = run(tmp_path / "s.jsonl", observation, budget)
    assert records == count and budget.stopped_by == "record_cap" and BLOCK < position < 2 * BLOCK
    expected = hashlib.sha256(data[:position]).hexdigest()
    if sha:
        assert observation.cursor_hash(position, "sha") == expected          # a verified checkpoint now: no second read
    else:
        assert observation.cursor_hash(position, "blocks") == source_blocks.BlockTable.build(
            position, data[:position].count(b"\n"),
            b"".join(hashlib.sha256(data[i * BLOCK:(i + 1) * BLOCK]).digest() for i in range(position // BLOCK)),
            hashlib.sha256(data[position // BLOCK * BLOCK:position]).digest()).hash


def test_a_stop_in_the_last_partial_block_is_compared_with_the_capture_s_digest_of_its_records(tmp_path):
    data = make(tmp_path / "s.jsonl", 14_000, tail=b'{"unfinished')
    observation = capture_source(tmp_path / "s.jsonl", sha=False, blocks=True)
    tamper(tmp_path / "s.jsonl", len(data) - 200)                         # inside the last block, after the stop point
    assert len(data) > BLOCK + 80_000                                    # (the last block, partial, holds the stop point)
    with pytest.raises(SourceChanged):
        run(tmp_path / "s.jsonl", observation, SliceBudget(0, max_records=len([e for e in boundaries(data) if e <= BLOCK + 50_000])))


def test_the_look_schedule(tmp_path):
    budget = SliceBudget(0, soft_deadline=5.0, clock=lambda: 0.0)
    assert budget.look(0, 0) == (None, 256)                                  # nothing to decide before the minimum
    assert budget.look(256, 1000) == (None, 320)                             # records are there, bytes are not: look again soon
    assert budget.look(400, 2 << 20) == (None, 464)                          # both: the clock is read every 64 records
    late = SliceBudget(0, soft_deadline=-1.0, clock=lambda: 0.0)
    assert late.look(400, 2 << 20)[0] == "soft_deadline" and late.look(100, 5 << 20)[0] is None
    assert SliceBudget(0, max_records=5).look(5, 0)[0] == "record_cap"
    assert SliceBudget(0, max_bytes=10).look(3, 10)[0] == "slice_cap" and SliceBudget(0, max_bytes=10).look(0, 0)[0] is None
