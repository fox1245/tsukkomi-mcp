"""The spool file of a staged re-index (roundtable 8, PR3): ``reindex_spool`` on its own, with synthetic rows.

* What a slice leaves: rows, then a trailer; the progress is the last valid trailer, a torn tail is cut at every possible
  byte, a slice is never staged twice, a spool of another session or version is never touched.
* What binds it: every part of the binding that changes discards the spool (and nothing but the spool).
* What protects it: a private directory and file, no symlinks, no other user's, one writer at a time, a disk guard, and a
  frame that does not verify is never published.
"""
from __future__ import annotations

import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from self_directing_mcp import reindex_spool as rs
from self_directing_mcp.reindex_spool import ReindexSpool, SpoolBinding, SpoolBusy, SpoolError, SpoolUnsafe

POSIX = os.name != "nt"


def binding(**changes) -> SpoolBinding:
    base = SpoolBinding(provider="codex", session_id="session-1", target_version=4, base_version=3, base_stamp="t0",
                        hash_kind="blocks", dev=7, ino=9, path="/synthetic/rollout.jsonl", size=5000, head_length=100,
                        head_hash="head-hash")
    return replace(base, **changes)


def rows(first, count, session="session-1", provider="codex"):
    return [(f"{provider}:{session}:{n * 10}:0:{n:016x}", provider, session, n * 10, 0,
             json.dumps({"chunk_id": f"c{n}", "text": f"event {n} é \"quoted\" \\ backslash"})) for n in range(first, first + count)]


def head_hash(length):
    return "head-hash"


def cursor_hash(offset):
    return f"prefix-{offset}"


def open_spool(tmp_path, bound=None, *, length=5000, head=head_hash, cursor=cursor_hash):
    return ReindexSpool.open(tmp_path, bound or binding(), source_length=length, head_hash=head, cursor_hash=cursor)


def stage(spool, *slices):
    """Stage ``(first row, rows, end offset)`` slices one after the other."""
    start = spool.staged_through
    for first, count, end in slices:
        assert spool.append_slice(rows(first, count), end, cursor_hash(end), start_offset=start)
        start = end


@pytest.fixture(autouse=True)
def forgetful():
    rs.forget_verified()
    yield
    rs.forget_verified()


def test_a_spool_is_created_by_its_first_slice_and_not_before(tmp_path):
    spool = open_spool(tmp_path)
    assert not spool.valid and spool.staged_through == 0 and spool.discard_reason is None
    assert not list(tmp_path.glob("reindex/*"))                      # opening a spool that does not exist makes no file
    stage(spool, (0, 5, 50))
    assert spool.valid and spool.path.exists() and spool.staged_through == 50 and spool.rows == 5 and spool.slices == 1
    spool.close()


def test_the_name_has_no_session_id_in_it_and_carries_the_target_version(tmp_path):
    spool = open_spool(tmp_path, binding(session_id="a/b c:d?e"))
    stage(spool, (0, 1, 10))
    assert spool.path.parent == tmp_path / "reindex" and spool.path.name.endswith(".4.spool")
    assert spool.path.name.startswith("codex-") and "a/b" not in spool.path.name and " " not in spool.path.name
    spool.close()


def test_a_reopened_spool_resumes_where_the_last_trailer_ends(tmp_path):
    spool = open_spool(tmp_path)
    stage(spool, (0, 5, 50), (5, 3, 80))
    spool.close()
    again = open_spool(tmp_path)
    assert again.valid and (again.staged_through, again.rows, again.slices) == (80, 8, 2)
    assert again.prefix_hash == "prefix-80" and again.discard_reason is None
    stage(again, (8, 2, 100))
    again.close()
    third = open_spool(tmp_path)
    assert (third.staged_through, third.rows, third.slices) == (100, 10, 3)
    third.close()


def test_the_row_source_has_the_interface_of_a_row_spool(tmp_path):
    spool = open_spool(tmp_path)
    spool.append_slice(rows(0, 4), 40, "prefix-40", start_offset=0, flags={"x": (True, False)},
                       tallies={"y": [("legacy_unlinked_results", 1), ("media_placeholders", 2)]})
    spool.append_slice(rows(4, 2), 60, "prefix-60", start_offset=40)
    source = spool.as_row_source()
    assert source.count == len(source) == 6
    assert source.ids == {row[0] for row in rows(0, 6)}
    assert source.flags == {"x": (True, False)}
    assert source.tallies == {"y": [("legacy_unlinked_results", 1), ("media_placeholders", 2)]}
    streamed = [row for batch in source.batches(4) for row in batch]
    assert streamed == rows(0, 6)                                     # the same tuples a RowSpool replays
    assert [len(batch) for batch in source.batches(4)] == [4, 2]
    source.close()
    spool.close()


def test_a_row_spool_staged_and_read_back_is_what_a_row_spool_replays(tmp_path):
    """The spool takes the text of a ``RowSpool`` as it is and brings back its rows, ids, flags, tallies (and, for OMP, the
    earliest byte of each tool call) exactly: the commit cannot tell the two row sources apart."""
    from self_directing_mcp.index.ingest import RowSpool
    from self_directing_mcp.session_events import make_chunk
    source = RowSpool(track_calls=True)
    chunks = [
        make_chunk("omp", "session-1", "tool_call", "[tool_call] bash", {"call_id": "c1", "tool_name": "bash"}, 1, 10, 50),
        make_chunk("omp", "session-1", "meta", "[parse_error] bad", {"parse_error": True}, 2, 50, 60),
        make_chunk("omp", "session-1", "meta", "[omp_event:x] caf\u00e9 \"q\" \\", {"unsupported_event": True}, 3, 60, 99),
        make_chunk("omp", "session-1", "tool_call", "[tool_call] bash", {"call_id": "c1", "tool_name": "bash"}, 4, 99, 120),
        make_chunk("omp", "session-1", "tool_result", "[tool_result:bash] ok", {"call_id": "c1", "success": True}, 5, 120, 150),
    ]
    for chunk in chunks:
        source.append(chunk)
    replayed = [row for batch in source.batches() for row in batch]
    spool = open_spool(tmp_path, binding(provider="omp", session_id="session-1"))
    assert spool.append_slice(source, 150, "prefix-150", start_offset=0)
    staged = spool.as_row_source()
    assert [row for batch in staged.batches(2) for row in batch] == replayed
    assert (staged.count, staged.ids, staged.flags) == (source.count, source.ids, source.flags)
    assert staged.tallies == source.tallies == {} and source.calls == {("c1", "bash"): 10}
    spool.close()
    again = open_spool(tmp_path, binding(provider="omp", session_id="session-1"))
    assert again.calls == {("c1", "bash"): 10}
    again.close()
    source.close()


def test_the_blocks_of_a_row_spool_are_its_lines(tmp_path):
    from self_directing_mcp.index.ingest import RowSpool
    from self_directing_mcp.session_events import make_chunk
    source = RowSpool()
    for n in range(50):
        source.append(make_chunk("codex", "session-1", "turn", f"event {n} " + "x" * 200, {"role": "user"}, n, n * 300, n * 300 + 299))
    blocks = list(source.blocks(limit=1000))
    assert len(blocks) > 5 and sum(count for _data, count in blocks) == 50
    assert all(isinstance(data, bytes) and data.endswith(b"\n") for data, _count in blocks)
    assert b"".join(data for data, _count in blocks).decode().splitlines() == \
        [json.dumps(list(row)) for batch in source.batches() for row in batch]
    source.close()


def test_the_index_of_a_slice_is_written_in_bounded_parts(tmp_path, monkeypatch):
    """The ids of a slice are never one document of the size of the slice: a commit decodes them a part at a time."""
    monkeypatch.setattr(rs, "INDEX_CHUNK", 7)
    spool = open_spool(tmp_path)
    flags = {f"codex:session-1:{n * 10}:0:{n:016x}": (True, False) for n in range(0, 40, 3)}
    tallies = {f"codex:session-1:{n * 10}:0:{n:016x}": [("k", n)] for n in range(0, 40, 2)}
    spool.append_slice(rows(0, 40), 400, "prefix-400", start_offset=0, flags=flags, tallies=tallies)
    assert len(spool._index_frames) == 6                              # 40 ids in parts of 7
    source = spool.as_row_source()
    assert source.ids == {row[0] for row in rows(0, 40)} and source.flags == flags and source.tallies == tallies
    spool.close()
    again = open_spool(tmp_path)
    assert len(again._index_frames) == 6 and again.as_row_source().ids == source.ids
    again.close()


def test_the_row_source_streams_a_frame_at_a_time(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "BLOCK_BYTES", 600)                       # a few rows per frame
    spool = open_spool(tmp_path)
    stage(spool, (0, 60, 600))
    source = spool.as_row_source()
    reads = []
    original = rs.StagedRows._frame
    monkeypatch.setattr(rs.StagedRows, "_frame", staticmethod(lambda stream, offset: reads.append(offset) or original(stream, offset)))
    frames = len(spool._row_frames)
    assert frames >= 10
    batches = source.batches(2)
    first = next(batches)
    assert len(first) == 2 and len(reads) == 1                        # not the whole file for the first rows
    assert sum(len(batch) for batch in batches) + len(first) == 60 and len(reads) == frames
    spool.close()


def test_a_slice_that_ends_where_the_staged_range_ends_is_a_rerun_and_writes_nothing(tmp_path):
    spool = open_spool(tmp_path)
    stage(spool, (0, 5, 50))
    size = spool.path.stat().st_size
    assert spool.append_slice(rows(0, 5), 50, "prefix-50", start_offset=0) is False
    assert spool.append_slice(rows(0, 3), 30, "prefix-30", start_offset=0) is False      # a shorter one is a rerun as well
    assert spool.path.stat().st_size == size and spool.rows == 5 and spool.slices == 1
    spool.close()
    again = open_spool(tmp_path)
    assert [row for batch in again.as_row_source().batches() for row in batch] == rows(0, 5)    # no duplicate, no gap
    again.close()


def test_a_slice_that_does_not_continue_the_range_is_refused(tmp_path):
    spool = open_spool(tmp_path)
    stage(spool, (0, 5, 50))
    with pytest.raises(SpoolError):
        spool.append_slice(rows(9, 2), 90, "prefix-90", start_offset=70)                  # a gap
    with pytest.raises(SpoolError):
        spool.append_slice(rows(3, 2), 90, "prefix-90", start_offset=30)                  # an overlap
    assert spool.staged_through == 50 and spool.rows == 5
    spool.close()


def test_a_torn_tail_is_cut_at_every_possible_byte(tmp_path):
    """The file is cut after each byte: what was staged before the cut slice is intact, a partial slice is never visible, a
    file cut inside its header is no spool at all."""
    spool = open_spool(tmp_path)
    stage(spool, (0, 5, 50))
    one = spool._end
    stage(spool, (5, 3, 80))
    two = spool._end
    spool.close()
    path, data = spool.path, spool.path.read_bytes()
    assert len(data) == two
    header = rs._FRAME.size + rs._FRAME.unpack(data[:rs._FRAME.size])[0]
    for cut in range(len(data) + 1):
        path.write_bytes(data[:cut])
        rs.forget_verified()
        reopened = open_spool(tmp_path)
        if cut < header:
            assert not reopened.valid and reopened.staged_through == 0
            assert reopened.discard_reason == (None if cut == 0 else "unreadable") and not path.exists()
        else:
            staged, kept = (80, two) if cut == two else (50, one) if cut >= one else (0, header)
            assert reopened.valid and reopened.staged_through == staged
            assert reopened.rows == {0: 0, 50: 5, 80: 8}[staged] and reopened.discard_reason is None
            assert path.stat().st_size == kept            # the torn tail is gone, the next slice goes after the last trailer
        reopened.close()


def test_the_slice_after_a_torn_tail_follows_the_last_trailer(tmp_path):
    spool = open_spool(tmp_path)
    stage(spool, (0, 5, 50))
    spool.close()
    with open(spool.path, "ab") as stream:          # a crash in the middle of the next slice: whole frames, no trailer
        stream.write(b"\x08\x00\x00\x00\x00\x00\x00\x00garbage!")
    rs.forget_verified()
    again = open_spool(tmp_path)
    assert again.valid and again.staged_through == 50
    stage(again, (5, 2, 70))
    again.close()
    rs.forget_verified()
    final = open_spool(tmp_path)
    assert (final.staged_through, final.rows, final.slices) == (70, 7, 2)
    assert [row for batch in final.as_row_source().batches() for row in batch] == rows(0, 7)
    final.close()


def test_rows_written_without_their_trailer_are_not_staged(tmp_path, monkeypatch):
    """The process stops after the row frames and before the trailer: those rows were never acknowledged."""
    spool = open_spool(tmp_path)
    stage(spool, (0, 5, 50))

    class Stop(Exception):
        pass

    def stop(self, trailer):
        raise Stop

    monkeypatch.setattr(ReindexSpool, "_write_trailer", stop)
    with pytest.raises(Stop):
        spool.append_slice(rows(5, 4), 90, "prefix-90", start_offset=50)
    assert spool.staged_through == 50 and spool.rows == 5
    assert spool.path.stat().st_size == spool._end                     # the unfinished slice was cut away at once
    spool.close()
    rs.forget_verified()
    assert open_spool(tmp_path).staged_through == 50


def test_a_frame_that_does_not_verify_before_the_last_trailer_discards_the_spool(tmp_path):
    spool = open_spool(tmp_path)
    stage(spool, (0, 40, 400), (40, 5, 450))
    spool.close()
    data = bytearray(spool.path.read_bytes())
    data[len(data) // 3] ^= 0x01                                       # one flipped bit inside the first slice's rows
    spool.path.write_bytes(bytes(data))
    rs.forget_verified()
    again = open_spool(tmp_path)
    assert not again.valid and again.discard_reason == "corrupt" and again.staged_through == 0
    assert not spool.path.exists()


def test_a_corrupt_frame_met_while_streaming_stops_the_commit(tmp_path):
    """Even a spool this process trusts is verified frame by frame when its rows are replayed."""
    spool = open_spool(tmp_path)
    stage(spool, (0, 40, 400))
    source = spool.as_row_source()
    data = bytearray(spool.path.read_bytes())
    data[len(data) // 2] ^= 0x01
    spool.path.write_bytes(bytes(data))
    with pytest.raises(SpoolError):
        list(source.batches())
    spool.close()


BINDING_CHANGES = [
    ("format", dict(hash_kind="sha")),                  # the prefix hashes are of another form
    ("base_version", dict(base_version=2)),             # the index it replaces is of another parser version
    ("orphan", dict(base_version=4)),                   # ... or already of the target: another process finished
    ("index_changed", dict(base_stamp="t1")),           # the cursor it replaces has been rewritten
    ("source_identity", dict(ino=10)),                  # another file
]


@pytest.mark.parametrize("reason,change", BINDING_CHANGES, ids=[item[0] for item in BINDING_CHANGES])
def test_a_spool_whose_binding_no_longer_holds_is_discarded(tmp_path, reason, change):
    spool = open_spool(tmp_path)
    stage(spool, (0, 5, 50))
    spool.close()
    other = replace(binding(), **change)
    again = ReindexSpool.open(tmp_path, other, source_length=5000, head_hash=head_hash, cursor_hash=cursor_hash)
    assert not again.valid and again.discard_reason == reason and again.staged_through == 0
    assert not spool.path.exists()


def test_a_source_that_ends_before_the_staged_range_discards_the_spool(tmp_path):
    spool = open_spool(tmp_path)
    stage(spool, (0, 5, 50))
    spool.close()
    again = open_spool(tmp_path, length=49)
    assert again.discard_reason == "source_shrunk" and not again.valid


def test_a_changed_head_discards_the_spool(tmp_path):
    spool = open_spool(tmp_path)
    stage(spool, (0, 5, 50))
    spool.close()
    again = open_spool(tmp_path, head=lambda length: "another head")
    assert again.discard_reason == "source_changed" and not again.valid


def test_a_changed_prefix_discards_the_spool(tmp_path):
    spool = open_spool(tmp_path)
    stage(spool, (0, 5, 50))
    spool.close()
    again = open_spool(tmp_path, cursor=lambda offset: "another prefix")
    assert again.discard_reason == "prefix_changed" and not again.valid and not spool.path.exists()


def test_the_path_names_the_source_where_the_platform_has_no_inodes(tmp_path):
    spool = open_spool(tmp_path, binding(ino=0, dev=0))
    stage(spool, (0, 5, 50))
    spool.close()
    kept = open_spool(tmp_path, binding(ino=0, dev=0))
    assert kept.valid
    kept.close()
    moved = open_spool(tmp_path, binding(ino=0, dev=0, path="/synthetic/elsewhere.jsonl"))
    assert moved.discard_reason == "source_identity"


def test_a_spool_for_another_target_version_is_removed_and_never_read(tmp_path):
    old = open_spool(tmp_path, binding(target_version=4))
    stage(old, (0, 5, 50))
    old.close()
    new = open_spool(tmp_path, binding(target_version=5, base_version=3))
    assert not new.valid and new.discard_reason == "target_version" and not old.path.exists()


def test_only_the_spools_of_the_session_are_touched(tmp_path):
    mine, other = open_spool(tmp_path), open_spool(tmp_path, binding(session_id="session-2"))
    other_provider = open_spool(tmp_path, binding(provider="omp"))
    for spool in (mine, other, other_provider):
        stage(spool, (0, 2, 20))
        spool.close()
    assert ReindexSpool.discard_session(tmp_path, "codex", "session-1") == 1
    assert not mine.path.exists() and other.path.exists() and other_provider.path.exists()
    assert ReindexSpool.discard_session(tmp_path, "codex", "session-1") == 0
    assert ReindexSpool.discard_session(tmp_path / "nowhere", "codex", "session-1") == 0


def test_peek_reads_the_progress_and_changes_nothing(tmp_path):
    assert ReindexSpool.peek(tmp_path, "codex", "session-1", 4) is None
    spool = open_spool(tmp_path)
    stage(spool, (0, 5, 50), (5, 3, 80))
    spool.close()
    before = spool.path.read_bytes()
    assert ReindexSpool.peek(tmp_path, "codex", "session-1", 4) == 80
    assert ReindexSpool.peek(tmp_path, "codex", "session-1", 5) is None
    assert ReindexSpool.peek(tmp_path, "codex", "session-2", 4) is None
    assert spool.path.read_bytes() == before


def test_an_empty_file_is_no_spool_and_is_removed_without_a_reason(tmp_path):
    directory = rs.ensure_directory(tmp_path)
    path = directory / rs.spool_name("codex", "session-1", 4)
    path.write_bytes(b"")
    spool = open_spool(tmp_path)
    assert not spool.valid and spool.discard_reason is None and not path.exists()


def test_a_header_that_cannot_be_read_discards_the_file(tmp_path):
    directory = rs.ensure_directory(tmp_path)
    path = directory / rs.spool_name("codex", "session-1", 4)
    path.write_bytes(b"this is not a spool at all, but longer than a frame header")
    spool = open_spool(tmp_path)
    assert not spool.valid and spool.discard_reason == "unreadable" and not path.exists()


def test_there_is_no_slice_without_progress_and_no_file_without_a_slice(tmp_path):
    spool = open_spool(tmp_path)
    assert spool.append_slice([], 0, "prefix-0") is False             # nothing to stage is not a slice
    spool.close()
    assert not spool.path.exists()


def test_a_slice_without_rows_is_still_progress(tmp_path):
    """A stretch of blank lines parses to no row; the range it covers is staged all the same."""
    spool = open_spool(tmp_path)
    assert spool.append_slice([], 90, "prefix-90", start_offset=0)
    assert (spool.staged_through, spool.rows, spool.slices) == (90, 0, 1)
    spool.close()
    again = open_spool(tmp_path)
    assert again.staged_through == 90 and list(again.as_row_source().batches()) == []
    again.close()


def test_the_calls_and_dependencies_a_next_slice_needs_survive(tmp_path):
    spool = open_spool(tmp_path)
    spool.append_slice(rows(0, 2), 20, "prefix-20", start_offset=0, calls={("call-1", "bash"): 10, ("call-2", "read"): 5},
                       deps={"v": 1, "first": 3, "steps": {"7": "aa", "9": "bb"}}, receipt_digest="digest-1")
    spool.append_slice(rows(2, 2), 40, "prefix-40", start_offset=20, calls={("call-1", "bash"): 4, ("call-3", "bash"): 30},
                       deps={"v": 1, "first": 2, "steps": {"11": "cc"}}, receipt_digest="digest-2")
    spool.close()
    again = open_spool(tmp_path)
    assert again.calls == {("call-1", "bash"): 4, ("call-2", "read"): 5, ("call-3", "bash"): 30}    # the earliest of each
    assert again.deps_steps == {7: "aa", 9: "bb", 11: "cc"} and again.deps_first == 2
    assert again.receipt_digest == "digest-2"
    again.close()


# --- privacy ---------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not POSIX, reason="POSIX permission bits")
def test_the_directory_is_0700_and_the_file_0600(tmp_path):
    spool = open_spool(tmp_path)
    stage(spool, (0, 3, 30))
    assert stat.S_IMODE((tmp_path / "reindex").stat().st_mode) == 0o700
    assert stat.S_IMODE(spool.path.stat().st_mode) == 0o600
    spool.close()


@pytest.mark.skipif(not POSIX, reason="POSIX permission bits")
def test_a_loose_directory_or_file_is_tightened(tmp_path):
    spool = open_spool(tmp_path)
    stage(spool, (0, 3, 30))
    spool.close()
    os.chmod(tmp_path / "reindex", 0o755)
    os.chmod(spool.path, 0o644)
    again = open_spool(tmp_path)
    assert again.valid
    assert stat.S_IMODE((tmp_path / "reindex").stat().st_mode) == 0o700
    assert stat.S_IMODE(spool.path.stat().st_mode) == 0o600
    again.close()


@pytest.mark.skipif(not POSIX, reason="needs POSIX permissions")
def test_a_umask_does_not_open_the_spool_to_others(tmp_path):
    old = os.umask(0)
    try:
        spool = open_spool(tmp_path)
        stage(spool, (0, 3, 30))
        assert stat.S_IMODE(spool.path.stat().st_mode) == 0o600 and stat.S_IMODE((tmp_path / "reindex").stat().st_mode) == 0o700
        spool.close()
    finally:
        os.umask(old)


def test_a_symlinked_directory_is_refused(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    index = tmp_path / "index"
    index.mkdir()
    (index / "reindex").symlink_to(target, target_is_directory=True)
    with pytest.raises(SpoolUnsafe):
        open_spool(index)
    assert not list(target.iterdir())


def test_a_symlinked_spool_is_refused_and_not_followed(tmp_path):
    spool = open_spool(tmp_path)
    stage(spool, (0, 3, 30))
    spool.close()
    secret = tmp_path / "secret.txt"
    secret.write_text("do not touch")
    spool.path.unlink()
    spool.path.symlink_to(secret)
    with pytest.raises(SpoolUnsafe):
        open_spool(tmp_path)
    assert secret.read_text() == "do not touch"


@pytest.mark.skipif(not POSIX, reason="ownership is a POSIX notion")
def test_a_directory_of_another_user_is_refused(tmp_path, monkeypatch):
    rs.ensure_directory(tmp_path)
    monkeypatch.setattr(os, "getuid", lambda: os.stat(tmp_path / "reindex").st_uid + 1)
    with pytest.raises(SpoolUnsafe):
        open_spool(tmp_path)


# --- one writer, and room ---------------------------------------------------------------------------------------------

@pytest.mark.skipif(not POSIX, reason="the lock of Windows is not exercised here")
def test_a_second_writer_is_refused_while_the_first_holds_the_spool(tmp_path):
    first = open_spool(tmp_path)
    stage(first, (0, 3, 30))
    with pytest.raises(SpoolBusy):
        open_spool(tmp_path)
    first.close()
    second = open_spool(tmp_path)
    assert second.valid and second.staged_through == 30
    second.close()


@pytest.mark.skipif(not POSIX, reason="the lock of Windows is not exercised here")
def test_a_writer_that_races_the_creation_is_refused(tmp_path):
    first, second = open_spool(tmp_path), open_spool(tmp_path)
    stage(first, (0, 3, 30))
    with pytest.raises(SpoolBusy):
        stage(second, (0, 3, 30))
    first.close()
    second.close()


def test_the_disk_guard_wants_twice_the_measured_ratio(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "free_space", lambda path: 9 * 1000 - 1)
    assert not rs.has_room(tmp_path, 1000)
    monkeypatch.setattr(rs, "free_space", lambda path: 9 * 1000)
    assert rs.has_room(tmp_path, 1000)
    monkeypatch.setattr(rs, "free_space", lambda path: (_ for _ in ()).throw(OSError("no such volume")))
    assert rs.has_room(tmp_path, 10 ** 12)          # a volume that cannot be asked is no reason to refuse


def test_a_write_that_fails_leaves_the_staged_range_as_it_was(tmp_path, monkeypatch):
    spool = open_spool(tmp_path)
    stage(spool, (0, 5, 50))

    def full_disk(self):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ReindexSpool, "_sync", full_disk)
    with pytest.raises(OSError):
        spool.append_slice(rows(5, 3), 80, "prefix-80", start_offset=50)
    monkeypatch.undo()
    assert spool.staged_through == 50 and spool.path.stat().st_size == spool._end
    spool.close()
    assert open_spool(tmp_path).staged_through == 50


def test_this_process_does_not_read_a_spool_in_full_twice_when_nothing_changed(tmp_path, monkeypatch):
    spool = open_spool(tmp_path)
    stage(spool, (0, 40, 400))
    spool.close()
    modes = []
    original = rs._scan
    monkeypatch.setattr(rs, "_scan", lambda path, **kwargs: modes.append(kwargs["verify"]) or original(path, **kwargs))
    first = open_spool(tmp_path)               # written by this process: its own writes were checked
    first.close()
    rs.forget_verified()
    second = open_spool(tmp_path)              # a new process knows nothing: every frame is verified
    second.close()
    third = open_spool(tmp_path)
    third.close()
    assert modes == [False, True, False]
