"""zstd-compressed Codex rollouts (``rollout-*.jsonl.zst``) through the engine: indexing, transitions, errors.

Synthetic sessions only. A plain rollout and its compressed twin are the same session: offsets, anchors, prefix
hashes and cached vectors address the decompressed stream, so Codex compressing a rollout in place (or a resume
decompressing it again) is not a new session. Stream-level behavior is in ``test_zstd_stream.py``.
"""
from __future__ import annotations

import hashlib
import json
import tracemalloc

import pytest

from self_directing_mcp import compressed_source
from self_directing_mcp.codex import discover
from self_directing_mcp.codex import parse as codex_parser
from self_directing_mcp.codex_hooks import _evidence_stamp
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from test_reliability import SID, RecordingEmbedder, call, message, result
from test_zstd_stream import BACKENDS, OTHER, jsonl, make_frame

pytestmark = pytest.mark.skipif(not BACKENDS, reason="needs Python 3.14+ or the zstandard package")


# ---------------------------------------------------------------- the engine

class World:
    def __init__(self, tmp_path, monkeypatch, **settings):
        for name in ("SELF_DIRECT_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER",
                     "SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR"):
            monkeypatch.delenv(name, raising=False)
        self.root = tmp_path / "sessions"
        self.root.mkdir(parents=True, exist_ok=True)
        self.engine = SelfDirectEngine(Settings(codex_sessions_dir=self.root, index_dir=tmp_path / "index",
                                                use_fake_embedder=True, _env_file=None, **settings))
        self.parses = []
        original = codex_parser.parse_session_chunks

        def counted(*args, **kwargs):
            self.parses.append(kwargs.get("start_byte", 0))
            return original(*args, **kwargs)

        monkeypatch.setattr(codex_parser, "parse_session_chunks", counted)

    name = f"rollout-2026-10-14T01-02-03-{SID}.jsonl"

    def plain(self, data, name=None):
        path = self.root / (name or self.name)
        path.write_bytes(data)
        return path.resolve()

    def packed(self, data, name=None, **options):
        path = self.root / ((name or self.name) + ".zst")
        path.write_bytes(make_frame(data, **options))
        return path.resolve()

    def sync(self, path=None, **kwargs):
        return self.engine.sync_session(session_id=SID, path=None if path is None else str(path), embed=False, **kwargs)

    def sync_authority_only(self, path):
        """Source and metadata work only (what an audit refresh does): no sparse or dense repair."""
        return self.engine._sync_session(session_id=SID, path=str(path), embed=False, derived=False)

    def cursor(self):
        return self.engine.store.cursor_record(SID, "codex")

    def events(self):
        return [(c.chunk_id, c.byte_start, c.byte_end, c.text) for c in self.engine.store.list_chunks(SID, "codex")]

    def contract(self, regex="curl"):
        self.engine.upsert_contracts([{"id": "no-curl", "type": "must_not", "scope": "tool_call", "regex": regex}])


@pytest.fixture
def world(tmp_path, monkeypatch):
    built = World(tmp_path, monkeypatch)
    try:
        yield built
    finally:
        built.engine.close()


def test_a_compressed_session_indexes_exactly_like_its_plain_twin(tmp_path, monkeypatch):
    data = jsonl(40, pad=25) + (json.dumps(call("curl http://example.test", "c1")) + "\n").encode()
    plain_world = World(tmp_path / "a", monkeypatch)  # separate index and root
    packed_world = World(tmp_path / "b", monkeypatch)
    try:
        info_plain = plain_world.sync(plain_world.plain(data))
        info_packed = packed_world.sync(packed_world.packed(data))
        assert info_plain["ok"] and info_packed["ok"]
        assert plain_world.events() == packed_world.events()  # same ids, anchors, offsets (decompressed) and text
        coverage = info_packed["coverage"]
        assert coverage["complete"] and coverage["issues"] == [] and coverage["pending_bytes"] == 0
        assert coverage["file_size"] == len(data) == info_packed["byte_offset"]
        assert info_packed["new_chunks"] == info_plain["new_chunks"] > 40 and info_packed["parser_version"] == info_plain["parser_version"]
        for each in (plain_world, packed_world):
            each.contract()
            audit = each.engine.audit_session(SID)
            assert audit["verdict"] == "violation" and audit["coverage"]["unread_bytes"] == 0
            assert audit["coverage"]["judged_through_byte"] == len(data) and audit["coverage"]["complete"]
        again = packed_world.sync()  # by session id: found through the compressed file's header
        assert again["ok"] and again["new_chunks"] == 0 and not again["reindexed"]
        assert again["path"] == str(packed_world.root / (packed_world.name + ".zst")) == packed_world.cursor()["path"]
    finally:
        plain_world.engine.close()
        packed_world.engine.close()


def test_session_id_comes_from_the_header_not_the_file_name(world):
    path = world.packed(jsonl(5), name=f"rollout-2026-10-14T01-02-03-{OTHER}_{SID}.jsonl")
    info = world.engine.sync_session(path=str(path), embed=False)  # no session id given
    assert info["ok"] and info["session_id"] == SID and info["new_chunks"] > 0
    assert world.sync()["ok"]  # and by id, through the header scan


def test_an_unterminated_final_record_stays_unread_never_half_parsed(world):
    tail = b'{"type": "response_item", "payload": {"type": "PARTIALTAIL'
    data = jsonl(10, tail=tail)
    info = world.sync(world.packed(data))
    assert info["ok"] and info["coverage"]["pending_bytes"] == len(tail) and "unread_bytes" in info["coverage"]["issues"]
    assert not any("PARTIALTAIL" in chunk[3] for chunk in world.events())
    assert info["byte_offset"] == len(data) - len(tail) and info["coverage"]["complete"] is False


def test_recompressing_the_same_content_is_no_change(world):
    data = jsonl(30)
    path = world.packed(data, level=1)
    first = world.sync(path)
    ids = world.events()
    path.write_bytes(make_frame(data, level=19))  # different compressed bytes, the same stream
    info = world.sync(path)
    assert info["ok"] and not info["reindexed"] and info["new_chunks"] == 0 and info["coverage"]["complete"]
    assert world.events() == ids and world.parses == [0] and first["new_chunks"] > 0


def test_rewriting_a_compressed_session_is_handled_like_any_rewrite(world):
    path = world.packed(jsonl(30))
    world.sync(path)
    path.write_bytes(make_frame(jsonl(30).replace(b"event 3 ", b"event 9 ")))  # an earlier record changed
    info = world.sync(path)
    assert info["ok"] and info["reindexed"] and info["reindex_reason"] == "source_changed"
    assert world.cursor()["byte_offset"] == len(jsonl(30).replace(b"event 3 ", b"event 9 "))
    path.write_bytes(make_frame(jsonl(45).replace(b"event 3 ", b"event 9 ")))  # re-compressed with more records
    before = world.parses[:]
    info = world.sync(path)
    assert info["ok"] and not info["reindexed"] and world.parses[len(before):] == [len(jsonl(30))]  # parsed from the cursor on


def test_corrupt_truncated_and_unavailable_sources_are_reported_and_index_nothing(world, monkeypatch):
    data = jsonl(300, pad=50)
    frame = make_frame(data)
    for label, blob, code in (("corrupt", frame[:12] + bytes(range(256)) * 12, "zstd_corrupt"),
                              ("truncated", frame[:-20], "zstd_truncated")):
        path = world.root / f"{label}-{world.name}.zst"
        path.write_bytes(blob)
        info = world.engine.sync_session(path=str(path), embed=False)
        assert info["ok"] is False and info["error"] == code and info["provider"] == "codex"
        assert world.cursor() is None and world.parses == []
        audit = world.engine.audit_session(SID, path=str(path))
        assert audit["verdict"] == "unknown" and audit["coverage"]["complete"] is False
        assert code in audit["coverage"]["issues"]
    path = world.packed(data)
    with monkeypatch.context() as module_missing:
        module_missing.setattr(compressed_source, "_PREFERENCE", ())
        info = world.sync(path)
        assert info["ok"] is False and info["error"] == "zstd_unavailable"
        plain = world.plain(data, name="other-" + world.name)
        assert world.sync(plain)["ok"]  # plain sessions never need the module
    assert world.sync(path)["ok"]


def test_a_session_over_the_configured_limit_is_not_decoded(tmp_path, monkeypatch):
    data = jsonl(200)
    built = World(tmp_path, monkeypatch, max_decompressed_session_bytes=len(data) - 1)
    try:
        info = built.sync(built.packed(data))
        assert info["ok"] is False and info["error"] == "zstd_too_large" and built.cursor() is None
    finally:
        built.engine.close()
    assert Settings(_env_file=None).max_decompressed_session_bytes == compressed_source.DEFAULT_MAX_DECOMPRESSED_BYTES
    monkeypatch.setenv("SELF_DIRECT_MAX_DECOMPRESSED_SESSION_BYTES", "12345")
    assert Settings(_env_file=None).max_decompressed_session_bytes == 12345


def test_an_indexed_compressed_source_that_goes_bad_is_reported_by_coverage(world):
    path = world.packed(jsonl(40))
    world.sync(path)
    world.contract()
    assert world.engine.audit_session(SID)["verdict"] == "clean"
    path.write_bytes(path.read_bytes()[:-5])
    audit = world.engine.audit_session(SID)
    assert audit["verdict"] == "unknown" and audit["coverage"]["complete"] is False
    assert any(issue.startswith("zstd_") for issue in audit["coverage"]["issues"]), audit["coverage"]["issues"]
    status = world.engine.audit_status(SID)["session"]
    assert "zstd_truncated" in status["issues"] and status["complete"] is False


# ---------------------------------------------------------------- transitions between the two forms

def check_same_session(world, before, info):
    assert info["ok"] and not info["reindexed"] and info["reindex_reason"] is None
    assert "ambiguous" not in json.dumps(info) and "source_changed" not in info["coverage"]["issues"]
    assert [event[:3] for event in world.events()[:len(before)]] == [event[:3] for event in before]
    assert {row[0] for row in world.engine.store._conn.execute("SELECT DISTINCT session_id FROM session_cursors")} == {SID}


def test_plain_replaced_by_compressed_keeps_ids_anchors_and_vectors(world, monkeypatch):
    data = jsonl(25, pad=20)
    plain = world.plain(data)
    world.engine.ensure_ready()
    embedder = RecordingEmbedder()
    world.engine.embedder = embedder
    first = world.engine.sync_session(session_id=SID, path=str(plain))
    assert first["ok"] and first["embedded"] > 0 and world.cursor()["path"] == str(plain)
    before, ids = world.events(), sorted(world.engine.store.chunk_ids(SID, "codex"))
    assert world.engine.dense.present(ids) == set(ids)
    embedded = len(embedder.documents)
    state = world.engine.store.state_token(SID, "codex")
    packed = world.packed(data)
    plain.unlink()  # Codex compresses in place: the compressed file replaces the plain one
    info = world.engine.sync_session(session_id=SID, embed=True)  # found by session id
    check_same_session(world, before, info)
    assert world.parses == [0] and info["new_chunks"] == 0 and world.events() == before
    assert info["embedded"] == 0 and info["dense_restored"] == 0 and info["embedding_pending"] == 0
    assert len(embedder.documents) == embedded  # no vector was recomputed
    assert world.engine.dense.present(ids) == set(ids)
    assert world.cursor()["path"] == str(packed) and world.cursor()["byte_offset"] == len(data)
    assert info["coverage"]["complete"] and "alias_of" not in info
    assert world.engine.store.state_token(SID, "codex") != state  # only the cursor path moved
    assert world.engine.store.chunk_count(SID, "codex") == len(before)


def test_plain_that_grew_before_it_was_compressed_continues_from_the_cursor(world):
    plain = world.plain(jsonl(20))
    world.sync(plain)
    before, size = world.events(), len(jsonl(20))
    grown = jsonl(20) + jsonl(32, start=20, header=False)
    packed = world.packed(grown)
    plain.unlink()
    info = world.sync()
    check_same_session(world, before, info)
    assert world.parses == [0, size] and info["new_chunks"] > 0
    assert world.cursor() == {**world.cursor(), "path": str(packed), "byte_offset": len(grown)}
    assert world.events()[:len(before)] == before


def test_compressed_decompressed_back_and_appended_continues_the_same_session(world):
    data = jsonl(20)
    packed = world.packed(data)
    world.sync(packed)
    before = world.events()
    plain = world.plain(data + jsonl(30, start=20, header=False))  # `codex resume` decompresses and appends
    packed.unlink()
    info = world.sync()
    check_same_session(world, before, info)
    assert world.parses == [0, len(data)] and info["new_chunks"] > 0 and world.cursor()["path"] == str(plain)
    assert world.sync()["new_chunks"] == 0 and world.parses == [0, len(data)]


def test_identical_plain_and_compressed_copies_follow_the_duplicate_source_rules(world):
    data = jsonl(15)
    plain, packed = world.plain(data), world.packed(data)
    assert discover.find_session_by_id(world.root, SID) == plain
    states = []
    for path in (plain, packed, plain, packed):
        info = world.sync(path)
        assert info["ok"] and not info["reindexed"] and info["coverage"]["complete"]
        states.append(world.engine.store.state_token(SID, "codex"))
        if path == packed:
            assert info["alias_of"] == str(plain) and info["coverage"]["indexed_path"] == str(plain)
    assert world.parses == [0] and len(set(states)) == 1 and world.cursor()["path"] == str(plain)


def test_compressed_first_then_an_identical_plain_copy(world):
    data = jsonl(15)
    packed, plain = world.packed(data), world.plain(data)
    assert world.sync(packed)["ok"]
    info = world.sync(plain)
    assert info["ok"] and not info["reindexed"] and info["alias_of"] == str(packed) and world.parses == [0]
    assert info["coverage"]["complete"] and world.cursor()["path"] == str(packed)


def test_a_shorter_compressed_prefix_of_the_indexed_plain_file_is_reported_as_a_prefix(world):
    short, long = jsonl(8), jsonl(8) + jsonl(14, start=8, header=False)
    plain, packed = world.plain(long), world.packed(short)
    world.sync(plain)
    state = world.engine.store.state_token(SID, "codex")
    info = world.sync(packed)
    assert info["ok"] and info["alias_of"] == str(plain) and not info["reindexed"] and world.parses == [0]
    assert "requested_source_is_prefix_of_indexed" in info["coverage"]["issues"] and info["coverage"]["complete"] is False
    assert world.engine.store.state_token(SID, "codex") == state


def test_a_longer_compressed_copy_continues_the_indexed_plain_file(world):
    short, long = jsonl(8), jsonl(8) + jsonl(14, start=8, header=False)
    plain, packed = world.plain(short), world.packed(long)
    world.sync(plain)
    info = world.sync(packed)
    assert info["ok"] and not info["reindexed"] and world.parses == [0, len(short)] and info["new_chunks"] > 0
    assert world.cursor()["path"] == str(packed) and world.cursor()["byte_offset"] == len(long)


def test_a_longer_plain_file_continues_the_indexed_compressed_one(world):
    short, long = jsonl(8), jsonl(8) + jsonl(14, start=8, header=False)
    packed, plain = world.packed(short), world.plain(long)
    world.sync(packed)
    before = world.events()
    info = world.sync(plain)  # the rollout was decompressed and resumed; the old .zst is still lying around
    check_same_session(world, before, info)
    assert world.parses == [0, len(short)] and info["new_chunks"] > 0
    assert world.cursor()["path"] == str(plain) and world.cursor()["byte_offset"] == len(long)
    stale = world.sync(packed)  # the leftover compressed file is now a strict prefix of the indexed one
    assert stale["alias_of"] == str(plain) and "requested_source_is_prefix_of_indexed" in stale["coverage"]["issues"]
    assert world.parses == [0, len(short)]


@pytest.mark.parametrize("indexed_form", ["plain", "packed"])
def test_diverged_plain_and_compressed_files_are_ambiguous(world, indexed_form):
    a, b = jsonl(10), jsonl(10).replace(b"event 4 ", b"event 5 ")
    plain, packed = (world.plain(a), world.packed(b)) if indexed_form == "plain" else (world.plain(b), world.packed(a))
    indexed, rival = (plain, packed) if indexed_form == "plain" else (packed, plain)
    world.sync(indexed)
    state, cursor = world.engine.store.state_token(SID, "codex"), world.cursor()
    info = world.sync(rival)
    assert info["ok"] is False and info["error"] == "ambiguous_session_source"
    assert info["indexed_path"] == str(indexed) and info["requested_path"] == str(rival)
    assert world.engine.store.state_token(SID, "codex") == state and world.cursor() == cursor and world.parses == [0]
    audit = world.engine.audit_session(SID, path=str(rival))
    assert audit["verdict"] == "unknown" and "ambiguous_session_source" in audit["coverage"]["issues"]


def test_the_active_hook_stamp_reads_a_compressed_transcript_as_a_stream(world):
    data = jsonl(12) + (json.dumps(result("c1")) + "\n").encode()
    plain, packed = world.plain(data), world.packed(data, name="twin.jsonl")
    assert _evidence_stamp(world.engine, plain, []) == _evidence_stamp(world.engine, packed, [])


# ---------------------------------------------------------------- bounded memory

def big_session(path, *, megabytes, header=True):
    """About ``megabytes`` of JSONL, 4 KB per record, written line by line."""
    size, i = 0, 0
    with path.open("wb") as stream:
        if header:
            stream.write(json.dumps({"type": "session_meta", "payload": {"id": SID}}).encode() + b"\n")
        while size < megabytes * 1024 * 1024:
            body = "".join(hashlib.sha256(f"{i}:{k}".encode()).hexdigest() for k in range(60))  # no repeats: ~2:1
            line = (json.dumps(message(f"{i:07d} " + body)) + "\n").encode()
            stream.write(line)
            size += len(line)
            i += 1
    return path.stat().st_size


def packed_copy(plain, target):
    """Compress with a streaming frame that still declares its size, without holding the file twice."""
    data = plain.read_bytes()
    target.write_bytes(make_frame(data))
    return len(data)


def test_syncing_a_large_compressed_session_never_materializes_it(world):
    plain = world.root / "big.jsonl"
    size = big_session(plain, megabytes=52)
    packed = world.root / f"rollout-2026-10-14T01-02-03-{SID}.jsonl.zst"
    assert packed_copy(plain, packed) == size
    plain.unlink()
    assert size >= 50 * 1024 * 1024
    tracemalloc.start()
    try:
        info = world.sync(packed)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert info["ok"] and info["total_chunks"] > 12000 and info["coverage"]["complete"]
    assert info["coverage"]["file_size"] == size and info["coverage"]["pending_bytes"] == 0
    print(f"zstd sync: decompressed {size / 1e6:.1f} MB, compressed {packed.stat().st_size / 1e6:.1f} MB, "
          f"peak traced {peak / 1e6:.2f} MB")
    # The identical plain file peaks at the same ~16 MB (sparse batches, spooled rows, id sets): decoding adds nothing.
    assert peak < 0.4 * size, (peak, size)
    tracemalloc.start()  # an unchanged re-sync decodes again for the checks but keeps nothing
    try:
        again = world.sync_authority_only(packed)
        repeat_peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert again["ok"] and again["new_chunks"] == 0 and repeat_peak < 0.1 * size, (repeat_peak, size)
    print(f"zstd re-sync: peak traced {repeat_peak / 1e6:.2f} MB")


def test_continuing_a_cursor_inside_a_large_compressed_source_streams_from_the_start(world):
    whole = world.root / "whole.jsonl"
    size = big_session(whole, megabytes=52)
    data = whole.read_bytes()
    whole.unlink()
    cut = data[:20 * 1024 * 1024]
    cut = cut[:cut.rindex(b"\n") + 1]
    plain = world.plain(cut)  # the first part is indexed as a plain file
    world.sync(plain)
    assert world.cursor()["byte_offset"] == len(cut)
    packed = world.packed(data)  # then the finished session is compressed in place
    plain.unlink()
    del data
    tracemalloc.start()
    try:
        info = world.sync_authority_only(packed)  # skips the indexed 20 MB by decoding and discarding, then parses the rest
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert info["ok"] and not info["reindexed"] and world.parses[-1] == len(cut) and info["new_chunks"] > 7000
    assert info["byte_offset"] == size and info["coverage"]["complete"]
    print(f"zstd continuation: peak traced {peak / 1e6:.2f} MB of {size / 1e6:.1f} MB")
    assert peak < 0.2 * size, (peak, size)
