"""Compressed Codex rollouts: discovery, session lookup by id and explicit path resolution.

Synthetic files only. ``rollout-*.jsonl.zst`` follows the same root and traversal validation as ``.jsonl``;
the session id comes from the ``session_meta`` header, not from a file name that may carry two UUIDs.
"""
from __future__ import annotations

import pytest

from self_directing_mcp import compressed_source
from self_directing_mcp.codex import discover
from self_directing_mcp.codex import parse as codex_parser
from self_directing_mcp.compressed_source import CompressedSourceError
from self_directing_mcp.session_events import capture_source
from test_zstd_stream import BACKENDS, OTHER, SID, jsonl, make_frame

pytestmark = pytest.mark.skipif(not BACKENDS, reason="needs Python 3.14+ or the zstandard package")


def test_without_a_zstd_module_a_compressed_source_is_an_explicit_error_and_plain_jsonl_still_works(tmp_path, monkeypatch):
    data = jsonl(50)
    (tmp_path / "a.jsonl.zst").write_bytes(make_frame(data))
    (tmp_path / "a.jsonl").write_bytes(data)
    monkeypatch.setattr(compressed_source, "_PREFERENCE", ())
    with pytest.raises(CompressedSourceError) as caught:
        capture_source(tmp_path / "a.jsonl.zst")
    assert caught.value.code == "zstd_unavailable" and "zstandard" in str(caught.value)
    assert capture_source(tmp_path / "a.jsonl").length == len(data)
    assert codex_parser.peek_session_id(tmp_path / "a.jsonl") == SID
    assert codex_parser.peek_session_id(tmp_path / "a.jsonl.zst") is None  # a peek never raises
    with pytest.raises(CompressedSourceError):
        codex_parser.read_session_id(tmp_path / "a.jsonl.zst")


def test_discovery_and_session_lookup_cover_compressed_rollouts(tmp_path):
    root = tmp_path / "sessions"
    day = root / "2026" / "10" / "14"
    day.mkdir(parents=True)
    plain = day / f"rollout-2026-10-14T01-02-03-{SID}.jsonl"
    forked = day / f"rollout-2026-10-14T04-05-06-{OTHER}_{SID}.jsonl.zst"  # header id is the second uuid
    archived = day / "rollout-2026-10-13T01-02-03-11111111-1111-1111-1111-111111111111.jsonl.zst"
    stray = day / "notes.jsonl.zst"  # not a rollout
    plain.write_bytes(jsonl(3))
    forked.write_bytes(make_frame(jsonl(3)))
    archived.write_bytes(make_frame(jsonl(3).replace(SID.encode(), b"11111111-1111-1111-1111-111111111111")))
    stray.write_bytes(make_frame(jsonl(3)))
    found = discover.discover_session_files(root)
    assert set(found) == {plain, forked, archived} and stray not in found
    assert discover.session_id_from_filename(forked) is None  # two uuids: the header decides
    assert discover.session_id_from_filename(archived) == "11111111-1111-1111-1111-111111111111"
    assert discover.find_session_by_id(root, "11111111-1111-1111-1111-111111111111") == archived.resolve()
    assert discover.find_session_by_id(root, SID) == plain.resolve()  # the plain, live file wins over the _uuid fork
    plain.unlink()
    assert discover.find_session_by_id(root, SID) == forked.resolve()  # found by its header, not its name
    assert discover.find_session_by_id(root, "00000000-0000-0000-0000-000000000000") is None
    assert discover.resolve_session_path(root, session_id=SID) == forked.resolve()
    assert discover.resolve_session_path(root, path=archived) == archived.resolve()


def test_plain_rollout_sorts_before_its_compressed_twin(tmp_path):
    root = tmp_path / "sessions"
    root.mkdir()
    name = f"rollout-2026-10-14T01-02-03-{SID}.jsonl"
    (root / name).write_bytes(jsonl(2))
    (root / (name + ".zst")).write_bytes(make_frame(jsonl(2)))
    assert discover.discover_session_files(root) == [root / name, root / (name + ".zst")]
    assert discover.find_session_by_id(root, SID) == (root / name).resolve()


def test_compressed_paths_keep_the_root_and_traversal_validation(tmp_path):
    root = tmp_path / "sessions"
    root.mkdir()
    outside = tmp_path / "elsewhere" / f"rollout-2026-10-14T01-02-03-{SID}.jsonl.zst"
    outside.parent.mkdir()
    outside.write_bytes(make_frame(jsonl(2)))
    for candidate in (outside, root / ".." / "elsewhere" / outside.name):
        with pytest.raises(discover.PathTraversalError):
            discover.resolve_session_path(root, path=candidate)


def test_an_unreadable_compressed_candidate_is_reported_not_skipped(tmp_path):
    root = tmp_path / "sessions"
    root.mkdir()
    (root / f"rollout-2026-10-14T01-02-03-{OTHER}_{SID}.jsonl.zst").write_bytes(b"garbage" * 30)
    with pytest.raises(CompressedSourceError) as caught:
        discover.find_session_by_id(root, SID)
    assert caught.value.code == "zstd_corrupt" and "may be among them" in str(caught.value)
