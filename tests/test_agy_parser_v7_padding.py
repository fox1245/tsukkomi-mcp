"""AGY parser v7: a host-written NUL run in front of a complete record.

Synthetic data only. The host once wrote a line made of a contiguous run of NUL bytes immediately followed by a
complete, valid JSON record, between two consecutive steps. The parser strips that leading run, accepts the
record and records the padding; every other use of NUL bytes stays a visible parse error.
"""
import json

import pytest

from self_directing_mcp import agy
from self_directing_mcp.agy import parse_session_chunks
from self_directing_mcp.config import Settings
from self_directing_mcp.embed.embedder import FakeEmbedder
from self_directing_mcp.engine import SelfDirectEngine

from test_agy_parser_v6 import (  # noqa: F401  (env is a fixture)
    SID, append, chunks_of, env, finding, generic, hook, logs, planner, sync, user, write)

PAD = 1550


def raw_lines(path, *lines):
    path.write_bytes(b"".join(line if line.endswith(b"\n") else line + b"\n" for line in lines))
    return path


def encoded(row):
    return json.dumps(row).encode()


def padded(row, count=PAD):
    return b"\x00" * count + encoded(row)


def session_with(tmp_path, middle, *, name="transcript.jsonl"):
    path = logs(tmp_path) / name
    raw_lines(path, encoded(user(0, "keep the logs")), middle, encoded(planner(2, "done")))
    return path


def parse_errors(chunks):
    return [c for c in chunks if c.meta.get("parse_error")]


def test_parser_version_has_reached_seven():
    assert agy.PARSER_VERSION >= 7   # v8 moved model-written tool labels out of the call text


def test_nul_padded_record_is_accepted_and_the_padding_is_recorded(tmp_path):
    middle = padded(planner(1, "I will run it", [("run_command", {"CommandLine": "npm test"})]))
    path = session_with(tmp_path, middle)
    chunks, offset, _sid = parse_session_chunks(path)
    assert not parse_errors(chunks)
    step = [c for c in chunks if c.meta.get("step_index") == 1]
    assert [c.text for c in step][0] == "[message:assistant] I will run it"
    assert any(c.meta.get("evidence_origin") == "model_proposal" and '"CommandLine": "npm test"' in c.text for c in step)
    assert all(c.meta["host_padding_bytes"] == PAD for c in step)
    neighbours = [c for c in chunks if c.meta.get("step_index") in (0, 2)]
    assert neighbours and not any("host_padding_bytes" in c.meta for c in neighbours)
    # The byte range keeps the host's own bytes: anchors and prefix digests cover the padding.
    assert {(c.byte_end - c.byte_start) for c in step} == {len(middle) + 1}
    assert offset == path.stat().st_size


def test_padding_is_counted_once_per_record_in_the_session_stats(env):
    engine, root = env
    path = session_with(root, padded(planner(1, "one", [("run_command", {"CommandLine": "ls"}),
                                                        ("run_command", {"CommandLine": "pwd"})])))
    info = sync(engine, path)
    assert info["coverage"]["complete"] and info["coverage"]["parse_error_count"] == 0
    assert info["coverage"]["agy_evidence"]["host_padded_records"] == 1
    append(path, planner(3, "again"))
    path.write_bytes(path.read_bytes() + padded(planner(4, "two")) + b"\n")
    info = sync(engine, path)
    assert not info["reindexed"]
    persisted = info["coverage"]["agy_evidence"]
    assert persisted["host_padded_records"] == 2
    engine.store._conn.execute("DELETE FROM session_agy_stats")  # force the scan fallback
    assert engine.store.agy_evidence_stats(SID) == persisted


def test_padding_does_not_excuse_a_gap_or_make_a_bad_record_good(env):
    engine, root = env
    path = logs(root) / "transcript.jsonl"
    # Steps 1 and 3 are the neighbours of a line that is only NULs: the missing step 2 stays visible.
    raw_lines(path, encoded(user(0, "go")), encoded(planner(1, "ok")), b"\x00" * PAD, encoded(planner(3, "later")))
    info = sync(engine, path)
    assert info["coverage"]["parse_error_count"] == 1 and not info["coverage"]["complete"]
    assert "parse_errors" in info["coverage"]["issues"]
    assert info["coverage"]["agy_evidence"]["host_padded_records"] == 0
    assert [c.meta["step_index"] for c in chunks_of(engine) if c.meta.get("step_index") is not None] == [0, 1, 3]


INVALID = {
    "only_nuls": lambda: b"\x00" * PAD,
    "nuls_then_whitespace": lambda: b"\x00" * PAD + b"  \t",
    "nuls_then_garbage": lambda: b"\x00" * PAD + b"not json at all",
    "nuls_then_truncated_record": lambda: b"\x00" * PAD + encoded(planner(1, "cut"))[:-8],
    "nuls_then_non_object": lambda: b"\x00" * PAD + b"[1, 2, 3]",
    "nuls_then_record_without_step": lambda: b"\x00" * PAD + encoded({"type": "USER_INPUT", "source": "USER_EXPLICIT",
                                                                    "content": "hi", "status": "DONE"}),
    "record_then_nuls": lambda: encoded(planner(1, "ok")) + b"\x00" * PAD,
    "nuls_inside_the_record": lambda: b"\x00" * 3 + encoded(planner(1, "ok"))[:20] + b"\x00" + encoded(planner(1, "ok"))[20:],
    "nul_inside_a_string": lambda: b"\x00" * 3 + b'{"step_index": 1, "type": "PLANNER_RESPONSE", "source": "MODEL", "content": "a\x00b"}',
    "space_then_nuls_then_record": lambda: b" " + b"\x00" * 4 + encoded(planner(1, "ok")),
    "nuls_around_a_record": lambda: b"\x00" * 4 + encoded(planner(1, "ok")) + b"\x00" * 4,
}


@pytest.mark.parametrize("name", sorted(INVALID))
def test_every_other_use_of_nul_bytes_stays_a_parse_error(env, name):
    engine, root = env
    path = session_with(root, INVALID[name]())
    chunks, _offset, _sid = parse_session_chunks(path)
    errors = parse_errors(chunks)
    assert len(errors) == 1
    assert not [c for c in chunks if c.meta.get("step_index") == 1]
    assert not any("host_padding_bytes" in c.meta for c in chunks)
    info = sync(engine, path)
    assert info["coverage"]["parse_error_count"] == 1 and not info["coverage"]["complete"]
    assert info["coverage"]["agy_evidence"]["host_padded_records"] == 0


def test_padding_in_the_middle_of_a_run_of_padded_and_plain_records(tmp_path):
    path = logs(tmp_path) / "transcript.jsonl"
    raw_lines(path, encoded(user(0, "go")), padded(planner(1, "a"), 1), padded(planner(2, "b"), 2),
              encoded(planner(3, "c")), padded(generic(4), 7))
    chunks, *_ = parse_session_chunks(path)
    assert not parse_errors(chunks)
    seen = {c.meta["step_index"]: c.meta.get("host_padding_bytes") for c in chunks}
    assert seen == {0: None, 1: 1, 2: 2, 3: None, 4: 7}


def test_the_full_transcript_fallback_reads_a_padded_line_too(tmp_path):
    folder = logs(tmp_path)
    full = padded(user(0, "Keep the logs for thirty days"))
    raw_lines(folder / "transcript_full.jsonl", full, encoded(planner(1, "ok")))
    display = write(folder / "transcript.jsonl", [user(0, "Keep the logs", truncated_fields=["content"]), planner(1, "ok")])
    chunks, *_ = parse_session_chunks(display)
    first = next(c for c in chunks if c.meta["step_index"] == 0)
    assert first.text == "[message:user] Keep the logs for thirty days"
    assert first.meta["source"] == "full" and first.meta["restored_fields"] == ["content"]
    assert first.meta["full_ref"]["byte_end"] - first.meta["full_ref"]["byte_start"] == len(full) + 1
    assert "host_padding_bytes" not in first.meta  # that is about the display line, which was not padded


def test_a_padded_full_transcript_is_parsed_directly_and_stays_padded_in_meta(tmp_path):
    path = session_with(tmp_path, padded(planner(1, "x")),
                        name="transcript_full.jsonl")
    chunks, *_ = parse_session_chunks(path)
    assert not parse_errors(chunks)
    assert next(c for c in chunks if c.meta["step_index"] == 1).meta["host_padding_bytes"] == PAD


def test_hook_receipts_work_across_a_padded_line_and_still_link(env):
    engine, root = env
    path = logs(root) / "transcript.jsonl"
    raw_lines(path, encoded(user(0, "run the tests")), padded(planner(1, "", [("run_command", {"CommandLine": "npm test"})])))
    engine.record_agy_hook(hook(path, 2, "npm test"), "PreToolUse")  # anchors on the prefix that holds the padded line
    append(path, generic(2, "tests finished", code=0))
    engine.record_agy_hook(hook(path, 2, "npm test"), "PostToolUse")
    info = sync(engine, path)
    assert info["coverage"]["complete"] and info["coverage"]["agy_evidence"]["host_padded_records"] == 1
    result = next(c for c in chunks_of(engine) if c.kind == "tool_result")
    assert result.meta["linkage"] == "host_step_receipt" and result.meta["success"] is True
    engine.upsert_contracts([{"id": "tests-first", "provider": "agy", "type": "must", "scope": "tool_call",
                              "regex": "npm test", "before_regex": "deploy", "requires_success": True}])
    check = engine.check_action(SID, {"tool_name": "run_command", "arguments": {"CommandLine": "deploy"}},
                                provider="agy", path=str(path))
    assert finding(check, "tests-first")["verdict"] == "clean"


def test_a_padded_native_result_is_hash_bound_like_any_other_result(env):
    engine, root = env
    path = logs(root) / "transcript.jsonl"
    raw_lines(path, encoded(user(0, "go")), encoded(planner(1, "", [("run_command", {"CommandLine": "ls"})])))
    engine.record_agy_hook(hook(path, 2, "ls"), "PreToolUse")
    with path.open("ab") as stream:
        stream.write(padded(generic(2, "listing", code=0), 99) + b"\n")
    engine.record_agy_hook(hook(path, 2, "ls"), "PostToolUse")
    info = sync(engine, path)
    assert info["coverage"]["complete"]
    result = next(c for c in chunks_of(engine) if c.kind == "tool_result")
    assert result.meta["linkage"] == "host_step_receipt" and result.meta["host_padding_bytes"] == 99


def test_a_parser_bump_turns_the_old_parse_error_into_the_record(env, monkeypatch):
    engine, root = env
    path = session_with(root, padded(planner(1, "ok")))
    with monkeypatch.context() as old:
        # What parser v6 did with the line: a plain JSON decode of the raw text.
        old.setattr(agy, "PARSER_VERSION", 6)
        old.setattr(agy, "load_line", lambda raw: (json.loads(raw), 0))
        first = sync(engine, path)
    assert first["coverage"]["parse_error_count"] == 1 and first["parser_version"] == 6
    second = sync(engine, path)
    assert second["reindexed"] and second["reindex_reason"] == "parser_updated" and second["parser_version"] == agy.PARSER_VERSION
    assert second["coverage"]["parse_error_count"] == 0 and second["coverage"]["complete"]
    assert not sync(engine, path)["reindexed"]


class RecordingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return super().embed_documents(texts)


def test_a_parser_bump_keeps_event_ids_anchors_and_cached_vectors(tmp_path, monkeypatch):
    root = tmp_path / "app"
    engine = SelfDirectEngine(Settings(_env_file=None, use_fake_embedder=True, index_dir=tmp_path / "index",
                                      agy_app_data_dirs=[root], openrouter_api_key_file=None))
    try:
        path = session_with(root, padded(planner(1, "I will run it")))
        engine.ensure_ready()
        spy = engine.embedder = RecordingEmbedder()
        with monkeypatch.context() as old:
            old.setattr(agy, "PARSER_VERSION", 6)
            old.setattr(agy, "load_line", lambda raw: (json.loads(raw), 0))  # v6: the padded line is a parse error
            first = engine.sync_session(session_id=SID, provider="agy", path=str(path), embed=True)
        assert first["parser_version"] == 6 and first["coverage"]["parse_error_count"] == 1
        before, calls = chunks_of(engine), spy.calls
        error = next(c.chunk_id for c in before if c.meta.get("parse_error"))
        kept = {c.chunk_id for c in before if not c.meta.get("parse_error")}
        assert kept <= engine.dense.ids()
        updated = engine.sync_session(session_id=SID, provider="agy", path=str(path), embed=False)
        assert updated["reindex_reason"] == "parser_updated" and updated["parser_version"] == agy.PARSER_VERSION
        after = {c.chunk_id for c in chunks_of(engine)}
        assert kept <= after and error not in after  # unchanged events keep their ids (and so any anchors)
        assert kept <= engine.dense.ids() and error not in engine.dense.ids() | engine.sparse.ids()
        assert spy.calls == calls  # nothing was sent to the embedding API: the new event waits for a later embed
        assert updated["embedding_pending"] == len(after - kept) == 1
        assert not engine.sync_session(session_id=SID, provider="agy", path=str(path), embed=False)["reindexed"]
    finally:
        engine.close()
