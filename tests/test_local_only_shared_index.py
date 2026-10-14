"""A local-only process (a lifecycle hook) can share an index with a server that keeps dense vectors.

It has no dense index of its own, so when a reparse makes chunk ids obsolete it must leave their vector cleanup
journalled; acknowledging the journal would orphan those vectors for good.
"""
from __future__ import annotations

import json

from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine

SID = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"


def _record(n, role, text):
    record = {"type": role, "uuid": f"00000000-0000-4000-8000-{n:012d}", "parentUuid": None, "sessionId": SID,
              "timestamp": f"2026-10-01T00:00:{n:02d}.000Z", "cwd": "/synthetic/project", "isSidechain": False}
    if role == "user":
        record.update(message={"role": "user", "content": text}, origin={"kind": "human"})
    else:
        record.update(message={"id": f"msg_{n}", "role": "assistant", "model": "claude-synthetic",
                               "content": [{"type": "text", "text": text}], "stop_reason": "end_turn"})
    return record


def _write(path, *texts):
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [_record(n, "user" if n % 2 == 0 else "assistant", text) for n, text in enumerate(texts)]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _engine(tmp_path, *, local_only):
    return SelfDirectEngine(Settings(_env_file=None, index_dir=tmp_path / "index", local_only=local_only,
                                     use_fake_embedder=not local_only, dense_backend="numpy",
                                     claude_projects_dir=tmp_path / "projects", openrouter_api_key_file=None))


def test_local_only_sync_keeps_vector_cleanup_journalled_for_the_dense_server(tmp_path):
    path = tmp_path / "projects" / "-synthetic-project" / f"{SID}.jsonl"
    _write(path, "first request alpha", "first answer bravo", "second request charlie")
    server = _engine(tmp_path, local_only=False)
    assert server.sync_session(session_id=SID, provider="claude", path=str(path), embed=True)["ok"]
    before = set(server.store.chunk_ids(SID, "claude"))
    assert server.dense.present(sorted(before)) == before
    server.close()

    _write(path, "rewritten request delta", "rewritten answer echo")       # not an append: a full reparse
    hook = _engine(tmp_path, local_only=True)
    try:
        assert hook.sync_session(session_id=SID, provider="claude", path=str(path), embed=False)["ok"]
        current = set(hook.store.chunk_ids(SID, "claude"))
        obsolete = before - current
        assert obsolete, "the rewrite must make some chunk ids obsolete"
        assert not hook.sparse.present(sorted(obsolete))                    # sparse cleanup is done here
        assert set(hook.store.pending_deletions(SID, "claude")) >= obsolete    # vectors are still owed
    finally:
        hook.close()

    server = _engine(tmp_path, local_only=False)
    try:
        assert server.sync_session(session_id=SID, provider="claude", path=str(path), embed=True)["ok"]
        assert not server.dense.present(sorted(obsolete))
        assert server.store.pending_deletions(SID, "claude") == []
    finally:
        server.close()


def test_a_purely_local_index_still_acknowledges_its_deletions(tmp_path):
    path = tmp_path / "projects" / "-synthetic-project" / f"{SID}.jsonl"
    _write(path, "first request alpha", "first answer bravo")
    hook = _engine(tmp_path, local_only=True)
    try:
        assert hook.sync_session(session_id=SID, provider="claude", path=str(path), embed=False)["ok"]
        _write(path, "rewritten request delta")
        assert hook.sync_session(session_id=SID, provider="claude", path=str(path), embed=False)["ok"]
        assert hook.store.pending_deletions(SID, "claude") == []           # no dense store: nothing is owed
    finally:
        hook.close()
