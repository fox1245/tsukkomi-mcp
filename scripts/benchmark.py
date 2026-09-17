"""Synthetic offline regression budget; not a production throughput claim."""
from __future__ import annotations

import argparse
import cProfile
import json
import os
from pathlib import Path
import tempfile
import time

from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine


def run(events=1000):
    for name in ("SELF_DIRECT_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER"):
        os.environ.pop(name, None)
    with tempfile.TemporaryDirectory(prefix="self-direct-benchmark-") as scratch:
        root = Path(scratch)
        sid = "11111111-2222-3333-4444-555555555555"
        path = root / f"rollout-2026-09-09T01-00-00-{sid}.jsonl"
        rows = [{"type": "session_meta", "payload": {"id": sid}}]
        rows += [{"type": "response_item", "payload": {"type": "function_call", "name": "shell",
                  "call_id": f"call-{i}", "arguments": {"command": f"echo synthetic-{i}"}}} for i in range(events)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        engine = SelfDirectEngine(Settings(codex_sessions_dir=root, index_dir=root / "index",
                                          use_fake_embedder=True, dense_backend="numpy", _env_file=None))
        try:
            start = time.perf_counter()
            first = engine.sync_session(session_id=sid)
            engine.upsert_contracts([{"id": "no-delete", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"}])
            audit = engine.audit_session(sid)
            elapsed = time.perf_counter() - start
            second_start = time.perf_counter()
            second = engine.sync_session(session_id=sid)
            second_elapsed = time.perf_counter() - second_start
            result = {"events": events + 1, "sync_and_audit_seconds": round(elapsed, 3),
                      "unchanged_sync_seconds": round(second_elapsed, 3), "second_embedded": second["embedded"],
                      "verdict": audit["verdict"], "backend": first["dense_backend"]}
            print(json.dumps(result))
            assert first["total_chunks"] == events + 1 and audit["verdict"] == "clean"
            assert second["embedded"] == 0 and elapsed < 15, result
        finally:
            engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    args = parser.parse_args()
    if args.profile:
        profiler = cProfile.Profile()
        profiler.runcall(run)
        profiler.dump_stats(args.profile)
    else:
        run()
