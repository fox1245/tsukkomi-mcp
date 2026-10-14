"""Large-session timeout regression. All records are synthetic and remain local."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine

SID = "11111111-2222-3333-4444-555555555555"
SMALL = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def settings(root, index=None):
    for name in ("SELF_DIRECT_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER"):
        os.environ.pop(name, None)
    return Settings(codex_sessions_dir=root, index_dir=index or root / "index",
                    use_fake_embedder=True, dense_backend="numpy", _env_file=None)


def worker(root, sid):
    from self_directing_mcp import server
    from self_directing_mcp.schemas import ProposedAction
    engine = SelfDirectEngine(settings(root))
    server._engine = engine
    values, busy_replies = [], 0
    async def check():
        nonlocal busy_replies
        for _ in range(3):
            start = time.perf_counter()
            for attempt in range(5):
                result = await server.check_action(sid, ProposedAction(tool_name="shell", arguments="echo safe"))
                if result.get("verdict") == "clean":
                    break
                # Retry only a genuine bounded engine/file lease conflict.
                # Deadline, cancellation and operation/storage failures are not busy.
                assert result.get("error") == "index_busy", result
                assert result["coverage"]["complete"] is False
                busy_replies += 1
                if attempt == 4 or time.perf_counter() - start >= 8.0:
                    raise AssertionError("No verified response within the bounded contention budget")
                await asyncio.sleep(0.05)
            values.append(time.perf_counter() - start)
    try:
        asyncio.run(check())
    finally:
        engine.close()
    return {"seconds": values, "busy_replies": busy_replies}


def run(root, events, baseline=None):
    root.mkdir(parents=True, exist_ok=True)
    text = ("synthetic harmless output evidence " * 130)[:4096]
    for sid, count in ((SID, events), (SMALL, 50)):
        path = root / f"rollout-2026-09-09T01-00-00-{sid}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"type": "session_meta", "payload": {"id": sid}}) + "\n")
            for i in range(count):
                stream.write(json.dumps({"type": "response_item", "payload": {
                    "type": "message", "role": "assistant", "content": f"{i} {text}"}}) + "\n")
    engine = SelfDirectEngine(settings(root))
    try:
        started = time.perf_counter()
        engine.sync_session(session_id=SID, embed=False)
        engine.sync_session(session_id=SMALL, embed=False)
        cold = time.perf_counter() - started
        engine.upsert_contracts([{"id": "no-forbidden", "type": "must_not", "regex": "FORBIDDEN_MARKER"}])
        before = engine.sparse._conn.total_changes
        started = time.perf_counter()
        checked = engine.check_action(SID, {"tool_name": "shell", "arguments": "echo safe"})
        warm = time.perf_counter() - started
        assert checked["verdict"] == "clean"
        assert engine.sparse._conn.total_changes == before
        started = time.perf_counter()
        audited = engine.audit_session(SID)
        full = time.perf_counter() - started
        assert audited["verdict"] == "clean"
    finally:
        engine.close()
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    def child(sid):
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", sid, "--work-dir", str(root)]
        result = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = list(pool.map(child, (SID, SMALL)))
    values = [value for group in concurrent for value in group["seconds"]]
    report = {"events": events + 52, "large_file_bytes": (root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl").stat().st_size,
              "cold_sync_seconds": round(cold, 3), "warm_check_seconds": round(warm, 3),
              "full_audit_seconds": round(full, 3), "concurrent_max_seconds": round(max(values), 3),
              "concurrent_calls": len(values), "busy_replies": sum(g["busy_replies"] for g in concurrent),
              "unchanged_fts_writes": 0}
    print(json.dumps(report), flush=True)
    assert warm < 8 and full < 10 and max(values) < 8, report
    if baseline:
        old_index = root / "baseline-index"
        shutil.copytree(root / "index", old_index)
        code = """import sys,time,json
from pathlib import Path
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
root=Path(sys.argv[1]); e=SelfDirectEngine(Settings(codex_sessions_dir=root,index_dir=root/'baseline-index',use_fake_embedder=True,dense_backend='numpy',_env_file=None))
start=time.perf_counter()
r=e.check_action(sys.argv[2],{'tool_name':'shell','arguments':'echo safe'})
print(json.dumps({'baseline_check_seconds':time.perf_counter()-start,'verdict':r['verdict']}))
e.close()
"""
        env["PYTHONPATH"] = str(Path(baseline) / "src")
        process = subprocess.Popen([sys.executable, "-B", "-c", code, str(root), SID],
                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, error = process.communicate(timeout=65)
            assert process.returncode == 0, error.decode(errors="replace")
            print(out.decode().strip(), flush=True)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            print(json.dumps({"baseline_check_seconds": ">65", "baseline_stopped": True}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--events", type=int, default=3000)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.work_dir, args.worker)))
    elif args.work_dir:
        run(args.work_dir, args.events, args.baseline)
    else:
        with tempfile.TemporaryDirectory(prefix="self-direct-large-") as directory:
            run(Path(directory), args.events, args.baseline)
