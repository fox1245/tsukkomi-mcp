"""Source-integrity micro-benchmark on a synthetic session. Nothing real is read.

Writes one synthetic Codex rollout (default 300 MB) into a temporary directory, then measures CPU and wall time
for the operations whose cost depends on the size of the already indexed prefix:

  first sync                    index the whole file (authority only, no sparse/dense derivation)
  nothing appended              check_action as a hook asks it (fast verification when available)
  small append                  the same after a few KB were appended
  audit_session                 full verification, evaluates every indexed event

check_action is measured with two contract sets: a prohibition only (what most hook-time checks evaluate: the
proposed action alone) and with a prerequisite gate added (a "must ... before ..." contract, which evaluates the
indexed history; that cost grows with the number of events, not with the bytes of the file).

Each row reports wall and CPU time, the CPU spent verifying the source ("verify cpu": the refresh of the
session index plus every later source check), and the bytes passed through SHA-256 as a multiple of the file
size ("passes"): 1.0 means the whole file was hashed once. The count comes from wrapping hashlib.sha256 for the
whole process, so a hash site that is not routed through the helpers cannot hide.

    python scripts/bench_integrity.py [--size-mb 300] [--repeat 3] [--dir DIR] [--json]

The script runs on commits before the fast path existed: it passes verify="fast" only when the engine accepts
it. Synthetic data only; the per-user TOML configuration is replaced by an empty file.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SID = "11111111-2222-3333-4444-555555555555"
FILLER = ("synthetic harmless output evidence for the integrity benchmark " * 70)[:4000]


class _Counted:
    """hashlib object proxy that counts the bytes fed to it (copies share the same counter)."""

    def __init__(self, inner, counter):
        self._inner, self._counter = inner, counter

    def update(self, data):
        self._counter[0] += len(data)
        self._inner.update(data)

    def copy(self):
        return _Counted(self._inner.copy(), self._counter)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def install_counter():
    counter = [0]
    real = hashlib.sha256

    def counting(data=b"", **kwargs):
        digest = _Counted(real(**kwargs), counter)
        if data:
            digest.update(data)
        return digest

    hashlib.sha256 = counting
    return counter, lambda: setattr(hashlib, "sha256", real)


def instrument(engine):
    """Accumulate the CPU time spent verifying the source: the refresh of the session index (capture, ingestion,
    coverage) and every later check of an observation. Nested calls are counted once."""
    from self_directing_mcp.session_events import SourceObservation

    total, depth = [0.0], [0]

    def timed(function):
        def wrapper(*args, **kwargs):
            depth[0] += 1
            started = time.process_time()
            try:
                return function(*args, **kwargs)
            finally:
                depth[0] -= 1
                if not depth[0]:
                    total[0] += time.process_time() - started
        return wrapper

    engine._refresh_evidence = timed(engine._refresh_evidence)
    SourceObservation.validate_prefix = timed(SourceObservation.validate_prefix)
    return total


def write_session(path: Path, size_mb: float) -> int:
    """A prerequisite (pytest, successful) up front, then filler messages up to ``size_mb``."""
    rows = [{"type": "session_meta", "payload": {"id": SID}},
            {"type": "response_item", "payload": {"type": "function_call", "name": "shell", "call_id": "t1",
                                                  "arguments": {"command": "pytest -q"}}},
            {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "t1",
                                                  "output": json.dumps({"exit_code": 0, "output": "ok"})}}]
    target = int(size_mb * 1024 * 1024)
    written = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            written += stream.write(json.dumps(row) + "\n")
        index = 0
        while written < target:
            line = json.dumps({"type": "response_item", "payload": {
                "type": "message", "role": "assistant", "content": f"{index} {FILLER}"}})
            written += stream.write(line + "\n")
            index += 1
    return path.stat().st_size


def append_events(path: Path, count: int, tag: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for index in range(count):
            stream.write(json.dumps({"type": "response_item", "payload": {
                "type": "message", "role": "assistant", "content": f"{tag} {index} appended"}}) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--size-mb", type=float, default=300.0, help="synthetic session size (default 300)")
    parser.add_argument("--repeat", type=int, default=3, help="repetitions of the repeatable rows (median reported)")
    parser.add_argument("--append-events", type=int, default=5, help="events appended for the small-append row")
    parser.add_argument("--dir", type=Path, default=None, help="parent directory for the temporary data")
    parser.add_argument("--keep", action="store_true", help="keep the temporary directory")
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    args = parser.parse_args(argv)

    work = Path(tempfile.mkdtemp(prefix="bench-integrity-", dir=args.dir))
    config = work / "empty-config.toml"
    config.write_text("", encoding="utf-8")
    for name in ("SELF_DIRECT_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER",
                 "SELF_DIRECT_INDEX_DIR", "SELF_DIRECT_OPENROUTER_API_KEY_FILE", "OPENROUTER_API_KEY"):
        os.environ.pop(name, None)
    os.environ["SELF_DIRECT_CONFIG_FILE"] = str(config)
    counter, restore = install_counter()
    try:
        from self_directing_mcp.config import Settings
        from self_directing_mcp.engine import SelfDirectEngine

        root = work / "sessions"
        root.mkdir()
        path = root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl"
        size = write_session(path, args.size_mb)
        engine = SelfDirectEngine(Settings(codex_sessions_dir=root, index_dir=work / "index", local_only=True,
                                           _env_file=None))
        engine.ensure_ready()
        verify_cpu = instrument(engine)
        fast = "verify" in inspect.signature(engine.check_action).parameters
        action = {"tool_name": "shell", "arguments": {"command": "deploy now"}}
        kwargs = {"verify": "fast"} if fast else {}
        results = []

        def measure(label, work_fn, repeats=1, tracked=True):
            samples = []
            for _ in range(repeats):
                before = (time.perf_counter(), time.process_time(), counter[0], verify_cpu[0])
                value = work_fn()
                samples.append((time.perf_counter() - before[0], time.process_time() - before[1],
                                (counter[0] - before[2]) / max(1, path.stat().st_size), verify_cpu[0] - before[3]))
                last = value
            results.append({"row": label, "wall_s": statistics.median(s[0] for s in samples),
                            "cpu_s": statistics.median(s[1] for s in samples),
                            "verify_cpu_s": statistics.median(s[3] for s in samples) if tracked else None,
                            "passes": statistics.median(s[2] for s in samples),
                            "verdict": last.get("verdict") if isinstance(last, dict) else None,
                            "mode": (last.get("integrity") or {}).get("mode") if isinstance(last, dict) else None})

        # Warm the process (imports, native stages) on a tiny session so rows below exclude one-time costs.
        warm = root / "rollout-2026-09-09T01-00-00-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
        warm.write_text(json.dumps({"type": "session_meta", "payload": {"id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}})
                        + "\n", encoding="utf-8")
        engine.upsert_contracts([
            {"id": "no-forbidden", "type": "must_not", "scope": "tool_call", "regex": "FORBIDDEN_MARKER"}])
        engine.check_action("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", action)

        mode = "fast" if fast else "no fast path"
        measure("first sync", lambda: engine._sync_session(session_id=SID, provider="codex", embed=False,
                                                           derived=False), tracked=False)
        measure(f"check_action, nothing appended ({mode}), prohibition",
                lambda: engine.check_action(SID, action, **kwargs), args.repeat)
        counter_tag = iter(range(10 ** 6))

        def small_append():
            append_events(path, args.append_events, f"small-{next(counter_tag)}")
            return engine.check_action(SID, action, **kwargs)

        measure(f"check_action, {args.append_events} events appended ({mode}), prohibition", small_append, args.repeat)
        engine.upsert_contracts([
            {"id": "tests-first", "type": "must", "scope": "tool_call", "regex": "pytest", "before_regex": "deploy"}])
        measure(f"check_action, nothing appended ({mode}), + prerequisite gate",
                lambda: engine.check_action(SID, action, **kwargs), args.repeat)
        measure("audit_session (full)", lambda: engine.audit_session(SID), args.repeat)
        engine.close()
    finally:
        restore()
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)

    report = {"size_bytes": size, "fast_path": fast, "rows": results}
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"synthetic session: {size / 1e6:.1f} MB; fast path available: {fast}")
    print(f"{'row':<62}{'wall s':>8}{'cpu s':>8}{'verify cpu s':>14}{'passes':>8}  verdict/mode")
    for row in results:
        verify = "-" if row["verify_cpu_s"] is None else f"{row['verify_cpu_s']:.3f}"
        print(f"{row['row']:<62}{row['wall_s']:>8.3f}{row['cpu_s']:>8.3f}{verify:>14}"
              f"{row['passes']:>8.2f}  {row['verdict']}/{row['mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
