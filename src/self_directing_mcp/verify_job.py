"""The detached verifier: the shadow comparison that a hook process does not wait for.

An incremental audit answers from stored monitor states; every so often (the adaptive schedule of ``audit.shadow``) the
answer is compared with a full evaluation of the whole session, which on a long transcript takes seconds. A command hook
lives for one event and must answer fast, so it leaves its planned comparisons to this job (``engine.defer_comparisons``,
``AuditPlan.prepare(hook=True)``): the hook starts it detached after it has answered (``claude_hooks.spawn_verifier``),
and the job compares the latest state of the session with a full evaluation exactly as the idle comparison of a resident
engine does (``SelfDirectEngine.verify_incremental``): on a mismatch the contract's incremental path is switched off and
the requests answered since the last match are recorded as needing review. A comparison that an answer calls for (a
violation, a violation that turned clean, a high-risk contract) is not left to the job: the hook makes it in the request.

One verifier runs at a time per index (a lock file next to the index); a hook that finds one running does not start
another, and the next planned comparison covers the session. Nothing is printed. Linux and macOS only (``fcntl``): on
other systems a hook process makes its comparisons itself, as before.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

LOCK_NAME = "verify-job.lock"

try:
    import fcntl
except ImportError:         # not POSIX: the hooks keep their comparisons in the request (see ``supported``)
    fcntl = None


def supported() -> bool:
    return fcntl is not None


def lock_path(index_dir) -> Path:
    return Path(index_dir) / LOCK_NAME


def running(index_dir) -> bool:
    """Whether a verifier holds the lock of this index right now (a hook then does not start another process that would
    only find the lock taken)."""
    if fcntl is None:
        return False
    try:
        with open(lock_path(index_dir), "a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(lock, fcntl.LOCK_UN)
    except OSError:
        return False
    return False


def run(provider: str, session_id: str, path: str | None, index_dir) -> str:
    """Compare one session. Returns what happened: ``"busy"`` (another verifier holds the lock), ``"compared"``,
    ``"mismatch"`` or ``"not_compared"``."""
    if fcntl is None:
        return "not_compared"
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    with open(lock_path(index_dir), "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return "busy"
        os.environ["SELF_DIRECT_LOCAL_ONLY"] = "true"          # a verifier never embeds
        os.environ["SELF_DIRECT_INDEX_DIR"] = str(index_dir)
        from self_directing_mcp.engine import SelfDirectEngine
        from self_directing_mcp.settings_core import load_hook_settings
        engine = SelfDirectEngine(load_hook_settings())
        try:
            engine.ensure_ready()                  # a background operation never initializes an engine itself
            # The budget of the idle comparison of a resident engine: a job that cannot finish in it gives up.
            result = engine.verify_incremental(session_id=session_id, provider=provider, path=path,
                                               _deadline=time.monotonic() + engine.settings.request_timeout_sec)
        finally:
            engine.close()
    outcome = "mismatch" if result.get("mismatch") else "compared" if result.get("compared") else "not_compared"
    if provider == "claude":
        from self_directing_mcp.claude_hook_state import ClaudeHookState
        ClaudeHookState(index_dir).verifier_done(f"{provider}:{session_id}", outcome != "not_compared")
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m self_directing_mcp.verify_job")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--path", default=None)
    parser.add_argument("--index-dir", required=True)
    args = parser.parse_args(argv)
    try:
        run(args.provider, args.session_id, args.path, args.index_dir)
    except Exception as exc:                # nothing waits for this process and nothing is printed: the class goes to stderr
        if args.provider == "claude":
            from self_directing_mcp.claude_hook_state import ClaudeHookState
            ClaudeHookState(args.index_dir).verifier_done(f"{args.provider}:{args.session_id}", False)
        print(type(exc).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
