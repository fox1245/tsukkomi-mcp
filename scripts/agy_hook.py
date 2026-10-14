#!/usr/bin/env python3
"""AGY command entry point; importing it never reads stdin or re-execs Python."""
import json
import os
import sys

EVENTS = ("PreToolUse", "PostToolUse", "PreInvocation", "Stop")


def _minimal_fail_open(event, reason):
    """Last resort when even the standard-library helper cannot be imported. Never deny, never continue."""
    if event == "PostToolUse":
        return {}
    if event == "PreInvocation":
        return {"injectSteps": [{"ephemeralMessage": reason + " Contract context was not injected; do not claim compliance."}]}
    return {"decision": "allow", "reason": reason}


def _fail_open(root, event, payload, error_class):
    """Fail open with a visible warning. Uses the stdlib-only state module when it can be imported."""
    try:
        sys.path.insert(0, str(root / "src"))
        from self_directing_mcp.agy_hook_state import AUDIT_UNAVAILABLE, fail_open
        return fail_open(event, payload, error_class, AUDIT_UNAVAILABLE)
    except Exception:
        reason = (f"Tsukkomi audit UNAVAILABLE ({error_class}): this {event or 'hook'} was not verified; "
                  "fail-open, not a compliance verdict.")
        print(reason, file=sys.stderr)
        return _minimal_fail_open(event, reason)


def _via_daemon():
    """Resident-daemon mode (``SELF_DIRECT_DAEMON=on``, Linux): a thin client that imports only the standard library
    and the daemon's client module (not even ``pathlib``, whose import costs more than the rest of the client).
    Returns False, having read nothing, when the daemon is not in use. With the daemon in use the answer is the
    daemon's or a visible fail-open; there is no in-process fallback."""
    here = os.path.realpath(__file__)
    root = os.path.dirname(os.path.dirname(here))
    try:
        sys.path.insert(0, os.path.join(root, "src"))
        from self_directing_mcp.daemon import client
        if not client.use_daemon():
            return False
    except Exception:
        return False        # the package is unusable: the standalone path below reports it, fail-open
    python = os.path.join(root, ".venv", "Scripts/python.exe" if os.name == "nt" else "bin/python")
    event = next((a.split("=", 1)[1] if a.startswith("--event=") else a
                  for a in sys.argv[1:] if a.startswith("--event=") or a in EVENTS), None)
    try:
        try:
            payload = json.load(sys.stdin)
        except Exception as exc:
            from self_directing_mcp import agy_hook_state as state
            response = state.fail_open(event, None, type(exc).__name__, state.CHECK_FAILED)
        else:
            response, _served = client.serve_hook("agy", event, payload, sys.argv[1:],
                                                  python=python if os.path.isfile(python) else None)
        json.dump(response, sys.stdout)
        sys.stdout.write("\n")
    except Exception as exc:
        # The daemon is in use, so there is no standalone fallback: a visible fail-open and nothing else.
        reason = (f"Tsukkomi audit UNAVAILABLE (daemon_unavailable:{type(exc).__name__}): this {event or 'hook'} "
                  "was not verified; fail-open, not a compliance verdict.")
        print(reason, file=sys.stderr)
        print(json.dumps(_minimal_fail_open(event, reason)))
    return True


def main():
    if _via_daemon():
        return
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    venv = root / ".venv"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    try:
        if python.is_file() and Path(sys.prefix).resolve() != venv.resolve():
            os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
        sys.path.insert(0, str(root / "src"))
        from self_directing_mcp.agy_hooks import main as run_hook
        run_hook()
    except Exception as exc:
        # This branch also covers package import and re-exec failure. Only the error class is reported:
        # messages can carry transcript data or credentials, and the stdin payload is never echoed.
        event = next((a.split("=", 1)[1] if a.startswith("--event=") else a
                      for a in sys.argv[1:] if a.startswith("--event=") or a in EVENTS), None)
        try:
            payload = json.load(sys.stdin)
        except Exception:
            payload = None
        print(json.dumps(_fail_open(root, event, payload, type(exc).__name__)))


if __name__ == "__main__":
    main()
