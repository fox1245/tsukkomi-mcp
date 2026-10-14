#!/usr/bin/env python3
"""AGY command entry point; importing it never reads stdin or re-execs Python."""
import json
import os
import sys
from pathlib import Path

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


def main():
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
