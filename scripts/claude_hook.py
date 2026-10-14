#!/usr/bin/env python3
"""Claude Code command-hook entry point; importing it never reads stdin or re-execs Python.

Always exits 0 with a JSON response (exit code 2 would block). If the package cannot be imported, a standard-
library fallback answers fail-open with a visible warning and records ``hook_health``.
"""
import json
import os
import sys
from pathlib import Path

EVENTS = ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PostToolUseFailure",
          "Stop", "SubagentStop")
CONTEXT_EVENTS = ("SessionStart", "UserPromptSubmit")


def _mask(response):
    """Best effort: the fallback only emits class names and contract ids, but nothing raw should leave unmasked."""
    try:
        from self_directing_mcp.security.mask import mask_value
        return mask_value(response)
    except Exception:
        return response


def _minimal_fail_open(event, reason):
    """Last resort when even the standard-library helper cannot be imported. Never deny, never block."""
    if event in CONTEXT_EVENTS:
        text = reason + " Contract context was not injected; do not claim compliance."
        return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}, "systemMessage": reason}
    return {"systemMessage": reason}


def _fail_open(root, event, payload, error_class):
    """Fail open with a visible warning. Uses the stdlib-only state module when it can be imported."""
    try:
        sys.path.insert(0, str(root / "src"))
        from self_directing_mcp.claude_hook_state import AUDIT_UNAVAILABLE, fail_open
        return fail_open(event, payload, error_class, AUDIT_UNAVAILABLE)
    except Exception:
        reason = (f"Tsukkomi audit UNAVAILABLE ({error_class}): this {event or 'hook'} was not verified; "
                  "fail-open, not a compliance verdict.")
        print(reason, file=sys.stderr)
        return _minimal_fail_open(event, reason)


def _event(argv, payload):
    event = next((a.split("=", 1)[1] if a.startswith("--event=") else a
                  for a in argv if a.startswith("--event=") or a in EVENTS), None)
    if event is None and isinstance(payload, dict) and payload.get("hook_event_name") in EVENTS:
        event = payload["hook_event_name"]
    return event


def _apply_index_dir(argv):
    """``--index-dir=PATH`` from the registered command wins; the fallback below records health there too."""
    for arg in argv:
        if arg.startswith("--index-dir=") and os.path.isabs(arg.split("=", 1)[1]):
            os.environ["SELF_DIRECT_INDEX_DIR"] = arg.split("=", 1)[1]


def _via_daemon(root, python):
    """Resident-daemon mode (``SELF_DIRECT_DAEMON=on``, Linux): a thin client that imports only the standard library
    and the daemon's client module. Returns False, having read nothing, when the daemon is not in use. With the
    daemon in use the answer is the daemon's or a visible fail-open; there is no in-process fallback."""
    try:
        sys.path.insert(0, str(root / "src"))
        from self_directing_mcp.daemon import client
        if not client.use_daemon():
            return False
    except Exception:
        return False        # the package is unusable: the standalone path below reports it, fail-open
    argv = sys.argv[1:]
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        from self_directing_mcp import claude_hook_state as state
        event = _event(argv, None)
        response, served = state.fail_open(event, None, type(exc).__name__, state.CHECK_FAILED), False
    else:
        response, served = client.serve_hook("claude", _event(argv, payload), payload, argv,
                                             python=str(python) if python.is_file() else None)
    json.dump(response if served else _mask(response), sys.stdout)
    sys.stdout.write("\n")
    return True


def main():
    _apply_index_dir(sys.argv[1:])
    root = Path(__file__).resolve().parent.parent
    venv = root / ".venv"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if _via_daemon(root, python):
        return
    try:
        if python.is_file() and Path(sys.prefix).resolve() != venv.resolve():
            os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
        sys.path.insert(0, str(root / "src"))
        from self_directing_mcp.claude_hooks import main as run_hook
        run_hook()
    except Exception as exc:
        # This branch also covers package import and re-exec failure. Only the error class is reported:
        # messages can carry transcript data or credentials, and the stdin payload is never echoed.
        try:
            payload = json.load(sys.stdin)
        except Exception:
            payload = None
        event = _event(sys.argv[1:], payload)
        print(json.dumps(_mask(_fail_open(root, event, payload, type(exc).__name__))))


if __name__ == "__main__":
    main()
