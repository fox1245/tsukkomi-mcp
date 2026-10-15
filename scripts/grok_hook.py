#!/usr/bin/env python3
"""Grok CLI command-hook entry point; importing it never reads stdin or re-execs Python.

Never exits 2 (Grok treats exit code 2 as an explicit deny). A judgement exits 0 with a JSON reply. If anything fails
the reply still allows (PreToolUse answers an explicit ``{"decision": "allow"}``, never ``{}``), the first line of
stderr names the failure and the exit code is 1, which Grok records as a failed hook ("exit code 1: <first stderr
line>") without blocking. If the package cannot be imported, a standard-library fallback does the same and records
``hook_health``. There is no resident-daemon mode for Grok.
"""
import json
import os
import sys

EVENTS = ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PostToolUseFailure", "Stop")
MODES = ("advisory", "confirmed_only", "enforced")


def _mask(response):
    """Best effort: the fallback only emits class names and contract ids, but nothing raw should leave unmasked."""
    try:
        from self_directing_mcp.security.mask import mask_value
        return mask_value(response)
    except Exception:
        return response


def _minimal_fail_open(event, reason):
    """Last resort when even the standard-library helper cannot be imported. Never deny, never block."""
    if event == "PreToolUse":
        return {"decision": "allow", "systemMessage": reason}
    return {"systemMessage": reason}


def _fail_open(root, event, payload, error_class):
    """Fail open with a visible warning. Uses the stdlib-only state module when it can be imported."""
    try:
        sys.path.insert(0, str(root / "src"))
        from self_directing_mcp.grok_hook_state import AUDIT_UNAVAILABLE, fail_open
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


def _apply_flags(argv):
    """``--index-dir=PATH`` and ``--mode=`` from the registered command win; the fallback honours them too."""
    for arg in argv:
        if arg.startswith("--index-dir=") and os.path.isabs(arg.split("=", 1)[1]):
            os.environ["SELF_DIRECT_INDEX_DIR"] = arg.split("=", 1)[1]
        elif arg.startswith("--mode=") and arg.split("=", 1)[1] in MODES:
            os.environ["SELF_DIRECT_GROK_ENFORCEMENT"] = arg.split("=", 1)[1]


def main():
    _apply_flags(sys.argv[1:])
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    venv = root / ".venv"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    try:
        if python.is_file() and Path(sys.prefix).resolve() != venv.resolve():
            os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
        sys.path.insert(0, str(root / "src"))
        from self_directing_mcp.grok_hooks import main as run_hook
        return run_hook()
    except Exception as exc:
        # This branch also covers package import and re-exec failure. Only the error class is reported:
        # messages can carry transcript data or credentials, and the stdin payload is never echoed.
        try:
            payload = json.load(sys.stdin)
        except Exception:
            payload = None
        event = _event(sys.argv[1:], payload)
        response = _mask(_fail_open(root, event, payload, type(exc).__name__))
        print(json.dumps(response))
        return 0 if response.get("decision") == "ask" else 1


if __name__ == "__main__":
    sys.exit(main())
