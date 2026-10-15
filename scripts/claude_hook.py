#!/usr/bin/env python3
"""Claude Code command-hook entry point; importing it never reads stdin or re-execs Python.

Always exits 0 with a JSON response (exit code 2 would block). If the package cannot be imported, a standard-
library fallback answers fail-open with a visible warning and records ``hook_health``.

With ``SELF_DIRECT_DAEMON=on`` (Linux; experimental, see README "Resident daemon") this script is a thin client of the
resident daemon: it imports only the standard library and ``self_directing_mcp.daemon.client``, and the answer is the
daemon's or a visible fail-open (``daemon_unavailable``), never an in-process judgement.
"""
import json
import os
import sys

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


_GROK_ENVELOPE = ("workspaceRoot", "sessionId", "toolUseId", "permissionMode")


def _segments(path):
    return [part for part in path.replace("\\", "/").split("/") if part]


def _is_grok_payload(text):
    """Whether ``text`` (the hook's stdin) is a Grok CLI payload. Standard library only, and cheap: a payload with
    no sign of Grok is not even parsed. Keep in step with ``self_directing_mcp.grok_ids.is_grok_payload`` (a test
    runs both over the same payloads).

    Grok loads Claude Code's settings file by default, so this hook, installed there, is also run on Grok's events
    with Grok's payload: a transcript outside the Claude projects root, which this hook would reject loudly on every
    tool call without protecting anything. Grok has its own hook (``grok_hook.py``).
    """
    if ".grok" not in text.lower() and "hookEventName" not in text:
        return False
    try:
        payload = json.loads(text)
    except ValueError:
        return False
    if not isinstance(payload, dict):
        return False
    path = next((payload[key] for key in ("transcript_path", "transcriptPath")
                 if isinstance(payload.get(key), str) and payload[key]), None)
    if path is not None:
        parts = [part.lower() for part in _segments(path)]
        if any(left == ".grok" and right == "sessions" for left, right in zip(parts, parts[1:])):
            return True
    if not isinstance(payload.get("hookEventName"), str):
        return False
    if path is not None and any(part.lower() == ".claude" for part in _segments(path)):
        return False
    return any(isinstance(payload.get(key), str) for key in _GROK_ENVELOPE)


def _reexec_target():
    """The repository's own interpreter when this process is not it (the hook then re-executes), else None."""
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    venv = os.path.join(root, ".venv")
    python = os.path.join(venv, "Scripts/python.exe" if os.name == "nt" else "bin/python")
    if os.path.isfile(python) and os.path.realpath(sys.prefix) != os.path.realpath(venv):
        return python
    return None


def _bypass_grok():
    """Answer a Grok payload with silence: nothing printed, nothing written, no health record, exit code 0.

    stdin is consumed here and replaced by a replayable copy, so every path below reads the same payload. When this
    process is about to re-execute into the repository's interpreter it does not read stdin (the next process would
    find it empty), and the check happens there instead.
    """
    if _reexec_target() is not None:
        return False
    try:
        text = sys.stdin.read()
    except Exception:
        text = ""
    if _is_grok_payload(text):
        return True
    import io
    sys.stdin = io.StringIO(text)
    return False


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
    argv = sys.argv[1:]
    event = _event(argv, None)
    try:
        try:
            payload = json.load(sys.stdin)
        except Exception as exc:
            from self_directing_mcp import claude_hook_state as state
            response, served = state.fail_open(event, None, type(exc).__name__, state.CHECK_FAILED), False
        else:
            event = _event(argv, payload)
            response, served = client.serve_hook("claude", event, payload, argv,
                                                 python=python if os.path.isfile(python) else None)
        json.dump(response if served else _mask(response), sys.stdout)
        sys.stdout.write("\n")
    except Exception as exc:
        # The daemon is in use, so there is no standalone fallback: a visible fail-open and nothing else.
        reason = (f"Tsukkomi audit UNAVAILABLE (daemon_unavailable:{type(exc).__name__}): this {event or 'hook'} "
                  "was not verified; fail-open, not a compliance verdict.")
        print(reason, file=sys.stderr)
        print(json.dumps(_minimal_fail_open(event, reason)))
    return True


def main():
    if _bypass_grok():
        return
    _apply_index_dir(sys.argv[1:])
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
