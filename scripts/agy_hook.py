#!/usr/bin/env python3
"""AGY command entry point; importing it never reads stdin or re-execs Python."""
import json
import os
import sys
from pathlib import Path


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
        # This branch also covers package import and re-exec failure. It uses
        # only the standard library so missing native dependencies cannot allow.
        event = next((a.split("=", 1)[1] if a.startswith("--event=") else a
                      for a in sys.argv[1:] if a.startswith("--event=") or a in
                      ("PreToolUse", "PostToolUse", "PreInvocation", "Stop")), None)
        try:
            payload = json.load(sys.stdin)
        except Exception:
            payload = None
        reason = f"Tsukkomi runtime unavailable ({type(exc).__name__})"
        if event == "PostToolUse":
            response = {}
        elif event == "PreInvocation":
            response = {"injectSteps": [{"ephemeralMessage": reason + "; do not claim compliance."}]}
        elif event == "Stop":
            resume = isinstance(payload, dict) and payload.get("terminationReason") in ("model_stop", "NO_TOOL_CALL") and payload.get("fullyIdle") is True
            response = {"decision": "continue" if resume else "allow", "reason": reason}
        else:
            response = {"decision": "deny", "reason": reason}
        print(reason, file=sys.stderr)
        print(json.dumps(response))


if __name__ == "__main__":
    main()
