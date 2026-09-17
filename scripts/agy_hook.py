#!/usr/bin/env python3
"""Executable entry point for Antigravity (AGY) lifecycle hooks.

Configured in .agents/hooks.json or global hooks.json.
"""
import os
import sys
from pathlib import Path

# Dynamically resolve repository root and virtualenv via pathlib
repo_root = Path(__file__).resolve().parent.parent
venv_dir = repo_root / ".venv"
venv_python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

# Auto re-exec inside the managed virtualenv if invoked with outside python
in_venv = Path(sys.prefix).resolve() == venv_dir.resolve()
if venv_python.is_file() and not in_venv:
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve())] + sys.argv[1:])

src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from self_directing_mcp.agy_hooks import main
    if __name__ == "__main__":
        main()
except Exception as exc:
    import json
    # Always exit 0 with safe JSON output so hooks never block the agent on unexpected errors
    json.dump({"decision": "allow", "reason": f"Self-directing MCP hook fallback: {type(exc).__name__}: {exc}"}, sys.stdout)
    sys.stdout.write("\n")
    sys.exit(0)

