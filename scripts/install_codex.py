#!/usr/bin/env python3
"""One-command Codex installation. Requires Python 3.12+ and Codex on PATH."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run(argv, **kwargs):
    completed = subprocess.run(
        [str(value) for value in argv], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0, **kwargs)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    completed.check_returncode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex-home", type=Path, default=Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")))
    parser.add_argument("--codex", default=shutil.which("codex"))
    parser.add_argument("--env-file", type=Path, help="Authorized dotenv file containing OPENROUTER_API_KEY")
    parser.add_argument("--vector-archive", type=Path, help="Official sqlite-vector release ZIP for offline setup")
    parser.add_argument("--trust-hooks", action="store_true",
                        help="Trust only this package's three exact reviewed hooks")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use only cached Python packages (uv)")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error("Run this script with Python 3.12 or newer")
    if not args.codex:
        parser.error("Codex CLI is not on PATH; install Codex or pass --codex")
    source = Path(__file__).resolve().parents[1]
    home = args.codex_home.expanduser().resolve()
    installation = home / "mcp-servers" / "self-directing-mcp"
    environment = installation / "venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if args.dry_run:
        print(json.dumps({"source": str(source), "environment": str(environment),
                          "files": [str(home / name) for name in ("config.toml", "hooks.json", "AGENTS.md")],
                          "trust_hooks": args.trust_hooks, "dry_run": True}, indent=2))
        return
    if not args.verify:
        installation.mkdir(parents=True, exist_ok=True)
        uv = shutil.which("uv")
        if not python.exists():
            if environment.exists() and any(environment.iterdir()):
                parser.error("Existing incomplete venv; refusing to replace unknown files")
            if uv:
                run([uv, "venv", "--python", sys.executable, environment])
            else:
                run([sys.executable, "-m", "venv", environment])
        if uv:
            command = [uv, "pip", "install", "--python", python]
            if args.offline:
                command.append("--offline")
            run([*command, source])
        else:
            if args.offline:
                parser.error("--offline requires uv")
            run([python, "-m", "ensurepip"])
            run([python, "-m", "pip", "install", source])
    if not python.exists():
        parser.error("Managed environment not installed")
    child_env = os.environ.copy()
    if not args.verify:
        key_file = (args.env_file or source / ".env").expanduser().resolve()
        if not key_file.is_file():
            parser.error("A real OpenRouter key file is required; create .env or pass --env-file")
        # Validation prints neither key nor file contents.
        run([python, "-c", "from self_directing_mcp.config import Settings; import sys; Settings(_env_file=None, openrouter_api_key_file=sys.argv[1]).resolve_api_key()", key_file])
        child_env["SELF_DIRECT_OPENROUTER_API_KEY_FILE"] = str(key_file)
        native = installation / "native"
        extension = native / ("vector.dll" if os.name == "nt" else "vector.dylib" if sys.platform == "darwin" else "vector.so")
        if not args.offline or args.vector_archive:
            setup = [python, source / "scripts/setup_sqlite_vector.py", "--output-dir", native]
            if args.vector_archive:
                setup += ["--archive", args.vector_archive]
            run(setup)
        elif not extension.is_file():
            parser.error("--offline requires an installed native library or --vector-archive")
        # Verify cached libraries too; fail before registration on incompatible binaries.
        run([python, "-c", "import sys; from setup_sqlite_vector import verify; verify(sys.argv[1])", extension], cwd=source / "scripts")
    command = [python, "-m", "self_directing_mcp.install_codex",
               "--codex-home", home, "--codex", args.codex]
    if args.trust_hooks:
        command.append("--trust-hooks")
    if args.verify:
        command.append("--verify")
    run(command, cwd=home, env=child_env)
    if not args.verify:
        marker = {"source": str(source), "python": str(python), "server": "self-directing-mcp"}
        (installation / "installation.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
