"""Fail-closed Linux process boundary for approved workflow programs.

Only the selected workspace and explicit runtime files are visible. Child output
is internal evidence, never a public diagnostic: it may contain hostile text.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time

from self_directing_mcp import config
from self_directing_mcp.config import Settings


class SandboxError(RuntimeError):
    pass


def _environment() -> dict[str, str]:
    return {"PATH": "/usr/bin:/bin", "HOME": "/home/sandbox", "TMPDIR": "/tmp",
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "LEAN_PATH": "/work",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}


def _runtime_paths(extra: tuple[Path, ...], workspace: Path) -> list[Path]:
    # Do not mount /, /home, /etc, a repository parent, or a whole venv parent.
    paths = {Path(p) for p in ("/usr/bin", "/usr/lib", "/usr/lib64", "/bin", "/lib", "/lib64")
             if Path(p).exists()}
    paths.add(Path(sysconfig.get_path("stdlib")))
    paths.add(Path(sysconfig.get_path("purelib")))
    paths.add(Path(sysconfig.get_path("platlib")))
    for executable in (Path(sys.executable), Path(getattr(sys, "_base_executable", sys.executable))):
        paths.update((executable, executable.resolve()))
    if sys.prefix != sys.base_prefix:
        paths.add(Path(sys.prefix) / "pyvenv.cfg")
    libdir = Path(sysconfig.get_config_var("LIBDIR") or sys.base_prefix)
    paths.update(libdir.glob("libpython*.so*"))
    paths.update(Path(path).expanduser().absolute() for path in extra)
    sensitive = [Path.home().resolve()]
    sensitive.extend(Path(value).expanduser().resolve() for key in
                     ("TSUKKOMI_WORKFLOW_STATE", "TSUKKOMI_WORKFLOW_APPROVAL")
                     if (value := os.environ.get(key)))
    # Keep dotenv protection independent of explicit key-file selection.
    sensitive.extend(path.resolve() for path in
                     (Path.cwd() / ".env", config._repo_root() / ".env")
                     if path.is_file())
    sensitive.append(config.get_config_path())
    key_file = Settings().openrouter_api_key_file
    if key_file is not None:
        sensitive.append(key_file.expanduser().resolve())
    if any(item == workspace or item.is_relative_to(workspace) for item in sensitive):
        raise SandboxError("sandbox_workspace_exposes_host_state")
    protected = [*sensitive, workspace]
    result = []
    for path in sorted(paths, key=str):
        if not path.exists():
            raise SandboxError("sandbox_runtime_missing")
        real = path.resolve()
        if real == Path("/") or any(item == real or item.is_relative_to(real) for item in protected):
            raise SandboxError("sandbox_runtime_exposes_host_state")
        result.append(path)
    return result


def _mapped(argument: str, workspace: Path) -> str:
    prefix = str(workspace)
    if argument == prefix:
        return "/work"
    if argument.startswith(prefix + "/"):
        return "/work/" + argument[len(prefix) + 1:]
    return argument


def _output(stream) -> tuple[str, str]:
    stream.seek(0)
    digest = hashlib.sha256()
    chunks = []
    remaining = 1024 * 1024
    while chunk := stream.read(65536):
        digest.update(chunk)
        if remaining:
            chunks.append(chunk[:remaining])
            remaining = max(0, remaining - len(chunk))
    return b"".join(chunks).decode("utf-8", errors="replace"), digest.hexdigest()


def _not_started(status: str) -> dict:
    return {"status": status, "exit_code": None, "output_hash": hashlib.sha256(b"").hexdigest(),
            "diagnostics": status, "started": False, "outcome_unknown": False,
            "stdout": "", "stderr": ""}


def run_process(argv: list[str], *, workspace: Path, writable: bool, timeout_sec: float,
                cancel_event=None, read_only_paths: tuple[Path, ...] = ()) -> dict:
    """Run in a private filesystem/network/PID namespace, or do not run at all.

    `started` means exec was observed, or may have occurred before interruption.
    `outcome_unknown` distinguishes that latter case; no claim about side effects
    is possible on cancellation, timeout, or loss of the launch-status channel.
    """
    if cancel_event is not None and cancel_event.is_set():
        return _not_started("cancelled")
    if timeout_sec <= 0:
        return _not_started("timeout")
    if not sys.platform.startswith("linux"):
        return _not_started("sandbox_unavailable")
    requested = os.environ.get("TSUKKOMI_BWRAP", "bwrap")
    bwrap = shutil.which(requested)
    if bwrap is None:
        return _not_started("sandbox_unavailable")
    workspace = Path(workspace).absolute()
    if not workspace.is_dir() or workspace.resolve() != workspace:
        return _not_started("sandbox_workspace_invalid")
    if not argv or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in argv):
        return _not_started("spawn_failed")
    try:
        runtimes = _runtime_paths(read_only_paths, workspace)
    except (OSError, SandboxError) as exc:
        return _not_started(str(exc) if isinstance(exc, SandboxError) else "sandbox_runtime_invalid")
    env = _environment()
    bins = [str(Path(path).absolute() / "bin") for path in read_only_paths if (Path(path) / "bin").is_dir()]
    if bins:
        env["PATH"] = ":".join([*bins, env["PATH"]])
    executable = argv[0]
    if "/" not in executable:
        executable = shutil.which(executable, path=env["PATH"])
        if executable is None:
            return _not_started("spawn_failed")
    elif not Path(executable).is_absolute():
        executable = str(workspace / executable)
    if not Path(executable).is_file() or not os.access(executable, os.X_OK):
        return _not_started("spawn_failed")
    command = [_mapped(executable, workspace), *(_mapped(arg, workspace) for arg in argv[1:])]
    deadline = time.monotonic() + timeout_sec
    with tempfile.TemporaryFile() as status_file, tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        sandbox = [str(Path(bwrap).resolve()), "--unshare-all", "--die-with-parent", "--cap-drop", "ALL",
                   "--clearenv", "--json-status-fd", str(status_file.fileno()),
                   "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                   "--dir", "/home/sandbox"]
        for path in runtimes:
            sandbox.extend(("--ro-bind", str(path), str(path)))
        sandbox.extend(("--bind" if writable else "--ro-bind", str(workspace), "/work", "--chdir", "/work"))
        for key, value in env.items():
            sandbox.extend(("--setenv", key, value))
        sandbox.extend(("--", *command))
        if cancel_event is not None and cancel_event.is_set():
            return _not_started("cancelled")
        if time.monotonic() >= deadline:
            return _not_started("timeout")
        try:
            process = subprocess.Popen(sandbox, cwd="/", env=env, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, pass_fds=(status_file.fileno(),),
                                       start_new_session=True)
        except OSError:
            return _not_started("spawn_failed")
        interrupted = None
        try:
            while process.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    interrupted = "cancelled"
                    break
                if time.monotonic() >= deadline:
                    interrupted = "timeout"
                    break
                time.sleep(min(.02, max(0, deadline - time.monotonic())))
        finally:
            # Killing the namespace's init also kills descendants which called
            # setsid; killing the host process group catches the bwrap monitor.
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
        status_file.seek(0)
        records = []
        try:
            records = [json.loads(line) for line in status_file.read(65536).splitlines()]
        except (ValueError, UnicodeError):
            pass
        exits = [record["exit-code"] for record in records if isinstance(record, dict) and "exit-code" in record]
        observed_exec = len(exits) == 1 and isinstance(exits[0], int)
        child_created = any(isinstance(record, dict) and "child-pid" in record for record in records)
        unknown = interrupted is not None or (not observed_exec and (child_created or process.returncode <= 0))
        started = observed_exec or unknown
        status = interrupted or ("passed" if observed_exec and exits[0] == 0 else
                                 "failed" if observed_exec else "sandbox_launch_failed")
        out, out_hash = _output(stdout)
        err, err_hash = _output(stderr)
        return {"status": status, "exit_code": exits[0] if observed_exec else None,
                "output_hash": hashlib.sha256((out_hash + err_hash).encode()).hexdigest(),
                "diagnostics": "" if status == "passed" else status,
                "started": started, "outcome_unknown": unknown, "stdout": out, "stderr": err}
