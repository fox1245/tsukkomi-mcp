"""The daemon's runtime directory and the files in it. Standard library only (clients import it).

One directory per user: ``$XDG_RUNTIME_DIR/tsukkomi/`` when that variable names a usable directory, else
``~/.cache/tsukkomi/run/``. It is created with mode 0700 and verified on every use: it must be a real directory
(never a symbolic link), owned by the current user, with mode exactly 0700. Otherwise ``RuntimeDirError`` is raised
and nothing is created or reused: a directory somebody else could write to would let them replace the socket.

The verified directory is held open as a file descriptor. Files are opened relative to it (``dir_fd``), and the
socket address goes through ``/proc/self/fd/<fd>/`` when the plain path would exceed the ~107-byte ``sun_path``
limit, so the check and the use cannot be separated by a swapped path component.

Contents (all mode 0600): ``daemon.sock`` (the Unix socket), ``daemon.lock`` (held for the daemon's lifetime),
``daemon.pid``, ``daemon.log`` (event names, sizes, durations, error classes), ``spawn.lock`` and ``spawns.json``
(spawn times, for the crash-loop guard).
"""
from __future__ import annotations

import os
import stat

DIR_MODE = 0o700
SOCKET_NAME = "daemon.sock"
LOCK_NAME = "daemon.lock"
PID_NAME = "daemon.pid"
LOG_NAME = "daemon.log"
SPAWN_LOCK_NAME = "spawn.lock"
SPAWNS_NAME = "spawns.json"
SUN_PATH_SAFE = 100          # sockaddr_un.sun_path holds 108 bytes on Linux, 104 on macOS; stay well below


class RuntimeDirError(RuntimeError):
    """The runtime directory is missing, unusable or unsafe. The message says which and why."""


def current_uid() -> int:
    """The user the directory and every peer must belong to (a single seam for tests)."""
    return os.geteuid()


def _xdg_usable(xdg: str | None) -> str | None:
    if not xdg or not os.path.isabs(xdg):
        return None
    try:
        info = os.stat(xdg)
    except OSError:
        return None
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != current_uid() or info.st_mode & 0o022:
        return None
    return xdg if os.access(xdg, os.W_OK | os.X_OK) else None


def runtime_dir_path(environ=None) -> str:
    environ = os.environ if environ is None else environ
    base = _xdg_usable(environ.get("XDG_RUNTIME_DIR"))
    if base is not None:
        return os.path.join(base, "tsukkomi")
    home = environ.get("HOME") or os.path.expanduser("~")
    return os.path.join(home, ".cache", "tsukkomi", "run")


class RuntimeDir:
    def __init__(self, path: str, fd: int) -> None:
        self.path, self.fd = path, fd

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def open_file(self, name: str, flags: int, mode: int = 0o600) -> int:
        return os.open(name, flags | os.O_CLOEXEC | os.O_NOFOLLOW, mode, dir_fd=self.fd)

    def unlink(self, name: str) -> None:
        try:
            os.unlink(name, dir_fd=self.fd)
        except FileNotFoundError:
            pass

    def exists(self, name: str) -> bool:
        try:
            os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            return True
        except FileNotFoundError:
            return False

    def address(self, name: str = SOCKET_NAME) -> str:
        """Unix-socket address of a file in this directory, safe against over-long runtime paths."""
        plain = os.path.join(self.path, name)
        if len(plain.encode()) <= SUN_PATH_SAFE:
            return plain
        return f"/proc/self/fd/{self.fd}/{name}"


def verify(info: os.stat_result, path) -> None:
    if stat.S_ISLNK(info.st_mode):
        raise RuntimeDirError(f"runtime directory {path} is a symbolic link; refusing to use it")
    if not stat.S_ISDIR(info.st_mode):
        raise RuntimeDirError(f"runtime path {path} is not a directory")
    if info.st_uid != current_uid():
        raise RuntimeDirError(f"runtime directory {path} is not owned by the current user (uid {info.st_uid}); "
                              "refusing to use it")
    mode = stat.S_IMODE(info.st_mode)
    if mode != DIR_MODE:
        raise RuntimeDirError(f"runtime directory {path} has mode {mode:04o}; it must be exactly 0700. "
                              "Fix it with chmod 700 or remove it so that it is recreated")


def open_runtime_dir(environ=None, *, create: bool = True) -> RuntimeDir:
    """Create (when allowed), open and verify the runtime directory. Raises ``RuntimeDirError``."""
    path = runtime_dir_path(environ)
    created = False
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        if not create:
            raise RuntimeDirError(f"runtime directory {path} does not exist") from None
        try:
            os.makedirs(os.path.dirname(path), mode=DIR_MODE, exist_ok=True)
            os.mkdir(path, DIR_MODE)
            created = True
        except FileExistsError:
            pass                                    # raced with another process; verified below
        except OSError as exc:
            raise RuntimeDirError(f"cannot create runtime directory {path}: {type(exc).__name__}") from None
        info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode):
        raise RuntimeDirError(f"runtime directory {path} is a symbolic link; refusing to use it")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise RuntimeDirError(f"cannot open runtime directory {path}: {type(exc).__name__}") from None
    try:
        if created:
            os.fchmod(fd, DIR_MODE)                 # a restrictive umask can only have removed bits; make it exact
        verify(os.fstat(fd), path)                  # the opened inode, not the path
    except BaseException:
        os.close(fd)
        raise
    return RuntimeDir(path, fd)
