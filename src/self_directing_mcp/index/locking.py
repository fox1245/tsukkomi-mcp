"""Inter-process lease for one local index directory."""
from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import time

from self_directing_mcp.request_control import IndexBusy, check_request


@contextmanager
def index_lock(directory: Path, timeout: float = 15):
    check_request()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".engine.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        deadline = time.monotonic() + timeout
        while True:
            check_request()
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                # POSIX flock reports EAGAIN/EACCES for a conflicting lease.
                # Windows locking may report its lock-violation winerror.
                contention = exc.errno in {errno.EACCES, errno.EAGAIN}
                if os.name == "nt" and getattr(exc, "winerror", None) is not None:
                    contention = exc.winerror == 33
                if not contention:
                    raise
                check_request()
                if time.monotonic() >= deadline:
                    raise IndexBusy() from exc
                time.sleep(min(0.02, max(0, deadline - time.monotonic())))
        try:
            check_request()
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
