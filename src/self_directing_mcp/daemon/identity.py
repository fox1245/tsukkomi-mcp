"""The code identity of this installation of the package. Standard library only (a hook client computes it).

The package version names a release; it does not name the code that runs. Two installations can carry the same version
string and different files (an older copy in another virtual environment, a checkout on another branch), and a daemon
started by one must not serve a client of the other. The identity is a SHA-256 digest over every ``.py`` file of the
package directory: for each file, in the order of its relative path, the relative path (POSIX separators, UTF-8), a
NUL byte, the file's length as eight big-endian bytes and its content. The length makes the framing unambiguous (a
file ``a`` holding ``bc`` and a file ``ab`` holding ``c`` digest differently). ``__pycache__`` is not part of it, and
neither is any file that is not Python source.

It is computed from the files once per process and then remembered. When any file cannot be read there is no identity
(``None``), and an absent identity equals nothing, not even another absent one: a process that cannot name its code is
never treated as running the same code as anybody.
"""
from __future__ import annotations

import os

try:
    from _sha2 import sha256       # CPython's own SHA-256: the hashlib wrapper costs about 1.7 ms of import in every hook process
except ImportError:                # (an interpreter without the C module)
    from hashlib import sha256

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_SUFFIX = ".py"
SKIPPED_DIRECTORY = "__pycache__"
LENGTH_BYTES = 8

_computed = False
_identity: str | None = None


def source_files(root: str) -> list[tuple[str, str]]:
    """``(relative path with POSIX separators, absolute path)`` of every ``.py`` file under ``root``, sorted by relative
    path. Raises ``OSError`` when a directory cannot be listed."""
    found: list[tuple[str, str]] = []
    pending = [("", root)]
    while pending:
        prefix, directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = prefix + entry.name
                if entry.is_dir(follow_symlinks=False):
                    if entry.name != SKIPPED_DIRECTORY:
                        pending.append((relative + "/", entry.path))
                elif entry.name.endswith(SOURCE_SUFFIX) and entry.is_file():
                    found.append((relative, entry.path))
    found.sort(key=lambda item: item[0].encode("utf-8"))
    return found


def digest_of(root: str) -> str | None:
    """The identity of the package directory ``root`` (hex digest), or ``None`` when it cannot be read or holds no source."""
    try:
        files = source_files(root)
        if not files:
            return None
        digest = sha256()
        for relative, path in files:
            with open(path, "rb") as stream:
                content = stream.read()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(len(content).to_bytes(LENGTH_BYTES, "big"))
            digest.update(content)
        return digest.hexdigest()
    except OSError:
        return None


def code_identity() -> str | None:
    """This process's identity, computed on first use."""
    global _computed, _identity
    if not _computed:
        _identity, _computed = digest_of(PACKAGE_ROOT), True
    return _identity


def same_code(own: str | None, other: str | None) -> bool:
    """Equal identities; an absent one is equal to nothing."""
    return own is not None and other is not None and own == other
