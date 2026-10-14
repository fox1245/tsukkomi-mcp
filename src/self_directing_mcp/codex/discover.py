from __future__ import annotations

import re
import os
from pathlib import Path, PureWindowsPath

from self_directing_mcp.compressed_source import SUFFIX, CompressedSourceError, is_compressed

# rollout-YYYY-MM-DDThh-mm-ss-<uuid>.jsonl (uuid may be dashed). Codex compresses older rollouts in place to
# rollout-...-<uuid>.jsonl.zst, and a forked or sub-agent rollout can carry a second "_<uuid>" suffix.
_UUID_RE = re.compile(
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)
_ROLLOUT_RE = re.compile(r"^rollout-.*\.jsonl$", re.IGNORECASE)
_ROLLOUT_ZST_RE = re.compile(r"^rollout-.*\.jsonl\.zst$", re.IGNORECASE)


class PathTraversalError(ValueError):
    """Raised when a session path escapes the configured sessions root."""


def _windows_comparison_path(path: Path) -> PureWindowsPath:
    """Compare resolved DOS/UNC paths in the same extended namespace.

    Preserve the original resolved path for I/O. Removing a prefix before
    resolving or opening can change long-path and trailing-dot semantics.
    Device namespaces are not session-file locations.
    """
    value = str(path)
    if value.startswith("\\\\?\\UNC\\"):
        return PureWindowsPath(value)
    if re.match(r"^\\\\\?\\[A-Za-z]:\\", value):
        return PureWindowsPath(value)
    if value.startswith(("\\\\?\\", "\\\\.\\")):
        raise PathTraversalError("unsupported Windows device namespace")
    if re.match(r"^[A-Za-z]:\\", value):
        return PureWindowsPath("\\\\?\\" + value)
    if value.startswith("\\\\"):
        return PureWindowsPath("\\\\?\\UNC\\" + value[2:])
    raise PathTraversalError("expected an absolute Windows filesystem path")


def ensure_under_root(path: Path, root: Path) -> Path:
    """Resolve path and ensure it is under root (path traversal guard)."""
    root_r = root.resolve()
    path_r = path.expanduser().resolve()
    root_compare, path_compare = root_r, path_r
    if os.name == "nt":
        root_compare = _windows_comparison_path(root_r)
        path_compare = _windows_comparison_path(path_r)
    try:
        path_compare.relative_to(root_compare)
    except ValueError as e:
        raise PathTraversalError(
            f"path {path_r} is outside sessions root {root_r}"
        ) from e
    return path_r


def session_id_from_filename(path: Path) -> str | None:
    """The one UUID in a rollout file name. Two UUIDs (a ``_<uuid>`` suffix) do not say which one is the
    session, so the answer is None and the ``session_meta`` header decides."""
    found = {m.lower() for m in _UUID_RE.findall(path.name)}
    return next(iter(found)) if len(found) == 1 else None


def _discovery_key(path: Path):
    # A plain rollout sorts before its compressed twin: while both exist the plain file is the live one.
    compressed = is_compressed(path)
    return (str(path)[:-len(SUFFIX)] if compressed else str(path), compressed)


def discover_session_files(sessions_dir: Path) -> list[Path]:
    """Find Codex rollout files under YYYY/MM/DD/ layout (or flat): ``*.jsonl`` and ``rollout-*.jsonl.zst``."""
    root = Path(sessions_dir)
    if not root.exists():
        return []
    found: list[Path] = []
    for p in root.rglob("*.jsonl"):
        if _ROLLOUT_RE.match(p.name) or p.suffix.lower() == ".jsonl":
            if p.is_file():
                found.append(p)
    for p in root.rglob("*" + SUFFIX):
        if _ROLLOUT_ZST_RE.match(p.name) and p.is_file():
            found.append(p)
    return sorted(found, key=_discovery_key)


def find_session_by_id(sessions_dir: Path, session_id: str) -> Path | None:
    """Resolve session by UUID in filename (preferred) under sessions root.

    A compressed rollout whose header cannot be inspected is never skipped silently: when no other file
    matches, the failure is raised as a ``CompressedSourceError`` instead of "not found".
    """
    root = Path(sessions_dir)
    sid = session_id.strip().lower()
    if not sid:
        return None
    files = discover_session_files(root)
    for p in files:
        fid = session_id_from_filename(p)
        if fid and fid == sid:
            return ensure_under_root(p, root)
    # Fallback: scan session_meta.payload.id (lazy import to avoid cycles). Names with no single UUID
    # (the "_<uuid>" suffix form) are the likeliest matches, so they go first.
    from self_directing_mcp.codex.parse import read_session_id

    unreadable: list[CompressedSourceError] = []
    for p in sorted(files, key=lambda q: session_id_from_filename(q) is not None):
        try:
            ensure_under_root(p, root)
        except PathTraversalError:
            continue
        try:
            meta_id = read_session_id(p)
        except CompressedSourceError as exc:
            unreadable.append(exc)
            continue
        except OSError:
            continue
        if meta_id and meta_id.lower() == sid:
            return p.resolve()
    if unreadable:
        raise CompressedSourceError(
            unreadable[0].code,
            f"{len(unreadable)} compressed session file(s) under the sessions root could not be inspected, so "
            f"session {sid} may be among them: {unreadable[0]}")
    return None


def resolve_session_path(
    sessions_dir: Path,
    *,
    session_id: str | None = None,
    path: str | Path | None = None,
) -> Path:
    """Resolve a session file by id or explicit path; always under sessions_dir."""
    root = Path(sessions_dir)
    if path is not None:
        return ensure_under_root(Path(path), root)
    if session_id:
        found = find_session_by_id(root, session_id)
        if found is None:
            raise FileNotFoundError(f"session not found for id={session_id!r} under {root}")
        return found
    raise ValueError("session_id or path is required")
