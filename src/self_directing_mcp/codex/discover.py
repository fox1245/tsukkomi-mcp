from __future__ import annotations

import re
import os
from pathlib import Path, PureWindowsPath

# rollout-YYYY-MM-DDThh-mm-ss-<uuid>.jsonl (uuid may be dashed)
_UUID_RE = re.compile(
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)
_ROLLOUT_RE = re.compile(r"^rollout-.*\.jsonl$", re.IGNORECASE)


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
    m = _UUID_RE.search(path.name)
    return m.group(1).lower() if m else None


def discover_session_files(sessions_dir: Path) -> list[Path]:
    """Find Codex rollout JSONL files under YYYY/MM/DD/ layout (or flat)."""
    root = Path(sessions_dir)
    if not root.exists():
        return []
    found: list[Path] = []
    for p in root.rglob("*.jsonl"):
        if _ROLLOUT_RE.match(p.name) or p.suffix.lower() == ".jsonl":
            if p.is_file():
                found.append(p)
    return sorted(found)


def find_session_by_id(sessions_dir: Path, session_id: str) -> Path | None:
    """Resolve session by UUID in filename (preferred) under sessions root."""
    root = Path(sessions_dir)
    sid = session_id.strip().lower()
    if not sid:
        return None
    for p in discover_session_files(root):
        fid = session_id_from_filename(p)
        if fid and fid == sid:
            return ensure_under_root(p, root)
    # Fallback: scan session_meta.payload.id (lazy import to avoid cycles)
    from self_directing_mcp.codex.parse import peek_session_id

    for p in discover_session_files(root):
        try:
            ensure_under_root(p, root)
        except PathTraversalError:
            continue
        meta_id = peek_session_id(p)
        if meta_id and meta_id.lower() == sid:
            return p.resolve()
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
