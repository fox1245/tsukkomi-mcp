from __future__ import annotations

import os
import re
from pathlib import Path

# Session folder/file stem: UUID or sand-subagent-* (and similar agent ids)
_SESSION_STEM_RE = re.compile(
    r"^(?:"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"|sand-subagent-[A-Za-z0-9._\-]+"
    r"|[A-Za-z0-9._\-]{8,}"
    r")$"
)


class PathTraversalError(ValueError):
    """Raised when a session path escapes the configured transcripts root."""


def ensure_under_root(path: Path, root: Path) -> Path:
    """Resolve path and ensure it is under root (path traversal guard)."""
    root_r = root.resolve()
    path_r = path.expanduser().resolve()
    try:
        path_r.relative_to(root_r)
    except ValueError as e:
        raise PathTraversalError(
            f"path {path_r} is outside transcripts root {root_r}"
        ) from e
    return path_r


def ensure_under_roots(path: Path, roots: list[Path]) -> Path:
    """Ensure path is under at least one configured root."""
    path_r = path.expanduser().resolve()
    errors: list[str] = []
    for root in roots:
        try:
            return ensure_under_root(path_r, root)
        except PathTraversalError as e:
            errors.append(str(e))
    joined = "; ".join(str(r.resolve()) for r in roots) or "(none)"
    raise PathTraversalError(
        f"path {path_r} is outside transcripts roots [{joined}]"
    )


def parse_transcripts_dirs(value: str | Path | list[Path] | None) -> list[Path]:
    """Parse env/list of transcript roots (os.pathsep, comma, or semicolon)."""
    if value is None:
        return []
    if isinstance(value, list):
        return [Path(p) for p in value if str(p).strip()]
    if isinstance(value, Path):
        s = str(value)
    else:
        s = str(value)
    s = s.strip()
    if not s:
        return []
    parts: list[str] = []
    for chunk in re.split(r"[;," + re.escape(os.pathsep) + r"]+", s):
        chunk = chunk.strip().strip('"').strip("'")
        if chunk:
            parts.append(chunk)
    return [Path(p) for p in parts]


def session_id_from_path(path: Path) -> str | None:
    """Session id = parent folder name when it matches file stem, else stem."""
    path = Path(path)
    stem = path.stem
    parent = path.parent.name
    if parent and parent == stem:
        return parent
    if _SESSION_STEM_RE.match(stem):
        return stem
    if parent and _SESSION_STEM_RE.match(parent):
        return parent
    return stem or None


def discover_session_files(transcripts_dirs: Path | list[Path]) -> list[Path]:
    """Find Grok Bot / Cursor agent transcript JSONL files under root(s).

    Preferred layout: ``<root>/<session_id>/<session_id>.jsonl``
    Also accepts flat ``<root>/*.jsonl`` and nested mirrors.
    """
    roots = (
        [Path(transcripts_dirs)]
        if isinstance(transcripts_dirs, (str, Path))
        else list(transcripts_dirs)
    )
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*.jsonl"):
            if not p.is_file():
                continue
            try:
                resolved = ensure_under_root(p, root)
            except PathTraversalError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(resolved)
    return sorted(found)


def find_session_by_id(
    transcripts_dirs: Path | list[Path], session_id: str
) -> Path | None:
    """Resolve session by folder/stem id under transcript roots."""
    sid = session_id.strip()
    if not sid:
        return None
    roots = (
        [Path(transcripts_dirs)]
        if isinstance(transcripts_dirs, (str, Path))
        else list(transcripts_dirs)
    )
    sid_l = sid.lower()

    # Fast path: <root>/<id>/<id>.jsonl
    for root in roots:
        root = Path(root)
        candidate = root / sid / f"{sid}.jsonl"
        if candidate.is_file():
            try:
                return ensure_under_root(candidate, root)
            except PathTraversalError:
                continue
        # Case-insensitive scan of direct children
        if root.is_dir():
            for child in root.iterdir():
                if child.is_dir() and child.name.lower() == sid_l:
                    nested = child / f"{child.name}.jsonl"
                    if nested.is_file():
                        try:
                            return ensure_under_root(nested, root)
                        except PathTraversalError:
                            continue

    for p in discover_session_files(roots):
        fid = session_id_from_path(p)
        if fid and fid.lower() == sid_l:
            return p.resolve()
    return None


def resolve_session_path(
    transcripts_dirs: Path | list[Path],
    *,
    session_id: str | None = None,
    path: str | Path | None = None,
) -> Path:
    """Resolve a transcript file by id or explicit path; always under a root."""
    roots = (
        [Path(transcripts_dirs)]
        if isinstance(transcripts_dirs, (str, Path))
        else list(transcripts_dirs)
    )
    if not roots:
        raise FileNotFoundError(
            "no Grok Bot transcripts dirs configured "
            "(set SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR)"
        )
    if path is not None:
        return ensure_under_roots(Path(path), roots)
    if session_id:
        found = find_session_by_id(roots, session_id)
        if found is None:
            joined = ", ".join(str(r) for r in roots)
            raise FileNotFoundError(
                f"session not found for id={session_id!r} under {joined}"
            )
        return found
    raise ValueError("session_id or path is required")
