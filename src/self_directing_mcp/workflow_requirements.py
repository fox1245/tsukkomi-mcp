"""Owner-approved requirements, captured provenance, and deterministic obligations.

Approval storage must be mounted read-only outside the agent's authority. Hashes
and exclusive creation detect corruption; they are not same-OS-user isolation.

Every Lean source (definitions and proof bodies alike) is an immutable approval
artifact. Scope globs select mutable implementation files, never implicit test
configuration. Verification snapshots contain only these tracked files and
captured approved artifacts, not a recursive copy of the live project.
"""
from __future__ import annotations

import copy
import difflib
import fnmatch
import hashlib
import json
import math
import os
import re
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ApprovalError(ValueError):
    """An approval is malformed, unsafe, incomplete, or no longer intact."""


_IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".lake", "node_modules", "build", "dist",
})
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")
_THEOREM = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*\Z")
_RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?\Z", re.I)
_TEST_CONFIG_NAMES = frozenset({
    "conftest.py", "sitecustomize.py", "usercustomize.py", "pytest.ini",
    ".pytest.ini", "pyproject.toml", "setup.cfg", "setup.py", "tox.ini",
    ".coveragerc", "coverage.ini", "lakefile.lean", "lakefile.toml",
    "lean-toolchain", "lake-manifest.json",
})
_GENERATED_SUFFIXES = frozenset({".pyc", ".pyo", ".olean", ".ilean"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity(value: Any) -> str:
    return _hash(_canonical(value))


def _json(data: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ApprovalError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        result = json.loads(data, object_pairs_hook=pairs,
                            parse_constant=lambda value: (_ for _ in ()).throw(
                                ApprovalError(f"Nonfinite JSON value: {value}")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ApprovalError("Invalid UTF-8 JSON") from exc
    if not isinstance(result, dict):
        raise ApprovalError("Expected a JSON object")
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ApprovalError(f"{label} must be a nonempty string")
    return value


def _path(value: Any, *, pattern: bool = False) -> str:
    value = _text(value, "Path")
    if "\\" in value or ":" in value or value.startswith("/"):
        raise ApprovalError(f"Path must be project-relative and portable: {value}")
    parts = value.split("/")
    for part in parts:
        if (part in {"", ".", ".."} or part[-1] in " ." or
                any(ord(char) < 32 or char in '<>|"' for char in part) or
                _RESERVED.fullmatch(part) or unicodedata.normalize("NFC", part) != part):
            raise ApprovalError(f"Unsafe path: {value}")
        if not pattern and any(char in part for char in "*?[]"):
            raise ApprovalError(f"Literal path required: {value}")
    if any(part in _IGNORED_DIRS for part in parts[:-1]):
        raise ApprovalError(f"Approval cannot track generated/cache paths: {value}")
    return value


def _matches(path: str, pattern: str) -> bool:
    """Segment globbing, with ** matching zero or more complete segments."""
    names, patterns = path.split("/"), pattern.split("/")
    def match(i: int, j: int) -> bool:
        if j == len(patterns):
            return i == len(names)
        if patterns[j] == "**":
            return match(i, j + 1) or (i < len(names) and match(i + 1, j))
        return (i < len(names) and fnmatch.fnmatchcase(names[i], patterns[j])
                and match(i + 1, j + 1))
    return match(0, 0)


def _inside(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def _safe_file(root: Path, relative: str) -> Path:
    _path(relative)
    current = root
    for part in relative.split("/"):
        current = current / part
        if current.is_symlink():
            raise ApprovalError(f"Symbolic links are not allowed in tracked paths: {relative}")
    if not _inside(current.resolve(), root):
        raise ApprovalError(f"Path escapes project: {relative}")
    return current


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _open_directory(path: Path) -> int:
    """Walk absolute directory components without following any symbolic link."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_file(root: Path, relative: str, *,
               identities: dict[str, tuple[int, ...]] | None = None) -> bytes:
    file = _safe_file(root, relative)
    directory_fd = None
    try:
        if not file.is_file():
            raise ApprovalError(f"Tracked file is missing or not regular: {relative}")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        if os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW"):
            directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            directory_fd = _open_directory(root)
            parts = relative.split("/")
            for part in parts[:-1]:
                next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = next_fd
            fd = os.open(parts[-1], flags, dir_fd=directory_fd)
        else:
            fd = os.open(file, flags)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ApprovalError(f"Tracked path is not a regular file: {relative}")
            data = stream.read()
            identity = _file_identity(before)
            if (identity != _file_identity(os.fstat(stream.fileno())) or
                    identity != _file_identity(_safe_file(root, relative).stat(follow_symlinks=False))):
                raise ApprovalError(f"Tracked file changed while reading: {relative}")
            if identities is not None:
                identities[relative] = identity
            return data
    except OSError as exc:
        raise ApprovalError(f"Cannot read tracked file: {relative}") from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def _strings(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ApprovalError(f"{label} must be {'a nonempty' if nonempty else 'a'} list")
    for item in value:
        _text(item, label)
    if len(value) != len(set(value)):
        raise ApprovalError(f"Duplicate {label}")
    return value


def _fields(value: Any, required: set[str], optional: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ApprovalError(f"{label} must be an object")
    missing, extra = required - value.keys(), value.keys() - required - optional
    if missing or extra:
        raise ApprovalError(f"Invalid {label} fields (missing={sorted(missing)}, unsupported={sorted(extra)})")


def _number(value: Any, label: str, low: float, high: float | None = None) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < low or (high is not None and value > high)):
        raise ApprovalError(f"Invalid {label}")


def _validate_manifest(manifest: dict) -> dict:
    manifest = copy.deepcopy(manifest)
    _fields(manifest, {"schema_version", "prd", "scope", "requirements", "tests", "proofs",
                       "completion_actions", "class_mapping_version"},
            {"protected_paths", "confidence_threshold", "external_context"}, "manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ApprovalError("Unsupported requirements schema")
    if manifest["class_mapping_version"] != "1":
        raise ApprovalError("Unsupported class mapping version")
    manifest.setdefault("protected_paths", [])
    manifest.setdefault("confidence_threshold", .7)
    manifest.setdefault("external_context", "disabled")
    _number(manifest["confidence_threshold"], "confidence_threshold", 0, 1)
    if manifest["external_context"] not in ("disabled", "synthetic"):
        raise ApprovalError("external_context must be disabled or synthetic")
    if not _path(manifest["prd"]).lower().endswith(".md"):
        raise ApprovalError("PRD must be Markdown")
    for key in ("scope", "protected_paths"):
        for pattern in _strings(manifest[key], key, nonempty=key == "scope"):
            _path(pattern, pattern=True)
    for kind in ("tests", "proofs"):
        entries = manifest[kind]
        if not isinstance(entries, dict):
            raise ApprovalError(f"{kind} must map IDs to definitions")
        ids = set()
        for identifier, entry in entries.items():
            if not _ID.fullmatch(identifier) or identifier.casefold() in ids:
                raise ApprovalError(f"Ambiguous {kind} ID: {identifier}")
            ids.add(identifier.casefold())
            if kind == "tests":
                _fields(entry, {"argv", "paths"}, {"timeout_sec"}, "test")
                # Repeated command arguments are legitimate.
                if not isinstance(entry["argv"], list) or not entry["argv"]:
                    raise ApprovalError("Test argv must be nonempty")
                for argument in entry["argv"]:
                    if not isinstance(argument, str) or "\x00" in argument:
                        raise ApprovalError("Invalid test argument")
                _text(entry["argv"][0], "Test executable")
                for path in _strings(entry["paths"], "test paths", nonempty=True):
                    _path(path)
                entry.setdefault("timeout_sec", 60)
                _number(entry["timeout_sec"], "test timeout", 0)
                if entry["timeout_sec"] == 0:
                    raise ApprovalError("Test timeout must be positive")
            else:
                _fields(entry, {"path", "theorem", "statement"}, set(), "proof")
                if not _path(entry["path"]).endswith(".lean"):
                    raise ApprovalError("Proof path must be a Lean source")
                if not _THEOREM.fullmatch(_text(entry["theorem"], "theorem")):
                    raise ApprovalError("Invalid qualified theorem name")
                _text(entry["statement"], "statement")
    requirements = manifest["requirements"]
    if not isinstance(requirements, list) or not requirements:
        raise ApprovalError("Requirements must be nonempty")
    ids = set()
    for requirement in requirements:
        _fields(requirement, {"id", "text", "source", "code", "specs", "tests", "proofs",
                              "required", "status"}, {"exceptions", "depends_on", "conflicts_with"}, "requirement")
        identifier = _text(requirement["id"], "requirement ID")
        if not _ID.fullmatch(identifier) or identifier.casefold() in ids:
            raise ApprovalError(f"Ambiguous requirement ID: {identifier}")
        _path(f"{identifier}.md")
        if identifier.casefold() == "index":
            raise ApprovalError("Requirement ID collides with wiki index")
        ids.add(identifier.casefold())
        _text(requirement["text"], "requirement text")
        if type(requirement["required"]) is not bool:
            raise ApprovalError("required must be a boolean")
        if requirement["status"] not in ("approved", "derived"):
            raise ApprovalError("Unknown requirement status")
        if requirement["required"] and requirement["status"] != "approved":
            raise ApprovalError(f"Required item is not approved: {identifier}")
        requirement.setdefault("exceptions", [])
        requirement.setdefault("depends_on", [])
        requirement.setdefault("conflicts_with", [])
        for key in ("code", "specs", "tests", "proofs", "exceptions", "depends_on", "conflicts_with"):
            _strings(requirement[key], key, nonempty=bool(requirement["required"] and
                                                        key in {"code", "specs", "tests", "proofs"}))
        for path in requirement["code"]:
            _path(path, pattern=True)
        for path in requirement["specs"]:
            _path(path)
        for kind in ("tests", "proofs"):
            if set(requirement[kind]) - manifest[kind].keys():
                raise ApprovalError(f"Dangling {kind} link in {identifier}")
        source = requirement["source"]
        _fields(source, {"path", "section", "quote"}, set(), "requirement source")
        if _path(source["path"]) != manifest["prd"]:
            raise ApprovalError("Requirement origins must reference the approved PRD")
        _text(source["section"], "source section")
        _text(source["quote"], "source quote")
    if not any(item["required"] for item in requirements):
        raise ApprovalError("At least one approved requirement must be required")
    by_id = {item["id"]: item for item in requirements}
    for requirement in requirements:
        for conflict in requirement["conflicts_with"]:
            if conflict not in by_id or conflict == requirement["id"]:
                raise ApprovalError(f"Invalid requirement conflict: {conflict}")
    visiting, visited = set(), set()
    def visit(identifier: str) -> None:
        if identifier in visiting:
            raise ApprovalError(f"Requirement dependency cycle: {identifier}")
        if identifier in visited:
            return
        visiting.add(identifier)
        for dependency in by_id[identifier]["depends_on"]:
            if dependency not in by_id:
                raise ApprovalError(f"Dangling requirement dependency: {dependency}")
            if by_id[identifier]["required"] and by_id[dependency]["status"] != "approved":
                raise ApprovalError(f"Required dependency is not approved: {dependency}")
            visit(dependency)
        visiting.remove(identifier)
        visited.add(identifier)
    for identifier in by_id:
        visit(identifier)
    required_closure = set()
    pending = [item["id"] for item in requirements if item["required"]]
    while pending:
        identifier = pending.pop()
        if identifier in required_closure:
            continue
        required_closure.add(identifier)
        item = by_id[identifier]
        if item["status"] != "approved" or not item["tests"] or not item["proofs"]:
            raise ApprovalError(f"Required dependency needs approval, tests and proofs: {identifier}")
        pending.extend(item["depends_on"])
    for identifier in required_closure:
        if required_closure.intersection(by_id[identifier]["conflicts_with"]):
            raise ApprovalError(f"Conflicting mandatory requirements: {identifier}")
    actions = manifest["completion_actions"]
    if not isinstance(actions, list) or not actions:
        raise ApprovalError("completion_actions must be nonempty")
    for action in actions:
        _fields(action, {"tool_name", "argument_pattern"}, set(), "completion action")
        _text(action["tool_name"], "completion tool")
        _text(action["argument_pattern"], "argument pattern")
        try:
            re.compile(action["argument_pattern"])
        except re.error as exc:
            raise ApprovalError("Invalid completion argument regex") from exc
    return manifest


def _sections(markdown: str) -> list[tuple[str, str]]:
    lines = markdown.splitlines(keepends=True)
    headings = []
    fence = None
    for index, line in enumerate(lines):
        stripped = line.lstrip(" ")
        marker = re.match(r"(`{3,}|~{3,})", stripped) if len(line) - len(stripped) <= 3 else None
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        heading = re.match(r" {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*(?:\r?\n)?$", line)
        if heading:
            headings.append((index, len(heading.group(1)), heading.group(2).strip()))
    result = []
    for position, (index, depth, title) in enumerate(headings):
        end = next((other[0] for other in headings[position + 1:] if other[1] <= depth), len(lines))
        result.append((title, "".join(lines[index + 1:end])))
    return result


def _source_errors(manifest: dict, artifacts: dict[str, bytes]) -> list[dict]:
    try:
        sections = _sections(artifacts[manifest["prd"]].decode("utf-8"))
    except (UnicodeError, KeyError):
        return [{"code": "invalid_prd", "requirement_id": "", "message": "PRD must be captured UTF-8 Markdown"}]
    errors = []
    for requirement in manifest["requirements"]:
        source = requirement["source"]
        matches = [body for title, body in sections if title == source["section"]]
        if len(matches) != 1 or source["quote"] not in matches[0]:
            errors.append({"code": "invalid_source", "requirement_id": requirement["id"],
                           "message": "Source section must be unique and contain the exact approved quote"})
    return errors


def _explicit_paths(manifest: dict, manifest_path: str) -> set[str]:
    paths = {manifest_path, manifest["prd"]}
    paths.update(path for requirement in manifest["requirements"] for path in requirement["specs"])
    paths.update(path for test in manifest["tests"].values() for path in test["paths"])
    paths.update(proof["path"] for proof in manifest["proofs"].values())
    return paths


def _code_patterns(manifest: dict) -> list[str]:
    # Requirement links must never disappear from tracking merely because a
    # manifest's broad scope glob omitted one of its explicit code links.
    return sorted(set(manifest["scope"]) |
                  {pattern for item in manifest["requirements"] for pattern in item["code"]})


def _protected_paths(manifest: dict, manifest_path: str, paths) -> set[str]:
    return _explicit_paths(manifest, manifest_path) | {
        path for path in paths if path.endswith(".lean") or
        any(_matches(path, pattern) for pattern in manifest["protected_paths"])
    }


def _test_configuration(relative: str) -> bool:
    path = Path(relative)
    return (path.name.casefold() in _TEST_CONFIG_NAMES or path.suffix.casefold() == ".pth" or
            path.name.startswith("test_") or path.name.endswith(("_test.py", "_spec.py")) or
            any(part.casefold() in {"test", "tests", "testing", "__tests__"} for part in path.parts[:-1]))


def _workspace(root: Path, manifest: dict, manifest_path: str, *,
               identities: dict[str, tuple[int, ...]] | None = None) -> dict[str, bytes]:
    explicit = _explicit_paths(manifest, manifest_path)
    patterns = _code_patterns(manifest) + manifest["protected_paths"]
    paths = set(explicit)
    test_paths = {path for test in manifest["tests"].values() for path in test["paths"]}
    def unreadable(error: OSError) -> None:
        raise ApprovalError("Cannot enumerate approved workspace") from error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=unreadable):
        dirs[:] = sorted(name for name in dirs if name not in _IGNORED_DIRS)
        for name in dirs + files:
            entry = Path(directory) / name
            relative = entry.relative_to(root).as_posix()
            # Reject linked directories even when the glob targets their descendants.
            if entry.is_symlink():
                raise ApprovalError(f"Workspace symbolic links are unsupported: {relative}")
        for name in files:
            relative = (Path(directory) / name).relative_to(root).as_posix()
            if Path(relative).suffix.casefold() in _GENERATED_SUFFIXES and relative not in explicit:
                continue
            if relative.endswith(".lean") or any(_matches(relative, pattern) for pattern in patterns):
                paths.add(relative)
    result, portable = {}, set()
    for relative in sorted(paths):
        _path(relative)
        if Path(relative).suffix.casefold() in _GENERATED_SUFFIXES:
            raise ApprovalError(f"Generated verification artifacts cannot be tracked: {relative}")
        if (_test_configuration(relative) and not relative.endswith(".lean") and relative not in test_paths and
                not any(_matches(relative, pattern) for pattern in manifest["protected_paths"])):
            raise ApprovalError(f"Test/configuration file requires explicit approved test protection: {relative}")
        folded = relative.casefold()
        if folded in portable:
            raise ApprovalError(f"Case-ambiguous tracked path: {relative}")
        portable.add(folded)
        file = _safe_file(root, relative)
        if file.exists():
            result[relative] = _read_file(root, relative, identities=identities)
    return result


def _hashes(artifacts: dict[str, bytes]) -> dict[str, str]:
    return {path: _hash(data) for path, data in sorted(artifacts.items())}


def _exclusive(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def _sync_directory(path: Path) -> None:
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


@dataclass(frozen=True)
class ApprovedProject:
    root: Path
    approval_dir: Path
    manifest: dict
    approval_digest: str
    _record: dict

    @classmethod
    def load(cls, approval_dir: Path) -> ApprovedProject:
        directory = Path(approval_dir).absolute()
        if directory.is_symlink() or directory.resolve() != directory:
            raise ApprovalError("Approval storage must not traverse symlinks")
        record_data = _read_file(directory, "approval.json")
        record = _json(record_data)
        _fields(record, {"schema_version", "root", "manifest_path", "manifest", "baseline"}, set(), "approval")
        if type(record["schema_version"]) is not int or record["schema_version"] != 1:
            raise ApprovalError("Unsupported approval schema")
        digest = _read_file(directory, "approval.sha256").decode("ascii").strip()
        if digest != _hash(record_data):
            raise ApprovalError("Approval record integrity failure")
        root = Path(record["root"])
        if not root.is_absolute() or root.resolve() != root or not root.is_dir():
            raise ApprovalError("Approved workspace is missing or changed location")
        if _inside(directory, root) or _inside(root, directory):
            raise ApprovalError("Approval storage and workspace must be disjoint")
        manifest = _validate_manifest(record["manifest"])
        manifest_path = _path(record["manifest_path"])
        baseline = record["baseline"]
        if not isinstance(baseline, dict) or not baseline:
            raise ApprovalError("Missing captured baseline")
        artifacts = {}
        for relative, expected in baseline.items():
            _path(relative)
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ApprovalError("Invalid captured content identity")
            data = _read_file(directory, f"blobs/{expected}")
            if _hash(data) != expected:
                raise ApprovalError(f"Captured artifact integrity failure: {relative}")
            artifacts[relative] = data
        if _explicit_paths(manifest, manifest_path) - artifacts.keys():
            raise ApprovalError("Missing captured approved artifact")
        if _validate_manifest(_json(artifacts[manifest_path])) != manifest:
            raise ApprovalError("Captured manifest does not match approval")
        errors = _source_errors(manifest, artifacts)
        if errors:
            raise ApprovalError(errors[0]["message"])
        return cls(root, directory, manifest, digest, record)

    def _current(self) -> dict[str, str]:
        return _hashes(_workspace(self.root, self.manifest, self._record["manifest_path"]))

    def _intact(self) -> ApprovedProject:
        intact = type(self).load(self.approval_dir)
        if (intact.approval_digest != self.approval_digest or intact.manifest != self.manifest or
                intact.root != self.root or intact._record != self._record):
            raise ApprovalError("Approval changed after loading")
        return intact

    def approved_hash(self, relative: str) -> str:
        """Return the validated captured identity, never a live workspace hash."""
        relative = _path(relative)
        intact = self._intact()
        try:
            return intact._record["baseline"][relative]
        except KeyError as exc:
            raise ApprovalError(f"Artifact was not captured by approval: {relative}") from exc

    def materialize(self, destination: Path) -> dict[str, str]:
        """Create a fresh isolated project layout and hash every emitted file.

        The existing parent and all its ancestors must be real directories;
        destination must be disjoint from both project and approval storage.
        Immutable artifacts come from validated approval blobs. Only mutable
        files selected by scope/code globs come from the current workspace.
        All emitted files are read-only and executable, permitting approved
        direct script commands without trusting live executable-mode changes.
        Failures raise ApprovalError; the caller owns cleanup of partial output.
        """
        if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
            raise ApprovalError("Safe snapshot materialization requires POSIX no-follow operations")
        try:
            destination = Path(destination).absolute()
            if (destination.resolve() != destination or destination.exists() or
                    any(_inside(destination, root) or _inside(root, destination)
                        for root in (self.root, self.approval_dir))):
                raise ApprovalError("Snapshot destination must be fresh, disjoint, and without symlinks")
        except (OSError, RuntimeError) as exc:
            raise ApprovalError("Cannot resolve snapshot destination safely") from exc
        self._intact()
        identities: dict[str, tuple[int, ...]] = {}
        artifacts = _workspace(self.root, self.manifest, self._record["manifest_path"],
                               identities=identities)
        current = _hashes(artifacts)
        baseline = self._record["baseline"]
        protected = _protected_paths(self.manifest, self._record["manifest_path"],
                                     current.keys() | baseline.keys())
        for relative in sorted(protected):
            if current.get(relative) != baseline.get(relative) or relative not in baseline:
                raise ApprovalError(f"Approved source changed; fresh owner approval required: {relative}")
            data = _read_file(self.approval_dir, f"blobs/{baseline[relative]}")
            if _hash(data) != baseline[relative]:
                raise ApprovalError(f"Captured artifact integrity failure: {relative}")
            artifacts[relative] = data
        parent_fd = destination_fd = None
        try:
            parent_fd = _open_directory(destination.parent)
            os.mkdir(destination.name, mode=0o700, dir_fd=parent_fd)
            destination_fd = os.open(destination.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                     dir_fd=parent_fd)
            destination_identity = os.fstat(destination_fd)
            emitted_identities: dict[str, tuple[int, ...]] = {}
            for relative, data in sorted(artifacts.items()):
                directory_fd = os.dup(destination_fd)
                try:
                    parts = relative.split("/")
                    for part in parts[:-1]:
                        try:
                            os.mkdir(part, mode=0o700, dir_fd=directory_fd)
                        except FileExistsError:
                            pass
                        next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                          dir_fd=directory_fd)
                        os.close(directory_fd)
                        directory_fd = next_fd
                    fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=directory_fd)
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                        os.fchmod(stream.fileno(), 0o555)
                        emitted_identities[relative] = _file_identity(os.fstat(stream.fileno()))
                finally:
                    os.close(directory_fd)
            after_identities: dict[str, tuple[int, ...]] = {}
            after = _workspace(self.root, self.manifest, self._record["manifest_path"],
                               identities=after_identities)
            if current != _hashes(after) or identities != after_identities:
                raise ApprovalError("Workspace changed during snapshot materialization")
            self._intact()
            placed = destination.stat(follow_symlinks=False)
            if (destination.resolve() != destination or
                    (placed.st_dev, placed.st_ino) !=
                    (destination_identity.st_dev, destination_identity.st_ino)):
                raise ApprovalError("Snapshot destination changed during materialization")
            hashes = _hashes(artifacts)
            observed_paths = set()
            for directory, dirs, files in os.walk(destination, followlinks=False):
                for name in dirs + files:
                    entry = Path(directory) / name
                    if entry.is_symlink():
                        raise ApprovalError("Snapshot gained a symbolic link during materialization")
                observed_paths.update((Path(directory) / name).relative_to(destination).as_posix()
                                      for name in files)
            if observed_paths != hashes.keys():
                raise ApprovalError("Snapshot file set changed during materialization")
            copied_identities: dict[str, tuple[int, ...]] = {}
            for relative, expected in hashes.items():
                if _hash(_read_file(destination, relative, identities=copied_identities)) != expected:
                    raise ApprovalError(f"Snapshot file changed during materialization: {relative}")
            if copied_identities != emitted_identities:
                raise ApprovalError("Snapshot file identity changed during materialization")
            return hashes
        except OSError as exc:
            raise ApprovalError(f"Cannot materialize snapshot: {exc.strerror}") from exc
        finally:
            if destination_fd is not None:
                os.close(destination_fd)
            if parent_fd is not None:
                os.close(parent_fd)

    def changed_paths(self) -> list[str]:
        current = self._current()
        baseline = self._record["baseline"]
        return sorted(path for path in current.keys() | baseline.keys()
                      if current.get(path) != baseline.get(path))

    def obligations(self, changed_paths: list[str] | None = None) -> list[dict]:
        # A caller may add context paths, but cannot hide actual workspace changes.
        changed = set(self.changed_paths())
        if changed_paths is not None:
            changed.update(_path(path) for path in changed_paths)
        by_id = {item["id"]: item for item in self.manifest["requirements"]}
        selected = {item["id"] for item in by_id.values() if item["required"] or
                    any(_matches(path, pattern) for path in changed for pattern in item["code"])}
        pending = list(selected)
        while pending:
            for dependency in by_id[pending.pop()]["depends_on"]:
                if dependency not in selected:
                    selected.add(dependency)
                    pending.append(dependency)
        return [copy.deepcopy(by_id[identifier]) for identifier in sorted(selected)]

    def versions(self) -> dict[str, str]:
        current = self._current()
        def subset(paths):
            return {path: current.get(path, "missing") for path in sorted(paths)}
        specs = {path for item in self.manifest["requirements"] for path in item["specs"]}
        specs.update(proof["path"] for proof in self.manifest["proofs"].values())
        specs.update(path for path in current.keys() | self._record["baseline"].keys()
                     if path.endswith(".lean"))
        tests = {path for test in self.manifest["tests"].values() for path in test["paths"]}
        code = {path for path in current.keys() | self._record["baseline"].keys()
                if any(_matches(path, pattern) for pattern in _code_patterns(self.manifest))}
        return {
            "requirement": _identity({"approval": self.approval_digest,
                                      "sources": subset({self.manifest["prd"], self._record["manifest_path"]})}),
            "code": _identity(subset(code)), "spec": _identity(subset(specs)),
            "test": _identity(subset(tests)),
            "policy": _identity({"configuration": {key: self.manifest[key] for key in
                                 ("completion_actions", "protected_paths", "confidence_threshold", "external_context")},
                                 "protected": subset(path for path in current.keys() | self._record["baseline"].keys()
                                                     if any(_matches(path, pattern) for pattern in self.manifest["protected_paths"]))}),
            "mapping": _identity({"version": self.manifest["class_mapping_version"],
                                  "choices": ["InsufficientEvidence", "NeedsRevision", "ReadyForVerification", "ReadyForCompletion"]}),
        }

    def lint(self) -> list[dict]:
        errors = []
        try:
            self._intact()
            current = self._current()
            protected = _explicit_paths(self.manifest, self._record["manifest_path"])
            for path in sorted(protected - current.keys()):
                errors.append({"code": "missing_artifact", "requirement_id": "",
                               "message": f"Approved source or verification reference is missing: {path}"})
            protected = _protected_paths(self.manifest, self._record["manifest_path"],
                                         current.keys() | self._record["baseline"].keys())
            for path in sorted(protected):
                if current.get(path) != self._record["baseline"].get(path):
                    errors.append({"code": "approval_invalidated", "requirement_id": "",
                                   "message": f"Approved source changed; fresh owner approval required: {path}"})
            active = self.obligations()
            active_ids = {item["id"] for item in active}
            for requirement in active:
                conflicts = active_ids.intersection(requirement["conflicts_with"])
                if conflicts:
                    errors.append({"code": "conflicting_requirements", "requirement_id": requirement["id"],
                                   "message": "Conflicts with active obligations: " + ", ".join(sorted(conflicts))})
                if requirement["status"] != "approved":
                    errors.append({"code": "unapproved_obligation", "requirement_id": requirement["id"],
                                   "message": "Derived obligation requires owner approval"})
                if not requirement["tests"] or not requirement["proofs"]:
                    errors.append({"code": "missing_verification", "requirement_id": requirement["id"],
                                   "message": "Active obligation requires tests and proofs"})
        except (ApprovalError, OSError, UnicodeError) as exc:
            errors.append({"code": "approval_integrity", "requirement_id": "", "message": str(exc)})
        return errors

    def context(self) -> dict:
        return {"approval_digest": self.approval_digest, "prd": self.manifest["prd"],
                "requirements": copy.deepcopy(sorted(self.manifest["requirements"], key=lambda item: item["id"])),
                "class_mapping_version": self.manifest["class_mapping_version"],
                "external_context": self.manifest["external_context"]}

    def change_context(self, max_bytes: int = 16384) -> list[dict]:
        """Bounded real code changes for an explicitly authorized JEV request.

        Never silently truncate code or treat filenames alone as change content.
        The caller must still enforce external-context approval and secret checks.
        """
        changes = []
        size = 0
        current = self._current()
        for path in self.changed_paths():
            if not any(_matches(path, pattern) for pattern in _code_patterns(self.manifest)):
                continue
            previous_hash = self._record["baseline"].get(path)
            previous = _read_file(self.approval_dir, f"blobs/{previous_hash}") if previous_hash else b""
            present = _read_file(self.root, path) if path in current else b""
            size += len(previous) + len(present)
            if size > max_bytes:
                raise ApprovalError("Changed code exceeds approved classifier context size")
            try:
                diff = "".join(difflib.unified_diff(
                    previous.decode("utf-8").splitlines(keepends=True),
                    present.decode("utf-8").splitlines(keepends=True),
                    fromfile="approved/" + path, tofile="current/" + path))
            except UnicodeError as exc:
                raise ApprovalError("Changed code is not UTF-8 classifier input") from exc
            changes.append({"path": path, "before_hash": previous_hash, "after_hash": current.get(path), "diff": diff})
        return changes

    def render_wiki(self, output_dir: Path) -> list[Path]:
        errors = self.lint()
        if errors:
            raise ApprovalError(f"Cannot render invalid approved requirements: {errors[0]['message']}")
        output = Path(output_dir).absolute()
        if output.resolve() != output or _inside(output, self.approval_dir):
            raise ApprovalError("Wiki output must not traverse symlinks or mutate approval storage")
        pages = {f"{item['id']}.md": item for item in self.manifest["requirements"]}
        pages["index.md"] = None
        for name in pages:
            destination = output / name
            if destination.exists() or destination.is_symlink():
                raise ApprovalError("Wiki output must be fresh; refusing to overwrite existing files")
        if _inside(output, self.root):
            for name in pages:
                relative = (output / name).relative_to(self.root).as_posix()
                if relative in self._record["baseline"] or any(
                        _matches(relative, pattern) for pattern in _code_patterns(self.manifest) + self.manifest["protected_paths"]):
                    raise ApprovalError("Derived wiki must be outside approved source/code scope")
        output.mkdir(parents=True, exist_ok=True)
        written = []
        for name, item in sorted(pages.items()):
            destination = output / name
            if item is None:
                text = "# Derived requirements wiki\n\n" + f"Approval: `{self.approval_digest}`\n\n"
                text += "".join(f"- [{entry['id']}]({entry['id']}.md)\n" for entry in
                                sorted(self.manifest["requirements"], key=lambda entry: entry["id"]))
            else:
                source = item["source"]
                text = (f"# {item['id']}\n\n{item['text']}\n\n"
                        f"Status: {item['status']}; required: {str(item['required']).lower()}\n\n"
                        f"## Approved origin\n\nPRD: `{source['path']}`; section: {source['section']}\n\n"
                        + "\n".join("> " + line for line in source["quote"].splitlines()) + "\n\n")
                for label, key in (("Code scope", "code"), ("Specifications", "specs"),
                                   ("Tests", "tests"), ("Proofs", "proofs"), ("Exceptions", "exceptions")):
                    text += f"## {label}\n\n" + "".join(f"- {value}\n" for value in item[key]) + "\n"
                text += "## Dependencies\n\n" + "".join(f"- [{identifier}]({identifier}.md)\n"
                                                          for identifier in item["depends_on"])
                text += "\n## Conflicts\n\n" + "".join(f"- [{identifier}]({identifier}.md)\n"
                                                        for identifier in item["conflicts_with"])
                text += f"\nApproval: `{self.approval_digest}`\n"
            with destination.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
            written.append(destination)
        return written


def approve_project(project_root: Path, manifest_path: Path, approval_dir: Path) -> ApprovedProject:
    """Capture a fresh approval; invoke only from an owner-controlled host CLI."""
    root = Path(project_root).absolute()
    if root.resolve() != root or not root.is_dir():
        raise ApprovalError("Project root must be an existing canonical directory without symlinks")
    source = Path(manifest_path)
    if not source.is_absolute():
        source = root / source
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ApprovalError("Manifest must be inside project") from exc
    manifest = _validate_manifest(_json(_read_file(root, relative)))
    directory = Path(approval_dir).absolute()
    if directory.resolve() != directory or _inside(directory, root) or _inside(root, directory):
        raise ApprovalError("Approval storage must be outside project without symlinks")
    if directory.exists():
        raise ApprovalError("Approval already exists; select a fresh approval directory")
    identities: dict[str, tuple[int, ...]] = {}
    artifacts = _workspace(root, manifest, relative, identities=identities)
    missing = _explicit_paths(manifest, relative) - artifacts.keys()
    if missing:
        raise ApprovalError(f"Missing approved files: {sorted(missing)}")
    errors = _source_errors(manifest, artifacts)
    if errors:
        raise ApprovalError(errors[0]["message"])
    if _validate_manifest(_json(artifacts[relative])) != manifest:
        raise ApprovalError("Manifest changed during approval")
    baseline = _hashes(artifacts)
    after_identities: dict[str, tuple[int, ...]] = {}
    after = _workspace(root, manifest, relative, identities=after_identities)
    if baseline != _hashes(after) or identities != after_identities:
        raise ApprovalError("Workspace changed during approval")
    record = {"schema_version": 1, "root": str(root), "manifest_path": relative,
              "manifest": manifest, "baseline": baseline}
    record_data = _canonical(record)
    directory.mkdir(parents=True, exist_ok=False)
    blobs = directory / "blobs"
    blobs.mkdir()
    written = set()
    for path, data in artifacts.items():
        digest = baseline[path]
        if digest not in written:
            _exclusive(blobs / digest, data)
            written.add(digest)
    _exclusive(directory / "approval.json", record_data)
    _exclusive(directory / "approval.sha256", (_hash(record_data) + "\n").encode("ascii"))
    blobs.chmod(0o555)
    directory.chmod(0o555)
    _sync_directory(blobs)
    _sync_directory(directory)
    _sync_directory(directory.parent)
    return ApprovedProject.load(directory)
