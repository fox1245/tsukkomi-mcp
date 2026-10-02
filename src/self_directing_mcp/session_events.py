"""Shared lossless JSONL and event helpers. Raw evidence stays local."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import io
import os
import json
import re
from pathlib import Path
from typing import Any, Iterator

from self_directing_mcp.schemas import Chunk
from self_directing_mcp.request_control import check_request


def source_stamp(path: Path) -> tuple[int, ...]:
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@dataclass(frozen=True)
class SourceObservation:
    path: Path
    stamp: tuple[int, ...]
    data: bytes

    def prefix_digest(self, length: int) -> str:
        if length < 0 or length > len(self.data):
            raise ValueError("source truncated during read")
        return hashlib.sha256(memoryview(self.data)[:length]).hexdigest()

    def validate(self) -> bool:
        check_request()
        try:
            return source_stamp(self.path) == self.stamp
        except OSError:
            return False


def capture_source(path: Path) -> SourceObservation:
    """Read one stable source generation; parsers and receipt hashes share its bytes."""
    check_request()
    path = Path(path).resolve()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        blocks = []
        while block := stream.read(1024 * 1024):
            check_request()
            blocks.append(block)
        after = os.fstat(stream.fileno())
    stamp = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    data = b"".join(blocks)
    observation = SourceObservation(path, stamp(after), data)
    if stamp(before) != observation.stamp or len(data) != after.st_size or not observation.validate():
        raise ValueError("source changed during preparation")
    return observation


_source_observation: ContextVar[SourceObservation | None] = ContextVar("source_observation", default=None)


def observed_source(path: Path) -> SourceObservation | None:
    observation = _source_observation.get()
    return observation if observation is not None and Path(path).resolve() == observation.path else None


@contextmanager
def source_scope(observation: SourceObservation):
    token = _source_observation.set(observation)
    try:
        yield observation
    finally:
        _source_observation.reset(token)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_raw_lines(path: Path, *, start_byte: int = 0) -> Iterator[tuple[int, int, int, str]]:
    """Cursors point to the start of a record. Leave unfinished final records unread."""
    check_request()
    observation = observed_source(path)
    with (io.BytesIO(observation.data) if observation else Path(path).open("rb")) as stream:
        if start_byte < 0:
            raise ValueError("negative byte cursor")
        prefix = stream.read(start_byte)
        if len(prefix) != start_byte or (start_byte and not prefix.endswith(b"\n")):
            raise ValueError("cursor is not on a complete record boundary")
        line_no = prefix.count(b"\n")
        while True:
            check_request()
            start = stream.tell()
            raw = stream.readline()
            if not raw or not raw.endswith(b"\n"):
                return
            line_no += 1
            # Invalid UTF-8 remains detectable instead of silently altering evidence.
            yield line_no, start, stream.tell(), raw.decode("utf-8", errors="surrogateescape")


def make_chunk(provider, sid, kind, text, meta, line, start, end, sub=0, timestamp=None):
    digest = content_hash(f"{kind}|{text}")
    # Include content digest to invalidate anchors after a source rewrite at the same offset.
    event_digest = content_hash(f"{digest}|{as_text(meta)}|{timestamp}")
    event_id = f"{provider}:{sid}:{start}:{sub}:{event_digest[:16]}"
    return Chunk(chunk_id=event_id, session_id=sid, provider=provider, kind=kind,
                 text=text, content_hash=digest, timestamp=timestamp,
                 line_start=line, line_end=line, byte_start=start, byte_end=end,
                 meta={**meta, "provider": provider, "sub_index": sub})


def parse_error(provider, sid, line, start, end, reason):
    return make_chunk(provider, sid, "meta", f"[parse_error] {reason}",
                      {"parse_error": True}, line, start, end)


def result_success(payload: Any) -> bool | None:
    """Use structured status / explicit runtime exit status, never prose such as 'passed'."""
    if isinstance(payload, str):
        try:
            return result_success(json.loads(payload))
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"(?im)^\s*(?:Process exited with code|Process exit code|Exit code)\s*:?\s*(-?\d+)\s*$", payload)
            return int(match.group(1)) == 0 if match else None
    if isinstance(payload, dict):
        for key in ("exit_code", "exitCode"):
            value = payload.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value == 0
        if payload.get("is_error") is True:
            return False
        for key in ("output", "content"):
            if key in payload:
                status = result_success(payload[key])
                if status is not None:
                    return status
    if isinstance(payload, list):
        statuses = [result_success(block.get("text")) for block in payload if isinstance(block, dict)]
        if False in statuses:
            return False
        if True in statuses:
            return True
    return None
