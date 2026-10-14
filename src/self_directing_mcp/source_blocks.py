"""Block digests of an append-only session source: the data model behind the fast verification path.

A source prefix ``[0, covered)`` is described by the SHA-256 of every complete 1 MiB block, the digest of the
partial last block (the "tail record") and a root over all of it. The table is built while the source is read
anyway and is extended, never recomputed, as the cursor advances: only new complete blocks and the partial tail
are hashed. It is stored beside the cursor in the index metadata (``SessionStore.source_block_record``).

What the table is and is not:

* It is a record of what the prefix hashed to when it was last read. Comparing blocks with the source proves
  that those blocks have the recorded bytes *now*, for exactly the blocks that were compared.
* A fast check compares the tail, the most recent blocks and a few random blocks. It says nothing about the
  other blocks and is never a statement about the whole current source. Only a full verification (every block)
  is, and only for the moment it ran.
* Roots bind the covered length and line count as well, so a table cannot be replayed for another offset.

This module holds pure data and arithmetic; reading files is ``session_events``' job.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import struct

BLOCK_SIZE = 1024 * 1024
HASH_TAG = "blocks1:"  # session_cursors.prefix_hash of a block-tabled cursor: HASH_TAG + root
DIGEST_SIZE = 32
_DOMAIN = b"tsukkomi-source-blocks/1\0"


def is_block_hash(value) -> bool:
    """Whether a cursor's ``prefix_hash`` is a block root (otherwise a plain SHA-256 of the prefix)."""
    return isinstance(value, str) and value.startswith(HASH_TAG)


def compute_root(block_size: int, covered: int, lines: int, digests: bytes, tail: bytes | None) -> str:
    digest = hashlib.sha256()
    digest.update(_DOMAIN)
    digest.update(struct.pack("<QQQ", block_size, covered, lines))
    digest.update(digests)
    digest.update(tail or b"")
    return digest.hexdigest()


@dataclass(frozen=True)
class BlockTable:
    """Digests of the prefix ``[0, covered)``; ``covered`` is a record boundary (the cursor offset)."""

    covered: int
    lines: int  # newline count of [0, covered)
    digests: bytes  # DIGEST_SIZE bytes per complete block
    tail: bytes | None  # digest of the partial last block [nfull * block_size, covered); None when aligned
    root: str
    block_size: int

    @classmethod
    def build(cls, covered: int, lines: int, digests: bytes, tail: bytes | None, block_size: int | None = None):
        size = block_size or BLOCK_SIZE
        return cls(covered, lines, digests, tail, compute_root(size, covered, lines, digests, tail), size)

    @property
    def nfull(self) -> int:
        return self.covered // self.block_size

    @property
    def tail_start(self) -> int:
        return self.nfull * self.block_size

    def digest(self, index: int) -> bytes:
        return self.digests[index * DIGEST_SIZE:(index + 1) * DIGEST_SIZE]

    def consistent(self) -> bool:
        """Structure and root agree (a damaged or hand-edited row fails this and is never trusted)."""
        return (self.block_size > 0 and self.covered >= 0 and self.lines >= 0
                and len(self.digests) == self.nfull * DIGEST_SIZE
                and (self.tail is None) == (self.covered % self.block_size == 0)
                and (self.tail is None or len(self.tail) == DIGEST_SIZE)
                and self.root == compute_root(self.block_size, self.covered, self.lines, self.digests, self.tail))

    @property
    def hash(self) -> str:
        return HASH_TAG + self.root


@dataclass(frozen=True)
class BlockRecord:
    """A stored table plus what is known about the source it described.

    ``dev``/``ino``/``size``/``mtime_ns`` are advisory signals only: a difference escalates to a full
    verification, equality proves nothing. They are never compared across machines as a guarantee.
    ``full_verified_at`` / ``full_verified_through`` describe the last verification of EVERY block (a past event).
    """

    provider: str
    session_id: str
    path: str
    dev: int
    ino: int
    size: int
    mtime_ns: int
    table: BlockTable
    synced_at: str
    full_verified_at: str | None
    full_verified_through: int | None

    def bound_to(self, cursor) -> bool:
        """The table describes exactly this cursor: same publication, path, offset and root."""
        return (cursor is not None and self.synced_at == cursor["synced_at"] and self.path == cursor["path"]
                and self.table.covered == cursor["byte_offset"] and self.table.consistent()
                and cursor["prefix_hash"] == self.table.hash)


@dataclass(frozen=True)
class BlockDraft:
    """What a publication stores next to the cursor (the ``synced_at`` binding is added by the store)."""

    table: BlockTable
    dev: int
    ino: int
    size: int
    mtime_ns: int
    full_verified_at: str | None
    full_verified_through: int | None


class FastFallback(Exception):
    """The fast check cannot vouch for this request; it must be done (or confirmed) in full instead."""

    def __init__(self, reason: str, block: int | None = None):
        super().__init__(reason)
        self.reason, self.block = reason, block


def new_rng() -> random.Random:
    """Fresh randomness per call: an attacker cannot know which blocks a given check will sample."""
    return random.SystemRandom()


def choose_blocks(nfull: int, recent: int, samples: int, rng) -> list[int]:
    """Indices of complete blocks a fast check compares: the last ``recent`` and ``samples`` random others."""
    recent_blocks = range(max(0, nfull - max(0, recent)), nfull)
    pool = nfull - len(recent_blocks)
    count = min(max(0, samples), pool)
    sampled = rng.sample(range(pool), count) if count else []
    return sorted({*recent_blocks, *sampled})


class BlockScan:
    """Block digests of ``[first_block * block_size, length)`` produced by one streaming pass.

    ``marks`` maps an observed offset to ``(digest of the partial block ending there | None, newline count of
    [0, offset))``; an offset on a block boundary has no partial block. ``base`` is the stored table the scan
    continues (a tail scan starts at its partial block and ``first_block == base.nfull``); ``None`` for a scan
    from byte 0.
    """

    def __init__(self, base: BlockTable | None, first_block: int, block_size: int):
        self.base, self.first_block, self.block_size = base, first_block, block_size
        self.digests: list[bytes] = []
        self.block_lines: list[int] = []  # newlines inside each complete scanned block
        self.marks: dict[int, tuple[bytes | None, int]] = {}
        # Hash state of the partial block at each REQUESTED checkpoint (the cursor offset): a parser that starts
        # there continues the block hash from it instead of reading the block's head again.
        self.states: dict[int, object] = {}
        self.origin = 0  # newlines before first_block * block_size

    def block_digest(self, index: int) -> bytes | None:
        if index < self.first_block:
            return self.base.digest(index) if self.base is not None and index < self.base.nfull else None
        position = index - self.first_block
        return self.digests[position] if position < len(self.digests) else None

    def table_at(self, offset: int) -> BlockTable | None:
        """The table describing ``[0, offset)``; None when ``offset`` was not observed by this scan."""
        mark = self.marks.get(offset)
        nfull = offset // self.block_size
        if mark is None or nfull < self.first_block or nfull - self.first_block > len(self.digests):
            return None
        head = self.base.digests[:self.first_block * DIGEST_SIZE] if self.first_block else b""
        if len(head) != self.first_block * DIGEST_SIZE:
            return None
        digests = head + b"".join(self.digests[:nfull - self.first_block])
        return BlockTable.build(offset, mark[1], digests, mark[0], self.block_size)
