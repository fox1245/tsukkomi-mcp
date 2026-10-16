from __future__ import annotations

from contextlib import closing, contextmanager
import functools
import re
import sqlite3
from pathlib import Path
import unicodedata

from self_directing_mcp.index import cjk
from self_directing_mcp.request_control import check_request, publication, request_operation
from self_directing_mcp.security.mask import mask_secrets

LEGACY_FILE = "sparse.sqlite"          # the first format: FTS5 table that stores a copy of every text
COMPACT_FILE = "sparse-v2.sqlite"      # the compact format: FTS5 contentless table, nothing but the inverted index
COMPACT_FORMAT = "compact-1"           # the value of sparse_meta.format in a compact file
CJK_FILE = "sparse-v3.sqlite"          # the CJK format: the compact table over text whose CJK runs are cut into two-character windows
CJK_FORMAT = "compact-cjk-1"           # the value of sparse_meta.format in a CJK file
CJK_TOKENIZATION = cjk.TOKENIZATION    # sparse_meta.tokenization of a CJK file ("cjk-bigram-1")
CJK_READY, CJK_BUILDING = "ready", "building"   # sparse_meta.state of a CJK file
CJK_TERM_LIMIT = 24                    # terms a query keeps that came from CJK runs
REBUILD_CJK = "python -m self_directing_mcp.maintenance sparse-rebuild --format cjk"
COMPACT_MIN_SQLITE = (3, 43, 0)        # FTS5 contentless_delete
LAZY_LOOKUPS = 4000                    # compact search with an allow list: rows resolved one by one before the joined query takes over


# Characters of the scripts whose words take suffixes or are written without spaces: Hangul (syllables, jamo, compatibility
# jamo, extended A and B), CJK ideographs (extension A, the unified block, compatibility, the supplementary planes),
# Hiragana and Katakana (with the phonetic extensions and the halfwidth Katakana).
_PREFIX_SCRIPT = re.compile(
    "[\u1100-\u11ff\u3040-\u30ff\u3130-\u318f\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\ua960-\ua97f\uac00-\ud7af"
    "\ud7b0-\ud7ff\uf900-\ufaff\uff66-\uff9f\U00020000-\U0002ffff]")
_QUERY_TOKEN = re.compile(r"[\w\-]+")
_HANGUL_SYLLABLES = re.compile("[\uac00-\ud7a3]+")
_PARTICLES = ("에서", "으로", "에게", "까지", "부터", "을", "를", "은", "는", "의")       # longest first
_STEM_MIN = 2                       # syllables a stem must keep for its particle to be cut
PREFIX_TERM_LIMIT = 8               # prefix terms in one query, the stems cut from particles included


def _cut_particle(word: str) -> str | None:
    """The stem of a word made of Hangul syllables only that ends in one of the listed particles, else None."""
    if _HANGUL_SYLLABLES.fullmatch(word) is None:
        return None
    for particle in _PARTICLES:
        if word.endswith(particle):
            stem = word[:-len(particle)]
            return stem if len(stem) >= _STEM_MIN else None
    return None


def _fts_query(raw: str) -> str:
    """The FTS5 MATCH string of a sparse search: quoted terms joined by `` OR ``, ``'""'`` (matches nothing) without any.

    Query side only. The stored index and its tokenizer (``porter unicode61``) are unchanged, so no rebuild is needed and
    both formats share this function; nothing heavy may be imported here (the hook processes import this module).

    The query is first put in NFC (not NFKC, which folds fullwidth Latin and would change ASCII bytes), and every length
    below is counted in NFC code points: a decomposed Hangul syllable of two jamo is one character, not two. Tokens are runs of
    Unicode word characters, ``_`` and ``-`` (``[\\w\\-]+``), so no quote, ``*``, ``^``, ``:``, parenthesis or ``NEAR`` can reach
    the MATCH string.

    * A token without Hangul, CJK ideographs, Hiragana or Katakana becomes the exact term ``"tok"`` when it has two
      characters or more. For ASCII this is the behaviour from before Korean queries were handled, and **the string for
      any query made of ASCII only is byte-identical to it**: one-character tokens skipped, ``foo-bar`` and ``foo_bar``
      kept whole (FTS5 reads them as phrases), duplicates kept, no limit on the number of terms. A non-ASCII token of
      another script (``café``, Cyrillic, Greek) is also exact: unicode61 folds case and diacritics in the query phrase as
      it did in the stored text, and these scripts have spaces, so no prefix is needed.
    * Korean attaches particles and endings to a word, and unicode61 stores one token per space-separated word: the
      text ``검색 색인을 다시`` holds ``검색``, ``색인을`` and ``다시``, so the exact term ``"색인"`` finds nothing but the
      prefix term ``"색인"*`` finds ``색인을``. The porter stemmer leaves Hangul alone, so a prefix is the one query-side way
      to match an inflected word. A token that contains such a character is split on ``-`` and ``_`` (unicode61 stores those
      as separators) and each part is handled alone: an ASCII part as above (``색인_v1`` gives ``"색인"*`` and ``"v1"``), a
      one-character part exactly (``"집"``: one-syllable nouns such as 집, 책 and 돈 exist), a longer part as ``"tok"*`` (the
      ``*`` outside the quotes). ``v2색인`` is one part, since nothing separates it.
    * A part made of Hangul syllables only that ends in a particle of the short list (``에서 으로 에게 까지 부터 을 를 은 는
      의``, longest first) with a stem of at least two syllables also gets the stem as a second prefix term, never instead of
      the word: ``색인을`` gives ``"색인을"* OR "색인"*`` (the word may be a noun that ends that way, or carry a particle
      that is not listed). The particles ``이 가 도 로 와 과 만 에`` are not cut because they end real nouns (도로, 어린이,
      비만, 논의). The two-syllable rule: ``가을은`` also gives ``"가을"*``, ``글을`` is not cut (its stem is one syllable).
    * Bounds, for the terms that come from non-ASCII text only: at most ``PREFIX_TERM_LIMIT`` (8) prefix terms in a query,
      the cut stems counted, and the words in query order get the budget before any stem; a word beyond it is an exact term.
      The same non-ASCII word typed twice is one term. A prefix on a frequent stem costs a scan of every word that starts
      with it, so the bound keeps one query from fanning out.

    Limits, also in the README: a prefix term matches longer words as well (``색인`` finds ``색인표``), and BM25 does not rank
    an exact word above a longer prefix match. Chinese and Japanese, and Korean text written without spaces inside
    code-mixed runs (``v2색인``), are one stored token per unspaced run, so a query matches from the start of that run
    only; an inner word needs ``mode="regex"``. A particle that is not in the list is not cut.
    """
    raw = unicodedata.normalize("NFC", raw)
    # Slots in query order: ("a", ASCII term), ("x", exact non-ASCII word), ("p", word with a prefix-script character).
    slots: list[tuple[str, str]] = []
    for token in _QUERY_TOKEN.findall(raw):
        check_request()
        if _PREFIX_SCRIPT.search(token) is None:
            parts = [token]
        else:
            parts = [part for part in re.split(r"[-_]+", token) if part]
        for part in parts:
            if part.isascii():
                if len(part) >= 2:
                    slots.append(("a", part))
            elif _PREFIX_SCRIPT.search(part) is None:
                if len(part) >= 2:
                    slots.append(("x", part))
            else:
                slots.append(("p", part))
    terms: list[list[str]] = []         # the terms of each slot that is kept: the word, then its stem
    seen: set[str] = set()              # non-ASCII words already handled
    prefixes = 0
    heads: list[tuple[int, str]] = []   # (index in terms, word) of the words that became prefix terms
    for kind, word in slots:
        if kind == "a":
            terms.append([f'"{word}"'])
        elif word not in seen:
            seen.add(word)
            if kind == "p" and len(word) >= 2 and prefixes < PREFIX_TERM_LIMIT:
                prefixes += 1
                heads.append((len(terms), word))
                terms.append([f'"{word}"*'])
            else:
                terms.append([f'"{word}"'])
    present = {term for group in terms for term in group}
    for index, word in heads:
        if prefixes >= PREFIX_TERM_LIMIT:
            break
        stem = _cut_particle(word)
        if stem is not None and f'"{stem}"*' not in present:
            terms[index].append(f'"{stem}"*')
            present.add(f'"{stem}"*')
            prefixes += 1
    flat = [term for group in terms for term in group]
    if not flat:
        return '""'
    return " OR ".join(flat)


def _fts_query_cjk(raw: str) -> str:
    """The FTS5 MATCH string of a search in the CJK format (``sparse-v3.sqlite``, tokenization ``cjk-bigram-1``).

    The stored text had every CJK run cut into overlapping two-character windows (``index/cjk.py``, the one definition of a
    run that the writer and this function share), so a run is queried as the **phrase of its windows**: ``색인을`` becomes
    ``"색인 인을"`` and finds that substring wherever it sits (``원탁회의를`` is found by ``회의``, ``v2색인`` by ``색인``).

    * The query is put in NFC like the text. Without any run character the string is exactly what ``_fts_query`` builds, so
      an ASCII-only query is byte-identical to the one the compact-1 format gets, with the same ids and scores.
    * A run of two or more characters is one quoted phrase term. A run longer than 33 characters keeps its first 32 windows
      (``cjk.MAX_QUERY_WINDOWS``): a longer phrase would cost more and match nothing a 33-character one does not.
    * A run of one character is a prefix term over the windows, ``"집"*``: it finds every window that starts with the
      character and every one-character run. It **misses a document where the character only ends a run** (``서집`` holds the
      windows ``서집`` and no token that starts with ``집``), which is the price of storing windows; search the character with a
      neighbour to find it.
    * The other tokens are the exact terms of ``_fts_query`` (ASCII two characters or longer, kept whole and in order,
      duplicates included; a non-ASCII word of another script, once). A hyphen or underscore that only touches a run is
      dropped from the token next to it, as the old builder cut such tokens at those characters.
    * Terms are joined by `` OR `` as always; at most ``CJK_TERM_LIMIT`` (24) terms come from CJK runs, the same run once.

    No quote, operator or ``*`` can come out of query text: a run consists of run characters only (letters of CJK scripts)
    and a token of ``[\\w\\-]``.

    Known limit, also in the README: FTS5 phrase adjacency ignores what separates two tokens, so two neighbouring runs whose
    border windows form the query's windows (``버스`` next to ``스톱`` for the query ``버스톱``) match although the text never
    holds ``버스톱``. Documents that hold the substring are never missed (tests/test_sparse_cjk.py).
    """
    raw = unicodedata.normalize("NFC", raw)
    if not cjk.has_run_character(raw):
        return _fts_query(raw)
    pieces = cjk.segments(raw)
    terms: list[str] = []
    seen_runs: set[str] = set()         # quoted terms of runs already added
    seen_words: set[str] = set()        # non-ASCII words of other scripts already added
    taken = 0                           # terms that came from runs
    for index, (is_run, piece) in enumerate(pieces):
        check_request()
        if is_run:
            if len(piece) >= 2:
                term = '"' + " ".join(cjk.windows(piece)[:cjk.MAX_QUERY_WINDOWS]) + '"'
            else:
                term = f'"{piece}"*'
            if term not in seen_runs and taken < CJK_TERM_LIMIT:
                seen_runs.add(term)
                taken += 1
                terms.append(term)
            continue
        for token in _QUERY_TOKEN.finditer(piece):      # the pieces alternate: a text piece borders runs, or the query's ends
            word = token.group(0)
            if index > 0 and token.start() == 0:
                word = word.lstrip("-_")
            if index + 1 < len(pieces) and token.end() == len(piece):
                word = word.rstrip("-_")
            if len(word) < 2:
                continue
            if word.isascii():
                terms.append(f'"{word}"')
            elif word not in seen_words:
                seen_words.add(word)
                terms.append(f'"{word}"')
    return " OR ".join(terms) if terms else '""'


class SparseIndex:
    """SQLite FTS5 BM25 sparse index over session chunks: the first ("legacy") format, ``sparse.sqlite``.

    The FTS5 table stores a copy of every (masked) text next to its inverted index. ``CompactSparseIndex`` returns the
    same results from a file about 70 % smaller; see its docstring and README "Sparse index formats".
    """

    format = "legacy"

    def __init__(self, db_path: Path) -> None:
        check_request()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            self._init_schema()
        except Exception:
            self._conn.close()
            raise

    def _init_schema(self) -> None:
        check_request()
        with self._conn:
            self._conn.execute("BEGIN")
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    session_id UNINDEXED,
                    kind UNINDEXED,
                    text,
                    tokenize = 'porter unicode61'
                )
                """
            )
            # Preserve the FTS table; migrate only a compact id -> rowid map.
            check_request()
            exists = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk_rows'").fetchone()
            if not exists:
                self._conn.execute("CREATE TABLE chunk_rows(chunk_id TEXT PRIMARY KEY, doc_rowid INTEGER UNIQUE NOT NULL)")
                self._conn.execute("INSERT INTO chunk_rows SELECT chunk_id,rowid FROM chunks_fts")
            with publication():
                self._conn.commit()


    def clear(self) -> None:
        check_request()
        with self._conn:
            self._conn.execute("DELETE FROM chunks_fts")
            self._conn.execute("DELETE FROM chunk_rows")
            with publication():
                self._conn.commit()

    @request_operation
    def ids(self) -> set[str]:
        out = set()
        with closing(self._conn.execute("SELECT chunk_id FROM chunk_rows")) as rows:
            for row in rows:
                check_request()
                out.add(row[0])
        return out

    @request_operation
    def present(self, ids) -> set[str]:
        """Subset of ``ids`` already indexed; queries only the given ids, in bounded batches."""
        ids = list(ids)
        out = set()
        for start in range(0, len(ids), 500):
            check_request()
            batch = ids[start:start + 500]
            out.update(row[0] for row in self._conn.execute(
                "SELECT chunk_id FROM chunk_rows WHERE chunk_id IN (" + ",".join("?" * len(batch)) + ")", batch))
        return out

    def upsert(self, chunk_id: str, session_id: str, kind: str, text: str) -> None:
        check_request()
        with self._conn:
            row = self._conn.execute("SELECT doc_rowid FROM chunk_rows WHERE chunk_id=?", (chunk_id,)).fetchone()
            if row:
                self._conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (row[0],))
            inserted = self._conn.execute(
                "INSERT INTO chunks_fts(chunk_id,session_id,kind,text) VALUES (?,?,?,?)",
                (chunk_id, session_id, kind, mask_secrets(text)))
            self._conn.execute("INSERT OR REPLACE INTO chunk_rows VALUES (?,?)", (chunk_id, inserted.lastrowid))
            with publication():
                self._conn.commit()

    def delete_ids(self, ids: list[str]) -> None:
        check_request()
        with self._conn:
            for cid in ids:
                check_request()
                row = self._conn.execute("SELECT doc_rowid FROM chunk_rows WHERE chunk_id=?", (cid,)).fetchone()
                if row:
                    self._conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (row[0],))
                    self._conn.execute("DELETE FROM chunk_rows WHERE chunk_id=?", (cid,))
            with publication():
                self._conn.commit()

    def upsert_many(self, chunks) -> None:
        check_request()
        with self._conn:
            for chunk in chunks:
                check_request()
                row = self._conn.execute("SELECT doc_rowid FROM chunk_rows WHERE chunk_id=?", (chunk.chunk_id,)).fetchone()
                if row:
                    self._conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (row[0],))
                inserted = self._conn.execute(
                    "INSERT INTO chunks_fts(chunk_id,session_id,kind,text) VALUES (?,?,?,?)",
                    (chunk.chunk_id, chunk.session_id, chunk.kind, mask_secrets(chunk.text)))
                self._conn.execute("INSERT OR REPLACE INTO chunk_rows VALUES (?,?)",
                                   (chunk.chunk_id, inserted.lastrowid))
            with publication():
                self._conn.commit()

    @request_operation
    def search(
        self,
        query: str,
        top_k: int = 20,
        *,
        session_id: str | None = None,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float, int]]:
        q = _fts_query(query)
        if session_id:
            cur = self._conn.execute(
                """
                SELECT chunk_id, bm25(chunks_fts) AS score
                FROM chunks_fts
                WHERE chunks_fts MATCH ? AND session_id = ?
                ORDER BY score
                LIMIT ?
                """,
                (q, session_id, -1 if allowed_ids is not None else top_k),
            )
        else:
            cur = self._conn.execute(
                """
                SELECT chunk_id, bm25(chunks_fts) AS score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (q, -1 if allowed_ids is not None else top_k),
            )
        out: list[tuple[str, float, int]] = []
        with closing(cur):
            for row in cur:
                check_request()
                if allowed_ids is not None and row["chunk_id"] not in allowed_ids:
                    continue
                if len(out) >= top_k:
                    break
                out.append((row["chunk_id"], float(-row["score"]), len(out) + 1))
        return out

    @request_operation
    def count(self, session_id: str | None = None) -> int:
        if session_id:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM chunks_fts WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        else:
            # chunk_rows mirrors every FTS document in the same transaction; counting the FTS5 table
            # itself scans the whole index while callers may hold the shared index lock.
            row = self._conn.execute("SELECT COUNT(*) AS c FROM chunk_rows").fetchone()
        return int(row["c"]) if row else 0


class SparseFormatError(RuntimeError):
    """The sparse index cannot be opened in the requested format. The message names the repair."""


# ------------------------------------------------------------------------------------------------------------------
# FTS5 statistics
#
# bm25() needs the number of documents and the total number of tokens ("averages": row 1 of ``<fts>_data``, the varints
# nTotalRow and one total per column). A contentless_delete table does not take a deleted row out of them (FTS5 does not
# know its tokens), so after a delete the scores drift away from an index that never held the row. The size of every
# deleted document is still in ``<fts>_docsize``, so the record is corrected in the transaction of the delete, before its
# first FTS5 write (FTS5 keeps its own copy of the totals once it has written and saves it at commit). The record format
# is part of the SQLite file format, which is stable; ``compact_support`` proves the whole mechanism on the running SQLite
# before a compact file is opened, and a self-check (``CompactSparseIndex.verify_statistics``) compares the record with
# the document sizes.
# ------------------------------------------------------------------------------------------------------------------
def _read_varints(blob: bytes) -> list[int]:
    out, i, n = [], 0, len(blob)
    while i < n:
        value = 0
        for position in range(9):
            if i >= n:
                raise SparseFormatError("truncated varint in an FTS5 statistics record")
            byte = blob[i]
            i += 1
            if position == 8:               # the ninth byte of a SQLite varint carries all eight bits
                value = (value << 8) | byte
                break
            value = (value << 7) | (byte & 0x7F)
            if not byte & 0x80:
                break
        out.append(value)
    return out


def _varint(value: int) -> bytes:
    if value < 0 or value >= 1 << 56:
        raise SparseFormatError("FTS5 statistic out of range")
    groups = [value & 0x7F]
    value >>= 7
    while value:
        groups.append(value & 0x7F)
        value >>= 7
    groups.reverse()
    return bytes([g | 0x80 for g in groups[:-1]] + [groups[-1]])


def _retire_statistics(conn: sqlite3.Connection, fts: str, rowids: list[int]) -> None:
    """Take the documents ``rowids`` out of the FTS5 statistics of the contentless table ``fts``. Call it inside the
    write transaction, before the first write to the table and before the documents are deleted."""
    rows = tokens = 0
    for start in range(0, len(rowids), 500):
        batch = rowids[start:start + 500]
        for _id, size in conn.execute(f"SELECT id, sz FROM {fts}_docsize WHERE id IN (" + ",".join("?" * len(batch)) + ")", batch):
            sizes = _read_varints(size)
            if len(sizes) != 1:
                raise SparseFormatError("unexpected FTS5 document size record")
            rows += 1
            tokens += sizes[0]
    if not rows:
        return
    row = conn.execute(f"SELECT block FROM {fts}_data WHERE id=1").fetchone()
    totals = _read_varints(row[0]) if row else []
    if len(totals) != 2 or totals[0] < rows or totals[1] < tokens:
        raise SparseFormatError(f"unexpected FTS5 statistics record {totals} while retiring {rows} documents")
    conn.execute(f"UPDATE {fts}_data SET block=? WHERE id=1", (_varint(totals[0] - rows) + _varint(totals[1] - tokens),))


def sqlite_version_info() -> tuple[int, ...]:
    """The version of the running SQLite (a function so that tests can pose as an old one)."""
    return tuple(sqlite3.sqlite_version_info)


@functools.lru_cache(maxsize=8)
def _statistics_probe(runtime: str) -> tuple[str | None, bool]:
    """Prove on ``runtime`` (the SQLite version, the cache key) what the compact format needs: contentless_delete, and
    bm25 scores after a delete that equal those of an index that never held the row. Returns ``(reason, needs_patch)``:
    a reason when something is missing, else whether this SQLite leaves the statistics stale on delete."""
    try:
        conn = sqlite3.connect(":memory:", isolation_level=None)
        try:
            documents = {1: "alpha beta gamma delta", 2: "alpha alpha epsilon zeta eta", 3: "beta gamma", 4: "alpha theta iota kappa lambda mu",
                         5: "gamma gamma gamma", 6: "alpha"}
            for name in ("ref", "plain", "patched"):
                conn.execute(f"CREATE VIRTUAL TABLE {name} USING fts5(text, content='', contentless_delete=1, tokenize='porter unicode61')")
                for rowid, text in documents.items():
                    if name != "ref" or rowid != 2:
                        conn.execute(f"INSERT INTO {name}(rowid, text) VALUES (?,?)", (rowid, text))
            conn.execute("DELETE FROM plain WHERE rowid=2")
            conn.execute("BEGIN IMMEDIATE")
            _retire_statistics(conn, "patched", [2])
            conn.execute("DELETE FROM patched WHERE rowid=2")
            conn.execute("COMMIT")

            def scores(name):
                return conn.execute(f"SELECT rowid, bm25({name}) AS score FROM {name} WHERE {name} MATCH ? ORDER BY score, rowid",
                                    ('"alpha" OR "gamma" OR "theta"',)).fetchall()

            expected = scores("ref")
            if scores("plain") == expected:
                return None, False
            if scores("patched") == expected:
                return None, True
            return ("this SQLite does not keep FTS5 bm25 statistics exact after a contentless delete and they cannot be corrected here",
                    False)
        finally:
            conn.close()
    except (sqlite3.Error, SparseFormatError) as exc:
        return f"{type(exc).__name__}: {exc}", False


_REPAIR = ("use a Python build with SQLite 3.43 or newer (a newer Python, or a different SQLite library), or keep the previous "
           "format with SELF_DIRECT_SPARSE_FORMAT=legacy")


def compact_support() -> str | None:
    """None when this process can read and write the compact format, else the reason and the repair."""
    version = sqlite_version_info()
    if tuple(version) < COMPACT_MIN_SQLITE:
        return (f"SQLite {'.'.join(map(str, version))} is older than {'.'.join(map(str, COMPACT_MIN_SQLITE))}, which the compact "
                f"sparse format needs (FTS5 contentless_delete): {_REPAIR}")
    reason, _needs_patch = _statistics_probe(sqlite3.sqlite_version)
    if reason is not None:
        return f"the compact sparse format cannot be used with SQLite {sqlite3.sqlite_version} ({reason}): {_REPAIR}"
    return None


class CompactSparseIndex(SparseIndex):
    """The compact format, ``sparse-v2.sqlite``: the same search results as ``SparseIndex`` from a file about 70 % smaller.

    * ``chunks_fts`` is a contentless FTS5 table (``content=''``, ``contentless_delete=1``) over the masked text, with the
      same tokenizer and so the same stemming: it stores the inverted index and document sizes, **no text**.
    * ``chunk_rows(rowid, chunk_id, session_no, kind_no)`` maps a chunk to its FTS rowid (the same number); sessions and kinds
      are small integers resolved through ``sparse_sessions`` and ``sparse_kinds``. ``sparse_meta`` holds the format
      marker and the SQLite version that created the file.

    Equivalence with the legacy format is the contract (tests/test_sparse_compact.py): the same chunks, in the same order
    of writes, give the same ranked ids and the same bm25 scores, including ties (ascending rowid, which is what the
    legacy query returns), after deletes and re-upserts (the statistics are kept exact, see ``_retire_statistics``),
    session and ``allowed_ids`` filters and ``top_k``. The file never holds more text than the legacy one did, since it
    holds none: ``mask_secrets`` still runs on every text before it is tokenized.

    Opening a file in this format on a SQLite that cannot use it raises ``SparseFormatError`` naming the repair; nothing
    is substituted.
    """

    format = "compact"
    marker = COMPACT_FORMAT                 # the value of sparse_meta.format that identifies this class's files

    def __init__(self, db_path: Path) -> None:
        reason = compact_support()          # before the file is created: an unsupported SQLite leaves nothing behind
        if reason is not None:
            raise SparseFormatError(reason)
        self._patch_statistics = _statistics_probe(sqlite3.sqlite_version)[1]
        super().__init__(db_path)

    _TABLES = ("sparse_meta", "chunks_fts", "chunk_rows", "sparse_sessions", "sparse_kinds")

    def _schema_ready(self) -> bool:
        """An existing, complete file of this format: opening it takes no write lock (a hook process that opens the index
        while a rebuild writes must not wait for it)."""
        names = {row[0] for row in self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN (" + ",".join("?" * len(self._TABLES)) + ")", self._TABLES)}
        if names != set(self._TABLES):
            return False
        found = self._conn.execute("SELECT value FROM sparse_meta WHERE key='format'").fetchone()
        if not found or found[0] != self.marker:
            raise SparseFormatError(f"{self.db_path.name} is not a {self.marker} sparse index "
                                    f"(format {found[0] if found else 'missing'!r}); it is left untouched")
        return True

    def _new_meta(self) -> list[tuple[str, str]]:
        """The ``sparse_meta`` rows of a file that is being created."""
        return [("format", self.marker), ("sqlite_version", sqlite3.sqlite_version)]

    def _init_schema(self) -> None:
        check_request()
        if self._schema_ready():
            return
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            existing = self._conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sparse_meta'").fetchone()
            if existing:
                found = self._conn.execute("SELECT value FROM sparse_meta WHERE key='format'").fetchone()
                if not found or found[0] != self.marker:
                    raise SparseFormatError(f"{self.db_path.name} is not a {self.marker} sparse index "
                                            f"(format {found[0] if found else 'missing'!r}); it is left untouched")
            elif self._conn.execute("SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchone():
                raise SparseFormatError(f"{self.db_path.name} holds tables of another format; it is left untouched")
            self._conn.execute("CREATE TABLE IF NOT EXISTS sparse_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID")
            self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='', "
                               "contentless_delete=1, tokenize='porter unicode61')")
            self._conn.execute("CREATE TABLE IF NOT EXISTS chunk_rows(rowid INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE, "
                               "session_no INTEGER NOT NULL, kind_no INTEGER NOT NULL)")
            self._conn.execute("CREATE TABLE IF NOT EXISTS sparse_sessions(session_no INTEGER PRIMARY KEY, session_id TEXT NOT NULL UNIQUE)")
            self._conn.execute("CREATE TABLE IF NOT EXISTS sparse_kinds(kind_no INTEGER PRIMARY KEY, kind TEXT NOT NULL UNIQUE)")
            for key, value in self._new_meta():
                self._conn.execute("INSERT OR IGNORE INTO sparse_meta(key, value) VALUES (?, ?)", (key, value))
            with publication():
                self._conn.commit()

    # ---- writes -----------------------------------------------------------------------------------------------------
    @contextmanager
    def _write(self):
        """One write transaction. The write lock is taken first (the statistics are read and corrected inside it)."""
        check_request()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            with publication():
                self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise

    def _intern(self, table: str, column: str, value: str, local: dict, *, create: bool) -> int | None:
        key = (table, value)
        if key in local:
            return local[key]
        row = self._conn.execute(f"SELECT rowid FROM {table} WHERE {column}=?", (value,)).fetchone()
        number = row[0] if row else (self._conn.execute(f"INSERT INTO {table}({column}) VALUES (?)", (value,)).lastrowid if create else None)
        local[key] = number
        return number

    def _rowids(self, ids) -> list[int]:
        out: list[int] = []
        for start in range(0, len(ids), 500):
            check_request()
            batch = ids[start:start + 500]
            out += [row[0] for row in self._conn.execute(
                "SELECT rowid FROM chunk_rows WHERE chunk_id IN (" + ",".join("?" * len(batch)) + ")", batch)]
        return out

    def _retire(self, ids: list[str]) -> list[int]:
        """The rowids of the stored ``ids``, taken out of the FTS statistics (before any FTS5 write of the transaction)."""
        rowids = self._rowids(ids)
        if rowids and self._patch_statistics:
            _retire_statistics(self._conn, "chunks_fts", rowids)
        return rowids

    def _drop(self, rowids: list[int]) -> None:
        # Highest rowid first. FTS5 records each contentless delete as a tombstone in the segment that holds the row, and
        # tombstones inserted in ascending rowid order made it rebuild its tombstone tables over and over (a 100,000-event
        # Codex-like index grew by 17 % in free pages after deleting 1,000 evenly spaced rows; descending, random and chunk-id
        # order did not).
        for rowid in sorted(rowids, reverse=True):
            check_request()
            self._conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (rowid,))
            self._conn.execute("DELETE FROM chunk_rows WHERE rowid=?", (rowid,))

    def _index_text(self, text: str) -> str:
        """The text as FTS5 indexes it: masked first, so that no raw secret is ever tokenized."""
        return mask_secrets(text)

    def _match(self, query: str) -> str:
        """The FTS5 MATCH string of a query in this format."""
        return _fts_query(query)

    def _add(self, chunk_id: str, session_id: str, kind: str, text: str, local: dict) -> None:
        masked = self._index_text(text)
        session_no = self._intern("sparse_sessions", "session_id", session_id, local, create=True)
        kind_no = self._intern("sparse_kinds", "kind", kind, local, create=True)
        rowid = self._conn.execute("INSERT INTO chunk_rows(chunk_id, session_no, kind_no) VALUES (?,?,?)",
                                   (chunk_id, session_no, kind_no)).lastrowid
        self._conn.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?,?)", (rowid, masked))

    def clear(self) -> None:
        with self._write():
            self._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")     # also resets the statistics
            self._conn.execute("DELETE FROM chunk_rows")

    def upsert(self, chunk_id: str, session_id: str, kind: str, text: str) -> None:
        with self._write():
            self._drop(self._retire([chunk_id]))
            self._add(chunk_id, session_id, kind, text, {})

    def delete_ids(self, ids: list[str]) -> int:
        """Delete the documents ``ids`` (an id that is not stored is ignored) in one transaction, keeping the statistics
        exact. Returns how many documents were actually removed."""
        ids = list(ids)
        check_request()
        with self._write():
            rowids = self._retire(ids)
            self._drop(rowids)
        return len(rowids)

    def upsert_many(self, chunks) -> None:
        latest: dict = {}
        for chunk in chunks:                  # a repeated id keeps its last text, at its last position (as the legacy loop does)
            latest.pop(chunk.chunk_id, None)
            latest[chunk.chunk_id] = chunk
        with self._write():
            self._drop(self._retire(list(latest)))
            local: dict = {}
            for chunk in latest.values():
                check_request()
                self._add(chunk.chunk_id, chunk.session_id, chunk.kind, chunk.text, local)

    # ---- reads ------------------------------------------------------------------------------------------------------
    # Every search ranks by (bm25 score, rowid): ties come out in ascending rowid, which is the order the legacy table's
    # query returns them in (rowids are assigned in write order in both formats). The FTS5 scan drives each query; the
    # chunk map is only consulted for the rows that are needed.
    @request_operation
    def search(
        self,
        query: str,
        top_k: int = 20,
        *,
        session_id: str | None = None,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float, int]]:
        q = self._match(query)
        if allowed_ids is not None and not allowed_ids:
            return []
        if session_id:
            row = self._conn.execute("SELECT session_no FROM sparse_sessions WHERE session_id=?", (session_id,)).fetchone()
            if row is None:
                return []
            return self._search_joined(q, top_k, row[0], allowed_ids)
        if allowed_ids is None:
            return self._search_top(q, top_k)
        found = self._search_lazy(q, top_k, allowed_ids)
        return found if found is not None else self._search_joined(q, top_k, None, allowed_ids)

    def _search_top(self, q: str, top_k: int) -> list[tuple[str, float, int]]:
        """No filter: rank first, then look up the ids of the ``top_k`` rows only."""
        cur = self._conn.execute(
            "SELECT r.chunk_id AS chunk_id, s.score AS score FROM (SELECT rowid AS rid, bm25(chunks_fts) AS score FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY score, rowid LIMIT ?) AS s CROSS JOIN chunk_rows AS r ON r.rowid = s.rid "
            "ORDER BY s.score, s.rid", (q, top_k))
        with closing(cur):
            out = []
            for row in cur:
                check_request()
                out.append((row["chunk_id"], float(-row["score"]), len(out) + 1))
            return out

    def _search_lazy(self, q: str, top_k: int, allowed_ids: set[str]) -> list[tuple[str, float, int]] | None:
        """An allow list (what the engine always passes: the ids of the searched scope): walk the ranking and look up one
        id at a time until ``top_k`` allowed ones are found. Gives up (None) after ``LAZY_LOOKUPS`` rows without being
        done, which is what a short allow list over a frequent term looks like; the joined query is better there."""
        out: list[tuple[str, float, int]] = []
        scanned = 0
        cur = self._conn.execute("SELECT rowid, bm25(chunks_fts) AS score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score, rowid", (q,))
        with closing(cur):
            for rowid, score in cur:
                check_request()
                scanned += 1
                if scanned > LAZY_LOOKUPS:
                    return None
                found = self._conn.execute("SELECT chunk_id FROM chunk_rows WHERE rowid=?", (rowid,)).fetchone()
                if found is None or found[0] not in allowed_ids:      # a document without a map row is unreachable, as in the join
                    continue
                chunk_id = found[0]
                out.append((chunk_id, float(-score), len(out) + 1))
                if len(out) >= top_k:
                    break
        return out

    def _search_joined(self, q: str, top_k: int, session_no: int | None, allowed_ids: set[str] | None) -> list[tuple[str, float, int]]:
        """A session filter (or a short allow list): the chunk map is joined to every match before ranking."""
        sql = ("SELECT r.chunk_id AS chunk_id, bm25(chunks_fts) AS score "
               "FROM chunks_fts CROSS JOIN chunk_rows AS r ON r.rowid = chunks_fts.rowid "   # the FTS5 scan drives; rowid is r's key
               "WHERE chunks_fts MATCH ?")
        args: list = [q]
        if session_no is not None:
            sql += " AND r.session_no = ?"
            args.append(session_no)
        sql += " ORDER BY score, chunks_fts.rowid LIMIT ?"
        args.append(-1 if allowed_ids is not None else top_k)
        cur = self._conn.execute(sql, args)
        out: list[tuple[str, float, int]] = []
        with closing(cur):
            for row in cur:
                check_request()
                if allowed_ids is not None and row["chunk_id"] not in allowed_ids:
                    continue
                if len(out) >= top_k:
                    break
                out.append((row["chunk_id"], float(-row["score"]), len(out) + 1))
        return out

    @request_operation
    def count(self, session_id: str | None = None) -> int:
        if session_id:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM chunk_rows WHERE session_no = (SELECT session_no FROM sparse_sessions WHERE session_id=?)",
                (session_id,)).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM chunk_rows").fetchone()
        return int(row["c"]) if row else 0

    def meta(self) -> dict[str, str]:
        return {row[0]: row[1] for row in self._conn.execute("SELECT key, value FROM sparse_meta")}

    @request_operation
    def verify_statistics(self) -> dict:
        """Compare the FTS5 statistics record with the stored document sizes and the chunk map. ``ok`` is the bar the
        bm25 parity rests on; the rebuild command and the tests call it."""
        rows = tokens = 0
        with closing(self._conn.execute("SELECT sz FROM chunks_fts_docsize")) as cur:
            for (size,) in cur:
                check_request()
                rows += 1
                tokens += _read_varints(size)[0]
        record = self._conn.execute("SELECT block FROM chunks_fts_data WHERE id=1").fetchone()
        totals = _read_varints(record[0]) if record and record[0] else [0, 0]
        mapped = self.count()
        return {"ok": totals == [rows, tokens] and mapped == rows, "documents": rows, "tokens": tokens,
                "recorded": totals, "mapped": mapped}


class CjkSparseIndex(CompactSparseIndex):
    """The CJK format, ``sparse-v3.sqlite``: the compact format over text whose CJK runs are cut into two-character windows.

    Everything but the text is the compact format (the contentless FTS5 table with the same ``porter unicode61`` tokenizer,
    the chunk map, the exact bm25 statistics, ``verify_statistics``); the differences are the two ends of the pipeline:

    * **Write.** ``cjk.pretokenize`` runs on the masked text, so ``색인을 다시`` is indexed as the tokens ``색인``, ``인을``,
      ``다시`` and ``v2색인`` as ``v2``, ``색인`` (``index/cjk.py`` has the exact rules). The document length FTS5 records,
      and so the bm25 statistics that ``_retire_statistics`` keeps exact, is the token count of that rewritten text.
    * **Query.** ``_fts_query_cjk``: a CJK run is the phrase of its windows, so any substring of two or more characters is
      found. A text without CJK characters is indexed and queried exactly as in the compact-1 format (same ids, same scores).

    ``sparse_meta`` holds ``format`` = ``compact-cjk-1``, ``tokenization`` = ``cjk-bigram-1`` and ``state``: ``ready``, or
    ``building`` while ``sparse-rebuild --format cjk`` has not finished the file. A process opens a file for answers only
    when the tokenization is the one it knows and the state is ``ready``; otherwise ``SparseFormatError`` names the repair,
    and nothing is substituted. ``build=True`` (the rebuild command) also opens a ``building`` file, and creates a new file
    in that state. A file that an engine creates on its own (a fresh install, or the explicit setting) is ``ready`` from the
    start: like the compact index it is filled lazily and every search repairs its scope before it answers.
    """

    format = "cjk"
    marker = CJK_FORMAT

    def __init__(self, db_path: Path, *, build: bool = False) -> None:
        self.build = build
        super().__init__(db_path)

    def _new_meta(self) -> list[tuple[str, str]]:
        return super()._new_meta() + [("tokenization", CJK_TOKENIZATION), ("state", CJK_BUILDING if self.build else CJK_READY)]

    def _check_usable(self) -> None:
        """Refuse an existing file this process may not answer from, before anything is written to it."""
        meta = self.meta()
        if meta.get("format") != self.marker:
            return                                  # _init_schema's own check names the foreign format
        name = self.db_path.name
        if meta.get("tokenization") != CJK_TOKENIZATION:
            raise SparseFormatError(
                f"{name} uses the tokenization {meta.get('tokenization')!r}, which this version does not know (it writes and reads "
                f"{CJK_TOKENIZATION!r}); the file is left untouched. Use a version that knows it, or set "
                f"SELF_DIRECT_SPARSE_FORMAT=compact (sparse-v2.sqlite)")
        state = meta.get("state")
        if state == CJK_BUILDING and not self.build:
            raise SparseFormatError(
                f"{name} is not finished (state {CJK_BUILDING!r}): run `{REBUILD_CJK}` to finish it, or set "
                f"SELF_DIRECT_SPARSE_FORMAT=compact to use sparse-v2.sqlite meanwhile")
        if state not in (CJK_READY, CJK_BUILDING):
            raise SparseFormatError(f"{name} has the unknown state {state!r}; the file is left untouched")

    def _init_schema(self) -> None:
        check_request()
        if self._conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sparse_meta'").fetchone():
            self._check_usable()                    # before anything is written to an existing file
        super()._init_schema()
        self._check_usable()                        # a file another process created while this one waited for the write lock

    def _index_text(self, text: str) -> str:
        return cjk.pretokenize(mask_secrets(text))          # mask first: the secret patterns see the text as it was written

    def _match(self, query: str) -> str:
        return _fts_query_cjk(query)

    # ---- the state of the file ------------------------------------------------------------------------------------
    def state(self) -> str | None:
        return self.meta().get("state")

    def set_state(self, state: str) -> None:
        """Set ``sparse_meta.state`` (one transaction; nothing is written when it already holds the value)."""
        if state not in (CJK_READY, CJK_BUILDING):
            raise ValueError(f"unknown sparse state {state!r}")
        if self.state() == state:
            return
        with self._write():
            self._conn.execute("INSERT OR REPLACE INTO sparse_meta(key, value) VALUES ('state', ?)", (state,))

    def finish_build(self, documents: int) -> bool:
        """Set the state to ``ready`` in one transaction, only when the file holds exactly ``documents`` documents (the
        canonical store's count, which the caller has just compared under the publication guard). False: nothing changed."""
        with self._write():
            held = self._conn.execute("SELECT COUNT(*) FROM chunk_rows").fetchone()[0]
            if held != documents:
                return False
            self._conn.execute("INSERT OR REPLACE INTO sparse_meta(key, value) VALUES ('state', ?)", (CJK_READY,))
        return True


FORMAT_FILES = {"legacy": LEGACY_FILE, "compact": COMPACT_FILE, "cjk": CJK_FILE}
SPARSE_SETTINGS = ("auto", "legacy", "compact", "cjk")


def sparse_file(index_dir, fmt: str) -> Path:
    return Path(index_dir) / FORMAT_FILES[fmt]


def look_at_sparse_file(path, *, count: bool = True) -> dict | None:
    """What a sparse file holds, read through a read-only connection (the file is never created, written or recovered):
    ``{"documents", "format", "tokenization", "state"}`` (the last three from ``sparse_meta``, None for a legacy file, which
    has none). ``documents`` is the size of the id map, or with ``count=False`` just 1 or 0 (does it hold anything: an
    O(1) probe for the open path). None when the file does not exist or holds no table at all (an empty file is a fresh one);
    ``documents`` is None when it cannot be read or has no id map."""
    path = Path(path)
    if not path.exists():
        return None
    info = {"documents": None, "format": None, "tokenization": None, "state": None}
    try:
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.25)
    except (sqlite3.Error, ValueError):
        return info
    try:
        try:
            if conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone() is None:
                return None
        except sqlite3.Error:
            return info
        try:
            info["documents"] = int(conn.execute("SELECT COUNT(*) FROM chunk_rows").fetchone()[0] if count else
                                    conn.execute("SELECT EXISTS(SELECT 1 FROM chunk_rows)").fetchone()[0])
        except sqlite3.Error:
            pass
        try:
            meta = dict(conn.execute("SELECT key, value FROM sparse_meta").fetchall())
            info.update(format=meta.get("format"), tokenization=meta.get("tokenization"), state=meta.get("state"))
        except sqlite3.Error:
            pass                                    # a legacy file has no sparse_meta
        return info
    finally:
        conn.close()


def _cjk_unusable(info: dict, *, build: bool) -> str | None:
    """Why ``sparse-v3.sqlite`` (as ``look_at_sparse_file`` saw it) cannot answer for this process, else None."""
    if info["documents"] is None and info["format"] is None:
        return "sparse-v3.sqlite could not be read"
    if info["format"] != CJK_FORMAT:
        return f"sparse-v3.sqlite is not a {CJK_FORMAT} file (format {info['format']!r})"
    if info["tokenization"] != CJK_TOKENIZATION:
        return (f"sparse-v3.sqlite uses the tokenization {info['tokenization']!r}, which this version does not know "
                f"(it uses {CJK_TOKENIZATION!r})")
    if info["state"] == CJK_BUILDING:
        return None if build else f"sparse-v3.sqlite is not finished (state {CJK_BUILDING!r}); run `{REBUILD_CJK}` to finish it"
    if info["state"] != CJK_READY:
        return f"sparse-v3.sqlite has the unknown state {info['state']!r}"
    return None


def choose_auto_format(index_dir, *, build: bool = False) -> tuple[str, dict]:
    """The format ``auto`` uses on a SQLite that can use the compact formats, and why: ``(format, details)`` with
    ``details["choice"]`` (the reason in words), ``details["cjk_problem"]`` (True when ``sparse-v3.sqlite`` exists and is
    unusable, which ``audit_status`` flags) and ``details["cjk_hint"]`` (what to do about the CJK format, or None).

    1. ``sparse-v3.sqlite`` is ready and its tokenization is known: it.
    2. It exists but is not usable (still ``building``, an unknown tokenization, unreadable): the compact format, visibly.
    3. It does not exist and an earlier file already holds data: that earlier choice (the compact format, as in 0.8.0) is kept,
       and the hint says how to build the CJK format. Nothing is rebuilt by itself.
    4. It does not exist and no sparse file holds a document (a fresh install): the CJK format.

    ``build=True`` is the rebuild command: a ``building`` file is the one to continue."""
    index_dir = Path(index_dir)
    v3 = look_at_sparse_file(index_dir / CJK_FILE, count=False)
    if v3 is not None:
        problem = _cjk_unusable(v3, build=build)
        if problem is None:
            return "cjk", {"choice": f"auto: {CJK_FILE} is {v3['state']}" + (" and being continued" if v3["state"] == CJK_BUILDING else ""),
                           "cjk_problem": False, "cjk_hint": None}
        return "compact", {"choice": f"auto: the compact format ({COMPACT_FILE}), because {problem}", "cjk_problem": True,
                           "cjk_hint": problem}
    older = [name for name in (COMPACT_FILE, LEGACY_FILE)       # a file that cannot be read counts as holding data: never guess
             if (found := look_at_sparse_file(index_dir / name, count=False)) is not None and found["documents"] != 0]
    if older:
        hint = f"{CJK_FILE} is not built: run `{REBUILD_CJK}` to build it from the canonical store (auto uses it once it is ready)"
        return "compact", {"choice": f"auto: the compact format ({COMPACT_FILE}); {older[0]} already holds data, so the CJK format "
                                     f"is not started by itself", "cjk_problem": False, "cjk_hint": hint}
    return "cjk", {"choice": f"auto: no sparse file holds data yet, so the CJK format ({CJK_FILE}) is used", "cjk_problem": False,
                   "cjk_hint": None}


def open_sparse_index(index_dir, setting: str = "auto", *, build: bool = False) -> tuple[SparseIndex, dict]:
    """Open the sparse index of ``index_dir`` in the format ``setting`` asks for (``auto``, ``legacy``, ``compact``, ``cjk``).

    ``compact`` and ``cjk`` raise ``SparseFormatError`` when this SQLite cannot use them, and ``cjk`` also when the file is
    not finished (state ``building``) or uses a tokenization this version does not know. ``auto`` follows
    ``choose_auto_format`` (the CJK format once it is ready or on a fresh install, else the compact format, else the legacy
    one on a SQLite that cannot use either) and says why in the returned resolution (``choice``, ``reason``, ``cjk_hint``,
    ``cjk_problem``), so that ``audit_status`` shows it. ``build=True`` is for the rebuild command: a CJK file in state
    ``building`` is opened (and a new one is created in that state). The file of a format that is not opened is never
    touched."""
    if setting not in SPARSE_SETTINGS:
        raise ValueError(f"SELF_DIRECT_SPARSE_FORMAT must be auto, legacy, compact or cjk, not {setting!r}")
    reason = None
    details: dict = {"choice": f"SELF_DIRECT_SPARSE_FORMAT={setting}", "cjk_problem": False, "cjk_hint": None}
    if setting == "legacy":
        fmt = "legacy"
    else:
        reason = compact_support()
        if reason is not None:
            if setting != "auto":
                raise SparseFormatError(reason)
            fmt = "legacy"
            details["choice"] = "auto: the legacy format, because this SQLite cannot use the compact formats"
        elif setting == "auto":
            fmt, details = choose_auto_format(index_dir, build=build)
        else:
            fmt = setting
    if fmt == "cjk":
        index = CjkSparseIndex(sparse_file(index_dir, fmt), build=build)
    else:
        index = (CompactSparseIndex if fmt == "compact" else SparseIndex)(sparse_file(index_dir, fmt))
    return index, {"setting": setting, "format": fmt, "reason": reason, **details}


def count_sparse_file(path) -> int | None:
    """Documents in a sparse file this process does not use (``audit_status`` shows every format). Read-only: the file is
    never created, written or recovered, and only its id map is read, not the FTS5 tables. None when it cannot be read."""
    try:
        conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=0.25)
    except (sqlite3.Error, ValueError):
        return None
    try:
        return int(conn.execute("SELECT COUNT(*) FROM chunk_rows").fetchone()[0])
    except sqlite3.Error:
        return None
    finally:
        conn.close()
