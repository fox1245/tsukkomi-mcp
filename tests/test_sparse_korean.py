"""Sparse queries in Korean and other non-ASCII text (``index/sparse.py::_fts_query``).

The index side was always fine (``porter unicode61`` stores Hangul, one token per space-separated word); the query builder
dropped every non-ASCII character, so a Korean query matched nothing. The builder now makes prefix terms for Hangul, CJK
and kana words (a Korean word carries its particle: the stored token is ``색인을``, never ``색인``), cuts a short list of
particles to add the stem, and leaves every ASCII-only query byte-identical to what it was. All text here is synthetic.
"""
from __future__ import annotations

import random
import re
import sqlite3
import unicodedata

import pytest

from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index import sparse as sparse_module
from self_directing_mcp.index.sparse import CompactSparseIndex, PREFIX_TERM_LIMIT, SparseIndex, _fts_query
from sparse_support import Doc, build
from test_reliability import SID, message, write

needs_compact = pytest.mark.skipif(sparse_module.compact_support() is not None,
                                   reason=f"this SQLite cannot use the compact format: {sparse_module.compact_support()}")

NFD = lambda text: unicodedata.normalize("NFD", text)           # noqa: E731


def old_fts_query(raw: str) -> str:
    """``_fts_query`` as it was before Korean queries were handled: the reference for every pure-ASCII query."""
    toks = re.findall(r"[A-Za-z0-9_\-]+", raw)
    cleaned: list[str] = []
    for t in toks:
        if len(t) < 2:
            continue
        cleaned.append(f'"{t}"')
    if not cleaned:
        return '""'
    return " OR ".join(cleaned)


# ---------------------------------------------------------------- ASCII-only queries are byte-identical to the old ones
ASCII_QUERIES = [
    "", " ", "\t\n", "a", "ab", "a b", "a bb c ddd", "foo-bar", "foo_bar", "foo-bar_baz qux", "foo foo foo", "Foo foo FOO",
    "--no-verify", "pre-commit", "a-", "-a", "--", "__", "a_b-c d", "1 2 3 45", "OR AND NOT", "NEAR(a b) NEAR/3", '"', '"quoted phrase"',
    "col:value", "x*", "^x", "(a OR b) AND c", "snake_case_name kebab-case-name", "tab\tsep\nnl", "sk-***", "x" * 300 + " y",
    "path/to/file.py:12", "https://example.com/a?b=c&d=e", " ".join(f"term{i}" for i in range(60)), "error " * 40,
]


def test_pure_ascii_queries_are_byte_identical_to_the_old_builder():
    for query in ASCII_QUERIES:
        assert _fts_query(query) == old_fts_query(query), query
    assert _fts_query("") == '""' and _fts_query("a b") == '""'
    assert _fts_query("foo-bar foo_bar foo foo x") == '"foo-bar" OR "foo_bar" OR "foo" OR "foo"'      # no dedupe, one character skipped
    assert _fts_query(" ".join(f"w{i}" for i in range(30))).count(" OR ") == 29                       # no cap on ASCII terms


def test_a_seeded_fuzz_of_pure_ascii_queries_is_byte_identical():
    rng = random.Random(40)
    alphabet = list("abcXYZ019_-  \t\n\"'()*^:.,;!@#$%&+=/\\|<>?[]{}~`") + ["OR", "AND", "NOT", "NEAR", "ab", "x", "--", "__"]
    for _ in range(4000):
        query = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 40)))
        assert query.isascii()
        assert _fts_query(query) == old_fts_query(query), repr(query)


def test_latin_tokens_of_a_mixed_query_keep_their_old_terms():
    rng = random.Random(41)
    latin = ["index", "Foo-Bar", "snake_case", "x", "v2", "a1", "--flag", "ID42", "pre-commit"]
    korean = ["검색", "색인을", "집", "테스트가", "도로는", "日本語", "カタカナ"]
    for _ in range(300):
        picks = [(rng.choice(latin), True) if rng.random() < 0.5 else (rng.choice(korean), False) for _ in range(rng.randrange(1, 9))]
        mixed = _fts_query(" ".join(text for text, _ in picks))
        ascii_terms = [term for term in mixed.split(" OR ") if term.isascii() and term != '""']
        old = old_fts_query(" ".join(text for text, is_latin in picks if is_latin))
        assert ascii_terms == ([] if old == '""' else old.split(" OR ")), picks


def test_the_token_loop_still_checks_the_request(monkeypatch):
    calls = []
    monkeypatch.setattr(sparse_module, "check_request", lambda: calls.append(1))
    _fts_query("alpha beta 검색 gamma-delta")
    assert len(calls) == 4                                      # one check per token, as before


# ---------------------------------------------------------------- Korean, CJK and kana words
def test_korean_words_become_prefix_terms_and_one_syllable_words_stay_exact():
    assert _fts_query("검색 색인") == '"검색"* OR "색인"*'
    assert _fts_query("검색") == '"검색"*'
    assert _fts_query("집") == '"집"' and _fts_query("책 돈") == '"책" OR "돈"'        # one-syllable nouns exist
    assert _fts_query("index 검색") == '"index" OR "검색"*'
    assert _fts_query("ㅋㅋㅋ") == '"ㅋㅋㅋ"*'                                          # compatibility jamo
    assert _fts_query("日本語のテキスト") == '"日本語のテキスト"*'                          # ideographs and kana
    assert _fts_query("カタカナ") == '"カタカナ"*' and _fts_query("ﾃｽﾄ") == '"ﾃｽﾄ"*'     # halfwidth Katakana is not folded
    assert _fts_query("本") == '"本"' and _fts_query("検索 本") == '"検索"* OR "本"'
    assert _fts_query("\U00020bb7野家") == '"\U00020bb7野家"*'                          # a supplementary-plane ideograph


def test_nfd_input_is_the_nfc_query_and_lengths_count_nfc_characters():
    assert _fts_query(NFD("검색")) == _fts_query("검색") == '"검색"*'
    assert _fts_query(NFD("색인을 다시")) == _fts_query("색인을 다시")
    for syllable in ("가", "각", "집", "책", "돈"):             # 2 or 3 jamo when decomposed, one character once composed
        assert len(NFD(syllable)) >= 2
        assert _fts_query(NFD(syllable)) == f'"{syllable}"', syllable
    assert _fts_query(NFD("집에서")) == '"집에서"*'                                      # the stem would be a single syllable
    assert _fts_query(NFD("café")) == _fts_query("café") == '"café"'


@pytest.mark.parametrize("particle", ["에서", "으로", "에게", "까지", "부터", "을", "를", "은", "는", "의"])
def test_each_listed_particle_adds_the_stem_as_a_second_prefix_term(particle):
    assert _fts_query(f"색인{particle}") == f'"색인{particle}"* OR "색인"*'
    assert _fts_query(f"도서관{particle}") == f'"도서관{particle}"* OR "도서관"*'


@pytest.mark.parametrize("particle", ["이", "가", "도", "로", "와", "과", "만", "에"])
def test_particles_that_end_real_nouns_are_not_cut(particle):
    assert _fts_query(f"색인{particle}") == f'"색인{particle}"*'


def test_particle_rules_edge_cases():
    assert _fts_query("도로는") == '"도로는"* OR "도로"*'
    assert _fts_query("가을은") == '"가을은"* OR "가을"*'                                 # two syllables are left
    assert _fts_query("글을") == '"글을"*' and _fts_query("집에서") == '"집에서"*'         # one syllable is not enough
    assert _fts_query("을") == '"을"' and _fts_query("은는") == '"은는"*'
    assert _fts_query("도로") == '"도로"*' and _fts_query("어린이") == '"어린이"*'
    assert _fts_query("비만") == '"비만"*' and _fts_query("논의") == '"논의"*'
    assert _fts_query("도서관에서는") == '"도서관에서는"* OR "도서관에서"*'                 # one particle, not a chain
    assert _fts_query("서울에서") == '"서울에서"* OR "서울"*'                              # the longer particle wins
    assert _fts_query("v2색인을") == '"v2색인을"*'                                       # not only Hangul syllables
    assert _fts_query("ㅋㅋ을") == '"ㅋㅋ을"*' and _fts_query("日本語を") == '"日本語を"*'
    assert _fts_query("색인을 색인") == '"색인을"* OR "색인"*'                           # a stem already in the query is not repeated
    assert _fts_query("색인 색인을") == '"색인"* OR "색인을"*'


def test_mixed_words_and_separators():
    assert _fts_query("v2색인") == '"v2색인"*'                    # nothing separates v2 from the Hangul: one stored token
    assert _fts_query("색인_v1") == '"색인"* OR "v1"'
    assert _fts_query("v1-색인") == '"v1" OR "색인"*'
    assert _fts_query("foo-bar-색인") == '"foo" OR "bar" OR "색인"*'
    assert _fts_query("x-색인") == '"색인"*'                      # a one-character ASCII part is skipped as before
    assert _fts_query("색인--검색__테스트") == '"색인"* OR "검색"* OR "테스트"*'
    assert _fts_query("색인-") == '"색인"*' and _fts_query("-색인_") == '"색인"*'
    assert _fts_query("café_색인") == '"café" OR "색인"*'
    assert _fts_query("API를") == '"API를"*'


def test_non_hangul_non_ascii_words_are_exact_terms():
    assert _fts_query("café") == '"café"' and _fts_query("Résumé naïve") == '"Résumé" OR "naïve"'
    assert _fts_query("привет Мир") == '"привет" OR "Мир"' and _fts_query("αβγ") == '"αβγ"'
    assert _fts_query("é") == '""' and _fts_query("é a") == '""'                       # one character, as for ASCII
    assert _fts_query("naïve-bayes") == '"naïve-bayes"'
    assert _fts_query("ＡＢＣ") == '"ＡＢＣ"'                                            # NFC only: fullwidth Latin is not folded
    assert _fts_query("café café") == '"café"'                                         # non-ASCII words are de-duplicated ...
    assert _fts_query("cafe cafe") == '"cafe" OR "cafe"'                               # ... ASCII terms never are
    assert _fts_query("Café café") == '"Café" OR "café"'


def test_no_token_still_gives_the_empty_phrase():
    for query in ("", "   ", "a", "\u00e9", "!!! ???", "\u200b", "\U0001f600 \U0001f600", "\u0301", "- _ -"):
        assert _fts_query(query) == '""', query


def test_prefix_terms_are_capped_and_the_cap_keeps_query_order():
    words = [chr(0xAC00 + 7 * i) + chr(0xAC00 + 7 * i + 1) for i in range(12)]
    out = _fts_query(" ".join(words)).split(" OR ")
    assert out == [f'"{w}"*' for w in words[:PREFIX_TERM_LIMIT]] + [f'"{w}"' for w in words[PREFIX_TERM_LIMIT:]]
    assert PREFIX_TERM_LIMIT == 8
    # The words get the budget before any stem: with eight words, no stem is cut ...
    inflected = [w + "을" for w in words[:8]]
    assert _fts_query(" ".join(inflected)).split(" OR ") == [f'"{w}"*' for w in inflected]
    # ... and with six, the two remaining prefix terms go to the first two stems, in query order.
    six = [w + "을" for w in words[:6]]
    out = _fts_query(" ".join(six)).split(" OR ")
    assert out == [f'"{six[0]}"*', f'"{words[0]}"*', f'"{six[1]}"*', f'"{words[1]}"*'] + [f'"{w}"*' for w in six[2:]]
    assert sum(term.endswith("*") for term in out) == 8
    # A word typed twice is one term, and costs no budget.
    assert _fts_query(" ".join(words[:8] + [words[0], words[9]])).split(" OR ") == \
        [f'"{w}"*' for w in words[:8]] + [f'"{words[9]}"']
    # ASCII terms are not capped and do not use the budget.
    ascii_terms = [f"term{i}" for i in range(30)]
    out = _fts_query(" ".join(ascii_terms + words[:10])).split(" OR ")
    assert out[:30] == [f'"{t}"' for t in ascii_terms] and sum(term.endswith("*") for term in out) == 8 and len(out) == 40


def test_the_exact_and_prefix_terms_of_one_word_do_not_repeat():
    assert _fts_query("집 집 집에서 집에서") == '"집" OR "집에서"*'
    assert _fts_query("色 色") == '"色"'


GRAMMAR = re.compile(r'(?:"[\w\-]+"\*? OR )*"[\w\-]+"\*?|""')


def test_operator_characters_never_reach_the_match_string_and_never_raise():
    rng = random.Random(42)
    pieces = list("abcXYZ019_- \t\n\"'()*^:.,;!@#$%&+=/\\|<>?[]{}~`") + list("검색인을집에서은는의가이ㅋ日本語テスト検索ﾃｽﾄ") + [
        "\u00e9", "e\u0301", "\u1100\u1161", "\u200b", "\u0301", "\U0001f600", "\u00df", "\u0130", "\u01c5", "\u00b2", "\u0661\u0662", "\u0e01\u0e47", "\u0939\u093f\u0928\u094d\u0926\u0940", "NEAR", " OR ", " AND ", " NOT ",
        "\u00a0", "\uff02", "\uff0a"]
    queries = ['"검색" OR (색인*) NEAR(a b) ^x col:y', "검색*", '색인"', "{col}: 검색", "NEAR(검색 색인, 2)", '"', "*", "^", ":", "()"]
    queries += ["".join(rng.choice(pieces) for _ in range(rng.randrange(1, 30))) for _ in range(300)]
    con = sqlite3.connect(":memory:")
    con.execute("CREATE VIRTUAL TABLE t USING fts5(text, tokenize='porter unicode61')")
    con.executemany("INSERT INTO t(text) VALUES (?)", [("검색 색인을 다시",), ("index café 日本語",), ("a b c",)])
    for query in queries:
        match = _fts_query(query)
        assert GRAMMAR.fullmatch(match), (query, match)
        con.execute("SELECT rowid FROM t WHERE t MATCH ?", (match,)).fetchall()      # a malformed MATCH raises sqlite3.OperationalError
    con.close()


# ---------------------------------------------------------------- the real indexes, both formats
KO_DOCS = {
    "stem": "검색 색인을 다시 만들었다",
    "subject": "색인이 깨져서 검색 실패했다",
    "bare": "색인 점검 끝났다",
    "longer": "색인표 목록 확인했다",
    "mixed": "index 검색 결과 확인",
    "english": "unrelated english words only",
    "home": "집에서 책을 읽었다",
    "cafe": "Café au lait and a résumé",
    "code": "v2색인 재작성",
    "api": "API를 호출했다",
    "tail": "우리 도로는 넓다 도로명 주소",
    "japanese": "日本語のテキスト 検索",
}


def ko_corpus():
    return [Doc(f"codex:s{n % 2}:{n}:0:{n:016x}", f"s{n % 2}", "turn", text) for n, text in enumerate(KO_DOCS.values())]


def ids_of(docs, *keys):
    by_key = dict(zip(KO_DOCS, (d.chunk_id for d in docs)))
    return {by_key[k] for k in keys}


# What each query must find in the corpus above, in both formats (a prefix finds longer words, an exact term does not).
EXPECTED = [
    ("검색", ("stem", "subject", "mixed")),
    ("색인", ("stem", "subject", "bare", "longer")),             # 색인을, 색인이, 색인, 색인표; not the unspaced run v2색인
    ("색인을", ("stem", "subject", "bare", "longer")),           # the word itself, and its stem as a prefix
    ("index 검색", ("stem", "subject", "mixed")),
    ("검색 색인", ("stem", "subject", "bare", "longer", "mixed")),
    ("집에서", ("home",)),
    ("집", ()),                                                  # an exact one-syllable term: stored words are 집에서, not 집
    ("café", ("cafe",)), ("CAFE résumé", ("cafe",)), ("Resume", ("cafe",)),
    ("v2색인", ("code",)), ("색인재", ()), ("api를", ("api",)), ("API를", ("api",)),
    ("도로", ("tail",)), ("도로는", ("tail",)), ("도로명", ("tail",)),
    ("日本語", ("japanese",)), ("検索", ("japanese",)), ("本語", ()),
    ("색인_v2", ("stem", "subject", "bare", "longer")),
    ("zzzqqq 없는말", ()),
]


@needs_compact
def test_the_issue_repro_on_both_formats_with_identical_results(tmp_path):
    docs = ko_corpus()
    legacy, compact = build(tmp_path / "legacy.sqlite", docs, SparseIndex), build(tmp_path / "compact.sqlite", docs, CompactSparseIndex)
    try:
        for query, keys in EXPECTED:
            a, b = legacy.search(query, 20), compact.search(query, 20)
            assert {h[0] for h in a} == ids_of(docs, *keys), (query, "legacy")
            assert {h[0] for h in b} == ids_of(docs, *keys), (query, "compact")
            assert a == b, (query, a, b)                                    # ids, scores (to the last bit) and ranks
            for hits in (a, b):
                assert [h[2] for h in hits] == list(range(1, len(hits) + 1))
        assert any(legacy.search(q, 20) for q, _ in EXPECTED[:6])
    finally:
        legacy._conn.close()
        compact._conn.close()


@needs_compact
def test_filters_and_top_k_give_identical_results_for_korean_queries(tmp_path):
    docs = ko_corpus()
    legacy, compact = build(tmp_path / "legacy.sqlite", docs, SparseIndex), build(tmp_path / "compact.sqlite", docs, CompactSparseIndex)
    try:
        allowed = {d.chunk_id for d in docs[::2]}
        for query in ("검색", "색인을", "index 검색", "café", "v2색인", "도로는"):
            for top_k in (1, 3, 20):
                for kwargs in ({}, {"session_id": "s0"}, {"session_id": "s1"}, {"allowed_ids": allowed}, {"session_id": "s1", "allowed_ids": allowed}):
                    assert legacy.search(query, top_k, **kwargs) == compact.search(query, top_k, **kwargs), (query, top_k, kwargs)
        assert legacy.search("검색", 1, session_id="s0") and not legacy.search("검색", 5, allowed_ids=set())
    finally:
        legacy._conn.close()
        compact._conn.close()


@needs_compact
def test_a_longer_word_is_not_ranked_below_the_exact_word(tmp_path):
    """The documented limit of a prefix term: BM25 counts ``색인표`` like ``색인``, so the exact word has no advantage. The
    two documents have the same length here and get the same score."""
    docs = ko_corpus()
    compact = build(tmp_path / "compact.sqlite", docs, CompactSparseIndex)
    try:
        score = {h[0]: h[1] for h in compact.search("색인", 20)}
        bare, longer = ids_of(docs, "bare").pop(), ids_of(docs, "longer").pop()
        assert score[bare] == score[longer] > 0
    finally:
        compact._conn.close()


# ---------------------------------------------------------------- the engine: sparse and hybrid search_history
def make_engine(tmp_path, monkeypatch, texts, *, local_only):
    for name in ("SELF_DIRECT_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER", "SELF_DIRECT_GROKBOT_TRANSCRIPTS_DIR",
                 "SELF_DIRECT_SPARSE_FORMAT"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "sessions"
    root.mkdir()
    write(root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl", [{"type": "session_meta", "payload": {"id": SID}}] + [message(t) for t in texts])
    engine = SelfDirectEngine(Settings(codex_sessions_dir=root, index_dir=tmp_path / "index", use_fake_embedder=True, local_only=local_only,
                                       _env_file=None))
    engine.ensure_ready()
    assert engine.sync_session(session_id=SID, embed=not local_only)["ok"]
    return engine


ENGINE_TEXTS = ["검색 색인을 다시 만들었다", "english filler about nothing", "집에서 쉬었다", "Café au lait", "plain index words"]


@pytest.mark.parametrize("local_only", [True, False])
def test_search_history_sparse_finds_what_regex_finds(tmp_path, monkeypatch, local_only):
    engine = make_engine(tmp_path, monkeypatch, ENGINE_TEXTS, local_only=local_only)
    try:
        for sparse_query, regex_query in (("색인", "색인"), ("색인을", "색인을"), ("검색 색인", "검색 색인"), ("집에서", "집에서"), ("café", "Café")):
            by_regex = engine.search_history(regex_query, session_id=SID, mode="regex")["hits"]
            by_sparse = engine.search_history(sparse_query, session_id=SID, mode="sparse")["hits"]
            assert by_regex and [h["chunk_id"] for h in by_sparse] == [h["chunk_id"] for h in by_regex], sparse_query
        mixed = engine.search_history("index 검색", session_id=SID, mode="sparse")["hits"]
        assert len(mixed) == 2                                        # the English word and the Korean one
        korean = engine.search_history("색인", session_id=SID, mode="sparse")["hits"]
        assert len(korean) == 1 and korean[0]["sparse_rank"] == 1
        assert [h["chunk_id"] for h in engine.search_history("색인", mode="sparse")["hits"]] == [korean[0]["chunk_id"]]     # all sessions too
        assert engine.search_history("없는말", session_id=SID, mode="sparse")["hits"] == []
    finally:
        engine.close()


def test_search_history_hybrid_uses_the_korean_sparse_channel(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch, ENGINE_TEXTS, local_only=False)
    try:
        hits = engine.search_history("색인", session_id=SID, mode="hybrid")["hits"]
        ranked = [h for h in hits if h["sparse_rank"] is not None]
        assert len(ranked) == 1 and ranked[0]["sparse_rank"] == 1 and ranked[0]["chunk_id"] == hits[0]["chunk_id"]
    finally:
        engine.close()
