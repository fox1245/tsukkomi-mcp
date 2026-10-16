"""The CJK sparse format (``sparse-v3.sqlite``, format ``compact-cjk-1``, tokenization ``cjk-bigram-1``): the pre-tokenizer
(``index/cjk.py``), the query builder (``_fts_query_cjk``), real FTS5 round trips on the index class, a seeded property test
against substring containment, the bm25 statistics, and size and time guards. Formats, selection, the rebuild command and the
engine are in ``test_sparse_cjk_formats.py``. All text here is synthetic; every test is small (the CI jobs are time-bound).

What the format promises, and the tests below check it: a query of two or more CJK characters finds every document that
contains that substring, wherever it sits (inside a compound, with a particle, in an unspaced run, glued to Latin text); it
may also find a document where two neighbouring runs happen to form the query's windows at their border (documented, and the
property test insists that every such extra is explained that way); a one-character query is a prefix query that misses a
character which only ends a run (documented); text without CJK characters is indexed and queried exactly as in compact-1.
"""
from __future__ import annotations

import random
import re
import sqlite3
import unicodedata

import pytest

from self_directing_mcp.index import cjk
from self_directing_mcp.index import sparse as sparse_module
from self_directing_mcp.index.sparse import CJK_TERM_LIMIT, CjkSparseIndex, CompactSparseIndex, _fts_query, _fts_query_cjk, compact_support
from sparse_support import Doc, assert_same_results, build, corpus, queries
from test_sparse_korean import ASCII_QUERIES, old_fts_query

pytestmark = pytest.mark.skipif(compact_support() is not None, reason=f"this SQLite cannot use the compact format: {compact_support()}")

NFD = lambda text: unicodedata.normalize("NFD", text)           # noqa: E731
NFC = lambda text: unicodedata.normalize("NFC", text)           # noqa: E731
ZERO_WIDTH = "\u200b\u200c\u200d\u2060\ufeff"


# ---------------------------------------------------------------------------------------------------- real tokenizer access
class Tokenizer:
    """The ``porter unicode61`` tokenizer of the index, asked through an FTS5 table and its instance vocabulary."""

    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='porter unicode61')")
        self.conn.execute("CREATE VIRTUAL TABLE v USING fts5vocab(t, 'instance')")

    def tokens(self, text: str) -> list[str]:
        """The tokens of ``text`` in order of position."""
        self.conn.execute("DELETE FROM t")
        self.conn.execute("INSERT INTO t(x) VALUES (?)", (text,))
        return [row[0] for row in self.conn.execute("SELECT term FROM v ORDER BY \"offset\"")]


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer()


# ---------------------------------------------------------------------------------------------------- the pre-tokenizer
UNTOUCHED = [
    "", " ", "plain ascii text", "/home/user/proj/src/main.py:12: error", "def f(x): return x ** 2  # comment",
    "https://example.com/a?b=c&d=e", "snake_case_name kebab-case-name --no-verify", "emoji \U0001f600 and \U0001f468\u200d\U0001f469",
    "fullwidth ＡＢＣ １２ and ！？", "accented café résumé naïve", "decomposed cafe\u0301 re\u0301sume\u0301",
    "Привет мир αβγ مرحبا שלום", "tab\tnewline\nzero\u200bwidth\u200dalone",
    "middle dot ・ only, ideographic 。、「」\u3000space", "radicals ⼀⺀ strokes ㇀ hexagram ䷀", "bopomofo ㄅㄆ",
    "combining sound marks alone \u3099\u309a ゛゜", "ellipsis … and · dot",
]


@pytest.mark.parametrize("text", UNTOUCHED)
def test_text_without_a_run_character_comes_back_as_the_same_string(text):
    assert not cjk.has_run_character(text)
    assert cjk.pretokenize(text) is text                       # the very object: not even NFC runs, the text is indexed as in compact-1
    assert cjk.segments(text) in ([], [(False, text)])


def test_a_run_is_cut_into_overlapping_windows_with_outer_spaces():
    assert cjk.pretokenize("색인을") == " 색인 인을 "
    assert cjk.pretokenize("a 색인을 b") == "a  색인 인을  b"
    assert cjk.windows("색인을") == ["색인", "인을"]
    assert cjk.windows("집") == ["집"] and cjk.windows("가나") == ["가나"]
    assert cjk.rewrite_run("집") == " 집 "                   # a one-character run is kept as a one-character token
    # Nothing but the runs changes: the words around them keep every character.
    assert cjk.pretokenize("path/to/한글.py:12 x") == "path/to/ 한글 .py:12 x"


def test_outer_spaces_separate_a_run_from_the_latin_text_glued_to_it(tokenizer):
    """The trap: without the outer spaces ``v2색인을`` stays one token and a query for ``색인`` never matches."""
    assert tokenizer.tokens("v2색인을") == ["v2색인을"]                      # what unicode61 does by itself
    assert cjk.pretokenize("v2색인을") == "v2 색인 인을 "
    assert tokenizer.tokens(cjk.pretokenize("v2색인을")) == ["v2", "색인", "인을"]
    for glued in ("API를", "한글abc", "x한글y", "3개월", "ABCＡＢ가나", "café한글", "한글_글한"):
        for token in tokenizer.tokens(cjk.pretokenize(glued)):
            kinds = {cjk.has_run_character(c) for c in token}
            assert len(kinds) == 1, (glued, token)                    # no token mixes a run character with anything else
    assert tokenizer.tokens(cjk.pretokenize("API를 호출")) == ["api", "를", "호출"]       # a one-character run is a token of its own


def test_a_one_character_run_is_kept_as_a_one_character_token(tokenizer):
    assert tokenizer.tokens(cjk.pretokenize("a가b")) == ["a", "가", "b"]
    assert tokenizer.tokens(cjk.pretokenize("집 책 돈")) == ["집", "책", "돈"]
    assert tokenizer.tokens(cjk.pretokenize("本")) == ["本"]


def test_kana_and_kanji_make_one_mixed_run(tokenizer):
    text = "日本語のテキストを検索"
    assert [seg for seg in cjk.segments(text)] == [(True, text)]
    assert tokenizer.tokens(cjk.pretokenize(text)) == [text[i:i + 2] for i in range(len(text) - 1)]
    assert tokenizer.tokens(cjk.pretokenize("ひらがなーカタカナ々")) == [
        "ひら", "らが", "がな", "なー", "ーカ", "カタ", "タカ", "カナ", "ナ々"]
    assert tokenizer.tokens(cjk.pretokenize("\U00020bb7野家")) == ["\U00020bb7野", "野家"]       # supplementary-plane ideograph


def test_halfwidth_katakana_is_a_run_and_fullwidth_latin_is_not(tokenizer):
    assert cjk.pretokenize("ｶﾀｶﾅ") == " ｶﾀ ﾀｶ ｶﾅ "
    assert tokenizer.tokens(cjk.pretokenize("ﾃﾞｽﾄﾞ")) == ["ﾃﾞ", "ﾞｽ", "ｽﾄ", "ﾄﾞ"]
    assert cjk.segments("\uff21\uff22\uff23 \uff11\uff12") == [(False, "\uff21\uff22\uff23 \uff11\uff12")]
    assert cjk.pretokenize("ＡＢ가나") == "ＡＢ 가나 "              # only NFC: fullwidth Latin is not folded (not NFKC)


def test_characters_that_end_a_run(tokenizer):
    """The middle dot, ideographic punctuation, fullwidth ASCII, radicals and strokes are not run characters."""
    runs = lambda text: [piece for is_run, piece in cjk.segments(text) if is_run]                  # noqa: E731
    assert runs("카타카나・테스트") == ["카타카나", "테스트"]
    assert runs("こんにちは。世界") == ["こんにちは", "世界"]
    assert runs("一、二，三") == ["一", "二", "三"]
    assert runs("ａｂ가나！") == ["가나"]
    assert runs("一⼀二") == ["一", "二"] and runs("一㇀二") == ["一", "二"]
    assert runs("カ゠カ") == ["カ", "カ"]                                       # the double hyphen
    for cp in list(range(0xFF01, 0xFF5F)) + [0x30FB, 0x30A0] + list(range(0x3000, 0x3005)) + list(range(0x3006, 0x3041)) + list(range(0x2E80, 0x2FE0)) + list(range(0x31C0, 0x31F0)):
        assert not cjk.has_run_character(chr(cp)), hex(cp)
    # The outer spaces keep the neighbours of a run apart, whatever they are.
    assert tokenizer.tokens(cjk.pretokenize("카타카나・테스트")) == ["카타", "타카", "카나", "테스", "스트"]


def test_nfd_text_is_the_nfc_text():
    for text in ("색인을 다시", "v2색인을", "한글 café", "가각간 내용", "がぎぐ パピプ", "한글 ㄱㄴ"):
        assert cjk.has_run_character(NFD(text)), text
        assert cjk.pretokenize(NFD(text)) == cjk.pretokenize(text), text
    # Jamo that do not combine stay jamo: they are run characters too.
    assert cjk.pretokenize("\u1100\u1100\u1100") == " \u1100\u1100 \u1100\u1100 "


def test_zero_width_characters_inside_a_run_do_not_cut_it(tokenizer):
    for zw in ZERO_WIDTH:
        assert cjk.pretokenize(f"한{zw}국{zw}{zw}어") == " 한국 국어 ", hex(ord(zw))
    assert tokenizer.tokens("한\u200b국어") == ["한", "국어"]                    # unicode61 cuts at it: this is why it is removed
    assert cjk.segments("한\u200d국") == [(True, "한국")]
    # At the border of a run they stay where they are (nothing joins across them): they are separators for the tokenizer.
    assert cjk.pretokenize("한\u200b a") == " 한 \u200b a" and cjk.pretokenize("a\u200b한") == "a\u200b 한 "
    assert cjk.pretokenize("가\u200b 나") == " 가 \u200b  나 "                         # a space after it: two runs
    assert cjk.pretokenize("一\ufeff二") == " 一二 "


def test_every_character_of_a_run_is_a_unicode61_token_character(tokenizer):
    """A window must never be cut by the tokenizer itself: every code point of the run class is, for the running SQLite, a token
    character that the tokenizer keeps unchanged (no folding, no separator). Checked one by one."""
    run = re.compile(f"[{cjk.RUN_CLASS}]")
    members = [cp for cp in range(0x110000) if run.match(chr(cp))]
    assert 100_000 < len(members) < 130_000
    # Assigned characters are letters or numbers; the class also covers the unassigned code points of its blocks.
    assert {unicodedata.category(chr(cp)) for cp in members} <= {"Lo", "Lm", "Nl", "Cn"}
    bad = []
    for start in range(0, len(members), 4000):
        chunk = members[start:start + 4000]
        seen = set(tokenizer.tokens(" ".join(f"a{chr(cp)}b" for cp in chunk)))
        bad += [hex(cp) for cp in chunk if f"a{chr(cp)}b" not in seen]
    assert not bad, bad[:20]


def test_windows_survive_the_tokenizer_unchanged(tokenizer):
    rng = random.Random(8)
    pools = ["가힣", "一鿿", "ぁゖ", "ァヺ", "\u1100\u11ff", "ㄱㆎ", "ｦﾟ", "\ud7b0\ud7c6", "㐀䶿", "\U00020000\U0002a6df"]
    for _ in range(40):
        run = "".join(chr(rng.randint(ord(pool[0]), ord(pool[-1]))) for pool in rng.choices(pools, k=rng.randrange(2, 30)))
        run = NFC(run)
        if not all(cjk.has_run_character(c) for c in run):
            continue                                                                            # NFC moved a character out of the class
        assert tokenizer.tokens(cjk.pretokenize(run)) == cjk.windows(NFC(run)), run


# ---------------------------------------------------------------------------------------------------- the query builder
def test_examples_of_the_match_string():
    assert _fts_query_cjk("색인을") == '"색인 인을"'
    assert _fts_query_cjk("v2색인") == '"v2" OR "색인"'
    assert _fts_query_cjk("집") == '"집"*'
    assert _fts_query_cjk("회의") == '"회의"'
    assert _fts_query_cjk("원탁회의를") == '"원탁 탁회 회의 의를"'
    assert _fts_query_cjk("검색 index 원탁회의") == '"검색" OR "index" OR "원탁 탁회 회의"'
    assert _fts_query_cjk("日本語のテキスト") == '"日本 本語 語の のテ テキ キス スト"'
    assert _fts_query_cjk("색인 색인") == '"색인"'                  # the same run twice is one term
    assert _fts_query_cjk("한글abc나다") == '"한글" OR "abc" OR "나다"'
    assert _fts_query_cjk("") == '""' and _fts_query_cjk("a 。") == '""'


def test_nfd_and_zero_width_in_a_query_are_ignored():
    assert _fts_query_cjk(NFD("색인을")) == _fts_query_cjk("색인을")
    assert _fts_query_cjk("한\u200b국\u2060어") == _fts_query_cjk("한국어") == '"한국 국어"'
    assert _fts_query_cjk("\u200b집\ufeff") == '"집"*'


def test_a_long_run_keeps_its_first_32_windows():
    run33 = "".join(chr(0xAC00 + 3 * i) for i in range(33))
    run40 = run33 + "".join(chr(0xAC00 + 3 * i) for i in range(33, 40))
    window_count = lambda match: len(match.strip('"').split(" "))                                   # noqa: E731
    assert window_count(_fts_query_cjk(run33)) == 32 and window_count(_fts_query_cjk(run40)) == 32
    assert _fts_query_cjk(run33) == _fts_query_cjk(run40)                                         # the first 33 characters are what is queried
    assert window_count(_fts_query_cjk(run33[:32])) == 31 and cjk.MAX_QUERY_WINDOWS == 32


def test_cjk_terms_are_capped_at_24_and_keep_query_order():
    runs = [chr(0xAC00 + 11 * i) + chr(0xAC00 + 11 * i + 1) for i in range(30)]
    out = _fts_query_cjk(" ".join(runs)).split(" OR ")
    assert out == [f'"{run}"' for run in runs[:CJK_TERM_LIMIT]] and CJK_TERM_LIMIT == 24
    mixed = _fts_query_cjk(" ".join(f"term{i} {run}" for i, run in enumerate(runs))).split(" OR ")
    assert len(mixed) == 30 + 24                                                                    # ASCII terms are not capped
    assert [t for t in mixed if not t.startswith('"term')] == [f'"{run}"' for run in runs[:CJK_TERM_LIMIT]]
    twice = _fts_query_cjk(" ".join(runs[:24] + runs[:6] + runs[24:])).split(" OR ")             # repeated runs cost nothing
    assert twice == [f'"{run}"' for run in runs[:24]]
    one_character = _fts_query_cjk(" ".join(chr(0xAC00 + 11 * i) for i in range(30))).split(" OR ")
    assert len(one_character) == 24 and all(t.endswith('"*') for t in one_character)


def test_hyphens_that_only_touch_a_run_are_dropped_and_the_rest_is_as_before():
    assert _fts_query_cjk("pytest-색인") == '"pytest" OR "색인"'
    assert _fts_query_cjk("색인_v1") == '"색인" OR "v1"'
    assert _fts_query_cjk("foo-bar 색인 --no-verify") == '"foo-bar" OR "색인" OR "--no-verify"'
    assert _fts_query_cjk("한--글") == '"한"* OR "글"*'
    assert _fts_query_cjk("café café 색인") == '"café" OR "색인"'     # non-ASCII words once, as in the old builder
    assert _fts_query_cjk("cafe cafe 색인") == '"cafe" OR "cafe" OR "색인"'


def test_a_query_without_any_run_character_is_the_old_builders_string():
    for query in ASCII_QUERIES:
        assert _fts_query_cjk(query) == _fts_query(query) == old_fts_query(query), query
    rng = random.Random(5)
    alphabet = list("abcXYZ019_-  \t\n\"'()*^:.,;!@#$%&+=/\\|<>?[]{}~`") + ["OR", "AND", "NOT", "NEAR", "ab", "x", "--", "__"]
    for _ in range(1500):
        query = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 40)))
        assert _fts_query_cjk(query) == old_fts_query(query), repr(query)
    for query in ("café Zürich", "привет Мир", "αβγ", "é", "ＡＢＣ", "e\u0301a", "ㄅㄆ ㄇ", "x \U0001f600 y"):
        assert _fts_query_cjk(query) == _fts_query(query), query


def test_latin_terms_of_a_mixed_query_are_the_old_terms():
    rng = random.Random(6)
    latin = ["index", "Foo-Bar", "snake_case", "x", "v2", "a1", "--flag", "ID42", "pre-commit"]
    cjk_words = ["검색", "색인을", "집", "테스트가", "日本語", "カタカナ"]
    for _ in range(300):
        picks = [(rng.choice(latin), True) if rng.random() < 0.5 else (rng.choice(cjk_words), False) for _ in range(rng.randrange(1, 9))]
        mixed = _fts_query_cjk(" ".join(text for text, _ in picks))
        ascii_terms = [term for term in mixed.split(" OR ") if term.isascii() and term != '""' and not term.endswith("*")]
        old = old_fts_query(" ".join(text for text, is_latin in picks if is_latin))
        assert ascii_terms == ([] if old == '""' else old.split(" OR ")), picks


GRAMMAR = re.compile(r'(?:"[^"*^:()]+"\*? OR )*"[^"*^:()]+"\*?|""')


def test_operator_characters_never_reach_the_match_string_and_never_raise():
    rng = random.Random(42)
    pieces = list("abcXYZ019_- \t\n\"'()*^:.,;!@#$%&+=/\\|<>?[]{}~`") + list("검색인을집에서은는ㄹ日本語テストﾃｽ") + [
        "é", "e\u0301", "\u1100\u1161", "\u200b", "\u0301", "\U0001f600", "\u3099", "・", "。", "ก\u0e47", "NEAR", " OR ", " AND ", " NOT ",
        "\u00a0", "＂", "＊", "NEAR(검색 색인, 2)", '"검색" OR', "색인*", "^집", "col:색인"]
    queries_ = ['"검색" OR (색인*) NEAR(a b) ^x col:y', "검색*", '색인"', "{col}: 검색", "NEAR(검색 색인, 2)", '"', "*", "^", ":", "()"]
    queries_ += ["".join(rng.choice(pieces) for _ in range(rng.randrange(1, 30))) for _ in range(300)]
    con = sqlite3.connect(":memory:")
    con.execute("CREATE VIRTUAL TABLE t USING fts5(text, tokenize='porter unicode61')")
    con.executemany("INSERT INTO t(text) VALUES (?)", [(cjk.pretokenize("검색 색인을 다시"),), ("index café 日本語",), ("a b c",)])
    for query in queries_:
        match = _fts_query_cjk(query)
        assert GRAMMAR.fullmatch(match), (query, match)
        for term in match.split(" OR "):
            inner = term.strip('*').strip('"')
            assert all(c.isalnum() or c in " _-" for c in inner), (query, term)                   # letters, digits, window separators
        con.execute("SELECT rowid FROM t WHERE t MATCH ?", (match,)).fetchall()                  # a malformed MATCH raises OperationalError
    con.close()


def test_the_run_loop_checks_the_request(monkeypatch):
    calls = []
    monkeypatch.setattr(sparse_module, "check_request", lambda: calls.append(1))
    _fts_query_cjk("alpha 검색 beta 색인")
    assert len(calls) == 4                                                      # one check per piece


# ---------------------------------------------------------------------------------------------------- real FTS5 round trips
COMPOUNDS = {
    "stem": "검색 색인을 다시 만들었다",
    "subject": "색인이 깨져서 검색 실패했다",
    "compound": "원탁회의를 열었다",
    "meeting": "회의 시간은 내일이다",
    "glued": "v2색인 재작성 완료",
    "latin": "index pytest error words only",
    "home": "내 집에 갔다",
    "home2": "집값이 올랐다",
    "tailhome": "이사집",                                         # the character only ends a run: the documented miss
    "alone": "대문 앞 집 앞",                               # a one-character run
    "chinese": "我喜欢自然语言处理和搜索",
    "japanese": "日本語のテキストを検索する",
    "kana": "ひらがなとカタカナのテスト",
    "mixed": "index 색인 and 日本語 together",
}


@pytest.fixture
def compounds(tmp_path):
    docs = [Doc(f"codex:s{n % 2}:{n}:0:{n:016x}", f"s{n % 2}", "turn", text) for n, text in enumerate(COMPOUNDS.values())]
    index = build(tmp_path / "v3.sqlite", docs, CjkSparseIndex)
    by_key = dict(zip(COMPOUNDS, (d.chunk_id for d in docs)))
    yield index, docs, by_key
    index._conn.close()


def ids(index, query, top_k=50, **kwargs):
    return {hit[0] for hit in index.search(query, top_k, **kwargs)}


def containing(docs, needle):
    return {d.chunk_id for d in docs if needle in NFC(d.text)}


@pytest.mark.parametrize("query,keys", [
    ("검색", ("stem", "subject")),                           # a two-character word
    ("색인을", ("stem",)),                                   # a word with its particle
    ("색인", ("stem", "subject", "glued", "mixed")),             # also inside v2색인
    ("회의", ("compound", "meeting")),                           # the inner piece of a compound: the thing the prefix fix cannot find
    ("탁회", ("compound",)), ("원탁회의", ("compound",)), ("원탁회의를", ("compound",)),
    ("자연언어", ()),                                         # a Hangul query does not match Chinese text
    ("自然语言", ("chinese",)), ("语言处理", ("chinese",)), ("处理和搜索", ("chinese",)),
    ("日本語", ("japanese", "mixed")), ("テキスト", ("japanese",)), ("検索する", ("japanese",)),
    ("語のテ", ("japanese",)),
    ("ひらがな", ("kana",)), ("カタカナ", ("kana",)), ("テスト", ("kana",)),
    ("집값", ("home2",)), ("집에", ("home",)),
    ("없는말", ()),
])
def test_a_substring_of_two_or_more_characters_is_found_wherever_it_sits(compounds, query, keys):
    index, docs, by_key = compounds
    assert ids(index, query) == {by_key[k] for k in keys}, query
    assert ids(index, query) == containing(docs, query)                       # and that is exactly the documents that contain it


def test_the_issue_40_prefix_fix_could_not_find_these(compounds, tmp_path):
    """The same corpus in the compact format with the issue-#40 prefix terms: the inner word of a compound, the word in an
    unspaced run and the word glued to Latin text are not found (a prefix reaches only the start of a stored word)."""
    _index, docs, by_key = compounds
    compact = build(tmp_path / "v2.sqlite", docs, CompactSparseIndex)
    try:
        assert ids(compact, "회의") == {by_key["meeting"]}                    # not the compound
        assert ids(compact, "색인") == {by_key[k] for k in ("stem", "subject", "mixed")}      # not v2색인
        assert ids(compact, "自然语言") == set() and ids(compact, "语言") == set()    # Chinese runs: only from the start
        assert ids(compact, "テキスト") == set()
    finally:
        compact._conn.close()


def test_a_one_character_query_is_a_prefix_query_with_a_documented_miss(compounds):
    index, docs, by_key = compounds
    assert _fts_query_cjk("집") == '"집"*'
    found = ids(index, "집")
    # Every window that starts with the character, and every one-character run ...
    assert found == {by_key["home"], by_key["home2"], by_key["alone"]}
    # ... but not a document where the character only ends its run (이사집 holds the windows 이사, 사집): the price of windows.
    assert by_key["tailhome"] in containing(docs, "집") and by_key["tailhome"] not in found
    # Searching it with its neighbour finds it.
    assert by_key["tailhome"] in ids(index, "사집")


def test_latin_queries_are_unchanged_and_a_mixed_query_ors_its_terms(compounds):
    index, docs, by_key = compounds
    assert ids(index, "index") == {by_key["latin"], by_key["mixed"]}
    assert ids(index, "pytest error") == {by_key["latin"]}
    assert ids(index, "words") == {by_key["latin"]}
    both = index.search("index 색인", 50)
    assert {h[0] for h in both} == ids(index, "index") | ids(index, "색인")
    assert both[0][0] == by_key["mixed"]                                       # the document with both terms ranks first
    assert ids(index, "v2색인") == {by_key["glued"]} | ids(index, "색인")


def test_filters_and_top_k_work_on_phrase_queries(compounds):
    index, docs, by_key = compounds
    everyone = index.search("회의", 50)
    assert [h[2] for h in everyone] == [1, 2]
    assert index.search("회의", 1) == everyone[:1]
    session = index.search("색인", 50, session_id="s0")
    assert {h[0] for h in session} == {d.chunk_id for d in docs if d.session_id == "s0"} & ids(index, "색인")
    allowed = {by_key["compound"]}
    assert ids(index, "회의", allowed_ids=allowed) == allowed and index.search("회의", 5, allowed_ids=set()) == []


def test_a_secret_is_masked_before_the_text_is_cut_into_windows(tmp_path):
    secret = "sk-" + "Zq9" * 12
    index = CjkSparseIndex(tmp_path / "v3.sqlite")
    try:
        index.upsert("codex:s:1:0:1", "s", "turn", f"토큰 {secret} 저장 완료")
        assert index.search("Zq9Zq9") == [] and ids(index, "저장") == {"codex:s:1:0:1"}
        assert index._conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE text IS NOT NULL").fetchone()[0] == 0        # no text is stored
    finally:
        index._conn.close()


# ---------------------------------------------------------------------------------------------------- the property test
HANGUL = list("회의색인을검집원탁학교버스톱")        # 스 and 톱 appear in 버스 and 스톱: coincidences happen
HAN = list("我喜欢自然语言日本語")
KANA = list("ひらがなカタカナー")
HALFWIDTH = list("ｶﾀﾞ")
RUN_ALPHABET = HANGUL + HAN + KANA + HALFWIDTH
BETWEEN = [" ", " ", ", ", ". ", "\n", "・", "-", "/", "v2", "ABC", "x ", "_", "\U0001f600", "ＡＢ", "。", "(", "\u2060 ", " \u200b"]


def make_property_corpus(seed=11, n=120):
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        parts = []
        for _ in range(rng.randrange(1, 7)):
            run = "".join(rng.choice(rng.choice((HANGUL, HANGUL, HAN, KANA, RUN_ALPHABET))) for _ in range(rng.randrange(1, 8)))
            if rng.random() < 0.25 and len(run) > 1:                       # a zero-width character inside the run
                cut = rng.randrange(1, len(run))
                run = run[:cut] + rng.choice(ZERO_WIDTH) + run[cut:]
            parts += [run, rng.choice(BETWEEN)]
        text = "".join(parts)
        if rng.random() < 0.3:
            text = NFD(text)                                              # decomposed Hangul and kana
        docs.append(Doc(f"codex:s{i % 3}:{i}:0:{i:016x}", f"s{i % 3}", "turn", text))
    # Hand-made documents of every kind the explanation of extras must know about.
    docs += [Doc(f"codex:s0:{n}:0:{n:016x}", "s0", "turn", text) for n, text in enumerate(
        ["버스 스톱 정류장", "버스, 스톱", "버스학교", "학교버스스톱", "我喜 喜欢"], start=1000)]
    return docs


REFERENCE_RUN = re.compile("[" + "".join(sorted({NFC(c) for c in RUN_ALPHABET})) + "]" + "(?:[" + ZERO_WIDTH + "]*[" + "".join(sorted({NFC(c) for c in RUN_ALPHABET})) + "])*")


def reference_tokens(text):
    """The tokens FTS5 must hold for ``text``, with the run each came from: an independent restatement of the format over the
    alphabet the corpus uses (a window per pair of neighbouring characters of a run, the character alone for a run of one)."""
    out = []
    for label, match in enumerate(REFERENCE_RUN.finditer(NFC(text))):
        run = re.sub(f"[{ZERO_WIDTH}]", "", match.group(0))
        out += [(run[i:i + 2], label) for i in range(len(run) - 1)] if len(run) > 1 else [(run, label)]
    return out


def straddles(text, query):
    """Does the phrase of the query's windows occur in the tokens of ``text`` across two or more runs?"""
    windows = [query[i:i + 2] for i in range(len(query) - 1)]
    tokens = reference_tokens(text)
    for start in range(len(tokens) - len(windows) + 1):
        window = tokens[start:start + len(windows)]
        if [w for w, _ in window] == windows and len({label for _, label in window}) > 1:
            return True
    return False


def squeeze(text):
    return re.sub(f"[{ZERO_WIDTH}]", "", NFC(text))


def test_every_document_with_the_substring_is_found_and_every_extra_is_a_run_boundary(tmp_path):
    docs = make_property_corpus()
    index = build(tmp_path / "v3.sqlite", docs, CjkSparseIndex, size=40)
    try:
        rng = random.Random(3)
        wanted = set()
        for doc in docs:                                                    # substrings of the documents' own runs
            for match in REFERENCE_RUN.finditer(NFC(doc.text)):
                run = re.sub(f"[{ZERO_WIDTH}]", "", match.group(0))
                if len(run) >= 2:
                    size = rng.randrange(2, min(len(run), 6) + 1)
                    start = rng.randrange(0, len(run) - size + 1)
                    wanted.add(run[start:start + size])
        wanted |= {"버스톱", "버스", "스톱", "버스스톱", "我喜欢", "喜欢"}
        wanted |= {"".join(rng.choice(RUN_ALPHABET) for _ in range(rng.randrange(2, 6))) for _ in range(60)}
        wanted = sorted(w for w in wanted if len(w) >= 2 and all(NFC(c) == c for c in w))
        assert len(wanted) > 150
        extras_seen = missing = 0
        for query in wanted:
            expected = {d.chunk_id for d in docs if query in squeeze(d.text)}
            hits = ids(index, query, top_k=1000)
            assert expected <= hits, ("false negative", query, sorted(expected - hits))
            for extra in hits - expected:
                text = next(d.text for d in docs if d.chunk_id == extra)
                assert straddles(text, query), ("unexplained extra", query, text)
                extras_seen += 1
            missing += not expected
        assert extras_seen >= 3, "the corpus never produced a run-boundary coincidence, so the explanation was not exercised"
        assert missing >= 5, "every query had a document: the no-match side was not exercised"
    finally:
        index._conn.close()


def test_the_documented_false_positive_is_exactly_the_run_boundary_one(tmp_path):
    docs = [Doc("c:1", "s", "turn", "버스 스톱"), Doc("c:2", "s", "turn", "버스톱"), Doc("c:3", "s", "turn", "버스"),
            Doc("c:4", "s", "turn", "스톱 버스")]
    index = build(tmp_path / "v3.sqlite", docs, CjkSparseIndex)
    try:
        assert ids(index, "버스톱") == {"c:1", "c:2"}                       # c:1 never holds the substring: its runs border on 스
        assert "버스톱" not in docs[0].text and straddles(docs[0].text, "버스톱")
        assert not straddles(docs[3].text, "버스톱") and "c:4" not in ids(index, "버스톱")      # the other order is not a match
    finally:
        index._conn.close()


def test_on_an_ascii_corpus_the_cjk_format_returns_the_compact_results_exactly(tmp_path):
    docs = [d for d in corpus(300) if d.text.isascii()]
    assert len(docs) > 200
    compact = build(tmp_path / "v2.sqlite", docs, CompactSparseIndex)
    cjk_index = build(tmp_path / "v3.sqlite", docs, CjkSparseIndex)
    try:
        query_set = queries(docs)
        assert assert_same_results(compact, cjk_index, query_set, "ascii") >= 40            # ids, ranks and scores
        assert all(_fts_query_cjk(q["q"]) == _fts_query(q["q"]) for q in query_set if q["q"].isascii())
        assert cjk_index.verify_statistics()["tokens"] == compact.verify_statistics()["tokens"]
    finally:
        compact._conn.close()
        cjk_index._conn.close()


# ---------------------------------------------------------------------------------------------------- statistics exactness
def scores(index, query_set):
    return [{h[0]: h[1] for h in index.search(q, 1000)} for q in query_set]


def test_statistics_stay_exact_after_deletes_and_reupserts(tmp_path):
    docs = make_property_corpus(seed=21, n=80)
    index = build(tmp_path / "v3.sqlite", docs, CjkSparseIndex, size=25)
    try:
        gone = {d.chunk_id for d in docs[::5]}
        index.delete_ids(sorted(gone))
        changed = [d._replace(text=d.text + " 색인 색인을 rewritten") for d in docs[1:30:4] if d.chunk_id not in gone]
        index.upsert_many(changed)
        index.upsert("codex:s9:new:0:0", "s9", "meta", "새로운 회의록 plus ascii words")
        index.delete_ids(["codex:s9:new:0:0", "codex:absent"])
        survivors = {d.chunk_id: d for d in docs if d.chunk_id not in gone}
        survivors.update({d.chunk_id: d for d in changed})
        assert index.verify_statistics()["ok"] and index.count() == len(survivors)
        fresh = build(tmp_path / "fresh.sqlite", list(survivors.values()), CjkSparseIndex, size=25)
        try:
            query_set = ["색인", "회의", "학교", "我喜欢", "가나", "버스", "색인을 회의", "ABC", "v2 색인", "rewritten"]
            for query, got, want in zip(query_set, scores(index, query_set), scores(fresh, query_set)):
                assert got.keys() == want.keys(), query
                for cid in got:
                    assert abs(got[cid] - want[cid]) <= 1e-12 * max(1.0, abs(want[cid])), (query, cid)
            a, b = index.verify_statistics(), fresh.verify_statistics()
            assert a["ok"] and b["ok"] and (a["documents"], a["tokens"]) == (b["documents"], b["tokens"])
            assert sum(map(bool, scores(index, query_set))) >= 6
        finally:
            fresh._conn.close()
    finally:
        index._conn.close()


def test_the_length_a_document_retires_is_the_count_of_its_windows(tmp_path):
    """``_retire_statistics`` takes the recorded document size out of the totals: the token count of the PRE-TOKENIZED text."""
    index = CjkSparseIndex(tmp_path / "v3.sqlite")
    try:
        index.upsert("c:1", "s", "turn", "색인을 다시 v2색인")                 # windows: 색인 인을 | 다시 | v2 색인
        assert index.verify_statistics() == {"ok": True, "documents": 1, "tokens": 5, "recorded": [1, 5], "mapped": 1}
        index.upsert("c:2", "s", "turn", "plain words")
        index.delete_ids(["c:1"])
        assert index.verify_statistics() == {"ok": True, "documents": 1, "tokens": 2, "recorded": [1, 2], "mapped": 1}
        index.clear()
        assert index.verify_statistics()["recorded"] == [0, 0]
    finally:
        index._conn.close()


# ---------------------------------------------------------------------------------------------------- size and time guards
def korean_like_corpus(seed=3, stems=150, docs=400, words=60):
    rng = random.Random(seed)
    syllables = [chr(c) for c in range(0xAC00, 0xAC00 + 11172, 37)]
    suffix = ["", "", "", "을", "를", "은", "는", "이", "가", "의", "에", "에서", "으로", "했다", "합니다"]
    vocabulary = ["".join(rng.choice(syllables) for _ in range(rng.choice((2, 2, 3, 3, 4)))) for _ in range(stems)]
    weights = [1 / (i + 1) for i in range(stems)]
    return [" ".join(s + rng.choice(suffix) for s in rng.choices(vocabulary, weights=weights, k=words)) for _ in range(docs)]


def test_the_file_is_not_much_bigger_than_the_text_and_bigger_than_the_compact_one(tmp_path):
    """A size guard (measured at 100,000 synthetic documents: 0.77 of the text bytes; a small file carries more overhead), and
    the honest cost: for CJK-heavy text the CJK format is bigger than the compact one."""
    texts = korean_like_corpus()
    docs = [Doc(f"codex:s{i % 4}:{i}:0:{i:016x}", f"s{i % 4}", "turn", text) for i, text in enumerate(texts)]
    text_bytes = sum(len(t.encode()) for t in texts)
    sizes = {}
    for name, cls in (("v2", CompactSparseIndex), ("v3", CjkSparseIndex)):
        index = build(tmp_path / f"{name}.sqlite", docs, cls, size=100)
        index._conn.commit()
        index._conn.close()
        sizes[name] = (tmp_path / f"{name}.sqlite").stat().st_size
    assert sizes["v3"] < 1.2 * text_bytes, (sizes, text_bytes)
    assert sizes["v2"] < sizes["v3"], sizes


def test_pretokenizing_costs_about_a_regex_pass():
    """No flaky wall-clock bound: a ratio to a plain regex substitution over the same documents (about 9 measured), best of
    three, with a generous limit."""
    import time
    texts = korean_like_corpus(docs=150)
    plain = re.compile("[가-힣]+")

    def best(function):
        times = []
        for _ in range(3):
            started = time.perf_counter()
            for text in texts:
                function(text)
            times.append(time.perf_counter() - started)
        return min(times)

    baseline = best(lambda text: plain.sub("", text))
    assert best(cjk.pretokenize) < 60 * max(baseline, 1e-4)
