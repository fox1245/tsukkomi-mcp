"""CJK pre-tokenization of the ``compact-cjk-1`` sparse format (``sparse-v3.sqlite``, tokenization ``cjk-bigram-1``).

SQLite's ``unicode61`` tokenizer cuts a text at spaces and punctuation only, so a Korean, Chinese or Japanese run is one
token: ``원탁회의를`` is stored whole and a search for ``회의`` cannot find it, and neither can ``색인`` find ``v2색인``.
This module rewrites every run of CJK-script characters into overlapping two-character windows *before* the text reaches
FTS5, on the write side and, with the very same run definition, on the query side::

    색인을 다시        ->   " 색인 인을 " + " " + " 다시 "     (tokens 색인, 인을, 다시)
    v2색인을           ->   "v2" + " 색인 인을 "              (tokens v2, 색인, 인을)

A query for a run is the FTS5 phrase of its windows, so any substring of two or more characters is found wherever it
sits. Standard library only (``re`` and ``unicodedata``): hook processes import the index package, so nothing heavy may
be imported here.

The definitions below are part of the file format. Changing any of them changes what an existing ``sparse-v3.sqlite``
holds, so it needs a new ``TOKENIZATION`` value (the index refuses a file whose tokenization it does not know instead of
answering from it).
"""
from __future__ import annotations

import re
import unicodedata

TOKENIZATION = "cjk-bigram-1"           # sparse_meta.tokenization of a sparse-v3.sqlite written with these definitions
MAX_QUERY_WINDOWS = 32                  # windows of one run that a query keeps (a run of up to 33 characters is matched whole)

# What makes a "run character": Hangul (syllables, jamo, compatibility jamo, jamo extended A and B), CJK ideographs (extension
# A, the unified block, compatibility, the supplementary planes up to extension H) and the ideographic iteration mark,
# Hiragana, Katakana (with the phonetic extensions) and halfwidth Katakana. Deliberately not in a run, so that a run ends
# there: fullwidth ASCII (FF01-FF5E; the unicode61 tokenizer keeps it with the Latin letters around it), the Katakana middle
# dot U+30FB and the double hyphen U+30A0, ideographic punctuation, radicals and strokes, the combining sound marks U+3099 and
# U+309A (NFC composes them into the kana whenever a composed kana exists) and every other script.
# Every character of this class is a token character of unicode61, never a separator, so a window is never cut by the
# tokenizer itself; tests/test_sparse_cjk.py proves that against the running SQLite for every code point of the class.
RUN_CLASS = (
    "\u1100-\u11ff"                     # Hangul Jamo
    "\u3005"                            # ideographic iteration mark
    "\u3041-\u3096\u309d-\u309f"        # Hiragana
    "\u30a1-\u30fa\u30fc-\u30ff\u31f0-\u31ff"   # Katakana and its phonetic extensions
    "\u3131-\u318e"                     # Hangul compatibility jamo
    "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af"       # CJK ideographs
    "\ua960-\ua97f\ud7b0-\ud7ff"        # Hangul jamo extended A and B
    "\uac00-\ud7a3"                     # Hangul syllables
    "\uff66-\uff9f"                     # halfwidth Katakana
)
# Zero-width characters between two run characters are dropped, so that they do not cut a run (unicode61 treats them as
# separators). Variation selectors are not in this list: an ideographic variation sequence cuts the run.
ZERO_WIDTH = "\u200b\u200c\u200d\u2060\ufeff"

_ANY_RUN_CHARACTER = re.compile(f"[{RUN_CLASS}]")
_RUN = re.compile(f"[{RUN_CLASS}](?:[{ZERO_WIDTH}]*[{RUN_CLASS}])*")
_ZERO_WIDTH = re.compile(f"[{ZERO_WIDTH}]+")


def has_run_character(text: str) -> bool:
    return _ANY_RUN_CHARACTER.search(text) is not None


def windows(run: str) -> list[str]:
    """The overlapping two-character windows of a run (``색인을`` gives ``색인``, ``인을``); a one-character run is its own
    only token."""
    return [run] if len(run) < 2 else [run[i:i + 2] for i in range(len(run) - 1)]


def rewrite_run(run: str) -> str:
    """One run as it is stored: the windows separated by single spaces and **a space on each side**. The outer spaces are
    what lets ``v2색인`` be found by ``색인``: without them ``v2`` and the first window would be one token."""
    return " " + " ".join(windows(run)) + " "


def clean_run(matched: str) -> str:
    """A matched run without the zero-width characters inside it."""
    return _ZERO_WIDTH.sub("", matched) if len(matched) > 1 else matched


def segments(text: str) -> list[tuple[bool, str]]:
    """``text`` (already in NFC) cut into ``(is_run, piece)`` pairs in order; the pieces of runs are clean (no zero-width
    characters) and the other pieces are untouched. The writer and the query builder both start from here."""
    out: list[tuple[bool, str]] = []
    position = 0
    for match in _RUN.finditer(text):
        if match.start() > position:
            out.append((False, text[position:match.start()]))
        out.append((True, clean_run(match.group(0))))
        position = match.end()
    if position < len(text):
        out.append((False, text[position:]))
    return out


def _rewrite_match(match: re.Match) -> str:
    return rewrite_run(clean_run(match.group(0)))


def pretokenize(text: str) -> str:
    """The text as FTS5 must index it. A text without any run character comes back as the very same string (so a corpus
    without CJK text is indexed exactly as in the ``compact-1`` format); otherwise the text is put in NFC (not NFKC: that
    would fold fullwidth Latin) and every run is rewritten with ``rewrite_run``. Everything outside the runs is untouched."""
    if _ANY_RUN_CHARACTER.search(text) is None:
        return text
    return _RUN.sub(_rewrite_match, unicodedata.normalize("NFC", text))
