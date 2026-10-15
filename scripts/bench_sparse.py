#!/usr/bin/env python3
"""Benchmark of the sparse (FTS5) index layouts on a SYNTHETIC coding-agent corpus. Nothing real is read.

The corpus is a seeded, deterministic stream of session events that look like what a coding agent writes: short chat
turns (English, Korean and mixed), shell commands and patch arguments, and multi-KB tool results (logs, pytest output,
diffs, JSON, stack traces, listings, code, grep hits) full of paths, hashes, uuids and snake_case / hyphenated
identifiers, with a heavy-tailed size distribution. Two profiles: ``codex`` (mean about 5.5 KB an event) and ``omp``
(mean about 1.7 KB). The stream of N events is a prefix of the stream of 2N events, so one 100k build also gives the 20k
point (``checkpoints``) and the linearity of size and time.

Every layout is built through the write path of its class (``upsert_many``, 500-event batches, ``mask_secrets`` on every
text) and measured for file size and its breakdown (``dbstat``), build time, search parity against the current layout
(``legacy``: ``SparseIndex`` as released), search latency per query class, ``delete_ids`` plus re-``upsert`` of 1000
events, ``integrity_check``, the FTS5 ``optimize`` command and ``VACUUM``; with ``--stages ops,maint,orphans`` also the cost of deleting a
tenth of the documents in batches (what ``sparse-rebuild`` does to orphans).

    python scripts/bench_sparse.py --profile codex --events 100000 --candidate a,c,e [--out results.json]
    python scripts/bench_sparse.py --stats --profile codex --events 20000      # corpus statistics only
    python scripts/bench_sparse.py --list

The generated databases live in a fresh directory under ``--tmpdir`` (default ``/tmp/claude-1000/sp``) that is removed when
the script ends. The search results of candidate ``a`` (the reference for parity) are kept in ``<tmpdir>/ref`` so that
later runs of other candidates can compare against them; ``--drop-ref`` removes them. Run one candidate at a time: a
100k Codex-like database is about 1 GB.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))     # this checkout, whatever is installed

from self_directing_mcp.index.sparse import CompactSparseIndex, SparseIndex, _fts_query    # noqa: E402
from self_directing_mcp.security.mask import mask_secrets               # noqa: E402

BATCH = 500
CHECKPOINT_EVERY = 20_000
DEFAULT_TMP = "/tmp/claude-1000/sp"
TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+")


# --------------------------------------------------------------------------------------------------------------------
# Synthetic corpus
# --------------------------------------------------------------------------------------------------------------------
class Doc(NamedTuple):
    chunk_id: str
    session_id: str
    kind: str
    text: str


_COMMON = """the be to of and a in that have it for not on with as you do at this but by from they we say or an will my one all
would there their what so up out if about who get which go me when make can like time no just him know take people into year
your good some could them see other than then now look only come its over think also back after use two how our work first
well way even new want because any these give day most us""".split()

_DEV = """error test file function return value class import module build run commit branch merge request response server client
config path directory install package version update delete create remove check fail pass warning exception traceback line column
index query result output input data json string number list array object token session event chunk search parse format default
option flag argument command shell script process thread lock timeout retry cache memory disk network socket port host header
body status code message log debug info trace stack frame handler service worker queue job task schedule metric counter gauge
report summary review patch diff hunk change revision release tag deploy rollback migrate schema table column row field record
store load save write read open close flush sync async await future promise stream buffer chunk block page offset length size
count limit batch window slice range filter sort group join merge split trim strip match replace regex pattern rule policy
contract audit verify validate sanitize escape encode decode encrypt decrypt hash digest sign secret credential permission role
user admin owner member team project repo workspace folder document markdown yaml toml config setting environment variable
runtime compiler linker loader debugger profiler benchmark latency throughput bandwidth capacity quota budget cost price
invoice payment order customer product catalog inventory shipment warehouse supplier vendor account balance ledger entry
transaction commitment rollback savepoint isolation snapshot checkpoint journal recovery crash restart reboot shutdown
startup initialize configure register subscribe publish notify dispatch route proxy gateway balancer cluster node shard
replica leader follower election consensus heartbeat gossip membership partition rebalance compaction vacuum optimize""".split()

_PREFIX = ["", "", "", "re", "un", "pre", "sub", "multi", "auto", "co"]
_ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "cr", "dr", "fl", "gr", "pl", "st", "tr", "ch", "sh"]
_VOWELS = ["a", "e", "i", "o", "u", "ai", "ea", "ou", "io"]
_CODAS = ["", "", "n", "r", "s", "t", "l", "x", "nd", "st", "rk"]
_KO_SYL = list("가각간감강개거건게결경고공과관광구국군그근기김나남내너노농누는다단대도동두드들라라래러런려로루류르리마만말망매머모목무문물미민바반방배백버번보부북분불비사산상새서석선성세소수스시신실심아안알암애야양어언얼엄여연영예오온와왕외요용우운울원위유은을음의이인일임자작잔장재저전정제조주중지진차착참창채처천철초추출치카컴코크타탄태터테토통트파판패펜포표프피하한함해허현호화환회후히")
_KO_END = ["", "", "", "은", "는", "이", "가", "을", "를", "에", "에서", "으로", "와", "과", "도", "만", "의", "에게", "까지", "부터"]
_KO_VERB = ["했어요", "합니다", "해줘", "됩니다", "안 돼요", "확인해줘", "실패했어", "성공했습니다", "수정했어요", "추가해줘", "보여줘", "고쳐줘"]
_LEVELS = ["INFO", "INFO", "INFO", "DEBUG", "DEBUG", "WARN", "ERROR"]
_EXT = [".py", ".py", ".py", ".ts", ".rs", ".go", ".md", ".json", ".toml", ".yaml", ".sh", ".c", ".cpp", ".h"]
_FLAGS = ["--no-verify", "--max-count=5", "--dry-run", "--force-with-lease", "--no-pager", "--follow-tags", "--cached",
          "--name-only", "--show-toplevel", "--fail-fast", "--maxfail=3", "-x", "-q", "-k", "-vv", "--tb=short", "--no-cache-dir"]
_SECRET_FORMS = ["sk-{a}", "ghp_{b}", "Authorization: Bearer {c}", "api_key={c}", "AKIA{d}"]


class Lexicon:
    """The vocabulary every corpus draws from. Fixed seed: it does not depend on the corpus seed."""

    def __init__(self) -> None:
        r = random.Random(20240131)
        words: list[str] = list(_COMMON)
        for w in _DEV:
            words.append(w)
        # Inflections give the porter stemmer work (fail/failed/failing/fails fold together).
        for w in _DEV[::2]:
            base = w[:-1] if w.endswith("e") else w
            words += [w + "s", base + "ed", base + "ing"]
        for _ in range(7000):   # a long tail of rare, pseudo-word vocabulary
            n = r.randrange(2, 4)
            words.append(r.choice(_PREFIX) + "".join(r.choice(_ONSETS) + r.choice(_VOWELS) + r.choice(_CODAS) for _ in range(n)))
        self.words = words
        self.pseudo_start = len(_COMMON) + len(_DEV) + 3 * len(_DEV[::2])
        total, cum = 0.0, []
        for rank in range(1, len(words) + 1):
            total += 1.0 / rank ** 1.05
            cum.append(total)
        self.cum = cum
        self.parts = [r.choice(_PREFIX) + "".join(r.choice(_ONSETS) + r.choice(_VOWELS) for _ in range(r.randrange(2, 4)))
                      for _ in range(2500)]      # path components and module names
        self.dev_rank = {w: i for i, w in enumerate(words)}
        self.ko_stems = ["".join(r.choice(_KO_SYL) for _ in range(r.randrange(1, 4))) for _ in range(1800)]
        total, cum = 0.0, []
        for rank in range(1, len(self.ko_stems) + 1):
            total += 1.0 / rank ** 1.0
            cum.append(total)
        self.ko_cum = cum


LEX = Lexicon()


class Profile(NamedTuple):
    name: str
    provider: str
    # (kind share, lognormal median in bytes, sigma, hard cap) per kind; tuned so the mean lands near ``target_mean``.
    sizes: dict
    target_mean: int
    session_median: int      # events per session, lognormal


PROFILES = {
    "codex": Profile("codex", "codex", {
        "turn": (0.22, 260, 1.05, 30_000), "tool_call": (0.28, 170, 1.25, 60_000),
        "tool_result": (0.45, 2300, 1.85, 400_000), "meta": (0.05, 220, 0.6, 4_000)}, 5500, 700),
    "omp": Profile("omp", "omp", {
        "turn": (0.28, 330, 1.0, 20_000), "tool_call": (0.27, 180, 1.1, 30_000),
        "tool_result": (0.40, 950, 1.6, 150_000), "meta": (0.05, 200, 0.6, 3_000)}, 1700, 300),
}


class Corpus:
    """Deterministic stream of ``Doc`` batches plus the facts the query builder needs."""

    def __init__(self, profile: str, events: int, seed: int = 7) -> None:
        self.profile = PROFILES[profile]
        self.events = events
        self.seed = seed
        self.rng = random.Random(f"{seed}:{profile}")      # str seeds are hashed deterministically by random.seed
        self.all_ids: list[str] = []
        self.session_of: list[int] = []
        self.sessions: list[str] = []
        self.session_bounds: list[tuple[int, int]] = []
        self.probes: list[tuple[int, list[str]]] = []      # (event index, tokens of a sampled event)
        self.keep: list[Doc] = []                          # every 100th event, for the delete / re-upsert measurement
        self.total_chars = 0
        self.kind_count: dict[str, int] = {}

    # ---- text builders ------------------------------------------------------------------------------------------
    def _uuid(self) -> str:
        h = "%032x" % self.rng.getrandbits(128)
        return f"{h[:8]}-{h[8:12]}-4{h[13:16]}-8{h[17:20]}-{h[20:]}"

    def _path(self) -> str:
        r, parts = self.rng, LEX.parts
        depth = r.randrange(2, 5)
        return "/home/user/proj/" + "/".join(r.choice(parts) for _ in range(depth)) + r.choice(_EXT)

    def _ident(self) -> str:
        r = self.rng
        words = r.choices(LEX.words, cum_weights=LEX.cum, k=r.randrange(1, 4))
        style = r.randrange(6)
        if style < 3:
            return "_".join(words)
        if style == 3:
            return words[0] + "".join(w.capitalize() for w in words[1:])
        if style == 4:
            return "-".join(words)
        return "_".join(words).upper()

    def _sentence(self) -> str:
        r = self.rng
        n = r.randrange(5, 15)
        s = " ".join(r.choices(LEX.words, cum_weights=LEX.cum, k=n))
        if r.random() < 0.25:
            s += " " + self._ident()
        if r.random() < 0.10:
            s += " `" + self._path() + "`"
        return s[0].upper() + s[1:] + r.choice([".", ".", ".", "?", ":"])

    def _korean(self) -> str:
        r = self.rng
        out = []
        for _ in range(r.randrange(3, 9)):
            stem = r.choices(LEX.ko_stems, cum_weights=LEX.ko_cum, k=1)[0]
            out.append(stem + r.choice(_KO_END))
        if r.random() < 0.5:
            out.append(self._ident() if r.random() < 0.6 else r.choice(LEX.words))
        out.append(r.choice(_KO_VERB))
        return " ".join(out)

    def _cmd(self) -> str:
        r = self.rng
        k = r.randrange(9)
        if k == 0:
            return f"git commit -m \"{self._sentence()[:60]}\" {r.choice(_FLAGS)}"
        if k == 1:
            return f"python -m pytest -q tests/test_{r.choice(LEX.parts)}.py::test_{self._ident()} {r.choice(_FLAGS)}"
        if k == 2:
            return f"rg -n \"{self._ident()}\" src/{r.choice(LEX.parts)}"
        if k == 3:
            return f"sed -n {r.randrange(1, 500)},{r.randrange(500, 900)}p {self._path()}"
        if k == 4:
            return f"git diff {'%07x' % r.getrandbits(28)}..{'%07x' % r.getrandbits(28)} -- {self._path()}"
        if k == 5:
            return f"cargo build --release {r.choice(_FLAGS)} -p {r.choice(LEX.parts)}-{r.choice(LEX.parts)}"
        if k == 6:
            return f"pip install {r.choice(LEX.parts)}-{r.choice(LEX.parts)}=={r.randrange(0, 9)}.{r.randrange(0, 30)}.{r.randrange(0, 9)}"
        if k == 7:
            return f"ls -la {self._path().rsplit('/', 1)[0]} && cat {self._path()}"
        return f"curl -sS https://api.example.test/v1/{r.choice(LEX.parts)}/{r.randrange(100000)} -H 'X-Request-Id: {self._uuid()}'"

    def _patch(self, target: int) -> str:
        r = self.rng
        out = [f"*** Begin Patch", f"*** Update File: {self._path()}"]
        size = 40
        while size < target:
            line = r.choice(("+", "-", " ")) + self._code_line()
            out.append(line)
            size += len(line) + 1
        out.append("*** End Patch")
        return "\n".join(out)

    def _code_line(self) -> str:
        r = self.rng
        indent = "    " * r.randrange(0, 4)
        k = r.randrange(8)
        if k == 0:
            return f"{indent}def {self._ident()}(self, {self._ident()}, {self._ident()}=None):"
        if k == 1:
            return f"{indent}return {self._ident()}({self._ident()}, {r.randrange(100)})"
        if k == 2:
            return f"{indent}{self._ident()} = {self._ident()}.{self._ident()}({self._ident()})"
        if k == 3:
            return f"{indent}# {self._sentence()}"
        if k == 4:
            return f"{indent}if {self._ident()} is not None and {self._ident()} > {r.randrange(1000)}:"
        if k == 5:
            return f"{indent}raise ValueError(f\"{self._ident()} must be {r.choice(LEX.words)}: {{{self._ident()}!r}}\")"
        if k == 6:
            return f"{indent}from {r.choice(LEX.parts)}.{r.choice(LEX.parts)} import {self._ident()}"
        return f"{indent}logger.{r.choice(['info', 'debug', 'warning'])}(\"{self._sentence()[:50]}\", extra={{\"id\": \"{self._uuid()}\"}})"

    def _log_line(self) -> str:
        r = self.rng
        ts = f"2026-{r.randrange(1, 13):02d}-{r.randrange(1, 29):02d}T{r.randrange(24):02d}:{r.randrange(60):02d}:{r.randrange(60):02d}.{r.randrange(1000):03d}Z"
        return (f"{ts} {r.choice(_LEVELS)} [{r.choice(LEX.parts)}-{r.randrange(16)}] {self._sentence()[:70]} "
                f"request_id={self._uuid()} path=/api/v1/{r.choice(LEX.parts)}/{r.randrange(100000)} status={r.choice((200, 200, 200, 201, 404, 500))} took={r.randrange(1, 900)}ms")

    def _pytest_line(self) -> str:
        r = self.rng
        res = r.choice(("PASSED", "PASSED", "PASSED", "FAILED", "SKIPPED"))
        return f"tests/test_{r.choice(LEX.parts)}_{r.choice(LEX.parts)}.py::test_{self._ident()} {res} [{r.randrange(101):3d}%]"

    def _diff_block(self) -> list[str]:
        r = self.rng
        a, b = '%07x' % r.getrandbits(28), '%07x' % r.getrandbits(28)
        p = self._path().replace("/home/user/proj/", "")
        lines = [f"diff --git a/{p} b/{p}", f"index {a}..{b} 100644", f"--- a/{p}", f"+++ b/{p}"]
        for _ in range(r.randrange(1, 4)):
            lines.append(f"@@ -{r.randrange(1, 900)},{r.randrange(3, 20)} +{r.randrange(1, 900)},{r.randrange(3, 24)} @@ {self._code_line().strip()[:40]}")
            for _ in range(r.randrange(4, 14)):
                lines.append(r.choice((" ", " ", "+", "-")) + self._code_line())
        return lines

    def _json_obj(self, depth: int = 0) -> str:
        r = self.rng
        items = []
        for _ in range(r.randrange(2, 7)):
            key = self._ident() if r.random() < 0.5 else r.choice(LEX.words)
            c = r.randrange(7)
            if c == 0:
                val = f"\"{self._uuid()}\""
            elif c == 1:
                val = str(r.randrange(10 ** r.randrange(1, 8)))
            elif c == 2:
                val = f"\"{self._sentence()[:50]}\""
            elif c == 3:
                val = f"\"{'%040x' % r.getrandbits(160)}\""
            elif c == 4 and depth < 2:
                val = self._json_obj(depth + 1)
            elif c == 5:
                val = "[" + ", ".join(f"\"{self._ident()}\"" for _ in range(r.randrange(1, 5))) + "]"
            else:
                val = r.choice(("true", "false", "null"))
            items.append(f"\"{key}\": {val}")
        return "{" + ", ".join(items) + "}"

    def _trace(self) -> list[str]:
        r = self.rng
        lines = ["Traceback (most recent call last):"]
        for _ in range(r.randrange(3, 9)):
            lines.append(f"  File \"{self._path()}\", line {r.randrange(1, 900)}, in {self._ident()}")
            lines.append("    " + self._code_line().strip())
        lines.append(f"{r.choice(('ValueError', 'KeyError', 'RuntimeError', 'sqlite3.OperationalError', 'TimeoutError'))}: {self._sentence()[:80]}")
        return lines

    def _listing_line(self) -> str:
        r = self.rng
        return f"-rw-r--r-- 1 user user {r.randrange(10 ** r.randrange(2, 8)):>8} {r.choice(('Jan', 'Feb', 'Mar', 'Apr', 'May'))} {r.randrange(1, 29):2d} {r.randrange(24):02d}:{r.randrange(60):02d} {r.choice(LEX.parts)}{r.choice(_EXT)}"

    def _hash_line(self) -> str:
        r = self.rng
        return f"sha256:{'%064x' % r.getrandbits(256)}  {self._path()}"

    def _grep_line(self) -> str:
        r = self.rng
        return f"{self._path().replace('/home/user/proj/', '')}:{r.randrange(1, 900)}:{self._code_line()}"

    def _blob(self) -> str:
        r = self.rng
        return "".join(r.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", k=r.randrange(300, 900)))

    def _result(self, target: int) -> str:
        r = self.rng
        flavor = r.choices(("log", "pytest", "diff", "json", "trace", "listing", "code", "grep", "hash"),
                           weights=(20, 14, 15, 14, 8, 6, 12, 8, 3))[0]
        lines = [f"exit_code: {r.choice((0, 0, 0, 1, 2))}", f"Duration: {r.randrange(1, 90)}.{r.randrange(10)}s", "Output:"]
        size = 60
        while size < target:
            if flavor == "log":
                chunk = [self._log_line()]
            elif flavor == "pytest":
                chunk = [self._pytest_line()]
            elif flavor == "diff":
                chunk = self._diff_block()
            elif flavor == "json":
                chunk = [self._json_obj()]
            elif flavor == "trace":
                chunk = self._trace() if r.random() < 0.3 else [self._log_line()]
            elif flavor == "listing":
                chunk = [self._listing_line()]
            elif flavor == "code":
                chunk = [self._code_line()]
            elif flavor == "grep":
                chunk = [self._grep_line()]
            else:
                chunk = [self._hash_line()]
            if r.random() < 0.004:
                chunk.append(self._blob())
            lines.extend(chunk)
            size += sum(len(x) + 1 for x in chunk)
        return "\n".join(lines)

    def _secret(self) -> str:
        r = self.rng
        form = r.choice(_SECRET_FORMS)
        return form.format(a="".join(r.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=40)),
                           b="".join(r.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJ0123456789", k=36)),
                           c="".join(r.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=32)),
                           d="".join(r.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", k=16)))

    # ---- events ---------------------------------------------------------------------------------------------------
    def _target(self, kind: str) -> int:
        _share, median, sigma, cap = self.profile.sizes[kind]
        return max(20, min(cap, int(self.rng.lognormvariate(math.log(median), sigma))))

    def _text(self, kind: str, target: int) -> str:
        r = self.rng
        if kind == "turn":
            parts = []
            size = 0
            korean = r.random() < 0.18
            while size < target:
                part = self._korean() if (korean and r.random() < 0.8) else self._sentence()
                if r.random() < 0.08:
                    part += "\n```\n" + "\n".join(self._code_line() for _ in range(r.randrange(2, 8))) + "\n```"
                parts.append(part)
                size += len(part) + 1
            text = "\n".join(parts)
        elif kind == "tool_call":
            if target > 900:
                text = self._patch(target)
            elif r.random() < 0.5:
                text = f"{{\"cmd\": [\"bash\", \"-lc\", \"{self._cmd()}\"], \"workdir\": \"/home/user/proj\", \"timeout_ms\": {r.choice((10000, 30000, 120000))}}}"
            else:
                text = self._cmd()
        elif kind == "tool_result":
            text = self._result(target)
        else:
            text = f"[session_meta] cwd=/home/user/proj model={r.choice(('gpt-5', 'claude-sonnet', 'qwen3'))} sandbox={r.choice(('workspace-write', 'read-only'))} id={self._uuid()}"
        if r.random() < 0.006:
            text += "\n" + self._secret()
        return text

    def batches(self):
        r, prof = self.rng, self.profile
        kinds = list(prof.sizes)
        weights = [prof.sizes[k][0] for k in kinds]
        batch: list[Doc] = []
        index = 0
        while index < self.events:
            sid = self._uuid()
            n = max(3, int(r.lognormvariate(math.log(prof.session_median), 1.1)))
            n = min(n, self.events - index)
            self.sessions.append(sid)
            self.session_bounds.append((index, index + n))
            offset = 0
            for _ in range(n):
                kind = r.choices(kinds, weights=weights)[0]
                text = self._text(kind, self._target(kind))
                chunk_id = f"{prof.provider}:{sid}:{offset}:0:{'%016x' % r.getrandbits(64)}"
                offset += len(text) + r.randrange(60, 400)
                doc = Doc(chunk_id, sid, kind, text)
                self.all_ids.append(chunk_id)
                self.session_of.append(len(self.sessions) - 1)
                self.total_chars += len(text)
                self.kind_count[kind] = self.kind_count.get(kind, 0) + 1
                if index % 97 == 0:
                    self.probes.append((index, TOKEN_RE.findall(text[:3000])))
                if index % 100 == 0:
                    self.keep.append(doc)
                index += 1
                batch.append(doc)
                if len(batch) >= BATCH:
                    yield batch
                    batch = []
        if batch:
            yield batch

    # ---- queries --------------------------------------------------------------------------------------------------
    def queries(self, count: int = 200) -> list[dict]:
        """``count`` queries from this corpus, a fixed share per class. Call after the stream was consumed."""
        r = random.Random(f"{self.seed}:{self.profile.name}:queries")
        tokens = [t for _i, toks in self.probes for t in toks]
        ids = [t for t in tokens if len(t) >= 8 and any(c.isdigit() for c in t) and "-" not in t and "_" not in t]
        hyph = [t for t in tokens if ("-" in t.strip("-") or "_" in t) and len(t) >= 5]
        words = [t for t in tokens if t.isalpha() and len(t) >= 3]
        common = [w for w in LEX.words[:60] if len(w) >= 3]
        mid = LEX.words[300:2500]
        per = count // 10
        out: list[dict] = []

        def add(cls, query, **extra):
            out.append({"cls": cls, "q": query, "top_k": extra.pop("top_k", r.choice((5, 20, 100))), **extra})

        for _ in range(per):
            add("freq1", r.choice(common))
        for _ in range(per):
            add("mid1", r.choice(mid))
        for _ in range(per):
            add("rare1", r.choice(ids or words))
        for _ in range(per):
            probe = r.choice(self.probes)[1] or ["error"]
            pick = [t for t in probe if len(t) >= 3]
            add("multi", " ".join(r.sample(pick, min(len(pick), r.randrange(2, 5))) or ["error"]))
        for _ in range(per):
            add("hyphen", r.choice(hyph or words) if r.random() < 0.7 else " ".join(r.sample(hyph or words, 2)))
        for _ in range(per):
            ko = LEX.ko_stems[r.randrange(40)] + r.choice(_KO_END)
            add("korean", ko if r.random() < 0.5 else ko + " " + r.choice(common))     # the query builder drops Hangul
        for _ in range(per):
            add("nomatch", r.choice(("zzqxv", "qqzzyy", "nonexistent_token_77", "xylophage-zz", "zq0042", "wwwvvv-uuu")) + str(r.randrange(10)))
        for _ in range(per):
            si = r.randrange(len(self.sessions))
            add("session", r.choice(common + mid[:200]), session_id=self.sessions[si])
        for _ in range(per):
            si = r.randrange(len(self.sessions))
            a, b = self.session_bounds[si]
            sub = {self.all_ids[i] for i in range(a, b)}
            add("allowed_session", r.choice(common + mid[:200]), session_id=self.sessions[si], allowed=sorted(sub))
        while len(out) < count:
            sample = {i for i in r.sample(range(len(self.all_ids)), len(self.all_ids) // 3)}
            add("allowed_third", r.choice(common + mid[:100]), allowed=sorted(self.all_ids[i] for i in sample))
        return out[:count]


# --------------------------------------------------------------------------------------------------------------------
# Layouts under test
# --------------------------------------------------------------------------------------------------------------------
class Prototype(SparseIndex):
    """The contentless layouts. ``SPEC`` selects the variant; the write path mirrors ``SparseIndex`` statement by
    statement (an upsert deletes the old document and inserts a new one, so rowids follow insertion order)."""

    SPEC: dict = {}

    def _init_schema(self) -> None:
        spec = self.SPEC
        opts = ["text", "content=''", "contentless_delete=1", f"tokenize='{spec.get('tokenize', 'porter unicode61')}'"]
        if spec.get("detail"):
            opts.append(f"detail={spec['detail']}")
        if spec.get("columnsize") is not None:
            opts.append(f"columnsize={spec['columnsize']}")
        with self._conn:
            self._conn.execute("BEGIN")
            self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(" + ", ".join(opts) + ")")
            if spec["map"] == "brief":
                self._conn.execute("CREATE TABLE IF NOT EXISTS chunk_rows(chunk_id TEXT PRIMARY KEY, doc_rowid INTEGER UNIQUE NOT NULL, "
                                   "session_id TEXT NOT NULL, kind TEXT NOT NULL)")
            else:       # "rowid": the FTS rowid is the chunk_rows rowid; sessions and kinds are small integers
                self._conn.execute("CREATE TABLE IF NOT EXISTS chunk_rows(rowid INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE, "
                                   "session_no INTEGER NOT NULL, kind INTEGER NOT NULL)")
                self._conn.execute("CREATE TABLE IF NOT EXISTS sparse_sessions(session_no INTEGER PRIMARY KEY, session_id TEXT NOT NULL UNIQUE)")
            if spec.get("pgsz"):
                self._conn.execute("INSERT INTO chunks_fts(chunks_fts, rank) VALUES('pgsz', ?)", (spec["pgsz"],))
            self._conn.commit()
        self._sessions: dict[str, int] = {}
        self._kinds = {"turn": 0, "tool_call": 1, "tool_result": 2, "policy": 3, "meta": 4}

    # ---- exact bm25 statistics --------------------------------------------------------------------------------------
    # FTS5 leaves nTotalRow and the total token count (the "averages" record, row 1 of chunks_fts_data) untouched when a
    # contentless_delete row is deleted, so bm25() (idf and average document length) drifts away from the layout that
    # stores its content. ``exact`` takes the deleted documents out of that record, in the transaction of the delete and
    # before its first FTS5 write (FTS5 keeps its own copy of the totals once it has written, and saves it at commit).
    def _retire(self, rowids: list[int]) -> None:
        if not rowids or not self.SPEC.get("exact"):
            return
        rows = tokens = 0
        for start in range(0, len(rowids), 500):
            batch = rowids[start:start + 500]
            for _id, sz in self._conn.execute("SELECT id, sz FROM chunks_fts_docsize WHERE id IN (" + ",".join("?" * len(batch)) + ")", batch):
                rows += 1
                tokens += _varints(sz)[0]
        block = self._conn.execute("SELECT block FROM chunks_fts_data WHERE id=1").fetchone()[0]
        n, t = _varints(block)
        self._conn.execute("UPDATE chunks_fts_data SET block=? WHERE id=1", (_varint(n - rows) + _varint(t - tokens),))

    def _rowids(self, ids) -> list[int]:
        out = []
        col = "doc_rowid" if self.SPEC["map"] == "brief" else "rowid"
        for start in range(0, len(ids), 500):
            batch = list(ids[start:start + 500])
            out += [r[0] for r in self._conn.execute(f"SELECT {col} FROM chunk_rows WHERE chunk_id IN (" + ",".join("?" * len(batch)) + ")", batch)]
        return out

    # ---- helpers --------------------------------------------------------------------------------------------------
    def _session_no(self, session_id: str, create: bool) -> int | None:
        no = self._sessions.get(session_id)
        if no is None:
            row = self._conn.execute("SELECT session_no FROM sparse_sessions WHERE session_id=?", (session_id,)).fetchone()
            if row is None and create:
                no = self._conn.execute("INSERT INTO sparse_sessions(session_id) VALUES (?)", (session_id,)).lastrowid
            elif row is not None:
                no = row[0]
            if no is not None:
                self._sessions[session_id] = no
        return no

    def _drop(self, chunk_id: str) -> None:
        if self.SPEC["map"] == "brief":
            row = self._conn.execute("SELECT doc_rowid FROM chunk_rows WHERE chunk_id=?", (chunk_id,)).fetchone()
            if row:
                self._conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (row[0],))
                self._conn.execute("DELETE FROM chunk_rows WHERE chunk_id=?", (chunk_id,))
        else:
            row = self._conn.execute("SELECT rowid FROM chunk_rows WHERE chunk_id=?", (chunk_id,)).fetchone()
            if row:
                self._conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (row[0],))
                self._conn.execute("DELETE FROM chunk_rows WHERE rowid=?", (row[0],))

    def _add(self, chunk_id, session_id, kind, text) -> None:
        masked = mask_secrets(text)
        if self.SPEC["map"] == "brief":
            rowid = self._conn.execute("INSERT INTO chunks_fts(text) VALUES (?)", (masked,)).lastrowid
            self._conn.execute("INSERT INTO chunk_rows VALUES (?,?,?,?)", (chunk_id, rowid, session_id, kind))
        else:
            no = self._session_no(session_id, True)
            rowid = self._conn.execute("INSERT INTO chunk_rows(chunk_id, session_no, kind) VALUES (?,?,?)",
                                       (chunk_id, no, self._kinds.get(kind, 9))).lastrowid
            self._conn.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?,?)", (rowid, masked))

    # ---- the SparseIndex interface -------------------------------------------------------------------------------
    def clear(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM chunks_fts")
            self._conn.execute("DELETE FROM chunk_rows")
            self._conn.commit()

    def upsert(self, chunk_id, session_id, kind, text) -> None:
        with self._conn:
            self._conn.execute("BEGIN")
            self._retire(self._rowids([chunk_id]))
            self._drop(chunk_id)
            self._add(chunk_id, session_id, kind, text)
            self._conn.commit()

    def delete_ids(self, ids) -> None:
        ids = list(ids)
        with self._conn:
            self._conn.execute("BEGIN")
            self._retire(self._rowids(ids))
            for cid in ids:
                self._drop(cid)
            self._conn.commit()

    def upsert_many(self, chunks) -> None:
        latest: dict[str, object] = {}
        for chunk in chunks:                       # the last of a repeated id wins, at its own position
            latest.pop(chunk.chunk_id, None)
            latest[chunk.chunk_id] = chunk
        with self._conn:
            self._conn.execute("BEGIN")
            self._retire(self._rowids(list(latest)))
            for chunk in latest.values():
                self._drop(chunk.chunk_id)
                self._add(chunk.chunk_id, chunk.session_id, chunk.kind, chunk.text)
            self._conn.commit()

    def search(self, query, top_k=20, *, session_id=None, allowed_ids=None):
        q = _fts_query(query)
        sql = "SELECT r.chunk_id AS chunk_id, bm25(chunks_fts) AS score FROM chunks_fts JOIN chunk_rows r ON r.{key} = chunks_fts.rowid WHERE chunks_fts MATCH ?"
        args: list = [q]
        if self.SPEC["map"] == "brief":
            sql = sql.format(key="doc_rowid")
            if session_id:
                sql += " AND r.session_id = ?"
                args.append(session_id)
        else:
            sql = sql.format(key="rowid")
            if session_id:
                no = self._session_no(session_id, False)
                if no is None:
                    return []
                sql += " AND r.session_no = ?"
                args.append(no)
        sql += " ORDER BY score LIMIT ?"
        args.append(-1 if allowed_ids is not None else top_k)
        out = []
        for row in self._conn.execute(sql, args):
            if allowed_ids is not None and row["chunk_id"] not in allowed_ids:
                continue
            if len(out) >= top_k:
                break
            out.append((row["chunk_id"], float(-row["score"]), len(out) + 1))
        return out


def _varints(blob: bytes) -> list[int]:
    out, i = [], 0
    while i < len(blob):
        v = 0
        for k in range(9):
            c = blob[i]
            i += 1
            if k == 8:
                v = (v << 8) | c
                break
            v = (v << 7) | (c & 0x7F)
            if not c & 0x80:
                break
        out.append(v)
    return out


def _varint(v: int) -> bytes:
    if v < 0x80:
        return bytes([v])
    groups = []
    while v:
        groups.append(v & 0x7F)
        v >>= 7
    groups.reverse()
    return bytes([g | 0x80 for g in groups[:-1]] + [groups[-1]])


def _proto(**spec):
    return type("Proto", (Prototype,), {"SPEC": spec})


CANDIDATES: dict[str, dict] = {
    # the released layout, and its page-size variants (overflow waste of the stored text)
    "a": dict(cls=SparseIndex, page=4096, note="current schema (content stored, porter unicode61, 4 KiB pages)"),
    "b8": dict(cls=SparseIndex, page=8192, note="current schema, 8 KiB pages"),
    "b16": dict(cls=SparseIndex, page=16384, note="current schema, 16 KiB pages"),
    "b32": dict(cls=SparseIndex, page=32768, note="current schema, 32 KiB pages"),
    # contentless
    "c": dict(cls=_proto(map="brief"), page=4096, note="contentless_delete=1; chunk_id/session_id/kind in chunk_rows (TEXT keys)"),
    "c16": dict(cls=_proto(map="brief"), page=16384, note="(c) with 16 KiB pages"),
    "d": dict(cls=_proto(map="brief", detail="column"), page=4096, note="(c) + detail=column"),
    "e": dict(cls=_proto(map="brief", tokenize="unicode61"), page=4096, note="(c) + tokenize=unicode61 (no porter)"),
    "f1": dict(cls=_proto(map="rowid"), page=4096, note="(c) with the FTS rowid = chunk_rows rowid, session/kind as integers"),
    "f1p16": dict(cls=_proto(map="rowid"), page=4096, pgsz=16000, note="(f1) + FTS5 leaf page size pgsz=16000"),
    "f1p4": dict(cls=_proto(map="rowid"), page=4096, pgsz=4000, note="(f1) + FTS5 leaf page size pgsz=4000"),
    "f1b8": dict(cls=_proto(map="rowid"), page=8192, note="(f1) with 8 KiB pages"),
    "f1b16": dict(cls=_proto(map="rowid"), page=16384, note="(f1) with 16 KiB pages"),
    "cx": dict(cls=_proto(map="brief", exact=True), page=4096, note="(c) + exact bm25 statistics (averages record corrected on delete)"),
    "f2": dict(cls=_proto(map="rowid", exact=True), page=4096, note="(f1) + exact bm25 statistics (averages record corrected on delete)"),
    "f2b16": dict(cls=_proto(map="rowid", exact=True), page=16384, note="(f2) with 16 KiB pages"),
    "p": dict(cls=CompactSparseIndex, page=4096, note="the shipped compact format (index/sparse.py CompactSparseIndex)"),
    "f1cs0": dict(cls=_proto(map="rowid", columnsize=0), page=4096, note="(f1) + columnsize=0 (no docsize table; bm25 expected to fail)"),
}
for _key in ("f1p16", "f1p4"):
    CANDIDATES[_key]["cls"] = _proto(map="rowid", exact=True, pgsz=CANDIDATES[_key]["pgsz"])
    CANDIDATES[_key]["note"] += " (exact statistics)"


def make_index(spec: dict, path: Path):
    """Open a candidate; the page size must be set before the first table exists."""
    page = spec["page"]
    if page != 4096:
        conn = sqlite3.connect(str(path))
        conn.execute(f"PRAGMA page_size={page}")
        conn.execute("CREATE TABLE _page(x)")
        conn.execute("DROP TABLE _page")
        conn.commit()
        conn.close()
    return spec["cls"](path)


# --------------------------------------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------------------------------------
def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(math.ceil(q * len(s))) - 1)]


def file_bytes(path: Path) -> int:
    total = 0
    for suffix in ("", "-journal", "-wal", "-shm"):
        p = Path(str(path) + suffix)
        if p.exists():
            total += p.stat().st_size
    return total


def dbstat(conn: sqlite3.Connection) -> dict[str, dict]:
    out = {}
    for name, size, unused, payload in conn.execute(
            "SELECT name, SUM(pgsize), SUM(unused), SUM(payload) FROM dbstat GROUP BY name ORDER BY SUM(pgsize) DESC"):
        out[name] = {"bytes": int(size), "unused": int(unused), "payload": int(payload)}
    return out


def run_queries(index, queries: list[dict], *, timed: bool = False) -> list[dict]:
    results = []
    for q in queries:
        allowed = set(q["allowed"]) if q.get("allowed") is not None else None
        start = time.perf_counter()
        try:
            hits = index.search(q["q"], top_k=q["top_k"], session_id=q.get("session_id"), allowed_ids=allowed)
            error = None
        except sqlite3.Error as exc:
            hits, error = [], f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start
        results.append({"hits": [[cid, score] for cid, score, _rank in hits], "error": error, "ms": elapsed * 1000.0})
    return results


def compare(reference: list[dict], other: list[dict], queries: list[dict], tolerance: float = 1e-9) -> dict:
    diffs = []
    for i, (a, b, q) in enumerate(zip(reference, other, queries)):
        if b["error"]:
            diffs.append({"i": i, "cls": q["cls"], "q": q["q"], "cause": "error: " + b["error"]})
            continue
        ids_a, ids_b = [h[0] for h in a["hits"]], [h[0] for h in b["hits"]]
        if ids_a != ids_b:
            common = len(set(ids_a) & set(ids_b))
            first = next((j for j, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y), min(len(ids_a), len(ids_b)))
            same_set = set(ids_a) == set(ids_b)
            diffs.append({"i": i, "cls": q["cls"], "q": q["q"], "cause": "order" if same_set else "set",
                          "n_ref": len(ids_a), "n_new": len(ids_b), "common": common, "first_diff": first})
            continue
        worst = max((abs(x[1] - y[1]) / max(1.0, abs(x[1])) for x, y in zip(a["hits"], b["hits"])), default=0.0)
        if worst > tolerance:
            diffs.append({"i": i, "cls": q["cls"], "q": q["q"], "cause": "score", "worst_rel": worst})
    census: dict[str, int] = {}
    for d in diffs:
        key = f"{d['cls']}:{d['cause'].split(':')[0]}"
        census[key] = census.get(key, 0) + 1
    return {"queries": len(queries), "divergent": len(diffs), "census": census, "divergences": diffs[:40]}


def latency(results: list[dict], queries: list[dict]) -> dict:
    by: dict[str, list[float]] = {}
    for r, q in zip(results, queries):
        by.setdefault(q["cls"], []).append(r["ms"])
    out = {cls: {"p50_ms": round(percentile(v, 0.5), 3), "p95_ms": round(percentile(v, 0.95), 3), "n": len(v)} for cls, v in by.items()}
    allv = [r["ms"] for r in results]
    out["all"] = {"p50_ms": round(percentile(allv, 0.5), 3), "p95_ms": round(percentile(allv, 0.95), 3), "n": len(allv)}
    return out


def hit_profile(results: list[dict], queries: list[dict]) -> dict:
    by: dict[str, list[int]] = {}
    for r, q in zip(results, queries):
        by.setdefault(q["cls"], []).append(len(r["hits"]))
    return {cls: {"min": min(v), "median": statistics.median(v), "max": max(v)} for cls, v in by.items()}


def measure(name: str, profile: str, events: int, work: Path, out: dict, *, queries_n: int, stages: set[str], ref_root: Path) -> None:
    spec = CANDIDATES[name]
    db = work / f"{name}-{profile}-{events}.sqlite"
    ref_dir = Path(ref_root)
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_path = lambda stage: ref_dir / f"{profile}-{events}-{stage}.json"      # noqa: E731
    corpus = Corpus(profile, events)
    rec: dict = {"candidate": name, "note": spec["note"], "profile": profile, "events": events,
                 "sqlite": sqlite3.sqlite_version, "checkpoints": []}
    out[f"{name}/{profile}/{events}"] = rec
    try:
        index = make_index(spec, db)
    except sqlite3.Error as exc:
        rec["build"] = {"events": 0, "file_bytes": 0, "bytes_per_event": 0, "upsert_seconds": 0, "error": f"{type(exc).__name__}: {exc}"}
        db.unlink(missing_ok=True)
        return
    t_upsert = t_total = 0.0
    done = 0
    wall = time.perf_counter()
    gen = corpus.batches()
    while True:
        g0 = time.perf_counter()
        batch = next(gen, None)
        if batch is None:
            break
        t0 = time.perf_counter()
        index.upsert_many(batch)
        t1 = time.perf_counter()
        t_upsert += t1 - t0
        t_total += t1 - g0
        done += len(batch)
        if done % CHECKPOINT_EVERY < BATCH or done == events:
            if not rec["checkpoints"] or rec["checkpoints"][-1]["events"] != done:
                rec["checkpoints"].append({"events": done, "file_bytes": file_bytes(db), "upsert_seconds": round(t_upsert, 2),
                                           "bytes_per_event": round(file_bytes(db) / done, 1)})
    rec["build"] = {"events": done, "file_bytes": file_bytes(db), "bytes_per_event": round(file_bytes(db) / done, 1),
                    "upsert_seconds": round(t_upsert, 2), "wall_seconds": round(time.perf_counter() - wall, 2),
                    "text_chars": corpus.total_chars, "mean_text_bytes": round(corpus.total_chars / done, 1),
                    "kinds": corpus.kind_count, "sessions": len(corpus.sessions)}
    rec["count"] = index.count()
    t = time.perf_counter()
    rec["dbstat"] = dbstat(index._conn)
    rec["dbstat_seconds"] = round(time.perf_counter() - t, 1)
    queries = corpus.queries(queries_n)
    is_ref = name == "a"
    # --- search parity and latency (before any delete)
    index.search("warm up", top_k=5)
    run_queries(index, queries)                                    # warm the page cache
    results = run_queries(index, queries)
    rec["latency"] = latency(results, queries)
    rec["hits"] = hit_profile(results, queries)
    if is_ref:
        ref_path("pre").write_text(json.dumps({"queries": queries, "results": results}))
    elif ref_path("pre").exists():
        ref = json.loads(ref_path("pre").read_text())
        rec["parity_pre"] = compare(ref["results"], results, queries)
    # --- delete 1000 events and re-upsert them
    if "ops" in stages:
        keep = corpus.keep[:1000]
        ids = [d.chunk_id for d in keep]
        t = time.perf_counter()
        index.delete_ids(ids)
        t_del = time.perf_counter() - t
        size_del = file_bytes(db)
        t = time.perf_counter()
        for start in range(0, len(keep), BATCH):
            index.upsert_many(keep[start:start + BATCH])
        t_up = time.perf_counter() - t
        rec["ops"] = {"events": len(keep), "delete_ids_seconds": round(t_del, 3), "reupsert_seconds": round(t_up, 3),
                      "size_after_delete": size_del, "size_after_reupsert": file_bytes(db), "count_after": index.count()}
        results = run_queries(index, queries)
        if is_ref:
            ref_path("post").write_text(json.dumps({"queries": queries, "results": results}))
        elif ref_path("post").exists():
            ref = json.loads(ref_path("post").read_text())
            rec["parity_post_ops"] = compare(ref["results"], results, queries)
    # --- integrity, optimize, vacuum
    if "maint" in stages:
        t = time.perf_counter()
        rec["integrity_check"] = {"pragma": index._conn.execute("PRAGMA integrity_check").fetchone()[0],
                                  "seconds": round(time.perf_counter() - t, 1)}
        t = time.perf_counter()
        try:
            index._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")
            rec["integrity_check"]["fts5"] = "ok"
        except sqlite3.Error as exc:
            rec["integrity_check"]["fts5"] = f"{type(exc).__name__}: {exc}"
        rec["integrity_check"]["fts5_seconds"] = round(time.perf_counter() - t, 1)
        index._conn.commit()
        rec["freelist"] = {"pages": index._conn.execute("PRAGMA freelist_count").fetchone()[0],
                           "page_count": index._conn.execute("PRAGMA page_count").fetchone()[0],
                           "page_size": index._conn.execute("PRAGMA page_size").fetchone()[0]}
        before = file_bytes(db)
        t = time.perf_counter()
        index._conn.execute("VACUUM")
        rec["vacuum"] = {"seconds": round(time.perf_counter() - t, 1), "before": before, "after": file_bytes(db),
                         "bytes_per_event": round(file_bytes(db) / index.count(), 1)}
        before = file_bytes(db)
        t = time.perf_counter()
        index._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
        index._conn.commit()
        rec["optimize"] = {"seconds": round(time.perf_counter() - t, 1), "before": before, "after": file_bytes(db)}
        results = run_queries(index, queries)
        rec["latency_after_optimize"] = latency(results, queries)
        if ref_path("post").exists() and not is_ref:
            rec["parity_after_optimize"] = compare(json.loads(ref_path("post").read_text())["results"], results, queries)
        before = file_bytes(db)
        t = time.perf_counter()
        index._conn.execute("VACUUM")
        rec["optimize_vacuum"] = {"seconds": round(time.perf_counter() - t, 1), "before": before, "after": file_bytes(db),
                                  "bytes_per_event": round(file_bytes(db) / index.count(), 1)}
        rec["dbstat_final"] = dbstat(index._conn)
    # --- orphans: delete a seeded tenth of the documents in sorted-id batches, as the rebuild command and a global search do
    if "orphans" in stages:
        victims = sorted(random.Random(11).sample(corpus.all_ids, max(1, len(corpus.all_ids) // 10)))
        before = file_bytes(db)
        times = []
        for start in range(0, len(victims), BATCH):
            t = time.perf_counter()
            index.delete_ids(victims[start:start + BATCH])
            times.append(time.perf_counter() - t)
        total = sum(times)
        rec["orphans"] = {"documents": len(victims), "seconds": round(total, 2), "per_10k_seconds": round(total * 10_000 / len(victims), 2),
                          "longest_batch_seconds": round(max(times), 3), "file_before": before, "file_after": file_bytes(db),
                          "free_bytes": index._conn.execute("PRAGMA freelist_count").fetchone()[0] * index._conn.execute("PRAGMA page_size").fetchone()[0],
                          "count_after": index.count()}
        if hasattr(index, "verify_statistics"):
            rec["orphans"]["statistics_ok"] = index.verify_statistics()["ok"]
    index._conn.close()
    db.unlink(missing_ok=True)


def mask_cost(profile: str, events: int) -> dict:
    corpus = Corpus(profile, events)
    t_mask = 0.0
    n = 0
    for batch in corpus.batches():
        t = time.perf_counter()
        for doc in batch:
            mask_secrets(doc.text)
        t_mask += time.perf_counter() - t
        n += len(batch)
    return {"events": n, "mask_seconds": round(t_mask, 2)}


def corpus_stats(profile: str, events: int) -> dict:
    t = time.perf_counter()
    corpus = Corpus(profile, events)
    sizes: list[int] = []
    tokens = 0
    df: dict[str, int] = {}
    for batch in corpus.batches():
        for doc in batch:
            sizes.append(len(doc.text.encode()))
            toks = {t.lower() for t in TOKEN_RE.findall(doc.text)}
            tokens += len(toks)
            for tok in toks:
                df[tok] = df.get(tok, 0) + 1
    sizes.sort()
    return {"profile": profile, "events": events, "gen_seconds": round(time.perf_counter() - t, 1),
            "mean_bytes": round(statistics.mean(sizes), 1), "median_bytes": sizes[len(sizes) // 2],
            "p90": sizes[int(len(sizes) * 0.9)], "p99": sizes[int(len(sizes) * 0.99)], "max": sizes[-1],
            "distinct_regex_tokens": len(df), "tokens_in_docs": tokens, "kinds": corpus.kind_count,
            "sessions": len(corpus.sessions), "singletons": sum(1 for v in df.values() if v == 1)}


def summary(rec: dict) -> str:
    b = rec["build"]
    if b.get("error"):
        return f"{rec['candidate']:6s} {rec['profile']:5s} cannot be built: {b['error']}  ({rec['note']})"
    lines = [f"{rec['candidate']:6s} {rec['profile']:5s} {rec['events']:>7d} events  sqlite {rec['sqlite']}  {b['file_bytes'] / 1e6:9.1f} MB  "
             f"{b['bytes_per_event']:9.1f} B/event  upsert {b['upsert_seconds']:7.1f} s  ({rec['note']})"]
    for key in ("parity_pre", "parity_post_ops", "parity_after_optimize"):
        if key in rec:
            p = rec[key]
            lines.append(f"       {key}: {p['divergent']}/{p['queries']} queries diverge" + (f" {p['census']}" if p["divergent"] else ""))
    if "latency" in rec:
        lines.append("       latency p50/p95 ms: " + "  ".join(f"{c}={v['p50_ms']}/{v['p95_ms']}" for c, v in rec["latency"].items()))
    if "ops" in rec:
        o = rec["ops"]
        lines.append(f"       delete_ids({o['events']}) {o['delete_ids_seconds']} s, re-upsert {o['reupsert_seconds']} s")
    if "orphans" in rec:
        o = rec["orphans"]
        lines.append(f"       delete {o['documents']} documents (orphans) {o['seconds']} s = {o['per_10k_seconds']} s per 10,000, longest batch "
                     f"{o['longest_batch_seconds']} s, file {o['file_before'] / 1e6:.1f} -> {o['file_after'] / 1e6:.1f} MB ({o['free_bytes'] / 1e6:.1f} MB free pages)"
                     + (f", statistics ok: {o['statistics_ok']}" if "statistics_ok" in o else ""))
    if "optimize" in rec:
        o, v, ov = rec["optimize"], rec["vacuum"], rec["optimize_vacuum"]
        lines.append(f"       vacuum {v['seconds']} s -> {v['after'] / 1e6:.1f} MB; optimize {o['seconds']} s {o['before'] / 1e6:.1f} -> {o['after'] / 1e6:.1f} MB; "
                     f"then vacuum {ov['seconds']} s -> {ov['after'] / 1e6:.1f} MB; freelist {rec['freelist']['pages']}/{rec['freelist']['page_count']} pages; "
                     f"integrity {rec['integrity_check']['pragma']} {rec['integrity_check']['seconds']} s, fts5 {rec['integrity_check']['fts5']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="codex")
    ap.add_argument("--events", type=int, default=20_000)
    ap.add_argument("--candidate", default="a", help="comma separated names (see --list); 'a' also writes the reference results")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--stages", default="ops,maint", help="comma separated: ops (delete + re-upsert), maint (integrity, optimize, vacuum), "
                    "orphans (delete a tenth of the documents in batches of 500, as sparse-rebuild does; off by default)")
    ap.add_argument("--out", help="write the results as JSON")
    ap.add_argument("--tmpdir", default=DEFAULT_TMP)
    ap.add_argument("--drop-ref", action="store_true", help="also delete the reference results (<tmpdir>/ref) when done")
    ap.add_argument("--stats", action="store_true", help="print corpus statistics and exit")
    ap.add_argument("--mask-cost", action="store_true", help="time mask_secrets over the corpus and exit")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        for key, spec in CANDIDATES.items():
            print(f"{key:8s} {spec['note']}")
        return 0
    if args.stats:
        print(json.dumps(corpus_stats(args.profile, args.events), indent=2))
        return 0
    if args.mask_cost:
        print(json.dumps(mask_cost(args.profile, args.events)))
        return 0
    Path(args.tmpdir).mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="bench-sparse-", dir=args.tmpdir))
    out: dict = {}
    try:
        for name in args.candidate.split(","):
            measure(name, args.profile, args.events, work, out, queries_n=args.queries,
                    stages=set(args.stages.split(",")) - {""}, ref_root=Path(args.tmpdir) / "ref")
            print(summary(out[f"{name}/{args.profile}/{args.events}"]), flush=True)
            if args.out:
                Path(args.out).write_text(json.dumps(out, indent=1))
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if args.drop_ref:
            shutil.rmtree(Path(args.tmpdir) / "ref", ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
