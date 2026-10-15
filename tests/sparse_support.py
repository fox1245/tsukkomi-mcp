"""Shared fixtures of the sparse-index format tests: a seeded synthetic corpus, a query set that covers every class the
benchmark uses, and the comparison of a legacy and a compact index (same ranked ids, same bm25 scores)."""
from __future__ import annotations

from collections import namedtuple
import hashlib
from pathlib import Path
import random

from self_directing_mcp.index.sparse import CompactSparseIndex, SparseIndex

Doc = namedtuple("Doc", "chunk_id session_id kind text")

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "run", "running", "runs", "ran", "test", "tests", "tested",
         "testing", "error", "errors", "file", "files", "path", "commit", "committed", "queue", "worker", "index"]
IDENTS = ["foo_bar", "pre-commit", "--no-verify", "max_retry_count", "snake_case_name", "kebab-case-name", "request_id", "a-b-c"]
KOREAN = ["테스트가 실패했어요", "파일을 수정해줘", "에러 확인", "커밋 메시지 추가"]
SECRET = "sk-" + "Zq9" * 12                 # masked on write: "sk-***"
KINDS = ("turn", "tool_call", "tool_result", "meta")
TIES = ["ok", "done", "ok done", "yes", "continue"]


def corpus(n: int = 400, seed: int = 5, sessions: int = 5, provider: str = "codex") -> list[Doc]:
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        roll = rng.random()
        if roll < 0.15:
            text = rng.choice(TIES)                                          # exact duplicates: equal scores, ties
        elif roll < 0.25:
            text = rng.choice(KOREAN) + " " + rng.choice(WORDS)
        else:
            parts = rng.choices(WORDS, k=rng.randrange(1, 40))
            for _ in range(rng.randrange(0, 3)):
                parts.insert(rng.randrange(len(parts) + 1), rng.choice(IDENTS))
            text = " ".join(parts)
        if rng.random() < 0.05:
            text += " token=" + SECRET
        sid = f"s{i % sessions}"
        docs.append(Doc(f"{provider}:{sid}:{i}:0:{i:016x}", sid, rng.choice(KINDS), text))
    return docs


def queries(docs: list[Doc], seed: int = 9) -> list[dict]:
    """Every class of the benchmark: single and multi token, hyphenated and snake_case, Korean, no match, empty, one
    character, a session filter, an ``allowed_ids`` filter (alone and with a session), and ``top_k`` 1, 5, 20 and 100."""
    rng = random.Random(seed)
    sessions = sorted({d.session_id for d in docs})
    ids = [d.chunk_id for d in docs]
    out = []

    def add(q, **extra):
        out.append({"q": q, "top_k": extra.pop("top_k", rng.choice((1, 5, 20, 100))), **extra})

    for word in WORDS:
        add(word)
    for _ in range(25):
        add(" ".join(rng.sample(WORDS, rng.randrange(2, 5))))
    for ident in IDENTS:
        add(ident)
        add(ident + " " + rng.choice(WORDS))
    for ko in KOREAN:
        add(ko)
        add(ko + " error")
    for tie in TIES:
        add(tie, top_k=100)
        add(tie, top_k=3)
    for q in ("zzzqqq", "nonexistent-token", "", "a", "x y z", '"', "OR AND NOT", "sk-***", "sk"):
        add(q)
    for sid in sessions + ["no-such-session"]:
        for word in rng.sample(WORDS, 3):
            add(word, session_id=sid)
    for _ in range(15):
        allowed = set(rng.sample(ids, len(ids) // 3))
        add(rng.choice(WORDS), allowed_ids=allowed)
        sid = rng.choice(sessions)
        add(rng.choice(WORDS) + " " + rng.choice(WORDS), session_id=sid, allowed_ids={i for i in allowed if f":{sid}:" in i})
    add("error", allowed_ids=set())                                          # an empty allow list matches nothing
    add("error", allowed_ids=set(ids), top_k=100)
    return out


def run(index, query_set: list[dict]) -> list[list[tuple]]:
    return [index.search(q["q"], q["top_k"], session_id=q.get("session_id"), allowed_ids=q.get("allowed_ids")) for q in query_set]


def assert_same_results(legacy, compact, query_set: list[dict], what: str = "") -> int:
    """Same ranked ids and bm25 scores for every query. Returns the number of queries that found something."""
    found = 0
    for q, a, b in zip(query_set, run(legacy, query_set), run(compact, query_set)):
        label = f"{what} {({k: v for k, v in q.items() if k != 'allowed_ids'})}" + (" allowed" if q.get("allowed_ids") is not None else "")
        assert [h[0] for h in a] == [h[0] for h in b], label
        assert [h[2] for h in a] == [h[2] for h in b], label
        for x, y in zip(a, b):
            assert abs(x[1] - y[1]) <= 1e-12 * max(1.0, abs(x[1])), (label, x, y)
        found += bool(a)
    return found


def batches(items, size=50):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def build(path: Path, docs: list[Doc], cls=SparseIndex, size: int = 50):
    index = cls(path)
    for batch in batches(docs, size):
        index.upsert_many(batch)
    return index


def pair(tmp_path: Path, docs: list[Doc], size: int = 50):
    """A legacy and a compact index holding the same documents, written in the same order."""
    return build(tmp_path / "legacy.sqlite", docs, SparseIndex, size), build(tmp_path / "compact.sqlite", docs, CompactSparseIndex, size)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
