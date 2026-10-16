"""scripts/bench_sparse.py: the synthetic corpus is seeded and deterministic, a run compares layouts with the reference, notices the
layouts that change results, and deletes its temporary directory."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def bench():
    spec = importlib.util.spec_from_file_location("bench_sparse_for_tests", ROOT / "scripts" / "bench_sparse.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stream(module, profile, events):
    return [doc for batch in module.Corpus(profile, events).batches() for doc in batch]


def test_the_corpus_is_deterministic_and_a_prefix_of_a_bigger_one():
    module = bench()
    first, again, longer = stream(module, "omp", 700), stream(module, "omp", 700), stream(module, "omp", 1400)
    assert first == again and len(first) == 700 and longer[:700] == first
    assert stream(module, "codex", 300) != stream(module, "omp", 300)
    kinds = {doc.kind for doc in longer}
    assert kinds == {"turn", "tool_call", "tool_result", "meta"}
    assert any(any("가" <= ch <= "힣" for ch in doc.text) for doc in longer)       # Korean text is in there
    assert any("sk-" in doc.text or "ghp_" in doc.text or "AKIA" in doc.text or "Bearer" in doc.text or "api_key=" in doc.text for doc in longer)


def test_the_profiles_have_their_mean_event_sizes():
    module = bench()
    for profile, low, high in (("codex", 3500, 9000), ("omp", 900, 2800)):
        docs = stream(module, profile, 3000)
        mean = sum(len(doc.text) for doc in docs) / len(docs)
        assert low < mean < high, (profile, mean)


def test_a_run_compares_the_layouts_and_deletes_its_temporary_directory(tmp_path, capsys):
    module = bench()
    work = tmp_path / "work"
    report = tmp_path / "report.json"
    code = module.main(["--profile", "omp", "--events", "1500", "--candidate", "a,p,c,d", "--queries", "60", "--tmpdir", str(work),
                        "--out", str(report), "--drop-ref"])
    assert code == 0
    runs = {key.split("/")[0]: value for key, value in json.loads(report.read_text()).items()}
    assert set(runs) == {"a", "p", "c", "d"}
    assert all(runs[name]["build"]["events"] == 1500 and runs[name]["count"] == 1500 for name in runs)
    legacy, shipped = runs["a"]["build"]["file_bytes"], runs["p"]["build"]["file_bytes"]
    assert shipped < 0.5 * legacy
    for stage in ("parity_pre", "parity_post_ops", "parity_after_optimize"):
        assert runs["p"][stage]["divergent"] == 0, stage                       # the shipped format is the reference, to the last bit
    assert runs["c"]["parity_pre"]["divergent"] == 0 and runs["c"]["parity_post_ops"]["divergent"] > 0      # stale statistics after deletes
    assert runs["d"]["parity_pre"]["divergent"] > 0 and runs["d"]["parity_pre"]["census"]                   # detail=column changes results
    for name in runs:
        assert runs[name]["integrity_check"]["pragma"] == "ok" and runs[name]["optimize_vacuum"]["after"] > 0
    assert "p      omp" in capsys.readouterr().out
    assert not work.exists() or not list(work.iterdir()), "the temporary directory and the reference results are deleted"


def test_the_candidates_are_listed(capsys):
    module = bench()
    assert module.main(["--list"]) == 0
    listed = capsys.readouterr().out
    for name in ("a ", "b16 ", "c ", "d ", "e ", "f2 ", "p "):
        assert any(line.startswith(name) for line in listed.splitlines()), name


def test_the_korean_stage_is_seeded_and_times_exact_against_prefix_terms(tmp_path, capsys):
    module = bench()
    first, again = module.KoreanCorpus(600), module.KoreanCorpus(600)
    docs = [doc for batch in first.batches() for doc in batch]
    assert docs == [doc for batch in again.batches() for doc in batch] and len(docs) == 600
    assert first.frequent == again.frequent and first.rare == again.rare and len(set(first.stems)) == len(first.stems) == 3000
    assert all(len(doc.text.split()) == 40 for doc in docs) and len({doc.chunk_id for doc in docs}) == 600
    assert all("가" <= ch <= "힣" or ch == " " for doc in docs for ch in doc.text)           # invented words, Hangul only
    work, report = tmp_path / "work", tmp_path / "report.json"
    assert module.main(["--stages", "korean", "--events", "600", "--tmpdir", str(work), "--out", str(report)]) == 0
    rec = json.loads(report.read_text())["korean/600"]
    assert list(json.loads(report.read_text())) == ["korean/600"]                          # alone, the stage runs no candidate
    assert rec["count"] == 600 and rec["suffixes"] == 40 and set(rec["queries"]) == {
        "frequent_exact", "frequent_prefix", "frequent_typed", "rare_exact", "rare_prefix", "rare_typed"}
    q = rec["queries"]
    assert q["frequent_exact"]["match"] == f'"{rec["frequent"]}"' and q["frequent_prefix"]["match"] == f'"{rec["frequent"]}"*'
    assert q["frequent_typed"]["match"] == f'"{rec["frequent"]}을"* OR "{rec["frequent"]}"*'
    for label in ("frequent", "rare"):
        assert q[f"{label}_prefix"]["documents"] >= q[f"{label}_exact"]["documents"]
        assert q[f"{label}_typed"]["documents"] == q[f"{label}_prefix"]["documents"]      # the stem is the prefix the particle form adds
    assert q["frequent_prefix"]["documents"] > 500 > q["rare_prefix"]["documents"]
    assert "frequent_prefix" in capsys.readouterr().out
    assert not work.exists() or not list(work.iterdir())
