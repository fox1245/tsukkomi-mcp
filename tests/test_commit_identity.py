"""scripts/check_commit_identity.py: only GitHub noreply addresses may be committed, and the address is never printed."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_commit_identity.py"
spec = importlib.util.spec_from_file_location("check_commit_identity", SCRIPT)
checker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checker           # dataclasses look their module up here
spec.loader.exec_module(checker)

NOREPLY = "12345+someone@users.noreply.github.com"
PLAIN_NOREPLY = "someone@users.noreply.github.com"
BOT = "49699333+dependabot[bot]@users.noreply.github.com"
GITHUB = "noreply@github.com"
PERSONAL = "person.name@example.invalid"      # a stand-in for any address that is not a noreply address


def _run(repo, *args, env=None):
    merged = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, **(env or {})}
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, env=merged, check=True)


def _commit(repo, name, author, committer=None, message="change"):
    path = Path(repo) / "file.txt"
    path.write_text(path.read_text() + message + "\n" if path.exists() else message + "\n")
    _run(repo, "add", "file.txt")
    _run(repo, "commit", "-q", "-m", message, env={
        "GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": author,
        "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": committer or author})
    return _run(repo, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture
def repo(tmp_path):
    _run(tmp_path, "init", "-q", "-b", "main")
    return tmp_path


def test_allowed_address_forms():
    for good in (NOREPLY, PLAIN_NOREPLY, BOT, GITHUB):
        assert checker.ALLOWED.match(good), good
    for bad in (PERSONAL, "x@gmail.com", "noreply@github.com.evil.example", "a@users.noreply.github.com.evil.example",
                "noreply@users.noreply.github.com@x", "", "no-at-sign"):
        assert not checker.ALLOWED.match(bad), bad


def test_noreply_commits_pass(repo):
    first = _commit(repo, "someone", NOREPLY, message="one")
    _commit(repo, "bot", BOT, message="two")
    head = _commit(repo, "GitHub", PLAIN_NOREPLY, GITHUB, message="three")
    assert checker.check(first, head, repo) == []


def test_a_personal_author_is_reported_with_the_hash_and_role_only(repo, capsys):
    base = _commit(repo, "someone", NOREPLY, message="base")
    bad = _commit(repo, "person", PERSONAL, PERSONAL, message="bad")
    assert checker.check(base, bad, repo) == [checker.Violation(bad, "author"), checker.Violation(bad, "committer")]
    code = checker.main(["--base", base, "--head", bad, "--repo", str(repo)])
    out = capsys.readouterr()
    assert code == 1
    assert bad in out.out
    assert "example.invalid" not in out.out + out.err and "person.name" not in out.out + out.err     # never printed


def test_a_personal_committer_alone_is_reported(repo):
    base = _commit(repo, "someone", NOREPLY, message="base")
    bad = _commit(repo, "someone", NOREPLY, PERSONAL, message="bad")
    assert checker.check(base, bad, repo) == [checker.Violation(bad, "committer")]


def test_only_the_new_range_is_checked(repo):
    old_bad = _commit(repo, "person", PERSONAL, message="old")           # already public, cannot be changed by this check
    good = _commit(repo, "someone", NOREPLY, message="new")
    assert checker.check(old_bad, good, repo) == []
    assert [v.commit for v in checker.check("", good, repo)] == []          # no base, no origin/main: the head commit alone


def test_a_branch_push_without_a_base_checks_what_main_lacks(repo):
    _commit(repo, "person", PERSONAL, message="on main already")
    _run(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _run(repo, "checkout", "-q", "-b", "topic")
    bad = _commit(repo, "person", PERSONAL, message="new on the branch")
    zero = "0" * 40
    assert {v.commit for v in checker.check(zero, bad, repo)} == {bad}
    good = _commit(repo, "someone", NOREPLY, message="fixed")
    assert {v.commit for v in checker.check(zero, good, repo)} == {bad}      # the bad one is still in the range


def test_a_missing_base_after_a_force_push_checks_the_head_commit(repo):
    head = _commit(repo, "person", PERSONAL, message="head")
    assert [v.commit for v in checker.check("f" * 40, head, repo)] == [head, head]


def test_usage_errors_exit_with_two(repo, capsys):
    assert checker.main(["--head", "f" * 40, "--repo", str(repo)]) == 2
    assert "not in this repository" in capsys.readouterr().err
