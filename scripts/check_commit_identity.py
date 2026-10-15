#!/usr/bin/env python3
"""Fail when a new commit carries an author or committer address that is not a GitHub noreply address.

A public repository keeps the author and committer address of every commit for good, and the history cannot be
cleaned up afterwards without rewriting it. This check runs on the commits a push or pull request adds and accepts only

* ``<name>@users.noreply.github.com`` and ``<id>+<name>@users.noreply.github.com`` (also for bot accounts), and
* ``noreply@github.com`` (commits made by GitHub itself, such as web merges).

It prints the hash of each offending commit and whether the author or the committer is at fault, never the address
itself: the output of a CI run is public too.

    python scripts/check_commit_identity.py --base <sha> --head <sha>

Without ``--base`` (or with an all-zero one, as a push that creates a branch reports) it checks the commits that are not
yet on ``origin/main``, or the head commit alone when that reference does not exist. A base that is not in the
repository (after a force push) is treated the same way. Exit status: 0 clean, 1 offending commits, 2 usage or git error.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys

ALLOWED = re.compile(r"^(?:[A-Za-z0-9._\-\[\]+]+@users\.noreply\.github\.com|noreply@github\.com)$")
ZERO = re.compile(r"^0+$")
_FIELD = "\x1f"


@dataclass(frozen=True)
class Violation:
    commit: str
    role: str                      # "author" or "committer"


def _git(args, cwd):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result


def _exists(rev, cwd) -> bool:
    return _git(["cat-file", "-e", f"{rev}^{{commit}}"], cwd).returncode == 0


def commit_range(base, head, cwd) -> list[str]:
    """The revision arguments that select the commits to check."""
    if base and not ZERO.match(base) and _exists(base, cwd):
        return [head, "--not", base]
    if _exists("refs/remotes/origin/main", cwd) and _git(["rev-parse", head], cwd).stdout.strip() != \
            _git(["rev-parse", "refs/remotes/origin/main"], cwd).stdout.strip():
        return [head, "--not", "refs/remotes/origin/main"]
    return ["-1", head]


def check(base, head, cwd=None) -> list[Violation]:
    cwd = Path(cwd) if cwd else Path.cwd()
    if not _exists(head, cwd):
        raise ValueError("the head commit is not in this repository")
    result = _git(["log", f"--format=%H{_FIELD}%ae{_FIELD}%ce", *commit_range(base, head, cwd)], cwd)
    if result.returncode != 0:
        raise ValueError("git log failed")
    violations = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        commit, author, committer = line.split(_FIELD)
        if not ALLOWED.match(author.strip()):
            violations.append(Violation(commit, "author"))
        if not ALLOWED.match(committer.strip()):
            violations.append(Violation(commit, "committer"))
    return violations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="", help="the commit before the new ones (all zeros or missing: see above)")
    parser.add_argument("--head", default="HEAD", help="the newest commit to check")
    parser.add_argument("--repo", default=".", help="the repository directory")
    args = parser.parse_args(argv)
    try:
        violations = check(args.base, args.head, args.repo)
    except (ValueError, OSError) as exc:
        print(f"check_commit_identity: {exc}", file=sys.stderr)
        return 2
    for item in violations:
        print(f"{item.commit}: the {item.role} address is not a GitHub noreply address")
    if violations:
        print("Commit with a GitHub noreply address (git config user.email <id>+<name>@users.noreply.github.com) and "
              "rewrite these commits before pushing. The address itself is not printed here on purpose.")
        return 1
    print("commit identities: only GitHub noreply addresses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
