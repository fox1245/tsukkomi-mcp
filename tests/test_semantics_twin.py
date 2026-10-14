"""The stage-1 semantics through the whole engine: an incremental engine against a twin that evaluates in full.

``Twin`` (tests/test_incremental_engine.py) feeds one growing Claude Code session to two engines, one auditing
incrementally with the shadow comparison off, the other in full, and requires identical payloads after every append.
Here the session walks through each semantic in turn: a refused call whose refusal arrives later, a masked and then a
plain test run that gate a commit, and a contract anchored to something that is not a user message.
"""
from __future__ import annotations

from self_directing_mcp.schemas import ContractRule
from semantic_support import (
    PROJECT, SID, bash, denied_result, finding, human, tool_result, write,
)
from test_incremental_engine import Twin

RM = {"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"}
GATE = {"id": "tests-first", "type": "must", "scope": "tool_call", "regex": "pytest", "before_regex": "git commit",
        "requires_success": True}
COMMIT = {"tool_name": "Bash", "arguments": {"command": "git commit -m change"}}


def test_incremental_and_full_engines_agree_through_every_semantic(tmp_path):
    root = tmp_path / "projects"
    path = root / PROJECT / f"{SID}.jsonl"
    write(path, human("never rm -rf, and run the tests before you commit"))
    twin = Twin(tmp_path, "claude", claude_projects_dir=root)
    try:
        twin.upsert([RM, GATE])
        twin.audit(SID, path=str(path))
        steps = [
            [bash("t1", "rm -rf /a")],                                    # executed as far as anyone knows
            [denied_result("t1")],                                        # the refusal arrives later
            [bash("t2", "pytest -q || true"), tool_result("t2", "3 passed")],   # masked: not success evidence
            [bash("t3", "pytest -q"), tool_result("t3", "3 passed")],           # a real run
            [bash("t4", "rm -rf /b"), tool_result("t4", "removed")],            # executed for real
            [bash("t5", "pytest -q | tee log"), tool_result("t5", "3 passed")],  # a newer masked run supersedes
        ]
        seen = []
        for records in steps:
            write(path, *records, mode="a")
            audit, _ = twin.audit(SID, path=str(path))
            action, _ = twin.check(SID, COMMIT, path=str(path))
            seen.append((finding(audit, "no-rm")["verdict"], finding(audit, "tests-first")["verdict"],
                         finding(action, "tests-first")["verdict"]))
        # (no-rm in the audit, tests-first in the audit: no commit happened, so clean; tests-first for the commit)
        assert seen == [
            ("violation", "clean", "violation"),
            ("clean", "clean", "violation"),
            ("clean", "clean", "unknown"),
            ("clean", "clean", "clean"),
            ("violation", "clean", "clean"),
            ("violation", "clean", "unknown"),
        ]
        refused = finding(audit, "no-rm")["proposals_rejected"]
        assert [entry["call_id"] for entry in refused] == ["t1"]
        # An anchor that is not a user message: only an older store (or an unresolved anchor) can hold one.
        result_id = next(c.chunk_id for c in twin.fast.store.list_chunks(SID, "claude") if c.kind == "tool_result")
        for engine in (twin.fast, twin.full):
            engine.contracts.upsert([ContractRule(**{**RM, "id": "odd-anchor", "source_event_id": result_id})])
        audit, _ = twin.audit(SID, path=str(path))
        assert finding(audit, "odd-anchor")["reason"].startswith("anchor_not_user:")
        modes = [report["mode"] for report in twin.reports if report]
        assert modes.count("delta") >= 5, modes
    finally:
        twin.close()
