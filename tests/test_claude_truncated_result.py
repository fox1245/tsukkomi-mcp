"""A Claude Code tool result cut by a read_truncation_notice is partial evidence.

A visible match still counts. A cut result never establishes success, and for lack of a match it can neither
clear a prohibition nor prove an obligation unmet: those findings become unknown with reason ``truncated_result``.
"""
from __future__ import annotations

import pytest

from self_directing_mcp.audit import runner
from self_directing_mcp.schemas import Chunk, ContractRule
from test_claude_provider import (  # noqa: F401  (fixtures)
    OTHER, SID, assistant, attachment, clean_environment, engine, human, main_file, root, sync, tool_result, use, write,
)

SECRET = "LEAKED-SECRET"
BAN = {"id": "no-secret-output", "type": "must_not", "scope": "tool_result", "regex": SECRET, "roles": ["tool"],
       "provider": "claude"}
NEED = {"id": "need-build-ok", "type": "must", "scope": "tool_result", "regex": "BUILD OK", "roles": ["tool"],
        "provider": "claude"}
GATE = {"id": "tests-first", "type": "must", "scope": "tool_call", "regex": "npm test", "before_regex": "deploy",
        "requires_success": True, "provider": "claude"}
DEPLOY = {"tool_name": "Bash", "arguments": {"command": "deploy now"}}


def notice(call_id="toolu_r"):
    return attachment("read_truncation_notice", banner="output was cut", toolUseID=call_id)


def reading(text, *, cut=True, error=False, target="toolu_r"):
    records = [human("read it"), assistant(use("toolu_r", "Read", file_path="/synthetic/a.txt")),
               tool_result("toolu_r", text, block_extra={"is_error": error})]
    return records + ([notice(target)] if cut else [])


def audit(engine, root, records, *rules):
    path = write(main_file(root), *records)
    engine.upsert_contracts(list(rules))
    return engine.audit_session(SID, provider="claude", path=str(path))


def only(result):
    (finding,) = result["findings"]
    return finding


# ---------------------------------------------------------------- must_not

def test_cut_result_without_a_visible_match_cannot_clear_a_prohibition(engine, root):
    result = audit(engine, root, reading("the first lines only"), BAN)
    assert result["coverage"]["complete"] is True
    assert result["verdict"] == "unknown" and only(result)["verdict"] == "unknown"
    assert only(result)["reason"].startswith("truncated_result:")
    assert "truncated_result" in only(result)["basis"]


def test_visible_match_in_a_cut_result_still_counts(engine, root):
    result = audit(engine, root, reading(f"head {SECRET} tail"), BAN)
    assert result["verdict"] == "violation" and only(result)["verdict"] == "violation"


def test_uncut_result_without_a_match_is_clean(engine, root):
    result = audit(engine, root, reading("the first lines only", cut=False), BAN)
    assert result["verdict"] == "clean" and result["coverage"]["complete"] is True


def test_a_notice_for_a_result_that_is_not_in_the_history_changes_nothing(engine, root):
    result = audit(engine, root, reading("the first lines only", target="toolu_elsewhere"), BAN)
    assert result["verdict"] == "clean"


def test_notice_before_its_result_still_marks_it(engine, root):
    records = [human("read it"), assistant(use("toolu_r", "Read", file_path="/synthetic/a.txt")), notice(),
               tool_result("toolu_r", "the first lines only", block_extra={"is_error": False})]
    assert audit(engine, root, records, BAN)["verdict"] == "unknown"


def test_rules_that_do_not_depend_on_tool_results_are_unaffected(engine, root):
    call_ban = {"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": "rm -rf", "provider": "claude",
                "roles": ["assistant"]}
    user_ban = {"id": "no-user-phrase", "type": "must_not", "scope": "message", "regex": "FORBIDDEN-PHRASE",
                "roles": ["user"], "provider": "claude"}
    result = audit(engine, root, reading("the first lines only"), call_ban, user_ban)
    assert result["verdict"] == "clean"
    assert [item["verdict"] for item in result["findings"]] == ["clean", "clean"]


# ---------------------------------------------------------------- must

def test_cut_result_cannot_prove_a_result_obligation_unmet(engine, root):
    control = audit(engine, root, reading("compiling...", cut=False), NEED)
    assert only(control)["verdict"] == "violation"
    cut = audit(engine, root, reading("compiling..."), NEED)
    assert only(cut)["verdict"] == "unknown" and only(cut)["reason"].startswith("truncated_result:")
    seen = audit(engine, root, reading("compiling...\nBUILD OK"), NEED)
    assert only(seen)["verdict"] == "clean"


def test_cut_result_never_establishes_success(engine, root):
    def check(records):
        write(main_file(root), *records)
        engine.upsert_contracts([GATE])
        return engine.check_action(SID, DEPLOY, provider="claude", path=str(main_file(root)))

    def tests(cut, error=False, text="ran"):
        return [human("ship it"), assistant(use("toolu_r", "Bash", command="npm test")),
                tool_result("toolu_r", text, block_extra={"is_error": error})] + ([notice()] if cut else [])

    assert check(tests(cut=False))["verdict"] == "clean"
    cut = check(tests(cut=True))
    assert cut["verdict"] == "unknown" and cut["coverage"]["complete"] is True
    assert cut["findings"][0]["reason"].startswith("truncated_result:")
    failed = check(tests(cut=True, error=True, text="Exit code 1"))
    assert failed["verdict"] == "violation"  # a visible failure is still a failure


def test_cut_result_does_not_hide_a_missing_call(engine, root):
    write(main_file(root), human("ship it"), assistant(use("toolu_r", "Read", file_path="/s/a.txt")),
          tool_result("toolu_r", "text", block_extra={"is_error": False}), notice())
    engine.upsert_contracts([GATE])
    result = engine.check_action(SID, DEPLOY, provider="claude", path=str(main_file(root)))
    assert result["verdict"] == "violation"  # no `npm test` call exists, whatever any result shows


# ---------------------------------------------------------------- evidence plumbing

def test_slim_audit_events_carry_the_truncation_link(engine, root):
    path = write(main_file(root), *reading("the first lines only"))
    assert sync(engine, path)["coverage"]["complete"]
    events = engine.store.capture_audit_events(SID, "claude", [("tool_result", ("tool",))]).events
    marks = [e.meta.get("truncated_result_for") for e in events]
    assert marks.count("toolu_r") == 1 and marks.count(None) == len(events) - 1
    results = [e for e in events if e.kind == "tool_result"]
    assert [e.meta["call_id"] for e in results] == ["toolu_r"]
    assert results[0].text == "[tool_result] the first lines only"


class Store:
    """Full chunks, as the non-slim audit path reads them."""

    def __init__(self, chunks):
        self.chunks = chunks

    def list_chunks(self, session_id=None, provider=None):
        return [c for c in self.chunks if provider in (None, c.provider)]

    def chunk_count(self, session_id=None, provider=None):
        return len(self.list_chunks(session_id, provider))

    def get_chunk(self, chunk_id):
        return next((c for c in self.chunks if c.chunk_id == chunk_id), None)


def chunk(provider, index, kind, text, **meta):
    return Chunk(chunk_id=f"{provider}-{index}", session_id="s", provider=provider, kind=kind, text=text,
                 content_hash=f"h{index}", meta=meta)


def history(provider, text="head only"):
    return [chunk(provider, 0, "tool_call", "[tool_call:Read] {}", role="assistant", call_id="t1", tool_name="Read"),
            chunk(provider, 1, "tool_result", f"[tool_result] {text}", role="tool", call_id="t1", success=True),
            chunk(provider, 2, "meta", "[attachment:read_truncation_notice] truncated_result_for=t1",
                  role="runtime", truncated_result_for="t1")]


def run(provider, rule, chunks):
    return runner.run_audit(session_id="s", contracts=[ContractRule(**rule)], store=Store(chunks), provider=provider,
                            coverage={"complete": True})


def test_only_claude_sessions_apply_the_truncation_link():
    ban = {**BAN, "provider": None}
    assert run("claude", ban, history("claude")).findings[0].verdict == "unknown"
    for provider in ("codex", "omp", "grokbot"):
        assert run(provider, ban, history(provider)).findings[0].verdict == "clean", provider
    assert run("claude", ban, history("claude", f"head {SECRET}")).findings[0].verdict == "violation"


def test_unlinked_or_non_result_events_are_never_cut():
    ban = {**BAN, "provider": None}
    chunks = history("claude")
    chunks[2].meta["truncated_result_for"] = "other-call"
    assert run("claude", ban, chunks).findings[0].verdict == "clean"
    chunks = history("claude")
    chunks[0].meta["call_id"] = "t1"  # a call with the same id is not a result
    chunks[1].kind = "turn"
    assert run("claude", ban, chunks).findings[0].verdict == "clean"


@pytest.mark.parametrize("flag", [None, 7, ["t1"], {"id": "t1"}])
def test_malformed_link_values_are_ignored(flag):
    ban = {**BAN, "provider": None}
    chunks = history("claude")
    chunks[2].meta["truncated_result_for"] = flag
    assert run("claude", ban, chunks).findings[0].verdict == "clean"
