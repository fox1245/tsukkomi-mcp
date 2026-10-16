"""Contract fragility report: reading a sensitive file through a host's file-read tool.

A contract regex written around shell readers (``(cat|head|tail|less)\\s+\\S*\\.env``) never matches the call of a host's
file-read tool, whose text is ``[tool_call:<name>] <JSON arguments>``. The library lists those calls as spellings of the
file-reading families. Every tool-call entry must be the text a parser or a hook really builds, so each one is derived
here from a synthetic transcript (or the committed Grok Bot fixture) and compared with the library, not typed twice.

Strictly informational, like the rest of the report: nothing here changes what a contract matches.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from self_directing_mcp import agy, agy_hook_state, omp
from self_directing_mcp.audit import variants
from self_directing_mcp.grokbot import parse as grok_parse
from self_directing_mcp.tool_payload import tool_call_text
from test_agy_parser_v6 import SID as AGY_SID, make as make_agy, planner, user as agy_user
from test_claude_provider import assistant, human, main_file, parse as parse_claude, use
from test_claude_provider import write as write_claude
from test_reliability import SID, session, write  # noqa: F401  (session is a fixture)

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = variants.load_library()
FAMILIES = {family.family: family for family in LIBRARY.families}

# family -> the file as a tool call names it (an absolute path)
FILES = {
    "read_secret_file": "/work/.env",
    "read_env_production": "/work/.env.production",
    "read_ssh_private_key": "/home/user/.ssh/id_rsa",
    "read_aws_credentials": "/home/user/.aws/credentials",
}
HOSTS = ("claude_read_tool", "agy_view_file_tool", "omp_read_tool")
# family -> (a shell-scoped regex of the kind a model writes, a regex that names only the file)
REGEXES = {
    "read_secret_file": (r"(cat|head|tail|less)\s+\S*\.env(?![\w.])", r"\.env(?![\w.])"),
    "read_env_production": (r"(cat|head|tail|less)\s+\S*\.env\.production", r"\.env\.production"),
    "read_ssh_private_key": (r"(cat|head|tail|less)\s+\S*\.ssh/id_rsa", r"\.ssh/id_rsa"),
    "read_aws_credentials": (r"(cat|head|tail|less)\s+\S*\.aws/credentials", r"\.aws/credentials"),
}
MOTIVATING = r"(cat|head|tail|less)\s+\S*\.env"
TOOL_REGEX = r"(?s)\[tool_call:[^\]]+\].*\.env"


def tool_entries(family_name: str) -> dict[str, variants.Variant]:
    return {v.id.split(".", 1)[1]: v for v in FAMILIES[family_name].variants if v.platform == "tool_call"}


def missed(report: dict, family_name: str) -> dict | None:
    return next((entry for entry in report["families"] if entry["family"] == family_name), None)


@pytest.fixture
def uncapped(monkeypatch):
    monkeypatch.setattr(variants, "MAX_MISSED_PER_FAMILY", 1000)


# ----------------------------------------------------------------------------------------------- the library

def test_the_vocabulary_declares_the_tool_call_platform_and_technique():
    assert "tool_call" in LIBRARY.platforms and "file_read_tool" in LIBRARY.techniques
    assert variants.TOOL_CALL_PLATFORM == "tool_call"
    assert LIBRARY.techniques.index("file_read_tool") == 1          # right after "standard": listed first in a capped list


def test_each_file_has_one_family_with_shell_readers_as_standard_spellings():
    for name in FILES:
        family = FAMILIES[name]
        canonical = [v.text.split()[0] for v in family.variants if v.canonical]
        assert canonical == ["cat", "head", "tail", "less", "more"], name
        assert all(v.platform == "posix" and v.technique == "standard" for v in family.variants if v.canonical)
    # one family per file: no two of them name the same file
    targets = {name: {v.text.split()[1] for v in FAMILIES[name].variants if v.canonical} for name in FILES}
    assert sorted(len(t) for t in targets.values()) == [1, 1, 1, 1]
    assert len({next(iter(t)) for t in targets.values()}) == len(FILES)


def test_each_file_reading_family_carries_the_three_host_tools_as_unverified_non_canonical_entries():
    for name, path in FILES.items():
        entries = tool_entries(name)
        assert set(entries) == set(HOSTS), name
        for entry in entries.values():
            assert entry.canonical is False and entry.verified is False
            assert entry.technique == "file_read_tool" and entry.preconditions
            assert entry.text.startswith("[tool_call:") and f'"{path}"' in entry.text
            assert entry.effect == FAMILIES[name].effect
    # no other family carries tool calls, and no tool call is a standard spelling
    carriers = {family.family for family in LIBRARY.families if any(v.platform == "tool_call" for v in family.variants)}
    assert carriers == set(FILES)
    assert not [v for f in LIBRARY.families for v in f.variants if v.platform == "tool_call" and v.canonical]


def test_the_shell_variants_of_the_new_files_cover_the_documented_readers():
    for name in FILES:
        texts = " | ".join(v.text for v in FAMILIES[name].variants if v.platform != "tool_call")
        for fragment in ("sed -n", "awk 1", "grep .", "cp ", "xxd", "base64", "python3 -c"):
            assert fragment in texts, (name, fragment)
    for name in ("read_secret_file", "read_env_production"):
        texts = [v.text for v in FAMILIES[name].variants]
        assert any(t.startswith("source ") for t in texts) and any(t.startswith(". ") for t in texts), name


# ----------------------------------------------- every tool-call entry is the text the hosts really produce

def claude_text(tmp_path, path):
    file = write_claude(main_file(tmp_path), human("go"), assistant(use("toolu_read", "Read", file_path=path)))
    calls = [chunk for chunk in parse_claude(file) if chunk.kind == "tool_call"]
    assert len(calls) == 1 and calls[0].meta["tool_name"] == "Read"
    return calls[0].text


def omp_text(tmp_path, path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    file = tmp_path / "omp-session.jsonl"
    records = [{"type": "session", "id": "omp-session"},
               {"type": "message", "message": {"role": "assistant", "content": [
                   {"type": "toolCall", "id": "call-1", "name": "read", "arguments": {"path": path}}]}}]
    file.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    calls = [chunk for chunk in omp.parse_session_chunks(file)[0] if chunk.kind == "tool_call"]
    assert len(calls) == 1 and calls[0].meta["tool_name"] == "read"
    return calls[0].text


def agy_texts(tmp_path, path):
    """``(the proposal AGY's transcript records, the call a hook is asked about)``."""
    file = make_agy(tmp_path, [agy_user(0, "go"), planner(1, "", [("view_file", {"AbsolutePath": path})])])
    proposals = [chunk for chunk in agy.parse_session_chunks(file)[0] if chunk.text.startswith("[proposed_tool_call:")]
    assert len(proposals) == 1
    hook_side = agy_hook_state.tool_call_text("agy", "view_file", {"AbsolutePath": path})
    return proposals[0].text, hook_side


@pytest.mark.parametrize("name,path", list(FILES.items()))
def test_the_claude_omp_and_agy_entries_equal_what_the_parsers_and_hooks_build(tmp_path, name, path):
    entries = tool_entries(name)
    assert entries["claude_read_tool"].text == claude_text(tmp_path / "claude", path)
    assert entries["omp_read_tool"].text == omp_text(tmp_path / "omp", path)
    proposal, hook_side = agy_texts(tmp_path / "agy", path)
    assert entries["agy_view_file_tool"].text == hook_side
    assert proposal == hook_side.replace("[tool_call:", "[proposed_tool_call:", 1)      # the transcript's own label
    # the one builder every writer shares renders the same text
    assert entries["claude_read_tool"].text == tool_call_text("claude", "Read", {"file_path": path})
    assert entries["omp_read_tool"].text == tool_call_text("omp", "read", {"path": path})


def test_the_grok_bot_fixture_read_call_has_the_text_of_the_omp_entry_up_to_case():
    fixture = next((ROOT / "fixtures" / "sample_grokbot_session").rglob("*.jsonl"))
    chunks = grok_parse.parse_session_chunks(fixture)[0]
    reads = [chunk for chunk in chunks if chunk.kind == "tool_call" and chunk.meta.get("tool_name") == "Read"]
    assert len(reads) == 1
    arguments = json.loads(reads[0].text.split("] ", 1)[1])
    assert list(arguments) == ["path"]                                          # no file_path key in this host
    rendered = tool_call_text("grokbot", "Read", {"path": "/work/.env"})
    omp_entry = tool_entries("read_secret_file")["omp_read_tool"].text
    assert rendered != omp_entry and rendered.lower() == omp_entry.lower()
    assert re.search(re.escape(omp_entry), rendered, re.IGNORECASE)              # the contract engine ignores case


def test_a_hooks_tool_call_text_is_the_same_text_a_regex_is_matched_against(tmp_path):
    # AGY's PreToolUse high-risk gate matches the saved regex against exactly the library's view_file text
    contracts = tmp_path / "contracts.json"

    def gate(regex):
        contracts.write_text(json.dumps({"contracts": [{"id": "no-env", "type": "must_not", "risk": "high",
                                                        "scope": "tool_call", "regex": regex}]}), encoding="utf-8")
        call = {"name": "view_file", "args": {"AbsolutePath": "/work/.env"}}
        return agy_hook_state.high_risk_match(str(contracts), AGY_SID, call)

    assert gate(MOTIVATING) is None                      # the shell-scoped regex does not see the tool call
    assert gate(r"\.env") == "no-env"                    # a regex that names the file does


# -------------------------------------------------------------------------------- the motivating case, end to end

def test_a_shell_scoped_dotenv_regex_is_told_about_the_file_read_tools():
    report = variants.analyze_pattern(MOTIVATING)
    entry = missed(report, "read_secret_file")
    assert report["status"] == "ok" and entry is not None
    assert entry["matched_canonical"] == ["cat .env", "head .env", "tail .env", "less .env"]
    listed = {item["id"]: item for item in entry["missed_variants"]}
    entries = tool_entries("read_secret_file")
    for host in HOSTS:                                   # listed under the default caps, one each
        item = listed[entries[host].id]
        assert item["text"] == entries[host].text
        assert item["platform"] == "tool_call" and item["technique"] == "file_read_tool" and item["verified"] is False
        assert item["preconditions"]
    assert "read_secret_file.more" in listed             # the standard spelling the regex does not name
    assert len(entry["missed_variants"]) == variants.MAX_MISSED_PER_FAMILY and entry["omitted"] > 0
    assert {item["technique"] for item in entry["missed_variants"]} >= {"file_read_tool", "standard"}


def test_a_regex_that_matches_the_tool_calls_is_told_nothing_about_them(uncapped):
    # matches only tool-call text: no standard spelling matches, so no family is selected at all
    assert variants.analyze_pattern(TOOL_REGEX) == {"status": "ok", "families": [], "families_omitted": 0}
    # names the file: selected through its shell spellings, and every tool call is matched
    for pattern in (r"\.env", r"(?s)\.env", r"(?s)(?:" + MOTIVATING + r")|\[tool_call:[^\]]+\].*\.env",
                    r"(?s)cat\s+\.env|\[tool_call:[^\]]+\].*\.env"):
        entry = missed(variants.analyze_pattern(pattern), "read_secret_file")
        assert entry is not None, pattern
        assert not [item for item in entry["missed_variants"] if item["platform"] == "tool_call"], pattern


@pytest.mark.parametrize("name", sorted(FILES))
def test_each_file_family_lists_the_tool_calls_for_a_shell_regex_and_none_for_a_file_regex(name, uncapped):
    shell_regex, file_regex = REGEXES[name]
    entry = missed(variants.analyze_pattern(shell_regex), name)
    assert entry is not None
    assert {item["id"] for item in entry["missed_variants"] if item["platform"] == "tool_call"} == {
        v.id for v in tool_entries(name).values()}
    entry = missed(variants.analyze_pattern(file_regex), name)             # a family nothing is missed in is not listed
    assert entry is None or not [i for i in entry["missed_variants"] if i["platform"] == "tool_call"]


def test_the_default_caps_still_list_the_three_tool_calls_for_each_file():
    for name, (shell_regex, _file_regex) in REGEXES.items():
        entry = missed(variants.analyze_pattern(shell_regex), name)
        assert len(entry["missed_variants"]) <= variants.MAX_MISSED_PER_FAMILY
        assert {item["id"] for item in entry["missed_variants"] if item["platform"] == "tool_call"} == {
            v.id for v in tool_entries(name).values()}, name


# ---------------------------------------------------------------------------- one family per file, and the prefix

def families_of(pattern: str) -> list[str]:
    return [entry["family"] for entry in variants.analyze_pattern(pattern)["families"]]


def test_a_regex_about_one_file_is_not_told_about_the_spellings_of_another_file():
    assert families_of(r"cat\s+~/\.ssh/id_rsa") == ["read_ssh_private_key"]
    assert families_of(r"cat\s+~/\.aws/credentials") == ["read_aws_credentials"]
    assert families_of(REGEXES["read_secret_file"][0]) == ["read_secret_file"]
    assert families_of(REGEXES["read_env_production"][0]) == ["read_env_production"]
    assert families_of(r"cat\s+\S*\.env\.production") == ["read_env_production"]       # not the .env family


def test_a_regex_naming_the_env_prefix_also_matches_the_production_spellings_and_is_reported_for_both():
    # Selection is a substring search and is not changed: `.env` is a prefix of `.env.production`, so a regex that
    # stops at `.env` matches the production standard spellings as well and is reported for both families (`.env` first).
    for pattern in (r"cat\s+\.env", r"(cat|head|tail|less)\s+\S*\.env", r"\bcat\b.*\.env"):
        assert families_of(pattern) == ["read_secret_file", "read_env_production"], pattern
    entry = missed(variants.analyze_pattern(r"cat\s+\.env"), "read_env_production")
    assert entry["matched_canonical"] == ["cat .env.production"]
    texts = [item["text"] for item in entry["missed_variants"] if item["platform"] == "tool_call"]
    assert texts and all(text.endswith('.env.production"}') for text in texts)       # its own file, not .env
    # a regex that ends the name stays with the .env family; nothing about the pattern's matching changed
    assert families_of(r"cat\s+\.env(?![\w.])") == ["read_secret_file"]
    assert families_of(r"cat\s+\.env\b") == ["read_secret_file", "read_env_production"]   # \b is a boundary before "."


def test_a_broad_shell_reader_regex_selects_every_file_family_within_the_family_cap():
    assert families_of(r"\bcat\b") == ["read_secret_file", "read_env_production", "read_ssh_private_key",
                                       "read_aws_credentials"]
    assert variants.MAX_FAMILIES_PER_CONTRACT == 4


# ----------------------------------------------------------------- upsert_contracts: informational, nothing changes

def read_call(path, call_id="read-1"):
    return {"type": "response_item", "payload": {"type": "function_call", "name": "Read", "call_id": call_id,
                                                 "arguments": {"file_path": path}}}


def test_the_report_names_the_gap_while_the_verdict_still_follows_the_saved_regex(session):
    engine, path = session
    shell_scoped = {"id": "no-dotenv", "type": "must_not", "scope": "tool_call", "regex": MOTIVATING,
                    "description": "Never print the secrets in .env."}
    response = engine.upsert_contracts([shell_scoped])
    assert response["ok"] is True and response["contracts"][0]["regex"] == MOTIVATING      # saved unchanged
    (contract,) = response["fragility"]["contracts"]
    entry = next(e for e in contract["families"] if e["family"] == "read_secret_file")
    assert {"[tool_call:Read] {\"file_path\": \"/work/.env\"}", "[tool_call:view_file] {\"AbsolutePath\": \"/work/.env\"}",
            "[tool_call:read] {\"path\": \"/work/.env\"}"} <= {i["text"] for i in entry["missed_variants"]}
    assert response["fragility"]["library_version"] == LIBRARY.version and response["fragility"]["verified"] is False
    # the gap is real: the file-read call is not a violation of the shell-scoped regex, and the report changed nothing
    write(path, [read_call("/work/.env")], "a")
    assert engine.audit_session(SID)["verdict"] == "clean"
    assert engine.list_contracts()["contracts"][0]["regex"] == MOTIVATING
    # a regex that names the file matches the same call and has nothing to report for it
    response = engine.upsert_contracts([{**shell_scoped, "id": "no-dotenv-file", "regex": r"\.env(?![\w.])"}])
    assert not [e for c in response["fragility"]["contracts"] for e in c["families"]
                if any(i["platform"] == "tool_call" for i in e["missed_variants"])]
    findings = {f["contract_id"]: f["verdict"] for f in engine.audit_session(SID)["findings"]}
    assert findings == {"no-dotenv": "clean", "no-dotenv-file": "violation"}


# ------------------------------------------------------------------------------- the authoring guidance sentence

def test_the_tool_description_and_the_readme_tell_the_author_about_file_read_tools():
    from self_directing_mcp import server
    tool = next(t for t in asyncio.run(server.mcp.list_tools()) if t.name == "upsert_contracts")
    description = " ".join(tool.description.split())
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    sentence = ("A contract about a file must also match the calls of the host's file-read tools "
                "(Claude Code Read, AGY view_file, OMP read), whose text is `[tool_call:<name>] {json}` and not a "
                "shell command, so name the file itself rather than only shell readers such as cat; "
                "the fragility report in the response lists such spellings.")
    assert description.count(sentence) == 1
    assert "A contract about a file must also match the calls of the host's file-read tools" in readme
    assert "`[tool_call:<name>] {json}` and not a shell command" in readme
