"""Contract fragility report (issue #15, B4): equivalent spellings of a risky action that a contract regex misses.

The report is informational: it never blocks a registration, edits a regex or changes a verdict, every library entry
is an unverified candidate, and hook paths never import the library."""
from __future__ import annotations

import ast
import asyncio
import copy
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from importlib import resources
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import pytest

from self_directing_mcp import codex_hooks
from self_directing_mcp.audit import patterns, variants
from self_directing_mcp.schemas import ContractRule
from self_directing_mcp.security.mask import mask_secrets
from test_reliability import SID, call, message, result, session, write  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "self_directing_mcp"
REQUIRED_FAMILIES = {"recursive_delete", "force_push", "discard_worktree", "world_writable", "remote_script_exec",
                     "read_secret_file", "stage_all", "db_cli_access", "skip_tests", "tls_verify_off"}
REQUIRED_FIELDS = {"id", "text", "platform", "technique", "preconditions", "verified"}
PATHOLOGICAL = r"^(.|.)+\d$"       # exponential on any long text without a trailing digit, even for the regex engine
LIBRARY = variants.load_library()
FAMILIES = {family.family: family for family in LIBRARY.families}


def raw_library() -> dict:
    resource = resources.files("self_directing_mcp").joinpath("resources", "contract_variants.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def exact(text: str) -> str:
    return r"\A(?:" + re.escape(text) + r")\Z"


@pytest.fixture
def tight(monkeypatch):
    monkeypatch.setattr(patterns, "SEARCH_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(patterns, "CONTRACT_BUDGET_SEC", 0.3)


@pytest.fixture
def uncapped(monkeypatch):
    monkeypatch.setattr(variants, "MAX_MISSED_PER_FAMILY", 1000)


def missed_ids(report_family: dict) -> set[str]:
    return {item["id"] for item in report_family["missed_variants"]}


# ----------------------------------------------------------------------------------------------- the library

def test_the_library_loads_from_package_data_and_is_schema_valid():
    raw = raw_library()
    library = variants.parse_library(raw)
    assert library.version == raw["library_version"] == variants.load_library().version
    assert re.fullmatch(r"\d+\.\d+\.\d+", library.version)
    assert {family.family for family in library.families} >= REQUIRED_FAMILIES
    for family in library.families:
        assert family.intent and family.effect
        assert sum(v.canonical for v in family.variants) >= 1
        assert sum(not v.canonical for v in family.variants) >= 8, family.family


def test_the_package_data_pattern_includes_the_library():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns_ = data["tool"]["setuptools"]["package-data"]["self_directing_mcp"]
    assert any(fnmatch.fnmatch("resources/contract_variants.json", pattern) for pattern in patterns_)


def test_every_variant_carries_the_required_fields():
    raw = raw_library()
    assert raw["verification"] and raw["matching"]
    for family in raw["families"]:
        assert {"family", "intent", "effect", "variants"} <= set(family)
        for item in family["variants"]:
            assert REQUIRED_FIELDS <= set(item), item.get("id")
            assert item["text"].strip() and isinstance(item["preconditions"], list)
            assert item["platform"] in raw["platforms"] and item["technique"] in raw["techniques"]
            assert isinstance(item["verified"], bool)
    for family in LIBRARY.families:
        for variant in family.variants:
            assert variant.effect and variant.id.startswith(family.family + ".")


def test_every_variant_is_an_unverified_candidate():
    assert all(not v.verified for family in LIBRARY.families for v in family.variants)
    assert LIBRARY.all_verified is False


def test_variant_texts_survive_response_masking_unchanged():
    for family in LIBRARY.families:
        for variant in family.variants:
            assert mask_secrets(variant.text) == variant.text, variant.id


def _mutations():
    def drop_verified(raw): del raw["families"][0]["variants"][0]["verified"]
    def bad_platform(raw): raw["families"][0]["variants"][1]["platform"] = "amiga"
    def bad_technique(raw): raw["families"][0]["variants"][1]["technique"] = "magic"
    def duplicate_id(raw): raw["families"][0]["variants"][1]["id"] = raw["families"][0]["variants"][0]["id"]
    def unprefixed_id(raw): raw["families"][0]["variants"][1]["id"] = "other.slug"
    def no_canonical(raw):
        for item in raw["families"][0]["variants"]:
            item.pop("canonical", None)
    def bad_version(raw): raw["library_version"] = "v1"
    def bad_schema(raw): raw["schema_version"] = 2
    def bad_preconditions(raw): raw["families"][0]["variants"][1]["preconditions"] = "none"
    def empty_text(raw): raw["families"][0]["variants"][1]["text"] = " "
    def repeated_text(raw):
        raw["families"][0]["variants"][1]["text"] = raw["families"][0]["variants"][2]["text"]
    def duplicate_family(raw): raw["families"].append(copy.deepcopy(raw["families"][0]))
    return [drop_verified, bad_platform, bad_technique, duplicate_id, unprefixed_id, no_canonical, bad_version,
            bad_schema, bad_preconditions, empty_text, repeated_text, duplicate_family]


@pytest.mark.parametrize("mutate", _mutations(), ids=lambda fn: fn.__name__)
def test_a_malformed_library_is_rejected(mutate):
    raw = raw_library()
    mutate(raw)
    with pytest.raises(ValueError, match="variant library"):
        variants.parse_library(raw)


def test_canonical_spellings_are_unique_across_the_library():
    texts = [v.text for family in LIBRARY.families for v in family.variants if v.canonical]
    assert len(texts) == len(set(texts))


# ------------------------------------------------------------------------------------- selection and matching

CASES = [(family.family, v.text) for family in LIBRARY.families for v in family.variants if v.canonical]


@pytest.mark.parametrize("family_name,canonical", CASES)
def test_a_regex_that_matches_only_one_canonical_form_reports_every_other_spelling(family_name, canonical, uncapped):
    family = FAMILIES[family_name]
    report = variants.analyze_pattern(exact(canonical))
    assert report["status"] == "ok" and report["families_omitted"] == 0
    assert [entry["family"] for entry in report["families"]] == [family_name]      # no other family is "about" it
    entry = report["families"][0]
    expected = {v.id for v in family.variants if v.text != canonical}
    assert entry["matched_canonical"] == [canonical]
    assert missed_ids(entry) == expected
    assert entry["missed_total"] == len(expected) and entry["omitted"] == 0
    assert entry["intent"] == family.intent and entry["effect"] == family.effect
    for item in entry["missed_variants"]:
        assert item["verified"] is False and item["platform"] in LIBRARY.platforms
        assert item["technique"] in LIBRARY.techniques


@pytest.mark.parametrize("family_name,canonical", CASES)
def test_the_default_caps_list_a_bounded_subset_and_count_the_rest(family_name, canonical):
    family = FAMILIES[family_name]
    entry = variants.analyze_pattern(exact(canonical))["families"][0]
    expected = {v.id for v in family.variants if v.text != canonical}
    listed = missed_ids(entry)
    assert listed <= expected
    assert len(listed) == min(len(expected), variants.MAX_MISSED_PER_FAMILY)
    assert entry["missed_total"] == len(expected) and entry["omitted"] == len(expected) - len(listed)


HEADLINE = [
    # regex, family, ids it misses, ids it matches (slug after the family)
    (r"rm\s+-rf", "recursive_delete",
     {"rm_fr", "split_flags", "long_options", "abbreviated_long", "continuation_flags", "python_rmtree", "find_delete"},
     {"rm_rf", "busybox", "absolute_path", "verbose_cluster", "option_terminator", "dot_slash_target"}),
    (r"^\s*rm\s+-rf", "recursive_delete", {"busybox", "absolute_path", "sudo", "python_rmtree"}, {"rm_rf"}),
    (r"git\s+push\b.*--force\b", "force_push",
     {"short_flag", "refspec_plus", "global_option", "python_subprocess", "continuation", "short_cluster"},
     {"long_flag", "flag_last", "lease"}),
    (r"chmod\s+777", "world_writable",
     {"symbolic_a_rwx", "mode_0777", "mode_666", "python_chmod", "recursive_flag", "continuation"},
     {"mode_777", "busybox"}),
    (r"git\s+reset\s+--hard", "discard_worktree",
     {"checkout_force_head", "restore_worktree", "abbreviated", "option_before", "global_option", "read_tree"},
     {"reset_hard", "reset_hard_head"}),
    (r"curl\b.*\|\s*(ba)?sh\b", "remote_script_exec",
     {"wget_sh", "command_substitution", "process_substitution", "two_step", "powershell_iex_alias", "continuation"},
     {"curl_sh", "curl_bash", "shell_args"}),
    (r"cat\s+\.env", "read_secret_file",
     {"sed_range", "head", "python_read", "stdin_redirect", "dot_slash", "quoted"}, {"cat", "busybox"}),
    (r"git\s+add\s+(-A|--all)\b", "stage_all",
     {"add_dot", "glob_star", "pathspec_top", "global_option", "cluster", "abbreviated"}, {"add_a", "add_all"}),
    (r"curl\s+(-k|--insecure)\b", "tls_verify_off",
     {"curl_cluster", "curl_option_last", "wget_no_check", "python_requests", "git_ssl_config"},
     {"curl_k", "curl_insecure"}),
    (r"\bpsql\b", "db_cli_access",
     {"mysql", "sqlite3", "mariadb", "sqlite_python", "pgcli"}, {"psql", "psql_uri", "psql_docker", "psql_continuation"}),
    (r"--no-verify\b", "skip_tests",
     {"commit_n", "commit_cluster", "hooks_path", "husky_env", "mvn_skip_tests"},
     {"commit_no_verify", "push_no_verify", "commit_option_last", "global_option", "continuation"}),
]


@pytest.mark.parametrize("pattern,family_name,missed,matched", HEADLINE, ids=[case[1] for case in HEADLINE])
def test_typical_contract_regexes_miss_the_documented_spellings(pattern, family_name, missed, matched, uncapped):
    report = variants.analyze_pattern(pattern)
    entry = next(e for e in report["families"] if e["family"] == family_name)
    listed = {i.removeprefix(family_name + ".") for i in missed_ids(entry)}
    assert missed <= listed
    assert not (matched & listed)


def test_a_broad_regex_reports_no_family():
    for pattern in ("(?s).", r"[\s\S]", r"\S"):
        assert variants.analyze_pattern(pattern) == {"status": "ok", "families": [], "families_omitted": 0}


def test_a_regex_listing_every_library_spelling_reports_no_family():
    every = "|".join(re.escape(v.text) for family in LIBRARY.families for v in family.variants)
    assert variants.analyze_pattern(every)["families"] == []


@pytest.mark.parametrize("pattern", [r"^terraform\s+destroy\b", r"kubectl\s+delete", r"DROP\s+TABLE", r"pip\s+install",
                                     "VERIFIED", r"\bdeploy\b"])
def test_an_unrelated_regex_reports_no_family(pattern):
    assert variants.analyze_pattern(pattern) == {"status": "ok", "families": [], "families_omitted": 0}


def test_matching_uses_the_contract_regex_engine_semantics(uncapped):
    # case-insensitive, like the audit
    entry = variants.analyze_pattern(r"\ARM -RF BUILD\Z")["families"][0]
    assert entry["family"] == "recursive_delete" and entry["matched_canonical"] == ["rm -rf build"]
    # multiline: ^ and $ work per line inside a variant, so the backslash-newline spelling is matched by its second line
    entry = variants.analyze_pattern(r"\Arm -rf build\Z|^-rf build$")["families"][0]
    assert "recursive_delete.continuation_flags" not in missed_ids(entry)
    assert "recursive_delete.split_flags" in missed_ids(entry)
    # the second line alone does not select a family: no canonical form matches
    assert variants.analyze_pattern(r"^-rf build$")["families"] == []


# ----------------------------------------------------------------------------- bounds, timeouts, time budget

def test_the_report_is_bounded():
    selector = "|".join(exact(v.text) for family in LIBRARY.families for v in family.variants if v.canonical)
    report = variants.analyze_pattern(selector)
    assert report["status"] == "ok"
    assert len(report["families"]) == variants.MAX_FAMILIES_PER_CONTRACT
    assert report["families_omitted"] == len(REQUIRED_FAMILIES) - variants.MAX_FAMILIES_PER_CONTRACT
    for entry in report["families"]:
        assert len(entry["missed_variants"]) <= variants.MAX_MISSED_PER_FAMILY
        assert entry["omitted"] == entry["missed_total"] - len(entry["missed_variants"])
    assert len(json.dumps(report)) < 20_000


def test_a_capped_list_shows_different_techniques_first():
    entry = variants.analyze_pattern(exact("rm -rf build"))["families"][0]
    assert entry["omitted"] > 0
    techniques = [item["technique"] for item in entry["missed_variants"]]
    assert len(set(techniques)) >= variants.MAX_MISSED_PER_FAMILY - 1
    assert [item["id"] for item in entry["missed_variants"]] == [
        v.id for v in FAMILIES["recursive_delete"].variants if v.id in missed_ids(entry)]       # library order


def test_the_number_of_analyzed_contracts_is_bounded():
    rules = [ContractRule(id=f"no-rm-{index}", type="must_not", scope="tool_call", regex=r"rm\s+-rf")
             for index in range(variants.MAX_CONTRACTS + 3)]
    report = variants.fragility_report(rules)
    assert report["analyzed"] == variants.MAX_CONTRACTS and report["contracts_not_analyzed"] == 3
    assert len(report["contracts"]) == variants.MAX_CONTRACTS


def test_a_pathological_regex_is_reported_as_regex_timeout_within_the_limit(tight):
    started = time.monotonic()
    report = variants.analyze_pattern(PATHOLOGICAL)
    assert time.monotonic() - started < 2
    assert report == {"status": "regex_timeout", "families": [], "families_omitted": 0}


def test_one_slow_contract_does_not_hide_the_others(tight):
    rules = [ContractRule(id="slow", type="must_not", scope="tool_call", regex=PATHOLOGICAL),
             ContractRule(id="no-rm", type="must_not", scope="tool_call", regex=r"rm\s+-rf")]
    started = time.monotonic()
    report = variants.fragility_report(rules)
    assert time.monotonic() - started < 3
    by_id = {entry["contract_id"]: entry for entry in report["contracts"]}
    assert by_id["slow"]["status"] == "regex_timeout" and by_id["slow"]["families"] == []
    assert by_id["no-rm"]["status"] == "ok" and by_id["no-rm"]["families"][0]["family"] == "recursive_delete"


def test_the_report_time_budget_is_explicit(monkeypatch):
    monkeypatch.setattr(variants, "REPORT_BUDGET_SEC", 0.0)
    rules = [ContractRule(id="no-rm", type="must_not", scope="tool_call", regex=r"rm\s+-rf")]
    report = variants.fragility_report(rules)
    assert report["contracts"] == [{"contract_id": "no-rm", "field": "regex", "status": "skipped_time_budget",
                                    "families": [], "families_omitted": 0}]


def test_the_shipped_high_risk_presets_can_be_analyzed():
    presets = json.loads((SRC / "resources" / "high_risk_presets.json").read_text(encoding="utf-8"))["contracts"]
    rules = [ContractRule.model_validate(item) for item in presets]
    report = variants.fragility_report(rules)
    assert report["analyzed"] == sum(1 for rule in rules if rule.regex)
    assert all(entry["status"] == "ok" for entry in report["contracts"])


# ----------------------------------------------------------------------------------------- upsert_contracts

def upsert(engine, *contracts):
    return engine.upsert_contracts(list(contracts))


NO_RM = {"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf",
         "description": "Never delete user files."}


def test_upsert_reports_the_missed_spellings_and_stores_the_contract_unchanged(session):
    engine, _path = session
    response = upsert(engine, NO_RM)
    assert response["ok"] is True and response["counts"] == {"upserted": 1, "store": 1}
    stored = response["contracts"][0]
    assert stored["regex"] == NO_RM["regex"] and stored["revision"] == 1
    assert stored["description"] == NO_RM["description"] and stored["enabled"] is True
    assert engine.list_contracts()["contracts"] == response["contracts"]
    fragility = response["fragility"]
    assert fragility["library_version"] == LIBRARY.version
    assert fragility["informational"] is True and fragility["verified"] is False
    assert "unverified" in fragility["note"] and "informational" in fragility["note"].lower()
    assert fragility["analyzed"] == 1
    (entry,) = fragility["contracts"]
    assert entry["contract_id"] == "no-rm" and entry["field"] == "regex" and entry["status"] == "ok"
    family = entry["families"][0]
    assert family["family"] == "recursive_delete" and family["matched_canonical"] == ["rm -rf build"]
    assert family["missed_total"] > len(family["missed_variants"]) == variants.MAX_MISSED_PER_FAMILY
    assert family["omitted"] == family["missed_total"] - len(family["missed_variants"])
    assert all(item["verified"] is False for item in family["missed_variants"])
    json.dumps(response)                                                       # the response stays serializable


def test_the_report_never_changes_a_verdict(session):
    engine, path = session
    write(path, [message("cleaning up"), call("rm -r -f build", "c1"), result("c1")], "a")
    response = upsert(engine, NO_RM)
    missed = missed_ids(response["fragility"]["contracts"][0]["families"][0])
    assert "recursive_delete.split_flags" in missed          # the report says the regex misses `rm -r -f build` ...
    assert engine.audit_session(SID)["verdict"] == "clean"    # ... and the verdict still follows the regex alone
    write(path, [call("rm -rf build", "c2"), result("c2")], "a")
    assert engine.audit_session(SID)["verdict"] == "violation"
    assert engine.list_contracts()["contracts"][0]["regex"] == NO_RM["regex"]     # never edited or broadened


def test_contracts_without_a_matching_action_pattern_report_nothing(session):
    engine, _path = session
    response = upsert(
        engine,
        {"id": "describe", "type": "must_not", "description": "no unreviewed deploys"},
        {"id": "verify", "type": "must", "scope": "message", "regex": "VERIFIED"},
        {"id": "no-destroy", "type": "must_not", "scope": "tool_call", "regex": r"terraform\s+destroy"})
    assert response["ok"] is True and response["counts"]["upserted"] == 3
    assert response["fragility"]["analyzed"] == 1 and response["fragility"]["contracts"] == []


def test_a_gated_action_of_an_obligation_is_analyzed_through_before_regex(session):
    engine, _path = session
    response = upsert(engine, {"id": "tests-first", "type": "must", "scope": "tool_call", "regex": "pytest",
                               "before_regex": r"git\s+push\s+--force"})
    (entry,) = response["fragility"]["contracts"]
    assert entry["contract_id"] == "tests-first" and entry["field"] == "before_regex"
    assert entry["families"][0]["family"] == "force_push"


def test_a_pathological_regex_is_explicit_in_the_response_and_still_registers(session, tight):
    engine, _path = session
    started = time.monotonic()
    response = upsert(engine, {"id": "slow", "type": "must_not", "scope": "tool_call", "regex": PATHOLOGICAL})
    assert time.monotonic() - started < 3
    assert response["ok"] is True and response["contracts"][0]["regex"] == PATHOLOGICAL
    (entry,) = response["fragility"]["contracts"]
    assert entry["contract_id"] == "slow" and entry["status"] == "regex_timeout" and entry["families"] == []
    assert engine.list_contracts()["contracts"][0]["id"] == "slow"


def test_a_failing_report_never_blocks_the_registration(session, monkeypatch):
    engine, _path = session

    def broken(_rules):
        raise RuntimeError("library unreadable")

    monkeypatch.setattr(variants, "fragility_report", broken)
    response = upsert(engine, NO_RM)
    assert response["ok"] is True and response["contracts"][0]["id"] == "no-rm"
    assert response["fragility"] == {"informational": True, "status": "unavailable", "error": "RuntimeError"}
    assert [c["id"] for c in engine.list_contracts()["contracts"]] == ["no-rm"]


def test_an_invalid_regex_is_still_rejected_before_any_report(session):
    engine, _path = session
    with pytest.raises(Exception):
        upsert(engine, {"id": "bad", "type": "must_not", "scope": "tool_call", "regex": "(unclosed"})
    assert engine.list_contracts()["contracts"] == []


# ---------------------------------------------------------------------------------------------- hook paths

def test_only_upsert_contracts_imports_the_library():
    importers = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"audit\.variants|audit\s+import\s+[^\n]*\bvariants\b|from\s+\.variants|from\s+\.\s+import\s+variants", text):
            importers.append(path.relative_to(SRC).as_posix())
    assert importers == ["engine.py"]
    tree = ast.parse((SRC / "engine.py").read_text(encoding="utf-8"))
    assert not [node for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom)) and "variants" in ast.dump(node)]   # not module-level
    method = next(node for cls in tree.body if isinstance(cls, ast.ClassDef)
                  for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "upsert_contracts")
    assert any(isinstance(node, ast.ImportFrom) and node.module == "self_directing_mcp.audit.variants"
               for node in ast.walk(method))


def test_importing_the_hook_modules_and_the_engine_does_not_load_the_library():
    code = ("import sys\n"
            "import self_directing_mcp.engine, self_directing_mcp.codex_hooks, self_directing_mcp.claude_hooks\n"
            "import self_directing_mcp.agy_hooks, self_directing_mcp.hook_policy, self_directing_mcp.server\n"
            "import self_directing_mcp.audit.runner, self_directing_mcp.audit.incremental\n"
            "assert 'self_directing_mcp.audit.variants' not in sys.modules, 'library imported by a hook path'\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr


def test_running_the_codex_hooks_does_not_load_the_library(session, monkeypatch):
    engine, path = session
    write(path, [message("hello")], "a")
    upsert(engine, NO_RM)                                              # imports the library for the report
    assert "self_directing_mcp.audit.variants" in sys.modules
    monkeypatch.delitem(sys.modules, "self_directing_mcp.audit.variants")
    prompt = codex_hooks.handle_hook(engine, "UserPromptSubmit", SID, str(path), None, None,
                                     prompt="Never run rm -rf on the user's files.")
    pre = codex_hooks.handle_hook(engine, "PreToolUse", SID, str(path), "exec_command", {"cmd": "rm -rf build"})
    stop = codex_hooks.handle_hook(engine, "Stop", SID, str(path), None, None)
    assert "violation" in pre["hookSpecificOutput"]["additionalContext"]
    assert isinstance(prompt, dict) and isinstance(stop, dict)
    assert "self_directing_mcp.audit.variants" not in sys.modules


# ---------------------------------------------------------------------------------------------- MCP server

def _server_env(tmp_path: Path) -> dict[str, str]:
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("SELF_DIRECT_") and k not in ("OPENROUTER_API_KEY", "CODEX_SESSIONS_DIR")}
    env.update(SELF_DIRECT_USE_FAKE_EMBEDDER="true", SELF_DIRECT_DENSE_BACKEND="numpy",
               SELF_DIRECT_CODEX_SESSIONS_DIR=str(sessions), SELF_DIRECT_INDEX_DIR=str(tmp_path / "index"),
               PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    return env


def test_the_mcp_tool_returns_the_report_and_documents_it(tmp_path):
    async def scenario():
        params = StdioServerParameters(command=sys.executable, args=["-m", "self_directing_mcp"],
                                       env=_server_env(tmp_path), cwd=str(tmp_path))
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                tool = next(t for t in (await client.list_tools()).tools if t.name == "upsert_contracts")
                assert "fragility" in tool.description and "unverified" in tool.description
                done = await client.call_tool("upsert_contracts", {"contracts": [NO_RM]})
                assert not done.isError
                body = json.loads(done.content[0].text)
                assert body["ok"] is True and body["contracts"][0]["regex"] == NO_RM["regex"]
                fragility = body["fragility"]
                assert fragility["library_version"] == LIBRARY.version and fragility["verified"] is False
                family = fragility["contracts"][0]["families"][0]
                assert family["family"] == "recursive_delete"
                assert any(item["text"] == "rm --recursive --force build" or item["technique"] == "option_split"
                           for item in family["missed_variants"])
    asyncio.run(asyncio.wait_for(scenario(), timeout=60))
