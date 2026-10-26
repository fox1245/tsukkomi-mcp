"""The documentation of portable contracts says what the code does: tools, statuses, limits, settings, the example file."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from self_directing_mcp import contract_bundles as cb
from self_directing_mcp import server, settings_core

ROOT = Path(__file__).resolve().parents[1]
TOP_README = (ROOT / "README.md").read_text(encoding="utf-8")
README = (ROOT / "docs" / "reference.md").read_text(encoding="utf-8")
SECTION = README[README.index("#### Sharing contracts across devices"):README.index("### 2. Checklist with evidence gating")]


def test_every_registered_tool_is_in_the_readmes_tool_reference():
    reference = README[README.index("## MCP tools reference"):README.index("### Host-side GraphRAG extraction")]
    top_ref = TOP_README[TOP_README.index("## MCP tools reference"):TOP_README.index("## Documentation")]
    for tool in asyncio.run(server.mcp.list_tools()):
        assert f"`{tool.name}`" in reference and f"`{tool.name}`" in top_ref, tool.name


def test_the_status_table_lists_exactly_the_statuses_the_code_reports():
    rows = re.findall(r"^\| `(\w+)` \|", SECTION, flags=re.MULTILINE)
    assert sorted(rows) == sorted(cb.STATUSES)


def test_the_documented_limits_are_the_enforced_ones():
    assert cb.MAX_BUNDLE_BYTES == 1024 * 1024 and "1 MiB" in SECTION
    assert cb.MAX_BUNDLE_CONTRACTS == 200 and "200 contracts" in SECTION
    assert cb.DESCRIPTION_SHOWN == 200 and "cut to 200 characters" in SECTION
    assert cb.ID_MAX == 200 and "at most 200 characters" in SECTION
    assert cb.REGEX_MAX == 1000 and "at most 1,000 characters" in SECTION
    assert cb.SHORT_HASH == 12 and "12 hex characters" in SECTION
    assert cb.BUNDLE_SCHEMA in SECTION and cb.PROJECT_BUNDLE_LABEL in SECTION


def test_the_readme_names_every_device_bound_field_and_the_example_file_is_a_valid_bundle():
    for name in cb.DEVICE_BOUND_FIELDS:
        assert f"`{name}`" in SECTION, name
    block = re.search(r"```json\n(.*?)\n```", SECTION, flags=re.DOTALL).group(1)
    bundle = cb.parse_bundle(block.encode("utf-8"), kind="project", path_label=cb.PROJECT_BUNDLE_LABEL)
    assert [c.rule.id for c in bundle.contracts] == ["no-rm", "tests-ran"] and json.loads(block)["schema"] == cb.BUNDLE_SCHEMA


def test_the_documented_refusal_reasons_exist_in_the_code():
    source = (ROOT / "src" / "self_directing_mcp" / "contract_bundles.py").read_text(encoding="utf-8")
    for reason in ("symlink_leaves_project", "not_a_regular_file", "not_owned_by_user", "world_writable", "too_large",
                   "not_json", "invalid_bundle"):
        assert reason in SECTION and f'"{reason}"' in source, reason


def test_both_settings_are_documented_where_users_look_for_them():
    for name in ("SELF_DIRECT_PROJECT_CONTRACTS", "SELF_DIRECT_CONTRACT_BUNDLE_PATHS"):
        assert README.count(name) >= 2 and name in (ROOT / ".env.example").read_text(encoding="utf-8"), name
    assert "contract_bundle_paths" in (ROOT / "config.example.toml").read_text(encoding="utf-8")
    assert settings_core.DEFAULT_PROJECT_CONTRACTS == "candidates" and "`candidates`, the default" in SECTION


def test_the_trust_rule_is_stated_in_the_changelog_the_agent_guide_and_the_skill():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release_notes = changelog[changelog.index("## 0.8.0"):changelog.index("## 0.7.0")]
    for fragment in ("list_contract_candidates", "adopt_contracts", "export_contracts", "never weakens or revokes",
                     "user message of this device's index", "ContractRepository", "SELF_DIRECT_PROJECT_CONTRACTS"):
        assert fragment in release_notes, fragment
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "## Portable contracts" in agents and "untrusted" in agents and "never weakens or revokes" in agents
    assert "adopt_contracts" in (ROOT / "skills" / "self-directing-mcp-setup" / "SKILL.md").read_text(encoding="utf-8")


def test_no_private_names_or_absolute_paths_in_what_this_change_documents():
    for text in (SECTION, (ROOT / "AGENTS.md").read_text(encoding="utf-8")):
        for needle in ("/home/", "/Users/", "/mnt/", "C:\\Users"):
            assert needle not in text, needle
