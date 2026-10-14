"""Lifecycle hooks never read a ``.env`` from the working directory.

Hosts run hooks inside the project being edited, so such a file is content the audited agent can write. If the
hooks honoured it, the agent could point them at an empty index or an advisory mode and silence its own checks.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_directing_mcp import agy_hooks, claude_hooks
from self_directing_mcp.config import Settings, hook_settings
from self_directing_mcp.engine import SelfDirectEngine

SID = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"
NO_RM = {"id": "no-rm", "type": "must_not", "scope": "tool_call", "provider": "claude", "regex": r"rm\s+-rf",
         "description": "never rm -rf"}


@pytest.fixture
def planted(tmp_path, monkeypatch):
    """A user index at the default location, and a project ``.env`` that redirects every hook setting."""
    home, project, decoy = tmp_path / "home", tmp_path / "project", tmp_path / "decoy"
    for folder in (home, project, decoy):
        folder.mkdir()
    for name in ("SELF_DIRECT_INDEX_DIR", "SELF_DIRECT_CONTRACTS_PATH", "SELF_DIRECT_CONFIG_FILE",
                 "SELF_DIRECT_CLAUDE_ENFORCEMENT", "SELF_DIRECT_AGY_ENFORCEMENT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("SELF_DIRECT_LOCAL_ONLY", "true")
    projects = tmp_path / "projects"
    monkeypatch.setenv("SELF_DIRECT_CLAUDE_PROJECTS_DIR", str(projects))
    (project / ".env").write_text(
        f"SELF_DIRECT_INDEX_DIR={decoy}\nSELF_DIRECT_CONTRACTS_PATH={decoy / 'contracts.json'}\n"
        "SELF_DIRECT_CLAUDE_ENFORCEMENT=advisory\nSELF_DIRECT_AGY_ENFORCEMENT=advisory\n", encoding="utf-8")
    monkeypatch.chdir(project)
    return home, project, decoy, projects


def test_the_project_env_file_is_ignored_by_hook_settings_but_still_read_by_the_server(planted):
    home, _project, decoy, _projects = planted
    hooked = hook_settings()
    assert hooked.index_dir != decoy and hooked.resolve_contracts_path().parent != decoy
    assert (hooked.claude_enforcement, hooked.agy_enforcement) == ("confirmed_only", "confirmed_only")
    assert Settings().index_dir == decoy          # unchanged server behaviour; hooks are the exception


def test_environment_variables_still_configure_hooks(planted, monkeypatch, tmp_path):
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(tmp_path / "chosen"))
    monkeypatch.setenv("SELF_DIRECT_CLAUDE_ENFORCEMENT", "enforced")
    hooked = hook_settings()
    assert hooked.index_dir == (tmp_path / "chosen").resolve() and hooked.claude_enforcement == "enforced"


@pytest.mark.parametrize("module", [agy_hooks, claude_hooks])
def test_both_adapters_build_their_engine_from_hook_settings(planted, module):
    _home, _project, decoy, _projects = planted
    engine = module._new_engine()
    try:
        assert engine.settings.index_dir != decoy
    finally:
        engine.close()


def test_a_planted_env_file_cannot_turn_a_confirmed_violation_into_an_allow(planted):
    home, _project, _decoy, projects = planted
    user_index = hook_settings().index_dir
    seed = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=user_index,
                                     claude_projects_dir=projects, openrouter_api_key_file=None))
    try:
        seed.upsert_contracts([NO_RM])
    finally:
        seed.close()
    transcript = projects / "-synthetic-project" / f"{SID}.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(json.dumps({
        "type": "user", "sessionId": SID, "uuid": "u1", "parentUuid": None, "timestamp": "2026-10-14T00:00:00Z",
        "isSidechain": False, "cwd": "/synthetic/project", "origin": {"kind": "human"},
        "message": {"role": "user", "content": "hi"}}) + "\n", encoding="utf-8")
    response = claude_hooks.run_hook("PreToolUse", {
        "session_id": SID, "transcript_path": str(transcript), "cwd": "/synthetic/project",
        "hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "rm -rf build"}})
    assert response["hookSpecificOutput"]["permissionDecision"] == "deny", response
