"""scripts/install_claude_hooks.py: print, merge, backup, update and uninstall on temporary settings files only."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "install_claude_hooks.py"
HOOK = ROOT / "scripts" / "claude_hook.py"
EVENTS = ["SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PostToolUseFailure", "Stop", "SubagentStop"]
TOOL_EVENTS = {"PreToolUse", "PostToolUse", "PostToolUseFailure"}


def _load():
    spec = importlib.util.spec_from_file_location("install_claude_hooks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


installer = _load()

USER_SETTINGS = {
    "model": "synthetic-model",
    "permissions": {"allow": ["Bash(ls:*)"], "deny": ["Read(./secrets/**)"]},
    "env": {"UNRELATED": "keep"},
    "hooks": {
        "PreToolUse": [
            {"matcher": "Bash", "hooks": [{"type": "command", "command": "/opt/audit/pre.sh", "timeout": 5}]},
            {"matcher": "Write|Edit", "hooks": [{"type": "command", "command": "/opt/lint/pre.sh"}]},
        ],
        "Notification": [{"hooks": [{"type": "command", "command": "notify-send done"}]}],
        "Stop": [{"hooks": [{"type": "command", "command": "/opt/stop-a.sh"},
                            {"type": "command", "command": "/opt/stop-b.sh"}]}],
    },
}


def run_cli(*args, home=None, cwd=None):
    environment = {**os.environ, "HOME": str(home or cwd or ROOT), "USERPROFILE": str(home or cwd or ROOT)}
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=30,
                          env=environment, cwd=cwd)


def apply(path, *extra):
    result = run_cli("--apply", "--settings", str(path), "--python", "/usr/bin/synthetic-python", *extra,
                     cwd=path.parent if path.parent.exists() else path.parent.parent)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def owned(settings):
    return [(event, handler) for event, groups in settings.get("hooks", {}).items() for group in groups
            for handler in group["hooks"] if installer.is_owned(handler)]


def backups(path):
    return sorted(path.parent.glob(path.name + ".tsukkomi-backup-*"))


@pytest.fixture
def settings_path(tmp_path):
    return tmp_path / "claude" / "settings.json"


@pytest.fixture
def user_file(settings_path):
    settings_path.parent.mkdir()
    settings_path.write_text(json.dumps(USER_SETTINGS, indent=2) + "\n", encoding="utf-8")
    return settings_path


# --- --print ----------------------------------------------------------------------------------------------

def test_print_is_the_default_shows_hooks_and_env_and_touches_nothing(tmp_path):
    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    result = run_cli(home=home, cwd=work)
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert set(document) == {"hooks", "env"}
    assert document["env"] == {"SELF_DIRECT_CLAUDE_ENFORCEMENT": "confirmed_only"}
    assert list(document["hooks"]) == EVENTS
    for event, groups in document["hooks"].items():
        [group] = groups
        [handler] = group["hooks"]
        assert handler["type"] == "command" and handler["timeout"] > 0
        assert handler["command"].endswith(f"--event={event}") and str(HOOK) in handler["command"]
        assert ("matcher" in group and group["matcher"] == "*") == (event in TOOL_EVENTS)
    assert list(home.iterdir()) == [] and list(work.iterdir()) == []              # no ~/.claude, no files at all
    assert run_cli("--print", home=home, cwd=work).stdout == result.stdout


def test_print_options_select_events_mode_python_and_local_only():
    result = run_cli("--events", "PreToolUse, Stop", "--mode", "advisory", "--python", "/opt/py 3/python",
                     "--local-only")
    document = json.loads(result.stdout)
    assert list(document["hooks"]) == ["PreToolUse", "Stop"]
    assert document["env"] == {"SELF_DIRECT_CLAUDE_ENFORCEMENT": "advisory"}
    command = document["hooks"]["Stop"][0]["hooks"][0]["command"]
    quoted = '"/opt/py 3/python" ' if os.name == "nt" else "'/opt/py 3/python' "
    assert command.startswith(quoted) and command.endswith("--event=Stop --local-only")
    assert run_cli("--events", "Nope").returncode == 2
    assert run_cli("--mode", "bogus").returncode == 2


def test_apply_and_uninstall_require_an_explicit_settings_path(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    for flag in ("--apply", "--uninstall"):
        result = run_cli(flag, cwd=work)
        assert result.returncode == 2 and "--settings" in result.stderr
    assert list(work.iterdir()) == []
    assert run_cli("--apply", "--uninstall", "--settings", str(tmp_path / "s.json")).returncode == 2


def test_printed_commands_run_the_real_hook_and_fail_open_on_bad_input(tmp_path):
    document = json.loads(run_cli("--python", sys.executable, "--local-only").stdout)
    for event in ("PreToolUse", "Stop", "UserPromptSubmit"):
        command = document["hooks"][event][0]["hooks"][0]["command"]
        environment = {**os.environ, "SELF_DIRECT_INDEX_DIR": str(tmp_path / "index"),
                       "SELF_DIRECT_CONFIG_FILE": str(tmp_path / "none.toml")}
        result = subprocess.run(command, shell=True, input="not json", capture_output=True, text=True, timeout=30,
                                cwd=tmp_path, env=environment)
        assert result.returncode == 0, result.stderr
        assert "not a compliance verdict" in json.loads(result.stdout)["systemMessage"]


# --- --apply ----------------------------------------------------------------------------------------------

def test_apply_creates_a_missing_settings_file_without_a_backup_and_is_idempotent(settings_path):
    first = apply(settings_path)
    assert first["written"] is True and first["backup"] is None
    settings = read(settings_path)
    assert [event for event, _ in owned(settings)] == EVENTS
    assert settings["env"] == {"SELF_DIRECT_CLAUDE_ENFORCEMENT": "confirmed_only"}
    before = settings_path.read_bytes()
    second = apply(settings_path)
    assert second == {"settings": str(settings_path), "backup": None, "changes": [], "written": False}
    assert settings_path.read_bytes() == before and backups(settings_path) == []


def test_apply_preserves_unrelated_hooks_settings_and_order_and_makes_a_backup(user_file):
    original = user_file.read_bytes()
    result = apply(user_file)
    [backup] = backups(user_file)
    assert result["backup"] == str(backup) and backup.read_bytes() == original
    merged = read(user_file)
    for key in ("model", "permissions"):
        assert merged[key] == USER_SETTINGS[key]
    assert merged["env"]["UNRELATED"] == "keep" and merged["env"]["SELF_DIRECT_CLAUDE_ENFORCEMENT"] == "confirmed_only"
    hooks = merged["hooks"]
    assert hooks["PreToolUse"][:2] == USER_SETTINGS["hooks"]["PreToolUse"]             # their order and content
    assert hooks["Notification"] == USER_SETTINGS["hooks"]["Notification"]
    assert hooks["Stop"][0] == USER_SETTINGS["hooks"]["Stop"][0]
    assert len(hooks["PreToolUse"]) == 3 and installer.is_owned(hooks["PreToolUse"][2]["hooks"][0])
    assert [event for event, _ in owned(merged)] == ["PreToolUse", "Stop", "SessionStart", "UserPromptSubmit",
                                                     "PostToolUse", "PostToolUseFailure", "SubagentStop"]
    assert list(merged) == ["model", "permissions", "env", "hooks"]                    # top-level order kept
    assert not any(installer.is_owned(h) for groups in USER_SETTINGS["hooks"].values()
                   for g in groups for h in g["hooks"])


def test_apply_twice_makes_one_backup_per_real_change(user_file):
    apply(user_file)
    assert len(backups(user_file)) == 1
    apply(user_file)
    assert len(backups(user_file)) == 1
    apply(user_file, "--mode", "advisory")                           # an actual change: new backup
    assert len(backups(user_file)) == 2
    assert read(user_file)["env"]["SELF_DIRECT_CLAUDE_ENFORCEMENT"] == "advisory"


def test_reapply_updates_owned_handlers_in_place_without_duplicates(user_file):
    apply(user_file)
    before = read(user_file)
    result = run_cli("--apply", "--settings", str(user_file), "--python", "/other/python", "--local-only",
                     cwd=user_file.parent)
    assert result.returncode == 0, result.stderr
    after = read(user_file)
    assert [e for e, _ in owned(after)] == [e for e, _ in owned(before)]
    assert all(h["command"].startswith("/other/python ") and h["command"].endswith("--local-only")
               for _, h in owned(after))
    assert after["hooks"]["PreToolUse"][:2] == USER_SETTINGS["hooks"]["PreToolUse"]


def test_owned_handler_sharing_a_group_with_a_foreign_one_is_updated_without_touching_the_foreign_one(settings_path):
    settings_path.parent.mkdir()
    foreign = {"type": "command", "command": "/opt/other.sh", "timeout": 9}
    old = {"type": "command", "command": f"/old/python {HOOK} --event=Stop", "timeout": 1}
    settings_path.write_text(json.dumps({"hooks": {"Stop": [{"hooks": [foreign, old]}]}}))
    apply(settings_path)
    [group] = read(settings_path)["hooks"]["Stop"]
    assert group["hooks"][0] == foreign and group["hooks"][1]["command"].startswith("/usr/bin/synthetic-python ")


def test_duplicate_owned_handlers_collapse_to_one(settings_path):
    settings_path.parent.mkdir()
    one = {"type": "command", "command": f"python {HOOK} --event=Stop"}
    settings_path.write_text(json.dumps({"hooks": {"Stop": [{"hooks": [one]}, {"hooks": [dict(one)]}]}}))
    result = apply(settings_path)
    assert any("duplicate Stop" in change for change in result["changes"])
    assert len([h for e, h in owned(read(settings_path)) if e == "Stop"]) == 1
    assert len(read(settings_path)["hooks"]["Stop"]) == 1


def test_narrowing_the_event_selection_removes_owned_handlers_of_unselected_events(user_file):
    apply(user_file)
    apply(user_file, "--events", "PreToolUse,Stop")
    merged = read(user_file)
    assert [event for event, _ in owned(merged)] == ["PreToolUse", "Stop"]
    assert "SessionStart" not in merged["hooks"] and "Notification" in merged["hooks"]


def test_a_similarly_named_foreign_hook_is_not_ours(user_file):
    lookalike = {"type": "command", "command": "python /home/u/claude_hook.py --event=Stop"}   # no scripts/ dir
    other = {"type": "command", "command": f"python {HOOK}-copy --event=Stop"}
    assert not installer.is_owned(lookalike) and not installer.is_owned(other)
    assert not installer.is_owned({"type": "command", "command": f"python {HOOK}"})              # no --event
    assert not installer.is_owned({"type": "prompt", "command": f"python {HOOK} --event=Stop"})
    assert installer.is_owned({"type": "command", "command": f'"C:\\Program Files\\x\\scripts\\claude_hook.py" --event=Stop'})


def test_indentation_trailing_newline_permissions_and_symlinks_are_respected(tmp_path):
    real = tmp_path / "dotfiles" / "settings.json"
    real.parent.mkdir()
    real.write_text(json.dumps({"model": "m"}, indent=4), encoding="utf-8")       # no trailing newline, 4 spaces
    real.chmod(0o600)
    link = tmp_path / "claude" / "settings.json"
    link.parent.mkdir()
    link.symlink_to(real)
    apply(link)
    assert link.is_symlink() and os.path.realpath(link) == str(real)
    text = real.read_text(encoding="utf-8")
    assert text.startswith('{\n    "model": "m"') and text.endswith("}\n")
    assert (real.stat().st_mode & 0o777) == 0o600
    assert len(list(real.parent.glob("*.tmp"))) == 0 and len(backups(real)) == 1


def test_unrelated_malformed_hook_entries_are_left_alone_but_a_malformed_managed_event_is_refused(settings_path):
    settings_path.parent.mkdir()
    settings_path.write_text(json.dumps({"hooks": {"Notification": "custom-format", "Stop": [{"hooks": []}]}}))
    apply(settings_path)
    assert read(settings_path)["hooks"]["Notification"] == "custom-format"
    settings_path.write_text(json.dumps({"hooks": {"PreToolUse": "not a list"}}))
    before, known = settings_path.read_bytes(), backups(settings_path)
    result = run_cli("--apply", "--settings", str(settings_path), cwd=settings_path.parent)
    assert result.returncode == 1 and "hooks.PreToolUse" in result.stderr
    assert settings_path.read_bytes() == before and backups(settings_path) == known


@pytest.mark.parametrize("content,message", [("{not json", "not valid JSON"), ("[1, 2]", "JSON object"),
                                              ('{"hooks": []}', '"hooks"'), ('{"env": "x"}', '"env"')])
def test_unusable_settings_are_refused_untouched_with_no_backup(settings_path, content, message):
    settings_path.parent.mkdir()
    settings_path.write_text(content)
    for flag in ("--apply", "--uninstall"):
        result = run_cli(flag, "--settings", str(settings_path), cwd=settings_path.parent)
        if flag == "--uninstall" and message == '"env"':
            continue                                           # nothing of ours can live in a non-object env
        assert result.returncode == 1 and message in result.stderr, (flag, result.stderr)
    assert settings_path.read_text() == content and backups(settings_path) == []


def test_apply_refuses_when_the_hook_script_is_missing(user_file, tmp_path):
    result = run_cli("--apply", "--settings", str(user_file), "--script", str(tmp_path / "gone" / "claude_hook.py"),
                     cwd=tmp_path)
    assert result.returncode == 1 and "not found" in result.stderr and backups(user_file) == []


# --- --uninstall ------------------------------------------------------------------------------------------

def test_uninstall_restores_the_original_settings_exactly(user_file):
    original = read(user_file)
    apply(user_file)
    result = run_cli("--uninstall", "--settings", str(user_file), cwd=user_file.parent)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["written"] is True and summary["backup"] and Path(summary["backup"]).exists()
    assert read(user_file) == original
    assert owned(read(user_file)) == []


def test_uninstall_removes_only_what_this_installer_owns(settings_path):
    settings_path.parent.mkdir()
    foreign = {"type": "command", "command": "/opt/keep.sh"}
    settings_path.write_text(json.dumps({"env": {"SELF_DIRECT_CLAUDE_ENFORCEMENT": "advisory", "KEEP": "1"},
                                         "hooks": {"Stop": [{"hooks": [foreign]}], "Other": [{"hooks": [foreign]}]}}))
    apply(settings_path)
    run_cli("--uninstall", "--settings", str(settings_path), cwd=settings_path.parent)
    assert read(settings_path) == {"env": {"KEEP": "1"}, "hooks": {"Stop": [{"hooks": [foreign]}],
                                                                  "Other": [{"hooks": [foreign]}]}}


def test_uninstall_of_a_file_with_nothing_installed_writes_and_backs_up_nothing(user_file):
    before = user_file.read_bytes()
    result = run_cli("--uninstall", "--settings", str(user_file), cwd=user_file.parent)
    assert json.loads(result.stdout) == {"settings": str(user_file), "backup": None, "changes": [], "written": False}
    assert user_file.read_bytes() == before and backups(user_file) == []


def test_uninstall_leaves_no_empty_scaffolding_and_does_not_create_a_missing_file(settings_path):
    settings_path.parent.mkdir()
    apply(settings_path)
    run_cli("--uninstall", "--settings", str(settings_path), cwd=settings_path.parent)
    assert read(settings_path) == {}
    missing = settings_path.with_name("absent.json")
    result = run_cli("--uninstall", "--settings", str(missing), cwd=settings_path.parent)
    assert result.returncode == 0 and not missing.exists()


# --- pure functions never mutate their input --------------------------------------------------------------

def test_merge_and_remove_do_not_modify_their_arguments():
    desired = installer.desired_hooks("python", str(HOOK))
    settings = copy.deepcopy(USER_SETTINGS)
    merged, _ = installer.merge_settings(settings, desired, {installer.ENV_KEY: "confirmed_only"})
    assert settings == USER_SETTINGS and merged != settings
    stripped, _ = installer.remove_settings(merged)
    assert stripped == USER_SETTINGS and merged["hooks"]["Stop"]


def test_a_matcher_group_with_keys_this_installer_does_not_know_survives_removal_of_its_handler(settings_path):
    settings_path.parent.mkdir()
    ours = {"type": "command", "command": f"python {HOOK} --event=Stop"}
    settings_path.write_text(json.dumps({"hooks": {"Stop": [{"note": "user metadata", "hooks": [ours]}]}}))
    run_cli("--uninstall", "--settings", str(settings_path), cwd=settings_path.parent)
    assert read(settings_path) == {"hooks": {"Stop": [{"note": "user metadata", "hooks": []}]}}
