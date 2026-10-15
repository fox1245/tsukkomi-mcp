"""scripts/install_grok_hooks.py: one dedicated hook file in a temporary Grok home, and nothing else.

Synthetic data only. A planted ``~/.claude/settings.json``, ``config.toml`` and ``trusted_folders.toml`` prove the
installer never reads or changes them; no ``grok`` binary is run.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "install_grok_hooks.py"
HOOK = ROOT / "scripts" / "grok_hook.py"
EVENTS = ["SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PostToolUseFailure", "Stop"]
TOOL_EVENTS = {"PreToolUse", "PostToolUse", "PostToolUseFailure"}


def _load():
    spec = importlib.util.spec_from_file_location("install_grok_hooks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


installer = _load()


def run_cli(*args, home, cwd=None, env=None):
    environment = {k: v for k, v in os.environ.items() if k != "GROK_HOME"}
    environment.update(HOME=str(home), USERPROFILE=str(home), **(env or {}))
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=180,
                          env=environment, cwd=cwd or home)


@pytest.fixture
def home(tmp_path):
    """A fake user home with files the installer must never touch."""
    root = tmp_path / "userhome"
    (root / ".claude").mkdir(parents=True)
    (root / ".grok").mkdir()
    (root / ".claude" / "settings.json").write_text('{"hooks": {"Stop": []}, "model": "synthetic"}\n', encoding="utf-8")
    (root / ".grok" / "config.toml").write_text('[hooks]\nmodel = "synthetic"\n', encoding="utf-8")
    (root / ".grok" / "trusted_folders.toml").write_text('trusted = ["/work/proj"]\n', encoding="utf-8")
    return root


def snapshot(root: Path):
    """Every file under ``root`` with its content digest."""
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def hook_file(home: Path) -> Path:
    return home / ".grok" / "hooks" / "tsukkomi.json"


def apply(home, *extra, grok_home=None):
    result = run_cli("--apply", "--python", "/usr/bin/synthetic-python", *extra,
                     *(["--grok-home", str(grok_home)] if grok_home else []), home=home)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def owned(document):
    return [(event, handler) for event, groups in document.get("hooks", {}).items() for group in groups
            for handler in group["hooks"] if installer.is_owned(handler)]


def backups(path):
    return sorted(path.parent.glob(path.name + ".tsukkomi-backup-*"))


# --- print -----------------------------------------------------------------------------------------------------

def test_print_is_the_default_and_changes_nothing(home):
    before = snapshot(home)
    result = run_cli("--python", "/usr/bin/synthetic-python", home=home)
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert set(document) == {"hooks"} and list(document["hooks"]) == EVENTS
    assert snapshot(home) == before and not hook_file(home).exists()


def test_every_event_runs_the_script_with_the_mode_and_the_flags_named(home):
    result = run_cli("--python", "/usr/bin/synthetic-python", "--local-only", "--index-dir", str(home / "idx"),
                     "--mode", "enforced", home=home)
    hooks = json.loads(result.stdout)["hooks"]
    for event in EVENTS:
        (group,) = hooks[event]
        (handler,) = group["hooks"]
        assert handler["type"] == "command" and handler["timeout"] > 0 and installer.is_owned(handler)
        command = handler["command"]
        assert command.startswith("/usr/bin/synthetic-python ") and f"--event={event}" in command
        assert str(HOOK.resolve()) in command and "--local-only" in command
        assert f"--index-dir={home / 'idx'}" in command and command.endswith("--mode=enforced")
        if event in TOOL_EVENTS:
            assert group["matcher"] == "*"
        else:
            assert "matcher" not in group            # Grok ignores a matcher on Stop and UserPromptSubmit


def test_without_flags_the_command_names_only_the_event_and_the_mode(home):
    handler = json.loads(run_cli("--python", "py", home=home).stdout)["hooks"]["PreToolUse"][0]["hooks"][0]
    assert "--local-only" not in handler["command"] and "--index-dir" not in handler["command"]
    assert handler["command"].endswith("--event=PreToolUse --mode=confirmed_only")


def test_an_event_subset_and_unknown_events(home):
    result = run_cli("--python", "py", "--events", "PreToolUse,Stop", home=home)
    assert list(json.loads(result.stdout)["hooks"]) == ["PreToolUse", "Stop"]
    bad = run_cli("--python", "py", "--events", "SubagentStop", home=home)
    assert bad.returncode != 0 and "unknown event" in bad.stderr


def test_a_relative_index_dir_or_grok_home_is_refused(home):
    assert run_cli("--python", "py", "--index-dir", "relative", home=home).returncode != 0
    result = run_cli("--apply", "--grok-home", "relative", home=home)
    assert result.returncode == 1 and "absolute" in result.stderr


# --- apply -----------------------------------------------------------------------------------------------------

def test_apply_writes_one_file_and_nothing_else(home):
    before = snapshot(home)
    result = apply(home)
    assert result["written"] is True and result["backup"] is None and result["hook_file"] == str(hook_file(home).resolve())
    after = snapshot(home)
    assert set(after) - set(before) == {os.path.join(".grok", "hooks", "tsukkomi.json")}
    assert {name: digest for name, digest in after.items() if name in before} == before     # every planted file unchanged
    document = read(hook_file(home))
    assert set(document) == {"hooks"} and [event for event, _h in owned(document)] == EVENTS


def test_apply_twice_is_idempotent_and_writes_no_backup(home):
    apply(home)
    first = hook_file(home).read_bytes()
    again = apply(home)
    assert again == {"hook_file": str(hook_file(home).resolve()), "backup": None, "changes": [], "written": False}
    assert hook_file(home).read_bytes() == first and backups(hook_file(home)) == []


def test_a_changed_setting_updates_in_place_and_backs_up_only_then(home):
    apply(home, "--local-only")
    result = apply(home, "--local-only", "--mode", "advisory")
    assert result["written"] is True and result["backup"] and Path(result["backup"]).is_file()
    assert sorted(result["changes"]) == sorted(f"updated {event}" for event in EVENTS)
    assert all(handler["command"].endswith("--mode=advisory") for _e, handler in owned(read(hook_file(home))))
    assert len(owned(read(hook_file(home)))) == len(EVENTS)               # no duplicates
    assert len(backups(hook_file(home))) == 1
    assert json.loads(Path(result["backup"]).read_text(encoding="utf-8")) != read(hook_file(home))


def test_apply_keeps_a_handler_someone_else_added_to_the_same_file(home):
    target = hook_file(home)
    target.parent.mkdir()
    foreign = {"hooks": {"PreToolUse": [{"matcher": "run_terminal_command",
                                         "hooks": [{"type": "command", "command": "/opt/audit/pre.sh"}]}],
                         "Notification": [{"hooks": [{"type": "command", "command": "notify-send done"}]}]},
               "note": "mine"}
    target.write_text(json.dumps(foreign, indent=4), encoding="utf-8")
    apply(home)
    merged = read(target)
    assert merged["note"] == "mine" and merged["hooks"]["Notification"] == foreign["hooks"]["Notification"]
    assert merged["hooks"]["PreToolUse"][0] == foreign["hooks"]["PreToolUse"][0]          # order is kept
    assert len(owned(merged)) == len(EVENTS) and len(backups(target)) == 1
    assert target.read_text(encoding="utf-8").startswith('{\n    "hooks"')                    # the file's indent is kept


def test_a_malformed_file_is_an_error_and_is_never_overwritten(home):
    target = hook_file(home)
    target.parent.mkdir()
    for bad in ("{not json", "[]", '{"hooks": []}', '{"hooks": {"PreToolUse": "x"}}'):
        target.write_text(bad, encoding="utf-8")
        result = run_cli("--apply", "--python", "py", home=home)
        assert result.returncode == 1 and result.stderr.startswith("error:"), bad
        assert target.read_text(encoding="utf-8") == bad and backups(target) == []


def test_apply_without_the_script_is_an_error(home, tmp_path):
    result = run_cli("--apply", "--script", str(tmp_path / "missing.py"), "--python", "py", home=home)
    assert result.returncode == 1 and "hook script not found" in result.stderr and not hook_file(home).exists()


def test_the_grok_home_option_and_the_environment_variable(home, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    result = apply(home, grok_home=elsewhere)
    assert result["hook_file"] == str((elsewhere / "hooks" / "tsukkomi.json").resolve())
    assert not hook_file(home).exists() and (elsewhere / "hooks" / "tsukkomi.json").is_file()
    via_env = tmp_path / "via-env"
    done = run_cli("--apply", "--python", "py", home=home, env={"GROK_HOME": str(via_env)})
    assert done.returncode == 0 and (via_env / "hooks" / "tsukkomi.json").is_file()
    explicit = tmp_path / "explicit"
    done = run_cli("--apply", "--python", "py", "--grok-home", str(explicit), home=home, env={"GROK_HOME": str(via_env)})
    assert (explicit / "hooks" / "tsukkomi.json").is_file()                  # the option beats the variable


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privileges on Windows")
def test_a_symlinked_hook_file_is_written_through_not_replaced(home, tmp_path):
    target = tmp_path / "dotfiles" / "tsukkomi.json"
    target.parent.mkdir()
    target.write_text("{}\n", encoding="utf-8")
    hook_file(home).parent.mkdir()
    hook_file(home).symlink_to(target)
    apply(home)
    assert hook_file(home).is_symlink() and len(owned(read(target))) == len(EVENTS)


# --- uninstall -------------------------------------------------------------------------------------------------

def test_uninstall_removes_the_file_when_only_our_handlers_were_in_it(home):
    before = snapshot(home)
    apply(home)
    result = run_cli("--uninstall", home=home)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["written"] is True and out["backup"] is None and out["changes"][-1] == "deleted the file"
    assert not hook_file(home).exists()
    after = snapshot(home)
    assert after == before                       # nothing else was ever touched, nothing left behind
    again = json.loads(run_cli("--uninstall", home=home).stdout)
    assert again["written"] is False and again["changes"] == []


def test_uninstall_leaves_foreign_handlers_and_foreign_hook_files_alone(home):
    other = home / ".grok" / "hooks" / "audit.json"
    other.parent.mkdir()
    other.write_text('{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "/opt/stop.sh"}]}]}}\n',
                     encoding="utf-8")
    target = hook_file(home)
    target.write_text(json.dumps({"hooks": {"PreToolUse": [
        {"matcher": "*", "hooks": [{"type": "command", "command": "/opt/mine.sh"}]}]}}), encoding="utf-8")
    apply(home)
    planted = snapshot(home)
    result = json.loads(run_cli("--uninstall", home=home).stdout)
    assert result["written"] is True and result["backup"] and target.is_file()
    assert read(target) == {"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": "/opt/mine.sh"}]}]}}
    now = snapshot(home)
    assert now[os.path.join(".grok", "hooks", "audit.json")] == planted[os.path.join(".grok", "hooks", "audit.json")]
    for name in (".claude/settings.json", ".grok/config.toml", ".grok/trusted_folders.toml"):
        assert now[os.path.normpath(name)] == planted[os.path.normpath(name)]


def test_a_hook_file_of_another_name_is_never_read_or_written(home):
    other = home / ".grok" / "hooks" / "audit.json"
    other.parent.mkdir()
    ours_by_hand = {"hooks": {"Stop": [{"hooks": [{"type": "command",
                                                  "command": "python /x/scripts/grok_hook.py --event=Stop"}]}]}}
    other.write_text(json.dumps(ours_by_hand), encoding="utf-8")
    digest = hashlib.sha256(other.read_bytes()).hexdigest()
    apply(home)
    run_cli("--uninstall", home=home)
    assert hashlib.sha256(other.read_bytes()).hexdigest() == digest          # only tsukkomi.json is ever owned


# --- ownership recognition ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("command,expected", [
    ("/venv/bin/python /repo/scripts/grok_hook.py --event=PreToolUse --mode=advisory", True),
    ("python \"C:\\repo\\scripts\\grok_hook.py\" --event=Stop", True),
    ("'/odd path/python' '/odd path/scripts/grok_hook.py' --event=Stop", True),
    ("python /repo/scripts/claude_hook.py --event=PreToolUse", False),
    ("python /repo/scripts/grok_hook.py", False),
    ("python /repo/scripts/not_grok_hook.py --event=Stop", False),
    ("python /repo/other/grok_hook.py --event=Stop", False),
])
def test_ownership_is_recognised_by_the_command(command, expected):
    assert installer.is_owned({"type": "command", "command": command}) is expected
    assert installer.is_owned({"type": "http", "command": command}) is False
    assert installer.is_owned({"type": "command"}) is False and installer.is_owned("x") is False


def test_merge_is_pure_and_idempotent():
    desired = installer.desired_hooks("py", "/r/scripts/grok_hook.py")
    document = {"hooks": {"Notification": [{"hooks": [{"type": "command", "command": "x"}]}]}, "k": 1}
    frozen = copy.deepcopy(document)
    merged, changes = installer.merge_document(document, desired)
    assert document == frozen and len(changes) == len(EVENTS)
    again, changes = installer.merge_document(merged, desired)
    assert again == merged and changes == []


def test_events_no_longer_selected_lose_only_our_handlers():
    full = installer.desired_hooks("py", "/r/scripts/grok_hook.py")
    merged, _c = installer.merge_document({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "/opt/s.sh"}]}]}}, full)
    narrowed, changes = installer.merge_document(merged, installer.desired_hooks("py", "/r/scripts/grok_hook.py",
                                                                                ("PreToolUse",)))
    assert list(narrowed["hooks"]) == ["Stop", "PreToolUse"] or set(narrowed["hooks"]) == {"Stop", "PreToolUse"}
    assert narrowed["hooks"]["Stop"] == [{"hooks": [{"type": "command", "command": "/opt/s.sh"}]}]
    assert any("no longer selected" in change for change in changes)


# --- verify ----------------------------------------------------------------------------------------------------

def test_verify_runs_the_registered_command_on_synthetic_data_only(home, tmp_path):
    flags = ("--python", sys.executable, "--local-only", "--index-dir", str(tmp_path / "must-stay-untouched"))
    apply_result = run_cli("--apply", *flags, home=home)
    assert apply_result.returncode == 0, apply_result.stderr
    before = snapshot(home)
    done = run_cli("--verify", *flags, home=home)
    assert done.returncode == 0, done.stdout + done.stderr
    report = json.loads(done.stdout)
    assert report["ok"] is True and report["registered"] is True and set(report["events"].values()) == {"ok"}
    assert report["smoke"]["forbidden"] == {"exit_code": 0, "decision": "deny", "expected": "deny"}
    assert report["smoke"]["harmless"] == {"exit_code": 0, "decision": "allow", "expected": "allow"}
    assert snapshot(home) == before and not (tmp_path / "must-stay-untouched").exists()      # the real index is never opened


def test_verify_reports_a_missing_or_drifted_installation(home):
    flags = ("--python", sys.executable, "--no-run")
    missing = run_cli("--verify", *flags, home=home)
    assert missing.returncode == 1 and json.loads(missing.stdout)["registered"] is False
    assert set(json.loads(missing.stdout)["events"].values()) == {"missing"}
    assert run_cli("--apply", *flags[:2], home=home).returncode == 0
    drift = run_cli("--verify", *flags, "--mode", "enforced", home=home)
    report = json.loads(drift.stdout)
    assert drift.returncode == 1 and set(report["events"].values()) == {"differs"} and report["smoke"] == {"ran": False}
    assert run_cli("--verify", *flags, home=home).returncode == 0


def test_verify_fails_when_the_hook_does_not_deny(home, tmp_path):
    broken = tmp_path / "repo" / "scripts"
    broken.mkdir(parents=True)
    script = broken / "grok_hook.py"
    script.write_text("import sys, json\nprint(json.dumps({'decision': 'allow'}))\n", encoding="utf-8")
    run_cli("--apply", "--python", sys.executable, "--script", str(script), home=home)
    done = run_cli("--verify", "--python", sys.executable, "--script", str(script), home=home)
    report = json.loads(done.stdout)
    assert done.returncode == 1 and report["ok"] is False and report["smoke"]["forbidden"]["decision"] == "allow"


def test_the_installer_never_names_the_other_hosts_files_or_the_binary():
    source = SCRIPT.read_text(encoding="utf-8")
    for needle in ("settings.json", "config.toml", "trusted_folders", ".claude"):
        # mentioned only in prose that says it never touches them
        assert source.count(needle) <= 2 and all(
            "never" in line or "no " in line for line in source.splitlines() if needle in line), needle
    assert "shutil.which(\"grok\")" not in source and "'grok'" not in source
    assert "subprocess.run([\"grok\"" not in source
