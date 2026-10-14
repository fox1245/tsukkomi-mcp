#!/usr/bin/env python3
"""Install, update or remove the tsukkomi Claude Code hooks in a settings.json.

Default is ``--print``: show the ``hooks`` block and ``env`` this installer would add, as JSON, and change
nothing. ``--apply --settings PATH`` merges them into that file; ``--uninstall --settings PATH`` removes exactly
what this installer owns. There is no default settings path: the file is always named explicitly.

Merging preserves unrelated hooks (including their order), unrelated settings and other handlers sharing a
matcher group. Owned handlers are recognised by their command (``.../scripts/claude_hook.py --event=...``).
Before any write the existing file is copied to a timestamped backup next to it; an unchanged result writes
nothing. The file is replaced atomically. Standard library only.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tempfile

SCRIPT_NAME = "claude_hook.py"
ENV_KEY = "SELF_DIRECT_CLAUDE_ENFORCEMENT"
MODES = ("confirmed_only", "advisory", "enforced")
# event -> (matcher or None, timeout seconds). Tool events keep "*": reads can be restricted and tool names
# do not prove the absence of side effects.
EVENT_SPECS = {
    "SessionStart": (None, 30),
    "UserPromptSubmit": (None, 30),
    "PreToolUse": ("*", 60),
    "PostToolUse": ("*", 30),
    "PostToolUseFailure": ("*", 30),
    "Stop": (None, 120),
    "SubagentStop": (None, 120),
}
_OWNED = re.compile(r"(?:^|[\\/\"' ])scripts[\\/]" + re.escape(SCRIPT_NAME) + r"(?=[\"' ]|$)")


class InstallError(Exception):
    """A settings file that cannot be merged safely; nothing is written."""


# --- commands and the desired hook block ------------------------------------------------------------------

def _quote(argument: str) -> str:
    if os.name == "nt":
        return f'"{argument}"' if re.search(r'[\s"&|^()]', argument) else argument
    return shlex.quote(argument)


def hook_command(python: str, script: str, event: str, *, local_only: bool = False,
                 index_dir: str | None = None) -> str:
    parts = [_quote(python), _quote(script), f"--event={event}"]
    if local_only:
        parts.append("--local-only")
    if index_dir:
        # Hooks never read a project .env, so the index is named on the command line (or in the user TOML).
        parts.append(_quote(f"--index-dir={index_dir}"))
    return " ".join(parts)


def desired_hooks(python: str, script: str, events=tuple(EVENT_SPECS), *, local_only: bool = False,
                  index_dir: str | None = None) -> dict:
    block: dict[str, list] = {}
    for event in events:
        matcher, timeout = EVENT_SPECS[event]
        handler = {"type": "command", "timeout": timeout,
                   "command": hook_command(python, script, event, local_only=local_only, index_dir=index_dir)}
        block[event] = [{"matcher": matcher, "hooks": [handler]} if matcher is not None else {"hooks": [handler]}]
    return block


def is_owned(handler) -> bool:
    return (isinstance(handler, dict) and handler.get("type") == "command"
            and isinstance(handler.get("command"), str) and "--event=" in handler["command"]
            and _OWNED.search(handler["command"]) is not None)


# --- merge and removal ------------------------------------------------------------------------------------

def _hooks_section(settings: dict) -> dict:
    hooks = settings.get("hooks", {})
    if not isinstance(hooks, dict):
        raise InstallError('"hooks" in the settings file is not an object')
    return hooks


def _well_formed(groups) -> bool:
    return isinstance(groups, list) and all(
        isinstance(group, dict) and isinstance(group.get("hooks", []), list) for group in groups)


def _groups(hooks: dict, event: str) -> list:
    """The matcher groups of an event we manage; a malformed list is an error, never a guess."""
    groups = hooks.get(event, [])
    if not _well_formed(groups):
        raise InstallError(f'"hooks.{event}" is not a list of matcher groups; refusing to guess')
    return groups


def _foreign_groups(hooks: dict, event: str) -> list | None:
    """The groups of an event we do not manage, or None if they are not ours to interpret."""
    groups = hooks.get(event)
    return groups if _well_formed(groups) else None


def _owned_slots(groups: list) -> list[tuple[int, int]]:
    return [(g, h) for g, group in enumerate(groups) for h, handler in enumerate(group.get("hooks", []))
            if is_owned(handler)]


def _strip(groups: list, slots: list[tuple[int, int]]) -> None:
    """Delete owned handlers (highest index first) and the matcher groups that only they kept alive."""
    for g, h in sorted(slots, reverse=True):
        del groups[g]["hooks"][h]
        if not groups[g]["hooks"] and set(groups[g]) <= {"matcher", "hooks"}:
            del groups[g]


def merge_settings(settings: dict, desired: dict, env: dict[str, str]) -> tuple[dict, list[str]]:
    """``(new settings, changes)``; the input is never modified. Idempotent."""
    merged = copy.deepcopy(settings)
    changes: list[str] = []
    hooks = _hooks_section(merged)
    for event in list(hooks):
        if event in desired:
            continue
        groups = _foreign_groups(hooks, event)   # events we no longer manage lose their owned handlers
        slots = _owned_slots(groups) if groups is not None else []
        if slots:
            _strip(groups, slots)
            changes.append(f"removed {event} (no longer selected)")
            if not groups:
                del hooks[event]
    for event, wanted in desired.items():
        groups = _groups(hooks, event)
        handler = wanted[0]["hooks"][0]
        slots = _owned_slots(groups)
        if not slots:
            groups.append(copy.deepcopy(wanted[0]))
            hooks[event] = groups
            changes.append(f"added {event}")
            continue
        g, h = slots[0]
        if groups[g]["hooks"][h] != handler:
            groups[g]["hooks"][h] = copy.deepcopy(handler)
            changes.append(f"updated {event}")
        if len(slots) > 1:
            _strip(groups, slots[1:])
            changes.append(f"removed {len(slots) - 1} duplicate {event} handler(s)")
    if hooks:
        merged["hooks"] = hooks
    if env:
        section = merged.get("env", {})
        if not isinstance(section, dict):
            raise InstallError('"env" in the settings file is not an object')
        for key, value in env.items():
            if section.get(key) != value:
                changes.append(f"set env {key}={value}")
            section[key] = value
        merged["env"] = section
    return merged, changes


def remove_settings(settings: dict) -> tuple[dict, list[str]]:
    merged = copy.deepcopy(settings)
    changes: list[str] = []
    hooks = _hooks_section(merged)
    for event in list(hooks):
        groups = _foreign_groups(hooks, event)
        slots = _owned_slots(groups) if groups is not None else []
        if slots:
            _strip(groups, slots)
            changes.append(f"removed {len(slots)} {event} handler(s)")
            if not groups:
                del hooks[event]
    if changes and not hooks:
        merged.pop("hooks", None)
    env = merged.get("env")
    if isinstance(env, dict) and ENV_KEY in env:
        del env[ENV_KEY]
        changes.append(f"removed env {ENV_KEY}")
        if not env:
            del merged["env"]
    return merged, changes


# --- files ------------------------------------------------------------------------------------------------

def load_settings(path: Path) -> tuple[dict, str]:
    """``(settings, raw text)``; a missing file is an empty object, anything unparsable is an error."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}, ""
    except (OSError, UnicodeError) as exc:
        raise InstallError(f"cannot read {path}: {type(exc).__name__}") from exc
    if not raw.strip():
        return {}, raw
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise InstallError(f"{path} is not valid JSON; refusing to modify it") from exc
    if not isinstance(data, dict):
        raise InstallError(f"{path} does not contain a JSON object; refusing to modify it")
    return data, raw


def _indent(raw: str) -> int:
    found = re.search(r'^\{[ \t]*\r?\n( +)"', raw)
    return len(found.group(1)) if found else 2


def backup(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.name}.tsukkomi-backup-{stamp}")
    counter = 1
    while target.exists():
        target = path.with_name(f"{path.name}.tsukkomi-backup-{stamp}-{counter}")
        counter += 1
    shutil.copy2(path, target)
    return target


def write_settings(path: Path, data: dict, raw: str) -> None:
    text = json.dumps(data, indent=_indent(raw), ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        if path.exists():
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def apply(settings_path: Path, desired: dict, env: dict[str, str], *, uninstall: bool = False) -> dict:
    target = Path(os.path.realpath(settings_path))      # write through a symlinked dotfile, never replace it
    settings, raw = load_settings(target)
    merged, changes = remove_settings(settings) if uninstall else merge_settings(settings, desired, env)
    if merged == settings:
        return {"settings": str(target), "backup": None, "changes": [], "written": False}
    saved = backup(target) if target.exists() else None
    write_settings(target, merged, raw)
    return {"settings": str(target), "backup": str(saved) if saved else None, "changes": changes, "written": True}


# --- command line -----------------------------------------------------------------------------------------

def _events(value: str) -> tuple[str, ...]:
    chosen = tuple(dict.fromkeys(name.strip() for name in value.split(",") if name.strip()))
    unknown = [name for name in chosen if name not in EVENT_SPECS]
    if unknown or not chosen:
        raise argparse.ArgumentTypeError(f"unknown event(s) {unknown or value!r}; choose from {', '.join(EVENT_SPECS)}")
    return chosen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--print", dest="show", action="store_true",
                        help="show the hooks block and env as JSON; change nothing (default)")
    action.add_argument("--apply", action="store_true", help="merge into --settings")
    action.add_argument("--uninstall", action="store_true", help="remove the installed hooks from --settings")
    parser.add_argument("--settings", type=Path, help="settings.json to modify (required with --apply/--uninstall)")
    parser.add_argument("--python", default=sys.executable, help="interpreter the hook commands run (default: this one)")
    parser.add_argument("--script", type=Path, default=Path(__file__).resolve().with_name(SCRIPT_NAME),
                        help="path of scripts/claude_hook.py")
    parser.add_argument("--mode", choices=MODES, default="confirmed_only", help=f"value of {ENV_KEY}")
    parser.add_argument("--events", type=_events, default=tuple(EVENT_SPECS),
                        help="comma-separated events to install (default: all)")
    parser.add_argument("--local-only", action="store_true",
                        help="run the hook engine local-only (no embedder or credentials); affects hooks only")
    parser.add_argument("--index-dir", type=Path,
                        help="absolute index directory the hooks use (shared with the MCP server whose contracts "
                             "apply); default: SELF_DIRECT_INDEX_DIR, the user TOML, then ~/.self_direct_index")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.apply or args.uninstall) and args.settings is None:
        parser.error("--settings is required with --apply and --uninstall")
    if args.index_dir is not None and not args.index_dir.is_absolute():
        parser.error("--index-dir must be an absolute path")
    script = str(args.script.resolve())
    index_dir = str(args.index_dir) if args.index_dir is not None else None
    desired = desired_hooks(args.python, script, args.events, local_only=args.local_only, index_dir=index_dir)
    env = {ENV_KEY: args.mode}
    try:
        if not (args.apply or args.uninstall):
            json.dump({"hooks": desired, "env": env}, sys.stdout, indent=2, ensure_ascii=False)
            sys.stdout.write("\n")
            return 0
        if args.apply and not Path(script).is_file():
            raise InstallError(f"hook script not found: {script}")
        result = apply(args.settings, desired, env, uninstall=args.uninstall)
    except InstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
