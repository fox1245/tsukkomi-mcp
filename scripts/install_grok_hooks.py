#!/usr/bin/env python3
"""Install, update, verify or remove the tsukkomi Grok CLI hooks.

Default is ``--print``: show the hook file this installer would write, as JSON, and change nothing.
``--apply`` writes ONE dedicated file, ``<grok home>/hooks/tsukkomi.json`` (``--grok-home``, else ``GROK_HOME``, else
``~/.grok``), and nothing else: it never reads or writes ``~/.claude/settings.json``, ``<grok home>/config.toml``,
``trusted_folders.toml`` or any other hook file, and it never runs ``grok``. Grok loads every ``*.json`` of its
global hooks directory, always trusted (no folder trust needed), so the file takes effect in the next Grok session.
``--uninstall`` removes exactly what this installer owns from that file and deletes the file when nothing else is
left in it. ``--verify`` checks the installed file against what ``--apply`` would write and runs the registered
command once on synthetic data in a temporary directory (never the real index or sessions).

Handlers are recognised by their command (``.../scripts/grok_hook.py --event=...``), so a hand-added handler in the
same file survives. Before an existing file is changed it is copied to a timestamped backup next to it; an unchanged
result writes nothing. The file is replaced atomically. Standard library only.

Grok loads Claude Code's settings file by default. If the Claude hooks are installed there, Grok runs them on its own
events as well; ``scripts/claude_hook.py`` recognises a Grok payload and answers silently.
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
import subprocess
import sys
import tempfile

SCRIPT_NAME = "grok_hook.py"
HOOK_FILE = "tsukkomi.json"
MODES = ("confirmed_only", "advisory", "enforced")
# event -> (matcher or None, timeout seconds). Grok ignores a matcher on Stop and UserPromptSubmit, and a SessionStart
# matcher tests the start source; tool events keep "*" because reads can be restricted and tool names do not prove
# the absence of side effects.
EVENT_SPECS = {
    "SessionStart": (None, 30),
    "UserPromptSubmit": (None, 30),
    "PreToolUse": ("*", 60),
    "PostToolUse": ("*", 30),
    "PostToolUseFailure": ("*", 30),
    "Stop": (None, 120),
}
_OWNED = re.compile(r"(?:^|[\\/\"' ])scripts[\\/]" + re.escape(SCRIPT_NAME) + r"(?=[\"' ]|$)")
SMOKE_SESSION = "00000000-0000-4000-8000-0000000000c1"
SMOKE_RULE = {"id": "installer-smoke-no-rm", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf",
              "description": "installer smoke test"}


class InstallError(Exception):
    """A hook file that cannot be merged safely; nothing is written."""


# --- commands and the desired hook block ------------------------------------------------------------------

def _quote(argument: str) -> str:
    if os.name == "nt":
        return f'"{argument}"' if re.search(r'[\s"&|^()]', argument) else argument
    return shlex.quote(argument)


def hook_command(python: str, script: str, event: str, *, mode: str = "confirmed_only", local_only: bool = False,
                 index_dir: str | None = None) -> str:
    parts = [_quote(python), _quote(script), f"--event={event}"]
    if local_only:
        parts.append("--local-only")
    if index_dir:
        # Hooks never read a project .env, so the index is named on the command line (or in the user TOML).
        parts.append(_quote(f"--index-dir={index_dir}"))
    parts.append(f"--mode={mode}")
    return " ".join(parts)


def desired_hooks(python: str, script: str, events=tuple(EVENT_SPECS), *, mode: str = "confirmed_only",
                  local_only: bool = False, index_dir: str | None = None) -> dict:
    block: dict[str, list] = {}
    for event in events:
        matcher, timeout = EVENT_SPECS[event]
        handler = {"type": "command", "timeout": timeout,
                   "command": hook_command(python, script, event, mode=mode, local_only=local_only, index_dir=index_dir)}
        block[event] = [{"matcher": matcher, "hooks": [handler]} if matcher is not None else {"hooks": [handler]}]
    return block


def is_owned(handler) -> bool:
    return (isinstance(handler, dict) and handler.get("type") == "command"
            and isinstance(handler.get("command"), str) and "--event=" in handler["command"]
            and _OWNED.search(handler["command"]) is not None)


# --- merge and removal ------------------------------------------------------------------------------------

def _hooks_section(document: dict) -> dict:
    hooks = document.get("hooks", {})
    if not isinstance(hooks, dict):
        raise InstallError('"hooks" in the hook file is not an object')
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


def merge_document(document: dict, desired: dict) -> tuple[dict, list[str]]:
    """``(new document, changes)``; the input is never modified. Idempotent."""
    merged = copy.deepcopy(document)
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
    return merged, changes


def remove_owned(document: dict) -> tuple[dict, list[str]]:
    merged = copy.deepcopy(document)
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
    return merged, changes


# --- files ------------------------------------------------------------------------------------------------

def grok_home(explicit: Path | None) -> Path:
    """``--grok-home``, else ``GROK_HOME``, else ``~/.grok`` (the directory Grok itself uses)."""
    chosen = explicit if explicit is not None else Path(os.environ.get("GROK_HOME") or Path.home() / ".grok")
    chosen = chosen.expanduser()
    if not chosen.is_absolute():
        raise InstallError("the Grok home must be an absolute path")
    return chosen


def hook_file(home: Path) -> Path:
    return home / "hooks" / HOOK_FILE


def load_document(path: Path) -> tuple[dict, str]:
    """``(document, raw text)``; a missing file is an empty object, anything unparsable is an error."""
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


def write_document(path: Path, data: dict, raw: str) -> None:
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


def apply(path: Path, desired: dict, *, uninstall: bool = False) -> dict:
    target = Path(os.path.realpath(path))        # write through a symlinked dotfile, never replace it
    document, raw = load_document(target)
    existed = target.is_file()
    merged, changes = remove_owned(document) if uninstall else merge_document(document, desired)
    unchanged = {"hook_file": str(target), "backup": None, "changes": [], "written": False}
    if uninstall:
        if not existed or not changes:
            return unchanged
        if not merged:
            target.unlink()           # everything in the file was ours: there is nothing of anyone else's to back up
            return {**unchanged, "changes": changes + ["deleted the file"], "written": True}
    elif existed and merged == document:
        return unchanged
    saved = backup(target) if existed and raw.strip() else None
    write_document(target, merged, raw)
    return {"hook_file": str(target), "backup": str(saved) if saved else None, "changes": changes, "written": True}


# --- verify -----------------------------------------------------------------------------------------------

def _payload(event: str, session: str, transcript: Path, command: str | None) -> dict:
    native = {"SessionStart": "session_start", "PreToolUse": "pre_tool_use"}[event]
    base = {"hookEventName": native, "hook_event_name": event, "sessionId": session, "session_id": session,
            "cwd": "/work/proj", "workspaceRoot": "/work/proj", "permissionMode": "default",
            "permission_mode": "default"}
    if event == "PreToolUse":
        arguments = {"command": command}
        base.update(transcriptPath=str(transcript), transcript_path=str(transcript), toolName="run_terminal_command",
                    tool_name="run_terminal_command", toolUseId="call-0", tool_use_id="call-0", toolInput=arguments,
                    tool_input=arguments)
    return base


def smoke(python: str, script: str, *, mode: str) -> dict:
    """Run the registered command on synthetic data: a forbidden command must be denied and a harmless one allowed.

    Everything lives in a temporary directory that is removed afterwards: the index, the contracts, the sessions and
    HOME. The registered command's own index and sessions are never opened."""
    result: dict = {"ran": True}
    work = Path(tempfile.mkdtemp(prefix="tsukkomi-grok-verify-"))
    try:
        index, sessions, home = work / "index", work / "sessions", work / "home"
        for directory in (index, home):
            directory.mkdir()
        transcript = sessions / "%2Fwork%2Fproj" / SMOKE_SESSION / "updates.jsonl"
        transcript.parent.mkdir(parents=True)
        record = {"timestamp": 1, "method": "session/update", "params": {"sessionId": SMOKE_SESSION, "update": {
            "sessionUpdate": "user_message_chunk", "content": {"type": "text", "text": "verify"},
            "_meta": {"promptIndex": 0}}}}
        transcript.write_text(json.dumps(record) + "\n", encoding="utf-8")
        (index / "contracts.json").write_text(json.dumps({"contracts": [SMOKE_RULE]}), encoding="utf-8")
        environment = {key: value for key, value in os.environ.items()
                       if not key.upper().startswith(("SELF_DIRECT_", "GROK_", "TSUKKOMI_", "OPENROUTER_"))}
        environment.update(HOME=str(home), USERPROFILE=str(home), SELF_DIRECT_GROK_SESSIONS_DIR=str(sessions),
                           SELF_DIRECT_INDEX_DIR=str(index), SELF_DIRECT_LOCAL_ONLY="true")
        base = [python, script, "--event=PreToolUse", "--local-only", f"--index-dir={index}", f"--mode={mode}"]
        for label, command, expected in (("forbidden", "rm -rf build", "deny"), ("harmless", "echo ok", "allow")):
            done = subprocess.run(base, input=json.dumps(_payload("PreToolUse", SMOKE_SESSION, transcript, command)),
                                  capture_output=True, text=True, timeout=120, env=environment)
            try:
                decision = json.loads(done.stdout).get("decision")
            except ValueError:
                decision = None
            result[label] = {"exit_code": done.returncode, "decision": decision, "expected": expected}
            if done.returncode != 0 or decision != expected:
                result["ok"] = False
                result["stderr_first_line"] = (done.stderr.splitlines() or [""])[0][:300]
        result.setdefault("ok", True)
    except (OSError, subprocess.SubprocessError) as exc:
        result.update(ok=False, error=type(exc).__name__)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return result


def verify(path: Path, desired: dict, *, python: str, script: str, mode: str, run: bool = True) -> dict:
    document, _raw = load_document(Path(os.path.realpath(path)))
    hooks = _hooks_section(document)
    events: dict[str, str] = {}
    for event, wanted in desired.items():
        groups = _foreign_groups(hooks, event) or []
        slots = _owned_slots(groups)
        if not slots:
            events[event] = "missing"
        elif len(slots) > 1:
            events[event] = "duplicate"
        else:
            g, h = slots[0]
            events[event] = "ok" if groups[g]["hooks"][h] == wanted[0]["hooks"][0] and (
                groups[g].get("matcher") == wanted[0].get("matcher")) else "differs"
    report = {"hook_file": str(path), "exists": Path(path).is_file(), "events": events,
              "script_exists": Path(script).is_file(), "python_exists": Path(python).is_file()
              or shutil.which(python) is not None}
    report["registered"] = report["exists"] and all(state == "ok" for state in events.values())
    report["smoke"] = smoke(python, script, mode=mode) if run and report["script_exists"] and report["python_exists"] \
        else {"ran": False}
    ran = bool(run)        # --no-run checks the file and the paths only; a run that could not start is a failure
    report["ok"] = bool(report["registered"] and report["script_exists"] and report["python_exists"]
                        and (report["smoke"].get("ok", False) if ran else True))
    return report


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
                        help="show the hook file as JSON; change nothing (default)")
    action.add_argument("--apply", action="store_true", help="write <grok home>/hooks/tsukkomi.json")
    action.add_argument("--uninstall", action="store_true", help="remove what this installer wrote from that file")
    action.add_argument("--verify", action="store_true",
                        help="check the installed file and run the registered command once on synthetic data")
    parser.add_argument("--grok-home", type=Path,
                        help="Grok's home directory (default: GROK_HOME, else ~/.grok); hooks go in <home>/hooks")
    parser.add_argument("--python", default=sys.executable, help="interpreter the hook commands run (default: this one)")
    parser.add_argument("--script", type=Path, default=Path(__file__).resolve().with_name(SCRIPT_NAME),
                        help="path of scripts/grok_hook.py")
    parser.add_argument("--mode", choices=MODES, default="confirmed_only",
                        help="enforcement mode written into every command (--mode=...)")
    parser.add_argument("--events", type=_events, default=tuple(EVENT_SPECS),
                        help="comma-separated events to install (default: all)")
    parser.add_argument("--local-only", action="store_true",
                        help="run the hook engine local-only (no embedder or credentials); affects hooks only")
    parser.add_argument("--index-dir", type=Path,
                        help="absolute index directory the hooks use (shared with the MCP server whose contracts "
                             "apply); default: SELF_DIRECT_INDEX_DIR, the user TOML, then ~/.self_direct_index")
    parser.add_argument("--no-run", action="store_true", help="with --verify: check the file only, run nothing")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.index_dir is not None and not args.index_dir.is_absolute():
        parser.error("--index-dir must be an absolute path")
    script = str(args.script.resolve())
    index_dir = str(args.index_dir) if args.index_dir is not None else None
    desired = desired_hooks(args.python, script, args.events, mode=args.mode, local_only=args.local_only,
                            index_dir=index_dir)
    try:
        if not (args.apply or args.uninstall or args.verify):
            json.dump({"hooks": desired}, sys.stdout, indent=2, ensure_ascii=False)
            sys.stdout.write("\n")
            return 0
        path = hook_file(grok_home(args.grok_home))
        if args.verify:
            report = verify(path, desired, python=args.python, script=script, mode=args.mode, run=not args.no_run)
            json.dump(report, sys.stdout, indent=2)
            sys.stdout.write("\n")
            return 0 if report["ok"] else 1
        if args.apply and not Path(script).is_file():
            raise InstallError(f"hook script not found: {script}")
        result = apply(path, desired, uninstall=args.uninstall)
    except InstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
