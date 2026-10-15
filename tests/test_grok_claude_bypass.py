"""Grok CLI loads Claude Code's settings file, so an installed Claude hook is also run on Grok's events.

``scripts/claude_hook.py`` must stay silent on a Grok payload (nothing printed, nothing written, exit code 0) instead of
rejecting its transcript on every tool call, and must still behave exactly as before for a real Claude Code payload.
Synthetic data only; the recorded Grok payloads come from ``tests/fixtures/grok``.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

import pytest

from self_directing_mcp import claude_hook_state, claude_hooks, grok_ids
from grok_support import FIXTURES, SID, hook_payload

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "claude_hook.py"
CLAUDE_SID = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"
EVENTS = ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PostToolUseFailure", "Stop")


def recorded():
    for name in ("hook_payloads_session1.jsonl", "hook_payloads_session2.jsonl"):
        for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines():
            yield json.loads(line)


def environment(tmp_path):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("SELF_DIRECT_", "CLAUDE_", "GROK_", "OPENROUTER_"))}
    return {**clean, "HOME": str(home), "USERPROFILE": str(home), "SELF_DIRECT_CLAUDE_PROJECTS_DIR": str(tmp_path / ".claude" / "projects")}


def run_claude_hook(event, payload, tmp_path, *args, script=SCRIPT, extra=None):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run([sys.executable, "-I", str(script), f"--event={event}", f"--index-dir={tmp_path / 'index'}",
                           "--local-only", *args], input=text, capture_output=True, text=True, timeout=60,
                          cwd=tmp_path, env={**environment(tmp_path), **(extra or {})})


def nothing_written(tmp_path):
    return not (tmp_path / "index").exists() and not any((tmp_path / "home").iterdir())


# --- the recorded and the synthetic Grok payloads ----------------------------------------------------------

@pytest.mark.parametrize("index", range(19))
def test_every_recorded_grok_payload_is_answered_with_silence(tmp_path, index):
    payloads = list(recorded())
    payload = payloads[index]
    event = payload["hook_event_name"]
    done = run_claude_hook(event if event in EVENTS else "Stop", payload, tmp_path)
    assert (done.returncode, done.stdout, done.stderr) == (0, "", "")
    assert nothing_written(tmp_path)


@pytest.mark.parametrize("event", EVENTS)
def test_all_six_events_are_silent_with_a_grok_payload(tmp_path, event):
    path = tmp_path / ".grok" / "sessions" / "%2Fwork%2Fproj" / SID / "updates.jsonl"
    done = run_claude_hook(event, hook_payload(event, path), tmp_path)
    assert (done.returncode, done.stdout, done.stderr) == (0, "", "")
    assert nothing_written(tmp_path)


@pytest.mark.parametrize("event", ["SessionStart", "UserPromptSubmit"])
def test_events_without_a_transcript_path_are_recognised_by_grok_s_own_envelope(tmp_path, event):
    payload = hook_payload(event, None)
    assert "transcript_path" not in payload and "transcriptPath" not in payload
    done = run_claude_hook(event, payload, tmp_path)
    assert (done.returncode, done.stdout, done.stderr) == (0, "", "") and nothing_written(tmp_path)


def test_a_grok_home_outside_dot_grok_is_still_recognised_by_the_envelope(tmp_path):
    payload = hook_payload("PreToolUse", tmp_path / "data" / "grokhome" / "sessions" / "g" / SID / "updates.jsonl")
    done = run_claude_hook("PreToolUse", payload, tmp_path)
    assert (done.returncode, done.stdout, done.stderr) == (0, "", "")


@pytest.mark.parametrize("path", [
    "C:\\Users\\someone\\.grok\\sessions\\C%3A%5Cwork\\" + SID + "\\updates.jsonl",
    "/home/user/.GROK/Sessions/x/" + SID + "/updates.jsonl",
    "//server/share/.grok//sessions/x/" + SID + "/updates.jsonl",
])
def test_either_slash_style_and_case_under_dot_grok_sessions(tmp_path, path):
    payload = {"hook_event_name": "PreToolUse", "session_id": SID, "transcript_path": path,
               "tool_name": "Bash", "tool_input": {"command": "ls"}}
    done = run_claude_hook("PreToolUse", payload, tmp_path)
    assert (done.returncode, done.stdout, done.stderr) == (0, "", "") and nothing_written(tmp_path)
    assert grok_ids.is_grok_payload(payload)


# --- what stays Claude Code's ----------------------------------------------------------------------------------

def claude_payload(path, event="PreToolUse", **extra):
    base = {"session_id": CLAUDE_SID, "transcript_path": str(path), "cwd": "/synthetic/project", "hook_event_name": event}
    if event == "PreToolUse":
        base.update(tool_name="Bash", tool_input={"command": "rm -rf build"})
    base.update(extra)
    return base


@pytest.fixture
def claude_world(tmp_path):
    projects, index = tmp_path / ".claude" / "projects", tmp_path / "index"
    transcript = projects / "-synthetic-project" / f"{CLAUDE_SID}.jsonl"
    transcript.parent.mkdir(parents=True)
    record = {"type": "user", "uuid": "00000000-0000-4000-8000-000000000001", "parentUuid": None,
              "sessionId": CLAUDE_SID, "timestamp": "2026-10-01T00:00:01.000Z", "cwd": "/synthetic/project",
              "isSidechain": False, "version": "9.9.9", "origin": {"kind": "human"}, "promptId": "p1",
              "message": {"role": "user", "content": "hello"}}
    transcript.write_text(json.dumps(record) + "\n", encoding="utf-8")
    index.mkdir()
    (index / "contracts.json").write_text(json.dumps({"contracts": [
        {"id": "no-rm", "type": "must_not", "scope": "tool_call", "provider": "claude", "regex": r"rm\s+-rf"}]}),
        encoding="utf-8")
    return transcript


def test_a_real_claude_payload_is_judged_exactly_as_before(tmp_path, claude_world):
    done = run_claude_hook("PreToolUse", claude_payload(claude_world), tmp_path)
    assert done.returncode == 0, done.stderr
    reply = json.loads(done.stdout)
    assert reply["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "no-rm" in reply["hookSpecificOutput"]["permissionDecisionReason"]
    clean = run_claude_hook("PreToolUse", claude_payload(claude_world, tool_input={"command": "ls"}), tmp_path)
    assert json.loads(clean.stdout) == {}


def test_a_claude_payload_that_merely_mentions_grok_is_still_claude_s(tmp_path, claude_world):
    sneaky = claude_payload(claude_world, tool_input={
        "command": "rm -rf ~/.grok/sessions/x", "hookEventName": "pre_tool_use", "workspaceRoot": "/x",
        "sessionId": SID, "transcriptPath": "/home/u/.grok/sessions/g/" + SID + "/updates.jsonl"})
    done = run_claude_hook("PreToolUse", sneaky, tmp_path)
    assert json.loads(done.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert not grok_ids.is_grok_payload(sneaky)


def test_a_claude_transcript_beside_grok_envelope_keys_is_not_a_grok_payload(tmp_path, claude_world):
    mixed = claude_payload(claude_world, hookEventName="pre_tool_use", workspaceRoot="/x")
    assert not grok_ids.is_grok_payload(mixed)
    done = run_claude_hook("PreToolUse", mixed, tmp_path)
    assert json.loads(done.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize("text", ["", "not json", "[]", "null", "5", '{"x": ".grok"}', '{"hookEventName": 5}',
                                  '{"hookEventName": "stop"}', '{"transcript_path": 7, "hookEventName": "stop"}'])
def test_garbage_is_never_mistaken_for_grok(tmp_path, text):
    done = run_claude_hook("PreToolUse", text, tmp_path)
    assert done.returncode == 0
    assert "FAILED" in json.loads(done.stdout)["systemMessage"] or "UNAVAILABLE" in json.loads(done.stdout)["systemMessage"]


# --- the script before the package ------------------------------------------------------------------------------

def test_the_bypass_runs_before_the_package_is_imported(tmp_path):
    code = ("import io, runpy, sys; sys.argv = [sys.argv[1], '--event=PreToolUse']; "
            "sys.stdin = io.StringIO(open(sys.argv[0] + '.in').read()); "
            "runpy.run_path(sys.argv[0], run_name='__main__'); "
            "print(sorted(m for m in sys.modules if m.startswith('self_directing_mcp')))")
    script = tmp_path / "claude_hook.py"
    script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    Path(str(script) + ".in").write_text(json.dumps(hook_payload("PreToolUse", Path("/x/.grok/sessions/g") / SID / "updates.jsonl")))
    done = subprocess.run([sys.executable, "-c", code, str(script)], capture_output=True, text=True, timeout=30,
                          cwd=tmp_path, env=environment(tmp_path))
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"                         # nothing of the package was imported, and nothing printed


def test_the_script_and_the_package_recognise_the_same_payloads():
    spec = importlib.util.spec_from_file_location("claude_hook_script", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    corpus = [dict(p) for p in recorded()]
    corpus += [hook_payload(event, "/x/.grok/sessions/g/" + SID + "/updates.jsonl") for event in EVENTS]
    corpus += [hook_payload(event, None) for event in ("SessionStart", "UserPromptSubmit")]
    corpus += [claude_payload("/home/u/.claude/projects/p/" + CLAUDE_SID + ".jsonl", event)
               for event in ("PreToolUse", "Stop", "SessionStart")]
    corpus += [{"hookEventName": "stop"}, {"hookEventName": "stop", "workspaceRoot": "/x"},
               {"hookEventName": "stop", "workspaceRoot": "/x", "transcript_path": "/u/.claude/projects/p/x.jsonl"},
               {"transcriptPath": "C:\\u\\.grok\\sessions\\x"}, {"transcript_path": "/a/.grok/other/x"},
               {"transcript_path": "/a/grok/sessions/x"}, {"transcript_path": 5}, {}, {"tool_input": {"hookEventName": "x"}}]
    for payload in corpus:
        assert script._is_grok_payload(json.dumps(payload)) is grok_ids.is_grok_payload(payload), payload
    assert script._is_grok_payload("[]") is False and script._is_grok_payload("{ .grok") is False


def test_the_package_defence_answers_silently_and_records_nothing(tmp_path, monkeypatch, capsys):
    index = tmp_path / "index"
    monkeypatch.setenv("SELF_DIRECT_INDEX_DIR", str(index))
    monkeypatch.setenv("SELF_DIRECT_LOCAL_ONLY", "true")
    for event in EVENTS:
        payload = hook_payload(event, "/x/.grok/sessions/g/" + SID + "/updates.jsonl")
        assert claude_hooks.run_hook(event, payload) == {}
        assert claude_hooks.dispatch_claude_hook(payload, event) == {}
    assert not index.exists()
    monkeypatch.setattr(sys, "argv", ["claude_hook.py", "--event=PreToolUse"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(hook_payload("PreToolUse", "/x/.grok/sessions/g/u.jsonl"))))
    claude_hooks.main()
    assert capsys.readouterr().out == "" and not index.exists()


def test_the_reexec_path_checks_in_the_interpreter_that_will_run_the_hook(tmp_path):
    """With a repository ``.venv`` that is not the running interpreter the hook re-executes first, so stdin is intact."""
    base = tmp_path / "repo"
    (base / "scripts").mkdir(parents=True)
    script = base / "scripts" / "claude_hook.py"
    script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    venv.create(base / ".venv", with_pip=False, symlinks=os.name != "nt")
    grok = hook_payload("PreToolUse", "/x/.grok/sessions/g/" + SID + "/updates.jsonl")
    done = run_claude_hook("PreToolUse", grok, tmp_path, script=script)
    assert (done.returncode, done.stdout, done.stderr) == (0, "", "")
    other = run_claude_hook("PreToolUse", claude_payload("/x/projects/p/" + CLAUDE_SID + ".jsonl"), tmp_path, script=script)
    assert other.returncode == 0 and "systemMessage" in json.loads(other.stdout)     # the package is absent there: visible fail-open
    assert "PreToolUse" in json.loads(other.stdout)["systemMessage"]                  # ... and the payload reached the child


def test_claude_state_is_untouched_by_grok_payloads(tmp_path):
    index = tmp_path / "index"
    index.mkdir()
    state = claude_hook_state.ClaudeHookState(index)
    state.record_error("check_failed", "PreToolUse", "ValueError")
    before = state.health()
    run_claude_hook("PreToolUse", hook_payload("PreToolUse", "/x/.grok/sessions/g/" + SID + "/updates.jsonl"), tmp_path)
    assert state.health() == before
