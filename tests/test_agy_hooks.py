import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from self_directing_mcp.agy import PathTraversalError, resolve_session_path
from self_directing_mcp.agy_hooks import (
    dispatch_agy_hook, failure_response, handle_pre_invocation,
    handle_pre_tool_use, handle_stop,
)
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine

SID = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"
ROOT = Path(__file__).resolve().parents[1]


def transcript_path(root, sid=SID):
    return root / "brain" / sid / ".system_generated" / "logs" / "transcript.jsonl"


@pytest.fixture
def engine(tmp_path):
    app = tmp_path / "app"
    path = transcript_path(app)
    path.parent.mkdir(parents=True)
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True,
        index_dir=tmp_path / "index", agy_app_data_dirs=[app],
        openrouter_api_key_file=None))
    yield engine, path
    engine.close()


def prohibit(engine, regex, **extra):
    engine.upsert_contracts([{"id": "restriction", "type": "must_not", "scope": "tool_call",
                             "provider": "agy", "regex": regex, **extra}])


@pytest.mark.parametrize("command", ["git branch -D feature", "git diff --output=output.patch",
                                     "git log --output=history.txt", "find . -delete"])
def test_mutating_read_looking_commands_are_denied(engine, command):
    obj, path = engine
    prohibit(obj, r"branch|diff|log|find")
    result = handle_pre_tool_use(obj, SID, {"name": "run_command", "args": {"CommandLine": command}}, str(path))
    assert result["decision"] == "deny"


@pytest.mark.parametrize("tool", ["view_file", "multi_replace_file_content", "self_directing_mcp_untrusted_tool"])
def test_restricted_reads_edits_and_internal_looking_names(engine, tool):
    obj, path = engine
    prohibit(obj, "private-file")
    result = handle_pre_tool_use(obj, SID, {"name": tool, "args": {"AbsolutePath": "/private-file"}}, str(path))
    assert result["decision"] == "deny"


def test_no_rules_differs_from_enforced_unknown_and_advisory(engine):
    obj, path = engine
    call = {"name": "run_command", "args": {"CommandLine": "deploy"}}
    assert handle_pre_tool_use(obj, SID, call, str(path))["decision"] == "allow"
    obj.upsert_contracts([{"id": "precondition", "provider": "agy", "type": "must", "scope": "tool_call",
                          "regex": "npm test", "before_regex": "deploy", "requires_success": True}])
    assert handle_pre_tool_use(obj, SID, call, str(path))["decision"] == "deny"
    obj.settings.agy_enforcement = "advisory"
    assert handle_pre_tool_use(obj, SID, call, str(path))["decision"] == "allow"


def test_exception_requires_fresh_approval(engine):
    obj, path = engine
    prohibit(obj, "deploy", exception_regex="staging")
    response = handle_pre_tool_use(obj, SID, {"name": "run_command", "args": {"CommandLine": "deploy staging"}}, str(path))
    assert response["decision"] == "force_ask"


def test_contract_provider_and_session_isolation(engine):
    obj, path = engine
    prohibit(obj, "private-file", session_id=OTHER)
    obj.upsert_contracts([{"id": "codex-only", "provider": "codex", "type": "must_not", "regex": "private-file"}])
    assert not obj.hook_obligations(SID, provider="agy", path=str(path))["contracts"]
    assert obj.hook_obligations(SID)["contracts"][0].id == "codex-only"


@pytest.mark.parametrize("normal_reason", ["model_stop", "NO_TOOL_CALL"])
def test_stop_pending_and_cancellation(engine, normal_reason):
    obj, path = engine
    obj.update_checklist(SID, provider="agy", add=[{"text": "Verify integration"}])
    payload = {"transcriptPath": str(path), "fullyIdle": True, "terminationReason": normal_reason}
    assert handle_stop(obj, SID, payload)["decision"] == "continue"
    for reason in ("user_cancel", "cancelled", "error", "max_steps_exceeded"):
        payload["terminationReason"] = reason
        assert handle_stop(obj, SID, payload)["decision"] == "allow"
    payload.update(terminationReason="model_stop", fullyIdle=False)
    assert handle_stop(obj, SID, payload)["decision"] == "allow"


def test_context_contains_only_current_agy_contracts(engine):
    obj, path = engine
    prohibit(obj, "restricted", description="Do not read restricted paths")
    obj.upsert_contracts([{"id": "other", "provider": "codex", "type": "must_not", "regex": "irrelevant"}])
    result = handle_pre_invocation(obj, SID, {"transcriptPath": str(path)})
    text = result["injectSteps"][0]["ephemeralMessage"]
    assert "restriction" in text and "irrelevant" not in text


def test_discovery_rejects_traversal_mismatch_symlinks_and_ambiguity(tmp_path):
    root = tmp_path / "app"
    path = transcript_path(root)
    path.parent.mkdir(parents=True)
    path.write_text("", encoding="utf-8")
    assert resolve_session_path([root], session_id=SID) == path
    with pytest.raises(ValueError):
        resolve_session_path([root], session_id=OTHER, path=str(path))
    with pytest.raises(PathTraversalError):
        resolve_session_path([root], session_id=SID, path=str(path.parent / ".." / "logs" / "transcript.jsonl"))
    outside = tmp_path / "outside.jsonl"
    outside.write_text("", encoding="utf-8")
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(PathTraversalError):
        resolve_session_path([root], session_id=SID, path=str(path))
    path.unlink()
    path.write_text("", encoding="utf-8")
    second = transcript_path(tmp_path / "other-app")
    second.parent.mkdir(parents=True)
    second.write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        resolve_session_path([root, tmp_path / "other-app"], session_id=SID)


@pytest.mark.parametrize("payload", [None, [], {}, {"conversationId": "../escape"},
    {"conversationId": SID, "transcriptPath": "/wrong", "workspacePaths": [], "stepIdx": 0,
     "toolCall": {"name": "view_file", "args": "not-an-object"}}])
def test_invalid_payload_is_not_permission(payload):
    with pytest.raises(ValueError):
        dispatch_agy_hook(payload, "PreToolUse")


def test_event_specific_failure_controls():
    assert failure_response("PreToolUse")["decision"] == "deny"
    assert failure_response("PostToolUse") == {}
    assert "injectSteps" in failure_response("PreInvocation")
    assert failure_response("Stop", {"terminationReason": "model_stop", "fullyIdle": True})["decision"] == "continue"
    assert failure_response("Stop", {"terminationReason": "cancelled", "fullyIdle": True})["decision"] == "allow"


def test_wrapper_import_has_no_input_or_reexec(tmp_path):
    script = ROOT / "scripts" / "agy_hook.py"
    command = ("import importlib.util,os,sys; "
               "os.execv=lambda *a: (_ for _ in ()).throw(AssertionError('reexec')); "
               "sys.stdin=None; "
               "s=importlib.util.spec_from_file_location('hook',sys.argv[1]); "
               "s.loader.exec_module(importlib.util.module_from_spec(s))")
    result = subprocess.run([sys.executable, "-c", command, str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and result.stdout == ""


@pytest.mark.parametrize("event,payload,key,decision", [
    ("PreToolUse", "broken JSON", "decision", "deny"),
    ("Stop", '{"fullyIdle":true,"terminationReason":"model_stop"}', "decision", "continue"),
    ("Stop", '{"fullyIdle":true,"terminationReason":"NO_TOOL_CALL"}', "decision", "continue"),
    ("Stop", '{"fullyIdle":true,"terminationReason":"cancelled"}', "decision", "allow"),
    ("PreInvocation", "{}", "injectSteps", None),
    ("PostToolUse", "{}", None, None),
])
def test_wrapper_native_import_failure_is_event_specific(tmp_path, event, payload, key, decision):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    wrapper = scripts / "agy_hook.py"
    shutil.copyfile(ROOT / "scripts" / "agy_hook.py", wrapper)
    package = tmp_path / "src" / "self_directing_mcp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "agy_hooks.py").write_text("raise ImportError('native runtime unavailable')\n")
    result = subprocess.run([sys.executable, "-I", str(wrapper), "--event=" + event], input=payload,
                            capture_output=True, text=True, timeout=10)
    response = json.loads(result.stdout)
    assert result.returncode == 0
    if key is None:
        assert response == {}
    elif decision is None:
        assert key in response
    else:
        assert response[key] == decision


def test_real_wrapper_rejects_malformed_json_without_echoing_input():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "agy_hook.py"), "--event=PreToolUse"],
        input="do-not-echo-this-malformed-payload", capture_output=True, text=True, timeout=15)
    assert result.returncode == 0
    assert json.loads(result.stdout)["decision"] == "deny"
    assert "do-not-echo" not in result.stdout + result.stderr


def test_unrelated_recovery_action_is_not_blocked_by_missing_prerequisite_history(engine):
    obj, path = engine
    obj.upsert_contracts([{"id": "tests-first", "provider": "agy", "type": "must", "scope": "tool_call",
                          "regex": "npm test", "before_regex": "deploy", "requires_success": True}])
    repair = {"name": "run_command", "args": {"CommandLine": "npm test"}}
    assert handle_pre_tool_use(obj, SID, repair, str(path))["decision"] == "allow"
    prohibit(obj, r"npm test")
    assert handle_pre_tool_use(obj, SID, repair, str(path))["decision"] == "deny"


def test_completion_only_must_does_not_prevent_its_own_test(engine):
    obj, path = engine
    obj.upsert_contracts([{"id": "finish-with-tests", "provider": "agy", "type": "must",
                          "scope": "tool_call", "regex": "npm test", "requires_success": True}])
    assert handle_pre_tool_use(obj, SID, {"name": "run_command", "args": {"CommandLine": "npm test"}}, str(path))["decision"] == "allow"
    assert handle_stop(obj, SID, {"transcriptPath": str(path), "fullyIdle": True, "terminationReason": "NO_TOOL_CALL"})["decision"] == "continue"


def test_exception_confirmation_cannot_waive_unknown_prerequisite(engine):
    obj, path = engine
    prohibit(obj, "deploy", exception_regex="staging")
    obj.upsert_contracts([{"id": "tests-first", "provider": "agy", "type": "must", "scope": "tool_call",
                          "regex": "npm test", "before_regex": "deploy", "requires_success": True}])
    response = handle_pre_tool_use(obj, SID, {"name": "run_command", "args": {"CommandLine": "deploy staging"}}, str(path))
    assert response["decision"] == "deny"


def test_forged_checklist_result_ids_do_not_unlock_stop(engine):
    obj, path = engine
    obj.update_checklist(SID, provider="agy", add=[{"text": "Verify the result"}])
    claimed = obj.update_checklist(SID, provider="agy", update=[
        {"item_id": "req-001", "status": "done", "verified": True, "evidence_chunk_ids": ["invented-result"]}])
    assert claimed["updated"][0]["status"] == "pending_verification"
    assert handle_stop(obj, SID, {"transcriptPath": str(path), "fullyIdle": True, "terminationReason": "NO_TOOL_CALL"})["decision"] == "continue"
