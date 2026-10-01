"""Real test/proof runners and adversarial classifier choices at the workflow boundary."""
from __future__ import annotations
import asyncio

import json
import os
from pathlib import Path
import shutil
from threading import Event

import httpx
import pytest

from self_directing_mcp import config, workflow, workflow_jev
from self_directing_mcp.workflow_jev import ClassifierResult, MODEL
from self_directing_mcp.workflow_requirements import approve_project


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    files = {
        "PRD.md": "# Orders\n\n## Retry\nA repeated ID creates one order.\n\n## Authorization\nAn untrusted caller cannot create an order.\n",
        "spec.txt": "insert(insert(orders,id),id) = insert(orders,id); unauthorized means False\n",
        "src/orders.py": "def insert(orders, key):\n    orders[key] = key\n    return orders[key]\n",
        "src/auth.py": "def allowed(trusted):\n    return trusted\n",
        "verify_retry.py": "from src.orders import insert\norders = {}\na = insert(orders, 'r')\nb = insert(orders, 'r')\nassert a == b and len(orders) == 1\n",
        "verify_auth.py": "from src.auth import allowed\nassert allowed(False) is False\nassert allowed(True) is True\n",
        "Proof.lean": "def remember (s : Bool) : Bool := true\ntheorem retry (s : Bool) : remember (remember s) = remember s := rfl\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    manifest = {
        "schema_version": 1, "prd": "PRD.md", "scope": ["src/**/*.py"],
        "class_mapping_version": "1", "external_context": "synthetic",
        "requirements": [
            {"id": "REQ-RETRY", "text": "Repeated IDs are idempotent", "required": True,
             "status": "approved", "source": {"path": "PRD.md", "section": "Retry", "quote": "A repeated ID creates one order."},
             "code": ["src/orders.py"], "specs": ["spec.txt"], "tests": ["retry"], "proofs": ["retry"]},
            {"id": "REQ-AUTH", "text": "Reject untrusted callers", "required": False,
             "status": "approved", "source": {"path": "PRD.md", "section": "Authorization", "quote": "An untrusted caller cannot create an order."},
             "code": ["src/auth.py"], "specs": ["spec.txt"], "tests": ["auth"], "proofs": ["retry"]},
        ],
        "tests": {"retry": {"argv": ["{python}", "verify_retry.py"], "paths": ["verify_retry.py"]},
                  "auth": {"argv": ["{python}", "verify_auth.py"], "paths": ["verify_auth.py"]}},
        "proofs": {"retry": {"path": "Proof.lean", "theorem": "retry", "statement": "∀ (s : Bool), remember (remember s) = remember s"}},
        "completion_actions": [{"tool_name": "workflow_command", "argument_pattern": ".*"}],
    }
    (root / "requirements.json").write_text(json.dumps(manifest))
    approved = approve_project(root, Path("requirements.json"), tmp_path / "approval")
    return approved


@pytest.fixture
def controller(project, tmp_path):
    lean = os.environ.get("TSUKKOMI_TEST_LEAN") or shutil.which("lean")
    if lean is None:
        pytest.skip("Real Lean toolchain required for workflow execution tests")
    if not shutil.which(os.environ.get("TSUKKOMI_BWRAP", "bwrap")):
        pytest.skip("Real bubblewrap runtime required for workflow execution tests")
    return workflow.WorkflowController(project.approval_dir, tmp_path / "state", lean=lean)


def choose(monkeypatch, controller, choice, session="case"):
    names = {0: "InsufficientEvidence", 1: "NeedsRevision", 2: "ReadyForVerification", 3: "ReadyForCompletion"}
    def adversarial_classifier(context, **kwargs):
        return ClassifierResult(choice, names[choice], {name: float(i == choice) for i, name in names.items()},
                                1.0, MODEL, workflow._digest(context), "1", "ok", "")
    monkeypatch.setattr(workflow, "classify", adversarial_classifier)
    result = controller.classify(session)
    assert result["ok"], result
    return result["classification_id"]


def advance(monkeypatch, controller, session="case"):
    for expected in ("formalize", "implement", "verify"):
        result = controller.transition(session, choose(monkeypatch, controller, 2, session))
        assert result["allowed"] and result["next_stage"] == expected, result


def command(text="done"):
    return {"tool_name": "workflow_command", "arguments": {"argv": ["{python}", "-c", f"print({text!r})"], "cwd": "."}}


@pytest.fixture
def isolated_openrouter_settings(tmp_path, monkeypatch):
    # Never read a developer's real dotenv during credential integration tests.
    monkeypatch.setattr(config, "_repo_root", lambda: tmp_path)
    monkeypatch.delenv("SELF_DIRECT_OPENROUTER_API_KEY_FILE", raising=False)
    monkeypatch.setattr(workflow, "Settings", lambda: config.Settings(_env_file=None))


@pytest.mark.parametrize("source", ["file", "setting", "environment"])
def test_classifier_uses_shared_openrouter_credential_authority(
        project, tmp_path, monkeypatch, isolated_openrouter_settings, source):
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-sentinel")
    expected_key = source + "-sentinel"
    if source == "file":
        key_file = tmp_path / "authorized.env"
        key_file.write_text("OPENROUTER_API_KEY=file-sentinel\n")
        monkeypatch.setenv("SELF_DIRECT_OPENROUTER_API_KEY_FILE", str(key_file))
    elif source == "setting":
        monkeypatch.setattr(workflow, "Settings", lambda: config.Settings(
            _env_file=None, OPENROUTER_API_KEY=expected_key))
    controller = workflow.WorkflowController(project.approval_dir, tmp_path / "credential-state")
    response_model = "typesafe/jev-1.13-20260917"

    def authenticate(request):
        if request.headers.get("Authorization") != "Bearer " + expected_key:
            return httpx.Response(401)
        return httpx.Response(200, json={
            "model": response_model,
            "answers": {"next_stage": {
                "type": "choice", "choice": "ReadyForVerification", "confidence": 1.0,
                "probabilities": {name: float(choice == 2) for choice, name in workflow_jev.CHOICES.items()},
            }},
        })

    client = httpx.Client
    transport = httpx.MockTransport(authenticate)
    monkeypatch.setattr(workflow_jev.httpx, "Client", lambda **kwargs: client(transport=transport, **kwargs))
    result = controller.classify("credential-case")
    assert result["ok"] and result["choice_id"] == 2 and result["model"] == response_model
    recorded = json.dumps(controller.status("credential-case"))
    assert expected_key not in recorded and "environment-sentinel" not in recorded


@pytest.mark.parametrize("file_content", [None, "", b"\xff", "ANTHROPIC_API_KEY=other-provider-sentinel\n"])
def test_configured_openrouter_file_failure_cannot_use_environment_or_other_provider(
        project, tmp_path, monkeypatch, isolated_openrouter_settings, file_content):
    key_file = tmp_path / "authorized.env"
    if isinstance(file_content, bytes):
        key_file.write_bytes(file_content)
    elif file_content is not None:
        key_file.write_text(file_content)
    monkeypatch.setenv("SELF_DIRECT_OPENROUTER_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-sentinel")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "other-provider-sentinel")
    controller = workflow.WorkflowController(project.approval_dir, tmp_path / "credential-state")

    def forbidden_request(request):
        raise AssertionError("A configured unusable key file must not issue a request")

    client = httpx.Client
    transport = httpx.MockTransport(forbidden_request)
    monkeypatch.setattr(workflow_jev.httpx, "Client", lambda **kwargs: client(transport=transport, **kwargs))
    result = controller.classify("credential-case")
    assert not result["ok"] and result["status"] == "missing_key" and result["choice_id"] == 0
    state = controller.status("credential-case")
    assert state["stage"] == "requirements"
    recorded = json.dumps(state)
    assert "environment-sentinel" not in recorded and "other-provider-sentinel" not in recorded


def test_wrong_completion_class_cannot_replace_actual_test_or_proof_evidence(controller, monkeypatch):
    classification = choose(monkeypatch, controller, 3)
    denied = controller.authorize("case", classification, command())
    assert not denied["ok"]
    assert {e["status"] for e in denied["evidence"]} == {"missing"}
    controller.project.root.joinpath("src/orders.py").write_text("def insert(orders, key):\n    orders[len(orders)] = key\n    return key\n")
    observed = controller.run_checks("case")
    assert not observed["ok"]
    assert next(e for e in observed["observations"] if e["kind"] == "test")["exit_code"] != 0
    advance(monkeypatch, controller)
    denied = controller.authorize("case", choose(monkeypatch, controller, 3), command())
    assert not denied["ok"] and "tests" in denied["failed"]
    controller.project.root.joinpath("src/orders.py").write_text("def insert(orders, key):\n    orders[key] = key\n    return key\n")
    assert controller.run_checks("case")["ok"]
    allowed = controller.authorize("case", choose(monkeypatch, controller, 3), command())
    executed = controller.execute("case", allowed["grant"], command())
    assert executed["ok"] and executed["action_executed"] and executed["outcome"] == "succeeded"
    assert controller.status("case")["stage"] == "complete"


def test_grant_binds_arguments_state_versions_and_single_use(controller, monkeypatch):
    assert controller.run_checks("case")["ok"]
    advance(monkeypatch, controller)
    classification = choose(monkeypatch, controller, 3)
    grant = controller.authorize("case", classification, command())["grant"]
    assert not controller.execute("other", grant, command())["action_executed"]
    assert not controller.execute("case", grant, command("changed"))["action_executed"]
    controller.project.root.joinpath("src/orders.py").write_text("def insert(orders, key):\n    return orders.setdefault(key, key)\n")
    assert not controller.execute("case", grant, command())["action_executed"]
    assert controller.run_checks("case")["ok"]
    renewed = controller.authorize("case", choose(monkeypatch, controller, 3), command())["grant"]
    assert controller.execute("case", renewed, command())["ok"]
    assert not controller.execute("case", renewed, command())["action_executed"]
    # Resume is a fresh observation, not trust in the previously saved completed stage.
    controller.project.root.joinpath("src/new.py").write_text("changed = True\n")
    reopened = workflow.WorkflowController(controller.project.approval_dir, controller.state_dir, lean=controller.lean)
    assert reopened.status("case")["stage"] == "verify"
    assert not reopened.status("case")["facts"]["tests"]


def test_authorization_obligation_comes_from_changed_code_not_classifier(controller, monkeypatch):
    controller.project.root.joinpath("src/auth.py").write_text("def allowed(trusted):\n    return True\n")
    observed = controller.run_checks("case")
    assert not observed["ok"]
    assert "REQ-AUTH" in observed["state"]["requirements"]
    assert next(e for e in observed["observations"] if e["name"] == "auth")["status"] == "failed"
    advance(monkeypatch, controller)
    denied = controller.authorize("case", choose(monkeypatch, controller, 3), command())
    assert not denied["ok"] and "tests" in denied["failed"]


def test_failed_execution_is_not_completed_and_consumes_grant(controller, monkeypatch):
    assert controller.run_checks("case")["ok"]
    advance(monkeypatch, controller)
    action = {"tool_name": "workflow_command", "arguments": {"argv": ["{python}", "-c", "raise SystemExit(7)"], "cwd": "."}}
    grant = controller.authorize("case", choose(monkeypatch, controller, 3), action)["grant"]
    result = controller.execute("case", grant, action)
    assert not result["ok"] and result["exit_code"] == 7 and result["outcome"] == "failed"
    assert controller.status("case")["stage"] == "verify"
    assert not controller.execute("case", grant, action)["action_executed"]


def test_low_confidence_and_forged_classification_cannot_advance(controller, monkeypatch):
    assert not controller.transition("case", "unrecorded-model-claim")["allowed"]
    monkeypatch.setattr(workflow, "classify", lambda *a, **kw: ClassifierResult(
        0, "InsufficientEvidence", {}, 0, MODEL, "0" * 64, "1", "low_confidence", "uncertain"))
    classification = controller.classify("case")
    assert not classification["ok"]
    result = controller.transition("case", classification["classification_id"])
    assert not result["allowed"] and result["failed"] == ["classification_low_confidence"]


@pytest.mark.parametrize("relative", ["verify_retry.py", "Proof.lean"])
def test_changed_expectations_require_owner_reapproval(controller, relative):
    controller.project.root.joinpath(relative).write_text("owner approval is required for this change\n")
    result = controller.run_checks("case")
    assert not result["ok"] and result["status"] == "approval_changed"
    assert not controller.status("case")["facts"]["approved"]


def test_readiness_transition_cannot_skip_owned_completion(controller, monkeypatch):
    assert controller.run_checks("case")["ok"]
    advance(monkeypatch, controller)
    ready = controller.transition("case", choose(monkeypatch, controller, 3))
    assert ready["allowed"] and ready["ready_for_completion"]
    assert ready["next_stage"] == "verify"
    assert controller.status("case")["stage"] == "verify"
    marker = controller.project.root / "completion-observed"
    action = {"tool_name": "workflow_command", "arguments": {
        "argv": ["{python}", "-c", "from pathlib import Path; Path('completion-observed').write_text('owned')"],
        "cwd": "."}}
    grant = controller.authorize("case", choose(monkeypatch, controller, 3), action)["grant"]
    assert controller.execute("case", grant, action)["ok"]
    assert marker.read_text() == "owned"
    assert controller.status("case")["stage"] == "complete"


def test_cancelled_request_cannot_mint_or_consume_grant(controller, monkeypatch):
    assert controller.run_checks("case")["ok"]
    advance(monkeypatch, controller)
    classification = choose(monkeypatch, controller, 3)
    cancelled = Event()
    cancelled.set()
    with pytest.raises(TimeoutError, match="cancelled"):
        controller.authorize("case", classification, command(), cancel_event=cancelled)
    grant = controller.authorize("case", classification, command())["grant"]
    with pytest.raises(TimeoutError, match="cancelled"):
        controller.execute("case", grant, command(), cancel_event=cancelled)
    assert controller.execute("case", grant, command())["ok"]


def test_cancellation_during_policy_check_rolls_back_start(controller, monkeypatch):
    assert controller.run_checks("case")["ok"]
    advance(monkeypatch, controller)
    action = command()
    grant = controller.authorize("case", choose(monkeypatch, controller, 3), action)["grant"]
    original = controller.policy.evaluate
    cancelled = Event()

    def cancel_after_real_policy(*args, **kwargs):
        decision = original(*args, **kwargs)
        cancelled.set()
        return decision

    monkeypatch.setattr(controller.policy, "evaluate", cancel_after_real_policy)
    with pytest.raises(TimeoutError, match="cancelled"):
        controller.execute("case", grant, action, cancel_event=cancelled)
    monkeypatch.setattr(controller.policy, "evaluate", original)
    assert controller.execute("case", grant, action)["ok"]


def test_missing_sandbox_consumes_grant_without_claiming_execution(controller, monkeypatch):
    assert controller.run_checks("case")["ok"]
    advance(monkeypatch, controller)
    action = command()
    grant = controller.authorize("case", choose(monkeypatch, controller, 3), action)["grant"]
    monkeypatch.setenv("TSUKKOMI_BWRAP", "/missing-workflow-bwrap")
    result = controller.execute("case", grant, action)
    assert result["status"] == "sandbox_unavailable"
    assert result["outcome"] == "not_started" and result["action_executed"] is False
    assert controller.status("case")["stage"] == "verify"
    assert not controller.execute("case", grant, action)["action_executed"]


def test_untracked_pytest_hooks_cannot_forge_passing_evidence(project, tmp_path):
    root = project.root
    manifest = json.loads((root / "requirements.json").read_text())
    manifest["tests"]["retry"] = {"argv": ["{python}", "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_retry.py"],
                                   "paths": ["test_retry.py"]}
    (root / "test_retry.py").write_text(
        "from src.orders import insert\n"
        "def test_retry():"
        "\n    orders = {}\n    insert(orders, 'x')\n    insert(orders, 'x')\n    assert len(orders) == 1\n")
    (root / "requirements.json").write_text(json.dumps(manifest))
    approved = approve_project(root, Path("requirements.json"), tmp_path / "pytest-approval")
    (root / "conftest.py").write_text("def pytest_pyfunc_call(pyfuncitem):\n    return True\n")
    (root / "sitecustomize.py").write_text("raise SystemExit(0)\n")
    (root / "src/orders.py").write_text("def insert(orders, key):\n    orders[len(orders)] = key\n")
    if not shutil.which(os.environ.get("TSUKKOMI_BWRAP", "bwrap")):
        pytest.skip("Real bubblewrap runtime required")
    controller = workflow.WorkflowController(approved.approval_dir, tmp_path / "pytest-state")
    observed = controller.run_checks("case", kind="test")
    assert not observed["ok"]
    result = observed["observations"][0]
    assert result["status"] == "failed" and result["exit_code"] == 1
    assert "stdout" not in result and "stderr" not in result
    assert not controller.status("case")["facts"]["tests"]


@pytest.mark.parametrize("cancel_request", [True, False])
def test_abandoned_mcp_request_cannot_execute_after_lock_release(controller, monkeypatch, cancel_request):
    from self_directing_mcp import server

    assert controller.run_checks("case")["ok"]
    advance(monkeypatch, controller)
    action = {"tool_name": "workflow_command", "arguments": {
        "argv": ["{python}", "-c", "open('completion-observed', 'w').write('owned')"], "cwd": "."}}
    grant = controller.authorize("case", choose(monkeypatch, controller, 3), action)["grant"]
    monkeypatch.setattr(workflow, "configured_workflow", lambda: controller)
    entered, finished = Event(), Event()
    execute = controller.execute

    def tracked_execute(*args, **kwargs):
        entered.set()
        try:
            return execute(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(controller, "execute", tracked_execute)

    async def scenario():
        with controller._db():
            pending = asyncio.create_task(server._dispatch_workflow(
                "execute", "case", grant, action, timeout_ms=5000 if cancel_request else 100))
            assert await asyncio.to_thread(entered.wait, 5)
            if cancel_request:
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                result = await pending
                assert result["outcome_unknown"] and result["action_executed"] is None
        assert await asyncio.to_thread(finished.wait, 5)

    asyncio.run(scenario())
    assert not (controller.project.root / "completion-observed").exists()
    assert controller.status("case")["stage"] == "verify"
    # No start was committed while blocked: the original grant remains usable.
    assert execute("case", grant, action)["ok"]
    assert (controller.project.root / "completion-observed").read_text() == "owned"


def test_installed_host_policy_change_invalidates_existing_grant(controller, monkeypatch):
    assert controller.run_checks("case")["ok"]
    advance(monkeypatch, controller)
    grant = controller.authorize("case", choose(monkeypatch, controller, 3), command())["grant"]
    monkeypatch.setattr(workflow, "host_policy_digest", lambda: "0" * 64)
    rejected = controller.execute("case", grant, command())
    assert rejected["ok"] is False and rejected["action_executed"] is False
    assert controller.status("case")["facts"]["tests"] is False
