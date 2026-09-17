import json
from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest

from self_directing_mcp.agy_hooks import (
    is_read_only_tool_call,
    handle_pre_tool_use,
    handle_stop,
    dispatch_agy_hook,
)
from self_directing_mcp.schemas import ContractRule
from self_directing_mcp.checklist import RequirementItem


def test_is_read_only():
    assert is_read_only_tool_call("view_file", {"AbsolutePath": "/a/b"}) is True
    assert is_read_only_tool_call("list_dir", {}) is True
    assert is_read_only_tool_call("grep_search", {}) is True
    assert is_read_only_tool_call("run_command", {"CommandLine": "ls -la"}) is True
    assert is_read_only_tool_call("run_command", {"CommandLine": "git status"}) is True
    assert is_read_only_tool_call("run_command", {"CommandLine": "rm -rf /"}) is False
    assert is_read_only_tool_call("write_to_file", {}) is False


def test_pre_tool_use_allow_read():
    engine = MagicMock()
    tool_call = {"name": "view_file", "args": {"AbsolutePath": "/foo"}}
    res = handle_pre_tool_use(engine, "s1", tool_call)
    assert res["decision"] == "allow"
    engine.check_action.assert_not_called()


def test_pre_tool_use_deny_violation():
    engine = MagicMock()
    engine.check_action.return_value = {
        "verdict": "violation",
        "findings": [{"contract_id": "no-delete", "verdict": "violation", "reason": "Cannot rm -rf"}],
    }
    tool_call = {"name": "run_command", "args": {"CommandLine": "rm -rf /"}}
    res = handle_pre_tool_use(engine, "s1", tool_call)
    assert res["decision"] == "deny"
    assert "no-delete" in res["reason"]


def test_pre_tool_use_ask_suspicious():
    engine = MagicMock()
    engine.check_action.return_value = {
        "verdict": "suspicious",
        "findings": [{"contract_id": "rule-deploy", "verdict": "suspicious", "reason": "Deploy requires check"}],
    }
    tool_call = {"name": "run_command", "args": {"CommandLine": "deploy --prod"}}
    res = handle_pre_tool_use(engine, "s1", tool_call)
    assert res["decision"] == "ask"
    assert "rule-deploy" in res["reason"]


def test_stop_with_pending_checklist():
    engine = MagicMock()
    engine.hook_obligations.return_value = {
        "pending": [
            RequirementItem(item_id="req1", text="Finish integration test", status="pending_verification", verified=False)
        ]
    }
    res = handle_stop(engine, "s1", {})
    assert res["decision"] == "continue"
    assert "Finish integration test" in res["reason"]


def test_stop_clean():
    engine = MagicMock()
    engine.hook_obligations.return_value = {"pending": []}
    res = handle_stop(engine, "s1", {})
    assert res == {}


def test_pre_invocation_injects_graphrag_context():
    from self_directing_mcp.agy_hooks import handle_pre_invocation

    engine = MagicMock()
    engine.hook_obligations.return_value = {"contracts": [], "pending": []}
    engine.get_graph_context.return_value = {
        "nodes": [{"node_id": "auth.py", "kind": "file"}, {"node_id": "test_auth.py", "kind": "file"}],
        "edges": [
            {"src": "test_auth.py", "dst": "auth.py", "relation": "DEPENDS_ON", "origin": "observed"}
        ],
    }

    res = handle_pre_invocation(engine, "s1", {"invocationNum": 1})
    assert "injectSteps" in res
    ephemeral = res["injectSteps"][0]["ephemeralMessage"]
    assert "test_auth.py -[DEPENDS_ON]-> auth.py" in ephemeral
    assert "Active Context & GraphRAG Guard" in ephemeral


def test_pre_invocation_quiet_when_empty():
    from self_directing_mcp.agy_hooks import handle_pre_invocation

    engine = MagicMock()
    engine.hook_obligations.return_value = {"contracts": [], "pending": []}
    engine.get_graph_context.return_value = {"nodes": [], "edges": []}

    res = handle_pre_invocation(engine, "s1", {"invocationNum": 1})
    assert res == {}


def test_pre_tool_use_with_graphrag_dependencies():
    engine = MagicMock()
    engine.check_action.return_value = {"verdict": "clean", "findings": []}
    mock_graph = MagicMock()
    mock_graph.neighbors.return_value = {
        "edges": [{"src": "test_auth.py", "dst": "auth_service.py", "relation": "DEPENDS_ON"}]
    }
    engine.graph = mock_graph

    tool_call = {"name": "replace_file_content", "args": {"TargetFile": "/src/auth_service.py"}}
    res = handle_pre_tool_use(engine, "s1", tool_call)
    assert res["decision"] == "allow"
    assert "GraphRAG Note" in res.get("reason", "")
    assert "test_auth.py -[DEPENDS_ON]-> auth_service.py" in res["reason"]

