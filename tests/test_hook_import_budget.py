"""The command-hook path must stay light: no NumPy, MCP, pydantic-settings or HTTP stack after a real judgement.

A hook is a fresh Python process per tool call, so every module it imports is paid for on every call (measured on the
synthetic benchmark, see README "Hook import budget"). This test runs *representative judgements* in subprocesses, not
bare imports: a PreToolUse that evaluates contracts against the transcript, a Stop checkpoint (sync and audit), a
PreInvocation, a SessionStart, and the bookkeeping events. Afterwards ``sys.modules`` must not contain a forbidden
module, and the only third-party packages loaded may be the documented exceptions below.

Synthetic data only. The hosts' entry scripts are covered by ``tests/test_claude_hooks.py`` and
``tests/test_agy_hooks.py``; here the package ``main`` functions they call are driven directly so that the
interpreter, and therefore ``sys.modules``, stays inspectable.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Never on the hook path (local-only). Each entry says why the hook has no use for it.
FORBIDDEN = {
    "numpy": "dense vectors only; local-only hooks never embed (loads ~40 ms and ~25 MB)",
    "mcp": "the server transport; hooks are plain processes",
    "pydantic_settings": "replaced by settings_core.load_hook_settings (~30 ms)",
    "httpx": "the OpenRouter client; hooks never call the network (~30 ms)",
    "dotenv": "key-file reading; hooks never need a key",
    "zstandard": "compressed Codex sessions only",
    "anyio": "an MCP/httpx dependency",
    "starlette": "an MCP dependency",
}
FORBIDDEN_OURS = ("self_directing_mcp.server", "self_directing_mcp.config")
# Documented exceptions: third-party packages the hook path does load, with the reason and the measured cost
# (python -X importtime, this machine; see README "Hook import budget").
ALLOWED_THIRD_PARTY = {
    "pydantic": "schemas.py models (ContractRule, Chunk, AuditResult ...) validate the contracts file and every "
                "chunk; ~20 ms import + ~15 ms building the models",
    "pydantic_core": "pydantic's compiled core",
    "annotated_types": "pydantic dependency",
    "typing_extensions": "pydantic dependency",
    "typing_inspection": "pydantic dependency",
    "neograph_engine": "project rule (AGENTS.md): audit judgements run through the NeoGraph native graph, with no "
                       "direct executor fallback; ~25 ms import, kept as a documented exception",
    "certifi": "imported by neograph_engine itself (TLS roots for its network nodes); well under 1 ms",
}

SID = "aaaaaaaa-1111-4111-8111-aaaaaaaa0001"
AGY_SID = "11111111-1111-4111-8111-111111111111"

RUNNER = r"""
import io, json, sys
baseline = set(sys.modules)
host, event, payload, argv = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
sys.argv = ["hook", "--event=" + event, *argv]
sys.stdin = io.StringIO(payload)
captured, real = io.StringIO(), sys.stdout
sys.stdout = captured
try:
    if host == "claude":
        from self_directing_mcp.claude_hooks import main
    else:
        from self_directing_mcp.agy_hooks import main
    main()
finally:
    sys.stdout = real
loaded = sorted(set(sys.modules) - baseline)
print(json.dumps({"response": captured.getvalue(), "loaded": loaded,
                  "stdlib": sorted(sys.stdlib_module_names)}))
"""


def _bench():
    spec = importlib.util.spec_from_file_location("bench_hooks_for_tests", ROOT / "scripts" / "bench_hooks.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CONTRACTS = {"contracts": [
    {"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf"},
    {"id": "tests-before-deploy", "type": "must", "scope": "tool_call", "regex": "pytest", "before_regex": "deploy",
     "requires_success": True},
    {"id": "tests-ran", "type": "must", "scope": "tool_call", "regex": "pytest"},
]}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Synthetic Claude and AGY sessions (each with a successful ``pytest`` call), one contracts file, one config."""
    bench = _bench()
    root = tmp_path_factory.mktemp("hook-budget")
    projects, app, index = root / "projects", root / "agy-app", root / "index"
    claude = bench.Transcript("claude", bench.claude_path(projects, SID), SID)
    claude.create(60_000)
    agy = bench.Transcript("agy", bench.agy_path(app, AGY_SID), AGY_SID)
    agy.create(60_000)
    index.mkdir()
    (index / "contracts.json").write_text(json.dumps(CONTRACTS), encoding="utf-8")
    config = root / "config.toml"
    config.write_text("[paths]\n" + "\n".join([
        f"index_dir = {json.dumps(str(index))}", f"claude_projects_dir = {json.dumps(str(projects))}",
        f"agy_app_data_dirs = [{json.dumps(str(app))}]"]) + "\n", encoding="utf-8")
    home = root / "home"
    home.mkdir()
    return {"bench": bench, "root": root, "index": index, "config": config, "home": home,
            "claude": claude, "agy": agy}


def _env(world, **extra):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SELF_DIRECT_", "CODEX_", "CLAUDE_", "OPENROUTER", "TSUKKOMI_"))}
    env.update(HOME=str(world["home"]), USERPROFILE=str(world["home"]), PYTHONPATH=str(ROOT / "src"),
               SELF_DIRECT_CONFIG_FILE=str(world["config"]), **extra)
    return env


def _run(world, host, event, command="deploy --prod", *, local_only=True):
    bench = world["bench"]
    if host == "claude":
        body = bench.claude_payload(event, SID, world["claude"].path, world["claude"].last_step + 1)
        if "tool_input" in body and event == "PreToolUse":
            body["tool_input"] = {"command": command}
        argv = [f"--index-dir={world['index']}"] + (["--local-only"] if local_only else [])
    else:
        body = bench.agy_payload(event, AGY_SID, world["agy"].path, world["agy"].last_step + 1)
        if event == "PreToolUse":
            body["toolCall"]["args"] = {"CommandLine": command}
        argv = []
    extra = {"SELF_DIRECT_LOCAL_ONLY": "true"} if (local_only and host == "agy") else (
        {} if local_only else {"SELF_DIRECT_USE_FAKE_EMBEDDER": "true", "SELF_DIRECT_DENSE_BACKEND": "numpy"})
    done = subprocess.run([sys.executable, "-c", RUNNER, host, event, json.dumps(body), json.dumps(argv)],
                          env=_env(world, **extra), capture_output=True, text=True, timeout=120, cwd=world["root"])
    assert done.returncode == 0, done.stderr[-800:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    return report, done.stderr


def _third_party(report):
    stdlib = set(report["stdlib"])
    # ``_sysconfigdata__<platform>`` is CPython's generated build-variable module, loaded by sysconfig.
    return {name.split(".")[0] for name in report["loaded"]
            if name.split(".")[0] not in stdlib and name.split(".")[0] != "self_directing_mcp"
            and not name.startswith("_sysconfigdata")}


JUDGEMENTS = [
    ("claude", "PreToolUse", "rm -rf /", "deny"),               # a confirmed violation: contracts really evaluated
    ("claude", "PreToolUse", "deploy --prod", "pass"),          # prerequisite satisfied by the pytest call
    ("claude", "Stop", None, "pass"),                           # sync + audit checkpoint
    ("claude", "SessionStart", None, "context"),
    ("claude", "UserPromptSubmit", None, "pass"),
    ("claude", "PostToolUse", None, "pass"),
    ("agy", "PreToolUse", "rm -rf /", "deny"),
    ("agy", "PreToolUse", "deploy --prod", "pass"),
    ("agy", "PreInvocation", None, "inject"),
    ("agy", "Stop", None, "pass"),
    ("agy", "PostToolUse", None, "pass"),
]


def _check_decision(host, expect, response):
    text = json.dumps(response)
    assert "UNAVAILABLE" not in text and "FAILED" not in text, f"the hook failed open instead of judging: {text[:300]}"
    if host == "claude":
        decision = response.get("hookSpecificOutput", {}).get("permissionDecision")
        blocked = response.get("decision") == "block"
    else:
        decision = response.get("decision")
        blocked = decision in ("continue", "force_ask", "ask")
    if expect == "deny":
        assert decision == "deny", text[:300]
    elif expect == "pass":
        assert decision in (None, "allow") and not blocked, text[:300]
    elif expect == "context":
        assert "Contract [" in text, text[:300]
    elif expect == "inject":
        assert "injectSteps" in response and "Contract [" in text, text[:300]


@pytest.mark.parametrize("host,event,command,expect", JUDGEMENTS,
                         ids=[f"{h}-{e}-{c or 'none'}" for h, e, c, _x in JUDGEMENTS])
def test_representative_judgements_leave_the_forbidden_modules_unloaded(world, host, event, command, expect):
    report, _stderr = _run(world, host, event, command or "deploy --prod")
    _check_decision(host, expect, json.loads(report["response"]))
    loaded = set(report["loaded"])
    top = {name.split(".")[0] for name in loaded}
    leaked = sorted(set(FORBIDDEN) & top)
    assert not leaked, ("forbidden modules loaded on the hook path: "
                        + ", ".join(f"{m} ({FORBIDDEN[m]})" for m in leaked))
    assert not sorted(set(FORBIDDEN_OURS) & loaded), sorted(set(FORBIDDEN_OURS) & loaded)
    unexpected = sorted(_third_party(report) - set(ALLOWED_THIRD_PARTY))
    assert not unexpected, ("undocumented third-party imports on the hook path: " + ", ".join(unexpected)
                            + ". Remove them from the hook path or document them in ALLOWED_THIRD_PARTY and README "
                            "'Hook import budget' with their reason and measured cost.")


@pytest.mark.parametrize("event", ["UserPromptSubmit", "PostToolUse", "PostToolUseFailure"])
def test_claude_bookkeeping_events_load_no_engine_models_or_native_runtime(world, event):
    """These events only deliver queued warnings: they fire after every tool call and must stay nearly free."""
    report, _stderr = _run(world, "claude", event)
    assert "UNAVAILABLE" not in report["response"] and "FAILED" not in report["response"], report["response"][:300]
    loaded = set(report["loaded"])
    top = {name.split(".")[0] for name in loaded}
    heavy = {"pydantic", "pydantic_core", "neograph_engine"} | set(FORBIDDEN)
    assert not heavy & top, sorted(heavy & top)
    assert not {"self_directing_mcp.engine", "self_directing_mcp.schemas", "self_directing_mcp.claude"} & loaded
    assert not _third_party(report) - {"typing_extensions"}, _third_party(report)


@pytest.mark.parametrize("host", ["claude", "agy"])
def test_default_mode_hooks_still_skip_the_server_and_http_stack(world, host):
    """Without --local-only the engine initializes the dense side (NumPy is then legitimate); nothing else is."""
    report, _stderr = _run(world, host, "PreToolUse", "rm -rf /", local_only=False)
    _check_decision(host, "deny", json.loads(report["response"]))
    top = {name.split(".")[0] for name in report["loaded"]}
    assert not (set(FORBIDDEN) - {"numpy"}) & top, sorted((set(FORBIDDEN) - {"numpy"}) & top)
    assert "self_directing_mcp.server" not in report["loaded"]


def test_numpy_loads_on_first_use_through_the_single_loading_point(tmp_path):
    probe = (
        "import os, sys\n"
        "from self_directing_mcp import _runtime\n"
        "from self_directing_mcp._runtime import np\n"
        "before = 'numpy' in sys.modules\n"
        "array = np.zeros(3)\n"
        "print(before, 'numpy' in sys.modules, type(array).__module__, np.asarray is sys.modules['numpy'].asarray,\n"
        "      os.environ.get('OPENBLAS_NUM_THREADS'), _runtime.numpy_loaded())\n")
    env = {k: v for k, v in os.environ.items() if k not in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                                           "MKL_NUM_THREADS")}
    env["PYTHONPATH"] = str(ROOT / "src")
    done = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=120,
                          check=True, cwd=tmp_path)
    assert done.stdout.split() == ["False", "True", "numpy", "True", "None", "True"], done.stdout
