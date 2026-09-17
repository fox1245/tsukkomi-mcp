"""Register one installed server, managed AGENTS text, and reviewed Codex hooks."""
from __future__ import annotations

import argparse
import asyncio
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tomllib
import uuid

from self_directing_mcp.codex_rpc import CodexRPC

SERVER = "self-directing-mcp"
HOOK_TOOL = "codex_session_hook"
BEGIN = "<!-- BEGIN self-directing-mcp managed guidance -->"
END = "<!-- END self-directing-mcp managed guidance -->"
GUIDANCE = """## Self-directing MCP

Use session history when evidence is missing and constraint checks when they apply.
Register only explicit user-added/changed/revoked constraints, using the real session id
and source wording. Questions, greetings and status requests create no new contracts.
Track ordinary outcomes with evidence-backed checklist items when useful; do not turn
every task description into a mandatory audit ritual.

Use check_action for actions affected by active constraints. Hooks skip absent contracts
and proven reads without an applicable restriction; dynamic commands remain conservative.
An exact fresh hook check avoids a redundant manual call. Audit/check_action refresh locally;
do not add a separate sync or history search at every turn. Verify outstanding obligations
at meaningful completion checkpoints. Reuse evidence only while its relevant state is unchanged.

violation/suspicious: stop the affected action and report the finding. unknown: resolve the
relevant evidence gap before claiming compliance; continue independent authorized work.
not_applicable/skipped/silent hooks are not clean and grant no permission. Description-only
contracts and retrieval rankings are not proof. Hooks never execute, authorize or restart work.
Keep session text local: sync_session(embed=False), regex/sparse search. Remote embeddings
require explicit authorization. Report unavailable checks without blocking unrelated useful work.
"""


def merge_guidance(original):
    if original.count(BEGIN) != original.count(END) or original.count(BEGIN) > 1:
        raise ValueError("Ambiguous or incomplete managed AGENTS markers")
    block = BEGIN + "\n" + GUIDANCE.rstrip() + "\n" + END
    if BEGIN in original:
        start, end = original.index(BEGIN), original.index(END) + len(END)
        if end < start:
            raise ValueError("Managed AGENTS markers are reversed")
        return original[:start] + block + original[end:]
    return original + ("\n" if original.endswith("\n") else "\n\n") + block + "\n"


def hook_handler(event):
    # Whole placeholders retain their JSON type in Codex mcp_tool hooks.
    template = {"event_name": event, "session_id": "$" + "{session_id}",
                "transcript_path": "$" + "{transcript_path}", "turn_id": "$" + "{turn_id}"}
    if event == "PreToolUse":
        template.update(tool_name="$" + "{tool_name}", tool_input="$" + "{tool_input}")
    if event == "UserPromptSubmit":
        # Existing connected servers already accept tool_input, allowing an
        # in-place definition update before their next process restart.
        template["tool_input"] = "$" + "{prompt}"
    if event == "Stop":
        template["stop_hook_active"] = "$" + "{stop_hook_active}"
    return {"type": "mcp_tool", "server": SERVER, "tool": HOOK_TOOL, "input": template,
            "timeout": 20, "statusMessage": "Checking session instructions"}


def merge_hooks(original):
    merged = copy.deepcopy(original)
    events = merged.setdefault("hooks", {})
    if not isinstance(events, dict):
        raise ValueError("hooks must be an object")
    for event in ("UserPromptSubmit", "PreToolUse", "Stop"):
        groups = events.setdefault(event, [])
        matches = [(group, i) for group in groups for i, handler in enumerate(group.get("hooks", []))
                   if handler.get("server") == SERVER and handler.get("tool") == HOOK_TOOL]
        if len(matches) > 1:
            raise ValueError(f"Duplicate owned {event} hooks; refusing to shift other hook trust keys")
        if matches:
            group, index = matches[0]
            # Existing group matcher could restrict our handler. Do not alter shared groups.
            if group.get("matcher") not in (None, "", "*"):
                raise ValueError("Owned hook is in a restricted matcher group")
            group["hooks"][index] = hook_handler(event)
        else:
            groups.append({"hooks": [hook_handler(event)]})
    return merged


def server_config(python, home, previous=None):
    result = copy.deepcopy(previous or {})
    if result.get("url"):
        raise ValueError("Existing server with this name uses HTTP; refusing to replace it")
    result.update(command=str(Path(python).resolve()), args=["-m", "self_directing_mcp"], enabled=True)
    result.setdefault("startup_timeout_sec", 30)
    result.setdefault("tool_timeout_sec", 60)
    env = result.setdefault("env", {})
    env.setdefault("SELF_DIRECT_INDEX_DIR", str(Path(home) / "mcp-servers" / SERVER / "index"))
    env.setdefault("SELF_DIRECT_CODEX_SESSIONS_DIR", str(Path(home) / "sessions"))
    env.setdefault("SELF_DIRECT_USE_FAKE_EMBEDDER", "false")
    env.setdefault("SELF_DIRECT_DENSE_BACKEND", "sqlite-vector")
    env.setdefault("SELF_DIRECT_SQLITE_VECTOR_PATH", str(Path(home) / "mcp-servers" / SERVER / "native" / ("vector.dll" if os.name == "nt" else "vector.dylib" if sys.platform == "darwin" else "vector.so")))
    if os.environ.get("SELF_DIRECT_OPENROUTER_API_KEY_FILE"):
        env.setdefault("SELF_DIRECT_OPENROUTER_API_KEY_FILE", os.environ["SELF_DIRECT_OPENROUTER_API_KEY_FILE"])
    env.setdefault("SELF_DIRECT_SEED_EXAMPLE_CONTRACTS", "false")
    return result


def atomic_write(path, content, expected):
    actual = path.read_bytes() if path.exists() else None
    if actual != expected:
        raise RuntimeError(f"{path.name} changed concurrently; no overwrite performed")
    temporary = path.with_name(path.name + ".self-directing-" + uuid.uuid4().hex + ".tmp")
    temporary.write_bytes(content)
    if expected is not None:
        shutil.copymode(path, temporary)
    elif os.name != "nt":
        temporary.chmod(0o600)
    temporary.replace(path)


def read_config(home):
    path = home / "config.toml"
    return tomllib.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def guidance_path(home):
    override = Path(home) / "AGENTS.override.md"
    if override.exists() and override.read_text(encoding="utf-8-sig").strip():
        return override
    return Path(home) / "AGENTS.md"


def verify_files(home):
    home = Path(home).resolve()
    config = read_config(home)
    server = config.get("mcp_servers", {}).get(SERVER, {})
    if not server.get("enabled", True) or not Path(server.get("command", "")).is_file():
        raise RuntimeError("Configured MCP executable is missing or disabled")
    text = guidance_path(home).read_text(encoding="utf-8-sig")
    if BEGIN not in text or merge_guidance(text) != text:
        raise RuntimeError("Effective global AGENTS guidance is missing or outdated")
    hooks = json.loads((home / "hooks.json").read_text(encoding="utf-8-sig"))
    if merge_hooks(hooks) != hooks:
        raise RuntimeError("Hook registration is incomplete or outdated")
    return {"agents_file": str(guidance_path(home)), "files": "verified"}


def owned_metadata(response, hooks_path):
    owned = {}
    for entry in response.get("data", []):
        for error in entry.get("errors", []):
            if Path(error["path"]).resolve() == hooks_path.resolve():
                raise RuntimeError("Codex rejected the generated hooks file")
        for hook in entry.get("hooks", []):
            if (hook.get("server") == SERVER and hook.get("tool") == HOOK_TOOL
                    and Path(hook["sourcePath"]).resolve() == hooks_path.resolve()):
                owned[hook["key"]] = hook
    if len(owned) != 3 or {h["eventName"] for h in owned.values()} != {"userPromptSubmit", "preToolUse", "stop"}:
        raise RuntimeError("Codex did not load exactly the three expected lifecycle hooks")
    if any(not h["enabled"] or h["isManaged"] for h in owned.values()):
        raise RuntimeError("Owned hook is disabled or unexpectedly managed")
    return list(owned.values())


def user_version(rpc, cwd, config_path):
    result = rpc.request("config/read", {"includeLayers": True, "cwd": str(cwd)})
    for layer in result.get("layers") or []:
        name = layer.get("name", {})
        if name.get("type") == "user" and Path(name["file"]).resolve() == config_path.resolve():
            return layer["version"]
    return None


def configure(home, python, codex, *, trust_hooks=False, dry_run=False):
    home = Path(home).expanduser().resolve()
    config_path, hooks_path, agents_path = home / "config.toml", home / "hooks.json", guidance_path(home)
    config = read_config(home)
    originals = {p: p.read_bytes() if p.exists() else None for p in (config_path, hooks_path, agents_path)}
    old_hooks = json.loads((originals[hooks_path] or b"{}").decode("utf-8-sig"))
    new_hooks = merge_hooks(old_hooks)
    old_guidance = (originals[agents_path] or b"").decode("utf-8-sig")
    new_guidance = merge_guidance(old_guidance)
    new_server = server_config(python, home, config.get("mcp_servers", {}).get(SERVER))
    planned = {"server": SERVER, "python": new_server["command"], "codex_home": str(home),
               "agents_file": str(agents_path),
               "events": ["UserPromptSubmit", "PreToolUse", "Stop"], "trust_requested": trust_hooks}
    if dry_run:
        return {**planned, "dry_run": True}
    home.mkdir(parents=True, exist_ok=True)
    changes = {}
    if new_hooks != old_hooks:
        changes[hooks_path] = (json.dumps(new_hooks, indent=2, ensure_ascii=False) + "\n").encode()
    if new_guidance != old_guidance:
        changes[agents_path] = new_guidance.encode()
    edits = []
    if new_server != config.get("mcp_servers", {}).get(SERVER):
        edits.append({"keyPath": f"mcp_servers.{SERVER}", "value": new_server, "mergeStrategy": "replace"})
    if config.get("features", {}).get("hooks") is not True:
        edits.append({"keyPath": "features.hooks", "value": True, "mergeStrategy": "replace"})
    backup = home / "backups" / ("self-directing-mcp-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    # Backups may contain existing credentials; leave them only under the user's Codex home.
    backup.mkdir(parents=True, mode=0o700)
    for path, content in originals.items():
        if content is not None:
            (backup / path.name).write_bytes(content)
            if os.name != "nt":
                (backup / path.name).chmod(0o600)
    version = None
    with CodexRPC(codex, home, home) as rpc:
        version = user_version(rpc, home, config_path)
        if edits:
            if (config_path.read_bytes() if config_path.exists() else None) != originals[config_path]:
                raise RuntimeError("config.toml changed concurrently")
            reply = rpc.request("config/batchWrite", {"edits": edits, "filePath": str(config_path), "expectedVersion": version})
            if reply["status"] != "ok":
                raise RuntimeError("Codex settings are overridden by another configuration layer")
            version = reply["version"]
        for path, content in changes.items():
            atomic_write(path, content, originals[path])
    # New app-server process guarantees hooks were reloaded from disk.
    expected_hooks = hooks_path.read_bytes()
    with CodexRPC(codex, home, home) as rpc:
        metadata = owned_metadata(rpc.request("hooks/list", {"cwds": [str(home)]}), hooks_path)
        if trust_hooks:
            trust_edits = []
            for hook in metadata:
                if hook["trustStatus"] != "trusted":
                    trust_edits.append({
                        "keyPath": "hooks.state." + json.dumps(hook["key"]) + ".trusted_hash",
                        "value": hook["currentHash"], "mergeStrategy": "replace"})
            if trust_edits:
                if hooks_path.read_bytes() != expected_hooks:
                    raise RuntimeError("Hooks changed during review; refusing to trust changed definitions")
                version = user_version(rpc, home, config_path)
                reply = rpc.request("config/batchWrite", {"edits": trust_edits, "filePath": str(config_path), "expectedVersion": version})
                if reply["status"] != "ok":
                    raise RuntimeError("Hook trust configuration is overridden")
    with CodexRPC(codex, home, home) as rpc:
        metadata = owned_metadata(rpc.request("hooks/list", {"cwds": [str(home)]}), hooks_path)
    trusted = all(h["trustStatus"] == "trusted" for h in metadata)
    if trust_hooks and not trusted:
        raise RuntimeError("Hooks registered but Codex did not confirm trust; inspect /hooks")
    return {**planned, "backup": str(backup), "registered": True, "trusted": trusted,
            "hooks": [{"event": h["eventName"], "key": h["key"], "trust": h["trustStatus"]} for h in metadata],
            "restart_required": True}


async def verify_mcp(home):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    config = read_config(Path(home))["mcp_servers"][SERVER]
    env = dict(os.environ)
    env.update(config.get("env", {}))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    params = StdioServerParameters(command=config["command"], args=config.get("args", []),
                                   env=env, cwd=config.get("cwd", str(home)))
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as client:
            await client.initialize()
            names = {tool.name for tool in (await client.list_tools()).tools}
            if not {"check_action", "audit_session", HOOK_TOOL} <= names:
                raise RuntimeError("Required MCP tools missing")
            reply = await client.call_tool(HOOK_TOOL, {"event_name": "UserPromptSubmit"})
            if reply.isError:
                raise RuntimeError("MCP hook smoke call failed")
            data = json.loads(reply.content[0].text)
            if data != {} and data.get("hookSpecificOutput", {}).get("hookEventName") != "UserPromptSubmit":
                raise RuntimeError("Invalid hook output contract")
            reply = await client.call_tool("audit_status", {})
            if reply.isError:
                raise RuntimeError("MCP runtime initialization failed; check API key source and native library")
            status = json.loads(reply.content[0].text)
            configured = config.get("env", {})
            fake = str(configured.get("SELF_DIRECT_USE_FAKE_EMBEDDER", "false")).lower() in ("true", "1", "yes", "on")
            expected_embedder = "FakeEmbedder" if fake else "OpenRouterEmbedder"
            expected_backend = configured.get("SELF_DIRECT_DENSE_BACKEND", "sqlite-vector")
            if status.get("embedder") != expected_embedder or status.get("dense_backend") != expected_backend:
                raise RuntimeError("MCP runtime does not match the configured embedder/backend")
            return {"tool_count": len(names), "hook_smoke": "passed", "session_text_uploaded": False,
                    "embedder": status["embedder"], "dense_backend": status["dense_backend"],
                    "vector_extension_version": status.get("vector_extension_version"),
                    "vector_compute_backend": status.get("vector_compute_backend")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-home", type=Path, default=Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")))
    parser.add_argument("--codex", default=shutil.which("codex"))
    parser.add_argument("--trust-hooks", action="store_true", help="Trust only this installer's exact three reviewed hook definitions")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.codex:
        parser.error("Codex CLI is required")
    try:
        if args.verify:
            with CodexRPC(args.codex, args.codex_home, args.codex_home) as rpc:
                hooks = owned_metadata(rpc.request("hooks/list", {"cwds": [str(args.codex_home.resolve())]}), args.codex_home / "hooks.json")
            result = {"registered": True, "trusted": all(h["trustStatus"] == "trusted" for h in hooks)}
            if not result["trusted"]:
                raise RuntimeError("Hooks are registered but require review/trust in /hooks")
        else:
            result = configure(args.codex_home, sys.executable, args.codex,
                               trust_hooks=args.trust_hooks, dry_run=args.dry_run)
        if not args.dry_run:
            result["files"] = verify_files(args.codex_home)
            result["mcp"] = asyncio.run(asyncio.wait_for(verify_mcp(args.codex_home), timeout=45))
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        # Do not print config values, tokens, or remote server output.
        print(json.dumps({"ok": False, "error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
