"""Documentation checks that need no host: example files match their generator, links resolve, nothing private."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = [ROOT / name for name in ("README.md", "AGENTS.md", "CHANGELOG.md", "SECURITY.md")] + sorted(
    (ROOT / "docs").glob("*.md"))
_LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)\)")
_FENCE = re.compile(r"^\s*```")


def _installer():
    spec = importlib.util.spec_from_file_location("install_claude_hooks", ROOT / "scripts" / "install_claude_hooks.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _slug(heading: str) -> str:
    """GitHub's heading anchor: lowercase, drop punctuation other than hyphen/underscore, spaces to hyphens."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower().replace("`", ""))
    return text.replace(" ", "-")


def _outside_fences(path: Path):
    inside = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if _FENCE.match(line):
            inside = not inside
        elif not inside:
            yield line


def _anchors(path: Path) -> set[str]:
    return {_slug(match.group(1)) for line in _outside_fences(path)
            if (match := re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line))}


def test_claude_hooks_example_is_what_the_installer_prints():
    installer = _installer()
    python = "/absolute/path/to/venv/bin/python"
    script = "/absolute/path/to/tsukkomi-mcp/scripts/claude_hook.py"
    expected = {"hooks": installer.desired_hooks(python, script),
                "env": {installer.ENV_KEY: "confirmed_only"}}
    assert json.loads((ROOT / "docs" / "claude-hooks.example.json").read_text(encoding="utf-8")) == expected


def test_grok_hooks_example_is_what_the_installer_prints():
    spec = importlib.util.spec_from_file_location("install_grok_hooks", ROOT / "scripts" / "install_grok_hooks.py")
    installer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = installer
    spec.loader.exec_module(installer)
    expected = {"hooks": installer.desired_hooks("/absolute/path/to/venv/bin/python",
                                                 "/absolute/path/to/tsukkomi-mcp/scripts/grok_hook.py")}
    assert json.loads((ROOT / "docs" / "grok-hooks.example.json").read_text(encoding="utf-8")) == expected


@pytest.mark.parametrize("path", MARKDOWN, ids=lambda p: p.name)
def test_relative_links_and_anchors_resolve(path):
    problems = []
    for line in _outside_fences(path):
        for target in _LINK.findall(line):
            if re.match(r"[a-z][a-z0-9+.-]*:", target):
                continue
            name, _, anchor = target.partition("#")
            destination = path if not name else (path.parent / name).resolve()
            if not destination.exists():
                problems.append(f"{target}: no such file")
            elif anchor and destination.suffix == ".md" and anchor not in _anchors(destination):
                problems.append(f"{target}: no such heading")
    assert not problems, problems


@pytest.mark.parametrize("path", MARKDOWN + [ROOT / ".env.example", ROOT / "docs" / "claude-hooks.example.json",
                                             ROOT / "docs" / "grok-hooks.example.json"],
                         ids=lambda p: p.name)
def test_no_private_paths_in_documentation(path):
    text = path.read_text(encoding="utf-8")
    for needle in ("/home/", "/Users/", "/mnt/c/", "C:\\Users"):
        assert needle not in text, f"{path.name} mentions {needle}"
