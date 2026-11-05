"""The command view of shell calls: data a shell does not run is out, anything it runs stays in.

Synthetic commands only. A contract about commands (``tools: ["Bash"]`` or ``target: "command"``) is matched against
the command view instead of the raw text, so a heredoc body, a quoted pattern or a commit message that merely names a
gated program no longer matches, while every way of really running it still does.
"""
from __future__ import annotations

import json

import pytest

from self_directing_mcp.audit import shellview
from self_directing_mcp.audit.shellview import command_view, targets_command, tool_view
from self_directing_mcp.schemas import ContractRule

GATED = "gh pr create"

# Text only: none of these runs the gated program.
DATA = {
    "heredoc written to a file": f"cat > notes.md <<'EOF'\nrun {GATED} later\nEOF\nls",
    "heredoc with a dash and tab-indented terminator": f"cat <<-EOF > f\n\t{GATED}\n\tEOF\necho done",
    "unquoted heredoc delimiter": f"cat <<EOF > f\n{GATED}\nEOF",
    "grep pattern in single quotes": "ps aux | grep -E 'gh pr (create|merge)' | wc -l",
    "grep pattern in double quotes": f'grep -n "{GATED}" script.sh',
    "echo": f"echo '{GATED}' > out.txt",
    "commit message": f'git commit -m "docs: mention {GATED} in the guide"',
    "commit message from a heredoc substitution": f"git commit -m \"$(cat <<'EOF'\nfix: don't {GATED}\nEOF\n)\"",
    "comment": f"ls  # {GATED}\npwd",
    "issue body": f'gh issue comment 5 --body "{GATED}"',
    "heredoc to a program that only stores it": f"tee notes.txt <<'EOF'\n{GATED}\nEOF",
    "script operand with quoted data": f"bash run.sh '{GATED}'",
    "here-string to a data consumer": f"wc -w <<< '{GATED}'",
}

# The same text, run: each of these really executes the gated program.
RUN = {
    "plain": GATED,
    "after a separator": f"cd /tmp && {GATED}",
    "after a pipe": f"echo x | {GATED}",
    "with an environment prefix": f"FOO=1 {GATED}",
    "behind a wrapper": f"timeout 5 {GATED}",
    "bash -c": f"bash -c '{GATED} --title x'",
    "sh -c double quoted": f'sh -c "{GATED} --title x"',
    "bash -lc cluster": f"bash -lc '{GATED}'",
    "behind env and bash -c": f"env A=1 bash -c '{GATED}'",
    "command substitution": f"x=$({GATED})",
    "command substitution in an argument": f"echo \"$({GATED})\"",
    "backticks": f"echo `{GATED}`",
    "process substitution": f"diff <({GATED}) other",
    "subshell": f"({GATED})",
    "eval": f'eval "{GATED}"',
    "ssh remote command": f"ssh host '{GATED}'",
    "heredoc to bash": f"bash <<EOF\n{GATED}\nEOF",
    "heredoc to sh through ssh": f"ssh host sh <<'EOF'\n{GATED}\nEOF",
    "heredoc piped to bash": f"cat <<'EOF' | bash\n{GATED}\nEOF",
    "echo piped to sh": f"echo '{GATED}' | sh",
    "here-string to bash": f"bash <<< '{GATED}'",
    "python -c": f"python3 -c \"import os; os.system('{GATED}')\"",
    "python script on stdin": f"python3 - <<'EOF'\nimport os\nos.system('{GATED}')\nEOF",
    "xargs with a shell": f"ls | xargs -I{{}} sh -c '{GATED} {{}}'",
    "find -exec with a shell": f"find . -name x -exec sh -c '{GATED}' \\;",
    "heredoc inside a substitution run by a shell": f"x=\"$(bash <<'EOF'\n{GATED}\nEOF\n)\"",
    "nested substitution": f"echo $(echo $({GATED}))",
    "after a comment line": f"# note\n{GATED}",
    "while loop body": f"while true; do {GATED}; done",
}


@pytest.mark.parametrize("name", sorted(DATA))
def test_text_that_is_not_run_leaves_the_view(name):
    assert GATED not in command_view(DATA[name]), name


@pytest.mark.parametrize("name", sorted(RUN))
def test_every_way_of_running_the_program_stays_in_the_view(name):
    assert GATED in command_view(RUN[name]), name


def test_the_view_only_removes_text_so_a_regex_written_for_the_raw_text_still_matches_what_runs():
    command = f"cd /tmp && git -C '/p' push origin main && echo \"done\"\ncat <<'EOF' > f\nbody\nEOF\nls -la"
    viewed = command_view(command)
    assert viewed.replace("''", "'x'").count("push origin main") == 1
    assert "cd /tmp && git -C" in viewed and viewed.rstrip().endswith("ls -la") and "body" not in viewed


def test_a_command_that_does_not_parse_or_is_too_large_is_returned_unchanged():
    broken = f"echo '{GATED}"                                    # unterminated quote
    assert command_view(broken) == broken
    deep = "$(" * (shellview.MAX_DEPTH + 3) + GATED + ")" * (shellview.MAX_DEPTH + 3)
    assert GATED in command_view(deep)
    huge = f"echo '{GATED}' " + "x" * shellview.MAX_CHARS
    assert command_view(huge) == huge


def test_nesting_deeper_than_the_interpreter_allows_is_returned_unchanged_not_raised():
    nested = '"$(' * 3000 + GATED + ')"' * 3000
    assert command_view(nested) == nested


def test_an_unlisted_program_is_a_data_consumer_and_listing_it_makes_its_arguments_commands(monkeypatch):
    unlisted = f"watchdog-runner '{GATED}'"
    assert GATED not in command_view(unlisted)
    data = json.loads(json.dumps(shellview.semantics()))
    data["string_carriers"].append("watchdog-runner")
    monkeypatch.setattr(shellview, "semantics", lambda: data)
    assert GATED in command_view(unlisted)


def _text(tool, **payload):
    return f"[tool_call:{tool}] " + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def test_tool_view_rewrites_the_command_of_a_shell_tool_only():
    text = _text("Bash", command=f"cat > f <<'EOF'\n{GATED}\nEOF", timeout=5)
    viewed = tool_view(text)
    assert GATED not in viewed and viewed.startswith("[tool_call:Bash] ") and '"timeout": 5' in viewed
    assert json.loads(viewed.split("] ", 1)[1])["command"] == "cat > f <<'EOF'\nEOF"
    proposed = text.replace("[tool_call:", "[proposed_tool_call:")
    assert GATED not in tool_view(proposed) and tool_view(proposed).startswith("[proposed_tool_call:Bash] ")
    other = _text("Write", file_path="f", content=f"{GATED}")
    assert tool_view(other) == other                              # not a shell tool: whole text
    monitor = _text("Monitor", command=f"echo '{GATED}'")
    assert GATED not in tool_view(monitor)
    assert tool_view("[tool_call:Bash] not json") == "[tool_call:Bash] not json"
    assert tool_view("plain text") == "plain text"
    unchanged = _text("Bash", command=GATED)
    assert tool_view(unchanged) is unchanged or tool_view(unchanged) == unchanged


def test_a_contract_targets_commands_when_it_says_so_or_names_only_shell_tools():
    base = {"id": "c", "type": "must_not", "regex": GATED}
    assert targets_command(ContractRule(**base, tools=["Bash"], scope="tool_call"))
    assert targets_command(ContractRule(**base, tools=["bash", "Monitor"], scope="tool_call"))
    assert not targets_command(ContractRule(**base, tools=["Bash", "Write"], scope="tool_call"))
    assert not targets_command(ContractRule(**base, scope="tool_call"))
    assert targets_command(ContractRule(**base, scope="tool_call", target="command"))
    assert not targets_command(ContractRule(**base, tools=["Bash"], scope="tool_call", target="text"))
    assert targets_command({"tools": ["Bash"]}) and not targets_command({"tools": ["Edit"]})
