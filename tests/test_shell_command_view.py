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
    "python script on stdin with an argument": f"python3 - \"$x\" <<'EOF'\nimport os\nos.system('{GATED}')\nEOF",
    "bash -s with an argument": f"bash -s arg <<'EOF'\n{GATED}\nEOF",
    "sh reading a dash": f"sh - <<'EOF'\n{GATED}\nEOF",
    "bash reading /dev/stdin": f"bash /dev/stdin <<'EOF'\n{GATED}\nEOF",
    "bash with a long option and stdin": f"bash --noprofile <<'EOF'\n{GATED}\nEOF",
    "node with the code attached to the flag": f"node --eval=\"require('child_process').execSync('{GATED}')\"",
    "python with clustered flags": f"python3 -uc \"import os; os.system('{GATED}')\"",
    "perl with clustered flags": f"perl -le 'system(\"{GATED}\")'",
    "bash with an option that takes an argument": f"bash -O extglob -c '{GATED}'",
    "unquoted here-document expands a substitution": f"cat <<EOF\n$({GATED})\nEOF",
    "backslash in the delimiter, then a command": f"cat <<E\\OF\ndata\nEOF\n{GATED}",
    "bash option with an argument": f"bash --rcfile /dev/null -c '{GATED}'",
    "python option with an argument": f"python3 -W ignore -c \"import os; os.system('{GATED}')\"",
    "perl option with an argument": f"perl -I lib -e 'system(\"{GATED}\")'",
    "script attached to the flag": f"bash -c'{GATED}'",
    "quoted flag": f'bash "-c" "{GATED}"',
    "clustered interpreter flags with the code attached": f"perl -lne'system(\"{GATED}\")'",
    "clustered python flags with the code attached": f"python3 -bc'import os; os.system(\"{GATED}\")'",
    "code attached to the flag": f"python3 -c'import os; os.system(\"{GATED}\")'",
    "script printed by a command": f"bash -c \"$(printf '%s' '{GATED}')\"",
    "eval of what a command prints": f"eval \"$(printf '%s\\n' '{GATED}')\"",
    "xargs running a shell on piped words": f"printf '%s\\n' '{GATED}' | xargs -I{{}} sh -c {{}}",
    "sourced process substitution": f". <(printf '%s\\n' '{GATED}')",
    "source of a process substitution": f"source <(echo '{GATED}')",
    "nodejs": f"nodejs -e \"require('child_process').execSync('{GATED}')\"",
    "shell that reads a pipe followed by another stage": f"printf '%s\\n' '{GATED}' | sh | cat",
    "script on stdin with positional arguments": f"echo '{GATED}' | bash -s arg1",
    "shell in the middle of a pipeline": f"echo '{GATED}' | bash | cat",
    "heredoc into a shell in the middle of a pipeline": f"cat <<'EOF' | bash | tee out.log\n{GATED}\nEOF",
    "double-quoted expansion with quotes inside, then a command": f"echo \"${{a:-'\"'}}\" && {GATED} && echo done",
    "ANSI-C quoting inside a substitution, then a command": f"echo \"$(echo $'\\')' && {GATED})\"",
    "parameter expansion with a parenthesis inside a substitution": f"echo \"$(echo ${{x%)*}} && {GATED})\"",
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


UNPARSED = {
    "arithmetic shift then a command": f"(( x = 1 << 4 ))\n{GATED}",
    "mistyped delimiter": f"cat <<EOF > f\nbody\nEOT\n{GATED}",
    "carriage return in the terminator": f"cat <<'EOF' > f\nbody\r\nEOF\r\n{GATED}",
    "never terminated": f"cat <<'EOF' > f\nbody {GATED}",
}


@pytest.mark.parametrize("name", sorted(UNPARSED))
def test_a_here_document_that_never_ends_is_not_guessed_at_the_command_is_returned_unchanged(name):
    assert command_view(UNPARSED[name]) == UNPARSED[name], name


def test_a_here_document_opened_on_the_last_line_has_no_body_and_is_fine():
    assert command_view("cat <<EOF") == "cat <<EOF"


@pytest.mark.parametrize("command", [
    'gh "pr" "create"', "gh 'pr' 'create'", '"gh" pr create', "cd \"/mnt/some/dir\" && git -C '/p' push origin main",
    'tool --flag="value" --other=\'x\'',
])
def test_a_quoted_word_the_shell_reads_as_if_it_were_unquoted_is_left_as_written(command):
    assert command_view(command) == command


def test_what_is_piped_into_xargs_becomes_words_of_a_command():
    assert "pr create" in command_view("echo 'pr create' | xargs gh")
    assert "pr create" not in command_view("echo 'pr create' | wc -c")


def test_a_typed_path_contract_is_matched_against_the_whole_text_even_with_a_shell_tool():
    from self_directing_mcp.schemas import ContractRule
    rule = ContractRule(id="p", type="must_not", tools=["Bash"], scope="tool_call", path_guard={"paths": ["**/.env"]})
    assert not targets_command(rule)
    assert targets_command(ContractRule(id="p", type="must_not", tools=["Bash"], scope="tool_call",
                                        path_guard={"paths": ["**/.env"]}, target="command"))


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
