"""Whether a recorded success belongs to the command a ``requires_success`` contract matched.

A host records one exit status per tool call: the status of the whole shell command line. That is the status of the
command a contract named only when nothing in the line replaces it. ``pytest -q || true`` always succeeds,
``pytest | tee log`` reports ``tee`` (unless ``pipefail`` is on), ``pytest; exit 0`` reports ``exit``, and
``pytest --collect-only`` succeeds without running a single test. A success recorded for such a line is not evidence
that the required command passed.

``unverifiable(text, regex)`` answers with the reason a success cannot be trusted, or ``None``. It is deterministic
and conservative in one direction only: a doubtful line is unverifiable (the finding becomes ``unknown``), never
verified. It needs no shell: a small quote-aware lexer splits the line into commands joined by ``;``, ``&&``,
``||``, ``|``, ``&`` and newlines (operators inside quotes, ``$( )``, backticks and escapes are not operators), the
command a contract's regex matched is located (the last one decides the line's status), and what follows it decides.
A line that cannot be lexed (an unterminated quote: not a command a shell would run) is left as it is.

Both evaluators call this one function, so they cannot disagree. Pure Python, no heavy imports.
"""
from __future__ import annotations

import json
import re
import shlex

MAX_CHARS = 100_000
MAX_DEPTH = 4
_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh", "ash"})
_COLLECT_ONLY = frozenset({"--collect-only", "--collectonly", "--co"})
_FAILING_EXIT = re.compile(r"\$\?|\"\$\?\"|\$\{\?\}|[1-9][0-9]*")
_SET_OPTION = re.compile(r"[-+][a-zA-Z]*o\Z")


class _Cmd:
    """One command of a line: its words (quotes removed), its raw text and the operator that follows it."""
    __slots__ = ("words", "raw", "op")

    def __init__(self, words, raw, op):
        self.words, self.raw, self.op = words, raw, op


# ---------------------------------------------------------------------------------------------------------- lexing

def _skip_quoted(src: str, i: int, quote: str) -> int:
    """Index after the quote that closes the one opening at ``i``; -1 when it never closes."""
    j = i + 1
    while j < len(src):
        c = src[j]
        if quote != "'" and c == "\\":
            j += 2
            continue
        if c == quote:
            return j + 1
        j += 1
    return -1


def _skip_parens(src: str, i: int) -> int:
    """Index after the ``)`` matching the ``(`` at ``i`` (quotes and nesting respected); -1 when unbalanced."""
    depth, j = 0, i
    while j < len(src):
        c = src[j]
        if c == "\\":
            j += 2
            continue
        if c in "'\"`":
            j = _skip_quoted(src, j, c)
            if j < 0:
                return -1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return -1


def _skip_braces(src: str, i: int) -> int:
    """Index after the standalone ``}`` closing the standalone ``{`` at ``i``; -1 when unbalanced."""
    depth, j = 0, i
    boundary = " \t\n;&|()"
    while j < len(src):
        c = src[j]
        if c == "\\":
            j += 2
            continue
        if c in "'\"`":
            j = _skip_quoted(src, j, c)
            if j < 0:
                return -1
            continue
        before = src[j - 1] if j else "\n"
        after = src[j + 1] if j + 1 < len(src) else "\n"
        if c == "{" and before in boundary and after in " \t\n":
            depth += 1
        elif c == "}" and before in boundary + "}" and after in boundary + "}":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return -1


def _lex(src: str):
    """Commands of ``src`` in order, or ``None`` when it is not lexable (an unterminated quote or group)."""
    cmds: list[_Cmd] = []
    words: list[str] = []
    cur: list[str] = []
    in_word = False
    start = 0
    i, n = 0, len(src)

    def end_word():
        nonlocal in_word
        if in_word:
            words.append("".join(cur))
            cur.clear()
            in_word = False

    def end_command(op, at, resume):
        nonlocal words, start
        end_word()
        raw = src[start:at].strip()
        if words or raw:
            cmds.append(_Cmd(words, raw, op))
        words = []
        start = resume

    while i < n:
        c = src[i]
        if c == "\\":
            if i + 1 < n and src[i + 1] == "\n":
                i += 2
                continue
            cur.append(src[i + 1] if i + 1 < n else "\\")
            in_word = True
            i += 2
        elif c in "'\"":
            close = _skip_quoted(src, i, c)
            if close < 0:
                return None
            cur.append(src[i + 1:close - 1])
            in_word = True
            i = close
        elif c == "`":
            close = _skip_quoted(src, i, "`")
            if close < 0:
                return None
            cur.append(src[i:close])
            in_word = True
            i = close
        elif c == "$" and src[i + 1:i + 2] == "(":
            close = _skip_parens(src, i + 1)
            if close < 0:
                return None
            cur.append(src[i:close])
            in_word = True
            i = close
        elif c == "(" and not in_word and not words:
            close = _skip_parens(src, i)
            if close < 0:
                return None
            words.append(src[i:close])
            in_word = False
            i = close
        elif c == "{" and not in_word and not words and src[i + 1:i + 2] in (" ", "\t", "\n"):
            close = _skip_braces(src, i)
            if close < 0:
                return None
            words.append(src[i:close])
            i = close
        elif c in " \t":
            end_word()
            i += 1
        elif c == "\n":
            end_command("\n", i, i + 1)
            i += 1
        elif c == "#" and not in_word:
            while i < n and src[i] != "\n":
                i += 1
        elif c == ";":
            end_command(";", i, i + (2 if src[i + 1:i + 2] == ";" else 1))
            i += 2 if src[i + 1:i + 2] == ";" else 1
        elif c == "&":
            if src[i + 1:i + 2] == "&":
                end_command("&&", i, i + 2)
                i += 2
            elif (i and src[i - 1] in "<>") or src[i + 1:i + 2] == ">":
                cur.append(c)               # a redirection (2>&1, &>file), not an operator
                in_word = True
                i += 1
            else:
                end_command("&", i, i + 1)
                i += 1
        elif c == "|":
            if src[i + 1:i + 2] == "|":
                end_command("||", i, i + 2)
                i += 2
            else:
                step = 2 if src[i + 1:i + 2] == "&" else 1   # ``|&`` also pipes
                end_command("|", i, i + step)
                i += step
        else:
            cur.append(c)
            in_word = True
            i += 1
    end_command(None, n, n)
    if cmds and cmds[-1].op in (";", "\n", "&&", "||", "|"):
        cmds[-1].op = None                 # a trailing separator or a dangling operator joins nothing
    return cmds


# ---------------------------------------------------------------------------------------------------- the analysis

def _group_inner(word: str) -> str | None:
    if word.startswith("(") and word.endswith(")"):
        return word[1:-1]
    if word.startswith("{") and word.endswith("}"):
        return word[1:-1].rstrip().rstrip(";")
    return None


def _plainly_fails(cmd: _Cmd, depth: int) -> bool:
    """The command after ``||`` keeps a failure: ``exit`` with no or a nonzero status, ``false``, or a group ending so."""
    if not cmd.words:
        return False
    head = cmd.words[0]
    if head in ("exit", "return"):
        return len(cmd.words) == 1 or bool(_FAILING_EXIT.fullmatch(cmd.words[1]))
    if head == "false":
        return True
    inner = _group_inner(head)
    if inner is not None and depth < MAX_DEPTH:
        cmds = _lex(inner)
        return bool(cmds) and _plainly_fails(cmds[-1], depth + 1)
    return False


def _sets_pipefail(words: list[str], current: bool) -> bool:
    if not words or words[0] != "set":
        return current
    for index in range(1, len(words) - 1):
        if _SET_OPTION.fullmatch(words[index]) and words[index + 1] == "pipefail":
            current = words[index][0] == "-"
    return current


def _basename(word: str) -> str:
    return word.rsplit("/", 1)[-1]


def _pytest(words: list[str]) -> bool:
    return any(_basename(word) in ("pytest", "py.test") for word in words)


def _collect_only(words: list[str]) -> bool:
    return _pytest(words) and any(word in _COLLECT_ONLY or word.startswith("--collect-only=") for word in words)


def _wrapped_script(words: list[str]):
    """``(script, pipefail)`` of ``bash -c '<script>'`` (also behind ``sudo``, ``env``, ``timeout``...), else None."""
    for index, word in enumerate(words):
        if _basename(word) not in _SHELLS:
            continue
        pipefail, position = False, index + 1
        while position < len(words):
            option = words[position]
            if not option.startswith(("-", "+")) or option == "--":
                return None                # a script file or an operand: nothing inline to read
            if _SET_OPTION.fullmatch(option):
                if words[position + 1:position + 2] == ["pipefail"]:
                    pipefail = option[0] == "-"
                position += 2              # ``-o name``: the next word is the option's name
                continue
            if re.fullmatch(r"-[a-zA-Z]+", option) and "c" in option and position + 1 < len(words):
                return words[position + 1], pipefail
            position += 1
        return None
    return None


def _analyze(src: str, regex, depth: int, pipefail: bool) -> str | None:
    cmds = _lex(src)
    if not cmds:
        return None
    matched = [i for i, cmd in enumerate(cmds) if regex is None or regex.search(cmd.raw)] or list(range(len(cmds)))
    state, flags = pipefail, []
    for cmd in cmds:                       # ``pipefail`` as it stands before each command
        flags.append(state)
        state = _sets_pipefail(cmd.words, state)
    m = matched[-1]                        # the last matched command is the one whose status the line can report
    cmd = cmds[m]
    if _collect_only(cmd.words):
        return "collect_only"
    if cmd.op == "|" and not flags[m]:
        return "pipeline_without_pipefail"
    end = m
    while end < len(cmds) and cmds[end].op == "|":
        end += 1                           # the pipeline's last stage; what follows is joined to the whole pipeline
    t = end
    while t < len(cmds) and cmds[t].op is not None:
        op = cmds[t].op
        rest = cmds[t + 1:]
        if op == "&":
            return "background"
        if op in (";", "\n"):
            if any(item.words for item in rest):
                if not (len(rest) == 1 and rest[0].words[:1] == ["exit"] and rest[0].words[1:] in (["$?"], ["\"$?\""])):
                    return "followed_by_command"
            break
        if op == "||" and rest and not _plainly_fails(rest[0], depth):
            return "or_masks_failure"
        t += 1
    if depth < MAX_DEPTH:
        inner = _group_inner(cmd.words[0]) if cmd.words else None
        wrapped = _wrapped_script(cmd.words)
        if inner is not None:
            return _analyze(inner, regex, depth + 1, flags[m])
        if wrapped is not None:
            return _analyze(wrapped[0], regex, depth + 1, wrapped[1] or flags[m])
    return None


def _leaves(value, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _leaves(item, out)
    elif isinstance(value, list):
        if value and all(isinstance(item, str) for item in value):
            out.append(shlex.join(value))      # an argv: the line a shell would see
        for item in value:
            _leaves(item, out)


def shell_texts(text: str) -> list[str]:
    """The strings of a tool call that a shell could run: the leaves of its JSON arguments, or the raw argument text."""
    body = text.split("] ", 1)[1] if text.startswith("[") and "] " in text else text
    try:
        value = json.loads(body)
    except ValueError:
        return [body]
    out: list[str] = []
    _leaves(value, out)
    return list(dict.fromkeys(out)) or [body]


def unverifiable(text: str, regex) -> str | None:
    """Why a success recorded for the call ``text`` cannot be attributed to what ``regex`` matched, or ``None``.

    ``regex`` is the contract's compiled pattern (``search``); without one every command of the line counts as matched.
    """
    if len(text) > MAX_CHARS:
        return "too_long_to_analyze"
    leaves = shell_texts(text)
    hit = [leaf for leaf in leaves if regex is None or regex.search(leaf)] or leaves
    for leaf in hit:
        why = _analyze(leaf, regex, 0, False)
        if why:
            return why
    return None


REASONS = {
    "collect_only": "pytest --collect-only collects tests without running them",
    "pipeline_without_pipefail": "in a pipeline without `set -o pipefail` the status is the last stage's",
    "background": "a command sent to the background reports no status",
    "followed_by_command": "a later command in the same line replaces its status",
    "or_masks_failure": "a `||` branch replaces its failure",
    "too_long_to_analyze": "the command is too long to analyze",
}
