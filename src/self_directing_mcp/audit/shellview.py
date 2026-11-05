"""The command view of a shell tool call: what a contract about commands is matched against.

A prohibition such as "never run X" is about a command that runs. A contract regex that is matched against the raw text
of a shell call also fires on text that is only quoted: the body of a here-document that is written to a file, the
pattern of ``grep -E 'a|b'``, a commit message, a command named inside a heredoc that a test script contains. The view
keeps what a shell would run (the command words, their unquoted arguments, command and process substitutions, scripts
handed to a shell or an interpreter) and removes what it only passes along as data (quoted arguments, here-document
bodies, comments). Nothing else changes: the text is the original with spans taken out, so a contract regex written for
the raw text still matches the commands that really run.

* The shell is parsed (quotes, escapes, ``$(...)``, backticks, ``<(...)``, here-documents with their delimiters, here-strings,
  pipelines, ``;`` ``&&`` ``||``, comments), not searched with a pattern.
* What a program does with its arguments is data: ``resources/shell_semantics.json`` names the shells (``bash -c '...'``,
  a script operand or standard input), the interpreters with their code flags (``python -c '...'``), and the programs that
  run their words as a command line (``ssh host '...'``, ``eval``). A quoted argument of any other program is data. In a
  pipeline whose later stage reads a script from standard input, the earlier stages feed it, so their data is kept.
* Anything that cannot be parsed (an unterminated quote) or is nested too deep is returned unchanged: matching more is the
  safe side of a prohibition.

Standard library only; hook processes load it.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

MAX_CHARS = 200_000
MAX_DEPTH = 8
# Words that open or close a compound command; they are grammar, not the name of the program that follows them.
_GRAMMAR = frozenset({"if", "then", "elif", "else", "fi", "for", "while", "until", "do", "done", "case", "esac",
                      "select", "function", "{", "}", "!", "time", "coproc", "in"})
_RESOURCE = Path(__file__).resolve().parent.parent / "resources" / "shell_semantics.json"


@functools.lru_cache(maxsize=1)
def semantics() -> dict:
    """The shell semantics data file (cached)."""
    return json.loads(_RESOURCE.read_text(encoding="utf-8"))


def shell_tools() -> dict[str, str]:
    """``{host tool name: the argument holding its shell command line}``."""
    return dict(semantics()["shell_tools"])


# ---------------------------------------------------------------------------------------------------- tokenizing

class _Part:
    """One piece of a word: ``lit`` (unquoted text), ``single``, ``double``, ``ansi`` (quoted text, ``inner`` is the
    span between the quotes) or ``subst`` (a command or process substitution, ``inner`` is its script)."""
    __slots__ = ("kind", "start", "end", "inner_start", "inner_end")

    def __init__(self, kind, start, end, inner_start=None, inner_end=None):
        self.kind, self.start, self.end = kind, start, end
        self.inner_start = start if inner_start is None else inner_start
        self.inner_end = end if inner_end is None else inner_end


class _Word:
    __slots__ = ("parts", "start", "end", "herestring")

    def __init__(self, start):
        self.parts, self.start, self.end, self.herestring = [], start, start, False


class _Heredoc:
    __slots__ = ("delimiter", "strip_tabs", "body_start", "body_end", "terminator_end")

    def __init__(self, delimiter, strip_tabs):
        self.delimiter, self.strip_tabs = delimiter, strip_tabs
        self.body_start = self.body_end = self.terminator_end = 0


class _Command:
    __slots__ = ("words", "heredocs", "op")

    def __init__(self):
        self.words: list[_Word] = []
        self.heredocs: list[_Heredoc] = []
        self.op: str | None = None       # the operator that follows (``|`` joins the next command into a pipeline)


def _close_quote(src: str, i: int, quote: str) -> int:
    """Index after the quote closing the one at ``i`` (backslash escapes outside single quotes); -1 when none."""
    j = i + 1
    while j < len(src):
        c = src[j]
        if c == "\\" and quote != "'":
            j += 2
            continue
        if c == quote:
            return j + 1
        if quote == '"' and c == "$" and src[j + 1:j + 2] == "(":
            nested = _close_paren(src, j + 1)
            if nested < 0:
                return -1
            j = nested
            continue
        if quote == '"' and c == "`":
            nested = _close_quote(src, j, "`")
            if nested < 0:
                return -1
            j = nested
            continue
        j += 1
    return -1


def _close_paren(src: str, i: int) -> int:
    """Index after the ``)`` matching the ``(`` at ``i``; quotes, escapes, comments, nesting and here-document bodies
    (which may hold unbalanced quotes or parentheses) respected; -1 if none."""
    depth, j, n = 0, i, len(src)
    pending: list[_Heredoc] = []
    while j < n:
        c = src[j]
        if c == "\\":
            j += 2
            continue
        if c in "'\"`":
            j = _close_quote(src, j, c)
            if j < 0:
                return -1
            continue
        if c == "#" and (j == 0 or src[j - 1] in " \t\n;&|("):
            while j < n and src[j] != "\n":
                j += 1
            continue
        if c == "<" and src[j:j + 2] == "<<" and src[j:j + 3] != "<<<":
            strip = src[j + 2:j + 3] == "-"
            k = j + (3 if strip else 2)
            while k < n and src[k] in " \t":
                k += 1
            start = k
            while k < n and src[k] not in " \t\n;&|()<>":
                if src[k] in "'\"":
                    k = _close_quote(src, k, src[k])
                    if k < 0:
                        return -1
                else:
                    k += 2 if src[k] == "\\" else 1
            doc = _Heredoc(_delimiter_text(src[start:k]), strip)
            pending.append(doc)
            j = k
            continue
        if c == "\n" and pending:
            j = _read_heredoc_bodies(src, j + 1, pending)
            pending = []
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return -1


def _delimiter_text(raw: str) -> str:
    """The delimiter of a here-document as the shell compares it: quotes and backslashes removed."""
    return "".join(ch for ch in raw if ch not in "'\"\\")


def _close_brace(src: str, i: int) -> int:
    """Index after the ``}`` closing the ``${`` whose brace is at ``i``; -1 if none."""
    depth, j, n = 0, i, len(src)
    while j < n:
        c = src[j]
        if c == "\\":
            j += 2
            continue
        if c in "'\"`":
            j = _close_quote(src, j, c)
            if j < 0:
                return -1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return -1


def _read_heredoc_bodies(src: str, pos: int, pending: list[_Heredoc]) -> int:
    """Consume the bodies of the here-documents opened on the line that just ended; return the new position."""
    for doc in pending:
        doc.body_start = pos
        while True:
            if pos >= len(src):
                doc.body_end = doc.terminator_end = len(src)         # never terminated: the rest is the body
                break
            end = src.find("\n", pos)
            line_end = len(src) if end < 0 else end
            line = src[pos:line_end]
            if (line.lstrip("\t") if doc.strip_tabs else line) == doc.delimiter:
                doc.body_end = pos
                doc.terminator_end = line_end
                pos = line_end + 1 if end >= 0 else line_end
                break
            pos = line_end + 1 if end >= 0 else len(src)
    return pos


def _parse(src: str) -> tuple[list[_Command], list[tuple[int, int]]] | None:
    """The commands of ``src`` with the spans of their words, here-documents and substitutions, and the spans of its
    comments; None when it does not parse."""
    cmds: list[_Command] = []
    comments: list[tuple[int, int]] = []
    cmd = _Command()
    word: _Word | None = None
    pending: list[_Heredoc] = []
    want_delimiter: _Heredoc | None = None
    n, i = len(src), 0

    def end_word(at):
        nonlocal word, want_delimiter
        if word is None:
            return
        word.end = at
        if want_delimiter is not None:
            text = "".join(src[p.inner_start:p.inner_end] if p.kind != "lit" else src[p.start:p.end] for p in word.parts)
            want_delimiter.delimiter = text
            want_delimiter = None
            word = None
            return
        cmd.words.append(word)
        word = None

    def end_command(op, at):
        nonlocal cmd
        end_word(at)
        if cmd.words or cmd.heredocs:
            cmd.op = op
            cmds.append(cmd)
        elif cmds and op is not None and cmds[-1].op is None:
            cmds[-1].op = op
        cmd = _Command()

    def add(part):
        nonlocal word
        if word is None:
            word = _Word(part.start)
        word.parts.append(part)

    while i < n:
        c = src[i]
        if c == "\\":
            if src[i + 1:i + 2] == "\n":
                i += 2
                continue
            add(_Part("lit", i, min(n, i + 2)))
            i += 2
        elif c == "'":
            close = _close_quote(src, i, "'")
            if close < 0:
                return None
            add(_Part("single", i, close, i + 1, close - 1))
            i = close
        elif c == '"':
            close = _close_quote(src, i, '"')
            if close < 0:
                return None
            add(_Part("double", i, close, i + 1, close - 1))
            i = close
        elif c == "$" and src[i + 1:i + 2] == "'":
            close = _close_quote(src, i + 1, "'")
            j = i + 2
            while j < n and src[j] != "'":          # $'...' honours backslash escapes
                j += 2 if src[j] == "\\" else 1
            if j >= n:
                return None
            add(_Part("ansi", i, j + 1, i + 2, j))
            i = j + 1
        elif c == "`":
            close = _close_quote(src, i, "`")
            if close < 0:
                return None
            add(_Part("subst", i, close, i + 1, close - 1))
            i = close
        elif c == "$" and src[i + 1:i + 2] == "(":
            close = _close_paren(src, i + 1)
            if close < 0:
                return None
            add(_Part("subst", i, close, i + 2, close - 1))
            i = close
        elif c == "$" and src[i + 1:i + 2] == "{":
            close = _close_brace(src, i + 1)
            if close < 0:
                return None
            add(_Part("lit", i, close))
            i = close
        elif c in "<>" and src[i + 1:i + 2] == "(" and word is None:
            close = _close_paren(src, i + 1)
            if close < 0:
                return None
            add(_Part("subst", i, close, i + 2, close - 1))
            i = close
        elif c == "<" and src[i:i + 3] == "<<<":
            end_word(i)
            word = _Word(i)
            word.herestring = True
            add(_Part("lit", i, i + 3))
            end_word(i + 3)
            i += 3
        elif c == "<" and src[i:i + 2] == "<<":
            end_word(i)
            strip = src[i + 2:i + 3] == "-"
            doc = _Heredoc("", strip)
            cmd.heredocs.append(doc)
            pending.append(doc)
            want_delimiter = doc
            i += 3 if strip else 2
            while i < n and src[i] in " \t":
                i += 1
        elif c in " \t":
            end_word(i)
            i += 1
        elif c == "\n":
            end_command("\n", i)
            i += 1
            if pending:
                i = _read_heredoc_bodies(src, i, pending)
                pending = []
        elif c == "#" and word is None:
            start = i
            while i < n and src[i] != "\n":
                i += 1
            comments.append((start, i))
        elif c == ";":
            step = 2 if src[i + 1:i + 2] == ";" else 1
            end_command(src[i:i + step], i)
            i += step
        elif c == "&":
            if src[i + 1:i + 2] == "&":
                end_command("&&", i)
                i += 2
            elif (i and src[i - 1] in "<>") or src[i + 1:i + 2] == ">":
                add(_Part("lit", i, i + 1))         # a redirection (2>&1, &>file), not an operator
                i += 1
            else:
                end_command("&", i)
                i += 1
        elif c == "|":
            if src[i + 1:i + 2] == "|":
                end_command("||", i)
                i += 2
            else:
                step = 2 if src[i + 1:i + 2] == "&" else 1
                end_command("|", i)
                i += step
        elif c in "()" and word is None:
            end_command(c, i)
            i += 1
        else:
            add(_Part("lit", i, i + 1))
            i += 1
    end_command(None, n)
    if pending:
        _read_heredoc_bodies(src, n, pending)        # a here-document opened on the last line has no body
    return cmds, comments


# ------------------------------------------------------------------------------------------------------ planning

def _word_text(src: str, word: _Word) -> str:
    """The text a shell would read the word as, for deciding which program it names (quotes removed, no expansion)."""
    out = []
    for part in word.parts:
        if part.kind == "lit":
            text = src[part.start:part.end]
            out.append(text[1:] if text.startswith("\\") and len(text) > 1 else text)
        elif part.kind == "subst":
            out.append(src[part.start:part.end])
        else:
            out.append(src[part.inner_start:part.inner_end])
    return "".join(out)


def _program(text: str) -> str:
    """The program name of a word: its last path component without a trailing version (``python3.12`` is ``python``)."""
    return text.rsplit("/", 1)[-1].rstrip("0123456789.")


def _unquote(raw: str, kind: str) -> str:
    """The text inside quotes after the shell removes the escapes (single quotes: none)."""
    if kind == "single":
        return raw
    if kind == "ansi":
        return raw.replace("\\'", "'").replace("\\\\", "\\")
    out, i = [], 0
    while i < len(raw):
        if raw[i] == "\\" and i + 1 < len(raw) and raw[i + 1] in '$`"\\\n':
            if raw[i + 1] != "\n":
                out.append(raw[i + 1])
            i += 2
        else:
            out.append(raw[i])
            i += 1
    return "".join(out)


def _substitutions_in_double(raw: str) -> list[tuple[int, int, int, int]]:
    """``(start, end, inner_start, inner_end)`` of the command substitutions inside the text of a double-quoted string."""
    found, i = [], 0
    while i < len(raw):
        c = raw[i]
        if c == "\\":
            i += 2
        elif c == "$" and raw[i + 1:i + 2] == "(":
            close = _close_paren(raw, i + 1)
            if close < 0:
                break
            found.append((i, close, i + 2, close - 1))
            i = close
        elif c == "`":
            close = _close_quote(raw, i, "`")
            if close < 0:
                break
            found.append((i, close, i + 1, close - 1))
            i = close
        else:
            i += 1
    return found


class _Plan:
    """Which words of a command are scripts for a shell, code for an interpreter, or run as a command line, and whether
    the command itself runs what it reads from standard input."""

    def __init__(self, src: str, cmd: _Command, sem: dict):
        self.shell_words: set[int] = set()
        self.code_words: set[int] = set()
        self.stdin_kind: str | None = None          # "shell" / "code": a shell or interpreter reading its script from stdin
        texts = [_word_text(src, w) for w in cmd.words]
        first = 0
        while first < len(texts) and (texts[first] in _GRAMMAR or _is_assignment(texts[first])):
            first += 1
        shells = set(sem["shells"])
        interpreters = {name: [flag.lower() for flag in flags] for name, flags in sem["interpreters"].items()}
        carriers = set(sem["string_carriers"])
        for index in range(first, len(texts)):
            program = _program(texts[index])
            if program in carriers:
                self.shell_words.update(w for w in range(index + 1, len(texts)) if _quoted(cmd.words[w]))
                if program == "eval":
                    self.shell_words.update(range(index + 1, len(texts)))
            if program in shells or program in interpreters:
                is_shell = program in shells
                script = self._script_word(texts, index, is_shell, ["-c"] if is_shell else interpreters[program])
                if script is not None:
                    (self.shell_words if is_shell else self.code_words).add(script)
                elif not any(not t.startswith("-") and not cmd.words[index + 1 + k].herestring
                             and not (index + k >= 0 and cmd.words[index + k].herestring)
                             for k, t in enumerate(texts[index + 1:])):
                    self.stdin_kind = self.stdin_kind or ("shell" if is_shell else "code")
        for number, word in enumerate(cmd.words[:-1]):
            if word.herestring and self.stdin_kind is not None:        # a here-string is the script of a shell
                (self.shell_words if self.stdin_kind == "shell" else self.code_words).add(number + 1)

    @staticmethod
    def _script_word(texts, index, is_shell, flags):
        position = index + 1
        while position < len(texts):
            option = texts[position]
            if is_shell:
                if option == "--" or not option.startswith(("-", "+")):
                    return None                  # a script file or another operand: nothing inline
                if len(option) > 1 and option[1] != "-" and "c" in option[1:]:
                    return position + 1 if position + 1 < len(texts) else None
                position += 2 if option in ("-o", "+o") else 1
            else:
                lowered = option.lower()
                if lowered in flags or (lowered.startswith("--") and lowered.split("=")[0] in flags and "=" in lowered):
                    return position + 1 if position + 1 < len(texts) else None
                if not option.startswith("-"):
                    return None
                position += 1
        return None


def _is_assignment(text: str) -> bool:
    name, sep, _value = text.partition("=")
    return bool(sep) and name.isidentifier()


def _quoted(word: _Word) -> bool:
    return any(part.kind in ("single", "double", "ansi") for part in word.parts)


def _view(src: str, depth: int, sem: dict) -> str:
    if depth > MAX_DEPTH or len(src) > MAX_CHARS:
        return src
    parsed = _parse(src)
    if parsed is None:
        return src
    cmds, comments = parsed
    edits: list[tuple[int, int, str]] = [(start, end, "") for start, end in comments]    # a comment is never run
    plans = [_Plan(src, cmd, sem) for cmd in cmds]
    # A pipeline whose last stage runs a script from standard input executes what the earlier stages print.
    feeds = [False] * len(cmds)
    start = 0
    while start < len(cmds):
        end = start
        while end < len(cmds) - 1 and cmds[end].op in ("|", "|&"):
            end += 1
        if plans[end].stdin_kind is not None and end > start:
            feeds[start:end] = [True] * (end - start)
        start = end + 1
    for cmd, plan, fed in zip(cmds, plans, feeds):
        for number, word in enumerate(cmd.words):
            for part in word.parts:
                if part.kind == "subst":
                    edits.append((part.inner_start, part.inner_end,
                                  _view(src[part.inner_start:part.inner_end], depth + 1, sem)))
                elif part.kind in ("single", "double", "ansi"):
                    raw = src[part.inner_start:part.inner_end]
                    if number in plan.code_words or fed:
                        continue                                      # code or a script's input: kept as it is
                    if number in plan.shell_words:
                        edits.append((part.inner_start, part.inner_end,
                                      _view(_unquote(raw, part.kind), depth + 1, sem)))
                    elif part.kind == "double":
                        kept = "".join("$(" + _view(raw[a:b], depth + 1, sem) + ")"
                                       for _s, _e, a, b in _substitutions_in_double(raw))
                        edits.append((part.inner_start, part.inner_end, kept))
                    else:
                        edits.append((part.inner_start, part.inner_end, ""))
        for doc in cmd.heredocs:
            body = src[doc.body_start:doc.body_end]
            if fed or plan.stdin_kind == "code":
                continue                                  # an interpreter reads it, or a later stage does: kept
            if plan.stdin_kind == "shell":
                edits.append((doc.body_start, doc.body_end, _view(body, depth + 1, sem)))
            else:
                edits.append((doc.body_start, doc.body_end, ""))
    return _apply(src, edits)


def _apply(src: str, edits: list[tuple[int, int, str]]) -> str:
    """The text with each span replaced; spans that overlap an earlier (outer) edit are dropped."""
    out, position = [], 0
    for start, end, text in sorted(edits, key=lambda e: (e[0], -e[1])):
        if start < position:
            continue
        out.append(src[position:start])
        out.append(text)
        position = end
    out.append(src[position:])
    return "".join(out)


def command_view(command: str) -> str:
    """``command`` with the data a shell does not run taken out (see the module text). Unchanged when it cannot be parsed."""
    return _view(command, 0, semantics())


# ------------------------------------------------------------------------------------------------- tool-call text

def targets_command(rule: Any) -> bool:
    """Whether a contract is matched against the command view of shell calls: ``target`` says so, and a contract that
    does not say is about commands when every tool it names is a shell tool (``tools: ["Bash"]``). A contract with no
    tool filter, or with other tools, is about the whole text of the call."""
    get = rule.get if isinstance(rule, dict) else lambda name, default=None: getattr(rule, name, default)
    target = get("target")
    if target is not None:
        return target == "command"
    tools = get("tools")
    shell = {name.lower() for name in shell_tools()}
    return bool(tools) and all(str(tool).lower() in shell for tool in tools)



_HEADERS = ("[tool_call:", "[proposed_tool_call:")


@functools.lru_cache(maxsize=8192)
def tool_view(text: str) -> str:
    """The text of a tool call (``[tool_call:NAME] <payload>``) as a contract about commands is matched against it.

    For a shell tool (``shell_tools`` in the semantics data) the payload's command line becomes its command view; any
    other call, or text that is not a tool call, is returned unchanged."""
    if not text.startswith(_HEADERS):
        return text
    close = text.find("] ")
    if close < 0:
        return text
    name = text[text.index(":") + 1:close]
    field = shell_tools().get(next((tool for tool in shell_tools() if tool.lower() == name.lower()), ""))
    if field is None:
        return text
    try:
        payload: Any = json.loads(text[close + 2:])
    except ValueError:
        return text
    if not isinstance(payload, dict) or not isinstance(payload.get(field), str):
        return text
    viewed = command_view(payload[field])
    if viewed == payload[field]:
        return text
    payload[field] = viewed
    return text[:close + 2] + json.dumps(payload, ensure_ascii=False, sort_keys=True)
