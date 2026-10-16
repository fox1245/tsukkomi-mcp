"""Typed path contracts: ``path_guard`` lowered to the one thing every evaluator understands, a contract regex.

A ``must_not`` contract about a FILE should say which file, not which shell spelling reads it. The contract regex
engine matches the text of a tool call, ``[tool_call:<name>] <JSON arguments>``, and every host renders its own tool
names and argument keys (Claude Code ``Read{file_path}``, AGY ``view_file{AbsolutePath}``, OMP ``read{path}``, Codex
``shell{command}`` and ``apply_patch``, Grok normalized to Claude's names by ``grok_tools``). A regex written around
``cat`` never sees them. ``path_guard`` names the path once::

    {"paths": [".env", "~/.ssh/id_*"], "except": [".env.example"], "modes": "any" | "read"}

and this module turns it into a regex through a versioned per-host tool map (``TOOL_MAP_VERSION``). The typed form is
what is stored, hashed, exported and compared; the regex is derived in memory when a contract is loaded
(``schemas.ContractRule``), so a better tool map reaches every adopted contract without rewriting any of them and
without changing a bundle's pin. This module has no heavy imports (the hooks load it): standard library only.

Semantics, version 1
--------------------
``modes``
    ``any`` (default): the path appears as a path argument of ANY tool call. ``read``: the path is read, that is the
    path argument of a file-read tool (Read, view_file, read, read_file) or of a content or name search (Grep, Glob),
    or, inside a shell command, an operand of a reader (cat, head, tail, less, more, sed, awk, grep, cp, xxd,
    base64, ``source``, ``.``, ``type``, Get-Content, ``git show`` ...), the source of a ``<`` redirection, or the
    argument of an interpreter's inline code (``python -c``, ``node -e``, a heredoc). Writes (Edit, Write, MultiEdit,
    NotebookEdit, write_to_file, replace_file_content, apply_patch, ``> file``) are matched by ``any`` and never by
    ``read``. ``write`` is not a mode in version 1 (deferred); validation rejects it.

A path token (an entry of ``paths`` or ``except``) is a glob: ``*`` stands for any characters inside one path
segment, ``**`` for any number of segments, ``?`` for one character, ``[...]`` for a class (``[!x]`` negates).
Forward and back slashes are equivalent, quoting around a token in the matched text is ignored, and matching is
case-insensitive like the contract engine.

* A bare name without a slash (``.env``) matches that basename in any directory (like a gitignore line); an entry
  with a slash (``config/secrets.yml``) matches as the end of a path, at a directory boundary; a leading ``./`` is
  dropped. A trailing ``/`` means everything below (``~/.ssh/`` is ``~/.ssh/**``).
* An entry that starts with ``/``, ``~/``, ``$HOME/``, ``${HOME}/``, ``%USERPROFILE%/`` or a drive letter is
  anchored. ``~/`` also matches ``$HOME/``, ``${HOME}/``, ``%USERPROFILE%\\``, ``$env:USERPROFILE\\`` and a home
  directory spelled out (``HOME_ROOTS``: the Linux, macOS, WSL and Windows roots followed by an account name).
* A token must not run into neighbouring name characters: ``.env`` does not match ``.environment``,
  ``.env.example``, ``my.env`` or ``foo.envrc`` unless a glob says so (``.env*``); and it is the last component of
  its path: ``.env`` does not match ``.env/bin/python`` (a virtualenv directory).
* ``except`` entries remove a whole path word from the guard: ``.env*`` except ``.env.example`` guards
  ``.env.production`` and not ``.env.example``.

Unknown tools and MCP tools: ``any`` looks at the string values (or lists of strings) of keys whose names end like a
path (``path``, ``file``, ``dir``, ``directory``, ``folder``, ``cwd``, ``workdir``, ``url``, ``uri``, ``glob``,
``target``, ``source``, ``dest`` ...: ``AbsolutePath``, ``TargetFile``, ``notebook_path`` ...) and at the shell text
of keys named ``command``, ``cmd`` or ``CommandLine``; never at any other field (``content``, ``description``,
``old_string``, ``prompt``, ``pattern`` of a Grep ...). ``read`` knows only the tools of the map (an MCP tool whose
last name segment is a known read tool counts as that tool); a read through an unknown tool is a miss, say
``modes: any`` if that matters. Unknown tools are best effort and the tool map says which entries are untraced.

Limits (class D: the guard does not see them; ``tests/test_contract_path_guard.py`` pins every one)
-----------------------------------------------------------------------------------------------------
Shell parsing is regex based and lexical. A shell segment is a simple command, up to ``;``, ``&``, ``|``, a backtick or
a newline; a reader and a path word in one segment match. Not seen: shell variables other than
``$HOME``/``%USERPROFILE%`` (``f=.env; cat $f``), ``cd`` chains (``cd ~/.ssh && cat id_rsa``), globs in the command
(``cat .en?``), string building in an interpreter (``python -c "open('.'+'env')"``), command substitution and ``eval``,
a separator inside quotes that ends the segment, symbolic links and hard links, a program that reads a path it is not
given on its command line, Codex's JavaScript ``exec`` wrapper, a command that names no path, a read through a tool that
is neither in the map nor named like a path (mode ``read``), and an absolute entry for a ``~user`` spelling. Matched on
purpose, and so false positives: a pattern argument that equals the guarded name (``grep '\\.env' src/``), every operand
of a reader (``cp new .env``), and a glob that starts with the name (``cat .env*``).

Cost: the regex of a typed contract is generated once per process and compiled by the contract engine on first use
(about 6 ms for mode ``any`` and 26 ms for mode ``read`` with one entry, more with more entries), and a search over a
hostile text can still reach the engine's time limit, which makes the contract ``unknown`` for that call.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import re
import time
from typing import Any, Iterable, Mapping, Sequence

TOOL_MAP_VERSION = 1

MODES = ("any", "read")
DEFAULT_MODE = "any"
MAX_ENTRIES = 16
MAX_ENTRY_CHARS = 256
MAX_LOWERED_CHARS = 32768       # the lowered regex of one contract
MAX_COMPILE_SEC = 2.0           # compiling it (both engines) must stay well inside the 2 s contract budget
MAX_DEEP = 32                   # path segments one ``**`` can stand for (keeps matching linear in the length of a path)

# ------------------------------------------------------------------------------------------------------- tool map

# SEARCH: a content search (Grep); FIND: a search by file name (Glob), whose ``pattern`` is a path. Both target the path.
FILE_READ, SEARCH, FIND, LIST, FILE_WRITE, SHELL, PATCH = (
    "file_read", "search", "find", "list", "file_write", "shell", "patch")
ROLES = (FILE_READ, SEARCH, FIND, LIST, FILE_WRITE, SHELL, PATCH)


@dataclass(frozen=True)
class ToolEntry:
    """One tool of one host: the name as rendered in ``[tool_call:<name>]`` and the argument keys that carry paths
    (for a SHELL tool the command text, for a PATCH tool nothing: the patch grammar names its files)."""
    host: str
    name: str
    role: str
    keys: tuple[str, ...]
    traced: bool
    source: str


# Every entry is derived from this repository's own parsers, fixtures and tests. ``traced=False`` marks what could not be
# found there (a name the owner listed, or a key a host documents): those entries are best effort and are listed as
# untraced in the README; ``tests/test_contract_path_guard_toolmap.py`` pins the list.
TOOL_ENTRIES: tuple[ToolEntry, ...] = (
    # Claude Code (also: Grok, whose calls are rendered with these names by grok_tools)
    ToolEntry("claude", "Read", FILE_READ, ("file_path",), True,
              "tests/test_claude_provider.py, tests/test_claude_hooks.py: Read{file_path}"),
    ToolEntry("claude", "Bash", SHELL, ("command",), True, "tests/test_claude_hooks.py: Bash{command}"),
    ToolEntry("claude", "Edit", FILE_WRITE, ("file_path",), True,
              "grok_tools.py row search_replace -> Edit: file_path, old_string, new_string"),
    ToolEntry("claude", "Write", FILE_WRITE, ("file_path",), True,
              "grok_tools.py row write -> Write: file_path, content"),
    ToolEntry("claude", "Grep", SEARCH, ("path", "glob"), False,
              "path: grok_tools.py row grep; glob: Claude Code's file filter, not in a fixture here"),
    ToolEntry("claude", "Glob", FIND, ("path", "pattern"), False,
              "grok_tools.py notes that Glob needs a pattern; the keys are not in a fixture here"),
    ToolEntry("claude", "LS", LIST, ("path",), True, "grok_tools.py row list_dir -> LS: path"),
    ToolEntry("claude", "MultiEdit", FILE_WRITE, ("file_path",), False, "named by the task brief; no fixture here"),
    ToolEntry("claude", "NotebookEdit", FILE_WRITE, ("notebook_path",), False, "named by the task brief; no fixture here"),
    ToolEntry("claude", "NotebookRead", FILE_READ, ("notebook_path",), False, "no fixture here"),
    # AGY
    ToolEntry("agy", "view_file", FILE_READ, ("AbsolutePath",), True,
              "tests/test_agy_parser_v7.py, tests/test_agy_hooks.py: view_file{AbsolutePath}"),
    ToolEntry("agy", "run_command", SHELL, ("CommandLine",), True,
              "fixtures/sample_agy_session: run_command{CommandLine, Cwd}"),
    ToolEntry("agy", "replace_file_content", FILE_WRITE, ("TargetFile",), True,
              "tests/test_agy_parser_v7.py: replace_file_content{TargetFile}"),
    ToolEntry("agy", "multi_replace_file_content", FILE_WRITE, ("TargetFile",), False,
              "the name is in tests/test_agy_hooks.py with an invented argument; TargetFile is untraced for it"),
    ToolEntry("agy", "write_to_file", FILE_WRITE, ("TargetFile",), False, "named by the task brief; no fixture here"),
    ToolEntry("agy", "list_dir", LIST, ("DirectoryPath",), True,
              "tests/test_agy_parser_v7.py: list_dir{DirectoryPath}"),
    ToolEntry("agy", "grep_search", SEARCH, ("SearchPath",), False, "no fixture here"),
    ToolEntry("agy", "find_by_name", FIND, ("SearchDirectory", "Pattern"), False, "no fixture here"),
    # OMP
    ToolEntry("omp", "read", FILE_READ, ("path",), True,
              "tests/test_contract_fragility_file_reads.py: the OMP parser renders read{path}"),
    ToolEntry("omp", "bash", SHELL, ("command",), True, "tests/slicing_support.py, tests/test_secret_egress.py: bash{command}"),
    ToolEntry("omp", "write", FILE_WRITE, ("path",), False, "no OMP fixture here"),
    ToolEntry("omp", "edit", FILE_WRITE, ("path",), False, "no OMP fixture here"),
    ToolEntry("omp", "grep", SEARCH, ("path",), False, "no OMP fixture here"),
    ToolEntry("omp", "find", FIND, ("path",), False, "no OMP fixture here"),
    ToolEntry("omp", "ls", LIST, ("path",), False, "no OMP fixture here"),
    # Codex
    ToolEntry("codex", "shell", SHELL, ("command",), True,
              "tests/test_reliability.py (command string), tests/test_success_masking.py (command argv list, workdir)"),
    ToolEntry("codex", "exec_command", SHELL, ("cmd",), True, "codex_hooks._read_command reads cmd (or command)"),
    ToolEntry("codex", "powershell", SHELL, ("command",), False,
              "the name is in codex_hooks._read_only; the key is untraced"),
    ToolEntry("codex", "read_file", FILE_READ, ("path",), False,
              "the name is in codex_hooks._READ_TOOLS; the key is untraced"),
    ToolEntry("codex", "apply_patch", PATCH, (), False,
              "named by the task brief; no Codex fixture here. Matched by the patch envelope's file headers"),
    # Grok Bot (fixtures/sample_grokbot_session)
    ToolEntry("grokbot", "Read", FILE_READ, ("path",), True, "tests/test_contract_fragility_file_reads.py: Read{path}"),
)

# Grok CLI calls are rendered with Claude Code's names by ``grok_tools`` before a contract sees them, so the map has no
# rows of its own for Grok; the native rows are listed for the reader of the table.
HOSTS = ("claude", "agy", "omp", "codex", "grokbot")
HOST_NOTES = {"grok": "rendered with Claude Code's tool names and keys by grok_tools.canonical_call (TOOL_MAP_VERSION there)"}

_PATCH_HEADERS = r"""\*\*\*\s*(?:(?:Add|Update|Delete)\s+File|Move\s+to):\s*(?:\\?["'])?"""

# Keys that carry a path for a tool the map does not know (best effort), and keys that carry shell text.
_PATH_KEY = (r"""[A-Za-z0-9_.\-]*(?:path|paths|file|files|filename|file_name|dir|dirs|directory|folder|cwd|workdir|"""
             r"""url|uri|glob|target|source|src|dest|destination|location)""")


def untraced() -> tuple[tuple[str, str], ...]:
    """``(host, tool name)`` of every map entry that no fixture, parser or test of this repository confirms."""
    return tuple((e.host, e.name) for e in TOOL_ENTRIES if not e.traced)


def tool_map_table() -> list[dict[str, Any]]:
    return [{"host": e.host, "tool": e.name, "role": e.role, "keys": list(e.keys), "traced": e.traced,
             "source": e.source} for e in TOOL_ENTRIES]


def _shell_keys() -> tuple[str, ...]:
    return tuple(dict.fromkeys(k for e in TOOL_ENTRIES if e.role == SHELL for k in e.keys))


def _shell_names() -> tuple[str, ...]:
    return tuple(dict.fromkeys(e.name for e in TOOL_ENTRIES if e.role == SHELL))


# ---------------------------------------------------------------------------------------------------- validation

class PathGuardError(ValueError):
    """A ``path_guard`` that cannot be used. ``field`` names the part (``paths``, ``paths[2]``, ``except``, ``modes``)."""

    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(f"{field}: {message}")


_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_HOME_ENTRY = re.compile(r"(?:~|\$home|\$\{home\}|%userprofile%|\$env:userprofile|\$\{env:userprofile\})/", re.IGNORECASE)
_DRIVE_ENTRY = re.compile(r"[A-Za-z]:/")


def _check_entry(field: str, index: int, entry: Any) -> str:
    where = f"{field}[{index}]"
    if not isinstance(entry, str):
        raise PathGuardError(where, f"must be a string, got {type(entry).__name__}")
    if not entry:
        raise PathGuardError(where, "is empty; name a path or a glob")
    if len(entry) > MAX_ENTRY_CHARS:
        raise PathGuardError(where, f"is longer than {MAX_ENTRY_CHARS} characters")
    if _CONTROL.search(entry):
        raise PathGuardError(where, "contains a control character (NUL, newline, tab ...)")
    if entry != entry.strip():
        raise PathGuardError(where, "has leading or trailing whitespace; remove it")
    if any(q in entry for q in "\"'`"):
        raise PathGuardError(where, "contains a quote; quoting around a path in a command is ignored, so leave it out")
    stripped = re.sub(r"\[[^\]]*\]", "", entry.replace("\\", "/"))
    if not re.sub(r"[*?/\s]", "", stripped):
        raise PathGuardError(where, f"{entry!r} has no literal character: a wildcard-only entry would guard every path")
    return entry


def check_modes(value: Any) -> str:
    if value == "write":
        raise PathGuardError("modes", "'write' is not supported in this version (deferred); use 'any' (every tool call "
                             "that names the path, writes included) or 'read'")
    if not isinstance(value, str) or value not in MODES:
        shown = repr(value) if not isinstance(value, (list, dict)) else type(value).__name__
        raise PathGuardError("modes", f"must be one of {', '.join(repr(m) for m in MODES)} (a string, not a list), got {shown}")
    return value


def check_entries(field: str, entries: Any, *, required: bool) -> tuple[str, ...]:
    if not isinstance(entries, (list, tuple)):
        raise PathGuardError(field, f"must be a list of strings, got {type(entries).__name__}")
    if required and not entries:
        raise PathGuardError(field, "must not be empty: name at least one path")
    if len(entries) > MAX_ENTRIES:
        raise PathGuardError(field, f"has {len(entries)} entries; at most {MAX_ENTRIES} are allowed")
    checked = tuple(_check_entry(field, i, entry) for i, entry in enumerate(entries))
    for i, entry in enumerate(checked):
        compile_token(f"{field}[{i}]", entry)          # a glob that does not compile is an error here, naming the entry
    return checked


# ----------------------------------------------------------------------------------------------- globs and tokens
# Regex building blocks. The text of a tool call is JSON (a literal backslash in a path is written ``\\``, and ``\n``, ``\t``
# and ``\"`` are escapes) or, for a tool whose arguments are a plain string, the string itself, where a backslash is
# only ever a separator. A path word is therefore read in one of two dialects, each with its own separators and
# boundaries, and the lowered regex has one section per dialect (JSON contexts, plain-string contexts).

Q = '"'
_ESC = "ntrbfu" + Q + "'"                                  # what may follow a backslash in a JSON string
_DELIMS = "\\s" + Q + "'`;&|<>(),="                        # characters that end a path word
_G = "[^/\\\\\\s" + Q + "'`;&|<>(){},]"                    # one character that a wildcard stands for
_USER = "[^/\\\\" + Q + "'`;&|<>(),\\n]+"                  # an account name (spaces are allowed in it)


@dataclass(frozen=True)
class _Dialect:
    s: str          # a run of separators
    dirs: str       # the directories (or ``HEAD:``, ``@``) before a name or a suffix
    rb: str         # the token ends here: no name character and no separator follows
    wstart: str     # a path word starts here
    home: str       # the spellings of the home directory, with the separator after it


def _after_escape(letters: str) -> str:
    """Just after a JSON escape of one of ``letters`` (``\\n``): an odd run of backslashes (``\\\\n`` is a backslash and an
    ``n``). Lookbehinds are fixed width, so runs of one and of three are covered."""
    return "(?:(?<=[^\\\\]\\\\[" + letters + "])|(?<=[^\\\\]\\\\\\\\\\\\[" + letters + "]))"


def _dialect(raw: bool) -> _Dialect:
    if raw:
        sep, rb, wstart = "[/\\\\]", "(?![\\w.\\-@+%~/\\\\])", "(?<![^" + _DELIMS + "])"
        s = sep + "+"
        dirs = "(?:[^" + _DELIMS + "/\\\\:@]*+(?:[/:@]|\\\\++))*"
    else:
        # A slash, a run of doubled backslashes (one separator, possessively), or a single backslash that does not begin a
        # JSON escape: ``\n``, ``\t`` and ``\"`` are escapes and never separators, so a newline inside a long text ends a
        # path word (and matching stays linear in the length of the text).
        sep = "(?:/|(?:\\\\\\\\)++|\\\\(?![" + _ESC + "\\\\]))"
        s = sep + "+"
        rb = "(?![\\w.\\-@+%~/]|\\\\(?![" + _ESC + "]))"
        wstart = "(?:(?<![^" + _DELIMS + "])|" + _after_escape("ntrbf") + ")"
        dirs = "(?:[^" + _DELIMS + "/\\\\:@]*+(?:[/:@]|(?:\\\\\\\\)++|\\\\(?![" + _ESC + "\\\\])))*"
    home = ("(?:~" + _G + "*"
            "|\\$\\{?HOME\\}?(?:\\\\?[" + Q + "'])?"
            "|\\$\\{?env:(?:USERPROFILE|HOME|HOMEPATH)\\}?(?:\\\\?[" + Q + "'])?"
            "|%(?:USERPROFILE|HOME)%|%HOMEDRIVE%%HOMEPATH%"
            "|/(?:home|Users)/" + _USER + "|/root|/mnt/[a-z]/Users/" + _USER + "|/[a-z]/Users/" + _USER +
            "|[a-z]:" + s + "Users" + s + _USER + ")") + s
    return _Dialect(s, dirs, rb, wstart, home)


JSON_TEXT, RAW_TEXT = _dialect(False), _dialect(True)
HOME_ROOTS = ("/home/<name>", "/Users/<name>", "/root", "/mnt/<drive>/Users/<name>", "<drive>:\\Users\\<name>")


def _class(body: str, negate: bool) -> str:
    return "(?:(?![/\\\\\\s" + Q + "'`;&|<>(){},])[" + ("^" if negate else "") + body + "])"


def _segment(field: str, seg: str) -> str:
    out, i, n = [], 0, len(seg)
    while i < n:
        c = seg[i]
        if c == "*":
            while i + 1 < n and seg[i + 1] == "*":
                i += 1
            out.append(_G + "*")
        elif c == "?":
            out.append(_G)
        elif c == "[":
            j = i + 1
            negate = j < n and seg[j] in "!^"
            if negate:
                j += 1
            start = j
            if j < n and seg[j] == "]":
                j += 1
            while j < n and seg[j] != "]":
                j += 1
            if j >= n:
                raise PathGuardError(field, f"unbalanced '[' in {seg!r}")
            body = seg[start:j]
            if not body:
                raise PathGuardError(field, f"empty character class in {seg!r}")
            body = body.replace("\\", "\\\\").replace("[", "\\[").replace("^", "\\^")
            try:
                re.compile("[" + body + "]")
            except re.error as exc:
                raise PathGuardError(field, f"invalid character class in {seg!r} ({exc.msg})") from None
            out.append(_class(body, negate))
            i = j
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


def _fold(field: str, segs: list[str], *, lead: bool, sep: str) -> str:
    """Segments joined by separators; ``**`` stands for any number of whole segments. ``lead``: a separator was just
    consumed (an anchored prefix), so a first ``**`` owns the separators after it."""
    out, owed, n = "", not lead, len(segs)
    for index, seg in enumerate(segs):
        last = index == n - 1
        if seg == "**":
            if last:
                out += ("(?:" + sep + _G + "+){1," + str(MAX_DEEP) + "}") if out else (
                    _G + "+(?:" + sep + _G + "+){0," + str(MAX_DEEP) + "}")
            elif out:
                out += "(?:" + sep + _G + "+){0," + str(MAX_DEEP) + "}"
            else:
                out += "(?:" + _G + "+" + sep + "){0," + str(MAX_DEEP) + "}"
                owed = False
            continue
        if owed and out:
            out += sep
        out += _segment(field, seg)
        owed = True
    return out


@dataclass(frozen=True)
class Token:
    entry: str
    kind: str            # home | abs | drive | name | suffix
    word: str            # regex for the whole path word the entry names; for a name or a suffix without the directories


def _normal(entry: str) -> str:
    text = re.sub(r"/+", "/", entry.replace("\\", "/"))
    while text.startswith("./") and len(text) > 2:
        text = text[2:]
    return text


def canonical(entry: str) -> str:
    """The entry as the matcher reads it: slashes forward and collapsed, no leading ``./``, lower case."""
    return _normal(entry).lower()


def compile_token(field: str, entry: str, dialect: _Dialect = JSON_TEXT) -> Token:
    text = _normal(entry)
    home = _HOME_ENTRY.match(text)
    if home:
        kind, rest, prefix = "home", text[home.end():], dialect.home
    elif text.startswith("/"):
        kind, rest, prefix = "abs", text[1:], "(?:file://)?[/\\\\]"
    elif _DRIVE_ENTRY.match(text):
        kind, rest, prefix = "drive", text[3:], "(?:file:///?)?" + re.escape(text[0]) + ":" + dialect.s
    elif text.startswith("~"):
        raise PathGuardError(field, f"{entry!r}: write the home directory as '~/...' (or an absolute path)")
    else:
        kind, rest, prefix = ("suffix" if "/" in text.rstrip("/") else "name"), text, ""
    if not re.sub(r"\[[^\]]*\]|[*?/]", "", rest):
        raise PathGuardError(field, f"{entry!r} has no literal part below its anchor: name a directory or a file")
    if rest.endswith("/"):
        rest = rest.rstrip("/") + "/**"                      # a directory: everything below it
    segs = rest.split("/")
    if kind in ("name", "suffix"):
        while segs and segs[0] == "**":
            segs = segs[1:]
        if not segs:
            raise PathGuardError(field, f"{entry!r} has no literal part")
    if any(seg == "" for seg in segs):
        raise PathGuardError(field, f"{entry!r} has an empty path segment")
    body = _fold(field, segs, lead=bool(prefix), sep=dialect.s)
    return Token(entry, kind, (prefix + body) if prefix else body)


def _words(tokens: Sequence[Token], dialect: _Dialect) -> str:
    """The alternatives for the whole path word: anchored entries from the word's start, names and suffixes after any
    directories (one scan of the directories for all of them)."""
    anchored = [t.word for t in tokens if t.kind in ("home", "abs", "drive")]
    relative = [t.word for t in tokens if t.kind in ("name", "suffix")]
    if relative:
        anchored.append(dialect.dirs + "(?:" + "|".join(relative) + ")")
    return "(?:" + "|".join(anchored) + ")"


# --------------------------------------------------------------------------------------------------- the contexts
# A context is the part of a tool call's text that comes before a path word: it ends exactly where the word may start.

_TC = "\\[(?:proposed_)?tool_call:"
_STRC = "(?:[^" + Q + "\\\\\\n]|\\\\.)"                      # one character of a JSON string
_STRW = "(?:\\w++|[^\\w" + Q + "\\\\\\n]|\\\\.)*?"                  # a lazy walk through a string, word by word
_STR = Q + "[^" + Q + "\\\\\\n]*(?:\\\\.[^" + Q + "\\\\\\n]*)*" + Q       # a whole JSON string
_VAL = "(?:\\[(?:\\s*" + _STR + "\\s*,)*\\s*)?" + Q                       # the start of a string value or list element
_MCP = "(?:mcp__[^\\]\\s]*?__)?"
# One shell segment (a simple command, up to ; & | a backtick or a newline) read in items: runs of word characters, other
# single characters and escapes. ``seg_item`` is what the lazy window walks over; ``seg_rest`` swallows the rest of a segment.
_JSON_SEG = {"item": "(?:\\w++|[^\\w" + Q + "\\\\;&|`\\n]|\\\\\\\\\\\\n|\\\\[^nr])",
             "rest": "(?:[^" + Q + "\\\\;&|`\\n]++|\\\\\\\\\\\\n|\\\\[^nr])*+",
             "other": "[^\\w" + Q + "\\\\\\n]", "any": "\\\\."}
_RAW_SEG = {"item": "(?:\\w++|[^\\w;&|`\\n]|\\\\\\n)",
            "rest": "(?:[^;&|`\\n]++|\\\\\\n)*+",
            "other": "[^\\w]", "any": "[\\s\\S]"}

READ_VERBS = (
    "cat", "tac", "nl", "head", "tail", "less", "more", "most", "bat", "batcat", "sed", "awk", "gawk", "mawk", "nawk",
    "grep", "egrep", "fgrep", "zgrep", "rg", "ag", "ack", "cp", "scp", "rsync", "dd", "xxd", "od", "hexdump", "hd",
    "strings", "base64", "base32", "basenc", "cut", "sort", "uniq", "paste", "diff", "cmp", "comm", "jq", "yq", "tar",
    "zip", "7z", "7za", "curl", "wget", "openssl", "gpg", "source", "type", "gc", "get-content", "select-string",
    "sls", "copy", "copy-item", "cpi", "perl", "ruby")
INTERPRETERS = ("python", "python2", "python3", "py", "node", "nodejs", "deno", "bun", "perl", "ruby", "php", "lua",
                "rscript")
_GIT_READ = "git(?:\\s+-\\S+(?:\\s+(?!-)\\S+)?)*\\s+(?:show|cat-file|diff|blame|log|grep|archive)"


_VSTART = "(?:(?<![\\w$.\\-])|" + _after_escape("ntr") + ")"          # a command word starts here


def _alt_body(names: Iterable[str]) -> str:
    return "|".join(re.escape(n) for n in sorted(set(names), key=lambda s: (-len(s), s)))


def _alt(names: Iterable[str]) -> str:
    return "(?:" + _alt_body(names) + ")"


# Skipping to a key: whole JSON strings and runs between them are consumed at once (possessively), so the search for
# the next key costs one step per string, not one per character of a 1 MB file body.
_STR_P = Q + "[^" + Q + "\\\\\\n]*+(?:\\\\.[^" + Q + "\\\\\\n]*+)*+" + Q
_SKIP = "(?:[^" + Q + "\\n]++|" + _STR_P + ")*?"


def _key_prefix(names: str, keys: str) -> str:
    """Up to the ``:`` after one of ``keys`` in a tool call named ``names``."""
    return _TC + names + "\\]\\s*" + _SKIP + Q + keys + Q + "\\s*:\\s*"


def _key_context(names: str, keys: str) -> str:
    """Up to the start of a string value (or list element) of one of ``keys`` in a tool call named ``names``."""
    return _key_prefix(names, keys) + _VAL + _STRW


def _shell_raw_context() -> str:
    return _TC + _alt(_shell_names()) + "\\]\\s+(?![\\[{])"


_VERB_HEAD = _VSTART + "(?:" + _alt_body(READ_VERBS) + "|" + _GIT_READ + ")" + JSON_TEXT.rb
_VERB_HEAD_RAW = "(?<![\\w$.\\-])(?:" + _alt_body(READ_VERBS) + "|" + _GIT_READ + ")" + RAW_TEXT.rb
_DOT_HEAD = "(?:(?<=[" + Q + "';&|(`{])|" + _after_escape("nrt") + ")\\s*\\.\\s+"


def _reader_tails(seg: dict, group: str) -> list[str]:
    """What may stand just before a path word that is read, inside one shell command string."""
    interp = _VSTART + _alt(INTERPRETERS) + "(?![\\w.\\-/\\\\])"
    quote = "(?P<" + group + ">\\\\?[" + Q + "'])"
    return [
        _VERB_HEAD + seg["item"] + "*?",                                                     # a reader, then its operand
        "(?<!<)<(?![<(])\\s*(?:\\\\?[" + Q + "'])?",                                         # < file
        _DOT_HEAD + seg["item"] + "*?",                                                      # . file
        interp + "[^;&|`\\n" + Q + "]*?\\s-(?:(?=[A-Za-z]*[ceEr])[A-Za-z]++|-eval)\\s+" + quote + "(?:(?!(?P=" + group + "))[^\\n])*?",
        interp + "[^;&|`\\n" + Q + "]*?<<-?\\s*\\\\?[" + Q + "']?\\w+\\\\?[" + Q + "']?[\\s\\S]*?",   # heredoc
    ]


def _walk(seg: dict) -> str:
    """The lazy walk from the start of a command string to a reader. A reader whose segment holds no path word
    makes every later reader of that segment pointless (their windows lie inside its), so the walk swallows the
    whole rest of the segment at once: each segment is read at most twice and the walk is linear. Each step is
    atomic (a reader is never also taken as a plain word), so the walk cannot be parsed in two ways."""
    return ("(?:(?>" + _VERB_HEAD + seg["rest"] + "|" + _DOT_HEAD + seg["rest"] + "|\\w++|" + seg["other"] + "|"
            + seg["any"] + "))*?")


def _shell_read_context() -> str:
    """The shell command strings of any tool, up to the word a reader may take: the string value (or one element of an
    argv list) holding the command, and an argv list that starts with a reader or an interpreter."""
    element = "(?:" + _STR + "\\s*,\\s*)"
    in_string = _VAL + _walk(_JSON_SEG) + "(?:" + "|".join(_reader_tails(_JSON_SEG, "qj")) + ")"
    argv_reader = ("\\[\\s*" + element + "*?" + Q + _alt(READ_VERBS) + Q + "\\s*,\\s*" + element + "*?" + Q + _STRW)
    argv_inline = ("\\[\\s*" + element + "*?" + Q + _alt(INTERPRETERS) + Q + "\\s*,\\s*" + element + "*?" + Q
                   + "-[A-Za-z]*[ceEr]" + Q + "\\s*,\\s*" + Q + _STRW)
    return _key_prefix("[^\\]\\n]*", _alt(_shell_keys())) + "(?:" + in_string + "|" + argv_reader + "|" + argv_inline + ")"


def contexts(modes: str) -> tuple[list[str], list[str]]:
    """The alternatives that lead up to a path word, for one mode: ``(JSON contexts, plain-string contexts)``."""
    out: list[str] = []
    raw: list[str] = []
    if modes == "any":
        out.append(_key_context("[^\\]\\n]*", "(?:" + _PATH_KEY + "|" + _alt_body(_shell_keys()) + ")"))
        unusual: dict[str, set[str]] = {}
        for entry in TOOL_ENTRIES:      # keys of a known tool that do not read like a path (a Glob's ``pattern``)
            if entry.role in (FILE_READ, SEARCH, FIND, LIST, FILE_WRITE):
                for key in entry.keys:
                    if not re.fullmatch(_PATH_KEY, key, re.IGNORECASE):
                        unusual.setdefault(key.lower(), set()).add(entry.name)
        for key, names in sorted(unusual.items()):
            out.append(_key_context(_alt(names), _alt([key])))
        raw.append(_shell_raw_context() + "[\\s\\S]*?")
        patch = [e.name for e in TOOL_ENTRIES if e.role == PATCH]
        if patch:       # a patch is a JSON string field of the call, or the whole plain-string argument
            out.append(_TC + _alt(patch) + "\\]\\s*\\{[\\s\\S]*?" + _PATCH_HEADERS)
            raw.append(_TC + _alt(patch) + "\\]\\s+(?![\\[{])[\\s\\S]*?" + _PATCH_HEADERS)
        return out, raw
    for role in (FILE_READ, SEARCH, FIND):      # one context per role: its tools, and the keys that carry its paths
        entries = [e for e in TOOL_ENTRIES if e.role == role and e.keys]
        out.append(_key_context(_MCP + _alt(e.name for e in entries), _alt(k for e in entries for k in e.keys)))
    out.append(_shell_read_context())
    # A plain-string argument (no JSON) is rare and short: a reader, or a ``<``, in the same line; no interpreter forms.
    raw.append(_shell_raw_context() + "[\\s\\S]*?(?:" + _VERB_HEAD_RAW + _RAW_SEG["item"] + "*?|(?<!<)<(?![<(])\\s*[" + Q + "']?)")
    return out, raw


# ------------------------------------------------------------------------------------------------------- lowering

def _compile_check(pattern: str) -> None:
    started = time.perf_counter()
    try:
        re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    except re.error as exc:
        raise PathGuardError("paths", f"the generated pattern does not compile ({exc.msg}); use plainer globs") from None
    if time.perf_counter() - started > MAX_COMPILE_SEC:
        raise PathGuardError("paths", f"the generated pattern takes longer than {MAX_COMPILE_SEC} s to compile; "
                             "use fewer or plainer globs")


def _section(contexts_: list[str], tokens: list[Token], skipped: list[Token], dialect: _Dialect) -> str:
    """One dialect: its contexts, then a path word that a guard entry names and no ``except`` entry does."""
    exclude = ("(?!" + _words(skipped, dialect) + dialect.rb + ")") if skipped else ""
    return ("(?:" + "|".join(contexts_) + ")" + dialect.wstart + exclude + _words(tokens, dialect) + dialect.rb)


@functools.lru_cache(maxsize=256)
def _lower(paths: tuple[str, ...], excepts: tuple[str, ...], modes: str) -> str:
    sections = []
    for contexts_, dialect in zip(contexts(modes), (JSON_TEXT, RAW_TEXT)):
        tokens = [compile_token(f"paths[{i}]", entry, dialect) for i, entry in enumerate(paths)]
        skipped = [compile_token(f"except[{i}]", entry, dialect) for i, entry in enumerate(excepts)]
        sections.append(_section(contexts_, tokens, skipped, dialect))
    pattern = "|".join(sections)
    if len(pattern) > MAX_LOWERED_CHARS:
        raise PathGuardError("paths", f"the generated pattern has {len(pattern)} characters, over the limit of "
                             f"{MAX_LOWERED_CHARS}; use fewer or plainer globs (or mode 'read', which is smaller)")
    _compile_check(pattern)
    return pattern


def normalize(paths: Any, excepts: Any = (), modes: Any = DEFAULT_MODE) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """The validated ``(paths, except, modes)``; ``PathGuardError`` names the field that is wrong."""
    return check_entries("paths", paths, required=True), check_entries("except", excepts, required=False), check_modes(modes)


def lower(paths: Any, excepts: Any = (), modes: Any = DEFAULT_MODE) -> str:
    """The contract regex of a guard under the current tool map. Raises ``PathGuardError``."""
    return _lower(*normalize(paths, excepts, modes))


def lower_mapping(guard: Any) -> str:
    """``lower`` for a ``path_guard`` read from JSON as it is on disk (a dictionary)."""
    if not isinstance(guard, Mapping):
        raise PathGuardError("path_guard", "must be an object")
    unknown = sorted(set(guard) - {"paths", "except", "modes"})
    if unknown:
        raise PathGuardError("path_guard", f"unknown key(s): {', '.join(map(str, unknown))}")
    return lower(guard.get("paths"), guard.get("except", ()), guard.get("modes", DEFAULT_MODE))


# ---------------------------------------------------------------------------------- identity, display and advice

def as_dict(guard: Any) -> dict[str, Any]:
    """The stored (typed) form of a guard, with every key present: what is hashed, exported and compared."""
    return {"paths": list(guard.paths), "except": list(guard.except_), "modes": guard.modes}


def describe(paths: Sequence[str], excepts: Sequence[str], modes: str) -> str:
    text = ", ".join(paths)
    if excepts:
        text += " (except " + ", ".join(excepts) + ")"
    return text + f"; mode {modes}"


def describe_guard(guard: Any) -> str:
    text = describe(guard.paths, guard.except_, guard.modes)
    return text if len(text) <= 200 else text[:199] + "..."


def verifier_text(rule: Any) -> str:
    """What to show of a contract's verifier when it has no description: its regex, or its typed guard."""
    guard = getattr(rule, "path_guard", None)
    return "path_guard " + describe_guard(guard) if guard is not None else (rule.regex or "")


def forbidden_label(rule: Any) -> str:
    """The subject of a finding's reason: ``forbidden regex`` or ``forbidden path <paths> (...)``."""
    guard = getattr(rule, "path_guard", None)
    return f"forbidden path {describe_guard(guard)}" if guard is not None else "forbidden regex"


def verifier_basis(rule: Any, outcome: str) -> str:
    """The basis of a finding: ``<id>.regex '<regex>' <outcome>`` for a regex, ``<id>.path_guard <paths> <outcome>``."""
    guard = getattr(rule, "path_guard", None)
    if guard is None:
        return f"{rule.id}.regex '{rule.regex}' {outcome}"
    return f"{rule.id}.path_guard {describe_guard(guard)} (tool map v{TOOL_MAP_VERSION}) {outcome}"


def verifier_field(rule: Any) -> str:
    return "path_guard" if getattr(rule, "path_guard", None) is not None else "regex"


_MODE_RANK = {"any": 0, "read": 1}


def weakening(previous: Any, new: Any) -> list[str]:
    """Why replacing the typed guard ``previous`` by ``new`` may enforce less; ``[]`` when it cannot.

    Entries only add alternatives, so a guard that keeps every old path, drops no exception and does not narrow the
    mode matches at least what the old one did. Anything else (a removed or replaced path, an added ``except``,
    ``any`` to ``read``) is judged a weakening, exactly like a changed regex."""
    reasons = []
    if not {canonical(p) for p in previous.paths} <= {canonical(p) for p in new.paths}:
        reasons.append("path_guard.paths narrowed or replaced")
    if {canonical(p) for p in new.except_} - {canonical(p) for p in previous.except_}:
        reasons.append("path_guard.except added")
    if _MODE_RANK[new.modes] > _MODE_RANK[previous.modes]:
        reasons.append(f"path_guard.modes narrowed ({previous.modes} -> {new.modes})")
    return reasons


_SHELL_READER = re.compile(r"\b(?:" + "|".join(re.escape(v) for v in READ_VERBS if v.isalpha()) + r")\b", re.IGNORECASE)
_CANDIDATE = re.compile(
    r"(?<![\w.\-])(?:~/|/)?(?:[\w\-]+/)*\.[A-Za-z][\w\-]*(?:\.[\w\-]+)*(?:/[\w\-]+(?:\.[\w\-]+)*)*"
    r"|(?<![\w.\-])(?:~/|/)[\w\-.]+(?:/[\w\-.]+)*")
_EXTENSIONS = frozenset("pem key crt cer p12 pfx jks json yml yaml toml ini cfg conf db sqlite sqlite3 log txt csv".split())
_WILD_BEFORE = re.compile(r"(?:\\S\*|\\w\*|\.\*|\*)$")


def suggest_from_regex(regex: str) -> list[str]:
    """Path-like tokens of a shell-scoped regex (a reader verb in it, and a dotfile or a path), or ``[]``.

    ``(cat|head|tail|less)\\s+\\S*\\.env`` gives ``['.env']``; ``cat\\s+\\S*\\.pem`` gives ``['*.pem']`` (a known extension
    after a wildcard is an extension). Purely lexical and informational: the caller shows it as a suggestion and
    never rewrites a contract."""
    if not isinstance(regex, str) or not _SHELL_READER.search(regex):
        return []
    plain = regex.replace("\\.", ".").replace("\\/", "/").replace("\\-", "-")
    found: list[str] = []
    for match in _CANDIDATE.finditer(plain):
        token = match.group(0).rstrip(".")
        if (not token or token.startswith("..") or token in (".", "~/", "/") or len(token) > MAX_ENTRY_CHARS
                or not re.search(r"[A-Za-z0-9_]", token)):
            continue
        if (token.startswith(".") and token[1:].lower() in _EXTENSIONS and _WILD_BEFORE.search(plain[:match.start()])):
            token = "*" + token
        if token not in found:
            found.append(token)
    return found[:MAX_ENTRIES]
