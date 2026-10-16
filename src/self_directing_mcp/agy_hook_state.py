"""Standard-library state and fail-open responses for the AGY hook adapter.

This module must stay importable when the rest of the runtime is not (missing native extension, broken
dependency): ``scripts/agy_hook.py`` uses it to answer, record ``hook_health`` and honour user-selected
high-risk contracts after an import failure. It therefore imports nothing outside the standard library, apart
from ``tool_payload`` (itself standard-library only, and the one builder of the tool-call text), which a copy of
this module without its package does without: it then matches the unfiltered text, the more cautious side.

Persistent state lives in ``<index_dir>/agy-hook-state.sqlite``:

* ``continue_budget``: Stop continuations already spent, keyed by conversation and reason. Turns never reset it.
* ``hook_health``: error class, counts and timestamps per failure kind and event. Never payload text.
* ``pending_warnings``: unverified-allow warnings waiting for the next PreInvocation injection.

Every method that touches the database swallows storage errors and reports them through its return value.
A hook must never fail because bookkeeping did, and an unreadable budget means "do not continue".
"""
from __future__ import annotations

from contextlib import closing
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
import tomllib

try:
    from self_directing_mcp.tool_payload import tool_call_text
except ImportError:   # a standalone copy next to a broken package: prose fields are then matched too (asks more)
    def tool_call_text(provider, tool_name, arguments, *, label="tool_call"):
        return f"[{label}:{tool_name}] " + (arguments if isinstance(arguments, str) else json.dumps(
            arguments, ensure_ascii=False, sort_keys=True))

try:
    from self_directing_mcp import contract_paths      # standard library only: lowers a typed ``path_guard`` to its regex
except ImportError:                                    # a standalone copy: a typed high-risk contract then always asks
    contract_paths = None

STATE_FILE = "agy-hook-state.sqlite"
MODES = ("advisory", "confirmed_only", "enforced")
DEFAULT_MODE = "confirmed_only"
EVENTS = ("PreToolUse", "PostToolUse", "PreInvocation", "Stop")

# At most this many Stop continuations per conversation, ever; the same reason never repeats.
CONTINUE_BUDGET = 1
# Consecutive failures before PreInvocation shows a banner.
BANNER_THRESHOLD = 3
# Warnings queued for the next PreInvocation are bounded per conversation.
MAX_PENDING_WARNINGS = 5

AUDIT_UNAVAILABLE = "audit_unavailable"   # no contract can be checked at all
CHECK_FAILED = "check_failed"             # one event's check failed; others still run


# --- settings the fallback can resolve without pydantic ---------------------------------------------------

def resolve_mode(environ=None) -> str:
    value = (environ if environ is not None else os.environ).get("SELF_DIRECT_AGY_ENFORCEMENT")
    return value if value in MODES else DEFAULT_MODE


def _toml_paths() -> tuple[dict, Path]:
    selected = os.environ.get("SELF_DIRECT_CONFIG_FILE")
    source = Path(selected or Path.home() / ".config" / "tsukkomi-mcp" / "config.toml").expanduser().resolve()
    try:
        with source.open("rb") as stream:
            paths = tomllib.load(stream).get("paths", {})
        return (paths if isinstance(paths, dict) else {}), source.parent
    except (OSError, ValueError):
        return {}, source.parent


def _toml_path(name: str) -> Path | None:
    paths, directory = _toml_paths()
    value = paths.get(name)
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else directory / path).resolve()


def resolve_index_dir() -> Path:
    """Environment, then the user TOML ``[paths]``, then the default; mirrors ``Settings`` for these keys."""
    override = os.environ.get("SELF_DIRECT_INDEX_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return _toml_path("index_dir") or Path.home() / ".self_direct_index"


def resolve_contracts_path() -> Path:
    override = os.environ.get("SELF_DIRECT_CONTRACTS_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return _toml_path("contracts_path") or resolve_index_dir() / "contracts.json"


# --- persistent state -------------------------------------------------------------------------------------

class HookState:
    """Per-host state. ``state_file`` selects an isolated database with the same schema (Claude Code uses its own)."""

    def __init__(self, index_dir, state_file: str | None = None) -> None:
        self.path = Path(index_dir) / (state_file or STATE_FILE)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=2.0, isolation_level=None)
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS continue_budget (
                    conversation_id TEXT NOT NULL, reason_key TEXT NOT NULL, claimed_at REAL NOT NULL,
                    PRIMARY KEY (conversation_id, reason_key));
                CREATE TABLE IF NOT EXISTS hook_health (
                    kind TEXT NOT NULL, event TEXT NOT NULL, total INTEGER NOT NULL, consecutive INTEGER NOT NULL,
                    first_at REAL NOT NULL, last_at REAL NOT NULL, last_error_class TEXT NOT NULL,
                    last_success_at REAL, PRIMARY KEY (kind, event));
                CREATE TABLE IF NOT EXISTS pending_warnings (
                    conversation_id TEXT NOT NULL, text TEXT NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY (conversation_id, text));""")
        except BaseException:
            conn.close()
            raise
        return conn

    # Stop continue budget.
    def claim_continue(self, conversation_id: str, reason_key: str, budget: int | None = None) -> bool:
        """Atomically spend one continuation. False for anything but a clean, recorded claim."""
        limit = CONTINUE_BUDGET if budget is None else budget
        try:
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    spent = conn.execute("SELECT COUNT(*) FROM continue_budget WHERE conversation_id=?",
                                         (conversation_id,)).fetchone()[0]
                    repeated = conn.execute("SELECT 1 FROM continue_budget WHERE conversation_id=? AND reason_key=?",
                                            (conversation_id, reason_key)).fetchone()
                    if spent >= limit or repeated:
                        conn.execute("ROLLBACK")
                        return False
                    conn.execute("INSERT INTO continue_budget VALUES (?,?,?)", (conversation_id, reason_key, time.time()))
                    conn.execute("COMMIT")
                    return True
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
        except Exception:
            return False

    def continue_claims(self, conversation_id: str) -> int | None:
        """Continuations spent for a conversation; None if the counter cannot be read."""
        if not self.path.exists():
            return 0
        try:
            with closing(self._connect()) as conn:
                return conn.execute("SELECT COUNT(*) FROM continue_budget WHERE conversation_id=?",
                                    (conversation_id,)).fetchone()[0]
        except Exception:
            return None

    # hook_health.
    def record_error(self, kind: str, event: str, error_class: str) -> bool:
        now = time.time()
        try:
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO hook_health (kind,event,total,consecutive,first_at,last_at,last_error_class)"
                    " VALUES (?,?,1,1,?,?,?) ON CONFLICT(kind,event) DO UPDATE SET total=total+1,"
                    " consecutive=consecutive+1, last_at=excluded.last_at, last_error_class=excluded.last_error_class",
                    (kind, event, now, now, error_class[:80]))
                conn.execute("COMMIT")
            return True
        except Exception:
            return False

    def record_success(self, event: str) -> bool:
        """A working event ends its own streak; any working event ends a full-unavailability streak."""
        if not self.path.exists():
            return True
        try:
            with closing(self._connect()) as conn:
                if conn.execute("SELECT 1 FROM hook_health WHERE consecutive>0 AND (event=? OR kind=?) LIMIT 1",
                                (event, AUDIT_UNAVAILABLE)).fetchone() is None:
                    return True  # healthy hot path: no write lock
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("UPDATE hook_health SET consecutive=0, last_success_at=?"
                             " WHERE consecutive>0 AND (event=? OR kind=?)", (time.time(), event, AUDIT_UNAVAILABLE))
                conn.execute("COMMIT")
            return True
        except Exception:
            return False

    def health(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            with closing(self._connect()) as conn:
                conn.row_factory = sqlite3.Row
                return [dict(row) for row in conn.execute("SELECT * FROM hook_health ORDER BY kind, event")]
        except Exception:
            return []

    def banner(self) -> str | None:
        rows = [row for row in self.health() if row["consecutive"] >= BANNER_THRESHOLD]
        lines = []
        inactive = [row for row in rows if row["kind"] == AUDIT_UNAVAILABLE]
        degraded = [row for row in rows if row["kind"] == CHECK_FAILED]
        if inactive:
            last = max(inactive, key=lambda row: row["last_at"])
            lines.append(
                f"Tsukkomi audit INACTIVE: the audit runtime failed {sum(r['consecutive'] for r in inactive)} "
                f"consecutive hook calls (last error: {last['last_error_class']}, {_stamp(last['last_at'])}). "
                "No contract is being checked; do not claim compliance. Repair the installation.")
        if degraded:
            last = max(degraded, key=lambda row: row["last_at"])
            events = ", ".join(sorted({r["event"] for r in degraded}))
            lines.append(
                f"Tsukkomi audit DEGRADED: {events} checks failed {sum(r['consecutive'] for r in degraded)} "
                f"consecutive times (last error: {last['last_error_class']}, {_stamp(last['last_at'])}). "
                "Those checks are not being applied; do not claim compliance for them.")
        return "\n".join(lines) or None

    # Warnings for the next PreInvocation (the host-supported injection channel).
    def queue_warning(self, conversation_id: str, text: str) -> bool:
        try:
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("INSERT OR IGNORE INTO pending_warnings VALUES (?,?,?)",
                             (conversation_id, text[:600], time.time()))
                conn.execute("DELETE FROM pending_warnings WHERE conversation_id=? AND rowid NOT IN ("
                             "SELECT rowid FROM pending_warnings WHERE conversation_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?)",
                             (conversation_id, conversation_id, MAX_PENDING_WARNINGS))
                conn.execute("COMMIT")
            return True
        except Exception:
            return False

    def drain_warnings(self, conversation_id: str) -> list[str]:
        if not self.path.exists():
            return []
        try:
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                rows = conn.execute("SELECT text FROM pending_warnings WHERE conversation_id=? ORDER BY created_at, rowid",
                                    (conversation_id,)).fetchall()
                conn.execute("DELETE FROM pending_warnings WHERE conversation_id=?", (conversation_id,))
                conn.execute("COMMIT")
            return [row[0] for row in rows]
        except Exception:
            return []


def _stamp(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


# --- user-selected high-risk policy without the runtime ----------------------------------------------------

def high_risk_match(contracts_path, session_id, tool_call, provider: str = "agy") -> str | None:
    """Id of an enabled ``risk: high`` contract that applies to this call, or None.

    Reads the contracts file with ``json`` only. An unreadable file means "unknown", never a guess. A
    high-risk contract with no usable pattern applies to every call, as it does in the audit itself.
    """
    try:
        document = json.loads(Path(contracts_path).read_text(encoding="utf-8"))
        name, args = tool_call["name"], tool_call["args"]
        text = tool_call_text(provider, name, args)   # the text the audit matches: prose fields excluded
        sid = session_id.lower() if isinstance(session_id, str) else None
        for rule in document.get("contracts", []):
            if (not isinstance(rule, dict) or rule.get("enabled", True) is not True or rule.get("risk") != "high"
                    or rule.get("provider") not in (None, provider) or rule.get("session_id") not in (None, sid)
                    or rule.get("scope", "any") not in ("tool_call", "any")
                    or "assistant" not in (rule.get("roles") or ["assistant"])):
                continue
            pattern = rule.get("regex") if rule.get("type") == "must_not" else rule.get("before_regex")
            if rule.get("type") == "must" and not pattern:
                continue                      # a completion obligation does not gate actions
            if rule.get("type") == "must_not" and rule.get("path_guard") is not None:
                # A typed contract stores no regex: it is derived from the guard (the same function the engine loads
                # it with) and run on the contract engine, whose time limit a hook needs.
                if _typed_match(rule["path_guard"], text):
                    return str(rule.get("id"))
                continue
            try:
                if not pattern or re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    return str(rule.get("id"))
            except re.error:
                return str(rule.get("id"))
    except Exception:
        return None
    return None


def _typed_match(guard, text) -> bool:
    """Whether a typed ``path_guard`` applies to this call text. Anything that cannot be decided applies (it asks)."""
    try:
        if contract_paths is None:
            return True
        from self_directing_mcp.audit.patterns import compile_contract
        return compile_contract(contract_paths.lower_mapping(guard)).search(text) is not None
    except Exception:
        return True


# --- fail-open responses ----------------------------------------------------------------------------------

def warning_text(event, error_class: str, kind: str) -> str:
    label = event or "hook"
    if kind == AUDIT_UNAVAILABLE:
        return (f"Tsukkomi audit UNAVAILABLE ({error_class}): this {label} was not verified; "
                "fail-open, not a compliance verdict.")
    return (f"Tsukkomi check FAILED ({error_class}): this {label} was not fully verified; "
            "fail-open, not a compliance verdict.")


def fail_open_response(event, reason: str, *, ask: str | None = None, banner: str | None = None) -> dict:
    """Never deny, never continue. PreToolUse asks only when a user-selected high-risk contract applies."""
    if event == "PostToolUse":
        return {}
    if event == "PreInvocation":
        text = reason + " Contract context was not injected; do not claim compliance."
        return {"injectSteps": [{"ephemeralMessage": text + (("\n" + banner) if banner else "")}]}
    if event == "PreToolUse" and ask:
        return {"decision": "ask", "reason": reason + " " + ask}
    return {"decision": "allow", "reason": reason}


def fail_open(event, payload, error_class: str, kind: str = CHECK_FAILED, *, mode: str | None = None,
              index_dir=None, contracts_path=None) -> dict:
    """The response for any internal hook error, after recording it. Stores and prints the class name only."""
    mode = mode if mode in MODES else resolve_mode()
    reason = warning_text(event, error_class, kind)
    recorded, banner, ask = False, None, None
    try:
        health = HookState(index_dir if index_dir is not None else resolve_index_dir())
        recorded = health.record_error(kind, event or "unknown", error_class)
        if event == "PreInvocation":
            banner = health.banner()
        elif event == "PostToolUse" and isinstance(payload, dict) and isinstance(payload.get("conversationId"), str):
            health.queue_warning(payload["conversationId"].lower()[:64],
                                 f"PostToolUse bookkeeping failed ({error_class}); recent tool results may be unrecorded.")
    except Exception:
        pass
    if event == "PreToolUse" and mode != "advisory" and isinstance(payload, dict):
        try:
            matched = high_risk_match(contracts_path if contracts_path is not None else resolve_contracts_path(),
                                      payload.get("conversationId"), payload.get("toolCall"))
        except Exception:
            matched = None
        if matched:
            ask = f"High-risk contract [{matched}] applies to this action and could not be verified; confirm manually."
    print(reason + ("" if recorded else " (hook health not recorded)"), file=sys.stderr)
    return fail_open_response(event, reason, ask=ask, banner=banner)
