"""Owner-approved workflows; observations authorize transitions, never model claims.

The approval, installed checker, and state directory are host trust boundaries.
Keep them outside agent write authority; child programs additionally require the
fail-closed Linux sandbox. Readiness is not successful owned execution.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import sys
import tempfile
import time
from typing import Any

from self_directing_mcp.config import Settings
from self_directing_mcp.index.locking import index_lock
from self_directing_mcp.neograph_runtime import graph_operation, run_stages
from self_directing_mcp.workflow_jev import MAPPING_HASH, classify
from self_directing_mcp.workflow_lean import LeanPolicy, policy_digest, verify_proof
from self_directing_mcp.workflow_process import run_process
from self_directing_mcp.workflow_requirements import ApprovalError, ApprovedProject


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def host_policy_digest() -> str:
    """Bind persisted decisions to the installed orchestration, not just Lean."""
    root = files("self_directing_mcp")
    digest = hashlib.sha256()
    for name in ("workflow.py", "workflow_requirements.py", "workflow_jev.py",
                 "workflow_lean.py", "workflow_process.py"):
        digest.update(name.encode())
        digest.update(root.joinpath(name).read_bytes())
    return digest.hexdigest()


def _deadline_check(deadline: float | None, cancel_event=None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise TimeoutError("workflow request cancelled")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("workflow request expired")


class WorkflowController:
    def __init__(self, approval_dir: Path, state_dir: Path, *, lean: str | Path | None = None):
        self.project = ApprovedProject.load(Path(approval_dir))
        self.state_dir = Path(state_dir).expanduser().resolve()
        if self.state_dir.is_relative_to(self.project.root):
            raise ValueError("workflow state must be outside the agent workspace")
        if self.state_dir == self.project.approval_dir or self.state_dir.is_relative_to(self.project.approval_dir):
            raise ValueError("workflow state must be separate from immutable approval")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lean = lean
        self.policy = LeanPolicy(self.state_dir / "lean", lean=lean)
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, stage TEXT NOT NULL, revision INTEGER NOT NULL,
                    versions TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY, session TEXT NOT NULL, kind TEXT NOT NULL,
                    name TEXT NOT NULL, versions TEXT NOT NULL, status TEXT NOT NULL,
                    payload TEXT NOT NULL, created REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS observations_session ON observations(session, kind, name, id);
                CREATE TABLE IF NOT EXISTS classifications (
                    id TEXT PRIMARY KEY, session TEXT NOT NULL, revision INTEGER NOT NULL,
                    versions TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS grants (
                    token TEXT PRIMARY KEY, session TEXT NOT NULL, revision INTEGER NOT NULL,
                    versions TEXT NOT NULL, action TEXT NOT NULL, classification TEXT NOT NULL,
                    status TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, session TEXT NOT NULL, kind TEXT NOT NULL,
                    payload TEXT NOT NULL, created REAL NOT NULL);
            """)

    @contextmanager
    def _db(self):
        with index_lock(self.state_dir, timeout=15):
            db = sqlite3.connect(self.state_dir / "workflow.sqlite", timeout=15)
            db.row_factory = sqlite3.Row
            try:
                with db:
                    yield db
            finally:
                db.close()

    def _key(self, session_id: str, provider: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
            raise ValueError("invalid workflow session ID")
        if provider not in {"codex", "grokbot", "omp", "agy"}:
            raise ValueError("invalid workflow provider")
        return _digest([self.project.approval_digest, provider, session_id])

    @staticmethod
    def _event(db, key, kind, payload):
        db.execute("INSERT INTO events(session,kind,payload,created) VALUES (?,?,?,?)",
                   (key, kind, _json(payload), time.time()))

    def _observe(self, db, key):
        # Reload immutable approval rather than trusting an old in-memory object.
        self.project = ApprovedProject.load(self.project.approval_dir)
        versions = {**self.project.versions(), "kernel": policy_digest(), "mapping": MAPPING_HASH,
                    "host_policy": host_policy_digest()}
        encoded = _json(versions)
        row = db.execute("SELECT * FROM sessions WHERE id=?", (key,)).fetchone()
        if row is None:
            db.execute("INSERT INTO sessions VALUES (?,?,?,?)", (key, "requirements", 0, encoded))
        elif row["versions"] != encoded:
            db.execute("UPDATE sessions SET stage=?,revision=revision+1,versions=? WHERE id=?",
                       ("verify" if row["stage"] == "complete" else row["stage"], encoded, key))
            db.execute("UPDATE grants SET status='stale' WHERE session=? AND status='authorized'", (key,))
            self._event(db, key, "versions_changed", {"versions": versions})
        row = db.execute("SELECT * FROM sessions WHERE id=?", (key,)).fetchone()
        obligations = self.project.obligations(self.project.changed_paths())
        problems = self.project.lint()
        tests = sorted({item for req in obligations for item in req["tests"]})
        proofs = sorted({item for req in obligations for item in req["proofs"]})
        evidence = []
        for kind, names in (("test", tests), ("proof", proofs)):
            for name in names:
                observation = db.execute(
                    "SELECT * FROM observations WHERE session=? AND kind=? AND name=? ORDER BY id DESC LIMIT 1",
                    (key, kind, name)).fetchone()
                evidence.append({"kind": kind, "name": name,
                                 "status": observation["status"] if observation else "missing",
                                 "current": bool(observation and observation["versions"] == encoded),
                                 "observation_id": observation["id"] if observation else None})
        current = lambda kind: all(e["current"] and e["status"] == "passed"
                                   for e in evidence if e["kind"] == kind)
        facts = {"approved": not problems,
                 "obligations": bool(obligations) and bool(tests) and bool(proofs),
                 "proofs": bool(proofs) and current("proof"),
                 "tests": bool(tests) and current("test"),
                 "fresh": not problems and all(e["current"] for e in evidence if e["kind"] == "proof")}
        return {"stage": row["stage"], "revision": row["revision"], "versions": versions,
                "requirements": [r["id"] for r in obligations], "tests": tests, "proofs": proofs,
                "evidence": evidence, "problems": problems, "facts": facts}

    def status(self, session_id: str, provider: str = "agy", *, _deadline=None, cancel_event=None) -> dict:
        _deadline_check(_deadline, cancel_event)
        key = self._key(session_id, provider)
        with self._db() as db:
            _deadline_check(_deadline, cancel_event)
            state = self._observe(db, key)
            history = db.execute("SELECT id,kind,payload,created FROM events WHERE session=? ORDER BY id DESC LIMIT 20",
                                 (key,)).fetchall()
        return {"ok": True, **state, "history": [{**dict(row), "payload": json.loads(row["payload"])} for row in history]}

    @graph_operation("workflow_classify")
    def classify(self, session_id: str, provider: str = "agy", *, _deadline=None, cancel_event=None) -> dict:
        _deadline_check(_deadline, cancel_event)
        key = self._key(session_id, provider)
        with self._db() as db:
            _deadline_check(_deadline, cancel_event)
            before = self._observe(db, key)
        if before["problems"]:
            return {"ok": False, "status": "approval_changed", "choice_id": 0, "problems": before["problems"]}
        if self.project.manifest.get("external_context", "disabled") != "synthetic":
            return {"ok": False, "status": "external_context_not_approved", "choice_id": 0}
        try:
            changes = self.project.change_context()
        except ApprovalError as exc:
            return {"ok": False, "status": "context_unavailable", "choice_id": 0, "reason": str(exc)}
        context = {"requirements": self.project.context(), "stage": before["stage"],
                   "changes": changes, "evidence": before["evidence"], "versions": before["versions"]}
        settings = Settings()
        result = classify(context, key_file=settings.openrouter_api_key_file,
                          api_key=settings.openrouter_api_key,
                          threshold=self.project.manifest.get("confidence_threshold", .7))
        payload = result.as_dict()
        classification_id = secrets.token_hex(16)
        with self._db() as db:
            after = self._observe(db, key)
            if (before["revision"], before["versions"]) != (after["revision"], after["versions"]):
                payload.update(status="stale", choice_id=0, reason="state changed during classification")
            _deadline_check(_deadline, cancel_event)
            db.execute("INSERT INTO classifications VALUES (?,?,?,?,?)",
                       (classification_id, key, after["revision"], _json(after["versions"]), _json(payload)))
            self._event(db, key, "classification", {"id": classification_id, **payload})
        return {"ok": payload["status"] == "ok", "classification_id": classification_id, **payload}

    @graph_operation("workflow_checks")
    def run_checks(self, session_id: str, provider: str = "agy", kind: str = "all", *,
                   _deadline=None, cancel_event=None) -> dict:
        _deadline_check(_deadline, cancel_event)
        if kind not in {"all", "proof", "test"}:
            raise ValueError("kind must be all, proof, or test")
        key = self._key(session_id, provider)
        observations = []
        with self._db() as db:
            _deadline_check(_deadline, cancel_event)
            before = self._observe(db, key)
            if before["problems"]:
                return {"ok": False, "status": "approval_changed", "problems": before["problems"]}
            _deadline_check(_deadline, cancel_event)
            db.execute("UPDATE sessions SET revision=revision+1,stage=? WHERE id=?",
                       ("verify" if before["stage"] == "complete" else before["stage"], key))
            db.execute("UPDATE grants SET status='stale' WHERE session=? AND status='authorized'", (key,))
            # Interrupted verification invalidates earlier passes before work.
            selected = [(k, name) for k, names in (("proof", before["proofs"]), ("test", before["tests"]))
                        for name in names if kind in {"all", k}]
            for check_kind, name in selected:
                self._observation(db, key, check_kind, name, before["versions"], "running", {})
        for check_kind, name in selected:
            _deadline_check(_deadline, cancel_event)
            started = time.monotonic()
            with self._db() as db:
                snapshot = self._observe(db, key)
            if snapshot["versions"] != before["versions"] or snapshot["problems"]:
                result = {"status": "stale", "reason": "inputs changed before execution", "started": False}
            else:
                try:
                    with tempfile.TemporaryDirectory(prefix="check-", dir=self.state_dir) as temporary:
                        work = Path(temporary) / "work"
                        hashes = self.project.materialize(work)
                        with self._db() as db:
                            materialized = self._observe(db, key)
                        if materialized["versions"] != before["versions"] or materialized["problems"]:
                            result = {"status": "stale", "reason": "inputs changed during snapshot", "started": False}
                        elif check_kind == "proof":
                            spec = self.project.manifest["proofs"][name]
                            result = verify_proof(
                                work / spec["path"], source_sha256=self.project.approved_hash(spec["path"]),
                                theorem=spec["theorem"], statement=spec["statement"],
                                cache_dir=self.state_dir / "proofs", lean=self.lean,
                                timeout_sec=self._budget(spec.get("timeout_sec", 60), _deadline),
                                cancel_event=cancel_event)
                        else:
                            spec = self.project.manifest["tests"][name]
                            argv = [sys.executable if arg == "{python}" else arg for arg in spec["argv"]]
                            result = run_process(argv, workspace=work, writable=False,
                                                 timeout_sec=self._budget(spec.get("timeout_sec", 60), _deadline),
                                                 cancel_event=cancel_event)
                        result["snapshot_hash"] = _digest(hashes)
                except ApprovalError as exc:
                    result = {"status": "stale", "reason": str(exc), "started": False}
            result = self._public_result(result)
            result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
            with self._db() as db:
                after = self._observe(db, key)
                current = after["versions"] == before["versions"] and not after["problems"]
                evidence_status = result["status"] if current else "stale"
                result["observed_status"] = result["status"]
                result["status"] = evidence_status
                observation_id = self._observation(db, key, check_kind, name, before["versions"], evidence_status, result)
                self._event(db, key, "verification", {"id": observation_id, "kind": check_kind,
                                                       "name": name, **result})
            observations.append({"id": observation_id, "kind": check_kind, "name": name, **result})
            if cancel_event is not None and cancel_event.is_set():
                break
        return {"ok": bool(observations) and len(observations) == len(selected)
                and all(x["status"] == "passed" for x in observations),
                "observations": observations, "state": self.status(session_id, provider)}

    @staticmethod
    def _budget(seconds, deadline):
        if deadline is not None:
            _deadline_check(deadline)
            seconds = min(seconds, deadline - time.monotonic())
        return max(.001, seconds)

    @staticmethod
    def _observation(db, key, kind, name, versions, status, payload):
        return db.execute("INSERT INTO observations(session,kind,name,versions,status,payload,created) VALUES (?,?,?,?,?,?,?)",
                          (key, kind, name, _json(versions), status, _json(payload), time.time())).lastrowid

    @staticmethod
    def _public_result(result):
        return {key: value for key, value in result.items() if key not in {"stdout", "stderr"}}

    def _classification(self, db, key, classification_id, state):
        row = db.execute("SELECT * FROM classifications WHERE id=? AND session=?", (classification_id, key)).fetchone()
        if row is None:
            return None, "missing_classification"
        payload = json.loads(row["payload"])
        if row["revision"] != state["revision"] or row["versions"] != _json(state["versions"]):
            return None, "stale_classification"
        if payload.get("status") != "ok":
            return None, "classification_" + str(payload.get("status", "missing"))
        return payload, None

    def transition(self, session_id: str, classification_id: str, provider: str = "agy", *,
                   _deadline=None, cancel_event=None) -> dict:
        key = self._key(session_id, provider)
        data = {}
        with self._db() as db:
            def observe():
                _deadline_check(_deadline, cancel_event)
                data["state"] = self._observe(db, key)
            def classify_request():
                data["classification"], data["error"] = self._classification(db, key, classification_id, data["state"])
            def evaluate():
                state = data["state"]
                choice = data["classification"]["choice_id"] if not data["error"] else 0
                data["decision"] = self.policy.evaluate(state["stage"], choice, state["facts"])
                if data["error"]:
                    data["decision"].update(allowed=False, failed=[data["error"]])
            def record():
                _deadline_check(_deadline, cancel_event)
                state, decision = data["state"], data["decision"]
                # Lean completion is readiness only here. Only the owned action
                # executor may persist a successfully completed workflow.
                if decision["allowed"] and decision["next_stage"] == "complete":
                    decision.update(policy_next_stage="complete", next_stage=state["stage"], ready_for_completion=True)
                current = self._observe(db, key)
                _deadline_check(_deadline, cancel_event)
                if current["versions"] != state["versions"]:
                    decision.update(allowed=False, next_stage=current["stage"], failed=["inputs_changed_during_check"])
                elif decision["allowed"] or decision["next_stage"] != state["stage"]:
                    db.execute("UPDATE sessions SET stage=?,revision=revision+1 WHERE id=?", (decision["next_stage"], key))
                self._event(db, key, "transition", {**decision, "requirements": state["requirements"],
                                                    "classification_id": classification_id, "versions": state["versions"]})
            execution = run_stages("workflow_transition", [("observe", observe), ("classify_request", classify_request),
                                                            ("lean_transition", evaluate), ("record", record)])
        return {"ok": data["decision"]["allowed"], **data["decision"], "execution": execution,
                "requirements": data["state"]["requirements"], "evidence": data["state"]["evidence"],
                "allowed_next_actions": ["read_evidence", "revise_implementation", "run_checks", "classify"]}

    def is_completion_action(self, action):
        name = action.get("tool_name")
        arguments = _json(action.get("arguments", {}))
        return any(rule["tool_name"] == name and re.search(rule["argument_pattern"], arguments)
                   for rule in self.project.manifest["completion_actions"])

    @graph_operation("workflow_authorize")
    def authorize(self, session_id: str, classification_id: str, action: dict, provider: str = "agy", *,
                  _deadline=None, cancel_event=None) -> dict:
        _deadline_check(_deadline, cancel_event)
        # The owned executor accepts argv, not interpolated shell strings.
        self._command(action)
        if not self.is_completion_action(action):
            return {"ok": False, "failed": ["action_not_in_approved_policy"]}
        key = self._key(session_id, provider)
        with self._db() as db:
            _deadline_check(_deadline, cancel_event)
            state = self._observe(db, key)
            classification, error = self._classification(db, key, classification_id, state)
            decision = self.policy.evaluate(state["stage"], classification["choice_id"] if classification else 0, state["facts"])
            if error or not decision["allowed"] or decision["next_stage"] != "complete":
                return {"ok": False, "failed": [error] if error else decision["failed"],
                        "requirements": state["requirements"], "evidence": state["evidence"],
                        "allowed_next_actions": ["read_evidence", "revise_implementation", "run_checks", "classify"]}
            _deadline_check(_deadline, cancel_event)
            # Re-read content after the checker, immediately before minting the grant.
            current = self._observe(db, key)
            if current["versions"] != state["versions"]:
                return {"ok": False, "failed": ["inputs_changed_during_check"]}
            token = secrets.token_urlsafe(32)
            _deadline_check(_deadline, cancel_event)
            db.execute("INSERT INTO grants VALUES (?,?,?,?,?,?,?)",
                       (_digest(token), key, state["revision"], _json(state["versions"]), _json(action), classification_id, "authorized"))
            self._event(db, key, "authorized", {"action_hash": _digest(action), "versions": state["versions"],
                                               "revision": state["revision"], "classification_id": classification_id})
            _deadline_check(_deadline, cancel_event)
        return {"ok": True, "grant": token, "action_hash": _digest(action), "revision": state["revision"],
                "versions": state["versions"], "action_executed": False}

    def _command(self, action):
        if set(action) != {"tool_name", "arguments"} or action["tool_name"] != "workflow_command":
            raise ValueError("owned executor requires a workflow_command action")
        args = action["arguments"]
        if not isinstance(args, dict) or set(args) != {"argv", "cwd"} or args["cwd"] != ".":
            raise ValueError("command must bind argv and cwd='.' (approved project root)")
        argv = args["argv"]
        if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or not a or "\0" in a for a in argv):
            raise ValueError("argv must contain nonempty strings without NUL")
        return [sys.executable if arg == "{python}" else arg for arg in argv]

    @graph_operation("workflow_execute")
    def execute(self, session_id: str, grant: str, action: dict, provider: str = "agy", *,
                _deadline=None, cancel_event=None) -> dict:
        _deadline_check(_deadline, cancel_event)
        argv = self._command(action)
        key = self._key(session_id, provider)
        with self._db() as db:
            _deadline_check(_deadline, cancel_event)
            state = self._observe(db, key)
            row = db.execute("SELECT * FROM grants WHERE token=? AND session=?", (_digest(grant), key)).fetchone()
            if (row is None or row["status"] != "authorized" or row["action"] != _json(action)
                    or row["revision"] != state["revision"] or row["versions"] != _json(state["versions"])):
                return {"ok": False, "failed": ["missing_used_or_changed_grant"], "action_executed": False}
            classification, error = self._classification(db, key, row["classification"], state)
            decision = self.policy.evaluate(state["stage"], classification["choice_id"] if classification else 0, state["facts"])
            if error or not decision["allowed"] or decision["next_stage"] != "complete":
                return {"ok": False, "failed": [error] if error else decision["failed"], "action_executed": False}
            current = self._observe(db, key)
            if current["versions"] != state["versions"] or current["revision"] != state["revision"]:
                return {"ok": False, "failed": ["inputs_changed_during_check"], "action_executed": False}
            timeout = self._budget(60, _deadline)
            _deadline_check(_deadline, cancel_event)
            db.execute("UPDATE grants SET status='started' WHERE token=?", (_digest(grant),))
            self._event(db, key, "execution_committed", {"action_hash": _digest(action), "process_started": False})
            _deadline_check(_deadline, cancel_event)
            db.commit()  # A crash consumes the grant but does not claim execution.
            try:
                result = self._public_result(run_process(argv, workspace=self.project.root, writable=True,
                                                         timeout_sec=timeout, cancel_event=cancel_event))
            except Exception:
                # The committed grant may have started work before a runner or
                # observation fault. Never turn uncertainty into \"not executed\".
                result = {"status": "runner_error", "started": True, "outcome_unknown": True}
            try:
                after = self._observe(db, key)
                unchanged = after["versions"] == state["versions"] and not after["problems"]
            except (ApprovalError, OSError) as exc:
                unchanged = False
                result["observation_error"] = type(exc).__name__
            success = result["status"] == "passed" and unchanged and not result.get("outcome_unknown", False)
            outcome = ("succeeded" if success else "outcome_unknown" if result.get("outcome_unknown")
                       else "failed" if result["started"] else "not_started")
            db.execute("UPDATE grants SET status=? WHERE token=?", (outcome, _digest(grant)))
            db.execute("UPDATE sessions SET stage=?,revision=revision+1 WHERE id=?", ("complete" if success else "verify", key))
            self._event(db, key, outcome, {"action_hash": _digest(action), "versions_unchanged": unchanged, **result})
        return {"ok": success, "action_executed": None if result.get("outcome_unknown") else result["started"],
                "outcome": outcome, "versions_unchanged": unchanged, **result}


def _optional_path(name):
    value = os.environ.get(name)
    return Path(value) if value else None


def configured_workflow() -> WorkflowController | None:
    approval = _optional_path("TSUKKOMI_WORKFLOW_APPROVAL")
    state = _optional_path("TSUKKOMI_WORKFLOW_STATE")
    if approval is None and state is None:
        return None
    if approval is None or state is None:
        raise ValueError("both workflow approval and state paths must be host-configured")
    return WorkflowController(approval, state, lean=os.environ.get("TSUKKOMI_LEAN"))


def guard_host_action(session_id: str, action: dict, provider: str = "agy") -> dict | None:
    controller = configured_workflow()
    if controller is None or not controller.is_completion_action(action):
        return None
    state = controller.status(session_id, provider)
    details = {"stage": state["stage"], "requirements": state["requirements"], "evidence": state["evidence"],
               "allowed_next_actions": ["workflow_run_checks", "workflow_classify", "workflow_authorize"]}
    return {"decision": "deny", "reason": "Managed completion requires workflow_authorize and workflow_execute; "
            "a hook allowance is not an execution grant. " + _json(details), **details}
