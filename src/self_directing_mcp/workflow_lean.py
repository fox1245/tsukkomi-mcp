"""Kernel-replayed, natively compiled Lean workflow policy.

The checker, packaged sources, toolchain and cache must be outside model write
authority. Candidate elaboration runs in a separate OS sandbox. Independent
kernel replay and a native audit process do not load candidate extensions.
Statements use trusted core syntax and fully qualified approved definitions;
source-defined notation, attributes and instances are not used by the audit.
The accepted foundational axioms are propext, Classical.choice and Quot.sound.
Like leanchecker, this trusts structurally valid native .olean serialization;
it does not claim comparator's validated export/external-checker guarantees.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import stat
import tempfile
import time
from contextlib import contextmanager
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Iterator

from .workflow_process import run_process

STAGES = ("requirements", "formalize", "implement", "verify", "complete")
FACTS = ("approved", "obligations", "proofs", "tests", "fresh")
_ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
# Independent review pins: changing policy/proof source requires updating these
# reviewed bindings, not merely trusting whatever theorem a resource now contains.
_SOURCE_HASHES = {
    "AuditSupport.lean": "e90dfd0be3e4c28303bd437ba12241b217c8e8ffb4ef354f7afad16acd7fb075",
    "Main.lean": "0fb245cdb3495acfd46f22f281359cd02d3c1c06deb078117ada7327d3d5f0b3",
    "Policy.lean": "5ab11d38d4b66cbae915135b3d41b43d81de6a39379303db5d0912664ec48eae",
}
_THEOREMS = {
    "Workflow.one_step_safe": "∀ (s : Workflow.Stage) (c : Workflow.Choice) (f : Workflow.Facts), (Workflow.transition s c f).next = Workflow.Stage.complete → f.all = true",
    "Workflow.advance_safe": "∀ (s : Workflow.Snapshot) (i : Workflow.Input), Workflow.Safe (Workflow.advance s i)",
    "Workflow.sequence_safe": "∀ (inputs : List Workflow.Input) (s : Workflow.Snapshot), Workflow.Safe s → Workflow.Safe (Workflow.run s inputs)",
}
_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9']*(?:\.[A-Za-z_][A-Za-z_0-9']*)*\Z")


class LeanPolicyError(RuntimeError):
    """No authorization may be issued when trusted policy preparation fails."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sources() -> dict[str, bytes]:
    root = files("self_directing_mcp").joinpath("resources", "workflow")
    result = {name: root.joinpath(name).read_bytes() for name in _SOURCE_HASHES}
    if any(_sha(data) != _SOURCE_HASHES[name] for name, data in result.items()):
        raise LeanPolicyError("packaged Lean source does not match approved policy")
    return result


def policy_digest() -> str:
    _sources()
    return _sha(_json({"sources": _SOURCE_HASHES, "theorems": _THEOREMS,
                       "allowed_axioms": sorted(_ALLOWED_AXIOMS), "protocol": 3,
                       "verifier": _sha(files("self_directing_mcp").joinpath("workflow_lean.py").read_bytes()),
                       "runner": _sha(files("self_directing_mcp").joinpath("workflow_process.py").read_bytes())}).encode())


def _sanitize(text: str) -> str:
    text = re.sub(r"(?i)(bearer\s+|api[_-]?key\s*[=:]\s*)[^\s\"']+", r"\1[REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    text = re.sub(r"[^\x09\x0a\x0d\x20-\x7e\u00a0-\uffff]", "", text)
    return text[-12000:]


def _env(work: Path, binary_dir: Path) -> dict[str, str]:
    # Even trusted native compilation receives no credentials or host settings.
    return {"PATH": str(binary_dir) + os.pathsep + os.defpath,
            "HOME": str(work), "TMPDIR": str(work), "LC_ALL": "C.UTF-8",
            "LEAN_PATH": str(work)}


def _run(argv: list[str], work: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=work, env=_env(work, Path(argv[0]).parent), text=True,
                          encoding="utf-8", errors="replace", capture_output=True,
                          timeout=timeout, check=False)


def _checked(argv: list[str], work: Path, timeout: float) -> str:
    result = _run(argv, work, timeout)
    if result.returncode:
        raise LeanPolicyError(_sanitize(result.stdout + result.stderr))
    return result.stdout + result.stderr


def _toolchain(lean: str | Path | None, work: Path) -> dict[str, str]:
    requested = str(lean) if lean is not None else "lean"
    resolved = shutil.which(requested)
    if resolved is None:
        raise LeanPolicyError("Lean 4.34.1+ and its native C toolchain are required")
    executable = Path(resolved).resolve()
    version = _checked([str(executable), "--version"], work, 15).strip()
    match = re.search(r"version (\d+)\.(\d+)\.(\d+)", version)
    if not match or tuple(map(int, match.groups())) < (4, 34, 1):
        raise LeanPolicyError("Lean 4.34.1 or newer is required")
    prefix = Path(_checked([str(executable), "--print-prefix"], work, 15).strip()).resolve()
    native_lean = prefix / "bin" / ("lean.exe" if os.name == "nt" else "lean")
    leanc = prefix / "bin" / ("leanc.exe" if os.name == "nt" else "leanc")
    checker = prefix / "bin" / ("leanchecker.exe" if os.name == "nt" else "leanchecker")
    if not native_lean.is_file() or not leanc.is_file() or not checker.is_file():
        raise LeanPolicyError("Lean native compiler and independent leanchecker are required")
    return {"lean": str(native_lean), "leanc": str(leanc), "checker": str(checker),
            "prefix": str(prefix), "version": version,
            "digest": _toolchain_digest(prefix, version)}


@lru_cache(maxsize=4)
def _toolchain_digest(prefix: Path, version: str) -> str:
    digest = hashlib.sha256(version.encode())
    # The installation is trusted read-only. Bind the compiler, native libraries,
    # and kernel import artifacts, not just a potentially ambiguous version label.
    candidates = {prefix / "bin" / name for name in
                  (("lean.exe", "leanc.exe", "leanchecker.exe") if os.name == "nt"
                   else ("lean", "leanc", "leanchecker"))}
    for pattern in ("*.so", "*.so.*", "*.dll", "*.dylib", "*.a"):
        candidates.update((prefix / "lib" / "lean").glob(pattern))
        candidates.update((prefix / "bin").glob(pattern))
    candidates.update((prefix / "lib" / "lean").rglob("*.olean"))
    for path in sorted(candidates):
        digest.update(str(path.relative_to(prefix)).encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _lock(path: Path, timeout: float = 300) -> Iterator[None]:
    deadline = time.monotonic() + timeout
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise LeanPolicyError("timed out waiting for Lean cache build lock")
            time.sleep(.05)
    try:
        yield
    finally:
        path.rmdir()


def _lean_string(value: str) -> str:
    # Lean's string escapes differ from JSON for control characters.
    return '"' + "".join('\\"' if c == '"' else '\\\\' if c == '\\'
                         else f"\\u{{{ord(c):x}}}" if ord(c) < 32 else c for c in value) + '"'


def _audit_source(module: str, expectations: dict[str, str]) -> str:
    return f"import AuditSupport\nimport {module}\nset_option autoImplicit false\n" + "".join(
        f"workflow_audit {_lean_string(name)} against {_lean_string(statement)}\n"
        for name, statement in expectations.items())


def _reports(output: str, expected: set[str]) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for line in output.splitlines():
        if "TSUKKOMI_AUDIT " not in line:
            continue
        item = json.loads(line.split("TSUKKOMI_AUDIT ", 1)[1])
        name = item.get("theorem")
        if name not in expected or name in reports:
            raise LeanPolicyError("unexpected or duplicate kernel audit record")
        if not isinstance(item.get("axioms"), list) or not all(isinstance(x, str) for x in item["axioms"]):
            raise LeanPolicyError("invalid kernel axiom report")
        reports[name] = item
    if set(reports) != expected:
        raise LeanPolicyError("kernel audit did not produce all expected theorem records")
    return reports


def _write_sources(work: Path, sources: dict[str, bytes]) -> None:
    for name, content in sources.items():
        (work / name).write_bytes(content)


class LeanPolicy:
    def __init__(self, cache_dir: Path, lean: str | Path | None = None):
        self.cache_dir = Path(cache_dir).resolve()
        self.lean = lean
        self._build: dict | None = None

    def build(self) -> dict:
        policy_hash = policy_digest()
        if self._build is not None:
            return dict(self._build)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        toolchain = _toolchain(self.lean, self.cache_dir)
        key = _sha((policy_hash + toolchain["digest"]).encode())
        target = self.cache_dir / key
        with _lock(self.cache_dir / (key + ".lock")):
            metadata_file = target / "build.json"
            if metadata_file.is_file():
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                if metadata.get("policy_hash") != policy_hash or metadata.get("toolchain") != toolchain:
                    raise LeanPolicyError("Lean cache identity mismatch")
                self._check_artifacts(target, metadata)
            else:
                with tempfile.TemporaryDirectory(prefix="policy-", dir=self.cache_dir) as temp:
                    work = Path(temp)
                    _write_sources(work, _sources())
                    compiler = toolchain["lean"]
                    for module in ("Policy", "AuditSupport"):
                        argv = [compiler, "-o", module + ".olean", "-c", module + ".c"]
                        _checked([*argv, module + ".lean"], work, 180)
                    (work / "Audit.lean").write_text(_audit_source("Policy", _THEOREMS), encoding="utf-8")
                    output = _checked([compiler, "Audit.lean"], work, 180)
                    reports = _reports(output, set(_THEOREMS))
                    if any(set(report["axioms"]) - _ALLOWED_AXIOMS for report in reports.values()):
                        raise LeanPolicyError("policy theorem depends on a forbidden axiom")
                    _checked([compiler, "-c", "Main.c", "Main.lean"], work, 180)
                    executable_name = "workflow-policy.exe" if os.name == "nt" else "workflow-policy"
                    _checked([toolchain["leanc"], "-o", executable_name, "Main.c", "Policy.c"], work, 180)
                    audit_name = "workflow-audit.exe" if os.name == "nt" else "workflow-audit"
                    _checked([toolchain["leanc"], "-rdynamic", "-o", audit_name, "AuditSupport.c",
                              "-Wl,--whole-archive", "-lLean", "-lStd", "-lInit",
                              "-Wl,--no-whole-archive"], work, 180)
                    artifact_names = [executable_name, audit_name, "Policy.olean", "AuditSupport.olean"]
                    metadata = {"policy_hash": policy_hash, "toolchain": toolchain,
                                "theorems": reports, "executable_name": executable_name,
                                "artifacts": {name: _sha((work / name).read_bytes()) for name in artifact_names}}
                    (work / "build.json").write_text(_json(metadata), encoding="utf-8")
                    if target.exists():
                        raise LeanPolicyError("incomplete or untrusted Lean cache directory")
                    work.rename(target)
        metadata["executable"] = str(target / metadata["executable_name"])
        metadata["auditor"] = str(target / ("workflow-audit.exe" if os.name == "nt" else "workflow-audit"))
        self._build = metadata
        return dict(metadata)

    @staticmethod
    def _check_artifacts(target: Path, metadata: dict) -> None:
        executable_name = "workflow-policy.exe" if os.name == "nt" else "workflow-policy"
        audit_name = "workflow-audit.exe" if os.name == "nt" else "workflow-audit"
        expected = {executable_name, audit_name, "Policy.olean", "AuditSupport.olean"}
        if metadata.get("executable_name") != executable_name or set(metadata.get("artifacts", {})) != expected:
            raise LeanPolicyError("invalid Lean cache artifact manifest")
        for name, digest in metadata["artifacts"].items():
            if _sha((target / name).read_bytes()) != digest:
                raise LeanPolicyError("Lean cache artifact integrity failure")

    def evaluate(self, stage: str, choice_id: int, facts: dict[str, bool]) -> dict:
        fallback = "verify" if stage == "complete" else stage if stage in STAGES else "requirements"
        try:
            digest = policy_digest()
            if stage not in STAGES or type(choice_id) is not int or choice_id not in range(4):
                raise LeanPolicyError("invalid stage or classifier choice")
            if set(facts) != set(FACTS) or any(type(facts[name]) is not bool for name in FACTS):
                raise LeanPolicyError("facts must contain exactly the five boolean evidence fields")
            metadata = self.build()
            executable = Path(metadata["executable"])
            if _sha(executable.read_bytes()) != metadata["artifacts"][executable.name]:
                raise LeanPolicyError("native policy executable integrity failure")
            output = _checked([str(executable), str(STAGES.index(stage)), str(choice_id),
                               *("1" if facts[name] else "0" for name in FACTS)], executable.parent, 10)
            result = json.loads(output)
            if (not isinstance(result, dict) or set(result) != {"allowed", "next_stage", "failed"}
                    or type(result["allowed"]) is not bool or result["next_stage"] not in STAGES
                    or not isinstance(result["failed"], list)
                    or not all(isinstance(reason, str) for reason in result["failed"])):
                raise LeanPolicyError("malformed native policy response")
            return {**result, "policy_hash": digest}
        except (OSError, ValueError, TypeError, KeyError, AttributeError,
                subprocess.SubprocessError, LeanPolicyError) as exc:
            return {"allowed": False, "next_stage": fallback, "failed": ["lean_policy_error"],
                    "policy_hash": locals().get("digest", ""), "diagnostics": _sanitize(str(exc))}


def verify_proof(source: Path, *, source_sha256: str, theorem: str, statement: str,
                 cache_dir: Path, lean: str | Path | None = None,
                 timeout_sec: float = 60, cancel_event=None) -> dict:
    """Check immutable approved source, independently replay, then audit its type.

    ``incomplete`` means missing evidence or a sorry-based proof; ``failed`` means
    a rejected proof/source/type/axiom; ``error`` is checker/toolchain failure.
    These are proof outcomes, never substitutes for actual test execution.
    Candidate compiler stdout is never interpreted as audit evidence.
    """
    result = {"status": "error", "source_hash": "", "theorem": theorem,
              "statement": statement, "axioms": [], "diagnostics": ""}
    try:
        if (not _NAME.fullmatch(theorem) or not isinstance(statement, str) or not statement.strip()
                or type(timeout_sec) not in (int, float) or not 0 < timeout_sec <= 3600):
            raise LeanPolicyError("invalid proof verification parameters")
        if cancel_event is not None and cancel_event.is_set():
            return {**result, "execution_status": "cancelled", "diagnostics": "proof verification cancelled"}
        source = Path(source)
        if not source.is_file():
            return {**result, "status": "incomplete", "diagnostics": "approved proof source is missing"}
        content = source.read_bytes()
        result["source_hash"] = _sha(content)
        if result["source_hash"] != source_sha256:
            return {**result, "status": "failed", "diagnostics": "proof source differs from owner-approved SHA-256"}
        cache = Path(cache_dir).resolve()
        metadata = LeanPolicy(cache, lean=lean).build()
        toolchain = metadata["toolchain"]
        auditor = Path(metadata["auditor"])
        if _sha(auditor.read_bytes()) != metadata["artifacts"][auditor.name]:
            raise LeanPolicyError("native audit executable integrity failure")
        prefix = Path(toolchain["prefix"])
        key = _sha(_json({"source": source_sha256, "theorem": theorem, "statement": statement,
                         "toolchain": toolchain["digest"], "policy": policy_digest()}).encode())
        record = cache / ("proof-" + key + ".json")
        with _lock(cache / ("proof-" + key + ".lock"), timeout_sec):
            if cancel_event is not None and cancel_event.is_set():
                return {**result, "execution_status": "cancelled", "diagnostics": "proof verification cancelled"}
            if record.is_file():
                cached = json.loads(record.read_text(encoding="utf-8"))
                if (cached.get("source_hash") != source_sha256 or cached.get("theorem") != theorem
                        or cached.get("statement") != statement or cached.get("status") != "passed"
                        or set(cached.get("axioms", [])) - _ALLOWED_AXIOMS):
                    raise LeanPolicyError("invalid cached proof identity")
                return cached
            with tempfile.TemporaryDirectory(prefix="proof-", dir=cache) as temp:
                root = Path(temp)
                work = root / "compile"
                work.mkdir()
                (work / "Domain.lean").write_bytes(content)
                compiled = run_process(
                    [toolchain["lean"], "-o", "Domain.olean", "Domain.lean"],
                    workspace=work, writable=True, timeout_sec=timeout_sec,
                    cancel_event=cancel_event, read_only_paths=(prefix,))
                result["compile_output_hash"] = compiled["output_hash"]
                if compiled["status"] != "passed":
                    return {**result, "status": "failed" if compiled["status"] == "failed" else "error",
                            "execution_status": compiled["status"],
                            "diagnostics": f"isolated Lean compilation {compiled['status']} "
                                           f"(exit {compiled['exit_code']})"}
                # The compiler cannot modify this new workspace or the auditor.
                # Copy only module artifacts, not candidate scripts, plugins,
                # replacement imports, audit output, or executable helpers.
                audit_work = root / "audit"
                audit_work.mkdir()
                for name in ("Domain.olean", "Domain.olean.server", "Domain.olean.private",
                             "Domain.ir", "Domain.ir.server", "Domain.ir.private"):
                    artifact = work / name
                    try:
                        info = artifact.lstat()
                    except FileNotFoundError:
                        if name == "Domain.olean":
                            raise LeanPolicyError("compiler did not produce a proof module")
                        continue
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise LeanPolicyError("proof module artifact is not a private regular file")
                    (audit_work / name).write_bytes(artifact.read_bytes())
                # Core imports come from the pinned read-only toolchain. Replay
                # this candidate module, not the entire unchanged Lean library.
                replayed = run_process(
                    [toolchain["checker"], "Domain"], workspace=audit_work,
                    writable=False, timeout_sec=timeout_sec, cancel_event=cancel_event,
                    read_only_paths=(prefix,))
                result["replay_output_hash"] = replayed["output_hash"]
                if replayed["status"] != "passed":
                    return {**result, "status": "failed" if replayed["status"] == "failed" else "error",
                            "execution_status": replayed["status"],
                            "diagnostics": f"independent Lean kernel replay {replayed['status']} "
                                           f"(exit {replayed['exit_code']})"}
                audited = run_process(
                    [str(auditor), "Domain", theorem, statement], workspace=audit_work,
                    writable=False, timeout_sec=timeout_sec, cancel_event=cancel_event,
                    read_only_paths=(prefix, auditor.parent))
                result["audit_output_hash"] = audited["output_hash"]
                if audited["status"] != "passed":
                    return {**result, "status": "failed" if audited["status"] == "failed" else "error",
                            "execution_status": audited["status"],
                            "diagnostics": f"trusted standalone theorem audit {audited['status']} "
                                           f"(exit {audited['exit_code']})"}
                # Only this trusted executable's entire stdout is the protocol.
                # There is no marker search through candidate compiler messages.
                report = json.loads(audited["stdout"])
                if (not isinstance(report, dict) or set(report) != {"theorem", "axioms"}
                        or report["theorem"] != theorem or not isinstance(report["axioms"], list)
                        or not all(isinstance(item, str) for item in report["axioms"])):
                    raise LeanPolicyError("invalid standalone kernel audit record")
                axioms = report["axioms"]
                forbidden = set(axioms) - _ALLOWED_AXIOMS
                status = "incomplete" if "sorryAx" in forbidden else "failed" if forbidden else "passed"
                result.update(status=status, axioms=axioms,
                              diagnostics="forbidden theorem assumptions" if forbidden else "")
                if status == "passed":
                    temporary = root / "result.json"
                    temporary.write_text(_json(result), encoding="utf-8")
                    temporary.replace(record)
                return result
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            subprocess.SubprocessError, LeanPolicyError) as exc:
        return {**result, "status": "error", "diagnostics": _sanitize(str(exc))}
