"""Behavioral coverage for immutable owner approval and workspace obligations."""
from __future__ import annotations

import copy
import hashlib
import errno
import json
from pathlib import Path

import pytest

from self_directing_mcp import workflow_requirements as requirements_module

from self_directing_mcp.workflow_requirements import ApprovalError, ApprovedProject, approve_project


def _project(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "project"
    root.mkdir()
    files = {
        "PRD.md": "# Product\n\n## Safety\nThe result must be safe.\n\n## Optional\nThe extra result must be safe.\n",
        "specs/safety.txt": "safe means True\n",
        "tests/test_safety.py": "def test_safe():\n    assert True\n",
        "proofs/Safety.lean": "theorem safety : True := True.intro\n",
        "src/main.py": "SAFE = True\n",
        "src/extra.py": "EXTRA = True\n",
        "config/policy.json": '{"enforce": true}\n',
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    requirement = {
        "id": "R1", "text": "Preserve safety", "required": True, "status": "approved",
        "source": {"path": "PRD.md", "section": "Safety", "quote": "The result must be safe."},
        "code": ["src/main.py"], "specs": ["specs/safety.txt"],
        "tests": ["T1"], "proofs": ["P1"],
    }
    optional = copy.deepcopy(requirement)
    optional.update(id="R2", text="Preserve optional safety", required=False, code=["src/extra*.py"])
    optional["source"] = {"path": "PRD.md", "section": "Optional", "quote": "The extra result must be safe."}
    optional["depends_on"] = ["R1"]
    manifest = {
        "schema_version": 1, "prd": "PRD.md", "scope": ["src/**/*.py"],
        "requirements": [requirement, optional],
        "tests": {"T1": {"argv": ["python", "-m", "pytest", "tests/test_safety.py"],
                         "paths": ["tests/test_safety.py"]}},
        "proofs": {"P1": {"path": "proofs/Safety.lean", "theorem": "safety", "statement": "True"}},
        "completion_actions": [{"tool_name": "publish", "argument_pattern": ".*"}],
        "protected_paths": ["config/*.json"], "class_mapping_version": "1",
    }
    _manifest(root, manifest)
    return root, manifest


def _manifest(root: Path, manifest: dict) -> None:
    (root / "requirements.json").write_text(json.dumps(manifest), encoding="utf-8")


def _approve(tmp_path: Path, root: Path) -> ApprovedProject:
    return approve_project(root, Path("requirements.json"), tmp_path / "approval")


def test_approval_reopens_and_preserves_sources_after_workspace_edits(tmp_path):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    original = (root / "PRD.md").read_bytes()
    (root / "PRD.md").write_text("# Product\nWeakened requirements\n")
    reopened = ApprovedProject.load(approved.approval_dir)
    assert reopened.approval_digest == approved.approval_digest
    assert reopened.context()["requirements"][0]["source"]["quote"] == "The result must be safe."
    blob = approved.approval_dir / "blobs" / hashlib.sha256(original).hexdigest()
    assert blob.read_bytes() == original
    assert any(item["code"] == "approval_invalidated" for item in reopened.lint())


@pytest.mark.parametrize("relative", ["PRD.md", "requirements.json", "tests/test_safety.py", "specs/safety.txt", "config/policy.json"])
def test_weakening_or_deleting_protected_sources_invalidates_approval(tmp_path, relative):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    previous = approved.versions()
    (root / relative).write_text("weakened\n")
    assert relative in approved.changed_paths()
    assert approved.versions() != previous
    assert any(relative in error["message"] for error in approved.lint())
    (root / relative).unlink()
    assert any(relative in error["message"] for error in approved.lint())


def test_fresh_owner_approval_is_required_and_existing_approval_never_overwritten(tmp_path):
    root, manifest = _project(tmp_path)
    approved = _approve(tmp_path, root)
    original = (approved.approval_dir / "approval.json").read_bytes()
    manifest["requirements"][0]["text"] = "Revised owner-approved wording"
    _manifest(root, manifest)
    with pytest.raises(ApprovalError, match="already exists"):
        _approve(tmp_path, root)
    assert (approved.approval_dir / "approval.json").read_bytes() == original
    revised = approve_project(root, Path("requirements.json"), tmp_path / "revised")
    assert revised.approval_digest != approved.approval_digest
    assert revised.lint() == []


def test_added_deleted_and_unknown_code_changes_are_versioned(tmp_path):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    before = approved.versions()
    (root / "src/main.py").unlink()
    (root / "src/new.py").write_text("NEW = True\n")
    (root / "src/nested").mkdir()
    (root / "src/nested/extra_new.py").write_text("NESTED = True\n")
    assert approved.changed_paths() == ["src/main.py", "src/nested/extra_new.py", "src/new.py"]
    assert approved.versions()["code"] != before["code"]
    assert approved.versions()["test"] == before["test"]


def test_required_changed_and_dependency_obligations_cannot_be_hidden(tmp_path):
    root, manifest = _project(tmp_path)
    dependency = copy.deepcopy(manifest["requirements"][1])
    dependency.update(id="R3", code=["src/unused.py"], depends_on=[])
    manifest["requirements"].append(dependency)
    manifest["requirements"][1]["depends_on"] = ["R3"]
    _manifest(root, manifest)
    approved = _approve(tmp_path, root)
    assert [item["id"] for item in approved.obligations()] == ["R1"]
    (root / "src/extra.py").write_text("EXTRA = False\n")
    assert [item["id"] for item in approved.obligations([])] == ["R1", "R2", "R3"]
    assert [item["id"] for item in approved.obligations(["src/main.py"])] == ["R1", "R2", "R3"]


def test_new_files_matching_optional_code_activate_obligation(tmp_path):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    (root / "src/extra_new.py").write_text("EXTRA = True\n")
    assert [item["id"] for item in approved.obligations()] == ["R1", "R2"]


def test_context_is_state_independent_and_cannot_mutate_approval(tmp_path):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    initial = approved.context()
    altered = approved.context()
    altered["requirements"][0]["tests"] = []
    (root / "src/main.py").write_text("SAFE = False\n")
    assert approved.context() == initial
    assert approved.obligations()[0]["tests"] == ["T1"]


def test_proof_body_changes_invalidate_approval_and_spec_fingerprint(tmp_path):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    before = approved.versions()
    (root / "proofs/Safety.lean").write_text("theorem safety : True := by trivial\n")
    assert approved.versions()["spec"] != before["spec"]
    assert approved.manifest["proofs"]["P1"]["statement"] == "True"
    assert any(error["code"] == "approval_invalidated" and "proofs/Safety.lean" in error["message"]
               for error in approved.lint())
    with pytest.raises(ApprovalError, match="fresh owner approval"):
        approved.materialize(tmp_path / "snapshot")


def test_proof_also_declared_as_spec_is_immutable(tmp_path):
    root, manifest = _project(tmp_path)
    manifest["requirements"][0]["specs"].append("proofs/Safety.lean")
    _manifest(root, manifest)
    approved = _approve(tmp_path, root)
    (root / "proofs/Safety.lean").write_text("theorem safety : False := by sorry\n")
    assert any("proofs/Safety.lean" in error["message"] for error in approved.lint())


def test_new_protected_file_invalidates_policy(tmp_path):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    before = approved.versions()["policy"]
    (root / "config/alternate.json").write_text("{}\n")
    assert approved.versions()["policy"] != before
    assert any("config/alternate.json" in error["message"] for error in approved.lint())


def test_caches_are_not_workspace_changes(tmp_path):
    root, manifest = _project(tmp_path)
    manifest["scope"] = ["src/**"]
    _manifest(root, manifest)
    approved = _approve(tmp_path, root)
    before = approved.versions()
    cache = root / "src/__pycache__"
    cache.mkdir()
    (cache / "main.pyc").write_bytes(b"cache")
    for name in ("main.pyc", "main.pyo", "Injected.olean", "Injected.ilean"):
        (root / "src" / name).write_bytes(b"unapproved generated artifact")
    assert approved.changed_paths() == []
    assert approved.versions() == before
    destination = tmp_path / "snapshot"
    hashes = approved.materialize(destination)
    assert {name for name in hashes if name.startswith("src/")} == {"src/main.py", "src/extra.py"}
    assert sorted(path.name for path in (destination / "src").iterdir()) == ["extra.py", "main.py"]


@pytest.mark.parametrize("field,value", [
    ("class_mapping_version", "2"), ("confidence_threshold", True),
    ("confidence_threshold", float("inf")), ("external_context", "real"),
    ("scope", []), ("requirements", []),
])
def test_unsupported_or_ambiguous_policy_is_rejected(tmp_path, field, value):
    root, manifest = _project(tmp_path)
    manifest[field] = value
    _manifest(root, manifest)
    with pytest.raises(ApprovalError):
        _approve(tmp_path, root)


@pytest.mark.parametrize("change", ["quote", "section", "duplicate_section", "source_path", "fake_heading"])
def test_origins_require_real_unique_prd_section_and_exact_quote(tmp_path, change):
    root, manifest = _project(tmp_path)
    if change == "quote":
        manifest["requirements"][0]["source"]["quote"] = "The result should be safe."
    elif change == "section":
        manifest["requirements"][0]["source"]["section"] = "Optional"
    elif change == "duplicate_section":
        with (root / "PRD.md").open("a") as stream:
            stream.write("\n## Safety\nThe result must be safe.\n")
    elif change == "source_path":
        manifest["requirements"][0]["source"]["path"] = "specs/safety.txt"
    else:
        (root / "PRD.md").write_text("# Product\n```markdown\n## Safety\nThe result must be safe.\n```\n## Optional\nThe extra result must be safe.\n")
    _manifest(root, manifest)
    with pytest.raises(ApprovalError):
        _approve(tmp_path, root)


@pytest.mark.parametrize("change", ["tests", "proofs", "dependency", "cycle", "derived", "duplicate_id", "case_id", "wiki_collision"])
def test_broken_links_and_missing_verification_rejected(tmp_path, change):
    root, manifest = _project(tmp_path)
    first, second = manifest["requirements"]
    if change in {"tests", "proofs"}:
        first[change] = []
    elif change == "dependency":
        first["depends_on"] = ["missing"]
    elif change == "cycle":
        first["depends_on"] = ["R2"]
    elif change == "derived":
        first["status"] = "derived"
    elif change == "duplicate_id":
        second["id"] = "R1"
    elif change == "case_id":
        second["id"] = "r1"
    elif change == "wiki_collision":
        second["id"] = "INDEX"
    _manifest(root, manifest)
    with pytest.raises(ApprovalError):
        _approve(tmp_path, root)


def test_derived_optional_obligation_cannot_unlock_verification(tmp_path):
    root, manifest = _project(tmp_path)
    manifest["requirements"][1].update(status="derived", tests=[], proofs=[])
    _manifest(root, manifest)
    approved = _approve(tmp_path, root)
    assert approved.lint() == []
    (root / "src/extra.py").write_text("EXTRA = False\n")
    assert {item["code"] for item in approved.lint()} == {"unapproved_obligation", "missing_verification"}


@pytest.mark.parametrize("path", ["../outside.py", "/outside.py", "C:/outside.py", "src\\outside.py", "src/../outside.py", "src/NUL.py", "src/bad. /file.py"])
def test_nonportable_and_escaping_paths_are_rejected(tmp_path, path):
    root, manifest = _project(tmp_path)
    manifest["tests"]["T1"]["paths"] = [path]
    _manifest(root, manifest)
    with pytest.raises(ApprovalError):
        _approve(tmp_path, root)


def test_approval_must_be_disjoint_from_project(tmp_path):
    root, _ = _project(tmp_path)
    for directory in (root / "approval", root.parent):
        with pytest.raises(ApprovalError):
            approve_project(root, Path("requirements.json"), directory)


def test_symlink_source_and_symlink_approval_parent_are_rejected(tmp_path):
    root, _ = _project(tmp_path)
    target = tmp_path / "outside.py"
    target.write_text("outside\n")
    (root / "src/link.py").symlink_to(target)
    with pytest.raises(ApprovalError, match="links"):
        _approve(tmp_path, root)
    (root / "src/link.py").unlink()
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ApprovalError, match="symlinks"):
        approve_project(root, Path("requirements.json"), link / "approval")


def test_symlink_introduced_after_approval_fails_closed(tmp_path):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    (root / "src/main.py").unlink()
    (root / "src/main.py").symlink_to(root / "src/extra.py")
    assert {item["code"] for item in approved.lint()} == {"approval_integrity"}
    with pytest.raises(ApprovalError):
        approved.versions()


def test_corrupted_captured_bytes_are_rejected_even_if_manifest_hash_is_unchanged(tmp_path):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    digest = hashlib.sha256((root / "PRD.md").read_bytes()).hexdigest()
    blob = approved.approval_dir / "blobs" / digest
    blob.chmod(0o644)
    blob.write_bytes(b"corruption")
    with pytest.raises(ApprovalError, match="artifact integrity"):
        ApprovedProject.load(approved.approval_dir)
    assert {item["code"] for item in approved.lint()} == {"approval_integrity"}


def test_manifest_metadata_cannot_override_captured_manifest(tmp_path):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    file = approved.approval_dir / "approval.json"
    record = json.loads(file.read_text())
    record["manifest"]["requirements"][0]["text"] = "Invented requirement"
    data = json.dumps(record).encode()
    file.chmod(0o644)
    file.write_bytes(data)
    checksum = approved.approval_dir / "approval.sha256"
    checksum.chmod(0o644)
    checksum.write_text(hashlib.sha256(data).hexdigest())
    with pytest.raises(ApprovalError, match="Captured manifest"):
        ApprovedProject.load(approved.approval_dir)


def test_wiki_is_derived_with_resolving_dependencies_and_does_not_mutate_sources(tmp_path):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    before = approved.versions()
    prd = (root / "PRD.md").read_bytes()
    output = tmp_path / "wiki"
    paths = approved.render_wiki(output)
    assert {path.name for path in paths} == {"index.md", "R1.md", "R2.md"}
    assert "[R1](R1.md)" in (output / "R2.md").read_text()
    assert "> The result must be safe." in (output / "R1.md").read_text()
    assert approved.approval_digest in (output / "index.md").read_text()
    assert (root / "PRD.md").read_bytes() == prd
    assert approved.versions() == before
    with pytest.raises(ApprovalError, match="overwrite"):
        approved.render_wiki(output)
    with pytest.raises(ApprovalError):
        approved.render_wiki(approved.approval_dir)


def test_wiki_cannot_overwrite_prd_or_become_tracked_code(tmp_path):
    root, manifest = _project(tmp_path)
    manifest["scope"] = ["src/**"]
    _manifest(root, manifest)
    approved = _approve(tmp_path, root)
    with pytest.raises(ApprovalError, match="outside approved"):
        approved.render_wiki(root / "src/wiki")
    assert not (root / "src/wiki").exists()


def test_duplicate_json_keys_cannot_hide_an_owner_requirement(tmp_path):
    root, _ = _project(tmp_path)
    manifest = root / "requirements.json"
    text = manifest.read_text()
    manifest.write_text(text.replace('"schema_version": 1', '"schema_version": 2, "schema_version": 1'))
    with pytest.raises(ApprovalError, match="Duplicate JSON key"):
        _approve(tmp_path, root)


def test_transitive_required_dependency_cannot_be_unapproved(tmp_path):
    root, manifest = _project(tmp_path)
    third = copy.deepcopy(manifest["requirements"][1])
    third.update(id="R3", status="derived", depends_on=[])
    manifest["requirements"].append(third)
    manifest["requirements"][0]["depends_on"] = ["R2"]
    manifest["requirements"][1]["depends_on"] = ["R3"]
    _manifest(root, manifest)
    with pytest.raises(ApprovalError, match="Required dependency"):
        _approve(tmp_path, root)


def test_explicit_requirement_code_cannot_escape_broad_scope_tracking(tmp_path):
    root, manifest = _project(tmp_path)
    manifest["requirements"][1]["code"] = ["other/*.py"]
    _manifest(root, manifest)
    approved = _approve(tmp_path, root)
    before = approved.versions()["code"]
    (root / "other").mkdir()
    (root / "other/new.py").write_text("EXTRA = True\n")
    assert approved.changed_paths() == ["other/new.py"]
    assert approved.versions()["code"] != before
    assert [item["id"] for item in approved.obligations([])] == ["R1", "R2"]


def test_wiki_preflights_all_destinations_before_writing(tmp_path):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    output = tmp_path / "wiki"
    output.mkdir()
    (output / "index.md").write_text("Owner-authored content\n")
    with pytest.raises(ApprovalError, match="overwrite"):
        approved.render_wiki(output)
    assert not (output / "R1.md").exists()
    assert (output / "index.md").read_text() == "Owner-authored content\n"


def test_deleted_proof_reference_is_linted_before_wiki_render(tmp_path):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    (root / "proofs/Safety.lean").unlink()
    assert any(error["code"] == "missing_artifact" and "proofs/Safety.lean" in error["message"]
               for error in approved.lint())
    with pytest.raises(ApprovalError):
        approved.render_wiki(tmp_path / "wiki")


def test_explicit_conflicts_block_only_when_both_obligations_are_active(tmp_path):
    root, manifest = _project(tmp_path)
    manifest["requirements"][0]["conflicts_with"] = ["R2"]
    _manifest(root, manifest)
    approved = _approve(tmp_path, root)
    assert approved.lint() == []
    (root / "src/extra.py").write_text("EXTRA = False\n")
    assert any(error["code"] == "conflicting_requirements" for error in approved.lint())
    with pytest.raises(ApprovalError):
        approved.render_wiki(tmp_path / "wiki")
    manifest["requirements"][1]["required"] = True
    _manifest(root, manifest)
    with pytest.raises(ApprovalError, match="Conflicting mandatory"):
        approve_project(root, Path("requirements.json"), tmp_path / "reapproval")


def test_classifier_change_context_contains_actual_diff_and_refuses_truncation(tmp_path):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    (root / "src/main.py").write_text("SAFE = False\n")
    change = approved.change_context()[0]
    assert change["path"] == "src/main.py"
    assert "-SAFE = True" in change["diff"] and "+SAFE = False" in change["diff"]
    assert change["before_hash"] != change["after_hash"]
    with pytest.raises(ApprovalError, match="exceeds"):
        approved.change_context(max_bytes=1)


@pytest.mark.parametrize("change", ["definition", "proof", "new_import"])
def test_all_lean_sources_require_owner_reapproval(tmp_path, change):
    root, _ = _project(tmp_path)
    definition = root / "proofs/Definitions.lean"
    definition.write_text("def Safe : Prop := True\n")
    proof = root / "proofs/Safety.lean"
    proof.write_text("import Definitions\ntheorem safety : Safe := True.intro\n")
    approved = _approve(tmp_path, root)
    before = approved.versions()["spec"]
    if change == "definition":
        definition.write_text("def Safe : Prop := False\n")
    elif change == "proof":
        proof.write_text("import Definitions\ntheorem safety : Safe := by trivial\n")
    else:
        (root / "proofs/Extra.lean").write_text("axiom unsafe : False\n")
    assert approved.versions()["spec"] != before
    assert any(error["code"] == "approval_invalidated" for error in approved.lint())
    with pytest.raises(ApprovalError, match="fresh owner approval"):
        approved.materialize(tmp_path / "snapshot")


def test_approved_hash_remains_captured_when_live_source_changes(tmp_path):
    root, _ = _project(tmp_path)
    original = (root / "src/main.py").read_bytes()
    approved = _approve(tmp_path, root)
    (root / "src/main.py").write_text("SAFE = False\n")
    assert approved.approved_hash("src/main.py") == hashlib.sha256(original).hexdigest()
    assert approved.approved_hash("src/main.py") != hashlib.sha256((root / "src/main.py").read_bytes()).hexdigest()
    for path in ("../src/main.py", "src/unknown.py"):
        with pytest.raises(ApprovalError):
            approved.approved_hash(path)
    blob = approved.approval_dir / "blobs" / hashlib.sha256(original).hexdigest()
    blob.chmod(0o600)
    blob.write_bytes(b"replaced")
    with pytest.raises(ApprovalError, match="artifact integrity"):
        approved.approved_hash("src/main.py")


def test_snapshot_uses_current_code_captured_artifacts_and_excludes_untracked_harness(tmp_path):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    (root / "src/main.py").write_text("SAFE = False\n")
    (root / "src/new.py").write_text("NEW = True\n")
    (root / "src/extra.py").unlink()
    injections = {
        "conftest.py": "raise RuntimeError('unapproved root hook')\n",
        "tests/conftest.py": "raise RuntimeError('unapproved test hook')\n",
        "sitecustomize.py": "raise RuntimeError('unapproved startup hook')\n",
        "pytest.ini": "[pytest]\naddopts = --collect-only\n",
        "tests/test_unapproved.py": "def test_unapproved(): assert False\n",
        "src/__pycache__/main.pyc": "unapproved bytecode",
    }
    for relative, data in injections.items():
        file = root / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(data)
    destination = tmp_path / "snapshot"
    hashes = approved.materialize(destination)
    expected = {
        "requirements.json", "PRD.md", "specs/safety.txt", "tests/test_safety.py",
        "proofs/Safety.lean", "config/policy.json", "src/main.py", "src/new.py",
    }
    assert hashes.keys() == expected
    assert {file.relative_to(destination).as_posix() for file in destination.rglob("*")
            if file.is_file()} == expected
    for relative, digest in hashes.items():
        assert digest == hashlib.sha256((destination / relative).read_bytes()).hexdigest()
        assert (destination / relative).read_bytes() == (root / relative).read_bytes()
    assert hashes["src/main.py"] != approved.approved_hash("src/main.py")
    assert hashes["proofs/Safety.lean"] == approved.approved_hash("proofs/Safety.lean")
    (root / "src/main.py").write_text("CHANGED_AFTER_SNAPSHOT = True\n")
    assert (destination / "src/main.py").read_text() == "SAFE = False\n"


@pytest.mark.parametrize("name", ["conftest.py", "sitecustomize.py", "pytest.ini", "pyproject.toml"])
def test_scope_cannot_make_test_configuration_mutable(tmp_path, name):
    root, manifest = _project(tmp_path)
    manifest["scope"] = ["**"]
    _manifest(root, manifest)
    config = root / name
    config.write_text("# owner configuration\n")
    with pytest.raises(ApprovalError, match="explicit approved test protection"):
        _approve(tmp_path, root)
    manifest["tests"]["T1"]["paths"].append(name)
    _manifest(root, manifest)
    approved = _approve(tmp_path, root)
    destination = tmp_path / "snapshot"
    assert approved.materialize(destination)[name] == approved.approved_hash(name)
    config.write_text("# changed test environment\n")
    with pytest.raises(ApprovalError, match="fresh owner approval"):
        approved.materialize(tmp_path / "second-snapshot")


@pytest.mark.parametrize("location", ["project", "approval", "ancestor", "existing", "missing_parent", "symlink_parent", "symlink_loop"])
def test_snapshot_requires_fresh_disjoint_real_destination(tmp_path, location):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_text("untouched\n")
    link = tmp_path / "linked"
    link.symlink_to(outside, target_is_directory=True)
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    destination = {
        "project": root / "snapshot", "approval": approved.approval_dir / "snapshot",
        "ancestor": tmp_path, "existing": outside, "missing_parent": tmp_path / "missing/snapshot",
        "symlink_parent": link / "snapshot",
        "symlink_loop": loop,
    }[location]
    with pytest.raises(ApprovalError):
        approved.materialize(destination)
    assert (outside / "sentinel").read_text() == "untouched\n"
    assert not (outside / "snapshot").exists()


def test_snapshot_rejects_live_source_symlinks(tmp_path):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    (root / "src/main.py").unlink()
    (root / "src/main.py").symlink_to(root / "src/extra.py")
    with pytest.raises(ApprovalError, match="links"):
        approved.materialize(tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


@pytest.mark.parametrize("same_bytes", [False, True])
def test_snapshot_rejects_source_changes_and_same_content_replacement_during_copy(tmp_path, monkeypatch, same_bytes):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    source = root / "src/main.py"
    original_fsync = requirements_module.os.fsync
    changed = False

    def replace_during_copy(fd):
        nonlocal changed
        original_fsync(fd)
        if not changed:
            changed = True
            replacement = tmp_path / "replacement"
            replacement.write_bytes(source.read_bytes() if same_bytes else b"SAFE = False\n")
            replacement.replace(source)

    monkeypatch.setattr(requirements_module.os, "fsync", replace_during_copy)
    with pytest.raises(ApprovalError, match="Workspace changed during snapshot"):
        approved.materialize(tmp_path / "snapshot")
    assert source.read_bytes() == (b"SAFE = True\n" if same_bytes else b"SAFE = False\n")


def test_snapshot_reports_copy_failure_without_returning_success(tmp_path, monkeypatch):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)

    def no_space(fd):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(requirements_module.os, "fsync", no_space)
    with pytest.raises(ApprovalError, match="Cannot materialize snapshot"):
        approved.materialize(tmp_path / "snapshot")
    assert (root / "src/main.py").read_text() == "SAFE = True\n"


@pytest.mark.parametrize("attack", ["symlink", "extra_file"])
def test_snapshot_rejects_destination_replacement_or_extra_harness_during_copy(tmp_path, monkeypatch, attack):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    destination = tmp_path / "snapshot"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_fsync = requirements_module.os.fsync
    changed = False

    def inject_during_copy(fd):
        nonlocal changed
        original_fsync(fd)
        if not changed:
            changed = True
            if attack == "symlink":
                destination.rename(tmp_path / "original-snapshot")
                destination.symlink_to(outside, target_is_directory=True)
            else:
                (destination / "conftest.py").write_text("raise RuntimeError('injected')\n")

    monkeypatch.setattr(requirements_module.os, "fsync", inject_during_copy)
    with pytest.raises(ApprovalError, match="Snapshot (destination|file set) changed"):
        approved.materialize(destination)
    assert list(outside.iterdir()) == []


def test_snapshot_rejects_fifo_source_without_opening_it(tmp_path):
    root, _ = _project(tmp_path)
    approved = _approve(tmp_path, root)
    source = root / "src/main.py"
    source.unlink()
    requirements_module.os.mkfifo(source)
    with pytest.raises(ApprovalError, match="not regular"):
        approved.materialize(tmp_path / "snapshot")
