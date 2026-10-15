"""Portable contract bundles: the format, strict validation, discovery as candidates, comparison and export safety.

Everything is synthetic. The engine-level behavior (adopt, conflicts, the two-device round trip) is in
``tests/test_portable_contracts.py``; this file covers ``self_directing_mcp.contract_bundles`` on its own: reading a file
is reading untrusted data, so every limit and refusal named in the README ("Sharing contracts across devices") is pinned
here.
"""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from self_directing_mcp import contract_bundles as cb
from self_directing_mcp.schemas import ContractOrigin, ContractRule

POSIX = hasattr(os, "getuid")
posix_only = pytest.mark.skipif(not POSIX, reason="ownership and mode checks are POSIX only")
RM = {"id": "no-rm", "type": "must_not", "scope": "tool_call", "regex": r"rm\s+-rf", "description": "Never delete."}
TESTS = {"id": "tests-ran", "type": "must", "scope": "tool_call", "regex": "pytest", "roles": ["assistant"]}


def doc(*contracts, label="team contracts", **extra):
    return {"schema": cb.BUNDLE_SCHEMA, "label": label, "contracts": list(contracts), **extra}


def raw(document) -> bytes:
    return json.dumps(document).encode("utf-8")


def parse(document, kind="project", label=".tsukkomi/contracts.json"):
    return cb.parse_bundle(raw(document) if not isinstance(document, bytes) else document, kind=kind, path_label=label)


def problems(document) -> str:
    with pytest.raises(cb.BundleError) as caught:
        parse(document)
    return " | ".join(caught.value.problems)


def put(path: Path, document, mode=0o644) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(document if isinstance(document, bytes) else raw(document))
    path.chmod(mode)
    return path


def strings(value):
    """Every string in a nested structure: used to prove that no path is ever returned."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from strings(key)
            yield from strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from strings(item)
    elif hasattr(value, "__dict__"):
        yield from strings(vars(value))


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = home / "work" / "app"
    project.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return {"tmp": tmp_path, "home": home.resolve(), "project": project.resolve()}


def found(world, directory=None, **kwargs):
    return cb.discover(project_dir=directory if directory is not None else world["project"], home=world["home"], **kwargs)


# ---------------------------------------------------------------------------------------------------- the format

def test_a_valid_bundle_is_read_as_data_with_its_hash_as_identity():
    data = raw(doc(RM, TESTS, label="  my team  "))
    bundle = cb.parse_bundle(data, kind="project", path_label=".tsukkomi/contracts.json")
    assert bundle.sha256 == cb.digest(data) and bundle.bundle_id == bundle.sha256[:12] and len(bundle.bundle_id) == 12
    assert bundle.label == "my team" and [c.rule.id for c in bundle.contracts] == ["no-rm", "tests-ran"]
    assert all(c.rule.revision == 1 and c.rule.source_event_id is None and c.rule.session_id is None
               for c in bundle.contracts)


def test_the_same_bytes_have_the_same_identity_and_one_changed_byte_another():
    first, again = parse(doc(RM)), parse(doc(RM))
    changed = parse(doc({**RM, "description": "Never delete!"}))
    assert first.sha256 == again.sha256 != changed.sha256


def test_a_leading_byte_order_mark_is_accepted_and_part_of_the_hash():
    data = b"\xef\xbb\xbf" + raw(doc(RM))
    assert cb.parse_bundle(data, kind="personal", path_label="personal#1/x.json").sha256 == cb.digest(data)


@pytest.mark.parametrize("name", sorted(cb.DEVICE_BOUND_FIELDS))
def test_a_device_bound_field_is_rejected_and_named(name):
    value = {"source_event_id": "claude:other-device:1", "applies_from_event_id": "x", "session_id": "s",
             "revision": 3, "origin": {"bundle_sha256": "0" * 64}}[name]
    message = problems(doc({**RM, name: value}))
    assert f"device-bound field '{name}'" in message and "no-rm" in message


def test_unknown_fields_are_rejected_at_both_levels():
    assert "unknown field 'surprise'" in problems(doc({**RM, "surprise": 1}))
    assert "unknown top-level field 'includes'" in problems(doc(RM, includes=["other.json"]))


@pytest.mark.parametrize("document,expected", [
    ({"schema": "tsukkomi.contracts-bundle/2", "label": "x", "contracts": []}, "schema must be"),
    ({"label": "x", "contracts": []}, "schema must be"),
    ({"schema": cb.BUNDLE_SCHEMA, "contracts": []}, "label must be"),
    ({"schema": cb.BUNDLE_SCHEMA, "label": "  ", "contracts": []}, "label must be"),
    ({"schema": cb.BUNDLE_SCHEMA, "label": "x"}, "contracts must be an array"),
    ({"schema": cb.BUNDLE_SCHEMA, "label": "x", "contracts": {"a": 1}}, "contracts must be an array"),
    ({"schema": cb.BUNDLE_SCHEMA, "label": "x" * 81, "contracts": []}, "label must be plain text"),
    ({"schema": cb.BUNDLE_SCHEMA, "label": "a‮b", "contracts": []}, "label must be plain text"),
    ({"schema": cb.BUNDLE_SCHEMA, "label": "a\nb", "contracts": []}, "label must be plain text"),
])
def test_the_envelope_is_strict(document, expected):
    assert expected in problems(document)


def test_the_top_level_must_be_an_object():
    with pytest.raises(cb.BundleError, match="invalid_bundle"):
        parse(b"[]")


@pytest.mark.parametrize("data,code", [
    (b"\xff\xfe not utf8", "not_utf8"), (b"{not json", "not_json"), (b"", "not_json"), (b'{"a": NaN}', "not_json"),
    (b'{"schema": "x", "schema": "y"}', "not_json"), (b"[" * 100000, "not_json"),
], ids=["not-utf8", "broken-json", "empty", "nan-constant", "duplicate-key", "deeply-nested"])   # explicit ids: a 100,000 character id overflows Windows' 32,767 character environment variable limit (PYTEST_CURRENT_TEST)
def test_bytes_that_are_not_utf8_json_are_refused(data, code):
    with pytest.raises(cb.BundleError) as caught:
        cb.parse_bundle(data, kind="project", path_label="x")
    assert caught.value.code == code


def test_a_bundle_over_one_mebibyte_is_refused_without_parsing():
    with pytest.raises(cb.BundleError) as caught:
        cb.parse_bundle(b" " * (cb.MAX_BUNDLE_BYTES + 1), kind="project", path_label="x")
    assert caught.value.code == "too_large"
    cb.parse_bundle(raw(doc()) + b" " * (cb.MAX_BUNDLE_BYTES - len(raw(doc()))), kind="project", path_label="x")


def test_a_bundle_holds_at_most_two_hundred_contracts():
    many = [{"id": f"c{n}", "type": "must_not", "regex": f"x{n}"} for n in range(cb.MAX_BUNDLE_CONTRACTS)]
    assert len(parse(doc(*many)).contracts) == 200
    with pytest.raises(cb.BundleError) as caught:
        parse(doc(*many, {"id": "one-more", "type": "must_not", "regex": "y"}))
    assert caught.value.code == "too_many_contracts"


@pytest.mark.parametrize("contract,expected", [
    ({"type": "must_not"}, "an id is required"),
    ({"id": "", "type": "must_not"}, "id must not be empty"),
    ({"id": "x" * 201, "type": "must_not"}, "id is longer than 200"),
    ({"id": "bad‮id", "type": "must_not"}, "id contains control"),
    ({"id": "line\nbreak", "type": "must_not"}, "id contains control"),
    ({"id": 5, "type": "must_not"}, "id must be a string"),
    ({"id": "a"}, "type"),
    ({"id": "a", "type": "forbidden"}, "type"),
    ({"id": "a", "type": "must_not", "enabled": "true"}, "enabled"),
    ({"id": "a", "type": "must_not", "roles": []}, "roles"),
    ({"id": "a", "type": "must_not", "roles": ["r"] * 9}, "at most 8 roles"),
    ({"id": "a", "type": "must_not", "roles": ["r" * 41]}, "roles is longer than 40"),
    ({"id": "a", "type": "must_not", "regex": "x" * 1001}, "regex is longer than 1000"),
    ({"id": "a", "type": "must", "regex": "x", "before_regex": "y" * 1001, "scope": "tool_call"},
     "before_regex is longer than 1000"),
    ({"id": "a", "type": "must", "regex": "x", "when": {"tool_call_regex": "y" * 1001}}, "tool_call_regex is longer"),
    ({"id": "a", "type": "must_not", "description": "d" * 2001}, "description is longer than 2000"),
    ({"id": "a", "type": "must_not", "description": "bidi ‮ text"}, "description contains control"),
    ({"id": "a", "type": "must_not", "regex": "("}, "invalid regex"),
    ({"id": "a", "type": "must_not", "regex": "ok", "exception_regex": "(?P<"}, "invalid regex"),
    ({"id": "a", "type": "must", "regex": "x", "when": {"tool_call_regex": "["}}, "invalid regex"),
    ({"id": "a", "type": "must_not", "risk": "high"}, "high-risk must_not needs a regex"),
    ({"id": "a", "type": "must_not", "before_regex": "x", "regex": "y"}, "before_regex requires"),
    ({"id": "a", "type": "must", "exception_regex": "x"}, "exception_regex applies only"),
    ({"id": "a", "type": "must_not", "claimed_origin_event_id": "e" * 401}, "claimed_origin_event_id is longer"),
    ({"id": "a", "type": "must_not", "source_quote": "q" * 501}, "source_quote is longer than 500"),
    ("not an object", "a contract must be an object"),
])
def test_every_contract_is_validated_as_strictly_as_upsert(contract, expected):
    assert expected in problems(doc(contract))


def test_the_regex_gate_is_the_one_upsert_uses():
    """One function validates a contract before it is stored, whichever way it arrives (the bundle's rules are the engine's)."""
    from self_directing_mcp.audit import contracts as audit_contracts
    assert cb.check_rule_patterns is audit_contracts.check_rule_patterns


def test_a_duplicate_id_in_one_bundle_is_an_error():
    assert "duplicate id" in problems(doc(RM, RM))


def test_the_problems_are_capped_and_never_repeat_the_text_of_a_pattern():
    marker = "SECRET_MARKER_PATTERN("
    many = [{"id": f"c{n}", "type": "must_not", "regex": marker} for n in range(25)]
    message = problems(doc(*many))
    assert message.count("invalid regex") == cb.MAX_PROBLEMS and "and 15 more" in message
    assert marker not in message


def test_a_quote_is_accepted_as_plain_text_but_never_becomes_the_stored_source_quote():
    bundle = parse(doc({**RM, "source_quote": "the author's own words", "claimed_origin_event_id": "claude:x:1"}))
    (contract,) = bundle.contracts
    assert contract.rule.source_quote is None
    assert contract.claimed_quote == "the author's own words" and contract.claimed_origin_event_id == "claude:x:1"


def test_content_ignores_everything_that_belongs_to_a_device_and_the_order_of_roles():
    local = ContractRule(id="no-rm", type="must_not", scope="tool_call", regex=r"rm\s+-rf", description="Never delete.",
                         source_event_id="claude:a:1", applies_from_event_id="claude:a:0", revision=7,
                         source_quote="never rm -rf", roles=["assistant"])
    assert cb.content_hash(cb.content_of(local)) == parse(doc(RM)).contracts[0].content_hash
    both = ContractRule(id="x", type="must_not", roles=["b", "a", "a"])
    assert cb.content_of(both)["roles"] == ["a", "b"]
    scoped = local.model_copy(update={"session_id": "s1"})
    assert cb.content_hash(cb.content_of(scoped)) != cb.content_hash(cb.content_of(local))


# --------------------------------------------------------------------------------------------- compare and statuses

def stored(rule: dict, origin=None, **extra) -> ContractRule:
    return ContractRule(**{**rule, "source_quote": "mine", "revision": 2, **extra}, origin=origin)


def origin_of(bundle: cb.Bundle, sha=None) -> ContractOrigin:
    return ContractOrigin(bundle_sha256=sha or bundle.sha256, bundle_label=bundle.label, origin_kind=bundle.kind,
                          bundle_path=bundle.path_label, adopted_by_event_id="claude:b:1")


def statuses(bundle, local):
    return {entry.contract.rule.id: entry.status for entry in cb.compare(bundle, local)}


def test_a_contract_is_new_same_adopted_differs_or_changed_since_adoption():
    bundle = parse(doc(RM, TESTS, {"id": "third", "type": "must_not", "regex": "z"}))
    older = parse(doc({**RM, "regex": "rm -r"}))                # the file as it was when no-rm was adopted
    elsewhere = parse(doc(TESTS), kind="personal", label="personal#1/other.json")
    local = {"no-rm": stored(RM, origin_of(bundle)),            # identical content, adopted
             "tests-ran": stored({**TESTS, "regex": "pytest -q"}, origin_of(elsewhere))}    # adopted from another file
    assert statuses(bundle, local) == {"no-rm": "adopted", "tests-ran": "differs", "third": "new"}
    same_content_written_here = {"no-rm": stored(RM)}
    assert statuses(bundle, same_content_written_here)["no-rm"] == "same"
    adopted_old = {"no-rm": stored(older.contracts[0].rule.model_dump(), origin_of(older))}
    entries = cb.compare(bundle, adopted_old)
    assert entries[0].status == "changed_since_adoption"
    assert [d["field"] for d in entries[0].differences] == ["regex"]


def test_a_stored_contract_edited_after_adoption_from_this_very_file_is_differs_not_changed():
    bundle = parse(doc(RM))
    edited = stored({**RM, "regex": "rm -rf --no-preserve-root"}, origin_of(bundle))
    (entry,) = cb.compare(bundle, {"no-rm": edited})
    assert entry.status == "differs" and entry.local_modified is True


def test_a_contract_adopted_from_another_file_that_differs_is_a_conflict_not_a_change():
    other = parse(doc({**RM, "regex": "other"}), label="docs/.tsukkomi/contracts.json")
    (entry,) = cb.compare(parse(doc(RM)), {"no-rm": stored({**RM, "regex": "other"}, origin_of(other))})
    assert entry.status == "differs"


def test_a_local_contract_limited_to_a_session_never_equals_a_global_candidate():
    (entry,) = cb.compare(parse(doc(RM)), {"no-rm": stored(RM, session_id="s1")})
    assert entry.status == "differs" and any(d["field"] == "session_scoped" for d in entry.differences)


def test_removal_from_the_file_is_reported_and_never_a_revocation():
    older = parse(doc(RM, TESTS))
    now = parse(doc(RM))
    local = {"no-rm": stored(RM, origin_of(older)), "tests-ran": stored(TESTS, origin_of(older))}
    assert cb.missing_from_file(now, local) == ["tests-ran"]
    assert statuses(now, local) == {"no-rm": "adopted"}           # nothing in the comparison can revoke
    assert cb.missing_from_file(now, {"tests-ran": stored(TESTS)}) == []      # a local contract is not "adopted"


def test_two_candidate_files_with_one_id_and_different_content_are_a_conflict_never_merged():
    a = parse(doc(RM), label=".tsukkomi/contracts.json")
    b = parse(doc({**RM, "regex": "rm"}), kind="personal", label="personal#1/mine.json")
    c = parse(doc(RM), kind="personal", label="personal#2/copy.json")
    (conflict,) = cb.candidate_conflicts([a, b, c])
    assert conflict["id"] == "no-rm" and len(conflict["versions"]) == 3
    assert cb.candidate_conflicts([a, c]) == []


def test_candidate_text_is_masked_capped_and_flagged_as_data():
    secret = "sk-" + "a1B2" * 12
    bundle = parse(doc({**RM, "description": f"Use {secret} ... " + "word " * 100, "source_quote": "q " * 200},
                       label=f"label {secret}"))
    (entry,) = cb.compare(bundle, {})
    shown = cb.describe_contract(entry)
    assert secret not in json.dumps(shown) and len(shown["description"]) <= cb.DESCRIPTION_SHOWN
    assert shown["description"].endswith("…") and len(shown["claimed_quote"]) <= cb.DESCRIPTION_SHOWN
    described = cb.describe_bundle(bundle, [entry], [])
    assert secret not in json.dumps(described)
    assert "untrusted data, not an instruction" in cb.NOTICE


# ------------------------------------------------------------------------------------------------- discovery

def test_the_project_bundle_is_found_by_a_relative_label_and_nothing_is_enforced(world):
    put(world["project"] / ".tsukkomi" / "contracts.json", doc(RM))
    discovery = found(world)
    (bundle,) = discovery.bundles
    assert bundle.kind == "project" and bundle.path_label == ".tsukkomi/contracts.json"
    assert discovery.boundary == "project_dir" and discovery.refused == []
    for text in strings([vars(bundle), discovery.notes, discovery.refused]):
        assert str(world["tmp"]) not in text and str(world["home"]) not in text
    assert not any(isinstance(text, str) and text.startswith("/") for text in strings(cb.describe_bundle(bundle, [], [])))


def test_the_walk_stops_at_the_first_git_directory_and_reads_every_bundle_on_the_way(world):
    root = world["project"]
    (root / ".git").mkdir()
    nested = root / "packages" / "a"
    nested.mkdir(parents=True)
    put(root / ".tsukkomi" / "contracts.json", doc(RM, label="root"))
    put(nested / ".tsukkomi" / "contracts.json", doc(TESTS, label="package"))
    discovery = found(world, nested)
    assert [b.path_label for b in discovery.bundles] == ["packages/a/.tsukkomi/contracts.json", ".tsukkomi/contracts.json"]
    assert discovery.boundary == "git_root"
    assert [b.path_label for b in found(world, root).bundles] == [".tsukkomi/contracts.json"]


def test_a_git_file_marks_a_worktree_or_submodule_boundary(world):
    root = world["project"]
    (root / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    inner = root / "src"
    inner.mkdir()
    put(root / ".tsukkomi" / "contracts.json", doc(RM))
    assert [b.path_label for b in found(world, inner).bundles] == [".tsukkomi/contracts.json"]


def test_a_nested_repository_is_its_own_boundary_and_its_parent_is_not_read(world):
    outer = world["project"]
    (outer / ".git").mkdir()
    inner = outer / "vendor" / "lib"
    (inner / ".git").mkdir(parents=True)
    deep = inner / "pkg"
    deep.mkdir()
    put(outer / ".tsukkomi" / "contracts.json", doc(RM, label="outer"))
    put(inner / ".tsukkomi" / "contracts.json", doc(TESTS, label="inner"))
    discovery = found(world, deep)
    assert [b.label for b in discovery.bundles] == ["inner"] and discovery.boundary == "git_root"


def test_without_a_git_directory_only_the_project_directory_itself_is_read(world):
    parent = world["project"].parent
    put(parent / ".tsukkomi" / "contracts.json", doc(RM, label="parent"))
    child = world["project"] / "sub"
    child.mkdir()
    assert found(world, child).bundles == []
    put(child / ".tsukkomi" / "contracts.json", doc(TESTS, label="child"))
    assert [b.label for b in found(world, child).bundles] == ["child"]


def test_the_walk_never_reaches_the_users_home_so_a_dotfiles_repository_is_not_a_boundary(world):
    (world["home"] / ".git").mkdir()
    put(world["home"] / ".tsukkomi" / "contracts.json", doc(RM, label="home bundle"))
    discovery = found(world)
    assert discovery.bundles == [] and discovery.boundary == "project_dir"
    inside_home = found(world, world["home"])               # asked about the home itself: only the home itself
    assert [b.label for b in inside_home.bundles] == ["home bundle"] and inside_home.boundary == "git_root"


def test_a_project_directory_above_home_or_the_filesystem_root_or_not_absolute_is_not_a_project(world):
    put(world["tmp"] / ".tsukkomi" / "contracts.json", doc(RM))
    for directory, fragment in ((world["tmp"], "above the user's home"), (Path(world["tmp"].anchor), "filesystem root"),
                                ("relative/dir", "absolute"), (world["tmp"] / "missing", "not a directory")):
        discovery = found(world, directory)
        assert discovery.bundles == [] and any(fragment in note for note in discovery.notes), directory


def test_project_discovery_can_be_switched_off_or_has_no_directory(world):
    put(world["project"] / ".tsukkomi" / "contracts.json", doc(RM))
    off = found(world, project_mode="off")
    assert off.bundles == [] and "SELF_DIRECT_PROJECT_CONTRACTS=off" in off.notes[0]
    none = cb.discover(project_dir=None, home=world["home"])
    assert none.bundles == [] and "no project directory" in none.notes[0]


@pytest.mark.parametrize("make,reason", [
    (lambda path: path.write_bytes(b"x" * (cb.MAX_BUNDLE_BYTES + 1)), "too_large"),
    (lambda path: path.write_text("{not json", encoding="utf-8"), "not_json"),
    (lambda path: path.write_bytes(raw(doc(*[{"id": f"c{n}", "type": "must_not", "regex": "x"} for n in range(201)]))),
     "too_many_contracts"),
    (lambda path: path.write_bytes(raw(doc({**RM, "source_event_id": "claude:other:1"}))), "invalid_bundle"),
])
def test_a_bad_project_file_is_listed_as_refused_with_a_reason_and_no_path(world, make, reason):
    path = world["project"] / ".tsukkomi" / "contracts.json"
    path.parent.mkdir(parents=True)
    make(path)
    discovery = found(world)
    assert discovery.bundles == [] and [item["reason"] for item in discovery.refused] == [reason]
    assert discovery.refused[0]["path_label"] == ".tsukkomi/contracts.json"
    for text in strings(discovery.refused):
        assert str(world["tmp"]) not in text


@posix_only
def test_a_symbolic_link_out_of_the_project_is_never_followed(world):
    outside = put(world["tmp"] / "elsewhere" / "contracts.json", doc(RM, label="from outside"))
    folder = world["project"] / ".tsukkomi"
    folder.mkdir()
    (folder / "contracts.json").symlink_to(outside)
    discovery = found(world)
    assert discovery.bundles == [] and discovery.refused[0]["reason"] == "symlink_leaves_project"


@posix_only
def test_a_symbolic_link_directory_out_of_the_project_is_never_followed(world):
    put(world["tmp"] / "elsewhere" / "contracts.json", doc(RM, label="from outside"))
    (world["project"] / ".tsukkomi").symlink_to(world["tmp"] / "elsewhere")
    discovery = found(world)
    assert discovery.bundles == [] and discovery.refused[0]["reason"] == "symlink_leaves_project"


@posix_only
def test_a_symbolic_link_that_stays_inside_the_project_is_read(world):
    real = put(world["project"] / "shared" / "team.json", doc(RM, label="shared"))
    folder = world["project"] / ".tsukkomi"
    folder.mkdir()
    (folder / "contracts.json").symlink_to(real)
    assert [b.label for b in found(world).bundles] == ["shared"]


@posix_only
def test_the_file_is_never_opened_through_a_link_swapped_in_after_the_check(world, monkeypatch):
    """The final path component is opened with O_NOFOLLOW: a link in its place is refused, not read."""
    target = put(world["tmp"] / "elsewhere" / "x.json", doc(RM))
    link = world["tmp"] / "swapped.json"
    link.symlink_to(target)
    with pytest.raises(cb.FileRefused) as caught:
        cb.read_checked(link)
    assert caught.value.code == "unreadable"


@posix_only
def test_a_file_owned_by_another_user_is_refused(world, monkeypatch):
    put(world["project"] / ".tsukkomi" / "contracts.json", doc(RM))
    monkeypatch.setattr(os, "getuid", lambda: os.stat(world["project"]).st_uid + 1)
    discovery = found(world)
    assert discovery.bundles == [] and discovery.refused[0]["reason"] == "not_owned_by_user"


@posix_only
@pytest.mark.parametrize("mode", [0o666, 0o646, 0o602])
def test_a_world_writable_file_is_refused(world, mode):
    put(world["project"] / ".tsukkomi" / "contracts.json", doc(RM), mode=mode)
    discovery = found(world)
    assert discovery.bundles == [] and discovery.refused[0]["reason"] == "world_writable"


@posix_only
@pytest.mark.skipif(POSIX and os.geteuid() == 0, reason="root can read any file")
def test_an_unreadable_file_is_refused_with_an_error_name_not_a_path(world):
    path = put(world["project"] / ".tsukkomi" / "contracts.json", doc(RM), mode=0o000)
    try:
        discovery = found(world)
    finally:
        path.chmod(0o644)
    assert discovery.refused[0]["reason"] == "unreadable" and discovery.refused[0]["detail"] == "EACCES"


@posix_only
def test_a_fifo_or_directory_in_place_of_the_file_is_not_a_regular_file(world):
    folder = world["project"] / ".tsukkomi"
    folder.mkdir()
    os.mkfifo(folder / "contracts.json")
    assert found(world).refused[0]["reason"] == "not_a_regular_file"


def test_a_directory_named_like_the_bundle_is_refused(world):
    (world["project"] / ".tsukkomi" / "contracts.json").mkdir(parents=True)
    assert found(world).refused[0]["reason"] in ("not_a_regular_file", "unreadable")


def test_more_than_sixteen_project_bundles_on_the_way_up_are_not_all_read(world):
    root = world["project"]
    (root / ".git").mkdir()
    directory = root
    for n in range(cb.MAX_PROJECT_BUNDLES + 3):
        put(directory / ".tsukkomi" / "contracts.json", doc({"id": f"c{n}", "type": "must_not", "regex": str(n)},
                                                              label=f"level {n}"))
        directory = directory / f"d{n}"
        directory.mkdir()
    discovery = found(world, directory)
    assert len(discovery.bundles) == cb.MAX_PROJECT_BUNDLES and any("more than 16" in n for n in discovery.notes)


def test_personal_bundles_come_from_the_configured_files_and_are_named_by_label(world):
    mine = put(world["tmp"] / "private" / "mine.json", doc(RM, label="mine"), mode=0o600)
    gone = world["tmp"] / "private" / "gone.json"
    discovery = found(world, personal_paths=[mine, gone, "relative.json"])
    assert [(b.kind, b.path_label, b.label) for b in discovery.bundles] == [("personal", "personal#1/mine.json", "mine")]
    assert [(r["path_label"], r["reason"]) for r in discovery.refused] == [
        ("personal#2/gone.json", "not_found"), ("personal#3/relative.json", "relative_path")]
    for text in strings([vars(b) for b in discovery.bundles] + discovery.refused):
        assert str(world["tmp"]) not in text


@posix_only
def test_a_personal_bundle_gets_the_same_file_checks(world):
    open_to_all = put(world["tmp"] / "open.json", doc(RM), mode=0o666)
    assert found(world, personal_paths=[open_to_all]).refused[0]["reason"] == "world_writable"


@posix_only
def test_a_symbolic_link_to_a_personal_bundle_is_resolved_because_the_owner_chose_that_path(world):
    real = put(world["tmp"] / "dotfiles" / "contracts.json", doc(RM, label="dotfile"), mode=0o600)
    link = world["tmp"] / "link.json"
    link.symlink_to(real)
    assert [b.label for b in found(world, personal_paths=[link]).bundles] == ["dotfile"]


def test_the_same_bytes_in_two_places_are_one_candidate_naming_both(world):
    put(world["project"] / ".tsukkomi" / "contracts.json", doc(RM))
    copy = put(world["tmp"] / "copy.json", doc(RM), mode=0o600)
    (bundle,) = found(world, personal_paths=[copy, copy]).bundles
    assert bundle.kind == "project" and bundle.also_at == ("personal#1/copy.json",)


# ----------------------------------------------------------------------------------------------------- export

def target_for(world, label, personal=(), boundary="project", allowed=True):
    return cb.resolve_export_target(label, boundary=world["project"] if boundary == "project" else None,
                                    personal_paths=personal, project_allowed=allowed)


def test_a_project_export_target_is_a_json_file_inside_the_project(world):
    first = target_for(world, ".tsukkomi/contracts.json")
    assert (first.kind, first.label, first.path) == ("project", ".tsukkomi/contracts.json",
                                                      world["project"] / ".tsukkomi" / "contracts.json")
    inside = target_for(world, str(world["project"] / "docs" / "c.json"))
    assert inside.label == "docs/c.json"


@pytest.mark.parametrize("label", ["../outside.json", ".tsukkomi/../../x.json", "/etc/passwd.json", "/tmp/x.json",
                                   ".tsukkomi/contracts.txt", ".", "", "a\x00.json", "personal#9", "personal#1/other.json"])
def test_every_other_export_path_is_refused(world, label):
    with pytest.raises(ValueError):
        target_for(world, label)


def test_an_export_target_can_be_a_configured_personal_bundle_by_label_or_path(world):
    mine = world["tmp"] / "private" / "mine.json"
    for label in ("personal#1", "personal#1/mine.json", str(mine)):
        target = target_for(world, label, personal=[mine], boundary=None)
        assert (target.kind, target.label, target.path) == ("personal", "personal#1/mine.json", mine)


def test_project_export_is_refused_when_project_bundles_are_off_or_no_project_is_known(world):
    with pytest.raises(ValueError, match="neither a configured personal bundle"):
        target_for(world, ".tsukkomi/contracts.json", allowed=False)
    with pytest.raises(ValueError, match="neither a configured personal bundle"):
        target_for(world, ".tsukkomi/contracts.json", boundary=None)


@posix_only
def test_an_export_path_that_leaves_the_project_through_a_link_is_refused(world):
    (world["project"] / ".tsukkomi").symlink_to(world["tmp"])
    with pytest.raises(ValueError, match="outside the project"):
        target_for(world, ".tsukkomi/contracts.json")


def exported(world, label=".tsukkomi/contracts.json", rules=None, kind=None, **personal):
    target = target_for(world, label, **personal)
    data = cb.serialize_bundle("exported", [ContractRule(**RM)] if rules is None else rules)
    cb.write_atomic(target, data)
    return target, data


@posix_only
@pytest.mark.parametrize("umask", [0o022, 0o077, 0o000])
def test_a_project_file_is_written_0644_and_a_personal_file_0600_whatever_the_umask(world, umask):
    old = os.umask(umask)
    try:
        project, _ = exported(world)
        mine = world["tmp"] / "private" / "mine.json"
        personal, _ = exported(world, "personal#1", personal=[mine], boundary=None)
    finally:
        os.umask(old)
    assert stat.S_IMODE(project.path.stat().st_mode) == 0o644 and stat.S_IMODE(personal.path.stat().st_mode) == 0o600


def test_the_written_file_reads_back_as_the_same_bundle(world):
    target, data = exported(world, rules=[ContractRule(**RM), ContractRule(**{**TESTS, "risk": "high", "enabled": False})])
    assert target.path.read_bytes() == data
    bundle = cb.parse_bundle(data, kind="project", path_label=target.label)
    assert [c.content_hash for c in bundle.contracts] == [
        cb.content_hash(cb.content_of(ContractRule(**RM))),
        cb.content_hash(cb.content_of(ContractRule(**{**TESTS, "risk": "high", "enabled": False})))]
    assert not [name for name in json.loads(data)["contracts"][0] if name in cb.DEVICE_BOUND_FIELDS]


def test_a_failed_write_leaves_the_old_file_and_no_temporary_file(world, monkeypatch):
    target, first = exported(world)
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("simulated")))
    with pytest.raises(OSError):
        cb.write_atomic(target, cb.serialize_bundle("second", [ContractRule(**TESTS)]))
    assert target.path.read_bytes() == first
    assert sorted(p.name for p in target.path.parent.iterdir()) == ["contracts.json"]


def test_an_existing_file_that_is_not_a_bundle_is_never_overwritten(world):
    target = target_for(world, ".tsukkomi/contracts.json")
    target.path.parent.mkdir()
    for content in (b"important notes\n", b'{"schema": "other", "contracts": []}', raw(doc({**RM, "source_event_id": "x"}))):
        target.path.write_bytes(content)
        with pytest.raises(ValueError, match="not a valid contracts bundle"):
            cb.write_atomic(target, cb.serialize_bundle("x", []))
        assert target.path.read_bytes() == content


@posix_only
def test_the_export_does_not_write_through_a_symbolic_link(world):
    target = target_for(world, ".tsukkomi/contracts.json")
    victim = put(world["tmp"] / "victim.json", doc(RM))
    target.path.parent.mkdir()
    target.path.symlink_to(victim)
    before = victim.read_bytes()
    with pytest.raises(ValueError, match="symbolic link"):
        cb.write_atomic(target, cb.serialize_bundle("x", []))
    assert victim.read_bytes() == before


def test_an_existing_bundle_is_replaced_atomically(world):
    target, _ = exported(world)
    new = cb.serialize_bundle("second", [ContractRule(**TESTS)])
    cb.write_atomic(target, new)
    assert target.path.read_bytes() == new


def test_the_module_imports_only_what_a_candidate_reader_needs():
    """Not loaded by hook paths (tests/test_hook_import_budget.py); here, that it needs no network or server stack."""
    forbidden = {"httpx", "mcp", "numpy", "pydantic_settings", "dotenv"}
    import subprocess
    code = ("import sys; import self_directing_mcp.contract_bundles; "
            "print(sorted(m for m in %r if m in sys.modules))" % sorted(forbidden))
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root,
                         env={**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
    assert out.returncode == 0 and out.stdout.strip() == "[]", out.stderr
