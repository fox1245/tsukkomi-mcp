"""Contract fragility report: which equivalent spellings of a risky action a contract regex does not match.

A regex over command text is easy to bypass with an equivalent spelling (``rm -r -f`` for ``rm -rf``), channel
(``python3 -c "shutil.rmtree(...)"``) or lexical trick (a backslash-newline, busybox, a GNU option abbreviation).
This module holds a versioned library of such variants (``resources/contract_variants.json``) and reports, for the
regex of a contract being saved, the variants of each family it is "about" that it does not match.

Strictly informational. It never blocks a registration, never edits or broadens a regex and never changes a verdict.
It is deterministic and offline (no model). Every library entry is an unverified candidate (``verified: false``):
equivalence was not verified by execution. It is imported only from ``SelfDirectEngine.upsert_contracts``; hook paths never
load it.

Selection: a family is reported only if the regex matches at least one of its canonical entries (the standard
spellings). Then every entry of that family the regex does not match is a missed variant; an unmatched canonical
entry (for example ``git push -f`` when the regex names ``--force``) counts as one. Families whose canonical entries
the regex does not match are not reported. Matching uses the contract regex engine (``audit.patterns``) with its
time limits: a regex that times out yields an explicit ``regex_timeout`` entry, never a crash or silence.

Tool calls. A contract is matched against the text of a tool call, ``[tool_call:<name>] <JSON arguments>`` (see
``tool_payload``), and that is not a shell command: a regex written around the shell readers (``cat``, ``head``,
``tail``, ``less`` followed by the file name) never matches the call of a host's file-read tool (Claude Code
``Read``, AGY ``view_file``, OMP ``read``). The file-reading families therefore carry such calls as non-canonical entries of the
platform ``tool_call``, each the exact text the parsers and the hooks build for that host. Their canonical entries
stay plain shell readers (so a regex that already matches the tool calls, for example any regex naming the file, is
told nothing about them) and a regex that matches only tool-call text selects no such family. Selection is by
substring search, so a regex that names ``.env`` also matches the canonical spellings of the ``.env.production``
family and is reported for both; a regex about ``.env.production`` does not match the ``.env`` ones. The report
lists each missed tool-call entry on its own (see ``_pick``).

Extension point: ``analyze_pattern(pattern, library)`` accepts any ``Library`` built by ``parse_library`` from a
document in the same schema. No generator of further candidates exists, and none is called here.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json
import re
import time
from typing import Any

from self_directing_mcp.audit.patterns import ContractRegexTimeout, compile_contract, contract_budget

LIBRARY_RESOURCE = ("resources", "contract_variants.json")
SUPPORTED_SCHEMA_VERSION = 1

MAX_MISSED_PER_FAMILY = 8          # variants listed per family; the rest are counted in "omitted"
MAX_FAMILIES_PER_CONTRACT = 4      # families listed per contract; the rest are counted in "families_omitted"
MAX_CONTRACTS = 20                 # contracts analyzed per call; the rest are counted in "contracts_omitted"
REPORT_BUDGET_SEC = 3.0            # all contracts of one call; later contracts are reported as skipped

NOTE = ("Informational only: contracts are saved unchanged and no verdict changes. Variants are unverified "
        "candidates (equivalence was not verified by execution), matched as plain text: a command line, or for "
        "the tool_call platform the text of a tool call, [tool_call:<name>] <JSON arguments>.")
TOOL_CALL_PLATFORM = "tool_call"

_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_ID = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")


@dataclass(frozen=True)
class Variant:
    id: str
    text: str
    platform: str
    technique: str
    preconditions: tuple[str, ...]
    effect: str
    verified: bool
    canonical: bool = False


@dataclass(frozen=True)
class Family:
    family: str
    intent: str
    effect: str
    variants: tuple[Variant, ...]
    notes: str = ""


@dataclass(frozen=True)
class Library:
    version: str
    schema_version: int
    platforms: frozenset[str]
    techniques: tuple[str, ...]          # in the library's priority order (see _pick)
    families: tuple[Family, ...]

    @property
    def all_verified(self) -> bool:
        return all(v.verified for family in self.families for v in family.variants)


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"variant library: {where} must be a non-empty string")
    return value


def parse_library(raw: Any) -> Library:
    """Validate the decoded JSON and build the library. Raises ``ValueError`` naming the first defect."""
    if not isinstance(raw, dict):
        raise ValueError("variant library: top level must be an object")
    if raw.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"variant library: unsupported schema_version {raw.get('schema_version')!r}")
    version = _string(raw.get("library_version"), "library_version")
    if not _VERSION.match(version):
        raise ValueError(f"variant library: library_version {version!r} is not MAJOR.MINOR.PATCH")
    platforms, techniques = raw.get("platforms"), raw.get("techniques")
    if not isinstance(platforms, dict) or not platforms or not isinstance(techniques, dict) or not techniques:
        raise ValueError("variant library: platforms and techniques must be non-empty objects")
    families, seen_ids, seen_canonical = [], set(), set()
    names = set()
    for fam in raw.get("families") or []:
        if not isinstance(fam, dict):
            raise ValueError("variant library: a family must be an object")
        name = _string(fam.get("family"), "family")
        if name in names:
            raise ValueError(f"variant library: duplicate family {name!r}")
        names.add(name)
        intent, effect = _string(fam.get("intent"), f"{name}.intent"), _string(fam.get("effect"), f"{name}.effect")
        variants = []
        for item in fam.get("variants") or []:
            if not isinstance(item, dict):
                raise ValueError(f"variant library: {name}: a variant must be an object")
            vid = _string(item.get("id"), f"{name}: id")
            if not _ID.match(vid) or not vid.startswith(name + "."):
                raise ValueError(f"variant library: variant id {vid!r} must look like '{name}.<slug>'")
            if vid in seen_ids:
                raise ValueError(f"variant library: duplicate variant id {vid!r}")
            seen_ids.add(vid)
            platform, technique = item.get("platform"), item.get("technique")
            if platform not in platforms:
                raise ValueError(f"variant library: {vid}: unknown platform {platform!r}")
            if technique not in techniques:
                raise ValueError(f"variant library: {vid}: unknown technique {technique!r}")
            pre = item.get("preconditions")
            if not isinstance(pre, list) or not all(isinstance(p, str) and p for p in pre):
                raise ValueError(f"variant library: {vid}: preconditions must be a list of strings")
            if not isinstance(item.get("verified"), bool):
                raise ValueError(f"variant library: {vid}: verified must be a boolean")
            canonical = item.get("canonical", False)
            if not isinstance(canonical, bool):
                raise ValueError(f"variant library: {vid}: canonical must be a boolean")
            text = _string(item.get("text"), f"{vid}.text")
            if canonical:
                if text in seen_canonical:
                    raise ValueError(f"variant library: canonical text {text!r} appears twice")
                seen_canonical.add(text)
            variants.append(Variant(vid, text, platform, technique, tuple(pre),
                                    _string(item["effect"], f"{vid}.effect") if "effect" in item else effect,
                                    item["verified"], canonical))
        if not any(v.canonical for v in variants):
            raise ValueError(f"variant library: family {name!r} has no canonical entry")
        if len({v.text for v in variants}) != len(variants):
            raise ValueError(f"variant library: family {name!r} repeats a text")
        families.append(Family(name, intent, effect, tuple(variants), str(fam.get("notes") or "")))
    if not families:
        raise ValueError("variant library: no families")
    return Library(version, SUPPORTED_SCHEMA_VERSION, frozenset(platforms), tuple(techniques), tuple(families))


_library: Library | None = None


def load_library() -> Library:
    """The packaged library (read and validated once per process)."""
    global _library
    if _library is None:
        resource = resources.files("self_directing_mcp").joinpath(*LIBRARY_RESOURCE)
        _library = parse_library(json.loads(resource.read_text(encoding="utf-8")))
    return _library


def _variant_view(variant: Variant, family: Family) -> dict[str, Any]:
    view: dict[str, Any] = {"id": variant.id, "text": variant.text, "platform": variant.platform,
                            "technique": variant.technique, "verified": variant.verified}
    if variant.canonical:
        view["canonical"] = True
    if variant.preconditions:
        view["preconditions"] = list(variant.preconditions)
    if variant.effect != family.effect:
        view["effect"] = variant.effect
    return view


def _kind(variant: Variant) -> tuple[str, str]:
    """What counts as one kind of gap in a capped list: a technique, except that every tool call is its own kind
    (each host names its file-read tool and its path argument differently, and a regex can miss any of them)."""
    return variant.technique, variant.id if variant.platform == TOOL_CALL_PLATFORM else ""


def _pick(missed: list[Variant], cap: int, priority: tuple[str, ...]) -> list[Variant]:
    """At most ``cap`` of the missed variants, in library order: unmatched canonical spellings first, then one
    variant per technique in turn (techniques in the library's priority order; each tool call counts as a
    technique of its own), so a capped list shows different kinds of gap rather than the first eight option
    spellings."""
    if len(missed) <= cap:
        return missed
    picked = [v for v in missed if v.canonical][:cap]
    queues: dict[tuple[str, str], list[Variant]] = {}
    for variant in missed:
        if not variant.canonical:
            queues.setdefault(_kind(variant), []).append(variant)
    rank = {technique: index for index, technique in enumerate(priority)}
    while len(picked) < cap and queues:
        for kind in sorted(queues, key=lambda kind: rank[kind[0]]):     # stable: equal ranks keep library order
            picked.append(queues[kind].pop(0))
            if not queues[kind]:
                del queues[kind]
            if len(picked) >= cap:
                break
    order = {id(variant): index for index, variant in enumerate(missed)}
    return sorted(picked, key=lambda variant: order[id(variant)])


def analyze_pattern(pattern: str, library: Library | None = None) -> dict[str, Any]:
    """Report the variants one contract regex misses. ``status`` is ``ok`` or ``regex_timeout``."""
    library = library or load_library()
    compiled = compile_contract(pattern)
    found: list[dict[str, Any]] = []
    try:
        with contract_budget():
            for family in library.families:
                matches = {variant.id: compiled.search(variant.text) is not None
                           for variant in family.variants if variant.canonical}
                if not any(matches.values()):
                    continue
                for variant in family.variants:
                    if variant.id not in matches:
                        matches[variant.id] = compiled.search(variant.text) is not None
                missed = [variant for variant in family.variants if not matches[variant.id]]
                if not missed:
                    continue
                listed = _pick(missed, MAX_MISSED_PER_FAMILY, library.techniques)
                found.append({
                    "family": family.family,
                    "intent": family.intent,
                    "effect": family.effect,
                    "matched_canonical": [v.text for v in family.variants if v.canonical and matches[v.id]],
                    "missed_variants": [_variant_view(v, family) for v in listed],
                    "missed_total": len(missed),
                    "omitted": len(missed) - len(listed),
                })
    except ContractRegexTimeout:
        return {"status": "regex_timeout", "families": [], "families_omitted": 0}
    return {"status": "ok", "families": found[:MAX_FAMILIES_PER_CONTRACT],
            "families_omitted": max(0, len(found) - MAX_FAMILIES_PER_CONTRACT)}


def _analyzed_pattern(rule: Any) -> tuple[str, str | None]:
    """The pattern that names the action a contract is about: a prohibition's regex, or the gated action
    (``before_regex``) of an obligation."""
    if getattr(rule, "type", None) == "must_not":
        return "regex", getattr(rule, "regex", None)
    return "before_regex", getattr(rule, "before_regex", None)


def fragility_report(rules: list[Any]) -> dict[str, Any]:
    """The ``fragility`` section of an ``upsert_contracts`` response for the saved rules."""
    library = load_library()
    deadline = time.monotonic() + REPORT_BUDGET_SEC
    candidates = []
    for rule in rules:
        field, pattern = _analyzed_pattern(rule)
        if pattern:
            candidates.append((rule, field, pattern))
    contracts: list[dict[str, Any]] = []
    for rule, field, pattern in candidates[:MAX_CONTRACTS]:
        if time.monotonic() >= deadline:
            result: dict[str, Any] = {"status": "skipped_time_budget", "families": [], "families_omitted": 0}
        else:
            result = analyze_pattern(pattern, library)
        if result["status"] != "ok" or result["families"]:
            contracts.append({"contract_id": rule.id, "field": field, **result})
    report: dict[str, Any] = {
        "library_version": library.version,
        "informational": True,
        "verified": library.all_verified,
        "note": NOTE,
        "analyzed": min(len(candidates), MAX_CONTRACTS),
        "contracts": contracts,
    }
    if len(candidates) > MAX_CONTRACTS:
        report["contracts_not_analyzed"] = len(candidates) - MAX_CONTRACTS
    return report
