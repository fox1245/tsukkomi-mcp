from __future__ import annotations

from contextlib import contextmanager
import json
import math
import re
import time
from typing import Any, Iterator

from self_directing_mcp.audit.contracts import check_rule_patterns
from self_directing_mcp.audit.patterns import (
    ContractRegexTimeout,
    SEARCH_TIMEOUT_SEC,
    _remaining,
    compile_contract,
    contract_budget,
)
from self_directing_mcp.audit.runner import _eligible
from self_directing_mcp.tool_payload import tool_call_parts
from self_directing_mcp.schemas import AuditEvent, ContractCase, ContractRule


DEFAULT_CASE_MAX_MS = 250.0
MAX_CASE_MAX_MS = 10000.0
MAX_CONTRACT_CASES = 64
MAX_SUITE_ELAPSED_MS = 5000.0
MAX_PREREQ_PROBES = 32
_RESERVED_EVENT_META_KEYS = frozenset({"role", "tool_name", "call_id", "session_id", "kind"})
_TOOL_CALL_NAME_HEADER_RE = re.compile(r"^(tool_call name=)([!-~]+)((?:\n.*)?)$", re.DOTALL)


@contextmanager
def _ensure_eval_budget(default_ms: float = DEFAULT_CASE_MAX_MS) -> Iterator[None]:
    """Ensure every regex/prerequisite evaluation runs under an active contract_budget bounded by default_ms and MAX_SUITE_ELAPSED_MS."""
    rem = _remaining.get()
    if rem is None:
        if (
            default_ms is None
            or isinstance(default_ms, bool)
            or not isinstance(default_ms, (int, float))
            or not math.isfinite(float(default_ms))
            or float(default_ms) <= 0.0
        ):
            eff_ms = DEFAULT_CASE_MAX_MS
        else:
            eff_ms = min(max(float(default_ms), 1.0), MAX_CASE_MAX_MS, MAX_SUITE_ELAPSED_MS)
        with contract_budget(eff_ms / 1000.0):
            yield
    else:
        if rem <= 0.0:
            raise ContractRegexTimeout("contract evaluation budget exhausted")
        yield


def _valid_max_ms(val: Any) -> float | None:
    """Validate and return a finite positive max_ms cap, raising ValueError on non-positive or NaN/inf inputs."""
    if val is None:
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError("ContractCase.max_ms must be a finite positive number")
    f = float(val)
    if not math.isfinite(f) or f <= 0.0:
        raise ValueError("ContractCase.max_ms must be a finite positive number")
    return min(f, MAX_CASE_MAX_MS)


def _valid_case_id(case_id: Any) -> str | None:
    """Validate explicit ContractCase.id and reject reserved internal '__anon_case_' namespace."""
    if case_id is None:
        return None
    if not isinstance(case_id, str) or not case_id or len(case_id) > 128:
        raise ValueError("ContractCase.id must be a non-empty ASCII printable token up to 128 characters")
    for ch in case_id:
        code = ord(ch)
        if code <= 0x20 or code >= 0x7F:
            raise ValueError("ContractCase.id must not contain whitespace, control, or non-ASCII characters")
    if case_id.startswith("__anon_case_"):
        raise ValueError("ContractCase.id cannot use reserved '__anon_case_' prefix")
    return case_id


def case_dedup_key(c: ContractCase) -> tuple[Any, ...]:
    """Canonical deduplication key across full semantic identity and input fields of a ContractCase."""
    args_norm = (
        ("dict", json.dumps(c.arguments, sort_keys=True, ensure_ascii=False))
        if isinstance(c.arguments, dict)
        else ("raw", c.arguments)
    )
    fields_set = frozenset(getattr(c, "model_fields_set", set()) - {"id", "max_ms"})
    return (
        c.id,
        "id" in getattr(c, "model_fields_set", set()),
        c.kind,
        c.role,
        c.tool_name,
        c.command,
        args_norm,
        c.text,
        c.expected_match,
        fields_set,
    )


def merge_contract_cases(
    candidate_cases: list[ContractCase],
    previous_cases: list[ContractCase],
    *,
    allow_id_override: bool = False,
) -> list[ContractCase]:
    """Merge candidate and baseline ContractCases without allowing an unanchored reused id to drop a distinct baseline case."""
    if len(candidate_cases) > MAX_CONTRACT_CASES:
        raise ValueError(f"Contract rule cannot exceed {MAX_CONTRACT_CASES} candidate test cases (got {len(candidate_cases)})")
    if len(previous_cases) > MAX_CONTRACT_CASES:
        raise ValueError(f"Contract rule cannot exceed {MAX_CONTRACT_CASES} baseline test cases (got {len(previous_cases)})")
    by_key: dict[tuple[Any, ...], ContractCase] = {}
    order: list[tuple[Any, ...]] = []
    used_ids: set[str] = set()
    candidate_explicit_ids: dict[str, tuple[Any, ...]] = {}
    for c in candidate_cases:
        _valid_case_id(c.id)
        c_max = _valid_max_ms(c.max_ms)
        if c_max is not None and c_max != c.max_ms:
            c = c.model_copy(update={"max_ms": c_max})
        k = case_dedup_key(c)
        if k not in by_key:
            if c.id and c.id in used_ids:
                raise ValueError(f"Duplicate ContractCase id {c.id!r} with conflicting definition in candidate cases")
            if len(order) >= MAX_CONTRACT_CASES:
                raise ValueError(f"Contract rule cannot exceed {MAX_CONTRACT_CASES} test cases")
            by_key[k] = c
            order.append(k)
            if c.id:
                used_ids.add(c.id)
                candidate_explicit_ids[c.id] = k
        else:
            existing = by_key[k]
            ex_max = _valid_max_ms(existing.max_ms)
            if c_max is not None and (ex_max is None or c_max < ex_max):
                by_key[k] = existing.model_copy(update={"max_ms": c_max})
    for idx, prev_c in enumerate(previous_cases):
        _valid_case_id(prev_c.id)
        p_max = _valid_max_ms(prev_c.max_ms)
        if p_max is not None and p_max != prev_c.max_ms:
            prev_c = prev_c.model_copy(update={"max_ms": p_max})
        if allow_id_override and prev_c.id and prev_c.id in candidate_explicit_ids:
            cand_key = candidate_explicit_ids[prev_c.id]
            cand_existing = by_key[cand_key]
            cand_max = _valid_max_ms(cand_existing.max_ms)
            if p_max is not None and (cand_max is None or p_max < cand_max):
                by_key[cand_key] = cand_existing.model_copy(update={"max_ms": p_max})
            continue
        k = case_dedup_key(prev_c)
        if k in by_key:
            existing = by_key[k]
            ex_max = _valid_max_ms(existing.max_ms)
            if p_max is not None and (ex_max is None or p_max < ex_max):
                by_key[k] = existing.model_copy(update={"max_ms": p_max})
            continue
        if len(order) >= MAX_CONTRACT_CASES:
            raise ValueError(f"Contract rule cannot exceed {MAX_CONTRACT_CASES} test cases")
        to_store = prev_c
        if prev_c.id and prev_c.id in used_ids:
            suffix = idx + 1
            disambiguated = False
            for _ in range(MAX_CONTRACT_CASES + 2):
                disambig_id = f"{prev_c.id}__baseline_{suffix}"
                if len(disambig_id) > 128:
                    raise ValueError(f"Unable to disambiguate overlong baseline ContractCase id {prev_c.id!r}")
                candidate_store = prev_c.model_copy(update={"id": disambig_id})
                cand_k = case_dedup_key(candidate_store)
                if disambig_id not in used_ids and cand_k not in by_key:
                    to_store = candidate_store
                    k = cand_k
                    disambiguated = True
                    break
                suffix += 1
            if not disambiguated:
                raise ValueError(f"Unable to disambiguate baseline ContractCase id {prev_c.id!r}")
        by_key[k] = to_store
        order.append(k)
        if to_store.id:
            used_ids.add(to_store.id)
    if len(order) > MAX_CONTRACT_CASES:
        raise ValueError(f"Contract rule cannot exceed {MAX_CONTRACT_CASES} test cases (got {len(order)})")
    final_cases: list[ContractCase] = []
    for out_idx, k in enumerate(order, start=1):
        item = by_key[k]
        if not item.id:
            auto_id = f"case_{out_idx}"
            if auto_id in used_ids:
                suffix = 1
                assigned = False
                for _ in range(MAX_CONTRACT_CASES + 2):
                    cand_auto = f"case_{out_idx}__auto_{suffix}"
                    if cand_auto not in used_ids:
                        auto_id = cand_auto
                        assigned = True
                        break
                    suffix += 1
                if not assigned:
                    raise ValueError("Unable to assign non-colliding ContractCase id")
            used_ids.add(auto_id)
            item = item.model_copy(update={"id": auto_id})
        final_cases.append(item)
    return final_cases


def _normalize_case_arguments(arguments: Any) -> dict[str, Any] | str:
    """Validate and return case arguments as a shallow dict copy or raw string without lossy coercion or JSON guessing."""
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        return arguments
    raise ValueError("ContractCase.arguments must be a dict or str")


def _is_valid_case_tool_name(tool_name: Any) -> bool:
    """Return True iff tool_name is a non-empty ASCII printable token without whitespace, control chars, or brackets."""
    if not isinstance(tool_name, str) or not tool_name or len(tool_name) > 128:
        return False
    for ch in tool_name:
        code = ord(ch)
        if code <= 0x20 or code >= 0x7F or ch in "[]":
            return False
    return True


def render_case_event(
    case: ContractCase,
    rule: ContractRule | None = None,
    index: int = 0,
    provider: str = "claude",
    *,
    override_tool: str | None = None,
    override_role: str | None = None,
) -> AuditEvent:
    """Synthesize a canonical AuditEvent from a ContractCase, inheriting rule scope/role/tool defaults when unspecified."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("render_case_event index must be a non-negative integer")
    explicit_id = _valid_case_id(case.id)
    case_id = explicit_id if explicit_id is not None else f"__anon_case_{index + 1}"
    effective_provider = (rule.provider if (rule is not None and rule.provider) else provider) or "claude"
    fields_set = getattr(case, "model_fields_set", set())
    eff_role = override_role if override_role is not None else case.role
    if override_role is None and "role" not in fields_set and rule is not None and getattr(rule, "roles", None):
        eff_role = rule.roles[0]
    eff_tool = override_tool if override_tool is not None else case.tool_name
    if override_tool is None and "tool_name" not in fields_set and rule is not None and getattr(rule, "tools", None):
        eff_tool = rule.tools[0]
    if eff_tool is not None and not _is_valid_case_tool_name(eff_tool):
        raise ValueError(f"invalid tool_name: {eff_tool!r}")

    if case.text is not None and (case.command is not None or case.arguments is not None):
        raise ValueError("ContractCase cannot specify both 'text' and 'command'/'arguments'")
    if (case.command is not None or case.arguments is not None) and case.text is None:
        if case.arguments is None:
            payload_args: dict[str, Any] | str = {"command": case.command}
        else:
            norm_args = _normalize_case_arguments(case.arguments)
            if isinstance(norm_args, str):
                if case.command is not None:
                    raise ValueError(
                        "ContractCase cannot combine string 'arguments' with 'command'; pass a dict for 'arguments'"
                    )
                payload_args = norm_args
            else:
                merged_args = dict(norm_args)
                if case.command is not None:
                    if "command" in merged_args and merged_args["command"] != case.command:
                        raise ValueError("conflicting case.command and case.arguments['command']")
                    merged_args["command"] = case.command
                payload_args = merged_args
        text, prose = tool_call_parts(
            effective_provider, eff_tool, payload_args
        )
        kind = case.kind or "tool_call"
        safe_prose = {
            k: v for k, v in (prose or {}).items() if k not in _RESERVED_EVENT_META_KEYS
        }
        meta = {
            **safe_prose,
            "role": eff_role,
            "tool_name": eff_tool,
            "call_id": case_id,
        }
        return AuditEvent(case_id, kind, text, meta)

    raw_text = case.text or ""
    first_line = raw_text.split("\n", 1)[0]
    starts_bracket = raw_text.startswith("[tool_call:") or raw_text.startswith("[proposed_tool_call:")
    if not starts_bracket and re.match(r"^\[(?:proposed_)?tool_call(?![A-Za-z0-9_])", raw_text):
        raise ValueError("malformed tool_call bracket header in ContractCase.text")
    if starts_bracket and ("]" not in first_line or first_line.split("]", 1)[1] != ""):
        raise ValueError("malformed tool_call bracket header in ContractCase.text")
    has_bracket_header = starts_bracket
    m_name_hdr = _TOOL_CALL_NAME_HEADER_RE.match(raw_text) if not has_bracket_header else None
    if not has_bracket_header and m_name_hdr is None and re.match(r"^(?:proposed_)?tool_call(?![A-Za-z0-9_])", raw_text):
        raise ValueError("malformed tool_call name= header in ContractCase.text")

    hdr_tool: str | None = None
    if has_bracket_header:
        header = first_line.split("]", 1)[0]
        if ":" not in header:
            raise ValueError("malformed tool_call bracket header in ContractCase.text")
        hdr_tool = header.split(":", 1)[1]
        if not _is_valid_case_tool_name(hdr_tool):
            raise ValueError(f"invalid tool_name in raw header: {hdr_tool!r}")
    elif m_name_hdr is not None:
        hdr_tool = m_name_hdr.group(2)
        if not _is_valid_case_tool_name(hdr_tool):
            raise ValueError(f"invalid tool_name in raw header: {hdr_tool!r}")

    if override_tool is None and "tool_name" in fields_set:
        if not _is_valid_case_tool_name(case.tool_name):
            raise ValueError(f"invalid tool_name: {case.tool_name!r}")
        if hdr_tool is not None and hdr_tool != case.tool_name:
            raise ValueError(
                f"conflicting ContractCase.tool_name {case.tool_name!r} and raw text header tool {hdr_tool!r}"
            )

    if override_tool is not None:
        inferred_tool = override_tool
    elif "tool_name" in fields_set:
        inferred_tool = case.tool_name
    elif hdr_tool is not None:
        inferred_tool = hdr_tool
    else:
        inferred_tool = eff_tool

    if inferred_tool is not None and not _is_valid_case_tool_name(inferred_tool):
        raise ValueError(f"invalid tool_name: {inferred_tool!r}")

    if inferred_tool is not None:
        if has_bracket_header and hdr_tool != inferred_tool:
            prefix_tag = "[proposed_tool_call:" if raw_text.startswith("[proposed_tool_call:") else "[tool_call:"
            rest = raw_text.split("]", 1)[1]
            raw_text = f"{prefix_tag}{inferred_tool}]{rest}"
        elif m_name_hdr is not None and hdr_tool != inferred_tool:
            raw_text = f"{m_name_hdr.group(1)}{inferred_tool}{m_name_hdr.group(3)}"

    if case.kind is not None:
        kind = case.kind
    elif has_bracket_header or m_name_hdr is not None:
        kind = "tool_call"
    elif rule is not None and rule.scope == "message":
        kind = "turn"
    elif rule is not None and rule.scope == "tool_result":
        kind = "tool_result"
    else:
        kind = "tool_call"

    meta = {"role": eff_role, "tool_name": inferred_tool, "call_id": case_id}
    return AuditEvent(case_id, kind, raw_text, meta)


def _search_at(compiled: Any, text: str, pos: int = 0, endpos: int | None = None) -> Any:
    """Search `text` starting at `pos` (preserving true string-start `\\A` and line-start `^` semantics) under `contract_budget`."""
    raw = getattr(compiled, "_compiled", None)
    if raw is None:
        raise TypeError("compiled pattern must be a GuardedPattern")
    rem = _remaining.get()
    default_limit = min(SEARCH_TIMEOUT_SEC, DEFAULT_CASE_MAX_MS / 1000.0, MAX_SUITE_ELAPSED_MS / 1000.0)
    limit = default_limit if rem is None else min(SEARCH_TIMEOUT_SEC, rem)
    if limit <= 0:
        raise ContractRegexTimeout(getattr(compiled, "pattern", ""))
    started = time.perf_counter()
    try:
        if pos == 0 and endpos is None:
            return raw.search(text, timeout=limit)
        if endpos is None:
            return raw.search(text, pos, timeout=limit)
        return raw.search(text, pos, endpos, timeout=limit)
    except TimeoutError:
        raise ContractRegexTimeout(getattr(compiled, "pattern", "")) from None
    finally:
        if rem is not None:
            new_rem = (_remaining.get() or 0.0) - (time.perf_counter() - started)
            _remaining.set(new_rem)
            if new_rem <= 0:
                raise ContractRegexTimeout(getattr(compiled, "pattern", ""))


def _require_non_nullable(compiled: Any, label: str) -> None:
    """Reject any contract pattern that matches the empty string."""
    if _search_at(compiled, "", 0) is not None:
        raise ValueError(f"{label} must not match the empty string")


def _search_nonempty_at(compiled: Any, text: str, pos: int = 0, endpos: int | None = None, *, label: str = "regex") -> Any:
    """Search `text` from `pos` and reject any zero-width match (`m.end() <= m.start()`)."""
    m = _search_at(compiled, text, pos=pos, endpos=endpos)
    if m is not None and m.end() <= m.start():
        raise ValueError(f"{label} must not produce a zero-width match")
    return m


def _is_regex_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _segment_has_prerequisite(
    req_compiled: Any,
    full_text: str,
    seg_start: int,
    seg_end: int,
) -> bool:
    """Return True iff full_text[seg_start:seg_end] contains a valid non-empty match of req_compiled
    whose boundary assertions match the real character context at seg_start and seg_end in O(seg_end - seg_start) memory."""
    if seg_end <= seg_start:
        return False
    with _ensure_eval_budget():
        _require_non_nullable(req_compiled, "prerequisite regex")
        text_len = len(full_text)
        if seg_start == 0:
            lead_a = ""
            lead_b = ""
            win_start = 0
        else:
            prev_ch = full_text[seg_start - 1]
            if prev_ch == "\n":
                lead_a = "\x00\n"
                lead_b = "\x01\n"
            elif _is_regex_word_char(prev_ch):
                lead_a = "\x00a"
                lead_b = "\x01b"
            else:
                lead_a = "\x00\x00"
                lead_b = "\x01\x01"
            win_start = 2

        if seg_end >= text_len:
            tail_a = ""
            tail_b = ""
        elif full_text[seg_end] == "\n":
            tail_a = "\n\x00"
            tail_b = "\n\x01"
        elif _is_regex_word_char(full_text[seg_end]):
            tail_a = "a\x00"
            tail_b = "b\x01"
        else:
            tail_a = "\x00\x00"
            tail_b = "\x01\x01"

        seg_slice = full_text[seg_start:seg_end]
        probe_a = lead_a + seg_slice + tail_a
        probe_b = lead_b + seg_slice + tail_b
        win_end = win_start + (seg_end - seg_start)

        cur_win_pos = win_start
        probes = 0
        while cur_win_pos < win_end:
            probes += 1
            if probes > MAX_PREREQ_PROBES:
                return False
            m_a = _search_nonempty_at(req_compiled, probe_a, pos=cur_win_pos, label="prerequisite regex")
            if m_a is None or m_a.start() >= win_end:
                return False
            if win_start <= m_a.start() < m_a.end() <= win_end:
                real_start = seg_start + (m_a.start() - win_start)
                real_end = seg_start + (m_a.end() - win_start)
                m_trunc = _search_nonempty_at(
                    req_compiled, full_text, pos=real_start, endpos=seg_end, label="prerequisite regex"
                )
                if m_trunc is not None and m_trunc.span() == (real_start, real_end):
                    m_full = _search_nonempty_at(
                        req_compiled, full_text, pos=real_start, label="prerequisite regex"
                    )
                    if m_full is not None and m_full.span() == (real_start, real_end):
                        if probe_b == probe_a:
                            return True
                        m_b = _search_nonempty_at(
                            req_compiled, probe_b, pos=m_a.start(), label="prerequisite regex"
                        )
                        if m_b is not None and m_b.span() == m_a.span():
                            return True
            cur_win_pos = m_a.start() + 1
        return False


def evaluate_rule_on_event(rule: ContractRule, event: AuditEvent) -> tuple[bool, str]:
    """Evaluate whether rule matches event under canonical scope/role/tool/regex semantics."""
    with _ensure_eval_budget():
        if not _eligible(event, rule):
            return False, "not_eligible"
        if rule.before_regex and rule.type != "must":
            raise ValueError(f"before_regex is only supported on 'must' contracts (got {rule.type!r})")
        if rule.type == "must" and rule.before_regex:
            if rule.exception_regex:
                exc_compiled = compile_contract(rule.exception_regex)
                _require_non_nullable(exc_compiled, "exception_regex")
                if _search_nonempty_at(exc_compiled, event.text, label="exception_regex") is not None:
                    return False, "exception_matched"
            before_compiled = compile_contract(rule.before_regex)
            _require_non_nullable(before_compiled, "before_regex")
            m_before = _search_nonempty_at(before_compiled, event.text, label="before_regex")
            before_hit = m_before is not None
            if not rule.regex:
                if not before_hit:
                    return False, "no_match"
                return True, "matched_before_regex"
            req_compiled = compile_contract(rule.regex)
            _require_non_nullable(req_compiled, "regex")
            m_req = _search_nonempty_at(req_compiled, event.text, label="regex")
            req_hit = m_req is not None
            if req_hit and not before_hit:
                return True, "matched"
            if req_hit and before_hit:
                assert m_req is not None and m_before is not None
                prev_end = 0
                pos = 0
                text_len = len(event.text)
                found_trigger = False
                trigger_steps = 0
                while pos <= text_len:
                    trigger_steps += 1
                    if trigger_steps > MAX_PREREQ_PROBES:
                        return False, "too_many_before_regex_triggers"
                    m_tr = _search_nonempty_at(before_compiled, event.text, pos=pos, label="before_regex")
                    if m_tr is None:
                        break
                    found_trigger = True
                    t_start, t_end = m_tr.start(), m_tr.end()
                    if not _segment_has_prerequisite(req_compiled, event.text, prev_end, t_start):
                        return False, "matched_both_prerequisite_and_before_regex"
                    prev_end = t_end
                    pos = t_end
                if not found_trigger:
                    if not _segment_has_prerequisite(req_compiled, event.text, 0, m_before.start()):
                        return False, "matched_both_prerequisite_and_before_regex"
                return True, "matched_prerequisite_before_trigger"
            if before_hit:
                return False, "matched_before_regex_without_prerequisite"
            return False, "no_match"
        pattern_text = rule.regex
        if not pattern_text:
            return False, "no_regex"
        compiled = compile_contract(pattern_text)
        _require_non_nullable(compiled, "regex")
        if _search_nonempty_at(compiled, event.text, label="regex") is None:
            return False, "no_match"
        if rule.exception_regex:
            exc = compile_contract(rule.exception_regex)
            _require_non_nullable(exc, "exception_regex")
            if _search_nonempty_at(exc, event.text, label="exception_regex") is not None:
                return False, "exception_matched"
        return True, "matched"


def _eval_single_case(
    rule: ContractRule,
    case: ContractCase,
    index: int,
    default_max_ms: float | None = None,
    provider: str = "claude",
    *,
    remaining_suite_ms: float | None = None,
) -> dict[str, Any]:
    case_id = case.id or f"case_{index + 1}"
    if (
        default_max_ms is None
        or isinstance(default_max_ms, bool)
        or not isinstance(default_max_ms, (int, float))
        or not math.isfinite(float(default_max_ms))
        or float(default_max_ms) <= 0
    ):
        effective_default = DEFAULT_CASE_MAX_MS
    else:
        effective_default = min(float(default_max_ms), MAX_CASE_MAX_MS)
    effective_provider = (rule.provider if rule.provider else provider) or "claude"
    fields_set = getattr(case, "model_fields_set", set())
    try:
        validated_case_max = _valid_max_ms(case.max_ms)
        limit_ms = validated_case_max if validated_case_max is not None else effective_default
    except Exception as exc:
        return {
            "id": case_id,
            "index": index,
            "rendered_text": "",
            "kind": case.kind or "tool_call",
            "tool_name": case.tool_name,
            "expected": "match" if case.expected_match else "no_match",
            "expected_match": case.expected_match,
            "actual": "no_match",
            "matched": False,
            "detail": f"render_error: {exc}",
            "elapsed_ms": 0.0,
            "max_ms": effective_default,
            "timed_out": False,
            "incomplete": True,
            "passed": False,
            "failure_reason": f"render_error: {exc}",
        }

    # Suite kill switch runs BEFORE render_case_event so exhausted suites perform zero further render or eval work.
    if remaining_suite_ms is not None and remaining_suite_ms <= 0.0:
        return {
            "id": case_id,
            "index": index,
            "rendered_text": "",
            "kind": case.kind or "tool_call",
            "tool_name": case.tool_name,
            "expected": "match" if case.expected_match else "no_match",
            "expected_match": case.expected_match,
            "actual": "no_match",
            "matched": False,
            "detail": "suite_timeout",
            "elapsed_ms": 0.0,
            "max_ms": limit_ms,
            "timed_out": True,
            "incomplete": True,
            "passed": False,
            "failure_reason": f"suite elapsed time exceeded {MAX_SUITE_ELAPSED_MS}ms budget",
        }

    eff_budget_ms = limit_ms if remaining_suite_ms is None else min(limit_ms, remaining_suite_ms)
    budget_sec = min(max(eff_budget_ms / 1000.0, 0.001), MAX_CASE_MAX_MS / 1000.0)
    try:
        with contract_budget(budget_sec):
            event = render_case_event(case, rule=rule, index=index, provider=effective_provider)
    except Exception as exc:
        return {
            "id": case_id,
            "index": index,
            "rendered_text": "",
            "kind": case.kind or "tool_call",
            "tool_name": case.tool_name,
            "expected": "match" if case.expected_match else "no_match",
            "expected_match": case.expected_match,
            "actual": "no_match",
            "matched": False,
            "detail": f"render_error: {exc}",
            "elapsed_ms": 0.0,
            "max_ms": limit_ms,
            "timed_out": False,
            "incomplete": True,
            "passed": False,
            "failure_reason": f"render_error: {exc}",
        }

    started = time.perf_counter()
    elapsed_ms = 0.0
    timed_out = False
    error: str | None = None
    matched = False
    detail = "no_match"
    provider_mismatch: str | None = None
    try:
        with contract_budget(budget_sec):
            t_primary = time.perf_counter()
            matched, detail = evaluate_rule_on_event(rule, event)
            # Measure single-evaluation latency directly on the primary evaluation (and take max over any single probe below,
            # rather than summing N fan-out probes, so multi-tool/multi-provider rules are not penalized vs 1-tool baselines).
            elapsed_ms = round((time.perf_counter() - t_primary) * 1000.0, 3)

            # 1) When rule applies across all hosts (not rule.provider) and the case synthesizes a command/arguments,
            #    verify both match boolean AND eligibility/detail consistency across claude, codex, agy, and caller provider.
            if not rule.provider and case.text is None and (case.command is not None or case.arguments is not None):
                all_provs = tuple(dict.fromkeys(p for p in ("claude", "codex", "agy", provider, effective_provider) if p))
                for other_prov in all_provs:
                    if other_prov == effective_provider:
                        continue
                    other_ev = render_case_event(case, rule=rule, index=index, provider=other_prov)
                    t_p = time.perf_counter()
                    other_matched, other_detail = evaluate_rule_on_event(rule, other_ev)
                    elapsed_ms = max(elapsed_ms, round((time.perf_counter() - t_p) * 1000.0, 3))
                    if other_detail == "not_eligible" or other_matched != matched or other_detail != detail:
                        provider_mismatch = (
                            f"provider rendering mismatch: {effective_provider}={'match' if matched else 'no_match'} ({detail}) "
                            f"vs {other_prov}={'match' if other_matched else 'no_match'} ({other_detail})"
                        )
                        break

            primary_tool = event.meta.get("tool_name")
            primary_tool_label = primary_tool or "default"
            ev_first_line = event.text.split("\n", 1)[0]
            has_hdr = (
                (
                    (ev_first_line.startswith("[tool_call:") or ev_first_line.startswith("[proposed_tool_call:"))
                    and "]" in ev_first_line
                    and ev_first_line.split("]", 1)[1].strip() == ""
                )
                or (_TOOL_CALL_NAME_HEADER_RE.match(event.text) is not None)
            )

            # 2) When case does not pin tool_name and rule.tools has multiple tools, verify both match boolean AND eligibility/detail
            #    consistency across ALL rule.tools.
            if (
                provider_mismatch is None
                and "tool_name" not in fields_set
                and getattr(rule, "tools", None)
                and len(rule.tools) > 1
            ):
                for alt_tool in rule.tools:
                    alt_ev = render_case_event(
                        case, rule=rule, index=index, provider=effective_provider, override_tool=alt_tool
                    )
                    t_p = time.perf_counter()
                    alt_matched, alt_detail = evaluate_rule_on_event(rule, alt_ev)
                    elapsed_ms = max(elapsed_ms, round((time.perf_counter() - t_p) * 1000.0, 3))
                    if alt_detail == "not_eligible" or alt_matched != matched or alt_detail != detail:
                        provider_mismatch = (
                            f"tool rendering mismatch across rule.tools: {primary_tool_label}={'match' if matched else 'no_match'} ({detail}) "
                            f"vs {alt_tool}={'match' if alt_matched else 'no_match'} ({alt_detail})"
                        )
                        break

            # 2b) Header-stability differential check for unheaded raw text tool_call/proposed_tool_call cases
            #     (applies regardless of whether rule.tools has 0, 1, or multiple tools).
            if (
                provider_mismatch is None
                and case.text is not None
                and event.kind in ("tool_call", "proposed_tool_call")
                and not has_hdr
            ):
                if "tool_name" not in fields_set and getattr(rule, "tools", None):
                    hdr_tools: tuple[str, ...] = tuple(rule.tools)
                else:
                    hdr_tools = (primary_tool or "Bash",)
                hdr_prefix = "[proposed_tool_call:" if event.kind == "proposed_tool_call" else "[tool_call:"
                for h_tool in hdr_tools:
                    hdr_ev = AuditEvent(
                        event.chunk_id,
                        event.kind,
                        f"{hdr_prefix}{h_tool}]\n{event.text}",
                        {**event.meta, "tool_name": h_tool},
                    )
                    t_h = time.perf_counter()
                    hdr_matched, hdr_detail = evaluate_rule_on_event(rule, hdr_ev)
                    elapsed_ms = max(elapsed_ms, round((time.perf_counter() - t_h) * 1000.0, 3))
                    if hdr_detail == "not_eligible" or hdr_matched != matched or hdr_detail != detail:
                        provider_mismatch = (
                            f"header stability mismatch: {primary_tool_label}={'match' if matched else 'no_match'} ({detail}) "
                            f"vs {hdr_prefix}{h_tool}]={'match' if hdr_matched else 'no_match'} ({hdr_detail})"
                        )
                        break

            # 3) When case does not pin role and rule.roles has multiple roles, verify both match boolean AND eligibility/detail across rule.roles.
            if (
                provider_mismatch is None
                and "role" not in fields_set
                and getattr(rule, "roles", None)
                and len(rule.roles) > 1
            ):
                primary_role_label = event.meta.get("role") or "default"
                for alt_role in rule.roles:
                    alt_ev = render_case_event(
                        case, rule=rule, index=index, provider=effective_provider, override_role=alt_role
                    )
                    t_r = time.perf_counter()
                    alt_matched, alt_detail = evaluate_rule_on_event(rule, alt_ev)
                    elapsed_ms = max(elapsed_ms, round((time.perf_counter() - t_r) * 1000.0, 3))
                    if alt_detail == "not_eligible" or alt_matched != matched or alt_detail != detail:
                        provider_mismatch = (
                            f"role mismatch across rule.roles: {primary_role_label}={'match' if matched else 'no_match'} ({detail}) "
                            f"vs {alt_role}={'match' if alt_matched else 'no_match'} ({alt_detail})"
                        )
                        break
    except ContractRegexTimeout:
        timed_out = True
        detail = "regex_timeout"
        elapsed_ms = max(elapsed_ms, round((time.perf_counter() - started) * 1000.0, 3))
    except Exception as exc:
        error = str(exc)
        detail = f"eval_error: {exc}"
        elapsed_ms = max(elapsed_ms, round((time.perf_counter() - started) * 1000.0, 3))

    expected = case.expected_match
    ineligible = detail == "not_eligible"
    incomplete = bool(timed_out or (error is not None))
    expect_ok = (
        (not timed_out)
        and (error is None)
        and (not ineligible)
        and (matched == expected)
        and (provider_mismatch is None)
    )
    latency_ok = (not timed_out) and (limit_ms is None or elapsed_ms <= limit_ms)
    passed = expect_ok and latency_ok

    failure_reasons: list[str] = []
    if timed_out:
        failure_reasons.append("regex_timeout")
    elif error is not None:
        failure_reasons.append(f"regex_error: {error}")
    elif ineligible:
        failure_reasons.append("event is not_eligible for rule scope/roles/tools")
    elif matched != expected:
        failure_reasons.append(
            f"expected {'match' if expected else 'no_match'}, got {'match' if matched else 'no_match'} ({detail})"
        )
    if provider_mismatch is not None:
        failure_reasons.append(provider_mismatch)
    if not latency_ok and not timed_out:
        failure_reasons.append(f"elapsed_ms {elapsed_ms} exceeded max_ms {limit_ms}")

    return {
        "id": event.chunk_id,
        "index": index,
        "rendered_text": event.text,
        "kind": event.kind,
        "tool_name": event.meta.get("tool_name"),
        "expected": "match" if expected else "no_match",
        "expected_match": expected,
        "actual": "match" if matched else "no_match",
        "matched": matched,
        "detail": detail,
        "elapsed_ms": elapsed_ms,
        "max_ms": limit_ms,
        "timed_out": timed_out,
        "incomplete": incomplete,
        "passed": passed,
        "failure_reason": "; ".join(failure_reasons) if failure_reasons else None,
    }


def evaluate_contract_cases(
    rule: ContractRule,
    cases: list[ContractCase],
    *,
    baseline_rule: ContractRule | None = None,
    default_max_ms: float | None = None,
    provider: str = "claude",
) -> dict[str, Any]:
    """Evaluate candidate rule (and optional baseline_rule) across cases."""
    if len(cases) > MAX_CONTRACT_CASES:
        raise ValueError(f"evaluate_contract_cases supports at most {MAX_CONTRACT_CASES} cases (got {len(cases)})")
    check_rule_patterns(rule)
    if baseline_rule is not None:
        check_rule_patterns(baseline_rule)

    effective_provider = (rule.provider if rule.provider else provider) or "claude"
    results: list[dict[str, Any]] = []
    differences: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    baseline_passed_count = 0
    cand_suite_elapsed_ms = 0.0
    base_suite_elapsed_ms = 0.0

    for idx, case in enumerate(cases):
        rem_suite = MAX_SUITE_ELAPSED_MS - cand_suite_elapsed_ms
        t_cand0 = time.perf_counter()
        item = _eval_single_case(
            rule,
            case,
            idx,
            default_max_ms=default_max_ms,
            provider=effective_provider,
            remaining_suite_ms=rem_suite,
        )
        cand_suite_elapsed_ms += (time.perf_counter() - t_cand0) * 1000.0
        if baseline_rule is not None:
            base_prov = (baseline_rule.provider if baseline_rule.provider else provider) or "claude"
            rem_suite_base = MAX_SUITE_ELAPSED_MS - base_suite_elapsed_ms
            t_base0 = time.perf_counter()
            base_item = _eval_single_case(
                baseline_rule,
                case,
                idx,
                default_max_ms=default_max_ms,
                provider=base_prov,
                remaining_suite_ms=rem_suite_base,
            )
            base_suite_elapsed_ms += (time.perf_counter() - t_base0) * 1000.0
            base_incomplete = bool(
                base_item.get("incomplete")
                or base_item.get("timed_out")
                or str(base_item.get("detail", "")).startswith(
                    ("render_error", "eval_error", "regex_timeout", "suite_timeout")
                )
            )
            item["baseline_matched"] = base_item["matched"]
            item["baseline_actual"] = base_item["actual"]
            item["baseline_passed"] = base_item["passed"]
            item["baseline_elapsed_ms"] = base_item["elapsed_ms"]
            item["baseline_incomplete"] = base_incomplete
            if base_item["passed"]:
                baseline_passed_count += 1
            if base_incomplete or base_item["matched"] != item["matched"] or base_item["passed"] != item["passed"]:
                diff_entry = {
                    "id": item["id"],
                    "index": idx,
                    "rendered_text": item["rendered_text"],
                    "expected": item["expected"],
                    "baseline_actual": base_item["actual"],
                    "candidate_actual": item["actual"],
                    "baseline_passed": base_item["passed"],
                    "candidate_passed": item["passed"],
                    "baseline_incomplete": base_incomplete,
                    "baseline_failure_reason": base_item.get("failure_reason"),
                }
                differences.append(diff_entry)
                # Fail closed on baseline non-completion (timeout/error) OR when a previously passing baseline case fails on candidate.
                if base_incomplete or (bool(base_item["passed"]) and not bool(item["passed"])):
                    regressions.append(diff_entry)
        results.append(item)

    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    failed_count = total_count - passed_count
    max_elapsed_ms = max((r["elapsed_ms"] for r in results), default=0.0)
    ok = (total_count > 0) and (failed_count == 0) and (len(regressions) == 0)

    report: dict[str, Any] = {
        "ok": ok,
        "contract_id": rule.id,
        "severity": rule.severity,
        "provider": effective_provider,
        "passed": passed_count,
        "failed": failed_count,
        "total": total_count,
        "max_elapsed_ms": max_elapsed_ms,
        "results": results,
        "differences": differences,
        "regressions": regressions,
    }
    if baseline_rule is not None:
        report["baseline"] = {
            "contract_id": baseline_rule.id,
            "revision": baseline_rule.revision,
            "severity": baseline_rule.severity,
            "passed": baseline_passed_count,
            "failed": total_count - baseline_passed_count,
            "total": total_count,
        }
    else:
        report["baseline"] = None
    return report
