"""Grok: a call a hook blocked is a rejected proposal, not an executed violation.

The recordings (``tests/fixtures/grok/refused_updates_*.jsonl``) are three live Grok CLI 1.0.52 runs in which a
``PreToolUse`` hook denied calls (sanitized: placeholder paths and session ids). What they show, and what these tests
pin:

* a denial takes four consecutive event numbers (the call n, the hook's report n + 1, the failed update n + 2, the
  host's note n + 3), but the FILE shows the report and the note before the call and its update;
* only that note, written by the host, is evidence. The failed update's "Hook denied:" text is a hook's, a tool output
  or a message could print the same words, and a call that merely failed is still an attempt;
* the join is made by the audit (``audit.hostblocks``), not by the parser, so any split of the file gives the same verdict.

Everything else here is synthetic and follows the recorded shapes. No Grok binary is run.
"""
from __future__ import annotations

import json
import re
import shutil

import pytest

from self_directing_mcp import grok
from self_directing_mcp.audit import runner
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.grok_hooks import run_hook
from self_directing_mcp.index.ingest import current_parser_version
from grok_support import (
    FIXTURES, GROUP, OTHER, SID, agent, annotation, call, completed, hook_execution, hook_payload, rec, session_file,
    update, user,
)
from semantic_support import finding

RECORDINGS = {1: "01a00000-0000-7000-8000-000000000004", 2: "01a00000-0000-7000-8000-000000000005",
              3: "01a00000-0000-7000-8000-000000000006"}
SOURCE = "project/tsukkomi"


def contract(contract_id, regex, **extra):
    return {"id": contract_id, "type": "must_not", "scope": "tool_call", "provider": "grok", "regex": regex, **extra}


SECRET, RM, NOTES = contract("secret", r"secret\.env"), contract("rm", r"rm\s+-rf"), contract("notes", r"notes\.txt")
TOOLS = [contract("write", r"\[tool_call:Write\]"), contract("task", r"\[tool_call:Task\]"),
         contract("grep", r"\[tool_call:Grep\]"), contract("ls", r"\[tool_call:LS\]")]
RULES = {1: [SECRET, RM, NOTES], 2: TOOLS, 3: [SECRET, RM, NOTES]}


def recording(number):
    return FIXTURES / f"refused_updates_{number}.jsonl"


def lines_of(number, sid):
    """The recording's records, one string each, for another session id (it is in every record and event number)."""
    text = recording(number).read_text(encoding="utf-8").replace(RECORDINGS[number], sid)
    return text.splitlines(keepends=True)


def install(root, number, sid=None, upto=None):
    """The recording (or its first ``upto`` records) as a session file; returns its path."""
    sid = sid or RECORDINGS[number]
    path = root / GROUP / sid / "updates.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines_of(number, sid)[:upto]), encoding="utf-8", newline="\n")
    return path


def extend(path, text):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def parse(path, **kwargs):
    return grok.parse_session_chunks(path, **kwargs)[0]


@pytest.fixture
def root(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    return sessions


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("SELF_DIRECT_GROK_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER", "GROK_HOME", "OPENROUTER_API_KEY",
                 "SELF_DIRECT_OPENROUTER_API_KEY_FILE", "SELF_DIRECT_LOCAL_ONLY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def engine(tmp_path, root):
    instance = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=tmp_path / "index",
                                         grok_sessions_dir=root, openrouter_api_key_file=None,
                                         audit_incremental_min_events=0))
    yield instance
    instance.close()


@pytest.fixture
def twin(tmp_path, root):
    """The engine that audits incrementally from stored state, and one that evaluates every audit in full."""
    base = dict(_env_file=None, local_only=True, openrouter_api_key_file=None, grok_sessions_dir=root,
                audit_incremental_min_events=0)
    fast = SelfDirectEngine(Settings(index_dir=tmp_path / "fast", audit_incremental=True, audit_shadow_every=0, **base))
    full = SelfDirectEngine(Settings(index_dir=tmp_path / "full", audit_incremental=False, **base))
    yield fast, full
    fast.close()
    full.close()


def audit(engine, path, sid):
    engine.sync_session(session_id=sid, provider="grok", path=str(path), embed=False)
    return engine.audit_session(sid, provider="grok", path=str(path))


def rejected(item):
    return [(entry["call_id"], entry["denied_by"]) for entry in item.get("proposals_rejected", [])]


def assert_attempt(item):
    assert item["verdict"] == "violation" and "proposals_rejected" not in item


def stop_note(engine, path, sid):
    reply = run_hook("Stop", hook_payload("Stop", path, sid=sid, lastAssistantMessage=None), engine=engine)
    return reply["systemMessage"]


# ---------------------------------------------------------------- the recordings themselves

def test_the_recordings_hold_only_placeholders():
    """Sanitized: every absolute path, id and name that is left is one of the neutral placeholders."""
    paths = re.compile(r"(?<![\w.])/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*")
    uuids = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
    for number, sid in RECORDINGS.items():
        text = recording(number).read_text(encoding="utf-8")
        assert sid in text and "@" not in text and not re.search(r"/(?:home|Users|mnt|root)/", text)
        assert all(path.startswith(("/work/proj", "/tmp/smoke-")) for path in paths.findall(text)), number
        assert all(found == sid or found.startswith("00000000-0000-4000-8000-") for found in uuids.findall(text)), number
        assert all(json.loads(line) for line in text.splitlines())


@pytest.mark.parametrize("number", sorted(RECORDINGS))
def test_the_file_shows_the_hosts_note_before_the_call_it_blocks(root, number):
    """The finding the design rests on: event numbers run call, report, update, note; the file does not."""
    chunks = parse(install(root, number))
    notes = [c for c in chunks if c.meta.get("evidence_origin") == "host_tool_outcome"]
    failed = {c.meta["host_event_id"]: c for c in chunks if c.kind == "tool_result" and c.meta["tool_status"] == "failed"}
    calls = {c.meta["call_id"]: c for c in chunks if c.kind == "tool_call"}
    assert notes and len(notes) == len(failed)
    for note in notes:
        result = failed[note.meta["host_event_id"] - 1]          # the host numbered its update just before its note
        owner = calls[result.meta["call_id"]]
        assert owner.meta["native_tool_name"] == note.meta["blocked_tool"]
        assert note.byte_start < owner.byte_start < result.byte_start     # ...yet the file shows the note first


@pytest.mark.parametrize("number", sorted(RECORDINGS))
def test_the_note_event_carries_no_text_a_hook_wrote(root, number):
    chunks = parse(install(root, number))
    for note in (c for c in chunks if c.meta.get("evidence_origin") == "host_tool_outcome"):
        assert note.kind == "meta" and note.meta["role"] == "runtime" and note.text == "[host_tool_outcome] blocked"
        assert note.meta["denied_by"] == SOURCE
        assert "forbidden" not in note.model_dump_json() and "matched" not in note.model_dump_json()
    assert not any(c.meta.get("unsupported_event") or c.meta.get("parse_error") for c in chunks)


# ---------------------------------------------------------------- the audit of the recordings, both evaluators

def test_recording_one_a_blocked_read_and_a_blocked_rm_are_rejected_proposals_and_the_allowed_read_is_not(engine, root):
    sid = RECORDINGS[1]
    engine.upsert_contracts(RULES[1])
    result = audit(engine, install(root, 1), sid)
    secret, rm, notes = (finding(result, name) for name in ("secret", "rm", "notes"))
    assert secret["verdict"] == "clean" and secret["reason"] == runner.rejected_clean(1)
    assert rm["verdict"] == "clean" and rm["reason"] == runner.rejected_clean(1)
    assert rejected(secret) == [("call-0000-1", SOURCE)] and rejected(rm) == [("call-0000-2", SOURCE)]
    # the read that ran is an executed match: the verdict of the whole audit is a violation of that rule alone
    assert notes["verdict"] == "violation" and "proposals_rejected" not in notes and result["verdict"] == "violation"
    chunks = {c.chunk_id: c for c in engine.store.list_chunks(sid, "grok")}
    for item, call_id, tool in ((secret, "call-0000-1", "read_file"), (rm, "call-0000-2", "run_terminal_command")):
        (entry,) = item["proposals_rejected"]
        owner, note = chunks[entry["event_id"]], chunks[entry["denial_event_id"]]
        assert owner.kind == "tool_call" and owner.meta["call_id"] == call_id and entry["at_byte"] == owner.byte_start
        assert note.meta["evidence_origin"] == "host_tool_outcome" and note.meta["blocked_tool"] == tool
        assert entry["reason"].startswith("host_denied:")


def test_recording_two_every_tool_blocked_gives_a_clean_audit_with_four_rejected_proposals(engine, root):
    engine.upsert_contracts(TOOLS)
    result = audit(engine, install(root, 2), RECORDINGS[2])
    assert result["verdict"] == "clean"
    for name, call_id in (("write", "call-0000-0"), ("task", "call-0000-1"), ("grep", "call-0000-2"),
                          ("ls", "call-0000-3")):
        item = finding(result, name)
        assert item["verdict"] == "clean" and item["reason"] == runner.rejected_clean(1)
        assert rejected(item) == [(call_id, SOURCE)]


def test_recording_three_parallel_calls_are_told_apart(engine, root):
    engine.upsert_contracts(RULES[3])
    result = audit(engine, install(root, 3), RECORDINGS[3])
    assert rejected(finding(result, "secret")) == [("call-0000-0", SOURCE)]
    assert rejected(finding(result, "rm")) == [("call-0000-2", SOURCE)]
    assert finding(result, "secret")["verdict"] == "clean" and finding(result, "rm")["verdict"] == "clean"
    notes = finding(result, "notes")                 # the allowed read, issued in parallel with the two blocked calls
    assert notes["verdict"] == "violation" and "proposals_rejected" not in notes


def test_the_first_probe_a_read_that_a_hook_blocked_with_another_source(engine, root):
    """``updates_session2.jsonl`` (the first live probe, another hook name): the same join, the same shape."""
    path = root / GROUP / SID / "updates.jsonl"
    path.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "updates_session2.jsonl", path)
    engine.upsert_contracts([SECRET, contract("notes", r"notes\.txt")])
    result = audit(engine, path, SID)
    assert rejected(finding(result, "secret")) == [("call-0000-1", "project/hook probe")]
    assert finding(result, "secret")["verdict"] == "clean"
    assert finding(result, "notes")["verdict"] == "violation" and "proposals_rejected" not in finding(result, "notes")


def test_the_recorded_cancelled_edit_is_still_an_attempt(engine, root):
    """A call the user cancelled is a failed result (``toolNotRun``) with no note: the recorded shape of a refusal by
    the user is not a hook's, so it is not read as one."""
    path = root / GROUP / OTHER / "updates.jsonl"
    path.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "updates_session3.jsonl", path)
    engine.upsert_contracts([contract("edits", r"\[tool_call:Edit\]")])
    assert_attempt(finding(audit(engine, path, OTHER), "edits"))


def test_a_rule_that_matches_both_a_blocked_and_an_executed_call_counts_only_the_executed_one(engine, root):
    engine.upsert_contracts([contract("any-read", r"\[tool_call:Read\]")])
    item = finding(audit(engine, install(root, 1), RECORDINGS[1]), "any-read")
    assert item["verdict"] == "violation" and item["reason"].startswith("forbidden regex matched 1 event(s)")
    assert rejected(item) == [("call-0000-1", SOURCE)]
    assert [e["snippet"] for e in item["evidence"]] == ['[tool_call:Read] {"file_path": "notes.txt"}']


@pytest.mark.parametrize("number,rules", [(1, [SECRET, RM]), (2, TOOLS), (3, [SECRET, RM])])
def test_the_stop_note_puts_rejected_proposals_apart_and_calls_nothing_violated(engine, root, number, rules):
    engine.upsert_contracts(rules)
    note = stop_note(engine, install(root, number), RECORDINGS[number])
    assert note.startswith("Rejected by the host, not executed: ")
    assert note.count("rejected proposals, not executed violations") == len(rules)
    assert "forbidden regex matched" not in note and "Not verified" not in note and "Contract checkpoint" not in note


def test_the_stop_note_keeps_an_executed_violation_beside_the_rejected_ones(engine, root):
    engine.upsert_contracts(RULES[1])
    note = stop_note(engine, install(root, 1), RECORDINGS[1])
    assert "[notes] forbidden regex matched 1 event(s)" in note
    assert "Rejected by the host, not executed: [secret]" in note and "[rm] 1 proposed call(s)" in note
    assert "[secret] forbidden" not in note and "[rm] forbidden" not in note


def test_without_the_join_a_blocked_call_is_an_executed_match_again(engine, root, monkeypatch):
    """The gap this closes, shown by switching the join off for Grok."""
    monkeypatch.setattr(runner, "REFUSAL_PROVIDERS", ("claude",))
    engine.upsert_contracts(TOOLS)
    note = stop_note(engine, install(root, 2), RECORDINGS[2])
    assert "forbidden regex matched" in note and "Rejected by the host" not in note


# ---------------------------------------------------------------- any split of the file gives the same verdict

def signature(result):
    """What an audit says about the calls, without the ids that name the session."""
    return [(item["contract_id"], item["verdict"], item["reason"],
             [(e["call_id"], e["denied_by"], e["at_byte"], e["reason"]) for e in item.get("proposals_rejected", [])],
             [e["snippet"] for e in item["evidence"]]) for item in result["findings"]], result["verdict"]


@pytest.mark.parametrize("number", sorted(RECORDINGS))
def test_parsing_in_two_pieces_equals_parsing_the_whole_at_every_record_boundary(root, number):
    sid = RECORDINGS[number]
    whole = parse(install(root, number))
    lines = lines_of(number, sid)
    for cut in range(1, len(lines)):
        path = install(root, number, upto=cut)
        first, offset, _ = grok.parse_session_chunks(path)
        extend(path, "".join(lines[cut:]))
        second, end, _ = grok.parse_session_chunks(path, start_byte=offset)
        assert offset == sum(len(line.encode()) for line in lines[:cut]) and end == path.stat().st_size
        assert [c.chunk_id for c in first + second] == [c.chunk_id for c in whole], cut
        assert [c.meta for c in first + second] == [c.meta for c in whole], cut


@pytest.mark.parametrize("number", sorted(RECORDINGS))
def test_the_audit_is_the_same_whether_the_file_is_read_whole_or_in_two_goes_at_any_boundary(twin, root, number):
    """Sync the first part, audit (the monitors and the facts are stored), sync the rest, audit from that state: the
    answer must be the one the full evaluator gives for the whole file. The join waits in the facts for its other half."""
    fast, full = twin
    for engine in (fast, full):
        engine.upsert_contracts(RULES[number])
    once = "01a00000-0000-7000-8000-000000000098"
    path = install(root, number, sid=once)
    want = signature(audit(full, path, once))
    assert signature(audit(fast, path, once)) == want
    lines = lines_of(number, RECORDINGS[number])
    for cut in range(1, len(lines)):
        sid = f"01a00000-0000-7000-8000-{cut:012d}"
        path = install(root, number, sid=sid, upto=cut)
        audit(fast, path, sid)
        extend(path, "".join(lines_of(number, sid)[cut:]))
        got = audit(fast, path, sid)
        assert got["incremental"]["mode"] == "delta", cut                      # from the state stored at the cut
        assert signature(got) == want, f"recording {number}, split after record {cut}"
        assert signature(audit(full, path, sid)) == want, cut


def test_a_read_that_ends_between_the_note_and_the_call_leaves_nothing_unread(engine, root):
    """The note, the call and its update are separate records: no cursor has to wait for the other half."""
    for cut in (6, 7, 8):               # records 5 and 6: the hook's report and the note; 7 and 8: the call and its update
        sid = f"01a00000-0000-7000-8000-{cut:012d}"
        path = install(root, 2, sid=sid, upto=cut)
        info = engine.sync_session(session_id=sid, provider="grok", path=str(path), embed=False)
        assert info["byte_offset"] == path.stat().st_size and info["coverage"]["complete"] is True
        assert "sync_in_progress" not in info["coverage"]["issues"] and "backlog_bytes" not in info["coverage"]
        assert info["new_chunks"] == len(parse(path))


def test_an_audit_between_the_note_and_its_call_waits_then_follows(engine, root):
    """Incremental state: the note is read first, the call and its update later; the verdict follows each read."""
    sid = RECORDINGS[2]
    lines = lines_of(2, sid)
    engine.upsert_contracts([TOOLS[0]])
    path = install(root, 2, upto=6)                                        # ... the hook's report and the note
    before = finding(audit(engine, path, sid), "write")
    assert before["verdict"] == "clean" and "proposals_rejected" not in before
    extend(path, lines[6])                                                 # the call: no result yet
    assert finding(audit(engine, path, sid), "write")["verdict"] == "violation"      # an attempt, as ever
    extend(path, lines[7])                                                 # its failed update
    item = finding(audit(engine, path, sid), "write")
    assert item["verdict"] == "clean" and rejected(item) == [("call-0000-0", SOURCE)]


# ---------------------------------------------------------------- what is never evidence

def eid(number, sid=SID):
    return {"eventId": f"{sid}-{number}"}


def result(call_id, number, text="Hook denied: x", status="failed", **kwargs):
    return update(call_id, status, text, envelope=eid(number), **kwargs)


def note(number, tool="read_file", source=SOURCE, reason="why", **kwargs):
    return annotation(f"`{tool}` blocked by {source}: {reason}", envelope=eid(number), **kwargs)


def raw_note(message, number=13, kind="tool_outcome", **kwargs):
    body = {"sessionUpdate": "hook_annotation", "message": message}
    if kind is not None:
        body["kind"] = kind
    return rec(body, method="_x.ai/session/update", prompt=None, envelope=eid(number), **kwargs)


def read(call_id, path="secret.env"):
    return call(call_id, "read_file", {"target_file": path})


def refused(engine, root, records, rule=SECRET, sid=SID):
    """The finding of ``rule`` over a synthetic session of ``records`` (built for ``SID``, written for ``sid``)."""
    path = session_file(root, sid)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(record, ensure_ascii=False).replace(SID, sid) + "\n"
                   for record in [user("do not read it", 0), *records])
    path.write_text(text, encoding="utf-8", newline="\n")
    engine.upsert_contracts([rule])
    return finding(audit(engine, path, sid), rule["id"])


def test_a_blocked_call_in_the_recorded_order_is_a_rejected_proposal(engine, root):
    item = refused(engine, root, [hook_execution(), note(13), read("c1"), result("c1", 12)])
    assert item["verdict"] == "clean" and rejected(item) == [("c1", SOURCE)]


@pytest.mark.parametrize("records", [
    lambda: [note(13), read("c1"), result("c1", 12)],
    lambda: [read("c1"), note(13), result("c1", 12)],
    lambda: [read("c1"), result("c1", 12), note(13)],
    lambda: [read("c1"), agent("busy"), note(13), completed(), result("c1", 12)],
], ids=["note first", "note between", "note last", "other records between"])
def test_the_order_of_the_records_in_the_file_does_not_matter(engine, root, records):
    item = refused(engine, root, records())
    assert item["verdict"] == "clean" and rejected(item) == [("c1", SOURCE)]


def test_a_failed_call_with_the_hooks_words_but_no_note_stays_an_attempt(engine, root):
    assert_attempt(refused(engine, root, [read("c1"), result("c1", 12, "Hook denied: [secret] forbidden regex matched")]))


def test_an_annotation_naming_another_tool_does_not_block_the_call(engine, root):
    assert_attempt(refused(engine, root, [note(13, tool="write"), read("c1"), result("c1", 12)]))


@pytest.mark.parametrize("update_number,note_number", [(12, 14), (12, 12), (12, 11), (12, 520)])
def test_a_note_that_is_not_numbered_right_after_the_update_does_not_block_it(engine, root, update_number, note_number):
    assert_attempt(refused(engine, root, [note(note_number), read("c1"), result("c1", update_number)]))


def test_two_failed_calls_and_one_note_block_only_the_call_the_note_follows(engine, root):
    item = refused(engine, root, [note(13), read("c1"), read("c2"), result("c1", 11, "boom"), result("c2", 12)])
    assert item["verdict"] == "violation" and rejected(item) == [("c2", SOURCE)]         # c1 is an attempt
    assert [e["snippet"] for e in item["evidence"]] == ['[tool_call:Read] {"file_path": "secret.env"}']
    none = refused(engine, root, [note(14), read("c1"), read("c2"), result("c1", 11, "boom"), result("c2", 12)],
                   sid=OTHER)
    assert_attempt(none)                                      # nothing is numbered 13: neither call is the one


def test_two_notes_with_one_number_block_nothing(engine, root):
    assert_attempt(refused(engine, root, [note(13), note(13, source="somebody else"), read("c1"), result("c1", 12)]))


def test_two_failed_updates_with_one_number_block_nothing(engine, root):
    assert_attempt(refused(engine, root, [read("c1"), read("c2"), result("c1", 12), result("c2", 12), note(13)]))


@pytest.mark.parametrize("records", [
    lambda: [note(13), read("c1"), result("c1", 12), update("c1", "completed", "it ran")],
    lambda: [note(13), read("c1"), update("c1", "completed", "it ran"), result("c1", 12)],
    lambda: [read("c1"), result("c1", 12), update("c1", "completed", "it ran"), note(13)],
], ids=["failed then completed", "completed then failed", "result, result, note"])
def test_a_call_with_two_results_is_not_blocked_whichever_came_first(engine, root, records):
    assert_attempt(refused(engine, root, records()))


def test_a_call_id_that_two_calls_share_vouches_for_neither(engine, root):
    assert_attempt(refused(engine, root, [note(13), read("c1"), read("c1", "other.txt"), result("c1", 12)]))


def test_a_call_that_ran_is_not_blocked_by_a_note_numbered_after_its_result(engine, root):
    assert_attempt(refused(engine, root, [note(13), read("c1"), update("c1", "completed", "x", envelope=eid(12))]))


def test_a_cancelled_call_is_an_attempt(engine, root):
    cancelled = update("c1", "failed", "User cancelled", envelope=eid(12), extra={"_meta": {"toolNotRun": "cancelled"}})
    assert_attempt(refused(engine, root, [read("c1"), cancelled]))


@pytest.mark.parametrize("record", [
    raw_note("`read_file` blocked by project/tsukkomi: x", kind=None),                  # no kind
    raw_note("`read_file` blocked by project/tsukkomi: x", kind="hook_failure"),        # another kind
    raw_note("`read_file` blocked: x"),                                                 # another wording
    raw_note("`read_file` was blocked by project/tsukkomi: x"),
    raw_note("read_file blocked by project/tsukkomi: x"),                               # not backtick-quoted
    raw_note("Note: `read_file` blocked by project/tsukkomi: x"),                       # not the start of the message
    raw_note(" `read_file` blocked by project/tsukkomi: x"),
    raw_note("`read_file` blocked by "),                                                # no source
    raw_note("`read file` blocked by project/tsukkomi: x"),                             # not a tool name
    raw_note("`read_file` blocked by project\nx: y"),                                   # the source spans lines
    raw_note(["`read_file` blocked by project/tsukkomi: x"]),
    raw_note(None),
    rec({"sessionUpdate": "hook_annotation", "message": "`read_file` blocked by project/tsukkomi: x",
         "kind": "tool_outcome"}, envelope=eid(13)),                                    # a session/update: the wrong method
    rec({"sessionUpdate": "hook_annotation", "message": "`read_file` blocked by project/tsukkomi: x",
         "kind": "tool_outcome"}, method="_x.ai/session/update", prompt=None),          # no event number
    rec({"sessionUpdate": "hook_annotation", "message": "`read_file` blocked by project/tsukkomi: x",
         "kind": "tool_outcome"}, method="_x.ai/session/update", prompt=None, envelope={"eventId": f"{SID}-x13"}),
    rec({"sessionUpdate": "hook_annotation", "message": "`read_file` blocked by project/tsukkomi: x",
         "kind": "tool_outcome"}, method="_x.ai/session/update", prompt=None, envelope=eid(13, OTHER)),
    rec({"sessionUpdate": "hook_annotation", "message": "`read_file` blocked by project/tsukkomi: x",
         "kind": "tool_outcome"}, method="_x.ai/session/update", prompt=None, sid=OTHER, envelope=eid(13, OTHER)),
], ids=["no-kind", "other-kind", "other-wording", "was-blocked", "no-backticks", "not-first", "leading-space",
        "no-source", "not-a-name", "source-spans-lines", "list-message", "no-message", "wrong-method", "no-number",
        "number-not-digits", "number-of-another-session", "record-of-another-session"])
def test_only_the_hosts_own_wording_in_its_own_record_is_a_note(engine, root, record):
    assert_attempt(refused(engine, root, [record, read("c1"), result("c1", 12)]))
    assert not [c for c in engine.store.list_chunks(SID, "grok") if c.meta.get("evidence_origin") == "host_tool_outcome"]


@pytest.mark.parametrize("where", ["tool output", "agent message", "user message", "thought", "failed update"])
def test_the_hosts_words_printed_by_something_else_are_not_a_note(engine, root, where):
    words = "`read_file` blocked by project/tsukkomi: x"
    carriers = {
        "tool output": [read("c0", "notes.txt"), update("c0", "completed", words, envelope=eid(9))],
        "agent message": [agent(words)],
        "user message": [user(words, 1)],
        "thought": [rec({"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": words}})],
        "failed update": [],
    }[where]
    failure = words if where == "failed update" else "Hook denied: x"
    assert_attempt(refused(engine, root, [*carriers, read("c1"), result("c1", 12, failure)]))
    assert not [c for c in engine.store.list_chunks(SID, "grok") if c.meta.get("evidence_origin") == "host_tool_outcome"]


def test_a_failed_result_numbered_by_another_session_has_no_number(root):
    foreign = update("c1", "failed", "x", envelope=eid(12, OTHER))
    path = session_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in (read("c1"), foreign)), encoding="utf-8", newline="\n")
    (failed,) = [c for c in parse(path) if c.kind == "tool_result"]
    assert "host_event_id" not in failed.meta and failed.meta["success"] is False


def test_a_note_too_far_from_the_update_it_names_is_forgotten(engine, root):
    """A number is Grok's counter of one process: a stale note must not meet a later, unrelated failure."""
    filler = [completed(prompt=f"p{n}") if n % 2 else agent(f"turn {n}") for n in range(600)]
    assert_attempt(refused(engine, root, [note(13), *filler, read("c1"), result("c1", 12)]))


def test_a_blocked_and_an_allowed_call_of_one_tool_are_not_confused(engine, root):
    item = refused(engine, root, [note(13), read("c1"), read("c2"), result("c1", 12),
                                  update("c2", "completed", "TOKEN", envelope=eid(14))])
    assert item["verdict"] == "violation" and rejected(item) == [("c1", SOURCE)]
    assert [e["snippet"] for e in item["evidence"]] == ['[tool_call:Read] {"file_path": "secret.env"}']


def test_a_note_in_the_middle_of_a_streamed_message_does_not_split_it(root):
    path = session_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [agent("rm -"), hook_execution(), note(13), agent("rf /"), read("c1"), result("c1", 12)]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8", newline="\n")
    chunks = parse(path)
    assert [c.text for c in chunks if c.kind == "turn"] == ["[message:assistant] rm -rf /"]
    assert [(c.kind, c.meta.get("evidence_origin")) for c in chunks] == [
        ("turn", None), ("meta", "host_tool_outcome"), ("tool_call", None), ("tool_result", None)]
    assert [c.byte_start for c in chunks] == sorted(c.byte_start for c in chunks)


# ---------------------------------------------------------------- not read for other hosts, and the version

def test_a_claude_shaped_decision_on_a_grok_result_is_nothing(engine, root):
    refusal = update("c1", "failed", "blocked", extra={"permissionDecision": {"decision": "reject", "source": "hook"}})
    assert_attempt(refused(engine, root, [read("c1"), refusal]))


def test_the_parser_version_was_raised_and_a_session_indexed_before_is_reparsed(engine, root, monkeypatch):
    sid = RECORDINGS[1]
    assert grok.PARSER_VERSION >= 2 and current_parser_version("grok") == grok.PARSER_VERSION
    path = install(root, 1)
    engine.upsert_contracts([SECRET])
    with monkeypatch.context() as old:
        # What parser v1 produced: the host's note was skipped, so the blocked read was an executed match.
        old.setattr(grok, "PARSER_VERSION", 1)
        old.setattr(grok, "_annotation", lambda base, update_, params, state: iter(()))
        first = engine.sync_session(session_id=sid, provider="grok", path=str(path), embed=False)
        assert first["parser_version"] == 1
        assert_attempt(finding(engine.audit_session(sid, provider="grok", path=str(path)), "secret"))
    second = engine.sync_session(session_id=sid, provider="grok", path=str(path), embed=False)
    assert second["reindexed"] and second["reindex_reason"] == "parser_updated"
    assert second["parser_version"] == grok.PARSER_VERSION
    fresh = finding(engine.audit_session(sid, provider="grok", path=str(path)), "secret")
    assert fresh["verdict"] == "clean" and rejected(fresh) == [("call-0000-1", SOURCE)]
