"""The warm judgement path: what is cached or skipped, and the proof that a change is seen on the very next request.

Every cache here is keyed by the exact content it was computed from (never by a timestamp, size or version that a
change could leave alone), and every skipped piece of work is one whose result cannot reach the answer. Each test
changes something between two requests and asserts that the second request sees it. Synthetic data only.
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time

import pytest

from self_directing_mcp import agy_receipts
from self_directing_mcp.agy_receipts import ReceiptStore
from self_directing_mcp.audit import contracts as contracts_module
from self_directing_mcp.audit import runner
from self_directing_mcp.audit.contracts import ContentMemo, parse_contracts
from self_directing_mcp.checklist import ChecklistDocument, RequirementItem
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index.deferred import DerivedBuffer
from self_directing_mcp.index.ingest import SessionStore
from self_directing_mcp.request_control import RequestControl, RequestStopped, check_request, request_scope
from self_directing_mcp.schemas import ContractsDocument
from self_directing_mcp.session_events import has_invalid_utf8
from incremental_support import Session, contracts as random_contracts, proposed_action
from test_claude_hooks import SID as CLAUDE_SID, assistant, human, main_file, run, text, tool_result, use, write
from test_reliability import SID, call, message, result
from test_reliability import write as write_codex


# --- parse_contracts: the key is the content ----------------------------------------------------------------------

RULE = {"id": "r1", "type": "must_not", "scope": "tool_call", "regex": "forbidden", "provider": "claude"}


def document(*rules):
    return json.dumps({"contracts": list(rules)}).encode()


def test_equal_bytes_share_one_validated_document_and_changed_bytes_are_parsed_again():
    first = parse_contracts(document(RULE))
    assert parse_contracts(document(RULE)) is first                      # same bytes: no second validation
    changed = parse_contracts(document({**RULE, "regex": "forbidden|other"}))
    assert changed is not first and changed.contracts[0].regex == "forbidden|other"
    assert parse_contracts(document(RULE)) is first or parse_contracts(document(RULE)).contracts[0].regex == "forbidden"


def test_a_one_byte_edit_of_the_same_length_is_seen():
    """The classic stale-cache bug keys on size and modification time; this one keys on the bytes."""
    raw = document(RULE)
    edited = raw.replace(b"forbidden", b"forbiddeX")
    assert len(raw) == len(edited)
    assert parse_contracts(raw).contracts[0].regex == "forbidden"
    assert parse_contracts(edited).contracts[0].regex == "forbiddeX"
    assert parse_contracts(raw).contracts[0].regex == "forbidden"


def test_invalid_bytes_raise_every_time_and_are_never_remembered():
    broken = b'{"contracts": [{"id": "x"}]}'
    for _ in range(3):
        with pytest.raises(ValueError):
            parse_contracts(broken)
    assert parse_contracts(None).contracts == []
    assert parse_contracts(document(RULE)).contracts[0].id == "r1"


def test_a_contracts_file_over_the_limit_is_validated_every_time(monkeypatch):
    monkeypatch.setattr(contracts_module, "MEMO_MAX_BYTES", 10)
    raw = document(RULE)
    assert parse_contracts(raw) is not parse_contracts(raw)


def test_the_memo_evicts_the_least_recently_used_entry():
    memo = ContentMemo(2)
    memo.put("a", 1)
    memo.put("b", 2)
    assert memo.get("a") == 1                                            # touching a makes b the oldest
    memo.put("c", 3)
    assert memo.get("b") is None and memo.get("a") == 1 and memo.get("c") == 3


def test_an_audit_never_edits_the_shared_document(tmp_path, monkeypatch):
    engine, path = claude_engine(tmp_path, monkeypatch)
    try:
        engine.upsert_contracts([RULE, {"id": "r2", "type": "must", "scope": "tool_call", "regex": "pytest",
                                        "before_regex": "deploy", "provider": "claude"}])
        raw = engine.settings.resolve_contracts_path().read_bytes()
        before = parse_contracts(raw).model_dump(mode="json")
        engine.audit_session(CLAUDE_SID, provider="claude", path=str(path))
        engine.check_action(CLAUDE_SID, {"tool_name": "Bash", "arguments": {"command": "deploy"}}, provider="claude",
                            path=str(path), verify="fast")
        engine.hook_obligations(CLAUDE_SID, provider="claude", path=str(path))
        assert parse_contracts(raw).model_dump(mode="json") == before == \
            ContractsDocument.model_validate_json(raw).model_dump(mode="json")
    finally:
        engine.close()


def claude_engine(tmp_path, monkeypatch, **options):
    projects, index = tmp_path / "projects", tmp_path / "index"
    index.mkdir(parents=True, exist_ok=True)
    path = write(main_file(projects), human("hi"), assistant(text("hello")))
    for name in ("SELF_DIRECT_SESSION_PROVIDER", "SELF_DIRECT_CONTRACTS_PATH"):
        monkeypatch.delenv(name, raising=False)
    engine = SelfDirectEngine(Settings(_env_file=None, local_only=True, index_dir=index, claude_projects_dir=projects,
                                       openrouter_api_key_file=None, **options))
    return engine, path


# --- hook_obligations: a function of four values ------------------------------------------------------------------

@pytest.fixture
def claude(tmp_path, monkeypatch):
    engine, path = claude_engine(tmp_path, monkeypatch)
    yield engine, path
    engine.close()


def obligations(engine, path, provider="claude", sid=CLAUDE_SID):
    return engine.hook_obligations(sid, provider=provider, path=str(path))


def test_a_changed_contracts_file_is_seen_by_the_next_hook(claude):
    engine, path = claude
    contracts_path = engine.settings.resolve_contracts_path()
    contracts_path.write_bytes(document(RULE))
    assert [r.id for r in obligations(engine, path)["contracts"]] == ["r1"]
    contracts_path.write_bytes(document(RULE, {**RULE, "id": "r2"}))
    assert [r.id for r in obligations(engine, path)["contracts"]] == ["r1", "r2"]
    contracts_path.write_bytes(document({**RULE, "enabled": False}, {**RULE, "id": "r2"}))
    assert [r.id for r in obligations(engine, path)["contracts"]] == ["r2"]


def test_an_edit_that_keeps_size_and_modification_time_is_still_seen(claude):
    engine, path = claude
    contracts_path = engine.settings.resolve_contracts_path()
    contracts_path.write_bytes(document(RULE))
    stamp = contracts_path.stat()
    first = obligations(engine, path)
    contracts_path.write_bytes(document(RULE).replace(b"forbidden", b"forbiddeX"))
    os.utime(contracts_path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert contracts_path.stat().st_size == stamp.st_size and contracts_path.stat().st_mtime_ns == stamp.st_mtime_ns
    second = obligations(engine, path)
    assert first["contracts"][0].regex == "forbidden" and second["contracts"][0].regex == "forbiddeX"
    assert first["snapshot"] != second["snapshot"]


def test_a_changed_checklist_is_seen_by_the_next_hook(claude):
    engine, path = claude
    engine.settings.resolve_contracts_path().write_bytes(document(RULE))
    engine.ensure_ready()
    assert obligations(engine, path)["pending"] == []
    item = RequirementItem(item_id="req-001", text="write the report", status="open")
    engine.checklists.save(ChecklistDocument(session_id=CLAUDE_SID, items=[item]))
    after = obligations(engine, path)
    assert [i.text for i in after["pending"]] == ["write the report"] and after["has_checklist"] is True
    engine.checklists.save(ChecklistDocument(session_id=CLAUDE_SID, items=[item.model_copy(update={"status": "revoked"})]))
    assert obligations(engine, path)["pending"] == []


def test_another_session_or_provider_is_another_entry(claude):
    engine, path = claude
    engine.settings.resolve_contracts_path().write_bytes(document(
        RULE,
        {**RULE, "id": "mine", "session_id": CLAUDE_SID},
        {**RULE, "id": "theirs", "session_id": "other-session", "provider": None},
        {**RULE, "id": "codex-only", "provider": "codex"}))
    ids = lambda **kwargs: [r.id for r in obligations(engine, path, **kwargs)["contracts"]]   # noqa: E731
    assert ids() == ["r1", "mine"]
    assert ids(provider="codex", sid="other-session") == ["theirs", "codex-only"]
    assert ids(provider="claude", sid="other-session") == ["r1", "theirs"]
    assert ids() == ["r1", "mine"]                                       # and the first entry is unaffected


def test_a_caller_that_edits_what_it_got_does_not_change_the_next_answer(claude):
    engine, path = claude
    engine.settings.resolve_contracts_path().write_bytes(document(RULE))
    first = obligations(engine, path)
    first["contracts"].clear()
    first["pending"].append("not an item")
    first["status"] = "mutated"
    second = obligations(engine, path)
    assert [r.id for r in second["contracts"]] == ["r1"] and second["pending"] == [] and second["status"] == "applicable"


def test_an_unreadable_contracts_file_fails_every_time_and_recovers_when_fixed(claude):
    engine, path = claude
    contracts_path = engine.settings.resolve_contracts_path()
    contracts_path.write_bytes(document(RULE))
    obligations(engine, path)
    contracts_path.write_bytes(b"{ not json")
    for _ in range(2):
        with pytest.raises(ValueError):
            obligations(engine, path)
    contracts_path.write_bytes(document(RULE))
    assert [r.id for r in obligations(engine, path)["contracts"]] == ["r1"]


def test_the_answer_is_what_an_uncached_read_would_say(claude):
    engine, path = claude
    contracts_path = engine.settings.resolve_contracts_path()
    contracts_path.write_bytes(document(RULE, {**RULE, "id": "r2", "risk": "high"}))
    cached = obligations(engine, path)
    obligations(engine, path)
    again = obligations(engine, path)
    for key in ("status", "has_checklist", "snapshot"):
        assert cached[key] == again[key]
    expected = ContractsDocument.model_validate_json(contracts_path.read_bytes()).contracts
    assert [r.model_dump(mode="json") for r in again["contracts"]] == [r.model_dump(mode="json") for r in expected]


# --- the next request sees a contract edit and an appended event ---------------------------------------------------

def test_a_contract_edit_and_an_appended_event_are_seen_by_the_very_next_hook(tmp_path, monkeypatch):
    engine, path = claude_engine(tmp_path, monkeypatch)
    try:
        action = {"command": "run forbidden"}
        assert run(engine, path, "PreToolUse", tool_input=action) == {}                       # no contract yet
        engine.upsert_contracts([RULE])
        assert run(engine, path, "PreToolUse", tool_input=action)["hookSpecificOutput"]["permissionDecision"] == "deny"
        edited = {**RULE, "regex": "something-else"}
        engine.contracts.upsert([contract(edited)])                                            # a weaker revision
        assert run(engine, path, "PreToolUse", tool_input=action) == {}
        engine.upsert_contracts([{"id": "needs-tests", "type": "must", "scope": "tool_call", "regex": "pytest",
                                  "before_regex": "deploy", "provider": "claude"}])
        deploy = {"command": "deploy now"}
        before = run(engine, path, "PreToolUse", tool_input=deploy)
        assert "needs-tests" in json.dumps(before)                                             # unmet: not clean
        write(path, assistant(use("t1", "Bash", command="pytest -q")), tool_result("t1", "3 passed"), mode="a")
        assert run(engine, path, "PreToolUse", tool_input=deploy) == {}                        # the prerequisite appeared
    finally:
        engine.close()


def contract(data):
    from self_directing_mcp.schemas import ContractRule
    return ContractRule(**data)


# --- the checklist read skips what cannot matter ------------------------------------------------------------------

def test_a_checklist_without_a_done_item_needs_no_evidence(claude, monkeypatch):
    engine, path = claude
    engine.sync_session(session_id=CLAUDE_SID, provider="claude", path=str(path), embed=False)
    items = [RequirementItem(item_id="req-001", text="open one", status="open"),
             RequirementItem(item_id="req-002", text="blocked one", status="blocked", blocked_reason="waiting"),
             RequirementItem(item_id="req-003", text="in progress", status="in_progress")]
    engine.checklists.save(ChecklistDocument(session_id=CLAUDE_SID, items=items))

    def forbidden(*args, **kwargs):
        raise AssertionError("no evidence is needed for a checklist without a done item")

    monkeypatch.setattr(SessionStore, "capture_evidence", forbidden)
    monkeypatch.setattr("self_directing_mcp.engine.capture_for_cursor", forbidden)
    reply = engine.get_checklist(CLAUDE_SID, provider="claude")
    assert reply["ok"] and [(i["item_id"], i["status"]) for i in reply["items"]] == \
        [("req-001", "open"), ("req-002", "blocked"), ("req-003", "in_progress")]
    assert engine.get_checklist("a-session-without-a-checklist", provider="claude")["items"] == []


def test_a_done_item_is_still_checked_against_the_evidence(claude):
    engine, path = claude
    engine.sync_session(session_id=CLAUDE_SID, provider="claude", path=str(path), embed=False)
    claimed = RequirementItem(item_id="req-001", text="claimed", status="done", verified=True,
                              evidence_chunk_ids=["claude:no-such-event"])
    engine.checklists.save(ChecklistDocument(session_id=CLAUDE_SID, items=[
        claimed, RequirementItem(item_id="req-002", text="open", status="open")]))
    reply = engine.get_checklist(CLAUDE_SID, provider="claude")
    assert {i["item_id"]: (i["status"], i["verified"]) for i in reply["items"]} == \
        {"req-001": ("pending_verification", False), "req-002": ("open", False)}


# --- hooks refresh the canonical events, not the search index -----------------------------------------------------

def test_a_stop_hook_leaves_the_search_index_to_the_next_search_which_repairs_it(tmp_path, monkeypatch):
    engine, path = claude_engine(tmp_path, monkeypatch)
    try:
        engine.upsert_contracts([{"id": "said-done", "type": "must", "scope": "message", "regex": "all done",
                                  "provider": "claude", "roles": ["assistant"]}])
        write(path, assistant(text("zebrafish in the middle of the session")), mode="a")
        run(engine, path, "Stop")
        indexed = engine.sparse.ids()
        stored = engine.store.chunk_ids(CLAUDE_SID, "claude")
        assert stored and not stored <= indexed                      # the hook published events, not full-text rows
        found = engine.search_history("zebrafish", mode="sparse", session_id=CLAUDE_SID, provider="claude")
        assert any("zebrafish" in hit["snippet"] for hit in found["hits"])
        assert stored <= engine.sparse.ids()                         # the search materialized what it lacked
    finally:
        engine.close()


def test_a_synchronous_sync_still_indexes_for_search(tmp_path, monkeypatch):
    engine, path = claude_engine(tmp_path, monkeypatch)
    try:
        engine.sync_session(session_id=CLAUDE_SID, provider="claude", path=str(path), embed=False)
        assert engine.store.chunk_ids(CLAUDE_SID, "claude") <= engine.sparse.ids()
        write(path, assistant(text("a later line")), mode="a")
        engine.sync_session(session_id=CLAUDE_SID, provider="claude", path=str(path), embed=False, derived=False)
        assert not engine.store.chunk_ids(CLAUDE_SID, "claude") <= engine.sparse.ids()
        engine.sync_session(session_id=CLAUDE_SID, provider="claude", path=str(path), embed=False)
        assert engine.store.chunk_ids(CLAUDE_SID, "claude") <= engine.sparse.ids()
    finally:
        engine.close()


def test_only_a_local_only_hook_engine_skips_the_search_index(claude, tmp_path):
    from self_directing_mcp import hook_policy
    engine, _path = claude
    assert hook_policy.wants_derived_work(engine) is False
    other = SelfDirectEngine(Settings(_env_file=None, local_only=False, use_fake_embedder=True,
                                      index_dir=tmp_path / "other", dense_backend="numpy",
                                      openrouter_api_key_file=None))
    assert hook_policy.wants_derived_work(other) is True


# --- the cooperative stop check is lock free but loses nothing -----------------------------------------------------

def test_a_check_does_not_wait_for_a_publication_in_progress():
    control = RequestControl()
    held, release = threading.Event(), threading.Event()

    def publisher():
        with control._gate:
            held.set()
            release.wait(10)

    thread = threading.Thread(target=publisher)
    thread.start()
    assert held.wait(5)
    try:
        with request_scope(control):
            started = time.monotonic()
            check_request()                                           # would block until the gate was released
            assert time.monotonic() - started < 1.0
    finally:
        release.set()
        thread.join(5)


def test_a_cancellation_from_another_thread_stops_a_running_loop():
    control = RequestControl()
    stopped = []

    def work():
        try:
            with request_scope(control):
                while True:
                    check_request()
        except RequestStopped as exc:
            stopped.append(exc.reason)

    thread = threading.Thread(target=work)
    thread.start()
    time.sleep(0.05)
    control.cancel()
    thread.join(5)
    assert stopped == ["request_cancelled"]
    with request_scope(control), pytest.raises(RequestStopped):
        check_request()                                               # after cancel() returned every check raises


def test_a_deadline_still_stops_the_work():
    with request_scope(deadline=time.monotonic() + 0.05):
        with pytest.raises(RequestStopped) as caught:
            while True:
                check_request()
    assert caught.value.reason == "request_deadline"


# --- the invalid-UTF-8 scan ----------------------------------------------------------------------------------------

def test_the_invalid_utf8_scan_agrees_with_the_loop_it_replaced():
    def reference(raw):
        return any(0xDC80 <= ord(char) <= 0xDCFF for char in raw)

    rng = random.Random(7)
    alphabet = ["a", "b", "\n", "é", "漢", "\U0001f600", "\udc80", "\udcff", "\udc7f", "\udd00", "퟿"]
    for _ in range(5000):
        raw = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        assert has_invalid_utf8(raw) == reference(raw), repr(raw)
    for data in (b"plain", b"\xff", b"a\x80b", "café".encode(), b"\xed\xb2\x80", b'{"k": "\xc3"}'):
        raw = data.decode("utf-8", errors="surrogateescape")
        assert has_invalid_utf8(raw) == reference(raw), data


# --- the append commit asks only about the ids it brings ----------------------------------------------------------

def chunk(index, session="s1"):
    from self_directing_mcp.session_events import make_chunk
    return make_chunk("codex", session, "turn", f"event {index}", {"role": "user"}, index + 1, index * 10, index * 10 + 9)


@pytest.fixture
def store(tmp_path):
    store = SessionStore(tmp_path / "meta.sqlite")
    yield store
    store._conn.close()


def commit(store, chunks, offset, **options):
    store.commit_events(chunks, session_id="s1", provider="codex", path=tmp_file(store), offset=offset,
                        digest=f"digest-{offset}", **options)


def tmp_file(store):
    path = store.db_path.parent / "session.jsonl"
    path.write_text("x")
    return path


def test_an_append_counts_only_new_ids_and_an_overwrite_starts_a_new_generation(store):
    commit(store, [chunk(0), chunk(1)], 20)
    assert store.session_metrics("s1", "codex")[0] == 2
    commit(store, [chunk(2), chunk(3)], 40)
    assert store.session_metrics("s1", "codex")[0] == 4
    generation = store.audit_lineage("s1", "codex")["generation"]
    commit(store, [chunk(3), chunk(4)], 50)                       # chunk 3 is already stored: the prefix was touched
    assert store.session_metrics("s1", "codex")[0] == 5
    assert store.audit_lineage("s1", "codex")["generation"] == generation + 1


def test_a_replacement_still_journals_the_ids_the_new_parse_no_longer_has(store):
    commit(store, [chunk(0), chunk(1), chunk(2)], 30)
    commit(store, [chunk(0), chunk(1)], 20, replace=True)
    assert store.session_metrics("s1", "codex")[0] == 2
    assert store.pending_deletions("s1", "codex") == [chunk(2).chunk_id]


# --- an audit of a growing Codex session after each of these changes is still the full evaluation ------------------

def test_the_whole_path_still_answers_like_the_full_evaluator(tmp_path, monkeypatch):
    for name in ("SELF_DIRECT_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR", "SELF_DIRECT_SESSION_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / f"rollout-2026-09-09T01-00-00-{SID}.jsonl"
    write_codex(path, [{"type": "session_meta", "payload": {"id": SID}}, message("go", "user")])
    base = dict(_env_file=None, local_only=True, openrouter_api_key_file=None, codex_sessions_dir=root)
    fast = SelfDirectEngine(Settings(index_dir=tmp_path / "fast", audit_incremental=True, audit_shadow_every=1, **base))
    full = SelfDirectEngine(Settings(index_dir=tmp_path / "full", audit_incremental=False, **base))
    rules = [{"id": "no-deploy", "type": "must_not", "scope": "tool_call", "regex": "deploy", "roles": ["assistant"]},
             {"id": "tests-first", "type": "must", "scope": "tool_call", "regex": "test", "before_regex": "deploy",
              "requires_success": True}]
    try:
        for engine in (fast, full):
            engine.upsert_contracts(rules)
        for step in range(6):
            write_codex(path, [call(f"echo {step}", f"c{step}"), result(f"c{step}")] +
                        ([call("test", "t")] if step == 2 else []) + ([call("deploy", "d")] if step == 4 else []), "a")
            a = fast.audit_session(SID, provider="codex")
            b = full.audit_session(SID, provider="codex")
            assert [(f["contract_id"], f["verdict"]) for f in a["findings"]] == \
                [(f["contract_id"], f["verdict"]) for f in b["findings"]] and a["verdict"] == b["verdict"]
    finally:
        fast.close()
        full.close()


# --- the reference evaluator shares work between rules, never answers -------------------------------------------------

@pytest.mark.parametrize("provider", ["codex", "claude", "agy", "omp"])
@pytest.mark.parametrize("seed", range(8))
def test_a_rule_judged_with_the_others_is_judged_as_if_it_were_alone(provider, seed):
    """One evaluation builds a position index, a call/result table per activation point and (AGY) the places of the rare
    evidence classes once, for every rule. Sharing may not let one rule's inputs reach another's finding: each finding of
    the whole set equals the finding of that rule evaluated by itself."""
    rng = random.Random(f"share/{provider}/{seed}")
    session = Session(provider, rng, call_reuse=rng.choice([0.1, 0.3]), deny_mismatch=rng.choice([0.1, 0.5]))
    session.extend(rng.choice([40, 120, 260]))
    rules = random_contracts(rng, provider, session.events, 12, descriptions=0.1)
    for rule in rules[::6]:
        rule.enabled = False
    for proposed in (None, proposed_action(rng)):
        together = runner.full_findings(rules, session.events, session_id="s", store=None, provider=provider,
                                        proposed=proposed, coverage={})
        for rule, joint in zip(rules, together):
            alone = runner.full_findings([rule], session.events, session_id="s", store=None, provider=provider,
                                         proposed=proposed, coverage={})[0]
            assert alone.model_dump() == joint.model_dump(), (rule.id, rule.type, rule.scope)


# --- a resident engine reads a conversation's receipts once per change, not a dozen times per request --------------

@pytest.fixture
def receipts(tmp_path):
    """A resident store (it has a buffer, so it reads through a persistent reader) and a plain one on the same file."""
    resident, plain = ReceiptStore(tmp_path, held=DerivedBuffer()), ReceiptStore(tmp_path)
    resident._connect().close()                                   # the tables
    yield resident, plain
    ReceiptStore.release(tmp_path)


def put(store, session, step, text="{}"):
    with store._connect() as connection:
        connection.execute("INSERT OR REPLACE INTO receipts VALUES (?,?,?)", (session, step, text))


def test_the_resident_token_is_the_plain_token_after_every_kind_of_change(receipts):
    resident, plain = receipts
    assert resident.authority_token("c1") == plain.authority_token("c1")
    for step, text in enumerate(["{}", '{"a": 1}', '{"a": 2}'], start=1):                       # inserts
        put(plain, "c1", step, text)
        assert resident.authority_token("c1") == plain.authority_token("c1")
    put(plain, "c1", 2, '{"a": 9}')                                                          # same size, other content
    assert resident.authority_token("c1") == plain.authority_token("c1")
    before = resident.authority_token("c1")
    put(plain, "c2", 1)                                                                      # another conversation
    assert resident.authority_token("c1") == before and resident.authority_token("c2") == plain.authority_token("c2")
    with plain._connect() as connection:
        connection.execute("DELETE FROM receipts WHERE session_id='c1' AND step=3")        # a removal
    assert resident.authority_token("c1") == plain.authority_token("c1") != before


def test_the_resident_reader_sees_a_commit_of_another_connection_on_the_very_next_read(receipts):
    resident, plain = receipts
    put(plain, "c1", 1)
    first = resident.read_capture("c1")
    assert resident.read_capture("c1") == first                       # served again (same rows) ...
    put(plain, "c1", 2, '{"later": true}')
    second = resident.read_capture("c1")
    assert second[0] != first[0] and [step for step, _ in second[0]] == [1, 2]    # ... until the database changes
    assert resident.checkpoints("c1", "/p") == plain.checkpoints("c1", "/p")
    with plain._connect() as connection:
        connection.execute("UPDATE receipts SET data='{\"prefix_length\": 77}' WHERE step=2")
    assert resident.checkpoints("c1", "/p") == plain.checkpoints("c1", "/p") == (77,)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="deleting or replacing a file another handle holds open")
def test_a_replaced_or_recreated_database_is_never_served_from_the_old_reader(receipts, tmp_path):
    resident, plain = receipts
    put(plain, "c1", 1, '{"generation": "old"}')
    old = resident.authority_token("c1")
    assert old[0] == "present"
    resident.path.unlink()
    assert resident.authority_token("c1") == ("missing", None, None) == plain.authority_token("c1")
    resident._connect().close()                                       # recreated, empty
    fresh = resident.authority_token("c1")
    assert fresh[0] == "present" and fresh[2] != old[2] and fresh == plain.authority_token("c1")
    assert resident.read_capture("c1")[0] == ()
    # replaced by a different file at the same path (rename over it)
    other = ReceiptStore(tmp_path / "elsewhere")
    put(other, "c1", 1, '{"generation": "replacement"}')
    os.replace(other.path, resident.path)
    replaced = resident.authority_token("c1")
    assert replaced == plain.authority_token("c1") and replaced[2] != fresh[2]
    assert resident.read_capture("c1")[0] == ((1, '{"generation": "replacement"}'),)


def test_concurrent_readers_and_a_writer_end_up_agreeing(receipts):
    resident, plain = receipts
    stop = threading.Event()
    seen, errors = [], []

    def read():
        while not stop.is_set():
            try:
                seen.append(resident.authority_token("c1")[2])
            except Exception as exc:                                  # noqa: BLE001 - reported below
                errors.append(exc)

    threads = [threading.Thread(target=read) for _ in range(3)]
    for thread in threads:
        thread.start()
    for step in range(1, 40):
        put(plain, "c1", step, json.dumps({"step": step}))
    stop.set()
    for thread in threads:
        thread.join(10)
    assert not errors and resident.authority_token("c1") == plain.authority_token("c1")
    assert len(set(seen)) > 1                                         # the readers did see the writer's progress


def test_readers_are_bounded_and_released(tmp_path):
    stores = [ReceiptStore(tmp_path / f"index-{n}", held=DerivedBuffer()) for n in range(agy_receipts._READER_LIMIT + 2)]
    for store in stores:
        store._connect().close()
        store.authority_token("c")
    assert len(agy_receipts._READERS) <= agy_receipts._READER_LIMIT
    for store in stores:
        ReceiptStore.release(store.path.parent)
    assert not agy_receipts._READERS
    assert ReceiptStore(tmp_path / "never-created", held=DerivedBuffer()).authority_token("c") == ("missing", None, None)
