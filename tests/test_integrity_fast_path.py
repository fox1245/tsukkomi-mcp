"""Hook-time source verification: fast checks, full verification, and what each may claim.

Synthetic sessions only, with a small block size (4 KiB) so tampering with "a middle block" is cheap. Every
tamper scenario runs for the fast and the full path where both apply. The rules under test:

* a fast check compares only the tail, the latest blocks and a random sample, then parses what was appended;
  it reports ``integrity.mode == "fast"`` and never presents itself as a verification of the whole source;
* the full path (audit_session, Stop, a stale last verification, the background worker) catches what the fast
  path may have missed, for any block;
* a mismatch is ``source_changed`` and a reindex, exactly as a whole-prefix hash mismatch was; it is never
  treated as a runtime failure.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
import random

import pytest

from self_directing_mcp import hook_policy, source_blocks
from self_directing_mcp.codex_hooks import handle_hook
from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index.ingest import IntegrityPolicy

SID = "11111111-2222-3333-4444-555555555555"
BLOCK = 4096  # the default block size of this module's worlds; see World(block=...)
DEPLOY = {"tool_name": "shell", "arguments": {"command": "deploy now"}}
FORBIDDEN = {"tool_name": "shell", "arguments": {"command": "FORBIDDEN_MARKER now"}}
SAFE = {"tool_name": "shell", "arguments": {"command": "echo hello"}}
FILLER = "x" * 180


def line(row):
    return (json.dumps(row) + "\n").encode()


def message(index, text=None):
    return {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                                 "content": text or f"{index:06d} {FILLER}"}}


def call(command, call_id):
    return {"type": "response_item", "payload": {"type": "function_call", "name": "shell", "call_id": call_id,
                                                 "arguments": {"command": command}}}


def result(call_id, code=0):
    return {"type": "response_item", "payload": {"type": "function_call_output", "call_id": call_id,
                                                 "output": json.dumps({"exit_code": code, "output": "ok"})}}


class Clock:
    """Controllable wall clock for the policy."""

    def __init__(self):
        self.now = datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


class Sample:
    """A generator that 'randomly' samples exactly the given blocks."""

    def __init__(self, *picked):
        self.picked = list(picked)

    def sample(self, _population, count):
        return self.picked[:count]


class World:
    """One synthetic Codex session of about 30 blocks with the policy under the test's control."""

    def __init__(self, tmp_path, monkeypatch, *, filler=480, prerequisite=True, recent=2, samples=0, rng=None,
                 block=BLOCK):
        monkeypatch.setattr(source_blocks, "BLOCK_SIZE", block)
        self.block = block
        self.root = tmp_path / "sessions"
        self.root.mkdir(parents=True)
        self.path = self.root / f"rollout-2026-10-15T01-00-00-{SID}.jsonl"
        rows = [{"type": "session_meta", "payload": {"id": SID}}]
        if prerequisite:
            rows += [call("pytest -q", "t1"), result("t1")]
        rows += [message(i) for i in range(filler)]
        self.path.write_bytes(b"".join(line(row) for row in rows))
        self.engine = SelfDirectEngine(Settings(codex_sessions_dir=self.root, index_dir=tmp_path / "index",
                                                local_only=True, _env_file=None))
        self.clock = Clock()
        self.configure(recent=recent, samples=samples, rng=rng)
        self.engine.upsert_contracts([
            {"id": "tests-first", "type": "must", "scope": "tool_call", "regex": "pytest", "before_regex": "deploy"},
            {"id": "no-forbidden", "type": "must_not", "scope": "tool_call", "regex": "FORBIDDEN_MARKER"}])

    def configure(self, *, recent=2, samples=0, rng=None, interval=300.0):
        self.engine.integrity_policy = IntegrityPolicy(
            recent_blocks=recent, sample_blocks=samples, full_verify_interval_sec=interval, clock=self.clock,
            rng=(lambda: rng) if rng is not None else (lambda: random.Random(7)))

    # -- operations
    def check(self, action=DEPLOY, verify="fast"):
        return self.engine.check_action(SID, action, verify=verify)

    def audit(self):
        return self.engine.audit_session(SID)

    def sync(self, verify="full"):
        return self.engine.sync_session(session_id=SID, embed=False, verify=verify)

    def append(self, *rows):
        with self.path.open("ab") as stream:
            for row in rows:
                stream.write(line(row))

    def record(self):
        return self.engine.store.source_block_record(SID, "codex")

    def nblocks(self):
        return self.path.stat().st_size // self.block

    # -- tampering (always the same size; the modification time is restored unless asked otherwise)
    def edit(self, offset, byte, *, restore_mtime=True):
        stat = self.path.stat()
        data = bytearray(self.path.read_bytes())
        data[offset] = byte
        self.path.write_bytes(bytes(data))
        if restore_mtime:
            os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    def filler_offset(self, block):
        """An offset inside a filler message of ``block``: flipping it keeps the file valid JSONL."""
        data = self.path.read_bytes()
        position = data.index(b"x" * 40, block * self.block + 2) + 20
        assert position < (block + 1) * self.block + 100
        return position

    def tamper_filler(self, block, **options):
        self.edit(self.filler_offset(block), ord("y"), **options)

    def tamper_prerequisite(self, **options):
        data = self.path.read_bytes()
        self.edit(data.index(b"pytest -q") + 2, ord("f"), **options)  # pytest -> pyfest: the prerequisite is gone

    def forge_tail(self, **options):
        data = self.path.read_bytes()
        self.edit(len(data) - 30, ord("z"), **options)


@pytest.fixture
def make_world(tmp_path, monkeypatch):
    worlds = []

    def build(**options):
        world = World(tmp_path / f"world{len(worlds)}", monkeypatch, **options)
        worlds.append(world)
        return world

    yield build
    for world in worlds:
        world.engine.close()


@pytest.fixture
def world(make_world):
    return make_world()


def integrity(result):
    return result["integrity"]


# ------------------------------------------------------------------------------------------------ the object

def test_every_relevant_response_carries_the_integrity_object(world):
    expected = {"mode", "full_verified_through_byte", "full_verified_at", "source_generation", "index_generation"}
    first = world.check(verify="full")
    synced = world.sync()
    fast = world.check()
    audited = world.audit()
    for response in (first, synced, fast, audited):
        assert expected <= set(response["integrity"]), response["integrity"]
    assert [first["integrity"]["mode"], synced["integrity"]["mode"], fast["integrity"]["mode"],
            audited["integrity"]["mode"]] == ["full", "full", "fast", "full"]
    size = world.path.stat().st_size
    assert first["integrity"]["full_verified_through_byte"] == size
    assert first["integrity"]["full_verified_at"] == world.clock().isoformat()
    # A fast check reports the PAST full verification (unchanged), the generation it observed and what it compared.
    assert fast["integrity"]["full_verified_at"] == first["integrity"]["full_verified_at"]
    assert fast["integrity"]["full_verified_through_byte"] == size
    assert fast["integrity"]["source_generation"] == first["integrity"]["source_generation"]
    assert fast["integrity"]["index_generation"] == first["integrity"]["index_generation"]
    assert fast["integrity"]["fast_checked"]["tail"] is True
    assert "fast_checked" not in audited["integrity"]


def test_a_full_verification_restamps_the_table_even_when_nothing_was_published(world):
    first = world.check(verify="full")["integrity"]
    world.clock.advance(100)
    audited = world.audit()["integrity"]  # nothing appended: no cursor write, only the verification stamp moves
    assert audited["full_verified_at"] == world.clock().isoformat() != first["full_verified_at"]
    assert world.record().full_verified_at == audited["full_verified_at"]
    assert world.check()["integrity"]["full_verified_at"] == audited["full_verified_at"]  # the next fast check sees it
    world.clock.advance(250)  # 350 s after the first verification but only 250 after the audit's
    assert world.check()["integrity"]["mode"] == "fast"


def test_generations_move_with_the_source_and_the_index(world):
    first = world.check(verify="full")["integrity"]
    world.append(message(9001))
    second = world.check()["integrity"]
    assert second["source_generation"] != first["source_generation"]
    assert second["index_generation"] != first["index_generation"]
    assert second["full_verified_through_byte"] == first["full_verified_through_byte"]  # a past fact, not advanced
    third = world.check()["integrity"]
    assert third["source_generation"] == second["source_generation"] and third["index_generation"] == second["index_generation"]


def test_hook_judgements_carry_it(make_world):
    world = make_world(prerequisite=False)  # the deploy gate is unmet, so the hook has something to report
    world.check(verify="full")
    seen = []
    real = world.engine.check_action

    def spy(*args, **kwargs):
        seen.append(kwargs.get("verify"))
        return real(*args, **kwargs)

    world.engine.check_action = spy
    out = handle_hook(world.engine, "PreToolUse", SID, str(world.path), "shell", {"command": "deploy now"})
    assert seen == ["fast"]  # hook time asks for the cheap verification
    facts = json.loads(out["hookSpecificOutput"]["additionalContext"].split("): ", 1)[1])
    assert facts["integrity"]["mode"] == "fast" and facts["integrity"]["full_verified_through_byte"] > 0
    assert facts["integrity"]["full_verified_at"] == world.clock().isoformat()


# ---------------------------------------------------------------------------------- what fast mode costs

def test_a_fast_check_compares_only_the_selected_blocks(make_world, hash_meter):
    world = make_world(filler=8000, block=32 * 1024, recent=2, samples=3, rng=random.Random(11))  # ~2 MiB, 64 blocks
    world.check(verify="full")
    hash_meter.lap()
    result = world.check()
    size = world.path.stat().st_size
    assert result["integrity"]["mode"] == "fast"
    checked = result["integrity"]["fast_checked"]
    assert len(checked["recent"]) == 2 and len(checked["sampled"]) == 3
    # 5 blocks compared, the tail block, the 64 KiB tail window a handful of times and the roots over the stored
    # digests: well under half of the file, and independent of how large the file is (see bench_integrity.py).
    assert hash_meter.lap() < size / 2


def test_a_fast_check_after_an_append_parses_only_what_was_appended(world, monkeypatch):
    from self_directing_mcp.codex import parse as codex_parser
    world.check(verify="full")
    starts = []
    real = codex_parser.parse_session_chunks

    def spy(*args, **kwargs):
        starts.append(kwargs.get("start_byte"))
        return real(*args, **kwargs)

    monkeypatch.setattr(codex_parser, "parse_session_chunks", spy)
    covered = world.record().table.covered
    world.append(message(7001), message(7002))
    result = world.check()
    assert result["integrity"]["mode"] == "fast" and starts == [covered]
    assert world.record().table.covered == world.path.stat().st_size
    assert world.engine.store.chunk_count(SID, "codex") == 2 + 2 + 480 + 1  # meta, prerequisite pair, filler, +2


# ------------------------------------------------------------------------- equivalence on untampered data

ACTIONS = {"deploy": DEPLOY, "forbidden": FORBIDDEN, "safe": SAFE}


def comparable(response):
    # ``timings_ms`` and ``incremental`` describe how this request was evaluated (the second call reads the state the
    # first one stored), not what it found.
    out = {key: value for key, value in response.items() if key not in ("timings_ms", "integrity", "incremental")}
    out["findings"] = [{key: value for key, value in finding.items() if key != "verified_through"}
                       for finding in response["findings"]]
    out["execution"] = response["execution"]["nodes"]  # the run id differs per run
    return out


@pytest.mark.parametrize("name", sorted(ACTIONS))
@pytest.mark.parametrize("step", ["unchanged", "benign_append", "prerequisite_appended", "unfinished_record"])
def test_fast_and_full_verdicts_and_coverage_agree_on_untampered_data(make_world, name, step):
    world = make_world(prerequisite=step != "prerequisite_appended")
    world.check(verify="full")
    if step == "benign_append":
        world.append(message(7001), message(7002))
    elif step == "prerequisite_appended":
        world.append(call("pytest -q", "t9"), result("t9"))
    elif step == "unfinished_record":
        with world.path.open("ab") as stream:
            stream.write(b'{"type": "response_item", "payload": {"type": "mess')
    action = ACTIONS[name]
    fast = world.check(action, verify="fast")
    full = world.check(action, verify="full")
    assert fast["integrity"]["mode"] in ("fast", "full")
    assert full["integrity"]["mode"] == "full"
    assert comparable(fast) == comparable(full)
    if step == "unfinished_record":
        assert fast["coverage"]["complete"] is False and "unread_bytes" in fast["coverage"]["issues"]


def test_sync_coverage_agrees_between_fast_and_full(world):
    world.check(verify="full")
    world.append(message(7001))
    fast = world.sync("fast")
    again = world.sync("full")
    assert fast["integrity"]["mode"] == "fast" and again["integrity"]["mode"] == "full"
    assert fast["coverage"] == again["coverage"] and fast["new_chunks"] == 1 and again["new_chunks"] == 0


# ---------------------------------------------------------------------------- tamper: fast and full paths

def reindexed(world, verify="full"):
    info = world.sync(verify)
    return info["reindexed"] and info["reindex_reason"] == "source_changed"


def test_baseline_untampered_is_not_reindexed_by_either_path(world):
    world.check(verify="full")
    assert not reindexed(world, "fast") and not reindexed(world, "full")


@pytest.mark.parametrize("path", ["full", "fast"])
def test_same_size_rewrite_with_the_modification_time_restored(make_world, path):
    """The hard case: size, identity and modification time all look untouched."""
    world = make_world(recent=2, samples=1, rng=Sample(0))  # the sample is block 0: where the prerequisite lives
    assert world.check(verify="full")["verdict"] == "clean"
    world.tamper_prerequisite()
    assert world.path.stat().st_size == world.record().size
    result = world.check(verify=path)
    assert result["verdict"] != "clean"
    assert integrity(result)["mode"] == "full"  # a fast check that finds a mismatch settles it in full
    assert integrity(result).get("full_reason") == ("block_mismatch" if path == "fast" else None)
    assert integrity(result)["reindex_reason"] == "source_changed"  # a mismatch is a finding, reported as such
    assert not any("pytest" in c.text for c in world.engine.store.list_chunks(SID, "codex"))


@pytest.mark.parametrize("path", ["full", "fast"])
def test_middle_block_tamper(make_world, path):
    world = make_world(recent=2, samples=1, rng=Sample(12))
    world.check(verify="full")
    world.tamper_filler(12)
    assert reindexed(world, path)


@pytest.mark.parametrize("path", ["full", "fast"])
def test_tail_tamper(make_world, path):
    """The tail block is always compared, even with no recent or sampled blocks."""
    world = make_world(recent=0, samples=0)
    world.check(verify="full")
    world.forge_tail()
    info = world.sync(path)
    assert info["reindexed"] and info["reindex_reason"] == "source_changed"
    assert info["integrity"]["mode"] == "full"
    if path == "fast":
        assert info["integrity"]["full_reason"] == "tail_mismatch"


@pytest.mark.parametrize("path", ["full", "fast"])
def test_truncation(make_world, path):
    world = make_world()
    world.check(verify="full")
    data = world.path.read_bytes()
    stat = world.path.stat()
    world.path.write_bytes(data[:len(data) * 3 // 4])
    os.utime(world.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    info = world.sync(path)
    assert info["reindexed"] and info["reindex_reason"] == "source_changed" and info["integrity"]["mode"] == "full"
    if path == "fast":
        assert info["integrity"]["full_reason"] == "shrunk"
    assert world.engine.store.cursor_record(SID, "codex")["byte_offset"] <= world.path.stat().st_size


@pytest.mark.parametrize("path", ["full", "fast"])
@pytest.mark.parametrize("content", ["identical", "tampered"])
def test_replacement_by_a_new_inode(make_world, path, content):
    world = make_world()
    world.check(verify="full")
    data = world.path.read_bytes()
    if content == "tampered":
        data = data.replace(b"pytest -q", b"pyfest -q", 1)
    replacement = world.path.with_name("replacement.tmp")
    replacement.write_bytes(data)
    os.replace(replacement, world.path)
    info = world.sync(path)
    assert info["integrity"]["mode"] == "full"  # a new inode is never settled by the cheap check
    if path == "fast":
        assert info["integrity"]["full_reason"] == "identity_changed"
    if content == "identical":
        assert not info["reindexed"]  # same bytes: not a change of content
    else:
        assert info["reindexed"] and info["reindex_reason"] == "source_changed"


@pytest.mark.parametrize("path", ["full", "fast"])
@pytest.mark.parametrize("where", ["sampled", "tail", "unsampled"])
def test_growth_plus_tamper(make_world, path, where):
    world = make_world(recent=1, samples=1, rng=Sample(3))
    world.check(verify="full")
    block = {"sampled": 3, "tail": world.record().table.nfull, "unsampled": 10}[where]
    world.tamper_filler(block)
    world.append(message(7001), message(7002))
    info = world.sync(path)
    if path == "full" or where != "unsampled":
        assert info["reindexed"] and info["reindex_reason"] == "source_changed" and info["integrity"]["mode"] == "full"
    else:
        # The stated limit of the fast path: block 10 is not compared. The check says so, and the appended
        # records are still not published on top of a laundered prefix: the stored table keeps the OLD digest.
        assert not info["reindexed"] and info["integrity"]["mode"] == "fast"
        assert world.sync("full")["reindexed"] is True  # the next full verification finds it


def test_a_tamper_in_a_sampled_block_is_found_an_unsampled_one_is_not(make_world):
    """Sampled vs unsampled: the same rewrite of the same block, depending only on which blocks were drawn."""
    sampled = make_world(recent=1, samples=1, rng=Sample(8))
    sampled.check(verify="full")
    sampled.tamper_filler(8)
    found = sampled.check(SAFE)
    assert found["integrity"]["mode"] == "full" and found["integrity"]["full_reason"] == "block_mismatch"

    unsampled = make_world(recent=1, samples=1, rng=Sample(9))
    unsampled.check(verify="full")
    unsampled.tamper_filler(8)
    missed = unsampled.check(SAFE)
    assert missed["integrity"]["mode"] == "fast"  # not drawn: not found, and honestly labelled fast
    assert missed["integrity"]["fast_checked"]["sampled"] == [9]
    assert unsampled.audit()["integrity"]["mode"] == "full"  # the full path covers every block
    assert unsampled.engine.store.cursor_record(SID, "codex")["prefix_hash"] == \
        source_blocks.HASH_TAG + unsampled.record().table.root  # (and re-anchored the table on the changed bytes)
    assert tampered_text(unsampled)


# ------------------------------------------- eventually guaranteed by the full path (event or time triggered)

def silent_tamper(world):
    """A rewrite no fast check can see with the default test policy: block 12, modification time restored."""
    world.tamper_filler(12)


def tampered_text(world):
    """Whether the index now holds the rewritten bytes (a filler message with a 'y' inside its 'x' run)."""
    return any(c.text.count("y") == 1 and "xx" in c.text for c in world.engine.store.list_chunks(SID, "codex"))


def test_audit_session_is_full_and_finds_what_fast_checks_missed(world):
    world.check(verify="full")
    silent_tamper(world)
    assert world.check(SAFE)["integrity"]["mode"] == "fast"  # missed, and labelled as fast
    assert not tampered_text(world)  # the index still describes the old bytes
    audit = world.audit()
    assert audit["integrity"]["mode"] == "full"
    assert tampered_text(world)  # the full path read every block and reindexed
    assert reindexed(world) is False  # nothing is left to find


def test_the_stale_last_verification_makes_the_next_check_full_and_finds_it(make_world):
    world = make_world()
    world.check(verify="full")
    silent_tamper(world)
    world.clock.advance(120)
    assert world.check(SAFE)["integrity"]["mode"] == "fast"  # inside the interval: still the cheap check
    world.clock.advance(181)  # 301 s after the last full verification
    late = world.check(SAFE)
    assert late["integrity"]["mode"] == "full" and late["integrity"]["full_reason"] == "interval_elapsed"
    assert late["integrity"]["verification_lag_sec"] == pytest.approx(1.0, abs=0.01)  # the deadline was missed by 1 s
    assert tampered_text(world)  # the time-triggered full verification found the silent rewrite
    assert world.check(SAFE)["integrity"]["mode"] == "fast"  # and the verification clock restarted


def test_the_interval_is_a_setting(make_world):
    world = make_world()
    world.configure(interval=10.0)
    world.check(verify="full")
    world.clock.advance(11)
    assert world.check(SAFE)["integrity"]["full_reason"] == "interval_elapsed"
    world.configure(interval=0.0)  # zero: every check is full
    assert world.check(SAFE)["integrity"]["mode"] == "full"


def test_the_stop_checkpoint_verifies_in_full(world):
    world.check(verify="full")
    silent_tamper(world)
    obligations = world.engine.hook_obligations(SID)
    hook_policy.checkpoint(world.engine, "codex", SID, str(world.path), obligations, "Stop")
    assert tampered_text(world)  # the Stop checkpoint read every block (event-triggered full verification)


def test_auto_continue_needs_a_full_audit(world):
    class Rule:
        id, type, auto_continue, enabled, revision, risk, regex, description = "a", "must", True, True, 3, "normal", "x", None

    audit = {"ok": True, "coverage": {"complete": True, "issues": []},
             "findings": [{"contract_id": "a", "verdict": "violation", "contract_revision": 3,
                           "applicability": "applicable", "reason": "missing"}]}
    obligations = {"contracts": [Rule()]}
    assert hook_policy.confirmed_misses(obligations, {**audit, "integrity": {"mode": "full"}})
    assert hook_policy.confirmed_misses(obligations, {**audit, "integrity": {"mode": "fast"}}) == []
    assert hook_policy.confirmed_misses(obligations, audit) == []


# --------------------------------------------------------------------- what a fast verdict may rely on

def test_a_fast_verdict_rests_only_on_evidence_inside_the_fully_verified_region(make_world):
    world = make_world(prerequisite=False)
    first = world.check(verify="full")
    assert first["verdict"] != "clean"  # no prerequisite yet: the gate is unmet
    world.append(call("pytest -q", "t9"), result("t9"))
    result_ = world.check()  # the satisfying evidence is newer than the last full verification
    # A fast verdict does not accept evidence it cannot place in a fully verified region: it is settled in full.
    assert result_["verdict"] == "clean" and result_["integrity"]["mode"] == "full"
    assert result_["integrity"]["full_reason"] == "evidence_beyond_verified_region"
    again = world.check()  # now the evidence lies inside the region the full verification covered
    assert again["verdict"] == "clean" and again["integrity"]["mode"] == "fast"
    gate = next(f for f in again["findings"] if f["contract_id"] == "tests-first")
    assert gate["verified_through"] == again["integrity"]["full_verified_through_byte"]
    assert all(int(e["chunk_id"].rsplit(":", 3)[1]) < gate["verified_through"] for e in gate["evidence"]
               if e.get("detail") != "triggering action")


def test_a_fast_verdict_never_turns_a_gap_into_a_satisfied_obligation(make_world):
    world = make_world(prerequisite=False)
    world.check(verify="full")
    for _ in range(3):
        result_ = world.check()
        assert result_["verdict"] != "clean" and result_["integrity"]["mode"] == "fast"
    with world.path.open("ab") as stream:
        stream.write(b'{"type": "response_item", "payload": {"type": "function_call", "name": "sh')
    partial = world.check()
    assert partial["verdict"] != "clean" and "unread_bytes" in partial["coverage"]["issues"]


def test_description_and_unknown_stay_unknown_in_fast_mode(world):
    world.engine.upsert_contracts([{"id": "idea", "type": "must_not", "description": "no roses"}])
    full = world.check(SAFE, verify="full")
    fast = world.check(SAFE)
    assert full["verdict"] == fast["verdict"] == "unknown"
    assert fast["integrity"]["mode"] == "fast"


# ---------------------------------------------------------------------------------------- compatibility

def test_a_plain_cursor_from_before_this_version_migrates_in_full_then_goes_fast(world):
    world.check(verify="full")
    cursor = world.engine.store.cursor_record(SID, "codex")
    from self_directing_mcp.session_events import capture_source
    plain = capture_source(world.path, checkpoints=(cursor["byte_offset"],)).prefix_digest(cursor["byte_offset"])
    store = world.engine.store
    with store._conn:
        store._conn.execute("UPDATE session_cursors SET prefix_hash=?", (plain,))
        store._conn.execute("DELETE FROM source_blocks")
    migrated = world.check()
    assert migrated["integrity"]["mode"] == "full" and migrated["integrity"]["full_reason"] == "migration"
    assert world.engine.store.cursor_record(SID, "codex")["prefix_hash"].startswith(source_blocks.HASH_TAG)
    assert world.check()["integrity"]["mode"] == "fast"
    assert not world.sync()["reindexed"]  # no reindex was needed to migrate


def test_a_cursor_written_by_an_older_process_is_not_trusted_with_a_stale_table(world):
    world.check(verify="full")
    store = world.engine.store
    cursor = store.cursor_record(SID, "codex")
    from self_directing_mcp.session_events import capture_source
    plain = capture_source(world.path, checkpoints=(cursor["byte_offset"],)).prefix_digest(cursor["byte_offset"])
    with store._conn:  # an older process republishes the cursor (new stamp, plain hash) and knows no table
        store._conn.execute("UPDATE session_cursors SET prefix_hash=?, synced_at='older-process-write'", (plain,))
        store._conn.execute("UPDATE session_parser_versions SET synced_at='older-process-write'")
    result = world.check()
    assert result["integrity"]["mode"] == "full" and result["integrity"]["full_reason"] == "migration"
    assert world.record().bound_to(store.cursor_record(SID, "codex"))


def test_a_damaged_block_row_is_never_trusted(world):
    world.check(verify="full")
    store = world.engine.store
    with store._conn:
        store._conn.execute("UPDATE source_blocks SET lines=lines+1")
    result = world.check()
    assert result["integrity"]["mode"] == "full" and result["integrity"]["full_reason"] == "no_record"
    assert world.check()["integrity"]["mode"] == "fast"  # repaired by that full verification


def test_the_schema_migration_adds_the_table_and_keeps_everything_else(tmp_path):
    from self_directing_mcp.index.ingest import SessionStore
    db = tmp_path / "meta.sqlite"
    store = SessionStore(db)
    with store._conn:
        store._conn.execute("INSERT INTO session_cursors VALUES ('codex','s','/p',10,'abc','2026-01-01T00:00:00+00:00')")
        store._conn.execute("DROP TABLE source_blocks")  # an index created by the previous version
    store._conn.close()
    reopened = SessionStore(db)
    tables = {row[0] for row in reopened._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "source_blocks" in tables
    assert reopened.cursor_record("s", "codex")["prefix_hash"] == "abc"  # nothing else was touched
    assert reopened.source_block_record("s", "codex") is None
    reopened._conn.close()


def test_a_non_block_source_form_keeps_full_verification(tmp_path):
    from self_directing_mcp.index.ingest import _blockable
    assert _blockable("codex", tmp_path / "a.jsonl") and _blockable("claude", tmp_path / "a.jsonl")
    assert not _blockable("agy", tmp_path / "a.jsonl") and not _blockable("codex", tmp_path / "a.jsonl.zst")


def test_file_ids_that_use_the_whole_unsigned_range_survive_the_database(tmp_path):
    """Windows file ids can exceed 2**63: SQLite integers are signed, so they are stored bit for bit."""
    from self_directing_mcp.index.ingest import SessionStore
    from self_directing_mcp.source_blocks import BlockDraft, BlockTable
    store = SessionStore(tmp_path / "meta.sqlite")
    table = BlockTable.build(0, 0, b"", None, BLOCK)
    for dev, ino in ((2 ** 64 - 1, 2 ** 63), (7, 2 ** 63 - 1), (0, 0)):
        draft = BlockDraft(table, dev, ino, 10, -5, "2026-10-15T00:00:00+00:00", 0)
        store.commit_events((), session_id="s", provider="codex", path=tmp_path / "p", offset=0,
                            digest=table.hash, blocks=draft)
        record = store.source_block_record("s", "codex")
        assert (record.dev, record.ino, record.size, record.mtime_ns) == (dev, ino, 10, -5)
        assert store.mark_full_verified("s", "codex", record, at="2026-10-15T00:01:00+00:00", through=0,
                                        dev=dev, ino=ino, size=10, mtime_ns=-5)
        assert store.source_block_record("s", "codex").ino == ino
    store._conn.close()


def test_a_table_replaced_meanwhile_is_never_stamped(tmp_path):
    from self_directing_mcp.index.ingest import SessionStore
    from self_directing_mcp.source_blocks import BlockDraft, BlockTable
    store = SessionStore(tmp_path / "meta.sqlite")
    first = BlockTable.build(0, 0, b"", None, BLOCK)
    publish = lambda table: store.commit_events(
        (), session_id="s", provider="codex", path=tmp_path / "p", offset=table.covered, digest=table.hash,
        blocks=BlockDraft(table, 1, 2, 3, 4, None, None))
    publish(first)
    stale = store.source_block_record("s", "codex")
    publish(BlockTable.build(0, 0, b"", None, BLOCK))  # a later publication of the same session
    assert not store.mark_full_verified("s", "codex", stale, at="2026-10-15T00:00:00+00:00", through=0,
                                        dev=1, ino=2, size=3, mtime_ns=4)  # compare-and-set on the publication
    assert store.source_block_record("s", "codex").full_verified_at is None
    store._conn.close()


# --------------------------------------------------------------------------- the settings and the surfaces

def test_the_settings_reach_the_policy_with_their_documented_defaults():
    settings = Settings(_env_file=None)
    policy = IntegrityPolicy.from_settings(settings)
    assert (policy.recent_blocks, policy.sample_blocks, policy.full_verify_interval_sec) == (4, 8, 300.0)
    assert settings.integrity_background_verify is True and settings.integrity_background_idle_sec == 2.0
    assert settings.integrity_background_max_scans == 1


def test_the_environment_overrides_the_integrity_settings(monkeypatch):
    monkeypatch.setenv("SELF_DIRECT_INTEGRITY_FAST_RECENT_BLOCKS", "1")
    monkeypatch.setenv("SELF_DIRECT_INTEGRITY_FAST_SAMPLE_BLOCKS", "2")
    monkeypatch.setenv("SELF_DIRECT_INTEGRITY_FULL_VERIFY_INTERVAL_SEC", "60")
    monkeypatch.setenv("SELF_DIRECT_INTEGRITY_BACKGROUND_VERIFY", "false")
    policy = IntegrityPolicy.from_settings(Settings(_env_file=None))
    assert (policy.recent_blocks, policy.sample_blocks, policy.full_verify_interval_sec) == (1, 2, 60.0)
    assert Settings(_env_file=None).integrity_background_verify is False


def test_the_mcp_tools_expose_the_verify_choice():
    import asyncio
    from self_directing_mcp import server
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    for name in ("check_action", "sync_session"):
        verify = tools[name].inputSchema["properties"]["verify"]
        assert verify["enum"] == ["full", "fast"] and verify["default"] == "full"
    assert "verify" not in tools["audit_session"].inputSchema["properties"]  # always full


def test_the_check_action_tool_passes_verify_through(world, monkeypatch):
    import asyncio
    from self_directing_mcp import server
    from self_directing_mcp.schemas import ProposedAction
    monkeypatch.setattr(server, "_engine", world.engine)
    world.check(verify="full")
    action = ProposedAction(tool_name="shell", arguments="echo hello")
    modes = {}
    for verify in ("fast", "full"):
        result = asyncio.run(server.check_action(SID, action, provider="codex", path=str(world.path), verify=verify))
        modes[verify] = result["integrity"]["mode"]
    default = asyncio.run(server.check_action(SID, action, provider="codex", path=str(world.path)))
    assert modes == {"fast": "fast", "full": "full"} and default["integrity"]["mode"] == "full"


def test_a_failed_refresh_claims_nothing_about_the_source(world, monkeypatch):
    world.check(verify="full")
    monkeypatch.setattr(world.engine, "_sync_session", lambda **_kwargs: {"ok": False, "error": "synthetic"})
    result = world.check(verify="fast")
    assert result["verdict"] == "unknown" and result["coverage"]["complete"] is False
    unavailable = result["integrity"]
    assert unavailable["verification"] == "unavailable" and unavailable["full_verified_through_byte"] is None
    assert unavailable["full_verified_at"] is None and unavailable["source_generation"] is None


def test_verify_is_validated(world):
    with pytest.raises(ValueError, match="verify must be"):
        world.engine.check_action(SID, DEPLOY, verify="quick")
    with pytest.raises(ValueError, match="verify must be"):
        world.engine.sync_session(session_id=SID, embed=False, verify="quick")


def test_hook_adapters_ask_for_the_fast_check(world):
    from self_directing_mcp.hook_policy import judge_action
    world.check(verify="full")
    seen = []
    real = world.engine.check_action

    def spy(*args, **kwargs):
        seen.append(kwargs.get("verify"))
        return real(*args, **kwargs)

    world.engine.check_action = spy
    judgement = judge_action(world.engine, "codex", SID, str(world.path), "shell", {"command": "deploy now"},
                             "confirmed_only")
    assert seen == ["fast"] and judgement is not None
