"""What a re-index does between its first slice and its commit (roundtable 8, PR3).

* While it is staged the index is the old one, byte for byte, and everything that reads it says so: audits judge the old events,
  the coverage keeps ``parser_outdated`` / ``source_changed`` and adds ``sync_in_progress``, a verdict is never clean, and a
  forbidden call in the old events is still a violation.
* When everything is staged but the commit does not fit the time left it is not started: ``ready_to_commit``, the spool is kept,
  nothing is reported as done, and a call with the time commits.
* A spool that no longer matches its source, receipts, parser or index is discarded as a whole, no mixed rows reach the index,
  the staging starts again from byte 0 and converges.
* A re-index that fits one call never makes a file; nothing is staged when slicing is off, the source is compressed, or the sync
  is not a re-index; a spool is private and is gone when its commit is.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import stat
import time

import pytest

from reindex_support import (BIG, Pair, canonical, edit_source, index_dump, old_parser, reindex_of, spool_files)
from self_directing_mcp import reindex_spool, request_stats, server
from self_directing_mcp.codex import parse as codex_parser
from self_directing_mcp.index import ingest
from self_directing_mcp.index.ingest import FlightBusy, SlicePolicy
from slicing_support import PROVIDERS, World, forced
from test_slice_flight import counted, stats  # noqa: F401  (the statistics fixture of the flight tests)

POSIX = os.name != "nt"


@pytest.fixture
def pair(tmp_path, monkeypatch):
    made = []

    def make(*args, **kwargs):
        made.append(Pair(tmp_path, monkeypatch, *args, **kwargs))
        return made[-1]

    yield make
    for item in made:
        item.close()


def far():
    return time.monotonic() + BIG


# --- the report -------------------------------------------------------------------------------------------------------

def test_the_report_has_the_five_states_in_a_fixed_order_of_precedence():
    from self_directing_mcp.index.ingest import REINDEX_STATES, _Stage

    class Spool:
        staged_through, rows = 700, 33

    stage = _Stage()
    stage.spool = Spool()
    assert stage.report(1000, 1000) == {"state": "staging", "staged_through": 700, "target_bytes": 1000, "rows": 33, "reason": None}
    stage.discarded = "prefix_changed"
    assert stage.report(1000, 1000)["state"] == "discarded" and stage.report(1000, 1000)["reason"] == "prefix_changed"
    stage.complete = True
    assert stage.report(1000, 1000)["state"] == "ready_to_commit" and stage.report(1000, 1000)["reason"] == "commit_exceeds_budget"
    stage.committed, stage.rows = True, 40
    assert stage.report(1000, 1000) == {"state": "committed", "staged_through": 1000, "target_bytes": 1000, "rows": 40,
                                        "reason": None}
    stage.disabled = "disk"
    assert stage.report(1000, 1000)["state"] == "disabled" and stage.report(1000, 1000)["reason"] == "disk"
    assert {"staging", "discarded", "ready_to_commit", "committed", "disabled"} == set(REINDEX_STATES)


@pytest.mark.parametrize("state,expected", [
    ("staging", ("partial", "soft_deadline", 0.25)),
    ("discarded", ("partial", "soft_deadline", 0.25)),
    ("ready_to_commit", ("partial", "eof", 1.0)),
    ("committed", ("complete", "eof", 1.0)),
    ("disabled", ("complete", "eof", 1.0)),
])
def test_a_sync_is_partial_while_a_reindex_waits_whatever_the_cursor_says(state, expected):
    from self_directing_mcp.index.ingest import _sync_report
    from self_directing_mcp.slice_budget import SliceBudget

    class Observation:
        complete_end = 1000

    budget = SliceBudget()
    budget.stop("soft_deadline", 10, 250)
    through = 1000 if state == "ready_to_commit" else 250
    report = _sync_report(1000, 1000, 0, budget, Observation(),
                          {"state": state, "staged_through": through, "target_bytes": 1000, "rows": 5, "reason": None})
    assert (report["state"], report["stopped_by"], report["progress"]) == expected
    assert report["indexed_through"] == 1000 and report["reindex"]["state"] == state
    assert report["retry_after_ms"] == (500 if expected[0] == "partial" else None)


# --- while it is staged -----------------------------------------------------------------------------------------------

def test_the_first_slice_stages_and_reports(pair):
    p = pair()
    info = p.stage(cap=30)
    sync, staged = info["sync"], reindex_of(info)
    assert info["ok"] is True and info["reindexed"] is False and info["new_chunks"] == 0
    assert info["reindex_reason"] == "parser_updated"            # why a re-index is under way, though it has not happened
    assert sync["state"] == "partial" and sync["stopped_by"] == "record_cap" and sync["retry_after_ms"] == 500
    assert set(staged) == {"state", "staged_through", "target_bytes", "rows", "reason"}
    assert staged["state"] == "staging" and staged["reason"] is None and staged["rows"] >= 30    # a record may be several rows
    assert 0 < staged["staged_through"] < staged["target_bytes"] == sync["target_bytes"]
    assert sync["progress"] == round(staged["staged_through"] / staged["target_bytes"], 4)
    assert sync["indexed_through"] == p.before["session_cursors"][0][3]       # the committed cursor is the old one
    assert p.intact() and len(spool_files(p.staged)) == 1


def test_the_coverage_keeps_the_old_issue_and_adds_the_new_one(pair):
    p = pair()
    coverage = p.stage()["coverage"]
    assert "parser_outdated" in coverage["issues"] and "sync_in_progress" in coverage["issues"]
    assert "reindex_ready_to_commit" not in coverage["issues"] and coverage["complete"] is False
    assert "backlog_bytes" not in coverage          # the old index covers the source's bytes; what is missing is the new parse
    changed = pair(trigger="source")
    coverage = changed.stage()["coverage"]
    assert "source_changed" in coverage["issues"] and "sync_in_progress" in coverage["issues"]


def test_the_nudge_says_how_far_the_staging_is_and_what_to_do(pair):
    p = pair()
    info = p.stage()
    assert "re-index is staged through" in info["nudge"] and "call sync_session again" in info["nudge"]
    assert "previous index stays in force" in info["nudge"] and "retry_after_ms=500" in info["nudge"]


def test_audits_and_checks_judge_the_old_events_while_a_reindex_is_staged(pair):
    p = pair()
    p.stage(cap=20)
    p.staged.slice_policy = forced(20)
    audit = p.staged.audit_session(p.sid, provider="codex", path=str(p.world.path), _deadline=far())
    cov = audit["coverage"]
    assert cov["complete"] is False and not cov.get("judged_prefix")
    assert {"parser_outdated", "sync_in_progress"} <= set(cov["issues"])
    assert audit["verdict"] == "violation"                      # a forbidden call in the old events is still found
    verdicts = {f["contract_id"]: f["verdict"] for f in audit["findings"]}
    assert verdicts["no-wipe"] == "violation"
    harmless = p.staged.check_action(p.sid, {"tool_name": "bash", "arguments": {"command": "echo hi"}}, provider="codex",
                                     path=str(p.world.path), _deadline=far())
    assert harmless["verdict"] == "unknown" and not harmless["coverage"].get("judged_prefix")
    dangerous = p.staged.check_action(p.sid, {"tool_name": "bash", "arguments": {"command": "rm -rf /data"}}, provider="codex",
                                      path=str(p.world.path), _deadline=far())
    assert dangerous["verdict"] == "violation"
    assert p.intact()                                            # three more slices were staged; the index never moved


def test_an_audit_with_a_clean_looking_old_index_is_still_not_clean(pair):
    """A session with nothing forbidden in it: the old index would be clean were it current. It is not current."""
    p = pair(count=120)
    p.staged.revoke_contract("no-wipe")
    p.stage(cap=15)
    p.staged.slice_policy = forced(15)
    audit = p.staged.audit_session(p.sid, provider="codex", path=str(p.world.path), _deadline=far())
    assert audit["verdict"] != "clean" and audit["coverage"]["complete"] is False
    assert "sync_in_progress" in audit["coverage"]["issues"]


def test_the_staged_calls_move_strictly_forward_and_the_last_one_publishes(pair):
    p = pair()
    through, calls = 0, 0
    while True:
        info = p.stage(cap=25)
        calls += 1
        staged = reindex_of(info)
        if info["sync"]["state"] == "complete":
            assert staged["state"] == "committed" and info["reindexed"] is True and info["new_chunks"] == staged["rows"]
            assert staged["staged_through"] == info["byte_offset"] == staged["target_bytes"] and staged["reason"] is None
            break
        assert staged["staged_through"] > through and staged["rows"] > 0 and p.intact()
        through = staged["staged_through"]
        assert calls < 200
    assert calls >= 8 and not spool_files(p.staged)
    assert canonical(p.staged, "codex", p.sid) == p.expected()


def test_a_reindex_that_fits_one_call_never_makes_a_file(pair):
    p = pair(count=60)
    info = p.finish()
    assert reindex_of(info)["state"] == "committed" and info["reindexed"] is True and info["sync"]["state"] == "complete"
    assert not spool_files(p.staged)
    assert canonical(p.staged, "codex", p.sid) == p.expected()


def test_an_explicit_sync_session_finishes_the_staging_and_says_what_it_did(pair, monkeypatch):
    p = pair()
    p.stage(cap=40)
    monkeypatch.setattr(server, "_engine", p.staged)
    p.staged.slice_policy = None
    answer = asyncio.run(server.sync_session(session_id=p.sid, provider="codex", embed=False, timeout_ms=60_000))
    assert answer["ok"] is True and answer["sync"]["state"] == "complete" and reindex_of(answer)["state"] == "committed"
    assert answer["reindexed"] is True and canonical(p.staged, "codex", p.sid) == p.expected()


# --- complete, but the commit does not fit -----------------------------------------------------------------------------

def slow_commit(monkeypatch, seconds_per_row=0.05):
    monkeypatch.setattr(ingest, "COMMIT_SECONDS_PER_ROW", seconds_per_row)


def test_a_staged_reindex_whose_commit_does_not_fit_waits_and_says_so(pair, monkeypatch):
    p = pair()
    slow_commit(monkeypatch)                                       # 0.05 s * 1.5 per row: minutes for a few hundred rows
    started = []
    original = ingest.SessionStore._commit_rows
    monkeypatch.setattr(ingest.SessionStore, "_commit_rows", lambda self, *a, **k: started.append(1) or original(self, *a, **k))
    p.staged.slice_policy = None
    info = p.staged.sync_session(**p.world.sync_args(), derived=False, _deadline=time.monotonic() + 10)
    sync, staged = info["sync"], reindex_of(info)
    assert started == []                                           # the transaction was not even started
    assert staged["state"] == "ready_to_commit" and staged["reason"] == "commit_exceeds_budget"
    assert staged["staged_through"] == staged["target_bytes"] and staged["rows"] > 100
    assert sync["state"] == "partial" and sync["progress"] == 1.0 and sync["stopped_by"] == "eof"
    assert info["reindexed"] is False and info["new_chunks"] == 0                # nothing is reported as done
    cov = info["coverage"]
    assert "reindex_ready_to_commit" in cov["issues"] and "sync_in_progress" not in cov["issues"]
    assert "parser_outdated" in cov["issues"] and cov["complete"] is False
    assert "commit does not fit" in info["nudge"] and "call sync_session again" in info["nudge"]
    assert p.intact() and len(spool_files(p.staged)) == 1
    audit = p.staged.audit_session(p.sid, provider="codex", path=str(p.world.path), _deadline=time.monotonic() + 10)
    assert audit["verdict"] == "violation" and "reindex_ready_to_commit" in audit["coverage"]["issues"]
    assert p.intact()


def test_a_call_with_more_time_commits_the_staged_reindex(pair, monkeypatch):
    p = pair()
    slow_commit(monkeypatch, 0.01)                                 # about 5 s for the rows this session has
    p.staged.slice_policy = None
    first = p.staged.sync_session(**p.world.sync_args(), derived=False, _deadline=time.monotonic() + 3)
    assert reindex_of(first)["state"] == "ready_to_commit" and p.intact()
    rows = reindex_of(first)["rows"]
    second = p.staged.sync_session(**p.world.sync_args(), derived=False, _deadline=time.monotonic() + 60)
    staged = reindex_of(second)
    assert staged["state"] == "committed" and staged["rows"] == rows and second["reindexed"] is True
    assert second["sync"]["state"] == "complete" and not spool_files(p.staged)
    assert "reindex_ready_to_commit" not in second["coverage"]["issues"] and "parser_outdated" not in second["coverage"]["issues"]
    assert canonical(p.staged, "codex", p.sid) == p.expected()


def test_the_prediction_is_eighteen_microseconds_a_row_with_a_margin(monkeypatch):
    assert ingest.COMMIT_SECONDS_PER_ROW == 18e-6 and ingest.COMMIT_SAFETY == 1.5
    from self_directing_mcp.request_control import request_scope
    with request_scope(deadline=time.monotonic() + 100):
        fits = lambda rows: ingest._commit_fits(rows)         # noqa: E731
        assert fits(3_000_000) and not fits(3_600_000)        # 18e-6 * 1.5 * rows against 90 % of a hundred seconds (3.33 M)
        monkeypatch.setattr(ingest, "COMMIT_SECONDS_PER_ROW", 36e-6)
        assert fits(1_500_000) and not fits(1_800_000)


def test_the_wall_of_one_transaction_is_about_one_and_a_half_million_rows_for_a_45_second_call():
    from self_directing_mcp.request_control import request_scope
    with request_scope(deadline=time.monotonic() + 45):
        assert ingest._commit_fits(1_400_000) and not ingest._commit_fits(1_700_000)


def test_the_statistics_count_a_waiting_commit_as_degraded_with_its_own_reason(pair, monkeypatch, stats):  # noqa: F811
    p = pair()
    monkeypatch.setattr(server, "_engine", p.staged)
    slow_commit(monkeypatch)
    p.staged.slice_policy = None
    sync = dict(session_id=p.sid, provider="codex", embed=False, derived=False, budget_override=10, heavy=False)
    answer = asyncio.run(server._dispatch(p.staged.sync_session, **sync))
    assert reindex_of(answer)["state"] == "ready_to_commit"
    assert counted(stats.snapshot(), "sync_session") == (1, {"degraded": 1}, {"degraded": {"reindex_ready_to_commit": 1}})
    slow_commit(monkeypatch, 18e-6)
    done = asyncio.run(server._dispatch(p.staged.sync_session, **dict(sync, budget_override=60)))
    assert reindex_of(done)["state"] == "committed"
    assert counted(stats.snapshot(), "sync_session") == (2, {"degraded": 1, "ok": 1}, {"degraded": {"reindex_ready_to_commit": 1}})


def test_a_staged_reindex_is_counted_in_progress_and_a_waiting_one_by_its_own_name():
    assert "reindex_ready_to_commit" in request_stats.REASONS and request_stats.REASONS[-1] == request_stats.OTHER
    assert request_stats.fold("reindex_ready_to_commit", request_stats.REASONS) == "reindex_ready_to_commit"
    unfinished = lambda *issues: server._unfinished_sync({"coverage": {"issues": list(issues)}})      # noqa: E731
    assert unfinished("parser_outdated", "sync_in_progress") == "sync_in_progress"
    assert unfinished("parser_outdated", "reindex_ready_to_commit") == "reindex_ready_to_commit"
    assert unfinished("sync_in_progress", "reindex_ready_to_commit") == "reindex_ready_to_commit"
    assert unfinished("reindex_ready_to_commit", "sync_deadline") == "sync_deadline"      # the older reason first


# --- a spool that no longer matches -------------------------------------------------------------------------------------

def assert_restarted(p, info, reason):
    """The spool was thrown away as a whole and the staging began again from byte 0; the index never moved."""
    staged = reindex_of(info)
    assert staged["state"] == "discarded" and staged["reason"] == reason, staged
    assert info["sync"]["state"] == "partial" and p.intact() and info["new_chunks"] == 0
    return staged


def converges(p, **kwargs):
    """Finish without limits and compare with the one-shot re-index of the source as it is now."""
    info = p.finish(**kwargs)
    assert reindex_of(info)["state"] == "committed" and not spool_files(p.staged)
    assert canonical(p.staged, p.provider, p.sid) == p.expected()
    return info


def test_a_source_changed_between_slices_discards_the_spool_and_starts_over(pair):
    p = pair()
    one = reindex_of(p.stage(cap=30))
    first = p.stage(cap=30, calls=2)
    assert reindex_of(first)["staged_through"] > one["staged_through"]
    edit_source(p.world.path)                                       # inside the head: the hash of the head changes
    p.reason = "source_changed"
    restarted = p.stage(cap=30)
    staged = assert_restarted(p, restarted, "source_changed")
    assert staged["rows"] == one["rows"] and staged["staged_through"] == one["staged_through"]     # from byte 0 again
    converges(p)


def test_a_change_in_the_middle_of_the_staged_range_discards_the_spool(pair):
    """Beyond the 64 KiB whose hash binds the head of the source: the hash of the staged prefix finds it."""
    p = pair(count=900)
    one = reindex_of(p.stage(cap=250))
    first = p.stage(cap=250, calls=2)
    through = reindex_of(first)["staged_through"]
    assert through > 70_000 and p.world.path.stat().st_size > through
    at = edit_source(p.world.path, after=66_000 / p.world.path.stat().st_size)
    assert 66_000 <= at < through
    p.reason = "source_changed"
    restarted = p.stage(cap=250)
    assert_restarted(p, restarted, "prefix_changed")
    assert reindex_of(restarted)["rows"] == one["rows"]
    converges(p)


def test_a_source_that_grew_since_is_continued_not_restarted(pair):
    """An append is no reason to throw the staged range away: the prefix hash still holds."""
    p = pair()
    first = p.stage(cap=40, calls=2)
    with p.world.path.open("ab") as stream:
        stream.write(b'{"type":"event_msg","payload":{"type":"agent_message","message":"appended later"}}\n' * 30)
    again = p.stage(cap=40)
    staged = reindex_of(again)
    assert staged["state"] == "staging" and staged["staged_through"] > reindex_of(first)["staged_through"]
    assert staged["rows"] == reindex_of(first)["rows"] + 40
    converges(p)


def test_a_target_parser_version_that_moved_on_discards_the_old_spool(pair, monkeypatch):
    p = pair()
    one = reindex_of(p.stage(cap=30))
    p.stage(cap=30)
    stale = spool_files(p.staged)
    assert len(stale) == 1 and stale[0].name.endswith(f".{codex_parser.PARSER_VERSION}.spool")
    monkeypatch.setattr(codex_parser, "PARSER_VERSION", codex_parser.PARSER_VERSION + 1)      # n -> n + 1
    info = p.stage(cap=30)
    assert_restarted(p, info, "target_version")
    assert not stale[0].exists() and len(spool_files(p.staged)) == 1
    assert spool_files(p.staged)[0].name.endswith(f".{codex_parser.PARSER_VERSION}.spool")
    assert reindex_of(info)["rows"] == one["rows"]
    converges(p)


@pytest.mark.parametrize("trigger", ["parser", "source"])
def test_a_second_process_that_finished_the_reindex_leaves_an_orphan_that_is_removed(pair, trigger):
    p = pair(trigger=trigger)
    p.stage(cap=30, calls=2)
    assert len(spool_files(p.staged)) == 1
    other = p.world.engine("another-process")                      # not the same index: copy it, then finish there
    # The same index directory, another engine: it re-indexes in one piece, without a deadline.
    from self_directing_mcp.config import Settings
    from self_directing_mcp.engine import SelfDirectEngine
    other = SelfDirectEngine(Settings(_env_file=None, local_only=True, openrouter_api_key_file=None,
                                      index_dir=p.staged.settings.index_dir, **p.world.paths))
    p.world.engines.append(other)
    done = other.sync_session(**p.world.sync_args(), derived=False)
    assert done["reindexed"] is True
    published = index_dump(other)
    info = p.stage(cap=30)                                         # this process's next sync: the index is current
    assert "reindex" not in info["sync"] and info["sync"]["state"] == "complete" and info["new_chunks"] == 0
    assert info["reindexed"] is False and not spool_files(p.staged)
    now = index_dump(p.staged)                                     # nothing of the old spool reached the index (only the
    assert {k: v for k, v in now.items() if k != "source_blocks"} == \
        {k: v for k, v in published.items() if k != "source_blocks"}      # verification stamp of the block table moved
    assert canonical(p.staged, "codex", p.sid) == p.expected()


def test_a_spool_staged_for_an_upgrade_is_an_orphan_once_the_index_has_the_target(pair):
    """The index reaches the target version some other way, then the source changes: the old spool starts from an index that is
    gone (its base version is not the stored one)."""
    p = pair()
    one = reindex_of(p.stage(cap=30))
    p.stage(cap=30)
    [spool] = spool_files(p.staged)
    left_behind = spool.read_bytes()
    from self_directing_mcp.config import Settings
    from self_directing_mcp.engine import SelfDirectEngine
    other = SelfDirectEngine(Settings(_env_file=None, local_only=True, openrouter_api_key_file=None,
                                      index_dir=p.staged.settings.index_dir, **p.world.paths))
    p.world.engines.append(other)
    other.sync_session(**p.world.sync_args(), derived=False)
    assert not spool_files(p.staged)             # a re-index that is published clears the spools of its session
    spool.write_bytes(left_behind)              # ... unless that process was an older build, or died before it could
    edit_source(p.world.path)
    p.reason = "source_changed"
    restarted = p.stage(cap=30)
    assert reindex_of(restarted)["state"] == "discarded" and reindex_of(restarted)["reason"] == "orphan"
    assert reindex_of(restarted)["rows"] == one["rows"]
    p.before = index_dump(p.staged)
    info = p.finish()
    assert reindex_of(info)["state"] == "committed" and not spool_files(p.staged)
    # Two re-indexes happened in this index (the other process's, then this one): the generation counts them, the rows are alike.
    done, expected = canonical(p.staged, "codex", p.sid), p.expected()
    assert done["generation"] == expected["generation"] + 1
    for table in (done, expected):
        table.pop("generation"), table.pop("commit")
    assert done == expected


def test_a_cursor_rewritten_since_the_spool_started_discards_it(pair):
    p = pair()
    one = reindex_of(p.stage(cap=30))
    p.stage(cap=30)
    p.staged.store._conn.execute("UPDATE session_cursors SET synced_at='2000-01-01T00:00:00+00:00'")
    p.staged.store._conn.execute("UPDATE session_parser_versions SET synced_at='2000-01-01T00:00:00+00:00'")
    p.staged.store._conn.commit()
    restarted = p.stage(cap=30)
    staged = reindex_of(restarted)
    assert staged["state"] == "discarded" and staged["reason"] == "index_changed" and staged["rows"] == one["rows"]
    p.before = index_dump(p.staged)
    converges(p)


def test_a_missing_volume_room_discards_the_spool_and_the_sync_is_the_old_one(pair, monkeypatch):
    p = pair()
    p.stage(cap=30, calls=2)
    assert len(spool_files(p.staged)) == 1
    monkeypatch.setattr(reindex_spool, "free_space", lambda path: 1000)         # the index volume is short of space
    info = p.finish()
    staged = reindex_of(info)
    assert staged == {"state": "disabled", "staged_through": 0, "target_bytes": info["sync"]["target_bytes"], "rows": 0,
                      "reason": "disk"}
    assert info["reindexed"] is True and info["sync"]["state"] == "complete"      # all or nothing, as before
    assert not spool_files(p.staged)
    assert canonical(p.staged, "codex", p.sid) == p.expected()


def test_without_room_a_spool_is_not_even_started(pair, monkeypatch):
    p = pair()
    monkeypatch.setattr(reindex_spool, "free_space", lambda path: 1000)
    info = p.stage(cap=30)
    assert reindex_of(info)["state"] == "disabled" and reindex_of(info)["reason"] == "disk"
    assert info["sync"]["state"] == "complete" and info["reindexed"] is True          # the forced stop is ignored, as before
    assert not spool_files(p.staged)


def test_a_symlinked_spool_directory_turns_staging_off_for_the_request(pair, tmp_path):
    p = pair()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (p.staged.settings.index_dir / "reindex").symlink_to(elsewhere, target_is_directory=True)
    info = p.stage(cap=30)
    assert reindex_of(info)["state"] == "disabled" and reindex_of(info)["reason"] == "unsafe"
    assert info["reindexed"] is True and not list(elsewhere.iterdir())
    assert canonical(p.staged, "codex", p.sid) == p.expected()


def test_a_spool_that_cannot_be_read_is_removed_by_any_sync_of_its_session(pair):
    p = pair()
    one = reindex_of(p.stage(cap=30))
    [spool] = spool_files(p.staged)
    spool.write_bytes(b"\x00" * 64)                               # a header that cannot be read
    info = p.stage(cap=30)
    assert reindex_of(info)["state"] == "discarded" and reindex_of(info)["reason"] == "unreadable"
    assert reindex_of(info)["rows"] == one["rows"]
    converges(p)


def test_a_leftover_after_the_commit_is_removed_by_the_next_sync(pair):
    """A crash between the commit and the unlink: the spool is complete and the index has it. Nothing else will use it."""
    p = pair()
    p.stage(cap=40, calls=3)
    [spool] = spool_files(p.staged)
    kept = spool.read_bytes()
    converges(p)
    spool.write_bytes(kept)
    assert len(spool_files(p.staged)) == 1
    again = p.staged.sync_session(**p.world.sync_args(), derived=False, _deadline=far())
    assert again["sync"]["state"] == "complete" and "reindex" not in again["sync"] and not spool_files(p.staged)


def test_a_spool_of_another_session_is_never_touched(pair):
    p = pair()
    p.stage(cap=30)
    [mine] = spool_files(p.staged)
    foreign = mine.with_name("codex-" + "0" * 32 + ".9.spool")
    foreign.write_bytes(b"another session's rows")
    p.stage(cap=30)
    p.finish()
    assert foreign.exists() and foreign.read_bytes() == b"another session's rows" and not mine.exists()


# --- AGY: the receipts a spool was parsed with ---------------------------------------------------------------------------

def test_changed_receipts_of_a_staged_results_discard_the_spool(pair):
    p = pair("agy", count=110)
    p.stage(cap=40, calls=2)
    through = reindex_of(p.stage(cap=1))["staged_through"]
    assert through > 0
    from self_directing_mcp.agy_receipts import ReceiptStore
    import json
    rows = {step: json.loads(data) for step, data in ReceiptStore(p.staged.settings.index_dir).read_capture(p.sid)[0]}
    step, row = next((s, r) for s, r in sorted(rows.items()) if r.get("paired") and r.get("result_hash"))
    for engine in (p.staged, p.ref):
        engine.record_agy_hook({"conversationId": p.sid, "transcriptPath": str(p.world.path), "workspacePaths": ["/synthetic"],
                                "stepIdx": step, "toolCall": json.loads(row["action"]), "error": "reported later"},
                               "PostToolUse")
    p.before = index_dump(p.staged)
    restarted = p.stage(cap=40)
    assert reindex_of(restarted)["state"] == "discarded" and reindex_of(restarted)["reason"] == "receipts_changed"
    converges(p)


def test_a_receipt_for_a_step_that_is_not_staged_yet_does_not_restart_the_staging(pair):
    """A session that is still being written gets receipts all the time: only those its staged events were derived from count."""
    p = pair("agy", count=110)
    first = p.stage(cap=30, calls=2)
    import json
    from self_directing_mcp.agy_receipts import ReceiptStore
    rows = {step: json.loads(data) for step, data in ReceiptStore(p.staged.settings.index_dir).read_capture(p.sid)[0]}
    step, row = [(s, r) for s, r in sorted(rows.items()) if r.get("paired") and r.get("result_hash")][-1]     # the last one
    assert row["prefix_length"] > reindex_of(first)["staged_through"]            # its result is not staged yet
    for engine in (p.staged, p.ref):
        engine.record_agy_hook({"conversationId": p.sid, "transcriptPath": str(p.world.path), "workspacePaths": ["/synthetic"],
                                "stepIdx": step, "toolCall": json.loads(row["action"]), "error": "reported later"},
                               "PostToolUse")
    again = p.stage(cap=30)
    staged = reindex_of(again)
    assert staged["state"] == "staging" and staged["rows"] > reindex_of(first)["rows"]       # continued, not restarted
    converges(p)


# --- torn writes and reruns, through the engine -------------------------------------------------------------------------

def test_a_torn_spool_resumes_from_its_last_trailer_without_duplicates_or_gaps(pair):
    p = pair()
    first = reindex_of(p.stage(cap=30, calls=2))
    [spool] = spool_files(p.staged)
    data = spool.read_bytes()
    spool.write_bytes(data[:-17])                                  # cut inside the trailer of the second slice
    staged = reindex_of(p.stage(cap=30))
    # Slice two was parsed again and staged once: the same range and the same rows as before the cut, no more.
    assert staged["state"] == "staging" and (staged["rows"], staged["staged_through"]) == (first["rows"], first["staged_through"])
    assert p.intact()
    converges(p)


def test_a_slice_that_is_staged_twice_is_in_the_spool_once(pair):
    """The file goes back to how it was after the first slice (the second never reached the disk): the second is parsed again."""
    p = pair()
    one = reindex_of(p.stage(cap=30))
    [spool] = spool_files(p.staged)
    after_one = spool.read_bytes()
    second = reindex_of(p.stage(cap=30))
    spool.write_bytes(after_one)
    again = reindex_of(p.stage(cap=30))
    assert second["rows"] > one["rows"]
    assert (again["staged_through"], again["rows"]) == (second["staged_through"], second["rows"])
    converges(p)


# --- hygiene --------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not POSIX, reason="POSIX permission bits")
def test_the_spool_and_its_directory_are_private(pair):
    p = pair()
    p.stage(cap=30)
    directory = p.staged.settings.index_dir / "reindex"
    [spool] = spool_files(p.staged)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700 and stat.S_IMODE(spool.stat().st_mode) == 0o600


def test_no_spool_is_left_after_a_success_or_a_discard(pair, monkeypatch):
    p = pair()
    p.stage(cap=30, calls=2)
    edit_source(p.world.path)
    p.reason = "source_changed"
    p.stage(cap=30)
    monkeypatch.setattr(reindex_spool, "free_space", lambda path: 1)
    p.finish()                                                    # the discard (disk) and the all-or-nothing re-index
    assert not spool_files(p.staged)
    assert not list((p.staged.settings.index_dir / "reindex").iterdir())


def test_the_spool_holds_no_row_text_in_a_log(pair, caplog):
    p = pair()
    with caplog.at_level("DEBUG"):
        p.stage(cap=30, calls=2)
    assert "alpha" not in caplog.text and "event" not in caplog.text.lower().replace("events", "")


def test_nothing_is_staged_when_slicing_is_off(pair):
    p = pair()
    p.staged.slice_policy = SlicePolicy(mode="off")
    info = p.staged.sync_session(**p.world.sync_args(), derived=False, _deadline=far())
    assert "sync" not in info and info["reindexed"] is True
    assert not (p.staged.settings.index_dir / "reindex").exists()


def test_nothing_is_staged_without_a_finite_deadline(pair):
    p = pair()
    p.staged.slice_policy = forced(10)
    info = p.staged.sync_session(**p.world.sync_args(), derived=False)
    assert "sync" not in info and info["reindexed"] is True
    assert not (p.staged.settings.index_dir / "reindex").exists()


def test_nothing_is_staged_for_a_compressed_source(tmp_path, monkeypatch):
    from test_zstd_stream import BACKENDS, make_frame
    if not BACKENDS:
        pytest.skip("needs Python 3.14+ or the zstandard package")
    world = World(tmp_path, monkeypatch, "codex", seed=2, count=300, partial=False)
    try:
        packed = world.path.with_name(world.path.name + ".zst")
        packed.write_bytes(make_frame(world.path.read_bytes()))
        world.path.unlink()
        engine = world.engine("zst")
        with old_parser("codex"):
            engine.sync_session(session_id=world.sid, provider="codex", path=str(packed), embed=False, derived=False)
        engine.slice_policy = forced(3, cap_bytes=1000)
        info = engine.sync_session(session_id=world.sid, provider="codex", path=str(packed), embed=False, derived=False,
                                   _deadline=far())
        assert info["reindexed"] is True and info["sync"]["state"] == "complete" and "reindex" not in info["sync"]
        assert not (engine.settings.index_dir / "reindex").exists()
    finally:
        world.close()


def test_nothing_is_staged_when_the_sync_is_not_a_reindex(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch, "codex", seed=6, count=200, partial=False)
    try:
        engine = world.engine("plain")
        engine.slice_policy = forced(25)
        cold = engine.sync_session(**world.sync_args(), derived=False, _deadline=far())          # a cold sync, in slices
        assert cold["sync"]["state"] == "partial" and "reindex" not in cold["sync"]
        engine.slice_policy = None
        engine.sync_session(**world.sync_args(), derived=False, _deadline=far())
        with world.path.open("ab") as stream:
            stream.write(b'{"type":"event_msg","payload":{"type":"agent_message","message":"more"}}\n' * 80)
        engine.slice_policy = forced(25)
        append = engine.sync_session(**world.sync_args(), derived=False, _deadline=far())        # an append, in slices
        assert append["sync"]["state"] == "partial" and "reindex" not in append["sync"]
        assert not (engine.settings.index_dir / "reindex").exists()
    finally:
        world.close()


# --- one writer ---------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not POSIX, reason="the lock of Windows is not exercised here")
def test_a_spool_held_by_another_process_is_judged_around_not_waited_for(pair):
    p = pair()
    p.stage(cap=30)
    [spool] = spool_files(p.staged)
    descriptor = os.open(spool, os.O_RDWR)
    import fcntl
    fcntl.flock(descriptor, fcntl.LOCK_EX)                         # what another process that is staging this session holds
    try:
        with pytest.raises(FlightBusy):
            p.staged.sync_session(**p.world.sync_args(), derived=False, _deadline=far())
        audit = p.staged.audit_session(p.sid, provider="codex", path=str(p.world.path), _deadline=far())
        assert "sync_in_progress" in audit["coverage"]["issues"] and audit["verdict"] == "violation"
        assert p.intact()
    finally:
        os.close(descriptor)
    assert reindex_of(p.stage(cap=30))["state"] == "staging"


# --- the staged range is verified by the one pass of the capture ------------------------------------------------------------

def test_the_capture_hashes_the_staged_range_on_its_one_pass_instead_of_reading_it_again(pair, monkeypatch):
    """A resumed staging asks the capture for the hash state at the offset the spool ends at (a hint read from the spool before
    the capture): both the parse guard and the check of the spool's prefix hash then cost no second read of the source."""
    from self_directing_mcp.session_events import SourceObservation
    p = pair(count=2500)
    assert reindex_of(p.stage(cap=250, calls=3))["state"] == "staging"
    opens = []
    original = SourceObservation._open
    monkeypatch.setattr(SourceObservation, "_open", lambda self: opens.append(1) or original(self))
    assert reindex_of(p.stage(cap=250))["state"] == "staging"
    with_hint = len(opens)
    opens.clear()
    monkeypatch.setattr(reindex_spool.ReindexSpool, "peek", classmethod(lambda cls, *args, **kwargs: None))
    assert reindex_of(p.stage(cap=250))["state"] == "staging"
    without_hint = len(opens)
    assert without_hint >= with_hint + 2          # the head of the block and the mark of the offset are read again without it
    monkeypatch.undo()
    converges(p)


def test_a_spool_directory_that_cannot_be_read_turns_staging_off_for_the_request(pair):
    if not POSIX or os.geteuid() == 0:
        pytest.skip("needs a user that permissions apply to")
    p = pair()
    p.stage(cap=30)
    directory = p.staged.settings.index_dir / "reindex"
    os.chmod(directory, 0)
    try:
        info = p.stage(cap=30)
        assert reindex_of(info)["state"] == "disabled" and reindex_of(info)["reason"] in ("unavailable", "unsafe")
        assert info["reindexed"] is True and info["sync"]["state"] == "complete"      # the re-index of before
    finally:
        os.chmod(directory, 0o700)


@pytest.mark.skipif(not POSIX, reason="the lock of Windows is not exercised here")
def test_a_request_that_fails_before_its_parse_does_not_keep_the_spool_locked(pair, monkeypatch):
    """The lock is the file descriptor: whatever ends the request, it goes with it."""
    p = pair()
    p.stage(cap=30)
    assert spool_files(p.staged)

    def no_room_for_a_temporary_file(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ingest, "RowSpool", no_room_for_a_temporary_file)
    with pytest.raises(OSError):
        p.stage(cap=30)
    monkeypatch.undo()
    assert reindex_of(p.stage(cap=30))["state"] == "staging"        # not FlightBusy: nobody holds the spool
    assert p.intact()
