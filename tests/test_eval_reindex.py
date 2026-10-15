"""A staged re-index and the cost bound of an audit do not size each other (roundtable 8, PR3 and PR4).

An audit that syncs a session and then judges it sizes the slice it syncs by what judging that slice will cost
(``slice_budget.Sizing``, ``SliceBudget.cost_budget``): a slice that is APPENDED to the index is what the evaluation that follows
reads. A slice of a RE-INDEX is not: it waits in a spool, the index stays what it was, and the evaluation reads the old index.

* A staged slice is sized by the share alone. It never asks for the sizing numbers (neither the contracts nor the index are read for
  a limit it would not use), says ``eval_us`` 0, ``limited_by`` is ``share`` or ``null`` and never ``cost``, and its ``sync`` object
  carries the ``reindex`` report next to ``cost``.
* The slice appended to the same session once the re-index is published is cost-sized as before, and the published re-index has
  dropped the monitor states (a new generation), so the first audit after it builds them.
* Neither a staged slice nor the commit that publishes it teaches the cost model what an appended event costs.
* A request that returns a partial re-index has an answer, and the idle worker is told about a stopped evaluation only: a slice that
  merely came back partial, or a deadline that ended the sync before anything was judged, is not noted.

The slices are cut by a scripted clock (``slicing_support.Tick``) or by counts, and the hard deadline is an hour away, so no wall
clock decides anything: the tests do not depend on the speed of the machine.
"""
from __future__ import annotations

import types

import pytest

from convergence_support import DEPLOY, SID, Session
from reindex_support import old_parser, spool_files
from self_directing_mcp import eval_cost, reindex_spool
from self_directing_mcp.audit import incremental
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.eval_cost import COSTS
from self_directing_mcp.index import ingest
from self_directing_mcp.index.ingest import SlicePolicy
from self_directing_mcp.request_control import RequestStopped, request_scope
from self_directing_mcp.shadow_worker import ShadowWorker
from self_directing_mcp.slice_budget import Sizing
from slicing_support import Tick, forced, scripted_deadline
from test_slice_convergence import Tripwire

EVENTS = 1200


@pytest.fixture
def stale(tmp_path):
    """A session whose index was made by the parser of the version before: its next sync is a re-index. Five contracts, four of
    them judge history, so a request that judges what it syncs has numbers to size by."""
    session = Session(tmp_path, EVENTS)
    engine = session.engine("stale")
    with old_parser("codex"):
        engine.sync_session(session_id=SID, provider="codex", path=str(session.path), embed=False, derived=False)
    try:
        yield session, engine
    finally:
        session.close()


@pytest.fixture
def sized(monkeypatch):
    """What a request asked its sizing for: ``resolved`` counts the slices that consulted the ``Sizing`` of the request,
    ``numbers`` the reads of the contracts and the index that fill it in. Both delegate to the real thing."""
    seen = types.SimpleNamespace(resolved=0, numbers=0)
    resolve, numbers = Sizing.resolve, SelfDirectEngine._sizing_numbers

    def resolving(self):
        seen.resolved += 1
        return resolve(self)

    def reading(engine, *args):
        seen.numbers += 1
        return numbers(engine, *args)

    monkeypatch.setattr(Sizing, "resolve", resolving)
    monkeypatch.setattr(SelfDirectEngine, "_sizing_numbers", reading)
    return seen


def hard_deadline(monkeypatch, nominal=3.0):
    """The deadline of every call of a test whose slices a scripted clock cuts (an hour away) and, in ``SYNC_SHARE``, the soft
    deadline of a ``nominal``-second request in scripted time. Called once per test."""
    return scripted_deadline(monkeypatch, nominal)


def scripted(step, cap=None):
    """A slice policy whose clock moves ``step`` seconds at every look (every 64 records). The minimum progress is waived."""
    return SlicePolicy(mode="auto", clock=Tick(step), min_records=1, min_bytes=0, max_records=cap)


def args(session):
    return dict(session_id=SID, provider="codex", path=str(session.path), embed=False, derived=False)


def lag(engine):
    return engine.audit_status(SID, provider="codex")["evaluation"]["session"]


# --- a staged slice is not sized by judging it -------------------------------------------------------------------------------

@pytest.mark.parametrize("tool", ["audit", "check"])
@pytest.mark.parametrize("stop", ["share", "record_cap", "commit_waits"])
def test_a_staged_slice_is_sized_by_the_share_alone(stale, sized, monkeypatch, stop, tool):
    """Whatever ended the slice (the soft deadline, a cap, or the end of the source with a commit that does not fit), the sizing of
    the request was not consulted, and the report says so: nothing is judged after a staged slice."""
    session, engine = stale
    deadline = hard_deadline(monkeypatch)
    if stop == "share":
        engine.slice_policy = scripted(0.3)                                    # the soft deadline passes at the 4th look
    elif stop == "record_cap":
        engine.slice_policy = scripted(0.0001, cap=100)
    else:
        engine.slice_policy = scripted(0.0001)                                 # parsed to the end in one call, and
        monkeypatch.setattr(ingest, "COMMIT_SECONDS_PER_ROW", 10.0)            # a commit that cannot fit the time left
    kwargs = dict(provider="codex", path=str(session.path), _deadline=deadline())
    answer = engine.audit_session(SID, **kwargs) if tool == "audit" else engine.check_action(SID, DEPLOY, **kwargs)
    sync = answer["sync"]
    waiting = "ready_to_commit" if stop == "commit_waits" else "staging"
    assert sync["state"] == "partial" and sync["reindex"]["state"] == waiting          # the re-index report ...
    assert sync["stopped_by"] == {"share": "soft_deadline", "record_cap": "record_cap", "commit_waits": "eof"}[stop]
    assert sync["cost"] == {"slice_us": 40.0, "eval_us": 0.0, "debt_events": 0,        # ... and the cost report, side by side
                            "limited_by": "share" if stop == "share" else None}
    assert (sized.resolved, sized.numbers) == (0, 0)                  # the contracts and the index were not read for numbers
    assert answer["coverage"]["complete"] is False                     # and the answer judged the old index: never clean
    issues = set(answer["coverage"]["issues"])                         # the coverage issues are the re-index's own (PR3)
    assert "parser_outdated" in issues
    assert ("reindex_ready_to_commit" in issues and "sync_in_progress" not in issues) if stop == "commit_waits" \
        else ("sync_in_progress" in issues and "reindex_ready_to_commit" not in issues)


def test_the_same_session_is_staged_then_appended_and_only_the_append_is_cost_sized(stale, sized, monkeypatch):
    session, engine = stale
    deadline = hard_deadline(monkeypatch)
    # 1. Audits stage the re-index slice by slice; the last one cannot commit what it staged (the commit would not fit).
    fits = ingest.COMMIT_SECONDS_PER_ROW
    monkeypatch.setattr(ingest, "COMMIT_SECONDS_PER_ROW", 10.0)
    engine.slice_policy = scripted(0.3)
    answers = []
    while not answers or answers[-1]["sync"]["reindex"]["state"] == "staging":
        answers.append(engine.audit_session(SID, provider="codex", path=str(session.path), _deadline=deadline()))
        assert len(answers) < 40
    assert len(answers) >= 3 and answers[-1]["sync"]["reindex"]["state"] == "ready_to_commit"
    assert all(a["sync"]["cost"]["eval_us"] == 0.0 and a["sync"]["cost"]["limited_by"] in ("share", None) for a in answers)
    assert (sized.resolved, sized.numbers) == (0, 0)                  # not one of the calls asked
    # 2. A call with the time publishes it. The new generation has no monitor state: every event is owed again.
    monkeypatch.setattr(ingest, "COMMIT_SECONDS_PER_ROW", fits)
    engine.slice_policy = None
    published = engine.sync_session(**args(session), _deadline=deadline())
    assert published["reindexed"] is True and published["sync"]["reindex"]["state"] == "committed"
    owed = lag(engine)
    assert owed["events_without_state"] == owed["events_indexed"] > 1000 and owed["states_through_byte"] is None
    # 3. The first audit after it is the state-less case: nothing is cut (a session this small is below the minimum progress of a
    # slice), the states are built in the call, and the sizing was still not asked.
    answer = engine.audit_session(SID, provider="codex", path=str(session.path), _deadline=deadline())
    assert answer["coverage"]["complete"] and "sync" not in answer and lag(engine)["events_without_state"] == 0
    assert (sized.resolved, sized.numbers) == (0, 0)
    # 4. What is appended to it afterwards is judged by the call that syncs it, and its slice is sized by that.
    session.append(600)
    prior = 20.0                                                      # seconds per event: the cost bound binds after ~100 records
    monkeypatch.setattr(eval_cost, "DEFAULT_EVAL_SEC", prior)
    engine.slice_policy = scripted(0.011)                             # (the soft deadline of 3 scripted seconds is far)
    answer = engine.audit_session(SID, provider="codex", path=str(session.path), _deadline=deadline())
    sync = answer["sync"]
    assert sync["state"] == "partial" and "reindex" not in sync and sync["stopped_by"] == "soft_deadline"
    assert sync["cost"]["limited_by"] == "cost" and sync["cost"]["debt_events"] == 0
    assert sync["cost"]["eval_us"] == pytest.approx(prior * eval_cost.UNMEASURED_MARGIN * 1e6)
    assert 0 < sync["indexed_through"] < sync["target_bytes"]
    assert (sized.resolved, sized.numbers) == (1, 1)                  # asked once, by the slice that is judged


def test_a_staged_slice_and_the_commit_that_publishes_it_teach_the_cost_model_nothing(stale, monkeypatch):
    """The model learns what parsing and committing an APPENDED event costs. The commit of a re-index replaces the session in one
    transaction, and the parse of its slices happened in other calls: dividing one by the other measures nothing."""
    monkeypatch.setattr(eval_cost, "MIN_SAMPLE_EVENTS", 1)          # every slice is big enough to be a measurement here
    session, engine = stale
    deadline = hard_deadline(monkeypatch)
    engine.slice_policy = scripted(0.3)
    for _ in range(40):
        info = engine.sync_session(**args(session), _deadline=deadline())
        if info["sync"]["state"] == "complete":
            break
    assert info["reindexed"] is True and info["sync"]["reindex"]["state"] == "committed"
    assert COSTS.snapshot()["slice_us"] == {}
    session.append(300)                                             # the control: a slice that is appended is measured
    engine.sync_session(**args(session), _deadline=deadline())
    assert list(COSTS.snapshot()["slice_us"]) == ["codex"]


# --- the idle worker ---------------------------------------------------------------------------------------------------------

def attach_worker(engine):
    worker = ShadowWorker(engine, idle_sec=0.0, poll_sec=0.01)
    engine.attach_shadow_worker(worker)
    return worker


def test_a_partial_reindex_is_an_answer_and_is_not_noted_to_the_idle_worker(stale, monkeypatch):
    session, engine = stale
    worker = attach_worker(engine)
    deadline = hard_deadline(monkeypatch)
    engine.slice_policy = scripted(0.3)
    answer = engine.audit_session(SID, provider="codex", path=str(session.path), _deadline=deadline())
    assert answer["sync"]["reindex"]["state"] == "staging" and answer["verdict"] == "violation"    # the old index, judged
    assert worker.snapshot()["pending_sessions"] == 0


def test_a_request_stopped_while_it_stages_loses_its_answer_and_is_not_noted(stale, monkeypatch):
    """The deadline ends the sync, not an evaluation: nothing was judged, so there is no stopped evaluation for the idle worker
    to finish. The slice that was staged is kept."""
    session, engine = stale
    worker = attach_worker(engine)
    engine.slice_policy = forced(40)
    control = Tripwire()
    append = reindex_spool.ReindexSpool.append_slice

    def appending(self, *a, **k):
        staged = append(self, *a, **k)
        control.armed = True                                        # the first check after the slice is a stop
        return staged

    monkeypatch.setattr(reindex_spool.ReindexSpool, "append_slice", appending)
    with pytest.raises(RequestStopped) as stopped:
        with request_scope(control):
            engine.audit_session(SID, provider="codex", path=str(session.path))
    assert stopped.value.reason == "request_deadline"
    assert worker.snapshot()["pending_sessions"] == 0 and len(spool_files(engine)) == 1


def test_a_deadline_that_ends_the_evaluation_of_the_old_index_is_still_noted(stale, monkeypatch):
    """The control of the two above: in this very setup the note does fire when the evaluation after a staged slice is the one
    the deadline cuts (the idle audit syncs first, so it goes on with the re-index before it judges)."""
    session, engine = stale
    worker = attach_worker(engine)
    engine.slice_policy = forced(40)
    control = Tripwire()
    prepare = incremental.AuditPlan.prepare

    def preparing(cls, *a, **k):
        control.armed = True                                        # the sync returned partial; the evaluation begins: stopped
        return prepare(*a, **k)

    monkeypatch.setattr(incremental.AuditPlan, "prepare", classmethod(preparing))
    with pytest.raises(RequestStopped) as stopped:
        with request_scope(control):
            engine.audit_session(SID, provider="codex", path=str(session.path))
    assert stopped.value.reason == "request_deadline" and worker.snapshot()["pending_sessions"] == 1
