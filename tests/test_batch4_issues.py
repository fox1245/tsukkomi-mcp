from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.index.ingest import (
    FlightBusy,
    session_flight,
    update_session_progress,
)


class _SlowGateEmbedder:
    """Deterministic fake embedder that pauses in embed_documents until released."""

    dim = 4
    cache_key = "slow-gate-v1"

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        self.calls += 1
        self.started.set()
        self.release.wait(timeout=10.0)
        return [np.ones(self.dim, dtype=np.float32) for _ in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return np.ones(self.dim, dtype=np.float32)


def _write_codex_rollout(path: Path, session_id: str, turns: list[tuple[str, str, str]]) -> None:
    """Write a minimal canonical Codex rollout JSONL file with user/assistant messages and tool calls."""
    lines = [
        json.dumps(
            {
                "timestamp": "2026-10-24T10:00:00Z",
                "type": "session_meta",
                "payload": {"id": session_id, "cwd": "/tmp"},
            }
        )
    ]
    for idx, (kind, role, text) in enumerate(turns, start=1):
        ts = f"2026-10-24T10:00:{idx:02d}Z"
        if kind == "message":
            lines.append(
                json.dumps(
                    {
                        "timestamp": ts,
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": role,
                            "content": [
                                {
                                    "type": "input_text" if role == "user" else "output_text",
                                    "text": text,
                                }
                            ],
                        },
                    }
                )
            )
        elif kind == "tool_call":
            lines.append(
                json.dumps(
                    {
                        "timestamp": ts,
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "Bash",
                            "call_id": f"call_{idx}",
                            "arguments": json.dumps({"command": text}),
                        },
                    }
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_issue51_embed_sync_does_not_block_sparse_or_readonly_tools(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    index_dir = tmp_path / "index"
    sid = "11111111-2222-3333-4444-555555555555"
    rollout = sessions_dir / f"rollout-2026-10-24T10-00-00-{sid}.jsonl"
    _write_codex_rollout(
        rollout,
        sid,
        [
            ("message", "user", "Please inspect the alpha_beta_gamma pipeline metrics."),
            ("message", "assistant", "Checking alpha_beta_gamma pipeline status now."),
        ],
    )

    settings = Settings(
        codex_sessions_dir=sessions_dir,
        index_dir=index_dir,
        local_only=False,
        use_fake_embedder=True,
        embedding_dim=4,
        _env_file=None,
    )
    engine = SelfDirectEngine(settings)
    slow_embedder = _SlowGateEmbedder()
    with engine._components():
        engine.embedder = slow_embedder
        engine.retriever.embedder = slow_embedder

    # Start async sync_session(embed=True); Phase 1 commits canonical + sparse, then Phase 2 pauses in slow_embedder
    job = engine.sync_session(
        session_id=sid,
        path=str(rollout),
        provider="codex",
        embed=True,
        async_mode=True,
    )
    assert job["ok"] is True
    assert job["async"] is True
    job_id = job["job_id"]

    assert slow_embedder.started.wait(timeout=5.0), "Phase 2 embedding did not start in time"

    try:
        # 1. job_status reports running + stage='embedding'
        status_running = engine.job_status(job_id=job_id)
        assert status_running["ok"] is True
        assert status_running["status"] == "running"
        assert status_running["sync_progress"] is not None
        assert status_running["sync_progress"]["stage"] == "embedding"
        assert status_running["sync_progress"]["embedded_total"] >= 1

        # 1b. Duplicate async_mode sync_session coalesces onto the same running job_id
        job_dup = engine.sync_session(
            session_id=sid,
            path=str(rollout),
            provider="codex",
            embed=True,
            async_mode=True,
        )
        assert job_dup["ok"] is True
        assert job_dup["coalesced"] is True
        assert job_dup["job_id"] == job_id

        # 2. Concurrent sync_session(embed=False) completes immediately without FlightBusy
        t0 = time.monotonic()
        fast_sync = engine.sync_session(
            session_id=sid,
            path=str(rollout),
            provider="codex",
            embed=False,
        )
        elapsed_fast = time.monotonic() - t0
        assert elapsed_fast < 1.0
        assert fast_sync["ok"] is True
        assert fast_sync["total_chunks"] >= 2
        assert fast_sync.get("sync_progress", {}).get("stage") == "embedding"

        # 3. Concurrent sparse search_history completes immediately from committed Phase 1 snapshot
        search_res = engine.search_history(
            "alpha_beta_gamma",
            session_id=sid,
            mode="sparse",
            provider="codex",
        )
        assert len(search_res["hits"]) >= 1
        assert search_res.get("sync_in_progress") is True
        assert search_res.get("sync_progress", {}).get("stage") == "embedding"

        # 4. Concurrent read-only tools (list_contracts, get_checklist, get_graph_context) complete immediately
        contracts_res = engine.list_contracts(session_id=sid)
        assert "contracts" in contracts_res

        checklist_res = engine.get_checklist(sid, provider="codex")
        assert checklist_res["session_id"] == sid

        graph_res = engine.get_graph_context(sid, provider="codex", path=str(rollout))
        assert graph_res["ok"] is True
    finally:
        slow_embedder.release.set()

    # Wait for background job to finish
    for _ in range(50):
        final_status = engine.job_status(job_id=job_id)
        if final_status["status"] != "running":
            break
        time.sleep(0.05)
    assert final_status["status"] == "completed"
    assert final_status["result"]["ok"] is True
    assert final_status["result"]["embedded"] >= 1
    engine.close()


def test_issue51_flight_busy_attaches_progress() -> None:
    key = ("/tmp/idx", "codex", "session", "sid-busy-test")
    update_session_progress(
        "sid-busy-test",
        "codex",
        flight_key=key,
        stage="parsing",
        bytes_done=2048,
        bytes_total=8192,
        chunks_done=10,
        chunks_total=40,
    )
    released = threading.Event()
    acquired = threading.Event()

    def _holder() -> None:
        with session_flight(key):
            acquired.set()
            released.wait(timeout=5.0)

    t = threading.Thread(target=_holder, daemon=True)
    t.start()
    assert acquired.wait(timeout=2.0)
    try:
        with pytest.raises(FlightBusy) as exc_info:
            with session_flight(key, max_wait=0.06):
                pass
        assert exc_info.value.progress is not None
        assert exc_info.value.progress["stage"] == "parsing"
        assert exc_info.value.progress["bytes_done"] == 2048
        assert exc_info.value.progress["bytes_total"] == 8192
        assert exc_info.value.progress["eta_ms"] is not None
    finally:
        released.set()
        t.join(timeout=2.0)
        update_session_progress("sid-busy-test", "codex", flight_key=key, clear=True, force=True)


def test_issue56_grant_once_validation_and_single_use_lifecycle(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    index_dir = tmp_path / "index"
    sid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    rollout = sessions_dir / f"rollout-2026-10-24T10-00-00-{sid}.jsonl"
    _write_codex_rollout(
        rollout,
        sid,
        [
            ("message", "user", "Please fix the failing unit tests first."),
            ("message", "user", "You have my explicit permission to run drop_staging_cache --force once."),
            ("message", "assistant", "Understood, I will request a one-time grant anchored to your message."),
        ],
    )

    settings = Settings(codex_sessions_dir=sessions_dir, index_dir=index_dir, local_only=True, _env_file=None)
    engine = SelfDirectEngine(settings)
    engine.sync_session(session_id=sid, path=str(rollout), provider="codex", embed=False)

    with engine._components():
        chunks = engine.store.list_chunks(sid, "codex")
    user_chunks = [c for c in chunks if c.kind in ("turn", "message") and c.meta.get("role") == "user"]
    unrelated_user_chunk = user_chunks[0]
    user_chunk = user_chunks[1]
    asst_chunk = next(c for c in chunks if c.kind in ("turn", "message") and c.meta.get("role") == "assistant")

    engine.upsert_contracts(
        [
            {
                "id": "no-drop-staging-cache",
                "description": "Prohibit drop_staging_cache --force without user grant",
                "type": "must_not",
                "severity": "critical",
                "regex": r"drop_staging_cache\s+--force",
                "scope": "tool_call",
                "session_id": sid,
                "source_event_id": user_chunk.chunk_id,
                "source_quote": user_chunk.text,
            },
            {
                "id": "must-run-tests",
                "description": "Run pytest",
                "type": "must",
                "severity": "medium",
                "regex": r"pytest",
                "scope": "tool_call",
                "session_id": sid,
                "source_event_id": user_chunk.chunk_id,
                "source_quote": user_chunk.text,
            },
        ]
    )

    # 1. Validation rejections (missing anchor, non-user anchor, unrelated user anchor, non-must_not, max_uses != 1, catch-all/wildcard/alternation regex, excessive TTL)
    res_no_anchor = engine.grant_once(
        "no-drop-staging-cache",
        r"drop_staging_cache\s+--force",
        "",
        session_id=sid,
        provider="codex",
    )
    assert res_no_anchor["ok"] is False
    assert res_no_anchor["error"] == "anchor_required"

    res_bad_anchor = engine.grant_once(
        "no-drop-staging-cache",
        r"drop_staging_cache\s+--force",
        asst_chunk.chunk_id,
        session_id=sid,
        provider="codex",
    )
    assert res_bad_anchor["ok"] is False
    assert res_bad_anchor["error"] == "anchor_not_user"

    # Unrelated user turn ("Please fix the failing unit tests first.") cannot self-authorize drop_staging_cache --force!
    res_unrelated_anchor = engine.grant_once(
        "no-drop-staging-cache",
        r"drop_staging_cache\s+--force",
        unrelated_user_chunk.chunk_id,
        session_id=sid,
        provider="codex",
    )
    assert res_unrelated_anchor["ok"] is False
    assert res_unrelated_anchor["error"] == "anchor_does_not_approve_action"

    res_wrong_type = engine.grant_once(
        "must-run-tests",
        r"pytest\s+tests",
        user_chunk.chunk_id,
        session_id=sid,
        provider="codex",
    )
    assert res_wrong_type["ok"] is False
    assert res_wrong_type["error"] == "invalid_contract_type"

    with pytest.raises(ValueError, match="max_uses=1"):
        engine.grant_once(
            "no-drop-staging-cache",
            r"drop_staging_cache\s+--force",
            user_chunk.chunk_id,
            session_id=sid,
            provider="codex",
            max_uses=5,
        )

    for bad_regex in (
        r".*",
        r"drop_staging_cache.*",
        r"drop_staging_cache|other_cmd",
        r"drop_[abc]_cache",
        r"drop_staging_cache\\.+--force",
        r"\S\S\S\S\S\S",
        r"drop_\w+_cache\s+--force",
    ):
        with pytest.raises(ValueError, match="overly broad"):
            engine.grant_once(
                "no-drop-staging-cache",
                bad_regex,
                user_chunk.chunk_id,
                session_id=sid,
                provider="codex",
            )

    with pytest.raises(ValueError, match="cannot exceed 3600s"):
        engine.grant_once(
            "no-drop-staging-cache",
            r"drop_staging_cache\s+--force",
            user_chunk.chunk_id,
            session_id=sid,
            provider="codex",
            expires_in="9999d",
        )

    # 2. Before grant_once, check_action blocks the command
    pre_check = engine.check_action(
        sid,
        {"tool_name": "Bash", "arguments": {"command": "drop_staging_cache --force"}},
        provider="codex",
        path=str(rollout),
    )
    finding_pre = next(f for f in pre_check["findings"] if f["contract_id"] == "no-drop-staging-cache")
    assert finding_pre["verdict"] == "violation"

    # 3. Create one-time grant anchored to user_chunk
    grant_res = engine.grant_once(
        "no-drop-staging-cache",
        r"drop_staging_cache\s+--force",
        user_chunk.chunk_id,
        session_id=sid,
        path=str(rollout),
        provider="codex",
        max_uses=1,
        expires_in="30m",
    )
    assert grant_res["ok"] is True
    grant_id = grant_res["grant"]["grant_id"]

    # 3b. Replaying the same source_event_id for a second grant on the same contract is rejected
    replay_res = engine.grant_once(
        "no-drop-staging-cache",
        r"drop_staging_cache\s+--force",
        user_chunk.chunk_id,
        session_id=sid,
        path=str(rollout),
        provider="codex",
    )
    assert replay_res["ok"] is False
    assert replay_res["error"] == "anchor_already_used"

    # 3c. Command chaining / piggybacking ("drop_staging_cache --force && echo pwned") is rejected and does NOT burn the grant
    chained_check = engine.check_action(
        sid,
        {"tool_name": "Bash", "arguments": {"command": "drop_staging_cache --force && echo pwned"}},
        provider="codex",
        path=str(rollout),
    )
    finding_chained = next(f for f in chained_check["findings"] if f["contract_id"] == "no-drop-staging-cache")
    assert finding_chained["verdict"] == "violation"

    # 4. First check_action succeeds with clean + allowed_by_grant receipt + matched_text_digest + preserved evidence
    post_check = engine.check_action(
        sid,
        {"tool_name": "Bash", "arguments": {"command": "drop_staging_cache --force"}},
        provider="codex",
        path=str(rollout),
    )
    finding_post = next(f for f in post_check["findings"] if f["contract_id"] == "no-drop-staging-cache")
    assert finding_post["verdict"] == "clean"
    assert "allowed_by_grant" in finding_post["reason"]
    assert len(finding_post["evidence"]) >= 1
    assert len(finding_post["grants_applied"]) == 1
    assert finding_post["grants_applied"][0]["grant_id"] == grant_id
    assert finding_post["grants_applied"][0]["source_event_id"] == user_chunk.chunk_id
    preview_digest = finding_post["grants_applied"][0]["matched_text_digest"]
    assert isinstance(preview_digest, str) and len(preview_digest) == 64
    assert finding_post["grants_applied"][0]["matched_text"] == "drop_staging_cache --force"

    # 4b. A second check_action with the exact same payload is DENIED (one-time grant already consumed on first check_action)
    second_check = engine.check_action(
        sid,
        {"tool_name": "Bash", "arguments": {"command": "drop_staging_cache --force"}},
        provider="codex",
        path=str(rollout),
    )
    finding_second_check = next(f for f in second_check["findings"] if f["contract_id"] == "no-drop-staging-cache")
    assert finding_second_check["verdict"] == "violation"

    # 5. Append the executed tool call to the rollout and audit_session -> linked to the same single-use receipt!
    _write_codex_rollout(
        rollout,
        sid,
        [
            ("message", "user", "Please fix the failing unit tests first."),
            ("message", "user", "You have my explicit permission to run drop_staging_cache --force once."),
            ("message", "assistant", "Understood, I will request a one-time grant anchored to your message."),
            ("tool_call", "assistant", "drop_staging_cache --force"),
        ],
    )
    audit_res1 = engine.audit_session(session_id=sid, path=str(rollout), provider="codex")
    finding_audit1 = next(f for f in audit_res1["findings"] if f["contract_id"] == "no-drop-staging-cache")
    assert finding_audit1["verdict"] == "clean"
    assert "allowed_by_grant" in finding_audit1["reason"]
    assert len(finding_audit1["evidence"]) == 1
    assert finding_audit1["grants_applied"][0]["grant_id"] == grant_id
    assert finding_audit1["grants_applied"][0]["matched_text_digest"] == preview_digest

    # 6. A second execution of the forbidden command is NOT covered (max_uses=1 exhausted)
    _write_codex_rollout(
        rollout,
        sid,
        [
            ("message", "user", "Please fix the failing unit tests first."),
            ("message", "user", "You have my explicit permission to run drop_staging_cache --force once."),
            ("message", "assistant", "Understood, I will request a one-time grant anchored to your message."),
            ("tool_call", "assistant", "drop_staging_cache --force"),
            ("tool_call", "assistant", "drop_staging_cache --force"),
        ],
    )
    audit_res2 = engine.audit_session(session_id=sid, path=str(rollout), provider="codex")
    finding_audit2 = next(f for f in audit_res2["findings"] if f["contract_id"] == "no-drop-staging-cache")
    assert finding_audit2["verdict"] == "violation"
    engine.close()


def test_issue57_test_contract_and_upsert_pre_save_verification(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    index_dir = tmp_path / "index"
    settings = Settings(codex_sessions_dir=sessions_dir, index_dir=index_dir, local_only=True, _env_file=None)
    engine = SelfDirectEngine(settings)

    candidate_rule = {
        "id": "no-force-push-main",
        "description": "Prevent git push --force to main",
        "type": "must_not",
        "severity": "critical",
        "regex": r"git\s+push\s+.*(?:--force|-f)\b.*\bmain\b",
        "scope": "tool_call",
        "cases": [
            {"id": "block-force-main", "command": "git push --force origin main", "expect": "match"},
            {"id": "allow-normal-main", "command": "git push origin main", "expect": "no_match"},
        ],
    }

    # 1. Standalone test_contract synthesizes canonical [tool_call:Bash] text and passes both cases
    report = engine.test_contract(candidate_rule, baseline="none")
    assert report["ok"] is True
    assert report["passed"] == 2
    assert report["failed"] == 0
    assert report["results"][0]["rendered_text"].startswith("[tool_call:Bash]")
    assert report["results"][0]["max_ms"] == 250.0

    # 2. upsert_contracts rejects a rule whose embedded test cases fail
    bad_rule = dict(candidate_rule)
    bad_rule["cases"] = [
        {"id": "unmatched-positive", "command": "git push origin main --force", "expect": "match"},
    ]
    with pytest.raises(ValueError, match="failed pre-save test cases"):
        engine.upsert_contracts([bad_rule])

    # 3. upsert_contracts accepts candidate_rule when all cases pass
    saved = engine.upsert_contracts([candidate_rule])
    assert len(saved["contracts"]) == 1
    assert "no-force-push-main" in saved.get("case_verification", {})

    # 3b. Updating without resending cases preserves the stored regression cases on revision 2!
    rev2_rule = dict(candidate_rule)
    rev2_rule["description"] = "Prevent git push --force/-f to main branch"
    rev2_rule["cases"] = []
    saved_rev2 = engine.upsert_contracts([rev2_rule])
    assert saved_rev2["contracts"][0]["revision"] == 2
    assert len(saved_rev2["contracts"][0].get("cases", [])) == 2

    # 4. Updating a critical contract with a regex that regresses on stored positive cases is rejected even on rev 3
    regressed_rule = dict(candidate_rule)
    regressed_rule["regex"] = r"git\s+push\s+--force-with-lease\s+origin\s+main"
    regressed_rule["cases"] = []

    cmp_report = engine.test_contract(regressed_rule, baseline="current")
    assert cmp_report["ok"] is False
    assert len(cmp_report["regressions"]) == 1
    assert cmp_report["regressions"][0]["id"] == "block-force-main"

    with pytest.raises(ValueError, match="regressed against stored revision"):
        engine.upsert_contracts([regressed_rule])

    # 5. Tightening an overly broad baseline regex to fix a false positive (expected="no_match") is an improvement, NOT a regression
    broad_baseline = dict(candidate_rule)
    broad_baseline["regex"] = r"git\s+push\s+.*\bmain\b"
    false_pos_fix_report = engine.test_contract(
        candidate_rule,
        cases=[
            {"id": "block-force-main", "command": "git push --force origin main", "expect": "match"},
            {"id": "allow-normal-main", "command": "git push origin main", "expect": "no_match"},
        ],
        baseline=broad_baseline,
    )
    assert false_pos_fix_report["ok"] is True
    assert len(false_pos_fix_report["regressions"]) == 0
    assert len(false_pos_fix_report["differences"]) == 1
    assert false_pos_fix_report["differences"][0]["id"] == "allow-normal-main"
    assert false_pos_fix_report["differences"][0]["baseline_passed"] is False
    assert false_pos_fix_report["differences"][0]["candidate_passed"] is True
    engine.close()


def test_batch4_server_mcp_tools_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    from self_directing_mcp import server

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    index_dir = tmp_path / "index"
    sid = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
    rollout = sessions_dir / f"rollout-2026-10-24T10-00-00-{sid}.jsonl"
    _write_codex_rollout(
        rollout,
        sid,
        [
            ("message", "user", "You may execute purge_temp_artifacts --confirm once in this session."),
        ],
    )

    settings = Settings(codex_sessions_dir=sessions_dir, index_dir=index_dir, local_only=True, _env_file=None)
    test_engine = SelfDirectEngine(settings)
    monkeypatch.setattr(server, "_engine", test_engine)

    try:
        sync_res = asyncio.run(
            server.sync_session(session_id=sid, path=str(rollout), provider="codex", embed=False)
        )
        assert sync_res["ok"] is True

        status_res = asyncio.run(server.job_status(session_id=sid, provider="codex"))
        assert status_res["ok"] is True
        assert status_res["status"] == "idle"

        with test_engine._components():
            user_chunk = next(
                c for c in test_engine.store.list_chunks(sid, "codex")
                if c.kind in ("turn", "message") and c.meta.get("role") == "user"
            )

        rule_payload = {
            "id": "no-purge-temp-artifacts",
            "description": "Block purge_temp_artifacts --confirm",
            "type": "must_not",
            "severity": "high",
            "regex": r"purge_temp_artifacts\s+--confirm",
            "scope": "tool_call",
            "session_id": sid,
            "source_event_id": user_chunk.chunk_id,
            "source_quote": user_chunk.text,
            "cases": [
                {"id": "pos-1", "command": "purge_temp_artifacts --confirm", "expect": "match"},
                {"id": "neg-1", "command": "purge_temp_artifacts --dry-run", "expect": "no_match"},
            ],
        }

        test_res = asyncio.run(server.test_contract(contract=rule_payload, baseline="none"))
        assert test_res["ok"] is True
        assert test_res["passed"] == 2

        upsert_res = asyncio.run(server.upsert_contracts(contracts=[rule_payload]))
        assert len(upsert_res["contracts"]) == 1

        grant_res = asyncio.run(
            server.grant_once(
                contract_id="no-purge-temp-artifacts",
                action_regex=r"purge_temp_artifacts\s+--confirm",
                source_event_id=user_chunk.chunk_id,
                session_id=sid,
                path=str(rollout),
                provider="codex",
                max_uses=1,
                expires_in="15m",
            )
        )
        assert grant_res["ok"] is True
        assert grant_res["grant"]["max_uses"] == 1
    finally:
        test_engine.close()


def test_canonical_english_normalization_for_multilingual_turns(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    index_dir = tmp_path / "index"
    sid = "cccccccc-dddd-eeee-ffff-000000000000"
    rollout = sessions_dir / f"rollout-2026-10-24T10-00-00-{sid}.jsonl"
    _write_codex_rollout(
        rollout,
        sid,
        [
            ("message", "user", "절대로 purge_temp_artifacts --confirm 을 임의로 실행하지 마. 항상 먼저 물어봐."),
            ("message", "user", "이번 한 번만 purge_temp_artifacts --confirm 실행해도 좋아."),
            ("message", "assistant", "알겠습니다. 일회성 승인을 등록하겠습니다."),
        ],
    )

    settings = Settings(codex_sessions_dir=sessions_dir, index_dir=index_dir, local_only=True, _env_file=None)
    engine = SelfDirectEngine(settings)
    engine.sync_session(session_id=sid, path=str(rollout), provider="codex", embed=False)

    with engine._components():
        chunks = engine.store.list_chunks(sid, "codex")
    user_chunks = [c for c in chunks if c.kind in ("turn", "message") and c.meta.get("role") == "user"]
    norm_turn = user_chunks[0]
    kr_approval_turn = user_chunks[1]
    asst_turn = next(c for c in chunks if c.kind in ("turn", "message") and c.meta.get("role") == "assistant")

    # 1. list_contract_candidates with Orchestration-LLM translated_turns extracts Canonical English candidates
    cand_res = engine.list_contract_candidates(
        session_id=sid,
        provider="codex",
        translated_turns={
            norm_turn.chunk_id: "Never run purge_temp_artifacts --confirm arbitrarily. Always ask the user first.",
            asst_turn.chunk_id: "Never do anything bad.",
            "nonexistent_chunk_999": "Always run tests.",
        },
    )
    assert cand_res["ok"] is True
    turn_cands = cand_res.get("turn_candidates", [])
    assert len(turn_cands) == 2
    assert {c["suggested_type"] for c in turn_cands} == {"must_not", "must"}
    assert all(c["source_event_id"] == norm_turn.chunk_id for c in turn_cands)
    assert "절대로" in turn_cands[0]["source_quote"]
    refused_turns = cand_res.get("refused_turns", [])
    assert {r["error"] for r in refused_turns} == {"anchor_not_user", "anchor_unresolved"}

    engine.upsert_contracts(
        [
            {
                "id": "no-purge-temp-kr",
                "description": "Block purge_temp_artifacts --confirm",
                "type": "must_not",
                "severity": "critical",
                "regex": r"purge_temp_artifacts\s+--confirm",
                "scope": "tool_call",
                "session_id": sid,
                "source_event_id": norm_turn.chunk_id,
                "source_quote": norm_turn.text,
            }
        ]
    )

    # 2. Calling grant_once on a Korean turn WITHOUT canonical_english_quote is rejected (no English approval cue)
    no_eng_res = engine.grant_once(
        "no-purge-temp-kr",
        r"purge_temp_artifacts\s+--confirm",
        kr_approval_turn.chunk_id,
        session_id=sid,
        path=str(rollout),
        provider="codex",
    )
    assert no_eng_res["ok"] is False
    assert no_eng_res["error"] == "anchor_does_not_approve_action"

    # 3. Calling grant_once WITH a negated canonical_english_quote is rejected
    neg_eng_res = engine.grant_once(
        "no-purge-temp-kr",
        r"purge_temp_artifacts\s+--confirm",
        kr_approval_turn.chunk_id,
        session_id=sid,
        path=str(rollout),
        provider="codex",
        canonical_english_quote="Do not execute purge_temp_artifacts --confirm under any circumstances.",
    )
    assert neg_eng_res["ok"] is False
    assert neg_eng_res["error"] == "anchor_does_not_approve_action"

    # 3b. Calling grant_once on a Korean prohibition turn (norm_turn) WITH a forged approving canonical_english_quote is rejected!
    forged_eng_res = engine.grant_once(
        "no-purge-temp-kr",
        r"purge_temp_artifacts\s+--confirm",
        norm_turn.chunk_id,
        session_id=sid,
        path=str(rollout),
        provider="codex",
        canonical_english_quote="You may execute purge_temp_artifacts --confirm once.",
    )
    assert forged_eng_res["ok"] is False
    assert forged_eng_res["error"] == "anchor_does_not_approve_action"

    # 4. Calling grant_once WITH Orchestration-LLM canonical_english_quote succeeds and records both quotes
    ok_eng_res = engine.grant_once(
        "no-purge-temp-kr",
        r"purge_temp_artifacts\s+--confirm",
        kr_approval_turn.chunk_id,
        session_id=sid,
        path=str(rollout),
        provider="codex",
        canonical_english_quote="You may execute purge_temp_artifacts --confirm once.",
    )
    assert ok_eng_res["ok"] is True
    assert ok_eng_res["grant"]["canonical_english_quote"] == "You may execute purge_temp_artifacts --confirm once."
    assert "이번 한 번만" in ok_eng_res["grant"]["source_quote"]
    engine.close()


def test_round3_security_invariants_for_grants_and_testing() -> None:
    from self_directing_mcp.audit.contract_testing import evaluate_rule_on_event, merge_contract_cases
    from self_directing_mcp.audit.grants import (
        _grant_match_digest,
        create_grant_record,
        verify_user_anchor_approves_grant,
    )
    from self_directing_mcp.schemas import AuditEvent, ContractCase, ContractRule

    rule_gc = ContractRule(
        id="no-gc-prune-tmp",
        description="Block gc --prune /tmp/foo",
        type="must_not",
        severity="critical",
        regex=r"gc\s+--prune\s+/tmp/foo",
        scope="tool_call",
    )

    # 1. Substring inside word ('gc' in 'subgc') or out-of-order tokens must NOT approve a grant
    ok_sub, _ = verify_user_anchor_approves_grant(
        anchor_text="You may run subgc cleanup of /tmp/foo with --prune.",
        action_regex=r"gc\s+--prune\s+/tmp/foo",
        rule=rule_gc,
    )
    assert ok_sub is False

    # 2. Conditional ('if'), epistemic observation, negated CJK, trailing retraction, or Unicode path suffix ('/tmp/foo백업') must NOT approve a grant
    for tricky_text in (
        "You may run gc --prune /tmp/foo if the build fails.",
        "You may run gc --prune /tmp/foo. Disregard that.",
        "You may run gc --prune /tmp/foo, but hold off for now.",
        "You may run gc --prune /tmp/foo.txt once.",
        "You may run gc --prune /tmp/foo백업 once.",
        "You may have seen gc --prune /tmp/foo in the logs.",
    ):
        ok_cond, _ = verify_user_anchor_approves_grant(
            anchor_text=tricky_text,
            action_regex=r"gc\s+--prune\s+/tmp/foo",
            rule=rule_gc,
        )
        assert ok_cond is False

    for tricky_kr in (
        "gc --prune /tmp/foo 는 허용하지 않아.",
        "gc --prune /tmp/foo 진행해도 되는지 물어보지 마.",
        "gc --prune /tmp/foo 不許可",
    ):
        ok_kr_neg, _ = verify_user_anchor_approves_grant(
            anchor_text=tricky_kr,
            action_regex=r"gc\s+--prune\s+/tmp/foo",
            rule=rule_gc,
            canonical_english_quote="You may run gc --prune /tmp/foo once.",
        )
        assert ok_kr_neg is False

    # 3. _grant_match_digest requires full-string single-line match and rejects embedded newlines/\r, and create_grant_record enforces session_id & max_anchor_events
    with pytest.raises(ValueError, match="chunks is required"):
        create_grant_record(
            session_id="s1",
            rule=rule_gc,
            action_regex=r"gc\s+--prune\s+/tmp/foo",
            source_event_id="ev1",
            source_quote="You may run gc --prune /tmp/foo once.",
        )
    grant_rec = create_grant_record(
        session_id="s1",
        rule=rule_gc,
        action_regex=r"gc\s+--prune\s+/tmp/foo",
        source_event_id="ev1",
        source_quote="You may run gc --prune /tmp/foo once.",
        chunks=[AuditEvent("ev1", "turn", "You may run gc --prune /tmp/foo once.", {"role": "user"}, session_id="s1")],
    )
    assert _grant_match_digest(grant_rec, rule_gc, "[tool_call:Bash]\ncommand: gc --prune /tmp/foo")[0] is True
    assert _grant_match_digest(grant_rec, rule_gc, "[tool_call:Bash]\ncommand: sudo gc --prune /tmp/foo /etc")[0] is False
    assert _grant_match_digest(grant_rec, rule_gc, "[tool_call:Bash]\ncommand: gc --prune /tmp/foo\necho pwned")[0] is False
    assert _grant_match_digest(grant_rec, rule_gc, "[tool_call:Bash]\ncommand: gc --prune /tmp/foo\recho pwned")[0] is False
    assert _grant_match_digest(grant_rec, rule_gc, "[tool_call:Write]\nsudo gc --prune /tmp/foo /etc")[0] is False

    # 4. evaluate_rule_on_event on a 'must' rule with both regex and before_regex requires prerequisite (regex) and NOT before_regex alone
    must_rule = ContractRule(
        id="must-test-before-push",
        description="Run pytest before git push",
        type="must",
        severity="high",
        regex=r"pytest",
        before_regex=r"git\s+push",
        scope="tool_call",
    )
    ev_push_only = AuditEvent("c1", "tool_call", "[tool_call:Bash]\ncommand: git push origin main", {"role": "assistant", "tool_name": "Bash"})
    ev_pytest_only = AuditEvent("c2", "tool_call", "[tool_call:Bash]\ncommand: pytest -q", {"role": "assistant", "tool_name": "Bash"})
    assert evaluate_rule_on_event(must_rule, ev_push_only)[0] is False
    assert evaluate_rule_on_event(must_rule, ev_pytest_only)[0] is True

    # 5. merge_contract_cases preserves distinct baseline case even when candidate reuses the same case id
    merged = merge_contract_cases(
        [
            ContractCase(id="c1", command="git push --force origin dev", expect="no_match"),
            ContractCase(id="c1__candidate", command="git push --force origin staging", expect="match"),
        ],
        [ContractCase(id="c1", command="git push --force origin main", expect="match")],
    )
    assert len(merged) == 3

    # 6. Bracketed negation prefix [never] is NOT stripped as a chunk header, and citation/extra-token non-ASCII clauses fail closed
    ok_bracket_neg, _ = verify_user_anchor_approves_grant(
        anchor_text="[never]\nYou may run gc --prune /tmp/foo once.",
        action_regex=r"gc\s+--prune\s+/tmp/foo",
        rule=rule_gc,
    )
    assert ok_bracket_neg is False

    for tricky_cite_or_extra in (
        "gc --prune /tmp/foo 라고 적혀 있어.",
        "gc --prune /tmp/foo /etc 해도 돼.",
    ):
        ok_extra, _ = verify_user_anchor_approves_grant(
            anchor_text=tricky_cite_or_extra,
            action_regex=r"gc\s+--prune\s+/tmp/foo",
            rule=rule_gc,
            canonical_english_quote="You may run gc --prune /tmp/foo once.",
        )
        assert ok_extra is False

    # 7. render_case_event merges case.command and case.arguments and respects explicit case.tool_name
    from self_directing_mcp.audit.contract_testing import render_case_event
    ev_merged = render_case_event(
        ContractCase(
            id="c_merge",
            command="git push",
            arguments={"cwd": "/tmp"},
            tool_name="CustomRunner",
            expect="match",
        )
    )
    assert ev_merged.meta["tool_name"] == "CustomRunner"
    assert "/tmp" in ev_merged.text and "git push" in ev_merged.text
    ev_explicit_tool = render_case_event(
        ContractCase(
            id="c_explicit_tool",
            tool_name="CustomRunner",
            text="command: git push",
            expect="match",
        )
    )
    assert ev_explicit_tool.meta["tool_name"] == "CustomRunner"
    with pytest.raises(ValueError):
        render_case_event(
            ContractCase(
                id="c_conflict_tool",
                tool_name="CustomRunner",
                text="[tool_call:Bash]\ncommand: git push",
                expect="match",
            )
        )

    # 8. evaluate_rule_on_event enforces m_req.start() < m_before.start() when both match inside a single event
    ev_wrong_order = AuditEvent("c_order", "tool_call", "[tool_call:Bash]\ncommand: git push && pytest", {"role": "assistant", "tool_name": "Bash"})
    ev_right_order = AuditEvent("c_order_ok", "tool_call", "[tool_call:Bash]\ncommand: pytest && git push", {"role": "assistant", "tool_name": "Bash"})
    assert evaluate_rule_on_event(must_rule, ev_wrong_order)[0] is False
    assert evaluate_rule_on_event(must_rule, ev_right_order)[0] is True


