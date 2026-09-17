"""Live end-to-end verification of the GraphRAG update flow with a real LLM.

Uses NeoGraph's OpenAI-compatible provider routed to OpenRouter
(deepseek/deepseek-v4-flash-0731) so the extractor agent is a real LLM, not a
mock. The MCP layer still validates schema/evidence; the LLM only proposes.

Run:  python scripts/graphrag_llm_e2e.py   (requires OPENROUTER_API_KEY)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from self_directing_mcp.config import Settings
from self_directing_mcp.engine import SelfDirectEngine
from self_directing_mcp.schemas import Chunk


MODEL = "deepseek/deepseek-v4-flash-0731"
SID = "llm00000-1111-2222-3333-444444444444"

EXTRACTOR_PROMPT = """You are a graph extractor. Read the session events and return
nodes/edges with evidence. Allowed relations: IMPLEMENTS, DEPENDS_ON, MODIFIES,
PRODUCES, CHECKS_VERSION, EVIDENCED_BY, SUPERSEDES.
Rules:
- Every edge MUST list evidence_chunk_ids copied from the provided event ids.
- origin: "observed" only when the event text directly states the relation;
  otherwise "inferred".
- Reply with ONLY a JSON object:
  {"nodes":[{"node_id","kind","label"}],
   "edges":[{"src","dst","relation","origin","evidence_chunk_ids"}]}

Session events:
{events}
"""


def main() -> int:
    from dotenv import load_dotenv
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("SKIP: OPENROUTER_API_KEY not set")
        return 2
    import neograph_engine as ng
    import neograph_engine.llm  # registers OpenAIProvider on the engine module

    import urllib.request

    def llm_complete(prompt_text):
        schema = {
            "type": "object",
            "properties": {
                "nodes": {"type": "array", "items": {"type": "object", "properties": {
                    "node_id": {"type": "string"}, "kind": {"type": "string"}, "label": {"type": "string"}
                }, "required": ["node_id", "kind", "label"], "additionalProperties": False}},
                "edges": {"type": "array", "items": {"type": "object", "properties": {
                    "src": {"type": "string"}, "dst": {"type": "string"}, "relation": {"type": "string"},
                    "origin": {"type": "string"}, "evidence_chunk_ids": {"type": "array", "items": {"type": "string"}}
                }, "required": ["src", "dst", "relation", "origin", "evidence_chunk_ids"], "additionalProperties": False}}
            },
            "required": ["nodes", "edges"],
            "additionalProperties": False,
        }
        body = json.dumps({"model": MODEL,
                           "messages": [{"role": "user", "content": prompt_text}],
                           "response_format": {"type": "json_schema",
                               "json_schema": {"name": "graph_proposal", "strict": True, "schema": schema}},
                           "reasoning": {"exclude": True, "effort": "low"},
                           "max_tokens": 8000}).encode()
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
            data=body, headers={"Authorization": "Bearer " + api_key,
                                "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.load(resp)
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        return content, data.get("usage", {}).get("total_tokens", 0)
    print("transport: urllib-direct")

    tmp = Path(tempfile.mkdtemp())
    settings = Settings(use_fake_embedder=True, dense_backend="numpy", index_dir=tmp / "index",
                        contracts_path=tmp / "index" / "contracts.json")
    engine = SelfDirectEngine(settings=settings)
    try:
        engine.ensure_ready()
        executor = "neograph" if engine.neograph else "direct"
        print("executor:", executor)

        events = [
            ("c000", "turn", "user: 한국어 보고서를 만들고 출처를 달아 PDF로 저장한 뒤 열리는지 확인해줘"),
            ("c001", "tool_call", "[tool_call:write] report_ko.md 작성 (한국어, 출처 3개 포함)"),
            ("c002", "tool_call", "[tool_call:write] references.md 작성 (arxiv 3편)"),
            ("c003", "tool_result", "pandoc report_ko.md -o report_ko.pdf -> success 12 pages"),
            ("c004", "tool_result", "open report_ko.pdf -> success, rendered 12 pages"),
        ]
        rows = [Chunk(chunk_id=cid, session_id=SID, provider="codex", kind=kind,
                      text=text, content_hash="h" + cid[1:],
                      meta={"role": "assistant" if kind != "turn" else "user"})
                for cid, kind, text in events]
        engine.store.commit_events(rows, session_id=SID, provider="codex",
                                   path=tmp / "s.jsonl", offset=500, digest="e2e")

        event_block = chr(10).join(f"- {cid} [{kind}] {text}" for cid, kind, text in events)
        prompt = EXTRACTOR_PROMPT.replace("{events}", event_block)
        raw, total = llm_complete(prompt)
        if raw is None:
            # Reasoning models may spend all tokens before emitting content.
            raise SystemExit("LLM returned no content; rerun or raise max_tokens")
        print("llm tokens used:", total)
        print("--- raw LLM output (first 500 chars) ---")
        print(raw[:500])

        start = raw.find("{")
        end = raw.rfind("}") + 1
        proposal = json.loads(raw[start:end])
        proposal["job_id"] = "job-llm-e2e-1"
        print("proposed nodes:", [n.get("node_id") for n in proposal.get("nodes", [])])
        print("proposed edges:", [(e.get("src"), e.get("relation"), e.get("dst")) for e in proposal.get("edges", [])])

        check = engine.propose_graph_update(SID, proposal)
        print("validation ok:", check["ok"], "| problems:", check.get("problems"))
        assert check["ok"], check

        result = engine.commit_graph_update(SID, proposal, base_graph_version=0)
        print("commit:", result["status"], "| graph_version:", result.get("graph_version"),
              "| through_order:", result.get("processed_through_order"))
        assert result["ok"] and result["status"] == "applied"

        replay = engine.commit_graph_update(SID, proposal, base_graph_version=0)
        print("replay:", replay["status"])
        assert replay["status"] == "already_applied"

        anchor = proposal.get("nodes", [])[0]["node_id"]
        nb = engine.graph.neighbors(SID, "codex", anchor, depth=2)
        print("traversal from", anchor, "->", sorted(nb["nodes"]), "| edges:", len(nb["edges"]))
        for edge in nb["edges"][:4]:
            print("   ", edge["src"], "-[", edge["relation"], "]->", edge["dst"],
                  "origin=" + edge["origin"], "evidence=" + str(edge["evidence_chunk_ids"]))
        assert nb["edges"], "graph traversal must return LLM-proposed edges"

        stale = engine.commit_graph_update(SID, {**proposal, "job_id": "job-llm-stale"}, base_graph_version=0)
        assert stale["ok"] is False and stale["status"] == "version_conflict"
        print("stale version rejected:", stale["status"])

        print("ALL LLM E2E CHECKS PASSED (real extractor:", MODEL + ")")
        return 0
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())






