"""Main-owned incremental GraphRAG update with NeoGraph execution and recovery.

Responsibilities (issue #5):
- MCP: durable unprocessed-range tracking, deterministic fact extraction,
  proposal validation (schema, evidence, version), idempotent apply + cursor.
- NeoGraph (required backend): runs audit and graph-update orchestration;
  a missing native dependency is an explicit startup error.

Relationship meaning comes from the main/extractor agent; this layer only
verifies evidence and stores. Relations are marked observed vs inferred and
never grant requirement satisfaction or execution authority.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Union

from self_directing_mcp.request_control import check_request, publication, request_operation


GRAPH_SCHEMA_VERSION = 1
ALLOWED_RELATIONS = {
    "IMPLEMENTS", "DEPENDS_ON", "MODIFIES", "PRODUCES",
    "CHECKS_VERSION", "EVIDENCED_BY", "SUPERSEDES",
}


class GraphStore:
    """SQLite graph with provenance and a processing cursor, separate file."""

    def __init__(self, path: Union[str, Path]) -> None:
        check_request()
        self.db_path = Path(path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            self._init_schema()
        except Exception:
            self._conn.close()
            raise

    def _init_schema(self):
        check_request()
        self._conn.executescript("""
            BEGIN;
            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id TEXT NOT NULL, provider TEXT NOT NULL, session_id TEXT NOT NULL,
                kind TEXT NOT NULL, label TEXT NOT NULL, version_added INTEGER NOT NULL,
                PRIMARY KEY(node_id, session_id, provider)
            );
            CREATE TABLE IF NOT EXISTS graph_edges (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                src TEXT NOT NULL, dst TEXT NOT NULL, relation TEXT NOT NULL,
                origin TEXT NOT NULL DEFAULT 'inferred',
                evidence_chunk_ids TEXT NOT NULL,
                session_id TEXT NOT NULL, provider TEXT NOT NULL,
                version_added INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS edges_session ON graph_edges(provider, session_id, version_added);
            CREATE TABLE IF NOT EXISTS graph_cursor (
                provider TEXT NOT NULL, session_id TEXT NOT NULL,
                processed_through_order INTEGER NOT NULL DEFAULT 0,
                graph_version INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(provider, session_id)
            );
            CREATE TABLE IF NOT EXISTS graph_jobs (
                job_id TEXT PRIMARY KEY,
                provider TEXT NOT NULL, session_id TEXT NOT NULL,
                range_start INTEGER NOT NULL, range_end INTEGER NOT NULL,
                input_digest TEXT NOT NULL, base_graph_version INTEGER NOT NULL,
                status TEXT NOT NULL, detail TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
        """)
        with publication():
            self._conn.commit()

    @request_operation
    def cursor(self, session_id: str, provider: str) -> dict[str, int]:
        return self.observed_cursor(session_id, provider)

    def observed_cursor(self, session_id: str, provider: str) -> dict[str, int]:
        """Read actual stored outcome during owned diagnostic/cleanup settling.

        The caller still owns the connection guard. Do not use this exemption
        for ordinary request work or to admit another graph publication.
        """
        row = self._conn.execute(
            "SELECT processed_through_order, graph_version FROM graph_cursor WHERE provider=? AND session_id=?",
            (provider, session_id)).fetchone()
        return {"processed_through_order": row[0], "graph_version": row[1]} if row else {"processed_through_order": 0, "graph_version": 0}

    @request_operation
    def find_job(self, job_id: str):
        row = self._conn.execute("SELECT * FROM graph_jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def record_job(self, job: dict[str, Any]) -> None:
        check_request()
        values = (job["job_id"], job["provider"], job["session_id"], job["range_start"],
                  job["range_end"], job["input_digest"], job["base_graph_version"],
                  job["status"], job.get("detail"), job["created_at"], job["updated_at"])
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO graph_jobs (job_id,provider,session_id,range_start,range_end,"
                "input_digest,base_graph_version,status,detail,created_at,updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)", values)
            with publication():
                self._conn.commit()

    def apply_proposal(self, proposal: dict[str, Any], *, session_id: str, provider: str,
                       base_version: int, through_order: int) -> dict[str, Any]:
        """Atomic apply: nodes/edges + cursor advance. Unknown outcome stays unclear."""
        check_request()
        edge_rows = []
        for edge in proposal.get("edges", []):
            check_request()
            edge_rows.append((edge["src"], edge["dst"], edge["relation"],
                              edge.get("origin", "inferred"),
                              json.dumps(edge.get("evidence_chunk_ids", []))))
        version = base_version + 1
        job_id = proposal.get("job_id")
        try:
            with self._conn:
                check_request()
                if job_id:
                    # Post-crash replay guard: an already-recorded successful
                    # job is never applied twice at the storage layer either.
                    prior = self._conn.execute(
                        "SELECT status FROM graph_jobs WHERE job_id=?", (job_id,)).fetchone()
                    if prior and prior[0] == "applied":
                        return {"ok": True, "status": "already_applied",
                                "graph_version": version - 1}
                for node in proposal.get("nodes", []):
                    check_request()
                    self._conn.execute(
                        "INSERT OR IGNORE INTO graph_nodes VALUES (?,?,?,?,?,?)",
                        (node["node_id"], provider, session_id, node.get("kind", "task"),
                         node.get("label", node["node_id"]), version))
                for edge in edge_rows:
                    check_request()
                    self._conn.execute(
                        "INSERT INTO graph_edges (src,dst,relation,origin,evidence_chunk_ids,session_id,provider,version_added)"
                        " VALUES (?,?,?,?,?,?,?,?)", (*edge, session_id, provider, version))
                self._conn.execute(
                    "INSERT OR REPLACE INTO graph_cursor VALUES (?,?,?,?)",
                    (provider, session_id, through_order, version))
                if job_id:
                    from datetime import datetime, timezone
                    now = datetime.now(timezone.utc).isoformat()
                    self._conn.execute(
                        "INSERT OR REPLACE INTO graph_jobs (job_id,provider,session_id,"
                        "range_start,range_end,input_digest,base_graph_version,status,"
                        "detail,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (job_id, provider, session_id, 0, through_order, "", base_version,
                         "applied", None, now, now))
                with publication():
                    self._conn.commit()
        except sqlite3.Error as exc:
            return {"ok": False, "status": "unclear",
                    "detail": f"graph write failed: {type(exc).__name__}; range not advanced"}
        return {"ok": True, "status": "applied", "graph_version": version,
                "processed_through_order": through_order}

    @request_operation
    def neighbors(self, session_id: str | None, provider: str, node_id: str, depth: int = 1):
        seen, frontier, out = {node_id}, [node_id], []
        for _ in range(max(1, min(depth, 3))):
            check_request()
            placeholders = ",".join("?" for _ in frontier)
            if session_id and session_id != "*":
                rows = self._conn.execute(
                    "SELECT src,dst,relation,origin,evidence_chunk_ids,version_added FROM graph_edges"
                    " WHERE session_id=? AND provider=? AND src IN (" + placeholders + ")"
                    " UNION ALL SELECT src,dst,relation,origin,evidence_chunk_ids,version_added"
                    " FROM graph_edges WHERE session_id=? AND provider=? AND dst IN (" + placeholders + ")",
                    (session_id, provider, *frontier, session_id, provider, *frontier)).fetchall()
                if not rows:
                    rows = self._conn.execute(
                        "SELECT src,dst,relation,origin,evidence_chunk_ids,version_added FROM graph_edges"
                        " WHERE provider=? AND src IN (" + placeholders + ")"
                        " UNION ALL SELECT src,dst,relation,origin,evidence_chunk_ids,version_added"
                        " FROM graph_edges WHERE provider=? AND dst IN (" + placeholders + ")",
                        (provider, *frontier, provider, *frontier)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT src,dst,relation,origin,evidence_chunk_ids,version_added FROM graph_edges"
                    " WHERE provider=? AND src IN (" + placeholders + ")"
                    " UNION ALL SELECT src,dst,relation,origin,evidence_chunk_ids,version_added"
                    " FROM graph_edges WHERE provider=? AND dst IN (" + placeholders + ")",
                    (provider, *frontier, provider, *frontier)).fetchall()
            frontier = []
            for src, dst, rel, origin, evidence, ver in rows:
                check_request()
                try:
                    ev_list = json.loads(evidence)
                except Exception:
                    ev_list = []
                out.append({"src": src, "dst": dst, "relation": rel, "origin": origin,
                            "evidence_chunk_ids": ev_list, "version": ver})
                for n in (src, dst):
                    if n not in seen:
                        seen.add(n)
                        frontier.append(n)
            if not frontier:
                break
        return {"nodes": sorted(seen), "edges": out}

    @request_operation
    def get_session_graph(self, session_id: str | None = None, provider: str = "codex", limit: int = 50) -> dict[str, Any]:
        """Fetch nodes and edges for a session, or across sessions if session has no entries."""
        where_nodes = "WHERE provider=?"
        where_edges = "WHERE provider=?"
        params_nodes: list[Any] = [provider]
        params_edges: list[Any] = [provider]

        if session_id and session_id != "*":
            count = self._conn.execute(
                "SELECT COUNT(*) FROM graph_nodes WHERE session_id=? AND provider=?",
                (session_id, provider)
            ).fetchone()[0]
            if count > 0:
                where_nodes += " AND session_id=?"
                where_edges += " AND session_id=?"
                params_nodes.append(session_id)
                params_edges.append(session_id)

        params_nodes.append(limit)
        params_edges.append(limit)

        nodes_rows = self._conn.execute(
            f"SELECT node_id, provider, session_id, kind, label, version_added FROM graph_nodes {where_nodes} ORDER BY version_added DESC LIMIT ?",
            params_nodes
        ).fetchall()

        edges_rows = self._conn.execute(
            f"SELECT edge_id, src, dst, relation, origin, evidence_chunk_ids, session_id, provider, version_added FROM graph_edges {where_edges} ORDER BY edge_id DESC LIMIT ?",
            params_edges
        ).fetchall()

        nodes = [dict(r) for r in nodes_rows]
        edges = []
        for r in edges_rows:
            check_request()
            d = dict(r)
            try:
                d["evidence_chunk_ids"] = json.loads(d["evidence_chunk_ids"])
            except Exception:
                pass
            edges.append(d)

        return {"nodes": nodes, "edges": edges}

    @request_operation
    def find_nodes_like(self, query: str, session_id: str | None = None, provider: str = "codex") -> list[dict[str, Any]]:
        """Find nodes matching a filename or query."""
        clean_q = f"%{query}%"
        if session_id and session_id != "*":
            rows = self._conn.execute(
                "SELECT node_id, provider, session_id, kind, label, version_added FROM graph_nodes WHERE session_id=? AND provider=? AND (node_id LIKE ? OR label LIKE ?)",
                (session_id, provider, clean_q, clean_q)
            ).fetchall()
            if rows:
                return [dict(r) for r in rows]
        rows = self._conn.execute(
            "SELECT node_id, provider, session_id, kind, label, version_added FROM graph_nodes WHERE provider=? AND (node_id LIKE ? OR label LIKE ?)",
            (provider, clean_q, clean_q)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self._conn.close()


@request_operation
def validate_proposal(proposal: dict[str, Any], *, store, session_id: str, provider: str) -> dict[str, Any]:
    """Schema, evidence and provenance checks; rejects without storing.

    Edges need evidence chunk ids that exist in the indexed session and stay
    inside it. origin distinguishes observed facts from agent-inferred
    relations. Rejections never partially update the graph.
    """
    problems = []
    known_chunks = set(store.chunk_ids(session_id, provider))
    node_ids = {n.get("node_id") for n in proposal.get("nodes", [])}
    for edge in proposal.get("edges", []):
        check_request()
        relation = edge.get("relation", "")
        if relation not in ALLOWED_RELATIONS:
            problems.append(f"edge {edge.get('src')}->{edge.get('dst')}: unknown relation {relation!r}")
        evidence = edge.get("evidence_chunk_ids") or []
        if not evidence:
            problems.append(f"edge {edge.get('src')}->{edge.get('dst')} ({relation}): no evidence_chunk_ids")
        else:
            missing = [c for c in evidence if c not in known_chunks]
            if missing:
                problems.append(f"edge ({relation}): evidence not in this session: {missing[:3]}")
        if edge.get("origin") not in (None, "observed", "inferred"):
            problems.append(f"edge ({relation}): origin must be observed|inferred")
    if len(node_ids := {n.get("node_id") for n in proposal.get("nodes", [])}) != len(proposal.get("nodes", [])):
        problems.append("duplicate node_id in proposal")
    for edge in proposal.get("edges", []):
        check_request()
        if edge.get("src") not in node_ids or edge.get("dst") not in node_ids:
            problems.append(f"edge {edge.get('src')}->{edge.get('dst')}: references node outside this proposal")
    return {"ok": not problems, "problems": problems}

