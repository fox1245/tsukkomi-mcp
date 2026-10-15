#!/usr/bin/env python3
"""pgvector next to sqlite-vector and NumPy as the dense index, on synthetic vectors (nothing real is read).

For each size N of unit float32 vectors (dimension 1024 by default, generated in seeded batches and never kept whole,
except for NumPy which holds them) the script measures, on the same data and machine:

* bulk upsert, rows per second (batches of ``--batch`` rows, one call, so one transaction, per batch);
* ``present`` of 500 stored ids;
* ``search`` (top_k = 20, exact) with no filter, with a 500-id ``allowed_ids`` and with a 50 000-id one (ids that are
  not stored pad the set when N is below 50 000: what is measured is the filter, not the hits);
* pgvector only: table, primary-key index, total size; the HNSW index (build time, size, search latency, recall@20 against
  the exact scan at several ``hnsw.ef_search``).

A throwaway pgvector container is started (``tests/pgvector_support.py``: loopback, tmpfs data directory, a random
password) unless ``--dsn-file`` names a server. ``--lan-delay-ms`` runs pgvector through a loopback proxy that delays
every forwarded chunk, to show what a network round trip costs (latency only; bandwidth is not limited).

    python scripts/bench_pgvector.py --sizes 1000,10000,100000 --sqlite-vector /abs/vector.so --out results.json
    python scripts/bench_pgvector.py --sizes 300000 --tmpfs 8g --shm 6g --maintenance-work-mem 4GB --out big.json
    python scripts/bench_pgvector.py --sizes 10000 --backends pgvector --lan-delay-ms 0.5 --out lan.json

Results go to ``--out`` (keep them outside the repository). Latencies are client-side wall clock of the index call.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "tests"))

import numpy as np                                                  # noqa: E402

from dense_support import normalized_vectors                        # noqa: E402

CLUSTERS = 500


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summary(milliseconds):
    return {"p50_ms": round(percentile(milliseconds, 50), 3), "p95_ms": round(percentile(milliseconds, 95), 3),
            "n": len(milliseconds)}


class Data:
    """Seeded vectors: ``uniform`` (random unit vectors, the hard case for an approximate index) or ``clustered`` (500 centres
    plus noise, closer to what embeddings look like). Batch ``b`` is always the same rows, so a smaller N is a prefix."""

    def __init__(self, kind: str, dim: int) -> None:
        self.kind, self.dim = kind, dim
        self.centres = normalized_vectors(CLUSTERS, dim, seed=4242) if kind == "clustered" else None

    def batch(self, index: int, size: int, seed_base: int = 1000) -> np.ndarray:
        if self.kind == "uniform":
            return normalized_vectors(size, self.dim, seed=seed_base + index)
        rng = np.random.default_rng(seed_base + index)
        centre = self.centres[rng.integers(0, CLUSTERS, size=size)]
        noise = rng.standard_normal((size, self.dim)).astype(np.float32)
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        matrix = centre + np.float32(0.75) * noise                  # cosine to its centre is about 0.8
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix.astype(np.float32)

    def stream(self, count: int, batch: int):
        for number, start in enumerate(range(0, count, batch)):
            size = min(batch, count - start)
            yield [f"c{n:07d}" for n in range(start, start + size)], self.batch(number, size)

    def queries(self, count: int) -> np.ndarray:
        return self.batch(10 ** 6, count, seed_base=7)


def timed(call, repeats, warmup=2):
    for _ in range(warmup):
        call()
    out = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        out.append((time.perf_counter() - started) * 1000)
    return out


def load(index, data: Data, n: int, batch: int, single_call: bool = False) -> float:
    """Upsert ``n`` rows; returns seconds spent inside the index calls (generating the vectors is not counted)."""
    spent = 0.0
    if single_call:
        pairs = []
        for ids, matrix in data.stream(n, batch):
            pairs.extend(zip(ids, matrix))
        started = time.perf_counter()
        index.upsert_many(pairs)
        return time.perf_counter() - started
    for ids, matrix in data.stream(n, batch):
        pairs = list(zip(ids, matrix))
        started = time.perf_counter()
        index.upsert_many(pairs)
        spent += time.perf_counter() - started
    return spent


def measure_index(index, data: Data, n: int, queries: np.ndarray, rng, *, repeats: int) -> dict:
    ids = [f"c{k:07d}" for k in range(n)]
    sample = [ids[k] for k in rng.choice(n, size=min(500, n), replace=False)]
    present = timed(lambda: index.present(sample), repeats)
    allowed500 = {ids[k] for k in rng.choice(n, size=min(500, n), replace=False)}
    real = [ids[k] for k in rng.choice(n, size=min(50_000, n), replace=False)]
    allowed50k = set(real) | {f"ghost{k}" for k in range(50_000 - len(real))}
    out = {"present500": summary(present), "search": {}}
    for name, allowed in (("no_filter", None), ("ids_500", allowed500), ("ids_50k", allowed50k)):
        cycle = iter(range(10 ** 9))
        latencies = timed(lambda: index.search(queries[next(cycle) % len(queries)], 20, allowed), repeats)
        out["search"][name] = summary(latencies)
    out["search"]["ids_50k"]["real_ids"] = len(real)
    return out


def recall_at_20(exact: list[set], approximate: list[list]) -> float:
    return float(np.mean([len(want & {cid for cid, _ in got}) / 20 for want, got in zip(exact, approximate)]))


def run_pgvector(server, conninfo, data, n, args, queries, rng):
    from self_directing_mcp.index.pgvector import PgVectorIndex

    schema = f"bench_{n}"
    server.drop_schema(schema)
    index = PgVectorIndex(conninfo, dim=args.dim, schema=schema, policy="bench")
    result: dict = {}
    try:
        if "load" in args.phases:
            seconds = load(index, data, n, args.batch)
            result["upsert_rows_per_s"] = round(n / seconds)
            server.execute(f'ANALYZE "{schema}".chunk_vectors')
            size = server.execute("SELECT pg_total_relation_size(%s), pg_relation_size(%s), pg_indexes_size(%s), "
                                  "pg_total_relation_size(%s) - pg_relation_size(%s) - pg_indexes_size(%s)",
                                  (f'"{schema}".chunk_vectors',) * 6)[0]
            result["bytes"] = {"total": int(size[0]), "heap": int(size[1]), "indexes": int(size[2]), "toast": int(size[3])}
            result.update(measure_index(index, data, n, queries, rng, repeats=args.repeats))
            from self_directing_mcp.index.pgvector import PLAIN_STORAGE_MAX_DIM
            result["storage"] = "plain (inline)" if args.dim <= PLAIN_STORAGE_MAX_DIM else "pgvector default (external)"
        if "hnsw" in args.phases and not args.no_hnsw and n <= args.hnsw_max:
            exact = [{cid for cid, _ in index.search(q, 20)} for q in queries[:args.recall_queries]]
            started = time.perf_counter()
            try:
                index.build_hnsw_index(maintenance_work_mem=args.maintenance_work_mem, timeout_ms=int(args.build_timeout * 1000))
            except Exception as exc:
                result["hnsw"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            else:
                built = time.perf_counter() - started
                size = server.execute("SELECT pg_relation_size(%s::regclass)", (f'"{schema}".chunk_vectors_embedding_hnsw',))[0][0]
                hnsw = {"build_s": round(built, 1), "index_bytes": int(size), "recall_at_20": {}, "search_ms": {}}
                for ef in args.ef:
                    approximate = PgVectorIndex(conninfo, dim=args.dim, schema=schema, policy="bench", hnsw=True,
                                                hnsw_ef_search=ef)
                    try:
                        found = [approximate.search(q, 20) for q in queries[:args.recall_queries]]
                        hnsw["recall_at_20"][str(ef)] = round(recall_at_20(exact, found), 4)
                        cycle = iter(range(10 ** 9))
                        hnsw["search_ms"][str(ef)] = summary(timed(
                            lambda: approximate.search(queries[next(cycle) % len(queries)], 20), args.repeats))
                    finally:
                        approximate.close()
                result["hnsw"] = hnsw
    finally:
        index.close()
        server.drop_schema(schema)
    return result


def run_sqlite(path, data, n, args, queries, rng, workdir):
    from self_directing_mcp.index.vector import SqliteVectorIndex

    file = Path(workdir) / f"dense-{n}.sqlite"
    file.unlink(missing_ok=True)
    index = SqliteVectorIndex(file, dim=args.dim, extension_path=Path(path))
    try:
        seconds = load(index, data, n, args.batch)
        result = {"upsert_rows_per_s": round(n / seconds), "bytes": {"file": file.stat().st_size}}
        result.update(measure_index(index, data, n, queries, rng, repeats=args.repeats))
        result["extension_version"] = index.extension_version
        return result
    finally:
        index.close()
        file.unlink(missing_ok=True)


def run_numpy(data, n, args, queries, rng):
    from self_directing_mcp.index.vector import NumpyVectorIndex

    available = int(next((line.split()[1] for line in open("/proc/meminfo") if line.startswith("MemAvailable")), 0)) * 1024
    needed = n * args.dim * 8 * 2.2                                  # the float64 matrix, and the copy np.stack makes
    if needed > available * 0.6:
        return {"skipped": f"needs about {needed / 2**30:.1f} GiB, {available / 2**30:.1f} GiB available"}
    index = NumpyVectorIndex(dim=args.dim)                          # in memory: no persistence file
    seconds = load(index, data, n, args.batch, single_call=True)    # one call: every call re-stacks the whole matrix
    result = {"upsert_rows_per_s": round(n / seconds), "bytes": {"matrix_float64": n * args.dim * 8}}
    result["note"] = "one upsert_many call for all rows (each call rebuilds the whole matrix); no persistence"
    result.update(measure_index(index, data, n, queries, rng, repeats=args.repeats))
    return result


def environment(server, args) -> dict:
    import psycopg

    meta = {"python": platform.python_version(), "platform": platform.platform(), "cpus": os.cpu_count(),
            "psycopg": psycopg.__version__, "dim": args.dim, "batch": args.batch, "repeats": args.repeats,
            "data": args.data, "tmpfs": args.tmpfs, "shm": args.shm, "shared_buffers": args.shared_buffers,
            "maintenance_work_mem": args.maintenance_work_mem, "lan_delay_ms": args.lan_delay_ms}
    meta["server_version"] = server.execute("SHOW server_version")[0][0]
    meta["pgvector"] = server.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")[0][0]
    meta["pg_settings"] = {name: server.execute("SELECT setting || coalesce(unit, '') FROM pg_settings WHERE name = %s", (name,))[0][0]
                           for name in ("shared_buffers", "max_parallel_workers_per_gather", "max_parallel_maintenance_workers",
                                        "work_mem", "effective_cache_size", "random_page_cost")}
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="1000,10000,100000,300000")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=30, help="searches per cell (after two warm-up calls)")
    parser.add_argument("--recall-queries", type=int, default=50)
    parser.add_argument("--data", choices=("uniform", "clustered"), default="uniform")
    parser.add_argument("--backends", default="pgvector,sqlite-vector,numpy")
    parser.add_argument("--phases", default="load,hnsw", help="load (upsert, present, search) and/or hnsw")
    parser.add_argument("--sqlite-vector", default=os.environ.get("SQLITE_VECTOR_TEST_PATH"))
    parser.add_argument("--dsn-file", help="a server to use instead of a throwaway container (a private file with one DSN)")
    parser.add_argument("--tmpfs", default="3g")
    parser.add_argument("--shm", default="1g")
    parser.add_argument("--shared-buffers", default="512MB")
    parser.add_argument("--maintenance-work-mem", default="1GB")
    parser.add_argument("--ef", default="40,100,200,400", type=lambda text: [int(part) for part in text.split(",")])
    parser.add_argument("--hnsw-max", type=int, default=300_000)
    parser.add_argument("--no-hnsw", action="store_true")
    parser.add_argument("--build-timeout", type=float, default=3600)
    parser.add_argument("--lan-delay-ms", type=float, default=None,
                        help="run pgvector through a loopback proxy that delays each direction by this much (0: the proxy alone)")
    parser.add_argument("--workdir", default=None, help="where the SQLite files go (default: a temporary directory)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    args.phases = set(args.phases.split(","))
    backends = set(args.backends.split(","))
    sizes = [int(part) for part in args.sizes.split(",")]

    import pgvector_support as support

    if args.dsn_file:
        os.environ["TSUKKOMI_PGVECTOR_TEST_DSN_FILE"] = args.dsn_file
        server = support.external_server()
    else:
        server = support.start_container(tmpfs_size=args.tmpfs, shm_size=args.shm, shared_buffers=args.shared_buffers)
    proxy = None
    results = []
    try:
        meta = environment(server, args)
        conninfo = server.conninfo()
        if args.lan_delay_ms is not None:
            proxy = support.TcpProxy(server.host, server.port)
            proxy.delay = args.lan_delay_ms / 1000
            conninfo = server.conninfo(host="127.0.0.1", port=proxy.port)
        print(json.dumps(meta), flush=True)
        data = Data(args.data, args.dim)
        queries = data.queries(max(args.repeats + 2, args.recall_queries))
        workdir = args.workdir or tempfile.mkdtemp(prefix="bench-pgvector-")
        for n in sizes:
            rng = np.random.default_rng(n)
            entry = {"n": n, "data": args.data}
            for name in ("pgvector", "sqlite-vector", "numpy"):
                if name not in backends:
                    continue
                started = time.time()
                try:
                    if name == "pgvector":
                        entry[name] = run_pgvector(server, conninfo, data, n, args, queries, rng)
                    elif name == "sqlite-vector":
                        entry[name] = (run_sqlite(args.sqlite_vector, data, n, args, queries, rng, workdir)
                                       if "load" in args.phases and args.sqlite_vector else {"skipped": "no --sqlite-vector"})
                    elif "load" in args.phases:
                        entry[name] = run_numpy(data, n, args, queries, rng)
                except Exception as exc:
                    entry[name] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                print(f"n={n} {name} done in {time.time() - started:.0f}s: {json.dumps(entry.get(name))[:400]}", flush=True)
            results.append(entry)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps({"meta": meta, "results": results}, indent=1), encoding="utf-8")
    finally:
        if proxy is not None:
            proxy.stop()
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
