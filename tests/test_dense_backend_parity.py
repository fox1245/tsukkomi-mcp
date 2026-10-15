"""The dense backends answer the same questions the same way.

The reference is ``NumpyVectorIndex``, the exact scan. ``PgVectorIndex`` (a real server; the throwaway container of
``pgvector_support``, skipped with a reason without one) and ``SqliteVectorIndex`` (the official sqlite-vector library; it
needs SQLITE_VECTOR_TEST_PATH or a library at the default location, skipped with a reason without one) are compared with it
on the same seeded unit float32 vectors, the same queries and the same ``allowed_ids``.

What "the same" means (stated, not hoped for):

* the hits are an exact top-k: the reference scores of the returned ids equal the reference top-k scores rank by rank, so the
  ids may differ from the reference only inside a group of exactly equal scores (which has no defined order); with no tie
  around the k-th place the ids and their order are identical;
* every reported score is within ``TOLERANCE`` of the reference score of that id. The backends compute in float32 (the
  reference in float64); the largest difference seen over these queries was 1.5e-7 for pgvector and 2.0e-7 for sqlite-vector
  (float32 epsilon is 1.2e-7), so 1e-6 is a margin of five;
* scores are cosine similarities (1 is identical), a zero vector scores 0 everywhere, and ``allowed_ids`` filters before the
  limit.

One documented difference: ``NumpyVectorIndex`` returns a dot product, which is a cosine only for unit vectors (what every
embedder here produces); pgvector and sqlite-vector return the cosine itself (see the last test)."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dense_support import assert_same_topk, exact_cosine, make_ids, normalized_vectors
from self_directing_mcp import settings_core
from self_directing_mcp.index.pgvector import PgVectorIndex
from self_directing_mcp.index.vector import NumpyVectorIndex, SqliteVectorIndex

TOLERANCE = 1e-6
CANDIDATES = ["pgvector", "sqlite-vector"]
DIMS = [1024, 24]
COUNT = 2000


def sqlite_vector_path() -> Path | None:
    configured = os.environ.get("SQLITE_VECTOR_TEST_PATH")
    if configured:
        return Path(configured)
    return settings_core.default_sqlite_vector_path()


def build_backends(request, tmp_path_factory, dim: int, label: str):
    """``(backends, skipped, cleanup)``: every backend that can be had, and why the others cannot."""
    backends, skipped, cleanups = {"numpy": NumpyVectorIndex(dim=dim)}, {}, []
    path = sqlite_vector_path()
    if path is None or not Path(path).is_file():
        skipped["sqlite-vector"] = ("the sqlite-vector library is not available offline: set SQLITE_VECTOR_TEST_PATH "
                                    "(scripts/setup_sqlite_vector.py installs the official release)")
    else:
        try:
            directory = tmp_path_factory.mktemp(f"sqlite-{label}")
            sqlite_index = SqliteVectorIndex(directory / "dense.sqlite", dim=dim, extension_path=Path(path))
            backends["sqlite-vector"] = sqlite_index
            cleanups.append(sqlite_index.close)
        except Exception as exc:
            skipped["sqlite-vector"] = f"the sqlite-vector library could not be loaded ({type(exc).__name__})"
    try:
        server = request.getfixturevalue("pgvector_server")
    except pytest.skip.Exception as exc:
        skipped["pgvector"] = str(exc)
    else:
        schema = "parity_" + os.urandom(4).hex()
        pg = PgVectorIndex(server.conninfo(), dim=dim, schema=schema, policy="parity")
        backends["pgvector"] = pg
        cleanups.append(pg.close)
        cleanups.append(lambda: server.drop_schema(schema))
    return backends, skipped, cleanups


@pytest.fixture(scope="module", params=DIMS, ids=[f"dim{dim}" for dim in DIMS])
def world(request, tmp_path_factory):
    dim = request.param
    ids = make_ids(COUNT)
    vectors = normalized_vectors(COUNT, dim, seed=dim)
    vectors[11] = vectors[10]                       # an exact duplicate: two ids with the same score
    vectors[700:706] = vectors[700]                 # a group of six identical vectors
    vectors[500] = 0.0                              # a zero vector (no cosine)
    backends, skipped, cleanups = build_backends(request, tmp_path_factory, dim, f"dim{dim}")
    for index in backends.values():
        for start in range(0, COUNT, 500):
            index.upsert_many(list(zip(ids[start:start + 500], vectors[start:start + 500])))
    queries = np.concatenate([normalized_vectors(26, dim, seed=1000 + dim), vectors[[11, 702, 500, 3]]])
    rng = np.random.default_rng(dim)
    filters = {
        "empty": set(), "one": {ids[1234]}, "seven": {ids[n] for n in rng.choice(COUNT, 7, replace=False)},
        "threehundred": {ids[n] for n in rng.choice(COUNT, 300, replace=False)},
        "half": {ids[n] for n in rng.choice(COUNT, COUNT // 2, replace=False)},
        "ninety": {ids[n] for n in rng.choice(COUNT, 1800, replace=False)},
        "all": set(ids), "with-unknown-ids": {ids[n] for n in range(0, COUNT, 3)} | {f"unknown-{n}" for n in range(500)},
        "tie-group": {ids[n] for n in range(698, 708)} | {ids[10], ids[11]},
    }
    try:
        yield SimpleNamespace(dim=dim, ids=ids, vectors=vectors, backends=backends, skipped=skipped, queries=queries,
                              filters=filters, reference_cache={})
    finally:
        for cleanup in reversed(cleanups):
            try:
                cleanup()
            except Exception:
                pass


def backend(world, name):
    if name not in world.backends:
        pytest.skip(world.skipped[name])
    return world.backends[name]


def reference_scores(world, query) -> np.ndarray:
    """The reference score of every id for ``query``, in id order: the exact scan's full ranking."""
    key = query.tobytes()
    if key not in world.reference_cache:
        ranking = dict(world.backends["numpy"].search(query, COUNT))
        world.reference_cache[key] = np.array([ranking[cid] for cid in world.ids])
    return world.reference_cache[key]


def test_the_reference_is_the_exact_cosine_for_unit_vectors(world):
    for query in world.queries[:6]:
        assert np.max(np.abs(reference_scores(world, query) - exact_cosine(world.vectors, query))) <= TOLERANCE


@pytest.mark.parametrize("name", CANDIDATES)
def test_the_same_ids_count_and_membership(world, name):
    index = backend(world, name)
    assert index.count() == world.backends["numpy"].count() == COUNT
    assert index.ids() == world.backends["numpy"].ids() == set(world.ids)
    probe = world.ids[::7] + ["unknown-a", "unknown-b", ""]
    assert index.present(probe) == world.backends["numpy"].present(probe) == set(world.ids[::7])
    assert index.present([]) == set() and index.present(["unknown"]) == set()


@pytest.mark.parametrize("name", CANDIDATES)
@pytest.mark.parametrize("k", [1, 5, 20, 100])
def test_unfiltered_top_k_equals_the_reference(world, name, k):
    index = backend(world, name)
    for number, query in enumerate(world.queries):
        assert_same_topk(index.search(query, k), world.ids, reference_scores(world, query), k, tolerance=TOLERANCE,
                         label=f"{name} dim {world.dim} query {number} k {k}")


@pytest.mark.parametrize("name", CANDIDATES)
@pytest.mark.parametrize("filter_name", ["empty", "one", "seven", "threehundred", "half", "ninety", "all",
                                         "with-unknown-ids", "tie-group"])
def test_filtered_top_k_equals_the_reference(world, name, filter_name):
    index = backend(world, name)
    allowed = world.filters[filter_name]
    for number, query in enumerate(world.queries):
        got = index.search(query, 20, allowed)
        reference = reference_scores(world, query)
        assert_same_topk(got, world.ids, reference, 20, allowed=allowed, tolerance=TOLERANCE,
                         label=f"{name} dim {world.dim} filter {filter_name} query {number}")
        assert {cid for cid, _ in got} <= allowed
    assert index.search(world.queries[0], 20, set()) == []


@pytest.mark.parametrize("name", CANDIDATES)
def test_search_limits_and_empty_cases(world, name):
    index = backend(world, name)
    query = world.queries[0]
    assert len(index.search(query, COUNT + 500)) == COUNT
    assert index.search(query, 0) == [] and index.search(query, -2) == []
    assert world.backends["numpy"].search(query, 0) == [] and world.backends["numpy"].search(query, -2) == []


@pytest.mark.parametrize("name", CANDIDATES)
def test_a_zero_vector_scores_zero_on_every_backend(world, name):
    index = backend(world, name)
    zero_query = np.zeros(world.dim, dtype=np.float32)
    for candidate in (world.backends["numpy"], index):
        assert {score for _cid, score in candidate.search(zero_query, 10)} == {0.0}
    assert dict(index.search(world.queries[0], COUNT))[world.ids[500]] == 0.0


@pytest.mark.parametrize("name", CANDIDATES)
def test_the_candidates_break_ties_by_id_in_byte_order(world, name):
    """Not required of the reference (NumPy's order inside a tie is arbitrary) but the other two agree with each other."""
    index = backend(world, name)
    hits = index.search(world.vectors[702], 6)
    assert [cid for cid, _ in hits] == world.ids[700:706]
    assert [cid for cid, _ in index.search(world.vectors[11], 2)] == world.ids[10:12]


@pytest.mark.parametrize("name", CANDIDATES)
def test_a_random_sequence_of_writes_keeps_the_backends_equal_to_a_model(request, tmp_path_factory, name):
    backends, skipped, cleanups = build_backends(request, tmp_path_factory, 12, f"model-{name}")
    try:
        if name not in backends:
            pytest.skip(skipped[name])
        candidate, reference = backends[name], backends["numpy"]
        rng = np.random.default_rng(42)
        model: dict[str, np.ndarray] = {}
        pool = make_ids(300, "m")
        for step in range(40):
            action = rng.choice(["upsert", "overwrite", "delete", "clear", "upsert"])
            if action == "clear" and rng.random() < 0.7:
                action = "delete"
            if action in ("upsert", "overwrite"):
                chosen = [pool[n] for n in rng.choice(300, size=int(rng.integers(1, 80)), replace=False)]
                if action == "overwrite" and model:
                    chosen = list(model)[: int(rng.integers(1, len(model) + 1))]
                fresh = normalized_vectors(len(chosen), 12, seed=10_000 + step)
                for index in (candidate, reference):
                    index.upsert_many(list(zip(chosen, fresh)))
                model.update(zip(chosen, fresh))
            elif action == "delete":
                doomed = [pool[n] for n in rng.choice(300, size=int(rng.integers(0, 60)), replace=False)] + ["never-stored"]
                for index in (candidate, reference):
                    index.delete_ids(doomed)
                for cid in doomed:
                    model.pop(cid, None)
            else:
                candidate.clear()
                reference.clear()
                model.clear()
            assert candidate.ids() == reference.ids() == set(model)
            assert candidate.count() == reference.count() == len(model)
            ids = sorted(model)
            query = normalized_vectors(1, 12, seed=20_000 + step)[0]
            expected = exact_cosine(np.array([model[cid] for cid in ids]), query) if ids else np.array([])
            allowed = set(ids[::2]) | {"never-stored"}
            for scope in (None, allowed):
                got = candidate.search(query, 15, scope)
                if not ids:
                    assert got == []
                else:
                    assert_same_topk(got, ids, expected, 15, allowed=scope, tolerance=TOLERANCE, label=f"{name} step {step}")
    finally:
        for cleanup in reversed(cleanups):
            try:
                cleanup()
            except Exception:
                pass


@pytest.mark.parametrize("name", CANDIDATES)
def test_numpy_returns_a_dot_product_so_it_needs_unit_vectors_and_the_others_return_the_cosine(request, tmp_path_factory, name):
    """The one documented divergence: for vectors that are not unit length the reference is a dot product."""
    backends, skipped, cleanups = build_backends(request, tmp_path_factory, 24, f"scaled-{name}")
    try:
        if name not in backends:
            pytest.skip(skipped[name])
        vectors = normalized_vectors(20, 24, seed=3)
        scaled = vectors * np.float32(3.0)
        for index in (backends[name], backends["numpy"]):
            index.upsert_many(list(zip(make_ids(20), scaled)))
        query = vectors[4]
        cosine = exact_cosine(scaled, query)
        got = dict(backends[name].search(query, 20))
        assert all(abs(got[cid] - cosine[n]) <= TOLERANCE for n, cid in enumerate(make_ids(20)))
        dot = dict(backends["numpy"].search(query, 20))
        assert dot[make_ids(20)[4]] == pytest.approx(3.0, abs=1e-5) and got[make_ids(20)[4]] == pytest.approx(1.0, abs=1e-5)
    finally:
        for cleanup in reversed(cleanups):
            try:
                cleanup()
            except Exception:
                pass
