from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np

from self_directing_mcp.index.native import NativeVectorStore


class VectorIndex(ABC):
    def refresh(self) -> None:
        """SQLite reads are current automatically; in-memory backends override this."""

    @abstractmethod
    def ids(self) -> set[str]: ...

    def upsert_many(self, pairs) -> None:
        for chunk_id, vector in pairs:
            self.upsert(chunk_id, vector)

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def upsert(self, chunk_id: str, vector: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int = 20, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def delete_ids(self, ids: Sequence[str]) -> None: ...


class NumpyVectorIndex(VectorIndex):
    def refresh(self) -> None:
        if self.persist_path and self.persist_path.exists():
            stamp = (self.persist_path.stat().st_mtime_ns, self.persist_path.stat().st_size)
            if stamp != self._stamp:
                self._load()
        elif self._stamp is not None:
            self._ids, self._mat, self._stamp = [], None, None

    def ids(self) -> set[str]:
        return set(self._ids)

    def upsert_many(self, pairs) -> None:
        if not pairs:
            return
        by_id = dict(zip(self._ids, self._mat)) if self._mat is not None else {}
        for chunk_id, vector in pairs:
            value = np.asarray(vector, dtype=np.float64)
            if value.shape != (self.dim,):
                raise ValueError("invalid vector shape")
            by_id[chunk_id] = value
        self._ids = list(by_id)
        self._mat = np.stack(list(by_id.values()))
        self._save()

    def __init__(self, dim: int = 1024, persist_path: Path | None = None) -> None:
        self.dim = dim
        self.persist_path = Path(persist_path) if persist_path else None
        self._ids: list[str] = []
        self._mat: np.ndarray | None = None
        self._stamp = None
        if self.persist_path and self.persist_path.exists():
            self._load()

    def clear(self) -> None:
        self._ids = []
        self._mat = None
        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()

    def upsert(self, chunk_id: str, vector: np.ndarray) -> None:
        v = np.asarray(vector, dtype=np.float64).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {v.shape[0]}")
        if chunk_id in self._ids:
            idx = self._ids.index(chunk_id)
            assert self._mat is not None
            self._mat[idx] = v
        else:
            self._ids.append(chunk_id)
            if self._mat is None:
                self._mat = v.reshape(1, -1)
            else:
                self._mat = np.vstack([self._mat, v.reshape(1, -1)])
        self._save()

    def delete_ids(self, ids: Sequence[str]) -> None:
        drop = set(ids)
        if not drop or not self._ids:
            return
        keep_idx = [i for i, cid in enumerate(self._ids) if cid not in drop]
        self._ids = [self._ids[i] for i in keep_idx]
        if not keep_idx:
            self._mat = None
        else:
            assert self._mat is not None
            self._mat = self._mat[keep_idx]
        self._save()

    def search(self, query: np.ndarray, top_k: int = 20, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        if not self._ids or self._mat is None:
            return []
        q = np.asarray(query, dtype=np.float64).reshape(-1)
        positions = [i for i, cid in enumerate(self._ids) if allowed_ids is None or cid in allowed_ids]
        if not positions:
            return []
        scores = self._mat[positions] @ q
        order = np.argsort(-scores)
        out: list[tuple[str, float]] = []
        for i in order[:top_k]:
            out.append((self._ids[positions[int(i)]], float(scores[int(i)])))
        return out

    def count(self) -> int:
        return len(self._ids)

    def _save(self) -> None:
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        if self._mat is None:
            if self.persist_path.exists():
                self.persist_path.unlink()
            return
        np.savez_compressed(
            self.persist_path,
            ids=np.array(self._ids, dtype=str),
            mat=self._mat,
        )
        self._stamp = (self.persist_path.stat().st_mtime_ns, self.persist_path.stat().st_size)

    def _load(self) -> None:
        assert self.persist_path is not None
        with np.load(self.persist_path, allow_pickle=False) as data:
            self._ids = [str(x) for x in data["ids"].tolist()]
            self._mat = data["mat"].copy()
        self._stamp = (self.persist_path.stat().st_mtime_ns, self.persist_path.stat().st_size)


class SqliteVectorIndex(NativeVectorStore, VectorIndex):
    """sqliteai/sqlite-vector native exact cosine search over FLOAT32 blobs."""

    table = "chunk_vectors"
    id_column = "chunk_id"


def build_vector_index(
    index_dir: Path, dim: int = 1024, *, backend: str = "sqlite-vector",
    extension_path: Path | None = None,
) -> tuple[VectorIndex, str]:
    """Use the selected backend. Native loading failures never select NumPy."""
    index_dir = Path(index_dir)
    if backend not in ("sqlite-vector", "numpy"):
        raise ValueError(f"Unknown dense backend: {backend}")
    index_dir.mkdir(parents=True, exist_ok=True)
    if backend == "numpy":
        return NumpyVectorIndex(dim=dim, persist_path=index_dir / "dense_numpy.npz"), "numpy"
    return SqliteVectorIndex(index_dir / "dense.sqlite", dim=dim, extension_path=extension_path), "sqlite-vector"
