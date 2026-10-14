from __future__ import annotations

import hashlib
import re
import time
from abc import ABC, abstractmethod
from typing import Sequence

import httpx
import numpy as np

from self_directing_mcp.security.mask import mask_secrets
from self_directing_mcp.request_control import check_request, current_request, request_operation


def format_query(
    query: str,
    *,
    task: str = "Given a session audit query, retrieve matching Codex session chunks",
) -> str:
    return f"Instruct: {task}\nQuery: {query}"


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n == 0.0:
        return vec
    return vec / n


class Embedder(ABC):
    dim: int

    @property
    def cache_key(self) -> str:
        identity = f"redacted-v2:{getattr(self, 'model', 'fake-v1')}:{self.dim}:{getattr(self, 'base_url', 'local')}"
        return hashlib.sha256(identity.encode()).hexdigest()

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        ...

    @abstractmethod
    def embed_queries(self, texts: Sequence[str]) -> list[np.ndarray]:
        ...


class FakeEmbedder(Embedder):
    """Deterministic bag-of-tokens embedder for offline tests."""

    SYNONYMS: dict[str, str] = {
        "curl": "http_fetch",
        "wget": "http_fetch",
        "fetch": "http_fetch",
        "http": "http_fetch",
        "api_key": "secret_tok",
        "apikey": "secret_tok",
        "secret": "secret_tok",
        "token": "secret_tok",
        "password": "secret_tok",
        "shell": "shell_tok",
        "bash": "shell_tok",
        "powershell": "shell_tok",
        "rm": "destructive",
        "delete": "destructive",
        "unlink": "destructive",
        "tool": "tool_tok",
        "function": "tool_tok",
        "call": "tool_tok",
    }

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _tokenize(self, text: str) -> list[str]:
        raw = re.findall(r"[a-z0-9_\-]+", text.lower())
        out: list[str] = []
        for t in raw:
            check_request()
            out.append(self.SYNONYMS.get(t, t))
            out.append(t)
        return out

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        toks = self._tokenize(text)
        if not toks:
            vec[0] = 1.0
            return l2_normalize(vec)
        for tok in toks:
            check_request()
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            weight = 1.0 + (h[5] / 255.0)
            vec[idx] += sign * weight
            idx2 = int.from_bytes(h[6:10], "little") % self.dim
            vec[idx2] += 0.35 * sign
        return l2_normalize(vec)

    @request_operation
    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [self._embed_one(t) for t in texts]

    @request_operation
    def embed_queries(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [self._embed_one(format_query(t)) for t in texts]


class OpenRouterEmbedder(Embedder):
    # Qwen3 Embedding accepts 32K tokens. At most 6K Unicode codepoints
    # occupy 24K UTF-8 bytes, leaving room even for byte-level tokenization.
    document_chunk_chars = 6000
    request_batch_size = 32

    @property
    def cache_key(self) -> str:
        identity = f"{super().cache_key}:split-{self.document_chunk_chars}-weighted-mean-v1"
        return hashlib.sha256(identity.encode()).hexdigest()

    def __init__(
        self,
        api_key: str,
        model: str = "qwen/qwen3-embedding-8b",
        dim: int = 1024,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.base_url = base_url.rstrip("/")

    def _call(self, texts: Sequence[str]) -> list[np.ndarray]:
        check_request()
        inputs = []
        for text in texts:
            check_request()
            inputs.append(mask_secrets(text))
        payload = {"model": self.model, "input": inputs, "dimensions": self.dim}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
            for attempt in range(3):
                check_request()
                control = current_request()
                remaining = 180.0
                if control is not None and control.deadline is not None:
                    remaining = min(remaining, max(.001, control.deadline - time.monotonic()))
                try:
                    resp = client.post(f"{self.base_url}/embeddings", json=payload, headers=headers,
                                       timeout=httpx.Timeout(remaining, connect=min(10.0, remaining)))
                    check_request()
                    resp.raise_for_status()
                    data = resp.json()["data"]
                    check_request()
                    break
                except httpx.HTTPStatusError as exc:
                    if attempt == 2 or exc.response.status_code not in (429, 500, 502, 503, 504):
                        raise
                except httpx.TransportError:
                    if attempt == 2:
                        raise
                check_request()
                delay = float(2 ** attempt)
                if control is None:
                    time.sleep(delay)
                else:
                    if control.deadline is not None:
                        delay = min(delay, max(0, control.deadline - time.monotonic()))
                    control.cancel_event.wait(delay)
                check_request()
        out: list[np.ndarray] = []
        if len(data) != len(texts) or sorted(item["index"] for item in data) != list(range(len(texts))):
            raise ValueError("embedding response indices mismatch")
        for item in sorted(data, key=lambda x: x["index"]):
            check_request()
            arr = np.asarray(item["embedding"], dtype=np.float64)
            if arr.shape != (self.dim,) or not np.isfinite(arr).all():
                raise ValueError("embedding response dimensions or values invalid")
            out.append(l2_normalize(arr))
        return out

    @request_operation
    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not texts:
            return []
        # Mask BEFORE splitting so credentials crossing a segment boundary
        # cannot evade redaction. Preserve every character of sanitized text.
        segments = []
        owners = []
        for index, original in enumerate(texts):
            check_request()
            text = mask_secrets(original)
            parts = [text[i:i + self.document_chunk_chars]
                     for i in range(0, len(text), self.document_chunk_chars)] or [""]
            for part in parts:
                check_request()
                segments.append(part)
                owners.append((index, max(1, len(part))))
        totals = np.zeros((len(texts), self.dim), dtype=np.float64)
        for offset in range(0, len(segments), self.request_batch_size):
            check_request()
            batch = segments[offset:offset + self.request_batch_size]
            values = self._call(batch)
            check_request()
            for (index, weight), value in zip(owners[offset:offset + len(batch)], values):
                check_request()
                totals[index] += weight * value
        return [l2_normalize(total) for total in totals]

    @request_operation
    def embed_queries(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not texts:
            return []
        return self._call([format_query(t) for t in texts])


def build_embedder(
    *,
    api_key: str | None,
    use_fake: bool = False,
    model: str = "qwen/qwen3-embedding-8b",
    dim: int = 1024,
    base_url: str = "https://openrouter.ai/api/v1",
) -> tuple[Embedder, bool]:
    if use_fake:
        return FakeEmbedder(dim=dim), False
    if not api_key or not api_key.strip():
        raise ValueError("OPENROUTER_API_KEY is required unless the fake embedder is explicitly enabled")
    return OpenRouterEmbedder(api_key=api_key, model=model, dim=dim, base_url=base_url), False
