"""Embeddings: text -> vector, so that similar meaning => nearby vectors (cosine similarity)."""

import hashlib
import math
import re
import time
from typing import Protocol

from openai import OpenAI

from app.config import Settings
from app.llm.clients import build_openai_client
from app.observability import LLM_LATENCY, LLM_REQUESTS, record_usage


class Embedder(Protocol):
    provider: str
    model: str

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """OpenAI / Azure OpenAI / Ollama embeddings endpoint, batched."""

    batch_size = 64

    def __init__(self, client: OpenAI, model: str, provider: str):
        self.client = client
        self.model = model
        self.provider = provider
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed(["dimension probe"])[0])
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            start = time.perf_counter()
            try:
                resp = self.client.embeddings.create(model=self.model, input=batch)
            except Exception:
                LLM_REQUESTS.labels(self.provider, self.model, "embedding", "error").inc()
                raise
            LLM_LATENCY.labels(self.provider, self.model, "embedding").observe(time.perf_counter() - start)
            LLM_REQUESTS.labels(self.provider, self.model, "embedding", "ok").inc()
            if resp.usage:
                record_usage(self.provider, self.model, resp.usage.prompt_tokens, 0)
            vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        self._dim = self._dim or (len(vectors[0]) if vectors else None)
        return vectors


class FastEmbedEmbedder:
    """Real semantic embeddings on CPU, no API key (ONNX model, ~130 MB download on first use)."""

    provider = "fastembed"

    def __init__(self, model: str):
        from fastembed import TextEmbedding  # optional dependency

        self.model = model
        self._model = TextEmbedding(model_name=model)
        self._dim = len(self.embed(["dimension probe"])[0])

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]


_STOPWORDS = set(
    "a an the and or of to in on for is are was were be been do does did i you we it this that "
    "what how when where who which can my our your with at by from as if not no".split()
)


class HashEmbedder:
    """Offline lexical embedder (feature hashing of words + bigrams).

    Matches on shared words, not meaning - good enough for tests/CI, not for production.
    """

    provider = "hash"
    model = "hash-bow"

    def __init__(self, dim: int = 512):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _tokens(self, text: str) -> list[str]:
        words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]
        words = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words]  # crude stemming
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self._dim
            for tok in self._tokens(text):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
                vec[h % self._dim] += 1.0 if h & 1 << 31 else -1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def build_embedder(settings: Settings) -> Embedder:
    p = settings.embedding_provider
    if p == "hash":
        return HashEmbedder(settings.hash_embedding_dim)
    if p == "fastembed":
        return FastEmbedEmbedder(settings.fastembed_model)
    model = {
        "openai": settings.openai_embedding_model,
        "azure": settings.azure_embedding_deployment,
        "ollama": settings.ollama_embedding_model,
    }[p]
    return OpenAIEmbedder(build_openai_client(settings, p), model, p)
