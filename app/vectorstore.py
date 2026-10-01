"""Vector database layer (Qdrant): stores chunk embeddings + metadata, finds nearest neighbours."""

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from app.rag.chunking import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class QdrantStore:
    def __init__(self, url: str, collection: str, dim: int, api_key: str = ""):
        self.remote = url.startswith(("http://", "https://"))
        if url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        elif self.remote:
            self.client = QdrantClient(url=url, api_key=api_key or None, timeout=30)
        else:
            self.client = QdrantClient(path=url)  # embedded, persisted to a folder
        self.collection = collection
        self.dim = dim
        self.ensure_collection()

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
        )
        if self.remote:  # payload index -> fast filter/delete by document (no-op in local mode)
            self.client.create_payload_index(self.collection, "source", models.PayloadSchemaType.KEYWORD)

    def reset(self) -> None:
        self.client.delete_collection(self.collection)
        self.ensure_collection()

    def delete_source(self, source: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=_source_filter(source)),
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            models.PointStruct(
                # Deterministic id => re-ingesting the same chunk overwrites instead of duplicating.
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c.source}#{c.index}")),
                vector=v,
                payload={"text": c.text, "source": c.source, "heading": c.heading, "index": c.index},
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(self.collection, points=points, wait=True)

    def search(self, vector: list[float], top_k: int, source: str | None = None) -> list[ScoredChunk]:
        result = self.client.query_points(
            self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            query_filter=_source_filter(source) if source else None,
        )
        return [
            ScoredChunk(
                Chunk(text=p.payload["text"], source=p.payload["source"],
                      heading=p.payload["heading"], index=p.payload["index"]),
                p.score,
            )
            for p in result.points
        ]

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def sources(self) -> list[str]:
        seen: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=["source"], with_vectors=False
            )
            seen.update(p.payload["source"] for p in points)
            if offset is None:
                return sorted(seen)


def _source_filter(source: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])
