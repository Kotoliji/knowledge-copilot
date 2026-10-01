"""RAG pipeline: ingest documents -> embed -> store; question -> retrieve -> grounded answer."""

import logging
import re
from dataclasses import dataclass, field

from app.config import Settings
from app.embeddings import Embedder
from app.llm.base import ChatModel, Usage
from app.llm.providers import NO_ANSWER
from app.observability import RAG_NO_ANSWER, RETRIEVAL_TOP_SCORE, log_event
from app.prompts.registry import PromptRegistry
from app.rag.chunking import chunk_document
from app.rag.loaders import iter_files, load_bytes
from app.vectorstore import QdrantStore, ScoredChunk

log = logging.getLogger(__name__)



@dataclass
class Citation:
    n: int
    source: str
    heading: str
    score: float
    text: str


@dataclass
class RagAnswer:
    answer: str
    citations: list[Citation]
    grounded: bool
    prompt_version: str
    model: str
    usage: Usage = field(default_factory=Usage)


class RagPipeline:
    def __init__(self, settings: Settings, embedder: Embedder, store: QdrantStore, llm: ChatModel,
                 prompts: PromptRegistry):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.prompts = prompts

    # ---- Ingestion -----------------------------------------------------------
    def ingest_text(self, text: str, source: str) -> int:
        chunks = chunk_document(text, source, self.settings.chunk_size, self.settings.chunk_overlap)
        self.store.delete_source(source)  # replace stale chunks of a re-uploaded doc
        if chunks:
            self.store.upsert(chunks, self.embedder.embed([c.text for c in chunks]))
        log_event(log, "ingested", source=source, chunks=len(chunks))
        return len(chunks)

    def ingest_file(self, name: str, data: bytes) -> int:
        return self.ingest_text(load_bytes(name, data), name)

    def ingest_directory(self, root: str) -> dict[str, int]:
        return {name: self.ingest_file(name, data) for name, data in iter_files(root)}

    # ---- Retrieval -----------------------------------------------------------
    def retrieve(self, question: str, top_k: int | None = None, source: str | None = None) -> list[ScoredChunk]:
        vector = self.embedder.embed([question])[0]
        hits = self.store.search(vector, top_k or self.settings.top_k, source=source)
        if hits:
            RETRIEVAL_TOP_SCORE.observe(hits[0].score)
        return [h for h in hits if h.score >= self.settings.min_score]

    @staticmethod
    def format_context(hits: list[ScoredChunk]) -> str:
        return "\n\n".join(f"[{i}] (source: {h.chunk.source})\n{h.chunk.text}" for i, h in enumerate(hits, 1))

    # ---- Generation ----------------------------------------------------------
    def answer(self, question: str, top_k: int | None = None, prompt_version: int | None = None) -> RagAnswer:
        prompt = self.prompts.get("rag_answer", prompt_version)
        hits = self.retrieve(question, top_k)
        if not hits:
            RAG_NO_ANSWER.inc()
            return RagAnswer(NO_ANSWER, [], False, prompt.id, self.llm.model)

        messages = [{"role": "user", "content": prompt.render(context=self.format_context(hits), question=question)}]
        resp = self.llm.chat(messages, temperature=self.settings.llm_temperature)
        text = (resp.content or "").strip()

        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        citations = [
            Citation(i, h.chunk.source, h.chunk.heading, round(h.score, 4), h.chunk.text)
            for i, h in enumerate(hits, 1)
            if i in cited
        ]
        grounded = bool(citations) and NO_ANSWER not in text
        if not grounded:
            RAG_NO_ANSWER.inc()
        log_event(log, "rag_answer", question=question, hits=len(hits), top_score=hits[0].score,
                  grounded=grounded, prompt=prompt.id, tokens=resp.usage.total_tokens)
        return RagAnswer(text, citations, grounded, prompt.id, resp.model or self.llm.model, resp.usage)
