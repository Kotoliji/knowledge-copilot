"""FastAPI service: /ask (RAG), /agent (tool-using agent), /ingest, /health, /metrics, / (UI)."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from app.agent.runner import Agent
from app.agent.tools import TICKETS, build_tools
from app.config import Settings, get_settings
from app.embeddings import build_embedder
from app.llm.providers import build_chat_model
from app.observability import HTTP_LATENCY, log_event, request_id_var, setup_logging
from app.prompts.registry import PromptRegistry
from app.rag.pipeline import RagPipeline
from app.vectorstore import QdrantStore

log = logging.getLogger("app")
STATIC = Path(__file__).parent / "static"


def build_app_state(settings: Settings) -> dict[str, Any]:
    """Wires every component. Used by the API, the eval script and tests."""
    embedder = build_embedder(settings)
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, embedder.dim, settings.qdrant_api_key)
    llm = build_chat_model(settings)
    prompts = PromptRegistry(settings.prompts_dir, settings.prompt_pins)
    rag = RagPipeline(settings, embedder, store, llm, prompts)
    agent = Agent(settings, llm, build_tools(rag), prompts)
    return {"settings": settings, "embedder": embedder, "store": store, "llm": llm,
            "prompts": prompts, "rag": rag, "agent": agent}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    state = build_app_state(settings)
    app.state.components = state
    log_event(log, "startup", llm=f"{state['llm'].provider}/{state['llm'].model}",
              embeddings=f"{state['embedder'].provider}/{state['embedder'].model}", qdrant=settings.qdrant_url)
    if settings.ingest_on_startup and Path(settings.docs_dir).is_dir() and state["store"].count() == 0:
        report = state["rag"].ingest_directory(settings.docs_dir)
        log_event(log, "startup_ingest", files=len(report), chunks=sum(report.values()))
    yield


app = FastAPI(title="Knowledge Copilot", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id_var.set(request.headers.get("x-request-id") or uuid.uuid4().hex[:12])
    start = time.perf_counter()
    response = await call_next(request)
    path = request.url.path if request.url.path in {"/ask", "/agent", "/ingest", "/health"} else "other"
    HTTP_LATENCY.labels(request.method, path, response.status_code).observe(time.perf_counter() - start)
    response.headers["x-request-id"] = request_id_var.get()
    return response


def components(request: Request) -> dict[str, Any]:
    return request.app.state.components


def require_api_key(request: Request, settings: Settings = Depends(get_settings)) -> None:
    if settings.api_key and request.headers.get("x-api-key") != settings.api_key:
        raise HTTPException(401, "Invalid or missing X-API-Key")


# ---- Schemas ---------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)
    prompt_version: int | None = None


class AgentRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=20)


# ---- Routes ----------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health(c: dict = Depends(components)):
    return {
        "status": "ok",
        "llm": {"provider": c["llm"].provider, "model": c["llm"].model},
        "embeddings": {"provider": c["embedder"].provider, "model": c["embedder"].model, "dim": c["embedder"].dim},
        "vector_store": {"url": c["settings"].qdrant_url, "chunks": c["store"].count()},
        "prompts": c["prompts"].list_all(),
    }


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/ask", dependencies=[Depends(require_api_key)])
def ask(req: AskRequest, c: dict = Depends(components)):
    try:
        result = c["rag"].answer(req.question, req.top_k, req.prompt_version)
    except FileNotFoundError:
        raise HTTPException(400, f"Unknown prompt version {req.prompt_version}") from None
    return {"request_id": request_id_var.get(), **asdict(result)}


@app.post("/agent", dependencies=[Depends(require_api_key)])
def agent(req: AgentRequest, c: dict = Depends(components)):
    result = c["agent"].run(req.question, req.history)
    return {"request_id": request_id_var.get(), **asdict(result)}


@app.post("/ingest", dependencies=[Depends(require_api_key)])
async def ingest(file: UploadFile = File(...), c: dict = Depends(components)):
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 10 MB)")
    try:
        chunks = c["rag"].ingest_file(file.filename or "upload.txt", data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"source": file.filename, "chunks": chunks, "total_chunks": c["store"].count()}


@app.get("/documents", dependencies=[Depends(require_api_key)])
def documents(c: dict = Depends(components)):
    return {"sources": c["store"].sources(), "chunks": c["store"].count()}


@app.delete("/documents/{source:path}", dependencies=[Depends(require_api_key)])
def delete_document(source: str, c: dict = Depends(components)):
    c["store"].delete_source(source)
    return {"deleted": source, "chunks": c["store"].count()}


@app.get("/tickets", dependencies=[Depends(require_api_key)])
def tickets():
    return {"tickets": TICKETS}
