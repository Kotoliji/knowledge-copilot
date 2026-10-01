"""Central configuration. Every value can be overridden with an env var or .env file."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Which backends to use ---------------------------------------------
    # "fake" / "hash" run fully offline (tests, CI, demo without keys).
    llm_provider: Literal["azure", "openai", "ollama", "fake"] = "fake"
    embedding_provider: Literal["azure", "openai", "ollama", "fastembed", "hash"] = "hash"

    # --- OpenAI ----------------------------------------------------------------
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Azure OpenAI / Azure AI Foundry ---------------------------------------
    # If azure_openai_api_key is empty, keyless auth (Managed Identity / az login) is used.
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_chat_deployment: str = "gpt-4.1-mini"
    azure_embedding_deployment: str = "text-embedding-3-small"

    # --- Ollama (local open-source models, OpenAI-compatible API) ---------------
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_chat_model: str = "llama3.1:8b"
    ollama_embedding_model: str = "nomic-embed-text"

    # --- Local embeddings ------------------------------------------------------
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    hash_embedding_dim: int = 512

    # --- Vector database (Qdrant) ----------------------------------------------
    # ":memory:" = in-process, "http://host:6333" = server, anything else = local folder.
    qdrant_url: str = ":memory:"
    qdrant_api_key: str = ""
    qdrant_collection: str = "knowledge"

    # --- RAG / agent behaviour -------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    min_score: float = 0.05
    agent_max_steps: int = 6
    llm_temperature: float = 0.1
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3

    # --- Prompts (versioned YAML files) ----------------------------------------
    prompts_dir: str = "app/prompts"
    # Pin versions, e.g. PROMPT_PINS='{"rag_answer": 1}'. Unpinned -> latest.
    prompt_pins: dict[str, int] = {}

    # --- Service ---------------------------------------------------------------
    docs_dir: str = "data/docs"
    ingest_on_startup: bool = True
    api_key: str = ""  # if set, requests must send X-API-Key
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
