import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app, build_app_state


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_provider="fake", embedding_provider="hash", qdrant_url=":memory:",
                    docs_dir="data/docs", ingest_on_startup=True, api_key="", _env_file=None)


@pytest.fixture
def state(settings):
    s = build_app_state(settings)
    s["rag"].ingest_directory(settings.docs_dir)
    return s


@pytest.fixture
def client(settings, monkeypatch):
    # lifespan() calls get_settings() directly; routes get it via Depends -> patch both paths
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
