def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["llm"]["provider"] == "fake"
    assert body["vector_store"]["chunks"] > 0
    assert body["prompts"]["rag_answer"] == [1, 2]


def test_ask_returns_citations_and_request_id(client):
    r = client.post("/ask", json={"question": "What is the Team plan price?"}, headers={"x-request-id": "abc"})
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "abc"
    body = r.json()
    assert body["grounded"] and body["citations"][0]["source"] == "product_faq.md"


def test_ask_validation(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_agent_endpoint(client):
    r = client.post("/agent", json={"question": "What is the SEV1 response time?"})
    assert r.status_code == 200
    assert r.json()["trace"][0]["tool"] == "search_knowledge_base"


def test_ingest_and_delete_roundtrip(client):
    r = client.post("/ingest", files={"file": ("office.md", b"# Office\n\nThe office is in Berlin.", "text/markdown")})
    assert r.status_code == 200 and r.json()["chunks"] == 1
    assert "office.md" in client.get("/documents").json()["sources"]
    assert client.post("/ask", json={"question": "Where is the office?"}).json()["grounded"]
    assert client.delete("/documents/office.md").status_code == 200
    assert "office.md" not in client.get("/documents").json()["sources"]


def test_ingest_rejects_unknown_type(client):
    r = client.post("/ingest", files={"file": ("x.exe", b"\x00", "application/octet-stream")})
    assert r.status_code == 400


def test_metrics_exposed(client):
    client.post("/ask", json={"question": "How many sick days?"})
    text = client.get("/metrics").text
    assert "llm_requests_total" in text and "rag_retrieval_top_score" in text
