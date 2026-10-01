from app.embeddings import HashEmbedder
from app.rag.pipeline import NO_ANSWER


def test_hash_embedder_is_normalised_and_similar_for_similar_text():
    e = HashEmbedder(256)
    a, b, c = e.embed(["vacation days per year", "how many vacation days", "database encryption keys"])
    dot = lambda x, y: sum(i * j for i, j in zip(x, y, strict=True))
    assert abs(dot(a, a) - 1) < 1e-6
    assert dot(a, b) > dot(a, c)


def test_ingest_is_idempotent(state):
    before = state["store"].count()
    state["rag"].ingest_directory(state["settings"].docs_dir)
    assert state["store"].count() == before


def test_retrieval_finds_right_source(state):
    hits = state["rag"].retrieve("How many vacation days do employees get?")
    assert hits and hits[0].chunk.source == "hr_vacation_policy.md"


def test_answer_is_grounded_with_citation(state):
    res = state["rag"].answer("How many vacation days do full-time employees get per year?")
    assert res.grounded
    assert "24" in res.answer
    assert res.citations and res.citations[0].source == "hr_vacation_policy.md"
    assert res.prompt_version == "rag_answer@v2"


def test_prompt_version_pinning(state):
    res = state["rag"].answer("What is the SEV1 response time?", prompt_version=1)
    assert res.prompt_version == "rag_answer@v1"


def test_offtopic_question_is_refused(state):
    res = state["rag"].answer("What is the capital of France?")
    assert not res.grounded
    assert res.answer == NO_ANSWER or not res.citations


def test_delete_source_removes_chunks(state):
    store = state["store"]
    store.delete_source("product_faq.md")
    assert "product_faq.md" not in store.sources()
    assert not state["rag"].retrieve("How much does the Team plan cost?", source="product_faq.md")
