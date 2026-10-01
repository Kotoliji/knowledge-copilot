# Knowledge Copilot — production-style RAG + agent service

A self-contained GenAI backend that answers questions over a company's documents **with citations**, refuses
when the documents don't contain the answer, and can **act** (tool-calling agent) — packaged the way an
LLM feature ships to production: typed config, provider abstraction, versioned prompts, Prometheus metrics,
an offline eval suite that gates CI, Docker, and an Azure (AI Foundry / Azure OpenAI) deployment with
keyless auth.

```
question ──> embed ──> Qdrant (cosine top-k) ──> prompt (versioned) ──> LLM ──> answer + [citations]
                                   ▲                                     │
   .md/.txt/.pdf ──> chunk ──> embed┘               agent loop: LLM ──> tool ──> LLM ──> … ──> answer
```

| Layer | What is here | Where |
|---|---|---|
| **LLM in prod** | One `ChatModel` interface, three real providers (Azure OpenAI, OpenAI, Ollama) + a deterministic fake for CI; retries/timeouts; token & cost accounting | [app/llm/](app/llm/) |
| **Embeddings** | Same provider abstraction for embeddings; local CPU option (FastEmbed) and an offline lexical fallback | [app/embeddings.py](app/embeddings.py) |
| **Vector DB** | Qdrant: in-memory / embedded / server, deterministic ids (idempotent re-ingest), per-document delete, metadata filter | [app/vectorstore.py](app/vectorstore.py) |
| **RAG** | Heading-aware chunking with overlap, top-k retrieval with score gate, grounded prompt, citation extraction, "I don't know" path | [app/rag/](app/rag/) |
| **Agentic AI** | Function-calling loop with tool registry, step budget, error feedback to the model, full trace; tools: KB search, calculator, date, ticket | [app/agent/](app/agent/) |
| **LLMOps** | Versioned prompt registry, Prometheus metrics (latency, tokens, USD, error rate, retrieval quality), JSON logs with request ids, Grafana dashboard + alerts, eval suite with CI gate | [app/prompts/](app/prompts/), [app/observability.py](app/observability.py), [monitoring/](monitoring/), [evals/](evals/) |
| **Cloud / Azure** | Bicep: Azure OpenAI + model deployments, ACR, Container Apps, Managed Identity (no keys), Log Analytics | [infra/azure/](infra/azure/) |

## Quick start (no API key needed)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload          # http://localhost:8000  (UI)  /docs (OpenAPI)
pytest -q                              # 24 tests, ~1 s
python -m evals.run_eval               # offline eval against evals/golden_set.jsonl
```

Defaults are `LLM_PROVIDER=fake` + `EMBEDDING_PROVIDER=hash`: the whole pipeline runs offline with a
deterministic stand-in model. Switch to real models by setting env vars (or `.env`, see [.env.example](.env.example)):

```bash
# OpenAI
LLM_PROVIDER=openai EMBEDDING_PROVIDER=openai OPENAI_API_KEY=sk-...   uvicorn app.main:app

# Azure OpenAI / AI Foundry (keyless: `az login` locally, Managed Identity in Azure)
LLM_PROVIDER=azure EMBEDDING_PROVIDER=azure AZURE_OPENAI_ENDPOINT=https://<res>.openai.azure.com/ uvicorn app.main:app

# Fully local open models
ollama pull llama3.1:8b && ollama pull nomic-embed-text
LLM_PROVIDER=ollama EMBEDDING_PROVIDER=ollama uvicorn app.main:app

# Real semantic embeddings on CPU without any API (ONNX, ~130 MB download)
pip install fastembed && EMBEDDING_PROVIDER=fastembed uvicorn app.main:app
```

Full stack with Qdrant server, Prometheus and Grafana:

```bash
cp .env.example .env
docker compose up --build
# API  http://localhost:8000     Qdrant http://localhost:6333/dashboard
# Prom http://localhost:9090     Grafana http://localhost:3000  (dashboard "Knowledge Copilot - LLMOps")
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /ask` `{question, top_k?, prompt_version?}` | RAG answer with `citations[]`, `grounded`, `prompt_version`, `usage` |
| `POST /agent` `{question, history?}` | Tool-using agent; returns `answer`, `trace[]` (every tool call + result), `steps` |
| `POST /ingest` (multipart file) | Add/replace a `.md` / `.txt` / `.pdf` document |
| `GET /documents`, `DELETE /documents/{source}` | List / remove indexed documents |
| `GET /health` | Active providers, model names, chunk count, prompt versions |
| `GET /metrics` | Prometheus exposition |
| `GET /` | Minimal UI showing answers, citations, scores and agent traces |

Every response carries an `x-request-id` header that also appears in the JSON logs. Set `API_KEY` to
require an `X-API-Key` header on the mutating/LLM endpoints.

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"How many vacation days do employees get per year?"}' | jq
```
```json
{
  "answer": "Every full-time employee receives 24 paid vacation days per calendar year. [1]",
  "citations": [{"n": 1, "source": "hr_vacation_policy.md", "heading": "Vacation and Leave Policy > Annual vacation allowance", "score": 0.4088, "text": "..."}],
  "grounded": true,
  "prompt_version": "rag_answer@v2",
  "model": "fake-llm",
  "usage": {"prompt_tokens": 262, "completion_tokens": 38}
}
```

## Design notes

**Provider abstraction.** Azure OpenAI, OpenAI and Ollama all expose the OpenAI chat/embeddings API, so one
client class ([app/llm/clients.py](app/llm/clients.py)) covers all three; the difference is auth and base URL.
Azure uses `DefaultAzureCredential` when no key is configured — the same code works with `az login` on a
laptop and with a Managed Identity in Container Apps. The `fake`/`hash` providers make the full pipeline
testable in CI in under a second with zero network.

**Grounding.** The prompt forbids answering outside the passages and demands `[n]` citations. The pipeline
parses citations out of the answer and returns `grounded=false` when there are none or when the model used
the refusal phrase; both cases increment `rag_no_context_total`. The retrieval score gate (`MIN_SCORE`) is
deliberately low — the model is the main judge of relevance; the metric `rag_retrieval_top_score` lets you
see when questions start drifting away from the corpus.

**Agent safety.** Tools are declared with JSON schemas; the loop has a hard step budget (`AGENT_MAX_STEPS`),
feeds tool exceptions back to the model as text (so it can retry with different arguments instead of
crashing), marks side-effecting tools, uses an AST-based calculator (no `eval`), and returns the entire trace
so every action is auditable.

**Prompt versioning.** Prompts are YAML files `name.vN.yaml`; the registry serves the latest or a pinned
version (`PROMPT_PINS='{"rag_answer": 1}'`), and every answer reports which one it used. Changing a prompt
means adding a file, so evals can compare versions: `python -m evals.run_eval --prompt-version 1`.

**Evaluation.** [evals/golden_set.jsonl](evals/golden_set.jsonl) holds answerable questions (expected source +
required facts) and unanswerable ones (must be refused). The runner reports retrieval hit@k, answer
accuracy, citation accuracy, refusal rate, hallucinations, p50/p95 latency, tokens and estimated cost, and
can fail the build (`--fail-under`). CI runs it offline on every PR and with real models on `main` when a
key is configured.

**Observability.** Prometheus counters/histograms for every LLM and embedding call (latency by kind, tokens
by type, estimated USD by model, errors), retrieval quality, agent tool calls and steps, HTTP latency per
endpoint. [monitoring/alerts.yml](monitoring/alerts.yml) pages on error rate, p95 latency, no-context spikes and
daily spend; the Grafana dashboard is provisioned automatically by `docker compose`.

## Deploy to Azure

Requires `az` CLI and Docker. Creates Azure OpenAI with `gpt-4.1-mini` + `text-embedding-3-small`
deployments, a Container Registry, a Container Apps environment with Log Analytics, and a user-assigned
Managed Identity with *Cognitive Services OpenAI User* + *AcrPull* — no API keys anywhere.

```bash
az login
bash infra/azure/deploy.sh          # prints the public URL
```

Then `curl https://<app>.azurecontainerapps.io/health`. Logs are in Log Analytics (`ContainerAppConsoleLogs_CL`,
one JSON object per line with `request_id`). For a persistent vector store in Azure, point `QDRANT_URL`
at a Qdrant Cloud cluster or a Qdrant container with an Azure Files volume.

## Project layout

```
app/
  config.py          typed settings (env / .env)
  main.py            FastAPI app, routes, middleware, wiring
  observability.py   Prometheus metrics, cost table, JSON logging
  embeddings.py      OpenAI/Azure/Ollama/FastEmbed/hash embedders
  vectorstore.py     Qdrant wrapper
  llm/               ChatModel interface, OpenAI-compatible clients, fake model, metrics decorator
  rag/               loaders, chunking, pipeline (ingest/retrieve/answer)
  agent/             tool registry + agent loop
  prompts/           versioned YAML prompts + registry
  static/            single-file UI
data/docs/           sample corpus (HR policy, on-call runbook, product FAQ, IT security)
evals/               golden set + eval runner
tests/               24 unit/integration tests (offline)
monitoring/          Prometheus config, alert rules, Grafana provisioning + dashboard
infra/azure/         Bicep template + deploy script
```

## What I would do next

- Hybrid retrieval (BM25 + vectors) and a cross-encoder re-ranker; measure the gain with the eval suite.
- Streaming responses (SSE) and conversation memory for `/ask`.
- LLM-as-judge metrics (faithfulness, answer relevance) in the eval runner for real-model runs.
- Semantic cache for repeated questions; per-tenant collections and auth.
- OpenTelemetry traces exported to Azure Monitor / Application Insights.
