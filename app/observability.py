"""LLMOps telemetry: Prometheus metrics, token/cost accounting, JSON logs with request ids."""

import contextvars
import json
import logging
import sys

from prometheus_client import Counter, Histogram

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# USD per 1M tokens (input, output). Approximate list prices - check the provider pricing page.
PRICES_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}

LLM_REQUESTS = Counter(
    "llm_requests_total", "LLM/embedding API calls", ["provider", "model", "kind", "status"]
)
LLM_TOKENS = Counter("llm_tokens_total", "Tokens consumed", ["provider", "model", "type"])
LLM_COST = Counter("llm_cost_usd_total", "Estimated spend in USD", ["provider", "model"])
LLM_LATENCY = Histogram(
    "llm_latency_seconds",
    "LLM call latency",
    ["provider", "model", "kind"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
)
RETRIEVAL_TOP_SCORE = Histogram(
    "rag_retrieval_top_score",
    "Similarity score of the best retrieved chunk (low = likely unanswerable question)",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
RAG_NO_ANSWER = Counter("rag_no_context_total", "Questions refused because nothing relevant was found")
AGENT_TOOL_CALLS = Counter("agent_tool_calls_total", "Tool invocations by the agent", ["tool", "status"])
AGENT_STEPS = Histogram("agent_steps", "LLM turns per agent run", buckets=(1, 2, 3, 4, 5, 6, 8, 10))
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP latency", ["method", "path", "status"]
)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = PRICES_PER_1M.get(model, (0.0, 0.0))
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def record_usage(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> None:
    LLM_TOKENS.labels(provider, model, "prompt").inc(prompt_tokens)
    LLM_TOKENS.labels(provider, model, "completion").inc(completion_tokens)
    LLM_COST.labels(provider, model).inc(estimate_cost(model, prompt_tokens, completion_tokens))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, msg: str, **fields) -> None:
    logger.info(msg, extra={"extra_fields": fields})
