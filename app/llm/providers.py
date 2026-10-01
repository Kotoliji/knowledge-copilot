"""Chat model implementations + a metrics decorator that wraps any of them."""

import json
import re
import time
from typing import Any

from openai import OpenAI

from app.config import Settings
from app.llm.base import ChatModel, ChatResponse, ToolCall, Usage
from app.llm.clients import build_openai_client
from app.observability import LLM_LATENCY, LLM_REQUESTS, record_usage


class OpenAIChatModel:
    """Works for OpenAI, Azure OpenAI (model = deployment name) and Ollama."""

    def __init__(self, client: OpenAI, model: str, provider: str):
        self.client = client
        self.model = model
        self.provider = provider

    def chat(self, messages, tools=None, temperature=0.1) -> ChatResponse:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        tool_calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = Usage(
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )
        return ChatResponse(content=msg.content, tool_calls=tool_calls, usage=usage, model=resp.model)


NO_ANSWER = "I don't know based on the provided documents."


class FakeChatModel:
    """Deterministic offline stand-in for an LLM, used by tests, CI and key-less demos.

    - With tools: first calls `search_knowledge_base`, then answers from the tool output.
    - Without tools: answers from passage [1] of the prompt with a [1] citation.
    - "Grounding check": if fewer than a third of the question's content words occur in the
      top passage, it refuses - a crude imitation of what a real model does with the rules.
    It is NOT intelligent - it only exercises the full pipeline end to end.
    """

    provider = "fake"
    model = "fake-llm"

    def chat(self, messages, tools=None, temperature=0.1) -> ChatResponse:
        prompt_tokens = sum(len(str(m.get("content") or "")) for m in messages) // 4
        user_msg = next(m["content"] for m in reversed(messages) if m["role"] == "user")

        if tools:
            tool_msgs = [m for m in messages if m["role"] == "tool"]
            if not tool_msgs:
                call = ToolCall(id="call_1", name="search_knowledge_base", arguments={"query": user_msg})
                return ChatResponse(None, [call], Usage(prompt_tokens, 10), self.model)
            question = next(m["content"] for m in messages if m["role"] == "user")
            passages = _parse_passages(tool_msgs[-1]["content"])
        else:
            question = user_msg.rsplit("Question:", 1)[-1]
            passages = _parse_passages(user_msg)

        if not passages or not _overlaps(question, passages[0]):
            return ChatResponse(NO_ANSWER, [], Usage(prompt_tokens, 10), self.model)
        text = _first_sentences(passages[0], 2)
        return ChatResponse(f"{text} [1]", [], Usage(prompt_tokens, len(text) // 4), self.model)


_PASSAGE = re.compile(r"^\[(\d+)\] \(source: [^)]*\)\n(.*?)(?=\n\[\d+\] \(source: |\n\nQuestion:|\Z)", re.S | re.M)
_STOP = set("a an the and or of to in on for is are was were be been do does did i you we it this that what how "
            "when where who which can my our your with at by from as if not no much many get must often per "
            "there here does".split())


def _parse_passages(text: str) -> list[str]:
    """Extract passage bodies from the `[n] (source: ...)` blocks that RagPipeline.format_context writes."""
    return [m.group(2).strip() for m in _PASSAGE.finditer(text)]


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3 and w not in _STOP}


def _overlaps(question: str, passage: str) -> bool:
    q = _stems(question)
    words = set(re.findall(r"[a-z0-9]+", passage.lower()))
    matched = sum(any(w.startswith(s) or (len(w) >= 4 and s.startswith(w)) for w in words) for s in q)
    return bool(q) and matched * 3 >= len(q)


def _first_sentences(text: str, n: int) -> str:
    lines = [line.strip("#-* ") for line in text.strip().splitlines() if line.strip()]
    if len(lines) > 1 and lines[0][-1:] not in ".!?:":  # first line is a heading path, not prose
        lines = lines[1:]
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(lines))
    return " ".join(sentences[:n]).strip()


class InstrumentedChatModel:
    """Decorator: adds Prometheus latency/token/cost/error metrics to any ChatModel."""

    def __init__(self, inner: ChatModel):
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model

    def chat(self, messages, tools=None, temperature=0.1) -> ChatResponse:
        kind = "chat_tools" if tools else "chat"
        start = time.perf_counter()
        try:
            resp = self.inner.chat(messages, tools=tools, temperature=temperature)
        except Exception:
            LLM_REQUESTS.labels(self.provider, self.model, kind, "error").inc()
            raise
        finally:
            LLM_LATENCY.labels(self.provider, self.model, kind).observe(time.perf_counter() - start)
        LLM_REQUESTS.labels(self.provider, self.model, kind, "ok").inc()
        record_usage(self.provider, self.model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return resp


def build_chat_model(settings: Settings) -> ChatModel:
    provider = settings.llm_provider
    if provider == "fake":
        model: ChatModel = FakeChatModel()
    else:
        name = {
            "openai": settings.openai_chat_model,
            "azure": settings.azure_chat_deployment,
            "ollama": settings.ollama_chat_model,
        }[provider]
        model = OpenAIChatModel(build_openai_client(settings, provider), name, provider)
    return InstrumentedChatModel(model)
