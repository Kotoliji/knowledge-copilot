"""Agent loop: LLM decides -> tool runs -> result goes back -> ... until a final answer.

Guard rails: step budget, per-tool error capture (the model sees the error and can recover),
and a full trace of every step for debugging and evaluation.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.agent.tools import ToolRegistry
from app.config import Settings
from app.llm.base import ChatModel, Usage
from app.observability import AGENT_STEPS, AGENT_TOOL_CALLS, log_event
from app.prompts.registry import PromptRegistry

log = logging.getLogger(__name__)


@dataclass
class TraceStep:
    step: int
    tool: str
    arguments: dict[str, Any]
    result: str
    error: bool = False


@dataclass
class AgentResult:
    answer: str
    trace: list[TraceStep]
    steps: int
    prompt_version: str
    model: str
    usage: Usage = field(default_factory=Usage)
    stopped_by_budget: bool = False


class Agent:
    def __init__(self, settings: Settings, llm: ChatModel, tools: ToolRegistry, prompts: PromptRegistry):
        self.settings = settings
        self.llm = llm
        self.tools = tools
        self.prompts = prompts

    def run(self, question: str, history: list[dict[str, str]] | None = None) -> AgentResult:
        system = self.prompts.get("agent_system")
        messages: list[dict[str, Any]] = [{"role": "system", "content": system.template}]
        messages += history or []
        messages.append({"role": "user", "content": question})

        trace: list[TraceStep] = []
        usage = Usage()
        specs = self.tools.specs()

        for step in range(1, self.settings.agent_max_steps + 1):
            resp = self.llm.chat(messages, tools=specs, temperature=self.settings.llm_temperature)
            usage += resp.usage

            if not resp.tool_calls:
                AGENT_STEPS.observe(step)
                log_event(log, "agent_done", steps=step, tools=[t.tool for t in trace], tokens=usage.total_tokens)
                return AgentResult((resp.content or "").strip(), trace, step, system.id,
                                   resp.model or self.llm.model, usage)

            messages.append({
                "role": "assistant",
                "content": resp.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in resp.tool_calls
                ],
            })
            for tc in resp.tool_calls:
                result, error = self._execute(tc.name, tc.arguments)
                trace.append(TraceStep(step, tc.name, tc.arguments, result, error))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        # Budget exhausted: ask for a best-effort final answer without tools.
        messages.append({"role": "user", "content": "Stop using tools and give your final answer now."})
        resp = self.llm.chat(messages, temperature=self.settings.llm_temperature)
        usage += resp.usage
        AGENT_STEPS.observe(self.settings.agent_max_steps)
        log_event(log, "agent_budget_exhausted", steps=self.settings.agent_max_steps)
        return AgentResult((resp.content or "").strip(), trace, self.settings.agent_max_steps,
                           system.id, resp.model or self.llm.model, usage, stopped_by_budget=True)

    def _execute(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        try:
            tool = self.tools.get(name)
            result = str(tool.fn(**arguments))
            AGENT_TOOL_CALLS.labels(name, "ok").inc()
            log_event(log, "tool_call", tool=name, arguments=arguments, side_effect=tool.side_effect)
            return result, False
        except Exception as exc:  # the model gets the error text and may retry differently
            AGENT_TOOL_CALLS.labels(name, "error").inc()
            log.warning("tool %s failed: %s", name, exc)
            return f"ERROR: {type(exc).__name__}: {exc}", True
