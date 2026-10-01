import pytest

from app.agent.runner import Agent
from app.agent.tools import TICKETS, calculator, create_ticket
from app.config import Settings
from app.llm.base import ChatResponse, ToolCall, Usage
from app.prompts.registry import PromptRegistry


def test_calculator_safe():
    assert calculator("24 - 7") == "17"
    assert calculator("10 / 4") == "2.5"
    with pytest.raises(ValueError):
        calculator("__import__('os').system('rm -rf /')")


def test_create_ticket_records():
    before = len(TICKETS)
    out = create_ticket("VPN broken", priority="high")
    assert len(TICKETS) == before + 1 and "TCK-" in out


def test_agent_uses_search_tool_and_answers(state):
    res = state["agent"].run("How many vacation days do employees get?")
    assert [t.tool for t in res.trace] == ["search_knowledge_base"]
    assert "24" in res.answer
    assert res.steps == 2 and not res.stopped_by_budget


class ScriptedLLM:
    """Replays a fixed list of responses, to test the loop logic in isolation."""

    provider, model = "fake", "scripted"

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools=None, temperature=0.1):
        self.calls.append(list(messages))  # snapshot: the agent mutates the same list between calls
        return self.script.pop(0)


def _tool(name, **args):
    return ChatResponse(None, [ToolCall(f"c{len(args)}", name, args)], Usage(5, 5))


def test_agent_chains_tools_and_recovers_from_errors(state):
    llm = ScriptedLLM([
        _tool("calculator", expression="not math"),       # tool error -> model sees ERROR
        _tool("calculator", expression="24 - 7"),         # retries correctly
        ChatResponse("17 days left", [], Usage(5, 5)),
    ])
    agent = Agent(state["settings"], llm, state["agent"].tools, state["prompts"])
    res = agent.run("24 minus 7?")
    assert res.answer == "17 days left"
    assert [t.error for t in res.trace] == [True, False]
    assert res.trace[1].result == "17"
    assert "ERROR" in llm.calls[1][-1]["content"]


def test_agent_step_budget(state):
    settings = Settings(llm_provider="fake", embedding_provider="hash", agent_max_steps=2, _env_file=None)
    llm = ScriptedLLM([_tool("get_current_date")] * 2 + [ChatResponse("gave up", [], Usage())])
    agent = Agent(settings, llm, state["agent"].tools, PromptRegistry(settings.prompts_dir))
    res = agent.run("loop forever")
    assert res.stopped_by_budget and res.answer == "gave up" and len(res.trace) == 2


def test_unknown_tool_is_reported_not_crashed(state):
    llm = ScriptedLLM([_tool("launch_rockets"), ChatResponse("cannot", [], Usage())])
    agent = Agent(state["settings"], llm, state["agent"].tools, state["prompts"])
    res = agent.run("do it")
    assert res.trace[0].error and "Unknown tool" in res.trace[0].result
