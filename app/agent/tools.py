"""Tools the agent may call. Each tool = JSON schema (what the LLM sees) + a Python function."""

import ast
import json
import operator
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.rag.pipeline import RagPipeline


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., str]
    side_effect: bool = False  # mutating tools are logged separately and can be disabled

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self.tools.values()]

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise KeyError(f"Unknown tool: {name}")
        return self.tools[name]


# --- Safe calculator (no eval) ----------------------------------------------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Only numbers and + - * / ** % are allowed")


def calculator(expression: str) -> str:
    tree = ast.parse(expression.strip(), mode="eval")
    result = _eval_node(tree.body)
    return str(int(result)) if float(result).is_integer() else f"{result:.4f}".rstrip("0")


# --- Ticket store (in-memory stand-in for Jira/ServiceNow) --------------------------
TICKETS: list[dict[str, Any]] = []


def create_ticket(title: str, description: str = "", priority: str = "medium") -> str:
    ticket = {"id": f"TCK-{uuid.uuid4().hex[:6].upper()}", "title": title,
              "description": description, "priority": priority, "status": "open"}
    TICKETS.append(ticket)
    return json.dumps(ticket)


def build_tools(rag: RagPipeline, allow_side_effects: bool = True) -> ToolRegistry:
    reg = ToolRegistry()

    def search_knowledge_base(query: str, top_k: int = 4) -> str:
        hits = rag.retrieve(query, top_k=top_k)
        return rag.format_context(hits) if hits else "No relevant passages found."

    reg.register(Tool(
        "search_knowledge_base",
        "Semantic search over the company knowledge base (policies, product docs, procedures). "
        "Returns numbered passages with their source file.",
        {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for, phrased as a full question"},
            "top_k": {"type": "integer", "description": "How many passages to return (1-8)", "default": 4},
        }, "required": ["query"]},
        search_knowledge_base,
    ))
    reg.register(Tool(
        "calculator",
        "Evaluate an arithmetic expression, e.g. '20 * 1.5 + 3'. Supports + - * / ** %.",
        {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        calculator,
    ))
    reg.register(Tool(
        "get_current_date",
        "Return today's date in ISO format (YYYY-MM-DD).",
        {"type": "object", "properties": {}},
        lambda: date.today().isoformat(),
    ))
    if allow_side_effects:
        reg.register(Tool(
            "create_ticket",
            "Create a support/IT ticket on behalf of the user. Only when explicitly asked.",
            {"type": "object", "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
            }, "required": ["title"]},
            create_ticket,
            side_effect=True,
        ))
    return reg
