"""The swappable model interface for the reviewer agent.

A backend answers ONE chat turn. It gets a neutral message list and neutral tool
specs, and returns an LLMReply. Each real backend (Ollama, Claude) converts to and
from its own wire format; the agent loop never sees that format.

Neutral messages:
  {"role": "user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": <json string>}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Union


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMReply:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""


class LLMBackend(Protocol):
    name: str

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMReply: ...


class ScriptedBackend:
    """Test stand-in: plays back prepared replies in order. An item may also be a
    function (messages -> LLMReply) to react to what the agent sent."""

    name = "scripted"

    def __init__(self, script: list[Union[LLMReply, Callable[[list[dict]], LLMReply]]]):
        self.script, self.calls = list(script), []

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMReply:
        self.calls.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        if not self.script:
            raise RuntimeError("ScriptedBackend ran out of replies")
        item = self.script.pop(0)
        return item(messages) if callable(item) else item
