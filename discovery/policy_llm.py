"""A deterministic stand-in for a model. NOT an LLM.

It follows a fixed procedure through the same tools, so the evaluation pipeline can be
validated end to end without any model: if this scores ~100%, the plumbing is right and a
real model's misses are the model's. It is also a baseline for "what a hand-written
procedure achieves".
"""
from __future__ import annotations

import json
import re

from .llm import LLMReply, ToolCall

_LEAD = re.compile(r"Lead message id (\d+) from (\S+) at")
_ANSWERED = re.compile(r"\b(sell|DMs?|in stock|shop|sorted|got one)\b", re.I)
URGENT = re.compile(r"\b(asap|today|tomorrow|urgent|right now)\b", re.I)
LARGE = re.compile(r"\b\d{2,}\s+\w+|company|office", re.I)


def _call(name, **args):
    return LLMReply(tool_calls=[ToolCall(f"p-{name}", name, args)], tokens_in=50, tokens_out=10, model="policy")


class PolicyBackend:
    name = "policy"

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMReply:
        first = messages[0]["content"]
        msg_id = int(_LEAD.search(first).group(1))
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        if not tool_msgs:
            return _call("check_replied", msg_id=msg_id)
        data = json.loads(tool_msgs[-1]["content"])
        answered = any(_ANSWERED.search(r["text"]) for r in data.get("replies", []))
        return _call("submit_assessment", answered=answered, urgent=bool(URGENT.search(first)),
                     large=bool(LARGE.search(first)), reason="fixed procedure")
