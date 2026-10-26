"""Ollama backend for the reviewer agent: translates our neutral format to /api/chat.

The three convert_* functions are pure so they can be tested without a server.
"""
from __future__ import annotations

import itertools
import json

from .backends import _post
from .llm import LLMReply, ToolCall

_ids = itertools.count(1)


def to_ollama_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools]


def to_ollama_messages(system: str, messages: list[dict]) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "assistant":
            msg = {"role": "assistant", "content": m.get("content", "")}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"function": {"name": c.name, "arguments": c.args}} for c in m["tool_calls"]]
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_name": m["name"], "content": m["content"]})
        else:
            out.append({"role": "user", "content": m["content"]})
    return out


def _text_tool_calls(text: str, tool_names) -> list:
    """Some models (llama3.1) write the call as JSON in the message text instead of tool_calls."""
    dec, calls, i = json.JSONDecoder(), [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        i = end
        if isinstance(obj, dict) and obj.get("name") in tool_names:
            args = obj.get("parameters", obj.get("arguments")) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            calls.append(ToolCall(id=f"ollama-{next(_ids)}", name=obj["name"],
                                  args=args if isinstance(args, dict) else {}))
    return calls


def from_ollama_response(resp: dict, model: str, tool_names=()) -> LLMReply:
    msg = resp.get("message", {})
    calls = []
    for c in msg.get("tool_calls") or []:
        fn = c.get("function", {})
        args = fn.get("arguments") or {}
        calls.append(ToolCall(id=f"ollama-{next(_ids)}", name=fn.get("name", ""),
                              args=args if isinstance(args, dict) else {}))
    if not calls and tool_names:
        calls = _text_tool_calls(msg.get("content") or "", tool_names)
    return LLMReply(text=msg.get("content") or "", tool_calls=calls,
                    tokens_in=resp.get("prompt_eval_count", 0), tokens_out=resp.get("eval_count", 0), model=model)


class OllamaChat:
    name = "ollama"

    def __init__(self, url: str = "http://localhost:11434", model: str = "qwen2.5:7b", timeout: int = 300):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMReply:
        resp = _post(f"{self.url}/api/chat", {
            "model": self.model, "stream": False, "options": {"temperature": 0},
            "messages": to_ollama_messages(system, messages), "tools": to_ollama_tools(tools),
        }, timeout=self.timeout)
        return from_ollama_response(resp, self.model, {t["name"] for t in tools})
