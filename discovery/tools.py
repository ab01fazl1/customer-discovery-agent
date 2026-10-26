"""Tools the reviewer agent can call. Plain functions over a chat log; no model code here.

Every tool returns a small JSON-serialisable dict. Bad input comes back as
{"error": ...} instead of raising, so a confused model can recover.
"""
from __future__ import annotations

import inspect
import json
import re
from collections import defaultdict
from typing import Any

from .stages import INTENT_KEYWORDS

# The funnel's own buying-keyword check plus a few phrasings it misses. A heuristic; on real
# data this is the first thing to replace with the funnel's embedding score.
BUYING_HINT = re.compile(INTENT_KEYWORDS.pattern + r"|\b(place to get|where to get|place to buy)\b", re.I)

MAX_TEXT = 300
MAX_ITEMS = 20
INT_ARGS = ("msg_id", "before_id", "limit")


def _coerce(args: dict) -> dict:
    """Small models send numbers as strings ("414") or floats (414.0)."""
    out = dict(args)
    for k in INT_ARGS:
        v = out.get(k)
        if isinstance(v, str) and v.strip().lstrip("-").isdigit():
            out[k] = int(v)
        elif isinstance(v, float) and v.is_integer():
            out[k] = int(v)
    return out


def _view(m: dict) -> dict:
    return {"id": m["id"], "author": m["author"], "timestamp": m["timestamp"],
            "text": m["text"][:MAX_TEXT], "reply_to": m.get("reply_to")}


class ChatLog:
    def __init__(self, messages: list[dict]):
        self.messages = sorted(messages, key=lambda m: m["id"])
        self.by_id = {m["id"]: m for m in self.messages}
        self.by_author: dict[str, list[dict]] = defaultdict(list)
        self.children: dict[int, list[dict]] = defaultdict(list)
        for m in self.messages:
            self.by_author[m["author"]].append(m)
            if m.get("reply_to") is not None:
                self.children[m["reply_to"]].append(m)

    @classmethod
    def from_jsonl(cls, path: str) -> "ChatLog":
        with open(path, encoding="utf-8") as f:
            return cls([json.loads(line) for line in f if line.strip()])


class ToolBox:
    def __init__(self, log: ChatLog):
        self.log = log
        self._tools = {
            "get_thread": self.get_thread,
            "get_sender_history": self.get_sender_history,
            "check_replied": self.check_replied,
        }

    def get_thread(self, msg_id: int) -> dict:
        """The whole reply thread the message belongs to, oldest first."""
        m = self.log.by_id.get(msg_id)
        if m is None:
            return {"error": f"no message with id {msg_id}"}
        root = m
        while root.get("reply_to") is not None and root["reply_to"] in self.log.by_id:
            root = self.log.by_id[root["reply_to"]]
        thread, stack = [], [root]
        while stack:
            cur = stack.pop()
            thread.append(cur)
            stack.extend(self.log.children.get(cur["id"], []))
        thread.sort(key=lambda x: x["id"])
        return {"root_id": root["id"], "count": len(thread),
                "messages": [_view(x) for x in thread[:MAX_ITEMS]]}

    def get_sender_history(self, author: str, before_id: int, limit: int = 10) -> dict:
        """Earlier messages from the same sender (before `before_id`), oldest first."""
        limit = max(1, min(int(limit), MAX_ITEMS))
        earlier = [x for x in self.log.by_author.get(author, []) if x["id"] < before_id]
        return {"author": author, "total_earlier": len(earlier),
                "earlier_buying_messages": sum(bool(BUYING_HINT.search(x["text"])) for x in earlier),
                "messages": [{**_view(x), "buying_intent": bool(BUYING_HINT.search(x["text"]))}
                             for x in earlier[-limit:]]}

    def sender_buying_count(self, author: str, before_id: int) -> int:
        """Earlier messages from this sender that look like buying intent. Plain code, not a model call."""
        return self.get_sender_history(author, before_id, MAX_ITEMS)["earlier_buying_messages"]

    def check_replied(self, msg_id: int) -> dict:
        """Direct replies to the message. Whether they actually answer it is for the caller to judge."""
        if msg_id not in self.log.by_id:
            return {"error": f"no message with id {msg_id}"}
        replies = self.log.children.get(msg_id, [])
        return {"msg_id": msg_id, "reply_count": len(replies),
                "replies": [_view(x) for x in replies[:MAX_ITEMS]]}

    def call(self, name: str, args: dict[str, Any], context: dict | None = None) -> dict:
        """Run a tool. `context` supplies defaults for missing arguments (the lead under review)."""
        fn = self._tools.get(name)
        if fn is None:
            return {"error": f"unknown tool {name!r}; available: {sorted(self._tools)}"}
        params = inspect.signature(fn).parameters
        args = _coerce(args or {})
        ignored = [k for k in args if k not in params]
        clean = {k: v for k, v in args.items() if k in params}
        for k in params:
            if k not in clean and context and k in context:
                clean[k] = context[k]
        try:
            result = fn(**clean)
        except (TypeError, ValueError) as e:
            return {"error": f"bad arguments for {name}: {e}"}
        if ignored and "error" not in result:
            result["ignored_arguments"] = ignored
        return result


# Provider-neutral definitions. Each model backend converts these to its own format.
# Every argument is optional and defaults to the lead under review.
TOOL_SPECS = [
    {"name": "get_thread",
     "description": "Get the full reply thread that the lead belongs to, oldest first. No arguments needed.",
     "input_schema": {"type": "object", "properties": {"msg_id": {"type": "integer"}}}},
    {"name": "get_sender_history",
     "description": "Get earlier messages from the lead's sender, with a count of how many looked like buying intent. No arguments needed.",
     "input_schema": {"type": "object",
                      "properties": {"author": {"type": "string"},
                                     "before_id": {"type": "integer"},
                                     "limit": {"type": "integer", "description": "max messages, default 10"}}}},
    {"name": "check_replied",
     "description": "List the direct replies to the lead. A reply is not necessarily an answer; read it. No arguments needed.",
     "input_schema": {"type": "object", "properties": {"msg_id": {"type": "integer"}}}},
]
