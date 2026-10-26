"""The reviewer agent: given a lead, produce alert / queue / drop.

The model drives the loop. It may call the context tools (up to max_steps) and ends by calling
submit_assessment with three judgments: answered, urgent, large. It does NOT pick the decision.
Code turns those judgments, plus a repeat-buyer count computed in code, into the decision using
discovery/policy.py. The loop only runs tools, feeds results back, validates and enforces the cap.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .cost import CostTracker
from .llm import LLMBackend
from .policy import ALERT, DECISIONS, DROP, QUEUE, REPEAT_BUYER_MIN, decide
from .tools import TOOL_SPECS, ToolBox

SYSTEM = """You help review sales leads found in a group chat. For each lead you answer three yes/no questions; a separate program turns your answers into a decision.

Questions:
- answered: has the buyer already been answered? True if a reply offers to sell, sends prices, or says to check DMs, or the buyer says it is resolved. Replies such as "no idea", "following", "same here" do NOT answer the buyer.
- urgent: does the buyer need it within a day or two (asap, today, tomorrow)?
- large: is it a business purchase or many units (about 20 or more)?

How to work:
- Call check_replied first and read the replies carefully.
- The tools need no arguments; they act on the lead you are reviewing.
- Then call submit_assessment. Write the reason first, then answered, urgent and large, and make the answers agree with your reason."""

SUBMIT_SPEC = {
    "name": "submit_assessment",
    "description": "Submit your answers for this lead. Call this exactly once, when done. Write the reason first, then the three answers, and make them agree with the reason.",
    "input_schema": {"type": "object",
                     "properties": {"reason": {"type": "string", "description": "one sentence on what the replies and the lead say"},
                                    "answered": {"type": "boolean"},
                                    "urgent": {"type": "boolean"},
                                    "large": {"type": "boolean"}},
                     "required": ["answered", "urgent", "large"]},
}
FACTS = ("answered", "urgent", "large")


@dataclass
class ReviewResult:
    lead_id: int
    decision: str = QUEUE
    reason: str = ""
    finished: bool = False          # False -> the agent never produced a valid assessment
    facts: dict = field(default_factory=dict)        # answered/urgent/large from the model, repeat_buyer from code
    steps: int = 0
    tool_calls: list = field(default_factory=list)   # [(name, args), ...]
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    cost_usd: float = 0.0
    seconds: float = 0.0
    forced: bool = False            # True -> the last step was a forced "submit now"
    trace: list = field(default_factory=list)        # readable log of what happened


def _lead_prompt(msg: dict) -> str:
    return (f"Lead message id {msg['id']} from {msg['author']} at {msg['timestamp']}:\n"
            f"\"{msg['text']}\"\n\nReview this lead.")


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false", "yes", "no"):
        return v.strip().lower() in ("true", "yes")
    return None


def _check_assessment(args: dict) -> tuple[dict | None, str | None]:
    """Small models send "true" as a string; accept that, reject anything else."""
    facts = {k: _as_bool(args.get(k)) for k in FACTS}
    missing = [k for k, v in facts.items() if v is None]
    if missing:
        return None, f"{', '.join(missing)} must be true or false"
    return facts, None


def _parse_text_assessment(text: str) -> dict | None:
    """Small models often answer with plain JSON text instead of calling the tool."""
    for m in re.finditer(r"\{.*?\}", text or "", re.S):
        try:
            d = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(d, dict) and _check_assessment(d)[0] is not None:
            return d
    return None


class ReviewAgent:
    def __init__(self, backend: LLMBackend, toolbox: ToolBox, max_steps: int = 6,
                 tracker: CostTracker | None = None):
        self.backend, self.toolbox, self.max_steps, self.tracker = backend, toolbox, max_steps, tracker

    def review(self, msg: dict) -> ReviewResult:
        t0 = time.perf_counter()
        res = self._run(msg)
        res.seconds = time.perf_counter() - t0
        if self.tracker:
            self.tracker.record(res, "5_reviewer", res.model, res.tokens_in, res.tokens_out)
        return res

    def _run(self, msg: dict) -> ReviewResult:
        res = ReviewResult(lead_id=msg["id"])
        context = {"msg_id": msg["id"], "author": msg["author"], "before_id": msg["id"]}
        messages = [{"role": "user", "content": _lead_prompt(msg)}]
        tools = TOOL_SPECS + [SUBMIT_SPEC]

        for step in range(1, self.max_steps + 1):
            step_tools = tools
            if step == self.max_steps and step > 1:
                step_tools, res.forced = [SUBMIT_SPEC], True
                messages.append({"role": "user", "content": "This is your last step. Call submit_assessment now with your best answers."})
                res.trace.append(f"step {step} forcing submission (only submit_assessment offered)")
            reply = self.backend.chat(SYSTEM, messages, step_tools)
            res.steps, res.model = step, reply.model or res.model
            res.tokens_in += reply.tokens_in
            res.tokens_out += reply.tokens_out
            messages.append({"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls})
            if reply.text.strip():
                res.trace.append(f"step {step} model says: {reply.text.strip()[:200]}")

            if not reply.tool_calls:
                parsed = _parse_text_assessment(reply.text)
                if parsed:
                    return self._finish(res, msg, parsed, "text")
                res.trace.append(f"step {step} no tool call and no valid assessment; asking again")
                messages.append({"role": "user", "content": "Call submit_assessment now with answered, urgent, large and a reason."})
                continue

            for tc in reply.tool_calls:
                if tc.name == "submit_assessment":
                    _, err = _check_assessment(tc.args)
                    if err is None:
                        return self._finish(res, msg, tc.args, "tool")
                    result = {"error": err}
                    res.trace.append(f"step {step} submit_assessment rejected: {err}")
                else:
                    res.tool_calls.append((tc.name, tc.args))
                    result = self.toolbox.call(tc.name, tc.args, context)
                    res.trace.append(f"step {step} tool {tc.name}({tc.args}) -> "
                                     f"{'ERROR' if 'error' in result else 'ok'}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                 "content": json.dumps(result)})

        res.decision, res.finished = QUEUE, False
        res.reason = f"agent did not finish within {self.max_steps} steps; defaulted to queue for a human"
        res.trace.append(res.reason)
        return res

    def _finish(self, res: ReviewResult, msg: dict, args: dict, via: str) -> ReviewResult:
        facts, _ = _check_assessment(args)
        repeat = self.toolbox.sender_buying_count(msg["author"], msg["id"]) >= REPEAT_BUYER_MIN
        res.facts = {**facts, "repeat_buyer": repeat}
        res.decision, policy_reason = decide(**res.facts)
        said = str(args.get("reason", "")).strip()
        res.reason = f"{policy_reason}" + (f" | model: {said}" if said else "")
        res.finished = True
        res.trace.append(f"assessment via {via}: {res.facts} -> {res.decision}")
        return res
