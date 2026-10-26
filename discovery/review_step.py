"""Optional step after the funnel: the reviewer agent decides alert / queue / drop per lead.

The reviewer's tools need the whole conversation (replies, earlier messages), so the run
records every message it reads and builds a ChatLog from them once the funnel is done.
Message ids must be numeric (true for Telegram and the synthetic data).
"""
from __future__ import annotations

from collections import Counter
from typing import Iterator, Optional

from .agent import ReviewAgent, ReviewResult
from .cost import CostTracker
from .models import Candidate, Message
from .policy import DECISIONS
from .tools import ChatLog, ToolBox

_RANK = {d: i for i, d in enumerate(DECISIONS)}


def _num(v):
    if v in (None, ""):
        return None
    s = str(v)
    return int(s) if s.lstrip("-").isdigit() else s


def to_chat(m: Message) -> dict:
    return {"id": _num(m.id), "author": m.author, "timestamp": m.timestamp, "text": m.text,
            "reply_to": _num(m.reply_to)}


class RecordingSource:
    """Passes messages through to the funnel and keeps a copy for the reviewer."""

    def __init__(self, source):
        self.source, self.seen = source, []

    def messages(self) -> Iterator[Message]:
        for m in self.source.messages():
            self.seen.append(m)
            yield m


def review_leads(leads: list[Candidate], seen: list[Message], backend, tracker: Optional[CostTracker] = None,
                 max_steps: int = 6, limit: Optional[int] = None, verbose: bool = False):
    log = ChatLog([to_chat(m) for m in seen])
    agent = ReviewAgent(backend, ToolBox(log), max_steps, tracker)
    out = []
    for c in leads[:limit] if limit else leads:
        lead = log.by_id[_num(c.message.id)]
        try:
            r = agent.review(lead)
        except Exception as e:  # one failed model call must not lose the other reviews
            r = ReviewResult(lead_id=lead["id"], reason=f"error: {e}")
        out.append((c, r))
        if verbose:
            print(f"  reviewed {lead['id']}: {r.decision} ({r.seconds:.1f}s)", flush=True)
    return out


def render_reviews(pairs: list, total_leads: int, top: int = 15) -> str:
    out = [f"\nREVIEW ({len(pairs)} of {total_leads} leads reviewed)"]
    counts = Counter(r.decision for _, r in pairs)
    out.append("  " + "   ".join(f"{d} {counts[d]}" for d in DECISIONS))
    unfinished = sum(not r.finished for _, r in pairs)
    forced = sum(r.forced for _, r in pairs)
    if unfinished or forced:
        out.append(f"  unfinished {unfinished} (defaulted to queue)   forced at the step cap {forced}")
    cost = sum(c.cost_usd + r.cost_usd for c, r in pairs)
    secs = sum(r.seconds for _, r in pairs)
    if pairs:
        out.append(f"  cost ${cost:.4f} total, ${cost / len(pairs):.5f} per lead   review time {secs:.1f}s")
    ordered = sorted(pairs, key=lambda p: (_RANK[p[1].decision], -(p[0].intent_score or 0)))
    for c, r in ordered[:top]:
        text = c.message.text if len(c.message.text) <= 80 else c.message.text[:77] + "..."
        out.append(f"  {r.decision.upper():<5} ${c.cost_usd + r.cost_usd:.4f}  {text}")
        out.append(f"        {r.reason[:110]}")
    if len(pairs) < total_leads:
        out.append(f"  ({total_leads - len(pairs)} lower-scored leads not reviewed; raise --review-max)")
    return "\n".join(out)
