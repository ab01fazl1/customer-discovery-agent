"""Shared types. Every stage and every source speaks only these."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Message:
    id: str
    text: str
    author: str = ""
    timestamp: str = ""
    chat: str = ""
    # Ground truth for synthetic/test data ONLY. Stages must never read this;
    # only the evaluation in the pipeline summary does.
    label: Optional[int] = None
    reply_to: Optional[str] = None   # id of the message this one replies to


@dataclass
class Candidate:
    """A message travelling through the funnel, accumulating results."""

    message: Message
    dropped_at: Optional[str] = None
    keyword_hit: bool = False    # stage 1 saw a buying keyword; stage 2 uses the looser threshold
    similarity: Optional[float] = None
    intent_score: Optional[float] = None
    urgency: str = "low"
    deal_size: str = "unknown"
    confidence: float = 1.0
    scored_by: str = ""
    needs_fallback: bool = False
    is_lead: bool = False
    cost_usd: float = 0.0
    trace: list = field(default_factory=list)
