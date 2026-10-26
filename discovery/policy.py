"""The definition of "valuable to review". The ONE place to change it.

The reviewer model only judges facts about a lead (answered? urgent? large?). This function
turns those facts into a decision, and the synthetic test labels use it too, so the agent and
the ground truth can never drift apart.
"""
from __future__ import annotations

ALERT, QUEUE, DROP = "alert", "queue", "drop"
DECISIONS = (ALERT, QUEUE, DROP)
REPEAT_BUYER_MIN = 2   # earlier buying messages from the same sender needed to count as a repeat buyer


def decide(answered: bool, urgent: bool, large: bool, repeat_buyer: bool) -> tuple[str, str]:
    if answered:
        return DROP, "already answered or resolved"
    if urgent and large:
        return ALERT, "urgent and large deal, still unanswered"
    if repeat_buyer:
        return ALERT, "returning sender with repeated buying intent"
    return QUEUE, "real intent but not urgent or not large; worth a look later"
