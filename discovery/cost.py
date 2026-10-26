from __future__ import annotations

from collections import defaultdict

# USD per million tokens (input, output). Local models are free.
PRICES = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-5-5": (3.0, 15.0),
}


class CostTracker:
    def __init__(self):
        self.by_stage = defaultdict(lambda: {"calls": 0, "tokens_in": 0, "tokens_out": 0, "usd": 0.0})

    def record(self, candidate, stage: str, model: str, tokens_in: int = 0, tokens_out: int = 0) -> float:
        p_in, p_out = PRICES.get(model, (0.0, 0.0))
        usd = (tokens_in * p_in + tokens_out * p_out) / 1_000_000
        row = self.by_stage[stage]
        row["calls"] += 1
        row["tokens_in"] += tokens_in
        row["tokens_out"] += tokens_out
        row["usd"] += usd
        candidate.cost_usd += usd
        return usd

    @property
    def total_usd(self) -> float:
        return sum(r["usd"] for r in self.by_stage.values())
