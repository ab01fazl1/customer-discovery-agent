"""Planted review scenarios: a fake chat with threads, repeat senders and replies,
plus the correct reviewer decision for each lead.

python -m discovery.scenarios --per-type 10 --out data/review

Writes <out>_chat.jsonl (id, timestamp, author, text, reply_to) and
<out>_labels.jsonl (lead_id, scenario, decision, reason).

The decision comes from discovery/policy.py, the working definition of "valuable to review".
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .policy import ALERT, DROP, QUEUE, REPEAT_BUYER_MIN, decide
from .synthetic import AUTHORS, CHATTER, PLURAL, ITEMS

LEAD_TEXT = {
    ("high", "large"): [
        "Need a quote for {n} {items} for our office, must order today",
        "We need {n} {items} delivered asap, who can send an invoice?",
        "Looking for a vendor for {items}, company needs {n} units by tomorrow",
    ],
    ("medium", "large"): [
        "Looking for a vendor for {items}, our company needs {n} units next month",
        "Need a quote for {n} {items} for the new branch, budget approved",
    ],
    ("low", "small"): [
        "thinking about getting a {item} soon, what do they go for?",
        "anyone have a {item} they'd part with?",
        "might need a {item} next month, any idea on pricing?",
    ],
}
HISTORY_TEXT = [
    "how much do {items} cost these days?",
    "anyone know a good place to get {items}?",
    "which {item} would you recommend?",
]
OFFER_REPLY = [
    "We sell {items}, sent you a DM with prices",
    "I can do {items} at a good price, check your DMs",
    "Our shop has them in stock: https://example.com/shop",
]
PEER_NOISE_REPLY = ["no idea, sorry", "good question, following", "I was wondering the same", "lol same"]
RESOLVED_REPLY = ["never mind, got one from a friend, thanks all", "all sorted, thanks everyone"]


@dataclass(frozen=True)
class Scenario:
    name: str
    urgency: str        # high | medium | low
    deal: str           # large | small
    answer: str         # none | offer | peer_noise | resolved
    history: int        # earlier intent messages from the same sender


SCENARIOS = [
    Scenario("hot", "high", "large", "none", 0),
    Scenario("hot_answered", "high", "large", "offer", 0),
    Scenario("hot_peer_noise", "high", "large", "peer_noise", 0),
    Scenario("medium_large", "medium", "large", "none", 0),
    Scenario("vague", "low", "small", "none", 0),
    Scenario("vague_returning", "low", "small", "none", 2),
    Scenario("vague_resolved", "low", "small", "resolved", 0),
]


def expected_decision(s: Scenario) -> tuple[str, str]:
    """Ground truth: the scenario's planted facts run through the same policy the agent uses."""
    return decide(answered=s.answer in ("offer", "resolved"), urgent=s.urgency == "high",
                  large=s.deal == "large", repeat_buyer=s.history >= REPEAT_BUYER_MIN)


def _fill(tpl: str, rng: random.Random, large: bool = False, item: str | None = None) -> str:
    item = item or rng.choice(ITEMS)
    n = rng.choice([20, 50]) if large else rng.choice([1, 2, 3])
    return tpl.format(item=item, items=PLURAL[item], n=n)


def generate(per_type: int, chatter: int, days: int, seed: int):
    rng = random.Random(seed)
    start = datetime(2026, 10, 1, 7, 0)
    span = days * 24 * 3600
    events, labels, buyer = [], [], 0

    def add(key, ts, author, text, reply_key=None):
        events.append({"key": key, "ts": ts, "author": author, "text": text, "reply_key": reply_key})

    for sc in SCENARIOS:
        for i in range(per_type):
            buyer += 1
            author = f"buyer_{buyer:03d}"
            lead_ts = start + timedelta(seconds=rng.randint(2 * 24 * 3600, span))
            lead_key = f"lead:{sc.name}:{i}"
            large = sc.deal == "large"
            item = rng.choice(ITEMS)
            add(lead_key, lead_ts, author,
                _fill(rng.choice(LEAD_TEXT[(sc.urgency, sc.deal)]), rng, large, item))
            for h in range(sc.history):
                add(f"hist:{lead_key}:{h}", lead_ts - timedelta(hours=rng.randint(20, 40) * (h + 1)),
                    author, _fill(rng.choice(HISTORY_TEXT), rng))
            if sc.answer == "offer":
                add(f"ans:{lead_key}", lead_ts + timedelta(minutes=rng.randint(5, 60)),
                    f"seller_{rng.randint(1, 5):02d}", _fill(rng.choice(OFFER_REPLY), rng, item=item), lead_key)
            elif sc.answer == "peer_noise":
                add(f"ans:{lead_key}", lead_ts + timedelta(minutes=rng.randint(2, 20)),
                    rng.choice(AUTHORS), rng.choice(PEER_NOISE_REPLY), lead_key)
            elif sc.answer == "resolved":
                add(f"ans:{lead_key}", lead_ts + timedelta(hours=rng.randint(1, 3)),
                    author, rng.choice(RESOLVED_REPLY), lead_key)
            decision, reason = expected_decision(sc)
            labels.append({"lead_key": lead_key, "scenario": sc.name, "decision": decision, "reason": reason})

    for j in range(chatter):
        add(f"chat:{j}", start + timedelta(seconds=rng.randint(0, span)),
            rng.choice(AUTHORS), rng.choice(CHATTER))

    events.sort(key=lambda e: e["ts"])
    ids = {e["key"]: i for i, e in enumerate(events, 1)}
    chat = [{"id": ids[e["key"]], "timestamp": e["ts"].isoformat(), "author": e["author"],
             "text": e["text"], "reply_to": ids[e["reply_key"]] if e["reply_key"] else None}
            for e in events]
    out_labels = [{"lead_id": ids[l.pop("lead_key")], **l} for l in labels]
    return chat, out_labels


def _write(path: str, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-type", type=int, default=10)
    p.add_argument("--chatter", type=int, default=400)
    p.add_argument("--days", type=int, default=5)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", default="data/review")
    a = p.parse_args()
    chat, labels = generate(a.per_type, a.chatter, a.days, a.seed)
    _write(f"{a.out}_chat.jsonl", chat)
    _write(f"{a.out}_labels.jsonl", labels)
    counts = {d: sum(1 for l in labels if l["decision"] == d) for d in (ALERT, QUEUE, DROP)}
    print(f"wrote {len(chat)} messages, {len(labels)} labelled leads {counts} to {a.out}_*.jsonl")


if __name__ == "__main__":
    main()
