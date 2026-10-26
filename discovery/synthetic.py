"""Generate a labelled fake group chat as CSV.

python -m discovery.synthetic --days 3 --per-day 1000 --out data/synthetic.csv

label=1 -> real purchase intent, label=0 -> everything else (incl. hard negatives
that contain buying keywords but are not leads).
"""
from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

ITEMS = ["standing desk", "solar panel", "used laptop", "CRM license", "3D printer",
         "office chair", "security camera", "espresso machine", "projector", "e-bike"]
PLURAL = {"standing desk": "standing desks", "solar panel": "solar panels", "used laptop": "laptops",
          "CRM license": "licenses", "3D printer": "3D printers", "office chair": "office chairs",
          "security camera": "security cameras", "espresso machine": "espresso machines",
          "projector": "projectors", "e-bike": "e-bikes"}

LEADS = [
    "Looking to buy {n} {items} this week, any recommendations?",
    "How much does a {item} cost? Need to order today",
    "Can anyone recommend a supplier for {items}? Budget is around ${b}",
    "Need a quote for {n} {items} for our office",
    "Where can I buy {items} in bulk?",
    "Anyone selling a {item}? I'll take it if the price is right",
    "We're ready to purchase a {item}, who can send an invoice?",
    "What's the price of a {item} and is it in stock? Want to order tomorrow",
    "Interested in buying {n} {items}, please send prices",
    "Who sells {items} and do they deliver? Need them asap",
    "Looking for a vendor for {items}, our company needs {n} units",
    "How much for {n} {items}? Ready to buy now",
]

AMBIGUOUS_LEADS = [  # real intent, but vague: should land near the fallback band
    "thinking about getting a {item} soon, what do they go for?",
    "anyone have a {item} they'd part with?",
    "might need a {item} next month, any idea on pricing?",
]

HARD_NEGATIVES = [  # contain buying keywords, but no purchase intent
    "The price of {items} is crazy lol",
    "I bought a {item} last year and it broke after a month",
    "how much did you pay for dinner last night?",
    "need coffee asap",
    "my cost of living is insane this year",
    "can anyone recommend a good movie for tonight?",
    "where can I get a decent pizza around here?",
    "haha that {item} meme is great, the price tag in the picture made me laugh",
    "we used to sell {items} but stopped last year",
    "is the group chat available for new members to join?",
    "just got my {item}, works great, thanks for the tips yesterday",
    "rent prices here are ridiculous, don't even ask",
    "I'm looking for a free ride to the airport on friday",
]

CHATTER = [
    "good morning everyone", "gm", "lol", "ok", "thanks!", "haha yes", "agreed", "same here",
    "what time is the meetup?", "anyone watching the game tonight?", "congrats!!", "nice one",
    "this weather is something else", "sent you a DM", "can you resend the photo?", "brb",
    "that's hilarious", "who's coming on saturday?", "happy birthday!", "see you all tomorrow",
    "did anyone see the news today?", "I'll check and let you know", "makes sense", "no worries",
    "interesting, never thought of it that way", "good point", "👍", "😂😂", "wow", "true",
    "just finished work, finally", "traffic was terrible this morning", "the new update broke my app again",
    "can someone explain how the voting works?", "I disagree but fair enough",
    "https://example.com/article/12345", "check this out https://example.com/video", "yes", "no", "maybe later",
]

AUTHORS = [f"user_{i:03d}" for i in range(1, 61)]


def _fill(tpl: str, rng: random.Random) -> str:
    item = rng.choice(ITEMS)
    return tpl.format(item=item, items=PLURAL[item], n=rng.choice([2, 5, 10, 20, 50]),
                      b=rng.choice([500, 1200, 5000, 20000]))


def generate(days: int, per_day: int, lead_rate: float, seed: int):
    rng = random.Random(seed)
    start = datetime(2026, 10, 1, 7, 0)
    rows, n = [], 0
    for d in range(days):
        for _ in range(per_day):
            r = rng.random()
            if r < lead_rate:
                text, label = _fill(rng.choice(LEADS), rng), 1
            elif r < lead_rate * 1.35:
                text, label = _fill(rng.choice(AMBIGUOUS_LEADS), rng), 1
            elif r < lead_rate * 1.35 + 0.16:
                text, label = _fill(rng.choice(HARD_NEGATIVES), rng), 0
            else:
                text, label = rng.choice(CHATTER), 0
            ts = start + timedelta(days=d, seconds=rng.randint(0, 16 * 3600))
            rows.append((ts, rng.choice(AUTHORS), text, label))
    rows.sort(key=lambda x: x[0])
    return [{"id": i, "timestamp": ts.isoformat(), "author": a, "text": t, "label": l}
            for i, (ts, a, t, l) in enumerate(rows, 1)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--per-day", type=int, default=1000)
    p.add_argument("--lead-rate", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="data/synthetic.csv")
    a = p.parse_args()
    rows = generate(a.days, a.per_day, a.lead_rate, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "timestamp", "author", "text", "label"])
        w.writeheader()
        w.writerows(rows)
    leads = sum(r["label"] for r in rows)
    print(f"wrote {len(rows)} messages ({leads} leads) to {a.out}")


if __name__ == "__main__":
    main()
