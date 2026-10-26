from __future__ import annotations

import sqlite3
import time

from .models import Candidate

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY AUTOINCREMENT, started REAL, source TEXT);
CREATE TABLE IF NOT EXISTS results(
  run_id INTEGER, msg_id TEXT, chat TEXT, author TEXT, ts TEXT, text TEXT,
  dropped_at TEXT, similarity REAL, intent_score REAL, urgency TEXT, deal_size TEXT,
  confidence REAL, scored_by TEXT, is_lead INTEGER, cost_usd REAL
);
CREATE INDEX IF NOT EXISTS idx_results_lead ON results(run_id, is_lead);
CREATE TABLE IF NOT EXISTS reviews(
  run_id INTEGER, msg_id TEXT, decision TEXT, reason TEXT, answered INTEGER, urgent INTEGER,
  large INTEGER, repeat_buyer INTEGER, finished INTEGER, forced INTEGER, steps INTEGER,
  model TEXT, tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL, seconds REAL
);
"""


class Store:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def start_run(self, source: str) -> int:
        cur = self.db.execute("INSERT INTO runs(started, source) VALUES (?,?)", (time.time(), source))
        self.db.commit()
        return cur.lastrowid

    def save(self, run_id: int, c: Candidate) -> None:
        m = c.message
        self.db.execute(
            "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, m.id, m.chat, m.author, m.timestamp, m.text, c.dropped_at, c.similarity,
             c.intent_score, c.urgency, c.deal_size, c.confidence, c.scored_by, int(c.is_lead), c.cost_usd))

    def save_review(self, run_id: int, c: Candidate, r) -> None:
        f = r.facts
        flag = lambda k: None if k not in f else int(f[k])
        self.db.execute(
            "INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, c.message.id, r.decision, r.reason, flag("answered"), flag("urgent"), flag("large"),
             flag("repeat_buyer"), int(r.finished), int(r.forced), r.steps, r.model, r.tokens_in,
             r.tokens_out, r.cost_usd, r.seconds))

    def commit(self) -> None:
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()
