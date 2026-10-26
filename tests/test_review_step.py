import csv
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing

from discovery import scenarios
from discovery.factory import build_pipeline
from discovery.models import Message
from discovery.policy_llm import PolicyBackend
from discovery.review_step import RecordingSource, render_reviews, review_leads, to_chat
from discovery.settings import Settings
from discovery.sources import CsvSource, JsonlSource
from discovery.storage import Store

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ReviewStepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.chat, labels = scenarios.generate(per_type=4, chatter=80, days=5, seed=8)
        self.labels = {l["lead_id"]: l["decision"] for l in labels}
        self.jsonl = os.path.join(self.tmp.name, "chat.jsonl")
        with open(self.jsonl, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(m) for m in self.chat))

    def tearDown(self):
        self.tmp.cleanup()

    def run_funnel_and_review(self, store=None):
        pipe = build_pipeline(Settings(db_path=""), use_cloud=False)
        pipe.store = store
        rec = RecordingSource(JsonlSource(self.jsonl))
        summary = pipe.run(rec, "test")
        return summary, review_leads(summary.leads, rec.seen, PolicyBackend(), pipe.tracker)

    def test_recording_source_keeps_every_message(self):
        rec = RecordingSource(JsonlSource(self.jsonl))
        self.assertEqual(len(list(rec.messages())), len(self.chat))
        self.assertEqual(len(rec.seen), len(self.chat))

    def test_reply_links_survive_jsonl_and_csv(self):
        replies = [m for m in self.chat if m["reply_to"] is not None]
        got = {m.id: m.reply_to for m in JsonlSource(self.jsonl).messages()}
        self.assertEqual(got[str(replies[0]["id"])], str(replies[0]["reply_to"]))
        path = os.path.join(self.tmp.name, "chat.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "timestamp", "author", "text", "reply_to"])
            w.writeheader()
            w.writerows({**m, "reply_to": m["reply_to"] or ""} for m in self.chat)
        got = {m.id: m.reply_to for m in CsvSource(path).messages()}
        self.assertEqual(got[str(replies[0]["id"])], str(replies[0]["reply_to"]))
        no_reply = next(m for m in self.chat if m["reply_to"] is None)
        self.assertIsNone(got[str(no_reply["id"])])

    def test_to_chat_uses_numeric_ids(self):
        d = to_chat(Message(id="12", text="t", author="a", reply_to="7"))
        self.assertEqual((d["id"], d["reply_to"]), (12, 7))
        self.assertIsNone(to_chat(Message(id="3", text="t")).get("reply_to"))

    def test_reviewed_leads_get_the_labelled_decision(self):
        summary, pairs = self.run_funnel_and_review()
        self.assertEqual(len(pairs), len(summary.leads))
        self.assertGreater(len(pairs), 5)
        for c, r in pairs:
            if int(c.message.id) in self.labels:
                self.assertEqual(r.decision, self.labels[int(c.message.id)], c.message.text)

    def test_limit_and_render(self):
        summary, _ = self.run_funnel_and_review()
        pipe = build_pipeline(Settings(db_path=""), use_cloud=False)
        rec = RecordingSource(JsonlSource(self.jsonl))
        s = pipe.run(rec, "x")
        pairs = review_leads(s.leads, rec.seen, PolicyBackend(), limit=3)
        self.assertEqual(len(pairs), 3)
        text = render_reviews(pairs, len(s.leads))
        self.assertIn("REVIEW (3 of", text)
        self.assertIn("not reviewed", text)

    def test_a_failing_backend_does_not_lose_other_reviews(self):
        class Boom:
            name = "boom"

            def chat(self, *a):
                raise TimeoutError("down")
        pipe = build_pipeline(Settings(db_path=""), use_cloud=False)
        rec = RecordingSource(JsonlSource(self.jsonl))
        s = pipe.run(rec, "x")
        pairs = review_leads(s.leads, rec.seen, Boom(), limit=2)
        self.assertEqual([r.decision for _, r in pairs], ["queue", "queue"])
        self.assertIn("down", pairs[0][1].reason)

    def test_reviews_are_saved_to_sqlite(self):
        db = os.path.join(self.tmp.name, "leads.db")
        store = Store(db)
        summary, pairs = self.run_funnel_and_review(store)
        for c, r in pairs:
            store.save_review(summary.run_id, c, r)
        store.close()
        with closing(sqlite3.connect(db)) as con:
            rows = con.execute("SELECT decision, answered FROM reviews WHERE run_id=?", (summary.run_id,)).fetchall()
        self.assertEqual(len(rows), len(pairs))
        self.assertTrue(all(a in (0, 1) for _, a in rows))

    def test_cli_end_to_end(self):
        db = os.path.join(self.tmp.name, "cli.db")
        out = subprocess.run([sys.executable, "-m", "discovery", "--jsonl", self.jsonl, "--db", db,
                              "--no-cloud", "--review", "--review-max", "5"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
        self.assertIn("REVIEW (5 of", out)
        with closing(sqlite3.connect(db)) as con:
            n = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        self.assertEqual(n, 5)

    def test_cli_without_review_is_unchanged(self):
        out = subprocess.run([sys.executable, "-m", "discovery", "--jsonl", self.jsonl, "--db", "", "--no-cloud"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
        self.assertIn("FUNNEL", out)
        self.assertNotIn("REVIEW (", out)


if __name__ == "__main__":
    unittest.main()
