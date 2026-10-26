import csv
import tempfile
import unittest
from pathlib import Path

from discovery.cost import CostTracker
from discovery.factory import build_pipeline
from discovery.models import Candidate, Message
from discovery.settings import Settings
from discovery.sources import CsvSource, JsonlSource
from discovery.synthetic import generate


class FakeSource:
    """Any object with .messages() works -- this is the whole source contract."""

    def __init__(self, texts):
        self.texts = texts

    def messages(self):
        for i, t in enumerate(self.texts):
            yield Message(id=str(i), text=t)


class PipelineTests(unittest.TestCase):
    def pipe(self):
        return build_pipeline(Settings(db_path=""), use_cloud=False)

    def test_custom_source_plugs_in(self):
        s = self.pipe().run(FakeSource(["good morning everyone", "How much does a solar panel cost? Need to order today"]))
        self.assertEqual(s.total, 2)
        self.assertEqual(len(s.leads), 1)
        self.assertEqual(s.leads[0].urgency, "high")

    def test_hard_negatives_are_not_leads(self):
        texts = ["The price of laptops is crazy lol", "I bought a desk last year and it broke", "need coffee asap"]
        self.assertEqual(len(self.pipe().run(FakeSource(texts)).leads), 0)

    def test_stages_never_read_label(self):
        p = self.pipe()
        a = p.process(Candidate(Message("1", "Looking to buy 5 laptops this week, any recommendations?", label=0)))
        b = p.process(Candidate(Message("1", "Looking to buy 5 laptops this week, any recommendations?", label=1)))
        self.assertEqual(a.is_lead, b.is_lead)
        self.assertEqual(a.intent_score, b.intent_score)

    def test_csv_and_jsonl_round_trip(self):
        rows = generate(days=1, per_day=50, lead_rate=0.1, seed=1)
        with tempfile.TemporaryDirectory() as d:
            cp, jp = Path(d, "m.csv"), Path(d, "m.jsonl")
            with open(cp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            import json
            jp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            c, j = list(CsvSource(str(cp)).messages()), list(JsonlSource(str(jp)).messages())
            self.assertEqual(len(c), 50)
            self.assertEqual([m.text for m in c], [m.text for m in j])
            self.assertIn(c[0].label, (0, 1))

    def test_fallback_flags_ambiguous_when_no_cloud(self):
        s = self.pipe().run(FakeSource(["haha the price of a laptop is crazy, where can I buy one lol"]))
        self.assertEqual(s.total, 1)

    def test_synthetic_accuracy_floor(self):
        rows = generate(days=2, per_day=1000, lead_rate=0.05, seed=3)
        with tempfile.TemporaryDirectory() as d:
            cp = Path(d, "m.csv")
            with open(cp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            s = self.pipe().run(CsvSource(str(cp)))
        self.assertGreater(s.tp / (s.tp + s.fp), 0.9)
        self.assertGreater(s.tp / (s.tp + s.fn), 0.6)

    def test_cost_tracker_prices_cloud_only(self):
        t, c = CostTracker(), Candidate(Message("1", "x"))
        self.assertEqual(t.record(c, "s3", "llama3.1:8b", 1000, 100), 0.0)
        self.assertAlmostEqual(t.record(c, "s4", "claude-sonnet-4-6", 1_000_000, 0), 3.0)


if __name__ == "__main__":
    unittest.main()
