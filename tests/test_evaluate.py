import unittest

from discovery import evaluate, scenarios
from discovery.agent import ReviewAgent
from discovery.cost import CostTracker
from discovery.llm import LLMBackend
from discovery.policy_llm import PolicyBackend
from discovery.tools import ChatLog, ToolBox


class Boom:
    name = "boom"

    def chat(self, system, messages, tools):
        raise TimeoutError("server timed out")


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        chat, self.labels = scenarios.generate(per_type=4, chatter=80, days=5, seed=6)
        self.log = ChatLog(chat)

    def run_with(self, backend, **kw):
        agent = ReviewAgent(backend, ToolBox(self.log), tracker=CostTracker())
        return evaluate.run_eval(agent, self.log, self.labels, **kw)

    def test_policy_backend_scores_perfectly(self):
        rows = self.run_with(PolicyBackend())
        self.assertEqual(len(rows), 4 * len(scenarios.SCENARIOS))
        self.assertEqual(evaluate._acc(rows), 1.0)
        self.assertTrue(all(r["finished"] for r in rows))

    def test_text_only_baseline_is_clearly_worse(self):
        rows = self.run_with(PolicyBackend())
        self.assertLess(evaluate._acc(rows, "baseline"), 0.8)

    def test_backend_errors_are_recorded_not_raised(self):
        rows = self.run_with(Boom(), limit=3)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(not r["finished"] and r["decision"] == "queue" for r in rows))
        self.assertIn("timed out", rows[0]["reason"])

    def test_limit(self):
        self.assertEqual(len(self.run_with(PolicyBackend(), limit=5)), 5)

    def test_render_contains_the_key_sections(self):
        text = evaluate.render(self.run_with(PolicyBackend()))
        for needle in ("agent accuracy", "baseline", "CONFUSION", "BY SCENARIO", "COST", "MISTAKES (0)"):
            self.assertIn(needle, text)

    def test_confusion_counts_a_wrong_answer(self):
        rows = self.run_with(PolicyBackend())
        rows[0] = dict(rows[0], decision="drop" if rows[0]["expected"] != "drop" else "alert")
        text = evaluate.render(rows)
        self.assertIn("MISTAKES (1)", text)

    def test_policy_backend_satisfies_the_interface(self):
        self.assertTrue(hasattr(PolicyBackend(), "chat") and PolicyBackend().name == "policy")


if __name__ == "__main__":
    unittest.main()
