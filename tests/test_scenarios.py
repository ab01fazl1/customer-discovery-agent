import unittest

from discovery.scenarios import ALERT, DROP, QUEUE, SCENARIOS, expected_decision, generate


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.chat, self.labels = generate(per_type=5, chatter=100, days=5, seed=1)
        self.by_id = {m["id"]: m for m in self.chat}

    def test_ids_unique_and_chronological(self):
        ids = [m["id"] for m in self.chat]
        self.assertEqual(ids, sorted(set(ids)))
        stamps = [m["timestamp"] for m in self.chat]
        self.assertEqual(stamps, sorted(stamps))

    def test_replies_point_to_earlier_messages(self):
        for m in self.chat:
            if m["reply_to"] is not None:
                self.assertLess(m["reply_to"], m["id"])

    def test_every_label_points_to_a_lead_message(self):
        self.assertEqual(len(self.labels), 5 * len(SCENARIOS))
        for l in self.labels:
            self.assertIn(l["lead_id"], self.by_id)
            self.assertIn(l["decision"], (ALERT, QUEUE, DROP))

    def test_all_three_decisions_occur(self):
        self.assertEqual({l["decision"] for l in self.labels}, {ALERT, QUEUE, DROP})

    def test_returning_senders_have_earlier_messages(self):
        for l in self.labels:
            if l["scenario"] == "vague_returning":
                lead = self.by_id[l["lead_id"]]
                earlier = [m for m in self.chat if m["author"] == lead["author"] and m["id"] < lead["id"]]
                self.assertEqual(len(earlier), 2)

    def test_answered_leads_have_a_reply(self):
        for l in self.labels:
            replies = [m for m in self.chat if m["reply_to"] == l["lead_id"]]
            if l["scenario"] in ("hot_answered", "hot_peer_noise", "vague_resolved"):
                self.assertEqual(len(replies), 1)
            else:
                self.assertEqual(replies, [])

    def test_decision_rule(self):
        got = {s.name: expected_decision(s)[0] for s in SCENARIOS}
        self.assertEqual(got, {
            "hot": ALERT, "hot_answered": DROP, "hot_peer_noise": ALERT, "medium_large": QUEUE,
            "vague": QUEUE, "vague_returning": ALERT, "vague_resolved": DROP})

    def test_deterministic(self):
        self.assertEqual(generate(5, 100, 5, 1), (self.chat, self.labels))


if __name__ == "__main__":
    unittest.main()
