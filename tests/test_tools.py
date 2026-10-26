import json
import unittest

from discovery.scenarios import generate
from discovery.tools import MAX_ITEMS, MAX_TEXT, TOOL_SPECS, ChatLog, ToolBox


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.chat, self.labels = generate(per_type=3, chatter=60, days=5, seed=2)
        self.tb = ToolBox(ChatLog(self.chat))
        self.lead = {l["scenario"]: l["lead_id"] for l in self.labels}  # one lead per scenario

    def author(self, msg_id):
        return next(m["author"] for m in self.chat if m["id"] == msg_id)

    def test_check_replied_unanswered(self):
        r = self.tb.check_replied(self.lead["hot"])
        self.assertEqual((r["reply_count"], r["replies"]), (0, []))

    def test_check_replied_returns_offer_and_noise_unjudged(self):
        offer = self.tb.check_replied(self.lead["hot_answered"])
        noise = self.tb.check_replied(self.lead["hot_peer_noise"])
        self.assertEqual(offer["reply_count"], 1)
        self.assertTrue(offer["replies"][0]["author"].startswith("seller_"))
        self.assertEqual(noise["reply_count"], 1)
        self.assertNotIn("label", json.dumps(noise))

    def test_thread_from_lead_and_from_reply_are_the_same(self):
        lead_id = self.lead["hot_answered"]
        reply_id = self.tb.check_replied(lead_id)["replies"][0]["id"]
        a, b = self.tb.get_thread(lead_id), self.tb.get_thread(reply_id)
        self.assertEqual(a, b)
        self.assertEqual([m["id"] for m in a["messages"]], [lead_id, reply_id])

    def test_sender_history_only_returns_earlier_messages(self):
        lead_id = self.lead["vague_returning"]
        h = self.tb.get_sender_history(self.author(lead_id), lead_id)
        self.assertEqual(h["total_earlier"], 2)
        self.assertTrue(all(m["id"] < lead_id for m in h["messages"]))
        self.assertEqual(self.tb.get_sender_history(self.author(self.lead["vague"]), self.lead["vague"])["total_earlier"], 0)

    def test_limit_and_truncation(self):
        long = {"id": 1, "author": "a", "timestamp": "t", "text": "x" * 1000, "reply_to": None}
        many = [{"id": i, "author": "a", "timestamp": "t", "text": "hi", "reply_to": None} for i in range(2, 60)]
        tb = ToolBox(ChatLog([long] + many))
        h = tb.get_sender_history("a", 100, limit=999)
        self.assertEqual(len(h["messages"]), MAX_ITEMS)
        self.assertEqual(len(tb.get_thread(1)["messages"][0]["text"]), MAX_TEXT)

    def test_errors_do_not_raise(self):
        self.assertIn("error", self.tb.check_replied(10 ** 9))
        self.assertIn("error", self.tb.get_thread(10 ** 9))
        self.assertIn("error", self.tb.call("nope", {}))
        self.assertIn("error", self.tb.call("check_replied", {}))
        self.assertIn("error", self.tb.call("check_replied", {"msg_id": "abc"}))

    def test_numeric_strings_and_floats_are_coerced(self):
        i = self.lead["hot_answered"]
        want = self.tb.check_replied(i)
        self.assertEqual(self.tb.call("check_replied", {"msg_id": str(i)}), want)
        self.assertEqual(self.tb.call("check_replied", {"msg_id": float(i)}), want)

    def test_unknown_arguments_are_ignored_but_reported(self):
        i = self.lead["hot_answered"]
        r = self.tb.call("check_replied", {"msg_id": i, "extra": 2})
        self.assertEqual(r["reply_count"], 1)
        self.assertEqual(r["ignored_arguments"], ["extra"])

    def test_context_supplies_missing_arguments_for_the_lead(self):
        lead = self.log_msg("vague_returning")
        ctx = {"msg_id": lead["id"], "author": lead["author"], "before_id": lead["id"]}
        h = self.tb.call("get_sender_history", {}, ctx)
        self.assertEqual(h["total_earlier"], 2)
        self.assertEqual(self.tb.call("check_replied", {}, ctx)["msg_id"], lead["id"])
        self.assertIn("error", self.tb.call("check_replied", {}))   # no context, no default

    def test_explicit_arguments_beat_context(self):
        i = self.lead["hot_answered"]
        r = self.tb.call("check_replied", {"msg_id": i}, {"msg_id": 999999})
        self.assertEqual(r["msg_id"], i)

    def log_msg(self, scenario):
        return next(m for m in self.chat if m["id"] == self.lead[scenario])

    def test_call_dispatch_matches_direct(self):
        i = self.lead["hot"]
        self.assertEqual(self.tb.call("get_thread", {"msg_id": i}), self.tb.get_thread(i))

    def test_specs_match_tools(self):
        self.assertEqual({s["name"] for s in TOOL_SPECS}, set(self.tb._tools))
        for s in TOOL_SPECS:
            json.dumps(s)
            self.assertEqual(s["input_schema"]["type"], "object")

    def test_jsonl_roundtrip(self):
        import os, tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.jsonl")
            with open(p, "w") as f:
                f.write("\n".join(json.dumps(m) for m in self.chat))
            self.assertEqual(len(ChatLog.from_jsonl(p).messages), len(self.chat))


if __name__ == "__main__":
    unittest.main()
