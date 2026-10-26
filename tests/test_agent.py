import json
import unittest

from discovery import scenarios
from discovery.agent import SUBMIT_SPEC, SYSTEM, ReviewAgent
from discovery.llm import LLMReply, ScriptedBackend, ToolCall
from discovery.policy import ALERT, DROP, QUEUE
from discovery.tools import ChatLog, ToolBox


def tc(name, **args):
    return ToolCall(id=f"c-{name}", name=name, args=args)


def assess(answered=False, urgent=False, large=False, reason="because", **extra):
    return LLMReply(tool_calls=[tc("submit_assessment", answered=answered, urgent=urgent, large=large,
                                   reason=reason, **extra)])


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        chat, labels = scenarios.generate(per_type=2, chatter=40, days=5, seed=4)
        self.chat, self.log = chat, ChatLog(chat)
        self.toolbox = ToolBox(self.log)
        self.lead = {l["scenario"]: self.log.by_id[l["lead_id"]] for l in labels}

    def agent(self, script, **kw):
        self.backend = ScriptedBackend(script)
        return ReviewAgent(self.backend, self.toolbox, **kw)

    def test_tool_then_assessment_is_turned_into_a_decision_by_code(self):
        lead = self.lead["hot_answered"]
        r = self.agent([LLMReply(tool_calls=[tc("check_replied", msg_id=lead["id"])], tokens_in=100, tokens_out=10),
                        assess(answered=True, urgent=True, large=True, reason="a seller answered")]).review(lead)
        self.assertEqual((r.decision, r.finished, r.steps), (DROP, True, 2))
        self.assertEqual(r.facts, {"answered": True, "urgent": True, "large": True, "repeat_buyer": False})
        self.assertIn("a seller answered", r.reason)
        tool_msg = self.backend.calls[1]["messages"][-1]
        self.assertEqual(json.loads(tool_msg["content"])["reply_count"], 1)

    def test_model_never_chooses_the_decision(self):
        # even if the model tries to smuggle a decision in, only the three facts matter
        r = self.agent([assess(urgent=True, large=True, decision="drop")]).review(self.lead["hot"])
        self.assertEqual(r.decision, ALERT)

    def test_repeat_buyer_is_computed_by_code_without_the_model_asking(self):
        r = self.agent([assess()]).review(self.lead["vague_returning"])   # model never called the history tool
        self.assertTrue(r.facts["repeat_buyer"])
        self.assertEqual(r.decision, ALERT)

    def test_not_a_repeat_buyer_without_history(self):
        r = self.agent([assess()]).review(self.lead["vague"])
        self.assertEqual((r.facts["repeat_buyer"], r.decision), (False, QUEUE))

    def test_answered_beats_everything(self):
        r = self.agent([assess(answered=True, urgent=True, large=True)]).review(self.lead["vague_returning"])
        self.assertEqual(r.decision, DROP)

    def test_string_booleans_are_accepted(self):
        reply = LLMReply(tool_calls=[tc("submit_assessment", answered="false", urgent="True", large="yes")])
        r = self.agent([reply]).review(self.lead["hot"])
        self.assertEqual((r.facts["answered"], r.facts["urgent"], r.facts["large"]), (False, True, True))

    def test_text_json_fallback(self):
        text = 'Sure. {"answered": false, "urgent": true, "large": true, "reason": "needs 50 by tomorrow"}'
        r = self.agent([LLMReply(text=text)]).review(self.lead["hot"])
        self.assertEqual((r.decision, r.finished), (ALERT, True))

    def test_nudge_then_finish(self):
        r = self.agent([LLMReply(text="hmm, hard to say"), assess()]).review(self.lead["vague"])
        self.assertTrue(r.finished)
        self.assertEqual(self.backend.calls[1]["messages"][-1]["role"], "user")

    def test_step_cap_defaults_to_queue_unfinished(self):
        loop = LLMReply(tool_calls=[tc("check_replied", msg_id=self.lead["hot"]["id"])])
        r = self.agent([loop] * 3, max_steps=3).review(self.lead["hot"])
        self.assertEqual((r.decision, r.finished, r.steps), (QUEUE, False, 3))
        self.assertIn("did not finish", r.reason)

    def test_incomplete_assessment_is_rejected_then_corrected(self):
        bad = LLMReply(tool_calls=[tc("submit_assessment", answered=True)])
        r = self.agent([bad, assess(answered=True)]).review(self.lead["hot_answered"])
        self.assertEqual((r.decision, r.steps), (DROP, 2))
        self.assertIn("error", json.loads(self.backend.calls[1]["messages"][-1]["content"]))

    def test_wrong_argument_names_fall_back_to_the_lead(self):
        lead = self.lead["hot_answered"]
        self.agent([LLMReply(tool_calls=[tc("check_replied", message=5)]), assess(answered=True)]).review(lead)
        got = json.loads(self.backend.calls[1]["messages"][-1]["content"])
        self.assertEqual((got["msg_id"], got["reply_count"]), (lead["id"], 1))
        self.assertEqual(got["ignored_arguments"], ["message"])

    def test_tool_with_no_arguments_acts_on_the_lead(self):
        lead = self.lead["vague_returning"]
        self.agent([LLMReply(tool_calls=[tc("get_sender_history")]), assess()]).review(lead)
        got = json.loads(self.backend.calls[1]["messages"][-1]["content"])
        self.assertEqual((got["author"], got["total_earlier"], got["earlier_buying_messages"]), (lead["author"], 2, 2))

    def test_bad_tool_name_does_not_crash(self):
        r = self.agent([LLMReply(tool_calls=[tc("delete_everything")]), assess()]).review(self.lead["vague"])
        self.assertTrue(r.finished)
        self.assertIn("error", json.loads(self.backend.calls[1]["messages"][-1]["content"]))

    def test_several_tool_calls_in_one_turn(self):
        lead = self.lead["vague_returning"]
        calls = [tc("check_replied", msg_id=lead["id"]),
                 ToolCall("c2", "get_sender_history", {"author": lead["author"], "before_id": lead["id"]})]
        r = self.agent([LLMReply(tool_calls=calls), assess()]).review(lead)
        self.assertEqual([n for n, _ in r.tool_calls], ["check_replied", "get_sender_history"])
        self.assertEqual(len([m for m in self.backend.calls[1]["messages"] if m["role"] == "tool"]), 2)

    def test_tokens_summed_and_prompt_and_tools_sent(self):
        r = self.agent([LLMReply(tool_calls=[tc("check_replied", msg_id=1)], tokens_in=100, tokens_out=10, model="m"),
                        LLMReply(tool_calls=[tc("submit_assessment", answered=False, urgent=False, large=False)],
                                 tokens_in=150, tokens_out=20)]).review(self.lead["hot"])
        self.assertEqual((r.tokens_in, r.tokens_out, r.model), (250, 30, "m"))
        self.assertEqual(self.backend.calls[0]["system"], SYSTEM)
        self.assertIn("submit_assessment", [t["name"] for t in self.backend.calls[0]["tools"]])
        self.assertEqual(SUBMIT_SPEC["input_schema"]["required"], ["answered", "urgent", "large"])

    def test_prompt_never_contains_labels(self):
        self.agent([assess()]).review(self.lead["hot"])
        self.assertNotIn("scenario", json.dumps(self.backend.calls[0]["messages"]))

    def test_last_step_offers_only_submit_and_flags_it(self):
        loop = LLMReply(tool_calls=[tc("check_replied")])
        r = self.agent([loop, loop, assess(urgent=True, large=True)], max_steps=3).review(self.lead["hot"])
        self.assertEqual((r.finished, r.forced, r.decision), (True, True, ALERT))
        last = self.backend.calls[2]
        self.assertEqual([t["name"] for t in last["tools"]], ["submit_assessment"])
        self.assertIn("last step", last["messages"][-1]["content"])
        self.assertEqual(len(self.backend.calls[0]["tools"]), 4)   # earlier steps keep every tool

    def test_finishing_early_is_not_forced(self):
        r = self.agent([assess()], max_steps=3).review(self.lead["hot"])
        self.assertFalse(r.forced)

    def test_reason_is_listed_before_the_answers(self):
        self.assertEqual(list(SUBMIT_SPEC["input_schema"]["properties"]), ["reason", "answered", "urgent", "large"])

    def test_system_prompt_does_not_contain_the_decision_rule(self):
        for word in ("alert", "queue", "drop"):
            self.assertNotIn(word, SYSTEM.lower())


if __name__ == "__main__":
    unittest.main()
