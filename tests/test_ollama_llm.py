import unittest
from unittest import mock

from discovery import ollama_llm
from discovery.agent import SUBMIT_SPEC, ReviewAgent
from discovery.cost import CostTracker
from discovery.llm import ToolCall
from discovery.tools import TOOL_SPECS, ChatLog, ToolBox


class OllamaConversionTests(unittest.TestCase):
    def test_tools_converted(self):
        t = ollama_llm.to_ollama_tools(TOOL_SPECS + [SUBMIT_SPEC])
        self.assertEqual(t[0]["type"], "function")
        self.assertEqual(t[0]["function"]["parameters"], TOOL_SPECS[0]["input_schema"])
        self.assertEqual(len(t), 4)

    def test_messages_converted(self):
        msgs = [{"role": "user", "content": "lead"},
                {"role": "assistant", "content": "", "tool_calls": [ToolCall("1", "check_replied", {"msg_id": 5})]},
                {"role": "tool", "tool_call_id": "1", "name": "check_replied", "content": "{}"}]
        out = ollama_llm.to_ollama_messages("SYS", msgs)
        self.assertEqual([m["role"] for m in out], ["system", "user", "assistant", "tool"])
        self.assertEqual(out[2]["tool_calls"][0]["function"], {"name": "check_replied", "arguments": {"msg_id": 5}})
        self.assertEqual(out[3]["tool_name"], "check_replied")

    def test_response_with_tool_call_and_tokens(self):
        resp = {"message": {"content": "", "tool_calls": [{"function": {"name": "check_replied", "arguments": {"msg_id": 7}}}]},
                "prompt_eval_count": 120, "eval_count": 15}
        r = ollama_llm.from_ollama_response(resp, "m")
        self.assertEqual((r.tokens_in, r.tokens_out, r.model), (120, 15, "m"))
        self.assertEqual((r.tool_calls[0].name, r.tool_calls[0].args), ("check_replied", {"msg_id": 7}))

    def test_response_plain_text_and_odd_arguments(self):
        r = ollama_llm.from_ollama_response({"message": {"content": "hello"}}, "m")
        self.assertEqual((r.text, r.tool_calls), ("hello", []))
        weird = {"message": {"tool_calls": [{"function": {"name": "x", "arguments": "not a dict"}}]}}
        self.assertEqual(ollama_llm.from_ollama_response(weird, "m").tool_calls[0].args, {})

    def test_tool_call_written_as_text_is_recovered(self):
        names = {"get_sender_history", "check_replied"}
        text = 'I will look at the history.\n\n{"name": "get_sender_history", "parameters": {"author": "b", "before_id": "7"}}'
        r = ollama_llm.from_ollama_response({"message": {"content": text}}, "m", names)
        self.assertEqual((r.tool_calls[0].name, r.tool_calls[0].args), ("get_sender_history", {"author": "b", "before_id": "7"}))
        alt = '{"name": "check_replied", "arguments": "{\\"msg_id\\": 3}"}'
        self.assertEqual(ollama_llm.from_ollama_response({"message": {"content": alt}}, "m", names).tool_calls[0].args, {"msg_id": 3})

    def test_text_that_is_not_a_known_tool_call_is_left_alone(self):
        names = {"check_replied"}
        for text in ('{"decision": "alert", "reason": "x"}', '{"name": "other_tool"}', "plain words", '{"name": "check_re'):
            self.assertEqual(ollama_llm.from_ollama_response({"message": {"content": text}}, "m", names).tool_calls, [])

    def test_real_tool_calls_win_over_text(self):
        resp = {"message": {"content": '{"name": "check_replied", "parameters": {}}',
                            "tool_calls": [{"function": {"name": "get_thread", "arguments": {}}}]}}
        calls = ollama_llm.from_ollama_response(resp, "m", {"check_replied", "get_thread"}).tool_calls
        self.assertEqual([c.name for c in calls], ["get_thread"])

    def test_full_review_through_fake_server_and_cost_tracking(self):
        chat = [{"id": 1, "author": "b", "timestamp": "t", "text": "need 50 desks asap", "reply_to": None}]
        replies = [
            {"message": {"tool_calls": [{"function": {"name": "check_replied", "arguments": {"msg_id": 1}}}]},
             "prompt_eval_count": 200, "eval_count": 20},
            {"message": {"tool_calls": [{"function": {"name": "submit_assessment",
                                                      "arguments": {"answered": False, "urgent": True, "large": True,
                                                                    "reason": "unanswered"}}}]},
             "prompt_eval_count": 260, "eval_count": 25},
        ]
        tracker = CostTracker()
        with mock.patch.object(ollama_llm, "_post", side_effect=replies) as post:
            agent = ReviewAgent(ollama_llm.OllamaChat(model="qwen2.5:7b"), ToolBox(ChatLog(chat)), tracker=tracker)
            r = agent.review(chat[0])
        self.assertEqual((r.decision, r.steps, r.finished), ("alert", 2, True))
        self.assertEqual((r.tokens_in, r.tokens_out, r.cost_usd), (460, 45, 0.0))
        self.assertGreaterEqual(r.seconds, 0.0)
        self.assertEqual(tracker.by_stage["5_reviewer"]["calls"], 1)
        self.assertEqual(post.call_args_list[0][0][0], "http://localhost:11434/api/chat")


class CostTests(unittest.TestCase):
    def test_cloud_model_cost_is_recorded_on_the_result(self):
        from discovery.agent import ReviewResult
        t, r = CostTracker(), ReviewResult(lead_id=1)
        t.record(r, "5_reviewer", "claude-sonnet-4-6", 1_000_000, 0)
        self.assertAlmostEqual(r.cost_usd, 3.0)


if __name__ == "__main__":
    unittest.main()
