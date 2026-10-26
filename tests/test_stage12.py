import json
import os
import tempfile
import unittest

from discovery import sweep
from discovery.models import Candidate, Message
from discovery.stages import EmbeddingStage, PreFilterStage


class FixedEmbedder:
    """Embeds every text to a vector whose cosine with the library vector is a chosen value."""

    def __init__(self, sims):
        self.sims = sims

    def embed(self, text):
        if text == "LIB":
            return [1.0, 0.0]
        s = self.sims[text]
        return [s, (1 - s * s) ** 0.5]


def cand(text):
    return Candidate(Message(id="1", text=text))


class PreFilterTests(unittest.TestCase):
    def setUp(self):
        self.pre = PreFilterStage(12)

    def test_drops_only_obvious_noise(self):
        for t in ["gm", "lol ok", "https://example.com/a/very/long/link", "👍👍👍👍👍👍👍👍", "!!!! ???? ....", "12345 678 910"]:
            self.assertFalse(self.pre.process(cand(t)), t)

    def test_keeps_messages_without_keywords_and_flags_them(self):
        vague = cand("thinking about getting a desk soon, what do they go for?")
        self.assertTrue(self.pre.process(vague))
        self.assertFalse(vague.keyword_hit)
        clear = cand("how much does a standing desk cost?")
        self.assertTrue(self.pre.process(clear))
        self.assertTrue(clear.keyword_hit)

    def test_non_latin_text_is_not_noise(self):
        self.assertTrue(self.pre.process(cand("کسی میز اداری برای فروش داره؟")))


class TwoThresholdTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lib = os.path.join(self.tmp.name, "lib.json")
        with open(self.lib, "w") as f:
            json.dump(["LIB"], f)

    def tearDown(self):
        self.tmp.cleanup()

    def test_keyword_messages_use_the_looser_threshold(self):
        emb = FixedEmbedder({"a": 0.6})
        st = EmbeddingStage(emb, self.lib, threshold=0.55, threshold_no_keyword=0.65)
        with_kw, without = cand("a"), cand("a")
        with_kw.keyword_hit = True
        self.assertTrue(st.process(with_kw))
        self.assertFalse(st.process(without))
        self.assertAlmostEqual(without.similarity, 0.6, places=6)

    def test_single_threshold_when_no_keyword_threshold_given(self):
        st = EmbeddingStage(FixedEmbedder({"a": 0.6}), self.lib, threshold=0.55)
        self.assertTrue(st.process(cand("a")))


class SweepTests(unittest.TestCase):
    def test_table_counts(self):
        rows = [(True, 0.9, True), (True, 0.4, False), (False, 0.7, True), (False, 0.62, False), (False, 0.3, True)]
        text = sweep.table(total=4, lost=1, rows=rows, kw_grid=[0.5], nk_grid=[0.6, 0.65])
        lines = [l.split() for l in text.splitlines() if l.strip().startswith("0.5")]
        gate, t60, t65 = lines
        self.assertEqual(gate[2], "1")                          # keyword path only
        self.assertEqual((t60[2], t60[4]), ("2", "1"))          # + 0.7 lead and 0.62 noise; 0.3 lead stays out
        self.assertEqual((t65[2], t65[4]), ("2", "0"))          # 0.62 noise now excluded
        self.assertIn("dropped 1", text)

    def test_collect_with_label_ids(self):
        msgs = [Message(id="1", text="how much does a standing desk cost?"), Message(id="2", text="gm")]

        class Emb:
            def embed(self, t):
                return [1.0, 0.0]

        tmp = tempfile.TemporaryDirectory()
        lib = os.path.join(tmp.name, "l.json")
        with open(lib, "w") as f:
            json.dump(["x"], f)
        total, lost, rows = sweep.collect(msgs, {1, 2}, Emb(), library=lib)
        tmp.cleanup()
        self.assertEqual((total, lost, len(rows)), (2, 1, 1))


if __name__ == "__main__":
    unittest.main()
