import itertools
import unittest

from discovery.policy import ALERT, DROP, QUEUE, decide


class PolicyTests(unittest.TestCase):
    def test_truth_table(self):
        for answered, urgent, large, repeat in itertools.product([False, True], repeat=4):
            got, _ = decide(answered, urgent, large, repeat)
            if answered:
                want = DROP
            elif (urgent and large) or repeat:
                want = ALERT
            else:
                want = QUEUE
            self.assertEqual(got, want, (answered, urgent, large, repeat))

    def test_reasons_are_never_empty(self):
        for combo in itertools.product([False, True], repeat=4):
            self.assertTrue(decide(*combo)[1])


if __name__ == "__main__":
    unittest.main()
