import unittest

import numpy as np

from dockops.variants import mutation_score


class VariantTests(unittest.TestCase):
    def test_score_is_log_ratio(self):
        lp = np.log(np.array([0.5, 0.1, 0.25, 0.15]))
        # mut_id 2 more likely than wt_id 1 -> positive score
        s = mutation_score(lp, wt_id=1, mut_id=2)
        self.assertAlmostEqual(s, np.log(0.25) - np.log(0.1), places=6)
        self.assertGreater(s, 0)

    def test_disfavored_mutation_negative(self):
        lp = np.log(np.array([0.5, 0.1, 0.25, 0.15]))
        s = mutation_score(lp, wt_id=0, mut_id=3)
        self.assertLess(s, 0)

    def test_identical_score_zero(self):
        lp = np.log(np.array([0.5, 0.1, 0.25, 0.15]))
        self.assertAlmostEqual(mutation_score(lp, 1, 1), 0.0)


if __name__ == "__main__":
    unittest.main()
