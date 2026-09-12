import unittest

import numpy as np

from tfm_accidentes.model_selection import (
    ExpandingYearSplit,
    optimize_blend_weight,
    select_threshold_for_recall,
)


class TemporalModelSelectionTests(unittest.TestCase):
    def test_expanding_split_never_uses_future_observations(self):
        years = np.array([2019, 2020, 2021, 2022, 2023])
        splitter = ExpandingYearSplit([2022, 2023])

        folds = list(splitter.split(np.zeros((5, 1)), groups=years))

        self.assertEqual(years[folds[0][0]].tolist(), [2019, 2020, 2021])
        self.assertEqual(years[folds[0][1]].tolist(), [2022])
        self.assertTrue((years[folds[1][0]] < years[folds[1][1]][0]).all())

    def test_threshold_reaches_requested_recall(self):
        y_true = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.4, 0.8])

        threshold = select_threshold_for_recall(y_true, probabilities, 0.5)

        recall = ((probabilities >= threshold) & (y_true == 1)).sum() / y_true.sum()
        self.assertGreaterEqual(recall, 0.5)

    def test_blend_can_select_first_model(self):
        y_true = np.array([0, 0, 1, 1])
        first = np.array([0.1, 0.2, 0.8, 0.9])
        second = np.array([0.9, 0.8, 0.2, 0.1])

        weight, _ = optimize_blend_weight(y_true, first, second, [0.0, 0.5, 1.0])

        self.assertEqual(weight, 1.0)


if __name__ == "__main__":
    unittest.main()
