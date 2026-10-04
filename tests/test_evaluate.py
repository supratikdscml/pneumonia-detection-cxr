import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneum_det.evaluate import (
    compute_binary_metrics,
    expected_calibration_error,
    fit_temperature,
    grouped_bootstrap_intervals,
    select_sensitivity_threshold,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        self.probabilities = np.asarray([0.1, 0.4, 0.65, 0.9], dtype=np.float64)

    def test_sensitivity_threshold_chooses_highest_feasible(self):
        threshold = select_sensitivity_threshold(self.labels, self.probabilities, 0.95)
        metrics = compute_binary_metrics(self.labels, self.probabilities, threshold)

        self.assertAlmostEqual(threshold, 0.65)
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)

    def test_calibration_metrics_are_finite(self):
        ece, bins = expected_calibration_error(self.labels, self.probabilities, n_bins=4)
        temperature = fit_temperature(self.labels, np.log(self.probabilities / (1 - self.probabilities)))

        self.assertTrue(np.isfinite(ece))
        self.assertEqual(len(bins), 4)
        self.assertTrue(np.isfinite(temperature))
        self.assertGreater(temperature, 0)

    def test_patient_group_bootstrap_returns_expected_intervals(self):
        records = [
            {"patient_id": f"patient-{index}", "label": "NORMAL" if label == 0 else "PNEUMONIA"}
            for index, label in enumerate(self.labels)
        ]
        intervals = grouped_bootstrap_intervals(
            records,
            self.labels,
            self.probabilities,
            threshold=0.5,
            n_resamples=40,
            seed=42,
        )

        self.assertIn("auroc", intervals)
        self.assertLessEqual(intervals["auroc"]["lower"], intervals["auroc"]["upper"])
        self.assertGreaterEqual(intervals["auroc"]["valid_resamples"], 20)


if __name__ == "__main__":
    unittest.main()