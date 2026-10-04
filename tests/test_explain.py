import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneum_det.explain import border_activation_fraction, select_error_examples


class ExplainTests(unittest.TestCase):
    def test_selects_requested_count_for_each_error_category(self):
        labels = ("PNEUMONIA", "NORMAL", "PNEUMONIA", "NORMAL")
        probabilities = np.asarray([0.9, 0.1, 0.1, 0.9])
        records = [
            {"path": f"image-{index}.jpeg", "label": label, "patient_id": f"patient-{index}"}
            for index, label in enumerate(labels)
        ]

        selected = select_error_examples(
            records,
            probabilities,
            threshold=0.5,
            examples_per_category=1,
            seed=42,
        )

        self.assertEqual(set(selected), {"TP", "TN", "FP", "FN"})
        self.assertTrue(all(len(examples) == 1 for examples in selected.values()))

    def test_border_activation_fraction_detects_border_mass(self):
        cam = np.zeros((20, 20), dtype=np.float32)
        cam[:, :2] = 1.0

        fraction = border_activation_fraction(cam, fraction=0.1)

        self.assertEqual(fraction, 1.0)


if __name__ == "__main__":
    unittest.main()