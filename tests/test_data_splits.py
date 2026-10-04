import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneum_det.data import assert_no_group_leakage, build_grouped_splits


class GroupedSplitTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for label, prefix in (("NORMAL", "normal"), ("PNEUMONIA", "person")):
            for group_number in range(6):
                group = f"{prefix}-{group_number}"
                image_count = 2 if group_number == 0 else 1
                for image_number in range(image_count):
                    rows.append(
                        {
                            "path": f"{group}-{image_number}.jpeg",
                            "label": label,
                            "split": "train",
                            "patient_id": group,
                            "subtype": "normal" if label == "NORMAL" else "bacteria",
                        }
                    )
        rows.append(
            {
                "path": "locked-test.jpeg",
                "label": "NORMAL",
                "split": "test",
                "patient_id": "normal-locked",
                "subtype": "normal",
            }
        )
        self.metadata = pd.DataFrame(rows)

    def test_folds_are_group_disjoint_and_test_is_excluded(self):
        splits = build_grouped_splits(self.metadata, n_splits=3, random_state=7)

        self.assertEqual(set(splits["fold"]), {0, 1, 2})
        self.assertNotIn("locked-test.jpeg", set(splits["path"]))
        for fold, fold_rows in splits.groupby("fold"):
            training = fold_rows[fold_rows["partition"] == "train"]
            validation = fold_rows[fold_rows["partition"] == "validation"]
            assert_no_group_leakage(training, validation, int(fold))
            self.assertEqual(set(training["label"]), {"NORMAL", "PNEUMONIA"})
            self.assertEqual(set(validation["label"]), {"NORMAL", "PNEUMONIA"})

        validation_counts = splits[splits["partition"] == "validation"]["path"].value_counts()
        self.assertTrue((validation_counts == 1).all())

    def test_mixed_label_group_is_rejected(self):
        inconsistent = self.metadata.copy()
        index = inconsistent.index[inconsistent["patient_id"] == "normal-0"][0]
        inconsistent.loc[index, "label"] = "PNEUMONIA"

        with self.assertRaisesRegex(ValueError, "one class label"):
            build_grouped_splits(inconsistent, n_splits=3)

    def test_leakage_assertion_rejects_shared_group(self):
        training = pd.DataFrame({"patient_id": ["person-1"]})
        validation = pd.DataFrame({"patient_id": ["person-1"]})

        with self.assertRaisesRegex(AssertionError, "fold 2"):
            assert_no_group_leakage(training, validation, fold=2)


if __name__ == "__main__":
    unittest.main()