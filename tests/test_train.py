import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneum_det.train import binary_focal_loss, compute_positive_weight, read_fold_records


class TrainingTests(unittest.TestCase):
    def test_positive_weight_uses_training_class_counts(self):
        records = [
            {"label": "NORMAL"},
            {"label": "NORMAL"},
            {"label": "PNEUMONIA"},
        ]

        weight = compute_positive_weight(records)

        self.assertEqual(weight.tolist(), [2.0])

    def test_focal_loss_is_finite_and_differentiable(self):
        logits = torch.tensor([-1.0, 1.0], requires_grad=True)
        targets = torch.tensor([0.0, 1.0])

        loss = binary_focal_loss(logits, targets, alpha=0.75, gamma=2.0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_training_reader_excludes_test_and_uses_one_partition(self):
        rows = [
            "fold,partition,path,label,patient_id,subtype,official_split",
            "0,train,a.jpeg,NORMAL,normal-a,normal,train",
            "0,validation,b.jpeg,PNEUMONIA,person-b,bacteria,train",
            "1,train,other.jpeg,NORMAL,normal-other,normal,train",
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            splits_path = Path(temporary_directory) / "splits.csv"
            splits_path.write_text("\n".join(rows), encoding="utf-8")

            training = read_fold_records(splits_path, fold=0, partition="train")
            validation = read_fold_records(splits_path, fold=0, partition="validation")
            contaminated_rows = rows + ["0,train,test.jpeg,NORMAL,normal-test,normal,test"]
            splits_path.write_text("\n".join(contaminated_rows), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "Official test"):
                read_fold_records(splits_path, fold=0, partition="train")

        self.assertEqual([record["path"] for record in training], ["a.jpeg"])
        self.assertEqual([record["path"] for record in validation], ["b.jpeg"])
        self.assertNotIn("test.jpeg", {record["path"] for record in training + validation})


if __name__ == "__main__":
    unittest.main()