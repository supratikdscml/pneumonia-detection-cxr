import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Grayscale, Resize, ToTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneum_det.ensemble import (
    _meta_train_holdout_indices,
    _predict_with_tta,
    _validate_tta_angles,
)


class EnsembleTests(unittest.TestCase):
    def test_tta_angles_must_be_small_unique_and_include_original(self):
        self.assertEqual(_validate_tta_angles([-5, 0, 5]), [-5, 0, 5])
        with self.assertRaisesRegex(ValueError, "include 0"):
            _validate_tta_angles([-5, 5])
        with self.assertRaisesRegex(ValueError, "unique"):
            _validate_tta_angles([0, 0])
        with self.assertRaisesRegex(ValueError, "within"):
            _validate_tta_angles([-15, 0, 5])

    def test_stacking_holdout_has_disjoint_patient_groups(self):
        labels = np.asarray([0] * 10 + [1] * 10)
        groups = np.asarray([f"normal-{i}" for i in range(10)] + [f"positive-{i}" for i in range(10)])

        train_indices, holdout_indices = _meta_train_holdout_indices(
            labels,
            groups,
            n_splits=2,
            seed=42,
        )

        self.assertFalse(set(groups[train_indices]) & set(groups[holdout_indices]))
        self.assertEqual(set(labels[train_indices]), {0, 1})
        self.assertEqual(set(labels[holdout_indices]), {0, 1})

    def test_tta_predictor_returns_original_and_rotated_probabilities(self):
        model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(3, 1),
        ).eval()
        transform = Compose([Grayscale(num_output_channels=3), Resize((32, 32)), ToTensor()])
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "sample.png"
            Image.new("L", (40, 40), color=128).save(image_path)
            records = [{"path": str(image_path), "label": "NORMAL", "patient_id": "sample"}]
            original, tta = _predict_with_tta(
                model,
                records,
                transform,
                device=torch.device("cpu"),
                batch_size=1,
                tta_angles=[-5, 0, 5],
            )

        self.assertEqual(original.shape, (1,))
        self.assertEqual(tta.shape, (1,))
        self.assertTrue(np.isfinite(original).all())
        self.assertTrue(np.isfinite(tta).all())
        self.assertTrue(np.all((tta >= 0.0) & (tta <= 1.0)))


if __name__ == "__main__":
    unittest.main()