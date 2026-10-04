import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneum_det.baselines import (
    _extract_hog_features,
    _set_training_mode,
    compute_binary_metrics,
)
from pneum_det.models import create_model


class BaselineTests(unittest.TestCase):
    def test_metric_values_for_known_predictions(self):
        metrics = compute_binary_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.1, 0.8, 0.7, 0.9]),
        )

        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 0)
        self.assertEqual(metrics["true_positive"], 2)
        self.assertEqual(metrics["recall_sensitivity"], 1.0)
        self.assertEqual(metrics["specificity"], 0.5)

    def test_hog_extractor_returns_finite_features(self):
        import cv2

        image = np.zeros((128, 128), dtype=np.uint8)
        cv2.line(image, (16, 16), (112, 112), color=255, thickness=3)
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "synthetic.png"
            self.assertTrue(cv2.imwrite(str(image_path), image))
            features = _extract_hog_features(
                [{"path": str(image_path), "label": "NORMAL"}],
                image_size=128,
                block_size=(16, 16),
                block_stride=(8, 8),
                cell_size=(8, 8),
                bins=9,
            )

        self.assertEqual(features.shape, (1, 8100))
        self.assertTrue(np.isfinite(features).all())

    def test_frozen_backbone_batch_norm_stays_in_eval_mode(self):
        model = create_model("resnet50", freeze_backbone=True)
        _set_training_mode(model, "resnet50", freeze_backbone=True)
        backbone_batch_norm = model.bn1
        running_mean = backbone_batch_norm.running_mean.clone()

        with torch.no_grad():
            model(torch.zeros(2, 3, 224, 224))

        self.assertFalse(backbone_batch_norm.training)
        self.assertTrue(model.fc.training)
        self.assertTrue(torch.equal(running_mean, backbone_batch_norm.running_mean))


if __name__ == "__main__":
    unittest.main()