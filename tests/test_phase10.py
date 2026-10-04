import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneum_det.models import create_lung_unet, create_model
from pneum_det.phase10 import (
    apply_lung_crop,
    audit_pipeline_credentials,
    crop_to_mask,
    export_onnx,
    export_torchscript,
    heuristic_lung_mask,
    three_class_fold_counts,
    three_class_index,
)


class Phase10Tests(unittest.TestCase):
    def test_three_class_indices(self):
        self.assertEqual(three_class_index("normal"), 0)
        self.assertEqual(three_class_index("bacteria"), 1)
        self.assertEqual(three_class_index("virus"), 2)

    def test_three_class_counts_exclude_test_and_keep_groups(self):
        splits = pd.DataFrame(
            {
                "fold": [0, 0, 0],
                "partition": ["train", "validation", "train"],
                "path": ["a.jpeg", "b.jpeg", "c.jpeg"],
                "label": ["NORMAL", "PNEUMONIA", "PNEUMONIA"],
                "patient_id": ["n1", "p1", "p2"],
                "subtype": ["normal", "bacteria", "virus"],
                "official_split": ["train", "val", "train"],
            }
        )
        counts = three_class_fold_counts(splits)
        self.assertEqual(set(counts["three_class"]), {"normal", "bacteria", "virus"})
        self.assertEqual(int(counts["n_images"].sum()), 3)

    def test_heuristic_mask_and_crop(self):
        image = np.full((64, 80), 200, dtype=np.uint8)
        image[10:50, 8:30] = 20
        image[10:50, 50:72] = 20
        mask = heuristic_lung_mask(image)
        cropped = crop_to_mask(image, mask)
        self.assertEqual(mask.shape, image.shape)
        self.assertGreater(int(mask.sum()), 0)
        self.assertLess(cropped.size, image.size)

    def test_three_class_and_unet_shapes(self):
        classifier = create_model("custom_cnn", num_classes=3)
        unet = create_lung_unet(base_channels=8)
        classifier.eval()
        unet.eval()
        with torch.inference_mode():
            class_logits = classifier(torch.zeros(1, 3, 224, 224))
            mask_logits = unet(torch.zeros(1, 1, 96, 96))
        self.assertEqual(tuple(class_logits.shape), (1, 3))
        self.assertEqual(tuple(mask_logits.shape[:2]), (1, 1))

    def test_export_roundtrip_files(self):
        model = create_model("custom_cnn", num_classes=1)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            script_path = export_torchscript(model, destination / "model.pt", image_size=32)
            onnx_path = export_onnx(model, destination / "model.onnx", image_size=32)
            self.assertTrue(script_path.is_file())
            loaded = torch.jit.load(str(script_path))
            with torch.inference_mode():
                output = loaded(torch.zeros(1, 3, 32, 32))
            self.assertEqual(tuple(output.shape), (1, 1))
            if onnx_path is not None:
                self.assertTrue(onnx_path.is_file())

    def test_credential_audit_returns_rows(self):
        root = Path(__file__).resolve().parents[1]
        table = audit_pipeline_credentials(root)
        self.assertGreater(len(table), 10)
        self.assertEqual(set(table.columns), {"item", "status", "evidence"})

    def test_apply_lung_crop_writes_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.jpeg"
            image = np.full((48, 48), 180, dtype=np.uint8)
            image[8:40, 6:20] = 15
            image[8:40, 28:42] = 15
            Image.fromarray(image).save(path)
            result = apply_lung_crop(path)
            self.assertEqual(result["mask_source"], "heuristic_otsu")
            self.assertEqual(result["original"].shape, (48, 48))


if __name__ == "__main__":
    unittest.main()
