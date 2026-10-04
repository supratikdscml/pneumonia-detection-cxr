import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneum_det.models import MODEL_NAMES, create_model


class ModelFactoryTests(unittest.TestCase):
    def test_all_models_return_one_logit(self):
        for model_name in MODEL_NAMES:
            with self.subTest(model_name=model_name):
                model = create_model(model_name, pretrained=False)
                model.eval()
                with torch.inference_mode():
                    logits = model(torch.zeros(1, 3, 224, 224))
                self.assertEqual(tuple(logits.shape), (1, 1))
                del model, logits

    def test_frozen_backbone_keeps_only_classifier_trainable(self):
        model = create_model("efficientnet_b0", freeze_backbone=True)

        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertTrue(trainable_names)
        self.assertTrue(all(name.startswith("classifier.") for name in trainable_names))

    def test_custom_cnn_rejects_backbone_options(self):
        with self.assertRaisesRegex(ValueError, "does not have pretrained"):
            create_model("custom_cnn", pretrained=True)
        with self.assertRaisesRegex(ValueError, "no pretrained backbone"):
            create_model("custom_cnn", freeze_backbone=True)


if __name__ == "__main__":
    unittest.main()