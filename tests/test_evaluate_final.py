import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneum_det.evaluate_final import claim_one_time_test_run, validate_final_selection


class FinalEvaluationGuardTests(unittest.TestCase):
    def test_unfrozen_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "selection_status"):
            validate_final_selection({}, Path.cwd())

    def test_official_test_lock_is_one_time(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            claim_one_time_test_run(temporary_directory)
            with self.assertRaisesRegex(RuntimeError, "one-time"):
                claim_one_time_test_run(temporary_directory)

    def test_smoke_checkpoint_fails_final_training_guard(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path = root / "checkpoint.pt"
            torch.save(
                {
                    "model_name": "densenet121",
                    "epoch": 1,
                    "config": {"training": {"epochs": 1, "patience": 5}},
                },
                checkpoint_path,
            )
            validation_report = root / "validation.json"
            validation_report.write_text(json.dumps({"official_test_used": False}), encoding="utf-8")
            selection = {
                "selection_status": "frozen",
                "selected_on": "validation",
                "threshold_source": "validation",
                "temperature_source": "validation",
                "final_training_confirmed": True,
                "threshold": 0.5,
                "temperature": 1.0,
                "kind": "single",
                "models": [{"model_name": "densenet121", "checkpoint": str(checkpoint_path), "fold": 0}],
                "validation_report": str(validation_report),
            }
            with self.assertRaisesRegex(ValueError, "smoke/partial"):
                validate_final_selection(selection, root)


if __name__ == "__main__":
    unittest.main()