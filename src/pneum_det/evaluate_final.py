"""One-time official test evaluation for a frozen validation selection."""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as transform_functional
import yaml
from scipy.special import expit, logit
from sklearn.metrics import precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from pneum_det.evaluate import (
    compute_binary_metrics,
    expected_calibration_error,
    grouped_bootstrap_intervals,
)
from pneum_det.models import create_model
from pneum_det.train import XrayDataset
from pneum_det.transforms import build_image_transforms

LOGGER = logging.getLogger(__name__)
_LABEL_TO_BINARY = {"NORMAL": 0, "PNEUMONIA": 1}


def validate_final_selection(
    selection: dict[str, Any],
    project_root: str | Path,
) -> list[dict[str, Any]]:
    """Validate a frozen validation-only model selection before touching test data."""
    if selection.get("selection_status") != "frozen":
        raise ValueError("Final test is locked until selection_status is 'frozen'")
    if selection.get("selected_on") != "validation":
        raise ValueError("Final models and calibration parameters must be selected on validation data")
    if selection.get("threshold_source") != "validation":
        raise ValueError("The decision threshold must come from validation data")
    if selection.get("temperature_source") != "validation":
        raise ValueError("Temperature scaling must be fitted on validation data")
    if selection.get("final_training_confirmed") is not True:
        raise ValueError("The final-training checkpoint must be explicitly confirmed")
    threshold = float(selection["threshold"])
    temperature = float(selection["temperature"])
    if not 0.0 <= threshold <= 1.0 or temperature <= 0.0:
        raise ValueError("Selection threshold/temperature is out of range")

    model_rows = selection.get("models")
    selection_kind = selection.get("kind")
    if not isinstance(model_rows, list) or not model_rows:
        raise ValueError("Selection manifest must contain a non-empty models list")
    if selection_kind == "ensemble" and len(model_rows) < 2:
        raise ValueError("An ensemble selection needs at least two model checkpoints")
    if selection_kind not in {"single", "ensemble"}:
        raise ValueError("Selection kind must be 'single' or 'ensemble'")
    angles = selection.get("tta_angles", [0])
    if not angles or 0 not in angles or any(abs(float(angle)) > 10 for angle in angles):
        raise ValueError("Frozen TTA angles must include 0 and remain within +/-10 degrees")

    root = Path(project_root)
    validated_models = []
    seen_models = set()
    for model_row in model_rows:
        model_name = str(model_row["model_name"])
        if model_name in seen_models:
            raise ValueError(f"Duplicate model in final selection: {model_name}")
        seen_models.add(model_name)
        checkpoint_path = Path(model_row["checkpoint"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = root / checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Frozen model checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_name") != model_name:
            raise ValueError(f"Checkpoint/model mismatch for {model_name}")
        training_config = checkpoint.get("config", {}).get("training", {})
        configured_epochs = int(training_config.get("epochs", 0))
        completed_epochs = int(checkpoint.get("epoch", 0))
        minimum_final_epochs = min(configured_epochs, int(training_config.get("patience", 5)) + 1)
        if configured_epochs < 2 or completed_epochs < minimum_final_epochs:
            raise ValueError(
                f"Checkpoint is a smoke/partial run, not a final model: {model_name} "
                f"completed {completed_epochs}/{configured_epochs} epochs; "
                f"expected at least {minimum_final_epochs}"
            )
        weight = float(model_row.get("weight", 1.0))
        if weight <= 0.0:
            raise ValueError("Ensemble weights must be positive")
        validated_models.append(
            {
                "model_name": model_name,
                "checkpoint_path": checkpoint_path,
                "checkpoint": checkpoint,
                "weight": weight,
                "fold": int(model_row["fold"]),
            }
        )

    validation_report = selection.get("validation_report")
    if not validation_report:
        raise ValueError("Selection manifest must identify its validation report")
    validation_report_path = Path(validation_report)
    if not validation_report_path.is_absolute():
        validation_report_path = root / validation_report_path
    if not validation_report_path.is_file():
        raise FileNotFoundError(f"Validation selection report not found: {validation_report_path}")
    report_data = json.loads(validation_report_path.read_text(encoding="utf-8"))
    if report_data.get("official_test_used") is not False:
        raise ValueError("Validation report indicates the official test may have been used")
    return validated_models


def claim_one_time_test_run(output_dir: str | Path) -> Path:
    """Create a permanent lock before official test records are read."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / "FINAL_TEST_STARTED.lock"
    try:
        with lock_path.open("x", encoding="utf-8") as lock_file:
            lock_file.write(datetime.now(timezone.utc).isoformat())
    except FileExistsError as exc:
        raise RuntimeError(
            "Official test evaluation is one-time and has already been started for this output directory"
        ) from exc
    return lock_path


def _load_official_test_records(audit_path: Path) -> list[dict[str, str]]:
    with audit_path.open(newline="", encoding="utf-8") as audit_file:
        records = [record for record in csv.DictReader(audit_file) if record["split"] == "test"]
    if not records:
        raise ValueError("Phase 1 audit contains no official test records")
    if any(record["label"] not in _LABEL_TO_BINARY for record in records):
        raise ValueError("Official test metadata contains unsupported class labels")
    missing_paths = [record["path"] for record in records if not Path(record["path"]).is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Official test images missing, e.g. {missing_paths[:5]}")
    return records


def _predict_selected_models(
    records: list[dict[str, str]],
    model_rows: list[dict[str, Any]],
    transform: Any,
    *,
    device: torch.device,
    batch_size: int,
    tta_angles: list[float],
) -> np.ndarray:
    loader = DataLoader(
        XrayDataset(records, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    ensemble_probabilities = []
    total_weight = sum(row["weight"] for row in model_rows)
    for model_row in model_rows:
        checkpoint = model_row["checkpoint"]
        model = create_model(
            model_row["model_name"],
            pretrained=False,
            freeze_backbone=bool(checkpoint["freeze_backbone"]),
            dropout=float(checkpoint["config"]["phase4"]["head_dropout"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        model_probabilities = []
        with torch.inference_mode():
            for images, _ in loader:
                images = images.to(device, non_blocking=True)
                tta_probabilities = []
                for angle in tta_angles:
                    variant = (
                        transform_functional.rotate(
                            images,
                            angle=float(angle),
                            interpolation=InterpolationMode.BILINEAR,
                            fill=0.0,
                        )
                        if angle
                        else images
                    )
                    logits = model(variant).squeeze(1)
                    tta_probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
                model_probabilities.extend(np.mean(tta_probabilities, axis=0).tolist())
        ensemble_probabilities.append(
            np.asarray(model_probabilities, dtype=np.float64) * model_row["weight"] / total_weight
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.sum(np.stack(ensemble_probabilities), axis=0)


def _save_final_figures(
    labels: np.ndarray,
    probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    n_bins: int,
    output_dir: Path,
) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(false_positive_rate, true_positive_rate)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set(xlabel="False-positive rate", ylabel="Sensitivity", title="Official test ROC")
    axes[1].plot(recall, precision)
    axes[1].axhline(float(labels.mean()), linestyle="--", color="gray", label="Test prevalence")
    axes[1].set(xlabel="Sensitivity / recall", ylabel="Precision", title="Official test precision-recall")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "test_roc_pr.png", dpi=160)
    plt.close(figure)

    _, before_bins = expected_calibration_error(labels, probabilities, n_bins)
    _, after_bins = expected_calibration_error(labels, calibrated_probabilities, n_bins)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    for bins, name in ((before_bins, "Before temperature"), (after_bins, "After temperature")):
        axis.plot(
            [row["confidence"] for row in bins],
            [row["accuracy"] for row in bins],
            marker="o",
            label=name,
        )
    axis.set(xlabel="Mean predicted probability", ylabel="Observed pneumonia rate", title="Official test reliability")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "test_calibration.png", dpi=160)
    plt.close(figure)


def run_final_test(
    *,
    selection_path: str | Path,
    config_path: str | Path,
    splits_path: str | Path,
    audit_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the frozen model/ensemble once on the official test data."""
    selection_file = Path(selection_path)
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    project_root = Path(config_path).resolve().parent
    model_rows = validate_final_selection(selection, project_root)
    destination = Path(output_dir)
    if (destination / "test_metrics.json").exists() or (destination / "test_predictions.csv").exists():
        raise RuntimeError("Final test output already exists; refusing a second evaluation")
    claim_one_time_test_run(destination)

    with Path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    audit_records = _load_official_test_records(Path(audit_path))
    test_records = [
        {key: record[key] for key in ("path", "label", "patient_id", "subtype")}
        for record in audit_records
    ]
    _, eval_transform = build_image_transforms(config["preprocessing"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(config["training"]["batch_sizes"].get(selection["models"][0]["model_name"], config["training"]["batch_size"]))
    tta_angles = [float(angle) for angle in selection.get("tta_angles", [0])]
    probabilities = _predict_selected_models(
        test_records,
        model_rows,
        eval_transform,
        device=device,
        batch_size=batch_size,
        tta_angles=tta_angles,
    )
    labels = np.asarray([_LABEL_TO_BINARY[record["label"]] for record in test_records], dtype=np.int64)
    threshold = float(selection["threshold"])
    temperature = float(selection["temperature"])
    clipped_probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    ensemble_logits = logit(clipped_probabilities)
    calibrated_probabilities = expit(ensemble_logits / temperature)
    metrics = compute_binary_metrics(labels, probabilities, threshold)
    bootstrap_intervals = grouped_bootstrap_intervals(
        test_records,
        labels,
        probabilities,
        threshold,
        n_resamples=int(config["evaluation"]["bootstrap_resamples"]),
        seed=int(config["seed"]),
    )
    ece_before, _ = expected_calibration_error(
        labels,
        probabilities,
        int(config["evaluation"]["calibration_bins"]),
    )
    ece_after, _ = expected_calibration_error(
        labels,
        calibrated_probabilities,
        int(config["evaluation"]["calibration_bins"]),
    )
    nll_before = float(np.mean(np.logaddexp(0.0, ensemble_logits) - labels * ensemble_logits))
    calibrated_logits = ensemble_logits / temperature
    nll_after = float(np.mean(np.logaddexp(0.0, calibrated_logits) - labels * calibrated_logits))

    with Path(audit_path).open(newline="", encoding="utf-8") as audit_file:
        all_audit_records = list(csv.DictReader(audit_file))
    development_records = [record for record in all_audit_records if record["split"] in {"train", "val"}]
    development_prevalence = sum(record["label"] == "PNEUMONIA" for record in development_records) / len(development_records)
    test_prevalence = float(labels.mean())
    report = {
        "selection_manifest": str(selection_file),
        "models": [
            {"model_name": row["model_name"], "checkpoint": str(row["checkpoint_path"]), "weight": row["weight"]}
            for row in model_rows
        ],
        "selected_on": "validation",
        "official_test_used": True,
        "test_images": len(test_records),
        "test_patient_groups": len({record["patient_id"] for record in test_records}),
        "threshold": threshold,
        "temperature": temperature,
        "tta_angles": tta_angles,
        "metrics_at_frozen_threshold": metrics,
        "patient_grouped_bootstrap_95_percent_ci": bootstrap_intervals,
        "calibration": {
            "nll_before": nll_before,
            "nll_after": nll_after,
            "ece_before": ece_before,
            "ece_after": ece_after,
        },
        "distribution_shift": {
            "development_pneumonia_prevalence": development_prevalence,
            "official_test_pneumonia_prevalence": test_prevalence,
            "test_minus_development_prevalence": test_prevalence - development_prevalence,
        },
        "device": str(device),
    }
    predictions_path = destination / "test_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as prediction_file:
        fields = ("path", "patient_id", "label", "probability", "calibrated_probability")
        writer = csv.DictWriter(prediction_file, fieldnames=fields)
        writer.writeheader()
        for record, probability, calibrated_probability in zip(
            test_records,
            probabilities,
            calibrated_probabilities,
        ):
            writer.writerow(
                {
                    "path": record["path"],
                    "patient_id": record["patient_id"],
                    "label": record["label"],
                    "probability": float(probability),
                    "calibrated_probability": float(calibrated_probability),
                }
            )
    report["predictions_csv"] = str(predictions_path)
    (destination / "test_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _save_final_figures(
        labels,
        probabilities,
        calibrated_probabilities,
        int(config["evaluation"]["calibration_bins"]),
        destination,
    )
    LOGGER.info(
        "One-time official test evaluation complete: n=%d AUROC=%.4f AUPRC=%.4f sensitivity=%.4f specificity=%.4f",
        len(test_records),
        metrics["auroc"],
        metrics["auprc"],
        metrics["sensitivity"],
        metrics["specificity"],
    )
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=project_root / "outputs" / "phase9" / "final_selection.json",
    )
    parser.add_argument("--config", type=Path, default=project_root / "config.yaml")
    parser.add_argument("--splits", type=Path, default=project_root / "outputs" / "splits.csv")
    parser.add_argument("--audit", type=Path, default=project_root / "outputs" / "phase1" / "image_audit.csv")
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs" / "phase9")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_final_test(
        selection_path=arguments.selection,
        config_path=arguments.config,
        splits_path=arguments.splits,
        audit_path=arguments.audit,
        output_dir=arguments.output_dir,
    )


if __name__ == "__main__":
    main()