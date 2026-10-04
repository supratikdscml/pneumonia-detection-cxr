"""Validation-only probability ensembles, non-flip TTA, and stacking."""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torchvision.transforms.functional as transform_functional
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from pneum_det.evaluate import compute_binary_metrics, select_sensitivity_threshold
from pneum_det.models import MODEL_NAMES, create_model
from pneum_det.train import XrayDataset, read_fold_records
from pneum_det.transforms import build_image_transforms

LOGGER = logging.getLogger(__name__)
_LABEL_TO_BINARY = {"NORMAL": 0, "PNEUMONIA": 1}
_METRIC_FIELDS = (
    "model_or_ensemble",
    "fold",
    "evaluation_subset",
    "threshold",
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "auroc",
    "auprc",
)


def _meta_train_holdout_indices(
    labels: np.ndarray,
    patient_groups: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create disjoint, class-stratified groups for stacker fitting/evaluation."""
    if n_splits < 2:
        raise ValueError("stacking holdout needs at least two splits")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    meta_train, meta_holdout = next(
        splitter.split(np.zeros(len(labels)), labels, groups=patient_groups)
    )
    if set(patient_groups[meta_train]) & set(patient_groups[meta_holdout]):
        raise AssertionError("Patient groups overlap in the stacking holdout")
    if np.unique(labels[meta_train]).size != 2 or np.unique(labels[meta_holdout]).size != 2:
        raise ValueError("Both classes must be present in stacking train and holdout subsets")
    return meta_train, meta_holdout


def _validate_tta_angles(angles: list[int]) -> list[int]:
    if not angles or 0 not in angles:
        raise ValueError("TTA angles must include 0 degrees for the unmodified view")
    if any(abs(angle) > 10 for angle in angles):
        raise ValueError("TTA rotations must stay within +/-10 degrees")
    if len(set(angles)) != len(angles):
        raise ValueError("TTA angles must be unique")
    return angles


def _predict_with_tta(
    model: torch.nn.Module,
    records: list[dict[str, str]],
    transform: Any,
    *,
    device: torch.device,
    batch_size: int,
    tta_angles: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return original-view and averaged small-rotation probabilities."""
    loader = DataLoader(
        XrayDataset(records, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    original_probabilities = []
    tta_probabilities = []
    model.eval()
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            angle_probabilities = []
            for angle in tta_angles:
                variant = (
                    transform_functional.rotate(
                        images,
                        angle=angle,
                        interpolation=InterpolationMode.BILINEAR,
                        fill=0.0,
                    )
                    if angle
                    else images
                )
                logits = model(variant).squeeze(1)
                angle_probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
            original_probabilities.extend(angle_probabilities[tta_angles.index(0)].tolist())
            tta_probabilities.extend(np.mean(angle_probabilities, axis=0).tolist())
    return (
        np.asarray(original_probabilities, dtype=np.float32),
        np.asarray(tta_probabilities, dtype=np.float32),
    )


def _write_metric_rows(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_validation_ensemble(
    *,
    model_names: list[str],
    checkpoint_paths: list[str | Path],
    fold: int,
    splits_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compare models, probability averages, TTA, and held-out stacking."""
    if len(model_names) < 3:
        raise ValueError("Phase 8 requires at least three diverse model checkpoints")
    if len(model_names) != len(checkpoint_paths):
        raise ValueError("Each ensemble model must have exactly one checkpoint path")
    if len(set(model_names)) != len(model_names):
        raise ValueError("Ensemble model names must be unique")
    if any(name not in MODEL_NAMES for name in model_names):
        raise ValueError(f"Models must be selected from {MODEL_NAMES}")

    validation_records = read_fold_records(splits_path, fold, "validation")
    training_records = read_fold_records(splits_path, fold, "train")
    train_groups = {record["patient_id"] for record in training_records}
    validation_groups = {record["patient_id"] for record in validation_records}
    if train_groups & validation_groups:
        raise AssertionError("Patient groups overlap between outer training and validation folds")
    if any(record["official_split"] == "test" for record in validation_records):
        raise AssertionError("Official test data must not enter Phase 8")

    validation_labels = np.asarray(
        [_LABEL_TO_BINARY[record["label"]] for record in validation_records],
        dtype=np.int64,
    )
    patient_groups = np.asarray([record["patient_id"] for record in validation_records])
    if np.unique(validation_labels).size != 2:
        raise ValueError("Validation fold must contain both classes")
    _, eval_transform = build_image_transforms(config["preprocessing"])
    batch_size_config = config["training"].get("batch_sizes", {})
    default_batch_size = int(config["training"]["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tta_angles = _validate_tta_angles(
        [int(angle) for angle in config["phase8"]["tta_angles"]]
    )

    original_probabilities: dict[str, np.ndarray] = {}
    tta_probabilities: dict[str, np.ndarray] = {}
    for model_name, checkpoint_path in zip(model_names, checkpoint_paths):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["model_name"] != model_name or int(checkpoint["fold"]) != fold:
            raise ValueError(f"Checkpoint does not match model={model_name}, fold={fold}: {checkpoint_path}")
        model = create_model(
            model_name,
            pretrained=False,
            freeze_backbone=bool(checkpoint["freeze_backbone"]),
            dropout=float(config["phase4"]["head_dropout"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        batch_size = int(batch_size_config.get(model_name, default_batch_size))
        original_probabilities[model_name], tta_probabilities[model_name] = _predict_with_tta(
            model,
            validation_records,
            eval_transform,
            device=device,
            batch_size=batch_size,
            tta_angles=tta_angles,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        LOGGER.info("Generated original/TTA validation predictions for %s", model_name)

    mean_probability = np.mean(np.stack(list(original_probabilities.values())), axis=0)
    mean_tta_probability = np.mean(np.stack(list(tta_probabilities.values())), axis=0)
    target_sensitivity = float(config["evaluation"]["target_sensitivity"])
    metric_rows = []
    single_model_metrics = {}
    for model_name in model_names:
        for method, probabilities in (
            ("original", original_probabilities[model_name]),
            ("tta", tta_probabilities[model_name]),
        ):
            threshold = select_sensitivity_threshold(labels=validation_labels, probabilities=probabilities, target_sensitivity=target_sensitivity)
            metrics = compute_binary_metrics(validation_labels, probabilities, threshold)
            row = {
                "model_or_ensemble": model_name,
                "fold": fold,
                "evaluation_subset": f"validation_{method}",
                **metrics,
            }
            metric_rows.append(row)
            if method == "tta":
                single_model_metrics[model_name] = metrics

    ensemble_threshold = select_sensitivity_threshold(
        validation_labels,
        mean_probability,
        target_sensitivity,
    )
    ensemble_tta_threshold = select_sensitivity_threshold(
        validation_labels,
        mean_tta_probability,
        target_sensitivity,
    )
    ensemble_metrics = compute_binary_metrics(validation_labels, mean_probability, ensemble_threshold)
    ensemble_tta_metrics = compute_binary_metrics(validation_labels, mean_tta_probability, ensemble_tta_threshold)
    metric_rows.extend(
        [
            {
                "model_or_ensemble": "probability_mean",
                "fold": fold,
                "evaluation_subset": "full_validation",
                **ensemble_metrics,
            },
            {
                "model_or_ensemble": "probability_mean_tta",
                "fold": fold,
                "evaluation_subset": "full_validation",
                **ensemble_tta_metrics,
            },
        ]
    )

    meta_train_indices, meta_holdout_indices = _meta_train_holdout_indices(
        validation_labels,
        patient_groups,
        n_splits=int(config["phase8"]["stacking_holdout_splits"]),
        seed=int(config["seed"]),
    )
    meta_features = np.column_stack([tta_probabilities[name] for name in model_names])
    stacker = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=int(config["seed"]),
    )
    stacker.fit(meta_features[meta_train_indices], validation_labels[meta_train_indices])
    stack_train_probabilities = stacker.predict_proba(meta_features[meta_train_indices])[:, 1]
    stack_holdout_probabilities = stacker.predict_proba(meta_features[meta_holdout_indices])[:, 1]
    stack_threshold = select_sensitivity_threshold(
        validation_labels[meta_train_indices],
        stack_train_probabilities,
        target_sensitivity,
    )
    stack_metrics = compute_binary_metrics(
        validation_labels[meta_holdout_indices],
        stack_holdout_probabilities,
        stack_threshold,
    )
    ensemble_meta_train = mean_tta_probability[meta_train_indices]
    ensemble_meta_holdout = mean_tta_probability[meta_holdout_indices]
    ensemble_meta_threshold = select_sensitivity_threshold(
        validation_labels[meta_train_indices],
        ensemble_meta_train,
        target_sensitivity,
    )
    ensemble_meta_metrics = compute_binary_metrics(
        validation_labels[meta_holdout_indices],
        ensemble_meta_holdout,
        ensemble_meta_threshold,
    )
    metric_rows.extend(
        [
            {
                "model_or_ensemble": "probability_mean_tta",
                "fold": fold,
                "evaluation_subset": "stacking_group_holdout",
                **ensemble_meta_metrics,
            },
            {
                "model_or_ensemble": "logistic_stacking_tta",
                "fold": fold,
                "evaluation_subset": "stacking_group_holdout",
                **stack_metrics,
            },
        ]
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results_path = destination / f"ensemble_fold_{fold}_metrics.csv"
    _write_metric_rows(metric_rows, results_path)
    predictions_path = destination / f"ensemble_fold_{fold}_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as predictions_file:
        writer = csv.writer(predictions_file)
        writer.writerow(
            ["path", "patient_id", "label"]
            + [field for name in model_names for field in (f"{name}_probability", f"{name}_tta_probability")]
            + ["mean_probability", "mean_tta_probability"]
        )
        for index, record in enumerate(validation_records):
            writer.writerow(
                [record["path"], record["patient_id"], record["label"]]
                + [
                    probability
                    for name in model_names
                    for probability in (original_probabilities[name][index], tta_probabilities[name][index])
                ]
                + [mean_probability[index], mean_tta_probability[index]]
            )

    best_single_name = max(model_names, key=lambda name: single_model_metrics[name]["auroc"])
    summary = {
        "fold": fold,
        "models": model_names,
        "tta_angles_degrees": tta_angles,
        "validation_images": len(validation_records),
        "validation_patient_groups": len(validation_groups),
        "official_test_used": False,
        "target_sensitivity": target_sensitivity,
        "best_single_tta_model_by_validation_auroc": best_single_name,
        "best_single_tta_metrics": single_model_metrics[best_single_name],
        "probability_mean_metrics": ensemble_metrics,
        "probability_mean_tta_metrics": ensemble_tta_metrics,
        "stacking_meta_train_patient_groups": len(set(patient_groups[meta_train_indices])),
        "stacking_meta_holdout_patient_groups": len(set(patient_groups[meta_holdout_indices])),
        "stacking_holdout_group_overlap": False,
        "stacking_holdout_probability_mean_metrics": ensemble_meta_metrics,
        "stacking_holdout_logistic_metrics": stack_metrics,
        "note": "Stacking is assessed on a patient-group-disjoint half of the validation fold; all results remain validation-only.",
        "metrics_csv": str(results_path),
        "predictions_csv": str(predictions_path),
    }
    summary_path = destination / f"ensemble_fold_{fold}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    LOGGER.info(
        "Fold %d validation ensemble: best single=%s AUROC=%.4f, mean TTA AUROC=%.4f, "
        "stacking holdout AUROC=%.4f",
        fold,
        best_single_name,
        single_model_metrics[best_single_name]["auroc"],
        ensemble_tta_metrics["auroc"],
        stack_metrics["auroc"],
    )
    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, required=True)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=project_root / "config.yaml")
    parser.add_argument("--splits", type=Path, default=project_root / "outputs" / "splits.csv")
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs" / "phase8")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(arguments.models) != len(arguments.checkpoints):
        parser.error("Provide one checkpoint for every model in --models")

    with arguments.config.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    fold = int(config["phase4"]["fold"] if arguments.fold is None else arguments.fold)
    run_validation_ensemble(
        model_names=arguments.models,
        checkpoint_paths=arguments.checkpoints,
        fold=fold,
        splits_path=arguments.splits,
        config=config,
        output_dir=arguments.output_dir,
    )


if __name__ == "__main__":
    main()