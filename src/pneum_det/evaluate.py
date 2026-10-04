"""Validation-fold metrics, calibration, thresholds, and uncertainty."""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from pneum_det.models import MODEL_NAMES, create_model
from pneum_det.train import XrayDataset, read_fold_records
from pneum_det.transforms import build_image_transforms

LOGGER = logging.getLogger(__name__)


def compute_binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Compute threshold and ranking metrics for pneumonia as the positive class."""
    predictions = (probabilities >= threshold).astype(np.int64)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(true_negative / (true_negative + false_positive))
        if true_negative + false_positive
        else 0.0,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def select_sensitivity_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_sensitivity: float,
) -> float:
    """Choose the highest threshold that still reaches target sensitivity."""
    if not 0.0 < target_sensitivity <= 1.0:
        raise ValueError("target_sensitivity must be in (0, 1]")
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    eligible = [
        float(threshold)
        for threshold in candidates
        if recall_score(labels, probabilities >= threshold, zero_division=0) >= target_sensitivity
    ]
    if not eligible:
        return 0.0
    return max(eligible)


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, list[dict[str, float | int]]]:
    """Calculate equal-width ECE and bin statistics."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.minimum(np.digitize(probabilities, edges[1:-1]), n_bins - 1)
    total = len(labels)
    ece = 0.0
    bin_rows = []
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        count = int(mask.sum())
        if not count:
            continue
        confidence = float(probabilities[mask].mean())
        observed_rate = float(labels[mask].mean())
        ece += count / total * abs(confidence - observed_rate)
        bin_rows.append(
            {
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "count": count,
                "confidence": confidence,
                "accuracy": observed_rate,
            }
        )
    return float(ece), bin_rows


def fit_temperature(labels: np.ndarray, logits: np.ndarray) -> float:
    """Fit a scalar temperature by minimizing validation negative log-likelihood."""
    targets = labels.astype(np.float64)
    values = logits.astype(np.float64)

    def negative_log_likelihood(log_temperature: float) -> float:
        scaled_logits = values / np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0.0, scaled_logits) - targets * scaled_logits))

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(float(np.log(0.05)), float(np.log(20.0))),
        method="bounded",
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Temperature fitting failed: {result.message}")
    return float(np.exp(result.x))


def grouped_bootstrap_intervals(
    records: list[dict[str, str]],
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    """Bootstrap by patient group to preserve within-patient image clusters."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record["patient_id"], []).append(index)
    group_names = np.asarray(sorted(groups))
    if group_names.size < 2:
        raise ValueError("At least two patient groups are required for bootstrap intervals")

    rng = np.random.default_rng(seed)
    metric_names = ("accuracy", "precision", "sensitivity", "specificity", "f1", "auroc", "auprc")
    samples = {metric: [] for metric in metric_names}
    valid_resamples = 0
    for _ in range(n_resamples):
        sampled_groups = rng.choice(group_names, size=len(group_names), replace=True)
        indices = np.concatenate([groups[group] for group in sampled_groups])
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        metrics = compute_binary_metrics(sampled_labels, probabilities[indices], threshold)
        valid_resamples += 1
        for metric_name in metric_names:
            samples[metric_name].append(float(metrics[metric_name]))
    if valid_resamples < max(20, n_resamples // 2):
        raise RuntimeError(
            f"Only {valid_resamples}/{n_resamples} valid patient-group bootstrap samples"
        )
    return {
        metric_name: {
            "lower": float(np.percentile(values, 2.5)),
            "upper": float(np.percentile(values, 97.5)),
            "valid_resamples": valid_resamples,
        }
        for metric_name, values in samples.items()
    }


def summarize_folds(metrics_paths: list[Path]) -> dict[str, Any]:
    """Return descriptive mean/std for the available fold metrics."""
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in metrics_paths]
    if not reports:
        return {"n_folds": 0, "metrics": {}}
    metric_names = ("auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1")
    values = {
        metric: [report["selected_threshold_metrics"][metric] for report in reports]
        for metric in metric_names
    }
    return {
        "n_folds": len(reports),
        "metrics": {
            metric: {
                "mean": float(np.mean(metric_values)),
                "std": float(np.std(metric_values, ddof=1)) if len(metric_values) > 1 else 0.0,
            }
            for metric, metric_values in values.items()
        },
    }


def _save_roc_pr_figure(labels: np.ndarray, probabilities: np.ndarray, path: Path) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(false_positive_rate, true_positive_rate, label=f"AUROC {roc_auc_score(labels, probabilities):.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set(xlabel="False-positive rate", ylabel="Sensitivity", title="ROC")
    axes[0].legend()
    axes[1].plot(recall, precision, label=f"AUPRC {average_precision_score(labels, probabilities):.3f}")
    axes[1].axhline(float(labels.mean()), linestyle="--", color="gray", label="Prevalence")
    axes[1].set(xlabel="Sensitivity / recall", ylabel="Precision", title="Precision-recall")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_calibration_figure(
    before_bins: list[dict[str, float | int]],
    after_bins: list[dict[str, float | int]],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    for bins, name in ((before_bins, "Before temperature scaling"), (after_bins, "After temperature scaling")):
        axis.plot(
            [row["confidence"] for row in bins],
            [row["accuracy"] for row in bins],
            marker="o",
            label=name,
        )
    axis.set(xlabel="Mean predicted probability", ylabel="Observed pneumonia rate", title="Reliability diagram")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate_fold(
    *,
    model_name: str,
    fold: int,
    checkpoint_path: str | Path,
    splits_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate one saved checkpoint using its validation fold only."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["model_name"] != model_name or int(checkpoint["fold"]) != fold:
        raise ValueError("Checkpoint model/fold does not match the requested evaluation")
    records = read_fold_records(splits_path, fold, "validation")
    if any(record["official_split"] == "test" for record in records):
        raise AssertionError("Official test images must not enter validation evaluation")

    _, eval_transform = build_image_transforms(config["preprocessing"])
    batch_size = int(config["training"]["batch_sizes"].get(model_name, config["training"]["batch_size"]))
    loader = DataLoader(
        XrayDataset(records, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    model = create_model(
        model_name,
        pretrained=False,
        freeze_backbone=bool(checkpoint["freeze_backbone"]),
        dropout=float(config["phase4"]["head_dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    logits = []
    labels = []
    with torch.inference_mode():
        for images, batch_labels in loader:
            batch_logits = model(images.to(device, non_blocking=True)).squeeze(1)
            logits.extend(batch_logits.float().cpu().tolist())
            labels.extend(batch_labels.int().tolist())
    logits_array = np.asarray(logits, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities = expit(logits_array)

    evaluation_config = config["evaluation"]
    default_threshold = float(evaluation_config["default_threshold"])
    target_sensitivity = float(evaluation_config["target_sensitivity"])
    threshold = select_sensitivity_threshold(labels_array, probabilities, target_sensitivity)
    default_metrics = compute_binary_metrics(labels_array, probabilities, default_threshold)
    selected_metrics = compute_binary_metrics(labels_array, probabilities, threshold)

    n_bins = int(evaluation_config["calibration_bins"])
    ece_before, reliability_before = expected_calibration_error(labels_array, probabilities, n_bins)
    temperature = fit_temperature(labels_array, logits_array)
    calibrated_probabilities = expit(logits_array / temperature)
    ece_after, reliability_after = expected_calibration_error(
        labels_array,
        calibrated_probabilities,
        n_bins,
    )
    nll_before = float(np.mean(np.logaddexp(0.0, logits_array) - labels_array * logits_array))
    scaled_logits = logits_array / temperature
    nll_after = float(np.mean(np.logaddexp(0.0, scaled_logits) - labels_array * scaled_logits))
    bootstrap_intervals = grouped_bootstrap_intervals(
        records,
        labels_array,
        probabilities,
        threshold,
        n_resamples=int(evaluation_config["bootstrap_resamples"]),
        seed=int(config["seed"]),
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    predictions_path = destination / f"{model_name}_fold_{fold}_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as prediction_file:
        fields = ("path", "patient_id", "label", "logit", "probability", "calibrated_probability")
        writer = csv.DictWriter(prediction_file, fieldnames=fields)
        writer.writeheader()
        for record, logit, probability, calibrated_probability in zip(
            records,
            logits_array,
            probabilities,
            calibrated_probabilities,
        ):
            writer.writerow(
                {
                    "path": record["path"],
                    "patient_id": record["patient_id"],
                    "label": record["label"],
                    "logit": float(logit),
                    "probability": float(probability),
                    "calibrated_probability": float(calibrated_probability),
                }
            )

    report = {
        "model": model_name,
        "fold": fold,
        "n_validation_images": len(records),
        "n_validation_patient_groups": len({record["patient_id"] for record in records}),
        "official_test_used": False,
        "default_threshold_metrics": default_metrics,
        "target_sensitivity": target_sensitivity,
        "selected_threshold_metrics": selected_metrics,
        "calibration": {
            "temperature": temperature,
            "nll_before": nll_before,
            "nll_after": nll_after,
            "ece_before": ece_before,
            "ece_after": ece_after,
        },
        "bootstrap_95_percent_ci_patient_grouped": bootstrap_intervals,
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "predictions_csv": str(predictions_path),
    }
    metrics_path = destination / f"{model_name}_fold_{fold}_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    model_metrics_paths = sorted(destination.glob(f"{model_name}_fold_*_metrics.json"))
    fold_summary = summarize_folds(model_metrics_paths)
    report["available_fold_summary"] = fold_summary
    metrics_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    _save_roc_pr_figure(
        labels_array,
        probabilities,
        destination / f"{model_name}_fold_{fold}_roc_pr.png",
    )
    _save_calibration_figure(
        reliability_before,
        reliability_after,
        destination / f"{model_name}_fold_{fold}_calibration.png",
    )
    LOGGER.info(
        "%s fold %d validation: AUROC=%.4f AUPRC=%.4f threshold@%.1f%% sensitivity=%.5f "
        "specificity=%.4f ECE %.4f -> %.4f temperature=%.4f",
        model_name,
        fold,
        default_metrics["auroc"],
        default_metrics["auprc"],
        target_sensitivity * 100,
        threshold,
        selected_metrics["specificity"],
        ece_before,
        ece_after,
        temperature,
    )
    LOGGER.info("Saved validation metrics and figures to %s", destination)
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--config", type=Path, default=project_root / "config.yaml")
    parser.add_argument("--splits", type=Path, default=project_root / "outputs" / "splits.csv")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs" / "phase6")
    parser.add_argument("--bootstrap-resamples", type=int)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with arguments.config.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    fold = int(config["phase4"]["fold"] if arguments.fold is None else arguments.fold)
    if arguments.bootstrap_resamples is not None:
        config["evaluation"] = dict(config["evaluation"])
        config["evaluation"]["bootstrap_resamples"] = arguments.bootstrap_resamples
    checkpoint_path = arguments.checkpoint or (
        project_root / "outputs" / "checkpoints" / f"{arguments.model}_fold_{fold}" / "best.pt"
    )
    evaluate_fold(
        model_name=arguments.model,
        fold=fold,
        checkpoint_path=checkpoint_path,
        splits_path=arguments.splits,
        config=config,
        output_dir=arguments.output_dir,
    )


if __name__ == "__main__":
    main()