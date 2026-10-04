"""Validation-only Grad-CAM, subtype errors, and robustness checks."""

import argparse
import csv
import json
import logging
import random
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
from PIL import Image, ImageEnhance
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import recall_score
from torch import nn

from pneum_det.evaluate import compute_binary_metrics
from pneum_det.models import MODEL_NAMES, create_model
from pneum_det.train import read_fold_records
from pneum_det.transforms import build_image_transforms

LOGGER = logging.getLogger(__name__)


class BinaryOutputTarget:
    """Select the positive or negative score from a one-logit model."""

    def __init__(self, pneumonia_class: bool) -> None:
        self.pneumonia_class = pneumonia_class

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        logit = model_output.reshape(-1)[0]
        return logit if self.pneumonia_class else -logit


def select_error_examples(
    records: list[dict[str, str]],
    probabilities: np.ndarray,
    threshold: float,
    *,
    examples_per_category: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Choose deterministic validation examples for TP, TN, FP, and FN."""
    if len(records) != len(probabilities):
        raise ValueError("records and probabilities must have the same length")
    if examples_per_category < 1:
        raise ValueError("examples_per_category must be positive")
    categories: dict[str, list[dict[str, Any]]] = {name: [] for name in ("TP", "TN", "FP", "FN")}
    for record, probability in zip(records, probabilities):
        actual_positive = record["label"] == "PNEUMONIA"
        predicted_positive = float(probability) >= threshold
        category = (
            "TP" if actual_positive else "FP"
        ) if predicted_positive else (
            "FN" if actual_positive else "TN"
        )
        categories[category].append({**record, "probability": float(probability)})

    sampler = random.Random(seed)
    for category, candidates in categories.items():
        if len(candidates) < examples_per_category:
            raise ValueError(
                f"Need {examples_per_category} {category} validation examples; found {len(candidates)}"
            )
        sampler.shuffle(candidates)
        categories[category] = candidates[:examples_per_category]
    return categories


def border_activation_fraction(cam_mask: np.ndarray, fraction: float = 0.1) -> float:
    """Measure CAM mass near image borders as a shortcut-risk heuristic."""
    if not 0.0 < fraction < 0.5:
        raise ValueError("fraction must be between 0 and 0.5")
    mask = np.asarray(cam_mask, dtype=np.float32)
    border_size = max(1, int(round(min(mask.shape) * fraction)))
    border = np.zeros(mask.shape, dtype=bool)
    border[:border_size, :] = True
    border[-border_size:, :] = True
    border[:, :border_size] = True
    border[:, -border_size:] = True
    total = float(mask.sum())
    return float(mask[border].sum() / total) if total > 0 else 0.0


def _target_layers(model: nn.Module, model_name: str) -> tuple[list[nn.Module], Any | None]:
    if model_name == "resnet50":
        return [model.layer4[-1]], None
    if model_name == "densenet121":
        return [model.features.denseblock4], None
    if model_name == "efficientnet_b0":
        return [model.features[-1]], None
    if model_name == "convnext_tiny":
        return [model.features[-1]], None
    if model_name == "vit_b_16":
        def reshape_transform(activations: torch.Tensor) -> torch.Tensor:
            spatial_tokens = activations[:, 1:, :]
            side = int(np.sqrt(spatial_tokens.shape[1]))
            if side * side != spatial_tokens.shape[1]:
                raise ValueError("ViT token count cannot be reshaped into a square feature map")
            return spatial_tokens.reshape(
                spatial_tokens.shape[0], side, side, spatial_tokens.shape[2]
            ).permute(0, 3, 1, 2)

        return [model.encoder.layers[-1].ln_1], reshape_transform
    if model_name == "custom_cnn":
        convolution_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        if not convolution_layers:
            raise ValueError("custom_cnn contains no convolution layer for Grad-CAM")
        return [convolution_layers[-1]], None
    raise ValueError(f"No Grad-CAM target layer is defined for {model_name}")


def _make_perturbed_image(
    image: Image.Image,
    condition: str,
    *,
    seed: int,
) -> Image.Image:
    grayscale = image.convert("L")
    if condition == "gaussian_noise_sigma_0.03":
        pixels = np.asarray(grayscale, dtype=np.float32) / 255.0
        generator = np.random.default_rng(seed)
        noisy = np.clip(pixels + generator.normal(0.0, 0.03, pixels.shape), 0.0, 1.0)
        return Image.fromarray(np.asarray(noisy * 255.0, dtype=np.uint8), mode="L")
    if condition == "contrast_0.7":
        return ImageEnhance.Contrast(grayscale).enhance(0.7)
    if condition == "center_crop_90pct":
        width, height = grayscale.size
        horizontal_margin = max(1, int(round(width * 0.05)))
        vertical_margin = max(1, int(round(height * 0.05)))
        return grayscale.crop(
            (
                horizontal_margin,
                vertical_margin,
                width - horizontal_margin,
                height - vertical_margin,
            )
        )
    if condition == "original":
        return grayscale
    raise ValueError(f"Unknown robustness condition: {condition}")


def _predict_condition(
    records: list[dict[str, str]],
    model: nn.Module,
    transform: Any,
    device: torch.device,
    *,
    condition: str,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    probabilities = []
    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, len(records), batch_size):
            batch_records = records[batch_start : batch_start + batch_size]
            tensors = []
            for offset, record in enumerate(batch_records):
                with Image.open(record["path"]) as source:
                    transformed_image = _make_perturbed_image(
                        source,
                        condition,
                        seed=seed + batch_start + offset,
                    )
                    tensors.append(transform(transformed_image))
            batch = torch.stack(tensors).to(device, non_blocking=True)
            logits = model(batch).squeeze(1)
            probabilities.extend(torch.sigmoid(logits).float().cpu().tolist())
    return np.asarray(probabilities, dtype=np.float32)


def _error_by_subtype(
    records: list[dict[str, str]],
    probabilities: np.ndarray,
    threshold: float,
) -> list[dict[str, str | float | int]]:
    rows = []
    pneumonia_subtypes = sorted(
        {
            record.get("subtype", "unknown")
            for record in records
            if record["label"] == "PNEUMONIA"
        }
    )
    for subtype in pneumonia_subtypes:
        indices = [
            index
            for index, record in enumerate(records)
            if record["label"] == "PNEUMONIA" and record.get("subtype", "unknown") == subtype
        ]
        subtype_probabilities = probabilities[indices]
        false_negatives = int(np.sum(subtype_probabilities < threshold))
        rows.append(
            {
                "subtype": subtype,
                "pneumonia_images": len(indices),
                "true_positives": len(indices) - false_negatives,
                "false_negatives": false_negatives,
                "sensitivity": float(1.0 - false_negatives / len(indices)),
            }
        )
    return rows


def run_phase7(
    *,
    model_name: str,
    fold: int,
    checkpoint_path: str | Path,
    predictions_path: str | Path,
    phase6_metrics_path: str | Path,
    splits_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    examples_per_category: int = 8,
) -> dict[str, Any]:
    """Generate validation-only Grad-CAM sheets and robustness summaries."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["model_name"] != model_name or int(checkpoint["fold"]) != fold:
        raise ValueError("Checkpoint model/fold does not match Phase 7 request")
    records = read_fold_records(splits_path, fold, "validation")
    if any(record["official_split"] == "test" for record in records):
        raise AssertionError("Official test images must not enter Phase 7")
    with Path(predictions_path).open(newline="", encoding="utf-8") as prediction_file:
        prediction_rows = list(csv.DictReader(prediction_file))
    predictions_by_path = {row["path"]: row for row in prediction_rows}
    record_paths = {record["path"] for record in records}
    if set(predictions_by_path) != record_paths:
        raise ValueError("Phase 6 predictions do not exactly match the validation fold")
    if any(predictions_by_path[record["path"]]["label"] != record["label"] for record in records):
        raise ValueError("Phase 6 prediction labels disagree with the validation split")
    probabilities = np.asarray(
        [float(predictions_by_path[record["path"]]["probability"]) for record in records],
        dtype=np.float32,
    )
    labels = np.asarray([int(record["label"] == "PNEUMONIA") for record in records], dtype=np.int64)
    phase6_report = json.loads(Path(phase6_metrics_path).read_text(encoding="utf-8"))
    threshold = float(phase6_report["selected_threshold_metrics"]["threshold"])
    categories = select_error_examples(
        records,
        probabilities,
        threshold,
        examples_per_category=examples_per_category,
        seed=int(config["seed"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_name,
        pretrained=False,
        freeze_backbone=bool(checkpoint["freeze_backbone"]),
        dropout=float(config["phase4"]["head_dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    target_layers, reshape_transform = _target_layers(model, model_name)
    _, eval_transform = build_image_transforms(config["preprocessing"])

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    border_review_rows = []
    figure, axes = plt.subplots(
        4,
        examples_per_category,
        figsize=(2.4 * examples_per_category, 8),
        squeeze=False,
    )
    with GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    ) as cam:
        for row_index, category in enumerate(("TP", "TN", "FP", "FN")):
            for column_index, record in enumerate(categories[category]):
                probability = float(record["probability"])
                predicted_pneumonia = probability >= threshold
                with Image.open(record["path"]) as source:
                    original = source.convert("L")
                    tensor = eval_transform(original).unsqueeze(0).to(device)
                    cam_mask = cam(
                        input_tensor=tensor,
                        targets=[BinaryOutputTarget(predicted_pneumonia)],
                    )[0]
                    image_size = config["preprocessing"]["image_size"]
                    display_image = original.resize(
                        (image_size, image_size),
                        resample=Image.Resampling.BILINEAR,
                    )
                    grayscale = np.asarray(display_image, dtype=np.float32) / 255.0
                rgb_image = np.repeat(grayscale[:, :, None], 3, axis=2)
                overlay = show_cam_on_image(rgb_image, cam_mask, use_rgb=True)
                border_fraction = border_activation_fraction(cam_mask)
                axis = axes[row_index, column_index]
                axis.imshow(overlay)
                axis.set_title(
                    f"{Path(record['path']).name[:18]}\np={probability:.2f} edge={border_fraction:.2f}",
                    fontsize=6,
                )
                axis.axis("off")
                border_review_rows.append(
                    {
                        "category": category,
                        "path": record["path"],
                        "patient_id": record["patient_id"],
                        "subtype": record.get("subtype", "unknown"),
                        "probability": probability,
                        "border_activation_fraction": border_fraction,
                        "border_heuristic_flag": border_fraction >= 0.35,
                    }
                )
    figure.suptitle(f"Fold {fold} validation Grad-CAM examples | target: model decision")
    figure.tight_layout()
    cam_path = destination / f"{model_name}_fold_{fold}_gradcam_examples.png"
    figure.savefig(cam_path, dpi=160)
    plt.close(figure)

    cam_review_path = destination / f"{model_name}_fold_{fold}_cam_review.csv"
    with cam_review_path.open("w", newline="", encoding="utf-8") as review_file:
        writer = csv.DictWriter(review_file, fieldnames=border_review_rows[0].keys())
        writer.writeheader()
        writer.writerows(border_review_rows)

    subtype_rows = _error_by_subtype(records, probabilities, threshold)
    subtype_path = destination / f"{model_name}_fold_{fold}_subtype_errors.csv"
    with subtype_path.open("w", newline="", encoding="utf-8") as subtype_file:
        writer = csv.DictWriter(
            subtype_file,
            fieldnames=("subtype", "pneumonia_images", "true_positives", "false_negatives", "sensitivity"),
        )
        writer.writeheader()
        writer.writerows(subtype_rows)

    baseline_metrics = compute_binary_metrics(labels, probabilities, threshold)
    condition_names = ("gaussian_noise_sigma_0.03", "contrast_0.7", "center_crop_90pct")
    robustness_rows = [
        {
            "condition": "original_validation",
            **baseline_metrics,
            "auroc_drop": 0.0,
            "sensitivity_drop": 0.0,
        }
    ]
    batch_size = int(config["training"]["batch_sizes"].get(model_name, config["training"]["batch_size"]))
    for condition in condition_names:
        perturbed_probabilities = _predict_condition(
            records,
            model,
            eval_transform,
            device,
            condition=condition,
            batch_size=batch_size,
            seed=int(config["seed"]),
        )
        metrics = compute_binary_metrics(labels, perturbed_probabilities, threshold)
        robustness_rows.append(
            {
                "condition": condition,
                **metrics,
                "auroc_drop": baseline_metrics["auroc"] - metrics["auroc"],
                "sensitivity_drop": baseline_metrics["sensitivity"] - metrics["sensitivity"],
            }
        )
    robustness_path = destination / f"{model_name}_fold_{fold}_robustness.csv"
    with robustness_path.open("w", newline="", encoding="utf-8") as robustness_file:
        writer = csv.DictWriter(robustness_file, fieldnames=robustness_rows[0].keys())
        writer.writeheader()
        writer.writerows(robustness_rows)

    flagged_examples = sum(bool(row["border_heuristic_flag"]) for row in border_review_rows)
    summary = {
        "model": model_name,
        "fold": fold,
        "validation_images": len(records),
        "official_test_used": False,
        "threshold": threshold,
        "error_examples_per_category": examples_per_category,
        "error_category_counts": {category: len(rows) for category, rows in categories.items()},
        "error_by_pneumonia_subtype": subtype_rows,
        "border_shortcut_heuristic_flagged_examples": flagged_examples,
        "border_shortcut_heuristic_total_cam_examples": len(border_review_rows),
        "border_shortcut_note": (
            "Border activation is a heuristic only, not an anatomical lung mask. Manually inspect CAM overlays "
            "for borders, corners, text, and markers before drawing shortcut-learning conclusions."
        ),
        "robustness": robustness_rows,
        "artifacts": {
            "gradcam_examples": str(cam_path),
            "cam_review_csv": str(cam_review_path),
            "subtype_errors_csv": str(subtype_path),
            "robustness_csv": str(robustness_path),
        },
    }
    summary_path = destination / f"{model_name}_fold_{fold}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    LOGGER.info(
        "Phase 7 validation analysis complete for %s fold %d: CAM=%d/category, border heuristic flagged %d/%d",
        model_name,
        fold,
        examples_per_category,
        flagged_examples,
        len(border_review_rows),
    )
    LOGGER.info("Saved Phase 7 outputs to %s", destination)
    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--config", type=Path, default=project_root / "config.yaml")
    parser.add_argument("--splits", type=Path, default=project_root / "outputs" / "splits.csv")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs" / "phase7")
    parser.add_argument("--examples-per-category", type=int, default=8)
    parser.add_argument("--batch-size", type=int)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with arguments.config.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    fold = int(config["phase4"]["fold"] if arguments.fold is None else arguments.fold)
    output_dir = arguments.output_dir
    checkpoint_path = arguments.checkpoint or (
        project_root / "outputs" / "checkpoints" / f"{arguments.model}_fold_{fold}" / "best.pt"
    )
    predictions_path = arguments.predictions or (
        project_root / "outputs" / "phase6" / f"{arguments.model}_fold_{fold}_predictions.csv"
    )
    metrics_path = arguments.metrics or (
        project_root / "outputs" / "phase6" / f"{arguments.model}_fold_{fold}_metrics.json"
    )
    if arguments.batch_size is not None:
        config["training"] = dict(config["training"])
        config["training"]["batch_sizes"] = {
            **config["training"].get("batch_sizes", {}),
            arguments.model: arguments.batch_size,
        }
    run_phase7(
        model_name=arguments.model,
        fold=fold,
        checkpoint_path=checkpoint_path,
        predictions_path=predictions_path,
        phase6_metrics_path=metrics_path,
        splits_path=arguments.splits,
        config=config,
        output_dir=output_dir,
        examples_per_category=arguments.examples_per_category,
    )


if __name__ == "__main__":
    main()