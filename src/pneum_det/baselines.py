"""Classical fold-0 baselines for chest X-ray classification."""

import argparse
import csv
import logging
import random
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOGGER = logging.getLogger(__name__)
_LABEL_TO_BINARY = {"NORMAL": 0, "PNEUMONIA": 1}
_RESULT_FIELDS = (
    "model",
    "fold",
    "threshold",
    "accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1",
    "auroc",
    "auprc",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
)


class _ImageDataset(Dataset):
    def __init__(self, records: list[dict[str, str]], transform: Any) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(record["path"]) as image:
            tensor = self.transform(image.convert("L"))
        label = torch.tensor(_LABEL_TO_BINARY[record["label"]], dtype=torch.float32)
        return tensor, label


def _save_result_rows(
    output_path: str | Path,
    result_rows: list[dict[str, float | int | str]],
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    if destination.exists():
        with destination.open(newline="", encoding="utf-8") as result_file:
            existing_rows = list(csv.DictReader(result_file))
    replacement_keys = {(str(row["model"]), str(row["fold"])) for row in result_rows}
    existing_rows = [
        row for row in existing_rows
        if (row["model"], row["fold"]) not in replacement_keys
    ]
    with destination.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=_RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows + result_rows)


def _set_training_mode(
    model: nn.Module,
    model_name: str,
    freeze_backbone: bool,
) -> None:
    model.train()
    if not freeze_backbone:
        return

    classifier_name = {
        "resnet50": "fc",
        "densenet121": "classifier",
        "efficientnet_b0": "classifier",
        "convnext_tiny": "classifier",
        "vit_b_16": "heads",
    }[model_name]
    for name, module in model.named_children():
        if name == classifier_name:
            module.train()
        else:
            module.eval()


def run_neural_pilot(
    splits_path: str | Path,
    output_path: str | Path,
    *,
    fold: int,
    model_name: str,
    pretrained: bool,
    freeze_backbone: bool,
    preprocessing_config: dict[str, Any],
    training_config: dict[str, Any],
    dropout: float,
    seed: int,
) -> dict[str, float | int | str]:
    """Run a one-epoch fold-screening pilot, not the final training recipe."""
    from pneum_det.models import create_model
    from pneum_det.transforms import build_image_transforms

    training = _load_fold_partition(Path(splits_path), fold, "train")
    validation = _load_fold_partition(Path(splits_path), fold, "validation")
    train_labels = _binary_labels(training)
    validation_labels = _binary_labels(validation)
    if np.unique(train_labels).size != 2 or np.unique(validation_labels).size != 2:
        raise ValueError("Both classes must be present in training and validation")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_transform, eval_transform = build_image_transforms(preprocessing_config)
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(training_config["batch_size"])
    training_loader = DataLoader(
        _ImageDataset(training, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        _ImageDataset(validation, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    ).to(device)
    positive_weight = torch.tensor(
        [(train_labels == 0).sum() / (train_labels == 1).sum()],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    head_markers = ("fc.", "classifier.", "heads.head.")
    if model_name == "custom_cnn":
        head_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        backbone_parameters = []
    else:
        named_trainable = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        head_parameters = [
            parameter
            for name, parameter in named_trainable
            if name.startswith(head_markers)
        ]
        backbone_parameters = [
            parameter
            for name, parameter in named_trainable
            if not name.startswith(head_markers)
        ]
    optimizer_groups = []
    if backbone_parameters:
        optimizer_groups.append(
            {
                "params": backbone_parameters,
                "lr": float(training_config["backbone_learning_rate"]),
            }
        )
    if head_parameters:
        optimizer_groups.append(
            {
                "params": head_parameters,
                "lr": float(training_config["head_learning_rate"]),
            }
        )
    if not optimizer_groups:
        raise ValueError(f"No trainable parameters found for {model_name}")
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-2)
    use_amp = device.type == "cuda" and bool(training_config["mixed_precision"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def train_batch(batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        _set_training_mode(model, model_name, freeze_backbone)
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    iterator = iter(training_loader)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    preflight_times = []
    for _ in range(min(3, len(training_loader))):
        preflight_start = time.perf_counter()
        batch = next(iterator)
        train_batch(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        preflight_times.append(time.perf_counter() - preflight_start)
    if device.type == "cuda":
        peak_memory_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        free_memory_gib, total_memory_gib = torch.cuda.mem_get_info(device)
    else:
        peak_memory_gib = 0.0
        free_memory_gib = 0.0
        total_memory_gib = 0.0
    steady_state_seconds = float(np.median(preflight_times[1:] or preflight_times))
    estimated_seconds = steady_state_seconds * len(training_loader)
    LOGGER.info(
        "%s one-epoch estimate: %.1f min; pretrained=%s frozen=%s batch=%d; "
        "preflight peak=%.2f GiB; free/total VRAM=%.2f/%.2f GiB; device=%s",
        model_name,
        estimated_seconds / 60,
        pretrained,
        freeze_backbone,
        batch_size,
        peak_memory_gib,
        free_memory_gib / 1024**3,
        total_memory_gib / 1024**3,
        device,
    )
    for batch in iterator:
        train_batch(batch)

    model.eval()
    validation_probabilities = []
    with torch.inference_mode():
        for images, _ in validation_loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images.to(device, non_blocking=True)).squeeze(1)
            validation_probabilities.extend(torch.sigmoid(logits).float().cpu().tolist())

    if model_name == "custom_cnn":
        result_name = "custom_cnn_one_epoch"
    elif freeze_backbone:
        result_name = f"{model_name}_frozen_one_epoch"
    else:
        result_name = f"{model_name}_finetuned_one_epoch"
    result = {
        "model": result_name,
        "fold": fold,
        **compute_binary_metrics(
            validation_labels,
            np.asarray(validation_probabilities, dtype=np.float32),
        ),
    }
    _save_result_rows(output_path, [result])
    return result


def _binary_labels(records: list[dict[str, str]]) -> np.ndarray:
    try:
        return np.asarray([_LABEL_TO_BINARY[record["label"]] for record in records], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unsupported class label: {exc.args[0]}") from exc


def compute_binary_metrics(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute screening metrics at a fixed, non-tuned threshold."""
    predictions = (probabilities >= threshold).astype(np.int64)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    specificity = (
        float(true_negative / (true_negative + false_positive))
        if true_negative + false_positive
        else 0.0
    )
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "precision": float(precision_score(true_labels, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(true_labels, predictions, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(true_labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(true_labels, probabilities)),
        "auprc": float(average_precision_score(true_labels, probabilities)),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def _load_fold_partition(
    splits_path: Path,
    fold: int,
    partition: str,
) -> list[dict[str, str]]:
    with splits_path.open(newline="", encoding="utf-8") as split_file:
        records = [
            record
            for record in csv.DictReader(split_file)
            if int(record["fold"]) == fold and record["partition"] == partition
        ]
    if not records:
        raise ValueError(f"No records found for fold {fold} partition {partition}")
    if any(record["official_split"] == "test" for record in records):
        raise AssertionError("Official test rows must not be used in Phase 4 baselines")
    return records


def _extract_hog_features(
    records: list[dict[str, str]],
    *,
    image_size: int,
    block_size: tuple[int, int],
    block_stride: tuple[int, int],
    cell_size: tuple[int, int],
    bins: int,
) -> np.ndarray:
    if image_size % cell_size[0] or image_size % cell_size[1]:
        raise ValueError("HOG image size must be divisible by the cell dimensions")
    if block_size[0] % cell_size[0] or block_size[1] % cell_size[1]:
        raise ValueError("HOG block dimensions must be divisible by the cell dimensions")
    if block_stride[0] % cell_size[0] or block_stride[1] % cell_size[1]:
        raise ValueError("HOG block stride must be divisible by the cell dimensions")

    features = []
    for record in records:
        image = cv2.imread(record["path"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"Could not read image: {record['path']}")
        resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        feature = _compute_hog_descriptor(
            resized,
            block_size=block_size,
            block_stride=block_stride,
            cell_size=cell_size,
            bins=bins,
        )
        if not np.isfinite(feature).all():
            raise ValueError(f"HOG feature extraction failed: {record['path']}")
        features.append(feature.reshape(-1))
    return np.stack(features).astype(np.float32, copy=False)


def _compute_hog_descriptor(
    image: np.ndarray,
    *,
    block_size: tuple[int, int],
    block_stride: tuple[int, int],
    cell_size: tuple[int, int],
    bins: int,
) -> np.ndarray:
    """Build unsigned-gradient HOG with orientation interpolation and L2-Hys."""
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    orientation = np.mod(np.degrees(np.arctan2(gradient_y, gradient_x)), 180.0)
    bin_position = orientation * (bins / 180.0)
    lower_bins = np.floor(bin_position).astype(np.int32) % bins
    upper_bins = (lower_bins + 1) % bins
    upper_weight = bin_position - np.floor(bin_position)
    lower_weight = 1.0 - upper_weight

    cell_width, cell_height = cell_size
    cells_x = image.shape[1] // cell_width
    cells_y = image.shape[0] // cell_height
    cell_histograms = np.zeros((cells_y, cells_x, bins), dtype=np.float32)
    for cell_y in range(cells_y):
        y_slice = slice(cell_y * cell_height, (cell_y + 1) * cell_height)
        for cell_x in range(cells_x):
            x_slice = slice(cell_x * cell_width, (cell_x + 1) * cell_width)
            region = np.s_[y_slice, x_slice]
            lower_histogram = np.bincount(
                lower_bins[region].ravel(),
                weights=(magnitude[region] * lower_weight[region]).ravel(),
                minlength=bins,
            )
            upper_histogram = np.bincount(
                upper_bins[region].ravel(),
                weights=(magnitude[region] * upper_weight[region]).ravel(),
                minlength=bins,
            )
            cell_histograms[cell_y, cell_x] = lower_histogram + upper_histogram

    block_cells_x = block_size[0] // cell_width
    block_cells_y = block_size[1] // cell_height
    stride_cells_x = block_stride[0] // cell_width
    stride_cells_y = block_stride[1] // cell_height
    block_descriptors = []
    for start_y in range(0, cells_y - block_cells_y + 1, stride_cells_y):
        for start_x in range(0, cells_x - block_cells_x + 1, stride_cells_x):
            block = cell_histograms[
                start_y : start_y + block_cells_y,
                start_x : start_x + block_cells_x,
            ].reshape(-1)
            normalized = block / np.sqrt(np.dot(block, block) + 1e-6)
            normalized = np.minimum(normalized, 0.2)
            normalized /= np.sqrt(np.dot(normalized, normalized) + 1e-6)
            block_descriptors.append(normalized)
    if not block_descriptors:
        raise ValueError("HOG settings produce no image blocks")
    return np.concatenate(block_descriptors)


def run_classical_baselines(
    splits_path: str | Path,
    output_path: str | Path,
    *,
    fold: int,
    hog_config: dict[str, Any],
    seed: int,
) -> list[dict[str, float | int | str]]:
    """Evaluate majority-class and HOG-logistic baselines on one fold."""
    training = _load_fold_partition(Path(splits_path), fold, "train")
    validation = _load_fold_partition(Path(splits_path), fold, "validation")
    train_labels = _binary_labels(training)
    validation_labels = _binary_labels(validation)
    if np.unique(train_labels).size != 2 or np.unique(validation_labels).size != 2:
        raise ValueError("Both classes must be present in training and validation")

    majority_label = int(np.bincount(train_labels).argmax())
    result_rows: list[dict[str, float | int | str]] = []
    for model_name, probabilities in (
        ("majority_class", np.full(validation_labels.shape, majority_label, dtype=np.float32)),
    ):
        result_rows.append(
            {"model": model_name, "fold": fold, **compute_binary_metrics(validation_labels, probabilities)}
        )

    hog_options = {
        "image_size": int(hog_config["image_size"]),
        "block_size": tuple(map(int, hog_config["block_size"])),
        "block_stride": tuple(map(int, hog_config["block_stride"])),
        "cell_size": tuple(map(int, hog_config["cell_size"])),
        "bins": int(hog_config["bins"]),
    }
    train_features = _extract_hog_features(training, **hog_options)
    validation_features = _extract_hog_features(validation, **hog_options)
    LOGGER.info(
        "Extracted HOG features: train=%s validation=%s",
        train_features.shape,
        validation_features.shape,
    )

    logistic_config = hog_config["logistic"]
    classifier = LogisticRegression(
        C=float(logistic_config["c"]),
        class_weight=logistic_config["class_weight"],
        max_iter=int(logistic_config["max_iter"]),
        random_state=seed,
        solver="liblinear",
    )
    classifier.fit(train_features, train_labels)
    probabilities = classifier.predict_proba(validation_features)[:, 1]
    result_rows.append(
        {
            "model": "hog_logistic_regression",
            "fold": fold,
            **compute_binary_metrics(validation_labels, probabilities),
        }
    )

    _save_result_rows(output_path, result_rows)
    LOGGER.info("Saved fold %d classical baselines to %s", fold, output_path)
    return result_rows


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_root / "config.yaml")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--splits", type=Path, default=project_root / "outputs" / "splits.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "outputs" / "phase4" / "results.csv",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with arguments.config.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    phase4_config = config["phase4"]
    results = run_classical_baselines(
        arguments.splits,
        arguments.output,
        fold=phase4_config["fold"] if arguments.fold is None else arguments.fold,
        hog_config=phase4_config["hog"],
        seed=int(config["seed"]),
    )
    for pilot_config in phase4_config.get("neural_pilots", []):
        results.append(
            run_neural_pilot(
                arguments.splits,
                arguments.output,
                fold=phase4_config["fold"] if arguments.fold is None else arguments.fold,
                model_name=pilot_config["model_name"],
                pretrained=bool(pilot_config["pretrained"]),
                freeze_backbone=bool(pilot_config["freeze_backbone"]),
                preprocessing_config=config["preprocessing"],
                training_config=pilot_config,
                dropout=float(phase4_config["head_dropout"]),
                seed=int(config["seed"]),
            )
        )
    for result in results:
        LOGGER.info(
            "%s fold=%d AUROC=%.4f accuracy=%.4f sensitivity=%.4f specificity=%.4f F1=%.4f",
            result["model"],
            result["fold"],
            result["auroc"],
            result["accuracy"],
            result["recall_sensitivity"],
            result["specificity"],
            result["f1"],
        )


if __name__ == "__main__":
    main()