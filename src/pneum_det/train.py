"""Train one binary classifier on a patient-grouped development fold."""

import argparse
import csv
import itertools
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from pneum_det.models import MODEL_NAMES, create_model
from pneum_det.transforms import build_image_transforms

LOGGER = logging.getLogger(__name__)
_LABEL_TO_BINARY = {"NORMAL": 0, "PNEUMONIA": 1}
_METRIC_FIELDS = (
    "epoch",
    "train_loss",
    "validation_loss",
    "validation_auroc",
    "validation_f1",
    "learning_rates",
    "epoch_seconds",
)


class XrayDataset(Dataset):
    """Read one grayscale image and label from a split record."""

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


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_fold_records(
    splits_path: str | Path,
    fold: int,
    partition: str,
) -> list[dict[str, str]]:
    """Read one fold partition while excluding official test rows."""
    if partition not in {"train", "validation"}:
        raise ValueError("partition must be 'train' or 'validation'")
    with Path(splits_path).open(newline="", encoding="utf-8") as split_file:
        records = [
            record
            for record in csv.DictReader(split_file)
            if int(record["fold"]) == fold and record["partition"] == partition
        ]
    if not records:
        raise ValueError(f"No records found for fold {fold} partition {partition}")
    if any(record["official_split"] == "test" for record in records):
        raise AssertionError("Official test rows must not be used by the trainer")
    if any(record["label"] not in _LABEL_TO_BINARY for record in records):
        raise ValueError("Split records contain an unsupported class label")
    return records


def compute_positive_weight(records: list[dict[str, str]]) -> torch.Tensor:
    """Compute negative/positive count ratio for BCEWithLogitsLoss."""
    positive_count = sum(record["label"] == "PNEUMONIA" for record in records)
    negative_count = sum(record["label"] == "NORMAL" for record in records)
    if not positive_count or not negative_count:
        raise ValueError("Training data must contain both NORMAL and PNEUMONIA")
    return torch.tensor([negative_count / positive_count], dtype=torch.float32)


def binary_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Return mean binary focal loss from logits and binary targets."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("focal alpha must be in [0, 1]")
    if gamma < 0.0:
        raise ValueError("focal gamma must be non-negative")
    probabilities = torch.sigmoid(logits)
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    target_probability = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_weight = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_weight * (1.0 - target_probability).pow(gamma) * cross_entropy).mean()


def _read_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _set_model_training_mode(model: nn.Module, model_name: str, freeze_backbone: bool) -> None:
    model.train()
    if not freeze_backbone:
        return
    classifier_name = {
        "resnet50": "fc",
        "densenet121": "classifier",
        "efficientnet_b0": "classifier",
        "convnext_tiny": "classifier",
        "vit_b_16": "heads",
        "swin_t": "head",
    }[model_name]
    for name, module in model.named_children():
        if name == classifier_name:
            module.train()
        else:
            module.eval()


def _make_optimizer(
    model: nn.Module,
    model_name: str,
    learning_rates: dict[str, float],
    weight_decay: float,
) -> torch.optim.Optimizer:
    if model_name == "custom_cnn":
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        groups = [{"params": parameters, "lr": learning_rates["custom_cnn"], "name": "custom_cnn"}]
    else:
        head_markers = ("fc.", "classifier.", "heads.head.", "head.")
        trainable_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        backbone = [parameter for name, parameter in trainable_parameters if not name.startswith(head_markers)]
        head = [parameter for name, parameter in trainable_parameters if name.startswith(head_markers)]
        groups = []
        if backbone:
            groups.append({"params": backbone, "lr": learning_rates["backbone"], "name": "backbone"})
        if head:
            groups.append({"params": head, "lr": learning_rates["head"], "name": "head"})
    if not groups or any(not group["params"] for group in groups):
        raise ValueError(f"No trainable parameters found for {model_name}")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def _compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_name: str,
    positive_weight: torch.Tensor,
    focal_alpha: float,
    focal_gamma: float,
) -> torch.Tensor:
    if loss_name == "focal":
        return binary_focal_loss(logits, targets, alpha=focal_alpha, gamma=focal_gamma)
    if loss_name == "bce_with_logits":
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=positive_weight,
        )
    raise ValueError(f"Unsupported training loss: {loss_name}")


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _write_history(history: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def _save_training_curves(history: list[dict[str, Any]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], marker="o", label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_auroc"] for row in history], marker="o", label="validation AUROC")
    axes[1].plot(epochs, [row["validation_f1"] for row in history], marker="o", label="validation F1")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train_fold(
    *,
    model_name: str,
    fold: int,
    config: dict[str, Any],
    splits_path: str | Path,
    output_dir: str | Path,
    pretrained: bool,
    freeze_backbone: bool,
    resume: bool = False,
) -> dict[str, Any]:
    """Train a model with best/latest checkpoints and AUROC early stopping."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}")
    if model_name == "custom_cnn" and pretrained:
        raise ValueError("custom_cnn does not support pretrained weights")
    if model_name == "custom_cnn" and freeze_backbone:
        raise ValueError("custom_cnn has no pretrained backbone to freeze")

    training_records = read_fold_records(splits_path, fold, "train")
    validation_records = read_fold_records(splits_path, fold, "validation")
    shared_groups = {row["patient_id"] for row in training_records} & {
        row["patient_id"] for row in validation_records
    }
    if shared_groups:
        raise AssertionError(f"Patient groups leak across fold {fold}: {sorted(shared_groups)[:10]}")

    training_config = config["training"]
    epochs = int(training_config["epochs"])
    if epochs < 1:
        raise ValueError("training.epochs must be at least 1")
    batch_size = int(training_config["batch_sizes"].get(model_name, training_config["batch_size"]))
    seed = int(config["seed"])
    seed_everything(seed)

    train_transform, validation_transform = build_image_transforms(config["preprocessing"])
    data_generator = torch.Generator().manual_seed(seed)
    train_dataset = XrayDataset(training_records, train_transform)
    validation_dataset = XrayDataset(validation_records, validation_transform)
    sampler = None
    if bool(training_config["weighted_random_sampler"]):
        class_counts = {
            label: sum(record["label"] == label for record in training_records)
            for label in _LABEL_TO_BINARY
        }
        sample_weights = [
            1.0 / class_counts[record["label"]]
            for record in training_records
        ]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=data_generator,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(training_config["num_workers"]),
        generator=data_generator,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(training_config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=float(config["phase4"]["head_dropout"]),
    ).to(device)
    learning_rates = {
        key: float(value)
        for key, value in training_config["learning_rate"].items()
    }
    optimizer = _make_optimizer(
        model,
        model_name,
        learning_rates,
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler_name = str(training_config["scheduler"])
    if scheduler_name != "cosine":
        raise ValueError("Only the configured cosine scheduler is currently supported")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(training_config["min_learning_rate"]),
    )
    use_amp = device.type == "cuda" and bool(training_config["mixed_precision"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    positive_weight = compute_positive_weight(training_records).to(device)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    latest_path = destination / "latest.pt"
    best_path = destination / "best.pt"
    metrics_path = destination / "metrics.csv"
    curves_path = destination / "training_curves.png"
    if latest_path.exists() and not resume:
        raise FileExistsError(
            f"Checkpoint already exists at {latest_path}; use --resume to continue it"
        )
    if resume and not latest_path.is_file():
        raise FileNotFoundError(f"Cannot resume: latest checkpoint is missing at {latest_path}")

    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_auroc = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    if resume:
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint["model_name"] != model_name or checkpoint["fold"] != fold:
            raise ValueError("Resume checkpoint model/fold does not match this run")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"])
        best_auroc = float(checkpoint["best_auroc"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        data_generator.set_state(checkpoint["data_generator_state"])
        _restore_rng_state(checkpoint["rng_state"])
        LOGGER.info("Resuming %s fold %d at epoch %d", model_name, fold, start_epoch + 1)

    if start_epoch >= epochs:
        raise ValueError(f"Checkpoint already reached configured training.epochs={epochs}")

    loss_name = str(training_config["loss"])
    label_smoothing = float(training_config["label_smoothing"])
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("training.label_smoothing must be in [0, 1)")

    def train_batch(batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[float, int]:
        _set_model_training_mode(model, model_name, freeze_backbone)
        images, targets = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if label_smoothing:
            targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images).squeeze(1)
            loss = _compute_loss(
                logits,
                targets,
                loss_name=loss_name,
                positive_weight=positive_weight,
                focal_alpha=float(training_config["focal_alpha"]),
                focal_gamma=float(training_config["focal_gamma"]),
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(training_config["gradient_clip_norm"]),
        )
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach().item()), len(targets)

    def evaluate() -> tuple[float, float, float]:
        model.eval()
        loss_total = 0.0
        sample_total = 0
        all_targets = []
        all_probabilities = []
        with torch.inference_mode():
            for images, targets in validation_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(images).squeeze(1)
                    loss = _compute_loss(
                        logits,
                        targets,
                        loss_name=loss_name,
                        positive_weight=positive_weight,
                        focal_alpha=float(training_config["focal_alpha"]),
                        focal_gamma=float(training_config["focal_gamma"]),
                    )
                batch_size_actual = len(targets)
                loss_total += float(loss.item()) * batch_size_actual
                sample_total += batch_size_actual
                all_targets.extend(targets.int().cpu().tolist())
                all_probabilities.extend(torch.sigmoid(logits).float().cpu().tolist())
        probabilities = np.asarray(all_probabilities, dtype=np.float32)
        targets_array = np.asarray(all_targets, dtype=np.int64)
        predictions = probabilities >= 0.5
        return (
            loss_total / sample_total,
            float(roc_auc_score(targets_array, probabilities)),
            float(f1_score(targets_array, predictions, zero_division=0)),
        )

    warmup_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    warmup_losses: list[tuple[float, int]] = []
    warmup_times = []
    first_epoch_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    iterator = iter(train_loader)
    for _ in range(min(3, len(train_loader))):
        batch_start = time.perf_counter()
        batch = next(iterator)
        loss_value, batch_samples = train_batch(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        warmup_batches.append(batch)
        warmup_losses.append((loss_value, batch_samples))
        warmup_times.append(time.perf_counter() - batch_start)
    median_batch_seconds = float(np.median(warmup_times[1:] or warmup_times))
    remaining_batches = max(0, len(train_loader) - len(warmup_batches))
    estimated_train_seconds = sum(warmup_times) + median_batch_seconds * remaining_batches
    if device.type == "cuda":
        peak_memory_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info(device)
        free_memory_gib = free_memory_bytes / 1024**3
        total_memory_gib = total_memory_bytes / 1024**3
    else:
        peak_memory_gib = free_memory_gib = total_memory_gib = 0.0
    estimated_total_hours = estimated_train_seconds * (epochs - start_epoch) / 3600
    LOGGER.info(
        "Preflight %s fold %d: train estimate %.1f min/epoch (%.1f h remaining, validation excluded); "
        "peak %.2f GiB, free/total VRAM %.2f/%.2f GiB, batch %d, device %s",
        model_name,
        fold,
        estimated_train_seconds / 60,
        estimated_total_hours,
        peak_memory_gib,
        free_memory_gib,
        total_memory_gib,
        batch_size,
        device,
    )

    for epoch_index in range(start_epoch, epochs):
        epoch_start = first_epoch_start if epoch_index == start_epoch else time.perf_counter()
        learning_rate_summary = ";".join(
            f"{group.get('name', index)}={group['lr']:.8g}"
            for index, group in enumerate(optimizer.param_groups)
        )
        loss_sum = 0.0
        sample_count = 0
        batches = warmup_batches if epoch_index == start_epoch else []
        if epoch_index == start_epoch:
            batch_iterator = itertools.chain(batches, iterator)
        else:
            batch_iterator = iter(train_loader)
        for batch_index, batch in enumerate(batch_iterator):
            if epoch_index == start_epoch and batch_index < len(warmup_losses):
                batch_loss, batch_samples = warmup_losses[batch_index]
            else:
                batch_loss, batch_samples = train_batch(batch)
            loss_sum += batch_loss * batch_samples
            sample_count += batch_samples

        validation_loss, validation_auroc, validation_f1 = evaluate()
        scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_start
        metric_row = {
            "epoch": epoch_index + 1,
            "train_loss": loss_sum / sample_count,
            "validation_loss": validation_loss,
            "validation_auroc": validation_auroc,
            "validation_f1": validation_f1,
            "learning_rates": learning_rate_summary,
            "epoch_seconds": epoch_seconds,
        }
        history.append(metric_row)
        LOGGER.info(
            "epoch %d/%d train_loss=%.5f val_loss=%.5f val_auroc=%.5f val_f1=%.5f lr=%s time=%.1fs",
            epoch_index + 1,
            epochs,
            metric_row["train_loss"],
            validation_loss,
            validation_auroc,
            validation_f1,
            learning_rate_summary,
            epoch_seconds,
        )

        improved = validation_auroc > best_auroc + float(training_config["min_delta"])
        if improved:
            best_auroc = validation_auroc
            best_epoch = epoch_index + 1
            stale_epochs = 0
        else:
            stale_epochs += 1

        checkpoint_payload = {
            "model_name": model_name,
            "fold": fold,
            "epoch": epoch_index + 1,
            "best_epoch": best_epoch,
            "best_auroc": best_auroc,
            "stale_epochs": stale_epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
            "data_generator_state": data_generator.get_state(),
            "rng_state": _read_rng_state(),
            "config": config,
            "pretrained": pretrained,
            "freeze_backbone": freeze_backbone,
        }
        _atomic_torch_save(checkpoint_payload, latest_path)
        if improved:
            _atomic_torch_save(checkpoint_payload, best_path)
        _write_history(history, metrics_path)
        _save_training_curves(history, curves_path)

        if stale_epochs >= int(training_config["patience"]):
            LOGGER.info(
                "Early stopping at epoch %d; best validation AUROC %.5f at epoch %d",
                epoch_index + 1,
                best_auroc,
                best_epoch,
            )
            break

    return {
        "model_name": model_name,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_validation_auroc": best_auroc,
        "epochs_completed": history[-1]["epoch"],
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "metrics_csv": str(metrics_path),
        "curves": str(curves_path),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--config", type=Path, default=project_root / "config.yaml")
    parser.add_argument("--splits", type=Path, default=project_root / "outputs" / "splits.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, help="Override training.epochs for a bounded run")
    parser.add_argument("--batch-size", type=int, help="Override the configured model batch size")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with arguments.config.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config["training"] = dict(config["training"])
    if arguments.epochs is not None:
        config["training"]["epochs"] = arguments.epochs
    if arguments.batch_size is not None:
        config["training"]["batch_sizes"] = {
            **config["training"].get("batch_sizes", {}),
            arguments.model: arguments.batch_size,
        }
    fold = int(config["phase4"]["fold"] if arguments.fold is None else arguments.fold)
    pretrained = bool(config["training"]["pretrained"] if arguments.pretrained is None else arguments.pretrained)
    if arguments.model == "custom_cnn":
        pretrained = False
    run_directory = arguments.output_dir or (
        project_root / "outputs" / "checkpoints" / f"{arguments.model}_fold_{fold}"
    )
    result = train_fold(
        model_name=arguments.model,
        fold=fold,
        config=config,
        splits_path=arguments.splits,
        output_dir=run_directory,
        pretrained=pretrained,
        freeze_backbone=arguments.freeze_backbone,
        resume=arguments.resume,
    )
    LOGGER.info(
        "Training complete: best AUROC=%.5f at epoch %d; checkpoint=%s",
        result["best_validation_auroc"],
        result["best_epoch"],
        result["best_checkpoint"],
    )


if __name__ == "__main__":
    main()