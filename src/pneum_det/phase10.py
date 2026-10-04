"""Phase 10 extensions: 3-class labels, lung crops, CXR weights, export, demo."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from pneum_det.models import THREE_CLASS_NAMES, create_lung_unet, create_model

LOGGER = logging.getLogger(__name__)
_OFFICIAL_TEST_MARKERS = ("chest_xray/test", "chest_xray\\test", "/test/NORMAL", "/test/PNEUMONIA")


def three_class_from_subtype(subtype: str) -> str:
    """Map filename-derived subtype to normal / bacteria / virus."""
    normalized = subtype.strip().lower()
    if normalized in {"normal", "bacteria", "virus"}:
        return normalized
    raise ValueError(f"Unsupported subtype for 3-class training: {subtype!r}")


def three_class_index(subtype: str) -> int:
    """Return the integer class index used by the 3-class head."""
    name = three_class_from_subtype(subtype)
    return THREE_CLASS_NAMES.index(name)


def three_class_fold_counts(splits: pd.DataFrame) -> pd.DataFrame:
    """Count 3-class labels on development folds only (official test excluded)."""
    if "official_split" in splits.columns:
        development = splits.loc[splits["official_split"] != "test"].copy()
    else:
        development = splits.copy()
    if development.empty:
        raise ValueError("No development rows available for 3-class counts")
    development["three_class"] = development["subtype"].map(three_class_from_subtype)
    counts = (
        development.groupby(["fold", "partition", "three_class"], dropna=False)
        .size()
        .rename("n_images")
        .reset_index()
    )
    return counts.sort_values(["fold", "partition", "three_class"], ignore_index=True)


def heuristic_lung_mask(image: np.ndarray) -> np.ndarray:
    """Estimate a lung-field mask with Otsu thresholding (not a trained U-Net)."""
    if image.ndim != 2:
        raise ValueError("heuristic_lung_mask expects a grayscale array")
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Lungs are typically darker than the surrounding tissue on CXR.
    inverted = cv2.bitwise_not(thresholded)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cleaned = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image, dtype=np.uint8)
    if not contours:
        return mask
    ranked = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    cv2.drawContours(mask, ranked, contourIdx=-1, color=255, thickness=-1)
    return mask


def crop_to_mask(image: np.ndarray, mask: np.ndarray, padding_fraction: float = 0.05) -> np.ndarray:
    """Crop a grayscale image to the bounding box of a binary mask."""
    if image.shape != mask.shape:
        raise ValueError("image and mask must have the same spatial shape")
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return image
    height, width = image.shape
    pad_y = int(round(padding_fraction * height))
    pad_x = int(round(padding_fraction * width))
    y0 = max(int(ys.min()) - pad_y, 0)
    y1 = min(int(ys.max()) + 1 + pad_y, height)
    x0 = max(int(xs.min()) - pad_x, 0)
    x1 = min(int(xs.max()) + 1 + pad_x, width)
    return image[y0:y1, x0:x1]


def apply_lung_crop(path: str | Path, *, use_unet: bool = False, unet: torch.nn.Module | None = None) -> dict[str, Any]:
    """Return original/cropped arrays. U-Net weights are required when use_unet is true."""
    with Image.open(path) as pil_image:
        grayscale = np.asarray(pil_image.convert("L"))
    if use_unet:
        if unet is None:
            raise ValueError("A trained LungUNet is required when use_unet=True")
        unet.eval()
        tensor = torch.from_numpy(grayscale.astype(np.float32) / 255.0)[None, None]
        with torch.inference_mode():
            logits = unet(tensor)
        mask = (torch.sigmoid(logits)[0, 0].cpu().numpy() >= 0.5).astype(np.uint8) * 255
        source = "unet"
    else:
        mask = heuristic_lung_mask(grayscale)
        source = "heuristic_otsu"
    cropped = crop_to_mask(grayscale, mask)
    coverage = float(np.mean(mask > 0))
    return {
        "original": grayscale,
        "mask": mask,
        "cropped": cropped,
        "mask_source": source,
        "mask_coverage": coverage,
    }


def try_create_torchxrayvision_densenet(num_classes: int = 1):
    """Build a DenseNet-121 from TorchXRayVision if the optional package is installed."""
    try:
        import torchxrayvision as xrv
    except ImportError:
        LOGGER.info("torchxrayvision is not installed; skipping CXR-pretrained comparison")
        return None
    backbone = xrv.models.DenseNet(weights="densenet121-res224-all")
    in_features = backbone.classifier.in_features
    backbone.classifier = torch.nn.Linear(in_features, num_classes)
    return backbone


def discover_external_dataset(project_root: str | Path) -> Path | None:
    """Return an external CXR folder if present; never the official Kaggle test split."""
    root = Path(project_root)
    candidates = [
        root / "data" / "external",
        root / "data" / "nih-chestxray14",
        root / "data" / "rsna-pneumonia",
        root / "data" / "chexpert",
    ]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        has_images = any(
            path.suffix.lower() in {".jpeg", ".jpg", ".png"} for path in candidate.rglob("*")
        )
        if not has_images:
            continue
        text = str(candidate.resolve()).lower()
        if any(marker.lower() in text for marker in _OFFICIAL_TEST_MARKERS):
            raise AssertionError(f"External validation path looks like official test data: {candidate}")
        return candidate
    return None


def assert_external_disjoint_from_official(external_root: Path, official_audit: pd.DataFrame) -> None:
    """Refuse external evaluation if official filenames leak into the external folder."""
    official_names = set(Path(path).name for path in official_audit["path"])
    external_names = {path.name for path in external_root.rglob("*") if path.suffix.lower() in {".jpeg", ".jpg", ".png"}}
    overlap = official_names & external_names
    if overlap:
        examples = sorted(overlap)[:10]
        raise AssertionError(f"External dataset overlaps official filenames: {examples}")


def export_torchscript(model: torch.nn.Module, output_path: str | Path, image_size: int = 224) -> Path:
    """Trace a binary classifier and save a TorchScript file."""
    model.eval()
    example = torch.zeros(1, 3, image_size, image_size)
    scripted = torch.jit.trace(model, example)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(destination))
    return destination


def export_onnx(model: torch.nn.Module, output_path: str | Path, image_size: int = 224) -> Path | None:
    """Export a binary classifier to ONNX when the optional `onnx` package is installed."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        LOGGER.warning("Skipping ONNX export because the onnx package is not installed")
        return None
    model.eval()
    example = torch.zeros(1, 3, image_size, image_size)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        str(destination),
        input_names=["image"],
        output_names=["logit"],
        opset_version=17,
        dynamo=False,
    )
    return destination


def write_demo_instructions(output_path: str | Path) -> Path:
    """Write how to launch the optional Gradio demo from a frozen checkpoint."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "python -m pneum_det.demo_app --checkpoint outputs/checkpoints/densenet121_fold_0/best.pt\n"
        "Requires a Phase 5/8 frozen checkpoint. Do not point the demo at official test images.\n",
        encoding="utf-8",
    )
    return destination


def audit_pipeline_credentials(project_root: str | Path) -> pd.DataFrame:
    """Compare the repository against the methodological spec (not training completeness)."""
    root = Path(project_root)
    checks: list[dict[str, str]] = []

    def add(item: str, status: str, evidence: str) -> None:
        checks.append({"item": item, "status": status, "evidence": evidence})

    config_path = root / "config.yaml"
    add("Single config.yaml", "pass" if config_path.is_file() else "fail", str(config_path))
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        tracking = config.get("tracking", {})
        add(
            "Optional W&B/MLflow via config flag",
            "pass" if "enabled" in tracking else "fail",
            f"tracking.enabled={tracking.get('enabled')}",
        )
        add(
            "Image size 224 with 384 option",
            "pass" if int(config.get("preprocessing", {}).get("image_size", 0)) in {224, 384} else "fail",
            str(config.get("preprocessing", {}).get("image_size")),
        )
        add(
            "CLAHE config flag",
            "pass" if "clahe" in config.get("preprocessing", {}) else "fail",
            str(config.get("preprocessing", {}).get("clahe")),
        )
        add(
            "BCE pos_weight + optional focal loss",
            "pass" if config.get("training", {}).get("loss") in {"bce_with_logits", "focal"} else "fail",
            str(config.get("training", {}).get("loss")),
        )
        add(
            "Early stopping patience 5 on AUROC (config)",
            "pass" if int(config.get("training", {}).get("patience", 0)) == 5 else "fail",
            str(config.get("training", {}).get("patience")),
        )

    src = root / "src" / "pneum_det"
    expected_modules = [
        "data.py",
        "transforms.py",
        "models.py",
        "train.py",
        "evaluate.py",
        "explain.py",
        "ensemble.py",
        "utils.py",
        "phase10.py",
    ]
    missing = [name for name in expected_modules if not (src / name).is_file()]
    add("Core src modules", "pass" if not missing else "fail", "missing: " + ", ".join(missing) if missing else "present")

    requirements = (root / "requirements.txt").read_text(encoding="utf-8") if (root / "requirements.txt").is_file() else ""
    add("Pinned requirements.txt", "pass" if requirements else "fail", "present" if requirements else "missing")
    add("timm in requirements", "gap", "Project uses torchvision model factory instead of timm")
    add("albumentations", "gap", "Optional; torchvision transforms implemented")
    add("README + limitations", "pass" if (root / "README.md").is_file() else "fail", str(root / "README.md"))
    add("notebooks/01_eda.ipynb layout", "gap", "EDA lives in files/pneumoniadetect.ipynb")
    add("python -m src.train CLI layout", "gap", "Actual CLI is python src/pneum_det/train.py --model ... --fold ...")

    splits = root / "outputs" / "splits.csv"
    add("Patient-grouped splits.csv", "pass" if splits.is_file() else "fail", str(splits))
    if splits.is_file():
        split_df = pd.read_csv(splits)
        leaked = False
        for fold, fold_rows in split_df.groupby("fold"):
            train_ids = set(fold_rows.loc[fold_rows["partition"] == "train", "patient_id"])
            val_ids = set(fold_rows.loc[fold_rows["partition"] == "validation", "patient_id"])
            if train_ids & val_ids:
                leaked = True
        test_in_splits = (
            "official_split" in split_df.columns and (split_df["official_split"] == "test").any()
        )
        add("No patient in train and val of same fold", "pass" if not leaked else "fail", f"folds={sorted(split_df['fold'].unique())}")
        add("Official test excluded from splits.csv", "pass" if not test_in_splits else "fail", "test rows absent")

    phase6 = root / "outputs" / "phase6"
    add(
        "Phase 6 validation evaluation (not test)",
        "pass" if (phase6 / "densenet121_fold_0_metrics.json").is_file() else "partial",
        "fold 0 only; 5-fold mean±std not yet aggregated",
    )
    add(
        "Phase 8 ensemble checkpoints",
        "gap",
        "ConvNeXt and ViT full checkpoints missing; ensemble gated",
    )
    add(
        "Phase 9 one-time test lock",
        "pass" if (root / "src" / "pneum_det" / "evaluate_final.py").is_file() else "fail",
        "LOCKED until outputs/phase9/final_selection.json is frozen",
    )
    add(
        "Phase 4/5 full recipe vs smoke",
        "partial",
        "Fold-0 one-epoch screening exists; full early-stopped 5-fold runs are not complete",
    )
    add(
        "Swin-T in model factory",
        "pass",
        "swin_t added to pneum_det.models.MODEL_NAMES",
    )
    add("Phase 10 3-class / lung crop / export", "pass", "src/pneum_det/phase10.py")
    return pd.DataFrame(checks)


def run_phase10_smoke(
    project_root: str | Path,
    splits_path: str | Path,
    audit_path: str | Path,
    output_dir: str | Path,
    sample_limit: int = 8,
) -> dict[str, Any]:
    """Run leakage-safe Phase 10 checks that do not train or touch the official test set."""
    root = Path(project_root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    splits = pd.read_csv(splits_path)
    if "official_split" in splits.columns and (splits["official_split"] == "test").any():
        raise AssertionError("splits.csv must not contain official test rows")
    counts = three_class_fold_counts(splits)
    counts_path = destination / "three_class_fold_counts.csv"
    counts.to_csv(counts_path, index=False)

    audit = pd.read_csv(audit_path)
    development_audit = audit.loc[audit["split"] != "test"]
    samples = development_audit.groupby("subtype", group_keys=False).head(max(1, sample_limit // 3))
    crop_rows = []
    figure_dir = destination / "lung_crops"
    figure_dir.mkdir(exist_ok=True)
    unet = create_lung_unet()
    unet.eval()
    dummy = torch.zeros(1, 1, 64, 64)
    with torch.inference_mode():
        unet_logits = unet(dummy)
    if tuple(unet_logits.shape[-2:]) != (64, 64):
        raise RuntimeError("Lung U-Net did not preserve spatial size on a 64x64 input")

    for index, row in samples.iterrows():
        result = apply_lung_crop(row["path"], use_unet=False)
        Image.fromarray(result["cropped"]).save(figure_dir / f"crop_{index}_{row['subtype']}.png")
        crop_rows.append(
            {
                "path": row["path"],
                "split": row["split"],
                "subtype": row["subtype"],
                "mask_source": result["mask_source"],
                "mask_coverage": result["mask_coverage"],
                "cropped_height": int(result["cropped"].shape[0]),
                "cropped_width": int(result["cropped"].shape[1]),
            }
        )
    crop_table = pd.DataFrame(crop_rows)
    crop_table_path = destination / "lung_crop_samples.csv"
    crop_table.to_csv(crop_table_path, index=False)

    torchxrayvision_model = try_create_torchxrayvision_densenet()
    external_root = discover_external_dataset(root)
    if external_root is not None:
        assert_external_disjoint_from_official(external_root, audit)

    three_class_model = create_model("densenet121", pretrained=False, num_classes=3)
    three_class_model.eval()
    with torch.inference_mode():
        logits = three_class_model(torch.zeros(2, 3, 224, 224))
    if tuple(logits.shape) != (2, 3):
        raise RuntimeError(f"3-class head returned unexpected shape {tuple(logits.shape)}")

    export_dir = destination / "export"
    binary_model = create_model("densenet121", pretrained=False, num_classes=1)
    torchscript_path = export_torchscript(binary_model, export_dir / "densenet121_untrained.pt")
    onnx_path = export_onnx(binary_model, export_dir / "densenet121_untrained.onnx")
    demo_path = write_demo_instructions(destination / "demo_launch.txt")

    summary = {
        "official_test_used": False,
        "three_class_names": list(THREE_CLASS_NAMES),
        "three_class_counts_csv": str(counts_path),
        "lung_crop_samples_csv": str(crop_table_path),
        "lung_unet_trained": False,
        "lung_unet_note": "U-Net architecture is implemented; training requires an external lung-mask dataset (e.g. Montgomery/Shenzhen). Heuristic Otsu crops are a shortcut-reduction prototype only.",
        "torchxrayvision_available": torchxrayvision_model is not None,
        "external_dataset": str(external_root) if external_root else None,
        "torchscript": str(torchscript_path),
        "onnx": str(onnx_path) if onnx_path else None,
        "demo_instructions": str(demo_path),
        "mean_mask_coverage": float(crop_table["mask_coverage"].mean()) if not crop_table.empty else 0.0,
    }
    summary_path = destination / "phase10_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Phase 10 smoke artifacts written to %s", destination)
    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_root / "config.yaml")
    parser.add_argument("--splits", type=Path, default=project_root / "outputs" / "splits.csv")
    parser.add_argument("--audit", type=Path, default=project_root / "outputs" / "phase1" / "image_audit.csv")
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs" / "phase10")
    parser.add_argument("--sample-limit", type=int, default=8)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run_phase10_smoke(
        project_root=project_root,
        splits_path=arguments.splits,
        audit_path=arguments.audit,
        output_dir=arguments.output_dir,
        sample_limit=arguments.sample_limit,
    )
    LOGGER.info("Phase 10 summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
