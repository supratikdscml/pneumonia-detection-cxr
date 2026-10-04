"""Optional Gradio demo for a frozen binary checkpoint. Never uses the test set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from pneum_det.models import create_model
from pneum_det.transforms import build_image_transforms


def load_frozen_classifier(checkpoint_path: Path, config: dict) -> tuple[torch.nn.Module, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = checkpoint["model_name"]
    model = create_model(model_name, pretrained=False, num_classes=1)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    _, eval_transform = build_image_transforms(config["preprocessing"])
    return model, eval_transform


def predict_probability(model: torch.nn.Module, transform, image: Image.Image) -> float:
    tensor = transform(image.convert("L")).unsqueeze(0)
    with torch.inference_mode():
        logit = model(tensor)
    return float(torch.sigmoid(logit).item())


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=project_root / "config.yaml")
    parser.add_argument("--share", action="store_true")
    arguments = parser.parse_args()
    import yaml

    with arguments.config.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    model, transform = load_frozen_classifier(arguments.checkpoint, config)

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install gradio to launch the demo: pip install gradio") from exc

    def infer(image: Image.Image) -> dict[str, float]:
        probability = predict_probability(model, transform, image)
        return {
            "pneumonia_probability": probability,
            "normal_probability": 1.0 - probability,
        }

    interface = gr.Interface(
        fn=infer,
        inputs=gr.Image(type="pil", label="Chest X-ray"),
        outputs=gr.Label(num_top_classes=2),
        title="Pneumonia detector (research prototype)",
        description="Not a clinical device. Thresholds must be chosen on validation data only.",
    )
    interface.launch(share=arguments.share)


if __name__ == "__main__":
    main()
