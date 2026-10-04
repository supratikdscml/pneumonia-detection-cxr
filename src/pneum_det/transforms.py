"""Image transforms for grayscale chest X-rays."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    Grayscale,
    InterpolationMode,
    Normalize,
    RandomAffine,
    RandomResizedCrop,
    RandomRotation,
    Resize,
    ToTensor,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class ApplyClahe:
    """Apply CLAHE to a PIL image using OpenCV."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "CLAHE is enabled but opencv-python-headless is not installed"
            ) from exc

        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        grayscale = np.asarray(image.convert("L"))
        enhanced = self._clahe.apply(grayscale)
        return Image.fromarray(enhanced, mode="L")


def build_image_transforms(
    config: Mapping[str, Any],
) -> tuple[Compose, Compose]:
    """Build training and evaluation transforms from preprocessing config."""
    image_size = int(config.get("image_size", 224))
    if image_size not in {224, 384}:
        raise ValueError("image_size must be 224 or 384")

    clahe_config = config.get("clahe", {})
    augmentation = config.get("augmentation", {})
    if not isinstance(clahe_config, Mapping) or not isinstance(augmentation, Mapping):
        raise TypeError("clahe and augmentation settings must be mappings")

    grayscale = Grayscale(num_output_channels=1)
    to_rgb_grayscale = Grayscale(num_output_channels=3)
    clahe_transform = []
    if bool(clahe_config.get("enabled", False)):
        tile_grid_size = tuple(clahe_config.get("tile_grid_size", (8, 8)))
        if len(tile_grid_size) != 2:
            raise ValueError("clahe.tile_grid_size must contain two integers")
        clahe_transform.append(
            ApplyClahe(
                clip_limit=float(clahe_config.get("clip_limit", 2.0)),
                tile_grid_size=(int(tile_grid_size[0]), int(tile_grid_size[1])),
            )
        )

    normalization = Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    train_transform = Compose(
        [
            grayscale,
            *clahe_transform,
            RandomResizedCrop(
                size=(image_size, image_size),
                scale=tuple(augmentation.get("crop_scale", (0.85, 1.0))),
                ratio=tuple(augmentation.get("crop_ratio", (0.9, 1.8))),
                interpolation=InterpolationMode.BILINEAR,
            ),
            RandomRotation(
                degrees=float(augmentation.get("rotation_degrees", 10.0)),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
            RandomAffine(
                degrees=0,
                translate=(
                    float(augmentation.get("translation_fraction", 0.04)),
                    float(augmentation.get("translation_fraction", 0.04)),
                ),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
            ColorJitter(
                brightness=float(augmentation.get("brightness", 0.08)),
                contrast=float(augmentation.get("contrast", 0.08)),
            ),
            to_rgb_grayscale,
            ToTensor(),
            normalization,
        ]
    )
    eval_transform = Compose(
        [
            grayscale,
            *clahe_transform,
            Resize(
                size=(image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
            ),
            to_rgb_grayscale,
            ToTensor(),
            normalization,
        ]
    )
    return train_transform, eval_transform