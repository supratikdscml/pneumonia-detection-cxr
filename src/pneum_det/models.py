"""Image-classification and optional lung-segmentation models for chest X-rays."""

from collections.abc import Callable

import torch
from torch import nn
from torchvision import models as torchvision_models

MODEL_NAMES = (
    "custom_cnn",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_b_16",
    "swin_t",
)

THREE_CLASS_NAMES = ("normal", "bacteria", "virus")

_TORCHVISION_MODELS: dict[str, tuple[Callable[..., nn.Module], object, str]] = {
    "resnet50": (torchvision_models.resnet50, torchvision_models.ResNet50_Weights, "fc"),
    "densenet121": (
        torchvision_models.densenet121,
        torchvision_models.DenseNet121_Weights,
        "classifier",
    ),
    "efficientnet_b0": (
        torchvision_models.efficientnet_b0,
        torchvision_models.EfficientNet_B0_Weights,
        "classifier",
    ),
    "convnext_tiny": (
        torchvision_models.convnext_tiny,
        torchvision_models.ConvNeXt_Tiny_Weights,
        "classifier",
    ),
    "vit_b_16": (
        torchvision_models.vit_b_16,
        torchvision_models.ViT_B_16_Weights,
        "heads",
    ),
    "swin_t": (torchvision_models.swin_t, torchvision_models.Swin_T_Weights, "head"),
}


class DoubleConv(nn.Module):
    """Two convolution-normalization-ReLU blocks used by the lung U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class LungUNet(nn.Module):
    """Compact U-Net that predicts a single lung-field logit map."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        self.down1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skip1 = self.down1(inputs)
        skip2 = self.down2(self.pool1(skip1))
        skip3 = self.down3(self.pool2(skip2))
        encoded = self.bottleneck(self.pool3(skip3))
        decoded = self.conv3(_crop_and_cat(self.up3(encoded), skip3))
        decoded = self.conv2(_crop_and_cat(self.up2(decoded), skip2))
        decoded = self.conv1(_crop_and_cat(self.up1(decoded), skip1))
        return self.head(decoded)


def _crop_and_cat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    min_height = min(upsampled.shape[-2], skip.shape[-2])
    min_width = min(upsampled.shape[-1], skip.shape[-1])
    return torch.cat(
        (upsampled[..., :min_height, :min_width], skip[..., :min_height, :min_width]),
        dim=1,
    )


def _build_custom_cnn(dropout: float, num_classes: int) -> nn.Module:
    blocks: list[nn.Module] = []
    in_channels = 3
    for out_channels in (32, 64, 128, 256):
        blocks.extend(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout2d(p=dropout / 2),
            ]
        )
        in_channels = out_channels
    return nn.Sequential(
        *blocks,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )


def _replace_classifier(model: nn.Module, model_name: str, dropout: float, num_classes: int) -> nn.Module:
    if model_name == "resnet50":
        input_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(input_features, num_classes))
    elif model_name == "densenet121":
        input_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(input_features, num_classes))
    elif model_name == "efficientnet_b0":
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_features, num_classes)
    elif model_name == "convnext_tiny":
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_features, num_classes)
    elif model_name == "vit_b_16":
        input_features = model.heads.head.in_features
        model.heads.head = nn.Linear(input_features, num_classes)
    elif model_name == "swin_t":
        input_features = model.head.in_features
        model.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(input_features, num_classes))
    else:
        raise ValueError(f"Unsupported torchvision model: {model_name}")
    return model


def _get_classifier(model: nn.Module, model_name: str) -> nn.Module:
    if model_name in {"resnet50", "densenet121"}:
        return getattr(model, "fc" if model_name == "resnet50" else "classifier")
    if model_name == "efficientnet_b0":
        return model.classifier[-1]
    if model_name == "convnext_tiny":
        return model.classifier[-1]
    if model_name == "vit_b_16":
        return model.heads.head
    if model_name == "swin_t":
        return model.head
    raise ValueError(f"No replaceable classifier for model: {model_name}")


def create_lung_unet(base_channels: int = 16) -> LungUNet:
    """Create an untrained lung-field U-Net (needs pixel masks before training)."""
    return LungUNet(in_channels=1, base_channels=base_channels)


def create_model(
    model_name: str,
    *,
    pretrained: bool = False,
    freeze_backbone: bool = False,
    dropout: float = 0.2,
    num_classes: int = 1,
) -> nn.Module:
    """Create a classifier that returns ``num_classes`` logits per image.

    ``num_classes=1`` is the binary pneumonia head. Torchvision ImageNet weights
    are used when ``pretrained`` is true. Requesting them may download weights
    that are not already cached.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")
    if model_name == "custom_cnn":
        if pretrained:
            raise ValueError("custom_cnn does not have pretrained weights")
        if freeze_backbone:
            raise ValueError("custom_cnn has no pretrained backbone to freeze")
        return _build_custom_cnn(dropout, num_classes)

    constructor, weights_enum, _ = _TORCHVISION_MODELS[model_name]
    weights = weights_enum.DEFAULT if pretrained else None
    model = constructor(weights=weights)
    model = _replace_classifier(model, model_name, dropout, num_classes)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in _get_classifier(model, model_name).parameters():
            parameter.requires_grad = True
    return model
