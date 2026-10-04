"""Shared configuration, seeding, and optional experiment tracking."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

LOGGER = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the project YAML config."""
    with Path(path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a mapping")
    return config


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a consistent log format for CLI modules."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def maybe_init_tracker(config: Mapping[str, Any]) -> str:
    """Initialize W&B or MLflow when enabled. Returns the active backend name."""
    tracking = config.get("tracking", {})
    if not isinstance(tracking, Mapping) or not bool(tracking.get("enabled", False)):
        LOGGER.info("Experiment tracking is disabled")
        return "disabled"
    backend = str(tracking.get("backend", "wandb")).lower()
    if backend == "wandb":
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("tracking.backend=wandb but wandb is not installed") from exc
        wandb.init(project=str(tracking.get("project", "pneumonia-dl")), config=dict(config))
        return "wandb"
    if backend == "mlflow":
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("tracking.backend=mlflow but mlflow is not installed") from exc
        mlflow.set_experiment(str(tracking.get("project", "pneumonia-dl")))
        mlflow.start_run()
        return "mlflow"
    raise ValueError(f"Unsupported tracking backend: {backend}")
