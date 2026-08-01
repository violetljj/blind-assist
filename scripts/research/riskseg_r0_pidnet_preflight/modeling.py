from __future__ import annotations

import hashlib
import importlib
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


INPUT_WIDTH = 512
INPUT_HEIGHT = 288
NUM_CLASSES = 4
CLASS_ORDER = (
    "walkable",
    "blocking_obstacle",
    "boundary_level_change",
    "unknown_nonwalkable",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


def official_repo_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _load_official_pidnet_class(repo: Path) -> type[nn.Module]:
    models_dir = repo / "models"
    init_path = models_dir / "__init__.py"
    if not init_path.is_file():
        raise FileNotFoundError(f"official PIDNet models package missing: {init_path}")

    package_name = "models"
    for loaded_name in tuple(sys.modules):
        if loaded_name == package_name or loaded_name.startswith(package_name + "."):
            del sys.modules[loaded_name]
    sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module("models.pidnet")
        return module.PIDNet
    finally:
        sys.path.remove(str(repo))


def build_pidnet_s(
    *,
    official_repo: Path,
    augment: bool,
) -> nn.Module:
    pidnet_class = _load_official_pidnet_class(official_repo)
    return pidnet_class(
        m=2,
        n=3,
        num_classes=NUM_CLASSES,
        planes=32,
        ppm_planes=96,
        head_planes=128,
        augment=augment,
    )


def load_imagenet_backbone(
    *,
    model: nn.Module,
    checkpoint_path: Path,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("ImageNet checkpoint must contain a state_dict mapping")
    pretrained_state = payload["state_dict"]
    model_state = model.state_dict()
    matched = {
        key: value
        for key, value in pretrained_state.items()
        if key in model_state and value.shape == model_state[key].shape
    }
    if not matched:
        raise ValueError("ImageNet checkpoint has no shape-compatible PIDNet-S tensors")
    model_state.update(matched)
    model.load_state_dict(model_state, strict=True)
    return {
        "checkpoint_top_level_keys": sorted(payload),
        "checkpoint_tensor_count": len(pretrained_state),
        "model_tensor_count": len(model_state),
        "matched_tensor_count": len(matched),
        "matched_parameter_count": sum(value.numel() for value in matched.values()),
        "model_parameter_count": sum(value.numel() for value in model_state.values()),
        "unmatched_model_keys": sorted(set(model_state) - set(matched)),
        "unused_checkpoint_keys": sorted(set(pretrained_state) - set(matched)),
    }


class DeploymentWrapper(nn.Module):
    """Frozen deployment surface: normalized NCHW RGB to full-size NCHW logits."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_rgb_normalized: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_rgb_normalized)
        return F.interpolate(
            logits,
            size=(INPUT_HEIGHT, INPUT_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
