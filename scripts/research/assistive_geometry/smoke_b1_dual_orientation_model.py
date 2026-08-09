#!/usr/bin/env python3
"""Run frozen-checkpoint dual-orientation forward/backward B1 implementation smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.assistive_geometry_model import (  # noqa: E402
    DepthArtAssistiveGeometry,
    compute_b1_losses,
)
from scripts.research.assistive_geometry.download_b0_arkitscenes_assets import (  # noqa: E402
    load_json,
    require,
    sha256_file,
    write_json_exclusive,
)


ACTIVE_A4 = (
    "masked_log_depth",
    "valid_neighbor_log_gradient",
    "ground_bce",
    "ground_plane_depth",
    "clearance_huber",
    "occupancy_bce",
    "false_clear_extra",
    "confidence_bce",
)


def synthetic_targets(height: int, width: int, device: torch.device) -> dict[str, torch.Tensor]:
    depth = torch.full((1, 1, height, width), 2.0, device=device)
    valid = torch.ones_like(depth, dtype=torch.bool)
    ground = torch.zeros_like(depth)
    ground[..., height // 2 :, :] = 1.0
    intrinsics = torch.tensor(
        [[[420.0, 0.0, width / 2.0], [0.0, 420.0, height / 2.0], [0.0, 0.0, 1.0]]],
        device=device,
    )
    return {
        "dense_depth_m": depth,
        "depth_valid": valid,
        "ground_probability": ground,
        "ground_label_valid": valid,
        "ground_plane_valid": torch.tensor([False], device=device),
        "camera_height_m": torch.tensor([1.5], device=device),
        "up_camera": torch.tensor([[0.0, -1.0, 0.0]], device=device),
        "intrinsics_tensor": intrinsics,
        "clearance_m": torch.tensor([[0.8, 1.2, 1.8]], device=device),
        "clearance_valid": torch.ones((1, 3), dtype=torch.bool, device=device),
        "occupancy": torch.tensor([[[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]], device=device),
        "occupancy_valid": torch.ones((1, 3, 3), dtype=torch.bool, device=device),
    }


def gradient_summary(model: torch.nn.Module) -> dict[str, Any]:
    encoder_nonzero = 0
    head_nonzero = 0
    finite = True
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        finite &= bool(torch.isfinite(parameter.grad).all().item())
        nonzero = bool(torch.count_nonzero(parameter.grad).item())
        if name.startswith("assistive_heads."):
            head_nonzero += int(nonzero)
        else:
            encoder_nonzero += int(nonzero)
    return {
        "all_finite": finite,
        "encoder_or_depth_parameters_with_nonzero_grad": encoder_nonzero,
        "assistive_head_parameters_with_nonzero_grad": head_nonzero,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-protocol", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.implementation_protocol.resolve()
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == "blindassist_assistive_geometry_b1_implementation_lock_protocol_v1", "implementation protocol schema drift")
    require(protocol["model_smoke"]["sha256"] == sha256_file(Path(__file__)), "model smoke SHA drift")
    checkpoint = args.checkpoint.resolve()
    require(checkpoint.is_file() and sha256_file(checkpoint) == protocol["checkpoint"]["sha256"], "checkpoint binding drift")
    source = args.source.resolve()

    deployment = Path(__file__).resolve().parents[1] / "hftf/deployment/depthart"
    sys.path.insert(0, str(deployment))
    from export_depthart_camera_external import install_timm_compat

    install_timm_compat()
    sys.path.insert(0, str(source / "metric"))
    sys.path.insert(0, str(source / "deploy" / "shared"))
    sys.path.insert(0, str(source / "deploy" / "shared" / "selective_scan"))
    from depthart_selective_scan import install_depthart
    from model import load_model
    from network import tvimblock

    torch.manual_seed(20260809)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "full DepthART backward smoke requires CUDA")
    base = load_model(checkpoint, "S", "indoor", str(device))
    install_depthart(tvimblock)
    model = DepthArtAssistiveGeometry(base).to(device).eval()

    rows: list[dict[str, Any]] = []
    for family, height, width in (("portrait", 608, 448), ("landscape", 448, 608)):
        model.zero_grad(set_to_none=True)
        image = torch.randn(1, 3, height, width, device=device)
        intrinsics = torch.tensor(
            [[[420.0, 0.0, width / 2.0], [0.0, 420.0, height / 2.0], [0.0, 0.0, 1.0]]],
            device=device,
        )
        outputs = model(image, intrinsics)
        losses = compute_b1_losses(outputs, synthetic_targets(height, width, device), ACTIVE_A4)
        require(bool(torch.isfinite(losses["total"]).item()), f"non-finite {family} loss")
        losses["total"].backward()
        gradients = gradient_summary(model)
        require(gradients["all_finite"], f"non-finite {family} gradients")
        require(gradients["encoder_or_depth_parameters_with_nonzero_grad"] > 0, f"no {family} encoder gradient")
        require(gradients["assistive_head_parameters_with_nonzero_grad"] > 0, f"no {family} head gradient")
        rows.append({
            "orientation_family": family,
            "image_shape": list(image.shape),
            "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
            "total_loss": float(losses["total"].detach().item()),
            "gradients": gradients,
        })
        del image, intrinsics, outputs, losses
        torch.cuda.empty_cache()

    receipt = {
        "schema": "blindassist_assistive_geometry_b1_dual_orientation_model_smoke_v1",
        "protocol_sha256": sha256_file(protocol_path),
        "producer_sha256": protocol["model_smoke"]["sha256"],
        "model_sha256": protocol["model"]["sha256"],
        "checkpoint": {"path": str(checkpoint), "bytes": checkpoint.stat().st_size, "sha256": protocol["checkpoint"]["sha256"]},
        "source": str(source),
        "device": str(device),
        "torch_version": torch.__version__,
        "tf32_disabled": True,
        "active_losses": list(ACTIVE_A4),
        "shapes": rows,
        "development_or_confirmation_outcome_opened": False,
        "formal_training_started": False,
        "terminal": "B1_DUAL_ORIENTATION_MODEL_FORWARD_BACKWARD_SMOKE_PASS",
        "authority": "Synthetic implementation smoke only; no task quality, Development, Confirmation, deployment, product or safety authority.",
    }
    write_json_exclusive(args.output.resolve(), receipt)
    print(json.dumps({"terminal": receipt["terminal"], "shapes": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
