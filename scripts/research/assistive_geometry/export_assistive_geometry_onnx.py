#!/usr/bin/env python3
"""Export a selected Assistive Geometry checkpoint with host camera prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import onnx
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.assistive_geometry_model import (  # noqa: E402
    DepthArtAssistiveGeometry,
)
from scripts.research.assistive_geometry.download_b0_arkitscenes_assets import (  # noqa: E402
    require,
    sha256_file,
)
from scripts.research.hftf.deployment.depthart.export_depthart_camera_external import (  # noqa: E402
    install_timm_compat,
)


OUTPUT_NAMES = (
    "dense_depth_m",
    "ground_logits",
    "clearance_m",
    "occupancy_logits",
    "confidence_logits",
)


class ExternalCameraAssistiveGeometry(torch.nn.Module):
    def __init__(self, model: DepthArtAssistiveGeometry) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        image: torch.Tensor,
        cam4: torch.Tensor,
        cam8: torch.Tensor,
        cam16: torch.Tensor,
        cam32: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        metric = self.model.metric_depthart
        cameras = [cam4, cam8, cam16, cam32]
        features = metric.pretrained.forward_with_adapters(
            image,
            adapters=[metric.daa1, metric.daa2, metric.daa3, metric.daa4],
            cams=cameras,
        )
        relative_depth, shared = self.model._decode(list(features), image.shape[-2:])
        scale = metric.sfh(features[3], cam32)
        depth_m = relative_depth * scale.view(-1, 1, 1, 1) * metric.max_depth
        heads = self.model.assistive_heads(shared, image.shape[-2:])
        return (
            depth_m,
            heads["ground_logits"],
            heads["clearance_m"],
            heads["occupancy_logits"],
            heads["confidence_logits"],
        )


def _tuple_outputs(outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(outputs[name] for name in OUTPUT_NAMES)


def parity_summary(reference: tuple[torch.Tensor, ...], candidate: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    require(len(reference) == len(candidate) == len(OUTPUT_NAMES), "export parity output count drift")
    summary: dict[str, Any] = {}
    for name, expected, observed in zip(OUTPUT_NAMES, reference, candidate, strict=True):
        require(expected.shape == observed.shape, f"export parity shape drift: {name}")
        delta = (expected.float() - observed.float()).abs()
        summary[name] = {
            "shape": list(expected.shape),
            "max_abs": float(delta.max().item()),
            "mean_abs": float(delta.mean().item()),
        }
        require(summary[name]["max_abs"] <= 1e-5, f"external camera parity failed: {name}")
    return summary


def load_selected_model(source: Path, initialization: Path, selected: Path, device: torch.device) -> tuple[DepthArtAssistiveGeometry, dict[str, Any]]:
    install_timm_compat()
    sys.path.insert(0, str(source / "metric"))
    sys.path.insert(0, str(source / "deploy" / "shared"))
    sys.path.insert(0, str(source / "deploy" / "shared" / "selective_scan"))
    from model import load_model

    checkpoint = torch.load(selected, map_location="cpu", weights_only=False)
    require(
        checkpoint.get("schema")
        in {
            "blindassist_assistive_geometry_b1_a0_checkpoint_v1",
            "blindassist_assistive_geometry_b1_additive_arm_checkpoint_v1",
        },
        "selected Assistive Geometry checkpoint schema drift",
    )
    torch.manual_seed(int(checkpoint["seed"]))
    base = load_model(initialization, "S", "indoor", str(device))
    model = DepthArtAssistiveGeometry(base).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--height", type=int, default=608)
    parser.add_argument("--width", type=int, default=448)
    args = parser.parse_args()
    require((args.height, args.width) in {(608, 448), (448, 608)}, "unfrozen Assistive Geometry export shape")
    require(not args.output.exists() and not args.metadata.exists(), "export output already exists")
    source = args.source.resolve()
    initialization = args.initialization.resolve()
    selected = args.selected_checkpoint.resolve()
    require(source.is_dir() and initialization.is_file() and selected.is_file(), "export input missing")
    device = torch.device("cuda")
    require(torch.cuda.is_available(), "Assistive Geometry export requires CUDA")
    model, checkpoint = load_selected_model(source, initialization, selected, device)

    from network import tvimblock
    from depthart_selective_scan import install_depthart, parameter_fingerprint, register_onnx_symbolic
    from export_helpers import install_exportable_sdpa

    install_depthart(tvimblock)
    register_onnx_symbolic(17)
    install_exportable_sdpa()
    wrapper = ExternalCameraAssistiveGeometry(model).to(device).eval()
    fingerprint = parameter_fingerprint(wrapper)
    image = torch.randn(1, 3, args.height, args.width, device=device)
    intrinsics = torch.tensor(
        [[[420.0, 0.0, args.width / 2.0], [0.0, 420.0, args.height / 2.0], [0.0, 0.0, 1.0]]],
        device=device,
    )
    cameras = model.metric_depthart.cam_embedder(intrinsics, args.height, args.width, device)
    with torch.inference_mode():
        reference = _tuple_outputs(model(image, intrinsics))
        external = wrapper(image, *cameras)
    parity = parity_summary(reference, external)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (image, *cameras),
            args.output,
            input_names=("image", "camera_prompt_4", "camera_prompt_8", "camera_prompt_16", "camera_prompt_32"),
            output_names=OUTPUT_NAMES,
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
            training=torch.onnx.TrainingMode.PRESERVE,
            dynamo=False,
        )
    require(parameter_fingerprint(wrapper) == fingerprint, "export mutated model parameters")
    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    scans = [node for node in graph.graph.node if node.domain == "com.depthart" and node.op_type == "SelectiveScan"]
    require(len(scans) == 5, "Assistive Geometry export SelectiveScan count drift")
    receipt = {
        "schema": "blindassist_assistive_geometry_mobile_export_v1",
        "selected_checkpoint": {
            "path": str(selected),
            "bytes": selected.stat().st_size,
            "sha256": sha256_file(selected),
            "schema": checkpoint["schema"],
            "arm": checkpoint.get("arm", "A0_DEPTH_ONLY"),
            "seed": checkpoint["seed"],
            "epoch": checkpoint["next_epoch"],
            "model_state_sha256": checkpoint["model_state_sha256"],
        },
        "initialization_sha256": sha256_file(initialization),
        "input_shape": [1, 3, args.height, args.width],
        "inputs": ["image", "camera_prompt_4", "camera_prompt_8", "camera_prompt_16", "camera_prompt_32"],
        "outputs": list(OUTPUT_NAMES),
        "host_camera_prompt_required": True,
        "onnx": {"path": str(args.output.resolve()), "bytes": args.output.stat().st_size, "sha256": sha256_file(args.output), "node_count": len(graph.graph.node), "selective_scan_count": len(scans)},
        "pytorch_external_camera_parity": parity,
        "parameter_sha256": fingerprint,
        "task_postprocess_inside_graph": False,
        "gravity_and_unknown_postprocess_required_on_host": True,
        "strict_g4d_reopened": False,
        "claim_ceiling": "Export mechanics only; no QAIRT/HTP parity, partition, task quality, performance, product or safety authority.",
        "terminal": "ASSISTIVE_GEOMETRY_ONNX_EXPORT_PASS",
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
