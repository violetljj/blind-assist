#!/usr/bin/env python3
"""Export a DepthART metric graph with host-computed camera prompts."""

from __future__ import annotations

import argparse
import hashlib
import sys
import types
from pathlib import Path

import onnx
import torch


def install_timm_compat() -> None:
    try:
        from timm.models.layers.helpers import to_2tuple  # type: ignore # noqa: F401
    except (ImportError, ValueError):
        from timm.layers.helpers import to_2tuple  # type: ignore

        module = types.ModuleType("timm.models.layers.helpers")
        module.to_2tuple = to_2tuple
        sys.modules["timm.models.layers.helpers"] = module


class ExternalCameraMetric(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, cam4: torch.Tensor, cam8: torch.Tensor,
                cam16: torch.Tensor, cam32: torch.Tensor) -> torch.Tensor:
        cameras = [cam4, cam8, cam16, cam32]
        features = self.model.pretrained.forward_with_adapters(
            image,
            adapters=[self.model.daa1, self.model.daa2, self.model.daa3, self.model.daa4],
            cams=cameras,
        )
        depth = self.model.depth_head(features, image.shape[-2], image.shape[-1])
        scale = self.model.sfh(features[3], cam32)
        return (depth * scale.view(-1, 1, 1, 1) * self.model.max_depth).squeeze(1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=448)
    args = parser.parse_args()

    install_timm_compat()
    source = args.source.resolve()
    sys.path.insert(0, str(source / "metric"))
    sys.path.insert(0, str(source / "deploy" / "shared"))
    sys.path.insert(0, str(source / "deploy" / "shared" / "selective_scan"))
    from model import load_model  # type: ignore
    from network import tvimblock  # type: ignore
    from depthart_selective_scan import install_depthart, parameter_fingerprint, register_onnx_symbolic  # type: ignore
    from export_helpers import install_exportable_sdpa  # type: ignore

    model = load_model(args.checkpoint, "S", "indoor", "cuda")
    wrapper = ExternalCameraMetric(model).cuda().eval()
    install_depthart(tvimblock)
    register_onnx_symbolic(17)
    install_exportable_sdpa()
    fingerprint = parameter_fingerprint(wrapper)
    image = torch.randn(1, 3, args.resolution, args.resolution, device="cuda")
    K = torch.tensor(
        [[[500.0, 0.0, args.resolution / 2.0],
          [0.0, 500.0, args.resolution / 2.0],
          [0.0, 0.0, 1.0]]],
        device="cuda",
    )
    cameras = model.cam_embedder(K, args.resolution, args.resolution, "cuda")
    with torch.inference_mode():
        reference = model(image, K)
        external = wrapper(image, *cameras)
    max_abs = float((reference - external).abs().max().item())
    mean_abs = float((reference - external).abs().mean().item())
    if max_abs > 1e-5:
        raise RuntimeError(f"external camera prompt parity failed: max_abs={max_abs}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (image, *cameras),
            args.output,
            input_names=("image", "camera_prompt_4", "camera_prompt_8", "camera_prompt_16", "camera_prompt_32"),
            output_names=("depth",),
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
            training=torch.onnx.TrainingMode.PRESERVE,
            dynamo=False,
        )
    if parameter_fingerprint(wrapper) != fingerprint:
        raise RuntimeError("export changed model parameters")
    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    scans = [node for node in graph.graph.node if node.domain == "com.depthart" and node.op_type == "SelectiveScan"]
    forbidden = [node for node in graph.graph.node if node.op_type in {"Acos", "Einsum"}]
    print({
        "output": str(args.output.resolve()),
        "sha256": sha256(args.output),
        "nodes": len(graph.graph.node),
        "selective_scan": len(scans),
        "forbidden_before_rewrite": {node.op_type for node in forbidden},
        "camera_prompt_parity_max_abs": max_abs,
        "camera_prompt_parity_mean_abs": mean_abs,
        "parameter_sha256": fingerprint,
    })


if __name__ == "__main__":
    main()
