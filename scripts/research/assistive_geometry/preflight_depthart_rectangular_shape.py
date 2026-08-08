#!/usr/bin/env python3
"""Synthetic-only DepthART rectangular PyTorch/export shape preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def static_shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append(None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=608)
    parser.add_argument("--width", type=int, default=448)
    args = parser.parse_args()

    if args.height <= 0 or args.width <= 0 or args.height % 32 or args.width % 32:
        raise ValueError("height and width must be positive multiples of 32")

    source = args.source.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    deployment = Path(__file__).resolve().parents[1] / "hftf/deployment/depthart"
    sys.path.insert(0, str(deployment))
    from export_depthart_camera_external import ExternalCameraMetric, install_timm_compat

    install_timm_compat()
    sys.path.insert(0, str(source / "metric"))
    sys.path.insert(0, str(source / "deploy" / "shared"))
    sys.path.insert(0, str(source / "deploy" / "shared" / "selective_scan"))
    from depthart_selective_scan import install_depthart, parameter_fingerprint, register_onnx_symbolic
    from export_helpers import install_exportable_sdpa
    from model import load_model
    from network import tvimblock

    torch.manual_seed(20260809)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint, "S", "indoor", device).eval()
    install_depthart(tvimblock)
    register_onnx_symbolic(17)
    install_exportable_sdpa()
    wrapper = ExternalCameraMetric(model).to(device).eval()
    fingerprint_before = parameter_fingerprint(wrapper)

    image = torch.randn(1, 3, args.height, args.width, device=device)
    intrinsics = torch.tensor(
        [[[420.0, 0.0, args.width / 2.0],
          [0.0, 420.0, args.height / 2.0],
          [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )
    cameras = model.cam_embedder(intrinsics, args.height, args.width, device)
    with torch.inference_mode():
        direct = model(image, intrinsics)
        external = wrapper(image, *cameras)
    difference = (direct - external).abs()
    parity_max_abs = float(difference.max().item())
    parity_mean_abs = float(difference.mean().item())
    if tuple(direct.shape) != (1, args.height, args.width):
        raise RuntimeError(f"unexpected PyTorch output shape: {tuple(direct.shape)}")
    if not bool(torch.isfinite(direct).all().item()):
        raise RuntimeError("non-finite PyTorch output")
    if parity_max_abs > 1e-5:
        raise RuntimeError(f"camera externalization parity failed: {parity_max_abs}")

    onnx_path = output_dir / f"depthart_metric_indoor_s_{args.height}x{args.width}_camera_external.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (image, *cameras),
            onnx_path,
            input_names=("image", "camera_prompt_4", "camera_prompt_8", "camera_prompt_16", "camera_prompt_32"),
            output_names=("depth",),
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
            training=torch.onnx.TrainingMode.PRESERVE,
            dynamo=False,
        )
    if parameter_fingerprint(wrapper) != fingerprint_before:
        raise RuntimeError("export changed model parameters")

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    input_shapes = {value.name: static_shape(value) for value in graph.graph.input}
    output_shapes = {value.name: static_shape(value) for value in graph.graph.output}
    if input_shapes.get("image") != [1, 3, args.height, args.width]:
        raise RuntimeError(f"exported image shape drift: {input_shapes.get('image')}")
    exported_depth_shape = output_shapes.get("depth")
    if exported_depth_shape is None or len(exported_depth_shape) != 3:
        raise RuntimeError(f"exported depth rank drift: {exported_depth_shape}")
    output_metadata_static = exported_depth_shape == [1, args.height, args.width]
    status = (
        "PYTORCH_AND_ONNX_STATIC_SHAPE_PASS"
        if output_metadata_static
        else "PYTORCH_SHAPE_PASS_ONNX_GRAPH_PASS_SYMBOLIC_OUTPUT_METADATA"
    )

    receipt = {
        "schema": "blindassist_assistive_geometry_b0_depthart_rectangular_shape_preflight_v1",
        "status": status,
        "authority": "SYNTHETIC_SHAPE_AND_CAMERA_EXTERNALIZATION_ONLY",
        "explicit_exclusions": [
            "TASK_QUALITY",
            "DATA_ADMISSION",
            "QNN_CONVERSION",
            "HTP_EXECUTION",
            "LATENCY",
            "DEFAULT_APP",
            "SAFETY"
        ],
        "device": device,
        "torch_version": torch.__version__,
        "tf32_disabled": True,
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint)
        },
        "pytorch": {
            "image_shape": list(image.shape),
            "camera_prompt_shapes": [list(value.shape) for value in cameras],
            "depth_shape": list(direct.shape),
            "depth_finite": True,
            "camera_externalization_max_abs": parity_max_abs,
            "camera_externalization_mean_abs": parity_mean_abs,
            "parameter_sha256": fingerprint_before
        },
        "onnx": {
            "path": str(onnx_path),
            "bytes": onnx_path.stat().st_size,
            "sha256": sha256(onnx_path),
            "input_shapes": input_shapes,
            "output_shapes": output_shapes,
            "output_shape_metadata_static": output_metadata_static,
            "expected_depth_shape_from_pytorch": [1, args.height, args.width],
            "nodes": len(graph.graph.node),
            "selective_scan_nodes": sum(
                node.domain == "com.depthart" and node.op_type == "SelectiveScan"
                for node in graph.graph.node
            ),
            "acos_nodes": sum(node.op_type == "Acos" for node in graph.graph.node),
            "einsum_nodes": sum(node.op_type == "Einsum" for node in graph.graph.node)
        }
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
