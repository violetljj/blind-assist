#!/usr/bin/env python3
"""Export a selected R1 FP32 checkpoint to a fixed-layout ONNX graph."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

try:
    from .models import ExportableNchwRawRgbSegmenter, ExportableRawRgbSegmenter, build_model, sha256_file
    from .train import resolve, write_json
except ImportError:  # pragma: no cover - direct script execution
    from models import ExportableNchwRawRgbSegmenter, ExportableRawRgbSegmenter, build_model, sha256_file
    from train import resolve, write_json


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    checkpoint = resolve(repo_root, args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = resolve(repo_root, args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite ONNX output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    model_kwargs = {
        "ddrnet_architecture_source": resolve(repo_root, args.ddrnet_architecture_source) if args.ddrnet_architecture_source else None,
        "ddrnet_checkpoint": resolve(repo_root, args.ddrnet_source_checkpoint) if args.ddrnet_source_checkpoint else None,
        "segformer_checkpoint_dir": resolve(repo_root, args.segformer_checkpoint_dir) if args.segformer_checkpoint_dir else None,
    }
    model = build_model(args.model_id, **model_kwargs).cpu().eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("FP32 checkpoint does not contain state_dict")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"FP32 checkpoint tensor identity mismatch: missing={missing}, unexpected={unexpected}")
    exportable = (
        ExportableNchwRawRgbSegmenter(model).eval()
        if args.onnx_layout == "nchw"
        else ExportableRawRgbSegmenter(model).eval()
    )
    dummy = torch.zeros(
        (1, 3, 256, 256) if args.onnx_layout == "nchw" else (1, 256, 256, 3),
        dtype=torch.float32,
    )
    with torch.no_grad():
        expected = exportable(dummy)
    expected_shape = (1, 4, 256, 256) if args.onnx_layout == "nchw" else (1, 256, 256, 4)
    if tuple(expected.shape) != expected_shape or not torch.isfinite(expected).all():
        raise ValueError(f"unexpected FP32 export shape/value: {tuple(expected.shape)}")
    torch.onnx.export(
        exportable,
        (dummy,),
        str(output),
        input_names=["rgb"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    import onnx

    graph = onnx.load(str(output))
    if args.model_id == "DDRNet-23-Slim":
        _freeze_ddrnet_resize_sizes(graph, onnx)
        onnx.save(graph, str(output))
    onnx.checker.check_model(graph)
    input_shape = [int(d.dim_value) for d in graph.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [int(d.dim_value) for d in graph.graph.output[0].type.tensor_type.shape.dim]
    expected_input_shape = [1, 3, 256, 256] if args.onnx_layout == "nchw" else [1, 256, 256, 3]
    expected_output_shape = [1, 4, 256, 256] if args.onnx_layout == "nchw" else [1, 256, 256, 4]
    if input_shape != expected_input_shape or output_shape != expected_output_shape:
        raise ValueError(f"unexpected ONNX contract: input={input_shape} output={output_shape}")
    receipt = {
        "schema_version": "blindassist.dual_loop_segmentation_model_selection_r1.onnx_receipt.v1",
        "protocol_id": "DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1",
        "status": "FP32_ONNX_EXPORTED",
        "model_id": args.model_id,
        "implementation_identity": model.build_receipt.implementation_identity,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "onnx": str(output.resolve()),
        "onnx_sha256": sha256_file(output),
        "onnx_opset": 17,
        "input_contract": {"shape": input_shape, "dtype": "float32", "range": "0..255", "layout": args.onnx_layout.upper()},
        "output_contract": {"shape": output_shape, "dtype": "float32", "layout": args.onnx_layout.upper(), "class_count": 4},
        "build_receipt": model.build_receipt.as_dict(),
        "fresh_holdout_consumed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path = output.with_suffix(".receipt.json")
    write_json(receipt_path, receipt)
    write_json(receipt_path.with_suffix(".sha256.json"), {"sha256": sha256_file(receipt_path)})
    return receipt


def _freeze_ddrnet_resize_sizes(graph: Any, onnx: Any) -> None:
    """Replace shape-derived DAPPM resize scales with constants for 256 input.

    The official PyTorch source expresses these sizes from intermediate
    tensor shapes. The R1 contract is fixed at 256x256, so making the
    constants explicit is semantics-preserving and avoids an onnx2tf
    scalar-shape ambiguity. These are internal NCHW tensors, so the ONNX
    ``sizes`` input uses the standard ``[N, C, H, W]`` order.
    """
    target_shapes = {
        "/model/core/Resize": (64, 32, 32),
        "/model/core/Resize_1": (64, 32, 32),
        "/model/core/spp/Resize": (128, 4, 4),
        "/model/core/spp/Resize_1": (128, 4, 4),
        "/model/core/spp/Resize_2": (128, 4, 4),
        "/model/core/spp/Resize_3": (128, 4, 4),
        "/model/core/Resize_2": (128, 32, 32),
        "/model/Resize": (4, 256, 256),
    }
    existing = {initializer.name for initializer in graph.graph.initializer}
    for index, node in enumerate(graph.graph.node):
        if node.op_type != "Resize" or node.name not in target_shapes:
            continue
        size_name = f"r1_fixed_resize_size_{index}"
        if size_name not in existing:
            channels, height, width = target_shapes[node.name]
            size_values = [1, channels, height, width]
            graph.graph.initializer.append(
                onnx.helper.make_tensor(
                    size_name,
                    onnx.TensorProto.INT64,
                    [4],
                    size_values,
                )
            )
            existing.add(size_name)
        # ONNX Resize requires exactly one of scales or sizes.
        node.input[2] = ""
        node.input[3] = size_name


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=("DDRNet-23-Slim", "SegFormer-B0"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ddrnet-architecture-source")
    parser.add_argument("--ddrnet-source-checkpoint")
    parser.add_argument("--segformer-checkpoint-dir")
    parser.add_argument("--onnx-layout", choices=("nhwc", "nchw"), default="nhwc")
    args = parser.parse_args(argv)
    if args.model_id == "DDRNet-23-Slim" and not (args.ddrnet_architecture_source and args.ddrnet_source_checkpoint):
        parser.error("DDRNet requires architecture and source checkpoint")
    if args.model_id == "SegFormer-B0" and not args.segformer_checkpoint_dir:
        parser.error("SegFormer requires checkpoint directory")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
