#!/usr/bin/env python3
"""Export a selected R1 FP32 checkpoint to a fixed-layout ONNX graph."""

from __future__ import annotations

import argparse
import json
import re
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
    elif args.model_id == "SegFormer-B0":
        _rewrite_segformer_layer_norm(graph, onnx)
        _rewrite_segformer_token_broadcasts(graph, onnx)
        _annotate_segformer_static_shapes(graph, onnx)
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


def _annotate_segformer_static_shapes(graph: Any, onnx: Any) -> None:
    """Annotate fixed-token LayerNorm tensors for onnx2tf conversion.

    The exported graph is semantically static at 256x256, but the legacy
    exporter leaves several ``Shape -> Reshape`` paths symbolically unknown.
    onnx2tf needs the final token dimension to construct LayerNormalization
    weights. This annotation only supplies shapes implied by the frozen input
    contract; it does not alter graph operators or tensor values.
    """
    stage_shapes = {
        0: (4096, 32),
        1: (1024, 64),
        2: (256, 160),
        3: (64, 256),
    }
    reduced_lengths = {0: 64, 1: 64, 2: 64}
    shape_info: dict[str, Any] = {
        value.name: value
        for value in list(graph.graph.value_info) + list(graph.graph.input) + list(graph.graph.output)
    }

    def set_shape(name: str, shape: tuple[int, ...]) -> None:
        value = shape_info.get(name)
        if value is None:
            value = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, list(shape))
            graph.graph.value_info.append(value)
            shape_info[name] = value
        tensor_shape = value.type.tensor_type.shape
        del tensor_shape.dim[:]
        for dimension in shape:
            tensor_shape.dim.add().dim_value = int(dimension)

    for node in graph.graph.node:
        if node.op_type != "LayerNormalization":
            continue
        match = re.search(r"stages\.(\d+)", node.name)
        if match is None:
            continue
        stage = int(match.group(1))
        if stage not in stage_shapes:
            continue
        sequence, channels = stage_shapes[stage]
        if "sequence_reduction/layer_norm" in node.name:
            sequence = reduced_lengths[stage]
        shape = (1, sequence, channels)
        set_shape(node.input[0], shape)
        set_shape(node.output[0], shape)

    inferred = onnx.shape_inference.infer_shapes(graph)
    graph.graph.value_info.extend(
        value
        for value in inferred.graph.value_info
        if value.name not in shape_info
    )


def _rewrite_segformer_layer_norm(graph: Any, onnx: Any) -> None:
    """Expand SegFormer LayerNormalization nodes into portable ONNX primitives.

    The exported graph is fixed to a last-dimension LayerNorm. Some onnx2tf
    versions cannot construct a Keras LayerNormalization layer when the token
    length is symbolic, even though the channel weight is static. The primitive
    expansion is algebraically equivalent and keeps the original scale/bias
    initializers, while avoiding a converter-specific dynamic-shape path.
    """
    rewritten: list[Any] = []
    for index, node in enumerate(graph.graph.node):
        if node.op_type != "LayerNormalization":
            rewritten.append(node)
            continue
        attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
        axis = int(attributes.get("axis", -1))
        if axis != -1:
            raise ValueError(f"SegFormer LayerNormalization axis is not -1: {node.name} axis={axis}")
        if len(node.input) < 3:
            raise ValueError(f"SegFormer LayerNormalization lacks scale/bias inputs: {node.name}")
        epsilon = float(attributes.get("epsilon", 1e-5))
        prefix = f"{node.name or 'r1_segformer_layer_norm'}_{index}"
        epsilon_name = f"{prefix}/epsilon"
        if not any(initializer.name == epsilon_name for initializer in graph.graph.initializer):
            graph.graph.initializer.append(
                onnx.helper.make_tensor(
                    epsilon_name,
                    onnx.TensorProto.FLOAT,
                    [],
                    [epsilon],
                )
            )
        input_name, scale_name, bias_name = node.input[:3]
        mean_name = f"{prefix}/mean"
        centered_name = f"{prefix}/centered"
        squared_name = f"{prefix}/squared"
        variance_name = f"{prefix}/variance"
        variance_epsilon_name = f"{prefix}/variance_epsilon"
        denominator_name = f"{prefix}/denominator"
        normalized_name = f"{prefix}/normalized"
        scaled_name = f"{prefix}/scaled"
        output_name = node.output[0]
        rewritten.extend(
            [
                onnx.helper.make_node(
                    "ReduceMean",
                    [input_name],
                    [mean_name],
                    name=f"{prefix}/ReduceMean",
                    axes=[-1],
                    keepdims=1,
                ),
                onnx.helper.make_node(
                    "Sub",
                    [input_name, mean_name],
                    [centered_name],
                    name=f"{prefix}/Sub",
                ),
                onnx.helper.make_node(
                    "Mul",
                    [centered_name, centered_name],
                    [squared_name],
                    name=f"{prefix}/Square",
                ),
                onnx.helper.make_node(
                    "ReduceMean",
                    [squared_name],
                    [variance_name],
                    name=f"{prefix}/Variance",
                    axes=[-1],
                    keepdims=1,
                ),
                onnx.helper.make_node(
                    "Add",
                    [variance_name, epsilon_name],
                    [variance_epsilon_name],
                    name=f"{prefix}/AddEpsilon",
                ),
                onnx.helper.make_node(
                    "Sqrt",
                    [variance_epsilon_name],
                    [denominator_name],
                    name=f"{prefix}/Sqrt",
                ),
                onnx.helper.make_node(
                    "Div",
                    [centered_name, denominator_name],
                    [normalized_name],
                    name=f"{prefix}/Div",
                ),
                onnx.helper.make_node(
                    "Mul",
                    [normalized_name, scale_name],
                    [scaled_name],
                    name=f"{prefix}/Scale",
                ),
                onnx.helper.make_node(
                    "Add",
                    [scaled_name, bias_name],
                    [output_name],
                    name=f"{prefix}/Bias",
                ),
            ]
        )
    graph.graph.ClearField("node")
    graph.graph.node.extend(rewritten)


def _rewrite_segformer_token_broadcasts(graph: Any, onnx: Any) -> None:
    """Make vector broadcasts explicit in the transformer token layout.

    SegFormer uses ``[batch, sequence, channel]`` tensors between attention and
    MLP projections. onnx2tf's generic rank-3 layout heuristic treats a vector
    affine parameter as NCW and materializes it as ``[1, channel, 1]`` while
    leaving the token tensor as NWC. For Add/Mul nodes that consume a rank-1
    initializer, this rewrite computes the same operation in explicit NCS
    layout and transposes back. The ONNX graph remains mathematically
    equivalent, but its broadcast contract is unambiguous to the converter.
    """
    import numpy as np

    initializer_names = {initializer.name for initializer in graph.graph.initializer}
    rewritten: list[Any] = []
    rewritten_count = 0
    for index, node in enumerate(graph.graph.node):
        if node.op_type not in {"Add", "Mul"} or len(node.input) != 2 or len(node.output) != 1:
            rewritten.append(node)
            continue
        constant_index = next(
            (input_index for input_index, input_name in enumerate(node.input) if input_name in initializer_names),
            None,
        )
        if constant_index is None:
            rewritten.append(node)
            continue
        initializer = next(
            initializer for initializer in graph.graph.initializer if initializer.name == node.input[constant_index]
        )
        if len(initializer.dims) != 1 or int(initializer.dims[0]) <= 1:
            rewritten.append(node)
            continue
        tensor_index = 1 - constant_index
        tensor_name = node.input[tensor_index]
        prefix = f"{node.name or 'r1_segformer_token_broadcast'}_{index}"
        transposed_tensor_name = f"{prefix}/transpose_input"
        channel_first_parameter_name = f"{prefix}/channel_first_parameter"
        channel_first_output_name = f"{prefix}/channel_first_output"
        values = onnx.numpy_helper.to_array(initializer).astype(np.float32, copy=False)
        parameter = onnx.numpy_helper.from_array(
            values.reshape(1, values.size, 1),
            name=channel_first_parameter_name,
        )
        graph.graph.initializer.append(parameter)
        op_inputs = [transposed_tensor_name, channel_first_parameter_name]
        rewritten.extend(
            [
                onnx.helper.make_node(
                    "Transpose",
                    [tensor_name],
                    [transposed_tensor_name],
                    name=f"{prefix}/TransposeInput",
                    perm=[0, 2, 1],
                ),
                onnx.helper.make_node(
                    node.op_type,
                    op_inputs,
                    [channel_first_output_name],
                    name=f"{prefix}/{node.op_type}",
                ),
                onnx.helper.make_node(
                    "Transpose",
                    [channel_first_output_name],
                    [node.output[0]],
                    name=f"{prefix}/TransposeOutput",
                    perm=[0, 2, 1],
                ),
            ]
        )
        rewritten_count += 1
    graph.graph.ClearField("node")
    graph.graph.node.extend(rewritten)
    if rewritten_count == 0:
        raise ValueError("SegFormer token broadcast rewrite found no vector Add/Mul nodes")


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
