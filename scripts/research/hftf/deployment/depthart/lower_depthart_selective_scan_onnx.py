#!/usr/bin/env python3
"""Lower the fixed-shape DepthART SelectiveScan contract to standard ONNX ops.

This is a correctness/graph-size feasibility path. It deliberately unrolls the
sequence recurrence and does not claim to be an efficient HTP implementation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


DOMAIN = "com.depthart"
OP_TYPE = "SelectiveScan"


def _shape_map(model: onnx.ModelProto) -> dict[str, list[int]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    result: dict[str, list[int]] = {}
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int] = []
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                break
            dims.append(dim.dim_value)
        else:
            result[value.name] = dims
    for item in inferred.graph.initializer:
        result[item.name] = list(item.dims)
    return result


def _attributes(node: onnx.NodeProto) -> dict[str, int]:
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "selective_scan"


def _lower_node(
    node: onnx.NodeProto,
    node_index: int,
    shapes: dict[str, list[int]],
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto], dict[str, object]]:
    if len(node.input) != 7 or len(node.output) != 1:
        raise ValueError(f"{node.name or node_index}: expected 7 inputs and 1 output")
    attributes = _attributes(node)
    if attributes.get("delta_softplus", 1) != 1 or attributes.get("out_float", 0) != 0:
        raise ValueError(f"{node.name or node_index}: only delta_softplus=1/out_float=0 is frozen")

    u, delta, matrix_a, matrix_b, matrix_c, skip_d, delta_bias = node.input
    u_shape = shapes.get(u)
    b_shape = shapes.get(matrix_b)
    if not u_shape or len(u_shape) != 3 or not b_shape or len(b_shape) != 4:
        raise ValueError(f"{node.name or node_index}: static rank-3 u and rank-4 B are required")
    batch, channels, length = u_shape
    b_batch, groups, state_dim, b_length = b_shape
    if batch != b_batch or length != b_length or channels % groups != 0:
        raise ValueError(f"{node.name or node_index}: incompatible static dimensions")

    prefix = f"depthart_ss_lower_{node_index}_{_safe_name(node.name)}"
    axes_0 = f"{prefix}_axes_0"
    axes_last = f"{prefix}_axes_last"
    axes_bias = f"{prefix}_axes_bias"
    reduce_last = f"{prefix}_reduce_last"
    group_index = f"{prefix}_group_index"
    initializers = [
        numpy_helper.from_array(np.asarray([0], dtype=np.int64), axes_0),
        numpy_helper.from_array(np.asarray([-1], dtype=np.int64), axes_last),
        numpy_helper.from_array(np.asarray([0, 2], dtype=np.int64), axes_bias),
        numpy_helper.from_array(np.asarray([-1], dtype=np.int64), reduce_last),
        numpy_helper.from_array(np.repeat(np.arange(groups), channels // groups).astype(np.int64), group_index),
    ]
    nodes: list[onnx.NodeProto] = []

    bias_3d = f"{prefix}_bias_3d"
    delta_biased = f"{prefix}_delta_biased"
    delta_active = f"{prefix}_delta_active"
    a_3d = f"{prefix}_a_3d"
    state = f"{prefix}_state_0"
    nodes.extend([
        helper.make_node("Unsqueeze", [delta_bias, axes_bias], [bias_3d], name=f"{prefix}_UnsqueezeBias"),
        helper.make_node("Add", [delta, bias_3d], [delta_biased], name=f"{prefix}_AddBias"),
        helper.make_node("Softplus", [delta_biased], [delta_active], name=f"{prefix}_Softplus"),
        helper.make_node("Unsqueeze", [matrix_a, axes_0], [a_3d], name=f"{prefix}_UnsqueezeA"),
        helper.make_node("Sub", [a_3d, a_3d], [state], name=f"{prefix}_ZeroState"),
    ])

    outputs: list[str] = []
    for step in range(length):
        step_index = f"{prefix}_step_{step}"
        initializers.append(numpy_helper.from_array(np.asarray(step, dtype=np.int64), step_index))
        u_t = f"{prefix}_u_{step}"
        u_3d = f"{prefix}_u_3d_{step}"
        dt = f"{prefix}_dt_{step}"
        dt_3d = f"{prefix}_dt_3d_{step}"
        transition_arg = f"{prefix}_transition_arg_{step}"
        transition = f"{prefix}_transition_{step}"
        previous = f"{prefix}_previous_{step}"
        b_group = f"{prefix}_b_group_{step}"
        b_channel = f"{prefix}_b_channel_{step}"
        input_dt = f"{prefix}_input_dt_{step}"
        input_term = f"{prefix}_input_term_{step}"
        next_state = f"{prefix}_state_{step + 1}"
        c_group = f"{prefix}_c_group_{step}"
        c_channel = f"{prefix}_c_channel_{step}"
        state_c = f"{prefix}_state_c_{step}"
        value = f"{prefix}_value_{step}"
        skip = f"{prefix}_skip_{step}"
        value_skip = f"{prefix}_value_skip_{step}"
        value_3d = f"{prefix}_value_3d_{step}"
        nodes.extend([
            helper.make_node("Gather", [u, step_index], [u_t], axis=2, name=f"{prefix}_GatherU_{step}"),
            helper.make_node("Unsqueeze", [u_t, axes_last], [u_3d], name=f"{prefix}_UnsqueezeU_{step}"),
            helper.make_node("Gather", [delta_active, step_index], [dt], axis=2, name=f"{prefix}_GatherDelta_{step}"),
            helper.make_node("Unsqueeze", [dt, axes_last], [dt_3d], name=f"{prefix}_UnsqueezeDelta_{step}"),
            helper.make_node("Mul", [dt_3d, a_3d], [transition_arg], name=f"{prefix}_TransitionArg_{step}"),
            helper.make_node("Exp", [transition_arg], [transition], name=f"{prefix}_Transition_{step}"),
            helper.make_node("Mul", [transition, state], [previous], name=f"{prefix}_Previous_{step}"),
            helper.make_node("Gather", [matrix_b, step_index], [b_group], axis=3, name=f"{prefix}_GatherBStep_{step}"),
            helper.make_node("Gather", [b_group, group_index], [b_channel], axis=1, name=f"{prefix}_GatherBGroup_{step}"),
            helper.make_node("Mul", [dt_3d, b_channel], [input_dt], name=f"{prefix}_InputDelta_{step}"),
            helper.make_node("Mul", [input_dt, u_3d], [input_term], name=f"{prefix}_InputTerm_{step}"),
            helper.make_node("Add", [previous, input_term], [next_state], name=f"{prefix}_State_{step}"),
            helper.make_node("Gather", [matrix_c, step_index], [c_group], axis=3, name=f"{prefix}_GatherCStep_{step}"),
            helper.make_node("Gather", [c_group, group_index], [c_channel], axis=1, name=f"{prefix}_GatherCGroup_{step}"),
            helper.make_node("Mul", [next_state, c_channel], [state_c], name=f"{prefix}_StateC_{step}"),
            helper.make_node("ReduceSum", [state_c, reduce_last], [value], keepdims=0, name=f"{prefix}_Reduce_{step}"),
            helper.make_node("Mul", [skip_d, u_t], [skip], name=f"{prefix}_Skip_{step}"),
            helper.make_node("Add", [value, skip], [value_skip], name=f"{prefix}_Value_{step}"),
            helper.make_node("Unsqueeze", [value_skip, axes_last], [value_3d], name=f"{prefix}_UnsqueezeValue_{step}"),
        ])
        state = next_state
        outputs.append(value_3d)
    nodes.append(helper.make_node("Concat", outputs, list(node.output), axis=2, name=f"{prefix}_ConcatOutput"))
    return nodes, initializers, {
        "name": node.name,
        "batch": batch,
        "channels": channels,
        "length": length,
        "groups": groups,
        "state_dim": state_dim,
        "lowered_node_count": len(nodes),
        "initializer_count": len(initializers),
    }


def lower_model(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, object]]:
    shapes = _shape_map(model)
    replacement: list[onnx.NodeProto] = []
    extra_initializers: list[onnx.TensorProto] = []
    records: list[dict[str, object]] = []
    original_count = len(model.graph.node)
    for index, node in enumerate(model.graph.node):
        if node.domain == DOMAIN and node.op_type == OP_TYPE:
            nodes, initializers, record = _lower_node(node, index, shapes)
            replacement.extend(nodes)
            extra_initializers.extend(initializers)
            records.append(record)
        else:
            replacement.append(node)
    if not records:
        raise ValueError("no com.depthart::SelectiveScan nodes found")
    del model.graph.node[:]
    model.graph.node.extend(replacement)
    model.graph.initializer.extend(extra_initializers)
    onnx.checker.check_model(model)
    return model, {
        "status": "PASS_STANDARD_ONNX_UNROLLED_GRAPH_GENERATED",
        "original_node_count": original_count,
        "output_node_count": len(model.graph.node),
        "selective_scan_nodes_lowered": len(records),
        "remaining_custom_selective_scan_nodes": sum(
            node.domain == DOMAIN and node.op_type == OP_TYPE for node in model.graph.node
        ),
        "lowered": records,
        "claim_boundary": "CORRECTNESS_AND_GRAPH_SIZE_FEASIBILITY_ONLY_NOT_AN_EFFICIENT_HTP_IMPLEMENTATION",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    model, receipt = lower_model(onnx.load(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    receipt["input"] = str(args.input)
    receipt["output"] = str(args.output)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
