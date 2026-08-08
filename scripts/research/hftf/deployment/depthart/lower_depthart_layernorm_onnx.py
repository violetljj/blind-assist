#!/usr/bin/env python3
"""Lower frozen last-axis DepthART LayerNormalization nodes to standard ONNX ops."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "layernorm"


def lower_model(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, object]]:
    replacement: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    records: list[dict[str, object]] = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "LayerNormalization" or node.domain not in ("", "ai.onnx"):
            replacement.append(node)
            continue
        axis = next(
            (helper.get_attribute_value(attr) for attr in node.attribute if attr.name == "axis"),
            -1,
        )
        epsilon = next(
            (helper.get_attribute_value(attr) for attr in node.attribute if attr.name == "epsilon"),
            1e-5,
        )
        stash_type = next(
            (helper.get_attribute_value(attr) for attr in node.attribute if attr.name == "stash_type"),
            1,
        )
        if axis != -1 or len(node.input) != 3 or len(node.output) != 1 or stash_type != 1:
            raise ValueError(f"{node.name or index}: unsupported LayerNormalization contract")

        prefix = f"depthart_ln_lower_{index}_{_safe_name(node.name)}"
        epsilon_name = f"{prefix}_epsilon"
        initializers.append(
            numpy_helper.from_array(np.asarray(epsilon, dtype=np.float32), epsilon_name)
        )
        mean = f"{prefix}_mean"
        centered = f"{prefix}_centered"
        square = f"{prefix}_square"
        variance = f"{prefix}_variance"
        variance_epsilon = f"{prefix}_variance_epsilon"
        stddev = f"{prefix}_stddev"
        inv_stddev = f"{prefix}_inv_stddev"
        normalized = f"{prefix}_normalized"
        scaled = f"{prefix}_scaled"
        replacement.extend(
            [
                helper.make_node("ReduceMean", [node.input[0]], [mean], name=f"{prefix}_Mean", axes=[-1], keepdims=1),
                helper.make_node("Sub", [node.input[0], mean], [centered], name=f"{prefix}_Center"),
                helper.make_node("Mul", [centered, centered], [square], name=f"{prefix}_Square"),
                helper.make_node("ReduceMean", [square], [variance], name=f"{prefix}_Variance", axes=[-1], keepdims=1),
                helper.make_node("Add", [variance, epsilon_name], [variance_epsilon], name=f"{prefix}_AddEpsilon"),
                helper.make_node("Sqrt", [variance_epsilon], [stddev], name=f"{prefix}_Sqrt"),
                helper.make_node("Reciprocal", [stddev], [inv_stddev], name=f"{prefix}_Reciprocal"),
                helper.make_node("Mul", [centered, inv_stddev], [normalized], name=f"{prefix}_Normalize"),
                helper.make_node("Mul", [normalized, node.input[1]], [scaled], name=f"{prefix}_Scale"),
                helper.make_node("Add", [scaled, node.input[2]], list(node.output), name=f"{prefix}_Bias"),
            ]
        )
        records.append({"name": node.name, "axis": int(axis), "epsilon": float(epsilon)})

    if not records:
        raise ValueError("no frozen LayerNormalization nodes found")
    del model.graph.node[:]
    model.graph.node.extend(replacement)
    model.graph.initializer.extend(initializers)
    onnx.checker.check_model(model)
    return model, {
        "status": "PASS_STANDARD_ONNX_LAYERNORM_LOWERING_GENERATED",
        "layernorm_nodes_lowered": len(records),
        "remaining_layernorm_nodes": sum(node.op_type == "LayerNormalization" for node in model.graph.node),
        "records": records,
        "claim_boundary": "EQUIVALENT_FORMULA_GRAPH_ONLY_FULL_MODEL_RUNTIME_PARITY_NOT_YET_PROVEN",
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
