#!/usr/bin/env python3
"""Wrap rank-3 DepthART LayerNormalization nodes in an equivalent rank-4 form."""

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


def rewrite_model(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, object]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    ranks = {
        value.name: len(value.type.tensor_type.shape.dim)
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
        if value.type.tensor_type.HasField("shape")
    }
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
        if ranks.get(node.input[0]) != 3:
            replacement.append(node)
            continue
        if axis not in (-1, 2) or len(node.output) != 1:
            raise ValueError(f"{node.name or index}: expected rank-3 last-axis LayerNormalization")

        prefix = f"depthart_ln4_{index}_{_safe_name(node.name)}"
        axes = f"{prefix}_axes"
        expanded = f"{prefix}_expanded"
        normalized = f"{prefix}_normalized"
        initializers.append(numpy_helper.from_array(np.asarray([1], dtype=np.int64), axes))
        replacement.append(
            helper.make_node("Unsqueeze", [node.input[0], axes], [expanded], name=f"{prefix}_Unsqueeze")
        )
        rewritten = onnx.NodeProto()
        rewritten.CopyFrom(node)
        rewritten.input[0] = expanded
        rewritten.output[0] = normalized
        rewritten.name = f"{prefix}_LayerNormalization"
        replacement.append(rewritten)
        replacement.append(
            helper.make_node("Squeeze", [normalized, axes], list(node.output), name=f"{prefix}_Squeeze")
        )
        records.append({"name": node.name, "axis": int(axis), "input_rank": 3, "runtime_rank": 4})

    if not records:
        raise ValueError("no rank-3 last-axis LayerNormalization nodes found")
    del model.graph.node[:]
    model.graph.node.extend(replacement)
    model.graph.initializer.extend(initializers)
    onnx.checker.check_model(model)
    return model, {
        "status": "PASS_EQUIVALENT_RANK4_LAYERNORM_GRAPH_GENERATED",
        "layernorm_nodes_wrapped": len(records),
        "records": records,
        "claim_boundary": "EXACT_SHAPE_REWRITE_ONLY_RUNTIME_ACCEPTANCE_NOT_YET_PROVEN",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    model, receipt = rewrite_model(onnx.load(args.input))
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
