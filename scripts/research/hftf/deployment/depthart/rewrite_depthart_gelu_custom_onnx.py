#!/usr/bin/env python3
"""Replace exact exported erf-GELU patterns with the float32 HTP reference op."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _scalar_value(
    tensor_name: str,
    model: onnx.ModelProto,
    producers: dict[str, int],
) -> float | None:
    initializers = {item.name: item for item in model.graph.initializer}
    if tensor_name in initializers:
        value = numpy_helper.to_array(initializers[tensor_name])
    elif tensor_name in producers:
        node = model.graph.node[producers[tensor_name]]
        if node.op_type != "Constant" or node.domain not in ("", "ai.onnx"):
            return None
        attribute = next((item for item in node.attribute if item.name == "value"), None)
        if attribute is None:
            return None
        value = numpy_helper.to_array(helper.get_attribute_value(attribute))
    else:
        return None
    array = np.asarray(value)
    return float(array.reshape(-1)[0]) if array.size == 1 else None


def _unique_consumer(
    tensor_name: str,
    consumers: dict[str, list[int]],
    nodes: list[onnx.NodeProto],
    op_type: str,
) -> int | None:
    matches = [index for index in consumers.get(tensor_name, []) if nodes[index].op_type == op_type]
    return matches[0] if len(matches) == 1 and len(consumers.get(tensor_name, [])) == 1 else None


def _other_input(node: onnx.NodeProto, known: str) -> str | None:
    inputs = list(node.input)
    if len(inputs) != 2 or inputs.count(known) != 1:
        return None
    return inputs[1] if inputs[0] == known else inputs[0]


def rewrite_model(model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[dict[str, object]]]:
    rewritten = onnx.ModelProto()
    rewritten.CopyFrom(model)
    nodes = list(rewritten.graph.node)
    producers = {output: index for index, node in enumerate(nodes) for output in node.output}
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        for tensor in node.input:
            consumers.setdefault(tensor, []).append(index)

    replacements: dict[int, onnx.NodeProto] = {}
    removed: set[int] = set()
    records: list[dict[str, object]] = []
    for erf_index, erf in enumerate(nodes):
        if erf.op_type != "Erf" or erf.domain not in ("", "ai.onnx") or len(erf.input) != 1:
            continue
        div_index = producers.get(erf.input[0])
        if div_index is None:
            continue
        div = nodes[div_index]
        if div.op_type != "Div" or len(div.input) != 2 or len(div.output) != 1:
            continue
        x_name, divisor_name = div.input
        divisor = _scalar_value(divisor_name, rewritten, producers)
        if divisor is None or not math.isclose(divisor, math.sqrt(2.0), rel_tol=1e-6, abs_tol=1e-6):
            continue
        if consumers.get(div.output[0]) != [erf_index] or len(erf.output) != 1:
            continue

        add_index = _unique_consumer(erf.output[0], consumers, nodes, "Add")
        if add_index is None:
            continue
        add = nodes[add_index]
        one_name = _other_input(add, erf.output[0])
        if one_name is None or _scalar_value(one_name, rewritten, producers) != 1.0 or len(add.output) != 1:
            continue

        first_mul_index = _unique_consumer(add.output[0], consumers, nodes, "Mul")
        if first_mul_index is None:
            continue
        first_mul = nodes[first_mul_index]
        if _other_input(first_mul, add.output[0]) != x_name or len(first_mul.output) != 1:
            continue

        final_mul_index = _unique_consumer(first_mul.output[0], consumers, nodes, "Mul")
        if final_mul_index is None:
            continue
        final_mul = nodes[final_mul_index]
        half_name = _other_input(final_mul, first_mul.output[0])
        if half_name is None or _scalar_value(half_name, rewritten, producers) != 0.5 or len(final_mul.output) != 1:
            continue

        family = {div_index, erf_index, add_index, first_mul_index, final_mul_index}
        if family & removed:
            raise ValueError("overlapping GELU patterns are not supported")
        name = final_mul.name or f"DepthArtGelu_{len(records)}"
        replacements[div_index] = helper.make_node(
            "DepthArtGelu", [x_name], [final_mul.output[0]], name=name, domain="com.depthart"
        )
        removed.update(family)
        records.append({
            "index": div_index,
            "name": name,
            "input": x_name,
            "output": final_mul.output[0],
            "removed_nodes": [nodes[index].name for index in sorted(family)],
        })

    if not records:
        raise ValueError("no exact erf-GELU patterns found")
    new_nodes: list[onnx.NodeProto] = []
    for index, node in enumerate(nodes):
        if index in replacements:
            new_nodes.append(replacements[index])
        elif index not in removed:
            new_nodes.append(node)
    del rewritten.graph.node[:]
    rewritten.graph.node.extend(new_nodes)
    if not any(item.domain == "com.depthart" for item in rewritten.opset_import):
        rewritten.opset_import.append(helper.make_opsetid("com.depthart", 1))
    return rewritten, records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    model, records = rewrite_model(onnx.load(args.input))
    onnx.checker.check_model(model, full_check=False, check_custom_domain=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    receipt = {
        "schema": "blindassist_depthart_custom_gelu_rewrite_v1",
        "status": "REWRITTEN_RUNTIME_NOT_EVALUATED",
        "rewritten_count": len(records),
        "nodes": records,
        "authority": "Exact structural GELU node-family rewrite only; no HTP parity or performance claim.",
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
