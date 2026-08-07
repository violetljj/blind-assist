#!/usr/bin/env python3
"""Fold fixed-S448 no-op broadcasts and constant Mod nodes in DepthART ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def tensor_shape(value: onnx.ValueInfoProto) -> tuple[int, ...] | None:
    dims = value.type.tensor_type.shape.dim
    if any(not dim.HasField("dim_value") for dim in dims):
        return None
    return tuple(dim.dim_value for dim in dims)


def is_noop_expand(input_shape: tuple[int, ...], target_shape: np.ndarray) -> bool:
    target = tuple(int(value) for value in target_shape.reshape(-1))
    return np.broadcast_shapes(input_shape, target) == input_shape


def constant_mod(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if np.any(right == 0):
        raise ValueError("cannot fold Mod with zero divisor")
    return np.mod(left, right)


class StaticEvaluator:
    def __init__(self, model: onnx.ModelProto):
        inferred = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
        values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
        self.shapes = {value.name: tensor_shape(value) for value in values}
        self.shapes.update({item.name: tuple(item.dims) for item in model.graph.initializer})
        self.producers = {output: node for node in model.graph.node for output in node.output}
        self.cache = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}

    def value(self, name: str) -> np.ndarray:
        if name in self.cache:
            return self.cache[name]
        node = self.producers[name]
        attrs = {item.name: helper.get_attribute_value(item) for item in node.attribute}
        if node.op_type == "Shape":
            shape = self.shapes.get(node.input[0])
            if shape is None:
                raise ValueError(f"dynamic Shape input for {node.name!r}")
            result = np.asarray(shape, dtype=np.int64)
        elif node.op_type == "Constant":
            result = numpy_helper.to_array(attrs["value"])
        else:
            inputs = [self.value(item) for item in node.input]
            if node.op_type == "Gather":
                result = np.take(inputs[0], inputs[1], axis=attrs.get("axis", 0))
            elif node.op_type == "Unsqueeze":
                axes = tuple(inputs[1].tolist()) if len(inputs) > 1 else tuple(attrs["axes"])
                result = np.expand_dims(inputs[0], axes)
            elif node.op_type == "Concat":
                result = np.concatenate(inputs, axis=attrs["axis"])
            elif node.op_type == "Reshape":
                result = np.reshape(inputs[0], inputs[1])
            elif node.op_type == "Mul":
                result = inputs[0] * inputs[1]
            elif node.op_type == "Equal":
                result = inputs[0] == inputs[1]
            elif node.op_type == "Where":
                result = np.where(inputs[0], inputs[1], inputs[2])
            elif node.op_type == "ConstantOfShape":
                fill = numpy_helper.to_array(attrs["value"]).reshape(-1)[0]
                result = np.full(tuple(inputs[0].tolist()), fill)
            elif node.op_type == "Cast":
                dtypes = {1: np.float32, 6: np.int32, 7: np.int64}
                result = inputs[0].astype(dtypes[attrs["to"]])
            elif node.op_type == "Add":
                result = inputs[0] + inputs[1]
            elif node.op_type == "Mod":
                result = constant_mod(inputs[0], inputs[1])
            else:
                raise ValueError(f"unsupported static op {node.op_type!r} at {node.name!r}")
        self.cache[name] = result
        return result


def prune_dead_nodes(model: onnx.ModelProto) -> Counter[str]:
    needed = {output.name for output in model.graph.output}
    kept: list[onnx.NodeProto] = []
    removed: Counter[str] = Counter()
    for node in reversed(model.graph.node):
        if any(output in needed for output in node.output):
            kept.append(node)
            needed.update(node.input)
        else:
            removed[node.op_type] += 1
    kept.reverse()
    del model.graph.node[:]
    model.graph.node.extend(kept)
    used = {item for node in kept for item in node.input} | needed
    initializers = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    return removed


def rewrite(input_path: Path, output_path: Path, receipt_path: Path) -> dict[str, object]:
    model = onnx.load(str(input_path))
    evaluator = StaticEvaluator(model)
    aliases: dict[str, str] = {}
    folded_mods: list[str] = []
    rewritten: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.op_type == "Expand":
            input_shape = evaluator.shapes.get(node.input[0])
            target = evaluator.value(node.input[1])
            if input_shape is None or not is_noop_expand(input_shape, target):
                raise ValueError(f"Expand is not a fixed-shape no-op: {node.name!r}")
            aliases[node.output[0]] = node.input[0]
            continue
        if node.op_type == "Mod":
            value = evaluator.value(node.output[0])
            if value.dtype.kind not in "iu":
                raise ValueError(f"non-integer constant Mod: {node.name!r}")
            rewritten.append(
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=list(node.output),
                    name=f"{node.name}/qairt_constant",
                    value=numpy_helper.from_array(value),
                )
            )
            folded_mods.append(node.name)
            continue
        rewritten.append(node)
    if len(aliases) != 6 or len(folded_mods) != 4:
        raise ValueError(f"unexpected static inventory: expands={len(aliases)}, mods={len(folded_mods)}")
    for node in rewritten:
        for index, name in enumerate(node.input):
            node.input[index] = aliases.get(name, name)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    removed_dead = prune_dead_nodes(model)
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    result: dict[str, object] = {
        "schema": "blindassist_depthart_qairt_static_shape_r0",
        "input_sha256": sha256(input_path),
        "output_sha256": sha256(output_path),
        "noop_expands_removed": len(aliases),
        "constant_mods_folded": len(folded_mods),
        "dead_nodes_removed": dict(removed_dead),
        "node_count": len(model.graph.node),
        "equivalence": "EXACT_FIXED_S448_BROADCAST_AND_INTEGER_CONSTANT_FOLD_FULL_RUNTIME_PARITY_PENDING",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(rewrite(args.input, args.output, args.receipt), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
