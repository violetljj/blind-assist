#!/usr/bin/env python3
"""Remove QAIRT-incompatible ONNX attributes only when semantics are unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import onnx
from onnx import helper


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def attributes(node: onnx.NodeProto) -> dict[str, object]:
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def remove_attributes(node: onnx.NodeProto, names: set[str]) -> None:
    kept = [attribute for attribute in node.attribute if attribute.name not in names]
    del node.attribute[:]
    node.attribute.extend(kept)


def clean_node(node: onnx.NodeProto) -> list[str]:
    attrs = attributes(node)
    removed: list[str] = []
    if node.op_type == "BatchNormalization" and "training_mode" in attrs:
        if attrs["training_mode"] != 0:
            raise ValueError(f"unsafe BatchNormalization training_mode on {node.name!r}")
        removed.append("training_mode")
    elif node.op_type == "Reshape" and "allowzero" in attrs:
        if attrs["allowzero"] != 0:
            raise ValueError(f"unsafe Reshape allowzero on {node.name!r}")
        removed.append("allowzero")
    elif node.op_type == "AveragePool":
        if attrs.get("ceil_mode") == 0:
            removed.append("ceil_mode")
        if attrs.get("count_include_pad") == 1:
            pads = list(attrs.get("pads", []))
            if pads and all(value == 0 for value in pads):
                removed.append("count_include_pad")
    remove_attributes(node, set(removed))
    return removed


def rewrite(input_path: Path, output_path: Path, receipt_path: Path) -> dict[str, object]:
    model = onnx.load(str(input_path))
    changes: Counter[str] = Counter()
    for node in model.graph.node:
        for name in clean_node(node):
            changes[f"{node.op_type}.{name}"] += 1
    expected = {
        "BatchNormalization.training_mode": 123,
        "Reshape.allowzero": 108,
        "AveragePool.ceil_mode": 4,
        "AveragePool.count_include_pad": 4,
    }
    if dict(changes) != expected:
        raise ValueError(f"unexpected hygiene inventory: {dict(changes)!r}")
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    result: dict[str, object] = {
        "schema": "blindassist_depthart_qairt_graph_hygiene_r0",
        "input_sha256": sha256(input_path),
        "output_sha256": sha256(output_path),
        "removed_attributes": dict(changes),
        "node_count": len(model.graph.node),
        "equivalence": "EXACT_ONNX_DEFAULT_OR_ZERO_PADDING_EQUIVALENCE_FULL_RUNTIME_PARITY_PENDING",
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
