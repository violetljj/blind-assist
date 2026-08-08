#!/usr/bin/env python3
"""Map frozen DepthART last-axis LayerNormalization nodes to the HTP custom op."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import helper


def rewrite_model(model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[dict[str, object]]]:
    rewritten = onnx.ModelProto()
    rewritten.CopyFrom(model)
    records: list[dict[str, object]] = []
    for index, node in enumerate(rewritten.graph.node):
        if node.op_type != "LayerNormalization" or node.domain not in ("", "ai.onnx"):
            continue
        attributes = {item.name: helper.get_attribute_value(item) for item in node.attribute}
        axis = int(attributes.get("axis", -1))
        epsilon = float(attributes.get("epsilon", 1e-5))
        stash_type = int(attributes.get("stash_type", 1))
        if len(node.input) != 3 or len(node.output) != 1 or axis != -1 or stash_type != 1:
            raise ValueError(f"{node.name or index}: unsupported LayerNormalization contract")
        original_name = node.name
        node.op_type = "DepthArtLayerNorm"
        node.domain = "com.depthart"
        del node.attribute[:]
        node.attribute.extend([helper.make_attribute("epsilon", epsilon)])
        records.append({"index": index, "name": original_name, "epsilon": epsilon})
    if not records:
        raise ValueError("no frozen LayerNormalization nodes found")
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
        "schema": "blindassist_depthart_custom_layernorm_rewrite",
        "schema_version": 1,
        "status": "REWRITTEN_RUNTIME_NOT_EVALUATED",
        "rewritten_count": len(records),
        "nodes": records,
        "authority": "Structural ONNX rewrite only; no QNN context, HTP execution, parity, performance, safety, or production claim.",
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
