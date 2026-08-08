#!/usr/bin/env python3
"""Map frozen DepthART inference BatchNormalization nodes to the HTP custom op."""

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
        if node.op_type != "BatchNormalization" or node.domain not in ("", "ai.onnx"):
            continue
        attributes = {
            item.name: helper.get_attribute_value(item) for item in node.attribute
        }
        epsilon = float(attributes.get("epsilon", 1e-5))
        training_mode = int(attributes.get("training_mode", 0))
        if len(node.input) != 5 or len(node.output) != 1 or training_mode != 0:
            raise ValueError(f"{node.name or index}: unsupported BatchNormalization contract")
        original_name = node.name
        node.op_type = "DepthArtBatchNorm2d"
        node.domain = "com.depthart"
        del node.attribute[:]
        node.attribute.extend([helper.make_attribute("epsilon", epsilon)])
        records.append({"index": index, "name": original_name, "epsilon": epsilon})
    if not records:
        raise ValueError("no frozen inference BatchNormalization nodes found")
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
        "schema": "blindassist_depthart_custom_batchnorm_rewrite_v1",
        "status": "REWRITTEN_RUNTIME_NOT_EVALUATED",
        "rewritten_count": len(records),
        "nodes": records,
        "authority": "Structural ONNX node-family rewrite only; no HTP parity or performance claim.",
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
