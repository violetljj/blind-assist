#!/usr/bin/env python3
"""Rewrite only DepthART's frozen first patch Conv to a correctness HTP op."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import helper


TARGET_NAME = "/patch_embed/patch_embed.0/c/Conv"


def rewrite_model(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, object]]:
    rewritten = onnx.ModelProto()
    rewritten.CopyFrom(model)
    matches = [
        (index, node)
        for index, node in enumerate(rewritten.graph.node)
        if node.name == TARGET_NAME
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {TARGET_NAME!r}, found {len(matches)}")
    index, node = matches[0]
    attributes = {
        item.name: helper.get_attribute_value(item) for item in node.attribute
    }
    expected = {
        "dilations": [1, 1],
        "group": 1,
        "kernel_shape": [3, 3],
        "pads": [1, 1, 1, 1],
        "strides": [2, 2],
    }
    normalized = {
        name: list(value) if isinstance(value, (list, tuple)) else value
        for name, value in attributes.items()
    }
    if (
        node.op_type != "Conv"
        or node.domain not in ("", "ai.onnx")
        or len(node.input) != 2
        or len(node.output) != 1
        or any(normalized.get(name) != value for name, value in expected.items())
    ):
        raise ValueError(f"{TARGET_NAME}: unsupported frozen Conv contract: {normalized}")
    node.op_type = "DepthArtPatchConv2d"
    node.domain = "com.depthart"
    del node.attribute[:]
    if not any(item.domain == "com.depthart" for item in rewritten.opset_import):
        rewritten.opset_import.append(helper.make_opsetid("com.depthart", 1))
    return rewritten, {
        "index": index,
        "name": TARGET_NAME,
        "inputs": list(node.input),
        "output": node.output[0],
        "frozen_contract": expected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    model, record = rewrite_model(onnx.load(args.input))
    onnx.checker.check_model(model, full_check=False, check_custom_domain=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    receipt = {
        "schema": "blindassist_depthart_first_patch_conv_custom_rewrite_v1",
        "status": "REWRITTEN_RUNTIME_NOT_EVALUATED",
        "rewritten_count": 1,
        "node": record,
        "authority": "Structural single-node rewrite only; no HTP parity or performance claim.",
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
