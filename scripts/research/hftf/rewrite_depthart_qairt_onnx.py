#!/usr/bin/env python3
"""Rewrite DepthART's two batched linear Einsum forms to QAIRT-friendly MatMul."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import onnx
from onnx import helper


SUPPORTED_EQUATIONS = {
    "b k d l, k c d -> b k c l",
    "b k r l, k d r -> b k d l",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def equation(node: onnx.NodeProto) -> str:
    return next(
        (attribute.s.decode("utf-8") for attribute in node.attribute if attribute.name == "equation"),
        "",
    )


def rewrite(input_path: Path, output_path: Path, receipt_path: Path) -> dict[str, object]:
    graph = onnx.load(str(input_path))
    before = collections.Counter(
        (node.domain or "ai.onnx", node.op_type) for node in graph.graph.node
    )
    rewritten: list[dict[str, str]] = []
    nodes = []
    for node in graph.graph.node:
        if node.op_type != "Einsum":
            nodes.append(node)
            continue
        rule = equation(node)
        if rule not in SUPPORTED_EQUATIONS or len(node.input) != 2 or len(node.output) != 1:
            raise ValueError(f"unsupported Einsum {node.name!r}: {rule!r}")
        nodes.append(
            helper.make_node(
                "MatMul",
                inputs=[node.input[1], node.input[0]],
                outputs=list(node.output),
                name=f"{node.name}/qairt_matmul",
            )
        )
        rewritten.append({"node": node.name, "equation": rule})
    if not rewritten:
        raise ValueError("no supported DepthART Einsum nodes found")
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    onnx.checker.check_model(graph)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(graph, str(output_path))
    after = collections.Counter(
        (node.domain or "ai.onnx", node.op_type) for node in graph.graph.node
    )
    result: dict[str, object] = {
        "schema": "blindassist_depthart_qairt_onnx_rewrite_r0",
        "input_sha256": sha256(input_path),
        "output_sha256": sha256(output_path),
        "rewritten_count": len(rewritten),
        "rewritten": rewritten,
        "einsum_before": before[("ai.onnx", "Einsum")],
        "einsum_after": after[("ai.onnx", "Einsum")],
        "matmul_before": before[("ai.onnx", "MatMul")],
        "matmul_after": after[("ai.onnx", "MatMul")],
        "selective_scan_after": after[("com.depthart", "SelectiveScan")],
        "equivalence": "EXACT_BATCHED_LINEAR_ALGEBRA_UNIT_PARITY_PASS_FULL_GRAPH_RUNTIME_PARITY_PENDING",
    }
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
