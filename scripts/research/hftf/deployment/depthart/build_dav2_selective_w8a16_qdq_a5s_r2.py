#!/usr/bin/env python3
"""Build the frozen A5S R2 weight-only QDQ ONNX quality candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from scripts.research.hftf.deployment.depthart.generate_dav2_selective_w8a16_overrides_a5_r0 import (
    select_static_linear_weights,
    sha256_file,
)
from onnx import helper, numpy_helper


def quantize_weight(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maximum = np.max(np.abs(weight), axis=0)
    scale = np.maximum(maximum / 127.0, np.finfo(np.float32).tiny).astype(np.float32)
    quantized = np.clip(np.rint(weight / scale[None, :]), -127, 127).astype(np.int8)
    return quantized, scale


def build_qdq_model(model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[dict[str, Any]]]:
    selected = select_static_linear_weights(model)
    selected_by_node = {node_name: (name, weight) for name, weight, node_name in selected}
    selected_names = {name for name, _weight, _node in selected}
    retained_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name not in selected_names
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_initializers)
    rewritten_nodes = []
    records = []
    for node in model.graph.node:
        selected_item = selected_by_node.get(node.name)
        if selected_item is None:
            rewritten_nodes.append(node)
            continue
        tensor_name, weight = selected_item
        quantized, scale = quantize_weight(weight)
        quantized_name = f"{tensor_name}.a5s_int8"
        scale_name = f"{tensor_name}.a5s_scale"
        zero_name = f"{tensor_name}.a5s_zero"
        dequantized_name = f"{tensor_name}.a5s_dequantized"
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(quantized, name=quantized_name),
                numpy_helper.from_array(scale, name=scale_name),
                numpy_helper.from_array(np.zeros(scale.shape, dtype=np.int8), name=zero_name),
            ]
        )
        rewritten_nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [quantized_name, scale_name, zero_name],
                [dequantized_name],
                name=f"{node.name}/A5SWeightDequantize",
                axis=1,
            )
        )
        rewritten = onnx.NodeProto()
        rewritten.CopyFrom(node)
        rewritten.input[1] = dequantized_name
        rewritten_nodes.append(rewritten)
        reconstructed = quantized.astype(np.float32) * scale[None, :]
        records.append(
            {
                "node_name": node.name,
                "source_tensor": tensor_name,
                "quantized_tensor": quantized_name,
                "shape": list(weight.shape),
                "axis": 1,
                "maximum_abs_reconstruction_error": float(np.max(np.abs(reconstructed - weight))),
                "mean_abs_reconstruction_error": float(np.mean(np.abs(reconstructed - weight))),
            }
        )
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    if len(records) != 48:
        raise ValueError(f"A5S R2 expected 48 QDQ weights, got {len(records)}")
    onnx.checker.check_model(model, full_check=True)
    return model, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_A5S_R2_QDQ_MODEL_BUILD":
        raise ValueError("A5S R2 protocol is not frozen")
    bindings = protocol["bindings"]
    if sha256_file(args.onnx) != bindings["onnx_sha256"]:
        raise ValueError("A5S R2 ONNX hash mismatch")
    if sha256_file(Path(__file__).resolve()) != bindings["qdq_builder_source_sha256"]:
        raise ValueError("A5S R2 builder source hash mismatch")
    model = onnx.load(str(args.onnx), load_external_data=True)
    model, records = build_qdq_model(model)
    args.output_root.mkdir(parents=True)
    output_path = args.output_root / "dav2-a5s-r2-selective-w8a16-qdq.onnx"
    onnx.save_model(model, str(output_path))
    receipt = {
        "schema": "blindassist_dav2_selective_w8a16_a5s_r2_qdq_receipt",
        "protocol_sha256": sha256_file(args.protocol),
        "source_onnx_sha256": sha256_file(args.onnx),
        "builder_source_sha256": sha256_file(Path(__file__).resolve()),
        "qdq_onnx": {
            "path": str(output_path.resolve()),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "quantized_weight_count": len(records),
        "activation_quantizers": 0,
        "dequantize_nodes": 48,
        "maximum_abs_reconstruction_error": max(
            record["maximum_abs_reconstruction_error"] for record in records
        ),
        "records": records,
    }
    (args.output_root / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in receipt.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
