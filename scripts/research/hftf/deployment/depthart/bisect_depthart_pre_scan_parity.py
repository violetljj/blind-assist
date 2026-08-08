#!/usr/bin/env python3
"""Prepare and evaluate ORT/HTP probes before the first SelectiveScan node.

Each probe is a dependency-pruned, standard-ONNX subgraph with exactly one
intermediate tensor exposed as its graph output.  This keeps the frozen input
and source graph fixed while allowing a logarithmic search for the first
ORT/HTP numerical divergence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto


SCHEMA = "blindassist_depthart_pre_scan_parity_bisect_v1"
RTOL = 3e-5
ATOL = 3e-6
FROZEN_ORT_VERSION = "1.27.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _shape(value: onnx.ValueInfoProto) -> list[int | str | None]:
    dimensions: list[int | str | None] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return dimensions


def build_plan(model_path: Path) -> dict[str, Any]:
    model = onnx.shape_inference.infer_shapes(onnx.load(model_path))
    scans = [
        (index, node)
        for index, node in enumerate(model.graph.node)
        if node.op_type == "SelectiveScan"
    ]
    if not scans:
        raise ValueError("no SelectiveScan node found")
    scan_index, scan = scans[0]
    target = scan.input[0]
    producers = {
        output: (index, node)
        for index, node in enumerate(model.graph.node)
        for output in node.output
    }
    required_tensors = {target}
    required_nodes: set[int] = set()
    stack = [target]
    while stack:
        tensor = stack.pop()
        producer = producers.get(tensor)
        if producer is None:
            continue
        index, node = producer
        if index in required_nodes:
            continue
        required_nodes.add(index)
        required_tensors.update(node.input)
        stack.extend(node.input)

    values = {
        value.name: value
        for value in (
            list(model.graph.input)
            + list(model.graph.value_info)
            + list(model.graph.output)
        )
    }
    checkpoints: list[dict[str, Any]] = []
    for node_index in sorted(required_nodes):
        node = model.graph.node[node_index]
        if node.op_type in {"Constant", "Shape"}:
            continue
        for output_index, output in enumerate(node.output):
            if output not in required_tensors:
                continue
            value = values.get(output)
            if value is None or value.type.tensor_type.elem_type != TensorProto.FLOAT:
                continue
            dimensions = _shape(value)
            checkpoints.append(
                {
                    "position": len(checkpoints),
                    "node_index": node_index,
                    "output_index": output_index,
                    "node_name": node.name,
                    "op_type": node.op_type,
                    "tensor": output,
                    "shape": dimensions,
                }
            )
    if not checkpoints or checkpoints[-1]["tensor"] != target:
        raise ValueError("failed to enumerate the first SelectiveScan input")

    graph_inputs = [
        value.name
        for value in model.graph.input
        if value.name in required_tensors
    ]
    if graph_inputs != ["image"]:
        raise ValueError(f"unexpected prefix inputs: {graph_inputs}")
    return {
        "schema": SCHEMA,
        "source_onnx": {
            "path": str(model_path.resolve()),
            "bytes": model_path.stat().st_size,
            "sha256": sha256(model_path),
        },
        "first_selective_scan": {
            "node_index": scan_index,
            "node_name": scan.name,
            "first_input": target,
        },
        "prefix_inputs": graph_inputs,
        "prefix_dependency_nodes": len(required_nodes),
        "checkpoints": checkpoints,
        "tolerance": {"rtol": RTOL, "atol": ATOL},
        "authority": "SYNTHETIC_PRE_FIRST_SELECTIVE_SCAN_NUMERICAL_DIAGNOSTIC_ONLY",
        "explicit_exclusions": [
            "G4_E_PARTITION_PURITY",
            "G4_F_PERFORMANCE",
            "DA2_REPLACEMENT",
            "PRODUCTIZATION",
        ],
    }


def prepare(args: argparse.Namespace) -> None:
    model_path = args.onnx.resolve()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "bisect-plan.json"
    plan = build_plan(model_path)
    if ort.__version__ != FROZEN_ORT_VERSION:
        raise RuntimeError(
            f"frozen ORT {FROZEN_ORT_VERSION} required, got {ort.__version__}"
        )
    plan["runtime"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "provider": "CPUExecutionProvider",
    }
    input_identity = {
        "path": "image.raw",
        "bytes": input_path.stat().st_size,
        "sha256": sha256(input_path),
    }
    plan["input"] = input_identity
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        stable_keys = (
            "schema",
            "source_onnx",
            "input",
            "first_selective_scan",
            "prefix_inputs",
            "prefix_dependency_nodes",
            "checkpoints",
            "tolerance",
        )
        if any(existing.get(key) != plan.get(key) for key in stable_keys):
            raise ValueError("existing bisect plan does not match source graph/input")
        plan_path.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        shutil.copy2(input_path, output_dir / "image.raw")
        plan_path.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    checkpoints = plan["checkpoints"]
    positions = sorted(set(args.positions))
    for position in positions:
        if position < 0 or position >= len(checkpoints):
            raise ValueError(
                f"checkpoint position {position} outside 0..{len(checkpoints) - 1}"
            )
        checkpoint = checkpoints[position]
        probe_dir = output_dir / (
            f"probe-{position:03d}-node-{checkpoint['node_index']:04d}-"
            f"{checkpoint['op_type'].lower()}"
        )
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_model = probe_dir / "prefix.onnx"
        if not probe_model.exists():
            onnx.utils.extract_model(
                str(model_path),
                str(probe_model),
                input_names=plan["prefix_inputs"],
                output_names=[checkpoint["tensor"]],
                check_model=True,
            )
        session = ort.InferenceSession(
            str(probe_model), providers=["CPUExecutionProvider"]
        )
        model_input = session.get_inputs()[0]
        input_value = np.fromfile(output_dir / "image.raw", dtype=np.float32)
        input_value = input_value.reshape(model_input.shape)
        ort_output = np.asarray(
            session.run([checkpoint["tensor"]], {model_input.name: input_value})[0],
            dtype=np.float32,
        )
        ort_path = probe_dir / "ort.raw"
        np.ascontiguousarray(ort_output).tofile(ort_path)
        (probe_dir / "input-list.txt").write_text(
            "image:=../image.raw\n", encoding="utf-8"
        )
        receipt = {
            "schema": SCHEMA,
            "checkpoint": checkpoint,
            "source_onnx_sha256": plan["source_onnx"]["sha256"],
            "input_sha256": input_identity["sha256"],
            "probe_onnx": {
                "bytes": probe_model.stat().st_size,
                "sha256": sha256(probe_model),
            },
            "ort_output": {
                "bytes": ort_path.stat().st_size,
                "sha256": sha256(ort_path),
                "minimum": float(ort_output.min()),
                "maximum": float(ort_output.max()),
            },
        }
        (probe_dir / "probe-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(plan, indent=2, sort_keys=True))


def compare(left: Path, right: Path) -> dict[str, Any]:
    a = np.fromfile(left, dtype=np.float32)
    b = np.fromfile(right, dtype=np.float32)
    if a.shape != b.shape or a.size == 0:
        raise ValueError(f"invalid output size: {left} vs {right}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"non-finite output: {left} vs {right}")
    difference = np.abs(a - b)
    return {
        "elements": int(a.size),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(a - b)))),
        "bit_exact": bool(np.array_equal(a, b)),
        "allclose": bool(np.allclose(a, b, rtol=RTOL, atol=ATOL)),
        "left_sha256": sha256(left),
        "right_sha256": sha256(right),
    }


def evaluate(args: argparse.Namespace) -> None:
    root = args.evidence_dir.resolve()
    plan = json.loads((root / "bisect-plan.json").read_text(encoding="utf-8"))
    device_input_sha256 = args.device_input_sha256.upper()
    if device_input_sha256 != plan["input"]["sha256"]:
        raise ValueError("device input SHA-256 does not match the frozen host input")
    results: list[dict[str, Any]] = []
    for probe_dir in sorted(root.glob("probe-*")):
        qnn_path = probe_dir / "htp.raw"
        if not qnn_path.exists():
            continue
        probe = json.loads(
            (probe_dir / "probe-receipt.json").read_text(encoding="utf-8")
        )
        result: dict[str, Any] = {
            "checkpoint": probe["checkpoint"],
            "comparison": compare(probe_dir / "ort.raw", qnn_path),
            "probe_dir": probe_dir.name,
            "assets": {
                "probe_onnx_sha256": probe["probe_onnx"]["sha256"],
                "probe_dlc_sha256": sha256(probe_dir / "prefix.dlc"),
                "execution_metadata_sha256": sha256(
                    probe_dir / "execution_metadata.yaml"
                ),
            },
        }
        cpu_path = probe_dir / "qnn-cpu.raw"
        if cpu_path.exists():
            result["controls"] = {
                "ort_vs_qnn_cpu": compare(probe_dir / "ort.raw", cpu_path),
                "qnn_cpu_vs_htp": compare(cpu_path, qnn_path),
            }
        results.append(result)
    results.sort(key=lambda item: item["checkpoint"]["position"])
    first_failure = next(
        (item for item in results if not item["comparison"]["allclose"]), None
    )
    prior_passes = [
        item
        for item in results
        if item["comparison"]["allclose"]
        and (
            first_failure is None
            or item["checkpoint"]["position"]
            < first_failure["checkpoint"]["position"]
        )
    ]
    precision_observation = None
    if first_failure is not None:
        probe_dir = root / first_failure["probe_dir"]
        ort_value = np.fromfile(probe_dir / "ort.raw", dtype=np.float32)
        htp_value = np.fromfile(probe_dir / "htp.raw", dtype=np.float32)
        ort_fp16 = ort_value.astype(np.float16).astype(np.float32)
        fp16_difference = np.abs(ort_fp16 - htp_value)
        precision_observation = {
            "ort_output_all_fp16_roundtrip_exact": bool(
                np.array_equal(ort_value, ort_fp16)
            ),
            "htp_output_all_fp16_roundtrip_exact": bool(
                np.array_equal(
                    htp_value, htp_value.astype(np.float16).astype(np.float32)
                )
            ),
            "ort_fp16_roundtrip_vs_htp_max_abs": float(fp16_difference.max()),
            "ort_fp16_roundtrip_vs_htp_mean_abs": float(fp16_difference.mean()),
        }
    cpu_control_pass = bool(
        first_failure is not None
        and first_failure.get("controls", {})
        .get("ort_vs_qnn_cpu", {})
        .get("allclose", False)
    )
    receipt = {
        "schema": SCHEMA,
        "status": (
            "FIRST_DIVERGENCE_LOCALIZED"
            if first_failure is not None
            and first_failure["checkpoint"]["position"] == 0
            else "BISECT_INCOMPLETE_OR_BRACKETED"
        ),
        "device": {
            "model": "Samsung SM-S9280",
            "soc": "Qualcomm SM8650",
            "htp": "v75",
            "serial": args.serial,
        },
        "source_onnx": plan["source_onnx"],
        "input": plan["input"],
        "device_input_sha256": device_input_sha256,
        "tolerance": plan["tolerance"],
        "runtime": plan["runtime"],
        "results": results,
        "last_pass_before_first_failure": prior_passes[-1] if prior_passes else None,
        "first_failure": first_failure,
        "precision_observation": precision_observation,
        "interpretation": {
            "same_dlc_qnn_cpu_control": "PASS" if cpu_control_pass else "NOT_EVALUATED_OR_FAIL",
            "first_observable_htp_divergence": (
                first_failure["checkpoint"] if first_failure is not None else None
            ),
            "cause_boundary": (
                "HTP_SPECIFIC_LAYOUT_OR_PRECISION_LOWERING_AT_OR_BEFORE_FIRST_CONV_OUTPUT; "
                "INTERNAL_PRIMITIVE_NOT_YET_PROVEN"
                if cpu_control_pass
                else "NOT_LOCALIZED_TO_HTP_BACKEND"
            ),
        },
        "authority": plan["authority"],
        "explicit_exclusions": plan["explicit_exclusions"],
    }
    output = root / "pre-scan-parity-bisect-receipt.json"
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--onnx", type=Path, required=True)
    prepare_parser.add_argument("--input", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--positions", type=int, nargs="+", required=True)
    prepare_parser.set_defaults(function=prepare)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--evidence-dir", type=Path, required=True)
    evaluate_parser.add_argument("--serial", default="R5CX10M8Y8X")
    evaluate_parser.add_argument("--device-input-sha256", required=True)
    evaluate_parser.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
