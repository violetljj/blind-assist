#!/usr/bin/env python3
"""Validate the immutable D2 Development quality terminal and its 24 chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.research.hftf.deployment.depthart.evaluate_depthart_task_preserving_d2_development_quality import (
    PROTOCOL_ID,
    evaluate,
)
from scripts.research.hftf.deployment.depthart.run_depthart_task_preserving_d2_development_chunk import (
    chunk_schedule,
)
from scripts.research.hftf.deployment.depthart.run_depthart_task_preserving_d2_train_only import (
    EXPECTED_DEPTH_BYTES,
    load_json,
    require,
    sha256,
)


def validate(protocol_path: Path, activation_path: Path, output_root: Path) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    activation = load_json(activation_path)
    payload_path = output_root / "development-quality-payload.json"
    result_path = output_root / "development-quality-result.json"
    attempt_path = output_root / "attempt.json"
    payload, result, attempt = load_json(payload_path), load_json(result_path), load_json(attempt_path)
    require(protocol["protocol_id"] == PROTOCOL_ID, "protocol id drift")
    require(activation["protocol_sha256"] == sha256(protocol_path), "activation/protocol drift")
    require(attempt["protocol_sha256"] == sha256(protocol_path), "attempt/protocol drift")
    require(attempt["activation_receipt_sha256"] == sha256(activation_path), "attempt/activation drift")
    require(payload["protocol_sha256"] == sha256(protocol_path), "payload/protocol drift")
    require(payload["activation_receipt_sha256"] == sha256(activation_path), "payload/activation drift")
    require(payload["head_checkpoint_sha256"] == protocol["head_checkpoint"]["sha256"], "payload/head drift")
    require(payload["training_or_tuning"] is False and payload["r2_accessed"] is False
            and payload["performance_measured"] is False, "payload authority drift")
    require(result["status"] == "FAIL"
            and result["terminal"] == "D2_DEVELOPMENT_FROZEN_HEAD_QUALITY_FAIL_STOP",
            "terminal drift")
    require(result["authority"] == {
        "identity_disjoint_development_feasibility": False,
        "r2_candidate_lock": False,
        "r2_access": False,
        "performance": False,
        "android_default": False,
        "production": False,
        "safety": False,
    }, "result authority drift")

    bad_outputs: list[str] = []
    output_count = 0
    output_bytes = 0
    receipt_hashes: list[dict[str, Any]] = []
    for chunk in chunk_schedule(protocol):
        chunk_root = output_root / f"chunk-{chunk['chunk_index']:02d}"
        materialization_path = chunk_root / "materialization-receipt.json"
        device_path = chunk_root / "device-run-receipt.json"
        cleanup_path = chunk_root / "input-cleanup-receipt.json"
        materialization = load_json(materialization_path)
        device = load_json(device_path)
        cleanup = load_json(cleanup_path)
        require(materialization["chunk"] == device["chunk"] == chunk, "chunk mapping drift")
        require(len(materialization["records"]) == len(device["outputs"]) == 50, "chunk count drift")
        require(device["exit_code"] == 0 and device["remote_input_sha256_verified"] is True,
                "device receipt did not pass")
        require(device["training_or_tuning"] is False and device["performance_measured"] is False
                and device["r2_accessed"] is False, "device authority drift")
        require(not (chunk_root / "inputs").exists(), "generated inputs were not cleaned")
        require(cleanup["materialization_receipt_sha256"] == sha256(materialization_path)
                and cleanup["device_receipt_sha256"] == sha256(device_path), "cleanup binding drift")
        for record, output in zip(materialization["records"], device["outputs"], strict=True):
            output_count += 1
            output_bytes += int(output["bytes"])
            path = Path(output["path"])
            if not (record["frame_id"] == output["frame_id"] and path.is_file()
                    and path.stat().st_size == int(output["bytes"]) == EXPECTED_DEPTH_BYTES
                    and sha256(path) == output["sha256"]):
                bad_outputs.append(f"{chunk['chunk_index']}:{output['result_index']}")
        receipt_hashes.append({
            "chunk_index": chunk["chunk_index"],
            "materialization_sha256": sha256(materialization_path),
            "device_sha256": sha256(device_path),
            "cleanup_sha256": sha256(cleanup_path),
        })
    require(not bad_outputs and output_count == 1200 and output_bytes == 1_307_443_200,
            "output set drift")
    require(payload["chunks"] == [
        {
            "chunk_index": row["chunk_index"],
            "materialization_receipt_sha256": row["materialization_sha256"],
            "device_run_receipt_sha256": row["device_sha256"],
            "cleanup_receipt_sha256": row["cleanup_sha256"],
            "output_count": 50,
        }
        for row in receipt_hashes
    ], "payload/chunk receipt binding drift")

    reproduced = evaluate(protocol, payload)
    for key in ("schema", "protocol_id", "status", "terminal", "counts", "baseline", "candidate", "gates", "authority"):
        require(reproduced[key] == result[key], f"re-evaluated {key} drift")
    require(result["bindings"]["payload"]["sha256"] == sha256(payload_path), "result/payload drift")
    require(result["bindings"]["attempt_sha256"] == sha256(attempt_path), "result/attempt drift")
    require(result["outcome_access"] == {
        "development_truth": True,
        "development_saved_context_output": True,
        "baseline_and_frozen_head_quality": True,
        "training_or_tuning": False,
        "r2": False,
        "performance": False,
    }, "outcome authority drift")
    return {
        "schema": "blindassist_depthart_task_preserving_d2_development_quality_validation_v1",
        "status": "PASS",
        "validated_terminal": result["terminal"],
        "protocol_sha256": sha256(protocol_path),
        "activation_receipt_sha256": sha256(activation_path),
        "attempt_sha256": sha256(attempt_path),
        "payload_sha256": sha256(payload_path),
        "result_sha256": sha256(result_path),
        "chunks": 24,
        "frames": 1200,
        "outputs": output_count,
        "output_bytes": output_bytes,
        "all_output_hashes_reverified": True,
        "deterministic_evaluation_reproduced": True,
        "training_or_tuning": False,
        "r2_access": False,
        "performance": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = validate(args.protocol.resolve(), args.activation_receipt.resolve(), args.output_root.resolve())
    require(not args.output.exists(), f"validation output exists: {args.output}")
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
