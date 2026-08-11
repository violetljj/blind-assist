#!/usr/bin/env python3
"""Independently validate the frozen D2 TRAIN-only outputs and head checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.hftf.deployment.depthart.run_depthart_task_preserving_d2_train_only import (
    chunk_schedule,
    train_head,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not path.exists() and not temporary.exists(), f"output already exists: {path}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def checkpoint_parameter_count(checkpoint: dict[str, Any]) -> int:
    state = checkpoint["state_dict"]
    expected = {
        "layers.0.weight": (16, 11),
        "layers.0.bias": (16,),
        "layers.2.weight": (5, 16),
        "layers.2.bias": (5,),
    }
    require(set(state) == set(expected), "checkpoint state keys drift")
    total = 0
    for name, shape in expected.items():
        values = np.asarray(state[name], dtype=np.float64)
        require(values.shape == shape and np.all(np.isfinite(values)), f"checkpoint tensor drift: {name}")
        total += values.size
    return total


def validate(protocol_path: Path, activation_path: Path, attempt_root: Path) -> dict[str, Any]:
    protocol, activation = load_json(protocol_path), load_json(activation_path)
    result_path = attempt_root / "train-result.json"
    result = load_json(result_path)
    require(result["status"] == "PASS" and
            result["terminal"] == "D2_TRAIN_ONLY_BASE_OUTPUT_AND_HEAD_TRAINING_PASS_HEAD_LOCKED",
            "TRAIN result terminal drift")
    require(result["protocol_sha256"] == sha256(protocol_path), "result/protocol SHA drift")
    require(result["activation_receipt_sha256"] == sha256(activation_path), "result/activation SHA drift")
    require(activation["protocol_sha256"] == sha256(protocol_path), "activation/protocol SHA drift")
    require(result["development_accessed"] is False and result["r2_accessed"] is False
            and result["performance_measured"] is False and result["reference_model_output_accessed"] is False,
            "result authority firewall drift")

    frame_ids: list[str] = []
    output_bytes = 0
    chunk_hashes = []
    expected_device = {key: str(protocol["device"][key])
                       for key in ("build_fingerprint", "model", "device", "soc", "abi")}
    for chunk in chunk_schedule(protocol):
        root = attempt_root / f"chunk-{chunk['chunk_index']:02d}"
        materialization_path = root / "materialization-receipt.json"
        device_path = root / "device-run-receipt.json"
        cleanup_path = root / "input-cleanup-receipt.json"
        materialization, device, cleanup = (
            load_json(materialization_path), load_json(device_path), load_json(cleanup_path)
        )
        require(materialization["chunk"] == device["chunk"] == chunk, "chunk schedule drift")
        require(len(materialization["records"]) == len(device["outputs"]) == 50, "chunk row count drift")
        require(materialization["train_truth_accessed"] is True
                and materialization["reference_model_output_accessed"] is False
                and materialization["development_accessed"] is False and materialization["r2_accessed"] is False,
                "materialization authority drift")
        require(device["device"] == expected_device and device["remote_input_sha256_verified"] is True,
                "device identity or input verification drift")
        require(device["performance_measured"] is False and device["development_accessed"] is False
                and device["r2_accessed"] is False and device["exit_code"] == 0, "device authority drift")
        require(cleanup["materialization_receipt_sha256"] == sha256(materialization_path)
                and cleanup["device_receipt_sha256"] == sha256(device_path), "cleanup receipt drift")
        require(not (root / "inputs").exists(), "generated inputs remain after verified cleanup")
        for record, output in zip(materialization["records"], device["outputs"], strict=True):
            require(record["frame_id"] == output["frame_id"], "frame/output mapping drift")
            path = Path(output["path"])
            require(path.is_file() and path.stat().st_size == 1089536 and output["bytes"] == 1089536,
                    "candidate output byte drift")
            require(sha256(path) == output["sha256"], "candidate output SHA drift")
            values = np.fromfile(path, dtype=np.float32)
            require(values.size == 608 * 448 and np.all(np.isfinite(values)), "candidate output numeric drift")
            frame_ids.append(record["frame_id"])
            output_bytes += path.stat().st_size
        chunk_hashes.append({"chunk_index": chunk["chunk_index"],
                             "materialization_sha256": sha256(materialization_path),
                             "device_run_sha256": sha256(device_path)})
    require(len(frame_ids) == len(set(frame_ids)) == 1200, "frame identity count or uniqueness drift")
    require(chunk_hashes == result["dataset"]["chunk_receipts"], "result chunk hash roster drift")

    dataset_path = Path(result["dataset"]["path"])
    require(dataset_path.is_file() and dataset_path.stat().st_size == result["dataset"]["bytes"]
            and sha256(dataset_path) == result["dataset"]["sha256"], "TRAIN dataset binding drift")
    with np.load(dataset_path, allow_pickle=False) as archive:
        require(set(archive.files) == {"features", "known", "occupied", "raw_clearance",
                                      "truth_clearance", "clearance_paired"}, "dataset keys drift")
        dataset = {name: archive[name] for name in archive.files}
    require(dataset["features"].shape == (3600, 11), "feature dataset shape drift")
    require(dataset["known"].shape == dataset["occupied"].shape == (3600, 3), "cell label shape drift")
    require(dataset["raw_clearance"].shape == dataset["truth_clearance"].shape
            == dataset["clearance_paired"].shape == (3600,), "clearance label shape drift")
    require(all(np.all(np.isfinite(value)) for name, value in dataset.items() if name != "clearance_paired"),
            "dataset non-finite value")

    checkpoint_path = Path(result["checkpoint"]["path"])
    checkpoint = load_json(checkpoint_path)
    require(checkpoint_path.stat().st_size == result["checkpoint"]["bytes"]
            and sha256(checkpoint_path) == result["checkpoint"]["sha256"], "checkpoint binding drift")
    require(checkpoint["step"] == 500 and checkpoint["seed"] == 17
            and checkpoint_parameter_count(checkpoint) == 277, "checkpoint recipe drift")
    reproduced_checkpoint, reproduced_stats = train_head(dataset, steps=500, seed=17)
    require(reproduced_checkpoint == checkpoint, "deterministic checkpoint reproduction drift")
    require(reproduced_stats == result["training"], "deterministic training stats drift")

    restart_receipts = sorted(attempt_root.glob("host-restart-receipt-*.json"))
    for path in restart_receipts:
        receipt = load_json(path)
        require(receipt["candidate_data_policy_or_training_recipe_changed"] is False
                and receipt["development_accessed"] is False and receipt["r2_accessed"] is False,
                "host restart scope drift")
        require(receipt.get("partial_device_output_consumed", False) is False,
                "partial device output was consumed")
    return {
        "schema": "blindassist_depthart_task_preserving_d2_train_only_governed_result_v1",
        "status": "PASS",
        "terminal": "D2_TRAIN_ONLY_GOVERNED_BASE_OUTPUT_AND_HEAD_TRAINING_PASS_HEAD_LOCKED",
        "train_result": {"path": str(result_path.resolve()), "bytes": result_path.stat().st_size,
                         "sha256": sha256(result_path)},
        "checkpoint": result["checkpoint"],
        "dataset": {"path": str(dataset_path.resolve()), "bytes": dataset_path.stat().st_size,
                    "sha256": sha256(dataset_path), "band_rows": 3600, "cell_labels": 10800},
        "verification": {"chunks": 24, "frames": 1200, "candidate_output_bytes": output_bytes,
                         "all_output_hashes_reverified": True, "deterministic_training_reproduced": True,
                         "host_restart_receipts": len(restart_receipts)},
        "development_accessed": False, "r2_accessed": False, "performance_measured": False,
        "next_gate": "EXPLICIT_D2_DEVELOPMENT_BASELINE_AND_FROZEN_HEAD_QUALITY_ACTIVATION",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.protocol.resolve(), args.activation_receipt.resolve(), args.attempt_root.resolve())
    atomic_json(args.output.resolve(), result)
    print(json.dumps({"status": result["status"], "terminal": result["terminal"],
                      "output_sha256": sha256(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
