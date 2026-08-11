#!/usr/bin/env python3
"""Freeze D2 Development rows and evaluate baseline versus the locked head once."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.arkitscenes_truth_reader import (  # noqa: E402
    TruthReaderPolicy,
    derive_assistive_truth,
)
from scripts.research.hftf.deployment.depthart.depthart_task_preserving_d2_task_head_canary import (  # noqa: E402
    TaskHeadPolicy,
    compose_band,
    sigmoid,
)
from scripts.research.hftf.deployment.depthart.evaluate_depthart_task_preserving_d2_development_quality import (  # noqa: E402
    PROTOCOL_ID,
    SCHEMA,
    evaluate,
)
from scripts.research.hftf.deployment.depthart.run_depthart_task_preserving_d2_development_chunk import (  # noqa: E402
    chunk_schedule,
)
from scripts.research.hftf.deployment.depthart.run_depthart_task_preserving_d2_train_only import (  # noqa: E402
    BANDS,
    EXPECTED_DEPTH_BYTES,
    FEATURE_ORDER,
    HORIZONS,
    atomic_json,
    candidate_features,
    clearance_payload,
    load_json,
    require,
    sha256,
    verify_file_binding,
)


class FrozenTaskHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(11, 16), nn.SiLU(), nn.Linear(16, 5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


def load_head(checkpoint_path: Path) -> tuple[FrozenTaskHead, np.ndarray, np.ndarray, float]:
    checkpoint = load_json(checkpoint_path)
    require(checkpoint["schema"] == "blindassist_depthart_task_preserving_d2_task_head_checkpoint_v1",
            "head checkpoint schema drift")
    require(checkpoint["architecture"] == "Linear(11,16)-SiLU-Linear(16,5)"
            and checkpoint["parameter_count"] == 277, "head architecture drift")
    require(checkpoint["dtype"] == "float64" and checkpoint["device"] == "cpu", "head runtime drift")
    require(checkpoint["feature_order"] == list(FEATURE_ORDER), "head feature order drift")
    require(checkpoint["step"] == 500 and checkpoint["seed"] == 17, "head selection drift")
    model = FrozenTaskHead().to(dtype=torch.float64, device="cpu").eval()
    state = {name: torch.as_tensor(value, dtype=torch.float64) for name, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    require(sum(parameter.numel() for parameter in model.parameters()) == 277, "head parameter count drift")
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float64)
    std = np.asarray(checkpoint["feature_std_population"], dtype=np.float64)
    epsilon = float(checkpoint["standardization_epsilon"])
    require(mean.shape == std.shape == (11,) and epsilon == 1e-6, "head standardization drift")
    return model, mean, std, epsilon


def clearance_arm(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "clearance_valid": bool(value["valid"]),
        "clearance_m": float(value["metres"]) if value["valid"] else None,
        "unknown_reasons": [] if value["valid"] else ["CLEARANCE_UNAVAILABLE"],
    }


def state_arm(value: bool | None, reason: str) -> dict[str, Any]:
    state = "OCCUPIED" if value is True else "CLEAR" if value is False else "UNKNOWN_GROUND"
    return {"state": state, "unknown_reasons": [] if value is not None else [reason]}


def head_state_arm(state: str) -> dict[str, Any]:
    return {"state": state, "unknown_reasons": [] if state != "UNKNOWN_GROUND" else ["HEAD_OR_EVIDENCE_UNKNOWN"]}


def build_rows(protocol: dict[str, Any], output_root: Path, checkpoint_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model, feature_mean, feature_std, epsilon = load_head(checkpoint_path)
    head_policy = TaskHeadPolicy()
    require(protocol["head_composition_policy"] == json.loads(json.dumps(asdict(head_policy))),
            "head composition policy drift")
    rows: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for chunk in chunk_schedule(protocol):
        chunk_root = output_root / f"chunk-{chunk['chunk_index']:02d}"
        materialization_path = chunk_root / "materialization-receipt.json"
        device_path = chunk_root / "device-run-receipt.json"
        cleanup_path = chunk_root / "input-cleanup-receipt.json"
        materialization = load_json(materialization_path)
        device = load_json(device_path)
        cleanup = load_json(cleanup_path)
        require(materialization["chunk"] == device["chunk"] == chunk, "chunk schedule drift")
        require(len(materialization["records"]) == len(device["outputs"]) == 50, "chunk count drift")
        require(cleanup["materialization_receipt_sha256"] == sha256(materialization_path)
                and cleanup["device_receipt_sha256"] == sha256(device_path), "cleanup binding drift")
        require(device["training_or_tuning"] is False and device["performance_measured"] is False
                and device["r2_accessed"] is False, "device authority drift")
        for record, output in zip(materialization["records"], device["outputs"], strict=True):
            require(record["frame_id"] == output["frame_id"], "frame/output mapping drift")
            path = Path(output["path"])
            require(path.is_file() and path.stat().st_size == output["bytes"] == EXPECTED_DEPTH_BYTES
                    and sha256(path) == output["sha256"], "candidate output SHA drift")
            depth = np.fromfile(path, dtype=np.float32).reshape(608, 448)
            require(np.all(np.isfinite(depth)), "non-finite candidate depth")
            geometry = derive_assistive_truth(
                depth,
                np.full(depth.shape, 2, dtype=np.uint8),
                np.asarray(record["intrinsics_tensor"], dtype=np.float64),
                np.asarray(record["up_camera"], dtype=np.float64),
                TruthReaderPolicy(),
            )
            frame_bands: list[dict[str, Any]] = []
            for band_index, band_name in enumerate(BANDS):
                truth = record["truth_bands"][band_index]
                require(truth["band"] == band_name, "truth band order drift")
                base_band = geometry.get("bands", {}).get(band_name) or {}
                base_clearance = clearance_payload(base_band)
                features, evidence = candidate_features(geometry, band_name)
                standardized = (np.asarray(features, dtype=np.float64) - feature_mean) / (feature_std + epsilon)
                require(np.all(np.isfinite(standardized)), "non-finite standardized head feature")
                with torch.inference_mode():
                    logits = model(torch.as_tensor(standardized[None, :], dtype=torch.float64))[0].numpy()
                require(logits.shape == (5,) and np.all(np.isfinite(logits)), "head output drift")
                composed = compose_band(
                    occupancy_logits=logits[2:5],
                    known_probability=sigmoid(float(logits[1])),
                    raw_clearance_m=float(base_clearance["metres"]) if base_clearance["valid"] else None,
                    residual_logit=float(logits[0]),
                    valid_depth_fraction=float(evidence["valid_depth_fraction"]),
                    ground_support_fraction=float(evidence["ground_support_fraction"]),
                    band_support_points=int(evidence["band_support_points"]),
                    ground_plane_available=bool(evidence["ground_plane_available"]),
                    policy=head_policy,
                )
                candidate_clearance = {
                    "valid": composed["clearance_m"] is not None,
                    "metres": composed["clearance_m"],
                }
                cells = []
                for horizon_index, horizon in enumerate(HORIZONS):
                    base_value = base_band.get("occupied_by_horizon", {}).get(str(horizon))
                    cells.append({
                        "horizon_m": horizon,
                        "truth": state_arm(truth["occupied_by_horizon"][horizon_index], "TRUTH_UNKNOWN"),
                        "reference": state_arm(base_value, "BASE_GEOMETRY_UNKNOWN"),
                        "candidate": head_state_arm(composed["states"][horizon_index]),
                    })
                frame_bands.append({
                    "band": band_name,
                    "truth": clearance_arm(truth["clearance"]),
                    "reference": clearance_arm(base_clearance),
                    "candidate": clearance_arm(candidate_clearance),
                    "cells": cells,
                })
            rows.append({
                "parent_id": record["parent_id"], "session_id": record["session_id"],
                "frame_index": record["frame_index"], "frame_id": record["frame_id"],
                "timestamp_ns": record["timestamp_ns"], "orientation": "portrait",
                "bands": frame_bands,
            })
        chunks.append({
            "chunk_index": chunk["chunk_index"],
            "materialization_receipt_sha256": sha256(materialization_path),
            "device_run_receipt_sha256": sha256(device_path),
            "cleanup_receipt_sha256": sha256(cleanup_path),
            "output_count": 50,
        })
    require(len(rows) == 1200, "Development row count drift")
    return rows, chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    protocol_path, activation_path = args.protocol.resolve(), args.activation_receipt.resolve()
    protocol, activation = load_json(protocol_path), load_json(activation_path)
    require(protocol["protocol_id"] == PROTOCOL_ID and protocol["status"] == "FROZEN_DEVELOPMENT_QUALITY_EXECUTION",
            "execution protocol drift")
    require(activation["status"] == "DEVELOPMENT_QUALITY_EXECUTION_ACTIVATED"
            and activation["execution_authorized"] is True, "Development execution is not activated")
    require(activation["protocol_sha256"] == sha256(protocol_path), "activation/protocol drift")
    require(protocol["bindings"]["finalizer"]["sha256"] == sha256(Path(__file__)), "finalizer SHA drift")
    for binding in protocol["bindings"].values():
        if isinstance(binding, dict) and {"path", "bytes", "sha256"}.issubset(binding):
            verify_file_binding(binding)
    checkpoint_path = args.head_checkpoint.resolve()
    require(sha256(checkpoint_path) == protocol["head_checkpoint"]["sha256"], "head checkpoint SHA drift")
    require(protocol["truth_reader_policy"] == json.loads(json.dumps(asdict(TruthReaderPolicy()))),
            "truth reader policy drift")
    output_root = args.output_root.resolve()
    attempt_path = output_root / "attempt.json"
    attempt = load_json(attempt_path)
    require(attempt["protocol_sha256"] == sha256(protocol_path)
            and attempt["activation_receipt_sha256"] == sha256(activation_path), "attempt binding drift")
    payload_path = output_root / "development-quality-payload.json"
    result_path = output_root / "development-quality-result.json"
    require(not payload_path.exists() and not result_path.exists(), "quality output already exists")
    rows, chunks = build_rows(protocol, output_root, checkpoint_path)
    payload = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256(protocol_path),
        "activation_receipt_sha256": sha256(activation_path),
        "head_checkpoint_sha256": sha256(checkpoint_path),
        "chunk_runner_sha256": protocol["bindings"]["chunk_runner"]["sha256"],
        "finalizer_sha256": sha256(Path(__file__)),
        "evaluator_sha256": protocol["bindings"]["evaluator"]["sha256"],
        "truth_reader_sha256": protocol["bindings"]["truth_reader"]["sha256"],
        "chunks": chunks,
        "rows": rows,
        "training_or_tuning": False,
        "r2_accessed": False,
        "performance_measured": False,
    }
    atomic_json(payload_path, payload)
    result = evaluate(protocol, payload)
    result["bindings"] = {
        "protocol_sha256": sha256(protocol_path),
        "activation_receipt_sha256": sha256(activation_path),
        "attempt_sha256": sha256(attempt_path),
        "payload": {"path": str(payload_path.resolve()), "bytes": payload_path.stat().st_size,
                    "sha256": sha256(payload_path)},
        "head_checkpoint_sha256": sha256(checkpoint_path),
        "chunk_receipts": chunks,
    }
    result["outcome_access"] = {
        "development_truth": True, "development_saved_context_output": True,
        "baseline_and_frozen_head_quality": True, "training_or_tuning": False,
        "r2": False, "performance": False,
    }
    atomic_json(result_path, result)
    print(json.dumps({
        "status": result["status"], "terminal": result["terminal"],
        "pooled_baseline": result["baseline"]["pooled"],
        "pooled_candidate": result["candidate"]["pooled"],
        "gates": result["gates"],
        "result_sha256": sha256(result_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
