#!/usr/bin/env python3
"""Materialize and evaluate the frozen DepthART admission R0 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scripts.research.hftf import evaluate_dav2_model_variant_gate_r0 as dav2_r0
from scripts.research.hftf import evaluate_dav2_model_variant_gate_r2 as dav2_r2
from scripts.research.hftf.prepare_bonn_rgbd_metric_depth_manifest import normalize_depth_image
from scripts.research.hftf.evaluate_metric3d_clearance_field_a0 import tum_depth_metres

SCHEMA = "blindassist_depthart_admission_r0_protocol"
RESULT_SCHEMA = "blindassist_depthart_admission_r0_result"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _protocol(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema") != SCHEMA:
        raise ValueError("DepthART admission protocol schema mismatch")
    return payload


def _git_head(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().lower()


def verify_assets(
    protocol: dict[str, Any], source_root: Path, checkpoint: Path
) -> dict[str, Any]:
    expected = protocol["candidate"]["official_assets"]
    source_commit = _git_head(source_root)
    checkpoint_hash = sha256_file(checkpoint)
    checks = {
        "source_commit": source_commit == str(expected["source_commit"]).lower(),
        "checkpoint_sha256": checkpoint_hash
        == str(expected["checkpoint_sha256"]).upper(),
        "checkpoint_size_bytes": checkpoint.stat().st_size
        == int(expected["checkpoint_size_bytes"]),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "observed": {
            "source_commit": source_commit,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_size_bytes": checkpoint.stat().st_size,
        },
    }


def install_timm_compat_shim() -> bool:
    """Bridge the one timm<=0.6 import removed by newer Python-compatible timm."""
    try:
        from timm.models.layers.helpers import to_2tuple  # type: ignore # noqa: F401

        return False
    except (ImportError, ValueError):
        from timm.layers.helpers import to_2tuple  # type: ignore

        module = types.ModuleType("timm.models.layers.helpers")
        module.to_2tuple = to_2tuple
        sys.modules["timm.models.layers.helpers"] = module
        return True


def _load_official_runtime(
    source_root: Path, checkpoint: Path, device: str
) -> tuple[Any, Any, Any, bool]:
    import torch

    compat_shim = install_timm_compat_shim()
    metric_root = source_root / "metric"
    sys.path.insert(0, str(metric_root))
    try:
        from common import make_K, preprocess  # type: ignore
        from model import load_model  # type: ignore
    finally:
        sys.path.pop(0)
    model = load_model(str(checkpoint), "S", "indoor", device)
    model.eval()
    return model, make_K, preprocess, compat_shim


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def materialize(
    protocol_path: Path,
    roster_path: Path,
    source_root: Path,
    official_source: Path,
    checkpoint: Path,
    output: Path,
    receipt_path: Path,
    progress_path: Path,
    device: str,
) -> dict[str, Any]:
    import torch

    protocol = _protocol(protocol_path)
    roster = read_json(roster_path)
    if sha256_file(roster_path) != protocol["cohort"]["roster_sha256"]:
        raise ValueError("roster SHA-256 mismatch")
    rows = roster.get("rows")
    if not isinstance(rows, list) or len(rows) != int(protocol["cohort"]["frames"]):
        raise ValueError("roster row count mismatch")
    assets = verify_assets(protocol, official_source, checkpoint)
    if not assets["passed"]:
        raise ValueError(f"official asset identity mismatch: {assets['checks']}")
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise FileExistsError("candidate output or partial output already exists")

    output.parent.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    cache = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32, shape=(len(rows), 480, 640)
    )
    load_started = time.perf_counter()
    model, make_K, preprocess, compat_shim = _load_official_runtime(
        official_source, checkpoint, device
    )
    load_seconds = time.perf_counter() - load_started
    latencies_ms: list[float] = []
    started = time.perf_counter()
    with progress_path.open("w", encoding="utf-8", buffering=1) as progress:
        for index, row in enumerate(rows):
            rgb_path = source_root / str(row["sequence_root"]) / str(row["rgb_path"])
            if sha256_file(rgb_path) != str(row["rgb_sha256"]).upper():
                raise ValueError(f"RGB SHA-256 mismatch: {row['frame_id']}")
            image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (480, 640):
                raise ValueError(f"unexpected RGB image: {rgb_path}")
            fx, fy, cx, cy = (float(value) for value in row["intrinsics_fx_fy_cx_cy"])
            tensor, intrinsics = preprocess(image, make_K(fx, fy, cx, cy), 640, 480)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            inference_started = time.perf_counter()
            with torch.inference_mode():
                prediction = model(tensor.to(device), intrinsics.to(device))
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - inference_started) * 1000.0
            prediction_np = prediction[0].detach().float().cpu().numpy()
            if prediction_np.shape != (480, 640) or not np.isfinite(prediction_np).all():
                raise ValueError(f"invalid prediction: {row['frame_id']}")
            cache[index] = prediction_np
            cache.flush()
            latencies_ms.append(latency_ms)
            event = {
                "index": index,
                "frame_id": row["frame_id"],
                "latency_ms": latency_ms,
                "minimum_m": float(np.min(prediction_np)),
                "maximum_m": float(np.max(prediction_np)),
            }
            progress.write(json.dumps(event, sort_keys=True) + "\n")
            print(
                f"frame {index + 1}/{len(rows)} latency_ms={latency_ms:.3f}",
                flush=True,
            )
    del cache
    os.replace(partial, output)
    elapsed_seconds = time.perf_counter() - started
    receipt = {
        "schema": "blindassist_depthart_admission_r0_materialization_receipt",
        "protocol_sha256": sha256_file(protocol_path),
        "roster_sha256": sha256_file(roster_path),
        "assets": assets,
        "candidate_id": protocol["candidate"]["candidate_id"],
        "device": device,
        "official_reference_scan_fallback": True,
        "timm_compat_shim_used": compat_shim,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "model_load_seconds": load_seconds,
        "frames": len(rows),
        "output_shape": [len(rows), 480, 640],
        "output_dtype": "float32",
        "output_sha256": sha256_file(output),
        "elapsed_seconds": elapsed_seconds,
        "latency_ms": {
            "median": statistics.median(latencies_ms),
            "p95": float(np.quantile(np.asarray(latencies_ms), 0.95)),
        },
        "claim_ceiling": protocol["claim_ceiling"],
    }
    _write_json(receipt_path, receipt)
    return receipt


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _le(value: Any, ceiling: Any) -> bool:
    return _finite(value) and _finite(ceiling) and float(value) <= float(ceiling)


def _ge(value: Any, floor: Any) -> bool:
    return _finite(value) and _finite(floor) and float(value) >= float(floor)


def decide_gates(
    protocol: dict[str, Any], projection: dict[str, Any], temporal: dict[str, Any]
) -> dict[str, Any]:
    r0 = projection["parent_r0_result"]
    baseline_task = r0["baseline"]["task"]
    candidate_task = r0["candidate"]["task"]
    baseline_truth = projection["baseline_truth_geometry"]
    candidate_truth = projection["candidate_truth_geometry"]
    tolerance = protocol["gate_tolerances"]
    task_gates = {
        "ground_recovery": _ge(
            candidate_task["ground_recovery_success_rate"],
            baseline_task["ground_recovery_success_rate"] - tolerance["ground_recovery_rate_drop"],
        ),
        "clearance_mae": _le(
            candidate_task["clearance_mae_m"],
            baseline_task["clearance_mae_m"] + tolerance["clearance_mae_increase_m"],
        ),
        "collision_agreement": _ge(
            candidate_task["collision_agreement"],
            baseline_task["collision_agreement"] - tolerance["collision_agreement_drop"],
        ),
        "false_clear": _le(
            candidate_task["false_clear_rate_all_known_decisions"],
            baseline_task["false_clear_rate_all_known_decisions"] + tolerance["false_clear_rate_increase"],
        ),
        "truth_status": _ge(
            candidate_truth["truth_status_exact_agreement"],
            baseline_truth["truth_status_exact_agreement"] - tolerance["truth_status_exact_drop"],
        ),
        "false_block": _le(
            candidate_truth["false_block_rate_all_known"],
            baseline_truth["false_block_rate_all_known"] + tolerance["false_block_rate_increase"],
        ),
    }
    temporal_gates = {
        "clearance_delta": _le(
            candidate_task["temporal_clearance_delta_mae_m"],
            baseline_task["temporal_clearance_delta_mae_m"]
            + tolerance["temporal_clearance_delta_mae_increase_m"],
        ),
        "depth_delta": _le(
            temporal["candidate_depth_delta_mae_m"],
            temporal["baseline_depth_delta_mae_m"] + tolerance["temporal_depth_delta_mae_increase_m"],
        ),
        "scale_drift": _le(
            temporal["candidate_scale_drift_median"],
            temporal["baseline_scale_drift_median"] + tolerance["frame_scale_drift_increase"],
        ),
    }
    return {
        "gate_1_task_quality": task_gates,
        "gate_1_passed": all(task_gates.values()),
        "gate_2_temporal_quality": temporal_gates,
        "gate_2_passed": all(temporal_gates.values()),
    }


def _diagnostics(
    roster: dict[str, Any], source_root: Path, baseline_path: Path, candidate_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = np.load(baseline_path, mmap_mode="r")
    candidate = np.load(candidate_path, mmap_mode="r")
    bins = [(0.25, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 6.0)]
    values: dict[str, dict[str, list[float]]] = {
        "baseline": {f"{lo:g}-{hi:g}m": [] for lo, hi in bins},
        "candidate": {f"{lo:g}-{hi:g}m": [] for lo, hi in bins},
    }
    scales = {"baseline": [], "candidate": []}
    delta_errors = {"baseline": [], "candidate": []}
    previous: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for index, row in enumerate(roster["rows"]):
        depth_path = source_root / str(row["sequence_root"]) / str(row["depth_path"])
        truth = tum_depth_metres(normalize_depth_image(cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED), depth_path))
        predictions = {
            "baseline": np.asarray(baseline[index], dtype=np.float32),
            "candidate": np.asarray(candidate[index], dtype=np.float32),
        }
        for name, prediction in predictions.items():
            valid = np.isfinite(truth) & (truth >= 0.25) & (truth <= 6.0) & np.isfinite(prediction) & (prediction > 0)
            if np.any(valid):
                scales[name].append(float(np.median(truth[valid] / prediction[valid])))
            for lo, hi in bins:
                mask = valid & (truth >= lo) & (truth < hi)
                if np.any(mask):
                    values[name][f"{lo:g}-{hi:g}m"].extend(
                        (np.abs(prediction[mask] - truth[mask]) / truth[mask]).astype(np.float64).tolist()
                    )
        sequence = str(row["sequence_id"])
        if sequence in previous:
            prev_truth, prev_baseline, prev_candidate = previous[sequence]
            valid_pair = np.isfinite(truth) & np.isfinite(prev_truth) & (truth >= 0.25) & (truth <= 6.0) & (prev_truth >= 0.25) & (prev_truth <= 6.0)
            truth_delta = truth - prev_truth
            for name, prediction, prev_prediction in (
                ("baseline", predictions["baseline"], prev_baseline),
                ("candidate", predictions["candidate"], prev_candidate),
            ):
                mask = valid_pair & np.isfinite(prediction) & np.isfinite(prev_prediction) & (prediction > 0) & (prev_prediction > 0)
                if np.any(mask):
                    delta_errors[name].append(float(np.mean(np.abs((prediction - prev_prediction)[mask] - truth_delta[mask]))))
        previous[sequence] = (truth, predictions["baseline"], predictions["candidate"])

    depth = {}
    for name in ("baseline", "candidate"):
        depth[name] = {
            "abs_rel_by_truth_range": {
                key: (float(np.median(items)) if items else None)
                for key, items in values[name].items()
            },
            "per_frame_metric_scale_median": statistics.median(scales[name]),
        }
    temporal = {
        "baseline_depth_delta_mae_m": statistics.median(delta_errors["baseline"]),
        "candidate_depth_delta_mae_m": statistics.median(delta_errors["candidate"]),
        "baseline_scale_drift_median": statistics.median(
            abs(b - a) for a, b in zip(scales["baseline"], scales["baseline"][1:])
        ),
        "candidate_scale_drift_median": statistics.median(
            abs(b - a) for a, b in zip(scales["candidate"], scales["candidate"][1:])
        ),
    }
    return depth, temporal


def evaluate(
    protocol_path: Path,
    r2_protocol_path: Path,
    r1_protocol_path: Path,
    r0_protocol_path: Path,
    roster_path: Path,
    source_root: Path,
    baseline_depth_path: Path,
    candidate_depth_path: Path,
) -> dict[str, Any]:
    protocol = _protocol(protocol_path)
    projection = dav2_r2.evaluate(
        r2_protocol_path,
        r1_protocol_path,
        r0_protocol_path,
        roster_path,
        source_root,
        baseline_depth_path,
        candidate_depth_path,
        protocol["candidate"]["candidate_id"],
    )
    roster = read_json(roster_path)
    depth, temporal = _diagnostics(roster, source_root, baseline_depth_path, candidate_depth_path)
    decisions = decide_gates(protocol, projection, temporal)
    quality_passed = decisions["gate_1_passed"] and decisions["gate_2_passed"]
    return {
        "schema": RESULT_SCHEMA,
        "protocol_sha256": sha256_file(protocol_path),
        "candidate_depth_sha256": sha256_file(candidate_depth_path),
        "candidate_id": protocol["candidate"]["candidate_id"],
        "data_role": protocol["data_role"],
        "claim_ceiling": protocol["claim_ceiling"],
        "depth_quality_diagnostics_not_vetoes": depth,
        "temporal_diagnostics": temporal,
        **decisions,
        "gate_3_export": "NOT_EVALUATED",
        "gate_4_snapdragon": "NOT_EVALUATED",
        "quality_admission_passed": quality_passed,
        "backbone_replacement_authorized": False,
        "terminal": (
            "QUALITY_PASS_HOLD_EXPORT_AND_SNAPDRAGON"
            if quality_passed
            else "DEPTHART_S_METRIC_INDOOR_R0_QUALITY_NOT_ADMITTED"
        ),
        "frozen_dav2_r2_projection": projection,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--protocol", type=Path, required=True)
    preflight.add_argument("--official-source", type=Path, required=True)
    preflight.add_argument("--checkpoint", type=Path, required=True)
    materializer = subparsers.add_parser("materialize")
    for name in ("protocol", "roster", "source-root", "official-source", "checkpoint", "output", "receipt", "progress"):
        materializer.add_argument(f"--{name}", type=Path, required=True)
    materializer.add_argument("--device", default="cuda")
    evaluator = subparsers.add_parser("evaluate")
    for name in ("protocol", "r2-protocol", "r1-protocol", "r0-protocol", "roster", "source-root", "baseline-depth", "candidate-depth", "output"):
        evaluator.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = verify_assets(_protocol(args.protocol), args.official_source, args.checkpoint)
    elif args.command == "materialize":
        result = materialize(args.protocol, args.roster, args.source_root, args.official_source, args.checkpoint, args.output, args.receipt, args.progress, args.device)
    else:
        result = evaluate(args.protocol, args.r2_protocol, args.r1_protocol, args.r0_protocol, args.roster, args.source_root, args.baseline_depth, args.candidate_depth)
        _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
