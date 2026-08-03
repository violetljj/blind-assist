#!/usr/bin/env python3
"""Evaluate frozen causal dense Metric3D residual propagation on consumed TUM."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from common import (
    REPO_ROOT,
    affine_depth,
    fit_dense_affine,
    frame_key,
    load_json,
    percentile,
    report_frames,
    resolve,
    sha256,
    write_json_new,
)
from torchvision.models.optical_flow import raft_small

HFTF_DIR = REPO_ROOT / "scripts" / "research" / "hftf"
sys.path.insert(0, str(HFTF_DIR))

from evaluate_metric3d_clearance_field_a0 import clearance_field, summarize
from produce_external_rgb_metric_depth_observations import intrinsics_matrix

SCHEMA = "blindassist_dense_metric_depth_propagation_r0_result"
DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "docs/research/hftf/DENSE_METRIC_DEPTH_PROPAGATION_R0_PROTOCOL_2026-08-03.json"
)


def remap(array: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        np.asarray(array, dtype=np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def flow_consistency_mask(
    current_to_anchor: np.ndarray,
    anchor_to_current: np.ndarray,
    threshold_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if current_to_anchor.shape != anchor_to_current.shape:
        raise ValueError("flow shapes differ")
    if current_to_anchor.ndim != 3 or current_to_anchor.shape[0] != 2:
        raise ValueError("flows must be 2xHxW")
    height, width = current_to_anchor.shape[1:]
    columns, rows = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = columns + current_to_anchor[0]
    map_y = rows + current_to_anchor[1]
    inside = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    sampled_forward_x = remap(anchor_to_current[0], map_x, map_y)
    sampled_forward_y = remap(anchor_to_current[1], map_x, map_y)
    error = np.hypot(
        current_to_anchor[0] + sampled_forward_x,
        current_to_anchor[1] + sampled_forward_y,
    )
    valid = inside & np.isfinite(error) & (error <= threshold_px)
    return valid, map_x, map_y


def propagate_residual(
    anchor_fast: np.ndarray,
    anchor_metric: np.ndarray,
    current_fast: np.ndarray,
    current_to_anchor: np.ndarray,
    anchor_to_current: np.ndarray,
    fit: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    base_anchor = affine_depth(anchor_fast, fit)
    base_current = affine_depth(current_fast, fit)
    residual = np.asarray(anchor_metric, dtype=np.float32) - base_anchor
    valid, map_x, map_y = flow_consistency_mask(
        current_to_anchor,
        anchor_to_current,
        float(config["forward_backward_consistency_px_max"]),
    )
    warped_residual = remap(residual, map_x, map_y)
    valid &= np.isfinite(warped_residual)
    coverage = float(np.mean(valid))
    if coverage < float(config["minimum_consistent_residual_coverage"]):
        return None, {"status": "UNKNOWN_FLOW_COVERAGE", "coverage": coverage}
    output = np.where(valid, base_current + warped_residual, base_current)
    output = np.clip(output, 0.0, 300.0).astype(np.float32)
    if not np.all(np.isfinite(output)):
        return None, {"status": "UNKNOWN_PROPAGATED_NONFINITE", "coverage": coverage}
    return output, {
        "status": "VALID",
        "coverage": coverage,
        "residual_pixels": int(np.sum(valid)),
        "da_fill_pixels": int(valid.size - np.sum(valid)),
    }


class RaftRunner:
    def __init__(self, weights_path: Path) -> None:
        self.model = raft_small(weights=None, progress=False)
        self.model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True),
            strict=True,
        )
        self.device = torch.device("cuda")
        self.model.to(self.device).eval()

    def tensor(self, frame_path: str) -> torch.Tensor:
        bgr = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"cannot decode {frame_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .float()
            .div_(127.5)
            .sub_(1.0)
        )

    def flow(self, source: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            value = self.model(
                source[None].to(self.device),
                target[None].to(self.device),
            )[-1][0]
        result = value.float().cpu().numpy()
        if not np.all(np.isfinite(result)):
            raise ValueError("RAFT produced non-finite flow")
        return result


def validate_cache(
    protocol: dict[str, Any], protocol_path: Path, cache_root: Path
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    manifest_path = cache_root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "blindassist_metric_depth_dense_cache_r0":
        raise ValueError("unexpected dense cache schema")
    if manifest.get("protocol_sha256") != sha256(protocol_path):
        raise ValueError("dense cache protocol binding mismatch")
    da_path = Path(manifest["outputs"]["dav2_depth"]["path"])
    metric_path = Path(manifest["outputs"]["metric3d_depth"]["path"])
    for key, path in (("dav2_depth", da_path), ("metric3d_depth", metric_path)):
        if sha256(path) != manifest["outputs"][key]["sha256"]:
            raise ValueError(f"dense cache hash mismatch: {key}")
    da = np.load(da_path, mmap_mode="r")
    metric = np.load(metric_path, mmap_mode="r")
    expected_shape = tuple(protocol["dense_cache"]["shape"])
    if da.shape != expected_shape or metric.shape != expected_shape:
        raise ValueError("dense cache shape mismatch")
    return manifest, da, metric


def load_reports(
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reports = []
    for key in ("metric_report", "fast_report"):
        receipt = protocol["inputs"][key]
        path = resolve(receipt["path"])
        if sha256(path) != receipt["sha256"]:
            raise ValueError(f"bound report mismatch: {key}")
        reports.append(report_frames(load_json(path)))
    metric, fast = reports
    if [frame_key(row) for row in metric] != [frame_key(row) for row in fast]:
        raise ValueError("report frame mismatch")
    return metric, fast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    if protocol.get("status") != "FROZEN_BEFORE_DENSE_OUTPUT_MATERIALIZATION":
        raise ValueError("protocol not frozen")
    manifest, da_depth, metric_depth = validate_cache(
        protocol, args.protocol, args.cache_root
    )
    metric_frames, fast_frames = load_reports(protocol)
    rows_by_path = {str(row["frame_path"]): row for row in manifest["rows"]}
    for row in metric_frames:
        row["intrinsics_fx_fy_cx_cy"] = rows_by_path[str(row["frame_path"])][
            "intrinsics_fx_fy_cx_cy"
        ]
    raft_receipt = protocol["inputs"]["raft"]
    weights = resolve(raft_receipt["weights"])
    if sha256(weights) != raft_receipt["weights_sha256"]:
        raise ValueError("RAFT weights mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("dense propagation requires CUDA")
    raft = RaftRunner(weights)
    config = protocol["candidate"]
    service_s = float(config["metric_service_time_ms"]) / 1000.0
    period = int(config["anchor_period_frames"])
    maximum_age = float(config["maximum_anchor_source_age_s"])
    candidate_rows: list[dict[str, Any]] = []
    trace_rows = []
    coverage_values = []
    age_values = []
    reason_counts: dict[str, int] = {}
    causality_violations = 0
    flow_latency_values = []
    tensors: dict[int, torch.Tensor] = {}

    sequences: dict[str, list[int]] = {}
    for index, row in enumerate(metric_frames):
        sequences.setdefault(str(row["sequence_id"]), []).append(index)
    for sequence, indices in sequences.items():
        active: dict[str, Any] | None = None
        latest: dict[str, Any] | None = None
        for position, index in enumerate(indices):
            now = float(metric_frames[index]["timestamp"])
            if active is not None and float(active["completion_timestamp"]) <= now:
                latest = active
                active = None
            if position % period == 0 and active is None:
                active = {
                    "index": index,
                    "source_timestamp": now,
                    "completion_timestamp": now + service_s,
                }
            candidate = {"status": "UNKNOWN_ANCHOR_STARTUP"}
            diagnostics: dict[str, Any] = {"status": candidate["status"]}
            age = None
            flow_latency = 0.0
            if latest is not None:
                if float(latest["completion_timestamp"]) > now:
                    causality_violations += 1
                age = now - float(latest["source_timestamp"])
                if age > maximum_age:
                    candidate = {"status": "UNKNOWN_STALE_ANCHOR"}
                    diagnostics = {"status": candidate["status"]}
                else:
                    anchor_index = int(latest["index"])
                    fit = fit_dense_affine(
                        da_depth[anchor_index],
                        metric_depth[anchor_index],
                        config["affine_fit"],
                    )
                    if fit["status"] != "VALID":
                        candidate = {"status": fit["status"]}
                        diagnostics = fit
                    else:
                        started = time.perf_counter()
                        if index not in tensors:
                            tensors[index] = raft.tensor(
                                str(metric_frames[index]["frame_path"])
                            )
                        if anchor_index not in tensors:
                            tensors[anchor_index] = raft.tensor(
                                str(metric_frames[anchor_index]["frame_path"])
                            )
                        current_to_anchor = raft.flow(
                            tensors[index], tensors[anchor_index]
                        )
                        anchor_to_current = raft.flow(
                            tensors[anchor_index], tensors[index]
                        )
                        flow_latency = (time.perf_counter() - started) * 1000.0
                        propagated, diagnostics = propagate_residual(
                            da_depth[anchor_index],
                            metric_depth[anchor_index],
                            da_depth[index],
                            current_to_anchor,
                            anchor_to_current,
                            fit,
                            config,
                        )
                        if propagated is not None:
                            candidate = clearance_field(
                                propagated,
                                intrinsics_matrix(metric_frames[index]),
                            )
                            diagnostics["clearance_status"] = candidate["status"]
                            coverage_values.append(float(diagnostics["coverage"]))
                            age_values.append(age)
                        else:
                            candidate = {"status": diagnostics["status"]}
            if candidate.get("status") != "VALID":
                reason = str(candidate.get("status", "UNKNOWN"))
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            flow_latency_values.append(flow_latency)
            candidate_rows.append(
                {
                    "sequence_root": metric_frames[index].get("sequence_root"),
                    "sequence_id": sequence,
                    "timestamp": now,
                    "frame_path": metric_frames[index]["frame_path"],
                    "latency_ms": float(fast_frames[index]["latency_ms"])
                    + flow_latency,
                    "candidate": candidate,
                }
            )
            trace_rows.append(
                {
                    "index": index,
                    "sequence_id": sequence,
                    "timestamp": now,
                    "anchor_index": int(latest["index"])
                    if latest is not None
                    else None,
                    "anchor_completion_timestamp": (
                        float(latest["completion_timestamp"])
                        if latest is not None
                        else None
                    ),
                    "anchor_source_age_s": age,
                    "flow_latency_ms": flow_latency,
                    "candidate_status": candidate.get("status"),
                    "diagnostics": diagnostics,
                }
            )

    joined_rows = []
    for index, row in enumerate(candidate_rows):
        joined = copy.deepcopy(row)
        joined["sensor"] = copy.deepcopy(metric_frames[index]["sensor"])
        joined_rows.append(joined)
    task = summarize(joined_rows)
    compact_task = {key: value for key, value in task.items() if key != "frames"}
    known_fraction = sum(
        row["candidate"].get("status") == "VALID" for row in candidate_rows
    ) / len(candidate_rows)
    system_gates = {
        "known_output_fraction": known_fraction
        >= float(protocol["gates"]["paired_valid_fraction_min"]),
        "causality_violations": causality_violations
        <= int(protocol["gates"]["causality_violations_max"]),
    }
    result = {
        "schema": SCHEMA,
        "protocol_sha256": sha256(args.protocol),
        "cache_manifest_sha256": sha256(args.cache_root / "manifest.json"),
        "data_role": protocol["authority"]["data_role"],
        "fresh_data_opened": False,
        "non_keyframe_metric3d_candidate_reads": 0,
        "task": compact_task,
        "system": {
            "known_output_fraction": known_fraction,
            "anchor_source_age_p95_s": percentile(age_values, 0.95),
            "flow_consistent_coverage_mean": (
                float(np.mean(coverage_values)) if coverage_values else None
            ),
            "flow_consistent_coverage_p05": percentile(coverage_values, 0.05),
            "flow_latency_mean_ms": float(np.mean(flow_latency_values)),
            "flow_latency_p95_ms": percentile(flow_latency_values, 0.95),
            "causality_violations": causality_violations,
            "unknown_reason_counts": reason_counts,
            "gates": system_gates,
        },
        "terminal": (
            "DENSE_PROPAGATION_CONSUMED_DEVELOPMENT_SUPPORTED"
            if all(task["gates"].values()) and all(system_gates.values())
            else "DENSE_PROPAGATION_CONSUMED_DEVELOPMENT_NOT_SUPPORTED"
        ),
        "claim_ceiling": "consumed TUM Development replay only; no fresh, final-camera, phone, alert, safety, or ToF-replacement authority",
    }
    write_json_new(args.output, result)
    write_json_new(
        args.trace_output,
        {
            "schema": "blindassist_dense_metric_depth_propagation_r0_trace",
            "rows": trace_rows,
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
