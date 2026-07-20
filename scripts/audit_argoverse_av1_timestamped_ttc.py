#!/usr/bin/env python3
"""GPU source-native physical-TTC audit for timestamped Argoverse 1 forecasting tracks.

This calculates target motion relative to the AV's instantaneous heading in the source city frame.
It is deliberately not a visual perception evaluation and does not map a vehicle coordinate frame
or collision radius onto a human body.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any

import numpy as np


AV_TRACK_ID = "00000000-0000-0000-0000-000000000000"


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def audit(root: Path, *, horizon_seconds: float = 3.0, vehicle_radius_m: float = 2.0) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Argoverse timestamped TTC audit")
    pairs: list[dict[str, Any]] = []
    source_periods: list[float] = []
    type_counts: Counter[str] = Counter()
    for path in sorted(root.glob("*.csv")):
        by_time: dict[float, dict[str, dict[str, str]]] = defaultdict(dict)
        for row in _load(path):
            by_time[float(row["TIMESTAMP"])][row["TRACK_ID"]] = row
            type_counts[row["OBJECT_TYPE"]] += 1
        timestamps = sorted(by_time)
        for first_t, second_t in zip(timestamps, timestamps[1:]):
            dt = second_t - first_t
            if dt <= 0.0 or AV_TRACK_ID not in by_time[first_t] or AV_TRACK_ID not in by_time[second_t]:
                continue
            ego_previous = np.array([float(by_time[first_t][AV_TRACK_ID]["X"]), float(by_time[first_t][AV_TRACK_ID]["Y"])], dtype=np.float32)
            ego_current = np.array([float(by_time[second_t][AV_TRACK_ID]["X"]), float(by_time[second_t][AV_TRACK_ID]["Y"])], dtype=np.float32)
            heading = ego_current - ego_previous
            norm = float(np.linalg.norm(heading))
            if norm < 1e-4:
                continue
            forward = heading / norm
            left = np.array([-forward[1], forward[0]], dtype=np.float32)
            common = set(by_time[first_t]) & set(by_time[second_t]) - {AV_TRACK_ID}
            for track_id in common:
                first = by_time[first_t][track_id]; second = by_time[second_t][track_id]
                target_previous = np.array([float(first["X"]), float(first["Y"])], dtype=np.float32)
                target_current = np.array([float(second["X"]), float(second["Y"])], dtype=np.float32)
                # Express both observations in the *current* AV local frame to remove ego motion.
                previous_in_current = np.array([np.dot(target_previous - ego_current, forward), np.dot(target_previous - ego_current, left)], dtype=np.float32)
                current_in_current = np.array([np.dot(target_current - ego_current, forward), np.dot(target_current - ego_current, left)], dtype=np.float32)
                if current_in_current[0] <= 0.0:
                    continue
                pairs.append({"scenario": path.stem, "track_id": track_id, "object_type": second["OBJECT_TYPE"], "timestamp_s": second_t, "dt_s": dt, "previous": previous_in_current, "current": current_in_current})
                source_periods.append(dt)
    if not pairs:
        raise ValueError("no timestamped front-facing AV/target track pairs")
    device = torch.device("cuda")
    previous = torch.tensor(np.stack([pair["previous"] for pair in pairs]), device=device)
    current = torch.tensor(np.stack([pair["current"] for pair in pairs]), device=device)
    delta_seconds = torch.tensor([pair["dt_s"] for pair in pairs], device=device)
    velocity = (current - previous) / delta_seconds[:, None]
    norm_sq = (velocity * velocity).sum(dim=1)
    closing_dot = (current * velocity).sum(dim=1)
    approaching = (closing_dot < 0.0) & (norm_sq >= 1e-8)
    ttc = (-closing_dot / norm_sq.clamp_min(1e-8)).clamp(0.0, horizon_seconds)
    closest = current + velocity * ttc[:, None]
    closest_distance = torch.linalg.vector_norm(closest, dim=1)
    within_horizon = approaching & (ttc < horizon_seconds) & (closest_distance <= vehicle_radius_m)
    report = {
        "format": "blindassist_argoverse_av1_source_native_timestamped_ttc_audit_v1",
        "source_root": str(root),
        "scenario_count": len({pair["scenario"] for pair in pairs}),
        "front_facing_track_pair_count": len(pairs),
        "object_type_rows": dict(type_counts),
        "timestamp": {"source_field": "TIMESTAMP", "pair_count": len(source_periods), "median_period_seconds": float(np.median(source_periods)), "min_period_seconds": float(min(source_periods)), "max_period_seconds": float(max(source_periods))},
        "kinematics": {
            "median_ego_compensated_relative_speed_mps": float(torch.linalg.vector_norm(velocity, dim=1).median().item()),
            "p95_ego_compensated_relative_speed_mps": float(torch.quantile(torch.linalg.vector_norm(velocity, dim=1), .95).item()),
            "approaching_track_pair_count": int(approaching.sum().item()),
            "ttc_collision_candidate_count_within_horizon": int(within_horizon.sum().item()),
            "minimum_ttc_seconds_for_approaching": float(ttc[approaching].min().item()) if bool(approaching.any()) else None,
            "median_ttc_seconds_for_approaching": float(ttc[approaching].median().item()) if bool(approaching.any()) else None,
        },
        "source_coordinate_convention": "city-frame positions transformed to current AV forward/left plane using adjacent AV displacement",
        "sample_ttc_candidates": [{"scenario": pairs[index]["scenario"], "track_id": pairs[index]["track_id"], "object_type": pairs[index]["object_type"], "timestamp_s": pairs[index]["timestamp_s"], "ttc_seconds": float(ttc[index].item()), "closest_distance_m": float(closest_distance[index].item())} for index in torch.nonzero(within_horizon, as_tuple=False).flatten().cpu().tolist()[:20]],
        "compute_backend": {"name": "torch", "cuda": True, "device": torch.cuda.get_device_name(0)},
        "source_ttc_seconds_available": True,
        "ustrf_motion_input_admitted": False,
        "reason": "real timestamped source trajectories validate physical TTC arithmetic only; no synchronized RGB-D, camera calibration, or verified human-body mapping is present",
    }
    qa = root / "qa"; qa.mkdir(exist_ok=True)
    (qa / "argoverse_timestamped_ttc_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--horizon-seconds", type=float, default=3.0)
    parser.add_argument("--vehicle-radius-m", type=float, default=2.0)
    args = parser.parse_args()
    report = audit(args.root, horizon_seconds=args.horizon_seconds, vehicle_radius_m=args.vehicle_radius_m)
    print(json.dumps({"pairs": report["front_facing_track_pair_count"], "period_s": report["timestamp"]["median_period_seconds"], "ttc_candidates": report["kinematics"]["ttc_collision_candidate_count_within_horizon"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
