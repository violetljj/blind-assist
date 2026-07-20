#!/usr/bin/env python3
"""Generate and CUDA-audit analytic ego-motion-compensated TTC track pairs.

The manifest represents already-associated metric target tracks in a body-local frame.  It is a
theory benchmark for the motion/TTC contracts, not a detector, a user trajectory, or a safety
authorization.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "blindassist_ustrf_synthetic_dynamic_ttc_v1"
INTERVAL_NS = 1_000_000_000
HORIZON_SECONDS = 3.0
COLLISION_RADIUS_M = 0.75


def scenarios() -> list[dict[str, Any]]:
    # Previous/current points are expressed in their respective user-local body frames.  The
    # body advances 0.5 m between frames unless a failure row explicitly prevents promotion.
    return [
        {"id": "static", "previous": (3.5, 0.0), "current": (3.0, 0.0), "pose_forward": .5, "verified": True, "confidence": 1.0},
        {"id": "oncoming-central", "previous": (3.5, 0.0), "current": (2.0, 0.0), "pose_forward": .5, "verified": True, "confidence": 1.0},
        {"id": "oncoming-left", "previous": (3.5, -.5), "current": (2.0, -.5), "pose_forward": .5, "verified": True, "confidence": 1.0},
        {"id": "diagonal-collision", "previous": (3.5, -1.0), "current": (2.0, -.5), "pose_forward": .5, "verified": True, "confidence": 1.0},
        {"id": "lateral-cross-clear", "previous": (3.5, -1.6), "current": (3.0, -.8), "pose_forward": .5, "verified": True, "confidence": 1.0},
        {"id": "receding", "previous": (3.5, .4), "current": (4.0, .4), "pose_forward": .5, "verified": True, "confidence": 1.0},
        {"id": "oncoming-right", "previous": (3.5, .5), "current": (2.0, .5), "pose_forward": .5, "verified": True, "confidence": 1.0},
        {"id": "unverified-pose", "previous": (3.5, 0.0), "current": (2.0, 0.0), "pose_forward": .5, "verified": False, "confidence": 1.0},
        {"id": "low-confidence-track", "previous": (3.5, 0.0), "current": (2.0, 0.0), "pose_forward": .5, "verified": True, "confidence": .5},
    ]


def _truth(row: dict[str, Any]) -> dict[str, Any]:
    previous = np.asarray(row["previous"], dtype=np.float32)
    current = np.asarray(row["current"], dtype=np.float32)
    previous_in_current = previous - np.asarray([row["pose_forward"], 0.0], dtype=np.float32)
    velocity = current - previous_in_current
    admitted = bool(row["verified"]) and row["confidence"] >= .70
    if not admitted:
        return {"admitted": False, "previous_in_current": previous_in_current, "velocity": velocity, "ttc_ms": None, "closest_distance_m": None, "collision": None}
    norm_sq = float(velocity @ velocity)
    if norm_sq < .0001:
        return {"admitted": True, "previous_in_current": previous_in_current, "velocity": velocity, "ttc_ms": None, "closest_distance_m": None, "collision": False}
    t = float(np.clip(-(current @ velocity) / norm_sq, 0.0, HORIZON_SECONDS))
    closest = current + velocity * t
    return {"admitted": True, "previous_in_current": previous_in_current, "velocity": velocity, "ttc_ms": int(t * 1000.0), "closest_distance_m": float(np.linalg.norm(closest)), "collision": bool(np.linalg.norm(closest) <= COLLISION_RADIUS_M)}


def generate(root: Path) -> dict[str, Any]:
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing benchmark root: {root}")
    root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios()):
        truth = _truth(scenario)
        rows.append({
            "sequence_id": f"synthetic-dynamic-{index:02d}-{scenario['id']}",
            "track_id": f"target-{index:02d}",
            "previous_frame_id": index * 2,
            "previous_captured_at_ns": 1_000_000_000 + index * 2 * INTERVAL_NS,
            "current_frame_id": index * 2 + 1,
            "current_captured_at_ns": 1_000_000_000 + (index * 2 + 1) * INTERVAL_NS,
            "previous_forward_m": scenario["previous"][0], "previous_lateral_m": scenario["previous"][1],
            "current_forward_m": scenario["current"][0], "current_lateral_m": scenario["current"][1],
            "pose_forward_m": scenario["pose_forward"], "pose_lateral_m": 0.0, "pose_yaw_rad": 0.0,
            "pose_verified": scenario["verified"], "track_confidence": scenario["confidence"],
            "expected_admitted": truth["admitted"],
            "expected_velocity_forward_mps": float(truth["velocity"][0]),
            "expected_velocity_lateral_mps": float(truth["velocity"][1]),
            "expected_ttc_ms": truth["ttc_ms"], "expected_closest_distance_m": truth["closest_distance_m"],
            "expected_collision": truth["collision"],
        })
    columns = list(rows[0])
    with (root / "kotlin_dynamic_ttc_replay.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else str(value).lower() if isinstance(value, bool) else value for key, value in row.items()})
    spec = {"format": SCHEMA, "production_authority": False, "coordinate_frame": "synthetic-body", "interval_ns": INTERVAL_NS, "horizon_seconds": HORIZON_SECONDS, "collision_radius_m": COLLISION_RADIUS_M, "sequence_count": len(rows), "admitted_sequence_count": sum(row["expected_admitted"] for row in rows)}
    (root / "dataset_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec


def audit(root: Path, *, require_cuda: bool) -> dict[str, Any]:
    import torch
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for synthetic dynamic TTC benchmark audit")
    if not torch.cuda.is_available():
        raise RuntimeError("this audit is intentionally GPU-only")
    with (root / "kotlin_dynamic_ttc_replay.tsv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    admitted_rows = [row for row in rows if row["expected_admitted"] == "true"]
    device = torch.device("cuda")
    previous = torch.tensor([[float(row["previous_forward_m"]), float(row["previous_lateral_m"])] for row in admitted_rows], device=device)
    current = torch.tensor([[float(row["current_forward_m"]), float(row["current_lateral_m"])] for row in admitted_rows], device=device)
    pose = torch.tensor([[float(row["pose_forward_m"]), float(row["pose_lateral_m"])] for row in admitted_rows], device=device)
    expected_velocity = torch.tensor([[float(row["expected_velocity_forward_mps"]), float(row["expected_velocity_lateral_mps"])] for row in admitted_rows], device=device)
    velocity = current - (previous - pose)
    velocity_error = torch.linalg.vector_norm(velocity - expected_velocity, dim=1)
    norm_sq = (velocity * velocity).sum(dim=1)
    moving = norm_sq >= .0001
    raw_ttc = -(current * velocity).sum(dim=1) / norm_sq.clamp_min(.0001)
    ttc = raw_ttc.clamp(0.0, HORIZON_SECONDS)
    closest = current + velocity * ttc[:, None]
    collision = torch.linalg.vector_norm(closest, dim=1) <= COLLISION_RADIUS_M
    expected_collision = torch.tensor([row["expected_collision"] == "true" for row in admitted_rows], device=device)
    expected_ttc = torch.tensor([float(row["expected_ttc_ms"] or 0.0) for row in admitted_rows], device=device)
    ttc_error = torch.abs((ttc * 1000.0).floor() - expected_ttc)
    report = {
        "format": "blindassist_ustrf_synthetic_dynamic_ttc_audit_v1",
        "dataset_format": json.loads((root / "dataset_spec.json").read_text(encoding="utf-8"))["format"],
        "sequence_count": len(rows), "admitted_sequence_count": len(admitted_rows), "rejected_sequence_count": len(rows) - len(admitted_rows),
        "moving_admitted_count": int(moving.sum().item()),
        "max_velocity_error_mps": float(velocity_error.max().item()),
        "max_ttc_error_ms": float(ttc_error[moving].max().item()) if bool(moving.any()) else 0.0,
        "collision_label_count": len(admitted_rows), "collision_label_accuracy": float((collision == expected_collision).float().mean().item()),
        "compute_backend": {"name": "torch", "cuda": True, "device": torch.cuda.get_device_name(0)},
        "production_authority": False,
    }
    qa = root / "qa"; qa.mkdir(exist_ok=True)
    (qa / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if not args.audit_only:
        generate(args.output)
    report = audit(args.output, require_cuda=args.require_cuda)
    print(json.dumps({"sequences": report["sequence_count"], "max_ttc_error_ms": report["max_ttc_error_ms"], "backend": report["compute_backend"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
