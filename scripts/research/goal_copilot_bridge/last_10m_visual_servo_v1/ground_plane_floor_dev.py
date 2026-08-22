#!/usr/bin/env python3
"""Evaluate public-depth ground planes against private TartanAir floor labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.ground_plane import ground_mask_from_depth
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import decode_depth
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import sha256


FLOOR_PRECISION_MIN = 0.80
FLOOR_RECALL_MIN = 0.50
MIN_FIT_COUNT = 40


def run_provider(public_path: Path, output_path: Path) -> dict:
    public = _read(public_path)
    masks_dir = output_path.parent / "ground_masks"
    masks_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in public["cases"]:
        depth_path = Path(case["range_sensor"]["depth_path"])
        _require(depth_path.is_file() and sha256(depth_path) == case["range_sensor"]["depth_sha256"], "ground-plane public depth drift")
        depth = decode_depth(depth_path)
        try:
            mask, plane = ground_mask_from_depth(depth)
            error = None
        except ValueError as exc:
            mask = np.zeros(depth.shape, dtype=bool)
            plane, error = None, str(exc)
        mask_path = masks_dir / f"{case['case_id']}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        rows.append({"case_id": case["case_id"], "ground_mask_path": str(mask_path.resolve()), "ground_mask_sha256": sha256(mask_path), "plane": plane, "fit_error": error})
    payload = {"schema_version": "blindassist_ground_plane_floor_development_prediction_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": "DEVELOPMENT_ONLY", "public_sha256": sha256(public_path), "private_truth_access": False, "cases": rows}
    _atomic_json(output_path, payload)
    return payload


def evaluate(public_path: Path, prediction_path: Path, label_map_path: Path, output_path: Path, floor_names: Sequence[str]) -> dict:
    public, prediction, label_map = _read(public_path), _read(prediction_path), _read(label_map_path)
    _require(prediction.get("private_truth_access") is False and prediction.get("public_sha256") == sha256(public_path), "ground-plane provider boundary mismatch")
    name_map = label_map["name_map"]
    floor_ids = {int(name_map[name]) for name in floor_names if name in name_map}
    _require(floor_ids, "no requested private floor labels exist")
    public_cases = {case["case_id"]: case for case in public["cases"]}
    prediction_cases = {case["case_id"]: case for case in prediction["cases"]}
    _require(set(public_cases) == set(prediction_cases), "ground-plane roster mismatch")
    rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for case_id in sorted(public_cases):
        case, observed = public_cases[case_id], prediction_cases[case_id]
        mask_path = Path(observed["ground_mask_path"])
        _require(mask_path.is_file() and sha256(mask_path) == observed["ground_mask_sha256"], "ground mask drift")
        with Image.open(mask_path) as opened:
            predicted = np.asarray(opened.convert("L")) > 0
        seg_path = Path(case["query"]["image_path"]).parent / "seg.png"
        _require(seg_path.is_file(), "private aligned segmentation missing")
        with Image.open(seg_path) as opened:
            segmentation = np.asarray(opened)
        if segmentation.ndim == 3:
            segmentation = segmentation[..., 0]
        truth = np.isin(segmentation, list(floor_ids))
        region = np.zeros_like(truth, dtype=bool)
        region[truth.shape[0] // 2 :, :] = True
        tp = int((predicted & truth & region).sum())
        fp = int((predicted & ~truth & region).sum())
        fn = int((~predicted & truth & region).sum())
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        rows.append({"case_id": case_id, "fit_success": observed["fit_error"] is None, "tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None, "iou": tp / (tp + fp + fn) if tp + fp + fn else None})
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    iou = totals["tp"] / sum(totals.values()) if sum(totals.values()) else 0.0
    fit_count = sum(row["fit_success"] for row in rows)
    payload = {"schema_version": "blindassist_ground_plane_floor_development_evaluation_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": "DEVELOPMENT_ONLY", "prediction_sha256": sha256(prediction_path), "label_map_sha256": sha256(label_map_path), "floor_names": list(floor_names), "fit_count": fit_count, "case_count": len(rows), "pixel_precision": precision, "pixel_recall": recall, "pixel_iou": iou, "totals": totals, "rows": rows, "terminal": "DEV_GROUND_PLANE_FLOOR_PROMISING" if fit_count >= MIN_FIT_COUNT and precision >= FLOOR_PRECISION_MIN and recall >= FLOOR_RECALL_MIN else "DEV_GROUND_PLANE_FLOOR_NOT_PROMISING"}
    _atomic_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--floor-names", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    prediction_path, evaluation_path = args.output / "prediction.json", args.output / "evaluation.json"
    run_provider(args.public, prediction_path)
    result = evaluate(args.public, prediction_path, args.label_map, evaluation_path, args.floor_names)
    print(json.dumps({key: result[key] for key in ("fit_count", "case_count", "pixel_precision", "pixel_recall", "pixel_iou", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
