#!/usr/bin/env python3
"""Development-only open-aperture selection from current-frame RGB-D depth structure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.ade20k_functional_aperture_dev import MIN_APPARENT_HEIGHT_M, MIN_DINO_IOU, MIN_HEIGHT_FRACTION, region_depth_percentile
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import decode_depth
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256, validated_box


PROTOCOL_ID = "BLINDASSIST_DEPTH_APERTURE_DEV_V1"
RUN_SCHEMA = "blindassist_depth_aperture_dev_run_v1"
INTERACTION_BOUNDARY_M = 2.0


def select_aperture(candidates: Sequence[Mapping[str, Any]], dino_candidates: Sequence[Mapping[str, Any]], depth: np.ndarray, width: int, height: int) -> dict[str, Any] | None:
    eligible = []
    for candidate in candidates:
        box = validated_box(candidate["bbox_xyxy"], "YOLOE candidate")
        if not box[0] <= width / 2.0 <= box[2]:
            continue
        height_fraction = (box[3] - box[1]) / height
        near_surface = region_depth_percentile(depth, box, width, height, 20.0)
        interior = region_depth_percentile(depth, box, width, height, 50.0)
        overlaps = [iou(box, validated_box(row["bbox_xyxy"], "DINO candidate")) for row in dino_candidates]
        consensus = max(overlaps, default=0.0)
        if near_surface is None or interior is None:
            continue
        if height_fraction >= MIN_HEIGHT_FRACTION and near_surface <= INTERACTION_BOUNDARY_M < interior and near_surface * height_fraction >= MIN_APPARENT_HEIGHT_M and consensus >= MIN_DINO_IOU:
            eligible.append(dict(candidate) | {"sensor_region_depth_p20_m": near_surface, "sensor_region_depth_p50_m": interior, "depth_aperture_span_m": interior - near_surface, "height_fraction": height_fraction, "apparent_height_proxy_m": near_surface * height_fraction, "dino_consensus_iou": consensus})
    return max(eligible, key=lambda row: (float(row["depth_aperture_span_m"]), float(row["dino_consensus_iou"]), float(row["proposal_score"]))) if eligible else None


def run(public_path: Path, sealed_run_path: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), "depth aperture development output already exists")
    public, sealed = _read(public_path), _read(sealed_run_path)
    _require(sealed.get("private_truth_access") is False and sealed.get("public_sha256") == sha256(public_path), "sealed S4 boundary mismatch")
    public_cases = {case["case_id"]: case for case in public["cases"]}
    sealed_cases = {case["case_id"]: case for case in sealed["cases"]}
    _require(set(public_cases) == set(sealed_cases), "depth aperture roster mismatch")
    rows = []
    for case_id in sorted(public_cases):
        case, observed = public_cases[case_id], sealed_cases[case_id]
        depth_path = Path(case["range_sensor"]["depth_path"])
        _require(depth_path.is_file() and sha256(depth_path) == case["range_sensor"]["depth_sha256"], "public depth drift")
        selected = select_aperture(observed["yoloe_candidates"], observed["dino_candidates"], decode_depth(depth_path), observed["image_width"], observed["image_height"])
        rows.append({"case_id": case_id, "selected_candidate": selected, "completion": selected is not None})
    payload = {"schema_version": RUN_SCHEMA, "protocol_id": PROTOCOL_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "public_sha256": sha256(public_path), "sealed_s4_run_sha256": sha256(sealed_run_path), "private_truth_access": False, "stateless_current_frame_only": True, "provider_public_aligned_sensor_depth": True, "development_only": True, "threshold_prompt_model_or_rule_sweep": False, "selection_rule": "centered tall DINO-consensus candidate whose p20 depth is at or inside and median depth is beyond the 2.0m interaction boundary; rank by depth span", "provider": {"interaction_boundary_m": INTERACTION_BOUNDARY_M, "near_surface_percentile": 20, "interior_percentile": 50, "minimum_height_fraction": MIN_HEIGHT_FRACTION, "minimum_apparent_height_m": MIN_APPARENT_HEIGHT_M, "minimum_dino_iou": MIN_DINO_IOU}, "cases": rows}
    _atomic_json(output_path, payload)
    return payload


def evaluate(private_path: Path, run_path: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), "depth aperture evaluation already exists")
    private, run_payload = _read(private_path), _read(run_path)
    _require(run_payload.get("schema_version") == RUN_SCHEMA and run_payload.get("private_truth_access") is False, "depth aperture evaluation boundary mismatch")
    truths, runs = ({case["case_id"]: case for case in payload["cases"]} for payload in (private, run_payload))
    _require(set(truths) == set(runs), "depth aperture evaluation roster mismatch")
    rows = []
    for case_id in sorted(truths):
        legal, selected = truths[case_id]["legal_targets"], runs[case_id]["selected_candidate"]
        targets = [validated_box(target["target_bbox_xyxy"], f"{case_id} target") for target in legal]
        opportunity = any(box[0] <= 320.0 <= box[2] and float(target["target_depth_median_m"]) <= INTERACTION_BOUNDARY_M for box, target in zip(targets, legal, strict=True))
        matched = None
        if selected is not None:
            overlaps = [iou(validated_box(selected["bbox_xyxy"], f"{case_id} selected"), target) for target in targets]
            matched = int(np.argmax(overlaps)) if overlaps and max(overlaps) >= 0.30 else None
        correct = selected is not None and matched is not None and float(legal[matched]["target_depth_median_m"]) <= INTERACTION_BOUNDARY_M
        rows.append({"case_id": case_id, "completion_opportunity": opportunity, "completion_decision": selected is not None, "target_selected": matched is not None, "correct_completion": correct, "false_completion": selected is not None and not correct, "missed_completion_opportunity": opportunity and not correct})
    opportunities = sum(row["completion_opportunity"] for row in rows)
    correct = sum(row["correct_completion"] for row in rows)
    false = sum(row["false_completion"] for row in rows)
    payload = {"schema_version": "blindassist_depth_aperture_dev_evaluation_v1", "protocol_id": PROTOCOL_ID, "case_count": len(rows), "completion_opportunity_count": opportunities, "completion_decision_count": correct + false, "correct_completion_count": correct, "false_completion_count": false, "missed_completion_opportunity_count": sum(row["missed_completion_opportunity"] for row in rows), "correct_completion_coverage": correct / opportunities if opportunities else None, "rows": rows, "development_only": True, "confirmation_claim_authorized": False, "terminal": "DEV_DEPTH_APERTURE_PROMISING" if false == 0 and opportunities >= 8 and correct / opportunities >= 0.50 else "DEV_DEPTH_APERTURE_NOT_PROMISING"}
    _atomic_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    for name in ("public", "sealed-run", "output"):
        execute.add_argument(f"--{name}", required=True, type=Path)
    evaluation = sub.add_parser("evaluate")
    for name in ("private", "run", "output"):
        evaluation.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.public, args.sealed_run, args.output)
    else:
        result = evaluate(args.private, args.run, args.output)
        print(json.dumps({key: result[key] for key in ("completion_opportunity_count", "correct_completion_count", "false_completion_count", "correct_completion_coverage", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
