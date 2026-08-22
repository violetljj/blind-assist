#!/usr/bin/env python3
"""Development adapter for aligned current-frame sensor depth fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require, region_depth_median
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.door_consensus_dev import select_consensus
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import decode_depth
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256, validated_box


PROTOCOL_ID = "BLINDASSIST_ALIGNED_SENSOR_DEPTH_FUSION_DEV_V1"


def run(public_path: Path, sealed_run_path: Path, dataset_root: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), "sensor-depth development run already exists")
    public, sealed = _read(public_path), _read(sealed_run_path)
    _require(sealed.get("private_truth_access") is False and sealed.get("public_sha256") == sha256(public_path), "sensor-depth parent boundary mismatch")
    public_cases = {case["case_id"]: case for case in public["cases"]}
    sealed_cases = {case["case_id"]: case for case in sealed["cases"]}
    _require(set(public_cases) == set(sealed_cases), "sensor-depth roster mismatch")
    rows = []
    for case_id in sorted(public_cases):
        environment, trajectory, frame_id = public_cases[case_id]["episode_id"].split("/")
        depth_path = dataset_root / environment / "Data_easy" / trajectory / "depth_lcam_front" / f"{frame_id}_lcam_front_depth.png"
        _require(depth_path.is_file(), "aligned sensor depth missing")
        observed = sealed_cases[case_id]
        depth = decode_depth(depth_path)
        candidates = []
        for candidate in observed["yoloe_candidates"]:
            sensor_depth = region_depth_median(depth, candidate["bbox_xyxy"], observed["image_width"], observed["image_height"])
            candidates.append(dict(candidate) | {"monocular_region_depth_m": candidate["predicted_region_depth_m"], "predicted_region_depth_m": sensor_depth})
        selected = select_consensus(candidates, observed["dino_candidates"], observed["image_width"] / 2.0)
        rows.append({"case_id": case_id, "depth_path": str(depth_path.resolve()), "depth_sha256": sha256(depth_path), "candidates": candidates, "selected_candidate": selected, "completion": selected is not None})
    payload = {"schema_version": "blindassist_aligned_sensor_depth_fusion_dev_run_v1", "protocol_id": PROTOCOL_ID, "public_sha256": sha256(public_path), "sealed_semantic_run_sha256": sha256(sealed_run_path), "private_truth_access": False, "provider_public_aligned_sensor_depth": True, "development_only": True, "threshold_prompt_model_or_rule_sweep": False, "selection_rule": "frozen DINO-YOLOE consensus with aligned sensor depth replacing monocular depth", "cases": rows}
    _atomic_json(output_path, payload)
    return payload


def evaluate(private_path: Path, run_path: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), "sensor-depth development evaluation already exists")
    private, run_payload = _read(private_path), _read(run_path)
    _require(run_payload.get("private_truth_access") is False and run_payload.get("provider_public_aligned_sensor_depth") is True, "sensor-depth boundary mismatch")
    truths = {case["case_id"]: case for case in private["cases"]}
    runs = {case["case_id"]: case for case in run_payload["cases"]}
    _require(set(truths) == set(runs), "sensor-depth evaluation roster mismatch")
    rows = []
    for case_id in sorted(truths):
        truth, observed = truths[case_id], runs[case_id]
        legal = truth["legal_targets"]
        targets = [validated_box(target["target_bbox_xyxy"], f"{case_id} target") for target in legal]
        # TartanAir S2 is 640x640 and the frozen semantic run only permits
        # completion candidates that cover the horizontal center ray.
        opportunity = any(box[0] <= 320.0 <= box[2] and float(target["target_depth_median_m"]) <= 2.0 for box, target in zip(targets, legal, strict=True))
        selected = observed["selected_candidate"]
        matched_index = None
        if selected is not None:
            box = validated_box(selected["bbox_xyxy"], f"{case_id} selected")
            overlaps = [iou(box, target) for target in targets]
            matched_index = int(np.argmax(overlaps)) if overlaps and max(overlaps) >= 0.30 else None
        matched_depth = float(legal[matched_index]["target_depth_median_m"]) if matched_index is not None else None
        correct = selected is not None and matched_depth is not None and matched_depth <= 2.0
        rows.append({"case_id": case_id, "completion_opportunity": opportunity, "completion_decision": selected is not None, "target_selected": matched_index is not None, "correct_completion": correct, "false_completion": selected is not None and not correct, "missed_completion_opportunity": opportunity and not correct})
    opportunities = sum(row["completion_opportunity"] for row in rows)
    correct = sum(row["correct_completion"] for row in rows)
    false = sum(row["false_completion"] for row in rows)
    payload = {"schema_version": "blindassist_aligned_sensor_depth_fusion_dev_evaluation_v1", "protocol_id": PROTOCOL_ID, "case_count": len(rows), "completion_opportunity_count": opportunities, "completion_decision_count": sum(row["completion_decision"] for row in rows), "correct_completion_count": correct, "false_completion_count": false, "missed_completion_opportunity_count": sum(row["missed_completion_opportunity"] for row in rows), "correct_completion_coverage": correct / opportunities if opportunities else None, "rows": rows, "development_only": True, "confirmation_claim_authorized": False, "terminal": "DEV_ALIGNED_SENSOR_DEPTH_FUSION_PROMISING" if false == 0 and opportunities >= 8 and correct / opportunities >= 0.75 else "DEV_ALIGNED_SENSOR_DEPTH_FUSION_NOT_PROMISING"}
    _atomic_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    for name in ("public", "sealed-run", "dataset-root", "output"):
        execute.add_argument(f"--{name}", required=True, type=Path)
    evaluation = sub.add_parser("evaluate")
    for name in ("private", "run", "output"):
        evaluation.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.public, args.sealed_run, args.dataset_root, args.output)
    else:
        result = evaluate(args.private, args.run, args.output)
        print(json.dumps({key: result[key] for key in ("completion_opportunity_count", "correct_completion_count", "false_completion_count", "correct_completion_coverage", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
