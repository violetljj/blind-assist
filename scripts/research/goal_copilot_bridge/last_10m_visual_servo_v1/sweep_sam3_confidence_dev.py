#!/usr/bin/env python3
"""Bounded development-only confidence sweep over cached SAM 3 proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.functional_region_completion_dev import CONTACT_DEPTH_MAX_M, CONTACT_PIXEL_MIN
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.sam3_functional_region_dev import MASK_HEIGHT_FRACTION_MIN
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256


CONFIDENCE_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


def sweep(prediction: dict, private: dict) -> list[dict]:
    truths = {case["case_id"]: case for case in private["cases"]}
    results = []
    for confidence in CONFIDENCE_GRID:
        opportunity_count = decision_count = correct_count = false_count = 0
        for observed in prediction["cases"]:
            targets = truths[observed["case_id"]]["legal_targets"]
            opportunity = any(
                target["target_bbox_xyxy"][0] <= observed["image_width"] / 2 <= target["target_bbox_xyxy"][2]
                and float(target["target_depth_median_m"]) <= 2.0
                for target in targets
            )
            opportunity_count += int(opportunity)
            candidates = [
                candidate
                for candidate in observed["candidates"]
                if candidate["proposal_score"] >= confidence
                and candidate["bbox_xyxy"][0] <= observed["image_width"] / 2 <= candidate["bbox_xyxy"][2]
                and candidate["mask_height_fraction"] >= MASK_HEIGHT_FRACTION_MIN
                and candidate["ground_contact_pixel_count"] >= CONTACT_PIXEL_MIN
                and candidate["ground_contact_depth_median_m"] is not None
                and candidate["ground_contact_depth_median_m"] <= CONTACT_DEPTH_MAX_M
            ]
            selected = max(candidates, key=lambda row: (row["proposal_score"], row["ground_contact_pixel_count"])) if candidates else None
            decision_count += int(selected is not None)
            correct = selected is not None and any(
                iou(selected["bbox_xyxy"], target["target_bbox_xyxy"]) >= 0.30
                and float(target["target_depth_median_m"]) <= 2.0
                for target in targets
            )
            correct_count += int(correct)
            false_count += int(selected is not None and not correct)
        results.append(
            {
                "confidence_threshold": confidence,
                "opportunity_count": opportunity_count,
                "decision_count": decision_count,
                "correct_count": correct_count,
                "false_count": false_count,
                "coverage": correct_count / opportunity_count if opportunity_count else None,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prediction, private = _read(args.prediction), _read(args.private)
    _require(prediction.get("role") == "DEVELOPMENT_ONLY", "confidence sweep requires development proposals")
    _require(float(prediction["provider"]["confidence_threshold"]) <= min(CONFIDENCE_GRID), "proposal floor exceeds sweep grid")
    rows = sweep(prediction, private)
    eligible = [row for row in rows if row["false_count"] == 0]
    selected = max(eligible, key=lambda row: (row["coverage"], row["confidence_threshold"])) if eligible else None
    payload = {
        "schema_version": "blindassist_sam3_confidence_development_sweep_v1",
        "role": "DEVELOPMENT_ONLY",
        "prediction_sha256": sha256(args.prediction),
        "private_sha256": sha256(args.private),
        "confidence_grid": list(CONFIDENCE_GRID),
        "rows": rows,
        "selected_zero_false_configuration": selected,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
