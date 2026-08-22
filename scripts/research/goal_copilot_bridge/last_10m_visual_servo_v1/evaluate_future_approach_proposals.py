#!/usr/bin/env python3
"""Evaluate semantic proposals against future-demonstrated positive doors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256


KS = (1, 3, 5, 10)


def overlap_metrics(candidate: list[float], target: list[float]) -> dict[str, float | bool]:
    x1, y1 = max(candidate[0], target[0]), max(candidate[1], target[1])
    x2, y2 = min(candidate[2], target[2]), min(candidate[3], target[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    target_area = max(0.0, target[2] - target[0]) * max(0.0, target[3] - target[1])
    candidate_area = max(0.0, candidate[2] - candidate[0]) * max(0.0, candidate[3] - candidate[1])
    target_center = ((target[0] + target[2]) / 2.0, (target[1] + target[3]) / 2.0)
    candidate_center = ((candidate[0] + candidate[2]) / 2.0, (candidate[1] + candidate[3]) / 2.0)
    center_containment = (
        candidate[0] <= target_center[0] <= candidate[2] and candidate[1] <= target_center[1] <= candidate[3]
    ) or (
        target[0] <= candidate_center[0] <= target[2] and target[1] <= candidate_center[1] <= target[3]
    )
    return {"iou": iou(candidate, target), "target_coverage": intersection / target_area if target_area else 0.0, "candidate_coverage": intersection / candidate_area if candidate_area else 0.0, "center_containment": center_containment}


def bearing_action(box: list[float], width: int) -> str:
    center = (box[0] + box[2]) / (2.0 * width)
    return "TURN_LEFT" if center < 0.42 else ("TURN_RIGHT" if center > 0.58 else "ADVANCE")


def evaluate_cases(prediction: dict, private: dict) -> list[dict]:
    predictions = {case["case_id"]: case for case in prediction["cases"]}
    rows = []
    for truth in private["cases"]:
        observed = predictions[truth["case_id"]]
        candidates = sorted(observed["candidates"], key=lambda row: row["provider_rank"])
        targets = truth["legal_targets"]
        hits = []
        all_metrics = []
        for candidate in candidates:
            metrics = [overlap_metrics(candidate["bbox_xyxy"], target["target_bbox_xyxy"]) for target in targets]
            all_metrics.extend(metrics)
            overlaps = [metric["iou"] for metric in metrics]
            best = max(overlaps) if overlaps else 0.0
            matched = overlaps.index(best) if best >= 0.30 else None
            hits.append((matched, best))
        recall = {f"recall_at_{k}": any(matched is not None for matched, _ in hits[:k]) for k in KS}
        first_hit_rank = next((index + 1 for index, (matched, _) in enumerate(hits) if matched is not None), None)
        matched_action = proposal_action = None
        if first_hit_rank is not None:
            matched_index = hits[first_hit_rank - 1][0]
            matched_action = targets[matched_index]["demonstrated_action"]
            proposal_action = bearing_action(candidates[first_hit_rank - 1]["bbox_xyxy"], observed["image_width"])
        rows.append({
            "case_id": truth["case_id"],
            "candidate_count": len(candidates),
            "first_hit_rank": first_hit_rank,
            "first_hit_iou": hits[first_hit_rank - 1][1] if first_hit_rank is not None else None,
            "demonstrated_action": matched_action,
            "proposal_bearing_action": proposal_action,
            "bearing_action_agreement": matched_action == proposal_action if matched_action is not None else None,
            "best_iou_any": max((metric["iou"] for metric in all_metrics), default=0.0),
            "best_target_coverage_any": max((metric["target_coverage"] for metric in all_metrics), default=0.0),
            "best_candidate_coverage_any": max((metric["candidate_coverage"] for metric in all_metrics), default=0.0),
            "any_center_containment": any(metric["center_containment"] for metric in all_metrics),
            **recall,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "future proposal evaluation already exists")
    public, private, prediction = _read(args.public), _read(args.private), _read(args.prediction)
    _require(private.get("positive_only_truth") is True, "future evaluator requires positive-only truth")
    _require(prediction.get("private_truth_access") is False and prediction.get("public_sha256") == sha256(args.public), "future proposal boundary mismatch")
    rows = evaluate_cases(prediction, private)
    count = len(rows)
    recalls = {f"recall_at_{k}": sum(row[f"recall_at_{k}"] for row in rows) / count if count else None for k in KS}
    agreements = [row["bearing_action_agreement"] for row in rows if row["bearing_action_agreement"] is not None]
    bearing_agreement = sum(agreements) / len(agreements) if agreements else None
    passed = count >= 12 and recalls["recall_at_10"] >= 0.80
    payload = {
        "schema_version": "blindassist_future_approach_proposal_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "DEVELOPMENT_ONLY",
        "positive_only_truth": True,
        "unobserved_alternatives_are_not_false_positives": True,
        "public_sha256": sha256(args.public),
        "private_sha256": sha256(args.private),
        "prediction_sha256": sha256(args.prediction),
        "case_count": count,
        **recalls,
        "recalled_case_bearing_action_agreement": bearing_agreement,
        "terminal": "DEV_FUTURE_APPROACH_PROPOSAL_AVAILABLE" if passed else "DEV_FUTURE_APPROACH_PROPOSAL_NOT_AVAILABLE",
        "rows": rows,
    }
    _atomic_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("case_count", "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "recalled_case_bearing_action_agreement", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
