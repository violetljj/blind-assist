#!/usr/bin/env python3
"""Evaluate bounded functional-region availability for route-demonstrated doors.

This evaluator deliberately keeps ordinary IoU recall separate.  A proposal can
also count as a functional-region hit when it tightly enough contains a small or
truncated demonstrated component, but near-full-frame proposals are rejected.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.evaluate_future_approach_proposals import bearing_action, overlap_metrics
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import sha256


KS = (1, 3, 5, 10)
IOU_THRESHOLD = 0.30
TARGET_COVERAGE_THRESHOLD = 0.80
MAX_CANDIDATE_AREA_FRACTION = 0.40
MIN_CONFIRMATION_CASES = 12
RECALL_AT_10_GATE = 0.80


def functional_region_hit(candidate: list[float], target: list[float], image_width: int, image_height: int) -> tuple[bool, dict]:
    metrics = overlap_metrics(candidate, target)
    candidate_area = max(0.0, candidate[2] - candidate[0]) * max(0.0, candidate[3] - candidate[1])
    image_area = float(image_width * image_height)
    metrics["candidate_area_fraction"] = candidate_area / image_area if image_area else 1.0
    hit = metrics["iou"] >= IOU_THRESHOLD or (
        metrics["target_coverage"] >= TARGET_COVERAGE_THRESHOLD
        and metrics["center_containment"]
        and metrics["candidate_area_fraction"] <= MAX_CANDIDATE_AREA_FRACTION
    )
    return hit, metrics


def evaluate_cases(prediction: dict, private: dict) -> list[dict]:
    predictions = {case["case_id"]: case for case in prediction["cases"]}
    rows = []
    for truth in private["cases"]:
        observed = predictions[truth["case_id"]]
        candidates = sorted(observed["candidates"], key=lambda row: row["provider_rank"])
        targets = truth["legal_targets"]
        hits = []
        for candidate in candidates:
            matches = []
            for target_index, target in enumerate(targets):
                hit, metrics = functional_region_hit(
                    candidate["bbox_xyxy"], target["target_bbox_xyxy"], observed["image_width"], observed["image_height"]
                )
                if hit:
                    matches.append((target_index, metrics))
            matches.sort(key=lambda row: (row[1]["iou"] >= IOU_THRESHOLD, row[1]["iou"], row[1]["target_coverage"]), reverse=True)
            hits.append(matches[0] if matches else None)
        recall = {f"functional_recall_at_{k}": any(hit is not None for hit in hits[:k]) for k in KS}
        first_hit_rank = next((index + 1 for index, hit in enumerate(hits) if hit is not None), None)
        matched_action = proposal_action = match_kind = None
        if first_hit_rank is not None:
            target_index, metrics = hits[first_hit_rank - 1]
            matched_action = targets[target_index]["demonstrated_action"]
            proposal_action = bearing_action(candidates[first_hit_rank - 1]["bbox_xyxy"], observed["image_width"])
            match_kind = "IOU" if metrics["iou"] >= IOU_THRESHOLD else "BOUNDED_CONTAINMENT"
        rows.append({
            "case_id": truth["case_id"],
            "candidate_count": len(candidates),
            "first_functional_hit_rank": first_hit_rank,
            "first_functional_hit_kind": match_kind,
            "demonstrated_action": matched_action,
            "proposal_bearing_action": proposal_action,
            "bearing_action_agreement": matched_action == proposal_action if matched_action is not None else None,
            **recall,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("DEVELOPMENT_ONLY", "CONFIRMATION_ONLY"), required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "functional-region evaluation already exists")
    public, private, prediction = _read(args.public), _read(args.private), _read(args.prediction)
    _require(private.get("positive_only_truth") is True, "functional-region evaluator requires positive-only truth")
    _require(prediction.get("private_truth_access") is False and prediction.get("public_sha256") == sha256(args.public), "functional-region proposal boundary mismatch")
    rows = evaluate_cases(prediction, private)
    count = len(rows)
    recalls = {f"functional_recall_at_{k}": sum(row[f"functional_recall_at_{k}"] for row in rows) / count if count else None for k in KS}
    agreements = [row["bearing_action_agreement"] for row in rows if row["bearing_action_agreement"] is not None]
    bearing_agreement = sum(agreements) / len(agreements) if agreements else None
    passed = count >= MIN_CONFIRMATION_CASES and recalls["functional_recall_at_10"] >= RECALL_AT_10_GATE
    payload = {
        "schema_version": "blindassist_functional_region_availability_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": args.role,
        "positive_only_truth": True,
        "unobserved_alternatives_are_not_false_positives": True,
        "metric_contract": {
            "ordinary_iou_threshold": IOU_THRESHOLD,
            "bounded_containment_target_coverage_threshold": TARGET_COVERAGE_THRESHOLD,
            "bounded_containment_requires_center_containment": True,
            "bounded_containment_max_candidate_area_fraction": MAX_CANDIDATE_AREA_FRACTION,
            "minimum_case_count": MIN_CONFIRMATION_CASES,
            "functional_recall_at_10_gate": RECALL_AT_10_GATE,
        },
        "public_sha256": sha256(args.public),
        "private_sha256": sha256(args.private),
        "prediction_sha256": sha256(args.prediction),
        "case_count": count,
        **recalls,
        "recalled_case_bearing_action_agreement": bearing_agreement,
        "terminal": "FUNCTIONAL_REGION_AVAILABILITY_ESTABLISHED" if passed else "FUNCTIONAL_REGION_AVAILABILITY_NOT_ESTABLISHED",
        "rows": rows,
    }
    _atomic_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("case_count", "functional_recall_at_1", "functional_recall_at_3", "functional_recall_at_5", "functional_recall_at_10", "recalled_case_bearing_action_agreement", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
