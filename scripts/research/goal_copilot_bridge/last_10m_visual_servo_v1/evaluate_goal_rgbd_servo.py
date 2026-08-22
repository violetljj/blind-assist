#!/usr/bin/env python3
"""Evaluate positive-only target action availability for the RGB-D servo."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.evaluate_functional_region_availability import functional_region_hit
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import sha256


KS = (1, 3)


def evaluate_cases(prediction: dict, private: dict) -> list[dict]:
    observed = {row["case_id"]: row for row in prediction["cases"]}
    rows = []
    for truth in private["cases"]:
        case = observed[truth["case_id"]]
        candidate_hits, region_hits = [], []
        for candidate in case["candidates"]:
            matches, region_match = [], False
            for target in truth["legal_targets"]:
                hit, metrics = functional_region_hit(candidate["bbox_xyxy"], target["target_bbox_xyxy"], case["image_width"], case["image_height"])
                region_match = region_match or hit
                if hit and candidate["action"] == target["desired_action"]:
                    matches.append(metrics)
            candidate_hits.append(bool(matches))
            region_hits.append(region_match)
        rows.append({
            "case_id": truth["case_id"],
            "phase": truth["phase"],
            "candidate_count": len(case["candidates"]),
            **{f"target_region_recall_at_{k}": any(region_hits[:k]) for k in KS},
            **{f"target_action_recall_at_{k}": any(candidate_hits[:k]) for k in KS},
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("DEVELOPMENT_ONLY", "CONFIRMATION_ONLY"), required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "goal RGB-D servo evaluation already exists")
    private, prediction = _read(args.private), _read(args.prediction)
    _require(private.get("positive_only_truth") is True and prediction.get("private_truth_access") is False, "goal RGB-D servo evaluation boundary mismatch")
    rows = evaluate_cases(prediction, private)
    def recall(phase: str | None, k: int) -> float | None:
        selected = [row for row in rows if phase is None or row["phase"] == phase]
        return sum(row[f"target_action_recall_at_{k}"] for row in selected) / len(selected) if selected else None
    metrics = {
        "target_region_recall_at_1": sum(row["target_region_recall_at_1"] for row in rows) / len(rows) if rows else None,
        "target_region_recall_at_3": sum(row["target_region_recall_at_3"] for row in rows) / len(rows) if rows else None,
        "target_action_recall_at_1": recall(None, 1),
        "target_action_recall_at_3": recall(None, 3),
        "far_target_action_recall_at_3": recall("FAR_GUIDANCE", 3),
        "stop_target_action_recall_at_3": recall("NEAR_STOP", 3),
    }
    passed = len(rows) >= 12 and all(metrics[key] is not None and metrics[key] >= 0.80 for key in ("far_target_action_recall_at_3", "stop_target_action_recall_at_3"))
    payload = {
        "schema_version": "blindassist_goal_rgbd_servo_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": args.role,
        "positive_only_truth": True,
        "unobserved_alternatives_are_not_false_actions": True,
        "prediction_sha256": sha256(args.prediction),
        "private_sha256": sha256(args.private),
        "case_count": len(rows),
        "far_case_count": sum(row["phase"] == "FAR_GUIDANCE" for row in rows),
        "stop_case_count": sum(row["phase"] == "NEAR_STOP" for row in rows),
        **metrics,
        "terminal": "GOAL_RGBD_SERVO_ACTION_AVAILABILITY_ESTABLISHED" if passed else "GOAL_RGBD_SERVO_ACTION_AVAILABILITY_NOT_ESTABLISHED",
        "rows": rows,
    }
    _atomic_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("case_count", "far_case_count", "stop_case_count", "target_region_recall_at_1", "target_region_recall_at_3", "target_action_recall_at_1", "target_action_recall_at_3", "far_target_action_recall_at_3", "stop_target_action_recall_at_3", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
