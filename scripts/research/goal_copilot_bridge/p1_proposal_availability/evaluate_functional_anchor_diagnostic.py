#!/usr/bin/env python3
"""Post-outcome functional-anchor diagnostic for a consumed PA3 cohort."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import evaluate, iou, sha256, validated_box


def _area(box: list[float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _intersection_area(left: list[float], right: list[float]) -> float:
    return max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(0.0, min(left[3], right[3]) - max(left[1], right[1]))


def anchor_diagnostic(target_box: list[float], candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    target = validated_box(target_box, "functional target")
    anchor = [(target[0] + target[2]) / 2.0, (target[1] + target[3]) / 2.0]
    for candidate in candidates:
        box = validated_box(candidate["bbox_xyxy"], "functional candidate")
        if box[0] <= anchor[0] <= box[2] and box[1] <= anchor[1] <= box[3]:
            return {
                "anchor_xy": anchor,
                "first_containing_rank": int(candidate["rank"]),
                "candidate_bbox_xyxy": box,
                "candidate_to_target_area_ratio": _area(box) / _area(target),
                "target_area_coverage": _intersection_area(box, target) / _area(target),
                "iou": iou(box, target),
            }
    return {
        "anchor_xy": anchor, "first_containing_rank": None, "candidate_bbox_xyxy": None,
        "candidate_to_target_area_ratio": None, "target_area_coverage": 0.0, "iou": 0.0,
    }


def diagnose(public_path: Path, private_path: Path, prediction_path: Path) -> dict[str, Any]:
    sealed_pa3 = evaluate(public_path, private_path, prediction_path)
    private = json.loads(private_path.read_text(encoding="utf-8"))
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    truth = {row["case_id"]: row for row in private["cases"]}
    predicted = {row["case_id"]: row for row in prediction["cases"]}
    rows = []
    for pa3_row in sealed_pa3["rows"]:
        if not pa3_row["primary_evaluable"]:
            continue
        case_id = pa3_row["case_id"]
        target_diagnostics = [
            anchor_diagnostic(box, predicted[case_id]["candidates"])
            for box in truth[case_id]["legal_target_bboxes_xyxy"]
        ]
        ranks = [row["first_containing_rank"] for row in target_diagnostics if row["first_containing_rank"] is not None]
        rows.append({
            "case_id": case_id,
            "first_any_legal_anchor_containing_rank": min(ranks) if ranks else None,
            "targets": target_diagnostics,
        })
    recall = {
        f"anchor_containment_at_{k}": (
            sum(row["first_any_legal_anchor_containing_rank"] is not None and row["first_any_legal_anchor_containing_rank"] <= k for row in rows) / len(rows)
            if rows else None
        )
        for k in (1, 3, 5, 10)
    }
    return {
        "schema_version": "blindassist_p1_functional_anchor_diagnostic_v1",
        "source_pa3_terminal": sealed_pa3["terminal"],
        "source_pa3_primary_iou_recall_at_10": sealed_pa3["candidate_availability"]["recall_at_10"],
        "inputs": {
            "public_input_sha256": sha256(public_path), "private_eval_input_sha256": sha256(private_path),
            "prediction_sha256": sha256(prediction_path),
        },
        "visible_case_count": len(rows),
        "functional_anchor_containment": recall,
        "rows": rows,
        "terminal": "P1_FUNCTIONAL_ANCHOR_POST_OUTCOME_DIAGNOSTIC_COMPLETE",
        "interpretation_rule": "Anchor containment is a representation diagnostic only; large or partial proposals are not promoted to PA3 IoU success.",
        "claim_role": "POST_OUTCOME_CONSUMED_MECHANISM_DIAGNOSTIC_ONLY",
        "claim_ceiling": "FUNCTIONAL_ANCHOR_OBSERVABILITY_DIAGNOSTIC_ONLY_NO_NEW_PROPOSAL_SUCCESS_IDENTITY_GENERALIZATION_PRODUCT_OR_SAFETY_CLAIM",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--private", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("functional-anchor diagnostic already exists")
    payload = diagnose(args.public, args.private, args.prediction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
