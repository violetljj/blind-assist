#!/usr/bin/env python3
"""Evaluate frozen functional-region proposals on a consumed PA3 development cohort."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_grounding_dino_s0_r1 as dino
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import (
    TARGET_VISIBILITY_STATES,
    content_sha256,
    iou,
    private_truth_body,
    sha256,
    validate_precedence_receipt,
    validate_public,
    validated_box,
)
from scripts.research.goal_copilot_bridge.p1_proposal_availability.run_grounding_dino_functional_region import (
    BOUNDED_POOL_SIZE,
    PREDICTION_SCHEMA,
    PROTOCOL_ID,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def evaluate_functional_region(public_path: Path, private_path: Path, prediction_path: Path, prompt_map_path: Path) -> dict[str, Any]:
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    prompt_map = json.loads(prompt_map_path.read_text(encoding="utf-8"))
    validate_public(public, prompt_map, public_path.resolve().parent)
    public_hash = sha256(public_path)
    _require(private.get("public_input_sha256") == public_hash, "private/public binding mismatch")
    _require(prediction.get("schema_version") == PREDICTION_SCHEMA and prediction.get("protocol_id") == PROTOCOL_ID, "FRG1 prediction contract mismatch")
    _require(prediction.get("public_input_sha256") == public_hash and prediction.get("private_truth_access") is False, "FRG1 public/private isolation mismatch")
    provider = prediction.get("provider", {})
    expected = {
        "model_repository": dino.MODEL_REPOSITORY, "model_revision": dino.MODEL_REVISION,
        "weights_sha256": dino.WEIGHTS_SHA256, "prompt": dino.PROMPT,
        "box_threshold": dino.BOX_THRESHOLD, "text_threshold": dino.TEXT_THRESHOLD,
        "nms_iou_threshold": dino.NMS_IOU_THRESHOLD,
        "provider_max_proposals": dino.MAX_PROPOSALS_PER_IMAGE,
        "bounded_pool_size": BOUNDED_POOL_SIZE, "identity_selection": "FORBIDDEN",
        "threshold_or_configuration_sweep": False,
    }
    _require(all(provider.get(key) == value for key, value in expected.items()), "FRG1 frozen provider drift")
    truth_body_hash = content_sha256(private_truth_body(private))
    for case in public["cases"]:
        goal = case["goal_contract"]
        receipt_path = Path(goal["precedence_receipt_path"])
        validate_precedence_receipt(json.loads(receipt_path.read_text(encoding="utf-8")), goal["c0_goal_receipt_body_sha256"])
        _require(json.loads(receipt_path.read_text(encoding="utf-8"))["private_truth_body_sha256"] == truth_body_hash, "precedence/private truth binding mismatch")
    truth = {row["case_id"]: row for row in private["cases"]}
    predicted = {row["case_id"]: row for row in prediction["cases"]}
    public_ids = [row["case_id"] for row in public["cases"]]
    _require(set(public_ids) == set(truth) == set(predicted), "FRG1 case identity mismatch")
    rows = []
    for case_id in public_ids:
        item = truth[case_id]
        visibility = item.get("target_visibility", "VISIBLE")
        _require(visibility in TARGET_VISIBILITY_STATES, "target visibility drift")
        target_boxes = [validated_box(box, f"{case_id} target") for box in item.get("legal_target_bboxes_xyxy", [])]
        candidates = predicted[case_id]["candidates"]
        _require(len(candidates) <= BOUNDED_POOL_SIZE, "FRG1 bounded pool exceeded")
        boxes = [validated_box(row["bbox_xyxy"], f"{case_id} candidate") for row in candidates]
        first_ranks = []
        best_ious = []
        for target in target_boxes:
            overlaps = [iou(box, target) for box in boxes]
            first_ranks.append(next((rank for rank, value in enumerate(overlaps, start=1) if value >= 0.30), None))
            best_ious.append(max(overlaps, default=0.0))
        rows.append({
            "case_id": case_id, "target_visibility": visibility,
            "primary_evaluable": visibility == "VISIBLE" and item["reference_mode"] != "AMBIGUOUS",
            "candidate_count": len(candidates),
            "first_any_legal_rank_iou_0_30": min((rank for rank in first_ranks if rank is not None), default=None),
            "best_iou_by_legal_target": best_ious,
        })
    evaluable = [row for row in rows if row["primary_evaluable"]]
    recall = {f"recall_at_{k}": sum(row["first_any_legal_rank_iou_0_30"] is not None and row["first_any_legal_rank_iou_0_30"] <= k for row in evaluable) / len(evaluable) if evaluable else None for k in (1, 3, 5, 10)}
    terminal = "P1_FRG1_NOT_EVALUABLE" if not evaluable else "P1_FRG1_FUNCTIONAL_REGION_AVAILABILITY_OBSERVED_ON_CONSUMED_COHORT" if recall["recall_at_10"] > 0 else "P1_FRG1_FUNCTIONAL_REGION_AVAILABILITY_NOT_OBSERVED_ON_CONSUMED_COHORT"
    return {
        "schema_version": "blindassist_p1_frg1_evaluation_v1", "protocol_id": PROTOCOL_ID,
        "inputs": {"public_input_sha256": public_hash, "private_eval_input_sha256": sha256(private_path), "prediction_sha256": sha256(prediction_path)},
        "primary_evaluable_case_count": len(evaluable), "candidate_availability_iou_0_30": recall,
        "rows": rows, "terminal": terminal,
        "claim_role": "POST_PA3_OUTCOME_CONSUMED_DEVELOPMENT_MECHANISM_ONLY",
        "claim_ceiling": "FUNCTIONAL_REGION_PROPOSAL_MECHANISM_ONLY_NO_FRESH_CONFIRMATION_IDENTITY_GENERALIZATION_PRODUCT_OR_SAFETY_CLAIM",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--private", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--prompt-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("FRG1 evaluation already exists")
    payload = evaluate_functional_region(args.public, args.private, args.prediction, args.prompt_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
