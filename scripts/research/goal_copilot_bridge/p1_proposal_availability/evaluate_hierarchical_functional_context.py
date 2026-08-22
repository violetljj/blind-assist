#!/usr/bin/env python3
"""Private evaluator for fresh HRG0 semantic-supported functional-context proposals."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_grounding_dino_s0_r1 as dino
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import (
    EXPECTED_MODEL_SHA256,
    EXPECTED_TEXT_ENCODER_SHA256,
    EXPECTED_ULTRALYTICS_VERSION,
    TARGET_VISIBILITY_STATES,
    content_sha256,
    iou,
    private_truth_body,
    sha256,
    validate_precedence_receipt,
    validate_public,
    validated_box,
)
from scripts.research.goal_copilot_bridge.p1_proposal_availability.run_hierarchical_functional_context import (
    BOUNDED_POOL_SIZE,
    CONTEXT_SCALE,
    PREDICTION_SCHEMA,
    PROTOCOL_ID,
    SEMANTIC_SUPPORT_POOL_SIZE,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def evaluate_hrg0(public_path: Path, private_path: Path, prediction_path: Path, prompt_map_path: Path) -> dict[str, Any]:
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    prompt_map = json.loads(prompt_map_path.read_text(encoding="utf-8"))
    validate_public(public, prompt_map, public_path.resolve().parent)
    public_hash = sha256(public_path)
    _require(private.get("public_input_sha256") == public_hash, "HRG0 private/public binding mismatch")
    _require(prediction.get("schema_version") == PREDICTION_SCHEMA and prediction.get("protocol_id") == PROTOCOL_ID, "HRG0 prediction contract mismatch")
    _require(prediction.get("public_input_sha256") == public_hash and prediction.get("private_truth_access") is False, "HRG0 public/private isolation mismatch")
    provider = prediction.get("provider", {})
    expected = {
        "yoloe_model_sha256": EXPECTED_MODEL_SHA256,
        "text_encoder_sha256": EXPECTED_TEXT_ENCODER_SHA256,
        "ultralytics_version": EXPECTED_ULTRALYTICS_VERSION,
        "grounding_dino_repository": dino.MODEL_REPOSITORY,
        "grounding_dino_revision": dino.MODEL_REVISION,
        "grounding_dino_weights_sha256": dino.WEIGHTS_SHA256,
        "functional_prompt": dino.PROMPT,
        "functional_box_threshold": dino.BOX_THRESHOLD,
        "functional_text_threshold": dino.TEXT_THRESHOLD,
        "functional_nms_iou": dino.NMS_IOU_THRESHOLD,
        "semantic_support_pool_size": SEMANTIC_SUPPORT_POOL_SIZE,
        "functional_context_scale": CONTEXT_SCALE,
        "bounded_pool_size": BOUNDED_POOL_SIZE,
        "ranking": "SEMANTIC_CENTER_SUPPORTED_FIRST_THEN_FUNCTIONAL_PROVIDER_RANK",
        "identity_selection": "FORBIDDEN",
        "threshold_prompt_model_or_scale_sweep": False,
    }
    _require(all(provider.get(key) == value for key, value in expected.items()), "HRG0 frozen provider or hierarchy drift")
    truth_body_hash = content_sha256(private_truth_body(private))
    for case in public["cases"]:
        goal = case["goal_contract"]
        receipt_path = Path(goal["precedence_receipt_path"])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        validate_precedence_receipt(receipt, goal["c0_goal_receipt_body_sha256"])
        _require(receipt["private_truth_body_sha256"] == truth_body_hash, "HRG0 precedence/private truth binding mismatch")
    truth = {row["case_id"]: row for row in private["cases"]}
    predicted = {row["case_id"]: row for row in prediction["cases"]}
    public_ids = [row["case_id"] for row in public["cases"]]
    _require(set(public_ids) == set(truth) == set(predicted), "HRG0 case identity mismatch")
    rows = []
    for case_id in public_ids:
        item = truth[case_id]
        visibility = item.get("target_visibility", "VISIBLE")
        _require(visibility in TARGET_VISIBILITY_STATES, "HRG0 target visibility drift")
        targets = [validated_box(box, f"{case_id} target") for box in item.get("legal_target_bboxes_xyxy", [])]
        candidates = predicted[case_id]["candidates"]
        _require(len(candidates) <= BOUNDED_POOL_SIZE, "HRG0 bounded pool exceeded")
        _require([row["rank"] for row in candidates] == list(range(1, len(candidates) + 1)), "HRG0 candidate rank drift")
        boxes = [validated_box(row["bbox_xyxy"], f"{case_id} candidate") for row in candidates]
        first_ranks = []
        best_ious = []
        for target in targets:
            overlaps = [iou(box, target) for box in boxes]
            first_ranks.append(next((rank for rank, value in enumerate(overlaps, start=1) if value >= 0.30), None))
            best_ious.append(max(overlaps, default=0.0))
        rows.append({
            "case_id": case_id, "target_visibility": visibility,
            "primary_evaluable": visibility == "VISIBLE" and item["reference_mode"] != "AMBIGUOUS",
            "candidate_count": len(candidates),
            "semantic_supported_candidate_count": sum(row["semantic_supported"] is True for row in candidates),
            "first_any_legal_rank_iou_0_30": min((rank for rank in first_ranks if rank is not None), default=None),
            "best_iou_by_legal_target": best_ious,
        })
    evaluable = [row for row in rows if row["primary_evaluable"]]
    recall = {
        f"recall_at_{k}": (
            sum(row["first_any_legal_rank_iou_0_30"] is not None and row["first_any_legal_rank_iou_0_30"] <= k for row in evaluable) / len(evaluable)
            if evaluable else None
        ) for k in (1, 3, 5, 10)
    }
    if not evaluable:
        terminal = "P1_HRG0_NOT_EVALUABLE_NO_VISIBLE_CASES"
    elif recall["recall_at_10"] == 0.0:
        terminal = "P1_HRG0_HIERARCHICAL_TARGET_AVAILABILITY_NOT_OBSERVED"
    elif recall["recall_at_10"] == 1.0:
        terminal = "P1_HRG0_FULL_HIERARCHICAL_TARGET_AVAILABILITY_ON_FRESH_COHORT"
    else:
        terminal = "P1_HRG0_PARTIAL_HIERARCHICAL_TARGET_AVAILABILITY_ON_FRESH_COHORT"
    return {
        "schema_version": "blindassist_p1_hrg0_evaluation_v1", "protocol_id": PROTOCOL_ID,
        "inputs": {"public_input_sha256": public_hash, "private_eval_input_sha256": sha256(private_path), "prediction_sha256": sha256(prediction_path)},
        "primary_evaluable_case_count": len(evaluable),
        "target_not_visible_case_count": sum(row["target_visibility"] == "NOT_VISIBLE" for row in rows),
        "unadjudicable_case_count": sum(row["target_visibility"] == "UNADJUDICABLE" for row in rows),
        "candidate_availability_iou_0_30": recall, "rows": rows, "terminal": terminal,
        "identity_selection": "NOT_EVALUATED",
        "claim_ceiling": "FRESH_HIERARCHICAL_PROPOSAL_AVAILABILITY_ONLY_NO_IDENTITY_GENERALIZATION_PRODUCT_OR_SAFETY_CLAIM",
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
        raise ValueError("HRG0 evaluation already exists")
    payload = evaluate_hrg0(args.public, args.private, args.prediction, args.prompt_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
