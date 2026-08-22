"""Contract and evaluator mechanics for P1-PA3 semantic proposals."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


PUBLIC_SCHEMA = "blindassist_p1_pa3_public_input_v1"
PRIVATE_SCHEMA = "blindassist_p1_pa3_private_eval_v1"
PREDICTION_SCHEMA = "blindassist_p1_pa3_prediction_v1"
PRECEDENCE_SCHEMA = "blindassist_p1_pa3_goal_truth_precedence_receipt_v1"
PROMPT_MAP_SCHEMA = "blindassist_p1_pa3_c0_prompt_map_v1"
PROTOCOL_ID = "P1-PA3-GOAL-SEMANTIC-PROPOSAL-AVAILABILITY-V1"
EXPECTED_MODEL_SHA256 = "1741c1f8da3cea47e2c01829c334a50dc0b9bbd05e685b90a3ce84fae32c8c1b"
EXPECTED_TEXT_ENCODER_SHA256 = "35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f"
EXPECTED_ULTRALYTICS_VERSION = "8.4.52"
EXPECTED_PROVIDER_CONFIGURATION = {
    "text_encoder_sha256": EXPECTED_TEXT_ENCODER_SHA256,
    "imgsz": 640,
    "confidence_floor": 0.001,
    "provider_max_det": 100,
    "bounded_pool_size": 10,
    "identity_selection": "FORBIDDEN",
    "threshold_or_configuration_sweep": False,
}
REFERENCE_MODES = {"UNIQUE", "SET_VALUED", "AMBIGUOUS"}
TARGET_VISIBILITY_STATES = {"VISIBLE", "NOT_VISIBLE", "UNADJUDICABLE"}
PRECEDENCE_MODES = {
    "PHYSICAL_CAPTURE_AFTER_GOAL",
    "GOAL_BEFORE_FIRST_PROJECT_PIXEL_ACCESS_AND_TRUTH",
}
FORBIDDEN_PUBLIC_KEYS = {
    "target_bbox_xyxy",
    "target_bboxes_xyxy",
    "target_mask",
    "target_category",
    "instance_name",
    "object_uid",
    "referent_id",
    "truth",
    "evaluator",
}


class Pa3ContractError(ValueError):
    """Raised when PA3 input would violate its public/private or precedence contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pa3ContractError(message)


def _text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be non-empty text")
    return value.strip()


def _reject_forbidden_public_keys(value: Any, path: str = "public") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require(str(key) not in FORBIDDEN_PUBLIC_KEYS, f"{path}.{key} is evaluator/identity truth")
            _reject_forbidden_public_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_public_keys(child, f"{path}[{index}]")


def validated_box(value: Any, label: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == 4, f"{label} must be four-value XYXY")
    box = [float(item) for item in value]
    _require(all(math.isfinite(item) for item in box), f"{label} must be finite")
    _require(box[2] > box[0] and box[3] > box[1], f"{label} must have positive area")
    return box


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def prompt_lookup(prompt_map: Mapping[str, Any]) -> dict[str, str]:
    _require(prompt_map.get("schema_version") == PROMPT_MAP_SCHEMA, "prompt-map schema mismatch")
    _require(prompt_map.get("mapping_rule") == "EXACT_GLOBAL_GOAL_TYPE_LOOKUP_NO_EPISODE_OVERRIDE", "prompt-map rule mismatch")
    entries = prompt_map.get("entries")
    _require(isinstance(entries, list) and bool(entries), "prompt map must contain entries")
    lookup: dict[str, str] = {}
    for entry in entries:
        _require(isinstance(entry, Mapping), "prompt-map entry must be an object")
        goal_type = _text(entry.get("goal_type"), "goal_type")
        prompt = _text(entry.get("canonical_prompt"), "canonical_prompt")
        _require(goal_type not in lookup, f"duplicate prompt mapping for {goal_type}")
        lookup[goal_type] = prompt
    return lookup


def _utc_timestamp(value: Any, label: str) -> datetime:
    text = _text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise Pa3ContractError(f"{label} must be ISO-8601") from error
    _require(parsed.tzinfo is not None and parsed.utcoffset() is not None, f"{label} must include timezone")
    return parsed


def private_truth_body(private: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": private.get("protocol_id"),
        "primary_iou_threshold": private.get("primary_iou_threshold"),
        "diagnostic_iou_thresholds": private.get("diagnostic_iou_thresholds"),
        "recall_at_k": private.get("recall_at_k"),
        "cases": private.get("cases"),
    }


def validate_precedence_receipt(receipt: Mapping[str, Any], expected_body_sha256: str) -> None:
    _require(receipt.get("schema_version") == PRECEDENCE_SCHEMA, "precedence receipt schema mismatch")
    _require(receipt.get("goal_receipt_body_sha256") == expected_body_sha256, "goal receipt body hash mismatch")
    precedence_mode = receipt.get("precedence_mode", "PHYSICAL_CAPTURE_AFTER_GOAL")
    _require(precedence_mode in PRECEDENCE_MODES, "precedence mode is invalid")
    if precedence_mode == "PHYSICAL_CAPTURE_AFTER_GOAL":
        _require(receipt.get("created_before_capture") is True, "goal-before-capture is not confirmed")
    else:
        _require(receipt.get("created_before_project_pixel_access") is True, "goal-before-project-pixel-access is not confirmed")
        _require(receipt.get("physical_capture_after_goal_claimed") is False, "public-source cohort must not claim post-goal physical capture")
    _require(receipt.get("created_before_truth") is True, "goal-before-truth is not confirmed")
    goal_at = _utc_timestamp(receipt.get("goal_recorded_at_utc"), "goal_recorded_at_utc")
    capture_at = _utc_timestamp(receipt.get("capture_created_at_utc"), "capture_created_at_utc")
    truth_at = _utc_timestamp(receipt.get("truth_created_at_utc"), "truth_created_at_utc")
    _require(goal_at < capture_at, "goal timestamp does not precede capture/access")
    _require(goal_at < truth_at, "goal timestamp does not precede truth")
    private_hash = _text(receipt.get("private_truth_body_sha256"), "private_truth_body_sha256")
    _require(len(private_hash) == 64 and all(character in "0123456789abcdef" for character in private_hash), "private truth hash must be lowercase SHA-256")


def validate_public(public: Mapping[str, Any], prompt_map: Mapping[str, Any], base_dir: Path) -> list[dict[str, Any]]:
    _require(public.get("schema_version") == PUBLIC_SCHEMA, "public schema mismatch")
    _require(public.get("protocol_id") == PROTOCOL_ID, "protocol mismatch")
    _require(public.get("private_truth_access") is False, "public input must deny private truth access")
    _reject_forbidden_public_keys(public)
    contract = public.get("provider_contract")
    _require(isinstance(contract, Mapping), "provider_contract is required")
    _require(contract.get("input") == "CURRENT_FRAME_PLUS_PRETRUTH_GOAL_CONTRACT", "provider input contract drift")
    _require(contract.get("maximum_candidates") == 10, "bounded candidate cap must be ten")
    _require(contract.get("identity_selection") == "FORBIDDEN", "identity selection must be forbidden")
    _require(public.get("prompt_map_sha256") == content_sha256(prompt_map), "prompt-map hash mismatch")
    lookup = prompt_lookup(prompt_map)
    cases = public.get("cases")
    _require(isinstance(cases, list) and bool(cases), "PA3 public cohort must be non-empty")
    case_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for raw in cases:
        _require(isinstance(raw, Mapping), "public case must be an object")
        case_id = _text(raw.get("case_id"), "case_id")
        _require(case_id not in case_ids, f"duplicate case_id: {case_id}")
        case_ids.add(case_id)
        query = raw.get("query")
        goal = raw.get("goal_contract")
        _require(isinstance(query, Mapping) and isinstance(goal, Mapping), f"{case_id} query and goal_contract are required")
        image_path = Path(_text(query.get("image_path"), f"{case_id} image_path"))
        if not image_path.is_absolute():
            image_path = (base_dir / image_path).resolve()
        _require(image_path.is_file(), f"{case_id} query image is missing")
        _require(sha256(image_path) == query.get("image_sha256"), f"{case_id} query image hash mismatch")
        goal_type = _text(goal.get("goal_type"), f"{case_id} goal_type")
        reference_mode = _text(goal.get("reference_mode"), f"{case_id} reference_mode")
        _require(reference_mode in REFERENCE_MODES, f"{case_id} reference_mode is invalid")
        _require(goal_type in lookup, f"{case_id} has no globally frozen semantic prompt")
        _require(goal.get("canonical_prompt") == lookup[goal_type], f"{case_id} canonical prompt override")
        _text(goal.get("goal_text_original"), f"{case_id} goal_text_original")
        body_sha = _text(goal.get("c0_goal_receipt_body_sha256"), f"{case_id} C0 body hash")
        _require(len(body_sha) == 64 and all(character in "0123456789abcdef" for character in body_sha), f"{case_id} C0 body hash must be lowercase SHA-256")
        receipt_path = Path(_text(goal.get("precedence_receipt_path"), f"{case_id} precedence receipt path"))
        if not receipt_path.is_absolute():
            receipt_path = (base_dir / receipt_path).resolve()
        _require(receipt_path.is_file(), f"{case_id} precedence receipt is missing")
        _require(sha256(receipt_path) == goal.get("precedence_receipt_sha256"), f"{case_id} precedence receipt hash mismatch")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        validate_precedence_receipt(receipt, body_sha)
        validated.append({
            "case_id": case_id,
            "image_path": image_path,
            "goal_type": goal_type,
            "reference_mode": reference_mode,
            "canonical_prompt": lookup[goal_type],
            "precedence_receipt_sha256": sha256(receipt_path),
        })
    return validated


def evaluate(public_path: Path, private_path: Path, prediction_path: Path) -> dict[str, Any]:
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    _require(private.get("schema_version") == PRIVATE_SCHEMA, "private schema mismatch")
    _require(prediction.get("schema_version") == PREDICTION_SCHEMA, "prediction schema mismatch")
    _require(private.get("protocol_id") == PROTOCOL_ID and prediction.get("protocol_id") == PROTOCOL_ID, "protocol mismatch")
    public_hash = sha256(public_path)
    _require(private.get("public_input_sha256") == public_hash, "private input is not bound to public input")
    _require(prediction.get("public_input_sha256") == public_hash, "prediction is not bound to public input")
    _require(prediction.get("private_truth_access") is False, "provider must declare zero private truth access")
    provider = prediction.get("provider")
    _require(isinstance(provider, Mapping), "prediction provider receipt is missing")
    _require(provider.get("model_sha256") == EXPECTED_MODEL_SHA256, "PA3 model identity drift")
    _require(provider.get("ultralytics_version") == EXPECTED_ULTRALYTICS_VERSION, "PA3 provider version drift")
    for key, expected in EXPECTED_PROVIDER_CONFIGURATION.items():
        _require(provider.get(key) == expected, f"PA3 provider configuration drift: {key}")
    truth_body_sha256 = content_sha256(private_truth_body(private))
    for case in public.get("cases", []):
        goal = case.get("goal_contract", {})
        receipt_path = Path(_text(goal.get("precedence_receipt_path"), f"{case.get('case_id')} precedence receipt path"))
        if not receipt_path.is_absolute():
            receipt_path = (public_path.resolve().parent / receipt_path).resolve()
        _require(receipt_path.is_file(), f"{case.get('case_id')} precedence receipt is missing")
        _require(sha256(receipt_path) == goal.get("precedence_receipt_sha256"), f"{case.get('case_id')} precedence receipt hash mismatch")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        validate_precedence_receipt(receipt, _text(goal.get("c0_goal_receipt_body_sha256"), "C0 body hash"))
        _require(receipt.get("private_truth_body_sha256") == truth_body_sha256, f"{case.get('case_id')} precedence receipt is not bound to private truth body")
    public_ids = [case["case_id"] for case in public.get("cases", [])]
    predicted = {case["case_id"]: case for case in prediction.get("cases", [])}
    truth = {case["case_id"]: case for case in private.get("cases", [])}
    _require(len(public_ids) == len(set(public_ids)), "duplicate public case ids")
    _require(set(public_ids) == set(predicted) == set(truth), "case identity mismatch")
    thresholds = [float(private.get("primary_iou_threshold", 0.30)), *map(float, private.get("diagnostic_iou_thresholds", [0.10, 0.50]))]
    ks = [int(value) for value in private.get("recall_at_k", [1, 3, 5, 10])]
    _require(ks == [1, 3, 5, 10], "PA3 requires Recall@1/3/5/10")
    rows = []
    for case_id in public_ids:
        target = truth[case_id]
        mode = _text(target.get("reference_mode"), f"{case_id} reference_mode")
        _require(mode in REFERENCE_MODES, f"{case_id} reference_mode invalid")
        visibility = _text(target.get("target_visibility", "VISIBLE"), f"{case_id} target_visibility")
        _require(visibility in TARGET_VISIBILITY_STATES, f"{case_id} target_visibility invalid")
        public_mode = next(case["goal_contract"]["reference_mode"] for case in public["cases"] if case["case_id"] == case_id)
        _require(mode == public_mode, f"{case_id} public/private reference mode mismatch")
        target_boxes = [validated_box(box, f"{case_id} target") for box in target.get("legal_target_bboxes_xyxy", [])]
        if visibility == "VISIBLE" and mode == "UNIQUE":
            _require(len(target_boxes) == 1, f"{case_id} UNIQUE requires exactly one legal target")
        elif visibility == "VISIBLE":
            _require(bool(target_boxes), f"{case_id} requires at least one legal target")
        else:
            _require(not target_boxes, f"{case_id} non-visible target must not carry target boxes")
        candidates = predicted[case_id].get("candidates", [])
        _require(isinstance(candidates, list) and len(candidates) <= 10, f"{case_id} candidate cap exceeded")
        _require([candidate.get("rank") for candidate in candidates] == list(range(1, len(candidates) + 1)), f"{case_id} candidate ranks are not contiguous")
        candidate_boxes = [validated_box(candidate.get("bbox_xyxy"), f"{case_id} candidate") for candidate in candidates]
        target_first_ranks = []
        target_best_ious = []
        for target_box in target_boxes:
            overlaps = [iou(candidate_box, target_box) for candidate_box in candidate_boxes]
            target_best_ious.append(max(overlaps, default=0.0))
            target_first_ranks.append({
                str(threshold): next((rank for rank, overlap in enumerate(overlaps, start=1) if overlap >= threshold), None)
                for threshold in thresholds
            })
        any_first_rank = {
            str(threshold): min(
                (ranks[str(threshold)] for ranks in target_first_ranks if ranks[str(threshold)] is not None),
                default=None,
            )
            for threshold in thresholds
        }
        rows.append({
            "case_id": case_id,
            "reference_mode": mode,
            "target_visibility": visibility,
            "primary_evaluable": mode != "AMBIGUOUS" and visibility == "VISIBLE",
            "candidate_count": len(candidates),
            "legal_target_count": len(target_boxes),
            "any_legal_first_rank": any_first_rank,
            "legal_target_recall_at_10": {
                str(threshold): sum(
                    ranks[str(threshold)] is not None and ranks[str(threshold)] <= 10 for ranks in target_first_ranks
                ) / len(target_first_ranks) if target_first_ranks else None
                for threshold in thresholds
            },
            "best_iou_by_legal_target": target_best_ious,
            "latency_ms": float(predicted[case_id]["latency_ms"]),
        })
    evaluable = [row for row in rows if row["primary_evaluable"]]
    primary = str(thresholds[0])
    recall = {
        f"recall_at_{k}": (
            sum(row["any_legal_first_rank"][primary] is not None and row["any_legal_first_rank"][primary] <= k for row in evaluable) / len(evaluable)
            if evaluable else None
        )
        for k in ks
    }
    recall_at_10 = recall["recall_at_10"]
    if recall_at_10 is None:
        terminal = "P1_PA3_NOT_EVALUABLE_NO_UNIQUE_OR_SET_VALUED_CASES"
    elif recall_at_10 == 0.0:
        terminal = "P1_PA3_SEMANTIC_TARGET_AVAILABILITY_NOT_OBSERVED_ON_COHORT"
    elif recall_at_10 == 1.0:
        terminal = "P1_PA3_FULL_BOUNDED_SEMANTIC_TARGET_AVAILABILITY_ON_COHORT"
    else:
        terminal = "P1_PA3_PARTIAL_BOUNDED_SEMANTIC_TARGET_AVAILABILITY_ON_COHORT"
    return {
        "schema_version": "blindassist_p1_pa3_evaluation_v1",
        "protocol_id": PROTOCOL_ID,
        "inputs": {
            "public_input_sha256": public_hash,
            "private_eval_input_sha256": sha256(private_path),
            "prediction_sha256": sha256(prediction_path),
        },
        "case_count": len(rows),
        "primary_evaluable_case_count": len(evaluable),
        "ambiguous_diagnostic_case_count": sum(row["reference_mode"] == "AMBIGUOUS" for row in rows),
        "target_not_visible_case_count": sum(row["target_visibility"] == "NOT_VISIBLE" for row in rows),
        "unadjudicable_case_count": sum(row["target_visibility"] == "UNADJUDICABLE" for row in rows),
        "primary_iou_threshold": thresholds[0],
        "candidate_availability": recall,
        "rows": rows,
        "terminal": terminal,
        "identity_selection": "NOT_EVALUATED",
        "ambiguous_specific_referent_accuracy": "NOT_EVALUABLE_BY_CONTRACT",
        "claim_ceiling": "PROSPECTIVE_GOAL_SEMANTIC_PROPOSAL_AVAILABILITY_ONLY_NO_IDENTITY_GENERALIZATION_PRODUCT_OR_SAFETY_CLAIM",
    }
