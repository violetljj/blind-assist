#!/usr/bin/env python3
"""Create and validate the private-denominator gate for one frozen PA3 execution."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any

from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import (
    PRIVATE_SCHEMA,
    PROTOCOL_ID,
    PUBLIC_SCHEMA,
    REFERENCE_MODES,
    TARGET_VISIBILITY_STATES,
    Pa3ContractError,
    content_sha256,
    private_truth_body,
    sha256,
    validated_box,
)


AUTHORIZATION_SCHEMA = "blindassist_p1_pa3_execution_authorization_v1"
MINIMUM_VISIBLE_EPISODES = 5
MINIMUM_VISIBLE_FRAMES = 8
AUTHORIZED_ARM = "YOLOE_26N_SEG_GOAL_SEMANTIC_TEXT_PROMPT_ONLY"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pa3ContractError(message)


def _text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be non-empty text")
    return value.strip()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def authorize_pa3(
    *,
    public_path: Path,
    private_path: Path,
    prediction_output: Path,
    dispatch_journal: Path,
    authorization_output: Path,
) -> dict[str, Any]:
    public_path = public_path.resolve()
    private_path = private_path.resolve()
    prediction_output = prediction_output.resolve()
    dispatch_journal = dispatch_journal.resolve()
    authorization_output = authorization_output.resolve()
    _require(not authorization_output.exists(), "PA3 authorization receipt already exists")
    _require(not prediction_output.exists(), "PA3 prediction output already exists")
    _require(not dispatch_journal.exists(), "PA3 dispatch journal already exists")
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    _require(public.get("schema_version") == PUBLIC_SCHEMA and public.get("protocol_id") == PROTOCOL_ID, "PA3 public contract mismatch")
    _require(public.get("private_truth_access") is False, "PA3 public input accessed private truth")
    provider_contract = public.get("provider_contract")
    _require(isinstance(provider_contract, Mapping), "PA3 provider contract is missing")
    _require(provider_contract.get("input") == "CURRENT_FRAME_PLUS_PRETRUTH_GOAL_CONTRACT", "PA3 provider input drift")
    _require(provider_contract.get("maximum_candidates") == 10, "PA3 candidate cap drift")
    _require(provider_contract.get("identity_selection") == "FORBIDDEN", "PA3 identity selection is not forbidden")
    _require(private.get("schema_version") == PRIVATE_SCHEMA and private.get("protocol_id") == PROTOCOL_ID, "PA3 private contract mismatch")
    _require(private.get("public_input_sha256") == sha256(public_path), "PA3 private input is not bound to public input")
    _require(float(private.get("primary_iou_threshold", 0.30)) == 0.30, "PA3 primary IoU threshold drift")
    _require(list(private.get("diagnostic_iou_thresholds", [])) == [0.10, 0.50], "PA3 diagnostic IoU thresholds drift")
    _require(list(private.get("recall_at_k", [])) == [1, 3, 5, 10], "PA3 Recall@K contract drift")

    public_cases = public.get("cases")
    private_cases = private.get("cases")
    _require(isinstance(public_cases, list) and bool(public_cases), "PA3 public cases are missing")
    _require(isinstance(private_cases, list) and bool(private_cases), "PA3 private cases are missing")
    public_by_case: dict[str, Mapping[str, Any]] = {}
    for case in public_cases:
        _require(isinstance(case, Mapping), "invalid PA3 public case")
        case_id = _text(case.get("case_id"), "PA3 public case id")
        _require(case_id not in public_by_case, f"duplicate PA3 public case: {case_id}")
        public_by_case[case_id] = case
    private_by_case: dict[str, Mapping[str, Any]] = {}
    for case in private_cases:
        _require(isinstance(case, Mapping), "invalid PA3 private case")
        case_id = _text(case.get("case_id"), "PA3 private case id")
        _require(case_id not in private_by_case, f"duplicate PA3 private case: {case_id}")
        private_by_case[case_id] = case
    _require(set(public_by_case) == set(private_by_case), "PA3 public/private case roster mismatch")

    visible_episode_ids: set[str] = set()
    visible_frame_count = 0
    for case_id, truth in private_by_case.items():
        goal_contract = public_by_case[case_id].get("goal_contract")
        _require(isinstance(goal_contract, Mapping), f"{case_id} public goal contract is missing")
        public_mode = _text(goal_contract.get("reference_mode"), f"{case_id} public reference mode")
        private_mode = _text(truth.get("reference_mode"), f"{case_id} private reference mode")
        _require(public_mode in REFERENCE_MODES and private_mode == public_mode, f"{case_id} public/private reference mode mismatch")
        visibility = _text(truth.get("target_visibility", "VISIBLE"), f"{case_id} target visibility")
        _require(visibility in TARGET_VISIBILITY_STATES, f"{case_id} target visibility invalid")
        legal_boxes = truth.get("legal_target_bboxes_xyxy")
        _require(isinstance(legal_boxes, list), f"{case_id} legal target boxes must be a list")
        for index, box in enumerate(legal_boxes):
            validated_box(box, f"{case_id} legal target {index}")
        if visibility == "VISIBLE":
            _require(bool(legal_boxes), f"{case_id} visible target set is empty")
            if private_mode == "UNIQUE":
                _require(len(legal_boxes) == 1, f"{case_id} UNIQUE target set must contain one box")
        else:
            _require(not legal_boxes, f"{case_id} non-visible target carries boxes")
        if visibility == "VISIBLE":
            visible_frame_count += 1
            visible_episode_ids.add(_text(public_by_case[case_id].get("episode_id"), f"{case_id} episode id"))
    visible_episode_count = len(visible_episode_ids)
    authorized = visible_episode_count >= MINIMUM_VISIBLE_EPISODES and visible_frame_count >= MINIMUM_VISIBLE_FRAMES
    body: dict[str, Any] = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "public_input_sha256": sha256(public_path),
        "private_eval_input_sha256": sha256(private_path),
        "private_truth_body_sha256": content_sha256(private_truth_body(private)),
        "authorization_derived_from_private_denominator": True,
        "private_case_identifiers_exposed_to_provider": False,
        "minimum_visible_episode_count": MINIMUM_VISIBLE_EPISODES,
        "minimum_visible_frame_count": MINIMUM_VISIBLE_FRAMES,
        "visible_episode_count": visible_episode_count,
        "visible_frame_count": visible_frame_count,
        "pa3_inference_authorized": authorized,
        "terminal": "P1_PA3_EXECUTION_AUTHORIZED" if authorized else "NOT_EVALUABLE_INPUT_CONTRACT",
        "authorized_arm": AUTHORIZED_ARM,
        "identity_selection": "FORBIDDEN",
        "functional_region_grounding": "FORBIDDEN",
        "threshold_prompt_model_or_pool_sweep": False,
        "provider_model_calls_before_authorization": 0,
        "prediction_output_path": str(prediction_output),
        "dispatch_journal_path": str(dispatch_journal),
        "retry_or_replay_authorized": False,
    }
    receipt = dict(body)
    receipt["authorization_body_sha256"] = content_sha256(body)
    _atomic_json(authorization_output, receipt)
    return receipt


def _validate_authorized_receipt_content(receipt: Mapping[str, Any]) -> None:
    _require(receipt.get("pa3_inference_authorized") is True, "PA3 inference is not authorized")
    _require(receipt.get("terminal") == "P1_PA3_EXECUTION_AUTHORIZED", "PA3 authorization terminal mismatch")
    _require(receipt.get("authorization_derived_from_private_denominator") is True, "PA3 authorization is not denominator-derived")
    _require(receipt.get("private_case_identifiers_exposed_to_provider") is False, "PA3 authorization exposes private case identity")
    _require(receipt.get("minimum_visible_episode_count") == MINIMUM_VISIBLE_EPISODES, "PA3 visible episode gate drift")
    _require(receipt.get("minimum_visible_frame_count") == MINIMUM_VISIBLE_FRAMES, "PA3 visible frame gate drift")
    _require(receipt.get("visible_episode_count", 0) >= MINIMUM_VISIBLE_EPISODES, "PA3 visible episode denominator is insufficient")
    _require(receipt.get("visible_frame_count", 0) >= MINIMUM_VISIBLE_FRAMES, "PA3 visible frame denominator is insufficient")
    _require(receipt.get("authorized_arm") == AUTHORIZED_ARM, "PA3 authorized arm drift")
    _require(receipt.get("identity_selection") == "FORBIDDEN", "PA3 authorization permits identity selection")
    _require(receipt.get("functional_region_grounding") == "FORBIDDEN", "PA3 authorization permits functional-region grounding")
    _require(receipt.get("threshold_prompt_model_or_pool_sweep") is False, "PA3 authorization permits a sweep")
    _require(receipt.get("provider_model_calls_before_authorization") == 0, "provider ran before PA3 authorization")
    _require(receipt.get("retry_or_replay_authorized") is False, "PA3 authorization unexpectedly permits replay")


def validate_execution_authorization(
    authorization_path: Path,
    public_path: Path,
    prediction_output: Path,
    dispatch_journal: Path,
) -> dict[str, Any]:
    authorization_path = authorization_path.resolve()
    public_path = public_path.resolve()
    prediction_output = prediction_output.resolve()
    dispatch_journal = dispatch_journal.resolve()
    receipt = json.loads(authorization_path.read_text(encoding="utf-8"))
    _require(receipt.get("schema_version") == AUTHORIZATION_SCHEMA and receipt.get("protocol_id") == PROTOCOL_ID, "PA3 authorization contract mismatch")
    declared = _text(receipt.get("authorization_body_sha256"), "PA3 authorization body hash")
    body = dict(receipt)
    body.pop("authorization_body_sha256", None)
    _require(content_sha256(body) == declared, "PA3 authorization body hash mismatch")
    _require(receipt.get("public_input_sha256") == sha256(public_path), "PA3 authorization public binding mismatch")
    _validate_authorized_receipt_content(receipt)
    _require(receipt.get("prediction_output_path") == str(prediction_output), "PA3 prediction output differs from authorization")
    _require(receipt.get("dispatch_journal_path") == str(dispatch_journal), "PA3 dispatch journal differs from authorization")
    _require(not prediction_output.exists(), "PA3 prediction output already exists; refusing replay")
    _require(not dispatch_journal.exists(), "PA3 dispatch journal already exists; refusing replay")
    return receipt


def validate_completed_execution(
    authorization_path: Path,
    public_path: Path,
    private_path: Path,
    prediction_path: Path,
    dispatch_journal: Path,
) -> None:
    authorization_path = authorization_path.resolve()
    public_path = public_path.resolve()
    private_path = private_path.resolve()
    prediction_path = prediction_path.resolve()
    dispatch_journal = dispatch_journal.resolve()
    receipt = json.loads(authorization_path.read_text(encoding="utf-8"))
    declared = _text(receipt.get("authorization_body_sha256"), "PA3 authorization body hash")
    body = dict(receipt)
    body.pop("authorization_body_sha256", None)
    _require(receipt.get("schema_version") == AUTHORIZATION_SCHEMA and receipt.get("protocol_id") == PROTOCOL_ID, "PA3 authorization contract mismatch")
    _require(content_sha256(body) == declared, "PA3 authorization body hash mismatch")
    _require(receipt.get("public_input_sha256") == sha256(public_path), "PA3 authorization public binding mismatch")
    _validate_authorized_receipt_content(receipt)
    _require(receipt.get("private_eval_input_sha256") == sha256(private_path), "PA3 authorization private binding mismatch")
    private = json.loads(private_path.read_text(encoding="utf-8"))
    _require(receipt.get("private_truth_body_sha256") == content_sha256(private_truth_body(private)), "PA3 authorization private truth binding mismatch")
    _require(receipt.get("prediction_output_path") == str(prediction_path), "PA3 prediction differs from authorized output")
    _require(receipt.get("dispatch_journal_path") == str(dispatch_journal), "PA3 journal differs from authorized journal")
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    _require(prediction.get("execution_authorization_sha256") == sha256(authorization_path), "PA3 prediction authorization binding mismatch")
    _require(prediction.get("dispatch_journal_path") == str(dispatch_journal), "PA3 prediction journal binding mismatch")
    journal = json.loads(dispatch_journal.read_text(encoding="utf-8"))
    _require(journal.get("schema_version") == "blindassist_p1_pa3_dispatch_journal_v1" and journal.get("protocol_id") == PROTOCOL_ID, "PA3 dispatch journal contract mismatch")
    _require(journal.get("status") == "COMPLETED", "PA3 dispatch did not complete")
    _require(journal.get("public_input_sha256") == sha256(public_path), "PA3 journal public binding mismatch")
    _require(journal.get("authorization_receipt_sha256") == sha256(authorization_path), "PA3 journal authorization binding mismatch")
    _require(journal.get("prediction_output_path") == str(prediction_path), "PA3 journal prediction path mismatch")
    _require(journal.get("prediction_sha256") == sha256(prediction_path), "PA3 journal prediction hash mismatch")
    dispatched = journal.get("provider_model_calls_dispatched")
    completed = journal.get("provider_model_calls_completed")
    _require(isinstance(dispatched, int) and dispatched == completed == len(prediction.get("cases", [])), "PA3 dispatch call accounting mismatch")
    _require(journal.get("retry_or_replay_authorized") is False, "PA3 dispatch journal permits replay")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--private", required=True, type=Path)
    parser.add_argument("--prediction-output", required=True, type=Path)
    parser.add_argument("--dispatch-journal", required=True, type=Path)
    parser.add_argument("--authorization-output", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = authorize_pa3(
        public_path=args.public,
        private_path=args.private,
        prediction_output=args.prediction_output,
        dispatch_journal=args.dispatch_journal,
        authorization_output=args.authorization_output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
