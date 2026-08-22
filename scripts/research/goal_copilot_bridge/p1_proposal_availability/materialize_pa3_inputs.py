#!/usr/bin/env python3
"""Materialize hash-bound PA3 public/private inputs from prospective receipts."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import (
    PRIVATE_SCHEMA,
    PRECEDENCE_SCHEMA,
    PROTOCOL_ID,
    PUBLIC_SCHEMA,
    Pa3ContractError,
    TARGET_VISIBILITY_STATES,
    PRECEDENCE_MODES,
    content_sha256,
    private_truth_body,
    prompt_lookup,
    sha256,
    validated_box,
)


C0_SCHEMA = "blindassist_p1_pa3_c0_public_goal_cohort_v1"
CAPTURE_SCHEMA = "blindassist_p1_pa3_capture_manifest_v1"
TRUTH_SCHEMA = "blindassist_p1_pa3_truth_body_v1"
PHYSICAL_SOURCE_ROLE = "PROSPECTIVE_FIRST_PERSON_PHYSICAL_CAPTURE_AFTER_GOAL"
PHYSICAL_CAPTURE_INSTRUCTION = "APPROACH_NAMED_BUILDING_AND_STOP_AFTER_ENTRANCE_IS_IN_VIEW_V1"
PHYSICAL_FRAME_SELECTION_RULE = "FIXED_OFFSETS_FROM_VIDEO_END_NO_PIXEL_OR_TRUTH_SELECTION"
PHYSICAL_FRAME_OFFSETS = [-2.5, -1.5, -0.5]
PHYSICAL_CAPTURE_TIME_SEMANTICS = "PHYSICAL_DEVICE_CAPTURE_TIMELINE_DERIVED_FROM_VIDEO_PTS"
PUBLIC_SPATIAL_SCHEMAS = {
    "blindassist_p1_pa3_public_spatial_goal_contract_v2",
    "blindassist_p1_pa3_public_spatial_goal_contract_v3",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pa3ContractError(message)


def _text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be non-empty text")
    return value.strip()


def _lower_sha256(value: Any, label: str) -> str:
    text = _text(value, label)
    _require(len(text) == 64 and all(character in "0123456789abcdef" for character in text), f"{label} must be lowercase SHA-256")
    return text


def _verify_capture_manifest(capture: Mapping[str, Any], c0_body_sha: str) -> tuple[str, str]:
    _require(capture.get("schema_version") == CAPTURE_SCHEMA, "capture manifest schema mismatch")
    declared = _lower_sha256(capture.get("capture_manifest_body_sha256"), "capture manifest body hash")
    body = dict(capture)
    body.pop("capture_manifest_body_sha256", None)
    _require(content_sha256(body) == declared, "capture manifest body hash mismatch")
    _require(capture.get("goal_receipt_body_sha256") == c0_body_sha, "capture manifest C0 binding mismatch")
    _require(capture.get("private_truth_access") is False, "capture manifest accessed private truth")
    _require(capture.get("provider_model_calls") == 0, "provider ran before PA3 authorization")
    mode = _text(capture.get("precedence_mode", "PHYSICAL_CAPTURE_AFTER_GOAL"), "capture precedence mode")
    _require(mode in PRECEDENCE_MODES, "capture precedence mode invalid")
    if mode == "PHYSICAL_CAPTURE_AFTER_GOAL":
        _require(capture.get("physical_capture_after_goal_claimed") is True, "physical capture claim is missing")
        _require(capture.get("source_role") == PHYSICAL_SOURCE_ROLE, "physical capture source role mismatch")
        _require(capture.get("capture_instruction_id") == PHYSICAL_CAPTURE_INSTRUCTION, "physical capture instruction drift")
        _require(capture.get("frame_selection_rule") == PHYSICAL_FRAME_SELECTION_RULE, "physical frame-selection rule drift")
        _require(list(capture.get("frame_offsets_from_end_seconds", [])) == PHYSICAL_FRAME_OFFSETS, "physical frame offsets drift")
        _lower_sha256(capture.get("prospective_capture_receipt_body_sha256"), "prospective capture receipt hash")
        _lower_sha256(capture.get("prospective_capture_plan_body_sha256"), "prospective capture plan hash")
    else:
        _require(capture.get("physical_capture_after_goal_claimed") is False, "public-source capture must not claim post-goal physical capture")
    return declared, mode


def _timestamp(value: Any, label: str) -> datetime:
    text = _text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise Pa3ContractError(f"{label} must be ISO-8601") from error
    _require(parsed.tzinfo is not None and parsed.utcoffset() is not None, f"{label} must include timezone")
    return parsed


def _verify_c0_receipt(c0: Mapping[str, Any]) -> str:
    _require(c0.get("schema_version") == C0_SCHEMA, "C0 receipt schema mismatch")
    body_hash = _text(c0.get("receipt_body_sha256"), "C0 receipt_body_sha256")
    body = dict(c0)
    body.pop("receipt_body_sha256", None)
    _require(content_sha256(body) == body_hash, "C0 receipt body hash mismatch")
    _require(c0.get("private_truth_access") is False, "C0 receipt must be provider-public")
    _require(c0.get("pa3_inference_authorized") is False, "C0 receipt authority drift")
    return body_hash


def _verify_public_spatial_contract(spatial: Mapping[str, Any], c0_body_sha: str) -> dict[str, Mapping[str, Any]]:
    _require(spatial.get("schema_version") in PUBLIC_SPATIAL_SCHEMAS, "public spatial contract schema mismatch")
    body_hash = _text(spatial.get("roster_body_sha256"), "public spatial contract body hash")
    body = dict(spatial)
    body.pop("roster_body_sha256", None)
    _require(content_sha256(body) == body_hash, "public spatial contract body hash mismatch")
    _require(spatial.get("goal_receipt_body_sha256") == c0_body_sha, "public spatial contract C0 binding mismatch")
    _require(spatial.get("provider_public") is True, "public spatial contract must be provider-public")
    _require(spatial.get("private_truth_access") is False, "public spatial contract cannot access private truth")
    _require(spatial.get("source_authority") == "OPENSTREETMAP_PRETRUTH", "public spatial source authority drift")
    _require(spatial.get("spatial_goal_role") == "PRODUCT_NAVIGATION_ROUTE_ENDPOINT_CANDIDATE", "public spatial goal role drift")
    _require(spatial.get("created_before_mapillary_metadata_pixels_and_truth") is True, "public spatial precedence is not confirmed")
    episodes = spatial.get("episodes")
    _require(isinstance(episodes, list) and bool(episodes), "public spatial contract episodes are required")
    by_episode = {str(row.get("episode_id")): row for row in episodes if isinstance(row, Mapping)}
    _require(len(by_episode) == len(episodes), "duplicate or invalid public spatial episode")
    return by_episode


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def materialize_inputs(
    *,
    c0: Mapping[str, Any],
    prompt_map: Mapping[str, Any],
    capture: Mapping[str, Any],
    truth: Mapping[str, Any],
    output_dir: Path,
    source_base_dir: Path,
    public_spatial_contract: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    output_dir = output_dir.resolve()
    source_base_dir = source_base_dir.resolve()
    c0_body_sha = _verify_c0_receipt(c0)
    spatial_by_episode = _verify_public_spatial_contract(public_spatial_contract, c0_body_sha) if public_spatial_contract is not None else None
    _require(c0.get("prompt_map_sha256") == content_sha256(prompt_map), "C0 prompt-map hash mismatch")
    prompts = prompt_lookup(prompt_map)
    _require(truth.get("schema_version") == TRUTH_SCHEMA, "truth body schema mismatch")
    _require(not output_dir.exists(), "PA3 materialization output already exists")
    capture_body_sha, precedence_mode = _verify_capture_manifest(capture, c0_body_sha)

    c0_episodes = c0.get("episodes", [])
    goals = {episode["episode_id"]: episode for episode in c0_episodes}
    _require(len(goals) == len(c0_episodes), "duplicate C0 episode_id")
    captures = capture.get("cases")
    truths = truth.get("cases")
    _require(isinstance(captures, list) and bool(captures), "capture manifest must be non-empty")
    _require(isinstance(truths, list) and bool(truths), "truth body must be non-empty")
    truth_by_case = {case["case_id"]: case for case in truths}
    _require(len(truth_by_case) == len(truths), "duplicate truth case_id")
    case_ids = [case["case_id"] for case in captures]
    _require(len(case_ids) == len(set(case_ids)), "duplicate capture case_id")
    _require(set(case_ids) == set(truth_by_case), "capture/truth case identity mismatch")
    truth_created_text = _text(truth.get("truth_created_at_utc"), "truth_created_at_utc")
    truth_created_at = _timestamp(truth_created_text, "truth_created_at_utc")
    _require(float(truth.get("primary_iou_threshold", 0.30)) == 0.30, "PA3 primary IoU threshold must remain 0.30")
    _require(list(truth.get("diagnostic_iou_thresholds", [0.10, 0.50])) == [0.10, 0.50], "PA3 diagnostic IoU thresholds drift")
    _require(list(truth.get("recall_at_k", [1, 3, 5, 10])) == [1, 3, 5, 10], "PA3 Recall@K contract drift")

    private = {
        "schema_version": PRIVATE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "public_input_sha256": "PENDING_PUBLIC_MATERIALIZATION",
        "primary_iou_threshold": float(truth.get("primary_iou_threshold", 0.30)),
        "diagnostic_iou_thresholds": list(truth.get("diagnostic_iou_thresholds", [0.10, 0.50])),
        "recall_at_k": list(truth.get("recall_at_k", [1, 3, 5, 10])),
        "cases": truths,
    }
    truth_body_sha = content_sha256(private_truth_body(private))
    public_cases = []
    receipts: list[tuple[Path, dict[str, Any]]] = []
    for captured in captures:
        case_id = _text(captured.get("case_id"), "case_id")
        episode_id = _text(captured.get("episode_id"), f"{case_id} episode_id")
        _require(episode_id in goals, f"{case_id} has no C0 goal episode")
        goal = goals[episode_id]
        goal_provenance = goal.get("goal_provenance")
        contract = goal.get("goal_contract")
        _require(isinstance(goal_provenance, Mapping) and isinstance(contract, Mapping), f"{case_id} C0 goal is incomplete")
        goal_recorded_text = _text(goal_provenance.get("goal_recorded_at_utc"), f"{case_id} goal_recorded_at_utc")
        capture_created_text = _text(captured.get("capture_created_at_utc"), f"{case_id} capture_created_at_utc")
        if precedence_mode == "GOAL_BEFORE_FIRST_PROJECT_PIXEL_ACCESS_AND_TRUTH":
            _require(captured.get("capture_time_semantics") == "FIRST_PROJECT_PIXEL_ACCESS_NOT_PHYSICAL_CAMERA_CAPTURE", f"{case_id} pixel-access time semantics mismatch")
            source_capture_time_semantics = captured.get("source_capture_time_semantics")
            if source_capture_time_semantics == "UNKNOWN_PREEXISTING_PUBLIC_DATASET_CAPTURE":
                _require("source_captured_at_utc" not in captured, f"{case_id} unknown source capture time must not be fabricated")
            elif source_capture_time_semantics == "DATASET_COLLECTION_DATE_NO_PER_IMAGE_TIMESTAMP":
                _require("source_captured_at_utc" not in captured, f"{case_id} collection date must not be represented as per-image capture time")
                _timestamp(captured.get("source_collection_date_utc"), f"{case_id} source_collection_date_utc")
            else:
                _timestamp(captured.get("source_captured_at_utc"), f"{case_id} source_captured_at_utc")
        else:
            _require(captured.get("capture_time_semantics") == PHYSICAL_CAPTURE_TIME_SEMANTICS, f"{case_id} physical capture time semantics mismatch")
            _require(captured.get("frame_selection_rule") == PHYSICAL_FRAME_SELECTION_RULE, f"{case_id} physical frame-selection rule drift")
            _lower_sha256(captured.get("source_media_sha256"), f"{case_id} source media hash")
            _require(isinstance(captured.get("source_video_timestamp_seconds"), (int, float)), f"{case_id} source video timestamp is missing")
        goal_recorded_at = _timestamp(goal_recorded_text, f"{case_id} goal_recorded_at_utc")
        capture_created_at = _timestamp(capture_created_text, f"{case_id} capture_created_at_utc")
        _require(goal_recorded_at < capture_created_at, f"{case_id} goal does not precede capture")
        _require(goal_recorded_at < truth_created_at, f"{case_id} goal does not precede truth")
        _require(capture_created_at < truth_created_at, f"{case_id} truth must be created after capture")
        image_path = Path(_text(captured.get("image_path"), f"{case_id} image_path"))
        if not image_path.is_absolute():
            image_path = (source_base_dir / image_path).resolve()
        _require(image_path.is_file(), f"{case_id} image is missing")
        image_sha = sha256(image_path)
        _require(image_sha == captured.get("image_sha256"), f"{case_id} image hash mismatch")
        goal_type = _text(contract.get("goal_type"), f"{case_id} goal_type")
        _require(goal_type in prompts, f"{case_id} goal type lacks frozen prompt")
        _require(goal.get("canonical_prompt") == prompts[goal_type], f"{case_id} C0 canonical prompt drift")
        reference_mode = _text(contract.get("reference_mode"), f"{case_id} reference_mode")
        _require(truth_by_case[case_id].get("reference_mode") == reference_mode, f"{case_id} public/private reference mode mismatch")
        visibility = _text(truth_by_case[case_id].get("target_visibility", "VISIBLE"), f"{case_id} target_visibility")
        _require(visibility in TARGET_VISIBILITY_STATES, f"{case_id} target_visibility invalid")
        legal_boxes = truth_by_case[case_id].get("legal_target_bboxes_xyxy")
        _require(isinstance(legal_boxes, list), f"{case_id} legal target boxes must be a list")
        for index, box in enumerate(legal_boxes):
            validated_box(box, f"{case_id} legal target {index}")
        if visibility == "VISIBLE" and reference_mode == "UNIQUE":
            _require(len(legal_boxes) == 1, f"{case_id} UNIQUE requires exactly one legal target")
        elif visibility == "VISIBLE":
            _require(bool(legal_boxes), f"{case_id} visible legal target set must be non-empty")
        else:
            _require(not legal_boxes, f"{case_id} non-visible target must not carry target boxes")
        receipt_path = output_dir / "precedence" / f"{case_id}.json"
        receipt = {
            "schema_version": PRECEDENCE_SCHEMA,
            "case_id": case_id,
            "episode_id": episode_id,
            "goal_receipt_body_sha256": c0_body_sha,
            "goal_recorded_at_utc": goal_recorded_text,
            "capture_created_at_utc": capture_created_text,
            "truth_created_at_utc": truth_created_text,
            "precedence_mode": precedence_mode,
            "created_before_capture": precedence_mode == "PHYSICAL_CAPTURE_AFTER_GOAL",
            "created_before_project_pixel_access": precedence_mode == "GOAL_BEFORE_FIRST_PROJECT_PIXEL_ACCESS_AND_TRUTH",
            "physical_capture_after_goal_claimed": capture.get("physical_capture_after_goal_claimed", precedence_mode == "PHYSICAL_CAPTURE_AFTER_GOAL"),
            "created_before_truth": True,
            "capture_manifest_body_sha256": capture_body_sha,
            "private_truth_body_sha256": truth_body_sha,
        }
        receipts.append((receipt_path, receipt))
        public_goal_contract = {
            "goal_text_original": goal["goal_text_original"],
            "goal_type": goal_type,
            "reference_mode": reference_mode,
            "task_semantics": contract["task_semantics"],
            "canonical_prompt": prompts[goal_type],
            "c0_goal_receipt_body_sha256": c0_body_sha,
            "precedence_receipt_path": str(receipt_path),
            "precedence_receipt_sha256": "PENDING_RECEIPT_WRITE",
        }
        if spatial_by_episode is not None:
            _require(episode_id in spatial_by_episode, f"{case_id} has no public spatial episode")
            spatial = spatial_by_episode[episode_id]
            _require(captured.get("public_spatial_contract_body_sha256") == public_spatial_contract.get("roster_body_sha256"), f"{case_id} public spatial body binding mismatch")
            if public_spatial_contract.get("schema_version") == "blindassist_p1_pa3_public_spatial_goal_contract_v2":
                endpoint = spatial.get("selected_entrance")
                _require(isinstance(endpoint, Mapping), f"{case_id} has no public route endpoint candidate")
                _require(captured.get("osm_entrance_node_id") == endpoint.get("osm_node_id"), f"{case_id} route endpoint candidate drift")
                endpoint_candidates = [{"candidate_id": f"osm-entrance-{endpoint['osm_node_id']}", **dict(endpoint)}]
                selected_candidate_id = endpoint_candidates[0]["candidate_id"]
            else:
                endpoint_candidates = spatial.get("route_endpoint_candidates")
                _require(isinstance(endpoint_candidates, list) and bool(endpoint_candidates), f"{case_id} has no public route endpoint candidates")
                selected_candidate_id = _text(captured.get("public_spatial_candidate_id"), f"{case_id} public spatial candidate id")
                _require(selected_candidate_id in {candidate.get("candidate_id") for candidate in endpoint_candidates if isinstance(candidate, Mapping)}, f"{case_id} route endpoint candidate drift")
            public_goal_contract["public_spatial_context"] = {
                "schema_version": public_spatial_contract["schema_version"],
                "spatial_goal_role": public_spatial_contract["spatial_goal_role"],
                "source_authority": public_spatial_contract["source_authority"],
                "spatial_contract_body_sha256": public_spatial_contract["roster_body_sha256"],
                "place_anchor": {
                    "osm_type": spatial.get("osm_type"),
                    "osm_id": spatial.get("osm_id"),
                    "lat": spatial.get("lat"),
                    "lon": spatial.get("lon"),
                },
                "selected_parent": spatial.get("selected_parent"),
                "route_endpoint_candidates": [dict(candidate) for candidate in endpoint_candidates],
                "observation_selected_candidate_id": selected_candidate_id,
            }
        public_cases.append({
            "case_id": case_id,
            "episode_id": episode_id,
            "query": {"image_path": str(image_path), "image_sha256": image_sha},
            "goal_contract": public_goal_contract,
        })

    output_dir.mkdir(parents=True, exist_ok=False)
    for receipt_path, receipt in receipts:
        _atomic_json(receipt_path, receipt)
        case = next(case for case in public_cases if case["case_id"] == receipt["case_id"])
        case["goal_contract"]["precedence_receipt_sha256"] = sha256(receipt_path)
    public = {
        "schema_version": PUBLIC_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "private_truth_access": False,
        "prompt_map_sha256": content_sha256(prompt_map),
        "provider_contract": {
            "input": "CURRENT_FRAME_PLUS_PRETRUTH_GOAL_CONTRACT",
            "maximum_candidates": 10,
            "identity_selection": "FORBIDDEN",
            "public_spatial_context": spatial_by_episode is not None,
        },
        "cases": public_cases,
        "claim_role": "PROSPECTIVE_GOAL_SEMANTIC_PROPOSAL_AVAILABILITY",
    }
    public_path = output_dir / "public_input.json"
    private_path = output_dir / "private_eval_input.json"
    _atomic_json(public_path, public)
    private["public_input_sha256"] = sha256(public_path)
    _atomic_json(private_path, private)
    return public_path, private_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c0-receipt", required=True, type=Path)
    parser.add_argument("--prompt-map", required=True, type=Path)
    parser.add_argument("--capture-manifest", required=True, type=Path)
    parser.add_argument("--private-truth", required=True, type=Path)
    parser.add_argument("--public-spatial-contract", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    values = [
        json.loads(args.c0_receipt.read_text(encoding="utf-8")),
        json.loads(args.prompt_map.read_text(encoding="utf-8")),
        json.loads(args.capture_manifest.read_text(encoding="utf-8")),
        json.loads(args.private_truth.read_text(encoding="utf-8")),
    ]
    spatial = json.loads(args.public_spatial_contract.read_text(encoding="utf-8")) if args.public_spatial_contract else None
    public_path, private_path = materialize_inputs(
        c0=values[0],
        prompt_map=values[1],
        capture=values[2],
        truth=values[3],
        output_dir=args.output_dir,
        source_base_dir=args.capture_manifest.resolve().parent,
        public_spatial_contract=spatial,
    )
    print(json.dumps({
        "public_input": str(public_path),
        "public_input_sha256": sha256(public_path),
        "private_eval_input": str(private_path),
        "private_eval_input_sha256": sha256(private_path),
        "pa3_inference_authorized": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
