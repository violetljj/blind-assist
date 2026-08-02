"""Freeze a small output-blind burned judge pilot and render RGB-only packets.

This command stops before any primitive review is created.  It selects a
deterministic slice of already frozen screening windows by their opaque
screening order, verifies RGB payloads against the frozen native-input plan,
and writes three isolated packets: two causal prefix-view packets and one
retrospective packet.  Visibility is explicitly current-frame-only in every
view; temporal primitives have a separate, declared evidence window.
No source mask, model output, YOLO box, oracle trace, category or action label
is copied into a reviewer packet.

The command accepts an explicitly named in-progress materialization staging
root for the burned calibration asset only.  The plan-bound MD5 checks still
run for every copied RGB frame; formal evaluation requires the completed native
manifest and the separate data-admission gate.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import shutil
import uuid
from pathlib import Path
from typing import Any

from .common import PROTOCOL_ID, read_json, sha256_file, sha256_json
from .freeze_screening_cohort import SCHEMA as SCREENING_COHORT_SCHEMA
from .materialize_screening_inputs import MATERIALIZED_SCHEMA, PLAN_SCHEMA


FREEZE_SCHEMA = "blindassist.eval_validity_r0.judge_pilot_freeze.v3"
PACKET_SCHEMA = "blindassist.eval_validity_r0.judge_primitive_packet.v3"
PRIVATE_MAP_SCHEMA = "blindassist.eval_validity_r0.judge_primitive_private_map.v1"
REVIEW_MAP_SCHEMA = "blindassist.eval_validity_r0.judge_review_map.v1"
REVIEW_SCHEMA = "blindassist.eval_validity_r0.judge_review.v5"
PRIMITIVE_POLICY_VERSION = "primitive_observability_v3"
VISIBILITY_POLICY_VERSION = "visibility_observability_v2"
VISIBILITY_EVIDENCE_WINDOW = "CURRENT_RGB_FRAME_ONLY"
GEOMETRIC_EVIDENCE_WINDOW = "CURRENT_RGB_FRAME_ONLY"
CAUSAL_TEMPORAL_EVIDENCE_WINDOW = "CURRENT_PLUS_PAST_PREFIX"
RETROSPECTIVE_TEMPORAL_EVIDENCE_WINDOW = "FULL_EVENT_RGB"
ROUTE_ANCHOR_DEFINITION = "The route is the pedestrian-support surface currently carrying the camera and its visible forward continuation."
ROUTE_ANCHOR_RULE = "Use the surface directly under the camera and connected forward; do not infer the route from object identity, a reminder policy or a parallel/background walkway."
ROUTE_ANCHOR_INCLUDED_SURFACES = ["SIDEWALK", "PATH", "TRAIL", "OTHER_CURRENT_WALKABLE_SUPPORT_SURFACE"]
ROUTE_ANCHOR_EXCLUDED_SURFACES = ["VEHICLE_LANE_NOT_CURRENTLY_OCCUPIED_BY_CAMERA", "PARALLEL_OR_BACKGROUND_WALKWAY_NOT_CONNECTED_TO_CURRENT_SURFACE", "OBJECT_SIDE_DETOUR_NOT_VISIBLE_AS_A_CONTINUATION"]
ROUTE_ANCHOR_BRANCH_RULE = "Count a branch only when two or more materially distinct walkable continuations of this current surface are visibly connected."
FIELD_EVIDENCE_WINDOWS = {
    "visibility": VISIBILITY_EVIDENCE_WINDOW,
    "path_relation": GEOMETRIC_EVIDENCE_WINDOW,
    "route_certainty": GEOMETRIC_EVIDENCE_WINDOW,
    "evidence_quality": GEOMETRIC_EVIDENCE_WINDOW,
    "motion_relation": CAUSAL_TEMPORAL_EVIDENCE_WINDOW,
    "phase": CAUSAL_TEMPORAL_EVIDENCE_WINDOW,
    "retrospective_motion_relation": RETROSPECTIVE_TEMPORAL_EVIDENCE_WINDOW,
    "retrospective_phase": RETROSPECTIVE_TEMPORAL_EVIDENCE_WINDOW,
}
PRIMITIVE_FIELDS = {
    "visibility": ["EVALUABLE", "NOT_EVALUABLE"],
    "path_relation": ["BLOCKING_PATH", "NON_BLOCKING_PATH", "AMBIGUOUS"],
    "motion_relation": ["APPROACHING", "LATERAL_PASS", "RECEDING", "STATIC_OR_UNCLEAR"],
    "phase": ["BEFORE_INTRUSION", "CURRENT_INTRUSION", "PASSED_CLEAR", "UNKNOWN"],
    "route_certainty": ["SINGLE_PLAUSIBLE_ROUTE", "MULTIPLE_PLAUSIBLE_ROUTES", "UNKNOWN"],
    "evidence_quality": ["CLEAR", "BLUR", "OCCLUSION", "CAMERA_ROTATION", "INSUFFICIENT"],
}


class PilotPreparationError(ValueError):
    """Raised when a burned pilot input is not safely materializable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotPreparationError(message)


def _md5_base64(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - matches the native GCS md5Hash receipt.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _validate_cohort(cohort: dict[str, Any]) -> list[dict[str, Any]]:
    _require(cohort.get("schema_version") == SCREENING_COHORT_SCHEMA, "screening cohort schema mismatch")
    _require(cohort.get("protocol_id") == PROTOCOL_ID, "screening cohort protocol mismatch")
    _require(cohort.get("status") == "OUTPUT_BLIND_SCREENING_COHORT_CONTINUOUS_WINDOWS_FROZEN", "screening cohort continuous windows are not frozen")
    _require(cohort.get("candidate_outputs_opened") is False and cohort.get("final_event_facts_frozen") is False, "screening cohort output state is not blind")
    items = cohort.get("items")
    _require(isinstance(items, list) and len(items) == 48, "screening cohort must contain 48 frozen windows")
    event_ids = [item.get("screening_event_id") for item in items if isinstance(item, dict)]
    sessions = [item.get("source_session_id") for item in items if isinstance(item, dict)]
    _require(len(event_ids) == 48 and len(set(event_ids)) == 48, "screening event identities are not unique")
    _require(len(sessions) == 48 and len(set(sessions)) == 48, "screening source sessions are not independent")
    return sorted(items, key=lambda item: item["screening_event_id"])


def _validate_plan(plan: dict[str, Any], cohort: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _require(plan.get("schema_version") == PLAN_SCHEMA, "native input plan schema mismatch")
    _require(plan.get("protocol_id") == PROTOCOL_ID, "native input plan protocol mismatch")
    _require(plan.get("screening_cohort_sha256") == sha256_json(cohort), "native input plan is not bound to screening cohort")
    items = plan.get("items")
    _require(isinstance(items, list) and len(items) == 48, "native input plan must contain 48 windows")
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        event_id = item.get("screening_event_id") if isinstance(item, dict) else None
        _require(isinstance(event_id, str) and event_id not in result, "native input plan event identity is invalid")
        result[event_id] = item
    return result


def _source_state(source_root: Path, plan: dict[str, Any], cohort: dict[str, Any], allow_staging: bool) -> dict[str, Any]:
    manifest_path = source_root / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        _require(manifest.get("schema_version") == MATERIALIZED_SCHEMA, "native materialized manifest schema mismatch")
        _require(manifest.get("protocol_id") == PROTOCOL_ID, "native materialized manifest protocol mismatch")
        _require(manifest.get("screening_cohort_sha256") == sha256_json(cohort), "native materialized manifest cohort binding mismatch")
        _require(manifest.get("native_asset_plan_sha256") == sha256_json(plan), "native materialized manifest plan binding mismatch")
        _require(manifest.get("candidate_outputs_opened") is False, "native materialized manifest is output-contaminated")
        return {
            "state": "COMPLETED_NATIVE_MANIFEST",
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_present": True,
        }
    _require(allow_staging, "native materialized manifest is missing; pass the explicit burned-only staging flag to use a plan-bound staging root")
    return {
        "state": "PLAN_BOUND_STAGING_ROOT_BURNED_ONLY",
        "manifest_sha256": None,
        "manifest_present": False,
    }


def _verify_rgb_window(source_root: Path, event_id: str, plan_item: dict[str, Any]) -> list[dict[str, Any]]:
    frames = plan_item.get("frames")
    _require(isinstance(frames, list) and frames, f"{event_id}: native plan frames missing")
    verified: list[dict[str, Any]] = []
    for ordinal, frame in enumerate(frames):
        _require(isinstance(frame, dict) and frame.get("ordinal") == ordinal, f"{event_id}: native frame ordinals are not contiguous")
        path = source_root / "events" / event_id / "rgb" / f"{ordinal:03d}.png"
        _require(path.is_file() and path.stat().st_size > 8, f"{event_id}: RGB payload is missing at ordinal {ordinal}")
        _require(path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"{event_id}: RGB payload is not a PNG at ordinal {ordinal}")
        expected_md5 = frame.get("rgb", {}).get("md5_base64")
        _require(isinstance(expected_md5, str) and expected_md5, f"{event_id}: native RGB MD5 receipt is missing")
        _require(_md5_base64(path) == expected_md5, f"{event_id}: RGB MD5 mismatch at ordinal {ordinal}")
        verified.append({
            "ordinal": ordinal,
            "source_frame_index": frame.get("source_frame_index"),
            "rgb_sha256": sha256_file(path),
            "source_path": path,
        })
    return verified


def _opaque_id() -> str:
    return f"opaque-{secrets.token_urlsafe(18)}"


def _packet_item(
    *, role: str, view: str, opaque_id: str, source_frames: list[dict[str, Any]], staged_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    asset_paths: list[str] = []
    for frame in source_frames:
        source = frame["source_path"]
        target = staged_root / "assets" / opaque_id / f"{frame['ordinal']:03d}.png"
        _link_or_copy(source, target)
        asset_paths.append(target.relative_to(staged_root).as_posix())
    temporal_frames = (
        [asset_paths[: ordinal + 1] for ordinal in range(len(asset_paths))]
        if view == "CAUSAL"
        else [asset_paths for _ in asset_paths]
    )
    frame_cards = [
        {
            "frame_ordinal": ordinal,
            "current_rgb_frame": asset_paths[ordinal],
            "visibility_rgb_frames": [asset_paths[ordinal]],
            "path_relation_rgb_frames": [asset_paths[ordinal]],
            "route_certainty_rgb_frames": [asset_paths[ordinal]],
            "evidence_quality_rgb_frames": [asset_paths[ordinal]],
            "temporal_rgb_frames": temporal_frames[ordinal],
        }
        for ordinal in range(len(asset_paths))
    ]
    item = {
        "review_item_id": opaque_id,
        "frame_cards": frame_cards,
        "response_fields": PRIMITIVE_FIELDS,
    }
    private = {
        "pilot_event_id": None,
        "source_frame_indices": [frame["source_frame_index"] for frame in source_frames],
        "frame_count": len(source_frames),
        "reviewer_role": role,
    }
    return item, private


def _render_packet(
    *, role: str, view: str, event_rows: list[dict[str, Any]], source_root: Path, staged_root: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    private: dict[str, dict[str, Any]] = {}
    for row in event_rows:
        opaque = _opaque_id()
        item, private_row = _packet_item(role=role, view=view, opaque_id=opaque, source_frames=row["verified_frames"], staged_root=staged_root)
        item_private = {
            **private_row,
            "pilot_event_id": row["pilot_event_id"],
            "source_screening_event_id": row["source_screening_event_id"],
            "source_session_id": row["source_session_id"],
        }
        items.append(item)
        private[opaque] = item_private
    packet = {
        "schema_version": PACKET_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "reviewer_role": role,
        "view": view,
        "future_frame_access": view == "RETROSPECTIVE",
        "status": "PRIMITIVE_REVIEW_PENDING",
        "disclosures": {
            "model_output_visible": False,
            "oracle_output_visible": False,
            "yolo_box_visible": False,
            "source_mask_visible": False,
            "source_session_or_event_identity_visible": False,
            "discovery_arm_visible": False,
            "reviewed_event_phase_visible_to_pair_builder": False,
            "direct_action_labels_requested": False,
            "future_frames_visible_before_current_frame": view == "RETROSPECTIVE",
            "visibility_current_frame_only": True,
            "visibility_evidence_window": VISIBILITY_EVIDENCE_WINDOW,
            "geometric_fields_current_frame_only": True,
            "geometric_evidence_window": GEOMETRIC_EVIDENCE_WINDOW,
            "temporal_fields_evidence_window": CAUSAL_TEMPORAL_EVIDENCE_WINDOW if view == "CAUSAL" else RETROSPECTIVE_TEMPORAL_EVIDENCE_WINDOW,
        },
        "review_instructions": {
            "submit_only_primitive_observations": True,
            "do_not_submit_actionability": True,
            "frame_index_for_submission": "frame_ordinal",
            "primitive_policy_version": PRIMITIVE_POLICY_VERSION,
            "unknown_rule": "Use AMBIGUOUS for path_relation when the current route/region relation cannot be localized; use UNKNOWN for route_certainty or phase when their defined evidence is unavailable; use INSUFFICIENT only for evidence_quality when no usable route anchor remains.",
            "route_anchor_definition": ROUTE_ANCHOR_DEFINITION,
            "route_anchor_rule": ROUTE_ANCHOR_RULE,
            "route_anchor_included_surfaces": ROUTE_ANCHOR_INCLUDED_SURFACES,
            "route_anchor_excluded_surfaces": ROUTE_ANCHOR_EXCLUDED_SURFACES,
            "route_anchor_branch_rule": ROUTE_ANCHOR_BRANCH_RULE,
            "visibility_policy_version": VISIBILITY_POLICY_VERSION,
            "visibility_definition": "EVALUABLE means the forward route scene is localized in the current RGB frame; object identity and obstacle presence are not required.",
            "visibility_not_evaluable_only_if": [
                "NO_LOCALIZED_ROUTE_OR_SCENE_REGION",
                "ROUTE_SCENE_OUT_OF_FRAME",
                "ROUTE_SCENE_FULLY_OCCLUDED",
            ],
            "visibility_evidence_window": VISIBILITY_EVIDENCE_WINDOW,
            "visibility_must_not_be_derived_from": [
                "path_relation",
                "motion_relation",
                "phase",
                "route_certainty",
                "evidence_quality",
                "actionability",
            ],
            "path_relation_policy_version": "path_relation_observability_v2",
            "path_relation_definition": "BLOCKING_PATH means a localized physical region or edge intersects the current route corridor or leaves no continuous passable width. NON_BLOCKING_PATH means the current route corridor is visibly clear, the region is outside it, or a continuous passable route around it is visible. AMBIGUOUS is reserved for an unlocalized route/region relation or hidden relevant width/edge; it is not a proxy for actionability uncertainty or unknown object class.",
            "path_relation_evidence_window": GEOMETRIC_EVIDENCE_WINDOW,
            "route_certainty_policy_version": "route_certainty_observability_v2",
            "route_certainty_definition": "SINGLE_PLAUSIBLE_ROUTE means one connected forward continuation of the current support surface is identifiable. MULTIPLE_PLAUSIBLE_ROUTES requires two or more materially distinct connected walkable continuations. Do not count a vehicle lane, parallel/background walkway or hypothetical detour.",
            "route_certainty_evidence_window": GEOMETRIC_EVIDENCE_WINDOW,
            "evidence_quality_policy_version": "evidence_quality_observability_v2",
            "evidence_quality_definition": "CLEAR is interpretable route/candidate geometry; BLUR, OCCLUSION and CAMERA_ROTATION are used only when they materially prevent that interpretation; minor softness and normal shadows remain CLEAR; INSUFFICIENT is reserved for no usable route anchor. Precedence is INSUFFICIENT, CAMERA_ROTATION, OCCLUSION, BLUR, CLEAR.",
            "evidence_quality_evidence_window": GEOMETRIC_EVIDENCE_WINDOW,
            "geometric_fields_must_not_use_temporal_frames": ["path_relation", "route_certainty", "evidence_quality"],
            "temporal_fields_evidence_window": CAUSAL_TEMPORAL_EVIDENCE_WINDOW if view == "CAUSAL" else RETROSPECTIVE_TEMPORAL_EVIDENCE_WINDOW,
        },
        "items": items,
        "submission_shape": {
            "schema_version": REVIEW_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "reviewer_role": role,
            "view": view,
            "future_frame_access": view == "RETROSPECTIVE",
            "primitive_policy_version": PRIMITIVE_POLICY_VERSION,
            "visibility_evidence_window": VISIBILITY_EVIDENCE_WINDOW,
            "field_evidence_windows": FIELD_EVIDENCE_WINDOWS,
            "temporal_fields_evidence_window": CAUSAL_TEMPORAL_EVIDENCE_WINDOW if view == "CAUSAL" else RETROSPECTIVE_TEMPORAL_EVIDENCE_WINDOW,
            "sealed_before_pair_selection": True,
            "isolated_context": True,
            "metadata_blind": True,
            "other_review_visible_before_submission": False,
            "model_output_visible": False,
            "candidate_metadata_visible": False,
            "selection_reason_visible": False,
            "semantic_bucket_visible": False,
            "source_session_visible": False,
            "items": [{
                "review_item_id": "copy from packet",
                "primitive_observations": [{
                    "frame_index": "frame_ordinal",
                    **{field: "one allowed primitive value" for field in PRIMITIVE_FIELDS},
                }],
            }],
        },
    }
    return packet, private


def prepare_pilot(
    *, cohort: dict[str, Any], plan: dict[str, Any], source_root: Path, output_root: Path, event_count: int, selection_start_index: int, allow_staging: bool
) -> dict[str, Any]:
    _require(8 <= event_count <= 12, "burned pilot event count must be 8-12")
    _require(selection_start_index >= 0, "selection start index must be non-negative")
    _require(not output_root.exists(), f"refusing to overwrite pilot output root: {output_root}")
    cohort_items = _validate_cohort(cohort)
    plan_by_event = _validate_plan(plan, cohort)
    source_state = _source_state(source_root, plan, cohort, allow_staging)
    selected = cohort_items[selection_start_index : selection_start_index + event_count]
    _require(len(selected) == event_count, "selection slice exceeds the frozen screening cohort")
    event_rows: list[dict[str, Any]] = []
    for ordinal, cohort_item in enumerate(selected, start=1):
        source_event_id = cohort_item["screening_event_id"]
        plan_item = plan_by_event.get(source_event_id)
        _require(plan_item is not None, f"{source_event_id}: missing native plan item")
        verified_frames = _verify_rgb_window(source_root, source_event_id, plan_item)
        window = cohort_item["source_window"]
        _require(len(verified_frames) == window["frame_count"], f"{source_event_id}: source window/frame count mismatch")
        fps = float(plan_item.get("source_fps", 0))
        _require(fps > 0, f"{source_event_id}: invalid source fps")
        timestamps = [round(index * 1000 / fps) for index in range(len(verified_frames))]
        event_rows.append({
            "pilot_event_id": f"pilot-event-{ordinal:03d}",
            "source_screening_event_id": source_event_id,
            "source_session_id": cohort_item["source_session_id"],
            "discovery_arm": "source_mask",
            "source_window": {
                "start_frame": window["start_frame"],
                "frame_count": window["frame_count"],
                "selection_time_slots": list(range(window["frame_count"])),
            },
            "source_fps": fps,
            "frame_indices": list(range(len(verified_frames))),
            "frame_timestamps_ms": timestamps,
            "verified_frames": verified_frames,
            "label_provenance": {
                "truth_constructible_without_yolo": True,
                "yolo_boxes_used": False,
                "oracle_outputs_used": False,
                "model_outputs_visible": False,
            },
        })
    staged = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    try:
        reviewer_specs = (("CAUSAL_A", "CAUSAL", "causal-a"), ("CAUSAL_B", "CAUSAL", "causal-b"), ("RETROSPECTIVE_C", "RETROSPECTIVE", "retrospective-c"))
        packet_receipts: dict[str, Any] = {}
        all_private: dict[str, Any] = {}
        review_map_items: list[dict[str, str]] = []
        for role, view, directory_name in reviewer_specs:
            reviewer_root = staged / "reviewer-packets" / directory_name
            packet, private = _render_packet(role=role, view=view, event_rows=event_rows, source_root=source_root, staged_root=reviewer_root)
            reviewer_root.mkdir(parents=True, exist_ok=True)
            packet_path = reviewer_root / "packet.json"
            packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            packet_receipts[role] = {
                "path": packet_path.relative_to(staged).as_posix(),
                "sha256": sha256_file(packet_path),
                "item_count": len(packet["items"]),
                "view": view,
            }
            all_private[role] = private
            for opaque_id, private_row in private.items():
                review_map_items.append({"review_item_id": opaque_id, "parent_event_id": private_row["pilot_event_id"], "reviewer_role": role})
        freeze = {
            "schema_version": FREEZE_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "CALIBRATION_BURNED_OUTPUT_BLIND_EVENTS_FROZEN",
            "cohort_role": "CALIBRATION_BURNED",
            "selection_rule": "screening_event_id_ascending_deterministic_slice",
            "selection_start_index": selection_start_index,
            "selection_end_index_exclusive": selection_start_index + event_count,
            "event_count": len(event_rows),
            "formal_denominator_inclusion": False,
            "source_role": "SOURCE_MASK_DISCOVERY_ONLY_CANDIDATE_INPUT",
            "source_materialization": source_state,
            "screening_cohort_sha256": sha256_json(cohort),
            "native_input_plan_sha256": sha256_json(plan),
            "candidate_outputs_opened": False,
            "reviews_created": False,
            "pairs_created": False,
            "oracle_traces_created": False,
            "items": [{
                key: row[key]
                for key in (
                    "pilot_event_id",
                    "source_screening_event_id",
                    "source_session_id",
                    "discovery_arm",
                    "source_window",
                    "source_fps",
                    "frame_indices",
                    "frame_timestamps_ms",
                    "label_provenance",
                )
            } for row in event_rows],
            "next_gate": "Complete and seal CAUSAL_A, CAUSAL_B and RETROSPECTIVE_C primitive reviews before any YOLO pair selection or oracle trace access.",
        }
        custodian = staged / "custodian"
        custodian.mkdir(parents=True, exist_ok=True)
        freeze_path = custodian / "pilot-event-freeze.json"
        freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        review_map = {
            "schema_version": REVIEW_MAP_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "PRIVATE_MAP_FROZEN_BEFORE_REVIEW_SUBMISSIONS",
            "pilot_freeze_sha256": sha256_file(freeze_path),
            "items": review_map_items,
        }
        (custodian / "review-map-seed.json").write_text(json.dumps(review_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        private = {
            "schema_version": PRIVATE_MAP_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "PRIVATE_MAP_FROZEN_BEFORE_REVIEW_SUBMISSIONS",
            "pilot_freeze_sha256": sha256_file(freeze_path),
            "review_map_sha256": sha256_json(review_map),
            "reviewer_packets": packet_receipts,
            "reviewer_maps": all_private,
            "sharing_rule": "Never disclose custodian/ or review-map-seed.json to a reviewer. Each reviewer receives only their own reviewer-packets directory.",
        }
        (custodian / "private-review-map.json").write_text(json.dumps(private, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        receipt = {
            "schema_version": "blindassist.eval_validity_r0.judge_pilot_pre_review_receipt.v1",
            "protocol_id": PROTOCOL_ID,
            "status": "PILOT_EVENTS_AND_OPAQUE_PACKETS_FROZEN_REVIEW_PENDING",
            "pilot_freeze_sha256": sha256_file(freeze_path),
            "event_count": len(event_rows),
            "reviewer_packet_roles": [spec[0] for spec in reviewer_specs],
            "reviewer_packet_receipts": packet_receipts,
            "source_materialization": source_state,
            "formal_review_access": False,
            "formal_denominator_inclusion": False,
            "pairs_not_created": True,
            "oracle_traces_not_created": True,
            "next_gate": "Receive three independent primitive review submissions, validate review schema, then seal bundle hash.",
        }
        (staged / "pilot-pre-review-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staged, output_root)
        return receipt
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-cohort", type=Path, required=True)
    parser.add_argument("--native-input-plan", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--event-count", type=int, default=8)
    parser.add_argument("--selection-start-index", type=int, default=0)
    parser.add_argument("--allow-in-progress-staging", action="store_true")
    args = parser.parse_args()
    result = prepare_pilot(
        cohort=read_json(args.screening_cohort),
        plan=read_json(args.native_input_plan),
        source_root=args.source_root,
        output_root=args.output_root,
        event_count=args.event_count,
        selection_start_index=args.selection_start_index,
        allow_staging=args.allow_in_progress_staging,
    )
    print(f"status={result['status']} event_count={result['event_count']} output={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
