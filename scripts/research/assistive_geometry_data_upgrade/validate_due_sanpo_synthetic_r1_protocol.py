"""Validate the non-execution AG-DUE SANPO-Synthetic R1 audit protocol lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = REPO_ROOT / "docs/research/assistive-geometry-data-upgrade/BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R1_SANPO_SYNTHETIC_SOURCE_SPECIFIC_INTEGRITY_AND_CAPABILITY_AUDIT_PROTOCOL_LOCK_2026-08-10.json"

PROTOCOL_ID = "BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R1_SANPO_SYNTHETIC_SOURCE_SPECIFIC_INTEGRITY_AND_CAPABILITY_AUDIT_PROTOCOL_LOCK_2026-08-10"
SOURCE_ID = "sanpo_synthetic_v0_train_discovery"
SESSION_ID = "17c7d6bc6d4d4573afecc730cabf4db65f66b04ced504396a71d1185920179cb"
NAMESPACED_PARENT = f"{SOURCE_ID}:{SESSION_ID}"
SUCCESSOR = "BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R1_SANPO_SYNTHETIC_METADATA_AND_OBJECT_INVENTORY_PREFLIGHT_EXECUTION"
OUTPUT_ROOT = "artifacts.local/evidence/assistive-geometry-data-upgrade/sanpo-synthetic-r1-metadata-preflight"
OBJECT_PREFIX = "sanpo_dataset/v0"


class ProtocolError(ValueError):
    """Raised when the R1 protocol broadens scope or authority."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_protocol(protocol: dict[str, Any], repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    require(set(protocol) == {
        "schema",
        "protocol_id",
        "route_id",
        "status",
        "research_mode",
        "research_style",
        "execution_profile",
        "scientific_question",
        "information_gain",
        "minimal_discriminating_action",
        "predecessors",
        "locked_source",
        "source_object_contract",
        "phased_audit",
        "capability_claim_contract",
        "role_and_contamination",
        "output_contract",
        "execution_authority",
        "decision_tree",
        "stop_conditions",
        "implementation",
        "unique_successor",
        "claim_ceiling",
    }, "protocol field set drift")
    require(protocol["schema"] == "blindassist.assistive_geometry_due.sanpo_synthetic_r1_audit_protocol.v1", "protocol schema drift")
    require(protocol["protocol_id"] == PROTOCOL_ID, "protocol identity drift")
    require(protocol["route_id"] == "BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R1", "route identity drift")
    require(protocol["status"] == "SOURCE_SPECIFIC_AUDIT_PROTOCOL_LOCKED_EXECUTION_NOT_AUTHORIZED", "protocol status drift")
    require(protocol["research_mode"] == "REVERSIBLE_EXPLORATION", "research mode drift")
    require(protocol["research_style"] == "WILD_LAB", "research style drift")
    require(protocol["execution_profile"] == "CANARY_LITE", "execution profile drift")

    expected_predecessors = {
        "r0_prescreen_result",
        "r0_synthetic_manifest",
        "r0_manifest_lock",
        "r0_gap_contract",
        "reference_acquirer",
        "reference_replay_auditor",
        "reference_metric_depth_auditor",
        "reference_pose_limitation",
        "r2_f1_protocol",
        "r2_f1_result",
    }
    require(set(protocol["predecessors"]) == expected_predecessors, "predecessor set drift")
    expected_roles = {
        "r0_prescreen_result": "governed PARTIAL/PARTIAL result and unique-successor authority",
        "r0_synthetic_manifest": "exact source identity and metadata-only capability ceiling",
        "r0_manifest_lock": "exact manifest hash and identity lock",
        "r0_gap_contract": "unchanged DCA-derived thresholds including the 12-parent R2 F1 gate",
        "reference_acquirer": "non-authoritative format reference only; network and payload execution remain forbidden",
        "reference_replay_auditor": "non-authoritative GCS receipt, hash and numeric-index mechanics reference only",
        "reference_metric_depth_auditor": "non-authoritative depth-structure and pose-binding limitation reference only",
        "reference_pose_limitation": "tracked prior evidence that pose rows lack explicit frame/timestamp binding; not current payload evidence",
        "r2_f1_protocol": "frozen factor-label, derivation and 12-parent supervision requirements",
        "r2_f1_result": "current F1 execution blocker and claim ceiling",
    }
    for name, binding in protocol["predecessors"].items():
        require(set(binding) == {"path", "sha256", "role"}, f"predecessor binding drift: {name}")
        path = repo_root / binding["path"]
        require(path.is_file(), f"predecessor missing: {name}")
        require(binding["sha256"] == sha256_file(path), f"predecessor SHA drift: {name}")
        require(binding["role"] == expected_roles[name], f"predecessor role drift: {name}")

    source = protocol["locked_source"]
    require(source == {
        "source_id": SOURCE_ID,
        "source_family": "SANPO_SYNTHETIC",
        "source_version": "v0_official_train_split_F9C5DC4C",
        "official_split": "train",
        "official_split_sha256": "F9C5DC4C289FA87342ABC0D2CC49F112FCC78C7E02E0B6B081E296A99344173C",
        "session_id": SESSION_ID,
        "namespaced_parent_id": NAMESPACED_PARENT,
        "camera_candidate": "camera_chest",
        "lens_candidate": "left",
        "parent_count": 1,
        "roster_expansion_authorized": False,
        "fallback_session_camera_or_lens_authorized": False,
    }, "locked source drift")

    objects = protocol["source_object_contract"]
    require(objects["session_prefix"] == f"{OBJECT_PREFIX}/sanpo-synthetic/{SESSION_ID}", "session prefix drift")
    require(objects["metadata_objects"] == [
        {"role": "session_description", "path": f"{OBJECT_PREFIX}/sanpo-synthetic/{SESSION_ID}/description.json"},
        {"role": "global_labelmap", "path": f"{OBJECT_PREFIX}/labelmap.json"},
        {"role": "frame_annotation_type", "path": f"{OBJECT_PREFIX}/sanpo-synthetic/{SESSION_ID}/camera_chest/left/frame_segmentation_annotation_type.json"},
        {"role": "camera_pose_table", "path": f"{OBJECT_PREFIX}/sanpo-synthetic/{SESSION_ID}/camera_chest/camera_poses.csv"},
    ], "metadata object contract drift")
    require(objects["frame_prefixes"] == [
        {"role": "rgb", "prefix": f"{OBJECT_PREFIX}/sanpo-synthetic/{SESSION_ID}/camera_chest/left/video_frames/", "suffix": ".png"},
        {"role": "panoptic_mask", "prefix": f"{OBJECT_PREFIX}/sanpo-synthetic/{SESSION_ID}/camera_chest/left/segmentation_masks/", "suffix": ".png"},
        {"role": "metric_depth", "prefix": f"{OBJECT_PREFIX}/sanpo-synthetic/{SESSION_ID}/camera_chest/left/depth_maps/", "suffix": ".float16.gz"},
    ], "frame prefix contract drift")
    require(objects["object_receipt_fields"] == ["name", "generation", "size", "md5_hash", "crc32c", "sha256_after_read"], "object receipt fields drift")
    require(objects["frame_index_rule"] == "numeric_filename_stem_only; duplicates rejected; no gap filling", "frame index rule drift")
    require(objects["aligned_inventory_rule"] == "set intersection of exact numeric RGB, panoptic-mask and metric-depth indices; counts remain inventory-only", "aligned inventory rule drift")
    require(objects["body_canary_index_selection_rule"] == "freeze the lowest 25 complete aligned numeric indices before any frame body read; fewer than 25 is NOT_EVALUABLE; no replacement or content selection", "body canary selection rule drift")
    require(objects["observed_counts"] == {
        "rgb_objects": 0,
        "panoptic_mask_objects": 0,
        "metric_depth_objects": 0,
        "aligned_indices": 0,
        "portrait_frames": 0,
        "landscape_frames": 0,
        "support_frames": 0,
        "boundary_frames": 0,
    }, "pre-execution observed count drift")

    phases = protocol["phased_audit"]
    require(set(phases) == {"CURRENT_PROTOCOL_LOCK", "NEXT_METADATA_AND_OBJECT_INVENTORY_PREFLIGHT", "FUTURE_FRAME_BODY_CANARY"}, "audit phase set drift")
    require(phases["CURRENT_PROTOCOL_LOCK"] == {
        "executed": True,
        "allowed_reads": "tracked repository metadata and code only",
        "network": False,
        "source_object_metadata": False,
        "metadata_object_bytes": False,
        "frame_body_bytes": False,
    }, "current phase authority drift")
    next_phase = phases["NEXT_METADATA_AND_OBJECT_INVENTORY_PREFLIGHT"]
    require(next_phase["executed"] is False, "next preflight already executed")
    require(next_phase["network_authorized_by_this_lock"] is False, "network prematurely authorized")
    require(next_phase["allowed_after_explicit_successor_execution"] == [
        "exact split/session object metadata HEAD/LIST under the locked prefixes",
        "exact metadata object bytes for description, labelmap, annotation-type map and camera-pose table",
        "camera-pose header and row-count characterization without transform use",
    ], "next preflight read scope drift")
    require(next_phase["forbidden_even_when_executed"] == [
        "RGB body bytes",
        "panoptic-mask body bytes",
        "metric-depth body bytes",
        "Teacher/model output",
        "label derivation or training",
    ], "next preflight forbidden scope drift")
    require(phases["FUTURE_FRAME_BODY_CANARY"] == {
        "executed": False,
        "authorized": False,
        "requires_separate_protocol_lock": True,
        "exact_candidate_aligned_frames": 25,
        "rgb_access": "object metadata only; RGB body and visual access forbidden",
        "geometry_body_access": ["25 exact metric-depth objects", "25 exact panoptic-mask objects"],
        "requires_object_count_byte_and_disk_budget_from_preflight": True,
        "purpose": "structural source integrity only, never task truth or model evidence",
    }, "future body canary authority drift")

    claims = protocol["capability_claim_contract"]
    require(set(claims) == {
        "oracle_depth_factor",
        "oracle_support_factor",
        "r2_obstacle_boundary_truth_materialized",
        "consecutive_temporal_pair",
        "explicit_timestamp_materialized",
        "pose_transform_materialized",
        "parent_gate",
        "other_dca_gaps",
    }, "capability claim set drift")
    require(claims["oracle_depth_factor"] == {
        "after_inventory_preflight_at_most": "SOURCE_OBJECTS_PRESENT_NOT_VALIDATED_FOR_CLAIM",
        "requires_body_canary": True,
        "metric_units_must_be_verified_from_source_bytes_or_first-party schema": True,
        "invalid_sentinel_and_finite_range_policy_must_be_verified": True,
        "rgb_depth_registration_and_resolution_receipt_required": True,
        "cannot_establish_scale_accuracy_or_task_utility": True,
    }, "depth claim drift")
    require(claims["oracle_support_factor"] == {
        "status": "ABSENT",
        "depth_is_not_support_truth": True,
        "future_deterministic_derivation_contract_required": True,
    }, "support claim drift")
    require(claims["r2_obstacle_boundary_truth_materialized"] == {
        "after_inventory_preflight_at_most": "PANOPTIC_OBJECTS_PRESENT_DERIVATION_NOT_RUN",
        "validated_for_claim": False,
        "requires_label_taxonomy_mapping": True,
        "requires_frozen_boundary_connectivity_and_unknown_policy": True,
        "unknown_or_void_is_negative": False,
        "teacher_or_vlm_fill_forbidden": True,
    }, "boundary claim drift")
    require(claims["consecutive_temporal_pair"]["after_inventory_preflight_at_most"] == "ALIGNED_NUMERIC_INDEX_CANDIDATES_PRESENT", "temporal pair claim drift")
    require(claims["consecutive_temporal_pair"]["frame_body_continuity_not_yet_verified"] is True, "temporal continuity upgraded")
    require(claims["explicit_timestamp_materialized"] == {
        "status": "ABSENT",
        "index_divided_by_fps_is_derived_cadence_not_explicit_timestamp": True,
    }, "timestamp claim drift")
    require(claims["pose_transform_materialized"] == {
        "status": "NOT_ADMITTED",
        "pose_row_order_is_frame_binding": False,
        "row_count_coverage_is_frame_binding": False,
        "requires_explicit_frame_or_timestamp_mapping_and_coordinate_receipt": True,
        "requires_quaternion_order_handedness_and_transform_direction_receipt": True,
        "coordinate_axis_assumption": "UNKNOWN_UNTIL_EXACT_SESSION_VERIFIER",
        "device_body_frame_authority": False,
    }, "pose claim drift")
    require(claims["parent_gate"] == {
        "observed_parent_count": 1,
        "r2_f1_required_joint_parents": 12,
        "r2_f1_parent_gate_pass": False,
        "source_data_support_established": False,
    }, "parent gate drift")
    require(claims["other_dca_gaps"] == {
        "qsf_right_censor_supported": False,
        "corridor_supported": False,
        "fci_truth_bundle_supported": False,
    }, "other DCA gap drift")

    require(protocol["role_and_contamination"] == {
        "history_roles": ["SOURCE_DISCOVERY", "TRAIN"],
        "fresh_confirmation_claim": False,
        "selection_or_tuning_influence": False,
        "real_source_included": False,
        "roster_expansion": False,
        "cross_source_or_split_claim": False,
    }, "role or contamination drift")
    require(protocol["output_contract"] == {
        "owned_root": OUTPUT_ROOT,
        "overwrite": False,
        "required_receipts": ["source_object_inventory.json", "metadata_schema_receipt.json", "preflight_result.json"],
        "tracked_payload": False,
        "result_can_establish_source_support": False,
    }, "output contract drift")
    require(protocol["execution_authority"] == {
        "protocol_lock": True,
        "static_validation": True,
        "metadata_and_object_inventory_preflight": False,
        "network_or_remote_object_access": False,
        "source_object_listing": False,
        "local_existing_payload_open": False,
        "frame_body_download_or_open": False,
        "rgb_visual_access": False,
        "geometry_payload_access": False,
        "source_specific_capability_count_audit": False,
        "derivation_or_teacher": False,
        "data_materialization": False,
        "model_or_training": False,
        "development_or_confirmation": False,
        "android_or_default_app": False,
    }, "execution authority drift")
    require(protocol["decision_tree"] == {
        "PASS": "METADATA_INVENTORY_VALID_BODY_CANARY_PROTOCOL_LOCK_ELIGIBLE",
        "PARTIAL_POSE_UNBOUND": "DEPTH_AND_PANOPTIC_BODY_CANARY_PROTOCOL_LOCK_ELIGIBLE_POSE_TEMPORAL_HELD",
        "NOT_EVALUABLE": "STOP_SOURCE_OBJECT_INVENTORY_OR_SCHEMA_INCOMPLETE",
        "REJECT": "STOP_IDENTITY_SPLIT_LICENSE_ANCESTRY_OR_INTEGRITY_CONFLICT",
    }, "decision tree drift")
    require(len(protocol["stop_conditions"]) >= 10, "stop conditions incomplete")
    stop_text = "\n".join(protocol["stop_conditions"])
    for phrase in ("different session", "frame body", "pose row", "explicit timestamp", "support truth", "boundary truth", "Teacher", "parent gate", "output root", "source support"):
        require(phrase in stop_text, f"stop condition missing: {phrase}")

    require(set(protocol["implementation"]) == {
        "scripts/research/assistive_geometry_data_upgrade/validate_due_sanpo_synthetic_r1_protocol.py",
        "scripts/research/assistive_geometry_data_upgrade/test_validate_due_sanpo_synthetic_r1_protocol.py",
    }, "implementation set drift")
    for logical_path, expected_sha in protocol["implementation"].items():
        require(sha256_file(repo_root / logical_path) == expected_sha, f"implementation SHA drift: {logical_path}")
    require(protocol["unique_successor"] == SUCCESSOR, "successor drift")
    require(protocol["claim_ceiling"] == "Protocol/schema lock only for one exact SANPO-Synthetic TRAIN session. No network or source-object access, payload body, source support, DCA pass, F1 parent-gate pass, pose/timestamp admission, derivation, Teacher, materialization, training, Development, Confirmation, default-App, product or safety authority.", "claim ceiling drift")
    return {"protocol_id": PROTOCOL_ID, "status": "VALID", "unique_successor": SUCCESSOR}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args()
    try:
        require(args.protocol.resolve() == PROTOCOL_PATH.resolve(), "custom protocol path is not authorized")
        result = validate_protocol(load_json(args.protocol))
    except (ProtocolError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "INVALID", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
