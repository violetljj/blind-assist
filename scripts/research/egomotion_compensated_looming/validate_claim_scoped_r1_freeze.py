#!/usr/bin/env python3
"""Validate the R1 controlled-capture protocol and claim-source subset freeze."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
R1_DIR = (
    ROOT
    / "artifacts.local"
    / "evidence"
    / "ustrf"
    / "egomotion_compensated_looming_r1"
)
PROTOCOL = R1_DIR / "controlled_capture" / "controlled_capture_protocol_receipt_r1.json"
SOURCE_PROGRAM = R1_DIR / "r1_claim_scoped_source_program_r0.json"


def _load(path: Path) -> dict:
    if not path.is_file():
        raise AssertionError(f"missing R1 receipt: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol(receipt: dict) -> None:
    assert receipt["protocol_status"] == "FROZEN_NOT_CAPTURED"
    assert receipt["human_participant_count"] == 0
    assert receipt["recorded_trial_count"] == 0
    assert receipt["minimum_capture_clusters"] == 3
    assert receipt["minimum_sessions_per_cluster"] == 1
    assert receipt["minimum_trial_count"] == (
        sum(receipt["trial_matrix_per_cluster"].values())
        * receipt["minimum_capture_clusters"]
    ) == 84
    assert set(receipt["common_required_roles"]) == {
        "CAMERA_RIG",
        "TIME_AUTHORITY",
        "SAFETY_CONTROLLER",
        "AUDIT_STORAGE",
    }
    claim_roles = receipt["claim_specific_required_roles"]
    assert "LINEAR_TRUTH" not in claim_roles["C1_ROTATION_LEAKAGE_SUPPRESSION"]
    assert "FULL_POSE_TRUTH" in claim_roles["C1_ROTATION_LEAKAGE_SUPPRESSION"]
    assert "FULL_POSE_TRUTH" not in claim_roles[
        "C2_STATIC_SURFACE_CLOSING_RETENTION"
    ]
    assert receipt["truth_contract"]["target_center_distance_forbidden"] is True
    assert receipt["truth_contract"]["signal_truth_shared_ancestor_forbidden"] is True
    assert receipt["isolation"]["producer_may_read_truth_or_cell"] is False
    assert receipt["isolation"]["decoded_pixel_and_near_duplicate_firewall_required"] is True
    assert receipt["isolation"]["r1a_open_role"] == "DISCOVERY_CONTROLLED_RIGID_R1"
    assert receipt["isolation"]["minimum_independent_discovery_clusters"] == 3
    assert receipt["isolation"][
        "validation_and_holdout_capture_allowed_in_r1a"
    ] is False
    assert receipt["isolation"]["future_validation_and_holdout_require_new_clusters"] is True
    assert receipt["automation"]["human_collection_or_acceptance_queue_allowed"] is False
    assert receipt["automation"]["visible_truth_marker_allowed"] is False
    assert receipt["authority"]["may_capture_now"] is False
    assert receipt["authority"]["may_run_signal"] is False
    assert receipt["authority"]["may_capture_human_or_safety_outcomes"] is False
    assert receipt["terminal"] == (
        "CONTROLLED_RIGID_TARGET_PROTOCOL_FROZEN_DEVICE_MANIFEST_REQUIRED"
    )
    assert receipt["status"] == "VALID"


def validate_source_program(receipt: dict) -> None:
    assert receipt["source_search_freeze_end_inclusive"] == "2026-08-08"
    assert set(receipt["claims"]) == {
        "C1_ROTATION_LEAKAGE_SUPPRESSION",
        "C2_STATIC_SURFACE_CLOSING_RETENTION",
        "C2_ACTIVE_TARGET_AND_LATERAL_PASS",
    }
    sources = {item["source_family"]: item for item in receipt["sources"]}
    assert sources["CONTROLLED_RIGID_TARGET_CAPTURE"]["status"] == (
        "HOLD_CONTROLLED_CAPTURE_HARDWARE_RECEIPT"
    )
    bonn = sources["BONN_RGBD_DYNAMIC"]
    assert bonn["status"] == (
        "DISCOVERY_TRACE_FROZEN_NONAUTHORITATIVE_SCORING_QUARANTINED"
    )
    assert {
        unit["session_id"] for unit in bonn["prior_inspected_units"]
    } == {
        "rgbd_bonn_crowd",
        "rgbd_bonn_moving_obstructing_box",
        "rgbd_bonn_person_tracking",
    }
    assert all(
        unit["disposition"] == "PRIOR_INSPECTED_ENGINEERING_ONLY"
        for unit in bonn["prior_inspected_units"]
    )
    assert bonn["claim_scoped_role_freeze_receipt"]["selected_sequence_count"] == 6
    assert bonn["discovery_archive_audit_receipt"]["archive_count"] == 2
    assert bonn["discovery_archive_audit_receipt"]["image_decode_count"] == 0
    assert bonn["pose_cell_ledger_receipt"][
        "c1_pose_mechanics_candidate_count"
    ] == 0
    assert bonn["pose_cell_ledger_receipt"][
        "c2_translation_mechanics_candidate_count"
    ] == 2
    assert bonn["pose_cell_ledger_receipt"]["cell_truth_proven"] is False
    assert bonn["static_map_acquisition_receipt"][
        "point_member_extracted"
    ] is False
    assert bonn["static_map_geometry_receipt"][
        "deterministic_selected_point_count"
    ] == 856075
    assert bonn["transform_validation_sample_freeze_receipt"][
        "depth_decode_count_at_freeze"
    ] == 0
    assert bonn["static_surface_truth_receipt"]["terminal"] == (
        "BONN_C2_STATIC_SURFACE_TRANSFORM_CANARY_FAILED"
    )
    assert bonn["static_surface_truth_receipt"][
        "eligible_cell_trajectory_count"
    ] == 0
    assert bonn["static_surface_truth_receipt"][
        "candidate_signal_computed"
    ] is False
    assert bonn["nonauthoritative_continuous_signal_evaluation_review"][
        "counts_toward_claim_or_algorithm_result"
    ] is False
    assert bonn["nonauthoritative_continuous_signal_evaluation_review"][
        "terminal"
    ] == "BONN_NONAUTHORITATIVE_CONTINUOUS_SIGNAL_EVALUATION_QUARANTINED"
    assert sources["REVEL"]["prior_inspected_units"][0]["disposition"] == (
        "PRIOR_INSPECTED_DISCOVERY_OR_MIGRATION_ONLY"
    )
    assert sources["REVEL"]["same_bag_segmented_role_split_allowed"] is False
    assert sources["JRDB"]["counts_toward_claim_confirmation"] is False
    assert all(source["signal_allowed"] is False for source in sources.values())
    assert receipt["hard_boundaries"]["candidate_signal_computed"] is True
    assert receipt["hard_boundaries"][
        "candidate_signal_result_evaluated"
    ] is True
    assert receipt["hard_boundaries"]["oracle_trace_computed"] is True
    assert receipt["hard_boundaries"]["truth_join_or_scoring_run"] is True
    assert receipt["hard_boundaries"][
        "authoritative_algorithm_result_available"
    ] is False
    assert receipt["hard_boundaries"]["route_or_event_truth_used"] is False
    assert receipt["terminal"] == (
        "R1_CLAIM_SCOPED_SOURCE_PROGRAM_NONAUTHORITATIVE_EVALUATION_QUARANTINED_INPUT_AUTHORITY_BLOCKED"
    )
    assert receipt["status"] == "VALID"


def main() -> None:
    protocol = _load(PROTOCOL)
    source_program = _load(SOURCE_PROGRAM)
    validate_protocol(protocol)
    validate_source_program(source_program)
    print(
        json.dumps(
            {
                "status": "VALID",
                "protocol_terminal": protocol["terminal"],
                "source_program_terminal": source_program["terminal"],
                "recorded_trial_count": protocol["recorded_trial_count"],
                "signal_run_count": int(
                    source_program["hard_boundaries"]["candidate_signal_computed"]
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
