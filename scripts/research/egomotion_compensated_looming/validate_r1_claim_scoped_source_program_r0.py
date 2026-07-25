#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_ARMS = {
    "RAW_FLOW_ENERGY",
    "BBOX_LOG_AREA_GROWTH",
    "UNCOMPENSATED_LOCAL_RADIAL_EXPANSION",
    "ORACLE_ROTATION_COMPENSATION",
    "FULL_6DOF_RESIDUAL_DIAGNOSTIC",
}
EXPECTED_UNIT_FIELDS = {
    "claim_id",
    "source_family",
    "capture_cluster_id",
    "session_id",
    "unit_id",
    "eligible",
    "evaluated",
    "abstained",
    "abstention_reason",
    "evidence_grade",
    "truth_provenance",
    "interpolated_fraction",
    "time_sync_status",
    "transform_chain_status",
}
ROOT = Path(__file__).resolve().parents[3]
R1_EVIDENCE_DIR = (
    ROOT
    / "artifacts.local"
    / "evidence"
    / "ustrf"
    / "egomotion_compensated_looming_r1"
)
EXPECTED_BONN_PRIOR_INSPECTED = {
    "rgbd_bonn_crowd",
    "rgbd_bonn_moving_obstructing_box",
    "rgbd_bonn_person_tracking",
}
EXPECTED_BONN_SELECTED = {
    "discovery": {
        "rgbd_bonn_person_tracking2",
        "rgbd_bonn_balloon",
    },
    "validation": {
        "rgbd_bonn_placing_nonobstructing_box3",
        "rgbd_bonn_removing_nonobstructing_box2",
    },
    "sealed_holdout": {
        "rgbd_bonn_moving_nonobstructing_box",
        "rgbd_bonn_crowd3",
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(receipt: dict[str, Any]) -> None:
    require(
        receipt.get("schema_version")
        == "egomotion_compensated_looming_r1_claim_scoped_source_program_r0",
        "schema mismatch",
    )
    require(receipt.get("status") == "VALID", "status must be VALID")
    require(
        receipt.get("terminal")
        == (
            "R1_CLAIM_SCOPED_SOURCE_PROGRAM_NONAUTHORITATIVE_EVALUATION_"
            "QUARANTINED_INPUT_AUTHORITY_BLOCKED"
        ),
        "terminal mismatch",
    )
    require(
        receipt.get("source_search_freeze_end_inclusive") == "2026-08-08",
        "source search freeze mismatch",
    )

    claims = receipt.get("claims")
    require(isinstance(claims, dict) and len(claims) == 3, "three claims required")
    for claim_id, claim in claims.items():
        families = claim.get("support_families", [])
        require(
            isinstance(families, list) and len(set(families)) == 2,
            f"{claim_id} must bind two independent families",
        )
        require(
            isinstance(claim.get("required_evidence"), list)
            and claim["required_evidence"],
            f"{claim_id} required evidence missing",
        )

    sources = {
        item["source_family"]: item
        for item in receipt.get("sources", [])
        if isinstance(item, dict) and isinstance(item.get("source_family"), str)
    }
    require(
        {
            "CONTROLLED_RIGID_TARGET_CAPTURE",
            "BONN_RGBD_DYNAMIC",
            "REVEL",
            "JRDB",
            "HOT3D",
        }
        <= sources.keys(),
        "source program incomplete",
    )
    require(
        sources["CONTROLLED_RIGID_TARGET_CAPTURE"].get("status")
        == "HOLD_CONTROLLED_CAPTURE_HARDWARE_RECEIPT",
        "controlled capture must remain on hardware hold",
    )
    bonn = sources["BONN_RGBD_DYNAMIC"]
    require(
        bonn.get("status")
        == "DISCOVERY_TRACE_FROZEN_NONAUTHORITATIVE_SCORING_QUARANTINED",
        "Bonn source status mismatch",
    )
    bonn_prior = bonn.get("prior_inspected_units", [])
    require(
        {
            unit.get("session_id")
            for unit in bonn_prior
            if isinstance(unit, dict)
        }
        == EXPECTED_BONN_PRIOR_INSPECTED,
        "Bonn prior-inspected denylist mismatch",
    )
    require(
        all(
            unit.get("disposition") == "PRIOR_INSPECTED_ENGINEERING_ONLY"
            for unit in bonn_prior
        ),
        "prior Bonn session not quarantined",
    )
    bonn_receipt_ref = bonn.get("claim_scoped_role_freeze_receipt", {})
    bonn_receipt_path = R1_EVIDENCE_DIR / str(bonn_receipt_ref.get("path", ""))
    require(bonn_receipt_path.is_file(), "Bonn role-freeze receipt missing")
    require(
        hashlib.sha256(bonn_receipt_path.read_bytes()).hexdigest()
        == bonn_receipt_ref.get("sha256"),
        "Bonn role-freeze receipt hash mismatch",
    )
    bonn_receipt = json.loads(bonn_receipt_path.read_text(encoding="utf-8"))
    require(
        bonn_receipt.get("terminal")
        == "BONN_CLAIM_SCOPED_ROLE_FREEZE_READY_PAYLOAD_NOT_ACQUIRED",
        "Bonn role-freeze terminal mismatch",
    )
    selection_contract = bonn_receipt.get("selection_contract", {})
    require(
        selection_contract.get("cohort_identity_sha256")
        == bonn_receipt_ref.get("cohort_identity_sha256"),
        "Bonn cohort identity mismatch",
    )
    selected = selection_contract.get("selected", [])
    require(
        len(selected) == bonn_receipt_ref.get("selected_sequence_count") == 6,
        "Bonn selected sequence count mismatch",
    )
    selected_by_role = {
        role: {
            unit.get("sequence_id")
            for unit in selected
            if unit.get("role") == role
        }
        for role in EXPECTED_BONN_SELECTED
    }
    require(
        selected_by_role == EXPECTED_BONN_SELECTED,
        "Bonn role assignment mismatch",
    )
    require(
        bonn_receipt.get("read_firewall", {}).get(
            "selected_archive_download_count"
        )
        == 0
        and bonn_receipt.get("read_firewall", {}).get(
            "selected_rgb_decode_count"
        )
        == 0,
        "Bonn selected payload boundary violated",
    )
    acquisition_ref = bonn.get("discovery_archive_audit_receipt", {})
    acquisition_path = R1_EVIDENCE_DIR / str(acquisition_ref.get("path", ""))
    require(acquisition_path.is_file(), "Bonn discovery acquisition receipt missing")
    require(
        hashlib.sha256(acquisition_path.read_bytes()).hexdigest()
        == acquisition_ref.get("sha256"),
        "Bonn discovery acquisition receipt hash mismatch",
    )
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    require(
        acquisition.get("terminal")
        == "BONN_DISCOVERY_ARCHIVES_ACQUIRED_METADATA_VALID_EXTRACTION_NOT_RUN",
        "Bonn discovery acquisition terminal mismatch",
    )
    require(
        {item.get("sequence_id") for item in acquisition.get("archives", [])}
        == EXPECTED_BONN_SELECTED["discovery"],
        "Bonn discovery archive identities mismatch",
    )
    require(
        acquisition.get("read_firewall", {}).get("image_member_decode_count") == 0
        and acquisition.get("sealed_roles", {}).get(
            "validation_archive_read_count"
        )
        == 0
        and acquisition.get("sealed_roles", {}).get("holdout_archive_read_count")
        == 0,
        "Bonn discovery acquisition read boundary violated",
    )
    pose_ref = bonn.get("pose_cell_ledger_receipt", {})
    pose_path = R1_EVIDENCE_DIR / str(pose_ref.get("path", ""))
    require(pose_path.is_file(), "Bonn pose ledger missing")
    require(
        hashlib.sha256(pose_path.read_bytes()).hexdigest() == pose_ref.get("sha256"),
        "Bonn pose ledger hash mismatch",
    )
    pose_ledger = json.loads(pose_path.read_text(encoding="utf-8"))
    require(
        pose_ledger.get("terminal")
        == "BONN_DISCOVERY_POSE_MECHANICS_LEDGER_AVAILABLE_CELL_TRUTH_PENDING",
        "Bonn pose ledger terminal mismatch",
    )
    require(
        pose_ledger.get("counts", {}).get("c1_pose_mechanics_candidate_count") == 0
        and pose_ledger.get("counts", {}).get(
            "c2_translation_mechanics_candidate_count"
        )
        == 2,
        "Bonn pose mechanics counts mismatch",
    )
    require(
        pose_ledger.get("read_firewall", {}).get(
            "image_member_read_or_decode_count"
        )
        == 0
        and pose_ledger.get("read_firewall", {}).get(
            "candidate_signal_computed"
        )
        is False,
        "Bonn pose ledger read boundary violated",
    )
    map_ref = bonn.get("static_map_acquisition_receipt", {})
    map_path = R1_EVIDENCE_DIR / str(map_ref.get("path", ""))
    require(map_path.is_file(), "Bonn static-map receipt missing")
    map_receipt = json.loads(map_path.read_text(encoding="utf-8"))
    require(
        map_receipt.get("terminal")
        == "BONN_STATIC_MAP_ACQUIRED_EXTRACTION_AND_TRANSFORM_VALIDATION_PENDING",
        "Bonn static-map terminal mismatch",
    )
    require(
        map_receipt.get("archive_sha256") == map_ref.get("archive_sha256")
        and map_receipt.get("point_member_extracted") is False
        and map_receipt.get("candidate_signal_computed") is False,
        "Bonn static-map boundary violated",
    )
    geometry_ref = bonn.get("static_map_geometry_receipt", {})
    geometry_path = R1_EVIDENCE_DIR / str(geometry_ref.get("path", ""))
    require(geometry_path.is_file(), "Bonn static-map geometry receipt missing")
    require(
        hashlib.sha256(geometry_path.read_bytes()).hexdigest()
        == geometry_ref.get("sha256"),
        "Bonn static-map geometry receipt hash mismatch",
    )
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    require(
        geometry.get("terminal")
        == "BONN_STATIC_MAP_STREAM_AUDITED_DOWNSAMPLE_AVAILABLE_TRANSFORM_VALIDATION_PENDING"
        and geometry.get("stream_audit", {}).get("point_record_read_count")
        == 54_676_774
        and geometry.get("stream_audit", {}).get("selected_point_count")
        == 856_075
        and geometry.get("read_firewall", {}).get(
            "rgb_member_read_or_decode_count"
        )
        == 0
        and geometry.get("read_firewall", {}).get(
            "depth_member_read_or_decode_count"
        )
        == 0,
        "Bonn static-map geometry boundary violated",
    )
    sample_ref = bonn.get("transform_validation_sample_freeze_receipt", {})
    sample_path = R1_EVIDENCE_DIR / str(sample_ref.get("path", ""))
    require(sample_path.is_file(), "Bonn transform sample freeze missing")
    require(
        hashlib.sha256(sample_path.read_bytes()).hexdigest()
        == sample_ref.get("sha256"),
        "Bonn transform sample freeze hash mismatch",
    )
    sample_freeze = json.loads(sample_path.read_text(encoding="utf-8"))
    require(
        sample_freeze.get("terminal")
        == "BONN_TRANSFORM_VALIDATION_SAMPLES_FROZEN"
        and sample_freeze.get("counts", {}).get("sample_count") == 6
        and sample_freeze.get("counts", {}).get(
            "depth_member_read_or_decode_count_at_freeze"
        )
        == 0,
        "Bonn transform sample freeze boundary violated",
    )
    truth_ref = bonn.get("static_surface_truth_receipt", {})
    truth_path = R1_EVIDENCE_DIR / str(truth_ref.get("path", ""))
    require(truth_path.is_file(), "Bonn static-surface truth receipt missing")
    require(
        hashlib.sha256(truth_path.read_bytes()).hexdigest()
        == truth_ref.get("sha256"),
        "Bonn static-surface truth receipt hash mismatch",
    )
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    require(
        truth.get("terminal")
        == "BONN_C2_STATIC_SURFACE_TRANSFORM_CANARY_FAILED"
        and truth.get("transform_canary", {}).get("usable_depth_frame_count")
        == 3
        and truth.get("contract", {})
        .get("depth_transform_canary", {})
        .get("minimum_usable_frames")
        == 4
        and truth.get("counts", {}).get("eligible_cell_trajectory_count") == 0
        and truth.get("counts", {}).get("candidate_signal_computed") is False
        and truth.get("counts", {}).get("rgb_member_read_or_decode_count") == 0
        and truth.get("counts", {}).get("validation_or_holdout_read_count") == 0,
        "Bonn static-surface truth boundary violated",
    )
    parallel_review_ref = bonn.get(
        "parallel_static_surface_ledger_authority_review", {}
    )
    parallel_review_path = R1_EVIDENCE_DIR / str(
        parallel_review_ref.get("path", "")
    )
    require(
        parallel_review_path.is_file(),
        "Bonn parallel-ledger authority review missing",
    )
    require(
        hashlib.sha256(parallel_review_path.read_bytes()).hexdigest()
        == parallel_review_ref.get("sha256"),
        "Bonn parallel-ledger authority review hash mismatch",
    )
    parallel_review = json.loads(
        parallel_review_path.read_text(encoding="utf-8")
    )
    parallel_receipt_path = R1_EVIDENCE_DIR / str(
        parallel_review_ref.get("parallel_receipt_path", "")
    )
    require(
        parallel_receipt_path.is_file()
        and hashlib.sha256(parallel_receipt_path.read_bytes()).hexdigest()
        == parallel_review_ref.get("parallel_receipt_sha256"),
        "Bonn parallel-ledger receipt identity mismatch",
    )
    require(
        parallel_review.get("terminal")
        == "PARALLEL_BONN_STATIC_SURFACE_LEDGER_QUARANTINED_AS_DIAGNOSTIC"
        and parallel_review.get("disposition")
        == "EXPLORATORY_DIAGNOSTIC_ONLY_NOT_CONFIRMATORY_AUTHORITY"
        and parallel_review_ref.get(
            "counts_toward_confirmation_or_signal_authority"
        )
        is False,
        "Bonn parallel ledger gained authority",
    )
    trace_review_ref = bonn.get(
        "parallel_base_flow_trace_boundary_review", {}
    )
    trace_review_path = R1_EVIDENCE_DIR / str(
        trace_review_ref.get("path", "")
    )
    require(
        trace_review_path.is_file(),
        "Bonn base-trace boundary review missing",
    )
    require(
        hashlib.sha256(trace_review_path.read_bytes()).hexdigest()
        == trace_review_ref.get("sha256"),
        "Bonn base-trace boundary review hash mismatch",
    )
    trace_path = R1_EVIDENCE_DIR / str(trace_review_ref.get("trace_path", ""))
    require(
        trace_path.is_file()
        and hashlib.sha256(trace_path.read_bytes()).hexdigest()
        == trace_review_ref.get("trace_sha256"),
        "Bonn base trace identity mismatch",
    )
    trace_review = json.loads(trace_review_path.read_text(encoding="utf-8"))
    require(
        trace_review.get("terminal")
        == "BONN_PARALLEL_BASE_FLOW_TRACE_RETAINED_WITHOUT_RESULT_AUTHORITY"
        and trace_review.get("authority_disposition")
        == "BASE_TRACE_ONLY_RETAINED_SCORING_AND_ORACLE_JOIN_BLOCKED"
        and trace_review.get("actual_execution", {}).get(
            "unique_discovery_rgb_member_decode_count"
        )
        == 598
        and trace_review.get("actual_execution", {}).get("pair_count") == 596
        and trace_review.get("actual_execution", {}).get(
            "truth_join_or_scoring_run"
        )
        is False
        and trace_review.get("actual_execution", {}).get(
            "algorithm_result_available"
        )
        is False
        and trace_review_ref.get("counts_toward_claim_or_algorithm_result")
        is False,
        "Bonn base trace gained result authority",
    )
    oracle_review_ref = bonn.get(
        "parallel_oracle_flow_trace_boundary_review", {}
    )
    oracle_review_path = R1_EVIDENCE_DIR / str(
        oracle_review_ref.get("path", "")
    )
    require(
        oracle_review_path.is_file(),
        "Bonn oracle-trace boundary review missing",
    )
    require(
        hashlib.sha256(oracle_review_path.read_bytes()).hexdigest()
        == oracle_review_ref.get("sha256"),
        "Bonn oracle-trace boundary review hash mismatch",
    )
    oracle_trace_path = R1_EVIDENCE_DIR / str(
        oracle_review_ref.get("trace_path", "")
    )
    require(
        oracle_trace_path.is_file()
        and hashlib.sha256(oracle_trace_path.read_bytes()).hexdigest()
        == oracle_review_ref.get("trace_sha256"),
        "Bonn oracle trace identity mismatch",
    )
    oracle_review = json.loads(
        oracle_review_path.read_text(encoding="utf-8")
    )
    require(
        oracle_review.get("terminal")
        == "BONN_PARALLEL_ORACLE_FLOW_TRACE_RETAINED_WITHOUT_RESULT_AUTHORITY"
        and oracle_review.get("authority_disposition")
        == "ORACLE_TRACE_ONLY_RETAINED_TRUTH_JOIN_AND_SCORING_BLOCKED"
        and oracle_review.get("actual_execution", {}).get(
            "oracle_evaluated_pair_count"
        )
        == 594
        and oracle_review.get("actual_execution", {}).get(
            "depth_member_decode_count"
        )
        == 594
        and oracle_review.get("actual_execution", {}).get(
            "closing_truth_ledger_read"
        )
        is False
        and oracle_review.get("actual_execution", {}).get(
            "truth_join_or_scoring_run"
        )
        is False
        and oracle_review.get("actual_execution", {}).get(
            "algorithm_result_available"
        )
        is False
        and oracle_review_ref.get("counts_toward_claim_or_algorithm_result")
        is False,
        "Bonn oracle trace gained result authority",
    )
    evaluation_review_ref = bonn.get(
        "nonauthoritative_continuous_signal_evaluation_review", {}
    )
    evaluation_review_path = R1_EVIDENCE_DIR / str(
        evaluation_review_ref.get("path", "")
    )
    require(
        evaluation_review_path.is_file(),
        "Bonn nonauthoritative evaluation review missing",
    )
    require(
        hashlib.sha256(evaluation_review_path.read_bytes()).hexdigest()
        == evaluation_review_ref.get("sha256"),
        "Bonn nonauthoritative evaluation review hash mismatch",
    )
    evaluation_path = R1_EVIDENCE_DIR / str(
        evaluation_review_ref.get("evaluation_path", "")
    )
    require(
        evaluation_path.is_file()
        and hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
        == evaluation_review_ref.get("evaluation_sha256"),
        "Bonn nonauthoritative evaluation identity mismatch",
    )
    evaluation_review = json.loads(
        evaluation_review_path.read_text(encoding="utf-8")
    )
    evaluation_recheck_path = R1_EVIDENCE_DIR / str(
        evaluation_review.get("reviewed_evaluation", {}).get(
            "independent_recheck_path", ""
        )
    )
    require(
        evaluation_review.get("terminal")
        == "BONN_NONAUTHORITATIVE_CONTINUOUS_SIGNAL_EVALUATION_QUARANTINED"
        and evaluation_review.get("disposition")
        == "QUARANTINED_NONAUTHORITATIVE_DIAGNOSTIC_GLOBAL_TO_CENTRAL_ROI_ASSOCIATION_WEAK"
        and evaluation_review.get("actual_execution", {}).get(
            "truth_join_or_scoring_run"
        )
        is True
        and evaluation_review.get("actual_execution", {}).get(
            "candidate_signal_result_evaluated"
        )
        is True
        and evaluation_review.get("authority_review", {}).get(
            "evaluation_had_claim_or_stop_authority"
        )
        is False
        and evaluation_review.get("authority_review", {}).get(
            "signal_truth_spatial_units_aligned"
        )
        is False
        and evaluation_review.get("authority_review", {}).get(
            "truth_self_reported_evidence_grade"
        )
        == "A"
        and evaluation_review.get("authority_review", {}).get(
            "frozen_required_grade_for_derived_independent_reconstruction"
        )
        == "B"
        and evaluation_review.get("allowed_claim")
        == "THE_CURRENT_GLOBAL_Q90_SUMMARY_HAS_WEAK_AND_SESSION_UNSTABLE_ASSOCIATION_WITH_AN_EXPLORATORY_CENTRAL_ROI_STATIC_DEPTH_PROXY"
        and evaluation_review.get("reviewed_evaluation", {}).get(
            "exact_reproduction_match"
        )
        is True
        and evaluation_recheck_path.is_file()
        and hashlib.sha256(evaluation_recheck_path.read_bytes()).hexdigest()
        == evaluation_review.get("reviewed_evaluation", {}).get(
            "independent_recheck_sha256"
        )
        and evaluation_review_ref.get(
            "counts_toward_claim_or_algorithm_result"
        )
        is False,
        "Bonn nonauthoritative evaluation gained result authority",
    )
    quarantine_ref = bonn.get("unselected_payload_quarantine_receipt", {})
    quarantine_path = R1_EVIDENCE_DIR / str(quarantine_ref.get("path", ""))
    require(quarantine_path.is_file(), "Bonn quarantine receipt missing")
    quarantine = json.loads(quarantine_path.read_text(encoding="utf-8"))
    require(
        quarantine.get("disposition")
        == "ACCIDENTAL_UNSELECTED_PAYLOAD_NEVER_EVALUATE"
        and quarantine.get("counts_toward_any_role_or_claim") is False
        and quarantine.get("image_decode_count") == 0,
        "Bonn quarantine boundary violated",
    )
    require(
        sources["REVEL"]["prior_inspected_units"][0].get("disposition")
        == "PRIOR_INSPECTED_DISCOVERY_OR_MIGRATION_ONLY",
        "prior REveL session not quarantined",
    )
    require(
        sources["REVEL"].get("same_bag_segmented_role_split_allowed") is False,
        "REveL same-bag role split must be forbidden",
    )
    require(
        sources["JRDB"].get("counts_toward_claim_confirmation") is False,
        "JRDB must remain diagnostic-only",
    )
    require(
        sources["HOT3D"].get("clip_tar_read_count") == 0
        and sources["HOT3D"].get("candidate_image_byte_read_count") == 0,
        "HOT3D payload boundary violated",
    )
    require(set(receipt.get("r1a_arms", [])) == EXPECTED_ARMS, "R1-A arms mismatch")
    require(
        set(receipt.get("required_unit_fields", [])) == EXPECTED_UNIT_FIELDS,
        "unit ledger fields mismatch",
    )

    boundaries = receipt.get("hard_boundaries", {})
    require(
        boundaries.get("old_15_pair_window_selection_tuning_acceptance_reads") == 0,
        "old-window boundary violated",
    )
    require(boundaries.get("route_or_event_truth_used") is False, "route/event used")
    require(boundaries.get("alarm_threshold_selected") is False, "threshold selected")
    require(
        boundaries.get("app_route_or_lifecycle_connected") is False,
        "product boundary violated",
    )
    require(
        boundaries.get("candidate_signal_computed") is True,
        "actual base trace execution is not recorded",
    )
    require(
        boundaries.get("candidate_signal_result_evaluated") is True
        and boundaries.get("oracle_trace_computed") is True
        and boundaries.get("truth_join_or_scoring_run") is True
        and boundaries.get("authoritative_algorithm_result_available")
        is False,
        "actual scoring or authority boundary mismatch",
    )
    require(
        all(source.get("signal_allowed") is False for source in sources.values()),
        "a source unexpectedly allows signal",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "artifacts.local/evidence/ustrf/egomotion_compensated_looming_r1/"
            "r1_claim_scoped_source_program_r0.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    validate(receipt)
    print(
        json.dumps(
            {
                "status": "VALID",
                "terminal": receipt["terminal"],
                "claim_count": len(receipt["claims"]),
                "signal_computed": receipt["hard_boundaries"][
                    "candidate_signal_computed"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
