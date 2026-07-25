#!/usr/bin/env python3
"""Rebuild and validate the independent person-trajectory truth authority audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import audit_independent_person_trajectory_truth_source_authority_and_admission_r0 as producer

VALIDATION_SCHEMA = (
    "blindassist_ustrf_independent_person_trajectory_truth_"
    "source_authority_and_admission_r0_validation"
)


def validate(repo: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ledger_path = repo / config["outputs"]["ledger"]
    receipt_path = repo / config["outputs"]["receipt"]
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rebuilt_ledger, rebuilt_receipt = producer.audit(repo, config_path)
    rebuilt_receipt["ledger_file_sha256"] = producer.sha256_file(ledger_path)

    expected_bands = [row["id"] for row in config["distance_bands_m"]]
    gates = receipt["admission_gates"]
    checks = {
        "config_identity": config["schema"] == producer.CONFIG_SCHEMA,
        "stage_identity": config["stage"] == producer.STAGE,
        "candidate_blind_freeze_status": config["status"]
        == "frozen_before_candidate_output_read",
        "candidate_outputs_invisible_in_config": config["canary"][
            "candidate_outputs_visible"
        ]
        is False,
        "source_schema_amendment_candidate_blind": config[
            "source_schema_amendment_before_candidate_output_read"
        ]["candidate_outputs_read"]
        is False,
        "forbidden_truth_substitutes_complete": {
            "JRDB 3D box center",
            "JRDB PCD point-in-box support",
            "JRDB box-conditioned point centroid",
            "manually selected trajectories or windows",
            "candidate algorithm outputs",
        }
        == set(config["forbidden_truth_substitutes"]),
        "acquisition_hash_bound": receipt["acquisition_sha256"]
        == producer.sha256_file(repo / config["outputs"]["acquisition"]),
        "ledger_exact_rebuild": ledger == rebuilt_ledger,
        "receipt_exact_rebuild": receipt == rebuilt_receipt,
        "ledger_file_hash_bound": receipt["ledger_file_sha256"]
        == producer.sha256_file(ledger_path),
        "whole_file_window": ledger["window"] == "entire_file",
        "all_nine_person_tracks_frozen": ledger["person_tracks"]
        == [f"Helmet_{value}" for value in range(2, 11)],
        "source_reference_track_frozen": ledger["reference_track"] == "Citi_1",
        "declared_and_observed_frames_equal": ledger["source_header"][
            "declared_frames"
        ]
        == ledger["source_header"]["observed_frames"]
        == 25912,
        "time_strictly_monotonic": ledger["time"]["strictly_monotonic"] is True,
        "denominator_conservation": ledger["denominators"]["conservation_met"] is True
        and ledger["denominators"]["person_frame_opportunities"]
        == ledger["denominators"]["valid_object_frames"]
        + ledger["denominators"]["missing_person_frames"]
        + ledger["denominators"]["missing_reference_opportunities"],
        "all_distance_bands_retained": set(expected_bands)
        == set(ledger["denominators"]["distance_bands_provisional_mm_conversion"]),
        "product_focus_bands_retained": set(
            ledger["denominators"]["product_core_provisional_gates"]
        )
        == {"5-10", "10-20"},
        "capability_boundary_retained": expected_bands[-1] == "40-plus"
        and "40-plus" in ledger["denominators"]["empty_bands"],
        "metric_counts_not_admitted_from_hypothesis": ledger[
            "conversion_authority"
        ]["metric_band_counts_admitted"]
        is False,
        "jrdb_circular_truth_rejected": receipt["source_decisions"][
            "jrdb_annotation_derived_person_geometry"
        ]
        == "REJECTED_CIRCULAR_TRUTH",
        "revel_not_admitted": receipt["source_decisions"]["revel_dynamic_vicon"][
            "decision"
        ]
        == "SOURCE_CANDIDATE_LIMITED_NOT_ADMITTED",
        "thor_not_admitted": receipt["source_decisions"]["thor_people_tracks_v1"]
        == "AUDITED_NOT_ADMITTED",
        "independent_chain_observed": gates["independent_measurement_chain"]["met"]
        is True,
        "candidate_blind_gate_met": gates["candidate_blind_freeze"]["met"] is True,
        "stable_id_gate_failed_closed": gates["stable_person_track_ids"]["met"]
        is False,
        "metric_3d_gate_failed_closed": gates["metric_3d_position"]["met"] is False,
        "time_sync_gate_failed_closed": gates["shared_time_binding"]["met"] is False,
        "coordinate_transform_gate_failed_closed": gates[
            "coordinate_frame_and_transform_semantics"
        ]["met"]
        is False,
        "error_calibration_gate_failed_closed": gates[
            "quantitative_error_or_calibration_statement"
        ]["met"]
        is False,
        "distance_denominator_gate_failed_closed": gates[
            "product_core_distance_denominators"
        ]["met"]
        is False,
        "all_admission_gates_not_met": receipt["all_admission_gates_met"] is False,
        "required_terminal": receipt["terminal_state"]
        == "INDEPENDENT_PERSON_TRAJECTORY_TRUTH_AUTHORITY_ABSENT",
        "terminal_registered": receipt["terminal_state"]
        in config["terminal_states"],
        "no_truth_authority_granted": receipt["authority_scope"][
            "independent_person_trajectory_truth_admitted"
        ]
        is False,
        "no_candidate_output_read": receipt["candidate_outputs_read"] is False
        and ledger["candidate_outputs_read"] is False,
        "no_algorithm_comparison": receipt["algorithm_comparison_performed"] is False
        and receipt["authority_scope"][
            "algorithm_comparison_or_selection_admitted"
        ]
        is False,
        "no_manual_trajectory_selection": receipt[
            "manual_trajectory_selection_performed"
        ]
        is False,
        "higher_authorities_closed": all(
            receipt["authority_scope"][field] is False
            for field in (
                "sensor_observation_to_mocap_spatial_extrinsic_admitted",
                "helmet_to_body_center_transform_admitted",
                "route_event_truth_admitted",
                "android_human_independent_walking_production_authority",
            )
        ),
    }
    status = "VALID" if all(checks.values()) else "INVALID"
    return {
        "schema": VALIDATION_SCHEMA,
        "stage": producer.STAGE,
        "status": status,
        "terminal_state": receipt["terminal_state"],
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "config_sha256": producer.sha256_file(config_path),
        "ledger_sha256": producer.sha256_file(ledger_path),
        "receipt_sha256": producer.sha256_file(receipt_path),
        "candidate_outputs_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = validate(repo, config_path)
    output = repo / config["outputs"]["validation"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": f"{result['checks_passed']}/{result['checks_total']}",
                "terminal_state": result["terminal_state"],
                "output": str(output),
            }
        )
    )
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
