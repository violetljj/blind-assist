#!/usr/bin/env python3
"""Independently rebuild and validate the THÖR R1 source-authority audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import audit_thor_source_native_id_time_transform_authority_recovery_r1 as producer

VALIDATION_SCHEMA = "blindassist_ustrf_thor_source_native_id_time_transform_authority_recovery_r1_validation"


def validate(repo: Path, config_path: Path) -> dict[str, Any]:
    config = producer.load_json(config_path)
    inventory_path = repo / config["outputs"]["inventory"]
    receipt_path = repo / config["outputs"]["receipt"]
    inventory = producer.load_json(inventory_path)
    receipt = producer.load_json(receipt_path)
    rebuilt_inventory, rebuilt_receipt = producer.audit(repo, config_path)
    rebuilt_receipt["inventory_sha256"] = producer.sha256_file(inventory_path)
    gates = receipt["authority_gates"]
    frozen = receipt["frozen_denominator_inherited_unchanged"]
    checks = {
        "config_identity": config["schema"] == producer.CONFIG_SCHEMA,
        "stage_identity": config["stage"] == producer.STAGE,
        "exact_two_terminal_states": config["terminal_states"]
        == [
            "INDEPENDENT_PERSON_TRAJECTORY_TRUTH_SOURCE_ADMITTED",
            "INDEPENDENT_PERSON_TRAJECTORY_TRUTH_AUTHORITY_ABSENT",
        ],
        "inventory_exact_rebuild": inventory == rebuilt_inventory,
        "receipt_exact_rebuild": receipt == rebuilt_receipt,
        "official_record_ids": inventory["official_records"]["people_tracks"][
            "record_id"
        ]
        == 3382145
        and inventory["official_records"]["point_clouds"]["record_id"] == 3405915,
        "no_raw_qtm_published": inventory["official_records"]["people_tracks"][
            "raw_qtm_files"
        ]
        == [],
        "no_recovery_mask_published": inventory["official_records"]["people_tracks"][
            "recovery_mask_or_provenance_files"
        ]
        == [],
        "frozen_run_has_no_qualisys_bag": inventory["frozen_member"][
            "qualisys_bag_published"
        ]
        is False,
        "frozen_run_has_no_point_cloud_bag": inventory["frozen_member"][
            "point_cloud_bag_published"
        ]
        is False,
        "reference_point_and_unit_defined": inventory["format_authority"][
            "helmet_reference_point_phrase_present"
        ]
        is True
        and inventory["format_authority"]["explicit_tsv_coordinate_unit_present"]
        is True,
        "velodyne_rigid_body_pose_present": inventory["frozen_payload_header"][
            "velodyne_rigid_body_present"
        ]
        is True,
        "identity_gate_failed": gates[
            "raw_trajectory_and_identity_recovery_authority"
        ]["met"]
        is False,
        "metric_reference_gate_closed": gates[
            "explicit_metric_unit_and_helmet_reference_point"
        ]["met"]
        is True,
        "clock_gate_failed": gates[
            "measured_qtm_velodyne_clock_offset_and_jitter"
        ]["met"]
        is False,
        "transform_gate_failed": gates[
            "complete_world_rigid_body_lidar_transform_and_error"
        ]["met"]
        is False,
        "whole_file_and_tracks_frozen": frozen["window"] == "entire_file"
        and frozen["person_tracks"] == [f"Helmet_{value}" for value in range(2, 11)],
        "reference_and_bands_frozen": frozen["reference_track"] == "Citi_1"
        and frozen["bands"] == ["0-5", "5-10", "10-20", "20-40", "40-plus"],
        "provisional_counts_unchanged": frozen["counts"]
        == {
            "0-5": 43821,
            "5-10": 41035,
            "10-20": 7286,
            "20-40": 0,
            "40-plus": 0,
        },
        "metric_counts_not_admitted": frozen["metric_band_counts_admitted"] is False,
        "required_terminal": receipt["terminal_state"]
        == "INDEPENDENT_PERSON_TRAJECTORY_TRUTH_AUTHORITY_ABSENT",
        "no_candidate_output_read": receipt["candidate_outputs_read"] is False
        and inventory["candidate_outputs_read"] is False,
        "no_forbidden_comparison": all(
            receipt[field] is False
            for field in (
                "centroid_comparison_performed",
                "tracker_comparison_performed",
                "deskew_comparison_performed",
            )
        ),
        "higher_authority_closed": receipt["authority_scope"][
            "algorithm_comparison_or_selection_admitted"
        ]
        is False
        and receipt["authority_scope"][
            "route_event_android_human_production_authority"
        ]
        is False,
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
        "inventory_sha256": producer.sha256_file(inventory_path),
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
    config = producer.load_json(config_path)
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
