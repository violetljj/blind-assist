#!/usr/bin/env python3
"""Audit THÖR source-native ID, unit, time and transform authority."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "blindassist_ustrf_thor_source_native_id_time_transform_authority_recovery_r1_config"
ACQUISITION_SCHEMA = "blindassist_ustrf_thor_source_native_id_time_transform_authority_recovery_r1_acquisition"
INVENTORY_SCHEMA = "blindassist_ustrf_thor_source_native_id_time_transform_authority_recovery_r1_inventory"
RECEIPT_SCHEMA = "blindassist_ustrf_thor_source_native_id_time_transform_authority_recovery_r1_receipt"
STAGE = "THOR_SOURCE_NATIVE_ID_TIME_TRANSFORM_AUTHORITY_RECOVERY_R1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_keys(record: dict[str, Any]) -> list[str]:
    return sorted(str(row["key"]) for row in record["files"])


def raw_qtm_files(keys: list[str]) -> list[str]:
    return [key for key in keys if Path(key).suffix.lower() in {".qtm", ".qtmproj"}]


def calibration_files(keys: list[str]) -> list[str]:
    tokens = ("calib", "extrinsic", "transform", "lever", "static_tf")
    return [key for key in keys if any(token in key.lower() for token in tokens)]


def recovery_files(keys: list[str]) -> list[str]:
    tokens = ("recovery", "repair", "id_switch", "mask", "provenance")
    return [key for key in keys if any(token in key.lower() for token in tokens)]


def clean_description(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(value))).strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(repo: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_json(config_path)
    if config.get("schema") != CONFIG_SCHEMA or config.get("stage") != STAGE:
        raise RuntimeError("config_identity")
    for parent in config["parents"]:
        actual = sha256_file(repo / parent["path"])
        if actual != parent["sha256"]:
            raise RuntimeError(f"parent_hash_drift:{parent['path']}:{actual}")
    acquisition_path = repo / config["outputs"]["acquisition"]
    acquisition = load_json(acquisition_path)
    if acquisition.get("schema") != ACQUISITION_SCHEMA:
        raise RuntimeError("acquisition_identity")
    snapshots = {row["source_id"]: row for row in acquisition["artifacts"]}
    for row in snapshots.values():
        if sha256_file(repo / row["path"]) != row["sha256"]:
            raise RuntimeError(f"snapshot_hash_drift:{row['source_id']}")
    people = load_json(repo / snapshots["thor_people_tracks_v1"]["path"])
    clouds = load_json(repo / snapshots["thor_point_clouds_v1"]["path"])
    people_keys = file_keys(people)
    cloud_keys = file_keys(clouds)
    all_keys = people_keys + cloud_keys
    selected = config["frozen_input"]["member"]
    selected_stem = "ex2_run2"
    selected_people_exports = [
        key for key in people_keys if key.lower().startswith("exp_2_run_2")
    ]
    selected_qualisys_bag = f"{selected_stem}_qualisys.bag"
    selected_cloud_bag = f"{selected_stem}.bag"
    description = clean_description(people["metadata"]["description"])
    qualisys_6dof = clean_description(
        (repo / snapshots["qualisys_6dof_tsv_format"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    qualisys_motion = clean_description(
        (repo / snapshots["qualisys_motion_tsv_format"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    qualisys_rigid_body = clean_description(
        (repo / snapshots["qualisys_rigid_body_definition"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    reference_phrase = "position of the centre of the mass of the markers defining the rigid body"
    explicit_tsv_unit = "The position in mm of the origin of the local coordinate system" in qualisys_6dof
    generic_origin_defined = (
        "origin of the local coordinate system of the rigid body is set to the geometric center"
        in qualisys_rigid_body
    )
    timestamp_warning_present = (
        "discouraged to use them for synchronization with data from other devices"
        in qualisys_motion
    )
    header_path = repo / config["parents"][4]["path"]
    header_lines = header_path.read_text(encoding="utf-8").splitlines()[:11]
    body_names = next(
        line.split("\t")[1:] for line in header_lines if line.startswith("BODY_NAMES\t")
    )
    r0_ledger = load_json(repo / config["parents"][1]["path"])
    r0_receipt = load_json(repo / config["parents"][2]["path"])

    inventory = {
        "schema": INVENTORY_SCHEMA,
        "stage": STAGE,
        "official_records": {
            "people_tracks": {
                "record_id": int(people["id"]),
                "version": people["metadata"].get("version"),
                "file_count": len(people_keys),
                "files": people_keys,
                "raw_qtm_files": raw_qtm_files(people_keys),
                "calibration_files": calibration_files(people_keys),
                "recovery_mask_or_provenance_files": recovery_files(people_keys),
            },
            "point_clouds": {
                "record_id": int(clouds["id"]),
                "version": clouds["metadata"].get("version"),
                "file_count": len(cloud_keys),
                "files": cloud_keys,
                "calibration_files": calibration_files(cloud_keys),
            },
        },
        "frozen_member": {
            "name": selected,
            "published_exports": selected_people_exports,
            "qualisys_bag_expected_name": selected_qualisys_bag,
            "qualisys_bag_published": selected_qualisys_bag in people_keys,
            "point_cloud_bag_expected_name": selected_cloud_bag,
            "point_cloud_bag_published": selected_cloud_bag in cloud_keys,
        },
        "format_authority": {
            "helmet_reference_point_phrase_present": reference_phrase in description,
            "helmet_reference_point": (
                "centre of mass of the markers defining the rigid body"
                if reference_phrase in description
                else None
            ),
            "explicit_tsv_coordinate_unit_present": explicit_tsv_unit,
            "qualisys_generic_rigid_body_origin_definition_present": generic_origin_defined,
            "unit_observation": "Qualisys' official 6DOF TSV format defines rigid-body X/Y/Z as the position in mm of the rigid body's local-coordinate-system origin.",
            "timestamp_cross_device_warning_present": timestamp_warning_present,
        },
        "frozen_payload_header": {
            "body_names": body_names,
            "velodyne_rigid_body_present": "Velodyne" in body_names,
            "citi_reference_present": "Citi_1" in body_names,
            "coordinate_unit_in_header": None,
        },
        "candidate_outputs_read": False,
    }
    gates = {
        "raw_trajectory_and_identity_recovery_authority": {
            "met": bool(
                inventory["official_records"]["people_tracks"]["raw_qtm_files"]
                and inventory["official_records"]["people_tracks"][
                    "recovery_mask_or_provenance_files"
                ]
            ),
            "evidence": "No raw .qtm/.qtmproj and no frame-level ID repair/recovery mask or provenance file is published in the official people-tracks record.",
        },
        "explicit_metric_unit_and_helmet_reference_point": {
            "met": bool(
                inventory["format_authority"]["explicit_tsv_coordinate_unit_present"]
                and inventory["format_authority"]["helmet_reference_point_phrase_present"]
            ),
            "reference_point_defined": inventory["format_authority"][
                "helmet_reference_point_phrase_present"
            ],
            "unit_defined": inventory["format_authority"][
                "explicit_tsv_coordinate_unit_present"
            ],
            "evidence": "The THOR record defines the rigid-body translation as the marker-set centre of mass, and Qualisys' official 6DOF TSV format defines rigid-body X/Y/Z in mm.",
        },
        "measured_qtm_velodyne_clock_offset_and_jitter": {
            "met": False,
            "selected_run_paired_bags_present": bool(
                inventory["frozen_member"]["qualisys_bag_published"]
                and inventory["frozen_member"]["point_cloud_bag_published"]
            ),
            "ntp_statement_is_measurement": False,
            "evidence": "The paper states common-NTP configuration, but the frozen run 2 has neither published Qualisys nor point-cloud bag and no source-native measured offset/jitter artifact.",
        },
        "complete_world_rigid_body_lidar_transform_and_error": {
            "met": False,
            "world_to_velodyne_rigid_body_pose_present": inventory[
                "frozen_payload_header"
            ]["velodyne_rigid_body_present"],
            "rigid_body_to_lidar_measurement_frame_extrinsic_present": False,
            "axes_handedness_lever_arm_present": False,
            "quantified_extrinsic_error_present": False,
            "evidence": "The frozen QTM export contains a Velodyne rigid-body pose, but the official records publish no calibration/extrinsic file binding that marker rigid body to the LiDAR measurement frame.",
        },
    }
    all_met = all(row["met"] for row in gates.values())
    terminal = (
        "INDEPENDENT_PERSON_TRAJECTORY_TRUTH_SOURCE_ADMITTED"
        if all_met
        else "INDEPENDENT_PERSON_TRAJECTORY_TRUTH_AUTHORITY_ABSENT"
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "stage": STAGE,
        "status": terminal,
        "terminal_state": terminal,
        "authority_gates": gates,
        "all_required_authority_closed": all_met,
        "frozen_denominator_inherited_unchanged": {
            "frames": r0_ledger["source_header"]["observed_frames"],
            "person_tracks": r0_ledger["person_tracks"],
            "window": r0_ledger["window"],
            "reference_track": r0_ledger["reference_track"],
            "bands": config["frozen_input"]["distance_bands_m"],
            "counts": r0_ledger["denominators"][
                "distance_bands_provisional_mm_conversion"
            ],
            "missing_policy_unchanged": True,
            "metric_band_counts_admitted": False,
        },
        "parent_terminal_preserved": r0_receipt["terminal_state"],
        "authority_scope": {
            "independent_person_trajectory_truth_admitted": all_met,
            "algorithm_comparison_or_selection_admitted": False,
            "route_event_android_human_production_authority": False,
        },
        "candidate_outputs_read": False,
        "centroid_comparison_performed": False,
        "tracker_comparison_performed": False,
        "deskew_comparison_performed": False,
        "acquisition_sha256": sha256_file(acquisition_path),
        "config_sha256": sha256_file(config_path),
    }
    return inventory, receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config.resolve()
    config = load_json(config_path)
    inventory, receipt = audit(repo, config_path)
    inventory_path = repo / config["outputs"]["inventory"]
    receipt_path = repo / config["outputs"]["receipt"]
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt["inventory_sha256"] = sha256_file(inventory_path)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"terminal_state": receipt["terminal_state"], "output": str(receipt_path)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
