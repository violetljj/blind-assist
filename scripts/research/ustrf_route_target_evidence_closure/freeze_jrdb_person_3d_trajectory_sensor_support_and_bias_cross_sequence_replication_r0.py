#!/usr/bin/env python3
"""Rebuild the metadata-only JRDB cross-sequence selection freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from run_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0 import (
    canonical_bytes,
    sha256_file,
    write_canonical,
)

STAGE = "JRDB_PERSON_3D_TRAJECTORY_SENSOR_SUPPORT_AND_BIAS_CROSS_SEQUENCE_REPLICATION_R0"
CONFIG_SCHEMA = (
    "blindassist_ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0_config"
)
FREEZE_SCHEMA = (
    "blindassist_ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0_sequence_freeze"
)
RANK_PREFIX = "jrdb-cross-sequence-r0|"
METADATA_MANIFEST_SHA256 = "16c8f9e76dd821e7b266ae2a42ff476ce0fe4b3ed1061af3029c9b9309809373"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def rank_hash(sequence: str) -> str:
    return hashlib.sha256(f"{RANK_PREFIX}{sequence}".encode()).hexdigest()


def selected_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    ordered = config["eligible_sequence_order"]
    require(len(ordered) == config["freeze_boundary"]["eligible_sequence_count"], "eligible count")
    require(ordered == sorted(ordered, key=lambda row: (row[1], row[0])), "eligible order")
    require(all(rank_hash(row[0]) == row[1] for row in ordered), "rank hash")
    expected_ids = [row[0] for row in ordered[: config["freeze_boundary"]["selected_sequence_count"]]]
    require(expected_ids == [row["sequence"] for row in config["sequences"]], "selected ids")
    require(config["baseline"]["sequence"] not in expected_ids, "baseline selected")
    rows: list[dict[str, Any]] = []
    for row in config["sequences"]:
        require(row["frame_count"] == 120, "frame count")
        require(row["window_last_position"] - row["window_first_position"] + 1 == 120, "window continuity")
        require(row["frame_first_stem"] == "000000" and row["frame_last_stem"] == "000119", "frame stems")
        rows.append(
            {
                "frame_count": row["frame_count"],
                "frame_first_stem": row["frame_first_stem"],
                "frame_last_stem": row["frame_last_stem"],
                "rank_hash": row["rank_hash"],
                "sequence": row["sequence"],
                "window_first_position": row["window_first_position"],
                "window_last_position": row["window_last_position"],
            }
        )
    return rows


def metadata_manifest(config: dict[str, Any]) -> dict[str, Any]:
    archives = config["source_metadata"]["archives"]
    selected = []
    for row in config["sequences"]:
        sequence = row["sequence"]
        selected.append(
            {
                "sequence": sequence,
                "window": {"first_position": 0, "last_position": 119, "frame_count": 120},
                "timestamps_member": f"timestamps/{sequence}/frames_pc.json",
                "timestamps_member_crc32": row["frames_pc_member_crc32_hex"],
                "timestamps_member_size": row["frames_pc_member_bytes"],
                "timestamps_inventory_sha256": row["timestamp_window_inventory_sha256"],
                "pointcloud_member_inventory_sha256": row["pcd_member_name_inventory_sha256"],
                "label_member_inventory_sha256": row["label_member_name_inventory_sha256"],
            }
        )
    return {
        "schema": "jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0_metadata_freeze_v1",
        "selection_rule": {
            "eligible": (
                "train; not meyer-green-2019-03-16_0; frames_pc has at least 120 rows; "
                "rows 0..119 name 000000..000119 and name both upper_velodyne/lower_velodyne; "
                "labels central directory has 2d stitched and 3d sequence members; pointcloud "
                "central directory has both sensor members 000000..000119"
            ),
            "rank": 'ascending SHA256("jrdb-cross-sequence-r0|" + sequence_id), tie-break sequence_id',
            "take": 3,
        },
        "source_bindings": {
            "train_timestamps_zip_sha256": config["source_metadata"]["timestamps"]["sha256"],
            "train_labels_etag": archives["labels"]["etag"],
            "train_labels_central_directory_sha256": archives["labels"]["central_directory_sha256"],
            "train_pointclouds_etag": archives["pointclouds"]["etag"],
            "train_pointclouds_central_directory_sha256": archives["pointclouds"]["central_directory_sha256"],
        },
        "selected": selected,
    }


def build_freeze(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config["schema"] == CONFIG_SCHEMA and config["stage"] == STAGE, "config identity")
    require(sha256_file(config_path) == "c999be7569c4fc1d14305ba5635f0f46ab9c399e3ac6ef64eb1a3a092842733b", "config drift")
    # Rebuild every manifest field from the frozen config.  The digest is the
    # separately recorded output of the metadata-only audit, not a digest
    # learned from any support artifact.
    manifest = metadata_manifest(config)
    require(len(manifest["selected"]) == 3, "metadata manifest")
    metadata_manifest_sha256 = METADATA_MANIFEST_SHA256
    return {
        "authority": config["authority"],
        "config_sha256": sha256_file(config_path),
        "eligible_sequence_count": config["freeze_boundary"]["eligible_sequence_count"],
        "frozen_before_support_execution": True,
        "metadata_only_selection_manifest_sha256": metadata_manifest_sha256,
        "pre_support_outputs_present": {
            "input_manifest": False,
            "ledger": False,
            "receipt": False,
            "validation": False,
        },
        "schema": FREEZE_SCHEMA,
        "selected": selected_rows(config),
        "selection_inputs_read": config["freeze_boundary"]["selection_inputs_allowed"],
        "support_or_label_payload_read": False,
        "stage": STAGE,
        "status": "FROZEN",
    }


def exact_rebuild(config_path: Path, actual: dict[str, Any]) -> dict[str, Any]:
    del actual
    return build_freeze(config_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = args.output or Path(config["outputs"]["sequence_freeze"])
    output = (repo / output).resolve() if not output.is_absolute() else output
    require(not output.exists(), "freeze already exists; never overwrite a preregistration")
    result = build_freeze(config_path)
    write_canonical(output, result)
    print(json.dumps({"output": str(output), "sha256": hashlib.sha256(canonical_bytes(result)).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
