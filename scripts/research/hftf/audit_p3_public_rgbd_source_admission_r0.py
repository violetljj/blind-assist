#!/usr/bin/env python3
"""Validate label-blind public RGB-D source-admission evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "blindassist_p3_public_rgbd_source_admission_r0_protocol"
CATALOG_SCHEMA = "blindassist_p3_public_rgbd_source_admission_r0_catalog"
RESULT_SCHEMA = "blindassist_p3_public_rgbd_source_admission_r0_result"
FORBIDDEN_KEYS = {
    "clearance", "clearance_m", "geometry_state", "transition_label",
    "transition_counts", "model_output", "model_outputs", "a2_output",
    "p3_output", "candidate_performance", "score", "scores",
}
SOURCE_FIELDS = {
    "dataset_id", "official_url", "access_mode", "license_reviewed",
    "parent_unit", "higher_cluster_unit", "published_parent_count",
    "rgb_available", "source_native_timestamps_available",
    "intrinsics_available", "independent_metric_sensor_types",
    "independent_metric_sensor_validity_available", "pose_available",
    "download_or_registration_state", "identity_inventory",
    "evidence",
}
IDENTITY_FIELDS = {
    "parent_id", "higher_cluster_id", "rgb_identity_count",
    "four_frame_continuity_confirmed", "raw_metric_sensor_assets_present",
    "rgb_files_sha256_complete", "metric_sensor_files_sha256_complete",
    "timestamp_files_sha256_complete", "ancestry_excluded",
}
EVIDENCE_FIELDS = {"official_url", "claim", "retrieved_on"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def reject_forbidden(value: Any) -> None:
    if isinstance(value, dict):
        collided = {str(key).lower() for key in value} & FORBIDDEN_KEYS
        require(not collided, f"forbidden label/model field: {sorted(collided)}")
        for nested in value.values():
            reject_forbidden(nested)
    elif isinstance(value, list):
        for nested in value:
            reject_forbidden(nested)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON object required")
    reject_forbidden(value)
    return value


def exact(value: dict[str, Any], fields: set[str], label: str) -> None:
    require(set(value) == fields, f"{label} field drift: {sorted(set(value) ^ fields)}")


def audit(protocol_path: Path, catalog_path: Path) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    catalog = load_json(catalog_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(catalog.get("schema") == CATALOG_SCHEMA, "catalog schema drift")
    exact(catalog, {"schema", "protocol_sha256", "sources", "runtime_state"}, "catalog")
    require(catalog["protocol_sha256"] == sha256_file(protocol_path), "catalog protocol SHA mismatch")
    runtime = catalog["runtime_state"]
    exact(runtime, {"model_outputs_read", "transition_labels_read", "candidate_performance_read"}, "runtime state")
    require(all(value is False for value in runtime.values()), "label/model boundary violated")
    frozen = protocol["source_universe"]
    require([row["dataset_id"] for row in catalog["sources"]] == list(frozen), "source universe/order drift")
    admissions = []
    eligible_parent_ids: set[tuple[str, str]] = set()
    eligible_cluster_ids: set[tuple[str, str]] = set()
    for source in catalog["sources"]:
        exact(source, SOURCE_FIELDS, f"source:{source.get('dataset_id')}")
        dataset_id = source["dataset_id"]
        expected = frozen[dataset_id]
        require(source["official_url"] == expected["official_url"], "official URL drift")
        require(source["parent_unit"] == expected["parent_unit"], "parent unit drift")
        require(source["higher_cluster_unit"] == expected["higher_cluster_unit"], "higher cluster unit drift")
        require(isinstance(source["evidence"], list) and source["evidence"], "official evidence missing")
        for evidence in source["evidence"]:
            exact(evidence, EVIDENCE_FIELDS, "evidence")
            require(evidence["official_url"].startswith(expected["official_origin"]), "non-official evidence origin")
        sensor_capable = bool(source["rgb_available"] and source["source_native_timestamps_available"] and source["independent_metric_sensor_types"])
        access_ready = source["download_or_registration_state"] in {"DIRECT_DOWNLOAD_AVAILABLE", "REGISTRATION_AVAILABLE"}
        status = "IDENTITY_AUDIT_ELIGIBLE" if sensor_capable and access_ready else "NOT_CURRENTLY_ADMISSIBLE"
        inventory_complete = True
        for identity in source["identity_inventory"]:
            exact(identity, IDENTITY_FIELDS, "identity")
            complete = bool(
                identity["four_frame_continuity_confirmed"]
                and identity["raw_metric_sensor_assets_present"]
                and identity["rgb_files_sha256_complete"]
                and identity["metric_sensor_files_sha256_complete"]
                and identity["timestamp_files_sha256_complete"]
                and not identity["ancestry_excluded"]
            )
            if complete:
                eligible_parent_ids.add((dataset_id, str(identity["parent_id"])))
                eligible_cluster_ids.add((dataset_id, str(identity["higher_cluster_id"])))
            else:
                inventory_complete = False
        admissions.append({
            "dataset_id": dataset_id,
            "status": status,
            "published_parent_count": source["published_parent_count"],
            "identity_inventory_count": len(source["identity_inventory"]),
            "identity_inventory_complete": bool(source["identity_inventory"]) and inventory_complete,
            "independent_metric_sensor_types": source["independent_metric_sensor_types"],
        })
    minimum = int(protocol["capacity_gate"]["minimum_holdout_parents"])
    target = int(protocol["capacity_gate"]["target_holdout_parents"])
    capacity_ready = len(eligible_parent_ids) >= minimum and len(eligible_cluster_ids) >= minimum
    return {
        "schema": RESULT_SCHEMA,
        "protocol_sha256": sha256_file(protocol_path),
        "catalog_sha256": sha256_file(catalog_path),
        "label_blind": True,
        "admissions": admissions,
        "fully_hashed_eligible_parent_count": len(eligible_parent_ids),
        "fully_hashed_eligible_higher_cluster_count": len(eligible_cluster_ids),
        "minimum_holdout_parents": minimum,
        "target_holdout_parents": target,
        "capacity_ready": capacity_ready,
        "terminal": (
            "P3_PUBLIC_RGBD_SOURCE_ADMISSION_R0_CAPACITY_READY_FOR_ROLE_FREEZE"
            if capacity_ready
            else "P3_PUBLIC_RGBD_SOURCE_ADMISSION_R0_PRELIMINARY_SOURCES_IDENTIFIED_IDENTITY_DOWNLOAD_NOT_COMPLETE"
        ),
    }


def write_new(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"overwrite forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.protocol.resolve(), args.catalog.resolve())
    write_new(args.output.resolve(), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
