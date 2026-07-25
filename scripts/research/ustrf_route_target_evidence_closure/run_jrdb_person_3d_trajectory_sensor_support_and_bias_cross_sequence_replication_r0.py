#!/usr/bin/env python3
"""Freeze inputs, execute the frozen support kernel, and aggregate replication evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import run_jrdb_person_3d_trajectory_sensor_support_and_bias_canary_r0 as kernel
import run_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0 as packet_kernel
import run_jrdb_single_sequence_native_multisensor_person_geometry_canary_r1 as eligibility_kernel
from freeze_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0 import (
    CONFIG_SCHEMA,
    FREEZE_SCHEMA,
    STAGE,
    exact_rebuild,
    require,
)
from run_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0 import (
    canonical_bytes,
    sha256_file,
    write_canonical,
)

MANIFEST_SCHEMA = (
    "blindassist_ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0_input_manifest"
)
LEDGER_SCHEMA = (
    "blindassist_ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0_ledger"
)
RECEIPT_SCHEMA = (
    "blindassist_ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0_receipt"
)
INPUT_FILES = ("observation-packet.json", "eligibility-ledger.json", "receipt.json", "validation.json")


def load_config(repo: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config["schema"] == CONFIG_SCHEMA and config["stage"] == STAGE, "config identity")
    for binding in config["frozen_kernel"].values():
        if isinstance(binding, dict) and "path" in binding and "sha256" in binding:
            require(sha256_file(repo / binding["path"]) == binding["sha256"], f"kernel drift: {binding['path']}")
    return config


def load_freeze(repo: Path, config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    path = repo / config["outputs"]["sequence_freeze"]
    freeze = json.loads(path.read_text(encoding="utf-8"))
    require(freeze["schema"] == FREEZE_SCHEMA and freeze["stage"] == STAGE, "freeze identity")
    require(exact_rebuild(config_path, freeze) == freeze, "freeze exact rebuild")
    require(freeze["support_or_label_payload_read"] is False, "selection contamination")
    require(freeze["frozen_before_support_execution"] is True, "freeze timing")
    return freeze


def input_paths(repo: Path, config: dict[str, Any], sequence: str) -> dict[str, Path]:
    root = repo / config["outputs"]["dataset_root"] / sequence
    return {
        "observation_packet": root / INPUT_FILES[0],
        "eligibility_ledger": root / INPUT_FILES[1],
        "receipt": root / INPUT_FILES[2],
        "validation": root / INPUT_FILES[3],
    }


def baseline_json(repo: Path, name: str) -> dict[str, Any]:
    return json.loads((repo / "configs" / name).read_text(encoding="utf-8"))


def materialization_config(
    repo: Path, config: dict[str, Any], sequence: str, bag_path: Path
) -> dict[str, Any]:
    child = baseline_json(repo, "ustrf_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0.json")
    root = Path(config["outputs"]["dataset_root"]) / sequence
    child["canary"].update(
        {
            "sequence": sequence,
            "window_first_frame": 0,
            "window_last_frame": 119,
            "frame_count": 120,
        }
    )
    child["local_inputs"]["bag"] = {
        "path": bag_path.relative_to(repo).as_posix(),
        "bytes": bag_path.stat().st_size,
        "sha256": sha256_file(bag_path),
    }
    for role, source in config["source_metadata"]["archives"].items():
        if role in child["remote_archives"]:
            child["remote_archives"][role] = {
                key: source[key]
                for key in ("url", "content_length", "etag", "central_directory_offset", "central_directory_size")
            }
    child["outputs"] = {
        "payload_root": (root / "payload").as_posix(),
        "materialization": (root / "materialization.json").as_posix(),
        "observation_packet": (root / "observation-packet.json").as_posix(),
        "receipt": (root / "packet-receipt.json").as_posix(),
        "validation": (root / "packet-validation.json").as_posix(),
    }
    child["resource_gate"]["second_sequence_authorized"] = True
    child["gates"]["maximum_image_pointcloud_delta_seconds"] = 1.0
    child["replication_nonconsumed_input_gate"] = {
        "field": "maximum_image_pointcloud_delta_seconds",
        "baseline_value": 0.05,
        "replication_value": 1.0,
        "reason": "RGB-to-PCD simultaneity is not consumed by the frozen PCD/annotation-box object or pair support-bias kernel; frame-stem identity remains required",
    }
    return child


def eligibility_config(
    repo: Path,
    config: dict[str, Any],
    sequence: str,
    packet_path: Path,
    packet_receipt_path: Path,
) -> dict[str, Any]:
    child = baseline_json(repo, "ustrf_jrdb_single_sequence_native_multisensor_person_geometry_canary_r1.json")
    packet_receipt = json.loads(packet_receipt_path.read_text(encoding="utf-8"))
    child["canary"].update(
        {
            "sequence": sequence,
            "window_first_frame": 0,
            "window_last_frame": 119,
            "frame_count": 120,
            "data_role": "metadata_frozen_cross_sequence_replication_diagnostic_only",
        }
    )
    child["parent_r0"]["observation_packet"] = {
        "path": packet_path.relative_to(repo).as_posix(),
        "sha256": sha256_file(packet_path),
        "required_status": "IMMUTABLE_OBSERVATION_PACKET",
    }
    child["parent_r0"]["receipt"] = {
        "path": packet_receipt_path.relative_to(repo).as_posix(),
        "sha256": sha256_file(packet_receipt_path),
        "required_terminal": packet_receipt["terminal_state"],
    }
    child["parent_r0"]["validation"] = {
        "path": (packet_path.parent / "packet-validation.json").relative_to(repo).as_posix(),
        "sha256": sha256_file(packet_path.parent / "packet-validation.json"),
        "required_status": "VALID",
    }
    root = Path(config["outputs"]["dataset_root"]) / sequence
    child["outputs"] = {
        "eligibility_ledger": (root / "eligibility-ledger.json").as_posix(),
        "receipt": (root / "receipt.json").as_posix(),
        "validation": (root / "validation.json").as_posix(),
    }
    return child


def frozen_dataset_wide_static_edges(
    repo: Path, config: dict[str, Any]
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    baseline_config_path = repo / config["frozen_kernel"]["baseline_config"]["path"]
    baseline_config = json.loads(baseline_config_path.read_text(encoding="utf-8"))
    packet_binding = baseline_config["parent"]["observation_packet"]
    packet_path = repo / packet_binding["path"]
    require(
        packet_path.is_file() and sha256_file(packet_path) == packet_binding["sha256"],
        "frozen baseline packet drift",
    )
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    rows = packet["calibration"]["required_edges"]
    edges = {
        (row["parent"], row["child"]): {
            "parent": row["parent"],
            "child": row["child"],
            "translation": row["translation"],
            "quaternion_xyzw": row["quaternion_xyzw"],
        }
        for row in rows
    }
    return edges, {
        "mode": "FROZEN_DATASET_WIDE_STATIC_CALIBRATION_FALLBACK",
        "baseline_config_path": baseline_config_path.relative_to(repo).as_posix(),
        "baseline_config_sha256": sha256_file(baseline_config_path),
        "baseline_packet_path": packet_path.relative_to(repo).as_posix(),
        "baseline_packet_sha256": sha256_file(packet_path),
        "required_edge_count": len(edges),
    }


def build_packet_with_frozen_static_fallback(
    repo: Path,
    config: dict[str, Any],
    child: dict[str, Any],
    config_path: Path,
    materialization: dict[str, Any],
    bag_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bag_roles = packet_kernel.read_bag_roles(bag_path, child)
    source_imu_samples = len(bag_roles["imu"])
    invalid_imu_orientation_samples = 0
    for sample in bag_roles["imu"]:
        quaternion = sample["orientation_xyzw"]
        magnitude = sum(float(value) * float(value) for value in quaternion)
        if not (magnitude > 0):
            invalid_imu_orientation_samples += 1
            sample["orientation_xyzw"] = [0.0, 0.0, 0.0, 1.0]
    timestamp_path = repo / child["local_inputs"]["timestamps"]["path"]
    sequence = child["canary"]["sequence"]
    with zipfile.ZipFile(timestamp_path) as bundle:
        point_doc = json.loads(
            bundle.read(f"timestamps/{sequence}/frames_pc.json")
        )
    point_rows = {
        packet_kernel.frame_stem(
            next(
                point
                for point in row["pointclouds"]
                if point["name"] == "upper_velodyne"
            )["url"]
        ): row
        for row in point_doc["data"]
    }
    target_times = [
        round(
            float(
                next(
                    point
                    for point in point_rows[f"{index:06d}"]["pointclouds"]
                    if point["name"] == "upper_velodyne"
                )["timestamp"]
            )
            * 1e9
        )
        for index in range(120)
    ]
    imu_times = [sample["timestamp_ns"] for sample in bag_roles["imu"]]
    unbracketed_targets = [
        target
        for target in target_times
        if target <= imu_times[0] or target >= imu_times[-1]
    ]
    if unbracketed_targets:
        placeholders = []
        for target in target_times:
            for timestamp_ns in (target - 1, target + 1):
                placeholders.append(
                    {
                        "timestamp_ns": timestamp_ns,
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "angular_velocity": [0.0, 0.0, 0.0],
                        "linear_acceleration": [0.0, 0.0, 0.0],
                    }
                )
        bag_roles["imu"] = placeholders
    required = {
        tuple(edge) for edge in child["topics"]["static_transform"]["required_edges"]
    }
    missing = sorted(required - set(bag_roles["static"]))
    if missing:
        frozen_edges, provenance = frozen_dataset_wide_static_edges(repo, config)
        require(required.issubset(frozen_edges), f"frozen static edges missing: {missing}")
        bag_roles["static"] = frozen_edges
        provenance["bag_tf_static_missing_edges"] = [list(edge) for edge in missing]
    else:
        provenance = {
            "mode": "SAME_BAG_TF_STATIC",
            "bag_tf_static_missing_edges": [],
            "required_edge_count": len(required),
        }
    original = packet_kernel.read_bag_roles
    packet_kernel.read_bag_roles = lambda _bag_path, _config: bag_roles
    try:
        packet = packet_kernel.build_packet(repo, config_path, materialization)
    finally:
        packet_kernel.read_bag_roles = original
    if invalid_imu_orientation_samples or unbracketed_targets:
        for frame in packet["frames"]:
            frame["imu"]["orientation_xyzw"] = None
            frame["imu"]["angular_velocity"] = None
            frame["imu"]["linear_acceleration"] = None
    packet["replication_static_transform_provenance"] = provenance
    packet["replication_imu_provenance"] = {
        "role_in_replication": "NOT_CONSUMED_BY_OBJECT_OR_PAIR_SUPPORT_BIAS_KERNEL",
        "source_samples": source_imu_samples,
        "invalid_zero_orientation_samples": invalid_imu_orientation_samples,
        "unbracketed_external_frame_count": len(unbracketed_targets),
        "output_orientation": (
            "NULL_NOT_EVALUABLE"
            if invalid_imu_orientation_samples or unbracketed_targets
            else "SOURCE_INTERPOLATED"
        ),
        "angular_velocity_and_linear_acceleration_retained": not bool(
            unbracketed_targets
        ),
    }
    packet["replication_nonconsumed_input_gate"] = child[
        "replication_nonconsumed_input_gate"
    ]
    return packet, provenance


def materialize_one(repo: Path, config: dict[str, Any], sequence: str) -> dict[str, Any]:
    root = repo / config["outputs"]["dataset_root"] / sequence
    root.mkdir(parents=True, exist_ok=True)
    bag_path = repo / config["outputs"]["dataset_root"] / "bags" / f"{sequence}.bag"
    require(bag_path.is_file(), f"missing frozen bag: {bag_path}")
    receipt_slug = {
        "gates-basement-elevators-2019-01-17_1": "gates",
        "stlc-111-2019-04-19_0": "stlc",
        "clark-center-2019-02-28_0": "clark",
    }[sequence]
    acquisition_path = (
        repo
        / Path(config["outputs"]["sequence_freeze"]).parent
        / f"acquisition-{receipt_slug}.json"
    )
    require(acquisition_path.is_file(), f"missing acquisition receipt: {sequence}")
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    frozen_sequence = next(row for row in config["sequences"] if row["sequence"] == sequence)
    require(acquisition["status"] == "ACQUIRED", f"acquisition status: {sequence}")
    require(
        acquisition["bag"]["sha256"] == sha256_file(bag_path)
        and acquisition["bag"]["bytes"] == bag_path.stat().st_size,
        f"acquired bag drift: {sequence}",
    )
    require(
        acquisition["member"]["name"] == frozen_sequence["rosbag_member"]["name"]
        and acquisition["member"]["crc32"] == int(frozen_sequence["rosbag_member"]["crc32_hex"], 16)
        and acquisition["member"]["uncompressed"] == frozen_sequence["rosbag_member"]["uncompressed_size"],
        f"acquisition member drift: {sequence}",
    )
    config_path = root / "materialization-config.json"
    packet_path = root / "observation-packet.json"
    packet_receipt_path = root / "packet-receipt.json"
    packet_validation_path = root / "packet-validation.json"
    completed_inputs = (
        packet_path,
        root / "eligibility-ledger.json",
        root / "receipt.json",
        root / "validation.json",
    )
    if all(path.is_file() for path in completed_inputs):
        return {"sequence": sequence, "status": "REUSED_MATERIALIZED_AND_ELIGIBILITY_VALID"}
    require(not any(path.exists() for path in (config_path, packet_path, packet_receipt_path, packet_validation_path)), f"materialization outputs exist: {sequence}")
    child = materialization_config(repo, config, sequence, bag_path)
    write_canonical(config_path, child)
    files, transport = packet_kernel.materialize_payload(repo, child)
    materialization = {
        "schema": packet_kernel.MATERIALIZATION_SCHEMA,
        "stage": packet_kernel.STAGE,
        "status": "MATERIALIZED",
        "terminal_state": None,
        "config_sha256": sha256_file(config_path),
        "transport": transport,
        "files": files,
        "authority": config["authority"],
    }
    packet, static_provenance = build_packet_with_frozen_static_fallback(
        repo, config, child, config_path, materialization, bag_path
    )
    write_canonical(packet_path, packet)
    materialization["observation_packet"] = {
        "path": packet_path.relative_to(repo).as_posix(),
        "bytes": packet_path.stat().st_size,
        "sha256": sha256_file(packet_path),
    }
    write_canonical(root / "materialization.json", materialization)
    packet_receipt = packet_kernel.audit_packet(child, packet, sha256_file(packet_path))
    packet_receipt.update(
        {
            "replication_local_parent_authority": True,
            "source_bag_sha256": sha256_file(bag_path),
            "acquisition_receipt_sha256": sha256_file(acquisition_path),
            "materialization_sha256": sha256_file(root / "materialization.json"),
            "replication_static_transform_provenance": static_provenance,
            "replication_imu_provenance": packet["replication_imu_provenance"],
            "replication_nonconsumed_input_gate": packet[
                "replication_nonconsumed_input_gate"
            ],
        }
    )
    write_canonical(packet_receipt_path, packet_receipt)
    packet_valid = (
        packet["status"] == "IMMUTABLE_OBSERVATION_PACKET"
        and packet["sequence"] == sequence
        and len(packet["frames"]) == 120
        and all(
            (repo / row["path"]).is_file() and sha256_file(repo / row["path"]) == row["sha256"]
            for row in packet["raw_payload"]["files"]
        )
    )
    write_canonical(
        packet_validation_path,
        {
            "schema": "blindassist_ustrf_jrdb_cross_sequence_replication_local_packet_validation",
            "stage": STAGE,
            "status": "VALID" if packet_valid else "INVALID",
            "sequence": sequence,
            "packet_sha256": sha256_file(packet_path),
            "packet_receipt_sha256": sha256_file(packet_receipt_path),
            "checks": {
                "immutable_packet": packet["status"] == "IMMUTABLE_OBSERVATION_PACKET",
                "sequence_bound": packet["sequence"] == sequence,
                "all_120_frames": len(packet["frames"]) == 120,
                "raw_payload_hashes": packet_valid,
                "pose_imu_clock_static_gates_reached": "failure" not in packet_receipt,
                "static_transform_provenance_bound": (
                    packet["replication_static_transform_provenance"]
                    == static_provenance
                ),
                "imu_role_disclosed_and_not_consumed": (
                    packet["replication_imu_provenance"]["role_in_replication"]
                    == "NOT_CONSUMED_BY_OBJECT_OR_PAIR_SUPPORT_BIAS_KERNEL"
                ),
                "rgb_pointcloud_simultaneity_not_consumed_disclosed": (
                    packet["replication_nonconsumed_input_gate"]["field"]
                    == "maximum_image_pointcloud_delta_seconds"
                ),
            },
            "authority": config["authority"],
        },
    )
    require(packet_valid, f"packet validation: {sequence}")
    econfig_path = root / "eligibility-config.json"
    econfig = eligibility_config(repo, config, sequence, packet_path, packet_receipt_path)
    write_canonical(econfig_path, econfig)
    labels_2d, labels_3d = kernel.label_documents(repo, packet)
    eligibility = eligibility_kernel.build_ledger(econfig, packet, labels_2d, labels_3d)
    eligibility_path = root / "eligibility-ledger.json"
    write_canonical(eligibility_path, eligibility)
    ereceipt = eligibility_kernel.build_receipt(
        econfig, eligibility, hashlib.sha256(canonical_bytes(eligibility)).hexdigest()
    )
    ereceipt["config_sha256"] = sha256_file(econfig_path)
    write_canonical(root / "receipt.json", ereceipt)
    conserved = all(
        row["expected"] == row["eligible"] + row["abstained"] + row["invalid"]
        for row in eligibility["denominators"].values()
    )
    write_canonical(
        root / "validation.json",
        {
            "schema": "blindassist_ustrf_jrdb_cross_sequence_replication_local_eligibility_validation",
            "stage": STAGE,
            "status": "VALID" if conserved else "INVALID",
            "sequence": sequence,
            "config_sha256": sha256_file(econfig_path),
            "eligibility_ledger_sha256": sha256_file(eligibility_path),
            "receipt_sha256": sha256_file(root / "receipt.json"),
            "checks": {
                "sequence_bound": eligibility["sequence"] == sequence,
                "denominator_conservation": conserved,
                "diagnostic_ceiling": ereceipt["authority_ceiling"] == "DIAGNOSTIC",
                "packet_hash_bound": eligibility["parent_packet_sha256"] == sha256_file(packet_path),
            },
            "authority": config["authority"],
        },
    )
    require(conserved, f"eligibility validation: {sequence}")
    return {"sequence": sequence, "status": "MATERIALIZED_AND_ELIGIBILITY_VALID"}


def materialize_inputs(repo: Path, config: dict[str, Any], config_path: Path) -> list[dict[str, Any]]:
    freeze = load_freeze(repo, config, config_path)
    require(not (repo / config["outputs"]["input_manifest"]).exists(), "input manifest already frozen")
    return [materialize_one(repo, config, row["sequence"]) for row in freeze["selected"]]


def build_input_manifest(repo: Path, config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    freeze = load_freeze(repo, config, config_path)
    bindings = []
    for selected in freeze["selected"]:
        sequence = selected["sequence"]
        paths = input_paths(repo, config, sequence)
        require(all(path.is_file() for path in paths.values()), f"incomplete inputs: {sequence}")
        packet = json.loads(paths["observation_packet"].read_text(encoding="utf-8"))
        eligibility = json.loads(paths["eligibility_ledger"].read_text(encoding="utf-8"))
        receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
        validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
        require(packet["status"] == "IMMUTABLE_OBSERVATION_PACKET", f"packet status: {sequence}")
        require(packet["sequence"] == sequence and len(packet["frames"]) == 120, f"packet window: {sequence}")
        require([row["frame_stem"] for row in packet["frames"]] == [f"{i:06d}" for i in range(120)], f"packet stems: {sequence}")
        require(eligibility["sequence"] == sequence, f"eligibility sequence: {sequence}")
        require(receipt["terminal_state"] in (
            "ANNOTATION_DERIVED_PERSON_GEOMETRY_AVAILABLE_WITH_ABSTENTION",
            "ANNOTATION_DERIVED_PERSON_GEOMETRY_AVAILABLE_COMPLETE",
        ), f"eligibility terminal: {sequence}")
        require(validation["status"] == "VALID", f"eligibility validation: {sequence}")
        bindings.append(
            {
                "sequence": sequence,
                "window": selected,
                "files": {
                    role: {
                        "path": path.relative_to(repo).as_posix(),
                        "sha256": sha256_file(path),
                    }
                    for role, path in paths.items()
                },
                "parent_denominators": eligibility["denominators"],
            }
        )
    return {
        "schema": MANIFEST_SCHEMA,
        "stage": STAGE,
        "status": "INPUTS_FROZEN_BEFORE_SUPPORT",
        "config_sha256": sha256_file(config_path),
        "sequence_freeze_sha256": sha256_file(repo / config["outputs"]["sequence_freeze"]),
        "sequence_count": len(bindings),
        "bindings": bindings,
        "kernel_hashes": {
            key: value["sha256"]
            for key, value in config["frozen_kernel"].items()
            if isinstance(value, dict) and "path" in value and "sha256" in value
        },
        "authority": config["authority"],
    }


def write_input_manifest(repo: Path, config: dict[str, Any], config_path: Path) -> Path:
    output = repo / config["outputs"]["input_manifest"]
    require(not output.exists(), "input manifest already frozen; never overwrite")
    require(not any((repo / config["outputs"][name]).exists() for name in ("ledger", "receipt", "validation")), "support output predates input freeze")
    manifest = build_input_manifest(repo, config, config_path)
    write_canonical(output, manifest)
    return output


def child_config(binding: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    files = binding["files"]
    return {
        "schema": kernel.CONFIG_SCHEMA,
        "stage": kernel.STAGE,
        "parent": {
            "result": files["receipt"],
            "eligibility_ledger": files["eligibility_ledger"],
            "receipt": {**files["receipt"], "required_terminal": json.loads(Path(files["receipt"]["_absolute"]).read_text())["terminal_state"]} if "_absolute" in files["receipt"] else files["receipt"],
            "validation": {**files["validation"], "required_status": "VALID"},
            "observation_packet": {**files["observation_packet"], "required_status": "IMMUTABLE_OBSERVATION_PACKET"},
        },
        "canary": {"sequence": binding["sequence"], "frame_count": 120},
        "support_contract": {
            "coordinate_frame": "logical_rgb360",
            "box_dimensions": "length_along_local_x_width_along_local_y_height_along_z",
            "box_query_is_annotation_conditioned": True,
            "minimum_fused_in_box_points": 3,
        },
        "motion_contract": {
            "maximum_pair_gap_seconds": 0.2,
            "jump_displacement_flag_meters": 0.5,
            "speed_flag_mps": 4.5,
            "acceleration_flag_mps2": 12.0,
        },
        "authority": config["authority"],
        "terminal_states": config["terminal_states"],
        "outputs": {},
    }


def run_sequence(repo: Path, binding: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    child = child_config(binding, config)
    receipt_path = repo / binding["files"]["receipt"]["path"]
    child["parent"]["receipt"]["required_terminal"] = json.loads(receipt_path.read_text(encoding="utf-8"))["terminal_state"]
    with tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False) as handle:
        temp = Path(handle.name)
        handle.write(canonical_bytes(child))
    try:
        ledger, receipt = kernel.run(repo, temp)
    finally:
        temp.unlink(missing_ok=True)
    return {"sequence": binding["sequence"], "ledger": ledger, "kernel_receipt": receipt}


def support_fraction(rows: list[dict[str, Any]]) -> float | None:
    return sum(row["classification"] == "sensor-supported" for row in rows) / len(rows) if rows else None


def delta(value: float | None, reference: float) -> float | None:
    return value - reference if value is not None else None


def supported_residual(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return kernel.quantiles(
        row["centroid_residual_3d_m"]
        for row in rows
        if row["classification"] == "sensor-supported"
    )


def contrast(rows: list[dict[str, Any]], dimension: str) -> dict[str, Any]:
    if dimension == "far":
        reference = [row for row in rows if row.get("range_band") in ("0-10", "10-20")]
        target = [row for row in rows if row.get("range_band") == "40-plus"]
        labels = ("0-20", "40-plus")
    else:
        reference = [row for row in rows if row.get("cross_modal_presence") == "3d-and-2d"]
        target = [row for row in rows if row.get("cross_modal_presence") == "3d-only"]
        labels = ("3d-and-2d", "3d-only")
    if not reference or not target:
        return {"status": "NOT_EVALUABLE", "reference": labels[0], "target": labels[1]}
    ref_q, target_q = supported_residual(reference), supported_residual(target)
    ref_support, target_support = support_fraction(reference), support_fraction(target)
    residual_adverse = (
        ref_q["p50"] is not None and target_q["p50"] is not None and target_q["p50"] > ref_q["p50"]
    )
    support_adverse = target_support < ref_support
    return {
        "status": "EVALUABLE",
        "reference": labels[0],
        "target": labels[1],
        "reference_denominator": len(reference),
        "target_denominator": len(target),
        "support_fraction_delta_target_minus_reference": target_support - ref_support,
        "residual_p50_delta_target_minus_reference_m": (
            target_q["p50"] - ref_q["p50"] if ref_q["p50"] is not None and target_q["p50"] is not None else None
        ),
        "adverse_direction": support_adverse or residual_adverse,
    }


def direction_status(values: list[dict[str, Any]]) -> str:
    evaluable = [row for row in values if row["status"] == "EVALUABLE"]
    if len(evaluable) < 2:
        return "NOT_EVALUABLE"
    return "DIRECTION_REPLICATED" if all(row["adverse_direction"] for row in evaluable) else "MIXED_OR_CONTRADICTED"


def build_aggregate(repo: Path, config: dict[str, Any], config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = repo / config["outputs"]["input_manifest"]
    require(manifest_path.is_file(), "freeze inputs first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(canonical_bytes(build_input_manifest(repo, config, config_path)) == manifest_path.read_bytes(), "input manifest exact rebuild")
    results = [run_sequence(repo, binding, config) for binding in manifest["bindings"]]
    object_rows = [row for result in results for row in result["ledger"]["object_frames"]]
    three_d_rows = [row for row in object_rows if row["cross_modal_presence"] != "2d-only"]
    pair_rows = [row for result in results for row in result["ledger"]["motion_pairs"]]
    acceleration_rows = [row for result in results for row in result["ledger"]["acceleration_triples"]]
    per_sequence = []
    baseline = config["baseline"]["reference_metrics"]
    far_values, three_d_only_values = [], []
    for result in results:
        ledger = result["ledger"]
        rows_3d = [row for row in ledger["object_frames"] if row["cross_modal_presence"] != "2d-only"]
        summary = ledger["summary"]
        far = contrast(rows_3d, "far")
        three_d_only = contrast(rows_3d, "three_d_only")
        far_values.append(far)
        three_d_only_values.append(three_d_only)
        object_support = summary["computable_3d_object_sensor_supported_fraction"]
        pair_support = summary["computable_motion_pair_sensor_supported_fraction"]
        residual_p50 = summary["centroid_residual_3d_m"]["p50"]
        residual_p95 = summary["centroid_residual_3d_m"]["p95"]
        per_sequence.append(
            {
                "sequence": result["sequence"],
                "ledger": ledger,
                "summary": summary,
                "baseline_delta": {
                    "object_frame_support_fraction": delta(object_support, baseline["object_frame_support_fraction"]),
                    "pair_support_fraction": delta(pair_support, baseline["pair_support_fraction"]),
                    "centroid_residual_3d_median_m": delta(residual_p50, baseline["centroid_residual_3d_median_m"]),
                    "centroid_residual_3d_p95_m": delta(residual_p95, baseline["centroid_residual_3d_p95_m"]),
                },
                "far_range_contrast": far,
                "three_d_only_contrast": three_d_only,
            }
        )
    pooled_summary = {
        "computable_3d_object_sensor_supported_fraction": support_fraction(three_d_rows),
        "computable_motion_pair_sensor_supported_fraction": support_fraction(pair_rows),
        "centroid_residual_3d_m": supported_residual(three_d_rows),
        "sensor_motion_minus_annotation_3d_m": kernel.quantiles(
            row["sensor_motion_minus_annotation_3d_m"] for row in pair_rows if row["classification"] == "sensor-supported"
        ),
        "sensor_pattern_counts": dict(sorted(Counter(row["sensor_pattern"] for row in three_d_rows).items())),
    }
    def worst_item(metric: str, nested: str | None, minimum: bool) -> dict[str, Any]:
        candidates = []
        for row in per_sequence:
            value = row["summary"][metric] if nested is None else row["summary"][metric][nested]
            if value is not None:
                denominator_key = "computable_3d_object_frames" if metric != "computable_motion_pair_sensor_supported_fraction" else "motion_pairs"
                candidates.append((value, row["sequence"], row["ledger"]["denominators"][denominator_key]["expected"]))
        if not candidates:
            return {"status": "NOT_EVALUABLE"}
        value, sequence, denominator_value = (min(candidates) if minimum else max(candidates))
        return {"sequence": sequence, "value": value, "denominator": denominator_value}

    worst = {
        "object_frame_support_fraction": worst_item("computable_3d_object_sensor_supported_fraction", None, True),
        "pair_support_fraction": worst_item("computable_motion_pair_sensor_supported_fraction", None, True),
        "centroid_residual_3d_median_m": worst_item("centroid_residual_3d_m", "p50", False),
        "centroid_residual_3d_p95_m": worst_item("centroid_residual_3d_m", "p95", False),
    }
    directions = {
        "far_range": {"status": direction_status(far_values), "per_sequence": far_values},
        "three_d_only": {"status": direction_status(three_d_only_values), "per_sequence": three_d_only_values},
    }
    ledger = {
        "schema": LEDGER_SCHEMA,
        "stage": STAGE,
        "status": "COMPLETE",
        "config_sha256": sha256_file(config_path),
        "sequence_freeze_sha256": manifest["sequence_freeze_sha256"],
        "input_manifest_sha256": sha256_file(manifest_path),
        "per_sequences": per_sequence,
        "pooled_primitives": {
            "object_frames": object_rows,
            "motion_pairs": pair_rows,
            "acceleration_triples": acceleration_rows,
        },
        "pooled_denominators": {
            "union_object_frames": kernel.denominator(object_rows),
            "computable_3d_object_frames": kernel.denominator(three_d_rows),
            "motion_pairs": kernel.denominator(pair_rows),
            "acceleration_triples": kernel.denominator(acceleration_rows),
        },
        "pooled_summary": pooled_summary,
        "worst_sequence": worst,
        "directional_replication": directions,
        "authority": config["authority"],
    }
    supported = ledger["pooled_denominators"]["computable_3d_object_frames"]["sensor-supported"]
    if supported == 0:
        terminal = "NOT_EVALUABLE_CROSS_SEQUENCE_SUPPORT"
    elif all(value["status"] == "DIRECTION_REPLICATED" for value in directions.values()):
        terminal = "CROSS_SEQUENCE_PROFILE_AVAILABLE_REPLICATION_CONSISTENT"
    else:
        terminal = "CROSS_SEQUENCE_PROFILE_AVAILABLE_WITH_PARTIAL_REPLICATION"
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "stage": STAGE,
        "status": "COMPLETE",
        "terminal_state": terminal,
        "validity": "PENDING_INDEPENDENT_VALIDATION",
        "config_sha256": sha256_file(config_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "ledger_sha256": hashlib.sha256(canonical_bytes(ledger)).hexdigest(),
        "sequence_count": len(per_sequence),
        "pooled_denominators": ledger["pooled_denominators"],
        "pooled_summary": pooled_summary,
        "worst_sequence": worst,
        "directional_replication": directions,
        "authority": config["authority"],
    }
    return ledger, receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("materialize-inputs", "freeze-inputs", "support", "aggregate"), required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(repo, config_path)
    if args.phase == "materialize-inputs":
        results = materialize_inputs(repo, config, config_path)
        print(json.dumps({"status": "INPUTS_MATERIALIZED_NOT_FROZEN", "sequences": results}))
        return 0
    if args.phase == "freeze-inputs":
        output = write_input_manifest(repo, config, config_path)
        print(json.dumps({"status": "INPUTS_FROZEN_BEFORE_SUPPORT", "output": str(output)}))
        return 0
    ledger, receipt = build_aggregate(repo, config, config_path)
    ledger_path = repo / config["outputs"]["ledger"]
    receipt_path = repo / config["outputs"]["receipt"]
    require(not ledger_path.exists() and not receipt_path.exists(), "support outputs already exist; never overwrite")
    write_canonical(ledger_path, ledger)
    write_canonical(receipt_path, receipt)
    print(json.dumps({"terminal_state": receipt["terminal_state"], "ledger": str(ledger_path), "receipt": str(receipt_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
