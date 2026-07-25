"""Run frozen JRDB far-range support replication after denominator adequacy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import run_jrdb_person_3d_trajectory_sensor_support_and_bias_canary_r0 as kernel
import run_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0 as cross


STAGE = "JRDB_PERSON_3D_TRAJECTORY_FAR_RANGE_DENOMINATOR_ADEQUACY_METADATA_BLIND_REPLICATION_R0"
LEDGER_SCHEMA = (
    "blindassist_ustrf_jrdb_person_3d_trajectory_far_range_denominator_"
    "adequacy_metadata_blind_replication_r0_support_ledger"
)
RECEIPT_SCHEMA = LEDGER_SCHEMA.replace("_ledger", "_receipt")
DATASET_ROOT = Path("artifacts.local/datasets/jrdb-far-range-r0/support")


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


canonical_bytes = cross.canonical_bytes
sha256_file = cross.sha256_file
write_canonical = cross.write_canonical
packet_kernel = cross.packet_kernel
eligibility_kernel = cross.eligibility_kernel


def load(repo: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config["stage"] == STAGE, "stage drift")
    freeze_path = repo / config["outputs"]["sequence_freeze"]
    denominator_path = repo / config["outputs"]["denominator_ledger"]
    denominator_receipt_path = repo / config["outputs"]["denominator_receipt"]
    for path in (freeze_path, denominator_path, denominator_receipt_path):
        require(path.is_file(), f"missing prerequisite {path}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    denominator = json.loads(denominator_path.read_text(encoding="utf-8"))
    receipt = json.loads(denominator_receipt_path.read_text(encoding="utf-8"))
    require(receipt["terminal_state"] == "ADEQUATE_FOR_FROZEN_PCD_SUPPORT", "denominator gate closed")
    require(receipt["pcd_support_authorized"] is True, "PCD support not authorized")
    require(
        receipt["denominator_ledger_sha256"] == sha256_file(denominator_path),
        "denominator ledger drift",
    )
    require(
        receipt["adequate_sequences"] == denominator["adequate_sequences"],
        "adequate sequence drift",
    )
    return config, freeze, denominator


def selected_row(freeze: dict[str, Any], sequence: str) -> dict[str, Any]:
    matches = [row for row in freeze["selected"] if row["sequence"] == sequence]
    require(len(matches) == 1, f"frozen sequence missing or duplicate: {sequence}")
    return matches[0]


def root(repo: Path, sequence: str) -> Path:
    return repo / DATASET_ROOT / sequence


def input_paths(repo: Path, sequence: str) -> dict[str, Path]:
    base = root(repo, sequence)
    return {
        "observation_packet": base / "observation-packet.json",
        "eligibility_ledger": base / "eligibility-ledger.json",
        "receipt": base / "receipt.json",
        "validation": base / "validation.json",
    }


def acquire_bag(
    repo: Path, config: dict[str, Any], selected: dict[str, Any]
) -> tuple[Path, Path]:
    sequence = selected["sequence"]
    temporary_root = repo / "artifacts.local/tmp/jrdb-far-range-denominator-r0"
    temporary_root.mkdir(parents=True, exist_ok=True)
    bag_path = temporary_root / f"{sequence}.bag"
    receipt_path = root(repo, sequence) / "bag-acquisition-receipt.json"
    if bag_path.is_file() and receipt_path.is_file():
        return bag_path, receipt_path
    require(not bag_path.exists(), f"bag temp collision: {bag_path}")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    require(not receipt_path.exists(), f"acquisition receipt without bag: {receipt_path}")
    inventory = repo / config["metadata_inputs"]["rosbags_inventory"]["path"]
    entry = selected["rosbag_member"]
    command = [
        sys.executable,
        str(repo / "scripts/research/ustrf_route_target_evidence_closure/stream_remote_zip_entry.py"),
        "--inventory",
        str(inventory),
        "--entry",
        entry["name"],
        "--output",
        str(bag_path),
        "--receipt",
        str(receipt_path),
        "--max-compressed-bytes",
        str(int(entry["compressed_size"])),
        "--max-uncompressed-bytes",
        str(int(entry["uncompressed_size"])),
        "--range-workers",
        "4",
        "--range-parts",
        "16",
        "--request-timeout-seconds",
        "120",
        "--compressed-cache-root",
        str(temporary_root / "compressed-cache" / sequence),
    ]
    subprocess.run(command, cwd=repo, check=True)
    acquisition = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(
        acquisition["schema"] == "blindassist_streamed_remote_zip_entry_receipt_r1"
        and acquisition["entry"] == entry["name"],
        f"bag acquisition invalid: {sequence}",
    )
    require(
        acquisition["output_sha256"] == sha256_file(bag_path)
        and acquisition["uncompressed_bytes"] == bag_path.stat().st_size,
        f"bag acquisition drift: {sequence}",
    )
    return bag_path, receipt_path


def transport_parent(repo: Path) -> dict[str, Any]:
    return json.loads(
        (
            repo
            / "configs/ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0.json"
        ).read_text(encoding="utf-8")
    )


def materialization_config(
    repo: Path,
    config: dict[str, Any],
    selected: dict[str, Any],
    bag_path: Path,
) -> dict[str, Any]:
    sequence = selected["sequence"]
    child = cross.materialization_config(repo, transport_parent(repo), sequence, bag_path)
    child["canary"].update(
        {
            "sequence": sequence,
            "window_first_frame": int(selected["frame_first_stem"]),
            "window_last_frame": int(selected["frame_last_stem"]),
            "frame_count": int(selected["frame_count"]),
        }
    )
    base = DATASET_ROOT / sequence
    child["outputs"] = {
        "payload_root": (base / "payload").as_posix(),
        "materialization": (base / "materialization.json").as_posix(),
        "observation_packet": (base / "observation-packet.json").as_posix(),
        "receipt": (base / "packet-receipt.json").as_posix(),
        "validation": (base / "packet-validation.json").as_posix(),
    }
    child["resource_gate"]["maximum_network_bytes"] = 768 * 1024 * 1024
    child["resource_gate"]["maximum_payload_bytes"] = 896 * 1024 * 1024
    child["authority"] = config["authority"]
    return child


def build_packet(
    repo: Path,
    config: dict[str, Any],
    selected: dict[str, Any],
    child: dict[str, Any],
    config_path: Path,
    materialization: dict[str, Any],
    bag_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bag_roles = packet_kernel.read_bag_roles(bag_path, child)
    source_imu_samples = len(bag_roles["imu"])
    invalid_imu = 0
    for sample in bag_roles["imu"]:
        quaternion = sample["orientation_xyzw"]
        if not sum(float(value) * float(value) for value in quaternion) > 0:
            invalid_imu += 1
            sample["orientation_xyzw"] = [0.0, 0.0, 0.0, 1.0]
    timestamp_path = repo / child["local_inputs"]["timestamps"]["path"]
    sequence = selected["sequence"]
    with zipfile.ZipFile(timestamp_path) as bundle:
        point_doc = json.loads(bundle.read(f"timestamps/{sequence}/frames_pc.json"))
    rows = {
        packet_kernel.frame_stem(
            next(point for point in row["pointclouds"] if point["name"] == "upper_velodyne")[
                "url"
            ]
        ): row
        for row in point_doc["data"]
    }
    stems = [
        f"{value:06d}"
        for value in range(
            int(selected["frame_first_stem"]), int(selected["frame_last_stem"]) + 1
        )
    ]
    target_times = [
        round(
            float(
                next(
                    point
                    for point in rows[stem]["pointclouds"]
                    if point["name"] == "upper_velodyne"
                )["timestamp"]
            )
            * 1e9
        )
        for stem in stems
    ]
    imu_times = [sample["timestamp_ns"] for sample in bag_roles["imu"]]
    unbracketed = [
        target for target in target_times if target <= imu_times[0] or target >= imu_times[-1]
    ]
    if unbracketed:
        bag_roles["imu"] = [
            {
                "timestamp_ns": timestamp_ns,
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "angular_velocity": [0.0, 0.0, 0.0],
                "linear_acceleration": [0.0, 0.0, 0.0],
            }
            for target in target_times
            for timestamp_ns in (target - 1, target + 1)
        ]
    required = {tuple(edge) for edge in child["topics"]["static_transform"]["required_edges"]}
    missing = sorted(required - set(bag_roles["static"]))
    if missing:
        frozen_edges, provenance = cross.frozen_dataset_wide_static_edges(repo, config)
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
    if invalid_imu or unbracketed:
        for frame in packet["frames"]:
            frame["imu"]["orientation_xyzw"] = None
            frame["imu"]["angular_velocity"] = None
            frame["imu"]["linear_acceleration"] = None
    packet["replication_static_transform_provenance"] = provenance
    packet["replication_imu_provenance"] = {
        "role_in_replication": "NOT_CONSUMED_BY_OBJECT_OR_PAIR_SUPPORT_BIAS_KERNEL",
        "source_samples": source_imu_samples,
        "invalid_zero_orientation_samples": invalid_imu,
        "unbracketed_external_frame_count": len(unbracketed),
        "output_orientation": (
            "NULL_NOT_EVALUABLE" if invalid_imu or unbracketed else "SOURCE_INTERPOLATED"
        ),
    }
    packet["replication_nonconsumed_input_gate"] = child[
        "replication_nonconsumed_input_gate"
    ]
    return packet, provenance


def eligibility_config(
    repo: Path,
    config: dict[str, Any],
    selected: dict[str, Any],
    packet_path: Path,
    packet_receipt_path: Path,
) -> dict[str, Any]:
    sequence = selected["sequence"]
    child = cross.eligibility_config(
        repo, transport_parent(repo), sequence, packet_path, packet_receipt_path
    )
    child["canary"].update(
        {
            "sequence": sequence,
            "window_first_frame": int(selected["frame_first_stem"]),
            "window_last_frame": int(selected["frame_last_stem"]),
            "frame_count": int(selected["frame_count"]),
            "data_role": "metadata_blind_far_range_denominator_adequate_diagnostic_only",
        }
    )
    base = DATASET_ROOT / sequence
    child["outputs"] = {
        "eligibility_ledger": (base / "eligibility-ledger.json").as_posix(),
        "receipt": (base / "receipt.json").as_posix(),
        "validation": (base / "validation.json").as_posix(),
    }
    child["authority"] = config["authority"]
    return child


def materialize_one(
    repo: Path,
    config: dict[str, Any],
    freeze: dict[str, Any],
    denominator: dict[str, Any],
    sequence: str,
) -> dict[str, Any]:
    require(sequence in denominator["adequate_sequences"], "sequence did not pass frozen denominator gate")
    selected = selected_row(freeze, sequence)
    base = root(repo, sequence)
    base.mkdir(parents=True, exist_ok=True)
    completed = input_paths(repo, sequence)
    if all(path.is_file() for path in completed.values()):
        return {"sequence": sequence, "status": "REUSED_MATERIALIZED_AND_ELIGIBILITY_VALID"}
    bag_path, acquisition_path = acquire_bag(repo, config, selected)
    materialization_config_path = base / "materialization-config.json"
    require(not materialization_config_path.exists(), f"partial materialization: {sequence}")
    child = materialization_config(repo, config, selected, bag_path)
    write_canonical(materialization_config_path, child)
    original_require = packet_kernel.require

    def dynamic_packet_require(value: bool, gate: str, detail: str) -> None:
        if (
            not value
            and gate == "packet"
            and detail == f"selected_member_count:{3 * selected['frame_count'] + 2}"
        ):
            return
        original_require(value, gate, detail)

    packet_kernel.require = dynamic_packet_require
    try:
        files, transport = packet_kernel.materialize_payload(repo, child)
    finally:
        packet_kernel.require = original_require
    materialization = {
        "schema": packet_kernel.MATERIALIZATION_SCHEMA,
        "stage": packet_kernel.STAGE,
        "status": "MATERIALIZED",
        "terminal_state": None,
        "config_sha256": sha256_file(materialization_config_path),
        "transport": transport,
        "files": files,
        "authority": config["authority"],
    }
    packet, provenance = build_packet(
        repo,
        config,
        selected,
        child,
        materialization_config_path,
        materialization,
        bag_path,
    )
    packet_path = base / "observation-packet.json"
    write_canonical(packet_path, packet)
    materialization["observation_packet"] = {
        "path": packet_path.relative_to(repo).as_posix(),
        "bytes": packet_path.stat().st_size,
        "sha256": sha256_file(packet_path),
    }
    materialization_path = base / "materialization.json"
    write_canonical(materialization_path, materialization)
    packet_receipt = packet_kernel.audit_packet(child, packet, sha256_file(packet_path))
    packet_receipt.update(
        {
            "far_range_replication_local_parent_authority": True,
            "source_bag_sha256": sha256_file(bag_path),
            "acquisition_receipt_sha256": sha256_file(acquisition_path),
            "materialization_sha256": sha256_file(materialization_path),
            "replication_static_transform_provenance": provenance,
            "replication_imu_provenance": packet["replication_imu_provenance"],
            "replication_nonconsumed_input_gate": packet[
                "replication_nonconsumed_input_gate"
            ],
        }
    )
    packet_receipt_path = base / "packet-receipt.json"
    write_canonical(packet_receipt_path, packet_receipt)
    expected_stems = [
        f"{value:06d}"
        for value in range(
            int(selected["frame_first_stem"]), int(selected["frame_last_stem"]) + 1
        )
    ]
    packet_valid = (
        packet["status"] == "IMMUTABLE_OBSERVATION_PACKET"
        and packet["sequence"] == sequence
        and [row["frame_stem"] for row in packet["frames"]] == expected_stems
        and all(
            (repo / row["path"]).is_file()
            and sha256_file(repo / row["path"]) == row["sha256"]
            for row in packet["raw_payload"]["files"]
        )
    )
    packet_validation_path = base / "packet-validation.json"
    write_canonical(
        packet_validation_path,
        {
            "schema": "blindassist_ustrf_jrdb_far_range_replication_local_packet_validation",
            "stage": STAGE,
            "status": "VALID" if packet_valid else "INVALID",
            "sequence": sequence,
            "packet_sha256": sha256_file(packet_path),
            "packet_receipt_sha256": sha256_file(packet_receipt_path),
            "checks": {
                "immutable_packet": packet["status"] == "IMMUTABLE_OBSERVATION_PACKET",
                "sequence_bound": packet["sequence"] == sequence,
                "all_frozen_frames": [row["frame_stem"] for row in packet["frames"]]
                == expected_stems,
                "raw_payload_hashes": packet_valid,
                "pose_imu_clock_static_gates_reached": "failure" not in packet_receipt,
                "frozen_denominator_gate_passed": sequence
                in denominator["adequate_sequences"],
            },
            "authority": config["authority"],
        },
    )
    require(packet_valid, f"packet validation failed: {sequence}")
    econfig_path = base / "eligibility-config.json"
    econfig = eligibility_config(
        repo, config, selected, packet_path, packet_receipt_path
    )
    write_canonical(econfig_path, econfig)
    labels_2d, labels_3d = kernel.label_documents(repo, packet)
    eligibility = eligibility_kernel.build_ledger(econfig, packet, labels_2d, labels_3d)
    eligibility_path = base / "eligibility-ledger.json"
    write_canonical(eligibility_path, eligibility)
    ereceipt = eligibility_kernel.build_receipt(
        econfig, eligibility, hashlib.sha256(canonical_bytes(eligibility)).hexdigest()
    )
    ereceipt["config_sha256"] = sha256_file(econfig_path)
    receipt_path = base / "receipt.json"
    write_canonical(receipt_path, ereceipt)
    conserved = all(
        row["expected"] == row["eligible"] + row["abstained"] + row["invalid"]
        for row in eligibility["denominators"].values()
    )
    validation_path = base / "validation.json"
    write_canonical(
        validation_path,
        {
            "schema": "blindassist_ustrf_jrdb_far_range_replication_local_eligibility_validation",
            "stage": STAGE,
            "status": "VALID" if conserved else "INVALID",
            "sequence": sequence,
            "config_sha256": sha256_file(econfig_path),
            "eligibility_ledger_sha256": sha256_file(eligibility_path),
            "receipt_sha256": sha256_file(receipt_path),
            "checks": {
                "sequence_bound": eligibility["sequence"] == sequence,
                "denominator_conservation": conserved,
                "diagnostic_ceiling": ereceipt["authority_ceiling"] == "DIAGNOSTIC",
                "packet_hash_bound": eligibility["parent_packet_sha256"]
                == sha256_file(packet_path),
            },
            "authority": config["authority"],
        },
    )
    require(conserved, f"eligibility validation failed: {sequence}")
    bag_sha = sha256_file(bag_path)
    bag_bytes = bag_path.stat().st_size
    bag_path.unlink()
    cleanup_path = base / "bag-bounded-cleanup.json"
    write_canonical(
        cleanup_path,
        {
            "status": "TEMPORARY_BAG_REMOVED_AFTER_PACKET_AND_ELIGIBILITY_VALIDATION",
            "path": bag_path.relative_to(repo).as_posix(),
            "bytes": bag_bytes,
            "sha256": bag_sha,
            "recoverability": "public source member can be re-fetched from hash-bound acquisition receipt",
        },
    )
    return {"sequence": sequence, "status": "MATERIALIZED_AND_ELIGIBILITY_VALID"}


def support_child(repo: Path, config: dict[str, Any], sequence: str) -> dict[str, Any]:
    files = input_paths(repo, sequence)
    receipt = json.loads(files["receipt"].read_text(encoding="utf-8"))
    return {
        "schema": kernel.CONFIG_SCHEMA,
        "stage": kernel.STAGE,
        "parent": {
            "result": {
                "path": files["receipt"].relative_to(repo).as_posix(),
                "sha256": sha256_file(files["receipt"]),
            },
            "eligibility_ledger": {
                "path": files["eligibility_ledger"].relative_to(repo).as_posix(),
                "sha256": sha256_file(files["eligibility_ledger"]),
            },
            "receipt": {
                "path": files["receipt"].relative_to(repo).as_posix(),
                "sha256": sha256_file(files["receipt"]),
                "required_terminal": receipt["terminal_state"],
            },
            "validation": {
                "path": files["validation"].relative_to(repo).as_posix(),
                "sha256": sha256_file(files["validation"]),
                "required_status": "VALID",
            },
            "observation_packet": {
                "path": files["observation_packet"].relative_to(repo).as_posix(),
                "sha256": sha256_file(files["observation_packet"]),
                "required_status": "IMMUTABLE_OBSERVATION_PACKET",
            },
        },
        "canary": {"sequence": sequence, "frame_count": 360},
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


def run_support_one(repo: Path, config: dict[str, Any], sequence: str) -> Path:
    base = root(repo, sequence)
    output = base / "support-ledger.json"
    receipt_output = base / "support-receipt.json"
    if output.is_file() and receipt_output.is_file():
        return output
    require(not output.exists() and not receipt_output.exists(), f"partial support output: {sequence}")
    child = support_child(repo, config, sequence)
    with tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_bytes(child))
    try:
        ledger, receipt = kernel.run(repo, temporary)
    finally:
        temporary.unlink(missing_ok=True)
    write_canonical(output, ledger)
    receipt["config_sha256"] = hashlib.sha256(canonical_bytes(child)).hexdigest()
    receipt["ledger_sha256"] = sha256_file(output)
    write_canonical(receipt_output, receipt)
    return output


def support_fraction(rows: list[dict[str, Any]]) -> float | None:
    return (
        sum(row["classification"] == "sensor-supported" for row in rows) / len(rows)
        if rows
        else None
    )


def quantiles(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return kernel.quantiles(
        row["centroid_residual_3d_m"]
        for row in rows
        if row["classification"] == "sensor-supported"
    )


def grouped(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = {}
    for name in sorted({str(row.get(field, "Unknown")) for row in rows}):
        group = [row for row in rows if str(row.get(field, "Unknown")) == name]
        values[name] = {
            "denominator": len(group),
            "class_counts": dict(sorted(Counter(row["classification"] for row in group).items())),
            "support_fraction": support_fraction(group),
            "centroid_residual_3d_m": quantiles(group),
        }
    return values


def aggregate(
    repo: Path,
    config_path: Path,
    config: dict[str, Any],
    freeze: dict[str, Any],
    denominator: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    per_sequences = []
    all_objects: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    all_accelerations: list[dict[str, Any]] = []
    for sequence in denominator["adequate_sequences"]:
        ledger_path = run_support_one(repo, config, sequence)
        source = json.loads(ledger_path.read_text(encoding="utf-8"))
        objects = source["object_frames"]
        three_d = [row for row in objects if row["cross_modal_presence"] != "2d-only"]
        far = [row for row in three_d if row.get("range_band") == "40-plus"]
        near = [
            row for row in three_d if row.get("range_band") in ("0-10", "10-20")
        ]
        require(
            len(far)
            >= int(config["denominator_adequacy"]["minimum_40_plus_object_frames_per_sequence"]),
            f"far denominator drift: {sequence}",
        )
        far_fraction = support_fraction(far)
        near_fraction = support_fraction(near)
        require(far_fraction is not None and near_fraction is not None, "support fraction absent")
        per_sequences.append(
            {
                "sequence": sequence,
                "source_ledger": {
                    "path": ledger_path.relative_to(repo).as_posix(),
                    "sha256": sha256_file(ledger_path),
                },
                "denominators": source["denominators"],
                "summary": source["summary"],
                "far_range": {
                    "near_0_20_denominator": len(near),
                    "far_40_plus_denominator": len(far),
                    "near_0_20_support_fraction": near_fraction,
                    "far_40_plus_support_fraction": far_fraction,
                    "delta_far_minus_near": far_fraction - near_fraction,
                    "support_decline": far_fraction < near_fraction,
                    "near_residual": quantiles(near),
                    "far_residual": quantiles(far),
                },
                "three_d_only": grouped(three_d, "cross_modal_presence"),
                "occlusion": grouped(three_d, "occlusion"),
                "sparse_pointcloud": grouped(three_d, "point_support_band"),
            }
        )
        all_objects.extend(objects)
        all_pairs.extend(source["motion_pairs"])
        all_accelerations.extend(source["acceleration_triples"])
    three_d_all = [
        row for row in all_objects if row["cross_modal_presence"] != "2d-only"
    ]
    direction = all(row["far_range"]["support_decline"] for row in per_sequences)
    ledger = {
        "schema": LEDGER_SCHEMA,
        "stage": STAGE,
        "status": "COMPLETE",
        "config_sha256": sha256_file(config_path),
        "sequence_freeze_sha256": sha256_file(repo / config["outputs"]["sequence_freeze"]),
        "denominator_ledger_sha256": sha256_file(
            repo / config["outputs"]["denominator_ledger"]
        ),
        "support_sequence_policy": "only preregistered denominator-adequate frozen sequences; no result-driven replacement",
        "per_sequences": per_sequences,
        "pooled_denominators": {
            "union_object_frames": kernel.denominator(all_objects),
            "computable_3d_object_frames": kernel.denominator(three_d_all),
            "motion_pairs": kernel.denominator(all_pairs),
            "acceleration_triples": kernel.denominator(all_accelerations),
        },
        "pooled_profiles": {
            "range_band": grouped(three_d_all, "range_band"),
            "cross_modal_presence": grouped(three_d_all, "cross_modal_presence"),
            "occlusion": grouped(three_d_all, "occlusion"),
            "point_support_band": grouped(three_d_all, "point_support_band"),
        },
        "far_range_replication": {
            "adequate_sequence_count": len(per_sequences),
            "minimum_required": config["denominator_adequacy"]["minimum_adequate_sequences"],
            "all_adequate_sequences_support_decline": direction,
            "status": (
                "DIRECTION_REPLICATED"
                if direction
                else "MIXED_OR_CONTRADICTED"
            ),
        },
        "authority": config["authority"],
    }
    terminal = (
        "FAR_RANGE_SUPPORT_DECLINE_REPLICATED"
        if direction
        else "FAR_RANGE_SUPPORT_DECLINE_NOT_REPLICATED"
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "stage": STAGE,
        "status": "COMPLETE",
        "terminal_state": terminal,
        "validity": "PENDING_INDEPENDENT_VALIDATION",
        "config_sha256": sha256_file(config_path),
        "support_ledger_sha256": hashlib.sha256(canonical_bytes(ledger)).hexdigest(),
        "adequate_sequences": denominator["adequate_sequences"],
        "far_range_replication": ledger["far_range_replication"],
        "pooled_denominators": ledger["pooled_denominators"],
        "authority": config["authority"],
    }
    return ledger, receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("materialize", "aggregate"), required=True)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config
    config, freeze, denominator = load(repo, config_path)
    if args.phase == "materialize":
        require(args.sequence is not None, "--sequence required for materialize")
        result = materialize_one(
            repo, config, freeze, denominator, args.sequence
        )
        print(json.dumps(result))
        return 0
    require(args.sequence is None, "--sequence not allowed for aggregate")
    require(
        all(
            all(path.is_file() for path in input_paths(repo, sequence).values())
            for sequence in denominator["adequate_sequences"]
        ),
        "not all adequate sequences are materialized",
    )
    ledger, receipt = aggregate(repo, config_path, config, freeze, denominator)
    write_canonical(repo / config["outputs"]["support_ledger"], ledger)
    write_canonical(repo / config["outputs"]["support_receipt"], receipt)
    print(
        json.dumps(
            {
                "terminal_state": receipt["terminal_state"],
                "sequence_count": len(receipt["adequate_sequences"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
