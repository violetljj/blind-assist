from __future__ import annotations

import argparse
from bisect import bisect_left
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any


OPENLORIS_WINDOW_ID = "corridor1-1:w004"
OPENLORIS_START = Decimal("1560000043.537699")
OPENLORIS_END = Decimal("1560000053.537699")
DLR_WINDOW_ID = "extreme_geometry/hexagon_01:w001"
DLR_START = Decimal("1634201323.995618343")
DLR_END = Decimal("1634201333.995618343")
PAIR_TOLERANCE = Decimal("0.005")

EXPECTED_SHA256 = {
    "docs/research/rcle/RCLE_SOURCE_DISCOVERY_R4_SEGMENT_LEVEL_COHORT_DESIGN_R1_2026-07-28.json": "98428718c4be6a25d5a3c08d3264d20521f290a5b11b57765df362d13de7ed14",
    "artifacts.local/evidence/rcle_source_discovery_r3/r4_segment_level_cohort_design_r1_independent_review.json": "6c3bf4174929acead912237390bf96d30521717e9526698f9c23a92e87d891c1",
    "artifacts.local/evidence/rcle_source_discovery_r3/r4_segment_level_cohort_design_r1_signature.json": "5aa92970662ed9e5eb9bed2d648af771b67c1c92a5c4abc2706be4bf3e34247e",
    "artifacts.local/evidence/rcle_source_authority_repair_r1/openloris_corridor1-1_7z_directory.json": "7137b377d81c719b2f7644318bc0bd7785b46b16c97094d47936a0d0711d063c",
    "artifacts.local/evidence/rcle_source_authority_repair_r1/openloris_geometry_only_transport_preflight.json": "06c55a12ddccf4dd8beb25c9769baa88f96c18dc30e32280a35e1d60a6d559d1",
    "artifacts.local/evidence/rcle_source_authority_repair_r1/geometry-r2/openloris/corridor1-1/result.json": "07964d8934c278a305916e49ec0bb61fd4592bba38da8694df2d9913c1db8461",
    "artifacts.local/evidence/rcle_source_authority_repair_r1/geometry-r2/openloris/corridor1-1/geometry_pair_ledger.jsonl": "772746c1d4654c2e2b67829d3d5f254a9e9b6be88a7671f10cfac3299e8ab903",
    "artifacts.local/work/rcle-source-discovery-r3/dlr-realsense-directory.json": "3ed6d21323d0056824459e0cc5fb690182883b43f5402b7152ba2c12f4f1a680",
    "artifacts.local/datasets/rcle_unseen_external_confirmation_source_discovery_r3/successor_geometry/dlr/extreme_geometry__hexagon_01/realsense.bag.identity.json": "bfbcb930df690231a6ba36a839b6751521384bda7f677bfdc0197af25bbf5ecf",
    "artifacts.local/evidence/rcle_source_discovery_r3/successor_geometry/dlr/extreme_geometry__hexagon_01/geometry_windows/result.json": "f08deaed664fc9a22f5a5b50128614c91165fed9e4ab59df49c7aac3336f8883",
    "artifacts.local/evidence/rcle_source_discovery_r3/successor_geometry/dlr/extreme_geometry__hexagon_01/geometry_windows/geometry_pair_ledger.jsonl": "d94b426cc7342ae3522868d2dc6b3038b3c87ead7df10a41d5b6929b7c740bab",
    "artifacts.local/evidence/rcle_source_discovery_r3/successor_geometry/dlr/extreme_geometry__hexagon_01/timegrid_gate.json": "98fbb6369f8103154b197c9318f48fe780a5df47c1ecd6b9899d15ceb8003940",
    "scripts/research/egomotion_compensated_looming/configs/phase_a_synthetic_signal_audit_r0.json": "d20e77f3ea5f7ac55376006f1d14feb0ffb5daffd10a42792912fb89cdb1b502",
    "docs/research/rcle/RCLE_OBSERVABLE_SUPPORT_RECOVERY_R0_DESIGN_LOCK_2026-07-26.json": "3fcc21e28ba84e18d10b1c236a9a0df167d2a6464ea5ebefcb52ce4395152bac",
    "scripts/research/egomotion_compensated_looming/rcle_observable_support_r0/RCLE_OBSERVABLE_SUPPORT_MANAGER_R0_IMPLEMENTATION_LOCK.json": "a1dc1388ea6b6cb8ff7e7541da407cb827b6384678cdf46a097426a2111a5497",
    "scripts/research/egomotion_compensated_looming/rcle_minimal/evaluation.py": "068cca009db7dbad33d54cfde7c5ac12e9a0d2fac445d849c9fc0b73e91d2f8c",
    "scripts/research/egomotion_compensated_looming/rcle_minimal/rotation_compensation.py": "ab964ebaf09456dffe6fe57248bc0501dbe8547113a1cab931f1953598e3ff24",
    "scripts/research/egomotion_compensated_looming/rcle_minimal_r1/sparse_flow.py": "36401e8335266896cd91f153b0772ef0718004ff03f527af965b63add70973ca",
    "scripts/research/egomotion_compensated_looming/rcle_minimal_r1/local_expansion.py": "41e67c0f30b85a9a67449d96df14188093560e581aa415c8957cc0377cc8dbd6",
    "scripts/research/egomotion_compensated_looming/rcle_observable_support_r0/support_manager.py": "83ac2ef3cce3fce625b46574aa67b38ed96679a9579e6d844066da709cfc7d08",
    "scripts/research/egomotion_compensated_looming/rcle_observable_support_r0/evaluation.py": "00b6cf601bf3073e9c261b04bbca9a8cb2b01de4c4b9bfed0bab62a5576a27b1",
    "scripts/research/egomotion_compensated_looming/rcle_low_reference_false_trigger_r1/temporal_confirmation.py": "3533a9164943dcb0a6252450e757661e31f4525791a7e735a6f48be70ee704ee",
    "docs/research/rcle/RCLE_LOW_REFERENCE_TEMPORAL_CONFIRMATION_R1_CONTRACT_2026-07-27.json": "9806211b4ce1b0585ec2c0fbd08b3d52aa72532b797ac225f6390ea1e2f092dd",
    "scripts/research/egomotion_compensated_looming/rgb_algorithm_development_canary_cid_sims_r0/producer.py": "23211b1acec517b3c487d183742035887ff250bf51019b0cd1e1dea1d809bec5",
    "scripts/research/egomotion_compensated_looming/rgb_segment_confirmation_r1/opaque_transport.py": "bc507980baa2ee6bffb9ffa515c2d79c8847aadd9387e537fad4e30bb41ee78f",
    "artifacts.local/evidence/rcle_rgb_segment_confirmation_r1/opaque_transport_runtime_lock.v1.json": "c579184c538ddc83c2fb528adf1bd2617a71e12fcfe9dece085541151d08fd0f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_window_timestamps(path: Path, window_id: str) -> list[Decimal]:
    timestamps: set[Decimal] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("window_id") != window_id:
                continue
            timestamps.add(Decimal(str(row["previous_timestamp_s"])))
            timestamps.add(Decimal(str(row["current_timestamp_s"])))
    return sorted(timestamps)


def timestamp_from_member(path: str) -> Decimal:
    return Decimal(Path(path).stem)


def nearest_unique_pairs(
    reference: list[Decimal], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    remaining = list(candidates)
    result: list[dict[str, Any]] = []
    for reference_ts in reference:
        values = [row["timestamp"] for row in remaining]
        index = bisect_left(values, reference_ts)
        choices = [item for item in (index - 1, index) if 0 <= item < len(remaining)]
        chosen = min(
            choices,
            key=lambda item: (abs(remaining[item]["timestamp"] - reference_ts), item),
        )
        member = remaining.pop(chosen)
        delta = member["timestamp"] - reference_ts
        result.append(
            {
                "geometry_timestamp_s": str(reference_ts),
                "rgb_timestamp_s": str(member["timestamp"]),
                "rgb_member_path": member["path"],
                "rgb_member_uncompressed_bytes": member["uncompressed_bytes"],
                "rgb_member_crc32": member["crc32"],
                "rgb_payload_sha256": None,
                "rgb_minus_geometry_delta_s": str(delta),
            }
        )
    return result


def enforce_open_pairing(pair_inventory: list[dict[str, Any]]) -> Decimal:
    if len({row["rgb_member_path"] for row in pair_inventory}) != len(pair_inventory):
        raise ValueError("OPENLORIS_RGB_MEMBER_REUSE")
    geometry_order = [
        Decimal(row["geometry_timestamp_s"]) for row in pair_inventory
    ]
    rgb_order = [Decimal(row["rgb_timestamp_s"]) for row in pair_inventory]
    if any(right <= left for left, right in zip(geometry_order, geometry_order[1:])):
        raise ValueError("OPENLORIS_GEOMETRY_TIMESTAMP_NOT_STRICT")
    if any(right <= left for left, right in zip(rgb_order, rgb_order[1:])):
        raise ValueError("OPENLORIS_RGB_TIMESTAMP_NOT_STRICT")
    max_abs_delta = max(
        abs(Decimal(row["rgb_minus_geometry_delta_s"])) for row in pair_inventory
    )
    if max_abs_delta > PAIR_TOLERANCE:
        raise ValueError(f"OPENLORIS_PAIR_TOLERANCE:{max_abs_delta}")
    return max_abs_delta


def binding(repo: Path, relative: str) -> dict[str, str]:
    path = repo / relative
    normalized = relative.replace("\\", "/")
    actual = sha256(path)
    expected = EXPECTED_SHA256.get(normalized)
    if expected is None:
        raise ValueError(f"UNFROZEN_BINDING:{normalized}")
    if actual != expected:
        raise ValueError(f"FROZEN_BINDING_DRIFT:{normalized}:{actual}")
    return {"path": normalized, "sha256": expected}


def build(repo: Path) -> dict[str, Any]:
    r4_design = (
        "docs/research/rcle/"
        "RCLE_SOURCE_DISCOVERY_R4_SEGMENT_LEVEL_COHORT_DESIGN_R1_2026-07-28.json"
    )
    r4_review = (
        "artifacts.local/evidence/rcle_source_discovery_r3/"
        "r4_segment_level_cohort_design_r1_independent_review.json"
    )
    r4_signature = (
        "artifacts.local/evidence/rcle_source_discovery_r3/"
        "r4_segment_level_cohort_design_r1_signature.json"
    )
    openloris_directory = (
        "artifacts.local/evidence/rcle_source_authority_repair_r1/"
        "openloris_corridor1-1_7z_directory.json"
    )
    openloris_result = (
        "artifacts.local/evidence/rcle_source_authority_repair_r1/geometry-r2/"
        "openloris/corridor1-1/result.json"
    )
    openloris_ledger = (
        "artifacts.local/evidence/rcle_source_authority_repair_r1/geometry-r2/"
        "openloris/corridor1-1/geometry_pair_ledger.jsonl"
    )
    openloris_transport_preflight = (
        "artifacts.local/evidence/rcle_source_authority_repair_r1/"
        "openloris_geometry_only_transport_preflight.json"
    )
    dlr_directory = (
        "artifacts.local/work/rcle-source-discovery-r3/dlr-realsense-directory.json"
    )
    dlr_identity = (
        "artifacts.local/datasets/rcle_unseen_external_confirmation_source_discovery_r3/"
        "successor_geometry/dlr/extreme_geometry__hexagon_01/realsense.bag.identity.json"
    )
    dlr_result = (
        "artifacts.local/evidence/rcle_source_discovery_r3/successor_geometry/dlr/"
        "extreme_geometry__hexagon_01/geometry_windows/result.json"
    )
    dlr_ledger = (
        "artifacts.local/evidence/rcle_source_discovery_r3/successor_geometry/dlr/"
        "extreme_geometry__hexagon_01/geometry_windows/geometry_pair_ledger.jsonl"
    )
    dlr_timegrid = (
        "artifacts.local/evidence/rcle_source_discovery_r3/successor_geometry/dlr/"
        "extreme_geometry__hexagon_01/timegrid_gate.json"
    )

    directory = read_json(repo / openloris_directory)
    transport_preflight = read_json(repo / openloris_transport_preflight)
    capture_preflight = transport_preflight["captures"]["corridor1-1"]
    folder_two = next(
        row
        for row in capture_preflight["geometry_solid_folders"]
        if row["folder_index"] == 2
    )
    preflight_window = next(
        row
        for row in capture_preflight["windows"]
        if row["window_id"] == "corridor1-1:W004"
    )
    if preflight_window["solid_folder_indices"] != [2]:
        raise ValueError("OPENLORIS_BOUND_SOLID_FOLDER")
    if folder_two["pack_bytes"] != 3_946_335_545:
        raise ValueError("OPENLORIS_BOUND_SOLID_PACK_BYTES")
    if folder_two["pack_bytes"] >= 3_947_000_000:
        raise ValueError("OPENLORIS_SOLID_PACK_EXCEEDS_BUDGET")
    color_members = []
    for item in directory["members"]:
        path = item["path"].replace("\\", "/")
        if "/color/" not in path or item.get("is_directory"):
            continue
        try:
            timestamp = timestamp_from_member(path)
        except Exception:
            continue
        color_members.append({**item, "path": path, "timestamp": timestamp})
    color_members.sort(key=lambda row: row["timestamp"])
    selected = [
        row
        for row in color_members
        if OPENLORIS_START <= row["timestamp"] < OPENLORIS_END
    ]
    if len(selected) != 300:
        raise ValueError(f"OPENLORIS_SELECTED_RGB_COUNT:{len(selected)}")
    first_index = color_members.index(selected[0])
    last_index = color_members.index(selected[-1])
    if first_index == 0 or last_index + 1 >= len(color_members):
        raise ValueError("OPENLORIS_GUARD_FRAME_ABSENT")
    guard_before = color_members[first_index - 1]
    guard_after = color_members[last_index + 1]

    openloris_geometry_timestamps = read_window_timestamps(
        repo / openloris_ledger, OPENLORIS_WINDOW_ID
    )
    if len(openloris_geometry_timestamps) != 300:
        raise ValueError(
            f"OPENLORIS_GEOMETRY_FRAME_COUNT:{len(openloris_geometry_timestamps)}"
        )
    pair_inventory = nearest_unique_pairs(openloris_geometry_timestamps, selected)
    max_abs_delta = enforce_open_pairing(pair_inventory)
    geometry_order = [Decimal(row["geometry_timestamp_s"]) for row in pair_inventory]
    rgb_order = [Decimal(row["rgb_timestamp_s"]) for row in pair_inventory]
    if not (
        guard_before["timestamp"] < rgb_order[0]
        and guard_after["timestamp"] > rgb_order[-1]
        and color_members[first_index - 1]["path"] == guard_before["path"]
        and color_members[last_index + 1]["path"] == guard_after["path"]
    ):
        raise ValueError("OPENLORIS_GUARD_NOT_IMMEDIATE")

    dlr_dir = read_json(repo / dlr_directory)
    dlr_members = dlr_dir.get("entries", dlr_dir.get("members", []))
    dlr_member = next(
        row
        for row in dlr_members
        if row["name"] == "extreme_geometry/hexagon_01.bag"
    )
    dlr_identity_data = read_json(repo / dlr_identity)
    dlr_geometry_timestamps = read_window_timestamps(repo / dlr_ledger, DLR_WINDOW_ID)
    if len(dlr_geometry_timestamps) != 299:
        raise ValueError(f"DLR_GEOMETRY_FRAME_COUNT:{len(dlr_geometry_timestamps)}")

    algorithm_paths = [
        "scripts/research/egomotion_compensated_looming/configs/phase_a_synthetic_signal_audit_r0.json",
        "docs/research/rcle/RCLE_OBSERVABLE_SUPPORT_RECOVERY_R0_DESIGN_LOCK_2026-07-26.json",
        "scripts/research/egomotion_compensated_looming/rcle_observable_support_r0/RCLE_OBSERVABLE_SUPPORT_MANAGER_R0_IMPLEMENTATION_LOCK.json",
        "scripts/research/egomotion_compensated_looming/rcle_minimal/evaluation.py",
        "scripts/research/egomotion_compensated_looming/rcle_minimal/rotation_compensation.py",
        "scripts/research/egomotion_compensated_looming/rcle_minimal_r1/sparse_flow.py",
        "scripts/research/egomotion_compensated_looming/rcle_minimal_r1/local_expansion.py",
        "scripts/research/egomotion_compensated_looming/rcle_observable_support_r0/support_manager.py",
        "scripts/research/egomotion_compensated_looming/rcle_observable_support_r0/evaluation.py",
        "scripts/research/egomotion_compensated_looming/rcle_low_reference_false_trigger_r1/temporal_confirmation.py",
        "docs/research/rcle/RCLE_LOW_REFERENCE_TEMPORAL_CONFIRMATION_R1_CONTRACT_2026-07-27.json",
        "scripts/research/egomotion_compensated_looming/rgb_algorithm_development_canary_cid_sims_r0/producer.py",
    ]
    source_bindings = [
        r4_design,
        r4_review,
        r4_signature,
        openloris_directory,
        openloris_transport_preflight,
        openloris_result,
        openloris_ledger,
        dlr_directory,
        dlr_identity,
        dlr_result,
        dlr_ledger,
        dlr_timegrid,
    ]

    return {
        "schema_version": "rcle.rgb_segment_confirmation.preaccess_lock.v1",
        "protocol_id": "RCLE_RGB_SEGMENT_CONFIRMATION_R1",
        "status": "IDENTITY_EXTRACTION_REVIEW_REQUIRED",
        "scope": {
            "selected_segment_count": 2,
            "new_source_discovery": False,
            "whole_source_review": False,
            "rgb_decode": False,
            "rgb_visualization": False,
            "rgb_algorithm_execution": False,
            "android_execution": False,
        },
        "source_bindings": [binding(repo, path) for path in source_bindings],
        "algorithm_bindings": [binding(repo, path) for path in algorithm_paths],
        "transport_bindings": [
            binding(
                repo,
                "scripts/research/egomotion_compensated_looming/"
                "rgb_segment_confirmation_r1/opaque_transport.py",
            )
        ],
        "runtime_bindings": [
            binding(
                repo,
                "artifacts.local/evidence/rcle_rgb_segment_confirmation_r1/"
                "opaque_transport_runtime_lock.v1.json",
            )
        ],
        "frozen_algorithm": {
            "continuous_estimator": "POSE_ROTATION_COMPENSATED_OBSERVABLE_THREE_FRAME_LOCAL_EXPANSION_R0",
            "threshold_per_s": "0.01",
            "threshold_operator": "STRICT_GT",
            "temporal_confirmation": "CAUSAL_THREE_PAIR_CONFIRMATION_R1",
            "reset": "WINDOW_BOUNDARY_OR_ABSTENTION_OR_AT_OR_BELOW_THRESHOLD",
            "tuning_or_candidate_substitution": False,
        },
        "segments": [
            {
                "source_family_id": "OPENLORIS_CORRIDOR",
                "role": "POSITIVE_APPROACH_WINDOW",
                "capture_id": "corridor1-1",
                "window_id": OPENLORIS_WINDOW_ID,
                "half_open_window_s": [str(OPENLORIS_START), str(OPENLORIS_END)],
                "geometry_frame_count": 300,
                "geometry_pair_count": 299,
                "rgb_frame_count": 300,
                "rgb_guard_frame_count": 2,
                "guard_before": {
                    "path": guard_before["path"],
                    "timestamp_s": str(guard_before["timestamp"]),
                    "uncompressed_bytes": guard_before["uncompressed_bytes"],
                    "crc32": guard_before["crc32"],
                },
                "guard_after": {
                    "path": guard_after["path"],
                    "timestamp_s": str(guard_after["timestamp"]),
                    "uncompressed_bytes": guard_after["uncompressed_bytes"],
                    "crc32": guard_after["crc32"],
                },
                "container": directory["bounded_remote_object"],
                "source_url": directory["source_url"],
                "member_identity": {
                    "preaccess_hash": "OUTER_SHA256_PLUS_7Z_MEMBER_CRC32",
                    "payload_sha256_required_before_rgb_execution": True,
                },
                "camera_relation": {
                    "rgb_stream": "color",
                    "aligned_depth_stream": "aligned_depth",
                    "aligned_depth_camera": "d400_color_optical_frame",
                    "pose_frame": "base_link",
                    "camera_extrinsic_applied_in_bound_geometry": True,
                    "pairing": "UNIQUE_NEAREST_RGB_TO_GEOMETRY_TIMESTAMP",
                    "maximum_observed_abs_pair_delta_s": str(max_abs_delta),
                    "maximum_allowed_abs_pair_delta_s": "0.005",
                },
                "frame_inventory": pair_inventory,
                "identity_state": "OPAQUE_MEMBER_SHA256_PENDING",
            },
            {
                "source_family_id": "DLR_RGBD_VICON",
                "role": "BELOW_TRIGGER_REFERENCE_WINDOW",
                "capture_id": "extreme_geometry/hexagon_01",
                "window_id": DLR_WINDOW_ID,
                "half_open_window_s": [str(DLR_START), str(DLR_END)],
                "geometry_frame_count": 299,
                "geometry_pair_count": 298,
                "required_rgb_guard_frame_count": 2,
                "closed_rgb_guard_frame_count": 0,
                "outer_url": dlr_dir["url"],
                "outer_object_bytes": dlr_dir["object_bytes"],
                "zip_member": dlr_member,
                "bag_identity": dlr_identity_data,
                "camera_relation": {
                    "rgb_topic": "PENDING_OPAQUE_TOPIC_INVENTORY",
                    "aligned_depth_topic": "/camera/aligned_depth_to_color/image_raw",
                    "camera_info_topic": "/camera/aligned_depth_to_color/camera_info",
                    "camera_frame": "camera_color_optical_frame",
                    "vicon_parent_frame": "art0",
                    "vicon_marker_frame": "realsense_gt_1",
                    "marker_to_camera_extrinsic_applied_in_bound_geometry": True,
                    "pairing": "PENDING_MONOTONIC_RGB_TO_GEOMETRY_TIMESTAMP_AUDIT",
                },
                "geometry_timestamps_s": [str(value) for value in dlr_geometry_timestamps],
                "rgb_frame_inventory": [],
                "identity_state": "OPAQUE_TOPIC_FRAME_AND_PAYLOAD_SHA256_PENDING",
            },
        ],
        "transport_identity_authority": {
            "allowed": [
                "opaque compressed-range acquisition",
                "member/message timestamp and metadata parsing",
                "member/message CRC32 and SHA256",
                "camera topic/frame/encoding inventory",
                "RGB-to-geometry timestamp pairing receipt",
            ],
            "forbidden": [
                "pixel decode or visualization",
                "optical flow or RGB algorithm execution",
                "full-source fallback",
                "window/source replacement",
                "threshold or algorithm changes",
            ],
            "openloris_bound_solid_folder_index": 2,
            "openloris_bound_solid_folder_pack_bytes": folder_two["pack_bytes"],
            "openloris_max_remote_bytes": 3947000000,
            "dlr_max_remote_compressed_bytes": 1073741824,
            "stop_if_budget_insufficient": "SEGMENT_IDENTITY_NOT_EVALUABLE",
        },
        "final_lock_requirements": {
            "every_scientific_and_guard_rgb_frame_has_payload_sha256": True,
            "ordered_rgb_timestamp_inventory_complete": True,
            "camera_relation_and_encoding_closed": True,
            "pairing_is_monotonic_unique_and_within_frozen_tolerance": True,
            "independent_review_and_signature_required": True,
            "activation_separate_from_review": True,
        },
        "claim_ceiling": {
            "source_role_confounding": True,
            "allowed": "Per-segment descriptive RGB-versus-geometry mechanism alignment only.",
            "forbidden": [
                "positive/below role discrimination",
                "unconfounded role effect",
                "generalization or performance qualification",
                "host replay, Android, product, or safety authority",
            ],
        },
        "legal_terminals": [
            "IDENTITY_EXTRACTION_REVIEW_PASS",
            "IDENTITY_EXTRACTION_REVIEW_FAIL",
            "SEGMENT_IDENTITY_NOT_EVALUABLE",
            "INVALID_PREACCESS_LOCK",
        ],
        "execution_authority": {
            "opaque_identity_extraction": False,
            "rgb_algorithm": False,
            "android": False,
        },
        "mvsec_supplement": {
            "included": False,
            "reason": "Exact RGB capture semantics, encoding, per-frame identity, and synchronization are not closed.",
            "no_automatic_successor": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = args.output
    if not output.is_absolute():
        output = repo / output
    result = build(repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256(output),
                "status": result["status"],
                "openloris_frames": len(result["segments"][0]["frame_inventory"]),
                "dlr_geometry_frames": len(result["segments"][1]["geometry_timestamps_s"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
