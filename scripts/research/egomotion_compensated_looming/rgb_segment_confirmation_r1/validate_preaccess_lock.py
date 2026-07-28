from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any


OPEN_WINDOW = ["1560000043.537699", "1560000053.537699"]
DLR_WINDOW = ["1634201323.995618343", "1634201333.995618343"]
OPEN_WINDOW_ID = "corridor1-1:w004"
DLR_WINDOW_ID = "extreme_geometry/hexagon_01:w001"
PAIR_TOLERANCE = Decimal("0.005")
OPEN_BUDGET = 3_947_000_000
DLR_BUDGET = 1_073_741_824

EXPECTED_BINDINGS = {
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


def strict(values: list[Decimal]) -> bool:
    return all(right > left for left, right in zip(values, values[1:]))


def validate_bindings(repo: Path, candidate: dict[str, Any], errors: list[str]) -> None:
    rows = (
        candidate.get("source_bindings", [])
        + candidate.get("algorithm_bindings", [])
        + candidate.get("transport_bindings", [])
        + candidate.get("runtime_bindings", [])
    )
    actual_map = {row.get("path"): row.get("sha256") for row in rows}
    if actual_map != EXPECTED_BINDINGS:
        errors.append("FROZEN_BINDING_SET_OR_DIGEST")
    for relative, expected in EXPECTED_BINDINGS.items():
        path = repo / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"LIVE_BINDING_DRIFT:{relative}")


def validate_openloris(repo: Path, segment: dict[str, Any], errors: list[str]) -> None:
    if segment.get("window_id") != OPEN_WINDOW_ID:
        errors.append("OPEN_WINDOW_ID")
    if segment.get("half_open_window_s") != OPEN_WINDOW:
        errors.append("OPEN_WINDOW_BOUNDS")
    inventory = segment.get("frame_inventory", [])
    if len(inventory) != 300:
        errors.append("OPEN_FRAME_COUNT")
        return
    geometry = [Decimal(row["geometry_timestamp_s"]) for row in inventory]
    rgb = [Decimal(row["rgb_timestamp_s"]) for row in inventory]
    paths = [row["rgb_member_path"] for row in inventory]
    if len(set(geometry)) != 300 or not strict(geometry):
        errors.append("OPEN_GEOMETRY_ORDER_OR_UNIQUENESS")
    if len(set(rgb)) != 300 or not strict(rgb) or len(set(paths)) != 300:
        errors.append("OPEN_RGB_ORDER_OR_UNIQUENESS")
    deltas = [rgb_value - geometry_value for rgb_value, geometry_value in zip(rgb, geometry)]
    if any(
        Decimal(row["rgb_minus_geometry_delta_s"]) != delta
        for row, delta in zip(inventory, deltas)
    ):
        errors.append("OPEN_PAIR_DELTA_VALUE")
    if max(abs(value) for value in deltas) > PAIR_TOLERANCE:
        errors.append("OPEN_PAIR_TOLERANCE")
    if any(row.get("rgb_payload_sha256") is not None for row in inventory):
        errors.append("OPEN_PREACCESS_PAYLOAD_SHA_PRESENT")

    directory_path = (
        repo
        / "artifacts.local/evidence/rcle_source_authority_repair_r1/"
        "openloris_corridor1-1_7z_directory.json"
    )
    directory = json.loads(directory_path.read_text(encoding="utf-8"))
    members = []
    for item in directory["members"]:
        path = item["path"].replace("\\", "/")
        if "/color/" not in path or item.get("is_directory"):
            continue
        try:
            timestamp = Decimal(Path(path).stem)
        except Exception:
            continue
        members.append((timestamp, path, item))
    members.sort()
    selected = [
        row for row in members if Decimal(OPEN_WINDOW[0]) <= row[0] < Decimal(OPEN_WINDOW[1])
    ]
    if len(selected) != 300:
        errors.append("OPEN_DIRECTORY_SELECTED_COUNT")
        return
    expected_rows = {
        row[1]: (row[0], row[2]["uncompressed_bytes"], row[2]["crc32"])
        for row in selected
    }
    for row in inventory:
        expected = expected_rows.get(row["rgb_member_path"])
        if expected is None or expected != (
            Decimal(row["rgb_timestamp_s"]),
            row["rgb_member_uncompressed_bytes"],
            row["rgb_member_crc32"],
        ):
            errors.append("OPEN_MEMBER_IDENTITY")
            break
    first_index = members.index(selected[0])
    last_index = members.index(selected[-1])
    guard_before = segment.get("guard_before", {})
    guard_after = segment.get("guard_after", {})
    if guard_before.get("path") != members[first_index - 1][1]:
        errors.append("OPEN_GUARD_BEFORE")
    if guard_after.get("path") != members[last_index + 1][1]:
        errors.append("OPEN_GUARD_AFTER")
    if Decimal(guard_before.get("timestamp_s", "0")) >= rgb[0]:
        errors.append("OPEN_GUARD_BEFORE_BOUNDARY")
    if Decimal(guard_after.get("timestamp_s", "0")) <= rgb[-1]:
        errors.append("OPEN_GUARD_AFTER_BOUNDARY")


def validate_dlr(segment: dict[str, Any], errors: list[str]) -> None:
    if segment.get("window_id") != DLR_WINDOW_ID:
        errors.append("DLR_WINDOW_ID")
    if segment.get("half_open_window_s") != DLR_WINDOW:
        errors.append("DLR_WINDOW_BOUNDS")
    if segment.get("rgb_frame_inventory") != []:
        errors.append("DLR_UNREVIEWED_RGB_INVENTORY")
    if segment.get("required_rgb_guard_frame_count") != 2:
        errors.append("DLR_REQUIRED_GUARD")
    if segment.get("closed_rgb_guard_frame_count") != 0:
        errors.append("DLR_CLOSED_GUARD_PREACCESS")
    relation = segment.get("camera_relation", {})
    if relation.get("rgb_topic") != "PENDING_OPAQUE_TOPIC_INVENTORY":
        errors.append("DLR_RGB_TOPIC_PREMATURE")
    if relation.get("pairing") != "PENDING_MONOTONIC_RGB_TO_GEOMETRY_TIMESTAMP_AUDIT":
        errors.append("DLR_PAIRING_PREMATURE")


def validate(repo: Path, candidate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if candidate.get("protocol_id") != "RCLE_RGB_SEGMENT_CONFIRMATION_R1":
        errors.append("PROTOCOL")
    validate_bindings(repo, candidate, errors)
    segments = candidate.get("segments", [])
    if len(segments) != 2:
        errors.append("SEGMENT_COUNT")
        return sorted(set(errors))
    validate_openloris(repo, segments[0], errors)
    validate_dlr(segments[1], errors)
    authority = candidate.get("execution_authority", {})
    if authority != {
        "opaque_identity_extraction": False,
        "rgb_algorithm": False,
        "android": False,
    }:
        errors.append("PRE_REVIEW_AUTHORITY")
    transport = candidate.get("transport_identity_authority", {})
    if transport.get("openloris_max_remote_bytes") != OPEN_BUDGET:
        errors.append("OPEN_BUDGET")
    if transport.get("dlr_max_remote_compressed_bytes") != DLR_BUDGET:
        errors.append("DLR_BUDGET")
    if (
        transport.get("stop_if_budget_insufficient")
        != "SEGMENT_IDENTITY_NOT_EVALUABLE"
    ):
        errors.append("IDENTITY_BUDGET_TERMINAL")
    if "full-source fallback" not in transport.get("forbidden", []):
        errors.append("FULL_SOURCE_FALLBACK_NOT_FORBIDDEN")
    preflight = json.loads(
        (
            repo
            / "artifacts.local/evidence/rcle_source_authority_repair_r1/"
            "openloris_geometry_only_transport_preflight.json"
        ).read_text(encoding="utf-8")
    )
    capture = preflight["captures"]["corridor1-1"]
    folder = next(
        row
        for row in capture["geometry_solid_folders"]
        if row["folder_index"] == 2
    )
    window = next(
        row
        for row in capture["windows"]
        if row["window_id"] == "corridor1-1:W004"
    )
    if (
        transport.get("openloris_bound_solid_folder_index") != 2
        or transport.get("openloris_bound_solid_folder_pack_bytes")
        != folder["pack_bytes"]
        or window["solid_folder_indices"] != [2]
        or not folder["pack_bytes"] < OPEN_BUDGET
        or not OPEN_BUDGET < int(segments[0]["container"]["length"])
    ):
        errors.append("OPEN_SOLID_FOLDER_BUDGET_BINDING")
    if not DLR_BUDGET < int(segments[1]["zip_member"]["compressed_bytes"]):
        errors.append("DLR_BUDGET_NOT_BOUNDED")
    ceiling = candidate.get("claim_ceiling", {})
    if ceiling.get("source_role_confounding") is not True:
        errors.append("SOURCE_ROLE_CONFOUNDING_REMOVED")
    if "generalization or performance qualification" not in ceiling.get("forbidden", []):
        errors.append("GENERALIZATION_CEILING_REMOVED")
    return sorted(set(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    candidate_path = args.candidate if args.candidate.is_absolute() else repo / args.candidate
    output_path = args.output if args.output.is_absolute() else repo / args.output
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    errors = validate(repo, candidate)
    result = {
        "schema_version": "rcle.rgb_segment_confirmation.preaccess_validation.v2",
        "protocol_id": "RCLE_RGB_SEGMENT_CONFIRMATION_R1",
        "candidate": {
            "path": candidate_path.relative_to(repo).as_posix(),
            "sha256": sha256(candidate_path),
        },
        "decision": (
            "PREACCESS_LOCK_MACHINE_VALIDATION_PASS"
            if not errors
            else "PREACCESS_LOCK_MACHINE_VALIDATION_FAIL"
        ),
        "valid": not errors,
        "validator_scope": "LOCAL_FILES_AND_METADATA_ONLY",
        "validator_network_code_path": False,
        "validator_pixel_decode_code_path": False,
        "errors": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
