#!/usr/bin/env python3
"""Audit independent person-trajectory truth authority without reading candidate outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = (
    "blindassist_ustrf_independent_person_trajectory_truth_"
    "source_authority_and_admission_r0_config"
)
ACQUISITION_SCHEMA = (
    "blindassist_ustrf_independent_person_trajectory_truth_"
    "source_authority_and_admission_r0_acquisition"
)
LEDGER_SCHEMA = (
    "blindassist_ustrf_independent_person_trajectory_truth_"
    "source_authority_and_admission_r0_denominator_ledger"
)
RECEIPT_SCHEMA = (
    "blindassist_ustrf_independent_person_trajectory_truth_"
    "source_authority_and_admission_r0_receipt"
)
STAGE = "INDEPENDENT_PERSON_TRAJECTORY_TRUTH_SOURCE_AUTHORITY_AND_ADMISSION_R0"
GROUP_WIDTH = 17
POSITION_WIDTH = 3


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_number(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"non_numeric_value:{value!r}") from exc
    require(math.isfinite(parsed), "non_finite_value")
    return parsed


def missing_rigid_body(values: list[float]) -> bool:
    return all(value == 0.0 for value in values)


def distance_band(distance_m: float, bands: list[dict[str, Any]]) -> str:
    for band in bands:
        lower = float(band["lower_inclusive"])
        upper = band["upper_exclusive"]
        if distance_m >= lower and (upper is None or distance_m < float(upper)):
            return str(band["id"])
    raise RuntimeError(f"distance_outside_bands:{distance_m}")


def audit_payload(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        metadata_rows = [next(reader) for _ in range(10)]
        metadata = {row[0]: row[1:] for row in metadata_rows}
        header = next(reader)
        bodies = metadata["BODY_NAMES"]
        require(int(metadata["NO_OF_FRAMES"][0]) > 0, "declared_frame_count")
        require(int(metadata["NO_OF_BODIES"][0]) == len(bodies), "body_count_drift")
        require(int(metadata["FREQUENCY"][0]) == 100, "frequency_drift")
        require(metadata["DATA_INCLUDED"][0] == "6D", "data_role_drift")
        require(len(header) == 2 + GROUP_WIDTH * len(bodies), "column_count_drift")
        reference = config["canary"]["reference_track"]
        person_tracks = [f"Helmet_{value}" for value in range(2, 11)]
        require(reference in bodies, "reference_track_absent")
        require(all(track in bodies for track in person_tracks), "person_track_absent")
        indices = {body: 2 + bodies.index(body) * GROUP_WIDTH for body in bodies}

        bands = config["distance_bands_m"]
        band_counts = {str(band["id"]): 0 for band in bands}
        band_tracks: dict[str, set[str]] = {str(band["id"]): set() for band in bands}
        per_track = {
            track: {
                "valid_object_frames": 0,
                "missing_person_frames": 0,
                "missing_reference_frames": 0,
                "bands": {str(band["id"]): 0 for band in bands},
            }
            for track in person_tracks
        }
        frame_count = 0
        expected_frame = 1
        previous_time: float | None = None
        time_deltas: list[float] = []
        reference_valid_frames = 0
        total_person_opportunities = 0
        valid_object_frames = 0
        missing_person_frames = 0
        missing_reference_opportunities = 0
        residual_values: list[float] = []

        for row in reader:
            require(len(row) == len(header), f"row_width:{expected_frame}")
            frame = int(row[0])
            timestamp = parse_number(row[1])
            require(frame == expected_frame, f"frame_sequence:{frame}:{expected_frame}")
            if previous_time is not None:
                require(timestamp > previous_time, f"nonmonotonic_time:{frame}")
                time_deltas.append(timestamp - previous_time)
            previous_time = timestamp
            frame_count += 1
            expected_frame += 1

            reference_start = indices[reference]
            reference_values = [
                parse_number(value)
                for value in row[reference_start : reference_start + GROUP_WIDTH - 1]
            ]
            reference_missing = missing_rigid_body(reference_values)
            if not reference_missing:
                reference_valid_frames += 1
            reference_position = reference_values[:POSITION_WIDTH]

            for track in person_tracks:
                total_person_opportunities += 1
                start = indices[track]
                values = [
                    parse_number(value)
                    for value in row[start : start + GROUP_WIDTH - 1]
                ]
                person_missing = missing_rigid_body(values)
                if reference_missing:
                    missing_reference_opportunities += 1
                    per_track[track]["missing_reference_frames"] += 1
                    continue
                if person_missing:
                    missing_person_frames += 1
                    per_track[track]["missing_person_frames"] += 1
                    continue

                # THÖR paper reports millimetre-scale discretization, while the TSV
                # header itself omits a units declaration. Counts below are retained
                # as a conversion hypothesis and never used to pass metric authority.
                distance_m = (
                    math.sqrt(
                        sum(
                            (values[index] - reference_position[index]) ** 2
                            for index in range(POSITION_WIDTH)
                        )
                    )
                    / 1000.0
                )
                band = distance_band(distance_m, bands)
                band_counts[band] += 1
                band_tracks[band].add(track)
                per_track[track]["bands"][band] += 1
                per_track[track]["valid_object_frames"] += 1
                valid_object_frames += 1
                residual_values.append(values[6])

        declared_frames = int(metadata["NO_OF_FRAMES"][0])
        require(frame_count == declared_frames, "frame_count_drift")
        require(
            total_person_opportunities
            == valid_object_frames
            + missing_person_frames
            + missing_reference_opportunities,
            "denominator_conservation",
        )

    product_gates = config["denominator_gates"]
    product_core = {}
    for band in bands:
        if band["role"] != "product_core":
            continue
        band_id = str(band["id"])
        product_core[band_id] = {
            "valid_object_frames": band_counts[band_id],
            "distinct_person_track_ids": len(band_tracks[band_id]),
            "minimum_valid_object_frames": int(
                product_gates["minimum_valid_object_frames_per_product_core_band"]
            ),
            "minimum_distinct_person_track_ids": int(
                product_gates["minimum_distinct_person_track_ids_per_product_core_band"]
            ),
            "provisional_gate_met": (
                band_counts[band_id]
                >= int(product_gates["minimum_valid_object_frames_per_product_core_band"])
                and len(band_tracks[band_id])
                >= int(product_gates["minimum_distinct_person_track_ids_per_product_core_band"])
            ),
        }

    empty_bands = [band_id for band_id, count in band_counts.items() if count == 0]
    residual_sorted = sorted(residual_values)
    return {
        "schema": LEDGER_SCHEMA,
        "stage": STAGE,
        "source_id": config["canary"]["source_id"],
        "member": config["canary"]["member"],
        "window": config["canary"]["window"],
        "person_tracks": person_tracks,
        "reference_track": config["canary"]["reference_track"],
        "source_header": {
            "declared_frames": declared_frames,
            "observed_frames": frame_count,
            "frequency_hz": int(metadata["FREQUENCY"][0]),
            "body_names": bodies,
            "timestamp": metadata["TIME_STAMP"],
            "unit_declared_in_header": False,
        },
        "time": {
            "strictly_monotonic": True,
            "first_seconds": 0.0,
            "last_seconds": previous_time,
            "minimum_delta_seconds": min(time_deltas),
            "maximum_delta_seconds": max(time_deltas),
        },
        "denominators": {
            "person_frame_opportunities": total_person_opportunities,
            "valid_object_frames": valid_object_frames,
            "missing_person_frames": missing_person_frames,
            "missing_reference_opportunities": missing_reference_opportunities,
            "reference_valid_frames": reference_valid_frames,
            "conservation_met": True,
            "distance_bands_provisional_mm_conversion": band_counts,
            "distinct_tracks_per_band": {
                band_id: len(tracks) for band_id, tracks in band_tracks.items()
            },
            "empty_bands": empty_bands,
            "product_core_provisional_gates": product_core,
        },
        "per_track": per_track,
        "source_residual_raw_units": {
            "count": len(residual_sorted),
            "median": residual_sorted[len(residual_sorted) // 2],
            "maximum": max(residual_sorted),
        },
        "conversion_authority": {
            "hypothesis": "raw Qualisys translations divided by 1000",
            "basis": "paper reports 1 mm spatial discretization and 2 mm average residual",
            "payload_header_declares_units": False,
            "metric_band_counts_admitted": False,
        },
        "candidate_outputs_read": False,
    }


def audit(repo: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config.get("schema") == CONFIG_SCHEMA, "config_identity")
    require(config.get("stage") == STAGE, "stage_identity")
    require(
        config.get("status") == "frozen_before_candidate_output_read",
        "candidate_blind_freeze_status",
    )
    require(config["canary"]["candidate_outputs_visible"] is False, "candidate_visibility")
    require(
        config["source_schema_amendment_before_candidate_output_read"][
            "candidate_outputs_read"
        ]
        is False,
        "amendment_candidate_visibility",
    )
    for parent in config["parents"]:
        path = repo / parent["path"]
        require(path.is_file(), f"missing_parent:{path}")
        require(sha256_file(path) == parent["sha256"], f"parent_sha_drift:{path}")

    acquisition_path = repo / config["outputs"]["acquisition"]
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    require(acquisition.get("schema") == ACQUISITION_SCHEMA, "acquisition_identity")
    payload_path = repo / acquisition["payload"]["path"]
    require(payload_path.is_file(), "payload_missing")
    require(payload_path.stat().st_size == acquisition["payload"]["bytes"], "payload_size")
    require(sha256_file(payload_path) == acquisition["payload"]["sha256"], "payload_sha")

    revel_binding = config["source_inventory"][1]["binding"]
    revel_path = repo / revel_binding["path"]
    require(revel_path.is_file(), "revel_receipt_missing")
    require(sha256_file(revel_path) == revel_binding["sha256"], "revel_receipt_sha")
    revel = json.loads(revel_path.read_text(encoding="utf-8"))
    ledger = audit_payload(payload_path, config)

    provisional_product_denominators = all(
        row["provisional_gate_met"]
        for row in ledger["denominators"]["product_core_provisional_gates"].values()
    )
    gates = {
        "independent_measurement_chain": {
            "met": True,
            "evidence": "Qualisys optical mocap is independent of JRDB annotations, PCD point-in-box queries and derived centroids.",
        },
        "stable_person_track_ids": {
            "met": False,
            "evidence": "Helmet_2..Helmet_10 are deterministic rigid-body columns, but the source reports manual ID-switch cleaning and lost-track recovery without a per-frame intervention mask or raw uncleaned authority.",
        },
        "metric_3d_position": {
            "met": False,
            "evidence": "The paper reports millimetre-scale mocap, but this frozen TSV header does not declare units; provisional metre conversion cannot grant authority.",
        },
        "shared_time_binding": {
            "met": False,
            "evidence": "QTM rows are monotonic at 100 Hz and the paper reports shared NTP, but no measured Velodyne-to-QTM offset, jitter or synchronization error bound is present.",
        },
        "coordinate_frame_and_transform_semantics": {
            "met": False,
            "evidence": "Helmet, Velodyne and Citi_1 rigid bodies share the mocap world, but the public authority does not close the Velodyne marker-rigid-body to LiDAR measurement-frame lever arm, axes, handedness or extrinsic error.",
        },
        "quantitative_error_or_calibration_statement": {
            "met": False,
            "evidence": "The source reports 2 mm average mocap residual, but provides no admitted person-reference recovery error or sensor extrinsic/synchronization uncertainty needed for paired truth.",
        },
        "product_core_distance_denominators": {
            "met": False,
            "evidence": (
                "Provisional 5-20 m counts meet the numeric floors under a /1000 "
                "hypothesis, but the payload unit and paired sensor transform are not "
                "authoritative; 40m+ remains empty."
            ),
            "provisional_numeric_floor_met": provisional_product_denominators,
        },
        "candidate_blind_freeze": {
            "met": True,
            "evidence": "Source, member, entire-file window, tracks, bands, denominators and missing policy were frozen with candidate outputs invisible.",
        },
    }
    all_gates_met = all(row["met"] for row in gates.values())
    terminal = (
        "INDEPENDENT_PERSON_TRAJECTORY_TRUTH_SOURCE_ADMITTED"
        if all_gates_met
        else "INDEPENDENT_PERSON_TRAJECTORY_TRUTH_AUTHORITY_ABSENT"
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "stage": STAGE,
        "status": "AUDIT_COMPLETE",
        "terminal_state": terminal,
        "config_sha256": sha256_file(config_path),
        "acquisition_sha256": sha256_file(acquisition_path),
        "ledger_sha256": canonical_sha256(ledger),
        "source_decisions": {
            "jrdb_annotation_derived_person_geometry": "REJECTED_CIRCULAR_TRUTH",
            "revel_dynamic_vicon": {
                "decision": "SOURCE_CANDIDATE_LIMITED_NOT_ADMITTED",
                "existing_external_metric_truth_claim": revel["admission"][
                    "external_metric_person_sensor_trajectory_truth_admitted"
                ],
                "maximum_ranges_m": {
                    track: row["relative_to_sensor"]["sensor_local_range_m"]["max"]
                    for track, row in revel["helmet_people"].items()
                },
                "r0_gap": "No R0 candidate-blind freeze, independent validator, quantitative Vicon/extrinsic uncertainty or 10-20/20-40/40-plus denominator closure.",
            },
            "thor_people_tracks_v1": "AUDITED_NOT_ADMITTED",
        },
        "admission_gates": gates,
        "all_admission_gates_met": all_gates_met,
        "authority_scope": {
            "independent_person_trajectory_truth_admitted": False,
            "provisional_source_profile_retained": True,
            "distance_bands_closed": [
                "0-5",
                "5-10",
                "10-20",
                "20-40",
                "40-plus",
            ],
            "product_focus": "5-20m",
            "capability_boundary": "40m+ retained with zero admitted denominator",
            **config["authority_limits"],
        },
        "candidate_outputs_read": False,
        "algorithm_comparison_performed": False,
        "manual_trajectory_selection_performed": False,
    }
    require(
        terminal in config["terminal_states"],
        "terminal_not_registered",
    )
    return ledger, receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ledger, receipt = audit(repo, config_path)
    ledger_path = repo / config["outputs"]["ledger"]
    receipt_path = repo / config["outputs"]["receipt"]
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt["ledger_file_sha256"] = sha256_file(ledger_path)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "terminal_state": receipt["terminal_state"],
                "ledger": str(ledger_path),
                "receipt": str(receipt_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
