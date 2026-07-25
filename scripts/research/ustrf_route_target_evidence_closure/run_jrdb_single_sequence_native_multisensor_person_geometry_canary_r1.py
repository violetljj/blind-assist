#!/usr/bin/env python3
"""Re-evaluate JRDB person geometry with claim-specific elastic eligibility."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0 import (
    apply_transform,
    canonical_bytes,
    sha256_file,
    transform_from_q,
    write_canonical,
)

STAGE = "JRDB_SINGLE_SEQUENCE_NATIVE_MULTISENSOR_PERSON_GEOMETRY_CANARY_R1"
CONFIG_SCHEMA = "blindassist_ustrf_jrdb_single_sequence_native_multisensor_person_geometry_canary_r1_config"
LEDGER_SCHEMA = "blindassist_ustrf_jrdb_single_sequence_native_multisensor_person_geometry_canary_r1_eligibility_ledger"
RECEIPT_SCHEMA = "blindassist_ustrf_jrdb_single_sequence_native_multisensor_person_geometry_canary_r1_receipt"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def coverage_band(value: float) -> str:
    if value >= 0.95:
        return "HIGH_COVERAGE"
    if value >= 0.8:
        return "MODERATE_COVERAGE"
    return "LOW_COVERAGE"


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "median": None, "p95": None, "maximum": None}
    ordered = sorted(values)

    def pick(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {"minimum": ordered[0], "median": pick(0.5), "p95": pick(0.95), "maximum": ordered[-1]}


def denominator(expected: int, eligible: int, abstained: int, invalid: int) -> dict[str, Any]:
    require(expected == eligible + abstained + invalid, "denominator_conservation")
    coverage = eligible / expected if expected else None
    return {
        "expected": expected,
        "eligible": eligible,
        "abstained": abstained,
        "invalid": invalid,
        "coverage": coverage,
        "coverage_band": coverage_band(coverage) if coverage is not None else None,
    }


def index_unique(objects: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    by_id: dict[str, dict[str, Any]] = {}
    ambiguous: set[str] = set()
    for item in objects:
        label_id = item.get("label_id")
        require(isinstance(label_id, str), "label_id_missing")
        if label_id in by_id:
            ambiguous.add(label_id)
        else:
            by_id[label_id] = item
    for label_id in ambiguous:
        by_id.pop(label_id, None)
    return by_id, ambiguous


def compact_ranges(values: list[int]) -> list[list[int]]:
    if not values:
        return []
    ordered = sorted(set(values))
    ranges: list[list[int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append([start, previous])
        start = previous = value
    ranges.append([start, previous])
    return ranges


def build_ledger(
    config: dict[str, Any],
    packet: dict[str, Any],
    labels_2d: dict[str, Any],
    labels_3d: dict[str, Any],
) -> dict[str, Any]:
    require(config["schema"] == CONFIG_SCHEMA and config["stage"] == STAGE, "config_identity")
    require(packet["status"] == "IMMUTABLE_OBSERVATION_PACKET", "parent_packet_status")
    require(packet["sequence"] == config["canary"]["sequence"], "sequence_drift")
    require(len(packet["frames"]) == int(config["canary"]["frame_count"]), "frame_count_drift")
    base_from_rgb = packet["calibration"]["base_link_from_logical_rgb360"]
    observations: list[dict[str, Any]] = []
    defects: list[dict[str, Any]] = []
    three_d_expected = 0
    two_d_expected = 0
    two_d_joined = 0
    two_d_only: dict[str, list[int]] = defaultdict(list)
    three_d_only: dict[str, list[int]] = defaultdict(list)
    ambiguous_ids: dict[str, list[int]] = defaultdict(list)
    observed_frames: set[int] = set()
    direct_3d_observations = 0
    interpolated_3d_observations = 0

    for frame in packet["frames"]:
        frame_index = int(frame["frame_index"])
        stem = frame["frame_stem"]
        objects_2d = labels_2d["labels"][f"{stem}.jpg"]
        objects_3d = labels_3d["labels"][f"{stem}.pcd"]
        by_2d, ambiguous_2d = index_unique(objects_2d)
        by_3d, ambiguous_3d = index_unique(objects_3d)
        three_d_expected += len(objects_3d)
        two_d_expected += len(objects_2d)
        ambiguous = sorted(ambiguous_2d | ambiguous_3d)
        for label_id in ambiguous:
            ambiguous_ids[label_id].append(frame_index)
            defects.append(
                {
                    "defect_id": f"ambiguous:{frame_index}:{label_id}",
                    "defect_class": "localized_ambiguity",
                    "scope_type": "observation_or_object",
                    "scope_ids": [packet["sequence"], stem, label_id],
                    "affected_modalities": [
                        role
                        for role, values in (("2d_label", ambiguous_2d), ("3d_label", ambiguous_3d))
                        if label_id in values
                    ],
                    "affected_claims": [
                        "robot_relative_3d_geometry",
                        "source_annotation_derived_3d_motion",
                        "cross_modal_2d_3d_identity",
                    ],
                    "reason_code": "duplicate_or_ambiguous_label_id",
                    "localized": True,
                    "denominator_impact": 1,
                    "evidence_refs": [f"frame:{stem}"],
                }
            )
        pose = frame["pose"]
        odom_from_base = transform_from_q(pose["translation"], pose["quaternion_xyzw"])
        for label_id, item in sorted(by_3d.items()):
            box = item["box"]
            center_rgb = [float(box[key]) for key in ("cx", "cy", "cz")]
            finite = all(math.isfinite(value) for value in center_rgb)
            if not finite:
                defects.append(
                    {
                        "defect_id": f"nonfinite:{frame_index}:{label_id}",
                        "defect_class": "structural_integrity",
                        "scope_type": "observation_or_object",
                        "scope_ids": [packet["sequence"], stem, label_id],
                        "affected_modalities": ["3d_label"],
                        "affected_claims": [
                            "robot_relative_3d_geometry",
                            "source_annotation_derived_3d_motion",
                        ],
                        "reason_code": "nonfinite_3d_center",
                        "localized": True,
                        "denominator_impact": 1,
                        "evidence_refs": [f"frame:{stem}"],
                    }
                )
                continue
            observed_frames.add(frame_index)
            center_base = apply_transform(base_from_rgb, center_rgb)
            center_odom = apply_transform(odom_from_base, center_base)
            source_interpolated = item.get("attributes", {}).get("interpolated") is True
            if source_interpolated:
                interpolated_3d_observations += 1
            else:
                direct_3d_observations += 1
            cross_modal = label_id in by_2d
            if cross_modal:
                two_d_joined += 1
            else:
                three_d_only[label_id].append(frame_index)
                defects.append(
                    {
                        "defect_id": f"missing2d:{frame_index}:{label_id}",
                        "defect_class": "missing_modality",
                        "scope_type": "observation_or_object",
                        "scope_ids": [packet["sequence"], stem, label_id],
                        "affected_modalities": ["2d_label"],
                        "affected_claims": ["cross_modal_2d_3d_identity"],
                        "reason_code": "unknown_missing_2d",
                        "localized": True,
                        "denominator_impact": 1,
                        "evidence_refs": [f"frame:{stem}", f"3d_label:{label_id}"],
                    }
                )
            observations.append(
                {
                    "frame_index": frame_index,
                    "frame_stem": stem,
                    "timestamp_ns": frame["time"]["upper_pointcloud_timestamp_ns"],
                    "label_id": label_id,
                    "source_interpolated": source_interpolated,
                    "direct_observation": not source_interpolated,
                    "cross_modal_2d_join": cross_modal,
                    "center_logical_rgb360_m": center_rgb,
                    "center_base_link_m": center_base,
                    "center_odom_m": center_odom,
                }
            )
        for label_id in sorted(set(by_2d) - set(by_3d)):
            two_d_only[label_id].append(frame_index)
            defects.append(
                {
                    "defect_id": f"missing3d:{frame_index}:{label_id}",
                    "defect_class": "missing_modality",
                    "scope_type": "observation_or_object",
                    "scope_ids": [packet["sequence"], stem, label_id],
                    "affected_modalities": ["3d_label"],
                    "affected_claims": [
                        "robot_relative_3d_geometry",
                        "source_annotation_derived_3d_motion",
                        "cross_modal_2d_3d_identity",
                    ],
                    "reason_code": "unknown_missing_3d",
                    "localized": True,
                    "denominator_impact": 1,
                    "evidence_refs": [f"frame:{stem}", f"2d_label:{label_id}"],
                }
            )

    tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        tracks[observation["label_id"]].append(observation)
    motion_pairs: list[dict[str, Any]] = []
    expected_motion_pairs = 0
    abstained_motion_pairs = 0
    invalid_motion_pairs = 0
    motion_tracks: set[str] = set()
    maximum_gap = float(
        config["claims"]["source_annotation_derived_3d_motion"]["maximum_pair_gap_seconds"]
    )
    for label_id, values in sorted(tracks.items()):
        values.sort(key=lambda row: row["frame_index"])
        for left, right in zip(values, values[1:]):
            if right["frame_index"] != left["frame_index"] + 1:
                continue
            expected_motion_pairs += 1
            gap = (right["timestamp_ns"] - left["timestamp_ns"]) / 1e9
            if not (0 < gap <= maximum_gap):
                abstained_motion_pairs += 1
                defects.append(
                    {
                        "defect_id": f"motion-gap:{left['frame_index']}:{right['frame_index']}:{label_id}",
                        "defect_class": "quality_bound",
                        "scope_type": "adjacent_pair",
                        "scope_ids": [left["frame_stem"], right["frame_stem"], label_id],
                        "affected_modalities": ["3d_label", "source_time"],
                        "affected_claims": ["source_annotation_derived_3d_motion"],
                        "reason_code": "motion_gap_exceeds_bound",
                        "localized": True,
                        "denominator_impact": 1,
                        "evidence_refs": [
                            f"left_timestamp_ns:{left['timestamp_ns']}",
                            f"right_timestamp_ns:{right['timestamp_ns']}",
                        ],
                    }
                )
                continue
            odom_velocity = [
                (b - a) / gap for a, b in zip(left["center_odom_m"], right["center_odom_m"])
            ]
            relative_velocity = [
                (b - a) / gap
                for a, b in zip(left["center_base_link_m"], right["center_base_link_m"])
            ]
            if not all(math.isfinite(value) for value in odom_velocity + relative_velocity):
                invalid_motion_pairs += 1
                continue
            motion_tracks.add(label_id)
            motion_pairs.append(
                {
                    "label_id": label_id,
                    "left_frame": left["frame_index"],
                    "right_frame": right["frame_index"],
                    "gap_seconds": gap,
                    "source_interpolated": left["source_interpolated"]
                    or right["source_interpolated"],
                    "direct_observation_pair": left["direct_observation"]
                    and right["direct_observation"],
                    "source_annotation_odom_velocity_mps": odom_velocity,
                    "source_annotation_odom_speed_mps": math.sqrt(
                        sum(value * value for value in odom_velocity)
                    ),
                    "robot_relative_velocity_mps": relative_velocity,
                    "robot_relative_speed_mps": math.sqrt(
                        sum(value * value for value in relative_velocity)
                    ),
                }
            )
    require(
        expected_motion_pairs
        == len(motion_pairs) + abstained_motion_pairs + invalid_motion_pairs,
        "motion_denominator_conservation",
    )
    cross_modal_3d_eligible = sum(1 for row in observations if row["cross_modal_2d_join"])
    cross_modal_2d_abstained = sum(len(values) for values in two_d_only.values())
    cross_modal_3d_abstained = sum(len(values) for values in three_d_only.values())
    three_d_invalid = three_d_expected - len(observations)
    two_d_invalid = two_d_expected - two_d_joined - cross_modal_2d_abstained
    robot_denominator = denominator(
        three_d_expected,
        len(observations),
        0,
        three_d_invalid,
    )
    motion_denominator = denominator(
        expected_motion_pairs,
        len(motion_pairs),
        abstained_motion_pairs,
        invalid_motion_pairs,
    )
    cross_modal_from_3d = denominator(
        three_d_expected,
        cross_modal_3d_eligible,
        cross_modal_3d_abstained,
        three_d_invalid,
    )
    cross_modal_from_2d = denominator(
        two_d_expected,
        two_d_joined,
        cross_modal_2d_abstained,
        two_d_invalid,
    )
    geometry_floor = (
        len(observed_frames)
        >= int(config["claims"]["robot_relative_3d_geometry"]["minimum_observed_frames"])
        and len(observations)
        >= int(config["claims"]["robot_relative_3d_geometry"]["minimum_object_observations"])
    )
    motion_floor = (
        len(observed_frames)
        >= int(
            config["claims"]["source_annotation_derived_3d_motion"]["minimum_observed_frames"]
        )
        and len(motion_pairs)
        >= int(
            config["claims"]["source_annotation_derived_3d_motion"][
                "minimum_valid_adjacent_pairs"
            ]
        )
        and len(motion_tracks)
        >= int(
            config["claims"]["source_annotation_derived_3d_motion"]["minimum_motion_tracks"]
        )
    )
    return {
        "schema": LEDGER_SCHEMA,
        "stage": STAGE,
        "status": "ELIGIBILITY_LEDGER_COMPLETE",
        "sequence": packet["sequence"],
        "window": packet["window"],
        "parent_packet_sha256": config["parent_r0"]["observation_packet"]["sha256"],
        "artifact_integrity": "VALID",
        "observations": observations,
        "motion_pairs": motion_pairs,
        "defects": defects,
        "denominators": {
            "robot_relative_3d_geometry": robot_denominator,
            "source_annotation_derived_3d_motion": motion_denominator,
            "cross_modal_from_3d": cross_modal_from_3d,
            "cross_modal_from_2d": cross_modal_from_2d,
        },
        "support": {
            "observed_frames": len(observed_frames),
            "unique_3d_tracks": len(tracks),
            "motion_track_count": len(motion_tracks),
            "geometry_floor_pass": geometry_floor,
            "motion_floor_pass": motion_floor,
            "direct_3d_observations": direct_3d_observations,
            "source_interpolated_3d_observations": interpolated_3d_observations,
            "direct_motion_pairs": sum(
                pair["direct_observation_pair"] for pair in motion_pairs
            ),
            "source_interpolated_motion_pairs": sum(
                pair["source_interpolated"] for pair in motion_pairs
            ),
        },
        "missingness_clusters": {
            "3d_only": {
                label_id: {
                    "count": len(frames),
                    "frame_ranges": compact_ranges(frames),
                }
                for label_id, frames in sorted(three_d_only.items())
            },
            "2d_only": {
                label_id: {
                    "count": len(frames),
                    "frame_ranges": compact_ranges(frames),
                }
                for label_id, frames in sorted(two_d_only.items())
            },
            "ambiguous": {
                label_id: {
                    "count": len(frames),
                    "frame_ranges": compact_ranges(frames),
                }
                for label_id, frames in sorted(ambiguous_ids.items())
            },
        },
        "authority": config["authority"],
    }


def claim_record(
    claim_id: str,
    config: dict[str, Any],
    denominator_row: dict[str, Any],
    status: str,
    defect_class: str,
    missingness_mechanism: str,
    cluster_distribution: dict[str, Any],
    bias_risk: str,
    claim_scope: str,
) -> dict[str, Any]:
    claim = config["claims"][claim_id]
    return {
        "claim_id": claim_id,
        "status": status,
        "required_roles": claim["required_roles"],
        "optional_roles": claim.get("optional_roles", []),
        "unit_of_analysis": claim["unit_of_analysis"],
        "defect_class": defect_class,
        "missingness_mechanism": missingness_mechanism,
        "union_denominator": denominator_row["expected"],
        "eligible_denominator": denominator_row["eligible"],
        "abstained_denominator": denominator_row["abstained"],
        "invalid_denominator": denominator_row["invalid"],
        "coverage": denominator_row["coverage"],
        "coverage_band": denominator_row["coverage_band"],
        "cluster_distribution": cluster_distribution,
        "propagation_evidence": "localized; no global propagation",
        "bias_risk": bias_risk,
        "disposition": (
            "ADMIT_COMPLETE"
            if status == "AVAILABLE_COMPLETE"
            else (
                "ADMIT_WITH_ABSTENTION"
                if status == "AVAILABLE_WITH_DEGRADATION"
                else "NOT_EVALUABLE_FOR_CLAIM"
            )
        ),
        "maximum_claim_scope": claim_scope,
        "authority_granted": ["DIAGNOSTIC"],
        "authority_closed": [
            "SELECTION",
            "ROUTE_RISK",
            "EVENT_LIFECYCLE",
            "ALERT_LOGIC",
            "ANDROID",
            "HUMAN_SAFETY",
            "PRODUCTION",
        ],
    }


def build_receipt(config: dict[str, Any], ledger: dict[str, Any], ledger_sha256: str) -> dict[str, Any]:
    support = ledger["support"]
    denominators = ledger["denominators"]
    geometry_status = (
        "AVAILABLE_COMPLETE"
        if support["geometry_floor_pass"]
        and denominators["robot_relative_3d_geometry"]["abstained"] == 0
        and denominators["robot_relative_3d_geometry"]["invalid"] == 0
        else "NOT_EVALUABLE_INSUFFICIENT_SUPPORT"
    )
    motion_status = (
        "AVAILABLE_COMPLETE"
        if support["motion_floor_pass"]
        and denominators["source_annotation_derived_3d_motion"]["abstained"] == 0
        and denominators["source_annotation_derived_3d_motion"]["invalid"] == 0
        else (
            "AVAILABLE_WITH_DEGRADATION"
            if support["motion_floor_pass"]
            else "NOT_EVALUABLE_INSUFFICIENT_SUPPORT"
        )
    )
    cross_status = (
        "AVAILABLE_COMPLETE"
        if denominators["cross_modal_from_3d"]["coverage"] == 1.0
        and denominators["cross_modal_from_2d"]["coverage"] == 1.0
        else "AVAILABLE_WITH_DEGRADATION"
    )
    claims = [
        claim_record(
            "robot_relative_3d_geometry",
            config,
            denominators["robot_relative_3d_geometry"],
            geometry_status,
            "normal_observation_missingness",
            "none_for_3d_native_claim",
            {},
            "all source 3d labels are interpolated annotations; geometry is not direct human measurement",
            "this sequence/window source-annotation-derived robot-relative geometry computability",
        ),
        claim_record(
            "source_annotation_derived_3d_motion",
            config,
            denominators["source_annotation_derived_3d_motion"],
            motion_status,
            "normal_observation_missingness",
            "pair-level gap or invalid values only",
            {},
            "all source 3d labels are interpolated annotations; velocity accuracy is unvalidated",
            "this sequence/window source-annotation-derived 3d motion computability",
        ),
        claim_record(
            "cross_modal_2d_3d_identity",
            config,
            denominators["cross_modal_from_3d"],
            cross_status,
            "normal_observation_missingness",
            "unknown_missing_2d_or_3d",
            ledger["missingness_clusters"],
            "missingness is concentrated in named track runs and may be visibility-related, but source cause is not proven",
            "this sequence/window bidirectional 2d/3d identity coverage only",
        ),
    ]
    available_geometry = geometry_status.startswith("AVAILABLE")
    available_motion = motion_status.startswith("AVAILABLE")
    terminal = (
        "ANNOTATION_DERIVED_PERSON_GEOMETRY_AVAILABLE_WITH_ABSTENTION"
        if available_geometry and available_motion and cross_status == "AVAILABLE_WITH_DEGRADATION"
        else (
            "ANNOTATION_DERIVED_PERSON_GEOMETRY_AVAILABLE_COMPLETE"
            if available_geometry and available_motion
            else "NOT_EVALUABLE_3D_GEOMETRY_SUPPORT_INSUFFICIENT"
        )
    )
    odom_speeds = [
        pair["source_annotation_odom_speed_mps"] for pair in ledger["motion_pairs"]
    ]
    relative_speeds = [
        pair["robot_relative_speed_mps"] for pair in ledger["motion_pairs"]
    ]
    ranges = [
        math.sqrt(sum(value * value for value in observation["center_base_link_m"]))
        for observation in ledger["observations"]
    ]
    return {
        "schema": RECEIPT_SCHEMA,
        "stage": STAGE,
        "status": "VALID_WITH_PARTIAL_OR_DEGRADED_CLAIMS",
        "terminal_state": terminal,
        "artifact_integrity": "VALID",
        "authority_ceiling": config["authority"]["ceiling"],
        "config_sha256": None,
        "eligibility_ledger_sha256": ledger_sha256,
        "parent_r0": {
            "terminal_preserved": True,
            "terminal_state": config["parent_r0"]["receipt"]["required_terminal"],
            "receipt_sha256": config["parent_r0"]["receipt"]["sha256"],
        },
        "claims": claims,
        "support": support,
        "denominators": denominators,
        "quality": {
            "robot_relative_range_meters": quantiles(ranges),
            "source_annotation_odom_speed_mps": quantiles(odom_speeds),
            "robot_relative_speed_mps": quantiles(relative_speeds),
        },
        "missingness_clusters": ledger["missingness_clusters"],
        "claim_language": config["source_annotation_provenance"]["maximum_claim_language"],
        "prohibited_claims": [
            "directly_measured_human_motion",
            "motion_accuracy",
            "route_risk",
            "event_lifecycle",
            "alert_logic",
            "android",
            "human_safety",
            "production",
        ],
        "authority": config["authority"],
    }


def load_inputs(repo: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config["schema"] == CONFIG_SCHEMA and config["stage"] == STAGE, "config_identity")
    for group in ("elastic_standard", "parent_r0"):
        for role, binding in config[group].items():
            if not isinstance(binding, dict) or "path" not in binding:
                continue
            path = repo / binding["path"]
            require(path.is_file(), f"missing_binding:{group}:{role}")
            require(sha256_file(path) == binding["sha256"], f"binding_drift:{group}:{role}")
    standard_validation = json.loads(
        (repo / config["elastic_standard"]["validation"]["path"]).read_text(encoding="utf-8")
    )
    parent_receipt = json.loads(
        (repo / config["parent_r0"]["receipt"]["path"]).read_text(encoding="utf-8")
    )
    parent_validation = json.loads(
        (repo / config["parent_r0"]["validation"]["path"]).read_text(encoding="utf-8")
    )
    require(
        standard_validation["status"] == config["elastic_standard"]["validation"]["required_status"],
        "standard_validation_status",
    )
    require(
        parent_receipt["terminal_state"] == config["parent_r0"]["receipt"]["required_terminal"],
        "parent_terminal_drift",
    )
    require(
        parent_validation["status"] == config["parent_r0"]["validation"]["required_status"],
        "parent_validation_status",
    )
    packet = json.loads(
        (repo / config["parent_r0"]["observation_packet"]["path"]).read_text(encoding="utf-8")
    )
    require(
        packet["status"] == config["parent_r0"]["observation_packet"]["required_status"],
        "packet_status",
    )
    file_rows = {row["member"]: row for row in packet["raw_payload"]["files"]}
    sequence = config["canary"]["sequence"]
    label_2d_member = f"labels/labels_2d_stitched/{sequence}.json"
    label_3d_member = f"labels/labels_3d/{sequence}.json"
    require(label_2d_member in file_rows and label_3d_member in file_rows, "label_payload_binding")
    label_docs: list[dict[str, Any]] = []
    for member in (label_2d_member, label_3d_member):
        row = file_rows[member]
        path = repo / row["path"]
        require(path.is_file() and sha256_file(path) == row["sha256"], f"label_payload_drift:{member}")
        label_docs.append(json.loads(path.read_text(encoding="utf-8")))
    return config, packet, label_docs[0], label_docs[1]


def run(repo: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config, packet, labels_2d, labels_3d = load_inputs(repo, config_path)
    ledger = build_ledger(config, packet, labels_2d, labels_3d)
    ledger_sha = sha256_bytes(canonical_bytes(ledger))
    receipt = build_receipt(config, ledger, ledger_sha)
    receipt["config_sha256"] = sha256_file(config_path)
    return ledger, receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ledger, receipt = run(repo, config_path)
    ledger_path = repo / config["outputs"]["eligibility_ledger"]
    receipt_path = repo / config["outputs"]["receipt"]
    write_canonical(ledger_path, ledger)
    write_canonical(receipt_path, receipt)
    print(
        json.dumps(
            {
                "ledger": ledger_path.as_posix(),
                "receipt": receipt_path.as_posix(),
                "status": receipt["status"],
                "terminal_state": receipt["terminal_state"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
