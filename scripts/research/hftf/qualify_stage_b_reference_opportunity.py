#!/usr/bin/env python3
"""Qualify one HFTF R3.1 source using the dense reference only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_swept_envelope_label_mechanics import (
    _ground_support,
    _swept_prism_counts,
    _swept_prism_probes_world,
)
from run_geometry_teacher_canary import (
    _anchor_basis,
    _known_field,
    _obstacle_points_world,
    _sha256,
    _theta_edges,
    _validate_authority,
)
from verify_sanpo_pose_geometry_authority import _load_json, _load_jsonl


SCHEMA = "blindassist_hftf_stage_b_reference_opportunity_source_result_r3_1"
PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_b_reference_only_opportunity_qualification_r3_1"
)
LEDGER_SCHEMA = "blindassist_hftf_r3_1_source_pool_burn_ledger"
QUALIFIED = "R3_1_SOURCE_REFERENCE_OPPORTUNITY_QUALIFIED"
REJECTED = "R3_1_SOURCE_REFERENCE_OPPORTUNITY_REJECTED"
LAYERS = ("foot", "body", "head")
EXPECTED_MECHANICS_SHA256 = (
    "a69d25d77f1e2b72f407980f005c758b965517fd032562a009f91746ea1e0e6a"
)


def _validate_burn_ledger(
    ledger: dict[str, Any], ledger_path: Path
) -> set[str]:
    if (
        ledger.get("schema") != LEDGER_SCHEMA
        or ledger.get("status") != "FROZEN_BEFORE_R3_1_QUALIFICATION"
    ):
        raise ValueError("R3.1 burn ledger is not frozen")
    burned = [str(value) for value in ledger["burned_session_ids"]]
    if (
        len(burned) != int(ledger["burned_session_count"])
        or len(burned) != len(set(burned))
    ):
        raise ValueError("R3.1 burn ledger count or uniqueness mismatch")
    for parent in ledger["parent_protocols"]:
        path = ledger_path.parent / str(parent["path"])
        if _sha256(path) != parent["sha256"]:
            raise ValueError(f"Burn-ledger parent hash mismatch: {path}")
    return set(burned)


def _reference_decision(
    known_coverage: dict[str, float],
    positive_by_height: dict[str, int],
    negative_by_height: dict[str, int],
    sensitivity_opportunity: dict[str, dict[str, int]],
    ground_known_coverage: float,
    ground_risk_cells: int,
    ground_risk_frames: int,
    ground_risk_directions: int,
    protocol: dict[str, Any],
) -> dict[str, bool]:
    obstacle = protocol["obstacle_opportunity_qualification"]
    ground = protocol["ground_opportunity_qualification"]
    checks = {
        "obstacle_known_coverage_each_height": all(
            known_coverage[name]
            >= float(obstacle["minimum_known_coverage_each_height"])
            for name in LAYERS
        ),
        "obstacle_primary_positive_each_height": all(
            positive_by_height[name]
            >= int(obstacle["minimum_positive_known_cells_each_height"])
            for name in LAYERS
        ),
        "obstacle_primary_negative_each_height": all(
            negative_by_height[name]
            >= int(obstacle["minimum_negative_known_cells_each_height"])
            for name in LAYERS
        ),
        "obstacle_all_sensitivity_thresholds_have_micro_opportunity": all(
            item["positive"] > 0 and item["negative"] > 0
            for item in sensitivity_opportunity.values()
        ),
        "ground_known_coverage": ground_known_coverage
        >= float(ground["minimum_ground_known_coverage"]),
        "ground_reference_risk_cells": ground_risk_cells
        >= int(ground["minimum_reference_risk_cells"]),
        "ground_reference_risk_frames": ground_risk_frames
        >= int(ground["minimum_distinct_frames_with_reference_risk"]),
        "ground_reference_risk_directions": ground_risk_directions
        >= int(
            ground["minimum_distinct_directions_with_reference_risk"]
        ),
    }
    return checks


def _expected_from_current_source(
    replay_root: Path,
    authority_path: Path,
    session_id: str,
) -> dict[str, str]:
    return {
        "source_session_id": session_id,
        "authority_report_sha256": _sha256(authority_path),
        "manifest_sha256": _sha256(
            replay_root / "manifest.replay.jsonl"
        ),
        "dataset_spec_sha256": _sha256(
            replay_root / "dataset_spec.json"
        ),
        "camera_poses_sha256": _sha256(
            replay_root / "source_metadata/camera_poses.csv"
        ),
    }


def run(
    protocol_path: Path,
    ledger_path: Path,
    mechanics_path: Path,
    replay_root: Path,
    authority_path: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    ledger = _load_json(ledger_path)
    mechanics = _load_json(mechanics_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status")
        != "FROZEN_QUALIFICATION_ONLY_ARM_OUTCOME_PROHIBITED"
    ):
        raise ValueError("R3.1 qualification protocol is not frozen")
    if _sha256(mechanics_path) != EXPECTED_MECHANICS_SHA256:
        raise ValueError("R3.1 mechanics protocol hash mismatch")
    burned = _validate_burn_ledger(ledger, ledger_path)
    replay_root = replay_root.resolve()
    rows = _load_jsonl(replay_root / "manifest.replay.jsonl")
    if not rows:
        raise ValueError("Replay manifest is empty")
    session_ids = {str(row.get("session_id")) for row in rows}
    if len(session_ids) != 1:
        raise ValueError("Qualification requires exactly one source session")
    session_id = next(iter(session_ids))
    if session_id in burned:
        raise ValueError(f"Burned source cannot enter R3.1: {session_id}")
    if len(rows) != int(protocol["replay_and_authority"]["frame_count"]):
        raise ValueError("Qualification replay frame count mismatch")
    authority, authority_validation = _validate_authority(
        replay_root,
        rows,
        authority_path,
        _expected_from_current_source(
            replay_root, authority_path, session_id
        ),
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "source_session_id": session_id,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "evidence_role": "REFERENCE_ONLY_OPPORTUNITY_QUALIFICATION",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "burn_ledger_path": str(ledger_path.resolve()),
        "burn_ledger_sha256": _sha256(ledger_path),
        "mechanics_protocol_path": str(mechanics_path.resolve()),
        "mechanics_protocol_sha256": _sha256(mechanics_path),
        "authority_report_path": str(authority_path.resolve()),
        "authority_report_sha256": _sha256(authority_path),
        "manifest_sha256": _sha256(
            replay_root / "manifest.replay.jsonl"
        ),
        "dataset_spec_sha256": _sha256(
            replay_root / "dataset_spec.json"
        ),
        "camera_poses_sha256": _sha256(
            replay_root / "source_metadata/camera_poses.csv"
        ),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "authority_validation": authority_validation,
        "reference_only_assertions": {
            "reference_grid_computed": True,
            "candidate_grid_computed": False,
            "angular_baseline_computed": False,
            "arm_metric_or_delta_computed": False,
        },
    }
    if not authority_validation["ok"]:
        result.update(
            {
                "terminal": REJECTED,
                "qualified": False,
                "checks": {"authority": False},
                "arm_outcome_authorized": False,
            }
        )
        return result
    spec = _load_json(replay_root / "dataset_spec.json")
    bindings = authority["source_pose_authority"]["bindings"]
    binding_by_id = {
        item["manifest_id"]: item for item in bindings
    }
    plane_by_id = {
        item["manifest_id"]: item["local_ground_plane"]
        for item in authority["ground_and_body_proxy_canary"]["per_frame"]
        if item.get("local_ground_plane") is not None
    }
    field = mechanics["field"]
    theta_edges = _theta_edges(field)
    distance_edges = np.asarray(
        field["distance_edges_m"], dtype=np.float64
    )
    height_bands = [
        tuple(float(value) for value in field["height_bands_m"][name])
        for name in LAYERS
    ]
    widths = np.asarray(
        [
            mechanics["standard_synthetic_envelope"][
                "effective_lateral_half_width_m"
            ][name]
            for name in LAYERS
        ],
        dtype=np.float64,
    )
    obstacle_contract = mechanics["obstacle_support"]
    known_contract = mechanics["known_support"]
    qualification_obstacle = protocol[
        "obstacle_opportunity_qualification"
    ]
    qualification_ground = protocol[
        "ground_opportunity_qualification"
    ]
    thresholds = [
        int(value)
        for value in qualification_obstacle[
            "all_sensitivity_thresholds_micro_positive_and_negative_required"
        ]
    ]
    primary = int(
        qualification_obstacle["primary_reference_count_threshold"]
    )
    known_count = np.zeros(3, dtype=np.int64)
    primary_positive = np.zeros(3, dtype=np.int64)
    primary_negative = np.zeros(3, dtype=np.int64)
    sensitivity = {
        threshold: {"positive": 0, "negative": 0}
        for threshold in thresholds
    }
    ground_known_count = 0
    ground_risk_count = 0
    ground_risk_frame_ids: set[str] = set()
    ground_risk_direction_ids: set[int] = set()
    ground_failure_atlas: list[dict[str, Any]] = []
    ground_ids = set(obstacle_contract["ground_semantic_class_ids"])
    for row in rows:
        binding = binding_by_id[row["id"]]
        basis = _anchor_basis(binding, plane_by_id[row["id"]])
        reference_points, reference_dynamic = _obstacle_points_world(
            replay_root,
            row,
            binding,
            spec["camera"],
            stride=int(
                qualification_ground["reference_grid_stride_xy"]
            ),
            offset=int(
                qualification_ground["reference_grid_offset_xy"]
            ),
            excluded_classes=set(
                obstacle_contract["excluded_semantic_class_ids"]
            ),
            dynamic_classes=set(
                obstacle_contract["dynamic_provenance_class_ids"]
            ),
        )
        reference_counts, _ = _swept_prism_counts(
            reference_points,
            reference_dynamic,
            basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
        )
        probes = _swept_prism_probes_world(
            basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
        )
        known, _ = _known_field(
            probes,
            replay_root,
            row,
            binding,
            spec["camera"],
            len(theta_edges) - 1,
            len(distance_edges) - 1,
            len(height_bands),
            float(known_contract["depth_front_tolerance_m"]),
            int(known_contract["minimum_passing_prism_probes"]),
        )
        known_count += known.sum(axis=(0, 1))
        primary_positive += (
            known & (reference_counts >= primary)
        ).sum(axis=(0, 1))
        primary_negative += (
            known & (reference_counts < primary)
        ).sum(axis=(0, 1))
        for threshold in thresholds:
            sensitivity[threshold]["positive"] += int(
                (known & (reference_counts >= threshold)).sum()
            )
            sensitivity[threshold]["negative"] += int(
                (known & (reference_counts < threshold)).sum()
            )
        ground_points, _ = _obstacle_points_world(
            replay_root,
            row,
            binding,
            spec["camera"],
            stride=int(
                qualification_ground["reference_grid_stride_xy"]
            ),
            offset=int(
                qualification_ground["reference_grid_offset_xy"]
            ),
            excluded_classes=set(range(256)) - ground_ids,
            dynamic_classes=set(),
        )
        ground_known, ground_risk, ground_atlas = _ground_support(
            ground_points,
            basis,
            theta_edges,
            distance_edges,
            half_width_m=float(widths[0]),
            section_count=int(
                qualification_ground["longitudinal_section_count"]
            ),
            section_half_length_m=float(
                mechanics["foot_ground_support"][
                    "section_half_length_m"
                ]
            ),
            minimum_points_per_section=int(
                qualification_ground[
                    "minimum_ground_points_per_section"
                ]
            ),
            minimum_supported_sections=int(
                qualification_ground[
                    "minimum_supported_sections_for_known"
                ]
            ),
            maximum_step_rise_m=float(
                qualification_ground["maximum_step_rise_m"]
            ),
            maximum_drop_m=float(
                qualification_ground["maximum_drop_m"]
            ),
        )
        ground_positive = ground_known & (ground_risk > 0.0)
        ground_known_count += int(ground_known.sum())
        ground_risk_count += int(ground_positive.sum())
        if ground_positive.any():
            ground_risk_frame_ids.add(str(row["id"]))
            ground_risk_direction_ids.update(
                int(index)
                for index in np.flatnonzero(
                    ground_positive.any(axis=1)
                )
            )
        if ground_atlas:
            ground_failure_atlas.append(
                {
                    "manifest_id": row["id"],
                    "items": ground_atlas[:5],
                }
            )
    denominator = (
        len(rows) * (len(theta_edges) - 1) * (len(distance_edges) - 1)
    )
    known_coverage = {
        name: float(known_count[index] / denominator)
        for index, name in enumerate(LAYERS)
    }
    positive_by_height = {
        name: int(primary_positive[index])
        for index, name in enumerate(LAYERS)
    }
    negative_by_height = {
        name: int(primary_negative[index])
        for index, name in enumerate(LAYERS)
    }
    sensitivity_output = {
        str(threshold): values
        for threshold, values in sensitivity.items()
    }
    ground_known_coverage = float(
        ground_known_count / denominator
    )
    checks = _reference_decision(
        known_coverage,
        positive_by_height,
        negative_by_height,
        sensitivity_output,
        ground_known_coverage,
        ground_risk_count,
        len(ground_risk_frame_ids),
        len(ground_risk_direction_ids),
        protocol,
    )
    checks["authority"] = authority_validation["ok"]
    qualified = all(checks.values())
    result.update(
        {
            "terminal": QUALIFIED if qualified else REJECTED,
            "qualified": qualified,
            "fixed_denominator_per_height": denominator,
            "reference_obstacle": {
                "known_cells_by_height": {
                    name: int(known_count[index])
                    for index, name in enumerate(LAYERS)
                },
                "known_coverage_by_height": known_coverage,
                "primary_threshold": primary,
                "primary_positive_known_cells_by_height": (
                    positive_by_height
                ),
                "primary_negative_known_cells_by_height": (
                    negative_by_height
                ),
                "sensitivity_micro_opportunity": sensitivity_output,
            },
            "reference_ground": {
                "known_cells": ground_known_count,
                "known_coverage": ground_known_coverage,
                "risk_cells": ground_risk_count,
                "distinct_risk_frames": len(ground_risk_frame_ids),
                "distinct_risk_directions": len(
                    ground_risk_direction_ids
                ),
                "failure_atlas": ground_failure_atlas[:30],
            },
            "checks": checks,
            "arm_outcome_authorized": False,
            "future_stage_c_authorized": False,
            "student_training_authorized": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
        }
    )
    return result


def _require_artifacts_output(path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    artifacts_root = (repo_root / "artifacts.local").resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError(
            f"Output must stay under {artifacts_root}: {resolved}"
        ) from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--burn-ledger", type=Path, required=True)
    parser.add_argument("--mechanics-protocol", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        report = run(
            args.protocol.resolve(),
            args.burn_ledger.resolve(),
            args.mechanics_protocol.resolve(),
            args.replay_root.resolve(),
            args.authority.resolve(),
        )
        payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "qualified": report["qualified"],
                    "output": str(output),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
