#!/usr/bin/env python3
"""Audit HFTF swept-human-envelope label mechanics on consumed sources."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from run_geometry_teacher_canary import (
    _anchor_basis,
    _bin_obstacle_support,
    _known_field,
    _obstacle_points_world,
    _sha256,
    _theta_edges,
    _validate_authority,
)
from verify_sanpo_pose_geometry_authority import _load_json, _load_jsonl


SCHEMA = "blindassist_hftf_stage_b_swept_envelope_label_mechanics_result_d0"
PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_b_swept_envelope_label_mechanics_canary_d0"
)
ADMITTED_TERMINAL = (
    "STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_ADMITTED_FOR_FRESH_R3"
)
NOT_READY_TERMINAL = "STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_NOT_READY"
STATE_UNKNOWN = 0
STATE_SAFE = 1
STATE_RISK = 2


def _trajectory_directions(
    theta_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0
    return np.cos(centers), np.sin(centers)


def _tri_state_field(
    known: np.ndarray, risk: np.ndarray
) -> np.ndarray:
    if known.shape != risk.shape:
        raise ValueError("Known and risk fields must have the same shape")
    state = np.full(known.shape, STATE_UNKNOWN, dtype=np.uint8)
    state[known & (risk == 0.0)] = STATE_SAFE
    state[known & (risk > 0.0)] = STATE_RISK
    return state


def _swept_prism_counts(
    points_world: np.ndarray,
    dynamic: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    theta_edges: np.ndarray,
    distance_edges: np.ndarray,
    height_bands: list[tuple[float, float]],
    lateral_half_widths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    theta_count = len(theta_edges) - 1
    distance_count = len(distance_edges) - 1
    if lateral_half_widths.shape != (len(height_bands),):
        raise ValueError("One lateral half-width is required per height band")
    if np.any(lateral_half_widths <= 0.0):
        raise ValueError("Lateral half-widths must be positive")
    origin, forward, right, up = basis
    relative = points_world - origin[:, None]
    forward_coordinate = forward @ relative
    right_coordinate = right @ relative
    height = up @ relative
    cosine, sine = _trajectory_directions(theta_edges)
    counts = np.zeros(
        (theta_count, distance_count, len(height_bands)),
        dtype=np.int64,
    )
    dynamic_counts = np.zeros_like(counts)
    for theta_index in range(theta_count):
        along = (
            forward_coordinate * cosine[theta_index]
            + right_coordinate * sine[theta_index]
        )
        cross = (
            -forward_coordinate * sine[theta_index]
            + right_coordinate * cosine[theta_index]
        )
        distance_index = np.searchsorted(
            distance_edges, along, side="right"
        ) - 1
        distance_index = np.where(
            np.isclose(
                along, distance_edges[-1], atol=1e-12, rtol=0.0
            ),
            distance_count - 1,
            distance_index,
        )
        for height_index, (lower, upper) in enumerate(height_bands):
            upper_ok = (
                height <= upper
                if height_index == len(height_bands) - 1
                else height < upper
            )
            valid = (
                (distance_index >= 0)
                & (distance_index < distance_count)
                & (np.abs(cross) <= lateral_half_widths[height_index])
                & (height >= lower)
                & upper_ok
            )
            np.add.at(
                counts,
                (
                    theta_index,
                    distance_index[valid],
                    height_index,
                ),
                1,
            )
            dynamic_valid = valid & dynamic
            np.add.at(
                dynamic_counts,
                (
                    theta_index,
                    distance_index[dynamic_valid],
                    height_index,
                ),
                1,
            )
    return counts, dynamic_counts


def _swept_prism_probes_world(
    basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    theta_edges: np.ndarray,
    distance_edges: np.ndarray,
    height_bands: list[tuple[float, float]],
    lateral_half_widths: np.ndarray,
) -> np.ndarray:
    origin, forward, right, up = basis
    cosine, sine = _trajectory_directions(theta_edges)
    probes: list[np.ndarray] = []
    for theta_index in range(len(theta_edges) - 1):
        direction = (
            forward * cosine[theta_index]
            + right * sine[theta_index]
        )
        lateral = (
            -forward * sine[theta_index]
            + right * cosine[theta_index]
        )
        for distance_index in range(len(distance_edges) - 1):
            lower = float(distance_edges[distance_index])
            upper = float(distance_edges[distance_index + 1])
            center = (lower + upper) / 2.0
            for height_index, (height_lower, height_upper) in enumerate(
                height_bands
            ):
                half_width = float(lateral_half_widths[height_index])
                height_center = (height_lower + height_upper) / 2.0
                local = [(center, 0.0, height_center)]
                local.extend(
                    (along, cross, height)
                    for along in (lower, upper)
                    for cross in (-half_width, half_width)
                    for height in (height_lower, height_upper)
                )
                probes.append(
                    np.stack(
                        [
                            origin
                            + direction * along
                            + lateral * cross
                            + up * height
                            for along, cross, height in local
                        ],
                        axis=1,
                    )
                )
    return np.stack(probes, axis=0)


def _ground_support(
    ground_points_world: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    theta_edges: np.ndarray,
    distance_edges: np.ndarray,
    *,
    half_width_m: float,
    section_count: int,
    section_half_length_m: float,
    minimum_points_per_section: int,
    minimum_supported_sections: int,
    maximum_step_rise_m: float,
    maximum_drop_m: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    theta_count = len(theta_edges) - 1
    distance_count = len(distance_edges) - 1
    known = np.zeros((theta_count, distance_count), dtype=bool)
    risk = np.zeros((theta_count, distance_count), dtype=np.float64)
    atlas: list[dict[str, Any]] = []
    origin, forward, right, up = basis
    relative = ground_points_world - origin[:, None]
    forward_coordinate = forward @ relative
    right_coordinate = right @ relative
    height = up @ relative
    cosine, sine = _trajectory_directions(theta_edges)
    for theta_index in range(theta_count):
        along = (
            forward_coordinate * cosine[theta_index]
            + right_coordinate * sine[theta_index]
        )
        cross = (
            -forward_coordinate * sine[theta_index]
            + right_coordinate * cosine[theta_index]
        )
        for distance_index in range(distance_count):
            lower = float(distance_edges[distance_index])
            upper = float(distance_edges[distance_index + 1])
            centers = np.linspace(
                lower + section_half_length_m,
                upper - section_half_length_m,
                section_count,
            )
            medians: list[float | None] = []
            for center in centers:
                section = (
                    (np.abs(along - center) <= section_half_length_m)
                    & (np.abs(cross) <= half_width_m)
                )
                values = height[section]
                medians.append(
                    float(np.median(values))
                    if len(values) >= minimum_points_per_section
                    else None
                )
            supported = sum(value is not None for value in medians)
            known[theta_index, distance_index] = (
                supported >= minimum_supported_sections
            )
            deltas = [
                float(right_value - left_value)
                for left_value, right_value in zip(
                    medians[:-1], medians[1:]
                )
                if left_value is not None and right_value is not None
            ]
            hazardous = any(
                delta > maximum_step_rise_m
                or delta < -maximum_drop_m
                for delta in deltas
            )
            risk[theta_index, distance_index] = (
                1.0 if hazardous else 0.0
            )
            if not known[theta_index, distance_index] or hazardous:
                atlas.append(
                    {
                        "theta_index": theta_index,
                        "distance_index": distance_index,
                        "supported_sections": supported,
                        "section_height_medians": medians,
                        "hazardous": hazardous,
                    }
                )
    return known, risk, atlas


def _structural_canaries() -> dict[str, bool]:
    basis = (
        np.zeros(3),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
    )
    theta_edges = np.radians(np.asarray([-45.0, 0.0, 45.0]))
    straight_theta_edges = np.radians(np.asarray([-15.0, 15.0]))
    distance_edges = np.asarray([0.0, 2.0])
    bands = [(0.05, 0.35), (0.35, 1.35), (1.35, 2.05)]
    dynamic = np.asarray([False])
    body_point = np.asarray([[1.0], [0.35], [0.8]])
    narrow, _ = _swept_prism_counts(
        body_point,
        dynamic,
        basis,
        straight_theta_edges,
        distance_edges,
        bands,
        np.asarray([0.1, 0.1, 0.1]),
    )
    wide, _ = _swept_prism_counts(
        body_point,
        dynamic,
        basis,
        straight_theta_edges,
        distance_edges,
        bands,
        np.asarray([0.4, 0.4, 0.4]),
    )
    head_point = np.asarray([[1.0], [0.0], [1.8]])
    head_counts, _ = _swept_prism_counts(
        head_point,
        dynamic,
        basis,
        straight_theta_edges,
        distance_edges,
        bands,
        np.asarray([0.3, 0.4, 0.28]),
    )
    direction = math.radians(22.5)
    directional_point = np.asarray(
        [[math.cos(direction)], [math.sin(direction)], [0.8]]
    )
    directional_counts, _ = _swept_prism_counts(
        directional_point,
        dynamic,
        basis,
        theta_edges,
        distance_edges,
        bands,
        np.asarray([0.05, 0.05, 0.05]),
    )
    ground_points = []
    for center_index, center in enumerate(
        np.linspace(0.2, 1.8, 5)
    ):
        ground_height = 0.25 if center_index >= 3 else 0.0
        for offset in (-0.05, 0.0, 0.05):
            ground_points.append([center + offset, 0.0, ground_height])
    ground_known, ground_risk, _ = _ground_support(
        np.asarray(ground_points, dtype=np.float64).T,
        basis,
        np.radians(np.asarray([-15.0, 15.0])),
        distance_edges,
        half_width_m=0.3,
        section_count=5,
        section_half_length_m=0.2,
        minimum_points_per_section=3,
        minimum_supported_sections=4,
        maximum_step_rise_m=0.18,
        maximum_drop_m=0.15,
    )
    empty_known, empty_risk, _ = _ground_support(
        np.zeros((3, 0), dtype=np.float64),
        basis,
        np.radians(np.asarray([-15.0, 15.0])),
        distance_edges,
        half_width_m=0.3,
        section_count=5,
        section_half_length_m=0.2,
        minimum_points_per_section=3,
        minimum_supported_sections=4,
        maximum_step_rise_m=0.18,
        maximum_drop_m=0.15,
    )
    return {
        "wider_envelope_collision_is_monotone_superset": bool(
            np.all(wide >= narrow) and wide.sum() > narrow.sum()
        ),
        "height_specific_obstacle_changes_only_intersected_height_layer": (
            bool(
                head_counts[:, :, 2].sum() > 0
                and head_counts[:, :, :2].sum() == 0
            )
        ),
        "lateral_dilation_catches_obstacle_inside_body_width": bool(
            wide[:, :, 1].sum() > narrow[:, :, 1].sum()
        ),
        "candidate_directions_remain_separable": bool(
            directional_counts[1, :, 1].sum() == 1
            and directional_counts[0, :, 1].sum() == 0
        ),
        "step_or_drop_affects_foot_layer": bool(
            ground_known.all() and ground_risk.max() == 1.0
        ),
        "missing_ground_support_cannot_be_safe": bool(
            not empty_known.any() and empty_risk.max() == 0.0
        ),
        "zero_obstacle_support_is_safe_only_when_known": bool(
            not empty_known.any()
        ),
    }


def _session_audit(
    replay_root: Path,
    authority_path: Path,
    mechanics_protocol: dict[str, Any],
    r2_protocol: dict[str, Any],
) -> dict[str, Any]:
    replay_root = replay_root.resolve()
    rows = _load_jsonl(replay_root / "manifest.replay.jsonl")
    spec = _load_json(replay_root / "dataset_spec.json")
    session_id = str(rows[0]["session_id"])
    expected_by_id = {
        item["source_session_id"]: item
        for item in r2_protocol["required_sessions"]
    }
    if session_id not in mechanics_protocol["parent_sessions"]:
        raise ValueError(f"Unfrozen mechanics source: {session_id}")
    authority, validation = _validate_authority(
        replay_root, rows, authority_path, expected_by_id[session_id]
    )
    result: dict[str, Any] = {
        "source_session_id": session_id,
        "authority_validation": validation,
        "frame_count": len(rows),
    }
    if not validation["ok"]:
        result["ok"] = False
        return result
    binding_by_id = {
        item["manifest_id"]: item
        for item in authority["source_pose_authority"]["bindings"]
    }
    plane_by_id = {
        item["manifest_id"]: item["local_ground_plane"]
        for item in authority["ground_and_body_proxy_canary"]["per_frame"]
        if item.get("local_ground_plane") is not None
    }
    field = mechanics_protocol["field"]
    theta_edges = _theta_edges(field)
    distance_edges = np.asarray(
        field["distance_edges_m"], dtype=np.float64
    )
    height_bands = [
        tuple(float(value) for value in field["height_bands_m"][name])
        for name in ("foot", "body", "head")
    ]
    envelope = mechanics_protocol["standard_synthetic_envelope"]
    widths = np.asarray(
        [
            envelope["effective_lateral_half_width_m"][name]
            for name in ("foot", "body", "head")
        ],
        dtype=np.float64,
    )
    obstacle = mechanics_protocol["obstacle_support"]
    known_contract = mechanics_protocol["known_support"]
    ground_contract = mechanics_protocol["foot_ground_support"]
    known_counts = np.zeros(3, dtype=np.int64)
    required_per_height = (
        len(rows) * (len(theta_edges) - 1) * (len(distance_edges) - 1)
    )
    height_disagreement = 0
    unique_swept = 0
    foot_ground_risk = 0
    foot_ground_unknown = 0
    dynamic_support = 0
    unknown_to_safe_violations = 0
    atlas: list[dict[str, Any]] = []
    for row in rows:
        binding = binding_by_id[row["id"]]
        basis = _anchor_basis(binding, plane_by_id[row["id"]])
        obstacle_points, dynamic = _obstacle_points_world(
            replay_root,
            row,
            binding,
            spec["camera"],
            stride=int(obstacle["point_sample_stride_xy"]),
            offset=int(obstacle["point_sample_offset_xy"]),
            excluded_classes=set(obstacle["excluded_semantic_class_ids"]),
            dynamic_classes=set(
                obstacle["dynamic_provenance_class_ids"]
            ),
        )
        ground_points, _ = _obstacle_points_world(
            replay_root,
            row,
            binding,
            spec["camera"],
            stride=int(obstacle["point_sample_stride_xy"]),
            offset=int(obstacle["point_sample_offset_xy"]),
            excluded_classes=(
                set(range(256))
                - set(obstacle["ground_semantic_class_ids"])
            ),
            dynamic_classes=set(),
        )
        swept_counts, dynamic_counts = _swept_prism_counts(
            obstacle_points,
            dynamic,
            basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
        )
        old_counts, _ = _bin_obstacle_support(
            obstacle_points,
            dynamic,
            basis,
            theta_edges,
            distance_edges,
            height_bands,
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
        ground_known, ground_risk, ground_atlas = _ground_support(
            ground_points,
            basis,
            theta_edges,
            distance_edges,
            half_width_m=float(widths[0]),
            section_count=int(
                ground_contract["longitudinal_section_count"]
            ),
            section_half_length_m=float(
                ground_contract["section_half_length_m"]
            ),
            minimum_points_per_section=int(
                ground_contract["minimum_ground_points_per_section"]
            ),
            minimum_supported_sections=int(
                ground_contract["minimum_supported_sections_for_known"]
            ),
            maximum_step_rise_m=float(
                ground_contract["maximum_step_rise_m"]
            ),
            maximum_drop_m=float(ground_contract["maximum_drop_m"]),
        )
        known[:, :, 0] &= ground_known
        saturation = float(
            obstacle["risk_support_saturation_point_count"]
        )
        risk = np.minimum(1.0, swept_counts / saturation)
        risk[:, :, 0] = np.maximum(risk[:, :, 0], ground_risk)
        state = _tri_state_field(known, risk)
        known_counts += known.sum(axis=(0, 1))
        jointly_known = np.all(known, axis=2)
        disagreement = (
            np.max(risk, axis=2) - np.min(risk, axis=2)
        ) >= 0.25
        height_disagreement += int((jointly_known & disagreement).sum())
        unique_swept += int(
            ((swept_counts > 0) & (old_counts == 0)).sum()
        )
        foot_ground_risk += int((ground_known & (ground_risk > 0)).sum())
        foot_ground_unknown += int((~ground_known).sum())
        dynamic_support += int(dynamic_counts.sum())
        unknown_to_safe_violations += int(
            ((~known) & (state == STATE_SAFE)).sum()
        )
        if ground_atlas or (jointly_known & disagreement).any():
            atlas.append(
                {
                    "manifest_id": row["id"],
                    "height_disagreement_count": int(
                        (jointly_known & disagreement).sum()
                    ),
                    "ground_failures": ground_atlas[:5],
                }
            )
    result.update(
        {
            "required_cells_per_height": required_per_height,
            "known_cells_by_height": {
                name: int(known_counts[index])
                for index, name in enumerate(("foot", "body", "head"))
            },
            "known_coverage_by_height": {
                name: float(known_counts[index] / required_per_height)
                for index, name in enumerate(("foot", "body", "head"))
            },
            "height_disagreement": {
                "numerator": height_disagreement,
                "denominator": required_per_height,
                "fraction": float(
                    height_disagreement / required_per_height
                ),
            },
            "unique_swept_collision_cells_vs_angular_point_support": (
                unique_swept
            ),
            "foot_ground_risk_cells": foot_ground_risk,
            "foot_ground_unknown_cells": foot_ground_unknown,
            "dynamic_provenance_support_points": dynamic_support,
            "unknown_to_safe_violation_count": (
                unknown_to_safe_violations
            ),
            "failure_atlas": atlas[:30],
            "ok": validation["ok"],
        }
    )
    return result


def run(
    mechanics_protocol_path: Path,
    r2_protocol_path: Path,
    session_inputs: list[tuple[Path, Path]],
) -> dict[str, Any]:
    mechanics = _load_json(mechanics_protocol_path)
    r2 = _load_json(r2_protocol_path)
    if (
        mechanics.get("schema") != PROTOCOL_SCHEMA
        or mechanics.get("status")
        != "FROZEN_DEVELOPMENT_CANARY_RESULT_NOT_RUN"
    ):
        raise ValueError("Mechanics protocol is not frozen D0")
    if len(session_inputs) != len(mechanics["parent_sessions"]):
        raise ValueError("Expected exactly four consumed session inputs")
    structural = _structural_canaries()
    sessions = [
        _session_audit(replay, authority, mechanics, r2)
        for replay, authority in session_inputs
    ]
    ids = [item["source_session_id"] for item in sessions]
    exact = (
        set(ids) == set(mechanics["parent_sessions"])
        and len(ids) == len(set(ids))
    )
    no_unknown_to_safe = all(
        item.get("unknown_to_safe_violation_count") == 0
        for item in sessions
    )
    nondegenerate = (
        sum(
            item.get("height_disagreement", {}).get("numerator", 0)
            for item in sessions
        )
        > 0
        and all(
            all(value > 0 for value in item["known_cells_by_height"].values())
            for item in sessions
        )
    )
    admitted = (
        all(structural.values())
        and exact
        and all(item.get("ok") for item in sessions)
        and no_unknown_to_safe
        and nondegenerate
    )
    return {
        "schema": SCHEMA,
        "terminal": ADMITTED_TERMINAL if admitted else NOT_READY_TERMINAL,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "evidence_role": "CONSUMED_MECHANICS_AUDIT_ONLY",
        "mechanics_protocol_path": str(
            mechanics_protocol_path.resolve()
        ),
        "mechanics_protocol_sha256": _sha256(
            mechanics_protocol_path
        ),
        "r2_protocol_path": str(r2_protocol_path.resolve()),
        "r2_protocol_sha256": _sha256(r2_protocol_path),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "structural_canaries": structural,
        "exact_consumed_session_set": exact,
        "sessions": sessions,
        "aggregate": {
            "height_disagreement_count": sum(
                item["height_disagreement"]["numerator"]
                for item in sessions
            ),
            "unique_swept_collision_cells": sum(
                item[
                    "unique_swept_collision_cells_vs_angular_point_support"
                ]
                for item in sessions
            ),
            "foot_ground_risk_cells": sum(
                item["foot_ground_risk_cells"] for item in sessions
            ),
            "foot_ground_unknown_cells": sum(
                item["foot_ground_unknown_cells"] for item in sessions
            ),
            "dynamic_provenance_support_points": sum(
                item["dynamic_provenance_support_points"]
                for item in sessions
            ),
            "unknown_to_safe_violation_count": sum(
                item["unknown_to_safe_violation_count"]
                for item in sessions
            ),
        },
        "admission_checks": {
            "all_structural_tests_pass": all(structural.values()),
            "all_four_consumed_sources_decode_and_bind": (
                exact and all(item.get("ok") for item in sessions)
            ),
            "no_unknown_to_safe_violation": no_unknown_to_safe,
            "nondegenerate_known_and_height_specific_outputs_exist": (
                nondegenerate
            ),
        },
        "fresh_r3_automatically_run": False,
        "student_training_authorized": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
    }


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
    parser.add_argument("--r2-protocol", type=Path, required=True)
    parser.add_argument(
        "--session",
        action="append",
        nargs=2,
        metavar=("REPLAY_ROOT", "AUTHORITY_REPORT"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        report = run(
            args.protocol.resolve(),
            args.r2_protocol.resolve(),
            [
                (Path(replay).resolve(), Path(authority).resolve())
                for replay, authority in args.session
            ],
        )
        payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
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
