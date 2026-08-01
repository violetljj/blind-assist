#!/usr/bin/env python3
"""Run formal HFTF Stage B swept-envelope reference comparison R3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_swept_envelope_label_mechanics import _ground_support
from pilot_swept_envelope_reference_metrics import (
    LAYERS,
    _cohort_metrics,
    _confusion,
    _empty_counts,
    _pixel_lattices_disjoint,
    _session_pilot,
    _summarize_counts,
)
from run_geometry_teacher_canary import (
    _anchor_basis,
    _obstacle_points_world,
    _sha256,
    _theta_edges,
    _validate_authority,
)
from verify_sanpo_pose_geometry_authority import _load_json, _load_jsonl


SCHEMA = (
    "blindassist_hftf_stage_b_swept_envelope_reference_comparison_result_r3"
)
PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_b_swept_envelope_reference_comparison_protocol_r3"
)
SOURCE_NOT_EVALUABLE = "R3_SOURCE_OR_REFERENCE_NOT_EVALUABLE"
OBSTACLE_STOP = "R3_SWEPT_ENVELOPE_REFERENCE_GAIN_NOT_SUPPORTED_STOP"
GROUND_NOT_EVALUABLE = (
    "R3_OBSTACLE_ENVELOPE_GAIN_SUPPORTED_GROUND_NOT_EVALUABLE"
)
GROUND_STOP = "R3_GROUND_PROXY_NOT_SUPPORTED_STOP"
SUPPORTED = "R3_STAGE_B_SWEPT_ENVELOPE_PROXY_SUPPORTED"


def _session_ground_comparison(
    replay_root: Path,
    authority_path: Path,
    protocol: dict[str, Any],
    mechanics: dict[str, Any],
) -> dict[str, Any]:
    replay_root = replay_root.resolve()
    rows = _load_jsonl(replay_root / "manifest.replay.jsonl")
    spec = _load_json(replay_root / "dataset_spec.json")
    session_id = str(rows[0]["session_id"])
    expected_by_id = {
        item["source_session_id"]: item
        for item in protocol["required_sessions"]
    }
    authority, validation = _validate_authority(
        replay_root,
        rows,
        authority_path,
        expected_by_id[session_id],
    )
    result: dict[str, Any] = {
        "source_session_id": session_id,
        "authority_validation": validation,
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
    field = mechanics["field"]
    theta_edges = _theta_edges(field)
    distance_edges = np.asarray(
        field["distance_edges_m"], dtype=np.float64
    )
    obstacle = mechanics["obstacle_support"]
    ground_ids = set(obstacle["ground_semantic_class_ids"])
    ground = protocol["ground_comparison"]
    foot_width = float(
        mechanics["standard_synthetic_envelope"][
            "effective_lateral_half_width_m"
        ]["foot"]
    )
    candidate_known_count = 0
    reference_known_count = 0
    shared_known_count = 0
    candidate_risk_count = 0
    reference_risk_count = 0
    counts = _empty_counts()
    atlas: list[dict[str, Any]] = []
    for row in rows:
        binding = binding_by_id[row["id"]]
        basis = _anchor_basis(binding, plane_by_id[row["id"]])
        candidate_points, _ = _obstacle_points_world(
            replay_root,
            row,
            binding,
            spec["camera"],
            stride=int(
                ground["candidate_point_sample_stride_xy"]
            ),
            offset=int(
                ground["candidate_point_sample_offset_xy"]
            ),
            excluded_classes=set(range(256)) - ground_ids,
            dynamic_classes=set(),
        )
        reference_points, _ = _obstacle_points_world(
            replay_root,
            row,
            binding,
            spec["camera"],
            stride=int(
                ground["reference_point_sample_stride_xy"]
            ),
            offset=int(
                ground["reference_point_sample_offset_xy"]
            ),
            excluded_classes=set(range(256)) - ground_ids,
            dynamic_classes=set(),
        )
        candidate_known, candidate_risk, candidate_atlas = (
            _ground_support(
                candidate_points,
                basis,
                theta_edges,
                distance_edges,
                half_width_m=foot_width,
                section_count=int(
                    ground["longitudinal_section_count"]
                ),
                section_half_length_m=float(
                    ground["section_half_length_m"]
                ),
                minimum_points_per_section=int(
                    ground[
                        "candidate_minimum_ground_points_per_section"
                    ]
                ),
                minimum_supported_sections=int(
                    ground[
                        "minimum_supported_sections_for_known"
                    ]
                ),
                maximum_step_rise_m=float(
                    ground["maximum_step_rise_m"]
                ),
                maximum_drop_m=float(ground["maximum_drop_m"]),
            )
        )
        reference_known, reference_risk, reference_atlas = (
            _ground_support(
                reference_points,
                basis,
                theta_edges,
                distance_edges,
                half_width_m=foot_width,
                section_count=int(
                    ground["longitudinal_section_count"]
                ),
                section_half_length_m=float(
                    ground["section_half_length_m"]
                ),
                minimum_points_per_section=int(
                    ground[
                        "reference_minimum_ground_points_per_section"
                    ]
                ),
                minimum_supported_sections=int(
                    ground[
                        "minimum_supported_sections_for_known"
                    ]
                ),
                maximum_step_rise_m=float(
                    ground["maximum_step_rise_m"]
                ),
                maximum_drop_m=float(ground["maximum_drop_m"]),
            )
        )
        shared_known = candidate_known & reference_known
        candidate_positive = candidate_risk > 0.0
        reference_positive = reference_risk > 0.0
        candidate_known_count += int(candidate_known.sum())
        reference_known_count += int(reference_known.sum())
        shared_known_count += int(shared_known.sum())
        candidate_risk_count += int(
            (shared_known & candidate_positive).sum()
        )
        reference_risk_count += int(
            (shared_known & reference_positive).sum()
        )
        frame_counts = _confusion(
            candidate_positive, reference_positive, shared_known
        )
        for field_name, value in frame_counts.items():
            counts[field_name] += int(value)
        if candidate_atlas or reference_atlas:
            atlas.append(
                {
                    "manifest_id": row["id"],
                    "candidate_failures": candidate_atlas[:3],
                    "reference_failures": reference_atlas[:3],
                }
            )
    denominator = (
        len(rows) * (len(theta_edges) - 1) * (len(distance_edges) - 1)
    )
    result.update(
        {
            "fixed_denominator": denominator,
            "candidate_known_cells": candidate_known_count,
            "reference_known_cells": reference_known_count,
            "shared_known_cells": shared_known_count,
            "candidate_known_coverage": float(
                candidate_known_count / denominator
            ),
            "reference_known_coverage": float(
                reference_known_count / denominator
            ),
            "shared_known_coverage": float(
                shared_known_count / denominator
            ),
            "candidate_risk_cells_on_shared_known": (
                candidate_risk_count
            ),
            "reference_risk_cells_on_shared_known": (
                reference_risk_count
            ),
            "candidate_vs_reference": _summarize_counts(counts),
            "failure_atlas": atlas[:30],
            "ok": validation["ok"],
        }
    )
    return result


def _aggregate_ground(
    sessions: list[dict[str, Any]]
) -> dict[str, Any]:
    counts = _empty_counts()
    for session in sessions:
        source = session["candidate_vs_reference"]
        for field_name in ("tp", "fp", "fn", "tn"):
            counts[field_name] += int(source[field_name])
    return {
        "candidate_vs_reference": _summarize_counts(counts),
        "reference_risk_opportunities": sum(
            item["reference_risk_cells_on_shared_known"]
            for item in sessions
        ),
        "candidate_risk_cells_on_shared_known": sum(
            item["candidate_risk_cells_on_shared_known"]
            for item in sessions
        ),
    }


def _metric_delta(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    metric: str,
) -> float:
    candidate_value = candidate[metric]
    baseline_value = baseline[metric]
    if candidate_value is None or baseline_value is None:
        raise ValueError(f"Metric {metric} is undefined")
    return float(candidate_value - baseline_value)


def _decide_terminal(
    source_ready: bool,
    obstacle_supported: bool,
    ground_opportunity_count: int,
    ground_supported: bool,
) -> str:
    if not source_ready:
        return SOURCE_NOT_EVALUABLE
    if not obstacle_supported:
        return OBSTACLE_STOP
    if ground_opportunity_count <= 0:
        return GROUND_NOT_EVALUABLE
    if not ground_supported:
        return GROUND_STOP
    return SUPPORTED


def run(
    protocol_path: Path,
    mechanics_path: Path,
    source_preparation_path: Path,
    session_inputs: list[tuple[Path, Path]],
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    mechanics = _load_json(mechanics_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "FROZEN_RESULT_NOT_RUN"
    ):
        raise ValueError("R3 protocol is not frozen")
    if _sha256(mechanics_path) != protocol["mechanics_contract"][
        "sha256"
    ]:
        raise ValueError("Mechanics protocol hash mismatch")
    if _sha256(source_preparation_path) != protocol[
        "source_preparation_contract"
    ]["sha256"]:
        raise ValueError("Source preparation protocol hash mismatch")
    if len(session_inputs) != protocol["required_session_count"]:
        raise ValueError("Expected exactly four R3 session inputs")
    lattice_disjoint = _pixel_lattices_disjoint(
        int(protocol["candidate"]["point_sample_stride_xy"]),
        int(protocol["candidate"]["point_sample_offset_xy"]),
        int(protocol["reference"]["point_sample_stride_xy"]),
        int(protocol["reference"]["point_sample_offset_xy"]),
    )
    obstacle_sessions = [
        _session_pilot(
            replay,
            authority,
            protocol,
            mechanics,
            protocol,
        )
        for replay, authority in session_inputs
    ]
    ground_sessions = [
        _session_ground_comparison(
            replay,
            authority,
            protocol,
            mechanics,
        )
        for replay, authority in session_inputs
    ]
    ids = [item["source_session_id"] for item in obstacle_sessions]
    exact = (
        set(ids) == set(protocol["parent_sessions"])
        and len(ids) == len(set(ids))
    )
    thresholds = [
        int(value)
        for value in protocol["reference"][
            "positive_count_threshold_sensitivity"
        ]
    ]
    cohort = _cohort_metrics(obstacle_sessions, thresholds)
    obstacle_gates = protocol["obstacle_gates"]
    source_authority_ready = (
        exact
        and lattice_disjoint
        and all(item.get("ok") for item in obstacle_sessions)
        and all(item.get("ok") for item in ground_sessions)
    )
    obstacle_known_ready = all(
        all(
            coverage
            >= float(
                obstacle_gates[
                    "minimum_known_coverage_each_height_each_session"
                ]
            )
            for coverage in item[
                "known_coverage_by_height"
            ].values()
        )
        for item in obstacle_sessions
    )
    opportunity_ready = all(
        all(
            item["reference_opportunity_by_threshold"][
                str(threshold)
            ]["positive"]
            > 0
            and item["reference_opportunity_by_threshold"][
                str(threshold)
            ]["negative"]
            > 0
            for threshold in thresholds
        )
        for item in obstacle_sessions
    )
    ground_contract = protocol["ground_comparison"]
    ground_known_ready = all(
        item["candidate_known_coverage"]
        >= float(
            ground_contract[
                "minimum_candidate_and_reference_known_coverage_each_session"
            ]
        )
        and item["reference_known_coverage"]
        >= float(
            ground_contract[
                "minimum_candidate_and_reference_known_coverage_each_session"
            ]
        )
        and item["shared_known_coverage"]
        >= float(
            ground_contract[
                "minimum_shared_known_coverage_each_session"
            ]
        )
        for item in ground_sessions
    )
    source_ready = (
        source_authority_ready
        and obstacle_known_ready
        and opportunity_ready
        and ground_known_ready
    )
    primary = str(
        protocol["reference"]["primary_positive_count_threshold"]
    )
    candidate_primary = cohort[primary]["candidate"][
        "micro_all_layers"
    ]
    baseline_primary = cohort[primary]["baseline"][
        "micro_all_layers"
    ]
    cohort_f1_delta = _metric_delta(
        candidate_primary, baseline_primary, "f1"
    )
    cohort_precision_delta = _metric_delta(
        candidate_primary, baseline_primary, "precision"
    )
    cohort_recall_delta = _metric_delta(
        candidate_primary, baseline_primary, "recall"
    )
    session_f1_deltas = [
        _metric_delta(
            item["metrics_by_reference_count_threshold"][primary][
                "candidate"
            ]["micro_all_layers"],
            item["metrics_by_reference_count_threshold"][primary][
                "baseline"
            ]["micro_all_layers"],
            "f1",
        )
        for item in obstacle_sessions
    ]
    height_primary_supported = all(
        _metric_delta(
            cohort[primary]["candidate"][layer],
            cohort[primary]["baseline"][layer],
            "f1",
        )
        > 0.0
        for layer in LAYERS
    )
    sensitivity_f1_supported = all(
        _metric_delta(
            cohort[str(threshold)]["candidate"][
                "micro_all_layers"
            ],
            cohort[str(threshold)]["baseline"][
                "micro_all_layers"
            ],
            "f1",
        )
        > 0.0
        for threshold in thresholds
    )
    sensitivity_paired_supported = all(
        cohort[str(threshold)]["candidate"][
            "micro_all_layers"
        ]["candidate_only_correct"]
        > cohort[str(threshold)]["candidate"][
            "micro_all_layers"
        ]["baseline_only_correct"]
        for threshold in thresholds
    )
    obstacle_supported = (
        source_ready
        and cohort_f1_delta
        >= float(
            obstacle_gates[
                "primary_minimum_cohort_micro_f1_delta"
            ]
        )
        and cohort_precision_delta
        >= float(
            obstacle_gates[
                "primary_minimum_cohort_precision_delta"
            ]
        )
        and cohort_recall_delta
        >= float(
            obstacle_gates[
                "primary_minimum_cohort_recall_delta"
            ]
        )
        and all(
            delta
            >= float(
                obstacle_gates[
                    "primary_minimum_session_micro_f1_delta"
                ]
            )
            for delta in session_f1_deltas
        )
        and height_primary_supported
        and sensitivity_f1_supported
        and sensitivity_paired_supported
    )
    ground_cohort = _aggregate_ground(ground_sessions)
    ground_opportunities = int(
        ground_cohort["reference_risk_opportunities"]
    )
    ground_metrics = ground_cohort["candidate_vs_reference"]
    ground_supported = (
        ground_opportunities
        >= int(
            ground_contract[
                "minimum_cohort_reference_risk_opportunities_for_full_stage_b"
            ]
        )
        and ground_metrics["precision"] is not None
        and ground_metrics["recall"] is not None
        and ground_metrics["precision"]
        >= float(
            ground_contract[
                "minimum_candidate_precision_if_opportunity_exists"
            ]
        )
        and ground_metrics["recall"]
        >= float(
            ground_contract[
                "minimum_candidate_recall_if_opportunity_exists"
            ]
        )
    )
    terminal = _decide_terminal(
        source_ready,
        obstacle_supported,
        ground_opportunities,
        ground_supported,
    )
    return {
        "schema": SCHEMA,
        "terminal": terminal,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "evidence_role": "FRESH_STAGE_B_GEOMETRY_PROXY_COMPARISON",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "mechanics_protocol_path": str(mechanics_path.resolve()),
        "mechanics_protocol_sha256": _sha256(mechanics_path),
        "source_preparation_path": str(
            source_preparation_path.resolve()
        ),
        "source_preparation_sha256": _sha256(
            source_preparation_path
        ),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "exact_fresh_session_set": exact,
        "candidate_reference_pixel_lattices_disjoint": (
            lattice_disjoint
        ),
        "obstacle_sessions": obstacle_sessions,
        "obstacle_cohort_metrics_by_reference_count_threshold": (
            cohort
        ),
        "ground_sessions": ground_sessions,
        "ground_cohort": ground_cohort,
        "ordered_checks": {
            "source_authority_and_exact_set": source_authority_ready,
            "obstacle_known_coverage": obstacle_known_ready,
            "reference_opportunity": opportunity_ready,
            "ground_known_coverage": ground_known_ready,
            "source_and_reference_ready": source_ready,
            "primary_cohort_micro_f1_delta": cohort_f1_delta,
            "primary_cohort_precision_delta": cohort_precision_delta,
            "primary_cohort_recall_delta": cohort_recall_delta,
            "primary_session_micro_f1_deltas": session_f1_deltas,
            "primary_height_f1_direction_supported": (
                height_primary_supported
            ),
            "all_sensitivity_f1_directions_supported": (
                sensitivity_f1_supported
            ),
            "all_sensitivity_paired_directions_supported": (
                sensitivity_paired_supported
            ),
            "obstacle_envelope_gain_supported": obstacle_supported,
            "reference_ground_risk_opportunities": (
                ground_opportunities
            ),
            "ground_proxy_supported": ground_supported,
        },
        "future_stage_c_protocol_freeze_authorized": (
            terminal == SUPPORTED
        ),
        "future_stage_c_execution_authorized": False,
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
    parser.add_argument("--mechanics-protocol", type=Path, required=True)
    parser.add_argument("--source-preparation", type=Path, required=True)
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
            args.mechanics_protocol.resolve(),
            args.source_preparation.resolve(),
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
