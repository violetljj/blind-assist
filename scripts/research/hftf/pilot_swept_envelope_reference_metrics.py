#!/usr/bin/env python3
"""Pilot HFTF Stage B reference-relative metrics on consumed sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_swept_envelope_label_mechanics import (
    _swept_prism_counts,
    _swept_prism_probes_world,
)
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


SCHEMA = "blindassist_hftf_stage_b_reference_metric_pilot_result_d1"
PROTOCOL_SCHEMA = "blindassist_hftf_stage_b_reference_metric_pilot_d1"
READY_TERMINAL = "D1_REFERENCE_METRICS_READY_FOR_R3_GATE_FREEZE"
NOT_READY_TERMINAL = "D1_REFERENCE_METRICS_NOT_READY"
LAYERS = ("foot", "body", "head")
COUNT_FIELDS = (
    "tp",
    "fp",
    "fn",
    "tn",
    "candidate_only_correct",
    "baseline_only_correct",
)


def _pixel_lattices_disjoint(
    first_stride: int,
    first_offset: int,
    second_stride: int,
    second_offset: int,
) -> bool:
    if min(first_stride, second_stride) <= 0:
        raise ValueError("Pixel strides must be positive")
    if not 0 <= first_offset < first_stride:
        raise ValueError("First offset must fall inside its stride")
    if not 0 <= second_offset < second_stride:
        raise ValueError("Second offset must fall inside its stride")
    limit = int(np.lcm(first_stride, second_stride))
    first = set(range(first_offset, limit, first_stride))
    second = set(range(second_offset, limit, second_stride))
    return first.isdisjoint(second)


def _confusion(
    prediction: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> dict[str, int]:
    if prediction.shape != reference.shape or mask.shape != reference.shape:
        raise ValueError("Prediction, reference and mask shapes must match")
    prediction = prediction.astype(bool, copy=False)
    reference = reference.astype(bool, copy=False)
    mask = mask.astype(bool, copy=False)
    return {
        "tp": int((mask & prediction & reference).sum()),
        "fp": int((mask & prediction & ~reference).sum()),
        "fn": int((mask & ~prediction & reference).sum()),
        "tn": int((mask & ~prediction & ~reference).sum()),
    }


def _paired_correctness(
    candidate: np.ndarray,
    baseline: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> dict[str, int]:
    candidate_correct = candidate == reference
    baseline_correct = baseline == reference
    return {
        "candidate_only_correct": int(
            (mask & candidate_correct & ~baseline_correct).sum()
        ),
        "baseline_only_correct": int(
            (mask & ~candidate_correct & baseline_correct).sum()
        ),
    }


def _empty_counts() -> dict[str, int]:
    return {field: 0 for field in COUNT_FIELDS}


def _add_counts(
    target: dict[str, int], source: dict[str, int]
) -> None:
    for field, value in source.items():
        target[field] += int(value)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _summarize_counts(counts: dict[str, int]) -> dict[str, Any]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1_denominator = 2 * tp + fp + fn
    f1 = (
        float(2 * tp / f1_denominator)
        if f1_denominator
        else None
    )
    return {
        **counts,
        "positive_reference_cells": tp + fn,
        "negative_reference_cells": fp + tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": _safe_ratio(tp + tn, tp + fp + fn + tn),
    }


def _evaluate_arm(
    prediction: np.ndarray,
    baseline: np.ndarray,
    reference: np.ndarray,
    known: np.ndarray,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    candidate_counts: dict[str, dict[str, int]] = {}
    baseline_counts: dict[str, dict[str, int]] = {}
    for layer_index, layer_name in enumerate(LAYERS):
        mask = known[:, :, layer_index]
        candidate_layer = prediction[:, :, layer_index]
        baseline_layer = baseline[:, :, layer_index]
        reference_layer = reference[:, :, layer_index]
        candidate_item = _confusion(
            candidate_layer, reference_layer, mask
        )
        candidate_item.update(
            _paired_correctness(
                candidate_layer,
                baseline_layer,
                reference_layer,
                mask,
            )
        )
        baseline_item = _confusion(
            baseline_layer, reference_layer, mask
        )
        paired = _paired_correctness(
            candidate_layer,
            baseline_layer,
            reference_layer,
            mask,
        )
        baseline_item.update(
            {
                "candidate_only_correct": paired[
                    "candidate_only_correct"
                ],
                "baseline_only_correct": paired[
                    "baseline_only_correct"
                ],
            }
        )
        candidate_counts[layer_name] = candidate_item
        baseline_counts[layer_name] = baseline_item
    candidate_micro = _confusion(prediction, reference, known)
    candidate_micro.update(
        _paired_correctness(prediction, baseline, reference, known)
    )
    baseline_micro = _confusion(baseline, reference, known)
    paired_micro = _paired_correctness(
        prediction, baseline, reference, known
    )
    baseline_micro.update(paired_micro)
    candidate_counts["micro_all_layers"] = candidate_micro
    baseline_counts["micro_all_layers"] = baseline_micro
    return candidate_counts, baseline_counts


def _session_pilot(
    replay_root: Path,
    authority_path: Path,
    pilot: dict[str, Any],
    mechanics: dict[str, Any],
    r2: dict[str, Any],
) -> dict[str, Any]:
    replay_root = replay_root.resolve()
    rows = _load_jsonl(replay_root / "manifest.replay.jsonl")
    spec = _load_json(replay_root / "dataset_spec.json")
    session_id = str(rows[0]["session_id"])
    if session_id not in pilot["parent_sessions"]:
        raise ValueError(f"Unfrozen D1 source: {session_id}")
    expected_by_id = {
        item["source_session_id"]: item
        for item in r2["required_sessions"]
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
    field = mechanics["field"]
    theta_edges = _theta_edges(field)
    distance_edges = np.asarray(
        field["distance_edges_m"], dtype=np.float64
    )
    height_bands = [
        tuple(float(value) for value in field["height_bands_m"][name])
        for name in LAYERS
    ]
    envelope = mechanics["standard_synthetic_envelope"]
    widths = np.asarray(
        [
            envelope["effective_lateral_half_width_m"][name]
            for name in LAYERS
        ],
        dtype=np.float64,
    )
    obstacle = mechanics["obstacle_support"]
    known_contract = mechanics["known_support"]
    candidate_contract = pilot["candidate"]
    reference_contract = pilot["reference"]
    thresholds = [
        int(value)
        for value in reference_contract[
            "positive_count_threshold_sensitivity"
        ]
    ]
    accumulated: dict[int, dict[str, dict[str, dict[str, int]]]] = {
        threshold: {
            "candidate": {
                name: _empty_counts()
                for name in (*LAYERS, "micro_all_layers")
            },
            "baseline": {
                name: _empty_counts()
                for name in (*LAYERS, "micro_all_layers")
            },
        }
        for threshold in thresholds
    }
    known_counts = np.zeros(3, dtype=np.int64)
    candidate_baseline_disagreement = 0
    for row in rows:
        binding = binding_by_id[row["id"]]
        basis = _anchor_basis(binding, plane_by_id[row["id"]])
        candidate_points, candidate_dynamic = _obstacle_points_world(
            replay_root,
            row,
            binding,
            spec["camera"],
            stride=int(candidate_contract["point_sample_stride_xy"]),
            offset=int(candidate_contract["point_sample_offset_xy"]),
            excluded_classes=set(
                obstacle["excluded_semantic_class_ids"]
            ),
            dynamic_classes=set(
                obstacle["dynamic_provenance_class_ids"]
            ),
        )
        reference_points, reference_dynamic = _obstacle_points_world(
            replay_root,
            row,
            binding,
            spec["camera"],
            stride=int(reference_contract["point_sample_stride_xy"]),
            offset=int(reference_contract["point_sample_offset_xy"]),
            excluded_classes=set(
                obstacle["excluded_semantic_class_ids"]
            ),
            dynamic_classes=set(
                obstacle["dynamic_provenance_class_ids"]
            ),
        )
        candidate_counts, _ = _swept_prism_counts(
            candidate_points,
            candidate_dynamic,
            basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
        )
        baseline_counts, _ = _bin_obstacle_support(
            candidate_points,
            candidate_dynamic,
            basis,
            theta_edges,
            distance_edges,
            height_bands,
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
        known_counts += known.sum(axis=(0, 1))
        candidate_positive = candidate_counts > 0
        baseline_positive = baseline_counts > 0
        candidate_baseline_disagreement += int(
            (known & (candidate_positive != baseline_positive)).sum()
        )
        for threshold in thresholds:
            reference_positive = reference_counts >= threshold
            candidate_frame, baseline_frame = _evaluate_arm(
                candidate_positive,
                baseline_positive,
                reference_positive,
                known,
            )
            for arm_name, frame_counts in (
                ("candidate", candidate_frame),
                ("baseline", baseline_frame),
            ):
                for layer_name, counts in frame_counts.items():
                    _add_counts(
                        accumulated[threshold][arm_name][layer_name],
                        counts,
                    )
    required_per_height = (
        len(rows) * (len(theta_edges) - 1) * (len(distance_edges) - 1)
    )
    results_by_threshold: dict[str, Any] = {}
    for threshold in thresholds:
        threshold_result: dict[str, Any] = {}
        for arm_name in ("candidate", "baseline"):
            threshold_result[arm_name] = {
                layer_name: _summarize_counts(counts)
                for layer_name, counts in accumulated[threshold][
                    arm_name
                ].items()
            }
        results_by_threshold[str(threshold)] = threshold_result
    opportunities = {
        str(threshold): {
            "positive": results_by_threshold[str(threshold)][
                "candidate"
            ]["micro_all_layers"]["positive_reference_cells"],
            "negative": results_by_threshold[str(threshold)][
                "candidate"
            ]["micro_all_layers"]["negative_reference_cells"],
        }
        for threshold in thresholds
    }
    result.update(
        {
            "required_cells_per_height": required_per_height,
            "known_cells_by_height": {
                name: int(known_counts[index])
                for index, name in enumerate(LAYERS)
            },
            "known_coverage_by_height": {
                name: float(known_counts[index] / required_per_height)
                for index, name in enumerate(LAYERS)
            },
            "candidate_baseline_known_disagreement_cells": (
                candidate_baseline_disagreement
            ),
            "reference_opportunity_by_threshold": opportunities,
            "metrics_by_reference_count_threshold": (
                results_by_threshold
            ),
            "ok": validation["ok"],
        }
    )
    return result


def _cohort_metrics(
    sessions: list[dict[str, Any]], thresholds: list[int]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for threshold in thresholds:
        threshold_key = str(threshold)
        threshold_result: dict[str, Any] = {}
        for arm_name in ("candidate", "baseline"):
            arm_result: dict[str, Any] = {}
            for layer_name in (*LAYERS, "micro_all_layers"):
                counts = _empty_counts()
                for session in sessions:
                    source = session[
                        "metrics_by_reference_count_threshold"
                    ][threshold_key][arm_name][layer_name]
                    _add_counts(
                        counts,
                        {field: source[field] for field in COUNT_FIELDS},
                    )
                arm_result[layer_name] = _summarize_counts(counts)
            threshold_result[arm_name] = arm_result
        result[threshold_key] = threshold_result
    return result


def run(
    pilot_path: Path,
    mechanics_path: Path,
    r2_path: Path,
    session_inputs: list[tuple[Path, Path]],
) -> dict[str, Any]:
    pilot = _load_json(pilot_path)
    mechanics = _load_json(mechanics_path)
    r2 = _load_json(r2_path)
    if (
        pilot.get("schema") != PROTOCOL_SCHEMA
        or pilot.get("status")
        != "FROZEN_DEVELOPMENT_PILOT_RESULT_NOT_RUN"
    ):
        raise ValueError("D1 pilot protocol is not frozen")
    if len(session_inputs) != len(pilot["parent_sessions"]):
        raise ValueError("Expected exactly four consumed session inputs")
    candidate = pilot["candidate"]
    reference = pilot["reference"]
    lattice_disjoint = _pixel_lattices_disjoint(
        int(candidate["point_sample_stride_xy"]),
        int(candidate["point_sample_offset_xy"]),
        int(reference["point_sample_stride_xy"]),
        int(reference["point_sample_offset_xy"]),
    )
    sessions = [
        _session_pilot(
            replay,
            authority,
            pilot,
            mechanics,
            r2,
        )
        for replay, authority in session_inputs
    ]
    ids = [item["source_session_id"] for item in sessions]
    exact = (
        set(ids) == set(pilot["parent_sessions"])
        and len(ids) == len(set(ids))
    )
    thresholds = [
        int(value)
        for value in reference[
            "positive_count_threshold_sensitivity"
        ]
    ]
    opportunity_ready = all(
        all(
            session["reference_opportunity_by_threshold"][
                str(threshold)
            ]["positive"]
            > 0
            and session["reference_opportunity_by_threshold"][
                str(threshold)
            ]["negative"]
            > 0
            for threshold in thresholds
        )
        for session in sessions
    )
    arm_disagreement_ready = all(
        session["candidate_baseline_known_disagreement_cells"] > 0
        for session in sessions
    )
    sources_ready = exact and all(item.get("ok") for item in sessions)
    ready = (
        sources_ready
        and lattice_disjoint
        and opportunity_ready
        and arm_disagreement_ready
    )
    return {
        "schema": SCHEMA,
        "terminal": READY_TERMINAL if ready else NOT_READY_TERMINAL,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "evidence_role": "CONSUMED_METRIC_DESIGN_ONLY",
        "pilot_protocol_path": str(pilot_path.resolve()),
        "pilot_protocol_sha256": _sha256(pilot_path),
        "mechanics_protocol_path": str(mechanics_path.resolve()),
        "mechanics_protocol_sha256": _sha256(mechanics_path),
        "r2_protocol_path": str(r2_path.resolve()),
        "r2_protocol_sha256": _sha256(r2_path),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "candidate_reference_pixel_lattices_disjoint": (
            lattice_disjoint
        ),
        "exact_consumed_session_set": exact,
        "sessions": sessions,
        "cohort_metrics_by_reference_count_threshold": _cohort_metrics(
            sessions, thresholds
        ),
        "readiness_checks": {
            "all_four_consumed_sources_decode_and_bind": sources_ready,
            "candidate_and_reference_pixel_lattices_are_disjoint": (
                lattice_disjoint
            ),
            "positive_and_negative_reference_opportunity_per_session_at_every_threshold": (
                opportunity_ready
            ),
            "candidate_and_baseline_differ_per_session": (
                arm_disagreement_ready
            ),
        },
        "primary_threshold_selected": False,
        "formal_r3_gate_frozen": False,
        "fresh_source_acquisition_authorized_by_this_result": False,
        "future_stage_c_authorized": False,
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
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--mechanics-protocol", type=Path, required=True)
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
            args.pilot.resolve(),
            args.mechanics_protocol.resolve(),
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
