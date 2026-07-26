from __future__ import annotations

from collections import Counter, defaultdict
import time
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from scripts.research.egomotion_compensated_looming.rcle_minimal import (
    evaluation as r0_evaluation,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.protocol import (
    TrialSpec,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.rotation_compensation import (
    compensate_current_to_previous,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.synthetic_generator import (
    generate_sequence,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal_r1.local_expansion import (
    LocalExpansionResult,
    fit_fixed_grid_local_affine,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal_r1.sparse_flow import (
    SparseTrackResult,
    detect_fixed_grid_features,
    track_features,
)

from .support_manager import (
    OBSERVABLE_OCCLUSION,
    ObservableTrackDiagnostics,
    activated_cell_indices,
    classify_prior_survivors,
    merge_path_correspondences,
    observable_occlusion_centers,
    select_spatial_supplements,
    track_observable_points,
)


IMPLEMENTATION_REVISION = "OBSERVABLE_THREE_FRAME_SUPPORT_MANAGER_R0"
_WARP_OVERLAP_FLOOR = 0.75


def _empty_diagnostics() -> ObservableTrackDiagnostics:
    points = np.empty((0, 2), dtype=np.float32)
    booleans = np.empty((0,), dtype=bool)
    errors = np.empty((0,), dtype=np.float32)
    return ObservableTrackDiagnostics(
        initial_points=points,
        forward_points=points.copy(),
        forward_available=booleans,
        forward_backward_errors=errors,
        forward_backward_pass=booleans.copy(),
        source_patch_valid=booleans.copy(),
        target_patch_valid=booleans.copy(),
        photometric_errors=errors.copy(),
        photometric_pass=booleans.copy(),
        accepted=booleans.copy(),
    )


def _select_diagnostics(
    diagnostics: ObservableTrackDiagnostics, selected: np.ndarray
) -> ObservableTrackDiagnostics:
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    if selected.shape != (diagnostics.requested_count,):
        raise ValueError("DIAGNOSTIC_SELECTION_SHAPE_MISMATCH")
    return ObservableTrackDiagnostics(
        initial_points=np.ascontiguousarray(
            diagnostics.initial_points[selected]
        ),
        forward_points=np.ascontiguousarray(
            diagnostics.forward_points[selected]
        ),
        forward_available=np.ascontiguousarray(
            diagnostics.forward_available[selected]
        ),
        forward_backward_errors=np.ascontiguousarray(
            diagnostics.forward_backward_errors[selected]
        ),
        forward_backward_pass=np.ascontiguousarray(
            diagnostics.forward_backward_pass[selected]
        ),
        source_patch_valid=np.ascontiguousarray(
            diagnostics.source_patch_valid[selected]
        ),
        target_patch_valid=np.ascontiguousarray(
            diagnostics.target_patch_valid[selected]
        ),
        photometric_errors=np.ascontiguousarray(
            diagnostics.photometric_errors[selected]
        ),
        photometric_pass=np.ascontiguousarray(
            diagnostics.photometric_pass[selected]
        ),
        accepted=np.ascontiguousarray(diagnostics.accepted[selected]),
    )


def _points_in_cells(
    points: np.ndarray,
    cells: Sequence[LocalExpansionResult],
    indices: Sequence[int],
) -> np.ndarray:
    selected = np.zeros(points.shape[0], dtype=bool)
    for index in indices:
        x0, y0, x1, y1 = cells[index].region
        selected |= (
            (points[:, 0] >= x0)
            & (points[:, 0] < x1)
            & (points[:, 1] >= y0)
            & (points[:, 1] < y1)
        )
    return selected


def _baseline_source_union(
    initial_points: np.ndarray,
    raw_tracks: SparseTrackResult,
    compensated_tracks: SparseTrackResult,
) -> np.ndarray:
    """Return path-accepted baseline sources in detector order."""

    accepted = {
        tuple(float(value) for value in point)
        for point in np.vstack(
            (raw_tracks.previous_points, compensated_tracks.previous_points)
        )
    }
    return np.ascontiguousarray(
        np.asarray(
            [
                point
                for point in initial_points
                if tuple(float(value) for value in point) in accepted
            ],
            dtype=np.float32,
        ).reshape(-1, 2)
    )


def _admit_shared_carried_points(
    raw: ObservableTrackDiagnostics,
    compensated: ObservableTrackDiagnostics,
    baseline_pool: np.ndarray,
    cells: Sequence[LocalExpansionResult],
    activated: Sequence[int],
) -> np.ndarray:
    """Apply the shared baseline->carry pool order, 5 px spacing and cell cap."""

    if raw.requested_count != compensated.requested_count or not np.allclose(
        raw.initial_points, compensated.initial_points
    ):
        raise ValueError("RAW_COMPENSATED_CARRY_ALIGNMENT_MISMATCH")
    in_activated = _points_in_cells(raw.initial_points, cells, activated)
    eligible = in_activated & (raw.accepted | compensated.accepted)
    admitted = np.zeros(raw.requested_count, dtype=bool)
    occupied = np.ascontiguousarray(baseline_pool.astype(np.float32))
    for index, point in enumerate(raw.initial_points):
        if not eligible[index]:
            continue
        cell_index = next(
            (
                candidate
                for candidate in activated
                if _points_in_cells(
                    point.reshape(1, 2), cells, (candidate,)
                )[0]
            ),
            None,
        )
        if cell_index is None:
            continue
        x0, y0, x1, y1 = cells[cell_index].region
        in_cell = (
            (occupied[:, 0] >= x0)
            & (occupied[:, 0] < x1)
            & (occupied[:, 1] >= y0)
            & (occupied[:, 1] < y1)
        )
        if int(np.count_nonzero(in_cell)) >= 80:
            continue
        if occupied.size:
            squared = np.sum(
                (occupied.astype(np.float64) - point.astype(np.float64)) ** 2,
                axis=1,
            )
            if bool(np.any(squared < 25.0)):
                continue
        admitted[index] = True
        occupied = np.vstack((occupied, point.reshape(1, 2)))
    return admitted


def _splice_activated_cells(
    baseline: Sequence[LocalExpansionResult],
    managed: Sequence[LocalExpansionResult],
    activated: Sequence[int],
) -> list[LocalExpansionResult]:
    activated_set = set(activated)
    return [
        managed[index] if index in activated_set else baseline[index]
        for index in range(len(baseline))
    ]


def _accepted_prior_tracks(
    diagnostics: Iterable[ObservableTrackDiagnostics],
) -> SparseTrackResult:
    """Retain observable tracks in fixed point-class order for the next pair."""

    previous: list[np.ndarray] = []
    current: list[np.ndarray] = []
    errors: list[float] = []
    for item in diagnostics:
        for source, target, error in zip(
            item.initial_points[item.accepted],
            item.forward_points[item.accepted],
            item.forward_backward_errors[item.accepted],
            strict=True,
        ):
            if previous:
                occupied = np.vstack(previous).astype(np.float64)
                squared = np.sum(
                    (occupied - source.astype(np.float64)) ** 2, axis=1
                )
                if bool(np.any(squared < 25.0)):
                    continue
            previous.append(np.asarray(source, dtype=np.float32))
            current.append(np.asarray(target, dtype=np.float32))
            errors.append(float(error))
    if not previous:
        return SparseTrackResult(
            previous_points=np.empty((0, 2), dtype=np.float32),
            current_points=np.empty((0, 2), dtype=np.float32),
            forward_backward_errors=np.empty((0,), dtype=np.float32),
            requested_count=0,
        )
    return SparseTrackResult(
        previous_points=np.ascontiguousarray(
            np.vstack(previous).astype(np.float32)
        ),
        current_points=np.ascontiguousarray(
            np.vstack(current).astype(np.float32)
        ),
        forward_backward_errors=np.ascontiguousarray(
            np.asarray(errors, dtype=np.float32)
        ),
        requested_count=sum(item.requested_count for item in diagnostics),
    )


def _finish_trial_result(
    spec: TrialSpec,
    protocol: dict[str, Any],
    pair_total: int,
    pair_trace: list[dict[str, Any]],
    abstentions: Counter[str],
    sequence: Any,
) -> dict[str, Any]:
    evaluable_pairs = [row for row in pair_trace if row["evaluable"]]
    minimum_pair_fraction = float(
        protocol["local_affine"]["minimum_evaluable_pair_fraction_per_trial"]
    )
    pair_fraction = len(evaluable_pairs) / max(pair_total, 1)
    trial_evaluable = pair_fraction >= minimum_pair_fraction
    if not trial_evaluable:
        abstentions["EVALUABLE_PAIR_FRACTION_BELOW_0_80"] += 1

    result: dict[str, Any] = {
        **spec.to_dict(),
        "implementation_revision": IMPLEMENTATION_REVISION,
        "planned_pair_count": pair_total,
        "evaluable_pair_count": len(evaluable_pairs),
        "evaluable_pair_fraction": pair_fraction,
        "evaluable": trial_evaluable,
        "abstention_reason": (
            None
            if trial_evaluable
            else "EVALUABLE_PAIR_FRACTION_BELOW_0_80"
        ),
        "truth_scale_rate_per_s": spec.scale_rate_per_s,
        "base_sha256": sequence.base_sha256,
        "sequence_sha256": sequence.sequence_sha256,
        "abstention_counts": dict(sorted(abstentions.items())),
        "pair_trace": pair_trace,
    }
    metric_keys = (
        "raw_rotation_leakage_per_s",
        "compensated_rotation_leakage_per_s",
        "paired_leakage_reduction_per_s",
        "raw_closing_estimate_per_s",
        "compensated_closing_estimate_per_s",
        "raw_closing_error_per_s",
        "compensated_closing_error_per_s",
        "compensated_minus_raw_closing_error_per_s",
        "raw_sign_correct",
        "compensated_sign_correct",
        "rsr",
        "crr",
    )
    if not trial_evaluable:
        result.update({key: None for key in metric_keys})
        result["rsr_status"] = "NOT_EVALUABLE_TRIAL"
        result["crr_status"] = "NOT_EVALUABLE_TRIAL"
        return result

    raw_leakage = float(
        np.median(
            [row["raw_abs_expansion_median_per_s"] for row in evaluable_pairs]
        )
    )
    compensated_leakage = float(
        np.median(
            [
                row["compensated_abs_expansion_median_per_s"]
                for row in evaluable_pairs
            ]
        )
    )
    raw_estimate = float(
        np.median(
            [row["raw_expansion_median_per_s"] for row in evaluable_pairs]
        )
    )
    compensated_estimate = float(
        np.median(
            [
                row["compensated_expansion_median_per_s"]
                for row in evaluable_pairs
            ]
        )
    )
    raw_error = abs(raw_estimate - spec.scale_rate_per_s)
    compensated_error = abs(compensated_estimate - spec.scale_rate_per_s)
    result.update(
        {
            "raw_rotation_leakage_per_s": raw_leakage,
            "compensated_rotation_leakage_per_s": compensated_leakage,
            "paired_leakage_reduction_per_s": (
                raw_leakage - compensated_leakage
            ),
            "raw_closing_estimate_per_s": raw_estimate,
            "compensated_closing_estimate_per_s": compensated_estimate,
            "raw_closing_error_per_s": raw_error,
            "compensated_closing_error_per_s": compensated_error,
            "compensated_minus_raw_closing_error_per_s": (
                compensated_error - raw_error
            ),
        }
    )
    zero_band = float(protocol["metrics"]["sign_accuracy_zero_band_per_s"])
    if spec.scale_rate_per_s != 0.0:
        result["raw_sign_correct"] = r0_evaluation._sign_correct(
            raw_estimate, spec.scale_rate_per_s, zero_band
        )
        result["compensated_sign_correct"] = r0_evaluation._sign_correct(
            compensated_estimate, spec.scale_rate_per_s, zero_band
        )
    else:
        result["raw_sign_correct"] = None
        result["compensated_sign_correct"] = None

    rsr_floor = float(protocol["metrics"]["rsr_denominator_floor_per_s"])
    if spec.motion_family == "pure_rotation" and raw_leakage >= rsr_floor:
        result["rsr"] = 1.0 - compensated_leakage / raw_leakage
        result["rsr_status"] = "EVALUABLE"
    elif spec.motion_family == "pure_rotation":
        result["rsr"] = None
        result["rsr_status"] = "NOT_EVALUABLE_DENOMINATOR_FLOOR"
    else:
        result["rsr"] = None
        result["rsr_status"] = "NOT_APPLICABLE"

    crr_floor = float(protocol["metrics"]["crr_denominator_floor_per_s"])
    if spec.scale_rate_per_s > 0.0 and raw_estimate >= crr_floor:
        result["crr"] = compensated_estimate / raw_estimate
        result["crr_status"] = "EVALUABLE"
    elif spec.scale_rate_per_s > 0.0:
        result["crr"] = None
        result["crr_status"] = "NOT_EVALUABLE_DENOMINATOR_FLOOR"
    else:
        result["crr"] = None
        result["crr_status"] = "NOT_APPLICABLE"
    return result


def run_trial(
    spec: TrialSpec,
    protocol: dict[str, Any],
    include_cell_details: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the locked candidate with an explicit, observable three-frame loop."""

    cv2.setRNGSeed(int(spec.seed % (2**31 - 1)))
    cv2.setNumThreads(1)
    sequence_started = time.perf_counter_ns()
    sequence = generate_sequence(spec, protocol)
    sequence_elapsed = time.perf_counter_ns() - sequence_started

    pair_total = len(sequence.frames) - 1
    lk_parameters = protocol["sparse_lk"]
    affine_parameters = protocol["local_affine"]
    minimum_common = int(
        affine_parameters["minimum_common_evaluable_cells_per_pair"]
    )
    pair_trace: list[dict[str, Any]] = []
    abstentions: Counter[str] = Counter()
    timings_ns: defaultdict[str, int] = defaultdict(int)
    timings_ns["synthetic_generation"] = sequence_elapsed

    prior_survivors: SparseTrackResult | None = None
    prior_dt: float | None = None

    for pair_index in range(pair_total):
        previous = sequence.frames[pair_index]
        current = sequence.frames[pair_index + 1]
        previous_valid = sequence.valid_masks[pair_index]
        current_valid = sequence.valid_masks[pair_index + 1]
        image_bounds_valid = np.full(previous.shape, 255, dtype=np.uint8)
        dt = (
            sequence.timestamps_seconds[pair_index + 1]
            - sequence.timestamps_seconds[pair_index]
        )
        if not np.isfinite(dt) or dt <= 0.0:
            prior_survivors = None
            prior_dt = None
            abstentions["NON_POSITIVE_OR_MISSING_DT"] += 1
            pair_trace.append(
                {
                    "pair_index": pair_index,
                    "timestamp_seconds": sequence.timestamps_seconds[
                        pair_index + 1
                    ],
                    "evaluable": False,
                    "reason": "NON_POSITIVE_OR_MISSING_DT",
                }
            )
            continue

        started = time.perf_counter_ns()
        compensation = compensate_current_to_previous(
            current,
            current_valid,
            previous_valid,
            sequence.rotation_homography_previous_to_current,
        )
        timings_ns["rotation_warp"] += time.perf_counter_ns() - started
        if compensation.overlap_fraction < _WARP_OVERLAP_FLOOR:
            prior_survivors = None
            prior_dt = None
            abstentions["ROTATION_WARP_VALID_COVERAGE_BELOW_0_75"] += 1
            pair_trace.append(
                {
                    "pair_index": pair_index,
                    "timestamp_seconds": sequence.timestamps_seconds[
                        pair_index + 1
                    ],
                    "evaluable": False,
                    "reason": "ROTATION_WARP_VALID_COVERAGE_BELOW_0_75",
                    "warp_overlap_fraction": compensation.overlap_fraction,
                }
            )
            continue

        started = time.perf_counter_ns()
        initial_points = detect_fixed_grid_features(
            previous, np.ascontiguousarray(previous_valid), lk_parameters
        )
        raw_tracks = track_features(
            previous,
            current,
            initial_points,
            current_valid,
            lk_parameters,
        )
        compensated_tracks = track_features(
            previous,
            compensation.image,
            initial_points,
            compensation.valid_mask,
            lk_parameters,
        )
        timings_ns["sparse_lk"] += time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        raw_cells = fit_fixed_grid_local_affine(
            raw_tracks, dt, previous.shape, affine_parameters
        )
        compensated_cells = fit_fixed_grid_local_affine(
            compensated_tracks, dt, previous.shape, affine_parameters
        )
        timings_ns["local_affine"] += time.perf_counter_ns() - started

        manager_started = time.perf_counter_ns()
        baseline_observable = track_observable_points(
            previous,
            current,
            initial_points,
            image_bounds_valid,
            image_bounds_valid,
            lk_parameters,
        )
        activated: tuple[int, ...] = ()
        raw_carry = _empty_diagnostics()
        compensated_carry = _empty_diagnostics()
        raw_supplements = _empty_diagnostics()
        compensated_supplements = _empty_diagnostics()
        raw_classifications = np.empty((0,), dtype=object)
        compensated_classifications = np.empty((0,), dtype=object)
        supplement_points = np.empty((0, 2), dtype=np.float32)
        baseline_pool = _baseline_source_union(
            initial_points, raw_tracks, compensated_tracks
        )

        # Pair zero is deliberately left as the unchanged R1 baseline. It only
        # establishes observable raw lifecycle for the next pair.
        if pair_index > 0:
            activated = activated_cell_indices(raw_cells, compensated_cells)
            if activated and prior_survivors is not None and prior_dt is not None:
                raw_carry_all = track_observable_points(
                    previous,
                    current,
                    prior_survivors.current_points,
                    image_bounds_valid,
                    image_bounds_valid,
                    lk_parameters,
                )
                compensated_carry_all = track_observable_points(
                    previous,
                    compensation.image,
                    prior_survivors.current_points,
                    image_bounds_valid,
                    compensation.valid_mask,
                    lk_parameters,
                )
                raw_classifications = classify_prior_survivors(
                    prior_survivors.current_points,
                    (
                        prior_survivors.current_points
                        - prior_survivors.previous_points
                    ),
                    raw_carry_all,
                    image_bounds_valid,
                    prior_dt_seconds=prior_dt,
                    current_dt_seconds=dt,
                )
                compensated_classifications = classify_prior_survivors(
                    prior_survivors.current_points,
                    (
                        prior_survivors.current_points
                        - prior_survivors.previous_points
                    ),
                    compensated_carry_all,
                    compensation.valid_mask,
                    prior_dt_seconds=prior_dt,
                    current_dt_seconds=dt,
                )
                carry_admitted = _admit_shared_carried_points(
                    raw_carry_all,
                    compensated_carry_all,
                    baseline_pool,
                    raw_cells,
                    activated,
                )
                raw_carry = _select_diagnostics(
                    raw_carry_all, carry_admitted
                )
                compensated_carry = _select_diagnostics(
                    compensated_carry_all, carry_admitted
                )
                exclusion_classes = np.where(
                    (raw_classifications == OBSERVABLE_OCCLUSION)
                    | (
                        compensated_classifications
                        == OBSERVABLE_OCCLUSION
                    ),
                    OBSERVABLE_OCCLUSION,
                    "",
                )
                exclusions = observable_occlusion_centers(
                    prior_survivors.current_points, exclusion_classes
                )
            else:
                exclusions = np.empty((0, 2), dtype=np.float32)

            if activated:
                existing = np.vstack(
                    (
                        baseline_pool,
                        raw_carry.initial_points,
                    )
                )
                selected: list[np.ndarray] = []
                for cell_index in activated:
                    points = select_spatial_supplements(
                        previous,
                        image_bounds_valid,
                        raw_cells[cell_index].region,
                        (
                            np.vstack((existing, *selected))
                            if selected
                            else existing
                        ),
                        exclusions,
                    )
                    if points.size:
                        selected.append(points)
                if selected:
                    supplement_points = np.ascontiguousarray(
                        np.vstack(selected).astype(np.float32)
                    )
                raw_supplements = track_observable_points(
                    previous,
                    current,
                    supplement_points,
                    image_bounds_valid,
                    image_bounds_valid,
                    lk_parameters,
                )
                compensated_supplements = track_observable_points(
                    previous,
                    compensation.image,
                    supplement_points,
                    image_bounds_valid,
                    compensation.valid_mask,
                    lk_parameters,
                )
                managed_raw_tracks = merge_path_correspondences(
                    raw_tracks, raw_carry, raw_supplements
                )
                managed_compensated_tracks = merge_path_correspondences(
                    compensated_tracks,
                    compensated_carry,
                    compensated_supplements,
                )
                managed_raw_cells = fit_fixed_grid_local_affine(
                    managed_raw_tracks, dt, previous.shape, affine_parameters
                )
                managed_compensated_cells = fit_fixed_grid_local_affine(
                    managed_compensated_tracks,
                    dt,
                    previous.shape,
                    affine_parameters,
                )
                raw_cells = _splice_activated_cells(
                    raw_cells, managed_raw_cells, activated
                )
                compensated_cells = _splice_activated_cells(
                    compensated_cells,
                    managed_compensated_cells,
                    activated,
                )
                raw_tracks = managed_raw_tracks
                compensated_tracks = managed_compensated_tracks

        prior_survivors = _accepted_prior_tracks(
            (baseline_observable, raw_carry, raw_supplements)
        )
        prior_dt = dt
        timings_ns["observable_support_manager"] += (
            time.perf_counter_ns() - manager_started
        )

        raw_values, compensated_values, common_indices = (
            r0_evaluation._common_cell_expansions(
                raw_cells, compensated_cells
            )
        )
        manager_trace = {
            "candidate_id": IMPLEMENTATION_REVISION,
            "baseline_only": pair_index == 0,
            "activated_cell_indices": list(activated),
            "prior_survivor_count": (
                0 if pair_index == 0 else raw_carry.requested_count
            ),
            "raw_observable_occlusion_count": int(
                np.count_nonzero(
                    raw_classifications == OBSERVABLE_OCCLUSION
                )
            ),
            "compensated_observable_occlusion_count": int(
                np.count_nonzero(
                    compensated_classifications == OBSERVABLE_OCCLUSION
                )
            ),
            "spatial_supplement_count": int(supplement_points.shape[0]),
            "raw_accepted_carry_count": raw_carry.accepted_count,
            "compensated_accepted_carry_count": (
                compensated_carry.accepted_count
            ),
            "raw_accepted_supplement_count": raw_supplements.accepted_count,
            "compensated_accepted_supplement_count": (
                compensated_supplements.accepted_count
            ),
        }
        if len(common_indices) < minimum_common:
            abstentions["COMMON_GRID_SUPPORT_BELOW_5_OF_9"] += 1
            for cell in raw_cells:
                if not cell.evaluable and cell.abstention_reason:
                    abstentions[f"raw:{cell.abstention_reason}"] += 1
            for cell in compensated_cells:
                if not cell.evaluable and cell.abstention_reason:
                    abstentions[f"comp:{cell.abstention_reason}"] += 1
            pair_row: dict[str, Any] = {
                "pair_index": pair_index,
                "timestamp_seconds": sequence.timestamps_seconds[
                    pair_index + 1
                ],
                "evaluable": False,
                "reason": "COMMON_GRID_SUPPORT_BELOW_5_OF_9",
                "common_cell_count": len(common_indices),
                "raw_track_count": raw_tracks.valid_count,
                "compensated_track_count": compensated_tracks.valid_count,
                "warp_overlap_fraction": compensation.overlap_fraction,
                "support_manager": manager_trace,
            }
        else:
            pair_row = {
                "pair_index": pair_index,
                "timestamp_seconds": sequence.timestamps_seconds[
                    pair_index + 1
                ],
                "evaluable": True,
                "common_cell_count": len(common_indices),
                "raw_expansion_median_per_s": float(np.median(raw_values)),
                "compensated_expansion_median_per_s": float(
                    np.median(compensated_values)
                ),
                "raw_abs_expansion_median_per_s": float(
                    np.median(np.abs(raw_values))
                ),
                "compensated_abs_expansion_median_per_s": float(
                    np.median(np.abs(compensated_values))
                ),
                "raw_track_count": raw_tracks.valid_count,
                "compensated_track_count": compensated_tracks.valid_count,
                "warp_overlap_fraction": compensation.overlap_fraction,
                "common_cell_indices": common_indices,
                "support_manager": manager_trace,
            }
        if include_cell_details:
            pair_row["raw_cells"] = [cell.to_dict() for cell in raw_cells]
            pair_row["compensated_cells"] = [
                cell.to_dict() for cell in compensated_cells
            ]
        pair_trace.append(pair_row)

    result = _finish_trial_result(
        spec, protocol, pair_total, pair_trace, abstentions, sequence
    )
    runtime = {
        "trial_id": spec.trial_id,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "pair_count": pair_total,
        "module_total_milliseconds": {
            key: value / 1_000_000.0 for key, value in sorted(timings_ns.items())
        },
        "total_milliseconds": sum(timings_ns.values()) / 1_000_000.0,
    }
    return result, runtime


summarize_and_decide = r0_evaluation.summarize_and_decide
wilson_interval = r0_evaluation.wilson_interval

__all__ = [
    "IMPLEMENTATION_REVISION",
    "run_trial",
    "summarize_and_decide",
    "wilson_interval",
]
