from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROTOCOL_ID = "RCLE_TEMPORAL_STRUCTURE_DIAGNOSTIC_R1"
CONTRACT_SCHEMA = "rcle.temporal_structure_diagnostic.contract.v1"
ANALYSIS_SCHEMA = "rcle.temporal_structure_diagnostic.analysis.v1"
VALIDATION_SCHEMA = "rcle.temporal_structure_diagnostic.validation.v1"
DIRECTION_SUMMARY_SCHEMA = (
    "rcle.temporal_structure_diagnostic.direction_summary.v1"
)
SESSIONS = (13, 14, 15, 17)
SEALED_SESSION = 16
PAIR_COUNT = 601
GRID_HZ = 60.0
POSE_BAND_HZ = (0.7, 3.0)
ENERGY_DENOMINATOR_HZ = (0.1, 10.0)
TAIL_FRACTION = 0.2
FLOAT_REL_TOL = 1e-9
FLOAT_ABS_TOL = 1e-12

DIRECTION_ROW_FLAGS = {
    "stage_1_response_blind": True,
    "response_accessed_during_extraction": False,
    "r3_pair_ledger_accessed": False,
    "r0_proxy_ledger_accessed": False,
    "risk_or_obstacle_label_accessed": False,
    "manual_gait_phase_accessed": False,
    "sealed_session_accessed": False,
}
PROXY_ROW_FLAGS = {
    "response_accessed_during_extraction": False,
    "risk_label_accessed": False,
}
ANALYSIS_FLAGS = {
    "risk_label_accessed": False,
    "obstacle_label_accessed": False,
    "manual_gait_label_accessed": False,
    "sealed_session_accessed": False,
}
FORBIDDEN_LABEL_KEYS = {
    "risk_label",
    "obstacle_label",
    "manual_gait_label",
    "manual_gait_phase",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalized_average_ranks(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("NONFINITE_RANK_INPUT")
    order = np.argsort(array, kind="mergesort")
    result = np.empty(len(array), dtype=np.float64)
    cursor = 0
    while cursor < len(array):
        stop = cursor + 1
        while stop < len(array) and array[order[stop]] == array[order[cursor]]:
            stop += 1
        result[order[cursor:stop]] = (cursor + stop - 1) / 2.0
        cursor = stop
    return result / max(1, len(array) - 1)


def rolling_center_median(values: np.ndarray, window: int) -> np.ndarray:
    radius = window // 2
    return np.asarray(
        [
            np.median(values[max(0, i - radius) : min(len(values), i + radius + 1)])
            for i in range(len(values))
        ],
        dtype=np.float64,
    )


def divide_or_none(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def maximum_true_streak(mask: np.ndarray) -> int:
    best = 0
    active = 0
    for item in np.asarray(mask, dtype=bool):
        active = active + 1 if item else 0
        best = max(best, active)
    return best


def contiguous_true_lengths(mask: np.ndarray) -> list[int]:
    lengths: list[int] = []
    active = 0
    for item in np.asarray(mask, dtype=bool):
        if item:
            active += 1
        elif active:
            lengths.append(active)
            active = 0
    if active:
        lengths.append(active)
    return lengths


def response_top_tail(r3_rows: list[dict[str, Any]]) -> np.ndarray:
    eligible = np.asarray(
        [
            row.get("evaluable") is True
            and row.get("compensated_expansion_median_per_s") is not None
            for row in r3_rows
        ],
        dtype=bool,
    )
    mask = np.zeros(len(r3_rows), dtype=bool)
    indices = np.flatnonzero(eligible)
    magnitudes = [
        abs(float(r3_rows[index]["compensated_expansion_median_per_s"]))
        for index in indices
    ]
    mask[indices] = normalized_average_ranks(magnitudes) >= 1.0 - TAIL_FRACTION
    return mask


def pose_spectrum(
    timestamps: np.ndarray, values: np.ndarray
) -> dict[str, Any]:
    if (
        len(timestamps) != len(values)
        or len(values) < 4
        or not np.isfinite(timestamps).all()
        or not np.isfinite(values).all()
        or np.any(np.diff(timestamps) <= 0.0)
    ):
        raise ValueError("POSE_SPECTRUM_INPUT")
    step = 1.0 / GRID_HZ
    sample_count = int(
        math.floor((timestamps[-1] - timestamps[0]) / step)
    ) + 1
    grid = timestamps[0] + np.arange(sample_count, dtype=np.float64) * step
    uniform = np.interp(grid, timestamps, values)
    detrended = uniform - rolling_center_median(uniform, 61)
    transform = np.fft.rfft(detrended)
    frequencies = np.fft.rfftfreq(len(detrended), d=step)
    power = np.abs(transform) ** 2
    denominator_mask = (
        (frequencies >= ENERGY_DENOMINATOR_HZ[0])
        & (frequencies <= ENERGY_DENOMINATOR_HZ[1])
    )
    band_mask = (
        (frequencies >= POSE_BAND_HZ[0])
        & (frequencies <= POSE_BAND_HZ[1])
    )
    denominator_energy = float(np.sum(power[denominator_mask]))
    band_energy = float(np.sum(power[band_mask]))
    band_indices = np.flatnonzero(band_mask)
    dominant = (
        None
        if len(band_indices) == 0 or band_energy == 0.0
        else float(
            frequencies[
                band_indices[int(np.argmax(power[band_indices]))]
            ]
        )
    )
    isolated = np.zeros_like(transform)
    isolated[band_mask] = transform[band_mask]
    return {
        "grid": grid,
        "bandpassed": np.fft.irfft(isolated, n=len(detrended)),
        "band_energy_fraction": divide_or_none(
            band_energy, denominator_energy
        ),
        "dominant_frequency_hz": dominant,
    }


def strongest_pose_component(
    timestamps: np.ndarray, rows: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    fields = (
        ("angular_x", "camera_angular_velocity_x_deg_per_s"),
        ("angular_y", "camera_angular_velocity_y_deg_per_s"),
        ("angular_z", "camera_angular_velocity_z_deg_per_s"),
        ("translation_x", "camera_translation_velocity_x_m_per_s"),
        ("translation_y", "camera_translation_velocity_y_m_per_s"),
        ("translation_z", "camera_translation_velocity_z_m_per_s"),
    )
    candidates = [
        (
            name,
            pose_spectrum(
                timestamps,
                np.asarray([float(row[field]) for row in rows]),
            ),
        )
        for name, field in fields
    ]
    return max(
        candidates,
        key=lambda item: (
            -1.0
            if item[1]["band_energy_fraction"] is None
            else item[1]["band_energy_fraction"],
            item[0],
        ),
    )


def upward_zero_crossings(
    timestamps: np.ndarray, values: np.ndarray
) -> list[float]:
    crossings: list[float] = []
    for index in range(1, len(values)):
        left = float(values[index - 1])
        right = float(values[index])
        if left < 0.0 <= right and right != left:
            fraction = -left / (right - left)
            crossings.append(
                float(
                    timestamps[index - 1]
                    + fraction
                    * (timestamps[index] - timestamps[index - 1])
                )
            )
    return crossings


def describe_cycles(
    grid: np.ndarray,
    bandpassed: np.ndarray,
    pair_timestamps: np.ndarray,
    evaluable: np.ndarray,
    high_response: np.ndarray,
    response_magnitude: np.ndarray,
) -> dict[str, Any]:
    cycles: list[dict[str, Any]] = []
    crossings = upward_zero_crossings(grid, bandpassed)
    longest_consecutive = 0
    current_consecutive = 0
    for start, end in zip(crossings, crossings[1:]):
        period = end - start
        if not 1.0 / POSE_BAND_HZ[1] <= period <= 1.0 / POSE_BAND_HZ[0]:
            current_consecutive = 0
            continue
        members = (pair_timestamps >= start) & (pair_timestamps < end)
        if not np.any(members):
            current_consecutive = 0
            continue
        coverage = float(np.mean(evaluable[members]))
        if coverage < 0.8:
            current_consecutive = 0
            continue
        eligible_indices = np.flatnonzero(members & evaluable)
        peak = eligible_indices[
            int(np.argmax(response_magnitude[eligible_indices]))
        ]
        contains_high_response = bool(np.any(high_response[members]))
        current_consecutive = (
            current_consecutive + 1 if contains_high_response else 0
        )
        longest_consecutive = max(longest_consecutive, current_consecutive)
        cycles.append(
            {
                "start_s": start,
                "end_s": end,
                "period_s": period,
                "evaluable_coverage": coverage,
                "contains_high_response": contains_high_response,
                "cycle_max_response_phase_rad": 2.0
                * math.pi
                * ((pair_timestamps[peak] - start) / period),
            }
        )
    contains = np.asarray(
        [cycle["contains_high_response"] for cycle in cycles], dtype=bool
    )
    phases = np.asarray(
        [cycle["cycle_max_response_phase_rad"] for cycle in cycles],
        dtype=np.float64,
    )
    return {
        "valid_cycle_count": len(cycles),
        "cycles": cycles,
        "high_response_cycle_fraction": (
            float(np.mean(contains)) if len(contains) else None
        ),
        "longest_consecutive_high_response_cycles": longest_consecutive,
        "cycle_max_absolute_response_axial_phase_locking_value": (
            float(abs(np.mean(np.exp(2j * phases))))
            if len(phases)
            else None
        ),
    }


def independently_direction_evaluable(row: dict[str, Any]) -> bool:
    median_error = row.get("median_forward_backward_error_px")
    spatial_resultant = row.get("spatial_direction_resultant")
    radial_consistency = row.get("radial_direction_consistency")
    direction_structure = bool(
        (
            spatial_resultant is not None
            and float(spatial_resultant) >= 0.5
        )
        or (
            int(row["radial_direction_track_count"]) >= 60
            and radial_consistency is not None
            and float(radial_consistency) >= 0.5
        )
    )
    return bool(
        int(row["direction_track_count"]) >= 60
        and float(row["forward_backward_consistent_fraction"]) >= 0.5
        and median_error is not None
        and float(median_error) <= 0.75
        and direction_structure
    )


def principal_axis(vectors: np.ndarray) -> np.ndarray:
    centered = vectors - np.mean(vectors, axis=0)
    covariance = centered.T @ centered / max(1, len(centered))
    _, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, -1]
    first_nonzero = next(
        (float(value) for value in axis if value != 0.0), 0.0
    )
    if first_nonzero < 0.0:
        axis = -axis
    return axis


def periodic_fit_r_squared(
    timestamps: np.ndarray, values: np.ndarray, frequency_hz: float
) -> float | None:
    if len(values) < 30 or not np.isfinite(values).all():
        return None
    phase = 2.0 * math.pi * frequency_hz * timestamps
    design = np.column_stack(
        (np.ones(len(values)), np.sin(phase), np.cos(phase))
    )
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    fitted = design @ coefficients
    total = float(np.sum((values - np.mean(values)) ** 2))
    residual = float(np.sum((values - fitted) ** 2))
    return (
        None
        if total == 0.0
        else max(0.0, 1.0 - residual / total)
    )


def describe_flow(
    timestamps: np.ndarray,
    rows: list[dict[str, Any]],
    pose_frequency_hz: float | None,
) -> tuple[dict[str, Any], np.ndarray]:
    evaluable = np.asarray(
        [independently_direction_evaluable(row) for row in rows], dtype=bool
    )
    vectors = np.asarray(
        [
            [
                0.0
                if row.get("median_flow_dx_px") is None
                else float(row["median_flow_dx_px"]),
                0.0
                if row.get("median_flow_dy_px") is None
                else float(row["median_flow_dy_px"]),
                0.0
                if row.get("median_radial_flow_px") is None
                else float(row["median_radial_flow_px"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    cosines: list[float] = []
    for index in range(1, len(vectors)):
        if not (evaluable[index - 1] and evaluable[index]):
            continue
        denominator = float(
            np.linalg.norm(vectors[index - 1])
            * np.linalg.norm(vectors[index])
        )
        if denominator > 0.0:
            cosines.append(
                float(
                    np.dot(vectors[index - 1], vectors[index])
                    / denominator
                )
            )
    if np.count_nonzero(evaluable) >= 2:
        axis = principal_axis(vectors[evaluable])
        projection = vectors[evaluable] @ axis
        periodic_r2 = (
            periodic_fit_r_squared(
                timestamps[evaluable], projection, pose_frequency_hz
            )
            if pose_frequency_hz is not None
            else None
        )
        axis_value: list[float] | None = [
            float(axis[0]),
            float(axis[1]),
            float(axis[2]),
        ]
    else:
        periodic_r2 = None
        axis_value = None
    return (
        {
            "direction_evaluable_pair_count": int(
                np.count_nonzero(evaluable)
            ),
            "direction_evaluable_fraction": float(np.mean(evaluable)),
            "median_adjacent_direction_cosine": (
                float(np.median(cosines)) if cosines else None
            ),
            "adjacent_direction_pair_count": len(cosines),
            "flow_principal_axis_xy_radial": axis_value,
            "flow_periodic_r_squared_at_pose_frequency": periodic_r2,
        },
        evaluable,
    )


def prevalence_ratio(
    target: np.ndarray, exposure: np.ndarray, eligible: np.ndarray
) -> float | None:
    exposed = eligible & exposure
    unexposed = eligible & ~exposure
    if not np.any(exposed) or not np.any(unexposed):
        return None
    return divide_or_none(
        float(np.mean(target[exposed])),
        float(np.mean(target[unexposed])),
    )


def describe_measurement_failure(
    proxy_rows: list[dict[str, Any]],
    high_response: np.ndarray,
    r3_evaluable: np.ndarray,
) -> dict[str, Any]:
    sharpness = np.asarray(
        [float(row["sharpness_laplacian_variance"]) for row in proxy_rows]
    )
    texture = np.asarray(
        [
            float(row["detected_features_per_valid_megapixel"])
            for row in proxy_rows
        ]
    )
    blur = normalized_average_ranks(sharpness) <= TAIL_FRACTION
    low_texture = normalized_average_ranks(texture) <= TAIL_FRACTION
    feature_collapse = np.asarray(
        [
            int(row["detected_feature_count"]) < 60
            or int(row["forward_backward_consistent_count"]) < 60
            or float(row["forward_backward_consistent_fraction"]) < 0.5
            for row in proxy_rows
        ],
        dtype=bool,
    )
    round_trip_failure = np.asarray(
        [
            row.get("median_forward_backward_error_px") is None
            or float(row["median_forward_backward_error_px"]) > 0.75
            for row in proxy_rows
        ],
        dtype=bool,
    )
    failure = feature_collapse | round_trip_failure | blur
    event_lengths = contiguous_true_lengths(failure)
    high_count = int(np.count_nonzero(high_response))
    return {
        "blur_pair_count": int(np.count_nonzero(blur)),
        "low_texture_pair_count": int(np.count_nonzero(low_texture)),
        "feature_collapse_pair_count": int(
            np.count_nonzero(feature_collapse)
        ),
        "round_trip_failure_pair_count": int(
            np.count_nonzero(round_trip_failure)
        ),
        "failure_pair_count": int(np.count_nonzero(failure)),
        "failure_event_count": len(event_lengths),
        "maximum_failure_event_pairs": max(event_lengths, default=0),
        "high_response_failure_overlap_fraction": divide_or_none(
            float(np.count_nonzero(high_response & failure)),
            float(high_count),
        ),
        "high_response_prevalence_ratio_failure_vs_nonfailure": (
            prevalence_ratio(high_response, failure, r3_evaluable)
        ),
        "_failure_mask": failure,
    }


def independently_summarize_session(
    session: int,
    direction_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
    r3_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    timestamps = np.asarray(
        [
            0.5
            * (
                float(row["previous_timestamp_s"])
                + float(row["current_timestamp_s"])
            )
            for row in direction_rows
        ]
    )
    selected_component, pose = strongest_pose_component(
        timestamps, direction_rows
    )
    high_response = response_top_tail(r3_rows)
    r3_evaluable = np.asarray(
        [row.get("evaluable") is True for row in r3_rows], dtype=bool
    )
    response_magnitude = np.asarray(
        [
            abs(float(row["compensated_expansion_median_per_s"]))
            if row.get("compensated_expansion_median_per_s") is not None
            else float("-inf")
            for row in r3_rows
        ]
    )
    cycles = describe_cycles(
        pose["grid"],
        pose["bandpassed"],
        timestamps,
        r3_evaluable,
        high_response,
        response_magnitude,
    )
    flow, direction_mask = describe_flow(
        timestamps, direction_rows, pose["dominant_frequency_hz"]
    )
    failure = describe_measurement_failure(
        proxy_rows, high_response, r3_evaluable
    )
    failure_mask = failure.pop("_failure_mask")
    high_count = int(np.count_nonzero(high_response))
    high_direction_fraction = divide_or_none(
        float(np.count_nonzero(high_response & direction_mask)),
        float(high_count),
    )
    pose_fraction = pose["band_energy_fraction"]
    adjacent_cosine = flow["median_adjacent_direction_cosine"]
    periodic_r2 = flow[
        "flow_periodic_r_squared_at_pose_frequency"
    ]
    cycle_fraction = cycles["high_response_cycle_fraction"]
    phase_locking = cycles[
        "cycle_max_absolute_response_axial_phase_locking_value"
    ]
    motion_support = bool(
        pose_fraction is not None
        and pose_fraction >= 0.35
        and cycles["valid_cycle_count"] >= 4
        and flow["direction_evaluable_fraction"] >= 0.70
        and adjacent_cosine is not None
        and adjacent_cosine >= 0.50
        and periodic_r2 is not None
        and periodic_r2 >= 0.20
        and cycle_fraction is not None
        and cycle_fraction >= 0.50
        and cycles["longest_consecutive_high_response_cycles"] >= 3
        and phase_locking is not None
        and phase_locking >= 0.40
        and high_direction_fraction is not None
        and high_direction_fraction >= 0.70
    )
    failure_ratio = failure[
        "high_response_prevalence_ratio_failure_vs_nonfailure"
    ]
    failure_support = bool(
        failure["high_response_failure_overlap_fraction"] is not None
        and failure["high_response_failure_overlap_fraction"] >= 0.50
        and failure_ratio is not None
        and failure_ratio >= 1.50
        and failure["maximum_failure_event_pairs"] >= 3
    )
    return {
        "session": session,
        "pair_count": PAIR_COUNT,
        "duration_s": float(
            direction_rows[-1]["current_timestamp_s"]
            - direction_rows[0]["previous_timestamp_s"]
        ),
        "r3_evaluable_pair_count": int(np.count_nonzero(r3_evaluable)),
        "high_response_pair_count": high_count,
        "selected_pose_component": selected_component,
        "pose_band_energy_fraction": pose_fraction,
        "pose_dominant_frequency_hz": pose["dominant_frequency_hz"],
        "pose_cycles": cycles,
        "flow_temporal_direction": flow,
        "high_response_direction_evaluable_fraction": (
            high_direction_fraction
        ),
        "measurement_failure": failure,
        "track_consistent_periodic_motion_support": motion_support,
        "measurement_failure_support": failure_support,
        "valid_for_cross_session_terminal": bool(
            cycles["valid_cycle_count"] >= 4
            and flow["direction_evaluable_fraction"] >= 0.70
        ),
    }


def independently_route(
    session_results: list[dict[str, Any]]
) -> tuple[dict[str, int], str]:
    counts = {
        "valid_for_terminal": sum(
            bool(row["valid_for_cross_session_terminal"])
            for row in session_results
        ),
        "track_consistent_periodic_motion_support": sum(
            bool(row["track_consistent_periodic_motion_support"])
            for row in session_results
        ),
        "measurement_failure_support": sum(
            bool(row["measurement_failure_support"])
            for row in session_results
        ),
    }
    if counts["valid_for_terminal"] < 3:
        terminal = "NOT_EVALUABLE"
    elif (
        counts["track_consistent_periodic_motion_support"] >= 3
        and counts["measurement_failure_support"] <= 1
    ):
        terminal = (
            "PRIORITIZE_MOTION_DECOMPOSITION_OR_TEMPORAL_MODELING"
        )
    elif (
        counts["measurement_failure_support"] >= 3
        and counts["track_consistent_periodic_motion_support"] <= 1
    ):
        terminal = "PRIORITIZE_QUALITY_GATE_REDESIGN"
    else:
        terminal = "HOLD_MIXED_OR_INSUFFICIENT_TEMPORAL_EVIDENCE"
    return counts, terminal


def compare_values(
    expected: Any,
    actual: Any,
    path: str,
    failures: list[str],
) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            failures.append(f"VALUE_MISMATCH:{path}")
        return
    if isinstance(expected, (int, float)) and isinstance(
        actual, (int, float)
    ):
        if not math.isclose(
            float(expected),
            float(actual),
            rel_tol=FLOAT_REL_TOL,
            abs_tol=FLOAT_ABS_TOL,
        ):
            failures.append(
                f"FLOAT_MISMATCH:{path}:{expected!r}:{actual!r}"
            )
        return
    if type(expected) is not type(actual):
        failures.append(f"TYPE_MISMATCH:{path}")
        return
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            failures.append(
                f"KEY_MISMATCH:{path}:"
                f"missing={sorted(expected_keys - actual_keys)}:"
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        for key in sorted(expected_keys & actual_keys):
            compare_values(
                expected[key], actual[key], f"{path}.{key}", failures
            )
        return
    if isinstance(expected, list):
        if len(expected) != len(actual):
            failures.append(
                f"LENGTH_MISMATCH:{path}:{len(expected)}:{len(actual)}"
            )
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare_values(left, right, f"{path}[{index}]", failures)
        return
    if expected != actual:
        failures.append(f"VALUE_MISMATCH:{path}")


def validate_contract(contract: dict[str, Any], failures: list[str]) -> None:
    if contract.get("schema") != CONTRACT_SCHEMA:
        failures.append("CONTRACT_SCHEMA_MISMATCH")
    if contract.get("protocol_id") != PROTOCOL_ID:
        failures.append("CONTRACT_PROTOCOL_ID_MISMATCH")
    frozen = contract.get("frozen_inputs", {})
    if (
        frozen.get("sessions") != list(SESSIONS)
        or frozen.get("pair_indices_per_session") != "0..600"
        or frozen.get("sealed_session") != SEALED_SESSION
        or frozen.get("sealed_session_policy")
        != "NO_DOWNLOAD_NO_EXTRACTION_NO_CONTENT_NO_ALGORITHM_ACCESS"
    ):
        failures.append("CONTRACT_FROZEN_SESSION_IDENTITY_MISMATCH")
    unchanged = contract.get("unchanged_algorithm_contract", {})
    if (
        unchanged.get("implementation")
        != "ADVIO_WXYZ_TCAMIMU_VALIDMASK_CONTINUOUS_R3"
        or unchanged.get("response_field")
        != "compensated_expansion_median_per_s"
        or unchanged.get("response_threshold_operator")
        != "strict_greater_than"
        or unchanged.get("response_threshold_per_s") != 0.01
        or unchanged.get("required_consecutive_evaluable_pairs") != 3
    ):
        failures.append("CONTRACT_R3_LOCK_MISMATCH")


def check_row_identity_and_flags(
    session: int,
    direction_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
    r3_rows: list[dict[str, Any]],
    failures: list[str],
) -> None:
    if (
        len(direction_rows) != PAIR_COUNT
        or len(proxy_rows) != PAIR_COUNT
        or len(r3_rows) != PAIR_COUNT
    ):
        failures.append(
            f"PAIR_COUNT_MISMATCH:{session}:"
            f"{len(direction_rows)}:{len(proxy_rows)}:{len(r3_rows)}"
        )
        return
    for index, joined in enumerate(
        zip(direction_rows, proxy_rows, r3_rows)
    ):
        for source_name, row in zip(
            ("direction", "proxy", "r3"), joined
        ):
            forbidden = sorted(FORBIDDEN_LABEL_KEYS.intersection(row))
            if forbidden:
                failures.append(
                    f"FORBIDDEN_LABEL_FIELD:{session}:{index}:"
                    f"{source_name}:{','.join(forbidden)}"
                )
        if any(
            "session" in row and row.get("session") != session
            for row in joined
        ):
            failures.append(f"SESSION_IDENTITY_MISMATCH:{session}:{index}")
        if any(row.get("pair_index") != index for row in joined):
            failures.append(f"PAIR_IDENTITY_MISMATCH:{session}:{index}")
        for field in (
            "previous_timestamp_s",
            "current_timestamp_s",
            "dt_s",
        ):
            values = [row.get(field) for row in joined]
            if any(value is None for value in values) or not all(
                math.isclose(
                    float(values[0]),
                    float(value),
                    rel_tol=FLOAT_REL_TOL,
                    abs_tol=FLOAT_ABS_TOL,
                )
                for value in values[1:]
            ):
                failures.append(
                    f"PAIR_TIMESTAMP_IDENTITY_MISMATCH:"
                    f"{session}:{index}:{field}"
                )
        for key, expected in DIRECTION_ROW_FLAGS.items():
            if direction_rows[index].get(key) is not expected:
                failures.append(
                    f"DIRECTION_FIREWALL_FLAG:{session}:{index}:{key}"
                )
        for key, expected in PROXY_ROW_FLAGS.items():
            if proxy_rows[index].get(key) is not expected:
                failures.append(
                    f"PROXY_FIREWALL_FLAG:{session}:{index}:{key}"
                )
        expected_direction = independently_direction_evaluable(
            direction_rows[index]
        )
        if direction_rows[index].get("direction_evaluable") is not (
            expected_direction
        ):
            failures.append(
                f"DIRECTION_EVALUABLE_FLAG:{session}:{index}"
            )


def check_direction_summary(
    session: int,
    summary: dict[str, Any],
    contract: dict[str, Any],
    contract_hash: str,
    ledger_hash: str,
    failures: list[str],
) -> None:
    frozen = contract["frozen_inputs"]["by_session"][str(session)]
    expected_inputs = {
        "frames.mov": frozen["frames_mov_sha256"],
        "frames.csv": frozen["frames_csv_sha256"],
        "pose.csv": frozen["pose_csv_sha256"],
    }
    if (
        summary.get("schema") != DIRECTION_SUMMARY_SCHEMA
        or summary.get("protocol_id") != PROTOCOL_ID
        or summary.get("session") != session
        or summary.get("pair_count") != PAIR_COUNT
    ):
        failures.append(f"DIRECTION_SUMMARY_IDENTITY:{session}")
    if summary.get("contract_sha256") != contract_hash:
        failures.append(f"DIRECTION_CONTRACT_HASH:{session}")
    if summary.get("input_sha256") != expected_inputs:
        failures.append(f"DIRECTION_FROZEN_INPUT_HASH:{session}")
    if summary.get("direction_ledger_sha256") != ledger_hash:
        failures.append(f"DIRECTION_LEDGER_HASH:{session}")
    if summary.get("runtime_versions") != contract.get("runtime_lock"):
        failures.append(f"DIRECTION_RUNTIME_LOCK:{session}")
    if summary.get("firewall_flags") != DIRECTION_ROW_FLAGS:
        failures.append(f"DIRECTION_FIREWALL_SUMMARY:{session}")
    for key, expected in DIRECTION_ROW_FLAGS.items():
        if summary.get(key) is not expected:
            failures.append(f"DIRECTION_FIREWALL_SUMMARY:{session}:{key}")


def check_proxy_summary(
    session: int,
    summary: dict[str, Any],
    contract: dict[str, Any],
    ledger_hash: str,
    failures: list[str],
) -> None:
    frozen = contract["frozen_inputs"]["by_session"][str(session)]
    expected_inputs = {
        "frames.mov": frozen["frames_mov_sha256"],
        "frames.csv": frozen["frames_csv_sha256"],
        "pose.csv": frozen["pose_csv_sha256"],
    }
    if (
        summary.get("session") != session
        or summary.get("pair_count") != PAIR_COUNT
    ):
        failures.append(f"PROXY_SUMMARY_IDENTITY:{session}")
    if (
        summary.get("contract_sha256")
        != contract["frozen_inputs"]["r0_contract_sha256"]
    ):
        failures.append(f"PROXY_R0_CONTRACT_HASH:{session}")
    if summary.get("input_sha256") != expected_inputs:
        failures.append(f"PROXY_FROZEN_INPUT_HASH:{session}")
    if summary.get("proxy_ledger_sha256") != ledger_hash:
        failures.append(f"PROXY_LEDGER_HASH:{session}")
    for key, expected in {
        **PROXY_ROW_FLAGS,
        "sealed_session_accessed": False,
    }.items():
        if summary.get(key) is not expected:
            failures.append(f"PROXY_FIREWALL_SUMMARY:{session}:{key}")


def check_r3_summary(
    session: int,
    summary: dict[str, Any],
    ledger_hash: str,
    failures: list[str],
) -> None:
    evidence = summary.get("evidence_context", {})
    execution = summary.get("execution", {})
    if (
        evidence.get("session_number") != session
        or evidence.get("implementation_version")
        != "ADVIO_WXYZ_TCAMIMU_VALIDMASK_CONTINUOUS_R3"
        or evidence.get("sealed_session_accessed") is not False
        or evidence.get("algorithm_adjustment") is not False
        or evidence.get("three_pair_rule_changed") is not False
        or evidence.get("threshold_changed") is not False
    ):
        failures.append(f"R3_IDENTITY_OR_FLAGS:{session}")
    if (
        execution.get("candidate_pair_count") != PAIR_COUNT
        or execution.get("threshold_per_s") != 0.01
        or execution.get("required_consecutive_pairs") != 3
    ):
        failures.append(f"R3_EXECUTION_LOCK:{session}")
    if summary.get("artifacts", {}).get("pair_ledger_sha256") != ledger_hash:
        failures.append(f"R3_LEDGER_HASH:{session}")


def validate(
    direction_root: Path,
    proxy_root: Path,
    r3_runs_root: Path,
    contract_path: Path,
    analysis_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    failures: list[str] = []
    artifact_hashes: dict[str, dict[str, str]] = {}
    recomputed_sessions: list[dict[str, Any]] = []

    try:
        contract = load_json(contract_path)
        analysis = load_json(analysis_path)
    except Exception as exc:
        contract = {}
        analysis = {}
        failures.append(f"JSON_LOAD_FAILED:{type(exc).__name__}:{exc}")

    contract_hash = (
        file_sha256(contract_path) if contract_path.is_file() else ""
    )
    analysis_hash = (
        file_sha256(analysis_path) if analysis_path.is_file() else ""
    )
    if contract:
        validate_contract(contract, failures)
    if analysis:
        if (
            analysis.get("schema") != ANALYSIS_SCHEMA
            or analysis.get("protocol_id") != PROTOCOL_ID
            or analysis.get("sessions") != list(SESSIONS)
        ):
            failures.append("ANALYSIS_IDENTITY_MISMATCH")
        if analysis.get("contract_sha256") != contract_hash:
            failures.append("ANALYSIS_CONTRACT_HASH_MISMATCH")
        for key, expected in ANALYSIS_FLAGS.items():
            if analysis.get(key) is not expected:
                failures.append(f"ANALYSIS_FIREWALL_FLAG:{key}")
        if (
            analysis.get("claim_ceiling")
            != "DEVELOPMENT_PRIORITY_ONLY_NO_FALSE_ALERT_NO_CAUSALITY"
            or analysis.get(
                "pair_records_are_longitudinal_not_independent_samples"
            )
            is not True
        ):
            failures.append("ANALYSIS_CLAIM_OR_GRAIN_MISMATCH")

    for session in SESSIONS:
        direction_dir = direction_root / f"advio-{session:02d}"
        proxy_dir = proxy_root / f"advio-{session:02d}"
        r3_dir = r3_runs_root / f"advio-{session:02d}_r3_fixed_601"
        paths = {
            "direction_ledger": direction_dir / "direction_ledger.jsonl",
            "direction_summary": direction_dir / "direction_summary.json",
            "proxy_ledger": proxy_dir / "proxy_ledger.jsonl",
            "proxy_summary": proxy_dir / "proxy_summary.json",
            "r3_pair_ledger": r3_dir / "pair_ledger.jsonl",
            "r3_summary": r3_dir / "summary.json",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            failures.extend(
                f"MISSING_INPUT:{session}:{name}" for name in missing
            )
            continue
        hashes = {name: file_sha256(path) for name, path in paths.items()}
        artifact_hashes[str(session)] = hashes
        try:
            direction_rows = load_jsonl(paths["direction_ledger"])
            proxy_rows = load_jsonl(paths["proxy_ledger"])
            r3_rows = load_jsonl(paths["r3_pair_ledger"])
            direction_summary = load_json(paths["direction_summary"])
            proxy_summary = load_json(paths["proxy_summary"])
            r3_summary = load_json(paths["r3_summary"])
            check_row_identity_and_flags(
                session,
                direction_rows,
                proxy_rows,
                r3_rows,
                failures,
            )
            if not (
                len(direction_rows)
                == len(proxy_rows)
                == len(r3_rows)
                == PAIR_COUNT
            ):
                continue
            check_direction_summary(
                session,
                direction_summary,
                contract,
                contract_hash,
                hashes["direction_ledger"],
                failures,
            )
            check_proxy_summary(
                session,
                proxy_summary,
                contract,
                hashes["proxy_ledger"],
                failures,
            )
            check_r3_summary(
                session,
                r3_summary,
                hashes["r3_pair_ledger"],
                failures,
            )
            frozen = contract["frozen_inputs"]["by_session"][str(session)]
            if hashes["proxy_ledger"] != frozen["r0_proxy_ledger_sha256"]:
                failures.append(f"PROXY_CONTRACT_LEDGER_HASH:{session}")
            if hashes["r3_pair_ledger"] != frozen["r3_pair_ledger_sha256"]:
                failures.append(f"R3_CONTRACT_LEDGER_HASH:{session}")
            expected_analysis_hashes = {
                "direction_ledger_sha256": hashes["direction_ledger"],
                "r0_proxy_ledger_sha256": hashes["proxy_ledger"],
                "r3_pair_ledger_sha256": hashes["r3_pair_ledger"],
            }
            if (
                analysis.get("input_hashes", {}).get(str(session))
                != expected_analysis_hashes
            ):
                failures.append(f"ANALYSIS_INPUT_HASHES:{session}")
            recomputed_sessions.append(
                independently_summarize_session(
                    session, direction_rows, proxy_rows, r3_rows
                )
            )
        except Exception as exc:
            failures.append(
                f"SESSION_VALIDATION_FAILED:{session}:"
                f"{type(exc).__name__}:{exc}"
            )

    if len(recomputed_sessions) == len(SESSIONS):
        compare_values(
            recomputed_sessions,
            analysis.get("session_results"),
            "$.session_results",
            failures,
        )
        counts, terminal = independently_route(recomputed_sessions)
        compare_values(
            counts,
            analysis.get("cross_session_counts"),
            "$.cross_session_counts",
            failures,
        )
        if analysis.get("terminal") != terminal:
            failures.append(
                f"TERMINAL_MISMATCH:{terminal}:"
                f"{analysis.get('terminal')}"
            )
    else:
        counts = {}
        terminal = None
        failures.append(
            f"INDEPENDENT_SESSION_RECOMPUTATION_INCOMPLETE:"
            f"{len(recomputed_sessions)}"
        )

    receipt = {
        "schema": VALIDATION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "VALID" if not failures else "INVALID",
        "failures": failures,
        "contract_sha256": contract_hash,
        "analysis_sha256": analysis_hash,
        "artifact_sha256_by_session": artifact_hashes,
        "recomputed_cross_session_counts": counts,
        "recomputed_terminal": terminal,
        "float_comparison_tolerance": {
            "relative": FLOAT_REL_TOL,
            "absolute": FLOAT_ABS_TOL,
        },
        "risk_label_accessed": False,
        "obstacle_label_accessed": False,
        "manual_gait_label_accessed": False,
        "sealed_session_accessed": False,
        "independent_implementation": True,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction-root", type=Path, required=True)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--r3-runs-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate(
        args.direction_root.resolve(),
        args.proxy_root.resolve(),
        args.r3_runs_root.resolve(),
        args.contract.resolve(),
        args.analysis.resolve(),
        args.receipt.resolve(),
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if receipt["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
