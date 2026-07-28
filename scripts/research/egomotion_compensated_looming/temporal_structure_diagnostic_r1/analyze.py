from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROTOCOL_ID = "RCLE_TEMPORAL_STRUCTURE_DIAGNOSTIC_R1"
SESSIONS = (13, 14, 15, 17)
PAIR_COUNT = 601
GRID_HZ = 60.0
POSE_BAND_HZ = (0.7, 3.0)
ENERGY_DENOMINATOR_HZ = (0.1, 10.0)
HIGH_RESPONSE_FRACTION = 0.2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rank_average(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("RANK_INPUT")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks / max(1, len(array) - 1)


def centered_rolling_median(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or width % 2 == 0:
        raise ValueError("ROLLING_WIDTH")
    array = np.asarray(values, dtype=np.float64)
    radius = width // 2
    return np.asarray(
        [
            np.median(array[max(0, i - radius) : min(len(array), i + radius + 1)])
            for i in range(len(array))
        ],
        dtype=np.float64,
    )


def longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def event_lengths(values: np.ndarray) -> list[int]:
    lengths: list[int] = []
    current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def high_response_mask(r3_rows: list[dict[str, Any]]) -> np.ndarray:
    evaluable = np.asarray(
        [
            row.get("evaluable") is True
            and row.get("compensated_expansion_median_per_s") is not None
            for row in r3_rows
        ],
        dtype=bool,
    )
    result = np.zeros(len(r3_rows), dtype=bool)
    eligible_indices = np.flatnonzero(evaluable)
    values = [
        abs(float(r3_rows[index]["compensated_expansion_median_per_s"]))
        for index in eligible_indices
    ]
    ranks = rank_average(values)
    result[eligible_indices] = ranks >= 1.0 - HIGH_RESPONSE_FRACTION
    return result


def interpolate_uniform(
    timestamps: np.ndarray, values: np.ndarray, hz: float = GRID_HZ
) -> tuple[np.ndarray, np.ndarray]:
    if len(timestamps) != len(values) or len(values) < 4:
        raise ValueError("UNIFORM_INPUT")
    if not np.isfinite(timestamps).all() or not np.isfinite(values).all():
        raise ValueError("UNIFORM_NONFINITE")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("TIMESTAMP_NOT_STRICT")
    step = 1.0 / hz
    count = int(math.floor((timestamps[-1] - timestamps[0]) / step)) + 1
    grid = timestamps[0] + np.arange(count, dtype=np.float64) * step
    return grid, np.interp(grid, timestamps, values)


def component_spectrum(
    timestamps: np.ndarray, values: np.ndarray
) -> dict[str, Any]:
    grid, uniform = interpolate_uniform(timestamps, values)
    detrended = uniform - centered_rolling_median(uniform, 61)
    transformed = np.fft.rfft(detrended)
    frequencies = np.fft.rfftfreq(len(detrended), d=1.0 / GRID_HZ)
    power = np.square(np.abs(transformed))
    denominator = (
        (frequencies >= ENERGY_DENOMINATOR_HZ[0])
        & (frequencies <= ENERGY_DENOMINATOR_HZ[1])
    )
    band = (frequencies >= POSE_BAND_HZ[0]) & (frequencies <= POSE_BAND_HZ[1])
    denominator_energy = float(np.sum(power[denominator]))
    band_energy = float(np.sum(power[band]))
    band_fraction = safe_ratio(band_energy, denominator_energy)
    band_indices = np.flatnonzero(band)
    if not len(band_indices) or band_energy == 0:
        dominant_frequency = None
    else:
        dominant_frequency = float(
            frequencies[band_indices[int(np.argmax(power[band_indices]))]]
        )
    filtered_spectrum = np.zeros_like(transformed)
    filtered_spectrum[band] = transformed[band]
    bandpassed = np.fft.irfft(filtered_spectrum, n=len(detrended))
    return {
        "grid_timestamps_s": grid,
        "detrended": detrended,
        "bandpassed": bandpassed,
        "band_energy": band_energy,
        "denominator_energy": denominator_energy,
        "band_energy_fraction": band_fraction,
        "dominant_frequency_hz": dominant_frequency,
    }


def select_pose_component(
    timestamps: np.ndarray, direction_rows: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    fields = {
        "angular_x": "camera_angular_velocity_x_deg_per_s",
        "angular_y": "camera_angular_velocity_y_deg_per_s",
        "angular_z": "camera_angular_velocity_z_deg_per_s",
        "translation_x": "camera_translation_velocity_x_m_per_s",
        "translation_y": "camera_translation_velocity_y_m_per_s",
        "translation_z": "camera_translation_velocity_z_m_per_s",
    }
    candidates: list[tuple[str, dict[str, Any]]] = []
    for name, field in fields.items():
        result = component_spectrum(
            timestamps,
            np.asarray([float(row[field]) for row in direction_rows]),
        )
        candidates.append((name, result))
    return max(
        candidates,
        key=lambda item: (
            -1.0
            if item[1]["band_energy_fraction"] is None
            else item[1]["band_energy_fraction"],
            item[0],
        ),
    )


def positive_crossing_times(timestamps: np.ndarray, values: np.ndarray) -> list[float]:
    crossings: list[float] = []
    for index in range(1, len(values)):
        left = float(values[index - 1])
        right = float(values[index])
        if left < 0.0 <= right and right != left:
            fraction = -left / (right - left)
            crossings.append(
                float(
                    timestamps[index - 1]
                    + fraction * (timestamps[index] - timestamps[index - 1])
                )
            )
    return crossings


def pose_cycles(
    grid_timestamps: np.ndarray,
    bandpassed: np.ndarray,
    pair_timestamps: np.ndarray,
    r3_evaluable: np.ndarray,
    high_response: np.ndarray,
    absolute_response: np.ndarray,
) -> dict[str, Any]:
    crossings = positive_crossing_times(grid_timestamps, bandpassed)
    valid_cycles: list[dict[str, Any]] = []
    longest_consecutive = 0
    current_consecutive = 0
    for start, end in zip(crossings, crossings[1:]):
        period = end - start
        if not (1.0 / POSE_BAND_HZ[1] <= period <= 1.0 / POSE_BAND_HZ[0]):
            current_consecutive = 0
            continue
        members = (pair_timestamps >= start) & (pair_timestamps < end)
        if not np.any(members):
            current_consecutive = 0
            continue
        coverage = float(np.mean(r3_evaluable[members]))
        if coverage < 0.8:
            current_consecutive = 0
            continue
        eligible_members = members & r3_evaluable
        member_indices = np.flatnonzero(eligible_members)
        peak_index = member_indices[
            int(np.argmax(absolute_response[member_indices]))
        ]
        phase = 2.0 * math.pi * (
            (pair_timestamps[peak_index] - start) / period
        )
        contains_high_response = bool(np.any(high_response[members]))
        current_consecutive = (
            current_consecutive + 1 if contains_high_response else 0
        )
        longest_consecutive = max(longest_consecutive, current_consecutive)
        valid_cycles.append(
            {
                "start_s": start,
                "end_s": end,
                "period_s": period,
                "evaluable_coverage": coverage,
                "contains_high_response": contains_high_response,
                "cycle_max_response_phase_rad": phase,
            }
        )
    contains = np.asarray(
        [row["contains_high_response"] for row in valid_cycles], dtype=bool
    )
    phases = np.asarray(
        [row["cycle_max_response_phase_rad"] for row in valid_cycles],
        dtype=np.float64,
    )
    axial_phase_locking = (
        float(abs(np.mean(np.exp(2j * phases)))) if len(phases) else None
    )
    return {
        "valid_cycle_count": len(valid_cycles),
        "cycles": valid_cycles,
        "high_response_cycle_fraction": (
            float(np.mean(contains)) if len(contains) else None
        ),
        "longest_consecutive_high_response_cycles": longest_consecutive,
        "cycle_max_absolute_response_axial_phase_locking_value": (
            axial_phase_locking
        ),
    }


def direction_evaluable(row: dict[str, Any]) -> bool:
    median_error = row.get("median_forward_backward_error_px")
    resultant = row.get("spatial_direction_resultant")
    radial_consistency = row.get("radial_direction_consistency")
    direction_structure = bool(
        (resultant is not None and float(resultant) >= 0.5)
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


def deterministic_flow_axis(vectors: np.ndarray) -> np.ndarray:
    centered = vectors - np.mean(vectors, axis=0)
    covariance = centered.T @ centered / max(1, len(centered))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    first_nonzero = next((value for value in axis if value != 0.0), 0.0)
    if first_nonzero < 0.0:
        axis = -axis
    return axis


def sinusoid_r_squared(
    timestamps: np.ndarray, values: np.ndarray, frequency_hz: float
) -> float | None:
    if len(values) < 30 or not np.isfinite(values).all():
        return None
    omega_t = 2.0 * math.pi * frequency_hz * timestamps
    design = np.column_stack(
        [np.ones(len(timestamps)), np.sin(omega_t), np.cos(omega_t)]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coefficients
    total = float(np.sum(np.square(values - np.mean(values))))
    residual = float(np.sum(np.square(values - fitted)))
    return None if total == 0.0 else max(0.0, 1.0 - residual / total)


def flow_temporal_metrics(
    timestamps: np.ndarray,
    direction_rows: list[dict[str, Any]],
    pose_frequency_hz: float | None,
) -> dict[str, Any]:
    mask = np.asarray([direction_evaluable(row) for row in direction_rows])
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
            for row in direction_rows
        ],
        dtype=np.float64,
    )
    adjacent_cosines: list[float] = []
    for index in range(1, len(vectors)):
        if not (mask[index - 1] and mask[index]):
            continue
        left = vectors[index - 1]
        right = vectors[index]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator > 0.0:
            adjacent_cosines.append(float(np.dot(left, right) / denominator))
    if np.count_nonzero(mask) >= 2:
        axis = deterministic_flow_axis(vectors[mask])
        projection = vectors[mask] @ axis
        periodic_r_squared = (
            sinusoid_r_squared(
                timestamps[mask], projection, pose_frequency_hz
            )
            if pose_frequency_hz is not None
            else None
        )
        axis_list: list[float] | None = [
            float(axis[0]),
            float(axis[1]),
            float(axis[2]),
        ]
    else:
        periodic_r_squared = None
        axis_list = None
    return {
        "direction_evaluable_pair_count": int(np.count_nonzero(mask)),
        "direction_evaluable_fraction": float(np.mean(mask)),
        "median_adjacent_direction_cosine": (
            float(np.median(adjacent_cosines)) if adjacent_cosines else None
        ),
        "adjacent_direction_pair_count": len(adjacent_cosines),
        "flow_principal_axis_xy_radial": axis_list,
        "flow_periodic_r_squared_at_pose_frequency": periodic_r_squared,
        "_direction_evaluable_mask": mask,
    }


def prevalence_ratio(
    target: np.ndarray, exposure: np.ndarray, eligible: np.ndarray
) -> float | None:
    exposed = eligible & exposure
    unexposed = eligible & ~exposure
    if not np.any(exposed) or not np.any(unexposed):
        return None
    return safe_ratio(float(np.mean(target[exposed])), float(np.mean(target[unexposed])))


def measurement_failure_metrics(
    proxy_rows: list[dict[str, Any]],
    high_response: np.ndarray,
    r3_evaluable: np.ndarray,
) -> dict[str, Any]:
    sharpness = np.asarray(
        [float(row["sharpness_laplacian_variance"]) for row in proxy_rows]
    )
    texture = np.asarray(
        [float(row["detected_features_per_valid_megapixel"]) for row in proxy_rows]
    )
    blur = rank_average(sharpness) <= HIGH_RESPONSE_FRACTION
    low_texture = rank_average(texture) <= HIGH_RESPONSE_FRACTION
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
    lengths = event_lengths(failure)
    high_count = int(np.count_nonzero(high_response))
    overlap = safe_ratio(
        float(np.count_nonzero(high_response & failure)), float(high_count)
    )
    return {
        "blur_pair_count": int(np.count_nonzero(blur)),
        "low_texture_pair_count": int(np.count_nonzero(low_texture)),
        "feature_collapse_pair_count": int(np.count_nonzero(feature_collapse)),
        "round_trip_failure_pair_count": int(np.count_nonzero(round_trip_failure)),
        "failure_pair_count": int(np.count_nonzero(failure)),
        "failure_event_count": len(lengths),
        "maximum_failure_event_pairs": max(lengths, default=0),
        "high_response_failure_overlap_fraction": overlap,
        "high_response_prevalence_ratio_failure_vs_nonfailure": prevalence_ratio(
            high_response, failure, r3_evaluable
        ),
        "_failure_mask": failure,
    }


def check_identity(
    session: int,
    direction_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
    r3_rows: list[dict[str, Any]],
) -> None:
    if not (
        len(direction_rows) == len(proxy_rows) == len(r3_rows) == PAIR_COUNT
    ):
        raise ValueError(f"PAIR_COUNT:{session}")
    for index, rows in enumerate(zip(direction_rows, proxy_rows, r3_rows)):
        if any(int(row["pair_index"]) != index for row in rows):
            raise ValueError(f"PAIR_IDENTITY:{session}:{index}")
        if any(
            "session" in row and int(row["session"]) != session
            for row in rows
        ):
            raise ValueError(f"SESSION_IDENTITY:{session}:{index}")


def summarize_session(
    session: int,
    direction_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
    r3_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    check_identity(session, direction_rows, proxy_rows, r3_rows)
    timestamps = np.asarray(
        [
            0.5
            * (
                float(row["previous_timestamp_s"])
                + float(row["current_timestamp_s"])
            )
            for row in direction_rows
        ],
        dtype=np.float64,
    )
    selected_component, pose = select_pose_component(timestamps, direction_rows)
    high_response = high_response_mask(r3_rows)
    r3_evaluable = np.asarray(
        [row.get("evaluable") is True for row in r3_rows], dtype=bool
    )
    absolute_response = np.asarray(
        [
            abs(float(row["compensated_expansion_median_per_s"]))
            if row.get("compensated_expansion_median_per_s") is not None
            else float("-inf")
            for row in r3_rows
        ]
    )
    cycles = pose_cycles(
        pose["grid_timestamps_s"],
        pose["bandpassed"],
        timestamps,
        r3_evaluable,
        high_response,
        absolute_response,
    )
    flow = flow_temporal_metrics(
        timestamps, direction_rows, pose["dominant_frequency_hz"]
    )
    failure = measurement_failure_metrics(
        proxy_rows, high_response, r3_evaluable
    )
    direction_mask = flow.pop("_direction_evaluable_mask")
    failure.pop("_failure_mask")
    high_count = int(np.count_nonzero(high_response))
    high_direction_fraction = safe_ratio(
        float(np.count_nonzero(high_response & direction_mask)), float(high_count)
    )
    pose_fraction = pose["band_energy_fraction"]
    adjacent_cosine = flow["median_adjacent_direction_cosine"]
    periodic_r_squared = flow["flow_periodic_r_squared_at_pose_frequency"]
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
        and periodic_r_squared is not None
        and periodic_r_squared >= 0.20
        and cycle_fraction is not None
        and cycle_fraction >= 0.50
        and cycles["longest_consecutive_high_response_cycles"] >= 3
        and phase_locking is not None
        and phase_locking >= 0.40
        and high_direction_fraction is not None
        and high_direction_fraction >= 0.70
    )
    failure_rr = failure[
        "high_response_prevalence_ratio_failure_vs_nonfailure"
    ]
    failure_support = bool(
        failure["high_response_failure_overlap_fraction"] is not None
        and failure["high_response_failure_overlap_fraction"] >= 0.50
        and failure_rr is not None
        and failure_rr >= 1.50
        and failure["maximum_failure_event_pairs"] >= 3
    )
    valid_for_cross_session = bool(
        cycles["valid_cycle_count"] >= 4
        and flow["direction_evaluable_fraction"] >= 0.70
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
        "high_response_direction_evaluable_fraction": high_direction_fraction,
        "measurement_failure": failure,
        "track_consistent_periodic_motion_support": motion_support,
        "measurement_failure_support": failure_support,
        "valid_for_cross_session_terminal": valid_for_cross_session,
    }


def analyze(
    direction_root: Path,
    proxy_root: Path,
    r3_runs_root: Path,
    contract_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("CONTRACT_PROTOCOL_ID")
    session_results: list[dict[str, Any]] = []
    input_hashes: dict[str, dict[str, str]] = {}
    for session in SESSIONS:
        direction_path = (
            direction_root / f"advio-{session:02d}" / "direction_ledger.jsonl"
        )
        proxy_path = (
            proxy_root / f"advio-{session:02d}" / "proxy_ledger.jsonl"
        )
        r3_path = (
            r3_runs_root
            / f"advio-{session:02d}_r3_fixed_601"
            / "pair_ledger.jsonl"
        )
        session_results.append(
            summarize_session(
                session,
                read_jsonl(direction_path),
                read_jsonl(proxy_path),
                read_jsonl(r3_path),
            )
        )
        input_hashes[str(session)] = {
            "direction_ledger_sha256": sha256_file(direction_path),
            "r0_proxy_ledger_sha256": sha256_file(proxy_path),
            "r3_pair_ledger_sha256": sha256_file(r3_path),
        }
    valid_count = sum(
        result["valid_for_cross_session_terminal"] for result in session_results
    )
    motion_count = sum(
        result["track_consistent_periodic_motion_support"]
        for result in session_results
    )
    failure_count = sum(
        result["measurement_failure_support"] for result in session_results
    )
    if valid_count < 3:
        terminal = "NOT_EVALUABLE"
    elif motion_count >= 3 and failure_count <= 1:
        terminal = "PRIORITIZE_MOTION_DECOMPOSITION_OR_TEMPORAL_MODELING"
    elif failure_count >= 3 and motion_count <= 1:
        terminal = "PRIORITIZE_QUALITY_GATE_REDESIGN"
    else:
        terminal = "HOLD_MIXED_OR_INSUFFICIENT_TEMPORAL_EVIDENCE"
    analysis = {
        "schema": "rcle.temporal_structure_diagnostic.analysis.v1",
        "protocol_id": PROTOCOL_ID,
        "contract_sha256": sha256_file(contract_path),
        "sessions": list(SESSIONS),
        "session_results": session_results,
        "cross_session_counts": {
            "valid_for_terminal": valid_count,
            "track_consistent_periodic_motion_support": motion_count,
            "measurement_failure_support": failure_count,
        },
        "terminal": terminal,
        "claim_ceiling": "DEVELOPMENT_PRIORITY_ONLY_NO_FALSE_ALERT_NO_CAUSALITY",
        "pair_records_are_longitudinal_not_independent_samples": True,
        "risk_label_accessed": False,
        "obstacle_label_accessed": False,
        "manual_gait_label_accessed": False,
        "sealed_session_accessed": False,
        "input_hashes": input_hashes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction-root", type=Path, required=True)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--r3-runs-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.direction_root.resolve(),
        args.proxy_root.resolve(),
        args.r3_runs_root.resolve(),
        args.contract.resolve(),
        args.output.resolve(),
    )
    print(json.dumps({"terminal": result["terminal"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
