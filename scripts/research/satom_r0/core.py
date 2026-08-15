"""Deterministic, past-only SATOM-R0 simulation and evaluation core.

The candidate never receives registered-depth truth.  Truth is held by the
evaluator and by the simulated ToF sensor.  The scan policy sees only the
current frozen prior summary and memory state available before the scan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Iterable, Sequence

import numpy as np


BANDS = ("left", "center", "right")
HORIZONS_M = (1.0, 1.5, 2.0)
STATES = ("CLEAR", "OCCUPIED", "UNKNOWN")
POLICIES = (
    "center_only",
    "random",
    "round_robin",
    "max_entropy",
    "task_weighted_information_gain",
)
CONTROLS = ("none", "shuffled_timestamp", "wrong_extrinsic", "wrong_roi")


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def validate(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")


@dataclass(frozen=True)
class Frame:
    parent_id: str
    frame_index: int
    timestamp_s: float
    truth_depth_m: np.ndarray
    prior_depth_m: np.ndarray
    intrinsics: Intrinsics
    world_from_camera: np.ndarray
    camera_height_m: float
    gravity_down_camera: np.ndarray
    prior_confidence: np.ndarray | None = None

    def validate(self) -> None:
        self.intrinsics.validate()
        if not self.parent_id:
            raise ValueError("parent_id is required")
        if self.frame_index < 0 or not math.isfinite(self.timestamp_s):
            raise ValueError("invalid frame identity")
        if self.truth_depth_m.ndim != 2 or self.truth_depth_m.shape != self.prior_depth_m.shape:
            raise ValueError("truth and prior depth must be same-shape HxW arrays")
        if self.prior_confidence is not None and self.prior_confidence.shape != self.truth_depth_m.shape:
            raise ValueError("prior confidence shape mismatch")
        if self.world_from_camera.shape != (4, 4):
            raise ValueError("world_from_camera must be 4x4")
        if not np.all(np.isfinite(self.world_from_camera)):
            raise ValueError("pose must be finite")
        if not math.isfinite(self.camera_height_m) or self.camera_height_m <= 0:
            raise ValueError("camera height must be positive")
        gravity = np.asarray(self.gravity_down_camera, dtype=np.float64)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("gravity_down_camera must be a finite 3-vector")
        if not 0.95 <= float(np.linalg.norm(gravity)) <= 1.05:
            raise ValueError("gravity_down_camera must be unit length")


@dataclass(frozen=True)
class TofConfig:
    min_range_m: float = 0.04
    max_range_m: float = 4.0
    noise_sigma_m: float = 0.025
    missing_probability: float = 0.03
    first_return_quantile: float = 0.10
    roi_vertical_fraction: tuple[float, float] = (0.22, 0.78)
    roi_horizontal_fractions: tuple[tuple[float, float], ...] = (
        (0.04, 0.38),
        (0.33, 0.67),
        (0.62, 0.96),
    )

    def validate(self) -> None:
        if not 0 < self.min_range_m < self.max_range_m:
            raise ValueError("invalid ToF range")
        if self.noise_sigma_m < 0 or not 0 <= self.missing_probability < 1:
            raise ValueError("invalid ToF noise or missing probability")
        if not 0 <= self.first_return_quantile <= 1:
            raise ValueError("invalid first-return quantile")


@dataclass(frozen=True)
class ArmConfig:
    name: str
    policy: str | None
    use_prior: bool
    use_memory: bool
    use_tof: bool
    control: str = "none"

    def validate(self) -> None:
        if self.policy is not None and self.policy not in POLICIES:
            raise ValueError(f"unsupported policy: {self.policy}")
        if self.control not in CONTROLS:
            raise ValueError(f"unsupported control: {self.control}")
        if self.use_tof != (self.policy is not None):
            raise ValueError("ToF arms require exactly one scan policy")


@dataclass
class Cell:
    occupied: float = 1.0
    free: float = 1.0
    age: int = 10_000

    @property
    def probability(self) -> float:
        return self.occupied / (self.occupied + self.free)

    @property
    def evidence(self) -> float:
        return self.occupied + self.free - 2.0


@dataclass
class PolarEvidenceMemory:
    """Three-band task memory with metric range bins and pose advection."""

    range_step_m: float = 0.20
    max_range_m: float = 4.0
    decay: float = 0.96
    cells: np.ndarray = field(init=False, repr=False)
    last_pose: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        count = int(math.ceil(self.max_range_m / self.range_step_m))
        self.cells = np.empty((len(BANDS), count), dtype=object)
        for index in np.ndindex(self.cells.shape):
            self.cells[index] = Cell()

    @property
    def range_bins(self) -> int:
        return self.cells.shape[1]

    def begin_frame(self, pose: np.ndarray) -> None:
        if self.last_pose is not None:
            self._warp(self.last_pose, pose)
        for cell in self.cells.flat:
            cell.occupied = 1.0 + (cell.occupied - 1.0) * self.decay
            cell.free = 1.0 + (cell.free - 1.0) * self.decay
            cell.age += 1
        self.last_pose = np.asarray(pose, dtype=np.float64).copy()

    def _warp(self, old_world_from_camera: np.ndarray, new_world_from_camera: np.ndarray) -> None:
        camera_from_world = np.linalg.inv(new_world_from_camera)
        warped = PolarEvidenceMemory(self.range_step_m, self.max_range_m, self.decay)
        for band_index, band in enumerate(BANDS):
            x_ratio = (-0.30, 0.0, 0.30)[band_index]
            for range_index in range(self.range_bins):
                source: Cell = self.cells[band_index, range_index]
                if source.evidence <= 1e-9:
                    continue
                z = (range_index + 0.5) * self.range_step_m
                point_old = np.array([x_ratio * z, 0.0, z, 1.0], dtype=np.float64)
                point_new = camera_from_world @ old_world_from_camera @ point_old
                if point_new[2] <= 0:
                    continue
                new_band = _band_from_ratio(float(point_new[0] / point_new[2]))
                new_range = int(float(point_new[2]) / self.range_step_m)
                if 0 <= new_range < self.range_bins:
                    target: Cell = warped.cells[new_band, new_range]
                    target.occupied += source.occupied - 1.0
                    target.free += source.free - 1.0
                    target.age = min(target.age, source.age)
        self.cells = warped.cells

    def update_ray(self, band_index: int, distance_m: float, free_weight: float, occupied_weight: float) -> None:
        if not math.isfinite(distance_m) or not 0 < distance_m <= self.max_range_m:
            return
        endpoint = min(self.range_bins - 1, int(distance_m / self.range_step_m))
        for range_index in range(endpoint):
            cell: Cell = self.cells[band_index, range_index]
            cell.free += free_weight
            cell.age = 0
        cell = self.cells[band_index, endpoint]
        cell.occupied += occupied_weight
        cell.age = 0

    def band_entropy(self, band_index: int, horizon_m: float = 2.0) -> float:
        end = min(self.range_bins, int(math.ceil(horizon_m / self.range_step_m)))
        values = []
        for range_index in range(end):
            cell: Cell = self.cells[band_index, range_index]
            p = min(1 - 1e-9, max(1e-9, cell.probability))
            values.append(-(p * math.log(p) + (1 - p) * math.log(1 - p)))
        return float(np.mean(values)) if values else 0.0

    def band_staleness(self, band_index: int, horizon_m: float = 2.0) -> float:
        end = min(self.range_bins, int(math.ceil(horizon_m / self.range_step_m)))
        ages = [min(30, self.cells[band_index, index].age) / 30.0 for index in range(end)]
        return float(np.mean(ages)) if ages else 1.0

    def query(self, horizon_m: float) -> tuple[list[str], list[float | None], list[float]]:
        states: list[str] = []
        clearances: list[float | None] = []
        probabilities: list[float] = []
        end = min(self.range_bins, int(math.ceil(horizon_m / self.range_step_m)))
        for band_index in range(len(BANDS)):
            occupied_ranges = []
            all_occupied_ranges = []
            observed = 0
            observed_all = 0
            max_probability = 0.5
            for range_index in range(self.range_bins):
                cell: Cell = self.cells[band_index, range_index]
                if cell.evidence >= 0.35:
                    observed_all += 1
                    if range_index < end:
                        observed += 1
                        max_probability = max(max_probability, cell.probability)
                if cell.evidence >= 0.35 and cell.probability >= 0.58:
                    distance = (range_index + 0.5) * self.range_step_m
                    all_occupied_ranges.append(distance)
                    if range_index < end:
                        occupied_ranges.append(distance)
            coverage = observed / max(1, end)
            coverage_all = observed_all / self.range_bins
            estimated_clearance = (
                min(all_occupied_ranges)
                if all_occupied_ranges
                else (self.max_range_m if coverage_all >= 0.55 else None)
            )
            if occupied_ranges:
                states.append("OCCUPIED")
            elif coverage >= 0.55:
                states.append("CLEAR")
            else:
                states.append("UNKNOWN")
            clearances.append(estimated_clearance)
            probabilities.append(max_probability)
        return states, clearances, probabilities


def _band_from_ratio(x_over_z: float) -> int:
    if x_over_z < -0.12:
        return 0
    if x_over_z > 0.12:
        return 2
    return 1


def _task_clearance(depth: np.ndarray, frame: Frame) -> list[float | None]:
    height, width = depth.shape
    yy, xx = np.mgrid[0:height, 0:width]
    z = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(z) & (z > 0.04) & (z <= 4.0)
    x = (xx - frame.intrinsics.cx) * z / frame.intrinsics.fx
    y = (yy - frame.intrinsics.cy) * z / frame.intrinsics.fy
    gravity = np.asarray(frame.gravity_down_camera, dtype=np.float64)
    drop_from_camera = gravity[0] * x + gravity[1] * y + gravity[2] * z
    above_ground = frame.camera_height_m - drop_from_camera
    valid &= (above_ground >= 0.05) & (above_ground <= 1.65)
    ratios = np.divide(x, z, out=np.zeros_like(x), where=z > 0)
    band_masks = (ratios < -0.12, np.abs(ratios) <= 0.12, ratios > 0.12)
    result: list[float | None] = []
    for mask in band_masks:
        values = z[valid & mask]
        result.append(float(np.quantile(values, 0.05)) if values.size else None)
    return result


def _prior_summary(frame: Frame) -> list[float | None]:
    return _task_clearance(frame.prior_depth_m, frame)


def _roi_bounds(shape: tuple[int, int], roi_index: int, config: TofConfig) -> tuple[slice, slice]:
    height, width = shape
    x0, x1 = config.roi_horizontal_fractions[roi_index]
    y0, y1 = config.roi_vertical_fraction
    return (
        slice(max(0, int(y0 * height)), min(height, int(math.ceil(y1 * height)))),
        slice(max(0, int(x0 * width)), min(width, int(math.ceil(x1 * width)))),
    )


def simulate_tof(frame: Frame, roi_index: int, config: TofConfig, rng: np.random.Generator) -> float | None:
    """Generate a scalar first-return-like range from registered depth truth."""
    ys, xs = _roi_bounds(frame.truth_depth_m.shape, roi_index, config)
    values = np.asarray(frame.truth_depth_m[ys, xs], dtype=np.float64)
    valid = values[np.isfinite(values) & (values >= config.min_range_m) & (values <= config.max_range_m)]
    if not valid.size or rng.random() < config.missing_probability:
        return None
    measured = float(np.quantile(valid, config.first_return_quantile))
    measured += float(rng.normal(0.0, config.noise_sigma_m))
    return min(config.max_range_m, max(config.min_range_m, measured))


def associate_tof_band(frame: Frame, roi_index: int, distance_m: float, config: TofConfig) -> int | None:
    """Associate a cone return to the most likely task-height prior surface."""
    ys, xs = _roi_bounds(frame.prior_depth_m.shape, roi_index, config)
    prior = np.asarray(frame.prior_depth_m[ys, xs], dtype=np.float64)
    if not prior.size:
        return None
    y_grid, x_grid = np.mgrid[ys.start : ys.stop, xs.start : xs.stop]
    valid = np.isfinite(prior) & (prior > 0.04) & (prior <= config.max_range_m)
    if not np.any(valid):
        return roi_index
    confidence = (
        np.ones_like(prior)
        if frame.prior_confidence is None
        else np.clip(np.asarray(frame.prior_confidence[ys, xs], dtype=np.float64), 0.0, 1.0)
    )
    center_x = (xs.start + xs.stop - 1) / 2.0
    center_y = (ys.start + ys.stop - 1) / 2.0
    radial = ((x_grid - center_x) / max(1.0, xs.stop - xs.start)) ** 2
    radial += ((y_grid - center_y) / max(1.0, ys.stop - ys.start)) ** 2
    log_weight = -np.abs(prior - distance_m) / 0.18 + np.log(0.15 + 0.85 * confidence) - 3.0 * radial
    log_weight[~valid] = -np.inf
    flat_index = int(np.argmax(log_weight))
    pixel_y, pixel_x = np.unravel_index(flat_index, prior.shape)
    z = float(prior[pixel_y, pixel_x])
    full_x = float(x_grid[pixel_y, pixel_x])
    full_y = float(y_grid[pixel_y, pixel_x])
    x_m = (full_x - frame.intrinsics.cx) * z / frame.intrinsics.fx
    y_m = (full_y - frame.intrinsics.cy) * z / frame.intrinsics.fy
    gravity = np.asarray(frame.gravity_down_camera, dtype=np.float64)
    drop_from_camera = gravity[0] * x_m + gravity[1] * y_m + gravity[2] * z
    above_ground = frame.camera_height_m - drop_from_camera
    if not 0.05 <= above_ground <= 1.65:
        return None
    return _band_from_ratio(x_m / max(z, 1e-6))


def _choose_roi(
    policy: str,
    memory: PolarEvidenceMemory,
    prior_clearance: Sequence[float | None],
    frame_index: int,
    rng: np.random.Generator,
) -> int:
    if policy == "center_only":
        return 1
    if policy == "random":
        return int(rng.integers(0, len(BANDS)))
    if policy == "round_robin":
        return frame_index % len(BANDS)
    entropy = np.asarray([memory.band_entropy(index) for index in range(len(BANDS))])
    if policy == "max_entropy":
        return int(np.argmax(entropy))
    if policy == "task_weighted_information_gain":
        staleness = np.asarray([memory.band_staleness(index) for index in range(len(BANDS))])
        overlap = np.asarray((0.85, 1.25, 0.85), dtype=np.float64)
        hazard = np.asarray([
            1.0 if value is None else 1.0 + max(0.0, 2.0 - float(value)) / 2.0
            for value in prior_clearance
        ])
        return int(np.argmax(overlap * (0.1 + entropy) * (0.1 + staleness) * hazard))
    raise ValueError(f"unsupported policy: {policy}")


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def default_arms() -> list[ArmConfig]:
    arms = [
        ArmConfig("single_frame_depthart", None, True, False, False),
        ArmConfig("uniform_multiframe_fusion", None, True, True, False),
        ArmConfig("tof_only_round_robin", "round_robin", False, True, True),
    ]
    arms.extend(ArmConfig(f"satom_{policy}", policy, True, True, True) for policy in POLICIES)
    arms.extend(
        ArmConfig(f"satom_task_weighted_information_gain_{control}", "task_weighted_information_gain", True, True, True, control)
        for control in CONTROLS
        if control != "none"
    )
    return arms


def _single_frame_prediction(frame: Frame, horizon_m: float) -> tuple[list[str], list[float | None], list[float]]:
    clearance = _prior_summary(frame)
    states = ["UNKNOWN" if value is None else ("OCCUPIED" if value <= horizon_m else "CLEAR") for value in clearance]
    probabilities = [0.5 if value is None else float(1.0 / (1.0 + math.exp(8.0 * (value - horizon_m)))) for value in clearance]
    return states, clearance, probabilities


def _update_prior(memory: PolarEvidenceMemory, prior_clearance: Sequence[float | None]) -> None:
    for band_index, distance_m in enumerate(prior_clearance):
        if distance_m is not None:
            memory.update_ray(band_index, float(distance_m), free_weight=0.045, occupied_weight=0.16)


def _metric_rows_for_parent(
    frames: Sequence[Frame],
    arm: ArmConfig,
    tof_config: TofConfig,
) -> list[dict[str, Any]]:
    arm.validate()
    memory = PolarEvidenceMemory(max_range_m=tof_config.max_range_m)
    policy_rng = np.random.default_rng(_stable_seed(frames[0].parent_id, arm.name, "policy"))
    delayed_measurements: list[float | None] = []
    rows: list[dict[str, Any]] = []
    for frame in frames:
        frame.validate()
        truth_clearance = _task_clearance(frame.truth_depth_m, frame)
        prior_clearance = _prior_summary(frame)
        if arm.use_memory:
            memory.begin_frame(frame.world_from_camera)
            if arm.use_prior:
                _update_prior(memory, prior_clearance)
        selected_roi: int | None = None
        measurement: float | None = None
        associated_band: int | None = None
        if arm.use_tof and arm.policy is not None:
            selected_roi = _choose_roi(arm.policy, memory, prior_clearance, frame.frame_index, policy_rng)
            measured_roi = (selected_roi + 1) % len(BANDS) if arm.control == "wrong_roi" else selected_roi
            sensor_rng = np.random.default_rng(
                _stable_seed(frame.parent_id, frame.frame_index, measured_roi, "common_sensor")
            )
            fresh_measurement = simulate_tof(frame, measured_roi, tof_config, sensor_rng)
            if arm.control == "shuffled_timestamp":
                delayed_measurements.append(fresh_measurement)
                measurement = delayed_measurements[-3] if len(delayed_measurements) >= 3 else None
            else:
                measurement = fresh_measurement
            if measurement is not None:
                if arm.use_prior:
                    associated_band = associate_tof_band(frame, selected_roi, measurement, tof_config)
                else:
                    associated_band = selected_roi
                if associated_band is not None:
                    if arm.control == "wrong_extrinsic":
                        associated_band = (associated_band + 1) % len(BANDS)
                    memory.update_ray(associated_band, measurement, free_weight=0.32, occupied_weight=1.10)
        for horizon_m in HORIZONS_M:
            if arm.use_memory:
                states, clearance, probabilities = memory.query(horizon_m)
            else:
                states, clearance, probabilities = _single_frame_prediction(frame, horizon_m)
            for band_index, band in enumerate(BANDS):
                truth_value = truth_clearance[band_index]
                if truth_value is None:
                    continue
                rows.append(
                    {
                        "parent_id": frame.parent_id,
                        "frame_index": frame.frame_index,
                        "timestamp_s": frame.timestamp_s,
                        "band": band,
                        "horizon_m": horizon_m,
                        "truth_clearance_m": truth_value,
                        "truth_occupied": truth_value <= horizon_m,
                        "state": states[band_index],
                        "predicted_clearance_m": clearance[band_index],
                        "occupied_probability": probabilities[band_index],
                        "selected_roi": None if selected_roi is None else BANDS[selected_roi],
                        "tof_range_m": measurement,
                        "associated_band": None if associated_band is None else BANDS[associated_band],
                    }
                )
    return rows


def _ece(rows: Sequence[dict[str, Any]], bins: int = 10) -> float | None:
    if not rows:
        return None
    probabilities = np.asarray([row["occupied_probability"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["truth_occupied"] for row in rows], dtype=np.float64)
    result = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = (probabilities >= lower) & ((probabilities <= upper) if index == bins - 1 else (probabilities < upper))
        if np.any(selected):
            result += float(np.mean(selected)) * abs(float(np.mean(probabilities[selected])) - float(np.mean(labels[selected])))
    return result


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no known task opportunities")
    predicted = [row for row in rows if row["state"] != "UNKNOWN"]
    occupied_truth = [row for row in rows if row["truth_occupied"]]
    clear_truth = [row for row in rows if not row["truth_occupied"]]
    clearance_rows = [row for row in predicted if row["predicted_clearance_m"] is not None]
    return {
        "opportunities": len(rows),
        "coverage": len(predicted) / len(rows),
        "false_clear": (
            sum(row["state"] == "CLEAR" for row in occupied_truth) / len(occupied_truth)
            if occupied_truth else None
        ),
        "false_block": (
            sum(row["state"] == "OCCUPIED" for row in clear_truth) / len(clear_truth)
            if clear_truth else None
        ),
        "clearance_mae_m": (
            float(np.mean([abs(row["predicted_clearance_m"] - row["truth_clearance_m"]) for row in clearance_rows]))
            if clearance_rows else None
        ),
        "calibration_error": _ece(rows),
    }


def _mean_optional(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _worst_optional(values: Iterable[float | None], lower_is_worse: bool = False) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return min(finite) if lower_is_worse else max(finite)


def evaluate_frames(
    frames: Sequence[Frame],
    *,
    arms: Sequence[ArmConfig] | None = None,
    tof_config: TofConfig | None = None,
    evidence_role: str,
    prior_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate arms without exposing truth to policies or memory updates."""
    if not frames:
        raise ValueError("frames are required")
    tof_config = tof_config or TofConfig()
    tof_config.validate()
    arms = list(arms or default_arms())
    grouped: dict[str, list[Frame]] = {}
    for frame in frames:
        frame.validate()
        grouped.setdefault(frame.parent_id, []).append(frame)
    for parent_frames in grouped.values():
        parent_frames.sort(key=lambda frame: (frame.timestamp_s, frame.frame_index))
        if len({frame.frame_index for frame in parent_frames}) != len(parent_frames):
            raise ValueError("frame_index must be unique within parent")
    results: dict[str, Any] = {}
    for arm in arms:
        all_rows: list[dict[str, Any]] = []
        by_parent: dict[str, Any] = {}
        for parent_id in sorted(grouped):
            rows = _metric_rows_for_parent(grouped[parent_id], arm, tof_config)
            by_parent[parent_id] = summarize_rows(rows)
            all_rows.extend(rows)
        pooled = summarize_rows(all_rows)
        metric_names = ("coverage", "false_clear", "false_block", "clearance_mae_m", "calibration_error")
        parent_macro = {name: _mean_optional(row[name] for row in by_parent.values()) for name in metric_names}
        worst_parent = {
            name: _worst_optional((row[name] for row in by_parent.values()), lower_is_worse=name == "coverage")
            for name in metric_names
        }
        results[arm.name] = {
            "config": {
                "policy": arm.policy,
                "use_prior": arm.use_prior,
                "use_memory": arm.use_memory,
                "use_tof": arm.use_tof,
                "control": arm.control,
            },
            "pooled": pooled,
            "parent_macro": parent_macro,
            "worst_parent": worst_parent,
            "by_parent": by_parent,
            "scan_trace": [
                {
                    key: row[key]
                    for key in ("parent_id", "frame_index", "selected_roi", "tof_range_m", "associated_band")
                }
                for row in all_rows
                if row["band"] == "center" and row["horizon_m"] == HORIZONS_M[0]
            ],
        }
    return {
        "schema": "blindassist.satom_r0.evaluation.v1",
        "status": "SATOM_R0_MECHANICS_EVALUATED_NO_SCIENTIFIC_CLAIM",
        "evidence_role": evidence_role,
        "claim_ceiling": (
            "deterministic host simulation only; no real ToF, device, trained SATOM model, "
            "Android integration, product, safety, or default-App authority"
        ),
        "causality": {
            "policy_inputs": ["past_memory", "current_frozen_prior_summary", "frame_index", "seeded_rng"],
            "policy_truth_access": False,
            "complete_parent_future_distribution_access": False,
            "negative_controls_are_candidate_arms": False,
        },
        "prior_provenance": prior_provenance,
        "parents": sorted(grouped),
        "frames": len(frames),
        "tof_config": {
            "min_range_m": tof_config.min_range_m,
            "max_range_m": tof_config.max_range_m,
            "noise_sigma_m": tof_config.noise_sigma_m,
            "missing_probability": tof_config.missing_probability,
            "first_return_quantile": tof_config.first_return_quantile,
        },
        "arms": results,
    }


def make_synthetic_frames(parent_count: int = 4, frames_per_parent: int = 24, seed: int = 20260815) -> list[Frame]:
    """Create a causal mechanics canary, never a DepthART or real-data result."""
    rng = np.random.default_rng(seed)
    height, width = 48, 72
    intrinsics = Intrinsics(fx=58.0, fy=58.0, cx=(width - 1) / 2, cy=(height - 1) / 2)
    frames: list[Frame] = []
    for parent_index in range(parent_count):
        parent_id = f"synthetic-parent-{parent_index:02d}"
        obstacle_band = parent_index % len(BANDS)
        obstacle_distance = 0.85 + 0.28 * parent_index
        for frame_index in range(frames_per_parent):
            truth = np.full((height, width), 4.0, dtype=np.float32)
            prior = np.full((height, width), 4.0, dtype=np.float32)
            thirds = ((2, 25), (24, 49), (48, 70))
            x0, x1 = thirds[obstacle_band]
            distance = obstacle_distance + 0.04 * math.sin(frame_index / 4)
            truth[15:40, x0:x1] = distance
            # Frozen-prior-like corruption: parent scale bias plus intermittent local miss.
            scale = 1.18 - 0.08 * (parent_index % 2)
            prior[:] = np.clip(truth * scale + rng.normal(0, 0.035, truth.shape), 0.05, 4.0)
            if frame_index % 5 in (0, 1):
                prior[15:40, x0:x1] = min(4.0, distance + 0.65)
            confidence = np.full_like(prior, 0.75)
            confidence[15:40, x0:x1] = 0.35 if frame_index % 5 in (0, 1) else 0.8
            pose = np.eye(4, dtype=np.float64)
            pose[2, 3] = frame_index * 0.015
            frames.append(
                Frame(
                    parent_id=parent_id,
                    frame_index=frame_index,
                    timestamp_s=frame_index / 10.0,
                    truth_depth_m=truth,
                    prior_depth_m=prior,
                    prior_confidence=confidence,
                    intrinsics=intrinsics,
                    world_from_camera=pose,
                    camera_height_m=1.25,
                    gravity_down_camera=np.array([0.0, 1.0, 0.0], dtype=np.float64),
                )
            )
    return frames
