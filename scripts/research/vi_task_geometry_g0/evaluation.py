"""Truth-isolated A0-A3 evaluator for VI-Task Geometry G0 mechanics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Any, Iterable


ARM_IDS = ("A0", "A1", "A2", "A3")
PRIMARY_ARM_IDS = ("A2", "A3")
OBSERVABLE_EPISODE_TYPES = {
    "EXCITED_WALK_TURN",
    "NATURAL_HEAD_MOTION",
    "POSTURE_HEIGHT_CHANGE",
    "STAIRS_OR_RAMP",
}
DEGENERACY_EPISODE_TYPES = {"STRAIGHT_LOW_EXCITATION_CONTROL", "STATIC_CONTROL"}


@dataclass(frozen=True)
class EvaluationPolicy:
    observable_parent_macro_coverage_min: float = 0.80
    observable_worst_parent_coverage_min: float = 0.65
    parent_macro_camera_height_mae_m_max: float = 0.10
    worst_parent_camera_height_mae_m_max: float = 0.15
    worst_parent_height_error_mad_m_max: float = 0.08
    worst_parent_scale_drift_p95_fraction_max: float = 0.08
    parent_macro_sparse_anchor_absrel_max: float = 0.12
    worst_parent_false_wall_ground_support_rate_max: float = 0.05
    parent_macro_false_wall_ground_support_rate_max: float = 0.02
    worst_parent_degeneracy_unsafe_valid_rate_max: float = 0.05


@dataclass(frozen=True)
class CandidateFrame:
    arm_id: str
    parent_id: str
    episode_id: str
    frame_id: str
    status: str
    camera_height_m: float | None
    sparse_depth_m: dict[str, float]
    ground_support_ids: tuple[str, ...]
    clearance_m: float | None = None

    @property
    def valid(self) -> bool:
        return self.status == "VALID_METRIC_GEOMETRY"

    def validate(self) -> None:
        if self.arm_id not in ARM_IDS or not self.parent_id or not self.episode_id or not self.frame_id:
            raise ValueError("candidate identity is invalid")
        if self.status != "VALID_METRIC_GEOMETRY" and not self.status.startswith("UNKNOWN_"):
            raise ValueError("candidate status must be metric-valid or fail-closed UNKNOWN")
        if self.valid:
            if self.camera_height_m is None or not math.isfinite(self.camera_height_m) or not 0.4 <= self.camera_height_m <= 2.4:
                raise ValueError("metric-valid candidate requires a plausible finite camera height")
            if not self.sparse_depth_m or not self.ground_support_ids:
                raise ValueError("metric-valid candidate requires sparse geometry and ground support")
        for value in self.sparse_depth_m.values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError("candidate sparse depth must be positive and finite")
        if len(self.ground_support_ids) != len(set(self.ground_support_ids)):
            raise ValueError("candidate ground support IDs must be unique")
        if self.clearance_m is not None and (not math.isfinite(self.clearance_m) or self.clearance_m < 0):
            raise ValueError("candidate clearance must be nonnegative and finite")


@dataclass(frozen=True)
class TruthFrame:
    parent_id: str
    episode_id: str
    frame_id: str
    reference_camera_height_m: float | None
    sparse_depth_m: dict[str, float]
    ground_labels: dict[str, str]
    clearance_m: float | None = None

    def validate(self) -> None:
        if not self.parent_id or not self.episode_id or not self.frame_id:
            raise ValueError("truth identity is invalid")
        if self.reference_camera_height_m is not None and (
            not math.isfinite(self.reference_camera_height_m)
            or not 0.4 <= self.reference_camera_height_m <= 2.4
        ):
            raise ValueError("truth camera height is invalid")
        for value in self.sparse_depth_m.values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError("truth sparse depth must be positive and finite")
        if any(label not in {"GROUND", "NON_GROUND", "UNRESOLVED"} for label in self.ground_labels.values()):
            raise ValueError("truth ground label is invalid")
        if self.clearance_m is not None and (not math.isfinite(self.clearance_m) or self.clearance_m < 0):
            raise ValueError("truth clearance must be nonnegative and finite")


def _mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return sum(rows) / len(rows) if rows else None


def _quantile(values: Iterable[float], q: float) -> float | None:
    rows = sorted(values)
    if not rows:
        return None
    position = (len(rows) - 1) * q
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return rows[left]
    fraction = position - left
    return rows[left] * (1.0 - fraction) + rows[right] * fraction


def _parent_metrics(rows: list[tuple[CandidateFrame, TruthFrame, str]]) -> dict[str, Any]:
    observable = [row for row in rows if row[2] in OBSERVABLE_EPISODE_TYPES and row[1].reference_camera_height_m is not None]
    valid = [row for row in observable if row[0].valid]
    height_errors = [
        abs(float(candidate.camera_height_m) - float(truth.reference_camera_height_m))
        for candidate, truth, _ in valid
    ]
    signed_height_errors = [
        float(candidate.camera_height_m) - float(truth.reference_camera_height_m)
        for candidate, truth, _ in valid
    ]
    scale_errors_by_episode: dict[str, list[float]] = {}
    sparse_absrel = []
    false_support = 0
    resolved_support = 0
    clearance_errors = []
    false_clear = 0
    false_block = 0
    clearance_count = 0
    for candidate, truth, _ in valid:
        scale_errors_by_episode.setdefault(candidate.episode_id, []).append(
            float(candidate.camera_height_m) / float(truth.reference_camera_height_m) - 1.0
        )
        for anchor_id, predicted in candidate.sparse_depth_m.items():
            if anchor_id not in truth.sparse_depth_m:
                raise ValueError("candidate sparse anchor lacks evaluator truth identity")
            reference = truth.sparse_depth_m[anchor_id]
            sparse_absrel.append(abs(predicted - reference) / reference)
    for candidate, truth, _ in (row for row in rows if row[0].valid):
        for support_id in candidate.ground_support_ids:
            if support_id not in truth.ground_labels:
                raise ValueError("candidate ground support lacks evaluator label identity")
            label = truth.ground_labels[support_id]
            if label != "UNRESOLVED":
                resolved_support += 1
                false_support += int(label == "NON_GROUND")
        if candidate.clearance_m is not None and truth.clearance_m is not None:
            predicted = candidate.clearance_m
            reference = truth.clearance_m
            clearance_errors.append(abs(predicted - reference))
            false_clear += int(predicted >= 0.60 and reference < 0.45)
            false_block += int(predicted < 0.45 and reference >= 0.60)
            clearance_count += 1
    scale_drift = []
    for values in scale_errors_by_episode.values():
        center = median(values)
        scale_drift.extend(abs(value - center) for value in values)
    degeneracy = [row for row in rows if row[2] in DEGENERACY_EPISODE_TYPES]
    center = median(signed_height_errors) if signed_height_errors else None
    return {
        "observable_frames": len(observable),
        "observable_valid_frames": len(valid),
        "observable_coverage": len(valid) / len(observable) if observable else None,
        "camera_height_mae_m": _mean(height_errors),
        "height_error_mad_m": (
            median(abs(value - float(center)) for value in signed_height_errors)
            if signed_height_errors else None
        ),
        "scale_drift_p95_fraction": _quantile(scale_drift, 0.95),
        "sparse_anchor_absrel": _mean(sparse_absrel),
        "false_wall_ground_support_rate": false_support / resolved_support if resolved_support else None,
        "degeneracy_unsafe_valid_rate": (
            sum(candidate.valid for candidate, _, _ in degeneracy) / len(degeneracy)
            if degeneracy else None
        ),
        "diagnostic_clearance_mae_m": _mean(clearance_errors),
        "diagnostic_false_clear_rate": false_clear / clearance_count if clearance_count else None,
        "diagnostic_false_block_rate": false_block / clearance_count if clearance_count else None,
    }


def evaluate_g0(
    candidates: list[CandidateFrame],
    truth: list[TruthFrame],
    episode_types: dict[str, str],
    policy: EvaluationPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or EvaluationPolicy()
    truth_by_id: dict[tuple[str, str, str], TruthFrame] = {}
    for row in truth:
        row.validate()
        key = (row.parent_id, row.episode_id, row.frame_id)
        if key in truth_by_id:
            raise ValueError("duplicate truth identity")
        truth_by_id[key] = row
    if not truth_by_id:
        raise ValueError("VITG G0 truth is required by the evaluator")
    by_arm: dict[str, list[CandidateFrame]] = {arm: [] for arm in ARM_IDS}
    seen = set()
    for row in candidates:
        row.validate()
        key = (row.arm_id, row.parent_id, row.episode_id, row.frame_id)
        if key in seen:
            raise ValueError("duplicate candidate identity")
        seen.add(key)
        by_arm[row.arm_id].append(row)
    truth_keys = set(truth_by_id)
    for arm, rows in by_arm.items():
        arm_keys = {(row.parent_id, row.episode_id, row.frame_id) for row in rows}
        if arm_keys != truth_keys:
            raise ValueError(f"{arm} candidate ledger does not exactly match evaluator truth identities")
    result_by_arm = {}
    for arm, arm_rows in by_arm.items():
        parent_rows: dict[str, list[tuple[CandidateFrame, TruthFrame, str]]] = {}
        for candidate in arm_rows:
            key = (candidate.parent_id, candidate.episode_id, candidate.frame_id)
            if candidate.episode_id not in episode_types:
                raise ValueError("candidate episode lacks a frozen episode type")
            parent_rows.setdefault(candidate.parent_id, []).append(
                (candidate, truth_by_id[key], episode_types[candidate.episode_id])
            )
        by_parent = {parent: _parent_metrics(rows) for parent, rows in sorted(parent_rows.items())}
        metric_names = (
            "observable_coverage", "camera_height_mae_m", "height_error_mad_m",
            "scale_drift_p95_fraction", "sparse_anchor_absrel",
            "false_wall_ground_support_rate", "degeneracy_unsafe_valid_rate",
        )
        macro = {name: _mean(value[name] for value in by_parent.values() if value[name] is not None) for name in metric_names}
        worst = {}
        for name in metric_names:
            values = [value[name] for value in by_parent.values() if value[name] is not None]
            worst[name] = (min(values) if name == "observable_coverage" else max(values)) if values else None
        available = lambda name: macro[name] is not None and worst[name] is not None
        gates = {
            "parent_count_exactly_8": len(by_parent) == 8,
            "observable_parent_macro_coverage": available("observable_coverage") and macro["observable_coverage"] >= policy.observable_parent_macro_coverage_min,
            "observable_worst_parent_coverage": available("observable_coverage") and worst["observable_coverage"] >= policy.observable_worst_parent_coverage_min,
            "parent_macro_camera_height_mae": available("camera_height_mae_m") and macro["camera_height_mae_m"] <= policy.parent_macro_camera_height_mae_m_max,
            "worst_parent_camera_height_mae": available("camera_height_mae_m") and worst["camera_height_mae_m"] <= policy.worst_parent_camera_height_mae_m_max,
            "worst_parent_height_error_mad": available("height_error_mad_m") and worst["height_error_mad_m"] <= policy.worst_parent_height_error_mad_m_max,
            "worst_parent_scale_drift_p95": available("scale_drift_p95_fraction") and worst["scale_drift_p95_fraction"] <= policy.worst_parent_scale_drift_p95_fraction_max,
            "parent_macro_sparse_anchor_absrel": available("sparse_anchor_absrel") and macro["sparse_anchor_absrel"] <= policy.parent_macro_sparse_anchor_absrel_max,
            "worst_parent_false_wall_ground_support": available("false_wall_ground_support_rate") and worst["false_wall_ground_support_rate"] <= policy.worst_parent_false_wall_ground_support_rate_max,
            "parent_macro_false_wall_ground_support": available("false_wall_ground_support_rate") and macro["false_wall_ground_support_rate"] <= policy.parent_macro_false_wall_ground_support_rate_max,
            "worst_parent_degeneracy_unsafe_valid": available("degeneracy_unsafe_valid_rate") and worst["degeneracy_unsafe_valid_rate"] <= policy.worst_parent_degeneracy_unsafe_valid_rate_max,
        }
        result_by_arm[arm] = {
            "passed_absolute_gates": arm in PRIMARY_ARM_IDS and all(gates.values()),
            "parent_macro": macro,
            "worst_parent": worst,
            "by_parent": by_parent,
            "gates": gates,
        }
    winners = [arm for arm in PRIMARY_ARM_IDS if result_by_arm[arm]["passed_absolute_gates"]]
    passed = bool(winners)
    return {
        "schema": "blindassist.vitg_g0.evaluation.v1",
        "status": "VITG_G0_METRIC_FRAME_OBSERVABILITY_PASS" if passed else "VITG_G0_METRIC_FRAME_OBSERVABILITY_FAIL_CLOSE",
        "passed": passed,
        "eligible_primary_arms_passing": winners,
        "arms": result_by_arm,
        "causality": {"candidate_truth_access": False, "truth_role": "evaluator_only"},
        "clearance_metrics_affect_g0_pass": False,
        "claim_ceiling": "metric camera-height and sparse-ground observability only; no clearance, active allocation, Android, product, safety or paper claim",
    }
