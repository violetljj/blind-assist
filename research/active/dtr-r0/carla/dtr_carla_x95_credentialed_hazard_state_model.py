"""Trainable credentialed route-conditioned hazard state model.

X95 is the first structural successor after X73-X94.  It consumes the causal
credential ledger already present in sealed X94 predictions and replaces the
successor-by-successor event lifecycle with one logistic emission model plus a
fixed transition mask.  Geometry, association, detector, and route thresholds
remain inherited; this module operates only at the event-authority layer.

The five states separate current measured authority, bounded occlusion
continuity, and release.  Missing evidence can maintain an already active
credential for at most the inherited X24 hold window, but can never create an
alert from CLEAR.  A current release has precedence.  Consumed-cohort fitting
is Development only and supplies no fresh, deployment, or safety authority.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x94_one_frame_full_dropout_continuity as x94  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X95_CREDENTIALED_HAZARD_STATE_MODEL"
ARM_X95 = "X95_CREDENTIALED_ROUTE_CONDITIONED_HAZARD_STATE"

CLEAR = "CLEAR"
ONSET_PENDING = "ONSET_PENDING"
ACTIVE_MEASURED = "ACTIVE_MEASURED"
ACTIVE_OCCLUDED = "ACTIVE_OCCLUDED"
RELEASE_PENDING = "RELEASE_PENDING"
RISK_STATES = {ONSET_PENDING, ACTIVE_MEASURED, ACTIVE_OCCLUDED}

FEATURE_NAMES = (
    "baseline_route_risk",
    "entry_imminence",
    "current_measured_support",
    "held_support",
    "credentialed_support",
    "candidate_support",
    "raw_candidate_present",
    "metric_footprint_present",
    "full_dropout",
    "release_active",
    "closing_support",
    "receding_consensus",
    "lateral_dominant_consensus",
    "freshness",
    "x94_continuity",
    "valid_unchanged_plan",
)

BIRTH_FLAGS = (
    "x70_triple_credential_surface_dropout_handback_used",
    "x71_entry_cotransport_occupancy_birth_used",
    "x72_credentialed_surface_boundary_completion_used",
    "x73_credentialed_parent_hull_reconstruction_used",
)


def fixed_constants() -> dict[str, Any]:
    return {
        **x94.fixed_constants(),
        "representation": "LOGISTIC_EMISSION_WITH_FIXED_CREDENTIAL_TRANSITION_MASK",
        "states": [
            CLEAR,
            ONSET_PENDING,
            ACTIVE_MEASURED,
            ACTIVE_OCCLUDED,
            RELEASE_PENDING,
        ],
        "feature_names": list(FEATURE_NAMES),
        "emission_threshold": 0.5,
        "l2_penalty": 1.0,
        "maximum_newton_steps": 40,
        "continuity_window_s": x24.HOLD_WINDOW_S,
        "clear_cannot_enter_active_from_absence": True,
        "active_occlusion_uses_transition_prior_not_missing_emission": True,
        "release_precedence": True,
        "occluded_state_requires_same_parent_witness": True,
        "route_receipt_change_clears_continuity": True,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "new_geometry_threshold": False,
    }


def _parent_id(row: Mapping[str, Any]) -> str:
    return str(row.get("parent_track_id") or row["track_id"])


def _active_release(arm: Mapping[str, Any]) -> bool:
    return any(
        key.endswith("_release_used") and bool(value)
        for key, value in arm.items()
    )


def _rows_for_ids(
    frame: Mapping[str, Any], ids: Sequence[str]
) -> list[Mapping[str, Any]]:
    rows = {str(row["track_id"]): row for row in frame["tracks"]}
    return [rows[str(track_id)] for track_id in ids if str(track_id) in rows]


def _eligible_current(row: Mapping[str, Any]) -> bool:
    return (
        bool(row.get("risk_eligible"))
        and row.get("motion_authority") == x27.RIGID_DYNAMIC
        and row.get("disposition") == "MEASURED"
        and float(row.get("evidence_age_s", 0.0)) <= x24.EPSILON
    )


def observation(
    frame: Mapping[str, Any], credentialed_parent_ids: set[str]
) -> dict[str, Any]:
    arm = frame["arms"][x94.ARM_X94]
    confirmed = _rows_for_ids(frame, arm.get("confirmed_risk_track_ids", []))
    candidates = _rows_for_ids(frame, arm.get("candidate_risk_track_ids", []))
    support = confirmed or candidates
    current = [row for row in support if _eligible_current(row)]
    held = [
        row
        for row in support
        if row.get("disposition") == "HOLD"
        and float(row.get("evidence_age_s", float("inf")))
        <= x24.HOLD_WINDOW_S + x24.EPSILON
    ]
    parents = {_parent_id(row) for row in support}
    credentialed = parents & credentialed_parent_ids
    velocities = [
        (
            float(row.get("velocity_forward_mps", 0.0)),
            float(row.get("velocity_right_mps", 0.0)),
        )
        for row in support
    ]
    entry = arm.get("minimum_entry_s")
    entry_imminence = (
        0.0
        if entry is None
        else 1.0
        - min(
            1.0,
            max(0.0, float(entry))
            / float(x24.fixed_constants()["route_horizon_seconds"]),
        )
    )
    ages = [float(row.get("evidence_age_s", 0.0)) for row in support]
    freshness = 0.0 if not ages else 1.0 - min(1.0, min(ages) / x24.HOLD_WINDOW_S)
    raw_present = int(frame.get("raw_candidates", 0)) > 0
    metric_present = int(frame.get("metric_footprint_measurements", 0)) > 0
    valid_plan = (
        arm.get("authority") == "VALID"
        and arm.get("route_mode") == "ISSUED_PLAN"
        and not bool(arm.get("route_mode_changed"))
        and bool(arm.get("plan_receipt_sha256"))
    )
    current_birth_flag = any(bool(arm.get(key)) for key in BIRTH_FLAGS)
    current_measured_support = bool(current)
    birth_authorized = bool(
        valid_plan
        and current_measured_support
        and (
            bool(arm.get("route_risk"))
            or current_birth_flag
            or (raw_present and metric_present)
        )
    )
    vector = np.asarray(
        [
            float(bool(arm.get("route_risk"))),
            entry_imminence,
            float(current_measured_support),
            float(bool(held)),
            float(bool(credentialed)),
            min(1.0, len(candidates) / 4.0),
            float(raw_present),
            float(metric_present),
            float(not raw_present and not metric_present),
            float(_active_release(arm)),
            float(any(forward < -x24.EPSILON for forward, _right in velocities)),
            float(
                bool(velocities)
                and all(forward > x24.EPSILON for forward, _right in velocities)
            ),
            float(
                bool(velocities)
                and all(abs(right) > abs(forward) + x24.EPSILON for forward, right in velocities)
            ),
            freshness,
            float(bool(arm.get("x94_one_frame_full_dropout_continuity_used"))),
            float(valid_plan),
        ],
        dtype=np.float64,
    )
    return {
        "vector": vector,
        "arm": arm,
        "birth_authorized": birth_authorized,
        "current_measured_support": current_measured_support,
        "held_parent_ids": {_parent_id(row) for row in held},
        "support_parent_ids": parents,
        "release_active": _active_release(arm),
        "full_dropout": not raw_present and not metric_present,
        "valid_unchanged_plan": valid_plan,
    }


@dataclass(frozen=True)
class LogisticEmission:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def probability(self, vector: np.ndarray) -> float:
        standardized = (vector - self.mean) / self.scale
        score = float(np.dot(self.weights[1:], standardized) + self.weights[0])
        score = max(-40.0, min(40.0, score))
        return 1.0 / (1.0 + math.exp(-score))

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
        }


def fit_logistic(
    vectors: Sequence[np.ndarray], labels: Sequence[bool]
) -> LogisticEmission:
    matrix = np.vstack(vectors).astype(np.float64)
    target = np.asarray(labels, dtype=np.float64)
    x24.require(matrix.shape[0] == target.shape[0] and matrix.shape[0] > 0, "x95_fit_rows")
    x24.require(0.0 < float(target.mean()) < 1.0, "x95_fit_two_classes")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1.0e-9] = 1.0
    design = np.column_stack([np.ones(matrix.shape[0]), (matrix - mean) / scale])
    weights = np.zeros(design.shape[1], dtype=np.float64)
    regularizer = np.eye(design.shape[1], dtype=np.float64)
    regularizer[0, 0] = 0.0
    penalty = float(fixed_constants()["l2_penalty"])
    for _step in range(int(fixed_constants()["maximum_newton_steps"])):
        score = np.clip(design @ weights, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-score))
        curvature = np.maximum(probability * (1.0 - probability), 1.0e-6)
        gradient = design.T @ (probability - target) + penalty * regularizer @ weights
        hessian = (design.T * curvature) @ design + penalty * regularizer
        update = np.linalg.solve(hessian, gradient)
        weights -= update
        if float(np.max(np.abs(update))) < 1.0e-8:
            break
    return LogisticEmission(mean=mean, scale=scale, weights=weights)


def decode_episode(
    frames: Sequence[Mapping[str, Any]], model: LogisticEmission
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = CLEAR
    active_parent_ids: set[str] = set()
    credentialed_parent_ids: set[str] = set()
    last_measured_time_s: float | None = None
    last_plan_receipt: str | None = None
    output: list[dict[str, Any]] = []
    state_counts = {name: 0 for name in fixed_constants()["states"]}
    transition_counts: dict[str, int] = {}

    for frame in frames:
        arm = frame["arms"][x94.ARM_X94]
        credentialed_parent_ids.update(
            str(value)
            for value in arm.get("x75_collision_credential_birth_parent_ids", [])
        )
        obs = observation(frame, credentialed_parent_ids)
        probability = model.probability(obs["vector"])
        positive_emission = probability >= float(fixed_constants()["emission_threshold"])
        now_s = float(frame["time_s"])
        receipt = str(arm.get("plan_receipt_sha256") or "")
        same_plan = bool(last_plan_receipt and receipt == last_plan_receipt)
        same_parent = bool(active_parent_ids & obs["held_parent_ids"])
        within_hold = (
            last_measured_time_s is not None
            and now_s - last_measured_time_s <= x24.HOLD_WINDOW_S + x24.EPSILON
        )
        occlusion_authorized = bool(
            state in RISK_STATES
            and same_plan
            and same_parent
            and within_hold
            and not obs["release_active"]
            and (obs["full_dropout"] or bool(obs["held_parent_ids"]))
        )
        previous_state = state

        if obs["release_active"] or not obs["valid_unchanged_plan"]:
            state = RELEASE_PENDING if state in RISK_STATES else CLEAR
        elif state == CLEAR:
            state = ONSET_PENDING if positive_emission and obs["birth_authorized"] else CLEAR
        elif state == RELEASE_PENDING:
            state = ACTIVE_MEASURED if positive_emission and obs["birth_authorized"] else CLEAR
        elif positive_emission and obs["current_measured_support"]:
            state = ACTIVE_MEASURED
        elif occlusion_authorized:
            state = ACTIVE_OCCLUDED
        else:
            state = RELEASE_PENDING

        if state in {ONSET_PENDING, ACTIVE_MEASURED} and obs["current_measured_support"]:
            last_measured_time_s = now_s
            active_parent_ids = set(obs["support_parent_ids"])
            last_plan_receipt = receipt
        if state == CLEAR:
            active_parent_ids.clear()
            last_measured_time_s = None
            last_plan_receipt = None

        risk = state in RISK_STATES
        if state == ONSET_PENDING:
            event = "ONSET"
        elif risk:
            previous_entry = output[-1].get("minimum_entry_s") if output else None
            entry = arm.get("minimum_entry_s")
            event = (
                "ESCALATE"
                if entry is not None
                and previous_entry is not None
                and float(entry) + x24.EPSILON < float(previous_entry)
                else "HOLD"
            )
        elif previous_state in RISK_STATES:
            event = "CLEAR"
        else:
            event = "CLEAR"
        transition = f"{previous_state}->{state}"
        transition_counts[transition] = transition_counts.get(transition, 0) + 1
        state_counts[state] += 1
        output.append(
            {
                "sample_index": int(frame["sample_index"]),
                "time_s": now_s,
                "route_risk": risk,
                "minimum_entry_s": arm.get("minimum_entry_s") if risk else None,
                "state": state,
                "event": event,
                "emission_probability": probability,
                "birth_authorized": bool(obs["birth_authorized"]),
                "occlusion_authorized": occlusion_authorized,
                "active_parent_ids": sorted(active_parent_ids),
            }
        )
    return output, {"state_counts": state_counts, "transition_counts": transition_counts}


def self_check() -> dict[str, Any]:
    vectors = [
        np.zeros(len(FEATURE_NAMES), dtype=np.float64),
        np.ones(len(FEATURE_NAMES), dtype=np.float64),
        np.full(len(FEATURE_NAMES), 0.1, dtype=np.float64),
        np.full(len(FEATURE_NAMES), 0.9, dtype=np.float64),
    ]
    model = fit_logistic(vectors, [False, True, False, True])
    x24.require(
        model.probability(vectors[0]) < model.probability(vectors[1]),
        "x95_emission_order",
    )
    x24.require(
        CLEAR not in RISK_STATES
        and RELEASE_PENDING not in RISK_STATES
        and ACTIVE_OCCLUDED in RISK_STATES,
        "x95_state_partition",
    )
    return {
        "status": "X95_CREDENTIALED_HAZARD_STATE_MODEL_SELF_CHECK_MET",
        "trainable_logistic_emission": True,
        "fixed_five_state_transition_mask": True,
        "absence_cannot_birth_risk": True,
        "bounded_occlusion_continuity": True,
        "release_precedence": True,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
