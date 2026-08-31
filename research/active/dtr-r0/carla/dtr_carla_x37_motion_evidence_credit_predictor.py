"""X37 reuse motion consensus as route-confirmation evidence.

The inherited rigid-dynamic authority already requires a multi-observation
causal motion history.  X37 lets a currently measured route candidate consume
that evidence once instead of always waiting for an additional route-candidate
frame.  Credit is limited to candidate branches in the parent's modal live
velocity group.  Minority motion hypotheses and non-measured rows still use the
ordinary consecutive route confirmation unchanged.

C16 is consumed same-source synthetic Development and cannot confirm X37.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x24_plan_route_core as route  # noqa: E402
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x30_adaptive_surface_interval_predictor as x30  # noqa: E402
import dtr_carla_x31_ambiguity_preserving_transport_predictor as x31  # noqa: E402
import dtr_carla_x35_dormant_flow_consensus_predictor as x35  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X37_MOTION_EVIDENCE_CONFIRMATION_CREDIT"
ARM_X37 = "X37_ISSUED_PLAN_MOTION_EVIDENCE_CONFIRMATION_CREDIT"
BASE_ARM_FRAME = x31.ambiguity_preserving_arm_frame


def fixed_constants() -> dict[str, Any]:
    return {
        **x35.fixed_constants(),
        "representation": "MOTION_EVIDENCE_ROUTE_CONFIRMATION_CREDIT",
        "credit_eligibility": (
            "CURRENTLY_MEASURED_RIGID_DYNAMIC_MODAL_VELOCITY_CANDIDATE"
        ),
        "credit_source": "INHERITED_MULTI_OBSERVATION_MOTION_AUTHORITY",
        "minority_velocity_confirmation": "INHERITED_CONSECUTIVE_ROUTE_CONFIRMATION",
        "hold_confirmation": "INHERITED_CONSECUTIVE_ROUTE_CONFIRMATION",
        "confirmation_numeric_threshold_change": False,
        "detector_threshold_change": False,
        "association_radius_change": False,
        "route_geometry_change": False,
        "score_threshold_change": False,
    }


def velocity_key(row: Mapping[str, Any]) -> tuple[float, float]:
    return (
        float(round(float(row["velocity_forward_mps"]), 9)),
        float(round(float(row["velocity_right_mps"]), 9)),
    )


def modal_velocity_keys(rows: Sequence[Mapping[str, Any]]) -> set[tuple[float, float]]:
    counts = Counter(velocity_key(row) for row in rows)
    if not counts:
        return set()
    maximum = max(counts.values())
    return {key for key, count in counts.items() if count == maximum}


def motion_evidence_credit_arm_frame(
    selection: route.RouteSelection,
    *,
    receipt: Mapping[str, Any] | None,
    observation: Any,
    wearer_position: tuple[float, float],
    wearer_velocity: tuple[float, float],
    tracks: Sequence[Mapping[str, Any]],
    confirmation: x24.RiskConfirmation,
    sample_period_s: float,
) -> dict[str, Any]:
    value = BASE_ARM_FRAME(
        selection,
        receipt=receipt,
        observation=observation,
        wearer_position=wearer_position,
        wearer_velocity=wearer_velocity,
        tracks=tracks,
        confirmation=confirmation,
        sample_period_s=sample_period_s,
    )
    candidate_ids = set(value.get("candidate_risk_track_ids", []))
    if not candidate_ids:
        value["motion_evidence_credit_used"] = False
        value["motion_evidence_credit_parent_track_ids"] = []
        return value

    by_parent: dict[str, list[Mapping[str, Any]]] = {}
    for row in tracks:
        parent_id = str(row.get("parent_track_id", row["track_id"]))
        by_parent.setdefault(parent_id, []).append(row)

    credited: list[Mapping[str, Any]] = []
    for rows in by_parent.values():
        eligible = [
            row
            for row in rows
            if row.get("disposition") == "MEASURED"
            and row.get("motion_authority") == x27.RIGID_DYNAMIC
            and int(row.get("transport_lineage_pairs", 0))
            >= x27.MINIMUM_AUTHORITY_PAIRS
        ]
        modal = modal_velocity_keys(eligible)
        credited.extend(
            row
            for row in eligible
            if str(row["track_id"]) in candidate_ids
            and velocity_key(row) in modal
        )
    if not credited:
        value["motion_evidence_credit_used"] = False
        value["motion_evidence_credit_parent_track_ids"] = []
        return value

    segments = route.build_route_segments(
        selection,
        receipt=receipt,
        now_s=observation.time_s,
        wearer_position_xy=wearer_position,
        wearer_velocity_xy=wearer_velocity,
    )
    credit_entries = [
        x30.first_contact_interval_entry_s(
            row["footprint_xy"],
            (row["velocity_forward_mps"], row["velocity_right_mps"]),
            segments,
        )
        for row in credited
    ]
    credited = [
        row for row, entry in zip(credited, credit_entries, strict=True) if entry is not None
    ]
    credit_entries = [entry for entry in credit_entries if entry is not None]
    credit_parents = sorted(
        {str(row.get("parent_track_id", row["track_id"])) for row in credited}
    )
    confirmed_ids = set(value.get("confirmed_risk_track_ids", []))
    confirmed_parents = set(value.get("confirmed_risk_parent_track_ids", []))
    confirmed_ids.update(str(row["track_id"]) for row in credited)
    confirmed_parents.update(credit_parents)
    existing_entry = value.get("minimum_entry_s")
    entries = [float(entry) for entry in credit_entries]
    if existing_entry is not None:
        entries.append(float(existing_entry))
    value.update(
        {
            "route_risk": bool(confirmed_ids),
            "minimum_entry_s": min(entries) if entries else existing_entry,
            "confirmed_risk_track_ids": sorted(confirmed_ids),
            "confirmed_risk_parent_track_ids": sorted(confirmed_parents),
            "motion_evidence_credit_used": bool(credit_parents),
            "motion_evidence_credit_parent_track_ids": credit_parents,
        }
    )
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    original_arm_frame = x31.ambiguity_preserving_arm_frame
    x31.ambiguity_preserving_arm_frame = motion_evidence_credit_arm_frame
    try:
        value = x35.predict_episode(episode, candidate_values, calibration)
    finally:
        x31.ambiguity_preserving_arm_frame = original_arm_frame
    value["arms"][ARM_X37] = value["arms"].pop(x35.ARM_X35)
    value["diagnostics"]["x37_route_mode_counts"] = value["diagnostics"].pop(
        "x35_route_mode_counts"
    )
    value["diagnostics"]["motion_evidence_credit_frames"] = sum(
        bool(frame["arms"][x35.ARM_X35].get("motion_evidence_credit_used"))
        for frame in value["frames"]
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X37] = frame["arms"].pop(x35.ARM_X35)
    return value


def self_check() -> dict[str, Any]:
    inherited = x35.self_check()
    rows = [
        {"velocity_forward_mps": -1.625, "velocity_right_mps": 0.0},
        {"velocity_forward_mps": -1.625, "velocity_right_mps": 0.0},
        {"velocity_forward_mps": -1.625, "velocity_right_mps": 1.625},
    ]
    x24.require(
        modal_velocity_keys(rows) == {(-1.625, 0.0)},
        "x37_modal_velocity_credit",
    )
    return {
        "status": "X37_MOTION_EVIDENCE_CONFIRMATION_CREDIT_STRUCTURAL_FALSIFIER_MET",
        "x35_structural_status": inherited["status"],
        "motion_authority_reused": True,
        "minority_velocity_requires_ordinary_confirmation": True,
        "numeric_confirmation_threshold_change": False,
    }
