"""X43 authority-preserving metric-credential belief continuation.

X43 preserves X42's risk decision while repairing one representation error.  A
continued risk must not be referenced to a current STATIC_SCENE surface branch.
If X42 does so, X43 carries forward the immediately previous authorized
RIGID_DYNAMIC confirmed hypothesis as an explicit belief track.  The belief is
not relabelled as a current measurement and the conflicting surface branch
remains in the frame as separate evidence.

The rule is causal and class-independent.  It adds no numeric speed, score,
duration, detector, association, or route threshold.

C20 is consumed posthoc synthetic Development and cannot confirm X43.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x31_ambiguity_preserving_transport_predictor as x31  # noqa: E402
import dtr_carla_x32_observation_conditioned_core_predictor as x32  # noqa: E402
import dtr_carla_x42_instant_closing_consensus_predictor as x42  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X43_AUTHORITY_PRESERVING_CREDENTIAL_BELIEF"
ARM_X43 = "X43_ISSUED_PLAN_AUTHORITY_PRESERVING_CREDENTIAL_BELIEF"


def fixed_constants() -> dict[str, Any]:
    return {
        **x42.fixed_constants(),
        "representation": "DUAL_REPRESENTATION_AUTHORIZED_CREDENTIAL_BELIEF",
        "credential_authority_rule": (
            "NON_RIGID_CONFIRMED_REFERENCE_REPLACED_BY_IMMEDIATELY_PREVIOUS_"
            "AUTHORIZED_RIGID_DYNAMIC_BELIEF"
        ),
        "belief_disposition": "CREDENTIALED_DYNAMIC_BELIEF_HOLD",
        "belief_is_current_measurement": False,
        "surface_identity_merge": False,
        "credential_class_rule": "CLASS_INDEPENDENT",
        "credential_numeric_speed_threshold": None,
        "credential_duration_threshold": None,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "score_threshold_change": False,
    }


def authorized_confirmed_rows(
    arm: Mapping[str, Any], tracks: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for track_id in arm.get("confirmed_risk_track_ids", []):
        row = tracks.get(str(track_id))
        if (
            row is not None
            and row.get("motion_authority") == x27.RIGID_DYNAMIC
            and bool(row.get("risk_eligible", False))
        ):
            rows.append(dict(row))
    return rows


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    value = x42.predict_episode(episode, candidate_values, calibration)
    carrier_frames = 0
    previous_authorized_rows: list[dict[str, Any]] = []

    for frame in value["frames"]:
        arm = frame["arms"][x42.ARM_X42]
        tracks = {str(row["track_id"]): row for row in frame["tracks"]}
        current_authorized = authorized_confirmed_rows(arm, tracks)
        authority_defect = bool(arm.get("route_risk")) and not current_authorized
        if authority_defect and previous_authorized_rows:
            carrier_frames += 1
            carriers: list[dict[str, Any]] = []
            for previous in previous_authorized_rows:
                track_id = str(previous["track_id"])
                carrier = {
                    **previous,
                    "disposition": "CREDENTIALED_DYNAMIC_BELIEF_HOLD",
                    "motion_authority": x27.RIGID_DYNAMIC,
                    "risk_eligible": True,
                    "x43_belief_carrier": True,
                    "x43_belief_source_track_id": track_id,
                }
                if track_id not in tracks:
                    frame["tracks"].append(carrier)
                    frame["risk_eligible_tracks"] = int(
                        frame["risk_eligible_tracks"]
                    ) + 1
                else:
                    tracks[track_id].update(carrier)
                carriers.append(carrier)
            track_ids = sorted(str(row["track_id"]) for row in carriers)
            parent_ids = sorted(str(row["parent_track_id"]) for row in carriers)
            arm.update(
                {
                    "candidate_risk_track_ids": track_ids,
                    "confirmed_risk_track_ids": track_ids,
                    "candidate_risk_parent_track_ids": parent_ids,
                    "confirmed_risk_parent_track_ids": parent_ids,
                    "x43_authority_belief_carrier_used": True,
                    "x43_authority_belief_carrier_track_ids": track_ids,
                }
            )
            current_authorized = carriers
        else:
            arm["x43_authority_belief_carrier_used"] = False
            arm["x43_authority_belief_carrier_track_ids"] = []

        previous_authorized_rows = (
            current_authorized if bool(arm.get("route_risk")) else []
        )

    value["arms"][ARM_X43] = value["arms"].pop(x42.ARM_X42)
    value["diagnostics"]["x43_route_mode_counts"] = value["diagnostics"].pop(
        "x42_route_mode_counts"
    )
    value["diagnostics"]["authority_belief_carrier_frames"] = carrier_frames
    for frame in value["frames"]:
        frame["arms"][ARM_X43] = frame["arms"].pop(x42.ARM_X42)
    return value


def self_check() -> dict[str, Any]:
    inherited = x42.self_check()
    arm = {
        "route_risk": True,
        "confirmed_risk_track_ids": ["track-a"],
    }
    rows = {
        "track-a": {
            "track_id": "track-a",
            "motion_authority": x27.RIGID_DYNAMIC,
            "risk_eligible": True,
        }
    }
    x24.require(
        len(authorized_confirmed_rows(arm, rows)) == 1,
        "x43_authorized_confirmed_projection",
    )
    return {
        "status": "X43_AUTHORITY_PRESERVING_CREDENTIAL_BELIEF_FALSIFIER_MET",
        "x42_structural_status": inherited["status"],
        "current_static_surface_identity_merge": False,
        "previous_authorized_dynamic_belief_required": True,
        "belief_is_current_measurement": False,
        "class_independent": True,
        "numeric_speed_threshold_added": False,
    }
