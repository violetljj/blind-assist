"""X41 metric-credentialed continuation of an adjudicated lateral parent.

X41 lets X40 suppress uncorroborated lateral-only surface risk by default.  A
suppressed lateral parent may continue only if that exact parent previously
issued X40 risk while an X24 route candidate was present, and the same parent
also carried X41 risk in the immediately preceding frame.  Thus metric support
can issue a credential but cannot create an alert by itself; parent continuity
can bridge a later metric dropout but cannot restart after a gap.

The rule is causal, class-independent, and adds no numeric speed, score,
duration, detector, association, or route threshold.

C16 is consumed same-source synthetic Development and cannot confirm X41.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x31_ambiguity_preserving_transport_predictor as x31  # noqa: E402
import dtr_carla_x40_cross_representation_lateral_adjudicator as x40  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X41_METRIC_CREDENTIALED_PARENT_CONTINUATION"
ARM_X41 = "X41_ISSUED_PLAN_METRIC_CREDENTIALED_PARENT_CONTINUATION"


def fixed_constants() -> dict[str, Any]:
    return {
        **x40.fixed_constants(),
        "representation": "DUAL_REPRESENTATION_METRIC_CREDENTIALED_CONTINUATION",
        "credential_issuer": "CURRENT_X24_ROUTE_RISK_CANDIDATE",
        "credential_subject": "EXACT_SURFACE_PARENT_IDENTITY",
        "continuation_requirement": "IMMEDIATELY_PREVIOUS_X41_RISK_SAME_PARENT",
        "continuation_can_originate_risk": False,
        "credential_class_rule": "CLASS_INDEPENDENT",
        "credential_numeric_speed_threshold": None,
        "credential_duration_threshold": None,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "score_threshold_change": False,
    }


def parent_ids(
    track_ids: Sequence[str], tracks: Mapping[str, Mapping[str, Any]]
) -> set[str]:
    return {
        str(tracks[track_id]["parent_track_id"])
        for track_id in track_ids
        if track_id in tracks
    }


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    value = x40.predict_episode(episode, candidate_values, calibration)
    metric = x24.predict_episode(episode, candidate_values, calibration)
    credentialed_parents: set[str] = set()
    previous_x41_risk = False
    previous_x41_parents: set[str] = set()
    continuation_frames = 0

    for fused_frame, metric_frame in zip(
        value["frames"], metric["frames"], strict=True
    ):
        x24.require(
            int(fused_frame["sample_index"]) == int(metric_frame["sample_index"])
            and abs(float(fused_frame["time_s"]) - float(metric_frame["time_s"]))
            <= x31.EPSILON,
            "x41_frame_alignment",
        )
        arm = fused_frame["arms"][x40.ARM_X40]
        metric_arm = metric_frame["arms"][x24.ARM_X24]
        tracks = {str(row["track_id"]): row for row in fused_frame["tracks"]}
        confirmed_ids = [str(value) for value in arm.get("confirmed_risk_track_ids", [])]
        confirmed_parents = parent_ids(confirmed_ids, tracks)

        if bool(arm["route_risk"]) and bool(
            metric_arm.get("candidate_risk_track_ids", [])
        ):
            credentialed_parents.update(confirmed_parents)

        selected_ids: list[str] = []
        if (
            not bool(arm["route_risk"])
            and bool(arm.get("cross_representation_adjudication_suppressed", False))
            and previous_x41_risk
        ):
            for track_id in arm.get("cross_representation_suppressed_track_ids", []):
                row = tracks.get(str(track_id))
                if row is None:
                    continue
                parent = str(row["parent_track_id"])
                if parent in previous_x41_parents and parent in credentialed_parents:
                    selected_ids.append(str(track_id))

        if selected_ids:
            continuation_frames += 1
            selected_ids = sorted(set(selected_ids))
            selected_parents = sorted(parent_ids(selected_ids, tracks))
            arm.update(
                {
                    "route_risk": True,
                    "minimum_entry_s": arm.get(
                        "cross_representation_suppressed_minimum_entry_s"
                    ),
                    "candidate_risk_track_ids": selected_ids,
                    "confirmed_risk_track_ids": selected_ids,
                    "candidate_risk_parent_track_ids": selected_parents,
                    "confirmed_risk_parent_track_ids": selected_parents,
                    "metric_credentialed_parent_continuation_used": True,
                    "metric_credentialed_parent_continuation_track_ids": selected_ids,
                }
            )
            confirmed_parents = set(selected_parents)
        else:
            arm["metric_credentialed_parent_continuation_used"] = False
            arm["metric_credentialed_parent_continuation_track_ids"] = []

        previous_x41_risk = bool(arm["route_risk"])
        previous_x41_parents = confirmed_parents if previous_x41_risk else set()

    value["arms"][ARM_X41] = value["arms"].pop(x40.ARM_X40)
    value["diagnostics"]["x41_route_mode_counts"] = value["diagnostics"].pop(
        "x40_route_mode_counts"
    )
    value["diagnostics"]["metric_credentialed_parent_continuation_frames"] = (
        continuation_frames
    )
    value["diagnostics"]["metric_credentialed_parent_count"] = len(
        credentialed_parents
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X41] = frame["arms"].pop(x40.ARM_X40)
    return value


def self_check() -> dict[str, Any]:
    inherited = x40.self_check()
    rows = {
        "track-a": {"parent_track_id": "parent-a"},
        "track-b": {"parent_track_id": "parent-b"},
    }
    x24.require(
        parent_ids(["track-a", "missing"], rows) == {"parent-a"},
        "x41_parent_identity_projection",
    )
    return {
        "status": "X41_METRIC_CREDENTIALED_CONTINUATION_STRUCTURAL_FALSIFIER_MET",
        "x40_structural_status": inherited["status"],
        "causal_previous_parent_required": True,
        "metric_candidate_credential_required": True,
        "class_independent": True,
        "numeric_speed_threshold_added": False,
    }
