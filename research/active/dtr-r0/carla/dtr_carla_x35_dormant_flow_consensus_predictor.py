"""X35 modal-velocity consensus for dormant occupancy flow.

Measured and ordinary HOLD states retain X31-X34's complete ambiguity set.
Only the extra dormant-flow state is narrowed to the velocity supported by the
largest number of live authorized transport branches.  Every geometry branch
inside that modal velocity group is preserved.  This removes a minority motion
hypothesis only while extrapolating without a current observation.

C16 results are iterative same-source synthetic Development only.
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
import dtr_carla_x33_dormant_transport_reactivation_predictor as x33  # noqa: E402
import dtr_carla_x34_bounded_dormant_occupancy_flow_predictor as x34  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X35_DORMANT_FLOW_VELOCITY_CONSENSUS"
ARM_X35 = "X35_ISSUED_PLAN_DORMANT_FLOW_VELOCITY_CONSENSUS"
EPSILON = x31.EPSILON
BASE_DORMANT_TRACKER = x33.DormantTransportReactivationTracker


def fixed_constants() -> dict[str, Any]:
    return {
        **x34.fixed_constants(),
        "representation": "DORMANT_FLOW_MODAL_VELOCITY_CONSENSUS",
        "dormant_velocity_rule": (
            "LARGEST_LIVE_AUTHORIZED_BRANCH_GROUP_BY_ROUNDED_VELOCITY"
        ),
        "dormant_geometry_rule": "ALL_BRANCH_GEOMETRIES_WITHIN_MODAL_VELOCITY",
        "measured_ambiguity_reduction": False,
        "ordinary_hold_ambiguity_reduction": False,
        "numeric_score_threshold_added": False,
    }


def velocity_key(velocity: Any) -> tuple[float, float]:
    return tuple(float(round(value, 9)) for value in velocity)


class DormantFlowConsensusTracker(x34.BoundedDormantOccupancyFlowTracker):
    def emitted(self, now_s: float, measured_ids: set[str]) -> list[dict[str, Any]]:
        output = BASE_DORMANT_TRACKER.emitted(self, now_s, measured_ids)
        for parent_track_id, track in sorted(self.tracks.items()):
            age_s = now_s - track.last_seen_s
            if age_s <= x24.HOLD_WINDOW_S + EPSILON:
                continue
            groups: dict[tuple[float, float], list[tuple[str, Any, Any]]] = {}
            for key, branch in track.transport_branches.items():
                authority, velocity = x31.resolve_branch_authority(branch, now_s)
                if authority == x27.RIGID_DYNAMIC:
                    groups.setdefault(velocity_key(velocity), []).append(
                        (key, branch, velocity)
                    )
            if not groups:
                continue
            modal_key, selected = max(
                groups.items(),
                key=lambda item: (
                    len(item[1]),
                    max(
                        x31._branch_priority(value[1], now_s)
                        for value in item[1]
                    ),
                    tuple(-value for value in item[0]),
                ),
            )
            for key, branch, velocity in selected:
                row = self._row(
                    track=track,
                    track_id=self._track_variant_id(parent_track_id, key),
                    parent_track_id=parent_track_id,
                    authority=x27.RIGID_DYNAMIC,
                    velocity=velocity,
                    lineage=branch.world_lineage,
                    last_shift_cells=branch.last_shift_cells,
                    branch=branch,
                    now_s=now_s,
                    measured_ids=measured_ids,
                    authorized_branch_count=len(selected),
                )
                row["disposition"] = "DORMANT_FLOW_CONSENSUS"
                row["dormant_flow_age_s"] = max(0.0, age_s)
                row["dormant_modal_velocity"] = list(modal_key)
                row["dormant_modal_velocity_branch_support"] = len(selected)
                row["dormant_authorized_velocity_groups"] = len(groups)
                output.append(row)
        return output


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    original_tracker = x34.BoundedDormantOccupancyFlowTracker
    x34.BoundedDormantOccupancyFlowTracker = DormantFlowConsensusTracker
    try:
        value = x34.predict_episode(episode, candidate_values, calibration)
    finally:
        x34.BoundedDormantOccupancyFlowTracker = original_tracker
    value["arms"][ARM_X35] = value["arms"].pop(x34.ARM_X34)
    value["diagnostics"]["x35_route_mode_counts"] = value["diagnostics"].pop(
        "x34_route_mode_counts"
    )
    value["diagnostics"]["dormant_consensus_flow_frames"] = sum(
        any(
            track.get("disposition") == "DORMANT_FLOW_CONSENSUS"
            for track in frame["tracks"]
        )
        for frame in value["frames"]
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X35] = frame["arms"].pop(x34.ARM_X34)
    return value


def self_check() -> dict[str, Any]:
    inherited = x34.self_check()
    x24.require(velocity_key((-1.625, 0.0)) == (-1.625, 0.0), "x35_key")
    return {
        "status": "X35_DORMANT_FLOW_CONSENSUS_STRUCTURAL_FALSIFIER_MET",
        "x34_structural_status": inherited["status"],
        "consensus_only_during_dormant_flow": True,
        "modal_group_geometry_preserved": True,
        "score_threshold_added": False,
    }
