"""X34 bounded dormant occupancy-flow propagation.

X34 keeps X33's observation-centric reactivation and propagates only the
already-authorized transport branches while their parent is dormant inside the
same bounded two-sample association memory.  This is a causal occupancy-flow
state, not a longer generic detector HOLD: unauthorized, ego-carried, static,
or newly created tracks cannot enter it, and a returning observation replaces
the dormant prediction through X33's normal measured update.

C16 remains consumed same-source synthetic Development and cannot confirm X34.
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


EXPERIMENT_ID = "DTR_CARLA_X34_BOUNDED_DORMANT_OCCUPANCY_FLOW"
ARM_X34 = "X34_ISSUED_PLAN_BOUNDED_DORMANT_OCCUPANCY_FLOW"
EPSILON = x31.EPSILON


def fixed_constants() -> dict[str, Any]:
    return {
        **x33.fixed_constants(),
        "representation": "BOUNDED_AUTHORIZED_DORMANT_OCCUPANCY_FLOW",
        "dormant_emission": "AUTHORIZED_TRANSPORT_BRANCHES_ONLY",
        "dormant_geometry": "BRANCH_LINEAGE_WARPED_BY_AUTHORIZED_VELOCITY",
        "dormant_flow_lifetime": "SAME_BOUNDED_TWO_SAMPLE_ASSOCIATION_MEMORY",
        "generic_detector_hold_extension": False,
        "detector_threshold_change": False,
        "association_radius_change": False,
        "route_threshold_change": False,
        "score_threshold_change": False,
    }


class BoundedDormantOccupancyFlowTracker(
    x33.DormantTransportReactivationTracker
):
    def emitted(self, now_s: float, measured_ids: set[str]) -> list[dict[str, Any]]:
        output = super().emitted(now_s, measured_ids)
        for parent_track_id, track in sorted(self.tracks.items()):
            age_s = now_s - track.last_seen_s
            if age_s <= x24.HOLD_WINDOW_S + EPSILON:
                continue
            authorized = [
                (key, branch, x31.resolve_branch_authority(branch, now_s)[1])
                for key, branch in track.transport_branches.items()
                if x31.resolve_branch_authority(branch, now_s)[0]
                == x27.RIGID_DYNAMIC
            ]
            for key, branch, velocity in authorized:
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
                    authorized_branch_count=len(authorized),
                )
                row["disposition"] = "DORMANT_FLOW"
                row["dormant_flow_age_s"] = max(0.0, age_s)
                output.append(row)
        return output


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    original_tracker = x33.DormantTransportReactivationTracker
    x33.DormantTransportReactivationTracker = BoundedDormantOccupancyFlowTracker
    try:
        value = x33.predict_episode(episode, candidate_values, calibration)
    finally:
        x33.DormantTransportReactivationTracker = original_tracker
    value["arms"][ARM_X34] = value["arms"].pop(x33.ARM_X33)
    value["diagnostics"]["x34_route_mode_counts"] = value["diagnostics"].pop(
        "x33_route_mode_counts"
    )
    value["diagnostics"]["dormant_occupancy_flow_frames"] = sum(
        any(track.get("disposition") == "DORMANT_FLOW" for track in frame["tracks"])
        for frame in value["frames"]
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X34] = frame["arms"].pop(x33.ARM_X33)
    return value


def self_check() -> dict[str, Any]:
    inherited = x33.self_check()
    x24.require(
        x33.ASSOCIATION_MEMORY_SAMPLE_PERIODS == 2,
        "x34_unbounded_dormant_flow",
    )
    return {
        "status": "X34_BOUNDED_DORMANT_OCCUPANCY_FLOW_STRUCTURAL_FALSIFIER_MET",
        "x33_structural_status": inherited["status"],
        "authorized_dynamic_only": True,
        "bounded_by_x33_association_memory": True,
        "generic_detector_hold_extension": False,
    }
