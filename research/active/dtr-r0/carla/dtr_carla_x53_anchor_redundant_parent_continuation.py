"""X53 anchor-redundant metric parent continuation.

X53 preserves X52 except for the inherited metric-credentialed parent
continuation branch.  That branch may carry route risk only when at least one
confirmed row has more direct transport-anchor pairs than lineage pairs.  The
relational rule requires observation redundancy beyond inherited ancestry and
introduces no numeric threshold.

C22 is consumed posthoc synthetic Development and cannot confirm X53.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x52_cross_parent_provisional_reidentification as x52  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X53_ANCHOR_REDUNDANT_PARENT_CONTINUATION"
ARM_X53 = "X53_ISSUED_PLAN_ANCHOR_REDUNDANT_PARENT_CONTINUATION"


def fixed_constants() -> dict[str, Any]:
    return {
        **x52.fixed_constants(),
        "representation": "CROSS_PARENT_REIDENTIFICATION_WITH_ANCHOR_REDUNDANCY",
        "metric_parent_continuation_rule": (
            "CONFIRMED_DIRECT_TRANSPORT_ANCHOR_PAIRS_STRICTLY_EXCEED_"
            "TRANSPORT_LINEAGE_PAIRS"
        ),
        "anchor_redundancy_numeric_threshold": None,
        "new_numeric_threshold_added": False,
    }


def anchor_redundant(row: Mapping[str, Any]) -> bool:
    return int(row.get("transport_anchor_pairs", 0)) > int(
        row.get("transport_lineage_pairs", 0)
    )


def suppress_parent_continuation(arm: dict[str, Any]) -> None:
    suppressed = sorted(str(value) for value in arm.get("confirmed_risk_track_ids", []))
    arm.update(
        {
            "route_risk": False,
            "minimum_entry_s": None,
            "candidate_risk_track_ids": [],
            "confirmed_risk_track_ids": [],
            "candidate_risk_parent_track_ids": [],
            "confirmed_risk_parent_track_ids": [],
            "x53_anchor_redundancy_suppressed": True,
            "x53_anchor_redundancy_suppressed_track_ids": suppressed,
        }
    )


def apply_anchor_redundancy_episode(
    episode: Any,
    value: dict[str, Any],
    calibration: Any,
) -> dict[str, Any]:
    del episode, calibration
    value = copy.deepcopy(value)
    suppressions = 0
    for frame in value["frames"]:
        arm = frame["arms"][x52.ARM_X52]
        tracks = {str(row["track_id"]): row for row in frame["tracks"]}
        confirmed = [
            tracks[str(track_id)]
            for track_id in arm.get("confirmed_risk_track_ids", [])
            if str(track_id) in tracks
        ]
        if (
            bool(arm.get("route_risk"))
            and bool(arm.get("metric_credentialed_parent_continuation_used", False))
            and not any(anchor_redundant(row) for row in confirmed)
        ):
            suppress_parent_continuation(arm)
            suppressions += 1
        else:
            arm["x53_anchor_redundancy_suppressed"] = False
            arm["x53_anchor_redundancy_suppressed_track_ids"] = []

    value["arms"][ARM_X53] = value["arms"].pop(x52.ARM_X52)
    value["diagnostics"]["x53_route_mode_counts"] = value["diagnostics"].pop(
        "x52_route_mode_counts"
    )
    value["diagnostics"]["x53_anchor_redundancy_suppressions"] = suppressions
    for frame in value["frames"]:
        frame["arms"][ARM_X53] = frame["arms"].pop(x52.ARM_X52)
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    return apply_anchor_redundancy_episode(
        episode,
        x52.predict_episode(episode, candidate_values, calibration),
        calibration,
    )


def self_check() -> dict[str, Any]:
    inherited = x52.self_check()
    x24.require(
        anchor_redundant({"transport_anchor_pairs": 4, "transport_lineage_pairs": 3}),
        "x53_anchor_redundancy_positive",
    )
    x24.require(
        not anchor_redundant(
            {"transport_anchor_pairs": 3, "transport_lineage_pairs": 3}
        ),
        "x53_lineage_only_rejected",
    )
    return {
        "status": "X53_ANCHOR_REDUNDANT_PARENT_CONTINUATION_FALSIFIER_MET",
        "x52_structural_status": inherited["status"],
        "direct_anchor_redundancy_required": True,
        "relational_rule": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
