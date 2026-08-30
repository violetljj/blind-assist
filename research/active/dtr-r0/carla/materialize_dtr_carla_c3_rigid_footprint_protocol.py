"""Materialize the C3 CARLA cohort for rigid-footprint route risk.

C3 keeps the consumed C2 occlusion pair, but turns the two showcase layouts
into fresh CONTACT/SAFE edge pairs.  The bus and bicycle centres remain outside
X24's uniform 0.65 m point tube; their physical side edges distinguish a
point-track from a metric occupancy footprint.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "dtr_carla_c2_rich_scene_protocol.json"
DEFAULT_OUTPUT = HERE / "dtr_carla_c3_rigid_footprint_protocol.json"


def stationary(forward_m: float, right_m: float, yaw_degrees: float) -> dict[str, Any]:
    return {
        "start_forward_m": forward_m,
        "start_right_m": right_m,
        "yaw_offset_degrees": yaw_degrees,
        "segments": [
            {
                "start_s": 0.0,
                "velocity_forward_mps": 0.0,
                "velocity_right_mps": 0.0,
            }
        ],
    }


def issued_plan(episode_id: str) -> dict[str, Any]:
    return {
        "plan_id": f"plan_{episode_id}",
        "session_id": f"session_{episode_id}",
        "issued_at_s": 0.0,
        "expires_at_s": 7.0,
        "time_parameterized_waypoints": [
            {"time_s": 0.0, "forward_m": -6.0, "right_m": 0.0},
            {"time_s": 7.0, "forward_m": 8.0, "right_m": 0.0},
        ],
    }


def scenario(
    episode_id: str,
    layout_id: str,
    role: str,
    outcome: str,
    responsible: list[str],
    asset_key: str,
    trajectory_key: str,
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "layout_id": layout_id,
        "scenario_role": role,
        "twin_role": "none",
        "navigation_session_id": f"session_{episode_id}",
        "expected_outcome": outcome,
        "expected_responsible_assets": responsible,
        "wearer_trajectory": "wearer_footprint_probe",
        "asset_trajectories": {asset_key: trajectory_key},
        "issued_plan": issued_plan(episode_id),
    }


def convert_asset_to_scenario_trajectory(
    protocol: dict[str, Any], layout_id: str, asset_key: str
) -> None:
    matches = [
        value
        for value in protocol["layouts"][layout_id]["assets"]
        if value["asset_key"] == asset_key
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {layout_id}/{asset_key} asset")
    asset = matches[0]
    asset.pop("fixed_pose", None)
    asset.pop("trajectory", None)
    asset["trajectory_key"] = asset_key
    asset["role"] = "rigid_footprint_target"


def materialize(base: dict[str, Any]) -> dict[str, Any]:
    protocol = deepcopy(base)
    protocol["schema_version"] = 3
    protocol["cohort_id"] = "DTR_CARLA_C3_RIGID_FOOTPRINT_DEVELOPMENT_V1"
    protocol["evidence_class"] = "synthetic_rich_multilayout_rigid_footprint_development_source"
    protocol["objective"] = (
        "Distinguish a metric rigid occupancy footprint from a single target point on "
        "the retained physical-occlusion pair plus large-bus and side-on-bicycle edge "
        "CONTACT/SAFE pairs, using the same dense truth-blind RGB-depth interface."
    )
    protocol["environment"]["sample_seconds"] = 0.10
    protocol["environment"]["fixed_delta_seconds"] = 0.10
    protocol["capture"]["render_quality_level"] = "Low"
    protocol["admission"]["expected_episode_count"] = 6

    protocol["layouts"]["layout_01"]["duration_seconds"] = 4.0
    protocol["layouts"]["layout_01"]["showcase_time_s"] = 1.3
    protocol["layouts"]["layout_02"]["duration_seconds"] = 7.0
    protocol["layouts"]["layout_02"]["showcase_time_s"] = 3.0
    # Stop after the bicycle edge decision and before the pre-existing 8.4 m
    # fountain footprint reaches the wearer route at 5.7 s.
    protocol["layouts"]["layout_03"]["duration_seconds"] = 5.5
    protocol["layouts"]["layout_03"]["showcase_time_s"] = 4.5

    convert_asset_to_scenario_trajectory(protocol, "layout_02", "city_bus")
    convert_asset_to_scenario_trajectory(protocol, "layout_03", "plaza_bike")

    # The bus occupies forward 2.4..12.7 m and reaches right 5.0 m in the SAFE
    # arm. Keep the station crowd dense but outside that rigid volume so CARLA
    # cannot resolve an overlap differently across fresh-server sensor shards.
    station_outer_crowd = {"commuter_01", "commuter_02", "commuter_03", "commuter_06"}
    for asset in protocol["layouts"]["layout_02"]["assets"]:
        if asset["asset_key"] in station_outer_crowd:
            asset["fixed_pose"]["right_m"] = 8.5

    library = protocol["trajectory_library"]
    library["wearer_footprint_probe"] = {
        "start_forward_m": -6.0,
        "start_right_m": 0.0,
        "segments": [
            {
                "start_s": 0.0,
                "velocity_forward_mps": 2.0,
                "velocity_right_mps": 0.0,
            }
        ],
    }
    # Mitsubishi Fuso Rosa half-width in this CARLA build is 1.972 m.
    # CONTACT: physical edge at +0.228 m. SAFE: edge at +1.028 m.
    library["bus_edge_contact"] = stationary(8.0, 2.20, 0.0)
    library["bus_edge_safe"] = stationary(8.0, 3.00, 0.0)
    # Crossbike half-length is 0.755 m and becomes lateral at 90 degrees.
    # CONTACT: physical edge at +0.245 m. SAFE: edge at +0.845 m.
    library["bike_edge_contact"] = stationary(5.0, 1.00, 90.0)
    library["bike_edge_safe"] = stationary(5.0, 1.60, 90.0)

    original_pair = [deepcopy(value) for value in protocol["scenarios"][:2]]
    protocol["scenarios"] = original_pair + [
        scenario(
            "ep_03",
            "layout_02",
            "large_bus_side_edge_contact",
            "CONTACT",
            ["city_bus"],
            "city_bus",
            "bus_edge_contact",
        ),
        scenario(
            "ep_04",
            "layout_02",
            "large_bus_centerline_near_miss_safe",
            "SAFE",
            [],
            "city_bus",
            "bus_edge_safe",
        ),
        scenario(
            "ep_05",
            "layout_03",
            "side_on_bicycle_edge_contact",
            "CONTACT",
            ["plaza_bike"],
            "plaza_bike",
            "bike_edge_contact",
        ),
        scenario(
            "ep_06",
            "layout_03",
            "side_on_bicycle_centerline_near_miss_safe",
            "SAFE",
            [],
            "plaza_bike",
            "bike_edge_safe",
        ),
    ]
    return protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    value = materialize(base)
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "cohort_id": value["cohort_id"],
                "episodes": len(value["scenarios"]),
                "frames_per_sensor": sum(
                    int(round(value["layouts"][row["layout_id"]]["duration_seconds"] / 0.1)) + 1
                    for row in value["scenarios"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
