"""Materialize the fresh C4 cohort for X26 support-consensus transport."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "dtr_carla_c3_rigid_footprint_protocol.json"
DEFAULT_OUTPUT = HERE / "dtr_carla_c4_support_consensus_protocol.json"


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


def only_asset(protocol: dict[str, Any], layout_id: str, key: str) -> dict[str, Any]:
    matches = [value for value in protocol["layouts"][layout_id]["assets"] if value["asset_key"] == key]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {layout_id}/{key}")
    return matches[0]


def materialize(base: dict[str, Any]) -> dict[str, Any]:
    protocol = deepcopy(base)
    protocol["schema_version"] = 4
    protocol["cohort_id"] = "DTR_CARLA_C4_SUPPORT_CONSENSUS_FRESH_DEVELOPMENT_V1"
    protocol["evidence_class"] = "synthetic_fresh_asset_support_consensus_development_source"
    protocol["objective"] = (
        "Evaluate support-consensus footprint transport on fresh HGV and side-on "
        "motorcycle edge CONTACT/SAFE pairs under changed weather and seed, while "
        "retaining one occlusion continuity pair."
    )
    protocol["capture"]["seed"] = 73129
    protocol["layouts"]["layout_01"]["weather"] = "WetCloudyNoon"
    protocol["layouts"]["layout_02"]["weather"] = "CloudyNoon"
    protocol["layouts"]["layout_03"]["weather"] = "ClearSunset"

    # Remove the consumed C3 bus target and promote the previously parked HGV.
    protocol["layouts"]["layout_02"]["assets"] = [
        value
        for value in protocol["layouts"]["layout_02"]["assets"]
        if value["asset_key"] != "city_bus"
    ]
    hgv = only_asset(protocol, "layout_02", "delivery_hgv")
    hgv.pop("fixed_pose", None)
    hgv.pop("trajectory", None)
    hgv["trajectory_key"] = "delivery_hgv"
    hgv["role"] = "fresh_rigid_footprint_target"

    # Restore the C3 bicycle as a parked distractor and promote a new motorcycle.
    bicycle = only_asset(protocol, "layout_03", "plaza_bike")
    bicycle.pop("trajectory_key", None)
    bicycle["fixed_pose"] = {
        "forward_m": 4.5,
        "right_m": -6.0,
        "yaw_offset_degrees": 180.0,
    }
    bicycle["role"] = "micromobility"
    motorcycle = only_asset(protocol, "layout_03", "harley")
    motorcycle.pop("fixed_pose", None)
    motorcycle.pop("trajectory", None)
    motorcycle["trajectory_key"] = "harley"
    motorcycle["role"] = "fresh_rigid_footprint_target"

    library = protocol["trajectory_library"]
    # HGV half-width 1.446 m: CONTACT edge +0.254 m; SAFE edge +1.054 m.
    library["hgv_edge_contact"] = stationary(8.0, 1.70, 0.0)
    library["hgv_edge_safe"] = stationary(8.0, 2.50, 0.0)
    # Harley half-length 1.18 m, rotated lateral: +0.22 m / +1.02 m edges.
    library["motorcycle_edge_contact"] = stationary(5.0, 1.40, 90.0)
    library["motorcycle_edge_safe"] = stationary(5.0, 2.20, 90.0)

    scenario_map = {value["episode_id"]: value for value in protocol["scenarios"]}
    for episode_id, trajectory, outcome, responsible, role in (
        ("ep_03", "hgv_edge_contact", "CONTACT", ["delivery_hgv"], "fresh_hgv_side_edge_contact"),
        ("ep_04", "hgv_edge_safe", "SAFE", [], "fresh_hgv_centerline_near_miss_safe"),
        ("ep_05", "motorcycle_edge_contact", "CONTACT", ["harley"], "fresh_side_on_motorcycle_edge_contact"),
        ("ep_06", "motorcycle_edge_safe", "SAFE", [], "fresh_side_on_motorcycle_near_miss_safe"),
    ):
        value = scenario_map[episode_id]
        value["scenario_role"] = role
        value["expected_outcome"] = outcome
        value["expected_responsible_assets"] = responsible
        value["asset_trajectories"] = {
            "delivery_hgv" if episode_id in {"ep_03", "ep_04"} else "harley": trajectory
        }
    return protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    value = materialize(json.loads(args.base.read_text(encoding="utf-8")))
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
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
