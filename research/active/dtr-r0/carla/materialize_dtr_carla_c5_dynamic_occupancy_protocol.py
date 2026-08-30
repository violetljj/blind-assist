"""Materialize the fresh C5 cohort for persistent occupancy authority.

C5 changes the information source, not the frozen matcher.  Each scored model
window is followed by a full three-second truth-only tail, so
``future_contact_within_horizon`` is never right-censored.  The fresh arms are
genuinely moving, oncoming Sprinter and pedestrian CONTACT/SAFE pairs.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "dtr_carla_c4_support_consensus_protocol.json"
DEFAULT_OUTPUT = HERE / "dtr_carla_c5_dynamic_occupancy_protocol.json"
TRUTH_TAIL_SECONDS = 3.0
SCORE_WINDOW_END_SECONDS = {
    "ep_01": 4.0,
    "ep_02": 4.0,
    "ep_03": 7.0,
    "ep_04": 7.0,
    "ep_05": 5.5,
    "ep_06": 5.5,
}


def linear(
    forward_m: float,
    right_m: float,
    yaw_degrees: float,
    velocity_forward_mps: float,
    velocity_right_mps: float = 0.0,
) -> dict[str, Any]:
    return {
        "start_forward_m": forward_m,
        "start_right_m": right_m,
        "yaw_offset_degrees": yaw_degrees,
        "segments": [
            {
                "start_s": 0.0,
                "velocity_forward_mps": velocity_forward_mps,
                "velocity_right_mps": velocity_right_mps,
            }
        ],
    }


def approach_then_hold(
    forward_m: float,
    right_m: float,
    yaw_degrees: float,
    velocity_forward_mps: float,
    hold_after_s: float,
) -> dict[str, Any]:
    value = linear(
        forward_m,
        right_m,
        yaw_degrees,
        velocity_forward_mps,
    )
    value["segments"].append(
        {
            "start_s": hold_after_s,
            "velocity_forward_mps": 0.0,
            "velocity_right_mps": 0.0,
        }
    )
    return value


def only_asset(protocol: dict[str, Any], layout_id: str, key: str) -> dict[str, Any]:
    matches = [
        value
        for value in protocol["layouts"][layout_id]["assets"]
        if value["asset_key"] == key
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {layout_id}/{key}")
    return matches[0]


def extend_plan(scenario: dict[str, Any], end_s: float, end_forward_m: float) -> None:
    plan = scenario["issued_plan"]
    plan["expires_at_s"] = end_s
    waypoints = plan["time_parameterized_waypoints"]
    if float(waypoints[-1]["time_s"]) < end_s:
        waypoints.append(
            {
                "time_s": end_s,
                "forward_m": end_forward_m,
                "right_m": 0.0,
            }
        )


def materialize(base: dict[str, Any]) -> dict[str, Any]:
    protocol = deepcopy(base)
    protocol["schema_version"] = 5
    protocol["cohort_id"] = "DTR_CARLA_C5_DYNAMIC_OCCUPANCY_FRESH_DEVELOPMENT_V1"
    protocol["evidence_class"] = (
        "synthetic_fresh_uncensored_dynamic_occupancy_development_source"
    )
    protocol["objective"] = (
        "Evaluate frozen persistent occupancy authority on genuinely moving "
        "oncoming Sprinter and pedestrian CONTACT/SAFE pairs, with a complete "
        "future-truth tail and the original physical-occlusion pair retained."
    )
    protocol["capture"]["seed"] = 84217
    # Freeze the renderer in the source identity.  This CARLA package's DX11
    # off-screen server never became RPC-ready; DX12 is admitted only under an
    # exclusive-memory preflight.
    protocol["capture"]["render_backend"] = "dx12"
    # Scripted contact truth is computed from frozen geometry.  Prevent CARLA's
    # damage path from deleting the sensor-bearing wearer during a near pass.
    protocol["wearer"]["scripted_invincible"] = True
    protocol["layouts"]["layout_01"]["weather"] = "SoftRainSunset"
    protocol["layouts"]["layout_02"]["weather"] = "WetCloudySunset"
    protocol["layouts"]["layout_03"]["weather"] = "MidRainSunset"
    protocol["evaluation_contract"] = {
        "truth_tail_seconds": TRUTH_TAIL_SECONDS,
        "future_truth_rule": "SCORE_ONLY_FRAMES_WITH_FULL_CAPTURED_HORIZON",
        "score_window_end_seconds": SCORE_WINDOW_END_SECONDS,
        "fresh_dynamic_contact_episodes": ["ep_03", "ep_05"],
        "fresh_dynamic_safe_episodes": ["ep_04", "ep_06"],
        "all_physical_obstacles_are_truth_relevant": True,
    }
    protocol["claim_boundary"].append(
        "C5 appends a three-second evaluator tail after every scored window; "
        "tail RGB-D is captured and predicted but excluded from frame metrics."
    )

    # Capture a full realized-future tail after every original model window.
    protocol["layouts"]["layout_01"]["duration_seconds"] = 7.0
    protocol["layouts"]["layout_02"]["duration_seconds"] = 10.0
    protocol["layouts"]["layout_03"]["duration_seconds"] = 8.5

    # The HGV is no longer the target; keep it as off-route visual clutter and
    # add the already-proven Sprinter blueprint as a genuinely moving target.
    hgv = only_asset(protocol, "layout_02", "delivery_hgv")
    hgv.pop("trajectory_key", None)
    hgv.pop("trajectory", None)
    hgv["fixed_pose"] = {
        "forward_m": 23.0,
        "right_m": 8.0,
        "yaw_offset_degrees": 0.0,
    }
    hgv["role"] = "delivery_vehicle"
    protocol["layouts"]["layout_02"]["assets"].append(
        {
            "asset_key": "oncoming_sprinter",
            "track_id": "b_08",
            "role": "fresh_dynamic_vehicle_target",
            "template": "v_sprinter",
            "trajectory_key": "oncoming_sprinter",
            "scripted_pose_authority": True,
        }
    )

    # Retire the stationary motorcycle target.  A new pedestrian identity is
    # used for the fresh motion-authority pair.
    harley = only_asset(protocol, "layout_03", "harley")
    harley.pop("trajectory_key", None)
    harley.pop("trajectory", None)
    harley["fixed_pose"] = {
        "forward_m": 18.0,
        "right_m": -7.5,
        "yaw_offset_degrees": 180.0,
    }
    harley["role"] = "micromobility"
    protocol["asset_templates"]["w23"] = {
        "kind": "walker",
        "blueprint_candidates": ["walker.pedestrian.0024"],
        "surface_offset_m": 0.8,
        "collision_relevant": True,
    }
    protocol["layouts"]["layout_03"]["assets"].append(
        {
            "asset_key": "oncoming_walker",
            "track_id": "c_31",
            "role": "fresh_dynamic_pedestrian_target",
            "template": "w23",
            "trajectory_key": "oncoming_walker",
            "scripted_pose_authority": True,
        }
    )

    # C4's fountain sat at f=10,r=0 and entered the model's three-second route
    # horizon after the recorded SAFE window.  Keep the landmarks, but move
    # them beyond the captured wearer path so the SAFE label is physically true.
    only_asset(protocol, "layout_03", "fountain")["fixed_pose"] = {
        "forward_m": 20.0,
        "right_m": 0.0,
        "yaw_offset_degrees": 0.0,
    }
    only_asset(protocol, "layout_03", "pergola")["fixed_pose"] = {
        "forward_m": 24.0,
        "right_m": 0.0,
        "yaw_offset_degrees": 0.0,
    }

    library = protocol["trajectory_library"]
    # Sprinter bbox half-width is 0.994 m in the admitted CARLA build.
    library["sprinter_oncoming_contact"] = approach_then_hold(
        8.0, 1.20, 180.0, -2.0, 4.0
    )
    # The first C5 geometry probe used r=1.90 and the CARLA walker disappeared
    # during the near pass despite analytical bbox clearance.  r=2.50 keeps a
    # full 1.51 m vehicle-edge separation without changing the motion evidence.
    library["sprinter_oncoming_safe"] = approach_then_hold(
        8.0, 2.50, 180.0, -2.0, 4.0
    )
    # A 2 m/s oncoming walker produces resolvable nonzero lattice translation.
    library["walker_oncoming_contact"] = approach_then_hold(
        8.0, 0.30, 180.0, -2.0, 4.0
    )
    library["walker_oncoming_safe"] = approach_then_hold(
        8.0, 1.15, 180.0, -2.0, 4.0
    )

    scenario_map = {value["episode_id"]: value for value in protocol["scenarios"]}
    for episode_id, trajectory, outcome, responsible, role, asset_key in (
        (
            "ep_03",
            "sprinter_oncoming_contact",
            "CONTACT",
            ["oncoming_sprinter"],
            "fresh_oncoming_sprinter_contact",
            "oncoming_sprinter",
        ),
        (
            "ep_04",
            "sprinter_oncoming_safe",
            "SAFE",
            [],
            "fresh_oncoming_sprinter_near_miss_safe",
            "oncoming_sprinter",
        ),
        (
            "ep_05",
            "walker_oncoming_contact",
            "CONTACT",
            ["oncoming_walker"],
            "fresh_oncoming_walker_contact",
            "oncoming_walker",
        ),
        (
            "ep_06",
            "walker_oncoming_safe",
            "SAFE",
            [],
            "fresh_oncoming_walker_near_miss_safe",
            "oncoming_walker",
        ),
    ):
        value = scenario_map[episode_id]
        value["scenario_role"] = role
        value["expected_outcome"] = outcome
        value["expected_responsible_assets"] = responsible
        value["asset_trajectories"] = {asset_key: trajectory}

    # Match issued-plan authority to the captured tail.  Scoring still ends at
    # the original windows declared above.
    for episode_id in ("ep_01", "ep_02"):
        extend_plan(scenario_map[episode_id], 7.0, 4.1)
    for episode_id in ("ep_03", "ep_04"):
        extend_plan(scenario_map[episode_id], 10.0, 14.0)
    for episode_id in ("ep_05", "ep_06"):
        extend_plan(scenario_map[episode_id], 8.5, 11.0)
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
                "truth_tail_seconds": TRUTH_TAIL_SECONDS,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
