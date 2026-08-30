"""Materialize the frozen C6 source-disjoint confirmation cohort for X29.

C6 retains the original physical-occlusion pair only as a regression stratum.
The confirmation stratum is six new episodes: vehicle, pedestrian, and
motorcycle CONTACT/SAFE pairs using target blueprints absent from C5, a new
capture seed, changed weather, and a complete three-second evaluator tail.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import dtr_carla_c2_rich_scene as c2


HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "dtr_carla_c5_dynamic_occupancy_protocol.json"
DEFAULT_OUTPUT = HERE / "dtr_carla_c6_source_disjoint_protocol.json"
SCHEMA_VERSION = 6
COHORT_ID = "DTR_CARLA_C6_X29_SOURCE_DISJOINT_CONFIRMATION_V1"
PARENT_COHORT_ID = "DTR_CARLA_C5_DYNAMIC_OCCUPANCY_FRESH_DEVELOPMENT_V1"
CAPTURE_SEED = 96317
TRUTH_TAIL_SECONDS = 3.0
EPISODES = tuple(f"ep_{value:02d}" for value in range(1, 9))
CONTACT_EPISODES = ("ep_01", "ep_03", "ep_05", "ep_07")
SAFE_EPISODES = ("ep_02", "ep_04", "ep_06", "ep_08")
DYNAMIC_CONTACT_EPISODES = ("ep_03", "ep_05", "ep_07")
DYNAMIC_SAFE_EPISODES = ("ep_04", "ep_06", "ep_08")
SCORE_WINDOW_END_SECONDS = {
    "ep_01": 4.0,
    "ep_02": 4.0,
    "ep_03": 7.0,
    "ep_04": 7.0,
    "ep_05": 6.0,
    "ep_06": 6.0,
    "ep_07": 6.0,
    "ep_08": 6.0,
}
SAFE_SEGMENT_START_SECONDS = {
    "ep_02": 2.7,
    "ep_04": 0.0,
    "ep_06": 0.0,
    "ep_08": 0.0,
}
WEATHER_BY_LAYOUT = {
    "layout_01": "HardRainNoon",
    "layout_02": "ClearNoon",
    "layout_03": "CloudySunset",
    "layout_04": "WetNoon",
}
NEW_TARGET_BLUEPRINTS = {
    "vehicle": "vehicle.lincoln.mkz_2020",
    "walker": "walker.pedestrian.0031",
    "motorcycle": "vehicle.yamaha.yzf",
}
PARENT_DYNAMIC_TARGET_BLUEPRINTS = {
    "vehicle.mercedes.sprinter",
    "walker.pedestrian.0024",
}
PAIR_SPECS = (
    {
        "kind": "vehicle",
        "contact": "ep_03",
        "safe": "ep_04",
        "layout_id": "layout_02",
        "asset_key": "c6_vehicle_target",
        "template": "v_c6_lincoln",
        "track_id": "b_90",
        "contact_trajectory": "c6_vehicle_contact",
        "safe_trajectory": "c6_vehicle_safe",
        "duration_s": 10.0,
        "score_end_s": 7.0,
    },
    {
        "kind": "walker",
        "contact": "ep_05",
        "safe": "ep_06",
        "layout_id": "layout_03",
        "asset_key": "c6_walker_target",
        "template": "w_c6_0031",
        "track_id": "c_90",
        "contact_trajectory": "c6_walker_contact",
        "safe_trajectory": "c6_walker_safe",
        "duration_s": 9.0,
        "score_end_s": 6.0,
    },
    {
        "kind": "motorcycle",
        "contact": "ep_07",
        "safe": "ep_08",
        "layout_id": "layout_04",
        "asset_key": "c6_motorcycle_target",
        "template": "v_c6_yamaha",
        "track_id": "d_90",
        "contact_trajectory": "c6_motorcycle_contact",
        "safe_trajectory": "c6_motorcycle_safe",
        "duration_s": 9.0,
        "score_end_s": 6.0,
    },
)


def require(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError(label)


def canonical_sha256(value: Any) -> str:
    return c2.sha256_bytes(c2.canonical_json_bytes(value))


def approach_then_hold(
    forward_m: float,
    right_m: float,
    yaw_degrees: float,
    velocity_forward_mps: float,
    hold_after_s: float,
) -> dict[str, Any]:
    return {
        "start_forward_m": forward_m,
        "start_right_m": right_m,
        "yaw_offset_degrees": yaw_degrees,
        "segments": [
            {
                "start_s": 0.0,
                "velocity_forward_mps": velocity_forward_mps,
                "velocity_right_mps": 0.0,
            },
            {
                "start_s": hold_after_s,
                "velocity_forward_mps": 0.0,
                "velocity_right_mps": 0.0,
            },
        ],
    }


def issued_plan(pair_name: str, score_end_s: float, duration_s: float) -> dict[str, Any]:
    return {
        "plan_id": f"plan_c6_{pair_name}",
        "session_id": f"session_c6_{pair_name}",
        "issued_at_s": 0.0,
        "expires_at_s": duration_s,
        "time_parameterized_waypoints": [
            {"time_s": 0.0, "forward_m": -6.0, "right_m": 0.0},
            {
                "time_s": score_end_s,
                "forward_m": -6.0 + 2.0 * score_end_s,
                "right_m": 0.0,
            },
            {
                "time_s": duration_s,
                "forward_m": -6.0 + 2.0 * duration_s,
                "right_m": 0.0,
            },
        ],
    }


def scenario(
    episode_id: str,
    pair_name: str,
    layout_id: str,
    asset_key: str,
    trajectory_key: str,
    *,
    outcome: str,
    score_end_s: float,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "layout_id": layout_id,
        "scenario_role": (
            f"c6_source_disjoint_{pair_name}_{outcome.casefold()}"
        ),
        "twin_role": outcome.casefold(),
        "navigation_session_id": f"session_c6_{pair_name}",
        "expected_outcome": outcome,
        "expected_responsible_assets": [asset_key] if outcome == "CONTACT" else [],
        "wearer_trajectory": "wearer_footprint_probe",
        "asset_trajectories": {asset_key: trajectory_key},
        "issued_plan": issued_plan(pair_name, score_end_s, duration_s),
    }


def remove_asset(layout: dict[str, Any], asset_key: str) -> None:
    matches = [value for value in layout["assets"] if value["asset_key"] == asset_key]
    require(len(matches) == 1, f"expected_one_asset:{asset_key}")
    layout["assets"] = [
        value for value in layout["assets"] if value["asset_key"] != asset_key
    ]


def append_target(layout: dict[str, Any], spec: Mapping[str, Any]) -> None:
    layout["assets"].append(
        {
            "asset_key": spec["asset_key"],
            "track_id": spec["track_id"],
            "role": f"c6_source_disjoint_dynamic_{spec['kind']}_target",
            "template": spec["template"],
            "trajectory_key": spec["asset_key"],
            "scripted_pose_authority": True,
        }
    )


def pair_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": spec["kind"],
        "contact_episode": spec["contact"],
        "safe_episode": spec["safe"],
        "layout_id": spec["layout_id"],
        "asset_key": spec["asset_key"],
        "target_blueprint": NEW_TARGET_BLUEPRINTS[str(spec["kind"])],
        "pair_difference": "TARGET_LATERAL_OFFSET_ONLY",
    }


def materialize(base: dict[str, Any]) -> dict[str, Any]:
    protocol = deepcopy(base)
    require(base.get("cohort_id") == PARENT_COHORT_ID, "unexpected_parent_cohort")
    protocol["schema_version"] = SCHEMA_VERSION
    protocol["cohort_id"] = COHORT_ID
    protocol["evidence_class"] = (
        "synthetic_fresh_source_disjoint_uncensored_dynamic_confirmation_source"
    )
    protocol["objective"] = (
        "Confirm frozen X29 temporal occupancy lineage against X24 on three "
        "fresh source-disjoint moving CONTACT/SAFE pairs (vehicle, walker, and "
        "motorcycle), while retaining the original physical-occlusion pair as "
        "a regression stratum and preserving a full future-truth tail."
    )
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["admission"]["expected_episode_count"] = len(EPISODES)
    protocol["admission"]["expected_layout_count"] = 4
    protocol["layouts"]["layout_01"]["weather"] = WEATHER_BY_LAYOUT["layout_01"]
    protocol["layouts"]["layout_02"]["weather"] = WEATHER_BY_LAYOUT["layout_02"]
    protocol["layouts"]["layout_03"]["weather"] = WEATHER_BY_LAYOUT["layout_03"]
    protocol["layouts"]["layout_02"]["duration_seconds"] = 10.0
    protocol["layouts"]["layout_02"]["showcase_time_s"] = 3.6
    protocol["layouts"]["layout_03"]["duration_seconds"] = 9.0
    protocol["layouts"]["layout_03"]["showcase_time_s"] = 3.8

    protocol["asset_templates"]["v_c6_lincoln"] = {
        "kind": "vehicle",
        "blueprint_candidates": [NEW_TARGET_BLUEPRINTS["vehicle"]],
        "surface_offset_m": 0.30,
        "collision_relevant": True,
    }
    protocol["asset_templates"]["w_c6_0031"] = {
        "kind": "walker",
        "blueprint_candidates": [NEW_TARGET_BLUEPRINTS["walker"]],
        "surface_offset_m": 0.80,
        "collision_relevant": True,
    }
    protocol["asset_templates"]["v_c6_yamaha"] = {
        "kind": "vehicle",
        "blueprint_candidates": [NEW_TARGET_BLUEPRINTS["motorcycle"]],
        "surface_offset_m": 0.25,
        "collision_relevant": True,
    }

    remove_asset(protocol["layouts"]["layout_02"], "oncoming_sprinter")
    remove_asset(protocol["layouts"]["layout_03"], "oncoming_walker")
    append_target(protocol["layouts"]["layout_02"], PAIR_SPECS[0])

    # The fourth layout reuses the proven Town10 plaza anchor and route basis,
    # then removes same-class motorcycle clutter.  Its asset composition,
    # target, and weather are new without inventing an unprobed map coordinate.
    layout_04 = deepcopy(protocol["layouts"]["layout_03"])
    layout_04["display_name"] = "recomposed_plaza_motorcycle_crossing"
    layout_04["weather"] = WEATHER_BY_LAYOUT["layout_04"]
    layout_04["duration_seconds"] = 9.0
    layout_04["showcase_time_s"] = 3.3
    layout_04["witness"]["right_m"] = 14.0
    retired_layout_04_assets = {"harley", "kawasaki", "plaza_scooter"}
    layout_04["assets"] = [
        value
        for value in layout_04["assets"]
        if value["asset_key"] not in retired_layout_04_assets
    ]
    append_target(layout_04, PAIR_SPECS[2])
    protocol["layouts"]["layout_04"] = layout_04
    append_target(protocol["layouts"]["layout_03"], PAIR_SPECS[1])

    library = protocol["trajectory_library"]
    # Lincoln half-width is 0.918 m; the frozen wearer radius is 0.45 m.
    library["c6_vehicle_contact"] = approach_then_hold(
        9.0, 1.10, 180.0, -2.20, 4.20
    )
    library["c6_vehicle_safe"] = approach_then_hold(
        9.0, 2.15, 180.0, -2.20, 4.20
    )
    # Walker half-width is 0.188 m in this CARLA package.
    library["c6_walker_contact"] = approach_then_hold(
        8.5, 0.20, 180.0, -1.80, 4.50
    )
    library["c6_walker_safe"] = approach_then_hold(
        8.5, 1.10, 180.0, -1.80, 4.50
    )
    # Yamaha YZF half-width is 0.433 m.
    library["c6_motorcycle_contact"] = approach_then_hold(
        9.0, 0.60, 180.0, -2.50, 4.00
    )
    library["c6_motorcycle_safe"] = approach_then_hold(
        9.0, 1.45, 180.0, -2.50, 4.00
    )

    base_scenarios = {value["episode_id"]: value for value in base["scenarios"]}
    scenarios = [deepcopy(base_scenarios["ep_01"]), deepcopy(base_scenarios["ep_02"])]
    for spec in PAIR_SPECS:
        pair_name = str(spec["kind"])
        scenarios.extend(
            [
                scenario(
                    str(spec["contact"]),
                    pair_name,
                    str(spec["layout_id"]),
                    str(spec["asset_key"]),
                    str(spec["contact_trajectory"]),
                    outcome="CONTACT",
                    score_end_s=float(spec["score_end_s"]),
                    duration_s=float(spec["duration_s"]),
                ),
                scenario(
                    str(spec["safe"]),
                    pair_name,
                    str(spec["layout_id"]),
                    str(spec["asset_key"]),
                    str(spec["safe_trajectory"]),
                    outcome="SAFE",
                    score_end_s=float(spec["score_end_s"]),
                    duration_s=float(spec["duration_s"]),
                ),
            ]
        )
    protocol["scenarios"] = scenarios
    protocol["evaluation_contract"] = {
        "truth_tail_seconds": TRUTH_TAIL_SECONDS,
        "future_truth_rule": "SCORE_ONLY_FRAMES_WITH_FULL_CAPTURED_HORIZON",
        "score_window_end_seconds": SCORE_WINDOW_END_SECONDS,
        "contact_episodes": list(CONTACT_EPISODES),
        "safe_episodes": list(SAFE_EPISODES),
        "fresh_dynamic_contact_episodes": list(DYNAMIC_CONTACT_EPISODES),
        "fresh_dynamic_safe_episodes": list(DYNAMIC_SAFE_EPISODES),
        "retained_occlusion_episodes": ["ep_01", "ep_02"],
        "safe_segment_start_seconds": SAFE_SEGMENT_START_SECONDS,
        "dynamic_pairs": [pair_contract(value) for value in PAIR_SPECS],
        "all_physical_obstacles_are_truth_relevant": True,
    }
    protocol["source_disjoint_contract"] = {
        "parent_cohort_id": PARENT_COHORT_ID,
        "parent_protocol_canonical_sha256": canonical_sha256(base),
        "retained_regression_episodes": ["ep_01", "ep_02"],
        "confirmation_episodes": list(EPISODES[2:]),
        "new_capture_seed": CAPTURE_SEED,
        "weather_by_layout": WEATHER_BY_LAYOUT,
        "new_dynamic_target_blueprints": NEW_TARGET_BLUEPRINTS,
        "parent_dynamic_target_blueprints_excluded": sorted(
            PARENT_DYNAMIC_TARGET_BLUEPRINTS
        ),
        "dynamic_pairs": [pair_contract(value) for value in PAIR_SPECS],
    }
    protocol["claim_boundary"] = [
        value
        for value in protocol["claim_boundary"]
        if not str(value).startswith("C5 appends")
    ] + [
        "C6 freezes X29 and its scorer before evaluator outcomes are opened; "
        "ep_03 through ep_08 are source-disjoint from the C5 dynamic targets.",
        "The original ep_01/ep_02 occlusion pair is retained only as regression "
        "evidence and does not count toward source-disjoint dynamic confirmation.",
        "Every scored frame has a three-second captured evaluator tail; tail "
        "RGB-D is predicted but excluded from frame metrics.",
        "This remains scripted synthetic CARLA evidence, not real-world, product, "
        "deployment, or safety authority.",
    ]
    return protocol


def validate_c6(protocol: dict[str, Any], base: dict[str, Any]) -> None:
    c2.validate_protocol(protocol)
    require(protocol.get("schema_version") == SCHEMA_VERSION, "c6_schema")
    require(protocol.get("cohort_id") == COHORT_ID, "c6_cohort")
    require(int(protocol["capture"]["seed"]) == CAPTURE_SEED, "c6_seed")
    require(
        int(base["capture"]["seed"]) != int(protocol["capture"]["seed"]),
        "c6_seed_not_fresh",
    )
    require(tuple(protocol["layouts"]) == tuple(WEATHER_BY_LAYOUT), "c6_layouts")
    require(
        {key: value["weather"] for key, value in protocol["layouts"].items()}
        == WEATHER_BY_LAYOUT,
        "c6_weather",
    )
    parent_weather = {value["weather"] for value in base["layouts"].values()}
    require(
        not (set(WEATHER_BY_LAYOUT.values()) & parent_weather),
        "c6_weather_not_disjoint",
    )
    scenarios = {value["episode_id"]: value for value in protocol["scenarios"]}
    base_scenarios = {value["episode_id"]: value for value in base["scenarios"]}
    require(tuple(scenarios) == EPISODES, "c6_episode_order")
    require(scenarios["ep_01"] == base_scenarios["ep_01"], "c6_occlusion_contact_drift")
    require(scenarios["ep_02"] == base_scenarios["ep_02"], "c6_occlusion_safe_drift")
    require(
        protocol["layouts"]["layout_01"]["assets"]
        == base["layouts"]["layout_01"]["assets"],
        "c6_occlusion_assets_drift",
    )
    require(protocol["occlusion_contracts"] == base["occlusion_contracts"], "c6_occlusion_contract_drift")
    require(protocol["twin_contracts"] == base["twin_contracts"], "c6_occlusion_twin_drift")

    evaluation = protocol["evaluation_contract"]
    require(float(evaluation["truth_tail_seconds"]) == TRUTH_TAIL_SECONDS, "c6_tail")
    require(
        evaluation["score_window_end_seconds"] == SCORE_WINDOW_END_SECONDS,
        "c6_score_windows",
    )
    require(tuple(evaluation["contact_episodes"]) == CONTACT_EPISODES, "c6_contacts")
    require(tuple(evaluation["safe_episodes"]) == SAFE_EPISODES, "c6_safe")
    require(
        tuple(evaluation["fresh_dynamic_contact_episodes"])
        == DYNAMIC_CONTACT_EPISODES,
        "c6_dynamic_contacts",
    )
    require(
        tuple(evaluation["fresh_dynamic_safe_episodes"]) == DYNAMIC_SAFE_EPISODES,
        "c6_dynamic_safe",
    )
    require(
        evaluation["safe_segment_start_seconds"] == SAFE_SEGMENT_START_SECONDS,
        "c6_safe_boundaries",
    )

    observed_blueprints: set[str] = set()
    for spec in PAIR_SPECS:
        contact_id = str(spec["contact"])
        safe_id = str(spec["safe"])
        contact = scenarios[contact_id]
        safe = scenarios[safe_id]
        asset_key = str(spec["asset_key"])
        require(contact["layout_id"] == safe["layout_id"] == spec["layout_id"], f"c6_pair_layout:{spec['kind']}")
        require(contact["expected_outcome"] == "CONTACT", f"c6_pair_contact:{spec['kind']}")
        require(safe["expected_outcome"] == "SAFE", f"c6_pair_safe:{spec['kind']}")
        require(contact["expected_responsible_assets"] == [asset_key], f"c6_pair_responsible:{spec['kind']}")
        require(safe["expected_responsible_assets"] == [], f"c6_pair_safe_responsible:{spec['kind']}")
        require(set(contact["asset_trajectories"]) == {asset_key}, f"c6_pair_contact_override:{spec['kind']}")
        require(set(safe["asset_trajectories"]) == {asset_key}, f"c6_pair_safe_override:{spec['kind']}")
        assets = {
            value["asset_key"]: value
            for value in c2.materialize_layout_assets(protocol, str(spec["layout_id"]))
        }
        target = assets[asset_key]
        blueprint = str(target["blueprint_candidates"][0])
        observed_blueprints.add(blueprint)
        require(blueprint == NEW_TARGET_BLUEPRINTS[str(spec["kind"])], f"c6_pair_blueprint:{spec['kind']}")
        require(blueprint not in PARENT_DYNAMIC_TARGET_BLUEPRINTS, f"c6_pair_parent_blueprint:{spec['kind']}")
        require(bool(target.get("scripted_pose_authority")), f"c6_pair_pose_authority:{spec['kind']}")
        first = deepcopy(protocol["trajectory_library"][contact["asset_trajectories"][asset_key]])
        second = deepcopy(protocol["trajectory_library"][safe["asset_trajectories"][asset_key]])
        contact_right = float(first.pop("start_right_m"))
        safe_right = float(second.pop("start_right_m"))
        require(first == second and contact_right < safe_right, f"c6_pair_geometry:{spec['kind']}")
        duration = float(protocol["layouts"][str(spec["layout_id"])]["duration_seconds"])
        score_end = float(SCORE_WINDOW_END_SECONDS[contact_id])
        require(duration + 1e-9 >= score_end + TRUTH_TAIL_SECONDS, f"c6_pair_tail:{spec['kind']}")
        for value in (contact, safe):
            plan = value["issued_plan"]
            require(float(plan["expires_at_s"]) == duration, f"c6_plan_expiry:{value['episode_id']}")
            require(float(plan["time_parameterized_waypoints"][-1]["time_s"]) == duration, f"c6_plan_tail:{value['episode_id']}")
    require(observed_blueprints == set(NEW_TARGET_BLUEPRINTS.values()), "c6_target_blueprints")

    source_disjoint = protocol["source_disjoint_contract"]
    require(source_disjoint["parent_cohort_id"] == PARENT_COHORT_ID, "c6_parent_identity")
    require(
        source_disjoint["parent_protocol_canonical_sha256"] == canonical_sha256(base),
        "c6_parent_hash",
    )
    require(
        tuple(source_disjoint["confirmation_episodes"]) == EPISODES[2:],
        "c6_confirmation_episodes",
    )
    require(
        source_disjoint["new_dynamic_target_blueprints"] == NEW_TARGET_BLUEPRINTS,
        "c6_disjoint_blueprints",
    )


def serialized_protocol(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Materialize twice and validate without writing a protocol.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    first = materialize(base)
    second = materialize(base)
    first_bytes = serialized_protocol(first)
    require(first_bytes == serialized_protocol(second), "c6_nondeterministic_materialization")
    validate_c6(first, base)
    if args.validate_only:
        if args.output.exists():
            require(args.output.read_bytes() == first_bytes, "c6_existing_protocol_drift")
    else:
        if args.output.exists() and args.output.read_bytes() != first_bytes:
            raise FileExistsError(f"existing C6 protocol differs: {args.output}")
        if not args.output.exists():
            args.output.write_bytes(first_bytes)
    frames_per_sensor = sum(
        int(
            round(
                float(first["layouts"][value["layout_id"]]["duration_seconds"])
                / float(first["environment"]["sample_seconds"])
            )
        )
        + 1
        for value in first["scenarios"]
    )
    print(
        json.dumps(
            {
                "status": "C6_PROTOCOL_VALID",
                "output": str(args.output.resolve()),
                "wrote_output": not args.validate_only,
                "cohort_id": first["cohort_id"],
                "episodes": len(first["scenarios"]),
                "layouts": len(first["layouts"]),
                "frames_per_sensor": frames_per_sensor,
                "protocol_sha256": c2.sha256_bytes(first_bytes),
                "parent_protocol_canonical_sha256": canonical_sha256(base),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
