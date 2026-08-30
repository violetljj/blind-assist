"""Materialize the fresh C7 source-disjoint confirmation cohort for X30.

C7 retains ep_01/ep_02 only as the original occlusion regression stratum.  Its
three dynamic CONTACT/SAFE pairs use target blueprints not used as C5/C6
dynamic targets, a new seed, four new weather presets, and the unchanged full
three-second evaluator truth tail.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import dtr_carla_c2_rich_scene as c2


HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "dtr_carla_c6_source_disjoint_protocol.json"
DEFAULT_OUTPUT = HERE / "dtr_carla_c7_x30_source_disjoint_protocol.json"
SCHEMA_VERSION = 7
COHORT_ID = "DTR_CARLA_C7_X30_SOURCE_DISJOINT_CONFIRMATION_V1"
PARENT_COHORT_ID = "DTR_CARLA_C6_X29_SOURCE_DISJOINT_CONFIRMATION_V1"
CAPTURE_SEED = 104729
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
    "layout_01": "CloudyNoon",
    "layout_02": "ClearSunset",
    "layout_03": "WetCloudyNoon",
    "layout_04": "SoftRainNoon",
}
NEW_TARGET_BLUEPRINTS = {
    "vehicle": "vehicle.audi.tt",
    "walker": "walker.pedestrian.0025",
    "motorcycle": "vehicle.kawasaki.ninja",
}
PAIR_SPECS = (
    {
        "kind": "vehicle",
        "contact": "ep_03",
        "safe": "ep_04",
        "layout_id": "layout_02",
        "old_asset_key": "c6_vehicle_target",
        "asset_key": "c7_vehicle_target",
        "old_template": "v_c6_lincoln",
        "template": "v_c7_audi_tt",
        "track_id": "b_91",
        "old_contact_trajectory": "c6_vehicle_contact",
        "old_safe_trajectory": "c6_vehicle_safe",
        "contact_trajectory": "c7_vehicle_contact",
        "safe_trajectory": "c7_vehicle_safe",
    },
    {
        "kind": "walker",
        "contact": "ep_05",
        "safe": "ep_06",
        "layout_id": "layout_03",
        "old_asset_key": "c6_walker_target",
        "asset_key": "c7_walker_target",
        "old_template": "w_c6_0031",
        "template": "w_c7_0025",
        "track_id": "c_91",
        "old_contact_trajectory": "c6_walker_contact",
        "old_safe_trajectory": "c6_walker_safe",
        "contact_trajectory": "c7_walker_contact",
        "safe_trajectory": "c7_walker_safe",
    },
    {
        "kind": "motorcycle",
        "contact": "ep_07",
        "safe": "ep_08",
        "layout_id": "layout_04",
        "old_asset_key": "c6_motorcycle_target",
        "asset_key": "c7_motorcycle_target",
        "old_template": "v_c6_yamaha",
        "template": "v_c7_kawasaki",
        "track_id": "d_91",
        "old_contact_trajectory": "c6_motorcycle_contact",
        "old_safe_trajectory": "c6_motorcycle_safe",
        "contact_trajectory": "c7_motorcycle_contact",
        "safe_trajectory": "c7_motorcycle_safe",
    },
)


def require(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError(label)


def canonical_sha256(value: Any) -> str:
    return c2.sha256_bytes(c2.canonical_json_bytes(value))


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


def _replace_target(protocol: dict[str, Any], spec: Mapping[str, Any]) -> None:
    old_template = str(spec["old_template"])
    template = str(spec["template"])
    require(old_template in protocol["asset_templates"], f"c7_old_template:{old_template}")
    value = deepcopy(protocol["asset_templates"].pop(old_template))
    value["blueprint_candidates"] = [NEW_TARGET_BLUEPRINTS[str(spec["kind"])]]
    require(template not in protocol["asset_templates"], f"c7_new_template_exists:{template}")
    protocol["asset_templates"][template] = value

    layout = protocol["layouts"][str(spec["layout_id"])]
    matches = [
        asset
        for asset in layout["assets"]
        if asset["asset_key"] == spec["old_asset_key"]
    ]
    require(len(matches) == 1, f"c7_old_asset:{spec['old_asset_key']}")
    asset = matches[0]
    asset["asset_key"] = spec["asset_key"]
    asset["track_id"] = spec["track_id"]
    asset["role"] = f"c7_source_disjoint_dynamic_{spec['kind']}_target"
    asset["template"] = template
    asset["trajectory_key"] = spec["asset_key"]

    for old_key, new_key in (
        (spec["old_contact_trajectory"], spec["contact_trajectory"]),
        (spec["old_safe_trajectory"], spec["safe_trajectory"]),
    ):
        require(old_key in protocol["trajectory_library"], f"c7_old_trajectory:{old_key}")
        require(new_key not in protocol["trajectory_library"], f"c7_new_trajectory:{new_key}")
        protocol["trajectory_library"][new_key] = protocol["trajectory_library"].pop(
            old_key
        )

    scenarios = {value["episode_id"]: value for value in protocol["scenarios"]}
    for episode_id, outcome, trajectory in (
        (spec["contact"], "CONTACT", spec["contact_trajectory"]),
        (spec["safe"], "SAFE", spec["safe_trajectory"]),
    ):
        scenario = scenarios[str(episode_id)]
        pair_name = str(spec["kind"])
        scenario["scenario_role"] = (
            f"c7_source_disjoint_{pair_name}_{outcome.casefold()}"
        )
        scenario["navigation_session_id"] = f"session_c7_{pair_name}"
        scenario["expected_responsible_assets"] = (
            [spec["asset_key"]] if outcome == "CONTACT" else []
        )
        scenario["asset_trajectories"] = {spec["asset_key"]: trajectory}
        scenario["issued_plan"]["plan_id"] = f"plan_c7_{pair_name}"
        scenario["issued_plan"]["session_id"] = f"session_c7_{pair_name}"


def materialize(base: dict[str, Any]) -> dict[str, Any]:
    require(base.get("cohort_id") == PARENT_COHORT_ID, "c7_unexpected_parent")
    protocol = deepcopy(base)
    protocol["schema_version"] = SCHEMA_VERSION
    protocol["cohort_id"] = COHORT_ID
    protocol["evidence_class"] = (
        "synthetic_fresh_source_disjoint_x30_confirmation_source"
    )
    protocol["objective"] = (
        "Confirm frozen X30 set-valued adaptive surface lineage and contact-"
        "interval risk against X24 on fresh source-disjoint vehicle, walker, "
        "and motorcycle CONTACT/SAFE pairs, while retaining ep_01/ep_02 only "
        "as an occlusion regression stratum and preserving a full truth tail."
    )
    protocol["capture"]["seed"] = CAPTURE_SEED
    for layout_id, weather in WEATHER_BY_LAYOUT.items():
        protocol["layouts"][layout_id]["weather"] = weather
    for spec in PAIR_SPECS:
        _replace_target(protocol, spec)

    dynamic_pairs = [pair_contract(spec) for spec in PAIR_SPECS]
    evaluation = protocol["evaluation_contract"]
    evaluation["dynamic_pairs"] = dynamic_pairs
    evaluation["contact_episodes"] = list(CONTACT_EPISODES)
    evaluation["safe_episodes"] = list(SAFE_EPISODES)
    evaluation["fresh_dynamic_contact_episodes"] = list(DYNAMIC_CONTACT_EPISODES)
    evaluation["fresh_dynamic_safe_episodes"] = list(DYNAMIC_SAFE_EPISODES)
    evaluation["score_window_end_seconds"] = SCORE_WINDOW_END_SECONDS
    evaluation["safe_segment_start_seconds"] = SAFE_SEGMENT_START_SECONDS
    evaluation["truth_tail_seconds"] = TRUTH_TAIL_SECONDS

    excluded = set(base["source_disjoint_contract"]["parent_dynamic_target_blueprints_excluded"])
    excluded.update(base["source_disjoint_contract"]["new_dynamic_target_blueprints"].values())
    protocol["source_disjoint_contract"] = {
        "parent_cohort_id": PARENT_COHORT_ID,
        "parent_protocol_canonical_sha256": canonical_sha256(base),
        "retained_regression_episodes": ["ep_01", "ep_02"],
        "confirmation_episodes": list(EPISODES[2:]),
        "new_capture_seed": CAPTURE_SEED,
        "weather_by_layout": WEATHER_BY_LAYOUT,
        "new_dynamic_target_blueprints": NEW_TARGET_BLUEPRINTS,
        "parent_dynamic_target_blueprints_excluded": sorted(excluded),
        "dynamic_pairs": dynamic_pairs,
    }
    claims = [
        value
        for value in protocol["claim_boundary"]
        if not str(value).startswith("C6 freezes X29")
        and "source-disjoint from the C5 dynamic targets" not in str(value)
    ]
    claims.extend(
        [
            "C7 freezes X30 and its scorer before evaluator outcomes are opened; ep_03 through ep_08 use target blueprints not used as C5/C6 dynamic targets.",
            "The original ep_01/ep_02 occlusion pair remains regression-only and does not count toward X30 source-disjoint dynamic confirmation.",
        ]
    )
    protocol["claim_boundary"] = claims
    return protocol


def validate_c7(protocol: dict[str, Any], base: dict[str, Any]) -> None:
    c2.validate_protocol(protocol)
    require(protocol.get("schema_version") == SCHEMA_VERSION, "c7_schema")
    require(protocol.get("cohort_id") == COHORT_ID, "c7_cohort")
    require(int(protocol["capture"]["seed"]) == CAPTURE_SEED, "c7_seed")
    require(int(base["capture"]["seed"]) != CAPTURE_SEED, "c7_seed_not_fresh")
    require(tuple(protocol["layouts"]) == tuple(WEATHER_BY_LAYOUT), "c7_layouts")
    require(
        {key: value["weather"] for key, value in protocol["layouts"].items()}
        == WEATHER_BY_LAYOUT,
        "c7_weather",
    )
    require(
        not (
            set(WEATHER_BY_LAYOUT.values())
            & {value["weather"] for value in base["layouts"].values()}
        ),
        "c7_weather_not_disjoint",
    )
    scenarios = {value["episode_id"]: value for value in protocol["scenarios"]}
    base_scenarios = {value["episode_id"]: value for value in base["scenarios"]}
    require(tuple(scenarios) == EPISODES, "c7_episode_order")
    require(scenarios["ep_01"] == base_scenarios["ep_01"], "c7_occlusion_contact_drift")
    require(scenarios["ep_02"] == base_scenarios["ep_02"], "c7_occlusion_safe_drift")
    require(
        protocol["layouts"]["layout_01"]["assets"]
        == base["layouts"]["layout_01"]["assets"],
        "c7_occlusion_assets_drift",
    )
    require(protocol["occlusion_contracts"] == base["occlusion_contracts"], "c7_occlusion_contract_drift")
    require(protocol["twin_contracts"] == base["twin_contracts"], "c7_occlusion_twin_drift")

    evaluation = protocol["evaluation_contract"]
    require(float(evaluation["truth_tail_seconds"]) == TRUTH_TAIL_SECONDS, "c7_tail")
    require(evaluation["score_window_end_seconds"] == SCORE_WINDOW_END_SECONDS, "c7_windows")
    require(tuple(evaluation["contact_episodes"]) == CONTACT_EPISODES, "c7_contacts")
    require(tuple(evaluation["safe_episodes"]) == SAFE_EPISODES, "c7_safe")
    require(tuple(evaluation["fresh_dynamic_contact_episodes"]) == DYNAMIC_CONTACT_EPISODES, "c7_dynamic_contacts")
    require(tuple(evaluation["fresh_dynamic_safe_episodes"]) == DYNAMIC_SAFE_EPISODES, "c7_dynamic_safe")
    require(evaluation["safe_segment_start_seconds"] == SAFE_SEGMENT_START_SECONDS, "c7_safe_boundaries")

    excluded = set(protocol["source_disjoint_contract"]["parent_dynamic_target_blueprints_excluded"])
    observed: set[str] = set()
    for spec in PAIR_SPECS:
        contact = scenarios[str(spec["contact"])]
        safe = scenarios[str(spec["safe"])]
        asset_key = str(spec["asset_key"])
        require(contact["layout_id"] == safe["layout_id"] == spec["layout_id"], f"c7_pair_layout:{spec['kind']}")
        require(contact["expected_outcome"] == "CONTACT", f"c7_contact:{spec['kind']}")
        require(safe["expected_outcome"] == "SAFE", f"c7_safe:{spec['kind']}")
        require(contact["expected_responsible_assets"] == [asset_key], f"c7_responsible:{spec['kind']}")
        require(safe["expected_responsible_assets"] == [], f"c7_safe_responsible:{spec['kind']}")
        require(set(contact["asset_trajectories"]) == {asset_key}, f"c7_contact_override:{spec['kind']}")
        require(set(safe["asset_trajectories"]) == {asset_key}, f"c7_safe_override:{spec['kind']}")
        assets = {
            value["asset_key"]: value
            for value in c2.materialize_layout_assets(protocol, str(spec["layout_id"]))
        }
        target = assets[asset_key]
        blueprint = str(target["blueprint_candidates"][0])
        observed.add(blueprint)
        require(blueprint == NEW_TARGET_BLUEPRINTS[str(spec["kind"])], f"c7_blueprint:{spec['kind']}")
        require(blueprint not in excluded, f"c7_parent_blueprint:{spec['kind']}")
        require(bool(target.get("scripted_pose_authority")), f"c7_pose_authority:{spec['kind']}")
        first = deepcopy(protocol["trajectory_library"][contact["asset_trajectories"][asset_key]])
        second = deepcopy(protocol["trajectory_library"][safe["asset_trajectories"][asset_key]])
        contact_right = float(first.pop("start_right_m"))
        safe_right = float(second.pop("start_right_m"))
        require(first == second and contact_right < safe_right, f"c7_pair_geometry:{spec['kind']}")
        duration = float(protocol["layouts"][str(spec["layout_id"])]["duration_seconds"])
        score_end = float(SCORE_WINDOW_END_SECONDS[str(spec["contact"])])
        require(duration + 1e-9 >= score_end + TRUTH_TAIL_SECONDS, f"c7_pair_tail:{spec['kind']}")
        for scenario in (contact, safe):
            plan = scenario["issued_plan"]
            require(float(plan["expires_at_s"]) == duration, f"c7_plan_expiry:{scenario['episode_id']}")
            require(float(plan["time_parameterized_waypoints"][-1]["time_s"]) == duration, f"c7_plan_tail:{scenario['episode_id']}")
    require(observed == set(NEW_TARGET_BLUEPRINTS.values()), "c7_target_blueprints")

    source = protocol["source_disjoint_contract"]
    require(source["parent_cohort_id"] == PARENT_COHORT_ID, "c7_parent")
    require(source["parent_protocol_canonical_sha256"] == canonical_sha256(base), "c7_parent_hash")
    require(tuple(source["confirmation_episodes"]) == EPISODES[2:], "c7_confirmation_episodes")
    require(source["new_dynamic_target_blueprints"] == NEW_TARGET_BLUEPRINTS, "c7_disjoint_blueprints")


def serialized_protocol(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    first = materialize(base)
    second = materialize(base)
    first_bytes = serialized_protocol(first)
    require(first_bytes == serialized_protocol(second), "c7_nondeterministic")
    validate_c7(first, base)
    if args.validate_only:
        if args.output.exists():
            require(args.output.read_bytes() == first_bytes, "c7_existing_protocol_drift")
    else:
        if args.output.exists() and args.output.read_bytes() != first_bytes:
            raise FileExistsError(f"existing C7 protocol differs: {args.output}")
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
                "status": "C7_PROTOCOL_VALID",
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
