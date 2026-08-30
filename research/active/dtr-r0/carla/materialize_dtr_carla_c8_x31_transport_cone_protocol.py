"""Materialize the fresh C8 X31 transport-cone continuity cohort.

C8 is a wholly fresh eight-episode source on the installed runner's proven
Town10HD_Opt map.  It reuses the C7 source protocol's numeric anchors, witnesses,
registered C2/C3/C7 asset families, and capture schema, but it does not retain
any prior episode or refer to prior evidence/results.  Four CONTACT/SAFE pairs
freeze a new seed, weather, dense support rearrangement, target, same-blueprint
alias, and physical-occluder trajectories before capture.  CONTACT and SAFE
targets differ causally in lateral position from the first frame while sharing
the same velocities and segment timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import dtr_carla_c2_rich_scene as c2


HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "dtr_carla_c7_x30_source_disjoint_protocol.json"
DEFAULT_ASSET_REGISTRY = HERE / "dtr_carla_c3_asset_registry.json"
DEFAULT_PRIOR_PROTOCOLS = (
    HERE / "dtr_carla_c3_rigid_footprint_protocol.json",
    HERE / "dtr_carla_c4_support_consensus_protocol.json",
    HERE / "dtr_carla_c5_dynamic_occupancy_protocol.json",
    HERE / "dtr_carla_c6_source_disjoint_protocol.json",
    DEFAULT_BASE,
)
DEFAULT_OUTPUT = HERE / "dtr_carla_c8_x31_transport_cone_protocol.json"

SCHEMA_VERSION = 8
COHORT_ID = "DTR_CARLA_C8_X31_TRANSPORT_CONE_SOURCE_DISJOINT_V1"
PARENT_COHORT_ID = "DTR_CARLA_C7_X30_SOURCE_DISJOINT_CONFIRMATION_V1"
MAP_ID = "Carla/Maps/Town10HD_Opt"
CAPTURE_SEED = 130363
DURATION_SECONDS = 9.0
SAMPLE_SECONDS = 0.1
TRUTH_TAIL_SECONDS = 3.0
SCORE_WINDOW_END_SECONDS = 6.0
SAFE_SEGMENT_START_SECONDS = 0.0
EPISODES = tuple(f"ep_{value:02d}" for value in range(1, 9))
CONTACT_EPISODES = ("ep_01", "ep_03", "ep_05", "ep_07")
SAFE_EPISODES = ("ep_02", "ep_04", "ep_06", "ep_08")

WEATHER_BY_LAYOUT = {
    "c8_l01": "HardRainSunset",
    "c8_l02": "WetSunset",
    "c8_l03": "MidRainyNoon",
    "c8_l04": "DustStorm",
}

# These numeric anchors are copied only from the frozen C7 source protocol.
# No captured evidence or outcome is consulted.
ANCHORS = {
    "c8_l01": {
        "center_xy_m": [-46.934195, 36.51329],
        "forward_xy": [0.999999564, 0.000933925],
        "right_xy": [-0.000933925, 0.999999564],
    },
    "c8_l02": {
        "center_xy_m": [-56.87, 140.54],
        "forward_xy": [0.999981, 0.006109],
        "right_xy": [-0.006109, 0.999981],
    },
    "c8_l03": {
        "center_xy_m": [99.38, -6.31],
        "forward_xy": [-0.006807, 0.999977],
        "right_xy": [-0.999977, -0.006807],
    },
    "c8_l04": {
        "center_xy_m": [99.38, -6.31],
        "forward_xy": [-0.006807, 0.999977],
        "right_xy": [-0.999977, -0.006807],
    },
}

PAIR_SPECS = (
    {
        "layout_id": "c8_l01",
        "base_layout_id": "layout_01",
        "contact": "ep_01",
        "safe": "ep_02",
        "kind": "walker",
        "family": "adult",
        "target_blueprint": "walker.pedestrian.0044",
        "template": "w_c8_0044",
        "template_base": "w_c7_0025",
        "old_assets": ("target_primary", "moving_occluder"),
        "target_key": "c8_l01_target",
        "alias_key": "c8_l01_alias",
        "occluder_key": "c8_l01_occluder",
        "occluder_template": "v_sprinter",
        "target_track": "e_91",
        "alias_track": "e_92",
        "occluder_track": "e_93",
        "contact_trajectory": "c8_l01_target_contact",
        "safe_trajectory": "c8_l01_target_safe",
        "alias_trajectory": "c8_l01_alias_safe",
        "occluder_trajectory": "c8_l01_occluder_cross",
        "planned_occlusion_window_s": [1.3, 2.4],
        "expected_contact_time_s": 4.15,
    },
    {
        "layout_id": "c8_l02",
        "base_layout_id": "layout_02",
        "contact": "ep_03",
        "safe": "ep_04",
        "kind": "vehicle",
        "family": "sedan",
        "target_blueprint": "vehicle.chevrolet.impala",
        "template": "v_c8_impala",
        "template_base": "v_c7_audi_tt",
        "old_assets": ("c7_vehicle_target",),
        "target_key": "c8_l02_target",
        "alias_key": "c8_l02_alias",
        "occluder_key": "c8_l02_occluder",
        "occluder_template": "v_bus",
        "target_track": "f_91",
        "alias_track": "f_92",
        "occluder_track": "f_93",
        "contact_trajectory": "c8_l02_target_contact",
        "safe_trajectory": "c8_l02_target_safe",
        "alias_trajectory": "c8_l02_alias_safe",
        "occluder_trajectory": "c8_l02_occluder_cross",
        "planned_occlusion_window_s": [1.3, 2.4],
        "expected_contact_time_s": 3.87,
    },
    {
        "layout_id": "c8_l03",
        "base_layout_id": "layout_03",
        "contact": "ep_05",
        "safe": "ep_06",
        "kind": "vehicle",
        "family": "van",
        "target_blueprint": "vehicle.volkswagen.t2",
        "template": "v_c8_t2",
        "template_base": "v_c7_audi_tt",
        "old_assets": ("c7_walker_target",),
        "target_key": "c8_l03_target",
        "alias_key": "c8_l03_alias",
        "occluder_key": "c8_l03_occluder",
        "occluder_template": "v_hgv",
        "target_track": "g_91",
        "alias_track": "g_92",
        "occluder_track": "g_93",
        "contact_trajectory": "c8_l03_target_contact",
        "safe_trajectory": "c8_l03_target_safe",
        "alias_trajectory": "c8_l03_alias_safe",
        "occluder_trajectory": "c8_l03_occluder_cross",
        "planned_occlusion_window_s": [1.2, 2.3],
        "expected_contact_time_s": 3.70,
    },
    {
        "layout_id": "c8_l04",
        "base_layout_id": "layout_04",
        "contact": "ep_07",
        "safe": "ep_08",
        "kind": "vehicle",
        "family": "emergency",
        "target_blueprint": "vehicle.dodge.charger_police",
        "template": "v_c8_charger_police",
        "template_base": "v_c7_audi_tt",
        "old_assets": ("c7_motorcycle_target",),
        "target_key": "c8_l04_target",
        "alias_key": "c8_l04_alias",
        "occluder_key": "c8_l04_occluder",
        "occluder_template": "v_firetruck",
        "target_track": "h_91",
        "alias_track": "h_92",
        "occluder_track": "h_93",
        "contact_trajectory": "c8_l04_target_contact",
        "safe_trajectory": "c8_l04_target_safe",
        "alias_trajectory": "c8_l04_alias_safe",
        "occluder_trajectory": "c8_l04_occluder_cross",
        "planned_occlusion_window_s": [1.5, 2.5],
        "expected_contact_time_s": 3.97,
    },
)


def require(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError(label)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"json_object:{path}")
    return value


def canonical_sha256(value: Any) -> str:
    return c2.sha256_bytes(c2.canonical_json_bytes(value))


def trajectory_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(c2.canonical_json_bytes(value)).hexdigest().upper()


def trajectory(
    start_forward_m: float,
    start_right_m: float,
    segments: tuple[tuple[float, float, float], ...],
) -> dict[str, Any]:
    return {
        "start_forward_m": start_forward_m,
        "start_right_m": start_right_m,
        "segments": [
            {
                "start_s": start_s,
                "velocity_forward_mps": velocity_forward_mps,
                "velocity_right_mps": velocity_right_mps,
            }
            for start_s, velocity_forward_mps, velocity_right_mps in segments
        ],
    }


def target_trajectories() -> dict[str, dict[str, Any]]:
    contact_specs = {
        "c8_l01": trajectory(
            8.5,
            0.20,
            ((0.0, -1.6, 0.0), (1.3, -2.0, 0.05), (2.4, -1.1, -0.05), (4.6, 0.0, 0.0)),
        ),
        "c8_l02": trajectory(
            10.0,
            0.50,
            ((0.0, -2.3, 0.0), (1.4, -1.6, -0.10), (2.5, -2.4, 0.10), (4.4, 0.0, 0.0)),
        ),
        "c8_l03": trajectory(
            9.2,
            -0.45,
            ((0.0, -2.0, 0.0), (1.2, -2.7, 0.15), (2.2, -1.8, -0.15), (4.5, 0.0, 0.0)),
        ),
        "c8_l04": trajectory(
            10.8,
            0.55,
            ((0.0, -2.8, 0.0), (1.5, -1.5, -0.20), (2.5, -2.2, 0.20), (4.6, 0.0, 0.0)),
        ),
    }
    safe_start_right_m = {
        "c8_l01": 1.45,
        "c8_l02": 2.50,
        "c8_l03": -2.65,
        "c8_l04": 2.95,
    }
    values: dict[str, dict[str, Any]] = {}
    for spec in PAIR_SPECS:
        layout_id = str(spec["layout_id"])
        contact = deepcopy(contact_specs[layout_id])
        safe = deepcopy(contact)
        safe["start_right_m"] = safe_start_right_m[layout_id]
        values[str(spec["contact_trajectory"])] = contact
        values[str(spec["safe_trajectory"])] = safe
    return values


ALIAS_TRAJECTORIES = {
    "c8_l01_alias_safe": trajectory(
        7.9, -2.2, ((0.0, -1.4, 0.05), (1.5, -1.0, 0.20), (2.5, -1.5, -0.20), (4.5, 0.0, 0.0))
    ),
    "c8_l02_alias_safe": trajectory(
        8.8, -3.2, ((0.0, -1.8, 0.0), (1.4, -2.2, 0.15), (2.5, -1.3, -0.15), (4.6, 0.0, 0.0))
    ),
    "c8_l03_alias_safe": trajectory(
        10.0, 3.4, ((0.0, -2.2, 0.0), (1.2, -1.5, -0.20), (2.3, -2.4, 0.20), (4.4, 0.0, 0.0))
    ),
    "c8_l04_alias_safe": trajectory(
        9.8, -3.6, ((0.0, -2.0, 0.0), (1.5, -2.5, 0.20), (2.6, -1.2, -0.20), (4.5, 0.0, 0.0))
    ),
}

OCCLUDER_TRAJECTORIES = {
    "c8_l01_occluder_cross": trajectory(
        1.4, 4.0, ((0.0, 0.0, 0.0), (0.5, 0.0, -4.0), (1.3, 0.0, -0.2), (2.4, 0.0, -5.0), (3.2, 0.0, 0.0))
    ),
    "c8_l02_occluder_cross": trajectory(
        1.7, 4.5, ((0.0, 0.0, 0.0), (0.4, 0.0, -4.5), (1.3, 0.0, -0.15), (2.35, 0.0, -5.2), (3.25, 0.0, 0.0))
    ),
    "c8_l03_occluder_cross": trajectory(
        1.6, -4.8, ((0.0, 0.0, 0.0), (0.45, 0.0, 4.6), (1.2, 0.0, 0.18), (2.3, 0.0, 5.1), (3.2, 0.0, 0.0))
    ),
    "c8_l04_occluder_cross": trajectory(
        1.8, 4.7, ((0.0, 0.0, 0.0), (0.6, 0.0, -4.2), (1.5, 0.0, -0.1), (2.5, 0.0, -5.5), (3.2, 0.0, 0.0))
    ),
}


def all_preferred_blueprints(protocol: Mapping[str, Any]) -> set[str]:
    values = {str(protocol["wearer"]["blueprint_candidates"][0])}
    values.update(
        str(template["blueprint_candidates"][0])
        for template in protocol["asset_templates"].values()
    )
    return values


def prior_contract(
    prior_protocols: Mapping[str, Mapping[str, Any]], asset_registry: Mapping[str, Any]
) -> dict[str, Any]:
    blueprints = {
        str(asset["blueprint_id"])
        for asset in asset_registry["assets"].values()
    }
    maps: set[str] = set()
    weather: set[str] = set()
    seeds: set[int] = set()
    trajectory_hashes: set[str] = set()
    protocol_hashes: dict[str, str] = {}
    for name, protocol in prior_protocols.items():
        blueprints.update(all_preferred_blueprints(protocol))
        maps.add(str(protocol["environment"]["map"]))
        weather.update(str(layout["weather"]) for layout in protocol["layouts"].values())
        seeds.add(int(protocol["capture"]["seed"]))
        trajectory_hashes.update(
            trajectory_hash(value) for value in protocol["trajectory_library"].values()
        )
        protocol_hashes[name] = canonical_sha256(protocol)
    return {
        "prior_protocol_canonical_sha256": protocol_hashes,
        "excluded_maps": sorted(maps),
        "excluded_weather_presets": sorted(weather),
        "excluded_capture_seeds": sorted(seeds),
        "excluded_blueprints": sorted(blueprints),
        "excluded_trajectory_sha256": sorted(trajectory_hashes),
    }


SUPPORT_REARRANGEMENT_PATTERN = (
    (-0.30, 0.20, -5.0),
    (0.25, -0.25, 4.0),
    (0.15, 0.30, -3.0),
    (-0.20, -0.15, 6.0),
)


def rearrange_dense_support(
    assets: list[dict[str, Any]], *, layout_number: int
) -> int:
    """Deterministically move every retained fixed-pose support asset."""

    rearranged = 0
    for ordinal, asset in enumerate(assets):
        fixed_pose = asset.get("fixed_pose")
        if not isinstance(fixed_pose, dict):
            continue
        delta_forward, delta_right, delta_yaw = SUPPORT_REARRANGEMENT_PATTERN[
            (ordinal + layout_number) % len(SUPPORT_REARRANGEMENT_PATTERN)
        ]
        fixed_pose["forward_m"] = round(float(fixed_pose["forward_m"]) + delta_forward, 6)
        fixed_pose["right_m"] = round(float(fixed_pose["right_m"]) + delta_right, 6)
        fixed_pose["yaw_offset_degrees"] = round(
            float(fixed_pose["yaw_offset_degrees"]) + delta_yaw, 6
        )
        rearranged += 1
    return rearranged


def make_layout(base: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    layout_id = str(spec["layout_id"])
    base_layout_id = str(spec["base_layout_id"])
    value = deepcopy(base["layouts"][base_layout_id])
    value["display_name"] = f"c8_transport_cone_{spec['family']}_alias_occlusion"
    value["weather"] = WEATHER_BY_LAYOUT[layout_id]
    value["duration_seconds"] = DURATION_SECONDS
    value["showcase_time_s"] = 2.0
    value["anchor"] = deepcopy(ANCHORS[layout_id])
    value["witness"] = deepcopy(base["layouts"][base_layout_id]["witness"])
    value["anchor_source"] = {
        "map": MAP_ID,
        "authority": "C7_SOURCE_PROTOCOL_NUMERIC_ANCHOR_REUSE",
        "base_layout_id": base_layout_id,
        "evidence_or_outcome_consulted": False,
    }
    removed = set(spec["old_assets"])
    value["assets"] = [
        deepcopy(asset) for asset in value["assets"] if asset["asset_key"] not in removed
    ]
    layout_number = int(layout_id[-2:])
    rearranged = rearrange_dense_support(value["assets"], layout_number=layout_number)
    value["dense_support_rearrangement"] = {
        "schema": "dtr-c8-dense-support-rearrangement-v1",
        "fixed_pose_assets_rearranged": rearranged,
        "pattern_forward_right_yaw": [list(value) for value in SUPPORT_REARRANGEMENT_PATTERN],
        "prior_evidence_or_outcome_consulted": False,
    }
    value["assets"].extend(
        [
            {
                "asset_key": spec["target_key"],
                "track_id": spec["target_track"],
                "role": f"c8_fresh_dynamic_{spec['family']}_target",
                "template": spec["template"],
                "trajectory_key": spec["target_key"],
                "scripted_pose_authority": True,
            },
            {
                "asset_key": spec["alias_key"],
                "track_id": spec["alias_track"],
                "role": f"c8_same_blueprint_{spec['family']}_alias",
                "template": spec["template"],
                "trajectory_key": spec["alias_key"],
                "scripted_pose_authority": True,
            },
            {
                "asset_key": spec["occluder_key"],
                "track_id": spec["occluder_track"],
                "role": "c8_physical_transport_cone_occluder",
                "template": spec["occluder_template"],
                "trajectory_key": spec["occluder_key"],
                "scripted_pose_authority": True,
            },
        ]
    )
    return value


def issued_plan(pair_number: int) -> dict[str, Any]:
    return {
        "plan_id": f"plan_c8_pair_{pair_number:02d}",
        "session_id": f"session_c8_pair_{pair_number:02d}",
        "issued_at_s": 0.0,
        "expires_at_s": DURATION_SECONDS,
        "time_parameterized_waypoints": [
            {"time_s": 0.0, "forward_m": -6.0, "right_m": 0.0},
            {"time_s": 1.0, "forward_m": -4.1, "right_m": 0.0},
            {"time_s": 5.0, "forward_m": 4.1, "right_m": 0.0},
            {"time_s": 9.0, "forward_m": 12.1, "right_m": 0.0},
        ],
    }


def make_scenarios() -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for pair_number, spec in enumerate(PAIR_SPECS, start=1):
        for episode_id, outcome, target_trajectory in (
            (spec["contact"], "CONTACT", spec["contact_trajectory"]),
            (spec["safe"], "SAFE", spec["safe_trajectory"]),
        ):
            values.append(
                {
                    "episode_id": episode_id,
                    "layout_id": spec["layout_id"],
                    "scenario_role": (
                        f"c8_transport_cone_{spec['family']}_{outcome.casefold()}"
                    ),
                    "twin_role": outcome.casefold(),
                    "navigation_session_id": f"session_c8_pair_{pair_number:02d}",
                    "expected_outcome": outcome,
                    "expected_responsible_assets": (
                        [spec["target_key"]] if outcome == "CONTACT" else []
                    ),
                    "wearer_trajectory": "c8_wearer_route",
                    "asset_trajectories": {
                        spec["target_key"]: target_trajectory,
                        spec["alias_key"]: spec["alias_trajectory"],
                        spec["occluder_key"]: spec["occluder_trajectory"],
                    },
                    "issued_plan": issued_plan(pair_number),
                }
            )
    return sorted(values, key=lambda value: value["episode_id"])


def pair_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "family": spec["family"],
        "contact_episode": spec["contact"],
        "safe_episode": spec["safe"],
        "layout_id": spec["layout_id"],
        "target_asset": spec["target_key"],
        "alias_asset": spec["alias_key"],
        "occluder_asset": spec["occluder_key"],
        "target_blueprint": spec["target_blueprint"],
        "pair_difference": "TARGET_INITIAL_LATERAL_POSITION_ONLY",
        "target_velocity_segments_identical": True,
        "causally_distinguishable_from_s": 0.0,
        "planned_occlusion_window_s": spec["planned_occlusion_window_s"],
        "expected_contact_time_s": spec["expected_contact_time_s"],
    }


def materialize(
    base: dict[str, Any],
    prior_protocols: Mapping[str, Mapping[str, Any]],
    asset_registry: Mapping[str, Any],
) -> dict[str, Any]:
    require(base.get("cohort_id") == PARENT_COHORT_ID, "c8_unexpected_parent")
    protocol = deepcopy(base)
    protocol["schema_version"] = SCHEMA_VERSION
    protocol["cohort_id"] = COHORT_ID
    protocol["evidence_class"] = "synthetic_fresh_source_disjoint_x31_transport_cone_source"
    protocol["objective"] = (
        "Falsify frozen X31 ambiguity-preserving set-valued surface transport "
        "continuity against X24 on four fresh same-blueprint alias and physical-"
        "occlusion CONTACT/SAFE pairs, without retaining prior outcomes."
    )
    protocol["environment"]["map"] = MAP_ID
    protocol["environment"]["sample_seconds"] = SAMPLE_SECONDS
    protocol["environment"]["fixed_delta_seconds"] = SAMPLE_SECONDS
    protocol["capture"]["seed"] = CAPTURE_SEED

    for spec in PAIR_SPECS:
        template = deepcopy(protocol["asset_templates"][str(spec["template_base"])])
        template["blueprint_candidates"] = [spec["target_blueprint"]]
        protocol["asset_templates"][str(spec["template"])] = template

    # Dense support actors retain their registered trajectories.  No retained
    # trajectory is used by a C8 target, alias, occluder, or wearer.
    protocol["trajectory_library"] = deepcopy(base["trajectory_library"])
    protocol["trajectory_library"].update(
        {
            "c8_wearer_route": trajectory(
                -6.0,
                0.0,
                ((0.0, 1.9, 0.0), (1.0, 2.05, 0.0), (5.0, 2.0, 0.0)),
            ),
            **target_trajectories(),
            **deepcopy(ALIAS_TRAJECTORIES),
            **deepcopy(OCCLUDER_TRAJECTORIES),
        }
    )
    protocol["layouts"] = {
        str(spec["layout_id"]): make_layout(base, spec) for spec in PAIR_SPECS
    }
    protocol["scenarios"] = make_scenarios()
    # C2's joiner requires selected occlusion indices to be identical across
    # every episode listed in one contract.  CONTACT/SAFE are intentionally
    # distinguishable at t=0, so each episode receives its own contract.
    protocol["twin_contracts"] = []
    protocol["occlusion_contracts"] = []
    for pair_index, spec in enumerate(PAIR_SPECS, start=1):
        for episode_id, outcome in (
            (str(spec["contact"]), "CONTACT"),
            (str(spec["safe"]), "SAFE"),
        ):
            protocol["occlusion_contracts"].append(
                {
                    "contract_id": f"c8_transport_cone_loss_{episode_id}",
                    "episodes": [episode_id],
                    "target_asset": spec["target_key"],
                    "occluder_asset": spec["occluder_key"],
                    "alias_asset": spec["alias_key"],
                    "pair_index": pair_index,
                    "minimum_pre_track_frames": 10,
                    "minimum_post_reappearance_frames": 8,
                    "minimum_trackable_pixel_fraction": 0.0002,
                    "complete_occlusion_pixel_fraction": 0.0,
                    "minimum_complete_occlusion_seconds": 0.6,
                    "maximum_complete_occlusion_seconds": 1.3,
                    "required_outcomes": {episode_id: outcome},
                    "alias_stress": {
                        "same_exact_blueprint_as_target": True,
                        "minimum_alias_visible_boundary_frames": 4,
                        "minimum_projected_bbox_iou": 0.25,
                        "minimum_bbox_iou_frames": 3,
                        "alias_must_remain_contact_safe": True,
                    },
                    "planned_occlusion_window_s": spec["planned_occlusion_window_s"],
                }
            )

    protocol["admission"]["expected_episode_count"] = len(EPISODES)
    protocol["admission"]["expected_layout_count"] = len(PAIR_SPECS)
    protocol["admission"]["minimum_active_assets_per_layout_excluding_wearer"] = 28
    protocol["admission"]["minimum_unique_actual_blueprints_across_pack"] = 60
    protocol["admission"]["c8_transport_cone_source_gates"] = {
        "all_episodes_new_capture": True,
        "minimum_full_occlusion_frames": 6,
        "maximum_full_occlusion_frames": 13,
        "minimum_pre_track_frames": 10,
        "minimum_post_reappearance_frames": 8,
        "one_episode_per_occlusion_contract": True,
        "cross_outcome_occlusion_indices_must_match": False,
        "same_blueprint_alias_required": True,
        "alias_and_occluder_must_be_contact_safe": True,
        "failure_disposition": "WHOLE_COHORT_NOT_EVALUABLE_NO_RESEED_OR_SUBSTITUTION",
    }

    dynamic_pairs = [pair_contract(spec) for spec in PAIR_SPECS]
    protocol["evaluation_contract"] = {
        "truth_tail_seconds": TRUTH_TAIL_SECONDS,
        "future_truth_rule": "SCORE_ONLY_FRAMES_WITH_FULL_CAPTURED_HORIZON",
        "score_window_end_seconds": {
            episode_id: SCORE_WINDOW_END_SECONDS for episode_id in EPISODES
        },
        "contact_episodes": list(CONTACT_EPISODES),
        "safe_episodes": list(SAFE_EPISODES),
        "fresh_dynamic_contact_episodes": list(CONTACT_EPISODES),
        "fresh_dynamic_safe_episodes": list(SAFE_EPISODES),
        "retained_occlusion_episodes": [],
        "safe_segment_start_seconds": {
            episode_id: SAFE_SEGMENT_START_SECONDS for episode_id in SAFE_EPISODES
        },
        "dynamic_pairs": dynamic_pairs,
        "all_physical_obstacles_are_truth_relevant": True,
        "transport_cone_falsification": {
            "contact_pre_loss_future_positive_risk_required": True,
            "zero_future_positive_risk_gaps_during_full_loss": True,
            "first_reappearance_future_positive_risk_required": True,
            "post_loss_risk_branch_must_descend_from_pre_loss_authorized_branch": True,
            "single_conflicting_shift_must_not_revoke_compatible_branch": True,
            "safe_alias_confirmed_risk_frames_maximum": 0,
        },
        "frozen_thresholds": {
            "minimum_dynamic_contact_lead_seconds": 2.0,
            "minimum_dynamic_contact_future_positive_recall": 0.80,
            "maximum_safe_risk_segments": 0,
            "minimum_aggregate_precision": 0.95,
            "minimum_aggregate_f1": 0.80,
            "x31_f1_must_strictly_exceed_x24": True,
        },
    }

    prior = prior_contract(prior_protocols, asset_registry)
    protocol["source_disjoint_contract"] = {
        "parent_cohort_id": PARENT_COHORT_ID,
        "parent_protocol_canonical_sha256": canonical_sha256(base),
        "prior_protocol_canonical_sha256": prior["prior_protocol_canonical_sha256"],
        "confirmation_episodes": list(EPISODES),
        "retained_regression_episodes": [],
        "map": MAP_ID,
        "map_disjoint_from_prior_protocols": False,
        "same_map_numeric_anchor_and_witness_reuse": True,
        "same_map_claim_boundary": (
            "SOURCE_DISJOINT_IN_SEED_WEATHER_TARGET_ALIAS_OCCLUDER_TRAJECTORY_AND_DENSE_SUPPORT_POSE_NOT_MAP"
        ),
        "new_capture_seed": CAPTURE_SEED,
        "weather_by_layout": WEATHER_BY_LAYOUT,
        "new_dynamic_target_blueprints": {
            str(spec["family"]): spec["target_blueprint"] for spec in PAIR_SPECS
        },
        "dynamic_pairs": dynamic_pairs,
        "excluded_prior_maps": prior["excluded_maps"],
        "excluded_prior_weather_presets": prior["excluded_weather_presets"],
        "excluded_prior_capture_seeds": prior["excluded_capture_seeds"],
        "excluded_prior_blueprints": prior["excluded_blueprints"],
        "excluded_prior_trajectory_sha256": prior["excluded_trajectory_sha256"],
        "prior_outcomes_or_evidence_referenced": False,
        "support_asset_family_reuse_allowed": True,
        "target_alias_occluder_trajectories_must_be_canonically_fresh": True,
    }
    protocol["capture_compatibility_contract"] = {
        "protocol_schema_compatible_with_c2_rich_capture": True,
        "numeric_anchor_fields_complete": True,
        "requested_server_map": MAP_ID,
        "generic_runner_launches_without_explicit_map_argument": True,
        "installed_runner_default_map_matches_protocol": True,
        "capture_entrypoint_rejects_map_mismatch": True,
        "capture_ready_with_current_generic_runner": True,
        "required_pre_capture_check": "CAPTURE_ENTRYPOINT_WORLD_MAP_NAME_MUST_MATCH_PROTOCOL",
    }
    protocol["claim_boundary"] = [
        "C8 is wholly fresh scripted CARLA Development source; no C3-C7 episode, evaluator outcome, prediction, or result enters its denominator.",
        "C8 deliberately reuses the proven Town10HD_Opt map plus C7 numeric anchors and witnesses; source-disjointness does not include map or anchor identity.",
        "C8 freezes new seed, weather, target blueprints, same-blueprint aliases, occluders, dynamic trajectories, and deterministic dense-support poses.",
        "Every CONTACT/SAFE target pair is distinguishable from t=0 by initial lateral position while retaining identical velocity segments; SAFE scoring begins at t=0.",
        "Each episode has its own physical-occlusion contract; no cross-outcome equality of realized occlusion indices is claimed.",
        "The model sees only dense wearable RGB, metric depth, calibration, current camera/wearer pose, and immutable issued-plan receipts; actor identity and outcome remain evaluator-only.",
        "X31 and the C8 scorer must be hash-frozen before capture and run once without threshold, detector, tracker, route, lifecycle, seed, or cohort sweeps.",
        "A failed source-admission contract is NOT_EVALUABLE and cannot be repaired by reseeding, substituting assets, or selecting successful episodes.",
        "The installed generic runner's Town10HD_Opt default matches this protocol, and the capture entrypoint still fails closed on any world-map mismatch.",
        "No result establishes real-world, product, deployment, user-benefit, reliability, or safety authority.",
    ]
    return protocol


def validate_c8(
    protocol: dict[str, Any],
    base: dict[str, Any],
    prior_protocols: Mapping[str, Mapping[str, Any]],
    asset_registry: Mapping[str, Any],
) -> None:
    c2.validate_protocol(protocol)
    require(protocol.get("schema_version") == SCHEMA_VERSION, "c8_schema")
    require(protocol.get("cohort_id") == COHORT_ID, "c8_cohort")
    require(protocol["environment"]["map"] == MAP_ID, "c8_map")
    require(float(protocol["environment"]["sample_seconds"]) == SAMPLE_SECONDS, "c8_sample")
    require(int(protocol["capture"]["seed"]) == CAPTURE_SEED, "c8_seed")
    require(tuple(protocol["layouts"]) == tuple(WEATHER_BY_LAYOUT), "c8_layout_order")
    require(
        {key: value["weather"] for key, value in protocol["layouts"].items()}
        == WEATHER_BY_LAYOUT,
        "c8_weather",
    )
    require(tuple(value["episode_id"] for value in protocol["scenarios"]) == EPISODES, "c8_episode_order")
    require(len(protocol["occlusion_contracts"]) == 8, "c8_occlusion_contract_count")
    require(protocol["twin_contracts"] == [], "c8_twin_contracts_not_empty")
    occlusion_episodes = [
        str(contract["episodes"][0])
        for contract in protocol["occlusion_contracts"]
        if len(contract["episodes"]) == 1
    ]
    require(tuple(occlusion_episodes) == EPISODES, "c8_single_episode_occlusion_order")
    require(protocol["evaluation_contract"]["retained_occlusion_episodes"] == [], "c8_retained")
    require(
        set(protocol["evaluation_contract"]["safe_segment_start_seconds"].values())
        == {0.0},
        "c8_safe_scoring_not_from_zero",
    )
    require(
        protocol["capture_compatibility_contract"]["capture_ready_with_current_generic_runner"] is True,
        "c8_runner_not_ready",
    )

    prior = prior_contract(prior_protocols, asset_registry)
    require(MAP_ID in set(prior["excluded_maps"]), "c8_same_map_boundary_missing")
    require(CAPTURE_SEED not in set(prior["excluded_capture_seeds"]), "c8_seed_not_fresh")
    require(
        not (set(WEATHER_BY_LAYOUT.values()) & set(prior["excluded_weather_presets"])),
        "c8_weather_not_fresh",
    )
    fresh_targets = {str(spec["target_blueprint"]) for spec in PAIR_SPECS}
    require(not (fresh_targets & set(prior["excluded_blueprints"])), "c8_blueprints_not_fresh")

    scenarios = {value["episode_id"]: value for value in protocol["scenarios"]}
    new_trajectory_names = {
        "c8_wearer_route",
        *(str(spec["contact_trajectory"]) for spec in PAIR_SPECS),
        *(str(spec["safe_trajectory"]) for spec in PAIR_SPECS),
        *(str(spec["alias_trajectory"]) for spec in PAIR_SPECS),
        *(str(spec["occluder_trajectory"]) for spec in PAIR_SPECS),
    }
    new_hashes = {
        trajectory_hash(protocol["trajectory_library"][name]) for name in new_trajectory_names
    }
    require(len(new_hashes) == len(new_trajectory_names), "c8_trajectory_hash_collision")
    require(
        not (new_hashes & set(prior["excluded_trajectory_sha256"])),
        "c8_trajectory_not_fresh",
    )
    for spec in PAIR_SPECS:
        layout_id = str(spec["layout_id"])
        layout = protocol["layouts"][layout_id]
        require(float(layout["duration_seconds"]) == DURATION_SECONDS, f"c8_duration:{layout_id}")
        require(layout["anchor"] == ANCHORS[layout_id], f"c8_anchor:{layout_id}")
        base_layout_id = str(spec["base_layout_id"])
        require(
            layout["anchor"] == base["layouts"][base_layout_id]["anchor"],
            f"c8_anchor_not_c7_source:{layout_id}",
        )
        require(
            layout["witness"] == base["layouts"][base_layout_id]["witness"],
            f"c8_witness_not_c7_source:{layout_id}",
        )
        require(layout["anchor_source"]["map"] == MAP_ID, f"c8_anchor_map:{layout_id}")
        require(
            int(layout["dense_support_rearrangement"]["fixed_pose_assets_rearranged"]) > 0,
            f"c8_dense_support_not_rearranged:{layout_id}",
        )
        assets = {value["asset_key"]: value for value in c2.materialize_layout_assets(protocol, layout_id)}
        target = assets[str(spec["target_key"])]
        alias = assets[str(spec["alias_key"])]
        require(
            target["blueprint_candidates"] == alias["blueprint_candidates"] == [spec["target_blueprint"]],
            f"c8_same_blueprint_alias:{layout_id}",
        )
        require(bool(target["scripted_pose_authority"]) and bool(alias["scripted_pose_authority"]), f"c8_pose_authority:{layout_id}")
        contact = scenarios[str(spec["contact"])]
        safe = scenarios[str(spec["safe"])]
        require(contact["layout_id"] == safe["layout_id"] == layout_id, f"c8_pair_layout:{layout_id}")
        require(contact["expected_responsible_assets"] == [spec["target_key"]], f"c8_contact_responsible:{layout_id}")
        require(safe["expected_responsible_assets"] == [], f"c8_safe_responsible:{layout_id}")
        contact_trajectory = deepcopy(
            protocol["trajectory_library"][str(spec["contact_trajectory"])]
        )
        safe_trajectory = deepcopy(
            protocol["trajectory_library"][str(spec["safe_trajectory"])]
        )
        contact_right = float(contact_trajectory.pop("start_right_m"))
        safe_right = float(safe_trajectory.pop("start_right_m"))
        require(contact_trajectory == safe_trajectory, f"c8_pair_velocity_drift:{layout_id}")
        require(contact_right != safe_right, f"c8_pair_not_distinct_at_t0:{layout_id}")
        require(
            DURATION_SECONDS + 1e-9 >= SCORE_WINDOW_END_SECONDS + TRUTH_TAIL_SECONDS,
            f"c8_truth_tail:{layout_id}",
        )
    source = protocol["source_disjoint_contract"]
    require(source["parent_protocol_canonical_sha256"] == canonical_sha256(base), "c8_parent_hash")
    require(source["prior_outcomes_or_evidence_referenced"] is False, "c8_outcome_reference")


def serialized_protocol(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--asset-registry", type=Path, default=DEFAULT_ASSET_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = read_json(args.base.resolve(strict=True))
    asset_registry = read_json(args.asset_registry.resolve(strict=True))
    prior_protocols = {
        path.name: read_json(path.resolve(strict=True)) for path in DEFAULT_PRIOR_PROTOCOLS
    }
    first = materialize(base, prior_protocols, asset_registry)
    second = materialize(base, prior_protocols, asset_registry)
    first_bytes = serialized_protocol(first)
    require(first_bytes == serialized_protocol(second), "c8_nondeterministic")
    validate_c8(first, base, prior_protocols, asset_registry)
    if args.validate_only:
        require(args.output.exists(), "c8_output_missing")
        require(args.output.read_bytes() == first_bytes, "c8_existing_protocol_drift")
    else:
        if args.output.exists() and args.output.read_bytes() != first_bytes:
            raise FileExistsError(f"existing C8 protocol differs: {args.output}")
        if not args.output.exists():
            args.output.write_bytes(first_bytes)
    frames_per_sensor = len(EPISODES) * (int(round(DURATION_SECONDS / SAMPLE_SECONDS)) + 1)
    print(
        json.dumps(
            {
                "status": "C8_PROTOCOL_VALID",
                "output": str(args.output.resolve()),
                "cohort_id": COHORT_ID,
                "episodes": len(EPISODES),
                "layouts": len(PAIR_SPECS),
                "frames_per_sensor": frames_per_sensor,
                "sensor_payload_frames": frames_per_sensor * len(first["capture"]["sensor_order"]),
                "protocol_sha256": c2.sha256_bytes(first_bytes),
                "capture_ready_with_current_generic_runner": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
