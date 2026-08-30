"""Materialize the one-shot C11 solid-body successor to terminal C10.

C10 is a consumed SOURCE_NOT_EVALUABLE source attempt.  Its failure result and
instance JSONL are used only to replace a raster-permeable bus silhouette with
an already observed solid-body firetruck representation.  No C10 algorithm,
model, evaluator score, X24 result, or X31 result exists or is opened here.
Every target-side and algorithm-side contract remains frozen.
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c8_x31_transport_cone_protocol as c8


HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "dtr_carla_c10_x31_opaque_single_pass_protocol.json"
DEFAULT_OUTPUT = HERE / "dtr_carla_c11_x31_solid_body_protocol.json"

SCHEMA_VERSION = 11
COHORT_ID = "DTR_CARLA_C11_X31_SOLID_BODY_SOURCE_DISJOINT_V1"
PARENT_COHORT_ID = "DTR_CARLA_C10_X31_OPAQUE_SINGLE_PASS_SOURCE_DISJOINT_V1"
PARENT_PROTOCOL_SHA256 = (
    "255402227C86133B95706F7B49957F9AB463A146EDC3D266853381794BC5078B"
)
PARENT_PROTOCOL_CANONICAL_SHA256 = (
    "A6AF4BA1B931AAE5502925904B11D96FDA4825DD0D90237B228241015723E951"
)
CAPTURE_SEED = 131365
C10_RUN_ID = "c10-x31-opaque-20260830-203500"
C10_RESULT_SHA256 = (
    "CD8F1C6043B4687044CF568C626ADCF54359DE63B35EA6A5317007C4721A363B"
)
OCCLUDER_TEMPLATE = "v_firetruck"
OCCLUDER_BLUEPRINT = "vehicle.carlamotors.firetruck"
OCCLUDER_BBOX_LOCATION = (-0.25344201922416687, 0.005309276282787323, 1.91335928440094)
OCCLUDER_BBOX_EXTENT = (4.234020709991455, 1.4455441236495972, 1.9137061834335327)
OCCLUDER_BBOX_ROTATION = (0.0, 0.0, 0.0)
OCCLUDER_START_FORWARD_M = 3.2
OCCLUDER_START_RIGHT_M = 8.0
OCCLUDER_CROSS_START_S = 0.9
OCCLUDER_CROSS_STOP_S = 2.0
OCCLUDER_RIGHT_VELOCITY_MPS = -10.0
OCCLUDER_FINAL_RIGHT_M = -3.0
MINIMUM_STATIC_CLEARANCE_M = 0.25
EXPECTED_CONTAINMENT_RUNS = {
    "ep_01": (14, 19),
    "ep_02": (13, 19),
    "ep_03": (14, 19),
    "ep_04": (13, 19),
    "ep_05": (14, 19),
    "ep_06": (16, 21),
    "ep_07": (14, 19),
    "ep_08": (12, 19),
}
LAYOUT_IDS = tuple(str(spec["layout_id"]) for spec in c8.PAIR_SPECS)
TARGET_BBOX_RECEIPTS = {
    "c8_l01": {
        "blueprint": "walker.pedestrian.0044",
        "surface_offset_m": 0.8,
        "location": (0.0, 0.0, 0.0),
        "extent": (0.18767888844013214, 0.18767888844013214, 0.9300000071525574),
        "rotation": (0.0, 0.0, 0.0),
    },
    "c8_l02": {
        "blueprint": "vehicle.chevrolet.impala",
        "surface_offset_m": 0.3,
        "location": (5.974524901830591e-05, 0.001786777633242309, 0.7014925479888916),
        "extent": (2.6787397861480713, 1.0166014432907104, 0.7053293585777283),
        "rotation": (0.0, 0.0, 0.0),
    },
    "c8_l03": {
        "blueprint": "vehicle.volkswagen.t2",
        "surface_offset_m": 0.3,
        "location": (0.0013318287674337626, -0.0005364218377508223, 1.0144717693328857),
        "extent": (2.2402184009552, 1.034657597541809, 1.0188959836959839),
        "rotation": (0.0, 0.0, 0.0),
    },
    "c8_l04": {
        "blueprint": "vehicle.dodge.charger_police",
        "surface_offset_m": 0.3,
        "location": (0.023636838421225548, 0.0006008386262692511, 0.7926706671714783),
        "extent": (2.487122058868408, 1.0192005634307861, 0.7710590958595276),
        "rotation": (0.0, 0.0, 0.0),
    },
}
PREDECESSOR_DISPOSITION = {
    "cohort_id": PARENT_COHORT_ID,
    "protocol_sha256": PARENT_PROTOCOL_SHA256,
    "run_id": C10_RUN_ID,
    "status": "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_NOT_EVALUABLE",
    "terminal_stage": "JOINED_SOURCE_PHYSICAL_OCCLUSION_ADMISSION",
    "result_sha256": C10_RESULT_SHA256,
    "instance_result_present": True,
    "instance_jsonl_and_manifests_present": True,
    "instance_source_outcome_accessed_for_c11_geometry_design": True,
    "instance_jsonl_and_manifests_accessed_for_c11_geometry_design": True,
    "source_model_package_present": True,
    "source_evaluator_truth_present": True,
    "algorithm_model_run_present": False,
    "scorer_result_present": False,
    "x24_or_x31_outcome_present": False,
    "algorithm_model_scorer_x24_or_x31_outcome_accessed": False,
    "resume_replay_reseed_substitution_or_partial_selection_authorized": False,
}


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(label)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def serialized_protocol(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def solid_body_single_pass_trajectory() -> dict[str, Any]:
    return {
        "start_forward_m": OCCLUDER_START_FORWARD_M,
        "start_right_m": OCCLUDER_START_RIGHT_M,
        "segments": [
            {"start_s": 0.0, "velocity_forward_mps": 0.0, "velocity_right_mps": 0.0},
            {
                "start_s": OCCLUDER_CROSS_START_S,
                "velocity_forward_mps": 0.0,
                "velocity_right_mps": OCCLUDER_RIGHT_VELOCITY_MPS,
            },
            {
                "start_s": OCCLUDER_CROSS_STOP_S,
                "velocity_forward_mps": 0.0,
                "velocity_right_mps": 0.0,
            },
        ],
    }


def geometry_receipt() -> dict[str, Any]:
    return {
        "schema": "dtr-c11-static-solid-body-single-pass-launch-falsifier-v1",
        "authority": "PRELAUNCH_FALSIFIER_NOT_PIXEL_OR_OUTCOME_PROOF",
        "source_run_id": C10_RUN_ID,
        "source_result_sha256": C10_RESULT_SHA256,
        "sample_seconds": c8.SAMPLE_SECONDS,
        "frames_per_episode": int(round(c8.DURATION_SECONDS / c8.SAMPLE_SECONDS)) + 1,
        "camera_model": {
            "relative_forward_m": 0.08,
            "relative_right_m": 0.0,
            "height_above_local_surface_m": 1.45,
            "pitch_degrees": -5.0,
            "fov_degrees": 90.0,
        },
        "solid_body_occluder": {
            "template": OCCLUDER_TEMPLATE,
            "blueprint": OCCLUDER_BLUEPRINT,
            "surface_offset_m": 0.3,
            "bbox_location": list(OCCLUDER_BBOX_LOCATION),
            "bbox_extent": list(OCCLUDER_BBOX_EXTENT),
            "bbox_rotation_degrees": list(OCCLUDER_BBOX_ROTATION),
            "c9_instance_pixel_occlusion_evidence": "FIRETRUCK_SOLID_BODY_OBSERVED",
            "c10_bus_raster_permeability_removed": True,
        },
        "target_bbox_by_layout": {
            key: {
                child_key: list(child_value) if isinstance(child_value, tuple) else child_value
                for child_key, child_value in value.items()
            }
            for key, value in TARGET_BBOX_RECEIPTS.items()
        },
        "expected_single_containment_runs_inclusive": {
            key: list(value) for key, value in EXPECTED_CONTAINMENT_RUNS.items()
        },
        "required_duration_seconds": [0.6, 1.3],
        "required_pre_track_frames": 10,
        "required_post_reappearance_frames": 8,
        "minimum_xy_clearance_beyond_wearer_radius_m": MINIMUM_STATIC_CLEARANCE_M,
        "expected_minimum_xy_clearance_beyond_wearer_radius_m": 1.099147,
        "expected_final_right_m": OCCLUDER_FINAL_RIGHT_M,
        "formal_instance_pixels_remain_authority": True,
    }


def materialize(base: Mapping[str, Any]) -> dict[str, Any]:
    require(base.get("cohort_id") == PARENT_COHORT_ID, "c11_unexpected_parent")
    protocol = deepcopy(base)
    protocol["schema_version"] = SCHEMA_VERSION
    protocol["cohort_id"] = COHORT_ID
    protocol["evidence_class"] = "synthetic_fresh_source_disjoint_x31_solid_body_development_source"
    protocol["objective"] = (
        "Run one fresh Development capture with one standardized solid-body "
        "firetruck and one monotonic lateral pass per layout, after consumed C10 "
        "proved the Fusorosa OBB reachable but its raster silhouette permeable."
    )
    protocol["capture"]["seed"] = CAPTURE_SEED

    trajectory = solid_body_single_pass_trajectory()
    occluder_keys: list[str] = []
    for index, layout_id in enumerate(LAYOUT_IDS, start=1):
        occluder_key = f"c8_l0{index}_occluder"
        trajectory_name = f"c8_l0{index}_occluder_cross"
        protocol["trajectory_library"][trajectory_name] = deepcopy(trajectory)
        occluder_keys.append(occluder_key)
        matched = [
            asset for asset in protocol["layouts"][layout_id]["assets"]
            if asset["asset_key"] == occluder_key
        ]
        require(len(matched) == 1, f"c11_occluder_asset_missing:{layout_id}")
        matched[0]["template"] = OCCLUDER_TEMPLATE
        matched[0]["collisions_enabled"] = False

    admission = protocol["admission"]
    prior_gates = admission.pop("c10_opaque_single_pass_source_gates")
    prior_gates.pop("all_four_occluders_exact_opaque_bus_blueprint", None)
    admission["c11_solid_body_source_gates"] = {
        **prior_gates,
        "all_four_occluders_exact_solid_body_firetruck_blueprint": True,
        "all_four_occluders_exact_single_pass_trajectory": True,
        "c10_raster_permeable_bus_representation_removed": True,
        "static_3d_obb_launch_falsifier_required": True,
        "static_falsifier_is_not_pixel_or_outcome_proof": True,
        "formal_instance_pixels_remain_occlusion_authority": True,
        "maximum_fresh_capture_attempts": 1,
        "resume_replay_reseed_substitution_or_partial_selection_allowed": False,
    }
    protocol.pop("c10_static_launch_falsifier")
    protocol["c11_static_launch_falsifier"] = geometry_receipt()

    protocol["source_disjoint_contract"] = {
        "parent_cohort_id": PARENT_COHORT_ID,
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "parent_protocol_canonical_sha256": PARENT_PROTOCOL_CANONICAL_SHA256,
        "predecessor_c10_disposition": deepcopy(PREDECESSOR_DISPOSITION),
        "confirmation_episodes": [],
        "development_episodes": list(c8.EPISODES),
        "retained_regression_episodes": [],
        "map": c8.MAP_ID,
        "map_disjoint_from_predecessor": False,
        "same_map_numeric_anchor_witness_weather_and_dense_support_reuse": True,
        "new_capture_seed": CAPTURE_SEED,
        "predecessor_capture_seed": 130365,
        "weather_by_layout": deepcopy(c8.WEATHER_BY_LAYOUT),
        "dynamic_pairs": deepcopy(base["evaluation_contract"]["dynamic_pairs"]),
        "reused_from_c10": [
            "MAP", "NUMERIC_ANCHORS", "WITNESSES", "WEATHER",
            "TARGET_ALIAS_BLUEPRINTS", "TARGET_ALIAS_WEARER_TRAJECTORIES",
            "DENSE_SUPPORT_POSES", "EPISODE_PAIRING", "SCENARIO_EXPECTED_LABELS",
            "OCCLUSION_CONTRACTS", "MODEL_DETECTOR_X31_ROUTE_THRESHOLDS",
            "EVALUATION_AND_SCORE_GATES", "SCRIPTED_COLLISION_DISABLE_POLICY",
        ],
        "structural_source_change": "RASTER_PERMEABLE_BUS_TO_SOLID_BODY_FIRETRUCK_SINGLE_PASS",
        "changed_occluder_assets": occluder_keys,
        "fresh_capture_identity": "SEED_131365_ONE_COMPLETE_CAPTURE_ONLY",
        "predecessor_instance_source_outcome_accessed_for_geometry_reachability": True,
        "predecessor_instance_jsonl_and_manifests_accessed_for_geometry_reachability": True,
        "predecessor_algorithm_model_scorer_x24_or_x31_outcome_existed": False,
        "predecessor_algorithm_model_scorer_x24_or_x31_outcome_accessed": False,
        "c9_firetruck_instance_jsonl_accessed_for_solid_body_reachability": True,
        "c9_algorithm_model_evaluator_scorer_x24_or_x31_outcome_existed": False,
        "c9_algorithm_model_evaluator_scorer_x24_or_x31_outcome_accessed": False,
        "development_only_never_confirmation": True,
        "one_fresh_capture_only": True,
        "resume_replay_reseed_substitution_or_partial_selection_allowed": False,
    }
    protocol["claim_boundary"] = [
        "C10 is terminal SOURCE_NOT_EVALUABLE and is not resumed, replayed, reseeded, substituted, or partially selected.",
        "C10 failure result and instance JSONL were accessed only for C11 source-representation reachability design.",
        "No C10 algorithm model run, scorer, X24 outcome, or X31 outcome existed or was accessed.",
        "C9 firetruck instance rows were accessed only as solid-body source reachability evidence; no C9 algorithm outcome existed or was accessed.",
        "C11 changes only the physical-occluder representation and matched crossing dwell: every layout uses the exact solid-body firetruck and exact single monotonic pass.",
        "Targets, aliases, wearer, map, weather, scenarios, expected labels, collision disable policy, X31, detector, model, route thresholds, and evaluation gates remain frozen.",
        "The static 3-D OBB construction is a prelaunch falsifier only; formal instance pixels remain the occlusion authority.",
        "C11 is fresh Development only and can never count as confirmation.",
        "No result establishes real-world, product, deployment, user-benefit, reliability, or safety authority.",
    ]
    return protocol


def _yaw_radians(trajectory: Mapping[str, Any], time_s: float) -> float:
    forward, right = c2.trajectory_velocity(dict(trajectory), time_s)
    return math.atan2(right, forward) if math.hypot(forward, right) > 1e-9 else 0.0


def _obb_vertices(
    center: tuple[float, float, float],
    location: tuple[float, float, float],
    extent: tuple[float, float, float],
    yaw: float,
) -> list[tuple[float, float, float]]:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    center_x = center[0] + cosine * location[0] - sine * location[1]
    center_y = center[1] + sine * location[0] + cosine * location[1]
    center_z = center[2] + location[2]
    return [
        (
            center_x + cosine * sx * extent[0] - sine * sy * extent[1],
            center_y + sine * sx * extent[0] + cosine * sy * extent[1],
            center_z + sz * extent[2],
        )
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]


def _projected_interval(
    vertices: list[tuple[float, float, float]],
    camera: tuple[float, float, float],
    pitch_degrees: float,
) -> tuple[float, float, float, float] | None:
    pitch = math.radians(pitch_degrees)
    projected: list[tuple[float, float]] = []
    for x_value, y_value, z_value in vertices:
        x_value -= camera[0]
        y_value -= camera[1]
        z_value -= camera[2]
        depth = math.cos(pitch) * x_value + math.sin(pitch) * z_value
        vertical = -math.sin(pitch) * x_value + math.cos(pitch) * z_value
        if depth > 1e-9:
            projected.append((y_value / depth, -vertical / depth))
    if not projected:
        return None
    return (
        min(value[0] for value in projected),
        max(value[0] for value in projected),
        min(value[1] for value in projected),
        max(value[1] for value in projected),
    )


def _contains(
    outer: tuple[float, ...] | None, inner: tuple[float, ...] | None
) -> bool:
    if outer is None or inner is None:
        return False
    return (
        inner[0] >= outer[0] - 1e-12
        and inner[1] <= outer[1] + 1e-12
        and inner[2] >= outer[2] - 1e-12
        and inner[3] <= outer[3] + 1e-12
    )


def _contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            runs.append((start, previous))
            start = index
        previous = index
    runs.append((start, previous))
    return runs


def _xy_clearance(
    wearer: tuple[float, float],
    occluder_center: tuple[float, float, float],
    yaw: float,
    wearer_radius_m: float,
) -> float:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    center_x = occluder_center[0] + cosine * OCCLUDER_BBOX_LOCATION[0] - sine * OCCLUDER_BBOX_LOCATION[1]
    center_y = occluder_center[1] + sine * OCCLUDER_BBOX_LOCATION[0] + cosine * OCCLUDER_BBOX_LOCATION[1]
    dx, dy = wearer[0] - center_x, wearer[1] - center_y
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    outside_x = max(abs(local_x) - OCCLUDER_BBOX_EXTENT[0], 0.0)
    outside_y = max(abs(local_y) - OCCLUDER_BBOX_EXTENT[1], 0.0)
    return math.hypot(outside_x, outside_y) - wearer_radius_m


def validate_static_launch_falsifier(protocol: Mapping[str, Any]) -> dict[str, Any]:
    receipt = protocol["c11_static_launch_falsifier"]
    require(receipt == geometry_receipt(), "c11_static_geometry_receipt_drift")
    trajectory = solid_body_single_pass_trajectory()
    wearer_trajectory = protocol["trajectory_library"]["c8_wearer_route"]
    sample_s = float(protocol["environment"]["sample_seconds"])
    frame_count = int(round(c8.DURATION_SECONDS / sample_s)) + 1
    wearer_radius = float(protocol["route_contract"]["wearer_body_radius_m"])
    pitch = float(protocol["capture"]["wearable_relative_transform"]["pitch_degrees"])
    camera_forward = float(protocol["capture"]["wearable_relative_transform"]["x_m"])
    camera_height = float(protocol["wearer"]["surface_offset_m"]) + float(
        protocol["capture"]["wearable_relative_transform"]["z_m"]
    )
    scenarios = {value["episode_id"]: value for value in protocol["scenarios"]}
    minimum_clearance = math.inf
    observed_runs: dict[str, list[int]] = {}
    for episode_id in c8.EPISODES:
        scenario = scenarios[episode_id]
        layout_id = str(scenario["layout_id"])
        pair_index = int(layout_id[-1])
        target_key = f"c8_l0{pair_index}_target"
        target_trajectory = protocol["trajectory_library"][
            scenario["asset_trajectories"][target_key]
        ]
        target_receipt = TARGET_BBOX_RECEIPTS[layout_id]
        contained_indices: list[int] = []
        for sample_index in range(frame_count):
            time_s = sample_index * sample_s
            wearer_xy = c2.trajectory_position(wearer_trajectory, time_s)
            occluder_xy = c2.trajectory_position(trajectory, time_s)
            target_xy = c2.trajectory_position(target_trajectory, time_s)
            occluder_yaw = _yaw_radians(trajectory, time_s)
            target_yaw = _yaw_radians(target_trajectory, time_s)
            camera = (wearer_xy[0] + camera_forward, wearer_xy[1], camera_height)
            occluder_center = (occluder_xy[0], occluder_xy[1], 0.3)
            target_center = (
                target_xy[0], target_xy[1], float(target_receipt["surface_offset_m"])
            )
            clearance = _xy_clearance(wearer_xy, occluder_center, occluder_yaw, wearer_radius)
            minimum_clearance = min(minimum_clearance, clearance)
            require(
                clearance >= MINIMUM_STATIC_CLEARANCE_M - 1e-9,
                f"c11_static_corridor_clearance:{episode_id}:{sample_index}",
            )
            occluder_interval = _projected_interval(
                _obb_vertices(
                    occluder_center,
                    OCCLUDER_BBOX_LOCATION,
                    OCCLUDER_BBOX_EXTENT,
                    occluder_yaw,
                ),
                camera,
                pitch,
            )
            target_interval = _projected_interval(
                _obb_vertices(
                    target_center,
                    tuple(target_receipt["location"]),
                    tuple(target_receipt["extent"]),
                    target_yaw,
                ),
                camera,
                pitch,
            )
            if _contains(occluder_interval, target_interval):
                require(
                    camera[0] < occluder_center[0] < target_center[0],
                    f"c11_static_depth_order:{episode_id}:{sample_index}",
                )
                contained_indices.append(sample_index)
        runs = _contiguous_runs(contained_indices)
        require(len(runs) == 1, f"c11_static_single_run:{episode_id}")
        require(runs[0] == EXPECTED_CONTAINMENT_RUNS[episode_id], f"c11_static_run:{episode_id}")
        duration = (runs[0][1] - runs[0][0] + 1) * sample_s
        contract = next(
            value for value in protocol["occlusion_contracts"]
            if value["episodes"] == [episode_id]
        )
        require(
            float(contract["minimum_complete_occlusion_seconds"]) - 1e-9
            <= duration
            <= float(contract["maximum_complete_occlusion_seconds"]) + 1e-9,
            f"c11_static_duration:{episode_id}",
        )
        require(runs[0][0] >= int(contract["minimum_pre_track_frames"]), f"c11_static_pre:{episode_id}")
        require(
            frame_count - runs[0][1] - 1 >= int(contract["minimum_post_reappearance_frames"]),
            f"c11_static_post:{episode_id}",
        )
        observed_runs[episode_id] = [runs[0][0], runs[0][1]]
    final_xy = c2.trajectory_position(trajectory, c8.DURATION_SECONDS)
    require(abs(final_xy[0] - OCCLUDER_START_FORWARD_M) <= 1e-12, "c11_final_forward")
    require(abs(final_xy[1] - OCCLUDER_FINAL_RIGHT_M) <= 1e-12, "c11_final_right")
    require(abs(minimum_clearance - 1.0991466000676156) <= 1e-6, "c11_min_clearance_receipt")
    return {
        "containment_runs_inclusive": observed_runs,
        "minimum_xy_clearance_beyond_wearer_radius_m": minimum_clearance,
        "final_right_m": final_xy[1],
        "formal_instance_pixels_remain_authority": True,
    }


def validate_c11(protocol: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    c2.validate_protocol(protocol)
    require(protocol.get("schema_version") == SCHEMA_VERSION, "c11_schema")
    require(protocol.get("cohort_id") == COHORT_ID, "c11_cohort")
    require(int(protocol["capture"]["seed"]) == CAPTURE_SEED, "c11_seed")
    for key in ("environment", "route_contract", "model_contract", "evaluation_contract", "occlusion_contracts", "scenarios", "asset_templates", "wearer"):
        require(protocol[key] == base[key], f"c11_frozen_contract_drift:{key}")
    for name, value in base["trajectory_library"].items():
        if name not in {f"c8_l0{index}_occluder_cross" for index in range(1, 5)}:
            require(protocol["trajectory_library"][name] == value, f"c11_non_occluder_trajectory_drift:{name}")
    for index, layout_id in enumerate(LAYOUT_IDS, start=1):
        base_layout = base["layouts"][layout_id]
        layout = protocol["layouts"][layout_id]
        for key in ("anchor", "witness", "weather", "duration_seconds", "showcase_time_s"):
            require(layout[key] == base_layout[key], f"c11_layout_drift:{layout_id}:{key}")
        base_assets = {value["asset_key"]: value for value in base_layout["assets"]}
        assets = {value["asset_key"]: value for value in layout["assets"]}
        require(tuple(assets) == tuple(base_assets), f"c11_asset_order:{layout_id}")
        occluder_key = f"c8_l0{index}_occluder"
        for asset_key, asset in assets.items():
            expected = deepcopy(base_assets[asset_key])
            if asset_key == occluder_key:
                expected["template"] = OCCLUDER_TEMPLATE
                expected["collisions_enabled"] = False
            require(asset == expected, f"c11_asset_drift:{layout_id}:{asset_key}")
        require(protocol["trajectory_library"][f"c8_l0{index}_occluder_cross"] == solid_body_single_pass_trajectory(), f"c11_occluder_trajectory:{layout_id}")
    source = protocol["source_disjoint_contract"]
    require(source["parent_protocol_sha256"] == PARENT_PROTOCOL_SHA256, "c11_parent_hash")
    require(source["predecessor_c10_disposition"] == PREDECESSOR_DISPOSITION, "c11_predecessor")
    require(source["development_only_never_confirmation"] is True, "c11_not_development_only")
    require(source["predecessor_algorithm_model_scorer_x24_or_x31_outcome_existed"] is False, "c11_c10_outcome_existed")
    require(source["predecessor_algorithm_model_scorer_x24_or_x31_outcome_accessed"] is False, "c11_c10_outcome_accessed")
    require(source["c9_firetruck_instance_jsonl_accessed_for_solid_body_reachability"] is True, "c11_c9_firetruck_source_not_recorded")
    require(source["c9_algorithm_model_evaluator_scorer_x24_or_x31_outcome_existed"] is False, "c11_c9_outcome_existed")
    require(source["c9_algorithm_model_evaluator_scorer_x24_or_x31_outcome_accessed"] is False, "c11_c9_outcome_accessed")
    require(source["one_fresh_capture_only"] is True and source["resume_replay_reseed_substitution_or_partial_selection_allowed"] is False, "c11_one_shot_policy")
    candidate_blueprints = {
        protocol["asset_templates"][asset.get("template", "")]["blueprint_candidates"][0]
        for layout in protocol["layouts"].values()
        for asset in layout["assets"]
        if asset.get("template") in protocol["asset_templates"]
    }
    candidate_blueprints.add(protocol["wearer"]["blueprint_candidates"][0])
    require(len(candidate_blueprints) >= int(protocol["admission"]["minimum_unique_actual_blueprints_across_pack"]), "c11_unique_blueprint_lower_bound")
    require(all(len(layout["assets"]) >= int(protocol["admission"]["minimum_active_assets_per_layout_excluding_wearer"]) for layout in protocol["layouts"].values()), "c11_active_asset_lower_bound")
    static_receipt = validate_static_launch_falsifier(protocol)
    static_receipt["candidate_unique_blueprints"] = len(candidate_blueprints)
    static_receipt["minimum_active_assets_excluding_wearer"] = min(
        len(layout["assets"]) for layout in protocol["layouts"].values()
    )
    return static_receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_path = args.base.resolve(strict=True)
    require(c2.sha256_file(base_path) == PARENT_PROTOCOL_SHA256, "c11_parent_file_hash")
    base = read_json(base_path)
    first = materialize(base)
    second = materialize(base)
    first_bytes = serialized_protocol(first)
    require(first_bytes == serialized_protocol(second), "c11_nondeterministic")
    static_receipt = validate_c11(first, base)
    if args.validate_only:
        require(args.output.exists(), "c11_output_missing")
        require(args.output.read_bytes() == first_bytes, "c11_existing_protocol_drift")
    else:
        if args.output.exists() and args.output.read_bytes() != first_bytes:
            raise FileExistsError(f"existing C11 protocol differs: {args.output}")
        if not args.output.exists():
            args.output.write_bytes(first_bytes)
    print(json.dumps({
        "status": "C11_PROTOCOL_STATIC_VALID",
        "output": str(args.output.resolve()),
        "cohort_id": COHORT_ID,
        "capture_seed": CAPTURE_SEED,
        "episodes": len(c8.EPISODES),
        "layouts": len(LAYOUT_IDS),
        "protocol_sha256": c2.sha256_bytes(first_bytes),
        "opened_source_predictions_evaluator_model_or_scorer": False,
        "predecessor_c10_instance_source_outcome_accessed_for_geometry_design": True,
        "predecessor_c10_algorithm_model_scorer_x24_or_x31_outcome_accessed": False,
        "c9_firetruck_instance_source_accessed_for_solid_body_reachability": True,
        "c9_algorithm_model_evaluator_scorer_x24_or_x31_outcome_accessed": False,
        "development_only_never_confirmation": True,
        "one_fresh_capture_only": True,
        "static_launch_falsifier": static_receipt,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
