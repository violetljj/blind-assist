"""Materialize the C12 opaque-skin successor to terminal C11.

C11 is a consumed SOURCE_NOT_EVALUABLE source attempt.  C12 keeps its exact
firetruck, trajectory, targets, scenarios, model, detector, route, scorer, and
evaluator gates.  The only scene change is a collision-irrelevant, co-rigid
skin of six CARLA-native advertisement panels inside the firetruck proxy in
each layout.  Materialization does not authorize a formal capture: an
instance-only raster probe must first meet the frozen C12 prelaunch gate.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import dtr_carla_c2_rich_scene as c2


HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "dtr_carla_c11_x31_solid_body_protocol.json"
DEFAULT_OUTPUT = HERE / "dtr_carla_c12_x31_opaque_skin_protocol.json"

SCHEMA_VERSION = 12
COHORT_ID = "DTR_CARLA_C12_X31_OPAQUE_SKIN_SOURCE_DISJOINT_V1"
PARENT_COHORT_ID = "DTR_CARLA_C11_X31_SOLID_BODY_SOURCE_DISJOINT_V1"
PARENT_PROTOCOL_SHA256 = (
    "744320C57ABBCD25835155760C15E83A421D66FC5DEDC35FB07F5F30BE0E5ACD"
)
PARENT_PROTOCOL_CANONICAL_SHA256 = (
    "F56DD108911F4371BD1577C8B2A47188F58CABE873F4B32E23C2E867FEEF34D7"
)
PARENT_RUN_ID = "c11-x31-solid-20260830-211827"
PARENT_RESULT_SHA256 = (
    "8B730BA1A0D5BE7BFF87750775FA4193E5BA3BB00F44A6D271C2073AFE499AAE"
)
PARENT_PHYSICAL_REPORT_SHA256 = (
    "64F5361F68A9C142C674711DBB9576E10069FFD95418C51EB59854E803D0C210"
)
CAPTURE_SEED = 132365

LAYOUT_IDS = ("c8_l01", "c8_l02", "c8_l03", "c8_l04")
EPISODE_IDS = tuple(f"ep_{index:02d}" for index in range(1, 9))
PANEL_COUNT_PER_LAYOUT = 6
PRIOR_SCRIPTED_COUNT_PER_LAYOUT = 3
EXPECTED_SCRIPTED_COUNT_PER_LAYOUT = 9
PRIOR_SCRIPTED_COUNT_GLOBAL = 12
EXPECTED_SCRIPTED_COUNT_GLOBAL = 36

PANEL_TEMPLATE = "p_opaque_skin_panel"
PANEL_BLUEPRINT = "static.prop.advertisement"
PANEL_ROLE = "c12_firetruck_opaque_skin_component"
PANEL_SURFACE_OFFSET_M = -0.02
PANEL_START_FORWARD_M = 1.84
PANEL_CENTER_OFFSETS_RIGHT_M = (-3.40, -2.04, -0.68, 0.68, 2.04, 3.40)
PANEL_YAW_OFFSET_BY_LAYOUT = {
    "c8_l01": 90.0,
    "c8_l02": 90.0,
    "c8_l03": -90.0,
    "c8_l04": -90.0,
}
PANEL_TRACK_PREFIX_BY_LAYOUT = {
    "c8_l01": "e",
    "c8_l02": "f",
    "c8_l03": "g",
    "c8_l04": "h",
}

# Frozen C11 firetruck receipt and C11 advertisement evidence.  The panel
# dimensions are nominal geometry only; raster opacity remains probe-owned.
PROXY_BLUEPRINT = "vehicle.carlamotors.firetruck"
PROXY_BBOX_LOCATION = (
    -0.25344201922416687,
    0.005309276282787323,
    1.91335928440094,
)
PROXY_BBOX_EXTENT = (
    4.234020709991455,
    1.4455441236495972,
    1.9137061834335327,
)
PROXY_START_FORWARD_M = 3.2
PROXY_START_RIGHT_M = 8.0
PROXY_CROSS_START_S = 0.9
PROXY_CROSS_STOP_S = 2.0
PROXY_RIGHT_VELOCITY_MPS = -10.0
PANEL_EVIDENCE_BBOX_LOCATION = (0.0, 0.0, 1.2214220762252808)
PANEL_EVIDENCE_BBOX_EXTENT = (
    0.07609862834215164,
    0.7840991020202637,
    1.2214220762252808,
)
PANEL_EVIDENCE_INSTANCE_PIXELS = 46455
PANEL_EVIDENCE_INSTANCE_BBOX_PIXELS = 49496
MINIMUM_EDGE_INSET_M = 0.04
MINIMUM_VERTICAL_MARGIN_M = 0.08
RASTER_PROBE_GATE_STATUS = "DTR_CARLA_C12_INSTANCE_RASTER_PROBE_GATE_MET"

PARENT_DISPOSITION = {
    "cohort_id": PARENT_COHORT_ID,
    "protocol_sha256": PARENT_PROTOCOL_SHA256,
    "protocol_canonical_sha256": PARENT_PROTOCOL_CANONICAL_SHA256,
    "run_id": PARENT_RUN_ID,
    "status": "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_NOT_EVALUABLE",
    "terminal_stage": "JOINED_SOURCE_PHYSICAL_OCCLUSION_ADMISSION",
    "physical_occlusion_contracts_met": 1,
    "physical_occlusion_contracts_required": 8,
    "result_sha256": PARENT_RESULT_SHA256,
    "physical_occlusion_report_sha256": PARENT_PHYSICAL_REPORT_SHA256,
    "instance_source_outcome_accessed_for_c12_representation_design": True,
    "instance_jsonl_accessed_for_c12_representation_design": True,
    "source_model_package_present": True,
    "algorithm_model_run_present": False,
    "detector_run_present": False,
    "x24_or_x31_prediction_present": False,
    "scorer_result_present": False,
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


def panel_key(layout_id: str, ordinal: int) -> str:
    return f"{layout_id}_occluder_skin_{ordinal:02d}"


def panel_track_id(layout_id: str, ordinal: int) -> str:
    return f"{PANEL_TRACK_PREFIX_BY_LAYOUT[layout_id]}_{93 + ordinal:02d}"


def panel_trajectory_name(yaw_offset_degrees: float, ordinal: int) -> str:
    facing = "pos90" if yaw_offset_degrees > 0.0 else "neg90"
    return f"c12_opaque_skin_{facing}_{ordinal:02d}"


def panel_wall_center_right_m() -> float:
    # While the firetruck moves along negative local-right, its local-x bbox
    # center maps to local-right with the opposite sign.
    return -PROXY_BBOX_LOCATION[0]


def panel_center_right_m(ordinal: int) -> float:
    return panel_wall_center_right_m() + PANEL_CENTER_OFFSETS_RIGHT_M[ordinal - 1]


def panel_start_right_m(ordinal: int) -> float:
    return PROXY_START_RIGHT_M + panel_center_right_m(ordinal)


def panel_trajectory(ordinal: int, yaw_offset_degrees: float) -> dict[str, Any]:
    return {
        "start_forward_m": PANEL_START_FORWARD_M,
        "start_right_m": panel_start_right_m(ordinal),
        "yaw_offset_degrees": yaw_offset_degrees,
        "segments": [
            {
                "start_s": 0.0,
                "velocity_forward_mps": 0.0,
                "velocity_right_mps": 0.0,
            },
            {
                "start_s": PROXY_CROSS_START_S,
                "velocity_forward_mps": 0.0,
                "velocity_right_mps": PROXY_RIGHT_VELOCITY_MPS,
            },
            {
                "start_s": PROXY_CROSS_STOP_S,
                "velocity_forward_mps": 0.0,
                "velocity_right_mps": 0.0,
            },
        ],
    }


def panel_asset(layout_id: str, ordinal: int) -> dict[str, Any]:
    yaw = PANEL_YAW_OFFSET_BY_LAYOUT[layout_id]
    return {
        "asset_key": panel_key(layout_id, ordinal),
        "track_id": panel_track_id(layout_id, ordinal),
        "role": PANEL_ROLE,
        "template": PANEL_TEMPLATE,
        "trajectory": panel_trajectory_name(yaw, ordinal),
        "scripted_pose_authority": True,
        "collisions_enabled": False,
    }


def panel_template() -> dict[str, Any]:
    return {
        "kind": "prop",
        "blueprint_candidates": [PANEL_BLUEPRINT],
        "surface_offset_m": PANEL_SURFACE_OFFSET_M,
        "collision_relevant": False,
    }


def geometry_values(base: Mapping[str, Any]) -> dict[str, float | list[float]]:
    proxy_center_right = panel_wall_center_right_m()
    proxy_min_right = proxy_center_right - PROXY_BBOX_EXTENT[0]
    proxy_max_right = proxy_center_right + PROXY_BBOX_EXTENT[0]
    panel_half_width = PANEL_EVIDENCE_BBOX_EXTENT[1]
    panel_min_right = panel_center_right_m(1) - panel_half_width
    panel_max_right = panel_center_right_m(PANEL_COUNT_PER_LAYOUT) + panel_half_width
    proxy_min_forward = (
        PROXY_START_FORWARD_M + PROXY_BBOX_LOCATION[1] - PROXY_BBOX_EXTENT[1]
    )
    proxy_max_forward = (
        PROXY_START_FORWARD_M + PROXY_BBOX_LOCATION[1] + PROXY_BBOX_EXTENT[1]
    )
    panel_min_forward = PANEL_START_FORWARD_M - PANEL_EVIDENCE_BBOX_EXTENT[0]
    panel_max_forward = PANEL_START_FORWARD_M + PANEL_EVIDENCE_BBOX_EXTENT[0]
    panel_bottom = PANEL_SURFACE_OFFSET_M
    panel_top = PANEL_SURFACE_OFFSET_M + 2.0 * PANEL_EVIDENCE_BBOX_EXTENT[2]
    adjacent_panel_overlap = (
        2.0 * PANEL_EVIDENCE_BBOX_EXTENT[1]
        - (PANEL_CENTER_OFFSETS_RIGHT_M[1] - PANEL_CENTER_OFFSETS_RIGHT_M[0])
    )
    target_tops = []
    for value in base["c11_static_launch_falsifier"]["target_bbox_by_layout"].values():
        target_tops.append(
            float(value["surface_offset_m"])
            + float(value["location"][2])
            + float(value["extent"][2])
        )
    highest_target_top = max(target_tops)
    return {
        "proxy_right_span_m": [proxy_min_right, proxy_max_right],
        "panel_right_span_m": [panel_min_right, panel_max_right],
        "left_edge_inset_m": panel_min_right - proxy_min_right,
        "right_edge_inset_m": proxy_max_right - panel_max_right,
        "proxy_forward_span_m": [proxy_min_forward, proxy_max_forward],
        "panel_forward_span_m": [panel_min_forward, panel_max_forward],
        "near_face_inset_m": panel_min_forward - proxy_min_forward,
        "panel_vertical_span_m": [panel_bottom, panel_top],
        "adjacent_panel_overlap_m": adjacent_panel_overlap,
        "highest_frozen_target_top_m": highest_target_top,
        "vertical_top_margin_m": panel_top - highest_target_top,
    }


def design_receipt(base: Mapping[str, Any]) -> dict[str, Any]:
    geometry = geometry_values(base)
    return {
        "schema": "dtr-c12-firetruck-opaque-skin-design-v1",
        "authority": "PRELAUNCH_DESIGN_ONLY_RASTER_PROBE_REQUIRED",
        "parent_run_id": PARENT_RUN_ID,
        "parent_result_sha256": PARENT_RESULT_SHA256,
        "parent_physical_occlusion_report_sha256": PARENT_PHYSICAL_REPORT_SHA256,
        "firetruck_proxy": {
            "blueprint": PROXY_BLUEPRINT,
            "bbox_location": list(PROXY_BBOX_LOCATION),
            "bbox_extent": list(PROXY_BBOX_EXTENT),
            "trajectory_unchanged_from_c11": True,
            "remains_named_occluder": True,
            "remains_collision_relevant": True,
        },
        "panel_asset_evidence": {
            "blueprint": PANEL_BLUEPRINT,
            "bbox_location": list(PANEL_EVIDENCE_BBOX_LOCATION),
            "bbox_extent": list(PANEL_EVIDENCE_BBOX_EXTENT),
            "large_face_width_m": 2.0 * PANEL_EVIDENCE_BBOX_EXTENT[1],
            "large_face_height_m": 2.0 * PANEL_EVIDENCE_BBOX_EXTENT[2],
            "observed_instance_pixels": PANEL_EVIDENCE_INSTANCE_PIXELS,
            "observed_instance_bbox_pixels": PANEL_EVIDENCE_INSTANCE_BBOX_PIXELS,
            "observed_instance_bbox_fill_fraction": (
                PANEL_EVIDENCE_INSTANCE_PIXELS / PANEL_EVIDENCE_INSTANCE_BBOX_PIXELS
            ),
            "evidence_does_not_prove_compound_pose_opacity": True,
        },
        "opaque_skin": {
            "panels_per_layout": PANEL_COUNT_PER_LAYOUT,
            "panel_start_forward_m": PANEL_START_FORWARD_M,
            "panel_wall_center_right_m": panel_wall_center_right_m(),
            "panel_center_offsets_right_m": list(PANEL_CENTER_OFFSETS_RIGHT_M),
            "panel_centers_right_at_proxy_crossing_m": [
                panel_center_right_m(index)
                for index in range(1, PANEL_COUNT_PER_LAYOUT + 1)
            ],
            "panel_start_right_m": [
                panel_start_right_m(index)
                for index in range(1, PANEL_COUNT_PER_LAYOUT + 1)
            ],
            "surface_offset_m": PANEL_SURFACE_OFFSET_M,
            "yaw_offset_degrees_by_layout": deepcopy(PANEL_YAW_OFFSET_BY_LAYOUT),
            "all_panels_scripted_pose_authority": True,
            "all_panels_collisions_enabled": False,
            "all_panels_collision_relevant": False,
            **geometry,
        },
        "required_instance_probe": {
            "evaluator": "evaluate_dtr_carla_c12_instance_probe.py",
            "sensor": "instance",
            "expected_episode_count": 8,
            "complete_occlusion_pixel_fraction": 0.0,
            "minimum_complete_occlusion_frames": 6,
            "maximum_complete_occlusion_frames": 13,
            "minimum_pre_track_frames": 10,
            "minimum_post_reappearance_frames": 8,
            "minimum_trackable_pixel_fraction": 0.0002,
            "line_of_sight_proxy": PROXY_BLUEPRINT,
            "required_gate_status": RASTER_PROBE_GATE_STATUS,
            "probe_is_not_formal_source_or_algorithm_evidence": True,
        },
        "panel_yaw_and_material_orientation_remain_probe_owned": True,
        "formal_launch_authorized_by_materialization_alone": False,
    }


def collision_policy() -> dict[str, Any]:
    return {
        "schema": "dtr-c12-scripted-opaque-skin-collision-policy-v1",
        "scope": "EVERY_LAYOUT_ASSET_WITH_SCRIPTED_POSE_AUTHORITY_TRUE",
        "required_collisions_enabled": False,
        "prior_scripted_asset_count_per_layout": PRIOR_SCRIPTED_COUNT_PER_LAYOUT,
        "expected_scripted_asset_count_per_layout": EXPECTED_SCRIPTED_COUNT_PER_LAYOUT,
        "prior_scripted_asset_count": PRIOR_SCRIPTED_COUNT_GLOBAL,
        "expected_scripted_asset_count": EXPECTED_SCRIPTED_COUNT_GLOBAL,
        "opaque_skin_component_count_per_layout": PANEL_COUNT_PER_LAYOUT,
        "physics_collision_response_only": True,
        "visual_rendering_change_limited_to_opaque_skin_components": True,
        "firetruck_proxy_projected_bbox_unchanged": True,
        "physical_occlusion_evaluation_unchanged": True,
        "evaluator_collision_relevance_unchanged": True,
        "original_asset_template_collision_relevant_fields_unchanged": True,
        "opaque_skin_components_collision_relevant": False,
    }


def materialize(base: Mapping[str, Any]) -> dict[str, Any]:
    require(base.get("cohort_id") == PARENT_COHORT_ID, "c12_unexpected_parent")
    protocol = deepcopy(base)
    protocol["schema_version"] = SCHEMA_VERSION
    protocol["cohort_id"] = COHORT_ID
    protocol["evidence_class"] = (
        "synthetic_fresh_source_disjoint_x31_opaque_skin_development_source"
    )
    protocol["objective"] = (
        "After consumed C11 isolated firetruck raster gaps, require an "
        "instance-only prelaunch probe of one six-panel CARLA-native opaque skin "
        "inside the unchanged firetruck proxy before authorizing one fresh "
        "Development capture."
    )
    protocol["capture"]["seed"] = CAPTURE_SEED

    protocol["asset_templates"][PANEL_TEMPLATE] = panel_template()
    for yaw in (90.0, -90.0):
        for ordinal in range(1, PANEL_COUNT_PER_LAYOUT + 1):
            protocol["trajectory_library"][panel_trajectory_name(yaw, ordinal)] = (
                panel_trajectory(ordinal, yaw)
            )
    changed_assets: list[str] = []
    for layout_id in LAYOUT_IDS:
        additions = [
            panel_asset(layout_id, ordinal)
            for ordinal in range(1, PANEL_COUNT_PER_LAYOUT + 1)
        ]
        protocol["layouts"][layout_id]["assets"].extend(additions)
        changed_assets.extend(value["asset_key"] for value in additions)

    prior_gates = protocol["admission"].pop("c11_solid_body_source_gates")
    prior_gates.pop("static_3d_obb_launch_falsifier_required", None)
    prior_gates.pop("static_falsifier_is_not_pixel_or_outcome_proof", None)
    protocol["admission"]["c12_opaque_skin_source_gates"] = {
        **prior_gates,
        "all_four_named_occluders_remain_exact_c11_firetruck_proxies": True,
        "six_exact_advertisement_skin_components_per_layout": True,
        "skin_components_co_rigid_with_firetruck_pass": True,
        "skin_components_inside_firetruck_proxy_lateral_span": True,
        "skin_components_collision_irrelevant_and_collision_disabled": True,
        "instance_only_raster_prelaunch_probe_required": True,
        "required_instance_probe_status": RASTER_PROBE_GATE_STATUS,
        "formal_launch_authorized_by_materialization_alone": False,
        "probe_pixels_are_not_formal_source_or_algorithm_evidence": True,
        "formal_instance_pixels_remain_occlusion_authority": True,
        "maximum_fresh_capture_attempts": 1,
        "resume_replay_reseed_substitution_or_partial_selection_allowed": False,
    }
    protocol.pop("c11_static_launch_falsifier")
    protocol["c12_opaque_skin_design_receipt"] = design_receipt(base)
    protocol["scripted_pose_collision_policy_contract"] = collision_policy()

    prior_source = base["source_disjoint_contract"]
    protocol["source_disjoint_contract"] = {
        "parent_cohort_id": PARENT_COHORT_ID,
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "parent_protocol_canonical_sha256": PARENT_PROTOCOL_CANONICAL_SHA256,
        "predecessor_c11_disposition": deepcopy(PARENT_DISPOSITION),
        "confirmation_episodes": [],
        "development_episodes": list(EPISODE_IDS),
        "retained_regression_episodes": [],
        "map": prior_source["map"],
        "map_disjoint_from_predecessor": False,
        "same_map_numeric_anchor_witness_weather_and_dense_support_reuse": True,
        "new_capture_seed": CAPTURE_SEED,
        "predecessor_capture_seed": int(base["capture"]["seed"]),
        "weather_by_layout": deepcopy(prior_source["weather_by_layout"]),
        "dynamic_pairs": deepcopy(base["evaluation_contract"]["dynamic_pairs"]),
        "reused_from_c11": [
            "MAP",
            "NUMERIC_ANCHORS",
            "WITNESSES",
            "WEATHER",
            "TARGET_ALIAS_BLUEPRINTS",
            "TARGET_ALIAS_WEARER_TRAJECTORIES",
            "DENSE_SUPPORT_POSES",
            "EPISODE_PAIRING",
            "SCENARIO_EXPECTED_LABELS",
            "OCCLUSION_CONTRACTS",
            "FIRETRUCK_PROXY_AND_SINGLE_PASS_TRAJECTORY",
            "MODEL_DETECTOR_X31_ROUTE_THRESHOLDS",
            "EVALUATION_AND_SCORE_GATES",
        ],
        "structural_source_change": (
            "FIRETRUCK_PROXY_PLUS_SIX_PANEL_CARLA_NATIVE_OPAQUE_SKIN"
        ),
        "unchanged_named_occluder_assets": [
            f"{layout_id}_occluder" for layout_id in LAYOUT_IDS
        ],
        "changed_opaque_skin_assets": changed_assets,
        "fresh_capture_identity": (
            "SEED_132365_ONE_COMPLETE_CAPTURE_ONLY_AFTER_INSTANCE_PROBE_GATE_MET"
        ),
        "predecessor_instance_source_accessed_for_representation_design": True,
        "predecessor_algorithm_model_detector_x24_x31_or_scorer_outcome_existed": False,
        "predecessor_algorithm_model_detector_x24_x31_or_scorer_outcome_accessed": False,
        "development_only_never_confirmation": True,
        "instance_probe_required_before_formal_launch": True,
        "instance_probe_pixels_not_reusable_as_formal_capture": True,
        "formal_launch_authorized_by_materialization_alone": False,
        "one_fresh_capture_only": True,
        "resume_replay_reseed_substitution_or_partial_selection_allowed": False,
    }
    protocol["claim_boundary"] = [
        "C11 is terminal SOURCE_NOT_EVALUABLE and is not resumed, replayed, reseeded, substituted, predicted, scored, or partially selected.",
        "C11 result and instance evidence were accessed only for C12 source-representation design.",
        "No C11 detector, model, X24, X31, or scorer outcome exists or was accessed.",
        "C12 keeps every C11 target, alias, wearer, map, weather, scenario, expected label, named firetruck proxy, proxy trajectory, detector, model, route threshold, evaluator threshold, and scorer gate unchanged.",
        "C12 adds only six collision-irrelevant CARLA-native advertisement skin components per layout.",
        "Materialization and static geometry checks do not authorize a formal capture.",
        "A separate direct instance-only raster probe must meet the frozen eight-episode gate before any formal C12 launch.",
        "Probe pixels are prelaunch reachability evidence only and cannot enter the formal C12 source or algorithm evaluation.",
        "C12 remains synthetic Development only and can never count as confirmation.",
        "No result establishes real-world, product, deployment, user-benefit, reliability, or safety authority.",
    ]
    return protocol


def validate_geometry(base: Mapping[str, Any]) -> dict[str, Any]:
    values = geometry_values(base)
    proxy_right = values["proxy_right_span_m"]
    panel_right = values["panel_right_span_m"]
    proxy_forward = values["proxy_forward_span_m"]
    panel_forward = values["panel_forward_span_m"]
    require(isinstance(proxy_right, list) and isinstance(panel_right, list), "c12_right_span_type")
    require(isinstance(proxy_forward, list) and isinstance(panel_forward, list), "c12_forward_span_type")
    require(panel_right[0] > proxy_right[0], "c12_panel_left_outside_proxy")
    require(panel_right[1] < proxy_right[1], "c12_panel_right_outside_proxy")
    require(float(values["left_edge_inset_m"]) >= MINIMUM_EDGE_INSET_M, "c12_left_inset")
    require(float(values["right_edge_inset_m"]) >= MINIMUM_EDGE_INSET_M, "c12_right_inset")
    require(panel_forward[0] > proxy_forward[0], "c12_panel_before_proxy_near_face")
    require(panel_forward[1] < proxy_forward[1], "c12_panel_after_proxy_far_face")
    require(float(values["near_face_inset_m"]) > 0.0, "c12_near_face_inset")
    require(float(values["adjacent_panel_overlap_m"]) > 0.2, "c12_panel_overlap")
    require(float(values["vertical_top_margin_m"]) >= MINIMUM_VERTICAL_MARGIN_M, "c12_vertical_margin")
    return values


def validate_c12(protocol: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    c2.validate_protocol(dict(protocol))
    require(protocol.get("schema_version") == SCHEMA_VERSION, "c12_schema")
    require(protocol.get("cohort_id") == COHORT_ID, "c12_cohort")
    require(int(protocol["capture"]["seed"]) == CAPTURE_SEED, "c12_seed")
    require(set(protocol["layouts"]) == set(base["layouts"]), "c12_layout_set")

    for key in (
        "experiment_id",
        "environment",
        "route_contract",
        "occlusion_contracts",
        "wearer",
        "scenarios",
        "twin_contracts",
        "model_contract",
        "evaluation_contract",
        "capture_compatibility_contract",
    ):
        require(protocol[key] == base[key], f"c12_frozen_contract_drift:{key}")
    expected_capture = deepcopy(base["capture"])
    expected_capture["seed"] = CAPTURE_SEED
    require(protocol["capture"] == expected_capture, "c12_capture_drift")
    require(
        set(protocol["admission"])
        == {*base["admission"], "c12_opaque_skin_source_gates"}
        - {"c11_solid_body_source_gates"},
        "c12_admission_field_set",
    )
    for key, value in base["admission"].items():
        if key != "c11_solid_body_source_gates":
            require(protocol["admission"][key] == value, f"c12_admission_drift:{key}")

    require(
        set(protocol["asset_templates"]) == {*base["asset_templates"], PANEL_TEMPLATE},
        "c12_template_set",
    )
    for key, value in base["asset_templates"].items():
        require(protocol["asset_templates"][key] == value, f"c12_template_drift:{key}")
    require(protocol["asset_templates"][PANEL_TEMPLATE] == panel_template(), "c12_panel_template")

    expected_extra_trajectories = {
        panel_trajectory_name(yaw, ordinal)
        for yaw in (90.0, -90.0)
        for ordinal in range(1, PANEL_COUNT_PER_LAYOUT + 1)
    }
    require(
        set(protocol["trajectory_library"]) == {
            *base["trajectory_library"],
            *expected_extra_trajectories,
        },
        "c12_trajectory_set",
    )
    for key, value in base["trajectory_library"].items():
        require(protocol["trajectory_library"][key] == value, f"c12_trajectory_drift:{key}")
    for yaw in (90.0, -90.0):
        for ordinal in range(1, PANEL_COUNT_PER_LAYOUT + 1):
            name = panel_trajectory_name(yaw, ordinal)
            require(
                protocol["trajectory_library"][name] == panel_trajectory(ordinal, yaw),
                f"c12_panel_trajectory:{name}",
            )

    all_panel_keys: list[str] = []
    scripted_global = 0
    for layout_id in LAYOUT_IDS:
        base_layout = base["layouts"][layout_id]
        layout = protocol["layouts"][layout_id]
        for key in ("anchor", "witness", "weather", "duration_seconds", "showcase_time_s"):
            require(layout[key] == base_layout[key], f"c12_layout_drift:{layout_id}:{key}")
        base_assets = list(base_layout["assets"])
        require(
            layout["assets"][: len(base_assets)] == base_assets,
            f"c12_original_asset_drift:{layout_id}",
        )
        additions = layout["assets"][len(base_assets) :]
        expected_additions = [
            panel_asset(layout_id, ordinal)
            for ordinal in range(1, PANEL_COUNT_PER_LAYOUT + 1)
        ]
        require(additions == expected_additions, f"c12_panel_assets:{layout_id}")
        all_panel_keys.extend(value["asset_key"] for value in additions)
        materialized = c2.materialize_layout_assets(dict(protocol), layout_id)
        scripted = [value for value in materialized if value.get("scripted_pose_authority") is True]
        require(len(scripted) == EXPECTED_SCRIPTED_COUNT_PER_LAYOUT, f"c12_scripted_count:{layout_id}")
        require(all(value.get("collisions_enabled") is False for value in scripted), f"c12_scripted_collision:{layout_id}")
        for value in materialized[-PANEL_COUNT_PER_LAYOUT:]:
            require(value["blueprint_candidates"] == [PANEL_BLUEPRINT], f"c12_panel_blueprint:{layout_id}")
            require(value["collision_relevant"] is False, f"c12_panel_collision_relevance:{layout_id}")
        scripted_global += len(scripted)
    require(scripted_global == EXPECTED_SCRIPTED_COUNT_GLOBAL, "c12_scripted_count_global")
    require(len(all_panel_keys) == 24 and len(set(all_panel_keys)) == 24, "c12_panel_key_count")

    require(
        protocol["scripted_pose_collision_policy_contract"] == collision_policy(),
        "c12_collision_policy",
    )
    source = protocol["source_disjoint_contract"]
    require(source["parent_protocol_sha256"] == PARENT_PROTOCOL_SHA256, "c12_parent_hash")
    require(source["parent_protocol_canonical_sha256"] == PARENT_PROTOCOL_CANONICAL_SHA256, "c12_parent_canonical_hash")
    require(source["predecessor_c11_disposition"] == PARENT_DISPOSITION, "c12_parent_disposition")
    require(source["changed_opaque_skin_assets"] == all_panel_keys, "c12_source_panel_keys")
    require(source["formal_launch_authorized_by_materialization_alone"] is False, "c12_materializer_authorized_launch")
    require(source["one_fresh_capture_only"] is True, "c12_one_shot")
    require(source["resume_replay_reseed_substitution_or_partial_selection_allowed"] is False, "c12_resume_policy")

    require(
        protocol["c12_opaque_skin_design_receipt"] == design_receipt(base),
        "c12_design_receipt_drift",
    )
    gates = protocol["admission"]["c12_opaque_skin_source_gates"]
    require(gates["instance_only_raster_prelaunch_probe_required"] is True, "c12_probe_not_required")
    require(gates["required_instance_probe_status"] == RASTER_PROBE_GATE_STATUS, "c12_probe_status")
    require(gates["formal_launch_authorized_by_materialization_alone"] is False, "c12_gate_authorized_launch")
    geometry = validate_geometry(base)
    return {
        "panel_count_per_layout": PANEL_COUNT_PER_LAYOUT,
        "scripted_asset_count_per_layout": EXPECTED_SCRIPTED_COUNT_PER_LAYOUT,
        "scripted_asset_count_global": scripted_global,
        "panel_keys": all_panel_keys,
        "geometry": geometry,
        "required_instance_probe_status": RASTER_PROBE_GATE_STATUS,
        "panel_yaw_and_material_orientation_remain_probe_owned": True,
        "formal_launch_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_path = args.base.resolve(strict=True)
    require(c2.sha256_file(base_path) == PARENT_PROTOCOL_SHA256, "c12_parent_file_hash")
    base = read_json(base_path)
    require(c2.sha256_json(base) == PARENT_PROTOCOL_CANONICAL_SHA256, "c12_parent_canonical_hash")
    first = materialize(base)
    second = materialize(base)
    first_bytes = serialized_protocol(first)
    require(first_bytes == serialized_protocol(second), "c12_nondeterministic")
    validation = validate_c12(first, base)
    if args.validate_only:
        require(args.output.exists(), "c12_output_missing")
        require(args.output.read_bytes() == first_bytes, "c12_existing_protocol_drift")
    else:
        if args.output.exists() and args.output.read_bytes() != first_bytes:
            raise FileExistsError(f"existing C12 protocol differs: {args.output}")
        if not args.output.exists():
            args.output.write_bytes(first_bytes)
    print(
        json.dumps(
            {
                "status": "C12_PROTOCOL_STATIC_VALID_PROBE_REQUIRED",
                "output": str(args.output.resolve()),
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "protocol_sha256": c2.sha256_bytes(first_bytes),
                "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
                "parent_protocol_canonical_sha256": PARENT_PROTOCOL_CANONICAL_SHA256,
                "opened_formal_source_detector_model_x24_x31_or_scorer": False,
                "started_carla": False,
                "validation": validation,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
