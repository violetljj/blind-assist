"""Materialize the one-shot C9 collision-decoupled successor to C8.

C8 produced no result, model contract, evaluator, or outcome: its instance
shard stopped at the authoritative scripted-pose gate.  C9 therefore preserves
the frozen visual cohort and every model/route/evaluation threshold while
changing capture mechanics only: each scripted-pose-authority layout actor has
physics collision response explicitly disabled.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c8_x31_transport_cone_protocol as c8


HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "dtr_carla_c8_x31_transport_cone_protocol.json"
DEFAULT_OUTPUT = HERE / "dtr_carla_c9_x31_collision_decoupled_protocol.json"

SCHEMA_VERSION = 9
COHORT_ID = "DTR_CARLA_C9_X31_COLLISION_DECOUPLED_SOURCE_DISJOINT_V1"
PARENT_COHORT_ID = c8.COHORT_ID
PARENT_PROTOCOL_SHA256 = (
    "180A9FF243FB6D07999AD936481F77429BE2759B0A9038D13E6B9821BC35ACF0"
)
PARENT_PROTOCOL_CANONICAL_SHA256 = (
    "AB738CA81CEC5D188D72A5A2254B475C71B83BB0108461F90C083586E5547724"
)
CAPTURE_SEED = 130364
COLLISION_POLICY_SCHEMA = "dtr-c9-scripted-pose-collision-decoupling-v1"
PREDECESSOR_DISPOSITION = {
    "cohort_id": PARENT_COHORT_ID,
    "protocol_sha256": PARENT_PROTOCOL_SHA256,
    "status": "SOURCE_NOT_EVALUABLE",
    "terminal_stage": "INSTANCE_SHARD_EP_01_AUTHORITATIVE_POSE_GATE",
    "partial_instance_payload_frames_before_failure": 43,
    "failed_scripted_asset": "c8_l01_alias",
    "observed_roll_degrees": -2.048,
    "observed_planar_residual_m": 0.0321,
    "result_present": False,
    "model_present": False,
    "evaluator_present": False,
    "outcome_or_evaluator_accessed_for_c9_design": False,
    "resume_replay_or_delete_authorized": False,
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


def materialize(base: Mapping[str, Any]) -> dict[str, Any]:
    require(base.get("cohort_id") == PARENT_COHORT_ID, "c9_unexpected_parent")
    protocol = deepcopy(base)
    protocol["schema_version"] = SCHEMA_VERSION
    protocol["cohort_id"] = COHORT_ID
    protocol["evidence_class"] = (
        "synthetic_fresh_source_disjoint_x31_collision_decoupled_source"
    )
    protocol["objective"] = (
        "Run one fresh capture of the frozen C8 X31 transport-cone cohort with "
        "scripted-pose actors mechanically decoupled from collision response, "
        "without changing visual geometry, trajectories, truth relevance, or "
        "any model, route, admission, or score threshold."
    )
    protocol["capture"]["seed"] = CAPTURE_SEED

    scripted_assets = 0
    for layout in protocol["layouts"].values():
        layout_count = 0
        for asset in layout["assets"]:
            if bool(asset.get("scripted_pose_authority", False)):
                asset["collisions_enabled"] = False
                layout_count += 1
        scripted_assets += layout_count
        layout["scripted_pose_collision_policy"] = {
            "schema": COLLISION_POLICY_SCHEMA,
            "scripted_pose_assets": layout_count,
            "collisions_enabled": False,
            "visual_presence_unchanged": True,
            "bbox_and_occlusion_presence_unchanged": True,
            "evaluator_collision_relevance_unchanged": True,
        }

    admission = protocol["admission"]
    gates = admission.pop("c8_transport_cone_source_gates")
    admission["c9_collision_decoupled_source_gates"] = {
        **gates,
        "all_scripted_pose_assets_explicitly_collision_disabled": True,
        "visual_bbox_occlusion_and_evaluator_relevance_unchanged": True,
        "maximum_fresh_capture_attempts": 1,
        "resume_replay_reseed_substitution_or_partial_selection_allowed": False,
    }
    protocol["scripted_pose_collision_policy_contract"] = {
        "schema": COLLISION_POLICY_SCHEMA,
        "scope": "EVERY_LAYOUT_ASSET_WITH_SCRIPTED_POSE_AUTHORITY_TRUE",
        "required_collisions_enabled": False,
        "expected_scripted_asset_count": scripted_assets,
        "physics_collision_response_only": True,
        "visual_rendering_unchanged": True,
        "projected_bbox_unchanged": True,
        "physical_occlusion_evaluation_unchanged": True,
        "evaluator_collision_relevance_unchanged": True,
        "asset_template_collision_relevant_fields_unchanged": True,
    }

    prior_source = base["source_disjoint_contract"]
    protocol["source_disjoint_contract"] = {
        "parent_cohort_id": PARENT_COHORT_ID,
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "parent_protocol_canonical_sha256": PARENT_PROTOCOL_CANONICAL_SHA256,
        "predecessor_c8_disposition": deepcopy(PREDECESSOR_DISPOSITION),
        "confirmation_episodes": list(c8.EPISODES),
        "retained_regression_episodes": [],
        "map": c8.MAP_ID,
        "map_disjoint_from_predecessor": False,
        "same_map_numeric_anchor_and_witness_reuse": True,
        "new_capture_seed": CAPTURE_SEED,
        "predecessor_capture_seed": c8.CAPTURE_SEED,
        "weather_by_layout": deepcopy(c8.WEATHER_BY_LAYOUT),
        "dynamic_pairs": deepcopy(base["evaluation_contract"]["dynamic_pairs"]),
        "new_dynamic_target_blueprints": deepcopy(
            prior_source["new_dynamic_target_blueprints"]
        ),
        "reused_from_c8": [
            "MAP",
            "NUMERIC_ANCHORS",
            "WITNESSES",
            "WEATHER",
            "TARGET_ALIAS_OCCLUDER_BLUEPRINTS",
            "ALL_TRAJECTORIES",
            "DENSE_SUPPORT_POSES",
            "EPISODE_PAIRING",
            "OCCLUSION_CONTRACTS",
            "MODEL_AND_ROUTE_THRESHOLDS",
            "EVALUATION_GATES",
        ],
        "structural_source_change": (
            "SCRIPTED_POSE_AUTHORITY_ACTORS_EXPLICITLY_COLLISION_DISABLED"
        ),
        "fresh_capture_identity": "SEED_130364_ONE_COMPLETE_CAPTURE_ONLY",
        "predecessor_source_failure_diagnostic_referenced": True,
        "predecessor_outcome_or_evaluator_referenced": False,
        "prior_outcomes_or_evidence_referenced": False,
        "one_fresh_capture_only": True,
        "resume_replay_reseed_substitution_or_partial_selection_allowed": False,
    }
    protocol["capture_compatibility_contract"][
        "scripted_asset_collisions_enabled_flag_supported"
    ] = True
    protocol["capture_compatibility_contract"][
        "capture_ready_with_current_generic_runner"
    ] = True
    protocol["claim_boundary"] = [
        "C8 is a terminal SOURCE_NOT_EVALUABLE predecessor: it has no result, model, evaluator, or outcome authority and is not resumed, replayed, or deleted.",
        "C9 uses one fresh seed-130364 capture only; failure remains whole-cohort NOT_EVALUABLE with no reseed, resume, replay, substitution, or partial selection.",
        "C9 preserves the C8 map, anchors, witnesses, weather, targets, aliases, occluders, trajectories, dense-support poses, episode pairing, model contract, route thresholds, and evaluation gates.",
        "Every layout asset with scripted_pose_authority=true explicitly sets collisions_enabled=false; this changes physics collision response only.",
        "Visual rendering, projected bbox, physical-occlusion evaluation, evaluator collision relevance, and asset-template collision_relevant fields remain unchanged.",
        "C9 design uses the C8 source failure diagnostic only; no C8 outcome, evaluator, result, model, or partially captured payload enters prediction features or the C9 denominator.",
        "The model sees only dense wearable RGB, metric depth, calibration, current camera/wearer pose, and immutable issued-plan receipts; actor identity and outcome remain evaluator-only.",
        "X31 and the C9 scorer are hash-frozen before capture and run once without threshold, detector, tracker, route, lifecycle, seed, or cohort sweeps.",
        "No result establishes real-world, product, deployment, user-benefit, reliability, or safety authority.",
    ]
    return protocol


def validate_c9(protocol: Mapping[str, Any], base: Mapping[str, Any]) -> None:
    c2.validate_protocol(protocol)
    require(protocol.get("schema_version") == SCHEMA_VERSION, "c9_schema")
    require(protocol.get("cohort_id") == COHORT_ID, "c9_cohort")
    require(int(protocol["capture"]["seed"]) == CAPTURE_SEED, "c9_seed")
    require(protocol["environment"] == base["environment"], "c9_environment_drift")
    require(protocol["route_contract"] == base["route_contract"], "c9_route_drift")
    require(protocol["model_contract"] == base["model_contract"], "c9_model_drift")
    require(protocol["trajectory_library"] == base["trajectory_library"], "c9_trajectory_drift")
    require(protocol["evaluation_contract"] == base["evaluation_contract"], "c9_evaluation_drift")
    require(protocol["occlusion_contracts"] == base["occlusion_contracts"], "c9_occlusion_drift")
    require(protocol["scenarios"] == base["scenarios"], "c9_scenario_drift")
    require(protocol["asset_templates"] == base["asset_templates"], "c9_template_drift")
    require(
        {key: value["weather"] for key, value in protocol["layouts"].items()}
        == c8.WEATHER_BY_LAYOUT,
        "c9_weather_drift",
    )

    scripted_count = 0
    for layout_id, layout in protocol["layouts"].items():
        base_layout = base["layouts"][layout_id]
        require(layout["anchor"] == base_layout["anchor"], f"c9_anchor_drift:{layout_id}")
        require(layout["witness"] == base_layout["witness"], f"c9_witness_drift:{layout_id}")
        base_assets = {value["asset_key"]: value for value in base_layout["assets"]}
        assets = {value["asset_key"]: value for value in layout["assets"]}
        require(tuple(assets) == tuple(base_assets), f"c9_asset_order:{layout_id}")
        layout_scripted = 0
        for asset_key, asset in assets.items():
            expected = deepcopy(base_assets[asset_key])
            if bool(expected.get("scripted_pose_authority", False)):
                expected["collisions_enabled"] = False
                layout_scripted += 1
                scripted_count += 1
            require(asset == expected, f"c9_asset_drift:{layout_id}:{asset_key}")
        policy = layout["scripted_pose_collision_policy"]
        require(policy["scripted_pose_assets"] == layout_scripted, f"c9_layout_policy_count:{layout_id}")
        require(policy["collisions_enabled"] is False, f"c9_layout_collision_policy:{layout_id}")
    policy = protocol["scripted_pose_collision_policy_contract"]
    require(scripted_count > 0, "c9_no_scripted_assets")
    require(policy["expected_scripted_asset_count"] == scripted_count, "c9_scripted_count")
    require(policy["required_collisions_enabled"] is False, "c9_collision_policy")
    require(
        policy["visual_rendering_unchanged"] is True
        and policy["projected_bbox_unchanged"] is True
        and policy["physical_occlusion_evaluation_unchanged"] is True
        and policy["evaluator_collision_relevance_unchanged"] is True
        and policy["asset_template_collision_relevant_fields_unchanged"] is True,
        "c9_visual_or_evaluator_semantics_drift",
    )
    source = protocol["source_disjoint_contract"]
    require(source["parent_protocol_sha256"] == PARENT_PROTOCOL_SHA256, "c9_parent_raw_hash")
    require(source["parent_protocol_canonical_sha256"] == PARENT_PROTOCOL_CANONICAL_SHA256, "c9_parent_canonical_hash")
    require(source["predecessor_c8_disposition"] == PREDECESSOR_DISPOSITION, "c9_predecessor_disposition")
    require(source["predecessor_outcome_or_evaluator_referenced"] is False, "c9_predecessor_truth_access")
    require(source["one_fresh_capture_only"] is True, "c9_not_one_shot")
    require(source["resume_replay_reseed_substitution_or_partial_selection_allowed"] is False, "c9_retry_policy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_path = args.base.resolve(strict=True)
    require(c2.sha256_file(base_path) == PARENT_PROTOCOL_SHA256, "c9_parent_file_hash")
    base = read_json(base_path)
    first = materialize(base)
    second = materialize(base)
    first_bytes = serialized_protocol(first)
    require(first_bytes == serialized_protocol(second), "c9_nondeterministic")
    validate_c9(first, base)
    if args.validate_only:
        require(args.output.exists(), "c9_output_missing")
        require(args.output.read_bytes() == first_bytes, "c9_existing_protocol_drift")
    else:
        if args.output.exists() and args.output.read_bytes() != first_bytes:
            raise FileExistsError(f"existing C9 protocol differs: {args.output}")
        if not args.output.exists():
            args.output.write_bytes(first_bytes)
    frames_per_sensor = len(c8.EPISODES) * (
        int(round(c8.DURATION_SECONDS / c8.SAMPLE_SECONDS)) + 1
    )
    print(
        json.dumps(
            {
                "status": "C9_PROTOCOL_VALID",
                "output": str(args.output.resolve()),
                "cohort_id": COHORT_ID,
                "episodes": len(c8.EPISODES),
                "layouts": len(c8.PAIR_SPECS),
                "frames_per_sensor": frames_per_sensor,
                "sensor_payload_frames": frames_per_sensor
                * len(first["capture"]["sensor_order"]),
                "scripted_pose_assets_collision_disabled": first[
                    "scripted_pose_collision_policy_contract"
                ]["expected_scripted_asset_count"],
                "one_fresh_capture_only": True,
                "protocol_sha256": c2.sha256_bytes(first_bytes),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
