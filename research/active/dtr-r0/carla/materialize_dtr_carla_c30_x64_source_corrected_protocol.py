"""Materialize source-corrected C30 confirmation for frozen X64."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c17_x42_counterfactual_protocol as c17
import materialize_dtr_carla_c28_x59_mixed_lighting_protocol as c28
import materialize_dtr_carla_c29_x64_sensor_topology_protocol as c29


COHORT_ID = "DTR_CARLA_C30_X64_SOURCE_CORRECTED_CONFIRMATION_V1"
CAPTURE_SEED = 301064
WEATHERS = {
    "c8_l01": "ClearNoon",
    "c8_l02": "DustStorm",
    "c8_l03": "MidRainyNoon",
    "c8_l04": "WetNight",
}
C29_PROTOCOL_SHA256 = (
    "32B94DB44397675533B98E9A4555D3EE944AEA3AF8555A1C252E0232E5EE036C"
)
C29_SOURCE_RESULT_SHA256 = (
    "B5208624C2C1A3B973758110584E458D85492D16E2C9BB3214AD5C91A08465F1"
)
COMPONENT_SHA256 = dict(c29.COMPONENT_SHA256)
CAPTURE_SCRIPT_SHA256 = c29.CAPTURE_SCRIPT_SHA256
JOIN_SCRIPT_SHA256 = c29.JOIN_SCRIPT_SHA256
X64_SHA256 = c29.X64_SHA256


def _restore_source_reachability(protocol: dict, base: dict) -> dict[str, str]:
    parent = c28.materialize(base)
    topology = {
        "c8_l01": "C29_STRICT_LONGITUDINAL_RETAINED",
        "c8_l02": "PARENT_LONGITUDINAL_DOMINANT_RESTORED",
        "c8_l03": "PARENT_CROSS_ROUTE_COMPONENT_RESTORED",
        "c8_l04": "C29_AMPLIFIED_CROSS_ROUTE_RETAINED",
    }
    for layout_id in ("c8_l02", "c8_l03"):
        for suffix in ("target_contact", "target_safe"):
            key = f"{layout_id}_{suffix}"
            protocol["trajectory_library"][key] = parent["trajectory_library"][key]
    return topology


def materialize(base: dict) -> dict:
    protocol = c29.materialize(base)
    protocol["schema_version"] = 30
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Confirm frozen X64 once on fresh 100-degree camera pixels after the "
        "two C29 source-reachability defects are corrected without algorithm access."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather
    motion_topology = _restore_source_reachability(protocol, base)

    old_prereg = protocol.pop("c29_x64_preregistration")
    protocol["c30_x64_preregistration"] = {
        **old_prereg,
        "schema": "dtr-carla-c30-x64-preregistration-v1",
        "single_x64_scored_invocation": True,
        "no_post_capture_algorithm_threshold_or_scenario_tuning": True,
    }
    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c30-x64-source-corrected-contract-v1",
        "failed_parent_cohort_id": c29.COHORT_ID,
        "failed_parent_protocol_sha256": C29_PROTOCOL_SHA256,
        "failed_parent_source_result_sha256": C29_SOURCE_RESULT_SHA256,
        "failed_parent_status": "SOURCE_NOT_EVALUABLE_6_OF_8_OCCLUSION_CONTRACTS",
        "failed_parent_algorithm_predictions_created": False,
        "source_only_corrections": {
            "ep03_post_reappearance_restored_by_parent_l02_trajectory": True,
            "ep05_pretrack_and_reappearance_restored_by_parent_l03_trajectory": True,
            "c29_l01_strict_longitudinal_topology_retained": True,
            "c29_l04_amplified_cross_route_topology_retained": True,
            "camera_fov_and_wearable_transform_unchanged_from_c29": True,
        },
        "new_capture_seed": CAPTURE_SEED,
        "fresh_pixels": True,
        "changed_weather_by_layout": WEATHERS,
        "target_motion_topology": motion_topology,
        "model_sensor_replay_projection": "ACTUAL_ALL_ACTORS",
        "critical_contact_outcome_truth_must_match_across_all_sensors": True,
        "prior_pixels_reused": False,
        "future_truth_visible_to_predictor": False,
        "single_scored_invocation": True,
        "capture_retry_policy": (
            "ONE_FAILED_SERVER_SHARD_WITH_ZERO_DURABLE_FRAMES_MAY_BE_RETRIED_"
            "ONCE_AFTER_ATTEMPT_LOGS_ARE_PRESERVED"
        ),
        "capture_in_doubt_policy": (
            "ANY_NONZERO_PARTIAL_SHARD_IS_SOURCE_NOT_EVALUABLE_AND_MUST_NOT_RETRY"
        ),
        "completed_shard_rerun_allowed": False,
    }
    protocol["claim_boundary"] = [
        "C29 is terminal SOURCE_NOT_EVALUABLE and is never rerun or scored.",
        "C30 changes only source reachability for the two failed C29 layouts plus seed and weather assignment; X64 remains byte-identical.",
        "C30 retains the C29 100-degree camera, lowered wearable pitch, strict-longitudinal l01, and amplified-cross-route l04 domains.",
        "X64 and every imported algorithm component are byte-frozen before C30 capture and consume no weather or lighting label.",
        "A C30 gate confirms X64 only as source-disjoint synthetic Development when both synchronized conflict handback and unanchored crossing release are exercised.",
        "C30 cannot establish unseen-map, open-world traffic, real-sensor, deployment, reliability, user-benefit, or safety authority.",
        "A failed C30 source gate is NOT_EVALUABLE; frozen X64 cannot be changed or rerun on C30 after pixels are opened.",
    ]
    c2.validate_protocol(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    frozen_files = {
        here / "capture_dtr_carla_c2_rich_scene.py": CAPTURE_SCRIPT_SHA256,
        here / "join_dtr_carla_c2_rich_scene.py": JOIN_SCRIPT_SHA256,
        here / "dtr_carla_c29_x64_sensor_topology_protocol.json": (
            C29_PROTOCOL_SHA256
        ),
        **{here / name: digest for name, digest in COMPONENT_SHA256.items()},
    }
    for path, expected in frozen_files.items():
        if c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C30 component drift: {path.name}")
    base = c17._read_json(c17.BASE_PROTOCOL)
    protocol = materialize(base)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "C30_X64_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "output": str(output),
                "sha256": c2.sha256_file(output),
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "weather_domains": WEATHERS,
                "fov_degrees": protocol["capture"]["fov_degrees"],
                "x64_sha256": X64_SHA256,
                "source_corrected_parent": c29.COHORT_ID,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
