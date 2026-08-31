"""Materialize camera-corrected C31 confirmation for frozen X64."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c17_x42_counterfactual_protocol as c17
import materialize_dtr_carla_c28_x59_mixed_lighting_protocol as c28
import materialize_dtr_carla_c30_x64_source_corrected_protocol as c30


COHORT_ID = "DTR_CARLA_C31_X64_CAMERA_CORRECTED_CONFIRMATION_V1"
CAPTURE_SEED = 311064
WEATHERS = {
    "c8_l01": "CloudyNight",
    "c8_l02": "WetCloudySunset",
    "c8_l03": "SoftRainNoon",
    "c8_l04": "HardRainNight",
}
C30_PROTOCOL_SHA256 = (
    "E98FE61CE4E5E602CF498EF85249C51FFAA0D673E84A94CABA2E7626EE5138A5"
)
C30_SOURCE_RESULT_SHA256 = (
    "344B8443F63918979937187A1AC6DF4B7004E90FA740793C4D385975F7EE967B"
)
COMPONENT_SHA256 = dict(c30.COMPONENT_SHA256)
CAPTURE_SCRIPT_SHA256 = c30.CAPTURE_SCRIPT_SHA256
JOIN_SCRIPT_SHA256 = c30.JOIN_SCRIPT_SHA256
X64_SHA256 = c30.X64_SHA256


def materialize(base: dict) -> dict:
    protocol = c30.materialize(base)
    validated_camera = c28.materialize(base)["capture"]
    protocol["schema_version"] = 31
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["capture"]["fov_degrees"] = validated_camera["fov_degrees"]
    protocol["capture"]["camera_calibration"] = copy.deepcopy(
        validated_camera["camera_calibration"]
    )
    protocol["capture"]["wearable_relative_transform"] = copy.deepcopy(
        validated_camera["wearable_relative_transform"]
    )
    protocol["objective"] = (
        "Confirm frozen X64 once on fresh pixels after restoring the validated "
        "camera geometry while retaining C30 motion-topology changes."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather

    old_prereg = protocol.pop("c30_x64_preregistration")
    protocol["c31_x64_preregistration"] = {
        **old_prereg,
        "schema": "dtr-carla-c31-x64-preregistration-v1",
        "single_x64_scored_invocation": True,
        "no_post_capture_algorithm_threshold_or_scenario_tuning": True,
    }
    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c31-x64-camera-corrected-contract-v1",
        "failed_parent_cohort_id": c30.COHORT_ID,
        "failed_parent_protocol_sha256": C30_PROTOCOL_SHA256,
        "failed_parent_source_result_sha256": C30_SOURCE_RESULT_SHA256,
        "failed_parent_status": "SOURCE_NOT_EVALUABLE_6_OF_8_OCCLUSION_CONTRACTS",
        "failed_parent_algorithm_predictions_created": False,
        "source_only_correction": (
            "RESTORE_C28_VALIDATED_90_DEGREE_CAMERA_AND_WEARABLE_TRANSFORM"
        ),
        "c30_motion_topology_retained": {
            "c8_l01": "STRICT_LONGITUDINAL",
            "c8_l02": "PARENT_LONGITUDINAL_DOMINANT",
            "c8_l03": "PARENT_CROSS_ROUTE_COMPONENT",
            "c8_l04": "AMPLIFIED_CROSS_ROUTE",
        },
        "new_capture_seed": CAPTURE_SEED,
        "fresh_pixels": True,
        "changed_weather_by_layout": WEATHERS,
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
        "C29 and C30 are terminal SOURCE_NOT_EVALUABLE and are never rerun or scored.",
        "C31 restores only the validated camera geometry and changes seed plus weather assignment; X64 remains byte-identical.",
        "C31 retains the new strict-longitudinal l01 and amplified-cross-route l04 motion topology.",
        "X64 and every imported algorithm component are byte-frozen before C31 capture and consume no weather or lighting label.",
        "A C31 gate confirms X64 only as source-disjoint synthetic Development when both synchronized conflict handback and unanchored crossing release are exercised.",
        "C31 cannot establish unseen-map, open-world traffic, real-sensor, deployment, reliability, user-benefit, or safety authority.",
        "A failed C31 source gate is NOT_EVALUABLE; frozen X64 cannot be changed or rerun on C31 after pixels are opened.",
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
        here / "dtr_carla_c30_x64_source_corrected_protocol.json": (
            C30_PROTOCOL_SHA256
        ),
        **{here / name: digest for name, digest in COMPONENT_SHA256.items()},
    }
    for path, expected in frozen_files.items():
        if c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C31 component drift: {path.name}")
    protocol = materialize(c17._read_json(c17.BASE_PROTOCOL))
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
                "status": "C31_X64_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "output": str(output),
                "sha256": c2.sha256_file(output),
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "weather_domains": WEATHERS,
                "fov_degrees": protocol["capture"]["fov_degrees"],
                "x64_sha256": X64_SHA256,
                "camera_corrected_parent": c30.COHORT_ID,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
