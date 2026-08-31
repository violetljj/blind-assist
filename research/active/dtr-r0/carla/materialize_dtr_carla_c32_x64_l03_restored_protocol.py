"""Materialize C32 with the C28 l03 occlusion trajectory truly restored."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c17_x42_counterfactual_protocol as c17
import materialize_dtr_carla_c31_x64_camera_corrected_protocol as c31


COHORT_ID = "DTR_CARLA_C32_X64_L03_RESTORED_CONFIRMATION_V1"
CAPTURE_SEED = 321064
WEATHERS = {
    "c8_l01": "WetCloudySunset",
    "c8_l02": "SoftRainNoon",
    "c8_l03": "HardRainNight",
    "c8_l04": "CloudyNight",
}
C31_PROTOCOL_SHA256 = (
    "FD8836C2044A81B0065B3641B4E5B86B001299226ED6089F5C81E4776806ECB9"
)
C31_SOURCE_RESULT_SHA256 = (
    "1230E255165A5059367E5EC21DC42C16F40C346366D412AEF75336FF0A5F641B"
)
C28_PROTOCOL_SHA256 = (
    "71C3C097301BAC9D23CA2AE161D1451BDE6D2639BBC325C5A333DC400B0F1E99"
)
COMPONENT_SHA256 = dict(c31.COMPONENT_SHA256)
CAPTURE_SCRIPT_SHA256 = c31.CAPTURE_SCRIPT_SHA256
JOIN_SCRIPT_SHA256 = c31.JOIN_SCRIPT_SHA256
X64_SHA256 = c31.X64_SHA256


def materialize(base: dict) -> dict:
    # Read the frozen C28 protocol itself. Re-materializing C28 is insufficient:
    # its parent chain also exports aliased trajectory objects that C31 mutates.
    validated_parent = c17._read_json(
        Path(__file__).with_name("dtr_carla_c28_x59_mixed_lighting_protocol.json")
    )
    protocol = c31.materialize(copy.deepcopy(base))
    for suffix in ("target_contact", "target_safe"):
        key = f"c8_l03_{suffix}"
        protocol["trajectory_library"][key] = copy.deepcopy(
            validated_parent["trajectory_library"][key]
        )

    protocol["schema_version"] = 32
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Confirm frozen X64 once on fresh pixels after eliminating the shared-"
        "mutable-base defect that prevented C31 l03 physical occlusion."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather

    old_prereg = protocol.pop("c31_x64_preregistration")
    protocol["c32_x64_preregistration"] = {
        **old_prereg,
        "schema": "dtr-carla-c32-x64-preregistration-v1",
        "single_x64_scored_invocation": True,
        "no_post_capture_algorithm_threshold_or_scenario_tuning": True,
    }
    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c32-x64-l03-restored-contract-v1",
        "failed_parent_cohort_id": c31.COHORT_ID,
        "failed_parent_protocol_sha256": C31_PROTOCOL_SHA256,
        "failed_parent_source_result_sha256": C31_SOURCE_RESULT_SHA256,
        "failed_parent_status": "SOURCE_NOT_EVALUABLE_7_OF_8_OCCLUSION_CONTRACTS",
        "failed_parent_algorithm_predictions_created": False,
        "source_only_correction": (
            "RESTORE_C28_L03_TARGET_LATERAL_VELOCITY_FROM_PRIVATE_BASE_COPY"
        ),
        "c31_l03_lateral_velocity_mps": [0.675, -0.675],
        "c32_l03_lateral_velocity_mps": [0.3, -0.3],
        "other_motion_topology_unchanged_from_c31": True,
        "camera_geometry_unchanged_from_c31": True,
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
        "C29, C30, and C31 are terminal SOURCE_NOT_EVALUABLE and are never rerun or scored.",
        "C32 changes only the defective l03 target trajectory materialization plus seed and weather assignment; X64 remains byte-identical.",
        "X64 and every imported algorithm component are byte-frozen before C32 capture and consume no weather or lighting label.",
        "A C32 gate confirms X64 only as source-disjoint synthetic Development when synchronized conflict handback and unanchored crossing release are both exercised.",
        "C32 cannot establish unseen-map, open-world traffic, real-sensor, deployment, reliability, user-benefit, or safety authority.",
        "A failed C32 source gate is NOT_EVALUABLE; frozen X64 cannot be changed or rerun on C32 after pixels are opened.",
    ]

    expected = validated_parent["trajectory_library"]["c8_l03_target_contact"]
    actual = protocol["trajectory_library"]["c8_l03_target_contact"]
    if actual != expected:
        raise RuntimeError("C32 l03 private-base restoration failed")
    lateral = [actual["segments"][1]["velocity_right_mps"], actual["segments"][2]["velocity_right_mps"]]
    if lateral != [0.3, -0.3]:
        raise RuntimeError(f"C32 l03 velocity drift: {lateral}")
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
        here / "dtr_carla_c31_x64_camera_corrected_protocol.json": (
            C31_PROTOCOL_SHA256
        ),
        here / "dtr_carla_c28_x59_mixed_lighting_protocol.json": (
            C28_PROTOCOL_SHA256
        ),
        **{here / name: digest for name, digest in COMPONENT_SHA256.items()},
    }
    for path, expected in frozen_files.items():
        if c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C32 component drift: {path.name}")
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
                "status": "C32_X64_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "output": str(output),
                "sha256": c2.sha256_file(output),
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "weather_domains": WEATHERS,
                "x64_sha256": X64_SHA256,
                "l03_lateral_velocity_mps": [0.3, -0.3],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
