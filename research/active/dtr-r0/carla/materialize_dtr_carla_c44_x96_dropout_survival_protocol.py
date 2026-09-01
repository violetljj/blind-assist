"""Freeze C44 with the last admitted trajectory source after C43 source failure."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2


HERE = Path(__file__).resolve().parent
PARENT_PROTOCOL = HERE / "dtr_carla_c43_x96_dropout_survival_protocol.json"
PARENT_PROTOCOL_SHA256 = (
    "3804A3F47122136A7AFFD842CD922108AA3BD13744318322E24274613410AED2"
)
ADMITTED_TRAJECTORY_PROTOCOL = HERE / "dtr_carla_c41_x82_fresh_confirmation_protocol.json"
ADMITTED_TRAJECTORY_PROTOCOL_SHA256 = (
    "67B806C47B9AA3B038C9CFD84E3BFF89C30D5944BBE84005CE269D2040BA08BE"
)
C43_SOURCE_RESULT_SHA256 = (
    "8BF25B77F1C3029BE9776DE337B61F06A1BE3A3121FD561E68B417CE476B74D7"
)
C43_SOURCE_STATUS = "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_NOT_EVALUABLE"
COHORT_ID = "DTR_CARLA_C44_X96_DROPOUT_SURVIVAL_STRESS_V1"
CAPTURE_SEED = 441096
WEATHERS = {
    "c8_l01": "ClearSunset",
    "c8_l02": "WetCloudyNoon",
    "c8_l03": "SoftRainNoon",
    "c8_l04": "CloudyNoon",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def materialize(parent: dict, admitted: dict) -> dict:
    protocol = copy.deepcopy(parent)
    protocol["schema_version"] = 44
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Evaluate the unchanged X96 four-arm 2/3/6-frame dropout stress on "
        "fresh seed, render assignment, plan receipts, and pixels while "
        "restoring the last source-admitted C41 trajectories after C43 failed "
        "the physical-occlusion source contract."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather

    admitted_by_episode = {
        str(scenario["episode_id"]): scenario for scenario in admitted["scenarios"]
    }
    for scenario in protocol["scenarios"]:
        episode_id = str(scenario["episode_id"])
        source = admitted_by_episode[episode_id]
        scenario["wearer_trajectory"] = source["wearer_trajectory"]
        scenario["asset_trajectories"] = copy.deepcopy(source["asset_trajectories"])
        episode_number = int(episode_id.split("_")[-1])
        session_id = f"c44_session_dropout_{episode_number:02d}"
        scenario["navigation_session_id"] = session_id
        scenario["issued_plan"] = copy.deepcopy(source["issued_plan"])
        scenario["issued_plan"]["plan_id"] = f"c44_plan_dropout_{episode_number:02d}"
        scenario["issued_plan"]["session_id"] = session_id

    prereg = protocol.pop("c43_x96_preregistration")
    prereg["schema"] = "dtr-carla-c44-x96-dropout-survival-preregistration-v1"
    prereg["c43_source_result_sha256"] = C43_SOURCE_RESULT_SHA256
    prereg["trajectory_authority"] = (
        "RESTORED_BYTE_IDENTICAL_C41_TRAJECTORY_BINDINGS_AFTER_C43_SOURCE_GATE_FAILURE"
    )
    protocol["c44_x96_preregistration"] = prereg
    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c44-x96-fresh-source-contract-v1",
        "parent_cohort_id": parent["cohort_id"],
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "parent_source_result_sha256": C43_SOURCE_RESULT_SHA256,
        "parent_source_status": C43_SOURCE_STATUS,
        "c43_failed_check": "track_then_complete_physical_occlusion_contract_met",
        "c44_is_new_cohort_not_c43_retry": True,
        "source_change": "NEW_CAPTURE_SEED_RENDER_DOMAIN_ASSIGNMENT_PLAN_RECEIPTS_AND_PIXELS",
        "new_capture_seed": CAPTURE_SEED,
        "changed_weather_by_layout": WEATHERS,
        "trajectory_source_protocol_sha256": ADMITTED_TRAJECTORY_PROTOCOL_SHA256,
        "target_occluder_alias_and_wearer_trajectories_restored_from_c41": True,
        "trajectory_disjoint_from_c41": False,
        "camera_and_route_geometry_unchanged": True,
        "model_sensor_replay_projection": "ACTUAL_ALL_ACTORS",
        "critical_contact_outcome_truth_must_match_across_all_sensors": True,
        "fresh_pixels": True,
        "prior_pixels_reused": False,
        "future_truth_visible_to_predictor": False,
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
        "C44 is a new cohort after C43 failed the source-only physical-occlusion contract; it is not a C43 retry.",
        "C44 uses new seed, weather assignment, plan receipts, and pixels but restores C41 trajectories byte-for-byte.",
        "C44 is not actor-trajectory-disjoint evidence and must not be described as such.",
        "The X96 algorithm, dropout lengths, placements, and gates remain unchanged from the pre-pixel C42 design.",
        "Candidate removal is a controlled sensor intervention, not natural detector-dropout prevalence evidence.",
        "D plan conflict is a controlled authority intervention, not a captured replanning event.",
        "A partition that misses its preregistered semantics is NOT_EVALUABLE and cannot be repaired on C44.",
        "C44 remains scripted synthetic Development, not real-sensor, deployment, reliability, user-benefit, or safety evidence.",
    ]
    c2.validate_protocol(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c43-source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if c2.sha256_file(PARENT_PROTOCOL) != PARENT_PROTOCOL_SHA256:
        raise RuntimeError("frozen C43 parent protocol drift")
    if c2.sha256_file(ADMITTED_TRAJECTORY_PROTOCOL) != ADMITTED_TRAJECTORY_PROTOCOL_SHA256:
        raise RuntimeError("admitted C41 trajectory protocol drift")
    c43_result = args.c43_source_result.resolve(strict=True)
    if c2.sha256_file(c43_result) != C43_SOURCE_RESULT_SHA256:
        raise RuntimeError("C43 source result drift")
    if read_json(c43_result).get("status") != C43_SOURCE_STATUS:
        raise RuntimeError("C43 source status drift")
    protocol = materialize(read_json(PARENT_PROTOCOL), read_json(ADMITTED_TRAJECTORY_PROTOCOL))
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
                "status": "C44_X96_DROPOUT_SURVIVAL_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "weather_domains": WEATHERS,
                "trajectory_authority": "C41_RESTORED_NOT_TRAJECTORY_DISJOINT",
                "output": str(output),
                "sha256": c2.sha256_file(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
