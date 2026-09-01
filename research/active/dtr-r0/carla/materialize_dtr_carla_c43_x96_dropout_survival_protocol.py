"""Freeze C43 after C42 ended with zero durable source frames."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2


HERE = Path(__file__).resolve().parent
PARENT_PROTOCOL = HERE / "dtr_carla_c42_x96_dropout_survival_protocol.json"
PARENT_PROTOCOL_SHA256 = (
    "48BF56D34E0B433BB2FD82DB6DA748C2C4E565EE3BF94F320120793893B825D1"
)
PARENT_TERMINAL = HERE / "DTR_CARLA_C42_X96_SOURCE_NOT_EVALUABLE_20260901.md"
PARENT_TERMINAL_SHA256 = (
    "F816D7A143645703768ECFD7870CD2AB423EC50FDBC3D002F1E26A3B4245EE1F"
)
COHORT_ID = "DTR_CARLA_C43_X96_DROPOUT_SURVIVAL_STRESS_V1"
CAPTURE_SEED = 431096
WEATHERS = {
    "c8_l01": "ClearNoon",
    "c8_l02": "WetSunset",
    "c8_l03": "HardRainNoon",
    "c8_l04": "CloudySunset",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def materialize(parent: dict) -> dict:
    protocol = copy.deepcopy(parent)
    protocol["schema_version"] = 43
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Execute the unchanged X96 2/3/6-frame four-arm dropout survival "
        "stress on a new C43 source after C42 produced zero durable frames."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather
    for scenario in protocol["scenarios"]:
        episode_number = int(str(scenario["episode_id"]).split("_")[-1])
        session_id = f"c43_session_dropout_{episode_number:02d}"
        scenario["navigation_session_id"] = session_id
        scenario["issued_plan"]["plan_id"] = f"c43_plan_dropout_{episode_number:02d}"
        scenario["issued_plan"]["session_id"] = session_id

    prereg = protocol.pop("c42_x96_preregistration")
    prereg["schema"] = "dtr-carla-c43-x96-dropout-survival-preregistration-v1"
    prereg["c42_zero_frame_terminal_sha256"] = PARENT_TERMINAL_SHA256
    protocol["c43_x96_preregistration"] = prereg
    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c43-x96-fresh-source-contract-v1",
        "parent_cohort_id": parent["cohort_id"],
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "parent_terminal_sha256": PARENT_TERMINAL_SHA256,
        "parent_terminal_status": "DTR_CARLA_C42_X96_SOURCE_NOT_EVALUABLE",
        "parent_durable_sensor_frames": 0,
        "c43_is_new_cohort_not_c42_retry": True,
        "source_change": "NEW_CAPTURE_SEED_RENDER_DOMAIN_ASSIGNMENT_AND_PIXELS",
        "new_capture_seed": CAPTURE_SEED,
        "changed_weather_by_layout": WEATHERS,
        "wearer_and_actor_trajectory_bindings_unchanged_from_c42": True,
        "camera_and_route_geometry_unchanged_from_c42": True,
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
        "C43 is a new cohort after C42 terminated with zero durable frames; it is not a C42 retry.",
        "C43 uses a new seed, weather assignment, plan receipts, and pixels while preserving the frozen X96 algorithm and intervention matrix.",
        "Dropout placements and lengths remain fixed and may not move after predictions or truth are opened.",
        "Candidate removal is a controlled sensor intervention, not natural detector-dropout prevalence evidence.",
        "D plan conflict is a controlled authority intervention, not a captured replanning event.",
        "A partition that misses its preregistered semantics is NOT_EVALUABLE and cannot be repaired on C43.",
        "C43 remains scripted synthetic Development, not real-sensor, deployment, reliability, user-benefit, or safety evidence.",
    ]
    c2.validate_protocol(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if c2.sha256_file(PARENT_PROTOCOL) != PARENT_PROTOCOL_SHA256:
        raise RuntimeError("frozen C42 parent protocol drift")
    if c2.sha256_file(PARENT_TERMINAL) != PARENT_TERMINAL_SHA256:
        raise RuntimeError("frozen C42 terminal drift")
    protocol = materialize(read_json(PARENT_PROTOCOL))
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
                "status": "C43_X96_DROPOUT_SURVIVAL_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "weather_domains": WEATHERS,
                "output": str(output),
                "sha256": c2.sha256_file(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
